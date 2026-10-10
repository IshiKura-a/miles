"""Serve keyword-averaged rewards using official relevance and prepared baselines.

Run:
    python -m examples.experimental.generative_retriever.reward_service \
        --baseline-dir /path/to/baseline-rewards --mapping-url http://127.0.0.1:30488 \
        --port 30489
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Annotated, Any
from urllib.error import HTTPError, URLError

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StrictInt, field_validator

from examples.experimental.generative_retriever.id_text_mapping import _validate_id, lookup_service
from examples.experimental.generative_retriever.relevance_reward import (
    BASELINE_TOP_K, PM_THRESHOLD, SM_THRESHOLD, RewardMode, build_baseline_set,
    calculate_docid_reward, calculate_keyword_rewards,
)

logger = logging.getLogger(__name__)


class RewardSample(BaseModel):
    prompt: str
    docid_tokens: tuple[StrictInt, StrictInt, StrictInt, StrictInt]

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, query: str) -> str:
        if not query.strip():
            raise ValueError("Query must be non-empty")
        return query

    @field_validator("docid_tokens")
    @classmethod
    def validate_tokens(cls, tokens: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        _validate_id((tokens[0] - 2, tokens[1] - 2, tokens[2] - 2, tokens[3] - 2))
        return tokens


class RewardRequest(BaseModel):
    samples: Annotated[list[RewardSample], Field(min_length=1, max_length=512)]
    reward_mode: RewardMode = "continuous_relevance_unique"


class KeywordRewardEngine:
    def __init__(self, baseline_dir: Path, mapping_url: str, device: str, batch_size: int) -> None:
        manifest = json.loads((baseline_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError("Baseline manifest must be complete")
        self.mapping_version = manifest["mapping_version"]
        self.mapping_url = mapping_url
        self.baselines: dict[str, frozenset[str]] = {}
        self._scores: dict[str, dict[str, tuple[float, float]]] = {}
        for index, summary in enumerate(manifest["queries"]):
            if (summary["baseline_docids"] != BASELINE_TOP_K
                    or summary["mapping_version"] != self.mapping_version
                    or summary["pm_threshold"] != PM_THRESHOLD or summary["sm_threshold"] != SM_THRESHOLD):
                raise ValueError("Baseline summary does not match the reward configuration")
            directory = baseline_dir / f"query-{index:04d}"
            query = summary["query"]
            if query in self.baselines:
                raise ValueError(f"Duplicate baseline query: {query!r}")
            scores = {}
            with (directory / "keyword_rewards.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["query"] != query or row["keyword"] in scores:
                        raise ValueError("Baseline keyword score records are inconsistent")
                    scores[row["keyword"]] = (row["pm_score"], row["sm_score"])
            baseline = build_baseline_set(scores)
            saved = json.loads((directory / "baseline_keywords.json").read_text(encoding="utf-8"))
            if baseline != frozenset(saved) or len(baseline) != summary["baseline_keyword_count"]:
                raise ValueError("Saved baseline set does not match its PM/SM scores")
            self.baselines[query] = baseline
            self._scores[query] = scores
        if not self.baselines:
            raise ValueError("At least one prepared baseline query is required")
        sys.path.insert(0, manifest["relevance_source"])
        from graphene.multi_cup.multi_cup_graphene_pipeline import MultiCupGraphenePipeline

        self.pipeline = MultiCupGraphenePipeline(
            manifest["relevance_model"], cup="large", device=device, dtype="bf16",
            batch_size=batch_size, cross_batch_size=256, show_progress=False,
        )
        self._lock = Lock()
        self._query_features: dict[str, np.ndarray] = {}

    def _score_missing(self, query: str, keywords: list[str]) -> None:
        if query not in self._query_features:
            self._query_features[query] = self.pipeline.encode_query([query])
        features = self.pipeline.encode_keyword(keywords)
        scores = self.pipeline.cross_score(
            np.repeat(self._query_features[query], len(keywords), axis=0), features,
        )
        if (
            scores.shape != (len(keywords), 4)
            or not np.isfinite(scores).all()
            or not ((scores >= 0) & (scores <= 1)).all()
        ):
            raise RuntimeError("Official relevance model returned invalid scores")
        for keyword, score in zip(keywords, scores, strict=True):
            self._scores[query][keyword] = (float(score[1]), float(score[2]))

    def score(self, request: RewardRequest) -> dict[str, Any]:
        for sample in request.samples:
            if sample.prompt not in self.baselines:
                raise KeyError(f"Prepared baseline missing for query {sample.prompt!r}")
        docids = [
            (sample.docid_tokens[0] - 2, sample.docid_tokens[1] - 2,
             sample.docid_tokens[2] - 2, sample.docid_tokens[3] - 2)
            for sample in request.samples
        ]
        mapping = lookup_service(
            self.mapping_url, docids, expected_mapping_version=self.mapping_version,
        )
        with self._lock:
            needed: dict[str, dict[str, None]] = {}
            for sample, docid in zip(request.samples, docids, strict=True):
                pending = needed.setdefault(sample.prompt, {})
                for keyword in mapping[docid]:
                    if keyword not in self._scores[sample.prompt]:
                        pending[keyword] = None
            for query, keywords in needed.items():
                if keywords:
                    self._score_missing(query, list(keywords))
            rewards = []
            details = []
            for sample, docid in zip(request.samples, docids, strict=True):
                scores = {keyword: self._scores[sample.prompt][keyword] for keyword in mapping[docid]}
                baseline = self.baselines[sample.prompt]
                breakdown = calculate_keyword_rewards(scores, baseline, mode=request.reward_mode)
                value = calculate_docid_reward(scores, baseline, mode=request.reward_mode)
                rewards.append(value)
                details.append({
                    "query": sample.prompt, "docid": docid, "keyword_count": len(scores),
                    "aggregation": "mean", "reward": value,
                    "keywords": [{"keyword": keyword, **asdict(result)}
                                 for keyword, result in breakdown.items()],
                })
                logger.info("reward_mode=%s docid=%s keywords=%d reward=%f",
                            request.reward_mode, docid, len(scores), value)
        return {"rewards": rewards, "details": details, "reward_mode": request.reward_mode,
                "mapping_version": self.mapping_version}


def create_app(baseline_dir: str | Path, mapping_url: str, *, device: str = "cpu", batch_size: int = 32) -> FastAPI:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = KeywordRewardEngine(Path(baseline_dir).expanduser().resolve(),
                                              mapping_url, device, batch_size)
        yield
        del app.state.engine

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        engine = app.state.engine
        return {"status": "ready", "mapping_version": engine.mapping_version,
                "baseline_queries": len(engine.baselines)}

    @app.post("/score")
    def score(request: RewardRequest) -> dict[str, Any]:
        try:
            return app.state.engine.score(request)
        except KeyError as error:
            logger.error("Baseline lookup failed: %s", error)
            raise HTTPException(status_code=409, detail=str(error)) from error
        except LookupError as error:
            logger.error("Official mapping lookup failed: %s", error)
            raise HTTPException(status_code=422, detail=str(error)) from error
        except (HTTPError, URLError) as error:
            logger.error("Mapping service unavailable: %s", error)
            raise HTTPException(status_code=502, detail=str(error)) from error
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--mapping-url", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30489)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(args.baseline_dir, args.mapping_url, device=args.device,
                           batch_size=args.batch_size), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
