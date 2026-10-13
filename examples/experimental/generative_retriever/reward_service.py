"""Serve keyword-averaged rewards using official relevance and prepared baselines.

Run:
    python -m examples.experimental.generative_retriever.reward_service \
        --baseline-dir /path/to/baseline-rewards --mapping-url http://127.0.0.1:30488 \
        --port 30489
"""

from __future__ import annotations

import argparse
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Annotated, Any
from urllib.error import HTTPError, URLError

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StrictInt, field_validator

from examples.experimental.generative_retriever.id_text_mapping import _validate_id, lookup_service
from examples.experimental.generative_retriever.lru_cache import LRUCache
from examples.experimental.generative_retriever.relevance_model import (
    load_relevance_pipeline_factory,
    resolve_relevance_source,
)
from examples.experimental.generative_retriever.relevance_reward import (
    RewardMode, build_baseline_set,
    calculate_docid_reward, calculate_keyword_rewards,
)

logger = logging.getLogger(__name__)
_SERVICE_API_VERSION = 3
_CACHE_SIZE = 8192


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
    reward_mode: RewardMode = "discrete_relevance_unique"


class KeywordRewardEngine:
    def __init__(
        self,
        baseline_url: str,
        mapping_url: str,
        mapping_version: str,
        relevance_model: str,
        device: str,
        batch_size: int,
        cache_size: int,
        baseline_top_k: int,
    ) -> None:
        self.mapping_version = mapping_version
        self.mapping_url = mapping_url
        self.baseline_url = baseline_url
        self.baseline_top_k = baseline_top_k
        self._baseline_queries: set[str] = set()
        self.baselines: LRUCache[str, frozenset[str]] = LRUCache(cache_size)
        self._score_cache: LRUCache[tuple[str, str], tuple[float, float]] = LRUCache(cache_size)
        relevance_source = resolve_relevance_source(relevance_model)
        pipeline_factory = load_relevance_pipeline_factory(relevance_source)
        self.pipeline = pipeline_factory(
            relevance_model, cup="large", device=device, dtype="bf16",
            batch_size=batch_size, cross_batch_size=256, show_progress=False,
        )
        self._lock = Lock()
        self._query_features: LRUCache[str, np.ndarray] = LRUCache(cache_size)

    def _score_missing(self, query: str, keywords: list[str]) -> None:
        query_features = self._query_features.get(query)
        if query_features is None:
            query_features = self.pipeline.encode_query([query])
            self._query_features.put(query, query_features)
        features = self.pipeline.encode_keyword(keywords)
        scores = self.pipeline.cross_score(
            np.repeat(query_features, len(keywords), axis=0), features,
        )
        if (
            scores.shape != (len(keywords), 4)
            or not np.isfinite(scores).all()
            or not ((scores >= 0) & (scores <= 1)).all()
        ):
            raise RuntimeError("Official relevance model returned invalid scores")
        for keyword, score in zip(keywords, scores, strict=True):
            self._score_cache.put((query, keyword), (float(score[1]), float(score[2])))

    def _get_baseline(self, query: str) -> frozenset[str]:
        cached = self.baselines.get(query)
        if cached is not None:
            return cached
        if not self.baseline_url:
            raise KeyError(f"Prepared baseline missing for query {query!r}")
        response = httpx.post(self.baseline_url, json={"queries": [query]}, timeout=300.0)
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results")
        if (
            not isinstance(results, list)
            or len(results) != 1
            or not isinstance(results[0], dict)
            or results[0].get("query") != query
        ):
            raise ValueError("Baseline service returned an invalid query group")
        rows = results[0].get("docids")
        if not isinstance(rows, list) or len(rows) != self.baseline_top_k:
            raise ValueError(
                f"Baseline service must return exactly {self.baseline_top_k} doc IDs"
            )
        docids = []
        for row in rows:
            docid = row.get("positional_docid") if isinstance(row, dict) else None
            if not isinstance(docid, list) or len(docid) != 4 or not all(isinstance(value, int) for value in docid):
                raise ValueError("Baseline service returned an invalid positional doc ID")
            docids.append((docid[0], docid[1], docid[2], docid[3]))
        if len(set(docids)) != self.baseline_top_k:
            raise ValueError("Baseline service returned duplicate doc IDs")
        mapping = lookup_service(
            self.mapping_url,
            docids,
            expected_mapping_version=self.mapping_version,
        )
        keywords = list(dict.fromkeys(keyword for docid in docids for keyword in mapping[docid]))
        missing = [keyword for keyword in keywords if self._score_cache.get((query, keyword)) is None]
        if missing:
            self._score_missing(query, missing)
        scores = {}
        for keyword in keywords:
            score = self._score_cache.get((query, keyword))
            if score is None:
                raise RuntimeError("Baseline keyword score disappeared from the resident LRU cache")
            scores[keyword] = score
        baseline = build_baseline_set(scores)
        self.baselines.put(query, baseline)
        self._baseline_queries.add(query)
        return baseline

    def score(self, request: RewardRequest) -> dict[str, Any]:
        started = time.perf_counter()
        before = {
            "baseline": self.baselines.stats(),
            "reward": self._score_cache.stats(),
            "query_features": self._query_features.stats(),
        }
        docids = [
            (sample.docid_tokens[0] - 2, sample.docid_tokens[1] - 2,
             sample.docid_tokens[2] - 2, sample.docid_tokens[3] - 2)
            for sample in request.samples
        ]
        mapping = lookup_service(
            self.mapping_url, docids, expected_mapping_version=self.mapping_version,
        )
        with self._lock:
            rewards = []
            details = []
            for sample, docid in zip(request.samples, docids, strict=True):
                baseline = self._get_baseline(sample.prompt)
                missing = []
                for keyword in mapping[docid]:
                    if self._score_cache.get((sample.prompt, keyword)) is None:
                        missing.append(keyword)
                if missing:
                    self._score_missing(sample.prompt, missing)
                scores = {}
                for keyword in mapping[docid]:
                    score = self._score_cache.get((sample.prompt, keyword))
                    if score is None:
                        raise RuntimeError("Reward score disappeared from the resident LRU cache")
                    scores[keyword] = score
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
        cache = {
            "baseline": asdict(self.baselines.stats()),
            "reward": asdict(self._score_cache.stats()),
            "query_features": asdict(self._query_features.stats()),
        }
        return {
            "rewards": rewards,
            "details": details,
            "reward_mode": request.reward_mode,
            "mapping_version": self.mapping_version,
            "latency_seconds": time.perf_counter() - started,
            "cache": cache,
            "cache_delta": {
                name: {
                    counter: cache[name][counter] - getattr(before[name], counter)
                    for counter in ("hits", "misses", "evictions")
                }
                for name in cache
            },
        }


def create_app(
    baseline_url: str,
    mapping_url: str,
    mapping_version: str,
    relevance_model: str,
    *,
    batch_size: int = 32,
    baseline_top_k: int = 16,
) -> FastAPI:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = KeywordRewardEngine(
            baseline_url,
            mapping_url,
            mapping_version,
            relevance_model,
            "cuda:0",
            batch_size,
            _CACHE_SIZE,
            baseline_top_k,
        )
        yield
        del app.state.engine

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        engine = app.state.engine
        return {
            "status": "ready",
            "mode": "worker",
            "worker_count": 1,
            "device": "cuda:0",
            "mapping_version": engine.mapping_version,
            "baseline_queries": len(engine._baseline_queries),
            "cache": {
                "baseline": asdict(engine.baselines.stats()),
                "reward": asdict(engine._score_cache.stats()),
                "query_features": asdict(engine._query_features.stats()),
            },
            "service_config": {
                "api_version": _SERVICE_API_VERSION,
                "baseline_url": baseline_url,
                "mapping_url": mapping_url,
                "mapping_version": mapping_version,
                "relevance_source": str(resolve_relevance_source(relevance_model)),
                "relevance_model": str(Path(relevance_model).expanduser().resolve()),
                "cache_size": _CACHE_SIZE,
                "baseline_top_k": baseline_top_k,
            },
        }

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
        except (HTTPError, URLError, httpx.HTTPError) as error:
            logger.error("Dependent service unavailable: %s", error)
            raise HTTPException(status_code=502, detail=str(error)) from error
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-url", required=True)
    parser.add_argument("--mapping-url", required=True)
    parser.add_argument("--mapping-version", required=True)
    parser.add_argument("--relevance-model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30489)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--baseline-top-k", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        create_app(
            args.baseline_url,
            args.mapping_url,
            mapping_version=args.mapping_version,
            relevance_model=args.relevance_model,
            batch_size=args.batch_size,
            baseline_top_k=args.baseline_top_k,
        ),
        host=args.host,
        port=args.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
