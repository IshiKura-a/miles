"""Score policy doc IDs by distinct keyword with both PM and SM relevance."""

from __future__ import annotations

from examples.experimental.generative_retriever.utils.paths import absolute_path

import argparse
from contextlib import asynccontextmanager
from typing import Annotated, Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from examples.experimental.generative_retriever.index.id_text_mapping import (
    PositionalID,
    lookup_service,
    validate_docid,
)
from examples.experimental.generative_retriever.reward.relevance_model import (
    MultiGpuRelevanceScorer,
    relevance_source_path,
)
from examples.experimental.generative_retriever.reward.relevance_reward import (
    MAX_DOCIDS_PER_QUERY,
    MAX_EVAL_BATCH_SIZE,
    PM_THRESHOLD,
    REWARD_PAIR_BATCH_SIZE,
    SM_THRESHOLD,
)

class GoodCountSample(BaseModel):
    query: str
    docids: Annotated[list[PositionalID], Field(min_length=1, max_length=MAX_DOCIDS_PER_QUERY)]

    @field_validator("query")
    @classmethod
    def validate_query(cls, query: str) -> str:
        if not query.strip():
            raise ValueError("Query must be non-empty")
        return query

    @field_validator("docids")
    @classmethod
    def validate_docids(cls, docids: list[PositionalID]) -> list[PositionalID]:
        for docid in docids:
            validate_docid(docid)
        return docids


class GoodCountRequest(BaseModel):
    samples: Annotated[list[GoodCountSample], Field(min_length=1, max_length=MAX_EVAL_BATCH_SIZE)]


class GoodCountPool:
    def __init__(
        self,
        *,
        mapping_url: str,
        mapping_version: str,
        relevance_model: str,
        devices: tuple[str, ...],
        batch_size: int,
        use_initial: bool = False,
    ) -> None:
        if not devices or len(set(devices)) != len(devices):
            raise ValueError("Good-count devices must be distinct and non-empty")
        if batch_size < 4 or batch_size % 4:
            raise ValueError("Good-count batch size must be a positive multiple of 4")
        self.mapping_url = mapping_url
        self.mapping_version = mapping_version
        self.use_initial = use_initial
        self.devices = devices
        self.relevance_model = str(absolute_path(relevance_model))
        self.scorer = MultiGpuRelevanceScorer(
            self.relevance_model,
            devices,
            relevance_source=relevance_source_path(self.relevance_model),
            batch_size=batch_size,
            score_cache_size=4_194_304,
            pair_batch_size=REWARD_PAIR_BATCH_SIZE,
        )
        self._client = httpx.Client(timeout=300.0, trust_env=False)

    def score(self, request: GoodCountRequest) -> dict[str, Any]:
        mapping = lookup_service(
            self.mapping_url,
            (docid for sample in request.samples for docid in sample.docids),

            use_initial=self.use_initial,
            client=self._client,
        )
        keywords_by_sample = [
            list(dict.fromkeys(keyword for docid in sample.docids for keyword in mapping[docid]))
            for sample in request.samples
        ]
        pairs = [
            (sample.query, keyword)
            for sample, keywords in zip(request.samples, keywords_by_sample, strict=True)
            for keyword in keywords
        ]
        pair_scores = self.scorer.score(pairs)
        results = []
        for sample, keywords in zip(request.samples, keywords_by_sample, strict=True):
            scores = [
                {
                    "keyword": keyword,
                    "pm_score": pair_scores[(sample.query, keyword)][0],
                    "sm_score": pair_scores[(sample.query, keyword)][1],
                }
                for keyword in keywords
            ]
            results.append({
                "query": sample.query,
                "keyword_count": len(scores),
                "keyword_scores": scores,
                "pm_keywords": [
                    row["keyword"] for row in scores if row["pm_score"] >= PM_THRESHOLD
                ],
                "sm_keywords": [
                    row["keyword"] for row in scores if row["sm_score"] >= SM_THRESHOLD
                ],
            })
        return {
            "results": results,
            "mapping_version": self.mapping_version,
            "relevance_model": self.relevance_model,
            "routed_samples": {
                f"worker_{index}": len(request.samples) if index == 0 else 0
                for index in range(len(self.devices))
            },
        }

    def close(self) -> None:
        self.scorer.close()
        self._client.close()


def create_app(
    *,
    mapping_url: str,
    mapping_version: str,
    relevance_model: str,
    devices: tuple[str, ...],
    batch_size: int = 8192,
    use_initial: bool = False,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pool = GoodCountPool(
            mapping_url=mapping_url,
            mapping_version=mapping_version,
            relevance_model=relevance_model,
            devices=devices,
            batch_size=batch_size,
            use_initial=use_initial,
        )
        try:
            yield
        finally:
            app.state.pool.close()
            del app.state.pool

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ready",
            "worker_count": len(app.state.pool.devices),
            "mapping_version": mapping_version,
            "relevance_model": app.state.pool.relevance_model,
        }

    @app.post("/score")
    def score(request: GoodCountRequest) -> dict[str, Any]:
        try:
            return app.state.pool.score(request)
        except (httpx.HTTPError, LookupError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-url", default="http://127.0.0.1:30488")
    parser.add_argument("--mapping-version", required=True)
    parser.add_argument("--relevance-model", required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--use-initial", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30589)
    args = parser.parse_args()
    devices = tuple(device.strip() for device in args.devices.split(",") if device.strip())
    uvicorn.run(
        create_app(
            mapping_url=args.mapping_url,
            mapping_version=args.mapping_version,
            relevance_model=args.relevance_model,
            devices=devices,
            batch_size=args.batch_size,
            use_initial=args.use_initial,
        ),
        host=args.host,
        port=args.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
