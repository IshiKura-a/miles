"""Score policy doc IDs by SM-qualified distinct keywords on multiple GPUs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Annotated, Any

import httpx
import uvicorn
import xxhash
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from examples.experimental.generative_retriever.id_text_mapping import (
    PositionalID,
    _validate_id,
    lookup_service,
)
from examples.experimental.generative_retriever.relevance_model import (
    RelevanceScorer,
    load_relevance_pipeline_factory,
    resolve_relevance_source,
)
from examples.experimental.generative_retriever.relevance_reward import SM_THRESHOLD

class GoodCountSample(BaseModel):
    query: str
    docids: Annotated[list[PositionalID], Field(min_length=1, max_length=16)]

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
            _validate_id(docid)
        return docids


class GoodCountRequest(BaseModel):
    samples: Annotated[list[GoodCountSample], Field(min_length=1, max_length=512)]


@dataclass
class _Worker:
    scorer: RelevanceScorer
    lock: Lock


class GoodCountPool:
    def __init__(
        self,
        *,
        mapping_url: str,
        mapping_version: str,
        relevance_model: str,
        devices: tuple[str, ...],
        batch_size: int,
    ) -> None:
        if not devices or len(set(devices)) != len(devices):
            raise ValueError("Good-count devices must be distinct and non-empty")
        if batch_size <= 0:
            raise ValueError("Good-count batch size must be positive")
        self.mapping_url = mapping_url
        self.mapping_version = mapping_version
        relevance_source = resolve_relevance_source(relevance_model)
        factory = load_relevance_pipeline_factory(relevance_source)
        self.workers = [
            _Worker(
                scorer=RelevanceScorer(factory(
                    relevance_model,
                    cup="large",
                    device=device,
                    dtype="bf16",
                    batch_size=batch_size,
                    cross_batch_size=256,
                    show_progress=False,
                )),
                lock=Lock(),
            )
            for device in devices
        ]
        self._executor = ThreadPoolExecutor(max_workers=len(self.workers))
        self._client = httpx.Client()

    @staticmethod
    def _score(worker: _Worker, samples: list[GoodCountSample], mapping: dict[PositionalID, tuple[str, ...]]):
        keywords_by_sample = [
            list(dict.fromkeys(keyword for docid in sample.docids for keyword in mapping[docid]))
            for sample in samples
        ]
        pairs = [
            (sample.query, keyword)
            for sample, keywords in zip(samples, keywords_by_sample, strict=True)
            for keyword in keywords
        ]
        with worker.lock:
            pair_scores = worker.scorer.score(pairs)
        return [
            {
                "query": sample.query,
                "keyword_count": len(keywords),
                "sm_keywords": [
                    keyword
                    for keyword in keywords
                    if pair_scores[(sample.query, keyword)][1] >= SM_THRESHOLD
                ],
            }
            for sample, keywords in zip(samples, keywords_by_sample, strict=True)
        ]

    def score(self, request: GoodCountRequest) -> dict[str, Any]:
        mapping = lookup_service(
            self.mapping_url,
            (docid for sample in request.samples for docid in sample.docids),
            expected_mapping_version=self.mapping_version, client=self._client,
        )
        buckets: list[list[tuple[int, GoodCountSample]]] = [[] for _ in self.workers]
        for index, sample in enumerate(request.samples):
            worker_index = max(
                range(len(self.workers)),
                key=lambda i: xxhash.xxh64_intdigest(f"{sample.query}\0{i}"),
            )
            buckets[worker_index].append((index, sample))
        futures = [
            (bucket, self._executor.submit(
                self._score,
                worker,
                [sample for _, sample in bucket],
                mapping,
            ))
            for worker, bucket in zip(self.workers, buckets, strict=True)
            if bucket
        ]
        results: dict[int, dict[str, Any]] = {}
        for bucket, future in futures:
            for (index, _), result in zip(bucket, future.result(), strict=True):
                results[index] = result
        return {
            "results": [results[index] for index in range(len(request.samples))],
            "routed_samples": {
                f"worker_{index}": len(bucket)
                for index, bucket in enumerate(buckets)
            },
        }

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        self._client.close()


def create_app(
    *,
    mapping_url: str,
    mapping_version: str,
    relevance_model: str,
    devices: tuple[str, ...],
    batch_size: int = 32,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pool = GoodCountPool(
            mapping_url=mapping_url,
            mapping_version=mapping_version,
            relevance_model=relevance_model,
            devices=devices,
            batch_size=batch_size,
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
            "worker_count": len(app.state.pool.workers),
            "mapping_version": mapping_version,
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
    parser.add_argument("--batch-size", type=int, default=32)
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
        ),
        host=args.host,
        port=args.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
