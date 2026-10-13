"""Resident two-GPU frozen baseline constrained-beam service."""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Annotated, Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from examples.experimental.generative_retriever.lru_cache import LRUCache
from examples.experimental.generative_retriever.transformers_infer import GenerationResult, SLMv3Retriever

logger = logging.getLogger(__name__)
_SERVICE_API_VERSION = 2
_CACHE_SIZE = 8192


class BaselineRequest(BaseModel):
    queries: Annotated[list[str], Field(min_length=1, max_length=512)]

    @field_validator("queries")
    @classmethod
    def validate_queries(cls, queries: list[str]) -> list[str]:
        if any(not query.strip() for query in queries):
            raise ValueError("Baseline queries must be non-empty")
        return queries


class _Worker:
    def __init__(self, retriever: SLMv3Retriever) -> None:
        self.retriever = retriever
        self.lock = Lock()

    def generate(self, queries: list[str], docids_per_query: int) -> list[GenerationResult]:
        with self.lock:
            return self.retriever.generate(
                queries,
                num_beams=docids_per_query,
                num_return_sequences=docids_per_query,
            )


class BaselinePool:
    def __init__(
        self,
        *,
        checkpoint: str,
        qwen_model: str,
        slm_root: str,
        id_mapping: str,
        devices: tuple[str, str],
        cache_size: int,
        docids_per_query: int,
    ) -> None:
        self.devices = devices
        self.cache: LRUCache[str, list[dict[str, Any]]] = LRUCache(cache_size)
        self.docids_per_query = docids_per_query
        self._cache_lock = Lock()
        self.workers = [
            _Worker(
                SLMv3Retriever(
                    checkpoint=checkpoint,
                    qwen_model=qwen_model,
                    slm_root=slm_root,
                    id_mapping=id_mapping,
                    device=device,
                )
            )
            for device in devices
        ]
        self._executor = ThreadPoolExecutor(max_workers=len(self.workers))

    @staticmethod
    def _serialize(results: list[GenerationResult]) -> list[dict[str, Any]]:
        return [
            {
                "decoder_token_ids": result.decoder_token_ids,
                "positional_docid": result.positional_docid,
                "rq_codes": result.rq_codes,
                "token_logprobs": result.token_logprobs,
            }
            for result in results
        ]

    def generate(self, request: BaselineRequest) -> dict[str, Any]:
        started = time.perf_counter()
        resolved: dict[int, list[dict[str, Any]]] = {}
        pending: list[tuple[int, str]] = []
        with self._cache_lock:
            for index, query in enumerate(request.queries):
                cached = self.cache.get(query)
                if cached is None:
                    pending.append((index, query))
                else:
                    resolved[index] = cached

        buckets: list[list[tuple[int, str]]] = [[] for _ in self.workers]
        for pending_index, item in enumerate(pending):
            buckets[pending_index % len(self.workers)].append(item)
        futures = []
        for worker, bucket in zip(self.workers, buckets, strict=True):
            if bucket:
                futures.append((
                    bucket,
                    self._executor.submit(
                        worker.generate,
                        [query for _, query in bucket],
                        self.docids_per_query,
                    ),
                ))
        for bucket, future in futures:
            generated = future.result()
            for query_index, (index, query) in enumerate(bucket):
                start = query_index * self.docids_per_query
                rows = self._serialize(generated[start:start + self.docids_per_query])
                if len(rows) != self.docids_per_query:
                    raise RuntimeError("Baseline worker returned an incomplete beam group")
                resolved[index] = rows
                with self._cache_lock:
                    self.cache.put(query, rows)

        results = [
            {"query": query, "docids": resolved[index]}
            for index, query in enumerate(request.queries)
        ]
        return {
            "results": results,
            "docids_per_query": self.docids_per_query,
            "latency_seconds": time.perf_counter() - started,
            "cache": asdict(self.cache.stats()),
        }

    def close(self) -> None:
        self._executor.shutdown(wait=True)


def create_app(
    *,
    checkpoint: str,
    qwen_model: str,
    slm_root: str,
    id_mapping: str,
    docids_per_query: int = 16,
) -> FastAPI:
    devices = ("cuda:0", "cuda:1")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pool = BaselinePool(
            checkpoint=checkpoint,
            qwen_model=qwen_model,
            slm_root=slm_root,
            id_mapping=id_mapping,
            devices=devices,
            cache_size=_CACHE_SIZE,
            docids_per_query=docids_per_query,
        )
        yield
        app.state.pool.close()
        del app.state.pool

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        pool = app.state.pool
        return {
            "status": "ready",
            "worker_count": len(pool.workers),
            "devices": pool.devices,
            "docids_per_query": pool.docids_per_query,
            "cache": asdict(pool.cache.stats()),
            "service_config": {
                "api_version": _SERVICE_API_VERSION,
                "checkpoint": str(Path(checkpoint).expanduser().resolve()),
                "qwen_model": str(Path(qwen_model).expanduser().resolve()),
                "slm_root": str(Path(slm_root).expanduser().resolve()),
                "id_mapping": str(Path(id_mapping).expanduser().resolve()),
                "devices": list(devices),
                "cache_size": _CACHE_SIZE,
                "docids_per_query": docids_per_query,
            },
        }

    @app.post("/generate")
    def generate(request: BaselineRequest) -> dict[str, Any]:
        return app.state.pool.generate(request)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--qwen-model", required=True)
    parser.add_argument("--slm-root", required=True)
    parser.add_argument("--id-mapping", required=True)
    parser.add_argument("--docids-per-query", type=int, default=16)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30487)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        create_app(
            checkpoint=args.checkpoint,
            qwen_model=args.qwen_model,
            slm_root=args.slm_root,
            id_mapping=args.id_mapping,
            docids_per_query=args.docids_per_query,
        ),
        host=args.host,
        port=args.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
