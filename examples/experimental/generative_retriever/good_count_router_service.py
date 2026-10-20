"""Route good-count scoring batches across isolated single-GPU workers."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
import xxhash
from fastapi import FastAPI, HTTPException

from examples.experimental.generative_retriever.good_count_service import GoodCountRequest


class GoodCountRouter:
    def __init__(self, worker_urls: tuple[str, ...]) -> None:
        if not worker_urls or len(set(worker_urls)) != len(worker_urls):
            raise ValueError("Good-count worker URLs must be distinct and non-empty")
        self.worker_urls = worker_urls
        self._executor = ThreadPoolExecutor(max_workers=len(worker_urls))
        self._client = httpx.Client(timeout=900.0)

    def _post(self, url: str, request: GoodCountRequest) -> dict[str, Any]:
        response = self._client.post(url, json=request.model_dump(mode="json"))
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Good-count worker response must be a JSON object")
        return payload

    def score(self, request: GoodCountRequest) -> dict[str, Any]:
        buckets = [[] for _ in self.worker_urls]
        for index, sample in enumerate(request.samples):
            worker_index = max(
                range(len(self.worker_urls)),
                key=lambda i: xxhash.xxh64_intdigest(f"{sample.query}\0{self.worker_urls[i]}"),
            )
            buckets[worker_index].append((index, sample))
        futures = [
            (bucket, self._executor.submit(
                self._post,
                url,
                GoodCountRequest(samples=[sample for _, sample in bucket]),
            ))
            for url, bucket in zip(self.worker_urls, buckets, strict=True)
            if bucket
        ]
        results: dict[int, dict[str, Any]] = {}
        versions: set[tuple[str, str]] = set()
        for bucket, future in futures:
            payload = future.result()
            worker_results = payload.get("results")
            if not isinstance(worker_results, list) or len(worker_results) != len(bucket):
                raise ValueError("Good-count worker returned an invalid result count")
            version = payload.get("mapping_version")
            model = payload.get("relevance_model")
            if not isinstance(version, str) or not isinstance(model, str) or not version or not model:
                raise ValueError("Good-count worker returned invalid scoring provenance")
            versions.add((version, model))
            for (index, sample), result in zip(bucket, worker_results, strict=True):
                if not isinstance(result, dict) or result.get("query") != sample.query:
                    raise ValueError("Good-count worker changed query ordering")
                results[index] = result
        if len(versions) != 1:
            raise ValueError("Good-count workers use different scoring models or mappings")
        mapping_version, relevance_model = versions.pop()
        return {
            "results": [results[index] for index in range(len(request.samples))],
            "mapping_version": mapping_version,
            "relevance_model": relevance_model,
            "routed_samples": {
                f"worker_{index}": len(bucket)
                for index, bucket in enumerate(buckets)
            },
        }

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        self._client.close()


def create_app(worker_urls: tuple[str, ...]) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.router = GoodCountRouter(worker_urls)
        try:
            yield
        finally:
            app.state.router.close()
            del app.state.router

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        try:
            responses = []
            for url in worker_urls:
                response = app.state.router._client.get(
                    url.removesuffix("/score") + "/health", timeout=10.0,
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or payload.get("status") != "ready":
                    raise ValueError("A good-count worker is not ready")
                responses.append(payload)
            versions = {
                (response.get("mapping_version"), response.get("relevance_model"))
                for response in responses
            }
            if len(versions) != 1:
                raise ValueError("Good-count workers use different scoring models or mappings")
            mapping_version, relevance_model = versions.pop()
            if not isinstance(mapping_version, str) or not mapping_version or not isinstance(relevance_model, str) or not relevance_model:
                raise ValueError("Good-count worker returned invalid scoring provenance")
            return {
                "status": "ready", "worker_count": len(worker_urls),
                "mapping_version": mapping_version, "relevance_model": relevance_model,
            }
        except (httpx.HTTPError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.post("/score")
    def score(request: GoodCountRequest) -> dict[str, Any]:
        try:
            return app.state.router.score(request)
        except (httpx.HTTPError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker-urls",
        default=",".join(f"http://127.0.0.1:{port}/score" for port in range(30590, 30594)),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30589)
    args = parser.parse_args()
    worker_urls = tuple(url.strip() for url in args.worker_urls.split(",") if url.strip())
    uvicorn.run(create_app(worker_urls), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
