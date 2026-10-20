"""Route reward batches across isolated single-GPU reward workers."""

from __future__ import annotations

import argparse
import math
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any

import httpx
import uvicorn
import xxhash
from fastapi import FastAPI, HTTPException

from examples.experimental.generative_retriever.reward_service import (
    BaselinePrepareRequest,
    RewardRequest,
)

_SERVICE_API_VERSION = 5


class RewardRouter:
    def __init__(self, worker_urls: tuple[str, ...]) -> None:
        if not worker_urls or len(set(worker_urls)) != len(worker_urls):
            raise ValueError("Reward router requires distinct worker URLs")
        self.worker_urls = worker_urls
        self._executor = ThreadPoolExecutor(max_workers=len(worker_urls))
        self._client = httpx.Client(timeout=900.0)
        self._routing_lock = Lock()
        self._routed_samples = [0] * len(worker_urls)
        self._prepared_queries = [0] * len(worker_urls)

    def _worker_index(self, query: str) -> int:
        scores = [
            xxhash.xxh64_intdigest(f"{query}\0{worker_url}")
            for worker_url in self.worker_urls
        ]
        return max(range(len(scores)), key=scores.__getitem__)

    def _post(self, url: str, request: RewardRequest | BaselinePrepareRequest) -> dict[str, Any]:
        response = self._client.post(
            url,
            json=request.model_dump(mode="json"),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Reward worker response must be a JSON object")
        return payload

    def score(self, request: RewardRequest) -> dict[str, Any]:
        buckets: list[list[tuple[int, Any]]] = [[] for _ in self.worker_urls]
        for index, sample in enumerate(request.samples):
            buckets[self._worker_index(sample.prompt)].append((index, sample))
        with self._routing_lock:
            for index, bucket in enumerate(buckets):
                self._routed_samples[index] += len(bucket)
        futures = [
            (index, self._executor.submit(
                self._post,
                url,
                RewardRequest(
                    samples=[sample for _, sample in bucket],
                    reward_mode=request.reward_mode,
                ),
            ))
            for index, (url, bucket) in enumerate(zip(self.worker_urls, buckets, strict=True))
            if bucket
        ]
        responses = {index: future.result() for index, future in futures}
        rewards: dict[int, float] = {}
        details: dict[int, dict[str, Any]] = {}
        for worker_index, response in responses.items():
            bucket = buckets[worker_index]
            worker_rewards = response.get("rewards")
            worker_details = response.get("details")
            if (
                not isinstance(worker_rewards, list)
                or len(worker_rewards) != len(bucket)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in worker_rewards
                )
                or not isinstance(worker_details, list)
                or len(worker_details) != len(bucket)
                or any(not isinstance(detail, dict) for detail in worker_details)
                or response.get("reward_mode") != request.reward_mode
            ):
                raise ValueError("Reward worker returned invalid rewards or details")
            for position, (index, _) in enumerate(bucket):
                rewards[index] = worker_rewards[position]
                details[index] = worker_details[position]
        return {
            "rewards": [rewards[index] for index in range(len(request.samples))],
            "details": [details[index] for index in range(len(request.samples))],
            "reward_mode": request.reward_mode,
            **self._response_metrics(responses),
            "routed_samples": {
                f"worker_{index}": len(bucket)
                for index, bucket in enumerate(buckets)
            },
        }

    def prepare(self, request: BaselinePrepareRequest) -> dict[str, Any]:
        buckets: list[list[str]] = [[] for _ in self.worker_urls]
        for query in dict.fromkeys(request.queries):
            buckets[self._worker_index(query)].append(query)
        with self._routing_lock:
            for index, bucket in enumerate(buckets):
                self._prepared_queries[index] += len(bucket)
        futures = [
            (index, self._executor.submit(
                self._post,
                url.removesuffix("/score") + "/prepare",
                BaselinePrepareRequest(queries=bucket),
            ))
            for index, (url, bucket) in enumerate(zip(self.worker_urls, buckets, strict=True))
            if bucket
        ]
        responses = {index: future.result() for index, future in futures}
        return {
            "prepared_queries": sum(
                response["prepared_queries"] for response in responses.values()
            ),
            **self._response_metrics(responses),
            "routed_queries": {
                f"worker_{index}": len(bucket)
                for index, bucket in enumerate(buckets)
            },
        }

    @staticmethod
    def _response_metrics(responses: dict[int, dict[str, Any]]) -> dict[str, Any]:
        version = next(iter(responses.values())).get("mapping_version")
        if not isinstance(version, str) or not version:
            raise ValueError("Reward worker returned an invalid mapping version")
        if any(response.get("mapping_version") != version for response in responses.values()):
            raise ValueError("Reward workers use different mappings")
        return {
            "mapping_version": version,
            "cache": {
                f"worker_{index}": response["cache"]
                for index, response in responses.items()
            },
            "cache_delta": {
                f"worker_{index}": response["cache_delta"]
                for index, response in responses.items()
            },
            "latency_seconds": max(response["latency_seconds"] for response in responses.values()),
        }

    def health(self) -> dict[str, Any]:
        responses = []
        for url in self.worker_urls:
            response = self._client.get(url.removesuffix("/score") + "/health", timeout=10.0)
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") != "ready" or payload.get("worker_count") != 1:
                raise RuntimeError(f"Reward worker is not ready: {payload}")
            responses.append(payload)
        mapping_versions = {response["mapping_version"] for response in responses}
        if len(mapping_versions) != 1:
            raise RuntimeError(f"Reward workers use different mappings: {mapping_versions}")
        with self._routing_lock:
            routed_samples = list(self._routed_samples)
            prepared_queries = list(self._prepared_queries)
        return {
            "status": "ready",
            "mode": "router",
            "worker_count": len(self.worker_urls),
            "mapping_version": mapping_versions.pop(),
            "workers": responses,
            "routed_samples": {
                f"worker_{index}": count
                for index, count in enumerate(routed_samples)
            },
            "prepared_queries": {
                f"worker_{index}": count
                for index, count in enumerate(prepared_queries)
            },
            "service_config": {
                "api_version": _SERVICE_API_VERSION,
                "worker_urls": list(self.worker_urls),
            },
        }

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        self._client.close()


def create_app(worker_urls: tuple[str, ...]) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.router = RewardRouter(worker_urls)
        try:
            yield
        finally:
            app.state.router.close()
            del app.state.router

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        try:
            return app.state.router.health()
        except (httpx.HTTPError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.post("/score")
    def score(request: RewardRequest) -> dict[str, Any]:
        try:
            return app.state.router.score(request)
        except (httpx.HTTPError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.post("/prepare")
    def prepare(request: BaselinePrepareRequest) -> dict[str, Any]:
        try:
            return app.state.router.prepare(request)
        except (httpx.HTTPError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker-urls",
        default="http://127.0.0.1:30490/score,http://127.0.0.1:30491/score",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30489)
    args = parser.parse_args()
    worker_urls = tuple(args.worker_urls.split(","))
    if not worker_urls:
        raise ValueError("--worker-urls must contain at least one URL")
    uvicorn.run(
        create_app(worker_urls),
        host=args.host,
        port=args.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
