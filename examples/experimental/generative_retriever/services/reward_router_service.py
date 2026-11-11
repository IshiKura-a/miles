"""Route reward batches across isolated single-GPU reward workers."""

from __future__ import annotations

import argparse
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any

import anyio.to_thread
import httpx
import uvicorn
import xxhash
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from examples.experimental.generative_retriever.services.mapping_service import SaveMappingRequest
from examples.experimental.generative_retriever.services.reward_service import (
    BaselinePrepareRequest,
    RewardRequest,
    reward_request_model,
)
from examples.experimental.generative_retriever.reward.relevance_reward import (
    DEFAULT_REWARD_BATCH_SIZE,
    DOCID_MAX_KEYWORDS,
)
from examples.experimental.generative_retriever.reward.reward_http import post_json_sync

REWARD_ROUTER_API_VERSION = 7
_SERVICE_API_VERSION = REWARD_ROUTER_API_VERSION
logger = logging.getLogger(__name__)


class StepRequest(BaseModel):
    rollout_id: int


class RewardRouter:
    def __init__(
        self, worker_urls: tuple[str, ...], *,
        max_samples_per_request: int = DEFAULT_REWARD_BATCH_SIZE, forward_concurrency: int = 1,
    ) -> None:
        if not worker_urls or len(set(worker_urls)) != len(worker_urls):
            raise ValueError("Reward router requires distinct worker URLs")
        if forward_concurrency <= 0:
            raise ValueError("Reward forwarding concurrency must be positive")
        self.worker_urls = worker_urls
        self._request_model = reward_request_model(max_samples_per_request)
        self.max_samples_per_request = max_samples_per_request
        self.forward_concurrency = forward_concurrency
        self._executor = ThreadPoolExecutor(max_workers=len(worker_urls) * forward_concurrency)
        connections = len(worker_urls) * (forward_concurrency + 1)
        self._client = httpx.Client(
            timeout=900.0,
            limits=httpx.Limits(
                max_connections=connections, max_keepalive_connections=0,
            ),
            trust_env=False,
        )
        self._routing_lock = Lock()
        self._routed_samples = [0] * len(worker_urls)
        self._prepared_queries = [0] * len(worker_urls)
        self._step_updates: dict[int, tuple[list[str], dict[str, Any]]] = {}

    def _worker_index(self, query: str) -> int:
        return xxhash.xxh64_intdigest(query) % len(self.worker_urls)

    def _post(self, url: str, request: RewardRequest | BaselinePrepareRequest) -> dict[str, Any]:
        return post_json_sync(self._client, url, request.model_dump(mode="json"))

    def score(self, request: RewardRequest) -> dict[str, Any]:
        started = time.perf_counter()
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
                self._request_model(
                    samples=[sample for _, sample in bucket],
                    reward_mode=request.reward_mode,
                ),
            ))
            for index, (url, bucket) in enumerate(zip(self.worker_urls, buckets, strict=True))
            if bucket
        ]
        responses = {index: future.result() for index, future in futures}
        forwarded_at = time.perf_counter()
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
        result = {
            "rewards": [rewards[index] for index in range(len(request.samples))],
            "details": [details[index] for index in range(len(request.samples))],
            "reward_mode": request.reward_mode,
            **self._response_metrics(responses),
            "routed_samples": {
                f"worker_{index}": len(bucket)
                for index, bucket in enumerate(buckets)
            },
        }
        logger.info(
            "reward_route_score samples=%d active_workers=%d forward_s=%.3f "
            "aggregate_s=%.3f total_s=%.3f",
            len(request.samples), len(responses), forwarded_at - started,
            time.perf_counter() - forwarded_at, time.perf_counter() - started,
        )
        return result

    def prepare(self, request: BaselinePrepareRequest) -> dict[str, Any]:
        started = time.perf_counter()
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
        result = {
            "prepared_queries": sum(
                response["prepared_queries"] for response in responses.values()
            ),
            **self._response_metrics(responses),
            "routed_queries": {
                f"worker_{index}": len(bucket)
                for index, bucket in enumerate(buckets)
            },
        }
        logger.info(
            "reward_route_prepare queries=%d active_workers=%d total_s=%.3f",
            len(request.queries), len(responses), time.perf_counter() - started,
        )
        return result

    def stage_step_updates(self, rollout_id: int) -> dict[str, int]:
        futures = [
            self._executor.submit(post_json_sync, self._client, url.removesuffix("/score") + "/harvest_step_delta", {})
            for url in self.worker_urls
        ]
        harvested = [future.result() for future in futures]
        mapping_urls = list(dict.fromkeys(item["mapping_url"].rstrip("/") for item in harvested))

        def resolve(item: dict[str, Any]) -> dict[str, Any]:
            keywords = list(dict.fromkeys(
                keyword for kind in ("additions", "removals")
                for entry in item[kind] for keyword in entry["keywords"]
            ))
            if not keywords:
                return {"additions": [], "removals": []}
            payload = post_json_sync(
                self._client, item["mapping_url"].rstrip("/") + "/resolve_keywords", {"keywords": keywords},
            )
            keyword_ids = dict(zip(keywords, payload["keyword_ids"], strict=True))
            return {
                kind: [
                    {"docid": entry["docid"], "keyword_ids": [keyword_ids[kw] for kw in entry["keywords"]]}
                    for entry in item[kind]
                ]
                for kind in ("additions", "removals")
            }

        resolved = list(self._executor.map(resolve, harvested))
        merged: dict[str, dict[tuple[int, ...], list[int]]] = {"additions": {}, "removals": {}}
        for item in resolved:
            for kind in merged:
                for entry in item[kind]:
                    bucket = merged[kind].setdefault(tuple(entry["docid"]), [])
                    bucket.extend(kid for kid in entry["keyword_ids"] if kid not in bucket)
        delta: dict[str, Any] = {
            kind: [{"docid": list(docid), "keyword_ids": kids} for docid, kids in entries.items()]
            for kind, entries in merged.items()
        }
        delta["k_max"] = DOCID_MAX_KEYWORDS
        with self._routing_lock:
            if rollout_id in self._step_updates:
                raise ValueError(f"Updates already staged for rollout {rollout_id}")
            self._step_updates[rollout_id] = (mapping_urls, delta)
        return {"staged_docids": sum(len(entries) for entries in merged.values())}

    def commit_step_updates(self, rollout_id: int) -> dict[str, int]:
        with self._routing_lock:
            steps = sorted(step for step in self._step_updates if step <= rollout_id)
        if not steps or steps[-1] != rollout_id:
            raise ValueError(f"Missing staged mapping updates for rollout {rollout_id}")
        updated_docids = 0
        for step in steps:
            urls, delta = self._step_updates[step]
            if delta["additions"] or delta["removals"]:
                futures = [
                    self._executor.submit(post_json_sync, self._client, url + "/update_delta_ids", delta)
                    for url in urls
                ]
                results = [future.result() for future in futures]
                updated_docids += results[0]["updated_docids"]
            with self._routing_lock:
                del self._step_updates[step]
        return {"updated_docids": updated_docids}

    def reset_step_updates(self) -> None:
        futures = [
            self._executor.submit(post_json_sync, self._client, url.removesuffix("/score") + "/reset_step_updates", {})
            for url in self.worker_urls
        ]
        for future in futures:
            future.result()
        with self._routing_lock:
            self._step_updates.clear()

    def save_mapping(self, output_dir: str) -> dict[str, Any]:
        started = time.perf_counter()
        response = self._client.get(
            self.worker_urls[0].removesuffix("/score") + "/health",
            timeout=10.0,
        )
        response.raise_for_status()
        payload = response.json()
        mapping_url = payload.get("service_config", {}).get("mapping_url")
        if not isinstance(mapping_url, str) or not mapping_url.strip():
            raise RuntimeError("Primary reward worker did not report a valid mapping_url")
        result = post_json_sync(
            self._client,
            mapping_url.rstrip("/") + "/save",
            {"output_dir": output_dir},
        )
        return {
            **result,
            "latency_seconds": time.perf_counter() - started,
        }

    @staticmethod
    def _response_metrics(responses: dict[int, dict[str, Any]]) -> dict[str, Any]:
        version = next(iter(responses.values())).get("mapping_version")
        if not isinstance(version, str) or not version:
            raise ValueError("Reward worker returned an invalid mapping version")
        if any(response.get("mapping_version") != version for response in responses.values()):
            raise ValueError("Reward workers use different mappings")
        merged_cache: dict[str, dict[str, int]] = {}
        merged_delta: dict[str, dict[str, int]] = {}
        for response in responses.values():
            for table_name, table_stats in response.get("cache", {}).items():
                if isinstance(table_stats, dict):
                    dst = merged_cache.setdefault(table_name, {})
                    for k, v in table_stats.items():
                        if isinstance(v, int) and not isinstance(v, bool):
                            dst[k] = dst.get(k, 0) + v
            for table_name, table_delta in response.get("cache_delta", {}).items():
                if isinstance(table_delta, dict):
                    dst = merged_delta.setdefault(table_name, {})
                    for k, v in table_delta.items():
                        if isinstance(v, int) and not isinstance(v, bool):
                            dst[k] = dst.get(k, 0) + v
        return {
            "mapping_version": version,
            "cache": merged_cache,
            "cache_delta": merged_delta,
            "latency_seconds": max(response["latency_seconds"] for response in responses.values()),
        }

    def health(self) -> dict[str, Any]:
        responses = []
        for url in self.worker_urls:
            response = self._client.get(url.removesuffix("/score") + "/health", timeout=10.0)
            response.raise_for_status()
            payload = response.json()
            worker_count = payload.get("worker_count")
            if (
                payload.get("status") != "ready"
                or not isinstance(worker_count, int)
                or worker_count < 1
            ):
                raise RuntimeError(f"Reward worker is not ready: {payload}")
            worker_limit = payload["service_config"]["max_samples_per_request"]
            if worker_limit < self.max_samples_per_request:
                raise RuntimeError(
                    f"Reward worker request limit {worker_limit} is below router limit "
                    f"{self.max_samples_per_request}"
                )
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
                "max_samples_per_request": self.max_samples_per_request,
                "forward_concurrency": self.forward_concurrency,
            },
        }

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        self._client.close()


def create_app(
    worker_urls: tuple[str, ...], *,
    max_samples_per_request: int = 4096, forward_concurrency: int = 1,
) -> FastAPI:
    request_model = reward_request_model(max_samples_per_request)
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        anyio.to_thread.current_default_thread_limiter().total_tokens = 1024
        app.state.router = RewardRouter(
            worker_urls, max_samples_per_request=max_samples_per_request,
            forward_concurrency=forward_concurrency,
        )
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

    def score(request: RewardRequest) -> dict[str, Any]:
        try:
            return app.state.router.score(request)
        except (httpx.HTTPError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    score.__annotations__["request"] = request_model
    app.post("/score")(score)

    @app.post("/prepare")
    def prepare(request: BaselinePrepareRequest) -> dict[str, Any]:
        try:
            return app.state.router.prepare(request)
        except (httpx.HTTPError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.post("/stage_step_updates")
    def stage_step_updates(request: StepRequest) -> dict[str, Any]:
        try:
            return app.state.router.stage_step_updates(request.rollout_id)
        except (httpx.HTTPError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.post("/commit_step_updates")
    def commit_step_updates(request: StepRequest) -> dict[str, Any]:
        try:
            return app.state.router.commit_step_updates(request.rollout_id)
        except (httpx.HTTPError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.post("/reset_step_updates")
    def reset_step_updates() -> None:
        try:
            app.state.router.reset_step_updates()
        except (httpx.HTTPError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.post("/save_mapping")
    def save_mapping(request: SaveMappingRequest) -> dict[str, Any]:
        try:
            return app.state.router.save_mapping(request.output_dir)
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
    parser.add_argument("--max-samples-per-request", type=int, default=4096)
    parser.add_argument("--forward-concurrency", type=int, default=1)
    args = parser.parse_args()
    worker_urls = tuple(args.worker_urls.split(","))
    if not worker_urls:
        raise ValueError("--worker-urls must contain at least one URL")
    uvicorn.run(
        create_app(
            worker_urls, max_samples_per_request=args.max_samples_per_request,
            forward_concurrency=args.forward_concurrency,
        ),
        host=args.host,
        port=args.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
