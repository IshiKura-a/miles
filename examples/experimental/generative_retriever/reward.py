"""HTTP reward hook for generative-retriever rollouts.

Configure this module with ``--custom-rm-path
examples.experimental.generative_retriever.reward.reward``.  The configured
endpoint receives ``{"samples": [...]}`` and must return
``{"rewards": [float, ...]}`` with exactly one finite reward per input sample.
"""

from __future__ import annotations

import math
import asyncio
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from weakref import WeakKeyDictionary

from miles.utils.types import Sample
from examples.experimental.generative_retriever.relevance_reward import REWARD_MODES
from examples.experimental.generative_retriever.reward_http import post_json_async

_MAX_REWARD_SAMPLES_PER_REQUEST = 2048
_REWARD_BATCH_WAIT_SECONDS = 0.01


@dataclass(frozen=True)
class _BatchResult:
    rewards: list[float]
    details: list[dict[str, Any]]
    cache: dict[str, Any]
    request_metrics: dict[str, Any]


@dataclass
class _PendingRequest:
    payloads: list[dict[str, Any]]
    future: asyncio.Future[_BatchResult]


class _RewardBatcher:
    def __init__(
        self, url: str, mode: str, timeout_seconds: float, *,
        max_samples: int = _MAX_REWARD_SAMPLES_PER_REQUEST, concurrency: int = 1,
    ) -> None:
        if max_samples <= 0 or concurrency <= 0:
            raise ValueError("Reward request batch size and concurrency must be positive")
        self._url = url
        self._mode = mode
        self._timeout_seconds = timeout_seconds
        self._max_samples = max_samples
        self._slots = asyncio.Semaphore(concurrency)
        self._requests: set[asyncio.Task[None]] = set()
        self._pending: deque[_PendingRequest] = deque()
        self._flush_task: asyncio.Task[None] | None = None

    async def submit(self, payloads: list[dict[str, Any]]) -> _BatchResult:
        if not 0 < len(payloads) <= self._max_samples:
            raise ValueError(
                f"reward batch must contain 1-{self._max_samples} samples"
            )
        future = asyncio.get_running_loop().create_future()
        self._pending.append(_PendingRequest(payloads=payloads, future=future))
        if self._flush_task is None:
            self._flush_task = asyncio.create_task(self._flush())
        return await future

    def _take_batch(self) -> list[_PendingRequest]:
        requests = []
        sample_count = 0
        while self._pending:
            request = self._pending[0]
            if requests and sample_count + len(request.payloads) > self._max_samples:
                break
            requests.append(self._pending.popleft())
            sample_count += len(request.payloads)
        return requests

    async def _flush(self) -> None:
        try:
            while self._pending:
                await asyncio.sleep(_REWARD_BATCH_WAIT_SECONDS)
                await self._slots.acquire()
                requests = self._take_batch()
                task = asyncio.create_task(self._dispatch(requests))
                self._requests.add(task)
                task.add_done_callback(self._requests.discard)
        finally:
            self._flush_task = None
            if self._pending:
                self._flush_task = asyncio.create_task(self._flush())

    async def _dispatch(self, requests: list[_PendingRequest]) -> None:
        try:
            result = await self._post(requests)
            self._resolve(requests, result)
        except asyncio.CancelledError:
            for request in requests:
                request.future.cancel()
            raise
        except Exception as error:
            for request in requests:
                if not request.future.done():
                    request.future.set_exception(error)
        finally:
            self._slots.release()

    async def _post(self, requests: list[_PendingRequest]) -> _BatchResult:
        payloads = [payload for request in requests for payload in request.payloads]
        async with asyncio.timeout(self._timeout_seconds):
            response = await post_json_async(
                self._url, {"samples": payloads, "reward_mode": self._mode},
                timeout=self._timeout_seconds,
            )
            rewards, details, cache, request_metrics = _parse_rewards(
                response,
                expected_count=len(payloads),
            )
        return _BatchResult(rewards, details, cache, request_metrics)

    @staticmethod
    def _resolve(requests: list[_PendingRequest], result: _BatchResult) -> None:
        offset = 0
        for index, request in enumerate(requests):
            end = offset + len(request.payloads)
            if not request.future.done():
                request.future.set_result(
                    _BatchResult(
                        rewards=result.rewards[offset:end],
                        details=result.details[offset:end],
                        cache=result.cache if index == 0 else {},
                        request_metrics=result.request_metrics if index == 0 else {},
                    )
                )
            offset = end


_BATCHERS: WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[tuple[str, str, float, int, int], _RewardBatcher]
] = WeakKeyDictionary()


def _get_batcher(
    url: str, mode: str, timeout_seconds: float, *,
    max_samples: int = _MAX_REWARD_SAMPLES_PER_REQUEST, concurrency: int = 1,
) -> _RewardBatcher:
    loop = asyncio.get_running_loop()
    batchers = _BATCHERS.setdefault(loop, {})
    key = (url, mode, timeout_seconds, max_samples, concurrency)
    batcher = batchers.get(key)
    if batcher is None:
        batcher = _RewardBatcher(
            url, mode, timeout_seconds, max_samples=max_samples, concurrency=concurrency,
        )
        batchers[key] = batcher
    return batcher


def _sample_payload(sample: Sample) -> dict[str, Any]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    retriever_metadata = metadata.get("generative_retriever", {})
    if not isinstance(retriever_metadata, dict):
        raise ValueError("generative retriever metadata must be a dictionary")
    docid_tokens = retriever_metadata.get("docid_tokens")
    if not isinstance(docid_tokens, list) or not all(isinstance(token, int) for token in docid_tokens):
        raise ValueError("generative retriever reward requires generated integer docid_tokens")

    return {
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
        "docid_tokens": docid_tokens,
        "metadata": metadata,
    }


def _parse_rewards(
    response: Any,
    expected_count: int,
) -> tuple[list[float], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if not isinstance(response, dict):
        raise ValueError("generative retriever reward endpoint must return a JSON object")
    raw_rewards = response.get("rewards")
    if not isinstance(raw_rewards, Sequence) or isinstance(raw_rewards, (str, bytes)):
        raise ValueError("generative retriever reward response is missing a rewards list")
    if len(raw_rewards) != expected_count:
        raise ValueError(
            "generative retriever reward response count does not match request: "
            f"expected {expected_count}, got {len(raw_rewards)}"
        )

    rewards = []
    for index, value in enumerate(raw_rewards):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"generative retriever reward at index {index} is not numeric: {value!r}")
        reward = float(value)
        if not math.isfinite(reward):
            raise ValueError(f"generative retriever reward at index {index} is not finite: {value!r}")
        rewards.append(reward)
    details = response.get("details")
    if not isinstance(details, list) or len(details) != expected_count:
        raise ValueError("generative retriever reward response has invalid details")
    if not all(isinstance(detail, dict) for detail in details):
        raise ValueError("generative retriever reward details must be JSON objects")
    cache = response.get("cache", {})
    if not isinstance(cache, dict):
        raise ValueError("generative retriever reward cache metrics must be a JSON object")
    cache_delta = response.get("cache_delta", {})
    if not isinstance(cache_delta, dict):
        raise ValueError("generative retriever reward cache deltas must be a JSON object")
    routed_samples = response.get("routed_samples", {})
    if not isinstance(routed_samples, dict):
        raise ValueError("generative retriever routing metrics must be a JSON object")
    latency = response.get("latency_seconds")
    if (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(latency)
        or latency < 0
    ):
        raise ValueError("generative retriever reward response has invalid latency")
    return rewards, details, cache, {
        "cache": cache_delta, "routing": routed_samples,
        "score_latency_seconds": float(latency),
    }


def _store_breakdown(
    sample: Sample,
    detail: dict[str, Any],
    cache: dict[str, Any],
    request_metrics: dict[str, Any],
) -> None:
    keywords = detail.get("keywords")
    if not isinstance(keywords, list) or not keywords:
        raise ValueError("generative retriever reward detail must contain keyword breakdowns")
    components = (
        "pm_score",
        "sm_score",
        "pm_pass",
        "sm_pass",
        "normalized_pm",
        "normalized_sm",
        "pm_relevance",
        "sm_relevance",
        "relevance_quality",
        "pm_unique_bonus",
        "sm_unique_bonus",
        "unique_bonus",
        "discrete_base",
        "continuous_base",
        "discrete_total",
        "continuous_total",
        "reward",
    )
    metrics = {}
    for component in components:
        values = [keyword.get(component) for keyword in keywords]
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            raise ValueError(f"invalid reward component {component!r}")
        metrics[component] = math.fsum(float(value) for value in values) / len(values)
    for metric, component in (
        ("pm_unique_keyword_rate", "is_pm_unique"),
        ("sm_unique_keyword_rate", "is_sm_unique"),
        ("unique_keyword_rate", "is_unique"),
        ("relevance_eligible_rate", "relevance_eligible"),
        ("pm_unique_eligible_rate", "pm_unique_eligible"),
        ("sm_unique_eligible_rate", "sm_unique_eligible"),
        ("unique_eligible_rate", "unique_eligible"),
    ):
        values = [keyword.get(component) for keyword in keywords]
        if not all(isinstance(value, bool) for value in values):
            raise ValueError(f"invalid reward component {component!r}")
        metrics[metric] = math.fsum(float(value) for value in values) / len(values)
    unique_quality = [
        float(keyword["relevance_quality"])
        for keyword in keywords
        if keyword["is_pm_unique"] or keyword["is_sm_unique"]
    ]
    metrics["unique_quality"] = (
        math.fsum(unique_quality) / len(unique_quality)
        if unique_quality
        else 0.0
    )
    metrics["train_total"] = metrics["reward"]
    if "score_latency_seconds" in request_metrics:
        metrics["score/latency_seconds"] = request_metrics["score_latency_seconds"]
        metrics["score/request_count"] = 1.0

    def collect_cache_metrics(prefix: str, values: dict[str, Any], metric_prefix: str) -> None:
        for name, value in values.items():
            path = f"{prefix}/{name}" if prefix else name
            if isinstance(value, dict):
                collect_cache_metrics(path, value, metric_prefix)
            elif (
                name in {"hits", "misses", "evictions", "size", "capacity"}
                and isinstance(value, int)
                and not isinstance(value, bool)
            ):
                metrics[f"{metric_prefix}/{path}"] = value

    collect_cache_metrics("", cache, "cache_gauge")
    collect_cache_metrics("", request_metrics.get("cache", {}), "cache_delta")
    routed_samples = request_metrics.get("routing", {})
    for worker, value in routed_samples.items():
        if isinstance(value, int) and not isinstance(value, bool):
            metrics[f"routing_delta/{worker}/samples"] = value
    sample.metadata.setdefault("generative_retriever", {})["reward_metrics"] = metrics


async def reward(args: Any, samples: Sample | list[Sample], **kwargs: Any) -> float | list[float]:
    """Score generated docids through the configured retrieval-reward service."""
    url = getattr(args, "generative_retriever_reward_url", None)
    if not url:
        raise ValueError(
            "--generative-retriever-reward-url is required when using "
            "examples.experimental.generative_retriever.reward.reward"
        )

    batch = samples if isinstance(samples, list) else [samples]
    timeout_seconds = getattr(args, "generative_retriever_reward_timeout", 30.0)
    if timeout_seconds <= 0:
        raise ValueError("generative_retriever_reward_timeout must be positive")

    mode = getattr(args, "generative_retriever_reward_mode", "discrete_relevance_unique")
    if mode not in REWARD_MODES:
        raise ValueError(f"Unknown reward mode: {mode!r}")
    rewards: list[float] = []
    max_samples = args.generative_retriever_reward_batch_size
    concurrency = args.generative_retriever_reward_concurrency
    batcher = _get_batcher(
        url, mode, timeout_seconds, max_samples=max_samples, concurrency=concurrency,
    )
    chunks = [batch[start:start + max_samples] for start in range(0, len(batch), max_samples)]
    results = await asyncio.gather(*(
        batcher.submit([_sample_payload(sample) for sample in chunk])
        for chunk in chunks
    ))
    for sample_chunk, result in zip(chunks, results, strict=True):
        rewards.extend(result.rewards)
        for index, (sample, detail) in enumerate(
            zip(sample_chunk, result.details, strict=True)
        ):
            _store_breakdown(
                sample,
                detail,
                result.cache if index == 0 else {},
                result.request_metrics if index == 0 else {},
            )

    return rewards if isinstance(samples, list) else rewards[0]
