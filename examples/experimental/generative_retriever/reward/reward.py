"""HTTP reward hook for generative-retriever rollouts.

Configure this module with ``--custom-rm-path
examples.experimental.generative_retriever.reward.reward.reward``.  The configured
endpoint receives ``{"samples": [...]}`` and must return
``{"rewards": [float, ...]}`` with exactly one finite reward per input sample.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any
from weakref import WeakKeyDictionary

from miles.utils.types import Sample
from examples.experimental.generative_retriever.reward.relevance_reward import REWARD_MODES
from examples.experimental.generative_retriever.reward.reward_http import (
    AsyncRequestBatcher,
    PendingBatchRequest,
    post_json_async,
)

_MAX_REWARD_SAMPLES_PER_REQUEST = 2048
_REWARD_BATCH_WAIT_SECONDS = 0.01
_INLINE_METRICS_SAMPLE_LIMIT = 16
_NUMERIC_COMPONENTS = (
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
_BOOLEAN_RATE_COMPONENTS = (
    ("pm_unique_keyword_rate", "is_pm_unique"),
    ("sm_unique_keyword_rate", "is_sm_unique"),
    ("unique_keyword_rate", "is_unique"),
    ("relevance_eligible_rate", "relevance_eligible"),
    ("pm_unique_eligible_rate", "pm_unique_eligible"),
    ("sm_unique_eligible_rate", "sm_unique_eligible"),
    ("unique_eligible_rate", "unique_eligible"),
)


@dataclass(frozen=True, slots=True)
class BatchResult:
    rewards: list[float]
    details: list[dict[str, Any]]
    cache: dict[str, Any]
    request_metrics: dict[str, Any]


class RewardBatcher(AsyncRequestBatcher[list[dict[str, Any]], BatchResult]):
    def __init__(
        self,
        url: str,
        mode: str,
        timeout_seconds: float,
        *,
        max_samples: int = _MAX_REWARD_SAMPLES_PER_REQUEST,
        concurrency: int = 1,
    ) -> None:
        super().__init__(
            max_batch_size=max_samples,
            concurrency=concurrency,
            batch_wait_seconds=_REWARD_BATCH_WAIT_SECONDS,
        )
        self._url = url
        self._mode = mode
        self._timeout_seconds = timeout_seconds

    async def submit(self, payloads: list[dict[str, Any]]) -> BatchResult:
        if not 0 < len(payloads) <= self._max_batch_size:
            raise ValueError(
                f"reward batch must contain 1-{self._max_batch_size} samples"
            )
        return await self.submit_weighted(payloads, weight=len(payloads))

    async def execute_batch(
        self,
        requests: list[PendingBatchRequest[list[dict[str, Any]], BatchResult]],
    ) -> None:
        payloads = [payload for request in requests for payload in request.payload]
        response = await post_json_async(
            self._url,
            {"samples": payloads, "reward_mode": self._mode},
            timeout=self._timeout_seconds,
        )
        rewards, details, cache, request_metrics = parse_rewards(
            response,
            expected_count=len(payloads),
        )
        result = BatchResult(rewards, details, cache, request_metrics)
        self.resolve_requests(requests, result)

    @staticmethod
    def resolve_requests(
        requests: list[PendingBatchRequest[list[dict[str, Any]], BatchResult]],
        result: BatchResult,
    ) -> None:
        offset = 0
        for index, request in enumerate(requests):
            end = offset + len(request.payload)
            if not request.future.done():
                request.future.set_result(
                    BatchResult(
                        rewards=result.rewards[offset:end],
                        details=result.details[offset:end],
                        cache=result.cache if index == 0 else {},
                        request_metrics=result.request_metrics if index == 0 else {},
                    )
                )
            offset = end


_BATCHERS: WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[tuple[str, str, float, int, int], RewardBatcher]
] = WeakKeyDictionary()


def get_reward_batcher(
    url: str,
    mode: str,
    timeout_seconds: float,
    *,
    max_samples: int = _MAX_REWARD_SAMPLES_PER_REQUEST,
    concurrency: int = 1,
) -> RewardBatcher:
    loop = asyncio.get_running_loop()
    batchers = _BATCHERS.setdefault(loop, {})
    key = (url, mode, timeout_seconds, max_samples, concurrency)
    batcher = batchers.get(key)
    if batcher is None:
        batcher = RewardBatcher(
            url, mode, timeout_seconds, max_samples=max_samples, concurrency=concurrency,
        )
        batchers[key] = batcher
    return batcher


def sample_payload(sample: Sample) -> dict[str, Any]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    retriever_metadata = metadata.get("generative_retriever", {})
    if not isinstance(retriever_metadata, dict):
        raise ValueError("generative retriever metadata must be a dictionary")
    docid_tokens = retriever_metadata.get("docid_tokens")
    if not isinstance(docid_tokens, list) or not all(isinstance(token, int) for token in docid_tokens):
        raise ValueError("generative retriever reward requires generated integer docid_tokens")

    payload: dict[str, Any] = {
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
        "docid_tokens": docid_tokens,
        "metadata": metadata,
    }
    baseline_docids = retriever_metadata.pop("baseline_docids", None)
    if baseline_docids is not None:
        payload["baseline_docids"] = baseline_docids
    return payload


def parse_rewards(
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
        reward_value = float(value)
        if not math.isfinite(reward_value):
            raise ValueError(f"generative retriever reward at index {index} is not finite: {value!r}")
        rewards.append(reward_value)
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
        "cache": cache_delta,
        "routing": routed_samples,
        "score_latency_seconds": float(latency),
    }


def extract_detail_metrics(detail: dict[str, Any]) -> dict[str, Any]:
    precomputed = detail.get("metrics")
    if isinstance(precomputed, dict) and precomputed:
        return dict(precomputed)

    keywords = detail.get("keywords")
    if not isinstance(keywords, list) or not keywords:
        raise ValueError("generative retriever reward detail must contain keyword breakdowns")
    metrics: dict[str, Any] = {}
    for component in _NUMERIC_COMPONENTS:
        values = [keyword.get(component) for keyword in keywords]
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            raise ValueError(f"invalid reward component {component!r}")
        metrics[component] = math.fsum(float(value) for value in values) / len(values)
    for metric, component in _BOOLEAN_RATE_COMPONENTS:
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
    return metrics


def attach_request_metrics(
    metrics: dict[str, Any],
    cache: dict[str, Any],
    request_metrics: dict[str, Any],
) -> None:
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


def store_chunk_breakdown(
    sample_chunk: list[Sample],
    result: BatchResult,
) -> None:
    if len(sample_chunk) <= _INLINE_METRICS_SAMPLE_LIMIT:
        for index, (sample, detail) in enumerate(zip(sample_chunk, result.details, strict=True)):
            metrics = extract_detail_metrics(detail)
            if index == 0:
                attach_request_metrics(metrics, result.cache, result.request_metrics)
            sample.metadata.setdefault("generative_retriever", {})["reward_metrics"] = metrics
        return

    detail_metrics = [extract_detail_metrics(detail) for detail in result.details]
    count = float(len(detail_metrics))
    aggregated: dict[str, Any] = {
        key: math.fsum(float(item[key]) for item in detail_metrics) / count
        for key in detail_metrics[0]
    }
    attach_request_metrics(aggregated, result.cache, result.request_metrics)
    sample_chunk[0].metadata.setdefault("generative_retriever", {})["reward_metrics"] = aggregated


async def reward(args: Any, samples: Sample | list[Sample], **kwargs: Any) -> float | list[float]:
    """Score generated docids through the configured retrieval-reward service."""
    del kwargs
    url = args.generative_retriever_reward_url
    if not url:
        raise ValueError(
            "--generative-retriever-reward-url is required when using "
            "examples.experimental.generative_retriever.reward.reward.reward"
        )

    batch = samples if isinstance(samples, list) else [samples]
    timeout_seconds = args.generative_retriever_reward_timeout
    if timeout_seconds <= 0:
        raise ValueError("generative_retriever_reward_timeout must be positive")

    mode = args.generative_retriever_reward_mode
    if mode not in REWARD_MODES:
        raise ValueError(f"Unknown reward mode: {mode!r}")
    rewards: list[float] = []
    max_samples = args.generative_retriever_reward_batch_size
    concurrency = args.generative_retriever_reward_concurrency
    batcher = get_reward_batcher(
        url, mode, timeout_seconds, max_samples=max_samples, concurrency=concurrency,
    )
    chunks = [batch[start:start + max_samples] for start in range(0, len(batch), max_samples)]
    results = await asyncio.gather(*(
        batcher.submit([sample_payload(sample) for sample in chunk])
        for chunk in chunks
    ))
    for sample_chunk, result in zip(chunks, results, strict=True):
        rewards.extend(result.rewards)
        store_chunk_breakdown(sample_chunk, result)

    return rewards if isinstance(samples, list) else rewards[0]
