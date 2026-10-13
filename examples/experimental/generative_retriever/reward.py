"""HTTP reward hook for generative-retriever rollouts.

Configure this module with ``--custom-rm-path
examples.experimental.generative_retriever.reward.reward``.  The configured
endpoint receives ``{"samples": [...]}`` and must return
``{"rewards": [float, ...]}`` with exactly one finite reward per input sample.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import aiohttp

from miles.utils.types import Sample
from examples.experimental.generative_retriever.relevance_reward import REWARD_MODES

_MAX_REWARD_SAMPLES_PER_REQUEST = 512


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
    return rewards, details, cache, {"cache": cache_delta, "routing": routed_samples}


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
        ("unique_keyword_rate", "is_unique"),
        ("relevance_eligible_rate", "relevance_eligible"),
        ("unique_eligible_rate", "unique_eligible"),
    ):
        values = [keyword.get(component) for keyword in keywords]
        if not all(isinstance(value, bool) for value in values):
            raise ValueError(f"invalid reward component {component!r}")
        metrics[metric] = math.fsum(float(value) for value in values) / len(values)
    unique_quality = [
        float(keyword["relevance_quality"])
        for keyword in keywords
        if keyword["is_unique"]
    ]
    metrics["unique_quality"] = (
        math.fsum(unique_quality) / len(unique_quality)
        if unique_quality
        else 0.0
    )
    metrics["train_total"] = metrics["reward"]

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

    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    mode = getattr(args, "generative_retriever_reward_mode", "discrete_relevance_unique")
    if mode not in REWARD_MODES:
        raise ValueError(f"Unknown reward mode: {mode!r}")
    rewards: list[float] = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for start in range(0, len(batch), _MAX_REWARD_SAMPLES_PER_REQUEST):
            sample_chunk = batch[start:start + _MAX_REWARD_SAMPLES_PER_REQUEST]
            async with session.post(
                url,
                json={
                    "samples": [_sample_payload(sample) for sample in sample_chunk],
                    "reward_mode": mode,
                },
            ) as response:
                response.raise_for_status()
                chunk_rewards, details, cache, request_metrics = _parse_rewards(
                    await response.json(),
                    expected_count=len(sample_chunk),
                )
            rewards.extend(chunk_rewards)
            for index, (sample, detail) in enumerate(zip(sample_chunk, details, strict=True)):
                _store_breakdown(
                    sample,
                    detail,
                    cache if index == 0 else {},
                    request_metrics if index == 0 else {},
                )

    return rewards if isinstance(samples, list) else rewards[0]
