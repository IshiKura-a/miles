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


def _parse_rewards(response: Any, expected_count: int) -> list[float]:
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
    return rewards


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
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json={"samples": [_sample_payload(sample) for sample in batch]}) as response:
            response.raise_for_status()
            rewards = _parse_rewards(await response.json(), expected_count=len(batch))

    return rewards if isinstance(samples, list) else rewards[0]
