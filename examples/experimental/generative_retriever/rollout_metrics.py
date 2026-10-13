"""Add GR reward-component and cache metrics to the normal rollout report."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def log_rollout(
    rollout_id: int,
    args: Any,
    samples: list[Any],
    rollout_extra_metrics: dict[str, Any] | None,
    rollout_time: float,
) -> bool:
    del rollout_id, args, rollout_time
    if rollout_extra_metrics is None:
        return False
    values: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        metadata = sample.metadata.get("generative_retriever", {})
        for name, value in metadata.get("reward_metrics", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                values[name].append(float(value))
    for name, metrics in values.items():
        if name.startswith(("cache_delta/", "routing_delta/")):
            value = math.fsum(metrics)
        elif name.startswith("cache_gauge/"):
            value = metrics[-1]
        else:
            value = math.fsum(metrics) / len(metrics)
        rollout_extra_metrics[f"rollout/generative_retriever/{name}"] = value
    groups: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    for sample in samples:
        metadata = sample.metadata.get("generative_retriever", {})
        positional_docid = metadata.get("positional_docid")
        if (
            sample.group_index is not None
            and isinstance(positional_docid, list)
            and all(isinstance(value, int) for value in positional_docid)
        ):
            groups[sample.group_index].append(tuple(positional_docid))
    if groups:
        distinct_ratios = [
            len(set(docids)) / len(docids)
            for docids in groups.values()
            if docids
        ]
        rollout_extra_metrics["rollout/generative_retriever/distinct_docid_ratio"] = (
            math.fsum(distinct_ratios) / len(distinct_ratios)
        )
    return False
