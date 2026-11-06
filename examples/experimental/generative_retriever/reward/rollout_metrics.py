"""Add GR reward-component and cache metrics to the normal rollout report."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any

import httpx
import numpy as np

logger = logging.getLogger(__name__)


def _mean_top_k(values: list[float], k: int) -> float:
    if not values:
        return 0.0
    if len(values) <= k:
        return math.fsum(values) / len(values)
    top_values = sorted(values, reverse=True)[:k]
    return math.fsum(top_values) / len(top_values)


def apply_hierarchical_tree_credit(args: Any, samples: list[Any]) -> None:
    grouped: dict[int, list[tuple[Any, tuple[int, int, int, int], float]]] = defaultdict(list)
    for sample in samples:
        if sample.group_index is None:
            continue
        metadata = sample.metadata.get("generative_retriever", {}) if isinstance(sample.metadata, dict) else {}
        docid = metadata.get("positional_docid")
        if (
            isinstance(docid, list)
            and len(docid) == 4
            and all(isinstance(v, int) for v in docid)
        ):
            raw_val = float(sample.get_reward_value(args))
            sample.metadata["raw_reward"] = raw_val
            grouped[int(sample.group_index)].append(
                (sample, (docid[0], docid[1], docid[2], docid[3]), raw_val)
            )
    for items in grouped.values():
        if len(items) <= 1:
            continue
        by_p0: dict[int, list[float]] = defaultdict(list)
        by_p1: dict[tuple[int, int], list[float]] = defaultdict(list)
        by_p2: dict[tuple[int, int, int], list[float]] = defaultdict(list)
        for _, d, r in items:
            by_p0[d[0]].append(r)
            by_p1[(d[0], d[1])].append(r)
            by_p2[(d[0], d[1], d[2])].append(r)
        v0_map = {k: _mean_top_k(v, 8) for k, v in by_p0.items()}
        v1_map = {k: _mean_top_k(v, 4) for k, v in by_p1.items()}
        v2_map = {k: _mean_top_k(v, 2) for k, v in by_p2.items()}
        for sample, d, r in items:
            v0 = v0_map[d[0]]
            v1 = v1_map[(d[0], d[1])]
            v2 = v2_map[(d[0], d[1], d[2])]
            sample.reward = 0.25 * (v0 + v1 + v2 + r)


def commit_coupled_doc2k_updates(args: Any, output: dict[str, Any]) -> None:
    reward_url = getattr(args, "generative_retriever_reward_url", None)
    if not isinstance(reward_url, str) or not reward_url:
        return
    commit_url = reward_url.removesuffix("/score") + "/commit_step_updates"
    try:
        with httpx.Client(timeout=60.0, trust_env=False) as client:
            response = client.post(commit_url, json={})
            response.raise_for_status()
            payload = response.json()
        for key in ("updated_docids", "added_edges", "removed_edges", "overlay_docids"):
            value = payload.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                output[f"rollout/generative_retriever/doc2k/{key}"] = float(value)
    except Exception as error:
        logger.warning("Failed to commit step doc2k updates at %s: %s", commit_url, error)


def _add_distribution_metrics(
    output: dict[str, Any],
    prefix: str,
    values: list[float],
) -> None:
    if not values:
        return
    data = np.asarray(values, dtype=np.float64)
    p10, median, p90 = np.quantile(data, [0.10, 0.50, 0.90])
    output.update({
        f"{prefix}/mean": float(np.mean(data)),
        f"{prefix}/median": float(median),
        f"{prefix}/p10": float(p10),
        f"{prefix}/p90": float(p90),
        f"{prefix}/min": float(np.min(data)),
        f"{prefix}/max": float(np.max(data)),
    })


def _normalize_group_rewards(args: Any, rewards: np.ndarray, reward_std: float) -> np.ndarray:
    if not args.rewards_normalization:
        return rewards
    centered = rewards - np.mean(rewards)
    if args.grpo_std_normalization and reward_std > 0:
        return centered / (reward_std + 1e-6)
    return centered


def _add_group_reward_metrics(
    args: Any,
    samples: list[Any],
    output: dict[str, Any],
) -> None:
    groups: dict[int, list[float]] = defaultdict(list)
    for sample in samples:
        if sample.group_index is not None:
            groups[int(sample.group_index)].append(float(sample.get_reward_value(args)))
    if not groups:
        return

    stds = []
    ranges = []
    unique_counts = []
    top_gaps = []
    group_advantage_abs_means = []
    advantage_abs_values = []
    zero_std_groups = 0
    for rewards in groups.values():
        data = np.asarray(rewards, dtype=np.float64)
        reward_std = float(np.std(data, ddof=1)) if data.size > 1 else 0.0
        stds.append(reward_std)
        ranges.append(float(np.max(data) - np.min(data)))
        unique_counts.append(float(np.unique(data).size))
        ordered = np.sort(data)
        top_gaps.append(float(ordered[-1] - ordered[-2]) if ordered.size > 1 else 0.0)
        zero_std_groups += int(np.all(data == data[0]))
        if args.advantage_estimator == "grpo" and args.kl_coef == 0:
            normalized = _normalize_group_rewards(args, data, reward_std)
            absolute = np.abs(normalized)
            advantage_abs_values.extend(absolute.tolist())
            group_advantage_abs_means.append(float(np.mean(absolute)))

    prefix = "rollout/generative_retriever/group_reward"
    _add_distribution_metrics(output, f"{prefix}/std", stds)
    _add_distribution_metrics(output, f"{prefix}/range", ranges)
    _add_distribution_metrics(output, f"{prefix}/unique_count", unique_counts)
    _add_distribution_metrics(output, f"{prefix}/top1_top2_gap", top_gaps)
    output[f"{prefix}/group_count"] = len(groups)
    output[f"{prefix}/zero_std_count"] = zero_std_groups
    output[f"{prefix}/zero_std_fraction"] = zero_std_groups / len(groups)
    if advantage_abs_values:
        _add_distribution_metrics(
            output,
            f"{prefix}/advantage_abs_group_mean",
            group_advantage_abs_means,
        )
        output[f"{prefix}/advantage_abs/mean"] = float(np.mean(advantage_abs_values))
        output[f"{prefix}/advantage_abs/max"] = float(np.max(advantage_abs_values))


def log_rollout(
    rollout_id: int,
    args: Any,
    samples: list[Any],
    rollout_extra_metrics: dict[str, Any] | None,
    rollout_time: float,
) -> bool:
    del rollout_id, rollout_time
    if rollout_extra_metrics is None:
        return False
    _add_group_reward_metrics(args, samples, rollout_extra_metrics)
    apply_hierarchical_tree_credit(args, samples)
    commit_coupled_doc2k_updates(args, rollout_extra_metrics)
    values: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        metadata = sample.metadata.get("generative_retriever", {})
        for source in ("prepare_metrics", "reward_metrics"):
            for name, value in metadata.get(source, {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                    values[name].append(float(value))
    for name, metrics in values.items():
        if name.endswith("/request_count") or name.startswith(
            ("cache_delta/", "routing_delta/", "prepare/cache_delta/", "prepare/routing_delta/")
        ):
            value = math.fsum(metrics)
        elif name.startswith(("cache_gauge/", "prepare/cache_gauge/")):
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
