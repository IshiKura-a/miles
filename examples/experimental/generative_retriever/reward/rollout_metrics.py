"""Add GR reward-component and cache metrics to the normal rollout report."""

from __future__ import annotations

from collections import defaultdict
import logging
import math
from typing import Any

import httpx
import numpy as np
import torch

from examples.experimental.generative_retriever.reward.relevance_reward import (
    RQ_CODE_NUM,
    TREE_CREDIT_LEVEL_WEIGHT,
    TREE_CREDIT_TOP_K,
)
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.tracking_utils import tracking

logger = logging.getLogger(__name__)


def mean_top_k(values: list[float], k: int) -> float:
    if not values:
        return 0.0
    if len(values) <= k:
        return math.fsum(values) / len(values)
    top_values = sorted(values, reverse=True)[:k]
    return math.fsum(top_values) / len(top_values)


def _topk_mean_from_sorted_rewards(
    s_keys: np.ndarray,
    s_rewards: np.ndarray,
    ord1: np.ndarray,
    k: int,
) -> np.ndarray:
    num_groups, group_size = s_keys.shape
    rows = np.arange(num_groups)[:, None]
    ord2 = np.argsort(s_keys, axis=1, kind="mergesort")
    k_s = s_keys[rows, ord2]
    r_s = s_rewards[rows, ord2]
    full_ord = ord1[rows, ord2]

    is_start = np.ones((num_groups, group_size), dtype=bool)
    is_start[:, 1:] = k_s[:, 1:] != k_s[:, :-1]
    idx = np.arange(group_size, dtype=np.int32)[None, :]
    start_pos = np.maximum.accumulate(np.where(is_start, idx, 0), axis=1)
    valid = (idx - start_pos) < k

    cs_r = np.cumsum(np.where(valid, r_s, 0.0), axis=1)
    cs_c = np.cumsum(valid.astype(np.int32), axis=1)

    is_end = np.ones((num_groups, group_size), dtype=bool)
    is_end[:, :-1] = is_start[:, 1:]
    end_pos = np.minimum.accumulate(
        np.where(is_end[:, ::-1], idx[:, ::-1], group_size - 1),
        axis=1,
    )[:, ::-1]

    prev_pos = np.maximum(0, start_pos - 1)
    has_prev = start_pos > 0
    group_sum = np.take_along_axis(cs_r, end_pos, axis=1) - np.where(
        has_prev, np.take_along_axis(cs_r, prev_pos, axis=1), 0.0
    )
    group_cnt = np.maximum(
        1,
        np.take_along_axis(cs_c, end_pos, axis=1)
        - np.where(has_prev, np.take_along_axis(cs_c, prev_pos, axis=1), 0),
    )
    means_sorted = group_sum / group_cnt.astype(np.float64)

    out = np.empty_like(means_sorted)
    out[rows, full_ord] = means_sorted
    return out


def _is_regular_group_layout(samples: list[Any], g_size: int) -> bool:
    return (
        bool(samples)
        and g_size > 1
        and len(samples) % g_size == 0
        and samples[0].group_index is not None
        and samples[-1].group_index == samples[0].group_index + (len(samples) // g_size) - 1
    )


def apply_hierarchical_tree_credit(args: Any, samples: list[Any]) -> None:
    if not samples:
        return
    g_size = getattr(args, "n_samples_per_prompt", 0)
    first_meta = (
        samples[0].metadata.get("generative_retriever", {})
        if isinstance(samples[0].metadata, dict)
        else {}
    )
    if _is_regular_group_layout(samples, g_size) and "group_raw_rewards" in first_meta:
        num_groups = len(samples) // g_size
        raw_1d: list[float] = []
        for g in range(num_groups):
            g_meta = samples[g * g_size].metadata["generative_retriever"]
            raw_1d.extend(g_meta.pop("group_raw_rewards"))
        first_meta["batch_raw_rewards_1d"] = raw_1d
        return

    if _is_regular_group_layout(samples, g_size) and "group_u32" in first_meta:
        num_groups = len(samples) // g_size
        u32_mat = np.stack(
            [samples[g * g_size].metadata["generative_retriever"]["group_u32"] for g in range(num_groups)],
            axis=0,
        )
        raw_flat = np.fromiter(
            (float(s.reward if isinstance(s.reward, (int, float)) else s.get_reward_value(args)) for s in samples),
            dtype=np.float64,
            count=len(samples),
        )
        first_meta["batch_raw_rewards_1d"] = raw_flat.tolist()
        r_mat = raw_flat.reshape(num_groups, g_size)
        rows = np.arange(num_groups)[:, None]
        ord1 = np.argsort(-r_mat, axis=1, kind="mergesort")
        u_sorted = u32_mat[rows, ord1]
        r_sorted = r_mat[rows, ord1]
        k0, k1, k2 = TREE_CREDIT_TOP_K
        v0 = _topk_mean_from_sorted_rewards(u_sorted >> np.uint32(24), r_sorted, ord1, k0)
        v1 = _topk_mean_from_sorted_rewards(u_sorted >> np.uint32(16), r_sorted, ord1, k1)
        v2 = _topk_mean_from_sorted_rewards(u_sorted >> np.uint32(8), r_sorted, ord1, k2)
        credited_flat = (TREE_CREDIT_LEVEL_WEIGHT * (v0 + v1 + v2 + r_mat)).reshape(-1).tolist()
        for sample, val in zip(samples, credited_flat, strict=True):
            sample.reward = val
        return

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
        k0, k1, k2 = TREE_CREDIT_TOP_K
        v0_map = {k: mean_top_k(v, k0) for k, v in by_p0.items()}
        v1_map = {k: mean_top_k(v, k1) for k, v in by_p1.items()}
        v2_map = {k: mean_top_k(v, k2) for k, v in by_p2.items()}
        for sample, d, r in items:
            v0 = v0_map[d[0]]
            v1 = v1_map[(d[0], d[1])]
            v2 = v2_map[(d[0], d[1], d[2])]
            sample.reward = TREE_CREDIT_LEVEL_WEIGHT * (v0 + v1 + v2 + r)


def stage_coupled_doc2k_updates(args: Any, rollout_id: int) -> None:
    reward_url = getattr(args, "generative_retriever_reward_url", None)
    if not isinstance(reward_url, str) or not reward_url:
        return
    stage_url = reward_url.removesuffix("/score") + "/stage_step_updates"
    with httpx.Client(timeout=300.0, trust_env=False) as client:
        response = client.post(stage_url, json={"rollout_id": rollout_id})
        response.raise_for_status()


def add_distribution_metrics(
    output: dict[str, Any],
    prefix: str,
    values: list[float] | np.ndarray,
) -> None:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return
    p10, median, p90 = np.quantile(data, [0.10, 0.50, 0.90])
    output.update({
        f"{prefix}/mean": float(np.mean(data)),
        f"{prefix}/median": float(median),
        f"{prefix}/p10": float(p10),
        f"{prefix}/p90": float(p90),
        f"{prefix}/min": float(np.min(data)),
        f"{prefix}/max": float(np.max(data)),
    })


def normalize_group_rewards(args: Any, rewards: np.ndarray, reward_std: float) -> np.ndarray:
    if not args.rewards_normalization:
        return rewards
    centered = rewards - np.mean(rewards)
    if args.grpo_std_normalization and reward_std > 0:
        return centered / (reward_std + 1e-6)
    return centered


def add_group_reward_metrics(
    args: Any,
    samples: list[Any],
    output: dict[str, Any],
) -> None:
    g_size = getattr(args, "n_samples_per_prompt", 0)
    if _is_regular_group_layout(samples, g_size):
        flat_r = np.fromiter(
            (float(s.reward if isinstance(s.reward, (int, float)) else s.get_reward_value(args)) for s in samples),
            dtype=np.float64,
            count=len(samples),
        )
        mat = flat_r.reshape(-1, g_size)
        num_groups = mat.shape[0]
        stds_arr = np.std(mat, axis=1, ddof=1)
        ordered_mat = np.sort(mat, axis=1)
        ranges_arr = ordered_mat[:, -1] - ordered_mat[:, 0]
        top_gaps_arr = ordered_mat[:, -1] - ordered_mat[:, -2]
        unique_counts_arr = 1.0 + np.sum(ordered_mat[:, 1:] != ordered_mat[:, :-1], axis=1).astype(np.float64)
        zero_std_groups = int(np.sum(ranges_arr == 0.0))
        stds: list[float] | np.ndarray = stds_arr
        ranges: list[float] | np.ndarray = ranges_arr
        unique_counts: list[float] | np.ndarray = unique_counts_arr
        top_gaps: list[float] | np.ndarray = top_gaps_arr
        group_advantage_abs_means: list[float] | np.ndarray = []
        advantage_abs_sum = 0.0
        advantage_abs_count = 0
        advantage_abs_max = 0.0
        if args.advantage_estimator == "grpo" and args.kl_coef == 0:
            centered = mat - np.mean(mat, axis=1, keepdims=True)
            if args.rewards_normalization and args.grpo_std_normalization:
                denom = np.where(stds_arr[:, None] > 0, stds_arr[:, None] + 1e-6, 1.0)
                norm_mat = centered / denom
            elif args.rewards_normalization:
                norm_mat = centered
            else:
                norm_mat = mat
            abs_mat = np.abs(norm_mat)
            advantage_abs_sum = float(np.sum(abs_mat))
            advantage_abs_count = int(abs_mat.size)
            advantage_abs_max = float(np.max(abs_mat))
            group_advantage_abs_means = np.mean(abs_mat, axis=1)
        group_count = num_groups
    else:
        groups_dict: dict[int, list[float]] = defaultdict(list)
        for sample in samples:
            if sample.group_index is not None:
                groups_dict[int(sample.group_index)].append(float(sample.get_reward_value(args)))
        if not groups_dict:
            return
        stds = []
        ranges = []
        unique_counts = []
        top_gaps = []
        group_advantage_abs_means = []
        advantage_abs_sum = 0.0
        advantage_abs_count = 0
        advantage_abs_max = 0.0
        zero_std_groups = 0
        for rewards in groups_dict.values():
            data = np.asarray(rewards, dtype=np.float64)
            reward_std = float(np.std(data, ddof=1)) if data.size > 1 else 0.0
            stds.append(reward_std)
            ranges.append(float(np.max(data) - np.min(data)))
            unique_counts.append(float(np.unique(data).size))
            ordered = np.sort(data)
            top_gaps.append(float(ordered[-1] - ordered[-2]) if ordered.size > 1 else 0.0)
            zero_std_groups += int(np.all(data == data[0]))
            if args.advantage_estimator == "grpo" and args.kl_coef == 0:
                normalized = normalize_group_rewards(args, data, reward_std)
                absolute = np.abs(normalized)
                advantage_abs_sum += float(np.sum(absolute))
                advantage_abs_count += int(absolute.size)
                advantage_abs_max = max(advantage_abs_max, float(np.max(absolute)))
                group_advantage_abs_means.append(float(np.mean(absolute)))
        group_count = len(groups_dict)

    prefix = "rollout/generative_retriever/group_reward"
    add_distribution_metrics(output, f"{prefix}/std", stds)
    add_distribution_metrics(output, f"{prefix}/range", ranges)
    add_distribution_metrics(output, f"{prefix}/unique_count", unique_counts)
    add_distribution_metrics(output, f"{prefix}/top1_top2_gap", top_gaps)
    output[f"{prefix}/group_count"] = group_count
    output[f"{prefix}/zero_std_count"] = zero_std_groups
    output[f"{prefix}/zero_std_fraction"] = zero_std_groups / group_count
    if advantage_abs_count > 0:
        add_distribution_metrics(
            output,
            f"{prefix}/advantage_abs_group_mean",
            group_advantage_abs_means,
        )
        output[f"{prefix}/advantage_abs/mean"] = advantage_abs_sum / advantage_abs_count
        output[f"{prefix}/advantage_abs/max"] = advantage_abs_max


def convert_gr_samples_to_train_data(args: Any, samples: list[Any]) -> dict[str, Any]:
    """Fast vectorized conversion of GR rollout samples to train_data."""
    n = len(samples)
    g_size = args.n_samples_per_prompt
    first_meta = (
        samples[0].metadata.get("generative_retriever", {})
        if (samples and isinstance(samples[0].metadata, dict))
        else {}
    )
    raw_rewards = first_meta.pop("batch_raw_rewards_1d", None)
    if raw_rewards is None:
        raw_rewards = [
            float(s.metadata["raw_reward"]) if (s.metadata and "raw_reward" in s.metadata) else float(s.get_reward_value(args))
            for s in samples
        ]
    rewards_t = torch.tensor(
        [float(s.reward if isinstance(s.reward, (int, float)) else s.get_reward_value(args)) for s in samples],
        dtype=torch.float32,
    )
    if args.rewards_normalization and n % g_size == 0:
        r_mat = rewards_t.view(-1, g_size)
        centered = r_mat - r_mat.mean(dim=1, keepdim=True)
        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
            std = r_mat.std(dim=1, keepdim=True)
            centered = torch.where(std > 0, centered / (std + 1e-6), centered)
        norm_rewards = centered.reshape(-1).tolist()
    else:
        norm_rewards = rewards_t.tolist()

    tokens = [s.tokens for s in samples]
    if n > 0 and n % g_size == 0:
        total_lengths = [len(samples[g].tokens) for g in range(0, n, g_size) for _ in range(g_size)]
    else:
        total_lengths = [len(t) for t in tokens]
    base_idx = int(samples[0].index) if (n > 0 and samples[0].index is not None) else 0
    sample_indices = list(range(base_idx, base_idx + n))
    shared_mask = [1] * RQ_CODE_NUM
    return {
        "tokens": tokens,
        "total_lengths": total_lengths,
        "response_lengths": [RQ_CODE_NUM] * n,
        "rewards": norm_rewards,
        "raw_reward": raw_rewards,
        "truncated": [0] * n,
        "sample_indices": sample_indices,
        "rollout_ids": sample_indices,
        "loss_masks": [shared_mask] * n,
        "rollout_mask_sums": [RQ_CODE_NUM] * n,
        "rollout_log_probs": [s.rollout_log_probs for s in samples],
    }


def log_rollout(
    rollout_id: int,
    args: Any,
    samples: list[Any],
    rollout_extra_metrics: dict[str, Any] | None,
    rollout_time: float,
) -> bool:
    stage_coupled_doc2k_updates(args, rollout_id)
    if rollout_extra_metrics is None:
        return False
    g_size = getattr(args, "n_samples_per_prompt", 0)
    is_regular = _is_regular_group_layout(samples, g_size)
    first_meta = (
        samples[0].metadata.get("generative_retriever", {})
        if (samples and isinstance(samples[0].metadata, dict))
        else {}
    )
    in_process_groups = is_regular and "group_u32" in first_meta
    if in_process_groups:
        apply_hierarchical_tree_credit(args, samples)
        add_group_reward_metrics(args, samples, rollout_extra_metrics)
    else:
        add_group_reward_metrics(args, samples, rollout_extra_metrics)
        apply_hierarchical_tree_credit(args, samples)

    values: dict[str, list[float]] = defaultdict(list)
    metric_Linear_samples = samples[::g_size] if in_process_groups else samples
    for sample in metric_Linear_samples:
        metadata = sample.metadata.get("generative_retriever", {}) if isinstance(sample.metadata, dict) else {}
        if "prepare_metrics" not in metadata and "reward_metrics" not in metadata:
            continue
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

    if in_process_groups:
        num_groups = len(samples) // g_size
        u32_mat = np.stack(
            [samples[g * g_size].metadata["generative_retriever"].pop("group_u32") for g in range(num_groups)],
            axis=0,
        )
        s_u32 = np.sort(u32_mat, axis=1)
        unique_per_group = 1.0 + np.sum(s_u32[:, 1:] != s_u32[:, :-1], axis=1).astype(np.float64)
        rollout_extra_metrics["rollout/generative_retriever/distinct_docid_ratio"] = float(
            np.mean(unique_per_group / float(g_size))
        )
        raw_1d = first_meta.get("batch_raw_rewards_1d")
        if raw_1d is not None and rollout_time > 0:
            n = len(samples)
            raw_arr = np.asarray(raw_1d, dtype=np.float64)
            raw_groups = raw_arr.reshape(num_groups, g_size)
            zero_std_mask = np.all(raw_groups == raw_groups[:, :1], axis=1)
            zero_count = int(np.sum(zero_std_mask & (np.round(raw_groups[:, 0], 1) == 0.0)))
            one_count = int(np.sum(zero_std_mask & (np.round(raw_groups[:, 0], 1) == 1.0)))
            total_resp_tokens = float(n * RQ_CODE_NUM)
            rollout_gpus = max(1, int(getattr(args, "rollout_num_gpus", 1) or 1))
            log_dict: dict[str, Any] = {
                **rollout_extra_metrics,
                "rollout/num_training_samples": n,
                "rollout/episode_raw_reward": float(np.mean(raw_arr)),
                "rollout/episode_response_length/mean": float(RQ_CODE_NUM),
                "rollout/episode_response_length/min": float(RQ_CODE_NUM),
                "rollout/episode_response_length/max": float(RQ_CODE_NUM),
                "rollout/episode_response_length/std": 0.0,
                "rollout/episode_total_response_length/mean": float(RQ_CODE_NUM),
                "rollout/response_len/mean": float(RQ_CODE_NUM),
                "rollout/response_len/min": float(RQ_CODE_NUM),
                "rollout/response_len/max": float(RQ_CODE_NUM),
                "rollout/response_len/std": 0.0,
                "rollout/zero_std/all_zero_percentage": zero_count / float(num_groups),
                "rollout/zero_std/all_one_percentage": one_count / float(num_groups),
                "rollout/repetition_frac": 0.0,
                "rollout/truncated_ratio": 0.0,
                "perf/rollout_time": rollout_time,
                "perf/tokens_per_gpu_per_sec": total_resp_tokens / rollout_time / rollout_gpus,
                "perf/longest_sample_tokens_per_sec": float(RQ_CODE_NUM) / rollout_time,
                "perf/effective_tokens_per_gpu_per_sec": total_resp_tokens / rollout_time / rollout_gpus,
                "perf/longest_effective_sample_tokens_per_sec": float(RQ_CODE_NUM) / rollout_time,
            }
            logger.info(f"perf {rollout_id}: {log_dict}")
            step = compute_rollout_step(args, rollout_id)
            log_dict["rollout/step"] = step
            tracking.log(args, log_dict, step_key="rollout/step")
            return True
        return False

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
