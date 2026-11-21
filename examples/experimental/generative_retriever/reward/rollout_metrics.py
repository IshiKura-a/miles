"""Causal RQ-tree credit assignment, group advantage normalization, and rollout telemetry."""

from __future__ import annotations

from collections import defaultdict
import logging
import math
from typing import Any

import numpy as np
import torch

from examples.experimental.generative_retriever.reward.relevance_reward import (
    RQ_CODE_NUM,
    TREE_CREDIT_TOP_K,
)
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.tracking_utils import tracking

logger = logging.getLogger(__name__)


def topk_mean_from_sorted_rewards(
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


def compute_tree_level_values(
    u32_mat: np.ndarray,
    raw_mat_f64: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_groups = raw_mat_f64.shape[0]
    rows = np.arange(num_groups)[:, None]
    ord1 = np.argsort(-raw_mat_f64, axis=1, kind="mergesort")
    u_sorted = u32_mat[rows, ord1]
    r_sorted = raw_mat_f64[rows, ord1]
    k0, k1, k2 = TREE_CREDIT_TOP_K
    v0 = topk_mean_from_sorted_rewards(u_sorted >> np.uint32(24), r_sorted, ord1, k0)
    v1 = topk_mean_from_sorted_rewards(u_sorted >> np.uint32(16), r_sorted, ord1, k1)
    v2 = topk_mean_from_sorted_rewards(u_sorted >> np.uint32(8), r_sorted, ord1, k2)
    return v0, v1, v2


def compute_causal_tree_credit_3d(u32_mat: np.ndarray, raw_mat: np.ndarray) -> np.ndarray:
    """Compute per-depth causal suffix-mean tree credit ``[B, G, 4]`` where leaf ``t=3`` uses ``raw`` only."""
    num_groups, group_size = raw_mat.shape
    r64 = raw_mat.astype(np.float64, copy=False)
    if num_groups == 0 or group_size <= 1:
        return np.repeat(r64[:, :, None], RQ_CODE_NUM, axis=-1).astype(np.float32)
    v0, v1, v2 = compute_tree_level_values(u32_mat, r64)
    return np.stack(
        (
            (v0 + v1 + v2 + r64) / 4.0,
            (v1 + v2 + r64) / 3.0,
            (v2 + r64) / 2.0,
            r64,
        ),
        axis=-1,
    ).astype(np.float32)


def normalize_group_advantages_tensor(
    returns_t: torch.Tensor,
    *,
    rewards_norm: bool,
    grpo_std_norm: bool,
) -> torch.Tensor:
    """Normalize ``[B, G]`` or ``[B, G, C]`` returns across the group dimension ``dim=1``."""
    if not rewards_norm:
        return returns_t
    centered = returns_t - returns_t.mean(dim=1, keepdim=True)
    if grpo_std_norm and returns_t.shape[1] > 1:
        std = returns_t.std(dim=1, keepdim=True)
        centered = torch.where(std > 0, centered / (std + 1e-6), centered)
    return centered


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
    g_size = int(args.n_samples_per_prompt)
    first_meta = (
        samples[0].metadata.get("generative_retriever", {})
        if isinstance(samples[0].metadata, dict)
        else {}
    )
    if not (_is_regular_group_layout(samples, g_size) and "group_u32" in first_meta):
        raise ValueError("GR tree credit requires regular query groups carrying group_u32 metadata")
    num_groups = len(samples) // g_size
    if "group_raw_rewards" in first_meta:
        raw_1d: list[float] = []
        for g in range(num_groups):
            g_meta = samples[g * g_size].metadata["generative_retriever"]
            raw_1d.extend(g_meta.pop("group_raw_rewards"))
    else:
        raw_1d = [
            float(s.reward if isinstance(s.reward, (int, float)) else s.get_reward_value(args))
            for s in samples
        ]
    first_meta["batch_raw_rewards_1d"] = raw_1d
    u32_mat = np.stack(
        [samples[g * g_size].metadata["generative_retriever"]["group_u32"] for g in range(num_groups)],
        axis=0,
    )
    r_mat = np.asarray(raw_1d, dtype=np.float64).reshape(num_groups, g_size)
    causal_3d = compute_causal_tree_credit_3d(u32_mat, r_mat)
    first_meta["batch_causal_returns_2d"] = causal_3d.reshape(-1, RQ_CODE_NUM)
    for sample, val in zip(samples, causal_3d[:, :, 0].reshape(-1).tolist(), strict=True):
        sample.reward = val


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


def add_group_reward_metrics(
    args: Any,
    samples: list[Any],
    output: dict[str, Any],
) -> None:
    g_size = int(args.n_samples_per_prompt)
    if not _is_regular_group_layout(samples, g_size):
        raise ValueError("GR group reward metrics require regular query groups")
    flat_r = np.fromiter(
        (float(s.reward if isinstance(s.reward, (int, float)) else s.get_reward_value(args)) for s in samples),
        dtype=np.float64,
        count=len(samples),
    )
    mat = flat_r.reshape(-1, g_size)
    group_count = mat.shape[0]
    stds_arr = np.std(mat, axis=1, ddof=1)
    ordered_mat = np.sort(mat, axis=1)
    ranges_arr = ordered_mat[:, -1] - ordered_mat[:, 0]
    top_gaps_arr = ordered_mat[:, -1] - ordered_mat[:, -2]
    unique_counts_arr = 1.0 + np.sum(ordered_mat[:, 1:] != ordered_mat[:, :-1], axis=1).astype(np.float64)
    zero_std_groups = int(np.sum(ranges_arr == 0.0))

    prefix = "rollout/generative_retriever/group_reward"
    add_distribution_metrics(output, f"{prefix}/std", stds_arr)
    add_distribution_metrics(output, f"{prefix}/range", ranges_arr)
    add_distribution_metrics(output, f"{prefix}/unique_count", unique_counts_arr)
    add_distribution_metrics(output, f"{prefix}/top1_top2_gap", top_gaps_arr)
    output[f"{prefix}/group_count"] = group_count
    output[f"{prefix}/zero_std_count"] = zero_std_groups
    output[f"{prefix}/zero_std_fraction"] = zero_std_groups / group_count
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
        add_distribution_metrics(output, f"{prefix}/advantage_abs_group_mean", np.mean(abs_mat, axis=1))
        output[f"{prefix}/advantage_abs/mean"] = float(np.sum(abs_mat)) / float(abs_mat.size)
        output[f"{prefix}/advantage_abs/max"] = float(np.max(abs_mat))


def convert_gr_samples_to_train_data(args: Any, samples: list[Any]) -> dict[str, Any]:
    """Fast vectorized conversion of GR rollout samples to train_data."""
    n = len(samples)
    g_size = int(args.n_samples_per_prompt)
    first_meta = (
        samples[0].metadata.get("generative_retriever", {})
        if (samples and isinstance(samples[0].metadata, dict))
        else {}
    )
    raw_rewards = first_meta.pop("batch_raw_rewards_1d", None)
    causal_2d = first_meta.pop("batch_causal_returns_2d", None)
    if n > 0 and (n % g_size != 0 or raw_rewards is None or causal_2d is None):
        raise ValueError("GR sample conversion requires precomputed causal returns and raw rewards on regular groups")
    grpo_std = args.advantage_estimator in ["grpo", "gspo"] and bool(args.grpo_std_normalization)
    if n > 0:
        r_3d = torch.from_numpy(causal_2d).view(-1, g_size, RQ_CODE_NUM)
        norm_3d = normalize_group_advantages_tensor(
            r_3d,
            rewards_norm=bool(args.rewards_normalization),
            grpo_std_norm=grpo_std,
        )
        norm_rewards: list[Any] = norm_3d.reshape(-1, RQ_CODE_NUM).tolist()
    else:
        raw_rewards = []
        norm_rewards = []

    tokens = [s.tokens for s in samples]
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


def _emit_rollout_tracking(rollout_id: int, args: Any, log_dict: dict[str, Any]) -> bool:
    logger.info(f"perf {rollout_id}: {log_dict}")
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    tracking.log(args, log_dict, step_key="rollout/step")
    return True


def log_rollout(
    rollout_id: int,
    args: Any,
    samples: list[Any],
    rollout_extra_metrics: dict[str, Any] | None,
    rollout_time: float,
) -> bool:
    if rollout_extra_metrics is None:
        return False
    if not samples:
        return _emit_rollout_tracking(rollout_id, args, dict(rollout_extra_metrics))

    g_size = int(args.n_samples_per_prompt)
    first_meta = (
        samples[0].metadata.get("generative_retriever", {})
        if isinstance(samples[0].metadata, dict)
        else {}
    )
    if not (_is_regular_group_layout(samples, g_size) and "group_u32" in first_meta):
        raise ValueError("GR rollout logging requires regular query groups carrying group_u32 metadata")

    apply_hierarchical_tree_credit(args, samples)
    add_group_reward_metrics(args, samples, rollout_extra_metrics)

    values: dict[str, list[float]] = defaultdict(list)
    for sample in samples[::g_size]:
        metadata = sample.metadata.get("generative_retriever", {}) if isinstance(sample.metadata, dict) else {}
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
    if raw_1d is None or rollout_time <= 0:
        return False

    n = len(samples)
    raw_arr = np.asarray(raw_1d, dtype=np.float64)
    raw_groups = raw_arr.reshape(num_groups, g_size)
    zero_std_mask = np.all(raw_groups == raw_groups[:, :1], axis=1)
    zero_count = int(np.sum(zero_std_mask & (np.round(raw_groups[:, 0], 1) == 0.0)))
    one_count = int(np.sum(zero_std_mask & (np.round(raw_groups[:, 0], 1) == 1.0)))
    total_resp_tokens = float(n * RQ_CODE_NUM)
    rollout_gpus = max(1, int(args.rollout_num_gpus or 1))
    return _emit_rollout_tracking(
        rollout_id,
        args,
        {
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
        },
    )
