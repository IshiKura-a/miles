"""Shared wire protocol, u32 DocID bit-packing, and linear unique_weight unpacking."""

from __future__ import annotations

from collections.abc import Sequence
import os
import sys

import numpy as np
import torch

from examples.experimental.generative_retriever.reward.relevance_reward import (
    DOCID_TOKEN_OFFSET,
    RQ_CODE_NUM,
)

# Per-beam wire columns: [tok0, tok1, tok2, tok3, lp0, lp1, lp2, lp3, base_docid_val, unit_ub_docid_val]
IN_PROCESS_BEAM_COLS = 10

IN_PROCESS_METRIC_KEYS: tuple[str, ...] = (
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
    "pm_unique_keyword_rate",
    "sm_unique_keyword_rate",
    "unique_keyword_rate",
    "relevance_eligible_rate",
    "pm_unique_eligible_rate",
    "sm_unique_eligible_rate",
    "unique_eligible_rate",
    "unique_quality",
    "train_total",
    "score/latency_seconds",
    "score/request_count",
    "cache_gauge/baseline/size",
    "cache_gauge/reward/size",
    "cache_gauge/query_features/size",
    "cache_gauge/keyword_features/size",
    "cache_delta/baseline/hits",
    "cache_delta/baseline/misses",
    "cache_delta/reward/hits",
    "cache_delta/reward/misses",
    "cache_delta/query_features/hits",
    "cache_delta/query_features/misses",
    "cache_delta/keyword_features/hits",
    "cache_delta/keyword_features/misses",
)


def expected_in_process_embedding_width(group_size: int) -> int:
    return group_size * IN_PROCESS_BEAM_COLS + len(IN_PROCESS_METRIC_KEYS)


def resolve_physical_gpu_id(
    device: torch.device | None = None,
    *,
    fallback_rank: int | None = None,
) -> int:
    """Resolve the node-local physical GPU index for per-GPU /tmp staging files."""
    argv = sys.argv
    for idx, arg in enumerate(argv[:-1]):
        if arg == "--base-gpu-id":
            try:
                return int(argv[idx + 1])
            except ValueError:
                pass
        if arg.startswith("--base-gpu-id="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                pass
    for env_key in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        val = os.environ.get(env_key, "").strip()
        if val and "," not in val:
            try:
                return int(val)
            except ValueError:
                pass
    local_rank = os.environ.get("LOCAL_RANK", "").strip()
    if local_rank.isdigit():
        return int(local_rank)
    if device is not None and device.type == "cuda" and device.index is not None:
        return int(device.index)
    if torch.cuda.is_available():
        return int(torch.cuda.current_device())
    if fallback_rank is not None:
        return int(fallback_rank)
    return 0


def pack_docids_u32_from_tokens(tok_ids: np.ndarray) -> np.ndarray:
    """Pack ``[..., 4]`` decoder token IDs (with ``DOCID_TOKEN_OFFSET``) into ``[...]`` ``uint32``."""
    pos = tok_ids.astype(np.int64, copy=False) - DOCID_TOKEN_OFFSET
    return (
        (pos[..., 0].astype(np.uint32) << np.uint32(24))
        | ((pos[..., 1] - 256).astype(np.uint32) << np.uint32(16))
        | ((pos[..., 2] - 512).astype(np.uint32) << np.uint32(8))
        | (pos[..., 3] - 768).astype(np.uint32)
    )


def pack_in_process_embedding(
    *,
    token_ids: torch.Tensor,
    token_logprobs: torch.Tensor,
    base_docid_mat: np.ndarray,
    unit_ub_docid_mat: np.ndarray,
    metrics_mat: np.ndarray,
) -> torch.Tensor:
    """Pack rollout tokens, logprobs, linear reward components, and summary metrics into ``[B, G*10+40]``."""
    device = token_ids.device
    base_t = torch.from_numpy(base_docid_mat).to(device=device, dtype=torch.float32).unsqueeze(-1)
    unit_ub_t = torch.from_numpy(unit_ub_docid_mat).to(device=device, dtype=torch.float32).unsqueeze(-1)
    core = torch.cat((token_ids.float(), token_logprobs.float(), base_t, unit_ub_t), dim=-1).flatten(1)
    metrics_t = torch.from_numpy(metrics_mat).to(device=device, dtype=torch.float32)
    return torch.cat((core, metrics_t), dim=-1)


def unpack_in_process_embedding(
    out_mat: np.ndarray,
    group_size: int,
    unique_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unpack ``[B, G*10+40]`` (or ``[G*10+40]``) and apply exact linear ``unique_weight`` scaling.

    Returns ``(tok_i32, lp_f32, u32_mat, raw_rewards_f64, scaled_metrics_f64)``.
    """
    squeeze_batch = out_mat.ndim == 1
    mat_2d = out_mat[None, :] if squeeze_batch else out_mat
    num_q = mat_2d.shape[0]
    core_cols = group_size * IN_PROCESS_BEAM_COLS
    core = mat_2d[:, :core_cols].reshape(num_q, group_size, IN_PROCESS_BEAM_COLS)

    tok_i32 = core[:, :, :4].astype(np.int32)
    lp_f32 = np.ascontiguousarray(core[:, :, 4:8], dtype=np.float32)
    base_f64 = core[:, :, 8].astype(np.float64)
    unit_ub_f64 = core[:, :, 9].astype(np.float64)
    raw_f64 = base_f64 + float(unique_weight) * unit_ub_f64
    u32_mat = pack_docids_u32_from_tokens(tok_i32)

    metrics_f64 = mat_2d[:, core_cols:].astype(np.float64, copy=True)
    w = float(unique_weight)
    metrics_f64[:, 9] *= w
    metrics_f64[:, 10] *= w
    metrics_f64[:, 11] *= w
    metrics_f64[:, 14] = metrics_f64[:, 12] + metrics_f64[:, 11]
    metrics_f64[:, 15] = metrics_f64[:, 13] + metrics_f64[:, 11]
    actual_mean_rew = metrics_f64[:, 16] + w * metrics_f64[:, 25]
    metrics_f64[:, 16] = actual_mean_rew
    metrics_f64[:, 25] = actual_mean_rew

    if squeeze_batch:
        return tok_i32[0], lp_f32[0], u32_mat[0], raw_f64[0], metrics_f64[0]
    return tok_i32, lp_f32, u32_mat, raw_f64, metrics_f64


def pack_group_tokens_flat(
    prompt_ids_list: Sequence[Sequence[int]],
    tok_i32: np.ndarray,
    *,
    pad_tok_id: int,
    compact_ipc: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack ``[B, G, 4]`` beam tokens with prompts into flat ``int32`` buffers."""
    num_q = len(prompt_ids_list)
    if num_q == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    group_size = int(tok_i32.shape[1])
    n_samples = num_q * group_size
    tok_chunks: list[np.ndarray] = []
    if compact_ipc and group_size > 1:
        tail_5 = np.empty((num_q, group_size - 1, RQ_CODE_NUM + 1), dtype=np.int32)
        tail_5[:, :, 0] = pad_tok_id
        tail_5[:, :, 1:] = tok_i32[:, 1:, :]
        tail_flat = tail_5.reshape(num_q, -1)
        len_2d = np.full((num_q, group_size), RQ_CODE_NUM + 1, dtype=np.int32)
        for q_i, p_ids in enumerate(prompt_ids_list):
            p_arr = np.asarray(p_ids, dtype=np.int32)
            len_2d[q_i, 0] = int(p_arr.shape[0]) + RQ_CODE_NUM
            tok_chunks.extend((p_arr, tok_i32[q_i, 0], tail_flat[q_i]))
        return np.concatenate(tok_chunks), len_2d.reshape(n_samples)

    len_2d = np.empty((num_q, group_size), dtype=np.int32)
    for q_i, p_ids in enumerate(prompt_ids_list):
        p_arr = np.asarray(p_ids, dtype=np.int32)
        full_len = int(p_arr.shape[0]) + RQ_CODE_NUM
        len_2d[q_i, :] = full_len
        grp_arr = np.empty((group_size, full_len), dtype=np.int32)
        grp_arr[:, : len(p_ids)] = p_arr[None, :]
        grp_arr[:, len(p_ids) :] = tok_i32[q_i]
        tok_chunks.append(grp_arr.reshape(-1))
    return np.concatenate(tok_chunks), len_2d.reshape(n_samples)
