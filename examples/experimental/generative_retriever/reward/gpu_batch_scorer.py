"""GPU-vectorized batch reward, novelty, metrics, and doc2k vote reducer."""

from __future__ import annotations

import time

import numpy as np
import torch

from examples.experimental.generative_retriever.index.doc2k_voting import (
    Doc2kVoteDelta,
    compute_head_cutoff,
    partition_head_and_tail_indices,
    record_query_enrichment_votes,
)
from examples.experimental.generative_retriever.reward.in_process_protocol import (
    IN_PROCESS_METRIC_KEYS,
)
from examples.experimental.generative_retriever.reward.precompiled_keywords import (
    BaselineKidSets,
    PackedPairScorer,
)
from examples.experimental.generative_retriever.reward.relevance_reward import (
    DOC2K_MIN_GROUP_SAMPLES,
    DOC2K_PRUNE_FLOOR,
    PM_THRESHOLD,
    SM_THRESHOLD,
)
from examples.experimental.generative_retriever.runtime.gr_config import DocidDenomMode, GRConfig
from examples.experimental.generative_retriever.services.lru_cache import LRUCache
from examples.experimental.generative_retriever.services.mapping_service import ResidentMapping

_TOTAL_METRIC_COLS = len(IN_PROCESS_METRIC_KEYS)


def gpu_not_in_sorted(keys_i64: torch.Tensor, sorted_set_i64: torch.Tensor) -> torch.Tensor:
    n_set = int(sorted_set_i64.numel())
    if n_set == 0 or keys_i64.numel() == 0:
        return torch.ones_like(keys_i64, dtype=torch.bool)
    pos = torch.searchsorted(sorted_set_i64, keys_i64)
    clamped = pos.clamp_max(n_set - 1)
    return (pos >= n_set) | (sorted_set_i64[clamped] != keys_i64)


def docid_denominator_gpu(
    lens_f64: torch.Tensor,
    *,
    denom_mode: DocidDenomMode,
    k_floor: int,
) -> torch.Tensor:
    floor_f = float(k_floor)
    if denom_mode == "linear_floor":
        return lens_f64.clamp_min(floor_f)
    if denom_mode == "sqrt_floor":
        return torch.sqrt(floor_f * lens_f64.clamp_min(floor_f))
    if denom_mode == "unit":
        return torch.ones_like(lens_f64)
    raise ValueError(f"Unknown docid_denom_mode: {denom_mode!r}")


def pack_vk_sorted(sets_per_v: list[frozenset[int]], device: torch.device) -> torch.Tensor:
    parts = [
        (np.int64(v_idx) << np.int64(32)) | np.fromiter(item, dtype=np.int64, count=len(item))
        for v_idx, item in enumerate(sets_per_v)
        if item
    ]
    if not parts:
        return torch.empty(0, dtype=torch.int64, device=device)
    merged = np.concatenate(parts)
    merged.sort()
    return torch.from_numpy(merged).to(device=device, non_blocking=True)


def _populate_missing_baselines(
    *,
    miss_q_list: list[str],
    base_edge_m_np: np.ndarray,
    base_edge_kid_np: np.ndarray,
    pm_sm_gpu: torch.Tensor,
    inv_base_gpu: torch.Tensor,
    baseline_kid_cache: LRUCache[str, BaselineKidSets],
) -> None:
    if not miss_q_list:
        return
    base_pm_ok_np = (pm_sm_gpu[inv_base_gpu, 0] >= PM_THRESHOLD).detach().cpu().numpy()
    base_sm_ok_np = (pm_sm_gpu[inv_base_gpu, 1] >= SM_THRESHOLD).detach().cpu().numpy()
    m_bounds = np.searchsorted(base_edge_m_np, np.arange(len(miss_q_list) + 1))
    for m_idx, q_str in enumerate(miss_q_list):
        st, ed = int(m_bounds[m_idx]), int(m_bounds[m_idx + 1])
        kids_sl = base_edge_kid_np[st:ed]
        full_pm = frozenset(int(x) for x in kids_sl[base_pm_ok_np[st:ed]].tolist())
        full_sm = frozenset(int(x) for x in kids_sl[base_sm_ok_np[st:ed]].tolist())
        baseline_kid_cache.put(q_str, BaselineKidSets(pm=full_pm, sm=full_sm))


def _reduce_edge_metrics_gpu(
    *,
    n_valid: int,
    group_size: int,
    pol_lens_gpu: torch.Tensor,
    pol_edge_s_gpu: torch.Tensor,
    pol_edge_v_gpu: torch.Tensor,
    pol_edge_kid_gpu: torch.Tensor,
    pm_sm_pol: torch.Tensor,
    sorted_bpm_gpu: torch.Tensor,
    sorted_bsm_gpu: torch.Tensor,
    gr_config: GRConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = pol_lens_gpu.device
    n_samples = n_valid * group_size
    e_pol = int(pol_edge_kid_gpu.numel())
    lens_f64 = pol_lens_gpu.to(torch.float64).clamp_min(1.0)
    inv_n_f64 = 1.0 / lens_f64

    pm_f = pm_sm_pol[:, 0].to(torch.float64)
    sm_f = pm_sm_pol[:, 1].to(torch.float64)
    norm_pm = torch.clamp(pm_f * (1.0 / PM_THRESHOLD), max=1.0)
    norm_sm = torch.clamp(sm_f * (1.0 / SM_THRESHOLD), max=1.0)
    p_ok = pm_f >= PM_THRESHOLD
    s_ok = sm_f >= SM_THRESHOLD
    pm_p = p_ok.to(torch.float64)
    sm_p = s_ok.to(torch.float64)
    disc_b = pm_p + sm_p
    cont_b = norm_pm + norm_sm
    rel_q = 0.5 * (norm_pm + norm_sm)

    vk_pol_gpu = (pol_edge_v_gpu << 32) | pol_edge_kid_gpu
    _, inv_vk, counts_vk = torch.unique(vk_pol_gpu, return_inverse=True, return_counts=True)
    inv_freq = 1.0 / counts_vk[inv_vk].to(torch.float64)

    u_pm = gpu_not_in_sorted(vk_pol_gpu, sorted_bpm_gpu)
    u_sm = gpu_not_in_sorted(vk_pol_gpu, sorted_bsm_gpu)
    pm_u_el = u_pm & p_ok
    sm_u_el = u_sm & s_ok
    u_el = pm_u_el | sm_u_el
    rel_el = p_ok | s_ok
    any_u = u_pm | u_sm

    # Store unit (unique_weight = 1.0) bonus components; caller scales linearly by step unique_weight.
    unit_pm_ub = torch.where(pm_u_el, 0.5 * rel_q, 0.0)
    unit_sm_ub = torch.where(sm_u_el, 0.5 * rel_q, 0.0)
    unit_ub = unit_pm_ub + unit_sm_ub

    is_cont = gr_config.reward_mode == "continuous_relevance_unique"
    base_raw = cont_b if is_cont else disc_b
    base_rew = base_raw * inv_freq
    unit_ub_rew = unit_ub * inv_freq

    edge_cols = torch.stack(
        (
            pm_f,
            sm_f,
            pm_p,
            sm_p,
            norm_pm,
            norm_sm,
            norm_pm if is_cont else pm_p,
            norm_sm if is_cont else sm_p,
            rel_q,
            unit_pm_ub,
            unit_sm_ub,
            unit_ub,
            disc_b,
            cont_b,
            disc_b + unit_ub,
            cont_b + unit_ub,
            base_rew,
            u_pm.to(torch.float64),
            u_sm.to(torch.float64),
            (u_pm & u_sm).to(torch.float64),
            rel_el.to(torch.float64),
            pm_u_el.to(torch.float64),
            sm_u_el.to(torch.float64),
            u_el.to(torch.float64),
            torch.where(any_u, rel_q, 0.0),
            any_u.to(torch.float64),
            unit_ub_rew,
        ),
        dim=1,
    )
    sample_sums = torch.zeros((n_samples, 27), dtype=torch.float64, device=device)
    if e_pol > 0:
        sample_sums.index_add_(0, pol_edge_s_gpu, edge_cols)

    denom_f64 = docid_denominator_gpu(
        pol_lens_gpu.to(torch.float64),
        denom_mode=gr_config.docid_denom_mode,
        k_floor=gr_config.docid_reward_floor,
    )
    base_docid_np = (sample_sums[:, 16] / denom_f64).view(n_valid, group_size).detach().cpu().numpy().astype(np.float32)
    unit_ub_docid_np = (sample_sums[:, 26] / denom_f64).view(n_valid, group_size).detach().cpu().numpy().astype(np.float32)
    sample_has_pm_np = (sample_sums[:, 2] > 0.0).view(n_valid, group_size).detach().cpu().numpy()

    per_sample_means = sample_sums[:, :24] * inv_n_f64.unsqueeze(1)
    any_u_cnt = sample_sums[:, 25]
    u_qual_sample = torch.where(any_u_cnt > 0.0, sample_sums[:, 24] / any_u_cnt.clamp_min(1.0), 0.0)
    unit_ub_rew_mean = (sample_sums[:, 26] * inv_n_f64).unsqueeze(1)
    group_acc_np = (
        torch.cat((per_sample_means, u_qual_sample.unsqueeze(1), unit_ub_rew_mean), dim=1)
        .view(n_valid, group_size, 26)
        .mean(dim=1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    ug_mask = p_ok & s_ok & u_pm & u_sm
    return base_docid_np, unit_ub_docid_np, group_acc_np, sample_has_pm_np, p_ok, rel_el, ug_mask, pol_edge_kid_gpu


def accumulate_batch_doc2k_votes(
    *,
    pol_valid_u32: np.ndarray,
    pol_lens_np: np.ndarray,
    pol_edge_s_gpu: torch.Tensor,
    pol_edge_kid_gpu: torch.Tensor,
    p_ok: torch.Tensor,
    rel_el: torch.Tensor,
    ug_mask: torch.Tensor,
    sample_has_pm_np: np.ndarray,
    token_logprobs: torch.Tensor,
    valid_rows: list[int],
    group_size: int,
    gr_config: GRConfig,
) -> Doc2kVoteDelta:
    """Extract fat-bucket survival edges and head-enrichment votes into a compact ``Doc2kVoteDelta``."""
    delta = Doc2kVoteDelta()
    n_valid = len(valid_rows)
    lens_2d = pol_lens_np.reshape(n_valid, group_size)
    r_offs = np.empty(len(pol_lens_np) + 1, dtype=np.int64)
    r_offs[0] = 0
    np.cumsum(pol_lens_np, out=r_offs[1:])

    pass_mask = p_ok if gr_config.pm_prune_only else rel_el
    fat_sample_mask_gpu = torch.from_numpy(pol_lens_np > DOC2K_PRUNE_FLOOR).to(device=p_ok.device)
    if pol_edge_s_gpu.numel() > 0:
        fp_idx = torch.nonzero(pass_mask & fat_sample_mask_gpu[pol_edge_s_gpu], as_tuple=False).squeeze(-1)
        fp_s_np = pol_edge_s_gpu[fp_idx].detach().cpu().numpy()
        fp_kid_np = pol_edge_kid_gpu[fp_idx].detach().cpu().numpy()
        ug_idx = torch.nonzero(ug_mask, as_tuple=False).squeeze(-1)
        ug_s_np = pol_edge_s_gpu[ug_idx].detach().cpu().numpy()
        ug_kid_np = pol_edge_kid_gpu[ug_idx].detach().cpu().numpy()
        all_kids_cpu = pol_edge_kid_gpu.detach().cpu().numpy()
    else:
        fp_s_np = fp_kid_np = ug_s_np = ug_kid_np = all_kids_cpu = np.empty(0, dtype=np.int64)

    fp_bounds = np.searchsorted(fp_s_np, np.arange(n_valid * group_size + 1))
    ug_bounds = np.searchsorted(ug_s_np, np.arange(n_valid * group_size + 1))
    rel_guided = gr_config.doc2k_relevance_guided
    lp_np_all = token_logprobs[valid_rows].detach().cpu().numpy().sum(axis=-1) if rel_guided else None
    head_cutoff = compute_head_cutoff(group_size)
    can_enrich = group_size >= DOC2K_MIN_GROUP_SAMPLES and group_size > head_cutoff

    for v in range(n_valid):
        s_base = v * group_size
        row_u32 = pol_valid_u32[v]
        row_lens = lens_2d[v]

        fat_cols = np.flatnonzero(row_lens > DOC2K_PRUNE_FLOOR)
        if fat_cols.size > 0:
            seen_fat: set[int] = set()
            for col in fat_cols.tolist():
                u32_val = int(row_u32[col])
                if u32_val in seen_fat:
                    continue
                seen_fat.add(u32_val)
                s_idx = s_base + col
                delta.docid_q_count[u32_val] = delta.docid_q_count.get(u32_val, 0) + 1
                fst, fed = int(fp_bounds[s_idx]), int(fp_bounds[s_idx + 1])
                for kid_p in fp_kid_np[fst:fed].tolist():
                    edge = (u32_val, int(kid_p))
                    delta.edge_passes[edge] = delta.edge_passes.get(edge, 0) + 1

        if not can_enrich or int(ug_bounds[s_base + group_size]) == int(ug_bounds[s_base]):
            continue
        head_indices, tail_indices = partition_head_and_tail_indices(
            group_size,
            lp_np_all[v] if lp_np_all is not None else None,
            relevance_guided=rel_guided,
        )
        head_chunks = [all_kids_cpu[r_offs[s_base + hi] : r_offs[s_base + hi + 1]] for hi in head_indices]
        head_kids_set = set(np.concatenate(head_chunks).tolist()) if head_chunks else set()
        tail_ug_slices = [ug_kid_np[int(ug_bounds[s_base + ti]) : int(ug_bounds[s_base + ti + 1])] for ti in tail_indices]
        record_query_enrichment_votes(
            row_u32=[int(x) for x in row_u32.tolist()],
            sample_lens=[int(x) for x in row_lens.tolist()],
            sample_has_pm=[bool(x) for x in sample_has_pm_np[v].tolist()],
            head_indices=head_indices,
            tail_indices=tail_indices,
            head_kids_set=head_kids_set,
            tail_unique_good_kids=tail_ug_slices,
            relevance_guided=rel_guided,
            delta=delta,
            k_floor=gr_config.docid_reward_floor,
        )
    return delta


def _lookup_policy_and_baseline_edges(
    *,
    mapping_engine: ResidentMapping,
    pol_valid_u32: np.ndarray,
    valid_queries: list[str],
    q_to_uidx: dict[str, int],
    baseline_u32_by_q: dict[str, np.ndarray],
    group_size: int,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, list[str], np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, int]:
    v_quidx_gpu = torch.as_tensor([q_to_uidx[q] for q in valid_queries], dtype=torch.int64, device=device)
    pol_lens_np, pol_edge_s_np, pol_edge_kid_np = mapping_engine.lookup_flat_kids_csr(
        pol_valid_u32.reshape(-1),
        use_initial=False,
    )
    e_pol = int(pol_edge_kid_np.shape[0])
    pol_edge_s_gpu = torch.from_numpy(pol_edge_s_np).to(device=device, non_blocking=True)
    pol_edge_kid_gpu = torch.from_numpy(pol_edge_kid_np).to(device=device, non_blocking=True)
    pol_edge_v_gpu = torch.div(pol_edge_s_gpu, group_size, rounding_mode="floor")
    pol_qk_gpu = (v_quidx_gpu[pol_edge_v_gpu] << 32) | pol_edge_kid_gpu

    miss_q_list = list(baseline_u32_by_q.keys())
    if miss_q_list:
        base_u32_stack = np.stack([baseline_u32_by_q[q] for q in miss_q_list], axis=0)
        b_cols = base_u32_stack.shape[1]
        _, base_edge_s_np, base_edge_kid_np = mapping_engine.lookup_flat_kids_csr(
            base_u32_stack.reshape(-1),
            use_initial=True,
        )
        base_edge_m_np = base_edge_s_np // b_cols
        miss_quidx_gpu = torch.as_tensor([q_to_uidx[q] for q in miss_q_list], dtype=torch.int64, device=device)
        base_edge_m_gpu = torch.from_numpy(base_edge_m_np).to(device=device, non_blocking=True)
        base_edge_kid_gpu = torch.from_numpy(base_edge_kid_np).to(device=device, non_blocking=True)
        all_qk_gpu = torch.cat((pol_qk_gpu, (miss_quidx_gpu[base_edge_m_gpu] << 32) | base_edge_kid_gpu), dim=0)
    else:
        base_edge_m_np = base_edge_kid_np = np.empty(0, dtype=np.int64)
        all_qk_gpu = pol_qk_gpu

    unique_qk_gpu, inv_all_gpu = torch.unique(all_qk_gpu, sorted=True, return_inverse=True)
    return (
        pol_lens_np,
        pol_edge_s_gpu,
        pol_edge_v_gpu,
        pol_edge_kid_gpu,
        miss_q_list,
        base_edge_m_np,
        base_edge_kid_np,
        unique_qk_gpu,
        inv_all_gpu,
        e_pol,
    )


def score_batch_gpu_vectorized(
    *,
    scorer: PackedPairScorer,
    mapping_engine: ResidentMapping,
    baseline_kid_cache: LRUCache[str, BaselineKidSets],
    queries: list[str | None],
    pol_u32: np.ndarray,
    token_logprobs: torch.Tensor,
    baseline_u32_by_q: dict[str, np.ndarray],
    base_hits: int,
    base_misses: int,
    gr_config: GRConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Doc2kVoteDelta]:
    """Score a batch of generated docID groups on GPU and return linear reward components + doc2k delta."""
    started = time.perf_counter()
    device = scorer.device
    batch_size, group_size = pol_u32.shape
    valid_rows: list[int] = []
    valid_queries: list[str] = []
    for idx, q in enumerate(queries):
        if q is not None:
            valid_rows.append(idx)
            valid_queries.append(q)
    n_valid = len(valid_rows)
    if n_valid == 0:
        zeros_2d = np.zeros((batch_size, group_size), dtype=np.float32)
        return (
            zeros_2d,
            zeros_2d.copy(),
            np.zeros((batch_size, _TOTAL_METRIC_COLS), dtype=np.float32),
            Doc2kVoteDelta(),
        )

    unique_queries: list[str] = list(dict.fromkeys(valid_queries))
    q_to_uidx: dict[str, int] = {q: idx for idx, q in enumerate(unique_queries)}
    pol_valid_u32 = pol_u32[valid_rows]
    (
        pol_lens_np,
        pol_edge_s_gpu,
        pol_edge_v_gpu,
        pol_edge_kid_gpu,
        miss_q_list,
        base_edge_m_np,
        base_edge_kid_np,
        unique_qk_gpu,
        inv_all_gpu,
        e_pol,
    ) = _lookup_policy_and_baseline_edges(
        mapping_engine=mapping_engine,
        pol_valid_u32=pol_valid_u32,
        valid_queries=valid_queries,
        q_to_uidx=q_to_uidx,
        baseline_u32_by_q=baseline_u32_by_q,
        group_size=group_size,
        device=device,
    )

    pm_sm_gpu, delta_stats = scorer.score_packed_qk_gpu(unique_queries, unique_qk_gpu)
    _populate_missing_baselines(
        miss_q_list=miss_q_list,
        base_edge_m_np=base_edge_m_np,
        base_edge_kid_np=base_edge_kid_np,
        pm_sm_gpu=pm_sm_gpu,
        inv_base_gpu=inv_all_gpu[e_pol:],
        baseline_kid_cache=baseline_kid_cache,
    )

    base_pm_list: list[frozenset[int]] = []
    base_sm_list: list[frozenset[int]] = []
    for q_str in valid_queries:
        bset = baseline_kid_cache.get(q_str)
        assert bset is not None
        base_pm_list.append(bset.pm)
        base_sm_list.append(bset.sm)

    (
        base_valid_np,
        unit_ub_valid_np,
        group_acc_np,
        sample_has_pm_np,
        p_ok,
        rel_el,
        ug_mask,
        pol_edge_kid_gpu,
    ) = _reduce_edge_metrics_gpu(
        n_valid=n_valid,
        group_size=group_size,
        pol_lens_gpu=torch.from_numpy(pol_lens_np).to(device=device, non_blocking=True),
        pol_edge_s_gpu=pol_edge_s_gpu,
        pol_edge_v_gpu=pol_edge_v_gpu,
        pol_edge_kid_gpu=pol_edge_kid_gpu,
        pm_sm_pol=pm_sm_gpu[inv_all_gpu[:e_pol]],
        sorted_bpm_gpu=pack_vk_sorted(base_pm_list, device),
        sorted_bsm_gpu=pack_vk_sorted(base_sm_list, device),
        gr_config=gr_config,
    )

    vote_delta = (
        accumulate_batch_doc2k_votes(
            pol_valid_u32=pol_valid_u32,
            pol_lens_np=pol_lens_np,
            pol_edge_s_gpu=pol_edge_s_gpu,
            pol_edge_kid_gpu=pol_edge_kid_gpu,
            p_ok=p_ok,
            rel_el=rel_el,
            ug_mask=ug_mask,
            sample_has_pm_np=sample_has_pm_np,
            token_logprobs=token_logprobs,
            valid_rows=valid_rows,
            group_size=group_size,
            gr_config=gr_config,
        )
        if not gr_config.doc2k_disabled
        else Doc2kVoteDelta()
    )

    base_mat = np.zeros((batch_size, group_size), dtype=np.float32)
    unit_ub_mat = np.zeros((batch_size, group_size), dtype=np.float32)
    metrics_mat = np.zeros((batch_size, _TOTAL_METRIC_COLS), dtype=np.float32)
    base_mat[valid_rows] = base_valid_np
    unit_ub_mat[valid_rows] = unit_ub_valid_np
    metrics_mat[valid_rows, :26] = group_acc_np

    elapsed = time.perf_counter() - started
    n_vf = float(max(1, n_valid))
    tail = np.array(
        [
            elapsed / n_vf,
            1.0,
            float(baseline_kid_cache.stats().size),
            0.0,
            float(scorer.query_cache_size),
            float(scorer.keyword_cache_size),
            float(base_hits) / n_vf,
            float(base_misses) / n_vf,
            0.0,
            float(int(unique_qk_gpu.numel())) / n_vf,
            float(delta_stats["q_hits"]) / n_vf,
            float(delta_stats["q_misses"]) / n_vf,
            float(delta_stats["k_hits"]) / n_vf,
            0.0,
        ],
        dtype=np.float32,
    )
    metrics_mat[valid_rows, 26:] = tail[None, :]
    return base_mat, unit_ub_mat, metrics_mat, vote_delta
