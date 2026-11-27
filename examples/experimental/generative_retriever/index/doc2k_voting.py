"""Shared doc2k target-bucket selection and per-query vote accumulation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from examples.experimental.generative_retriever.reward.relevance_reward import (
    DOC2K_HEAD_RANK_CUTOFF,
    DOC2K_MAX_ADDS_PER_QUERY,
    DOC2K_MIN_GROUP_SAMPLES,
    DOC2K_MIN_HEAD_RANKS,
    DOC2K_PRUNE_FLOOR,
    DOCID_REWARD_FLOOR,
)


@dataclass(slots=True)
class Doc2kVoteDelta:
    """Per-batch sparse vote increments emitted by rollout scoring."""

    cand_votes: dict[tuple[int, int], int] = field(default_factory=dict)
    docid_q_count: dict[int, int] = field(default_factory=dict)
    edge_passes: dict[tuple[int, int], int] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (self.cand_votes or self.docid_q_count or self.edge_passes)


def compute_head_cutoff(group_size: int) -> int:
    return min(DOC2K_HEAD_RANK_CUTOFF, max(DOC2K_MIN_HEAD_RANKS, group_size // 2))


def compute_natural_winner_counts(rewards_2d: np.ndarray) -> np.ndarray:
    """Compute per-query natural winner counts ``M_+(q)`` above positive-reward mean ``E[R | R > 0]``."""
    rw = np.asarray(rewards_2d, dtype=np.float64)
    if rw.ndim != 2 or rw.shape[1] <= 1:
        return np.zeros(rw.shape[0] if rw.ndim >= 1 else 0, dtype=np.int32)
    std = rw.std(axis=1, ddof=1, keepdims=True)
    has_var = std > 1.0e-12
    grp_mean = rw.mean(axis=1, keepdims=True)
    pos_mask = rw > 0.0
    pos_count = np.maximum(1, pos_mask.sum(axis=1, keepdims=True))
    pos_mean = np.where(pos_mask, rw, 0.0).sum(axis=1, keepdims=True) / pos_count
    has_strict = (rw > (pos_mean + 1.0e-12)).any(axis=1, keepdims=True)
    thresh = np.where(has_strict, pos_mean, grp_mean)
    winners = has_var & (rw > thresh)
    return winners.sum(axis=1, dtype=np.int32)


def partition_head_and_tail_indices(
    group_size: int,
    lp_sums_1d: np.ndarray,
    *,
    head_cutoff: int | None = None,
) -> tuple[list[int], list[int]]:
    cutoff = (
        compute_head_cutoff(group_size)
        if head_cutoff is None
        else max(0, min(int(head_cutoff), group_size))
    )
    beam_order = np.argsort(-lp_sums_1d, kind="mergesort").tolist()
    return beam_order[:cutoff], beam_order[cutoff:]


def build_candidate_head_tiers(
    head_indices: Sequence[int],
    sample_lens: Sequence[int],
    sample_has_pm: Sequence[bool],
    *,
    k_floor: int = DOCID_REWARD_FLOOR,
    prune_floor: int = DOC2K_PRUNE_FLOOR,
) -> tuple[list[int], list[int]]:
    under_floor: list[int] = []
    under_prune: list[int] = []
    for hi in head_indices:
        if not bool(sample_has_pm[hi]):
            continue
        hlen = int(sample_lens[hi])
        if hlen < k_floor:
            under_floor.append(hi)
        elif hlen < prune_floor:
            under_prune.append(hi)
    return under_floor, under_prune


def select_doc2k_target_u32(
    tail_u32: int,
    row_u32: Sequence[int],
    candidate_tiers: tuple[Sequence[int], Sequence[int]],
) -> int | None:
    """Select the target head docID for tail enrichment.

    Matches L3 ``(c0, c1, c2)`` siblings first, then L2 ``(c0, c1)`` cousins in
    policy log-prob rank order, without falling back to coarse L1 ``(c0)``.
    """
    tail_p3 = tail_u32 >> 8
    tail_p2 = tail_u32 >> 16
    for tier in candidate_tiers:
        if not tier:
            continue
        for hi in tier:
            hu = int(row_u32[hi])
            if (hu >> 8) == tail_p3:
                return hu
        for hi in tier:
            hu = int(row_u32[hi])
            if (hu >> 16) == tail_p2:
                return hu
    return None


def record_query_enrichment_votes(
    *,
    row_u32: Sequence[int],
    sample_lens: Sequence[int],
    sample_has_pm: Sequence[bool],
    head_indices: Sequence[int],
    tail_indices: Sequence[int],
    head_kids_set: set[int],
    tail_unique_good_kids: Sequence[Iterable[int]],
    delta: Doc2kVoteDelta,
    k_floor: int = DOCID_REWARD_FLOOR,
    prune_floor: int = DOC2K_PRUNE_FLOOR,
    max_adds_per_query: int = DOC2K_MAX_ADDS_PER_QUERY,
) -> None:
    group_size = len(row_u32)
    if group_size < DOC2K_MIN_GROUP_SAMPLES or not head_indices or not tail_indices:
        return
    tiers = build_candidate_head_tiers(
        head_indices,
        sample_lens,
        sample_has_pm,
        k_floor=k_floor,
        prune_floor=prune_floor,
    )
    if not tiers[0] and not tiers[1]:
        return

    added_for_query = 0
    seen_tail_kids: set[int] = set()
    seen_query_edges: set[tuple[int, int]] = set()
    for local_pos, ti in enumerate(tail_indices):
        if added_for_query >= max_adds_per_query:
            break
        ug_iter = tail_unique_good_kids[local_pos]
        ug_list = [int(k) for k in ug_iter]
        if not ug_list:
            continue
        target_u32 = select_doc2k_target_u32(
            int(row_u32[ti]),
            row_u32,
            tiers,
        )
        if target_u32 is None:
            continue
        for kid_i in ug_list:
            if kid_i in head_kids_set or kid_i in seen_tail_kids:
                continue
            seen_tail_kids.add(kid_i)
            ckey = (target_u32, kid_i)
            if ckey not in seen_query_edges:
                seen_query_edges.add(ckey)
                delta.cand_votes[ckey] = delta.cand_votes.get(ckey, 0) + 1
            added_for_query += 1
            if added_for_query >= max_adds_per_query:
                break
