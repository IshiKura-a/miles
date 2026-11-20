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


def partition_head_and_tail_indices(
    group_size: int,
    lp_sums_1d: np.ndarray | None,
    *,
    relevance_guided: bool,
) -> tuple[list[int], list[int]]:
    head_cutoff = compute_head_cutoff(group_size)
    if relevance_guided and lp_sums_1d is not None:
        beam_order = np.argsort(-lp_sums_1d, kind="mergesort").tolist()
        return beam_order[:head_cutoff], beam_order[head_cutoff:]
    return list(range(head_cutoff)), list(range(head_cutoff, group_size))


def build_candidate_head_tiers(
    head_indices: Sequence[int],
    sample_lens: Sequence[int],
    sample_has_pm: Sequence[bool],
    *,
    relevance_guided: bool,
    k_floor: int = DOCID_REWARD_FLOOR,
    prune_floor: int = DOC2K_PRUNE_FLOOR,
) -> tuple[list[int], list[int]]:
    under_floor: list[int] = []
    under_prune: list[int] = []
    for hi in head_indices:
        if relevance_guided and not bool(sample_has_pm[hi]):
            continue
        hlen = int(sample_lens[hi])
        if hlen < k_floor:
            under_floor.append(hi)
        elif hlen < prune_floor:
            under_prune.append(hi)
    if not relevance_guided:
        under_floor.sort(key=lambda hi: int(sample_lens[hi]))
        under_prune.sort(key=lambda hi: int(sample_lens[hi]))
    return under_floor, under_prune


def select_doc2k_target_u32(
    tail_u32: int,
    row_u32: Sequence[int],
    candidate_tiers: tuple[Sequence[int], Sequence[int]],
    *,
    relevance_guided: bool,
) -> int | None:
    """Select the target head docID for tail enrichment.

    In ``relevance_guided`` mode, matches L3 ``(c0, c1, c2)`` siblings first, then L2 ``(c0, c1)``
    cousins, preserving policy log-prob rank order and never falling back to noisy L1 ``(c0)``.
    In ``legacy`` mode, matches L2 then L1 in ascending bucket-length order.
    """
    tail_p3 = tail_u32 >> 8
    tail_p2 = tail_u32 >> 16
    tail_p1 = tail_u32 >> 24
    for tier in candidate_tiers:
        if not tier:
            continue
        if relevance_guided:
            for hi in tier:
                hu = int(row_u32[hi])
                if (hu >> 8) == tail_p3:
                    return hu
            for hi in tier:
                hu = int(row_u32[hi])
                if (hu >> 16) == tail_p2:
                    return hu
        else:
            for hi in tier:
                hu = int(row_u32[hi])
                if (hu >> 16) == tail_p2:
                    return hu
            for hi in tier:
                hu = int(row_u32[hi])
                if (hu >> 24) == tail_p1:
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
    relevance_guided: bool,
    delta: Doc2kVoteDelta,
    k_floor: int = DOCID_REWARD_FLOOR,
    prune_floor: int = DOC2K_PRUNE_FLOOR,
    max_adds_per_query: int = DOC2K_MAX_ADDS_PER_QUERY,
) -> None:
    group_size = len(row_u32)
    head_cutoff = compute_head_cutoff(group_size)
    if group_size < DOC2K_MIN_GROUP_SAMPLES or group_size <= head_cutoff:
        return
    tiers = build_candidate_head_tiers(
        head_indices,
        sample_lens,
        sample_has_pm,
        relevance_guided=relevance_guided,
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
            relevance_guided=relevance_guided,
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
