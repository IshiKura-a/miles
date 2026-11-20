from __future__ import annotations

from argparse import Namespace
import math
from pathlib import Path
import threading

import numpy as np
import pytest
import torch

from examples.experimental.generative_retriever.index.doc2k_voting import (
    build_candidate_head_tiers,
    select_doc2k_target_u32,
)
from examples.experimental.generative_retriever.reward.gpu_batch_scorer import (
    score_batch_gpu_vectorized,
)
from examples.experimental.generative_retriever.reward.in_process_protocol import (
    IN_PROCESS_METRIC_KEYS,
    pack_in_process_embedding,
    unpack_in_process_embedding,
)
from examples.experimental.generative_retriever.reward.precompiled_keywords import (
    BaselineKidSets,
)
from examples.experimental.generative_retriever.reward.relevance_reward import (
    BaselineSets,
    DOCID_TOKEN_OFFSET,
    score_and_summarize_docid,
)
from examples.experimental.generative_retriever.runtime.gr_config import GRConfig
from examples.experimental.generative_retriever.services.lru_cache import LRUCache
from miles.backends.fsdp_utils.checkpoint import (
    _checkpoint_sync_sort_key,
    _sync_staged_checkpoint,
)
from miles.backends.training_utils.data import _unpack_compact_token_shard
from miles.utils.types import COMPACT_TOKEN_SHARD_KEY


def test_gr_config_unique_bonus_annealing_modes() -> None:
    cfg_none = GRConfig(unique_anneal_mode="none", unique_weight_start=2.0, unique_weight_end=0.0)
    assert cfg_none.resolve_unique_weight(0, 100) == 2.0
    assert cfg_none.resolve_unique_weight(99, 100) == 2.0

    cfg_off = GRConfig(unique_anneal_mode="off", unique_weight_start=2.0)
    assert cfg_off.resolve_unique_weight(0, 100) == 0.0
    assert cfg_off.resolve_unique_weight(50, 100) == 0.0

    cfg_lin = GRConfig(
        unique_anneal_mode="linear",
        unique_weight_start=2.0,
        unique_weight_end=0.4,
        unique_anneal_warmup_ratio=0.2,
    )
    assert cfg_lin.resolve_unique_weight(0, 101) == pytest.approx(2.0)
    assert cfg_lin.resolve_unique_weight(20, 101) == pytest.approx(2.0)
    assert cfg_lin.resolve_unique_weight(60, 101) == pytest.approx(1.2)
    assert cfg_lin.resolve_unique_weight(100, 101) == pytest.approx(0.4)

    cfg_cos = GRConfig(unique_anneal_mode="cosine", unique_weight_start=2.0, unique_weight_end=0.0)
    assert cfg_cos.resolve_unique_weight(0, 101) == pytest.approx(2.0)
    assert cfg_cos.resolve_unique_weight(50, 101) == pytest.approx(1.0)
    assert cfg_cos.resolve_unique_weight(100, 101) == pytest.approx(0.0)

    roundtrip = GRConfig.from_env(cfg_lin.to_env_dict())
    assert roundtrip == cfg_lin


def test_relevance_guided_doc2k_prefers_l3_sibling_over_earlier_l2_cousin() -> None:
    # Beam 0 shares L2 (10, 20, 99, 1), Beam 1 shares full L3 (10, 20, 30, 2), Beam 2 has no PM.
    row_u32 = [
        (10 << 24) | (20 << 16) | (99 << 8) | 1,
        (10 << 24) | (20 << 16) | (30 << 8) | 2,
        (10 << 24) | (20 << 16) | (30 << 8) | 3,
    ]
    tail_u32 = (10 << 24) | (20 << 16) | (30 << 8) | 4
    tiers = build_candidate_head_tiers(
        head_indices=[2, 0, 1],
        sample_lens=[4, 4, 2],
        sample_has_pm=[True, True, False],
        relevance_guided=True,
    )
    # Non-PM beam 2 must be filtered out; L3 sibling beam 1 must beat earlier L2 cousin beam 0.
    assert tiers == ([0, 1], [])
    target = select_doc2k_target_u32(tail_u32, row_u32, tiers, relevance_guided=True)
    assert target == row_u32[1]

    # L1-only match must be rejected in relevance_guided mode, accepted in legacy mode.
    l1_only_tail = (10 << 24) | (77 << 16) | (88 << 8) | 9
    assert select_doc2k_target_u32(l1_only_tail, row_u32, tiers, relevance_guided=True) is None
    assert select_doc2k_target_u32(l1_only_tail, row_u32, tiers, relevance_guided=False) == row_u32[0]


class _FakeMapping:
    def __init__(self, u32_to_kids: dict[int, tuple[int, ...]], num_keywords: int) -> None:
        self._map = u32_to_kids
        self.num_keywords = num_keywords

    def lookup_flat_kids_csr(
        self,
        u32_1d: np.ndarray,
        *,
        use_initial: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del use_initial
        lens = []
        rows = []
        kids = []
        for idx, u32 in enumerate(u32_1d.tolist()):
            tup = self._map[int(u32)]
            lens.append(len(tup))
            rows.extend([idx] * len(tup))
            kids.extend(tup)
        return (
            np.asarray(lens, dtype=np.int32),
            np.asarray(rows, dtype=np.int64),
            np.asarray(kids, dtype=np.int64),
        )


class _FakePairScorer:
    def __init__(self, scores_by_kid: dict[int, tuple[float, float]]) -> None:
        self.device = torch.device("cpu")
        self._scores = scores_by_kid

    @property
    def query_cache_size(self) -> int:
        return 1

    @property
    def keyword_cache_size(self) -> int:
        return len(self._scores)

    def score_packed_qk_gpu(
        self,
        unique_queries: list[str],
        unique_qk_gpu: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, int]]:
        del unique_queries
        kids = (unique_qk_gpu & 0xFFFFFFFF).tolist()
        out = torch.tensor([self._scores[int(k)] for k in kids], dtype=torch.float32)
        return out, {"q_hits": 1, "q_misses": 0, "k_hits": len(kids)}


@pytest.mark.parametrize("denom_mode", ["sqrt_floor", "unit", "linear_floor"])
@pytest.mark.parametrize("unique_weight", [0.0, 0.75, 2.0])
def test_gpu_batch_scorer_linear_unique_weight_matches_reference(denom_mode: str, unique_weight: float) -> None:
    u32_to_kids = {
        101: (0, 1, 2),
        102: tuple(range(12)),
        103: (1, 12),
        104: (2, 3, 13),
    }
    scores_by_kid = {
        i: (float(np.float32(0.21 + 0.05 * (i % 15))), float(np.float32(0.51 + 0.04 * (i % 12))))
        for i in range(14)
    }
    pair_scores = {("q0", f"kw_{k}"): v for k, v in scores_by_kid.items()}

    base_u32 = np.array([101, 103], dtype=np.uint32)
    pol_u32 = np.array([[101, 102, 103, 104]], dtype=np.uint32)
    cfg = GRConfig(docid_denom_mode=denom_mode, doc2k_mode="disabled")  # type: ignore[arg-type]
    base_cache: LRUCache[str, BaselineKidSets] = LRUCache(16)
    token_ids = torch.full((1, 4, 4), DOCID_TOKEN_OFFSET, dtype=torch.long)
    token_lp = torch.zeros((1, 4, 4), dtype=torch.float32)

    base_mat, unit_ub_mat, metrics_mat, _ = score_batch_gpu_vectorized(
        scorer=_FakePairScorer(scores_by_kid),
        mapping_engine=_FakeMapping(u32_to_kids, 14),  # type: ignore[arg-type]
        baseline_kid_cache=base_cache,
        queries=["q0"],
        pol_u32=pol_u32,
        token_logprobs=token_lp,
        baseline_u32_by_q={"q0": base_u32},
        base_hits=0,
        base_misses=1,
        gr_config=cfg,
    )
    packed = pack_in_process_embedding(
        token_ids=token_ids,
        token_logprobs=token_lp,
        base_docid_mat=base_mat,
        unit_ub_docid_mat=unit_ub_mat,
        metrics_mat=metrics_mat,
    ).numpy()
    _, _, _, raw_f64, scaled_metrics = unpack_in_process_embedding(packed, 4, unique_weight)

    base_kids = {k for u in base_u32.tolist() for k in u32_to_kids[int(u)]}
    ref_base = BaselineSets(
        pm=frozenset(f"kw_{k}" for k in base_kids if scores_by_kid[k][0] >= 0.58),
        sm=frozenset(f"kw_{k}" for k in base_kids if scores_by_kid[k][1] >= 0.82),
    )
    freq: dict[str, int] = {}
    for u in pol_u32[0].tolist():
        for k in u32_to_kids[int(u)]:
            key = f"kw_{k}"
            freq[key] = freq.get(key, 0) + 1

    ref_vals = []
    ref_metrics_acc = np.zeros(26, dtype=np.float64)
    for u in pol_u32[0].tolist():
        kws = [f"kw_{k}" for k in u32_to_kids[int(u)]]
        val, m_dict, _, _ = score_and_summarize_docid(
            kws,
            pair_scores,
            "q0",
            ref_base,
            denom_mode=denom_mode,  # type: ignore[arg-type]
            unique_weight=unique_weight,
            keyword_frequencies=freq,
        )
        ref_vals.append(val)
        for idx, m_key in enumerate(IN_PROCESS_METRIC_KEYS[:26]):
            ref_metrics_acc[idx] += m_dict[m_key] / 4.0

    assert np.allclose(raw_f64[0], np.asarray(ref_vals), atol=1e-6)
    assert np.allclose(scaled_metrics[0, :26], ref_metrics_acc, atol=1e-6)


def test_atomic_staged_checkpoint_sync_commits_tracker_only_after_all_ranks(tmp_path: Path) -> None:
    assert _checkpoint_sync_sort_key(Path("model/__0_0.distcp")) < _checkpoint_sync_sort_key(
        Path("mapping/overlay.parquet")
    )
    assert _checkpoint_sync_sort_key(Path("mapping/overlay.parquet")) < _checkpoint_sync_sort_key(Path("meta.json"))

    base_dir = tmp_path / "checkpoints"
    ckpt_dir = base_dir / "iter_0000005"
    sync_done = ckpt_dir / ".sync_done"
    sync_done.mkdir(parents=True)

    stage0 = tmp_path / "stage_r0"
    stage1 = tmp_path / "stage_r1"
    w0 = stage0 / "iter_0000005"
    w1 = stage1 / "iter_0000005"
    (w0 / "model").mkdir(parents=True)
    (w1 / "model").mkdir(parents=True)
    (w0 / "model/__0_0.distcp").write_bytes(b"r0")
    (w0 / "meta.json").write_text('{"iteration": 5}', encoding="utf-8")
    (w1 / "model/__1_0.distcp").write_bytes(b"r1")

    t0 = threading.Thread(
        target=_sync_staged_checkpoint,
        kwargs={
            "stage_root": stage0,
            "write_ckpt_dir": w0,
            "checkpoint_dir": ckpt_dir,
            "base_dir": base_dir,
            "sync_done_dir": sync_done,
            "step_id": 5,
            "rank": 0,
            "world_size": 2,
        },
    )
    t0.start()
    # Rank 1 has not synced yet, so tracker MUST NOT exist even after Rank 0 copies meta.json.
    assert not (base_dir / "latest_checkpointed_iteration.txt").exists()

    _sync_staged_checkpoint(
        stage_root=stage1,
        write_ckpt_dir=w1,
        checkpoint_dir=ckpt_dir,
        base_dir=base_dir,
        sync_done_dir=sync_done,
        step_id=5,
        rank=1,
        world_size=2,
    )
    t0.join(timeout=5.0)
    assert not t0.is_alive()
    assert (base_dir / "latest_checkpointed_iteration.txt").read_text(encoding="utf-8") == "5"
    assert (ckpt_dir / "model/__0_0.distcp").read_bytes() == b"r0"
    assert (ckpt_dir / "model/__1_0.distcp").read_bytes() == b"r1"
    assert not sync_done.exists()


def test_compact_token_shard_validates_contract_and_unpacks() -> None:
    shard = {
        COMPACT_TOKEN_SHARD_KEY: True,
        "total_lengths": [6, 5],
        "response_length": 4,
        "base_sample_idx": 10,
        "flat_tokens_i32": np.array([100, 101, 2, 3, 4, 5, 0, 6, 7, 8, 9], dtype=np.int32),
        "rollout_log_probs_2d": np.zeros((2, 4), dtype=np.float32),
    }
    bad_args = Namespace(enable_witness=True, qkv_format="thd", true_on_policy_mode=False)
    with pytest.raises(ValueError, match=COMPACT_TOKEN_SHARD_KEY):
        _unpack_compact_token_shard(bad_args, dict(shard), current_device="cpu")  # type: ignore[arg-type]

    good_args = Namespace(enable_witness=False, qkv_format="thd", true_on_policy_mode=False)
    out = _unpack_compact_token_shard(good_args, dict(shard), current_device="cpu")  # type: ignore[arg-type]
    tokens = out["tokens"]
    assert isinstance(tokens, list)
    assert math.isclose(float(torch.as_tensor(out["rollout_mask_sums"])[0].item()), 4.0)
    assert [int(torch.as_tensor(t).numel()) for t in tokens] == [6, 5]
    assert list(out["sample_indices"]) == [10, 11]
