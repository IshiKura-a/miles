"""SGLang policy model for SLMv3 query-to-docid rollout with in-process per-GPU reward."""

from __future__ import annotations

import copy
from collections.abc import Iterable
import os
import sys
import pickle
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, cast

from filelock import FileLock
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
from transformers import AutoTokenizer, T5Config
from transformers.models.t5.modeling_t5 import T5Stack

from examples.experimental.generative_retriever.index.id_text_mapping import COLUMNAR_INDEX_FILES
from examples.experimental.generative_retriever.model.constrained_decode import (
    build_generation_model,
    generate_beams,
    generate_samples,
    load_constraint,
)
from examples.experimental.generative_retriever.reward.relevance_model import (
    RelevanceScorer,
    _stage_local_relevance_model,
    _stage_local_relevance_source,
    load_relevance_pipeline_factory,
    relevance_source_path,
)
from examples.experimental.generative_retriever.reward.relevance_reward import (
    BaselineSets,
    DOC2K_HEAD_RANK_CUTOFF,
    DOC2K_MAX_ADDS_PER_QUERY,
    DOC2K_MIN_GROUP_SAMPLES,
    DOC2K_MIN_HEAD_RANKS,
    DOC2K_PRUNE_FLOOR,
    DOCID_REWARD_FLOOR,
    DOCID_TOKEN_OFFSET,
    RewardMode,
    TREE_CREDIT_LEVEL_WEIGHT,
    TREE_CREDIT_TOP_K,
    build_baseline_set,
    score_and_summarize_docid,
)
from examples.experimental.generative_retriever.services.lru_cache import LRUCache
from examples.experimental.generative_retriever.services.mapping_service import ResidentMapping
from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3Model
from sglang.srt.models.qwen3_classification import Qwen3ForPooledOutput
from sglang.srt.utils import add_prefix

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
def _resolve_physical_gpu_id(device: torch.device | None = None) -> int:
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
    if device is not None and device.type == "cuda" and device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device()) if torch.cuda.is_available() else 0


_PROMPT_PREFIX = 'Given the user Query: "'
_PROMPT_SUFFIX = '", summarize it in one token:'


def _stage_local_mapping_dir(mapping_path: str | Path) -> Path:
    resolved = Path(mapping_path).resolve()
    local_root = Path(tempfile.gettempdir())
    target = resolved if resolved.is_relative_to(local_root) else local_root / f"gr_mapping_{resolved.name}"
    sentinel = target / ".staged_arrow_ok"
    if not sentinel.is_file():
        with FileLock(f"{target}.lock"):
            if not sentinel.is_file():
                target.mkdir(parents=True, exist_ok=True)
                for fname in COLUMNAR_INDEX_FILES:
                    src = resolved / fname
                    if not src.is_file():
                        raise FileNotFoundError(f"Columnar mapping file missing: {src}")
                    dst = target / fname
                    if dst != src and (not dst.is_file() or dst.stat().st_size != src.stat().st_size):
                        tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{fname}.", dir=target)
                        os.close(tmp_fd)
                        tmp_path = Path(tmp_name)
                        shutil.copyfile(src, tmp_path)
                        tmp_path.replace(dst)
                ipc_dst = target / "keywords_only.arrow"
                if not ipc_dst.is_file():
                    tmp_fd, tmp_name = tempfile.mkstemp(prefix=".keywords_only.arrow.", dir=target)
                    os.close(tmp_fd)
                    tmp_ipc = Path(tmp_name)
                    tbl = pq.read_table(target / "keywords.parquet", columns=["keyword"])
                    with pa.OSFile(str(tmp_ipc), "wb") as sink:
                        with pa.ipc.new_file(sink, tbl.schema) as writer:
                            writer.write_table(tbl)
                    tmp_ipc.replace(ipc_dst)
                sentinel.write_text(str(resolved), encoding="utf-8")
    return target


def _level_top_k_means(sorted_keys: np.ndarray, sorted_rewards: np.ndarray, ord_desc: np.ndarray, k: int) -> np.ndarray:
    n = len(sorted_keys)
    ord_group = np.argsort(sorted_keys, kind="mergesort")
    k_g = sorted_keys[ord_group]
    r_g = sorted_rewards[ord_group]
    orig_idx = ord_desc[ord_group]

    is_start = np.ones(n, dtype=bool)
    is_start[1:] = k_g[1:] != k_g[:-1]
    idx = np.arange(n, dtype=np.int32)
    start_pos = np.maximum.accumulate(np.where(is_start, idx, 0))
    valid = (idx - start_pos) < k

    cs_r = np.cumsum(np.where(valid, r_g, 0.0))
    cs_c = np.cumsum(valid.astype(np.int32))

    is_end = np.ones(n, dtype=bool)
    is_end[:-1] = is_start[1:]
    end_pos = np.minimum.accumulate(np.where(is_end[::-1], idx[::-1], n - 1))[::-1]

    prev_pos = start_pos - 1
    has_prev = prev_pos >= 0
    safe_prev = np.maximum(0, prev_pos)
    group_sum = cs_r[end_pos] - np.where(has_prev, cs_r[safe_prev], 0.0)
    group_cnt = np.maximum(1, cs_c[end_pos] - np.where(has_prev, cs_c[safe_prev], 0))
    means_sorted = group_sum / group_cnt.astype(np.float64)

    out = np.empty(n, dtype=np.float64)
    out[orig_idx] = means_sorted
    return out


def _compute_group_tree_credit_1d(u32_group: np.ndarray, raw_rewards: list[float]) -> list[float]:
    n = len(raw_rewards)
    if n <= 1:
        return list(raw_rewards)
    r = np.asarray(raw_rewards, dtype=np.float64)
    ord_desc = np.argsort(-r, kind="mergesort")
    u_sorted = u32_group[ord_desc]
    r_sorted = r[ord_desc]
    k0, k1, k2 = TREE_CREDIT_TOP_K
    v0 = _level_top_k_means(u_sorted >> np.uint32(24), r_sorted, ord_desc, k0)
    v1 = _level_top_k_means(u_sorted >> np.uint32(16), r_sorted, ord_desc, k1)
    v2 = _level_top_k_means(u_sorted >> np.uint32(8), r_sorted, ord_desc, k2)
    return (TREE_CREDIT_LEVEL_WEIGHT * (v0 + v1 + v2 + r)).tolist()


class GRQwen3ForDocIDGeneration(Qwen3ForPooledOutput):
    """Frozen Qwen3 encoder with SLMv3 Stage5 policy head, baseline head, and in-process GPU reward."""

    _qwen_tokenizer: Any
    _mapping_engine: ResidentMapping | None
    _relevance_scorer: RelevanceScorer | None

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        raw_gr_config = getattr(config, "gr_config", None)
        if not isinstance(raw_gr_config, dict):
            raise ValueError("GRQwen3ForDocIDGeneration requires config.gr_config")

        self.gr_config = T5Config.from_dict(raw_gr_config)
        self.docid_size = int(raw_gr_config["docid_size"])
        self.docid_dim = int(raw_gr_config["docid_dim"])
        self.num_beams = int(raw_gr_config["num_beams"])
        self.num_return_sequences = int(raw_gr_config["num_return_sequences"])
        self.decoding_mode = raw_gr_config["decoding_mode"]
        self.temperature = float(raw_gr_config["temperature"])
        if self.decoding_mode not in {"beam", "beam_sample", "sample"}:
            raise ValueError("GR decoding_mode must be 'beam', 'beam_sample', or 'sample'")
        if self.decoding_mode in {"beam", "beam_sample"} and not 1 <= self.num_return_sequences <= self.num_beams:
            raise ValueError("Require 1 <= num_return_sequences <= self.num_beams")
        self.id_mapping_path = str(raw_gr_config["id_mapping"])
        self.qwen_model_path = str(raw_gr_config.get("qwen_model") or os.environ.get("GR_QWEN_MODEL", ""))
        self.relevance_model_path = str(
            raw_gr_config.get("relevance_model") or os.environ.get("GR_RELEVANCE_MODEL", "")
        )
        self.reward_mode: RewardMode = cast(
            RewardMode,
            raw_gr_config.get("reward_mode") or os.environ.get("GR_REWARD_MODE", "discrete_relevance_unique"),
        )
        self.constraint = load_constraint(
            self.id_mapping_path, self.docid_size, self.docid_dim,
        )

        decoder_config = self.gr_config
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = True
        decoder_config.use_cache = False
        cast(Any, decoder_config).num_layers = decoder_config.num_decoder_layers
        decoder_vocab_size = self.docid_size * self.docid_dim + 2
        decoder_config.vocab_size = decoder_vocab_size

        self.docEnc2decoder = nn.Linear(
            config.hidden_size,
            decoder_config.d_model,
            bias=False,
        )
        self.docid_embedding = nn.Embedding(
            decoder_vocab_size,
            decoder_config.d_model,
            padding_idx=decoder_config.pad_token_id,
        )
        self.decoder = T5Stack(decoder_config)
        self.decoder.embed_tokens = self.docid_embedding
        self.lm_head = nn.Linear(
            decoder_config.d_model,
            decoder_vocab_size,
            bias=False,
        )
        generation_model = build_generation_model(
            slm_root=raw_gr_config["slm_root"], config=decoder_config,
            projection=self.docEnc2decoder, decoder=self.decoder, lm_head=self.lm_head,
        )
        object.__setattr__(self, "_generation_model", generation_model)

        self.baseline_model = Qwen3Model(
            config, quant_config=quant_config, prefix=add_prefix("baseline_model", prefix)
        )
        self.baseline_docEnc2decoder = copy.deepcopy(self.docEnc2decoder)
        self.baseline_docid_embedding = copy.deepcopy(self.docid_embedding)
        self.baseline_decoder = copy.deepcopy(self.decoder)
        self.baseline_decoder.embed_tokens = self.baseline_docid_embedding
        self.baseline_lm_head = copy.deepcopy(self.lm_head)
        for mod in (
            self.baseline_model,
            self.baseline_docEnc2decoder,
            self.baseline_docid_embedding,
            self.baseline_decoder,
            self.baseline_lm_head,
        ):
            mod.requires_grad_(False)
        baseline_generation_model = build_generation_model(
            slm_root=raw_gr_config["slm_root"], config=decoder_config,
            projection=self.baseline_docEnc2decoder,
            decoder=self.baseline_decoder,
            lm_head=self.baseline_lm_head,
        )
        object.__setattr__(self, "_baseline_generation_model", baseline_generation_model)
        if self.relevance_model_path:
            tokenizer = AutoTokenizer.from_pretrained(self.qwen_model_path, trust_remote_code=True)
            staged_map_dir = _stage_local_mapping_dir(self.id_mapping_path)
            mapping_engine = ResidentMapping(staged_map_dir, read_only=True)
            object.__setattr__(self, "_qwen_tokenizer", tokenizer)
            object.__setattr__(self, "_mapping_engine", mapping_engine)
        else:
            object.__setattr__(self, "_qwen_tokenizer", None)
            object.__setattr__(self, "_mapping_engine", None)
        object.__setattr__(self, "_relevance_scorer", None)
        self._baseline_sets_cache: LRUCache[str, BaselineSets] = LRUCache(32_768)
        self._baseline_initialized = False
        self._has_uncommitted_rollout = False
        self._step_cand_last_q: dict[tuple[int, int], int] = {}
        self._step_cand_votes: dict[tuple[int, int], int] = {}
        self._step_docid_q_count: dict[int, int] = {}
        self._step_docid_kids: dict[int, tuple[int, ...]] = {}
        self._step_edge_passes: dict[tuple[int, int], int] = {}
        self._query_seq_id = 0
        self._doc2k_sync_step = 0
        self._local_gpu_id = _resolve_physical_gpu_id()

    def _ensure_in_process_reward_engine(self, device: torch.device) -> None:
        if self._relevance_scorer is not None:
            return
        if self._qwen_tokenizer is None or self._mapping_engine is None:
            tokenizer = AutoTokenizer.from_pretrained(self.qwen_model_path, trust_remote_code=True)
            staged_map_dir = _stage_local_mapping_dir(self.id_mapping_path)
            mapping_engine = ResidentMapping(staged_map_dir, read_only=True)
            object.__setattr__(self, "_qwen_tokenizer", tokenizer)
            object.__setattr__(self, "_mapping_engine", mapping_engine)
        rel_src = _stage_local_relevance_source(relevance_source_path(self.relevance_model_path))
        rel_model = _stage_local_relevance_model(self.relevance_model_path)
        factory = load_relevance_pipeline_factory(rel_src)
        dev_str = f"cuda:{device.index or 0}" if device.type == "cuda" else str(device)
        pipeline = factory(
            rel_model,
            cup="large",
            device=dev_str,
            dtype="bf16",
            batch_size=4096,
            cross_batch_size=16384,
            show_progress=False,
        )
        scorer = RelevanceScorer(
            pipeline,
            feature_cache_size=32_768,
            score_cache_size=524_288,
            pair_batch_size=65_536,
            keyword_cache_size=67_108_864,
        )
        object.__setattr__(self, "_relevance_scorer", scorer)

    def _extract_queries(self, input_ids: torch.Tensor, forward_batch: ForwardBatch) -> list[str | None] | None:
        tokenizer = self._qwen_tokenizer
        lengths = forward_batch.extend_seq_lens_cpu
        if not lengths or sum(lengths) != input_ids.shape[0]:
            return None
        seqs = input_ids.detach().cpu().split(lengths)
        decoded_batch = tokenizer.batch_decode([seq.tolist() for seq in seqs], skip_special_tokens=False)
        queries: list[str | None] = []
        has_valid = False
        for decoded in decoded_batch:
            start = decoded.find(_PROMPT_PREFIX)
            if start < 0:
                queries.append(None)
                continue
            content_start = start + len(_PROMPT_PREFIX)
            end = decoded.rfind(_PROMPT_SUFFIX)
            if end < content_start:
                queries.append(None)
                continue
            q_str = decoded[content_start:end]
            queries.append(q_str if q_str.strip() else " ")
            has_valid = True
        return queries if has_valid else None

    @staticmethod
    def _split_hidden_states(
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_lens = forward_batch.extend_prefix_lens_cpu
        if prefix_lens is None or len(prefix_lens) != forward_batch.batch_size:
            raise RuntimeError("GR policy requires CPU prefill prefix lengths for every query")
        if any(prefix_lens):
            raise RuntimeError("GR policy requires --disable-radix-cache so every query has full hidden states")
        lengths = forward_batch.extend_seq_lens_cpu
        if not lengths or sum(lengths) != hidden_states.shape[0]:
            raise RuntimeError(
                f"invalid GR prefill lengths {lengths!r} for {hidden_states.shape[0]} hidden states"
            )

        sequences = hidden_states.split(lengths)
        max_length = max(lengths)
        padded = hidden_states.new_zeros((len(sequences), max_length, hidden_states.shape[-1]))
        attention_mask = torch.zeros(
            (len(sequences), max_length),
            dtype=torch.long,
            device=hidden_states.device,
        )
        for index, sequence in enumerate(sequences):
            padded[index, : sequence.shape[0]] = sequence
            attention_mask[index, : sequence.shape[0]] = 1
        return padded, attention_mask

    @staticmethod
    def _pack_u32_matrix(pos_docids_3d: np.ndarray) -> np.ndarray:
        c0 = pos_docids_3d[:, :, 0].astype(np.uint32)
        c1 = (pos_docids_3d[:, :, 1] - 256).astype(np.uint32)
        c2 = (pos_docids_3d[:, :, 2] - 512).astype(np.uint32)
        c3 = (pos_docids_3d[:, :, 3] - 768).astype(np.uint32)
        return (c0 << np.uint32(24)) | (c1 << np.uint32(16)) | (c2 << np.uint32(8)) | c3

    def _score_batch_in_process(
        self,
        queries: list[str | None],
        token_ids: torch.Tensor,
        token_logprobs: torch.Tensor,
        baseline_hidden_states: torch.Tensor | None,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        started = time.perf_counter()
        assert self._relevance_scorer is not None
        assert self._mapping_engine is not None
        scorer: RelevanceScorer = self._relevance_scorer
        mapping_engine: ResidentMapping = self._mapping_engine
        baseline_cache = self._baseline_sets_cache
        batch_size = len(queries)
        group_size = self.num_return_sequences
        valid_rows = [idx for idx, q in enumerate(queries) if q is not None]

        pol_pos_np = token_ids.detach().cpu().numpy().astype(np.int32) - DOCID_TOKEN_OFFSET
        pol_u32 = self._pack_u32_matrix(pol_pos_np)

        missing_base_indices: list[int] = []
        seen_missing_q: dict[str, int] = {}
        for idx in valid_rows:
            q = queries[idx]
            assert q is not None
            if baseline_cache.get(q) is None and q not in seen_missing_q:
                seen_missing_q[q] = len(missing_base_indices)
                missing_base_indices.append(idx)
        base_hits = len(valid_rows) - len(missing_base_indices)
        base_misses = len(missing_base_indices)

        base_u32_by_q: dict[str, np.ndarray] = {}
        u32_parts: list[np.ndarray] = [pol_u32[valid_rows].reshape(-1)]
        if missing_base_indices:
            assert baseline_hidden_states is not None
            idx_t = torch.tensor(missing_base_indices, dtype=torch.long, device=attention_mask.device)
            base_ids_t, _ = generate_beams(
                self._baseline_generation_model,
                baseline_hidden_states.index_select(0, idx_t),
                attention_mask.index_select(0, idx_t),
                self.constraint,
                code_num=self.docid_size,
                num_beams=self.num_beams,
                num_return_sequences=group_size,
                do_sample=False,
                temperature=1.0,
            )
            base_pos_np = base_ids_t.detach().cpu().numpy().astype(np.int32) - DOCID_TOKEN_OFFSET
            base_u32_mat = self._pack_u32_matrix(base_pos_np)
            u32_parts.append(base_u32_mat.reshape(-1))
            for pos, q_idx in enumerate(missing_base_indices):
                q_key = queries[q_idx]
                assert q_key is not None
                base_u32_by_q[q_key] = base_u32_mat[pos]

        pol_unique_u32 = np.unique(u32_parts[0])
        u32_to_kws, u32_to_kids = mapping_engine.lookup_u32_with_kids(pol_unique_u32)
        base_u32_to_kws = (
            mapping_engine.lookup_u32(np.unique(u32_parts[1]), use_initial=True)
            if len(u32_parts) > 1
            else {}
        )

        pairs_dict: dict[tuple[str, str], None] = {}
        base_kws_by_q: dict[str, list[str]] = {}
        for q_str, b_u32_row in base_u32_by_q.items():
            kws = list(dict.fromkeys(kw for u32_val in b_u32_row.tolist() for kw in base_u32_to_kws[int(u32_val)]))
            base_kws_by_q[q_str] = kws
            for kw in kws:
                pairs_dict[(q_str, kw)] = None

        pol_kws_per_query: dict[int, list[list[str]]] = {}
        for b in valid_rows:
            q_str = queries[b]
            assert q_str is not None
            row_u32 = pol_u32[b].tolist()
            q_sample_kws: list[list[str]] = []
            for u32_val in row_u32:
                kws = u32_to_kws[int(u32_val)]
                q_sample_kws.append(kws)
                for kw in kws:
                    pairs_dict[(q_str, kw)] = None
            pol_kws_per_query[b] = q_sample_kws

        pair_scores, scoring_delta = scorer.score_with_deltas(list(pairs_dict))

        for q_str, kws in base_kws_by_q.items():
            baseline_cache.put(
                q_str,
                build_baseline_set({kw: pair_scores[(q_str, kw)] for kw in kws}),
            )

        credited_mat = np.zeros((batch_size, group_size), dtype=np.float32)
        raw_mat = np.zeros((batch_size, group_size), dtype=np.float32)
        metrics_mat = np.zeros((batch_size, len(IN_PROCESS_METRIC_KEYS)), dtype=np.float32)
        metric_names = IN_PROCESS_METRIC_KEYS[:26]
        inv_group = 1.0 / float(group_size)
        head_cutoff = min(DOC2K_HEAD_RANK_CUTOFF, max(DOC2K_MIN_HEAD_RANKS, group_size // 2))

        for b in valid_rows:
            q_str = queries[b]
            assert q_str is not None
            self._query_seq_id += 1
            q_seq = self._query_seq_id
            b_set = baseline_cache.get(q_str)
            assert b_set is not None
            q_sample_kws = pol_kws_per_query[b]
            row_u32_list = [int(x) for x in pol_u32[b].tolist()]
            freq_map: dict[str, int] = {}
            for kws in q_sample_kws:
                for kw in kws:
                    freq_map[kw] = freq_map.get(kw, 0) + 1
            raw_rewards_b: list[float] = []
            sample_unique_good: list[list[str]] = []
            seen_fat_u32: set[int] = set()
            acc = [0.0] * 26
            for u32_val, kws in zip(row_u32_list, q_sample_kws, strict=True):
                val, m_dict, passed_kws, uniq_good_kws = score_and_summarize_docid(
                    kws,
                    pair_scores,
                    q_str,
                    b_set,
                    mode=self.reward_mode,
                    keyword_frequencies=freq_map,
                    k_floor=DOCID_REWARD_FLOOR,
                )
                raw_rewards_b.append(val)
                sample_unique_good.append(uniq_good_kws)
                if len(kws) > DOC2K_PRUNE_FLOOR and u32_val not in seen_fat_u32:
                    seen_fat_u32.add(u32_val)
                    kids_tup = u32_to_kids[u32_val]
                    self._step_docid_q_count[u32_val] = self._step_docid_q_count.get(u32_val, 0) + 1
                    self._step_docid_kids[u32_val] = kids_tup
                    if passed_kws:
                        passed_set = set(passed_kws)
                        for kw_s, kid_i in zip(kws, kids_tup, strict=True):
                            if kw_s in passed_set:
                                edge = (u32_val, kid_i)
                                self._step_edge_passes[edge] = self._step_edge_passes.get(edge, 0) + 1
                for m_idx, m_key in enumerate(metric_names):
                    acc[m_idx] += float(m_dict[m_key])
            raw_mat[b] = raw_rewards_b
            credited_mat[b] = _compute_group_tree_credit_1d(pol_u32[b], raw_rewards_b)
            for m_idx in range(26):
                metrics_mat[b, m_idx] = acc[m_idx] * inv_group

            if group_size >= DOC2K_MIN_GROUP_SAMPLES and group_size > head_cutoff:
                under_floor_heads: list[int] = []
                under_prune_heads: list[int] = []
                head_keywords: set[str] = set()
                for hi in range(head_cutoff):
                    hkws = q_sample_kws[hi]
                    head_keywords.update(hkws)
                    hlen = len(hkws)
                    if hlen < DOCID_REWARD_FLOOR:
                        under_floor_heads.append(hi)
                    elif hlen < DOC2K_PRUNE_FLOOR:
                        under_prune_heads.append(hi)
                if not under_floor_heads and not under_prune_heads:
                    continue
                under_floor_heads.sort(key=lambda hi: len(q_sample_kws[hi]))
                under_prune_heads.sort(key=lambda hi: len(q_sample_kws[hi]))
                candidate_tiers = (under_floor_heads, under_prune_heads)

                added_for_query = 0
                seen_tail: set[str] = set()
                for ti in range(head_cutoff, group_size):
                    if added_for_query >= DOC2K_MAX_ADDS_PER_QUERY:
                        break
                    ug_kws = sample_unique_good[ti]
                    if not ug_kws:
                        continue
                    tail_u32 = row_u32_list[ti]
                    tail_p2 = tail_u32 >> 16
                    tail_p1 = tail_u32 >> 24
                    target_u32: int | None = None
                    for tier in candidate_tiers:
                        if not tier:
                            continue
                        for hi in tier:
                            hu = row_u32_list[hi]
                            if (hu >> 16) == tail_p2:
                                target_u32 = hu
                                break
                        if target_u32 is not None:
                            break
                        for hi in tier:
                            hu = row_u32_list[hi]
                            if (hu >> 24) == tail_p1:
                                target_u32 = hu
                                break
                        if target_u32 is not None:
                            break
                    if target_u32 is None:
                        continue
                    tail_kws_list = q_sample_kws[ti]
                    tail_kids_tup = u32_to_kids[tail_u32]
                    kw_to_kid_local = dict(zip(tail_kws_list, tail_kids_tup, strict=True))
                    for kw in ug_kws:
                        if kw not in head_keywords and kw not in seen_tail:
                            seen_tail.add(kw)
                            kid = kw_to_kid_local.get(kw)
                            if kid is None:
                                continue
                            ckey = (target_u32, kid)
                            if self._step_cand_last_q.get(ckey) != q_seq:
                                self._step_cand_last_q[ckey] = q_seq
                                self._step_cand_votes[ckey] = self._step_cand_votes.get(ckey, 0) + 1
                            added_for_query += 1
                            if added_for_query >= DOC2K_MAX_ADDS_PER_QUERY:
                                break

        elapsed = time.perf_counter() - started
        q_stats = scorer.query_features.stats()
        k_stats = scorer.keyword_features.stats()
        s_stats = scorer.scores.stats()
        b_stats = baseline_cache.stats()
        q_delta = scoring_delta["query_features"]
        k_delta = scoring_delta["keyword_features"]
        r_delta = scoring_delta["reward"]
        n_valid = float(max(1, len(valid_rows)))

        tail = np.array(
            [
                elapsed / n_valid,
                1.0,
                float(b_stats.size),
                float(s_stats.size),
                float(q_stats.size),
                float(k_stats.size),
                float(base_hits) / n_valid,
                float(base_misses) / n_valid,
                float(r_delta["hits"]) / n_valid,
                float(r_delta["misses"]) / n_valid,
                float(q_delta["hits"]) / n_valid,
                float(q_delta["misses"]) / n_valid,
                float(k_delta["hits"]) / n_valid,
                float(k_delta["misses"]) / n_valid,
            ],
            dtype=np.float32,
        )
        metrics_mat[valid_rows, 26:] = tail[None, :]

        device = token_ids.device
        credited_t = torch.from_numpy(credited_mat).to(device=device).unsqueeze(-1)
        raw_t = torch.from_numpy(raw_mat).to(device=device).unsqueeze(-1)
        core = torch.cat((token_ids.float(), token_logprobs.float(), credited_t, raw_t), dim=-1).flatten(1)
        self._local_gpu_id = _resolve_physical_gpu_id(token_ids.device)
        self._stage_local_doc2k_votes()
        self._has_uncommitted_rollout = True
        metrics_t = torch.from_numpy(metrics_mat).to(device=device)
        packed = torch.cat((core, metrics_t), dim=-1)
        if device.type == "cuda" and (
            torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device) > 4 * 2**30
        ):
            torch.cuda.empty_cache()
        return packed

    def _stage_local_doc2k_votes(self) -> None:
        local_path = Path(tempfile.gettempdir()) / f"gr_doc2k_local_gpu{self._local_gpu_id}.pkl"
        tmp_path = local_path.with_suffix(".tmp")
        payload = {
            "cand_votes": self._step_cand_votes,
            "docid_q_count": self._step_docid_q_count,
            "docid_kids": self._step_docid_kids,
            "edge_passes": self._step_edge_passes,
        }
        tmp_path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        tmp_path.replace(local_path)

    def _generate_docids(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        baseline_hidden_states: torch.Tensor | None = None,
        queries: list[str | None] | None = None,
    ) -> EmbeddingPoolerOutput:
        if self.decoding_mode in {"beam", "beam_sample"}:
            token_ids, token_logprobs = generate_beams(
                self._generation_model, hidden_states, attention_mask, self.constraint,
                code_num=self.docid_size, num_beams=self.num_beams,
                num_return_sequences=self.num_return_sequences,
                do_sample=self.decoding_mode == "beam_sample",
                temperature=self.temperature,
            )
        else:
            token_ids, token_logprobs = generate_samples(
                self._generation_model, hidden_states, attention_mask, self.constraint,
                code_num=self.docid_size, num_return_sequences=self.num_return_sequences,
                temperature=self.temperature,
            )
        if queries is not None and self.num_return_sequences == self.num_beams:
            packed = self._score_batch_in_process(
                queries, token_ids, token_logprobs, baseline_hidden_states, attention_mask,
            )
            del token_ids, token_logprobs
            return self._dp_compatible_output(packed)
        base_hs = baseline_hidden_states if baseline_hidden_states is not None else hidden_states
        baseline_ids, _ = generate_beams(
            self._baseline_generation_model, base_hs, attention_mask, self.constraint,
            code_num=self.docid_size, num_beams=self.num_beams,
            num_return_sequences=self.num_return_sequences,
            do_sample=False,
            temperature=1.0,
        )
        packed = torch.cat(
            (token_ids.float(), token_logprobs.float(), baseline_ids.float()),
            dim=-1,
        ).flatten(1)
        del token_ids, token_logprobs, baseline_ids
        return self._dp_compatible_output(packed)

    @staticmethod
    def _dp_compatible_output(embeddings: torch.Tensor) -> EmbeddingPoolerOutput:
        output = EmbeddingPoolerOutput(embeddings=embeddings)
        cast(Any, output).next_token_logits = embeddings.new_empty((0, 0))
        cast(Any, output).hidden_states = None
        return output

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        get_embedding: bool = True,
    ) -> EmbeddingPoolerOutput:
        if not get_embedding:
            raise ValueError("GRQwen3ForDocIDGeneration only supports embedding-mode rollout")
        if not self._baseline_initialized and (
            bool(self.relevance_model_path) and self.num_return_sequences == self.num_beams
        ) and forward_batch.batch_size > 1:
            for b_param, p_param in zip(self.baseline_model.parameters(), self.model.parameters(), strict=True):
                b_param.data.copy_(p_param.data)
            self._baseline_initialized = True
        use_in_process_reward = bool(self.relevance_model_path) and self.num_return_sequences == self.num_beams
        packed_width = (
            self.num_return_sequences * IN_PROCESS_BEAM_COLS + len(IN_PROCESS_METRIC_KEYS)
            if use_in_process_reward
            else self.num_return_sequences * self.docid_size * 3
        )
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        if hidden_states.shape[0] == 0:
            return self._dp_compatible_output(
                hidden_states.new_empty((0, packed_width)),
            )
        queries: list[str | None] | None = None
        need_baseline_hs = not use_in_process_reward
        if use_in_process_reward:
            self._ensure_in_process_reward_engine(hidden_states.device)
            queries = self._extract_queries(input_ids, forward_batch)
            if queries is None:
                return self._dp_compatible_output(
                    hidden_states.new_zeros((forward_batch.batch_size, packed_width), dtype=torch.float32),
                )
            need_baseline_hs = any(
                q is not None and self._baseline_sets_cache.get(q) is None
                for q in queries
            )
        baseline_hs_padded: torch.Tensor | None = None
        if need_baseline_hs:
            base_hs_raw = self.baseline_model(input_ids, positions, forward_batch, input_embeds)
            baseline_hs_padded, _ = self._split_hidden_states(base_hs_raw, forward_batch)
        hidden_states, attention_mask = self._split_hidden_states(hidden_states, forward_batch)
        return self._generate_docids(
            hidden_states,
            attention_mask,
            baseline_hidden_states=baseline_hs_padded,
            queries=queries,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        params = dict(self.named_parameters())
        loaded_head_keys: list[str] = []
        qwen_weights: list[tuple[str, torch.Tensor]] = []
        for original_name, loaded_weight in weights:
            if original_name.startswith("encoder_list.0.model.encoder."):
                qwen_name = original_name.removeprefix("encoder_list.0.model.encoder.")
                qwen_weights.append((qwen_name, loaded_weight))
                continue

            name = original_name
            if name in {"docid_embedding.weight", "decoder.embed_tokens.weight"}:
                target_name = "docid_embedding.weight"
            elif name.startswith(("docEnc2decoder.", "decoder.", "lm_head.")):
                target_name = name
            else:
                continue

            parameter = params.get(target_name)
            if parameter is None:
                raise KeyError(f"GR rollout parameter not found: {target_name}")
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            loader(parameter, loaded_weight)
            if not self._baseline_initialized:
                baseline_target = f"baseline_{target_name}"
                baseline_param = params.get(baseline_target)
                if baseline_param is not None:
                    b_loader = getattr(baseline_param, "weight_loader", default_weight_loader)
                    b_loader(baseline_param, loaded_weight)
                    loaded_head_keys.append(baseline_target)
        if qwen_weights:
            super().load_weights(qwen_weights)
        if self._has_uncommitted_rollout and (loaded_head_keys or qwen_weights):
            self._has_uncommitted_rollout = False
            self._commit_step_doc2k_updates()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _commit_step_doc2k_updates(self) -> None:
        if self._mapping_engine is None:
            return
        mapping_engine = self._mapping_engine
        self._doc2k_sync_step += 1
        self._step_cand_last_q.clear()
        self._step_cand_votes.clear()
        self._step_docid_q_count.clear()
        self._step_docid_kids.clear()
        self._step_edge_passes.clear()

        global_path = Path(tempfile.gettempdir()) / f"gr_doc2k_global_gpu{self._local_gpu_id}.pkl"
        if global_path.is_file():
            payload = pickle.loads(global_path.read_bytes())
            global_path.unlink(missing_ok=True)
            additions: dict[int, list[int]] = payload["additions"]
            removals: dict[int, set[int]] = payload["removals"]
        else:
            additions = {}
            removals = {}

        stats = mapping_engine.apply_u32_kid_delta(additions, removals)
        print(
            f"[GR doc2k global] step={self._doc2k_sync_step} gpu={self._local_gpu_id} "
            f"updated_docids={stats['updated_docids']} added_edges={stats['added_edges']} "
            f"removed_edges={stats['removed_edges']} overlay_docids={stats['overlay_docids']}",
            flush=True,
        )


EntryClass = GRQwen3ForDocIDGeneration
