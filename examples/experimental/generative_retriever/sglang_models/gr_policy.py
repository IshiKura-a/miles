"""SGLang policy model for SLMv3 query-to-docid rollout with in-process per-GPU reward."""

from __future__ import annotations

import copy
from collections.abc import Iterable
import logging
import os
import pickle
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, T5Config
from transformers.models.t5.modeling_t5 import T5Stack

from examples.experimental.generative_retriever.index.fast_doc2k_sync import (
    append_local_doc2k_delta,
    global_doc2k_delta_path,
)
from examples.experimental.generative_retriever.model.constrained_decode import (
    build_generation_model,
    generate_beams,
    generate_samples,
    load_constraint,
)
from examples.experimental.generative_retriever.reward.gpu_batch_scorer import (
    score_batch_gpu_vectorized,
)
from examples.experimental.generative_retriever.reward.in_process_protocol import (
    expected_in_process_embedding_width,
    pack_docids_u32_from_tokens,
    pack_in_process_embedding,
    resolve_physical_gpu_id,
)
from examples.experimental.generative_retriever.reward.precompiled_keywords import (
    BaselineKidSets,
    FastPrecompileScorer,
    LiveRelevancePairScorer,
    PackedPairScorer,
    PrecompiledKeywordStore,
    resolve_precompiled_kw_paths,
)
from examples.experimental.generative_retriever.reward.relevance_model import (
    RelevanceScorer,
    load_relevance_pipeline_factory,
    relevance_source_path,
    stage_local_relevance_model,
    stage_local_relevance_source,
)
from examples.experimental.generative_retriever.runtime.gr_config import GRConfig, RewardModeLiteral
from examples.experimental.generative_retriever.services.lru_cache import LRUCache
from examples.experimental.generative_retriever.services.mapping_service import ResidentMapping
from examples.experimental.generative_retriever.services.mapping_staging import stage_local_mapping_dir
from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3Model
from sglang.srt.models.qwen3_classification import Qwen3ForPooledOutput
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)
_PROMPT_PREFIX = 'Given the user Query: "'
_PROMPT_SUFFIX = '", summarize it in one token:'


class GRQwen3ForDocIDGeneration(Qwen3ForPooledOutput):
    """Frozen Qwen3 encoder with SLMv3 Stage5 policy head, baseline head, and in-process GPU reward."""

    _qwen_tokenizer: Any
    _mapping_engine: ResidentMapping | None
    _pair_scorer: PackedPairScorer | None

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

        self.runtime_cfg = GRConfig.from_env()
        bundle_reward_mode = raw_gr_config.get("reward_mode")
        if bundle_reward_mode and bundle_reward_mode != self.runtime_cfg.reward_mode:
            self.runtime_cfg = GRConfig(
                **{
                    **{k: getattr(self.runtime_cfg, k) for k in self.runtime_cfg.__dataclass_fields__},
                    "reward_mode": cast(RewardModeLiteral, bundle_reward_mode),
                }
            )

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
        self.constraint = load_constraint(self.id_mapping_path, self.docid_size, self.docid_dim)

        decoder_config = self._configure_decoder_config(self.gr_config)
        self._init_policy_and_baseline_heads(
            config=config,
            decoder_config=decoder_config,
            slm_root=str(raw_gr_config["slm_root"]),
            quant_config=quant_config,
            prefix=prefix,
        )

        if self.relevance_model_path:
            tokenizer = AutoTokenizer.from_pretrained(self.qwen_model_path, trust_remote_code=True)
            mapping_engine = ResidentMapping(stage_local_mapping_dir(self.id_mapping_path), read_only=True)
            object.__setattr__(self, "_qwen_tokenizer", tokenizer)
            object.__setattr__(self, "_mapping_engine", mapping_engine)
        else:
            object.__setattr__(self, "_qwen_tokenizer", None)
            object.__setattr__(self, "_mapping_engine", None)
        object.__setattr__(self, "_pair_scorer", None)
        self._baseline_kid_cache: LRUCache[str, BaselineKidSets] = LRUCache(32_768)
        self._baseline_initialized = False
        self._has_uncommitted_rollout = False
        self._doc2k_sync_step = 0
        self._local_gpu_id = resolve_physical_gpu_id()

    def _configure_decoder_config(self, decoder_config: T5Config) -> T5Config:
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = True
        decoder_config.use_cache = False
        cast(Any, decoder_config).num_layers = decoder_config.num_decoder_layers
        decoder_config.vocab_size = self.docid_size * self.docid_dim + 2
        return decoder_config

    def _init_policy_and_baseline_heads(
        self,
        *,
        config: Any,
        decoder_config: T5Config,
        slm_root: str,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        vocab_size = decoder_config.vocab_size
        self.docEnc2decoder = nn.Linear(config.hidden_size, decoder_config.d_model, bias=False)
        self.docid_embedding = nn.Embedding(
            vocab_size,
            decoder_config.d_model,
            padding_idx=decoder_config.pad_token_id,
        )
        self.decoder = T5Stack(decoder_config)
        self.decoder.embed_tokens = self.docid_embedding
        self.lm_head = nn.Linear(decoder_config.d_model, vocab_size, bias=False)
        generation_model = build_generation_model(
            slm_root=slm_root,
            config=decoder_config,
            projection=self.docEnc2decoder,
            decoder=self.decoder,
            lm_head=self.lm_head,
            grouped_cross_attention=self.runtime_cfg.grouped_cross_attention,
        )
        object.__setattr__(self, "_generation_model", generation_model)

        self.baseline_model = Qwen3Model(
            config,
            quant_config=quant_config,
            prefix=add_prefix("baseline_model", prefix),
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
            slm_root=slm_root,
            config=decoder_config,
            projection=self.baseline_docEnc2decoder,
            decoder=self.baseline_decoder,
            lm_head=self.baseline_lm_head,
            grouped_cross_attention=self.runtime_cfg.grouped_cross_attention,
        )
        object.__setattr__(self, "_baseline_generation_model", baseline_generation_model)

    def _ensure_in_process_reward_engine(self, device: torch.device) -> None:
        if self._pair_scorer is not None:
            return
        if self._qwen_tokenizer is None or self._mapping_engine is None:
            tokenizer = AutoTokenizer.from_pretrained(self.qwen_model_path, trust_remote_code=True)
            mapping_engine = ResidentMapping(stage_local_mapping_dir(self.id_mapping_path), read_only=True)
            object.__setattr__(self, "_qwen_tokenizer", tokenizer)
            object.__setattr__(self, "_mapping_engine", mapping_engine)
        assert self._mapping_engine is not None

        pre_paths = resolve_precompiled_kw_paths(self._mapping_engine.path, self.id_mapping_path)
        if pre_paths is not None:
            kw_store = PrecompiledKeywordStore(pre_paths[0], pre_paths[1])
            fast_scorer = FastPrecompileScorer(
                self.relevance_model_path,
                kw_store,
                device,
                gr_config=self.runtime_cfg,
                kw_degrees=self._mapping_engine.keyword_degrees,
            )
            object.__setattr__(self, "_pair_scorer", fast_scorer)
            hbm_gib = (
                fast_scorer.hbm_cache.total_slots
                * kw_store.num_tokens
                * kw_store.feature_dim
                * 2
                / (1024**3)
            )
            logger.info(
                "[GR precompiled kw] gpu=%d ready: rows=%d hbm_slots=%d (%.1f GiB) src=%s",
                self._local_gpu_id,
                kw_store.num_keywords,
                fast_scorer.hbm_cache.total_slots,
                hbm_gib,
                kw_store.active_path,
            )
            return

        rel_src = stage_local_relevance_source(relevance_source_path(self.relevance_model_path))
        rel_model = stage_local_relevance_model(self.relevance_model_path)
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
        live_scorer = RelevanceScorer(
            pipeline,
            feature_cache_size=32_768,
            score_cache_size=524_288,
            pair_batch_size=65_536,
            keyword_cache_size=33_554_432,
        )
        object.__setattr__(self, "_pair_scorer", LiveRelevancePairScorer(live_scorer, self._mapping_engine))

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

    def _score_batch_in_process(
        self,
        queries: list[str | None],
        token_ids: torch.Tensor,
        token_logprobs: torch.Tensor,
        baseline_hidden_states: torch.Tensor | None,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        assert self._pair_scorer is not None and self._mapping_engine is not None
        device = token_ids.device
        batch_size = len(queries)
        valid_rows = [idx for idx, q in enumerate(queries) if q is not None]
        if not valid_rows:
            return torch.zeros(
                (batch_size, expected_in_process_embedding_width(self.num_return_sequences)),
                dtype=torch.float32,
                device=device,
            )

        missing_base_indices: list[int] = []
        seen_missing_q: set[str] = set()
        for idx in valid_rows:
            q = queries[idx]
            assert q is not None
            if self._baseline_kid_cache.get(q) is None and q not in seen_missing_q:
                seen_missing_q.add(q)
                missing_base_indices.append(idx)
        base_hits = len(valid_rows) - len(missing_base_indices)
        base_misses = len(missing_base_indices)

        pol_u32 = pack_docids_u32_from_tokens(token_ids.detach().cpu().numpy())
        base_u32_by_q: dict[str, np.ndarray] = {}
        if missing_base_indices:
            assert baseline_hidden_states is not None
            miss_idx_t = torch.tensor(missing_base_indices, dtype=torch.long, device=device)
            baseline_ids, _ = generate_beams(
                self._baseline_generation_model,
                baseline_hidden_states.index_select(0, miss_idx_t),
                attention_mask.index_select(0, miss_idx_t),
                self.constraint,
                code_num=self.docid_size,
                num_beams=self.num_beams,
                num_return_sequences=self.num_return_sequences,
                do_sample=False,
                temperature=1.0,
                query_chunk_size=self.runtime_cfg.tree_query_chunk_size,
            )
            base_u32_mat = pack_docids_u32_from_tokens(baseline_ids.detach().cpu().numpy())
            del baseline_ids
            for pos, orig_idx in enumerate(missing_base_indices):
                q_key = queries[orig_idx]
                assert q_key is not None
                base_u32_by_q[q_key] = base_u32_mat[pos]

        base_mat, unit_ub_mat, metrics_mat, vote_delta = score_batch_gpu_vectorized(
            scorer=self._pair_scorer,
            mapping_engine=self._mapping_engine,
            baseline_kid_cache=self._baseline_kid_cache,
            queries=queries,
            pol_u32=pol_u32,
            token_logprobs=token_logprobs,
            baseline_u32_by_q=base_u32_by_q,
            base_hits=base_hits,
            base_misses=base_misses,
            gr_config=self.runtime_cfg,
        )
        self._local_gpu_id = resolve_physical_gpu_id(device)
        if not self.runtime_cfg.doc2k_disabled:
            append_local_doc2k_delta(self._local_gpu_id, vote_delta)
            self._has_uncommitted_rollout = True
        packed = pack_in_process_embedding(
            token_ids=token_ids,
            token_logprobs=token_logprobs,
            base_docid_mat=base_mat,
            unit_ub_docid_mat=unit_ub_mat,
            metrics_mat=metrics_mat,
        )
        if device.type == "cuda" and (
            torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device) > 2 * 2**30
        ):
            torch.cuda.empty_cache()
        return packed

    def _generate_docids(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        baseline_hidden_states: torch.Tensor | None = None,
        queries: list[str | None] | None = None,
    ) -> EmbeddingPoolerOutput:
        is_beam_sample = self.decoding_mode == "beam_sample"
        if self.decoding_mode in {"beam", "beam_sample"}:
            token_ids, token_logprobs = generate_beams(
                self._generation_model,
                hidden_states,
                attention_mask,
                self.constraint,
                code_num=self.docid_size,
                num_beams=self.num_beams,
                num_return_sequences=self.num_return_sequences,
                do_sample=is_beam_sample,
                temperature=self.temperature,
                step_temperatures=self.runtime_cfg.step_temperatures if is_beam_sample else None,
                query_chunk_size=self.runtime_cfg.tree_query_chunk_size,
            )
        else:
            token_ids, token_logprobs = generate_samples(
                self._generation_model,
                hidden_states,
                attention_mask,
                self.constraint,
                code_num=self.docid_size,
                num_return_sequences=self.num_return_sequences,
                temperature=self.temperature,
            )
        if queries is not None and self.num_return_sequences == self.num_beams:
            packed = self._score_batch_in_process(
                queries,
                token_ids,
                token_logprobs,
                baseline_hidden_states,
                attention_mask,
            )
            del token_ids, token_logprobs
            return self._dp_compatible_output(packed)
        base_hs = baseline_hidden_states if baseline_hidden_states is not None else hidden_states
        baseline_ids, _ = generate_beams(
            self._baseline_generation_model,
            base_hs,
            attention_mask,
            self.constraint,
            code_num=self.docid_size,
            num_beams=self.num_beams,
            num_return_sequences=self.num_return_sequences,
            do_sample=False,
            temperature=1.0,
            query_chunk_size=self.runtime_cfg.tree_query_chunk_size,
        )
        packed = torch.cat((token_ids.float(), token_logprobs.float(), baseline_ids.float()), dim=-1).flatten(1)
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
        use_in_process_reward = bool(self.relevance_model_path) and self.num_return_sequences == self.num_beams
        if not self._baseline_initialized and use_in_process_reward and forward_batch.batch_size > 1:
            for b_param, p_param in zip(self.baseline_model.parameters(), self.model.parameters(), strict=True):
                b_param.data.copy_(p_param.data)
            self._baseline_initialized = True
        packed_width = (
            expected_in_process_embedding_width(self.num_return_sequences)
            if use_in_process_reward
            else self.num_return_sequences * self.docid_size * 3
        )
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        if hidden_states.shape[0] == 0:
            return self._dp_compatible_output(hidden_states.new_empty((0, packed_width)))
        queries: list[str | None] | None = None
        need_baseline_hs = not use_in_process_reward
        if use_in_process_reward:
            self._ensure_in_process_reward_engine(hidden_states.device)
            queries = self._extract_queries(input_ids, forward_batch)
            if queries is None:
                return self._dp_compatible_output(
                    hidden_states.new_zeros((forward_batch.batch_size, packed_width), dtype=torch.float32)
                )
            need_baseline_hs = any(q is not None and self._baseline_kid_cache.get(q) is None for q in queries)
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
                qwen_weights.append((original_name.removeprefix("encoder_list.0.model.encoder."), loaded_weight))
                continue
            if original_name in {"docid_embedding.weight", "decoder.embed_tokens.weight"}:
                target_name = "docid_embedding.weight"
            elif original_name.startswith(("docEnc2decoder.", "decoder.", "lm_head.")):
                target_name = original_name
            else:
                continue
            parameter = params.get(target_name)
            if parameter is None:
                raise KeyError(f"GR rollout parameter not found: {target_name}")
            getattr(parameter, "weight_loader", default_weight_loader)(parameter, loaded_weight)
            if not self._baseline_initialized:
                baseline_target = f"baseline_{target_name}"
                baseline_param = params.get(baseline_target)
                if baseline_param is not None:
                    getattr(baseline_param, "weight_loader", default_weight_loader)(baseline_param, loaded_weight)
                    loaded_head_keys.append(baseline_target)
        had_qwen_weights = bool(qwen_weights)
        if had_qwen_weights:
            super().load_weights(qwen_weights)
            qwen_weights.clear()
        if not self._baseline_initialized and loaded_head_keys and had_qwen_weights:
            for b_param, p_param in zip(self.baseline_model.parameters(), self.model.parameters(), strict=True):
                b_param.data.copy_(p_param.data)
            self._baseline_initialized = True
        if self._has_uncommitted_rollout and (loaded_head_keys or had_qwen_weights):
            self._has_uncommitted_rollout = False
            self._commit_step_doc2k_updates()

    def _commit_step_doc2k_updates(self) -> None:
        if self.runtime_cfg.doc2k_disabled or self._mapping_engine is None:
            return
        self._doc2k_sync_step += 1
        global_path = global_doc2k_delta_path(self._local_gpu_id)
        if global_path.is_file():
            payload = pickle.loads(global_path.read_bytes())
            global_path.unlink(missing_ok=True)
            additions: dict[int, list[int]] = payload["additions"]
            removals: dict[int, set[int]] = payload["removals"]
        else:
            additions = {}
            removals = {}
        stats = self._mapping_engine.apply_u32_kid_delta(additions, removals)
        logger.info(
            "[GR doc2k global] step=%d gpu=%d updated_docids=%d added_edges=%d removed_edges=%d overlay_docids=%d",
            self._doc2k_sync_step,
            self._local_gpu_id,
            stats["updated_docids"],
            stats["added_edges"],
            stats["removed_edges"],
            stats["overlay_docids"],
        )


EntryClass = GRQwen3ForDocIDGeneration
