"""Resolve the source snapshot shipped with a Graphene relevance checkpoint."""

from __future__ import annotations

from collections import OrderedDict
import importlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
import torch.nn.functional as F

from examples.experimental.generative_retriever.services.lru_cache import CacheStats, LRUCache

EXPECTED_GRAPHENE_COMMIT = "a57cb597de0891c1e0a2612d12366188b96aa0cc"
_PIPELINE_PATH = Path("graphene/multi_cup/multi_cup_graphene_pipeline.py")
_QUERY_PREFIX = "Search query: "
_KEYWORD_PREFIX = "Document: "
_CROSS_CHUNK_SIZE = 8192
_TOWER_MAX_BATCH_SIZE = 1024
_KEYWORD_PAD_THRESHOLD = 64
_KEYWORD_PAD_MULTIPLE = 256
_QUERY_PAD_THRESHOLD = 16
_QUERY_PAD_MULTIPLE = 16
_MIN_PAD_MULTIPLE = 4
GRAPHENE_NUM_TOKENS = 2
GRAPHENE_FEATURE_DIM = 128
GRAPHENE_NUM_SCORE_HEADS = 4
DEFAULT_QUERY_CACHE_SIZE = 16384
DEFAULT_KEYWORD_CACHE_SIZE = 4194304
DEFAULT_SCORE_CACHE_SIZE = 1048576
DEFAULT_PAIR_BATCH_SIZE = 32768
logger = logging.getLogger(__name__)


def fast_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
    **kwargs: object,
) -> torch.Tensor:
    del kwargs
    kernel_size = weight.shape[-1]
    windows = F.pad(x, (kernel_size - 1, 0)).unfold(-1, kernel_size, 1)
    out = (windows.float() * weight.view(1, -1, 1, kernel_size).float()).sum(dim=-1)
    if bias is not None:
        out = out + bias.view(1, -1, 1).float()
    out_typed = out.to(dtype=x.dtype)
    return F.silu(out_typed) if activation in ("silu", "swish") else out_typed


def make_fast_single_chunk_gated_delta_rule(orig_fn):
    def _fast_rule(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        chunk_size: int = 64,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        **kwargs: object,
    ):
        seq_len = key.shape[1]
        if initial_state is not None or output_final_state or seq_len > chunk_size:
            return orig_fn(
                query,
                key,
                value,
                g,
                beta,
                chunk_size=chunk_size,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                **kwargs,
            )
        initial_dtype = query.dtype
        if use_qk_l2norm_in_kernel:
            query = query * torch.rsqrt((query * query).sum(dim=-1, keepdim=True) + 1e-6).to(query.dtype)
            key = key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6).to(key.dtype)
        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32)
            for x in (query, key, value, beta, g)
        ]
        query = query * (query.shape[-1] ** -0.5)
        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)
        mask0 = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device), diagonal=0)
        g = g.cumsum(dim=-1)
        decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask0, 0)
        for idx in range(1, seq_len):
            row = attn[..., idx, :idx].clone()
            sub = attn[..., :idx, :idx].clone()
            attn[..., idx, :idx] = row + (row.unsqueeze(-1) * sub).sum(-2)
        attn = attn + torch.eye(seq_len, dtype=attn.dtype, device=attn.device)
        v_new = attn @ v_beta
        attn_qk = (query @ key.transpose(-1, -2)) * decay_mask
        core_attn_out = attn_qk @ v_new
        return core_attn_out.transpose(1, 2).contiguous().to(initial_dtype), None

    return _fast_rule


def optimize_graphene_pipeline(pipeline: object) -> None:
    for attr in ("query_engine", "keyword_engine"):
        engine: Any = getattr(pipeline, attr, None)
        model = getattr(engine, "model", None)
        if engine is None or model is None:
            continue
        cur_batch_size = getattr(engine, "batch_size", None)
        if isinstance(cur_batch_size, int):
            engine.batch_size = min(cur_batch_size, _TOWER_MAX_BATCH_SIZE)
        for mod in model.modules():
            if type(mod).__name__ == "Qwen3_5GatedDeltaNet":
                if getattr(mod, "causal_conv1d_fn", None) is None:
                    mod.causal_conv1d_fn = fast_causal_conv1d
                orig_chunk = getattr(mod, "chunk_gated_delta_rule", None)
                if callable(orig_chunk) and getattr(orig_chunk, "__name__", "") != "_fast_rule":
                    mod.chunk_gated_delta_rule = make_fast_single_chunk_gated_delta_rule(orig_chunk)


class RelevancePipeline(Protocol):
    device: torch.device
    dtype: torch.dtype
    cross_batch_size: int
    query_engine: Any
    keyword_engine: Any
    cross_model: Any

    def encode_query(self, queries: list[str]) -> np.ndarray: ...

    def encode_keyword(self, keywords: list[str]) -> np.ndarray: ...

    def cross_score(self, query_vectors: np.ndarray, keyword_vectors: np.ndarray) -> np.ndarray: ...


class RelevancePipelineFactory(Protocol):
    def __call__(
        self,
        model_path: str,
        cup: str,
        device: str,
        dtype: str,
        batch_size: int,
        cross_batch_size: int,
        show_progress: bool,
    ) -> RelevancePipeline: ...


class GpuEmbeddingTable:
    """Preallocated in-VRAM bfloat16 embedding table with LRU slot management."""

    def __init__(
        self,
        capacity: int,
        *,
        num_tokens: int = GRAPHENE_NUM_TOKENS,
        feature_dim: int = GRAPHENE_FEATURE_DIM,
        device: torch.device | str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if capacity <= 0:
            raise ValueError("Embedding table capacity must be positive")
        self.capacity = capacity
        self.num_tokens = num_tokens
        self.feature_dim = feature_dim
        self.device = torch.device(device)
        self.dtype = dtype
        self.storage: torch.Tensor | None = None
        self._slots: OrderedDict[str, int] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def ensure_storage(self, num_tokens: int, feature_dim: int) -> torch.Tensor:
        if self.storage is None:
            self.num_tokens = num_tokens
            self.feature_dim = feature_dim
            self.storage = torch.empty(
                (self.capacity, num_tokens, feature_dim),
                dtype=self.dtype,
                device=self.device,
            )
        return self.storage

    def lookup_and_reserve(
        self, texts: list[str],
    ) -> tuple[dict[str, int], list[str], list[int], dict[str, int]]:
        unique_texts = list(dict.fromkeys(texts))
        if len(unique_texts) > self.capacity:
            raise ValueError(
                f"Batch unique texts ({len(unique_texts)}) exceed embedding table capacity ({self.capacity})"
            )
        slot_map: dict[str, int] = {}
        missing_texts: list[str] = []
        hits = 0
        misses = 0
        evictions = 0
        for text in unique_texts:
            slot = self._slots.pop(text, None)
            if slot is not None:
                self._slots[text] = slot
                self._hits += 1
                hits += 1
                slot_map[text] = slot
            else:
                self._misses += 1
                misses += 1
                missing_texts.append(text)
        missing_slots: list[int] = []
        for text in missing_texts:
            if len(self._slots) < self.capacity:
                slot = len(self._slots)
            else:
                _, slot = self._slots.popitem(last=False)
                self._evictions += 1
                evictions += 1
            self._slots[text] = slot
            slot_map[text] = slot
            missing_slots.append(slot)
        return slot_map, missing_texts, missing_slots, {
            "hits": hits,
            "misses": misses,
            "evictions": evictions,
        }

    def stats(self) -> CacheStats:
        return CacheStats(
            capacity=self.capacity,
            size=len(self._slots),
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
        )


class RelevanceScorer:
    """Two-level GPU relevance cache: L1 (q, k) score cache + L2 (q_emb, k_emb) GPU tables.

    The caller serializes access to the pipeline and caches.
    """

    def __init__(
        self,
        pipeline: RelevancePipeline,
        *,
        feature_cache_size: int = DEFAULT_QUERY_CACHE_SIZE,
        score_cache_size: int = DEFAULT_SCORE_CACHE_SIZE,
        pair_batch_size: int = DEFAULT_PAIR_BATCH_SIZE,
        keyword_cache_size: int = DEFAULT_KEYWORD_CACHE_SIZE,
    ) -> None:
        if pair_batch_size <= 0:
            raise ValueError("Relevance pair batch size must be positive")
        optimize_graphene_pipeline(pipeline)
        self.pipeline = pipeline
        self.pair_batch_size = pair_batch_size
        self.query_features = GpuEmbeddingTable(
            feature_cache_size,
            device=pipeline.device,
            dtype=pipeline.dtype,
        )
        self.keyword_features = GpuEmbeddingTable(
            keyword_cache_size,
            device=pipeline.device,
            dtype=pipeline.dtype,
        )
        self.scores: LRUCache[tuple[str, str], tuple[float, float]] = LRUCache(
            score_cache_size
        )

    @torch.no_grad()
    def _encode_gpu_table(
        self,
        texts: list[str],
        *,
        query: bool,
    ) -> tuple[torch.Tensor, dict[str, int], dict[str, int]]:
        table = self.query_features if query else self.keyword_features
        slot_map, missing_texts, missing_slots, delta = table.lookup_and_reserve(texts)
        if missing_texts:
            engine = self.pipeline.query_engine if query else self.pipeline.keyword_engine
            prefix = _QUERY_PREFIX if query else _KEYWORD_PREFIX
            pad_multiple = (
                _KEYWORD_PAD_MULTIPLE
                if (not query and len(missing_texts) >= _KEYWORD_PAD_THRESHOLD)
                else (
                    _QUERY_PAD_MULTIPLE
                    if (query and len(missing_texts) >= _QUERY_PAD_THRESHOLD)
                    else _MIN_PAD_MULTIPLE
                )
            )
            padded = missing_texts + [missing_texts[-1]] * (-len(missing_texts) % pad_multiple)
            prompts = [prefix + text for text in padded]
            batch_size = int(getattr(engine, "batch_size", _TOWER_MAX_BATCH_SIZE))
            chunks: list[torch.Tensor] = []
            for start in range(0, len(prompts), batch_size):
                token_ids, attention = engine._prepare_batch(prompts[start : start + batch_size])
                chunks.append(engine.model(token_ids, attention).to(dtype=table.dtype))
            encoded = torch.cat(chunks, dim=0)[: len(missing_texts)]
            if encoded.ndim != 3 or not bool(torch.isfinite(encoded).all()):
                raise RuntimeError("Official relevance model returned invalid GPU tower features")
            storage = table.ensure_storage(encoded.shape[1], encoded.shape[2])
            slot_tensor = torch.tensor(missing_slots, dtype=torch.long, device=storage.device)
            storage[slot_tensor] = encoded
        storage = table.ensure_storage(table.num_tokens, table.feature_dim)
        return storage, slot_map, delta

    @torch.no_grad()
    def encode_queries_cached(
        self, queries: list[str],
    ) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
        unique_queries = list(dict.fromkeys(queries))
        if not unique_queries:
            return {}, {"hits": 0, "misses": 0, "evictions": 0}
        q_storage, q_map, q_delta = self._encode_gpu_table(unique_queries, query=True)
        slots = torch.tensor([q_map[q] for q in unique_queries], dtype=torch.long, device=q_storage.device)
        fetched = q_storage[slots].detach().cpu()
        return (
            {query: fetched[idx] for idx, query in enumerate(unique_queries)},
            q_delta,
        )

    @torch.no_grad()
    def _cross_score_gpu(
        self,
        q_embeds_gpu: torch.Tensor,
        k_storage: torch.Tensor,
        q_indices: list[int],
        k_slots: list[int],
    ) -> np.ndarray:
        device = k_storage.device
        q_idx = torch.tensor(q_indices, dtype=torch.long, device=device)
        k_idx = torch.tensor(k_slots, dtype=torch.long, device=device)
        cross_chunk = max(_CROSS_CHUNK_SIZE, int(self.pipeline.cross_batch_size))
        scores_gpu: list[torch.Tensor] = []
        for start in range(0, len(q_indices), cross_chunk):
            end = start + cross_chunk
            scores_gpu.append(
                self.pipeline.cross_model(q_embeds_gpu[q_idx[start:end]], k_storage[k_idx[start:end]]).float()
            )
        all_scores = torch.cat(scores_gpu, dim=0)
        if (
            all_scores.shape != (len(q_indices), GRAPHENE_NUM_SCORE_HEADS)
            or not bool(torch.isfinite(all_scores).all())
            or not bool(((all_scores >= 0) & (all_scores <= 1)).all())
        ):
            raise RuntimeError("Official relevance model returned invalid scores")
        return all_scores[:, 1:3].cpu().numpy()

    @torch.no_grad()
    def score_with_query_embeddings(
        self,
        query_embeddings: torch.Tensor,
        keywords: list[str],
        q_indices: list[int],
        k_indices: list[int],
    ) -> tuple[list[tuple[float, float]], dict[str, int]]:
        if len(q_indices) != len(k_indices):
            raise ValueError("q_indices and k_indices must have equal length")
        if not q_indices:
            return [], {"hits": 0, "misses": 0, "evictions": 0}
        started = time.perf_counter()
        q_gpu = query_embeddings.to(device=self.pipeline.device, dtype=self.pipeline.dtype)
        phase = time.perf_counter()
        k_storage, k_map, k_delta = self._encode_gpu_table(keywords, query=False)
        keyword_seconds = time.perf_counter() - phase
        mapped_k_slots = [k_map[kw] for kw in keywords]
        pair_k_slots = [mapped_k_slots[i] for i in k_indices]
        phase = time.perf_counter()
        pm_sm = self._cross_score_gpu(
            q_gpu,
            k_storage,
            q_indices,
            pair_k_slots,
        )
        cross_seconds = time.perf_counter() - phase
        results = [(float(row[0]), float(row[1])) for row in pm_sm]
        logger.info(
            "graphene_score_stage pairs=%d keywords=%d keyword_s=%.3f cross_s=%.3f total_s=%.3f",
            len(q_indices), len(keywords), keyword_seconds, cross_seconds, time.perf_counter() - started,
        )
        return results, k_delta

    def score(self, pairs: list[tuple[str, str]]) -> dict[tuple[str, str], tuple[float, float]]:
        started = time.perf_counter()
        resolved = {}
        missing = []
        for pair in dict.fromkeys(pairs):
            score = self.scores.get(pair)
            if score is None:
                missing.append(pair)
            else:
                resolved[pair] = score
        lookup_seconds = time.perf_counter() - started
        query_seconds = keyword_seconds = cross_seconds = store_seconds = 0.0
        if missing:
            unique_queries = list(dict.fromkeys(query for query, _ in missing))
            unique_keywords = list(dict.fromkeys(keyword for _, keyword in missing))
            phase = time.perf_counter()
            q_storage, q_map, _ = self._encode_gpu_table(unique_queries, query=True)
            query_seconds = time.perf_counter() - phase
            phase = time.perf_counter()
            k_storage, k_map, _ = self._encode_gpu_table(unique_keywords, query=False)
            keyword_seconds = time.perf_counter() - phase
            phase = time.perf_counter()
            pm_sm = self._cross_score_gpu(
                q_storage,
                k_storage,
                [q_map[query] for query, _ in missing],
                [k_map[keyword] for _, keyword in missing],
            )
            cross_seconds = time.perf_counter() - phase
            phase = time.perf_counter()
            for pair, row in zip(missing, pm_sm, strict=True):
                score = (float(row[0]), float(row[1]))
                self.scores.put(pair, score)
                resolved[pair] = score
            store_seconds = time.perf_counter() - phase
        logger.info(
            "graphene_score_stage pairs=%d misses=%d lookup_s=%.3f query_s=%.3f "
            "keyword_s=%.3f cross_s=%.3f store_s=%.3f total_s=%.3f",
            len(pairs), len(missing), lookup_seconds, query_seconds, keyword_seconds,
            cross_seconds, store_seconds, time.perf_counter() - started,
        )
        return resolved


def _validate_source(path: Path) -> Path:
    source = path.expanduser().resolve()
    if not (source / _PIPELINE_PATH).is_file():
        raise FileNotFoundError(f"Graphene pipeline is missing from relevance source: {source}")
    commit_path = source / "GIT_COMMIT.txt"
    commit = commit_path.read_text(encoding="utf-8").strip()
    if commit != EXPECTED_GRAPHENE_COMMIT:
        raise ValueError(
            f"Graphene source commit mismatch: expected {EXPECTED_GRAPHENE_COMMIT}, got {commit!r}"
        )
    return source


def resolve_relevance_source(
    relevance_model: str | Path,
    relevance_source: str | Path | None = None,
) -> Path:
    if relevance_source is not None:
        return _validate_source(Path(relevance_source))

    model = Path(relevance_model).expanduser().resolve()
    pointer = model / "SOURCE_CHECKPOINT.txt"
    if not pointer.is_file():
        raise FileNotFoundError(f"Relevance model has no SOURCE_CHECKPOINT.txt: {model}")
    source_checkpoint = Path(pointer.read_text(encoding="utf-8").strip())
    direct_source = source_checkpoint / "code"
    if direct_source.is_dir():
        return _validate_source(direct_source)

    cosmos_root = os.environ.get("cosmos_data_path")
    marker = "/local/"
    source_text = os.fspath(source_checkpoint)
    if not cosmos_root or marker not in source_text:
        raise FileNotFoundError(
            "Cannot resolve relevance source snapshot; set cosmos_data_path or pass --relevance-source"
        )
    relative_checkpoint = source_text.split(marker, 1)[1]
    return _validate_source(Path(cosmos_root) / "local" / relative_checkpoint / "code")


def load_relevance_pipeline_factory(source: str | Path) -> RelevancePipelineFactory:
    resolved = _validate_source(Path(source))
    source_text = os.fspath(resolved)
    if source_text not in sys.path:
        sys.path.append(source_text)
    module = importlib.import_module("graphene.multi_cup.multi_cup_graphene_pipeline")
    factory = getattr(module, "MultiCupGraphenePipeline", None)
    if not callable(factory):
        raise TypeError("Graphene source does not expose MultiCupGraphenePipeline")
    return cast(RelevancePipelineFactory, factory)
