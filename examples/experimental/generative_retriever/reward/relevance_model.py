"""Resolve the source snapshot shipped with a Graphene relevance checkpoint."""

from __future__ import annotations

from examples.experimental.generative_retriever.utils.paths import absolute_path

from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import gc
import importlib
import logging
import multiprocessing as mp
from multiprocessing.connection import Connection
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, cast

import numpy as np
from filelock import FileLock
import torch
import torch.nn.functional as F
import xxhash

from examples.experimental.generative_retriever.services.lru_cache import CacheStats, LRUCache

EXPECTED_GRAPHENE_COMMIT = "a57cb597de0891c1e0a2612d12366188b96aa0cc"
_PIPELINE_PATH = Path("graphene/multi_cup/multi_cup_graphene_pipeline.py")
_QUERY_PREFIX = "Search query: "
_KEYWORD_PREFIX = "Document: "
_CROSS_CHUNK_SIZE = 16384
_TOWER_MAX_BATCH_SIZE = 4096
_KEYWORD_PAD_THRESHOLD = 64
_KEYWORD_PAD_MULTIPLE = 256
_QUERY_PAD_THRESHOLD = 16
_QUERY_PAD_MULTIPLE = 16
_MIN_PAD_MULTIPLE = 4
GRAPHENE_NUM_TOKENS = 2
GRAPHENE_FEATURE_DIM = 128
GRAPHENE_NUM_SCORE_HEADS = 4
DEFAULT_QUERY_CACHE_SIZE = 16384
DEFAULT_KEYWORD_CACHE_SIZE = 16777216
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


def make_fast_rms_norm_gated(norm_mod: Any, eps: float):

    def _fast_norm(hidden_states: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hs_f32 = hidden_states.to(torch.float32)
        variance = hs_f32.pow(2).mean(-1, keepdim=True)
        normed = norm_mod.weight * (hs_f32 * torch.rsqrt(variance + eps)).to(input_dtype)
        if gate is not None:
            normed = normed * F.silu(gate.to(torch.float32))
        return normed.to(input_dtype)

    return _fast_norm


def _make_fast_prepare_batch(engine: Any):
    tokenizer = engine.tokenizer
    max_len = int(engine.max_length)
    pad_id = int(tokenizer.pad_token_id)
    add_emb = bool(getattr(engine, "add_embedding_token", False))
    dev = engine.device

    def _fast_prepare_batch(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        tokenized = tokenizer(texts, padding=False, truncation=True, max_length=max_len)["input_ids"]
        n = len(tokenized)
        ids_np = np.full((n, max_len), pad_id, dtype=np.int64)
        att_np = np.zeros((n, max_len), dtype=np.int64)
        for i, row in enumerate(tokenized):
            if add_emb and row and row[-1] != pad_id:
                row = list(row[: max_len - 1]) + [pad_id]
            else:
                row = row[:max_len]
            k = len(row)
            if k > 0:
                arr = np.asarray(row, dtype=np.int64)
                ids_np[i, max_len - k :] = arr
                att_np[i, max_len - k :] = (arr != pad_id).astype(np.int64)
            if add_emb:
                att_np[i, -1] = 1
        return torch.from_numpy(ids_np).to(device=dev), torch.from_numpy(att_np).to(device=dev)

    return _fast_prepare_batch


def optimize_graphene_pipeline(pipeline: object) -> None:
    for attr in ("query_engine", "keyword_engine"):
        engine: Any = getattr(pipeline, attr, None)
        model = getattr(engine, "model", None)
        if engine is None or model is None:
            continue
        cur_batch_size = getattr(engine, "batch_size", None)
        if isinstance(cur_batch_size, int):
            engine.batch_size = min(cur_batch_size, _TOWER_MAX_BATCH_SIZE)
        engine._prepare_batch = _make_fast_prepare_batch(engine)
        for mod in model.modules():
            if type(mod).__name__ == "Qwen3_5GatedDeltaNet":
                if getattr(mod, "causal_conv1d_fn", None) is None:
                    mod.causal_conv1d_fn = fast_causal_conv1d
                orig_chunk = getattr(mod, "chunk_gated_delta_rule", None)
                if callable(orig_chunk) and getattr(orig_chunk, "__name__", "") != "_fast_rule":
                    mod.chunk_gated_delta_rule = make_fast_single_chunk_gated_delta_rule(orig_chunk)
                norm_mod = getattr(mod, "norm", None)
                if norm_mod is not None and getattr(norm_mod.forward, "__name__", "") != "_fast_norm":
                    norm_mod.forward = make_fast_rms_norm_gated(norm_mod, float(mod.layer_norm_epsilon))


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
        self.storage: torch.Tensor = torch.empty(
            (self.capacity, self.num_tokens, self.feature_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self._slots: OrderedDict[str, int] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def ensure_storage(self, num_tokens: int, feature_dim: int) -> torch.Tensor:
        if self.storage.shape[1:] != (num_tokens, feature_dim):
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
    """Two-level in-VRAM GPU relevance cache: L1 (q, k) score cache + L2 (q_emb, k_emb) GPU tables."""

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
            batch_size = int(engine.batch_size)
            chunks: list[torch.Tensor] = []
            for start in range(0, len(prompts), batch_size):
                token_ids, attention = engine._prepare_batch(prompts[start : start + batch_size])
                chunks.append(engine.model(token_ids, attention).to(dtype=table.dtype))
            encoded = torch.cat(chunks, dim=0)[: len(missing_texts)]
            if encoded.ndim != 3:
                raise RuntimeError("Official relevance model returned invalid GPU tower features")
            storage = table.ensure_storage(encoded.shape[1], encoded.shape[2])
            slot_tensor = torch.tensor(missing_slots, dtype=torch.long).to(device=storage.device)
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
        slots = torch.tensor([q_map[q] for q in unique_queries], dtype=torch.long).to(device=q_storage.device)
        fetched = q_storage[slots].detach().cpu()
        if not bool(torch.isfinite(fetched).all()):
            raise RuntimeError("Official relevance model returned non-finite query features")
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
        q_idx = torch.tensor(q_indices, dtype=torch.long).to(device=device)
        k_idx = torch.tensor(k_slots, dtype=torch.long).to(device=device)
        cross_chunk = max(_CROSS_CHUNK_SIZE, int(self.pipeline.cross_batch_size))
        scores_gpu: list[torch.Tensor] = []
        for start in range(0, len(q_indices), cross_chunk):
            end = start + cross_chunk
            scores_gpu.append(
                self.pipeline.cross_model(q_embeds_gpu[q_idx[start:end]], k_storage[k_idx[start:end]]).float()
            )
        all_scores_cpu = torch.cat(scores_gpu, dim=0).cpu().numpy()
        if (
            all_scores_cpu.shape != (len(q_indices), GRAPHENE_NUM_SCORE_HEADS)
            or not bool(np.isfinite(all_scores_cpu).all())
            or not bool(((all_scores_cpu >= 0.0) & (all_scores_cpu <= 1.0)).all())
        ):
            raise RuntimeError("Official relevance model returned invalid scores")
        return all_scores_cpu[:, 1:3]

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
        for start in range(0, len(missing), self.pair_batch_size):
            chunk = missing[start:start + self.pair_batch_size]
            unique_queries = list(dict.fromkeys(query for query, _ in chunk))
            unique_keywords = list(dict.fromkeys(keyword for _, keyword in chunk))
            phase = time.perf_counter()
            q_storage, q_map, _ = self._encode_gpu_table(unique_queries, query=True)
            query_seconds += time.perf_counter() - phase
            phase = time.perf_counter()
            k_storage, k_map, _ = self._encode_gpu_table(unique_keywords, query=False)
            keyword_seconds += time.perf_counter() - phase
            phase = time.perf_counter()
            pm_sm = self._cross_score_gpu(
                q_storage,
                k_storage,
                [q_map[query] for query, _ in chunk],
                [k_map[keyword] for _, keyword in chunk],
            )
            cross_seconds += time.perf_counter() - phase
            phase = time.perf_counter()
            for pair, row in zip(chunk, pm_sm, strict=True):
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


def _gpu_worker_loop(
    conn: Connection,
    relevance_model: str,
    relevance_source: Path,
    device: str,
    batch_size: int,
    cross_batch_size: int,
    query_cache_size: int,
    keyword_cache_size: int,
    pair_batch_size: int,
) -> None:
    try:
        local_device = "cuda:0" if device.startswith("cuda") else device
        if local_device.startswith("cuda:"):
            torch.cuda.set_device(0)
        factory = load_relevance_pipeline_factory(relevance_source)
        pipeline = factory(
            relevance_model,
            cup="large",
            device=local_device,
            dtype="bf16",
            batch_size=batch_size,
            cross_batch_size=cross_batch_size,
            show_progress=False,
        )
        scorer = RelevanceScorer(
            pipeline,
            feature_cache_size=query_cache_size,
            score_cache_size=1024,
            pair_batch_size=pair_batch_size,
            keyword_cache_size=keyword_cache_size,
        )
        gc.collect()
        gc.freeze()
        gc.set_threshold(100_000, 500, 500)
        conn.send(("ready", None))
        while True:
            msg = conn.recv()
            if msg is None:
                break
            cmd, payload = msg
            if cmd == "encode_queries":
                q_map, q_delta = scorer.encode_queries_cached(payload)
                encoded_u16 = {
                    q: t.contiguous().view(torch.uint16).numpy()
                    for q, t in q_map.items()
                }
                conn.send(("ok", (encoded_u16, q_delta, scorer.query_features.stats())))
            elif cmd == "score_pairs":
                q_u16, keywords, q_indices, k_indices = payload
                q_tensor = torch.from_numpy(q_u16).view(torch.bfloat16)
                scores, k_delta = scorer.score_with_query_embeddings(
                    q_tensor, keywords, q_indices, k_indices,
                )
                conn.send(("ok", (scores, k_delta, scorer.keyword_features.stats())))
            elif cmd == "stats":
                conn.send(("ok", (scorer.query_features.stats(), scorer.keyword_features.stats())))
            elif cmd == "empty_cache":
                torch.cuda.empty_cache()
                conn.send(("ok", None))
            else:
                conn.send(("error", f"Unknown command: {cmd}"))
    except BaseException as exc:
        try:
            conn.send(("error", f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
        raise
    finally:
        conn.close()


class MultiGpuRelevanceScorer:
    """Per-node multi-process GPU relevance scorer: node-wide L1 cache + 8 isolated GPU subprocesses."""

    def __init__(
        self,
        relevance_model: str,
        devices: Sequence[str],
        *,
        relevance_source: str | Path,
        batch_size: int = 8192,
        cross_batch_size: int = _CROSS_CHUNK_SIZE,
        feature_cache_size: int = DEFAULT_QUERY_CACHE_SIZE,
        score_cache_size: int = DEFAULT_SCORE_CACHE_SIZE,
        pair_batch_size: int = DEFAULT_PAIR_BATCH_SIZE,
        keyword_cache_size: int = DEFAULT_KEYWORD_CACHE_SIZE,
    ) -> None:
        if not devices:
            raise ValueError("At least one GPU device is required")
        self.devices = tuple(devices)
        self.num_gpus = len(self.devices)
        self.pair_batch_size = pair_batch_size
        self.scores: LRUCache[tuple[str, str], tuple[float, float]] = LRUCache(
            score_cache_size
        )
        self._cache_lock = Lock()
        per_gpu_q_cache = max(2048, feature_cache_size // self.num_gpus)
        self._q_capacity = per_gpu_q_cache * self.num_gpus
        self._query_emb_cache: LRUCache[str, np.ndarray] = LRUCache(self._q_capacity)
        self._k_capacity = keyword_cache_size * self.num_gpus
        self._q_stats = [
            CacheStats(capacity=per_gpu_q_cache, size=0, hits=0, misses=0, evictions=0)
            for _ in range(self.num_gpus)
        ]
        self._k_stats = [
            CacheStats(capacity=keyword_cache_size, size=0, hits=0, misses=0, evictions=0)
            for _ in range(self.num_gpus)
        ]
        staged_source = _stage_local_relevance_source(Path(relevance_source))
        staged_model = _stage_local_relevance_model(relevance_model)
        ctx = mp.get_context("spawn")
        self._executor = ThreadPoolExecutor(max_workers=self.num_gpus)
        self._conns: list[Connection] = []
        self._procs: list[Any] = []
        self._conn_locks = [Lock() for _ in range(self.num_gpus)]
        saved_vis = {
            k: os.environ.get(k)
            for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")
        }
        try:
            for dev in self.devices:
                if dev.startswith("cuda:"):
                    gpu_id = dev.split(":", 1)[1]
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                    os.environ.pop("HIP_VISIBLE_DEVICES", None)
                    os.environ.pop("NVIDIA_VISIBLE_DEVICES", None)
                    os.environ["ROCR_VISIBLE_DEVICES"] = gpu_id
                parent_conn, child_conn = ctx.Pipe(duplex=True)
                proc = ctx.Process(
                    target=_gpu_worker_loop,
                    args=(
                        child_conn,
                        staged_model,
                        staged_source,
                        dev,
                        batch_size,
                        cross_batch_size,
                        per_gpu_q_cache,
                        keyword_cache_size,
                        pair_batch_size,
                    ),
                    daemon=True,
                )
                proc.start()
                child_conn.close()
                self._conns.append(parent_conn)
                self._procs.append(proc)
        finally:
            for k, v in saved_vis.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        for idx, conn in enumerate(self._conns):
            status, detail = conn.recv()
            if status != "ready":
                self.close()
                raise RuntimeError(f"GPU subprocess {self.devices[idx]} failed to start: {detail}")

    def check_alive(self) -> None:
        for idx, proc in enumerate(self._procs):
            if not proc.is_alive():
                raise RuntimeError(f"GPU subprocess {self.devices[idx]} is not alive")

    def _call_worker(self, shard_idx: int, cmd: str, payload: Any) -> Any:
        with self._conn_locks[shard_idx]:
            proc = self._procs[shard_idx]
            if not proc.is_alive():
                raise RuntimeError(f"GPU subprocess {self.devices[shard_idx]} exited unexpectedly")
            conn = self._conns[shard_idx]
            conn.send((cmd, payload))
            status, result = conn.recv()
        if status != "ok":
            raise RuntimeError(f"GPU subprocess {self.devices[shard_idx]} error: {result}")
        return result

    def query_feature_stats(self) -> CacheStats:
        with self._cache_lock:
            ram_stats = self._query_emb_cache.stats()
            return CacheStats(
                capacity=sum(s.capacity for s in self._q_stats),
                size=max(ram_stats.size, sum(s.size for s in self._q_stats)),
                hits=ram_stats.hits + sum(s.hits for s in self._q_stats),
                misses=sum(s.misses for s in self._q_stats),
                evictions=sum(s.evictions for s in self._q_stats),
            )

    def keyword_feature_stats(self) -> CacheStats:
        with self._cache_lock:
            return CacheStats(
                capacity=sum(s.capacity for s in self._k_stats),
                size=sum(s.size for s in self._k_stats),
                hits=sum(s.hits for s in self._k_stats),
                misses=sum(s.misses for s in self._k_stats),
                evictions=sum(s.evictions for s in self._k_stats),
            )

    def _encode_queries_sharded(
        self, queries: list[str],
    ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        unique_queries = list(dict.fromkeys(queries))
        if not unique_queries:
            return {}, {"hits": 0, "misses": 0, "evictions": 0}
        q_embeds: dict[str, np.ndarray] = {}
        missing_queries: list[str] = []
        ram_hits = 0
        with self._cache_lock:
            for q in unique_queries:
                cached = self._query_emb_cache.get(q)
                if cached is not None:
                    q_embeds[q] = cached
                    ram_hits += 1
                else:
                    missing_queries.append(q)
        if not missing_queries:
            return q_embeds, {"hits": ram_hits, "misses": 0, "evictions": 0}
        buckets: list[list[str]] = [[] for _ in range(self.num_gpus)]
        for q in missing_queries:
            buckets[xxhash.xxh64_intdigest(q) % self.num_gpus].append(q)
        futures = [
            (idx, self._executor.submit(self._call_worker, idx, "encode_queries", bucket))
            for idx, bucket in enumerate(buckets)
            if bucket
        ]
        q_delta_total = {"hits": ram_hits, "misses": 0, "evictions": 0}
        for idx, future in futures:
            encoded_u16, q_delta, q_stat = future.result()
            q_embeds.update(encoded_u16)
            for key in ("hits", "misses", "evictions"):
                q_delta_total[key] += q_delta[key]
            with self._cache_lock:
                self._q_stats[idx] = q_stat
                for q, emb in encoded_u16.items():
                    self._query_emb_cache.put(q, emb)
        return q_embeds, q_delta_total

    def _score_k_shard(
        self,
        shard_idx: int,
        shard_pairs: list[tuple[str, str]],
        query_embeddings: dict[str, np.ndarray],
    ) -> tuple[dict[tuple[str, str], tuple[float, float]], dict[str, int]]:
        q_map: dict[str, int] = {}
        k_map: dict[str, int] = {}
        q_arrays: list[np.ndarray] = []
        keywords: list[str] = []
        q_indices: list[int] = []
        k_indices: list[int] = []
        for q, k in shard_pairs:
            qi = q_map.get(q)
            if qi is None:
                qi = len(q_arrays)
                q_map[q] = qi
                q_arrays.append(query_embeddings[q])
            ki = k_map.get(k)
            if ki is None:
                ki = len(keywords)
                k_map[k] = ki
                keywords.append(k)
            q_indices.append(qi)
            k_indices.append(ki)
        q_u16 = np.stack(q_arrays, axis=0)
        scores, k_delta, k_stat = self._call_worker(
            shard_idx, "score_pairs", (q_u16, keywords, q_indices, k_indices),
        )
        with self._cache_lock:
            self._k_stats[shard_idx] = k_stat
        return dict(zip(shard_pairs, scores, strict=True)), k_delta

    def score_with_deltas(
        self, pairs: list[tuple[str, str]],
    ) -> tuple[dict[tuple[str, str], tuple[float, float]], dict[str, dict[str, int]]]:
        started = time.perf_counter()
        zero_counter = {"hits": 0, "misses": 0, "evictions": 0}
        unique_pairs = list(dict.fromkeys(pairs))
        if not unique_pairs:
            return {}, {
                "baseline": dict(zero_counter),
                "reward": dict(zero_counter),
                "query_features": dict(zero_counter),
                "keyword_features": dict(zero_counter),
            }
        resolved: dict[tuple[str, str], tuple[float, float]] = {}
        missing: list[tuple[str, str]] = []
        r_hits = r_misses = 0
        with self._cache_lock:
            for pair in unique_pairs:
                score = self.scores.get(pair)
                if score is None:
                    missing.append(pair)
                    r_misses += 1
                else:
                    resolved[pair] = score
                    r_hits += 1
        if not missing:
            return resolved, {
                "baseline": dict(zero_counter),
                "reward": {"hits": r_hits, "misses": 0, "evictions": 0},
                "query_features": dict(zero_counter),
                "keyword_features": dict(zero_counter),
            }
        lookup_seconds = time.perf_counter() - started
        phase = time.perf_counter()
        q_embeds, q_delta = self._encode_queries_sharded([q for q, _ in missing])
        query_seconds = time.perf_counter() - phase
        phase = time.perf_counter()
        buckets: list[list[tuple[str, str]]] = [[] for _ in range(self.num_gpus)]
        for pair in missing:
            buckets[xxhash.xxh64_intdigest(pair[1]) % self.num_gpus].append(pair)
        futures = [
            self._executor.submit(self._score_k_shard, idx, bucket, q_embeds)
            for idx, bucket in enumerate(buckets)
            if bucket
        ]
        newly_scored: dict[tuple[str, str], tuple[float, float]] = {}
        k_delta_total = {"hits": 0, "misses": 0, "evictions": 0}
        for future in futures:
            shard_scores, shard_k_delta = future.result()
            newly_scored.update(shard_scores)
            for key in ("hits", "misses", "evictions"):
                k_delta_total[key] += shard_k_delta[key]
        keyword_seconds = time.perf_counter() - phase
        phase = time.perf_counter()
        with self._cache_lock:
            evictions_before = self.scores.stats().evictions
            for pair in missing:
                score = newly_scored[pair]
                self.scores.put(pair, score)
                resolved[pair] = score
            r_evictions = self.scores.stats().evictions - evictions_before
        store_seconds = time.perf_counter() - phase
        logger.info(
            "graphene_score_stage pairs=%d misses=%d gpus=%d lookup_s=%.3f query_s=%.3f "
            "keyword_cross_s=%.3f store_s=%.3f total_s=%.3f",
            len(pairs), len(missing), self.num_gpus, lookup_seconds, query_seconds,
            keyword_seconds, store_seconds, time.perf_counter() - started,
        )
        return resolved, {
            "baseline": dict(zero_counter),
            "reward": {"hits": r_hits, "misses": r_misses, "evictions": r_evictions},
            "query_features": q_delta,
            "keyword_features": k_delta_total,
        }

    def score(self, pairs: list[tuple[str, str]]) -> dict[tuple[str, str], tuple[float, float]]:
        resolved, _ = self.score_with_deltas(pairs)
        return resolved

    def empty_cache(self) -> None:
        futures = [
            self._executor.submit(self._call_worker, idx, "empty_cache", None)
            for idx in range(self.num_gpus)
        ]
        for future in futures:
            future.result()

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        for conn in self._conns:
            try:
                conn.send(None)
                conn.close()
            except Exception:
                pass
        for proc in self._procs:
            if proc.is_alive():
                proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()


def _validate_source(path: Path) -> Path:
    source = absolute_path(path)
    if not (source / _PIPELINE_PATH).is_file():
        raise FileNotFoundError(f"Graphene pipeline is missing from relevance source: {source}")
    commit_path = source / "GIT_COMMIT.txt"
    commit = commit_path.read_text(encoding="utf-8").strip()
    if commit != EXPECTED_GRAPHENE_COMMIT:
        raise ValueError(
            f"Graphene source commit mismatch: expected {EXPECTED_GRAPHENE_COMMIT}, got {commit!r}"
        )
    return source


def relevance_source_path(relevance_model: str | Path) -> Path:
    local_target = Path(tempfile.gettempdir()) / f"gr_relevance_source_{EXPECTED_GRAPHENE_COMMIT[:12]}"
    if local_target.is_dir():
        return _validate_source(local_target)
    cosmos_root = os.environ.get("cosmos_data_path")
    if not cosmos_root or not cosmos_root.strip():
        raise ValueError("cosmos_data_path must be set for the relevance source")
    staged_model = Path(_stage_local_relevance_model(relevance_model))
    pointer = staged_model / "SOURCE_CHECKPOINT.txt"
    source_checkpoint = pointer.read_text(encoding="utf-8").strip()
    _, marker, relative = source_checkpoint.partition("/local/")
    if not marker:
        raise ValueError(f"Relevance source checkpoint has no /local/ prefix: {source_checkpoint}")
    return _validate_source(Path(cosmos_root) / "local" / relative / "code")


_RELEVANCE_SUBDIRS = ("checkpoint-query", "checkpoint-keyword", "checkpoint-cross-large")
_RELEVANCE_FILES = (
    "chat_template.jinja",
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _stage_local_relevance_model(model_dir: str | Path) -> str:
    resolved = absolute_path(model_dir)
    local_root = Path(tempfile.gettempdir())
    if resolved.is_relative_to(local_root):
        return os.fspath(resolved)
    target = local_root / f"gr_relevance_model_{resolved.name}"
    sentinel = target / ".staged_ok"
    if not sentinel.is_file():
        with FileLock(f"{target}.lock"):
            if not sentinel.is_file():
                target.mkdir(parents=True, exist_ok=True)
                src_ptr = resolved / "SOURCE_CHECKPOINT.txt"
                dst_ptr = target / "SOURCE_CHECKPOINT.txt"
                if not dst_ptr.is_file() and src_ptr.is_file():
                    shutil.copyfile(src_ptr, dst_ptr)
                for sub in _RELEVANCE_SUBDIRS:
                    sub_dst = target / sub
                    sub_dst.mkdir(parents=True, exist_ok=True)
                    sub_src = resolved / sub
                    for fname in _RELEVANCE_FILES:
                        f_dst = sub_dst / fname
                        if not f_dst.is_file():
                            f_src = sub_src / fname
                            if f_src.is_file():
                                shutil.copyfile(f_src, f_dst)
                sentinel.write_text(os.fspath(resolved), encoding="utf-8")
    return os.fspath(target)


def _stage_local_relevance_source(source: Path) -> Path:
    resolved = _validate_source(source)
    local_root = Path(tempfile.gettempdir())
    if resolved.is_relative_to(local_root):
        return resolved
    target = local_root / f"gr_relevance_source_{EXPECTED_GRAPHENE_COMMIT[:12]}"
    if not target.exists():
        with FileLock(f"{target}.lock"):
            if not target.exists():
                with tempfile.TemporaryDirectory(prefix=f".gr_relevance_source_{EXPECTED_GRAPHENE_COMMIT[:12]}.", dir=local_root) as tmp:
                    staging = Path(tmp)
                    for metadata_name in ("GIT_COMMIT.txt", "SHA256SUMS", "SOURCE_REPOSITORY.txt"):
                        metadata_src = resolved / metadata_name
                        if metadata_src.is_file():
                            shutil.copyfile(metadata_src, staging / metadata_name)
                    shutil.copytree(
                        resolved / "graphene",
                        staging / "graphene",
                        copy_function=shutil.copyfile,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                    )
                    os.replace(staging, target)
    return _validate_source(target)


def load_relevance_pipeline_factory(source: str | Path) -> RelevancePipelineFactory:
    staged = _stage_local_relevance_source(Path(source))
    source_text = os.fspath(staged)
    if source_text not in sys.path:
        sys.path.append(source_text)
    module = importlib.import_module("graphene.multi_cup.multi_cup_graphene_pipeline")
    factory = getattr(module, "MultiCupGraphenePipeline", None)
    if not callable(factory):
        raise TypeError("Graphene source does not expose MultiCupGraphenePipeline")
    return cast(RelevancePipelineFactory, factory)
