"""Resolve the source snapshot shipped with a Graphene relevance checkpoint."""

from __future__ import annotations

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

from examples.experimental.generative_retriever.services.lru_cache import LRUCache

EXPECTED_GRAPHENE_COMMIT = "a57cb597de0891c1e0a2612d12366188b96aa0cc"
_PIPELINE_PATH = Path("graphene/multi_cup/multi_cup_graphene_pipeline.py")
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
            engine.batch_size = min(cur_batch_size, 1024)
        for mod in model.modules():
            if type(mod).__name__ == "Qwen3_5GatedDeltaNet":
                if getattr(mod, "causal_conv1d_fn", None) is None:
                    mod.causal_conv1d_fn = fast_causal_conv1d
                orig_chunk = getattr(mod, "chunk_gated_delta_rule", None)
                if callable(orig_chunk) and getattr(orig_chunk, "__name__", "") != "_fast_rule":
                    mod.chunk_gated_delta_rule = make_fast_single_chunk_gated_delta_rule(orig_chunk)


class RelevancePipeline(Protocol):
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


class RelevanceScorer:
    """Batch distinct tower inputs; keep active results independent of LRU eviction.

    The caller serializes access to the pipeline and caches.
    """

    def __init__(
        self,
        pipeline: RelevancePipeline,
        *,
        feature_cache_size: int = 8192,
        score_cache_size: int = 131072,
        pair_batch_size: int = 4096,
        keyword_cache_size: int = 8192,
    ) -> None:
        if pair_batch_size <= 0:
            raise ValueError("Relevance pair batch size must be positive")
        optimize_graphene_pipeline(pipeline)
        self.pipeline = pipeline
        self.pair_batch_size = pair_batch_size
        self.query_features: LRUCache[str, np.ndarray] = LRUCache(feature_cache_size)
        self.keyword_features: LRUCache[str, np.ndarray] = LRUCache(keyword_cache_size)
        self.scores: LRUCache[tuple[str, str], tuple[float, float]] = LRUCache(
            score_cache_size
        )

    def _encode(
        self, texts: list[str], *, query: bool,
    ) -> dict[str, np.ndarray]:
        cache = self.query_features if query else self.keyword_features
        resolved = {}
        missing = []
        for text in dict.fromkeys(texts):
            features = cache.get(text)
            if features is None:
                missing.append(text)
            else:
                resolved[text] = features
        if missing:
            encode = self.pipeline.encode_query if query else self.pipeline.encode_keyword
            # The tower changes a row's output for batches of 1-3; coarse bucketing avoids MIOpen conv1d retuning.
            pad_multiple = 256 if (not query and len(missing) >= 64) else (16 if (query and len(missing) >= 16) else 4)
            padded = missing + [missing[-1]] * (-len(missing) % pad_multiple)
            encoded = encode(padded)
            if (
                encoded.ndim != 3
                or encoded.shape[0] != len(padded)
                or not np.isfinite(encoded).all()
            ):
                raise RuntimeError("Official relevance model returned invalid tower features")
            for index, text in enumerate(missing):
                # Copy so a cached row does not retain the entire encoded batch.
                features = encoded[index:index + 1].copy()
                cache.put(text, features)
                resolved[text] = features
        return resolved

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
            phase = time.perf_counter()
            queries = self._encode([query for query, _ in chunk], query=True)
            query_seconds += time.perf_counter() - phase
            phase = time.perf_counter()
            keywords = self._encode([keyword for _, keyword in chunk], query=False)
            keyword_seconds += time.perf_counter() - phase
            phase = time.perf_counter()
            scores = self.pipeline.cross_score(
                np.concatenate([queries[query] for query, _ in chunk], axis=0),
                np.concatenate([keywords[keyword] for _, keyword in chunk], axis=0),
            )
            cross_seconds += time.perf_counter() - phase
            if (
                scores.shape != (len(chunk), 4)
                or not np.isfinite(scores).all()
                or not ((scores >= 0) & (scores <= 1)).all()
            ):
                raise RuntimeError("Official relevance model returned invalid scores")
            phase = time.perf_counter()
            for pair, row in zip(chunk, scores, strict=True):
                score = (float(row[1]), float(row[2]))
                self.scores.put(pair, score)
                resolved[pair] = score
            store_seconds += time.perf_counter() - phase
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
