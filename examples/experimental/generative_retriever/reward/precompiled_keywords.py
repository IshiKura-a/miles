"""Node-shared /dev/shm precompiled Graphene keyword table and unified pair-scorer protocol."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Protocol

from filelock import FileLock
import numpy as np
import torch

from examples.experimental.generative_retriever.reward.hbm_keyword_cache import HbmKeywordCache
from examples.experimental.generative_retriever.reward.relevance_model import (
    EXPECTED_GRAPHENE_COMMIT,
    GRAPHENE_FEATURE_DIM,
    GRAPHENE_NUM_SCORE_HEADS,
    GRAPHENE_NUM_TOKENS,
    load_graphene_tower_and_cross_builders,
    optimize_graphene_pipeline,
    relevance_source_path,
    stage_local_relevance_model,
    stage_local_relevance_source,
)
from examples.experimental.generative_retriever.runtime.gr_config import GRConfig
from examples.experimental.generative_retriever.services.lru_cache import LRUCache
from examples.experimental.generative_retriever.services.mapping_staging import resolve_mapping_origin_dir

PRECOMPILED_KW_BIN_NAME = "keyword_features_gv21_large.bf16.bin"
PRECOMPILED_KW_META_NAME = "keyword_features_gv21_large.meta.json"
_QUERY_PREFIX = "Search query: "
_QUERY_PAD_MULTIPLE = 16
_MIN_PAD_MULTIPLE = 4
_CROSS_CHUNK_SIZE = 131_072
_COPY_CHUNK_BYTES = 64 * 1024 * 1024
_SHM_HEADROOM_BYTES = 32 * 1024**3


@dataclass(frozen=True, slots=True)
class BaselineKidSets:
    pm: frozenset[int]
    sm: frozenset[int]


class PackedPairScorer(Protocol):
    """Unified GPU scorer contract for ``(unique_queries, unique_qk_gpu)`` packed pairs."""

    device: torch.device

    @property
    def query_cache_size(self) -> int: ...

    @property
    def keyword_cache_size(self) -> int: ...

    def score_packed_qk_gpu(
        self,
        unique_queries: list[str],
        unique_qk_gpu: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, int]]: ...


def resolve_precompiled_kw_paths(
    staged_map_dir: Path,
    original_map_dir: str | Path,
) -> tuple[Path, Path]:
    """Locate and validate precompiled keyword feature files for the active index."""
    raw_candidates = (
        Path("/scratch/gr_kw_precompile"),
        staged_map_dir.resolve(),
        resolve_mapping_origin_dir(staged_map_dir),
        resolve_mapping_origin_dir(original_map_dir),
    )
    candidates = list(dict.fromkeys(raw_candidates))
    for folder in candidates:
        bin_p = folder / PRECOMPILED_KW_BIN_NAME
        meta_p = folder / PRECOMPILED_KW_META_NAME
        if bin_p.is_file() and meta_p.is_file():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            if meta.get("graphene_commit") != EXPECTED_GRAPHENE_COMMIT:
                raise ValueError(
                    f"Precompiled keyword commit mismatch in {meta_p}: {meta.get('graphene_commit')}"
                )
            expected_bytes = int(meta["num_keywords"]) * GRAPHENE_NUM_TOKENS * GRAPHENE_FEATURE_DIM * 2
            if bin_p.stat().st_size != expected_bytes:
                raise ValueError(
                    f"Precompiled keyword file size mismatch in {bin_p}: "
                    f"expected {expected_bytes}, got {bin_p.stat().st_size}"
                )
            return bin_p, meta_p
    searched = ", ".join(os.fspath(p) for p in candidates)
    raise FileNotFoundError(
        f"Precompiled keyword table ({PRECOMPILED_KW_BIN_NAME}, {PRECOMPILED_KW_META_NAME}) "
        f"not found in any candidate directory: {searched}"
    )


def stage_precompiled_keywords_to_shm(bin_path: Path, meta_path: Path) -> Path:
    """Copy the validated bf16 keyword feature table into node-shared /dev/shm if space permits."""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_bytes = int(meta["num_keywords"]) * int(meta["num_tokens"]) * int(meta["feature_dim"]) * 2
    if bin_path.stat().st_size != expected_bytes:
        raise ValueError(f"Precompiled keyword file size mismatch: {bin_path}")
    tag = str(meta.get("mapping_name", "part25"))
    shm_root = Path("/dev/shm")
    if not shm_root.is_dir():
        return bin_path
    shm_dst = shm_root / f"gr_precompiled_kw_{tag}_gv21.bf16.bin"
    with FileLock(str(shm_root / f"gr_precompiled_kw_{tag}_gv21.lock")):
        if shm_dst.is_file() and shm_dst.stat().st_size == expected_bytes:
            return shm_dst
        if shutil.disk_usage(shm_root).free >= expected_bytes + _SHM_HEADROOM_BYTES:
            tmp_dst = shm_dst.with_suffix(f".tmp.{os.getpid()}")
            try:
                with open(bin_path, "rb") as src_f, open(tmp_dst, "wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f, length=_COPY_CHUNK_BYTES)
                tmp_dst.replace(shm_dst)
                return shm_dst
            finally:
                tmp_dst.unlink(missing_ok=True)
    return bin_path


def ensure_precompiled_keywords_staged(staged_map_dir: Path, original_map_dir: str | Path) -> Path:
    bin_path, meta_path = resolve_precompiled_kw_paths(staged_map_dir, original_map_dir)
    return stage_precompiled_keywords_to_shm(bin_path, meta_path)


class PrecompiledKeywordStore:
    """Read-only node-shared /dev/shm (or local NVMe) bf16 keyword feature table."""

    def __init__(self, bin_path: Path, meta_path: Path, *, stage_to_shm: bool = True) -> None:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.num_keywords = int(meta["num_keywords"])
        self.num_tokens = int(meta["num_tokens"])
        self.feature_dim = int(meta["feature_dim"])
        expected_bytes = self.num_keywords * self.num_tokens * self.feature_dim * 2
        if bin_path.stat().st_size != expected_bytes:
            raise ValueError(f"Precompiled keyword file size mismatch: {bin_path}")

        self.tag = str(meta.get("mapping_name", "part25"))
        self.active_path = stage_precompiled_keywords_to_shm(bin_path, meta_path) if stage_to_shm else bin_path
        self._mmap = np.memmap(
            self.active_path,
            dtype=np.uint16,
            mode="r",
            shape=(self.num_keywords, self.num_tokens, self.feature_dim),
        )

    @property
    def mmap_u16(self) -> np.memmap:
        return self._mmap


class FastPrecompileScorer:
    """Graphene scorer loading only query_engine + cross_model backed by precompiled keyword features."""

    def __init__(
        self,
        relevance_model_path: str,
        kw_store: PrecompiledKeywordStore,
        device: torch.device,
        *,
        gr_config: GRConfig,
        kw_degrees: np.ndarray | None = None,
    ) -> None:
        rel_src = stage_local_relevance_source(relevance_source_path(relevance_model_path))
        rel_model = stage_local_relevance_model(relevance_model_path)
        load_tower, resolve_dtype, cup_model_classes = load_graphene_tower_and_cross_builders(rel_src)

        dev_str = f"cuda:{device.index or 0}" if device.type == "cuda" else str(device)
        self.device = torch.device(dev_str)
        self.dtype = resolve_dtype("bf16")
        self.cross_batch_size = _CROSS_CHUNK_SIZE
        self.query_engine = load_tower(rel_model, "query", 24, self.device, 8192, self.dtype)
        self.keyword_engine = None
        cross_path = os.path.join(rel_model, "checkpoint-cross-large")
        self.cross_model = (
            cup_model_classes["large"]
            .from_pretrained(cross_path, torch_dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        optimize_graphene_pipeline(self)
        self.kw_store = kw_store
        self.hbm_cache = HbmKeywordCache(
            mmap_u16=kw_store.mmap_u16,
            num_keywords=kw_store.num_keywords,
            num_tokens=kw_store.num_tokens,
            feature_dim=kw_store.feature_dim,
            device=self.device,
            dtype=self.dtype,
            kw_degrees=kw_degrees,
            tag=kw_store.tag,
            enabled=gr_config.hbm_kw_cache_enabled,
            reserve_gib=gr_config.hbm_kw_reserve_gib,
            max_gib=gr_config.hbm_kw_max_gib,
            explicit_gib=gr_config.hbm_kw_explicit_gib,
        )
        self._query_cache: LRUCache[str, torch.Tensor] = LRUCache(65_536)

    @property
    def query_cache_size(self) -> int:
        return int(self._query_cache.stats().size)

    @property
    def keyword_cache_size(self) -> int:
        return int(self.kw_store.num_keywords)

    @torch.no_grad()
    def _encode_queries(self, unique_queries: list[str]) -> tuple[torch.Tensor, int, int]:
        if not unique_queries:
            empty = torch.empty((0, GRAPHENE_NUM_TOKENS, GRAPHENE_FEATURE_DIM), device=self.device, dtype=self.dtype)
            return empty, 0, 0
        hits = 0
        missing_q: list[str] = []
        missing_pos: list[int] = []
        out_gpu = torch.empty(
            (len(unique_queries), GRAPHENE_NUM_TOKENS, GRAPHENE_FEATURE_DIM),
            device=self.device,
            dtype=self.dtype,
        )
        for idx, q in enumerate(unique_queries):
            cached = self._query_cache.get(q)
            if cached is not None:
                out_gpu[idx] = cached
                hits += 1
            else:
                missing_q.append(q)
                missing_pos.append(idx)
        misses = len(missing_q)
        if missing_q:
            engine = self.query_engine
            pad_mult = _QUERY_PAD_MULTIPLE if misses >= _QUERY_PAD_MULTIPLE else _MIN_PAD_MULTIPLE
            padded = missing_q + [missing_q[-1]] * (-misses % pad_mult)
            prompts = [_QUERY_PREFIX + q for q in padded]
            bs = int(engine.batch_size)
            chunks = [
                engine.model(*engine._prepare_batch(prompts[start : start + bs])).to(dtype=self.dtype)
                for start in range(0, len(prompts), bs)
            ]
            encoded = torch.cat(chunks, dim=0)[:misses]
            for local_i, q_pos in enumerate(missing_pos):
                vec = encoded[local_i]
                out_gpu[q_pos] = vec
                self._query_cache.put(unique_queries[q_pos], vec)
        return out_gpu, hits, misses

    @torch.no_grad()
    def score_packed_qk_gpu(
        self,
        unique_queries: list[str],
        unique_qk_gpu: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, int]]:
        n_pairs = int(unique_qk_gpu.numel())
        if n_pairs == 0:
            return (
                torch.empty((0, 2), dtype=torch.float32, device=self.device),
                {"q_hits": 0, "q_misses": 0, "k_hits": 0},
            )
        q_gpu, q_hits, q_misses = self._encode_queries(unique_queries)
        q_idx_t = unique_qk_gpu >> 32
        raw_kids_t = unique_qk_gpu & 0xFFFFFFFF
        unique_kids_t, k_inv_t = torch.unique(raw_kids_t, sorted=True, return_inverse=True)
        k_gpu = self.hbm_cache.gather(unique_kids_t)

        chunk_sz = int(self.cross_batch_size)
        if n_pairs <= chunk_sz:
            all_scores = self.cross_model(q_gpu[q_idx_t], k_gpu[k_inv_t]).float()
        else:
            scores_gpu = [
                self.cross_model(q_gpu[q_idx_t[st : st + chunk_sz]], k_gpu[k_inv_t[st : st + chunk_sz]]).float()
                for st in range(0, n_pairs, chunk_sz)
            ]
            all_scores = torch.cat(scores_gpu, dim=0)
        valid_ok = (
            all_scores.shape == (n_pairs, GRAPHENE_NUM_SCORE_HEADS)
            and bool(torch.isfinite(all_scores).all().item())
            and bool(((all_scores >= 0.0) & (all_scores <= 1.0)).all().item())
        )
        if not valid_ok:
            raise RuntimeError("Official relevance model returned invalid scores")
        return all_scores[:, 1:3], {
            "q_hits": q_hits,
            "q_misses": q_misses,
            "k_hits": int(unique_kids_t.numel()),
        }

