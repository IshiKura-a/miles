"""Resident official doc-ID to keyword-list service for the unified columnar index.

Run:
    python -m examples.experimental.generative_retriever.services.mapping_service \
        --mapping /path/to/slmgrv3_part25_na_intl_20261028_20261029 --port 30488
"""

from __future__ import annotations

import atexit
import threading

from examples.experimental.generative_retriever.utils.paths import absolute_path, symlink_target

import argparse
import gc
import logging
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from threading import RLock
from typing import Annotated, Any
from collections.abc import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, StrictInt, field_validator

from examples.experimental.generative_retriever.index.id_text_mapping import (
    PositionalID,
    is_columnar_index_dir,
    pack_docid_u32,
    validate_docid,
)
from examples.experimental.generative_retriever.reward.relevance_reward import DOC2K_PRUNE_FLOOR, DOCID_MAX_KEYWORDS

logger = logging.getLogger(__name__)
MAPPING_SERVICE_API_VERSION = 3
_SERVICE_API_VERSION = MAPPING_SERVICE_API_VERSION
_NA_LOCALE_ALIASES = frozenset({"NA", "EN-US", "EN-CA", "ES-US", "FR-CA"})


class LookupRequest(BaseModel):
    docids: Annotated[list[tuple[StrictInt, StrictInt, StrictInt, StrictInt]], Field(min_length=1, max_length=4096)]
    market: str | None = None
    use_initial: bool = False

    @field_validator("docids")
    @classmethod
    def validate_docids(cls, values: list[PositionalID]) -> list[PositionalID]:
        for value in values:
            validate_docid(value)
        return values


class KeywordResult(BaseModel):
    docid: PositionalID
    keywords: list[str]
    status: str


class LookupResponse(BaseModel):
    mapping_version: str
    results: list[KeywordResult]


class DocidKeywordPair(BaseModel):
    docid: tuple[StrictInt, StrictInt, StrictInt, StrictInt]
    keyword_ids: Annotated[list[StrictInt], Field(min_length=1)]

    @field_validator("docid")
    @classmethod
    def validate_docid(cls, value: PositionalID) -> PositionalID:
        validate_docid(value)
        return value


class DocidKeywordTextPair(BaseModel):
    docid: tuple[StrictInt, StrictInt, StrictInt, StrictInt]
    keywords: Annotated[list[str], Field(min_length=1)]

    @field_validator("docid")
    @classmethod
    def validate_docid(cls, value: PositionalID) -> PositionalID:
        validate_docid(value)
        return value


class UpdateMappingRequest(BaseModel):
    entries: Annotated[list[DocidKeywordPair], Field(min_length=1)]
    replace_all: bool = False


class DeltaMappingRequest(BaseModel):
    additions: list[DocidKeywordTextPair] = Field(default_factory=list)
    removals: list[DocidKeywordTextPair] = Field(default_factory=list)
    k_max: int = DOCID_MAX_KEYWORDS


class DeltaMappingIdsRequest(BaseModel):
    additions: list[DocidKeywordPair] = Field(default_factory=list)
    removals: list[DocidKeywordPair] = Field(default_factory=list)
    k_max: int = DOCID_MAX_KEYWORDS


class ResolveKeywordsRequest(BaseModel):
    keywords: list[str]


class ClaimMappingRequest(BaseModel):
    owner: str
    checkpoint: str | None = None


class SaveMappingRequest(BaseModel):
    output_dir: str


class ResidentMapping:
    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        if not is_columnar_index_dir(path):
            raise ValueError(f"Mapping path must be a columnar index directory: {path}")
        self.path = path
        self.read_only = read_only
        self.version = (
            symlink_target(path / "keywords.parquet").parent.name
            if path.name == "mapping" and (path / "keywords.parquet").is_symlink()
            else path.name
        )
        self._lock = RLock()
        self._owner: str | None = None
        self._active_requests = 0
        self._overlay_kw_ids: dict[int, tuple[int, ...]] = {}
        self._docid_kw_cache: dict[int, list[str]] = {}
        self._text_to_kid: dict[str, int] = {}
        self._decoded_kws: dict[int, str] = {}
        self._kw_array: Any = None
        self._in_na_array: np.ndarray = np.empty(0, dtype=np.bool_)
        self._markets_array: Any = None
        self._unique_docids_u32: np.ndarray = np.empty(0, dtype=np.uint32)
        self._csr_offsets: np.ndarray = np.empty(0, dtype=np.int64)
        self._csr_kw_ids: np.ndarray = np.empty(0, dtype=np.uint32)
        self._kw_degree: np.ndarray = np.empty(0, dtype=np.int32)
        self._async_save_lock = threading.Lock()
        self._async_save_thread: threading.Thread | None = None
        atexit.register(self.flush_async_saves)
        self._load_columnar(path)

    def _load_columnar(self, directory: Path) -> None:
        logger.info("Loading columnar index into memory: %s", directory)
        kw_path = directory / "keywords.parquet"
        pairs_path = directory / "docid_keywords.parquet"
        if self.read_only:
            ipc_path = directory / "keywords_only.arrow"
            if ipc_path.is_file():
                mmap = pa.memory_map(str(ipc_path), "r")
                kw_table = pa.ipc.open_file(mmap).read_all()
            else:
                kw_table = pq.read_table(kw_path, columns=["keyword"], memory_map=True)
            self._kw_array = kw_table.column("keyword")
            n_kw = len(self._kw_array)
        else:
            kw_table = pq.read_table(
                kw_path, columns=["keyword_id", "keyword", "in_na", "markets"], memory_map=True,
            )
            self._kw_array = kw_table.column("keyword").combine_chunks()
            n_kw = len(self._kw_array)
            if n_kw > 0:
                kid_col = kw_table.column("keyword_id")
                if int(kid_col[0].as_py()) != 0 or int(kid_col[n_kw - 1].as_py()) != n_kw - 1:
                    raise ValueError("keywords.parquet keyword_id column must be contiguous 0..N-1")
            self._in_na_array = kw_table.column("in_na").to_numpy(zero_copy_only=False)
            self._markets_array = kw_table.column("markets").combine_chunks()

        u_path = directory / "docid_unique.u32.npy"
        off_path = directory / "docid_offsets.i64.npy"
        kid_path = directory / "docid_kw_ids.u32.npy"
        self._unique_docids_u32 = np.load(u_path, mmap_mode="r")
        self._csr_offsets = np.load(off_path, mmap_mode="r")
        self._csr_kw_ids = np.load(kid_path, mmap_mode="r")

        self.rows = int(len(self._csr_kw_ids))
        if not self.read_only:
            self._kw_degree = np.bincount(self._csr_kw_ids, minlength=n_kw).astype(np.int32)
        self.byte_count = int(kw_path.stat().st_size + pairs_path.stat().st_size)
        logger.info(
            "Columnar mapping ready: %d IDs, %d keywords, %d pairs",
            len(self._unique_docids_u32),
            n_kw,
            self.rows,
        )


    @property
    def id_count(self) -> int:
        with self._lock:
            return int(len(self._unique_docids_u32))

    def _market_allows(self, kw_id: int, target_market_upper: str) -> bool:
        if target_market_upper in _NA_LOCALE_ALIASES and bool(self._in_na_array[kw_id]):
            return True
        mkt_str: str = self._markets_array[kw_id].as_py()
        if not mkt_str:
            return False
        if mkt_str == target_market_upper:
            return True
        n = len(target_market_upper)
        pos = mkt_str.find(target_market_upper)
        while pos != -1:
            left_ok = pos == 0 or mkt_str[pos - 1] == ","
            right_ok = pos + n == len(mkt_str) or mkt_str[pos + n] == ","
            if left_ok and right_ok:
                return True
            pos = mkt_str.find(target_market_upper, pos + 1)
        return False

    def _base_kw_ids_one(self, target_u32: int) -> tuple[int, ...] | None:
        idx = int(np.searchsorted(self._unique_docids_u32, np.uint32(target_u32)))
        n_unique = len(self._unique_docids_u32)
        if idx >= n_unique or int(self._unique_docids_u32[idx]) != int(target_u32):
            return None
        start = int(self._csr_offsets[idx])
        end = int(self._csr_offsets[idx + 1])
        return tuple(int(k) for k in self._csr_kw_ids[start:end])

    def _decode_kid(self, kid: int) -> str:
        cached = self._decoded_kws.get(kid)
        if cached is not None:
            return cached
        raw_bytes: bytes = self._kw_array[kid].as_py()
        text = raw_bytes.decode("utf-8")
        self._decoded_kws[kid] = text
        if not self.read_only:
            self._text_to_kid[text] = kid
        return text

    @contextmanager
    def access(self, owner: str | None) -> Iterator[None]:
        with self._lock:
            if owner != self._owner:
                raise ValueError("Mapping service is owned by another run")
            self._active_requests += 1
        try:
            yield
        finally:
            with self._lock:
                self._active_requests -= 1

    def replace_overlay(self, overlay: dict[int, tuple[int, ...]]) -> None:
        for docid in self._overlay_kw_ids.keys() | overlay.keys():
            base = self._base_kw_ids_one(docid) or ()
            previous = set(self._overlay_kw_ids.get(docid, base))
            current = set(overlay.get(docid, base))
            for kid in previous - current:
                self._kw_degree[kid] -= 1
            for kid in current - previous:
                self._kw_degree[kid] += 1
        self._overlay_kw_ids = overlay
        self._docid_kw_cache.clear()

    def lookup_u32(
        self,
        packed_u32: np.ndarray,
        *,
        use_initial: bool = False,
    ) -> dict[int, list[str]]:
        """Fast vectorized lookup from packed uint32 docids to deduplicated keyword lists."""
        u32_list: list[int] = packed_u32.tolist()
        out: dict[int, list[str]] = {}
        overlay_map = self._overlay_kw_ids
        cache = self._docid_kw_cache
        missing_u32: list[int] = []
        for u32_int in u32_list:
            if not use_initial and u32_int in overlay_map:
                cached = cache.get(u32_int)
                if cached is not None:
                    out[u32_int] = cached
                else:
                    kws = [self._decode_kid(int(k)) for k in dict.fromkeys(overlay_map[u32_int])]
                    cache[u32_int] = kws
                    out[u32_int] = kws
            elif not use_initial or u32_int not in overlay_map:
                cached = cache.get(u32_int)
                if cached is not None:
                    out[u32_int] = cached
                else:
                    missing_u32.append(u32_int)
            else:
                missing_u32.append(u32_int)
        if missing_u32:
            missing_arr = np.asarray(missing_u32, dtype=np.uint32)
            positions = np.searchsorted(self._unique_docids_u32, missing_arr)
            n_unique = len(self._unique_docids_u32)
            for u32_int, pos in zip(missing_u32, positions.tolist(), strict=True):
                idx = int(pos)
                if idx >= n_unique or int(self._unique_docids_u32[idx]) != u32_int:
                    raise LookupError(f"Official mapping packed u32 ID not found: {u32_int}")
                start = int(self._csr_offsets[idx])
                end = int(self._csr_offsets[idx + 1])
                kws = [self._decode_kid(int(k)) for k in dict.fromkeys(int(x) for x in self._csr_kw_ids[start:end])]
                if not use_initial or u32_int not in overlay_map:
                    cache[u32_int] = kws
                out[u32_int] = kws
        return out

    def lookup_u32_with_kids(
        self,
        packed_u32: np.ndarray,
    ) -> tuple[dict[int, list[str]], dict[int, tuple[int, ...]]]:
        u32_list: list[int] = packed_u32.tolist()
        out_kws: dict[int, list[str]] = {}
        out_kids: dict[int, tuple[int, ...]] = {}
        overlay_map = self._overlay_kw_ids
        cache = self._docid_kw_cache
        missing_u32: list[int] = []
        for u32_int in u32_list:
            ov = overlay_map.get(u32_int)
            if ov is not None:
                dedup_ov = tuple(dict.fromkeys(int(k) for k in ov))
                out_kids[u32_int] = dedup_ov
                cached = cache.get(u32_int)
                if cached is not None and len(cached) == len(dedup_ov):
                    out_kws[u32_int] = cached
                else:
                    kws = [self._decode_kid(k) for k in dedup_ov]
                    cache[u32_int] = kws
                    out_kws[u32_int] = kws
            else:
                missing_u32.append(u32_int)
        if missing_u32:
            missing_arr = np.asarray(missing_u32, dtype=np.uint32)
            positions = np.searchsorted(self._unique_docids_u32, missing_arr)
            n_unique = len(self._unique_docids_u32)
            for u32_int, pos in zip(missing_u32, positions.tolist(), strict=True):
                idx = int(pos)
                if idx >= n_unique or int(self._unique_docids_u32[idx]) != u32_int:
                    raise LookupError(f"Official mapping packed u32 ID not found: {u32_int}")
                start = int(self._csr_offsets[idx])
                end = int(self._csr_offsets[idx + 1])
                raw_kids = tuple(dict.fromkeys(int(k) for k in self._csr_kw_ids[start:end]))
                out_kids[u32_int] = raw_kids
                cached = cache.get(u32_int)
                if cached is not None and len(cached) == len(raw_kids):
                    out_kws[u32_int] = cached
                else:
                    kws = [self._decode_kid(k) for k in raw_kids]
                    cache[u32_int] = kws
                    out_kws[u32_int] = kws
        return out_kws, out_kids

    def resolve_kid_in_docid(self, u32: int, keyword: str) -> int | None:
        kids = self._overlay_kw_ids.get(u32)
        if kids is None:
            kids = self._base_kw_ids_one(u32)
            if not kids:
                return None
        for raw_kid in kids:
            kid = int(raw_kid)
            if self._decode_kid(kid) == keyword:
                return kid
        return None

    def apply_u32_kid_delta(
        self,
        additions: dict[int, list[int]],
        removals: dict[int, set[int]],
        *,
        k_max: int = DOCID_MAX_KEYWORDS,
    ) -> dict[str, int]:
        updated_u32: set[int] = set()
        added_edges = 0
        removed_edges = 0
        for u32, rem_kids in removals.items():
            if not rem_kids:
                continue
            current_kids = self._overlay_kw_ids.get(u32)
            if current_kids is None:
                current_kids = self._base_kw_ids_one(u32)
                if not current_kids:
                    continue
            if len(current_kids) <= DOC2K_PRUNE_FLOOR:
                continue
            filtered_list: list[int] = []
            for idx_k, kid in enumerate(current_kids):
                remaining_slots = len(current_kids) - idx_k
                if (len(filtered_list) + remaining_slots) > DOC2K_PRUNE_FLOOR and kid in rem_kids:
                    removed_edges += 1
                else:
                    filtered_list.append(kid)
            if len(filtered_list) < len(current_kids):
                self._overlay_kw_ids[u32] = tuple(filtered_list)
                updated_u32.add(u32)
        for u32, add_kids in additions.items():
            if not add_kids:
                continue
            current_kids = self._overlay_kw_ids.get(u32)
            if current_kids is None:
                current_kids = self._base_kw_ids_one(u32)
                if not current_kids:
                    continue
            kid_list = list(current_kids)
            existing_kids = set(kid_list)
            changed = False
            for kid in add_kids:
                if len(kid_list) >= k_max:
                    break
                if kid not in existing_kids:
                    kid_list.append(kid)
                    existing_kids.add(kid)
                    added_edges += 1
                    changed = True
            if changed:
                self._overlay_kw_ids[u32] = tuple(kid_list)
                updated_u32.add(u32)
        for u32 in updated_u32:
            self._docid_kw_cache.pop(u32, None)
        return {
            "updated_docids": len(updated_u32),
            "added_edges": added_edges,
            "removed_edges": removed_edges,
            "overlay_docids": len(self._overlay_kw_ids),
        }

    def apply_delta(self, request: DeltaMappingRequest) -> dict[str, int]:
        def ids(entries: list[DocidKeywordTextPair]) -> list[DocidKeywordPair]:
            with self._lock:
                result = []
                for entry in entries:
                    kids = [self._text_to_kid[kw] for kw in entry.keywords if kw in self._text_to_kid]
                    if kids:
                        result.append(DocidKeywordPair(docid=entry.docid, keyword_ids=kids))
                return result

        return self.apply_id_delta(DeltaMappingIdsRequest(
            additions=ids(request.additions), removals=ids(request.removals), k_max=request.k_max,
        ))

    def apply_id_delta(self, request: DeltaMappingIdsRequest) -> dict[str, int]:
        if request.k_max < 1:
            raise ValueError("k_max must be at least 1")
        with self._lock:
            updated_u32: set[int] = set()
            added_edges = 0
            removed_edges = 0
            for item in request.removals:
                u32 = pack_docid_u32(item.docid)
                current_kids = self._overlay_kw_ids.get(u32)
                if current_kids is None:
                    current_kids = self._base_kw_ids_one(u32)
                    if not current_kids:
                        continue
                remove_kids = set(item.keyword_ids)
                if not remove_kids or len(current_kids) <= DOC2K_PRUNE_FLOOR:
                    continue
                filtered_list: list[int] = []
                has_deg = len(self._kw_degree) > 0
                for idx_k, kid in enumerate(current_kids):
                    remaining_slots = len(current_kids) - idx_k
                    can_remove = (len(filtered_list) + remaining_slots) > DOC2K_PRUNE_FLOOR
                    if (
                        can_remove
                        and kid in remove_kids
                        and (not has_deg or int(self._kw_degree[kid]) > 1)
                    ):
                        if has_deg:
                            self._kw_degree[kid] -= 1
                        removed_edges += 1
                    else:
                        filtered_list.append(kid)
                if len(filtered_list) < len(current_kids):
                    self._overlay_kw_ids[u32] = tuple(filtered_list)
                    updated_u32.add(u32)
            for item in request.additions:
                u32 = pack_docid_u32(item.docid)
                current_kids = self._overlay_kw_ids.get(u32)
                if current_kids is None:
                    current_kids = self._base_kw_ids_one(u32)
                    if not current_kids:
                        continue
                kid_list = list(current_kids)
                existing_kids = set(kid_list)
                changed = False
                for kid in item.keyword_ids:
                    if len(kid_list) >= request.k_max:
                        break
                    if kid not in existing_kids:
                        kid_list.append(kid)
                        existing_kids.add(kid)
                        if len(self._kw_degree) > 0:
                            self._kw_degree[kid] += 1
                        added_edges += 1
                        changed = True
                if changed:
                    self._overlay_kw_ids[u32] = tuple(kid_list)
                    updated_u32.add(u32)
            for u32 in updated_u32:
                self._docid_kw_cache.pop(u32, None)
            return {
                "updated_docids": len(updated_u32),
                "added_edges": added_edges,
                "removed_edges": removed_edges,
                "overlay_docids": len(self._overlay_kw_ids),
            }

    def update_docid_mappings(
        self,
        entries: list[DocidKeywordPair],
        *,
        replace_all: bool = False,
    ) -> dict[str, int]:
        with self._lock:
            max_kw_id = len(self._kw_array)
            overlay = {} if replace_all else dict(self._overlay_kw_ids)
            for entry in entries:
                u32 = pack_docid_u32(entry.docid)
                dedup_kids: list[int] = []
                seen: set[int] = set()
                for kid in entry.keyword_ids:
                    if not 0 <= kid < max_kw_id:
                        raise ValueError(f"keyword_id {kid} out of range [0, {max_kw_id})")
                    if kid not in seen:
                        dedup_kids.append(int(kid))
                        seen.add(int(kid))
                overlay[u32] = tuple(dedup_kids)
            self.replace_overlay(overlay)
            return {
                "id_count": int(len(self._unique_docids_u32)),
                "overlay_docids": len(self._overlay_kw_ids),
            }

    def flush_async_saves(self) -> None:
        with self._async_save_lock:
            thread = self._async_save_thread
            self._async_save_thread = None
        if thread is not None and thread.is_alive():
            thread.join()

    def save_docid_mappings(
        self,
        output_dir: str | Path,
        *,
        stage_dir: str | Path | None = None,
        async_transfer: bool = True,
    ) -> dict[str, Any]:
        import os
        import shutil
        import tempfile
        import time

        self.flush_async_saves()
        with self._lock:
            overlay_snapshot = dict(self._overlay_kw_ids)
        target_dir = absolute_path(output_dir)
        raw_base = os.environ.get("GR_ID_MAPPING")
        base_dir = Path(raw_base).resolve() if raw_base and Path(raw_base).is_dir() else self.path
        unique_docids = self._unique_docids_u32
        n_unique = len(unique_docids)
        counts = np.diff(self._csr_offsets).copy()
        if overlay_snapshot:
            sorted_ov = sorted(overlay_snapshot.items())
            ov_u32 = np.asarray([u for u, _ in sorted_ov], dtype=np.uint32)
            ov_pos = np.searchsorted(unique_docids, ov_u32)
            valid_items: list[tuple[int, np.ndarray]] = []
            for (u32, kids), pos in zip(sorted_ov, ov_pos.tolist(), strict=True):
                idx_pos = int(pos)
                if idx_pos < n_unique and int(unique_docids[idx_pos]) == int(u32):
                    dedup_k = np.asarray(list(dict.fromkeys(int(k) for k in kids)), dtype=np.uint32)
                    counts[idx_pos] = len(dedup_k)
                    valid_items.append((idx_pos, dedup_k))
            csr_offsets = np.empty(n_unique + 1, dtype=np.int64)
            csr_offsets[0] = 0
            np.cumsum(counts, out=csr_offsets[1:])
            total_rows = int(csr_offsets[-1])
            csr_kw_ids = np.empty(total_rows, dtype=np.uint32)
            prev_idx = 0
            for idx_pos, dedup_k in valid_items:
                if idx_pos > prev_idx:
                    old_s = int(self._csr_offsets[prev_idx])
                    old_e = int(self._csr_offsets[idx_pos])
                    new_s = int(csr_offsets[prev_idx])
                    csr_kw_ids[new_s : new_s + (old_e - old_s)] = self._csr_kw_ids[old_s:old_e]
                new_s = int(csr_offsets[idx_pos])
                csr_kw_ids[new_s : new_s + len(dedup_k)] = dedup_k
                prev_idx = idx_pos + 1
            if prev_idx < n_unique:
                old_s = int(self._csr_offsets[prev_idx])
                old_e = int(self._csr_offsets[n_unique])
                new_s = int(csr_offsets[prev_idx])
                csr_kw_ids[new_s : new_s + (old_e - old_s)] = self._csr_kw_ids[old_s:old_e]
        else:
            csr_offsets = self._csr_offsets
            csr_kw_ids = self._csr_kw_ids

        overlay_table = pa.table({
            "docid_u32": pa.array(list(overlay_snapshot), type=pa.uint32()),
            "keyword_ids": pa.array(list(overlay_snapshot.values()), type=pa.list_(pa.uint32())),
        }).replace_schema_metadata({b"base_mapping": str(base_dir).encode()})

        def _populate_local_dir(local_d: Path) -> None:
            local_d.mkdir(parents=True, exist_ok=True)
            np.save(local_d / "docid_offsets.i64.npy", csr_offsets)
            np.save(local_d / "docid_kw_ids.u32.npy", csr_kw_ids)
            pq.write_table(overlay_table, local_d / "overlay.parquet", compression="zstd")
            for link_name in ("keywords.parquet", "docid_keywords.parquet", "docid_unique.u32.npy"):
                src_f = base_dir / link_name
                if not src_f.is_file():
                    src_f = self.path / link_name
                dst_f = local_d / link_name
                if not dst_f.exists() and not dst_f.is_symlink() and src_f.is_file():
                    dst_f.symlink_to(src_f)

        if stage_dir is not None:
            _populate_local_dir(Path(stage_dir))
            return {
                "output_dir": str(target_dir),
                "staged_dir": str(stage_dir),
                "id_count": int(n_unique),
                "rows": int(len(csr_kw_ids)),
            }

        tmp_dir = Path(tempfile.mkdtemp(prefix="gr_save_map_"))
        _populate_local_dir(tmp_dir)

        def _sync_worker() -> None:
            t0 = time.perf_counter()
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
                for fname in ("docid_offsets.i64.npy", "docid_kw_ids.u32.npy", "overlay.parquet"):
                    dst_tmp = target_dir / f"{fname}.tmp"
                    shutil.copyfile(tmp_dir / fname, dst_tmp)
                    dst_tmp.replace(target_dir / fname)
                for link_name in ("keywords.parquet", "docid_keywords.parquet", "docid_unique.u32.npy"):
                    src_link = tmp_dir / link_name
                    dst_link = target_dir / link_name
                    if src_link.is_symlink() and not dst_link.exists() and not dst_link.is_symlink():
                        dst_link.symlink_to(os.readlink(src_link))
                logger.info(
                    "[GR doc2k save] Synced staged mapping to %s in %.2fs",
                    target_dir,
                    time.perf_counter() - t0,
                )
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        if async_transfer:
            with self._async_save_lock:
                t = threading.Thread(
                    target=_sync_worker,
                    name=f"gr-map-sync-{target_dir.name}",
                    daemon=False,
                )
                self._async_save_thread = t
                t.start()
        else:
            _sync_worker()

        return {
            "output_dir": str(target_dir),
            "id_count": int(n_unique),
            "rows": int(len(csr_kw_ids)),
        }


def create_app(mapping: str | Path) -> FastAPI:
    path = absolute_path(mapping)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.mapping = ResidentMapping(path)
        gc.collect()
        gc.freeze()
        gc.set_threshold(100_000, 500, 500)
        yield
        del app.state.mapping

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        loaded = app.state.mapping
        return {"status": "ready", "mapping_version": loaded.version,
                "id_count": loaded.id_count, "rows": loaded.rows, "bytes": loaded.byte_count,
                "service_config": {
                    "api_version": _SERVICE_API_VERSION,
                    "mapping": str(path),
                }}

    @app.get("/runs/{owner}/health")
    def run_health(owner: str) -> dict[str, Any]:
        try:
            with app.state.mapping.access(owner):
                return health()
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/claim")
    def claim(request: ClaimMappingRequest) -> None:
        try:
            app.state.mapping.claim(request.owner, request.checkpoint)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/release")
    def release(request: ClaimMappingRequest) -> None:
        try:
            app.state.mapping.release(request.owner)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/lookup", response_model=LookupResponse)
    @app.post("/runs/{owner}/lookup", response_model=LookupResponse)
    def lookup(request: LookupRequest, http_request: Request) -> LookupResponse:
        try:
            with app.state.mapping.access(http_request.path_params.get("owner")):
                return app.state.mapping.lookup(
                    request.docids, market=request.market, use_initial=request.use_initial,
                )
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/runs/{owner}/resolve_keywords")
    def resolve_keywords(request: ResolveKeywordsRequest, http_request: Request) -> dict[str, list[int]]:
        try:
            with app.state.mapping.access(http_request.path_params["owner"]):
                return {"keyword_ids": app.state.mapping.resolve_keywords(request.keywords)}
        except (ValueError, KeyError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/update_delta")
    @app.post("/runs/{owner}/update_delta")
    def update_delta(request: DeltaMappingRequest, http_request: Request) -> dict[str, int]:
        try:
            with app.state.mapping.access(http_request.path_params.get("owner")):
                return app.state.mapping.apply_delta(request)
        except (ValueError, KeyError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/runs/{owner}/update_delta_ids")
    def update_delta_ids(request: DeltaMappingIdsRequest, http_request: Request) -> dict[str, int]:
        try:
            with app.state.mapping.access(http_request.path_params["owner"]):
                return app.state.mapping.apply_id_delta(request)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/update")
    @app.post("/runs/{owner}/update")
    def update(request: UpdateMappingRequest, http_request: Request) -> dict[str, int]:
        try:
            with app.state.mapping.access(http_request.path_params.get("owner")):
                return app.state.mapping.update_docid_mappings(request.entries, replace_all=request.replace_all)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/save")
    @app.post("/runs/{owner}/save")
    def save(request: SaveMappingRequest, http_request: Request) -> dict[str, Any]:
        try:
            with app.state.mapping.access(http_request.path_params.get("owner")):
                return app.state.mapping.save_docid_mappings(request.output_dir)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30488)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(args.mapping), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
