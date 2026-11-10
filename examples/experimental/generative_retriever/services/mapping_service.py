"""Resident official doc-ID to keyword-list service for the unified columnar index.

Run:
    python -m examples.experimental.generative_retriever.services.mapping_service \
        --mapping /path/to/slmgrv3_part25_na_intl_20261028_20261029 --port 30488
"""

from __future__ import annotations

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
from examples.experimental.generative_retriever.reward.relevance_reward import DOCID_MAX_KEYWORDS

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
    def __init__(self, path: Path) -> None:
        if not is_columnar_index_dir(path):
            raise ValueError(f"Mapping path must be a columnar index directory: {path}")
        self.path = path
        self.version = (
            symlink_target(path / "keywords.parquet").parent.name
            if path.name == "mapping" and (path / "keywords.parquet").is_symlink()
            else path.name
        )
        self._lock = RLock()
        self._owner: str | None = None
        self._active_requests = 0
        self._overlay_kw_ids: dict[int, tuple[int, ...]] = {}
        self._text_to_kid: dict[str, int] = {}
        self._decoded_kws: list[str | None] = []
        self._kw_array: Any = None
        self._in_na_array: np.ndarray = np.empty(0, dtype=np.bool_)
        self._markets_array: Any = None
        self._unique_docids_u32: np.ndarray = np.empty(0, dtype=np.uint32)
        self._csr_offsets: np.ndarray = np.empty(0, dtype=np.int64)
        self._csr_kw_ids: np.ndarray = np.empty(0, dtype=np.uint32)
        self._kw_degree: np.ndarray = np.empty(0, dtype=np.int32)
        self._load_columnar(path)

    def _load_columnar(self, directory: Path) -> None:
        logger.info("Loading columnar index into memory: %s", directory)
        kw_path = directory / "keywords.parquet"
        pairs_path = directory / "docid_keywords.parquet"
        kw_table = pq.read_table(kw_path, columns=["keyword_id", "keyword", "in_na", "markets"])
        self._kw_array = kw_table.column("keyword").combine_chunks()
        n_kw = len(self._kw_array)
        self._decoded_kws = [None] * n_kw
        if n_kw > 0:
            kid_col = kw_table.column("keyword_id")
            if int(kid_col[0].as_py()) != 0 or int(kid_col[n_kw - 1].as_py()) != n_kw - 1:
                raise ValueError("keywords.parquet keyword_id column must be contiguous 0..N-1")
        self._in_na_array = kw_table.column("in_na").to_numpy(zero_copy_only=False)
        self._markets_array = kw_table.column("markets").combine_chunks()

        u_path = directory / "docid_unique.u32.npy"
        off_path = directory / "docid_offsets.i64.npy"
        kid_path = directory / "docid_kw_ids.u32.npy"
        self._unique_docids_u32 = np.load(u_path)
        self._csr_offsets = np.load(off_path)
        self._csr_kw_ids = np.load(kid_path)

        self.rows = int(len(self._csr_kw_ids))
        self._kw_degree = np.bincount(self._csr_kw_ids, minlength=n_kw).astype(np.int32)
        self.byte_count = int(kw_path.stat().st_size + pairs_path.stat().st_size)
        logger.info(
            "Columnar mapping ready: %d IDs, %d keywords, %d pairs",
            len(self._unique_docids_u32),
            len(self._kw_array),
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
        cached = self._decoded_kws[kid]
        if cached is not None:
            return cached
        raw_bytes: bytes = self._kw_array[kid].as_py()
        text = raw_bytes.decode("utf-8")
        self._decoded_kws[kid] = text
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

    def claim(self, owner: str, checkpoint: str | None) -> None:
        overlay: dict[int, tuple[int, ...]] = {}
        if checkpoint is not None:
            table = pq.read_table(Path(checkpoint) / "overlay.parquet")
            metadata = table.schema.metadata or {}
            if metadata.get(b"base_mapping") != str(self.path).encode():
                raise ValueError("Checkpoint belongs to a different base mapping")
            overlay = {
                int(docid): tuple(kids)
                for docid, kids in zip(
                    table.column("docid_u32").to_pylist(),
                    table.column("keyword_ids").to_pylist(), strict=True,
                )
            }
        with self._lock:
            if not owner or self._owner is not None or self._active_requests:
                raise ValueError("Mapping service is already in use")
            self.replace_overlay(overlay)
            self._owner = owner

    def release(self, owner: str) -> None:
        with self._lock:
            if self._owner != owner or self._active_requests:
                raise ValueError("Mapping service is still in use or owned by another run")
            self._owner = None

    def resolve_keywords(self, keywords: list[str]) -> list[int]:
        with self._lock:
            return [self._text_to_kid[keyword] for keyword in keywords]

    def lookup(
        self,
        docids: list[PositionalID],
        *,
        market: str | None = None,
        use_initial: bool = False,
    ) -> LookupResponse:
        target_mkt = market.strip().upper() if market else None
        packed_targets = np.asarray([pack_docid_u32(d) for d in docids], dtype=np.uint32)
        positions = np.searchsorted(self._unique_docids_u32, packed_targets)
        results = []
        n_unique = len(self._unique_docids_u32)
        overlay_map = None if use_initial else self._overlay_kw_ids
        for docid, target_u32, pos in zip(docids, packed_targets, positions, strict=True):
            u32_int = int(target_u32)
            overlay_kids = None if overlay_map is None else overlay_map.get(u32_int)
            if overlay_kids is not None:
                kw_ids_iter: Any = overlay_kids
            else:
                idx = int(pos)
                if idx >= n_unique or int(self._unique_docids_u32[idx]) != u32_int:
                    logger.warning("Columnar mapping ID not found: %s", docid)
                    results.append(KeywordResult(docid=docid, keywords=[], status="not_found"))
                    continue
                start = int(self._csr_offsets[idx])
                end = int(self._csr_offsets[idx + 1])
                kw_ids_iter = self._csr_kw_ids[start:end]
            texts = []
            for raw_kid in kw_ids_iter:
                kid = int(raw_kid)
                if target_mkt is not None and not self._market_allows(kid, target_mkt):
                    continue
                texts.append(self._decode_kid(kid))
            results.append(KeywordResult(docid=docid, keywords=texts, status="ok"))
        return LookupResponse(mapping_version=self.version, results=results)

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
                if not remove_kids or len(current_kids) <= 1:
                    continue
                filtered_list: list[int] = []
                for idx_k, kid in enumerate(current_kids):
                    remaining_slots = len(current_kids) - idx_k
                    can_remove = (len(filtered_list) + remaining_slots) > 1
                    if (
                        can_remove
                        and kid in remove_kids
                        and int(self._kw_degree[kid]) > 1
                    ):
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
                        self._kw_degree[kid] += 1
                        added_edges += 1
                        changed = True
                if changed:
                    self._overlay_kw_ids[u32] = tuple(kid_list)
                    updated_u32.add(u32)
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

    def save_docid_mappings(self, output_dir: str | Path) -> dict[str, Any]:
        with self._lock:
            overlay_snapshot = dict(self._overlay_kw_ids)
        target_dir = absolute_path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        counts = np.diff(self._csr_offsets)
        base_docids = np.repeat(self._unique_docids_u32, counts)
        if overlay_snapshot:
            overlay_u32 = np.asarray(list(overlay_snapshot.keys()), dtype=np.uint32)
            docid_in_overlay = np.isin(self._unique_docids_u32, overlay_u32, assume_unique=False)
            keep_mask = np.repeat(~docid_in_overlay, counts)
            ov_docids_list: list[int] = []
            ov_kids_list: list[int] = []
            for u32, kids in overlay_snapshot.items():
                ov_docids_list.extend([u32] * len(kids))
                ov_kids_list.extend(kids)
            merged_docids = np.concatenate([
                base_docids[keep_mask],
                np.asarray(ov_docids_list, dtype=np.uint32),
            ])
            merged_kids = np.concatenate([
                self._csr_kw_ids[keep_mask],
                np.asarray(ov_kids_list, dtype=np.uint32),
            ])
            packed = (merged_docids.astype(np.uint64) << np.uint64(32)) | merged_kids.astype(np.uint64)
            packed = np.unique(packed)
            expanded_docids = (packed >> np.uint64(32)).astype(np.uint32)
            csr_kw_ids = (packed & np.uint64(0xFFFFFFFF)).astype(np.uint32)
            change_idx = np.flatnonzero(expanded_docids[1:] != expanded_docids[:-1]) + 1
            unique_docids = np.empty(len(change_idx) + 1, dtype=np.uint32)
            unique_docids[0] = expanded_docids[0]
            unique_docids[1:] = expanded_docids[change_idx]
            csr_offsets = np.empty(len(unique_docids) + 1, dtype=np.int64)
            csr_offsets[0] = 0
            csr_offsets[1:-1] = change_idx
            csr_offsets[-1] = len(expanded_docids)
        else:
            expanded_docids = base_docids
            unique_docids = self._unique_docids_u32
            csr_offsets = self._csr_offsets
            csr_kw_ids = self._csr_kw_ids
        pairs_schema = pa.schema([
            ("docid_u32", pa.uint32()),
            ("keyword_id", pa.uint32()),
        ])
        pairs_path = target_dir / "docid_keywords.parquet"
        pq.write_table(
            pa.Table.from_arrays(
                [pa.array(expanded_docids, type=pa.uint32()), pa.array(csr_kw_ids, type=pa.uint32())],
                schema=pairs_schema,
            ),
            pairs_path,
            compression="zstd",
        )
        np.save(target_dir / "docid_unique.u32.npy", unique_docids)
        np.save(target_dir / "docid_offsets.i64.npy", csr_offsets)
        np.save(target_dir / "docid_kw_ids.u32.npy", csr_kw_ids)
        src_kw = self.path / "keywords.parquet"
        dst_kw = target_dir / "keywords.parquet"
        if not dst_kw.exists() and src_kw.is_file():
            dst_kw.symlink_to(src_kw)
        overlay_table = pa.table({
            "docid_u32": pa.array(list(overlay_snapshot), type=pa.uint32()),
            "keyword_ids": pa.array(list(overlay_snapshot.values()), type=pa.list_(pa.uint32())),
        }).replace_schema_metadata({b"base_mapping": str(self.path).encode()})
        temporary = target_dir / "overlay.parquet.tmp"
        pq.write_table(overlay_table, temporary, compression="zstd")
        temporary.replace(target_dir / "overlay.parquet")
        return {
            "output_dir": str(target_dir),
            "id_count": int(len(unique_docids)),
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
