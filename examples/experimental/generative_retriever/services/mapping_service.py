"""Resident official doc-ID to keyword-list service.

Run:
    python -m examples.experimental.generative_retriever.services.mapping_service \
        --mapping /path/to/asiindex_mapping.ta_listing.tsv --port 30488
"""

from __future__ import annotations

import argparse
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from threading import RLock
from typing import Annotated, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StrictInt, field_validator

from examples.experimental.generative_retriever.index.id_text_mapping import (
    PositionalID,
    is_columnar_index_dir,
    pack_docid_u32,
    validate_docid,
)

logger = logging.getLogger(__name__)
_SERVICE_API_VERSION = 2
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
    k_max: int = 150


class SaveMappingRequest(BaseModel):
    output_dir: str


class ResidentMapping:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = RLock()
        self._is_columnar = is_columnar_index_dir(path)
        self._keywords: dict[bytes, list[bytes]] = {}
        self._overlay_texts: dict[PositionalID, list[str]] = {}
        self._kw_array: Any = None
        self._in_na_array: np.ndarray | None = None
        self._markets_array: Any = None
        self._unique_docids_u32: np.ndarray | None = None
        self._csr_offsets: np.ndarray | None = None
        self._csr_kw_ids: np.ndarray | None = None

        if self._is_columnar:
            self._load_columnar(path)
        else:
            self._load_tsv(path)

    def _load_columnar(self, directory: Path) -> None:
        logger.info("Loading columnar index into memory: %s", directory)
        kw_path = directory / "keywords.parquet"
        pairs_path = directory / "docid_keywords.parquet"
        kw_table = pq.read_table(kw_path, columns=["keyword", "in_na", "markets"])
        self._kw_array = kw_table.column("keyword").combine_chunks()
        self._in_na_array = kw_table.column("in_na").to_numpy(zero_copy_only=False)
        self._markets_array = kw_table.column("markets").combine_chunks()

        u_path = directory / "docid_unique.u32.npy"
        off_path = directory / "docid_offsets.i64.npy"
        kid_path = directory / "docid_kw_ids.u32.npy"
        if u_path.is_file() and off_path.is_file() and kid_path.is_file():
            self._unique_docids_u32 = np.load(u_path)
            self._csr_offsets = np.load(off_path)
            self._csr_kw_ids = np.load(kid_path)
        else:
            pairs_table = pq.read_table(pairs_path, columns=["docid_u32", "keyword_id"])
            docid_u32 = pairs_table.column("docid_u32").to_numpy(zero_copy_only=False)
            kw_id = pairs_table.column("keyword_id").to_numpy(zero_copy_only=False)
            self._rebuild_csr(docid_u32, kw_id)

        assert self._unique_docids_u32 is not None and self._csr_kw_ids is not None
        self.rows = int(len(self._csr_kw_ids))
        self.byte_count = int(kw_path.stat().st_size + pairs_path.stat().st_size)
        logger.info(
            "Columnar mapping ready: %d IDs, %d keywords, %d pairs",
            len(self._unique_docids_u32),
            len(self._kw_array),
            self.rows,
        )

    def _rebuild_csr(self, docid_u32: np.ndarray, kw_id: np.ndarray) -> None:
        if len(docid_u32) == 0:
            raise ValueError("docid-to-keyword mapping cannot be empty")
        packed = (docid_u32.astype(np.uint64) << np.uint64(32)) | kw_id.astype(np.uint64)
        packed = np.unique(packed)
        sorted_docids = (packed >> np.uint64(32)).astype(np.uint32)
        sorted_kw_ids = (packed & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        change_idx = np.flatnonzero(sorted_docids[1:] != sorted_docids[:-1]) + 1
        unique_docids = np.empty(len(change_idx) + 1, dtype=np.uint32)
        unique_docids[0] = sorted_docids[0]
        unique_docids[1:] = sorted_docids[change_idx]
        offsets = np.empty(len(unique_docids) + 1, dtype=np.int64)
        offsets[0] = 0
        offsets[1:-1] = change_idx
        offsets[-1] = len(sorted_docids)
        self._unique_docids_u32 = unique_docids
        self._csr_offsets = offsets
        self._csr_kw_ids = sorted_kw_ids
        self.rows = int(len(sorted_kw_ids))

    def _load_tsv(self, path: Path) -> None:
        initial = path.stat()
        keywords: dict[bytes, list[bytes]] = {}
        scanned_bytes = scanned_rows = 0
        previous_id = None
        bucket: list[bytes] = []
        logger.info("Loading official mapping into memory: %s", path)
        with path.open("rb", buffering=8 * 1024 * 1024) as stream:
            for line_number, line in enumerate(stream, 1):
                raw_id, separator, rest = line.partition(b"\t")
                if not separator:
                    raise ValueError(f"Official mapping row {line_number} has no text column")
                raw_text = rest.partition(b"\t")[0].rstrip(b"\r\n")
                if not raw_text.strip():
                    raise ValueError(f"Official mapping row {line_number} has empty text")
                if raw_id != previous_id:
                    components = [int(value) for value in raw_id.split()]
                    if len(components) != 4:
                        raise ValueError(f"Official mapping row {line_number} must have four ID components")
                    values = (components[0], components[1], components[2], components[3])
                    validate_docid(values)
                    canonical_id = " ".join(map(str, values)).encode("ascii")
                    bucket = keywords.setdefault(canonical_id, [])
                    previous_id = raw_id
                bucket.append(raw_text)
                scanned_bytes += len(line)
                scanned_rows += 1
                if scanned_rows % 10_000_000 == 0:
                    logger.info("Loaded %d rows, %d IDs", scanned_rows, len(keywords))
        final = path.stat()
        if (scanned_bytes, final.st_size, final.st_mtime_ns) != (initial.st_size, initial.st_size, initial.st_mtime_ns):
            raise RuntimeError("Official mapping changed during startup")
        self._keywords = keywords
        self.rows = scanned_rows
        self.byte_count = scanned_bytes
        logger.info("Official mapping ready: %d IDs, %d rows", len(keywords), scanned_rows)

    @property
    def id_count(self) -> int:
        with self._lock:
            if self._is_columnar and self._unique_docids_u32 is not None:
                return int(len(self._unique_docids_u32))
            return len(self._keywords)

    def _market_allows(self, kw_id: int, target_market_upper: str) -> bool:
        assert self._in_na_array is not None and self._markets_array is not None
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

    def _base_lookup_one(self, docid: PositionalID, target_mkt: str | None) -> list[str] | None:
        if not self._is_columnar:
            raw_id = " ".join(map(str, docid)).encode("ascii")
            values = self._keywords.get(raw_id)
            if values is None:
                return None
            return [value.decode("utf-8") for value in dict.fromkeys(values)]
        assert (
            self._unique_docids_u32 is not None
            and self._csr_offsets is not None
            and self._csr_kw_ids is not None
            and self._kw_array is not None
        )
        target_u32 = np.uint32(pack_docid_u32(docid))
        idx = int(np.searchsorted(self._unique_docids_u32, target_u32))
        n_unique = len(self._unique_docids_u32)
        if idx >= n_unique or int(self._unique_docids_u32[idx]) != int(target_u32):
            return None
        start = int(self._csr_offsets[idx])
        end = int(self._csr_offsets[idx + 1])
        kw_ids = self._csr_kw_ids[start:end]
        texts = []
        for raw_kid in kw_ids:
            kid = int(raw_kid)
            if target_mkt is not None and not self._market_allows(kid, target_mkt):
                continue
            raw_bytes: bytes = self._kw_array[kid].as_py()
            texts.append(raw_bytes.decode("utf-8"))
        return texts

    def lookup(
        self,
        docids: list[PositionalID],
        *,
        market: str | None = None,
        use_initial: bool = False,
    ) -> LookupResponse:
        target_mkt = market.strip().upper() if market else None
        with self._lock:
            if not self._is_columnar:
                results = []
                for docid in docids:
                    if not use_initial and docid in self._overlay_texts:
                        results.append(
                            KeywordResult(
                                docid=docid,
                                keywords=list(self._overlay_texts[docid]),
                                status="ok",
                            )
                        )
                        continue
                    raw_id = " ".join(map(str, docid)).encode("ascii")
                    values = self._keywords.get(raw_id)
                    if values is None:
                        logger.warning("Official mapping ID not found: %s", docid)
                        results.append(KeywordResult(docid=docid, keywords=[], status="not_found"))
                        continue
                    texts = [value.decode("utf-8") for value in dict.fromkeys(values)]
                    results.append(KeywordResult(docid=docid, keywords=texts, status="ok"))
                return LookupResponse(mapping_version=self.path.name, results=results)

            assert (
                self._unique_docids_u32 is not None
                and self._csr_offsets is not None
                and self._csr_kw_ids is not None
                and self._kw_array is not None
            )
            packed_targets = np.asarray([pack_docid_u32(d) for d in docids], dtype=np.uint32)
            positions = np.searchsorted(self._unique_docids_u32, packed_targets)
            results = []
            n_unique = len(self._unique_docids_u32)
            for docid, target_u32, pos in zip(docids, packed_targets, positions, strict=True):
                if not use_initial and docid in self._overlay_texts:
                    results.append(
                        KeywordResult(
                            docid=docid,
                            keywords=list(self._overlay_texts[docid]),
                            status="ok",
                        )
                    )
                    continue
                idx = int(pos)
                if idx >= n_unique or int(self._unique_docids_u32[idx]) != int(target_u32):
                    logger.warning("Columnar mapping ID not found: %s", docid)
                    results.append(KeywordResult(docid=docid, keywords=[], status="not_found"))
                    continue
                start = int(self._csr_offsets[idx])
                end = int(self._csr_offsets[idx + 1])
                kw_ids = self._csr_kw_ids[start:end]
                texts = []
                for raw_kid in kw_ids:
                    kid = int(raw_kid)
                    if target_mkt is not None and not self._market_allows(kid, target_mkt):
                        continue
                    raw_bytes: bytes = self._kw_array[kid].as_py()
                    texts.append(raw_bytes.decode("utf-8"))
                results.append(KeywordResult(docid=docid, keywords=texts, status="ok"))
            return LookupResponse(mapping_version=self.path.name, results=results)

    def apply_delta(self, request: DeltaMappingRequest) -> dict[str, int]:
        if request.k_max < 1:
            raise ValueError("k_max must be at least 1")
        with self._lock:
            updated_docids: set[PositionalID] = set()
            added_edges = 0
            removed_edges = 0
            for item in request.removals:
                docid = item.docid
                current = self._overlay_texts.get(docid)
                if current is None:
                    base = self._base_lookup_one(docid, None)
                    if not base:
                        continue
                    current = list(base)
                remove_set = {kw for kw in item.keywords if kw.strip()}
                filtered = [kw for kw in current if kw not in remove_set]
                if not filtered:
                    filtered = [current[0]]
                if len(filtered) < len(current):
                    removed_edges += len(current) - len(filtered)
                    self._overlay_texts[docid] = filtered
                    updated_docids.add(docid)
            for item in request.additions:
                docid = item.docid
                current = self._overlay_texts.get(docid)
                if current is None:
                    base = self._base_lookup_one(docid, None)
                    if not base:
                        continue
                    current = list(base)
                existing = set(current)
                changed = False
                for kw in item.keywords:
                    if len(current) >= request.k_max:
                        break
                    if kw.strip() and kw not in existing:
                        current.append(kw)
                        existing.add(kw)
                        added_edges += 1
                        changed = True
                if changed:
                    self._overlay_texts[docid] = current
                    updated_docids.add(docid)
            return {
                "updated_docids": len(updated_docids),
                "added_edges": added_edges,
                "removed_edges": removed_edges,
                "overlay_docids": len(self._overlay_texts),
            }

    def update_docid_mappings(
        self,
        entries: list[DocidKeywordPair],
        *,
        replace_all: bool = False,
    ) -> dict[str, int]:
        with self._lock:
            if not self._is_columnar or self._kw_array is None:
                raise ValueError("In-memory docid2keyword update requires a columnar index")
            max_kw_id = len(self._kw_array)
            new_docids: list[int] = []
            new_kw_ids: list[int] = []
            for entry in entries:
                u32 = pack_docid_u32(entry.docid)
                for kid in entry.keyword_ids:
                    if not 0 <= kid < max_kw_id:
                        raise ValueError(f"keyword_id {kid} out of range [0, {max_kw_id})")
                    new_docids.append(u32)
                    new_kw_ids.append(kid)
            new_docids_arr = np.asarray(new_docids, dtype=np.uint32)
            new_kw_ids_arr = np.asarray(new_kw_ids, dtype=np.uint32)
            if not replace_all and self._unique_docids_u32 is not None and self._csr_offsets is not None and self._csr_kw_ids is not None:
                counts = np.diff(self._csr_offsets)
                existing_docids = np.repeat(self._unique_docids_u32, counts)
                updated_set = np.unique(new_docids_arr)
                keep_mask = ~np.isin(existing_docids, updated_set)
                merged_docids = np.concatenate([existing_docids[keep_mask], new_docids_arr])
                merged_kw_ids = np.concatenate([self._csr_kw_ids[keep_mask], new_kw_ids_arr])
                self._rebuild_csr(merged_docids, merged_kw_ids)
            else:
                self._rebuild_csr(new_docids_arr, new_kw_ids_arr)
            assert self._unique_docids_u32 is not None
            return {"id_count": int(len(self._unique_docids_u32)), "rows": self.rows}

    def save_docid_mappings(self, output_dir: str | Path) -> dict[str, Any]:
        with self._lock:
            if not self._is_columnar or self._unique_docids_u32 is None or self._csr_offsets is None or self._csr_kw_ids is None:
                raise ValueError("Saving dynamic docid2keyword mappings requires a columnar index")
            target_dir = Path(output_dir).expanduser().resolve()
            target_dir.mkdir(parents=True, exist_ok=True)
            counts = np.diff(self._csr_offsets)
            expanded_docids = np.repeat(self._unique_docids_u32, counts)
            pairs_schema = pa.schema([
                ("docid_u32", pa.uint32()),
                ("keyword_id", pa.uint32()),
            ])
            pairs_path = target_dir / "docid_keywords.parquet"
            pq.write_table(
                pa.Table.from_arrays(
                    [pa.array(expanded_docids, type=pa.uint32()), pa.array(self._csr_kw_ids, type=pa.uint32())],
                    schema=pairs_schema,
                ),
                pairs_path,
                compression="zstd",
            )
            np.save(target_dir / "docid_unique.u32.npy", self._unique_docids_u32)
            np.save(target_dir / "docid_offsets.i64.npy", self._csr_offsets)
            np.save(target_dir / "docid_kw_ids.u32.npy", self._csr_kw_ids)
            src_kw = self.path / "keywords.parquet"
            dst_kw = target_dir / "keywords.parquet"
            if not dst_kw.exists() and src_kw.is_file():
                dst_kw.symlink_to(src_kw)
            return {
                "output_dir": str(target_dir),
                "id_count": int(len(self._unique_docids_u32)),
                "rows": self.rows,
            }


def create_app(mapping: str | Path) -> FastAPI:
    path = Path(mapping).expanduser().resolve()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.mapping = ResidentMapping(path)
        yield
        del app.state.mapping

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        loaded = app.state.mapping
        return {"status": "ready", "mapping_version": path.name,
                "id_count": loaded.id_count, "rows": loaded.rows, "bytes": loaded.byte_count,
                "service_config": {
                    "api_version": _SERVICE_API_VERSION,
                    "mapping": str(path),
                }}

    @app.post("/lookup", response_model=LookupResponse)
    def lookup(request: LookupRequest) -> LookupResponse:
        return app.state.mapping.lookup(
            request.docids,
            market=request.market,
            use_initial=request.use_initial,
        )

    @app.post("/update_delta")
    def update_delta(request: DeltaMappingRequest) -> dict[str, int]:
        try:
            return app.state.mapping.apply_delta(request)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/update")
    def update(request: UpdateMappingRequest) -> dict[str, int]:
        try:
            return app.state.mapping.update_docid_mappings(
                request.entries,
                replace_all=request.replace_all,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/save")
    def save(request: SaveMappingRequest) -> dict[str, Any]:
        try:
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
