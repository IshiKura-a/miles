"""Resident official doc-ID to keyword-list service.

Run:
    python -m examples.experimental.generative_retriever.mapping_service \
        --mapping /path/to/asiindex_mapping20261005022223.tsv --port 30488
"""

from __future__ import annotations

import argparse
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field, StrictInt, field_validator

from examples.experimental.generative_retriever.id_text_mapping import PositionalID, _validate_id

logger = logging.getLogger(__name__)


class LookupRequest(BaseModel):
    docids: Annotated[list[tuple[StrictInt, StrictInt, StrictInt, StrictInt]], Field(min_length=1, max_length=4096)]

    @field_validator("docids")
    @classmethod
    def validate_docids(cls, values: list[PositionalID]) -> list[PositionalID]:
        for value in values:
            _validate_id(value)
        return values


class KeywordResult(BaseModel):
    docid: PositionalID
    keywords: list[str]
    status: str


class LookupResponse(BaseModel):
    mapping_version: str
    results: list[KeywordResult]


class ResidentMapping:
    def __init__(self, path: Path) -> None:
        self.path = path
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
                    _validate_id(values)
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
        return len(self._keywords)

    def lookup(self, docids: list[PositionalID]) -> LookupResponse:
        results = []
        for docid in docids:
            raw_id = " ".join(map(str, docid)).encode("ascii")
            values = self._keywords.get(raw_id)
            if values is None:
                logger.warning("Official mapping ID not found: %s", docid)
                results.append(KeywordResult(docid=docid, keywords=[], status="not_found"))
                continue
            texts = [value.decode("utf-8") for value in dict.fromkeys(values)]
            results.append(KeywordResult(docid=docid, keywords=texts, status="ok"))
        return LookupResponse(mapping_version=self.path.name, results=results)


def create_app(mapping: str | Path) -> FastAPI:
    path = Path(mapping).expanduser().resolve()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.mapping = ResidentMapping(path)
        yield
        del app.state.mapping

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str | int]:
        loaded = app.state.mapping
        return {"status": "ready", "mapping_version": path.name,
                "id_count": loaded.id_count, "rows": loaded.rows, "bytes": loaded.byte_count}

    @app.post("/lookup", response_model=LookupResponse)
    def lookup(request: LookupRequest) -> LookupResponse:
        return app.state.mapping.lookup(request.docids)

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
