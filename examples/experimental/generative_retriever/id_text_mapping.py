"""Look up selected positional GR IDs in the official serving mapping TSV."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import httpx

PositionalID = tuple[int, int, int, int]


@dataclass(frozen=True)
class MappingLookup:
    texts: dict[PositionalID, tuple[str, ...]]
    matching_rows: int
    scanned_rows: int
    scanned_bytes: int


def _validate_id(docid: PositionalID) -> None:
    if len(docid) != 4:
        raise ValueError("GR positional ID must have four components")
    for position, code in enumerate(docid):
        if isinstance(code, bool) or not isinstance(code, int):
            raise TypeError("GR positional ID components must be integers")
        if not position * 256 <= code < (position + 1) * 256:
            raise ValueError(f"GR ID component {code} is outside position {position}'s codebook")


def lookup_texts(mapping: str | Path, docids: Iterable[PositionalID]) -> MappingLookup:
    """Scan the official TSV once, preserving every distinct matching text per ID."""
    path = Path(mapping).expanduser().resolve()
    requested = set(docids)
    if not requested:
        raise ValueError("At least one GR ID is required")
    for docid in requested:
        _validate_id(docid)
    encoded = {(" ".join(map(str, docid))).encode("ascii"): docid for docid in requested}
    texts: dict[PositionalID, dict[str, None]] = {docid: {} for docid in requested}
    matching_rows = scanned_rows = scanned_bytes = 0
    initial_stat = path.stat()
    with path.open("rb", buffering=8 * 1024 * 1024) as stream:
        for line_number, line in enumerate(stream, 1):
            scanned_rows += 1
            scanned_bytes += len(line)
            raw_id, separator, rest = line.partition(b"\t")
            docid = encoded.get(b" ".join(raw_id.split()))
            if docid is None:
                continue
            if not separator:
                raise ValueError(f"Mapping row {line_number} has no text column")
            raw_text = rest.partition(b"\t")[0].rstrip(b"\r\n")
            try:
                text = raw_text.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"Mapping row {line_number} has invalid UTF-8 text") from error
            if not text.strip():
                raise ValueError(f"Mapping row {line_number} has empty keyword text")
            texts[docid][text] = None
            matching_rows += 1
    final_stat = path.stat()
    if (
        scanned_bytes != initial_stat.st_size
        or final_stat.st_size != initial_stat.st_size
        or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
    ):
        raise RuntimeError("Official mapping changed during lookup; results are not a consistent snapshot")
    return MappingLookup(
        texts={docid: tuple(values) for docid, values in texts.items()},
        matching_rows=matching_rows,
        scanned_rows=scanned_rows,
        scanned_bytes=scanned_bytes,
    )


def lookup_service(
    service_url: str, docids: Iterable[PositionalID], *, timeout: float = 30.0,
    expected_mapping_version: str | None = None,
    client: httpx.Client | None = None,
) -> dict[PositionalID, tuple[str, ...]]:
    requested = list(dict.fromkeys(docids))
    if not requested:
        raise ValueError("Mapping service lookup requires at least one ID")
    for docid in requested:
        _validate_id(docid)
    url = f"{service_url.rstrip('/')}/lookup"
    texts = {}
    with httpx.Client(timeout=timeout) if client is None else nullcontext(client) as session:
        for start in range(0, len(requested), 4096):
            chunk = requested[start:start + 4096]
            response = session.post(url, json={"docids": chunk}, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Mapping service response must be a JSON object")
            if expected_mapping_version is not None and (
                payload.get("mapping_version") != expected_mapping_version
            ):
                raise ValueError(f"Mapping service version does not match {expected_mapping_version!r}")
            results = payload.get("results")
            if not isinstance(results, list) or len(results) != len(chunk):
                raise ValueError("Mapping service returned an invalid result count")
            for docid, result in zip(chunk, results, strict=True):
                if not isinstance(result, dict) or result.get("docid") != list(docid):
                    raise ValueError("Mapping service returned results in an unexpected ID order")
                if result.get("status") != "ok":
                    raise LookupError(f"Official mapping ID unavailable: {docid}, status={result.get('status')!r}")
                keywords = result.get("keywords")
                if not isinstance(keywords, list) or not keywords or any(
                    not isinstance(text, str) or not text.strip() for text in keywords
                ):
                    raise ValueError(f"Mapping service returned invalid keyword texts for {docid}")
                texts[docid] = tuple(keywords)
    return texts
