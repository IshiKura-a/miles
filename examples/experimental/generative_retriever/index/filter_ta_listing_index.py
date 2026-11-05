"""Filter a nine-column ASI serving index to TA listings without changing its rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path


def filter_index(source: Path, output: Path) -> dict[str, object]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_file() or source == output:
        raise ValueError(f"Source must exist and differ from output: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    manifest = output.with_suffix(".json")
    if output.exists() or partial.exists() or manifest.exists():
        raise FileExistsError(f"Refusing to replace index output: {output}")

    initial = source.stat()
    source_hash = hashlib.sha256()
    filtered_hash = hashlib.sha256()
    types: Counter[str] = Counter()
    started = time.monotonic()
    with source.open("rb", buffering=8 * 1024 * 1024) as infile, partial.open(
        "wb", buffering=8 * 1024 * 1024,
    ) as outfile:
        for row_number, line in enumerate(infile, 1):
            source_hash.update(line)
            if line.count(b"\t") != 8:
                raise ValueError(f"Expected nine tab-separated fields at source row {row_number}")
            raw_type = line.rsplit(b"\t", 1)[1].rstrip(b"\r\n")
            try:
                kind = raw_type.decode("ascii")
            except UnicodeDecodeError as error:
                raise ValueError(f"Invalid source type at row {row_number}") from error
            types[kind] += 1
            if kind == "SALLTAListing":
                outfile.write(line)
                filtered_hash.update(line)
            if row_number % 10_000_000 == 0:
                print(
                    f"scanned={row_number} ta_listing={types['SALLTAListing']} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
        outfile.flush()
        os.fsync(outfile.fileno())
    final = source.stat()
    if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
        raise RuntimeError("Source index changed while filtering")
    if not types["SALLTAListing"]:
        raise ValueError("Source index has no TA listings")
    os.replace(partial, output)
    result: dict[str, object] = {
        "source": str(source),
        "source_bytes": initial.st_size,
        "source_sha256": source_hash.hexdigest(),
        "source_rows": sum(types.values()),
        "source_types": dict(sorted(types.items())),
        "output": str(output),
        "output_rows": types["SALLTAListing"],
        "output_bytes": output.stat().st_size,
        "output_sha256": filtered_hash.hexdigest(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = filter_index(args.source, args.output)
    print(f"INDEX_READY rows={result['output_rows']} path={result['output']}", flush=True)


if __name__ == "__main__":
    main()
