"""Convert SLMv3 queryfeat RecordIO parts to Miles JSONL prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_reader(slm_root: Path):
    root = slm_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"SLMv3 source directory does not exist: {root}")
    sys.path.insert(0, str(root))
    try:
        from mxjsondataset.dataset_mt_with_DocEncJson import MXIndexedRecordIO
    except ModuleNotFoundError as error:
        if error.name == "mxnet":
            raise RuntimeError(
                "queryfeat conversion requires MXNet because the source data is MXRecordIO"
            ) from error
        raise
    return MXIndexedRecordIO


def _parse_record(record: bytes, source: Path, index: int) -> dict[str, Any]:
    try:
        value = json.loads(record.decode("utf-8"))
        if isinstance(value, str):
            value = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON record at {source}, index {index}") from error
    if not isinstance(value, dict):
        raise ValueError(f"queryfeat record at {source}, index {index} is not an object")
    query = value.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"queryfeat record at {source}, index {index} has no query")
    return value


def convert_queryfeat(
    *,
    parts: list[str | Path],
    output: str | Path,
    slm_root: str | Path,
    limit: int | None = None,
) -> int:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    reader_class = _load_reader(Path(slm_root))
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for raw_part in parts:
            part = Path(raw_part).expanduser().resolve()
            record_path = Path(f"{part}.rec")
            index_path = Path(f"{part}.np.bin")
            if not record_path.is_file() or not index_path.is_file():
                raise FileNotFoundError(
                    f"queryfeat part requires {record_path} and {index_path}"
                )
            reader = reader_class(str(index_path), str(record_path), "r")
            try:
                for index in range(len(reader.idx)):
                    value = _parse_record(reader.read_idx(index), part, index)
                    output_file.write(
                        json.dumps(
                            {
                                "prompt": value["query"],
                                "label": value.get("label"),
                                "metadata": {
                                    "source": value.get("task"),
                                    "qid": value.get("qid"),
                                },
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    count += 1
                    if limit is not None and count >= limit:
                        return count
            finally:
                reader.close()
    return count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--slm-root", required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    count = convert_queryfeat(
        parts=args.part,
        output=args.output,
        slm_root=args.slm_root,
        limit=args.limit,
    )
    print(count)


if __name__ == "__main__":
    main()
