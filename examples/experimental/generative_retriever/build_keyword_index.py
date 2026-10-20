"""Index TA-Keyword RecordIO keywords with the frozen Stage5 document encoder and RQVAE."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from .id_text_mapping import _validate_id
from .model_loading import (
    build_run_arguments,
    checkpoint_tensor_shape,
    import_slmv3,
    load_checkpoint,
    require_directory,
)

_RECORD_MAGIC = 0xCED7230A


def _record(mm: mmap.mmap, offsets: np.memmap, index: int) -> dict[str, Any]:
    offset = int(offsets[index])
    magic, length = struct.unpack_from("<II", mm, offset)
    if magic != _RECORD_MAGIC or length >> 29:
        raise ValueError(f"Unsupported RecordIO header at row {index}")
    end = offset + 8 + length
    next_offset = int(offsets[index + 1]) if index + 1 < len(offsets) else len(mm)
    if end > len(mm) or next_offset != (end + 3) & ~3:
        raise ValueError(f"Invalid RecordIO length at row {index}")
    value = json.loads(mm[offset + 8:end])
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict) or value.get("task") != "TA-Keyword":
        raise ValueError(f"Expected a TA-Keyword object at row {index}")
    keyword = value.get("keyword")
    if not isinstance(keyword, str) or not keyword.strip() or any(c in keyword for c in "\t\r\n"):
        raise ValueError(f"Invalid TSV keyword at row {index}")
    if not isinstance(value.get("query"), str):
        raise ValueError(f"Invalid query field at row {index}")
    return value


def _load_model(checkpoint: Path, qwen_model: Path, slm_root: Path, device: str) -> Any:
    import torch
    from transformers import T5Config, T5ForConditionalGeneration

    slmv3 = import_slmv3(slm_root)
    args = build_run_arguments(slmv3["RunArguments"], checkpoint, qwen_model)
    config = T5Config.from_pretrained(checkpoint)
    config.vocab_size = checkpoint_tensor_shape(checkpoint, "query_encoder.embed_tokens.weight")[0]
    backbone = T5ForConditionalGeneration(config)
    docencoder = slmv3["Encoder_QWen1216t"](args)
    model = slmv3["T5MultiWithGivenQueryDocEncoderDSI"](
        config, backbone, encoder=None, docencoder=docencoder,
        docid_dim=args.code_dim, docid_size=args.code_num,
        margin=args.margin, running_args=args, domain_idx=[0],
    )
    load_checkpoint(model, checkpoint)
    return model.eval().to(device=device, dtype=torch.bfloat16), slmv3["DataSupport_QWen1216t"](args)


def _encode(
    model: Any, support: Any, records: list[dict[str, Any]], device: str,
) -> list[tuple[int, int, int, int]]:
    import torch

    prompts = [
        support.cutomized_apply_chat_template(
            support.customized_tokenize(Query=record["query"], Doc=record)["doc_str"]
        )
        for record in records
    ]
    tokenized = support.processor(
        text=prompts, padding=True, truncation=True,
        max_length=support.max_len, return_tensors="pt",
    )
    with torch.inference_mode():
        inputs = {name: tensor.to(device) for name, tensor in tokenized.items()}
        embeddings = model.obtain_slm_embeddings(**inputs)
        tokens = model.infer_document_encoder(
            input_embs=embeddings, return_docid_statistics=False,
        )[3]
        if tokens.shape != (len(records), 4):
            raise RuntimeError(f"Invalid RQVAE doc ID tensor shape: {tuple(tokens.shape)}")
        codes = tokens.to(dtype=torch.int32).cpu().tolist()
    result = []
    for row in codes:
        docid = (row[0] - 2, row[1] - 2, row[2] - 2, row[3] - 2)
        _validate_id(docid)
        result.append(docid)
    return result


def build_shard(args: argparse.Namespace) -> None:
    checkpoint = require_directory(args.checkpoint, "Stage5 checkpoint")
    qwen_model = require_directory(args.qwen_model, "Qwen model")
    slm_root = require_directory(args.slm_root, "SLMv3 source")
    source = Path(args.part).expanduser().resolve()
    index_path = Path(f"{source}.np.bin")
    record_path = Path(f"{source}.rec")
    if not index_path.is_file() or not record_path.is_file():
        raise FileNotFoundError(f"RecordIO part requires {index_path} and {record_path}")
    if args.batch_size <= 0 or not 0 <= args.shard < args.shards:
        raise ValueError("Require a positive batch size and 0 <= shard < shards")
    offsets = np.memmap(index_path, dtype="<i8", mode="r")
    total = len(offsets) if args.limit is None else min(args.limit, len(offsets))
    if total <= 0 or total != 10_000_000 and args.limit is None:
        raise ValueError(f"Unexpected input record count: {total}")
    start = total * args.shard // args.shards
    end = total * (args.shard + 1) // args.shards
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"part-{args.shard:02d}-of-{args.shards:02d}.tsv"
    partial = output.with_suffix(".tsv.partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"Refusing to replace existing shard: {output} or {partial}")
    model, support = _load_model(checkpoint, qwen_model, slm_root, args.device)
    started = time.monotonic()
    hasher = hashlib.sha256()
    with record_path.open("rb") as source_file, mmap.mmap(
        source_file.fileno(), 0, access=mmap.ACCESS_READ,
    ) as mm, partial.open("wb", buffering=8 * 1024 * 1024) as stream:
        for first in range(start, end, args.batch_size):
            last = min(first + args.batch_size, end)
            records = [_record(mm, offsets, index) for index in range(first, last)]
            docids = _encode(model, support, records, args.device)
            for record, docid in zip(records, docids, strict=True):
                line = f"{' '.join(map(str, docid))}\t{record['keyword']}\n".encode()
                stream.write(line)
                hasher.update(line)
            if (last - start) % 100_000 < args.batch_size or last == end:
                print(f"shard={args.shard} rows={last - start}/{end - start} elapsed={time.monotonic() - started:.1f}s", flush=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, output)
    manifest = {
        "source": str(source),
        "source_records": len(offsets),
        "start": start,
        "end": end,
        "rows": end - start,
        "checkpoint": str(checkpoint),
        "qwen_model": str(qwen_model),
        "slm_root": str(slm_root),
        "sha256": hasher.hexdigest(),
        "seconds": round(time.monotonic() - started, 3),
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")


def merge_shards(args: argparse.Namespace) -> None:
    output_dir = Path(args.output).expanduser().resolve()
    manifests = []
    shards = []
    for shard in range(args.shards):
        path = output_dir / f"part-{shard:02d}-of-{args.shards:02d}.tsv"
        manifest_path = path.with_suffix(".json")
        if not path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Missing completed shard: {path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest["rows"] != manifest["end"] - manifest["start"]:
            raise ValueError(f"Invalid shard row count: {path}")
        if manifests and (
            manifest["start"] != manifests[-1]["end"]
            or any(manifest[key] != manifests[0][key]
                   for key in ("source", "source_records", "checkpoint", "qwen_model", "slm_root"))
        ):
            raise ValueError(f"Inconsistent shard provenance: {path}")
        manifests.append(manifest)
        shards.append(path)
    if manifests[0]["start"] != 0 or manifests[-1]["end"] != manifests[0]["source_records"]:
        raise ValueError("Shards do not cover the complete source corpus")
    output = output_dir / "sid_keyword.tsv"
    partial = output_dir / "sid_keyword.tsv.partial"
    if output.exists() or partial.exists():
        raise FileExistsError(f"Refusing to replace an existing index: {output}")
    with partial.open("wb") as stream:
        subprocess.run(
            ["sort", "-u", "-S", "8G", "-T", str(output_dir), "--", *map(str, shards)],
            check=True,
            env={**os.environ, "LC_ALL": "C"},
            stdout=stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    digest = hashlib.sha256()
    rows = 0
    roots: set[int] = set()
    with partial.open("rb", buffering=8 * 1024 * 1024) as stream:
        for line in stream:
            digest.update(line)
            rows += 1
            raw_id, separator, raw_keyword = line.partition(b"\t")
            if not separator or not raw_keyword.strip():
                raise ValueError(f"Invalid merged TSV at row {rows}")
            components = [int(code) for code in raw_id.split()]
            if len(components) != 4:
                raise ValueError(f"Invalid merged doc ID at row {rows}")
            docid = (components[0], components[1], components[2], components[3])
            _validate_id(docid)
            roots.add(docid[0])
    if roots != set(range(256)):
        raise ValueError(f"Index misses first-level codes: {sorted(set(range(256)) - roots)}")
    os.replace(partial, output)
    (output_dir / "index.json").write_text(json.dumps({
        "source": manifests[0]["source"],
        "source_records": manifests[0]["source_records"],
        "checkpoint": manifests[0]["checkpoint"],
        "qwen_model": manifests[0]["qwen_model"],
        "slm_root": manifests[0]["slm_root"],
        "rows": rows,
        "sha256": digest.hexdigest(),
        "format": "positional_id_4_space_separated\\tkeyword",
    }, indent=2) + "\n")
    print(f"INDEX_READY rows={rows} sha256={digest.hexdigest()} path={output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    shard = modes.add_parser("shard")
    merge = modes.add_parser("merge")
    for mode in (shard, merge):
        mode.add_argument("--output", required=True)
        mode.add_argument("--shards", type=int, default=8)
    shard.add_argument("--part", required=True)
    shard.add_argument("--checkpoint", required=True)
    shard.add_argument("--qwen-model", required=True)
    shard.add_argument("--slm-root", required=True)
    shard.add_argument("--shard", type=int, default=0)
    shard.add_argument("--batch-size", type=int, default=128)
    shard.add_argument("--device", default="cuda:0")
    shard.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.shards <= 0:
        parser.error("--shards must be positive")
    if args.mode == "shard":
        build_shard(args)
    else:
        merge_shards(args)


if __name__ == "__main__":
    main()
