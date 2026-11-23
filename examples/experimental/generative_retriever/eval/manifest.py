"""Evaluation dataset manifest parsing and deterministic source-keyed naming."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from examples.experimental.generative_retriever.utils.paths import (
    atomic_stream_copy_file,
    stream_copy_file,
)


@dataclass(frozen=True)
class EvalManifest:
    source: str
    dataset_tag: str
    manifest_sha256: str
    source_digest: str


def source_slug(value: str) -> str:
    parts = [part for part in value.strip().strip("/").split("/") if part and not part.startswith("$")]
    if not parts:
        raise ValueError(f"Cannot derive dataset tag from {value!r}")
    raw = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    slug = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw).strip("._-")
    if not slug:
        raise ValueError(f"Cannot derive dataset tag from {value!r}")
    return slug


def parse_eval_manifest(metadata_json: str) -> EvalManifest:
    meta = json.loads(metadata_json)
    if not isinstance(meta, dict):
        raise ValueError("evaluation.json must contain a JSON object")
    raw_source = meta.get("source")
    if not isinstance(raw_source, str) or not raw_source.strip():
        raise ValueError("evaluation.json must contain a non-empty 'source' string")
    raw_sha = meta.get("manifest_sha256")
    if not isinstance(raw_sha, str) or not raw_sha.strip():
        raise ValueError("evaluation.json must contain a non-empty 'manifest_sha256' string")

    source = raw_source.strip().rstrip("/")
    manifest_sha256 = raw_sha.strip()
    raw_dataset_id = meta.get("dataset_id")
    dataset_tag = (
        source_slug(raw_dataset_id)
        if isinstance(raw_dataset_id, str) and raw_dataset_id.strip()
        else source_slug(source)
    )
    source_digest = hashlib.sha256(f"{source}\n{manifest_sha256}".encode()).hexdigest()[:12]
    return EvalManifest(
        source=source,
        dataset_tag=dataset_tag,
        manifest_sha256=manifest_sha256,
        source_digest=source_digest,
    )


def load_eval_manifest(eval_dir: Path) -> EvalManifest:
    return parse_eval_manifest((eval_dir / "evaluation.json").read_text(encoding="utf-8"))


def manifests_match(left: EvalManifest, right: EvalManifest) -> bool:
    return left.source == right.source and left.manifest_sha256 == right.manifest_sha256


def eval_dir_name(manifest: EvalManifest, *, docids_per_query: int, step: int) -> str:
    return f"{manifest.dataset_tag}_top{docids_per_query}_step{step}"


def baseline_cache_path(
    output_root: Path,
    manifest: EvalManifest,
    *,
    docids_per_query: int,
    mapping_version: str,
) -> Path:
    filename = (
        f"baseline_{manifest.dataset_tag}_{manifest.source_digest}"
        f"_top{docids_per_query}_{mapping_version}.jsonl"
    )
    return output_root / "eval_cache" / filename


def stage_eval_inputs(
    eval_root: Path,
    *,
    base_eval_root: Path,
    base_manifest: EvalManifest,
) -> EvalManifest:
    eval_root.mkdir(parents=True, exist_ok=True)
    try:
        manifest = load_eval_manifest(eval_root)
    except FileNotFoundError:
        for filename in ("queries.jsonl", "evaluation.json"):
            stream_copy_file(base_eval_root / filename, eval_root / filename)
        return base_manifest
    if not manifests_match(manifest, base_manifest):
        raise ValueError(
            f"Existing eval_root source ({manifest.source}) does not match "
            f"base_eval_root ({base_manifest.source})"
        )
    return manifest


def publish_baseline_cache(source_baseline: Path, cached_baseline: Path) -> None:
    atomic_stream_copy_file(source_baseline, cached_baseline)
