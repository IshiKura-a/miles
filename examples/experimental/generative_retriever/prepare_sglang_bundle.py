"""Create a lightweight SGLang model bundle for an SLMv3 checkpoint.

The bundle contains Qwen configuration/tokenizer files and symlinks to the GR
checkpoint shards. Model weights are not copied.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .model_loading import require_directory


def require_sglang_abort_patch() -> None:
    patch = Path(__file__).resolve().parent / "patches/sglang-embedding-abort.patch"
    sglang_root = Path(sys.prefix) / "src/sglang"
    result = subprocess.run(
        ["git", "-C", str(sglang_root), "apply", "--reverse", "--check", str(patch)],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"SGLang embedding abort patch is missing from {sglang_root}; "
            f"apply {patch} before launching training: {result.stderr.strip()}"
        )


_QWEN_ASSETS = (
    "generation_config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _replace_symlink(link: Path, target: Path) -> None:
    if link.is_symlink() and link.resolve() == target:
        return
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"refusing to replace existing bundle file: {link}")
    link.symlink_to(target)


def prepare_bundle(
    *,
    checkpoint: str | Path,
    qwen_model: str | Path,
    output: str | Path,
    slm_root: str | Path,
    id_mapping: str | Path,
    num_beams: int = 64,
    num_return_sequences: int = 64,
    temperature: float = 1.0,
    decoding_mode: str = "stochastic_beam",
) -> Path:
    checkpoint_path = require_directory(checkpoint, "checkpoint")
    qwen_path = require_directory(qwen_model, "Qwen model")
    slm_path = require_directory(slm_root, "SLMv3 source")
    mapping_path = Path(id_mapping).expanduser().resolve()
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Official ID mapping is missing: {mapping_path}")
    output_path = Path(output).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    if decoding_mode not in {"beam", "sample", "stochastic_beam"}:
        raise ValueError("decoding_mode must be 'beam', 'sample', or 'stochastic_beam'")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if decoding_mode in {"beam", "stochastic_beam"} and not 1 <= num_return_sequences <= num_beams:
        raise ValueError("Require 1 <= num_return_sequences <= num_beams")

    with (qwen_path / "config.json").open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    with (checkpoint_path / "config.json").open(encoding="utf-8") as config_file:
        gr_config = json.load(config_file)

    config["architectures"] = ["GRQwen3ForDocIDGeneration"]
    config["gr_config"] = {
        **gr_config,
        "docid_size": 4,
        "docid_dim": 256,
        "temperature": temperature,
        "slm_root": str(slm_path),
        "id_mapping": str(mapping_path),
        "num_beams": num_beams,
        "num_return_sequences": num_return_sequences,
        "decoding_mode": decoding_mode,
    }
    (output_path / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for filename in _QWEN_ASSETS:
        source = qwen_path / filename
        if not source.is_file():
            raise FileNotFoundError(f"Qwen tokenizer asset is missing: {source}")
        _replace_symlink(output_path / filename, source)
    for filename in ("chat_template.jinja", "chat_template.json"):
        source = qwen_path / filename
        if source.is_file():
            _replace_symlink(output_path / filename, source)

    index_source = checkpoint_path / "pytorch_model.bin.index.json"
    if not index_source.is_file():
        raise FileNotFoundError(f"checkpoint index is missing: {index_source}")
    _replace_symlink(output_path / index_source.name, index_source)

    with index_source.open(encoding="utf-8") as index_file:
        weight_map = json.load(index_file).get("weight_map", {})
    if not weight_map:
        raise ValueError(f"checkpoint index has no weight_map: {index_source}")
    for filename in sorted(set(weight_map.values())):
        source = checkpoint_path / filename
        if not source.is_file():
            raise FileNotFoundError(f"checkpoint shard is missing: {source}")
        _replace_symlink(output_path / filename, source)

    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--qwen-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--slm-root", required=True)
    parser.add_argument("--id-mapping", required=True)
    parser.add_argument("--num-beams", type=int, default=64)
    parser.add_argument("--num-return-sequences", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--decoding-mode",
        choices=("beam", "sample", "stochastic_beam"),
        default="stochastic_beam",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = prepare_bundle(
        checkpoint=args.checkpoint,
        qwen_model=args.qwen_model,
        output=args.output,
        slm_root=args.slm_root,
        id_mapping=args.id_mapping,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        temperature=args.temperature,
        decoding_mode=args.decoding_mode,
    )
    print(os.fspath(output))


if __name__ == "__main__":
    main()
