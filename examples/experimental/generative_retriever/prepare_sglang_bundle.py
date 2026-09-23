"""Create a lightweight SGLang model bundle for an SLMv3 checkpoint.

The bundle contains Qwen configuration/tokenizer files and symlinks to the GR
checkpoint shards. Model weights are not copied.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .model_loading import require_directory


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
    temperature: float = 1.0,
) -> Path:
    checkpoint_path = require_directory(checkpoint, "checkpoint")
    qwen_path = require_directory(qwen_model, "Qwen model")
    output_path = Path(output).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    if temperature < 0:
        raise ValueError("temperature must be non-negative")

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
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = prepare_bundle(
        checkpoint=args.checkpoint,
        qwen_model=args.qwen_model,
        output=args.output,
        temperature=args.temperature,
    )
    print(os.fspath(output))


if __name__ == "__main__":
    main()
