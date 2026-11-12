"""Create a lightweight SGLang model bundle for an SLMv3 checkpoint.

The bundle contains Qwen configuration/tokenizer files and symlinks to the GR
checkpoint shards. Model weights are not copied.
"""

from __future__ import annotations

from examples.experimental.generative_retriever.utils.paths import absolute_path, symlink_target

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from examples.experimental.generative_retriever.model.model_loading import require_directory
from examples.experimental.generative_retriever.index.id_text_mapping import is_columnar_index_dir
from examples.experimental.generative_retriever.reward.relevance_reward import (
    DEFAULT_TRAIN_GROUP_SIZE,
    RQ_CODE_DIM,
    RQ_CODE_NUM,
)


def require_sglang_abort_patch() -> None:
    patch = Path(__file__).absolute().parent / "patches/sglang-embedding-abort.patch"
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
    if link.is_symlink():
        if symlink_target(link) == target:
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"refusing to replace existing bundle file: {link}")
    link.symlink_to(target)


def prepare_bundle(
    *,
    checkpoint: str | Path,
    qwen_model: str | Path,
    output: str | Path,
    slm_root: str | Path,
    id_mapping: str | Path,
    relevance_model: str | Path | None = None,
    reward_mode: str = "discrete_relevance_unique",
    num_beams: int = DEFAULT_TRAIN_GROUP_SIZE,
    num_return_sequences: int = DEFAULT_TRAIN_GROUP_SIZE,
    temperature: float = 1.0,
    decoding_mode: str = "beam_sample",
) -> Path:
    checkpoint_path = require_directory(checkpoint, "checkpoint")
    qwen_path = require_directory(qwen_model, "Qwen model")
    slm_path = require_directory(slm_root, "SLMv3 source")
    mapping_path = absolute_path(id_mapping)
    if not is_columnar_index_dir(mapping_path):
        raise FileNotFoundError(f"Incomplete official mapping index: {mapping_path}")
    resolved_relevance = (
        str(require_directory(relevance_model, "relevance model"))
        if relevance_model is not None
        else os.environ.get("GR_RELEVANCE_MODEL", "")
    )
    output_path = absolute_path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    if decoding_mode not in {"beam", "beam_sample", "sample"}:
        raise ValueError("decoding_mode must be 'beam', 'beam_sample', or 'sample'")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if decoding_mode in {"beam", "beam_sample"} and not 1 <= num_return_sequences <= num_beams:
        raise ValueError("Require 1 <= num_return_sequences <= num_beams")

    with (qwen_path / "config.json").open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    with (checkpoint_path / "config.json").open(encoding="utf-8") as config_file:
        gr_config = json.load(config_file)

    config["architectures"] = ["GRQwen3ForDocIDGeneration"]
    config["gr_config"] = {
        **gr_config,
        "docid_size": RQ_CODE_NUM,
        "docid_dim": RQ_CODE_DIM,
        "temperature": temperature,
        "slm_root": str(slm_path),
        "qwen_model": str(qwen_path),
        "id_mapping": str(mapping_path),
        "relevance_model": resolved_relevance,
        "reward_mode": reward_mode,
        "num_beams": num_beams,
        "num_return_sequences": num_return_sequences,
        "decoding_mode": decoding_mode,
    }
    rendered_config = json.dumps(config, indent=2, sort_keys=True) + "\n"
    config_target = output_path / "config.json"
    if not config_target.is_file() or config_target.read_text(encoding="utf-8") != rendered_config:
        config_target.write_text(rendered_config, encoding="utf-8")

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
    parser.add_argument("--relevance-model", default=None)
    parser.add_argument("--reward-mode", default="discrete_relevance_unique")
    parser.add_argument("--num-beams", type=int, default=DEFAULT_TRAIN_GROUP_SIZE)
    parser.add_argument("--num-return-sequences", type=int, default=DEFAULT_TRAIN_GROUP_SIZE)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--decoding-mode",
        choices=("beam", "beam_sample", "sample"),
        default="beam_sample",
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
        relevance_model=args.relevance_model,
        reward_mode=args.reward_mode,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        temperature=args.temperature,
        decoding_mode=args.decoding_mode,
    )
    print(os.fspath(output))


if __name__ == "__main__":
    main()
