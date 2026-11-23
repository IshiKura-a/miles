"""Export an FSDP DCP checkpoint into a standalone SLMv3 checkpoint directory.

Merges the trained Qwen3-4B encoder (`encoder_list.0.model.encoder.model.*`) and
T5 generative decoder (`docEnc2decoder.*`, `decoder.*`, `lm_head.*`) weights
into the two-shard `pytorch_model-00001/00002-of-00002.bin` layout expected by
`SLMv3` (`run_mt.py`, `transfer_to_onnx_slm_and_rqvae.py`, and vLLM serving),
while synchronizing tied embedding aliases (`encoder_list.0.model.encoder.lm_head.weight`,
`docid_embedding.weight`, and `glue.embeds`).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from examples.experimental.generative_retriever.model.model_loading import (
    export_slmv3_checkpoint,
    require_directory,
)
from examples.experimental.generative_retriever.utils.paths import absolute_path

_DEFAULT_BASE_CHECKPOINT = (
    "$cosmos_data_path/local/Users/tianchiyang/Checkpoints/ASI_RQ/"
    "SLMv3-202511-4x256-stage5/epoch1-part24/checkpoint-30553"
)
_DEFAULT_QWEN_MODEL = "$cosmos_data_path/local/Users/tianchiyang/LLM/Qwen/Qwen3-4B"
_DEFAULT_RUNS_ROOT = "$cosmos_data_path/local/User/zihao/online_rl/gr_offline_syn"


def _expand_required_path(raw: str, name: str) -> Path:
    expanded = os.path.expandvars(raw)
    if "$" in expanded:
        raise ValueError(f"{name} contains an unresolved environment variable: {raw!r}")
    return absolute_path(expanded)


def _resolve_dcp_and_default_output(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.dcp_model_dir is not None:
        if args.run_id is not None or args.step is not None:
            raise ValueError("--dcp-model-dir cannot be combined with --run-id or --step")
        if args.output_dir is None:
            raise ValueError("--output-dir is required when using --dcp-model-dir")
        return (
            _expand_required_path(args.dcp_model_dir, "--dcp-model-dir"),
            _expand_required_path(args.output_dir, "--output-dir"),
        )

    if not args.run_id:
        raise ValueError("Provide either --run-id or --dcp-model-dir")
    runs_root = require_directory(_expand_required_path(args.runs_root, "--runs-root"), "runs root")
    run_dir = require_directory(runs_root / args.run_id, "training run")
    checkpoints_root = require_directory(run_dir / "checkpoints", "checkpoints")

    step = args.step
    if step is None:
        latest_file = checkpoints_root / "latest_checkpointed_iteration.txt"
        if not latest_file.is_file():
            raise FileNotFoundError(f"Missing latest checkpoint marker: {latest_file}")
        step = int(latest_file.read_text(encoding="utf-8").strip())
    if step <= 0:
        raise ValueError(f"--step must be positive, got {step}")

    dcp_dir = require_directory(checkpoints_root / f"iter_{step:07d}" / "model", "DCP model checkpoint")
    out_dir = (
        _expand_required_path(args.output_dir, "--output-dir")
        if args.output_dir is not None
        else run_dir / "exported" / f"checkpoint-{step}"
    )
    return dcp_dir, out_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None, help="Training run ID under --runs-root")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (defaults to latest)")
    parser.add_argument("--runs-root", default=_DEFAULT_RUNS_ROOT, help="Root directory containing training runs")
    parser.add_argument("--dcp-model-dir", default=None, help="Explicit path to checkpoints/iter_XXXXXXX/model")
    parser.add_argument("--base-checkpoint", default=_DEFAULT_BASE_CHECKPOINT, help="Base SLMv3 checkpoint directory")
    parser.add_argument(
        "--qwen-model",
        default=_DEFAULT_QWEN_MODEL,
        help="Qwen3-4B directory whose config/tokenizer files are bundled into <output-dir>/qwen_config",
    )
    parser.add_argument("--output-dir", default=None, help="Target directory for exported SLMv3 checkpoint")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite non-empty --output-dir if it exists")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dcp_dir, out_dir = _resolve_dcp_and_default_output(args)
    base_ckpt = _expand_required_path(args.base_checkpoint, "--base-checkpoint")
    qwen_dir = _expand_required_path(args.qwen_model, "--qwen-model") if args.qwen_model else None
    summary = export_slmv3_checkpoint(
        dcp_dir,
        base_checkpoint=base_ckpt,
        output_dir=out_dir,
        qwen_model_dir=qwen_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps({"dcp_model_dir": str(dcp_dir), "output_dir": str(out_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
