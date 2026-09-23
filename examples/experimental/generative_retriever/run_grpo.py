"""Single-node ROCm GRPO launcher for SLMv3 generative retrieval."""

from __future__ import annotations

import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import typer

_MILES_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_MILES_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_MILES_REPO_ROOT))

import miles.utils.external_utils.command_utils as U


def _required_directory(value: str, name: str) -> str:
    if not value:
        raise ValueError(f"{name} is required")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{name} directory does not exist: {path}")
    return str(path)


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    checkpoint: str = ""
    qwen_model: str = ""
    slm_root: str = ""
    prompt_data: str = ""
    output_dir: str = "/tmp/gr-rl"
    reward_url: str = ""
    num_gpus_per_node: int = 8
    actor_num_gpus_per_node: int = 4
    rollout_num_gpus: int = 4
    rollout_num_gpus_per_engine: int = 1
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    global_batch_size: int = 256
    num_rollout: int = 1000
    temperature: float = 1.0
    lr: float = 1e-6
    extra_args: str = ""


def execute(args: ScriptArgs) -> None:
    checkpoint = _required_directory(args.checkpoint, "checkpoint")
    qwen_model = _required_directory(args.qwen_model, "Qwen model")
    slm_root = _required_directory(args.slm_root, "SLMv3 source")
    prompt_data = Path(args.prompt_data).expanduser().resolve()
    if not prompt_data.is_file():
        raise FileNotFoundError(f"prompt data does not exist: {prompt_data}")
    if not args.reward_url:
        raise ValueError("reward_url is required until an in-process retrieval reward is available")
    if args.actor_num_gpus_per_node + args.rollout_num_gpus > args.num_gpus_per_node:
        raise ValueError(
            "actor_num_gpus_per_node + rollout_num_gpus cannot exceed num_gpus_per_node"
        )
    if args.rollout_num_gpus % args.rollout_num_gpus_per_engine:
        raise ValueError("rollout_num_gpus must be divisible by rollout_num_gpus_per_engine")

    bundle = Path(args.output_dir).expanduser().resolve() / args.run_id / "sglang_bundle"
    quote = shlex.quote
    U.exec_command_cpu(
        " ".join(
            (
                "python -m examples.experimental.generative_retriever.prepare_sglang_bundle",
                f"--checkpoint {quote(checkpoint)}",
                f"--qwen-model {quote(qwen_model)}",
                f"--output {quote(os.fspath(bundle))}",
                f"--temperature {args.temperature}",
            )
        )
    )

    rollout_args = (
        f"--prompt-data {quote(os.fspath(prompt_data))} "
        "--input-key prompt "
        "--label-key label "
        "--metadata-key metadata "
        "--rollout-shuffle "
        "--custom-generate-function-path "
        "examples.experimental.generative_retriever.sglang_encode_generate.generate "
        "--custom-rm-path examples.experimental.generative_retriever.reward.reward "
        f"--generative-retriever-reward-url {quote(args.reward_url)} "
        "--generative-retriever-docid-size 4 "
        "--generative-retriever-docid-dim 256 "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        "--rollout-max-response-len 4 "
        f"--rollout-temperature {args.temperature} "
        "--rollout-top-p 1 "
        f"--global-batch-size {args.global_batch_size} "
    )
    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.lr} "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )
    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-coef 0 "
        "--kl-loss-coef 0 "
        "--entropy-coef 0 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        "--use-rollout-logprobs "
    )
    sglang_args = (
        f"--rollout-num-gpus {args.rollout_num_gpus} "
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        "--sglang-is-embedding "
        "--sglang-disable-radix-cache "
        "--sglang-attention-backend triton "
        "--sglang-disable-cuda-graph "
        "--sglang-mem-fraction-static 0.7 "
    )
    learner_args = (
        "--train-backend fsdp "
        "--custom-fsdp-model-factory-path "
        "examples.experimental.generative_retriever.fsdp_policy.build_gr_stage5_policy "
        "--weight-sync-include-prefixes "
        "docEnc2decoder. decoder. lm_head. "
        "--update-weight-buffer-size 536870912 "
    )
    misc_args = (
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.actor_num_gpus_per_node} "
    )
    train_args = (
        f"--hf-checkpoint {quote(os.fspath(bundle))} "
        f"{rollout_args}"
        f"{optimizer_args}"
        f"{grpo_args}"
        f"{sglang_args}"
        f"{learner_args}"
        f"{misc_args}"
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{args.extra_args}"
    )
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=None,
        extra_env_vars={
            "GR_CHECKPOINT": checkpoint,
            "GR_QWEN_MODEL": qwen_model,
            "GR_SLM_ROOT": slm_root,
            "SGLANG_EXTERNAL_MODEL_PACKAGE": (
                "examples.experimental.generative_retriever.sglang_models"
            ),
            "SGLANG_USE_AITER": "0",
            **{
                key: os.environ[key]
                for key in (
                    "PYTHONPATH",
                    "CUDA_VISIBLE_DEVICES",
                    "HIP_VISIBLE_DEVICES",
                    "ROCR_VISIBLE_DEVICES",
                    "MILES_HOST_IP",
                )
                if key in os.environ
            },
        },
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    execute(args)


if __name__ == "__main__":
    typer.run(main)
