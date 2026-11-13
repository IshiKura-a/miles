"""Single-node eight-GPU GRPO launcher for SLMv3 generative retrieval."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from ray._private import ray_constants
from ray._private.node import Node
from ray._private.parameter import RayParams
import torch
import typer

_MILES_REPO_ROOT = Path(__file__).absolute().parents[3]
if str(_MILES_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_MILES_REPO_ROOT))

from examples.experimental.generative_retriever.index.id_text_mapping import is_columnar_index_dir
from examples.experimental.generative_retriever.model.model_loading import stage_local_model_dir
from examples.experimental.generative_retriever.reward.relevance_model import (
    _stage_local_relevance_model,
    _stage_local_relevance_source,
    relevance_source_path,
)
from examples.experimental.generative_retriever.reward.relevance_reward import REWARD_MODES
from examples.experimental.generative_retriever.runtime.prepare_rocm_env import (
    derive_sglang_mem_fraction,
    is_rocm,
    locked_accelerator_env,
)
from examples.experimental.generative_retriever.runtime.prepare_sglang_bundle import require_sglang_abort_patch
from examples.experimental.generative_retriever.runtime.training_common import (
    ScriptArgs,
    build_algorithm_args,
    build_optimizer_args,
    build_sglang_args,
    expanded_path,
    prepare_prompt_data,
    required_directory,
)
from examples.experimental.generative_retriever.sglang_models.gr_policy import _stage_local_mapping_dir
from examples.experimental.generative_retriever.utils.paths import absolute_path
import miles.utils.external_utils.command_utils as U
from miles.utils.misc import get_current_node_ip


def _validate_gpu_topology(args: ScriptArgs) -> None:
    if args.num_gpus_per_node != 8:
        raise ValueError("Generative retriever requires exactly eight physical GPUs")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible != "0,1,2,3,4,5,6,7":
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must be unset or exactly 0,1,2,3,4,5,6,7 before launch"
        )
    if torch.cuda.device_count() != 8:
        raise RuntimeError(f"Expected eight visible GPUs, found {torch.cuda.device_count()}")
    if args.actor_num_gpus_per_node != args.num_gpus_per_node or args.rollout_num_gpus != args.num_gpus_per_node:
        raise ValueError("Colocated single-node training requires all 8 GPUs for both rollout and learner")


def execute(args: ScriptArgs) -> None:
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "WANDB_API_KEY must be provided through the process environment "
            "for generative retriever training"
        )
    require_sglang_abort_patch()
    wandb_args = (
        "--use-wandb "
        "--wandb-project miles-run_grpo "
        f"--wandb-group {shlex.quote(args.run_id)} "
        "--disable-wandb-random-suffix "
    )
    if args.reward_mode not in REWARD_MODES:
        raise ValueError(f"Unknown reward mode: {args.reward_mode!r}")
    _validate_gpu_topology(args)
    checkpoint = required_directory(args.checkpoint, "checkpoint")
    qwen_model = required_directory(args.qwen_model, "Qwen model")
    slm_root = required_directory(args.slm_root, "SLMv3 source")
    relevance_model = required_directory(args.relevance_model, "relevance model")
    mapping = absolute_path(os.path.expandvars(args.id_mapping))
    if not is_columnar_index_dir(mapping):
        raise FileNotFoundError(f"Incomplete official mapping index: {mapping}")
    if args.temperature <= 0:
        raise ValueError("Training sampling temperature must be positive")
    if args.micro_batch_size <= 0 or args.micro_batch_size % args.n_samples_per_prompt != 0:
        raise ValueError("micro_batch_size must be a positive multiple of n_samples_per_prompt")
    if args.rollout_max_in_flight_groups <= 0:
        raise ValueError("rollout_max_in_flight_groups must be positive")
    if args.sglang_server_concurrency < args.n_samples_per_prompt:
        raise ValueError("SGLang concurrency must fit at least one complete query group")
    total_rollout_samples = args.rollout_batch_size * args.n_samples_per_prompt
    if args.global_batch_size <= 0 or total_rollout_samples % args.global_batch_size != 0:
        raise ValueError(
            "rollout_batch_size * n_samples_per_prompt must be a positive multiple of global_batch_size"
        )
    if args.save_interval <= 0:
        raise ValueError("save_interval must be positive")
    prompt_source = absolute_path(os.path.expandvars(args.prompt_data))
    if not prompt_source.is_file():
        raise FileNotFoundError(f"prompt data does not exist: {prompt_source}")
    if args.rollout_num_gpus % args.rollout_num_gpus_per_engine:
        raise ValueError("rollout_num_gpus must be divisible by rollout_num_gpus_per_engine")

    output_dir = expanded_path(args.output_dir, "output_dir", directory=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / args.run_id / "checkpoints"
    if args.resume:
        tracker = checkpoint_dir / "latest_checkpointed_iteration.txt"
        if not tracker.is_file():
            raise FileNotFoundError(f"GR resume checkpoint tracker is missing: {tracker}")
        iteration = int(tracker.read_text().strip())
        if iteration <= 0 or not (
            checkpoint_dir / f"iter_{iteration:07d}" / "meta.json"
        ).is_file():
            raise ValueError(f"GR resume checkpoint is incomplete at iteration {iteration}")
        if not (
            checkpoint_dir / "rollout" / f"global_dataset_state_dict_{iteration - 1}.pt"
        ).is_file():
            raise FileNotFoundError("GR resume requires the matching rollout dataset state")
    prompt_data = prepare_prompt_data(
        prompt_source,
        output_dir / args.run_id / "prompt_data.jsonl",
    )
    required_queries = args.num_rollout * args.rollout_batch_size
    if required_queries > prompt_data.sample_count:
        raise ValueError(
            "Fixed-step training must not reuse dataset rows implicitly: "
            f"num_rollout * rollout_batch_size requires {required_queries} queries, "
            f"but the dataset contains {prompt_data.sample_count}. "
            "Use an explicit epoch-based recipe to repeat the dataset."
        )
    local_ckpt = str(stage_local_model_dir(checkpoint, "checkpoint"))
    local_qwen = str(stage_local_model_dir(qwen_model, "Qwen model"))
    local_mapping = str(_stage_local_mapping_dir(mapping))
    _stage_local_relevance_source(relevance_source_path(relevance_model))
    local_relevance = _stage_local_relevance_model(relevance_model)
    bundle = Path(f"/tmp/gr_sglang_bundle_{args.run_id}")
    quote = shlex.quote
    U.exec_command_cpu(
        " ".join(
            (
                f"{quote(sys.executable)} -m examples.experimental.generative_retriever.runtime.prepare_sglang_bundle",
                f"--checkpoint {quote(local_ckpt)}",
                f"--qwen-model {quote(local_qwen)}",
                f"--slm-root {quote(slm_root)}",
                f"--id-mapping {quote(local_mapping)}",
                f"--relevance-model {quote(local_relevance)}",
                f"--reward-mode {quote(args.reward_mode)}",
                f"--num-beams {args.n_samples_per_prompt}",
                f"--num-return-sequences {args.n_samples_per_prompt}",
                f"--output {quote(os.fspath(bundle))}",
                f"--temperature {args.temperature}",
                "--decoding-mode beam_sample",
            )
        )
    )

    optimizer_args = shlex.join(build_optimizer_args(args.lr)) + " "
    grpo_args = shlex.join(build_algorithm_args()) + " "
    mem_fraction = (
        args.sglang_mem_fraction_static
        if args.sglang_mem_fraction_static > 0
        else derive_sglang_mem_fraction(
            args.sglang_server_concurrency,
            args.n_samples_per_prompt,
            args.rollout_num_gpus,
        )
    )
    sglang_args = shlex.join(build_sglang_args(args, mem_fraction)) + " "
    learner_args = (
        "--train-backend fsdp "
        "--data-source-path "
        "examples.experimental.generative_retriever.index.data_source.GRRolloutDataSource "
        "--custom-weight-sync-function-path "
        "examples.experimental.generative_retriever.index.data_source.sync_mapping_on_weight_update "
        "--custom-fsdp-model-factory-path "
        "examples.experimental.generative_retriever.model.fsdp_policy.build_gr_stage5_policy "
        "--weight-sync-include-prefixes "
        "encoder_list.0.model.encoder.model. docEnc2decoder. decoder. lm_head. "
        "--update-weight-buffer-size 2147483648 "
        f"--micro-batch-size {args.micro_batch_size} "
    )
    checkpoint_args = (
        f"--save {quote(os.fspath(checkpoint_dir))} "
        f"{f'--load {quote(os.fspath(checkpoint_dir))} ' if args.resume else ''}"
        f"--save-interval {args.save_interval} "
    )
    misc_args = (
        "--colocate "
        "--no-offload-train "
        "--no-offload-rollout "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.actor_num_gpus_per_node} "
    )
    rollout_args = (
        f"--prompt-data {quote(os.fspath(prompt_data.path))} "
        f"--input-key {quote(prompt_data.input_key)} "
        f"{f'--label-key {quote(prompt_data.label_key)} ' if prompt_data.label_key else ''}"
        f"{f'--metadata-key {quote(prompt_data.metadata_key)} ' if prompt_data.metadata_key else ''}"
        "--rollout-shuffle "
        "--custom-generate-function-path "
        "examples.experimental.generative_retriever.model.sglang_encode_generate.generate "
        "--custom-rm-path examples.experimental.generative_retriever.reward.reward.reward "
        "--custom-rollout-log-function-path "
        "examples.experimental.generative_retriever.reward.rollout_metrics.log_rollout "
        "--custom-convert-samples-to-train-data-path "
        "examples.experimental.generative_retriever.reward.rollout_metrics.convert_gr_samples_to_train_data "
        f"--generative-retriever-reward-mode {args.reward_mode} "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--rollout-max-in-flight-groups {args.rollout_max_in_flight_groups} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        "--rollout-max-response-len 4 "
        "--rollout-temperature 1.0 "
        "--rollout-top-p 1 "
        f"--global-batch-size {args.global_batch_size} "
    )
    train_args = (
        f"--hf-checkpoint {quote(os.fspath(bundle))} "
        f"{rollout_args}"
        f"{optimizer_args}"
        f"{grpo_args}"
        f"{sglang_args}"
        f"{learner_args}"
        f"{checkpoint_args}"
        f"{misc_args}"
        f"{wandb_args}"
        f"{args.extra_args}"
    )
    original_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    original_host_ip = os.environ.get("MILES_HOST_IP")
    node_ip = get_current_node_ip()
    cuda_env = locked_accelerator_env()
    os.environ["MILES_HOST_IP"] = node_ip
    no_proxy_hosts = f"127.0.0.1,localhost,{node_ip},{os.environ.get('MASTER_ADDR', 'node-0')}"
    extra_env = {
        "GR_CHECKPOINT": local_ckpt,
        "GR_QWEN_MODEL": local_qwen,
        "GR_SLM_ROOT": slm_root,
        "GR_ID_MAPPING": local_mapping,
        "GR_RELEVANCE_MODEL": local_relevance,
        "GR_REWARD_MODE": args.reward_mode,
        "SGLANG_EXTERNAL_MODEL_PACKAGE": (
            "examples.experimental.generative_retriever.sglang_models"
        ),
        "SGLANG_USE_AITER": "0",
        "MILES_HOST_IP": node_ip,
        "MASTER_ADDR": node_ip,
        "PYTHONPATH": f"{_MILES_REPO_ROOT}:" + ":".join(
            p for p in os.environ.get("PYTHONPATH", "").split(":")
            if p and "/opt/venv" not in p
        ),
        "no_proxy": no_proxy_hosts,
        "NO_PROXY": no_proxy_hosts,
        **cuda_env,
    }
    try:
        if is_rocm():
            os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
            ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False
            os.environ.update(extra_env)
            ray_port = 16389
            ray_temp = f"/tmp/gr_1node_{hashlib.sha256(args.run_id.encode()).hexdigest()[:10]}"
            head = Node(
                RayParams(
                    node_ip_address=node_ip,
                    gcs_server_port=ray_port,
                    num_cpus=64,
                    num_gpus=args.num_gpus_per_node,
                    object_store_memory=128 * 2**30,
                    temp_dir=ray_temp,
                    include_dashboard=False,
                ),
                head=True,
                shutdown_at_exit=True,
                spawn_reaper=True,
            )
            try:
                driver_code = "\n".join((
                    "import asyncio, json, os, sys, ray",
                    "from miles.utils.arguments import parse_args",
                    "from miles.utils.tracking_utils.tracking import finish_tracking",
                    "from train import train",
                    "ray.init(address=os.environ['RAY_ADDRESS'], "
                    "runtime_env={'env_vars': json.loads(os.environ['GR_RAY_RUNTIME_ENV_JSON'])})",
                    "args = parse_args()",
                    "try:",
                    "    asyncio.run(train(args))",
                    "finally:",
                    "    finish_tracking()",
                    "    ray.shutdown()",
                ))
                train_argv = [
                    sys.executable,
                    "-u",
                    "-c",
                    driver_code,
                    "--num-gpus-per-node",
                    str(args.num_gpus_per_node),
                    *shlex.split(train_args),
                ]
                child_env = {
                    **os.environ,
                    **extra_env,
                    "RAY_ADDRESS": f"{node_ip}:{ray_port}",
                    "GR_RAY_RUNTIME_ENV_JSON": json.dumps(extra_env),
                }
                log_dir = Path(f"/tmp/gr_logs_{args.run_id}")
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / "training.log"
                with log_file.open("a", encoding="utf-8") as log_stream:
                    proc = subprocess.Popen(
                        train_argv,
                        cwd=_MILES_REPO_ROOT,
                        env=child_env,
                        stdout=log_stream,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    print(
                        f"Started 8-GPU GRPO training pid={proc.pid}, log={log_file}",
                        flush=True,
                    )
                    ret = proc.wait()
                    if ret != 0:
                        raise RuntimeError(f"train.py exited with code {ret}; see {log_file}")
            finally:
                head.kill_all_processes(check_alive=False, allow_graceful=True, wait=True)
        else:
            U.execute_train(
                train_args=train_args,
                num_gpus_per_node=args.num_gpus_per_node,
                megatron_model_type=None,
                train_script="train.py",
                extra_env_vars=extra_env,
            )
    finally:
        if original_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_visible
        if original_host_ip is None:
            os.environ.pop("MILES_HOST_IP", None)
        else:
            os.environ["MILES_HOST_IP"] = original_host_ip


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    execute(args)


if __name__ == "__main__":
    typer.run(main)
