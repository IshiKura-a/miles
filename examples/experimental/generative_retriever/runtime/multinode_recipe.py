"""Recipe parameters for paired-node asynchronous GRPO training."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

from examples.experimental.generative_retriever.reward.relevance_reward import (
    DEFAULT_TRAIN_GROUP_SIZE,
    PROMPT_DATASET_ROWS,
)
from examples.experimental.generative_retriever.runtime.gr_config import (
    GRConfig,
    RAY_GRPC_KEEPALIVE_ENV,
)
from examples.experimental.generative_retriever.runtime.prepare_rocm_env import (
    derive_sglang_mem_fraction,
)
from examples.experimental.generative_retriever.runtime.training_common import (
    ScriptArgs,
    build_algorithm_args,
    build_checkpoint_args,
    build_learner_args,
    build_optimizer_args,
    build_rollout_args,
    build_sglang_args,
    build_wandb_args,
    resolve_rollout_steps,
)

_REPO = Path(__file__).absolute().parents[4]

REWARD_GPUS_PER_NODE = 8
_LEARNER_GPUS_PER_NODE = 8
ROLLOUT_GPUS_PER_PAIR = 16
REWARD_NODES_PER_PAIR = 2

QUERIES_PER_STEP_PER_PAIR = 16_000
IN_FLIGHT_GROUPS_PER_PAIR = 512
SAMPLES_PER_PROMPT = DEFAULT_TRAIN_GROUP_SIZE
MAX_STEPS_PER_PAIR = PROMPT_DATASET_ROWS // QUERIES_PER_STEP_PER_PAIR
_SGLANG_CONCURRENCY_PER_PAIR = 16_384


def steps_per_epoch(pairs: int) -> int:
    if pairs <= 0:
        raise ValueError("pairs must be positive")
    steps = MAX_STEPS_PER_PAIR // pairs
    if steps < 1:
        raise ValueError(f"Prompt dataset too small for {pairs} node pairs")
    return steps


def build_multinode_recipe(
    run_id: str,
    pairs: int,
    num_rollout: int,
    *,
    base_args: ScriptArgs | None = None,
) -> ScriptArgs:
    if not run_id or "/" in run_id or run_id in {".", ".."}:
        raise ValueError("run-id must be a nonempty directory name without slashes")
    resolved_rollout = resolve_rollout_steps(
        steps_per_epoch=steps_per_epoch(pairs),
        num_rollout=num_rollout,
    )

    template = ScriptArgs() if base_args is None else base_args
    mem_fraction = (
        template.sglang_mem_fraction_static
        if template.sglang_mem_fraction_static > 0
        else min(
            derive_sglang_mem_fraction(
                _SGLANG_CONCURRENCY_PER_PAIR * pairs,
                SAMPLES_PER_PROMPT,
                ROLLOUT_GPUS_PER_PAIR * pairs,
            ),
            0.18,
        )
    )
    mbs = template.micro_batch_size
    return replace(
        template,
        run_id=run_id,
        num_rollout=resolved_rollout,
        rollout_batch_size=QUERIES_PER_STEP_PER_PAIR * pairs,
        rollout_max_in_flight_groups=IN_FLIGHT_GROUPS_PER_PAIR * pairs,
        n_samples_per_prompt=SAMPLES_PER_PROMPT,
        global_batch_size=QUERIES_PER_STEP_PER_PAIR * SAMPLES_PER_PROMPT * pairs,
        actor_num_gpus_per_node=_LEARNER_GPUS_PER_NODE,
        rollout_num_gpus=ROLLOUT_GPUS_PER_PAIR * pairs,
        rollout_num_gpus_per_engine=1,
        sglang_dp_size=1,
        micro_batch_size=mbs,
        max_tokens_per_gpu=max(262144, 6 * mbs),
        sglang_mem_fraction_static=mem_fraction,
        sglang_server_concurrency=_SGLANG_CONCURRENCY_PER_PAIR * pairs,
    )


def build_multinode_train_args(
    args: ScriptArgs,
    *,
    addresses: list[str],
    bundle: Path,
    prompt: Path,
    checkpoint_dir: Path,
    resume: bool,
) -> list[str]:
    blocks = [
        build_learner_args(bundle=bundle, micro_batch_size=args.micro_batch_size),
        build_rollout_args(args, prompt_path=prompt),
        build_optimizer_args(args.lr) + build_algorithm_args(),
        [
            "--colocate",
            "--no-offload-train",
            "--no-offload-rollout",
            "--actor-num-nodes", str(len(addresses)),
            "--actor-num-gpus-per-node", str(args.actor_num_gpus_per_node),
            "--num-gpus-per-node", str(args.num_gpus_per_node),
            "--dp-replicate-size", str(len(addresses)),
            "--router-policy", "round_robin",
        ] + build_sglang_args(args, args.sglang_mem_fraction_static, colocate=True),
        build_checkpoint_args(
            checkpoint_dir=checkpoint_dir,
            save_interval=args.save_interval,
            resume=resume,
        ) + build_wandb_args(args.run_id),
    ]
    return [item for block in blocks for item in block]


def build_multinode_runtime_env(
    *,
    checkpoint: str,
    qwen: str,
    slm_root: str,
    mapping: str,
    relevance_model: str,
    reward_mode: str,
    addresses: list[str],
    gr_config: GRConfig | None = None,
) -> dict[str, str]:
    inherited = (
        "ROCM_HOME", "ROCM_PATH", "PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH",
        "CPATH", "CMAKE_PREFIX_PATH", "HSA_NO_SCRATCH_RECLAIM", "PYTORCH_ROCM_ARCH",
        "GPU_ARCHS", "GPU_ARCH", "AMDGPU_TARGET", "AITER_USE_SYSTEM_TRITON",
        "NCCL_SOCKET_IFNAME", "NCCL_IB_HCA", "NCCL_IB_TIMEOUT",
        "NCCL_IB_PCI_RELAXED_ORDERING", "NCCL_NET_GDR_LEVEL", "cosmos_data_path",
        "WANDB_MODE",
    )
    active_cfg = GRConfig.from_env() if gr_config is None else gr_config
    return {
        **{key: os.environ[key] for key in inherited if key in os.environ},
        "GPU_MAX_HW_QUEUES": "1",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "GPU_PINNED_MIN_XFER_SIZE": "100000",
        "GR_CHECKPOINT": checkpoint,
        "GR_QWEN_MODEL": qwen,
        "GR_SLM_ROOT": slm_root,
        "GR_ID_MAPPING": mapping,
        "GR_RELEVANCE_MODEL": relevance_model,
        "GR_REWARD_MODE": reward_mode,
        **active_cfg.to_env_dict(),
        "SGLANG_EXTERNAL_MODEL_PACKAGE": "examples.experimental.generative_retriever.sglang_models",
        "SGLANG_USE_AITER": "0",
        "MILES_HARDWARE_PLATFORM": "rocm",
        "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
        **RAY_GRPC_KEEPALIVE_ENV,
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "8"),
        "MASTER_ADDR": addresses[0],
        "PYTHONPATH": f"{_REPO}:{os.environ.get('PYTHONPATH', '')}",
        "no_proxy": ",".join(("127.0.0.1", "localhost", *addresses)),
        "NO_PROXY": ",".join(("127.0.0.1", "localhost", *addresses)),
    }
