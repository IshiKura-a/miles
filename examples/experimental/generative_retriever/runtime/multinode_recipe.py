"""Recipe parameters for paired-node asynchronous GRPO training."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

from examples.experimental.generative_retriever.runtime.training_common import (
    ScriptArgs, build_algorithm_args, build_optimizer_args, build_sglang_args,
)
from examples.experimental.generative_retriever.reward.relevance_reward import (
    DEFAULT_TRAIN_GROUP_SIZE,
    PROMPT_DATASET_ROWS,
)
from examples.experimental.generative_retriever.runtime.prepare_rocm_env import (
    derive_reward_cache_sizes,
    derive_sglang_mem_fraction,
)

_REPO = Path(__file__).absolute().parents[4]

# Each pair has one rollout node and one learner node.
REWARD_GPUS_PER_NODE = 8
_LEARNER_GPUS_PER_NODE = 8
ROLLOUT_GPUS_PER_PAIR = 16
REWARD_NODES_PER_PAIR = 2

# Batch size and step limit scale with the number of node pairs.
QUERIES_PER_STEP_PER_PAIR = 16_000
IN_FLIGHT_GROUPS_PER_PAIR = 512
SAMPLES_PER_PROMPT = DEFAULT_TRAIN_GROUP_SIZE
MAX_STEPS_PER_PAIR = PROMPT_DATASET_ROWS // QUERIES_PER_STEP_PER_PAIR

# SGLang capacity per pair.
_SGLANG_CONCURRENCY_PER_PAIR = 16_384


def build_multinode_recipe(run_id: str, pairs: int, num_rollout: int) -> ScriptArgs:
    if not run_id or "/" in run_id or run_id in {".", ".."}:
        raise ValueError("run-id must be a nonempty directory name without slashes")
    if pairs <= 0:
        raise ValueError("pairs must be positive")
    max_rollouts = MAX_STEPS_PER_PAIR // pairs
    if max_rollouts < 1 or not 1 <= num_rollout <= max_rollouts:
        raise ValueError(f"num-rollout must be between 1 and {max_rollouts}")
    query_cache, kw_cache, score_cache = derive_reward_cache_sizes(
        QUERIES_PER_STEP_PER_PAIR * pairs,
        SAMPLES_PER_PROMPT,
        REWARD_NODES_PER_PAIR * pairs,
    )
    mem_fraction = derive_sglang_mem_fraction(
        _SGLANG_CONCURRENCY_PER_PAIR * pairs,
        SAMPLES_PER_PROMPT,
        ROLLOUT_GPUS_PER_PAIR * pairs,
    )
    return replace(
        ScriptArgs(),
        run_id=run_id,
        num_rollout=num_rollout,
        rollout_batch_size=QUERIES_PER_STEP_PER_PAIR * pairs,
        rollout_max_in_flight_groups=IN_FLIGHT_GROUPS_PER_PAIR * pairs,
        n_samples_per_prompt=SAMPLES_PER_PROMPT,
        global_batch_size=QUERIES_PER_STEP_PER_PAIR * SAMPLES_PER_PROMPT * pairs,
        actor_num_gpus_per_node=_LEARNER_GPUS_PER_NODE,
        rollout_num_gpus=ROLLOUT_GPUS_PER_PAIR * pairs,
        rollout_num_gpus_per_engine=1,
        sglang_dp_size=1,
        micro_batch_size=10 * SAMPLES_PER_PROMPT,
        max_tokens_per_gpu=262144,
        sglang_mem_fraction_static=mem_fraction,
        sglang_server_concurrency=_SGLANG_CONCURRENCY_PER_PAIR * pairs,
        reward_timeout=1800,
        reward_request_concurrency=2 * REWARD_NODES_PER_PAIR * pairs,
        prepare_concurrency=2 * REWARD_NODES_PER_PAIR * pairs,
        reward_query_cache_size=query_cache,
        reward_keyword_cache_size=kw_cache,
        reward_score_cache_size=score_cache,
    )


def build_multinode_train_args(
    args: ScriptArgs, *, addresses: list[str], bundle: Path, prompt: Path,
    checkpoint_dir: Path, resume: bool,
) -> list[str]:
    blocks = [
        [
            "--hf-checkpoint", str(bundle),
            "--train-backend", "fsdp",
            "--custom-fsdp-model-factory-path",
            "examples.experimental.generative_retriever.model.fsdp_policy.build_gr_stage5_policy",
            "--data-source-path",
            "examples.experimental.generative_retriever.index.data_source.GRRolloutDataSource",
            "--custom-weight-sync-function-path",
            "examples.experimental.generative_retriever.index.data_source.sync_mapping_on_weight_update",
            "--weight-sync-include-prefixes",
            "encoder_list.0.model.encoder.model.", "docEnc2decoder.", "decoder.", "lm_head.",
            "--update-weight-buffer-size", "2147483648",
            "--micro-batch-size", str(args.micro_batch_size),
        ],
        [
            "--prompt-data", str(prompt),
            "--input-key", "query",
            "--rollout-shuffle",
            "--custom-generate-function-path",
            "examples.experimental.generative_retriever.model.sglang_encode_generate.generate",
            "--custom-rm-path", "examples.experimental.generative_retriever.reward.reward.reward",
            "--custom-rollout-log-function-path",
            "examples.experimental.generative_retriever.reward.rollout_metrics.log_rollout",
            "--custom-convert-samples-to-train-data-path",
            "examples.experimental.generative_retriever.reward.rollout_metrics.convert_gr_samples_to_train_data",
            "--generative-retriever-reward-mode", args.reward_mode,
            "--num-rollout", str(args.num_rollout),
            "--rollout-batch-size", str(args.rollout_batch_size),
            "--rollout-max-in-flight-groups", str(args.rollout_max_in_flight_groups),
            "--n-samples-per-prompt", str(args.n_samples_per_prompt),
            "--rollout-max-response-len", "4",
            "--rollout-temperature", "1.0",
            "--rollout-top-p", "1",
            "--global-batch-size", str(args.global_batch_size),
        ],
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
        ] + build_sglang_args(args, args.sglang_mem_fraction_static),
        [
            "--save", str(checkpoint_dir),
            *(["--load", str(checkpoint_dir)] if resume else []),
            "--save-interval", str(args.save_interval),
            "--use-wandb",
            "--wandb-project", "miles-run_grpo",
            "--wandb-group", args.run_id,
            "--disable-wandb-random-suffix",
        ],
    ]
    return [item for block in blocks for item in block]


def build_multinode_runtime_env(
    *, checkpoint: str, qwen: str, slm_root: str, mapping: str,
    relevance_model: str, reward_mode: str, addresses: list[str],
) -> dict[str, str]:
    inherited = (
        "ROCM_HOME", "ROCM_PATH", "PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH",
        "CPATH", "CMAKE_PREFIX_PATH", "HSA_NO_SCRATCH_RECLAIM", "PYTORCH_ROCM_ARCH",
        "GPU_ARCHS", "GPU_ARCH", "AMDGPU_TARGET", "AITER_USE_SYSTEM_TRITON",
        "NCCL_SOCKET_IFNAME", "NCCL_IB_HCA", "NCCL_IB_TIMEOUT",
        "NCCL_IB_PCI_RELAXED_ORDERING", "NCCL_NET_GDR_LEVEL", "cosmos_data_path",
        "WANDB_MODE",
    )
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
        "SGLANG_EXTERNAL_MODEL_PACKAGE":
            "examples.experimental.generative_retriever.sglang_models",
        "SGLANG_USE_AITER": "0",
        "MILES_HARDWARE_PLATFORM": "rocm",
        "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "8"),
        "MASTER_ADDR": addresses[0],
        "PYTHONPATH": f"{_REPO}:{os.environ.get('PYTHONPATH', '')}",
        "no_proxy": ",".join(("127.0.0.1", "localhost", *addresses)),
        "NO_PROXY": ",".join(("127.0.0.1", "localhost", *addresses)),
    }
