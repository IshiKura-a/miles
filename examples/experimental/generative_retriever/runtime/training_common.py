"""Shared configuration and arguments for GRPO launchers."""

from __future__ import annotations

from examples.experimental.generative_retriever.utils.paths import absolute_path, symlink_target

from dataclasses import dataclass
import json
import os
import tempfile
from pathlib import Path

import miles.utils.external_utils.command_utils as U
from examples.experimental.generative_retriever.reward.relevance_reward import (
    DEFAULT_REWARD_BATCH_SIZE,
    DEFAULT_TRAIN_GROUP_SIZE,
    PROMPT_DATASET_ROWS,
    RewardMode,
)

_QUERIES_PER_STEP = 4000
_GLOBAL_BATCH_SIZE = _QUERIES_PER_STEP * DEFAULT_TRAIN_GROUP_SIZE
_MAX_STEPS = PROMPT_DATASET_ROWS // _QUERIES_PER_STEP
_IN_FLIGHT_GROUPS = 512
_REWARD_DEVICES = tuple(f"cuda:{i}" for i in range(8))


def required_directory(value: str, name: str) -> str:
    if not value:
        raise ValueError(f"{name} is required")
    path = absolute_path(os.path.expandvars(value))
    if "$" in os.fspath(path):
        raise ValueError(f"{name} contains an unresolved environment variable")
    if not path.is_dir():
        raise FileNotFoundError(f"{name} directory does not exist: {path}")
    return str(path)


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    checkpoint: str = (
        "$cosmos_data_path/local/Users/tianchiyang/Checkpoints/ASI_RQ/"
        "SLMv3-202511-4x256-stage5/epoch1-part24/checkpoint-30553/"
    )
    qwen_model: str = "$cosmos_data_path/local/Users/tianchiyang/LLM/Qwen/Qwen3-4B/"
    slm_root: str = "/home/aiscuser/SLMGR_dev/SLMv3"
    id_mapping: str = (
        "$cosmos_data_path/local/User/zihao/online_rl/gr_offline_syn/index/"
        "slmgrv3_part25_na_intl_20261028_20261029"
    )
    prompt_data: str = (
        "$cosmos_data_path/local/Users/tianchiyang/Data/MZLog/"
        "TA_PA_UnifiedData_202604to202605_Weekly/TAKeyword/"
        "train_15M.queryfeat.2026-04-01/Part0.json"
    )
    output_dir: str = "$cosmos_data_path/local/User/zihao/online_rl/gr_offline_syn/"
    service_log_dir: str = "~/.local/state/miles/generative_retriever/"
    relevance_model: str = (
        "$cosmos_data_path/local/User/yugao/share/qualitymodel/"
        "GCQC/GV2.1/ForArcher/Large/"
    )
    reward_mode: RewardMode = "discrete_relevance_unique"
    num_gpus_per_node: int = 8
    actor_num_gpus_per_node: int = 8
    rollout_num_gpus: int = 8
    rollout_num_gpus_per_engine: int = 1
    sglang_dp_size: int = 1
    sglang_mem_fraction_static: float = 0.0
    sglang_server_concurrency: int = 8192
    rollout_batch_size: int = _QUERIES_PER_STEP
    rollout_max_in_flight_groups: int = _IN_FLIGHT_GROUPS
    n_samples_per_prompt: int = DEFAULT_TRAIN_GROUP_SIZE
    global_batch_size: int = _GLOBAL_BATCH_SIZE
    micro_batch_size: int = 10 * DEFAULT_TRAIN_GROUP_SIZE
    max_tokens_per_gpu: int = 131072
    reward_timeout: float = 1800.0
    reward_batch_size: int = DEFAULT_REWARD_BATCH_SIZE
    reward_request_concurrency: int = 2 * len(_REWARD_DEVICES)
    prepare_concurrency: int = 2
    reward_forward_concurrency: int = len(_REWARD_DEVICES)
    reward_query_cache_size: int = 0
    reward_keyword_cache_size: int = 0
    reward_score_cache_size: int = 0
    num_rollout: int = _MAX_STEPS
    one_step_async: bool = True
    temperature: float = 1.0
    lr: float = 2e-6
    save_interval: int = 20
    resume: bool = False
    extra_args: str = ""


def expanded_path(value: str, name: str, *, directory: bool = True) -> Path:
    path = absolute_path(os.path.expandvars(value))
    if "$" in os.fspath(path):
        raise ValueError(f"{name} contains an unresolved environment variable")
    if directory and not path.is_dir():
        raise FileNotFoundError(f"{name} directory does not exist: {path}")
    return path


@dataclass(frozen=True)
class PromptData:
    path: Path
    input_key: str
    label_key: str | None
    metadata_key: str | None
    sample_count: int


def count_records(path: Path) -> int:
    stat = path.stat()
    cache_file = Path.home() / ".local/state/miles/generative_retriever/prompt_counts.json"
    key = f"counted-v1:{path}:{stat.st_size}:{stat.st_mtime_ns}"
    if cache_file.is_file():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if not isinstance(cached, dict):
            raise ValueError(f"Invalid prompt count cache: {cache_file}")
        if key in cached:
            count = cached[key]
            if type(count) is not int or count <= 0:
                raise ValueError(f"Invalid prompt count in {cache_file}: {count!r}")
            return count
    with path.open("rb", buffering=16 * 1024 * 1024) as stream:
        count = sum(bool(line.strip()) for line in stream)
    final_stat = path.stat()
    if (final_stat.st_size, final_stat.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        raise RuntimeError(f"Prompt dataset changed while counting: {path}")
    if count == 0:
        raise ValueError(f"Prompt dataset is empty: {path}")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_file.parent) as tmp:
        temporary = Path(tmp) / cache_file.name
        temporary.write_text(json.dumps({key: count}), encoding="utf-8")
        temporary.replace(cache_file)
    return count


def prepare_prompt_data(source: Path, jsonl_alias: Path) -> PromptData:
    with source.open(encoding="utf-8") as stream:
        first = json.loads(stream.readline())
    if isinstance(first, dict) and isinstance(first.get("prompt"), str):
        if source.suffix != ".jsonl":
            raise ValueError(f"Prompt dataset with prompt fields must use a .jsonl suffix: {source}")
        return PromptData(
            path=source,
            input_key="prompt",
            label_key="label",
            metadata_key="metadata",
            sample_count=count_records(source),
        )
    if not isinstance(first, dict) or not isinstance(first.get("query"), str):
        raise ValueError(f"Prompt data must contain either prompt or query fields: {source}")
    jsonl_alias.parent.mkdir(parents=True, exist_ok=True)
    if jsonl_alias.is_symlink():
        if symlink_target(jsonl_alias) != source:
            raise FileExistsError(f"Prompt alias points to a different dataset: {jsonl_alias}")
    else:
        if jsonl_alias.exists():
            raise FileExistsError(f"Prompt alias is not a symlink: {jsonl_alias}")
        jsonl_alias.symlink_to(source)
    return PromptData(
        path=jsonl_alias,
        input_key="query",
        label_key=None,
        metadata_key=None,
        sample_count=count_records(source),
    )


def build_optimizer_args(lr: float) -> list[str]:
    return [
        "--optimizer", "adam", "--lr", str(lr), "--lr-decay-style", "constant",
        "--weight-decay", "0.1", "--adam-beta1", "0.9", "--adam-beta2", "0.98",
    ]


def build_algorithm_args() -> list[str]:
    return [
        "--advantage-estimator", "grpo", "--kl-coef", "0", "--kl-loss-coef", "0",
        "--entropy-coef", "0", "--eps-clip", "0.2", "--eps-clip-high", "0.28",
        "--use-rollout-logprobs", "--skip-actor-forward-only",
    ]


def build_sglang_args(args: ScriptArgs, mem_fraction: float) -> list[str]:
    return [
        "--rollout-num-gpus", str(args.rollout_num_gpus),
        "--rollout-num-gpus-per-engine", str(args.rollout_num_gpus_per_engine),
        "--sglang-dp-size", str(args.sglang_dp_size),
        *(["--sglang-enable-dp-attention"] if args.sglang_dp_size > 1 else []),
        "--sglang-is-embedding", "--sglang-disable-radix-cache",
        "--sglang-attention-backend", "triton", "--sglang-disable-cuda-graph",
        "--sglang-server-concurrency", str(args.sglang_server_concurrency),
        "--sglang-mem-fraction-static", str(mem_fraction),
    ]
