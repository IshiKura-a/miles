"""Shared configuration and arguments for GRPO launchers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

from examples.experimental.generative_retriever.reward.relevance_reward import (
    DEFAULT_TRAIN_GROUP_SIZE,
    PROMPT_DATASET_ROWS,
    RewardMode,
)
from examples.experimental.generative_retriever.runtime.gr_config import (
    AntiCollapseMode,
    DEFAULT_HBM_KW_MEMORY_FRACTION,
    DEFAULT_SGLANG_WATCHDOG_TIMEOUT_SECONDS,
    Doc2kMode,
    GRConfig,
    MIN_ACTIVE_ENTROPY_COEF,
    UniqueAnnealMode,
    parse_step_temperatures,
)
from examples.experimental.generative_retriever.utils.paths import absolute_path, symlink_target
import miles.utils.external_utils.command_utils as U

_QUERIES_PER_STEP = 4000
_GLOBAL_BATCH_SIZE = _QUERIES_PER_STEP * DEFAULT_TRAIN_GROUP_SIZE
_MAX_STEPS = PROMPT_DATASET_ROWS // _QUERIES_PER_STEP
_IN_FLIGHT_GROUPS = 512


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
    micro_batch_size: int = 50 * DEFAULT_TRAIN_GROUP_SIZE
    max_tokens_per_gpu: int = 131072
    num_rollout: int = _MAX_STEPS
    temperature: float = 0.4
    step_temperatures: str = "0.2,0.25,0.6,0.8"
    doc2k_mode: Doc2kMode = "relevance_guided"
    unique_anneal_mode: UniqueAnnealMode = "cosine"
    anti_collapse_mode: AntiCollapseMode = "none"
    unique_weight_start: float = 2.0
    unique_weight_end: float = 0.0
    unique_anneal_warmup_ratio: float = 0.0
    local_rollout_batch_size: int = 125
    tree_query_chunk_size: int = 128
    hbm_kw_memory_fraction: float = DEFAULT_HBM_KW_MEMORY_FRACTION
    lr: float = 2e-6
    save_interval: int = 5
    resume: bool = False
    extra_args: str = ""

    def build_gr_config(self) -> GRConfig:
        return GRConfig(
            reward_mode=self.reward_mode,
            step_temperatures=parse_step_temperatures(self.step_temperatures),
            doc2k_mode=self.doc2k_mode,
            unique_anneal_mode=self.unique_anneal_mode,
            anti_collapse_mode=self.anti_collapse_mode,
            unique_weight_start=self.unique_weight_start,
            unique_weight_end=self.unique_weight_end,
            unique_anneal_warmup_ratio=self.unique_anneal_warmup_ratio,
            local_rollout_batch_size=self.local_rollout_batch_size,
            tree_query_chunk_size=self.tree_query_chunk_size,
            hbm_kw_memory_fraction=self.hbm_kw_memory_fraction,
        )


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


def build_algorithm_args(anti_collapse_mode: AntiCollapseMode = "none") -> list[str]:
    entropy_coef = str(MIN_ACTIVE_ENTROPY_COEF) if anti_collapse_mode == "target_entropy" else "0"
    return [
        "--advantage-estimator", "grpo", "--kl-coef", "0", "--kl-loss-coef", "0",
        "--entropy-coef", entropy_coef, "--observe-training-entropy",
        "--eps-clip", "0.2", "--eps-clip-high", "0.28",
        "--eps-clip-c", "3.0",
        "--use-rollout-logprobs", "--skip-actor-forward-only",
    ]


def build_sglang_args(
    args: ScriptArgs,
    mem_fraction: float,
    *,
    colocate: bool = False,
) -> list[str]:
    return [
        *([] if colocate else ["--rollout-num-gpus", str(args.rollout_num_gpus)]),
        "--rollout-num-gpus-per-engine", str(args.rollout_num_gpus_per_engine),
        "--sglang-dp-size", str(args.sglang_dp_size),
        *(["--sglang-enable-dp-attention"] if args.sglang_dp_size > 1 else []),
        "--sglang-is-embedding", "--sglang-disable-radix-cache",
        "--sglang-attention-backend", "triton", "--sglang-disable-cuda-graph",
        "--sglang-server-concurrency", str(args.sglang_server_concurrency),
        "--sglang-mem-fraction-static", str(mem_fraction),
        "--sglang-watchdog-timeout", str(DEFAULT_SGLANG_WATCHDOG_TIMEOUT_SECONDS),
    ]
