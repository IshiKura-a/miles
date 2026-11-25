"""Shared configuration and arguments for GRPO launchers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import time

from examples.experimental.generative_retriever.reward.relevance_reward import (
    DEFAULT_TRAIN_GROUP_SIZE,
    PROMPT_DATASET_ROWS,
    RewardMode,
)
from examples.experimental.generative_retriever.runtime.gr_config import (
    AdvantageMode,
    DEFAULT_HBM_KW_MEMORY_FRACTION,
    DEFAULT_SGLANG_WATCHDOG_TIMEOUT_SECONDS,
    Doc2kMode,
    GRConfig,
    UniqueAnnealMode,
    parse_step_temperatures,
)
from examples.experimental.generative_retriever.utils.paths import absolute_path, symlink_target
import miles.utils.external_utils.command_utils as U

_QUERIES_PER_STEP = 4000
_GLOBAL_BATCH_SIZE = _QUERIES_PER_STEP * DEFAULT_TRAIN_GROUP_SIZE
_MAX_STEPS = PROMPT_DATASET_ROWS // _QUERIES_PER_STEP
_IN_FLIGHT_GROUPS = 512
_WANDB_PROJECT = "miles-run_grpo"
_UPDATE_WEIGHT_BUFFER_BYTES = 2_147_483_648
_WEIGHT_SYNC_INCLUDE_PREFIXES: tuple[str, ...] = (
    "encoder_list.0.model.encoder.model.",
    "docEnc2decoder.",
    "decoder.",
    "lm_head.",
)


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
    epochs: int = 1
    num_rollout: int = _MAX_STEPS
    temperature: float = 1.0
    step_temperatures: str = "1.0"
    doc2k_mode: Doc2kMode = "relevance_guided"
    unique_anneal_mode: UniqueAnnealMode = "cosine"
    advantage_mode: AdvantageMode = "pos_mean_quota"
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
            advantage_mode=self.advantage_mode,
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


def resolve_rollout_steps(
    *,
    steps_per_epoch: int,
    epochs: int = 1,
    num_rollout: int | None = None,
) -> int:
    if steps_per_epoch < 1:
        raise ValueError(f"steps_per_epoch must be positive, got {steps_per_epoch}")
    if epochs < 1:
        raise ValueError(f"epochs must be positive, got {epochs}")
    resolved = steps_per_epoch * epochs if num_rollout is None else num_rollout
    if resolved < 1:
        raise ValueError(f"num_rollout must be positive, got {resolved}")
    return resolved


def validate_resume_checkpoint(checkpoint_dir: Path) -> int:
    tracker = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise FileNotFoundError(f"Resume checkpoint missing: {tracker}")
    raw = tracker.read_text(encoding="utf-8").strip()
    if not raw.isdigit():
        raise ValueError(f"Invalid checkpoint iteration in {tracker}: {raw!r}")
    iteration = int(raw)
    if iteration <= 0 or not (checkpoint_dir / f"iter_{iteration:07d}" / "meta.json").is_file():
        raise ValueError(f"Incomplete checkpoint at iteration {iteration}")
    if not (checkpoint_dir / "rollout" / f"global_dataset_state_dict_{iteration - 1}.pt").is_file():
        raise FileNotFoundError("Resume requires the matching rollout dataset state")
    return iteration


def wait_completed_checkpoint(
    checkpoint_dir: Path,
    expected_step: int,
    *,
    timeout: float = 120.0,
) -> int:
    tracker = checkpoint_dir / "latest_checkpointed_iteration.txt"
    target_meta = checkpoint_dir / f"iter_{expected_step:07d}" / "model" / ".metadata"
    deadline = time.monotonic() + timeout
    completed = -1
    while True:
        if target_meta.is_file():
            return expected_step
        if tracker.is_file():
            raw = tracker.read_text(encoding="utf-8").strip()
            if raw.isdigit():
                completed = int(raw)
                if completed == expected_step:
                    return completed
        if time.monotonic() >= deadline:
            return completed
        time.sleep(2.0)


def build_learner_args(*, bundle: Path, micro_batch_size: int) -> list[str]:
    return [
        "--hf-checkpoint", str(bundle),
        "--train-backend", "fsdp",
        "--custom-fsdp-model-factory-path",
        "examples.experimental.generative_retriever.model.fsdp_policy.build_gr_stage5_policy",
        "--data-source-path",
        "examples.experimental.generative_retriever.index.data_source.GRRolloutDataSource",
        "--weight-sync-include-prefixes", *_WEIGHT_SYNC_INCLUDE_PREFIXES,
        "--update-weight-buffer-size", str(_UPDATE_WEIGHT_BUFFER_BYTES),
        "--micro-batch-size", str(micro_batch_size),
    ]


def build_rollout_args(
    args: ScriptArgs,
    *,
    prompt_path: Path,
    input_key: str = "query",
    label_key: str | None = None,
    metadata_key: str | None = None,
) -> list[str]:
    return [
        "--prompt-data", str(prompt_path),
        "--input-key", input_key,
        *(["--label-key", label_key] if label_key else []),
        *(["--metadata-key", metadata_key] if metadata_key else []),
        "--rollout-shuffle",
        "--rollout-function-path",
        "examples.experimental.generative_retriever.model.local_coloc_rollout.GRLocalColocatedRolloutFn",
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
    ]


def build_checkpoint_args(*, checkpoint_dir: Path, save_interval: int, resume: bool) -> list[str]:
    return [
        "--save", str(checkpoint_dir),
        *(["--load", str(checkpoint_dir)] if resume else []),
        "--save-interval", str(save_interval),
    ]


def build_wandb_args(run_id: str) -> list[str]:
    return [
        "--use-wandb",
        "--wandb-project", _WANDB_PROJECT,
        "--wandb-group", run_id,
        "--disable-wandb-random-suffix",
    ]


def build_optimizer_args(lr: float) -> list[str]:
    return [
        "--optimizer", "adam", "--lr", str(lr), "--lr-decay-style", "constant",
        "--weight-decay", "0.1", "--adam-beta1", "0.9", "--adam-beta2", "0.98",
    ]


def build_algorithm_args() -> list[str]:
    return [
        "--advantage-estimator", "grpo", "--kl-coef", "0", "--kl-loss-coef", "0",
        "--entropy-coef", "0", "--observe-training-entropy",
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
