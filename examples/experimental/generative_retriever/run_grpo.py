"""Single-node eight-GPU GRPO launcher for SLMv3 generative retrieval."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import torch
import typer

_MILES_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_MILES_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_MILES_REPO_ROOT))

import miles.utils.external_utils.command_utils as U
from miles.utils.misc import get_current_node_ip
from examples.experimental.generative_retriever.relevance_reward import REWARD_MODES, RewardMode
from examples.experimental.generative_retriever.relevance_model import resolve_relevance_source

_TRAIN_GPUS = "0,1"
_ROLLOUT_GPUS = "2,3"
_BASELINE_GPUS = "4,5"
_REWARD_GPUS = "6,7"
_MAPPING_PORT = 30488
_BASELINE_PORT = 30487
_REWARD_PORT = 30489
_REWARD_WORKER_PORTS = (30490, 30491)
_SERVICE_START_TIMEOUT = 1800.0


def _required_directory(value: str, name: str) -> str:
    if not value:
        raise ValueError(f"{name} is required")
    path = Path(os.path.expandvars(value)).expanduser().resolve()
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
        "globalindex1_latest_ta_listing_20261019/asiindex_mapping.ta_listing.tsv"
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
    actor_num_gpus_per_node: int = 2
    rollout_num_gpus: int = 2
    rollout_num_gpus_per_engine: int = 2
    sglang_dp_size: int = 2
    sglang_mem_fraction_static: float = 0.7
    sglang_server_concurrency: int = 4096
    rollout_batch_size: int = 1024
    n_samples_per_prompt: int = 64
    global_batch_size: int = 65536
    max_tokens_per_gpu: int = 33024
    reward_timeout: float = 1200.0
    num_rollout: int = 1000
    one_step_async: bool = True
    temperature: float = 1.0
    lr: float = 2e-6
    save_interval: int = 20
    resume: bool = False
    extra_args: str = ""


def _expanded_path(value: str, name: str, *, directory: bool = True) -> Path:
    path = Path(os.path.expandvars(value)).expanduser().resolve()
    if "$" in os.fspath(path):
        raise ValueError(f"{name} contains an unresolved environment variable")
    if directory and not path.is_dir():
        raise FileNotFoundError(f"{name} directory does not exist: {path}")
    return path


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
    if args.actor_num_gpus_per_node != 2 or args.rollout_num_gpus != 2:
        raise ValueError("Learner and train rollout must each use exactly two GPUs")


@dataclass(frozen=True)
class _PromptData:
    path: Path
    input_key: str
    label_key: str | None
    metadata_key: str | None
    sample_count: int


def _count_records(path: Path) -> int:
    with path.open("rb") as stream:
        count = sum(bool(line.strip()) for line in stream)
    if count == 0:
        raise ValueError(f"Prompt dataset is empty: {path}")
    return count


def _prepare_prompt_data(source: Path, jsonl_alias: Path) -> _PromptData:
    with source.open(encoding="utf-8") as stream:
        first = json.loads(stream.readline())
    if isinstance(first, dict) and isinstance(first.get("prompt"), str):
        if source.suffix != ".jsonl":
            raise ValueError(f"Prompt dataset with prompt fields must use a .jsonl suffix: {source}")
        return _PromptData(
            path=source,
            input_key="prompt",
            label_key="label",
            metadata_key="metadata",
            sample_count=_count_records(source),
        )
    if not isinstance(first, dict) or not isinstance(first.get("query"), str):
        raise ValueError(f"Prompt data must contain either prompt or query fields: {source}")
    jsonl_alias.parent.mkdir(parents=True, exist_ok=True)
    if jsonl_alias.exists() or jsonl_alias.is_symlink():
        if jsonl_alias.resolve() != source:
            raise FileExistsError(f"Prompt dataset alias points to another file: {jsonl_alias}")
    else:
        jsonl_alias.symlink_to(source)
    return _PromptData(
        path=jsonl_alias,
        input_key="query",
        label_key=None,
        metadata_key=None,
        sample_count=_count_records(source),
    )


def _service_env(gpus: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["NVIDIA_VISIBLE_DEVICES"] = gpus
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _locked_cuda_env() -> dict[str, str]:
    cuda_home = Path(sys.prefix) / "cuda"
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.is_file():
        raise FileNotFoundError(
            "uv CUDA facade is missing; run "
            "`uv run python -m examples.experimental.generative_retriever.prepare_uv_cuda`"
        )
    lib64 = cuda_home / "lib64"
    if not lib64.is_dir():
        raise FileNotFoundError(f"uv-locked CUDA library directory is missing: {lib64}")
    cudart_link = lib64 / "libcudart.so"
    if not cudart_link.is_file():
        raise FileNotFoundError(f"uv-locked CUDA runtime linker input is missing: {cudart_link}")
    return {
        "CUDA_HOME": os.fspath(cuda_home),
        "PATH": f"{cuda_home / 'bin'}:{os.environ.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{lib64}:{os.environ.get('LD_LIBRARY_PATH', '')}",
    }


def _wait_for_health(
    name: str,
    url: str,
    process: subprocess.Popen[str],
    timeout: float,
    expected: dict[str, Any],
    log_path: Path,
) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"{name} service exited with code {process.returncode}; see {log_path}"
            )
        try:
            response = httpx.get(url, timeout=5.0)
            response.raise_for_status()
            payload = response.json()
            if all(payload.get(key) == value for key, value in expected.items()):
                return
            last_error = RuntimeError(f"unexpected health payload: {payload}")
        except (httpx.HTTPError, ValueError) as error:
            last_error = error
        time.sleep(2)
    raise TimeoutError(f"{name} service did not become ready: {last_error}; see {log_path}")


def _health_matches(url: str, expected: dict[str, Any]) -> bool:
    try:
        response = httpx.get(url, timeout=5.0)
    except httpx.ConnectError:
        return False
    response.raise_for_status()
    payload = response.json()
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Resident service configuration mismatch at {url}: {mismatches}")
    return True


def _ensure_resident_services(
    args: ScriptArgs,
    *,
    checkpoint: str,
    qwen_model: str,
    slm_root: str,
    mapping: Path,
    service_log_dir: Path,
) -> str:
    relevance_model = _expanded_path(args.relevance_model, "Relevance model")
    relevance_source = resolve_relevance_source(
        relevance_model,
    )
    service_dir = service_log_dir
    service_dir.mkdir(parents=True, exist_ok=True)
    mapping_url = f"http://127.0.0.1:{_MAPPING_PORT}"
    baseline_url = f"http://127.0.0.1:{_BASELINE_PORT}/generate"
    reward_url = f"http://127.0.0.1:{_REWARD_PORT}/score"
    specs = [
        (
            "mapping",
            [
                sys.executable,
                "-m",
                "examples.experimental.generative_retriever.mapping_service",
                "--mapping",
                os.fspath(mapping),
                "--port",
                str(_MAPPING_PORT),
            ],
            _service_env(""),
            f"{mapping_url}/health",
            {
                "status": "ready",
                "mapping_version": mapping.name,
                "service_config": {"api_version": 1, "mapping": str(mapping)},
            },
        ),
        (
            "baseline",
            [
                sys.executable,
                "-m",
                "examples.experimental.generative_retriever.baseline_service",
                "--checkpoint",
                checkpoint,
                "--qwen-model",
                qwen_model,
                "--slm-root",
                slm_root,
                "--id-mapping",
                os.fspath(mapping),
                "--docids-per-query",
                str(args.n_samples_per_prompt),
                "--port",
                str(_BASELINE_PORT),
            ],
            _service_env(_BASELINE_GPUS),
            f"http://127.0.0.1:{_BASELINE_PORT}/health",
            {
                "status": "ready",
                "worker_count": 2,
                "docids_per_query": args.n_samples_per_prompt,
                "service_config": {
                    "api_version": 2,
                    "checkpoint": checkpoint,
                    "qwen_model": qwen_model,
                    "slm_root": slm_root,
                    "id_mapping": str(mapping),
                    "devices": ["cuda:0", "cuda:1"],
                    "cache_size": 8192,
                    "docids_per_query": args.n_samples_per_prompt,
                },
            },
        ),
    ]
    for index, (gpu, port) in enumerate(
        zip(_REWARD_GPUS.split(","), _REWARD_WORKER_PORTS, strict=True)
    ):
        specs.append(
            (
                f"reward-worker-{index}",
                [
                    sys.executable,
                    "-m",
                    "examples.experimental.generative_retriever.reward_service",
                    "--baseline-url",
                    baseline_url,
                    "--mapping-url",
                    mapping_url,
                    "--mapping-version",
                    mapping.name,
                    "--relevance-model",
                    os.fspath(relevance_model),
                    "--baseline-top-k",
                    str(args.n_samples_per_prompt),
                    "--batch-size",
                    "2048",
                    "--pair-batch-size",
                    "8192",
                    "--port",
                    str(port),
                ],
                _service_env(gpu),
                f"http://127.0.0.1:{port}/health",
                {
                    "status": "ready",
                    "mode": "worker",
                    "worker_count": 1,
                    "mapping_version": mapping.name,
                    "service_config": {
                        "api_version": 8,
                        "baseline_url": baseline_url,
                        "mapping_url": mapping_url,
                        "mapping_version": mapping.name,
                        "relevance_source": str(relevance_source),
                        "relevance_model": str(relevance_model),
                        "query_cache_size": 8192,
                        "keyword_cache_size": 8192,
                        "score_cache_size": 131072,
                        "batch_size": 2048,
                        "pair_batch_size": 8192,
                        "baseline_top_k": args.n_samples_per_prompt,
                    },
                },
            )
        )
    specs.append(
        (
            "reward-router",
            [
                sys.executable,
                "-m",
                "examples.experimental.generative_retriever.reward_router_service",
                "--worker-urls",
                ",".join(
                    f"http://127.0.0.1:{port}/score" for port in _REWARD_WORKER_PORTS
                ),
                "--port",
                str(_REWARD_PORT),
            ],
            _service_env(""),
            f"http://127.0.0.1:{_REWARD_PORT}/health",
            {
                "status": "ready",
                "mode": "router",
                "worker_count": 2,
                "mapping_version": mapping.name,
                "service_config": {
                    "api_version": 5,
                    "worker_urls": [
                        f"http://127.0.0.1:{port}/score"
                        for port in _REWARD_WORKER_PORTS
                    ]
                },
            },
        )
    )
    started: list[
        tuple[
            str,
            subprocess.Popen[str],
            Any,
            str,
            dict[str, Any],
            Path,
        ]
    ] = []
    for name, command, env, health_url, expected in specs:
        if _health_matches(health_url, expected):
            continue
        log_path = service_dir / f"{name}.log"
        log_stream = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=_MILES_REPO_ROOT,
            env=env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        started.append((name, process, log_stream, health_url, expected, log_path))
    try:
        for name, process, log_stream, health_url, expected, log_path in started:
            _wait_for_health(
                name,
                health_url,
                process,
                _SERVICE_START_TIMEOUT,
                expected,
                log_path,
            )
            log_stream.close()
    except BaseException:
        for _, process, log_stream, _, _, _ in started:
            if process.poll() is None:
                process.terminate()
            if process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            if not log_stream.closed:
                log_stream.close()
        raise
    return reward_url


def execute(args: ScriptArgs) -> None:
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "WANDB_API_KEY must be provided through the process environment "
            "for generative retriever training"
        )
    wandb_args = (
        "--use-wandb "
        "--wandb-project miles-run_grpo "
        f"--wandb-group {shlex.quote(args.run_id)} "
        "--disable-wandb-random-suffix "
    )
    if args.reward_mode not in REWARD_MODES:
        raise ValueError(f"Unknown reward mode: {args.reward_mode!r}")
    _validate_gpu_topology(args)
    checkpoint = _required_directory(args.checkpoint, "checkpoint")
    qwen_model = _required_directory(args.qwen_model, "Qwen model")
    slm_root = _required_directory(args.slm_root, "SLMv3 source")
    mapping = Path(os.path.expandvars(args.id_mapping)).expanduser().resolve()
    if not args.id_mapping or not mapping.is_file():
        raise FileNotFoundError(f"Official ID mapping is required: {args.id_mapping!r}")
    if args.temperature <= 0:
        raise ValueError("Training sampling temperature must be positive")
    if args.reward_timeout <= 0:
        raise ValueError("reward_timeout must be positive")
    if args.max_tokens_per_gpu <= 0:
        raise ValueError("max_tokens_per_gpu must be positive")
    if args.sglang_server_concurrency < args.n_samples_per_prompt:
        raise ValueError("SGLang concurrency must fit at least one complete query group")
    if args.rollout_batch_size * args.n_samples_per_prompt != args.global_batch_size:
        raise ValueError(
            "One training step must contain rollout_batch_size * n_samples_per_prompt "
            "== global_batch_size"
        )
    if args.save_interval <= 0:
        raise ValueError("save_interval must be positive")
    prompt_source = Path(os.path.expandvars(args.prompt_data)).expanduser().resolve()
    if not prompt_source.is_file():
        raise FileNotFoundError(f"prompt data does not exist: {prompt_source}")
    if args.actor_num_gpus_per_node + args.rollout_num_gpus > args.num_gpus_per_node:
        raise ValueError(
            "actor_num_gpus_per_node + rollout_num_gpus cannot exceed num_gpus_per_node"
        )
    if args.rollout_num_gpus % args.rollout_num_gpus_per_engine:
        raise ValueError("rollout_num_gpus must be divisible by rollout_num_gpus_per_engine")

    output_dir = _expanded_path(args.output_dir, "output_dir", directory=False)
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
    service_log_dir = _expanded_path(
        args.service_log_dir,
        "service_log_dir",
        directory=False,
    )
    service_log_dir.mkdir(parents=True, exist_ok=True)
    prompt_data = _prepare_prompt_data(
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
    bundle = output_dir / args.run_id / "sglang_bundle"
    quote = shlex.quote
    U.exec_command_cpu(
        " ".join(
            (
                "python -m examples.experimental.generative_retriever.prepare_sglang_bundle",
                f"--checkpoint {quote(checkpoint)}",
                f"--qwen-model {quote(qwen_model)}",
                f"--slm-root {quote(slm_root)}",
                f"--id-mapping {quote(str(mapping))}",
                "--num-beams 1",
                f"--num-return-sequences {args.n_samples_per_prompt}",
                f"--output {quote(os.fspath(bundle))}",
                f"--temperature {args.temperature}",
                "--decoding-mode sample",
            )
        )
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
        "--skip-actor-forward-only "
    )
    sglang_args = (
        f"--rollout-num-gpus {args.rollout_num_gpus} "
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        f"--sglang-dp-size {args.sglang_dp_size} "
        "--sglang-enable-dp-attention "
        "--sglang-is-embedding "
        "--sglang-disable-radix-cache "
        "--sglang-attention-backend triton "
        "--sglang-disable-cuda-graph "
        f"--sglang-server-concurrency {args.sglang_server_concurrency} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
    )
    learner_args = (
        "--train-backend fsdp "
        "--data-source-path "
        "examples.experimental.generative_retriever.data_source.GRRolloutDataSource "
        "--custom-fsdp-model-factory-path "
        "examples.experimental.generative_retriever.fsdp_policy.build_gr_stage5_policy "
        "--weight-sync-include-prefixes "
        "docEnc2decoder. decoder. lm_head. "
        "--update-weight-buffer-size 536870912 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
    )
    checkpoint_args = (
        f"--save {quote(os.fspath(checkpoint_dir))} "
        f"{f'--load {quote(os.fspath(checkpoint_dir))} ' if args.resume else ''}"
        f"--save-interval {args.save_interval} "
    )
    misc_args = (
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.actor_num_gpus_per_node} "
    )
    reward_url = _ensure_resident_services(
        args,
        checkpoint=checkpoint,
        qwen_model=qwen_model,
        slm_root=slm_root,
        mapping=mapping,
        service_log_dir=service_log_dir,
    )
    reward_prepare_url = reward_url.removesuffix("/score") + "/prepare"
    rollout_args = (
            f"--prompt-data {quote(os.fspath(prompt_data.path))} "
            f"--input-key {quote(prompt_data.input_key)} "
            f"{f'--label-key {quote(prompt_data.label_key)} ' if prompt_data.label_key else ''}"
            f"{f'--metadata-key {quote(prompt_data.metadata_key)} ' if prompt_data.metadata_key else ''}"
            "--rollout-shuffle "
            "--custom-generate-function-path "
            "examples.experimental.generative_retriever.sglang_encode_generate.generate "
            "--custom-rm-path examples.experimental.generative_retriever.reward.reward "
            "--custom-rollout-log-function-path "
            "examples.experimental.generative_retriever.rollout_metrics.log_rollout "
            f"--generative-retriever-reward-url {quote(reward_url)} "
            f"--generative-retriever-reward-prepare-url {quote(reward_prepare_url)} "
            f"--generative-retriever-reward-mode {args.reward_mode} "
            f"--generative-retriever-reward-timeout {args.reward_timeout} "
            f"--num-rollout {args.num_rollout} "
            f"--rollout-batch-size {args.rollout_batch_size} "
            f"--n-samples-per-prompt {args.n_samples_per_prompt} "
            "--rollout-max-response-len 4 "
            f"--rollout-temperature {args.temperature} "
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
    cuda_env = _locked_cuda_env()
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{_TRAIN_GPUS},{_ROLLOUT_GPUS}"
    os.environ["MILES_HOST_IP"] = node_ip
    try:
        U.execute_train(
                train_args=train_args,
                num_gpus_per_node=4,
                megatron_model_type=None,
                train_script="train_async.py" if args.one_step_async else "train.py",
                extra_env_vars={
                    "GR_CHECKPOINT": checkpoint,
                    "GR_QWEN_MODEL": qwen_model,
                    "GR_SLM_ROOT": slm_root,
                    "GR_ID_MAPPING": str(mapping),
                    "SGLANG_EXTERNAL_MODEL_PACKAGE": (
                        "examples.experimental.generative_retriever.sglang_models"
                    ),
                    "SGLANG_USE_AITER": "0",
                    "CUDA_VISIBLE_DEVICES": f"{_TRAIN_GPUS},{_ROLLOUT_GPUS}",
                    "MILES_HOST_IP": node_ip,
                    "no_proxy": f"127.0.0.1,{node_ip},{os.environ.get('MASTER_ADDR', 'node-0')}",
                    **cuda_env,
                },
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
