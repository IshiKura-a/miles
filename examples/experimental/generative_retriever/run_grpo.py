"""Single-node eight-GPU GRPO launcher for SLMv3 generative retrieval."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from ray._private import ray_constants
from ray._private.node import Node
from ray._private.parameter import RayParams
import torch
import typer

_MILES_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_MILES_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_MILES_REPO_ROOT))

import miles.utils.external_utils.command_utils as U
from miles.utils.misc import get_current_node_ip
from examples.experimental.generative_retriever.runtime.prepare_rocm_env import (
    build_service_env,
    derive_reward_cache_sizes,
    derive_sglang_mem_fraction,
    is_rocm,
    locked_accelerator_env,
)
from examples.experimental.generative_retriever.runtime.prepare_sglang_bundle import require_sglang_abort_patch
from examples.experimental.generative_retriever.runtime.service_processes import (
    claim_mapping, health_matches, mapping_service_matches, release_mapping, stop_process,
)
from examples.experimental.generative_retriever.runtime.training_common import (
    ScriptArgs, build_algorithm_args, build_optimizer_args, build_sglang_args,
    expanded_path, prepare_prompt_data, required_directory,
)
from examples.experimental.generative_retriever.reward.relevance_reward import (
    REWARD_MODES,
    REWARD_PAIR_BATCH_SIZE,
    REWARD_TOWER_BATCH_SIZE,
)
from examples.experimental.generative_retriever.reward.relevance_model import resolve_relevance_source
from examples.experimental.generative_retriever.services.baseline_service import (
    BASELINE_CACHE_SIZE,
    BASELINE_SERVICE_API_VERSION,
)
from examples.experimental.generative_retriever.services.mapping_service import (
    MAPPING_SERVICE_API_VERSION,
)
from examples.experimental.generative_retriever.services.reward_router_service import (
    REWARD_ROUTER_API_VERSION,
)
from examples.experimental.generative_retriever.services.reward_service import (
    REWARD_SERVICE_API_VERSION,
)

_TRAIN_GPUS = "0,1"
_ROLLOUT_GPUS = "2,3"
_BASELINE_GPUS = "4,5"
_REWARD_GPUS = "0,1,2,3,4,5,6,7"
_REWARD_DEVICES = tuple(f"cuda:{i}" for i in range(len(_REWARD_GPUS.split(","))))
_MAPPING_PORT = 30488
_BASELINE_PORT = 30487
_REWARD_PORT = 30489
_REWARD_WORKER_PORTS = (30490,)
_SERVICE_START_TIMEOUT = 1800.0


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
            with httpx.Client(timeout=30.0, trust_env=False) as client:
                response = client.get(url)
            response.raise_for_status()
            payload = response.json()
            if all(payload.get(key) == value for key, value in expected.items()):
                return
            last_error = RuntimeError(f"unexpected health payload: {payload}")
        except (httpx.HTTPError, ValueError) as error:
            last_error = error
        time.sleep(2)
    raise TimeoutError(f"{name} service did not become ready: {last_error}; see {log_path}")


def _ensure_resident_services(
    args: ScriptArgs,
    *,
    checkpoint: str,
    qwen_model: str,
    slm_root: str,
    mapping: Path,
    service_log_dir: Path,
) -> str:
    relevance_model = expanded_path(args.relevance_model, "Relevance model")
    relevance_source = resolve_relevance_source(
        relevance_model,
    )
    service_dir = service_log_dir
    service_dir.mkdir(parents=True, exist_ok=True)
    mapping_base_url = f"http://127.0.0.1:{_MAPPING_PORT}"
    mapping_url = f"{mapping_base_url}/runs/single"
    baseline_url = f"http://127.0.0.1:{_BASELINE_PORT}/generate"
    reward_url = f"http://127.0.0.1:{_REWARD_PORT}/score"
    worker_urls = [f"http://127.0.0.1:{port}/score" for port in _REWARD_WORKER_PORTS]
    peer_urls_csv = ",".join(worker_urls)
    derived_query_cache, derived_kw_cache, derived_score_cache = derive_reward_cache_sizes(
        args.rollout_batch_size,
        args.n_samples_per_prompt,
        len(_REWARD_WORKER_PORTS),
    )
    query_cache_size = (
        args.reward_query_cache_size
        if args.reward_query_cache_size > 0
        else derived_query_cache
    )
    kw_cache_size = (
        args.reward_keyword_cache_size
        if args.reward_keyword_cache_size > 0
        else derived_kw_cache
    )
    score_cache_size = (
        args.reward_score_cache_size
        if args.reward_score_cache_size > 0
        else derived_score_cache
    )
    specs = [
        (
            "mapping",
            [
                sys.executable,
                "-m",
                "examples.experimental.generative_retriever.services.mapping_service",
                "--mapping",
                os.fspath(mapping),
                "--port",
                str(_MAPPING_PORT),
            ],
            build_service_env(""),
            f"{mapping_base_url}/health",
            {
                "status": "ready",
                "mapping_version": mapping.name,
                "service_config": {"api_version": MAPPING_SERVICE_API_VERSION, "mapping": str(mapping)},
            },
        ),
        (
            "baseline",
            [
                sys.executable,
                "-m",
                "examples.experimental.generative_retriever.services.baseline_service",
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
            build_service_env(_BASELINE_GPUS),
            f"http://127.0.0.1:{_BASELINE_PORT}/health",
            {
                "status": "ready",
                "worker_count": 2,
                "docids_per_query": args.n_samples_per_prompt,
                "service_config": {
                    "api_version": BASELINE_SERVICE_API_VERSION,
                    "checkpoint": checkpoint,
                    "qwen_model": qwen_model,
                    "slm_root": slm_root,
                    "id_mapping": str(mapping),
                    "devices": ["cuda:0", "cuda:1"],
                    "cache_size": BASELINE_CACHE_SIZE,
                    "docids_per_query": args.n_samples_per_prompt,
                },
            },
        ),
        (
            "reward-worker-0",
            [
                sys.executable,
                "-m",
                "examples.experimental.generative_retriever.services.reward_service",
                "--baseline-url",
                baseline_url,
                "--mapping-url",
                mapping_url,
                "--mapping-version",
                mapping.name,
                "--relevance-model",
                os.fspath(relevance_model),
                "--devices",
                ",".join(_REWARD_DEVICES),
                "--docids-per-query",
                str(args.n_samples_per_prompt),
                "--batch-size",
                str(REWARD_TOWER_BATCH_SIZE),
                "--pair-batch-size",
                str(REWARD_PAIR_BATCH_SIZE),
                "--max-samples-per-request",
                str(args.reward_batch_size),
                "--query-cache-size",
                str(query_cache_size),
                "--keyword-cache-size",
                str(kw_cache_size),
                "--score-cache-size",
                str(score_cache_size),
                "--port",
                str(_REWARD_WORKER_PORTS[0]),
            ],
            build_service_env(_REWARD_GPUS),
            f"http://127.0.0.1:{_REWARD_WORKER_PORTS[0]}/health",
            {
                "status": "ready",
                "mode": "worker",
                "worker_count": len(_REWARD_DEVICES),
                "mapping_version": mapping.name,
                "service_config": {
                    "api_version": REWARD_SERVICE_API_VERSION,
                    "baseline_url": baseline_url,
                    "mapping_url": mapping_url,
                    "mapping_version": mapping.name,
                    "relevance_source": str(relevance_source),
                    "relevance_model": str(relevance_model),
                    "devices": list(_REWARD_DEVICES),
                    "query_cache_size": query_cache_size,
                    "keyword_cache_size": kw_cache_size,
                    "score_cache_size": score_cache_size,
                    "batch_size": REWARD_TOWER_BATCH_SIZE,
                    "pair_batch_size": REWARD_PAIR_BATCH_SIZE,
                    "docids_per_query": args.n_samples_per_prompt,
                    "max_samples_per_request": args.reward_batch_size,
                },
            },
        ),
        (
            "reward-router",
            [
                sys.executable,
                "-m",
                "examples.experimental.generative_retriever.services.reward_router_service",
                "--worker-urls",
                peer_urls_csv,
                "--port",
                str(_REWARD_PORT),
                "--max-samples-per-request",
                str(args.reward_batch_size),
                "--forward-concurrency",
                str(args.reward_forward_concurrency),
            ],
            build_service_env(""),
            f"http://127.0.0.1:{_REWARD_PORT}/health",
            {
                "status": "ready",
                "mode": "router",
                "worker_count": len(_REWARD_WORKER_PORTS),
                "mapping_version": mapping.name,
                "service_config": {
                    "api_version": REWARD_ROUTER_API_VERSION,
                    "worker_urls": worker_urls,
                    "max_samples_per_request": args.reward_batch_size,
                    "forward_concurrency": args.reward_forward_concurrency,
                },
            },
        ),
    ]
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
        if (
            mapping_service_matches(mapping_base_url, mapping)
            if name == "mapping" else health_matches(health_url, expected)
        ):
            print(f"[service] reusing healthy {name} at {health_url}", flush=True)
            continue
        log_path = service_dir / f"{name}.log"
        log_stream = log_path.open("a", encoding="utf-8")
        print(f"[service] starting {name} (parallel) -> {log_path}", flush=True)
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
            print(f"[service] {name} ready at {health_url}", flush=True)
    except BaseException:
        for _, proc, stream, _, _, _ in started:
            stop_process(proc, timeout=30, kill_timeout=10)
            if not stream.closed:
                stream.close()
        raise
    return reward_url


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
    mapping = Path(os.path.expandvars(args.id_mapping)).expanduser().resolve()
    if not args.id_mapping or (
        not mapping.is_file() and not (mapping.is_dir() and (mapping / "docid_keywords.parquet").is_file())
    ):
        raise FileNotFoundError(f"Official ID mapping is required: {args.id_mapping!r}")
    if args.temperature <= 0:
        raise ValueError("Training sampling temperature must be positive")
    if args.reward_timeout <= 0:
        raise ValueError("reward_timeout must be positive")
    if min(
        args.reward_batch_size, args.reward_request_concurrency, args.prepare_concurrency,
        args.reward_forward_concurrency,
    ) <= 0:
        raise ValueError("Reward batch and concurrency settings must be positive")
    if args.max_tokens_per_gpu <= 0:
        raise ValueError("max_tokens_per_gpu must be positive")
    if args.sglang_server_concurrency < args.n_samples_per_prompt:
        raise ValueError("SGLang concurrency must fit at least one complete query group")
    total_rollout_samples = args.rollout_batch_size * args.n_samples_per_prompt
    if args.global_batch_size <= 0 or total_rollout_samples % args.global_batch_size != 0:
        raise ValueError(
            "rollout_batch_size * n_samples_per_prompt must be a positive multiple of global_batch_size"
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

    output_dir = expanded_path(args.output_dir, "output_dir", directory=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / args.run_id / "checkpoints"
    mapping_checkpoint = None
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
        mapping_checkpoint = str(checkpoint_dir / f"iter_{iteration:07d}" / "mapping")
        if not (Path(mapping_checkpoint) / "overlay.parquet").is_file():
            raise FileNotFoundError("GR resume requires the matching mapping overlay checkpoint")
    service_log_dir = expanded_path(
        args.service_log_dir,
        "service_log_dir",
        directory=False,
    )
    service_log_dir.mkdir(parents=True, exist_ok=True)
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
    bundle = Path(f"/tmp/gr_sglang_bundle_{args.run_id}")
    quote = shlex.quote
    U.exec_command_cpu(
        " ".join(
            (
                f"{quote(sys.executable)} -m examples.experimental.generative_retriever.runtime.prepare_sglang_bundle",
                f"--checkpoint {quote(checkpoint)}",
                f"--qwen-model {quote(qwen_model)}",
                f"--slm-root {quote(slm_root)}",
                f"--id-mapping {quote(str(mapping))}",
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
            "examples.experimental.generative_retriever.model.sglang_encode_generate.generate "
            "--custom-rm-path examples.experimental.generative_retriever.reward.reward.reward "
            "--custom-rollout-log-function-path "
            "examples.experimental.generative_retriever.reward.rollout_metrics.log_rollout "
            f"--generative-retriever-reward-url {quote(reward_url)} "
            f"--generative-retriever-reward-prepare-url {quote(reward_prepare_url)} "
            f"--generative-retriever-reward-mode {args.reward_mode} "
            f"--generative-retriever-reward-timeout {args.reward_timeout} "
            f"--generative-retriever-reward-batch-size {args.reward_batch_size} "
            f"--generative-retriever-reward-concurrency {args.reward_request_concurrency} "
            f"--generative-retriever-prepare-concurrency {args.prepare_concurrency} "
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
    cuda_env = locked_accelerator_env()
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{_TRAIN_GPUS},{_ROLLOUT_GPUS}"
    os.environ["MILES_HOST_IP"] = node_ip
    no_proxy_hosts = f"127.0.0.1,localhost,{node_ip},{os.environ.get('MASTER_ADDR', 'node-0')}"
    extra_env = {
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
        "MASTER_ADDR": node_ip,
        "PYTHONPATH": f"{_MILES_REPO_ROOT}:" + ":".join(
            p for p in os.environ.get("PYTHONPATH", "").split(":")
            if p and "/opt/venv" not in p
        ),
        "no_proxy": no_proxy_hosts,
        "NO_PROXY": no_proxy_hosts,
        **cuda_env,
    }
    mapping_claimed = False
    try:
        claim_mapping(f"http://127.0.0.1:{_MAPPING_PORT}", "single", mapping_checkpoint)
        mapping_claimed = True
        with httpx.Client(timeout=300.0, trust_env=False) as client:
            response = client.post(reward_url.removesuffix("/score") + "/reset_step_updates")
            response.raise_for_status()
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
                    num_cpus=32,
                    num_gpus=4,
                    object_store_memory=32 * 2**30,
                    temp_dir=ray_temp,
                    include_dashboard=False,
                ),
                head=True,
                shutdown_at_exit=True,
                spawn_reaper=True,
            )
            try:
                train_script = "train_async.py" if args.one_step_async else "train.py"
                train_module = "train_async" if args.one_step_async else "train"
                driver_code = "\n".join((
                    "import asyncio, json, os, sys, ray",
                    "from miles.utils.arguments import parse_args",
                    "from miles.utils.tracking_utils.tracking import finish_tracking",
                    f"from {train_module} import train",
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
                    "4",
                    *shlex.split(train_args),
                ]
                child_env = {
                    **os.environ,
                    **extra_env,
                    "RAY_ADDRESS": f"{node_ip}:{ray_port}",
                    "GR_RAY_RUNTIME_ENV_JSON": json.dumps(extra_env),
                }
                log_file = output_dir / args.run_id / "training.log"
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
                        raise RuntimeError(f"{train_script} exited with code {ret}; see {log_file}")
            finally:
                head.kill_all_processes(check_alive=False, allow_graceful=True, wait=True)
        else:
            U.execute_train(
                train_args=train_args,
                num_gpus_per_node=4,
                megatron_model_type=None,
                train_script="train_async.py" if args.one_step_async else "train.py",
                extra_env_vars=extra_env,
            )
    finally:
        if mapping_claimed:
            release_mapping(f"http://127.0.0.1:{_MAPPING_PORT}", "single")
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
