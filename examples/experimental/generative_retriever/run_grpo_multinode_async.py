"""Run one asynchronous GRPO job across pairs of eight-GPU ROCm nodes.

Requires matching runtime Miles and SLMv3 source on all hosts, shared Cosmos
paths, passwordless SSH, and WANDB_API_KEY. Source .venv/activate-rocm.sh on
the first host before launching:

    uv run --no-sync --env-file .env python -m \
        examples.experimental.generative_retriever.run_grpo_multinode_async \
        --hosts node-0,node-1 --run-id gr-16gpu-001

Each pair consists of a service/rollout node followed by a learner node.
Only processes started for this run are stopped on exit.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, replace
import filecmp
import hashlib
import json
import os
from pathlib import Path
import select
import shlex
import signal
import socket
import subprocess
import sys
import time
from typing import IO, TypedDict, cast

import httpx
import ray
from ray._private import ray_constants
from ray._private.node import Node
from ray._private.parameter import RayParams

from examples.experimental.generative_retriever import run_grpo
from examples.experimental.generative_retriever.prepare_sglang_bundle import (
    prepare_bundle,
    require_sglang_abort_patch,
)
from miles.utils.external_utils.command_utils import create_run_id

_REPO = Path(__file__).resolve().parents[3]
_MODULE = "examples.experimental.generative_retriever.run_grpo_multinode_async"
_RESIDENT_MAPPING_PORT = 31480
_MAPPING_PORT = 31580
_BASELINE_PORT = 31581
_REWARD_PORTS = tuple(range(31583, 31591))
_ROUTER_PORT = 31593
_RAY_PORT = 16380
_SERVICE_TIMEOUT = 3600


def _disable_uv_runtime_env() -> None:
    # Ray would replay uv --env-file .env inside a packaged directory that excludes .env.
    os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False


class Manifest(TypedDict):
    run_id: str
    hosts: list[str]
    addresses: list[str]
    head_ip: str
    ray_address: str
    ray_temp: str
    run_dir: str
    checkpoint: str
    qwen_model: str
    slm_root: str
    mapping: str
    mapping_url: str
    mapping_owned: bool
    relevance_model: str
    prompt_source: str
    num_rollout: int
    train_argv: list[str]
    runtime_env: dict[str, str]


@ray.remote
class _ReserveServiceGpus:
    def gpu_ids(self) -> list[int]:
        return [int(value) for value in ray.get_gpu_ids()]


@dataclass
class _Child:
    name: str
    process: subprocess.Popen[str]
    stream: IO[str]


class _Processes:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.children: list[_Child] = []
        self.persistent: list[_Child] = []
        log_dir.mkdir(parents=True, exist_ok=True)

    def start(
        self, name: str, argv: list[str], *,
        gpus: str | None = None, stdin: bool = False, training: bool = False,
        persistent: bool = False,
    ) -> subprocess.Popen[str]:
        env = os.environ.copy()
        if not training:
            env.pop("WANDB_API_KEY", None)
        if gpus is None:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            env["CUDA_VISIBLE_DEVICES"] = gpus
        stream = (self.log_dir / f"{name}.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            argv, cwd=_REPO, env=env,
            stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, text=True,
            start_new_session=persistent,
        )
        if persistent:
            self.persistent.append(_Child(name, process, stream))
            stream.close()
        else:
            self.children.append(_Child(name, process, stream))
        print(f"{name}: pid={process.pid}, log={self.log_dir / (name + '.log')}", flush=True)
        return process

    def check(self, except_name: str | None = None) -> None:
        for child in (*self.children, *self.persistent):
            if child.name != except_name and child.process.poll() is not None:
                raise RuntimeError(
                    f"{child.name} exited with {child.process.returncode}: "
                    f"{self.log_dir / (child.name + '.log')}"
                )

    def close(self) -> None:
        for child in reversed(self.children):
            process = child.process
            if process.poll() is None and process.stdin is not None:
                try:
                    process.stdin.write("STOP\n")
                    process.stdin.close()
                except BrokenPipeError:
                    pass
        for child in reversed(self.children):
            process = child.process
            if process.poll() is None:
                if process.stdin is not None:
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                else:
                    process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
            child.stream.close()


def _hosts(text: str) -> tuple[list[str], list[str]]:
    hosts = text.split(",")
    if len(hosts) < 2 or len(hosts) % 2 or len(set(hosts)) != len(hosts) or any(
        not host.strip() or host != host.strip() for host in hosts
    ):
        raise ValueError("Pass distinct hostnames in service,learner pairs (2n nodes)")
    addresses = [socket.gethostbyname(host) for host in hosts]
    if len(set(addresses)) != len(addresses):
        raise ValueError("Each host must resolve to a different node IP")
    return hosts, addresses


def _recipe(run_id: str, pairs: int, num_rollout: int) -> run_grpo.ScriptArgs:
    if not run_id or "/" in run_id or run_id in {".", ".."}:
        raise ValueError("run-id must be a nonempty directory name without slashes")
    if 3750 % pairs:
        raise ValueError(f"Cannot cover 15M rows exactly with {pairs} pairs at 4000n queries/step")
    if not 1 <= num_rollout <= 3750 // pairs:
        raise ValueError(f"num-rollout must be between 1 and {3750 // pairs}")
    return replace(
        run_grpo.ScriptArgs(),
        run_id=run_id,
        num_rollout=num_rollout,
        rollout_batch_size=4000 * pairs,
        global_batch_size=256000 * pairs,
        actor_num_gpus_per_node=8,
        rollout_num_gpus=4 * pairs,
        max_tokens_per_gpu=524288,
        sglang_server_concurrency=8192 * pairs,
        reward_timeout=1800,
        reward_request_concurrency=8 * pairs,
        prepare_concurrency=pairs,
    )


def _train_arguments(
    args: run_grpo.ScriptArgs, *, addresses: list[str], bundle: Path, prompt: Path,
    reward_url: str, checkpoint_dir: Path, resume: bool,
) -> list[str]:
    pairs = len(addresses) // 2
    blocks = [
        [
            "--hf-checkpoint", str(bundle),
            "--train-backend", "fsdp",
            "--custom-fsdp-model-factory-path",
            "examples.experimental.generative_retriever.fsdp_policy.build_gr_stage5_policy",
            "--data-source-path",
            "examples.experimental.generative_retriever.data_source.GRRolloutDataSource",
            "--weight-sync-include-prefixes", "docEnc2decoder.", "decoder.", "lm_head.",
            "--update-weight-buffer-size", "536870912",
            "--use-dynamic-batch-size",
            "--max-tokens-per-gpu", str(args.max_tokens_per_gpu),
        ],
        [
            "--prompt-data", str(prompt),
            "--input-key", "query",
            "--rollout-shuffle",
            "--custom-generate-function-path",
            "examples.experimental.generative_retriever.sglang_encode_generate.generate",
            "--custom-rm-path", "examples.experimental.generative_retriever.reward.reward",
            "--custom-rollout-log-function-path",
            "examples.experimental.generative_retriever.rollout_metrics.log_rollout",
            "--generative-retriever-reward-url", reward_url + "/score",
            "--generative-retriever-reward-prepare-url", reward_url + "/prepare",
            "--generative-retriever-reward-mode", args.reward_mode,
            "--generative-retriever-reward-timeout", str(args.reward_timeout),
            "--generative-retriever-reward-batch-size", str(args.reward_batch_size),
            "--generative-retriever-reward-concurrency", str(args.reward_request_concurrency),
            "--generative-retriever-prepare-concurrency", str(args.prepare_concurrency),
            "--num-rollout", str(args.num_rollout),
            "--rollout-batch-size", str(args.rollout_batch_size),
            "--n-samples-per-prompt", str(args.n_samples_per_prompt),
            "--rollout-max-response-len", "4",
            "--rollout-temperature", str(args.temperature),
            "--rollout-top-p", "1",
            "--global-batch-size", str(args.global_batch_size),
        ],
        [
            "--optimizer", "adam",
            "--lr", str(args.lr),
            "--lr-decay-style", "constant",
            "--weight-decay", "0.1",
            "--adam-beta1", "0.9",
            "--adam-beta2", "0.98",
            "--advantage-estimator", "grpo",
            "--kl-coef", "0",
            "--kl-loss-coef", "0",
            "--entropy-coef", "0",
            "--eps-clip", "0.2",
            "--eps-clip-high", "0.28",
            "--use-rollout-logprobs",
            "--skip-actor-forward-only",
        ],
        [
            "--actor-num-nodes", str(pairs),
            "--actor-num-gpus-per-node", str(args.actor_num_gpus_per_node),
            "--num-gpus-per-node", str(args.num_gpus_per_node),
            "--rollout-num-gpus", str(args.rollout_num_gpus),
            "--rollout-num-gpus-per-engine", str(args.rollout_num_gpus_per_engine),
            "--sglang-dp-size", str(args.sglang_dp_size),
            "--sglang-enable-dp-attention",
            "--sglang-is-embedding",
            "--sglang-disable-radix-cache",
            "--sglang-attention-backend", "triton",
            "--sglang-disable-cuda-graph",
            "--sglang-server-concurrency", str(args.sglang_server_concurrency),
            "--sglang-mem-fraction-static", str(args.sglang_mem_fraction_static),
        ],
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


def _runtime_env(
    *, checkpoint: str, qwen: str, slm_root: str, mapping: str,
    addresses: list[str],
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
        "GR_CHECKPOINT": checkpoint,
        "GR_QWEN_MODEL": qwen,
        "GR_SLM_ROOT": slm_root,
        "GR_ID_MAPPING": mapping,
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


def _mapping_service(mapping: Path, head_ip: str) -> tuple[str, bool]:
    resident_url = f"http://{head_ip}:{_RESIDENT_MAPPING_PORT}"
    with httpx.Client(timeout=10, trust_env=False) as client:
        try:
            response = client.get(f"{resident_url}/health")
        except httpx.ConnectError:
            return f"http://{head_ip}:{_MAPPING_PORT}", True
    response.raise_for_status()
    health = response.json()
    config = health.get("service_config")
    if (
        health.get("status") != "ready"
        or health.get("mapping_version") != mapping.name
        or health.get("rows") != 238_836_331
        or health.get("bytes") != mapping.stat().st_size
        or not isinstance(config, dict)
        or config.get("api_version") != 1
        or not isinstance(config.get("mapping"), str)
    ):
        raise RuntimeError(f"Resident mapping at {resident_url} does not match the documented mapping")
    resident_path = Path(config["mapping"]).resolve()
    if not resident_path.is_file():
        raise FileNotFoundError(f"Resident mapping file is missing: {resident_path}")
    if resident_path != mapping:
        original = (mapping.stat().st_size, mapping.stat().st_mtime_ns)
        resident = (resident_path.stat().st_size, resident_path.stat().st_mtime_ns)
        if not filecmp.cmp(resident_path, mapping, shallow=False):
            raise RuntimeError(f"Resident mapping content differs from {mapping}: {resident_path}")
        if (
            original != (mapping.stat().st_size, mapping.stat().st_mtime_ns)
            or resident != (resident_path.stat().st_size, resident_path.stat().st_mtime_ns)
        ):
            raise RuntimeError("Mapping changed while verifying the resident service")
    return resident_url, False


def _manifest(
    args: run_grpo.ScriptArgs, *, hosts: list[str], addresses: list[str],
    create: bool, resume: bool,
) -> Manifest:
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY must be set for GRPO training")
    checkpoint = run_grpo._required_directory(args.checkpoint, "Stage5 checkpoint")
    qwen = run_grpo._required_directory(args.qwen_model, "Qwen encoder")
    slm_root = run_grpo._required_directory(args.slm_root, "SLMv3 source")
    relevance = run_grpo._required_directory(args.relevance_model, "relevance model")
    mapping = Path(os.path.expandvars(args.id_mapping)).resolve()
    prompt = Path(os.path.expandvars(args.prompt_data)).resolve()
    if not mapping.is_file() or not prompt.is_file():
        raise FileNotFoundError(f"Official mapping or prompt dataset missing: {mapping}, {prompt}")
    mapping_url, mapping_owned = _mapping_service(mapping, addresses[0])
    output = run_grpo._expanded_path(args.output_dir, "output_dir", directory=False)
    run_dir = output / args.run_id
    checkpoint_dir = run_dir / "checkpoints"
    alias = run_dir / "prompt_data.jsonl"
    bundle = run_dir / "sglang_bundle"
    count = run_grpo._count_records(prompt)
    if count != 15_000_000 or args.rollout_batch_size * args.num_rollout > count:
        raise ValueError(f"Expected 15M source rows without wrapping, got {count}")
    if create:
        if resume:
            tracker = checkpoint_dir / "latest_checkpointed_iteration.txt"
            if not tracker.is_file():
                raise FileNotFoundError(f"Resume checkpoint missing: {tracker}")
            iteration = int(tracker.read_text().strip())
            if iteration <= 0 or not (checkpoint_dir / f"iter_{iteration:07d}" / "meta.json").is_file():
                raise ValueError(f"Incomplete checkpoint at iteration {iteration}")
            if not (
                checkpoint_dir / "rollout" / f"global_dataset_state_dict_{iteration - 1}.pt"
            ).is_file():
                raise FileNotFoundError("Resume requires the matching rollout dataset state")
        else:
            run_dir.mkdir(parents=True, exist_ok=False)
        run_grpo._prepare_prompt_data(prompt, alias)
        prepare_bundle(
            checkpoint=checkpoint, qwen_model=qwen, slm_root=slm_root,
            id_mapping=mapping, output=bundle, num_beams=64,
            num_return_sequences=64, temperature=1.0, decoding_mode="beam_sample",
        )
    return {
        "run_id": args.run_id,
        "hosts": hosts,
        "addresses": addresses,
        "head_ip": addresses[0],
        "ray_address": f"{addresses[0]}:{_RAY_PORT}",
        "ray_temp": f"/tmp/miles_grpo_{args.run_id}",
        "run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "qwen_model": qwen,
        "slm_root": slm_root,
        "mapping": str(mapping),
        "mapping_url": mapping_url,
        "mapping_owned": mapping_owned,
        "relevance_model": relevance,
        "prompt_source": str(prompt),
        "num_rollout": args.num_rollout,
        "train_argv": _train_arguments(
            args, addresses=addresses, bundle=bundle, prompt=alias,
            reward_url=f"http://{addresses[0]}:{_ROUTER_PORT}",
            checkpoint_dir=checkpoint_dir, resume=resume,
        ),
        "runtime_env": _runtime_env(
            checkpoint=checkpoint, qwen=qwen, slm_root=slm_root,
            mapping=str(mapping), addresses=addresses,
        ),
    }


def _wait_health(
    processes: _Processes, host: str, name: str, port: int,
    expected: dict[str, object],
) -> None:
    deadline = time.monotonic() + _SERVICE_TIMEOUT
    url = f"http://{host}:{port}/health"
    with httpx.Client(timeout=10, trust_env=False) as client:
        while time.monotonic() < deadline:
            processes.check()
            try:
                response = client.get(url)
            except (httpx.ConnectError, httpx.TimeoutException):
                time.sleep(3)
                continue
            if response.status_code == 503:
                time.sleep(3)
                continue
            response.raise_for_status()
            result = response.json()
            service_config = expected.get("service_config")
            actual_config = result.get("service_config")
            if (
                result.get("status") != "ready"
                or any(
                    result.get(key) != value
                    for key, value in expected.items() if key != "service_config"
                )
                or (
                    service_config is not None
                    and (
                        not isinstance(service_config, dict)
                        or not isinstance(actual_config, dict)
                        or any(
                            actual_config.get(key) != value
                            for key, value in service_config.items()
                        )
                    )
                )
            ):
                raise RuntimeError(f"{name} returned unexpected health at {url}: {result}")
            return
    raise TimeoutError(f"{name} did not become ready at {url}; see {processes.log_dir}")


def _start_service(
    processes: _Processes, name: str, module: str, host: str, port: int,
    argv: list[str], *, gpus: str | None = None, persistent: bool = False,
) -> subprocess.Popen[str]:
    return processes.start(
        name,
        [sys.executable, "-u", "-m",
         f"examples.experimental.generative_retriever.{module}",
         *argv, "--host", host, "--port", str(port)],
        gpus=gpus, persistent=persistent,
    )


def _start_baseline(
    processes: _Processes, manifest: Manifest, host: str,
) -> tuple[subprocess.Popen[str], dict[str, object]] | None:
    mapping = manifest["mapping"]
    expected: dict[str, object] = {
        "status": "ready",
        "worker_count": 4,
        "docids_per_query": 64,
        "service_config": {
            "api_version": 2,
            "checkpoint": manifest["checkpoint"],
            "qwen_model": manifest["qwen_model"],
            "slm_root": manifest["slm_root"],
            "id_mapping": mapping,
            "devices": [f"cuda:{gpu}" for gpu in range(4)],
            "cache_size": 8192,
            "docids_per_query": 64,
        },
    }
    url = f"http://{host}:{_BASELINE_PORT}/health"
    if run_grpo._health_matches(url, expected):
        print(f"Reusing verified frozen baseline: {url}", flush=True)
        return None
    _available_ports(host, (_BASELINE_PORT,))
    process = _start_service(
        processes, "baseline", "baseline_service", host, _BASELINE_PORT,
        [
            "--checkpoint", manifest["checkpoint"],
            "--qwen-model", manifest["qwen_model"],
            "--slm-root", manifest["slm_root"],
            "--id-mapping", mapping, "--docids-per-query", "64",
            "--devices", "cuda:0,cuda:1,cuda:2,cuda:3",
        ],
        gpus="0,1,2,3", persistent=True,
    )
    return process, expected


def _exit_on_sigterm(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


def _service_node(manifest: Manifest, index: int) -> None:
    os.environ.update(manifest["runtime_env"])
    host = manifest["addresses"][index]
    mapping = manifest["mapping"]
    processes = _Processes(Path(manifest["run_dir"]) / f"services-{index}")
    previous_handler = signal.signal(signal.SIGTERM, _exit_on_sigterm)
    baseline: tuple[subprocess.Popen[str], dict[str, object]] | None = None
    try:
        baseline = _start_baseline(processes, manifest, host)
        reward_health = []
        for gpu, port in enumerate(_REWARD_PORTS):
            name = f"reward-{gpu}"
            _start_service(
                processes, name, "reward_service", host, port,
                [
                    "--baseline-url", f"http://{host}:{_BASELINE_PORT}/generate",
                    "--mapping-url", manifest["mapping_url"],
                    "--mapping-version", Path(mapping).name,
                    "--relevance-model", manifest["relevance_model"],
                    "--batch-size", "8192", "--pair-batch-size", "32768",
                    "--baseline-top-k", "64", "--max-samples-per-request", "2048",
                    "--keyword-cache-size", "8192", "--score-cache-size", "131072",
                ],
                gpus=str(gpu),
            )
            reward_health.append((name, port, {
                "worker_count": 1, "mapping_version": Path(mapping).name,
                "service_config": {
                    "baseline_url": f"http://{host}:{_BASELINE_PORT}/generate",
                    "mapping_url": manifest["mapping_url"],
                    "relevance_model": manifest["relevance_model"],
                    "max_samples_per_request": 2048,
                },
            }))
        if baseline is not None:
            _wait_health(processes, host, "baseline", _BASELINE_PORT, baseline[1])
            print(
                f"Frozen baseline will remain on GPU 0-3 after this run: "
                f"http://{host}:{_BASELINE_PORT}/health",
                flush=True,
            )
            baseline = None
        for name, port, expected in reward_health:
            _wait_health(processes, host, name, port, expected)
        print(f"service node {index} ready", flush=True)
        while True:
            processes.check()
            ready, _, _ = select.select([sys.stdin], [], [], 2)
            if ready:
                if sys.stdin.readline().strip() == "STOP":
                    break
                raise RuntimeError("Service manager input closed without STOP")
    except BaseException:
        if baseline is not None and baseline[0].poll() is None:
            baseline[0].terminate()
            try:
                baseline[0].wait(timeout=30)
            except subprocess.TimeoutExpired:
                baseline[0].kill()
                baseline[0].wait(timeout=30)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        processes.close()


def _available_ports(host: str, ports: tuple[int, ...]) -> None:
    for port in ports:
        with socket.socket() as connection:
            connection.settimeout(1)
            if connection.connect_ex((host, port)) == 0:
                raise RuntimeError(f"Port {host}:{port} occupied; will not interrupt another job")


def _remote(host: str, role: str, path: Path, index: int) -> list[str]:
    remote = (
        f"cd {shlex.quote(str(_REPO))} && "
        f"source {shlex.quote(str(_REPO / '.venv/activate-rocm.sh'))} && "
        f"exec {shlex.quote(sys.executable)} -u -m {_MODULE} {role} "
        f"{shlex.quote(str(path))} {index}"
    )
    return ["ssh", "-o", "BatchMode=yes", host, "bash", "-c", shlex.quote(remote)]


def _prepare_node_assets(manifest: Manifest) -> None:
    os.environ.update(manifest["runtime_env"])
    run_dir = Path(manifest["run_dir"])
    source = Path(manifest["prompt_source"])
    alias = run_dir / "prompt_data.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"Prompt dataset is missing on this node: {source}")
    if alias.is_symlink():
        if alias.resolve() != source:
            raise FileExistsError(f"Prompt alias points to a different dataset: {alias}")
    elif alias.exists():
        raise FileExistsError(f"Prompt alias is not a symlink: {alias}")
    else:
        alias.symlink_to(source)
    if not alias.is_file():
        raise FileNotFoundError(f"Prompt alias is not readable on this node: {alias}")

    bundle = prepare_bundle(
        checkpoint=manifest["checkpoint"],
        qwen_model=manifest["qwen_model"],
        slm_root=manifest["slm_root"],
        id_mapping=manifest["mapping"],
        output=run_dir / "sglang_bundle",
        num_beams=64, num_return_sequences=64, temperature=1.0,
        decoding_mode="beam_sample",
    )
    for name in ("config.json", "tokenizer.json", "pytorch_model.bin.index.json"):
        if not (bundle / name).is_file():
            raise FileNotFoundError(f"Model bundle asset is not readable on this node: {bundle / name}")


def _check_miles_checkouts(hosts: list[str]) -> None:
    local_commit = subprocess.check_output(
        ["git", "-C", str(_REPO), "rev-parse", "HEAD"], text=True,
    ).strip()
    local_changes = subprocess.check_output(
        ["git", "-C", str(_REPO), "status", "--porcelain", "--untracked-files=all"], text=True,
    )
    if local_changes:
        raise RuntimeError("Miles checkout on the launch node is dirty; deploy a committed version before training")
    for host in hosts[1:]:
        remote_checkout = f"git -C {shlex.quote(str(_REPO))}"
        checkout = subprocess.run(
            [
                "ssh", "-o", "BatchMode=yes", host,
                f"{remote_checkout} rev-parse HEAD && "
                f"{remote_checkout} status --porcelain --untracked-files=all",
            ],
            check=True, capture_output=True, text=True,
        )
        lines = checkout.stdout.splitlines()
        if not lines or lines[0] != local_commit:
            raise RuntimeError(f"Miles checkout on {host} must be at commit {local_commit}")
        if len(lines) != 1:
            raise RuntimeError(f"Miles checkout on {host} is dirty; deploy a clean checkout before training")


def _check_remote_sources(manifest: Manifest) -> None:
    _check_miles_checkouts(manifest["hosts"])
    run_dir = Path(manifest["run_dir"])
    require_sglang_abort_patch()
    sglang_patch = Path(__file__).resolve().parent / "patches/sglang-embedding-abort.patch"
    sglang_root = Path(sys.prefix) / "src/sglang"
    files = (
        sglang_root / "python/sglang/srt/entrypoints/openai/serving_embedding.py",
        Path(manifest["slm_root"]) / "model.py",
        Path(manifest["checkpoint"]) / "pytorch_model.bin.index.json",
        Path(manifest["qwen_model"]) / "model.safetensors.index.json",
    )
    expected = [hashlib.sha256(path.read_bytes()).hexdigest() for path in files]
    for index, host in enumerate(manifest["hosts"][1:], start=1):
        remote = "sha256sum " + " ".join(shlex.quote(str(path)) for path in files)
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", host, remote],
            check=True, capture_output=True, text=True,
        )
        actual = [line.split(maxsplit=1)[0] for line in result.stdout.splitlines()]
        if actual != expected:
            raise RuntimeError(f"Source/model indices differ on {host}; synchronize the checkout first")
        patch_check = subprocess.run(
            [
                "ssh", "-o", "BatchMode=yes", host,
                f"git -C {shlex.quote(str(sglang_root))} apply --reverse --check -",
            ],
            input=sglang_patch.read_text(), capture_output=True, text=True,
        )
        if patch_check.returncode:
            raise RuntimeError(f"SGLang embedding abort patch is missing on {host}: {patch_check.stderr.strip()}")
        required = (
            (run_dir / "manifest.json", False),
            (run_dir / "sglang_bundle/config.json", False),
            (Path(manifest["prompt_source"]), False),
            (Path(manifest["mapping"]), False),
            (Path(sys.executable), True),
            (_REPO / ".venv/activate-rocm.sh", False),
        )
        for file, executable in required:
            probe = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", host,
                 f"test -{'x' if executable else 'f'} {shlex.quote(str(file))}"],
                capture_output=True, text=True,
            )
            if probe.returncode == 1:
                raise FileNotFoundError(f"Node {host} cannot read required file: {file}")
            if probe.returncode:
                raise RuntimeError(f"Node {host} prerequisite check failed: {probe.stderr.strip()}")
        subprocess.run(
            _remote(host, "_prepare_assets", run_dir / "manifest.json", index),
            check=True,
        )


def _worker(manifest: Manifest, index: int) -> None:
    os.environ.update(manifest["runtime_env"])
    key = sys.stdin.readline()
    if not key.endswith("\n") or not key.strip() or "\r" in key:
        raise RuntimeError("Ray worker did not receive a valid WANDB_API_KEY through SSH stdin")
    os.environ["WANDB_API_KEY"] = key[:-1]
    node = Node(
        RayParams(
            gcs_address=manifest["ray_address"],
            node_ip_address=manifest["addresses"][index],
            num_cpus=64, num_gpus=8, object_store_memory=32 * 2**30,
            temp_dir=manifest["ray_temp"], include_dashboard=False,
        ),
        head=False, shutdown_at_exit=True, spawn_reaper=True,
    )
    try:
        print(f"Ray node {index} ready", flush=True)
        for line in sys.stdin:
            if line.strip() == "STOP":
                break
    finally:
        node.kill_all_processes(check_alive=False, allow_graceful=True, wait=True)


def _train(manifest: Manifest) -> None:
    from train_async import train
    from miles.utils.arguments import parse_args
    from miles.utils.tracking_utils.tracking import finish_tracking

    key = os.environ.get("WANDB_API_KEY")
    if not key:
        raise RuntimeError("WANDB_API_KEY must be set in the training driver")
    runtime_env = {"env_vars": manifest["runtime_env"]}
    ray.init(address=manifest["ray_address"], runtime_env=runtime_env)
    sys.argv = [str(_REPO / "train_async.py"), *manifest["train_argv"]]
    try:
        args = parse_args()
        args.actor_node_ips = manifest["addresses"][1::2]
        args.rollout_node_ips = manifest["addresses"][::2]
        try:
            asyncio.run(train(args))
        finally:
            finish_tracking()
    finally:
        ray.shutdown()


def _run(manifest: Manifest) -> None:
    run_dir = Path(manifest["run_dir"])
    path = run_dir / "manifest.json"
    processes = _Processes(run_dir)
    head: Node | None = None
    ray_connected = False
    try:
        if ray.util.get_node_ip_address() != manifest["head_ip"]:
            raise RuntimeError("Launch on the first service/rollout host listed in --hosts")
        _check_remote_sources(manifest)
        if not manifest["mapping_owned"]:
            if _mapping_service(Path(manifest["mapping"]), manifest["head_ip"])[0] != manifest["mapping_url"]:
                raise RuntimeError("Resident mapping service changed before training started")
            print(f"Reusing verified resident mapping: {manifest['mapping_url']}", flush=True)
        _available_ports(
            manifest["head_ip"],
            (_RAY_PORT, _ROUTER_PORT, *_REWARD_PORTS)
            + ((_MAPPING_PORT,) if manifest["mapping_owned"] else ()),
        )
        for host in manifest["addresses"][2::2]:
            _available_ports(host, _REWARD_PORTS)
        if manifest["mapping_owned"]:
            print(f"Starting isolated mapping: {manifest['mapping_url']}", flush=True)
            processes.start(
                "mapping",
                [sys.executable, "-u", "-m",
                 "examples.experimental.generative_retriever.mapping_service",
                 "--mapping", manifest["mapping"], "--host", manifest["head_ip"],
                 "--port", str(_MAPPING_PORT)],
            )
        for index, host in enumerate(manifest["hosts"][::2]):
            node = index * 2
            argv = (
                [sys.executable, "-u", "-m", _MODULE, "_service_node", str(path), str(node)]
                if node == 0 else _remote(host, "_service_node", path, node)
            )
            processes.start(f"services-{node}", argv, stdin=True)
        worker_urls = [
            f"http://{host}:{port}/score"
            for host in manifest["addresses"][::2] for port in _REWARD_PORTS
        ]
        _start_service(
            processes, "reward-router", "reward_router_service",
            manifest["head_ip"], _ROUTER_PORT,
            ["--worker-urls", ",".join(worker_urls),
             "--max-samples-per-request", "2048", "--forward-concurrency", "1"],
        )
        os.environ.update(manifest["runtime_env"])
        head = Node(
            RayParams(
                node_ip_address=manifest["head_ip"], gcs_server_port=_RAY_PORT,
                num_cpus=32, num_gpus=8, object_store_memory=32 * 2**30,
                temp_dir=manifest["ray_temp"], include_dashboard=False,
            ),
            head=True, shutdown_at_exit=True, spawn_reaper=True,
        )
        for index, host in enumerate(manifest["hosts"][1:], start=1):
            worker = processes.start(
                f"ray-worker-{index}", _remote(host, "_worker", path, index),
                stdin=True,
            )
            key = os.environ.get("WANDB_API_KEY")
            if not key or "\n" in key or "\r" in key or worker.stdin is None:
                raise RuntimeError("Cannot securely supply WANDB_API_KEY to a Ray worker")
            worker.stdin.write(key + "\n")
            worker.stdin.flush()
        ray.init(address=manifest["ray_address"])
        ray_connected = True
        deadline = time.monotonic() + 180
        required = 8 * len(manifest["hosts"])
        while ray.cluster_resources().get("GPU", 0) != required:
            processes.check()
            if time.monotonic() > deadline:
                raise TimeoutError(f"Expected {required} Ray GPUs across all nodes")
            time.sleep(2)
        reservations = [
            # Ray's decorator supplies .options/.remote dynamically.
            _ReserveServiceGpus.options(  # pyright: ignore[reportFunctionMemberAccess]
                num_gpus=4, num_cpus=0,
                resources={f"node:{host}": 0.001},
            ).remote()
            for host in manifest["addresses"][::2]
        ]
        for host, actor in zip(manifest["addresses"][::2], reservations, strict=True):
            assigned = sorted(ray.get(actor.gpu_ids.remote()))  # pyright: ignore[reportFunctionMemberAccess]
            if assigned != [0, 1, 2, 3]:
                raise RuntimeError(f"Service node {host} GPU reservation was {assigned}, not 0-3")
        if manifest["mapping_owned"]:
            _wait_health(
                processes, manifest["head_ip"], "mapping", _MAPPING_PORT,
                {"mapping_version": Path(manifest["mapping"]).name,
                 "bytes": Path(manifest["mapping"]).stat().st_size,
                 "service_config": {"mapping": manifest["mapping"]}},
            )
        for index, host in enumerate(manifest["addresses"][::2]):
            _wait_health(
                processes, host, f"baseline-{index}", _BASELINE_PORT,
                {"worker_count": 4, "docids_per_query": 64},
            )
            for port in _REWARD_PORTS:
                _wait_health(
                    processes, host, f"reward-{index}-{port}", port,
                    {"worker_count": 1, "mapping_version": Path(manifest["mapping"]).name},
                )
        _wait_health(
            processes, manifest["head_ip"], "reward-router", _ROUTER_PORT,
            {"worker_count": 8 * (len(manifest["hosts"]) // 2),
             "mapping_version": Path(manifest["mapping"]).name,
             "service_config": {"worker_urls": worker_urls}},
        )
        training = processes.start(
            "training",
            [sys.executable, "-u", "-m", _MODULE, "_train", str(path)],
            training=True,
        )
        while training.poll() is None:
            processes.check(except_name="training")
            time.sleep(5)
        if training.returncode:
            raise RuntimeError(f"train_async.py exited {training.returncode}: {run_dir / 'training.log'}")
        tracker = run_dir / "checkpoints/latest_checkpointed_iteration.txt"
        completed = int(tracker.read_text().strip())
        if completed != manifest["num_rollout"]:
            raise RuntimeError(f"Expected {manifest['num_rollout']} checkpointed steps, got {completed}")
        print(f"Completed {completed} GRPO optimizer steps: {run_dir}", flush=True)
    finally:
        if ray_connected:
            ray.shutdown()
        processes.close()
        if head is not None:
            head.kill_all_processes(check_alive=False, allow_graceful=True, wait=True)


def main() -> None:
    _disable_uv_runtime_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", default="node-0,node-1")
    parser.add_argument("--run-id", default=create_run_id())
    parser.add_argument("--num-rollout", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "_role", nargs="?",
        choices=("_worker", "_service_node", "_prepare_assets", "_train"),
    )
    parser.add_argument("_manifest", nargs="?")
    parser.add_argument("_index", nargs="?", type=int)
    cli = parser.parse_args()
    if cli._role:
        if not cli._manifest:
            parser.error("Internal role requires a manifest")
        manifest = cast(Manifest, json.loads(Path(cli._manifest).read_text()))
        if cli._role == "_train":
            _train(manifest)
        else:
            if cli._index is None or not 0 <= cli._index < len(manifest["hosts"]):
                parser.error("Internal node role requires a valid node index")
            if cli._role == "_worker":
                _worker(manifest, cli._index)
            elif cli._role == "_service_node":
                _service_node(manifest, cli._index)
            else:
                _prepare_node_assets(manifest)
        return
    if cli._manifest or cli._index is not None:
        parser.error("Manifest path and index are only accepted with an internal role")
    hosts, addresses = _hosts(cli.hosts)
    pairs = len(hosts) // 2
    args = _recipe(
        cli.run_id, pairs,
        cli.num_rollout if cli.num_rollout is not None else 3750 // pairs,
    )
    manifest = _manifest(
        args, hosts=hosts, addresses=addresses,
        create=not cli.dry_run, resume=cli.resume,
    )
    if cli.dry_run:
        print(json.dumps({
            "hosts": hosts,
            "num_rollout": manifest["num_rollout"],
            "prompt_source": manifest["prompt_source"],
            "mapping": manifest["mapping"],
            "mapping_url": manifest["mapping_url"],
            "mapping_owned": manifest["mapping_owned"],
            "checkpoint": manifest["checkpoint"],
            "qwen_model": manifest["qwen_model"],
            "relevance_model": manifest["relevance_model"],
            "train_argv": manifest["train_argv"],
        }, indent=2))
        return
    path = Path(manifest["run_dir"]) / "manifest.json"
    if cli.resume:
        previous = cast(Manifest, json.loads(path.read_text()))
        unchanged = (
            "hosts", "addresses", "checkpoint", "qwen_model", "mapping",
            "prompt_source", "relevance_model", "num_rollout", "run_id",
        )
        if any(previous[key] != manifest[key] for key in unchanged):
            raise ValueError("Resume configuration differs from the original run")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Async GRPO run manifest: {path}", flush=True)
    _run(manifest)


if __name__ == "__main__":
    main()
