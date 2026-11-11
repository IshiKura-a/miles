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

from examples.experimental.generative_retriever.utils.paths import absolute_path, symlink_target

import argparse
import asyncio
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
import tempfile
import time
import uuid
from typing import cast

import ray
from ray._private import ray_constants
from ray._private.node import Node
from ray._private.parameter import RayParams

from examples.experimental.generative_retriever.index.id_text_mapping import COLUMNAR_INDEX_FILES, is_columnar_index_dir
from examples.experimental.generative_retriever.runtime.training_common import (
    ScriptArgs, count_records, expanded_path, prepare_prompt_data, required_directory,
)
from examples.experimental.generative_retriever.model.model_loading import (
    stage_local_model_dir,
)
from examples.experimental.generative_retriever.reward.relevance_reward import (
    DEFAULT_REWARD_BATCH_SIZE,
    PROMPT_DATASET_ROWS,
    REWARD_PAIR_BATCH_SIZE,
    REWARD_TOWER_BATCH_SIZE,
)
from examples.experimental.generative_retriever.runtime.service_processes import (
    ProcessSupervisor,
    claim_mapping,
    health_matches,
    select_mapping_service,
    release_mapping,
    start_service,
    stop_process,
    wait_health,
)
from examples.experimental.generative_retriever.runtime.prepare_rocm_env import (
    derive_reward_cache_sizes,
)
from examples.experimental.generative_retriever.runtime.prepare_sglang_bundle import (
    prepare_bundle,
    require_sglang_abort_patch,
)
from examples.experimental.generative_retriever.runtime.multinode_recipe import (
    MAX_STEPS_PER_PAIR,
    QUERIES_PER_STEP_PER_PAIR,
    REWARD_GPUS_PER_NODE,
    REWARD_NODES_PER_PAIR,
    ROLLOUT_GPUS_PER_PAIR,
    SAMPLES_PER_PROMPT,
    build_multinode_recipe,
    build_multinode_runtime_env,
    build_multinode_train_args,
)
from examples.experimental.generative_retriever.runtime.multinode_setup import (
    Manifest,
    check_resume,
    distribute_manifest,
    prepare_hosts,
    remove_remote_manifests,
    validate_cluster,
    write_record,
)
from examples.experimental.generative_retriever.services.baseline_service import (
    BASELINE_CACHE_SIZE,
    BASELINE_SERVICE_API_VERSION,
)
from miles.utils.external_utils.command_utils import create_run_id

_REPO = Path(__file__).absolute().parents[3]
_MODULE = "examples.experimental.generative_retriever.run_grpo_multinode_async"
_RESIDENT_MAPPING_PORT = 31480
_BASELINE_PORT = 31581
_REWARD_PORTS = (31583,)
_REWARD_DEVICES = tuple(f"cuda:{i}" for i in range(REWARD_GPUS_PER_NODE))
_ROUTER_PORT = 31593
_RAY_PORT = 16380




def disable_uv_runtime_env() -> None:
    # Ray would replay uv --env-file .env inside a packaged directory that excludes .env.
    os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False


@ray.remote
class ReserveServiceGpus:
    def gpu_ids(self) -> list[int]:
        return [int(value) for value in ray.get_gpu_ids()]


def resolve_hosts(text: str) -> tuple[list[str], list[str]]:
    hosts = text.split(",")
    if len(hosts) < 2 or len(hosts) % 2 or len(set(hosts)) != len(hosts) or any(
        not host.strip() or host != host.strip() for host in hosts
    ):
        raise ValueError("Pass distinct hostnames in service,learner pairs (2n nodes)")
    addresses = [socket.gethostbyname(host) for host in hosts]
    if len(set(addresses)) != len(addresses):
        raise ValueError("Each host must resolve to a different node IP")
    return hosts, addresses


def build_manifest(
    args: ScriptArgs, *, hosts: list[str], addresses: list[str],
    validate_checkpoints: bool, resume: bool,
) -> Manifest:
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY must be set for GRPO training")
    checkpoint = required_directory(args.checkpoint, "Stage5 checkpoint")
    qwen = required_directory(args.qwen_model, "Qwen encoder")
    slm_root = required_directory(args.slm_root, "SLMv3 source")
    relevance = required_directory(args.relevance_model, "relevance model")
    mapping = absolute_path(os.path.expandvars(args.id_mapping))
    prompt = absolute_path(os.path.expandvars(args.prompt_data))
    if not is_columnar_index_dir(mapping):
        raise FileNotFoundError(f"Incomplete official mapping index: {mapping}")
    if not prompt.is_file():
        raise FileNotFoundError(f"Prompt dataset is missing: {prompt}")
    mapping_url, mapping_owned = select_mapping_service(mapping, addresses[0], resident_port=_RESIDENT_MAPPING_PORT)
    output = expanded_path(args.output_dir, "output_dir", directory=False)
    run_dir = output / args.run_id
    checkpoint_dir = run_dir / "checkpoints"
    alias = run_dir / "prompt_data.jsonl"
    local_bundle = local_bundle_path(args.run_id)
    count = count_records(prompt)
    if count != PROMPT_DATASET_ROWS or args.rollout_batch_size * args.num_rollout > count:
        raise ValueError(f"Expected {PROMPT_DATASET_ROWS} source rows without wrapping, got {count}")
    mapping_checkpoint = None
    if validate_checkpoints:
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
            mapping_checkpoint = str(checkpoint_dir / f"iter_{iteration:07d}" / "mapping")
            if not is_columnar_index_dir(mapping_checkpoint) or not (Path(mapping_checkpoint) / "overlay.parquet").is_file():
                raise FileNotFoundError("Resume requires the matching mapping overlay checkpoint")
        elif (checkpoint_dir / "latest_checkpointed_iteration.txt").is_file():
            raise FileExistsError(f"Run directory already contains checkpoints; pass --resume: {run_dir}")
    return {
        "run_id": args.run_id,
        "hosts": hosts,
        "addresses": addresses,
        "head_ip": addresses[0],
        "ray_address": f"{addresses[0]}:{_RAY_PORT}",
        "ray_temp": f"/tmp/gr_{hashlib.sha256(args.run_id.encode()).hexdigest()[:12]}",
        "run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "qwen_model": qwen,
        "slm_root": slm_root,
        "mapping": str(mapping),
        "mapping_url": mapping_url,
        "mapping_owned": mapping_owned,
        "mapping_owner": uuid.uuid4().hex,
        "mapping_checkpoint": mapping_checkpoint,
        "relevance_model": relevance,
        "prompt_source": str(prompt),
        "num_rollout": args.num_rollout,
        "train_argv": build_multinode_train_args(
            args, addresses=addresses, bundle=local_bundle, prompt=alias,
            reward_url=f"http://{addresses[0]}:{_ROUTER_PORT}",
            checkpoint_dir=checkpoint_dir, resume=resume,
        ),
        "runtime_env": build_multinode_runtime_env(
            checkpoint=checkpoint, qwen=qwen, slm_root=slm_root,
            mapping=str(mapping), addresses=addresses,
        ),
    }


def start_baseline(
    processes: ProcessSupervisor, manifest: Manifest, host: str,
) -> tuple[subprocess.Popen[str], dict[str, object]] | None:
    mapping = manifest["mapping"]
    expected: dict[str, object] = {
        "status": "ready",
        "worker_count": ROLLOUT_GPUS_PER_PAIR,
        "docids_per_query": SAMPLES_PER_PROMPT,
        "service_config": {
            "api_version": BASELINE_SERVICE_API_VERSION,
            "checkpoint": manifest["checkpoint"],
            "qwen_model": manifest["qwen_model"],
            "slm_root": manifest["slm_root"],
            "id_mapping": mapping,
            "devices": [f"cuda:{gpu}" for gpu in range(ROLLOUT_GPUS_PER_PAIR)],
            "cache_size": BASELINE_CACHE_SIZE,
            "docids_per_query": SAMPLES_PER_PROMPT,
        },
    }
    url = f"http://{host}:{_BASELINE_PORT}/health"
    if health_matches(url, expected):
        print(f"Reusing verified frozen baseline: {url}", flush=True)
        return None
    check_available_ports(host, (_BASELINE_PORT,))
    process = start_service(
        processes, "baseline", "baseline_service", host, _BASELINE_PORT,
        [
            "--checkpoint", manifest["checkpoint"],
            "--qwen-model", manifest["qwen_model"],
            "--slm-root", manifest["slm_root"],
            "--id-mapping", mapping, "--docids-per-query", str(SAMPLES_PER_PROMPT),
            "--devices", "cuda:0,cuda:1,cuda:2,cuda:3",
        ],
        gpus="0,1,2,3", persistent=True,
    )
    return process, expected


def exit_on_sigterm(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


def run_service_node(manifest: Manifest, index: int) -> None:
    os.environ.update(manifest["runtime_env"])
    host = manifest["addresses"][index]
    baseline_host = manifest["addresses"][(index // 2) * 2]
    baseline_url = f"http://{baseline_host}:{_BASELINE_PORT}/generate"
    mapping = manifest["mapping"]
    processes = ProcessSupervisor(Path(f"/tmp/gr_logs_{manifest['run_id']}") / f"services-{index}")
    previous_handler = signal.signal(signal.SIGTERM, exit_on_sigterm)
    baseline: tuple[subprocess.Popen[str], dict[str, object]] | None = None
    mapping_process: subprocess.Popen[str] | None = None
    mapping_claimed = False
    mapping_base_url = ""
    try:
        mapping_base_url, local_mapping_owned = select_mapping_service(
            Path(mapping), host, resident_port=_RESIDENT_MAPPING_PORT,
        )
        local_mapping_url = f"{mapping_base_url}/runs/{manifest['mapping_owner']}"
        if local_mapping_owned:
            check_available_ports(host, (_RESIDENT_MAPPING_PORT,))
            mapping_process = start_service(
                processes, "mapping", "mapping_service", host, _RESIDENT_MAPPING_PORT,
                ["--mapping", mapping], persistent=True,
            )
        baseline = None
        reward_health = []
        pairs = len(manifest["hosts"]) // 2
        query_cache, kw_cache, score_cache = derive_reward_cache_sizes(
            QUERIES_PER_STEP_PER_PAIR * pairs,
            SAMPLES_PER_PROMPT,
            REWARD_NODES_PER_PAIR * pairs,
        )
        check_available_ports(host, _REWARD_PORTS)
        port = _REWARD_PORTS[0]
        name = "reward-0"
        start_service(
            processes, name, "reward_service", host, port,
            [
                "--baseline-url", baseline_url,
                "--mapping-url", local_mapping_url,
                "--mapping-version", Path(mapping).name,
                "--relevance-model", manifest["relevance_model"],
                "--devices", ",".join(_REWARD_DEVICES),
                "--batch-size", str(REWARD_TOWER_BATCH_SIZE),
                "--pair-batch-size", str(REWARD_PAIR_BATCH_SIZE),
                "--docids-per-query", str(SAMPLES_PER_PROMPT),
                "--max-samples-per-request", str(DEFAULT_REWARD_BATCH_SIZE),
                "--query-cache-size", str(query_cache),
                "--keyword-cache-size", str(kw_cache),
                "--score-cache-size", str(score_cache),
            ],
            gpus=",".join(str(g) for g in range(REWARD_GPUS_PER_NODE)),
        )
        reward_health.append((name, port, {
            "worker_count": REWARD_GPUS_PER_NODE,
            "mapping_version": Path(mapping).name,
            "service_config": {
                "baseline_url": baseline_url,
                "mapping_url": local_mapping_url,
                "relevance_model": manifest["relevance_model"],
                "devices": list(_REWARD_DEVICES),
                "docids_per_query": SAMPLES_PER_PROMPT,
                "max_samples_per_request": DEFAULT_REWARD_BATCH_SIZE,
            },
        }))
        if local_mapping_owned:
            wait_health(
                processes, host, "mapping", _RESIDENT_MAPPING_PORT,
                {
                    "mapping_version": Path(mapping).name,
                    "service_config": {"mapping": mapping},
                },
            )
        claim_mapping(mapping_base_url, manifest["mapping_owner"], manifest["mapping_checkpoint"])
        mapping_claimed = True
        if baseline is not None:
            wait_health(processes, host, "baseline", _BASELINE_PORT, baseline[1])
            print(
                f"Frozen baseline will remain on GPU 0-3 after this run: "
                f"http://{host}:{_BASELINE_PORT}/health",
                flush=True,
            )
            baseline = None
        for name, port, expected in reward_health:
            wait_health(processes, host, name, port, expected)
        print(f"service node {index} ready", flush=True)
        while True:
            processes.check()
            ready, _, _ = select.select([sys.stdin], [], [], 2)
            if ready:
                if sys.stdin.readline().strip() == "STOP":
                    break
                raise RuntimeError("Service manager input closed without STOP")
    except BaseException:
        if mapping_process is not None and not mapping_claimed:
            stop_process(mapping_process, timeout=30, kill_timeout=30)
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
        try:
            processes.close()
        finally:
            if mapping_claimed:
                release_mapping(mapping_base_url, manifest["mapping_owner"])


def check_available_ports(host: str, ports: tuple[int, ...]) -> None:
    for port in ports:
        with socket.socket() as connection:
            connection.settimeout(1)
            if connection.connect_ex((host, port)) == 0:
                raise RuntimeError(f"Port {host}:{port} occupied; will not interrupt another job")


def remote_command(host: str, role: str, path: Path, index: int) -> list[str]:
    remote = (
        f"cd {shlex.quote(str(_REPO))} && "
        f"source {shlex.quote(str(_REPO / '.venv/activate-rocm.sh'))} && "
        f"exec {shlex.quote(sys.executable)} -u -m {_MODULE} {role} "
        f"{shlex.quote(str(path))} {index}"
    )
    return ["ssh", "-o", "BatchMode=yes", host, "bash", "-c", shlex.quote(remote)]


def ray_temp_path(run_id: str) -> str:
    return f"/tmp/gr_{hashlib.sha256(run_id.encode()).hexdigest()[:12]}"


def local_bundle_path(run_id: str) -> Path:
    return Path(f"/tmp/gr_bundle_{hashlib.sha256(run_id.encode()).hexdigest()[:12]}")


def prepare_node_assets(manifest: Manifest) -> None:
    os.environ.update(manifest["runtime_env"])
    run_dir = Path(manifest["run_dir"])
    source = Path(manifest["prompt_source"])
    alias = run_dir / "prompt_data.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"Prompt dataset is missing on this node: {source}")
    if alias.is_symlink():
        if symlink_target(alias) != source:
            raise FileExistsError(f"Prompt alias points to a different dataset: {alias}")
    elif alias.exists():
        raise FileExistsError(f"Prompt alias is not a symlink: {alias}")
    else:
        alias.symlink_to(source)
    if not alias.is_file():
        raise FileNotFoundError(f"Prompt alias is not readable on this node: {alias}")

    local_ckpt = stage_local_model_dir(manifest["checkpoint"], "checkpoint")
    local_qwen = stage_local_model_dir(manifest["qwen_model"], "Qwen model")
    bundle = prepare_bundle(
        checkpoint=local_ckpt,
        qwen_model=local_qwen,
        slm_root=manifest["slm_root"],
        id_mapping=manifest["mapping"],
        output=local_bundle_path(manifest["run_id"]),
        num_beams=SAMPLES_PER_PROMPT,
        num_return_sequences=SAMPLES_PER_PROMPT,
        temperature=1.0,
        decoding_mode="beam_sample",
    )
    for name in ("config.json", "tokenizer.json", "pytorch_model.bin.index.json"):
        if not (bundle / name).is_file():
            raise FileNotFoundError(f"Model bundle asset is not readable on this node: {bundle / name}")


def validate_run_cluster(manifest: Manifest) -> None:
    require_sglang_abort_patch()
    sglang_root = Path(sys.prefix) / "src/sglang"
    patch = Path(__file__).absolute().parent / "runtime/patches/sglang-embedding-abort.patch"
    source_files = (
        sglang_root / "python/sglang/srt/entrypoints/openai/serving_embedding.py",
        Path(manifest["slm_root"]) / "model.py",
        Path(manifest["slm_root"]) / "utils.py",
        Path(manifest["slm_root"]) / "CustomizedEncoder/__init__.py",
        Path(manifest["slm_root"]) / "CustomizedEncoder/Encoder_QW1216.py",
        Path(manifest["slm_root"]) / "CustomizedEncoder/Encoder_QW1216text.py",
        Path(manifest["checkpoint"]) / "pytorch_model.bin.index.json",
        Path(manifest["qwen_model"]) / "model.safetensors.index.json",
    )
    required = (
        (Path(manifest["prompt_source"]), "f"),
        *((Path(manifest["mapping"]) / name, "f") for name in COLUMNAR_INDEX_FILES),
        (Path(sys.executable), "x"),
        (_REPO / ".venv/activate-rocm.sh", "f"),
    )
    validate_cluster(
        hosts=manifest["hosts"], repo=_REPO, sglang_root=sglang_root,
        source_files=source_files, required=required, patch=patch.read_text(),
    )


def run_worker(manifest: Manifest, index: int) -> None:
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
            temp_dir=ray_temp_path(manifest["run_id"]), include_dashboard=False,
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


def run_training(manifest: Manifest) -> None:
    from train import train
    from miles.utils.arguments import parse_args
    from miles.utils.tracking_utils.tracking import finish_tracking

    key = os.environ.get("WANDB_API_KEY")
    if not key:
        raise RuntimeError("WANDB_API_KEY must be set in the training driver")
    runtime_env = {"env_vars": manifest["runtime_env"]}
    ray.init(address=manifest["ray_address"], runtime_env=runtime_env)
    sys.argv = [str(_REPO / "train.py"), *manifest["train_argv"]]
    try:
        args = parse_args()
        try:
            asyncio.run(train(args))
        finally:
            finish_tracking()
    finally:
        ray.shutdown()


def run_cluster(manifest: Manifest) -> None:
    run_dir = Path(manifest["run_dir"])
    if ray.util.get_node_ip_address() != manifest["head_ip"]:
        raise RuntimeError("Launch on the first service/rollout host listed in --hosts")
    validate_run_cluster(manifest)
    run_dir.mkdir(parents=True, exist_ok=True)
    prepare_prompt_data(Path(manifest["prompt_source"]), run_dir / "prompt_data.jsonl")
    processes = ProcessSupervisor(Path(f"/tmp/gr_logs_{manifest['run_id']}"))
    local_dir = tempfile.TemporaryDirectory(prefix="gr_manifest_", dir="/tmp")
    path = Path(local_dir.name) / "manifest.json"
    head: Node | None = None
    ray_connected = False
    distributed = False
    try:
        record, digest = write_record(manifest)
        manifest_text = json.dumps(manifest, indent=2) + "\n"
        path.write_text(manifest_text, encoding="utf-8")
        distributed = True
        distribute_manifest(manifest["hosts"], path, manifest_text, record, digest)
        prepare_hosts(
            manifest["hosts"],
            lambda host, index: (
                [sys.executable, "-u", "-m", _MODULE, "_prepare_assets", str(path), str(index)]
                if index == 0 else remote_command(host, "_prepare_assets", path, index)
            ),
        )
        print(f"Async GRPO run manifest: {record}", flush=True)
        check_available_ports(manifest["head_ip"], (_RAY_PORT, _ROUTER_PORT))
        for node, host in enumerate(manifest["hosts"]):
            argv = (
                [sys.executable, "-u", "-m", _MODULE, "_service_node", str(path), str(node)]
                if node == 0 else remote_command(host, "_service_node", path, node)
            )
            processes.start(f"services-{node}", argv, stdin=True)
        worker_urls = [
            f"http://{host}:{port}/score"
            for host in manifest["addresses"] for port in _REWARD_PORTS
        ]
        start_service(
            processes, "reward-router", "reward_router_service",
            manifest["head_ip"], _ROUTER_PORT,
            ["--worker-urls", ",".join(worker_urls),
             "--max-samples-per-request", str(DEFAULT_REWARD_BATCH_SIZE),
             "--forward-concurrency", "2"],
        )
        os.environ.update(manifest["runtime_env"])
        head = Node(
            RayParams(
                node_ip_address=manifest["head_ip"], gcs_server_port=_RAY_PORT,
                num_cpus=32, num_gpus=8, object_store_memory=32 * 2**30,
                temp_dir=ray_temp_path(manifest["run_id"]), include_dashboard=False,
            ),
            head=True, shutdown_at_exit=True, spawn_reaper=True,
        )
        for index, host in enumerate(manifest["hosts"][1:], start=1):
            worker = processes.start(
                f"ray-worker-{index}", remote_command(host, "_worker", path, index),
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
        # All 64 GPUs are colocated for rollout, baseline, reward, and training.
        for node, host in enumerate(manifest["addresses"]):
            wait_health(
                processes, host, f"mapping-{node}", _RESIDENT_MAPPING_PORT,
                {"mapping_version": Path(manifest["mapping"]).name},
                health_path=f"/runs/{manifest['mapping_owner']}/health",
            )
            for port in _REWARD_PORTS:
                wait_health(
                    processes, host, f"reward-{node}-{port}", port,
                    {"worker_count": REWARD_GPUS_PER_NODE, "mapping_version": Path(manifest["mapping"]).name},
                )
        wait_health(
            processes, manifest["head_ip"], "reward-router", _ROUTER_PORT,
            {"worker_count": len(manifest["hosts"]),
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
        if distributed:
            remove_remote_manifests(manifest["hosts"], path)
        local_dir.cleanup()


def main() -> None:
    disable_uv_runtime_env()
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
            run_training(manifest)
        else:
            if cli._index is None or not 0 <= cli._index < len(manifest["hosts"]):
                parser.error("Internal node role requires a valid node index")
            if cli._role == "_worker":
                run_worker(manifest, cli._index)
            elif cli._role == "_service_node":
                run_service_node(manifest, cli._index)
            else:
                prepare_node_assets(manifest)
        return
    if cli._manifest or cli._index is not None:
        parser.error("Manifest path and index are only accepted with an internal role")
    hosts, addresses = resolve_hosts(cli.hosts)
    pairs = len(hosts) // 2
    args = build_multinode_recipe(
        cli.run_id, pairs,
        cli.num_rollout if cli.num_rollout is not None else MAX_STEPS_PER_PAIR // pairs,
    )
    manifest = build_manifest(
        args, hosts=hosts, addresses=addresses,
        validate_checkpoints=not cli.dry_run, resume=cli.resume,
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
    if cli.resume:
        check_resume(manifest)
    run_cluster(manifest)


if __name__ == "__main__":
    main()
