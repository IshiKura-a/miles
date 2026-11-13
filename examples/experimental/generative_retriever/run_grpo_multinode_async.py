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
import shlex
import socket
import sys
import tempfile
import time
from typing import cast

import ray
from ray._private import ray_constants
from ray._private.node import Node
from ray._private.parameter import RayParams

from examples.experimental.generative_retriever.index.id_text_mapping import COLUMNAR_INDEX_FILES, is_columnar_index_dir
from examples.experimental.generative_retriever.model.model_loading import (
    stage_local_model_dir,
)
from examples.experimental.generative_retriever.reward.relevance_model import (
    _stage_local_relevance_model,
    _stage_local_relevance_source,
    relevance_source_path,
)
from examples.experimental.generative_retriever.reward.relevance_reward import (
    PROMPT_DATASET_ROWS,
)
from examples.experimental.generative_retriever.runtime.multinode_recipe import (
    MAX_STEPS_PER_PAIR,
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
from examples.experimental.generative_retriever.runtime.prepare_sglang_bundle import (
    prepare_bundle,
    require_sglang_abort_patch,
)
from examples.experimental.generative_retriever.runtime.service_processes import (
    ProcessSupervisor,
)
from examples.experimental.generative_retriever.runtime.training_common import (
    ScriptArgs, count_records, expanded_path, prepare_prompt_data, required_directory,
)
from examples.experimental.generative_retriever.sglang_models.gr_policy import (
    _stage_local_mapping_dir,
)
from miles.utils.external_utils.command_utils import create_run_id

_REPO = Path(__file__).absolute().parents[3]
_MODULE = "examples.experimental.generative_retriever.run_grpo_multinode_async"
_RAY_PORT = 16380


def disable_uv_runtime_env() -> None:
    # Ray would replay uv --env-file .env inside a packaged directory that excludes .env.
    os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False


def resolve_hosts(text: str) -> tuple[list[str], list[str]]:
    hosts = text.split(",")
    if len(hosts) < 2 or len(hosts) % 2 or len(set(hosts)) != len(hosts) or any(
        not host.strip() or host != host.strip() for host in hosts
    ):
        raise ValueError("Pass distinct hostnames in node pairs (2n nodes)")
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
    output = expanded_path(args.output_dir, "output_dir", directory=False)
    run_dir = output / args.run_id
    checkpoint_dir = run_dir / "checkpoints"
    alias = run_dir / "prompt_data.jsonl"
    local_bundle = local_bundle_path(args.run_id)
    count = count_records(prompt)
    if count != PROMPT_DATASET_ROWS or args.rollout_batch_size * args.num_rollout > count:
        raise ValueError(f"Expected {PROMPT_DATASET_ROWS} source rows without wrapping, got {count}")
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
        "relevance_model": relevance,
        "reward_mode": args.reward_mode,
        "prompt_source": str(prompt),
        "num_rollout": args.num_rollout,
        "train_argv": build_multinode_train_args(
            args, addresses=addresses, bundle=local_bundle, prompt=alias,
            checkpoint_dir=checkpoint_dir, resume=resume,
        ),
        "runtime_env": build_multinode_runtime_env(
            checkpoint=checkpoint, qwen=qwen, slm_root=slm_root,
            mapping=str(mapping), relevance_model=relevance,
            reward_mode=args.reward_mode, addresses=addresses,
        ),
    }


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
    local_mapping = _stage_local_mapping_dir(manifest["mapping"])
    _stage_local_relevance_source(relevance_source_path(manifest["relevance_model"]))
    local_relevance = _stage_local_relevance_model(manifest["relevance_model"])
    bundle = prepare_bundle(
        checkpoint=local_ckpt,
        qwen_model=local_qwen,
        slm_root=manifest["slm_root"],
        id_mapping=local_mapping,
        relevance_model=local_relevance,
        reward_mode=manifest["reward_mode"],
        output=local_bundle_path(manifest["run_id"]),
        num_beams=SAMPLES_PER_PROMPT,
        num_return_sequences=SAMPLES_PER_PROMPT,
        temperature=0.4,
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
            num_cpus=64, num_gpus=8, object_store_memory=128 * 2**30,
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
        check_available_ports(manifest["head_ip"], (_RAY_PORT,))
        os.environ.update(manifest["runtime_env"])
        head = Node(
            RayParams(
                node_ip_address=manifest["head_ip"], gcs_server_port=_RAY_PORT,
                num_cpus=32, num_gpus=8, object_store_memory=128 * 2**30,
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
        # All GPUs are colocated for in-process rollout, baseline, reward, and training.
        training = processes.start(
            "training",
            [sys.executable, "-u", "-m", _MODULE, "_train", str(path)],
            training=True,
        )
        while training.poll() is None:
            processes.check(except_name="training")
            time.sleep(5)
        if training.returncode:
            raise RuntimeError(f"train.py exited {training.returncode}: {run_dir / 'training.log'}")
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
        choices=("_worker", "_prepare_assets", "_train"),
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
