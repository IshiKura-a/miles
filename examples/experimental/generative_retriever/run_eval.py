"""Single-node eight-GPU offline good-count evaluation launcher for SLMv3 generative retrieval.

Exports the trainable policy head from an FSDP DCP checkpoint, starts local
mapping, policy-generation, and relevance-scoring services on dedicated GPUs,
runs the resumable 10,000-query top-64 evaluation, and writes summary.json.

Example:
    uv run --env-file .env python -m examples.experimental.generative_retriever.run_eval \
        --run-id 261027-bs64-v2 --step 1560
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Literal

import httpx

from examples.experimental.generative_retriever.eval.good_count_eval import run as run_good_count_eval
from examples.experimental.generative_retriever.model.model_loading import export_policy_head_checkpoint
from examples.experimental.generative_retriever.reward.relevance_reward import DEFAULT_EVAL_GROUP_SIZE
from examples.experimental.generative_retriever.runtime.prepare_rocm_env import build_service_env

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_GENERATOR_DEVICES = "cuda:0,cuda:1,cuda:2,cuda:3"
_DEFAULT_SCORER_DEVICES = "cuda:4,cuda:5,cuda:6,cuda:7"
_MAPPING_PORT = 30488
_GENERATOR_PORT = 30588
_SCORER_PORT = 30689
_SERVICE_START_TIMEOUT = 1800.0


@dataclass(frozen=True)
class EvalConfig:
    run_id: str = "261027-bs64-v2"
    step: int | None = None
    output_dir: str = "$cosmos_data_path/local/User/zihao/online_rl/gr_offline_syn"
    base_eval_root: str = (
        "$cosmos_data_path/local/User/zihao/online_rl/gr_offline_syn/"
        "261021-114423-121/eval/valid_100k_10k_top64_step840"
    )
    eval_root: str | None = None
    checkpoint: str = (
        "$cosmos_data_path/local/Users/tianchiyang/Checkpoints/ASI_RQ/"
        "SLMv3-202511-4x256-stage5/epoch1-part24/checkpoint-30553"
    )
    qwen_model: str = "$cosmos_data_path/local/Users/tianchiyang/LLM/Qwen/Qwen3-4B"
    slm_root: str = "/home/aiscuser/SLMGR_dev/SLMv3"
    id_mapping: str = (
        "$cosmos_data_path/local/User/zihao/online_rl/gr_offline_syn/index/"
        "slmgrv3_part25_na_intl_20261028_20261029"
    )
    relevance_model: str = (
        "$cosmos_data_path/local/User/yugao/share/qualitymodel/"
        "GCQC/GV2.1/ForArcher/Large"
    )
    generator_devices: str = _DEFAULT_GENERATOR_DEVICES
    scorer_devices: str = _DEFAULT_SCORER_DEVICES
    batch_size: int = DEFAULT_EVAL_GROUP_SIZE
    scorer_batch_size: int = 256
    docids_per_query: int = DEFAULT_EVAL_GROUP_SIZE
    score_head: Literal["sm", "pm"] = "pm"
    mapping_port: int = _MAPPING_PORT
    generator_port: int = _GENERATOR_PORT
    scorer_port: int = _SCORER_PORT


@dataclass(frozen=True)
class _ManagedService:
    name: str
    port: int
    process: subprocess.Popen[bytes]
    log_handle: IO[str]


def _expand_path(value: str, name: str, *, expect_dir: bool = False, expect_file: bool = False) -> Path:
    expanded = Path(os.path.expandvars(value)).expanduser().resolve()
    if "$" in os.fspath(expanded):
        raise ValueError(f"{name} contains an unresolved environment variable: {value}")
    if expect_dir and not expanded.is_dir():
        raise FileNotFoundError(f"{name} directory does not exist: {expanded}")
    if expect_file and not expanded.is_file():
        raise FileNotFoundError(f"{name} file does not exist: {expanded}")
    return expanded


def _resolve_step(run_dir: Path, requested_step: int | None) -> int:
    if requested_step is not None:
        if requested_step <= 0:
            raise ValueError("Evaluation step must be positive")
        return requested_step
    ckpt_root = run_dir / "checkpoints"
    latest_file = ckpt_root / "latest_checkpointed_iteration.txt"
    if latest_file.is_file():
        raw = latest_file.read_text(encoding="utf-8").strip()
        if raw.isdigit():
            return int(raw)
    candidates = sorted(
        int(entry.name.removeprefix("iter_"))
        for entry in ckpt_root.glob("iter_*")
        if entry.is_dir()
        and entry.name.removeprefix("iter_").isdigit()
        and (entry / "model/.metadata").is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"No complete iter_* checkpoints found in {ckpt_root}")
    return candidates[-1]


def _prepare_eval_directory(
    *,
    eval_root: Path,
    base_eval_root: Path,
    dcp_model_dir: Path,
    step: int,
) -> Path:
    eval_root.mkdir(parents=True, exist_ok=True)
    for filename in ("queries.jsonl", "evaluation.json", "baseline.jsonl"):
        target = eval_root / filename
        source = base_eval_root / filename
        if not target.is_file() and source.is_file():
            shutil.copy2(source, target)
    for required in ("queries.jsonl", "baseline.jsonl"):
        if not (eval_root / required).is_file():
            raise FileNotFoundError(f"Missing required evaluation input: {eval_root / required}")

    policy_pt = eval_root / f"policy-head-step{step}.pt"
    if not policy_pt.is_file() or policy_pt.stat().st_size == 0:
        tensor_count = export_policy_head_checkpoint(dcp_model_dir, policy_pt)
        print(f"Exported {tensor_count} policy-head tensors -> {policy_pt}", flush=True)
    return policy_pt


def _assert_port_free(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"Evaluation port 127.0.0.1:{port} is already occupied")


def _start_service(
    *,
    name: str,
    module: str,
    argv: list[str],
    port: int,
    log_path: Path,
) -> _ManagedService:
    _assert_port_free(port)
    env = build_service_env("", _REPO_ROOT)
    log_handle = log_path.open("a", encoding="utf-8")
    command = [
        sys.executable,
        "-u",
        "-m",
        module,
        *argv,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    process = subprocess.Popen(
        command,
        cwd=_REPO_ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return _ManagedService(name=name, port=port, process=process, log_handle=log_handle)


def _wait_health(service: _ManagedService, timeout: float = _SERVICE_START_TIMEOUT) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{service.port}/health"
    while time.monotonic() < deadline:
        exit_code = service.process.poll()
        if exit_code is not None:
            raise RuntimeError(f"Service {service.name} on port {service.port} exited with code {exit_code}")
        try:
            response = httpx.get(url, timeout=5.0)
            if response.status_code == 200:
                payload = response.json()
                if isinstance(payload, dict) and payload.get("status") == "ready":
                    return payload
        except httpx.HTTPError:
            pass
        time.sleep(2.0)
    raise TimeoutError(f"Timed out waiting for {service.name} at {url}")


def _stop_services(services: list[_ManagedService]) -> None:
    for service in reversed(services):
        if service.process.poll() is None:
            service.process.terminate()
            try:
                service.process.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                service.process.kill()
                service.process.wait(timeout=15.0)
        service.log_handle.close()


def run_evaluation(config: EvalConfig) -> dict[str, Any] | None:
    output_root = _expand_path(config.output_dir, "output_dir", expect_dir=True)
    run_dir = output_root / config.run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Training run directory does not exist: {run_dir}")
    step = _resolve_step(run_dir, config.step)
    step_pad = f"{step:07d}"
    dcp_model_dir = run_dir / "checkpoints" / f"iter_{step_pad}" / "model"
    eval_root = (
        _expand_path(config.eval_root, "eval_root")
        if config.eval_root is not None
        else run_dir / "eval" / f"valid_100k_10k_top64_step{step}"
    )
    base_eval_root = _expand_path(config.base_eval_root, "base_eval_root", expect_dir=True)
    stage5_ckpt = _expand_path(config.checkpoint, "checkpoint", expect_dir=True)
    qwen_model = _expand_path(config.qwen_model, "qwen_model", expect_dir=True)
    slm_root = _expand_path(config.slm_root, "slm_root", expect_dir=True)
    id_mapping = _expand_path(config.id_mapping, "id_mapping")
    if not id_mapping.exists():
        raise FileNotFoundError(f"id_mapping does not exist: {id_mapping}")
    relevance_model = _expand_path(config.relevance_model, "relevance_model", expect_dir=True)

    policy_pt = _prepare_eval_directory(
        eval_root=eval_root,
        base_eval_root=base_eval_root,
        dcp_model_dir=dcp_model_dir,
        step=step,
    )

    services: list[_ManagedService] = []
    pkg = "examples.experimental.generative_retriever.services."
    try:
        mapping_svc = _start_service(
            name="mapping",
            module=pkg + "mapping_service",
            argv=["--mapping", str(id_mapping)],
            port=config.mapping_port,
            log_path=eval_root / "mapping.log",
        )
        services.append(mapping_svc)
        mapping_health = _wait_health(mapping_svc)
        mapping_version = str(mapping_health["mapping_version"])

        generator_svc = _start_service(
            name="policy_gen",
            module=pkg + "baseline_service",
            argv=[
                "--checkpoint",
                str(stage5_ckpt),
                "--policy-checkpoint",
                str(policy_pt),
                "--qwen-model",
                str(qwen_model),
                "--slm-root",
                str(slm_root),
                "--id-mapping",
                str(id_mapping),
                "--docids-per-query",
                str(config.docids_per_query),
                "--devices",
                config.generator_devices,
            ],
            port=config.generator_port,
            log_path=eval_root / "policy_gen.log",
        )
        services.append(generator_svc)

        scorer_svc = _start_service(
            name="good_count",
            module=pkg + "good_count_service",
            argv=[
                "--mapping-url",
                f"http://127.0.0.1:{config.mapping_port}",
                "--mapping-version",
                mapping_version,
                "--relevance-model",
                str(relevance_model),
                "--devices",
                config.scorer_devices,
                "--batch-size",
                str(config.scorer_batch_size),
            ],
            port=config.scorer_port,
            log_path=eval_root / "good_count.log",
        )
        services.append(scorer_svc)

        _wait_health(generator_svc)
        _wait_health(scorer_svc)

        return run_good_count_eval(
            mode="policy",
            manifest=eval_root / "queries.jsonl",
            baseline_results=eval_root / "baseline.jsonl",
            output=eval_root / "policy.jsonl",
            summary_output=eval_root / "summary.json",
            generate_url=f"http://127.0.0.1:{config.generator_port}/generate",
            score_url=f"http://127.0.0.1:{config.scorer_port}/score",
            batch_size=config.batch_size,
            docids_per_query=config.docids_per_query,
            score_head=config.score_head,
        )
    finally:
        _stop_services(services)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="261027-bs64-v2")
    parser.add_argument("--step", type=int)
    parser.add_argument(
        "--output-dir",
        default="$cosmos_data_path/local/User/zihao/online_rl/gr_offline_syn",
    )
    parser.add_argument(
        "--base-eval-root",
        default=(
            "$cosmos_data_path/local/User/zihao/online_rl/gr_offline_syn/"
            "261021-114423-121/eval/valid_100k_10k_top64_step840"
        ),
    )
    parser.add_argument("--eval-root")
    parser.add_argument(
        "--checkpoint",
        default=(
            "$cosmos_data_path/local/Users/tianchiyang/Checkpoints/ASI_RQ/"
            "SLMv3-202511-4x256-stage5/epoch1-part24/checkpoint-30553"
        ),
    )
    parser.add_argument(
        "--qwen-model",
        default="$cosmos_data_path/local/Users/tianchiyang/LLM/Qwen/Qwen3-4B",
    )
    parser.add_argument("--slm-root", default="/home/aiscuser/SLMGR_dev/SLMv3")
    parser.add_argument(
        "--id-mapping",
        default=(
            "$cosmos_data_path/local/User/zihao/online_rl/gr_offline_syn/index/"
            "slmgrv3_part25_na_intl_20261028_20261029"
        ),
    )
    parser.add_argument(
        "--relevance-model",
        default=(
            "$cosmos_data_path/local/User/yugao/share/qualitymodel/"
            "GCQC/GV2.1/ForArcher/Large"
        ),
    )
    parser.add_argument("--generator-devices", default=_DEFAULT_GENERATOR_DEVICES)
    parser.add_argument("--scorer-devices", default=_DEFAULT_SCORER_DEVICES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_EVAL_GROUP_SIZE)
    parser.add_argument("--scorer-batch-size", type=int, default=256)
    parser.add_argument("--docids-per-query", type=int, default=DEFAULT_EVAL_GROUP_SIZE)
    parser.add_argument("--score-head", choices=("sm", "pm"), default="pm")
    parser.add_argument("--mapping-port", type=int, default=_MAPPING_PORT)
    parser.add_argument("--generator-port", type=int, default=_GENERATOR_PORT)
    parser.add_argument("--scorer-port", type=int, default=_SCORER_PORT)
    args = parser.parse_args()
    run_evaluation(
        EvalConfig(
            run_id=args.run_id,
            step=args.step,
            output_dir=args.output_dir,
            base_eval_root=args.base_eval_root,
            eval_root=args.eval_root,
            checkpoint=args.checkpoint,
            qwen_model=args.qwen_model,
            slm_root=args.slm_root,
            id_mapping=args.id_mapping,
            relevance_model=args.relevance_model,
            generator_devices=args.generator_devices,
            scorer_devices=args.scorer_devices,
            batch_size=args.batch_size,
            scorer_batch_size=args.scorer_batch_size,
            docids_per_query=args.docids_per_query,
            score_head=args.score_head,
            mapping_port=args.mapping_port,
            generator_port=args.generator_port,
            scorer_port=args.scorer_port,
        )
    )


if __name__ == "__main__":
    main()
