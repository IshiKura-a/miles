"""Single-node eight-GPU offline good-count evaluation launcher for SLMv3 generative retrieval.

Exports the trainable policy head from an FSDP DCP checkpoint, starts local
mapping, policy-generation, and relevance-scoring services on dedicated GPUs,
runs the resumable 10,000-query top-64 evaluation, and writes summary.json.

Example:
    uv run --env-file .env python -m examples.experimental.generative_retriever.run_eval \
        --run-id 261027-bs64-v2 --step 1560
"""

from __future__ import annotations

from examples.experimental.generative_retriever.utils.paths import absolute_path

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import IO, Any, Literal

import httpx

from examples.experimental.generative_retriever.eval.manifest import (
    EvalManifest,
    baseline_cache_path,
    eval_dir_name,
    load_eval_manifest,
    publish_baseline_cache,
    stage_eval_inputs,
)
from examples.experimental.generative_retriever.eval.good_count_eval import run as run_good_count_eval
from examples.experimental.generative_retriever.index.id_text_mapping import is_columnar_index_dir
from examples.experimental.generative_retriever.model.model_loading import export_policy_head_checkpoint
from examples.experimental.generative_retriever.reward.relevance_reward import (
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_EVAL_GROUP_SIZE,
)
from examples.experimental.generative_retriever.runtime.prepare_rocm_env import build_service_env
from examples.experimental.generative_retriever.runtime.service_processes import stop_process

_REPO_ROOT = Path(__file__).absolute().parents[3]
_DEFAULT_GENERATOR_DEVICES = "cuda:0,cuda:1,cuda:2,cuda:3"
_DEFAULT_SCORER_DEVICES = "cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7"
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
        "eval_sets/valid_100k_10k"
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
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE
    scorer_batch_size: int = 8192
    docids_per_query: int = DEFAULT_EVAL_GROUP_SIZE
    policy_mapping_mode: Literal["checkpoint", "initial"] = "checkpoint"
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
    expanded = absolute_path(os.path.expandvars(value))
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
    output_root: Path,
    eval_root: Path,
    base_eval_root: Path,
    base_manifest: EvalManifest,
    dcp_model_dir: Path,
    step: int,
    initial_mapping_version: str,
    docids_per_query: int,
) -> tuple[Path, Path, bool]:
    manifest = stage_eval_inputs(
        eval_root,
        base_eval_root=base_eval_root,
        base_manifest=base_manifest,
    )
    cached_baseline = baseline_cache_path(
        output_root,
        manifest,
        docids_per_query=docids_per_query,
        mapping_version=initial_mapping_version,
    )
    policy_pt = eval_root / f"policy-head-step{step}.pt"
    if not policy_pt.is_file() or policy_pt.stat().st_size == 0:
        tensor_count = export_policy_head_checkpoint(dcp_model_dir, policy_pt)
        print(f"Exported {tensor_count} policy-head tensors -> {policy_pt}", flush=True)
    return policy_pt, cached_baseline, not cached_baseline.is_file()


def _scorer_already_ready(port: int, mapping_version: str, relevance_model: str) -> bool:
    try:
        resp = httpx.get(f"http://127.0.0.1:{port}/health", timeout=3.0)
        if resp.status_code != 200:
            return False
        data = resp.json()
        return (
            isinstance(data, dict)
            and data.get("status") == "ready"
            and data.get("mapping_version") == mapping_version
            and data.get("relevance_model") == relevance_model
        )
    except Exception:
        return False


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
    gpus: str | None = None,
) -> _ManagedService:
    _assert_port_free(port)
    env = build_service_env(gpus, _REPO_ROOT)
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
        stop_process(service.process, timeout=15.0, kill_timeout=15.0)
        service.log_handle.close()


def run_evaluation(config: EvalConfig) -> dict[str, Any] | None:
    output_root = _expand_path(config.output_dir, "output_dir", expect_dir=True)
    run_dir = output_root / config.run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Training run directory does not exist: {run_dir}")
    step = _resolve_step(run_dir, config.step)
    step_pad = f"{step:07d}"
    dcp_model_dir = run_dir / "checkpoints" / f"iter_{step_pad}" / "model"
    base_eval_root = _expand_path(config.base_eval_root, "base_eval_root", expect_dir=True)
    base_manifest = load_eval_manifest(base_eval_root)
    default_dir_name = eval_dir_name(base_manifest, docids_per_query=config.docids_per_query, step=step)
    if config.policy_mapping_mode == "initial":
        default_dir_name = f"{default_dir_name}_policy_only"
    eval_root = (
        _expand_path(config.eval_root, "eval_root")
        if config.eval_root is not None
        else run_dir / "eval" / default_dir_name
    )
    stage5_ckpt = _expand_path(config.checkpoint, "checkpoint", expect_dir=True)
    qwen_model = _expand_path(config.qwen_model, "qwen_model", expect_dir=True)
    slm_root = _expand_path(config.slm_root, "slm_root", expect_dir=True)
    id_mapping = _expand_path(config.id_mapping, "id_mapping")
    if not is_columnar_index_dir(id_mapping):
        raise FileNotFoundError(f"Incomplete initial mapping index: {id_mapping}")
    if config.policy_mapping_mode == "initial":
        policy_mapping = id_mapping
    else:
        ckpt_mapping_dir = run_dir / "checkpoints" / f"iter_{step_pad}" / "mapping"
        if not is_columnar_index_dir(ckpt_mapping_dir):
            raise FileNotFoundError(f"Incomplete checkpoint mapping index: {ckpt_mapping_dir}")
        policy_mapping = ckpt_mapping_dir
    relevance_model = _expand_path(config.relevance_model, "relevance_model", expect_dir=True)
    policy_pt, cached_baseline_path, need_baseline = _prepare_eval_directory(
        output_root=output_root,
        eval_root=eval_root,
        base_eval_root=base_eval_root,
        base_manifest=base_manifest,
        dcp_model_dir=dcp_model_dir,
        step=step,
        initial_mapping_version=id_mapping.name,
        docids_per_query=config.docids_per_query,
    )

    pkg = "examples.experimental.generative_retriever.services."
    target_baseline = eval_root / "baseline.jsonl"
    services: list[_ManagedService] = []
    try:
        scorer_svc: _ManagedService | None = None
        if not _scorer_already_ready(config.scorer_port, id_mapping.name, str(relevance_model)):
            scorer_svc = _start_service(
                name="good_count",
                module=pkg + "good_count_service",
                argv=[
                    "--mapping-url",
                    f"http://127.0.0.1:{config.mapping_port}",
                    "--mapping-version",
                    id_mapping.name,
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

        generator_argv = [
            "--checkpoint",
            str(stage5_ckpt),
            "--qwen-model",
            str(qwen_model),
            "--slm-root",
            str(slm_root),
            "--id-mapping",
            str(policy_mapping),
            "--docids-per-query",
            str(config.docids_per_query),
            "--devices",
            config.generator_devices,
        ]
        if not need_baseline:
            generator_argv.extend(["--policy-checkpoint", str(policy_pt)])
        generator_svc = _start_service(
            name="policy_gen",
            module=pkg + "baseline_service",
            argv=generator_argv,
            port=config.generator_port,
            log_path=eval_root / "policy_gen.log",
        )
        services.append(generator_svc)

        if need_baseline:
            print(
                f"Generating fresh baseline.jsonl for mapping {id_mapping.name} -> {target_baseline}",
                flush=True,
            )
            base_map_svc = _start_service(
                name="baseline_mapping",
                module=pkg + "mapping_service",
                argv=["--mapping", str(id_mapping)],
                port=config.mapping_port,
                log_path=eval_root / "baseline_mapping.log",
                gpus="",
            )
            try:
                _wait_health(base_map_svc)
                _wait_health(generator_svc)
                if scorer_svc is not None:
                    _wait_health(scorer_svc)

                run_good_count_eval(
                    mode="baseline",
                    manifest=eval_root / "queries.jsonl",
                    baseline_results=None,
                    output=target_baseline,
                    summary_output=eval_root / "baseline_summary.json",
                    generate_url=f"http://127.0.0.1:{config.generator_port}/generate",
                    score_url=f"http://127.0.0.1:{config.scorer_port}/score",
                    batch_size=config.batch_size,
                    docids_per_query=config.docids_per_query,
                    score_head=config.score_head,
                )
                publish_baseline_cache(target_baseline, cached_baseline_path)
            finally:
                _stop_services([base_map_svc])

            resp = httpx.post(
                f"http://127.0.0.1:{config.generator_port}/load_policy_head",
                json={"policy_checkpoint": str(policy_pt)},
                timeout=120.0,
            )
            resp.raise_for_status()

        mapping_svc = _start_service(
            name="mapping",
            module=pkg + "mapping_service",
            argv=["--mapping", str(policy_mapping)],
            port=config.mapping_port,
            log_path=eval_root / "mapping.log",
            gpus="",
        )
        services.append(mapping_svc)

        _wait_health(mapping_svc)
        _wait_health(generator_svc)
        if scorer_svc is not None:
            _wait_health(scorer_svc)

        return run_good_count_eval(
            mode="policy",
            manifest=eval_root / "queries.jsonl",
            baseline_results=cached_baseline_path,
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
    defaults = EvalConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=defaults.run_id)
    parser.add_argument("--step", type=int)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--base-eval-root", default=defaults.base_eval_root)
    parser.add_argument("--eval-root")
    parser.add_argument("--checkpoint", default=defaults.checkpoint)
    parser.add_argument("--qwen-model", default=defaults.qwen_model)
    parser.add_argument("--slm-root", default=defaults.slm_root)
    parser.add_argument("--id-mapping", default=defaults.id_mapping)
    parser.add_argument("--relevance-model", default=defaults.relevance_model)
    parser.add_argument("--generator-devices", default=defaults.generator_devices)
    parser.add_argument("--scorer-devices", default=defaults.scorer_devices)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--scorer-batch-size", type=int, default=defaults.scorer_batch_size)
    parser.add_argument("--docids-per-query", type=int, default=defaults.docids_per_query)
    parser.add_argument(
        "--policy-mapping-mode",
        choices=("checkpoint", "initial"),
        default=defaults.policy_mapping_mode,
    )
    parser.add_argument("--score-head", choices=("sm", "pm"), default=defaults.score_head)
    parser.add_argument("--mapping-port", type=int, default=defaults.mapping_port)
    parser.add_argument("--generator-port", type=int, default=defaults.generator_port)
    parser.add_argument("--scorer-port", type=int, default=defaults.scorer_port)
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
            policy_mapping_mode=args.policy_mapping_mode,
            score_head=args.score_head,
            mapping_port=args.mapping_port,
            generator_port=args.generator_port,
            scorer_port=args.scorer_port,
        )
    )


if __name__ == "__main__":
    main()
