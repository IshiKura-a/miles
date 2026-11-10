"""Process ownership and health checks for GRPO services."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import IO, Any

import httpx

from examples.experimental.generative_retriever.runtime.prepare_rocm_env import build_service_env
from examples.experimental.generative_retriever.services.mapping_service import MAPPING_SERVICE_API_VERSION

_REPO = Path(__file__).absolute().parents[4]
_SERVICE_TIMEOUT = 3600


def health_matches(url: str, expected: dict[str, Any]) -> bool:
    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            response = client.get(url)
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


def mapping_service_matches(url: str, mapping: Path) -> bool:
    """Check the resident service's base index configuration."""
    return health_matches(f"{url}/health", {
        "status": "ready",
        "mapping_version": mapping.name,
        "service_config": {"api_version": MAPPING_SERVICE_API_VERSION, "mapping": str(mapping)},
    })


def select_mapping_service(mapping: Path, host: str, *, resident_port: int) -> tuple[str, bool]:
    resident_url = f"http://{host}:{resident_port}"
    if mapping_service_matches(resident_url, mapping):
        return resident_url, False
    return resident_url, True


def claim_mapping(url: str, owner: str, checkpoint: str | None = None) -> str:
    with httpx.Client(timeout=600, trust_env=False) as client:
        response = client.post(f"{url}/claim", json={"owner": owner, "checkpoint": checkpoint})
        response.raise_for_status()
    return f"{url}/runs/{owner}"


def release_mapping(url: str, owner: str) -> None:
    with httpx.Client(timeout=30, trust_env=False) as client:
        response = client.post(f"{url}/release", json={"owner": owner})
        response.raise_for_status()


def stop_process(process: subprocess.Popen[Any], *, timeout: float, kill_timeout: float) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=kill_timeout)


@dataclass
class _Child:
    name: str
    process: subprocess.Popen[str]
    stream: IO[str]


class ProcessSupervisor:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.children: list[_Child] = []
        self.persistent: list[_Child] = []
        log_dir.mkdir(parents=True, exist_ok=True)

    def start(
        self, name: str, argv: list[str], *,
        gpus: str | None = None, stdin: bool = False, training: bool = False,
        persistent: bool = False, service: bool = False,
    ) -> subprocess.Popen[str]:
        if service:
            env = build_service_env(gpus, _REPO)
        else:
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
                        pass
            stop_process(process, timeout=30, kill_timeout=30)
            child.stream.close()


def wait_health(
    processes: ProcessSupervisor, host: str, name: str, port: int,
    expected: dict[str, object], *, health_path: str = "/health",
) -> None:
    deadline = time.monotonic() + _SERVICE_TIMEOUT
    url = f"http://{host}:{port}{health_path}"
    with httpx.Client(timeout=10, trust_env=False) as client:
        while time.monotonic() < deadline:
            processes.check()
            try:
                response = client.get(url)
            except (httpx.ConnectError, httpx.TimeoutException):
                time.sleep(3)
                continue
            if response.status_code in (409, 503):
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


def start_service(
    processes: ProcessSupervisor, name: str, module: str, host: str, port: int,
    argv: list[str], *, gpus: str | None = None, persistent: bool = False,
) -> subprocess.Popen[str]:
    return processes.start(
        name,
        [sys.executable, "-u", "-m",
         f"examples.experimental.generative_retriever.services.{module}",
         *argv, "--host", host, "--port", str(port)],
        gpus=gpus, persistent=persistent, service=True,
    )
