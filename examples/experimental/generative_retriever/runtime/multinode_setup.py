"""Validate a training cluster and distribute its run manifest in parallel."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from typing import TypedDict, cast


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
    relevance_model: str
    reward_mode: str
    prompt_source: str
    num_rollout: int
    train_argv: list[str]
    runtime_env: dict[str, str]


_IDENTITY_KEYS = (
    "hosts", "addresses", "checkpoint", "qwen_model", "mapping", "prompt_source",
    "relevance_model", "num_rollout", "run_id",
)


def _identity(manifest: Manifest) -> dict[str, object]:
    values = dict(manifest)
    return {key: values[key] for key in _IDENTITY_KEYS}


def _record_path(manifest: Manifest) -> Path:
    encoded = json.dumps(_identity(manifest), sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    return Path(manifest["run_dir"]) / f"manifest-{digest}.json"


def check_resume(manifest: Manifest) -> None:
    """Check the immutable record for this run."""
    path = _record_path(manifest)
    previous = cast(Manifest, json.loads(path.read_text(encoding="utf-8")))
    if _identity(previous) != _identity(manifest):
        raise ValueError("Resume configuration differs from the original run")


def write_record(manifest: Manifest) -> tuple[Path, str]:
    """Persist an immutable record; the digest verifies what other nodes can read."""
    path = _record_path(manifest)
    if path.exists():
        stored = cast(Manifest, json.loads(path.read_text(encoding="utf-8")))
        if _identity(stored) != _identity(manifest):
            raise ValueError("Stored manifest configuration differs from the current run")
        data = path.read_bytes()
    else:
        data = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
        path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


def _parallel(hosts: Sequence[str], action: Callable[[str], None]) -> None:
    if hosts:
        with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
            list(pool.map(action, hosts))


def validate_cluster(
    *, hosts: Sequence[str], repo: Path, sglang_root: Path,
    source_files: Sequence[Path], required: Sequence[tuple[Path, str]], patch: str,
) -> None:
    """Validate every node before any node is prepared for this run."""
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    changes = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"], text=True,
    )
    if changes:
        raise RuntimeError("Miles checkout on the launch node is dirty; deploy a committed version before training")
    with ThreadPoolExecutor(max_workers=len(source_files)) as pool:
        expected = list(pool.map(lambda p: hashlib.sha256(p.read_bytes()).hexdigest(), source_files))

    git_repo = f"git -C {shlex.quote(str(repo))}"
    checks = [
        "set -e",
        f'test "$({git_repo} rev-parse HEAD)" = {shlex.quote(commit)} '
        '|| { echo "Miles checkout commit differs" >&2; exit 1; }',
        f'test -z "$({git_repo} status --porcelain --untracked-files=all)" '
        '|| { echo "Miles checkout is dirty" >&2; exit 1; }',
        "sha256sum " + " ".join(shlex.quote(str(path)) for path in source_files),
        f"git -C {shlex.quote(str(sglang_root))} apply --reverse --check -",
        *(
            f"test -{flag} {shlex.quote(str(path))} "
            f"|| {{ echo {shlex.quote(f'Cannot read required path: {path}')} >&2; exit 1; }}"
            for path, flag in required
        ),
    ]
    command = " && ".join(checks)

    def validate(host: str) -> None:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", host, command], input=patch,
            capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError(f"Preflight failed on {host}: {result.stderr.strip()}")
        actual = [line.split(maxsplit=1)[0] for line in result.stdout.splitlines()]
        if actual != expected:
            raise RuntimeError(f"Source/model indices differ on {host}; synchronize the checkout first")

    _parallel(hosts[1:], validate)


def distribute_manifest(
    hosts: Sequence[str], local_path: Path, text: str, record_path: Path, record_digest: str,
) -> None:
    """Write node-local copies and verify the shared recovery record on each node."""
    directory = shlex.quote(str(local_path.parent))
    target = shlex.quote(str(local_path))
    record = shlex.quote(str(record_path))
    command = f"umask 077 && mkdir -m 700 {directory} && cat > {target} && sha256sum {record}"

    def distribute(host: str) -> None:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", host, command], input=text,
            capture_output=True, text=True, check=True,
        )
        if result.stdout.split(maxsplit=1)[0] != record_digest:
            raise RuntimeError(f"Saved manifest differs on {host}: {record_path}")

    _parallel(hosts[1:], distribute)


def prepare_hosts(hosts: Sequence[str], command: Callable[[str, int], list[str]]) -> None:
    indexes = {host: index for index, host in enumerate(hosts)}

    def prepare(host: str) -> None:
        subprocess.run(command(host, indexes[host]), check=True)

    _parallel(hosts, prepare)


def remove_remote_manifests(hosts: Sequence[str], local_path: Path) -> None:
    target = shlex.quote(str(local_path))
    directory = shlex.quote(str(local_path.parent))

    def remove(host: str) -> None:
        try:
            subprocess.run(
                ["ssh", "-o", "BatchMode=yes", host, f"rm -f -- {target}; rmdir -- {directory}"],
                capture_output=True, text=True, timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    _parallel(hosts[1:], remove)
