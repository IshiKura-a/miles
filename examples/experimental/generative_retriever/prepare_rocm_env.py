"""Build and synchronize the locked ROCm 10 + SGLang environment across MI300X hosts.

Run on the first host before launching multi-node GRPO training:

    python3 -m examples.experimental.generative_retriever.prepare_rocm_env \
        --hosts node-0,node-1 --slm-root /home/aiscuser/SLMGR_dev/SLMv3
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess

_REPO = Path(__file__).resolve().parents[3]
_DIR = Path(__file__).resolve().parent
_FREEZE = _DIR / "requirements-rocm.freeze.txt"
_SLMV3_PATCH = _DIR / "patches/slmv3-transformers-cache.patch"
_SGLANG_PATCH = _DIR / "patches/sglang-embedding-abort.patch"

_ROCM_INDEX_URL = "https://stable.repo.amd.com/rocm/whl-next/"
_ROCM_PACKAGES = (
    "rocm==10.0.0",
    "rocm-bootstrap==0.3.0",
    "rocm-sdk-core==10.0.0",
    "rocm-sdk-libraries==10.0.0",
    "rocm-sdk-devel==10.0.0",
    "rocm-sdk-device-gfx942==10.0.0",
    "torch==2.11.0+rocm10.0.0",
    "torchvision==0.26.0+rocm10.0.0",
    "torchaudio==2.11.0+rocm10.0.0",
    "amd-torch-device-gfx942==2.11.0+rocm10.0.0",
    "amd-torchvision-device-gfx942==0.26.0+rocm10.0.0",
    "triton==3.8.0+git4cff872c.rocm10.0.0",
)
_GIT_WHEELS = (
    "mbridge @ git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c",
    "megatron-bridge @ git+https://github.com/radixark/Megatron-Bridge.git@8cd3466d14d2337c8492827b3712482c2b3e4866",
    "torch-memory-saver @ git+https://github.com/fzyzcjy/torch_memory_saver.git@06caa534822c5b980b61733a5d79ac731f9a5f9c",
)
_SKIP_PREFIXES = (
    "-e ",
    "sglang-kernel @",
    "torch-memory-saver @",
    "megatron-bridge @",
    "mbridge @",
    "rocm==",
    "rocm-bootstrap==",
    "rocm-sdk-",
    "torch==",
    "torchaudio==",
    "torchvision==",
    "amd-torch-",
    "triton==",
)
_MEGATRON_COMMIT = "fd15ee20a4f03b03680529baf4b8f7eeed64df1d"
_SGLANG_COMMIT = "7f79d11d948aa37b49821475e075369bb0e801f3"
_AITER_COMMIT = "acf8fdf9307431ece8ee275971c41cb3d1a7020b"
_AITER_CHERRYPICKS = (
    "7b481fbcaf834ce98b66004ea329c2ca2bba87d7",
    "24a62b1c122f23645a19b9d8b0abd4750c59359b",
)


def _run(argv: list[str], *, cwd: Path = _REPO, env: dict[str, str] | None = None) -> None:
    subprocess.run(argv, cwd=cwd, env=env, check=True)


def _rocm_env(venv: Path) -> dict[str, str]:
    rocm_home = venv / "lib/python3.12/site-packages/_rocm_sdk_devel"
    env = os.environ.copy()
    env.update({
        "VIRTUAL_ENV": str(venv),
        "ROCM_HOME": str(rocm_home),
        "ROCM_PATH": str(rocm_home),
        "PATH": f"{venv / 'bin'}:{rocm_home / 'llvm/bin'}:{rocm_home / 'bin'}:{Path.home() / '.local/bin'}:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{rocm_home / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":"),
        "LIBRARY_PATH": f"{rocm_home / 'lib'}:{env.get('LIBRARY_PATH', '')}".rstrip(":"),
        "CPATH": f"{rocm_home / 'include'}:{env.get('CPATH', '')}".rstrip(":"),
        "CMAKE_PREFIX_PATH": f"{rocm_home}:{env.get('CMAKE_PREFIX_PATH', '')}".rstrip(":"),
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "PYTORCH_ROCM_ARCH": "gfx942",
        "GPU_ARCHS": "gfx942",
        "GPU_ARCH": "gfx942",
        "AMDGPU_TARGET": "gfx942",
        "AITER_USE_SYSTEM_TRITON": "1",
    })
    return env


def _write_activate_script(venv: Path) -> Path:
    rocm_home = venv / "lib/python3.12/site-packages/_rocm_sdk_devel"
    script = venv / "activate-rocm.sh"
    script.write_text(
        f'source {shlex.quote(str(venv / "bin/activate"))}\n'
        f'export ROCM_HOME={shlex.quote(str(rocm_home))}\n'
        'export ROCM_PATH="$ROCM_HOME"\n'
        f'export PATH="{venv / "bin"}:$ROCM_HOME/llvm/bin:$ROCM_HOME/bin:{Path.home() / ".local/bin"}:$PATH"\n'
        'export LD_LIBRARY_PATH="$ROCM_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n'
        'export LIBRARY_PATH="$ROCM_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"\n'
        'export CPATH="$ROCM_HOME/include${CPATH:+:$CPATH}"\n'
        'export CMAKE_PREFIX_PATH="$ROCM_HOME${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"\n'
        'export HSA_NO_SCRATCH_RECLAIM="1"\n'
        'export PYTORCH_ROCM_ARCH="gfx942"\n'
        'export GPU_ARCHS="gfx942"\n'
        'export GPU_ARCH="gfx942"\n'
        'export AMDGPU_TARGET="gfx942"\n'
        'export AITER_USE_SYSTEM_TRITON="1"\n'
        'export NCCL_SOCKET_IFNAME="eth0"\n'
        'export NCCL_IB_HCA=""\n'
        'export NCCL_IB_TIMEOUT="22"\n'
        'export NCCL_IB_PCI_RELAXED_ORDERING="1"\n'
        'export NCCL_NET_GDR_LEVEL="5"\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _apply_patch_idempotent(repo: Path, patch: Path) -> None:
    reverse = subprocess.run(
        ["git", "-C", str(repo), "apply", "--reverse", "--check", str(patch)],
        capture_output=True, text=True,
    )
    if reverse.returncode == 0:
        return
    _run(["git", "-C", str(repo), "apply", "--check", str(patch)])
    _run(["git", "-C", str(repo), "apply", str(patch)])


def _patch_aiter_tree(aiter_dir: Path) -> None:
    prefill = aiter_dir / "aiter/ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4_prefill.py"
    text = prefill.read_text(encoding="utf-8")
    text = text.replace("from functools import lru_cache", "from functools import cache")
    text = text.replace("@lru_cache(maxsize=32)", "@cache")
    prefill.write_text(text, encoding="utf-8")

    torch_utils = aiter_dir / "csrc/cpp_itfs/torch_utils.py"
    utils_text = torch_utils.read_text(encoding="utf-8")
    old = (
        "        elif isinstance(arg, torch.cuda.Stream):\n"
        "            c_args.append(ctypes.cast(arg.cuda_stream, ctypes.c_void_p))\n"
    )
    new = (
        "        elif isinstance(arg, torch.Stream):\n"
        '            handle = getattr(arg, "cuda_stream", None)\n'
        "            if handle is None:\n"
        "                handle = torch.cuda.Stream(\n"
        "                    stream_id=arg.stream_id,\n"
        "                    device_index=arg.device_index,\n"
        "                    device_type=arg.device_type,\n"
        "                ).cuda_stream\n"
        "            c_args.append(ctypes.cast(handle, ctypes.c_void_p))\n"
    )
    if old in utils_text:
        torch_utils.write_text(utils_text.replace(old, new), encoding="utf-8")


def _prepare_sources(venv: Path, env: dict[str, str]) -> None:
    src = venv / "src"
    src.mkdir(parents=True, exist_ok=True)
    python = str(venv / "bin/python")
    uv = str(shutil.which("uv") or (Path.home() / ".local/bin/uv"))

    megatron = src / "Megatron-LM"
    if not megatron.exists():
        _run(["git", "clone", "--no-checkout", "https://github.com/radixark/Megatron-LM.git", str(megatron)])
        _run(["git", "-C", str(megatron), "checkout", _MEGATRON_COMMIT])

    sglang = src / "sglang"
    if not sglang.exists():
        _run(["git", "clone", "--no-checkout", "https://github.com/sgl-project/sglang.git", str(sglang)])
        _run(["git", "-C", str(sglang), "checkout", _SGLANG_COMMIT])
    _apply_patch_idempotent(sglang, _SGLANG_PATCH)

    aiter = src / "aiter"
    if not aiter.exists():
        _run(["git", "clone", "https://github.com/ROCm/aiter.git", str(aiter)])
        _run(["git", "-C", str(aiter), "checkout", "-f", _AITER_COMMIT])
        for commit in _AITER_CHERRYPICKS:
            _run(["git", "-C", str(aiter), "cherry-pick", "--no-commit", commit])
        _patch_aiter_tree(aiter)
        _run(["git", "-C", str(aiter), "submodule", "update", "--init", "--recursive"])

    _run(
        [
            str(venv / "bin/pip"), "install", "--no-deps", "--no-build-isolation",
            "--config-settings", "editable_mode=compat", "-e", str(aiter),
        ],
        env=env,
    )

    aot_dir = sglang / "python/sglang/kernels/aot"
    wheels = list((aot_dir / "dist").glob("sglang_kernel-*.whl"))
    if not wheels:
        pyproject = aot_dir / "pyproject.toml"
        backup = aot_dir / "pyproject.toml.bak"
        shutil.copy2(pyproject, backup)
        try:
            shutil.copy2(aot_dir / "pyproject_rocm.toml", pyproject)
            _run([python, "setup_rocm.py", "bdist_wheel"], cwd=aot_dir, env={**env, "MAX_JOBS": "32"})
        finally:
            shutil.move(backup, pyproject)
        wheels = list((aot_dir / "dist").glob("sglang_kernel-*.whl"))
    _run([uv, "pip", "install", "--python", python, "--no-deps", str(wheels[0])], env=env)

    sgl_py = sglang / "python"
    sgl_pyproject = sgl_py / "pyproject.toml"
    sgl_backup = sgl_py / "pyproject.toml.bak"
    shutil.copy2(sgl_pyproject, sgl_backup)
    try:
        shutil.copy2(sgl_py / "pyproject_other.toml", sgl_pyproject)
        _run(
            [uv, "pip", "install", "--python", python, "--no-deps", "--no-build-isolation", "-e", str(sgl_py)],
            env={**env, "SGLANG_BUILD_RUST_EXTS": "none"},
        )
    finally:
        shutil.move(sgl_backup, sgl_pyproject)

    _run([uv, "pip", "install", "--python", python, "--no-deps", "--no-build-isolation", "-e", str(megatron)], env=env)
    _run([uv, "pip", "install", "--python", python, "--no-deps", "--no-build-isolation", "-e", str(_REPO)], env=env)


def prepare(*, hosts: list[str], slm_root: Path) -> None:
    slmgr_repo = slm_root.resolve().parent
    _apply_patch_idempotent(slmgr_repo, _SLMV3_PATCH)

    venv = _REPO / ".venv"
    uv = str(shutil.which("uv") or (Path.home() / ".local/bin/uv"))
    python = str(venv / "bin/python")
    if not venv.exists():
        _run([uv, "venv", str(venv), "--python", "/usr/bin/python3.12"])

    _run([uv, "pip", "install", "--python", python, "--index-url", _ROCM_INDEX_URL, *_ROCM_PACKAGES])
    _run([str(venv / "bin/rocm-sdk"), "init"])

    site_packages = venv / "lib/python3.12/site-packages"
    core_libs = sorted((site_packages / "_rocm_sdk_core/lib").glob("libamd_smi.so.*"))
    devel_lib = site_packages / "_rocm_sdk_devel/lib/libamd_smi.so"
    if core_libs and devel_lib.exists() and not devel_lib.is_symlink():
        devel_lib.unlink()
        devel_lib.symlink_to(core_libs[0])

    rocm_devel = site_packages / "_rocm_sdk_devel"
    _run(["sudo", "mkdir", "-p", "/usr/lib64"])
    _run(["sudo", "ln", "-sf", "/lib/x86_64-linux-gnu/libc.so", "/usr/lib64/libc.so"])
    _run(["sudo", "ln", "-sfn", str(rocm_devel), "/opt/rocm"])
    activate = _write_activate_script(venv)
    env = _rocm_env(venv)

    pinned = [
        line.strip()
        for line in _FREEZE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(_SKIP_PREFIXES)
    ]
    _run([uv, "pip", "install", "--python", python, "--no-deps", *pinned], env=env)
    _run([uv, "pip", "install", "--python", python, "--no-deps", "--no-build-isolation", *_GIT_WHEELS], env=env)
    _prepare_sources(venv, env)
    _run(
        [python, "-c", "import torch, aiter, sglang, sgl_kernel, miles, gin; assert torch.cuda.device_count() == 8"],
        env=env,
    )

    for host in hosts[1:]:
        _run([
            "ssh", "-o", "BatchMode=yes", host,
            f"git -C {shlex.quote(str(slmgr_repo))} apply --reverse --check {shlex.quote(str(_SLMV3_PATCH))} "
            f"|| git -C {shlex.quote(str(slmgr_repo))} apply {shlex.quote(str(_SLMV3_PATCH))}",
        ])
        _run(["rsync", "-a", f"{venv}/", f"{host}:{venv}/"])
        _run([
            "ssh", "-o", "BatchMode=yes", host,
            f"sudo mkdir -p /usr/lib64 && sudo ln -sf /lib/x86_64-linux-gnu/libc.so /usr/lib64/libc.so "
            f"&& sudo ln -sfn {shlex.quote(str(rocm_devel))} /opt/rocm "
            f"&& source {shlex.quote(str(activate))} "
            f"&& python -c 'import torch, aiter, sglang, sgl_kernel, miles, gin; assert torch.cuda.device_count() == 8'",
        ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", default="node-0,node-1")
    parser.add_argument("--slm-root", default="/home/aiscuser/SLMGR_dev/SLMv3")
    args = parser.parse_args()
    hosts = [host.strip() for host in args.hosts.split(",") if host.strip()]
    prepare(hosts=hosts, slm_root=Path(args.slm_root))


if __name__ == "__main__":
    main()
