"""Build and synchronize the locked ROCm 10 + SGLang environment across MI300X hosts.

Run on the first host before launching multi-node GRPO training:

    python3 -m examples.experimental.generative_retriever.runtime.prepare_rocm_env \
        --hosts node-0,node-1 --slm-root /home/aiscuser/SLMGR_dev/SLMv3
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import torch

_REPO = Path(__file__).resolve().parents[4]
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


def _next_pow2(value: int) -> int:
    return 1 << max(0, (int(value) - 1).bit_length())


def derive_sglang_mem_fraction(
    sglang_server_concurrency: int,
    n_samples_per_prompt: int,
    rollout_num_gpus: int,
    *,
    max_prompt_tokens: int = 64,
    total_gpu_mem_gib: float | None = None,
) -> float:
    if total_gpu_mem_gib is not None:
        gpu_gib = float(total_gpu_mem_gib)
    elif torch.cuda.is_available():
        gpu_gib = float(torch.cuda.get_device_properties(0).total_memory) / float(2**30)
    else:
        gpu_gib = 191.45
    concurrent_queries_per_gpu = max(
        64,
        math.ceil(
            (sglang_server_concurrency / max(1, n_samples_per_prompt))
            / max(1, rollout_num_gpus)
        ),
    )
    kv_tokens = max(32768, 4 * concurrent_queries_per_gpu * max_prompt_tokens)
    kv_bytes_per_token = 36 * 2 * 8 * 128 * 2
    kv_gib = (kv_tokens * kv_bytes_per_token) / (2**30)
    model_and_workspace_gib = 5.5 + 4.0 + (0.01 * n_samples_per_prompt)
    fraction = (model_and_workspace_gib + kv_gib) / max(1.0, gpu_gib)
    return round(min(0.70, max(0.15, fraction)), 3)


def derive_reward_cache_sizes(
    rollout_batch_size: int,
    n_samples_per_prompt: int,
    num_workers: int,
    *,
    baseline_top_k: int = 64,
    mean_keywords_per_docid: float = 6.7,
) -> tuple[int, int, int]:
    queries_per_worker = max(1, math.ceil(rollout_batch_size / max(1, num_workers)))
    query_cache = max(16384, _next_pow2(4 * queries_per_worker))
    pairs_per_worker = int(
        math.ceil(2.0 * queries_per_worker * (n_samples_per_prompt + baseline_top_k) * mean_keywords_per_docid)
    )
    score_cache = min(8388608, max(131072, _next_pow2(pairs_per_worker)))
    keyword_cache = min(4194304, max(16384, _next_pow2(score_cache // 2)))
    return query_cache, keyword_cache, score_cache


def is_rocm() -> bool:
    version_mod = getattr(torch, "version", None)
    return getattr(version_mod, "hip", None) is not None


def rocm_runtime_env(venv: Path | None = None) -> dict[str, str]:
    root = venv if venv is not None else Path(sys.prefix)
    rocm_home = root / "lib/python3.12/site-packages/_rocm_sdk_devel"
    if not rocm_home.is_dir():
        raise FileNotFoundError(
            f"ROCm SDK devel directory is missing at {rocm_home}; run "
            "`python3 -m examples.experimental.generative_retriever.runtime.prepare_rocm_env`"
        )
    return {
        "ROCM_HOME": os.fspath(rocm_home),
        "ROCM_PATH": os.fspath(rocm_home),
        "PATH": f"{root / 'bin'}:{rocm_home / 'llvm/bin'}:{rocm_home / 'bin'}:{os.environ.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{rocm_home / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":"),
        "LIBRARY_PATH": f"{rocm_home / 'lib'}:{os.environ.get('LIBRARY_PATH', '')}".rstrip(":"),
        "CPATH": f"{rocm_home / 'include'}:{os.environ.get('CPATH', '')}".rstrip(":"),
        "CMAKE_PREFIX_PATH": f"{rocm_home}:{os.environ.get('CMAKE_PREFIX_PATH', '')}".rstrip(":"),
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "PYTORCH_ROCM_ARCH": "gfx942",
        "GPU_ARCHS": "gfx942",
        "GPU_ARCH": "gfx942",
        "AMDGPU_TARGET": "gfx942",
        "AITER_USE_SYSTEM_TRITON": "1",
        "MILES_HARDWARE_PLATFORM": "rocm",
        "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
        "NCCL_NVLS_ENABLE": "0",
    }


def locked_accelerator_env() -> dict[str, str]:
    if is_rocm():
        return rocm_runtime_env()
    cuda_home = Path(sys.prefix) / "cuda"
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.is_file():
        raise FileNotFoundError(
            "uv CUDA facade is missing; run "
            "`uv run python -m examples.experimental.generative_retriever.runtime.prepare_uv_cuda`"
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


def build_service_env(gpus: str, repo_root: Path = _REPO) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("WANDB_API_KEY", None)
    if is_rocm():
        env.update(rocm_runtime_env())
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["NVIDIA_VISIBLE_DEVICES"] = gpus
    env["PYTHONUNBUFFERED"] = "1"
    clean_py_path = ":".join(
        p for p in os.environ.get("PYTHONPATH", "").split(":")
        if p and "/opt/venv" not in p
    )
    env["PYTHONPATH"] = f"{repo_root}:{clean_py_path}".rstrip(":")
    return env


def _rocm_env(venv: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(rocm_runtime_env(venv))
    env["VIRTUAL_ENV"] = str(venv)
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
    pyvenv_cfg = venv / "pyvenv.cfg"
    if pyvenv_cfg.is_file():
        cfg_text = pyvenv_cfg.read_text(encoding="utf-8")
        if "include-system-site-packages = true" in cfg_text:
            pyvenv_cfg.write_text(
                cfg_text.replace(
                    "include-system-site-packages = true",
                    "include-system-site-packages = false",
                ),
                encoding="utf-8",
            )
    stale_pth = venv / "lib/python3.12/site-packages/host-runtime.pth"
    if stale_pth.is_file():
        stale_pth.unlink()

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
