"""Prepare a CUDA_HOME facade for the CUDA compiler installed by uv."""

from __future__ import annotations

import sys
from pathlib import Path


def prepare() -> Path:
    environment = Path(sys.prefix)
    package_root = (
        environment
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
        / "cu13"
    )
    destination = environment / "cuda"
    destination.mkdir(exist_ok=True)
    for name in ("bin", "include", "nvvm"):
        link = destination / name
        if not link.exists():
            link.symlink_to(package_root / name, target_is_directory=True)
    lib64 = destination / "lib64"
    lib64.mkdir(exist_ok=True)
    for library in (package_root / "lib").iterdir():
        link = lib64 / library.name
        if not link.exists():
            link.symlink_to(library)
    cudart = lib64 / "libcudart.so"
    if cudart.is_symlink() and not cudart.exists():
        cudart.unlink()
    if not cudart.exists():
        cudart.symlink_to(package_root / "lib" / "libcudart.so.13")
    return destination


def main() -> None:
    print(prepare())


if __name__ == "__main__":
    main()
