"""Lexical paths and buffered stream-copy utilities for GR artifacts."""

from __future__ import annotations

import os
from pathlib import Path
import shutil

COPY_BUFFER_BYTES = 64 * 1024 * 1024


def absolute_path(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.path.expanduser(path)))


def symlink_target(path: Path) -> Path:
    return absolute_path(path.parent / os.readlink(path))


def stream_copy_file(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
    """Copy a file via user-space buffered read/write without kernel sendfile."""
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with src_path.open("rb") as reader, dst_path.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=COPY_BUFFER_BYTES)


def atomic_stream_copy_file(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
    """Stream-copy to a sibling ``.tmp`` file and atomically replace ``dst``."""
    dst_path = Path(dst)
    tmp_path = dst_path.with_name(f".{dst_path.name}.tmp.{os.getpid()}")
    try:
        stream_copy_file(src, tmp_path)
        tmp_path.replace(dst_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
