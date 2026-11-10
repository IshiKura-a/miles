"""Lexical paths for GR artifacts and their known symlinks."""

from __future__ import annotations

import os
from pathlib import Path


def absolute_path(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.path.expanduser(path)))


def symlink_target(path: Path) -> Path:
    return absolute_path(path.parent / os.readlink(path))
