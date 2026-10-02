"""Publish files and directory bundles durably on POSIX filesystems."""

from __future__ import annotations

import os
from pathlib import Path


def sync_directory(path: Path) -> None:
    """Persist directory entries, including completed renames."""

    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def ensure_directory(path: Path) -> None:
    """Create a permanent directory and persist each newly created ancestor."""

    if not path.is_dir():
        ensure_directory(path.parent)
        path.mkdir(exist_ok=True)
    sync_directory(path.parent)


def flush_to_disk(path: Path) -> None:
    """Flush files and then directory entries, working from children to parents."""

    if path.is_dir():
        for child in path.iterdir():
            flush_to_disk(child)
        sync_directory(path)
    else:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())


def publish(temporary: Path, final: Path) -> None:
    """Flush and rename a temporary sibling, then persist its final name."""

    flush_to_disk(temporary)
    os.replace(temporary, final)
    sync_directory(final.parent)
