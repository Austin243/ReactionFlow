"""Publish files and directory bundles durably on POSIX filesystems."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def file_digest(path: Path) -> str:
    """SHA-256 of a file's contents."""

    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


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


@contextmanager
def publishing(temporary: Path, final: Path) -> Iterator[None]:
    """Make a temporary sibling directory to fill, publish it, and remove it after an Exception."""

    temporary.mkdir()
    try:
        yield
        publish(temporary, final)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
