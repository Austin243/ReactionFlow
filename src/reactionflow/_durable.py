"""Flush files to disk before the rename that publishes them."""

from __future__ import annotations

import os
from pathlib import Path


def flush_to_disk(path: Path) -> None:
    """Force a file, or every file under a directory, to stable storage.

    Published artifacts are flushed first so that nothing durable can name data that a node
    failure lost.
    """

    for item in (path,) if path.is_file() else path.rglob("*"):
        if item.is_file():
            with item.open("rb") as handle:
                os.fsync(handle.fileno())
