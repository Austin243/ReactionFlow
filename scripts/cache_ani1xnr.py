#!/usr/bin/env python3
"""Download the pinned ANI-1xnr state dict and verify its content digest."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

from reactionflow.adapters.ani1xnr import (
    MODEL_FILENAME,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    MODEL_SHA256,
)


def _sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=os.environ.get("TORCHANI_DATA_DIR"),
        help="TorchANI data root (defaults to TORCHANI_DATA_DIR)",
    )
    arguments = parser.parse_args()
    if arguments.data_dir is None:
        parser.error("use --data-dir or set TORCHANI_DATA_DIR")
    data_dir = arguments.data_dir.expanduser().resolve()
    state_dicts = data_dir / "StateDicts"
    state_dicts.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=MODEL_REPOSITORY,
            filename=MODEL_FILENAME,
            revision=MODEL_REVISION,
            local_dir=state_dicts,
        )
    ).resolve()
    digest = _sha256(downloaded)
    if digest != MODEL_SHA256:
        raise RuntimeError(
            f"ANI-1xnr weights failed SHA-256 verification: {digest} != {MODEL_SHA256}"
        )
    print(f"ANI-1xnr weights verified: {downloaded} ({digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
