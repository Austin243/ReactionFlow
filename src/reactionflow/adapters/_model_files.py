"""Small, explicit helpers for optional model packages and cached weights."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.request import urlopen
from uuid import uuid4

from .._durable import ensure_directory, publish, sync_directory
from .ase import _sha256


def require_package(name: str, expected: str, extra: str) -> None:
    try:
        installed = version(name)
    except PackageNotFoundError:
        installed = "not installed"
    if installed != expected:
        raise RuntimeError(
            f"{name} {expected} is required (found {installed}); "
            f"install 'reactionflow[{extra}]' or use 'reactionflow prepare campaign.json --install'"
        )


def model_cache(options: Mapping[str, object], backend: str) -> Path:
    raw = options.get("cache_dir", os.environ.get("REACTIONFLOW_MODEL_CACHE"))
    if raw is None:
        root = Path.home() / ".cache" / "reactionflow" / "models"
    else:
        if not isinstance(raw, str) or not raw:
            raise ValueError("cache_dir must be an absolute path string")
        root = Path(raw).expanduser()
        if not root.is_absolute():
            raise ValueError("cache_dir must be an absolute path")
    return root / backend


def _cached_file(root: Path, url: str) -> Path:
    try:
        manifest = json.loads((root / "source.json").read_text(encoding="utf-8"))
        path = root / "model"
        if manifest["url"] != url or _sha256(path) != manifest["sha256"]:
            raise ValueError("model source or checksum differs")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"cached model failed integrity verification: {root}") from error
    # A previous publish may have renamed the bundle before its parent sync failed.
    sync_directory(root.parent)
    return path


def cached_download(url: str, cache_dir: Path, *, download: bool) -> Path:
    """Publish one complete URL/checksum/weight bundle, serializing concurrent setup."""

    if not url.startswith("https://"):
        raise ValueError("pretrained model downloads require an HTTPS URL")
    key = hashlib.sha256(url.encode()).hexdigest()
    final = cache_dir / key
    if final.exists():
        return _cached_file(final, url)
    if not download:
        raise FileNotFoundError(
            "model is not cached; run 'reactionflow prepare campaign.json' "
            "or 'reactionflow run campaign.json --download' first"
        )
    ensure_directory(cache_dir)
    with (cache_dir / f".{key}.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if final.exists():
            return _cached_file(final, url)
        temporary = cache_dir / f".{key}-{uuid4().hex}.tmp"
        temporary.mkdir()
        try:
            path = temporary / "model"
            with urlopen(url, timeout=60) as response, path.open("wb") as handle:
                expected_size = response.headers.get("Content-Length")
                shutil.copyfileobj(response, handle)
                if expected_size is not None and handle.tell() != int(expected_size):
                    raise ValueError(
                        "model download size does not match the advertised Content-Length"
                    )
            if path.stat().st_size == 0:
                raise ValueError("model download returned an empty file")
            (temporary / "source.json").write_text(
                json.dumps({"url": url, "sha256": _sha256(path)}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            publish(temporary, final)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    return final / "model"
