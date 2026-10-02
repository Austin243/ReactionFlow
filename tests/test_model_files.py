from __future__ import annotations

from io import BytesIO
from typing import ClassVar

import pytest

from reactionflow.adapters import _model_files as models


def _response(content):
    response = BytesIO(content)
    response.headers = {"Content-Length": str(len(content))}
    return response


def test_download_is_cached_and_offline_resolution_verifies_content(tmp_path, monkeypatch):
    requests = []

    def fetch(url, *, timeout):
        requests.append((url, timeout))
        return _response(b"model weights")

    monkeypatch.setattr(models, "urlopen", fetch)
    url = "https://models.example/model.pt"
    path = models.cached_download(url, tmp_path, download=True)
    assert path.read_bytes() == b"model weights"
    assert models.cached_download(url, tmp_path, download=False) == path
    assert models.cached_download(url, tmp_path, download=True) == path
    assert requests == [(url, 60)]

    path.write_bytes(b"replaced weights")
    with pytest.raises(ValueError, match="integrity"):
        models.cached_download(url, tmp_path, download=True)
    assert len(requests) == 1


def test_offline_cache_miss_does_not_download_or_create_directories(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("offline mode used the network")

    monkeypatch.setattr(models, "urlopen", unexpected)
    cache = tmp_path / "absent"
    with pytest.raises(FileNotFoundError, match="prepare"):
        models.cached_download("https://models.example/model.pt", cache, download=False)
    assert not cache.exists()


def test_failed_download_does_not_publish_a_partial_model(tmp_path, monkeypatch):
    class Interrupted(BytesIO):
        headers: ClassVar[dict[str, str]] = {}

        def read(self, *args):
            raise OSError("download interrupted")

    monkeypatch.setattr(models, "urlopen", lambda *args, **kwargs: Interrupted())
    url = "https://models.example/model.pt"
    with pytest.raises(OSError, match="interrupted"):
        models.cached_download(url, tmp_path, download=True)
    assert not any(path.is_dir() for path in tmp_path.iterdir())

    monkeypatch.setattr(models, "urlopen", lambda *args, **kwargs: _response(b"complete"))
    assert models.cached_download(url, tmp_path, download=True).read_bytes() == b"complete"


def test_concurrent_preparation_downloads_one_complete_copy(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    started = Barrier(2)
    requests = []

    def fetch(*args, **kwargs):
        requests.append(args)
        return _response(b"complete model")

    def prepare(_index):
        started.wait(timeout=5)
        return models.cached_download("https://models.example/model.pt", tmp_path, download=True)

    monkeypatch.setattr(models, "urlopen", fetch)
    with ThreadPoolExecutor(max_workers=2) as workers:
        paths = list(workers.map(prepare, range(2)))
    assert paths[0] == paths[1]
    assert paths[0].read_bytes() == b"complete model"
    assert len(requests) == 1


def test_clean_eof_before_content_length_is_not_cached(tmp_path, monkeypatch):
    from http.client import HTTPResponse

    class Socket:
        def makefile(self, *args, **kwargs):
            return BytesIO(b"HTTP/1.1 200 OK\r\nContent-Length: 10000\r\n\r\ntruncated")

    response = HTTPResponse(Socket())
    response.begin()
    monkeypatch.setattr(models, "urlopen", lambda *args, **kwargs: response)
    url = "https://models.example/model.pt"
    with pytest.raises(ValueError, match="Content-Length"):
        models.cached_download(url, tmp_path, download=True)
    with pytest.raises(FileNotFoundError):
        models.cached_download(url, tmp_path, download=False)
    assert not any(path.is_dir() for path in tmp_path.iterdir())
    monkeypatch.setattr(models, "urlopen", lambda *args, **kwargs: _response(b"complete"))
    assert models.cached_download(url, tmp_path, download=True).read_bytes() == b"complete"


def test_retry_syncs_cache_after_post_rename_parent_sync_failure(tmp_path, monkeypatch):
    from reactionflow import _durable

    url = "https://models.example/model.pt"
    monkeypatch.setattr(models, "urlopen", lambda *args, **kwargs: _response(b"complete"))
    original_sync = _durable.sync_directory
    failed = False

    def fail_after_rename(path):
        nonlocal failed
        if path == tmp_path and not failed:
            failed = True
            raise OSError("parent sync failed")
        original_sync(path)

    monkeypatch.setattr(_durable, "sync_directory", fail_after_rename)
    with pytest.raises(OSError, match="parent sync failed"):
        models.cached_download(url, tmp_path, download=True)
    published = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(published) == 1
    assert (published[0] / "model").read_bytes() == b"complete"
    synced = []

    def record_sync(path):
        synced.append(path)
        original_sync(path)

    monkeypatch.setattr(models, "sync_directory", record_sync)
    assert models.cached_download(url, tmp_path, download=False) == published[0] / "model"
    assert synced == [tmp_path]


def test_cache_path_is_explicit_and_backend_scoped(tmp_path, monkeypatch):
    monkeypatch.setenv("REACTIONFLOW_MODEL_CACHE", str(tmp_path))
    assert models.model_cache({}, "uma") == tmp_path / "uma"
    assert models.model_cache({"cache_dir": str(tmp_path / "other")}, "mace") == (
        tmp_path / "other/mace"
    )
    with pytest.raises(ValueError, match="absolute"):
        models.model_cache({"cache_dir": "relative"}, "mace")
