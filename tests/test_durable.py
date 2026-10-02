from __future__ import annotations

import json
import os

import pytest

import reactionflow._durable as durable
from reactionflow import ReactionRun


def test_nested_bundle_is_flushed_before_its_name_is_published(tmp_path, monkeypatch) -> None:
    temporary = tmp_path / ".checkpoint.tmp"
    nested = temporary / "runtime"
    nested.mkdir(parents=True)
    payload = nested / "state.json"
    payload.write_text("complete")
    paths = {path.stat().st_ino: path.name for path in (payload, nested, temporary, tmp_path)}
    events = []
    fsync, replace = os.fsync, os.replace

    def record_sync(descriptor):
        events.append(paths[os.fstat(descriptor).st_ino])
        fsync(descriptor)

    def record_replace(source, destination):
        events.append("rename")
        replace(source, destination)

    monkeypatch.setattr(durable.os, "fsync", record_sync)
    monkeypatch.setattr(durable.os, "replace", record_replace)
    final = tmp_path / "checkpoint"
    durable.publish(temporary, final)

    assert events == ["state.json", "runtime", ".checkpoint.tmp", "rename", tmp_path.name]
    assert (final / "runtime/state.json").read_text() == "complete"
    assert not temporary.exists()


def test_new_directory_ancestors_are_persisted(tmp_path, monkeypatch) -> None:
    parents = []
    sync_directory = durable.sync_directory

    def record_sync(path):
        parents.append(path)
        sync_directory(path)

    monkeypatch.setattr(durable, "sync_directory", record_sync)
    destination = tmp_path / "campaign" / "run"
    durable.ensure_directory(destination)

    assert destination.is_dir()
    assert parents[-2:] == [tmp_path, tmp_path / "campaign"]


@pytest.mark.parametrize("fail_directory_sync", [False, True])
def test_state_publication_failure_preserves_previous_checkpoint(
    tmp_path, monkeypatch, fail_directory_sync
) -> None:
    run = ReactionRun.create(tmp_path)
    previous = run.runtime_checkpoints / "previous"
    previous.mkdir()
    (previous / "marker").write_text("recoverable")
    run._active_checkpoint = previous
    run._write_state()

    current = run.runtime_checkpoints / "current"
    current.mkdir()
    run._active_checkpoint = current
    fsync, sync_directory = os.fsync, durable.sync_directory

    def fail_file_sync(descriptor):
        temporary = run.state_path.with_suffix(".json.tmp")
        if os.fstat(descriptor).st_ino == temporary.stat().st_ino:
            raise OSError("injected file sync failure")
        fsync(descriptor)

    def fail_parent_sync(path):
        if path == run.root:
            raise OSError("injected directory sync failure")
        sync_directory(path)

    with monkeypatch.context() as patch:
        if fail_directory_sync:
            patch.setattr(durable, "sync_directory", fail_parent_sync)
        else:
            patch.setattr(durable.os, "fsync", fail_file_sync)
        with pytest.raises(OSError, match=r"injected .* sync failure"):
            run._write_state()

    # Either name may survive a failed publication. Neither checkpoint may be pruned yet.
    assert (previous / "marker").read_text() == "recoverable"
    assert current.is_dir()
    state = json.loads(run.state_path.read_text())
    expected = current if fail_directory_sync else previous
    assert run.root / state["active_checkpoint"] == expected

    run._write_state()
    assert not previous.exists()
    assert current.is_dir()
