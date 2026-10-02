from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from reactionflow.adapters import _torch, mace
from reactionflow.campaign import TrajectorySpec


class FakeMACECalculator(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

    def __init__(self, *, model_paths, device, default_dtype, head=None):
        super().__init__()
        self.model_paths = model_paths
        self.device = device
        self.default_dtype = default_dtype
        self.scale = float(Path(model_paths[0]).read_text())
        # MACE 0.3.16 warns and substitutes the final head on an unknown name.
        self.head = head if head in ("first", "second") else "Default"

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": float(0.5 * self.scale * np.sum(self.atoms.positions**2)),
            "forces": -self.scale * self.atoms.positions,
            "stress": np.zeros(6),
        }


@pytest.fixture
def backend(monkeypatch):
    package = ModuleType("mace")
    calculators = ModuleType("mace.calculators")
    calculators.MACECalculator = FakeMACECalculator
    registry = ModuleType("mace.calculators.foundations_models")
    registry.mace_mp_urls = {"medium-mpa-0": "https://example.test/mp.model"}
    registry.mace_off_urls = {"medium": "https://example.test/off.model"}
    monkeypatch.setitem(sys.modules, "mace", package)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators)
    monkeypatch.setitem(sys.modules, "mace.calculators.foundations_models", registry)
    checked = []
    monkeypatch.setattr(mace, "require_package", lambda *args: checked.append(args))
    return SimpleNamespace(calculators=calculators, checked=checked)


def _trajectory() -> TrajectorySpec:
    return TrajectorySpec.from_dict(
        {
            "id": "mace-test",
            "total_steps": 6,
            "timestep_fs": 0.25,
            "temperature_K": 300.0,
            "pressure_GPa": 0.1,
            "seed": 77,
        }
    )


def _atoms() -> Atoms:
    return Atoms("H2", positions=[[0.1, 0.2, 0.3], [1.1, 0.2, 0.3]], cell=[5] * 3, pbc=True)


def _checkpoint(tmp_path) -> Path:
    checkpoint = tmp_path / "model[1].model"
    checkpoint.write_text("0.5")
    return checkpoint


@pytest.mark.parametrize(
    ("family", "model", "url"),
    [
        ("mp", "medium-mpa-0", "https://example.test/mp.model"),
        ("off", "medium", "https://example.test/off.model"),
    ],
)
def test_prepare_uses_pinned_registry_without_constructing_calculator(
    tmp_path, monkeypatch, backend, family, model, url
) -> None:
    checkpoint = _checkpoint(tmp_path)
    calls = []

    def download(source, cache, *, download):
        calls.append((source, cache, download))
        return checkpoint

    def unexpected_calculator(**kwargs):
        pytest.fail("preparation must not load the model")

    monkeypatch.setattr(mace, "cached_download", download)
    monkeypatch.setattr(backend.calculators, "MACECalculator", unexpected_calculator)
    options = {"family": family, "model": model, "cache_dir": str(tmp_path)}
    assert mace.prepare(options) == {"checkpoint": checkpoint}
    assert mace.prepare(options, download=False) == {"checkpoint": checkpoint}
    assert [(source, allowed) for source, _, allowed in calls] == [(url, True), (url, False)]
    assert all(cache.is_absolute() for _, cache, _ in calls)
    assert backend.checked == [("mace-torch", "0.3.16", "mace"), ("e3nn", "0.4.4", "mace")] * 2


def test_local_checkpoint_never_uses_download_helper(tmp_path, monkeypatch, backend) -> None:
    checkpoint = _checkpoint(tmp_path)

    def unexpected_download(*args, **kwargs):
        pytest.fail("local checkpoint must not invoke a downloader")

    monkeypatch.setattr(mace, "cached_download", unexpected_download)
    assert mace.prepare({"checkpoint": str(checkpoint)}) == {"checkpoint": checkpoint}
    with pytest.raises(FileNotFoundError, match="checkpoint does not exist"):
        mace.prepare({"checkpoint": str(tmp_path / "missing")}, download=False)


def test_adapter_creation_cannot_download_a_missing_named_model(tmp_path, monkeypatch, backend):
    def missing_cache(url, cache, *, download):
        assert download is False
        raise FileNotFoundError("run prepare first")

    monkeypatch.setattr(mace, "cached_download", missing_cache)
    with pytest.raises(FileNotFoundError, match="prepare first"):
        mace.create_adapter(
            trajectory=_trajectory(),
            options={"model": "medium-mpa-0", "cache_dir": str(tmp_path)},
        )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({}, "exactly one"),
        ({"model": "medium", "checkpoint": "/tmp/model"}, "exactly one"),
        ({"model": None}, "explicit non-empty"),
        ({"model": "unknown"}, "choose from.*medium-mpa-0"),
        ({"model": "medium", "family": "wrong"}, "family"),
        ({"checkpoint": "relative.model"}, "absolute path"),
        ({"model": "medium-mpa-0", "cache_dir": "relative"}, "absolute path"),
        ({"model": "medium-mpa-0", "device": "auto"}, "device"),
        ({"model": "medium-mpa-0", "dtype": "float16"}, "dtype"),
        ({"model": "medium-mpa-0", "head": ""}, "head"),
        ({"model": "medium-mpa-0", "dispersion": True}, "unknown.*options"),
    ],
)
def test_bad_model_options_fail_before_download(options, message, monkeypatch, backend):
    def unexpected_download(*args, **kwargs):
        pytest.fail("invalid options must not download")

    monkeypatch.setattr(mace, "cached_download", unexpected_download)
    with pytest.raises(ValueError, match=message):
        mace.prepare(options)


def test_calculator_uses_one_literal_checkpoint_and_records_resolved_head(tmp_path, backend):
    checkpoint = _checkpoint(tmp_path)
    adapter = mace.create_adapter(
        trajectory=_trajectory(),
        options={"checkpoint": str(checkpoint), "device": "cpu", "head": "second"},
    )
    calculator = adapter._new_calculator()
    assert calculator.model_paths == [str(checkpoint)]
    assert calculator.device == "cpu"
    assert calculator.default_dtype == "float64"
    assert adapter._extra_metadata(calculator) == {"resolved_head": "second"}


def test_silently_substituted_model_head_is_rejected(tmp_path, backend):
    adapter = mace.create_adapter(
        trajectory=_trajectory(),
        options={"checkpoint": str(_checkpoint(tmp_path)), "device": "cpu", "head": "typo"},
    )
    with pytest.raises(ValueError, match="did not select requested head 'typo'"):
        adapter._new_calculator()


def test_restart_preserves_trajectory_and_rejects_changed_weights_or_head(
    tmp_path, monkeypatch, backend
):
    monkeypatch.setattr(_torch, "_load_torch", lambda device: SimpleNamespace())
    monkeypatch.setattr(_torch, "_torch_environment", lambda torch, device: {"device": device})
    monkeypatch.setattr(_torch, "_preserve_rng", lambda torch: nullcontext())
    monkeypatch.setattr(_torch, "version", lambda name: "fake-version")
    checkpoint = _checkpoint(tmp_path)
    options = {"checkpoint": str(checkpoint), "device": "cpu", "head": "first"}
    adapter = mace.create_adapter(trajectory=_trajectory(), options=options)
    with adapter.start(_atoms()) as runtime:
        runtime.run(6)
        full = runtime.snapshot()
    with adapter.start(_atoms()) as runtime:
        runtime.run(2)
        split = runtime.snapshot()
    fresh = mace.create_adapter(trajectory=_trajectory(), options=options)
    with fresh.restore(split) as runtime:
        runtime.run(4)
        resumed = runtime.snapshot()
    assert resumed.calculator == full.calculator
    assert resumed.dynamics.metadata == full.dynamics.metadata
    for key in full.atoms.arrays:
        np.testing.assert_array_equal(resumed.atoms.arrays[key], full.atoms.arrays[key])
    np.testing.assert_array_equal(resumed.atoms.cell.array, full.atoms.cell.array)
    for key in full.dynamics.arrays:
        np.testing.assert_array_equal(resumed.dynamics.arrays[key], full.dynamics.arrays[key])
    changed = mace.create_adapter(trajectory=_trajectory(), options={**options, "head": "second"})
    with pytest.raises(ValueError, match="environment differs"), changed.restore(split):
        pass
    checkpoint.write_text("0.75")
    changed = mace.create_adapter(trajectory=_trajectory(), options=options)
    with pytest.raises(ValueError, match="environment differs"), changed.restore(split):
        pass
