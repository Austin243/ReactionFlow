from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes

from reactionflow.adapters import _torch, sevennet
from reactionflow.campaign import TrajectorySpec


@pytest.fixture
def backend(tmp_path, monkeypatch):
    torch = ModuleType("torch")
    torch.float32 = "float32"
    state = {"dtype": "float64"}
    torch.set_default_dtype = lambda dtype: state.update(dtype=dtype)
    calls = []

    class FakeCalculator(Calculator):
        implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

        def __init__(self, **kwargs):
            assert state["dtype"] == "float32"
            super().__init__()
            calls.append(kwargs)
            self.modal = kwargs["modal"] if kwargs["modal"] != "ignored" else None
            self.scale = float(Path(kwargs["model"]).read_text())

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {
                "energy": float(self.scale * np.sum(self.atoms.positions**2) / 2),
                "forces": -self.scale * self.atoms.positions,
                "stress": np.zeros(6),
            }

    package = ModuleType("sevenn")
    calculator = ModuleType("sevenn.calculator")
    calculator.SevenNetCalculator = FakeCalculator
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "sevenn", package)
    monkeypatch.setitem(sys.modules, "sevenn.calculator", calculator)
    monkeypatch.setattr(sevennet, "require_package", lambda *args: None)
    monkeypatch.setattr(
        sevennet,
        "distribution",
        lambda name: SimpleNamespace(locate_file=lambda path: tmp_path / path),
    )
    for path in sevennet._BUNDLED.values():
        file = tmp_path / "sevenn/pretrained_potentials" / path
        file.parent.mkdir(parents=True)
        file.write_text("0.5")
    for key in (
        "SEVENNET_ENABLE_CUEQ",
        "SEVENNET_ENABLE_FLASH",
        "SEVENNET_ENABLE_OEQ",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    ):
        monkeypatch.delenv(key, raising=False)
    return SimpleNamespace(calls=calls, torch=torch)


def _trajectory():
    return TrajectorySpec(
        id="sevennet", total_steps=3, timestep_fs=0.1, temperature_K=50, pressure_GPa=0.1, seed=77
    )


def _local(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_text("0.5")
    return {"checkpoint": str(checkpoint), "device": "cpu"}


def test_catalog_has_released_names_and_case_sensitive_modals_without_optional_imports(monkeypatch):
    def unexpected(*args):
        pytest.fail("catalog must not require SevenNet")

    monkeypatch.setattr(sevennet, "require_package", unexpected)
    catalog = json.loads(json.dumps(sevennet.catalog()))
    assert catalog["backend"] == "sevennet"
    models = {entry["name"]: entry for entry in catalog["models"]}
    assert len(models) == 9
    assert models["7net-mf-0"]["modals"] == ["PBE", "R2SCAN"]
    assert models["7net-mf-ompa"]["modals"] == ["mpa", "omat24"]
    assert len(models["7net-omni"]["modals"]) == 13
    assert not any("nano" in name for name in models)


@pytest.mark.parametrize("model", list(sevennet._BUNDLED))
def test_prepare_bundled_weights_never_downloads_or_loads_model(
    tmp_path, monkeypatch, backend, model
):
    def unexpected(*args, **kwargs):
        pytest.fail("bundled weights must not be downloaded")

    monkeypatch.setattr(sevennet, "cached_download", unexpected)
    options = {"model": model}
    if model == "7net-mf-0":
        options["modal"] = "PBE"
    result = sevennet.prepare(options, download=False)
    assert result["checkpoint"].is_file()
    assert result["checkpoint"].is_absolute()
    assert backend.calls == []


@pytest.mark.parametrize("model", list(sevennet._DOWNLOADS))
def test_prepare_download_control_and_release_urls(tmp_path, monkeypatch, backend, model):
    calls = []
    file = Path(_local(tmp_path)["checkpoint"])

    def download(url, cache, *, download):
        calls.append((url, cache, download))
        return file

    monkeypatch.setattr(sevennet, "cached_download", download)
    options = {"model": model, "cache_dir": str(tmp_path)}
    if model in sevennet._MODALS:
        options["modal"] = "mpa"
    for allowed in (False, True):
        assert sevennet.prepare(options, download=allowed) == {"checkpoint": file}
    assert [call[2] for call in calls] == [False, True]
    assert all(
        call[0].startswith("https://github.com/MDIL-SNU/SevenNet/releases/download/v0.")
        for call in calls
    )
    assert all(call[1] == tmp_path / "sevennet" for call in calls)
    assert backend.calls == []


def test_constructor_is_offline_and_uses_fixed_float32_e3nn(tmp_path, monkeypatch, backend):
    def unexpected(*args, **kwargs):
        pytest.fail("local inference must not download")

    monkeypatch.setattr(sevennet, "cached_download", unexpected)
    options = {**_local(tmp_path), "modal": "mpa"}
    adapter = sevennet.create_adapter(trajectory=_trajectory(), options=options)
    calculator = adapter._new_calculator()
    assert backend.calls == [
        {
            "model": options["checkpoint"],
            "file_type": "checkpoint",
            "device": "cpu",
            "modal": "mpa",
            "enable_cueq": False,
            "enable_flash": False,
            "enable_oeq": False,
            "compute_atomic_virial": False,
        }
    ]
    assert adapter._extra_metadata(calculator)["resolved_modal"] == "mpa"


def test_missing_cached_named_weights_cannot_trigger_runtime_download(
    tmp_path, monkeypatch, backend
):
    def missing(url, cache, *, download):
        assert download is False
        raise FileNotFoundError("prepare first")

    monkeypatch.setattr(sevennet, "cached_download", missing)
    with pytest.raises(FileNotFoundError, match="prepare first"):
        sevennet.create_adapter(
            trajectory=_trajectory(), options={"model": "7net-omni", "modal": "mpa"}
        )
    assert backend.calls == []


@pytest.mark.parametrize(
    "options,message",
    [
        ({}, "exactly one"),
        ({"model": "7net-0", "checkpoint": "/tmp/weights"}, "exactly one"),
        ({"model": "7net-nano-5.5"}, "unknown SevenNet model"),
        ({"model": []}, "unknown SevenNet model"),
        ({"model": "7net-omni"}, "requires explicit modal"),
        ({"model": "7net-mf-0", "modal": "pbe"}, "requires explicit modal"),
        ({"model": "7net-0", "modal": "mpa"}, "has no modal"),
        ({"checkpoint": "/tmp/weights", "modal": []}, "nonempty string"),
        ({"checkpoint": "relative"}, "absolute path"),
        ({"model": "7net-0", "cache_dir": "relative"}, "absolute path"),
        ({"model": "7net-0", "device": "auto"}, "device"),
        ({"model": "7net-0", "dtype": "float64"}, "unknown SevenNet options"),
    ],
)
def test_invalid_options_fail_before_optional_setup(options, message, monkeypatch):
    def unexpected(*args):
        pytest.fail("invalid configuration must fail before optional imports")

    monkeypatch.setattr(sevennet, "require_package", unexpected)
    with pytest.raises(ValueError, match=message):
        sevennet.prepare(options)


@pytest.mark.parametrize(
    "variable",
    [
        "SEVENNET_ENABLE_CUEQ",
        "SEVENNET_ENABLE_FLASH",
        "SEVENNET_ENABLE_OEQ",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    ],
)
def test_environment_cannot_silently_change_inference(tmp_path, monkeypatch, backend, variable):
    adapter = sevennet.create_adapter(trajectory=_trajectory(), options=_local(tmp_path))
    monkeypatch.setenv(variable, "1")
    with pytest.raises(ValueError, match="acceleration overrides"):
        adapter._new_calculator()
    assert backend.calls == []


def test_local_checkpoint_cannot_ignore_requested_modality(tmp_path, backend):
    adapter = sevennet.create_adapter(
        trajectory=_trajectory(), options={**_local(tmp_path), "modal": "ignored"}
    )
    with pytest.raises(ValueError, match="did not select"):
        adapter._new_calculator()


def test_restart_binds_weights_and_modal(tmp_path, monkeypatch, backend):
    monkeypatch.setattr(_torch, "_load_torch", lambda device: backend.torch)
    monkeypatch.setattr(_torch, "_torch_environment", lambda torch, device: {"device": device})
    monkeypatch.setattr(_torch, "_preserve_rng", lambda torch: nullcontext())
    monkeypatch.setattr(_torch, "version", lambda name: "fake-version")
    options = {**_local(tmp_path), "modal": "mpa"}
    trajectory = _trajectory()
    atoms = bulk("Si", "diamond", a=5.43)
    adapter = sevennet.create_adapter(trajectory=trajectory, options=options)
    with adapter.start(atoms.copy()) as runtime:
        runtime.run(3)
        full = runtime.snapshot()
    with adapter.start(atoms.copy()) as runtime:
        runtime.run(1)
        partial = runtime.snapshot()
    fresh = sevennet.create_adapter(trajectory=trajectory, options=options)
    with fresh.restore(partial) as runtime:
        runtime.run(2)
        resumed = runtime.snapshot()
    assert resumed.calculator == full.calculator
    for name in full.atoms.arrays:
        np.testing.assert_array_equal(resumed.atoms.arrays[name], full.atoms.arrays[name])
    np.testing.assert_array_equal(resumed.atoms.cell.array, full.atoms.cell.array)
    for name in full.dynamics.arrays:
        np.testing.assert_array_equal(resumed.dynamics.arrays[name], full.dynamics.arrays[name])
    changed = sevennet.create_adapter(trajectory=trajectory, options={**options, "modal": "omat24"})
    with pytest.raises(ValueError, match="environment differs"), changed.restore(partial):
        pass
    Path(options["checkpoint"]).write_text("0.7")
    changed = sevennet.create_adapter(trajectory=trajectory, options=options)
    with pytest.raises(ValueError, match="environment differs"), changed.restore(partial):
        pass
