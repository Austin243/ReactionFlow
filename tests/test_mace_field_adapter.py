from __future__ import annotations

import hashlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from reactionflow.adapters import _torch, mace_field
from reactionflow.campaign import TrajectorySpec


@pytest.fixture
def backend(monkeypatch):
    torch = ModuleType("torch")
    torch.float32 = "float32"
    torch.float64 = "float64"
    state = {"dtype": "float32"}
    torch.get_default_dtype = lambda: state["dtype"]
    torch.set_default_dtype = lambda dtype: state.update(dtype=dtype)
    loaded, constructed = [], []

    def load(path, *, map_location, weights_only):
        loaded.append((path, map_location, weights_only))
        return SimpleNamespace(scale=float(Path(path).read_text()))

    torch.load = load

    class FakeCalculator(Calculator):
        implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

        def __init__(self, **kwargs):
            super().__init__()
            constructed.append(kwargs)
            self.head = kwargs["head"] if kwargs["head"] != "typo" else "other"
            self.kwargs = kwargs
            torch.set_default_dtype(kwargs["default_dtype"])

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            assert torch.get_default_dtype() == self.kwargs["default_dtype"]
            super().calculate(atoms, properties, system_changes)
            scale = self.kwargs["models"][0].scale
            field = np.asarray(self.kwargs["electric_field"])
            self.results = {
                "energy": float(
                    0.5 * scale * np.sum(self.atoms.positions**2)
                    - np.sum(self.atoms.positions * field)
                ),
                "forces": -scale * self.atoms.positions + field,
                "stress": np.zeros(6),
            }

    package = ModuleType("mace")
    calculators = ModuleType("mace.calculators")
    calculators.MACECalculator = FakeCalculator
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "mace", package)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators)
    monkeypatch.setattr(mace_field, "_require_fork", lambda: None)
    return SimpleNamespace(torch=torch, loaded=loaded, constructed=constructed)


def _trajectory():
    return TrajectorySpec.from_dict(
        {
            "id": "field-test",
            "total_steps": 4,
            "timestep_fs": 0.25,
            "temperature_K": 100,
            "pressure_GPa": 0.1,
            "seed": 77,
        }
    )


def _atoms():
    return Atoms("H2", positions=[[0.1, 0.2, 0.3], [1.1, 0.2, 0.3]], cell=[5] * 3, pbc=True)


def _options(tmp_path):
    checkpoint = tmp_path / "field.model"
    checkpoint.write_text("0.5")
    return {
        "checkpoint": str(checkpoint),
        "head": "mp-dielectric",
        "electric_field": [0.0, 0.0, 0.01],
        "device": "cpu",
        "dtype": "float64",
    }


def test_fork_provenance_requires_exact_repository_and_commit(monkeypatch):
    checked = []
    monkeypatch.setattr(mace_field, "require_package", lambda *args: checked.append(args))
    correct = {
        "url": mace_field.SOURCE_URL,
        "vcs_info": {
            "vcs": "git",
            "commit_id": mace_field.SOURCE_COMMIT,
        },
    }
    for origin in [
        None,
        "invalid-json",
        json.dumps({}),
        json.dumps(
            {
                **correct,
                "url": "https://github.com/ACEsuit/mace.git",
            }
        ),
        json.dumps({**correct, "vcs_info": {"vcs": "git", "commit_id": "wrong"}}),
    ]:
        monkeypatch.setattr(
            mace_field,
            "distribution",
            lambda name, origin=origin: SimpleNamespace(read_text=lambda filename: origin),
        )
        with pytest.raises(RuntimeError, match="pinned Git source"):
            mace_field._require_fork()
    monkeypatch.setattr(
        mace_field,
        "distribution",
        lambda name: SimpleNamespace(read_text=lambda filename: json.dumps(correct)),
    )
    mace_field._require_fork()
    assert checked[-2:] == [("mace-torch", "0.3.15", "mace-field"), ("e3nn", "0.4.4", "mace-field")]


def test_catalog_lists_verified_heads_without_importing_backend(monkeypatch):
    def unexpected():
        pytest.fail("listing must be independent of the installed fork")

    monkeypatch.setattr(mace_field, "_require_fork", unexpected)
    catalog = json.loads(json.dumps(mace_field.catalog()))
    assert catalog["backend"] == "mace_field"
    assert catalog["factory"] == "reactionflow.adapters.mace_field:create_adapter"
    assert catalog["models"][0]["heads"] == ["pt_head", "mp-dielectric", "mp-ferroelectric"]


def test_missing_package_points_to_supported_install_command(monkeypatch):
    def missing(*args):
        raise RuntimeError("missing dependency")

    monkeypatch.setattr(mace_field, "require_package", missing)
    with pytest.raises(RuntimeError, match=r"prepare campaign\.json --install") as error:
        mace_field._require_fork()
    assert mace_field.INSTALL_REQUIREMENT in str(error.value)
    assert "reactionflow[mace-field]" not in str(error.value)


def test_prepare_checks_named_checksum_and_never_constructs_calculator(
    tmp_path, monkeypatch, backend
):
    local = _options(tmp_path)
    checkpoint = Path(local.pop("checkpoint"))
    options = {**local, "model": "MACEField-MH-0-omat-dielectric", "cache_dir": str(tmp_path)}
    calls = []

    def download(url, cache, *, download):
        calls.append((url, cache, download))
        return checkpoint

    monkeypatch.setattr(mace_field, "cached_download", download)
    with pytest.raises(ValueError, match="checksum differs"):
        mace_field.prepare(options)
    monkeypatch.setattr(mace_field, "_SHA256", hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    assert mace_field.prepare(options, download=False) == {"checkpoint": checkpoint}
    assert calls[-1][0].endswith("/1.0.2/MACEField-MH-0-omat-dielectric.model")
    assert calls[-1][1] == tmp_path / "mace-field"
    assert calls[-1][2] is False
    assert backend.loaded == backend.constructed == []


def test_local_weights_fixed_field_and_precision_are_used(tmp_path, monkeypatch, backend):
    options = _options(tmp_path)

    def unexpected(*args, **kwargs):
        pytest.fail("local weights must never invoke downloading")

    monkeypatch.setattr(mace_field, "cached_download", unexpected)
    adapter = mace_field.create_adapter(trajectory=_trajectory(), options=options)
    calculator = adapter._new_calculator()
    assert backend.loaded == [(Path(options["checkpoint"]), "cpu", False)]
    kwargs = backend.constructed[0]
    assert kwargs["electric_field"] == [0.0, 0.0, 0.01]
    assert kwargs["model_type"] == "MACEField"
    assert kwargs["compile_mode"] is None
    assert kwargs["enable_cueq"] is kwargs["enable_oeq"] is False
    backend.torch.set_default_dtype("float32")
    calculator.get_forces(_atoms())
    assert backend.torch.get_default_dtype() == "float32"
    assert adapter._extra_metadata(calculator)["source_commit"] == mace_field.SOURCE_COMMIT


@pytest.mark.parametrize(
    "override, message",
    [
        ({"head": ""}, "explicit head"),
        ({"electric_field": None}, "three finite"),
        ({"electric_field": [0, 0]}, "three finite"),
        ({"electric_field": [True, 0, 0]}, "three finite"),
        ({"electric_field": [0, float("nan"), 0]}, "three finite"),
        ({"device": "auto"}, "device"),
        ({"dtype": "float16"}, "dtype"),
        ({"checkpoint": "relative"}, "absolute path"),
        ({"time_dependent_field": True}, "unknown MACE-Field"),
    ],
)
def test_invalid_options_rejected_before_setup(tmp_path, monkeypatch, override, message):
    options = {**_options(tmp_path), **override}

    def unexpected():
        pytest.fail("invalid options must be rejected before setup")

    monkeypatch.setattr(mace_field, "_require_fork", unexpected)
    with pytest.raises(ValueError, match=message):
        mace_field.prepare(options)


def test_wrong_local_head_is_not_silently_substituted(tmp_path, backend):
    adapter = mace_field.create_adapter(
        trajectory=_trajectory(), options={**_options(tmp_path), "head": "typo"}
    )
    with pytest.raises(ValueError, match="did not select requested head"):
        adapter._new_calculator()


def test_preflight_rejects_zero_cell_volume(tmp_path, backend):
    adapter = mace_field.create_adapter(trajectory=_trajectory(), options=_options(tmp_path))
    with pytest.raises(ValueError, match="nonzero cell volume"):
        adapter.preflight(Atoms("He"))
    adapter.preflight(_atoms())


def test_exact_restart_rejects_changed_field_head_and_weights(tmp_path, monkeypatch, backend):
    monkeypatch.setattr(_torch, "_load_torch", lambda device: backend.torch)
    monkeypatch.setattr(_torch, "_torch_environment", lambda torch, device: {"device": device})
    monkeypatch.setattr(_torch, "_preserve_rng", lambda torch: nullcontext())
    monkeypatch.setattr(_torch, "version", lambda name: "fake-version")
    options = _options(tmp_path)
    adapter = mace_field.create_adapter(trajectory=_trajectory(), options=options)
    with adapter.start(_atoms()) as runtime:
        runtime.run(4)
        full = runtime.snapshot()
    with adapter.start(_atoms()) as runtime:
        runtime.run(1)
        partial = runtime.snapshot()
    fresh = mace_field.create_adapter(trajectory=_trajectory(), options=options)
    with fresh.restore(partial) as runtime:
        runtime.run(3)
        resumed = runtime.snapshot()
    assert full.calculator == resumed.calculator
    for key in full.atoms.arrays:
        np.testing.assert_array_equal(full.atoms.arrays[key], resumed.atoms.arrays[key])
    np.testing.assert_array_equal(full.atoms.cell.array, resumed.atoms.cell.array)
    for key in full.dynamics.arrays:
        np.testing.assert_array_equal(full.dynamics.arrays[key], resumed.dynamics.arrays[key])
    for updates in ({"head": "mp-ferroelectric"}, {"electric_field": [0, 0, 0.02]}):
        changed = mace_field.create_adapter(
            trajectory=_trajectory(), options={**options, **updates}
        )
        with pytest.raises(ValueError, match="environment differs"), changed.restore(partial):
            pass
    Path(options["checkpoint"]).write_text("0.75")
    changed = mace_field.create_adapter(trajectory=_trajectory(), options=options)
    with pytest.raises(ValueError, match="environment differs"), changed.restore(partial):
        pass


def test_field_under_pressure_needs_a_cell_that_pathways_do_not_rotate(
    tmp_path, monkeypatch, backend
):
    monkeypatch.setattr(_torch, "_load_torch", lambda device: backend.torch)
    monkeypatch.setattr(_torch, "_torch_environment", lambda torch, device: {"device": device})
    monkeypatch.setattr(_torch, "_preserve_rng", lambda torch: nullcontext())
    monkeypatch.setattr(_torch, "version", lambda name: "fake-version")
    # SSNEB would rotate this cell's a vector onto x, and the molecule with it, but not the field.
    tilted = _atoms()
    tilted.set_cell([[0, 5, 0], [-5, 0, 0], [0, 0, 5]], scale_atoms=True)
    adapter = mace_field.create_adapter(trajectory=_trajectory(), options=_options(tmp_path))
    with pytest.raises(ValueError, match="lower-triangular cell"), adapter.start(tilted):
        pass
    zero = mace_field.create_adapter(
        trajectory=_trajectory(), options={**_options(tmp_path), "electric_field": [0, 0, 0]}
    )
    with zero.start(tilted) as runtime:
        runtime.run(1)
