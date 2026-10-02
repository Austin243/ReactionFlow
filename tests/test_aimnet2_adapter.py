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
from ase.calculators.calculator import Calculator, PropertyNotImplementedError, all_changes

from reactionflow.adapters import _torch, aimnet2
from reactionflow.campaign import TrajectorySpec


class FakePredictor:
    def __init__(self, path, *, device, compile_model, train):
        assert Path(path).is_absolute() and Path(path).is_file()
        assert compile_model is False and train is False
        self.payload = json.loads(Path(path).read_text())
        self.metadata = self.payload["metadata"]
        self.is_nse = self.payload.get("is_nse", False)
        self.has_external_coulomb = True
        self.device = device
        self.method = "simple"

    def set_lrcoulomb_method(self, method):
        self.method = method


class FakeASE(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

    def __init__(self, base_calc, *, validate_species):
        super().__init__()
        assert validate_species is True
        self.base_calc = base_calc

    def check_state(self, atoms, tol=1e-15):
        state = super().check_state(atoms, tol=tol)
        if not state and self.atoms.info != atoms.info:
            state.append("info")
        return state

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        properties = properties or ["energy"]
        if set(atoms.numbers) - set(self.base_calc.metadata["implemented_species"]):
            raise ValueError("species not implemented")
        if "stress" in properties and self.base_calc.payload.get("no_stress"):
            raise PropertyNotImplementedError("stress unavailable")
        scale = self.base_calc.payload.get("scale", 0.5)
        self.results = {
            "energy": float(0.5 * scale * np.sum(atoms.positions**2))
            + atoms.info["charge"]
            + atoms.info["spin"],
            "forces": -scale * atoms.positions,
            "stress": np.zeros(6),
        }
        if self.base_calc.payload.get("no_stress"):
            self.results.pop("stress")
        if self.base_calc.payload.get("nonfinite"):
            self.results["forces"][:] = np.nan


@pytest.fixture
def backend(tmp_path, monkeypatch):
    path = tmp_path / "weights.pt"
    data = tmp_path / "dftd3_data.pt"
    data.write_bytes(b"dispersion-reference-data")
    payload = {
        "metadata": {
            "format_version": 2,
            "family": "rxn",
            "supports_charged_systems": False,
            "implemented_species": [1, 6, 7, 8],
        }
    }
    path.write_text(json.dumps(payload))
    calculators = ModuleType("aimnet.calculators")
    calculators.AIMNet2Calculator = FakePredictor
    calculators.AIMNet2ASE = FakeASE
    monkeypatch.setitem(sys.modules, "aimnet", ModuleType("aimnet"))
    monkeypatch.setitem(sys.modules, "aimnet.calculators", calculators)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(float32="float32", set_default_dtype=lambda dtype: None),
    )
    monkeypatch.setattr(aimnet2, "require_package", lambda *args: None)
    monkeypatch.setattr(aimnet2, "_package_file", lambda relative: data)
    monkeypatch.setattr(_torch, "_load_torch", lambda device: None)
    monkeypatch.setattr(_torch, "_preserve_rng", lambda torch: nullcontext())
    monkeypatch.setattr(_torch, "_torch_environment", lambda torch, device: {"device": device})
    monkeypatch.setattr(_torch, "version", lambda name: "test-version")
    return SimpleNamespace(path=path, data=data, payload=payload, calculators=calculators)


def _trajectory(pressure=None):
    return TrajectorySpec.from_dict(
        {
            "id": "aimnet-test",
            "total_steps": 4,
            "timestep_fs": 0.1,
            "temperature_K": 100.0,
            "pressure_GPa": pressure,
            "seed": 19,
        }
    )


def _atoms(periodic=False):
    atoms = Atoms("H2", positions=[[0.1, 0.2, 0.3], [0.8, 0.2, 0.3]])
    atoms.info.update(charge=0, spin=1)
    if periodic:
        atoms.set_cell([5] * 3)
        atoms.pbc = True
    return atoms


def _adapter(backend, *, pressure=None, **options):
    return aimnet2.create_adapter(
        trajectory=_trajectory(pressure),
        options={"checkpoint": str(backend.path), "device": "cpu", **options},
    )


def test_catalog_is_json_serializable_without_optional_backend(monkeypatch):
    def unexpected(*args):
        pytest.fail("the model catalog must not load optional packages")

    monkeypatch.setattr(aimnet2, "require_package", unexpected)
    monkeypatch.setattr(aimnet2, "distribution", unexpected)
    catalog = json.loads(json.dumps(aimnet2.catalog()))
    assert catalog["package"] == "aimnet[ase]==0.2.0"
    assert len(catalog["models"]) == 6
    assert "aimnet2-rxn" in [item["name"] for item in catalog["models"]]


def test_prepare_resolves_one_member_and_published_hash_without_loading_predictor(
    tmp_path, monkeypatch, backend
):
    calls = []
    registry = {
        "aliases": {"aimnet2-rxn": "aimnet2-rxn_0"},
        "models": {
            "aimnet2-rxn_2": {
                "url": "https://models.example/rxn2.pt",
                "sha256": hashlib.sha256(backend.path.read_bytes()).hexdigest(),
            }
        },
    }
    monkeypatch.setattr(aimnet2, "_registry", lambda: registry)

    def fetch(url, cache, *, download):
        calls.append((url, download))
        return backend.path

    def unexpected_predictor(*args, **kwargs):
        pytest.fail("preparation must not load the model")

    monkeypatch.setattr(aimnet2, "cached_download", fetch)
    monkeypatch.setattr(backend.calculators, "AIMNet2Calculator", unexpected_predictor)
    options = {"model": "aimnet2-rxn", "model_index": 2, "cache_dir": str(tmp_path)}
    assert aimnet2.prepare(options) == {"checkpoint": backend.path, "dftd3": backend.data}
    assert aimnet2.prepare(options, download=False)["checkpoint"] == backend.path
    assert calls == [
        ("https://models.example/rxn2.pt", True),
        ("https://models.example/rxn2.pt", False),
    ]
    backend.path.write_text("corrupt")
    with pytest.raises(ValueError, match="published SHA-256"):
        aimnet2.prepare(options, download=False)


def test_normal_adapter_creation_never_downloads(monkeypatch, backend):
    monkeypatch.setattr(
        aimnet2,
        "_registry",
        lambda: {
            "aliases": {"aimnet2": "aimnet2-wb97m-d3_0"},
            "models": {"aimnet2-wb97m-d3_0": {"url": "https://models.example/general.pt"}},
        },
    )

    def missing_cache(url, cache, *, download):
        assert download is False
        raise FileNotFoundError("prepare the model first")

    monkeypatch.setattr(aimnet2, "cached_download", missing_cache)
    with pytest.raises(FileNotFoundError, match="prepare"):
        aimnet2.create_adapter(trajectory=_trajectory(), options={"model": "aimnet2"})


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"model": "aimnet2", "checkpoint": "/tmp/model"},
        {"model": "unknown"},
        {"model": "aimnet2", "model_index": True},
        {"model": "aimnet2", "model_index": 4},
        {"checkpoint": "/tmp/model", "model_index": 0},
        {"checkpoint": "relative.pt"},
        {"model": "aimnet2", "cache_dir": "relative"},
        {"model": "aimnet2", "device": "auto"},
        {"model": "aimnet2", "dtype": "float64"},
        {"model": "aimnet2", "coulomb_method": "unknown"},
    ],
)
def test_invalid_options_fail_before_loading_or_downloading(options, monkeypatch):
    def unexpected(*args):
        pytest.fail("invalid options reached an optional dependency")

    monkeypatch.setattr(aimnet2, "require_package", unexpected)
    with pytest.raises(ValueError):
        aimnet2.prepare(options)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"charge": None}, "explicit integer"),
        ({"charge": False}, "explicit integer"),
        ({"spin": 0}, "positive multiplicity"),
        ({"spin": 3}, "singlets only"),
        ({"charge": 1}, "net charge zero"),
        ({"mult": 3}, "must agree"),
    ],
)
def test_electronic_state_is_checked_even_before_cached_results(backend, changes, message):
    atoms = _atoms()
    with _adapter(backend).calculator("neb") as calculator:
        atoms.calc = calculator
        assert np.isfinite(atoms.get_potential_energy())
        atoms.info.update(changes)
        with pytest.raises(ValueError, match=message):
            atoms.get_potential_energy()


def test_open_shell_model_uses_changed_charge_and_spin_without_stale_cache(backend):
    backend.payload["is_nse"] = True
    backend.payload["metadata"].update(family="nse", supports_charged_systems=True)
    backend.path.write_text(json.dumps(backend.payload))
    atoms = _atoms()
    with _adapter(backend).calculator("neb") as calculator:
        atoms.calc = calculator
        original = atoms.get_potential_energy()
        atoms.info["charge"] = 1
        assert atoms.get_potential_energy() == original + 1
        atoms.info["spin"] = 3
        assert atoms.get_potential_energy() == original + 3


def test_auto_coulomb_does_not_retain_periodic_mode_for_molecules(backend):
    with _adapter(backend).calculator("neb") as calculator:
        periodic = _atoms(True)
        periodic.calc = calculator
        periodic.get_forces()
        assert calculator.base_calc.method == "dsf"
        molecule = _atoms()
        molecule.calc = calculator
        molecule.get_forces()
        assert calculator.base_calc.method == "simple"


def test_periodic_use_does_not_silently_keep_embedded_all_pairs_coulomb(backend):
    backend.payload["metadata"]["coulomb_mode"] = "full_embedded"
    backend.path.write_text(json.dumps(backend.payload))
    with pytest.raises(ValueError, match="embedded Coulomb method"):
        _adapter(backend, pressure=0.0).preflight(_atoms(True))


def test_npt_preflight_rejects_missing_cell_or_stress_and_nonfinite_output(backend):
    adapter = _adapter(backend, pressure=0.0)
    with pytest.raises(ValueError, match="fully periodic"):
        adapter.preflight(_atoms())
    backend.payload["no_stress"] = True
    backend.path.write_text(json.dumps(backend.payload))
    with pytest.raises(PropertyNotImplementedError, match="stress unavailable"):
        _adapter(backend, pressure=0.0).preflight(_atoms(True))
    backend.payload.pop("no_stress")
    backend.payload["nonfinite"] = True
    backend.path.write_text(json.dumps(backend.payload))
    with pytest.raises(ValueError, match="non-finite"):
        _adapter(backend).preflight(_atoms())


@pytest.mark.parametrize("pressure", [None, 0.0])
def test_restart_is_exact_and_binds_model_and_dispersion_data(backend, pressure):
    adapter = _adapter(backend, pressure=pressure)
    atoms = _atoms(pressure is not None)
    adapter.preflight(atoms)
    with adapter.start(atoms.copy()) as runtime:
        runtime.run(4)
        full = runtime.snapshot()
    with adapter.start(atoms.copy()) as runtime:
        runtime.run(2)
        split = runtime.snapshot()
    with _adapter(backend, pressure=pressure).restore(split) as runtime:
        runtime.run(2)
        resumed = runtime.snapshot()
    assert resumed.calculator == full.calculator
    assert resumed.dynamics.metadata == full.dynamics.metadata
    for name in full.atoms.arrays:
        np.testing.assert_array_equal(resumed.atoms.arrays[name], full.atoms.arrays[name])
    np.testing.assert_array_equal(resumed.atoms.cell.array, full.atoms.cell.array)
    for name in full.dynamics.arrays:
        np.testing.assert_array_equal(resumed.dynamics.arrays[name], full.dynamics.arrays[name])
    assert "dftd3" in full.calculator.metadata["model_files"]
    backend.data.write_bytes(b"changed dispersion parameters")
    with pytest.raises(ValueError, match="environment differs"), _adapter(backend).restore(split):
        pass
