from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from reactionflow.adapters import nep
from reactionflow.campaign import TrajectorySpec


class FakeCPUNEP(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

    def __init__(self, model_filename):
        super().__init__()
        self.path = Path(model_filename)
        self.model_type = self.path.read_text().split(":")[0]
        self.natoms = None
        self.nepy = None
        self._nepy_atoms = None
        self.loads = 0

    def _setup_nepy(self):
        self.factor = float(self.path.read_text().split(":")[1])
        self.natoms = len(self.atoms)
        self.nepy = object()
        self._nepy_atoms = self.atoms.copy()
        self.loads += 1

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if self.nepy is None or self._nepy_atoms is None or self.natoms != len(self.atoms):
            self._setup_nepy()
        self.results = {
            "energy": float(self.factor * np.sum(self.atoms.positions**2) / 2),
            "forces": -self.factor * self.atoms.positions,
            "stress": np.zeros(6),
        }


@pytest.fixture
def backend(tmp_path, monkeypatch):
    checkpoint = tmp_path / "nep.txt"
    checkpoint.write_text("potential:1")
    native = tmp_path / "_nepy.so"
    native.write_bytes(b"native implementation")
    monkeypatch.setattr(nep, "require_package", lambda *args: None)
    monkeypatch.setitem(sys.modules, "_nepy", SimpleNamespace(__file__=str(native)))
    monkeypatch.setitem(sys.modules, "calorine.calculators", SimpleNamespace(CPUNEP=FakeCPUNEP))
    trajectory = TrajectorySpec(
        id="nep", total_steps=2, timestep_fs=0.1, temperature_K=100, pressure_GPa=None, seed=17
    )

    def create():
        return nep.create_adapter(trajectory=trajectory, options={"checkpoint": str(checkpoint)})

    return SimpleNamespace(checkpoint=checkpoint, native=native, create=create)


@pytest.fixture
def atoms():
    return Atoms("Si2", positions=[[0.1, 0.2, 0.3], [1.3, 1.4, 1.5]], cell=[5] * 3, pbc=True)


def test_catalog_is_json_serializable_without_optional_imports():
    assert json.loads(json.dumps(nep.catalog()))["models"][0]["name"] == "nep89"


@pytest.mark.parametrize("download", [False, True])
def test_named_model_respects_explicit_download_and_pinned_source(backend, monkeypatch, download):
    fetch = Mock(return_value=backend.checkpoint)
    monkeypatch.setattr(nep, "cached_download", fetch)
    files = nep.prepare(
        {"model": "nep89", "cache_dir": str(backend.checkpoint.parent)}, download=download
    )
    assert files == {"checkpoint": backend.checkpoint}
    assert fetch.call_args.kwargs == {"download": download}
    assert "/05982941c85b257fdf5a5c69fb7922ce7d7a5c80/" in fetch.call_args.args[0]


def test_local_model_never_downloads_and_missing_file_fails(backend, monkeypatch):
    monkeypatch.setattr(nep, "cached_download", lambda *args, **kwargs: pytest.fail("network"))
    assert backend.create().path == backend.checkpoint
    backend.checkpoint.unlink()
    with pytest.raises(FileNotFoundError):
        backend.create()


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"model": "unknown"},
        {"model": "nep89", "checkpoint": "/tmp/nep.txt"},
        {"checkpoint": "relative.txt"},
        {"model": "nep89", "device": "cuda"},
        {"model": "nep89", "cache_dir": "relative"},
        {"model": "nep89", "dtype": "float32"},
    ],
)
def test_invalid_options_fail_before_importing_backend(monkeypatch, options):
    monkeypatch.setattr(nep, "require_package", lambda *args: pytest.fail("optional import"))
    with pytest.raises(ValueError):
        nep.prepare(options)


@pytest.mark.parametrize("model_type", ["dipole", "polarizability", "potential_with_charges"])
def test_nonpotential_models_rejected_before_native_loading(backend, model_type):
    backend.checkpoint.write_text(f"{model_type}:1")
    with pytest.raises(ValueError, match="energy model"), backend.create().calculator("neb"):
        pytest.fail("unsupported model was leased")


@pytest.mark.parametrize(
    "invalid",
    [
        "nonperiodic",
        "partial",
        "cell",
        "nan_cell",
        "nan_positions",
        "charge",
        "spin",
        "charges",
        "magmoms",
    ],
)
def test_invalid_inputs_rejected_even_after_cached_prediction(backend, atoms, invalid):
    adapter = backend.create()
    with adapter.calculator("neb") as calculator:
        atoms.calc = calculator
        atoms.get_potential_energy()
        if invalid == "nonperiodic":
            atoms.pbc = False
        elif invalid == "partial":
            atoms.pbc = [True, True, False]
        elif invalid == "cell":
            atoms.cell = [5, 5, 0]
        elif invalid == "nan_cell":
            atoms.cell[0, 0] = np.nan
        elif invalid == "nan_positions":
            atoms.positions[0, 0] = np.nan
        elif invalid == "charges":
            atoms.set_initial_charges(np.ones(len(atoms)))
        elif invalid == "magmoms":
            atoms.set_initial_magnetic_moments(np.ones(len(atoms)))
        else:
            atoms.info[invalid] = 2
        with pytest.raises(ValueError):
            adapter.preflight(atoms)
        with pytest.raises(ValueError):
            atoms.get_potential_energy()


def test_lazy_loading_and_free_energy_alias(backend, atoms):
    with backend.create().calculator("neb") as calculator:
        assert calculator.loads == 0
        atoms.calc = calculator
        energy = atoms.get_potential_energy(force_consistent=True)
        assert energy == atoms.get_potential_energy()
        atoms.positions += 0.1
        atoms.get_forces()
        assert calculator.loads == 1


@pytest.mark.parametrize("when", ["before_first_prediction", "before_atom_count_change"])
def test_checkpoint_drift_before_lazy_native_load_is_rejected(backend, atoms, when):
    with backend.create().calculator("neb") as calculator:
        atoms.calc = calculator
        if when == "before_atom_count_change":
            atoms.get_potential_energy()
            atoms += Atoms("Si", positions=[[2, 2, 2]])
        backend.checkpoint.write_text("potential:2")
        with pytest.raises(ValueError, match="before native model loading"):
            atoms.get_potential_energy()


def test_checkpoint_drift_during_native_load_is_rejected(backend, atoms, monkeypatch):
    setup = FakeCPUNEP._setup_nepy

    def replace(self):
        backend.checkpoint.write_text("potential:2")
        setup(self)

    monkeypatch.setattr(FakeCPUNEP, "_setup_nepy", replace)
    with backend.create().calculator("neb") as calculator:
        atoms.calc = calculator
        with pytest.raises(ValueError, match="during native model loading"):
            atoms.get_potential_energy()
        # A failed load must not leave a usable native model with the rejected bytes.
        with pytest.raises(ValueError, match="checkpoint changed"):
            atoms.get_potential_energy()


@pytest.mark.parametrize("changed", ["checkpoint", "native"])
def test_restart_binds_model_and_native_implementation(backend, atoms, changed):
    adapter = backend.create()
    with adapter.start(atoms) as runtime:
        snapshot = runtime.snapshot()
    metadata = snapshot.calculator.metadata
    assert metadata["device"] == "cpu" and metadata["dtype"] == "float64"
    assert metadata["native_library_sha256"] == nep.file_digest(backend.native)
    assert metadata["calculator_source_sha256"] == nep.file_digest(Path(__file__))
    path = getattr(backend, changed)
    path.write_text("potential:2" if changed == "checkpoint" else "different native implementation")
    with pytest.raises(ValueError, match="changed during the run"), adapter.calculator("neb"):
        pass
    with pytest.raises(ValueError, match="environment differs"), backend.create().restore(snapshot):
        pass
