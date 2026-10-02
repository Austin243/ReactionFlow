from __future__ import annotations

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

from reactionflow.adapters import _torch, orb
from reactionflow.campaign import TrajectorySpec


class FakeModel:
    def __init__(self, path, precision):
        self.scale = float(Path(path).read_text())
        self.dtype = precision
        self.has_stress = False
        self.conservative = True

    def enable_stress(self):
        self.has_stress = True

    def parameters(self):
        return iter([SimpleNamespace(dtype=self.dtype)])


class FakeORBCalculator(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

    def __init__(self, model, atoms_adapter, *, device, edge_method, half_supercell):
        super().__init__()
        self.model = model
        self.adapter = atoms_adapter
        self.device = device
        self.edge_method = edge_method
        self.half_supercell = half_supercell
        self.conservative = model.conservative
        self.evaluations = 0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.adapter.from_ase_atoms(atoms=self.atoms, device=self.device)
        self.evaluations += 1
        self.results = {
            "energy": float(0.5 * self.model.scale * np.sum(self.atoms.positions**2))
            + self.atoms.info.get("charge", 0)
            + self.atoms.info.get("spin", 0),
            "forces": -self.model.scale * self.atoms.positions,
            "stress": np.zeros(6),
        }


@pytest.fixture
def backend(monkeypatch):
    package = ModuleType("orb_models")
    forcefield = ModuleType("orb_models.forcefield")
    pretrained = ModuleType("orb_models.forcefield.pretrained")
    inference = ModuleType("orb_models.forcefield.inference")
    calculator = ModuleType("orb_models.forcefield.inference.calculator")
    calculator.ORBCalculator = FakeORBCalculator
    forcefield.pretrained = pretrained
    constructed, graphs = [], []

    def constructor(**kwargs):
        constructed.append(kwargs)
        model = FakeModel(kwargs["weights_path"], kwargs["precision"])

        def from_ase_atoms(**kwargs):
            assert kwargs["output_dtype"] == model.dtype
            assert kwargs["graph_construction_dtype"] == model.dtype
            graphs.append(kwargs)
            return None

        return model, SimpleNamespace(from_ase_atoms=from_ase_atoms)

    for name in ("orbmol_v2", "orb_v3_conservative_inf_omat", "orb_v3_conservative_inf_mpa"):
        setattr(pretrained, name, constructor)
    for name, module in {
        "orb_models": package,
        "orb_models.forcefield": forcefield,
        "orb_models.forcefield.pretrained": pretrained,
        "orb_models.forcefield.inference": inference,
        "orb_models.forcefield.inference.calculator": calculator,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    checked = []
    monkeypatch.setattr(orb, "require_package", lambda *args: checked.append(args))
    return SimpleNamespace(
        constructed=constructed, graphs=graphs, checked=checked, pretrained=pretrained
    )


def _trajectory():
    return TrajectorySpec.from_dict(
        {
            "id": "orb-test",
            "total_steps": 4,
            "timestep_fs": 0.25,
            "temperature_K": 300,
            "pressure_GPa": 0.1,
            "seed": 77,
        }
    )


def _atoms():
    atoms = Atoms("H2", positions=[[0.1, 0.2, 0.3], [1.1, 0.2, 0.3]], cell=[5] * 3, pbc=True)
    atoms.info.update(charge=0, spin=1)
    return atoms


def _options(tmp_path, *, model="orbmol-v2", precision="float32-highest"):
    checkpoint = tmp_path / "weights.ckpt"
    checkpoint.write_text("0.5")
    return {"model": model, "checkpoint": str(checkpoint), "device": "cpu", "precision": precision}


def test_catalog_is_available_without_model_dependencies(monkeypatch):
    def unexpected_package(*args):
        pytest.fail("catalog listing must not check optional packages")

    monkeypatch.setattr(orb, "require_package", unexpected_package)
    catalog = json.loads(json.dumps(orb.catalog()))
    assert catalog["package"] == "orb-models==0.7.0"
    assert [model["name"] for model in catalog["models"]] == [
        "orbmol-v2",
        "orb-v3-conservative-inf-omat",
        "orb-v3-conservative-inf-mpa",
    ]


@pytest.mark.parametrize("model", [item["name"] for item in orb.catalog()["models"]])
def test_prepare_uses_explicit_published_weight_without_loading_model(
    tmp_path, monkeypatch, backend, model
):
    checkpoint = Path(_options(tmp_path)["checkpoint"])
    calls = []

    def download(url, cache, *, download):
        calls.append((url, cache, download))
        return checkpoint

    monkeypatch.setattr(orb, "cached_download", download)
    for allowed in (True, False):
        assert orb.prepare({"model": model, "cache_dir": str(tmp_path)}, download=allowed) == {
            "checkpoint": checkpoint
        }
    assert [item[2] for item in calls] == [True, False]
    assert all(item[0].startswith("https://orbitalmaterials-public-models.s3.") for item in calls)
    assert all(item[1] == tmp_path / "orb" for item in calls)
    assert backend.constructed == []
    assert backend.checked == [("orb-models", "0.7.0", "orb")] * 2


def test_local_checkpoint_and_constructor_are_offline(tmp_path, monkeypatch, backend):
    options = _options(tmp_path)

    def unexpected_download(*args, **kwargs):
        pytest.fail("local checkpoints must not use the downloader")

    monkeypatch.setattr(orb, "cached_download", unexpected_download)
    assert orb.prepare(options)["checkpoint"] == Path(options["checkpoint"])
    adapter = orb.create_adapter(trajectory=_trajectory(), options=options)
    calculator = adapter._new_calculator()
    assert backend.constructed == [
        {
            "weights_path": options["checkpoint"],
            "device": "cpu",
            "precision": "float32-highest",
            "compile": False,
        }
    ]
    assert calculator.model.has_stress
    assert calculator.edge_method == "knn_alchemi"
    assert calculator.half_supercell is False


def test_named_adapter_cannot_download_missing_weights(tmp_path, monkeypatch, backend):
    def missing(url, cache, *, download):
        assert download is False
        raise FileNotFoundError("prepare first")

    monkeypatch.setattr(orb, "cached_download", missing)
    with pytest.raises(FileNotFoundError, match="prepare first"):
        orb.create_adapter(trajectory=_trajectory(), options={"model": "orbmol-v2"})
    assert backend.constructed == []


@pytest.mark.parametrize(
    "options, message",
    [
        ({}, "supported conservative model"),
        ({"model": ["orbmol-v2"]}, "supported conservative model"),
        ({"model": "orb-v3-direct-inf-omat"}, "supported conservative model"),
        ({"model": "orbmol-v2", "precision": "float32-high"}, "precision"),
        ({"model": "orbmol-v2", "device": "auto"}, "device"),
        ({"model": "orbmol-v2", "checkpoint": "relative.ckpt"}, "absolute path"),
        ({"model": "orbmol-v2", "cache_dir": "relative"}, "absolute path"),
        ({"model": "orbmol-v2", "compile": True}, "unknown ORB"),
    ],
)
def test_bad_options_fail_before_setup(options, message, monkeypatch):
    def unexpected_package(*args):
        pytest.fail("bad options must fail before package checks or downloads")

    monkeypatch.setattr(orb, "require_package", unexpected_package)
    with pytest.raises(ValueError, match=message):
        orb.prepare(options)


@pytest.mark.parametrize("precision", ["float32-highest", "float64"])
def test_graph_dtype_is_bound_to_model_not_process_default(tmp_path, backend, precision):
    adapter = orb.create_adapter(
        trajectory=_trajectory(), options=_options(tmp_path, precision=precision)
    )
    calculator = adapter._new_calculator()
    calculator.get_forces(_atoms())
    assert backend.graphs[0]["output_dtype"] == precision
    assert backend.graphs[0]["graph_construction_dtype"] == precision


@pytest.mark.parametrize("info", [{}, {"charge": 0.0, "spin": 1}, {"charge": 0, "spin": 0}])
def test_molecular_metadata_is_checked_before_cached_results(tmp_path, backend, info):
    adapter = orb.create_adapter(trajectory=_trajectory(), options=_options(tmp_path))
    calculator = adapter._new_calculator()
    atoms = _atoms()
    calculator.get_potential_energy(atoms)
    atoms.info = info
    with pytest.raises(ValueError, match="OrbMol"):
        calculator.get_potential_energy(atoms)
    with pytest.raises(ValueError, match="OrbMol"):
        adapter.preflight(atoms)


def test_changed_charge_or_spin_invalidates_ase_cache(tmp_path, backend):
    adapter = orb.create_adapter(trajectory=_trajectory(), options=_options(tmp_path))
    calculator = adapter._new_calculator()
    atoms = _atoms()
    initial = calculator.get_potential_energy(atoms)
    atoms.info["charge"] = 1
    assert calculator.get_potential_energy(atoms) == initial + 1
    atoms.info["spin"] = 2
    assert calculator.get_potential_energy(atoms) == initial + 2
    assert calculator.evaluations == 3


def test_partial_periodicity_rejected_and_materials_need_no_charge_spin(tmp_path, backend):
    options = _options(tmp_path)
    adapter = orb.create_adapter(trajectory=_trajectory(), options=options)
    atoms = _atoms()
    atoms.pbc = [True, False, False]
    with pytest.raises(ValueError, match="fully periodic or fully nonperiodic"):
        adapter.preflight(atoms)
    atoms.info = {}
    materials = orb.create_adapter(
        trajectory=_trajectory(), options={**options, "model": "orb-v3-conservative-inf-omat"}
    )
    materials.preflight(atoms)
    assert np.isfinite(materials._new_calculator().get_potential_energy(atoms))


def test_restart_preserves_dynamics_and_rejects_changed_weights(tmp_path, monkeypatch, backend):
    monkeypatch.setattr(_torch, "_load_torch", lambda device: SimpleNamespace())
    monkeypatch.setattr(_torch, "_torch_environment", lambda torch, device: {"device": device})
    monkeypatch.setattr(_torch, "_preserve_rng", lambda torch: nullcontext())
    monkeypatch.setattr(_torch, "version", lambda name: "fake-version")
    options = _options(tmp_path)
    adapter = orb.create_adapter(trajectory=_trajectory(), options=options)
    with adapter.start(_atoms()) as runtime:
        runtime.run(4)
        full = runtime.snapshot()
    with adapter.start(_atoms()) as runtime:
        runtime.run(1)
        partial = runtime.snapshot()
    fresh = orb.create_adapter(trajectory=_trajectory(), options=options)
    with fresh.restore(partial) as runtime:
        runtime.run(3)
        resumed = runtime.snapshot()
    assert resumed.calculator == full.calculator
    assert resumed.dynamics.metadata == full.dynamics.metadata
    for key in full.atoms.arrays:
        np.testing.assert_array_equal(resumed.atoms.arrays[key], full.atoms.arrays[key])
    np.testing.assert_array_equal(resumed.atoms.cell.array, full.atoms.cell.array)
    for key in full.dynamics.arrays:
        np.testing.assert_array_equal(resumed.dynamics.arrays[key], full.dynamics.arrays[key])
    Path(options["checkpoint"]).write_text("0.75")
    changed = orb.create_adapter(trajectory=_trajectory(), options=options)
    with pytest.raises(ValueError, match="environment differs"), changed.restore(partial):
        pass
