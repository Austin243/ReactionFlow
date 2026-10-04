from __future__ import annotations

import json
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms
from ase.io import write

from reactionflow.campaign import CampaignConfig
from reactionflow.cli import main
from reactionflow.search import cli as search_cli
from reactionflow.search import records
from reactionflow.search.eon import ProcessResult


class DoubleWell(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x, y, z = self.atoms.positions[0]
        self.results = {
            "energy": (x * x - 1) ** 2 + y * y + z * z,
            "forces": np.array([[-4 * x * (x * x - 1), -2 * y, -2 * z]]),
        }


class Backend:
    """Stands in for EON: the saddle at x=0 joins the minima at x=-1 and x=+1."""

    searches = 0

    def __init__(self, calculator, settings, temperature_K):
        assert isinstance(calculator, DoubleWell)
        self.calculator = calculator
        self.contract = {"backend": "fixture"}

    def energy(self, atoms):
        atoms = atoms.copy()
        atoms.calc = self.calculator
        return atoms.get_potential_energy()

    def relax(self, atoms):
        return atoms, self.energy(atoms)

    def search(self, atoms, *, seed):
        Backend.searches += 1
        saddle, product = atoms.copy(), atoms.copy()
        saddle.positions[0, 0] = 0
        product.positions[0, 0] = -atoms.positions[0, 0]
        energies = [self.energy(item) for item in (atoms, saddle, product)]
        return ProcessResult("good", "", saddle, product, *energies, 1e13, 1e13)

    def same(self, first, first_energy, second, second_energy):
        return np.allclose(first.positions, second.positions)


@pytest.fixture(autouse=True)
def no_scheduler_environment(monkeypatch):
    for name in ("SLURM_NTASKS", "SLURM_PROCID"):
        monkeypatch.delenv(name, raising=False)


def campaign(tmp_path, atoms=None, **overrides):
    if atoms is None:
        atoms = Atoms("H", positions=[[-1, 0, 0]], cell=[12, 12, 12], pbc=True)
    write(tmp_path / "start.extxyz", atoms)
    value = {
        "schema_version": 2,
        "mode": "eon",
        "structure": "start.extxyz",
        "output_root": "runs",
        "require_gpu": False,
        "adapter": {
            "factory": "reactionflow.adapters.ase:create_adapter",
            "options": {"calculator_factory": f"{__name__}:DoubleWell"},
        },
        "akmc": {"steps": 1, "confidence": 0.5, "seed": 7},
        **overrides,
    }
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(value))
    return path


def forbid_resources(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("offline command attempted to load a backend or model")

    monkeypatch.setattr(search_cli, "EONBackend", unexpected)
    monkeypatch.setattr(search_cli, "load_mlip_adapter", unexpected)
    monkeypatch.setattr(records, "ensure_directory", unexpected)


def test_validate_plan_and_status_are_offline(tmp_path, monkeypatch, capsys):
    path = campaign(tmp_path, adapter={"factory": "uninstalled_model:create_adapter"})
    forbid_resources(monkeypatch)
    for command in ("validate", "plan"):
        assert main([command, str(path)]) == 0
        output = json.loads(capsys.readouterr().out)
        assert output["mode"] == "eon" and output["tasks"] == 1 and output["atoms"] == 1
    assert main(["status", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "not_started"
    assert not (tmp_path / "runs").exists()
    # A free molecule's rate also depends on its rotations, which harmonic prefactors omit.
    molecule = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=[12, 12, 12])
    with pytest.raises(ValueError, match="harmonic prefactors"):
        main(["validate", str(campaign(tmp_path, molecule))])
    assert main(["validate", str(campaign(tmp_path, molecule, eon={"prefactor": 1e13}))]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["rigid_rotation"] and not output["warnings"]
    # One fixed atom cannot stop a molecule turning about it.
    molecule.set_constraint(FixAtoms(indices=[0]))
    assert main(["validate", str(campaign(tmp_path, molecule, eon={"prefactor": 1e13}))]) == 0
    output = json.loads(capsys.readouterr().out)
    assert not output["rigid_rotation"] and "turning" in output["warnings"][0]


def test_run_resume_and_read_only_status(tmp_path, monkeypatch, capsys):
    path = campaign(tmp_path)
    monkeypatch.setattr(search_cli, "EONBackend", Backend)
    Backend.searches = 0
    assert main(["run", str(path), "--index", "0"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["steps"] == 1 and output["current"] == "state-000001"
    assert output["attempt_statuses"] == {"new": 1, "repeat": 2}
    row = output["step_rows"][0]
    assert row["barrier_eV"] == pytest.approx(1) and row["bonds"] == "no bond change"
    assert main(["run", str(path)]) == 0
    capsys.readouterr()
    assert Backend.searches == 3
    (tmp_path / "start.extxyz").unlink()
    root = tmp_path / "runs"
    before = {
        p.relative_to(root): (p.stat().st_mtime_ns, p.stat().st_size) for p in root.rglob("*")
    }
    forbid_resources(monkeypatch)
    assert main(["status", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == output
    assert main(["status", str(path)]) == 0
    assert "state-000000 -> state-000001" in capsys.readouterr().out
    after = {p.relative_to(root): (p.stat().st_mtime_ns, p.stat().st_size) for p in root.rglob("*")}
    assert before == after


def test_md_mode_is_optional_and_explicit_md_keeps_validation_output(tmp_path, capsys):
    path = campaign(tmp_path)
    data = {
        "schema_version": 2,
        "structure": "start.extxyz",
        "output_root": "md-runs",
        "require_gpu": False,
        "adapter_profiles": {"fixture": {"factory": "not_imported:factory"}},
        "trajectories": [
            {
                "id": "a",
                "adapter_profile": "fixture",
                "total_steps": 1,
                "timestep_fs": 1,
                "temperature_K": 0,
                "seed": 1,
            }
        ],
    }
    path.write_text(json.dumps(data))
    legacy = CampaignConfig.load(path)
    assert main(["validate", str(path)]) == 0
    original = json.loads(capsys.readouterr().out)
    data["mode"] = "md"
    path.write_text(json.dumps(data))
    assert CampaignConfig.load(path) == legacy
    assert main(["validate", str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == original
