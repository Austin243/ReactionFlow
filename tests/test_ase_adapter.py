from __future__ import annotations

import hashlib
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from reactionflow.adapters.ase import ASECalculatorAdapter
from reactionflow.campaign import AdapterSpec, TrajectorySpec
from reactionflow.mlip import load_mlip_adapter


class HarmonicSolid(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

    def __init__(self, *, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = self.atoms.positions
        self.results = {
            "energy": float(0.5 * self.scale * np.sum(positions**2)),
            "forces": -self.scale * positions.copy(),
            "stress": np.zeros(6),
        }


def create_harmonic_calculator(*, scale: float = 1.0) -> Calculator:
    return HarmonicSolid(scale=scale)


def create_invalid_calculator() -> object:
    return object()


def _trajectory(**overrides) -> TrajectorySpec:
    value = {
        "id": "generic-mlip",
        "total_steps": 6,
        "timestep_fs": 0.25,
        "temperature_K": 300.0,
        "pressure_GPa": 0.1,
        "seed": 77,
        "conditions": {
            "hydrostatic": True,
            "thermostat_tau_fs": 25.0,
            "barostat_tau_fs": 250.0,
        },
        **overrides,
    }
    return TrajectorySpec.from_dict(value)


def _atoms() -> Atoms:
    return Atoms(
        "H2",
        positions=[[0.2, 0.1, 0.3], [1.1, 0.4, 0.2]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )


def _options(model: Path) -> dict[str, object]:
    return {
        "calculator_factory": f"{__name__}:create_harmonic_calculator",
        "calculator_kwargs": {"scale": 0.5},
        "model_files": [str(model)],
        "packages": ["ase"],
    }


def test_campaign_factory_loads_a_configured_ase_calculator(tmp_path) -> None:
    model = tmp_path / "model.ckpt"
    model.write_bytes(b"weights-v1")
    spec = AdapterSpec(
        factory="reactionflow.adapters.ase:create_adapter",
        options=_options(model),
    )

    adapter = load_mlip_adapter(spec, _trajectory())

    assert isinstance(adapter, ASECalculatorAdapter)
    with adapter.calculator("neb") as calculator:
        assert isinstance(calculator, HarmonicSolid)
        assert calculator.scale == 0.5


def test_generic_adapter_restart_matches_uninterrupted_npt_exactly(tmp_path) -> None:
    model = tmp_path / "model.ckpt"
    model.write_bytes(b"weights-v1")
    adapter = ASECalculatorAdapter(trajectory=_trajectory(), options=_options(model))

    uninterrupted_atoms = _atoms()
    with adapter.start(uninterrupted_atoms) as uninterrupted:
        uninterrupted.run(6)
        uninterrupted_snapshot = uninterrupted.snapshot()

    split_atoms = _atoms()
    with adapter.start(split_atoms) as split:
        split.run(2)
        checkpoint = split.snapshot()
    with adapter.restore(checkpoint) as resumed:
        resumed.run(4)
        resumed_snapshot = resumed.snapshot()

    assert resumed_snapshot.calculator == uninterrupted_snapshot.calculator
    assert resumed_snapshot.dynamics.metadata == uninterrupted_snapshot.dynamics.metadata
    for name in resumed_snapshot.atoms.arrays:
        np.testing.assert_array_equal(
            resumed_snapshot.atoms.arrays[name],
            uninterrupted_snapshot.atoms.arrays[name],
        )
    np.testing.assert_array_equal(
        resumed_snapshot.atoms.cell.array,
        uninterrupted_snapshot.atoms.cell.array,
    )
    for name in resumed_snapshot.dynamics.arrays:
        np.testing.assert_array_equal(
            resumed_snapshot.dynamics.arrays[name],
            uninterrupted_snapshot.dynamics.arrays[name],
        )


def test_model_file_and_environment_are_bound_to_exact_restart(tmp_path) -> None:
    model = tmp_path / "model.ckpt"
    model.write_bytes(b"weights-v1")
    adapter = ASECalculatorAdapter(trajectory=_trajectory(), options=_options(model))

    with adapter.start(_atoms()) as runtime:
        checkpoint = runtime.snapshot()

    metadata = checkpoint.calculator.metadata
    assert metadata["model_files"] == {str(model): hashlib.sha256(b"weights-v1").hexdigest()}
    assert metadata["package_versions"]["ase"]

    model.write_bytes(b"weights-v2")
    changed_adapter = ASECalculatorAdapter(trajectory=_trajectory(), options=_options(model))
    with (
        pytest.raises(ValueError, match="environment differs"),
        changed_adapter.restore(checkpoint),
    ):
        pass


def test_generic_adapter_rejects_invalid_configuration_and_factory(tmp_path) -> None:
    model = tmp_path / "model.ckpt"
    model.write_bytes(b"weights")

    with pytest.raises(ValueError, match="calculator_factory"):
        ASECalculatorAdapter(trajectory=_trajectory(), options={})
    with pytest.raises(ValueError, match="unknown ASE calculator adapter options"):
        ASECalculatorAdapter(
            trajectory=_trajectory(),
            options={**_options(model), "mystery": True},
        )
    with pytest.raises(TypeError, match="model_files"):
        ASECalculatorAdapter(
            trajectory=_trajectory(),
            options={**_options(model), "model_files": str(model)},
        )
    with pytest.raises(ValueError, match="unknown ASE trajectory conditions"):
        ASECalculatorAdapter(
            trajectory=_trajectory(conditions={"unknown": True}),
            options=_options(model),
        )

    invalid = ASECalculatorAdapter(
        trajectory=_trajectory(),
        options={
            "calculator_factory": f"{__name__}:create_invalid_calculator",
            "model_files": [str(model)],
        },
    )
    with pytest.raises(TypeError, match="not an ASE Calculator"), invalid.calculator("neb"):
        pass


def test_generic_adapter_requires_each_declared_model_file(tmp_path) -> None:
    missing = tmp_path / "missing.ckpt"
    adapter = ASECalculatorAdapter(trajectory=_trajectory(), options=_options(missing))

    with pytest.raises(FileNotFoundError, match="model file is missing"), adapter.start(_atoms()):
        pass


def test_model_neutral_perlmutter_setup_is_separate_from_ani() -> None:
    root = Path(__file__).parents[1]
    setup = (root / "scripts/setup-perlmutter.sh").read_text(encoding="utf-8")
    core = (root / "requirements/perlmutter-core.txt").read_text(encoding="utf-8")
    ani = (root / "requirements/perlmutter-ani1xnr.txt").read_text(encoding="utf-8")

    assert "module load pytorch/2.11.0" in setup
    assert "PYTHONUSERBASE" in setup
    assert "perlmutter-core.txt" in setup
    assert "torchani" not in setup.lower()
    assert "cache_ani1xnr" not in setup
    assert "torchani" not in core.lower()
    assert "-r perlmutter-core.txt" in ani
