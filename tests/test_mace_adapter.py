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
from ase.constraints import Hookean
from ase.stress import full_3x3_to_voigt_6_stress

from reactionflow.adapters import _torch, mace
from reactionflow.campaign import TrajectorySpec


class FakePolarModel:
    pass


class FakeMACECalculator(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, *, model_paths, device, default_dtype, head=None, model_type="MACE"):
        super().__init__()
        self.model_paths = model_paths
        self.device = device
        self.default_dtype = default_dtype
        self.model_type = model_type
        self.models = [FakePolarModel() if model_type == "PolarMACE" else object()]
        self.calculations = 0
        self.scale = float(Path(model_paths[0]).read_text())
        # MACE 0.3.16 warns and substitutes the final head on an unknown name.
        self.head = (
            head
            if head
            in (
                "first",
                "second",
                "matpes_r2scan",
                "mp_pbe_refit_add",
                "spice_wB97M",
                "oc20_usemppbe",
                "omol",
                "omat_pbe",
            )
            else "Default"
        )

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calculations += 1
        self.results = {
            "energy": float(0.5 * self.scale * np.sum(self.atoms.positions**2)),
            "free_energy": float(0.5 * self.scale * np.sum(self.atoms.positions**2)),
            "forces": -self.scale * self.atoms.positions,
            "stress": np.zeros(6),
        }


@pytest.fixture
def backend(monkeypatch):
    package = ModuleType("mace")
    calculators = ModuleType("mace.calculators")
    calculators.MACECalculator = FakeMACECalculator
    modules = ModuleType("mace.modules")
    modules.PolarMACE = FakePolarModel
    registry = ModuleType("mace.calculators.foundations_models")
    registry.mace_mp_urls = {
        "medium-mpa-0": "https://example.test/mp.model",
        "mh-1": "https://example.test/mh-1.model",
    }
    registry.mace_off_urls = {"medium": "https://example.test/off.model"}
    registry.polar_model_urls = {
        f"polar-1-{size}": f"https://example.test/polar-{size}.model" for size in ("s", "m", "l")
    }
    monkeypatch.setitem(sys.modules, "mace", package)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators)
    monkeypatch.setitem(sys.modules, "mace.modules", modules)
    monkeypatch.setitem(sys.modules, "mace.calculators.foundations_models", registry)
    checked = []
    monkeypatch.setattr(mace, "require_package", lambda *args: checked.append(args))
    return SimpleNamespace(calculators=calculators, checked=checked)


def _trajectory(*, pressure=0.1) -> TrajectorySpec:
    return TrajectorySpec.from_dict(
        {
            "id": "mace-test",
            "total_steps": 6,
            "timestep_fs": 0.25,
            "temperature_K": 300.0,
            "pressure_GPa": pressure,
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


@pytest.mark.parametrize(
    "head",
    ["matpes_r2scan", "mp_pbe_refit_add", "spice_wB97M", "oc20_usemppbe", "omol", "omat_pbe"],
)
def test_named_mh1_supports_each_released_head_without_runtime_download(
    tmp_path, monkeypatch, backend, head
):
    checkpoint = _checkpoint(tmp_path)
    downloads = []

    def cached(url, cache, *, download):
        downloads.append((url, download))
        return checkpoint

    monkeypatch.setattr(mace, "cached_download", cached)
    options = {"model": "mh-1", "head": head, "device": "cpu"}
    mace.prepare(options)
    adapter = mace.create_adapter(trajectory=_trajectory(), options=options)
    assert adapter._new_calculator().head == head
    assert downloads == [
        ("https://example.test/mh-1.model", True),
        ("https://example.test/mh-1.model", False),
    ]


@pytest.mark.parametrize(
    "options", [{"model": "mh-1"}, {"model": "mh-0"}, {"model": "mh-1", "head": "rgd1_b3lyp"}]
)
def test_multhead_selection_errors_fail_before_packages_or_download(monkeypatch, options):
    def forbidden(*args):
        pytest.fail("invalid head selection must fail before setup")

    monkeypatch.setattr(mace, "require_package", forbidden)
    with pytest.raises(ValueError, match="head"):
        mace.prepare(options)


@pytest.mark.parametrize("size", ["s", "m", "l"])
def test_each_polar_size_uses_polar_calculator_and_offline_cache(
    tmp_path, monkeypatch, backend, size
):
    downloads = []
    checkpoint = _checkpoint(tmp_path)

    def cached(url, cache, *, download):
        downloads.append((url, download))
        return checkpoint

    monkeypatch.setattr(mace, "cached_download", cached)
    adapter = mace.create_adapter(
        trajectory=_trajectory(pressure=None),
        options={"family": "polar", "model": f"polar-1-{size}"},
    )
    assert adapter._new_calculator().inner.model_type == "PolarMACE"
    assert "graph_longrange" in adapter.packages
    assert downloads == [(f"https://example.test/polar-{size}.model", False)]


def _polar_calculator(tmp_path, **options):
    adapter = mace.create_adapter(
        trajectory=_trajectory(pressure=None),
        options={
            "family": "polar",
            "checkpoint": str(_checkpoint(tmp_path)),
            "device": "cpu",
            **options,
        },
    )
    return adapter._new_calculator()


def test_polar_metadata_changes_invalidate_cache_including_in_place_field(tmp_path, backend):
    calculator = _polar_calculator(tmp_path)
    atoms = _atoms()
    atoms.info.update(charge=0, spin=1, external_field=np.zeros(3))
    calculator.get_potential_energy(atoms)
    calculator.get_potential_energy(atoms)
    assert calculator.inner.calculations == 1
    atoms.info["charge"] = 1
    calculator.get_potential_energy(atoms)
    atoms.info["spin"] = 3
    calculator.get_potential_energy(atoms)
    atoms.info["external_field"][2] = 0.01
    calculator.get_potential_energy(atoms)
    assert calculator.inner.calculations == 4


@pytest.mark.parametrize(
    "info",
    [
        {},
        {"charge": 0},
        {"charge": False, "spin": 1},
        {"charge": 0, "spin": 0},
        {"charge": 0, "spin": 1, "external_field": [0, 0]},
        {"charge": 0, "spin": 1, "external_field": [0, 0, float("nan")]},
    ],
)
def test_polar_rejects_missing_or_invalid_physical_inputs_before_cached_results(
    tmp_path, backend, info
):
    calculator = _polar_calculator(tmp_path)
    atoms = _atoms()
    atoms.info.update(charge=0, spin=1)
    calculator.get_potential_energy(atoms)
    atoms.info = info
    with pytest.raises(ValueError, match="MACE-POLAR"):
        calculator.get_potential_energy(atoms)


def test_polar_missing_dependency_names_setup_command(tmp_path, monkeypatch, backend):
    def require(name, *args):
        if name == "graph_longrange":
            raise RuntimeError("not installed")

    monkeypatch.setattr(mace, "require_package", require)
    with pytest.raises(RuntimeError, match=r"prepare campaign\.json --install"):
        _polar_calculator(tmp_path)


@pytest.mark.parametrize("pressure", [0.0, 0.1])
def test_polar_npt_requires_float64_for_numerical_stress(tmp_path, backend, pressure):
    with pytest.raises(ValueError, match="NPT requires dtype='float64'"):
        mace.create_adapter(
            trajectory=_trajectory(pressure=pressure),
            options={
                "family": "polar",
                "checkpoint": str(_checkpoint(tmp_path)),
                "dtype": "float32",
            },
        )


def test_polar_float32_fixed_cell_is_allowed_but_stress_requires_float64(tmp_path, backend):
    calculator = _polar_calculator(tmp_path, dtype="float32")
    atoms = _atoms()
    atoms.info.update(charge=0, spin=1)
    assert np.isfinite(calculator.get_potential_energy(atoms))
    with pytest.raises(ValueError, match="numerical stress requires dtype='float64'"):
        calculator.get_stress(atoms)


@pytest.mark.parametrize("constrained", [False, True])
def test_polar_stress_is_lazy_full_energy_derivative_preserving_original_results(
    tmp_path, backend, constrained
):
    calculator = _polar_calculator(tmp_path)
    atoms = Atoms(
        "H2",
        positions=[[0.1, 0.2, 0.3], [1.1, 0.5, 0.7]],
        cell=[[5, 0.2, 0.1], [0.3, 4.7, 0.4], [0.2, 0.5, 4.5]],
        pbc=True,
    )
    atoms.info.update(charge=0, spin=1)
    if constrained:
        atoms.set_constraint(Hookean(0, 1, rt=0.1, k=10.0))
    original = atoms.copy()
    energy = calculator.get_potential_energy(atoms)
    forces = calculator.get_forces(atoms)
    assert "stress" not in calculator.results
    assert calculator.inner.calculations == 1
    tensor = calculator.inner.scale * atoms.positions.T @ atoms.positions / atoms.get_volume()
    np.testing.assert_allclose(
        calculator.get_stress(atoms), full_3x3_to_voigt_6_stress(tensor), rtol=0, atol=1e-10
    )
    assert calculator.inner.calculations == 13
    assert calculator.get_potential_energy(atoms) == energy
    np.testing.assert_array_equal(calculator.get_forces(atoms), forces)
    np.testing.assert_array_equal(calculator.atoms.positions, original.positions)
    np.testing.assert_array_equal(atoms.positions, original.positions)
    np.testing.assert_array_equal(atoms.cell, original.cell)
    assert calculator.inner.calculations == 13


def test_failed_polar_stress_probe_leaves_original_atoms_and_no_partial_results(
    tmp_path, backend, monkeypatch
):
    calculator = _polar_calculator(tmp_path)
    atoms = _atoms()
    atoms.info.update(charge=0, spin=1)
    original = atoms.copy()
    expected = calculator.get_potential_energy(atoms)

    def failed_probe(probe, **kwargs):
        probe.set_cell(probe.cell * 1.2, scale_atoms=True)
        probe.get_potential_energy()
        raise RuntimeError("failed strained evaluation")

    monkeypatch.setattr(mace, "calculate_numerical_stress", failed_probe)
    with pytest.raises(RuntimeError, match="failed strained"):
        calculator.get_stress(atoms)
    assert calculator.results == {}
    np.testing.assert_array_equal(atoms.positions, original.positions)
    np.testing.assert_array_equal(atoms.cell, original.cell)
    assert calculator.get_potential_energy(atoms) == expected


@pytest.mark.parametrize("family", [None, "mp", "off", "polar"])
def test_local_checkpoint_architecture_cannot_bypass_polar_guards(tmp_path, backend, family):
    class MislabeledCalculator(FakeMACECalculator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.models = [object() if family == "polar" else FakePolarModel()]

        def calculate(self, *args, **kwargs):
            pytest.fail("a mismatched model must be rejected before inference")

    backend.calculators.MACECalculator = MislabeledCalculator
    options = {"checkpoint": str(_checkpoint(tmp_path)), "device": "cpu"}
    if family is not None:
        options["family"] = family
    adapter = mace.create_adapter(
        trajectory=_trajectory(pressure=None if family == "polar" else 0.0), options=options
    )
    with pytest.raises(ValueError, match="checkpoint architecture does not match family"):
        adapter._new_calculator()


def test_catalog_is_available_without_optional_packages_and_reports_actual_mh1_heads(monkeypatch):
    def forbidden(*args):
        pytest.fail("catalog must not inspect an optional package")

    monkeypatch.setattr(mace, "require_package", forbidden)
    listing = json.loads(json.dumps(mace.catalog()))
    heads = {
        "matpes_r2scan",
        "mp_pbe_refit_add",
        "spice_wB97M",
        "oc20_usemppbe",
        "omol",
        "omat_pbe",
    }
    assert set(listing["heads"]) == heads
    mh = next(item for item in listing["models"] if item["name"] == "mh-1")
    assert set(mh["heads"]) == heads
    assert {item["name"] for item in listing["models"] if item["family"] == "polar"} == {
        "polar-1-s",
        "polar-1-m",
        "polar-1-l",
    }
