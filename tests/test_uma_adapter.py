from __future__ import annotations

import ast
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

from reactionflow.adapters import _torch, uma
from reactionflow.campaign import TrajectorySpec


@pytest.fixture
def package(monkeypatch, tmp_path):
    registry = tmp_path / "pretrained_models.json"
    registry.write_text(
        json.dumps(
            {
                "uma-new-registry-entry": {
                    "repo_id": "facebook/UMA",
                    "filename": "weights.pt",
                    "subfolder": "checkpoints",
                    "atom_refs": {"filename": "atoms.yaml", "subfolder": "references"},
                    "form_elem_refs": {"filename": "formation.yaml", "subfolder": "references"},
                },
                "different-family": {"repo_id": "facebook/OMol25", "filename": "other.pt"},
            }
        )
    )
    monkeypatch.setattr(uma, "require_package", lambda *args: None)
    monkeypatch.setattr(
        uma, "distribution", lambda name: SimpleNamespace(locate_file=lambda name: registry)
    )
    return registry


@pytest.fixture
def hub(monkeypatch, tmp_path, package):
    class HTTPError(Exception):
        def __init__(self, status=403):
            self.response = SimpleNamespace(status_code=status)

    class MissingFile(HTTPError):
        pass

    calls = []
    cache = tmp_path / "cache" / "uma"
    current_revision = "a" * 40
    state = SimpleNamespace(error=None, missing=None)

    def download(**kwargs):
        nonlocal current_revision
        calls.append(kwargs)
        if state.error:
            raise state.error
        if kwargs["filename"] == state.missing:
            raise MissingFile()
        revision = kwargs["revision"]
        if revision is None or revision == "release-branch":
            revision = current_revision
            # Simulate the branch advancing between checkpoint and reference downloads.
            current_revision = "b" * 40
        snapshot = cache / "models--facebook--UMA" / "snapshots" / revision
        path = snapshot / kwargs["subfolder"] / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        # Real HF snapshot entries can be symlinks into blobs. Resolve the revision first.
        blob = cache / "models--facebook--UMA" / "blobs" / (revision + kwargs["filename"])
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_text(kwargs["filename"])
        if not path.exists():
            path.symlink_to(blob)
        return str(path)

    module = ModuleType("huggingface_hub")
    module.hf_hub_download = download
    errors = ModuleType("huggingface_hub.errors")
    errors.HfHubHTTPError = HTTPError
    errors.LocalEntryNotFoundError = MissingFile
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    return SimpleNamespace(calls=calls, state=state, HTTPError=HTTPError, cache=cache)


def _named_options(tmp_path, **overrides):
    return {
        "model": "uma-new-registry-entry",
        "task": "omol",
        "device": "cpu",
        "cache_dir": str(tmp_path / "cache"),
        **overrides,
    }


@pytest.mark.parametrize("download", [False, True])
def test_registry_downloads_one_revision_without_loading_predictor(tmp_path, hub, download):
    files = uma.prepare(_named_options(tmp_path, revision="release-branch"), download=download)

    assert set(files) == {"checkpoint", "atom_refs", "form_elem_refs"}
    assert all(path.is_file() for path in files.values())
    assert [call["revision"] for call in hub.calls] == ["release-branch", "a" * 40, "a" * 40]
    assert all(call["local_files_only"] is not download for call in hub.calls)
    assert all(call["cache_dir"] == str(hub.cache) for call in hub.calls)
    assert all("a" * 40 in path.name for path in files.values())


def test_offline_missing_reference_fails_instead_of_using_another_revision(tmp_path, hub):
    hub.state.missing = "atoms.yaml"
    with pytest.raises(FileNotFoundError, match="prepare this model"):
        uma.prepare(_named_options(tmp_path), download=False)
    assert [call["revision"] for call in hub.calls] == [None, "a" * 40]
    assert all(call["local_files_only"] for call in hub.calls)


def test_gated_download_explains_user_authentication(tmp_path, hub):
    hub.state.error = hub.HTTPError(403)
    with pytest.raises(RuntimeError, match="hf auth login"):
        uma.prepare(_named_options(tmp_path))
    assert len(hub.calls) == 1


def test_preparation_rejects_other_families_and_unpinned_cache_paths(tmp_path, hub, monkeypatch):
    with pytest.raises(ValueError, match="unknown UMA model"):
        uma.prepare(_named_options(tmp_path, model="different-family"))
    assert hub.calls == []
    checkpoint = tmp_path / "loose.pt"
    checkpoint.write_bytes(b"weights")
    monkeypatch.setattr(
        sys.modules["huggingface_hub"], "hf_hub_download", lambda **kwargs: str(checkpoint)
    )
    with pytest.raises(ValueError, match="cache snapshot"):
        uma.prepare(_named_options(tmp_path))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"HF_TOKEN": "must-not-be-serialized"}, "unknown UMA"),
        ({"checkpoint": "/also-selected.pt"}, "exactly one"),
        ({"task": ""}, "explicit task"),
        ({"device": "mps"}, "device"),
        ({"cache_dir": "relative"}, "absolute path"),
        ({"revision": None}, "revision"),
        ({"atom_refs": "/manual.yaml"}, "only valid with checkpoint"),
    ],
)
def test_uma_rejects_ambiguous_or_hidden_options(tmp_path, package, changes, message):
    with pytest.raises(ValueError, match=message):
        uma.prepare(_named_options(tmp_path, **changes))


def _local_options(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    atom_refs = tmp_path / "atom_refs.yaml"
    atom_refs.write_text(repr({"omol_elem_refs": {1: {0: -13.6}}}))
    form_refs = tmp_path / "form_refs.yaml"
    form_refs.write_text(repr({"refs": {"omat": {"H": -3.0}}}))
    return {
        "checkpoint": str(checkpoint),
        "atom_refs": str(atom_refs),
        "form_elem_refs": str(form_refs),
        "task": "omol",
        "device": "cpu",
    }


def test_local_paths_need_no_registry_or_hub_and_must_exist(tmp_path, package, monkeypatch):
    options = _local_options(tmp_path)

    def no_registry():
        raise AssertionError("local paths must not inspect the model registry")

    monkeypatch.setattr(uma, "_registry", no_registry)
    files = uma.prepare(options, download=False)
    assert set(files) == {"checkpoint", "atom_refs", "form_elem_refs"}
    Path(options["atom_refs"]).unlink()
    with pytest.raises(FileNotFoundError, match="atom_refs"):
        uma.prepare(options, download=False)
    with pytest.raises(ValueError, match="absolute path"):
        uma.prepare({**options, "checkpoint": "relative.pt"})


class FakeCalculator(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

    def __init__(self, predictor, *, task_name):
        super().__init__()
        if task_name not in predictor.dataset_to_tasks:
            raise ValueError("Invalid task_name")
        self.predictor = predictor
        self.task_name = task_name

    def check_state(self, atoms, tol=1e-15):
        state = super().check_state(atoms, tol=tol)
        if not state and self.atoms.info != atoms.info:
            state.append("info")
        return state

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": float(0.5 * np.sum(self.atoms.positions**2)),
            "forces": -self.atoms.positions.copy(),
            "stress": np.zeros(6),
        }


@pytest.fixture
def backend(monkeypatch, package):
    loaded = []

    def load(path, **kwargs):
        loaded.append((path, kwargs))
        return SimpleNamespace(
            **kwargs,
            dataset_to_tasks={
                task: []
                for task in ("omol", "omat", "omc", "odac", "oc20", "oc22", "oc25", "custom")
            },
        )

    modules = {
        name: ModuleType(name)
        for name in (
            "fairchem",
            "fairchem.core",
            "fairchem.core.units",
            "fairchem.core.units.mlip_unit",
            "fairchem.core.units.mlip_unit.api",
            "fairchem.core.units.mlip_unit.api.inference",
            "omegaconf",
        )
    }
    modules["fairchem.core"].FAIRChemCalculator = FakeCalculator
    modules["fairchem.core.units.mlip_unit"].load_predict_unit = load
    modules["fairchem.core.units.mlip_unit.api.inference"].InferenceSettings = SimpleNamespace
    modules["omegaconf"].OmegaConf = SimpleNamespace(
        load=lambda path: ast.literal_eval(Path(path).read_text())
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(_torch, "_load_torch", lambda device: None)
    monkeypatch.setattr(_torch, "_torch_environment", lambda torch, device: {"device": device})
    monkeypatch.setattr(_torch, "_preserve_rng", lambda torch: nullcontext())
    monkeypatch.setattr(_torch, "version", lambda name: "test-version")
    return loaded


def _trajectory():
    return TrajectorySpec(
        id="uma",
        total_steps=4,
        timestep_fs=0.25,
        temperature_K=300,
        pressure_GPa=None,
        seed=123,
    )


def _atoms():
    atoms = Atoms("H2", positions=[[0.2, 0.1, 0.3], [1.1, 0.4, 0.2]])
    atoms.info.update(charge=0, spin=1)
    return atoms


def test_local_calculator_uses_fixed_settings_and_preserves_reference_integer_keys(
    tmp_path, backend
):
    options = _local_options(tmp_path)
    adapter = uma.create_adapter(trajectory=_trajectory(), options=options)
    with adapter.calculator("neb") as calculator:
        assert calculator.task_name == "omol"
        assert np.isfinite(calculator.get_potential_energy(_atoms()))
    path, kwargs = backend[0]
    assert path == Path(options["checkpoint"])
    assert kwargs["atom_refs"]["omol_elem_refs"][1][0] == -13.6
    assert kwargs["form_elem_refs"] == {"omat": {"H": -3.0}}
    assert kwargs["seed"] == 0 and kwargs["workers"] == 1 and kwargs["device"] == "cpu"
    assert vars(kwargs["inference_settings"]) == {
        "merge_mole": False,
        "compile": False,
        "tf32": False,
        "execution_mode": "general",
        "base_precision_dtype": "float32",
        "activation_checkpointing": False,
        "auto_add_default_untrained_tasks": True,
    }


@pytest.mark.parametrize(
    "info", [{}, {"charge": 0}, {"charge": False, "spin": 1}, {"charge": 0, "spin": 0}]
)
def test_omol_requires_explicit_charge_and_positive_spin(tmp_path, backend, info):
    adapter = uma.create_adapter(trajectory=_trajectory(), options=_local_options(tmp_path))
    with adapter.calculator("neb") as calculator:
        atoms = _atoms()
        calculator.get_potential_energy(atoms)
        atoms.info = info
        with pytest.raises(ValueError, match="UMA omol"):
            calculator.get_potential_energy(atoms)


def test_task_is_explicit_and_nonmolecular_task_needs_no_omol_metadata(tmp_path, backend):
    options = {**_local_options(tmp_path), "task": "omat"}
    adapter = uma.create_adapter(trajectory=_trajectory(), options=options)
    with adapter.calculator("neb") as calculator:
        atoms = _atoms()
        atoms.info = {}
        assert np.isfinite(calculator.get_potential_energy(atoms))
    invalid = uma.create_adapter(trajectory=_trajectory(), options={**options, "task": "unknown"})
    with pytest.raises(ValueError, match="Invalid task_name"), invalid.calculator("neb"):
        pass


def test_fake_uma_restart_matches_continuation_and_reference_change_is_rejected(tmp_path, backend):
    options = _local_options(tmp_path)
    adapter = uma.create_adapter(trajectory=_trajectory(), options=options)
    with adapter.start(_atoms()) as full:
        full.run(4)
        expected = full.snapshot()
    with adapter.start(_atoms()) as split:
        split.run(2)
        checkpoint = split.snapshot()
    reopened = uma.create_adapter(trajectory=_trajectory(), options=options)
    with reopened.restore(checkpoint) as continuation:
        continuation.run(2)
        actual = continuation.snapshot()
    for key in expected.atoms.arrays:
        np.testing.assert_array_equal(actual.atoms.arrays[key], expected.atoms.arrays[key])
    assert actual.calculator == expected.calculator
    Path(options["atom_refs"]).write_text(repr({"omol_elem_refs": {1: {0: -14.0}}}))
    changed = uma.create_adapter(trajectory=_trajectory(), options=options)
    with pytest.raises(ValueError, match="environment differs"), changed.restore(checkpoint):
        pass


@pytest.mark.parametrize("task", ["omol", "omat", "omc", "odac", "oc20", "oc22", "oc25"])
def test_all_uma_tasks_are_passed_as_strings_to_checkpoint_validation(tmp_path, backend, task):
    adapter = uma.create_adapter(
        trajectory=_trajectory(), options={**_local_options(tmp_path), "task": task}
    )
    with adapter.calculator("neb") as calculator:
        assert calculator.task_name == task
        assert np.isfinite(calculator.get_potential_energy(_atoms()))
    assert adapter._contract.metadata["options"]["task"] == task


def test_catalog_lists_seven_tasks_and_model_availability_without_backend_imports(monkeypatch):
    def forbidden(*args):
        pytest.fail("catalog must not inspect an optional package or download weights")

    monkeypatch.setattr(uma, "require_package", forbidden)
    monkeypatch.setattr(uma, "_registry", forbidden)
    listing = json.loads(json.dumps(uma.catalog()))
    all_tasks = {"omol", "omat", "omc", "odac", "oc20", "oc22", "oc25"}
    assert set(listing["tasks"]) == all_tasks
    models = {model["name"]: set(model["tasks"]) for model in listing["models"]}
    assert models["uma-s-1p2p1"] == models["uma-s-1p2"] == all_tasks
    assert models["uma-s-1p1"] == models["uma-m-1p1"] == all_tasks - {"oc22", "oc25"}


@pytest.mark.parametrize("model", ["uma-s-1p1", "uma-m-1p1"])
@pytest.mark.parametrize("task", ["oc22", "oc25"])
def test_older_named_models_reject_untrained_tasks_before_setup(monkeypatch, model, task):
    def forbidden(*args):
        pytest.fail("invalid model/task must fail before package checks or downloads")

    monkeypatch.setattr(uma, "require_package", forbidden)
    with pytest.raises(ValueError, match="was not trained"):
        uma.prepare({"model": model, "task": task})


def test_named_task_typo_fails_early_but_custom_checkpoint_tasks_are_allowed(
    tmp_path, backend, monkeypatch
):
    def forbidden(*args):
        pytest.fail("invalid named task must fail before package checks or downloads")

    with monkeypatch.context() as context:
        context.setattr(uma, "require_package", forbidden)
        with pytest.raises(ValueError, match="unknown named UMA task"):
            uma.prepare({"model": "uma-s-1p2p1", "task": "omoll"})
    adapter = uma.create_adapter(
        trajectory=_trajectory(), options={**_local_options(tmp_path), "task": "custom"}
    )
    with adapter.calculator("neb") as calculator:
        assert calculator.task_name == "custom"


@pytest.mark.parametrize(
    ("model", "task"),
    [
        ("uma-s-1p2p1", "oc22"),
        ("uma-s-1p2p1", "oc25"),
        ("uma-s-1p2", "oc22"),
        ("uma-s-1p2", "oc25"),
        ("uma-s-1p1", "omat"),
        ("uma-m-1p1", "omol"),
    ],
)
def test_named_model_task_matrix_prepares_one_revision(tmp_path, package, hub, model, task):
    registry = json.loads(package.read_text())
    registry[model] = registry["uma-new-registry-entry"]
    package.write_text(json.dumps(registry))
    files = uma.prepare(_named_options(tmp_path, model=model, task=task))
    assert set(files) == {"checkpoint", "atom_refs", "form_elem_refs"}
    assert [call["revision"] for call in hub.calls] == [None, "a" * 40, "a" * 40]
