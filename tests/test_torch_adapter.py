from __future__ import annotations

import random
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from reactionflow.adapters import _torch as backend
from reactionflow.campaign import TrajectorySpec


class FakeTorch:
    def __init__(self):
        self.__version__ = "test-torch"
        self.state = b"caller torch RNG"
        self.cuda_state = [b"caller cuda RNG"]
        self.dtype = "float32"
        self.num_threads = 1
        self.num_interop_threads = 1
        self.deterministic = True
        self.warn_only = False
        self.precision = "highest"
        self.backends = SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
            cudnn=SimpleNamespace(allow_tf32=False, deterministic=True, benchmark=False),
        )
        self.random = SimpleNamespace(
            get_rng_state=lambda: self.state,
            set_rng_state=lambda state: setattr(self, "state", state),
        )
        self.cuda = SimpleNamespace(
            is_initialized=lambda: True,
            get_rng_state_all=lambda: self.cuda_state.copy(),
            set_rng_state_all=lambda state: setattr(self, "cuda_state", state),
        )

    def get_default_dtype(self):
        return self.dtype

    def set_default_dtype(self, dtype):
        self.dtype = dtype

    def get_num_threads(self):
        return self.num_threads

    def get_num_interop_threads(self):
        return self.num_interop_threads

    def are_deterministic_algorithms_enabled(self):
        return self.deterministic

    def is_deterministic_algorithms_warn_only_enabled(self):
        return self.warn_only

    def get_float32_matmul_precision(self):
        return self.precision

    def use_deterministic_algorithms(self, enabled):
        self.deterministic = enabled
        self.warn_only = False

    def set_float32_matmul_precision(self, precision):
        self.precision = precision


class Harmonic(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": float(np.sum(self.atoms.positions**2) / 2),
            "forces": -self.atoms.positions,
            "stress": np.zeros(6),
        }


class Model(backend.TorchModelAdapter):
    def _new_calculator(self):
        random.seed(0)
        np.random.seed(0)
        return Harmonic()


def _model(path, *, pressure=None):
    trajectory = TrajectorySpec.from_dict(
        {
            "id": "model",
            "total_steps": 6,
            "temperature_K": 200,
            "timestep_fs": 0.2,
            "pressure_GPa": pressure,
            "seed": 19,
        }
    )
    return Model(
        trajectory=trajectory,
        options={"device": "cpu", "model": "test"},
        files={"checkpoint": path},
        packages=("ase",),
        kind="test.model",
    )


@pytest.fixture
def fake_torch(monkeypatch):
    torch = FakeTorch()
    monkeypatch.setattr(backend, "_load_torch", lambda device: torch)
    monkeypatch.setattr(backend, "_torch_environment", lambda torch, device: {"device": device})
    return torch


@pytest.mark.parametrize("pressure", [None, 0.1])
def test_local_model_restart_matches_uninterrupted_dynamics(tmp_path, fake_torch, pressure):
    path = tmp_path / "weights"
    path.write_bytes(b"checkpoint")
    adapter = _model(path, pressure=pressure)
    atoms = Atoms("H2", positions=[[0.2, 0.1, 0.3], [1.1, 0.4, 0.2]], cell=[5, 5, 5], pbc=True)
    with adapter.start(atoms.copy()) as runtime:
        runtime.run(6)
        expected = runtime.snapshot()
    with adapter.start(atoms.copy()) as runtime:
        runtime.run(2)
        snapshot = runtime.snapshot()
    with _model(path, pressure=pressure).restore(snapshot) as runtime:
        runtime.run(4)
        actual = runtime.snapshot()
    assert actual.calculator == expected.calculator
    assert actual.dynamics.metadata == expected.dynamics.metadata
    for name in actual.atoms.arrays:
        np.testing.assert_array_equal(actual.atoms.arrays[name], expected.atoms.arrays[name])
    np.testing.assert_array_equal(actual.atoms.cell, expected.atoms.cell)
    for name in actual.dynamics.arrays:
        np.testing.assert_array_equal(actual.dynamics.arrays[name], expected.dynamics.arrays[name])


def test_changed_weights_rejected_between_leases_and_on_restore(tmp_path, fake_torch):
    path = tmp_path / "weights"
    path.write_bytes(b"checkpoint")
    adapter = _model(path)
    with adapter.start(Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])) as runtime:
        snapshot = runtime.snapshot()
    path.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="changed during"), adapter.calculator("neb"):
        pass
    with pytest.raises(ValueError, match="environment differs"), _model(path).restore(snapshot):
        pass


def test_checkpoint_replacement_during_construction_is_rejected_and_calculator_closed(
    tmp_path, monkeypatch, fake_torch
):
    path = tmp_path / "weights"
    path.write_bytes(b"checkpoint")
    closed = []

    def replacing_constructor(self):
        path.write_bytes(b"new checkpoint loaded by the model")
        calculator = Harmonic()
        calculator.close = lambda: closed.append(True)
        return calculator

    monkeypatch.setattr(Model, "_new_calculator", replacing_constructor)
    with (
        pytest.raises(ValueError, match="changed while the calculator was being constructed"),
        _model(path).calculator("neb"),
    ):
        pytest.fail("a model with a stale file digest must not be leased")
    assert closed == [True]


@pytest.mark.parametrize("fail", [False, True])
def test_model_construction_preserves_all_rng_states_and_default_dtype(fake_torch, fail):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with (
        pytest.raises(RuntimeError) if fail else nullcontext(),
        backend._preserve_rng(fake_torch),
    ):
        random.seed(0)
        np.random.seed(0)
        fake_torch.state = b"changed"
        fake_torch.cuda_state = [b"changed"]
        fake_torch.dtype = "float64"
        if fail:
            raise RuntimeError("constructor failed")
    assert random.getstate() == python_state
    restored = np.random.get_state()
    assert restored[0] == numpy_state[0]
    np.testing.assert_array_equal(restored[1], numpy_state[1])
    assert restored[2:] == numpy_state[2:]
    assert fake_torch.state == b"caller torch RNG"
    assert fake_torch.cuda_state == [b"caller cuda RNG"]
    assert fake_torch.dtype == "float32"


@pytest.mark.parametrize("field", ["num_threads", "num_interop_threads"])
def test_changed_cpu_parallelism_is_rejected_on_restart(tmp_path, monkeypatch, field):
    torch = FakeTorch()
    monkeypatch.setattr(backend, "_load_torch", lambda device: torch)
    path = tmp_path / "weights"
    path.write_bytes(b"checkpoint")
    with _model(path).start(Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])) as runtime:
        snapshot = runtime.snapshot()
    assert snapshot.calculator.metadata[field] == 1
    setattr(torch, field, 2)
    with pytest.raises(ValueError, match="environment differs"), _model(path).restore(snapshot):
        pass


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("deterministic", False),
        ("warn_only", True),
        ("precision", "high"),
        ("backends.cuda.matmul.allow_tf32", True),
        ("backends.cudnn.allow_tf32", True),
        ("backends.cudnn.deterministic", False),
        ("backends.cudnn.benchmark", True),
    ],
)
def test_constructor_cannot_silently_relax_required_inference_settings(
    tmp_path, monkeypatch, field, value
):
    torch = FakeTorch()
    monkeypatch.setattr(backend, "_load_torch", lambda device: torch)

    def changing_constructor(self):
        owner = torch
        *parents, name = field.split(".")
        for parent in parents:
            owner = getattr(owner, parent)
        setattr(owner, name, value)
        return Harmonic()

    monkeypatch.setattr(Model, "_new_calculator", changing_constructor)
    path = tmp_path / "weights"
    path.write_bytes(b"checkpoint")
    with (
        pytest.raises(RuntimeError, match="backend changed required Torch inference settings"),
        _model(path).calculator("neb"),
    ):
        pass


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cuda_is_initialized_before_rng_capture_only_for_cuda_models(monkeypatch, device):
    torch = FakeTorch()
    torch.cuda.initialized = False
    torch.cuda.is_initialized = lambda: torch.cuda.initialized
    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: 1
    torch.cuda.init = lambda: setattr(torch.cuda, "initialized", True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    backend._load_torch(device)

    assert torch.cuda.initialized is (device == "cuda")
    if device == "cuda":
        with backend._preserve_rng(torch):
            torch.cuda_state = [b"constructor seeded CUDA"]
        assert torch.cuda_state == [b"caller cuda RNG"]


@pytest.mark.parametrize("visible_devices", [0, 2])
def test_cuda_requires_one_visible_device_before_initialization(monkeypatch, visible_devices):
    torch = FakeTorch()
    torch.cuda.is_available = lambda: visible_devices > 0
    torch.cuda.device_count = lambda: visible_devices

    def unexpected_initialization():
        pytest.fail("a rejected CUDA configuration must not be initialized")

    torch.cuda.init = unexpected_initialization
    monkeypatch.setitem(sys.modules, "torch", torch)
    with pytest.raises(RuntimeError, match="exactly one usable CUDA"):
        backend._load_torch("cuda")
