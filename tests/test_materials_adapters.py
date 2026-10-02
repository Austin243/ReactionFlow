from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from reactionflow.adapters import chgnet, mattersim
from reactionflow.campaign import TrajectorySpec


@pytest.fixture
def trajectory():
    return TrajectorySpec(
        id="materials", total_steps=2, timestep_fs=0.1, temperature_K=300, pressure_GPa=0, seed=2
    )


@pytest.mark.parametrize("backend", [chgnet, mattersim])
def test_local_checkpoint_is_resolved_without_download(tmp_path, monkeypatch, backend):
    monkeypatch.setattr(backend, "require_package", lambda *args: None)
    path = tmp_path / "weights.pt"
    path.write_bytes(b"model")
    assert backend.prepare({"checkpoint": str(path)}, download=False) == {"checkpoint": path}
    path.unlink()
    with pytest.raises(FileNotFoundError):
        backend.prepare({"checkpoint": str(path)}, download=False)


@pytest.mark.parametrize("backend", [chgnet, mattersim])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"model": "unknown"},
        {"model": "unknown", "checkpoint": "/tmp/model"},
        {"checkpoint": "relative.pt"},
        {"checkpoint": "/tmp/model", "device": "mps"},
        {"checkpoint": "/tmp/model", "typo": True},
    ],
)
def test_bad_configuration_fails_before_loading_backend(monkeypatch, backend, options):
    def unexpected(*args):
        raise AssertionError("invalid configuration must fail before package setup")

    monkeypatch.setattr(backend, "require_package", unexpected)
    with pytest.raises(ValueError):
        backend.prepare(options)


def test_mattersim_download_uses_pinned_release_and_respects_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(mattersim, "require_package", lambda *args: None)
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"weights")
    download = Mock(return_value=checkpoint)
    monkeypatch.setattr(mattersim, "cached_download", download)
    result = mattersim.prepare(
        {"model": "mattersim-v1.0.0-5M", "cache_dir": str(tmp_path)}, download=False
    )
    assert result == {"checkpoint": checkpoint}
    assert download.call_args.args == (
        "https://raw.githubusercontent.com/microsoft/mattersim/"
        "40a1eb8f1189a53af310957b4f2c5dfbfe68d647/pretrained_models/mattersim-v1.0.0-5M.pth",
        tmp_path / "mattersim",
    )
    assert download.call_args.kwargs == {"download": False}


@pytest.mark.parametrize("name", ["0.3.0", "r2scan"])
def test_chgnet_resolves_bundled_weights_without_importing_backend(tmp_path, monkeypatch, name):
    monkeypatch.setattr(chgnet, "require_package", lambda *args: None)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    locate = Mock(return_value=checkpoint)
    monkeypatch.setattr(chgnet, "distribution", lambda name: SimpleNamespace(locate_file=locate))
    assert chgnet.prepare({"model": name}, download=False) == {"checkpoint": checkpoint}
    assert locate.call_args.args[0] == f"chgnet/pretrained/{chgnet._MODELS[name]}"


@pytest.mark.parametrize("backend", [chgnet, mattersim])
def test_calculator_uses_hashed_path_and_rejects_nonperiodic_inputs(
    tmp_path, monkeypatch, trajectory, backend
):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"weights")
    monkeypatch.setattr(backend, "require_package", lambda *args: None)
    monkeypatch.setattr(backend, "prepare", lambda *args, **kwargs: {"checkpoint": checkpoint})
    calls = []

    class FakeCalculator(Calculator):
        def __init__(self, **kwargs):
            super().__init__()
            calls.append(kwargs)

    if backend is chgnet:
        monkeypatch.setitem(
            sys.modules, "torch", SimpleNamespace(float32="float32", set_default_dtype=Mock())
        )
        model = object()
        load = Mock(return_value=model)
        monkeypatch.setitem(
            sys.modules,
            "chgnet.model.model",
            SimpleNamespace(CHGNet=SimpleNamespace(from_file=load)),
        )
        monkeypatch.setitem(
            sys.modules, "chgnet.model.dynamics", SimpleNamespace(CHGNetCalculator=FakeCalculator)
        )
        options = {"model": "r2scan", "device": "cpu"}
    else:
        model = Mock()
        model.to.return_value = model
        model.load_state_dict.return_value = SimpleNamespace(
            missing_keys=["sbf.coef"], unexpected_keys=[]
        )
        load = Mock(
            return_value={
                "model_name": "m3gnet",
                "model_args": {"units": 256},
                "model": {"weight": 1},
            }
        )
        monkeypatch.setitem(
            sys.modules,
            "torch",
            SimpleNamespace(
                load=load, float32="float32", float64="float64", set_default_dtype=Mock()
            ),
        )
        build_model = Mock(return_value=model)
        potential = object()
        build_potential = Mock(return_value=potential)
        monkeypatch.setitem(
            sys.modules, "mattersim.forcefield.m3gnet.m3gnet", SimpleNamespace(M3Gnet=build_model)
        )
        monkeypatch.setitem(
            sys.modules,
            "mattersim.forcefield",
            SimpleNamespace(MatterSimCalculator=FakeCalculator, Potential=build_potential),
        )
        options = {"model": "mattersim-v1.0.0-5M", "dtype": "float64", "device": "cpu"}
    adapter = backend.create_adapter(trajectory=trajectory, options=options)
    calculator = adapter._new_calculator()
    if backend is chgnet:
        load.assert_called_once_with(str(checkpoint), version="r2scan", mlp_out_bias=False)
        assert calls == [
            {
                "model": model,
                "use_device": "cpu",
                "check_cuda_mem": False,
                "on_isolated_atoms": "error",
            }
        ]
    else:
        load.assert_called_once_with(checkpoint, map_location="cpu", weights_only=False)
        build_model.assert_called_once_with(device="cpu", units=256)
        model.load_state_dict.assert_called_once_with({"weight": 1}, strict=False)
        build_potential.assert_called_once_with(
            model, device="cpu", allow_tf32=False, model_name="m3gnet"
        )
        assert calls == [
            {
                "potential": potential,
                "device": "cpu",
                "dtype": "float64",
                "compute_stress": True,
                "compile": False,
                "batch_converter": False,
                "direct_graph": False,
            }
        ]
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], cell=[4, 4, 4], pbc=True)
    adapter.preflight(atoms)
    calculator.check_state(atoms)
    atoms.pbc = False
    with pytest.raises(ValueError, match="fully periodic"):
        adapter.preflight(atoms)
    with pytest.raises(ValueError, match="fully periodic"):
        calculator.check_state(atoms)
    with pytest.raises(ValueError, match="fully periodic"):
        calculator.calculate(atoms)
