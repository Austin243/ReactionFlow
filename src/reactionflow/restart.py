"""Portable, explicit state for exact trajectory restarts."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from ase import Atoms
from ase.io import read, write

_TYPE_MARKER = "__reactionflow_type__"


def _digest(path: Path) -> str:
    checksum = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _encode_json(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("restart metadata cannot contain non-finite floats")
        return value
    if isinstance(value, np.generic):
        return _encode_json(value.item())
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("restart metadata cannot contain object arrays")
        return {
            _TYPE_MARKER: "ndarray",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": value.tolist(),
        }
    if isinstance(value, tuple):
        return {_TYPE_MARKER: "tuple", "items": [_encode_json(item) for item in value]}
    if isinstance(value, list):
        return [_encode_json(item) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("restart metadata keys must be strings")
            if key == _TYPE_MARKER:
                raise ValueError(f"{_TYPE_MARKER!r} is reserved in restart metadata")
            result[key] = _encode_json(item)
        return result
    raise TypeError(f"unsupported restart metadata value {type(value).__name__}")


def _decode_json(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_json(item) for item in value]
    if not isinstance(value, dict):
        return value
    marker = value.get(_TYPE_MARKER)
    if marker == "tuple":
        return tuple(_decode_json(item) for item in value["items"])
    if marker == "ndarray":
        array = np.asarray(value["data"], dtype=np.dtype(value["dtype"]))
        return array.reshape(tuple(map(int, value["shape"])))
    if marker is not None:
        raise ValueError(f"unsupported restart metadata marker {marker!r}")
    return {key: _decode_json(item) for key, item in value.items()}


@dataclass(frozen=True, slots=True)
class ComponentState:
    """Versioned exact state for one runtime component."""

    kind: str
    version: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)
    arrays: Mapping[str, np.ndarray] = field(default_factory=dict)
    exact: bool = True

    def __post_init__(self) -> None:
        if not self.kind or not isinstance(self.kind, str):
            raise ValueError("component kind must be a non-empty string")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("component version must be a positive integer")
        if type(self.exact) is not bool:
            raise TypeError("component exact flag must be a boolean")
        if not isinstance(self.metadata, Mapping) or not isinstance(self.arrays, Mapping):
            raise TypeError("component metadata and arrays must be mappings")
        _encode_json(self.metadata)
        copied: dict[str, np.ndarray] = {}
        for name, value in self.arrays.items():
            if not isinstance(name, str) or not name:
                raise ValueError("component array names must be non-empty strings")
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise TypeError("component arrays cannot use object dtype")
            copied[name] = array.copy()
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "arrays", copied)


@dataclass(frozen=True, slots=True)
class ExactRestartSnapshot:
    """Calculator-free atoms plus exact dynamics and calculator state."""

    atoms: Atoms
    dynamics: ComponentState
    calculator: ComponentState

    def __post_init__(self) -> None:
        if not self.dynamics.exact:
            raise ValueError("dynamics component does not provide exact restart state")
        if not self.calculator.exact:
            raise ValueError("calculator component does not provide exact restart state")
        atoms = self.atoms.copy()
        atoms.calc = None
        object.__setattr__(self, "atoms", atoms)

    def write(self, path: str | Path) -> Path:
        """Atomically publish a new restart directory."""

        final = Path(path).resolve()
        if final.exists():
            raise FileExistsError(final)
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary = final.parent / f".{final.name}-{uuid4().hex}.tmp"
        temporary.mkdir()
        try:
            atoms_path = temporary / "atoms.traj"
            arrays_path = temporary / "arrays.npz"
            state_path = temporary / "restart.json"
            write(atoms_path, self.atoms, format="traj")

            archive: dict[str, np.ndarray] = {}
            components: dict[str, dict[str, Any]] = {}
            for label, component in (
                ("dynamics", self.dynamics),
                ("calculator", self.calculator),
            ):
                names: dict[str, str] = {}
                for index, (name, array) in enumerate(sorted(component.arrays.items())):
                    archive_name = f"{label}_{index:04d}"
                    archive[archive_name] = array
                    names[name] = archive_name
                components[label] = {
                    "kind": component.kind,
                    "version": component.version,
                    "exact": component.exact,
                    "metadata": _encode_json(component.metadata),
                    "arrays": names,
                }
            np.savez_compressed(arrays_path, **archive)
            manifest = {
                "schema_version": 1,
                "components": components,
                "files": {
                    "atoms.traj": _digest(atoms_path),
                    "arrays.npz": _digest(arrays_path),
                },
            }
            state_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, final)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return final

    @classmethod
    def read(cls, path: str | Path) -> ExactRestartSnapshot:
        """Load and integrity-check a version-1 restart directory."""

        root = Path(path).resolve()
        manifest = json.loads((root / "restart.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported exact-restart schema")
        files = manifest.get("files")
        if not isinstance(files, dict) or set(files) != {"atoms.traj", "arrays.npz"}:
            raise ValueError("exact-restart manifest has an invalid file set")
        for name, expected in files.items():
            file_path = root / name
            if not file_path.is_file() or _digest(file_path) != expected:
                raise ValueError(f"restart artifact failed integrity check: {name}")

        atoms = read(root / "atoms.traj")
        components = manifest["components"]
        if not isinstance(components, dict) or set(components) != {"dynamics", "calculator"}:
            raise ValueError("exact-restart manifest has an invalid component set")
        with np.load(root / "arrays.npz", allow_pickle=False) as archive:
            restored: dict[str, ComponentState] = {}
            referenced_arrays: set[str] = set()
            for label in ("dynamics", "calculator"):
                value = components[label]
                referenced_arrays.update(value.get("arrays", {}).values())
                arrays = {
                    name: np.asarray(archive[archive_name]).copy()
                    for name, archive_name in value.get("arrays", {}).items()
                }
                restored[label] = ComponentState(
                    kind=value["kind"],
                    version=value["version"],
                    exact=value["exact"],
                    metadata=_decode_json(value.get("metadata", {})),
                    arrays=arrays,
                )
            if referenced_arrays != set(archive.files):
                raise ValueError("exact-restart array archive does not match its manifest")
        return cls(
            atoms=atoms,
            dynamics=restored["dynamics"],
            calculator=restored["calculator"],
        )


__all__ = ["ComponentState", "ExactRestartSnapshot"]
