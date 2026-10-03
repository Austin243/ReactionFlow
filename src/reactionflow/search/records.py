"""Immutable, checksummed JSON records on a single-writer POSIX filesystem."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io.jsonio import decode, encode

from .._durable import ensure_directory, publish, sync_directory
from ..detection import assign_atom_ids, atom_ids

_KINDS = {"states", "processes", "attempts", "steps"}
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _name(value: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError("record IDs must be safe non-empty names")
    return value


def _json(value: object) -> str:
    def check(item: object) -> None:
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise TypeError("JSON object keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif item is not None and not isinstance(item, (str, bool, int, float)):
            raise TypeError("record data must contain only JSON values")

    check(value)
    return json.dumps(value, allow_nan=False, sort_keys=True, indent=2) + "\n"


def _array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biufcSU":
        raise ValueError("record arrays must have numeric, boolean, or string dtype")
    if array.dtype.kind in "iufc" and not np.isfinite(array).all():
        raise ValueError("record arrays must be finite")
    return array


@dataclass(frozen=True)
class SearchRecord:
    """Loaded data and independent calculator-free structure/array copies."""

    data: dict
    structures: dict[str, Atoms]
    arrays: dict[str, np.ndarray]


class SearchStore:
    """One atomic JSON file per record; a runner owns the single-writer lock."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def _path(self, kind: str, record_id: str | None = None) -> Path:
        if kind not in _KINDS:
            raise ValueError(f"unsupported search record kind: {kind!r}")
        directory = self.root / kind
        return directory if record_id is None else directory / f"{_name(record_id)}.json"

    def ids(self, kind: str) -> tuple[str, ...]:
        """List published record IDs in lexical order, ignoring interrupted temporary files."""

        return tuple(
            sorted(
                path.stem
                for path in self._path(kind).glob("*.json")
                if path.is_file() and _NAME.fullmatch(path.stem)
            )
        )

    def write(
        self,
        kind: str,
        record_id: str,
        data: dict,
        *,
        structures: dict[str, Atoms] | None = None,
        arrays: dict[str, np.ndarray] | None = None,
    ) -> SearchRecord:
        """Save a new record, or return an identical existing one; reject conflicts."""

        if not isinstance(data, dict):
            raise TypeError("record data must be a JSON object")
        final = self._path(kind, record_id)
        encoded_structures = {}
        for name, source in (structures or {}).items():
            atoms = assign_atom_ids(source.copy())
            atoms.calc = None
            for array in atoms.arrays.values():
                _array(array)
            # ASE JSON preserves custom arrays, info, cell displacement, and constraints.
            encoded_structures[name] = json.loads(encode(atoms))
        payload = {
            "schema_version": 1,
            "kind": kind,
            "record_id": record_id,
            "data": data,
            "structures": encoded_structures,
            "arrays": {
                name: json.loads(encode(_array(array))) for name, array in (arrays or {}).items()
            },
        }
        document = {"payload": payload, "sha256": sha256(_json(payload).encode()).hexdigest()}
        if final.exists():
            existing = self.read(kind, record_id)
            if json.loads(final.read_text(encoding="utf-8")) != document:
                raise ValueError(f"search record {kind}/{record_id} has conflicting data")
            sync_directory(final.parent)
            return existing
        ensure_directory(final.parent)
        temporary = final.with_name(f".{final.name}.tmp")
        try:
            temporary.write_text(_json(document), encoding="utf-8")
            publish(temporary, final)
        finally:
            temporary.unlink(missing_ok=True)
        return self.read(kind, record_id)

    def read(self, kind: str, record_id: str) -> SearchRecord:
        """Verify the complete payload before decoding ASE structures and arrays."""

        document = json.loads(self._path(kind, record_id).read_text(encoding="utf-8"))
        payload = document["payload"]
        if sha256(_json(payload).encode()).hexdigest() != document["sha256"]:
            raise ValueError(f"search record failed integrity check: {kind}/{record_id}")
        if (payload.get("schema_version"), payload.get("kind"), payload.get("record_id")) != (
            1,
            kind,
            record_id,
        ):
            raise ValueError("unsupported or mismatched search record")
        if not isinstance(payload["data"], dict):
            raise ValueError("record data must be a JSON object")
        structures = {}
        for name, value in payload["structures"].items():
            atoms = decode(json.dumps(value))
            if not isinstance(atoms, Atoms):
                raise ValueError("record structure is not ASE Atoms")
            atom_ids(atoms)
            structures[name] = atoms
        arrays = {
            name: _array(decode(json.dumps(value))) for name, value in payload["arrays"].items()
        }
        return SearchRecord(payload["data"], structures, arrays)
