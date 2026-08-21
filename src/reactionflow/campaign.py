"""Validated, scheduler-neutral trajectory campaign configuration."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from .run import ReactionRunConfig

_TRAJECTORY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be an object with string keys")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_float(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        qualifier = f" greater than or equal to {minimum}" if minimum is not None else " finite"
        raise ValueError(f"{name} must be{qualifier}")
    return result


def _json_mapping(value: object, name: str) -> dict[str, Any]:
    mapping = dict(_mapping(value, name))
    try:
        return json.loads(json.dumps(mapping, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain only finite JSON values") from error


@dataclass(frozen=True, slots=True)
class AdapterSpec:
    """Importable MLIP adapter factory and campaign-wide options."""

    factory: str
    options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: object) -> AdapterSpec:
        mapping = _mapping(value, "adapter")
        unknown = set(mapping) - {"factory", "options"}
        if unknown:
            raise ValueError(f"unknown adapter keys: {sorted(unknown)}")
        factory = mapping.get("factory")
        if (
            not isinstance(factory, str)
            or factory.count(":") != 1
            or any(not part for part in factory.split(":"))
        ):
            raise ValueError("adapter.factory must be 'module:callable'")
        return cls(
            factory=factory,
            options=_json_mapping(mapping.get("options", {}), "adapter.options"),
        )


@dataclass(frozen=True, slots=True)
class TrajectorySpec:
    """Conditions and deterministic identity for one independent trajectory."""

    id: str
    total_steps: int
    timestep_fs: float
    temperature_K: float
    pressure_GPa: float | None
    seed: int
    conditions: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: object) -> TrajectorySpec:
        mapping = _mapping(value, "trajectory")
        allowed = {
            "id",
            "total_steps",
            "timestep_fs",
            "temperature_K",
            "pressure_GPa",
            "seed",
            "conditions",
        }
        unknown = set(mapping) - allowed
        if unknown:
            raise ValueError(f"unknown trajectory keys: {sorted(unknown)}")
        trajectory_id = mapping.get("id")
        if (
            not isinstance(trajectory_id, str)
            or not _TRAJECTORY_ID.fullmatch(trajectory_id)
            or trajectory_id in {".", ".."}
        ):
            raise ValueError("trajectory.id must be a safe non-empty file name")
        seed = mapping.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**64:
            raise ValueError("trajectory.seed must be an integer in [0, 2**64)")
        pressure = mapping.get("pressure_GPa")
        timestep_fs = _finite_float(
            mapping.get("timestep_fs"),
            "trajectory.timestep_fs",
            minimum=0.0,
        )
        if timestep_fs == 0:
            raise ValueError("trajectory.timestep_fs must be positive")
        return cls(
            id=trajectory_id,
            total_steps=_positive_int(mapping.get("total_steps"), "trajectory.total_steps"),
            timestep_fs=timestep_fs,
            temperature_K=_finite_float(
                mapping.get("temperature_K"),
                "trajectory.temperature_K",
                minimum=0.0,
            ),
            pressure_GPa=(
                None if pressure is None else _finite_float(pressure, "trajectory.pressure_GPa")
            ),
            seed=int(seed),
            conditions=_json_mapping(mapping.get("conditions", {}), "trajectory.conditions"),
        )


def _run_config(value: object) -> ReactionRunConfig:
    overrides = _mapping(value, "reaction_run")
    merged = ReactionRunConfig().to_dict()
    unknown = set(overrides) - set(merged)
    if unknown:
        raise ValueError(f"unknown reaction_run keys: {sorted(unknown)}")
    for key, item in overrides.items():
        if key in {"detector", "pathway"}:
            nested = _mapping(item, f"reaction_run.{key}")
            unknown_nested = set(nested) - set(merged[key])  # type: ignore[arg-type]
            if unknown_nested:
                raise ValueError(f"unknown reaction_run.{key} keys: {sorted(unknown_nested)}")
            merged[key] = {**merged[key], **nested}  # type: ignore[dict-item]
        else:
            merged[key] = item
    return ReactionRunConfig.from_dict(merged)


@dataclass(frozen=True, slots=True)
class CampaignConfig:
    """One structure, one adapter, and independently parameterized trajectories."""

    source: Path
    structure: Path
    output_root: Path
    adapter: AdapterSpec
    reaction_run: ReactionRunConfig
    trajectories: tuple[TrajectorySpec, ...]
    require_gpu: bool = True

    @classmethod
    def load(cls, path: str | Path) -> CampaignConfig:
        source = Path(path).resolve()
        value = _mapping(json.loads(source.read_text(encoding="utf-8")), "campaign")
        allowed = {
            "schema_version",
            "structure",
            "output_root",
            "adapter",
            "reaction_run",
            "trajectories",
            "require_gpu",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown campaign keys: {sorted(unknown)}")
        if value.get("schema_version") != 1:
            raise ValueError("unsupported campaign schema")
        structure_value = value.get("structure")
        output_value = value.get("output_root")
        if not isinstance(structure_value, str) or not structure_value:
            raise ValueError("campaign.structure must be a path string")
        if not isinstance(output_value, str) or not output_value:
            raise ValueError("campaign.output_root must be a path string")
        structure = (source.parent / structure_value).resolve()
        if not structure.is_file():
            raise FileNotFoundError(structure)
        raw_trajectories = value.get("trajectories")
        if (
            not isinstance(raw_trajectories, Sequence)
            or isinstance(raw_trajectories, (str, bytes))
            or not raw_trajectories
        ):
            raise ValueError("campaign.trajectories must be a non-empty array")
        trajectories = tuple(TrajectorySpec.from_dict(item) for item in raw_trajectories)
        identifiers = [trajectory.id for trajectory in trajectories]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("trajectory IDs must be unique")
        require_gpu = value.get("require_gpu", True)
        if type(require_gpu) is not bool:
            raise TypeError("campaign.require_gpu must be a boolean")
        return cls(
            source=source,
            structure=structure,
            output_root=(source.parent / output_value).resolve(),
            adapter=AdapterSpec.from_dict(value.get("adapter")),
            reaction_run=_run_config(value.get("reaction_run", {})),
            trajectories=trajectories,
            require_gpu=require_gpu,
        )

    def trajectory(self, index: int) -> TrajectorySpec:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("trajectory index must be an integer")
        if not 0 <= index < len(self.trajectories):
            raise IndexError(f"trajectory index {index} is outside this campaign")
        return self.trajectories[index]


__all__ = ["AdapterSpec", "CampaignConfig", "TrajectorySpec"]
