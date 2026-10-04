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
    """Importable MLIP adapter factory and its options."""

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
    """Starting structures, named adapter profiles, and independently parameterized trajectories."""

    source: Path
    output_root: Path
    reaction_run: ReactionRunConfig
    trajectories: tuple[TrajectorySpec, ...]
    adapter_profiles: Mapping[str, AdapterSpec]
    trajectory_profiles: tuple[str, ...]
    trajectory_structures: tuple[Path, ...]
    require_gpu: bool = True

    @classmethod
    def load(cls, path: str | Path) -> CampaignConfig:
        source = Path(path).resolve()
        value = _mapping(json.loads(source.read_text(encoding="utf-8")), "campaign")
        if value.get("schema_version") != 2:
            raise ValueError("unsupported campaign schema; use schema_version 2")
        if value.get("mode", "md") != "md":
            raise ValueError("MD campaign requires mode 'md'")
        unknown = set(value) - {
            "schema_version",
            "mode",
            "structure",
            "output_root",
            "reaction_run",
            "trajectories",
            "require_gpu",
            "adapter_profiles",
        }
        if unknown:
            raise ValueError(f"unknown campaign keys: {sorted(unknown)}")
        output_value = value.get("output_root")
        if not isinstance(output_value, str) or not output_value:
            raise ValueError("campaign.output_root must be a path string")

        def structure_path(raw: object, name: str) -> Path:
            if not isinstance(raw, str) or not raw:
                raise ValueError(f"{name} must be a path string")
            path = (source.parent / raw).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            return path

        # A trajectory's own structure overrides the campaign-wide one.
        default_structure = (
            structure_path(value["structure"], "campaign.structure")
            if "structure" in value
            else None
        )

        raw_profiles = _mapping(value.get("adapter_profiles"), "campaign.adapter_profiles")
        if not raw_profiles:
            raise ValueError("campaign.adapter_profiles must be a non-empty object")
        adapter_profiles: dict[str, AdapterSpec] = {}
        for name, raw_profile in raw_profiles.items():
            if not _TRAJECTORY_ID.fullmatch(name) or name in {".", ".."}:
                raise ValueError(
                    f"adapter profile name must be a safe non-empty identifier: {name!r}"
                )
            adapter_profiles[name] = AdapterSpec.from_dict(raw_profile)

        raw_trajectories = value.get("trajectories")
        if (
            not isinstance(raw_trajectories, Sequence)
            or isinstance(raw_trajectories, (str, bytes))
            or not raw_trajectories
        ):
            raise ValueError("campaign.trajectories must be a non-empty array")
        trajectories: list[TrajectorySpec] = []
        trajectory_profiles: list[str] = []
        trajectory_structures: list[Path] = []
        for raw_trajectory in raw_trajectories:
            trajectory = dict(_mapping(raw_trajectory, "trajectory"))
            profile = trajectory.pop("adapter_profile", None)
            if not isinstance(profile, str) or not profile:
                raise ValueError("trajectory.adapter_profile must name an adapter profile")
            if profile not in adapter_profiles:
                raise ValueError(
                    f"trajectory.adapter_profile references unknown adapter profile: {profile!r}"
                )
            if "structure" in trajectory:
                structure = structure_path(trajectory.pop("structure"), "trajectory.structure")
            elif default_structure is not None:
                structure = default_structure
            else:
                raise ValueError("set campaign.structure or a structure for every trajectory")
            trajectories.append(TrajectorySpec.from_dict(trajectory))
            trajectory_profiles.append(profile)
            trajectory_structures.append(structure)
        identifiers = [trajectory.id for trajectory in trajectories]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("trajectory IDs must be unique")
        require_gpu = value.get("require_gpu", True)
        if type(require_gpu) is not bool:
            raise TypeError("campaign.require_gpu must be a boolean")
        return cls(
            source=source,
            output_root=(source.parent / output_value).resolve(),
            reaction_run=_run_config(value.get("reaction_run", {})),
            trajectories=tuple(trajectories),
            adapter_profiles=adapter_profiles,
            trajectory_profiles=tuple(trajectory_profiles),
            trajectory_structures=tuple(trajectory_structures),
            require_gpu=require_gpu,
        )

    def trajectory(self, index: int) -> TrajectorySpec:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("trajectory index must be an integer")
        if not 0 <= index < len(self.trajectories):
            raise IndexError(f"trajectory index {index} is outside this campaign")
        return self.trajectories[index]

    def adapter_profile_for(self, index: int) -> str:
        """Return the name of the adapter profile assigned to one trajectory."""

        self.trajectory(index)
        return self.trajectory_profiles[index]

    def adapter_for(self, index: int) -> AdapterSpec:
        """Return the adapter configuration assigned to one trajectory."""

        return self.adapter_profiles[self.adapter_profile_for(index)]

    def structure_for(self, index: int) -> Path:
        """Return the starting structure of one trajectory."""

        self.trajectory(index)
        return self.trajectory_structures[index]


__all__ = ["AdapterSpec", "CampaignConfig", "TrajectorySpec"]
