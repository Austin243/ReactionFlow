"""Validated, scheduler-neutral trajectory campaign configuration."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from ._durable import SAFE_NAME
from .pathway import _RETIRED_KEYS
from .run import DEFAULT_MAX_HOURS_PER_BARRIER, ReactionRunConfig


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be an object with string keys")
    return value


def _reject_unknown(given: Iterable[str], allowed: Iterable[str], label: str) -> None:
    unknown = set(given) - set(allowed)
    if unknown:
        raise ValueError(f"unknown {label}: {sorted(unknown)}")


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
        _reject_unknown(mapping, {"factory", "options"}, "adapter keys")
        factory = mapping.get("factory")
        if not isinstance(factory, str) or factory.count(":") != 1 or not all(factory.split(":")):
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
        _reject_unknown(mapping, [item.name for item in fields(TrajectorySpec)], "trajectory keys")
        trajectory_id = mapping.get("id")
        if not isinstance(trajectory_id, str) or not SAFE_NAME.fullmatch(trajectory_id):
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
        steps = mapping.get("total_steps")
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
            raise ValueError("trajectory.total_steps must be a positive integer")
        return cls(
            id=trajectory_id,
            total_steps=int(steps),
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
    _reject_unknown(overrides, merged, "reaction_run keys")
    for key, item in overrides.items():
        if key in {"detector", "pathway"}:
            nested = _mapping(item, f"reaction_run.{key}")
            known = set(merged[key])  # type: ignore[call-overload]
            if key == "pathway":
                known |= _RETIRED_KEYS
            _reject_unknown(nested, known, f"reaction_run.{key} keys")
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
    max_hours_per_barrier: float | None = DEFAULT_MAX_HOURS_PER_BARRIER
    require_gpu: bool = True
    refine_after_md: bool = False

    @classmethod
    def load(cls, path: str | Path) -> CampaignConfig:
        source = Path(path).resolve()
        value = _mapping(json.loads(source.read_text(encoding="utf-8")), "campaign")
        if value.get("schema_version") != 2:
            raise ValueError("unsupported campaign schema; use schema_version 2")
        if value.get("mode", "md") != "md":
            raise ValueError("MD campaign requires mode 'md'")
        known = (
            "schema_version",
            "mode",
            "structure",
            "output_root",
            "reaction_run",
            "trajectories",
            "require_gpu",
            "refine_after_md",
            "adapter_profiles",
            "max_hours_per_barrier",
        )
        _reject_unknown(value, known, "campaign keys")
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
        default_structure = None
        if "structure" in value:
            default_structure = structure_path(value["structure"], "campaign.structure")

        raw_profiles = _mapping(value.get("adapter_profiles"), "campaign.adapter_profiles")
        if not raw_profiles:
            raise ValueError("campaign.adapter_profiles must be a non-empty object")
        adapter_profiles: dict[str, AdapterSpec] = {}
        for name, raw_profile in raw_profiles.items():
            if not SAFE_NAME.fullmatch(name):
                raise ValueError(
                    f"adapter profile name must be a safe non-empty identifier: {name!r}"
                )
            adapter_profiles[name] = AdapterSpec.from_dict(raw_profile)

        raw_trajectories = value.get("trajectories")
        if not isinstance(raw_trajectories, list) or not raw_trajectories:
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
        if len({trajectory.id for trajectory in trajectories}) != len(trajectories):
            raise ValueError("trajectory IDs must be unique")
        max_hours = value.get("max_hours_per_barrier", DEFAULT_MAX_HOURS_PER_BARRIER)
        if max_hours is not None:
            max_hours = _finite_float(max_hours, "campaign.max_hours_per_barrier", minimum=0.0)
            if max_hours == 0:
                raise ValueError("campaign.max_hours_per_barrier must be positive, or null")
        require_gpu = value.get("require_gpu", True)
        if type(require_gpu) is not bool:
            raise TypeError("campaign.require_gpu must be a boolean")
        refine_after_md = value.get("refine_after_md", False)
        if type(refine_after_md) is not bool:
            raise TypeError("campaign.refine_after_md must be a boolean")
        return cls(
            source=source,
            output_root=(source.parent / output_value).resolve(),
            reaction_run=_run_config(value.get("reaction_run", {})),
            trajectories=tuple(trajectories),
            adapter_profiles=adapter_profiles,
            trajectory_profiles=tuple(trajectory_profiles),
            trajectory_structures=tuple(trajectory_structures),
            max_hours_per_barrier=max_hours,
            require_gpu=require_gpu,
            refine_after_md=refine_after_md,
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
