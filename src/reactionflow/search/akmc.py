"""Adaptive kinetic Monte Carlo rules, following EON's AKMC defaults.

A state is searched until its confidence, 1 - 1/N after N consecutive searches that repeat a known
process inside the thermal window, reaches the target. A kinetic Monte Carlo step then picks one
process in the window with probability proportional to its harmonic transition-state rate and
advances the clock by an exponentially distributed residence time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

BOLTZMANN_EV_PER_K = 8.617333262e-5


@dataclass(frozen=True)
class AKMCConfig:
    temperature_K: float = 300.0
    steps: int = 100
    confidence: float = 0.95
    thermal_window_kT: float = 20.0
    seed: int = 1

    def __post_init__(self) -> None:
        for name in ("temperature_K", "thermal_window_kT"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not value > 0:
                raise ValueError(f"akmc.{name} must be a positive number")
            if not math.isfinite(value):
                raise ValueError(f"akmc.{name} must be finite")
        for name, minimum in (("steps", 1), ("seed", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"akmc.{name} must be an integer >= {minimum}")
        if self.seed >= 2**64:
            raise ValueError("akmc.seed must be less than 2**64")
        value = self.confidence
        if isinstance(value, bool) or not isinstance(value, Real) or not 0 < value < 1:
            raise ValueError("akmc.confidence must be between 0 and 1")

    @property
    def kT(self) -> float:
        return BOLTZMANN_EV_PER_K * self.temperature_K


def confidence(repeats: int) -> float:
    """EON's default confidence after `repeats` consecutive repeats of known processes."""

    return 0.0 if repeats < 1 else 1.0 - 1.0 / repeats


def rate(prefactor: float, barrier: float, kT: float) -> float:
    """Harmonic transition-state rate in 1/s."""

    return prefactor * math.exp(-barrier / kT)


def in_window(barrier: float, lowest: float, config: AKMCConfig) -> bool:
    return barrier <= lowest + config.thermal_window_kT * config.kT


def kmc_step(rates: list[float], seed: list[int]) -> tuple[int, float]:
    """Pick one process with probability proportional to its rate; return it and the time step."""

    pick, wait = np.random.default_rng(seed).random(2)
    cumulative = np.cumsum(rates)
    index = int(np.searchsorted(cumulative, pick * cumulative[-1], side="right"))
    return min(index, len(rates) - 1), -math.log(1.0 - wait) / float(cumulative[-1])
