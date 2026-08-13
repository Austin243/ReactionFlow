"""General-purpose reaction discovery and pathway refinement."""

from importlib.metadata import PackageNotFoundError, version

from .detection import (
    BondChangeDetector,
    BondDetectorConfig,
    BondEvent,
    assign_atom_ids,
    atom_ids,
)

try:
    __version__ = version("reactionflow")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"

__all__ = [
    "BondChangeDetector",
    "BondDetectorConfig",
    "BondEvent",
    "__version__",
    "assign_atom_ids",
    "atom_ids",
]
