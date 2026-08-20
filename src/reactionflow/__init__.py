"""General-purpose reaction discovery and pathway refinement."""

from importlib.metadata import PackageNotFoundError, version

from .candidates import ReactionCandidate, ReactionTracker, same_reaction
from .detection import (
    BondChangeDetector,
    BondDetectorConfig,
    BondEvent,
    assign_atom_ids,
    atom_ids,
)
from .pathway import PathwayConfig, PathwayOutcome, refine_pathway
from .restart import ComponentState, ExactRestartSnapshot
from .run import ReactionRun, ReactionRunConfig, RunSummary
from .runtime import ExactDynamicsRuntime, ExactRuntimeProvider

try:
    __version__ = version("reactionflow")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"

__all__ = [
    "BondChangeDetector",
    "BondDetectorConfig",
    "BondEvent",
    "ComponentState",
    "ExactDynamicsRuntime",
    "ExactRestartSnapshot",
    "ExactRuntimeProvider",
    "PathwayConfig",
    "PathwayOutcome",
    "ReactionCandidate",
    "ReactionRun",
    "ReactionRunConfig",
    "ReactionTracker",
    "RunSummary",
    "__version__",
    "assign_atom_ids",
    "atom_ids",
    "refine_pathway",
    "same_reaction",
]
