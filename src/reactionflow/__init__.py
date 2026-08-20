"""General-purpose reaction discovery and pathway refinement."""

from importlib.metadata import PackageNotFoundError, version

from .campaign import AdapterSpec, CampaignConfig, TrajectorySpec
from .candidates import ReactionCandidate, ReactionTracker, same_reaction
from .detection import (
    BondChangeDetector,
    BondDetectorConfig,
    BondEvent,
    assign_atom_ids,
    atom_ids,
)
from .mlip import MLIPAdapter, load_mlip_adapter
from .pathway import PathwayConfig, PathwayOutcome, refine_pathway
from .restart import ComponentState, ExactRestartSnapshot
from .run import ReactionRun, ReactionRunConfig, RunSummary
from .runtime import ExactDynamicsRuntime, ExactRuntimeProvider

try:
    __version__ = version("reactionflow")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"

__all__ = [
    "AdapterSpec",
    "BondChangeDetector",
    "BondDetectorConfig",
    "BondEvent",
    "CampaignConfig",
    "ComponentState",
    "ExactDynamicsRuntime",
    "ExactRestartSnapshot",
    "ExactRuntimeProvider",
    "MLIPAdapter",
    "PathwayConfig",
    "PathwayOutcome",
    "ReactionCandidate",
    "ReactionRun",
    "ReactionRunConfig",
    "ReactionTracker",
    "RunSummary",
    "TrajectorySpec",
    "__version__",
    "assign_atom_ids",
    "atom_ids",
    "load_mlip_adapter",
    "refine_pathway",
    "same_reaction",
]
