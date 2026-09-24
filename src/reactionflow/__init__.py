"""General-purpose reaction discovery and pathway refinement."""

from ._version import __version__
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
from .pathway import (
    ConnectivityCheck,
    FrequencyValidation,
    PathwayConfig,
    PathwayOutcome,
    refine_pathway,
)
from .restart import ComponentState, ExactRestartSnapshot
from .run import ReactionRun, ReactionRunConfig, RunSummary
from .runtime import ExactDynamicsRuntime, ExactRuntimeProvider

__all__ = [
    "AdapterSpec",
    "BondChangeDetector",
    "BondDetectorConfig",
    "BondEvent",
    "CampaignConfig",
    "ComponentState",
    "ConnectivityCheck",
    "ExactDynamicsRuntime",
    "ExactRestartSnapshot",
    "ExactRuntimeProvider",
    "FrequencyValidation",
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
