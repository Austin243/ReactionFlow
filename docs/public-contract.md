# Provisional public contract

This contract guides extraction and tests. It may change before the first alpha; after
`v0.1.0a1`, changes follow semantic versioning.

## User facade

The intended stable surface is small:

```python
from reactionflow import ReactionRun, ReactionRunConfig

run = ReactionRun.create(
    root="run-01",
    config=ReactionRunConfig(...),
)

summary = run.run_ase(
    atoms,
    md_calculator_provider=md_calculator_provider,
    pathway_calculator_provider=pathway_calculator_provider,
    dynamics_factory=make_dynamics,
    total_steps=1_000_000,
)
```

Strict runtime continuation uses the same facade with a provider that explicitly snapshots and
restores its dynamics and calculator:

```python
summary = run.run_exact(
    atoms,
    runtime_provider=exact_runtime_provider,
    pathway_calculator_provider=pathway_calculator_provider,
    total_steps=1_000_000,
)
```

Recovery after a completed structural checkpoint is explicit:

```python
run = ReactionRun.open("run-01")
run.refine_pending(pathway_calculator_provider)
segment = run.resume_segment()
```

The provisional top-level facade includes:

- `ReactionRun`, `ReactionRunConfig`, and `RunSummary`;
- `ExactDynamicsRuntime`, `ExactRuntimeProvider`, `ExactRestartSnapshot`, and `ComponentState`;
- `CampaignConfig`, `TrajectorySpec`, `AdapterSpec`, and `MLIPAdapter`;
- `BondChangeDetector`, `BondDetectorConfig`, and `BondEvent`;
- `ReactionTracker` and `ReactionCandidate`;
- `PathwayConfig`, `PathwayOutcome`, and `refine_pathway()`; and
- stable-ID helpers and `same_reaction()`.

Low-level implementation helpers remain module-level APIs unless deliberately promoted.

RF-1 deliberately promotes the reusable detection surface at the top level:

- `BondChangeDetector`, `BondDetectorConfig`, and `BondEvent`; and
- `assign_atom_ids()` and `atom_ids()`.

Detector checkpoints are small versioned dictionaries. The contract may grow when the durable run
store has a concrete need for migrations or typed errors.

RF-2a promotes the in-memory candidate surface:

- `ReactionTracker` aggregates stable topologies; and
- `ReactionCandidate` carries independent reactant/product snapshots and one connected changed
  region.

RF-2b adds `same_reaction()` for exact forward/reverse topology identity. NetworkX graph objects,
class IDs, and graph serialization remain implementation details.

RF-2c adds the provisional module-level `OccurrenceStore` and `OccurrenceRecord` APIs. They retain
and reopen immutable candidate bundles; `ReactionRun` owns their normal use.

RF-3 promotes `PathwayConfig`, `PathwayOutcome`, and `refine_pathway()` for one in-memory candidate.
The context-managed `CalculatorProvider` type remains module-level; an `MLIPAdapter` supplies it
without changing the pathway primitive.

RF-4 adds provisional module-level `SegmentStore`, `SegmentGeneration`, and `ResumeToken` APIs for
structural or exact checkpoint/resume. They remain low-level APIs normally reached through
`ReactionRun`.

RF-5 promotes `ReactionRun`, `ReactionRunConfig`, and `RunSummary`. Manual scheduler-neutral
methods are `start`, `observe`, `checkpoint`, `refine_pending`, `resume_segment`, and `complete`;
`run_ase()` is the structural synchronous executor. `run_exact()` uses an `ExactRuntimeProvider`
and publishes a complete runtime plus monitor checkpoint at every observation boundary. It can
reopen an active MD segment without silently resetting detector persistence, tracker state,
integrator, thermostat, RNG, or calculator state.

## Calculator provider

A calculator provider returns a context-managed lease containing an ASE calculator. Acquisition is
stage-aware; release is explicit and occurs even after failure. The local serial-GPU executor never
holds more than one lease. Serializable provider specifications are deferred.

## Adapter interface

An MLIP factory receives one `TrajectorySpec` and a JSON options mapping. Its `MLIPAdapter`
provides exact start/restore contexts plus sequential pathway calculator contexts. Campaign
processes map directly from a zero-based task index to one trajectory; no scheduler identifiers,
queues, or output references enter the chemistry or restart models.

## Compatibility

Durable JSON and SQLite records carry explicit schema versions. The package version in
`pyproject.toml` is the source of truth, and release tags use the matching `v<version>` form.
