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
    pathway_calculator=calculator_provider,
)

summary = run.run_ase(
    atoms,
    md_calculator=md_calculator_provider,
    dynamics_factory=make_dynamics,
    total_steps=1_000_000,
)
```

An interrupted run is reopened explicitly:

```python
run = ReactionRun.open("run-01")
summary = run.resume(...)
```

The top-level stable names are planned to be:

- `ReactionRun`, `ReactionRunConfig`, and `RunSummary`;
- `BondDetectorConfig` and `PathwayConfig`;
- `CalculatorProvider` and its stage context;
- `ReactionCandidate` and `PathwayOutcome`; and
- typed lifecycle, outcome, and error enums.

Low-level implementation helpers remain module-level APIs unless deliberately promoted.

## Calculator provider

A calculator provider returns a context-managed lease containing an ASE calculator. Acquisition is
stage-aware; release is explicit and occurs even after failure. The local serial-GPU executor never
holds more than one lease. Serializable adapters use an importable factory plus JSON-compatible
parameters rather than pickling a live calculator.

## Adapter interface

Adapters consume scheduler-neutral coordinator facts and commands. Initial facts cover candidate
observation, segment completion/failure, pathway completion/failure, and monitor failure. Initial
commands request a safe stop, run a pathway, start a segment, or finish the run.

Flux resources, MatEnsemble chore IDs, queues, and output references do not appear in core models.
MatEnsemble translates at its boundary and stores its own metadata under
`adapters/matensemble/`.

## Compatibility

Every public result and durable artifact carries an explicit schema version. Schema readers either
migrate supported older data or fail with a typed compatibility error; they never silently reinterpret
reaction identity. The package version in `pyproject.toml` is the source of truth, and release tags
use the matching `v<version>` form.
