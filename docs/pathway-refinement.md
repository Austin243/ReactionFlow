# Pathway refinement

`refine_pathway()` relaxes one resolved candidate and runs a serial, fixed-cell climbing-image
NEB. It is an in-memory scientific primitive; `ReactionRun` publishes its outcome while direct
callers retain ownership of artifacts and retry policy.

```python
from contextlib import contextmanager

from reactionflow import PathwayConfig, refine_pathway


@contextmanager
def calculators(stage):
    calculator = make_calculator(stage)
    try:
        yield calculator
    finally:
        release_calculator(calculator)


outcome = refine_pathway(
    candidate,
    calculator_provider=calculators,
    detector_config=detector_config,
    config=PathwayConfig(images=7),
)
```

For a stored occurrence, load both inputs that were published together:

```python
candidate = store.load(occurrence_id)
detector_config = store.load_detector_config(occurrence_id)
```

The provider is entered sequentially for `relax_reactant`, `relax_product`, and `neb`. Only one
calculator lease is live at a time, and the NEB images share one calculator in serial execution.

## Scientific workflow

- Product atoms are aligned to reactant order by stable ID, not array position.
- Modest fully periodic cell drift is mapped fractionally onto the fixed reactant cell. Larger
  volume or strain changes are rejected by configuration.
- Atoms in changed bonds and neighbors within `active_radius` remain active; all others are fixed
  to the reactant position.
- Both endpoints are relaxed and checked against the exact bond thresholds used for detection.
  Collapsed, ambiguous, and unexpectedly changed endpoints do not proceed to NEB.
- Intermediate images use ASE IDPP interpolation with minimum-image distances, followed by serial
  improved-tangent climbing-image NEB.

`PathwayOutcome.status` is one of `unresolved`, `collapsed`, `relaxation_failed`, `neb_failed`,
`neb_converged`, or `failed`. Outcomes retain calculator-free endpoint or NEB snapshots as far as
the attempt progressed. Only `neb_converged` includes image energies and a forward barrier in eV.

## Current limits

This layer does not write files, schedule work, run images concurrently, vary the cell along the
path, or select among multiple mechanisms. A converged CI-NEB result is a potential-energy path;
it is not a vibrationally validated transition state, an IRC, a free-energy barrier, or kinetics.
