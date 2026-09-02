# Pathway refinement

`refine_pathway()` relaxes one resolved candidate, runs a serial fixed-cell NEB followed by a
climbing-image NEB stage, and classifies the highest-energy interior image with a constrained
active-region frequency calculation. It is an in-memory scientific primitive; `ReactionRun`
publishes its outcome while direct callers retain ownership of artifacts and retry policy.

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
calculator lease is live at a time. The NEB images share one calculator in serial execution, and
frequency validation reuses that lease before it is released.

## Scientific workflow

- Product atoms are aligned to reactant order by stable ID, not array position.
- Modest fully periodic cell drift is mapped fractionally onto the fixed reactant cell. Larger
  volume or strain changes are rejected by configuration.
- Atoms in changed bonds and neighbors within `active_radius` remain active; all others are fixed
  to the reactant position.
- Both endpoints are relaxed and checked against the exact bond thresholds used for detection.
  Collapsed, ambiguous, and unexpectedly changed endpoints do not proceed to NEB.
- Intermediate images use ASE IDPP interpolation with minimum-image distances. A serial
  improved-tangent NEB converges first; only then is climbing enabled for the CI-NEB stage.
- After CI-NEB convergence, central finite differences displace only the same active atoms used by
  the pathway. Frequencies with imaginary magnitude at or above the configured cutoff are counted,
  while the complete signed spectrum remains available for diagnosing small numerical modes.

`PathwayOutcome.status` is one of `unresolved`, `collapsed`, `relaxation_failed`, `neb_failed`,
`ci_neb_failed`, `ci_neb_converged`, or `failed`. Outcomes retain calculator-free endpoint or NEB
snapshots as far as the attempt progressed. Only `ci_neb_converged` includes image energies and a
forward barrier in eV. It also includes a `frequency_validation` diagnostic classified as zero,
one, or multiple significant imaginary modes, or `failed`. That nested status does not change the
successful pathway status or prevent MD from resuming.

The diagnostic records the saddle-image index, active atom IDs, finite-difference displacement,
imaginary-mode cutoff, active-atom residual force, complete signed spectrum, significant mode
indices, and the primary imaginary-mode displacement when present. It is a partial Hessian with a
fixed environment and fixed cell. One significant imaginary mode is therefore consistent with a
constrained first-order saddle, not proof that it connects the intended endpoints.

`frequency_delta` defaults to `0.01` Å and `imaginary_frequency_cutoff_cm1` defaults to
`50.0` cm⁻¹. Imaginary frequencies are stored as negative real values. The primary mode vector is
normalized over its active Cartesian components, and its sign is arbitrary.

## Current limits

This layer does not write files, schedule work, run images concurrently, vary the cell along the
path, or select among multiple mechanisms. It does not run an IRC, calculate endpoint thermal or
free-energy corrections, refine higher-order saddles automatically, or produce kinetics.
