# Pathway refinement

`refine_pathway()` relaxes one resolved candidate, runs NEB and climbing-image NEB (SSNEB for NPT),
then applies the constrained frequency diagnostic to the highest-energy or highest-enthalpy interior
image. It is an in-memory scientific primitive; `ReactionRun` publishes its outcome while direct
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
    pressure_GPa=20.0,  # SSNEB at this pressure; omit or use None for fixed-cell NEB
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
- NVT retains the fixed reactant cell and configured cell-drift mapping limits. NPT retains and
  relaxes both endpoint cells at the target pressure, with no fixed-cell mapping limit. SSNEB
  requires fully periodic, nonsingular, right-handed cells and removes rigid cell rotation.
- Atoms in changed bonds and neighbors within `active_radius` remain active. Other atoms are fixed
  to the reactant position for NVT, or to its fractional coordinates for NPT, so they follow affine
  cell deformation.
- Both endpoints are relaxed and checked against the exact bond thresholds used for detection.
  Collapsed, ambiguous, and unexpectedly changed endpoints do not proceed to NEB.
- NVT uses ASE IDPP interpolation; NPT interpolates reference-cell atomic and deformation coordinates
  after periodic alignment. Improved-tangent NEB converges before climbing is enabled. SSNEB includes
  both atomic and cell forces in its tangent and climbing projection, using enthalpy E + PV.
- After CI-NEB convergence, central finite differences displace only the same active atoms used by
  the pathway. Frequencies with imaginary magnitude at or above the configured cutoff are counted,
  while the complete signed spectrum remains available for diagnosing small numerical modes.

`PathwayOutcome.status` is one of `unresolved`, `collapsed`, `relaxation_failed`, `neb_failed`,
`ci_neb_failed`, `ci_neb_converged`, or `failed`. Outcomes retain calculator-free endpoint or NEB
snapshots as far as the attempt progressed. Only `ci_neb_converged` includes image energies and a
forward barrier in eV: potential energy for NVT, enthalpy for NPT. The `frequency_validation`
diagnostic is classified as zero, one, or multiple significant imaginary modes, or `failed`.
That nested status does not change the successful pathway status or prevent MD from resuming.

The diagnostic records the saddle-image index, active atom IDs, finite-difference displacement,
imaginary-mode cutoff, active-atom residual force, complete signed spectrum, significant mode
indices, and the primary imaginary-mode displacement when present. It is a partial Hessian with a
fixed environment and fixed cell. One significant imaginary mode is consistent with a constrained
first-order saddle, not proof of connectivity. A coupled atom/cell SSNEB saddle can have zero
imaginary modes in this fixed-cell diagnostic.

`frequency_delta` defaults to `0.01` Å and `imaginary_frequency_cutoff_cm1` defaults to
`50.0` cm⁻¹. Imaginary frequencies are stored as negative real values. The primary mode vector is
normalized over its active Cartesian components, and its sign is arbitrary.

## Constant-pressure SSNEB

Campaigns pass `pressure_GPa` automatically. Direct calls to refinement or run methods must supply
it on each call: `None` selects NVT, and a number (including zero) selects NPT. Positive pressure
means compression. The calculator must supply stress; otherwise refinement records `failed`.
SSNEB relaxes volume and shape even when MD uses hydrostatic scaling. Existing force tolerances
apply to the combined atomic/cell gradient, with the cell scaling documented in `ssneb.py`.

Results identify `method`, `pressure_GPa`, and `barrier_quantity`. `energies_eV` remains potential
energy; NPT adds `enthalpies_eV` and `volumes_A3`, and `barrier_eV` uses enthalpy. Saved images retain
their individual cells. Older results load as fixed-cell NEB and are not recalculated.

## Current limits

This layer does not write files, schedule work, run images concurrently, select among mechanisms,
run an IRC, calculate thermal or free-energy corrections, refine higher-order saddles, or produce
kinetics.
