# Pathway refinement

`refine_pathway()` relaxes one resolved candidate, runs NEB and climbing-image NEB (SSNEB for NPT),
then applies the constrained frequency diagnostic to the highest-energy or highest-enthalpy interior
image and checks where that saddle leads. It is an in-memory scientific primitive; `ReactionRun`
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
- Every atom relaxes, in the endpoints, the NEB or SSNEB band, and the connectivity check; for NPT
  the cell relaxes too. The product keeps its own coordinates for atoms in changed bonds and
  neighbors within `active_radius`. Its other atoms start from the reactant's positions for NVT,
  or its fractional coordinates for NPT, so both endpoints relax from the same surroundings.
- Both endpoints are relaxed and checked against the exact bond thresholds used for detection.
  Collapsed, ambiguous, and unexpectedly changed endpoints do not proceed to NEB. Neither does a
  pair of endpoints whose bonds differ anywhere else in the cell: relaxation that forms or breaks
  a bond away from the reaction would otherwise enter the path and the barrier.
- NVT uses ASE IDPP interpolation; NPT interpolates reference-cell atomic and deformation coordinates
  after periodic alignment. Improved-tangent NEB converges before climbing is enabled. SSNEB includes
  both atomic and cell forces in its tangent and climbing projection, using enthalpy E + PV.
- After CI-NEB convergence, central finite differences displace only atoms in changed bonds and
  neighbors within `active_radius`, holding the rest of the relaxed cell fixed. Frequencies with
  imaginary magnitude at or above the configured cutoff are counted, while the complete signed
  spectrum remains available for diagnosing small numerical modes.

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

## Saddle connectivity

When the frequency diagnostic finds a significant imaginary mode, the saddle is displaced 0.1 Å
each way along its primary mode, as the total displacement of the active atoms. Each copy is then
relaxed exactly like the endpoints: every atom free, the same `relax_fmax` and `relax_steps`, and
the cell filter at the target pressure for NPT. Each relaxed side is classified with the detector's
bond thresholds:

- `reactant`, `product`, or `other`;
- `ambiguous` when a changed bond stays inside the hysteresis gap;
- `not_converged` when the relaxation does not converge; and
- `no_step` when the displaced copy already meets the force tolerance, so the mode is too soft to
  test at this displacement.

`connectivity.status` is `connects_endpoints` when one side reaches the reactant and the other the
product. It is `inconclusive` when either side is `not_converged` or `no_step`,
`does_not_connect` otherwise, and `failed` if the check raises. The check runs inside the NEB
calculator lease and is recorded in `result.json`. It never changes the top-level
`ci_neb_converged` status, and it is null when no significant imaginary mode was found. A saddle
with one imaginary mode that connects both endpoints is strong evidence for the transition state
of the detected reaction, but it is not an intrinsic reaction coordinate.

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
