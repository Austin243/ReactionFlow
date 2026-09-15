# Pathway refinement

`refine_pathway()` relaxes one resolved candidate and runs a serial NEB followed by a
climbing-image NEB stage. A constrained active-region frequency calculation then classifies the
interior image with the highest potential energy (NVT) or enthalpy (NPT).
It is an in-memory scientific primitive; `ReactionRun`
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
- For NVT (`pressure_GPa=None`), modest fully periodic cell drift is mapped fractionally onto the
  fixed reactant cell. The existing volume/strain mapping limits and fixed-cell relaxation remain
  unchanged.
- For NPT (numeric `pressure_GPa`, including zero), endpoint cells are retained and independently
  relaxed at the target pressure. Fully periodic, nonsingular, right-handed cells are required.
  Rigid cell rotation is removed by putting each endpoint in lower-triangular form while preserving
  its internal geometry. The fixed-cell mapping limits do not apply to SSNEB.
- Atoms in changed bonds and neighbors within `active_radius` remain active. Other atoms are fixed
  to the reactant position for NVT, or to its fractional coordinates for NPT, so they follow affine
  cell deformation. Candidate and MD checkpoint structures are never modified by refinement.
- Both endpoints are relaxed and checked against the exact bond thresholds used for detection.
  Collapsed, ambiguous, and unexpectedly changed endpoints do not proceed to NEB.
- NVT intermediate images use ASE IDPP interpolation with minimum-image distances. NPT uses linear
  interpolation in combined reference-cell atomic and deformation coordinates after periodic
  endpoint alignment. A serial improved-tangent NEB converges first; only then is climbing enabled
  for the CI-NEB stage. For SSNEB, the tangent, springs, and climbing projection include cell forces
  as well as atomic forces, and image ordering uses enthalpy E + PV.
- After CI-NEB convergence, central finite differences displace only the same active atoms used by
  the pathway. Frequencies with imaginary magnitude at or above the configured cutoff are counted,
  while the complete signed spectrum remains available for diagnosing small numerical modes.

`PathwayOutcome.status` is one of `unresolved`, `collapsed`, `relaxation_failed`, `neb_failed`,
`ci_neb_failed`, `ci_neb_converged`, or `failed`. Outcomes retain calculator-free endpoint or NEB
snapshots as far as the attempt progressed. Only `ci_neb_converged` includes image energies and a
forward barrier in eV. For NPT this is an enthalpy barrier; NVT retains its potential-energy barrier.
It also includes a `frequency_validation` diagnostic classified as zero,
one, or multiple significant imaginary modes, or `failed`. That nested status does not change the
successful pathway status or prevent MD from resuming.

The diagnostic records the saddle-image index, active atom IDs, finite-difference displacement,
imaginary-mode cutoff, active-atom residual force, complete signed spectrum, significant mode
indices, and the primary imaginary-mode displacement when present. It is a partial Hessian with a
fixed environment and fixed cell, including at an SSNEB saddle. It does not test coupled atom/cell
curvature. One significant imaginary mode is therefore consistent with a
constrained first-order saddle, not proof that it connects the intended endpoints.
A coupled SSNEB saddle can have zero imaginary modes in this fixed-cell diagnostic.

`frequency_delta` defaults to `0.01` Å and `imaginary_frequency_cutoff_cm1` defaults to
`50.0` cm⁻¹. Imaginary frequencies are stored as negative real values. The primary mode vector is
normalized over its active Cartesian components, and its sign is arbitrary.

## Constant-pressure SSNEB

Campaign workers pass each trajectory's `pressure_GPa` automatically, including when resuming.
Direct callers of `refine_pathway()`, `ReactionRun.refine_pending()`, `run_exact()`, or `run_ase()`
must pass the same target pressure on every call for NPT; the default `None` retains fixed-cell NEB.
Already-published outcomes are loaded as recorded and are not recalculated.

`ssneb.py` combines ASE's improved-tangent NEB with `UnitCellFilter` gradients of E + PV. All images
share the relaxed reactant's reference cell and the length scale
`J = sqrt(N) * (V_reference/N)^(1/3)` for cell deformation. Six independent deformation components
allow cell volume and shape to relax under isotropic external pressure; this also applies when the
MD barostat restricts motion to hydrostatic scaling. The existing force tolerances apply to the
combined atomic/cell gradient in eV/Å, with cell forces scaled by J. There is no added stress cutoff
or calculation stopping policy. See the [SSNEB method](https://doi.org/10.1063/1.3684549) for the
combined-coordinate approach.

The MLIP must supply ASE stress in eV/Å³. Positive `pressure_GPa` means compression; SSNEB adds
`P * ase.units.GPa * V` to the potential energy and the corresponding pressure term to the cell
gradient. A missing stress implementation produces the existing handled `failed` pathway outcome.

Outcomes record `method` (`neb` or `ssneb`), `pressure_GPa`, and `barrier_quantity`
(`potential_energy` or `enthalpy`). `energies` always contains raw potential energies. Converged NPT
outcomes additionally contain `enthalpies` and `volumes`; in `result.json` these arrays are named
`energies_eV`, `enthalpies_eV`, and `volumes_A3`, and `barrier_eV` uses the stated quantity.
`images.traj` contains the physical images with their individual cells. Older results without these
fields load as fixed-cell NEB. The method does not include finite-temperature free-energy effects.

## Current limits

This layer does not write files, schedule work, run images concurrently, or select among multiple
mechanisms. It does not run an IRC, calculate endpoint thermal or
free-energy corrections, refine higher-order saddles automatically, or produce kinetics.
