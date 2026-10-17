# Pathway refinement

`refine_pathway()` relaxes one resolved candidate, runs NEB and climbing-image NEB (SSNEB for NPT),
then tests whether the highest-energy or highest-enthalpy interior image is a saddle between the
band's two ends. Only a band that passes that test records a barrier. It is an in-memory scientific
primitive; `ReactionRun` publishes its outcome while direct callers retain ownership of artifacts
and retry policy.

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
the saddle check reuses that lease before it is released. The fixed endpoints are evaluated
once with that lease, so each NEB, CI-NEB, or SSNEB step evaluates only the moving images.

## Scientific workflow

- Product atoms are aligned to reactant order by stable ID, not array position.
- NVT retains the fixed reactant cell and configured cell-drift mapping limits. NPT relaxes each
  endpoint cell at the target pressure, with no fixed-cell mapping limit. SSNEB requires fully
  periodic, nonsingular, right-handed cells and removes rigid cell rotation.
- Every atom relaxes, in the endpoints, the NEB or SSNEB band, and the connectivity check; for NPT
  the cell relaxes too. The reactant relaxes first. The product then starts from the relaxed
  reactant, cell included, with only the atoms in changed bonds and the atoms bonded to them placed
  where they sat in the product frame. Every other atom starts at the same minimum in both
  endpoints, so the band spends its images on the reaction rather than on hot-frame motion around
  it, and a hot frame cannot relax the surroundings differently in each.
- Both endpoints are relaxed and checked against the exact bond thresholds used for detection.
  Collapsed, ambiguous, and unexpectedly changed endpoints do not proceed to NEB. Neither does a
  pair of endpoints whose bonds differ anywhere else in the cell: relaxation that forms or breaks
  a bond away from the reaction would otherwise enter the path and the barrier.
- NVT uses ASE IDPP interpolation; NPT interpolates reference-cell atomic and deformation coordinates
  after periodic alignment. Improved-tangent NEB converges before climbing is enabled. SSNEB includes
  both atomic and cell forces in its tangent and climbing projection, using enthalpy E + PV.

`PathwayOutcome.status` is `unresolved`, `collapsed`, `relaxation_failed`, `neb_failed`,
`ci_neb_failed`, or `failed` when refinement stops before the band converges. A converged band ends
with the saddle check's verdict below. Outcomes retain calculator-free endpoint or NEB snapshots as
far as the attempt progressed. Every converged band keeps its image energies, and only
`saddle_validated` adds a forward barrier in eV: potential energy for NVT, enthalpy for NPT. No
status stops MD from resuming.

## Saddle check

`check_saddle()` tests a converged band. It runs inside the NEB calculator lease after every
CI-NEB, and on any band file through `reactionflow check`. The saddle is the highest interior image
of the profile: energy for NVT, enthalpy for NPT.

- **Active atoms.** The atoms that move more than 0.05 Å between the two images next to the saddle.
  When fewer than six move, the six that move most.
- **Modes.** Central differences of the forces, with `frequency_delta` steps (0.01 Å), give the
  Hessian over the active atoms, with the cell and every other atom fixed. A mode is an eigenvalue
  below `-curvature_cutoff_eV_A2` (0.05 eV/Å² by default). A curvature cutoff treats every element
  alike, where a frequency cutoff counts more hydrogen noise as a mode.
- **Connectivity.** When there is a mode, the saddle is moved both ways along the lowest one and each
  side is relaxed, as described below.

The check ends a converged band as one of these:

| Status | When |
| --- | --- |
| `saddle_validated` | one mode, and the two sides relax to the band's two ends |
| `no_saddle` | an endpoint is the highest point, or the saddle image has no mode |
| `saddle_rejected` | more than one mode, or the sides do not reach both ends |
| `saddle_unchecked` | the Hessian failed, or the connectivity check failed or was inconclusive |

The outcome's message says which case applies. A saddle lies above both endpoints, so when an
endpoint is the highest point of the band, there is no saddle to test, and the outcome has no
frequency diagnostic or connectivity check.

`frequency_validation` records the saddle-image index, active atom IDs, finite-difference step,
curvature cutoff, active-atom residual force, and every Hessian eigenvalue in eV/Å², in ascending
order. It also records the frequencies of the same Hessian with mass weighting, in cm⁻¹ with
imaginary frequencies as negative values. A Hessian has the same number of negative eigenvalues
with or without mass weighting. The lowest mode's overlap with the band tangent at the saddle (the
displacement between the images on either side, over the active atoms) is recorded too, and so is
the lowest mode itself when it counts as a mode. The mode is normalized over its active Cartesian
components, and its sign is arbitrary. Its status is `zero_imaginary_modes`, `one_imaginary_mode`,
`multiple_imaginary_modes`, or `failed`.

The Hessian is partial, with a fixed environment and a fixed cell. A saddle that is mostly a change
of cell, which SSNEB can find, may have no mode at fixed cell; it then ends as `no_saddle`.

## Saddle connectivity

The saddle is moved each way along its lowest mode until the atom that moves most has moved 0.1 Å.
Each copy is then relaxed exactly like the endpoints: every atom free, the same `relax_fmax` and
`relax_steps`, and the cell filter at the target pressure for NPT. Each relaxed side is classified
with the detector's bond thresholds:

- `reactant`, `product`, or `other`;
- `ambiguous` when any atom pair stays inside its hysteresis gap;
- `not_converged` when the relaxation does not converge; and
- `no_step` when the moved copy already meets the force tolerance, so the mode is too soft to test
  at this displacement.

A side is labeled `reactant` or `product` only when its whole-cell bond topology matches the
corresponding end of the band. A bond change elsewhere in the cell makes that side `other`. When
both ends have the same bonds, as in a conformer change, a side reaches the end it lies nearer to
by root-mean-square displacement. An end with any pair in the hysteresis gap makes the check
`inconclusive` before launching the descents. Matching unambiguous bond topology does not establish
identical conformations.

`connectivity.status` is `connects_endpoints` when one side reaches the reactant and the other the
product. It is `inconclusive` when either side is `ambiguous`, `not_converged`, or `no_step`,
`does_not_connect` otherwise, and `failed` if the check raises. A saddle with one mode that
connects both ends is strong evidence for the transition state of the detected reaction, but it is
not an intrinsic reaction coordinate.

## Checking a band

```bash
reactionflow check campaign.json runs/<trajectory>/pathways/<occurrence>/images.traj
```

`reactionflow check` runs the same check on any band file ASE can read, both ends included: a
pathway's `images.traj`, a band from another code, or an EON saddle written as three images
(minimum, saddle, minimum). The campaign supplies the model, the bond thresholds, and the pathway
settings. `--index` picks the trajectory whose model and pressure are used; a campaign with several
trajectories needs it, even inside a Slurm job. `--pressure` replaces that pressure. The verdict is written beside the band as `<band>.check.json`, in the form
of a pathway's `result.json`, and its status, barrier, and message are printed.

Campaign and state files written before the saddle check may name `active_radius` and
`imaginary_frequency_cutoff_cm1`. They still open, and those two settings are ignored. A trajectory
started before the check cannot resume under it, because its pathway settings, and so its campaign
contract, have changed. Finish it with the version that started it, and test its bands with
`reactionflow check`. Its results keep their `ci_neb_converged` status.

## Constant-pressure SSNEB

Campaigns pass `pressure_GPa` automatically. Direct calls to refinement or run methods must supply
it on each call: `None` selects NVT, and a number (including zero) selects NPT. Positive pressure
means compression. The calculator must supply stress; otherwise refinement records `failed`.
SSNEB relaxes volume and shape even when MD uses hydrostatic scaling. Existing force tolerances
apply to the combined atomic/cell gradient, with the cell scaling documented in `ssneb.py`.

Results identify `method`, `pressure_GPa`, and `barrier_quantity`. `energies_eV` remains potential
energy; NPT adds `enthalpies_eV` and `volumes_A3`, and `barrier_eV` uses enthalpy. Saved images retain
their individual cells.

## Current limits

This layer does not write files, schedule work, run images concurrently, select among mechanisms,
run an IRC, calculate thermal or free-energy corrections, refine higher-order saddles, or produce
kinetics.
