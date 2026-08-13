# Extraction provenance

Every pull request adapting implementation from MatEnsemble must update this ledger. Record the
destination, exact upstream file and commit, and material changes. Preserve upstream copyright and
license notices.

Extraction baseline: `Austin243/MatEnsemble-private` at
`c1b212296a0b6a3b7dec3a927e81cb7bd6d0125e`.

| ReactionFlow destination | MatEnsemble source | Adaptation status |
| --- | --- | --- |
| `detection.py` | `src/matensemble/analysis/bonds.py` at baseline commit; tests adapted from `tests/test_bond_detector.py` | RF-1 implemented; preserved hysteresis, persistence, MIC distances, stable topology, and checkpoint recovery. Uses explicit stable integer IDs, a fresh ASE neighbor list per frame, and no element-specific control flow. |
| `candidates.py` | Tracking, changed-region, and exact graph-matching portions of `src/matensemble/reactions/core.py` at baseline commit; tests adapted from `tests/test_reaction_identity.py` | RF-2a/b implemented; retains candidate tracking and element/change-labeled forward/reverse identity. Hydrogen handoff/coordination policy was not copied. |
| `store.py` | Registry portions of `src/matensemble/reactions/core.py` and publication portions of `src/matensemble/analysis/live_bonds.py` at baseline commit | RF-2c implemented as one occurrence table plus atomic endpoint bundles. Retains all occurrences, records detector settings, uses exact identity, and allows only resolved representatives. Status lifecycle, controller coupling, and hash indexes were not copied. |
| `pathway.py` | Endpoint alignment, active-region, relaxation, topology-gate, IDPP, and CI-NEB portions of `src/matensemble/reactions/pathway.py` at baseline commit; test adapted from `tests/test_reaction_pathway.py` | RF-3 implemented as a pure in-memory primitive with context-managed calculator leases and calculator-free outcomes. Global factory registration, calculator specs, file I/O, controller coupling, and transition-state validation claims were not copied. |
| `segments.py` | Structural checkpoint and generation portions of `src/matensemble/reactions/segments.py` at baseline commit; tests adapted from `tests/test_reaction_segments.py` | RF-4 implemented with atomic checkpoint bundles and explicit structural fidelity. Run IDs, stop files, driver reflection, exact-state codecs, and scheduler hooks were not copied. |
| `trajectory.py` | `src/matensemble/analysis/live_bonds.py` | Planned; retain only supported file-monitor guarantees. |
| `coordinator.py` | State policy from `src/matensemble/reactions/controller.py` | Planned rewrite; do not copy MatEnsemble manager coupling. |
