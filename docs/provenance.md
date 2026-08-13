# Extraction provenance

Every pull request adapting implementation from MatEnsemble must update this ledger. Record the
destination, exact upstream file and commit, and material changes. Preserve upstream copyright and
license notices.

Extraction baseline: `Austin243/MatEnsemble-private` at
`c1b212296a0b6a3b7dec3a927e81cb7bd6d0125e`.

| ReactionFlow destination | MatEnsemble source | Adaptation status |
| --- | --- | --- |
| `detection.py` | `src/matensemble/analysis/bonds.py`; tracking portions of `src/matensemble/reactions/core.py` | Planned; remove element-specific transition policy and unify atom IDs. |
| `store.py` | Registry portions of `src/matensemble/reactions/core.py`; publication portions of `src/matensemble/analysis/live_bonds.py` | Planned; version schemas and drain every occurrence. |
| `pathway.py` | `src/matensemble/reactions/pathway.py` | Planned; preserve scientific gates and rename convergence claims. |
| `segments.py` | `src/matensemble/reactions/segments.py` | Planned; make checkpoint publication atomic and clarify fidelity. |
| `trajectory.py` | `src/matensemble/analysis/live_bonds.py` | Planned; retain only supported file-monitor guarantees. |
| `coordinator.py` | State policy from `src/matensemble/reactions/controller.py` | Planned rewrite; do not copy MatEnsemble manager coupling. |
