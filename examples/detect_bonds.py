"""Minimal persistent bond-formation example."""

from __future__ import annotations

import json

from ase import Atoms

from reactionflow import BondChangeDetector, BondDetectorConfig, assign_atom_ids


def main() -> None:
    atoms = assign_atom_ids(Atoms("CO", positions=[[0, 0, 0], [2.0, 0, 0]]))
    detector = BondChangeDetector(
        BondDetectorConfig(
            persistence_frames=2,
            pair_thresholds={"C-O": (1.5, 1.8)},
        )
    )

    detector.process(atoms, frame=0)
    atoms.positions[1, 0] = 1.4
    detector.process(atoms, frame=1)
    atoms.positions[1, 0] = 1.3
    events = detector.process(atoms, frame=2)

    print(json.dumps(events[0].to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
