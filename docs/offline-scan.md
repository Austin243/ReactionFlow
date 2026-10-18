# Offline scan

`reactionflow scan` replays saved frames through bond detection and candidate tracking without
loading the model. It needs no GPU and changes nothing in a run, so it is safe while the run is
going.

```bash
reactionflow scan campaign.json                          # every trajectory, with its own settings
reactionflow scan campaign.json --index 0 --radius none  # one trajectory, whole components
reactionflow scan XDATCAR --pair C-O 1.6 2.0 --frame-ps 0.05
```

## Campaign runs

A run saves one frame per observation in `segments/<generation>/trajectory.traj`. A new generation
starts with the frame it resumed from, and the scan reads that frame once. It stops at the first
frame it cannot read, such as the one a running job is writing, and it prints `"scan": null` for a
trajectory with no saved frames yet.

With no options the scan uses the run's own settings from `state.json`, so it finds the candidates
the run recorded. `candidates.csv` gives each one's occurrence ID, and in `summary.json`
`matching_recorded` equals `recorded_by_run` for a finished run. Options change the replay and
nothing else:

- `--radius R` sets the region radius in bonds, and `--radius none` takes whole bonded components.
- `--persistence N` sets how many observations a crossing must last to become a bond change.
- `--stability N` sets how many observations a region's new bonds must hold.
- `--pair A-B FORM BREAK` sets the formation and break distances in Å for one element pair. It can
  be repeated, and it adds to or replaces the run's pair thresholds.
- `--every K` sets the frames between species counts (default 10).
- `--out DIR` writes each trajectory's tables to `DIR/<trajectory-id>/` in place of
  `<run>/scan/`.

## Trajectory files from other codes

Any source that is not a `.json` campaign file is read as a trajectory file, in any format ASE
reads, for example an XDATCAR, an extxyz file, or a LAMMPS dump. Atoms keep the IDs the file
carries, such as an `atom_id` column in extxyz, and otherwise get IDs 0 to N−1 in file order, so
the file must keep its atom order. ASE sorts a LAMMPS dump by its `id` column, and it names the
elements from an `element` or `mass` column, because types alone do not. The settings start from
the defaults of a campaign (thresholds from covalent radii, persistence 3, stability 3, radius 2)
and take the same options. `--frame-ps` gives the time between frames for the time columns, which
are empty without it. The tables go to `<file>-scan/` beside the file unless `--out` names another
directory.

## Output

- `bond_changes.csv` has one row per confirmed change: the frame and time the crossing began, the
  frame it was confirmed, formed or broken, the two atom IDs and elements, the distance in Å, and
  whether the two atoms started in different molecules. Molecules are the bonded components of the
  first frame.
- `candidates.csv` has one row per candidate: the reactant, product and observed frames and the
  times of the first two, resolved or not, whether it was still changing at the end, its number of
  atoms, its bond changes, its local class and label as `reactionflow status` names them (radius 2),
  and the run's occurrence ID when the run recorded the same candidate.
- `species.csv` counts each formula among the bonded components every `--every` frames, with the
  number of starting molecules in each component, which is the chain length in a polymer.
- `summary.json` holds the settings and the counts, which are also printed:
  - `reverted_crossings`, crossings that left their threshold before they persisted;
  - `bonds_formed` and `bonds_broken`, the confirmed changes;
  - `candidates_resolved` and `candidates_unresolved`, with `still_changing_at_end` among the
    unresolved;
  - `net_changes_in_resolved` and `classes_in_resolved`, the bond changes and local classes in the
    resolved candidates;
  - `recorded_by_run` and `matching_recorded` for a campaign run.

These counts separate the stages between a threshold crossing and a candidate: what persistence
filters, what is confirmed, how confirmed changes group into candidates, and how many never come
out resolved.

## From Python

```python
from reactionflow import BondDetectorConfig
from reactionflow.scan import file_frames, scan_frames

detector = BondDetectorConfig(pair_thresholds={"C-O": (1.6, 2.0)})
result = scan_frames(file_frames("XDATCAR", frame_ps=0.05), detector_config=detector)
result.write("XDATCAR-scan")
print(result.summary["candidates_resolved"])
```

`run_frames(run_directory, timestep_fs=...)` yields a run's saved frames in the same form, and
`recorded_occurrences(run_directory)` reads the candidates the run recorded for `occurrences=`.

## Limits

The scan sees nothing finer than the saved frames, so a crossing between two observations is
invisible to it, as it is to the run. It refines nothing and adds nothing to a run's store. Counts
by atom role, such as links between particular carbons, belong in analysis scripts, which can read
`bond_changes.csv`.
