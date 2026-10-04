# Figures

The figures in the main README are drawn by code in this folder. Each figure comes as a light and
a dark SVG, and the README shows the one that matches the reader's GitHub theme. Only the SVGs are
committed. The build can also write 2x PNGs to `png/` and a gallery page, `index.html`, with every
figure and a light/dark switch.

```bash
node docs/figures/build.mjs                    # every figure, SVG and PNG
node docs/figures/build.mjs pathway detection  # PNGs for the named figures only
node docs/figures/build.mjs --no-png           # SVG only; Chrome is not needed
```

The build needs Node 20 or newer and no packages. PNG export runs headless Google Chrome; set
`CHROME` to use another Chromium binary.

| Figure | Shows | Used in |
| --- | --- | --- |
| `hero` | An MD trajectory crossing between two basins and the refined path over the saddle | README banner |
| `how-it-works` | One trajectory on a timeline: MD, bond monitor, checkpoint, refinement, restore | How it works |
| `models` | The built-in model families with their model counts, the generic and custom adapters, and the commands | Models |
| `ensembles` | Fixed-cell NEB against variable-cell SSNEB, with enthalpy barriers at three pressures | Constant volume and constant pressure |
| `campaign` | Eight trajectories on two GPU nodes across a resubmitted job | Many trajectories on a cluster |
| `run-directory` | The files one trajectory writes | Output |
| `outcomes` | Each refinement stage and the status it ends with | Refinement outcomes |
| `detection` | The two distance thresholds and the persistence and stability counts for one atom pair | Bond detection |
| `identity` | A reaction as a change graph and how occurrences group into classes | Reaction classes |
| `pathway` | Endpoint relaxation, NEB, climbing image, and saddle checks on a model surface | Pathway refinement |
| `restart` | What a checkpoint holds, and a restored run compared with one that never stopped | Exact restart |
| `phases` | The phases a trajectory records in `state.json` | not used yet |
| `eon-landscape` | The states and saddles EON mode found on a model landscape, the kinetic Monte Carlo steps, and the energy against simulated time | Rare events with EON |
| `eon-loop` | Searches from the current state until the confidence target, then a kinetic Monte Carlo step, replayed from the start of the model run | Rare events with EON |
| `eon-search` | One EON search in stages (push, dimer climb, both sides relaxed) and every search from the starting state | EON searches |
| `eon-outcomes` | The checks that end a search and the confidence rule | EON searches |
| `eon-step` | The rates, the random pick, and the waiting time of one kinetic Monte Carlo step | Kinetic Monte Carlo steps |
| `eon-run-directory` | The files an EON campaign writes | Output |
| `hero-animated` | The banner as an animation (12 s loop) | not used yet |
| `band-animation` | The band relaxing and the highest image climbing (10.5 s loop, 760 units wide) | not used yet |

## Computed data

Nine figures show output of ReactionFlow itself, written to `data/` by `make_data.py`:

- `detection`: `BondChangeDetector` and `ReactionTracker` with default settings on a prescribed
  C–N distance trace.
- `pathway`: `refine_pathway()` with default settings on a two-dimensional model potential. The
  band, the 0.622 eV barrier, the 476i cm⁻¹ mode, and the connectivity result come from that run;
  `band-animation` replays its optimizer steps.
- `ensembles`: SSNEB on the volume-coupled double well from `tests/test_ssneb.py` at 0, 0.4, and
  0.8 GPa.
- `restart`: the generic ASE adapter on 32 copper atoms with EMT in NPT at 600 K and 1 GPa,
  checkpointed to disk and restored at step 150.
- `models`: the catalog that `reactionflow models` prints. The build fails if a backend has no
  tile.
- `eon-landscape`, `eon-loop`, `eon-search`, and `eon-step` show one run of `run_exploration()`
  with the EON backend, default settings, and 12 kinetic Monte Carlo steps at 300 K on a
  two-dimensional landscape of Gaussian wells (`LANDSCAPE` in `make_data.py`, drawn by
  `src/landscape.mjs`). Every force evaluation is logged, so the searches from the starting state
  are drawn as EON ran them.

Each data file records the version and commit it was made from. Rerun the script when one of
these results could change, from the repository root and with ReactionFlow installed from this
checkout:

```bash
python -m pip install -e '.[eon]'
python docs/figures/make_data.py           # every data file
python docs/figures/make_data.py eon       # only eon.json
```

Only `eon.json` needs the EON backend, which has prebuilt wheels for Linux x86_64 only.

The banner trajectory is a short Langevin run on the model potential, computed in `src/toy.mjs`
and not by ReactionFlow. The timings in `campaign` and `how-it-works` are drawn by hand.

## Style

Every figure is 1200 units wide except `band-animation`. Text uses three sizes, 20, 22, and 26
units, set in `TYPE` in `src/lib.mjs`; at README width that is about 14 to 18 px, the same in every
figure. Colors are set in `THEMES` in the same file: blue for molecular dynamics, orange for
detection, green for pathway refinement. Explanations belong in the README text, not inside a
figure. Each figure's alt text is under `alt` in its file in `src/figures/`.
