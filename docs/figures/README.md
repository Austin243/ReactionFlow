# Figures

The figures in the main README are drawn by code in this folder. Each figure comes as a light and
a dark SVG, and the README shows the one that matches the reader's GitHub theme. Only the SVGs are
committed. The build can also write 2x PNGs to `png/` and a gallery page, `index.html`, with every
figure and a light/dark switch.

```bash
node docs/figures/build.mjs                    # every figure, SVG and PNG
node docs/figures/build.mjs md-mode eon-mode   # PNGs for the named figures only
node docs/figures/build.mjs --no-png           # SVG only; Chrome is not needed
```

The build needs Node 20 or newer and no packages. PNG export runs headless Google Chrome; set
`CHROME` to use another Chromium binary.

| Figure | Shows | Used in |
| --- | --- | --- |
| `overview` | MD on a model surface with the refined path, and the states, saddles, and kinetic Monte Carlo steps EON mode found on a model landscape, each with the loop it runs | README banner |
| `md-mode` | Bond detection on one atom pair, the refinement stages and barrier on a model surface, and how occurrences group into reaction classes | How it works |
| `eon-mode` | The searches from the starting state, the energy along one process, searches until the confidence target, one kinetic Monte Carlo step, and the energy against simulated time | Rare events with EON |
| `campaigns` | Built-in models by family, fixed-cell NEB against variable-cell SSNEB, an exact restart against a run that never stopped, and trajectories across a resubmitted job | Models |
| `hero` | An MD trajectory crossing between two basins and the refined path over the saddle | replaced by `overview` |
| `how-it-works` | One trajectory on a timeline: MD, bond monitor, checkpoint, refinement, restore | replaced by `overview` |
| `models` | The built-in model families with their model counts, the generic and custom adapters, and the commands | replaced by `campaigns` |
| `ensembles` | Fixed-cell NEB against variable-cell SSNEB, with enthalpy barriers at three pressures | replaced by `campaigns` |
| `campaign` | Eight trajectories on two GPU nodes across a resubmitted job | replaced by `campaigns` |
| `run-directory` | The files one trajectory writes | replaced by a text tree |
| `outcomes` | Each refinement stage and the status it ends with | replaced by a table |
| `detection` | The two distance thresholds and the persistence and stability counts for one atom pair | replaced by `md-mode` |
| `identity` | A reaction as a change graph and how occurrences group into classes | replaced by `md-mode` |
| `pathway` | Endpoint relaxation, NEB, climbing image, and saddle checks on a model surface | replaced by `md-mode` |
| `restart` | What a checkpoint holds, and a restored run compared with one that never stopped | replaced by `campaigns` |
| `phases` | The phases a trajectory records in `state.json` | not used yet |
| `eon-landscape` | The states and saddles EON mode found on a model landscape, the kinetic Monte Carlo steps, and the energy against simulated time | replaced by `overview` and `eon-mode` |
| `eon-loop` | Searches from the current state until the confidence target, then a kinetic Monte Carlo step, replayed from the start of the model run | replaced by `eon-mode` |
| `eon-search` | One EON search in stages (push, dimer climb, both sides relaxed) and every search from the starting state | replaced by `eon-mode` |
| `eon-outcomes` | The checks that end a search and the confidence rule | replaced by `eon-mode` and a table |
| `eon-step` | The rates, the random pick, and the waiting time of one kinetic Monte Carlo step | replaced by `eon-mode` |
| `eon-run-directory` | The files an EON campaign writes | replaced by a text tree |
| `hero-animated` | The banner as an animation (12 s loop) | not used yet |
| `band-animation` | The band relaxing and the highest image climbing (10.5 s loop, 760 units wide) | not used yet |

## Computed data

The README figures show output of ReactionFlow itself, written to `data/` by `make_data.py`.
`overview` draws on `pathway.json` and `eon.json`, `md-mode` on `detection.json` and
`pathway.json`, `eon-mode` on `eon.json`, and `campaigns` on `models.json`, `ssneb.json`, and
`restart.json`. The earlier figures use the same files.

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

The MD trajectory in `overview` and `hero` is a short Langevin run on the model potential,
computed in `src/toy.mjs` and not by ReactionFlow. The timings in `campaigns`, `campaign`, and
`how-it-works` are drawn by hand.

## Style

Every figure is 1200 units wide except `band-animation`. The four README figures have lettered
panels and text in three sizes, 24, 26, and 30 units, set in `HERO_TYPE` in `src/lib.mjs`. At
README width that is about 17 to 21 px, the size of the README text. The earlier figures use
`TYPE`, 20, 22, and 26 units. Colors are set in `THEMES` in the same file: blue for molecular
dynamics, orange for detection, green for pathway refinement. Explanations belong in the README
text, not inside a figure. Each figure's alt text is under `alt` in its file in `src/figures/`.
