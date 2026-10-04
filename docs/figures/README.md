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

## Computed data

Every panel shows output of ReactionFlow itself, written to `data/` by `make_data.py`, except
three. The MD trajectory in `overview` is a short Langevin run on the model potential, computed in
`src/toy.mjs`. The change graphs in `md-mode` d are a diagram of the hydrogen shift between
acetonitrile and ketenimine, and the job timings in `campaigns` d are drawn by hand.

- `detection.json` is one C–N pair passed through `BondChangeDetector` and `ReactionTracker` with
  default settings (`md-mode` a).
- `pathway.json` is a `refine_pathway()` run with default settings on a two-dimensional model
  potential (`overview` a, `md-mode` b and c). The band, the 0.622 eV barrier, the 476i cm⁻¹ mode,
  and the connectivity result come from that run.
- `eon.json` is one run of `run_exploration()` with the EON backend, default settings, and 12
  kinetic Monte Carlo steps at 300 K on a two-dimensional landscape of Gaussian wells
  (`LANDSCAPE` in `make_data.py`, drawn by `src/landscape.mjs`). Every force evaluation is logged,
  so the searches from the starting state are drawn as EON ran them (`overview` b, `eon-mode`).
- `models.json` is the catalog that `reactionflow models` prints (`campaigns` a). The build fails
  if a backend has no name in `campaigns.mjs`.
- `ssneb.json` is SSNEB on the volume-coupled double well from `tests/test_ssneb.py` at 0, 0.4, and
  0.8 GPa (`campaigns` b).
- `restart.json` is the generic ASE adapter on 32 copper atoms with EMT in NPT at 600 K and 1 GPa,
  checkpointed to disk and restored at step 150 (`campaigns` c).

Each data file records the version and commit it was made from. Rerun the script when one of
these results could change, from the repository root and with ReactionFlow installed from this
checkout:

```bash
python -m pip install -e '.[eon]'
python docs/figures/make_data.py           # every data file
python docs/figures/make_data.py eon       # only eon.json
```

Only `eon.json` needs the EON backend, which has prebuilt wheels for Linux x86_64 only.

## Style

Every figure is 1200 units wide and has lettered panels. Text uses three sizes, 24, 26, and 30
units, set in `TYPE` in `src/lib.mjs`. At README width that is about 17 to 21 px, the size of the
README text. Colors are set in `THEMES` in the same file: blue for molecular dynamics, orange for
detection, green for pathway refinement. Explanations belong in the README text, not inside a
figure. Each figure's alt text is under `alt` in its file in `src/figures/`.
