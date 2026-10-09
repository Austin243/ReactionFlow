# EON adaptive kinetic Monte Carlo

MD finds reactions that happen within the simulated time. EON mode is for events too rare for
that: it runs adaptive kinetic Monte Carlo (AKMC) with EON's own process searches and the same
models as MD.

From the current state, each search gives the structure a random push, finds a saddle with the
dimer method, relaxes both sides of the saddle, keeps the result only if exactly one side is the
starting state, and computes harmonic transition-state (Vineyard) prefactors from the Hessians.
Searching continues until EON's confidence rule says the low-barrier exits have been found. A
kinetic Monte Carlo step then picks one exit with probability proportional to its rate,
`prefactor × exp(-barrier / kT)`, and advances the clock by a random residence time with mean
`1 / (sum of rates)`. The result is a sequence of reactions in simulated time, like an MD run but
for slower chemistry.

## CPU example

From the repository root, install the optional backend and run the bundled analytic example:

```sh
python -m pip install -e '.[eon]'
PYTHONPATH=examples/eon reactionflow validate examples/eon/campaign.json
PYTHONPATH=examples/eon reactionflow run examples/eon/campaign.json
reactionflow status examples/eon/campaign.json
reactionflow status examples/eon/campaign.json --json
```

EON mode requires `pyeonclient[ase]==0.4.2`. The supported prebuilt-wheel environment is
Linux x86_64; source builds on other platforms, including macOS, are not covered by the
integration check. See the
[EON Python client and ASE bridge](https://eondocs.org/user_guide/pyeonclient#ase-bridge)
for upstream installation and interface details. The example needs neither a GPU nor model weights.
Configuration validation and status inspection do not import the EON backend or the model.

[campaign.json](../examples/eon/campaign.json) selects `mode: "eon"` and loads
[toy.py](../examples/eon/toy.py) through the generic ASE calculator adapter. Structure and output
paths are relative to the campaign file. `PYTHONPATH` in these commands makes the example
calculator importable from the repository root. Outputs go to `outputs/eon-demo`.

The two atoms are labels for a mathematical model, not a helium force field. Atom 0 is fixed
in a periodic cell; atom 1 moves in three dimensions. With its displacement from the anchor
written as `(x, y, z)` in angstroms, the energy in eV is

```text
E = (x² - 1)² + 0.8 (x³/3 - x) + 8 (y² + z²).
```

| Process | Barrier (eV) | Harmonic prefactor (1/s) | Rate at 300 K (1/s) |
| --- | ---: | ---: | ---: |
| A (x = -1) → B (x = 1) | 0.5461 | 1.977 × 10¹³ | 1.32 × 10⁴ |
| B → A | 1.6128 | 2.421 × 10¹³ | 1.95 × 10⁻¹⁴ |

The saddle is at x = -0.2. With one free atom of mass 4.0026 u, each prefactor is the frequency
of the x vibration in the starting well. The run alternates between A and B: about 76 µs in A,
then about 1.6 million years in B.

## Models

The `adapter` block takes the same form as an MD adapter profile, so EON can use any built-in
model or the generic ASE calculator adapter. ANI-1xnr on one GPU, for example:

```json
"require_gpu": true,
"adapter": {
  "factory": "reactionflow.adapters.ani1xnr:create_adapter",
  "options": {"device": "cuda", "dtype": "float32", "model_index": 0, "strategy": "pyaev"}
}
```

`reactionflow models` lists the choices. `reactionflow prepare` downloads the model files,
`prepare --install` also installs the model packages and `pyeonclient`, and `run --download`
prepares the model before searching. As in MD campaigns, `require_gpu` defaults to true and
then requires exactly one visible CUDA device.

## Settings

`akmc`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `temperature_K` | 300 | Temperature of the rates |
| `steps` | 100 | Kinetic Monte Carlo steps to take |
| `confidence` | 0.95 | Search a state until 1 - 1/N reaches this, N being consecutive repeats of known processes in the window (EON's rule; its default is 0.99) |
| `thermal_window_kT` | 20 | Keep processes up to this many kT above the state's lowest barrier |
| `seed` | 1 | Seeds every search and every step |
| `basins` | `"none"` | `"bonds"` merges states joined without a bond change into one chemical basin, so `steps` counts only chemical steps; see Chemical basins |

`eon`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `displace` | `"all"` | Push every free atom, or `"local"`: only atoms near one chosen atom |
| `displace_size_A` | 0.5 | Total length of the random push, or how far a guided pair moves, in Å |
| `displace_radius_A` | 4.0 | Local push: atoms within this distance of the center move |
| `displace_centers` | `[]` | Local push: element symbols or atom indices the center is chosen from; empty means any free atom |
| `guide` | `null` | Push one atom pair of listed element types together or apart instead of the random push; see Guided searches |
| `force_tolerance` | 0.01 | Convergence of saddle searches and relaxations, eV/Å |
| `force_metric` | `"norm"` | What `force_tolerance` bounds: the norm of the whole force vector, or `"max_atom"`: the largest force on any atom |
| `max_iterations` | 1000 | Limit for each saddle search and relaxation |
| `max_energy_eV` | 20 | Abandon a saddle search this far above the state |
| `prefactor` | `null` | `null` computes harmonic prefactors; a number in 1/s fixes them |
| `match_distance_A` | 0.1 | Largest atom displacement between two structures counted as the same |
| `match_energy_eV` | 0.01 | Largest energy difference between two structures counted as the same |
| `equivalent_atoms` | true | Atoms of one element may exchange places, so a methyl group turned by 120° is the same state; not in a freely turning structure of two or more elements (see Symmetry) |
| `rigid_rotation` | `"auto"` | Treat turning the whole structure as no change; see Symmetry |
| `log` | `"file"` | EON's own log goes to `eon.log` in the output directory; `"terminal"` prints it |

The total push is spread over the moved atoms, so its size does not grow with the system.
EON computes harmonic prefactors from the atoms that move most in each process (those carrying
90% of the displacement) and rejects values outside 10⁹–10²¹ 1/s.

## Chemical basins

In molecules on surfaces and in molecular crystals, most exits from a state are conformational:
a methyl group turns, a molecule tilts or slides to a neighboring site. Their barriers are small,
so they dominate the rate table, every kinetic Monte Carlo step is short, and the clock rarely
reaches a reaction. With `basins: "bonds"`, two states belong to one chemical basin when a known
process joins them without forming or breaking a bond, using the same bond labels as `status`.

A step then leaves the part of the current basin that has been searched to the confidence target.
Its exits are the chemical processes of those states, and the bond-preserving processes into
conformers that have not been searched yet. The exit probabilities and the mean escape time come
from the absorbing Markov chain of the explored basin, as in EON's mean rate method, so the time
spent moving between conformers is counted without taking a step for each move. The time step is
drawn from an exponential distribution with that mean, which is exact for a single state and
assumes a memoryless escape when the basin has several. The basin states are eliminated with only
sums of positive numbers, so the result stays accurate when conformational rates exceed the
chemical ones by many orders of magnitude.

An exit into an unexplored conformer moves the run there, searches it, and adds it to the basin; it
is recorded as a step but does not count toward `steps`. Chemical exits count. Each step record
says which kind it was (`chemical`), how many states the basin held, the escape probability of the
chosen exit, and the mean escape time. When every state of the basin has been searched and none
has a chemical exit, the run stops with `no_chemical_exit`.

The thermal window applies among chemical exits only: a chemical process is kept when it lies
within `thermal_window_kT` of the lowest chemical barrier of its state, and bond-preserving
processes are always kept. Without basins, a 0.05 eV methyl rotation would push every reaction
more than 20 kT above it, 0.52 eV at 300 K, out of the rate table.

EON keeps a search only when one of the two minima beside its saddle is the searched state. In
clusters, liquids, and other soft systems the start often settles into a nearby conformer while
the saddle search climbs, for example a water molecule turning to a better hydrogen bond. Neither
minimum then matches the start, and EON rejects the search as not connected. With basins, such a
search still counts when one minimum keeps every bond of the searched state and the other forms or
breaks one: the process leads from the searched state to the second minimum. Its barrier is
measured from the lower of the searched state and the start-side minimum, so a better conformer
never makes a barrier look lower. Its prefactors come from the start-side minimum, or are the fixed
`prefactor`. The attempt records the start-side energy and the process keeps its structure.

## Guided searches

A random push seldom finds a reaction in a molecular system: most saddles near the start move
whole molecules, while a reaction needs two particular atoms to meet or part. When you know which
bonds the chemistry makes or breaks, name their element pairs and each search pushes one such pair
instead:

```json
"eon": {"guide": {"form": ["C-N", "C-C"], "break": ["C-H"], "within_A": 3.5}}
```

A search picks one pair at random among all that match. A pair to form is not bonded, is closer
than `within_A` (3.5 Å by default), and has no bonded neighbor in common, which leaves out atoms
already joined through a third. A pair to break is bonded. Bonds use the default distances of MD
reaction detection. The pair moves `displace_size_A` closer or apart, each atom in proportion to
the other's mass and a fixed atom not at all, and a random push of 0.1 Å in total over the free
atoms makes a pair picked again start differently. The dimer starts along the pair's motion. The
rest is EON's own search: both sides of the saddle are relaxed 0.2 Å along its mode, compared
with the start, and given prefactors. The potential decides what each push becomes, so a guide
only says which bonds to try. The attempt records the pair it pushed.

In clusters and liquids the start usually settles into a nearby conformer during a guided search,
so use `guide` with `basins: "bonds"` (see Chemical basins).

## Symmetry

A motion that changes nothing physical must not count as a new state or a new process, and it has
no vibrational frequency.

- Translation. When no atom is fixed, EON removes the net force and compares structures without
  their overall translation. Periodic cells and slabs need nothing more.
- Rotation. A nonperiodic structure with no fixed atom, such as a molecule or cluster in vacuum,
  can turn freely. With `rigid_rotation: "auto"`, saddle searches then project rotation out of each
  step and structures are compared up to a rotation. EON can do this only when no atom is fixed,
  so fixed atoms fewer than three, or all on one line, leave a rotation free; `validate` and `run`
  warn about that case. EON's comparison up to both a rotation and exchanged atoms fails for two or
  more elements, so such a structure is compared without `equivalent_atoms`.
- Harmonic prefactors require a structure that cannot turn as a whole. A free molecule's rate also
  depends on how its rotations change between the minimum and the saddle, which EON does not
  compute, so such a campaign needs a fixed `prefactor`; `init` asks for one.
- Symmetry-equivalent sites give separate processes with equal barriers. Their rates add, which
  is correct.

## Reports

`status` prints the clock, the steps taken, and the current state's confidence and processes. Each
step and process is labeled with the bonds it forms and breaks, using the default bond distances
of MD reaction detection: for example `formed C-N; broken C-H`, or `no bond change` for a
conformational change. EON's own log of each search goes to `eon.log` in the output directory.

## Resume and records

After interruption, rerun the identical `run` command. Completed searches and steps are reused
exactly once, and an interrupted search is repeated with its saved seed. Keeping the same seed
does not promise bitwise agreement across EON builds or numerical environments.

The output directory holds the model/settings contract in `search-contract.json`, progress in
`search-state.json`, and immutable checksummed records under `states/`, `processes/`,
`attempts/`, and `steps/`. A process record keeps its saddle, product, barrier, prefactors, and
bond changes; the reverse of each process is recorded for the product state. An attempt that EON
rejected after its saddle converged keeps the saddle, both minima with their energies, and the
bonds each minimum forms and breaks relative to the searched state; `status` counts them. With
basins, such a search can still become a process (see Chemical basins). One process writes an
exploration at a time.

Resume checks bind the starting structure, settings, the model identity that an MD restart also
records, and the search code. To compare different settings, copy the campaign and choose a new
output directory.

## Supported scope

- Harmonic transition-state theory: prefactors come from vibrations at the minimum and saddle,
  which suits barriers well above kT and stiff wells.
- A search that returns to the same state, such as equivalent atoms trading places, does not change
  the state-to-state kinetics. It is recorded as a search but kept out of the rate table and the
  confidence count.
- Without `basins`, two states joined by a very low barrier make every step short, and such pairs
  show up in the step list. `basins: "bonds"` merges states only when no bond changes between
  them; a low-barrier reaction is still a separate step.
- A state is searched until it reaches the confidence target, with no cap on searches; `status`
  shows the searches and confidence of the current state.
- Searches use a fixed, nonsingular cell, fully periodic or fully nonperiodic, with `FixAtoms` as
  the only constraint.
- By default EON converges saddles and minima on the norm of the whole force vector
  (`force_tolerance`, 0.01 eV/Å), which adds up the small residual forces on every free atom. In
  cells of a few hundred free atoms that norm can stay above the tolerance while no atom carries
  more than a few meV/Å, and TF32-style fast arithmetic alone leaves that much force noise. There,
  converge on the largest atomic force (`force_metric: "max_atom"`, as ASE's `fmax` does), for
  example at 0.05 eV/Å, and run the model in float64 or full float32.
- Searches climb from a small push along the softest mode, which suits solids and surfaces. In
  liquids the softest modes are collective solvent motions that lead to no saddle, so searches
  rarely connect; use the MD workflow there.
- Small periodic cells admit collective processes, such as a whole surface row sliding through the
  boundary. Check which atoms move in the lowest barriers before trusting the kinetics.
- The analytic example verifies software behavior; it does not validate a model for real
  chemistry.
