# Models and preparation

ReactionFlow includes optional MACE, MACE-FIELD, UMA, AIMNet2, ORB, MatterSim, CHGNet, SevenNet,
NEP89, and ANI-1xnr adapters. Other models can use the
[generic ASE calculator adapter or a custom adapter](campaigns.md#mlip-adapter). Model selection
uses the campaign's named `adapter_profiles`, each assigned to trajectories by name.

List supported models, heads, tasks, and their intended domains without installing any backend:

```bash
reactionflow models
reactionflow models --backend mace
reactionflow models --backend uma
```

The command prints JSON and never downloads weights. These are model choices, not a universal
accuracy ranking: molecular reaction paths, surfaces, crystals, and electric-field response
require different validation.

| Backend | Included choices |
| --- | --- |
| `mace` | MACE-MH-1 (six heads), MH-0, MP/MPA/OMAT/MatPES, OFF23 sizes, POLAR-1 small/medium/large |
| `mace_field` | MACEField-MH-0-omat-dielectric, three heads, fixed external electric field |
| `uma` | Small 1.2.1/1.2/1.1 and medium 1.1; all seven published task names |
| `aimnet2` | General AIMNet2, 2025, B97-3c, NSE, Pd, RXN; ensemble members 0–3 |
| `orb` | OrbMol-v2; conservative ORB-v3 `-inf` OMAT and MPA |
| `mattersim` | MatterSim-v1.0.0-1M and -5M |
| `chgnet` | MPtrj 0.3.0 and r2SCAN |
| `sevennet` | SevenNet-0, L3i5, MF-0, MF-ompa, OMAT, and Omni/Omni-i8/Omni-i12, with explicit modalities |
| `nep` | NEP89 (89 elements), double-precision CPU inference |

## Install and prepare

Activate an environment for the chosen backend and install ReactionFlow from this checkout:

```bash
python -m pip install .
reactionflow prepare campaign.json --install
```

The optional `--install` runs pip in the current Python environment for the referenced built-in
backends. It then starts a fresh Python process to
prepare the model files, because pip may have changed already imported dependencies. Installation
or preparation failures return a failing exit status. It does not create an environment for you.

The PyPI backend pins are `mace-torch==0.3.16`, `fairchem-core==2.23.0`, `aimnet[ase]==0.2.0`,
`orb-models==0.7.0`, `mattersim==1.2.5`, `chgnet==0.4.2`, `sevenn==0.13.0`, and `calorine==4.0`.
SevenNet setup also installs `torch>=2.8,<3`. POLAR additionally installs
`graph_longrange` 0.4.0 at commit `0e21d5546c482d08388a08eb4d948e833227ce47`; MACE-FIELD installs
its fork at commit `136e4ef040d7c51a5b051a7a609ffb1f29307478`. Those two source dependencies
require Git. They are installed by `prepare --install`, not embedded as direct Git dependencies
in ReactionFlow's package metadata.

Alternatively, install a PyPI backend extra yourself: `.[mace]`, `.[uma]`, `.[aimnet2]`, `.[orb]`,
`.[mattersim]`, `.[chgnet]`, `.[sevennet]`, or `.[nep]`, then omit `--install`:

```bash
reactionflow prepare campaign.json
reactionflow prepare campaign.json --index 2
```

By default, preparation visits each distinct adapter configuration referenced by a trajectory once.
`--index` selects only that trajectory's configuration and can also be combined with `--install`.
Preparation resolves/downloads model files and reports their paths without constructing a
calculator, starting MD, or creating trajectory output. Unreferenced profiles are ignored.

Keep incompatible backends in separate environments and campaigns. The pinned MACE package needs
`e3nn==0.4.4`, whereas UMA, MatterSim, and SevenNet need `e3nn>=0.5`. MACE-FIELD also needs e3nn 0.4.4 and
replaces the same `mace` Python namespace as official MACE; it must use a separate environment.
UMA and ORB require incompatible `nvalchemi-toolkit-ops` versions. The tested MatterSim setup
also uses a newer version through TorchSim; setup keeps MatterSim and ORB separate too.
UMA's
FAIR-Chem version also requires Torch 2.13, while the ANI-1xnr setup pins Torch 2.11. Installing UMA
into the ANI environment would change its dependencies and invalidate exact restarts. The bundled
`setup-perlmutter-ani1xnr.sh` and Perlmutter job scripts configure that ANI environment; they are
not setup scripts for MACE or UMA. Use your backend's environment in your own batch script.
`--install` rejects these unsupported combinations before running pip; select one backend
with `--index` in a fresh environment.

Once prepared, normal built-in runs use only local files and do not download weights or install
packages:

```bash
reactionflow run campaign.json --index 0
```

For an explicit download immediately before running the selected trajectory, use
`reactionflow run campaign.json --index 0 --download`. This convenience never runs pip. On a
cluster, prepare on a network-enabled node first and make the cache visible to compute nodes.
`validate`, `plan`, and `status` remain independent of optional model imports and downloads.

ANI-1xnr and arbitrary ASE/custom adapters manage their own dependencies and files. `prepare`
reports them as `external_setup` without executing their code; `run --download` rejects them with
an unsupported-adapter error. The existing ANI setup workflow is unchanged.

## MACE profile

This is a profile block to include in a campaign's `adapter_profiles`:

```json
{
  "adapter_profiles": {
    "mace-materials": {
      "factory": "reactionflow.adapters.mace:create_adapter",
      "options": {
        "family": "mp",
        "model": "mh-1",
        "head": "omol",
        "device": "cuda",
        "dtype": "float64"
      }
    }
  }
}
```

Set a trajectory's `adapter_profile` to `mace-materials`. `family` is `mp` (default), `off`, or `polar`;
`model` must be an explicit name from that family's
[pinned MACE registry](https://github.com/ACEsuit/mace/blob/v0.3.16/mace/calculators/foundations_models.py).
For example, `family: "off"` with `model: "small"` selects MACE-OFF23 small. `device` is `cuda`
(default) or `cpu`, and `dtype` is `float64` (default) or `float32`. An optional `head` selects a
named head supported by that checkpoint; the selected head is recorded in the restart contract.
All MD and refinement stages use the same model, head, and precision. Named MH-0/MH-1 require
an explicit `head`; the loaded checkpoint must actually contain it.

The released MH-1 checkpoint contains:

| Head | Training domain |
| --- | --- |
| `omat_pbe` | OMat24, inorganic materials |
| `omol` | OMol25 neutral subset, molecules and organometallics |
| `spice_wB97M` | SPICE molecular chemistry |
| `oc20_usemppbe` | OC20 surfaces and adsorbates |
| `matpes_r2scan` | MatPES r²SCAN materials |
| `mp_pbe_refit_add` | Materials Project trajectories |

`rgd1_b3lyp` is absent from the current MH-1 artifact despite older release text; the maintainer
directs users to MH-0 for that head. See the [upstream correction](https://github.com/ACEsuit/mace/discussions/1462).
This adapter does not add an extra D3 correction to MACE predictions.

### MACE-POLAR sizes

```json
{
  "factory": "reactionflow.adapters.mace:create_adapter",
  "options": {"family": "polar", "model": "polar-1-s", "device": "cuda", "dtype": "float64"}
}
```

Choose `polar-1-s`, `polar-1-m`, or `polar-1-l`. Structures need explicit integer `charge` and
positive integer `spin` (multiplicity) in `atoms.info`. Optional `atoms.info['external_field']` is a finite
three-vector in V/Å; omitted means zero. Changes to these inputs invalidate cached predictions.

NPT and variable-cell refinement are supported with `dtype: "float64"`. The pinned upstream
stress omits part of the long-range response, matching
[issue #1642](https://github.com/ACEsuit/mace/issues/1642). ReactionFlow instead differentiates
the full model energy with respect to cell strain using central differences of size `1e-5`,
holding fractional coordinates and the external field fixed. Forces remain analytic. Each fresh
stress needs twelve additional energy evaluations; the method and strain size are recorded in the
restart contract. Float32 remains available for fixed-cell NVT (`pressure_GPa: null`), but is
rejected when stress is requested because numerical differentiation needs the extra precision.

Real Polar-S checks cover all six stress components in a triclinic cell at nonzero field,
comparison against an independent energy-strain calculation, and exact short NPT restarts.
The [PolarMACE guide](https://mace-docs.readthedocs.io/en/latest/guide/polar_mace.html)
describes the physical inputs; its current dependency recommendation does not match MACE 0.3.16,
so preparation installs the separately tested graph_longrange 0.4.0 revision above.

### MACE-FIELD

```json
{
  "factory": "reactionflow.adapters.mace_field:create_adapter",
  "options": {
    "model": "MACEField-MH-0-omat-dielectric",
    "head": "mp-dielectric",
    "electric_field": [0.0, 0.0, 0.02],
    "device": "cuda",
    "dtype": "float64"
  }
}
```

The published checkpoint has `pt_head`, `mp-dielectric`, and `mp-ferroelectric` heads. Both `head`
and a finite `electric_field` three-vector in V/Å are explicit options, including a zero field.
The field stays fixed across MD, endpoint relaxation, and NEB and is part of the restart contract.
The structure needs a cell with nonzero volume for the dielectric observables. Reported energies
are electric enthalpies at the configured field; NPT refinement additionally includes PV.
The fork's source revision is checked; official MACE cannot substitute for it. See the
[MACE-FIELD release](https://github.com/mdi-group/mace-field/releases/tag/1.0.2).

Choose a model for its training domain and intended observables. The
[MACE foundation-model guide](https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html)
describes coverage and limitations. Weight licenses differ across families/checkpoints; follow
the license and citation instructions linked from the selected model's official release.

## UMA profile

```json
{
  "adapter_profiles": {
    "uma-molecules": {
      "factory": "reactionflow.adapters.uma:create_adapter",
      "options": {
        "model": "uma-s-1p2p1",
        "task": "omol",
        "device": "cuda"
      }
    }
  }
}
```

`model` must name a UMA checkpoint in the pinned FAIR-Chem registry. `task` is required and selects
the scientific domain/level of theory, for example `omol` for molecules or `omat` for inorganic
materials. See the [official UMA guide](https://github.com/facebookresearch/fairchem/blob/main/docs/core/uma.md)
for the available tasks and their domains. `device` is `cuda` (default) or `cpu`. Optional
`revision` selects a Hugging Face revision; use an immutable commit for reproducible fresh
downloads. Preparation fetches the checkpoint and companion reference files from the same resolved
repository revision.

| Task | Domain | Checkpoint versions |
| --- | --- | --- |
| `omol` | Molecules | 1.1, 1.2, 1.2.1 |
| `omat` | Inorganic materials | 1.1, 1.2, 1.2.1 |
| `omc` | Molecular crystals | 1.1, 1.2, 1.2.1 |
| `odac` | CO₂/H₂O adsorption in MOFs | 1.1, 1.2, 1.2.1 |
| `oc20` | Heterogeneous catalysis | 1.1, 1.2, 1.2.1 |
| `oc22` | Oxide catalysis | 1.2, 1.2.1 |
| `oc25` | Electrolyte/inorganic interfaces | 1.2, 1.2.1 |

The available model identifiers are `uma-s-1p2p1`, `uma-s-1p2`, `uma-s-1p1`, and `uma-m-1p1`.

For `task: "omol"`, the input structure must contain explicit integer `charge` and positive
integer `spin` in `atoms.info`. `spin` means multiplicity: 1 for a singlet, 2 for a doublet, and so
on. Save that metadata in the campaign's structure file, for example:

```python
from ase.io import read, write

atoms = read("molecule.xyz")
atoms.info.update(charge=0, spin=1)
write("structure.extxyz", atoms)
```

NPT and variable-cell refinement require stress. OMAT has stress training; OMOL was trained on
aperiodic molecular data, although stress can be derived from energy gradients. A callable stress
does not by itself establish accuracy for periodic/high-pressure molecular systems. Choose and
validate the task for your system; use `pressure_GPa: null` for fixed-cell NVT. See FAIR-Chem's
[task guidance](https://github.com/facebookresearch/fairchem/blob/main/docs/core/uma.md) and
[gradient-based stress explanation](https://facebookresearch.github.io/fairchem/fine-tuning/).

UMA weights require the access conditions on the [official model page](https://huggingface.co/facebook/UMA).
Complete that one-time access step and authenticate with `hf auth login` or an existing `HF_TOKEN`
before downloading. ReactionFlow uses Hugging Face's authentication and does not put credentials
in campaign files or checkpoints. No token is needed to run already prepared local files.

## AIMNet2 and RXN

```json
{
  "factory": "reactionflow.adapters.aimnet2:create_adapter",
  "options": {"model": "aimnet2-rxn", "model_index": 0, "device": "cuda"}
}
```

Choose `aimnet2`, `aimnet2-2025`, `aimnet2-b973c`, `aimnet2-nse`, `aimnet2-pd`, or `aimnet2-rxn`.
Each has `model_index` 0–3 (default 0). RXN targets neutral H/C/N/O reactive systems; NSE supports
open shells; the other families require singlets. Explicit integer `charge` and positive integer
`spin` multiplicity are required in `atoms.info`; if `mult` is present it must agree with `spin`.
Unsupported species and electronic states fail before a new CLI trajectory is created.

The ASE backend uses float32. `coulomb_method` is `auto` (default), `dsf`, `ewald`, or `pme`;
auto uses molecular Coulomb for nonperiodic inputs and DSF for periodic ones. Ewald/PME require
fully periodic inputs. NPT preflight checks finite stress as well as energy and forces. See the
[official model guide](https://github.com/isayevlab/aimnetcentral) for reference energies,
element coverage, and each family's scientific limits; changing families changes the potential.

## OrbMol-v2 and ORB-v3

```json
{
  "factory": "reactionflow.adapters.orb:create_adapter",
  "options": {"model": "orbmol-v2", "device": "cuda", "precision": "float32-highest"}
}
```

Other choices are `orb-v3-conservative-inf-omat` and `orb-v3-conservative-inf-mpa` for materials.
Only conservative energy-gradient forces are included. `precision` is `float32-highest` (default)
or `float64`; compilation is disabled. OrbMol-v2 requires explicit integer `charge` and positive
integer `spin` multiplicity, and either fully periodic or fully nonperiodic inputs. See the
[upstream model descriptions](https://github.com/orbital-materials/orb-models/blob/v0.7.0/MODELS.md).

## MatterSim and CHGNet

```json
{
  "factory": "reactionflow.adapters.mattersim:create_adapter",
  "options": {"model": "mattersim-v1.0.0-1M", "device": "cuda", "dtype": "float32"}
}
```

MatterSim also offers `mattersim-v1.0.0-5M`; `dtype` is `float32` (default) or `float64`.
The versioned checkpoint download is tied to an immutable release commit. See
[MatterSim](https://github.com/microsoft/mattersim) for its materials training domain.

```json
{
  "factory": "reactionflow.adapters.chgnet:create_adapter",
  "options": {"model": "r2scan", "device": "cuda"}
}
```

CHGNet also offers `model: "0.3.0"` for MPtrj PBE+U. Its weights are already included in the
installed package, so preparation just resolves and verifies the local file. See
[CHGNet's releases](https://github.com/CederGroupHub/chgnet/releases/tag/v0.4.2).
Both materials adapters require fully periodic structures with a 3D cell; CHGNet rejects
isolated atoms. These are additional materials choices, not general molecular reaction models.

## SevenNet

```json
{
  "factory": "reactionflow.adapters.sevennet:create_adapter",
  "options": {"model": "7net-omni", "modal": "mpa", "device": "cuda"}
}
```

The released model names and selectable modalities are:

| Model | `modal` |
| --- | --- |
| `7net-0`, `7net-0_22may2024`, `7net-l3i5`, `7net-omat` | Omit; these have a single task |
| `7net-mf-0` | `PBE`, `R2SCAN` (case-sensitive) |
| `7net-mf-ompa` | `mpa`, `omat24` |
| `7net-omni`, `7net-omni-i8`, `7net-omni-i12` | `mpa`, `omat24`, `matpes_pbe`, `matpes_r2scan`, `mp_r2scan`, `oc20`, `oc22`, `odac23`, `omol25_low`, `omol25_high`, `spice`, `qcml`, `pet_mad` |

Multitask models require an explicit `modal`; the loaded checkpoint must contain it. This adapter
uses conservative energy-gradient forces/stress in float32 with the e3nn backend and no extra D3.
Acceleration overrides are rejected so the execution contract stays explicit. Charge and spin
are not explicit calculator inputs. SevenNet-0 (both dates), L3i5, and MF-0 resolve package-bundled weights locally;
OMAT, MF-ompa, and Omni variants download from versioned upstream releases during preparation.
See [SevenNet's model guide](https://sevennet.readthedocs.io/en/latest/user_guide/pretrained.html)
for task coverage and scientific limitations.

## NEP89

```json
{
  "factory": "reactionflow.adapters.nep:create_adapter",
  "options": {"model": "nep89", "device": "cpu"}
}
```

Set campaign `require_gpu: false`. NEP89 uses Calorine's in-memory CPU calculator in float64;
it has no Torch dependency. Preparation downloads the official 2025-04-09 NEP89 energy model
from an immutable GPUMD release commit. Calorine may need a C++17 compiler during installation.
An absolute `checkpoint` can instead select another local NEP energy-model text file.

This backend always applies periodic boundaries, so ReactionFlow requires fully periodic
structures with a finite 3D cell, including for NVT. Nonzero `atoms.info['charge']`, nonsinglet
`atoms.info['spin']`, and nonzero ASE initial charge/magnetic-moment arrays are rejected because
the model does not take them as inputs. Conservative forces and stress support NPT and variable-cell refinement. The native
library binary, calculator source, and weight hashes are bound to exact restarts. See
[the NEP89 release](https://github.com/brucefan1983/GPUMD/tree/v5.0/potentials/nep/nep89_20250409)
and [Calorine's calculator documentation](https://calorine.materialsmodeling.org/get_started/ase_calculators.html).

## Local files and cache

MACE, MACE-FIELD, UMA, AIMNet2, MatterSim, CHGNet, SevenNet, and NEP accept an absolute `checkpoint` path instead
of `model`; supply exactly one. For ORB, retain `model` to select its architecture and optionally
provide `checkpoint` to override the weights. AIMNet local files use current metadata-bearing `.pt`
format; `model_index` is only valid with named models.
For UMA local checkpoints, also supply the applicable `atom_refs` and `form_elem_refs` YAML files
from the same model release, for example:

```json
{
  "checkpoint": "/models/uma/uma-s-1p2p1.pt",
  "atom_refs": "/models/uma/iso_atom_elem_refs.yaml",
  "form_elem_refs": "/models/uma/form_elem_refs.yaml",
  "task": "omol",
  "device": "cuda"
}
```

Local paths must be absolute and visible on the execution node. `revision` applies only to named
UMA models. Reference-path options apply only to local UMA checkpoints; named models resolve their
reference files automatically.

The default download cache is `~/.cache/reactionflow/models`, with separate backend directories.
Set `REACTIONFLOW_MODEL_CACHE` to an absolute directory, or set an absolute `cache_dir` in the
adapter options to override it for that profile. Keep the same cache/path configuration when
resuming. Direct downloads verify cached file hashes; UMA uses the Hugging Face cache. CHGNet
uses package-bundled files and has no `cache_dir` option. Exact checkpoints record hashes of
every resolved weight/reference file for all adapters.

## Execution and restart scope

The built-in adapters use the shared Langevin BAOAB NVT/NPT runtime and the selected calculator for
every pathway stage. For CPU runs, set `device: "cpu"` and campaign `require_gpu: false`; a CUDA
worker must see exactly one usable GPU.

Exact resume binds model files, options, package versions, adapter/runtime source, and the recorded
Torch/platform/device environment (or the native Calorine binary for NEP). Changing those inputs requires a new trajectory output rather
than treating it as exact continuation. Preserve the environment and files of an active campaign.
For Torch backends, deterministic algorithms are enabled and TF32 is disabled; unsupported nondeterministic
operations fail. UMA uses fixed float32 inference without compilation or MoLE merging, so its
upstream accelerated presets do not describe this adapter's performance.

The adapter tests cover setup, configuration, and restart contracts with controlled calculators.
Opt-in integration tests use prepared real weights without downloads. CPU checks include finite
energy/forces and two-step versus one-step/fresh-restore/one-step comparisons for MACE-MP small,
AIMNet2/RXN (NVT and NPT), OrbMol-v2 (NVT and NPT), MACE-FIELD (NPT), MatterSim 1M (NPT), and
CHGNet 0.3.0/r2SCAN (NPT), SevenNet-0/MF-0/Omni (NPT), and NEP89 (NPT).
All six MH-1 heads have finite CPU energy/force checks; Polar-S has force, numerical-stress,
NVT, and NPT restart checks.
These short checks do not establish GPU reproducibility, real gated UMA behavior, every model
variant, or scientific accuracy for a new system. Model integration alone is not validation of a
reaction barrier or stress model.
