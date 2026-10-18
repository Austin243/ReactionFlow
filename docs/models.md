# Models and preparation

ReactionFlow includes optional MACE, UMA, AIMNet2, and ANI-1xnr adapters. Other models, among them
ORB, MatterSim, CHGNet, SevenNet, and NEP89, run through the
[generic ASE calculator adapter](#other-models-through-the-ase-adapter), and a model with state
of its own can use a [custom adapter](campaigns.md#write-a-custom-adapter). Model selection
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
| `uma` | Small 1.2.1/1.2/1.1 and medium 1.1; all seven published task names |
| `aimnet2` | General AIMNet2, 2025, B97-3c, NSE, Pd, RXN; ensemble members 0–3 |
| `ani1xnr` | ANI-1xnr reactive H/C/N/O potential; ensemble members 0–7 |

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
and `torchani==2.8.4`. ANI-1xnr setup also installs `torch==2.11.0`. POLAR additionally installs
`graph_longrange` 0.4.0 at commit `0e21d5546c482d08388a08eb4d948e833227ce47`. That source
dependency requires Git. It is installed by `prepare --install`, not embedded as direct Git dependencies
in ReactionFlow's package metadata.

Alternatively, install a PyPI backend extra yourself: `.[mace]`, `.[uma]`, `.[aimnet2]`, or
`.[ani1xnr]`, then omit `--install`:

```bash
reactionflow prepare campaign.json
reactionflow prepare campaign.json --index 2
```

By default, preparation visits each distinct adapter configuration referenced by a trajectory once.
`--index` selects only that trajectory's configuration and can also be combined with `--install`.
Preparation resolves/downloads model files and reports their paths without constructing a
calculator, starting MD, or creating trajectory output. Unreferenced profiles are ignored.

Keep incompatible backends in separate environments and campaigns. The pinned MACE package needs
`e3nn==0.4.4`, whereas UMA needs `e3nn>=0.5`. UMA's FAIR-Chem version also requires Torch 2.13, while ANI-1xnr pins Torch 2.11, so those two need
separate environments as well. The bundled
`setup-perlmutter-ani1xnr.sh` and Perlmutter job scripts configure an ANI-1xnr environment; they
are not setup scripts for the other backends. Use your backend's environment in your own batch
script.
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
`validate` and `status` remain independent of optional model imports and downloads.

Arbitrary ASE and custom adapters manage their own dependencies and files. `prepare` reports them
as `external_setup` without executing their code; `run --download` rejects them with an
unsupported-adapter error.

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
A nonzero field has a direction in the laboratory frame, which pathway refinement and EON must
keep: under pressure the cell must be lower-triangular (as `cell.standard_form()` gives) with
hydrostatic NPT, and EON needs `rigid_rotation: false` for a freely turning structure.

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

## ANI-1xnr

```json
{
  "factory": "reactionflow.adapters.ani1xnr:create_adapter",
  "options": {"device": "cuda", "dtype": "float32", "model_index": 0, "strategy": "pyaev"}
}
```

ANI-1xnr is a reactive potential for H, C, N, and O trained at the BLYP/TZV2P level. The adapter
always loads one pinned weight file, so it has no `model` or `checkpoint` option. Preparation
downloads that file from a fixed revision of the
[`roitberg-group/ani1xnr`](https://huggingface.co/roitberg-group/ani1xnr) repository and checks
its SHA-256; the adapter checks it again when a worker starts. `model_index` selects one of
the eight ensemble members, `dtype` is `float32` (default) or `float64`, and `strategy` is `pyaev`
(default) or `cuaev`. The adapter requires `torch` 2.11.0 and `torchani` 2.8.4. See the
[ANI-1xnr paper](https://www.nature.com/articles/s41557-023-01427-3) for the training data and
scope.

## Local files and cache

MACE, UMA, and AIMNet2 accept an absolute `checkpoint` path instead
of `model`; supply exactly one. AIMNet local files use current metadata-bearing `.pt`
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
adapter options to override it for that profile. The bundled Perlmutter scripts set
`REACTIONFLOW_MODEL_CACHE` to `.cache/models` inside the checkout unless it is already set. Keep
the same cache/path configuration when
resuming. Direct downloads verify cached file hashes; UMA uses the Hugging Face cache. ANI-1xnr
keeps its weights in TorchANI's `StateDicts` layout inside its backend directory. Exact checkpoints record hashes of
every resolved weight/reference file for all adapters.

## Execution and restart scope

The built-in adapters use the shared Langevin BAOAB NVT/NPT runtime and the selected calculator for
every pathway stage. For CPU runs, set `device: "cpu"` and campaign `require_gpu: false`; a CUDA
worker must see exactly one usable GPU.

Each checkpoint records the model files, options, package versions, adapter/runtime source, and the
Torch/platform/device environment. A changed model file or option refuses the resume. A changed
version, source hash, or device continues with a warning and a line in
`restart-environment.jsonl`, because the run is then no longer bitwise exact
([exact restart](exact-restart.md)). Keep the environment of an active campaign when you can.
For Torch backends, deterministic algorithms are enabled and TF32 is disabled; unsupported nondeterministic
operations fail. UMA uses fixed float32 inference without compilation or MoLE merging, so its
upstream accelerated presets do not describe this adapter's performance. The adapter makes two
UMA steps deterministic: expert routing sums each structure's composition average in a fixed
order, because fairchem's reduction has no deterministic CUDA kernel, and each forward pass draws
its random edge-rotation angles from the same seed.

The adapter tests cover setup, configuration, and restart contracts with controlled calculators.
Opt-in integration tests use prepared real weights without downloads. CPU checks include finite
energy/forces and two-step versus one-step/fresh-restore/one-step comparisons for MACE-MP small,
and AIMNet2/RXN (NVT and NPT).
All six MH-1 heads have finite CPU energy/force checks; Polar-S has force, numerical-stress,
NVT, and NPT restart checks.
UMA-S 1.1, 1.2 and 1.2.1 (`omol`, and 1.2.1 `omc`) ran with real weights on an A100: energies agree
with fairchem's own CPU reduction within 1e-6 eV, and repeated and fresh evaluations are bitwise
identical on GPU and CPU.
These short checks do not establish GPU reproducibility for the other backends, every model
variant, or scientific accuracy for a new system. Model integration alone is not validation of a
reaction barrier or stress model.

## Other models through the ASE adapter

Any model with an ASE calculator runs through `reactionflow.adapters.ase:create_adapter`
([options](campaigns.md#use-an-ase-calculator-directly)). Install the model's own package, download
its weights yourself, and list them in `model_files` so each checkpoint records their hashes. For a
Torch model, set `torch_device` to `cpu` or `cuda`. ReactionFlow then turns on deterministic Torch
inference before building the calculator and records the Torch environment in each checkpoint, as
the built-in adapters do; without it, restarts of Torch models are not bitwise exact. The profiles
below ran two NPT steps of crystalline Si on a CPU, and a restart from step one matched the
uninterrupted run bit for bit, except for ORB (below). Replace `cpu` with `cuda` on a GPU.

CHGNet 0.3.0, whose weights come with `chgnet`:

```json
"chgnet": {
  "factory": "reactionflow.adapters.ase:create_adapter",
  "options": {
    "calculator_factory": "chgnet.model.dynamics:CHGNetCalculator",
    "calculator_kwargs": {"use_device": "cpu"},
    "torch_device": "cpu"
  }
}
```

SevenNet-0, whose weights come with `sevenn` (for a multitask model, add its `modal`):

```json
"sevennet": {
  "factory": "reactionflow.adapters.ase:create_adapter",
  "options": {
    "calculator_factory": "sevenn.calculator:SevenNetCalculator",
    "calculator_kwargs": {"model": "7net-0", "device": "cpu"},
    "torch_device": "cpu"
  }
}
```

MatterSim, with a checkpoint from
[its repository](https://github.com/microsoft/mattersim/tree/main/pretrained_models):

```json
"mattersim": {
  "factory": "reactionflow.adapters.ase:create_adapter",
  "options": {
    "calculator_factory": "mattersim.forcefield:MatterSimCalculator",
    "calculator_kwargs": {"load_path": "/abs/path/mattersim-v1.0.0-1M.pth", "device": "cpu"},
    "model_files": ["/abs/path/mattersim-v1.0.0-1M.pth"],
    "torch_device": "cpu"
  }
}
```

NEP89 through Calorine, which runs on CPUs and does not use Torch, with the model file from
[the NEP89 release](https://github.com/brucefan1983/GPUMD/tree/v5.0/potentials/nep/nep89_20250409):

```json
"nep89": {
  "factory": "reactionflow.adapters.ase:create_adapter",
  "options": {
    "calculator_factory": "calorine.calculators:CPUNEP",
    "calculator_kwargs": {"model_filename": "/abs/path/nep89_20250409.txt"},
    "model_files": ["/abs/path/nep89_20250409.txt"]
  }
}
```

MatterSim, CHGNet, and NEP89 are materials models and expect a periodic cell.

ORB builds its calculator in two steps, so it needs a small factory in a module on `PYTHONPATH`,
here `orb_calculator.py`:

```python
import torch
from orb_models.forcefield import pretrained
from orb_models.forcefield.inference.calculator import ORBCalculator


class _Float32Graphs:
    def __init__(self, adapter):
        self.adapter = adapter

    def from_ase_atoms(self, *, atoms, **kwargs):
        return self.adapter.from_ase_atoms(
            atoms=atoms,
            **kwargs,
            output_dtype=torch.float32,
            graph_construction_dtype=torch.float32,
        )


def orb_v3(weights_path, device="cpu"):
    model, adapter = pretrained.orb_v3_conservative_inf_omat(
        weights_path=weights_path, device=device, precision="float32-highest", compile=False
    )
    return ORBCalculator(model, _Float32Graphs(adapter), device=device)
```

```json
"orb-v3": {
  "factory": "reactionflow.adapters.ase:create_adapter",
  "options": {
    "calculator_factory": "orb_calculator:orb_v3",
    "calculator_kwargs": {"weights_path": "/abs/path/orb-v3-conservative-inf-omat.ckpt", "device": "cpu"},
    "model_files": ["/abs/path/orb-v3-conservative-inf-omat.ckpt"],
    "packages": ["orb-models"],
    "torch_device": "cpu"
  }
}
```

The weights are listed in ORB's
[model table](https://github.com/orbital-materials/orb-models/blob/v0.7.0/MODELS.md).
`packages` records the ORB version, which the adapter cannot infer from a factory in your own
module. The first ORB evaluation in a process differs from later ones by about 10⁻¹² Å in the
positions after one step, so ORB restarts agree to that level rather than bit for bit. ORB 0.7.0
needs Python 3.12, because one of its dependencies does not build on 3.13.
