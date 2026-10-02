# Models and preparation

ReactionFlow includes optional MACE, UMA, and ANI-1xnr adapters. Other models can use the
[generic ASE calculator adapter or a custom adapter](campaigns.md#mlip-adapter). Model selection
uses the existing campaign format: one `adapter` in schema version 1, or named `adapter_profiles`
assigned to trajectories in schema version 2. No new campaign fields are required.

## Install and prepare

Activate an environment for the chosen backend and install ReactionFlow from this checkout:

```bash
python -m pip install .
reactionflow prepare campaign.json --install
```

The optional `--install` runs pip in the current Python environment for the referenced built-in
backends: `mace-torch==0.3.16` or `fairchem-core==2.23.0`. It then starts a fresh Python process to
prepare the model files, because pip may have changed already imported dependencies. Installation
or preparation failures return a failing exit status. It does not create an environment for you.

Alternatively, install the selected extra yourself with `python -m pip install '.[mace]'` or,
in a separate environment, `python -m pip install '.[uma]'`, then omit `--install`:

```bash
reactionflow prepare campaign.json
reactionflow prepare campaign.json --index 2
```

By default, preparation visits each distinct adapter configuration referenced by a trajectory once.
`--index` selects only that trajectory's configuration and can also be combined with `--install`.
Preparation resolves/downloads model files and reports their paths without constructing a
calculator, starting MD, or creating trajectory output. Unreferenced profiles are ignored.

Keep incompatible backends in separate environments and campaigns. The pinned MACE package needs
`e3nn==0.4.4`, whereas UMA needs `e3nn>=0.5`; they cannot share this dependency environment. UMA's
FAIR-Chem version also requires Torch 2.13, while the ANI-1xnr setup pins Torch 2.11. Installing UMA
into the ANI environment would change its dependencies and invalidate exact restarts. The bundled
`setup-perlmutter-ani1xnr.sh` and Perlmutter job scripts configure that ANI environment; they are
not setup scripts for MACE or UMA. Use your backend's environment in your own batch script.
`--install` rejects these known incompatible combinations before running pip; select one backend
with `--index` in a fresh environment.

Once prepared, normal MACE/UMA runs use only local files and do not download weights or install
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

This is a profile block to include in a schema version 2 campaign; the inner factory/options
object also works as the schema version 1 `adapter`:

```json
{
  "adapter_profiles": {
    "mace-materials": {
      "factory": "reactionflow.adapters.mace:create_adapter",
      "options": {
        "family": "mp",
        "model": "medium-mpa-0",
        "device": "cuda",
        "dtype": "float64"
      }
    }
  }
}
```

Set a trajectory's `adapter_profile` to `mace-materials`. `family` is `mp` (default) or `off`;
`model` must be an explicit name from that family's
[pinned MACE registry](https://github.com/ACEsuit/mace/blob/v0.3.16/mace/calculators/foundations_models.py).
For example, `family: "off"` with `model: "small"` selects MACE-OFF23 small. `device` is `cuda`
(default) or `cpu`, and `dtype` is `float64` (default) or `float32`. An optional `head` selects a
named head supported by that checkpoint; the selected head is recorded in the restart contract.
All MD and refinement stages use the same model, head, and precision.

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

## Local files and cache

Both adapters accept an absolute `checkpoint` path instead of `model`; supply exactly one.
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
resuming. MACE verifies cached file hashes; UMA uses the Hugging Face cache. Exact checkpoints
record hashes of every resolved weight/reference file for both adapters.

## Execution and restart scope

The built-in adapters use the shared Langevin BAOAB NVT/NPT runtime and the selected calculator for
every pathway stage. For CPU runs, set `device: "cpu"` and campaign `require_gpu: false`; a CUDA
worker must see exactly one usable GPU.

Exact resume binds model files, options, package versions, adapter/runtime source, and the recorded
Torch/platform/device environment. Changing those inputs requires a new trajectory output rather
than treating it as exact continuation. Preserve the environment and files of an active campaign.
Torch deterministic algorithms are enabled and TF32 is disabled; unsupported nondeterministic
operations fail. UMA uses fixed float32 inference without compilation or MoLE merging, so its
upstream accelerated presets do not describe this adapter's performance.

The adapter tests cover setup, configuration, and restart contracts with controlled calculators.
A CPU smoke check with MACE 0.3.16, Torch 2.14.1, and the official MACE-MP small checkpoint
downloaded the model and completed two NPT steps. With downloads blocked, a one-step run followed
by exact restoration and a second step matched the uninterrupted two-step atomic state, cell, and
dynamics arrays bit for bit. This short check does not establish GPU reproducibility, real UMA
checkpoint behavior, or scientific accuracy for a new system. Model integration alone is not
validation of a reaction barrier or stress model.
