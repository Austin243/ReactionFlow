# Command-line reference

`reactionflow` has nine commands. This page lists every argument and option of each one and links
to the page that explains it. The [README](../README.md#run-your-own-system) shows the commands in
the order a campaign uses them.

| Command | What it does |
| --- | --- |
| [`init`](#init) | Write a campaign file by answering questions |
| [`models`](#models) | List the built-in models, heads, and tasks |
| [`validate`](#validate) | Check a campaign without importing its model |
| [`run`](#run) | Run or resume one trajectory, or an EON search |
| [`prepare`](#prepare) | Download model files, and install backend packages with `--install` |
| [`check`](#check) | Test whether a band's highest image is a saddle between its two ends |
| [`scan`](#scan) | Replay saved frames through bond detection without loading the model |
| [`export-electrace`](#export-electrace) | Lay out a refined band as VASP frame folders for ElecTrace |
| [`status`](#status) | Summarize trajectories and reactions without changing any run |

Every command takes `-h` or `--help`, which prints its arguments. `campaign` is the path of a
campaign file. Trajectories are numbered from 0 in the order the campaign lists them, and `--index`
takes that number.

A campaign file with `"mode": "eon"` is an EON campaign, and one without `mode` runs MD. Where a
command treats an EON campaign differently, its section says how.

## init

```text
reactionflow init [campaign]
```

`init` asks whether to write an MD or an EON campaign, asks for its settings, and writes the file
([write a campaign by answering questions](campaigns.md#write-a-campaign-by-answering-questions)).

| Argument | Meaning |
| --- | --- |
| `campaign` | The file to write (default `campaign.json`) |

It asks before it replaces an existing file, and exits with status 1 when it writes nothing.

## models

```text
reactionflow models [--backend BACKEND]
```

`models` prints the built-in models with their heads and tasks as JSON
([models and preparation](models.md)).

| Option | Meaning |
| --- | --- |
| `--backend BACKEND` | Show only one backend: `mace`, `uma`, `aimnet2`, or `ani1xnr` |

## validate

```text
reactionflow validate campaign
```

`validate` checks a campaign and reads its structures without importing the model, then prints a
one-line JSON summary. For an MD campaign the summary has the number of trajectories, the atoms in
each structure (`structure_atoms`), the trajectories per model profile (`adapter_profile_counts`),
and any `warnings` about settings that quietly shape the result
([campaign guide](campaigns.md#when-barriers-are-refined)).

For an EON campaign the summary has the number of atoms, the kinetic Monte Carlo steps, the
temperature, the adapter factory, and `rigid_rotation`, with a warning when fixed atoms leave a
rotation free ([symmetry](eon-search.md#symmetry)).

## run

```text
reactionflow run [--index INDEX] [--download] campaign
```

`run` runs one trajectory of the campaign in this process. Rerun the same command to resume it from
its last checkpoint
([many trajectories on a cluster](how-it-works.md#many-trajectories-on-a-cluster)).

| Option | Meaning |
| --- | --- |
| `--index INDEX` | The trajectory to run |
| `--download` | Prepare the selected built-in model before running ([install and prepare](models.md#install-and-prepare)) |

Without `--index`, a campaign with one trajectory runs it, and under Slurm each task runs the
trajectory numbered by its `SLURM_PROCID`. A campaign with several trajectories needs `--index`
outside Slurm. Under Slurm the number of tasks must equal the number of trajectories, and an
`--index` that differs from `SLURM_PROCID` is refused.

`run` prints one JSON line when it stops, with the trajectory's phase and counts or with the
error, which it also writes to `last-error.json`
([campaign guide](campaigns.md#perlmutter-mapping)).

For an EON campaign, `run` runs or resumes the search
([resume and records](eon-search.md#resume-and-records)). The campaign is one task, so `--index`
can only be 0 and a Slurm job must have one task. `--download` prepares the model before
searching. When the search stops, `run` prints the report of `status --json` on one line. After an
error it prints no report and writes the error to `last-error.json`.

## prepare

```text
reactionflow prepare [--index INDEX] [--install] campaign
```

`prepare` downloads the files of the built-in models a campaign uses and prints their paths as
JSON, without creating a trajectory ([install and prepare](models.md#install-and-prepare)).

| Option | Meaning |
| --- | --- |
| `--index INDEX` | Prepare only this trajectory's model |
| `--install` | Install the backend packages of the selected models with pip, then prepare |

For an EON campaign, `--install` also installs `pyeonclient` ([EON models](eon-search.md#models)).
The campaign has one model, so `--index` can only be 0.

## check

```text
reactionflow check [--index INDEX] [--pressure PRESSURE] campaign band
```

`check` tests whether the highest image of a band is a saddle between its two ends, and writes the
verdict beside the band ([checking a band](pathway-refinement.md#checking-a-band)).

| Argument | Meaning |
| --- | --- |
| `campaign` | An MD campaign, which supplies the model, the bond thresholds, and the pathway settings |
| `band` | A band file ASE can read, both ends included |
| `--index INDEX` | The trajectory whose model and pressure to use; a campaign with several needs it |
| `--pressure PRESSURE` | A pressure in GPa, in place of the trajectory's |

`check` refuses an EON campaign.

## scan

```text
reactionflow scan [--index INDEX] [--radius RADIUS] [--persistence PERSISTENCE]
                  [--stability STABILITY] [--pair A-B FORM BREAK] [--every EVERY]
                  [--frame-ps FRAME_PS] [--out OUT] source
```

`scan` replays saved frames through bond detection and candidate tracking without loading the
model, and writes its tables to a directory ([offline scan](offline-scan.md)).

| Argument | Meaning |
| --- | --- |
| `source` | An MD campaign `.json` file, or any trajectory file ASE can read |
| `--index INDEX` | Scan only this campaign trajectory |
| `--radius RADIUS` | Region radius in bonds, or `none` for whole components |
| `--persistence PERSISTENCE` | Observations a bond change must last |
| `--stability STABILITY` | Observations a reaction region must hold |
| `--pair A-B FORM BREAK` | Formation and break distances in Å for one element pair; can be repeated |
| `--every EVERY` | Frames between species counts (default 10) |
| `--frame-ps FRAME_PS` | Picoseconds between the frames of a trajectory file |
| `--out OUT` | Output directory |

The options are explained under [campaign runs](offline-scan.md#campaign-runs) and
[trajectory files from other codes](offline-scan.md#trajectory-files-from-other-codes). `--index`
is refused with a trajectory file, and `--frame-ps` with a campaign. `scan` refuses an EON
campaign, which saves no frames.

## export-electrace

```text
reactionflow export-electrace --out OUT pathway
```

`export-electrace` lays out one refined band as numbered VASP frame folders with an ElecTrace
configuration, and prints a one-line JSON summary
([electronic analysis with ElecTrace](pathway-refinement.md#electronic-analysis-with-electrace)).

| Argument | Meaning |
| --- | --- |
| `pathway` | A band's folder, `<trajectory>/pathways/<occurrence>`, or one of its `step-N` folders |
| `--out OUT` | A new or empty folder to write to; required |

## status

```text
reactionflow status [--json] [--radius RADIUS] campaign
```

`status` summarizes a campaign's trajectories and reactions without changing any run
([campaign status](campaigns.md#campaign-status)).

| Option | Meaning |
| --- | --- |
| `--json` | Print the report as JSON |
| `--radius RADIUS` | Bonds around a bond change that define its reaction class (default 2); see [local classes](reaction-identity.md#local-classes) |

For an EON campaign, `status` reports the search instead ([reports](eon-search.md#reports)).
`--json` prints that report as JSON, and `--radius` has no effect.
