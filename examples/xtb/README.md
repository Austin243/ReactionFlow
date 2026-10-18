# Proton transfer with GFN2-xTB on a CPU

This example runs ReactionFlow from start to finish on a laptop. A 15-atom cluster of
malonaldehyde and two water molecules runs 20 ps of MD at 500 K with GFN2-xTB. The enol proton
hops between the two oxygens several times. ReactionFlow detects each hop, refines its path with
NEB and CI-NEB, and checks the saddle.

GFN2-xTB comes from [tblite](https://github.com/tblite/tblite) and runs through the generic ASE
adapter, so the campaign names `tblite.ase:TBLite` as its calculator and needs no model download.

From the repository root:

```bash
python -m pip install . tblite==0.7.0
export OMP_NUM_THREADS=1
reactionflow run examples/xtb/campaign.json
reactionflow status examples/xtb/campaign.json
```

`OMP_NUM_THREADS=1` keeps tblite on one core, which is fastest for 15 atoms. The run takes about
2 minutes on an Apple M4 Pro and 5 minutes on one core of a cloud server, refinements included,
and writes to `outputs/xtb-demo/500K-seed1`. `status` then prints something like this.

```text
reaction                         model     P (GPa)  trajectories  events  converged  validated  barrier eV (min / median / max)
broken H-O; formed H-O [C3H3O2]  gfn2-xtb  NVT      1             9       9/9        9          0.122 / 0.156 / 0.164

Saddle check: 9 validated, 0 no saddle, 0 rejected, 0 unchecked. Barriers come only from validated saddles.
```

Each proton transfer is one occurrence of the same reaction class, the O-H bond that breaks and
the O-H bond that forms in malonaldehyde. Every saddle has one imaginary mode near 1150i cm⁻¹ and
connects the two enols. The number of events and their barriers change with the machine, because
floating-point differences grow along a chaotic trajectory. Eight 20 ps runs from 400 to 800 K
gave 55 transfers, all validated, at 0.04 to 0.19 eV.

Each refinement is in `pathways/<occurrence>/`: `images.traj` holds the band, and `result.json`
the barrier, the saddle check and the frames the endpoints came from.
