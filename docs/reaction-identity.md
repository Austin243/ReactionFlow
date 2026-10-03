# Reaction topology identity

`same_reaction()` compares two candidates by exact labeled graph isomorphism:

```python
from reactionflow import same_reaction

if same_reaction(first, second):
    print("same reaction topology")
```

Graph nodes are labeled by element. Edges are labeled `unchanged`, `formed`, or `broken` from the
candidate's reactant and product bonds. Atom IDs and ASE array order are graph keys rather than
identity labels, so renumbered candidates can match. Geometry, frame numbers, and resolved status
are also ignored.

Forward and reverse occurrences are equivalent: all `formed` and `broken` labels may swap as one
global direction reversal. Element labels, connectivity, and the change pattern must otherwise
match exactly.

## Local classes

By default the graph covers the candidate's whole bonded region, every atom connected to a changed
bond. In a polymer or network solid that region grows with each reaction, so two copies of the same
local step, such as one more molecule adding to chains of different length, do not match.
`radius` limits the graph to the atoms of the changed bonds and the atoms within that many bonds of
them:

```python
same_reaction(first, second, radius=2)
```

`radius=0` compares only the reacting atoms, `radius=1` adds their bonded neighbors, and so on. A
radius larger than the region compares the whole region. Candidates that match as whole regions
match at every radius, so a smaller radius only merges classes.

Each trajectory's [occurrence store](occurrence-store.md) classifies by the whole region.
`reactionflow status` merges those classes with `--radius 2` by default. For acetonitrile
polymerization, that keeps chain initiation, chain growth, and ring closure apart and counts every
growth step as one class. Each class label ends with the formula of the compared atoms, so classes
with the same bond changes can be told apart, and `--json` names one example occurrence per class.

This first identity definition does not include geometry, bond order, charge, spin,
stereochemistry, or periodic-image shifts. A match identifies the same geometric topology change;
it does not establish a mechanism, transition state, barrier, or kinetics.
