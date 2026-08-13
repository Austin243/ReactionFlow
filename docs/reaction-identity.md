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

This first identity definition does not include geometry, bond order, charge, spin,
stereochemistry, or periodic-image shifts. A match identifies the same geometric topology change;
it does not establish a mechanism, transition state, barrier, or kinetics.
