#!/bin/bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)

module load pytorch/2.11.0

export PYTHONUSERBASE=${REACTIONFLOW_PYTHONUSERBASE:-"$repo_root/.perlmutter-python"}
export TORCHANI_DATA_DIR=${REACTIONFLOW_TORCHANI_DATA_DIR:-"$repo_root/.cache/torchani"}
export PATH="$PYTHONUSERBASE/bin:$PATH"
unset PYTHONNOUSERSITE || true

python - <<'PY'
import torch

if torch.__version__.split("+", 1)[0] != "2.11.0":
    raise SystemExit(f"expected NERSC torch 2.11.0, found {torch.__version__}")
print(f"Using torch {torch.__version__}")
PY

python -m pip install --user --requirement "$repo_root/requirements/perlmutter-ani1xnr.txt"
python -m pip install --user --force-reinstall --no-deps "$repo_root"
python "$repo_root/scripts/cache_ani1xnr.py" --data-dir "$TORCHANI_DATA_DIR"
python - <<'PY'
import os

import torch
import torchani
from torchani.models import ANI1xnr

model = ANI1xnr(model_index=0, device="cpu")
assert len(model.symbols) == 4
print(
    f"ReactionFlow ANI setup passed: torch={torch.__version__} "
    f"torchani={torchani.__version__} data={os.environ['TORCHANI_DATA_DIR']}"
)
PY

reactionflow validate \
    "$repo_root/examples/perlmutter/acn_20gpa_ani1xnr/campaign.json"

printf '\nSetup complete. Submit the example from this checkout with:\n'
printf '  sbatch -A <GPU_PROJECT> -q regular %s\n' \
    "$repo_root/examples/perlmutter/acn_20gpa_ani1xnr/submit.sbatch"
