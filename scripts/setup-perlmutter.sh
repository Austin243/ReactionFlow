#!/bin/bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)

module load pytorch/2.11.0

export PYTHONUSERBASE=${REACTIONFLOW_PYTHONUSERBASE:-"$repo_root/.perlmutter-python"}
export PATH="$PYTHONUSERBASE/bin:$PATH"
unset PYTHONNOUSERSITE || true

python -m pip install --user --requirement "$repo_root/requirements/perlmutter-core.txt"
python -m pip install --user --force-reinstall --no-deps "$repo_root"

python - <<'PY'
import ase
import networkx
import numpy
import reactionflow

print(
    "ReactionFlow core setup passed: "
    f"reactionflow={reactionflow.__version__} ase={ase.__version__} "
    f"networkx={networkx.__version__} numpy={numpy.__version__}"
)
PY

printf '\nCore setup complete. Install your ASE calculator package into this environment with:\n'
printf '  module load pytorch/2.11.0\n'
printf '  export PYTHONUSERBASE=%q\n' "$PYTHONUSERBASE"
printf '  python -m pip install --user <your-mlip-package>\n'
