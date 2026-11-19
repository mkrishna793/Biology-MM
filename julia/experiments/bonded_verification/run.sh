#!/usr/bin/env bash
# Lane 4: bonded-term convention check against an external engine.
#
#   ./run.sh            use the frozen reference tables
#   ./run.sh --refresh  re-extract parameters (InterMol/ParmEd env) and
#                       recompute the OpenMM per-term reference
#
# Stage 1 (InterMol env): GROMACS .top -> per-term parameters, cross-checked by
#                          two parsers.
# Stage 2 (OpenMM env):    those ParmEd-derived parameters -> real OpenMM
#                          per-term energies and forces at the frozen geometry.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JULIA_ROOT=${JULIA_ROOT:-$HOME/.cache/biology-mm}
JULIA_BIN=${JULIA_BIN:-$JULIA_ROOT/julia-1.13.3/bin/julia}
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-$JULIA_ROOT/julia-depot}

if [[ "${1:-}" == "--refresh" ]]; then
    INTERMOL_PYTHON=${INTERMOL_PYTHON:-$JULIA_ROOT/intermol-verify/bin/python}
    OPENMM_PYTHON=${OPENMM_PYTHON:-$JULIA_ROOT/.venv/bin/python}
    [[ -x "$INTERMOL_PYTHON" ]] || { echo "no InterMol python at $INTERMOL_PYTHON" >&2; exit 1; }
    [[ -x "$OPENMM_PYTHON" ]] || { echo "no OpenMM python at $OPENMM_PYTHON" >&2; exit 1; }
    "$INTERMOL_PYTHON" "$HERE/reference/extract_reference.py" > "$HERE/reference/terms.tsv"
    "$OPENMM_PYTHON" "$HERE/reference/openmm_terms.py" "$HERE/reference/terms.tsv" \
        > "$HERE/reference/openmm_terms.tsv"
    echo "refreshed reference/terms.tsv and reference/openmm_terms.tsv"
fi

"$JULIA_BIN" --project="$HERE/../.." "$HERE/check.jl"
