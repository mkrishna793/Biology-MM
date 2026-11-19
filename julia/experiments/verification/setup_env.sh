#!/usr/bin/env bash
# Create the isolated per-lane Julia environment and Python venv used by this
# harness. Nothing is written inside the repository: packages go to
# $JULIA_ROOT/verify-depot and the environment to $JULIA_ROOT/verification-env.
#
#   ./setup_env.sh
#
# Tested versions are recorded in NOTES.md; this script intentionally does not
# overwrite an existing environment unless --force is passed.
set -euo pipefail

JULIA_ROOT=${JULIA_ROOT:-$HOME/.cache/biology-mm}
JULIA_BIN=${JULIA_BIN:-$JULIA_ROOT/julia-1.13.3/bin/julia}
SHARED_DEPOT=${SHARED_DEPOT:-$JULIA_ROOT/julia-depot}
LANE_DEPOT=${LANE_DEPOT:-$JULIA_ROOT/verify-depot}
VERIFY_ENV=${VERIFY_ENV:-$JULIA_ROOT/verification-env}
PYENV=${PYENV:-$JULIA_ROOT/pyenv}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)

if [[ ! -x "$JULIA_BIN" ]]; then
    echo "SKIP no Julia at $JULIA_BIN; set JULIA_BIN and JULIA_ROOT" >&2
    exit 1
fi
if [[ -e "$VERIFY_ENV/Project.toml" && "${1:-}" != "--force" ]]; then
    echo "environment already exists at $VERIFY_ENV (use --force to reinstall)"
else
    mkdir -p "$LANE_DEPOT" "$VERIFY_ENV"
    export REPO_ROOT
    JULIA_DEPOT_PATH="$LANE_DEPOT:$SHARED_DEPOT" "$JULIA_BIN" --project="$VERIFY_ENV" -e '
        using Pkg
        Pkg.develop(path=joinpath(ENV["REPO_ROOT"], "foundation", "julia"))
        Pkg.add(["TaylorSeries", "IntervalArithmetic", "Enzyme",
                 "Test", "Random", "Printf", "Statistics", "LinearAlgebra"])
        Pkg.status()
    '
fi

if [[ -x "$PYENV/bin/python" ]]; then
    echo "python venv already exists at $PYENV"
else
    python3 -m venv "$PYENV"
    "$PYENV/bin/pip" install --quiet "numpy==1.26.4" "pymbar==4.0.3"
fi
"$PYENV/bin/python" -c 'import numpy, pymbar; print(f"numpy {numpy.__version__} pymbar {pymbar.__version__}")'
