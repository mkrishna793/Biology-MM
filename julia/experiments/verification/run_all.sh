#!/usr/bin/env bash
# Run the verification harness: human-readable lane report, assertion suite,
# then the optional pymbar cross-check of the shared MBAR dataset.
#
#   ./run_all.sh
#
# Environment overrides:
#   JULIA_BIN     Julia executable        (default ~/.cache/.../julia-1.13.3/bin/julia)
#   VERIFY_ENV    isolated project        (default ~/.cache/.../verification-env)
#   VERIFY_DEPOT  per-lane depot          (default ~/.cache/.../verify-depot)
#   PYTHON        python with pymbar      (default ~/.cache/.../pyenv/bin/python)
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JULIA_ROOT=${JULIA_ROOT:-$HOME/.cache/biology-mm}
JULIA_BIN=${JULIA_BIN:-$JULIA_ROOT/julia-1.13.3/bin/julia}
VERIFY_ENV=${VERIFY_ENV:-$JULIA_ROOT/verification-env}
LANE_DEPOT=${VERIFY_DEPOT:-$JULIA_ROOT/verify-depot}
SHARED_DEPOT=${SHARED_DEPOT:-$JULIA_ROOT/julia-depot}
export JULIA_DEPOT_PATH="$LANE_DEPOT:$SHARED_DEPOT"
export VERIFICATION_ARTIFACTS=${VERIFICATION_ARTIFACTS:-$JULIA_ROOT/verification-artifacts}

echo "== lane report =="
"$JULIA_BIN" --project="$VERIFY_ENV" "$HERE/run_all.jl"

echo "== assertions =="
"$JULIA_BIN" --project="$VERIFY_ENV" "$HERE/test/runtests.jl"

echo "== pymbar cross-check =="
PYTHON=${PYTHON:-$JULIA_ROOT/pyenv/bin/python}
if [[ -x "$PYTHON" ]]; then
    "$PYTHON" "$HERE/python/mbar_pymbar.py"
else
    echo "SKIP pymbar: no interpreter at $PYTHON (see setup_env.sh)"
fi
