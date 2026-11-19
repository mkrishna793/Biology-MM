#!/bin/sh
# Reproduces every proof, every test suite, and the reference-data validation.
#
# Everything here is either implemented and checked, or reported as a concrete
# blocker. Nothing is skipped silently: a missing optional dependency produces a
# visible BLOCKED line; a missing optional dependency does NOT fail the run, but
# the core suites failing DOES.
#
#   sh verify.sh            # full run
#   sh verify.sh --core     # proofs + core suites only (no optional deps)
#
# Exit code 0 means: all proofs check, all core tests pass, and the reference
# data validation passes. Optional external cross-checks report their own status.
set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LEAN_DIR="$REPO_ROOT/lean"
ROCQ_DIR="$REPO_ROOT/rocq"
EXP_DIR="$REPO_ROOT/experiment"
JULIA_DIR="$REPO_ROOT/julia"
NATIVE_DIR="$REPO_ROOT/native"
CORE_ONLY=0
[ "${1:-}" = "--core" ] && CORE_ONLY=1

WORK_TMP="$(mktemp -d "${TMPDIR:-/tmp}/biology-mm-verify.XXXXXX")"
cleanup() {
  rm -rf "$WORK_TMP"
}
trap cleanup EXIT INT TERM

# External toolchains and all temporary audit/build files live outside the tree.
DEFAULT_TOOLS="${BIOLOGY_MM_TOOLS_DIR:-$HOME/.cache/biology-mm}"
if [ -z "${JULIA_BIN:-}" ]; then
  if [ -n "${BIOLOGY_MM_JULIA:-}" ] && [ -x "${BIOLOGY_MM_JULIA}" ]; then
    JULIA_BIN="$BIOLOGY_MM_JULIA"
  elif command -v julia >/dev/null 2>&1; then
    JULIA_BIN="$(command -v julia)"
  elif [ -x "$DEFAULT_TOOLS/julia-1.13.3/bin/julia" ]; then
    JULIA_BIN="$DEFAULT_TOOLS/julia-1.13.3/bin/julia"
  else
    JULIA_BIN=""
  fi
fi
if [ -n "${JULIA_DEPOT_PATH:-}" ]; then
  export JULIA_DEPOT_PATH
elif [ -d "$DEFAULT_TOOLS/julia-depot" ]; then
  JULIA_DEPOT_PATH="$DEFAULT_TOOLS/julia-depot"
  export JULIA_DEPOT_PATH
fi
if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "$DEFAULT_TOOLS/.venv/bin/python" ]; then
    PYTHON_BIN="$DEFAULT_TOOLS/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    PYTHON_BIN=""
  fi
fi
NIST_DIR="${NIST_DIR:-$DEFAULT_TOOLS/nist/data}"

echo "=============================================================="
echo " 0. Native C++/GPU bridges (CMake)"
echo "=============================================================="
if ! ls "$JULIA_DIR/lib/"*trium_mm_bridge.* >/dev/null 2>&1; then
  if command -v cmake >/dev/null 2>&1 && [ -f "$NATIVE_DIR/CMakeLists.txt" ]; then
    cmake -S "$NATIVE_DIR" -B "$NATIVE_DIR/build"
    cmake --build "$NATIVE_DIR/build" -j
    echo "Native build: OK"
  else
    echo "Native build: skipped (cmake not found; pure-Julia/Python fallback)"
  fi
else
  echo "Native libraries: present in $JULIA_DIR/lib"
fi

echo "=============================================================="
echo " 1. Lean proofs"
echo "=============================================================="
[ -d "$HOME/.elan/bin" ] && export PATH="$HOME/.elan/bin:$PATH"
if ! command -v lake >/dev/null 2>&1; then
  echo "Lean: BLOCKED (lake not found in PATH or $HOME/.elan/bin)."
  exit 1
fi
cd "$LEAN_DIR"
lake build
echo "Lean: build OK"

LEAN_AUDIT="$WORK_TMP/lean-audit.lean"
cat > "$LEAN_AUDIT" <<'EOF'
import Foundation
open Foundation
#print axioms Foundation.pairSum_decomp
#print axioms Foundation.forceOnA_eq
#print axioms Foundation.forceOnA_injective
#print axioms Foundation.total_force_zero
#print axioms Foundation.harmEnergy_pert
#print axioms Foundation.pairSum_decomp_general
#print axioms Foundation.shadow_invariant
#print axioms Foundation.verlet_energy_defect
#print axioms Foundation.energy_is_shadow_plus_defect
#print axioms Foundation.shadow_coeff_pos
#print axioms Foundation.shadow_energy_band
#print axioms Foundation.energy_band_abs
#print axioms Foundation.energy_after_step_eq
EOF
if ! lake env lean "$LEAN_AUDIT" >"$WORK_TMP/lean-audit.txt" 2>&1; then
  cat "$WORK_TMP/lean-audit.txt"
  echo "Lean audit: FAIL (Lean command returned nonzero)"
  exit 1
fi
cat "$WORK_TMP/lean-audit.txt"
if grep -q "sorryAx" "$WORK_TMP/lean-audit.txt"; then
  echo "Lean audit: FAIL (a proof depends on sorryAx)"
  exit 1
fi
# every theorem must depend only on the three standard axioms
N_STD=$(grep -c "propext, Classical.choice, Quot.sound" "$WORK_TMP/lean-audit.txt" || true)
if [ "$N_STD" -ne 13 ]; then
  echo "Lean audit: FAIL (expected 13 audited theorems, found $N_STD standard-axiom lines)"
  exit 1
fi
echo "Lean audit: OK ($N_STD theorems, standard axioms only, no sorryAx)"

echo ""
echo "=============================================================="
echo " 1b. Lean ternary symplectic & weighing/Farkas audit"
echo "=============================================================="
TERNARY_AUDIT="$WORK_TMP/lean-ternary-audit.lean"
cat > "$TERNARY_AUDIT" <<'EOF'
import Foundation
open Foundation
#print axioms Foundation.trit_neg_involutive
#print axioms Foundation.dual_rail_neg_involutive
#print axioms Foundation.btha_conservation
#print axioms Foundation.btfa_conservation
#print axioms Foundation.carry_free_step_conservation
#print axioms Foundation.verletKDK_exact_reversible
#print axioms Foundation.verletKDK_n_steps_reversible
#print axioms Foundation.hamiltonian_energy_conservation
#print axioms Foundation.cw4_valid
#print axioms Foundation.cw7_valid
#print axioms Foundation.cw13_valid
#print axioms Foundation.cw4_orthogonal
#print axioms Foundation.cw7_orthogonal
#print axioms Foundation.cw13_orthogonal
#print axioms Foundation.weighing_left_inverse
#print axioms Foundation.weighing_parseval_energy
#print axioms Foundation.radix3_compression_sum_eq
#print axioms Foundation.farkas_refutation_sound
#print axioms Foundation.amo_steric_exclusion_sound
EOF
if ! lake env lean "$TERNARY_AUDIT" >"$WORK_TMP/lean-ternary-audit.txt" 2>&1; then
  cat "$WORK_TMP/lean-ternary-audit.txt"
  echo "Lean ternary/weighing audit: FAIL (Lean command returned nonzero)"
  exit 1
fi
cat "$WORK_TMP/lean-ternary-audit.txt"
if grep -qE "sorryAx|Lean\.ofReduceBool" "$WORK_TMP/lean-ternary-audit.txt"; then
  echo "Lean ternary/weighing audit: FAIL (depends on sorryAx or native_decide)"
  exit 1
fi
N_TERNARY=$(grep -cE "depends on axioms:|does not depend on any axioms" "$WORK_TMP/lean-ternary-audit.txt" || true)
if [ "$N_TERNARY" -ne 19 ]; then
  echo "Lean ternary/weighing audit: FAIL (expected 19 audited theorems, found $N_TERNARY)"
  exit 1
fi
echo "Lean ternary/weighing audit: OK ($N_TERNARY theorems, zero sorryAx, zero native_decide)"

echo ""
echo "=============================================================="
echo " 2. Rocq proofs"
echo "=============================================================="
COQC_BIN="${COQC_BIN:-}"
if [ -z "$COQC_BIN" ]; then
  if [ -x "$HOME/.opam/rocq/bin/coqc" ]; then
    COQC_BIN="$HOME/.opam/rocq/bin/coqc"
  elif command -v coqc >/dev/null 2>&1; then
    COQC_BIN="$(command -v coqc)"
  fi
fi
if [ -z "$COQC_BIN" ] || [ ! -x "$COQC_BIN" ]; then
  echo "Rocq: BLOCKED (coqc not found in $HOME/.opam/rocq/bin or PATH)."
  exit 1
fi
ROCQ_WORK=$(mktemp -d "$WORK_TMP/rocq-work.XXXXXX")
cp "$ROCQ_DIR/Core.v" "$ROCQ_WORK/Core.v"
cd "$ROCQ_WORK"
COQCORE="$HOME/.opam/rocq/lib/rocq-core/rocq.d"
COQSTDLIB="$HOME/.opam/rocq/lib/rocq-stdlib/rocq.d"
if [ -d "$COQCORE" ] && [ -d "$COQSTDLIB" ]; then
  export CAML_LD_LIBRARY_PATH="$HOME/.opam/rocq/lib/stublibs:$HOME/.opam/rocq/lib/ocaml/stublibs:$HOME/.opam/rocq/lib/ocaml"
  "$COQC_BIN" -Q "$COQCORE" Corelib -Q "$COQSTDLIB" Stdlib \
    -o "$ROCQ_WORK/Core.vo" Core.v
else
  "$COQC_BIN" -o "$ROCQ_WORK/Core.vo" Core.v
fi
echo "Rocq: OK"

echo ""
echo "=============================================================="
echo " 3. Original numerical experiment (algebraic demonstrator)"
echo "=============================================================="
if [ -z "$PYTHON_BIN" ] || [ ! -x "$PYTHON_BIN" ]; then
  echo "Experiment: BLOCKED (no python interpreter found)."
  exit 1
fi
cd "$EXP_DIR"
"$PYTHON_BIN" test_group_harmonic.py
echo "Experiment: OK"

echo ""
echo "=============================================================="
echo " 4. Julia core test suites"
echo "=============================================================="
if [ -z "$JULIA_BIN" ] || [ ! -x "$JULIA_BIN" ]; then
  echo "Julia: BLOCKED (no julia binary found)."
  echo "Install Julia and set JULIA_BIN, or see INSTALL.md."
  exit 1
fi
export CHEM_PYTHON="${CHEM_PYTHON:-$PYTHON_BIN}"
export FOUNDATION_OPENMM_PYTHON="${FOUNDATION_OPENMM_PYTHON:-$PYTHON_BIN}"
export BIOLOGY_MM_JULIA="${BIOLOGY_MM_JULIA:-$JULIA_BIN}"
"$JULIA_BIN" --version
cd "$JULIA_DIR"
"$JULIA_BIN" --project=. test/run_all.jl
"$JULIA_BIN" -t 4 --project=. test/test_threading.jl
echo "Julia core suites: OK"

echo ""
echo "=============================================================="
echo " 4b. Phase-2 chemistry Python tests (offline)"
echo "=============================================================="
if [ -n "$PYTHON_BIN" ] && [ -x "$PYTHON_BIN" ]; then
  (
    cd "$REPO_ROOT"
    set -- "$REPO_ROOT/chemistry/tools/test_extractors.py" \
           "$REPO_ROOT/chemistry/tools/test_reference_protocol.py"
    [ -f "$REPO_ROOT/chemistry/tools/test_verify_protein_ligand.py" ] && \
      set -- "$@" "$REPO_ROOT/chemistry/tools/test_verify_protein_ligand.py"
    [ -f "$REPO_ROOT/chemistry/tools/test_register_fixture.py" ] && \
      set -- "$@" "$REPO_ROOT/chemistry/tools/test_register_fixture.py"
    [ -f "$REPO_ROOT/chemistry/tools/test_term_audit.py" ] && \
      set -- "$@" "$REPO_ROOT/chemistry/tools/test_term_audit.py"
    "$PYTHON_BIN" -m pytest -q "$@"
  )
  echo "Phase-2 chemistry Python tests: OK"
else
  echo "Phase-2 chemistry Python tests: BLOCKED (no python interpreter found)"
  exit 1
fi

echo ""
echo "=============================================================="
echo " 4c. Python SDK (biology_mm) tests"
echo "=============================================================="
if [ -n "$PYTHON_BIN" ] && [ -x "$PYTHON_BIN" ]; then
  (
    cd "$REPO_ROOT"
    PYTHONPATH="$REPO_ROOT/python" "$PYTHON_BIN" -m pytest -q "$REPO_ROOT/python/tests/test_biology_mm_sdk.py"
  )
  echo "Python SDK (biology_mm) tests: OK"
else
  echo "Python SDK (biology_mm) tests: BLOCKED (no python interpreter found)"
  exit 1
fi

if [ "$CORE_ONLY" = "1" ]; then
  echo "NIST and optional external checks: AVAILABLE-NOT-RUN (--core requested)"
  echo ""
  echo "=============================================================="
  echo " CORE VERIFICATION PASSED (--core: proofs + experiment + Julia suites + chemistry & SDK Python tests)"
  echo "=============================================================="
  exit 0
fi

echo ""
echo "=============================================================="
echo " 5. NIST Lennard-Jones reference validation"
echo "=============================================================="
if [ -d "$NIST_DIR" ]; then
  "$JULIA_BIN" --project=. -e "
    include(\"bench/nist_validate.jl\")
    using .NISTValidate
    r = NISTValidate.run_nist_validation(\"$NIST_DIR\")
    exit(r.allpass ? 0 : 1)
  "
  echo "NIST validation: OK"
else
  echo "NIST validation: BLOCKED"
  echo "  Reference data not present at $NIST_DIR"
  echo "  Retrieve with:"
  echo "    curl -sSL -A 'Mozilla/5.0' -o lj.tar.gz \\"
  echo "      https://www.nist.gov/document/ljsampleconfigurations-targz"
  echo "    mkdir -p $NIST_DIR && tar xzf lj.tar.gz -C $NIST_DIR"
  exit 1
fi

echo ""
echo "=============================================================="
echo " 6. Optional third-party cross-checks"
echo "=============================================================="
if [ -n "$PYTHON_BIN" ] && [ -x "$PYTHON_BIN" ]; then
  echo "--- dependency probe ---"
  "$PYTHON_BIN" "$REPO_ROOT/validation/adapters/probe_environment.py"
echo "--- OpenMM polarization adapter ---"
  if "$PYTHON_BIN" "$REPO_ROOT/validation/adapters/openmm_polarization_check.py"; then
    echo "OpenMM polarization adapter: PASS"
  else
    echo "OpenMM polarization adapter: FAIL (optional check; core checks remain valid)"
  fi
else
  echo "Python env: BLOCKED (no python interpreter found)"
fi

echo "--- FoundationCore / OpenMM matched LJ and electrostatics ---"
if [ -n "$PYTHON_BIN" ] && [ -x "$PYTHON_BIN" ] && [ -n "$JULIA_BIN" ] && [ -x "$JULIA_BIN" ]; then
  if "$PYTHON_BIN" -c 'import openmm' >/dev/null 2>&1; then
    if [ "${RUN_MATCHED_OPENMM:-0}" = "1" ]; then
      if "$PYTHON_BIN" "$REPO_ROOT/validation/adapters/openmm_matched_models.py"; then
        echo "Matched FoundationCore/OpenMM models: PASS"
      else
        rc=$?
        if [ "$rc" -eq 2 ]; then
          echo "Matched FoundationCore/OpenMM models: BLOCKED"
        else
          echo "Matched FoundationCore/OpenMM models: FAIL"
          exit "$rc"
        fi
      fi
    else
      echo "Matched FoundationCore/OpenMM models: AVAILABLE-NOT-RUN (set RUN_MATCHED_OPENMM=1)"
    fi
  else
    echo "Matched FoundationCore/OpenMM models: BLOCKED (OpenMM unavailable in $PYTHON_BIN)"
  fi
elif [ -z "$PYTHON_BIN" ] || [ ! -x "$PYTHON_BIN" ]; then
  echo "Matched FoundationCore/OpenMM models: BLOCKED (Python unavailable)"
else
  echo "Matched FoundationCore/OpenMM models: BLOCKED (Julia unavailable)"
fi

echo "--- Explicit-parameter TIP3P fixture ---"
if [ -n "$PYTHON_BIN" ] && [ -x "$PYTHON_BIN" ] && [ -n "$JULIA_BIN" ] && [ -x "$JULIA_BIN" ]; then
  if "$PYTHON_BIN" -c 'import openmm' >/dev/null 2>&1; then
    if [ "${RUN_TIP3P_OPENMM:-0}" = "1" ]; then
      if "$PYTHON_BIN" "$REPO_ROOT/validation/adapters/tip3p_matched.py"; then
        echo "TIP3P matched fixture: PASS"
      else
        rc=$?
        if [ "$rc" -eq 2 ]; then
          echo "TIP3P matched fixture: BLOCKED"
        else
          echo "TIP3P matched fixture: FAIL"
          exit "$rc"
        fi
      fi
    else
      echo "TIP3P matched fixture: AVAILABLE-NOT-RUN (set RUN_TIP3P_OPENMM=1)"
    fi
  else
    echo "TIP3P matched fixture: BLOCKED (OpenMM unavailable in $PYTHON_BIN)"
  fi
else
  echo "TIP3P matched fixture: BLOCKED (configured Python or Julia unavailable)"
fi

echo "--- MDAnalysis structure cross-check on real BPTI ---"
BPTI="$REPO_ROOT/wrappers/python/tests/systems/bpti.pdb"
if [ -n "$PYTHON_BIN" ] && [ -x "$PYTHON_BIN" ] && [ -f "$BPTI" ]; then
  if "$PYTHON_BIN" "$REPO_ROOT/validation/adapters/structure_check.py" "$BPTI"; then
    echo "MDAnalysis structure cross-check: PASS"
  else
    echo "MDAnalysis structure cross-check: FAIL (optional check; core checks remain valid)"
  fi
else
  echo "structure cross-check: BLOCKED"
fi

echo "--- Molly.jl independent LJ cross-check ---"
if [ -d "$JULIA_DIR/validation-env" ]; then
  if NIST_DIR="$NIST_DIR" "$JULIA_BIN" --project="$JULIA_DIR/validation-env" \
    "$REPO_ROOT/validation/adapters/molly_crosscheck.jl"; then
    echo "Molly LJ cross-check: PASS"
  else
    echo "Molly LJ cross-check: FAIL (optional check; core checks remain valid)"
  fi
else
  echo "Molly cross-check: BLOCKED (validation-env not set up)"
fi

echo "--- Synthetic MSD analysis ---"
if [ -n "$PYTHON_BIN" ] && [ -x "$PYTHON_BIN" ] && "$PYTHON_BIN" -c 'import numpy' >/dev/null 2>&1; then
  "$PYTHON_BIN" -m unittest discover -s "$JULIA_DIR/experiments/biomolecular_validation" -p 'test*.py'
  echo "Synthetic MSD analysis: PASS"
else
  echo "Synthetic MSD analysis: BLOCKED (configured Python/NumPy unavailable)"
fi

echo "--- Bonded convention fixture ---"
if [ -f "$JULIA_DIR/experiments/bonded_verification/reference/terms.tsv" ]; then
  bash "$JULIA_DIR/experiments/bonded_verification/run.sh"
  echo "Bonded frozen reference fixture: PASS"
else
  echo "Bonded frozen reference fixture: BLOCKED (reference table missing)"
fi
echo "TIP3P bulk trajectory and long reference diffusion: AVAILABLE-NOT-RUN (offline experiments; see validation/STATUS.md)"
echo "PME scaling sweep: AVAILABLE-NOT-RUN (stored measurements; see julia/experiments/pme_scaling/results)"

echo ""
echo "=============================================================="
echo " ALL CORE VERIFICATION PASSED"
echo " Optional cross-checks reported above; see validation/STATUS.md"
echo "=============================================================="
