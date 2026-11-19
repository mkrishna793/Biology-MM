# Setup notes and blockers

Environment created outside the repository on 2026-11-12 (darwin, arm64,
Julia 1.13.3). Nothing here installs into the global depot, modifies
`julia/Project.toml`, or writes inside the repository at runtime.

## Layout

| Path (under `~/.cache/biology-mm/`) | Contents |
| --- | --- |
| `julia-1.13.3/bin/julia` | existing isolated Julia used by the project |
| `julia-depot/` | existing shared read-only depot (FFTW, Molly, ...) |
| `verify-depot/` | **per-lane depot** — every new package installs here |
| `verification-env/` | isolated project (`Project.toml`/`Manifest.toml`) for this harness |
| `verification-artifacts/` | MBAR dataset, Julia estimate, pymbar JSON |
| `pyenv/` | isolated Python venv with pymbar |

`JULIA_DEPOT_PATH="…/verify-depot:…/julia-depot"` makes new installs land in the
lane depot while the existing depot stays the source for shared JLLs.

## Exact versions (from `Pkg.status()` / `python -c "import …"`)

```
Julia              1.13.3
FoundationCore     0.1.0   (dev path: <repo>/julia)
TaylorSeries       0.23.4
IntervalArithmetic 1.0.14
Enzyme             0.13.219
Python             3.9.6   (system python3)
numpy              1.26.4
pymbar             4.0.3
```

## Installation commands (already executed)

```sh
export JULIA_DEPOT_PATH="$HOME/.cache/biology-mm/verify-depot:$HOME/.cache/biology-mm/julia-depot"
export REPO_ROOT="/path/to/Biology MM"
JULIA="$HOME/.cache/biology-mm/julia-1.13.3/bin/julia"
ENV_DIR="$HOME/.cache/biology-mm/verification-env"

"$JULIA" --project="$ENV_DIR" -e 'using Pkg;
  Pkg.add(["TaylorSeries", "IntervalArithmetic"]);
  Pkg.add("Enzyme");
  Pkg.develop(path=joinpath(ENV["REPO_ROOT"], "foundation", "julia"));
  Pkg.add(["Test", "Random", "Printf", "Statistics", "LinearAlgebra"])'

python3 -m venv "$HOME/.cache/biology-mm/pyenv"
"$HOME/.cache/biology-mm/pyenv/bin/pip" install "numpy==1.26.4" "pymbar==4.0.3"
```

`./setup_env.sh` reproduces the same state (idempotent; `--force` reuses the
install path but re-adds packages).

## Blockers met and how they were handled

1. **`MonteCarloBarostat` is not part of `FoundationCore`.** Its own docstring
   says it is a standalone adapter; `FoundationCore.jl` does not `include` it.
   Loading it at `Main` scope fails because it declares `using ..Types`.
   `common.jl` defines a host module that exposes `Types` and then `include`s
   the engine file, so the lane exercises the exact engine implementation
   (no copy):
   ```julia
   module BarostatHost
   using FoundationCore
   using FoundationCore: Types
   include(joinpath(pkgdir(FoundationCore), "src", "MonteCarloBarostat.jl"))
   end
   ```

2. **Enzyme fails on the engine's `Context` without runtime activity.**
   Plain `Enzyme.gradient(Enzyme.Reverse, …)` raises
   `EnzymeRuntimeActivityError: Detected potential need for runtime activity…`
   and reports the `AbstractBox` field stored in `Context` as the erring value.
   With `set_runtime_activity(Reverse)` it then raises
   `IllegalTypeAnalysisException` from the `Union` dispatch in
   `Types.unwrap_displacement`. Working combination:
   ```julia
   Enzyme.API.strictAliasing!(false)
   Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), energy, q)
   ```
   Result: gradient agrees with the engine force to ~7e-16 relative.

3. **IntervalArithmetic shows `_com_NG`, not plain `_com`.** `NG` is the
   *guarantee* flag: IntervalArithmetic 1.0 will not certify that a plain
   `Float64` literal is the intended real number, so mixed scalar/interval
   operations are tagged NG (see `src/intervals/exact_literals.jl` in the
   package). The endpoints are still outward-rounded; the flag only says the
   *inputs* were not marked exact. The lanes therefore assert
   `decoration(I) == com` (the function is defined on the whole box) and
   containment of engine values, and do not require the NG flag to be clear.
   Wrapping literals in `exact(…)` clears it, but was not necessary and would
   have leaked IntervalArithmetic types into `common.jl`.

4. **pymbar needs its own venv.** A `--system-site-packages` venv picked up the
   user's `~/Library/Python/3.9` numpy alongside the venv numpy; pymbar 4.0.3
   then failed in `compute_free_energy_differences` with
   `LinAlgError: SVD did not converge in Linear Least Squares` (corrupted
   mixed install, not a pymbar bug — the same code ran once the venv was
   clean). Use a plain `python3 -m venv`.
   pymbar 4.0.3 also has **no `method=` keyword** (older docs show one), and
   prints its JAX/citation banner on import; the banner is harmless.

5. **Julia 1.13 is newer than most package compat bounds.** Enzyme 0.13.219
   resolves and runs on macOS arm64; TaylorSeries 0.23.4 and
   IntervalArithmetic 1.0.14 are pure Julia. No pinning was required, but the
   exact resolved versions are listed above.

6. **`ConstraintLib.project_velocities!` mass indexing.** Natural but wrong:
   `v = p ./ m` broadcasts the mass vector along dimension 1 (rows), whereas
   the engine divides each atom *column* by that atom's mass. The lane uses
   `p ./ reshape(m, 1, :)`, matching `Constraints.tangent_residuals`. With the
   wrong convention the AD tangent check reported ~9e-2 instead of ~8e-17.

## Commands and observed results

Run from anywhere:

```sh
cd julia/experiments/verification
./run_all.sh
```

Observed on 2026-11-12 (Julia 1.13.3, 1 thread):

```
lane_derivatives : energy_parity=0.000e+00 taylor_err=4.564e-16 enzyme_err=7.417e-16 fd_err=1.588e-09 tangent=7.974e-17
lane_intervals   : energy_width=1.194e-03 contained=200/200 bond_force=true constraint=true/true barostat=true decorations=true
lane_intervals   : barostat_symmetry=2.220e-16 barostat_step=7.002e-16 bond_force_width=1.285e-04
lane_mbar        : iterations=18 samples=8000 max_abs_err=1.615e-02 max_bootstrap_se=1.470e-02
lane_mbar        : Delta_f_est   = 0.00000 0.33791 0.67981 1.02357
lane_mbar        : Delta_f_exact = 0.00000 0.34657 0.69315 1.03972
Test Summary:      | Pass  Total   Time
verification lanes |   19     19  40.5s
pymbar 4.0.3 numpy 1.26.4
max |pymbar-analytic| = 0.016149852780049434  tolerance = 0.05635928098772002
max |pymbar-julia|    = 4.884981308350689e-14
PASS pymbar vs analytic within 4 combined standard errors
PASS pymbar vs Julia estimate within 1e-3 kT
```

Isolated reruns (useful while debugging):

```sh
export JULIA_DEPOT_PATH="$HOME/.cache/biology-mm/verify-depot:$HOME/.cache/biology-mm/julia-depot"
JULIA="$HOME/.cache/biology-mm/julia-1.13.3/bin/julia"
ENV_DIR="$HOME/.cache/biology-mm/verification-env"
"$JULIA" --project="$ENV_DIR" run_all.jl
"$JULIA" --project="$ENV_DIR" test/runtests.jl
```

## What is not established

- These are numerical checks at `Float64`, not machine-checked proofs. The
  Lean/Rocq development in `lean`, `rocq` is untouched.
- Interval containment is checked against sampled engine evaluations inside
  the box, not against a formally verified floor of the engine code.
- MBAR validates the estimator on data drawn from the exact ensemble; it does
  not validate any production sampler's ergodicity.
- The barostat lane checks the acceptance exponent and returned bookkeeping of
  `MonteCarloBarostat.step!`; it does not establish physical NPT correctness,
  which the adapter itself disclaims.
