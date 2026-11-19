# Verification lanes: derivatives, intervals, synthetic MBAR

Small, fast experiments that cross-check engine mathematics with independent
numerical tools. This directory is additive: it does not modify
`FoundationCore`, its `Project.toml`, or any other experiment, and it writes
no files inside the repository.

| Lane | File | Tools | Claims checked |
| --- | --- | --- | --- |
| A — derivatives | `lane_derivatives.jl` | TaylorSeries, Enzyme | bonded forces equal `-dU/dq`; projected velocities satisfy the first-order constraint tangency |
| B — intervals | `lane_intervals.jl` | IntervalArithmetic | engine energies/forces lie in interval enclosures; rigid-water projection; barostat exponent symmetry |
| C — MBAR | `lane_mbar.jl`, `python/mbar_pymbar.py` | Julia + pymbar | synthetic 4-state MBAR reproduces `0.5 log k` and matches pymbar on the same data |

## Run

```sh
cd julia/experiments/verification
./setup_env.sh     # once: isolated Julia env + Python venv, all outside the repo
./run_all.sh       # lane report, assertions, pymbar cross-check
```

`run_all.sh` uses the existing Julia (`~/.cache/biology-mm/julia-1.13.3`)
and records nothing in the repository. See `NOTES.md` for exact versions,
installation commands, and the blockers hit (Enzyme runtime activity,
standalone barostat module, IntervalArithmetic `NG` flag, pymbar venv).

## Lane A — derivatives

Fixture: six-atom bonded chain with 5 bonds, 4 angles, 3 proper dihedrals,
evaluated through `Factors.total_energy_forces` (exact engine kernels).

1. `chain_energy` in `common.jl` is an independent generic implementation of
   the same three laws. `Float64` energy parity is asserted **first**
   (`energy_parity = 0`), which pins that implementation to the engine.
2. TaylorSeries expands `chain_energy` to first order in each coordinate; the
   coefficients are derivatives to machine precision (better than finite
   differences by construction).
3. Enzyme reverse-mode differentiates the *engine's own* code path. Runtime
   activity and relaxed aliasing are required because `Context` stores an
   `AbstractBox` and `unwrap_displacement` uses `Union` dispatch.
4. Central finite differences along one direction are a low-order sanity
   check only.
5. A rigid-water fixture is projected with `Constraints.project_positions!`
   and `project_velocities!`; TaylorSeries differentiates the O–H distance
   along the projected velocity, checking `r · (v_j - v_i) = 0` at first
   order (observed ~8e-17).

## Lane B — interval arithmetic

The hull of a `1e-6` ball around the chain configuration is propagated with
IntervalArithmetic. 200 engine evaluations sampled inside the ball must lie in
the computed energy enclosure; a `HarmonicBond` force must lie in its
componentwise force enclosure. The water fixture is projected after a 0.05
displacement, then `1e-13` position intervals must contain each constrained
distance and enclose zero for `|x_j - x_i|^2 - d^2`.

The barostat lane uses the exact `MonteCarloBarostat` module (loaded through a
host module because `FoundationCore` does not include it). It checks:

- interval enclosure of `log_acceptance` over a box of `(dU, V, V')`,
- exact antisymmetry `l(dU,V,V') = -l(-dU,V',V)` (observed 2e-16),
- for 300 `step!` calls, that the returned `old/new` energies, `old/new`
  volumes, `delta_log_volume`, box update, and `log_acceptance` are mutually
  consistent with the documented proposal (observed 7e-16).

Decorations must be `com` for every enclosure; the `NG` display flag is the
input-literal guarantee flag and is explained in `NOTES.md`.

## Lane C — synthetic MBAR

States `U_k(x) = k x^2 / 2`, `k = 1,2,4,8`, `beta = 1`, sampled exactly from
`N(0, 1/k)` (2000 samples each). The exact reduced free-energy difference is
`Delta f_{k,0} = 0.5 log k`.

`lane_mbar.jl` solves the MBAR fixed point with log-sum-exp (18 iterations,
tolerance 1e-12), reports the maximum deviation from the analytic value
(1.6e-2, i.e. ~1.1 bootstrap standard errors) and a 100-repetition
within-state bootstrap standard error (1.5e-2). It writes `u_kn.csv`,
`n_k.csv`, `f_julia.csv`, `k_values.csv` and `meta.txt` to
`VERIFICATION_ARTIFACTS` (default under `~/.cache/biology-mm/`).

`python/mbar_pymbar.py` reads that dataset, solves the identical equations with
pymbar 4.0.3, and asserts (a) agreement with the analytic value within four
combined standard errors and (b) agreement with the Julia estimate within
1e-3 kT (observed 4.9e-14).

## Limits

Numerical checks at `Float64`, not machine-checked proofs. Interval
containment is validated against sampled evaluations, not a verified floor of
the engine. MBAR is validated on exact-ensemble samples, not on a production
sampler. The barostat checks its documented configurational move and
bookkeeping, not physical NPT correctness.
