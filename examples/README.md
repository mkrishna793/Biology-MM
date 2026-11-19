# Runnable examples for the Julia foundation core

Five self-contained scripts that drive `FoundationCore` (`../julia/src/`) end to end, print
what they measure, and gate on explicit numeric thresholds. Each exits `0` only if every gate
passes; a failing gate prints its threshold and the measured value.

They are demonstrations of the pipeline and its error accounting, **not** studies of matter.
Every quantity in the output that comes from a finite run is an estimate, and the scripts label
it as one in the printed text (sample counts, tolerance arguments, single-chain diagnostics).
Nothing here is, or claims to be, production-ready simulation.

## Running

From `julia`, so that `--project=.` resolves the package:

```sh
cd julia
julia --project=. ../examples/01_energy_forces.jl; echo "exit $?"
julia --project=. ../examples/02_dynamics_error_control.jl; echo "exit $?"
julia --project=. ../examples/03_sampling_double_well.jl; echo "exit $?"
julia --project=. ../examples/04_polarization.jl; echo "exit $?"
julia --project=. ../examples/equilibrium_sensitivity.jl; echo "exit $?"
```

The same five work from the repository root:

```sh
julia --project=julia examples/01_energy_forces.jl
julia --project=julia examples/equilibrium_sensitivity.jl
```

Exit status: `0` all gates passed, `1` a gate failed (`01`/`02` stop at the first failed check;
`03`/`04` and `equilibrium_sensitivity` run every gate and print a failure count), `2` `FoundationCore` could not be loaded.
The last case is only possible for `01`/`02`, which use `using FoundationCore` and therefore need
`--project`; `03`/`04` and `equilibrium_sensitivity` `include` the source file directly and run even with no project at all
(verified: they exit `0`, `01` exits `2` and prints the command to use).

These are examples, not the test suite: the package's own tests live in `../julia/test/`.

No network access and no registered packages: `Project.toml` depends only on stdlib
(`LinearAlgebra`, `Random`, `Printf`, `Statistics`, `Test`). Tested with Julia 1.13.3. A full
pass of the original four costs about 20 s of wall time on one machine; the equilibrium example
adds its own HMC runs.

## The five examples

| File | What it demonstrates | What you should see (measured) | Wall time |
| --- | --- | --- | --- |
| `01_energy_forces.jl` | One 50-atom Lennard-Jones fixture through the whole slice: typed state -> cell-list pair list -> pair factors -> `(U, F)` from the single VJP accumulation point, with `F` checked against finite differences of `U` | FD relative error `1.556e-09` vs threshold `1e-05`; `sum_j F_j` at `5.395e-16` relative; 558 listed pairs, 431 inside `rc`, **0 missing / 0 extra** vs brute force; `U = -63.839860114617`; cutoff-mode table (`:trunc` discontinuous, `:shift` C^0 not C^1, `:lfs` force-consistent) | ~3.3 s |
| `02_dynamics_error_control.jl` | Velocity Verlet on a 2-particle harmonic bond, interrogated with the three diagnostics the core provides (drift statistic, reversibility, offline `dt` ladder) plus one measured naive-adaptive control run | `drift_fraction = 0.00206` (threshold `0.5`); round-trip reversibility `2.220e-15` over 100 000 steps (threshold `1e-9`); fitted convergence order `2.000`; `select_dt` -> `0.0200`; adaptive round trip `2.641e-05`, i.e. `3.5e+09`x worse than fixed | ~3.5 s |
| `03_sampling_double_well.jl` | HMC and replica exchange against targets whose answers are known exactly: a Gaussian with `Var[x] = 1/(beta k)`, a 1-D double well with a quadrature CDF. Where the reported precision is right, and where it is wrong | 14 gates, 0 failed; leapfrog involution `2.220e-16`; plain HMC at a 16 kT barrier: **0** crossings, `<x> = -1.975`; 9-replica exchange: **1720** crossings, `<x> = -0.0395` within `0.284`; trapped chain fails the CDF test (`D = 0.5001` vs critical `0.0068`) while its own error bar looks excellent | ~8.5-9.8 s |
| `04_polarization.jl` | Induced dipoles: solve `A mu = E`, check `mu_i = alpha_i E_local,i`, and test the envelope theorem (fixed-`mu` force vs gradient of the re-solved energy); stability of the interaction tensor | 8 gates, 0 failed; solver residual `1.70e-16`; `mu = alpha E_loc` to `6.94e-18`; envelope gap floor `2.94e-11` at `h = 1e-5` (threshold `1e-5`); force error slope `1.00` and energy error slope `2.00` in the dipole-convergence error; undamped critical radius `1.587401` == `(2*alpha)^(1/3)` | ~4.2 s |
| `equilibrium_sensitivity.jl` | HMC samples plus analytic factor gradients into fixed-measure sensitivity estimates; harmonic exact answer, toy two-leg subtraction and charge-tangent projection | 0 failed gates in current run; ligand `dA/dk = 0.124308 ± 0.001381` vs exact `0.125`; toy two-leg z = `+0.34`; no physical binding claim | varies |

Times are single runs on one machine (1 vs 4 Julia threads changes only the scripts' own
wall-time line); expect them to vary with load.

## What each script gates on

A gate is a numeric threshold checked in the script, not a prose claim. The measured value is
always printed next to it.

### `01_energy_forces.jl`

| Check | Threshold | Measured |
| --- | --- | --- |
| Configuration is fit for finite differences: closest pair `r_min` | `> 0.70` sigma | `1.058` sigma |
| Cell-list pair list vs exhaustive enumeration: missing pairs, and listed pairs beyond `rc + skin` | 0 and 0 | 0 missing, 0 extra (558 listed, 431 within `rc`) |
| `F == -grad(U)` by central differences (`h = 1e-6`, frozen pair list), relative | `<= 1e-05` | `1.556e-09` |
| Newton's third law in the accumulator: largest component of `sum_j F_j`, over the largest force component | `<= 1e-12` | `5.395e-16` |
| Per-factor breakdown vs the driver's own energy: absolute difference of the two sums | `<= 1e-09` | `7.105e-15` |

Printed for context, not gated: the 127 skin pairs contribute exactly `0.0` and
`needs_rebuild!` trips at `skin/2 = 0.150` (false fresh, true after a 0.180 displacement); the
cutoff-mode table (`:trunc` jumps by `U(rc)` across the cutoff, `:shift` is C^0 not C^1, only
`:lfs` is force-consistent); and a synthetic cancellation series `[1e16, 2e5 x 1.0, -1e16]`
where naive summation errs by `1.060e+02` while Kahan and Neumaier err by `0.000e+00` - the real
50-atom breakdown does **not** stress summation, and the script says so.

### `02_dynamics_error_control.jl`

| Check | Threshold | Measured |
| --- | --- | --- |
| `drift_fraction` (slope of the compensated energy series extrapolated over the run, against the oscillation band) | `<= 0.5` | `0.002060` |
| The same drift report's boundedness flag | must say bounded | `bounded = true` (band `4.495e-05`, slope `1.852e-10`) |
| Time reversibility: 50 000 steps out, negate `p`, 50 000 back; round-trip position error over the coordinate scale | `<= 1e-9` | `2.220e-15` (farthest excursion `0.9466`, so the round trip is not trivial; the printed momentum counterpart is `2.498e-14`) |
| Velocity-Verlet order from global position error vs the analytic `x(t)` | `1.8 < order < 2.2` | fitted `2.000` |
| `select_dt` over a ladder of fixed steps: some `dt` must satisfy step-doubling defect `<= 1e-5` AND `drift_fraction <= 0.5` | at least one accepted | `dt = 0.0200` selected (defect `1.004e-06`); `dt = 0.0500` fails on the defect at `1.580e-05`; re-running the selection gives the same `dt` |

Printed for context, not gated: the detector calibration on synthetic series of the same length
and band (bounded oscillation `0.001909` true; pure linear leak `0.5000` **false**; leak +
oscillation `0.3749` **`bounded = true` - a deliberate false negative**, shown because a small
secular leak hides under the band); and the naive adaptive-`dt` control run (round trip
`2.641e-05` vs `7.550e-15` fixed, i.e. `3.5e+09`x worse, with `dt` shrunk on 20 000 of 20 000
steps). The adaptive numbers are reported rather than gated because the claim being tested is
"reversibility is lost", which the measurement shows directly - there is no threshold to pass.

### `03_sampling_double_well.jl` (14 gates, listed in the order printed)

| Gate | Condition in the script | Measured |
| --- | --- | --- |
| `[involution]` | `dev < 1e-12` | `2.22e-16`; at q of about 1.7 the roundoff floor is `2.2e-15`. The momentum negation is what makes the Metropolis ratio exact |
| `[ergodic dt hits the known answer]` | `abs(zeff) < 4` at `dt = 0.08` | `z = -0.5` naive sigma, `-0.5` ESS-corrected, against the exact Gaussian variance |
| `[correlated dt explained by ESS]` | `abs(z) > 4 && abs(zeff) < 4` at `dt = 0.02` | naive `z = 5.0`, `tau = 23.6`: a wrong *error bar*, not a wrong answer |
| `[the leapfrog map predicts rho_1]` | signs agree AND `maxmapdev < 0.15` AND the two worst rows are the two lowest-acceptance rows | max miss `0.116`, misses at `dt = 0.5, 0.55`; `rho_1` is the map's own phase advance `cos(Omega*L)`, predicted not fitted - rejection is the only thing the deterministic formula omits |
| `[ESS correction is calibrated where it is valid]` | `0.7 < scatter ratio < 1.4` at `dt = 0.02` | measured / tau-predicted seed scatter `= 1.06` |
| `[ESS correction fails when correlations alternate]` | `scatter ratio > 4` at `dt = 0.45` | `16.0x` what `tau = 1.0` allows: the initial-positive-sequence estimator truncates an alternating correlation |
| `[resonant bias is NOT established here]` | six-seed mean within `3` sigma of the exact `0.25` | `0.23095 +- 0.01506`, i.e. `-1.3 sigma` - the single run's `15` sigma reading was an error-bar artifact, so no bias is claimed |
| `[plain HMC stays trapped]` | `crossings == 0 && abs(mean(x)) > 1.5` | 40 000 samples at a 16 kT barrier: `0` crossings, `<x> = -1.975`, while looking fully converged |
| `[replica exchange escapes]` | `crossings > 100 && abs(mean(x)) < 5*sigma` | 9 replicas: `1720` crossings, `<x> = -0.0395` inside `0.284` |
| `[trapped chain fails the CDF test]` | `D > 0.25` | `D = 0.5001` against an ESS-corrected critical value of `0.0068` |
| `[the trapped chain is uncorrelated AND wrong]` | `D > 0.25 && tau_block < 1.5 && rho_1 < 0` | blocking `tau = 0.69`, `rho_1 = -0.217`, yet `D = 0.5001`: ESS measures noise, not bias |
| `[the anti-correlation is expected, not predicted]` | both negative AND `abs(measured - predicted) < 0.10` | `U''(-2) = 32` gives `cos = -0.186` (leapfrog dispersion `-0.147`) vs measured `-0.217`; the **sign** follows, the magnitude is not attributed |
| `[PT chain matches the CDF]` | `D < crit_ess` | `D = 0.01305 < 0.03858` |
| `[ladder ranking is seed-stable]` | ESS spread `> 10%` AND every crossing spread `< 10%` AND `crossings(R=3) < crossings(denser)` | ESS spread up to `29%` peak-to-peak, crossing spreads all `< 10%`, `R = 3` at `786` vs `>= 1139` for denser ladders: the crossing count is the seed-stable quantity, a single run's ESS is not |

### `04_polarization.jl` (8 gates)

| Gate | Condition in the script | Measured |
| --- | --- | --- |
| `[solve converged]` | `converged` AND the solver's residual `< 1e-10` AND a residual recomputed here from `A`, `E`, `mu` `< 1e-10` | both `1.70e-16` |
| `[spd flag matches the spectrum]` | `spd == (min eig > 0)` | `spd = true`, `min eig = +1.9809`, `cond = 1.711`; the spectrum is computed here because the solver does not fill it - see below |
| `[mu_i = alpha_i * E_local,i]` | max deviation `< 1e-10` | `6.94e-18` |
| `[envelope theorem holds]` | best relative gap over the `h` sweep `< 1e-5` | `2.94e-11` at `h = 1e-5`; the sweep shows clean second-order slopes down to that roundoff floor |
| `[force error scales linearly with mu error]` | every fitted slope in `(0.7, 1.3)` | slopes `1.00`, `1.00` in the deliberate dipole-convergence error |
| `[energy error scales quadratically]` | every fitted slope in `(1.7, 2.3)` | slopes `2.00`, `2.00` - so at `eps = 1e-4` the energy is off by `1.1e-05` relative while the forces are off by `1.3e-04` |
| `[tensor convention matches closed form]` | difference from the closed form `< 1e-6` | undamped `r_c = 1.587401 == (2*alpha)^(1/3)`; the damped value `1.579400` buys under 1% of that separation and does not cure the instability |
| `[indefinite A is surfaced, not hidden]` | `!spd && method === :lstsq` | at `r = 0.8`: `spd = false`, `method = :lstsq`, and the resulting energy fall (`1/2 lambda_min t^2`) is measured rather than asserted |

## Reproducibility

All four are seeded and use fixed sample/step counts, so the printed numbers are the numbers:
no gate depends on wall time, every threshold is printed next to the value it measured, and
repeated runs give identical output. On this machine the output is byte-identical under
`JULIA_NUM_THREADS=1` and `JULIA_NUM_THREADS=4` except for each script's own wall-time line
(verified for all four). Changing a seed in `03` does change its sampling results - which is the
point of its `[ladder ranking is seed-stable]` gate: it reports which diagnostics survive a seed
change and which do not.

## What these examples do NOT demonstrate

Stated plainly, because the passing gates above are easy to over-read:

- **No long protein or biomolecular trajectories.** No solvent, no real force-field
  parameterisation, no equilibrated thermodynamics. `01` is a 50-atom unit-scale fixture;
  `03` uses 1-D targets precisely because their answers are known - agreement there is not
  evidence about a high-dimensional free-energy landscape.
- **No long-range electrostatics: no Ewald, no PME, no FMM.** `01` uses a cell list and a
  switching function; `04`'s dipole operator is a direct O(N^2) minimum-image sum. A switched
  Coulomb and a lattice sum are different models, not versions of one another.
- **Constraints are narrow.** Pair-distance RATTLE is tested, but SHAKE, SETTLE, and general
  coupled constraint networks are not validated. No example composes constrained periodic
  electrostatics end-to-end.
- **Triclinic simulation is partial.** Nearest-image geometry and wrapping are tested, but
  triclinic neighbor stencils and force consumers remain unsupported; examples use orthorhombic
  boxes only.
- **No production NPT or barostat validation.** A standalone configurational volume-move prototype
  is tested algebraically and for reproducibility, but has no pressure-convergence evidence.
  `02` and `03` are microcanonical
  integration and Metropolis-weighted sampling respectively; nothing here samples a canonical
  molecular system through a heat bath, and `03`'s sampler sweeps are not physical time - the
  crossing counts measure mixing, not kinetics.
- **No polarizable force field for a real material.** `04` is an interface: no charge transfer,
  no metallic/band-structure response, no magnetism, no non-exponential Thole damping family,
  no periodic dipole Ewald sum, no applied external field (a one-line addition to `E`, not
  implemented), no analytic tensor derivatives (its induced forces are finite
  differences of the quadratic form - correct, not fast), and no AMOEBA/Drude parameterisation.
  `04` prints the module's own `extension_boundaries()` table for exactly this list.
- **No AI/ML content of any kind**, and no convergence certificates: finite runs cannot prove
  convergence, and "the numbers stopped moving" is precisely the failure mode `03` is built to
  show is undetectable from inside a single run.

Each script also carries its own, sharper `not shown, and not implied` block at the end of its
output; those blocks are the authoritative version of this list per example.

## Things the examples work around, so you do not mistake them for bugs

- **`Polarizable.InducedDipoleResult.min_eig` is always `NaN` and `n_iter` is always `0` for the
  direct solver** (`solve_dipoles!` never assigns them). `04` says so in its own output and
  computes the spectrum itself with `eigvals` instead of reading those fields as if they meant
  something.
- **`Sampling.doublewell_cdf_exact` costs O(probes x 400 001) quadrature.** `03` keeps the probe
  count small (121 probes ~ 0.5 s) and linearly interpolates the exact CDF between probes for
  the KS comparison; that interpolation is an approximation and `03` labels it as one.
- The KS critical value scaled by the effective sample size is a **heuristic**, not an exact test
  for correlated chains, and `03` prints it as a heuristic.

## Files

```
examples/
  01_energy_forces.jl            pair lists, energy/force consistency, cutoff conventions
  02_dynamics_error_control.jl   drift statistic, reversibility, offline dt validation
  03_sampling_double_well.jl     HMC / replica exchange against known 1-D targets
  04_polarization.jl             A mu = E, the envelope theorem, tensor stability
  equilibrium_sensitivity.jl     fixed-measure equilibrium parameter sensitivity
```

These scripts read `../julia/src/` and write nothing anywhere; they create no files, take no
arguments, and require no environment variables beyond a Julia that can resolve the project.
