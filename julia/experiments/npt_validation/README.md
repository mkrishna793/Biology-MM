# NPT barostat validation (isotropic Monte-Carlo volume moves)

Status: **ideal-gas limit validated; scoped interacting tests pass in the pre-change audit.**
`MonteCarloBarostat` is exported by `FoundationCore`. The 34-suite pre-change
audit passed interacting LJ and 64-water TIP3P NPT checks with supported-factor
box derivatives. A current targeted rerun passes; final frozen-snapshot verification is pending (see
[`validation/STATUS.md`](../../../validation/STATUS.md)). No replicated production
NPT trajectory or bulk compressibility result is qualified. The low-level kernel
and `Workflow` have distinct constraint gates; `Workflow` refuses constrained NPT.

## What was built

`julia/src/MonteCarloBarostat.jl`

* `VolumeMoveBarostat` / `step!` — one isotropic log-volume Metropolis move.
  Proposes `dlogV ~ U(-dmax, dmax)`, sets `s = exp(dlogV/3)`, scales **both**
  positions (`q → s*q`) and the orthorhombic box (`L → s*L`), recomputes the
  energy at the scaled configuration, and accepts with

  ```
  ln a = -beta*(ΔU + P_target*ΔV) + (N+1)*dlogV .
  ```

  `q` is modified only on acceptance; the returned `box` is the new box. The
  realised box volume is used in the ratio, so the Jacobian refers to the state
  the caller adopts.
* `NPTModel(energy; virial, masses)` — bundles the energy, the virial
  `W = Σ_i r_i·F_i` from the *same* force field, and the masses.
* `NPTBarostat(beta, P, dmax, model)` / `npt_step!(barostat, q, p, box, rng)` —
  volume move plus a configurational-pressure record
  `P = (2K + W)/(3V)`. Designed to be called between fixed-volume dynamics
  steps; the experiment composes it with BAOAB (`Dynamics.langevin_baoab_step!`).
* `configurational_pressure(model, q, p, box)` — the mechanical pressure above.
* Typed refusals:
  * `ConstraintIncompatibilityError` — uniform dilation changes every absolute
    bond length by `s`; rescaling constraint targets and re-projecting is not
    implemented, so a non-empty constraint list is refused rather than silently
    violated.
  * `MissingVirialError` — NPT pressure without a virial provider.
  * `MissingMassError` — kinetic pressure without masses.
  * `Types.UnsupportedGeometry` — non-orthorhombic boxes.

`julia/test/test_barostat.jl` — focused checks (below).
`julia/experiments/npt_validation/` — this README, `analysis.jl`
(IAT / block statistics), `run.jl` (full experiment).

## The Jacobian, verified rather than assumed

Work in scaled coordinates `s = q/L` and volume `V`. The classical NPT
configurational weight is

```
pi(s,V) ∝ exp(-beta*(U + P*V)) * V^N .
```

The volume move holds `s` fixed (positions and box are dilated by the same
factor), so this is the marginal the kernel must preserve. (In absolute
coordinates the weight is `exp(-beta(U+PV)) dq dV`; the deterministic dilation
`(q,V) -> (a q, a^3 V)` has Jacobian `a^(3N) * a^3 = (V1/V0)^(N+1)`, giving the
same `N+1` without a `V^N` in the weight.)

The proposal is symmetric in `x = log V`. Changing variables adds `dV/dx = V`,
so the target in proposal coordinates is `exp(-beta*(U+P*V)) * V^(N+1)`. Hence
the `(N+1)ΔlogV` term. Omitting it targets `V^{-1} exp(-beta*P*V)`, which is not
normalisable at fixed proposal step: the sampled volume collapses and the
collapse depends on `dmax`. The negative-control run measures exactly that
(389× and 123× smaller mean volume for `dmax = 0.15, 0.05`, with the two control
runs differing by 3.16×).

## Exact ideal-gas check

For `U = 0`, `p(V)` is exactly `Gamma(shape = N+1, scale = 1/(beta P))`, so

```
<V> = (N+1)/(beta P),   var(V) = (N+1)/(beta P)^2,
var(V)/<V>^2 = 1/(N+1) = kT*kappa_T/<V>   (kappa_T = 1/P for an ideal gas).
```

Note the finite-`N` `N+1`: the classical NPT partition function carries `V^N`,
and the volume mean is `(N+1)/(beta P)`, not `N/(beta P)`. The equation of state
`P = N/(beta V)` holds as an *estimator* statement: `<N/(beta V)> = P` exactly
(and `<2K/(3V)> = P` in the dynamics run). For `N = 32` the difference between
`N/(beta P)` and `(N+1)/(beta P)` is 1.5%; a run matching `N/(beta P)` would be
sampling the wrong measure.

Tolerances used: **4 autocorrelation-corrected SE** for a single run against an
exact value, **3 combined SE** for two-seed agreement, **10%** for fluctuation
magnitudes.

### Result (two seeds, `N=32`, `beta=1`, `P_target=0.5`, 100k thinned samples)

| quantity | exact | seed 2024 | seed 2025 | z |
|---|---|---|---|---|
| `<V>` (pure volume moves) | 66.0 | 66.0593 ± 0.1201 | 66.1345 ± 0.1188 | 0.49, 1.13 |
| `var(V)` | 132.0 | 133.012 ± 1.688 | 132.426 ± 1.461 | 0.60, 0.29 |
| `<N/(beta V)>` | 0.5 | 0.49961 ± 0.00095 | — | 0.41 |
| two-seed `<V>` | — | \|Δ\|/(combined SE) = 0.44 | KS = 0.0052 (5% crit. 0.0061) | — |
| `var/<V>^2` | 0.030303 | 0.030481 | 0.030277 | 0.6%, 0.1% rel. error |

Naive `std/sqrt(n)` is **3.3× too small** here (0.0365 vs 0.1201): the IAT is
`tau ≈ 5.4` thinned samples. All tolerances above use the corrected error.

### Dynamics-loop check (BAOAB + volume moves, 30k thinned samples)

| quantity | exact | seed 2024 | seed 2025 |
|---|---|---|---|
| `<V>` | 66.0 | 66.251 ± 0.213 | 66.135 ± 0.206 |
| `<P_config>` | 0.5 | 0.49952 ± 0.00207 | 0.49840 ± 0.00213 |
| `var(V)` | 132.0 | 132.439 ± 3.149 | 128.709 ± 2.127 |
| two-seed | — | z(`<V>`) = 0.39, z(`<P>`) = 0.38 | KS = 0.0026 (crit. 0.0111) |

`P_config` is the module's `(2K+W)/(3V)` with the analytic ideal-gas `W = 0`.
Both seeds reproduce the target pressure; the two seeds agree within combined
uncertainty.

### Enthalpy stationarity (`H = U + P V`, 20 blocks)

| seed | `<H>` (exact 33.0) | half₁−half₂ | block slope |
|---|---|---|---|
| 2024 | 33.1255 ± 0.1062 | +0.0613 ± 0.2125 (z=0.29) | z=1.35 |
| 2025 | 33.0676 ± 0.1028 | −0.0710 ± 0.2057 (z=0.35) | z=0.14 |

No drift or half-window offset at the stated tolerance.

### Negative control

| run | `<V>` | z vs exact | acceptance |
|---|---|---|---|
| correct (`dmax=0.15`) | 66.0593 | 0.5 | 0.831 |
| no Jacobian (`dmax=0.15`) | 0.1700 | 736.9 | 0.997 |
| no Jacobian (`dmax=0.05`) | 0.5370 | 238.6 | 0.996 |

The Jacobian control is a deliberate corruption of `log_acceptance`, not a
supported mode: the test subtracts `(N+1)*dlogV` from the correct result. Its
step-size dependence is the proof that the term is necessary.

## Test suite

```
cd julia
JULIA_DEPOT_PATH=~/.cache/biology-mm/julia-depot \
  ~/.cache/biology-mm/julia-1.13.3/bin/julia --project=. \
  test/test_barostat.jl
```

9 checks, all passing (~3 s):

1. analytic log-ratio / Jacobian algebra (`N+1`, not `N`);
2. scaled-coordinate invariance and exact proposal reversal (up then inverse
   down returns to the start configuration and box);
3. seeded replay determinism (same seed bit-identical, different seed differs);
4. ideal-gas volume mean/variance/EOS against the exact Gamma moments, plus the
   pressure formula `(2K+W)/(3V)` against a hand-computed value;
5. Jacobian negative control (collapsed and step-size-dependent volume);
6. constraint refusal with `ConstraintIncompatibilityError` (typed, not
   `ArgumentError`);
7. missing-virial and missing-mass refusals with `MissingVirialError` /
   `MissingMassError`;
8. non-orthorhombic geometry refusal;
9. BAOAB+barostat pressure mean and equipartition `<K> = 3N/(2beta)`.

## Verdict

**Validated and exportable for the configurational volume-move kernel in the
exact ideal-gas limit.** The ideal-gas volume distribution, the equation of
state in estimator form, two-seed reproducibility, the fluctuation/compressibility
relation, and enthalpy stationarity all pass at the stated autocorrelation-corrected
tolerances. The Jacobian is demonstrated to be necessary by a decisive negative
control.

**Scoped interacting checks now pass.** The current `test_barostat.jl` rerun
passed LJ and 64-water TIP3P NPT checks. TIP3P pressure was −24.4 ± 68.8 bar
against 1 bar, with broad uncertainty. This does not qualify a replicated
production density/compressibility result. Source files changed during the run;
final frozen-snapshot verification remains assigned to the primary session.

## Limitations

* Orthorhombic boxes only.
* `Workflow` refuses constrained NPT; low-level molecular volume moves and the
  64-water test are a separate path. No qualified production TIP3P NPT trajectory.
* Only the ideal gas has the exact distribution target described above.
  Interacting-system tests use composed analytic virial backends.
* `N = 32`; no finite-size extrapolation. The `N+1` vs `N` distinction is
  explicit, and is 1.5% at this size.
* Statistics use IAT/block errors on one chain per seed; no replica-exchange or
  multiple independent chains beyond the two seeds.
* The pressure estimator requires `W = Σ_i r_i·F_i`; the provider must cover
  every factor in the energy. The module cannot introspect a closure, so a
  backend that silently omits a factor's virial would not be detected here.

## Wiring

The module is written for `FoundationCore`'s namespace: it uses
`using ..Types` and no other package internals (`Types`, `Random` only). The
test and experiment therefore load it with

```julia
module Host
using FoundationCore
const Types = FoundationCore.Types
include(".../src/MonteCarloBarostat.jl")
end
```

so the box type is exactly `FoundationCore.Types.OrthorhombicBox`. To make it a
first-class package submodule (already included/exported), the package uses
after `include("Types.jl")`:

```julia
include("MonteCarloBarostat.jl")
```

`MonteCarloBarostat` is already exported and `test_barostat` is registered in
`run_all.jl`. The `Host` example above describes the historical standalone
experiment harness; no additional package wiring is required.
