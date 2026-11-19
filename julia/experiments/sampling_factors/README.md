# Molecular HMC and parallel tempering on real factor energies

This directory connects `Sampling`'s HMC / replica-exchange to the **real
molecular energy field**: `Dynamics.HarmonicSystem`, whose `U(q)` and `-∂U/∂q`
come from `Dynamics.forces!` over any `Factors.AbstractFactor` set. Until this
change, HMC only accepted analytic closures (`U`, `gradU`) on a 1-D double well
and had never touched a molecular system.

The library code is `julia/src/Sampling.jl` (molecular backend added;
the closure API is unchanged). Tests are in
`julia/test/test_sampling.jl`.

```bash
cd julia
JULIA_DEPOT_PATH=~/.cache/biology-mm/julia-depot \
  ~/.cache/biology-mm/julia-1.13.3/bin/julia --project=. \
  experiments/sampling_factors/run.jl
```

Runtime is about 50 s on the development machine (most of it is the `N = 32`
Lennard-Jones reference runs).

## What is sampled

`hmc_trajectory(s::HarmonicSystem, q0, nsamples)` runs `L` leapfrog steps of size
`dt` on the factor energy, with momenta drawn from `N(0, M k_B T)`, and accepts
with the full Hamiltonian `H = ½pᵀM⁻¹p + U(q)`. The proposal is

    T = (momentum negation) ∘ (L leapfrog steps),

which is an exact involution and volume preserving; both are verified
numerically on the molecular code path (not assumed to carry over from the
closure path). The target is therefore exactly `exp(-βU(q))dq` for any
force-consistent factor set.

`replica_exchange(s::HarmonicSystem, q0, nsamples; temperatures, ...)` runs the
same move on a temperature ladder and exchanges adjacent slots with

    min(1, exp((β_i − β_j)(U_i − U_j))).

The same formula is exported as `swap_log_ratio` / `swap_accept_probability` and
is tested against an exactly enumerable two-state target.

Units are the system's own: `Sampling.molecular_beta(s, T) = 1/(k_B T)`, with
`k_B = 1` for `Types.REDUCED`.

## Experiment 1 — Lennard-Jones fluid, HMC vs BAOAB

All-pairs Lennard-Jones in reduced units, `ε = σ = m = 1`, `N = 32`,
`ρ = 0.8`, `L = 3.420`, force-consistent `:lfs` cutoff `rc = 2.5`, `T = 1.0`.
The fluid is equilibrated with BAOAB (`dt = 0.002`, `γ = 1`, 100 000 steps),
then BAOAB production is 200 000 steps and HMC production is 4 000 samples
(`dt = 0.004`, `L = 100`, i.e. 0.4 time units per sample), both started from the
same equilibrated configuration.

| quantity | HMC (factor energy) | BAOAB |
| --- | --- | --- |
| acceptance | 0.993 | — (no Metropolis test) |
| `U/N` | **−4.0050 ± 0.0043** | **−3.9949 ± 0.0072** |
| autocorrelation time of `U` | 2.5 samples = 1.02 time units | 352 steps = 0.70 time units |
| ESS of `U` | 1570.9 | 568.0 |
| force evaluations | 414 100 | 200 000 |
| **ESS per force evaluation** | **3.79 × 10⁻³** | **2.84 × 10⁻³** |

Difference `Δ(U/N) = −0.0102`, combined SE `0.0084`, `z = −1.22`. A BAOAB run
with the timestep halved (`dt = 0.001`, 100 000 steps) gives
`U/N = −4.0095 ± 0.0128`, i.e. it moves toward the HMC value. That is the
expected direction: BAOAB carries an `O(dt²)` discretisation bias in its
invariant measure, while HMC's Metropolis correction makes the target exact.
The remaining difference is therefore consistent with both statistics and a
small integration bias; it is **not** evidence that either sampler is wrong.

At this state point HMC delivers about 1.34× the ESS per force evaluation of
BAOAB. That is a statement about this system, timestep and trajectory length,
not a general ranking: tuned BAOAB (larger `dt`, different `γ`) or a different
HMC trajectory length would change the ratio.

The raw `U/N` is the cutoff energy of the `:lfs` model (no long-range tail
correction), so it is **not** directly comparable to a full-LJ literature value
with a tail correction. The comparison here is HMC against BAOAB on the same
potential, which is the point of the experiment.

## Experiment 2 — two-cluster barrier, cold-chain trap vs tempering

The model has two three-atom clusters, each held by three harmonic bonds
(`k = 200`, `r0 = 1`) and internal Lennard-Jones pairs. The only inter-cluster
interaction is a **coarse-grained** separation potential

    U(r) = 8.0 ((r − 2.5)² − 0.9²)²

on the distance `r` between the cluster centres of mass: two degenerate wells at
`r = 1.6` and `r = 3.4`, and a barrier of height `8.0` at `r = 2.5`. This is a
model potential chosen to give a controlled symmetric barrier; it is not a
physical inter-cluster force field. The clusters themselves are real bonded/LJ
factor systems.

Because the potential separates as `U_dw(r) + U_internal`, the exact marginal of
`r` is `p(r) ∝ r² exp(−βU_dw(r))`, whose inner-well fraction can be computed by
quadrature. That is an independent analytic reference, not another sampler.

At `T = 0.35` the barrier is `8.0 = 22.9 kT`.

* **Single cold HMC chain**, `T = 0.35`, 4 000 moves (`dt = 0.005`, `L = 100`):
  acceptance 0.996, **0 barrier crossings**, inner-well fraction **1.000**.
  The exact equilibrium inner-well fraction is **0.1849**. The chain is trapped
  in the wrong well and its distribution is simply wrong.
* **Parallel tempering**, ladder `[0.35, 0.5, 0.7, 1.0, 1.5, 2.2, 3.2]`,
  4 000 sweeps, swaps every 2 sweeps, 3 040 107 force evaluations:
  * per-slot acceptance 0.994–0.997;
  * adjacent-pair swap acceptance 0.587–0.660;
  * **cold slot: 376 barrier crossings**, hot slot: 573 crossings;
  * cold-slot inner-well fraction **0.186 ± 0.034** (autocorrelation SE),
    vs exact **0.1849**, `z = +0.03`;
  * **38 cold → hot → cold walker round trips** through the ladder.

The cold slot samples the exact marginal, while a single chain at the *same*
temperature never crosses. That is the intended demonstration: tempering does
not change the target, it restores the exchange that a trapped chain lacks.

## Tests added to `test/test_sampling.jl`

1. `factor_energy_hmc_harmonic` — a 1-D harmonic factor through
   `HarmonicSystem`/`forces!`; mean and variance match `x0` and `T/k` within
   autocorrelation-corrected SEs (acceptance 0.998, 20 000 samples).
2. `molecular_proposal_involution_volume` — molecular proposal on a LJ dimer:
   involution to `5.6e-17`, numerical `|det J| = 1` to `3e-10`.
3. `acceptance_matches_formula` — observed acceptance 0.7832 vs
   `mean(min(1, exp(−βΔH))) = 0.7827`, binomial SE 0.0029. The configuration
   deliberately has ~22 % rejections; an all-accept run would test nothing.
4. `swap_two_state_exact` — swap acceptance on an exactly enumerable two-state
   target: exact 0.732863 vs observed 0.733845, SE 0.000989. The sampler's own
   swap loop is then checked against the recorded per-attempt energies
   (`swap_u_low`/`swap_u_high`) in `molecular_pt_swap_statistics`.
5. `constraints_refused` — `hmc_trajectory`, `hmc_step!` and `replica_exchange`
   all throw `SamplingConstraintError` when given constraints;
   `sampling_constraints_supported == false`.

The original six closure-based tests are unchanged and still pass, so the
closure-based API is backward compatible.

## Constraints: refused, not silently mishandled

A constrained HMC proposal must preserve the measure on the constraint manifold.
Projecting positions at the end of a leapfrog step and using the unconstrained
`exp(−βΔH)` acceptance is not correct: the projected map is not the time
reversal of itself with the right Jacobian, and the manifold measure
`√det G(q) dσ` is not the unconstrained Lebesgue measure. `Dynamics` implements
RATTLE for deterministic trajectories, but that is a different goal.

The molecular entry points therefore accept a `constraints` keyword and throw
`SamplingConstraintError` for any non-`nothing` value. This cannot catch a
caller who *has* a constrained physical model but omits the keyword — the
docstrings say so explicitly. That case remains the caller's responsibility
until a measure-preserving constrained proposal is implemented.

## What is not validated

* **Not biomolecular.** `N = 32` LJ particles and a 6-atom model are not a
  protein, water box, or membrane. No PME, no water, no ions, no constraints, no
  barostat. Nothing here says anything about folding, kinetics, or rates.
* **One LJ state point.** `ρ = 0.8`, `T = 1.0`, `rc = 2.5`, one system size.
  No finite-size or cutoff study. `U/N` is the raw cutoff energy without a
  long-range tail correction.
* **One HMC tuning.** `dt = 0.004`, `L = 100`. The ESS/force-eval ratio depends
  on the trajectory length and on the (fully Metropolis-corrected) step size;
  it was not optimised.
* **Error bars are estimates.** `integrated_autocorrelation` is the
  initial-positive-sequence estimator; its own uncertainty grows with `τ`. The
  reported SEs are therefore approximate, and the HMC/BAOAB agreement at
  `z = −1.22` should be read as "consistent within these estimates", not as a
  certified equality.
* **The barrier model is coarse-grained.** The separation double well is a
  model potential, not a physical inter-cluster interaction. The *exact*
  inner-well fraction is exact for that model (the internal modes factorise);
  it is not a physical free-energy benchmark.
* **Single-chain connectivity.** `hmc_trajectory` uses one chain per call; there
  is no multi-chain diagnostic, no split-R̂, and no proof of ergodicity. Finite
  runs are evidence, not proof.
* **The tempering demo is one seed.** The crossing and round-trip counts are
  from a single reproducible run, not a distribution over seeds.
