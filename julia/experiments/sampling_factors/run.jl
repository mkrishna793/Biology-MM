# Molecular HMC + parallel tempering on real factor energies.
#
#     cd julia
#     JULIA_DEPOT_PATH=~/.cache/biology-mm/julia-depot \
#       julia --project=. experiments/sampling_factors/run.jl
#
# Two experiments:
#   1. a homogeneous Lennard-Jones fluid (N = 32, rho = 0.8, T = 1.0) sampled
#      with HMC on the factor energy, compared against a long BAOAB run at the
#      same temperature;
#   2. a two-cluster barrier model where a single cold chain is trapped in one
#      well and parallel tempering restores exchange, with barrier crossings,
#      walker round trips and the exact radial-marginal well population.
#
# Everything is measured, including the things that are disappointing. No
# physical (biomolecular) claim is made: these are reduced-unit model systems.

include(joinpath(@__DIR__, "models.jl"))

using .SamplingFactorModels
using FoundationCore
using FoundationCore: Types, Factors, Dynamics, Sampling
using Random, Statistics, LinearAlgebra, Printf

const SA = FoundationCore.Sampling
const D = FoundationCore.Dynamics

# ---------------------------------------------------------------------------
# exact reference for the barrier model
# ---------------------------------------------------------------------------

"""
Exact fraction of the radial marginal `p(r) ∝ r² exp(-βU(r))` that lies in the
inner well `r < rmid`. The internal cluster modes factor out of the marginal
(the potential is `U_dw(r) + U_internal`), so this is exact for the model, not
an approximation. Checked by doubling the quadrature resolution.
"""
function exact_inner_fraction(a, rmid, d, T; lo = 0.02, hi = 9.0, n = 200_001)
    u(r) = a * ((r - rmid)^2 - d^2)^2
    function fraction(nn)
        rr = range(lo, hi, length = nn)
        h = step(rr)
        inner = 0.0
        outer = 0.0
        for (i, r) in enumerate(rr)
            w = (i == 1 || i == nn) ? 1.0 : (isodd(i) ? 4.0 : 2.0)
            p = w * r^2 * exp(-u(r) / T)
            if r < rmid
                inner += p
            else
                outer += p
            end
        end
        return inner / (inner + outer)
    end
    f1 = fraction(n)
    f2 = fraction(2n - 1)
    return f1, abs(f1 - f2)
end

# ---------------------------------------------------------------------------
# experiment 1: LJ fluid, HMC vs BAOAB
# ---------------------------------------------------------------------------

function lj_benchmark(; N = 32, rho = 0.8, T = 1.0, rc = 2.5,
                      dt_baoab = 0.002, eq_steps = 100_000, prod_steps = 200_000,
                      dt_hmc = 0.004, L_hmc = 100, n_hmc = 4000, warm_hmc = 100,
                      check_steps = 100_000)
    println("="^78)
    println("EXPERIMENT 1  Lennard-Jones fluid: molecular HMC vs BAOAB")
    println("="^78)
    sys, q0, Lbox = lj_fluid_system(N, rho; rc = rc)
    U0 = Dynamics.potential_energy(sys, q0)
    @printf("system: N=%d  rho=%.2f  L=%.3f  rc=%.2f  T=%.2f  pairs=%d  U0/N=%.2f\n",
            N, rho, Lbox, rc, T, N * (N - 1) ÷ 2, U0 / N)

    t0 = time()
    Ueq, qeq = baoab_run(sys, q0, T, dt_baoab, eq_steps; gamma = 1.0, seed = 11)
    t_eq = time() - t0
    @printf("equilibration: BAOAB %d steps (dt=%.3f, gamma=1), %.2f s, U/N last half=%.3f\n",
            eq_steps, dt_baoab, t_eq, mean(Ueq[(eq_steps ÷ 2):end]) / N)

    t1 = time()
    Ubao, qprod = baoab_run(sys, qeq, T, dt_baoab, prod_steps; gamma = 1.0, seed = 12)
    t_bao = time() - t1
    bao = SA.correlated_mean(Ubao)
    ess_bao = SA.effective_sample_size(Ubao)
    @printf("BAOAB production: %d steps (%.1f time units), %.2f s\n",
            prod_steps, prod_steps * dt_baoab, t_bao)
    @printf("  U/N = %.4f +/- %.4f (autocorr SE)   tau = %.1f steps (%.3f time units)   ESS = %.1f\n",
            bao.mean / N, bao.se / N, bao.tau, bao.tau * dt_baoab, ess_bao)
    @printf("  ESS / force evaluation = %.3e   (1 force eval per step)\n",
            ess_bao / prod_steps)

    t2 = time()
    hmc = SA.hmc_trajectory(sys, qeq, n_hmc; T = T, dt = dt_hmc, L = L_hmc,
                            nwarmup = warm_hmc, rng = MersenneTwister(13))
    t_hmc = time() - t2
    hm = SA.correlated_mean(hmc.U)
    ess_hmc = SA.effective_sample_size(hmc.U)
    @printf("HMC production: %d samples (dt=%.3f, L=%d, trajectory %.2f time units), %.2f s\n",
            n_hmc, dt_hmc, L_hmc, dt_hmc * L_hmc, t_hmc)
    @printf("  acceptance = %.3f   U/N = %.4f +/- %.4f   tau = %.1f samples (%.3f time units)\n",
            hmc.acceptance_rate, hm.mean / N, hm.se / N,
            hm.tau, hm.tau * dt_hmc * L_hmc)
    @printf("  ESS = %.1f   ESS / force evaluation = %.3e   (%d force evals)\n",
            ess_hmc, ess_hmc / hmc.force_evals, hmc.force_evals)

    z = (hm.mean - bao.mean) / sqrt(hm.se^2 + bao.se^2)
    @printf("comparison: Delta(U/N) = %+.4f  combined SE = %.4f  z = %+.2f\n",
            (hm.mean - bao.mean) / N, sqrt(hm.se^2 + bao.se^2) / N, z)
    @printf("cost: HMC %.2e ESS/force-eval vs BAOAB %.2e  (ratio %.2f)\n",
            ess_hmc / hmc.force_evals, ess_bao / prod_steps,
            (ess_hmc / hmc.force_evals) / (ess_bao / prod_steps))

    # BAOAB has an O(dt^2) discretisation bias; HMC has none (Metropolis makes
    # the target exact). A halved-dt BAOAB run quantifies the size of that bias
    # so a small HMC/BAOAB difference is not over-interpreted.
    Ub2, _ = baoab_run(sys, qprod, T, dt_baoab / 2, check_steps; gamma = 1.0, seed = 14)
    bao2 = SA.correlated_mean(Ub2)
    @printf("BAOAB dt=%.4f bias check: U/N = %.4f +/- %.4f (%.1f time units)\n",
            dt_baoab / 2, bao2.mean / N, bao2.se / N, check_steps * dt_baoab / 2)

    return (N = N, rho = rho, T = T, U_hmc = hm.mean / N, U_hmc_se = hm.se / N,
            U_baoab = bao.mean / N, U_baoab_se = bao.se / N, z = z,
            U_baoab_dt2 = bao2.mean / N, U_baoab_dt2_se = bao2.se / N,
            acc_hmc = hmc.acceptance_rate, ess_hmc = ess_hmc,
            ess_baoab = ess_bao, evals_hmc = hmc.force_evals,
            evals_baoab = prod_steps, tau_hmc_time = hm.tau * dt_hmc * L_hmc,
            tau_baoab_time = bao.tau * dt_baoab)
end

# ---------------------------------------------------------------------------
# experiment 2: two-cluster barrier, cold-chain trap vs tempering
# ---------------------------------------------------------------------------

function barrier_demo(; Tmin = 0.35, temperatures = [0.35, 0.5, 0.7, 1.0, 1.5, 2.2, 3.2],
                      dt = 0.005, L = 100, nsamples = 4000, warmup = 300,
                      a = 8.0, rmid = 2.5, d = 0.9, rstar = 2.5)
    println("")
    println("="^78)
    println("EXPERIMENT 2  Two-cluster barrier: single cold chain vs tempering")
    println("="^78)
    sys, q0 = two_cluster_system(; a = a, rmid = rmid, d = d)
    exact_inner, quad_err = exact_inner_fraction(a, rmid, d, Tmin)
    @printf("model: 2 clusters x 3 atoms, harmonic bonds + intra-cluster LJ\n")
    @printf("       separation double well U(r)=%.1f((r-%.1f)^2-%.1f^2)^2\n", a, rmid, d)
    @printf("       wells r=%.1f and r=%.1f, barrier %.1f at r=%.1f;  r* = %.2f\n",
            rmid - d, rmid + d, a, rmid, rstar)
    @printf("       T_ladder = %s\n", string(temperatures))
    @printf("       cold barrier/kT = %.1f\n", a / Tmin)

    # single cold chain, same temperature, same dt/L, same number of stored moves
    cold = SA.hmc_trajectory(sys, q0, nsamples; T = Tmin, dt = dt, L = L,
                             nwarmup = warmup, rng = MersenneTwister(21))
    rs = separation_series(cold.samples)
    inner_flag = Float64.(rs .< rstar)
    single_inner = SA.correlated_mean(inner_flag)
    single_cross = separation_crossings(rs, rstar)
    @printf("\nsingle cold chain (T=%.2f): acceptance=%.3f\n", Tmin, cold.acceptance_rate)
    @printf("  barrier crossings = %d   inner-well fraction = %.3f +/- %.3f (corr SE)\n",
            single_cross, single_inner.mean, single_inner.se)
    @printf("  exact equilibrium inner-well fraction = %.4f (quadrature err %.1e)\n",
            exact_inner, quad_err)

    # parallel tempering at the same cold temperature
    t0 = time()
    pt = SA.replica_exchange(sys, q0, nsamples; temperatures = temperatures,
                             dt = dt, L = L, swap_every = 2, nwarmup = warmup,
                             rng = MersenneTwister(22))
    elapsed = time() - t0
    cold_r = [cluster_separation(reshape(pt.samples[1][:, k], 3, 6)) for k in 1:nsamples]
    hot_r = [cluster_separation(reshape(pt.samples[end][:, k], 3, 6)) for k in 1:nsamples]
    pt_inner = SA.correlated_mean(Float64.(cold_r .< rstar))
    pt_cross = separation_crossings(cold_r, rstar)
    hot_cross = separation_crossings(hot_r, rstar)
    trips = walker_round_trips(pt.replica_index, 1)
    sw = SA.swap_acceptance(pt)

    @printf("\nparallel tempering (%d replicas, %d sweeps, %.2f s):\n",
            length(temperatures), nsamples, elapsed)
    @printf("  acceptance per slot: %s\n", string(round.(pt.acceptance, digits = 3)))
    @printf("  swap acceptance per adjacent pair: %s\n", string(round.(sw, digits = 3)))
    @printf("  total force evaluations = %d\n", pt.force_evals)
    @printf("  COLD slot: barrier crossings = %d   inner-well fraction = %.3f +/- %.3f\n",
            pt_cross, pt_inner.mean, pt_inner.se)
    @printf("  HOT  slot: barrier crossings = %d\n", hot_cross)
    @printf("  cold->hot->cold walker round trips = %d\n", trips)
    @printf("  exact inner-well fraction = %.4f\n", exact_inner)
    zc = (pt_inner.mean - exact_inner) / max(pt_inner.se, 1e-12)
    @printf("  cold-slot inner fraction vs exact: z = %+.2f\n", zc)

    println("\nreading:")
    if single_cross == 0 && pt_cross > 0
        @printf("  the cold chain never crosses (barrier %.1f = %.0f kT at T=%.2f) and sits in the\n",
                a, a / Tmin, Tmin)
        @printf("  wrong well: inner fraction %.2f vs exact %.2f. Tempering produces %d crossings\n",
                single_inner.mean, exact_inner, pt_cross)
        @printf("  at the SAME cold temperature, moving the cold slot toward the exact marginal.\n")
    else
        println("  cold chain was not trapped at this setting; the comparison is not sharp.")
    end

    return (a = a, Tmin = Tmin, single_cross = single_cross,
            single_inner = single_inner.mean, single_inner_se = single_inner.se,
            pt_cross = pt_cross, pt_inner = pt_inner.mean, pt_inner_se = pt_inner.se,
            exact_inner = exact_inner, z = zc, trips = trips, swap = sw,
            acceptance = pt.acceptance, evals = pt.force_evals)
end

function main()
    lj = lj_benchmark()
    bar = barrier_demo()
    println("")
    println("="^78)
    println("Both experiments used Sampling.hmc_trajectory / replica_exchange on")
    println("HarmonicSystem with real AbstractFactor forces. This is not a")
    println("biomolecular benchmark: no PME, no constraints, no water, N=32 and N=6.")
    println("="^78)
end

main()
