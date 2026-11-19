# Sampling tests: HMC and replica exchange on analytic targets.
#
# The point of these tests is that the *correctness* of the sampler is measured,
# not assumed: the target distribution is known in closed form (or by quadrature),
# and the empirical distribution is compared to it with a statistic whose
# finite-sample null distribution is known.
#
# Explicit limitation, stated once and meant: a finite test is evidence, not proof
# of ergodicity, and nothing here says anything about protein folding.

using FoundationCore
using FoundationCore.Sampling
using Random, Statistics, Printf, LinearAlgebra

const SA = FoundationCore.Sampling

# ---------------------------------------------------------------------------
# 1. harmonic target: HMC must reproduce mean and variance
# ---------------------------------------------------------------------------
function test_hmc_reproduces_gaussian()
    gt = SA.GaussianTarget(1.0, 0.0)
    β = 1.0
    x0 = [1.0]
    m = [1.0]
    dt = 0.15
    L = 12
    nsteps = 60_000
    rng = MersenneTwister(11)
    res = SA.hmc_trajectory(x0, gt, x -> SA.grad(gt, x), m, dt, L, nsteps, β, rng)
    xs = vec(res.samples)
    μexp = 0.0
    σ2exp = 1.0 / (β * gt.k)
    τ = SA.integrated_autocorrelation(xs)
    n_eff = length(xs) / τ
    μ = mean(xs); v = var(xs)
    se_mu = sqrt(σ2exp / n_eff)
    se_v = sqrt(2 * σ2exp^2 / n_eff)
    @printf("  acceptance=%.3f  mean=%.4f±%.4f (0)  var=%.4f±%.4f (%.3f)  tau=%.1f\n",
            res.acceptance_rate, μ, se_mu, v, se_v, σ2exp, τ)
    @assert res.acceptance_rate > 0.5 "acceptance too low: dt/L poorly chosen"
    @assert abs(μ - μexp) < 5 * se_mu + 1e-3 "mean off"
    @assert abs(v - σ2exp) < 5 * se_v + 1e-3 "variance off"
    return true
end

# ---------------------------------------------------------------------------
# 2. the leapfrog proposal must be an exact involution
# ---------------------------------------------------------------------------
function test_proposal_is_involution()
    rng = MersenneTwister(22)
    gt = SA.GaussianTarget(1.7, 0.3)
    gradU = x -> SA.grad(gt, x)
    m = [1.0, 1.0, 1.0]
    worst = 0.0
    for _ in 1:200
        q = randn(rng, 3)
        p = randn(rng, 3)
        dt = 0.05 + 0.1 * rand(rng)
        L = rand(rng, 2:15)
        q1, p1 = SA.leapfrog(q, p, gradU, m, dt, L)
        # the returned momentum is already negated; to invert we apply the map
        # again to (q1, p1) and must recover (q, p)
        q2, p2 = SA.leapfrog(q1, p1, gradU, m, dt, L)
        worst = max(worst, maximum(abs, q2 - q), maximum(abs, p2 - p))
    end
    @printf("  max deviation from involution over 200 random proposals: %.3e\n", worst)
    @assert worst < 1e-9 "proposal is not an involution: Metropolis would be biased"
    return true
end

# ---------------------------------------------------------------------------
# 3. the proposal must be volume preserving
# ---------------------------------------------------------------------------
function test_proposal_volume_preserving()
    gt = SA.GaussianTarget(1.3, -0.2)
    gradU = x -> SA.grad(gt, x)
    m = [1.0, 1.0]
    dt = 0.07
    L = 9
    # numerical Jacobian of the (q, p) -> (q', p') proposal map
    q = [0.4, -0.9]; p = [0.3, 0.6]
    n = 4
    z0 = vcat(q, p)
    function prop(z)
        q1, p1 = SA.leapfrog(z[1:2], z[3:4], gradU, m, dt, L)
        return vcat(q1, p1)
    end
    h = 1e-6
    J = zeros(n, n)
    for i in 1:n
        zp = copy(z0); zp[i] += h
        zm = copy(z0); zm[i] -= h
        J[:, i] = (prop(zp) - prop(zm)) / (2h)
    end
    detJ = det(J)
    @printf("  |det J| = %.15f (exactly 1 expected for a symplectic map)\n", abs(detJ))
    @assert abs(abs(detJ) - 1.0) < 1e-6 "proposal does not preserve volume"
    return true
end

# ---------------------------------------------------------------------------
# 4. double well: distribution correctness where plain HMC does mix
# ---------------------------------------------------------------------------
function test_doublewell_distribution_moderate_temperature()
    dw = SA.DoubleWell(1.0, 1.0)   # minima at +-1, barrier height 1
    β = 2.0
    # Reference CDF by quadrature
    grid = collect(range(-4.0, 4.0, length = 2001))
    refcdf = let g = grid
        cdfg = SA.doublewell_cdf_exact(dw, β, g)
        x -> cdfg[searchsortedfirst(g, x) |> i -> clamp(i, 1, length(g))]
    end
    rng = MersenneTwister(33)
    res = SA.hmc_trajectory([0.5], dw, x -> SA.grad(dw, x), [1.0], 0.12, 15,
                            80_000, β, rng)
    xs = vec(res.samples)
    D = SA.ks_distance(xs, refcdf)
    crit = SA.ks_critical(length(xs))
    mom = SA.doublewell_moments_exact(dw, β)
    @printf("  acceptance=%.3f  KS D=%.4f  critical(5%%)=%.4f  empirical var=%.3f exact=%.3f\n",
            res.acceptance_rate, D, crit, var(xs), mom.var)
    @assert D < crit "KS distance exceeds the 5% critical value: distribution is wrong"
    return true
end

# ---------------------------------------------------------------------------
# 5. the actual enhanced-sampling claim: tempering beats a cold single chain
# ---------------------------------------------------------------------------
function test_replica_exchange_traverses_barrier()
    dw = SA.DoubleWell(1.0, 1.0)
    # Cold replicas: barrier/kT = beta*a*b^2 = 12. A short-trajectory HMC chain at
    # beta=12 essentially cannot cross, so its samples live in one well.
    betas = [12.0, 8.0, 5.0, 3.0, 1.5]
    rng = MersenneTwister(44)
    q0 = [1.0]           # start in the right well
    m = [1.0]
    dt = 0.10
    L = 10
    nsteps = 25_000
    swap_every = 5

    # plain cold chain
    rng_plain = MersenneTwister(45)
    plain = SA.hmc_trajectory(q0, dw, x -> SA.grad(dw, x), m, dt, L, nsteps,
                              betas[1], rng_plain)
    xs_plain = vec(plain.samples)
    cross_plain = SA.barrier_crossings(xs_plain)
    frac_left_plain = mean(xs_plain .< 0)

    pt = SA.replica_exchange(q0, dw, x -> SA.grad(dw, x), m, betas, dt, L,
                             nsteps, swap_every, rng)
    xs_cold = vec(pt.samples[1])
    cross_cold = SA.barrier_crossings(xs_cold)
    rt = SA.round_trip_rate(xs_cold)
    frac_left_cold = mean(xs_cold .< 0)

    # cold-replica target correctness against the exact beta_1 distribution
    grid = collect(range(-3.0, 3.0, length = 3001))
    refcdf = let g = grid, b = betas[1]
        cdfg = SA.doublewell_cdf_exact(dw, b, g)
        x -> cdfg[clamp(searchsortedfirst(g, x), 1, length(g))]
    end
    D_plain = SA.ks_distance(xs_plain, refcdf)
    D_cold = SA.ks_distance(xs_cold, refcdf)
    crit = SA.ks_critical(length(xs_cold))

    sw = SA.swap_acceptance(pt)
    @printf("  plain cold chain: crossings=%d  frac(left)=%.3f  KS D=%.4f\n",
            cross_plain, frac_left_plain, D_plain)
    @printf("  tempering cold:   crossings=%d  round_trips=%d  frac(left)=%.3f  KS D=%.4f (crit %.4f)\n",
            cross_cold, rt, frac_left_cold, D_cold, crit)
    @printf("  swap acceptance per adjacent pair: %s\n",
            string(round.(sw, digits = 3)))

    # The honest claim: tempering produces far more crossing traffic and a cold
    # replica that is much closer to the true two-well distribution.
    @assert cross_cold > 5 * cross_plain "tempering did not increase barrier traffic"
    @assert D_cold < D_plain "tempering did not improve the cold-replica distribution"
    # and the swap acceptance must be non-trivial for the ladder to be useful
    @assert all(x -> x > 0.1, sw) "a swap acceptance below 0.1 makes the ladder useless"
    return true
end

# ===========================================================================
# molecular (real factor energy) backend
# ===========================================================================
#
# The functions above sample analytic closures `U(q)`. Everything below goes
# through `Dynamics.HarmonicSystem` / `Dynamics.forces!`, i.e. the same
# `Factors.AbstractFactor` path used by Verlet and BAOAB. A small external
# harmonic factor is defined here so the target is exactly Gaussian and the
# molecular machinery can be checked against a known answer.

struct TestHarmonic <: FoundationCore.Factors.AbstractFactor
    idx::Int
    k::Float64
    x0::Float64
end
FoundationCore.Factors.factor_support_size(::TestHarmonic) = 1
FoundationCore.Factors.read_support(f::TestHarmonic) = (f.idx,)
function FoundationCore.Factors.evaluate!(F::AbstractMatrix{Float64}, f::TestHarmonic,
                                          q::AbstractMatrix{Float64}, ctx)
    dx = q[1, f.idx] - f.x0
    F[1, f.idx] -= f.k * dx          # F = -dU/dq
    return 0.5 * f.k * dx^2
end

function _harmonic_system(k, x0; N = 1, m = ones(N))
    factors = [TestHarmonic(j, k, x0) for j in 1:N]
    return FoundationCore.Dynamics.HarmonicSystem(
        factors, m, FoundationCore.Factors.Context(FoundationCore.Types.REDUCED, N))
end

function _lj_dimer_system(; ε = 1.0, σ = 1.0)
    fac = [FoundationCore.Factors.LennardJonesPair(1, 2, ε, σ; mode = :none)]
    return FoundationCore.Dynamics.HarmonicSystem(
        fac, [1.0, 1.0], FoundationCore.Factors.Context(FoundationCore.Types.REDUCED, 2))
end

# ---------------------------------------------------------------------------
# 7. molecular HMC on a 1-D harmonic: mean and variance vs analytic
# ---------------------------------------------------------------------------
function test_factor_energy_hmc_harmonic()
    k = 2.0
    x0 = 0.3
    sys = _harmonic_system(k, x0)
    T = 1.0
    rng = MersenneTwister(101)
    res = SA.hmc_trajectory(sys, reshape([x0, 0.0, 0.0], 3, 1), 40_000;
                            T = T, dt = 0.2, L = 10, nwarmup = 500, rng = rng)
    xs = [q[1, 1] for q in res.samples]
    est = SA.correlated_mean(xs)
    @assert res.acceptance_rate > 0.5 "molecular HMC acceptance too low: $(res.acceptance_rate)"
    @assert all(isfinite, res.H) "non-finite Hamiltonian in a molecular sample"
    @assert res.force_evals == (length(res.samples) + 500) * 11
    μexp = x0
    σ2exp = T / k
    se_v = sqrt(2 * σ2exp^2 / est.n_eff) + 1e-3
    # sample variance is corrected (divide by n-1), the estimate above is fine at this n
    @printf("  factor HMC: acceptance=%.3f  mean=%.4f±%.4f (%.3f)  var=%.4f±%.4f (%.3f)  tau=%.1f  n_eff=%.0f\n",
            res.acceptance_rate, est.mean, est.se, μexp, var(xs), se_v, σ2exp,
            est.tau, est.n_eff)
    @assert abs(est.mean - μexp) < 5 * est.se + 1e-3 "molecular HMC mean is biased"
    @assert abs(var(xs) - σ2exp) < 5 * se_v "molecular HMC variance is biased"

    # momentum draw: p ~ N(0, M k_B T), so var(p_j)/m_j = T per component and
    # <K> = (3N/2) T. This is checked directly rather than assumed.
    m = [2.0, 0.5, 1.5]
    sm = _harmonic_system(1.0, 0.0; N = 3, m = m)
    pm = zeros(3, 3)
    p2 = zeros(3, 3)
    nrep = 20_000
    rngp = MersenneTwister(106)
    Ksum = 0.0
    for _ in 1:nrep
        SA.maxwell_momenta!(pm, sm, T, rngp)
        Ksum += FoundationCore.Dynamics.kinetic_energy(sm, pm)
        @inbounds for j in 1:3, d in 1:3
            p2[d, j] += pm[d, j]^2
        end
    end
    for j in 1:3
        vj = p2[1, j] / nrep              # all components have the same law
        @assert abs(vj / m[j] - T) < 5 * sqrt(2 * T^2 / nrep) + 1e-3 "momentum variance not mass-weighted"
    end
    Kbar = Ksum / nrep
    seK = sqrt(4.5 * T^2 / nrep) * sqrt(2)   # loose, but only a sanity bound
    @printf("  momentum draw: <K>=%.4f (%.1f expected)  var(p_j)/m_j=%.4f (%.1f)\n",
            Kbar, 4.5, p2[1, 1] / nrep / m[1], T)
    @assert abs(Kbar - 4.5) < 4 * seK + 1e-3 "kinetic energy is not (3N/2)T"
    return true
end

# ---------------------------------------------------------------------------
# 6. honestly report cost per effective sample
# ---------------------------------------------------------------------------
function test_cost_per_effective_sample_report()
    dw = SA.DoubleWell(1.0, 1.0)
    # Compare at a temperature where BOTH can mix, so the comparison is about
    # efficiency rather than about one method being broken.
    β = 1.5
    m = [1.0]; dt = 0.10; L = 10
    nsteps = 20_000
    rng = MersenneTwister(55)
    t0 = time()
    plain = SA.hmc_trajectory([1.0], dw, x -> SA.grad(dw, x), m, dt, L, nsteps, β, rng)
    t_plain = time() - t0
    ess_plain = SA.effective_sample_size(vec(plain.samples))

    # replica exchange needs R times the force evaluations per sweep, so comparing
    # wall time per *cold* sample is the honest accounting
    betas = [β, 1.0, 0.7, 0.5]
    t1 = time()
    pt = SA.replica_exchange([1.0], dw, x -> SA.grad(dw, x), m, betas, dt, L,
                             nsteps, 5, MersenneTwister(56))
    t_pt = time() - t1
    ess_pt = SA.effective_sample_size(vec(pt.samples[1]))

    rate_plain = ess_plain / t_plain
    rate_pt = ess_pt / t_pt
    @printf("  plain HMC:  ESS=%.1f  wall=%.2fs  ESS/s=%.2f\n", ess_plain, t_plain, rate_plain)
    @printf("  tempering:  ESS=%.1f  wall=%.2fs  ESS/s=%.2f  (cold replica, %d replicas)\n",
            ess_pt, t_pt, rate_pt, length(betas))
    if rate_pt < rate_plain
        @printf("  NOTE: tempering is SLOWER per effective sample here (%.2fx). That is\n", rate_plain / rate_pt)
        println("        expected at a temperature where a single chain already mixes; the")
        println("        benefit is only realized on the cold end where a single chain traps.")
    else
        @printf("  tempering faster per effective sample by %.2fx\n", rate_pt / rate_plain)
    end
    # No assertion that tempering always wins: that would be false. The test
    # asserts the *measurement machinery* works and reports the truth.
    @assert ess_plain > 1 && ess_pt > 1 "ESS estimate degenerate"
    return true
end

# ---------------------------------------------------------------------------
# 8. proposal involution and volume preservation on the molecular path
# ---------------------------------------------------------------------------
function test_molecular_proposal_involution_volume()
    sys = _lj_dimer_system()
    rng = MersenneTwister(102)
    dt = 0.01
    L = 12
    q0 = [0.1 1.25; -0.05 0.1; 0.02 -0.08]
    p0 = [0.3 -0.2; 0.1 0.25; -0.15 0.05]
    q1, p1 = SA.hmc_proposal(q0, p0, sys, dt, L)
    q2, p2 = SA.hmc_proposal(q1, p1, sys, dt, L)
    dev = max(maximum(abs, q2 - q0), maximum(abs, p2 - p0))
    # numerical Jacobian of the (q,p) -> (q',p') map on the molecular path
    z0 = vcat(vec(q0), vec(p0))
    prop = z -> begin
        qq = reshape(z[1:6], 3, 2)
        pp = reshape(z[7:12], 3, 2)
        qn, pn = SA.hmc_proposal(qq, pp, sys, dt, L)
        return vcat(vec(qn), vec(pn))
    end
    h = 1e-6
    J = zeros(12, 12)
    for i in 1:12
        zp = copy(z0); zp[i] += h
        zm = copy(z0); zm[i] -= h
        J[:, i] = (prop(zp) - prop(zm)) / (2h)
    end
    detJ = det(J)
    @printf("  molecular proposal: involution max deviation=%.3e  |det J|=%.12f\n",
            dev, abs(detJ))
    @assert dev < 1e-9 "molecular proposal is not an involution"
    @assert abs(abs(detJ) - 1.0) < 1e-6 "molecular proposal does not preserve volume"
    return true
end

# ---------------------------------------------------------------------------
# 9. observed acceptance == mean(min(1, exp(-dH))) within binomial error
# ---------------------------------------------------------------------------
function test_acceptance_matches_formula()
    sys = _harmonic_system(2.0, 0.0)
    q0 = reshape([0.0, 0.0, 0.0], 3, 1)
    res = SA.hmc_trajectory(sys, q0, 20_000; T = 1.0, dt = 1.0, L = 5,
                            nwarmup = 200, rng = MersenneTwister(103))
    p = mean(min.(1.0, exp.(-res.beta .* res.dH)))
    obs = mean(res.accepted)
    se = sqrt(max(p * (1 - p), 1e-12) / length(res.dH))
    @printf("  acceptance: observed=%.4f  mean[min(1,exp(-βdH))]=%.4f  binom se=%.4f\n",
            obs, p, se)
    # a test with all-accept (or all-reject) would not exercise the ratio at all
    @assert 0.3 < obs < 0.98 "test configuration is degenerate: acceptance=$(obs)"
    @assert abs(obs - p) < 4 * se + 1e-3 "acceptance does not equal the Metropolis probability"
    return true
end

# ---------------------------------------------------------------------------
# 10. swap acceptance against the exact 2-state formula
# ---------------------------------------------------------------------------
function test_swap_two_state_exact()
    U0, U1 = 0.0, 1.7
    βcold, βhot = 2.0, 0.5
    # exact two-state equilibrium weights
    z(b) = 1.0 + exp(-b * U1)
    pc0, pc1 = 1.0 / z(βcold), exp(-βcold * U1) / z(βcold)
    ph0, ph1 = 1.0 / z(βhot), exp(-βhot * U1) / z(βhot)
    exact = 0.0
    for (pc, ph, Ui, Uj) in ((pc0, ph0, U0, U0), (pc0, ph1, U0, U1),
                             (pc1, ph0, U1, U0), (pc1, ph1, U1, U1))
        exact += pc * ph * SA.swap_accept_probability(βcold, βhot, Ui, Uj)
    end
    rng = MersenneTwister(104)
    n = 200_000
    accepts = 0
    for _ in 1:n
        ci = rand(rng) < pc1
        hj = rand(rng) < ph1
        Ui = ci ? U1 : U0
        Uj = hj ? U1 : U0
        accepts += rand(rng) < SA.swap_accept_probability(βcold, βhot, Ui, Uj)
    end
    obs = accepts / n
    se = sqrt(exact * (1 - exact) / n)
    @printf("  2-state swap: exact=%.6f  observed=%.6f  binom se=%.6f\n", exact, obs, se)
    @assert abs(obs - exact) < 4 * se + 1e-4 "swap acceptance deviates from the formula"
    # degenerate sanity checks: identical energies always swap, and the
    # direction of the inequality is the one derived from the joint target
    @assert SA.swap_accept_probability(2.0, 0.5, 1.0, 1.0) == 1.0
    @assert SA.swap_accept_probability(2.0, 0.5, 0.0, 2.0) ≈ exp((2.0 - 0.5) * (0.0 - 2.0))
    return true
end

# ---------------------------------------------------------------------------
# 11. the sampler's own swap loop matches the formula on recorded energies
# ---------------------------------------------------------------------------
function test_molecular_pt_swap_statistics()
    sys = _harmonic_system(2.0, 0.0)
    temperatures = [0.4, 0.8, 1.4]
    q0 = reshape([0.0, 0.0, 0.0], 3, 1)
    res = SA.replica_exchange(sys, q0, 4000; temperatures = temperatures,
                              dt = 0.2, L = 10, swap_every = 2,
                              nwarmup = 200, rng = MersenneTwister(105))
    sw = SA.swap_acceptance(res)
    for r in 1:2
        pred = mean(SA.swap_accept_probability.(Ref(res.beta[r]), Ref(res.beta[r + 1]),
                                                res.swap_u_low[1:res.swap_attempts[r], r],
                                                res.swap_u_high[1:res.swap_attempts[r], r]))
        se = sqrt(max(pred * (1 - pred), 1e-12) / res.swap_attempts[r])
        @printf("  pair %d: swap observed=%.4f predicted=%.4f se=%.4f (n=%d)\n",
                r, sw[r], pred, se, res.swap_attempts[r])
        @assert abs(sw[r] - pred) < 4 * se + 1e-2 "sampler swap acceptance deviates from the formula"
        @assert sw[r] > 0.1 "swap acceptance below 0.1 makes the ladder useless"
    end
    @assert size(res.replica_index) == (4000, 3)
    return true
end

# ---------------------------------------------------------------------------
# 12. constraints must be refused with a typed error, never sampled silently
# ---------------------------------------------------------------------------
function test_constraints_refused()
    sys = _harmonic_system(2.0, 0.0)
    q0 = reshape([0.0, 0.0, 0.0], 3, 1)
    constraints = [(1, 2, 1.0)]
    @assert SA.sampling_constraints_supported == false
    threw = 0
    for f in (
        () -> SA.hmc_trajectory(sys, q0, 10; dt = 0.1, L = 5,
                                rng = MersenneTwister(1), constraints = constraints),
        () -> SA.hmc_step!(reshape([0.0, 0.0, 0.0], 3, 1),
                           reshape([0.0, 0.0, 0.0], 3, 1), sys, 0.1, 5;
                           rng = MersenneTwister(1), constraints = constraints),
        () -> SA.replica_exchange(sys, q0, 10; temperatures = [0.5, 1.0],
                                  dt = 0.1, L = 5, rng = MersenneTwister(1),
                                  constraints = constraints),
    )
        try
            f()
        catch e
            @assert e isa SA.SamplingConstraintError "wrong error type: $(typeof(e))"
            threw += 1
        end
    end
    @printf("  constrained entry points refused: %d/3 with SamplingConstraintError\n", threw)
    @assert threw == 3 "a constrained entry point did not refuse"
    return true
end

# ---------------------------------------------------------------------------
# 13. blocking & Flyvbjerg-Petersen autocorrelation-aware summaries
# ---------------------------------------------------------------------------
function test_blocking_and_observable_summary()
    rng = MersenneTwister(2026)
    rho = 0.8
    tau_exact = (1.0 + rho) / (1.0 - rho)   # 9.0
    n = 40_000
    x = Vector{Float64}(undef, n)
    x[1] = randn(rng)
    s_innov = sqrt(1.0 - rho^2)
    for t in 2:n
        x[t] = rho * x[t - 1] + s_innov * randn(rng)
    end
    se_exact = sqrt(tau_exact / n)

    est = SA.correlated_mean(x)
    blk = SA.blocked_mean(x; n_blocks = 40)
    fp = SA.flyvbjerg_petersen_se(x; min_blocks = 16)
    summ = SA.summarize_observable(x; n_blocks = 40, min_blocks = 16)

    @printf("  AR(1) rho=0.8: tau=%.2f (exact %.1f)  se_ess=%.4f  se_blk=%.4f  se_fp=%.4f  (exact %.4f)\n",
            est.tau, tau_exact, summ.se_ess, summ.se_block, summ.se_fp, se_exact)
    @assert abs(est.tau - tau_exact) < 1.5
    @assert 0.7 * se_exact < summ.se_ess < 1.4 * se_exact
    @assert 0.7 * se_exact < summ.se_block < 1.4 * se_exact
    @assert 0.7 * se_exact < fp.se < 1.5 * se_exact
    @assert summ.se == max(summ.se_ess, summ.se_block, summ.se_fp)
    @assert abs(summ.mean) < 4 * summ.se
    @assert isapprox(blk.se, summ.se_block; atol = 1e-14)
    @assert SA.block_variance_se(x; n_blocks = 40) > 0

    # Input validation
    threw = 0
    for f in (
        () -> SA.block_means([1.0, 2.0, 3.0], 5),
        () -> SA.block_standard_error([1.0, NaN, 2.0, 3.0]; n_blocks = 2),
        () -> SA.flyvbjerg_petersen_se([1.0, 2.0]; min_blocks = 8),
    )
        try
            f()
        catch e
            @assert e isa ArgumentError
            threw += 1
        end
    end
    @assert threw == 3
    return true
end

function _run_all()
    for (name, fn) in [
        ("hmc_reproduces_gaussian", test_hmc_reproduces_gaussian),
        ("proposal_involution", test_proposal_is_involution),
        ("proposal_volume_preserving", test_proposal_volume_preserving),
        ("doublewell_distribution", test_doublewell_distribution_moderate_temperature),
        ("replica_exchange_barrier", test_replica_exchange_traverses_barrier),
        ("cost_per_ess_report", test_cost_per_effective_sample_report),
        ("factor_energy_hmc_harmonic", test_factor_energy_hmc_harmonic),
        ("molecular_proposal_involution_volume", test_molecular_proposal_involution_volume),
        ("acceptance_matches_formula", test_acceptance_matches_formula),
        ("swap_two_state_exact", test_swap_two_state_exact),
        ("molecular_pt_swap_statistics", test_molecular_pt_swap_statistics),
        ("constraints_refused", test_constraints_refused),
        ("blocking_and_observable_summary", test_blocking_and_observable_summary),
    ]
        println("TEST ", name)
        fn()
    end
    println("")
    println("ALL TESTS PASSED")
end

_run_all()
