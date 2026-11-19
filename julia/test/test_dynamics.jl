# Dynamics and numerical-accuracy-controller tests.
#
# Every test here measures a number. Where a claim is about behaviour over time
# (energy bounded vs drifting) the test reports the OLS drift slope and its
# standard error, so the assertion is on the *statistic* rather than on a
# hand-picked energy threshold.
#
# Analytic reference for the oscillator: for U = (k/2)(x1-x2)^2 with equal masses
# m, the relative coordinate obeys a harmonic oscillator of angular frequency
# omega = sqrt(2k/m). The exact energy is E0 and the exact trajectory is a cosine.
# Velocity Verlet is known to conserve a *shadow* Hamiltonian, so its energy error
# oscillates with a bounded amplitude ~ O(dt^2) and its drift slope is zero.

using FoundationCore
using FoundationCore.Types
using FoundationCore.Factors
using FoundationCore.Dynamics
using FoundationCore.Sampling: integrated_autocorrelation
using Random, Statistics, Printf, LinearAlgebra

const T = FoundationCore.Types
const FA = FoundationCore.Factors
const DY = FoundationCore.Dynamics

"Two particles, U = (k/2)|q1-q2|^2, centre of mass at rest."
function oscillator(k = 1.0, m = 1.0, x0 = 1.0)
    q = zeros(3, 2)
    q[1, 1] = -x0 / 2
    q[1, 2] = x0 / 2
    p = zeros(3, 2)
    fac = FA.HarmonicBond(1, 2, k, 0.0)
    ctx = FA.Context(T.REDUCED, 2)
    sys = DY.HarmonicSystem([fac], [m, m], ctx)
    return DY.State(q, p), sys
end

"Two atoms at r = 1.12 with a deliberately tight LJ cutoff rc = 1.2, no box."
function _lj_dimer(mode)
    q = zeros(3, 2)
    q[1, 2] = 1.12
    p = zeros(3, 2)
    fac = FA.LennardJonesPair(1, 2, 1.0, 1.0; mode = mode, rc = 1.2)
    return DY.State(q, p),
           DY.HarmonicSystem([fac], [1.0, 1.0], FA.Context(T.REDUCED, 2))
end

"Run `f`, returning the caught exception or `nothing`."
function _caught(f)
    try
        f()
    catch e
        return e
    end
    return nothing
end

function test_energy_is_bounded_not_drifting()
    st, sys = oscillator(1.0, 1.0, 1.0)
    dt = 0.01
    n = 200_000
    recs, es = DY.run_trajectory(st, sys, dt, n; record_every = 20)
    rep = DY.energy_drift_report(recs, es)
    @printf("  n=%d steps dt=%.3f  E0=%.12f  max|ΔE|=%.3e  slope=%.3e ± %.3e\n",
            n, dt, rep.E0, rep.max_abs_dev, rep.slope, rep.slope_stderr)
    @printf("  predicted_drift/band = %.4f (threshold %.1f)\n",
            rep.drift_fraction, DY.DRIFT_FRACTION_MAX)
    # Symplectic integrators do not drift secularly. The statistic that tests
    # this is the drift slope in units of its own standard error.
    @assert rep.bounded "predicted drift is $(round(rep.drift_fraction, digits=3)) of the band: NOT bounded"
    # the oscillation amplitude must be O(dt^2), not growing with n
    @assert rep.max_abs_dev < 1e-3 "oscillation amplitude $(rep.max_abs_dev) implausibly large"
    # A real leak would make the *slope* grow in the second half. Comparing the
    # maximum excursion is phase-dependent noise, so test the statistic that
    # actually encodes drift: both halves must be consistent with zero slope.
    half = length(recs) ÷ 2
    r1 = DY.energy_drift_report(recs[1:half], DY.EnergySeries(es.naive[1:half], es.compensated[1:half]))
    r2 = DY.energy_drift_report(recs[half+1:end], DY.EnergySeries(es.naive[half+1:end], es.compensated[half+1:end]))
    @printf("  first half frac=%.4f   second half frac=%.4f\n", r1.drift_fraction, r2.drift_fraction)
    @assert r1.bounded && r2.bounded "drift appears in one half only: real leak"
    # A single endpoint value is one phase sample inside the oscillation band, so
    # |E_final - E_0| is not a drift measure. The drift measure is the slope
    # extrapolated over the run length, compared with the band.
    T_run = rep.n > 1 ? (recs[end].t - recs[1].t) : 0.0
    predicted_drift = abs(rep.slope) * T_run
    @printf("  |E_final - E_0| = %.3e (one phase sample)   slope*T = %.3e   band = %.3e\n",
            abs(rep.Efinal - rep.E0), predicted_drift, rep.max_abs_dev)
    @assert predicted_drift < rep.max_abs_dev "slope extrapolates to a drift larger than the band"
    return true
end

function test_time_reversibility()
    # Integrate forward, negate momenta, integrate the same number of steps: a
    # time-reversible integrator returns to the start. This tests reversibility
    # directly and is independent of any energy argument.
    st, sys = oscillator(1.3, 0.7, 1.0)
    st.p[1, 1] = 0.21; st.p[1, 2] = -0.21   # give it some motion
    q0 = copy(st.q); p0 = copy(st.p)
    dt = 0.005
    for _ in 1:2000
        DY.velocity_verlet_step!(st, sys, dt)
    end
    st.p .= -st.p
    for _ in 1:2000
        DY.velocity_verlet_step!(st, sys, dt)
    end
    st.p .= -st.p
    dq = maximum(abs, st.q .- q0) / max(maximum(abs, q0), 1e-30)
    dp = maximum(abs, st.p .- p0) / max(maximum(abs, p0), 1e-30)
    @printf("  after forward+reverse: rel|Δq|=%.3e  rel|Δp|=%.3e\n", dq, dp)
    @assert dq < 1e-9 && dp < 1e-9 "time reversibility violated"
    return true
end

function test_second_order_convergence()
    # Global position error at a fixed final time must fall as dt^2 for a
    # second-order integrator. Fit the exponent rather than asserting a constant,
    # so the test reports what the code actually does.
    dts = [0.02, 0.01, 0.005, 0.0025]
    errs = Float64[]
    Tend = 2.0
    omega = sqrt(2.0 * 1.0 / 1.0)
    for dt in dts
        st, sys = oscillator(1.0, 1.0, 1.0)
        n = round(Int, Tend / dt)
        for _ in 1:n
            DY.velocity_verlet_step!(st, sys, dt)
        end
        # exact: x_rel(t) = cos(omega t) with x_rel(0) = 1, zero velocity
        xexact = cos(omega * n * dt)
        xgot = st.q[1, 2] - st.q[1, 1]
        push!(errs, abs(xgot - xexact))
    end
    # fit log(err) = log(C) + p log(dt)
    xs = log.(dts); ys = log.(errs)
    p = (length(xs) * sum(xs .* ys) - sum(xs) * sum(ys)) /
        (length(xs) * sum(abs2, xs) - sum(xs)^2)
    @printf("  global error at T=%.1f for dt=%s\n", Tend, string(dts))
    @printf("  errors = %s\n", string(round.(errs, sigdigits = 4)))
    @printf("  fitted convergence order p = %.3f (expect ~2)\n", p)
    @assert 1.8 < p < 2.2 "convergence order $p is not second order"
    return true
end

function test_step_doubling_defect_scales_as_dt3()
    # The local step-doubling defect must scale as dt^3 for a second-order
    # integrator. Reporting the fitted exponent makes the "heuristic estimate"
    # claim checkable.
    dts = [0.04, 0.02, 0.01, 0.005]
    defects = Float64[]
    for dt in dts
        st, sys = oscillator(1.0, 1.0, 1.0)
        st.p[1, 1] = 0.37; st.p[1, 2] = -0.37
        push!(defects, DY.step_doubling_defect(st, sys, dt))
    end
    xs = log.(dts); ys = log.(defects)
    p = (length(xs) * sum(xs .* ys) - sum(xs) * sum(ys)) /
        (length(xs) * sum(abs2, xs) - sum(xs)^2)
    @printf("  defects = %s\n", string(round.(defects, sigdigits = 4)))
    @printf("  fitted local-defect order = %.3f (expect ~3)\n", p)
    @assert 2.7 < p < 3.3 "defect order $p is not dt^3"
    return true
end

function test_compensated_summation_matters()
    # Construct a series where naive summation demonstrably loses precision: a
    # large constant plus many small terms whose total is comparable to the small
    # terms. Compare against the exact sum computed in BigFloat.
    n = 1_000_000
    xs = vcat([1.0e16], fill(1.0, n), [-1.0e16])
    exact = sum(BigFloat.(xs))
    naive = sum(xs)
    kahan = DY.kahan_sum(xs)
    neum = DY.neumaier_sum(xs)
    e_naive = abs(naive - Float64(exact))
    e_kahan = abs(kahan - Float64(exact))
    e_neum = abs(neum - Float64(exact))
    @printf("  naive error=%.3e  kahan error=%.3e  neumaier error=%.3e  (exact=%.1f)\n",
            e_naive, e_kahan, e_neum, Float64(exact))
    @assert e_neum < 1e-3 "Neumaier summation should be near-exact here"
    @assert e_naive > 1.0 "this test needs a series where naive summation really fails"
    return true
end

function test_energy_drift_report_detects_a_real_leak()
    # A diagnostic that cannot detect a leak is useless. Feed it a synthetic
    # linear leak and require it to flag it; then feed a bounded oscillation and
    # require it NOT to flag it. This validates the detector itself.
    t = collect(0.0:0.01:100.0)
    E_osc = 1.0 .+ 1e-6 .* sin.(2pi .* t)
    E_leak = 1.0 .+ 1e-8 .* t
    r_osc = DY.energy_drift_report(t, E_osc)
    r_leak = DY.energy_drift_report(t, E_leak)
    @printf("  bounded oscillation: frac=%.4f (bounded=%s)\n", r_osc.drift_fraction, r_osc.bounded)
    @printf("  true linear leak:    frac=%.4f (bounded=%s)\n", r_leak.drift_fraction, r_leak.bounded)
    @assert r_osc.bounded "false positive: bounded oscillation flagged as drift"
    @assert !r_leak.bounded "false negative: real leak not detected"
    return true
end

function test_select_dt_picks_a_valid_fixed_step()
    st, sys = oscillator(1.0, 1.0, 1.0)
    sel = DY.select_dt(st, sys, [0.05, 0.02, 0.01, 0.005, 0.0025, 0.001];
                       nsteps = 3000, max_defect = 1e-5)
    println("    dt        defect      drift_frac  bounded")
    for row in sel.table
        @printf("    %.4f   %.3e   %10.4f   %s\n", row.dt, row.defect, row.drift_fraction, row.bounded)
    end
    @assert sel.accepted "no dt passed; the ladder or tolerances need revisiting"
    @printf("  selected dt = %.4f (largest passing both criteria)\n", sel.dt)
    # the selection must be a *constant*, and re-running it must be deterministic
    sel2 = DY.select_dt(st, sys, [0.05, 0.02, 0.01, 0.005, 0.0025, 0.001];
                        nsteps = 3000, max_defect = 1e-5)
    @assert sel.dt == sel2.dt "step selection is not deterministic"
    return true
end

function test_rattle_distance_constraints()
    q = zeros(3, 2); q[1, 2] = 1.0
    p = zeros(3, 2); p[2, 1] = -0.25; p[2, 2] = 0.25
    st = DY.State(q, p)
    fac = FA.HarmonicBond(1, 2, 2.0, 1.0)
    sys = DY.HarmonicSystem([fac], [2.0, 3.0], FA.Context(T.REDUCED, 2))
    constraints = [(1, 2, 1.0)]
    @assert DY.constraints_supported()
    qfd = [0.0 1.3; 0.1 0.2; 0.0 0.0]
    Ffd = zeros(3, 2)
    DY.forces!(sys, qfd, Ffd)
    hfd = 1e-6
    qplus = copy(qfd); qminus = copy(qfd)
    qplus[2, 2] += hfd; qminus[2, 2] -= hfd
    dU = (DY.potential_energy(sys, qplus) - DY.potential_energy(sys, qminus)) / (2hfd)
    @assert abs(Ffd[2, 2] + dU) < 1e-8 "force is inconsistent with the energy gradient"
    DY.rattle_project!(st, sys, constraints)
    constraint_error(x) = abs(sum(abs2, x.q[:, 2] .- x.q[:, 1]) - 1.0)
    tangent_error(x) = abs(dot(x.q[:, 2] .- x.q[:, 1],
        x.p[:, 2] ./ sys.m[2] .- x.p[:, 1] ./ sys.m[1]))
    @assert constraint_error(st) < 1e-12 && tangent_error(st) < 1e-12
    tangent = st.p ./ reshape(sys.m, 1, :)
    eps_tangent = 1e-7
    q_tangent = st.q .+ eps_tangent .* tangent
    @assert abs(constraint_error(DY.State(q_tangent, st.p))) / eps_tangent < 1e-6

    # The correction uses inverse masses, not an unweighted Cartesian average.
    displaced = DY.State([0.0 1.2; 0 0; 0 0], zeros(3, 2))
    DY.rattle_project!(displaced, sys, constraints)
    @assert constraint_error(displaced) < 1e-12
    @assert abs(displaced.q[1, 1]) > abs(displaced.q[1, 2] - 1.2)

    dt = 0.002
    energies = Float64[DY.total_energy(sys, st.q, st.p)]
    for step in 1:10_000
        DY.velocity_verlet_step!(st, sys, dt; constraints)
        if step % 100 == 0
            @assert constraint_error(st) < 1e-12
            @assert tangent_error(st) < 1e-11
            push!(energies, DY.total_energy(sys, st.q, st.p))
        end
    end
    energy_excursion = maximum(abs.(energies .- energies[1]))
    @printf("  constrained max|ΔE|=%.3e\n", energy_excursion)
    @assert energy_excursion < 1e-4

    # Reversibility includes the tangent-momentum projection at both endpoints.
    reverse = DY.copy(st)
    reverse.p .*= -1
    for _ in 1:10_000
        DY.velocity_verlet_step!(reverse, sys, dt; constraints)
    end
    reverse.p .*= -1
    @assert maximum(abs, reverse.q .- q) < 1e-8
    @assert maximum(abs, reverse.p .- p) < 1e-8

    err = _caught(() -> DY.velocity_verlet_step!(DY.State([0.0 1.2; 0 0; 0 0], zeros(3, 2)),
        sys, dt; constraints, constraint_max_iterations = 1, constraint_tolerance = 1e-15))
    @assert err isa DY.ConstraintConvergenceError
    @assert err.stage == :position
    coupled_error = _caught(() -> DY.rattle_project!(DY.State([0.0 1.0 2.0; 0 0 0; 0 0 0], zeros(3, 3)),
        DY.HarmonicSystem(FA.AbstractFactor[], [1.0, 1.0, 1.0], FA.Context(T.REDUCED, 3)),
        [(1, 2, 1.0), (2, 3, 1.0)]))
    @assert coupled_error isa DY.UnsupportedConstraintTopology
    println("  pair RATTLE: 10,000-step residuals, reversal, energy bound, and typed rejection passed")
    return true
end

function test_baoab_samples_canonical_gaussian()
    # A 1-D harmonic oscillator sampled with Langevin BAOAB must reproduce the
    # canonical distribution N(0, 1/(beta k)). We compare the first two moments
    # against their *statistical* errors, which is the only honest statement for a
    # finite run.
    k = 1.0
    m = 1.0
    beta = 2.0
    gamma = 1.0
    dt = 0.01
    rng = MersenneTwister(9)
    q = zeros(3, 2)
    q[1, 1] = -1.0; q[1, 2] = 1.0
    p = zeros(3, 2)
    fac = FA.HarmonicBond(1, 2, k, 0.0)
    ctx = FA.Context(T.REDUCED, 2)
    sys = DY.HarmonicSystem([fac], [m, m], ctx)
    F = zeros(3, 2)
    DY.forces!(sys, q, F)
    xs = Float64[]
    n = 400_000
    burn = 20_000
    for step in 1:n
        DY.langevin_baoab_step!(q, p, F, sys, dt, gamma, beta, rng)
        if step > burn && step % 10 == 0
            push!(xs, q[1, 2] - q[1, 1])
        end
    end
    # theoretical: the relative coordinate has effective stiffness 2k and reduced
    # mass m/2, so variance = 1/(beta * 2k) ... computed directly instead:
    # U = (k/2)x^2 => variance = 1/(beta*k).
    μexp = 0.0
    σ2exp = 1.0 / (beta * k)
    nS = length(xs)
    μ = mean(xs); v = var(xs)
    # Successive samples are correlated (dt is small relative to the relaxation
    # time), so the naive standard error sigma/sqrt(n) would be far too optimistic.
    # Use the integrated autocorrelation time to inflate it: this is the standard
    # correction and it is reported so the reader can see the effective n.
    τx = integrated_autocorrelation(xs)
    n_eff = nS / τx
    se_mu = sqrt(σ2exp / n_eff)
    se_v = sqrt(2 * σ2exp^2 / n_eff)
    @printf("  BAOAB n=%d (tau=%.1f, n_eff=%.0f)  mean=%.4f ± %.4f (exact %.1f)  var=%.4f ± %.4f (exact %.3f)\n",
            nS, τx, n_eff, μ, se_mu, μexp, v, se_v, σ2exp)
    @assert abs(μ - μexp) < 5 * se_mu + 1e-3 "mean off by more than 5 autocorrelation-corrected standard errors"
    @assert abs(v - σ2exp) < 5 * se_v + 1e-3 "variance off by more than 5 autocorrelation-corrected standard errors"
    return true
end

function test_force_model_gate_refuses_inconsistent_cutoffs()
    # A factor whose force is not the derivative of its energy may still be
    # constructed and evaluated (energy comparison, NIST tables); what it may not
    # do is move (q, p). Truncated and shifted LJ are the canonical traps: their
    # force is correct inside rc, so a test started there would look healthy until
    # a step crossed the cutoff. The refusal is therefore structural, not a
    # sampled-configuration check.
    for mode in (:trunc, :shift)
        st, sys = _lj_dimer(mode)
        @assert !DY.force_model_supported(sys)
        U = DY.potential_energy(sys, st.q)
        @assert isfinite(U) "energy-only evaluation must remain available (mode=$mode)"
        q0 = copy(st.q); p0 = copy(st.p)
        err = _caught(() -> DY.velocity_verlet_step!(st, sys, 1e-3))
        @assert err isa ErrorException "mode=$mode was not refused by velocity Verlet"
        msg = sprint(showerror, err)
        @assert occursin("non-force-consistent", msg) &&
                occursin("mode = :" * string(mode), msg) "refusal is not actionable: $msg"
        @assert st.q == q0 && st.p == p0 "state moved before the refusal"
        for (label, f) in [
            ("buffer Verlet", () -> DY.velocity_verlet_step!(st.q, st.p, st.F, sys, 1e-3)),
            ("run_trajectory", () -> DY.run_trajectory(st, sys, 1e-3, 10)),
            ("step_doubling", () -> DY.step_doubling_defect(st, sys, 1e-3)),
            ("select_dt", () -> DY.select_dt(st, sys, [1e-3]; nsteps = 10)),
            ("BAOAB", () -> DY.langevin_baoab_step!(st.q, st.p, st.F, sys, 1e-3,
                                                    1.0, 1.0, MersenneTwister(1))),
        ]
            e = _caught(f)
            @assert e isa ErrorException "mode=$mode was not refused by $label"
            @assert occursin("non-force-consistent", sprint(showerror, e)) "wrong refusal from $label"
        end
        @printf("  mode=:%-5s refused by every evolving entry point; U=%.6f still evaluable\n",
                mode, U)
    end

    # :lfs subtracts U(rc) and U'(rc)(r-rc), so it is C^1; force_consistent is
    # true and the same dimer must integrate with a bounded energy excursion.
    st, sys = _lj_dimer(:lfs)
    @assert DY.force_model_supported(sys)
    recs, es = DY.run_trajectory(st, sys, 1e-4, 500; record_every = 10)
    rep = DY.energy_drift_report(recs, es)
    @assert all(isfinite, es.compensated) && all(isfinite, es.naive)
    @assert rep.max_abs_dev < 1e-8 "lfs energy excursion $(rep.max_abs_dev) is too large"
    # the other integrators accept the same compatible model
    @assert _caught(() -> DY.step_doubling_defect(copy(st), sys, 1e-4)) === nothing
    @assert _caught(() -> DY.langevin_baoab_step!(st.q, st.p, st.F, sys, 1e-4,
                                                  1.0, 1.0, MersenneTwister(2))) === nothing
    @printf("  mode=:lfs   accepted: E0=%.9f  max|ΔE|=%.3e over %d steps\n",
            rep.E0, rep.max_abs_dev, length(recs))
    # the gate is a per-call scan, so a bad factor appended after construction is
    # still refused (a cached "validated" flag would be unsound here)
    push!(sys.factors, FA.LennardJonesPair(1, 2, 1.0, 1.0; mode = :trunc, rc = 1.2))
    @assert !DY.force_model_supported(sys)
    @assert _caught(() -> DY.velocity_verlet_step!(st, sys, 1e-4)) isa ErrorException
    println("  post-construction :trunc factor is refused as well")
    return true
end

function test_integrator_state_machine_and_classification()
    vv = DY.integrator_spec(:velocity_verlet)
    rv = DY.integrator_spec(:rattle_verlet)
    bo = DY.integrator_spec(:baoab)
    cb = DY.integrator_spec(:constrained_baoab)

    @assert DY.is_reversible(vv) && !DY.is_stochastic(vv)
    @assert DY.is_reversible(rv) && !DY.is_stochastic(rv)
    @assert !DY.is_reversible(bo) && DY.is_stochastic(bo)
    @assert !DY.is_reversible(cb) && DY.is_stochastic(cb)
    @assert all(s -> DY.is_reversible(s) || s.classification === :diagnostic, vv.stages)
    @assert count(DY.is_stochastic, bo.stages) == 1
    @assert any(s -> s.name === :position_constraints, rv.stages)
    @assert any(s -> s.name === :velocity_constraints, cb.stages)

    # Verify live execution traces match the declared state machine stages.
    st, sys = oscillator(1.0, 1.0, 1.0)
    DY.forces!(sys, st.q, st.F); st.force_valid = true
    vv_trace = Symbol[]
    DY.velocity_verlet_step!(st, sys, 0.01; stage_hook = s -> push!(vv_trace, s))
    @assert vv_trace == [:half_kick, :position_drift, :force_evaluation, :half_kick]
    @assert all(s -> DY.stage_classification(s) === :reversible, vv_trace)

    st_r = DY.State([0.0 1.0; 0.0 0.0; 0.0 0.0], [0.0 0.0; -0.25 0.25; 0.0 0.0])
    sys_r = DY.HarmonicSystem([FA.HarmonicBond(1, 2, 2.0, 1.0)], [2.0, 3.0], FA.Context(T.REDUCED, 2))
    DY.rattle_project!(st_r, sys_r, [(1, 2, 1.0)])
    rv_trace = Symbol[]
    DY.velocity_verlet_step!(st_r, sys_r, 0.002; constraints = [(1, 2, 1.0)],
                             stage_hook = s -> push!(rv_trace, s))
    @assert rv_trace == [:force_evaluation, :half_kick, :position_drift,
                         :position_constraints, :force_evaluation, :half_kick, :velocity_constraints]

    bo_trace = Symbol[]
    DY.langevin_baoab_step!(st.q, st.p, st.F, sys, 0.01, 1.0, 1.0, MersenneTwister(1);
                            stage_hook = s -> push!(bo_trace, s))
    @assert bo_trace == [:half_kick, :position_drift, :thermostat_ou,
                         :position_drift, :force_evaluation, :half_kick]
    @assert count(s -> DY.stage_classification(s) === :stochastic, bo_trace) == 1
    println("  state machine & classification: live stage traces match declared specs")
    return true
end

function test_rattle_convergence_momentum_and_status()
    # Two constrained particles under central-force harmonic bond + asymmetric masses,
    # also coupled to an external central potential so angular momentum is nontrivial or conserved.
    # For a central pair potential, BOTH total linear momentum and total angular momentum are conserved!
    q0 = [0.0 1.0; 0.0 0.0; 0.0 0.0]
    p0 = [0.1 -0.1; -0.4 0.4; 0.2 -0.2]
    fac = FA.HarmonicBond(1, 2, 3.0, 1.2)
    sys = DY.HarmonicSystem([fac], [2.0, 3.0], FA.Context(T.REDUCED, 2))
    cs = [(1, 2, 1.0)]
    st0 = DY.State(q0, p0)
    res_status = DY.rattle_project_with_status!(st0, sys, cs)
    @assert res_status.converged && res_status.position_residual <= 1e-12 && res_status.tangent_residual <= 1e-12

    # Atomic refusal on non-convergence in rattle_project_with_status!
    st_bad = DY.State([0.0 1.3; 0.0 0.0; 0.0 0.0], zeros(3, 2))
    q_bad0 = copy(st_bad.q); p_bad0 = copy(st_bad.p)
    bad_status = DY.rattle_project_with_status!(st_bad, sys, cs; tolerance = 1e-16, max_iterations = 1)
    @assert !bad_status.converged
    @assert st_bad.q == q_bad0 && st_bad.p == p_bad0
    @assert bad_status.position_residual > 1e-16

    # Timestep-halving O(dt^2) convergence of constrained RATTLE Verlet against fine reference.
    Tend = 1.0
    dt_ref = 0.00025
    st_ref = DY.copy(st0)
    for _ in 1:round(Int, Tend / dt_ref)
        DY.velocity_verlet_step!(st_ref, sys, dt_ref; constraints = cs)
    end
    errs = Float64[]
    for dt in (0.02, 0.01, 0.005, 0.0025)
        st = DY.copy(st0)
        for _ in 1:round(Int, Tend / dt)
            DY.velocity_verlet_step!(st, sys, dt; constraints = cs)
        end
        push!(errs, maximum(abs, st.q .- st_ref.q))
    end
    xs = log.([0.02, 0.01, 0.005, 0.0025]); ys = log.(errs)
    order = (length(xs) * sum(xs .* ys) - sum(xs) * sum(ys)) /
            (length(xs) * sum(abs2, xs) - sum(xs)^2)
    @assert 1.8 < order < 2.2 "RATTLE global convergence order $order is not O(dt^2)"

    # Linear and angular momentum conservation over 2000 RATTLE steps.
    st_m = DY.copy(st0)
    P0 = vec(sum(st_m.p, dims = 2))
    L0 = sum(cross(st_m.q[:, j], st_m.p[:, j]) for j in 1:2)
    max_dP = 0.0; max_dL = 0.0
    for _ in 1:2000
        DY.velocity_verlet_step!(st_m, sys, 0.002; constraints = cs)
        Pt = vec(sum(st_m.p, dims = 2))
        Lt = sum(cross(st_m.q[:, j], st_m.p[:, j]) for j in 1:2)
        max_dP = max(max_dP, maximum(abs, Pt .- P0))
        max_dL = max(max_dL, maximum(abs, Lt .- L0))
    end
    rp, rt = DY.rattle_residuals(st_m, sys, cs)
    @assert max_dP < 1e-11 && max_dL < 1e-11
    @assert rp < 1e-12 && rt < 1e-12
    @printf("  RATTLE O(dt^2) order=%.3f, max|ΔP|=%.2e, max|ΔL|=%.2e, res=(%.2e, %.2e)\n",
            order, max_dP, max_dL, rp, rt)
    return true
end

function test_checkpoint_deterministic_replay()
    # 1. Deterministic Velocity Verlet replay
    st, sys = oscillator(1.2, 0.9, 1.0)
    st.p[1, 1] = 0.3; st.p[1, 2] = -0.3
    for _ in 1:50
        DY.velocity_verlet_step!(st, sys, 0.005)
    end
    cp = DY.checkpoint(st; step = 50, t = 0.25)
    st_cont = DY.copy(st)
    for _ in 1:100
        DY.velocity_verlet_step!(st_cont, sys, 0.005)
    end
    st_replay = DY.State(zeros(3, 2), zeros(3, 2))
    DY.restore!(st_replay, cp)
    for _ in 1:100
        DY.velocity_verlet_step!(st_replay, sys, 0.005)
    end
    @assert st_replay.q == st_cont.q && st_replay.p == st_cont.p && st_replay.F == st_cont.F

    # 2. Deterministic BAOAB stochastic replay with RNG snapshot
    rng = MersenneTwister(42)
    DY.forces!(sys, st.q, st.F); st.force_valid = true
    for _ in 1:25
        DY.langevin_baoab_step!(st.q, st.p, st.F, sys, 0.005, 1.0, 2.0, rng)
    end
    cp_stoch = DY.checkpoint(st; step = 25, t = 0.125, rng = rng)
    st_s_cont = DY.copy(st)
    for _ in 1:50
        DY.langevin_baoab_step!(st_s_cont.q, st_s_cont.p, st_s_cont.F, sys, 0.005, 1.0, 2.0, rng)
    end
    st_s_rep = DY.State(zeros(3, 2), zeros(3, 2))
    rng_rep = MersenneTwister(999)
    DY.restore!(st_s_rep, cp_stoch; rng = rng_rep)
    for _ in 1:50
        DY.langevin_baoab_step!(st_s_rep.q, st_s_rep.p, st_s_rep.F, sys, 0.005, 1.0, 2.0, rng_rep)
    end
    @assert st_s_rep.q == st_s_cont.q && st_s_rep.p == st_s_cont.p && st_s_rep.F == st_s_cont.F
    println("  deterministic checkpoint replay: bit-identical for Verlet and stochastic BAOAB")
    return true
end

function _run_all()
    for (name, fn) in [
        ("energy_bounded_not_drifting", test_energy_is_bounded_not_drifting),
        ("time_reversibility", test_time_reversibility),
        ("second_order_convergence", test_second_order_convergence),
        ("step_doubling_defect_order", test_step_doubling_defect_scales_as_dt3),
        ("compensated_summation", test_compensated_summation_matters),
        ("drift_detector_validation", test_energy_drift_report_detects_a_real_leak),
        ("select_dt", test_select_dt_picks_a_valid_fixed_step),
        ("rattle_constraints", test_rattle_distance_constraints),
        ("force_model_gate", test_force_model_gate_refuses_inconsistent_cutoffs),
        ("baoab_canonical", test_baoab_samples_canonical_gaussian),
        ("integrator_state_machine", test_integrator_state_machine_and_classification),
        ("rattle_convergence_momentum", test_rattle_convergence_momentum_and_status),
        ("checkpoint_replay", test_checkpoint_deterministic_replay),
    ]
        println("TEST ", name)
        fn()
    end
    println("")
    println("ALL TESTS PASSED")
end

_run_all()
