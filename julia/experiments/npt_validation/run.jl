"""End-to-end validation of the isotropic NPT Monte-Carlo barostat.

Run from `julia`:

    JULIA_DEPOT_PATH=~/.cache/biology-mm/julia-depot \
      ~/.cache/biology-mm/julia-1.13.3/bin/julia --project=. \
      experiments/npt_validation/run.jl

What is measured
----------------
1. Ideal-gas volume distribution. A non-interacting `N`-particle system has the
   exact NPT volume density `p(V) ∝ V^N exp(-beta*P*V)`, i.e.
   `Gamma(shape = N+1, scale = 1/(beta*P))`. Mean `(N+1)/(beta*P)`, variance
   `(N+1)/(beta*P)^2`. This is an exact analytic target, not a fit.
2. The equation of state in estimator form: `N/(beta*V)` has ensemble mean
   exactly `P`, and the configurational estimator `2K/(3V)` (virial is zero for
   an ideal gas) has the same mean. The run that samples the *volume* uses the
   real barostat kernel inside a BAOAB dynamics loop.
3. Two-seed reproducibility of mean volume, mean pressure, and volume variance.
4. Volume fluctuations against `kT*kappa_T / <V>` with the ideal-gas
   `kappa_T = 1/P`; for `N+1` Gamma sampling this relation is exact:
   `var(V)/<V>^2 = 1/(N+1) = kT*kappa_T/<V>`.
5. Enthalpy stationarity, block-averaged, over the production window.
6. A negative control that deliberately omits the `(N+1)*dlogV` Jacobian and
   shows the sampled volume collapses; the collapse is proposal-step dependent,
   which is the signature of the non-normalisable measure the omitted factor
   produces.

The position/box scaling, proposal reversal, seeded replay, constraint refusal
and missing-virial refusal are checked in `test/test_barostat.jl`.
"""

using FoundationCore
using Random
using Statistics
using Printf

module BarostatHost
using FoundationCore
const Types = FoundationCore.Types
include(normpath(joinpath(@__DIR__, "..", "..", "src", "MonteCarloBarostat.jl")))
end

include(joinpath(@__DIR__, "analysis.jl"))
using .NPTAnalysis

const MCB = BarostatHost.MonteCarloBarostat
const T = FoundationCore.Types
const Dyn = FoundationCore.Dynamics
const Fac = FoundationCore.Factors

# ---------------------------------------------------------------------------
# exact ideal-gas reference values
# ---------------------------------------------------------------------------

const N = 32
const BETA = 1.0
const PTARGET = 0.5
const V_EXACT = (N + 1) / (BETA * PTARGET)          # 66.0
const VAR_EXACT = (N + 1) / (BETA * PTARGET)^2      # 132.0
const H_EXACT = (N + 1) / BETA                      # 33.0 (U = 0)
const COMPRESS_EXACT = 1 / (N + 1)                  # 0.030303...
const EOS_EXACT = PTARGET

const SEED_A = 2024
const SEED_B = 2025
const DLNV = 0.15
const NBURN = 5_000
const NPROD = 200_000
const THIN = 2

ideal_energy(q, box) = 0.0
ideal_virial(q, box) = 0.0

# ---------------------------------------------------------------------------
# (1) pure volume-move sampling of the ideal-gas NPT measure
# ---------------------------------------------------------------------------

"""
    ideal_gas_volume_run(seed; dlnv, nburn, nprod, thin, jacobian)

Pure configurational volume moves on an ideal gas. `jacobian = false` is the
deliberately wrong negative control: it subtracts `(N+1)*dlogV` from the correct
log ratio and must never be used for production.
"""
function ideal_gas_volume_run(seed; dlnv = DLNV, nburn = NBURN, nprod = NPROD,
                              thin = THIN, jacobian = true)
    rng = MersenneTwister(seed)
    L = V_EXACT^(1 / 3)
    box = T.OrthorhombicBox((L, L, L))
    q = zeros(3, N)
    volumes = Float64[]
    accepted = 0
    total = 0

    if jacobian
        barostat = MCB.VolumeMoveBarostat(BETA, PTARGET, dlnv)
        for i in 1:(nburn + nprod)
            r = MCB.step!(barostat, q, box, ideal_energy, rng)
            box = r.box
            accepted += r.accepted
            total += 1
            i > nburn && (i - nburn) % thin == 0 && push!(volumes, T.box_volume(box))
        end
        return (volumes = volumes, acceptance = accepted / total, underflow = false)
    end

    # Negative control: identical proposal machinery, Jacobian term deleted.
    V = V_EXACT
    underflow = false
    for i in 1:(nburn + nprod)
        d = (2rand(rng) - 1) * dlnv
        newV = V * exp(d)
        if !(newV > 0) || !isfinite(newV)
            underflow = true
            break
        end
        log_ratio = MCB.log_acceptance(BETA, PTARGET, 0.0, V, newV, N) - (N + 1) * d
        if log(rand(rng)) < min(0.0, log_ratio)
            V = newV
            accepted += 1
        end
        total += 1
        i > nburn && (i - nburn) % thin == 0 && push!(volumes, V)
    end
    return (volumes = volumes, acceptance = accepted / max(total, 1), underflow = underflow)
end

# ---------------------------------------------------------------------------
# (2) BAOAB dynamics loop + NPT volume moves
# ---------------------------------------------------------------------------

"""
    ideal_gas_hybrid_run(seed; dlnv, nburn, nprod, thin, dt, gamma, md)

Alternate `md` BAOAB Langevin steps at fixed volume with one Monte-Carlo volume
move. Both kernels leave the NPT target invariant (BAOAB preserves the canonical
factor at fixed `V`; the volume move preserves the `V^N exp(-beta*P*V)` marginal),
so their composition is a valid NPT sampler. Returns volume, configurational
pressure, kinetic energy and potential energy samples.
"""
function ideal_gas_hybrid_run(seed; dlnv = DLNV, nburn = NBURN, nprod = 60_000,
                              thin = THIN, dt = 0.002, gamma = 5.0, md = 5)
    rng = MersenneTwister(seed)
    L = V_EXACT^(1 / 3)
    box = T.OrthorhombicBox((L, L, L))
    q = rand(rng, 3, N) .* L
    p = randn(rng, 3, N) ./ sqrt(BETA)
    F = zeros(3, N)
    system = Dyn.HarmonicSystem(Fac.AbstractFactor[], ones(N),
                                Fac.Context(T.NoBox(), T.REDUCED, N))
    model = MCB.NPTModel(ideal_energy; virial = ideal_virial, masses = 1.0)
    barostat = MCB.NPTBarostat(BETA, PTARGET, dlnv, model)

    volumes = Float64[]
    pressures = Float64[]
    kinetics = Float64[]
    potentials = Float64[]
    accepted = 0
    total = 0

    for i in 1:(nburn + nprod)
        for _ in 1:md
            Dyn.langevin_baoab_step!(q, p, F, system, dt, gamma, BETA, rng)
        end
        r = MCB.npt_step!(barostat, q, p, box, rng)
        box = r.box
        accepted += r.accepted
        total += 1
        # Periodic relabelling is exact for this non-interacting system; it keeps
        # free-flight coordinates bounded without touching the volume measure.
        for d in 1:3
            @views q[d, :] .= mod.(q[d, :], box.L[d])
        end
        if i > nburn && (i - nburn) % thin == 0
            push!(volumes, T.box_volume(box))
            push!(pressures, r.pressure_new)
            push!(kinetics, 0.5 * sum(abs2, p))
            push!(potentials, r.new_energy)
        end
    end
    return (volumes = volumes, pressures = pressures, kinetics = kinetics,
            potentials = potentials, acceptance = accepted / total)
end

# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------

function summarize_volumes(run)
    V = run.volumes
    m = mean(V)
    se = mean_stderr(V)
    v = var(V)
    vse = block_var_stderr(V, 20)
    return (n = length(V), mean = m, se = se, naive_se = naive_stderr(V),
            tau = tau_int(V), var = v, var_se = vse,
            rel_var = v / m^2, acceptance = run.acceptance)
end

function summarize_pressure(run)
    P = run.pressures
    return (n = length(P), mean = mean(P), se = mean_stderr(P),
            tau = tau_int(P), acceptance = run.acceptance)
end

function enthalpy_stationarity(run)
    H = PTARGET .* run.volumes .+ run.potentials
    nb = 20
    bm = block_means(H, nb)
    half = half_mean_agreement(H)
    slope = linear_slope(bm)
    return (mean = mean(H), se = mean_stderr(H), blocks = bm,
            block_se = std(bm) / sqrt(nb), half = half, slope = slope)
end

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

function main()
    println("=" ^ 78)
    println("NPT BAROSTAT VALIDATION — ideal gas, exact NPT reference")
    println("=" ^ 78)
    @printf("N=%d  beta=%.3f  P_target=%.3f  dlnv=%.3f\n", N, BETA, PTARGET, DLNV)
    @printf("Exact Gamma(N+1, 1/(beta P)):  <V>=%.6f  var(V)=%.6f  sd=%.6f\n",
            V_EXACT, VAR_EXACT, sqrt(VAR_EXACT))
    @printf("Exact <H>=%.6f   exact var/<V>^2 = kT kappa_T/<V> = %.8f (kappa_T=1/P)\n\n",
            H_EXACT, COMPRESS_EXACT)

    # ---- (1) pure volume moves, two seeds --------------------------------
    println("-" ^ 78)
    println("(1) Pure volume-move sampling of p(V) ∝ V^N exp(-beta P V)")
    println("-" ^ 78)
    mc_a = ideal_gas_volume_run(SEED_A)
    mc_b = ideal_gas_volume_run(SEED_B)
    sa, sb = summarize_volumes(mc_a), summarize_volumes(mc_b)
    for (label, s) in (("seed $SEED_A", sa), ("seed $SEED_B", sb))
        @printf("%-12s n=%d  <V>=%.4f ± %.4f  (IAT tau=%.2f, naive se=%.4f)\n",
                label, s.n, s.mean, s.se, s.tau, s.naive_se)
        @printf("%-12s var(V)=%.3f ± %.3f  var/<V>^2=%.6f  acceptance=%.3f\n",
                "", s.var, s.var_se, s.rel_var, s.acceptance)
        @printf("%-12s z(<V> vs exact)=%.2f  z(var vs exact)=%.2f\n", "",
                zscore(s.mean, V_EXACT, s.se),
                zscore(s.var, VAR_EXACT, s.var_se))
    end
    zV_2seed_mc = abs(sa.mean - sb.mean) / sqrt(sa.se^2 + sb.se^2)
    ks_mc = two_sample_ks(mc_a.volumes, mc_b.volumes)
    ks_crit_mc = 1.36 * sqrt((sa.n + sb.n) / (sa.n * sb.n))
    @printf("two-seed volume: |Δ<V>|/combined se = %.2f ; KS=%.4f (5%% critical %.4f)\n",
            zV_2seed_mc, ks_mc, ks_crit_mc)

    eos_a = N ./ (BETA .* mc_a.volumes)
    @printf("EOS estimator N/(beta V): <%.5f> ± %.5f  vs exact P=%.5f  z=%.2f\n",
            mean(eos_a), mean_stderr(eos_a), EOS_EXACT,
            zscore(mean(eos_a), EOS_EXACT, mean_stderr(eos_a)))

    comp_err_a = abs(sa.rel_var - COMPRESS_EXACT) / COMPRESS_EXACT
    comp_err_b = abs(sb.rel_var - COMPRESS_EXACT) / COMPRESS_EXACT
    @printf("compressibility: rel var seed A error=%.2f%%, seed B error=%.2f%%\n\n",
            100 * comp_err_a, 100 * comp_err_b)

    # ---- (2) BAOAB + MC hybrid ------------------------------------------
    println("-" ^ 78)
    println("(2) BAOAB dynamics loop + NPT volume moves (configurational pressure)")
    println("-" ^ 78)
    hy_a = ideal_gas_hybrid_run(SEED_A)
    hy_b = ideal_gas_hybrid_run(SEED_B)
    ha, hb = summarize_volumes(hy_a), summarize_volumes(hy_b)
    pa, pb = summarize_pressure(hy_a), summarize_pressure(hy_b)
    for (label, s, pr) in (("seed $SEED_A", ha, pa), ("seed $SEED_B", hb, pb))
        @printf("%-12s n=%d  <V>=%.4f ± %.4f  z(exact)=%.2f  acceptance=%.3f\n",
                label, s.n, s.mean, s.se, zscore(s.mean, V_EXACT, s.se), s.acceptance)
        @printf("%-12s <P_config>=%.5f ± %.5f  z(P_target)=%.2f  (IAT tau=%.2f)\n",
                "", pr.mean, pr.se, zscore(pr.mean, PTARGET, pr.se), pr.tau)
        @printf("%-12s var(V)=%.3f ± %.3f  var/<V>^2=%.6f  (exact %.6f)\n",
                "", s.var, s.var_se, s.rel_var, COMPRESS_EXACT)
    end
    zV_2seed_hy = abs(ha.mean - hb.mean) / sqrt(ha.se^2 + hb.se^2)
    zP_2seed_hy = abs(pa.mean - pb.mean) / sqrt(pa.se^2 + pb.se^2)
    ks_hy = two_sample_ks(hy_a.volumes, hy_b.volumes)
    ks_crit_hy = 1.36 * sqrt((ha.n + hb.n) / (ha.n * hb.n))
    @printf("two-seed: z(<V>)=%.2f  z(<P>)=%.2f ; KS=%.4f (critical %.4f)\n\n",
            zV_2seed_hy, zP_2seed_hy, ks_hy, ks_crit_hy)

    # ---- (3) enthalpy stationarity --------------------------------------
    println("-" ^ 78)
    println("(3) Enthalpy-like H = U + P V, block-averaged stationarity")
    println("-" ^ 78)
    ea, eb = enthalpy_stationarity(hy_a), enthalpy_stationarity(hy_b)
    for (label, e) in (("seed $SEED_A", ea), ("seed $SEED_B", eb))
        @printf("%-12s <H>=%.4f ± %.4f  (exact %.4f)\n", label, e.mean, e.se, H_EXACT)
        @printf("%-12s half1-half2 = %+.4f ± %.4f (z=%.2f); block slope=%+.2e ± %.2e (z=%.2f)\n",
                "", e.half.difference, e.half.se, e.half.z, e.slope.slope, e.slope.se, e.slope.z)
    end
    println()

    # ---- (4) Jacobian negative control ----------------------------------
    println("-" ^ 78)
    println("(4) NEGATIVE CONTROL: omit the (N+1) dlogV Jacobian")
    println("-" ^ 78)
    @printf("%-18s %12s %14s %12s\n", "run", "<V>", "z vs exact", "acceptance")
    @printf("%-18s %12.4f %14.1f %12.3f\n", "correct (dl=0.15)", sa.mean,
            zscore(sa.mean, V_EXACT, sa.se), sa.acceptance)
    control = Dict{Float64, Any}()
    for dl in (0.15, 0.05)
        c = ideal_gas_volume_run(SEED_A; dlnv = dl, jacobian = false)
        s = summarize_volumes(c)
        control[dl] = s
        @printf("%-18s %12.4f %14.1f %12.3f  underflow=%s\n",
                "no Jacobian (dl=$dl)", s.mean, zscore(s.mean, V_EXACT, s.se),
                s.acceptance, string(c.underflow))
    end
    ratio_15 = sa.mean / control[0.15].mean
    ratio_05 = sa.mean / control[0.05].mean
    cutoff_ratio = control[0.05].mean / control[0.15].mean
    @printf("correct / no-Jacobian mean-volume ratios: %.1fx (dl=0.15), %.1fx (dl=0.05)\n",
            ratio_15, ratio_05)
    @printf("no-Jacobian mean is proposal-step dependent: <V>(dl=0.05)/<V>(dl=0.15)=%.2f\n",
            cutoff_ratio)
    println("=> the omitted term is the Jacobian of the log-volume proposal; without it")
    println("   the sampled V-measure is non-normalisable at fixed step size and the")
    println("   volume collapses toward zero. The factor is necessary, not cosmetic.")
    println()

    # ---- verdict ---------------------------------------------------------
    println("=" ^ 78)
    println("CHECKS (pass thresholds stated)")
    println("=" ^ 78)
    checks = Pair{String,Bool}[
        "pure MC <V> within 4 autocorr SE of (N+1)/(betaP)" =>
            zscore(sa.mean, V_EXACT, sa.se) <= 4 && zscore(sb.mean, V_EXACT, sb.se) <= 4,
        "pure MC var(V) within 4 block SE of exact" =>
            zscore(sa.var, VAR_EXACT, sa.var_se) <= 4 && zscore(sb.var, VAR_EXACT, sb.var_se) <= 4,
        "EOS <N/(beta V)> within 4 SE of P" =>
            zscore(mean(eos_a), EOS_EXACT, mean_stderr(eos_a)) <= 4,
        "two-seed pure-MC <V> within 3 combined SE" => zV_2seed_mc <= 3,
        "two-sample KS below 5% critical (pure MC)" => ks_mc <= ks_crit_mc,
        "hybrid <P_config> within 4 SE of P_target" =>
            zscore(pa.mean, PTARGET, pa.se) <= 4 && zscore(pb.mean, PTARGET, pb.se) <= 4,
        "hybrid <V> within 4 SE of exact" =>
            zscore(ha.mean, V_EXACT, ha.se) <= 4 && zscore(hb.mean, V_EXACT, hb.se) <= 4,
        "two-seed hybrid <V> and <P> within 3 combined SE" =>
            zV_2seed_hy <= 3 && zP_2seed_hy <= 3,
        "volume fluctuation within 10% of 1/(N+1)" =>
            comp_err_a <= 0.10 && comp_err_b <= 0.10,
        "hybrid var(V) within 10% of exact and 4 block SE" =>
            abs(ha.var - VAR_EXACT) / VAR_EXACT <= 0.10 &&
            abs(hb.var - VAR_EXACT) / VAR_EXACT <= 0.10 &&
            zscore(ha.var, VAR_EXACT, ha.var_se) <= 4 &&
            zscore(hb.var, VAR_EXACT, hb.var_se) <= 4,
        "enthalpy stationary: half means and block slope within 3 SE" =>
            ea.half.z <= 3 && eb.half.z <= 3 && ea.slope.z <= 3 && eb.slope.z <= 3,
        "Jacobian control collapses <V> by >10x" => ratio_15 > 10 && ratio_05 > 10,
        "Jacobian control is proposal-step dependent (>2x)" => cutoff_ratio > 2,
    ]
    npass = 0
    for (name, ok) in checks
        @printf("[%s] %s\n", ok ? "PASS" : "FAIL", name)
        ok && (npass += 1)
    end
    @printf("\n%d/%d checks passed\n", npass, length(checks))
    if npass == length(checks)
        println("VERDICT: barostat kernel validated in the exact ideal-gas NPT limit.")
    else
        println("VERDICT: NOT validated — see failing checks above.")
    end
    println("LIMITATION: no interacting-system pressure mean is reported because no")
    println("force-field factor in this snapshot exposes a virial backend; the")
    println("barostat refuses such a model (MissingVirialError) rather than")
    println("substituting an estimator. Interacting-system NPT remains unvalidated.")
    return npass == length(checks)
end

main() || exit(1)
