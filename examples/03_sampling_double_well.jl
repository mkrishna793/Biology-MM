# ============================================================================
#  examples/03_sampling_double_well.jl
#
#  Sampling: HMC and replica exchange, and the difference between "converged"
#  and "correct".
#
#  Run from julia:  julia --project=. ../examples/03_sampling_double_well.jl
#  Deterministic: every RNG below is an explicitly seeded MersenneTwister.
#
#  The point of this example is that a sampler can pass every *internal*
#  diagnostic - acceptance near 1, a short autocorrelation time, a stable
#  estimator - and still be sampling the wrong distribution. The only way to
#  see that is to compare against targets whose answers are known exactly:
#    - a Gaussian, where <x> and Var[x] are closed form;
#    - a SYMMETRIC double well, where <x> = 0 by symmetry whatever the barrier;
#    - the double-well CDF, which Sampling integrates by quadrature
#      (`doublewell_cdf_exact`), so an empirical CDF can be tested against it.
#
#  What is NOT demonstrated: anything about real molecules. These are 1-D
#  targets chosen because their answers are known. Sampler sweeps are not
#  physical time, no kinetics or rates are measured here, and agreement with a
#  1-D reference is not evidence about a high-dimensional molecular target.
# ============================================================================

using Printf
using Random
using Statistics

include(joinpath(@__DIR__, "..", "julia", "src", "FoundationCore.jl"))
using .FoundationCore
import .FoundationCore.Sampling as SP

const T0 = time_ns()
wall() = (time_ns() - T0) / 1e9

nfail = 0

"""Records a verdict and prints it. `detail` is pre-formatted text."""
function checkgate(name::AbstractString, ok::Bool, detail...)
    ok || (global nfail = nfail + 1)
    print(rpad("  [$name] $(ok ? "PASS" : "FAIL")", 37), " ")
    for d in detail
        print(d)
    end
    println()
    return ok
end

"""Lag-`lag` autocorrelation. Defined here because Sampling exposes only the
integrated time, and an oscillating chain has to be diagnosed at the level of
individual correlations."""
function rho_at(x::AbstractVector{Float64}, lag::Int)
    mu = mean(x)
    num = sum((x[i] - mu) * (x[i + lag] - mu) for i in 1:(length(x) - lag))
    den = sum(abs2, x .- mu)
    return num / den
end

"Phase advance per step of the leapfrog map for a harmonic target of frequency w."
function rho1_dispersion(dt::Float64, L::Int, w::Float64)
    c = 1.0 - (w * dt)^2 / 2.0          # cos(Omega*dt) for leapfrog on q'' = -w^2 q
    return cos(acos(clamp(c, -1.0, 1.0)) * L)
end

println("=" ^ 78)
println(" Foundation core :: 03 - HMC and replica exchange against known targets")
println("=" ^ 78)

# ---------------------------------------------------------------------------
# 1. The proposal map must be an involution, or the Metropolis correction is
#    not exact. leapfrog() negates the momentum at the end precisely so that
#    T(T(z)) = z; that is a numerical claim, so measure it.
# ---------------------------------------------------------------------------
println("\n[1] Reversibility of the proposal map, T(T(z)) = z")
dw = SP.DoubleWell(1.0, 4.0)                  # U(x) = (x^2 - 4)^2, minima at +-2
Ud(x) = dw(x)
Gd(x) = SP.grad(dw, x)
mass = [1.0]
dt = 0.08
Lstep = 10

q0 = [-1.7]
p0 = [0.9]
qa, pa = SP.leapfrog(q0, p0, Gd, mass, dt, Lstep)
qb, pb = SP.leapfrog(qa, pa, Gd, mass, dt, Lstep)
dev = max(abs(qb[1] - q0[1]), abs(pb[1] - p0[1]))
@printf(
    "   two applications return to the start: dq = %.3e  dp = %.3e  (from q = %.3f, p = %.3f)\n",
    abs(qb[1] - q0[1]), abs(pb[1] - p0[1]), q0[1], p0[1])
println("   this holds to roundoff only because leapfrog negates p at the end. Without")
println("   that negation T is not an involution, and the Metropolis ratio is wrong.")
checkgate("involution", dev < 1e-12, "deviation ", @sprintf("%.2e", dev),
    ", roundoff floor at |q| ~ 1.7 is ", @sprintf("%.1e", 10 * eps()))

# ---------------------------------------------------------------------------
# 2. HMC against a Gaussian: U = k(x-mu)^2 with k = 4, beta = 1, so the target
#    variance is exactly 1/(beta*k) = 0.25. Sweep the step size.
# ---------------------------------------------------------------------------
println("\n[2] HMC against an exactly known Gaussian, target Var[x] = 1/(beta*k) = 0.25")
g = SP.GaussianTarget(4.0, 0.7)
Ug(x) = g(x)
Gg(x) = SP.grad(g, x)
w_gauss = sqrt(4.0 / mass[1])          # the target's own oscillation frequency
LG = 10

function sweep_gaussian(dtlist; nsteps = 20_000, L = LG)
    out = NamedTuple[]
    for dts in dtlist
        r = SP.hmc_trajectory([0.7], Ug, Gg, mass, dts, L, nsteps, 1.0, MersenneTwister(3))
        x = vec(r.samples)
        v = var(x)
        se = v * sqrt(2.0 / length(x))          # error bar for n INDEPENDENT draws
        tau = SP.integrated_autocorrelation(x)
        push!(out, (dt = dts, acc = SP.acceptance_rate(r), mn = mean(x), var = v,
            z = (v - 0.25) / se, zeff = (v - 0.25) / (se * sqrt(tau)), rho1 = rho_at(x, 1),
            pred = rho1_dispersion(dts, L, w_gauss), tau = tau, traj = dts * L))
    end
    return out
end

"Variance estimate of one chain per seed: what the run-to-run scatter really is."
function var_by_seed(dt; nsteps = 20_000, L = LG, seeds = (3, 4, 5, 6, 7, 8))
    return [var(vec(SP.hmc_trajectory([0.7], Ug, Gg, mass, dt, L, nsteps, 1.0,
        MersenneTwister(s)).samples)) for s in seeds]
end

rows = sweep_gaussian((0.02, 0.08, 0.20, 0.35, 0.45, 0.50, 0.55, 0.60))
@printf(
    "   %-6s %-8s %-11s %-10s %-7s %-9s %-8s %-8s %-6s %s\n",
    "dt", "accept", "mean", "var", "z", "z/sqrt(tau)", "rho_1", "rho_1 map", "tau", "L*dt")
for r in rows
    @printf(
        "   %-6.2f %-8.4f %-11.5f %-10.5f %-7.1f %-9.1f %-8.3f %-8.3f %-6.2f %.2f\n",
        r.dt, r.acc, r.mn, r.var, r.z, r.zeff, r.rho1, r.pred, r.tau, r.traj)
end
row008 = only(filter(r -> r.dt == 0.08, rows))
row45 = only(filter(r -> r.dt == 0.45, rows))
row002 = only(filter(r -> r.dt == 0.02, rows))
nnaive = count(r -> abs(r.z) > 4.0, rows)
devs = sort(rows; by = r -> -abs(r.rho1 - r.pred))
lowacc = sort(rows; by = r -> r.acc)
worst2 = [d.dt for d in devs[1:2]]
low2 = [a.dt for a in lowacc[1:2]]
maxmapdev = maximum(r -> abs(r.rho1 - r.pred), rows)
signs_ok = all(r -> r.rho1 * r.pred > 0.0, rows)
phase45 = acos(clamp(1.0 - (w_gauss * 0.45)^2 / 2.0, -1.0, 1.0)) * LG
println("   z is the deviation of the measured variance from the exact 0.25 in units of")
println("   the error bar 20,000 INDEPENDENT draws would have; z/sqrt(tau) is the same")
println("   deviation with the chain's effective sample size. Acceptance is above 0.91 in")
println("   every row, so acceptance alone would have flagged nothing.")
println("   The column rho_1 map is not a fit. It is the leapfrog map's own phase")
println("   advance: cos(Omega*dt) = 1 - (w*dt)^2/2 for a harmonic target of frequency")
@printf("   w = sqrt(k/m) = %.1f, so a trajectory of L = %d steps rotates phase by\n",
    w_gauss, LG)
println("   Omega*L, and the lag-1 correlation of x must be cos(Omega*L). The next table")
println("   is the same eight rows sorted by how well that deterministic prediction is")
println("   met; every row has the predicted sign.")
@printf("   %-8s %-9s %-11s %-9s %-10s %s\n",
    "dt", "rho_1", "rho_1 (map)", "miss", "accept", "reject/step")
for d in devs
    @printf("   %-8.2f %-9.3f %-11.3f %-9.3f %-10.4f %.4f\n",
        d.dt, d.rho1, d.pred, d.rho1 - d.pred, d.acc, 1.0 - d.acc)
end
@printf("   The two largest misses, %.3f and %.3f, fall at dt = %.2f and %.2f: the two\n",
    abs(devs[1].rho1 - devs[1].pred), abs(devs[2].rho1 - devs[2].pred), devs[1].dt, devs[2].dt)
@printf("   lowest-acceptance rows in the sweep (%.4f and %.4f). The deterministic formula\n",
    lowacc[1].acc, lowacc[2].acc)
println("   knows nothing about Metropolis rejections, so the residual sits exactly where")
println("   it must. Resonance is then not mysterious:")
@printf("   at dt = 0.45 the rotation is %.3f rad, %.2f periods, so a sample lands almost\n",
    phase45, phase45 / (2pi))
println("   exactly opposite the one before it and the chain cycles instead of mixing.")
println("   rho_1 = -0.97 says so; the initial-positive-sequence estimator truncates at")
println("   the first non-positive correlation, so tau reports 1.0 and ESS = n. The blind")
println("   spot, in the open.")
println()
@printf("   %d rows disagree with the known answer by more than 4 naive sigmas, and the\n", nnaive)
println("   two fail in opposite ways:")
println("    - dt = 0.02: rho_1 ~ 0.93, tau ~ 24. Small steps make the chain a correlated")
println("      random walk, so the naive error bar is too small. The effective sample")
println("      size repairs the error bar, and the answer then agrees.")
println("    - dt = 0.45: tau reports 1.0, so the effective-sample-size correction does")
println("      nothing and the deviation still reads 15 sigma. Whether that means the")
println("      sampler is biased or the error bar is fiction is NOT settled by one run.")
println("      It is settled by repeating the run:")
@printf("   %-8s %-9s %-42s %-15s %s\n",
    "dt", "tau(IPS)", "variance over 6 seeds", "sd (measured)", "sd / tau-corrected")
sc = Tuple{Float64, Float64, Vector{Float64}, Float64, Float64, Float64, Float64}[]
for dts in (0.02, 0.45)
    vs = var_by_seed(dts)
    tau_here = only(filter(r -> r.dt == dts, rows)).tau
    sdv = std(vs)
    se_corr = mean(vs) * sqrt(2.0 * tau_here / 20_000)
    push!(sc, (dts, tau_here, vs, sdv, sdv / se_corr, mean(vs), sdv / sqrt(length(vs))))
end
for s in sc
    vt = join(map(v -> @sprintf("%.5f", v), s[3]), " ")
    @printf("   %-8.2f %-9.2f %-42s %-15.5f %.2f\n", s[1], s[2], vt, s[4], s[5])
end
sm002 = sc[1]
sm045 = sc[2]
@printf("   At dt = 0.02 the six seeds scatter by %.5f and the tau-corrected bar predicts\n", sm002[4])
@printf("   %.5f: right to %.0f%%. The ESS machinery works where its assumption holds,\n",
    mean(sm002[3]) * sqrt(2.0 * sm002[2] / 20_000), 100 * abs(sm002[5] - 1.0))
println("   that correlations decay monotonically.")
@printf("   At dt = 0.45 it does not. The scatter is %.5f, which is %.1f times the bar\n", sm045[4], sm045[5])
println("   that its tau = 1.0 licenses - and with tau = 1.0 that bar is the naive one,")
println("   so no correction is even attempted.")
@printf("   The single run's z/sqrt(tau) = %.1f was therefore not a 15-sigma discovery\n", row45.zeff)
println("   about the sampler; it was a 15-sigma statement made with an error bar short")
println("   by an order of magnitude.")
@printf("   And the six-seed mean is %.5f +- %.5f against an exact 0.25000: %.1f sigma.\n",
    sm045[6], sm045[7], (sm045[6] - 0.25) / sm045[7])
println("   So this file does NOT establish that the resonant step size is biased. What it")
println("   does establish is worse in practice: at a resonant step size the reported")
println("   precision is wrong by about an order of magnitude, the usual diagnostics")
println("   cannot see it, and a single run cannot tell an off answer from an")
println("   unresolvable one. Repetition on different seeds is the only thing in this")
println("   section that measures an error bar instead of asserting one. The Metropolis")
println("   correction protects the DISTRIBUTION from integrator error; it protects")
println("   neither the chain from a resonant step size nor the error bar from either.")
checkgate("ergodic dt hits the known answer", abs(row008.zeff) < 4.0,
    "at dt = 0.08, z = ", @sprintf("%.1f", row008.z), ", z/sqrt(tau) = ",
    @sprintf("%.1f", row008.zeff))
checkgate("correlated dt explained by ESS", abs(row002.z) > 4.0 && abs(row002.zeff) < 4.0,
    "at dt = 0.02 the naive z = ", @sprintf("%.1f", row002.z), " is a wrong error bar, not",
    " a wrong answer (tau = ", @sprintf("%.1f", row002.tau), ")")
checkgate("the leapfrog map predicts rho_1",
    signs_ok && maxmapdev < 0.15 && Set(worst2) == Set(low2),
    "signs agree; max miss ", @sprintf("%.3f", maxmapdev),
    "; the two largest misses are the two lowest acceptances, dt = ",
    join(string.(worst2), ", "))
checkgate("ESS correction is calibrated where it is valid",
    0.7 < sm002[5] < 1.4, "dt = 0.02: measured / tau-predicted scatter = ",
    @sprintf("%.2f", sm002[5]))
checkgate("ESS correction fails when correlations alternate",
    sm045[5] > 4.0, "dt = 0.45: measured scatter is ", @sprintf("%.1f", sm045[5]),
    "x what tau = ", @sprintf("%.1f", sm045[2]), " allows")
checkgate("resonant bias is NOT established here",
    abs((sm045[6] - 0.25) / sm045[7]) < 3.0, "6-seed mean ", @sprintf("%.5f", sm045[6]),
    " +- ", @sprintf("%.5f", sm045[7]), " vs exact 0.25 = ",
    @sprintf("%.1f", (sm045[6] - 0.25) / sm045[7]), " sigma")

# ---------------------------------------------------------------------------
# 3. The centerpiece: a symmetric double well with a 16 kT barrier.
#    U(x) = (x^2 - 4)^2 has minima at x = +-2 with U = 0, and U(0) = 16.
#    Symmetry makes the exact answer <x> = 0 whatever beta is, so a chain stuck
#    in one well is provably wrong while still looking perfectly converged.
# ---------------------------------------------------------------------------
println("\n[3] Symmetric double well, 16 kT barrier: trapped chain vs replica exchange")
@printf(
    "   U(0) - U(+-2) = %.1f kT; exp(-16) = %.2e. An Arrhenius guess in units of one\n",
    Ud([0.0]) - Ud([2.0]), exp(-16.0))
@printf("   attempted move would put the waiting time near %.1e moves - an estimate,\n", exp(16.0))
println("   not a rate, because one HMC step is not a unit of time. Meanwhile the exact")
println("   <x> is 0 by symmetry at ANY barrier height, which is what makes a trapped")
println("   chain refutable rather than merely suspicious.")
mom = SP.doublewell_moments_exact(dw, 1.0)
@printf("   exact by quadrature: <x> = %+.4e   Var[x] = %.6f\n", mom.mean, mom.var)

nsm = 40_000
@printf("\n   step sizes below are dt = %.2f, L = %d steps - the pair that passed the\n", dt, Lstep)
println("   Gaussian test in section [2]. The failure measured here is not an artifact")
println("   of a bad integrator: the same parameters sample a harmonic target perfectly.")
rh = SP.hmc_trajectory([-2.0], Ud, Gd, mass, dt, Lstep, nsm, 1.0, MersenneTwister(11))
xplain = vec(rh.samples)
taup = SP.integrated_autocorrelation(xplain)
essp = length(xplain) / taup
@printf("\n   plain HMC from the cold start x = -2:  acceptance %.4f\n", SP.acceptance_rate(rh))
@printf(
    "     <x> = %+.5f   <x^2> = %.5f   crossings = %d   round trips = %d\n",
    mean(xplain), mean(xplain .^ 2), SP.barrier_crossings(xplain), SP.round_trip_rate(xplain))
@printf("     tau = %.1f  ->  ESS = %.0f of %d draws\n", taup, essp, length(xplain))
println("     every internal diagnostic in that block is green: high acceptance, tau ~ 1,")
println("     a tight-looking estimate. The chain is still wrong, because it never left")
println("     the well it started in.")

betas = [(0.05)^((i - 1) / 8) for i in 1:9]
pt = SP.replica_exchange([-2.0], Ud, Gd, mass, betas, dt, Lstep, nsm, 10, MersenneTwister(7))
xc = vec(pt.samples[1])
tauc = SP.integrated_autocorrelation(xc)
essc = length(xc) / tauc
sig_pt = 2.0 / sqrt(essc)
@printf("\n   replica exchange, 9 replicas, beta from 1.0 down to %.3f:\n", betas[end])
@printf(
    "     cold replica: <x> = %+.5f   crossings = %d   round trips = %d\n",
    mean(xc), SP.barrier_crossings(xc), SP.round_trip_rate(xc))
@printf("     tau = %.1f  ->  ESS = %.0f of %d draws\n", tauc, essc, length(xc))
println("     swap acceptance per adjacent pair: ",
    join(map(a -> @sprintf("%.2f", a), SP.swap_acceptance(pt)), " "))
println("     barrier crossings per replica:     ",
    join(map(s -> string(SP.barrier_crossings(vec(s))), pt.samples), " "))
@printf("\n   the cold replica sits %+.5f from the exact answer 0, and its own sigma is\n", mean(xc))
@printf("   ~%.3f, so it is consistent with 0. The trapped chain is off by %+.5f - about\n", sig_pt, mean(xplain))
@printf(
    "   %.0f of the sigmas it would quote for itself. That is this file's whole point.\n",
    abs(mean(xplain)) / (2.0 / sqrt(essp)))
checkgate("plain HMC stays trapped", SP.barrier_crossings(xplain) == 0 && abs(mean(xplain)) > 1.5,
    "crossings ", @sprintf("%d", SP.barrier_crossings(xplain)), ", <x> ",
    @sprintf("%.3f", mean(xplain)))
checkgate("replica exchange escapes", SP.barrier_crossings(xc) > 100 && abs(mean(xc)) < 5 * sig_pt,
    "crossings ", @sprintf("%d", SP.barrier_crossings(xc)), ", <x> ",
    @sprintf("%.4f", mean(xc)), " within ", @sprintf("%.3f", 5 * sig_pt))

# ---------------------------------------------------------------------------
# 4. Distributional test against the quadrature CDF, with the right sample size.
# ---------------------------------------------------------------------------
println("\n[4] Empirical CDF against doublewell_cdf_exact, and the n-vs-ESS trap")
probes = collect(range(-6.0, 6.0; length = 121))
Fex = SP.doublewell_cdf_exact(dw, 1.0, probes)

function cdf_interp(x::Float64)
    t = (x + 6.0) / 0.1
    i = clamp(floor(Int, t) + 1, 1, length(probes) - 1)
    u = clamp(t - (i - 1), 0.0, 1.0)
    return Fex[i] + (Fex[i + 1] - Fex[i]) * u
end

Dplain = SP.ks_distance(xplain, cdf_interp)
Dpt = SP.ks_distance(xc, cdf_interp)
critn = SP.ks_critical(length(xc), 0.05)
critess = SP.ks_critical(max(2, round(Int, essc)), 0.05)

const NBLK = 20

"lag-1 correlation of x, and a block check of whether 'independent draws' is fair."
function trapped_correlation(x)
    n = length(x)
    m = mean(x)
    v = sum(xi -> (xi - m)^2, x) / n
    r1 = sum((x[i] - m) * (x[i+1] - m) for i in 1:(n-1)) / ((n - 1) * v)
    blk = n ÷ NBLK
    bm = [mean(x[(k-1)*blk+1:k*blk]) for k in 1:NBLK]
    tau_blk = var(bm) / (v / blk)        # 1.0 if the draws really are independent
    return r1, tau_blk, n / tau_blk
end

rho1trap, tauxblk, essblk = trapped_correlation(xplain)
# what the anti-correlation should be, from the small-oscillation frequency alone
hfdd = 1e-4
uppmin = (Ud([-2.0 + hfdd]) - 2.0 * Ud([-2.0]) + Ud([-2.0 - hfdd])) / (hfdd * hfdd)
omega_min = sqrt(uppmin / mass[1])
phase = omega_min * Lstep * dt              # continuous rotation over one trajectory
rho1_pred = cos(phase)
# leapfrog's own dispersion: cos(Omega*dt) = 1 - omega^2 dt^2 / 2
rho1_disc = cos(2.0 * asin(min(1.0, omega_min * dt / 2.0)) * Lstep)

"Lag-1 correlation computed separately within each of nb equal blocks of x."
function rho1_by_block(x, nb::Int)
    n = length(x)
    m = mean(x)
    v = sum(xi -> (xi - m)^2, x) / n
    blk = n ÷ nb
    return [sum((x[i] - m) * (x[i+1] - m) for i in (k-1)*blk+1:k*blk-1) /
        ((blk - 1) * v) for k in 1:nb]
end

rho1_blocks = rho1_by_block(xplain, 4)
rho1_scatter = maximum(rho1_blocks) - minimum(rho1_blocks)
secblk = sqrt(2.0 / (NBLK - 1))          # relative standard error of a 20-sample variance
@printf("   trapped chain: D = %.5f\n", Dplain)
@printf(
    "     the iid DKW bound 2exp(-2nD^2) on a deviation that large is 10^%.0f.\n",
    log10(2.0) - 2.0 * length(xplain) * Dplain^2 / log(10.0))
println("     That bound is entitled to 40000 INDEPENDENT draws. Rather than assume it")
println("     has them, check:")
@printf(
    "       rho_1        = %+.4f\n",
    rho1trap)
@printf(
    "       tau (blocks) = %.2f  (+- %.2f from the estimator's own scatter)\n",
    tauxblk, tauxblk * secblk)
@printf("       ESS (blocks) = %.0f  of %d draws\n", essblk, length(xplain))
@printf("       tau (IPS)    = %.1f  ->  ESS = %.0f\n", taup, length(xplain) / taup)
println("     The blocking estimate is BELOW one, which is not a physical statement: with")
println("     only 20 blocks, a variance carries a relative standard error of")
@printf("     sqrt(2/19) = %.2f, so tau = %.2f could be anything from about %.1f to %.1f\n",
    secblk, tauxblk, tauxblk * (1 - secblk), tauxblk * (1 + secblk))
println("     and the 'ESS larger than n' it implies is a fact about the estimator. The")
println("     sign of rho_1, though, is reproducible - and it has a reason:")
@printf("       U''(-2) by finite differences = %.3f, so omega = sqrt(U''/m) = %.4f\n",
    uppmin, omega_min)
@printf("       one trajectory advances the phase by omega * L * dt = %.3f rad\n",
    omega_min * Lstep * dt)
@printf("       an exactly harmonic target would then give rho_1 = cos(that) = %.4f\n",
    rho1_pred)
@printf("       with leapfrog's own dispersion relation, rho_1 = %.4f\n", rho1_disc)
@printf("       measured here: rho_1 = %.4f\n", rho1trap)
println("     So the anti-correlation is expected. The samples are separated by most of a")
println("     small-oscillation period, and a rotation of more than a quarter and less")
println("     than three quarters of a period correlates negatively. The measured value")
println("     is the most negative of the three, and this file does NOT explain that")
println("     residual. It does size the estimator's own scatter, so that the residual can")
println("     be compared with something:")
qt = join(map(v -> @sprintf("%+.3f", v), rho1_blocks), " ")
@printf("       rho_1 by quarter of the chain = %s   (spread %.3f)\n", qt, rho1_scatter)
println("     The spread across quarters of one chain is of the same order as the gap")
println("     between measurement and rotation estimate, so no attribution of the gap is")
println("     on offer here. The candidates are the well's anharmonicity at this")
println("     amplitude and the correlations Metropolis rejections introduce; separating")
println("     them is not done. What is claimed is only that the sign is expected.")
println("     None of that softens the conclusion, which is the uncomfortable part. The")
println("     trapped chain's draws ARE independent, the iid bound IS entitled to them,")
println("     its error bar is as tight as a run could ask for, and the answer is still")
println("     0.5 in CDF distance from correct. An error bar is not an accuracy, and no")
println("     amount of precision detects a bias.")
@printf("   PT cold chain: D = %.5f\n", Dpt)
@printf("   the KS 5%% critical value with n = %d is %.5f, which REJECTS the PT chain on\n", length(xc), critn)
@printf("   a run length it cannot support; with n = ESS = %.0f the value is %.5f instead.\n", essc, critess)
println("   Replacing the effective size by the step count is the commonest way this test")
println("   is misapplied. Note the converse trap in the first row: the trapped chain has")
println("   ESS ~ n, so its statistics look precise. Precision and accuracy are unrelated")
println("   here, and a tight wrong answer is exactly what a converged-looking run")
println("   delivers. Putting ESS into an iid critical value is itself an engineering")
println("   approximation, not an exact test.")
checkgate("trapped chain fails the CDF test", Dplain > 0.25,
    "D = ", @sprintf("%.4f", Dplain), " vs critical ", @sprintf("%.4f", critn))
checkgate("the trapped chain is uncorrelated AND wrong",
    Dplain > 0.25 && tauxblk < 1.5 && rho1trap < 0.0,
    "blocking tau ", @sprintf("%.2f", tauxblk), ", rho_1 ", @sprintf("%.3f", rho1trap),
    ", D ", @sprintf("%.4f", Dplain))
checkgate("the anti-correlation is expected, not predicted",
    rho1_pred < 0.0 && rho1trap < 0.0 && abs(rho1trap - rho1_pred) < 0.10,
    "measured ", @sprintf("%.3f", rho1trap), " vs rotation ", @sprintf("%.3f", rho1_pred),
    ", vs leapfrog dispersion ", @sprintf("%.3f", rho1_disc))
checkgate("PT chain matches the CDF", Dpt < critess,
    "D = ", @sprintf("%.5f", Dpt), " < ", @sprintf("%.5f", critess))

# ---------------------------------------------------------------------------
# 5. How a ladder should be chosen: measured, not asserted.
# ---------------------------------------------------------------------------
println("\n[5] Ladder spacing: swap acceptance vs what the cold replica actually does")
function ladder(R::Int; nsteps = 30_000)
    bs = [(0.05)^((i - 1) / (R - 1)) for i in 1:R]
    p = SP.replica_exchange([-2.0], Ud, Gd, mass, bs, dt, Lstep, nsteps, 10, MersenneTwister(5))
    y = vec(p.samples[1])
    t = SP.integrated_autocorrelation(y)
    sa = SP.swap_acceptance(p)
    return (R = R, ratio = (0.05)^(1 / (R - 1)), amin = minimum(sa), amax = maximum(sa),
        cross = SP.barrier_crossings(y), ess = length(y) / t, mn = mean(y))
end

ladrows = [ladder(R) for R in (3, 5, 9, 13)]
@printf(
    "   %-4s %-12s %-17s %-15s %-10s %s\n",
    "R", "beta ratio", "swap acceptance", "cold crossings", "cold ESS", "cold <x>")
for L5 in ladrows
    accrng = string(round(L5.amin, digits = 2), "-", round(L5.amax, digits = 2))
    @printf(
        "   %-4d %-12.3f %-17s %-15d %-10.0f %+.4f\n",
        L5.R, L5.ratio, accrng, L5.cross, L5.ess, L5.mn)
end
cross3 = ladrows[1].cross
min_dense = minimum(L5 -> L5.cross, ladrows[2:end])
println("   Per-pair acceptance RISES as the ladder gets denser, which makes it the easy")
println("   number to watch, but it is the wrong target: what matters is whether the cold")
println("   replica crosses the barrier. A coarse ladder can keep a respectable")
println("   acceptance while the cold replica still visits the other well rarely.")
println("   Nothing here recommends a number of replicas for any other problem; it says")
println("   which quantity to measure, and how much of the difference is noise:")
"Cold-replica ESS and crossing count for ladder `R` at three seeds."
function ladder_spread(R::Int, seeds)
    es = Float64[]
    cr = Int[]
    for sd in seeds
        bs = [(0.05)^((i - 1) / (R - 1)) for i in 1:R]
        p = SP.replica_exchange([-2.0], Ud, Gd, mass, bs, dt, Lstep, 30_000, 10,
            MersenneTwister(sd))
        y = vec(p.samples[1])
        push!(es, length(y) / SP.integrated_autocorrelation(y))
        push!(cr, SP.barrier_crossings(y))
    end
    return es, cr
end

@printf("   %-8s %-32s %s\n", "ladder", "cold ESS over 3 seeds", "cold crossings over 3 seeds")
spread = Tuple{Int, Float64, Float64}[]
for R in (5, 9, 13)
    es, cr = ladder_spread(R, (5, 6, 7))
    pess = (maximum(es) - minimum(es)) / mean(es)
    pcross = (maximum(cr) - minimum(cr)) / mean(cr)
    push!(spread, (R, pess, pcross))
    @printf("   R = %-5d %-32s %s\n", R,
        join(map(v -> @sprintf("%.0f", v), es), "  "), join(map(string, cr), "  "))
end
println("   Peak-to-peak spread within a fixed ladder, over three seeds:")
for sp in spread
    @printf(
        "     R = %-3d  ESS %4.1f%%   crossings %4.1f%%\n",
        sp[1], 100 * sp[2], 100 * sp[3])
end
worst = maximum(sp -> sp[2], spread)
@printf(
    "   The ESS of a single run wobbles by up to %.0f%% peak-to-peak on a change of\n",
    100 * worst)
println("   seed alone, so the ESS difference between R = 9 and R = 13 in the table above")
println("   is not a real ordering. The crossing count does separate the coarse ladder")
@printf(
    "   (R = %d, %d crossings) from the denser ones (%d and up), and it does so\n",
    ladrows[1].R, cross3, min_dense)
println("   consistently across seeds. Use the quantity that survives a change of seed,")
println("   and quote a spread when you use the one that does not.")
checkgate("ladder ranking is seed-stable",
    worst > 0.10 && all(sp -> sp[3] < 0.10, spread) && cross3 < min_dense,
    "single-run ESS spread up to ", @sprintf("%.0f%%", 100 * worst),
    ", crossing spread all < 10%, R = 3 at ", @sprintf("%d", cross3),
    " vs ", @sprintf("%d", min_dense), " for the denser ladders")

# ---------------------------------------------------------------------------
# 6. What is and is not shown
# ---------------------------------------------------------------------------
println("\n[6] Reading this honestly")
println("   shown, by measurement:")
println("    - leapfrog's momentum-negated map is an involution to machine precision,")
println("      which is what makes the Metropolis ratio exact rather than approximate;")
println("    - HMC reproduces a Gaussian whose variance is known exactly, at step sizes")
println("      whose chains mix; the lag-1 correlation of every row is the leapfrog map's")
println("      own phase advance, predicted rather than fitted, and the residual against")
println("      that prediction sits at the lowest-acceptance rows, where Metropolis")
println("      rejections are the only thing the deterministic formula omits;")
println("    - at a resonant step size the reported precision, not the answer, is what")
println("      breaks: the naive error bar is wrong by 12x, the effective-sample-size")
println("      correction cannot see it because it truncates an alternating correlation,")
println("      and six seeds show the single run's 15-sigma reading was an error-bar")
println("      artifact rather than a bias - no bias is established here;")
println("    - the same correction, at a step size whose correlations decay monotonically,")
println("      is right to a few percent: it is a tool with a stated range, not a charm;")
println("    - the double-well runs use exactly the dt and L that passed the Gaussian")
println("      test above, so the trapped-chain bias is a sampling failure, not an")
println("      integrator failure;")
println("    - one replica cannot cross a 16 kT barrier in 4e4 steps, and is then biased")
println("      in <x> by ~2 while looking fully converged;")
println("    - replica exchange removes that bias: the cold replica's <x> moves to ~0 and")
println("      its empirical CDF agrees with quadrature within the ESS-corrected")
println("      tolerance;")
println("    - KS tests and error bars must use the effective sample size, and a trapped")
println("      chain's ESS is large, so ESS measures noise, not bias;")
println("    - the trapped chain is genuinely uncorrelated within its own well: rho_1 is")
println("      negative and the blocking estimate of tau is below one, so its tight")
println("      error bar is HONEST and still useless. Precision and accuracy are separate")
println("      properties, and only a known answer separates them;")
println("    - the sign of that anti-correlation follows from the small-oscillation phase")
println("      advance over one trajectory; its magnitude is NOT attributed here;")
println("    - which of those diagnostics survives a change of seed: the crossing count")
println("      separates ladder densities, a single run's ESS does not;")
println("    - and the only error bar in this file that is MEASURED rather than asserted")
println("      is the one obtained by repeating a run on new seeds. Everything else is a")
println("      single chain's own account of itself, which is exactly what the two")
println("      sections above show can be wrong in both directions.")
println("   not shown, and not implied:")
println("    - no molecule, no solvent, no high dimension. 1-D targets are used precisely")
println("      because their answers are known; agreement here is not evidence about a")
println("      protein free-energy landscape;")
println("    - no rates: crossing counts here measure mixing, not kinetics, and one HMC")
println("      step of L = 10 is not a unit of time;")
println("    - no claim that 9 replicas, or any ladder, is right for another system;")
println("    - the ESS-corrected KS critical value is a heuristic, not an exact test;")
println("    - finite runs cannot prove convergence. 'The numbers stopped moving' is")
println("      precisely the failure mode this file is built to show is undetectable from")
println("      inside a single run.")

println()
println("=" ^ 78)
@printf(" RESULT: %s   %d gate(s) failed   wall time = %.2f s\n", nfail == 0 ? "PASS" : "FAIL", nfail, wall())
println("=" ^ 78)
exit(nfail == 0 ? 0 : 1)
