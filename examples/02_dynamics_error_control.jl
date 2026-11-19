# Example 02 -- dynamics with explicit error control.
#
# A two-particle harmonic oscillator (HarmonicBond with r0 = 0) integrated with
# velocity Verlet, then interrogated with the three diagnostics this core
# actually provides:
#
#   1. energy_drift_report -- bounded oscillation vs secular leak, as a statistic
#   2. time reversibility  -- forward N steps, negate p, backward N steps
#   3. select_dt           -- OFFLINE validation over a ladder of FIXED steps
#
# plus one deliberately naive state-dependent-dt control experiment, measured
# rather than argued about.
#
# Run:
#   cd julia
#   julia --project=. ../examples/02_dynamics_error_control.jl
#
# ---------------------------------------------------------------------------
# WHY THERE IS NO ADAPTIVE TIME STEP (the design decision, stated up front)
# ---------------------------------------------------------------------------
# Velocity Verlet is worth using for exactly two properties:
#
#   * symplecticity: it conserves a *shadow* Hamiltonian H' = H + O(dt^2), so the
#     energy error stays inside a bounded band instead of accumulating;
#   * time reversibility: the map is an involution up to p -> -p, which is what
#     makes the long-time error oscillatory rather than one-directional, and what
#     makes HMC's Metropolis correction exact.
#
# Both arguments are per-map arguments. A step size that depends on the state,
# dt = dt(q, p), is a *different map at every step*: the backward leg then uses a
# different dt than the forward leg did, so the composition is no longer the
# identity, and the modified-equation argument that produces the shadow
# Hamiltonian no longer applies either. Every individual step can look locally
# accurate -- that is exactly what an adaptive controller measures -- while the
# trajectory as a whole leaks energy. `Dynamics` therefore refuses the design:
# `select_dt` runs short trial trajectories at a LADDER OF FIXED dt values,
# offline, and returns a single constant, so the production integrator stays
# symplectic and reversible. Rigorous adaptive symplectic schemes do exist in the
# literature (time-transformed / regularised reversible integrators); they are a
# different algorithm and are NOT implemented here. Section 5 below measures the
# failure instead of asking you to trust this paragraph.
# ---------------------------------------------------------------------------

try
    using FoundationCore
catch
    println(stderr, "FoundationCore is not on the load path. Run this example as:")
    println(stderr, "  cd julia && julia --project=. ../examples/02_dynamics_error_control.jl")
    exit(2)
end

using FoundationCore.Types
using FoundationCore.Factors
using FoundationCore.Dynamics
using Random
using Printf
using LinearAlgebra
using Statistics

const T  = FoundationCore.Types
const FA = FoundationCore.Factors
const DY = FoundationCore.Dynamics

const BAR = "=" ^ 78

# ---------------------------------------------------------------------------
# the system: U = (k/2)|q1 - q2|^2, centre of mass at rest
# ---------------------------------------------------------------------------
# In the relative coordinate x = q2 - q1 this is an oscillator with effective mass
# m/2 and stiffness k, so omega = sqrt(2k/m), and p = m*qdot (no constraints, so
# momenta are just masses times velocities). That closed form is the independent
# reference used below; the drift and reversibility verdicts do not depend on it.
function oscillator(k::Float64, m::Float64, x0::Float64, vrel::Float64)
    q = zeros(3, 2)
    q[1, 1] = -x0 / 2
    q[1, 2] = x0 / 2
    p = zeros(3, 2)
    p[1, 1] = -m * vrel / 2          # qdot1 = -xdot/2 for a stationary centre of mass
    p[1, 2] =  m * vrel / 2
    fac = FA.HarmonicBond(1, 2, k, 0.0)
    ctx = FA.Context(T.REDUCED, 2)
    return DY.State(q, p), DY.HarmonicSystem([fac], [m, m], ctx)
end

const k    = 1.0
const m    = 1.0
const x0   = 1.0
const vrel = -0.42
const omega = sqrt(2 * k / m)
const E_exact = 0.5 * (m / 2) * vrel^2 + 0.5 * k * x0^2
const dt_prod = 0.01
const NSTEPS  = 50_000
const REC     = 10

t_start = time()

println(BAR)
println("EXAMPLE 02   dynamics with explicit error control")
println(@sprintf("  system   : 2 particles, HarmonicBond k=%.1f r0=0, m=%.1f -> omega = sqrt(2k/m) = %.6f", k, m, omega))
println(@sprintf("             E_exact = %.10f  (relative mode: effective mass m/2, stiffness k)", E_exact))
println(@sprintf(
    "  run      : velocity Verlet, FIXED dt = %.2f, %d steps -> T = %.0f (%.0f periods)",
    dt_prod, NSTEPS, dt_prod * NSTEPS, dt_prod * NSTEPS * omega / (2pi)))
println("  checks   : energy-drift statistic | time reversibility | offline dt ladder")
println("             | one measured naive-adaptive control run")
println("  status   : diagnostics on a 1-degree-of-freedom integrable fixture. Not a")
println("             many-body MD run, and these tolerances do not transfer to a real")
println("             force field without repeating the whole procedure.")
println(BAR)

# ---------------------------------------------------------------------------
# 1. the drift report
# ---------------------------------------------------------------------------
# A symplectic integrator does not conserve energy; it conserves a shadow
# Hamiltonian. "Is the energy error bounded, or is it leaking?" is therefore a
# question about a statistic, not about a threshold on the last value:
#   predicted_drift = |slope| * T_run,  band = 2 * max|E - E0|,
#   drift_fraction  = predicted_drift / band.
# A bounded oscillation scores ~0 (its fitted slope is ~0); a pure linear leak
# scores ~2, because for a leak the band IS the leak. Decision threshold: 0.5.

st, sys = oscillator(k, m, x0, vrel)
E_initial = DY.total_energy(sys, st.q, st.p)
recs, es = DY.run_trajectory(st, sys, dt_prod, NSTEPS; record_every = REC, compensated = true)
rep = DY.energy_drift_report(recs, es)

println("")
println("  1. ENERGY DRIFT (Neumaier-compensated series, from run_trajectory)")
println(@sprintf(
    "     samples: %d recorded (every %d of %d steps), t spans %.2f .. %.2f",
    rep.n, REC, NSTEPS, recs[1].t, recs[end].t))
println(@sprintf("     E_initial (before any step)  = %+.14f", E_initial))
println(@sprintf(
    "     E0 = first recorded sample   = %+.14f   (|E0 - E_initial| = %.3e: the band already at t=%.2f)",
    rep.E0, abs(rep.E0 - E_initial), recs[1].t))
println(@sprintf("     Efinal                       = %+.14f", rep.Efinal))
println(@sprintf("     max_abs_dev                  = %.6e   (O(dt^2) shadow-Hamiltonian band)", rep.max_abs_dev))
println(@sprintf("     band = 2 * max_abs_dev       = %.6e", rep.band))
println(@sprintf("     slope (OLS of E on t)        = %+.6e", rep.slope))
println(@sprintf("     slope_stderr                 = %.6e", rep.slope_stderr))
println(@sprintf("     ratio = slope/stderr         = %+.6e   (reported, but NOT the verdict)", rep.ratio))
println(@sprintf("     predicted_drift = |slope|*T  = %.6e", rep.predicted_drift))
println(@sprintf("     drift_fraction               = %.6f   (threshold %.1f)", rep.drift_fraction, DY.DRIFT_FRACTION_MAX))
println(@sprintf("     bounded                      = %s", rep.bounded))
println(@sprintf("     |E_final - E0| = %.3e is one phase sample inside the band;", abs(rep.Efinal - rep.E0)))
println("     the drift measure is the slope extrapolated over the run length, compared")
println("     with the band.")
println("     why NOT slope/stderr: a symplectic run is a DETERMINISTIC oscillation, so")
println("     the residual sum of squares is pure roundoff, stderr is microscopic, and a")
println("     vanishing slope divided by a vanishing error still looks significant.")
half = length(recs) ÷ 2
r1 = DY.energy_drift_report(recs[1:half], DY.EnergySeries(es.naive[1:half], es.compensated[1:half]))
r2 = DY.energy_drift_report(recs[half+1:end], DY.EnergySeries(es.naive[half+1:end], es.compensated[half+1:end]))
println(@sprintf(
    "     split-half check: first drift_fraction = %.5f, second = %.5f, both bounded = %s",
    r1.drift_fraction, r2.drift_fraction, (r1.bounded && r2.bounded)))
println("     (a genuine leak would show up in one half; a phase artifact usually does not)")

# --- calibrating the detector on synthetic series (what the statistic can and
# --- cannot see). Measured, not asserted: the threshold is 0.5 and a PURE linear
# --- leak lands exactly on 0.5 by construction, because for a leak the excursion
# --- band IS the leak (predicted = c*T, band = 2*c*T).
tt = collect(0.0:0.01:500.0)
E_osc  = 1.0 .+ rep.max_abs_dev .* sin.(2pi .* tt)          # bounded oscillation
E_pure = 1.0 .+ rep.max_abs_dev ./ 500 .* tt                # pure leak, same end-to-end excursion
E_mixed = 1.0 .+ rep.max_abs_dev .* sin.(2pi .* tt) .+ 3 * rep.max_abs_dev ./ 500 .* tt
d_osc  = DY.energy_drift_report(tt, E_osc)
d_pure = DY.energy_drift_report(tt, E_pure)
d_mix  = DY.energy_drift_report(tt, E_mixed)
println("     detector calibration on synthetic series (same length and band as the run):")
println(@sprintf(
    "       bounded oscillation  : drift_fraction = %.6f  bounded = %s  (correct: no leak)",
    d_osc.drift_fraction, d_osc.bounded))
println(@sprintf("       pure linear leak     : drift_fraction = %.15f  bounded = %s", d_pure.drift_fraction, d_pure.bounded))
println(@sprintf(
    "       leak + oscillation   : drift_fraction = %.6f  bounded = %s  (FALSE NEGATIVE, shown on purpose)",
    d_mix.drift_fraction, d_mix.bounded))
println("     read this honestly: a pure leak scores exactly 0.5 (predicted = c*T and the")
println("     band = 2*c*T are the same quantity), so the flag there rests on the last")
println("     roundoff bit; and a leak sharing the run with an oscillation of comparable")
println("     size lands UNDER the threshold and is not flagged at all. What the")
println("     statistic separates is 'the fitted slope explains the excursion' from 'the")
println("     fitted slope is zero'. It is a diagnostic, not a certificate.")

# ---------------------------------------------------------------------------
# 2. time reversibility
# ---------------------------------------------------------------------------
st2, sys2 = oscillator(k, m, x0, vrel)
q0r = copy(st2.q); p0r = copy(st2.p)
for _ in 1:NSTEPS
    DY.velocity_verlet_step!(st2, sys2, dt_prod)
end
farthest = maximum(abs, st2.q .- q0r)
st2.p .= -st2.p
for _ in 1:NSTEPS
    DY.velocity_verlet_step!(st2, sys2, dt_prod)
end
st2.p .= -st2.p
scaleq = max(maximum(abs, q0r), 1e-30)
scalep = max(maximum(abs, p0r), 1e-30)
dq_rev = maximum(abs, st2.q .- q0r) / scaleq
dp_rev = maximum(abs, st2.p .- p0r) / scalep

println("")
println(@sprintf("  2. TIME REVERSIBILITY (%d steps forward at dt=%.2f, negate p, %d steps back)", NSTEPS, dt_prod, NSTEPS))
println(@sprintf("     farthest excursion from the start: %.6f (so the round trip is not trivial)", farthest))
println(@sprintf("     rel|dq| after the round trip = %.6e", dq_rev))
println(@sprintf("     rel|dp| after the round trip = %.6e", dp_rev))
println(@sprintf(
    "     machine eps = %.3e: the residual is accumulated roundoff over %d steps, not truncation.",
    eps(Float64), 2 * NSTEPS))
println("     A reversible map is exactly invertible in exact arithmetic; anything much")
println("     above eps is a bug in the step (this is the test that catches a Verlet")
println("     implementation that silently became a first-order kick-drift scheme).")

# ---------------------------------------------------------------------------
# 3. global accuracy against the closed form
# ---------------------------------------------------------------------------
A_amp = sqrt(2 * E_exact / k)
phi0  = atan(-vrel / (A_amp * omega), x0 / A_amp)
Tend  = 50.0
dts   = [0.02, 0.01, 0.005, 0.0025]
errs  = Float64[]
println("")
println(@sprintf("  3. GLOBAL POSITION ERROR vs the analytic x(t) = A cos(omega t + phi), A = %.6f, phi = %.6f", A_amp, phi0))
for dtx in dts
    stx, sysx = oscillator(k, m, x0, vrel)
    n = round(Int, Tend / dtx)
    for _ in 1:n
        DY.velocity_verlet_step!(stx, sysx, dtx)
    end
    x_exact = A_amp * cos(omega * n * dtx + phi0)
    err = abs((stx.q[1, 2] - stx.q[1, 1]) - x_exact)
    push!(errs, err)
    println(@sprintf("     dt = %-7.4f  n = %-6d  |x_num - x_exact| = %.6e", dtx, n, err))
end
lx = log.(dts); ly = log.(errs)
pfit = (length(lx) * sum(lx .* ly) - sum(lx) * sum(ly)) / (length(lx) * sum(abs2, lx) - sum(lx)^2)
println(@sprintf("     fitted convergence order = %.3f  (a second-order scheme: expect ~2)", pfit))
println(@sprintf(
    "     contrast with section 1: position error %.3e at dt=%.2f over T=%.0f, whereas the energy",
    errs[1], dts[1], Tend))
println(@sprintf("     band is only %.3e. Bounded energy does not mean an accurate trajectory: it means the", rep.max_abs_dev))
println("     error does not grow secularly. Both numbers are needed, and they answer")
println("     different questions.")

# ---------------------------------------------------------------------------
# 4. offline dt selection over a ladder of fixed steps
# ---------------------------------------------------------------------------
st3, sys3 = oscillator(k, m, x0, vrel)
ladder = [0.08, 0.05, 0.02, 0.01, 0.005, 0.0025, 0.001]
sel = DY.select_dt(st3, sys3, ladder; nsteps = 3000, max_defect = 1e-5)
sel2 = DY.select_dt(st3, sys3, ladder; nsteps = 3000, max_defect = 1e-5)

println("")
println("  4. select_dt -- OFFLINE validation over a ladder of fixed steps")
println("     criteria: step-doubling defect <= 1e-5  AND  drift_fraction <= 0.5")
println("       dt         defect      drift_fraction   bounded   passes")
for row in sel.table
    ok = row.defect <= 1e-5 && row.bounded
    println(@sprintf("       %-10.4f %-11.4e %-16.6f %-9s %s", row.dt, row.defect, row.drift_fraction, row.bounded, ok))
end
println(@sprintf("     selected dt = %.4f  (the largest passing both)   accepted = %s", sel.dt, sel.accepted))
println(@sprintf("     re-running the selection gives the same dt: %s (no RNG involved)", sel.dt == sel2.dt))
println("     the defect is a LOCAL heuristic estimate (one step of dt versus two of")
println("     dt/2): it is not a certified bound on the global error and it is not a")
println("     Lyapunov exponent. What select_dt returns is a CONSTANT, which is the")
println("     whole point -- the production integrator then never changes the map.")

# ---------------------------------------------------------------------------
# 5. control experiment: what a naive adaptive dt actually does
# ---------------------------------------------------------------------------
"""
Velocity Verlet with dt recomputed from the CURRENT state at every step: the
tempting "shrink the step where the local error is large" controller. The
controller variable is the step-doubling defect (the standard local error proxy),
so this is a faithful implementation of the naive idea rather than a strawman.
"""
function verlet_adaptive!(st::DY.State, s::DY.HarmonicSystem, dt0::Float64, tol::Float64, floor::Float64)
    d = DY.step_doubling_defect(st, s, dt0)
    dt = max(floor, min(dt0, dt0 * sqrt(tol / max(d, eps(Float64)))))
    DY.velocity_verlet_step!(st, s, dt)          # <-- state-dependent step
    return dt, d
end

println("")
println("  5. CONTROL EXPERIMENT: the naive adaptive controller, measured")
nad_n = 20_000
NAD_TOL = 1e-8            # tight enough that the controller genuinely has to intervene
Es, dt_used, df_used = Float64[], Float64[], Float64[]
sta, sysa = oscillator(k, m, x0, vrel)
for step in 1:nad_n
    dti, di = verlet_adaptive!(sta, sysa, dt_prod, NAD_TOL, 5e-4)
    push!(dt_used, dti); push!(df_used, di)
    if step % 10 == 0
        push!(Es, DY.total_energy(sysa, sta.q, sta.p))
    end
end
t_all = cumsum(dt_used)                      # the adaptive run's own time axis
ts = t_all[10:10:nad_n]
rep_a = DY.energy_drift_report(ts, Es)
n_shrunk = count(d -> d < dt_prod - 1e-15, dt_used)
def_fixed = DY.step_doubling_defect(sta, sysa, dt_prod)
println(@sprintf(
    "     dt actually used: %.6f .. %.6f (mean %.6f) over T = %.2f",
    minimum(dt_used), maximum(dt_used), mean(dt_used), t_all[end]))
println(@sprintf("     steps in which the controller shrank dt: %d of %d", n_shrunk, nad_n))
println(@sprintf(
    "     defect PROBE at dt0 (what the controller scales from): %.3e .. %.3e; dt0 probe target %.1e",
    minimum(df_used), maximum(df_used), NAD_TOL))
ach = df_used .* (dt_used ./ dt_prod).^3
println(@sprintf(
    "     defect actually achieved at the dt used: %.3e .. %.3e (<= target: %s)",
    minimum(ach), maximum(ach), maximum(ach) <= NAD_TOL))
println(@sprintf("     the same state at fixed dt = %.2f has probe defect %.3e, so the adaptive run really", dt_prod, def_fixed))
println("     is being asked to do something the fixed run is not: hold a smaller local error")
println(@sprintf(
    "     energy over its own time axis: max|dE| = %.6e   slope = %+.6e   drift_fraction = %.6f   bounded = %s",
    rep_a.max_abs_dev, rep_a.slope, rep_a.drift_fraction, rep_a.bounded))
println(@sprintf(
    "     for scale: the fixed-dt production run in section 1 had max|dE| = %.6e at dt = %.2f.",
    rep.max_abs_dev, dt_prod))
println(@sprintf(
    "     drift_fraction: adaptive %.5f versus fixed-dt %.5f. The adaptive band is smaller,",
    rep_a.drift_fraction, rep.drift_fraction))
println("     but its fitted slope is not -- that is a leak-shaped component, even though this")
println("     integrable fixture is too short and too regular for the effect to be decisive.")

# the same round trip, same step count, fixed dt: the reference for the comparison
stc, sysc = oscillator(k, m, x0, vrel)
q0c = copy(stc.q)
for _ in 1:nad_n
    DY.velocity_verlet_step!(stc, sysc, dt_prod)
end
stc.p .= -stc.p
for _ in 1:nad_n
    DY.velocity_verlet_step!(stc, sysc, dt_prod)
end
stc.p .= -stc.p
dq_ref = maximum(abs, stc.q .- q0c) / scaleq

stb, sysb = oscillator(k, m, x0, vrel)
for _ in 1:nad_n
    verlet_adaptive!(stb, sysb, dt_prod, NAD_TOL, 5e-4)
end
stb.p .= -stb.p
for _ in 1:nad_n
    verlet_adaptive!(stb, sysb, dt_prod, NAD_TOL, 5e-4)
end
stb.p .= -stb.p
dq_ad = maximum(abs, stb.q .- q0c) / scaleq
println(@sprintf("     round trip, %d steps out and %d back, identical step count:", nad_n, nad_n))
println(@sprintf("       fixed dt  : rel|dq| = %.6e", dq_ref))
println(@sprintf("       adaptive  : rel|dq| = %.6e      (%.3e x worse)", dq_ad, dq_ad / max(dq_ref, eps(Float64))))
if dq_ad > 100 * dq_ref
    println("     -> the map is not an involution any more: the return leg inverts a DIFFERENT")
    println("        step than the outbound leg used, so the round trip does not close.")
else
    println("     (this fixture did not open a >100x reversibility gap; the argument above is")
    println("      structural, and one short run is not evidence either way)")
end
println("     The controller does make every step locally accurate -- that is literally what")
println("     it measures. What it costs is reversibility, and with it the modified-")
println("     equation argument for a bounded shadow energy. That is why this core refuses")
println("     to adapt dt in production and validates a fixed one offline instead.")

# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------
if dq_rev > 1e-9
    println(@sprintf("RESULT: FAIL -- reversibility rel|dq| = %.3e exceeds 1e-9", dq_rev))
    exit(1)
end
if rep.drift_fraction > 0.5
    println(@sprintf("RESULT: FAIL -- drift_fraction = %.6f exceeds 0.5", rep.drift_fraction))
    exit(1)
end
if !rep.bounded
    println("RESULT: FAIL -- drift report says the energy is not bounded")
    exit(1)
end
if !sel.accepted
    println("RESULT: FAIL -- no dt on the ladder passed the acceptance criteria")
    exit(1)
end
if !(1.8 < pfit < 2.2)
    println(@sprintf("RESULT: FAIL -- fitted convergence order %.3f is not second order", pfit))
    exit(1)
end

println("")
println(BAR)
println(@sprintf(
    "RESULT: PASS   drift_fraction = %.5f (<= 0.5)   reversibility rel|dq| = %.3e (<= 1e-9)   selected dt = %.4f",
    rep.drift_fraction, dq_rev, sel.dt))
println(@sprintf("  fitted order %.3f (expect 2)   wall time = %.2f s", pfit, time() - t_start))
println("  NOT demonstrated here: many-body or chaotic dynamics (where the band itself")
println("  can grow and the shadow-Hamiltonian picture weakens), distance constraints")
println("  (not implemented: Dynamics.rattle_project! raises), thermostats/Langevin,")
println("  or any transfer of these tolerances to a real molecular force field.")
println(BAR)
exit(0)
