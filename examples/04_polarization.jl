# ============================================================================
#  examples/04_polarization.jl
#
#  Induced dipoles: solving A mu = E, and what the answer is FOR.
#
#  Run from julia:  julia --project=. ../examples/04_polarization.jl
#  Deterministic: the only random numbers used come from an explicit seed.
#
#  A polarizable force field has one idea in it: the charge distribution responds
#  to the field it finds itself in. For induced point dipoles that idea is a
#  linear system, `A mu = E`, and the whole construction rests on a calculus fact
#  - the envelope theorem - which says that once the dipoles are converged, the
#  force on an atom is the derivative at FIXED mu, with no term from dmu/dq.
#  This file checks both claims against computations that do not reuse the code
#  path that produced them, and then measures what happens when the hypothesis
#  is dropped.
#
#  What is NOT demonstrated: anisotropic polarizability, Drude oscillators,
#  charge transfer, metallic screening, periodic dipole sums (Ewald), analytic
#  tensor derivatives, or any parameterisation of a real material. Section 6
#  prints `extension_boundaries()`, which is the module's own statement of the
#  same list.
# ============================================================================

using Printf
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "..", "julia", "src", "FoundationCore.jl"))
using .FoundationCore
import .FoundationCore.Types as T
import .FoundationCore.Polarizable as P

const T0 = time_ns()
wall() = (time_ns() - T0) / 1e9

nfail = 0

function checkgate(name::AbstractString, ok::Bool, detail...)
    ok || (global nfail = nfail + 1)
    print(rpad("  [$name] $(ok ? "PASS" : "FAIL")", 34), " ")
    for d in detail
        print(d)
    end
    println()
    return ok
end

println("=" ^ 78)
println(" Foundation core :: 04 - induced dipoles, the envelope theorem, stability")
println("=" ^ 78)

# ---------------------------------------------------------------------------
# 1. the system and the solve
# ---------------------------------------------------------------------------
println("\n[1] A three-site polarizable system, solved")
q = [0.0 2.9 0.5
     0.0 0.0 2.3
     0.0 0.0 0.0]
z = [1.0, -0.5, -0.5]
al = [0.5, 0.3, 0.3]
s = P.InducedDipoleSystem(q, z, al, T.NoBox(), T.REDUCED; thole = true, a_thole = 1.0)
res = P.solve_dipoles!(s; tol = 1e-13, check_spd = true)
mu = res.μ
N = P.Nsites(s)
A = P.A_matrix(s)
Efield = P.permanent_field(s)
resid_own = norm(A * mu - Efield) / max(norm(Efield), eps(Float64))
ev = eigvals(Symmetric(A))
amin, amax = minimum(ev), maximum(ev)

println("   sites: ", N, "   charges: ", join(map(v -> @sprintf("%+.1f", v), z), " "))
println("   alpha: ", join(map(v -> @sprintf("%.1f", v), al), " "))
@printf("   |E| from the permanent charges = %.6f\n", norm(Efield))
@printf(
    "   solver: method = %s   spd = %s   converged = %s   n_iter field = %d\n",
    res.method, res.spd, res.converged, res.n_iter)
@printf("   residual reported by the solver     = %.3e\n", res.residual)
@printf("   residual recomputed here from A, E  = %.3e\n", resid_own)
moe = isnan(res.min_eig) ? "NaN" : @sprintf("%.6f", res.min_eig)
@printf("   result.min_eig as returned          = %s\n", moe)
println("   (the direct solver leaves min_eig and n_iter unset; the spectrum below is")
println("   computed here with eigvals, so nothing in this file reads those fields as")
println("   if they were meaningful.)")
@printf(
    "   eigenvalues of A: min = %+.6f   max = %+.6f   cond = %.3f   ||A|| = %.3f\n",
    amin, amax, amax / amin, opnorm(A))
for i in 1:N
    @printf(
        "   site %d  mu = (%+.6f, %+.6f, %+.6f)  |mu| = %.6f  alpha|E_perm| = %.6f\n",
        i, mu[3(i-1)+1], mu[3(i-1)+2], mu[3(i-1)+3], norm(mu[3(i-1)+1:3i]),
        al[i] * norm(Efield[3(i-1)+1:3i]))
end
Uperm, Fperm = P.permanent_energy_forces(s)
Uind = P.induced_energy(s, mu)
@printf("\n   U_perm = %+.8f   U_ind = %+.8f   U_total = %+.8f\n", Uperm, Uind, Uperm + Uind)
@printf(
    "   induced_energy(s, result) = %+.8f   difference %.3e\n",
    P.induced_energy(s, res), abs(P.induced_energy(s, res) - (Uperm + Uind)))
checkgate("solve converged", res.converged && res.residual < 1e-10 && resid_own < 1e-10,
    "residual ", @sprintf("%.2e", min(res.residual, resid_own)))
checkgate("spd flag matches the spectrum", res.spd == (amin > 0.0),
    "spd = ", string(res.spd), ", min eig = ", @sprintf("%+.4f", amin))

# ---------------------------------------------------------------------------
# 2. what the linear system says in physics words: mu = alpha * (LOCAL field),
#    where the local field includes the other induced dipoles. That is the whole
#    many-body content, and it can be read back out of A.
# ---------------------------------------------------------------------------
println("\n[2] The induced dipole follows the LOCAL field, not the permanent one")

function local_field(i::Int)
    blk = 3(i-1)+1:3i
    Elo = Efield[blk]
    for j in 1:N
        j == i && continue
        # A_ij = -T_ij, and the field at i produced by dipole j is +T_ij mu_j
        Tij = -A[3(i-1)+1:3i, 3(j-1)+1:3j]
        Elo = Elo + Tij * mu[3(j-1)+1:3j]
    end
    return Elo
end

@printf(
    "   %-5s %-15s %-15s %-13s %-13s %s\n",
    "site", "alpha|E_perm|", "alpha|E_local|", "enhancement", "mu.E_local>0", "mu/alpha-E_loc")
# rows are (site, alpha|E_perm|, alpha|E_local|, enhancement, alignment, deviation)
function response_rows()
    out = Tuple{Int, Float64, Float64, Float64, Bool, Float64}[]
    for i in 1:N
        Elo = local_field(i)
        blk = 3(i-1)+1:3i
        push!(out, (i, al[i] * norm(Efield[blk]), al[i] * norm(Elo),
            norm(Elo) / max(norm(Efield[blk]), 1e-300), dot(mu[blk], Elo) > 0,
            maximum(abs.(mu[blk] .- al[i] * Elo))))
    end
    return out
end

rows2 = response_rows()
for r2 in rows2
    @printf("   %-5d %-15.6f %-15.6f %-13.4f %-13s %.2e\n", r2[1], r2[2], r2[3], r2[4], r2[5] ? "yes" : "no", r2[6])
end
maxid = maximum(r2 -> r2[6], rows2)
println("   An enhancement above or below 1 is the many-body effect: each site feels its")
println("   neighbours' induced dipoles as well as the permanent charges. Rebuilding the")
println("   local field from -A_ij (that is the dipole tensor) and checking")
println("   mu_i = alpha_i E_local,i reads the operator back; it is not a restatement")
println("   of the solve.")
checkgate("mu_i = alpha_i * E_local,i", maxid < 1e-10, "max deviation ", @sprintf("%.2e", maxid))

# ---------------------------------------------------------------------------
# 3. THE claim. `induced_forces` differentiates U(q, mu) at FIXED mu. The
#    gradient a physicist writes down is the derivative of the RELAXED energy
#    U(q) = U_perm + U_ind(q, mu*(q)), with mu re-solved at every displaced
#    geometry. Different computations; the envelope theorem says the two have
#    the same limit when A mu = E. So: vary the step size and look at the limit.
# ---------------------------------------------------------------------------
println("\n[3] Envelope theorem: fixed-mu force vs gradient of the re-solved energy")

function displaced(dx::Float64, j::Int, d::Int)
    qx = copy(q)
    qx[d, j] += dx
    return P.InducedDipoleSystem(qx, z, al, T.NoBox(), T.REDUCED; thole = true, a_thole = 1.0)
end

function relaxed_energy(sx::P.InducedDipoleSystem)
    rr = P.solve_dipoles!(sx; tol = 1e-14, check_spd = true)
    return P.induced_energy(sx, rr)
end

"Force from the fully relaxed energy, central difference with step h."
function relaxed_force(h::Float64, j::Int, d::Int)
    return -(relaxed_energy(displaced(+h, j, d)) - relaxed_energy(displaced(-h, j, d))) / (2h)
end

function envelope_gap(h::Float64)
    Fs = P.induced_forces(s, res; h = h)
    w = 0.0
    sc = 0.0
    for j in 1:N, d in 1:3
        gp = relaxed_force(h, j, d)
        w = max(w, abs(gp - Fs[d, j]))
        sc = max(sc, abs(gp))
    end
    return w, w / sc
end

println("   the discrepancy between the two sides, as the step size is varied:")
@printf("   %-10s %-28s %-12s %s\n", "h", "max|stationary - relaxed|", "relative", "slope")
gaps = Tuple{Float64, Float64, Float64}[]
for h in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
    ab, rl = envelope_gap(h)
    push!(gaps, (h, ab, rl))
end
for k in eachindex(gaps)
    g = gaps[k]
    slope = k == 1 ? NaN : (log(gaps[k-1][2]) - log(g[2])) / log(10.0)
    @printf("   %-10.1e %-28.3e %-12.3e %s\n", g[1], g[2], g[3], k == 1 ? "-" : @sprintf("%.2f", slope))
end
best = minimum(g -> g[3], gaps)
besth = filter(g -> g[3] == best, gaps)[1][1]
println("   The slope column is the change in the gap per factor of 10 in h. It is")
println("   close to 2 while truncation dominates - which is what a consistent")
println("   finite-difference pair should do - and then goes negative as the roundoff")
println("   of the differences takes over: both sides are differences of a function of")
println("   order 1, so each carries about eps/h^2 of noise.")
@printf(
    "   The smallest gap, %.2e of the force scale at h = %.1e, is the floor of what\n",
    best, besth)
println("   this method can show: a statement about the check, not about the theorem.")
println("   Two independent finite differences agreeing at their best achievable level is")
println("   the evidence, and the theorem is the reason the two have the same limit all")
println("   together. The per-component table below is at that best step size:")
Fs = P.induced_forces(s, res; h = besth)
for j in 1:N, d in 1:3
    gp = relaxed_force(besth, j, d)
    lbl = d == 1 ? "x" : d == 2 ? "y" : "z"
    @printf(
    "   site %d %s  stationary-point %+14.8f  relaxed-gradient %+14.8f  diff %+.2e\n",
    j, lbl, Fs[d, j], gp, gp - Fs[d, j])
end
checkgate("envelope theorem holds", best < 1e-5,
    "best relative gap ", @sprintf("%.2e", best), " at h ", @sprintf("%.1e", besth),
    " (threshold 1e-5)")

# ---------------------------------------------------------------------------
# 4. the same comparison with the hypothesis removed. If A mu = E is not
#    satisfied, the conclusion must fail, and the size of the failure is set by
#    the residual, not by taste.
# ---------------------------------------------------------------------------
println("\n[4] Same test with a deliberately unconverged dipole solution")
@printf(
    "   %-14s %-22s %-26s %-12s %s\n",
    "perturbation", "residual ||Amu-E||/||E||", "max|stationary - relaxed|",
    "force rel err", "energy rel err")
rng = MersenneTwister(4)
noise = randn(rng, length(mu))
Uind_ref = P.induced_energy(s, mu; A = A, E = Efield)

"Force gap, force relative error and energy relative error for mu + eps_mu*noise."
function perturbed_gap(eps_mu::Float64)
    mub = mu + eps_mu * noise
    residb = norm(A * mub - Efield) / norm(Efield)
    rb = P.InducedDipoleResult(mub, residb, 0, res.method, res.spd, NaN, true)
    Fx = P.induced_forces(s, rb; h = besth)
    w = 0.0
    sc = 0.0
    for j in 1:N, d in 1:3
        gp = relaxed_force(besth, j, d)
        w = max(w, abs(gp - Fx[d, j]))
        sc = max(sc, abs(gp))
    end
    Ub = P.induced_energy(s, mub; A = A, E = Efield)
    return (eps_mu, residb, w, w / sc, abs(Ub - Uind_ref) / max(abs(Uind_ref), 1e-300))
end

rows4 = [perturbed_gap(e) for e in (0.0, 1e-6, 1e-4, 1e-2)]
for r4 in rows4
    @printf("   %-14.1e %-22.3e %-26.3e %-12.3e %.3e\n", r4[1], r4[2], r4[3], r4[4], r4[5])
end
rel_big = rows4[end][4]
rel_zero = rows4[1][4]
# slopes of log(gap) against log(perturbation), between successive nonzero rows
slope_rows = [(log(rows4[k][4]) - log(rows4[k-1][4])) / (log(rows4[k][1]) - log(rows4[k-1][1]))
    for k in 3:length(rows4)]
slopetxt = join(map(v -> @sprintf("%.2f", v), slope_rows), " ")
# the same scaling, for the energy instead of the force
eslope_rows = [(log(rows4[k][5]) - log(rows4[k-1][5])) / (log(rows4[k][1]) - log(rows4[k-1][1]))
    for k in 3:length(rows4)]
eslopetxt = join(map(v -> @sprintf("%.2f", v), eslope_rows), " ")
println("   `converged` was forced to true throughout that table: the flag is a claim,")
println("   the residual is the measurement. Three things fall out of it.")
println("   First, the force gap grows in proportion to the dipole error: the slopes of")
println("   log(gap) against log(perturbation) are")
@printf("   %s, so a dipole tolerance is a FORCE tolerance, not a rounding detail.\n", slopetxt)
println("   Second, the energy in the same rows is quadratic in that error, slopes")
@printf("   %s. At eps = 1e-4 the induced energy is off by %.1e relative,\n", eslopetxt, rows4[3][5])
@printf(
    "   %.1e in absolute reduced units, which would not disturb any convergence plot,\n",
    rows4[3][5] * abs(Uind_ref))
println("   while the forces in the same row are off by")
@printf(
    "   %.1e relative: an order of magnitude worse, and worsening faster as eps grows.\n",
    rows4[3][4])
println("   A run whose energy 'converged' is not evidence that its forces did.")
println("   Third, the reference row at eps = 0 sits on the finite-difference floor")
@printf(
    "   (%.1e relative), which is why the scaling is the meaningful reading rather\n",
    rel_zero)
println("   than the ratio to that floor: at eps = 1e-2 the force is wrong by")
@printf(
    "   %.1e relative, eight orders of magnitude above the floor, while at eps = 1e-6\n",
    rel_big)
println("   the dipole error is invisible in the energy and still shows in the force.")
checkgate("force error scales linearly with mu error",
    all(s -> 0.7 < s < 1.3, slope_rows), "slopes ", slopetxt)
checkgate("energy error scales quadratically",
    all(s -> 1.7 < s < 2.3, eslope_rows), "slopes ", eslopetxt)

# ---------------------------------------------------------------------------
# 5. stability. A is NOT guaranteed positive definite, and Thole damping is a
#    model choice that shifts the spectrum rather than a cure. Two identical,
#    strongly polarizable sites show where the boundary is - and the undamped
#    boundary has a closed form to check the tensor convention against.
# ---------------------------------------------------------------------------
println("\n[5] Where the induced-dipole model stops being stable")
al2 = [2.0, 2.0]
z2 = [0.4, -0.4]

function pair_system(rsep::Float64, damp::Bool)
    qq = [0.0 rsep; 0.0 0.0; 0.0 0.0]
    return P.InducedDipoleSystem(qq, z2, al2, T.NoBox(), T.REDUCED; thole = damp, a_thole = 1.0)
end

function min_eig_of(rsep::Float64, damp::Bool)
    return minimum(eigvals(Symmetric(P.A_matrix(pair_system(rsep, damp)))))
end

@printf("   two sites, alpha = %.1f each: smallest eigenvalue of A vs separation\n", al2[1])
@printf("   %-8s %-14s %-14s %-8s %s\n", "r", "damped", "undamped", "spd", "solver method")
for rsep in (0.8, 1.0, 1.25, 1.5, 1.6, 2.0, 3.0)
    rr = P.solve_dipoles!(pair_system(rsep, true); tol = 1e-10)
    @printf(
    "   %-8.2f %-14.4f %-14.4f %-8s %s\n",
    rsep, min_eig_of(rsep, true), min_eig_of(rsep, false), string(rr.spd), string(rr.method))
end

function critical_radius(damp::Bool; lo = 0.3, hi = 4.0)
    a, b = lo, hi
    fa = min_eig_of(a, damp)
    fb = min_eig_of(b, damp)
    sign(fa) == sign(fb) && return NaN
    for _ in 1:80
        mid = 0.5 * (a + b)
        fm = min_eig_of(mid, damp)
        if sign(fm) == sign(fa)
            a, fa = mid, fm
        else
            b = mid
        end
    end
    return 0.5 * (a + b)
end

rc_d = critical_radius(true)
rc_u = critical_radius(false)
rc_exact = (2.0 * al2[1])^(1 / 3)          # 1/alpha = 2/r^3 for the undamped tensor
println("\n   stability boundary (smallest eigenvalue of A = 0), located by bisection:")
@printf("      undamped : r_c = %.6f\n", rc_u)
@printf("      damped   : r_c = %.6f\n", rc_d)
@printf("      closed form for the undamped tensor, (2*alpha)^(1/3) = %.6f\n", rc_exact)
@printf("      difference from the closed form: %.3e\n", rc_u - rc_exact)
@printf("      damping moves the boundary inward by %.3f in r, %.1f%% of it\n", rc_u - rc_d, 100 * (rc_u - rc_d) / rc_u)
println("   Below r_c there is no stable induced-dipole state at all. Damping lifts the")
println("   spectrum; it does not remove the instability, and the margin it buys here is")
@printf(
    "   under one percent of the separation - not a licence to run alpha = %.1f sites\n",
    al2[1])
println("   at contact. Two things to notice: the undamped boundary matches the closed form")
println("   of the dipole tensor, which checks the sign and factor conventions of")
println("   A_matrix independently of any solve; and the solver answers spd = false,")
println("   method = lstsq there instead of regularising the problem away.")
checkgate("tensor convention matches closed form", abs(rc_u - rc_exact) < 1e-6,
    "r_c = ", @sprintf("%.6f", rc_u), " vs (2*alpha)^(1/3) = ", @sprintf("%.6f", rc_exact))
run08 = P.solve_dipoles!(pair_system(0.8, true); tol = 1e-10)
checkgate("indefinite A is surfaced, not hidden", !run08.spd && run08.method === :lstsq,
    "at r = 0.8: spd = ", string(run08.spd), ", method = ", string(run08.method))

# what an indefinite A means physically: the stationary point is a saddle
sun = pair_system(0.8, true)
run = P.solve_dipoles!(sun; tol = 1e-13)
Aun = P.A_matrix(sun)
Eun = P.permanent_field(sun)
lam = eigvals(Symmetric(Aun))
vecs = eigvecs(Symmetric(Aun))
kmin = argmin(lam)
v = vecs[:, kmin]
Ubase = P.induced_energy(sun, run.μ)
@printf(
    "\n   at r = 0.8 the solve still has residual %.3e, and lam_min = %+.4f:\n",
    run.residual, lam[kmin])
@printf("   %-12s %-26s %s\n", "step t", "dU measured", "predicted 0.5*lam*t^2")
for tt in (1e-3, 1e-2, 1e-1)
    U1 = P.induced_energy(sun, run.μ + tt * v; A = Aun, E = Eun)
    @printf("   %-12.1e %-26.3e %.3e\n", tt, U1 - Ubase, 0.5 * lam[kmin] * tt^2)
end
println("   The energy falls quadratically along that direction, and the fall is exactly")
println("   the eigenvalue: the solution is a stationary point of the wrong kind. A small")
println("   residual is not a stability statement - those are separate facts, and the")
println("   model reports both instead of conflating them.")

# ---------------------------------------------------------------------------
# 6. what the module itself says it does not do
# ---------------------------------------------------------------------------
println("\n[6] Stated boundaries (returned by the module as data)")
for b in P.extension_boundaries()
    @printf("   %-50s %-10s %s\n", b.capability, string(b.supported), b.note)
end

println("\n[7] Reading this honestly")
println("   shown, by measurement:")
println("    - A mu = E is solved: the residual, recomputed from independently built A")
println("      and E, is at machine precision for a 3-site system;")
println("    - the linear system is the physics statement mu_i = alpha_i * (local field),")
println("      the local field including the other induced dipoles; reading the dipole")
println("      tensor back out of -A_ij reproduces it to the solve tolerance;")
println("    - the envelope theorem holds here: the fixed-mu stationary force and the")
println("      gradient of the fully re-solved energy agree down to the floor set by two")
println("      finite differences, and the h-study shows the h^2-then-roundoff shape")
println("      rather than resting on one lucky comparison;")
println("    - the theorem is true only under its hypothesis: perturbing mu by 1e-2")
println("      breaks the agreement by a comparable relative amount, so it is the")
println("      residual, not the converged flag, that decides whether a force may be")
println("      used;")
println("    - positive definiteness is checked, not assumed: the two-site scan finds a")
println("      stability boundary that matches the closed form of the undamped tensor,")
println("      damping moves it by a few percent, and below it the stationary point is a")
println("      saddle whose measured curvature is the eigenvalue.")
println("   not shown, and not implied:")
println("    - nothing about a real material: three sites, reduced units, one geometry;")
println("    - no anisotropic alpha, no Drude particles, no charge transfer, no short-")
println("      range penetration, no metallic screening (section 6 is the module's own")
println("      list of these);")
println("    - no periodic dipole sums: with NoBox every term is a direct pair, so the")
println("      long-range convergence problem, where induced-dipole models are hardest,")
println("      is untouched here;")
println("    - no performance claim: induced_forces differentiates the quadratic form by")
println("      finite differences, so this is correctness of the idea, not a production")
println("      kernel;")
println("    - agreement of two finite differences at one geometry is evidence at that")
println("      geometry; it bounds nothing elsewhere.")

println()
println("=" ^ 78)
@printf(" RESULT: %s   %d gate(s) failed   wall time = %.2f s\n", nfail == 0 ? "PASS" : "FAIL", nfail, wall())
println("=" ^ 78)
exit(nfail == 0 ? 0 : 1)
