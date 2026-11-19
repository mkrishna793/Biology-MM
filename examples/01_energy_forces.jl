# Example 01 -- energy and forces: the vertical slice.
#
# One small Lennard-Jones configuration pushed through the whole pipeline:
#
#   typed state (AtomSet + OrthorhombicBox + System)
#     -> spatial pair list (cell list + skin, cross-checked against brute force)
#     -> factors (LennardJonesPair, mode = :lfs, rc = 2.5)
#     -> (U, F) from the single VJP accumulation point `total_energy_forces`
#     -> F verified against central finite differences of U
#
# Run:
#   cd julia
#   julia --project=. ../examples/01_energy_forces.jl
#
# Everything printed here is measured in this run. The claim being tested is the
# narrow one in `Factors`: for the frozen pair graph, the scattered force equals
# -grad(U). Nothing here claims a physical equation of state, a transport property,
# or converged thermodynamics: 50 atoms at rho ~ 0.3 with rc = 2.5 sigma is a
# unit-scale fixture, not a study of the liquid state.

try
    using FoundationCore
catch
    println(stderr, "FoundationCore is not on the load path. Run this example as:")
    println(stderr, "  cd julia && julia --project=. ../examples/01_energy_forces.jl")
    exit(2)
end

using FoundationCore.Types
using FoundationCore.Factors
using FoundationCore.Neighbors
using FoundationCore.Dynamics
using Random
using Printf
using LinearAlgebra
using Statistics

const T  = FoundationCore.Types
const FA = FoundationCore.Factors
const NB = FoundationCore.Neighbors
const DY = FoundationCore.Dynamics

const BAR = "=" ^ 78

# ---------------------------------------------------------------------------
# deterministic configuration
# ---------------------------------------------------------------------------

"""
Jittered-sublattice configuration: a deterministic, near-uniform packing.

Sites come from a `(4,4,5)` grid over the box (spacings 1.30/1.40/1.20 sigma), a
seeded random subset of `n` of them is taken, and each is displaced by at most
`jitter` in each direction. Why not plain random insertion? At rho ~ 0.3 with a
hard-core floor of 0.85 sigma the rejected-region volume is already at the random
sequential adsorption jamming limit, so insertion would either fail or produce a
configuration with near-coincident pairs; a singular pair makes the finite
difference meaningless.

The result is a deterministic fixture with a realistic first-shell distance. It is
NOT an equilibrated liquid: there is no temperature attached to it and no
equation of state should be read off these numbers.
"""
function lattice_fluid(rng, box::T.OrthorhombicBox, n::Int, dims, jitter::Float64)
    L = box.L
    nx, ny, nz = dims
    nx * ny * nz >= n || error("grid too small: need >= $n sites")
    sp = (L[1] / nx, L[2] / ny, L[3] / nz)
    idx = vec(collect(Iterators.product(1:nx, 1:ny, 1:nz)))
    sel = idx[randperm(rng, length(idx))[1:n]]   # subset WITHOUT replacement
    q = Matrix{Float64}(undef, 3, n)
    for (c, (i, j, k)) in enumerate(sel)
        q[1, c] = (i - 0.5) * sp[1] + (2 * rand(rng) - 1) * jitter
        q[2, c] = (j - 0.5) * sp[2] + (2 * rand(rng) - 1) * jitter
        q[3, c] = (k - 0.5) * sp[3] + (2 * rand(rng) - 1) * jitter
    end
    return q
end

"Exhaustive minimum-image pair list within `rc` (the independent reference)."
function brute_pairs(q, box, rc::Float64)
    n = size(q, 2)
    out = Tuple{Int,Int}[]
    r2 = rc * rc
    for i in 1:n-1
        for j in i+1:n
            d = T.unwrap_displacement(box, q, i, j)
            T.norm2(d) <= r2 && push!(out, (i, j))
        end
    end
    return sort(out)
end

"Smallest minimum-image separation present in the configuration."
function min_separation(q, box)
    n = size(q, 2)
    best = Inf
    for i in 1:n-1, j in i+1:n
        best = min(best, sqrt(T.norm2(T.unwrap_displacement(box, q, i, j))))
    end
    return best
end

"""
Relative discrepancy between the scattered force and central differences of U.

`h = 1e-6` on a double-precision radial potential balances the truncation term
O(h^2) against the subtraction roundoff O(eps/h); for this system both sit far
below the 1e-5 verdict threshold, and the measured value is printed rather than
assumed.
"""
function fd_force_error(factors, q, ctx, h::Float64 = 1e-6)
    U0, Fana = FA.total_energy_forces(factors, q, ctx)
    G = zeros(size(q))
    for j in 1:size(q, 2), d in 1:3
        qp = copy(q); qp[d, j] += h
        qm = copy(q); qm[d, j] -= h
        G[d, j] = (FA.total_energy_forces(factors, qp, ctx)[1] -
                   FA.total_energy_forces(factors, qm, ctx)[1]) / (2h)
    end
    scale = max(maximum(abs, G), 1e-12)
    return U0, maximum(abs, Fana .+ G) / scale, scale
end

# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------

const N      = 50
const RC     = 2.5
const SKIN   = 0.30
const SEED   = 20261111
const DIMS   = (4, 4, 5)
const JITTER = 0.10
const FD_TOL = 1e-5
const RMIN_SAFE = 0.70

t_start = time()

box = T.OrthorhombicBox((5.2, 5.6, 6.0))
rng = MersenneTwister(SEED)
q   = lattice_fluid(rng, box, N, DIMS, JITTER)

println(BAR)
println("EXAMPLE 01   energy and forces -- the vertical slice")
println(@sprintf(
    "  system : %d Lennard-Jones atoms, orthorhombic periodic box L=(%.1f, %.1f, %.1f),",
    N, box.L[1], box.L[2], box.L[3]))
println(@sprintf("           rc = %.1f sigma, cutoff convention :lfs, neighbour skin = %.2f", RC, SKIN))
println("  checks : pair list vs brute force | F vs -grad(U) by finite difference |")
println("           Newton's third law | compensated energy breakdown | cutoff modes")
println("  status : demonstration of the implemented pipeline. Not a production")
println("           engine, and not a thermodynamic study.")
println(BAR)

p     = zeros(3, N)              # zero momenta: this example never integrates
m     = fill(1.0, N)             # REDUCED units: sigma = epsilon = mass = 1
z     = zeros(N)                 # uncharged: no Coulomb factors in this example
atoms = T.AtomSet(q, p, m, z, fill(:Ar, N))
sys   = T.System(atoms, box, T.ChemicalTopology(); units = T.REDUCED, temperature = 0.0)

V    = T.box_volume(box)
rho  = N / V
rmin = min_separation(q, box)
sp   = (box.L[1] / DIMS[1], box.L[2] / DIMS[2], box.L[3] / DIMS[3])

println(@sprintf(
    "  state   : AtomSet(N=%d, sum m=%.1f, total charge=%.1f) in box L=(%.1f, %.1f, %.1f)",
    length(atoms.m), sum(atoms.m), sum(atoms.z), box.L[1], box.L[2], box.L[3]))
println(@sprintf(
    "            volume=%.3f  rho=%.4f sigma^-3  jittered %dx%dx%d sublattice (spacing %.2f/%.2f/%.2f sigma, jitter +/-%.2f)",
    V, rho, DIMS[1], DIMS[2], DIMS[3], sp[1], sp[2], sp[3], JITTER))
println(@sprintf("            r_min=%.3f sigma: must exceed %.2f, because a near-singular pair", rmin, RMIN_SAFE))
println("            makes the finite difference meaningless (U -> Inf at r -> 0).")
if rmin < RMIN_SAFE
    println("RESULT: FAIL -- configuration contains a near-singular pair (r_min = ", rmin, ")")
    exit(1)
end
println(@sprintf(
    "            seed=%d  rc=%.2f vs L_min/2=%.2f -> %s",
    SEED, RC, minimum(box.L) / 2, (RC <= minimum(box.L) / 2 ? "minimum image unambiguous" : "NOT guaranteed")))
println("            the AtomSet constructor is what validated this state: it rejects")
println("            non-3xN shapes, non-positive masses, and any non-finite coordinate.")

# ---------------------------------------------------------------------------
# pair list: cell list + skin, cross-checked against exhaustive enumeration
# ---------------------------------------------------------------------------

nl          = NB.build_neighbors(q, box, RC, SKIN)
pairs_nl    = sort(NB.neighbor_pairs(nl))
within_rc   = brute_pairs(q, box, RC)
within_skin = brute_pairs(q, box, RC + SKIN)
missing     = length(setdiff(within_rc, pairs_nl))
extra       = length(setdiff(pairs_nl, within_skin))
n_skin_only = length(pairs_nl) - length(within_rc)

println("")
println(@sprintf("  pairs   : cell-list search radius rc+skin = %.2f -> %d pairs in list", RC + SKIN, length(pairs_nl)))
println(@sprintf(
    "            brute force says %d pairs have r <= rc; missing = %d, beyond rc+skin = %d",
    length(within_rc), missing, extra))
println(@sprintf("            %d listed pairs sit in rc < r <= rc+skin and contribute exactly 0.0", n_skin_only))
println("            (that is the deliberate price of a skin: extra zero terms now, in")
println("             exchange for not re-running the geometric search every step)")
println(@sprintf(
    "            Verlet rebuild threshold skin/2 = %.3f sigma: needs_rebuild!(fresh) = %s",
    SKIN / 2, NB.needs_rebuild!(nl, q)))
qtest = copy(q); qtest[1, 1] += 0.6 * SKIN
println(@sprintf(
    "            one atom displaced %.3f sigma (> skin/2): needs_rebuild! = %s",
    0.6 * SKIN, NB.needs_rebuild!(nl, qtest)))
if missing != 0 || extra != 0
    println("RESULT: FAIL -- neighbour list disagrees with exhaustive enumeration")
    exit(1)
end

# ---------------------------------------------------------------------------
# factors and the (U, F) accumulation
# ---------------------------------------------------------------------------
# The pair list below is FROZEN: the finite-difference test differentiates U over
# this fixed graph, which is exactly what a neighbour-list-driven force does within
# one step. That is the honest scope of the check -- it verifies that the scattered
# force is the derivative of the energy actually being summed. For :lfs the
# potential is C^1 at rc, so a pair crossing rc under a 1e-6 perturbation cannot
# inject a jump into the difference; for :trunc it can (see CHECK 4).

factors = FA.AbstractFactor[
    FA.LennardJonesPair(i, j, 1.0, 1.0; mode = :lfs, rc = RC) for (i, j) in pairs_nl
]
ctx  = FA.Context(box, T.REDUCED, N)
hsys = DY.HarmonicSystem(sys, factors)   # the same factors wrapped as a Hamiltonian

U, F = FA.total_energy_forces(factors, q, ctx)
Uham = DY.potential_energy(hsys, q)

println("")
println(@sprintf("  energy  : U = %.12f reduced units, summed over %d :lfs pair factors", U, length(factors)))
println(@sprintf("            via HarmonicSystem/Dynamics path: U = %.12f (identical: %s)", Uham, Uham == U))
println(@sprintf("  forces  : max|F| = %.6f   rms|F| = %.6f   accumulator is 3x%d", maximum(abs, F), sqrt(mean(F .^ 2)), N))
println("            one pass: each factor returns its local energy and scatters its")
println("            own gradient -- no NxN object is ever formed.")

# --- CHECK 1: analytic force vs central finite difference of the energy --------
U0, relF, fscale = fd_force_error(factors, q, ctx, 1e-6)
println("")
println("  CHECK 1  F == -grad(U): central differences, h = 1e-6, frozen pair list")
println(@sprintf("            max|F_analytic + grad_fd| / max|grad_fd| = %.3e   (threshold %.0e)", relF, FD_TOL))
println(@sprintf("            max|grad_fd| = %.6f   U(driver) vs U(recomputed): %.3e", fscale, abs(U0 - U)))
println("            verdict: ", relF <= FD_TOL ? "PASS" : "FAIL")

# --- CHECK 2: total force on a periodic system ---------------------------------
Fsum    = vec(sum(F, dims = 2))
res_rel = maximum(abs, Fsum) / max(maximum(abs, F), 1e-30)
println("")
println("  CHECK 2  internal forces cancel: sum_j F_j must vanish")
println(@sprintf("            sum_j F_j = (%+.3e, %+.3e, %+.3e)   max|sum| / max|F| = %.3e", Fsum[1], Fsum[2], Fsum[3], res_rel))
println("            each pair scatters +g and -g, so this measures accumulation roundoff")
println("            in the scatter, not a physical assumption about the system.")

# --- CHECK 3: per-factor breakdown with compensated summation ------------------
Fbuf     = zeros(3, N)
contribs = [FA.evaluate!(Fbuf, f, q, ctx) for f in factors]
U_naive  = sum(contribs)
U_kahan  = DY.kahan_sum(contribs)
U_neum   = DY.neumaier_sum(contribs)
n_pos    = count(>(0.0), contribs)
n_neg    = count(<(0.0), contribs)
n_zero   = count(iszero, contribs)
rwall    = 2.0^(1 / 6)                       # minimum of the LJ pair potential
n_rep    = count(pairs_nl) do (i, j)
    d = T.unwrap_displacement(box, q, i, j)
    r = sqrt(T.norm2(d))
    r <= RC && r < rwall                     # inside the wall and actually contributing
end
neg_nonzero = [c for c in contribs if c < 0.0]
println("")
println(@sprintf("  CHECK 3  per-factor breakdown over %d terms", length(contribs)))
println(@sprintf(
    "            energy signs: %d positive, %d negative, %d exactly zero (zeros = skin pairs)",
    n_pos, n_neg, n_zero))
println(@sprintf(
    "            %d contributing pairs are inside the potential minimum r < %.4f sigma (repulsive force)",
    n_rep, rwall))
println(@sprintf(
    "            most negative term = %+.6f   |U| = %.6f   sum of |terms| = %.6f",
    minimum(neg_nonzero), abs(U), sum(abs, contribs)))
println(@sprintf("            naive float sum      = %.15f", U_naive))
println(@sprintf("            Kahan compensated    = %.15f", U_kahan))
println(@sprintf("            Neumaier compensated = %.15f", U_neum))
println(@sprintf("            |naive - Neumaier| = %.3e   |Neumaier - driver| = %.3e", abs(U_naive - U_neum), abs(U_neum - U)))
println("            honest caveat: every pair here is beyond the LJ zero crossing at r = 1")
println(@sprintf(
    "            sigma, so all %d nonzero terms share one sign (|U| / sum|terms| = %.3f):",
    n_neg, abs(U) / sum(abs, contribs)))
println("            and this breakdown does NOT stress the summation. The summation hazard")
println("            is cancellation, which needs a synthetic series to show:")
xs    = vcat([1.0e16], fill(1.0, 200_000), [-1.0e16])
exact = Float64(sum(BigFloat.(xs)))          # reference computed in BigFloat, not float
sn    = sum(xs)
sk    = DY.kahan_sum(xs)
sne   = DY.neumaier_sum(xs)
println(@sprintf("            series [1e16, 200000 x 1.0, -1e16]   exact (BigFloat) = %.1f", exact))
println(@sprintf(
    "            naive error = %.3e   Kahan error = %.3e   Neumaier error = %.3e",
    abs(sn - exact), abs(sk - exact), abs(sne - exact)))
println("            that is what compensated summation buys, and why the drift statistic")
println("            in Example 02 is taken from the compensated series.")

# --- CHECK 4: the four cutoff conventions are different potentials -------------
println("")
println("  CHECK 4  cutoff conventions are not cosmetic -- one pair at r = rc +/- 1e-9")
ctx2 = FA.Context(T.NoBox(), T.REDUCED, 2)
rp   = RC - 1e-9
rm   = RC + 1e-9
qp   = [0.0 rp; 0.0 0.0; 0.0 0.0]
qm   = [0.0 rm; 0.0 0.0; 0.0 0.0]
println(@sprintf("            %-8s %-16s %-16s %s\n", "mode", "U just inside", "U just outside", "force_consistent"))
for mode in (:trunc, :shift, :lfs)
    fac = FA.LennardJonesPair(1, 2, 1.0, 1.0; mode = mode, rc = RC)
    up  = FA.total_energy_forces([fac], qp, ctx2)[1]
    um  = FA.total_energy_forces([fac], qm, ctx2)[1]
    println(@sprintf("            %-8s %-16.6e %-16.6e %s", string(mode), up, um, FA.force_consistent(fac)))
end
println("            :trunc jumps by U(rc) across the cutoff: its energy is discontinuous,")
println("            so its force is not the derivative of the energy it reports, and")
println("            `force_consistent` returns false -- the integrator refuses that mix.")
println("            :shift is C^0 but not C^1. Only :lfs (and :none) pass CHECK 1.")

# --- what is NOT in this example ----------------------------------------------
println("")
println("  NOT demonstrated here:")
println("    * long-range electrostatics: no Ewald/PME in this core; minimum-image")
println("      Coulomb is a different model, not a switched version of this one")
println("    * triclinic cells: Types.triclinic_guard refuses them (UnsupportedGeometry)")
println("    * distance constraints (SHAKE/RATTLE): not implemented, refused at runtime")
println("    * real force-field parameterisation, or equilibrated thermodynamics")

if relF > FD_TOL
    println(@sprintf("RESULT: FAIL -- finite-difference relative error %.3e exceeds %.0e", relF, FD_TOL))
    exit(1)
end
if res_rel > 1e-12
    println("RESULT: FAIL -- internal forces do not cancel")
    exit(1)
end
if abs(U_neum - U) > 1e-9
    println("RESULT: FAIL -- per-factor breakdown disagrees with the driver")
    exit(1)
end

println("")
println(BAR)
println(@sprintf("RESULT: PASS   FD relative error = %.3e (threshold 1e-5)   wall time = %.2f s", relF, time() - t_start))
println(BAR)
exit(0)
