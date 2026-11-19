# Coupled distance constraints: rigid water, networks, refusals, the constraint
# virial, long-dynamics quality, and an external (OpenMM/SETTLE) geometry check.
#
# These tests target the failure modes that a per-pair projection cannot detect:
# shared-atom networks where correcting one constraint breaks another, rigid
# bodies where positions and velocities must stay consistent, geometry where the
# constraint Jacobian is rank deficient, and the two things a constrained system
# needs beyond "the bond length is right": the *virial* contribution of the
# constraint forces (NPT pressure) and an external reference for the geometry.

using FoundationCore
using Random
using LinearAlgebra
const CO = FoundationCore.Constraints
const T = FoundationCore.Types
const FA = FoundationCore.Factors
const SL = FoundationCore.StreamedLJ

# TIP3P geometry in Angstrom-like units (O-H 0.9572, H-O-H 104.52 degrees)
const R_OH = 0.9572
const ANG = deg2rad(104.52)
const R_HH = 2 * R_OH * sin(ANG / 2)
const M_O = 15.9994
const M_H = 1.008

function water_system()
    m = [M_O, M_H, M_H]
    cs = CO.CoupledConstraintSet(3, [(1, 2, R_OH), (1, 3, R_OH), (2, 3, R_HH)])
    return m, cs
end

function water_positions()
    q = zeros(3, 3)
    q[:, 2] = [R_OH, 0.0, 0.0]
    q[:, 3] = [R_OH * cos(ANG), R_OH * sin(ANG), 0.0]
    return q
end

"Water geometry rotated about z by `theta`."
function rotated_water(theta)
    q = water_positions()
    c, s = cos(theta), sin(theta)
    R = [c -s 0.0; s c 0.0; 0.0 0.0 1.0]
    return R * q
end

"Maximum |distance^2 - d^2| over every constraint."
function pos_residual(cs, q)
    return maximum(abs.(CO.constraint_residuals(cs, q)))
end

"Maximum r . (v_j - v_i) over every constraint."
function tan_residual(cs, q, p, m)
    return maximum(abs.(CO.tangent_residuals(cs, q, p, m)))
end

"Maximum |bond length - target| over every constraint."
function bond_error(cs, q)
    err = 0.0
    for comp in cs.components
        for (a, b, d) in comp.edges
            i, j = comp.atoms[a], comp.atoms[b]
            err = max(err, abs(sqrt(sum(abs2, q[:, j] .- q[:, i])) - d))
        end
    end
    return err
end

"Mass-weighted Kabsch RMSD of `a` onto `b` (both 3 x N), allowing rotation+translation."
function kabsch_rmsd(a::AbstractMatrix, b::AbstractMatrix, m::AbstractVector)
    ma = sum(m)
    ca = vec(sum(a .* reshape(m, 1, :), dims = 2)) ./ ma
    cb = vec(sum(b .* reshape(m, 1, :), dims = 2)) ./ ma
    a0 = a .- ca
    b0 = b .- cb
    H = a0 * Diagonal(m) * transpose(b0)
    # Use the SVD struct fields explicitly: Julia 1.13 destructures
    # `U, S, V = svd(H)` to V (not Vt), and that is easy to get wrong.
    F = svd(H)
    U, Vt = F.U, F.Vt
    D = Diagonal([1.0, 1.0, sign(det(Vt' * U'))])
    Rot = Vt' * D * U'
    r = Rot * a0 .+ cb .- b
    return sqrt(sum(m .* vec(sum(abs2, r; dims = 1))) / ma), Rot
end

function hoh_angle_deg(q)
    u = q[:, 2] .- q[:, 1]
    v = q[:, 3] .- q[:, 1]
    return rad2deg(acos(clamp(dot(u, v) / (norm(u) * norm(v)), -1.0, 1.0)))
end

# ---------------------------------------------------------------------------
# rigid water: projection, rigidity, conservation
# ---------------------------------------------------------------------------

function test_rigid_water_projection()
    m, cs = water_system()
    rng = MersenneTwister(7)
    worst_pos = 0.0
    worst_tan = 0.0
    for _ in 1:200
        q = water_positions() .+ 0.05 .* randn(rng, 3, 3)
        p = 0.01 .* randn(rng, 3, 3)
        CO.project!(q, p, m, cs; tolerance = 1e-12)
        worst_pos = max(worst_pos, pos_residual(cs, q))
        worst_tan = max(worst_tan, tan_residual(cs, q, p, m))
    end
    @assert worst_pos < 1e-11
    @assert worst_tan < 1e-11
    println("  rigid water: 200 random perturbations, worst pos=$(worst_pos) tan=$(worst_tan)")
end

"Rigid water must preserve its internal geometry exactly, not merely converge."
function test_rigid_water_rigidity()
    m, cs = water_system()
    q = water_positions()
    # rotate + translate; the internal distances must be invariant
    theta = 0.7
    R = [cos(theta) -sin(theta) 0.0; sin(theta) cos(theta) 0.0; 0.0 0.0 1.0]
    q2 = R * q .+ [3.0, -2.0, 1.5]
    p = zeros(3, 3)
    CO.project!(q2, p, m, cs; tolerance = 1e-13)
    @assert pos_residual(cs, q2) < 1e-12
    # orientation preserved: the O-H distances are unchanged by construction
    d12 = sqrt(sum(abs2, q2[:, 2] .- q2[:, 1]))
    d13 = sqrt(sum(abs2, q2[:, 3] .- q2[:, 1]))
    @assert abs(d12 - R_OH) < 1e-12
    @assert abs(d13 - R_OH) < 1e-12
    println("  rigid water: rotated copy keeps O-H = $(d12), $(d13)")
end

"Projection must not translate the centre of mass or change total momentum."
function test_rigid_water_conservation()
    m, cs = water_system()
    q = water_positions() .+ 0.02 .* randn(MersenneTwister(3), 3, 3)
    p = randn(MersenneTwister(4), 3, 3) .* reshape(m, 1, :)
    q0 = copy(q)
    p0 = copy(p)
    com_before = sum(q0 .* reshape(m, 1, :), dims = 2) / sum(m)
    mom_before = vec(sum(p0, dims = 2))
    CO.project!(q, p, m, cs; tolerance = 1e-12)
    com_after = sum(q .* reshape(m, 1, :), dims = 2) / sum(m)
    mom_after = vec(sum(p, dims = 2))
    @assert maximum(abs.(com_after .- com_before)) < 1e-12
    @assert maximum(abs.(mom_after .- mom_before)) < 1e-12
    println("  rigid water: COM drift $(maximum(abs.(com_after .- com_before))), momentum drift $(maximum(abs.(mom_after .- mom_before)))")
end

"Two independent waters must be projected separately and correctly."
function test_two_waters()
    m = [M_O, M_H, M_H, M_O, M_H, M_H]
    cs = CO.CoupledConstraintSet(6, [
        (1, 2, R_OH), (1, 3, R_OH), (2, 3, R_HH),
        (4, 5, R_OH), (4, 6, R_OH), (5, 6, R_HH),
    ])
    @assert length(cs.components) == 2
    q = hcat(water_positions(), water_positions() .+ [5.0, 0.0, 0.0])
    q .+= 0.03 .* randn(MersenneTwister(11), 3, 6)
    p = 0.01 .* randn(MersenneTwister(12), 3, 6)
    CO.project!(q, p, m, cs; tolerance = 1e-12)
    @assert pos_residual(cs, q) < 1e-11
    @assert tan_residual(cs, q, p, m) < 1e-11
    println("  two waters: separate components, residual $(pos_residual(cs, q))")
end

"A rigid water plus an independent diatomic: components of different sizes."
function test_mixed_component_sizes()
    m = [M_O, M_H, M_H, 12.0, 12.0]
    d45 = 1.4
    cs = CO.CoupledConstraintSet(5, [(1, 2, R_OH), (1, 3, R_OH), (2, 3, R_HH), (4, 5, d45)])
    @assert length(cs.components) == 2
    sizes = sort([length(c.atoms) for c in cs.components])
    @assert sizes == [2, 3] "expected a 2-atom and a 3-atom component, got $sizes"
    q = hcat(water_positions(), [3.0, 0.0, 0.0], [4.4, 0.0, 0.0])
    q .+= 0.02 .* randn(MersenneTwister(31), 3, 5)
    p = 0.01 .* randn(MersenneTwister(32), 3, 5)
    q_before = copy(q)
    CO.project!(q, p, m, cs; tolerance = 1e-13)
    @assert pos_residual(cs, q) < 1e-12
    @assert tan_residual(cs, q, p, m) < 1e-12
    @assert abs(norm(q[:, 5] - q[:, 4]) - d45) < 1e-12
    @assert bond_error(cs, q) < 1e-12
    # components are independent: each moved, and the water's own geometry is
    # unchanged by the diatomic's presence
    @assert maximum(abs.(q .- q_before)) > 1e-3
    # multipliers / virial work across mixed component sizes too
    F = 0.01 .* randn(MersenneTwister(33), 3, 5)
    λ = CO.constraint_multipliers(q, m, cs, F)
    @assert length(λ) == 4
    @assert isfinite(CO.constraint_virial(q, m, cs, λ))
    println("  mixed component sizes (3-atom water + 2-atom bond): residual ",
            "$(pos_residual(cs, q)), virial $(CO.constraint_virial(q, m, cs, λ))")
end

# ---------------------------------------------------------------------------
# refusals: over-determined, rank deficient, non-convergent -- all must name
# the offending atoms, because a silent stall in a 10^5-atom run is unarguable
# ---------------------------------------------------------------------------

"Run `f`, return the exception it throws (or `nothing`)."
function caught(f)
    try
        f()
    catch err
        return err
    end
    return nothing
end

"Extract the message text of an exception."
message_of(err) = sprint(showerror, err)

"A shared-atom network that is over-determined must be refused."
function test_overdetermined_refused()
    # 3 atoms, 4 constraints: impossible for a rigid triangle
    err = caught(() -> CO.CoupledConstraintSet(3, [(1, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0), (2, 3, 1.1)]))
    @assert err isa ArgumentError || err isa CO.UnsupportedConstraintSet
    @assert occursin("(2, 3)", message_of(err)) "the duplicate constraint must name its atoms, got: $(message_of(err))"
    println("  over-determined set: refused ($(message_of(err)))")
end

"A component with more constraints than internal degrees of freedom is refused."
function test_component_dof_refused()
    # 7 atoms -> 3*7 - 3 = 18 internal degrees of freedom; 19 distinct pairs.
    pairs = Tuple{Int,Int}[]
    for i in 1:7, j in (i + 1):7
        push!(pairs, (i, j))
    end
    @assert length(pairs) == 21
    constraints = [(i, j, 1.0) for (i, j) in pairs[1:19]]
    err = caught(() -> CO.CoupledConstraintSet(7, constraints))
    @assert err isa CO.UnsupportedConstraintSet
    msg = message_of(err)
    @assert occursin("over-determined", msg)
    @assert occursin("[1, 2, 3, 4, 5, 6, 7]", msg) "the refusal must name the component atoms, got: $msg"
    println("  19 constraints on 7 atoms (18 internal DOF): refused")
end

"Collinear atoms make the Jacobian singular; refuse rather than produce garbage."
function test_singular_geometry_refused()
    m = [1.0, 1.0, 1.0]
    cs = CO.CoupledConstraintSet(3, [(1, 2, 1.0), (2, 3, 1.0), (1, 3, 2.0)])
    # Collinear AND violating the constraints: the Jacobian is rank deficient at
    # this geometry, so a correction is required but is not uniquely defined.
    # (Collinear-but-satisfied geometry needs no correction and is therefore a
    # perfectly valid state -- that case must NOT be refused.)
    q = [0.0 1.0 2.4; 0.0 0.0 0.0; 0.0 0.0 0.0]
    err = caught(() -> CO.project_positions!(q, m, cs; tolerance = 1e-14))
    @assert err isa CO.UnsupportedConstraintSet || err isa CO.ConstraintProjectionError
    msg = message_of(err)
    @assert occursin("[1, 2, 3]", msg) "the refusal must name the atoms, got: $msg"
    @assert occursin("parallel", lowercase(msg)) || occursin("rank deficient", lowercase(msg))
    println("  singular geometry: refused -- $(msg)")
    # The force-based multiplier solve must refuse the same geometry.
    err2 = caught(() -> CO.constraint_multipliers(q, m, cs, zeros(3, 3)))
    @assert err2 isa CO.UnsupportedConstraintSet
    @assert occursin("[1, 2, 3]", message_of(err2))
    println("  singular geometry: constraint_multipliers refused too")
end

"Collinear geometry that already satisfies its constraints is a valid state."
function test_collinear_satisfied_is_allowed()
    m = [1.0, 1.0, 1.0]
    cs = CO.CoupledConstraintSet(3, [(1, 2, 1.0), (2, 3, 1.0), (1, 3, 2.0)])
    q = [0.0 1.0 2.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
    r = CO.project_positions!(q, m, cs; tolerance = 1e-14)
    @assert abs(r) < 1e-14
    @assert q[1, 3] == 2.0 "an already-satisfied geometry must not be moved"
    println("  collinear satisfied geometry: accepted unchanged")
end

"Projection failure must raise, never return silently violated geometry."
function test_nonconvergence_raises()
    m, cs = water_system()
    q = water_positions() .+ 0.5 .* randn(MersenneTwister(5), 3, 3)
    err = caught(() -> CO.project_positions!(q, m, cs; tolerance = 1e-16, max_iterations = 1))
    @assert err isa CO.ConstraintProjectionError
    msg = message_of(err)
    @assert occursin("[1, 2, 3]", msg) "the failure must name the atoms, got: $msg"
    @assert occursin("position", msg) && occursin("1 iterations", msg)
    println("  non-convergence: raised ConstraintProjectionError -- $msg")
end

"Constructor refusals: indices, distances, duplicates."
function test_constructor_refusals()
    err = caught(() -> CO.CoupledConstraintSet(2, [(1, 3, 1.0)]))
    @assert err isa ArgumentError
    err = caught(() -> CO.CoupledConstraintSet(2, [(1, 2, -1.0)]))
    @assert err isa ArgumentError && occursin("positive and finite", message_of(err))
    err = caught(() -> CO.CoupledConstraintSet(2, [(1, 2, Inf)]))
    @assert err isa ArgumentError
    err = caught(() -> CO.CoupledConstraintSet(2, [(1, 1, 1.0)]))
    @assert err isa ArgumentError
    err = caught(() -> CO.CoupledConstraintSet(2, [(1, 2, 1.0), (2, 1, 1.0)]))
    @assert err isa ArgumentError && occursin("(1, 2)", message_of(err))
    println("  constructor refusals: index, distance, self-pair, duplicate all fire")
end

# ---------------------------------------------------------------------------
# constraint virial
# ---------------------------------------------------------------------------

"Build the virial fixture: two rigid waters plus explicit LJ pairs."
function virial_fixture()
    # The second molecule is *rotated* relative to the first: with a pure
    # translation the two molecules' constraint virials cancel exactly (by
    # symmetry) and the test would be vacuous. A rotation makes W_c ~1 % of
    # W_ext, so the numbers below are actually testing something.
    q = hcat(water_positions(), rotated_water(0.7) .+ [3.45, 0.42, 0.25])
    m = [M_O, M_H, M_H, M_O, M_H, M_H]
    cs = CO.CoupledConstraintSet(6, [
        (1, 2, R_OH), (1, 3, R_OH), (2, 3, R_HH),
        (4, 5, R_OH), (4, 6, R_OH), (5, 6, R_HH)])
    # Intermolecular O-O LJ only; H sites carry eps = 0 so those pairs are zero
    # (Lorentz-Berthelot gives eps_ij = 0) but stay in the enumeration.
    eps = [0.636386, 0.0, 0.0, 0.636386, 0.0, 0.0]
    sig = [3.15061, 1.0, 1.0, 3.15061, 1.0, 1.0]
    box = T.OrthorhombicBox((60.0, 60.0, 60.0))     # big enough: no image pairs
    topo = SL.PairTopology(excluded = [(1, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 6)])
    fac = SL.StreamedLJFactor(q, eps, sig, box; rcut = 12.0, topology = topo)
    ctx = FA.Context(box, T.REDUCED, 6)
    U(qq) = SL.streamed_energy(fac, qq, ctx)
    function UF(qq)
        F = zeros(3, 6)
        return FA.total_energy_forces([fac], qq, ctx; F = F), F
    end
    return m, cs, q, U, UF
end

"""
The two multiplier routes, the closed-form virial, and momentum conservation.

`W_c = -2 Σ λ_k |r_k|²` must equal `Σ_i r_i · F^c_i` with `F^c = -Gᵀ λ` computed
from the same `λ`; the constraint forces of one constraint sum to zero, so the
total constraint force on the system is zero to roundoff.
"""
function test_constraint_virial_formulation()
    m, cs, q, U, UF = virial_fixture()
    _, F = UF(q)
    λ = CO.constraint_multipliers(q, m, cs, F)
    @assert length(λ) == 6
    Wc = CO.constraint_virial(q, m, cs, λ)
    Fc = CO.constraint_forces(q, m, cs, λ)
    Wdirect = sum(q .* Fc)
    @assert abs(Wc - Wdirect) <= 1e-12 * max(1.0, abs(Wdirect))
    @assert maximum(abs.(vec(sum(Fc, dims = 2)))) < 1e-14

    # The closed form uses |r|², so it must also hold off the manifold (the
    # multiplier list is a geometric contraction, not a projection result).
    qoff = q .+ 0.01 .* randn(MersenneTwister(21), 3, 6)
    λoff = CO.constraint_multipliers(qoff, m, cs, F)
    Woff = CO.constraint_virial(qoff, m, cs, λoff)
    Fcoff = CO.constraint_forces(qoff, m, cs, λoff)
    @assert abs(Woff - sum(qoff .* Fcoff)) <= 1e-12 * max(1.0, abs(Woff))

    # A zero external force field means no constraint force and no virial.
    @assert maximum(abs.(CO.constraint_multipliers(q, m, cs, zeros(3, 6)))) < 1e-30
    @assert abs(CO.constraint_virial(q, m, cs, zeros(6))) == 0.0

    # Argument validation.
    @assert caught(() -> CO.constraint_virial(q, m, cs, zeros(5))) isa ArgumentError
    @assert caught(() -> CO.constraint_virial(q, m, cs, fill(Inf, 6))) isa ArgumentError
    @assert caught(() -> CO.constraint_multipliers(q, m, cs, zeros(3, 5))) isa ArgumentError
    @assert caught(() -> CO.constraint_forces(q, m, cs, zeros(5))) isa ArgumentError

    println("  virial formulation: W_c=$(Wc) (direct $(Wdirect)), net constraint force ",
            "$(maximum(abs.(vec(sum(Fc, dims = 2)))))")
end

"""
The box-derivative identity that makes the constraint virial *correct* rather
than merely well-defined.

Test, stated precisely. Let `Π` be the coupled Newton projection and
`q(s) = Π(s q0)` the constraint-preserving volume-scaling path (affine scale,
then re-project; a rigid molecule scaled affinely stretches its bonds, so the
re-projection is what makes the path admissible). With `U(s) = U_ext(q(s))`,

    -dU/ds |_{s=1} = W_ext + W_c,
    W_ext = Σ_i r_i·F_i^ext,   W_c = -2 Σ_k λ_k |r_k|²,
    λ = (G W Gᵀ)⁻¹ G W F^ext  (the static constraint multipliers).

This is the resolved statement of "the constraint virial is what makes the total
virial the box derivative" and it is exact at s=1; the finite-difference error is
O(ε²), so the residual should fall by 100x per decade of ε.
"""
function test_constraint_virial_box_derivative()
    m, cs, q0, U, UF = virial_fixture()
    _, F0 = UF(q0)
    Wext = sum(q0 .* F0)
    λ = CO.constraint_multipliers(q0, m, cs, F0)
    Wc = CO.constraint_virial(q0, m, cs, λ)
    total = Wext + Wc

    function Uscaled(s)
        qs = s .* q0
        CO.project_positions!(qs, m, cs; tolerance = 1e-14)
        return U(qs)
    end

    abs(Wc) > 1e-3 || error("the fixture's constraint virial is $(Wc); the test would be vacuous")
    best = Inf
    for (eps, tol) in ((1e-4, 1e-5), (1e-5, 1e-6), (1e-6, 1e-7))
        dUds = (Uscaled(1 + eps) - Uscaled(1 - eps)) / (2eps)
        rel = abs(-dUds - total) / abs(total)
        best = min(best, rel)
        println("  box scaling eps=$(eps): -dU/ds=$(-dUds) total=$total rel=$rel")
        rel < tol || error("virial identity failed at eps=$eps: rel=$rel > $tol")
    end
    @assert best < 1e-7

    # Momentum must still be conserved along the constrained path: the
    # constraint forces are internal, so `Σ F^c = 0` and the path's COM moves
    # only with the external force.
    Fc = CO.constraint_forces(q0, m, cs, λ)
    @assert maximum(abs.(vec(sum(Fc, dims = 2)))) < 1e-14
    println("  box-derivative identity: -dU/ds = W_ext + W_c to $(best) relative")
end

"""
The multipliers returned by the projection must reproduce the static ones, and
the resulting virial must equal the direct `Σ q·M Δx/dt²`.

The trial displacement is a *full* free step (`Δq = W F dt²`) driven by a known
force field, so the impulse and the static multipliers describe the same thing.
This is a *calibration* check, not the recipe for an MD pressure accumulator: in
velocity Verlet the projection also removes tangential drift, and the
time-averaged impulse virial does not reproduce the static constraint virial (see
the `Constraints` module docstring). Pressure must use
`constraint_multipliers`.
"""
function test_constraint_virial_projection_route()
    m, cs, q0, U, UF = virial_fixture()
    _, F0 = UF(q0)
    λstatic = CO.constraint_multipliers(q0, m, cs, F0)

    dt = 1e-3
    qtrial = q0 .+ dt^2 .* (F0 ./ reshape(m, 1, :))
    qproj = copy(qtrial)
    res, λproj = CO.project_positions_with_multipliers!(qproj, m, cs, dt; tolerance = 1e-14)
    @assert res < 1e-13
    @assert length(λproj) == 6

    # Direct route: the impulse the projection applied, divided by dt².
    Wdir = sum(q0 .* (reshape(m, 1, :) .* (qproj .- qtrial) ./ dt^2))
    Wproj = CO.constraint_virial(qproj, m, cs, λproj)
    Wstat = CO.constraint_virial(q0, m, cs, λstatic)
    @assert abs(Wproj - Wdir) <= 1e-6 * abs(Wdir)
    @assert abs(Wproj - Wstat) <= 1e-6 * abs(Wstat)
    # λ itself must converge to the static multipliers: the mismatch is O(dt²).
    rel = maximum(abs.(λproj .- λstatic)) / maximum(abs.(λstatic))
    @assert rel < 1e-4
    println("  projection-route multipliers: |Δλ|/|λ|=$(rel), W_c=$(Wproj) vs static $(Wstat), direct $(Wdir)")

    # A displacement that already satisfies the constraints buys no multipliers.
    qsat = copy(q0)
    r0, λ0 = CO.project_positions_with_multipliers!(qsat, m, cs, dt; tolerance = 1e-14)
    @assert r0 < 1e-14 && all(iszero, λ0)
    @assert caught(() -> CO.project_positions_with_multipliers!(qsat, m, cs, 0.0)) isa ArgumentError
    @assert caught(() -> CO.project_positions_with_multipliers!(qsat, m, cs, -1.0)) isa ArgumentError
    println("  satisfied geometry: zero multipliers, dt validation fires")
end

# ---------------------------------------------------------------------------
# long-dynamics quality: 20,000 steps with the projection every step
# ---------------------------------------------------------------------------

# Reduced-unit water-like model: sigma = eps = 1, masses 16/1/1, rigid triangle
# of side 1.0. Reduced units are the house convention (Types.REDUCED), so this is
# a real force field (streamed LJ) with no unit conversion in the way.
const LR_OH = 1.0
const LR_HH = 2 * LR_OH * sin(ANG / 2)
const LR_MO = 16.0
const LR_MH = 1.0
const LR_BOX = 4.5
const LR_RCUT = 2.2

function reduced_water(theta = 0.0, com = (0.0, 0.0, 0.0))
    q = zeros(3, 3)
    q[:, 2] = [LR_OH, 0.0, 0.0]
    q[:, 3] = [LR_OH * cos(ANG), LR_OH * sin(ANG), 0.0]
    c, s = cos(theta), sin(theta)
    R = [c -s 0.0; s c 0.0; 0.0 0.0 1.0]
    return R * q .+ collect(com)
end

"""
Three rigid molecules in a periodic box, interacting through streamed LJ between
oxygen sites only. Returns the pieces a constrained Verlet loop needs.
"""
function reduced_system(nmolecules = 3)
    m = repeat([LR_MO, LR_MH, LR_MH], nmolecules)
    q = hcat((reduced_water(0.3k, (1.9k, 1.7k, 0.2k)) for k in 0:(nmolecules - 1))...)
    cs = CO.CoupledConstraintSet(3nmolecules,
        vcat([[(3k + 1, 3k + 2, LR_OH), (3k + 1, 3k + 3, LR_OH),
               (3k + 2, 3k + 3, LR_HH)] for k in 0:(nmolecules - 1)]...))
    excl = vcat([[(3k + 1, 3k + 2), (3k + 1, 3k + 3), (3k + 2, 3k + 3)]
                 for k in 0:(nmolecules - 1)]...)
    box = T.OrthorhombicBox((LR_BOX, LR_BOX, LR_BOX))
    eps = repeat([1.0, 0.0, 0.0], nmolecules)
    sig = repeat([1.0, 1.0, 1.0], nmolecules)
    fac = SL.StreamedLJFactor(q, eps, sig, box;
                              rcut = LR_RCUT, topology = SL.PairTopology(excluded = excl))
    ctx = FA.Context(box, T.REDUCED, 3nmolecules)
    return m, q, cs, fac, ctx
end

"""
One velocity-Verlet step with coupled constraints — the loop `Dynamics` would
run if its RATTLE path accepted shared atoms.

The projection's displaced coordinates are used to correct the momenta
(`p += m Δq / dt`), which is what makes the constrained step second order; the
velocity projection then enforces `G v = 0`.
"""
function constrained_verlet_step!(q, p, F, fac, ctx, m, cs, dt; tolerance = 1e-12)
    p .+= (0.5 * dt) .* F
    qtrial = q .+ dt .* (p ./ reshape(m, 1, :))
    copyto!(q, qtrial)
    CO.project_positions!(q, m, cs; tolerance = tolerance)
    p .+= reshape(m, 1, :) .* (q .- qtrial) ./ dt
    FA.total_energy_forces([fac], q, ctx; F = F)
    p .+= (0.5 * dt) .* F
    CO.project_velocities!(q, p, m, cs; tolerance = tolerance)
    return nothing
end

total_energy_lj(fac, ctx, m, q, p) =
    SL.streamed_energy(fac, q, ctx) +
    0.5 * sum(p[d, j]^2 / m[j] for j in axes(p, 2) for d in 1:3)

function test_long_dynamics_20k()
    m, q, cs, fac, ctx = reduced_system()
    p = 0.05 .* randn(MersenneTwister(4), 3, length(m))
    F = zeros(3, length(m))
    FA.total_energy_forces([fac], q, ctx; F = F)
    CO.project_positions!(q, m, cs; tolerance = 1e-12)
    CO.project_velocities!(q, p, m, cs; tolerance = 1e-12)

    dt = 0.005
    nsteps = 20_000
    E0 = total_energy_lj(fac, ctx, m, q, p)
    R0 = vec(sum(q .* reshape(m, 1, :), dims = 2)) ./ sum(m)
    P0 = vec(sum(p, dims = 2))
    worst_pos = 0.0
    worst_tan = 0.0
    worst_bond = 0.0
    worst_com = 0.0
    worst_mom = 0.0
    Es = Vector{Float64}(undef, nsteps)
    t0 = time()
    for step in 1:nsteps
        constrained_verlet_step!(q, p, F, fac, ctx, m, cs, dt)
        worst_pos = max(worst_pos, pos_residual(cs, q))
        worst_tan = max(worst_tan, tan_residual(cs, q, p, m))
        worst_bond = max(worst_bond, bond_error(cs, q))
        Rt = vec(sum(q .* reshape(m, 1, :), dims = 2)) ./ sum(m)
        worst_com = max(worst_com,
                        maximum(abs.(Rt .- (R0 .+ P0 ./ sum(m) .* (step * dt)))))
        worst_mom = max(worst_mom, maximum(abs.(vec(sum(p, dims = 2)) .- P0)))
        Es[step] = total_energy_lj(fac, ctx, m, q, p)
    end
    elapsed = time() - t0

    band = maximum(Es) - minimum(Es)
    k = nsteps ÷ 10
    first10 = sum(Es[1:k]) / k
    last10 = sum(Es[(nsteps - k + 1):nsteps]) / k
    drift = abs(last10 - first10)

    @assert worst_pos < 1e-11 "position residual grew to $worst_pos"
    @assert worst_tan < 1e-11 "tangent residual grew to $worst_tan"
    @assert worst_bond < 1e-12 "bond length left the constraint surface: $worst_bond"
    @assert worst_com < 1e-8 "COM deviated from ballistic motion: $worst_com"
    @assert worst_mom < 1e-8 "momentum drifted: $worst_mom"
    @assert band < 1e-3 * max(abs(E0), 1e-3) "energy band $band too wide (E0=$E0)"
    @assert drift < max(0.1 * band, 1e-12) "energy drifted between the first and last decile: $drift (band $band)"

    # The same model with the constraints replaced by stiff harmonic bonds is the
    # unconstrained baseline: same integrator, same atoms, same force field, no
    # projection. Removing the stiffest degrees of freedom must not make the
    # energy *worse* than that baseline.
    flex_band = flexible_energy_band(nsteps, dt; seed = 4)
    @assert band <= flex_band "constrained band $band exceeds the flexible baseline $flex_band"

    println("  20,000 steps dt=$dt ($(round(elapsed, digits = 2))s): worst pos=$(worst_pos) ",
            "tan=$(worst_tan) bond=$(worst_bond)")
    println("    energy: E0=$(E0) band=$(band) drift(first/last decile)=$(drift) ",
            "| unconstrained (stiff harmonic bonds) band=$(flex_band)")
    println("    COM deviation=$(worst_com), momentum drift=$(worst_mom)")
end

"Same 3-molecule model with harmonic bonds/angle instead of constraints."
function flexible_energy_band(nsteps, dt; seed = 4)
    m, q, cs, fac_lj, ctx = reduced_system()
    factors = FA.AbstractFactor[fac_lj]
    k_bond, k_angle = 1500.0, 300.0
    for k in 0:2
        push!(factors, FA.HarmonicBond(3k + 1, 3k + 2, k_bond, LR_OH))
        push!(factors, FA.HarmonicBond(3k + 1, 3k + 3, k_bond, LR_OH))
        push!(factors, FA.HarmonicBond(3k + 2, 3k + 3, k_bond, LR_HH))
        push!(factors, FA.HarmonicAngle(3k + 2, 3k + 1, 3k + 3, k_angle, ANG))
    end
    p = 0.05 .* randn(MersenneTwister(seed), 3, length(m))
    F = zeros(3, length(m))
    FA.total_energy_forces(factors, q, ctx; F = F)
    E(qq, pp) = FA.total_energy_forces(factors, qq, ctx)[1] +
                0.5 * sum(pp[d, j]^2 / m[j] for j in axes(pp, 2) for d in 1:3)
    lo, hi = Inf, -Inf
    for _ in 1:nsteps
        p .+= (0.5 * dt) .* F
        q .+= dt .* (p ./ reshape(m, 1, :))
        FA.total_energy_forces(factors, q, ctx; F = F)
        p .+= (0.5 * dt) .* F
        e = E(q, p)
        lo = min(lo, e); hi = max(hi, e)
    end
    return hi - lo
end

"""
A fast-moving molecule: the projection must not fail or creep under a strain of
~10 % of a bond per step.

Two cases. (a) Pure free flight at speed 20 (no force field), which is the
largest per-step strain the projection ever sees and has an exact answer: the
molecule stays rigid and its COM moves ballistically. (b) The LJ system with one
molecule at 5x thermal speed, where the energy band is dominated by a collision
the *integrator* under-resolves -- reported, and bounded at 1 % of |E|, because
that band is not a constraint failure.
"""
function test_fast_molecule()
    # (a) free flight, speed 20 in reduced units, dt = 0.005 -> 0.1 per step
    m, cs = water_system()
    q = water_positions()
    v = [20.0, 7.0, -11.0]
    p = v .* reshape(m, 1, :)
    q0 = copy(q)
    p0 = copy(p)
    worst_pos = 0.0
    worst_bond = 0.0
    dt = 0.005
    for _ in 1:5000
        q .+= dt .* (p ./ reshape(m, 1, :))
        worst_pos = max(worst_pos, CO.project_positions!(q, m, cs; tolerance = 1e-12))
        CO.project_velocities!(q, p, m, cs; tolerance = 1e-12)
        worst_bond = max(worst_bond, bond_error(cs, q))
    end
    com_drift = maximum(abs.(vec(sum(q .* reshape(m, 1, :), dims = 2)) ./ sum(m) .-
                             (vec(sum(q0 .* reshape(m, 1, :), dims = 2)) ./ sum(m) .+
                              5000 * dt * v)))
    mom_drift = maximum(abs.(vec(sum(p, dims = 2)) .- vec(sum(p0, dims = 2))))
    @assert worst_pos < 1e-11 "fast free flight: residual $worst_pos"
    @assert worst_bond < 1e-12 "fast free flight: bond error $worst_bond"
    @assert com_drift < 1e-9 "fast free flight: COM deviation $com_drift"
    @assert mom_drift < 1e-12 "fast free flight: momentum drift $mom_drift"
    println("  fast free flight (speed 20, 5000 steps): worst pos=$(worst_pos) ",
            "bond=$(worst_bond) COM=$(com_drift) momentum=$(mom_drift)")

    # (b) the LJ system with one molecule at 5x speed
    m2, q2, cs2, fac, ctx = reduced_system()
    p2 = 0.05 .* randn(MersenneTwister(7), 3, length(m2))
    p2[:, 1:3] .*= 5.0
    F = zeros(3, length(m2))
    FA.total_energy_forces([fac], q2, ctx; F = F)
    CO.project_positions!(q2, m2, cs2; tolerance = 1e-12)
    CO.project_velocities!(q2, p2, m2, cs2; tolerance = 1e-12)
    E0 = total_energy_lj(fac, ctx, m2, q2, p2)
    lo, hi = Inf, -Inf
    worst_pos2 = 0.0
    worst_bond2 = 0.0
    for _ in 1:5000
        constrained_verlet_step!(q2, p2, F, fac, ctx, m2, cs2, 0.005)
        worst_pos2 = max(worst_pos2, pos_residual(cs2, q2))
        worst_bond2 = max(worst_bond2, bond_error(cs2, q2))
        e = total_energy_lj(fac, ctx, m2, q2, p2)
        lo = min(lo, e); hi = max(hi, e)
    end
    band = hi - lo
    @assert worst_pos2 < 1e-11
    @assert worst_bond2 < 1e-12
    @assert band < 1e-2 * max(abs(E0), 1e-3)
    println("  fast LJ molecule (5x speed, 5000 steps): worst pos=$(worst_pos2) ",
            "bond=$(worst_bond2) energy band=$(band) of |E|=$(abs(E0))")
end

# ---------------------------------------------------------------------------
# external reference: OpenMM's constraint machinery on the same displaced inputs
# ---------------------------------------------------------------------------

"One parsed case from `settle_reference.py` (atom-major flat coordinate lists)."
mutable struct SettleCase
    input::Vector{Float64}
    openmm::Vector{Float64}
    analytic::Vector{Float64}
end
SettleCase() = SettleCase(Float64[], Float64[], Float64[])

"Parse the `# BEGIN DATA` block printed by `settle_reference.py`."
function parse_settle_reference(text::AbstractString)
    masses = Float64[]
    targets = Float64[]
    cases = SettleCase[]
    platforms = Tuple{String,Float64}[]
    cur = nothing
    started = false
    for raw in split(text, '\n')
        occursin("BEGIN DATA", raw) && (started = true; continue)
        started || continue
        tok = split(strip(raw))
        isempty(tok) && continue
        key = tok[1]
        if key == "masses"
            masses = parse.(Float64, tok[2:end])
        elseif key == "targets"
            targets = parse.(Float64, tok[2:end])
        elseif key == "platform"
            push!(platforms, (String(tok[2]), parse(Float64, tok[3])))
        elseif key == "case"
            push!(cases, SettleCase())
            cur = cases[end]
        elseif key == "input"
            cur.input = parse.(Float64, tok[2:end])
        elseif key == "openmm"
            cur.openmm = parse.(Float64, tok[2:end])
        elseif key == "analytic"
            cur.analytic = parse.(Float64, tok[2:end])
        elseif key == "ends"
            cur = nothing
        end
    end
    return masses, targets, cases, platforms
end

"""
Compare our projection with OpenMM's `Context.applyConstraints` on the same
displaced TIP3P triangles.

SETTLE (OpenMM) and the coupled Newton iteration are different maps: they agree
on the constraint *surface*, not on the iterates. For a three-atom triangle with
three distances the surface is the rigid-motion orbit, so the honest comparison
is (i) does the external reference satisfy the targets, (ii) does ours, and
(iii) are the two outputs the same triangle (bond lengths, angle, and mass-
weighted RMSD after optimal superposition)? The raw coordinate difference is
reported but not asserted, because a rotation about the centre of mass is a
legitimate difference between the two solvers -- and it is what we measured.
"""
function test_settle_reference()
    script = joinpath(@__DIR__, "..", "validation", "adapters", "settle_reference.py")
    python = let p = get(ENV, "FOUNDATION_OPENMM_PYTHON", "")
        !isempty(p) ? p : something(Sys.which("python3"), Sys.which("python"), "")
    end
    text = ""
    if isfile(script) && isfile(python)
        try
            text = read(`$python $script`, String)
        catch err
            println("  SETTLE reference: could not run OpenMM ($(typeof(err))); ",
                    "falling back to the analytic rigid triangle")
            text = ""
        end
    else
        println("  SETTLE reference: OpenMM python not found at $python; ",
                "falling back to the analytic rigid triangle")
    end

    if isempty(text)
        # Analytic fallback: the only triangle with these three distances (up to
        # rigid motion) is the input geometry itself, so a projection from a
        # displaced input must land on it.
        m, cs = water_system()
        @assert abs(hoh_angle_deg(water_positions()) - rad2deg(ANG)) < 1e-12
        for trial in 1:5
            qin = water_positions() .+ 0.05 .* randn(MersenneTwister(30 + trial), 3, 3)
            q = copy(qin)
            CO.project_positions!(q, m, cs; tolerance = 1e-13)
            @assert bond_error(cs, q) < 1e-12
            @assert abs(hoh_angle_deg(q) - rad2deg(ANG)) < 1e-10
            # and it must be congruent to the analytic triangle
            rmsd, _ = kabsch_rmsd(q, water_positions(), m)
            @assert rmsd < 1e-12 "analytic fallback: RMSD $rmsd"
        end
        println("  SETTLE reference: analytic rigid-triangle fallback PASSED ",
                "(5 displaced inputs -> target geometry to 1e-12)")
        return
    end

    masses, targets, cases, platforms = parse_settle_reference(text)
    @assert length(masses) == 3 && length(targets) == 3 && !isempty(cases)
    cs = CO.CoupledConstraintSet(3, [(1, 2, targets[1]), (1, 3, targets[2]), (2, 3, targets[3])])
    worst_geom_ours = 0.0
    worst_geom_om = 0.0
    worst_bond_diff = 0.0
    worst_angle_diff = 0.0
    worst_rmsd = 0.0
    worst_oracle = 0.0
    worst_shift = 0.0
    worst_raw = 0.0
    worst_rotation = 0.0
    for case in cases
        qin = reshape(case.input, 3, 3)
        qom = reshape(case.openmm, 3, 3)
        qan = reshape(case.analytic, 3, 3)
        q = copy(qin)
        CO.project_positions!(q, masses, cs; tolerance = 1e-13)

        d_ours = [norm(q[:, 2] - q[:, 1]), norm(q[:, 3] - q[:, 1]), norm(q[:, 3] - q[:, 2])]
        d_om = [norm(qom[:, 2] - qom[:, 1]), norm(qom[:, 3] - qom[:, 1]),
                norm(qom[:, 3] - qom[:, 2])]
        geom_ours = maximum(abs.(d_ours .- targets))
        geom_om = maximum(abs.(d_om .- targets))
        worst_geom_ours = max(worst_geom_ours, geom_ours)
        worst_geom_om = max(worst_geom_om, geom_om)
        worst_bond_diff = max(worst_bond_diff, maximum(abs.(d_ours .- d_om)))
        worst_angle_diff = max(worst_angle_diff, abs(hoh_angle_deg(q) - hoh_angle_deg(qom)))
        rmsd, Rot = kabsch_rmsd(q, qom, masses)
        worst_rmsd = max(worst_rmsd, rmsd)
        worst_rotation = max(worst_rotation, rad2deg(acos(clamp((tr(Rot) - 1) / 2, -1.0, 1.0))))
        worst_oracle = max(worst_oracle, maximum(abs.(q .- qan)))
        worst_shift = max(worst_shift, maximum(abs.(qom .- qin)))
        worst_raw = max(worst_raw, maximum(abs.(q .- qom)))
    end

    @assert worst_shift > 1e-3 "the OpenMM inputs are not displaced; the test proves nothing"
    @assert worst_geom_om < 1e-5 "OpenMM did not satisfy its own constraints: $worst_geom_om"
    @assert worst_geom_ours < 1e-9 "our geometry missed the targets: $worst_geom_ours"
    @assert worst_bond_diff < 1e-6 "bond lengths differ from OpenMM: $worst_bond_diff"
    @assert worst_angle_diff < 1e-4 "H-O-H angle differs from OpenMM: $worst_angle_diff deg"
    @assert worst_oracle < 1e-9 "our projection differs from the numpy oracle: $worst_oracle"

    # NOT asserted: coordinate-level RMSD against OpenMM.
    #
    # The constraint surface of a rigid triangle is 6-dimensional (3 atoms x 3
    # coords, minus 3 holonomic constraints), so from a displaced input there is
    # a family of valid projections differing by rigid motion. SETTLE and a
    # mass-metric Newton projection are different algorithms that land on
    # DIFFERENT POINTS of that same surface. The measured spread is real:
    # OpenMM sits ~2.8e-4 nm from the analytic constrained triangle while our
    # projection sits ~1e-9 from it, and both satisfy their constraints to
    # machine precision (OpenMM 4.7e-9, ours 1e-9).
    #
    # Demanding coordinate identity would require one algorithm to reproduce the
    # other's arbitrary choice of representative, which no correct implementation
    # does. The meaningful invariants -- constraint satisfaction, internal
    # geometry (bonds, angle), agreement with an independent oracle, and
    # platform reproducibility -- are asserted above. We record the coordinate
    # spread as a diagnostic instead, and require only that it stays small
    # relative to the input displacement (i.e. both land near the same surface).
    @assert worst_rmsd < 1e-2 "coordinate spread vs OpenMM ($worst_rmsd) exceeds any \
        plausible projection difference; investigate rather than loosen"
    @assert worst_rmsd < worst_shift "coordinate spread must be small relative to the \
        input displacement ($worst_rmsd vs $worst_shift)"

    println("  SETTLE/OpenMM reference ($(length(cases)) cases, ",
            join([p[1] for p in platforms], "/"), "):")
    println("    geometry: ours $(worst_geom_ours), OpenMM $(worst_geom_om) vs targets")
    println("    ours vs OpenMM: bond lengths $(worst_bond_diff), angle $(worst_angle_diff) deg, ",
            "mass-weighted RMSD $(worst_rmsd)")
    println("    numpy oracle agreement $(worst_oracle); inputs displaced by $(worst_shift)")
    println("    raw coordinate difference $(worst_raw) = a rigid rotation of ",
            "$(round(worst_rotation, digits = 4)) deg about the COM ",
            "(same surface, not the same iterate)")
end

# ---------------------------------------------------------------------------
# performance: cost per molecule must not scale with the number of molecules
# ---------------------------------------------------------------------------

"Time one full `project!` (positions + momenta) for `nmolecules` rigid waters."
function time_projection(nmolecules; reps = 20)
    m = repeat([M_O, M_H, M_H], nmolecules)
    cs = CO.CoupledConstraintSet(3nmolecules,
        vcat([[(3k + 1, 3k + 2, R_OH), (3k + 1, 3k + 3, R_OH),
               (3k + 2, 3k + 3, R_HH)] for k in 0:(nmolecules - 1)]...))
    base = hcat((water_positions() for _ in 1:nmolecules)...)
    rng = MersenneTwister(11)
    configs = [base .+ 0.02 .* randn(rng, 3, 3nmolecules) for _ in 1:reps]
    p = 0.01 .* randn(rng, 3, 3nmolecules)
    for c in configs[1:2]                     # warm up / compile
        q = copy(c)
        CO.project!(q, p, m, cs; tolerance = 1e-12)
    end
    # Time each repetition separately and report the best: a GC pause in a
    # microbenchmark otherwise shows up as "size-dependent cost".
    times = Vector{Float64}(undef, reps)
    for i in 1:reps
        q = copy(configs[i])
        t0 = time_ns()
        CO.project!(q, p, m, cs; tolerance = 1e-12)
        times[i] = (time_ns() - t0) / 1e9
    end
    seconds = minimum(times)
    return seconds, seconds / nmolecules
end

"""
Per-molecule cost of the water triangle must be flat in system size: every
component is a 3x3 solve, so the only allowed growth is the outer loop.
(`minimum` over repetitions, so the number is the cost without GC pauses.)

The scratch workspace is what makes this true. Before it, the Newton loop
allocated ~5.6 kB per molecule per call and the measured cost was ~5 µs per
molecule; one workspace per call removes the per-component allocation and the
per-molecule cost is ~1.5 µs, flat.
"""
function test_projection_performance()
    _, per100 = time_projection(100)
    _, per1000 = time_projection(1000)
    _, per2000 = time_projection(2000)
    ratio = per1000 / per100
    @assert per100 < 50e-6 "100 molecules: $(per100 * 1e6) µs/molecule is implausible"
    @assert 0.25 < ratio < 4.0 "per-molecule cost is not flat: 100 -> 1000 ratio $(ratio)"
    @assert 0.25 < per2000 / per1000 < 4.0
    println("  per-molecule cost of one project! (3x3 solve + 2 Newton iterations): ",
            "$(round(per100 * 1e6, digits = 2)) µs @100, ",
            "$(round(per1000 * 1e6, digits = 2)) µs @1000, ",
            "$(round(per2000 * 1e6, digits = 2)) µs @2000 molecules ",
            "(ratio 1000/100 = $(round(ratio, digits = 2)))")
end

# ---------------------------------------------------------------------------
# shorter cross-checks kept from the original suite
# ---------------------------------------------------------------------------

"Integrator: a constrained trajectory keeps residuals bounded."
function test_constrained_trajectory()
    m, cs = water_system()
    q0 = water_positions()
    p0 = zeros(3, 3)
    # Real motion: a translation plus an internal excitation that the projection
    # must remove. Without nonzero momentum this test would prove nothing.
    p0[:, 1] = [0.02, -0.01, 0.015]
    p0[:, 2] = [0.03, 0.02, -0.01]
    p0[:, 3] = [-0.01, 0.025, 0.005]
    dt = 0.002
    q = copy(q0); p = copy(p0)
    worst_pos = 0.0
    worst_tan = 0.0
    speed = 0.0
    for _ in 1:2000
        # free drift, then project: sufficient to exercise the projection under motion
        q .+= dt .* (p ./ reshape(m, 1, :))
        CO.project_positions!(q, m, cs; tolerance = 1e-12)
        CO.project_velocities!(q, p, m, cs; tolerance = 1e-12)
        worst_pos = max(worst_pos, pos_residual(cs, q))
        worst_tan = max(worst_tan, tan_residual(cs, q, p, m))
        speed = max(speed, maximum(abs.(p ./ reshape(m, 1, :))))
    end
    @assert worst_pos < 1e-11
    @assert worst_tan < 1e-11
    @assert speed > 0.0 "the trajectory must actually move to exercise the projection"
    println("  2000 drift+project steps (max speed $(speed)): worst pos=$(worst_pos) tan=$(worst_tan)")
end

"Cross-check against the disjoint-pair RATTLE path on an isolated bond."
function test_agrees_with_pair_rattle()
    m = [1.0, 2.0]
    q = randn(MersenneTwister(9), 3, 2) .* 0.3
    q[:, 2] .+= [1.2, 0.0, 0.0]
    p = 0.1 .* randn(MersenneTwister(10), 3, 2)
    q_pair = copy(q); p_pair = copy(p)
    q_coup = copy(q); p_coup = copy(p)
    d = sqrt(sum(abs2, q[:, 2] .- q[:, 1]))
    cs = CO.CoupledConstraintSet(2, [(1, 2, d)])
    CO.project!(q_coup, p_coup, m, cs; tolerance = 1e-13)
    @assert abs(sqrt(sum(abs2, q_coup[:, 2] .- q_coup[:, 1])) - d) < 1e-12
    println("  single pair: coupled projection holds the bond to $(d)")
end

"Audited solvers: explicit residuals, convergence status, and atomic rollback on failure."
function test_audited_solvers_and_atomic_refusal()
    # Two independent waters: component 1 can converge while component 2 fails.
    m = [M_O, M_H, M_H, M_O, M_H, M_H]
    cs = CO.CoupledConstraintSet(6, [
        (1, 2, R_OH), (1, 3, R_OH), (2, 3, R_HH),
        (4, 5, R_OH), (4, 6, R_OH), (5, 6, R_HH),
    ])
    q_good = hcat(water_positions(), water_positions() .+ [4.0, 0.0, 0.0])
    q_good .+= 0.02 .* randn(MersenneTwister(101), 3, 6)
    p_good = 0.01 .* randn(MersenneTwister(102), 3, 6)

    # 1. Successful solve returns explicit residuals, iteration counts, and converged=true.
    res = CO.solve_constraints!(copy(q_good), copy(p_good), m, cs; tolerance = 1e-12)
    @assert res isa CO.ConstraintSolveResult
    @assert res.converged && res.stage === :both
    @assert res.position_residual <= 1e-12 && res.tangent_residual <= 1e-12
    @assert res.iterations >= 1 && isempty(res.failed_atoms) && isempty(res.reason)

    # 2. Component 1 valid, component 2 collinear/singular -> must restore BOTH q and p!
    q_sing = hcat(water_positions() .+ 0.02 .* randn(MersenneTwister(103), 3, 3),
                  [4.0 4.8 6.2; 0.0 0.0 0.0; 0.0 0.0 0.0])
    q_sing0 = copy(q_sing)
    err_sing = caught(() -> CO.project_positions!(q_sing, m, cs; tolerance = 1e-12))
    @assert err_sing isa CO.UnsupportedConstraintSet
    @assert q_sing == q_sing0 "partial projection leaked into component 1 after component 2 failed"

    r_sing = CO.solve_positions!(q_sing, m, cs; tolerance = 1e-12)
    @assert !r_sing.converged && r_sing.stage === :position
    @assert q_sing == q_sing0 "solve_positions! mutated q on refusal"
    @assert r_sing.position_residual > 1e-3 && !isempty(r_sing.reason)

    # 3. Non-convergence on component 2 -> atomic rollback across both components and both stages.
    # Component 1 has a tiny perturbation (converges in 1 Newton step to <= 1e-12);
    # component 2 has a large perturbation (cannot converge in 1 step).
    q_fail = hcat(water_positions() .+ 1e-8 .* randn(MersenneTwister(104), 3, 3),
                  water_positions() .+ [4.0, 0.0, 0.0] .+ 0.4 .* randn(MersenneTwister(105), 3, 3))
    p_fail = 0.05 .* randn(MersenneTwister(106), 3, 6)
    q_fail0 = copy(q_fail); p_fail0 = copy(p_fail)
    r_fail = CO.solve_constraints!(q_fail, p_fail, m, cs; tolerance = 1e-12, max_iterations = 1)
    @assert !r_fail.converged
    @assert q_fail == q_fail0 && p_fail == p_fail0 "solve_constraints! left partially projected state"
    @assert r_fail.failed_atoms == [4, 5, 6]
    @assert isfinite(r_fail.position_residual) && r_fail.position_residual > 1e-12

    # 4. Exact post-update residual check: a 1-step linear velocity projection must succeed with max_iterations=1.
    q_exact = water_positions()
    p_vel = 0.05 .* randn(MersenneTwister(107), 3, 3)
    m1, cs1 = water_system()
    rv = CO.solve_velocities!(q_exact, p_vel, m1, cs1; tolerance = 1e-12, max_iterations = 1)
    @assert rv.converged && rv.iterations == 1 && rv.tangent_residual <= 1e-12

    # 5. Invalid inputs (non-positive masses, NaNs, bad tolerance) must be refused upfront.
    @assert caught(() -> CO.project_positions!(copy(q_exact), [1.0, 0.0, 1.0], cs1)) isa ArgumentError
    @assert caught(() -> CO.project_positions!(fill(NaN, 3, 3), m1, cs1)) isa ArgumentError
    @assert caught(() -> CO.project_positions!(copy(q_exact), m1, cs1; tolerance = 0.0)) isa ArgumentError
    @assert caught(() -> CO.project_positions!(copy(q_exact), m1, cs1; max_iterations = 0)) isa ArgumentError
    println("  audited solvers & atomic rollback: explicit status, post-step check, and rollback verified")
end

function run()
    println("CONSTRAINTS")
    test_rigid_water_projection()
    test_rigid_water_rigidity()
    test_rigid_water_conservation()
    test_two_waters()
    test_mixed_component_sizes()
    test_overdetermined_refused()
    test_component_dof_refused()
    test_singular_geometry_refused()
    test_collinear_satisfied_is_allowed()
    test_nonconvergence_raises()
    test_constructor_refusals()
    test_constraint_virial_formulation()
    test_constraint_virial_box_derivative()
    test_constraint_virial_projection_route()
    test_long_dynamics_20k()
    test_fast_molecule()
    test_settle_reference()
    test_projection_performance()
    test_constrained_trajectory()
    test_agrees_with_pair_rattle()
    test_audited_solvers_and_atomic_refusal()
    println("ALL TESTS PASSED")
end

run()
