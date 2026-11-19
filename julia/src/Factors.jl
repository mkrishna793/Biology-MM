"""
Factor protocol: the compositional core.

`U(q) = Σ_α U_α( q|_{S_α} )` (FRAMEWORK.md §3). Each factor has a static
read-support `S_α` and a local energy. A factor implements

    evaluate!(F, f, q, ctx) -> Float64

which returns its local energy and *scatters* its own gradient into `F` (the
`3 x N` force accumulator, sign convention `F = -∂U/∂q`). That scatter is a
hand-written reverse-mode sweep (VJP) for the local map, so the total cost is
`O(Σ_α |S_α|)` and no `N x N` object is ever formed.

Every factor is independently checked in `test/test_factors.jl` for

1. **adjoint consistency** — the scattered `F` equals `-∇U` to finite-difference
   accuracy, *including* at the cutoff for switched potentials; and
2. **support discipline** — the factor writes only into its own read-support,
   so disjoint factors cannot overwrite each other's contributions.

CDC connection, stated precisely: this protocol is a concrete instance of the
sum rule and chain rule that a Cartesian Differential Category abstracts, and
the scatter is the adjoint (reverse differential) of the local map. What is
*implemented* is the concrete pipeline. What is *proved* in Lean is the
algebraic identity for one model. The category theory is design guidance here,
not a machine-checked category implementation.
"""
module Factors

using ..Types
using ..Types: SVector3, norm2, unwrap_displacement, AbstractBox, NoBox

export AbstractFactor, read_support, evaluate!, total_energy_forces,
       HarmonicBond, HarmonicAngle, PeriodicDihedral, LennardJonesPair,
       CoulombPair, BipartiteHarmonicGroup, factor_support_size, is_local,
       force_consistent, Context, lj_energy, lj_dudr

abstract type AbstractFactor end

"Number of atoms this factor reads. Static, independent of configuration."
function factor_support_size end

"True when the factor's interaction set is a fixed list (no spatial predicate)."
is_local(::AbstractFactor) = true

"True when the factor's scattered force is exactly the derivative of its energy."
force_consistent(::AbstractFactor) = true

# ---------------------------------------------------------------------------
# evaluation context
# ---------------------------------------------------------------------------

"""
Evaluation context: the box (for minimum image) and the unit system (for the
Coulomb prefactor). Deliberately immutable and small — the neighbour structure
is passed to the *drivers* that enumerate pairs, not baked into every factor, so
a factor never has to reason about rebuild policy.
"""
struct Context
    box::AbstractBox
    units::Units
    N::Int
end
Context(units::Units, N::Int) = Context(NoBox(), units, N)

# ---------------------------------------------------------------------------
# bonded factors (fixed graph, exact pair evaluation, no summary)
# ---------------------------------------------------------------------------

"`U = (k/2)(r - r0)^2`, `r = |q_i - q_j|`. The force is a VECTOR, not a scalar."
struct HarmonicBond <: AbstractFactor
    i::Int; j::Int; k::Float64; r0::Float64
end
factor_support_size(::HarmonicBond) = 2
read_support(f::HarmonicBond) = (f.i, f.j)

function evaluate!(F::AbstractMatrix{Float64}, f::HarmonicBond, q::AbstractMatrix{Float64}, ctx::Context)
    d = unwrap_displacement(ctx.box, q, f.i, f.j)
    r2 = norm2(d)
    # r -> 0 is a singular configuration; preparation must refuse it. Returning
    # Inf here makes the failure diagnosable instead of silently poisoning a run.
    r2 < 1e-24 && return Inf
    r = sqrt(r2)
    dr = r - f.r0
    u = 0.5 * f.k * dr * dr
    # dU/dq_i = k*dr*(d/r)  =>  F_i = -k*dr*d/r
    c = -f.k * dr / r
    @inbounds for m in 1:3
        g = c * _comp(d, m)
        F[m, f.i] += g
        F[m, f.j] -= g
    end
    return u
end

"`U = (kθ/2)(θ - θ0)^2` for the angle at `j` between `i` and `k`."
struct HarmonicAngle <: AbstractFactor
    i::Int; j::Int; k::Int; ktheta::Float64; theta0::Float64
end
factor_support_size(::HarmonicAngle) = 3
read_support(f::HarmonicAngle) = (f.i, f.j, f.k)

function evaluate!(F::AbstractMatrix{Float64}, f::HarmonicAngle, q::AbstractMatrix{Float64}, ctx::Context)
    u1 = unwrap_displacement(ctx.box, q, f.i, f.j)
    u2 = unwrap_displacement(ctx.box, q, f.k, f.j)
    r1 = sqrt(norm2(u1)); r2 = sqrt(norm2(u2))
    (r1 < 1e-12 || r2 < 1e-12) && return Inf
    ct = clamp(_dot(u1, u2) / (r1 * r2), -1.0, 1.0)
    th = acos(ct)
    dth = th - f.theta0
    u = 0.5 * f.ktheta * dth * dth
    st = sin(th)
    # θ = 0 or π is a cusp of θ(q): sin θ = 0, gradient undefined. Reject.
    abs(st) < 1e-12 && return Inf
    # dθ/dq_i = -(1/(r1 r2 sinθ)) (u2 - ct (r2/r1) u1)
    # dθ/dq_k = -(1/(r1 r2 sinθ)) (u1 - ct (r1/r2) u2)
    # dθ/dq_j = -(dθ/dq_i + dθ/dq_k)
    c = -1.0 / (r1 * r2 * st)
    e1 = SVector3(u2.x - ct * (r2 / r1) * u1.x,
                  u2.y - ct * (r2 / r1) * u1.y,
                  u2.z - ct * (r2 / r1) * u1.z)
    e2 = SVector3(u1.x - ct * (r1 / r2) * u2.x,
                  u1.y - ct * (r1 / r2) * u2.y,
                  u1.z - ct * (r1 / r2) * u2.z)
    dU = f.ktheta * dth   # dU/dθ
    @inbounds for m in 1:3
        a = dU * c * _comp(e1, m)      # dU/dq_i
        b = dU * c * _comp(e2, m)      # dU/dq_k
        F[m, f.i] -= a
        F[m, f.k] -= b
        F[m, f.j] += a + b
    end
    return u
end

"""
`U = kφ [1 + cos(nφ - φ0)]`, proper dihedral about the `j-k` axis.

Gradient of the torsion angle φ is the textbook expression below; it is checked
against central differences of `atan2` in `test/test_factors.jl` rather than
trusted.
"""
struct PeriodicDihedral <: AbstractFactor
    i::Int; j::Int; k::Int; l::Int; kphi::Float64; n::Int; phi0::Float64
end
factor_support_size(::PeriodicDihedral) = 4
read_support(f::PeriodicDihedral) = (f.i, f.j, f.k, f.l)

"Returns `(φ, ∇φ)` or `nothing` at a degenerate (collinear) geometry."
function dihedral_angle_grad(q::AbstractMatrix{Float64}, box::AbstractBox, i, j, k, l)
    b1 = unwrap_displacement(box, q, j, i)
    b2 = unwrap_displacement(box, q, k, j)
    b3 = unwrap_displacement(box, q, l, k)
    n1 = _cross(b1, b2)
    n2 = _cross(b2, b3)
    n1sq = norm2(n1); n2sq = norm2(n2); b2sq = norm2(b2)
    (n1sq < 1e-24 || n2sq < 1e-24 || b2sq < 1e-24) && return nothing
    mb = _cross(n1, n2)
    lb2 = sqrt(b2sq)
    x = _dot(n1, n2)
    y = _dot(mb, b2) / lb2
    phi = atan(y, x)
    gi = _scale(n1, -(lb2 / n1sq))
    gl = _scale(n2, lb2 / n2sq)
    c12 = _dot(b1, b2) / b2sq
    c32 = _dot(b3, b2) / b2sq
    gj = _add(_scale(gi, -(1.0 + c12)), _scale(gl, c32))
    gk = _add(_scale(gi, c12), _scale(gl, -(1.0 + c32)))
    return phi, (gi, gj, gk, gl)
end

function evaluate!(F::AbstractMatrix{Float64}, f::PeriodicDihedral, q::AbstractMatrix{Float64}, ctx::Context)
    t = dihedral_angle_grad(q, ctx.box, f.i, f.j, f.k, f.l)
    t === nothing && return Inf
    phi, (gi, gj, gk, gl) = t
    arg = f.n * phi - f.phi0
    u = f.kphi * (1.0 + cos(arg))
    # F = -dU/dq = kφ n sin(arg) * ∇φ
    mult = f.kphi * f.n * sin(arg)
    @inbounds for m in 1:3
        F[m, f.i] += mult * _comp(gi, m)
        F[m, f.j] += mult * _comp(gj, m)
        F[m, f.k] += mult * _comp(gk, m)
        F[m, f.l] += mult * _comp(gl, m)
    end
    return u
end

# ---------------------------------------------------------------------------
# Lennard-Jones radial functions (shared with the NIST validation driver)
# ---------------------------------------------------------------------------

@inline function lj_energy(ε::Float64, σ::Float64, r::Float64)
    s6 = (σ / r)^6
    return 4.0 * ε * (s6 * s6 - s6)
end

"`dV_LJ/dr`."
@inline function lj_dudr(ε::Float64, σ::Float64, r::Float64)
    s6 = (σ / r)^6
    return 24.0 * ε * (s6 - 2.0 * s6 * s6) / r
end

"""
Lennard-Jones 12-6 pair factor with explicit cutoff convention.

`U = 4ε[(σ/r)^12 - (σ/r)^6]`. The modes are *different potentials*, not
cosmetic choices:

* `:none`   no cutoff; exact all-pairs pair sum.
* `:trunc`  hard truncation. Energy is discontinuous at `rc`, so the force is
            not the derivative of the truncated energy there. Kept because NIST
            reference data is defined this way; `force_consistent` is false and
            the integrator refuses it.
* `:shift`  `U - U(rc)`, force unchanged: `C^0` but not `C^1`.
* `:lfs`    `U - U(rc) - U'(rc)(r - rc)`: `C^1`, force is the exact derivative.
            This is the only mode the production integrator accepts.

Mixing a `:trunc` energy with an `:lfs` force is the classic silent-drift bug.
`force_consistent` exists so the integrator can refuse the mixture structurally.
"""
struct LennardJonesPair <: AbstractFactor
    i::Int; j::Int; ε::Float64; σ::Float64
    mode::Symbol
    rc::Float64
    lj_scale::Float64
end
LennardJonesPair(i, j, ε, σ; mode = :none, rc = Inf, lj_scale = 1.0) =
    LennardJonesPair(i, j, ε, σ, mode, rc, lj_scale)

factor_support_size(::LennardJonesPair) = 2
read_support(f::LennardJonesPair) = (f.i, f.j)
is_local(f::LennardJonesPair) = f.mode === :none
force_consistent(f::LennardJonesPair) = f.mode in (:none, :lfs)

function evaluate!(F::AbstractMatrix{Float64}, f::LennardJonesPair, q::AbstractMatrix{Float64}, ctx::Context)
    d = unwrap_displacement(ctx.box, q, f.i, f.j)
    r2 = norm2(d)
    r2 < 1e-24 && return Inf
    r = sqrt(r2)
    if f.mode !== :none && r > f.rc
        return 0.0
    end
    u = lj_energy(f.ε, f.σ, r)
    dudr = lj_dudr(f.ε, f.σ, r)
    if f.mode === :lfs
        uc = lj_energy(f.ε, f.σ, f.rc)
        duc = lj_dudr(f.ε, f.σ, f.rc)
        u = u - uc - duc * (r - f.rc)
        dudr = dudr - duc
    elseif f.mode === :shift
        u = u - lj_energy(f.ε, f.σ, f.rc)
    elseif f.mode !== :trunc && f.mode !== :none
        error("unknown LennardJonesPair mode $(f.mode)")
    end
    u *= f.lj_scale
    # dU/dq_i = (dU/dr)(d/r)  =>  F_i = -dudr * d / r
    c = -dudr * f.lj_scale / r
    @inbounds for m in 1:3
        g = c * _comp(d, m)
        F[m, f.i] += g
        F[m, f.j] -= g
    end
    return u
end

"""
Plain Coulomb pair `U = pref·λ·z_i z_j / r`.

No cutoff flag: periodic long-range electrostatics is a *different model*
(Ewald/PME or reaction field), not a switch on this factor. Minimum image with a
cutoff is not equivalent to a periodic Coulomb sum (FRAMEWORK.md §5.3), so a
periodic box is refused explicitly with `UnsupportedGeometry`.
"""
struct CoulombPair <: AbstractFactor
    i::Int; j::Int
    qi::Float64; qj::Float64
    coulomb_scale::Float64
end
CoulombPair(i, j, qi, qj; coulomb_scale = 1.0) = CoulombPair(i, j, qi, qj, coulomb_scale)
factor_support_size(::CoulombPair) = 2
read_support(f::CoulombPair) = (f.i, f.j)
is_local(::CoulombPair) = true

# This core has no long-range electrostatic solver. Evaluating a Coulomb pair in
# a periodic box via minimum image would silently substitute a different model,
# so the configuration is refused *before* any force is scattered.
@inline _require_nonperiodic_coulomb(::NoBox) = nothing
@inline function _require_nonperiodic_coulomb(box::AbstractBox)
    throw(UnsupportedGeometry(
        "CoulombPair is non-periodic only: got $(typeof(box)). Minimum-image " *
        "Coulomb is not a periodic Coulomb sum (long-range electrostatics is " *
        "unimplemented), so it is refused rather than silently approximated. " *
        "Use NoBox, or an engine that implements Ewald/PME (this is a " *
        "documented limitation, not a bug)."))
end

function evaluate!(F::AbstractMatrix{Float64}, f::CoulombPair, q::AbstractMatrix{Float64}, ctx::Context)
    _require_nonperiodic_coulomb(ctx.box)
    d = unwrap_displacement(ctx.box, q, f.i, f.j)
    r2 = norm2(d)
    r2 < 1e-24 && return Inf
    r = sqrt(r2)
    pre = ctx.units.coulomb_prefactor * f.coulomb_scale * f.qi * f.qj
    u = pre / r
    # dU/dr = -pre/r^2 => F_i = pre * d / r^3
    c = pre / (r2 * r)
    @inbounds for m in 1:3
        g = c * _comp(d, m)
        F[m, f.i] += g
        F[m, f.j] -= g
    end
    return u
end

# ---------------------------------------------------------------------------
# the one exactly summarised factor
# ---------------------------------------------------------------------------

"""
Complete-bipartite quadratic harmonic group interaction

    U = (k/2) Σ_{i∈A} Σ_{j∈B} ||q_i - q_j||^2

evaluated by the centroid/residual closed form proved exact in
`lean/Foundation/Harmonic.lean`:

    U = (k/2)[ n·m·||a-b||² + m·resid(A,a) + n·resid(B,b) ].

Energy *and* per-atom forces come from the summary, so the pair count drops from
`n·m` to `n+m`. Reading `n+m` coordinates and writing `n+m` forces is still
`Ω(n+m)`: the reduction is in pair evaluations, not per-atom work. Each atom
keeps its own distinct force (Lean `forceOnA_injective`).

This is the only factor here whose nonbonded cost is sub-quadratic *and exactly
certified*. Everything else is direct or explicitly approximate.
"""
struct BipartiteHarmonicGroup <: AbstractFactor
    A::Vector{Int}; B::Vector{Int}; k::Float64
end
factor_support_size(f::BipartiteHarmonicGroup) = length(f.A) + length(f.B)
read_support(f::BipartiteHarmonicGroup) = (f.A..., f.B...)
is_local(f::BipartiteHarmonicGroup) = false

function evaluate!(F::AbstractMatrix{Float64}, f::BipartiteHarmonicGroup, q::AbstractMatrix{Float64}, ctx::Context)
    A, B, k = f.A, f.B, f.k
    n = length(A); m = length(B)
    (n == 0 || m == 0) && return 0.0
    ax = ay = az = 0.0
    @inbounds for i in A
        ax += q[1, i]; ay += q[2, i]; az += q[3, i]
    end
    ax /= n; ay /= n; az /= n
    bx = by = bz = 0.0
    @inbounds for j in B
        bx += q[1, j]; by += q[2, j]; bz += q[3, j]
    end
    bx /= m; by /= m; bz /= m
    ra = 0.0
    @inbounds for i in A
        dx = q[1, i] - ax; dy = q[2, i] - ay; dz = q[3, i] - az
        ra += dx * dx + dy * dy + dz * dz
    end
    rb = 0.0
    @inbounds for j in B
        dx = q[1, j] - bx; dy = q[2, j] - by; dz = q[3, j] - bz
        rb += dx * dx + dy * dy + dz * dz
    end
    ab = (ax - bx)^2 + (ay - by)^2 + (az - bz)^2
    u = 0.5 * k * (n * m * ab + m * ra + n * rb)
    cm = -k * m
    cn = -k * n
    @inbounds for i in A
        F[1, i] += cm * (q[1, i] - bx)
        F[2, i] += cm * (q[2, i] - by)
        F[3, i] += cm * (q[3, i] - bz)
    end
    @inbounds for j in B
        F[1, j] += cn * (q[1, j] - ax)
        F[2, j] += cn * (q[2, j] - ay)
        F[3, j] += cn * (q[3, j] - az)
    end
    return u
end

# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

"""
`total_energy_forces(factors, q, ctx)` returns `(U, F)` with `F = -∂U/∂q`.

The single VJP accumulation point. `F` may be supplied to avoid allocation; it is
always zeroed first, because a stale accumulator is a silent double-count.
"""
function total_energy_forces(factors, q::AbstractMatrix{Float64}, ctx::Context;
                             F::Union{Nothing,Matrix{Float64}} = nothing)
    Fbuf = F === nothing ? zeros(3, ctx.N) : fill!(F, 0.0)
    U = 0.0
    @inbounds for f in factors
        U += evaluate!(Fbuf, f, q, ctx)
    end
    return U, Fbuf
end

# ---------------------------------------------------------------------------
# vector helpers
# ---------------------------------------------------------------------------

@inline _comp(v::SVector3, m::Int) = m == 1 ? v.x : (m == 2 ? v.y : v.z)
@inline _dot(a::SVector3, b::SVector3) = a.x * b.x + a.y * b.y + a.z * b.z
@inline _scale(a::SVector3, c::Float64) = SVector3(c * a.x, c * a.y, c * a.z)
@inline _add(a::SVector3, b::SVector3) = SVector3(a.x + b.x, a.y + b.y, a.z + b.z)
@inline _cross(a::SVector3, b::SVector3) =
    SVector3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)

end # module Factors
