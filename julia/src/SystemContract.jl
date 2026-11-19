"""
`SystemContract` — the execution contract of a resolved system.

A [`ResolvedSystem`](@ref) is *specification* state: atom identities, box, topology,
factor list, units and declared capabilities. Execution is a separate, explicit
contract implemented here:

* `energy`, `forces`, `energy_forces` (and the `evaluate_system` alias) delegate to
  `Factors.total_energy_forces`, so the factor protocol's calling convention and
  sign convention (`F = -∂U/∂q`) are preserved bit-for-bit.
* `component_energies` evaluates each factor against its own private force
  accumulator, so the per-component energies are independent and exactly additive.
* `box_derivative` returns the analytic `G[a,b] = ∂U/∂h[a,b]` at **fixed fractional
  coordinates**, with `h` the lattice matrix whose columns are the cell vectors.
  It is only defined where every factor supplies an analytic stress; there is no
  finite-difference substitute and no zero-fill. A missing capability throws
  [`MissingBoxDerivative`](@ref) naming the component and the reason.

## Sign conventions (exactly these)

* `F = -∂U/∂q`, `energy_forces` returns `(U, F)`.
* Periodic `h`: `OrthorhombicBox` -> `Diagonal(L)`, `TriclinicBox` -> `H`; the
  columns are lattice vectors and `r_i = h s_i`.
* `box_derivative` returns `G` with `G[a,b] = ∂U/∂h[a,b]` at fixed `s`.
* `virial(rs,q)` returns `Ξ = -h*G'`, so `Ξ[a,b] = Σ_i r_i[a] F_i[b]`.
* `affine_pressure(rs,q) = -tr(G*h')/(3V)`, the fixed-fractional affine value.
* `pressure(rs,q,p; extra_virial=0.0) = (2K + tr(Ξ) + extra_virial)/(3V)`, with
  `K = 0.5 Σ p²/m`. `extra_virial` is where a caller adds a constraint virial; with
  the default the value is the *unconstrained* affine pressure.

## Refusals, stated plainly

* Virtual sites are representable (mass-0 declared sites) but execution is refused;
  every energy/force/pressure entry point throws
  `ResolvedSystem.UnsupportedCapability(:virtual_sites, ...)`.
* `EwaldFactor` and `CoulombPair` have no analytic periodic box derivative here,
  so `box_derivative`, `virial`, `affine_pressure` and `pressure` refuse a
  system that contains one. A local bond in the same system can still be
  differentiated with `box_derivative_component`. `PMEFactor` and
  `StreamedLJFactor` (`:lfs`) supply analytic box derivatives on both
  `OrthorhombicBox` and `TriclinicBox`.
* `NoBox` systems have no lattice matrix: fractional coordinates, box derivatives
  and pressures refuse rather than pretending the derivative is zero.
"""
module SystemContract

using ..Types
using ..Factors
using ..Constraints
using ..StreamedLJ
using ..Electrostatics
using ..PME
using LinearAlgebra

import ..ResolvedSystem
import ..ResolvedSystem: accumulate_box_derivative!, box_derivative,
                          UnsupportedChemistry, InvalidGeometry, NumericalFailure

"Short alias for the module; the struct inside it has the same name."
const RS = ResolvedSystem

export MissingBoxDerivative, energy, forces, energy_forces, component_energies,
       component_energy, box_derivative, box_derivative_component,
       accumulate_box_derivative!,
       supports_box_derivative, fractional_coordinates, cartesian_coordinates,
       fractional_derivative, virial, pressure, kinetic_energy, affine_pressure,
       resolved_context, evaluate_system, resolved_from_system, to_atomset,
       check_numerical_stability,
       UnsupportedChemistry, InvalidGeometry, NumericalFailure,
       place_virtual_sites!, place_virtual_sites, redistribute_virtual_site_forces!,
       cartesian_forces!, energy_and_forces!, box_derivative!

# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

"""
Raised when an analytic box derivative is requested for a component (or a whole
system) that does not have one.

`component` names the offending component (`:system` for the total), `reason`
states why. It is never acceptable to return zeros or a finite-difference
approximation instead: a pressure built on a partial stress is a wrong number with
no diagnostic.
"""
struct MissingBoxDerivative <: Exception
    component::Symbol
    reason::String
end

Base.showerror(io::IO, e::MissingBoxDerivative) =
    print(io, "MissingBoxDerivative[", e.component, "]: ", e.reason)

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

"Lattice matrix: columns are the cell vectors."
_hmat(b::Types.OrthorhombicBox) =
    [b.L[1] 0.0 0.0; 0.0 b.L[2] 0.0; 0.0 0.0 b.L[3]]
_hmat(b::Types.TriclinicBox) = copy(b.H)
_hmat(::Types.NoBox) =
    throw(RS.UnsupportedCapability(:box_matrix,
        "no periodic box: there is no lattice matrix, so fractional coordinates, " *
        "box derivatives and pressures are undefined"))

"`Hinv_d * d` as an `NTuple{3,Float64}` — the fractional displacement."
@inline function _frac_disp(b::Types.OrthorhombicBox, d::Types.SVector3)
    L = b.L
    return (d.x / L[1], d.y / L[2], d.z / L[3])
end

@inline function _frac_disp(b::Types.TriclinicBox, d::Types.SVector3)
    M = b.Hinv
    return (M[1, 1] * d.x + M[1, 2] * d.y + M[1, 3] * d.z,
            M[2, 1] * d.x + M[2, 2] * d.y + M[2, 3] * d.z,
            M[3, 1] * d.x + M[3, 2] * d.y + M[3, 3] * d.z)
end

@inline _frac_disp(::Types.NoBox, ::Types.SVector3) =
    throw(RS.UnsupportedCapability(:fractional_coordinates,
        "no periodic box: fractional displacements need lattice vectors"))

@inline _vadd(a::NTuple{3,Float64}, b::NTuple{3,Float64}) =
    (a[1] + b[1], a[2] + b[2], a[3] + b[3])

"`G += c * d * w'` (outer product), component form."
@inline function _outer_add!(G::Matrix{Float64}, c::Float64,
                             dx::Float64, dy::Float64, dz::Float64,
                             wx::Float64, wy::Float64, wz::Float64)
    gx = c * dx; gy = c * dy; gz = c * dz
    @inbounds begin
        G[1, 1] += gx * wx; G[1, 2] += gx * wy; G[1, 3] += gx * wz
        G[2, 1] += gy * wx; G[2, 2] += gy * wy; G[2, 3] += gy * wz
        G[3, 1] += gz * wx; G[3, 2] += gz * wy; G[3, 3] += gz * wz
    end
    return G
end

@inline _outer_add!(G::Matrix{Float64}, c::Float64, d::Types.SVector3,
                    w::NTuple{3,Float64}) =
    _outer_add!(G, c, d.x, d.y, d.z, w[1], w[2], w[3])

function _check_positions(rs::RS.ResolvedSystem, q::AbstractMatrix{Float64})
    N = RS.n_atoms(rs)
    size(q) == (3, N) ||
        throw(ArgumentError("positions must be 3 x $(N), got $(size(q))"))
    all(isfinite, q) ||
        throw(InvalidGeometry("positions must be finite, got non-finite coordinate in q"))
    return nothing
end

function _check_momenta(rs::RS.ResolvedSystem, p::AbstractMatrix{Float64})
    N = RS.n_atoms(rs)
    size(p) == (3, N) ||
        throw(ArgumentError("momenta must be 3 x $(N), got $(size(p))"))
    all(isfinite, p) ||
        throw(InvalidGeometry("momenta must be finite, got non-finite coordinate in p"))
    return nothing
end

"""
    check_numerical_stability(u::Float64, F::AbstractMatrix{Float64})
    check_numerical_stability(rs::ResolvedSystem, q::AbstractMatrix{Float64})

Verify that evaluated potential energy and forces are finite.
Throws `NumericalFailure` if NaN or Inf is encountered.
"""
function check_numerical_stability(u::Float64, F::AbstractMatrix{Float64})
    isfinite(u) || throw(NumericalFailure("potential energy evaluated to non-finite value: $u"))
    all(isfinite, F) || throw(NumericalFailure("forces contain non-finite values"))
    return true
end

function check_numerical_stability(rs::RS.ResolvedSystem, q::AbstractMatrix{Float64})
    u, F = energy_forces(rs, q)
    return check_numerical_stability(u, F)
end

"""
Refuse execution for a system that declares virtual sites unless declared. Virtual sites are
specification state unless :virtual_sites is declared in capabilities.
"""
function _refuse_virtual_sites(rs::RS.ResolvedSystem)
    if !isempty(rs.atoms.virtual_sites) && !(:virtual_sites in rs.declared)
        throw(RS.UnsupportedCapability(:virtual_sites,
            "ResolvedSystem contains $(length(rs.atoms.virtual_sites)) virtual site(s); this core stores the " *
            "representation only; execution is not implemented. Include :virtual_sites in declared capabilities to enable virtual site placement and force redistribution"))
    end
    return nothing
end

"""
    place_virtual_sites!(q::AbstractMatrix{Float64}, rs::RS.ResolvedSystem) -> q

Compute and place the Cartesian coordinates of all declared virtual sites in `q` in-place
using parent coordinates reconstructed under minimum-image periodic boundary conditions.
Returns `q`.
"""
function place_virtual_sites!(q::AbstractMatrix{Float64}, rs::RS.ResolvedSystem)
    for vs in rs.atoms.virtual_sites
        p1 = vs.parents[1]
        rx = q[1, p1]
        ry = q[2, p1]
        rz = q[3, p1]
        for k in 2:length(vs.parents)
            pk = vs.parents[k]
            wk = vs.coefficients[k]
            d = Types.unwrap_displacement(rs.box, q, pk, p1)
            rx += wk * d.x
            ry += wk * d.y
            rz += wk * d.z
        end
        q[1, vs.site] = rx
        q[2, vs.site] = ry
        q[3, vs.site] = rz
    end
    return q
end

"""
    place_virtual_sites(q::AbstractMatrix{Float64}, rs::RS.ResolvedSystem) -> Matrix{Float64}

Return a copy of `q` with declared virtual sites placed from parent coordinates.
"""
place_virtual_sites(q::AbstractMatrix{Float64}, rs::RS.ResolvedSystem) =
    place_virtual_sites!(copy(q), rs)

"""
    redistribute_virtual_site_forces!(F::AbstractMatrix{Float64}, rs::RS.ResolvedSystem) -> F

Redistribute forces on virtual sites onto parent atoms according to the chain rule:
    F[:, p_k] += w_k * F[:, vs.site],   F[:, vs.site] .= 0
Iterates through virtual sites in reverse order so nested sites redistribute correctly.
Returns `F`.
"""
function redistribute_virtual_site_forces!(F::AbstractMatrix{Float64}, rs::RS.ResolvedSystem)
    for vs in Iterators.reverse(rs.atoms.virtual_sites)
        s = vs.site
        fx = F[1, s]; fy = F[2, s]; fz = F[3, s]
        for (p, w) in zip(vs.parents, vs.coefficients)
            F[1, p] += w * fx
            F[2, p] += w * fy
            F[3, p] += w * fz
        end
        F[1, s] = 0.0
        F[2, s] = 0.0
        F[3, s] = 0.0
    end
    return F
end

"Evaluation context for the factor protocol."
resolved_context(rs::RS.ResolvedSystem) =
    Factors.Context(rs.box, rs.units, RS.n_atoms(rs))

# ---------------------------------------------------------------------------
# energy and forces
# ---------------------------------------------------------------------------

"""
    energy(rs, q) -> U

Total potential energy. Delegates to `Factors.total_energy_forces`, so the factor
protocol (including minimum-image geometry and the unit system's Coulomb prefactor)
is used unchanged.
"""
function energy(rs::RS.ResolvedSystem, q::AbstractMatrix{Float64})
    _check_positions(rs, q)
    _refuse_virtual_sites(rs)
    q_placed = isempty(rs.atoms.virtual_sites) ? q : place_virtual_sites(q, rs)
    return Factors.total_energy_forces(rs.factors, q_placed, resolved_context(rs))[1]
end

"Total Cartesian force `F = -∂U/∂q` as a `3 x N` matrix."
function forces(rs::RS.ResolvedSystem, q::AbstractMatrix{Float64})
    return energy_forces(rs, q)[2]
end

"""
    energy_forces(rs, q; F=nothing) -> (U, F)

`Factors.total_energy_forces(rs.factors, q, context; F=F)` with the resolved
system's box, units and atom count. `F` is always zeroed before the factor sweep,
so a reused buffer cannot double count.
"""
function energy_forces(rs::RS.ResolvedSystem, q::AbstractMatrix{Float64};
                       F::Union{Nothing,Matrix{Float64}} = nothing)
    _check_positions(rs, q)
    _refuse_virtual_sites(rs)
    q_placed = isempty(rs.atoms.virtual_sites) ? q : place_virtual_sites(q, rs)
    u, Fout = Factors.total_energy_forces(rs.factors, q_placed, resolved_context(rs); F = F)
    if !isempty(rs.atoms.virtual_sites)
        redistribute_virtual_site_forces!(Fout, rs)
    end
    return (u, Fout)
end

function energy_and_forces!(F::AbstractMatrix{Float64}, rs::RS.ResolvedSystem, q::AbstractMatrix{Float64})
    u, _ = energy_forces(rs, q; F = F isa Matrix{Float64} ? F : Matrix{Float64}(F))
    return u, F
end

function cartesian_forces!(F::AbstractMatrix{Float64}, rs::RS.ResolvedSystem, q::AbstractMatrix{Float64})
    energy_and_forces!(F, rs, q)
    return F
end

"""
    evaluate_system(rs, q; F=nothing) -> (U, F)
    evaluate_system(rs, q, F) -> (U, F)

Documented alias of [`energy_forces`](@ref). It exists so callers written against
the lower-level `Factors.total_energy_forces(factors, q, ctx; F=F)` convention keep
their `(energy, forces)` unpacking and their in-place buffer reuse unchanged.
"""
evaluate_system(rs::RS.ResolvedSystem, q::AbstractMatrix{Float64};
                F::Union{Nothing,Matrix{Float64}} = nothing) =
    energy_forces(rs, q; F = F)

evaluate_system(rs::RS.ResolvedSystem, q::AbstractMatrix{Float64},
                F::Matrix{Float64}) = energy_forces(rs, q; F = F)

# ---------------------------------------------------------------------------
# named components
# ---------------------------------------------------------------------------

"""
    component_energies(rs, q) -> NamedTuple

Per-component energies keyed by `rs.component_names`. Each factor is evaluated
against its **own** zeroed `3 x N` accumulator, so a component's value cannot depend
on which other components are present, and `sum(values(...))` equals
[`energy`](@ref) (same factors, same order, no shared state).
"""
function component_energies(rs::RS.ResolvedSystem, q::AbstractMatrix{Float64})
    _check_positions(rs, q)
    _refuse_virtual_sites(rs)
    q_placed = isempty(rs.atoms.virtual_sites) ? q : place_virtual_sites(q, rs)
    ctx = resolved_context(rs)
    names = Tuple(rs.component_names)
    values = Vector{Float64}(undef, length(rs.factors))
    @inbounds for k in eachindex(rs.factors)
        Fprivate = zeros(Float64, 3, ctx.N)
        values[k] = Factors.evaluate!(Fprivate, rs.factors[k], q_placed, ctx)
    end
    return NamedTuple{names}(Tuple(values))
end

"""
    component_energy(rs, q, name::Symbol) -> U

Energy of the single component named `name`. Throws
`ResolvedSystem.ResolvedSystemError(:component_name, ...)` when no component has that
name.
"""
function component_energy(rs::RS.ResolvedSystem, q::AbstractMatrix{Float64}, name::Symbol)
    _check_positions(rs, q)
    _refuse_virtual_sites(rs)
    k = findfirst(==(name), rs.component_names)
    k === nothing && throw(RS.ResolvedSystemError(:component_name,
        "no component named $(repr(name)); components are $(rs.component_names)"))
    q_placed = isempty(rs.atoms.virtual_sites) ? q : place_virtual_sites(q, rs)
    Fprivate = zeros(Float64, 3, RS.n_atoms(rs))
    return Factors.evaluate!(Fprivate, rs.factors[k], q_placed, resolved_context(rs))
end

# ---------------------------------------------------------------------------
# fractional coordinates
# ---------------------------------------------------------------------------

"""
    fractional_coordinates(rs, q) -> 3 x N

`s = h⁻¹ q` (`q ./ L` for an orthorhombic box, `Hinv*q` for a triclinic one).
`NoBox` refuses with `UnsupportedCapability(:fractional_coordinates, ...)`.
"""
function fractional_coordinates(rs::RS.ResolvedSystem, q::AbstractMatrix{Float64})
    _check_positions(rs, q)
    b = rs.box
    b isa Types.NoBox && throw(RS.UnsupportedCapability(:fractional_coordinates,
        "no periodic box: fractional coordinates need lattice vectors; use " *
        "Cartesian coordinates for a NoBox system"))
    N = RS.n_atoms(rs)
    s = Matrix{Float64}(undef, 3, N)
    @inbounds for j in 1:N
        x = Types.SVector3(q[1, j], q[2, j], q[3, j])
        w = _frac_disp(b, x)
        s[1, j] = w[1]; s[2, j] = w[2]; s[3, j] = w[3]
    end
    return s
end

"""
    cartesian_coordinates(rs, s) -> 3 x N

`q = h s`. `NoBox` refuses: there is no lattice matrix to apply.
"""
function cartesian_coordinates(rs::RS.ResolvedSystem, s::AbstractMatrix{Float64})
    N = RS.n_atoms(rs)
    size(s) == (3, N) ||
        throw(ArgumentError("fractional coordinates must be 3 x $(N), got $(size(s))"))
    rs.box isa Types.NoBox && throw(RS.UnsupportedCapability(:cartesian_coordinates,
        "no periodic box: fractional coordinates are undefined, so there is no " *
        "lattice matrix to apply; use Cartesian coordinates for a NoBox system"))
    return _hmat(rs.box) * s
end

"""
    fractional_derivative(rs, q; h=1e-6) -> 3 x N

Central differences of [`energy`](@ref) over fractional coordinates, returning
`∂U/∂s`. This is a *diagnostic* used to check the analytic force and the analytic
box derivative; it is never used to build a pressure.
"""
function fractional_derivative(rs::RS.ResolvedSystem, q::AbstractMatrix{Float64};
                               h::Float64 = 1e-6)
    (isfinite(h) && h > 0) ||
        throw(ArgumentError("fractional difference step must be finite and > 0, got $h"))
    _check_positions(rs, q)
    s0 = fractional_coordinates(rs, q)
    N = RS.n_atoms(rs)
    g = Matrix{Float64}(undef, 3, N)
    stencil = copy(s0)
    @inbounds for j in 1:N, a in 1:3
        up = stencil[a, j]
        stencil[a, j] = up + h
        uplus = energy(rs, cartesian_coordinates(rs, stencil))
        stencil[a, j] = up - h
        uminus = energy(rs, cartesian_coordinates(rs, stencil))
        stencil[a, j] = up
        g[a, j] = (uplus - uminus) / (2.0 * h)
    end
    return g
end

# ---------------------------------------------------------------------------
# analytic box derivative: support gate
# ---------------------------------------------------------------------------

"""
    supports_box_derivative(f, box) -> Bool

`true` when `f` has an analytic, force-consistent box derivative *and* the box is
periodic. `EwaldFactor` is orthorhombic-only and refused on a triclinic box even
though the box type itself supports a lattice matrix.
"""
function supports_box_derivative(f::Factors.AbstractFactor, box::Types.AbstractBox)
    return _box_derivative_reason(f, box) === nothing
end

"`nothing` when supported, otherwise the concrete refusal reason."
function _box_derivative_reason(f::Factors.AbstractFactor, box::Types.AbstractBox)
    box isa Types.NoBox && return "no periodic box: the box derivative is undefined " *
                                   "without lattice vectors"
    reason = RS.box_derivative_support(f)
    reason === nothing || return reason
    if box isa Types.TriclinicBox && f isa Electrostatics.EwaldFactor
        return "$(nameof(typeof(f))) supports only an orthorhombic box; no triclinic " *
               "stress is implemented"
    end
    return nothing
end

# ---------------------------------------------------------------------------
# analytic box derivative: per-factor kernels
#
# Every kernel accumulates G[a,b] = ∂U/∂h[a,b] at fixed fractional coordinates.
# For a minimum-image displacement `d = unwrap_displacement(box, q, i, j) = h w`
# with `w = Hinv_d d` (the fixed fractional displacement), each term of the local
# energy contributes via ∂U/∂h[a,b] = (∂U/∂d)_a w_b.
# ---------------------------------------------------------------------------

function _accumulate_box_derivative!(G::Matrix{Float64}, f::Factors.AbstractFactor,
                                     q::AbstractMatrix{Float64},
                                     ctx::Factors.Context, name::Symbol)
    if RS._has_factor_box_derivative(f)
        if hasmethod(RS.accumulate_box_derivative!, Tuple{Matrix{Float64}, typeof(f), Matrix{Float64}, Factors.Context}) ||
           hasmethod(RS.accumulate_box_derivative!, Tuple{Matrix{Float64}, typeof(f), AbstractMatrix{Float64}, Factors.Context})
            return RS.accumulate_box_derivative!(G, f, q, ctx)
        elseif hasmethod(RS.box_derivative, Tuple{typeof(f), Matrix{Float64}, Factors.Context}) ||
               hasmethod(RS.box_derivative, Tuple{typeof(f), AbstractMatrix{Float64}, Factors.Context})
            G .+= RS.box_derivative(f, q, ctx)
            return G
        end
    end
    throw(MissingBoxDerivative(name,
        "no analytic box derivative implemented for $(nameof(typeof(f)))"))
end

# U = (k/2)(r-r0)^2 with d = h w:  dU/dh = (k dr / r) d w'.
function RS.accumulate_box_derivative!(G::Matrix{Float64}, f::Factors.HarmonicBond,
                                      q::AbstractMatrix{Float64},
                                      ctx::Factors.Context)
    d = Types.unwrap_displacement(ctx.box, q, f.i, f.j)
    r2 = Types.norm2(d)
    r = sqrt(r2)
    dr = r - f.r0
    _outer_add!(G, f.k * dr / r, d, _frac_disp(ctx.box, d))
    return G
end

_accumulate_box_derivative!(G::Matrix{Float64}, f::Factors.HarmonicBond,
                            q::AbstractMatrix{Float64}, ctx::Factors.Context, ::Symbol) =
    RS.accumulate_box_derivative!(G, f, q, ctx)

# U = (kθ/2)(θ-θ0)^2. With u1 = h w1, u2 = h w2 and ∂θ/∂q_i = c e1, ∂θ/∂q_k = c e2
# (c, e1, e2 exactly as Factors.evaluate!), the fixed-fractional box derivative is
# kθ dθ (c e1) w1' + kθ dθ (c e2) w2'.
function RS.accumulate_box_derivative!(G::Matrix{Float64}, f::Factors.HarmonicAngle,
                                      q::AbstractMatrix{Float64},
                                      ctx::Factors.Context)
    u1 = Types.unwrap_displacement(ctx.box, q, f.i, f.j)
    u2 = Types.unwrap_displacement(ctx.box, q, f.k, f.j)
    r1 = sqrt(Types.norm2(u1))
    r2 = sqrt(Types.norm2(u2))
    # A singular geometry (r -> 0) or a cusp (theta = 0 or pi) makes `c` infinite, so
    # `G` becomes non-finite. That is the derivative's analogue of evaluation
    # returning `Inf`: loud, never a silent zero.
    ct = clamp((u1.x * u2.x + u1.y * u2.y + u1.z * u2.z) / (r1 * r2), -1.0, 1.0)
    th = acos(ct)
    dth = th - f.theta0
    st = sin(th)
    c = -1.0 / (r1 * r2 * st)
    e1 = Types.SVector3(u2.x - ct * (r2 / r1) * u1.x,
                        u2.y - ct * (r2 / r1) * u1.y,
                        u2.z - ct * (r2 / r1) * u1.z)
    e2 = Types.SVector3(u1.x - ct * (r1 / r2) * u2.x,
                        u1.y - ct * (r1 / r2) * u2.y,
                        u1.z - ct * (r1 / r2) * u2.z)
    coef = f.ktheta * dth * c
    _outer_add!(G, coef, e1, _frac_disp(ctx.box, u1))
    _outer_add!(G, coef, e2, _frac_disp(ctx.box, u2))
    return G
end

_accumulate_box_derivative!(G::Matrix{Float64}, f::Factors.HarmonicAngle,
                            q::AbstractMatrix{Float64}, ctx::Factors.Context, ::Symbol) =
    RS.accumulate_box_derivative!(G, f, q, ctx)

# U = kφ[1 + cos(nφ - φ0)], mult = kφ n sin(nφ - φ0) = -dU/dφ.
# b1 = q_j-q_i = h w1, b2 = q_k-q_j = h w2, b3 = q_l-q_k = h w3, so with the gauge
# s_i = 0 the fixed-fractional fractional offsets are (0, w1, w1+w2, w1+w2+w3):
#     dU/dh = -mult (g_j w1' + g_k (w1+w2)' + g_l (w1+w2+w3)').
# The three offsets matter: using w1, w2, w3 directly (without the cumulative
# sums) is not the derivative of φ and fails finite-difference comparison.
function RS.accumulate_box_derivative!(G::Matrix{Float64}, f::Factors.PeriodicDihedral,
                                      q::AbstractMatrix{Float64},
                                      ctx::Factors.Context)
    t = Factors.dihedral_angle_grad(q, ctx.box, f.i, f.j, f.k, f.l)
    if t === nothing
        # Matches evaluation, which returns Inf at a degenerate (collinear) geometry.
        G .= Inf
        return G
    end
    phi, (gi, gj, gk, gl) = t
    arg = f.n * phi - f.phi0
    mult = f.kphi * f.n * sin(arg)
    w1 = _frac_disp(ctx.box, Types.unwrap_displacement(ctx.box, q, f.j, f.i))
    w2 = _frac_disp(ctx.box, Types.unwrap_displacement(ctx.box, q, f.k, f.j))
    w3 = _frac_disp(ctx.box, Types.unwrap_displacement(ctx.box, q, f.l, f.k))
    s_k = _vadd(w1, w2)
    s_l = _vadd(s_k, w3)
    _outer_add!(G, -mult, gj, w1)
    _outer_add!(G, -mult, gk, s_k)
    _outer_add!(G, -mult, gl, s_l)
    return G
end

_accumulate_box_derivative!(G::Matrix{Float64}, f::Factors.PeriodicDihedral,
                            q::AbstractMatrix{Float64}, ctx::Factors.Context, ::Symbol) =
    RS.accumulate_box_derivative!(G, f, q, ctx)

# The bipartite summary uses raw Cartesian coordinates, so dU/dh = -Σ_a F_a s_a'.
function RS.accumulate_box_derivative!(G::Matrix{Float64}, f::Factors.BipartiteHarmonicGroup,
                                      q::AbstractMatrix{Float64},
                                      ctx::Factors.Context)
    N = size(q, 2)
    F = zeros(Float64, 3, N)
    Factors.evaluate!(F, f, q, ctx)
    @inbounds for a in 1:N
        w = _frac_disp(ctx.box, Types.SVector3(q[1, a], q[2, a], q[3, a]))
        _outer_add!(G, -1.0, F[1, a], F[2, a], F[3, a], w[1], w[2], w[3])
    end
    return G
end

_accumulate_box_derivative!(G::Matrix{Float64}, f::Factors.BipartiteHarmonicGroup,
                            q::AbstractMatrix{Float64}, ctx::Factors.Context, ::Symbol) =
    RS.accumulate_box_derivative!(G, f, q, ctx)

# U = λ U_mode(r), r = |h w|: dU/dh = (λ dU/dr / r) d w'.
function RS.accumulate_box_derivative!(G::Matrix{Float64}, f::Factors.LennardJonesPair,
                                      q::AbstractMatrix{Float64},
                                      ctx::Factors.Context)
    (f.mode === :none || f.mode === :lfs) || throw(MissingBoxDerivative(:lj_pair,
        "mode $(f.mode) is not force-consistent at the cutoff"))
    d = Types.unwrap_displacement(ctx.box, q, f.i, f.j)
    r = sqrt(Types.norm2(d))
    if f.mode !== :none && r > f.rc
        return G
    end
    dudr = Factors.lj_dudr(f.ε, f.σ, r)
    if f.mode === :lfs
        dudr -= Factors.lj_dudr(f.ε, f.σ, f.rc)
    end
    _outer_add!(G, f.lj_scale * dudr / r, d, _frac_disp(ctx.box, d))
    return G
end

function _accumulate_box_derivative!(G::Matrix{Float64}, f::Factors.LennardJonesPair,
                                     q::AbstractMatrix{Float64},
                                     ctx::Factors.Context, name::Symbol)
    (f.mode === :none || f.mode === :lfs) || throw(MissingBoxDerivative(name,
        "mode $(f.mode) is not force-consistent at the cutoff"))
    return RS.accumulate_box_derivative!(G, f, q, ctx)
end

function RS.accumulate_box_derivative!(G::Matrix{Float64}, f::StreamedLJ.StreamedLJFactor,
                                      q::AbstractMatrix{Float64},
                                      ctx::Factors.Context)
    f.mode === :lfs || throw(MissingBoxDerivative(:streamed_lj,
        "StreamedLJ mode $(f.mode) is not C1/force-consistent"))
    box = ctx.box
    (box isa Types.OrthorhombicBox || box isa Types.TriclinicBox) ||
        throw(MissingBoxDerivative(:streamed_lj,
            "StreamedLJFactor requires an orthorhombic or triclinic periodic box; got $(typeof(box))"))
    StreamedLJ.streamed_box_derivative!(f.ws.box_derivative, f, q, ctx)
    @inbounds for b in 1:3, a in 1:3
        G[a, b] += f.ws.box_derivative[a, b]
    end
    return G
end

function _accumulate_box_derivative!(G::Matrix{Float64}, f::StreamedLJ.StreamedLJFactor,
                                     q::AbstractMatrix{Float64},
                                     ctx::Factors.Context, name::Symbol)
    f.mode === :lfs || throw(MissingBoxDerivative(name,
        "StreamedLJ mode $(f.mode) is not C1/force-consistent"))
    box = ctx.box
    (box isa Types.OrthorhombicBox || box isa Types.TriclinicBox) ||
        throw(MissingBoxDerivative(name,
            "StreamedLJFactor requires an orthorhombic or triclinic periodic box; got $(typeof(box))"))
    return RS.accumulate_box_derivative!(G, f, q, ctx)
end

function RS.accumulate_box_derivative!(G::Matrix{Float64}, f::PME.PMEFactor,
                                      q::AbstractMatrix{Float64},
                                      ctx::Factors.Context)
    box = ctx.box
    (box isa Types.OrthorhombicBox || box isa Types.TriclinicBox) ||
        throw(MissingBoxDerivative(:pme,
            "PMEFactor requires an orthorhombic or triclinic periodic box; got $(typeof(box))"))
    PME.pme_factor_box_derivative!(G, f, q, ctx; accumulate = true)
    return G
end

function RS.box_derivative(f::PME.PMEFactor, q::AbstractMatrix{Float64},
                           ctx::Factors.Context)
    box = ctx.box
    (box isa Types.OrthorhombicBox || box isa Types.TriclinicBox) ||
        throw(MissingBoxDerivative(:pme,
            "PMEFactor requires an orthorhombic or triclinic periodic box; got $(typeof(box))"))
    return PME.pme_factor_box_derivative(f, q, ctx)
end

function _accumulate_box_derivative!(G::Matrix{Float64}, f::PME.PMEFactor,
                                     q::AbstractMatrix{Float64},
                                     ctx::Factors.Context, name::Symbol)
    box = ctx.box
    (box isa Types.OrthorhombicBox || box isa Types.TriclinicBox) ||
        throw(MissingBoxDerivative(name,
            "PMEFactor requires an orthorhombic or triclinic periodic box; got $(typeof(box))"))
    return RS.accumulate_box_derivative!(G, f, q, ctx)
end

# ---------------------------------------------------------------------------
# analytic box derivative: public entry points
# ---------------------------------------------------------------------------

"""
    box_derivative_component(rs, q, name::Symbol) -> 3 x 3

`G[a,b] = ∂U_component/∂h[a,b]` at fixed fractional coordinates. Throws
[`MissingBoxDerivative`](@ref) when the component has no analytic stress, including
when the system has no periodic box.
"""
function box_derivative_component(rs::RS.ResolvedSystem, q::AbstractMatrix{Float64},
                                 name::Symbol)
    _check_positions(rs, q)
    _refuse_virtual_sites(rs)
    k = findfirst(==(name), rs.component_names)
    k === nothing && throw(RS.ResolvedSystemError(:component_name,
        "no component named $(repr(name)); components are $(rs.component_names)"))
    f = rs.factors[k]
    reason = _box_derivative_reason(f, rs.box)
    reason === nothing || throw(MissingBoxDerivative(name, reason))
    q_placed = isempty(rs.atoms.virtual_sites) ? q : place_virtual_sites(q, rs)
    G = zeros(Float64, 3, 3)
    return _accumulate_box_derivative!(G, f, q_placed, resolved_context(rs), name)
end

"""
    box_derivative(rs, q) -> 3 x 3

Total `G[a,b] = ∂U/∂h[a,b]` at fixed fractional coordinates. Requires a periodic box
and an analytic box derivative for **every** component: one unsupported component
refuses the total with [`MissingBoxDerivative`](@ref) naming it. No zeros are
substituted and no finite differences are used.
"""
function box_derivative(rs::RS.ResolvedSystem, q::AbstractMatrix{Float64})
    _check_positions(rs, q)
    _refuse_virtual_sites(rs)
    rs.box isa Types.NoBox && throw(MissingBoxDerivative(:system,
        "no periodic box: a box derivative requires lattice vectors"))
    q_placed = isempty(rs.atoms.virtual_sites) ? q : place_virtual_sites(q, rs)
    ctx = resolved_context(rs)
    G = zeros(Float64, 3, 3)
    @inbounds for k in eachindex(rs.factors)
        name = rs.component_names[k]
        f = rs.factors[k]
        reason = _box_derivative_reason(f, rs.box)
        reason === nothing || throw(MissingBoxDerivative(name, reason))
        _accumulate_box_derivative!(G, f, q_placed, ctx, name)
    end
    return G
end

function box_derivative!(G::AbstractMatrix{Float64}, rs::RS.ResolvedSystem, q::AbstractMatrix{Float64})
    G_calc = box_derivative(rs, q)
    G .= G_calc
    return G
end

"""
    virial(rs, q) -> 3 x 3

`Ξ = -h*G'` with `Ξ[a,b] = Σ_i r_i[a] F_i[b]`. Refuses whenever
[`box_derivative`](@ref) refuses.
"""
function virial(rs::RS.ResolvedSystem, q::AbstractMatrix{Float64})
    G = box_derivative(rs, q)
    h = _hmat(rs.box)
    return -(h * transpose(G))
end

"""
    affine_pressure(rs, q) -> Float64

`-tr(G*h')/(3V)`, the fixed-fractional affine (virial) pressure. Equals
`tr(Ξ)/(3V)` and is the potential part of [`pressure`](@ref). Refuses whenever the
box derivative is unavailable.
"""
function affine_pressure(rs::RS.ResolvedSystem, q::AbstractMatrix{Float64})
    G = box_derivative(rs, q)
    h = _hmat(rs.box)
    V = Types.box_volume(rs.box)
    return -tr(G * transpose(h)) / (3.0 * V)
end

"""
    kinetic_energy(rs, p) -> Float64

`K = 0.5 Σ p²/m`. For systems with declared virtual sites (mass 0), momentum must be
strictly zero and massless degrees of freedom are excluded from kinetic energy.
"""
function kinetic_energy(rs::RS.ResolvedSystem, p::AbstractMatrix{Float64})
    _check_momenta(rs, p)
    _refuse_virtual_sites(rs)
    m = rs.atoms.masses
    N = RS.n_atoms(rs)
    s = 0.0
    @inbounds for j in 1:N
        if m[j] == 0.0
            (p[1, j] == 0.0 && p[2, j] == 0.0 && p[3, j] == 0.0) ||
                throw(ArgumentError("massless virtual site $j has nonzero momentum"))
            continue
        end
        for a in 1:3
            s += p[a, j]^2 / m[j]
        end
    end
    return 0.5 * s
end

"""
    pressure(rs, q, p; extra_virial=0.0) -> Float64

`(2K + tr(Ξ) + extra_virial)/(3V)`. `extra_virial` is where a caller adds a
constraint virial (`Constraints.constraint_virial`); with the default this is the
**unconstrained affine** pressure. Refuses whenever the box derivative is
unavailable, so a partial stress can never be reported as a pressure.
"""
function pressure(rs::RS.ResolvedSystem, q::AbstractMatrix{Float64},
                  p::AbstractMatrix{Float64}; extra_virial::Float64 = 0.0)
    _check_positions(rs, q)
    _check_momenta(rs, p)
    _refuse_virtual_sites(rs)
    Ξ = virial(rs, q)
    V = Types.box_volume(rs.box)
    K = kinetic_energy(rs, p)
    return (2.0 * K + tr(Ξ) + extra_virial) / (3.0 * V)
end

# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------

"""
    resolved_from_system(s::Types.System, factors; kwargs...) -> ResolvedSystem

Build a [`ResolvedSystem`](@ref) from a legacy `Types.System`: masses, charges,
names, box, topology and units are taken from `s`. The constructor validates the
result, so a `System` that was legal as *state* but not as a *specification* (for
example a `CoulombPair` whose charges disagree with the atom charges) is refused.
"""
function resolved_from_system(s::Types.System, factors;
                              component_names = nothing, constraints = nothing,
                              virtual_sites = RS.VirtualSite[],
                              provenance = RS.Provenance(),
                              declared = [:energy, :cartesian_forces,
                                          :component_energies],
                              schema = RS.RESOLVED_SCHEMA_VERSION)
    return RS.ResolvedSystem(;
        masses = s.atoms.m, charges = s.atoms.z, names = s.atoms.names,
        box = s.box, topology = s.topology, factors = factors,
        component_names = component_names, constraints = constraints,
        virtual_sites = virtual_sites, units = s.units,
        provenance = provenance, declared = declared, schema = schema)
end

"""
    to_atomset(rs, q, p) -> Types.AtomSet

Legacy-state adapter. A system with declared virtual sites refuses, because
`Types.AtomSet` requires strictly positive masses and this core has no
site-construction kernel.
"""
function to_atomset(rs::RS.ResolvedSystem, q::AbstractMatrix{Float64},
                    p::AbstractMatrix{Float64})
    isempty(rs.atoms.virtual_sites) || throw(RS.UnsupportedCapability(:virtual_sites,
        "$(length(rs.atoms.virtual_sites)) virtual site(s) are representation only; " *
        "execution is not implemented, so no AtomSet can be produced"))
    _check_positions(rs, q)
    _check_momenta(rs, p)
    return Types.AtomSet(q, p, rs.atoms.masses, rs.atoms.charges, rs.atoms.names)
end

end # module
