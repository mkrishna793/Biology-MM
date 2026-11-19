"""
NEMD — Non-Equilibrium Molecular Dynamics & Reverse-NEMD (RNEMD) for Biology MM.

Implements:
1. `ExternalElectricFieldFactor <: Factors.AbstractFactor` + `estimate_ionic_conductivity`
2. `NanofluidicBodyForceFactor <: Factors.AbstractFactor` + `estimate_mass_flux`
3. `WallRestraintFactor <: Factors.AbstractFactor` (`:harmonic_flat_bottom`,
   `:lj93` with C¹ `:lfs` shift, and `:cylindrical_pore`)
4. Exact analytical 3-entry orthorhombic & 9-entry triclinic fixed-fractional
   `ResolvedSystem.accumulate_box_derivative!` for all NEMD factors
5. Müller–Plathe Reverse-NEMD momentum & thermal swaps (`muller_plathe_shear_step!`,
   `muller_plathe_thermal_step!`, `estimate_shear_viscosity`, `estimate_thermal_conductivity`)
   with exact momentum and kinetic energy conservation for arbitrary masses.
"""
module NEMD

using LinearAlgebra
using Statistics
using ..Types
using ..Factors
import ..ResolvedSystem

if !isdefined(Factors, :accumulate_box_derivative!)
    @eval Factors function accumulate_box_derivative! end
end

export ExternalElectricFieldFactor, estimate_ionic_conductivity,
       NanofluidicBodyForceFactor, estimate_mass_flux,
       WallRestraintFactor,
       muller_plathe_shear_step!, estimate_shear_viscosity,
       muller_plathe_thermal_step!, estimate_thermal_conductivity

# ==============================================================================
# Shared Box-Derivative Helper for External Cartesian Factors
# ==============================================================================

@inline function _frac_coords(box::Types.OrthorhombicBox, q::AbstractMatrix{Float64}, i::Int)::NTuple{3,Float64}
    L = box.L
    @inbounds return (q[1, i] / L[1], q[2, i] / L[2], q[3, i] / L[3])
end

@inline function _frac_coords(box::Types.TriclinicBox, q::AbstractMatrix{Float64}, i::Int)::NTuple{3,Float64}
    Hinv = box.Hinv
    @inbounds begin
        x = q[1, i]; y = q[2, i]; z = q[3, i]
        s1 = Hinv[1, 1] * x + Hinv[1, 2] * y + Hinv[1, 3] * z
        s2 = Hinv[2, 1] * x + Hinv[2, 2] * y + Hinv[2, 3] * z
        s3 = Hinv[3, 1] * x + Hinv[3, 2] * y + Hinv[3, 3] * z
    end
    return (s1, s2, s3)
end

@inline function _accum_cartesian_box_deriv!(G::AbstractMatrix{Float64},
                                             box::Types.AbstractBox,
                                             q::AbstractMatrix{Float64},
                                             i::Int,
                                             fx::Float64, fy::Float64, fz::Float64)
    # ∂U/∂H[a,b] = (-F_i[a]) * (H^{-1} q_i)[b]
    s = _frac_coords(box, q, i)
    @inbounds begin
        G[1, 1] -= fx * s[1]; G[1, 2] -= fx * s[2]; G[1, 3] -= fx * s[3]
        G[2, 1] -= fy * s[1]; G[2, 2] -= fy * s[2]; G[2, 3] -= fy * s[3]
        G[3, 1] -= fz * s[1]; G[3, 2] -= fz * s[2]; G[3, 3] -= fz * s[3]
    end
    return G
end

function _require_periodic_box(box::Types.AbstractBox)
    box isa Types.NoBox && throw(ArgumentError("box derivative requires a periodic box (OrthorhombicBox or TriclinicBox)"))
    return nothing
end

# ==============================================================================
# 1. ExternalElectricFieldFactor & Ionic Conductivity
# ==============================================================================

"""
    ExternalElectricFieldFactor(E, charges, atom_indices, n_atoms)

Uniform external electric field factor:
`U = -∑_{i ∈ S} q_i (E ⋅ r_i)`, `F_i = q_i E`.
`charges` may be given either with length `length(atom_indices)` or `n_atoms`.
"""
struct ExternalElectricFieldFactor <: Factors.AbstractFactor
    E::NTuple{3,Float64}
    charges::Vector{Float64}
    atom_indices::Vector{Int}
    n_atoms::Int
    function ExternalElectricFieldFactor(E::NTuple{3,<:Real},
                                         charges::AbstractVector{<:Real},
                                         atom_indices::AbstractVector{<:Integer},
                                         n_atoms::Integer)
        E_t = (Float64(E[1]), Float64(E[2]), Float64(E[3]))
        all(isfinite, E_t) || throw(ArgumentError("electric field E must be finite"))
        na = Int(n_atoms)
        na >= 1 || throw(ArgumentError("n_atoms must be >= 1"))
        idxs = Vector{Int}(atom_indices)
        all(i -> 1 <= i <= na, idxs) || throw(BoundsError("atom_indices out of 1:$na"))
        qs = Vector{Float64}(charges)
        all(isfinite, qs) || throw(ArgumentError("charges must be finite"))
        (length(qs) == length(idxs) || length(qs) == na) ||
            throw(DimensionMismatch("charges length $(length(qs)) must match atom_indices ($(length(idxs))) or n_atoms ($na)"))
        new(E_t, qs, idxs, na)
    end
end

function ExternalElectricFieldFactor(E::NTuple{3,<:Real}, charges::AbstractVector{<:Real})
    na = length(charges)
    return ExternalElectricFieldFactor(E, charges, collect(1:na), na)
end

function ExternalElectricFieldFactor(E::AbstractVector{<:Real}, charges::AbstractVector{<:Real},
                                     atom_indices::AbstractVector{<:Integer}, n_atoms::Integer)
    length(E) == 3 || throw(DimensionMismatch("E must have length 3"))
    return ExternalElectricFieldFactor((E[1], E[2], E[3]), charges, atom_indices, n_atoms)
end

function ExternalElectricFieldFactor(E::AbstractVector{<:Real}, charges::AbstractVector{<:Real})
    length(E) == 3 || throw(DimensionMismatch("E must have length 3"))
    return ExternalElectricFieldFactor((E[1], E[2], E[3]), charges)
end

Factors.factor_support_size(f::ExternalElectricFieldFactor) = length(f.atom_indices)
Factors.read_support(f::ExternalElectricFieldFactor) = Tuple(f.atom_indices)
Factors.is_local(::ExternalElectricFieldFactor) = true
Factors.force_consistent(::ExternalElectricFieldFactor) = true

@inline function _efield_charge(f::ExternalElectricFieldFactor, k::Int, idx::Int)::Float64
    return length(f.charges) == f.n_atoms && length(f.atom_indices) != f.n_atoms ?
           @inbounds(f.charges[idx]) : @inbounds(f.charges[k])
end

function Factors.evaluate!(F::AbstractMatrix{Float64}, f::ExternalElectricFieldFactor,
                           q::AbstractMatrix{Float64}, ctx::Factors.Context)
    Ex, Ey, Ez = f.E
    U = 0.0
    @inbounds for (k, i) in enumerate(f.atom_indices)
        qi = _efield_charge(f, k, i)
        U -= qi * (Ex * q[1, i] + Ey * q[2, i] + Ez * q[3, i])
        F[1, i] += qi * Ex
        F[2, i] += qi * Ey
        F[3, i] += qi * Ez
    end
    return U
end

function Factors.accumulate_box_derivative!(G::AbstractMatrix{Float64},
                                            f::ExternalElectricFieldFactor,
                                            q::AbstractMatrix{Float64},
                                            ctx::Factors.Context)
    size(G) == (3, 3) || throw(DimensionMismatch("G must be 3x3"))
    _require_periodic_box(ctx.box)
    Ex, Ey, Ez = f.E
    @inbounds for (k, i) in enumerate(f.atom_indices)
        qi = _efield_charge(f, k, i)
        _accum_cartesian_box_deriv!(G, ctx.box, q, i, qi * Ex, qi * Ey, qi * Ez)
    end
    return G
end

ResolvedSystem.accumulate_box_derivative!(G::Matrix{Float64},
                                          f::ExternalElectricFieldFactor,
                                          q::AbstractMatrix{Float64},
                                          ctx::Factors.Context) =
    Factors.accumulate_box_derivative!(G, f, q, ctx)
ResolvedSystem.box_derivative_support(::ExternalElectricFieldFactor) = nothing

"""
    estimate_ionic_conductivity(velocities_history, charges, E_vec, volume) -> Float64

Estimate ionic conductivity `σ = ⟨J ⋅ Ê⟩ / ‖E‖`, where current density is
`J = (1/V) ∑_i q_i v_i`. Supports a vector of `3 × N` velocity frames, a `3 × N × T`
array, or a single `3 × N` matrix.
"""
function estimate_ionic_conductivity(velocities_history,
                                     charges::AbstractVector{<:Real},
                                     E_vec,
                                     volume::Real)::Float64
    V = Float64(volume)
    (isfinite(V) && V > 0.0) || throw(ArgumentError("volume must be positive and finite"))
    Ev = (Float64(E_vec[1]), Float64(E_vec[2]), Float64(E_vec[3]))
    Enorm = sqrt(Ev[1]^2 + Ev[2]^2 + Ev[3]^2)
    Enorm > 0.0 || throw(ArgumentError("electric field magnitude must be positive"))
    ehx, ehy, ehz = Ev[1] / Enorm, Ev[2] / Enorm, Ev[3] / Enorm
    N = length(charges)

    j_par_sum = 0.0
    n_frames = 0
    if velocities_history isa AbstractMatrix
        size(velocities_history) == (3, N) || throw(DimensionMismatch("velocity matrix must be 3 x $N"))
        jx = 0.0; jy = 0.0; jz = 0.0
        @inbounds for i in 1:N
            qi = Float64(charges[i])
            jx += qi * velocities_history[1, i]
            jy += qi * velocities_history[2, i]
            jz += qi * velocities_history[3, i]
        end
        j_par_sum = (jx * ehx + jy * ehy + jz * ehz) / V
        n_frames = 1
    elseif velocities_history isa AbstractArray{<:Real,3}
        size(velocities_history, 1) == 3 && size(velocities_history, 2) == N ||
            throw(DimensionMismatch("velocity array must be 3 x $N x T"))
        T = size(velocities_history, 3)
        T > 0 || throw(ArgumentError("velocities_history must be non-empty"))
        @inbounds for t in 1:T
            jx = 0.0; jy = 0.0; jz = 0.0
            for i in 1:N
                qi = Float64(charges[i])
                jx += qi * velocities_history[1, i, t]
                jy += qi * velocities_history[2, i, t]
                jz += qi * velocities_history[3, i, t]
            end
            j_par_sum += (jx * ehx + jy * ehy + jz * ehz) / V
        end
        n_frames = T
    else
        isempty(velocities_history) && throw(ArgumentError("velocities_history must be non-empty"))
        for v in velocities_history
            size(v) == (3, N) || throw(DimensionMismatch("each velocity frame must be 3 x $N"))
            jx = 0.0; jy = 0.0; jz = 0.0
            @inbounds for i in 1:N
                qi = Float64(charges[i])
                jx += qi * v[1, i]
                jy += qi * v[2, i]
                jz += qi * v[3, i]
            end
            j_par_sum += (jx * ehx + jy * ehy + jz * ehz) / V
            n_frames += 1
        end
    end

    return (j_par_sum / n_frames) / Enorm
end

# ==============================================================================
# 2. NanofluidicBodyForceFactor & Mass Flux
# ==============================================================================

"""
    NanofluidicBodyForceFactor(force, atom_indices, n_atoms)

Constant body-force / Poiseuille pressure-gradient factor:
`U = -∑_{i ∈ S} (f ⋅ r_i)`, `F_i = f`.
"""
struct NanofluidicBodyForceFactor <: Factors.AbstractFactor
    force::NTuple{3,Float64}
    atom_indices::Vector{Int}
    n_atoms::Int
    function NanofluidicBodyForceFactor(force::NTuple{3,<:Real},
                                        atom_indices::AbstractVector{<:Integer},
                                        n_atoms::Integer)
        ft = (Float64(force[1]), Float64(force[2]), Float64(force[3]))
        all(isfinite, ft) || throw(ArgumentError("body force must be finite"))
        na = Int(n_atoms)
        na >= 1 || throw(ArgumentError("n_atoms must be >= 1"))
        idxs = Vector{Int}(atom_indices)
        all(i -> 1 <= i <= na, idxs) || throw(BoundsError("atom_indices out of 1:$na"))
        new(ft, idxs, na)
    end
end

NanofluidicBodyForceFactor(force::NTuple{3,<:Real}, n_atoms::Integer) =
    NanofluidicBodyForceFactor(force, collect(1:Int(n_atoms)), Int(n_atoms))

function NanofluidicBodyForceFactor(force::AbstractVector{<:Real},
                                    atom_indices::AbstractVector{<:Integer},
                                    n_atoms::Integer)
    length(force) == 3 || throw(DimensionMismatch("force must have length 3"))
    return NanofluidicBodyForceFactor((force[1], force[2], force[3]), atom_indices, n_atoms)
end

NanofluidicBodyForceFactor(force::AbstractVector{<:Real}, n_atoms::Integer) =
    NanofluidicBodyForceFactor(force, collect(1:Int(n_atoms)), Int(n_atoms))

Factors.factor_support_size(f::NanofluidicBodyForceFactor) = length(f.atom_indices)
Factors.read_support(f::NanofluidicBodyForceFactor) = Tuple(f.atom_indices)
Factors.is_local(::NanofluidicBodyForceFactor) = true
Factors.force_consistent(::NanofluidicBodyForceFactor) = true

function Factors.evaluate!(F::AbstractMatrix{Float64}, f::NanofluidicBodyForceFactor,
                           q::AbstractMatrix{Float64}, ctx::Factors.Context)
    fx, fy, fz = f.force
    U = 0.0
    @inbounds for i in f.atom_indices
        U -= fx * q[1, i] + fy * q[2, i] + fz * q[3, i]
        F[1, i] += fx
        F[2, i] += fy
        F[3, i] += fz
    end
    return U
end

function Factors.accumulate_box_derivative!(G::AbstractMatrix{Float64},
                                            f::NanofluidicBodyForceFactor,
                                            q::AbstractMatrix{Float64},
                                            ctx::Factors.Context)
    size(G) == (3, 3) || throw(DimensionMismatch("G must be 3x3"))
    _require_periodic_box(ctx.box)
    fx, fy, fz = f.force
    @inbounds for i in f.atom_indices
        _accum_cartesian_box_deriv!(G, ctx.box, q, i, fx, fy, fz)
    end
    return G
end

ResolvedSystem.accumulate_box_derivative!(G::Matrix{Float64},
                                          f::NanofluidicBodyForceFactor,
                                          q::AbstractMatrix{Float64},
                                          ctx::Factors.Context) =
    Factors.accumulate_box_derivative!(G, f, q, ctx)
ResolvedSystem.box_derivative_support(::NanofluidicBodyForceFactor) = nothing

"""
    estimate_mass_flux(q_start, q_end, dt_total, cross_section_area, masses,
                       flow_axis=3; indices=nothing) -> Float64

Compute net mass flux `J_m = ∑_i m_i (q_end[d, i] - q_start[d, i]) / (A * Δt)`
from unwrapped start and end coordinates.
"""
function estimate_mass_flux(q_start::AbstractMatrix{<:Real},
                            q_end::AbstractMatrix{<:Real},
                            dt_total::Real,
                            cross_section_area::Real,
                            masses::AbstractVector{<:Real},
                            flow_axis::Int = 3;
                            indices = nothing)::Float64
    1 <= flow_axis <= 3 || throw(ArgumentError("flow_axis must be 1, 2, or 3"))
    size(q_start) == size(q_end) || throw(DimensionMismatch("q_start and q_end shape mismatch"))
    size(q_start, 1) == 3 || throw(DimensionMismatch("positions must be 3 x N"))
    N = size(q_start, 2)
    length(masses) == N || throw(DimensionMismatch("masses length != $N"))
    dt = Float64(dt_total)
    A = Float64(cross_section_area)
    (isfinite(dt) && dt > 0.0) || throw(ArgumentError("dt_total must be positive and finite"))
    (isfinite(A) && A > 0.0) || throw(ArgumentError("cross_section_area must be positive and finite"))
    idxs = indices === nothing ? (1:N) : indices
    dm = 0.0
    @inbounds for i in idxs
        dm += Float64(masses[i]) * (Float64(q_end[flow_axis, i]) - Float64(q_start[flow_axis, i]))
    end
    return dm / (A * dt)
end

# ==============================================================================
# 3. WallRestraintFactor
# ==============================================================================

@inline function _lj93_energy(eps::Float64, sig::Float64, z::Float64)::Float64
    s3 = (sig / z)^3
    s9 = s3 * s3 * s3
    return eps * ((2.0 / 15.0) * s9 - s3)
end

@inline function _lj93_dudz(eps::Float64, sig::Float64, z::Float64)::Float64
    s3 = (sig / z)^3
    s9 = s3 * s3 * s3
    return eps * (3.0 * s3 - 1.2 * s9) / z
end

@inline function _lj93_lfs(eps::Float64, sig::Float64, rc::Float64, z::Float64)::Tuple{Float64,Float64}
    z <= 0.0 && return (Inf, 0.0)
    z >= rc && return (0.0, 0.0)
    u = _lj93_energy(eps, sig, z)
    du = _lj93_dudz(eps, sig, z)
    uc = _lj93_energy(eps, sig, rc)
    duc = _lj93_dudz(eps, sig, rc)
    return (u - uc - duc * (z - rc), du - duc)
end

"""
    WallRestraintFactor

C¹ force-consistent confining wall / nanopore restraint factor supporting:
- `:harmonic_flat_bottom`: flat potential in `[lo, hi]` along `axis`, harmonic outside with stiffness `k`
- `:lj93`: planar 9-3 LJ slab walls at `lo` and `hi` along `axis` with C¹ `:lfs` shift at `rc`
- `:cylindrical_pore`: cylindrical pore along `axis` centered at `(c1, c2)` with radius `r_max`;
  uses harmonic wall `0.5*k*(r - r_max)^2` for `r > r_max` when `k > 0`, or C¹ `:lfs` 9-3 LJ wall
  at radial clearance `z = r_max - r` when `k == 0 && epsilon > 0`.
"""
struct WallRestraintFactor <: Factors.AbstractFactor
    mode::Symbol
    axis::Int
    lo::Float64
    hi::Float64
    k::Float64
    epsilon::Float64
    sigma::Float64
    rc::Float64
    center::NTuple{2,Float64}
    r_max::Float64
    atom_indices::Vector{Int}
    n_atoms::Int
end

function WallRestraintFactor(mode::Symbol,
                             atom_indices::AbstractVector{<:Integer},
                             n_atoms::Integer;
                             axis::Integer = 3,
                             lo::Real = -Inf,
                             hi::Real = Inf,
                             bounds::Union{Nothing,Tuple{Real,Real}} = nothing,
                             k::Real = 0.0,
                             epsilon::Real = 0.0,
                             sigma::Real = 1.0,
                             rc::Real = 2.5,
                             center::NTuple{2,<:Real} = (0.0, 0.0),
                             r_max::Real = Inf)
    mode in (:harmonic_flat_bottom, :lj93, :cylindrical_pore) ||
        throw(ArgumentError("unknown WallRestraintFactor mode :$mode"))
    ax = Int(axis)
    1 <= ax <= 3 || throw(ArgumentError("axis must be 1, 2, or 3"))
    na = Int(n_atoms)
    na >= 1 || throw(ArgumentError("n_atoms must be >= 1"))
    idxs = Vector{Int}(atom_indices)
    all(i -> 1 <= i <= na, idxs) || throw(BoundsError("atom_indices out of 1:$na"))

    lo_f = bounds === nothing ? Float64(lo) : Float64(bounds[1])
    hi_f = bounds === nothing ? Float64(hi) : Float64(bounds[2])
    lo_f <= hi_f || throw(ArgumentError("wall lo ($lo_f) must be <= hi ($hi_f)"))
    k_f = Float64(k)
    eps_f = Float64(epsilon)
    sig_f = Float64(sigma)
    rc_f = Float64(rc)
    c_t = (Float64(center[1]), Float64(center[2]))
    rmax_f = Float64(r_max)

    k_f >= 0.0 || throw(ArgumentError("k must be non-negative"))
    eps_f >= 0.0 || throw(ArgumentError("epsilon must be non-negative"))
    sig_f > 0.0 || throw(ArgumentError("sigma must be positive"))
    rc_f > 0.0 || throw(ArgumentError("rc must be positive"))
    all(isfinite, c_t) || throw(ArgumentError("center must be finite"))
    if mode === :cylindrical_pore
        (isfinite(rmax_f) && rmax_f > 0.0) || throw(ArgumentError("r_max must be positive and finite for :cylindrical_pore"))
    end

    return WallRestraintFactor(mode, ax, lo_f, hi_f, k_f, eps_f, sig_f, rc_f, c_t, rmax_f, idxs, na)
end

WallRestraintFactor(mode::Symbol, n_atoms::Integer; kwargs...) =
    WallRestraintFactor(mode, collect(1:Int(n_atoms)), Int(n_atoms); kwargs...)

Factors.factor_support_size(f::WallRestraintFactor) = length(f.atom_indices)
Factors.read_support(f::WallRestraintFactor) = Tuple(f.atom_indices)
Factors.is_local(::WallRestraintFactor) = true
Factors.force_consistent(::WallRestraintFactor) = true

@inline function _transverse_axes(axis::Int)::Tuple{Int,Int}
    return axis == 1 ? (2, 3) : (axis == 2 ? (1, 3) : (1, 2))
end

@inline function _eval_wall_atom(f::WallRestraintFactor,
                                 x1::Float64, x2::Float64, x3::Float64)::Tuple{Float64,Float64,Float64,Float64}
    if f.mode === :harmonic_flat_bottom
        ax = f.axis
        x = ax == 1 ? x1 : (ax == 2 ? x2 : x3)
        u = 0.0
        fa = 0.0
        if x < f.lo
            dx = x - f.lo
            u = 0.5 * f.k * dx * dx
            fa = -f.k * dx
        elseif x > f.hi
            dx = x - f.hi
            u = 0.5 * f.k * dx * dx
            fa = -f.k * dx
        end
        return ax == 1 ? (u, fa, 0.0, 0.0) : (ax == 2 ? (u, 0.0, fa, 0.0) : (u, 0.0, 0.0, fa))
    elseif f.mode === :lj93
        ax = f.axis
        x = ax == 1 ? x1 : (ax == 2 ? x2 : x3)
        u = 0.0
        fa = 0.0
        if isfinite(f.lo)
            z = x - f.lo
            z <= 0.0 && return (Inf, 0.0, 0.0, 0.0)
            ul, dul = _lj93_lfs(f.epsilon, f.sigma, f.rc, z)
            u += ul
            fa -= dul
        end
        if isfinite(f.hi)
            z = f.hi - x
            z <= 0.0 && return (Inf, 0.0, 0.0, 0.0)
            uh, duh = _lj93_lfs(f.epsilon, f.sigma, f.rc, z)
            u += uh
            fa += duh
        end
        return ax == 1 ? (u, fa, 0.0, 0.0) : (ax == 2 ? (u, 0.0, fa, 0.0) : (u, 0.0, 0.0, fa))
    else
        d1, d2 = _transverse_axes(f.axis)
        p1 = d1 == 1 ? x1 : (d1 == 2 ? x2 : x3)
        p2 = d2 == 1 ? x1 : (d2 == 2 ? x2 : x3)
        dx1 = p1 - f.center[1]
        dx2 = p2 - f.center[2]
        r2 = dx1 * dx1 + dx2 * dx2
        r = sqrt(r2)
        u = 0.0
        f1 = 0.0
        f2 = 0.0
        if f.k > 0.0
            if r > f.r_max
                dr = r - f.r_max
                u = 0.5 * f.k * dr * dr
                c = -f.k * dr / r
                f1 = c * dx1
                f2 = c * dx2
            end
        elseif f.epsilon > 0.0
            z = f.r_max - r
            z <= 0.0 && return (Inf, 0.0, 0.0, 0.0)
            if z < f.rc && r > 0.0
                ul, dul = _lj93_lfs(f.epsilon, f.sigma, f.rc, z)
                u = ul
                c = dul / r  # F = -dU/dr * (dx/r) = -(-dul) * dx/r = +dul * dx/r
                f1 = c * dx1
                f2 = c * dx2
            elseif z < f.rc && r == 0.0
                ul, _ = _lj93_lfs(f.epsilon, f.sigma, f.rc, z)
                u = ul
            end
        end
        if f.axis == 1
            return (u, 0.0, f1, f2)
        elseif f.axis == 2
            return (u, f1, 0.0, f2)
        else
            return (u, f1, f2, 0.0)
        end
    end
end

function Factors.evaluate!(F::AbstractMatrix{Float64}, f::WallRestraintFactor,
                           q::AbstractMatrix{Float64}, ctx::Factors.Context)
    U = 0.0
    @inbounds for i in f.atom_indices
        u, fx, fy, fz = _eval_wall_atom(f, q[1, i], q[2, i], q[3, i])
        !isfinite(u) && return Inf
        U += u
        F[1, i] += fx
        F[2, i] += fy
        F[3, i] += fz
    end
    return U
end

function Factors.accumulate_box_derivative!(G::AbstractMatrix{Float64},
                                            f::WallRestraintFactor,
                                            q::AbstractMatrix{Float64},
                                            ctx::Factors.Context)
    size(G) == (3, 3) || throw(DimensionMismatch("G must be 3x3"))
    _require_periodic_box(ctx.box)
    @inbounds for i in f.atom_indices
        u, fx, fy, fz = _eval_wall_atom(f, q[1, i], q[2, i], q[3, i])
        if !isfinite(u)
            G .= Inf
            return G
        end
        _accum_cartesian_box_deriv!(G, ctx.box, q, i, fx, fy, fz)
    end
    return G
end

ResolvedSystem.accumulate_box_derivative!(G::Matrix{Float64},
                                          f::WallRestraintFactor,
                                          q::AbstractMatrix{Float64},
                                          ctx::Factors.Context) =
    Factors.accumulate_box_derivative!(G, f, q, ctx)
ResolvedSystem.box_derivative_support(::WallRestraintFactor) = nothing

# ==============================================================================
# 4. Müller–Plathe Reverse-NEMD (Shear Viscosity & Thermal Conductivity)
# ==============================================================================

@inline function _slab_index(box::Types.OrthorhombicBox, q::AbstractMatrix{Float64},
                             i::Int, grad_axis::Int, n_slabs::Int)::Int
    @inbounds s = mod(q[grad_axis, i] / box.L[grad_axis], 1.0)
    return clamp(floor(Int, s * n_slabs) + 1, 1, n_slabs)
end

@inline function _slab_index(box::Types.TriclinicBox, q::AbstractMatrix{Float64},
                             i::Int, grad_axis::Int, n_slabs::Int)::Int
    Hinv = box.Hinv
    @inbounds s_raw = Hinv[grad_axis, 1] * q[1, i] +
                      Hinv[grad_axis, 2] * q[2, i] +
                      Hinv[grad_axis, 3] * q[3, i]
    s = mod(s_raw, 1.0)
    return clamp(floor(Int, s * n_slabs) + 1, 1, n_slabs)
end

"""
    muller_plathe_shear_step!(state, masses, box; n_slabs=10, flow_axis=1, grad_axis=3) -> Float64

Müller–Plathe reverse-NEMD momentum exchange step.
Finds atom `a` in slab 1 with maximum `+v_flow` and atom `b` in slab `n_slabs ÷ 2 + 1`
with maximum `-v_flow` (minimum `v_flow`). Performs exact 1D elastic velocity exchange
along `flow_axis`:
`v'_a = ((m_a - m_b) v_a + 2 m_b v_b) / (m_a + m_b)`,
`v'_b = ((m_b - m_a) v_b + 2 m_a v_a) / (m_a + m_b)`,
which reduces identically to `v'_a = v_b, v'_b = v_a` when `m_a == m_b` and conserves
BOTH linear momentum and kinetic energy to machine precision for arbitrary masses.
Returns the non-negative transferred momentum `dp = p_new[flow, b] - p_old[flow, b]`
(or `0.0` if either slab is empty or `v_a <= v_b`).
"""
function muller_plathe_shear_step!(state, masses::AbstractVector{<:Real},
                                   box::Types.AbstractBox;
                                   n_slabs::Int = 10,
                                   flow_axis::Int = 1,
                                   grad_axis::Int = 3)::Float64
    box isa Types.NoBox && throw(Types.UnsupportedGeometry("Müller–Plathe RNEMD requires a periodic box"))
    n_slabs >= 2 || throw(ArgumentError("n_slabs must be >= 2"))
    1 <= flow_axis <= 3 || throw(ArgumentError("flow_axis must be 1, 2, or 3"))
    1 <= grad_axis <= 3 || throw(ArgumentError("grad_axis must be 1, 2, or 3"))
    flow_axis != grad_axis || throw(ArgumentError("flow_axis and grad_axis must be distinct"))

    q = state.q
    p = state.p
    N = size(q, 2)
    length(masses) == N || throw(DimensionMismatch("masses length != N=$N"))
    mid_slab = div(n_slabs, 2) + 1

    best_a = 0
    best_va = -Inf
    best_b = 0
    best_vb = Inf

    @inbounds for i in 1:N
        slab = _slab_index(box, q, i, grad_axis, n_slabs)
        vi = p[flow_axis, i] / Float64(masses[i])
        if slab == 1 && vi > best_va
            best_va = vi
            best_a = i
        elseif slab == mid_slab && vi < best_vb
            best_vb = vi
            best_b = i
        end
    end

    (best_a == 0 || best_b == 0 || best_va <= best_vb) && return 0.0

    ma = Float64(masses[best_a])
    mb = Float64(masses[best_b])
    invm = 1.0 / (ma + mb)
    va_new = ((ma - mb) * best_va + 2.0 * mb * best_vb) * invm
    vb_new = ((mb - ma) * best_vb + 2.0 * ma * best_va) * invm

    @inbounds begin
        p_b_old = p[flow_axis, best_b]
        p_b_new = mb * vb_new
        p[flow_axis, best_a] = ma * va_new
        p[flow_axis, best_b] = p_b_new
    end
    return p_b_new - p_b_old
end

"""
    estimate_shear_viscosity(total_dp, total_time, cross_area, dv_dx) -> Float64

Shear viscosity `η = j_z(p_x) / |∂v_x/∂z|`, where `j_z(p_x) = total_dp / (2 * cross_area * total_time)`.
"""
function estimate_shear_viscosity(total_dp::Real, total_time::Real,
                                  cross_area::Real, dv_dx::Real)::Float64
    dp = Float64(total_dp)
    dt = Float64(total_time)
    A = Float64(cross_area)
    g = abs(Float64(dv_dx))
    (isfinite(dt) && dt > 0.0) || throw(ArgumentError("total_time must be positive"))
    (isfinite(A) && A > 0.0) || throw(ArgumentError("cross_area must be positive"))
    (isfinite(g) && g > 0.0) || throw(ArgumentError("velocity gradient |dv_dx| must be positive"))
    jz = dp / (2.0 * A * dt)
    return jz / g
end

"""
    muller_plathe_thermal_step!(state, masses, box; n_slabs=10, grad_axis=3) -> Float64

Müller–Plathe reverse-NEMD thermal kinetic-energy swap step.
Finds the hottest atom `a` (maximum kinetic energy) in cold slab 1 and the coldest
atom `b` (minimum kinetic energy) in hot slab `n_slabs ÷ 2 + 1`. Applies exact 3D
elastic velocity exchange conserving BOTH total 3D linear momentum and total kinetic
energy to machine precision (reducing to bit-exact `v_a ↔ v_b` for equal masses).
Returns the transferred kinetic energy `dK = K'_b - K_b > 0` (or `0.0` if no positive
energy transfer occurs).
"""
function muller_plathe_thermal_step!(state, masses::AbstractVector{<:Real},
                                     box::Types.AbstractBox;
                                     n_slabs::Int = 10,
                                     grad_axis::Int = 3)::Float64
    box isa Types.NoBox && throw(Types.UnsupportedGeometry("Müller–Plathe RNEMD requires a periodic box"))
    n_slabs >= 2 || throw(ArgumentError("n_slabs must be >= 2"))
    1 <= grad_axis <= 3 || throw(ArgumentError("grad_axis must be 1, 2, or 3"))

    q = state.q
    p = state.p
    N = size(q, 2)
    length(masses) == N || throw(DimensionMismatch("masses length != N=$N"))
    mid_slab = div(n_slabs, 2) + 1

    best_a = 0
    best_ka = -Inf
    best_b = 0
    best_kb = Inf

    @inbounds for i in 1:N
        slab = _slab_index(box, q, i, grad_axis, n_slabs)
        mi = Float64(masses[i])
        ki = 0.5 * (p[1, i]^2 + p[2, i]^2 + p[3, i]^2) / mi
        if slab == 1 && ki > best_ka
            best_ka = ki
            best_a = i
        elseif slab == mid_slab && ki < best_kb
            best_kb = ki
            best_b = i
        end
    end

    (best_a == 0 || best_b == 0 || best_ka <= best_kb) && return 0.0

    ma = Float64(masses[best_a])
    mb = Float64(masses[best_b])
    invm = 1.0 / (ma + mb)

    @inbounds begin
        vax = p[1, best_a] / ma; vay = p[2, best_a] / ma; vaz = p[3, best_a] / ma
        vbx = p[1, best_b] / mb; vby = p[2, best_b] / mb; vbz = p[3, best_b] / mb

        vax_new = ((ma - mb) * vax + 2.0 * mb * vbx) * invm
        vay_new = ((ma - mb) * vay + 2.0 * mb * vby) * invm
        vaz_new = ((ma - mb) * vaz + 2.0 * mb * vbz) * invm

        vbx_new = ((mb - ma) * vbx + 2.0 * ma * vax) * invm
        vby_new = ((mb - ma) * vby + 2.0 * ma * vay) * invm
        vbz_new = ((mb - ma) * vbz + 2.0 * ma * vaz) * invm

        kb_new = 0.5 * mb * (vbx_new^2 + vby_new^2 + vbz_new^2)
        dK = kb_new - best_kb
        dK <= 0.0 && return 0.0

        p[1, best_a] = ma * vax_new; p[2, best_a] = ma * vay_new; p[3, best_a] = ma * vaz_new
        p[1, best_b] = mb * vbx_new; p[2, best_b] = mb * vby_new; p[3, best_b] = mb * vbz_new
        return dK
    end
end

"""
    estimate_thermal_conductivity(total_dK, total_time, cross_area, dT_dz) -> Float64

Thermal conductivity `κ = J_z / |dT/dz|`, where `J_z = total_dK / (2 * cross_area * total_time)`.
"""
function estimate_thermal_conductivity(total_dK::Real, total_time::Real,
                                       cross_area::Real, dT_dz::Real)::Float64
    dK = Float64(total_dK)
    dt = Float64(total_time)
    A = Float64(cross_area)
    g = abs(Float64(dT_dz))
    (isfinite(dt) && dt > 0.0) || throw(ArgumentError("total_time must be positive"))
    (isfinite(A) && A > 0.0) || throw(ArgumentError("cross_area must be positive"))
    (isfinite(g) && g > 0.0) || throw(ArgumentError("temperature gradient |dT_dz| must be positive"))
    Jz = dK / (2.0 * A * dt)
    return Jz / g
end

end # module NEMD
