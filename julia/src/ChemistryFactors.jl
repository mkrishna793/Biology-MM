"""
Phase-2 factor additions: the small terms that established parameter files
require but Phase 1 did not have.

Everything here follows the Phase-1 factor protocol
(`evaluate!(F, f, q, ctx) -> U`, scattering `F = -∂U/∂q`), so these factors
compose with the existing bonded and nonbonded terms and are checked by the
same finite-difference adjoint tests. No OpenMM type is referenced.

Units are the importer's units (nm, kJ/mol, e); the Coulomb prefactor comes from
`ctx.units`, never a hard-coded constant.
"""
module ChemistryFactors

using ..Types
using ..Types: SVector3, norm2, unwrap_displacement
using ..Factors
import ..ResolvedSystem
import ..SystemContract

const _RS = ResolvedSystem
const _SC = SystemContract

export MultiPeriodicDihedral, HarmonicImproper, UreyBradley, PairTableNonbonded, PairC64Table,
       with_lfs_c64, torsion_angle_grad, NBFIXPairTable,
       accumulate_box_derivative!, accumulate_box_derivative, FIXED_POINT_SCALE

const FIXED_POINT_SCALE = Float64(Int64(1) << 40)

@inline function _cf_frac_disp(b::Types.OrthorhombicBox, d::SVector3)
    L = b.L
    return (d.x / L[1], d.y / L[2], d.z / L[3])
end

@inline function _cf_frac_disp(b::Types.TriclinicBox, d::SVector3)
    M = b.Hinv
    return (M[1, 1] * d.x + M[1, 2] * d.y + M[1, 3] * d.z,
            M[2, 1] * d.x + M[2, 2] * d.y + M[2, 3] * d.z,
            M[3, 1] * d.x + M[3, 2] * d.y + M[3, 3] * d.z)
end

@inline _cf_frac_disp(::Types.NoBox, ::SVector3) =
    throw(_RS.UnsupportedCapability(:fractional_coordinates,
        "no periodic box: fractional displacements need lattice vectors"))

@inline function _cf_outer_add!(G::Matrix{Float64}, c::Float64,
                                d::SVector3, w::NTuple{3,Float64})
    gx = c * d.x; gy = c * d.y; gz = c * d.z
    @inbounds begin
        G[1, 1] += gx * w[1]; G[1, 2] += gx * w[2]; G[1, 3] += gx * w[3]
        G[2, 1] += gy * w[1]; G[2, 2] += gy * w[2]; G[2, 3] += gy * w[3]
        G[3, 1] += gz * w[1]; G[3, 2] += gz * w[2]; G[3, 3] += gz * w[3]
    end
    return G
end

@inline function _cf_torsion_box_derivative!(G::Matrix{Float64},
                                             box::Types.AbstractBox,
                                             q::AbstractMatrix{Float64},
                                             i::Int, j::Int, k::Int, l::Int,
                                             dU_dphi::Float64,
                                             gj::SVector3, gk::SVector3, gl::SVector3)
    w1 = _cf_frac_disp(box, unwrap_displacement(box, q, j, i))
    w2 = _cf_frac_disp(box, unwrap_displacement(box, q, k, j))
    w3 = _cf_frac_disp(box, unwrap_displacement(box, q, l, k))
    sk = (w1[1] + w2[1], w1[2] + w2[2], w1[3] + w2[3])
    sl = (sk[1] + w3[1], sk[2] + w3[2], sk[3] + w3[3])
    _cf_outer_add!(G, dU_dphi, gj, w1)
    _cf_outer_add!(G, dU_dphi, gk, sk)
    _cf_outer_add!(G, dU_dphi, gl, sl)
    return G
end

# Same signed torsion convention as Factors.PeriodicDihedral. Kept as a thin
# alias so a test can check that all multi-term torsions use one angle
# definition rather than a second, silently different one.
torsion_angle_grad(q, box, i, j, k, l) = Factors.dihedral_angle_grad(q, box, i, j, k, l)

# ---------------------------------------------------------------------------
# multiple proper-torsion multiplicities on one atom quadruple
# ---------------------------------------------------------------------------

"""
`U = Σ_m k_m [1 + cos(n_m φ - φ0_m)]` for one `i-j-k-l` quadruple.

AMBER prmtop files list each multiplicity as a separate dihedral entry; CHARMM
parameter files use a type list. Collapsing them to the first term is a silent
physics error, which is why the importer groups the terms into this factor
instead of emitting duplicate `PeriodicDihedral` factors (the resolved-system
schema forbids duplicate interaction keys).
"""
struct MultiPeriodicDihedral <: Factors.AbstractFactor
    i::Int
    j::Int
    k::Int
    l::Int
    ks::Vector{Float64}
    ns::Vector{Int}
    phases::Vector{Float64}
    function MultiPeriodicDihedral(i::Integer, j::Integer, k::Integer, l::Integer,
                                   ks::AbstractVector{<:Real},
                                   ns::AbstractVector{<:Integer},
                                   phases::AbstractVector{<:Real})
        length(ks) == length(ns) == length(phases) ||
            throw(ArgumentError("MultiPeriodicDihedral needs one k, n and phase per term"))
        isempty(ks) && throw(ArgumentError("MultiPeriodicDihedral needs at least one term"))
        all(>(0), ns) || throw(ArgumentError("torsion multiplicity must be >= 1"))
        all(isfinite, ks) && all(isfinite, phases) ||
            throw(ArgumentError("torsion parameters must be finite"))
        return new(Int(i), Int(j), Int(k), Int(l), collect(Float64, ks),
                   collect(Int, ns), collect(Float64, phases))
    end
end

Factors.factor_support_size(::MultiPeriodicDihedral) = 4
Factors.read_support(f::MultiPeriodicDihedral) = (f.i, f.j, f.k, f.l)

function Factors.evaluate!(F::AbstractMatrix{Float64}, f::MultiPeriodicDihedral,
                           q::AbstractMatrix{Float64}, ctx::Factors.Context)
    t = torsion_angle_grad(q, ctx.box, f.i, f.j, f.k, f.l)
    t === nothing && return Inf
    phi, (gi, gj, gk, gl) = t
    u = 0.0
    mult = 0.0
    @inbounds for m in eachindex(f.ks)
        arg = f.ns[m] * phi - f.phases[m]
        u += f.ks[m] * (1.0 + cos(arg))
        mult += f.ks[m] * f.ns[m] * sin(arg)
    end
    # F = -dU/dq = (Σ k n sin(arg)) ∇φ
    @inbounds for d in 1:3
        F[d, f.i] += mult * Factors._comp(gi, d)
        F[d, f.j] += mult * Factors._comp(gj, d)
        F[d, f.k] += mult * Factors._comp(gk, d)
        F[d, f.l] += mult * Factors._comp(gl, d)
    end
    return u
end

_RS.box_derivative_support(::MultiPeriodicDihedral) = nothing

function _RS.accumulate_box_derivative!(G::Matrix{Float64}, f::MultiPeriodicDihedral,
                                        q::AbstractMatrix{Float64}, ctx::Factors.Context)
    t = torsion_angle_grad(q, ctx.box, f.i, f.j, f.k, f.l)
    if t === nothing
        G .= Inf
        return G
    end
    phi, (_, gj, gk, gl) = t
    dU_dphi = 0.0
    @inbounds for m in eachindex(f.ks)
        dU_dphi -= f.ks[m] * f.ns[m] * sin(f.ns[m] * phi - f.phases[m])
    end
    return _cf_torsion_box_derivative!(G, ctx.box, q, f.i, f.j, f.k, f.l,
                                       dU_dphi, gj, gk, gl)
end

# ---------------------------------------------------------------------------
# CHARMM/AMBER harmonic improper
# ---------------------------------------------------------------------------

"Wrap an angle difference into `[-π, π]`."
@inline function _wrap_pi(x::Float64)
    return x - 2.0 * pi * round(x / (2.0 * pi))
end

"""
`U = k (wrap(φ - φ0))²` with `k` in energy/rad².

This is the form used by AMBER impropers (OpenMM's `CustomTorsionForce` in
`amber_file_parser.py`) and by CHARMM harmonic impropers (`n = 0`). The angle
`φ` is the same signed torsion as `Factors.PeriodicDihedral`; the wrapped
difference makes the potential 2π-periodic and finite across the atan2 branch
cut. `k` may be negative (CHARMM parameter files do use negative `psi_k`).
"""
struct HarmonicImproper <: Factors.AbstractFactor
    i::Int
    j::Int
    k::Int
    l::Int
    k_phi::Float64
    phi0::Float64
    function HarmonicImproper(i::Integer, j::Integer, k::Integer, l::Integer,
                              k_phi::Real, phi0::Real)
        isfinite(k_phi) && isfinite(phi0) ||
            throw(ArgumentError("improper parameters must be finite"))
        return new(Int(i), Int(j), Int(k), Int(l), Float64(k_phi), Float64(phi0))
    end
end

Factors.factor_support_size(::HarmonicImproper) = 4
Factors.read_support(f::HarmonicImproper) = (f.i, f.j, f.k, f.l)

function Factors.evaluate!(F::AbstractMatrix{Float64}, f::HarmonicImproper,
                           q::AbstractMatrix{Float64}, ctx::Factors.Context)
    t = torsion_angle_grad(q, ctx.box, f.i, f.j, f.k, f.l)
    t === nothing && return Inf
    phi, (gi, gj, gk, gl) = t
    dphi = _wrap_pi(phi - f.phi0)
    u = f.k_phi * dphi * dphi
    # F = -dU/dq = -2 k dphi ∇φ
    mult = -2.0 * f.k_phi * dphi
    @inbounds for d in 1:3
        F[d, f.i] += mult * Factors._comp(gi, d)
        F[d, f.j] += mult * Factors._comp(gj, d)
        F[d, f.k] += mult * Factors._comp(gk, d)
        F[d, f.l] += mult * Factors._comp(gl, d)
    end
    return u
end

_RS.box_derivative_support(::HarmonicImproper) = nothing

function _RS.accumulate_box_derivative!(G::Matrix{Float64}, f::HarmonicImproper,
                                        q::AbstractMatrix{Float64}, ctx::Factors.Context)
    t = torsion_angle_grad(q, ctx.box, f.i, f.j, f.k, f.l)
    if t === nothing
        G .= Inf
        return G
    end
    phi, (_, gj, gk, gl) = t
    dphi = _wrap_pi(phi - f.phi0)
    dU_dphi = 2.0 * f.k_phi * dphi
    return _cf_torsion_box_derivative!(G, ctx.box, q, f.i, f.j, f.k, f.l,
                                       dU_dphi, gj, gk, gl)
end

# ---------------------------------------------------------------------------
# Urey-Bradley 1-3 harmonic
# ---------------------------------------------------------------------------

"""
`U = k (r13 - r0)²` for the 1-3 distance of an angle.

CHARMM's functional form has no `1/2` factor; the importer converts `k` from
kcal mol⁻¹ Å⁻² to kJ mol⁻¹ nm⁻² and stores the coefficient of exactly this
expression.
"""
struct UreyBradley <: Factors.AbstractFactor
    i::Int
    j::Int
    k::Float64
    r0::Float64
    function UreyBradley(i::Integer, j::Integer, k::Real, r0::Real)
        isfinite(k) && isfinite(r0) && r0 > 0 ||
            throw(ArgumentError("Urey-Bradley k must be finite and r0 must be positive"))
        return new(Int(i), Int(j), Float64(k), Float64(r0))
    end
end

Factors.factor_support_size(::UreyBradley) = 2
Factors.read_support(f::UreyBradley) = (f.i, f.j)

function Factors.evaluate!(F::AbstractMatrix{Float64}, f::UreyBradley,
                           q::AbstractMatrix{Float64}, ctx::Factors.Context)
    d = unwrap_displacement(ctx.box, q, f.i, f.j)
    r2 = norm2(d)
    r2 < 1e-24 && return Inf
    r = sqrt(r2)
    dr = r - f.r0
    u = f.k * dr * dr
    c = -2.0 * f.k * dr / r
    @inbounds for m in 1:3
        g = c * Factors._comp(d, m)
        F[m, f.i] += g
        F[m, f.j] -= g
    end
    return u
end

_RS.box_derivative_support(::UreyBradley) = nothing

function _RS.accumulate_box_derivative!(G::Matrix{Float64}, f::UreyBradley,
                                        q::AbstractMatrix{Float64}, ctx::Factors.Context)
    d = unwrap_displacement(ctx.box, q, f.i, f.j)
    r2 = norm2(d)
    if r2 < 1e-24
        G .= Inf
        return G
    end
    r = sqrt(r2)
    dr = r - f.r0
    _cf_outer_add!(G, 2.0 * f.k * dr / r, d, _cf_frac_disp(ctx.box, d))
    return G
end

# ---------------------------------------------------------------------------
# explicit non-periodic pair table
# ---------------------------------------------------------------------------

"""
Complete non-periodic nonbonded list for imported small systems (no box).

Each record is one unordered pair with precomputed Lorentz–Berthelot
`sigma`/`epsilon`, `qi*qj`, and the 1-4 multipliers. This is not a neighbour
list and it does not rebuild: the factor is the *specification* of a
non-periodic pair sum, with all pairs listed explicitly. It is used for
parameter files that carry no periodic box (e.g. an unsolvated DNA duplex), so
there is no cutoff and no minimum-image model.

`mode = :none` is the exact pair sum; `mode = :trunc` reproduces an unshifted
12-6 cutoff for reference evaluations (discontinuous at `rcut`, so
`force_consistent` is false).
"""
struct PairTableNonbonded <: Factors.AbstractFactor
    i::Vector{Int}
    j::Vector{Int}
    sigma::Vector{Float64}
    epsilon::Vector{Float64}
    qiqj::Vector{Float64}
    lj_scale::Vector{Float64}
    coul_scale::Vector{Float64}
    mode::Symbol
    rcut::Float64
    support::Vector{Int}
    function PairTableNonbonded(i::AbstractVector{<:Integer},
                                j::AbstractVector{<:Integer},
                                sigma::AbstractVector{<:Real},
                                epsilon::AbstractVector{<:Real},
                                qiqj::AbstractVector{<:Real},
                                lj_scale::AbstractVector{<:Real},
                                coul_scale::AbstractVector{<:Real};
                                mode::Symbol = :none, rcut::Real = Inf)
        n = length(i)
        length(j) == length(sigma) == length(epsilon) == length(qiqj) ==
            length(lj_scale) == length(coul_scale) == n ||
            throw(ArgumentError("pair table columns must have equal length"))
        mode in (:none, :trunc) || throw(ArgumentError("mode must be :none or :trunc"))
        if mode === :trunc
            isfinite(rcut) && rcut > 0 || throw(ArgumentError(":trunc requires a positive finite rcut"))
        end
        seen = Set{Tuple{Int,Int}}()
        for p in 1:n
            i[p] != j[p] || throw(ArgumentError("pair $p has identical atoms"))
            i[p] > 0 && j[p] > 0 || throw(ArgumentError("pair $p has a non-positive index"))
            isfinite(sigma[p]) && sigma[p] >= 0 || throw(ArgumentError("pair $p has invalid sigma"))
            isfinite(epsilon[p]) && epsilon[p] >= 0 || throw(ArgumentError("pair $p has invalid epsilon"))
            isfinite(qiqj[p]) || throw(ArgumentError("pair $p has invalid charge product"))
            isfinite(lj_scale[p]) && lj_scale[p] >= 0 || throw(ArgumentError("pair $p has invalid LJ scale"))
            isfinite(coul_scale[p]) && coul_scale[p] >= 0 || throw(ArgumentError("pair $p has invalid Coulomb scale"))
            key = i[p] < j[p] ? (Int(i[p]), Int(j[p])) : (Int(j[p]), Int(i[p]))
            key in seen && throw(ArgumentError("duplicate pair $(key)"))
            push!(seen, key)
        end
        supp = sort!(unique!(vcat(collect(Int, i), collect(Int, j))))
        return new(collect(Int, i), collect(Int, j), collect(Float64, sigma),
                   collect(Float64, epsilon), collect(Float64, qiqj),
                   collect(Float64, lj_scale), collect(Float64, coul_scale),
                   mode, Float64(rcut), supp)
    end
end

Factors.factor_support_size(f::PairTableNonbonded) = length(f.support)
Factors.read_support(f::PairTableNonbonded) = (f.support...,)
Factors.is_local(::PairTableNonbonded) = true
Factors.force_consistent(f::PairTableNonbonded) = f.mode === :none
_RS.box_derivative_support(::PairTableNonbonded) =
    "PairTableNonbonded is non-periodic only (NoBox); no periodic box derivative exists"

function Factors.evaluate!(F::AbstractMatrix{Float64}, f::PairTableNonbonded,
                           q::AbstractMatrix{Float64}, ctx::Factors.Context)
    ctx.box isa Types.NoBox ||
        throw(Types.UnsupportedGeometry(
            "PairTableNonbonded is a non-periodic explicit pair list; got $(typeof(ctx.box)). " *
            "Periodic systems use StreamedLJ + PME instead."))
    size(q, 1) == 3 && size(F) == size(q) ||
        throw(ArgumentError("q and F must be 3 x N, got q=$(size(q)), F=$(size(F))"))
    (isempty(f.support) || f.support[end] <= size(q, 2)) ||
        throw(BoundsError(q, (3, f.support[end])))
    pref = ctx.units.coulomb_prefactor
    u = 0.0
    @inbounds for p in eachindex(f.i)
        i = f.i[p]; j = f.j[p]
        d = unwrap_displacement(ctx.box, q, i, j)
        r2 = norm2(d)
        if f.mode === :trunc && r2 > f.rcut * f.rcut
            continue
        end
        r2 < 1e-24 && return Inf
        r = sqrt(r2)
        uu = 0.0
        dudr = 0.0
        if f.lj_scale[p] != 0.0 && f.epsilon[p] != 0.0
            s6 = (f.sigma[p] / r)^6
            uu += f.lj_scale[p] * 4.0 * f.epsilon[p] * (s6 * s6 - s6)
            dudr += f.lj_scale[p] * 24.0 * f.epsilon[p] * (s6 - 2.0 * s6 * s6) / r
        end
        if f.coul_scale[p] != 0.0 && f.qiqj[p] != 0.0
            uu += f.coul_scale[p] * pref * f.qiqj[p] / r
            dudr += -f.coul_scale[p] * pref * f.qiqj[p] / r2
        end
        u += uu
        c = -dudr / r
        @inbounds for m in 1:3
            g = c * Factors._comp(d, m)
            F[m, i] += g
            F[m, j] -= g
        end
    end
    return u
end


# ---------------------------------------------------------------------------
# AMBER 1-6-4 dispersion correction (-C/r^4)
# ---------------------------------------------------------------------------

"""
`U = Σ_pairs -C_ij / r^4` for the AMBER "1-6-4" Lennard-Jones variant.

AMBER prmtops that carry `LENNARD_JONES_CCOEF` define a per-type-pair `C`
coefficient in addition to the usual 12-6 terms (OpenMM implements it as a
`CustomNonbondedForce` with `-c/r^4`). It is a distinct pair term, not a
rescaling of σ/ε, so it gets its own factor. The explicit pair list is produced
by the importer after removing excluded and 1-4 pairs, exactly like OpenMM's
`CustomNonbondedForce` exclusions.

`mode = :trunc` is the reference (unshifted) cutoff used by OpenMM; it is not
force-consistent across the discontinuity, so the dynamics gate refuses it.
"""
struct PairC64Table <: Factors.AbstractFactor
    i::Vector{Int}
    j::Vector{Int}
    c::Vector{Float64}       # kJ/mol * nm^4
    mode::Symbol
    rcut::Float64
    support::Vector{Int}
    threaded::Bool
    fixed_point::Bool
    function PairC64Table(i::AbstractVector{<:Integer}, j::AbstractVector{<:Integer},
                          c::AbstractVector{<:Real}; mode::Symbol = :trunc,
                          rcut::Real = Inf, threaded::Bool = false, fixed_point::Bool = false)
        n = length(i)
        length(j) == length(c) == n ||
            throw(ArgumentError("1-6-4 pair columns must have equal length"))
        mode in (:none, :trunc, :lfs) || throw(ArgumentError("mode must be :none, :trunc, or :lfs"))
        if mode in (:trunc, :lfs)
            isfinite(rcut) && rcut > 0 ||
                throw(ArgumentError(":$mode requires a positive finite rcut"))
        end
        seen = Set{Tuple{Int,Int}}()
        for p in 1:n
            i[p] != j[p] || throw(ArgumentError("1-6-4 pair $p has identical atoms"))
            i[p] > 0 && j[p] > 0 || throw(ArgumentError("1-6-4 pair $p has a non-positive index"))
            isfinite(c[p]) || throw(ArgumentError("1-6-4 pair $p has a non-finite C"))
            key = i[p] < j[p] ? (Int(i[p]), Int(j[p])) : (Int(j[p]), Int(i[p]))
            key in seen && throw(ArgumentError("duplicate 1-6-4 pair $(key)"))
            push!(seen, key)
        end
        supp = sort!(unique!(vcat(collect(Int, i), collect(Int, j))))
        return new(collect(Int, i), collect(Int, j), collect(Float64, c),
                   mode, Float64(rcut), supp, threaded, fixed_point)
    end
end

"""
    with_lfs_c64(factors) -> Vector{Factors.AbstractFactor}

Return a copy of `factors` where any `PairC64Table` with `mode == :trunc` is
upgraded to `mode = :lfs` (C¹ linear force-shifted cutoff at `rcut`) for conservative MD.
"""
function with_lfs_c64(f::PairC64Table)
    return f.mode === :trunc ? PairC64Table(f.i, f.j, f.c; mode = :lfs, rcut = f.rcut,
                                           threaded = f.threaded, fixed_point = f.fixed_point) : f
end

function with_lfs_c64(factors::AbstractVector)
    return Factors.AbstractFactor[
        (f isa PairC64Table && f.mode === :trunc) ? with_lfs_c64(f) : f
        for f in factors
    ]
end

function with_lfs_c64(rs::_RS.ResolvedSystem)
    any(f -> f isa PairC64Table && f.mode === :trunc, rs.factors) || return rs
    new_factors = with_lfs_c64(rs.factors)
    return _RS.ResolvedSystem(
        masses = rs.atoms.masses,
        charges = rs.atoms.charges,
        names = rs.atoms.names,
        box = rs.box,
        topology = rs.topology,
        factors = new_factors,
        component_names = rs.component_names,
        constraints = rs.constraints,
        virtual_sites = rs.atoms.virtual_sites,
        units = rs.units,
        provenance = rs.provenance,
        declared = rs.declared,
        schema = rs.schema,
    )
end

Factors.factor_support_size(f::PairC64Table) = length(f.support)
Factors.read_support(f::PairC64Table) = (f.support...,)
Factors.is_local(::PairC64Table) = true
Factors.force_consistent(f::PairC64Table) = f.mode in (:none, :lfs)
_RS.box_derivative_support(f::PairC64Table) =
    f.mode in (:none, :lfs) ? nothing : "mode $(f.mode) is not force-consistent at the cutoff"

function Factors.evaluate!(F::AbstractMatrix{Float64}, f::PairC64Table,
                           q::AbstractMatrix{Float64}, ctx::Factors.Context)
    size(q, 1) == 3 && size(F) == size(q) ||
        throw(ArgumentError("q and F must be 3 x N, got q=$(size(q)), F=$(size(F))"))
    (isempty(f.support) || f.support[end] <= size(q, 2)) ||
        throw(BoundsError(q, (3, f.support[end])))
    rc = f.rcut
    rc2 = rc * rc
    is_lfs = f.mode === :lfs
    inv_rc4 = is_lfs ? 1.0 / (rc2 * rc2) : 0.0
    inv_rc5 = is_lfs ? inv_rc4 / rc : 0.0
    n = size(q, 2)

    if f.threaded && Threads.nthreads() > 1
        nth = Threads.maxthreadid()
        F_ints = [zeros(Int64, 3, n) for _ in 1:nth]
        u_threads = zeros(Float64, nth)
        has_inf = Threads.Atomic{Bool}(false)

        Threads.@threads :static for p in eachindex(f.i)
            has_inf[] && continue
            tid = Threads.threadid()
            i = f.i[p]; j = f.j[p]
            d = unwrap_displacement(ctx.box, q, i, j)
            r2 = norm2(d)
            f.mode !== :none && r2 > rc2 && continue
            if r2 < 1e-24
                has_inf[] = true
                continue
            end
            r4 = r2 * r2
            cp = f.c[p]
            if is_lfs
                r = sqrt(r2)
                u_threads[tid] += -cp / r4 + cp * inv_rc4 - (r - rc) * (4.0 * cp * inv_rc5)
                coeff = -4.0 * cp * (1.0 / (r4 * r2) - inv_rc5 / r)
            else
                u_threads[tid] -= cp / r4
                coeff = -4.0 * cp / (r4 * r2)
            end
            gx = coeff * d.x; gy = coeff * d.y; gz = coeff * d.z
            fx_int = round(Int64, gx * FIXED_POINT_SCALE)
            fy_int = round(Int64, gy * FIXED_POINT_SCALE)
            fz_int = round(Int64, gz * FIXED_POINT_SCALE)
            F_ints[tid][1, i] += fx_int
            F_ints[tid][2, i] += fy_int
            F_ints[tid][3, i] += fz_int
            F_ints[tid][1, j] -= fx_int
            F_ints[tid][2, j] -= fy_int
            F_ints[tid][3, j] -= fz_int
        end
        has_inf[] && return Inf

        F_int = zeros(Int64, 3, n)
        for tid in 1:nth
            @inbounds for a in 1:n, m in 1:3
                F_int[m, a] += F_ints[tid][m, a]
            end
        end
        inv_scale = 1.0 / FIXED_POINT_SCALE
        @inbounds for a in 1:n, m in 1:3
            F[m, a] += Float64(F_int[m, a]) * inv_scale
        end
        return sum(u_threads)
    elseif f.fixed_point
        F_int = zeros(Int64, 3, n)
        u = 0.0
        @inbounds for p in eachindex(f.i)
            i = f.i[p]; j = f.j[p]
            d = unwrap_displacement(ctx.box, q, i, j)
            r2 = norm2(d)
            f.mode !== :none && r2 > rc2 && continue
            r2 < 1e-24 && return Inf
            r4 = r2 * r2
            cp = f.c[p]
            if is_lfs
                r = sqrt(r2)
                u += -cp / r4 + cp * inv_rc4 - (r - rc) * (4.0 * cp * inv_rc5)
                coeff = -4.0 * cp * (1.0 / (r4 * r2) - inv_rc5 / r)
            else
                u -= cp / r4
                coeff = -4.0 * cp / (r4 * r2)
            end
            gx = coeff * d.x; gy = coeff * d.y; gz = coeff * d.z
            fx_int = round(Int64, gx * FIXED_POINT_SCALE)
            fy_int = round(Int64, gy * FIXED_POINT_SCALE)
            fz_int = round(Int64, gz * FIXED_POINT_SCALE)
            F_int[1, i] += fx_int
            F_int[2, i] += fy_int
            F_int[3, i] += fz_int
            F_int[1, j] -= fx_int
            F_int[2, j] -= fy_int
            F_int[3, j] -= fz_int
        end
        inv_scale = 1.0 / FIXED_POINT_SCALE
        @inbounds for a in 1:n, m in 1:3
            F[m, a] += Float64(F_int[m, a]) * inv_scale
        end
        return u
    else
        u = 0.0
        @inbounds for p in eachindex(f.i)
            i = f.i[p]; j = f.j[p]
            d = unwrap_displacement(ctx.box, q, i, j)
            r2 = norm2(d)
            f.mode !== :none && r2 > rc2 && continue
            r2 < 1e-24 && return Inf
            r4 = r2 * r2
            cp = f.c[p]
            if is_lfs
                r = sqrt(r2)
                u += -cp / r4 + cp * inv_rc4 - (r - rc) * (4.0 * cp * inv_rc5)
                coeff = -4.0 * cp * (1.0 / (r4 * r2) - inv_rc5 / r)
            else
                u -= cp / r4
                coeff = -4.0 * cp / (r4 * r2)
            end
            @inbounds for m in 1:3
                g = coeff * Factors._comp(d, m)
                F[m, i] += g
                F[m, j] -= g
            end
        end
        return u
    end
end

Factors.evaluate!(f::PairC64Table, q::AbstractMatrix{Float64}, F::AbstractMatrix{Float64}, ctx) =
    Factors.evaluate!(F, f, q, ctx)

function _RS.accumulate_box_derivative!(G::Matrix{Float64}, f::PairC64Table,
                                        q::AbstractMatrix{Float64}, ctx::Factors.Context)
    f.mode in (:none, :lfs) || throw(_SC.MissingBoxDerivative(:lj146,
        "mode $(f.mode) is not force-consistent at the cutoff"))
    size(q, 1) == 3 || throw(ArgumentError("q must be 3 x N, got $(size(q))"))
    (isempty(f.support) || f.support[end] <= size(q, 2)) ||
        throw(BoundsError(q, (3, f.support[end])))
    rc = f.rcut
    rc2 = rc * rc
    is_lfs = f.mode === :lfs
    inv_rc5 = is_lfs ? 1.0 / (rc2 * rc2 * rc) : 0.0
    @inbounds for p in eachindex(f.i)
        d = unwrap_displacement(ctx.box, q, f.i[p], f.j[p])
        r2 = norm2(d)
        f.mode !== :none && r2 > rc2 && continue
        if r2 < 1e-24
            G .= Inf
            return G
        end
        r4 = r2 * r2
        cp = f.c[p]
        scale = is_lfs ? 4.0 * cp * (1.0 / (r4 * r2) - inv_rc5 / sqrt(r2)) : 4.0 * cp / (r4 * r2)
        _cf_outer_add!(G, scale, d, _cf_frac_disp(ctx.box, d))
    end
    return G
end

# ---------------------------------------------------------------------------
# Generalized Off-Diagonal Lennard-Jones Pair Table (NBFIXPairTable)
# ---------------------------------------------------------------------------

"""
`NBFIXPairTable <: Factors.AbstractFactor`

Biomolecular force fields (CHARMM36 NBFIX, AMBER off-diagonal ion-carbonyl/phosphate overrides)
require pair-specific `(sigma_ij, epsilon_ij)` that override Lorentz-Berthelot combining rules.
Supports `:none`, `:trunc`, and C¹ linear force-shifted `:lfs` modes across `NoBox`,
`OrthorhombicBox`, and `TriclinicBox`.
"""
struct NBFIXPairTable <: Factors.AbstractFactor
    i::Vector{Int}
    j::Vector{Int}
    c6::Vector{Float64}
    c12::Vector{Float64}
    scale::Vector{Float64}
    mode::Symbol
    rcut::Float64
    threaded::Bool
    fixed_point::Bool
end

function NBFIXPairTable(i::AbstractVector{<:Integer}, j::AbstractVector{<:Integer},
                        c6::AbstractVector{<:Real}, c12::AbstractVector{<:Real},
                        scale::AbstractVector{<:Real};
                        mode::Symbol = :none, rcut::Real = Inf,
                        threaded::Bool = false, fixed_point::Bool = false)
    n = length(i)
    length(j) == length(c6) == length(c12) == length(scale) == n ||
        throw(ArgumentError("NBFIXPairTable columns must have equal length"))
    mode in (:none, :trunc, :lfs) || throw(ArgumentError("mode must be :none, :trunc, or :lfs"))
    if mode in (:trunc, :lfs)
        isfinite(rcut) && rcut > 0 || throw(ArgumentError(":$mode requires a positive finite rcut"))
    end
    for k in 1:n
        i[k] != j[k] || throw(ArgumentError("pair $k has identical atoms"))
        i[k] > 0 && j[k] > 0 || throw(ArgumentError("pair $k has non-positive atom index"))
        isfinite(c6[k]) && c6[k] >= 0 || throw(ArgumentError("pair $k has invalid c6"))
        isfinite(c12[k]) && c12[k] >= 0 || throw(ArgumentError("pair $k has invalid c12"))
        isfinite(scale[k]) && scale[k] >= 0 || throw(ArgumentError("pair $k has invalid scale"))
    end
    return NBFIXPairTable(collect(Int, i), collect(Int, j), collect(Float64, c6),
                          collect(Float64, c12), collect(Float64, scale),
                          mode, Float64(rcut), threaded, fixed_point)
end

function NBFIXPairTable(i::AbstractVector{<:Integer}, j::AbstractVector{<:Integer},
                        sigma::AbstractVector{<:Real}, epsilon::AbstractVector{<:Real};
                        scale::AbstractVector{<:Real} = ones(Float64, length(i)),
                        mode::Symbol = :none, rcut::Real = Inf,
                        threaded::Bool = false, fixed_point::Bool = false)
    n = length(i)
    length(j) == length(sigma) == length(epsilon) == length(scale) == n ||
        throw(ArgumentError("NBFIXPairTable columns must have equal length"))
    c6 = zeros(Float64, n)
    c12 = zeros(Float64, n)
    for k in 1:n
        isfinite(sigma[k]) && sigma[k] > 0 || throw(ArgumentError("pair $k has invalid sigma (must be > 0)"))
        isfinite(epsilon[k]) && epsilon[k] >= 0 || throw(ArgumentError("pair $k has invalid epsilon (must be >= 0)"))
        s6 = Float64(sigma[k])^6
        eps = Float64(epsilon[k])
        c6[k] = 4.0 * eps * s6
        c12[k] = 4.0 * eps * s6 * s6
    end
    return NBFIXPairTable(i, j, c6, c12, scale; mode = mode, rcut = rcut,
                          threaded = threaded, fixed_point = fixed_point)
end

Factors.is_local(::NBFIXPairTable) = false
Factors.force_consistent(f::NBFIXPairTable) = f.mode !== :trunc
Factors.factor_support_size(f::NBFIXPairTable) = isempty(f.i) ? 0 : length(unique(vcat(f.i, f.j)))
Factors.read_support(f::NBFIXPairTable) = isempty(f.i) ? () : Tuple(sort!(unique(vcat(f.i, f.j))))
_RS.box_derivative_support(f::NBFIXPairTable) =
    f.mode in (:none, :lfs) ? nothing : "mode $(f.mode) is not force-consistent at the cutoff"

function Factors.evaluate!(F::AbstractMatrix{Float64}, f::NBFIXPairTable,
                           q::AbstractMatrix{Float64}, ctx)
    box = hasproperty(ctx, :box) ? ctx.box : ctx
    size(q, 1) == 3 && size(F) == size(q) ||
        throw(ArgumentError("q and F must be 3 x N, got q=$(size(q)), F=$(size(F))"))
    n = size(q, 2)
    if !isempty(f.i)
        max_idx = max(maximum(f.i), maximum(f.j))
        max_idx <= n || throw(BoundsError(q, (3, max_idx)))
    end

    rc = f.rcut
    rc2 = rc * rc
    is_lfs = f.mode === :lfs

    if f.threaded && Threads.nthreads() > 1
        nth = Threads.maxthreadid()
        F_ints = [zeros(Int64, 3, n) for _ in 1:nth]
        u_threads = zeros(Float64, nth)
        has_inf = Threads.Atomic{Bool}(false)

        Threads.@threads :static for k in eachindex(f.i)
            has_inf[] && continue
            tid = Threads.threadid()
            i = f.i[k]; j = f.j[k]
            d = unwrap_displacement(box, q, i, j)
            r2 = norm2(d)
            f.mode !== :none && r2 > rc2 && continue
            if r2 < 1e-24
                has_inf[] = true
                continue
            end
            r4 = r2 * r2
            r6 = r4 * r2
            r12 = r6 * r6
            c6 = f.c6[k]; c12 = f.c12[k]; sc = f.scale[k]
            u0 = c12 / r12 - c6 / r6
            if is_lfs
                r = sqrt(r2)
                rc6 = rc2^3
                rc12 = rc6^2
                uc = c12 / rc12 - c6 / rc6
                duc = -12.0 * c12 / (rc12 * rc) + 6.0 * c6 / (rc6 * rc)
                u_threads[tid] += sc * (u0 - uc - duc * (r - rc))
                coeff = sc * (12.0 * c12 / (r12 * r2) - 6.0 * c6 / (r6 * r2) + duc / r)
            else
                u_threads[tid] += sc * u0
                coeff = sc * (12.0 * c12 / (r12 * r2) - 6.0 * c6 / (r6 * r2))
            end
            gx = coeff * d.x; gy = coeff * d.y; gz = coeff * d.z
            fx_int = round(Int64, gx * FIXED_POINT_SCALE)
            fy_int = round(Int64, gy * FIXED_POINT_SCALE)
            fz_int = round(Int64, gz * FIXED_POINT_SCALE)
            F_ints[tid][1, i] += fx_int
            F_ints[tid][2, i] += fy_int
            F_ints[tid][3, i] += fz_int
            F_ints[tid][1, j] -= fx_int
            F_ints[tid][2, j] -= fy_int
            F_ints[tid][3, j] -= fz_int
        end
        has_inf[] && return Inf

        F_int = zeros(Int64, 3, n)
        for tid in 1:nth
            @inbounds for a in 1:n, m in 1:3
                F_int[m, a] += F_ints[tid][m, a]
            end
        end
        inv_scale = 1.0 / FIXED_POINT_SCALE
        @inbounds for a in 1:n, m in 1:3
            F[m, a] += Float64(F_int[m, a]) * inv_scale
        end
        return sum(u_threads)
    elseif f.fixed_point
        F_int = zeros(Int64, 3, n)
        u = 0.0
        @inbounds for k in eachindex(f.i)
            i = f.i[k]; j = f.j[k]
            d = unwrap_displacement(box, q, i, j)
            r2 = norm2(d)
            f.mode !== :none && r2 > rc2 && continue
            r2 < 1e-24 && return Inf
            r4 = r2 * r2
            r6 = r4 * r2
            r12 = r6 * r6
            c6 = f.c6[k]; c12 = f.c12[k]; sc = f.scale[k]
            u0 = c12 / r12 - c6 / r6
            if is_lfs
                r = sqrt(r2)
                rc6 = rc2^3
                rc12 = rc6^2
                uc = c12 / rc12 - c6 / rc6
                duc = -12.0 * c12 / (rc12 * rc) + 6.0 * c6 / (rc6 * rc)
                u += sc * (u0 - uc - duc * (r - rc))
                coeff = sc * (12.0 * c12 / (r12 * r2) - 6.0 * c6 / (r6 * r2) + duc / r)
            else
                u += sc * u0
                coeff = sc * (12.0 * c12 / (r12 * r2) - 6.0 * c6 / (r6 * r2))
            end
            gx = coeff * d.x; gy = coeff * d.y; gz = coeff * d.z
            fx_int = round(Int64, gx * FIXED_POINT_SCALE)
            fy_int = round(Int64, gy * FIXED_POINT_SCALE)
            fz_int = round(Int64, gz * FIXED_POINT_SCALE)
            F_int[1, i] += fx_int
            F_int[2, i] += fy_int
            F_int[3, i] += fz_int
            F_int[1, j] -= fx_int
            F_int[2, j] -= fy_int
            F_int[3, j] -= fz_int
        end
        inv_scale = 1.0 / FIXED_POINT_SCALE
        @inbounds for a in 1:n, m in 1:3
            F[m, a] += Float64(F_int[m, a]) * inv_scale
        end
        return u
    else
        u = 0.0
        @inbounds for k in eachindex(f.i)
            i = f.i[k]; j = f.j[k]
            d = unwrap_displacement(box, q, i, j)
            r2 = norm2(d)
            f.mode !== :none && r2 > rc2 && continue
            r2 < 1e-24 && return Inf
            r4 = r2 * r2
            r6 = r4 * r2
            r12 = r6 * r6
            c6 = f.c6[k]; c12 = f.c12[k]; sc = f.scale[k]
            u0 = c12 / r12 - c6 / r6
            if is_lfs
                r = sqrt(r2)
                rc6 = rc2^3
                rc12 = rc6^2
                uc = c12 / rc12 - c6 / rc6
                duc = -12.0 * c12 / (rc12 * rc) + 6.0 * c6 / (rc6 * rc)
                u += sc * (u0 - uc - duc * (r - rc))
                coeff = sc * (12.0 * c12 / (r12 * r2) - 6.0 * c6 / (r6 * r2) + duc / r)
            else
                u += sc * u0
                coeff = sc * (12.0 * c12 / (r12 * r2) - 6.0 * c6 / (r6 * r2))
            end
            gx = coeff * d.x; gy = coeff * d.y; gz = coeff * d.z
            F[1, i] += gx; F[2, i] += gy; F[3, i] += gz
            F[1, j] -= gx; F[2, j] -= gy; F[3, j] -= gz
        end
        return u
    end
end

Factors.evaluate!(f::NBFIXPairTable, q::AbstractMatrix{Float64}, F::AbstractMatrix{Float64}, ctx) =
    Factors.evaluate!(F, f, q, ctx)

function accumulate_box_derivative!(G::AbstractMatrix{Float64}, f::NBFIXPairTable,
                                    q::AbstractMatrix{Float64}, box::Types.OrthorhombicBox)
    f.mode in (:none, :lfs) || throw(ArgumentError("NBFIXPairTable box derivative refuses mode $(f.mode)"))
    size(G) == (3, 3) || throw(DimensionMismatch("G must be 3x3"))
    size(q, 1) == 3 || throw(ArgumentError("q must be 3 x N"))
    if !isempty(f.i)
        max_idx = max(maximum(f.i), maximum(f.j))
        max_idx <= size(q, 2) || throw(BoundsError(q, (3, max_idx)))
    end
    rc = f.rcut; rc2 = rc * rc
    is_lfs = f.mode === :lfs
    rc6 = is_lfs ? rc2^3 : 0.0
    rc12 = is_lfs ? rc6^2 : 0.0
    L = box.L
    @inbounds for k in eachindex(f.i)
        i = f.i[k]; j = f.j[k]
        d = unwrap_displacement(box, q, i, j)
        r2 = norm2(d)
        f.mode !== :none && r2 > rc2 && continue
        if r2 < 1e-24
            G .= Inf
            return G
        end
        r4 = r2 * r2
        r6 = r4 * r2
        r12 = r6 * r6
        c6 = f.c6[k]; c12 = f.c12[k]; sc = f.scale[k]
        scale = sc * (-12.0 * c12 / (r12 * r2) + 6.0 * c6 / (r6 * r2))
        if is_lfs
            r = sqrt(r2)
            duc = -12.0 * c12 / (rc12 * rc) + 6.0 * c6 / (rc6 * rc)
            scale -= sc * duc / r
        end
        w = (d.x / L[1], d.y / L[2], d.z / L[3])
        _cf_outer_add!(G, scale, d, w)
    end
    return G
end

function accumulate_box_derivative!(G::AbstractMatrix{Float64}, f::NBFIXPairTable,
                                    q::AbstractMatrix{Float64}, box::Types.TriclinicBox)
    f.mode in (:none, :lfs) || throw(ArgumentError("NBFIXPairTable box derivative refuses mode $(f.mode)"))
    size(G) == (3, 3) || throw(DimensionMismatch("G must be 3x3"))
    size(q, 1) == 3 || throw(ArgumentError("q must be 3 x N"))
    if !isempty(f.i)
        max_idx = max(maximum(f.i), maximum(f.j))
        max_idx <= size(q, 2) || throw(BoundsError(q, (3, max_idx)))
    end
    rc = f.rcut; rc2 = rc * rc
    is_lfs = f.mode === :lfs
    rc6 = is_lfs ? rc2^3 : 0.0
    rc12 = is_lfs ? rc6^2 : 0.0
    @inbounds for k in eachindex(f.i)
        i = f.i[k]; j = f.j[k]
        d = unwrap_displacement(box, q, i, j)
        r2 = norm2(d)
        f.mode !== :none && r2 > rc2 && continue
        if r2 < 1e-24
            G .= Inf
            return G
        end
        r4 = r2 * r2
        r6 = r4 * r2
        r12 = r6 * r6
        c6 = f.c6[k]; c12 = f.c12[k]; sc = f.scale[k]
        scale = sc * (-12.0 * c12 / (r12 * r2) + 6.0 * c6 / (r6 * r2))
        if is_lfs
            r = sqrt(r2)
            duc = -12.0 * c12 / (rc12 * rc) + 6.0 * c6 / (rc6 * rc)
            scale -= sc * duc / r
        end
        w = _cf_frac_disp(box, d)
        _cf_outer_add!(G, scale, d, w)
    end
    return G
end

function accumulate_box_derivative!(G::AbstractMatrix{Float64}, f::NBFIXPairTable,
                                    q::AbstractMatrix{Float64}, box::Types.NoBox)
    throw(ArgumentError("no periodic box: box derivative ∂U/∂h is undefined for NoBox"))
end

function accumulate_box_derivative!(G::AbstractMatrix{Float64}, f::NBFIXPairTable,
                                    q::AbstractMatrix{Float64}, ctx::Factors.Context)
    return accumulate_box_derivative!(G, f, q, ctx.box)
end

function accumulate_box_derivative(f::NBFIXPairTable, q::AbstractMatrix{Float64}, box)
    G = zeros(Float64, 3, 3)
    accumulate_box_derivative!(G, f, q, box)
    return G
end

_RS.accumulate_box_derivative!(G::Matrix{Float64}, f::NBFIXPairTable,
                                q::AbstractMatrix{Float64}, ctx::Factors.Context) =
    accumulate_box_derivative!(G, f, q, ctx.box)

end # module ChemistryFactors
