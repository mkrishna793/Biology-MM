"""
`ReactiveNano` — reactive many-body potentials (Tersoff, Stillinger–Weber),
Morse bond factor, universal symmetry-function MLIP hook (`SymmetryFunctionMLIP`,
`MLIPFactor`), and periodic nanostructure builders (graphene, CNTs, nanopores,
diamond lattices).
"""
module ReactiveNano

using LinearAlgebra
using ..Types
using ..Factors
using ..Neighbors

const _HAS_RESOLVED_SYSTEM = isdefined(parentmodule(@__MODULE__), :ResolvedSystem)
if _HAS_RESOLVED_SYSTEM
    import ..ResolvedSystem
end

export TersoffParams, TersoffFactor, tersoff_carbon, tersoff_silicon,
       StillingerWeberParams, StillingerWeberFactor, stillinger_weber_silicon,
       MorseBondFactor,
       SymmetryFunctionMLIP, MLIPFactor,
       build_graphene_sheet, build_cnt, build_nanopore_membrane, build_diamond_lattice

# 1 eV in kJ/mol (CODATA 2018: e * N_A / 1000)
const _EV_TO_KJMOL = 1.602176634e-19 * 6.02214076e23 / 1000.0
const _ANG_TO_NM = 0.1
const _INV_ANG_TO_INV_NM = 10.0

# ---------------------------------------------------------------------------
# Fractional coordinates & box-derivative outer-product helpers
# ---------------------------------------------------------------------------

@inline function _frac_disp(b::Types.OrthorhombicBox, dx::Float64, dy::Float64, dz::Float64)
    L = b.L
    return (dx / L[1], dy / L[2], dz / L[3])
end

@inline function _frac_disp(b::Types.TriclinicBox, dx::Float64, dy::Float64, dz::Float64)
    M = b.Hinv
    return (M[1, 1] * dx + M[1, 2] * dy + M[1, 3] * dz,
            M[2, 1] * dx + M[2, 2] * dy + M[2, 3] * dz,
            M[3, 1] * dx + M[3, 2] * dy + M[3, 3] * dz)
end

@inline function _frac_disp(::Types.NoBox, ::Float64, ::Float64, ::Float64)
    throw(Types.UnsupportedGeometry("no periodic box: box derivative requires lattice vectors"))
end

@inline function _outer_add!(G::Matrix{Float64}, c::Float64,
                             dx::Float64, dy::Float64, dz::Float64,
                             w::NTuple{3,Float64})
    gx = c * dx; gy = c * dy; gz = c * dz
    @inbounds begin
        G[1, 1] += gx * w[1]; G[1, 2] += gx * w[2]; G[1, 3] += gx * w[3]
        G[2, 1] += gy * w[1]; G[2, 2] += gy * w[2]; G[2, 3] += gy * w[3]
        G[3, 1] += gz * w[1]; G[3, 2] += gz * w[2]; G[3, 3] += gz * w[3]
    end
    return G
end

# ---------------------------------------------------------------------------
# Box-aware neighbor adjacency builder (O(N) via Neighbors cell lists when
# cell widths allow, with exact O(N^2) MIC fallback for NoBox or small cells)
# ---------------------------------------------------------------------------

struct _NbrEntry
    j::Int
    dx::Float64
    dy::Float64
    dz::Float64
    r::Float64
end

function _can_use_cell_list(box::Types.OrthorhombicBox, rc::Float64)::Bool
    return 2.0 * rc < min(box.L[1], box.L[2], box.L[3])
end
function _can_use_cell_list(box::Types.TriclinicBox, rc::Float64)::Bool
    return 2.0 * rc < min(box.widths[1], box.widths[2], box.widths[3])
end
_can_use_cell_list(::Types.AbstractBox, ::Float64)::Bool = false

function _build_directed_neighbors(q::AbstractMatrix{Float64}, box::Types.AbstractBox,
                                   rc::Float64)::Union{Nothing,Vector{Vector{_NbrEntry}}}
    N = size(q, 2)
    nbrs = [_NbrEntry[] for _ in 1:N]
    r2cut = rc * rc

    if box isa Types.OrthorhombicBox && _can_use_cell_list(box, rc)
        cl = Neighbors.make_cell_list(q, box, rc)
        nc = cl.nc
        L = cl.L
        @inbounds for j in 1:N
            c1 = Neighbors._bucket(q[1, j], L[1], nc[1])
            c2 = Neighbors._bucket(q[2, j], L[2], nc[2])
            c3 = Neighbors._bucket(q[3, j], L[3], nc[3])
            for id in Neighbors._neighbor_cells(c1, c2, c3, nc)
                for k in cl.members[id]
                    k <= j && continue
                    d = Types.unwrap_displacement(box, q, k, j) # q_k - q_j
                    r2 = d.x * d.x + d.y * d.y + d.z * d.z
                    r2 < 1e-24 && return nothing
                    if r2 < r2cut
                        r = sqrt(r2)
                        push!(nbrs[j], _NbrEntry(k, d.x, d.y, d.z, r))
                        push!(nbrs[k], _NbrEntry(j, -d.x, -d.y, -d.z, r))
                    end
                end
            end
        end
    elseif box isa Types.TriclinicBox && _can_use_cell_list(box, rc)
        cl = Neighbors.make_triclinic_cell_list(q, box, rc)
        nc = cl.nc
        @inbounds for j in 1:N
            cid = cl.cell_of[j]
            c1 = mod1(cid, nc[1])
            rem1 = div(cid - c1, nc[1])
            c2 = mod1(rem1 + 1, nc[2])
            c3 = div(rem1 + 1 - c2, nc[2]) + 1
            for id in Neighbors._neighbor_cells(c1, c2, c3, nc)
                for k in cl.members[id]
                    k <= j && continue
                    d = Types.unwrap_displacement(box, q, k, j)
                    r2 = d.x * d.x + d.y * d.y + d.z * d.z
                    r2 < 1e-24 && return nothing
                    if r2 < r2cut
                        r = sqrt(r2)
                        push!(nbrs[j], _NbrEntry(k, d.x, d.y, d.z, r))
                        push!(nbrs[k], _NbrEntry(j, -d.x, -d.y, -d.z, r))
                    end
                end
            end
        end
    else
        @inbounds for j in 1:(N - 1)
            for k in (j + 1):N
                d = Types.unwrap_displacement(box, q, k, j)
                r2 = d.x * d.x + d.y * d.y + d.z * d.z
                r2 < 1e-24 && return nothing
                if r2 < r2cut
                    r = sqrt(r2)
                    push!(nbrs[j], _NbrEntry(k, d.x, d.y, d.z, r))
                    push!(nbrs[k], _NbrEntry(j, -d.x, -d.y, -d.z, r))
                end
            end
        end
    end
    return nbrs
end

# ---------------------------------------------------------------------------
# Tersoff bond-order potential
# ---------------------------------------------------------------------------

"""
    TersoffParams(A, B, lambda1, lambda2, beta, n, c, d, h, lambda3, R, D)

Parameters for the Abell–Tersoff reactive bond-order potential.
"""
struct TersoffParams
    A::Float64
    B::Float64
    lambda1::Float64
    lambda2::Float64
    beta::Float64
    n::Float64
    c::Float64
    d::Float64
    h::Float64
    lambda3::Float64
    R::Float64
    D::Float64
    function TersoffParams(A::Real, B::Real, lambda1::Real, lambda2::Real,
                           beta::Real, n::Real, c::Real, d::Real, h::Real,
                           lambda3::Real, R::Real, D::Real)
        vals = Float64.((A, B, lambda1, lambda2, beta, n, c, d, h, lambda3, R, D))
        all(isfinite, vals) || throw(ArgumentError("all TersoffParams must be finite"))
        (vals[1] > 0 && vals[2] > 0 && vals[3] > 0 && vals[4] > 0 &&
         vals[5] > 0 && vals[6] > 0 && vals[8] > 0 && vals[12] > 0 &&
         vals[11] > vals[12]) ||
            throw(ArgumentError("invalid TersoffParams domain (expected positive A, B, lambdas, beta, n, d, D and R > D)"))
        return new(vals...)
    end
end

"""Tersoff (1989) Carbon parameters. `units` may be `:md` (nm, kJ/mol) or `:ev` (Å, eV)."""
function tersoff_carbon(; units::Symbol = :md)::TersoffParams
    if units === :md
        return TersoffParams(
            1393.6 * _EV_TO_KJMOL,
            346.74 * _EV_TO_KJMOL,
            3.4879 * _INV_ANG_TO_INV_NM,
            2.2119 * _INV_ANG_TO_INV_NM,
            1.5724e-7,
            0.72751,
            3.8049e4,
            4.3484,
            -0.57058,
            0.0,
            1.95 * _ANG_TO_NM,
            0.15 * _ANG_TO_NM,
        )
    elseif units === :ev
        return TersoffParams(
            1393.6, 346.74, 3.4879, 2.2119,
            1.5724e-7, 0.72751, 3.8049e4, 4.3484, -0.57058,
            0.0, 1.95, 0.15,
        )
    end
    throw(ArgumentError("unsupported units :$units; expected :md or :ev"))
end

"""Tersoff (1989) Silicon parameters. `units` may be `:md` (nm, kJ/mol) or `:ev` (Å, eV)."""
function tersoff_silicon(; units::Symbol = :md)::TersoffParams
    if units === :md
        return TersoffParams(
            1830.8 * _EV_TO_KJMOL,
            471.18 * _EV_TO_KJMOL,
            2.4799 * _INV_ANG_TO_INV_NM,
            1.7322 * _INV_ANG_TO_INV_NM,
            1.1000e-6,
            0.78734,
            1.0039e5,
            16.217,
            -0.59825,
            0.0,
            2.85 * _ANG_TO_NM,
            0.15 * _ANG_TO_NM,
        )
    elseif units === :ev
        return TersoffParams(
            1830.8, 471.18, 2.4799, 1.7322,
            1.1000e-6, 0.78734, 1.0039e5, 16.217, -0.59825,
            0.0, 2.85, 0.15,
        )
    end
    throw(ArgumentError("unsupported units :$units; expected :md or :ev"))
end

"""
    TersoffFactor(n_atoms::Int, params::TersoffParams)

Reactive Tersoff bond-order factor with exact analytical forces and 9-entry
box derivatives.
"""
struct TersoffFactor <: Factors.AbstractFactor
    n_atoms::Int
    params::TersoffParams
    function TersoffFactor(n_atoms::Int, params::TersoffParams)
        n_atoms >= 1 || throw(ArgumentError("n_atoms must be >= 1"))
        return new(n_atoms, params)
    end
end
TersoffFactor(params::TersoffParams, n_atoms::Int) = TersoffFactor(n_atoms, params)

Factors.factor_support_size(f::TersoffFactor) = f.n_atoms
Factors.read_support(f::TersoffFactor) = Tuple(1:f.n_atoms)
Factors.is_local(::TersoffFactor) = false
Factors.force_consistent(::TersoffFactor) = true

@inline function _tersoff_fc(r::Float64, R::Float64, D::Float64)::Tuple{Float64,Float64}
    rmin = R - D
    rmax = R + D
    if r <= rmin
        return (1.0, 0.0)
    elseif r >= rmax
        return (0.0, 0.0)
    else
        arg = 0.5 * pi * (r - R) / D
        fc = 0.5 - 0.5 * sin(arg)
        dfc = -(0.25 * pi / D) * cos(arg)
        return (fc, dfc)
    end
end

function _tersoff_eval!(F::Union{Nothing,AbstractMatrix{Float64}},
                        G::Union{Nothing,Matrix{Float64}},
                        f::TersoffFactor,
                        q::AbstractMatrix{Float64},
                        box::Types.AbstractBox)::Float64
    p = f.params
    rc = p.R + p.D
    nbrs = _build_directed_neighbors(q, box, rc)
    if nbrs === nothing
        G !== nothing && (G .= Inf)
        return Inf
    end

    N = size(q, 2)
    c2 = p.c * p.c
    d2 = p.d * p.d
    c2_d2 = c2 / d2
    h = p.h
    l3_3 = p.lambda3^3
    has_l3 = p.lambda3 != 0.0
    U_tot = 0.0

    @inbounds for i in 1:N
        ni = length(nbrs[i])
        ni == 0 && continue
        for a in 1:ni
            eij = nbrs[i][a]
            j = eij.j
            rij = eij.r
            fc_ij, dfc_ij = _tersoff_fc(rij, p.R, p.D)
            (fc_ij <= 0.0 && dfc_ij == 0.0) && continue

            inv_rij = 1.0 / rij
            ux_ij = eij.dx * inv_rij
            uy_ij = eij.dy * inv_rij
            uz_ij = eij.dz * inv_rij

            # Compute bond-order coordination sum zeta_ij
            zeta = 0.0
            for b in 1:ni
                b == a && continue
                eik = nbrs[i][b]
                rik = eik.r
                fc_ik, _ = _tersoff_fc(rik, p.R, p.D)
                fc_ik <= 0.0 && continue
                inv_rik = 1.0 / rik
                ct = clamp((eij.dx * eik.dx + eij.dy * eik.dy + eij.dz * eik.dz) * (inv_rij * inv_rik), -1.0, 1.0)
                dh = h - ct
                g = 1.0 + c2_d2 - c2 / (d2 + dh * dh)
                om = has_l3 ? exp(l3_3 * (rij - rik)^3) : 1.0
                zeta += fc_ik * g * om
            end

            # Critical numerical guard when zeta <= 0 (e.g., isolated dimer)
            b_ij = 1.0
            db_dzeta = 0.0
            if zeta > 0.0
                tn = (p.beta * zeta)^p.n
                S = 1.0 + tn
                b_ij = S^(-0.5 / p.n)
                db_dzeta = -0.5 * b_ij * tn / (S * zeta)
            end

            fR = p.A * exp(-p.lambda1 * rij)
            dfR = -p.lambda1 * fR
            fA = -p.B * exp(-p.lambda2 * rij)
            dfA = -p.lambda2 * fA

            U_tot += 0.5 * fc_ij * (fR + b_ij * fA)

            if F !== nothing || G !== nothing
                dE_dr = 0.5 * (dfc_ij * (fR + b_ij * fA) + fc_ij * (dfR + b_ij * dfA))
                s_pair = dE_dr * inv_rij
                if F !== nothing
                    fx = s_pair * eij.dx; fy = s_pair * eij.dy; fz = s_pair * eij.dz
                    F[1, j] -= fx; F[2, j] -= fy; F[3, j] -= fz
                    F[1, i] += fx; F[2, i] += fy; F[3, i] += fz
                end
                w_ij = G !== nothing ? _frac_disp(box, eij.dx, eij.dy, eij.dz) : (0.0, 0.0, 0.0)
                if G !== nothing
                    _outer_add!(G, s_pair, eij.dx, eij.dy, eij.dz, w_ij)
                end

                if db_dzeta != 0.0 && fc_ij > 0.0
                    pref = 0.5 * fc_ij * fA * db_dzeta
                    for b in 1:ni
                        b == a && continue
                        eik = nbrs[i][b]
                        k = eik.j
                        rik = eik.r
                        fc_ik, dfc_ik = _tersoff_fc(rik, p.R, p.D)
                        (fc_ik <= 0.0 && dfc_ik == 0.0) && continue

                        inv_rik = 1.0 / rik
                        ux_ik = eik.dx * inv_rik
                        uy_ik = eik.dy * inv_rik
                        uz_ik = eik.dz * inv_rik
                        ct = clamp(ux_ij * ux_ik + uy_ij * uy_ik + uz_ij * uz_ik, -1.0, 1.0)
                        dh = h - ct
                        denom = d2 + dh * dh
                        g = 1.0 + c2_d2 - c2 / denom
                        dg = -2.0 * c2 * dh / (denom * denom)

                        dr_ijk = rij - rik
                        om = has_l3 ? exp(l3_3 * dr_ijk^3) : 1.0
                        dom = has_l3 ? 3.0 * l3_3 * (dr_ijk * dr_ijk) * om : 0.0

                        c_ij_rad = pref * fc_ik * g * dom
                        c_ij_ang = pref * fc_ik * om * dg * inv_rij
                        gx_ij = c_ij_rad * ux_ij + c_ij_ang * (ux_ik - ct * ux_ij)
                        gy_ij = c_ij_rad * uy_ij + c_ij_ang * (uy_ik - ct * uy_ij)
                        gz_ij = c_ij_rad * uz_ij + c_ij_ang * (uz_ik - ct * uz_ij)

                        c_ik_rad = pref * (dfc_ik * om - fc_ik * dom) * g
                        c_ik_ang = pref * fc_ik * om * dg * inv_rik
                        gx_ik = c_ik_rad * ux_ik + c_ik_ang * (ux_ij - ct * ux_ik)
                        gy_ik = c_ik_rad * uy_ik + c_ik_ang * (uy_ij - ct * uy_ik)
                        gz_ik = c_ik_rad * uz_ik + c_ik_ang * (uz_ij - ct * uz_ik)

                        if F !== nothing
                            F[1, j] -= gx_ij; F[2, j] -= gy_ij; F[3, j] -= gz_ij
                            F[1, k] -= gx_ik; F[2, k] -= gy_ik; F[3, k] -= gz_ik
                            F[1, i] += gx_ij + gx_ik; F[2, i] += gy_ij + gy_ik; F[3, i] += gz_ij + gz_ik
                        end
                        if G !== nothing
                            _outer_add!(G, 1.0, gx_ij, gy_ij, gz_ij, w_ij)
                            w_ik = _frac_disp(box, eik.dx, eik.dy, eik.dz)
                            _outer_add!(G, 1.0, gx_ik, gy_ik, gz_ik, w_ik)
                        end
                    end
                end
            end
        end
    end
    return U_tot
end

function Factors.evaluate!(F::AbstractMatrix{Float64}, f::TersoffFactor,
                           q::AbstractMatrix{Float64}, ctx::Factors.Context)
    return _tersoff_eval!(F, nothing, f, q, ctx.box)
end

function accumulate_box_derivative!(G::Matrix{Float64}, f::TersoffFactor,
                                    q::AbstractMatrix{Float64}, ctx::Factors.Context)
    ctx.box isa Types.NoBox && throw(Types.UnsupportedGeometry("box derivative requires a periodic box"))
    _tersoff_eval!(nothing, G, f, q, ctx.box)
    return G
end

# ---------------------------------------------------------------------------
# Stillinger-Weber 2+3-body potential
# ---------------------------------------------------------------------------

"""
    StillingerWeberParams(epsilon, sigma, a, lambda, gamma, costheta0, A, B, p, q)

Parameters for the Stillinger–Weber 2-body + 3-body potential.
"""
struct StillingerWeberParams
    epsilon::Float64
    sigma::Float64
    a::Float64
    lambda::Float64
    gamma::Float64
    costheta0::Float64
    A::Float64
    B::Float64
    p::Int
    q::Int
    function StillingerWeberParams(epsilon::Real, sigma::Real, a::Real, lambda::Real,
                                   gamma::Real, costheta0::Real, A::Real, B::Real,
                                   p::Integer, q::Integer)
        vals = Float64.((epsilon, sigma, a, lambda, gamma, costheta0, A, B))
        all(isfinite, vals) || throw(ArgumentError("all StillingerWeberParams must be finite"))
        (vals[1] > 0 && vals[2] > 0 && vals[3] > 0 && vals[4] >= 0 && vals[5] > 0) ||
            throw(ArgumentError("invalid StillingerWeberParams domain"))
        return new(vals..., Int(p), Int(q))
    end
end

"""Standard Stillinger–Weber (1985) Silicon parameters (`:md` or `:ev`)."""
function stillinger_weber_silicon(; units::Symbol = :md)::StillingerWeberParams
    if units === :md
        return StillingerWeberParams(
            2.1683 * _EV_TO_KJMOL,
            2.0951 * _ANG_TO_NM,
            1.80,
            21.0,
            1.20,
            -1.0 / 3.0,
            7.049556277,
            0.6022245584,
            4,
            0,
        )
    elseif units === :ev
        return StillingerWeberParams(
            2.1683, 2.0951, 1.80, 21.0, 1.20, -1.0 / 3.0,
            7.049556277, 0.6022245584, 4, 0,
        )
    end
    throw(ArgumentError("unsupported units :$units; expected :md or :ev"))
end

"""
    StillingerWeberFactor(n_atoms::Int, params::StillingerWeberParams)

Stillinger–Weber 2-body + 3-body factor with exact analytical forces and
9-entry box derivatives.
"""
struct StillingerWeberFactor <: Factors.AbstractFactor
    n_atoms::Int
    params::StillingerWeberParams
    function StillingerWeberFactor(n_atoms::Int, params::StillingerWeberParams)
        n_atoms >= 1 || throw(ArgumentError("n_atoms must be >= 1"))
        return new(n_atoms, params)
    end
end
StillingerWeberFactor(params::StillingerWeberParams, n_atoms::Int) = StillingerWeberFactor(n_atoms, params)

Factors.factor_support_size(f::StillingerWeberFactor) = f.n_atoms
Factors.read_support(f::StillingerWeberFactor) = Tuple(1:f.n_atoms)
Factors.is_local(::StillingerWeberFactor) = false
Factors.force_consistent(::StillingerWeberFactor) = true

function _sw_eval!(F::Union{Nothing,AbstractMatrix{Float64}},
                   G::Union{Nothing,Matrix{Float64}},
                   f::StillingerWeberFactor,
                   q::AbstractMatrix{Float64},
                   box::Types.AbstractBox)::Float64
    pr = f.params
    rc = pr.a * pr.sigma
    nbrs = _build_directed_neighbors(q, box, rc)
    if nbrs === nothing
        G !== nothing && (G .= Inf)
        return Inf
    end

    N = size(q, 2)
    eps = pr.epsilon
    sig = pr.sigma
    inv_sig = 1.0 / sig
    a_cut = pr.a
    lam = pr.lambda
    gam = pr.gamma
    ct0 = pr.costheta0
    A = pr.A
    B = pr.B
    pp = pr.p
    qq = pr.q
    U_tot = 0.0

    @inbounds for i in 1:N
        ni = length(nbrs[i])
        ni == 0 && continue
        for a in 1:ni
            eij = nbrs[i][a]
            j = eij.j
            rij = eij.r
            s_ij = rij * inv_sig
            u_ij = s_ij - a_cut
            u_ij >= -1e-12 && continue

            # 2-body term counted once per pair (j > i)
            if j > i
                e2 = exp(1.0 / u_ij)
                sp = pp == 0 ? 1.0 : s_ij^(-pp)
                sq = qq == 0 ? 1.0 : s_ij^(-qq)
                bracket = B * sp - sq
                U_tot += eps * A * bracket * e2
                if F !== nothing || G !== nothing
                    dsp = pp == 0 ? 0.0 : -pp * sp / s_ij
                    dsq = qq == 0 ? 0.0 : -qq * sq / s_ij
                    d_bracket = B * dsp - dsq
                    dphi2_dr = (eps * A * inv_sig) * e2 * (d_bracket - bracket / (u_ij * u_ij))
                    s_pair = dphi2_dr / rij
                    if F !== nothing
                        fx = s_pair * eij.dx; fy = s_pair * eij.dy; fz = s_pair * eij.dz
                        F[1, j] -= fx; F[2, j] -= fy; F[3, j] -= fz
                        F[1, i] += fx; F[2, i] += fy; F[3, i] += fz
                    end
                    if G !== nothing
                        w_ij = _frac_disp(box, eij.dx, eij.dy, eij.dz)
                        _outer_add!(G, s_pair, eij.dx, eij.dy, eij.dz, w_ij)
                    end
                end
            end

            # 3-body terms centered at i for unordered pairs {j, k}
            a == ni && continue
            inv_rij = 1.0 / rij
            ux_ij = eij.dx * inv_rij
            uy_ij = eij.dy * inv_rij
            uz_ij = eij.dz * inv_rij
            w_ij = G !== nothing ? _frac_disp(box, eij.dx, eij.dy, eij.dz) : (0.0, 0.0, 0.0)

            for b in (a + 1):ni
                eik = nbrs[i][b]
                k = eik.j
                rik = eik.r
                u_ik = rik * inv_sig - a_cut
                u_ik >= -1e-12 && continue

                Erad = eps * lam * exp(gam / u_ij + gam / u_ik)
                inv_rik = 1.0 / rik
                ux_ik = eik.dx * inv_rik
                uy_ik = eik.dy * inv_rik
                uz_ik = eik.dz * inv_rik
                ct = clamp(ux_ij * ux_ik + uy_ij * uy_ik + uz_ij * uz_ik, -1.0, 1.0)
                dc = ct - ct0
                U_tot += Erad * dc * dc

                if F !== nothing || G !== nothing
                    c_ang = 2.0 * dc * Erad
                    c_rad_ij = -gam * dc * dc * Erad * inv_sig / (u_ij * u_ij)
                    c_rad_ik = -gam * dc * dc * Erad * inv_sig / (u_ik * u_ik)

                    c_ang_ij = c_ang * inv_rij
                    gx_ij = c_rad_ij * ux_ij + c_ang_ij * (ux_ik - ct * ux_ij)
                    gy_ij = c_rad_ij * uy_ij + c_ang_ij * (uy_ik - ct * uy_ij)
                    gz_ij = c_rad_ij * uz_ij + c_ang_ij * (uz_ik - ct * uz_ij)

                    c_ang_ik = c_ang * inv_rik
                    gx_ik = c_rad_ik * ux_ik + c_ang_ik * (ux_ij - ct * ux_ik)
                    gy_ik = c_rad_ik * uy_ik + c_ang_ik * (uy_ij - ct * uy_ik)
                    gz_ik = c_rad_ik * uz_ik + c_ang_ik * (uz_ij - ct * uz_ik)

                    if F !== nothing
                        F[1, j] -= gx_ij; F[2, j] -= gy_ij; F[3, j] -= gz_ij
                        F[1, k] -= gx_ik; F[2, k] -= gy_ik; F[3, k] -= gz_ik
                        F[1, i] += gx_ij + gx_ik; F[2, i] += gy_ij + gy_ik; F[3, i] += gz_ij + gz_ik
                    end
                    if G !== nothing
                        _outer_add!(G, 1.0, gx_ij, gy_ij, gz_ij, w_ij)
                        w_ik = _frac_disp(box, eik.dx, eik.dy, eik.dz)
                        _outer_add!(G, 1.0, gx_ik, gy_ik, gz_ik, w_ik)
                    end
                end
            end
        end
    end
    return U_tot
end

function Factors.evaluate!(F::AbstractMatrix{Float64}, f::StillingerWeberFactor,
                           q::AbstractMatrix{Float64}, ctx::Factors.Context)
    return _sw_eval!(F, nothing, f, q, ctx.box)
end

function accumulate_box_derivative!(G::Matrix{Float64}, f::StillingerWeberFactor,
                                    q::AbstractMatrix{Float64}, ctx::Factors.Context)
    ctx.box isa Types.NoBox && throw(Types.UnsupportedGeometry("box derivative requires a periodic box"))
    _sw_eval!(nothing, G, f, q, ctx.box)
    return G
end

# ---------------------------------------------------------------------------
# Morse Bond Factor
# ---------------------------------------------------------------------------

"""
    MorseBondFactor(i, j, De, a, r0)

Reactive dissociable Morse bond factor `U = De * (1 - exp(-a * (r - r0)))^2`.
"""
struct MorseBondFactor <: Factors.AbstractFactor
    i::Int
    j::Int
    De::Float64
    a::Float64
    r0::Float64
    function MorseBondFactor(i::Int, j::Int, De::Real, a::Real, r0::Real)
        (i >= 1 && j >= 1 && i != j) || throw(ArgumentError("invalid bond indices ($i, $j)"))
        (De_f, a_f, r0_f) = Float64.((De, a, r0))
        (isfinite(De_f) && De_f >= 0 && isfinite(a_f) && a_f > 0 && isfinite(r0_f) && r0_f > 0) ||
            throw(ArgumentError("invalid MorseBondFactor parameters"))
        return new(i, j, De_f, a_f, r0_f)
    end
end

Factors.factor_support_size(::MorseBondFactor) = 2
Factors.read_support(f::MorseBondFactor) = (f.i, f.j)
Factors.is_local(::MorseBondFactor) = true
Factors.force_consistent(::MorseBondFactor) = true

function Factors.evaluate!(F::AbstractMatrix{Float64}, f::MorseBondFactor,
                           q::AbstractMatrix{Float64}, ctx::Factors.Context)
    d = Types.unwrap_displacement(ctx.box, q, f.i, f.j)
    r2 = Types.norm2(d)
    r2 < 1e-24 && return Inf
    r = sqrt(r2)
    e = exp(-f.a * (r - f.r0))
    om = 1.0 - e
    u = f.De * om * om
    dudr = 2.0 * f.De * f.a * om * e
    c = -dudr / r
    @inbounds begin
        F[1, f.i] += c * d.x; F[2, f.i] += c * d.y; F[3, f.i] += c * d.z
        F[1, f.j] -= c * d.x; F[2, f.j] -= c * d.y; F[3, f.j] -= c * d.z
    end
    return u
end

function accumulate_box_derivative!(G::Matrix{Float64}, f::MorseBondFactor,
                                    q::AbstractMatrix{Float64}, ctx::Factors.Context)
    ctx.box isa Types.NoBox && throw(Types.UnsupportedGeometry("box derivative requires a periodic box"))
    d = Types.unwrap_displacement(ctx.box, q, f.i, f.j)
    r2 = Types.norm2(d)
    if r2 < 1e-24
        G .= Inf
        return G
    end
    r = sqrt(r2)
    e = exp(-f.a * (r - f.r0))
    om = 1.0 - e
    dudr = 2.0 * f.De * f.a * om * e
    w = _frac_disp(ctx.box, d.x, d.y, d.z)
    _outer_add!(G, dudr / r, d.x, d.y, d.z, w)
    return G
end

# ---------------------------------------------------------------------------
# Symmetry-Function MLIP & Universal MLIPFactor Hook
# ---------------------------------------------------------------------------

"""
    SymmetryFunctionMLIP(rc; g2_eta=..., g2_rs=..., g4_eta=..., g4_zeta=..., g4_lambda=..., weights=..., bias=0.0)

Smooth Behler–Parrinello `G² + G⁴` linear descriptor potential with cosine cutoff
`f_c(r) = 0.5 * (cos(pi * r / rc) + 1)` and exact analytical derivatives wrt `(q, h)`.
Callable as `model(q, box, species) -> (U, F, G_or_nothing)`.
"""
struct SymmetryFunctionMLIP
    rc::Float64
    g2_eta::Vector{Float64}
    g2_rs::Vector{Float64}
    g4_eta::Vector{Float64}
    g4_zeta::Vector{Float64}
    g4_lambda::Vector{Float64}
    weights::Vector{Float64}
    bias::Float64
    function SymmetryFunctionMLIP(rc::Real;
                                  g2_eta::AbstractVector{<:Real} = [4.0, 12.0],
                                  g2_rs::AbstractVector{<:Real} = [0.15, 0.25],
                                  g4_eta::AbstractVector{<:Real} = [2.0, 2.0],
                                  g4_zeta::AbstractVector{<:Real} = [1.0, 2.0],
                                  g4_lambda::AbstractVector{<:Real} = [1.0, -1.0],
                                  weights::Union{Nothing,AbstractVector{<:Real}} = nothing,
                                  bias::Real = 0.0)
        rc_f = Float64(rc)
        (isfinite(rc_f) && rc_f > 0.0) || throw(ArgumentError("rc must be finite and > 0"))
        length(g2_eta) == length(g2_rs) || throw(ArgumentError("g2_eta and g2_rs must have equal length"))
        length(g4_eta) == length(g4_zeta) == length(g4_lambda) ||
            throw(ArgumentError("g4_eta, g4_zeta, g4_lambda must have equal length"))
        n_desc = length(g2_eta) + length(g4_eta)
        w = weights === nothing ? fill(1.0, n_desc) : Float64.(collect(weights))
        length(w) == n_desc || throw(ArgumentError("weights length $(length(w)) != descriptor count $n_desc"))
        all(z -> z >= 1.0, g4_zeta) || throw(ArgumentError("g4_zeta entries must be >= 1.0"))
        all(l -> l == 1.0 || l == -1.0, g4_lambda) || throw(ArgumentError("g4_lambda entries must be +/-1.0"))
        return new(rc_f, Float64.(collect(g2_eta)), Float64.(collect(g2_rs)),
                   Float64.(collect(g4_eta)), Float64.(collect(g4_zeta)), Float64.(collect(g4_lambda)),
                   w, Float64(bias))
    end
end

@inline function _bp_fc(r::Float64, rc::Float64)::Tuple{Float64,Float64}
    r >= rc && return (0.0, 0.0)
    arg = pi * r / rc
    fc = 0.5 * (cos(arg) + 1.0)
    dfc = -(0.5 * pi / rc) * sin(arg)
    return (fc, dfc)
end

function (m::SymmetryFunctionMLIP)(q::AbstractMatrix{Float64},
                                   box::Types.AbstractBox,
                                   species::AbstractVector{Int})
    N = size(q, 2)
    F = zeros(Float64, 3, N)
    has_box = !(box isa Types.NoBox)
    G = has_box ? zeros(Float64, 3, 3) : nothing

    nbrs = _build_directed_neighbors(q, box, m.rc)
    if nbrs === nothing
        G !== nothing && (G .= Inf)
        return (Inf, F, G)
    end

    rc = m.rc
    n_g2 = length(m.g2_eta)
    n_g4 = length(m.g4_eta)
    U_tot = N * m.bias

    @inbounds for i in 1:N
        ni = length(nbrs[i])
        ni == 0 && continue

        # Radial G2 descriptors
        for a in 1:ni
            eij = nbrs[i][a]
            j = eij.j
            rij = eij.r
            fc_ij, dfc_ij = _bp_fc(rij, rc)
            fc_ij == 0.0 && continue
            inv_rij = 1.0 / rij
            w_ij = has_box ? _frac_disp(box, eij.dx, eij.dy, eij.dz) : (0.0, 0.0, 0.0)

            dU_drij = 0.0
            for m2 in 1:n_g2
                w = m.weights[m2]
                w == 0.0 && continue
                eta = m.g2_eta[m2]
                dr = rij - m.g2_rs[m2]
                eg = exp(-eta * dr * dr)
                U_tot += w * eg * fc_ij
                dU_drij += w * eg * (dfc_ij - 2.0 * eta * dr * fc_ij)
            end
            if dU_drij != 0.0
                s_pair = dU_drij * inv_rij
                fx = s_pair * eij.dx; fy = s_pair * eij.dy; fz = s_pair * eij.dz
                F[1, j] -= fx; F[2, j] -= fy; F[3, j] -= fz
                F[1, i] += fx; F[2, i] += fy; F[3, i] += fz
                if has_box
                    _outer_add!(G, s_pair, eij.dx, eij.dy, eij.dz, w_ij)
                end
            end
        end

        # Angular G4 descriptors
        (n_g4 == 0 || ni < 2) && continue
        for a in 1:(ni - 1)
            eij = nbrs[i][a]
            j = eij.j
            rij = eij.r
            fc_ij, dfc_ij = _bp_fc(rij, rc)
            fc_ij == 0.0 && continue
            inv_rij = 1.0 / rij
            ux_ij = eij.dx * inv_rij; uy_ij = eij.dy * inv_rij; uz_ij = eij.dz * inv_rij
            w_ij = has_box ? _frac_disp(box, eij.dx, eij.dy, eij.dz) : (0.0, 0.0, 0.0)

            for b in (a + 1):ni
                eik = nbrs[i][b]
                k = eik.j
                rik = eik.r
                fc_ik, dfc_ik = _bp_fc(rik, rc)
                fc_ik == 0.0 && continue

                djk = Types.unwrap_displacement(box, q, k, j)
                rjk2 = Types.norm2(djk)
                rjk2 < 1e-24 && continue
                rjk = sqrt(rjk2)
                fc_jk, dfc_jk = _bp_fc(rjk, rc)
                fc_jk == 0.0 && continue

                inv_rik = 1.0 / rik
                ux_ik = eik.dx * inv_rik; uy_ik = eik.dy * inv_rik; uz_ik = eik.dz * inv_rik
                inv_rjk = 1.0 / rjk
                ux_jk = djk.x * inv_rjk; uy_jk = djk.y * inv_rjk; uz_jk = djk.z * inv_rjk

                ct = clamp(ux_ij * ux_ik + uy_ij * uy_ik + uz_ij * uz_ik, -1.0, 1.0)
                rsum2 = rij * rij + rik * rik + rjk * rjk
                C = fc_ij * fc_ik * fc_jk

                w_ik = has_box ? _frac_disp(box, eik.dx, eik.dy, eik.dz) : (0.0, 0.0, 0.0)
                w_jk = has_box ? _frac_disp(box, djk.x, djk.y, djk.z) : (0.0, 0.0, 0.0)

                for m4 in 1:n_g4
                    w = m.weights[n_g2 + m4]
                    w == 0.0 && continue
                    eta = m.g4_eta[m4]
                    zeta = m.g4_zeta[m4]
                    lam = m.g4_lambda[m4]

                    base = 1.0 + lam * ct
                    base <= 0.0 && continue
                    pref2 = 2.0^(1.0 - zeta)
                    A_ang = pref2 * base^zeta
                    dA_dct = zeta == 1.0 ? pref2 * lam : pref2 * zeta * lam * base^(zeta - 1.0)
                    E_rad = exp(-eta * rsum2)

                    U_tot += w * A_ang * E_rad * C

                    dEC_drij = E_rad * fc_ik * fc_jk * (dfc_ij - 2.0 * eta * rij * fc_ij)
                    dEC_drik = E_rad * fc_ij * fc_jk * (dfc_ik - 2.0 * eta * rik * fc_ik)
                    dEC_drjk = E_rad * fc_ij * fc_ik * (dfc_jk - 2.0 * eta * rjk * fc_jk)

                    c_ang_ij = w * E_rad * C * dA_dct * inv_rij
                    c_rad_ij = w * A_ang * dEC_drij
                    gx_ij = c_rad_ij * ux_ij + c_ang_ij * (ux_ik - ct * ux_ij)
                    gy_ij = c_rad_ij * uy_ij + c_ang_ij * (uy_ik - ct * uy_ij)
                    gz_ij = c_rad_ij * uz_ij + c_ang_ij * (uz_ik - ct * uz_ij)

                    c_ang_ik = w * E_rad * C * dA_dct * inv_rik
                    c_rad_ik = w * A_ang * dEC_drik
                    gx_ik = c_rad_ik * ux_ik + c_ang_ik * (ux_ij - ct * ux_ik)
                    gy_ik = c_rad_ik * uy_ik + c_ang_ik * (uy_ij - ct * uy_ik)
                    gz_ik = c_rad_ik * uz_ik + c_ang_ik * (uz_ij - ct * uz_ik)

                    c_rad_jk = w * A_ang * dEC_drjk
                    gx_jk = c_rad_jk * ux_jk
                    gy_jk = c_rad_jk * uy_jk
                    gz_jk = c_rad_jk * uz_jk

                    F[1, i] += gx_ij + gx_ik; F[2, i] += gy_ij + gy_ik; F[3, i] += gz_ij + gz_ik
                    F[1, j] -= gx_ij - gx_jk; F[2, j] -= gy_ij - gy_jk; F[3, j] -= gz_ij - gz_jk
                    F[1, k] -= gx_ik + gx_jk; F[2, k] -= gy_ik + gy_jk; F[3, k] -= gz_ik + gz_jk

                    if has_box
                        _outer_add!(G, 1.0, gx_ij, gy_ij, gz_ij, w_ij)
                        _outer_add!(G, 1.0, gx_ik, gy_ik, gz_ik, w_ik)
                        _outer_add!(G, 1.0, gx_jk, gy_jk, gz_jk, w_jk)
                    end
                end
            end
        end
    end

    return (U_tot, F, G)
end

"""
    MLIPFactor(n_atoms, rc, model_fn; species=fill(1, n_atoms))

Universal machine-learned interatomic potential factor wrapping `model_fn(q, box, species)`
which returns `(U, F)` or `(U, F, G)`.
"""
struct MLIPFactor{MF} <: Factors.AbstractFactor
    n_atoms::Int
    rc::Float64
    model_fn::MF
    species::Vector{Int}
    function MLIPFactor(n_atoms::Int, rc::Real, model_fn::MF;
                        species::AbstractVector{Int} = fill(1, n_atoms)) where {MF}
        n_atoms >= 1 || throw(ArgumentError("n_atoms must be >= 1"))
        rc_f = Float64(rc)
        (isfinite(rc_f) && rc_f > 0.0) || throw(ArgumentError("rc must be finite and > 0"))
        length(species) == n_atoms || throw(ArgumentError("species length $(length(species)) != n_atoms=$n_atoms"))
        return new{MF}(n_atoms, rc_f, model_fn, Int.(collect(species)))
    end
end

Factors.factor_support_size(f::MLIPFactor) = f.n_atoms
Factors.read_support(f::MLIPFactor) = Tuple(1:f.n_atoms)
Factors.is_local(::MLIPFactor) = false
Factors.force_consistent(::MLIPFactor) = true

function Factors.evaluate!(F::AbstractMatrix{Float64}, f::MLIPFactor,
                           q::AbstractMatrix{Float64}, ctx::Factors.Context)
    res = f.model_fn(q, ctx.box, f.species)
    U = Float64(res[1])
    F_local = res[2]
    size(F_local) == size(F) ||
        throw(DimensionMismatch("MLIPFactor model_fn returned force shape $(size(F_local)), expected $(size(F))"))
    F .+= F_local
    return U
end

function accumulate_box_derivative!(G::Matrix{Float64}, f::MLIPFactor,
                                    q::AbstractMatrix{Float64}, ctx::Factors.Context)
    ctx.box isa Types.NoBox && throw(Types.UnsupportedGeometry("box derivative requires a periodic box"))
    res = f.model_fn(q, ctx.box, f.species)
    if length(res) >= 3 && res[3] !== nothing
        G_local = res[3]
        size(G_local) == (3, 3) || throw(DimensionMismatch("expected 3x3 box derivative G, got $(size(G_local))"))
        G .+= G_local
        return G
    end
    throw(ArgumentError("MLIPFactor model_fn did not provide box derivative G"))
end

if _HAS_RESOLVED_SYSTEM
    ResolvedSystem.box_derivative_support(::TersoffFactor) = nothing
    ResolvedSystem.accumulate_box_derivative!(G::Matrix{Float64}, f::TersoffFactor,
                                              q::AbstractMatrix{Float64}, ctx::Factors.Context) =
        accumulate_box_derivative!(G, f, q, ctx)

    ResolvedSystem.box_derivative_support(::StillingerWeberFactor) = nothing
    ResolvedSystem.accumulate_box_derivative!(G::Matrix{Float64}, f::StillingerWeberFactor,
                                              q::AbstractMatrix{Float64}, ctx::Factors.Context) =
        accumulate_box_derivative!(G, f, q, ctx)

    ResolvedSystem.box_derivative_support(::MorseBondFactor) = nothing
    ResolvedSystem.accumulate_box_derivative!(G::Matrix{Float64}, f::MorseBondFactor,
                                              q::AbstractMatrix{Float64}, ctx::Factors.Context) =
        accumulate_box_derivative!(G, f, q, ctx)

    ResolvedSystem.box_derivative_support(::MLIPFactor) = nothing
    ResolvedSystem.accumulate_box_derivative!(G::Matrix{Float64}, f::MLIPFactor,
                                              q::AbstractMatrix{Float64}, ctx::Factors.Context) =
        accumulate_box_derivative!(G, f, q, ctx)
end

# ---------------------------------------------------------------------------
# Nanostructure Builders
# ---------------------------------------------------------------------------

"""
    build_graphene_sheet(nx, ny; bond_length_nm=0.142, vacuum_z_nm=3.0, element_mass=12.011)

Build a periodic orthorhombic graphene sheet of `nx × ny` 4-atom rectangular cells.
"""
function build_graphene_sheet(nx::Int, ny::Int;
                              bond_length_nm::Float64 = 0.142,
                              vacuum_z_nm::Float64 = 3.0,
                              element_mass::Float64 = 12.011)
    (nx >= 1 && ny >= 1) || throw(ArgumentError("nx, ny must be >= 1"))
    (bond_length_nm > 0 && vacuum_z_nm > 0 && element_mass > 0) ||
        throw(ArgumentError("bond_length_nm, vacuum_z_nm, element_mass must be > 0"))
    b = bond_length_nm
    ax = sqrt(3.0) * b
    ay = 3.0 * b
    basis_x = (0.0, 0.5 * ax, 0.5 * ax, 0.0)
    basis_y = (0.0, 0.5 * b, 1.5 * b, 2.0 * b)

    N = 4 * nx * ny
    q = Matrix{Float64}(undef, 3, N)
    z0 = 0.5 * vacuum_z_nm
    idx = 0
    @inbounds for ix in 0:(nx - 1), iy in 0:(ny - 1), k in 1:4
        idx += 1
        q[1, idx] = ix * ax + basis_x[k]
        q[2, idx] = iy * ay + basis_y[k]
        q[3, idx] = z0
    end
    box = Types.OrthorhombicBox((nx * ax, ny * ay, vacuum_z_nm))
    masses = fill(element_mass, N)
    names = fill(:C, N)
    return (q = q, box = box, masses = masses, names = names)
end

"""
    build_cnt(n, m, n_cells=2; bond_length_nm=0.142, pad_xy_nm=2.0, element_mass=12.011)

Build a periodic `(n, m)` single-walled carbon nanotube aligned along the `z`-axis,
with periodic `Lz = n_cells * |T|` seamlessly stitching C–C bonds across the `z`-boundary.
"""
function build_cnt(n::Int, m::Int, n_cells::Int = 2;
                   bond_length_nm::Float64 = 0.142,
                   pad_xy_nm::Float64 = 2.0,
                   element_mass::Float64 = 12.011)
    (n >= 1 && m >= 0 && m <= n && n_cells >= 1) ||
        throw(ArgumentError("require n >= 1, 0 <= m <= n, n_cells >= 1; got ($n, $m, $n_cells)"))
    (bond_length_nm > 0 && pad_xy_nm > 0 && element_mass > 0) ||
        throw(ArgumentError("bond_length_nm, pad_xy_nm, element_mass must be > 0"))

    b = bond_length_nm
    a0 = sqrt(3.0) * b
    a1x, a1y = a0, 0.0
    a2x, a2y = 0.5 * a0, 0.5 * sqrt(3.0) * a0

    dR = gcd(2 * m + n, 2 * n + m)
    t1 = div(2 * m + n, dR)
    t2 = -div(2 * n + m, dR)

    Chx = n * a1x + m * a2x
    Chy = n * a1y + m * a2y
    Tx = t1 * a1x + t2 * a2x
    Ty = t1 * a1y + t2 * a2y

    C = hypot(Chx, Chy)
    T = hypot(Tx, Ty)
    R = C / (2.0 * pi)
    chiral_angle_rad = atan(sqrt(3.0) * m, 2 * n + m)

    det_M = n * t2 - m * t1
    N_hex = abs(det_M)

    imin = min(0, n, t1, n + t1) - 2
    imax = max(0, n, t1, n + t1) + 2
    jmin = min(0, m, t2, m + t2) - 2
    jmax = max(0, m, t2, m + t2) + 2

    unit_uv = NTuple{2,Float64}[]
    seen = Set{Tuple{Int,Int}}()
    basis_offsets = ((0.0, 0.0), (1.0 / 3.0, 1.0 / 3.0))
    for i in imin:imax, j in jmin:jmax, (bi, bj) in basis_offsets
        ii = i + bi
        jj = j + bj
        u = (t2 * ii - t1 * jj) / det_M
        v = (-m * ii + n * jj) / det_M
        u -= floor(u)
        v -= floor(v)
        u >= 1.0 - 1e-9 && (u = 0.0)
        v >= 1.0 - 1e-9 && (v = 0.0)
        key = (round(Int, u * 1e7), round(Int, v * 1e7))
        if !(key in seen)
            push!(seen, key)
            push!(unit_uv, (u, v))
        end
    end
    length(unit_uv) == 2 * N_hex ||
        error("internal CNT enumeration mismatch: got $(length(unit_uv)), expected $(2 * N_hex)")

    Lx = 2.0 * R + 2.0 * pad_xy_nm
    Ly = 2.0 * R + 2.0 * pad_xy_nm
    Lz = n_cells * T
    cx = 0.5 * Lx
    cy = 0.5 * Ly

    n_unit = length(unit_uv)
    N = n_cells * n_unit
    q = Matrix{Float64}(undef, 3, N)
    idx = 0
    @inbounds for cell in 0:(n_cells - 1)
        for (u, v) in unit_uv
            idx += 1
            phi = 2.0 * pi * u
            q[1, idx] = cx + R * cos(phi)
            q[2, idx] = cy + R * sin(phi)
            q[3, idx] = (v + cell) * T
        end
    end

    box = Types.OrthorhombicBox((Lx, Ly, Lz))
    masses = fill(element_mass, N)
    names = fill(:C, N)
    return (q = q, box = box, masses = masses, names = names,
            radius_nm = R, chiral_angle_rad = chiral_angle_rad)
end

"""
    build_nanopore_membrane(nx, ny; pore_radius_nm=0.45, bond_length_nm=0.142, box_z_nm=4.0, element_mass=12.011)

Build a graphene membrane with a circular nanopore carved around the `(x, y)` center.
"""
function build_nanopore_membrane(nx::Int, ny::Int;
                                 pore_radius_nm::Float64 = 0.45,
                                 bond_length_nm::Float64 = 0.142,
                                 box_z_nm::Float64 = 4.0,
                                 element_mass::Float64 = 12.011)
    pore_radius_nm >= 0.0 || throw(ArgumentError("pore_radius_nm must be >= 0"))
    sheet = build_graphene_sheet(nx, ny; bond_length_nm = bond_length_nm,
                                 vacuum_z_nm = box_z_nm, element_mass = element_mass)
    Lx, Ly, _ = sheet.box.L
    cx = 0.5 * Lx
    cy = 0.5 * Ly
    r2_pore = pore_radius_nm * pore_radius_nm
    keep = Int[]
    q0 = sheet.q
    @inbounds for i in 1:size(q0, 2)
        dx = q0[1, i] - cx
        dy = q0[2, i] - cy
        if dx * dx + dy * dy >= r2_pore
            push!(keep, i)
        end
    end
    !isempty(keep) || throw(ArgumentError("pore_radius_nm=$pore_radius_nm removed all atoms"))
    N = length(keep)
    q = q0[:, keep]
    masses = fill(element_mass, N)
    names = fill(:C, N)
    return (q = q, box = sheet.box, masses = masses, names = names,
            center_xy = (cx, cy), pore_radius_nm = pore_radius_nm)
end

const _DIAMOND_BASIS = (
    (0.0, 0.0, 0.0),
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 0.5, 0.0),
    (0.25, 0.25, 0.25),
    (0.25, 0.75, 0.75),
    (0.75, 0.25, 0.75),
    (0.75, 0.75, 0.25),
)

const _DIAMOND_MASSES = Dict{Symbol,Float64}(
    :Si => 28.085,
    :C  => 12.011,
    :Ge => 72.630,
)

"""
    build_diamond_lattice(nx, ny, nz; a_nm=0.5431, element=:Si)

Build an `nx × ny × nz` supercell of the 8-atom conventional cubic diamond lattice.
"""
function build_diamond_lattice(nx::Int, ny::Int, nz::Int;
                               a_nm::Float64 = 0.5431,
                               element::Symbol = :Si)
    (nx >= 1 && ny >= 1 && nz >= 1) || throw(ArgumentError("nx, ny, nz must be >= 1"))
    a_nm > 0.0 || throw(ArgumentError("a_nm must be > 0"))
    haskey(_DIAMOND_MASSES, element) ||
        throw(ArgumentError("unsupported diamond element :$element; expected :Si, :C, or :Ge"))
    mass = _DIAMOND_MASSES[element]
    N = 8 * nx * ny * nz
    q = Matrix{Float64}(undef, 3, N)
    idx = 0
    @inbounds for ix in 0:(nx - 1), iy in 0:(ny - 1), iz in 0:(nz - 1), (bx, by, bz) in _DIAMOND_BASIS
        idx += 1
        q[1, idx] = (ix + bx) * a_nm
        q[2, idx] = (iy + by) * a_nm
        q[3, idx] = (iz + bz) * a_nm
    end
    box = Types.OrthorhombicBox((nx * a_nm, ny * a_nm, nz * a_nm))
    masses = fill(mass, N)
    names = fill(element, N)
    return (q = q, box = box, masses = masses, names = names)
end

end # module ReactiveNano
