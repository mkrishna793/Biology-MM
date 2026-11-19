# Alchemical free-energy module: generalized soft-core nonbonded factor,
# coupling schedules, and estimators (TI, FEP, BAR, MBAR).
#
# Follows the FoundationCore factor protocol:
#   evaluate!(F, f, q, ctx) -> U, scattering F = -∂U/∂q
#
# Provides exact analytical derivatives with respect to coordinates,
# periodic box matrices (orthorhombic and triclinic), and alchemical parameters,
# ensuring C1 continuity and singularity avoidance at r = 0.

module Alchemical

using LinearAlgebra
using Statistics
using ..Types
using ..Factors
using ..EquilibriumSensitivities

import ..Factors: evaluate!, AbstractFactor, is_local, force_consistent, factor_support_size, read_support

export AlchemicalState, AlchemicalSchedule, evaluate_schedule, schedule_state, schedule_derivatives,
       AlchemicalNonbondedFactor, with_lambdas, alchemical_derivatives, schedule_derivative,
       accumulate_box_derivative!, alchemical_box_derivative,
       TIResult, estimate_ti,
       FEPResult, estimate_fep,
       BARResult, estimate_bar,
       MBARResult, estimate_mbar,
       EvalContext

const EvalContext = Factors.Context

# ---------------------------------------------------------------------------
# Alchemical states and schedules
# ---------------------------------------------------------------------------

"""
    AlchemicalState(lambda_lj, lambda_elec)

Immutable container for coupling parameters `lambda_lj` and `lambda_elec`,
both validated to lie in `[0.0, 1.0]`.
"""
struct AlchemicalState
    lambda_lj::Float64
    lambda_elec::Float64
    function AlchemicalState(lambda_lj::Real, lambda_elec::Real)
        0.0 <= lambda_lj <= 1.0 || throw(ArgumentError("lambda_lj must be in [0, 1], got $lambda_lj"))
        0.0 <= lambda_elec <= 1.0 || throw(ArgumentError("lambda_elec must be in [0, 1], got $lambda_elec"))
        new(Float64(lambda_lj), Float64(lambda_elec))
    end
end

"""
    AlchemicalSchedule(mode::Symbol = :simultaneous; lambda_split::Real = 0.5)

Alchemical coupling schedule defining the mapping from a single scalar progression
parameter `lambda in [0, 1]` to `(lambda_lj, lambda_elec)`.

Supported modes:
- `:simultaneous`: `lambda_lj = lambda`, `lambda_elec = lambda`.
- `:staged_decouple` / `:staged_annihilate`: Electrostatics decouples smoothly on `[0, lambda_split]`
  (from 1 down to 0 while LJ stays 1), and soft-core LJ decouples smoothly on `[lambda_split, 1]`
  (from 1 down to 0 while electrostatics stays 0).
- `:staged_couple` / `:staged_grow`: Soft-core LJ grows from 0 to 1 on `[0, lambda_split]`,
  followed by electrostatics growing from 0 to 1 on `[lambda_split, 1]`.
"""
struct AlchemicalSchedule
    mode::Symbol
    lambda_split::Float64
    function AlchemicalSchedule(mode::Symbol = :simultaneous; lambda_split::Real = 0.5)
        mode in (:simultaneous, :staged_decouple, :staged_annihilate, :staged_couple, :staged_grow) ||
            throw(ArgumentError("schedule mode must be :simultaneous, :staged_decouple, :staged_annihilate, :staged_couple, or :staged_grow; got $(repr(mode))"))
        0.0 < lambda_split < 1.0 ||
            throw(ArgumentError("lambda_split must be in (0, 1), got $lambda_split"))
        new(mode, Float64(lambda_split))
    end
end

"""
    schedule_state(sched::AlchemicalSchedule, lambda::Real) -> AlchemicalState

Evaluate the alchemical state `(lambda_lj, lambda_elec)` at progression parameter `lambda in [0, 1]`.
"""
function schedule_state(sched::AlchemicalSchedule, lambda::Real)
    0.0 <= lambda <= 1.0 || throw(ArgumentError("lambda must be in [0, 1], got $lambda"))
    lam = Float64(lambda)
    s = sched.lambda_split
    if sched.mode === :simultaneous
        return AlchemicalState(lam, lam)
    elseif sched.mode in (:staged_decouple, :staged_annihilate)
        if lam <= s
            lam_elec = 1.0 - lam / s
            lam_lj = 1.0
        else
            lam_elec = 0.0
            lam_lj = 1.0 - (lam - s) / (1.0 - s)
        end
        return AlchemicalState(lam_lj, lam_elec)
    elseif sched.mode in (:staged_couple, :staged_grow)
        if lam <= s
            lam_lj = lam / s
            lam_elec = 0.0
        else
            lam_lj = 1.0
            lam_elec = (lam - s) / (1.0 - s)
        end
        return AlchemicalState(lam_lj, lam_elec)
    end
end

evaluate_schedule(sched::AlchemicalSchedule, lambda::Real) = schedule_state(sched, lambda)

"""
    schedule_derivatives(sched::AlchemicalSchedule, lambda::Real) -> (dlambda_lj::Float64, dlambda_elec::Float64)

Analytic derivatives `(d lambda_lj / d lambda, d lambda_elec / d lambda)` along the schedule.
"""
function schedule_derivatives(sched::AlchemicalSchedule, lambda::Real)
    0.0 <= lambda <= 1.0 || throw(ArgumentError("lambda must be in [0, 1], got $lambda"))
    lam = Float64(lambda)
    s = sched.lambda_split
    if sched.mode === :simultaneous
        return (dlambda_lj = 1.0, dlambda_elec = 1.0)
    elseif sched.mode in (:staged_decouple, :staged_annihilate)
        if lam < s
            return (dlambda_lj = 0.0, dlambda_elec = -1.0 / s)
        elseif lam > s
            return (dlambda_lj = -1.0 / (1.0 - s), dlambda_elec = 0.0)
        else
            return (dlambda_lj = -0.5 / (1.0 - s), dlambda_elec = -0.5 / s)
        end
    elseif sched.mode in (:staged_couple, :staged_grow)
        if lam < s
            return (dlambda_lj = 1.0 / s, dlambda_elec = 0.0)
        elseif lam > s
            return (dlambda_lj = 0.0, dlambda_elec = 1.0 / (1.0 - s))
        else
            return (dlambda_lj = 0.5 / s, dlambda_elec = 0.5 / (1.0 - s))
        end
    end
end

# ---------------------------------------------------------------------------
# Soft-Core Nonbonded Factor
# ---------------------------------------------------------------------------

"""
    AlchemicalNonbondedFactor <: Factors.AbstractFactor

Generalized soft-core nonbonded interaction factor supporting arbitrary pair topologies,
Lorentz-Berthelot Lennard-Jones scaling, and soft-core electrostatics (Beutler et al. / Steinbrecher).
"""
struct AlchemicalNonbondedFactor <: Factors.AbstractFactor
    i::Vector{Int}
    j::Vector{Int}
    sigma::Vector{Float64}
    epsilon::Vector{Float64}
    qi_qj::Vector{Float64}
    lj_scale::Vector{Float64}
    elec_scale::Vector{Float64}
    lambda_lj::Float64
    lambda_elec::Float64
    alpha_lj::Float64
    alpha_elec::Float64
    coulomb_prefactor::Float64
    mode::Symbol
    rcut::Float64

    # Memberwise constructor with full validation
    function AlchemicalNonbondedFactor(
        i::AbstractVector{<:Integer},
        j::AbstractVector{<:Integer},
        sigma::AbstractVector{<:Real},
        epsilon::AbstractVector{<:Real},
        qi_qj::AbstractVector{<:Real},
        lj_scale::AbstractVector{<:Real},
        elec_scale::AbstractVector{<:Real},
        lambda_lj::Real,
        lambda_elec::Real,
        alpha_lj::Real,
        alpha_elec::Real,
        coulomb_prefactor::Real,
        mode::Symbol,
        rcut::Real
    )
        n = length(i)
        length(j) == length(sigma) == length(epsilon) == length(qi_qj) == length(lj_scale) == length(elec_scale) == n ||
            throw(ArgumentError("pair arrays must all have identical length, got i=$n, j=$(length(j)), sigma=$(length(sigma))"))
        mode in (:none, :lfs) || throw(ArgumentError("mode must be :none or :lfs, got $(repr(mode))"))
        if mode === :lfs
            (isfinite(rcut) && rcut > 0) || throw(ArgumentError(":lfs mode requires a positive finite rcut, got $rcut"))
        end
        0.0 <= lambda_lj <= 1.0 || throw(ArgumentError("lambda_lj must be in [0, 1], got $lambda_lj"))
        0.0 <= lambda_elec <= 1.0 || throw(ArgumentError("lambda_elec must be in [0, 1], got $lambda_elec"))
        alpha_lj >= 0.0 || throw(ArgumentError("alpha_lj must be non-negative, got $alpha_lj"))
        alpha_elec >= 0.0 || throw(ArgumentError("alpha_elec must be non-negative, got $alpha_elec"))
        coulomb_prefactor >= 0.0 || throw(ArgumentError("coulomb_prefactor must be non-negative, got $coulomb_prefactor"))

        for p in 1:n
            i[p] > 0 && j[p] > 0 || throw(ArgumentError("atom indices must be positive, got i=$(i[p]), j=$(j[p])"))
            i[p] != j[p] || throw(ArgumentError("self-interaction pair forbidden at index $p: i=j=$(i[p])"))
        end

        i_vec = i isa Vector{Int} ? i : collect(Int, i)
        j_vec = j isa Vector{Int} ? j : collect(Int, j)
        sigma_vec = sigma isa Vector{Float64} ? sigma : collect(Float64, sigma)
        epsilon_vec = epsilon isa Vector{Float64} ? epsilon : collect(Float64, epsilon)
        qi_qj_vec = qi_qj isa Vector{Float64} ? qi_qj : collect(Float64, qi_qj)
        lj_scale_vec = lj_scale isa Vector{Float64} ? lj_scale : collect(Float64, lj_scale)
        elec_scale_vec = elec_scale isa Vector{Float64} ? elec_scale : collect(Float64, elec_scale)

        new(
            i_vec, j_vec,
            sigma_vec, epsilon_vec,
            qi_qj_vec, lj_scale_vec,
            elec_scale_vec,
            Float64(lambda_lj), Float64(lambda_elec),
            Float64(alpha_lj), Float64(alpha_elec),
            Float64(coulomb_prefactor), mode, Float64(rcut)
        )
    end
end

"""
    AlchemicalNonbondedFactor(i, j, sigma, epsilon, qi_qj; ...)

Direct pair constructor with keyword options for scales, lambdas, soft-core parameters, and cutoff.
"""
function AlchemicalNonbondedFactor(
    i::AbstractVector{<:Integer},
    j::AbstractVector{<:Integer},
    sigma::AbstractVector{<:Real},
    epsilon::AbstractVector{<:Real},
    qi_qj::AbstractVector{<:Real};
    lj_scale::AbstractVector{<:Real} = ones(length(i)),
    elec_scale::AbstractVector{<:Real} = ones(length(i)),
    lambda_lj::Real = 1.0,
    lambda_elec::Real = 1.0,
    alpha_lj::Real = 0.5,
    alpha_elec::Real = 0.5,
    coulomb_prefactor::Real = 1.0,
    mode::Symbol = :none,
    rcut::Real = Inf
)
    return AlchemicalNonbondedFactor(
        i, j, sigma, epsilon, qi_qj,
        lj_scale, elec_scale,
        lambda_lj, lambda_elec,
        alpha_lj, alpha_elec,
        coulomb_prefactor, mode, rcut
    )
end

"""
    AlchemicalNonbondedFactor(n_atoms, alchemical_atoms, sigma_atom, epsilon_atom, charges; ...)

Convenience constructor building all environment-alchemical pairs (and optionally intra-alchemical pairs)
using Lorentz-Berthelot combining rules:
    σ_ij = 0.5 * (σ_i + σ_j)
    ε_ij = sqrt(ε_i * ε_j)
    q_i q_j = q_i * q_j
"""
function AlchemicalNonbondedFactor(
    n_atoms::Int,
    alchemical_atoms::AbstractVector{<:Integer},
    sigma_atom::AbstractVector{<:Real},
    epsilon_atom::AbstractVector{<:Real},
    charges::AbstractVector{<:Real};
    excluded_pairs::AbstractSet{Tuple{Int,Int}} = Set{Tuple{Int,Int}}(),
    scale_14_pairs::AbstractDict{Tuple{Int,Int},Tuple{Float64,Float64}} = Dict{Tuple{Int,Int},Tuple{Float64,Float64}}(),
    include_intra_alchemical::Bool = false,
    lambda_lj::Real = 1.0,
    lambda_elec::Real = 1.0,
    alpha_lj::Real = 0.5,
    alpha_elec::Real = 0.5,
    coulomb_prefactor::Real = 1.0,
    mode::Symbol = :none,
    rcut::Real = Inf
)
    n_atoms > 0 || throw(ArgumentError("n_atoms must be positive, got $n_atoms"))
    length(sigma_atom) == length(epsilon_atom) == length(charges) == n_atoms ||
        throw(ArgumentError("per-atom parameter arrays must have length $n_atoms"))

    alch_set = Set{Int}(alchemical_atoms)
    for a in alch_set
        1 <= a <= n_atoms || throw(ArgumentError("alchemical atom index $a out of bounds 1:$n_atoms"))
    end

    i_vec = Int[]
    j_vec = Int[]
    sig_vec = Float64[]
    eps_vec = Float64[]
    qiqj_vec = Float64[]
    lj_sc = Float64[]
    el_sc = Float64[]

    seen = Set{Tuple{Int,Int}}()

    # Helper to add pair
    function add_pair(a::Int, b::Int)
        idx1 = min(a, b)
        idx2 = max(a, b)
        pair = (idx1, idx2)
        pair in seen && return
        push!(seen, pair)

        if pair in excluded_pairs || (idx2, idx1) in excluded_pairs
            return
        end

        lj_s = 1.0
        el_s = 1.0
        if haskey(scale_14_pairs, pair)
            lj_s, el_s = scale_14_pairs[pair]
        elseif haskey(scale_14_pairs, (idx2, idx1))
            lj_s, el_s = scale_14_pairs[(idx2, idx1)]
        end

        s_ij = 0.5 * (Float64(sigma_atom[idx1]) + Float64(sigma_atom[idx2]))
        e_ij = sqrt(Float64(epsilon_atom[idx1]) * Float64(epsilon_atom[idx2]))
        q_ij = Float64(charges[idx1]) * Float64(charges[idx2])

        push!(i_vec, idx1)
        push!(j_vec, idx2)
        push!(sig_vec, s_ij)
        push!(eps_vec, e_ij)
        push!(qiqj_vec, q_ij)
        push!(lj_sc, lj_s)
        push!(el_sc, el_s)
    end

    # Environment - Alchemical pairs
    for a in alchemical_atoms
        for e in 1:n_atoms
            if !(e in alch_set)
                add_pair(Int(a), Int(e))
            end
        end
    end

    # Intra-alchemical pairs if requested
    if include_intra_alchemical
        alch_sorted = sort(collect(alch_set))
        for p1 in 1:length(alch_sorted)-1
            for p2 in p1+1:length(alch_sorted)
                add_pair(alch_sorted[p1], alch_sorted[p2])
            end
        end
    end

    return AlchemicalNonbondedFactor(
        i_vec, j_vec, sig_vec, eps_vec, qiqj_vec;
        lj_scale = lj_sc, elec_scale = el_sc,
        lambda_lj = lambda_lj, lambda_elec = lambda_elec,
        alpha_lj = alpha_lj, alpha_elec = alpha_elec,
        coulomb_prefactor = coulomb_prefactor,
        mode = mode, rcut = rcut
    )
end

"""
    with_lambdas(f::AlchemicalNonbondedFactor, lambda_lj::Real, lambda_elec::Real) -> AlchemicalNonbondedFactor

Return a shallow copy of factor `f` with updated lambda parameters without reallocating pair topology arrays.
"""
function with_lambdas(f::AlchemicalNonbondedFactor, lambda_lj::Real, lambda_elec::Real)
    0.0 <= lambda_lj <= 1.0 || throw(ArgumentError("lambda_lj must be in [0, 1], got $lambda_lj"))
    0.0 <= lambda_elec <= 1.0 || throw(ArgumentError("lambda_elec must be in [0, 1], got $lambda_elec"))
    return AlchemicalNonbondedFactor(
        f.i, f.j, f.sigma, f.epsilon, f.qi_qj,
        f.lj_scale, f.elec_scale,
        Float64(lambda_lj), Float64(lambda_elec),
        f.alpha_lj, f.alpha_elec, f.coulomb_prefactor,
        f.mode, f.rcut
    )
end

# Factor protocol metadata
Factors.is_local(::AlchemicalNonbondedFactor) = false
Factors.force_consistent(::AlchemicalNonbondedFactor) = true
Factors.factor_support_size(f::AlchemicalNonbondedFactor) =
    isempty(f.i) ? 0 : length(unique(vcat(f.i, f.j)))
Factors.read_support(f::AlchemicalNonbondedFactor) =
    (unique(vcat(f.i, f.j))...,)

# ---------------------------------------------------------------------------
# Core Analytical Kernels (Beutler LJ and Steinbrecher Electrostatics)
# ---------------------------------------------------------------------------

@inline function _soft_lj_raw(r::Float64, sigma::Float64, eps::Float64, lam::Float64, alpha::Float64)
    s6 = sigma^6
    s12 = s6 * s6
    r6 = (r * r)^3
    S = alpha * (1.0 - lam) * s6 + r6
    invS = 1.0 / S
    invS2 = invS * invS
    invS3 = invS2 * invS

    # Raw energy
    u = 4.0 * eps * lam * (s12 * invS2 - s6 * invS)

    # Raw radial derivative dudr and (1/r)*dudr
    if r == 0.0
        dudr = 0.0
        inv_r_dudr = 0.0
    else
        r5 = r * (r * r)^2
        dudr = -24.0 * eps * lam * r5 * s6 * (2.0 * s6 - S) * invS3
        inv_r_dudr = -24.0 * eps * lam * (r * r)^2 * s6 * (2.0 * s6 - S) * invS3
    end

    # Raw lambda derivative
    dudlam = 4.0 * eps * (s12 * invS2 - s6 * invS) + 4.0 * eps * lam * alpha * s12 * (2.0 * s6 - S) * invS3

    return (u = u, dudr = dudr, inv_r_dudr = inv_r_dudr, dudlam = dudlam)
end

@inline function _soft_lj_d2udrdlam_raw(rc::Float64, sigma::Float64, eps::Float64, lam::Float64, alpha::Float64)
    s6 = sigma^6
    rc6 = (rc * rc)^3
    Sc = alpha * (1.0 - lam) * s6 + rc6
    invSc = 1.0 / Sc
    invSc3 = invSc^3
    invSc4 = invSc3 * invSc
    rc5 = rc * (rc * rc)^2

    term1 = (2.0 * s6 - Sc) * invSc3
    term2 = lam * alpha * s6 * (6.0 * s6 - 2.0 * Sc) * invSc4
    return -24.0 * eps * rc5 * s6 * (term1 + term2)
end

@inline function _soft_elec_raw(r::Float64, qiqj::Float64, lam::Float64, alpha::Float64, pref::Float64)
    S = alpha * (1.0 - lam) + r * r
    Re = sqrt(S)
    invRe = 1.0 / Re
    invRe3 = invRe^3

    kq = pref * qiqj
    u = kq * lam * invRe

    if r == 0.0
        dudr = 0.0
        inv_r_dudr = -kq * lam * invRe3
    else
        dudr = -kq * lam * r * invRe3
        inv_r_dudr = -kq * lam * invRe3
    end

    dudlam = kq * (invRe + 0.5 * lam * alpha * invRe3)
    return (u = u, dudr = dudr, inv_r_dudr = inv_r_dudr, dudlam = dudlam)
end

@inline function _soft_elec_d2udrdlam_raw(rc::Float64, qiqj::Float64, lam::Float64, alpha::Float64, pref::Float64)
    S = alpha * (1.0 - lam) + rc * rc
    Re = sqrt(S)
    invRe = 1.0 / Re
    invRe3 = invRe^3
    invRe5 = invRe3 * (invRe * invRe)
    kq = pref * qiqj
    return -kq * rc * (invRe3 + 1.5 * lam * alpha * invRe5)
end

# ---------------------------------------------------------------------------
# Factor Evaluation (energy + Cartesian forces)
# ---------------------------------------------------------------------------

function _evaluate_alchemical!(
    F::AbstractMatrix{Float64},
    f::AlchemicalNonbondedFactor,
    q::AbstractMatrix{Float64},
    box::Types.AbstractBox
)
    size(q, 1) == 3 && size(F) == size(q) ||
        throw(ArgumentError("q and F must be 3 x N, got q=$(size(q)), F=$(size(F))"))
    n_pairs = length(f.i)
    total_u = 0.0
    pref = f.coulomb_prefactor
    is_lfs = f.mode === :lfs
    rc = f.rcut
    rc2 = rc * rc

    for p in 1:n_pairs
        i = f.i[p]
        j = f.j[p]
        d = Types.unwrap_displacement(box, q, i, j)
        r2 = Types.norm2(d)
        if is_lfs && r2 >= rc2
            continue
        end
        r = sqrt(r2)
        s_lj = f.lj_scale[p]
        s_elec = f.elec_scale[p]

        u_pair = 0.0
        inv_r_dudr_pair = 0.0

        # Lennard-Jones
        if s_lj != 0.0 && f.epsilon[p] != 0.0 && f.lambda_lj != 0.0
            raw = _soft_lj_raw(r, f.sigma[p], f.epsilon[p], f.lambda_lj, f.alpha_lj)
            if !is_lfs
                u_pair += s_lj * raw.u
                inv_r_dudr_pair += s_lj * raw.inv_r_dudr
            else
                raw_c = _soft_lj_raw(rc, f.sigma[p], f.epsilon[p], f.lambda_lj, f.alpha_lj)
                u_lfs = raw.u - raw_c.u - (r - rc) * raw_c.dudr
                u_pair += s_lj * u_lfs
                if r > 0.0
                    inv_r_dudr_lfs = (raw.dudr - raw_c.dudr) / r
                    inv_r_dudr_pair += s_lj * inv_r_dudr_lfs
                end
            end
        end

        # Electrostatics
        if s_elec != 0.0 && f.qi_qj[p] != 0.0 && f.lambda_elec != 0.0
            raw_el = _soft_elec_raw(r, f.qi_qj[p], f.lambda_elec, f.alpha_elec, pref)
            if !is_lfs
                u_pair += s_elec * raw_el.u
                inv_r_dudr_pair += s_elec * raw_el.inv_r_dudr
            else
                raw_el_c = _soft_elec_raw(rc, f.qi_qj[p], f.lambda_elec, f.alpha_elec, pref)
                u_el_lfs = raw_el.u - raw_el_c.u - (r - rc) * raw_el_c.dudr
                u_pair += s_elec * u_el_lfs
                if r > 0.0
                    inv_r_dudr_el_lfs = (raw_el.dudr - raw_el_c.dudr) / r
                    inv_r_dudr_pair += s_elec * inv_r_dudr_el_lfs
                end
            end
        end

        total_u += u_pair

        if r > 0.0 && inv_r_dudr_pair != 0.0
            c = -inv_r_dudr_pair
            gx = c * d.x
            gy = c * d.y
            gz = c * d.z
            F[1, i] += gx; F[2, i] += gy; F[3, i] += gz
            F[1, j] -= gx; F[2, j] -= gy; F[3, j] -= gz
        end
    end
    return total_u
end

# Protocol implementations
function Factors.evaluate!(F::AbstractMatrix{Float64}, f::AlchemicalNonbondedFactor,
                           q::AbstractMatrix{Float64}, ctx::Factors.Context)
    return _evaluate_alchemical!(F, f, q, ctx.box)
end

function Factors.evaluate!(F::AbstractMatrix{Float64}, f::AlchemicalNonbondedFactor,
                           q::AbstractMatrix{Float64}, box::Types.AbstractBox)
    return _evaluate_alchemical!(F, f, q, box)
end

function Factors.evaluate!(f::AlchemicalNonbondedFactor, q::AbstractMatrix{Float64},
                           F::AbstractMatrix{Float64}, ctx)
    box = ctx isa Factors.Context ? ctx.box : (ctx isa Types.AbstractBox ? ctx : ctx.box)
    return _evaluate_alchemical!(F, f, q, box)
end

# ---------------------------------------------------------------------------
# Box Derivatives (Periodic Orthorhombic & Triclinic)
# ---------------------------------------------------------------------------

"""
    accumulate_box_derivative!(G::AbstractMatrix{Float64}, f::AlchemicalNonbondedFactor, q, box::OrthorhombicBox)
    accumulate_box_derivative!(G::AbstractMatrix{Float64}, f::AlchemicalNonbondedFactor, q, box::TriclinicBox)

In-place accumulation of the exact analytic periodic box derivative `G[a, b] += ∂U / ∂H[a, b]`
at fixed fractional coordinates.
"""
function accumulate_box_derivative!(
    G::AbstractMatrix{Float64},
    f::AlchemicalNonbondedFactor,
    q::AbstractMatrix{Float64},
    box::Types.OrthorhombicBox
)
    size(G) == (3, 3) || throw(ArgumentError("G must be 3x3, got $(size(G))"))
    n_pairs = length(f.i)
    pref = f.coulomb_prefactor
    is_lfs = f.mode === :lfs
    rc = f.rcut
    rc2 = rc * rc
    L = box.L

    for p in 1:n_pairs
        i = f.i[p]
        j = f.j[p]
        d = Types.unwrap_displacement(box, q, i, j)
        r2 = Types.norm2(d)
        if is_lfs && r2 >= rc2
            continue
        end
        r = sqrt(r2)
        r == 0.0 && continue

        s_lj = f.lj_scale[p]
        s_elec = f.elec_scale[p]
        inv_r_dudr_pair = 0.0

        if s_lj != 0.0 && f.epsilon[p] != 0.0 && f.lambda_lj != 0.0
            raw = _soft_lj_raw(r, f.sigma[p], f.epsilon[p], f.lambda_lj, f.alpha_lj)
            if !is_lfs
                inv_r_dudr_pair += s_lj * raw.inv_r_dudr
            else
                raw_c = _soft_lj_raw(rc, f.sigma[p], f.epsilon[p], f.lambda_lj, f.alpha_lj)
                inv_r_dudr_pair += s_lj * ((raw.dudr - raw_c.dudr) / r)
            end
        end

        if s_elec != 0.0 && f.qi_qj[p] != 0.0 && f.lambda_elec != 0.0
            raw_el = _soft_elec_raw(r, f.qi_qj[p], f.lambda_elec, f.alpha_elec, pref)
            if !is_lfs
                inv_r_dudr_pair += s_elec * raw_el.inv_r_dudr
            else
                raw_el_c = _soft_elec_raw(rc, f.qi_qj[p], f.lambda_elec, f.alpha_elec, pref)
                inv_r_dudr_pair += s_elec * ((raw_el.dudr - raw_el_c.dudr) / r)
            end
        end

        inv_r_dudr_pair == 0.0 && continue

        # Fractional displacement for Orthorhombic cell
        wx = d.x / L[1]; wy = d.y / L[2]; wz = d.z / L[3]
        gx = inv_r_dudr_pair * d.x
        gy = inv_r_dudr_pair * d.y
        gz = inv_r_dudr_pair * d.z

        G[1, 1] += gx * wx; G[1, 2] += gx * wy; G[1, 3] += gx * wz
        G[2, 1] += gy * wx; G[2, 2] += gy * wy; G[2, 3] += gy * wz
        G[3, 1] += gz * wx; G[3, 2] += gz * wy; G[3, 3] += gz * wz
    end
    return G
end

function accumulate_box_derivative!(
    G::AbstractMatrix{Float64},
    f::AlchemicalNonbondedFactor,
    q::AbstractMatrix{Float64},
    box::Types.TriclinicBox
)
    size(G) == (3, 3) || throw(ArgumentError("G must be 3x3, got $(size(G))"))
    n_pairs = length(f.i)
    pref = f.coulomb_prefactor
    is_lfs = f.mode === :lfs
    rc = f.rcut
    rc2 = rc * rc
    M = box.Hinv

    for p in 1:n_pairs
        i = f.i[p]
        j = f.j[p]
        d = Types.unwrap_displacement(box, q, i, j)
        r2 = Types.norm2(d)
        if is_lfs && r2 >= rc2
            continue
        end
        r = sqrt(r2)
        r == 0.0 && continue

        s_lj = f.lj_scale[p]
        s_elec = f.elec_scale[p]
        inv_r_dudr_pair = 0.0

        if s_lj != 0.0 && f.epsilon[p] != 0.0 && f.lambda_lj != 0.0
            raw = _soft_lj_raw(r, f.sigma[p], f.epsilon[p], f.lambda_lj, f.alpha_lj)
            if !is_lfs
                inv_r_dudr_pair += s_lj * raw.inv_r_dudr
            else
                raw_c = _soft_lj_raw(rc, f.sigma[p], f.epsilon[p], f.lambda_lj, f.alpha_lj)
                inv_r_dudr_pair += s_lj * ((raw.dudr - raw_c.dudr) / r)
            end
        end

        if s_elec != 0.0 && f.qi_qj[p] != 0.0 && f.lambda_elec != 0.0
            raw_el = _soft_elec_raw(r, f.qi_qj[p], f.lambda_elec, f.alpha_elec, pref)
            if !is_lfs
                inv_r_dudr_pair += s_elec * raw_el.inv_r_dudr
            else
                raw_el_c = _soft_elec_raw(rc, f.qi_qj[p], f.lambda_elec, f.alpha_elec, pref)
                inv_r_dudr_pair += s_elec * ((raw_el.dudr - raw_el_c.dudr) / r)
            end
        end

        inv_r_dudr_pair == 0.0 && continue

        # Fractional displacement for Triclinic cell: w = Hinv * d
        wx = M[1, 1] * d.x + M[1, 2] * d.y + M[1, 3] * d.z
        wy = M[2, 1] * d.x + M[2, 2] * d.y + M[2, 3] * d.z
        wz = M[3, 1] * d.x + M[3, 2] * d.y + M[3, 3] * d.z

        gx = inv_r_dudr_pair * d.x
        gy = inv_r_dudr_pair * d.y
        gz = inv_r_dudr_pair * d.z

        G[1, 1] += gx * wx; G[1, 2] += gx * wy; G[1, 3] += gx * wz
        G[2, 1] += gy * wx; G[2, 2] += gy * wy; G[2, 3] += gy * wz
        G[3, 1] += gz * wx; G[3, 2] += gz * wy; G[3, 3] += gz * wz
    end
    return G
end

function accumulate_box_derivative!(
    G::AbstractMatrix{Float64},
    f::AlchemicalNonbondedFactor,
    q::AbstractMatrix{Float64},
    box::Types.NoBox
)
    throw(ArgumentError("no periodic box: box derivative ∂U/∂H is undefined for NoBox"))
end

function accumulate_box_derivative!(
    G::AbstractMatrix{Float64},
    f::AlchemicalNonbondedFactor,
    q::AbstractMatrix{Float64},
    ctx::Factors.Context
)
    return accumulate_box_derivative!(G, f, q, ctx.box)
end

"""
    alchemical_box_derivative(f, q, box) -> Matrix{Float64}

Allocate and return a fresh 3x3 box derivative matrix `∂U / ∂H`.
"""
function alchemical_box_derivative(
    f::AlchemicalNonbondedFactor,
    q::AbstractMatrix{Float64},
    box::Types.AbstractBox
)
    G = zeros(Float64, 3, 3)
    accumulate_box_derivative!(G, f, q, box)
    return G
end

function alchemical_box_derivative(
    f::AlchemicalNonbondedFactor,
    q::AbstractMatrix{Float64},
    ctx::Factors.Context
)
    return alchemical_box_derivative(f, q, ctx.box)
end

if isdefined(parentmodule(@__MODULE__), :ResolvedSystem)
    import ..ResolvedSystem as _RS
    _RS.box_derivative_support(::AlchemicalNonbondedFactor) = nothing
    _RS.accumulate_box_derivative!(G::Matrix{Float64}, f::AlchemicalNonbondedFactor,
                                    q::AbstractMatrix{Float64}, ctx::Factors.Context) =
        accumulate_box_derivative!(G, f, q, ctx.box)
end

# ---------------------------------------------------------------------------
# Alchemical Parameter Derivatives (∂U/∂λ_LJ and ∂U/∂λ_elec)
# ---------------------------------------------------------------------------

"""
    alchemical_derivatives(f::AlchemicalNonbondedFactor, q, box = NoBox())
        -> (energy::Float64, dU_dlambda_lj::Float64, dU_dlambda_elec::Float64)

Compute exact analytical total energy and derivatives with respect to lambda_lj and lambda_elec.
"""
function alchemical_derivatives(
    f::AlchemicalNonbondedFactor,
    q::AbstractMatrix{Float64},
    box::Types.AbstractBox = Types.NoBox()
)
    size(q, 1) == 3 || throw(ArgumentError("q must have shape 3 x N, got $(size(q))"))
    n_pairs = length(f.i)
    total_u = 0.0
    total_dU_dlj = 0.0
    total_dU_delec = 0.0
    pref = f.coulomb_prefactor
    is_lfs = f.mode === :lfs
    rc = f.rcut
    rc2 = rc * rc

    for p in 1:n_pairs
        i = f.i[p]
        j = f.j[p]
        d = Types.unwrap_displacement(box, q, i, j)
        r2 = Types.norm2(d)
        if is_lfs && r2 >= rc2
            continue
        end
        r = sqrt(r2)
        s_lj = f.lj_scale[p]
        s_elec = f.elec_scale[p]

        # Lennard-Jones
        if s_lj != 0.0 && f.epsilon[p] != 0.0
            raw = _soft_lj_raw(r, f.sigma[p], f.epsilon[p], f.lambda_lj, f.alpha_lj)
            if !is_lfs
                total_u += s_lj * raw.u
                total_dU_dlj += s_lj * raw.dudlam
            else
                raw_c = _soft_lj_raw(rc, f.sigma[p], f.epsilon[p], f.lambda_lj, f.alpha_lj)
                u_lfs = raw.u - raw_c.u - (r - rc) * raw_c.dudr
                d2 = _soft_lj_d2udrdlam_raw(rc, f.sigma[p], f.epsilon[p], f.lambda_lj, f.alpha_lj)
                dudlam_lfs = raw.dudlam - raw_c.dudlam - (r - rc) * d2
                total_u += s_lj * u_lfs
                total_dU_dlj += s_lj * dudlam_lfs
            end
        end

        # Electrostatics
        if s_elec != 0.0 && f.qi_qj[p] != 0.0
            raw_el = _soft_elec_raw(r, f.qi_qj[p], f.lambda_elec, f.alpha_elec, pref)
            if !is_lfs
                total_u += s_elec * raw_el.u
                total_dU_delec += s_elec * raw_el.dudlam
            else
                raw_el_c = _soft_elec_raw(rc, f.qi_qj[p], f.lambda_elec, f.alpha_elec, pref)
                u_el_lfs = raw_el.u - raw_el_c.u - (r - rc) * raw_el_c.dudr
                d2_el = _soft_elec_d2udrdlam_raw(rc, f.qi_qj[p], f.lambda_elec, f.alpha_elec, pref)
                dudlam_el_lfs = raw_el.dudlam - raw_el_c.dudlam - (r - rc) * d2_el
                total_u += s_elec * u_el_lfs
                total_dU_delec += s_elec * dudlam_el_lfs
            end
        end
    end
    return (energy = total_u, dU_dlambda_lj = total_dU_dlj, dU_dlambda_elec = total_dU_delec)
end

function alchemical_derivatives(
    f::AlchemicalNonbondedFactor,
    q::AbstractMatrix{Float64},
    ctx::Factors.Context
)
    return alchemical_derivatives(f, q, ctx.box)
end

"""
    schedule_derivative(f::AlchemicalNonbondedFactor, q, sched::AlchemicalSchedule, lambda::Real; box = NoBox())
        -> (energy, dU_dlambda, dU_dlambda_lj, dU_dlambda_elec)

Compute exact analytical derivative along the coupling schedule using the chain rule:
    dU / dλ = (∂U / ∂λ_LJ) * (dλ_LJ / dλ) + (∂U / ∂λ_elec) * (dλ_elec / dλ)
"""
function schedule_derivative(
    f::AlchemicalNonbondedFactor,
    q::AbstractMatrix{Float64},
    sched::AlchemicalSchedule,
    lambda::Real;
    box::Types.AbstractBox = Types.NoBox()
)
    st = schedule_state(sched, lambda)
    f_lam = with_lambdas(f, st.lambda_lj, st.lambda_elec)
    res = alchemical_derivatives(f_lam, q, box)
    dlam = schedule_derivatives(sched, lambda)
    dU_dlambda = res.dU_dlambda_lj * dlam.dlambda_lj + res.dU_dlambda_elec * dlam.dlambda_elec
    return (
        energy = res.energy,
        dU_dlambda = dU_dlambda,
        dU_dlambda_lj = res.dU_dlambda_lj,
        dU_dlambda_elec = res.dU_dlambda_elec
    )
end

function schedule_derivative(
    f::AlchemicalNonbondedFactor,
    q::AbstractMatrix{Float64},
    sched::AlchemicalSchedule,
    lambda::Real,
    ctx::Factors.Context
)
    return schedule_derivative(f, q, sched, lambda; box = ctx.box)
end

# ---------------------------------------------------------------------------
# Free Energy Estimators (TI, FEP, BAR, MBAR)
# ---------------------------------------------------------------------------

"""
    TIResult

Result container for Thermodynamic Integration.
- `delta_f`: Integrated free energy difference.
- `se`: Propagated standard error across windows.
- `window_means`: Per-window mean dU/dlambda.
- `window_ses`: Per-window standard errors.
- `lambdas`: Sorted lambda values.
"""
struct TIResult
    delta_f::Float64
    se::Float64
    window_means::Vector{Float64}
    window_ses::Vector{Float64}
    lambdas::Vector{Float64}
end

"""
    estimate_ti(lambdas::AbstractVector{<:Real}, dU_samples::AbstractVector{<:AbstractVector{<:Real}}) -> TIResult

Thermodynamic integration using trapezoidal quadrature with autocorrelation-aware standard errors.
"""
function estimate_ti(
    lambdas::AbstractVector{<:Real},
    dU_samples::AbstractVector{<:AbstractVector{<:Real}}
)
    K = length(lambdas)
    K >= 2 || throw(ArgumentError("TI requires at least 2 lambda windows, got $K"))
    length(dU_samples) == K || throw(ArgumentError("dU_samples length $(length(dU_samples)) != lambdas length $K"))

    perm = sortperm(lambdas)
    lam_sorted = Float64.(lambdas[perm])
    for k in 1:K-1
        lam_sorted[k+1] > lam_sorted[k] ||
            throw(ArgumentError("lambdas must be strictly increasing, got $(lam_sorted[k]) and $(lam_sorted[k+1])"))
    end

    means = zeros(Float64, K)
    ses = zeros(Float64, K)

    for k in 1:K
        s = Float64.(dU_samples[perm[k]])
        n = length(s)
        n >= 1 || throw(ArgumentError("window $k has 0 samples"))
        means[k] = mean(s)
        if n == 1
            ses[k] = 0.0
        else
            v = var(s)
            if n >= 8
                ess = EquilibriumSensitivities.effective_sample_size(s)
                ess_safe = (isfinite(ess) && ess > 1.0) ? min(Float64(n), ess) : 1.0
                ses[k] = sqrt(v / ess_safe)
            else
                ses[k] = sqrt(v / n)
            end
        end
    end

    # Trapezoidal weights
    w = zeros(Float64, K)
    w[1] = 0.5 * (lam_sorted[2] - lam_sorted[1])
    w[K] = 0.5 * (lam_sorted[K] - lam_sorted[K-1])
    for k in 2:K-1
        w[k] = 0.5 * (lam_sorted[k+1] - lam_sorted[k-1])
    end

    delta_f = sum(w .* means)
    se_total = sqrt(sum((w .* ses).^2))

    return TIResult(delta_f, se_total, means, ses, lam_sorted)
end

"""
    FEPResult

Result container for Free Energy Perturbation (Zwanzig relation).
- `delta_f`: Estimated free energy difference.
- `se`: Standard error estimated via delta method.
- `ess`: Effective sample size of Boltzmann weights.
"""
struct FEPResult
    delta_f::Float64
    se::Float64
    ess::Float64
end

"""
    estimate_fep(delta_u::AbstractVector{<:Real}, beta::Real) -> FEPResult

Zwanzig Free Energy Perturbation: `ΔF = - (1/β) log < exp(-β ΔU) >`
using log-sum-exp stabilization and Kish's effective sample size.
"""
function estimate_fep(delta_u::AbstractVector{<:Real}, beta::Real)
    beta > 0.0 || throw(ArgumentError("beta must be positive, got $beta"))
    n = length(delta_u)
    n >= 1 || throw(ArgumentError("delta_u must contain at least 1 sample"))

    b = Float64(beta)
    du = Float64.(delta_u)
    all(isfinite, du) || throw(ArgumentError("delta_u contains non-finite values"))

    x = -b .* du
    m = maximum(x)
    weights = exp.(x .- m)
    sum_w = sum(weights)
    mean_w = sum_w / n

    delta_f = -(1.0 / b) * (m + log(mean_w))

    # Kish ESS
    norm_w = weights ./ sum_w
    ess = 1.0 / sum(norm_w.^2)

    # Delta method SE
    if n > 1
        s_w = std(weights)
        ess_safe = max(1.0, min(Float64(n), ess))
        se = (1.0 / b) * (s_w / (mean_w * sqrt(ess_safe)))
    else
        se = 0.0
    end

    return FEPResult(delta_f, se, ess)
end

"""
    BARResult

Result container for Bennett Acceptance Ratio.
- `delta_f`: Optimal maximum-likelihood free energy difference.
- `se`: Bennett asymptotic standard error.
- `overlap`: Phase space overlap metric in `[0, 1]`.
- `iterations`: Number of root-finding iterations.
- `converged`: Convergence status.
"""
struct BARResult
    delta_f::Float64
    se::Float64
    overlap::Float64
    iterations::Int
    converged::Bool
end

"""
    estimate_bar(w_forward, w_reverse, beta; tol = 1e-12, max_iter = 500) -> BARResult

Bennett Acceptance Ratio solving:
    ∑_i f(β w_{F, i} - C) = ∑_j f(β w_{R, j} + C)
where C = β ΔF - log(N_R / N_F) and f(z) = 1 / (1 + exp(z)).
"""
function estimate_bar(
    w_forward::AbstractVector{<:Real},
    w_reverse::AbstractVector{<:Real},
    beta::Real;
    tol::Float64 = 1e-12,
    max_iter::Int = 500
)
    beta > 0.0 || throw(ArgumentError("beta must be positive, got $beta"))
    nf = length(w_forward)
    nr = length(w_reverse)
    nf >= 1 && nr >= 1 || throw(ArgumentError("forward and reverse samples must be non-empty"))

    b = Float64(beta)
    wf = Float64.(w_forward)
    wr = Float64.(w_reverse)
    all(isfinite, wf) && all(isfinite, wr) || throw(ArgumentError("work samples must be finite"))

    # Initial guess from FEP forward/reverse
    fep_f = estimate_fep(wf, b).delta_f
    fep_r = -estimate_fep(wr, b).delta_f
    df_init = isfinite(fep_f) && isfinite(fep_r) ? 0.5 * (fep_f + fep_r) : 0.5 * (mean(wf) - mean(wr))
    C = b * df_init - log(nr / nf)

    converged = false
    iter = 0
    sf = 0.0
    sr = 0.0

    for it in 1:max_iter
        iter = it
        g_val = 0.0
        gp_val = 0.0
        sf = 0.0
        sr = 0.0

        for w in wf
            z = b * w - C
            fz = z >= 0.0 ? exp(-z) / (1.0 + exp(-z)) : 1.0 / (1.0 + exp(z))
            dfz = fz * (1.0 - fz)
            g_val += fz
            gp_val += dfz
            sf += dfz
        end

        for w in wr
            z = b * w + C
            fz = z >= 0.0 ? exp(-z) / (1.0 + exp(-z)) : 1.0 / (1.0 + exp(z))
            dfz = fz * (1.0 - fz)
            g_val -= fz
            gp_val += dfz
            sr += dfz
        end

        if abs(g_val) < tol * (nf + nr)
            converged = true
            break
        end

        gp_safe = max(gp_val, 1e-15)
        step = clamp(g_val / gp_safe, -5.0, 5.0)
        C -= step

        if abs(step) < tol
            converged = true
            break
        end
    end

    delta_f = (C + log(nr / nf)) / b
    v_asymp = 1.0 / (sf + sr) - 1.0 / nf - 1.0 / nr
    se = (1.0 / b) * sqrt(max(0.0, v_asymp))
    overlap = clamp(4.0 * (sf + sr) / (nf + nr), 0.0, 1.0)

    return BARResult(delta_f, se, overlap, iter, converged)
end

"""
    MBARResult

Result container for Multistate Bennett Acceptance Ratio.
- `f_k`: Dimensionless free energies for all states (pinned to `f_1 = 0`).
- `delta_f`: Dimensionless free energy difference `f_K - f_1`.
- `cov_f`: Covariance matrix of free energies.
- `se_delta_f`: Asymptotic standard error of `delta_f`.
- `overlap_matrix`: K x K state overlap / transition matrix.
- `iterations`: Iteration count.
- `converged`: Convergence status.
"""
struct MBARResult
    f_k::Vector{Float64}
    delta_f::Float64
    cov_f::Matrix{Float64}
    se_delta_f::Float64
    overlap_matrix::Matrix{Float64}
    iterations::Int
    converged::Bool
end

"""
    estimate_mbar(u_kn::AbstractMatrix{<:Real}, N_k::AbstractVector{<:Integer}; tol = 1e-10, max_iter = 1000) -> MBARResult

Multistate Bennett Acceptance Ratio (Shirts & Chodera 2008).
`u_kn` is the K x N_tot reduced potential matrix evaluated at all K states.
"""
function estimate_mbar(
    u_kn::AbstractMatrix{<:Real},
    N_k::AbstractVector{<:Integer};
    tol::Float64 = 1e-10,
    max_iter::Int = 1000
)
    K = length(N_k)
    K >= 2 || throw(ArgumentError("MBAR requires at least 2 states, got $K"))
    size(u_kn, 1) == K || throw(ArgumentError("u_kn row count $(size(u_kn, 1)) != state count $K"))
    N_tot = sum(N_k)
    size(u_kn, 2) == N_tot || throw(ArgumentError("u_kn col count $(size(u_kn, 2)) != total samples $N_tot"))

    U = Matrix{Float64}(u_kn)
    all(isfinite, U) || throw(ArgumentError("u_kn contains non-finite entries"))
    all(n -> n > 0, N_k) || throw(ArgumentError("each state must have at least 1 sample"))

    f = zeros(Float64, K)
    log_Nk = log.(Float64.(N_k))
    converged = false
    iter = 0

    log_D = zeros(Float64, N_tot)
    f_new = zeros(Float64, K)

    for it in 1:max_iter
        iter = it
        f_old = copy(f)
        v = f .+ log_Nk

        # Denominator log D_n for each configuration n
        for n in 1:N_tot
            m = -Inf
            for j in 1:K
                val = v[j] - U[j, n]
                if val > m
                    m = val
                end
            end
            s = 0.0
            for j in 1:K
                s += exp(v[j] - U[j, n] - m)
            end
            log_D[n] = m + log(s)
        end

        # Numerator sum for each state i
        for i in 1:K
            m_i = -Inf
            for n in 1:N_tot
                val = -U[i, n] - log_D[n]
                if val > m_i
                    m_i = val
                end
            end
            s_i = 0.0
            for n in 1:N_tot
                s_i += exp(-U[i, n] - log_D[n] - m_i)
            end
            f_new[i] = -(m_i + log(s_i))
        end

        # Pin f_1 = 0.0
        f_new .-= f_new[1]
        delta_max = maximum(abs.(f_new .- f_old))
        f .= f_new

        if delta_max < tol
            converged = true
            break
        end
    end

    # Weight matrix W of size N_tot x K
    W = zeros(Float64, N_tot, K)
    for n in 1:N_tot
        v = f .+ log_Nk
        m = -Inf
        for j in 1:K
            val = v[j] - U[j, n]
            if val > m
                m = val
            end
        end
        s = 0.0
        for j in 1:K
            s += exp(v[j] - U[j, n] - m)
        end
        log_Dn = m + log(s)

        for k in 1:K
            W[n, k] = exp(f[k] - U[k, n] - log_Dn)
        end
    end

    # Overlap matrix: O_ij = sum_n W_ni (N_j W_nj)
    O = zeros(Float64, K, K)
    for i in 1:K
        for j in 1:K
            s = 0.0
            Nj = Float64(N_k[j])
            for n in 1:N_tot
                s += W[n, i] * (Nj * W[n, j])
            end
            O[i, j] = s
        end
    end

    # Covariance matrix of f_k
    # Theta_ij = sum_n W_ni W_nj
    Theta = W' * W
    D_mat = Diagonal(Float64.(N_k))
    H_mat = D_mat - D_mat * Theta * D_mat

    cov_f = zeros(Float64, K, K)
    if K > 1
        H_sub = H_mat[2:K, 2:K]
        # Robust pseudo-inverse for covariance
        Sigma_sub = pinv(Matrix(H_sub))
        cov_f[2:K, 2:K] .= Sigma_sub
    end

    delta_f = f[K] - f[1]
    se_delta_f = sqrt(max(0.0, cov_f[K, K]))

    return MBARResult(f, delta_f, cov_f, se_delta_f, O, iter, converged)
end

end # module Alchemical
