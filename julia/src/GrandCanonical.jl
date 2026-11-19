"""
GrandCanonical — Grand Canonical (μVT) Monte Carlo, Farkas AMO Steric Screening,
Widom Test-Particle Insertion, Balanced-Ternary Lattice Gas (`TritWord64`), and
Constant-pH Monte Carlo for Biology MM / Trium-XBT.

Implements:
1. `GCMCSpecies` & `thermal_de_broglie`
2. `StericCertificate` & `farkas_steric_screen` (`GSI-R-0013` AMO hard-core screen)
3. `GrandCanonicalSampler` (multi-species μVT with exact detailed balance,
   host-guest framework interactions, `:lfs`/`:trunc`/`:shift`/`:none` LJ)
4. `widom_excess_chemical_potential` (log-sum-exp Widom insertion estimator)
5. `TernaryLatticeGas` (64-site bit-sliced `TriumXBT.TritWord64` periodic lattice GCMC)
6. `TitrationSite` & `ConstantPHSampler` (Henderson–Hasselbalch + electrostatic ΔU)
"""
module GrandCanonical

using LinearAlgebra
using Random
using Statistics
using ..Types
using ..Factors
using ..TriumXBT

export GCMCSpecies, thermal_de_broglie,
       StericCertificate, farkas_steric_screen,
       GrandCanonicalSampler,
       gcmc_insert_move!, gcmc_delete_move!, gcmc_displace_move!,
       gcmc_swap_species_move!, run_gcmc!,
       widom_excess_chemical_potential,
       TernaryLatticeGas, count_species_A, count_species_B,
       get_site, set_site!, run_lattice_gcmc!,
       TitrationSite, ConstantPHSampler, titration_mc_step!, run_constant_ph!

const _PLANCK_SI = 6.62607015e-34  # J·s (exact SI 2019 definition)
const _LN10 = log(10.0)

# ==============================================================================
# 1. Species & Thermal de Broglie Wavelength
# ==============================================================================

"""
    GCMCSpecies(name, mass, mu, lambda_de_broglie, sigma, epsilon, charge=0.0)

Thermodynamic and interaction parameters for one guest species in μVT GCMC.
"""
struct GCMCSpecies
    name::Symbol
    mass::Float64
    mu::Float64
    lambda_de_broglie::Float64
    sigma::Float64
    epsilon::Float64
    charge::Float64
    function GCMCSpecies(name::Symbol, mass::Real, mu::Real,
                         lambda_de_broglie::Real, sigma::Real, epsilon::Real,
                         charge::Real = 0.0)
        m = Float64(mass); μ = Float64(mu); Λ = Float64(lambda_de_broglie)
        σ = Float64(sigma); ε = Float64(epsilon); q = Float64(charge)
        (isfinite(m) && m > 0.0) || throw(ArgumentError("mass must be positive and finite"))
        isfinite(μ) || throw(ArgumentError("chemical potential mu must be finite"))
        (isfinite(Λ) && Λ > 0.0) || throw(ArgumentError("lambda_de_broglie must be positive and finite"))
        (isfinite(σ) && σ >= 0.0) || throw(ArgumentError("sigma must be non-negative and finite"))
        (isfinite(ε) && ε >= 0.0) || throw(ArgumentError("epsilon must be non-negative and finite"))
        isfinite(q) || throw(ArgumentError("charge must be finite"))
        new(name, m, μ, Λ, σ, ε, q)
    end
end

"""
    thermal_de_broglie(mass, temperature, units=Types.REDUCED) -> Float64

Thermal de Broglie wavelength `Λ = h / sqrt(2π m k_B T)` in `units.length`.
For `Types.REDUCED` (dimensionless unit scales `length=energy=mass=boltzmann=1`),
uses reduced Planck constant `h = 1.0`, giving `Λ = 1 / sqrt(2π m k_B T)`.
For physical unit systems, converts SI Planck constant `h = 6.62607015e-34 J·s`
into `units.length`.
"""
function thermal_de_broglie(mass::Real, temperature::Real,
                            units::Types.Units = Types.REDUCED)::Float64
    m = Float64(mass)
    T = Float64(temperature)
    (isfinite(m) && m > 0.0) || throw(ArgumentError("mass must be positive and finite"))
    (isfinite(T) && T > 0.0) || throw(ArgumentError("temperature must be positive and finite"))
    kBT = units.boltzmann * T
    if units.length == 1.0 && units.energy == 1.0 && units.mass == 1.0 && units.boltzmann == 1.0
        return 1.0 / sqrt(2.0 * pi * m * kBT)
    end
    m_si = m * units.mass
    kBT_si = kBT * units.energy
    lambda_si = _PLANCK_SI / sqrt(2.0 * pi * m_si * kBT_si)
    return lambda_si / units.length
end

# ==============================================================================
# 2. Periodic Geometry & Minimum-Image Helpers
# ==============================================================================

@inline function _wrap_pos(box::Types.NoBox, pos::NTuple{3,Float64})::NTuple{3,Float64}
    return pos
end

@inline function _wrap_pos(box::Types.OrthorhombicBox, pos::NTuple{3,Float64})::NTuple{3,Float64}
    L = box.L
    return (pos[1] - L[1] * floor(pos[1] / L[1]),
            pos[2] - L[2] * floor(pos[2] / L[2]),
            pos[3] - L[3] * floor(pos[3] / L[3]))
end

@inline function _wrap_pos(box::Types.TriclinicBox, pos::NTuple{3,Float64})::NTuple{3,Float64}
    Hinv = box.Hinv
    H = box.H
    f1 = Hinv[1, 1] * pos[1] + Hinv[1, 2] * pos[2] + Hinv[1, 3] * pos[3]
    f2 = Hinv[2, 1] * pos[1] + Hinv[2, 2] * pos[2] + Hinv[2, 3] * pos[3]
    f3 = Hinv[3, 1] * pos[1] + Hinv[3, 2] * pos[2] + Hinv[3, 3] * pos[3]
    f1 -= floor(f1); f2 -= floor(f2); f3 -= floor(f3)
    return (H[1, 1] * f1 + H[1, 2] * f2 + H[1, 3] * f3,
            H[2, 1] * f1 + H[2, 2] * f2 + H[2, 3] * f3,
            H[3, 1] * f1 + H[3, 2] * f2 + H[3, 3] * f3)
end

@inline function _sample_uniform_box(box::Types.OrthorhombicBox, rng::AbstractRNG)::NTuple{3,Float64}
    L = box.L
    return (rand(rng) * L[1], rand(rng) * L[2], rand(rng) * L[3])
end

@inline function _sample_uniform_box(box::Types.TriclinicBox, rng::AbstractRNG)::NTuple{3,Float64}
    f1 = rand(rng); f2 = rand(rng); f3 = rand(rng)
    H = box.H
    return (H[1, 1] * f1 + H[1, 2] * f2 + H[1, 3] * f3,
            H[2, 1] * f1 + H[2, 2] * f2 + H[2, 3] * f3,
            H[3, 1] * f1 + H[3, 2] * f2 + H[3, 3] * f3)
end

function _sample_uniform_box(box::Types.NoBox, ::AbstractRNG)
    throw(Types.UnsupportedGeometry("GrandCanonical requires a periodic box (OrthorhombicBox or TriclinicBox)"))
end

@inline function _mic_dist_sq(box::Types.NoBox, pos::NTuple{3,Float64},
                              q::AbstractMatrix{Float64}, j::Int)::Float64
    @inbounds begin
        dx = pos[1] - q[1, j]
        dy = pos[2] - q[2, j]
        dz = pos[3] - q[3, j]
    end
    return dx * dx + dy * dy + dz * dz
end

@inline function _mic_dist_sq(box::Types.OrthorhombicBox, pos::NTuple{3,Float64},
                              q::AbstractMatrix{Float64}, j::Int)::Float64
    L = box.L
    @inbounds begin
        dx = pos[1] - q[1, j]
        dy = pos[2] - q[2, j]
        dz = pos[3] - q[3, j]
    end
    dx -= L[1] * round(dx / L[1])
    dy -= L[2] * round(dy / L[2])
    dz -= L[3] * round(dz / L[3])
    return dx * dx + dy * dy + dz * dz
end

@inline function _mic_dist_sq(box::Types.TriclinicBox, pos::NTuple{3,Float64},
                              q::AbstractMatrix{Float64}, j::Int)::Float64
    @inbounds begin
        dx = pos[1] - q[1, j]
        dy = pos[2] - q[2, j]
        dz = pos[3] - q[3, j]
    end
    Hinv = box.Hinv
    H = box.H
    f1 = Hinv[1, 1] * dx + Hinv[1, 2] * dy + Hinv[1, 3] * dz
    f2 = Hinv[2, 1] * dx + Hinv[2, 2] * dy + Hinv[2, 3] * dz
    f3 = Hinv[3, 1] * dx + Hinv[3, 2] * dy + Hinv[3, 3] * dz
    n01 = round(Int, f1); n02 = round(Int, f2); n03 = round(Int, f3)
    best_x = dx - (H[1, 1] * n01 + H[1, 2] * n02 + H[1, 3] * n03)
    best_y = dy - (H[2, 1] * n01 + H[2, 2] * n02 + H[2, 3] * n03)
    best_z = dz - (H[3, 1] * n01 + H[3, 2] * n02 + H[3, 3] * n03)
    best2 = best_x * best_x + best_y * best_y + best_z * best_z
    b0 = sqrt(best2)
    b1 = sqrt(Hinv[1, 1]^2 + Hinv[1, 2]^2 + Hinv[1, 3]^2) * b0
    b2 = sqrt(Hinv[2, 1]^2 + Hinv[2, 2]^2 + Hinv[2, 3]^2) * b0
    b3 = sqrt(Hinv[3, 1]^2 + Hinv[3, 2]^2 + Hinv[3, 3]^2) * b0
    for c in ceil(Int, f3 - b3):floor(Int, f3 + b3)
        for b in ceil(Int, f2 - b2):floor(Int, f2 + b2)
            for a in ceil(Int, f1 - b1):floor(Int, f1 + b1)
                rx = dx - (H[1, 1] * a + H[1, 2] * b + H[1, 3] * c)
                ry = dy - (H[2, 1] * a + H[2, 2] * b + H[2, 3] * c)
                rz = dz - (H[3, 1] * a + H[3, 2] * b + H[3, 3] * c)
                r2 = rx * rx + ry * ry + rz * rz
                if r2 < best2
                    best2 = r2
                end
            end
        end
    end
    return best2
end

# ==============================================================================
# 3. Farkas AMO Steric Clash Certificate (`GSI-R-0013`)
# ==============================================================================

"""
    StericCertificate(feasible, clash_atom, reason, min_dist_sq)

Certificate produced by `farkas_steric_screen`.
- `feasible::Bool`: `true` iff no active particle violates `r < r_min`.
- `clash_atom::Int`: index of first clashing atom (`0` when feasible).
- `reason::Symbol`: `:feasible`, `:cell_clash`, or `:steric_clash`.
- `min_dist_sq::Float64`: minimum squared distance found (or clashing `d^2` on early exit).
"""
struct StericCertificate
    feasible::Bool
    clash_atom::Int
    reason::Symbol
    min_dist_sq::Float64
end

@inline function _orthorhombic_cell_coords(pos::NTuple{3,Float64}, L::NTuple{3,Float64},
                                           n_cells::NTuple{3,Int64})::NTuple{3,Int64}
    cx = mod(floor(Int64, (pos[1] / L[1]) * n_cells[1]), n_cells[1])
    cy = mod(floor(Int64, (pos[2] / L[2]) * n_cells[2]), n_cells[2])
    cz = mod(floor(Int64, (pos[3] / L[3]) * n_cells[3]), n_cells[3])
    return (cx, cy, cz)
end

"""
    farkas_steric_screen(pos::NTuple{3,Float64}, q_existing::AbstractMatrix{Float64},
                         box::Types.AbstractBox, r_min::Float64;
                         active_mask=nothing) -> StericCertificate

Exact minimum-image / integer-grid AMO steric clash pre-screening certificate.
Subdivides an orthorhombic box into integer cells of width strictly smaller than
`r_min / sqrt(3)` so same-cell occupancy is an O(1) integer certificate of `r < r_min`,
and verifies exact minimum-image squared distance `r^2 < r_min^2` across active atoms.
Preserves exact GCMC detailed balance when overlapping hard cores have Boltzmann weight 0.
"""
function farkas_steric_screen(pos::NTuple{3,Float64},
                              q_existing::AbstractMatrix{Float64},
                              box::Types.AbstractBox,
                              r_min::Float64;
                              active_mask::Union{Nothing,AbstractVector{Bool}} = nothing)::StericCertificate
    size(q_existing, 1) == 3 || throw(DimensionMismatch("q_existing must be 3 x N"))
    N = size(q_existing, 2)
    if active_mask !== nothing && length(active_mask) != N
        throw(DimensionMismatch("active_mask length $(length(active_mask)) != N=$N"))
    end
    if r_min <= 0.0 || N == 0
        return StericCertificate(true, 0, :feasible, Inf)
    end
    r_min_sq = r_min * r_min
    min_d2 = Inf

    if box isa Types.OrthorhombicBox
        L = box.L
        # Sub-cell diagonal < r_min strictly when L_d / n_d < r_min / sqrt(3)
        sqrt3_over_r = sqrt(3.0) / r_min
        nx = max(Int64(1), floor(Int64, L[1] * sqrt3_over_r) + Int64(1))
        ny = max(Int64(1), floor(Int64, L[2] * sqrt3_over_r) + Int64(1))
        nz = max(Int64(1), floor(Int64, L[3] * sqrt3_over_r) + Int64(1))
        n_cells = (nx, ny, nz)
        c_pos = _orthorhombic_cell_coords(pos, L, n_cells)

        @inbounds for j in 1:N
            active_mask !== nothing && !active_mask[j] && continue
            pj = (q_existing[1, j], q_existing[2, j], q_existing[3, j])
            c_j = _orthorhombic_cell_coords(pj, L, n_cells)
            d2 = _mic_dist_sq(box, pos, q_existing, j)
            if c_j == c_pos && d2 < r_min_sq
                return StericCertificate(false, j, :cell_clash, d2)
            elseif d2 < r_min_sq
                return StericCertificate(false, j, :steric_clash, d2)
            end
            if d2 < min_d2
                min_d2 = d2
            end
        end
    else
        @inbounds for j in 1:N
            active_mask !== nothing && !active_mask[j] && continue
            d2 = _mic_dist_sq(box, pos, q_existing, j)
            if d2 < r_min_sq
                return StericCertificate(false, j, :steric_clash, d2)
            end
            if d2 < min_d2
                min_d2 = d2
            end
        end
    end

    return StericCertificate(true, 0, :feasible, min_d2)
end

# ==============================================================================
# 4. Pairwise LJ Potential Helpers (Lorentz-Berthelot + Cutoff Modes)
# ==============================================================================

@inline function _lj_pair_energy(r2::Float64, eps_ij::Float64, sig_ij::Float64,
                                 rc::Float64, mode::Symbol)::Float64
    eps_ij == 0.0 && return 0.0
    sig_ij == 0.0 && return 0.0
    r2 < 1e-24 && return Inf
    r = sqrt(r2)
    if mode !== :none && r > rc
        return 0.0
    end
    u = Factors.lj_energy(eps_ij, sig_ij, r)
    if mode === :lfs
        uc = Factors.lj_energy(eps_ij, sig_ij, rc)
        duc = Factors.lj_dudr(eps_ij, sig_ij, rc)
        u = u - uc - duc * (r - rc)
    elseif mode === :shift
        u = u - Factors.lj_energy(eps_ij, sig_ij, rc)
    elseif mode !== :trunc && mode !== :none
        throw(ArgumentError("unknown LJ mode :$mode; expected :lfs, :trunc, :shift, or :none"))
    end
    return u
end

# ==============================================================================
# 5. GrandCanonicalSampler (μVT Monte Carlo)
# ==============================================================================

"""
    GrandCanonicalSampler

Multi-species μVT Grand Canonical Monte Carlo sampler with host framework atoms,
Farkas steric clash screening, and exact multi-species detailed balance.
"""
mutable struct GrandCanonicalSampler
    species::Vector{GCMCSpecies}
    box::Types.AbstractBox
    temperature::Float64
    units::Types.Units
    rc::Float64
    mode::Symbol
    hard_core_frac::Float64
    q_host::Matrix{Float64}
    sigma_host::Vector{Float64}
    epsilon_host::Vector{Float64}
    q_guest::Matrix{Float64}
    guest_species::Vector{Int}
    species_counts::Vector{Int}
    pair_energy_fn::Union{Nothing,Function}
    current_energy::Float64
    insert_attempts::Int
    insert_accepts::Int
    delete_attempts::Int
    delete_accepts::Int
    displace_attempts::Int
    displace_accepts::Int
    swap_attempts::Int
    swap_accepts::Int
    steric_rejections::Int
end

function GrandCanonicalSampler(species::AbstractVector{GCMCSpecies},
                               box::Types.AbstractBox,
                               temperature::Real;
                               units::Types.Units = Types.REDUCED,
                               rc::Real = 2.5,
                               mode::Symbol = :lfs,
                               hard_core_frac::Real = 0.5,
                               q_host::AbstractMatrix{<:Real} = zeros(Float64, 3, 0),
                               sigma_host::AbstractVector{<:Real} = Float64[],
                               epsilon_host::AbstractVector{<:Real} = Float64[],
                               q_guest::AbstractMatrix{<:Real} = zeros(Float64, 3, 0),
                               guest_species::AbstractVector{<:Integer} = Int[],
                               pair_energy_fn::Union{Nothing,Function} = nothing)
    isempty(species) && throw(ArgumentError("species vector must be non-empty"))
    box isa Types.NoBox && throw(Types.UnsupportedGeometry("GrandCanonicalSampler requires a periodic box"))
    T = Float64(temperature)
    (isfinite(T) && T > 0.0) || throw(ArgumentError("temperature must be positive and finite"))
    rc_f = Float64(rc)
    mode in (:lfs, :trunc, :shift, :none) || throw(ArgumentError("unknown LJ mode :$mode"))
    (mode === :none || (isfinite(rc_f) && rc_f > 0.0)) || throw(ArgumentError("rc must be positive"))
    hcf = Float64(hard_core_frac)
    (isfinite(hcf) && hcf >= 0.0) || throw(ArgumentError("hard_core_frac must be non-negative"))

    qh = Matrix{Float64}(q_host)
    size(qh, 1) == 3 || throw(DimensionMismatch("q_host must be 3 x N_host"))
    Nh = size(qh, 2)
    sh = isempty(sigma_host) && Nh > 0 ? zeros(Float64, Nh) : Vector{Float64}(sigma_host)
    eh = isempty(epsilon_host) && Nh > 0 ? zeros(Float64, Nh) : Vector{Float64}(epsilon_host)
    length(sh) == Nh || throw(DimensionMismatch("sigma_host length != N_host=$Nh"))
    length(eh) == Nh || throw(DimensionMismatch("epsilon_host length != N_host=$Nh"))

    qg = Matrix{Float64}(q_guest)
    size(qg, 1) == 3 || throw(DimensionMismatch("q_guest must be 3 x N_guest"))
    Ng = size(qg, 2)
    gs = Vector{Int}(guest_species)
    length(gs) == Ng || throw(DimensionMismatch("guest_species length != N_guest=$Ng"))
    S = length(species)
    sc = zeros(Int, S)
    for sp_id in gs
        1 <= sp_id <= S || throw(BoundsError("species index $sp_id out of 1:$S"))
        sc[sp_id] += 1
    end

    sampler = GrandCanonicalSampler(
        Vector{GCMCSpecies}(species), box, T, units, rc_f, mode, hcf,
        qh, sh, eh, qg, gs, sc, pair_energy_fn, 0.0,
        0, 0, 0, 0, 0, 0, 0, 0, 0
    )
    sampler.current_energy = _total_guest_energy(sampler)
    return sampler
end

function GrandCanonicalSampler(sp::GCMCSpecies, box::Types.AbstractBox, temperature::Real; kwargs...)
    return GrandCanonicalSampler([sp], box, temperature; kwargs...)
end

function _particle_energy(sampler::GrandCanonicalSampler, pos::NTuple{3,Float64},
                          sp_id::Int; skip_guest::Int = 0)::Float64
    sp = sampler.species[sp_id]
    rc = sampler.rc
    mode = sampler.mode
    box = sampler.box
    u = 0.0

    # Host-guest LJ
    if sp.epsilon > 0.0
        qh = sampler.q_host
        sh = sampler.sigma_host
        eh = sampler.epsilon_host
        @inbounds for h in axes(qh, 2)
            eh[h] == 0.0 && continue
            eps_ij = sqrt(sp.epsilon * eh[h])
            sig_ij = 0.5 * (sp.sigma + sh[h])
            r2 = _mic_dist_sq(box, pos, qh, h)
            du = _lj_pair_energy(r2, eps_ij, sig_ij, rc, mode)
            !isfinite(du) && return Inf
            u += du
        end

        # Guest-guest LJ
        qg = sampler.q_guest
        gs = sampler.guest_species
        spp = sampler.species
        @inbounds for j in axes(qg, 2)
            j == skip_guest && continue
            spj = spp[gs[j]]
            spj.epsilon == 0.0 && continue
            eps_ij = sqrt(sp.epsilon * spj.epsilon)
            sig_ij = 0.5 * (sp.sigma + spj.sigma)
            r2 = _mic_dist_sq(box, pos, qg, j)
            du = _lj_pair_energy(r2, eps_ij, sig_ij, rc, mode)
            !isfinite(du) && return Inf
            u += du
        end
    end

    if sampler.pair_energy_fn !== nothing
        u += Float64(sampler.pair_energy_fn(pos, sp_id, sampler.q_host, box))
    end
    return u
end

function _total_guest_energy(sampler::GrandCanonicalSampler)::Float64
    qg = sampler.q_guest
    gs = sampler.guest_species
    Ng = size(qg, 2)
    Ng == 0 && return 0.0
    u = 0.0
    rc = sampler.rc
    mode = sampler.mode
    box = sampler.box
    qh = sampler.q_host
    sh = sampler.sigma_host
    eh = sampler.epsilon_host
    spp = sampler.species

    @inbounds for i in 1:Ng
        pos = (qg[1, i], qg[2, i], qg[3, i])
        sp_i = spp[gs[i]]
        if sp_i.epsilon > 0.0
            for h in axes(qh, 2)
                eh[h] == 0.0 && continue
                eps_ij = sqrt(sp_i.epsilon * eh[h])
                sig_ij = 0.5 * (sp_i.sigma + sh[h])
                r2 = _mic_dist_sq(box, pos, qh, h)
                u += _lj_pair_energy(r2, eps_ij, sig_ij, rc, mode)
            end
            for j in (i + 1):Ng
                sp_j = spp[gs[j]]
                sp_j.epsilon == 0.0 && continue
                eps_ij = sqrt(sp_i.epsilon * sp_j.epsilon)
                sig_ij = 0.5 * (sp_i.sigma + sp_j.sigma)
                r2 = _mic_dist_sq(box, pos, qg, j)
                u += _lj_pair_energy(r2, eps_ij, sig_ij, rc, mode)
            end
        end
        if sampler.pair_energy_fn !== nothing
            u += Float64(sampler.pair_energy_fn(pos, gs[i], qh, box))
        end
    end
    return u
end

@inline function _steric_screen_insert(sampler::GrandCanonicalSampler, pos::NTuple{3,Float64},
                                       sp::GCMCSpecies; skip_guest::Int = 0)::Bool
    (sampler.hard_core_frac <= 0.0 || sp.epsilon <= 0.0 || sp.sigma <= 0.0) && return true
    r_min = sampler.hard_core_frac * sp.sigma
    if !isempty(sampler.q_host)
        cert_h = farkas_steric_screen(pos, sampler.q_host, sampler.box, r_min)
        if !cert_h.feasible
            sampler.steric_rejections += 1
            return false
        end
    end
    Ng = size(sampler.q_guest, 2)
    if Ng > 0
        mask = skip_guest == 0 ? nothing : Bool[j != skip_guest for j in 1:Ng]
        cert_g = farkas_steric_screen(pos, sampler.q_guest, sampler.box, r_min; active_mask = mask)
        if !cert_g.feasible
            sampler.steric_rejections += 1
            return false
        end
    end
    return true
end

"""
    gcmc_insert_move!(sampler, sp_id, rng; proposal_ratio=1.0) -> Bool

Propose uniform particle insertion of species `sp_id`, pre-screen via
`farkas_steric_screen`, and accept/reject by exact μVT Metropolis rule.
"""
function gcmc_insert_move!(sampler::GrandCanonicalSampler, sp_id::Int,
                           rng::AbstractRNG; proposal_ratio::Float64 = 1.0)::Bool
    1 <= sp_id <= length(sampler.species) || throw(BoundsError("species $sp_id out of range"))
    sampler.insert_attempts += 1
    sp = sampler.species[sp_id]
    pos = _sample_uniform_box(sampler.box, rng)

    if !_steric_screen_insert(sampler, pos, sp)
        return false
    end

    dU = _particle_energy(sampler, pos, sp_id)
    !isfinite(dU) && return false

    V = Types.box_volume(sampler.box)
    beta = 1.0 / (sampler.units.boltzmann * sampler.temperature)
    N_sp = sampler.species_counts[sp_id]
    Λ3 = sp.lambda_de_broglie^3
    log_acc = log(proposal_ratio * V / (Λ3 * (N_sp + 1))) - beta * (dU - sp.mu)

    if log_acc >= 0.0 || log(rand(rng)) < log_acc
        Ng = size(sampler.q_guest, 2)
        new_q = Matrix{Float64}(undef, 3, Ng + 1)
        if Ng > 0
            copyto!(view(new_q, :, 1:Ng), sampler.q_guest)
        end
        @inbounds begin
            new_q[1, Ng + 1] = pos[1]
            new_q[2, Ng + 1] = pos[2]
            new_q[3, Ng + 1] = pos[3]
        end
        sampler.q_guest = new_q
        push!(sampler.guest_species, sp_id)
        sampler.species_counts[sp_id] += 1
        sampler.current_energy += dU
        sampler.insert_accepts += 1
        return true
    end
    return false
end

function gcmc_insert_move!(sampler::GrandCanonicalSampler, rng::AbstractRNG;
                           proposal_ratio::Float64 = 1.0)::Bool
    sp_id = rand(rng, 1:length(sampler.species))
    return gcmc_insert_move!(sampler, sp_id, rng; proposal_ratio = proposal_ratio)
end

"""
    gcmc_delete_move!(sampler, rng; proposal_ratio=1.0) -> Bool

Propose uniform species selection `sp_id ~ Uniform(1:S)` (matching insertion
species selection for exact multi-species detailed balance), then uniform particle
deletion within species `sp_id`.
"""
function gcmc_delete_move!(sampler::GrandCanonicalSampler, sp_id::Int,
                           rng::AbstractRNG; proposal_ratio::Float64 = 1.0)::Bool
    1 <= sp_id <= length(sampler.species) || throw(BoundsError("species $sp_id out of range"))
    sampler.delete_attempts += 1
    N_sp = sampler.species_counts[sp_id]
    N_sp == 0 && return false

    # Pick uniformly among particles of species sp_id
    target_rank = rand(rng, 1:N_sp)
    idx = 0
    seen = 0
    @inbounds for j in eachindex(sampler.guest_species)
        if sampler.guest_species[j] == sp_id
            seen += 1
            if seen == target_rank
                idx = j
                break
            end
        end
    end

    pos = @inbounds (sampler.q_guest[1, idx], sampler.q_guest[2, idx], sampler.q_guest[3, idx])
    u_old = _particle_energy(sampler, pos, sp_id; skip_guest = idx)
    dU = -u_old
    sp = sampler.species[sp_id]
    V = Types.box_volume(sampler.box)
    beta = 1.0 / (sampler.units.boltzmann * sampler.temperature)
    Λ3 = sp.lambda_de_broglie^3
    log_acc = log(proposal_ratio * (Λ3 * N_sp) / V) - beta * (dU + sp.mu)

    if log_acc >= 0.0 || log(rand(rng)) < log_acc
        Ng = size(sampler.q_guest, 2)
        new_q = Matrix{Float64}(undef, 3, Ng - 1)
        if idx > 1
            copyto!(view(new_q, :, 1:(idx - 1)), view(sampler.q_guest, :, 1:(idx - 1)))
        end
        if idx < Ng
            copyto!(view(new_q, :, idx:(Ng - 1)), view(sampler.q_guest, :, (idx + 1):Ng))
        end
        sampler.q_guest = new_q
        deleteat!(sampler.guest_species, idx)
        sampler.species_counts[sp_id] -= 1
        sampler.current_energy += dU
        sampler.delete_accepts += 1
        return true
    end
    return false
end

function gcmc_delete_move!(sampler::GrandCanonicalSampler, rng::AbstractRNG;
                           proposal_ratio::Float64 = 1.0)::Bool
    sp_id = rand(rng, 1:length(sampler.species))
    return gcmc_delete_move!(sampler, sp_id, rng; proposal_ratio = proposal_ratio)
end

"""
    gcmc_displace_move!(sampler, max_disp, rng) -> Bool

Propose uniform random translation of a uniformly chosen guest particle.
"""
function gcmc_displace_move!(sampler::GrandCanonicalSampler, max_disp::Real,
                             rng::AbstractRNG)::Bool
    sampler.displace_attempts += 1
    Ng = size(sampler.q_guest, 2)
    Ng == 0 && return false
    md = Float64(max_disp)
    idx = rand(rng, 1:Ng)
    sp_id = sampler.guest_species[idx]
    sp = sampler.species[sp_id]

    old_pos = @inbounds (sampler.q_guest[1, idx], sampler.q_guest[2, idx], sampler.q_guest[3, idx])
    trial_raw = (old_pos[1] + (2.0 * rand(rng) - 1.0) * md,
                 old_pos[2] + (2.0 * rand(rng) - 1.0) * md,
                 old_pos[3] + (2.0 * rand(rng) - 1.0) * md)
    new_pos = _wrap_pos(sampler.box, trial_raw)

    if !_steric_screen_insert(sampler, new_pos, sp; skip_guest = idx)
        return false
    end

    u_old = _particle_energy(sampler, old_pos, sp_id; skip_guest = idx)
    u_new = _particle_energy(sampler, new_pos, sp_id; skip_guest = idx)
    !isfinite(u_new) && return false
    dU = u_new - u_old

    beta = 1.0 / (sampler.units.boltzmann * sampler.temperature)
    log_acc = -beta * dU
    if log_acc >= 0.0 || log(rand(rng)) < log_acc
        @inbounds begin
            sampler.q_guest[1, idx] = new_pos[1]
            sampler.q_guest[2, idx] = new_pos[2]
            sampler.q_guest[3, idx] = new_pos[3]
        end
        sampler.current_energy += dU
        sampler.displace_accepts += 1
        return true
    end
    return false
end

"""
    gcmc_swap_species_move!(sampler, rng) -> Bool

Propose identity mutation of a uniformly selected guest particle from species `a`
to a distinct species `b` chosen uniformly from the remaining `S - 1` species.
Because particle selection is uniform over all `N` guests, the combinatorial factor
`N_a / (N_b + 1)` cancels with `q_rev / q_fwd = (N_b + 1) / N_a`, giving exact
acceptance `min(1, (Λ_a / Λ_b)^3 * exp(-β(ΔU - (μ_b - μ_a))))`.
"""
function gcmc_swap_species_move!(sampler::GrandCanonicalSampler, rng::AbstractRNG)::Bool
    sampler.swap_attempts += 1
    S = length(sampler.species)
    Ng = size(sampler.q_guest, 2)
    (S < 2 || Ng == 0) && return false

    idx = rand(rng, 1:Ng)
    a = sampler.guest_species[idx]
    r = rand(rng, 1:(S - 1))
    b = r >= a ? r + 1 : r

    sp_a = sampler.species[a]
    sp_b = sampler.species[b]
    pos = @inbounds (sampler.q_guest[1, idx], sampler.q_guest[2, idx], sampler.q_guest[3, idx])

    if !_steric_screen_insert(sampler, pos, sp_b; skip_guest = idx)
        return false
    end

    u_old = _particle_energy(sampler, pos, a; skip_guest = idx)
    u_new = _particle_energy(sampler, pos, b; skip_guest = idx)
    !isfinite(u_new) && return false
    dU = u_new - u_old

    beta = 1.0 / (sampler.units.boltzmann * sampler.temperature)
    log_acc = 3.0 * log(sp_a.lambda_de_broglie / sp_b.lambda_de_broglie) -
              beta * (dU - (sp_b.mu - sp_a.mu))

    if log_acc >= 0.0 || log(rand(rng)) < log_acc
        sampler.guest_species[idx] = b
        sampler.species_counts[a] -= 1
        sampler.species_counts[b] += 1
        sampler.current_energy += dU
        sampler.swap_accepts += 1
        return true
    end
    return false
end

"""
    run_gcmc!(sampler, n_moves; rng=MersenneTwister(1), weights=(0.35, 0.35, 0.25, 0.05),
              burnin=0, max_disp=0.5)

Execute `n_moves` μVT Monte Carlo steps with move weights `(insert, delete, displace, swap)`.
Returns NamedTuple `(n_history, species_counts, energy_history, acceptance_rates, mean_loading, std_loading)`.
"""
function run_gcmc!(sampler::GrandCanonicalSampler, n_moves::Int;
                   rng::AbstractRNG = MersenneTwister(1),
                   weights::NTuple{4,Float64} = (0.35, 0.35, 0.25, 0.05),
                   burnin::Int = 0,
                   max_disp::Real = 0.5)
    n_moves > 0 || throw(ArgumentError("n_moves must be positive"))
    (0 <= burnin < n_moves) || throw(ArgumentError("burnin must satisfy 0 <= burnin < n_moves"))
    all(w -> isfinite(w) && w >= 0.0, weights) || throw(ArgumentError("weights must be non-negative"))

    w_ins, w_del, w_disp, w_swap = weights
    if length(sampler.species) < 2
        w_swap = 0.0
    end
    w_tot = w_ins + w_del + w_disp + w_swap
    w_tot > 0.0 || throw(ArgumentError("sum of move weights must be positive"))
    p_ins = w_ins / w_tot
    p_del = w_del / w_tot
    p_disp = w_disp / w_tot
    ins_ratio = (w_ins > 0.0 && w_del > 0.0) ? (w_del / w_ins) : 1.0
    del_ratio = (w_ins > 0.0 && w_del > 0.0) ? (w_ins / w_del) : 1.0

    S = length(sampler.species)
    n_keep = n_moves - burnin
    n_history = Vector{Int}(undef, n_keep)
    species_counts = Matrix{Int}(undef, S, n_keep)
    energy_history = Vector{Float64}(undef, n_keep)

    ins0, del0, disp0, swap0 = sampler.insert_attempts, sampler.delete_attempts,
                               sampler.displace_attempts, sampler.swap_attempts
    insa0, dela0, dispa0, swapa0 = sampler.insert_accepts, sampler.delete_accepts,
                                   sampler.displace_accepts, sampler.swap_accepts

    @inbounds for step in 1:n_moves
        u = rand(rng)
        if u < p_ins
            gcmc_insert_move!(sampler, rng; proposal_ratio = ins_ratio)
        elseif u < p_ins + p_del
            gcmc_delete_move!(sampler, rng; proposal_ratio = del_ratio)
        elseif u < p_ins + p_del + p_disp
            gcmc_displace_move!(sampler, max_disp, rng)
        else
            gcmc_swap_species_move!(sampler, rng)
        end

        if step > burnin
            idx = step - burnin
            n_history[idx] = size(sampler.q_guest, 2)
            for s in 1:S
                species_counts[s, idx] = sampler.species_counts[s]
            end
            energy_history[idx] = sampler.current_energy
        end
    end

    d_ins = sampler.insert_attempts - ins0
    d_del = sampler.delete_attempts - del0
    d_disp = sampler.displace_attempts - disp0
    d_swap = sampler.swap_attempts - swap0
    a_ins = sampler.insert_accepts - insa0
    a_del = sampler.delete_accepts - dela0
    a_disp = sampler.displace_accepts - dispa0
    a_swap = sampler.swap_accepts - swapa0
    tot_att = d_ins + d_del + d_disp + d_swap
    tot_acc = a_ins + a_del + a_disp + a_swap

    acceptance_rates = (
        insert = d_ins > 0 ? a_ins / d_ins : 0.0,
        delete = d_del > 0 ? a_del / d_del : 0.0,
        displace = d_disp > 0 ? a_disp / d_disp : 0.0,
        swap = d_swap > 0 ? a_swap / d_swap : 0.0,
        overall = tot_att > 0 ? tot_acc / tot_att : 0.0,
    )

    return (
        n_history = n_history,
        species_counts = species_counts,
        energy_history = energy_history,
        acceptance_rates = acceptance_rates,
        mean_loading = mean(n_history),
        std_loading = length(n_history) > 1 ? std(n_history) : 0.0,
    )
end

# ==============================================================================
# 6. Widom Test-Particle Insertion Estimator
# ==============================================================================

"""
    widom_excess_chemical_potential(q_host, q_guest, sigma_all, epsilon_all,
                                    box, sp, temperature, units=Types.REDUCED;
                                    n_trials=2000, rc=2.5, mode=:lfs,
                                    rng=MersenneTwister(1)) -> (mu_ex, std_err)

Compute excess chemical potential `μ_ex = -k_B T ln ⟨exp(-β ΔU)⟩` via Widom
test-particle insertion using numerically stable log-sum-exp and delta-method
standard error. `sigma_all` and `epsilon_all` correspond to `[q_host q_guest]`.
"""
function widom_excess_chemical_potential(q_host::AbstractMatrix{<:Real},
                                         q_guest::AbstractMatrix{<:Real},
                                         sigma_all::AbstractVector{<:Real},
                                         epsilon_all::AbstractVector{<:Real},
                                         box::Types.AbstractBox,
                                         sp::GCMCSpecies,
                                         temperature::Real,
                                         units::Types.Units = Types.REDUCED;
                                         n_trials::Int = 2000,
                                         rc::Real = 2.5,
                                         mode::Symbol = :lfs,
                                         rng::AbstractRNG = MersenneTwister(1))::Tuple{Float64,Float64}
    box isa Types.NoBox && throw(Types.UnsupportedGeometry("Widom insertion requires a periodic box"))
    n_trials > 1 || throw(ArgumentError("n_trials must be > 1"))
    T = Float64(temperature)
    (isfinite(T) && T > 0.0) || throw(ArgumentError("temperature must be positive"))
    rc_f = Float64(rc)
    qh = Matrix{Float64}(q_host)
    qg = Matrix{Float64}(q_guest)
    Nh = size(qh, 2)
    Ng = size(qg, 2)
    Ntot = Nh + Ng
    q_all = Ntot == 0 ? zeros(Float64, 3, 0) : hcat(qh, qg)
    sig_all = Vector{Float64}(sigma_all)
    eps_all = Vector{Float64}(epsilon_all)
    length(sig_all) == Ntot || throw(DimensionMismatch("sigma_all length $(length(sig_all)) != $Ntot"))
    length(eps_all) == Ntot || throw(DimensionMismatch("epsilon_all length $(length(eps_all)) != $Ntot"))

    beta = 1.0 / (units.boltzmann * T)
    log_w = Vector{Float64}(undef, n_trials)
    w_max = -Inf

    @inbounds for t in 1:n_trials
        pos = _sample_uniform_box(box, rng)
        dU = 0.0
        if sp.epsilon > 0.0
            for j in 1:Ntot
                eps_all[j] == 0.0 && continue
                eps_ij = sqrt(sp.epsilon * eps_all[j])
                sig_ij = 0.5 * (sp.sigma + sig_all[j])
                r2 = _mic_dist_sq(box, pos, q_all, j)
                du = _lj_pair_energy(r2, eps_ij, sig_ij, rc_f, mode)
                if !isfinite(du)
                    dU = Inf
                    break
                end
                dU += du
            end
        end
        wt = -beta * dU
        log_w[t] = wt
        if wt > w_max
            w_max = wt
        end
    end

    if !isfinite(w_max) && w_max < 0.0
        return (Inf, NaN)
    end

    scaled = Vector{Float64}(undef, n_trials)
    @inbounds for t in 1:n_trials
        scaled[t] = exp(log_w[t] - w_max)
    end
    m = mean(scaled)
    s = std(scaled)
    mu_ex = -(w_max + log(m)) / beta
    std_err = (s / sqrt(n_trials)) / (beta * m)
    return (mu_ex, std_err)
end

# ==============================================================================
# 7. TernaryLatticeGas (Built on TriumXBT.TritWord64)
# ==============================================================================

"""
    TernaryLatticeGas

1D/2D/3D periodic ternary lattice gas stored in bit-sliced `Vector{TriumXBT.TritWord64}`
(64 sites per word).
- Trit `+1` (`P` rail) = Species A
- Trit `-1` (`N` rail) = Species B
- Trit `0` = Vacant
Optional `blocked` bitmask locks pore-wall sites to vacant (`0`).
Invariant `(w.P & w.N) == 0` is enforced by `TriumXBT.TritWord64`.
"""
mutable struct TernaryLatticeGas
    dims::NTuple{3,Int}
    n_sites::Int
    words::Vector{TriumXBT.TritWord64}
    blocked::Vector{UInt64}
    J_AA::Float64
    J_BB::Float64
    J_AB::Float64
    eps_A::Vector{Float64}
    eps_B::Vector{Float64}
    mu_A::Float64
    mu_B::Float64
    temperature::Float64
    units::Types.Units
end

function TernaryLatticeGas(dims::NTuple{3,Int};
                           J_AA::Real = 0.0, J_BB::Real = 0.0, J_AB::Real = 0.0,
                           eps_A::Union{Real,AbstractVector{<:Real}} = 0.0,
                           eps_B::Union{Real,AbstractVector{<:Real}} = 0.0,
                           mu_A::Real = 0.0, mu_B::Real = 0.0,
                           temperature::Real = 1.0,
                           units::Types.Units = Types.REDUCED,
                           blocked_sites::AbstractVector{<:Integer} = Int[])
    all(d -> d >= 1, dims) || throw(ArgumentError("lattice dims must be >= 1, got $dims"))
    n_sites = dims[1] * dims[2] * dims[3]
    n_words = cld(n_sites, 64)
    words = [TriumXBT.TritWord64() for _ in 1:n_words]
    blocked = zeros(UInt64, n_words)
    for idx in blocked_sites
        1 <= idx <= n_sites || throw(BoundsError("blocked site $idx out of 1:$n_sites"))
        w_idx = ((idx - 1) >>> 6) + 1
        bit = UInt64(1) << ((idx - 1) & 63)
        blocked[w_idx] |= bit
    end
    eA = eps_A isa Real ? fill(Float64(eps_A), n_sites) : Vector{Float64}(eps_A)
    eB = eps_B isa Real ? fill(Float64(eps_B), n_sites) : Vector{Float64}(eps_B)
    length(eA) == n_sites || throw(DimensionMismatch("eps_A length != $n_sites"))
    length(eB) == n_sites || throw(DimensionMismatch("eps_B length != $n_sites"))
    T = Float64(temperature)
    (isfinite(T) && T > 0.0) || throw(ArgumentError("temperature must be positive"))
    return TernaryLatticeGas(dims, n_sites, words, blocked,
                             Float64(J_AA), Float64(J_BB), Float64(J_AB),
                             eA, eB, Float64(mu_A), Float64(mu_B), T, units)
end

TernaryLatticeGas(L::Int; kwargs...) = TernaryLatticeGas((L, 1, 1); kwargs...)
TernaryLatticeGas(Lx::Int, Ly::Int; kwargs...) = TernaryLatticeGas((Lx, Ly, 1); kwargs...)
TernaryLatticeGas(Lx::Int, Ly::Int, Lz::Int; kwargs...) = TernaryLatticeGas((Lx, Ly, Lz); kwargs...)

@inline count_species_A(lg::TernaryLatticeGas)::Int =
    sum(w -> Int(count_ones(w.P)), lg.words)

@inline count_species_B(lg::TernaryLatticeGas)::Int =
    sum(w -> Int(count_ones(w.N)), lg.words)

@inline function is_blocked(lg::TernaryLatticeGas, idx::Int)::Bool
    w_idx = ((idx - 1) >>> 6) + 1
    bit = UInt64(1) << ((idx - 1) & 63)
    return (lg.blocked[w_idx] & bit) != UInt64(0)
end

@inline function get_site(lg::TernaryLatticeGas, idx::Int)::Int8
    @boundscheck (1 <= idx <= lg.n_sites || throw(BoundsError("site $idx")))
    w_idx = ((idx - 1) >>> 6) + 1
    shift = (idx - 1) & 63
    w = @inbounds lg.words[w_idx]
    p = (w.P >> shift) & UInt64(1)
    n = (w.N >> shift) & UInt64(1)
    return Int8(p) - Int8(n)
end

@inline function set_site!(lg::TernaryLatticeGas, idx::Int, val::Int8)
    @boundscheck (1 <= idx <= lg.n_sites || throw(BoundsError("site $idx")))
    w_idx = ((idx - 1) >>> 6) + 1
    bit = UInt64(1) << ((idx - 1) & 63)
    w = @inbounds lg.words[w_idx]
    P = w.P & ~bit
    N = w.N & ~bit
    if val == Int8(1)
        P |= bit
    elseif val == Int8(-1)
        N |= bit
    elseif val != Int8(0)
        throw(ArgumentError("lattice site value must be -1, 0, or +1, got $val"))
    end
    @inbounds lg.words[w_idx] = TriumXBT.TritWord64(P, N)
    return val
end

@inline function _lattice_neighbors(dims::NTuple{3,Int}, idx::Int)::NTuple{6,Int}
    Lx, Ly, Lz = dims
    z0 = div(idx - 1, Lx * Ly)
    rem = (idx - 1) - z0 * Lx * Ly
    y0 = div(rem, Lx)
    x0 = rem - y0 * Lx
    @inline lin(x, y, z) = 1 + x + Lx * (y + Ly * z)
    n1 = Lx > 1 ? lin(mod(x0 + 1, Lx), y0, z0) : 0
    n2 = Lx > 1 ? lin(mod(x0 - 1, Lx), y0, z0) : 0
    n3 = Ly > 1 ? lin(x0, mod(y0 + 1, Ly), z0) : 0
    n4 = Ly > 1 ? lin(x0, mod(y0 - 1, Ly), z0) : 0
    n5 = Lz > 1 ? lin(x0, y0, mod(z0 + 1, Lz)) : 0
    n6 = Lz > 1 ? lin(x0, y0, mod(z0 - 1, Lz)) : 0
    return (n1, n2, n3, n4, n5, n6)
end

@inline function _site_grand_energy(lg::TernaryLatticeGas, idx::Int, s::Int8)::Float64
    s == Int8(0) && return 0.0
    @inbounds base = s == Int8(1) ? (lg.eps_A[idx] - lg.mu_A) : (lg.eps_B[idx] - lg.mu_B)
    nn_e = 0.0
    if lg.J_AA != 0.0 || lg.J_BB != 0.0 || lg.J_AB != 0.0
        nbrs = _lattice_neighbors(lg.dims, idx)
        for nb in nbrs
            nb == 0 && continue
            s_nb = get_site(lg, nb)
            s_nb == Int8(0) && continue
            if s == Int8(1)
                nn_e += (s_nb == Int8(1) ? lg.J_AA : lg.J_AB)
            else
                nn_e += (s_nb == Int8(-1) ? lg.J_BB : lg.J_AB)
            end
        end
    end
    return base + nn_e
end

function _total_lattice_grand_energy(lg::TernaryLatticeGas)::Float64
    e = 0.0
    @inbounds for idx in 1:lg.n_sites
        s = get_site(lg, idx)
        s == Int8(0) && continue
        e += s == Int8(1) ? (lg.eps_A[idx] - lg.mu_A) : (lg.eps_B[idx] - lg.mu_B)
        if lg.J_AA != 0.0 || lg.J_BB != 0.0 || lg.J_AB != 0.0
            nbrs = _lattice_neighbors(lg.dims, idx)
            for nb in nbrs
                nb <= idx && continue
                s_nb = get_site(lg, nb)
                s_nb == Int8(0) && continue
                if s == Int8(1)
                    e += (s_nb == Int8(1) ? lg.J_AA : lg.J_AB)
                else
                    e += (s_nb == Int8(-1) ? lg.J_BB : lg.J_AB)
                end
            end
        end
    end
    return e
end

"""
    run_lattice_gcmc!(lg::TernaryLatticeGas, n_sweeps::Int; burnin=0, rng=MersenneTwister(1))

Perform `n_sweeps` grand-canonical sweeps (`n_sites` single-site symmetric
Metropolis proposals per sweep) on `TernaryLatticeGas`.
Returns `(theta_A, theta_B, theta_vacant, selectivity, loading_ratio, theta_A_history, theta_B_history, energy_history)`.
"""
function run_lattice_gcmc!(lg::TernaryLatticeGas, n_sweeps::Int;
                           burnin::Int = 0,
                           rng::AbstractRNG = MersenneTwister(1))
    n_sweeps > 0 || throw(ArgumentError("n_sweeps must be positive"))
    (0 <= burnin < n_sweeps) || throw(ArgumentError("burnin must satisfy 0 <= burnin < n_sweeps"))

    n_sites = lg.n_sites
    n_blocked = sum(w -> Int(count_ones(w)), lg.blocked)
    n_active = n_sites - n_blocked
    n_active > 0 || throw(ArgumentError("lattice has no unblocked sites"))

    beta = 1.0 / (lg.units.boltzmann * lg.temperature)
    n_keep = n_sweeps - burnin
    tA_hist = Vector{Float64}(undef, n_keep)
    tB_hist = Vector{Float64}(undef, n_keep)
    e_hist = Vector{Float64}(undef, n_keep)

    @inbounds for sweep in 1:n_sweeps
        for _ in 1:n_sites
            idx = rand(rng, 1:n_sites)
            is_blocked(lg, idx) && continue
            s_old = get_site(lg, idx)
            # Symmetric proposal over the other two states in {-1, 0, +1}
            pick_first = rand(rng, Bool)
            s_new = if s_old == Int8(0)
                pick_first ? Int8(1) : Int8(-1)
            elseif s_old == Int8(1)
                pick_first ? Int8(0) : Int8(-1)
            else
                pick_first ? Int8(0) : Int8(1)
            end
            dOmega = _site_grand_energy(lg, idx, s_new) - _site_grand_energy(lg, idx, s_old)
            log_acc = -beta * dOmega
            if log_acc >= 0.0 || log(rand(rng)) < log_acc
                set_site!(lg, idx, s_new)
            end
        end

        if sweep > burnin
            k = sweep - burnin
            nA = count_species_A(lg)
            nB = count_species_B(lg)
            tA_hist[k] = nA / n_active
            tB_hist[k] = nB / n_active
            e_hist[k] = _total_lattice_grand_energy(lg)
        end
    end

    theta_A = mean(tA_hist)
    theta_B = mean(tB_hist)
    theta_vac = 1.0 - theta_A - theta_B
    loading_ratio = theta_A / max(theta_B, 1e-12)
    selectivity = loading_ratio * exp(-beta * (lg.mu_A - lg.mu_B))
    return (
        theta_A = theta_A,
        theta_B = theta_B,
        theta_vacant = theta_vac,
        selectivity = selectivity,
        loading_ratio = loading_ratio,
        theta_A_history = tA_hist,
        theta_B_history = tB_hist,
        energy_history = e_hist,
    )
end

# ==============================================================================
# 8. ConstantPHSampler
# ==============================================================================

"""
    TitrationSite(atom_indices, pKa, q_prot, q_deprot, protonated=true)

Single titratable residue/site with protonated and deprotonated partial charge arrays.
"""
mutable struct TitrationSite
    atom_indices::Vector{Int}
    pKa::Float64
    q_prot::Vector{Float64}
    q_deprot::Vector{Float64}
    protonated::Bool
    function TitrationSite(atom_indices::AbstractVector{<:Integer}, pKa::Real,
                           q_prot::AbstractVector{<:Real}, q_deprot::AbstractVector{<:Real},
                           protonated::Bool = true)
        idxs = Vector{Int}(atom_indices)
        qp = Vector{Float64}(q_prot)
        qd = Vector{Float64}(q_deprot)
        length(idxs) == length(qp) == length(qd) ||
            throw(DimensionMismatch("atom_indices, q_prot, and q_deprot must have equal length"))
        new(idxs, Float64(pKa), qp, qd, protonated)
    end
end

"""
    ConstantPHSampler(sites, pH, temperature, units=Types.REDUCED; n_atoms=...)

Discrete-state constant-pH Monte Carlo sampler obeying exact Henderson–Hasselbalch
thermodynamics coupled with electrostatic energy evaluation `energy_fn(charges)`.
"""
mutable struct ConstantPHSampler
    sites::Vector{TitrationSite}
    pH::Float64
    temperature::Float64
    units::Types.Units
    charges::Vector{Float64}
    attempts::Int
    accepts::Int
end

function ConstantPHSampler(sites::AbstractVector{TitrationSite}, pH::Real,
                           temperature::Real, units::Types.Units = Types.REDUCED;
                           base_charges::Union{Nothing,AbstractVector{<:Real}} = nothing)
    isempty(sites) && throw(ArgumentError("sites must be non-empty"))
    T = Float64(temperature)
    (isfinite(T) && T > 0.0) || throw(ArgumentError("temperature must be positive"))
    max_idx = maximum(maximum(s.atom_indices) for s in sites)
    charges = base_charges === nothing ? zeros(Float64, max_idx) : Vector{Float64}(base_charges)
    length(charges) >= max_idx || throw(DimensionMismatch("base_charges too short for site indices"))
    for s in sites
        q_cur = s.protonated ? s.q_prot : s.q_deprot
        @inbounds for (k, aidx) in enumerate(s.atom_indices)
            charges[aidx] = q_cur[k]
        end
    end
    return ConstantPHSampler(Vector{TitrationSite}(sites), Float64(pH), T, units, charges, 0, 0)
end

"""
    titration_mc_step!(sampler, energy_fn, rng) -> Bool

Propose protonation/deprotonation toggle of a random `TitrationSite`.
Uses exact ΔG = s (pKa - pH) k_B T ln(10) + ΔU_elec, where s = +1 for
deprotonation (prot -> deprot) and s = -1 for protonation (deprot -> prot).
"""
function titration_mc_step!(sampler::ConstantPHSampler, energy_fn, rng::AbstractRNG)::Bool
    sampler.attempts += 1
    k = rand(rng, 1:length(sampler.sites))
    site = sampler.sites[k]
    was_prot = site.protonated
    s = was_prot ? 1.0 : -1.0  # +1 deprotonate, -1 protonate

    u_old = energy_fn === nothing ? 0.0 : Float64(energy_fn(sampler.charges))
    q_target = was_prot ? site.q_deprot : site.q_prot
    q_restore = was_prot ? site.q_prot : site.q_deprot

    @inbounds for (j, aidx) in enumerate(site.atom_indices)
        sampler.charges[aidx] = q_target[j]
    end
    u_new = energy_fn === nothing ? 0.0 : Float64(energy_fn(sampler.charges))
    dU_elec = u_new - u_old

    kBT = sampler.units.boltzmann * sampler.temperature
    beta = 1.0 / kBT
    # P(deprot)/P(prot) = 10^(pH - pKa) exp(-β ΔU_elec)
    dG = s * (site.pKa - sampler.pH) * kBT * _LN10 + dU_elec
    log_acc = -beta * dG

    if isfinite(log_acc) && (log_acc >= 0.0 || log(rand(rng)) < log_acc)
        site.protonated = !was_prot
        sampler.accepts += 1
        return true
    else
        @inbounds for (j, aidx) in enumerate(site.atom_indices)
            sampler.charges[aidx] = q_restore[j]
        end
        return false
    end
end

"""
    run_constant_ph!(sampler, energy_fn, n_steps; burnin=0, rng=MersenneTwister(1))

Run `n_steps` titration Monte Carlo proposals and return
`(protonation_fractions, site_histories, acceptance_rate)`.
"""
function run_constant_ph!(sampler::ConstantPHSampler, energy_fn, n_steps::Int;
                          burnin::Int = 0, rng::AbstractRNG = MersenneTwister(1))
    n_steps > 0 || throw(ArgumentError("n_steps must be positive"))
    (0 <= burnin < n_steps) || throw(ArgumentError("burnin must satisfy 0 <= burnin < n_steps"))
    ns = length(sampler.sites)
    n_keep = n_steps - burnin
    site_histories = Matrix{Bool}(undef, ns, n_keep)
    att0 = sampler.attempts
    acc0 = sampler.accepts

    @inbounds for step in 1:n_steps
        titration_mc_step!(sampler, energy_fn, rng)
        if step > burnin
            col = step - burnin
            for j in 1:ns
                site_histories[j, col] = sampler.sites[j].protonated
            end
        end
    end

    fracs = [mean(@view site_histories[j, :]) for j in 1:ns]
    d_att = sampler.attempts - att0
    d_acc = sampler.accepts - acc0
    return (
        protonation_fractions = fracs,
        site_histories = site_histories,
        acceptance_rate = d_att > 0 ? d_acc / d_att : 0.0,
    )
end

end # module GrandCanonical
