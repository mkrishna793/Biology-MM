# 3-Zone Multiscale & Clash-Relaxation module for FoundationCore.
#
# Provides:
#   1. SymmetryAssemblySpec & expand_symmetry_assembly: allocation-free affine expansion
#      of asymmetric units (3 x M) across K 4x4 transforms into 3 x (K*M) assemblies.
#   2. MultiscalePartition & assign_multiscale_zones: MIC-aware 3-zone spatial partitioning
#      (Zone 0: All-Atom, Zone 1: Coarse-Grained buffer, Zone 2: Rigid Scaffold).
#   3. AdaptiveMultiscaleFactor <: Factors.AbstractFactor: combines an inner All-Atom factor,
#      a Coarse-Grained elastic-network + C1 force-shifted LJ buffer coupling (Zone 1),
#      and a translationally invariant O(|Z2|) Rigid-Scaffold harmonic summary tether (Zone 2),
#      with exact Cartesian forces and analytic periodic box derivatives.
#   4. softcore_relax_clashes!: O(N) linked-cell Beutler soft-core LJ steepest-descent
#      clash minimizer returning a ClashRelaxationResult.
#   5. DirtyTileDeltaTracker: per-atom displacement tracker emitting sparse dirty
#      32-atom Morton tile indices.

module Multiscale

using LinearAlgebra
using ..Types
using ..Types: AbstractBox, NoBox, OrthorhombicBox, TriclinicBox,
               SVector3, norm2, unwrap_displacement
using ..Factors
using ..Neighbors
using ..Alchemical

import ..Factors: evaluate!, AbstractFactor, is_local, force_consistent,
                  factor_support_size, read_support

export ZONE_ALL_ATOM, ZONE_COARSE_GRAINED, ZONE_RIGID_SCAFFOLD,
       SymmetryAssemblySpec, expand_symmetry_assembly,
       MultiscalePartition, assign_multiscale_zones, zone_counts,
       AdaptiveMultiscaleFactor, accumulate_box_derivative!,
       ClashRelaxationResult, softcore_relax_clashes!,
       DirtyTileDeltaTracker, update_dirty_tiles!, commit_tiles!,
       rebuild_morton!, tile_atoms

# ---------------------------------------------------------------------------
# Zone constants
# ---------------------------------------------------------------------------

const ZONE_ALL_ATOM       = 0x00
const ZONE_COARSE_GRAINED = 0x01
const ZONE_RIGID_SCAFFOLD = 0x02

# ---------------------------------------------------------------------------
# 1. Symmetry Assembly Expansion
# ---------------------------------------------------------------------------

"""
    SymmetryAssemblySpec(asu_positions, asu_masses, asu_charges, transforms)

Specification for expanding a `3 × M` asymmetric unit (ASU) by `K` rigid/affine
`4 × 4` homogeneous transforms into a `3 × (K*M)` biological assembly.
"""
struct SymmetryAssemblySpec
    asu_positions::Matrix{Float64}
    asu_masses::Vector{Float64}
    asu_charges::Vector{Float64}
    transforms::Vector{Matrix{Float64}}

    function SymmetryAssemblySpec(
        asu_positions::AbstractMatrix{<:Real},
        asu_masses::AbstractVector{<:Real},
        asu_charges::AbstractVector{<:Real},
        transforms::AbstractVector{<:AbstractMatrix{<:Real}}
    )
        size(asu_positions, 1) == 3 ||
            throw(ArgumentError("asu_positions must be 3 x M, got $(size(asu_positions))"))
        M = size(asu_positions, 2)
        M > 0 || throw(ArgumentError("asu_positions must contain at least one atom"))
        length(asu_masses) == M ||
            throw(ArgumentError("asu_masses length $(length(asu_masses)) != M=$M"))
        length(asu_charges) == M ||
            throw(ArgumentError("asu_charges length $(length(asu_charges)) != M=$M"))
        !isempty(transforms) ||
            throw(ArgumentError("transforms must contain at least one 4x4 matrix"))

        pos = Matrix{Float64}(asu_positions)
        masses = Vector{Float64}(asu_masses)
        charges = Vector{Float64}(asu_charges)

        all(isfinite, pos) || throw(ArgumentError("asu_positions contains non-finite values"))
        all(m -> isfinite(m) && m > 0.0, masses) ||
            throw(ArgumentError("asu_masses must be finite and strictly positive"))
        all(isfinite, charges) || throw(ArgumentError("asu_charges contains non-finite values"))

        t_vec = Vector{Matrix{Float64}}(undef, length(transforms))
        for k in eachindex(transforms)
            Tk = transforms[k]
            size(Tk) == (4, 4) ||
                throw(ArgumentError("transform $k must be 4x4, got $(size(Tk))"))
            M4 = Matrix{Float64}(Tk)
            all(isfinite, M4) ||
                throw(ArgumentError("transform $k contains non-finite entries"))
            t_vec[k] = M4
        end

        return new(pos, masses, charges, t_vec)
    end
end

"""
    expand_symmetry_assembly(spec::SymmetryAssemblySpec)
        -> (positions::Matrix{Float64}, masses::Vector{Float64}, charges::Vector{Float64})

Expand a `3 × M` asymmetric unit across `K` `4 × 4` affine transforms into
`3 × (K*M)` coordinates, masses, and charges without intermediate tuple allocations.
"""
function expand_symmetry_assembly(spec::SymmetryAssemblySpec)
    asu_pos = spec.asu_positions
    asu_m = spec.asu_masses
    asu_q = spec.asu_charges
    transforms = spec.transforms

    M = size(asu_pos, 2)
    K = length(transforms)
    N = K * M

    positions = Matrix{Float64}(undef, 3, N)
    masses = Vector{Float64}(undef, N)
    charges = Vector{Float64}(undef, N)

    @inbounds for k in 1:K
        T = transforms[k]
        r11 = T[1, 1]; r12 = T[1, 2]; r13 = T[1, 3]; tx = T[1, 4]
        r21 = T[2, 1]; r22 = T[2, 2]; r23 = T[2, 3]; ty = T[2, 4]
        r31 = T[3, 1]; r32 = T[3, 2]; r33 = T[3, 3]; tz = T[3, 4]
        base = (k - 1) * M
        for m in 1:M
            idx = base + m
            x = asu_pos[1, m]
            y = asu_pos[2, m]
            z = asu_pos[3, m]
            positions[1, idx] = r11 * x + r12 * y + r13 * z + tx
            positions[2, idx] = r21 * x + r22 * y + r23 * z + ty
            positions[3, idx] = r31 * x + r32 * y + r33 * z + tz
            masses[idx] = asu_m[m]
            charges[idx] = asu_q[m]
        end
    end

    return (positions = positions, masses = masses, charges = charges)
end

# ---------------------------------------------------------------------------
# 2. 3-Zone Spatial Multiscale Partition
# ---------------------------------------------------------------------------

"""
    MultiscalePartition(zones::Vector{UInt8}, center::NTuple{3,Float64},
                        aa_radius::Float64, cg_radius::Float64)

3-zone concentric spatial partition around `center`:
- `ZONE_ALL_ATOM (0x00)`: `r <= aa_radius`
- `ZONE_COARSE_GRAINED (0x01)`: `aa_radius < r <= cg_radius`
- `ZONE_RIGID_SCAFFOLD (0x02)`: `r > cg_radius`
"""
struct MultiscalePartition
    zones::Vector{UInt8}
    center::NTuple{3,Float64}
    aa_radius::Float64
    cg_radius::Float64

    function MultiscalePartition(
        zones::Vector{UInt8},
        center::NTuple{3,Float64},
        aa_radius::Float64,
        cg_radius::Float64
    )
        all(isfinite, center) || throw(ArgumentError("center must be finite, got $center"))
        (isfinite(aa_radius) && aa_radius >= 0.0) ||
            throw(ArgumentError("aa_radius must be finite and non-negative, got $aa_radius"))
        (isfinite(cg_radius) && cg_radius >= aa_radius) ||
            throw(ArgumentError("cg_radius ($cg_radius) must be finite and >= aa_radius ($aa_radius)"))
        @inbounds for i in eachindex(zones)
            z = zones[i]
            (z == ZONE_ALL_ATOM || z == ZONE_COARSE_GRAINED || z == ZONE_RIGID_SCAFFOLD) ||
                throw(ArgumentError("invalid zone code $z at atom $i"))
        end
        return new(zones, center, aa_radius, cg_radius)
    end
end

"""
    assign_multiscale_zones(q, center, aa_radius, cg_radius; box = Types.NoBox()) -> MultiscalePartition

Assign each atom in `3 × N` coordinate matrix `q` to one of the three multiscale zones
using minimum-image distances under `box` (`NoBox`, `OrthorhombicBox`, or `TriclinicBox`).
"""
function assign_multiscale_zones(
    q::AbstractMatrix{<:Real},
    center::NTuple{3,<:Real},
    aa_radius::Real,
    cg_radius::Real;
    box::AbstractBox = NoBox()
)
    size(q, 1) == 3 || throw(ArgumentError("q must be 3 x N, got $(size(q))"))
    cx = Float64(center[1])
    cy = Float64(center[2])
    cz = Float64(center[3])
    r_aa = Float64(aa_radius)
    r_cg = Float64(cg_radius)
    (isfinite(r_aa) && r_aa >= 0.0) ||
        throw(ArgumentError("aa_radius must be finite and non-negative, got $aa_radius"))
    (isfinite(r_cg) && r_cg >= r_aa) ||
        throw(ArgumentError("cg_radius ($cg_radius) must be finite and >= aa_radius ($aa_radius)"))

    N = size(q, 2)
    zones = Vector{UInt8}(undef, N)
    r_aa2 = r_aa * r_aa
    r_cg2 = r_cg * r_cg

    @inbounds for i in 1:N
        dx = Float64(q[1, i]) - cx
        dy = Float64(q[2, i]) - cy
        dz = Float64(q[3, i]) - cz
        d = Neighbors.minimum_image(box, dx, dy, dz)
        r2 = norm2(d)
        if r2 <= r_aa2
            zones[i] = ZONE_ALL_ATOM
        elseif r2 <= r_cg2
            zones[i] = ZONE_COARSE_GRAINED
        else
            zones[i] = ZONE_RIGID_SCAFFOLD
        end
    end

    return MultiscalePartition(zones, (cx, cy, cz), r_aa, r_cg)
end

function assign_multiscale_zones(
    q::AbstractMatrix{<:Real},
    center::AbstractVector{<:Real},
    aa_radius::Real,
    cg_radius::Real;
    box::AbstractBox = NoBox()
)
    length(center) == 3 || throw(ArgumentError("center must have length 3, got $(length(center))"))
    return assign_multiscale_zones(
        q, (Float64(center[1]), Float64(center[2]), Float64(center[3])),
        aa_radius, cg_radius; box = box
    )
end

"""
    zone_counts(part::MultiscalePartition) -> (all_atom::Int, coarse_grained::Int, rigid_scaffold::Int)

Return the atom counts in each of the three multiscale zones.
"""
function zone_counts(part::MultiscalePartition)
    n_aa = 0
    n_cg = 0
    n_rigid = 0
    @inbounds for z in part.zones
        if z == ZONE_ALL_ATOM
            n_aa += 1
        elseif z == ZONE_COARSE_GRAINED
            n_cg += 1
        else
            n_rigid += 1
        end
    end
    return (all_atom = n_aa, coarse_grained = n_cg, rigid_scaffold = n_rigid)
end

# ---------------------------------------------------------------------------
# Internal O(N) linked-cell helpers (Orthorhombic, Triclinic, and NoBox)
# ---------------------------------------------------------------------------

@inline function _orthorhombic_bucket(x::Float64, L::Float64, n::Int)
    xx = x - L * floor(x / L)
    return min(n, max(1, floor(Int, xx / L * n) + 1))
end

function _dedup_periodic_neighbor_cells(c1::Int, c2::Int, c3::Int, nc::NTuple{3,Int})
    out = Vector{Int}()
    sizehint!(out, 27)
    n1 = nc[1]
    n12 = nc[1] * nc[2]
    @inbounds for dz in -1:1, dy in -1:1, dx in -1:1
        d1 = mod1(c1 + dx, nc[1])
        d2 = mod1(c2 + dy, nc[2])
        d3 = mod1(c3 + dz, nc[3])
        id = (d3 - 1) * n12 + (d2 - 1) * n1 + d1
        found = false
        for prev in out
            if prev == id
                found = true
                break
            end
        end
        if !found
            push!(out, id)
        end
    end
    return out
end

function _build_nobox_spatial_hash(q::AbstractMatrix{Float64}, cell_size::Float64)
    N = size(q, 2)
    inv_cs = 1.0 / cell_size
    grid = Dict{NTuple{3,Int}, Vector{Int}}()
    sizehint!(grid, N)
    coords = Vector{NTuple{3,Int}}(undef, N)
    @inbounds for i in 1:N
        cx = floor(Int, q[1, i] * inv_cs)
        cy = floor(Int, q[2, i] * inv_cs)
        cz = floor(Int, q[3, i] * inv_cs)
        key = (cx, cy, cz)
        coords[i] = key
        bucket = get!(Vector{Int}, grid, key)
        push!(bucket, i)
    end
    return grid, coords
end

"""
Enumerate all unordered pairs `(i, j)` (`i < j`) within `rcut` in O(N) time using
linked cells (orthorhombic / triclinic) or spatial hashing (`NoBox`), invoking `f(i, j, d, r2)`.
"""
function _foreach_neighbor_pair(f::F, q::AbstractMatrix{Float64}, box::OrthorhombicBox, rcut::Float64) where {F}
    N = size(q, 2)
    N < 2 && return nothing
    L = box.L
    nc = ntuple(d -> max(1, floor(Int, L[d] / rcut)), 3)
    n1 = nc[1]
    n12 = nc[1] * nc[2]
    ncells = n12 * nc[3]
    members = [Int[] for _ in 1:ncells]
    cell_of = Vector{Int}(undef, N)
    @inbounds for i in 1:N
        c1 = _orthorhombic_bucket(q[1, i], L[1], nc[1])
        c2 = _orthorhombic_bucket(q[2, i], L[2], nc[2])
        c3 = _orthorhombic_bucket(q[3, i], L[3], nc[3])
        id = (c3 - 1) * n12 + (c2 - 1) * n1 + c1
        cell_of[i] = id
        push!(members[id], i)
    end
    rc2 = rcut * rcut
    @inbounds for c3 in 1:nc[3], c2 in 1:nc[2], c1 in 1:nc[1]
        cid = (c3 - 1) * n12 + (c2 - 1) * n1 + c1
        atoms_c = members[cid]
        isempty(atoms_c) && continue
        nbrs = _dedup_periodic_neighbor_cells(c1, c2, c3, nc)
        for nid in nbrs
            nid < cid && continue
            atoms_n = members[nid]
            isempty(atoms_n) && continue
            if nid == cid
                len = length(atoms_c)
                for a in 1:len-1
                    i = atoms_c[a]
                    for b in a+1:len
                        j = atoms_c[b]
                        ii, jj = i < j ? (i, j) : (j, i)
                        d = unwrap_displacement(box, q, ii, jj)
                        r2 = norm2(d)
                        if r2 < rc2
                            f(ii, jj, d, r2)
                        end
                    end
                end
            else
                for i in atoms_c
                    for j in atoms_n
                        ii, jj = i < j ? (i, j) : (j, i)
                        d = unwrap_displacement(box, q, ii, jj)
                        r2 = norm2(d)
                        if r2 < rc2
                            f(ii, jj, d, r2)
                        end
                    end
                end
            end
        end
    end
    return nothing
end

function _foreach_neighbor_pair(f::F, q::AbstractMatrix{Float64}, box::TriclinicBox, rcut::Float64) where {F}
    N = size(q, 2)
    N < 2 && return nothing
    nc = ntuple(d -> max(1, floor(Int, box.widths[d] / rcut)), 3)
    n1 = nc[1]
    n12 = nc[1] * nc[2]
    ncells = n12 * nc[3]
    members = [Int[] for _ in 1:ncells]
    Hinv = box.Hinv
    @inbounds for i in 1:N
        q1 = q[1, i]; q2 = q[2, i]; q3 = q[3, i]
        s1 = Hinv[1, 1] * q1 + Hinv[1, 2] * q2 + Hinv[1, 3] * q3
        s2 = Hinv[2, 1] * q1 + Hinv[2, 2] * q2 + Hinv[2, 3] * q3
        s3 = Hinv[3, 1] * q1 + Hinv[3, 2] * q2 + Hinv[3, 3] * q3
        f1 = s1 - floor(s1)
        f2 = s2 - floor(s2)
        f3 = s3 - floor(s3)
        c1 = min(nc[1], max(1, floor(Int, f1 * nc[1]) + 1))
        c2 = min(nc[2], max(1, floor(Int, f2 * nc[2]) + 1))
        c3 = min(nc[3], max(1, floor(Int, f3 * nc[3]) + 1))
        id = (c3 - 1) * n12 + (c2 - 1) * n1 + c1
        push!(members[id], i)
    end
    rc2 = rcut * rcut
    @inbounds for c3 in 1:nc[3], c2 in 1:nc[2], c1 in 1:nc[1]
        cid = (c3 - 1) * n12 + (c2 - 1) * n1 + c1
        atoms_c = members[cid]
        isempty(atoms_c) && continue
        nbrs = _dedup_periodic_neighbor_cells(c1, c2, c3, nc)
        for nid in nbrs
            nid < cid && continue
            atoms_n = members[nid]
            isempty(atoms_n) && continue
            if nid == cid
                len = length(atoms_c)
                for a in 1:len-1
                    i = atoms_c[a]
                    for b in a+1:len
                        j = atoms_c[b]
                        ii, jj = i < j ? (i, j) : (j, i)
                        d = unwrap_displacement(box, q, ii, jj)
                        r2 = norm2(d)
                        if r2 < rc2
                            f(ii, jj, d, r2)
                        end
                    end
                end
            else
                for i in atoms_c
                    for j in atoms_n
                        ii, jj = i < j ? (i, j) : (j, i)
                        d = unwrap_displacement(box, q, ii, jj)
                        r2 = norm2(d)
                        if r2 < rc2
                            f(ii, jj, d, r2)
                        end
                    end
                end
            end
        end
    end
    return nothing
end

function _foreach_neighbor_pair(f::F, q::AbstractMatrix{Float64}, box::NoBox, rcut::Float64) where {F}
    N = size(q, 2)
    N < 2 && return nothing
    grid, _ = _build_nobox_spatial_hash(q, rcut)
    rc2 = rcut * rcut
    @inbounds for (key, atoms_c) in grid
        cx, cy, cz = key
        # Intra-cell pairs
        len = length(atoms_c)
        for a in 1:len-1
            i = atoms_c[a]
            for b in a+1:len
                j = atoms_c[b]
                ii, jj = i < j ? (i, j) : (j, i)
                d = unwrap_displacement(box, q, ii, jj)
                r2 = norm2(d)
                if r2 < rc2
                    f(ii, jj, d, r2)
                end
            end
        end
        # Half-shell neighbor cells (lexicographically greater keys only)
        for dz in -1:1, dy in -1:1, dx in -1:1
            nkey = (cx + dx, cy + dy, cz + dz)
            nkey <= key && continue
            atoms_n = get(grid, nkey, nothing)
            atoms_n === nothing && continue
            for i in atoms_c
                for j in atoms_n
                    ii, jj = i < j ? (i, j) : (j, i)
                    d = unwrap_displacement(box, q, ii, jj)
                    r2 = norm2(d)
                    if r2 < rc2
                        f(ii, jj, d, r2)
                    end
                end
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# 3. AdaptiveMultiscaleFactor
# ---------------------------------------------------------------------------

"""
    AdaptiveMultiscaleFactor{AF} <: Factors.AbstractFactor

3-zone multiscale energy/force factor:
1. **Zone 0 (All-Atom + Zone 0-1 boundary)**: delegates to `aa_factor` (any `Factors.AbstractFactor`
   or `nothing`).
2. **Zone 1 (Coarse-Grained buffer)**: elastic-network harmonic bonds
   `U_ENM = (k_cg / 2) Σ (r_ij - r0_ij)^2` plus C1 linear-force-shifted (`:lfs`)
   coarse-grained Lennard-Jones pairs.
3. **Zone 2 (Rigid Scaffold)**: translationally invariant `O(|Z2|)` centroid-relative
   harmonic summary tether:
   `U_rigid = (k_rigid / 2) Σ_{i ∈ Z2} ||(q_i - c_2) - (q_ref_i - c_2_ref)||^2`,
   preserving exact Newton's 3rd law (`Σ F_i = 0`) and exact analytic box derivatives.
"""
struct AdaptiveMultiscaleFactor{AF} <: Factors.AbstractFactor
    aa_factor::AF
    partition::MultiscalePartition
    cg_bond_i::Vector{Int}
    cg_bond_j::Vector{Int}
    cg_bond_r0::Vector{Float64}
    cg_k::Float64
    cg_lj_i::Vector{Int}
    cg_lj_j::Vector{Int}
    cg_sigma::Float64
    cg_epsilon::Float64
    cg_rcut::Float64
    rigid_indices::Vector{Int}
    rigid_ref_rel::Matrix{Float64}
    k_rigid::Float64

    function AdaptiveMultiscaleFactor(
        aa_factor::AF,
        partition::MultiscalePartition,
        cg_bond_i::Vector{Int},
        cg_bond_j::Vector{Int},
        cg_bond_r0::Vector{Float64},
        cg_k::Real,
        cg_lj_i::Vector{Int},
        cg_lj_j::Vector{Int},
        cg_sigma::Real,
        cg_epsilon::Real,
        cg_rcut::Real,
        rigid_indices::Vector{Int},
        rigid_ref_rel::Matrix{Float64},
        k_rigid::Real
    ) where {AF}
        (aa_factor === nothing || aa_factor isa Factors.AbstractFactor) ||
            throw(ArgumentError("aa_factor must be nothing or a Factors.AbstractFactor, got $(typeof(aa_factor))"))
        length(cg_bond_i) == length(cg_bond_j) == length(cg_bond_r0) ||
            throw(ArgumentError("cg_bond arrays must have identical length"))
        length(cg_lj_i) == length(cg_lj_j) ||
            throw(ArgumentError("cg_lj_i and cg_lj_j must have identical length"))
        size(rigid_ref_rel) == (3, length(rigid_indices)) ||
            throw(ArgumentError("rigid_ref_rel must be 3 x length(rigid_indices)"))
        (isfinite(cg_k) && cg_k >= 0.0) || throw(ArgumentError("cg_k must be finite and >= 0"))
        (isfinite(cg_sigma) && cg_sigma > 0.0) || throw(ArgumentError("cg_sigma must be finite and > 0"))
        (isfinite(cg_epsilon) && cg_epsilon >= 0.0) || throw(ArgumentError("cg_epsilon must be finite and >= 0"))
        (isfinite(cg_rcut) && cg_rcut > 0.0) || throw(ArgumentError("cg_rcut must be finite and > 0"))
        (isfinite(k_rigid) && k_rigid >= 0.0) || throw(ArgumentError("k_rigid must be finite and >= 0"))

        return new{AF}(
            aa_factor, partition,
            cg_bond_i, cg_bond_j, cg_bond_r0, Float64(cg_k),
            cg_lj_i, cg_lj_j, Float64(cg_sigma), Float64(cg_epsilon), Float64(cg_rcut),
            rigid_indices, rigid_ref_rel, Float64(k_rigid)
        )
    end
end

"""
    AdaptiveMultiscaleFactor(aa_factor, partition, q_ref; box=NoBox(), cg_k=10.0,
                             cg_enm_cutoff=0.8, cg_sigma=0.35, cg_epsilon=0.5,
                             cg_rcut=1.0, k_rigid=50.0, include_cg_lj=true,
                             include_zone12_coupling=true)

Construct an `AdaptiveMultiscaleFactor` from an inner All-Atom factor, a `MultiscalePartition`,
and reference coordinates `q_ref` (`3 × N`), automatically building Zone 1 elastic-network
bonds, Zone 1 coarse-grained LJ pairs, and the Zone 2 centroid-relative rigid scaffold summary.
"""
function AdaptiveMultiscaleFactor(
    aa_factor,
    partition::MultiscalePartition,
    q_ref::AbstractMatrix{<:Real};
    box::AbstractBox = NoBox(),
    cg_k::Real = 10.0,
    cg_enm_cutoff::Real = 0.8,
    cg_sigma::Real = 0.35,
    cg_epsilon::Real = 0.5,
    cg_rcut::Real = 1.0,
    k_rigid::Real = 50.0,
    include_cg_lj::Bool = true,
    include_zone12_coupling::Bool = true
)
    size(q_ref, 1) == 3 || throw(ArgumentError("q_ref must be 3 x N, got $(size(q_ref))"))
    N = size(q_ref, 2)
    length(partition.zones) == N ||
        throw(ArgumentError("partition length $(length(partition.zones)) != N=$N"))
    qref_f64 = q_ref isa Matrix{Float64} ? q_ref : Matrix{Float64}(q_ref)

    zones = partition.zones
    cg_bond_i = Int[]
    cg_bond_j = Int[]
    cg_bond_r0 = Float64[]
    cg_lj_i = Int[]
    cg_lj_j = Int[]

    search_r = max(Float64(cg_enm_cutoff), include_cg_lj ? Float64(cg_rcut) : 0.0)
    enm_r2 = Float64(cg_enm_cutoff)^2
    lj_r2 = Float64(cg_rcut)^2

    if search_r > 0.0 && N >= 2
        _foreach_neighbor_pair(qref_f64, box, search_r) do i, j, _d, r2
            zi = zones[i]
            zj = zones[j]
            # Zone 1 - Zone 1 pairs
            if zi == ZONE_COARSE_GRAINED && zj == ZONE_COARSE_GRAINED
                if r2 > 1e-16 && r2 <= enm_r2 && cg_k > 0.0
                    push!(cg_bond_i, i)
                    push!(cg_bond_j, j)
                    push!(cg_bond_r0, sqrt(r2))
                end
                if include_cg_lj && r2 > 1e-16 && r2 < lj_r2 && cg_epsilon > 0.0
                    push!(cg_lj_i, i)
                    push!(cg_lj_j, j)
                end
            # Optional Zone 1 - Zone 2 boundary elastic coupling
            elseif include_zone12_coupling && cg_k > 0.0 &&
                   ((zi == ZONE_COARSE_GRAINED && zj == ZONE_RIGID_SCAFFOLD) ||
                    (zi == ZONE_RIGID_SCAFFOLD && zj == ZONE_COARSE_GRAINED))
                if r2 > 1e-16 && r2 <= enm_r2
                    push!(cg_bond_i, i)
                    push!(cg_bond_j, j)
                    push!(cg_bond_r0, sqrt(r2))
                end
            end
        end
    end

    # Collect Zone 2 rigid scaffold atoms and compute centroid-relative reference geometry
    rigid_indices = Int[]
    @inbounds for i in 1:N
        if zones[i] == ZONE_RIGID_SCAFFOLD
            push!(rigid_indices, i)
        end
    end
    n_rigid = length(rigid_indices)
    rigid_ref_rel = Matrix{Float64}(undef, 3, n_rigid)
    if n_rigid > 0
        cx = 0.0; cy = 0.0; cz = 0.0
        @inbounds for idx in rigid_indices
            cx += qref_f64[1, idx]
            cy += qref_f64[2, idx]
            cz += qref_f64[3, idx]
        end
        inv_n = 1.0 / n_rigid
        cx *= inv_n; cy *= inv_n; cz *= inv_n
        @inbounds for m in 1:n_rigid
            idx = rigid_indices[m]
            rigid_ref_rel[1, m] = qref_f64[1, idx] - cx
            rigid_ref_rel[2, m] = qref_f64[2, idx] - cy
            rigid_ref_rel[3, m] = qref_f64[3, idx] - cz
        end
    end

    return AdaptiveMultiscaleFactor(
        aa_factor, partition,
        cg_bond_i, cg_bond_j, cg_bond_r0, cg_k,
        cg_lj_i, cg_lj_j, cg_sigma, cg_epsilon, cg_rcut,
        rigid_indices, rigid_ref_rel, k_rigid
    )
end

# Factor protocol traits
Factors.is_local(::AdaptiveMultiscaleFactor) = false
Factors.force_consistent(f::AdaptiveMultiscaleFactor) =
    f.aa_factor === nothing ? true : Factors.force_consistent(f.aa_factor)
Factors.factor_support_size(f::AdaptiveMultiscaleFactor) = length(f.partition.zones)
Factors.read_support(f::AdaptiveMultiscaleFactor) = Tuple(1:length(f.partition.zones))

function Factors.evaluate!(
    F::AbstractMatrix{Float64},
    f::AdaptiveMultiscaleFactor,
    q::AbstractMatrix{Float64},
    ctx::Factors.Context
)
    size(q, 1) == 3 && size(F) == size(q) ||
        throw(ArgumentError("q and F must be 3 x N, got q=$(size(q)), F=$(size(F))"))
    box = ctx.box
    total_u = 0.0

    # 1. Zone 0 (and Zone 0-1 boundary) inner All-Atom factor
    if f.aa_factor !== nothing
        total_u += Factors.evaluate!(F, f.aa_factor, q, ctx)
    end

    # 2a. Zone 1 Coarse-Grained elastic network bonds
    k_cg = f.cg_k
    if k_cg != 0.0 && !isempty(f.cg_bond_i)
        bi = f.cg_bond_i
        bj = f.cg_bond_j
        br0 = f.cg_bond_r0
        @inbounds for p in eachindex(bi)
            i = bi[p]
            j = bj[p]
            d = unwrap_displacement(box, q, i, j)
            r2 = norm2(d)
            r2 < 1e-24 && return Inf
            r = sqrt(r2)
            dr = r - br0[p]
            total_u += 0.5 * k_cg * dr * dr
            c = -k_cg * dr / r
            gx = c * d.x
            gy = c * d.y
            gz = c * d.z
            F[1, i] += gx; F[2, i] += gy; F[3, i] += gz
            F[1, j] -= gx; F[2, j] -= gy; F[3, j] -= gz
        end
    end

    # 2b. Zone 1 Coarse-Grained C1 force-shifted (:lfs) LJ buffer pairs
    eps = f.cg_epsilon
    if eps != 0.0 && !isempty(f.cg_lj_i)
        sig = f.cg_sigma
        rc = f.cg_rcut
        rc2 = rc * rc
        uc = Factors.lj_energy(eps, sig, rc)
        duc = Factors.lj_dudr(eps, sig, rc)
        li = f.cg_lj_i
        lj = f.cg_lj_j
        @inbounds for p in eachindex(li)
            i = li[p]
            j = lj[p]
            d = unwrap_displacement(box, q, i, j)
            r2 = norm2(d)
            r2 >= rc2 && continue
            r2 < 1e-24 && return Inf
            r = sqrt(r2)
            u_raw = Factors.lj_energy(eps, sig, r)
            du_raw = Factors.lj_dudr(eps, sig, r)
            total_u += u_raw - uc - duc * (r - rc)
            c = -(du_raw - duc) / r
            gx = c * d.x
            gy = c * d.y
            gz = c * d.z
            F[1, i] += gx; F[2, i] += gy; F[3, i] += gz
            F[1, j] -= gx; F[2, j] -= gy; F[3, j] -= gz
        end
    end

    # 3. Zone 2 Rigid-Scaffold centroid-relative harmonic summary tether (O(|Z2|), ΣF = 0)
    k_rig = f.k_rigid
    ridx = f.rigid_indices
    n_rig = length(ridx)
    if k_rig != 0.0 && n_rig > 0
        cx = 0.0; cy = 0.0; cz = 0.0
        @inbounds for m in 1:n_rig
            idx = ridx[m]
            cx += q[1, idx]
            cy += q[2, idx]
            cz += q[3, idx]
        end
        inv_n = 1.0 / n_rig
        cx *= inv_n; cy *= inv_n; cz *= inv_n
        ref_rel = f.rigid_ref_rel
        @inbounds for m in 1:n_rig
            idx = ridx[m]
            dx = (q[1, idx] - cx) - ref_rel[1, m]
            dy = (q[2, idx] - cy) - ref_rel[2, m]
            dz = (q[3, idx] - cz) - ref_rel[3, m]
            total_u += 0.5 * k_rig * (dx * dx + dy * dy + dz * dz)
            F[1, idx] -= k_rig * dx
            F[2, idx] -= k_rig * dy
            F[3, idx] -= k_rig * dz
        end
    end

    return total_u
end

function Factors.evaluate!(
    F::AbstractMatrix{Float64},
    f::AdaptiveMultiscaleFactor,
    q::AbstractMatrix{Float64},
    box::AbstractBox
)
    ctx = Factors.Context(box, Types.REDUCED, size(q, 2))
    return Factors.evaluate!(F, f, q, ctx)
end

# ---------------------------------------------------------------------------
# Box derivatives for AdaptiveMultiscaleFactor (Orthorhombic & Triclinic)
# ---------------------------------------------------------------------------

@inline function _accum_outer!(
    G::AbstractMatrix{Float64},
    gx::Float64, gy::Float64, gz::Float64,
    wx::Float64, wy::Float64, wz::Float64
)
    @inbounds begin
        G[1, 1] += gx * wx; G[1, 2] += gx * wy; G[1, 3] += gx * wz
        G[2, 1] += gy * wx; G[2, 2] += gy * wy; G[2, 3] += gy * wz
        G[3, 1] += gz * wx; G[3, 2] += gz * wy; G[3, 3] += gz * wz
    end
    return nothing
end

@inline function _fractional_vec(box::OrthorhombicBox, vx::Float64, vy::Float64, vz::Float64)
    L = box.L
    return (vx / L[1], vy / L[2], vz / L[3])
end

@inline function _fractional_vec(box::TriclinicBox, vx::Float64, vy::Float64, vz::Float64)
    M = box.Hinv
    @inbounds begin
        wx = M[1, 1] * vx + M[1, 2] * vy + M[1, 3] * vz
        wy = M[2, 1] * vx + M[2, 2] * vy + M[2, 3] * vz
        wz = M[3, 1] * vx + M[3, 2] * vy + M[3, 3] * vz
    end
    return (wx, wy, wz)
end

function _accumulate_inner_box_derivative!(
    G::AbstractMatrix{Float64},
    aa_factor,
    q::AbstractMatrix{Float64},
    ctx::Factors.Context
)
    aa_factor === nothing && return G
    if isdefined(parentmodule(@__MODULE__), :ResolvedSystem)
        RS = getfield(parentmodule(@__MODULE__), :ResolvedSystem)
        if hasmethod(RS.accumulate_box_derivative!, Tuple{Matrix{Float64}, typeof(aa_factor), Matrix{Float64}, Factors.Context}) ||
           hasmethod(RS.accumulate_box_derivative!, Tuple{Matrix{Float64}, typeof(aa_factor), AbstractMatrix{Float64}, Factors.Context})
            RS.accumulate_box_derivative!(G, aa_factor, q, ctx)
            return G
        end
    end
    if isdefined(Factors, :accumulate_box_derivative!) &&
       hasmethod(Factors.accumulate_box_derivative!, Tuple{typeof(G), typeof(aa_factor), typeof(q), Factors.Context})
        Factors.accumulate_box_derivative!(G, aa_factor, q, ctx)
        return G
    end
    if aa_factor isa Alchemical.AlchemicalNonbondedFactor
        Alchemical.accumulate_box_derivative!(G, aa_factor, q, ctx.box)
        return G
    end
    throw(ArgumentError("inner aa_factor of type $(typeof(aa_factor)) does not implement accumulate_box_derivative!"))
end

"""
    accumulate_box_derivative!(G, f::AdaptiveMultiscaleFactor, q, ctx) -> G
    accumulate_box_derivative!(G, f::AdaptiveMultiscaleFactor, q, box) -> G

In-place accumulation of the exact analytic periodic box derivative `G[a, b] += ∂U / ∂H[a, b]`
at fixed fractional coordinates for `OrthorhombicBox` and `TriclinicBox`.
"""
function accumulate_box_derivative!(
    G::AbstractMatrix{Float64},
    f::AdaptiveMultiscaleFactor,
    q::AbstractMatrix{Float64},
    ctx::Factors.Context
)
    size(G) == (3, 3) || throw(ArgumentError("G must be 3x3, got $(size(G))"))
    box = ctx.box
    box isa NoBox && throw(ArgumentError("no periodic box: box derivative ∂U/∂H is undefined for NoBox"))
    (box isa OrthorhombicBox || box isa TriclinicBox) ||
        throw(Types.UnsupportedGeometry("unsupported box type $(typeof(box)) for box derivative"))

    # 1. Inner All-Atom factor
    if f.aa_factor !== nothing
        _accumulate_inner_box_derivative!(G, f.aa_factor, q, ctx)
    end

    # 2a. Zone 1 elastic-network bonds
    k_cg = f.cg_k
    if k_cg != 0.0 && !isempty(f.cg_bond_i)
        bi = f.cg_bond_i
        bj = f.cg_bond_j
        br0 = f.cg_bond_r0
        @inbounds for p in eachindex(bi)
            i = bi[p]
            j = bj[p]
            d = unwrap_displacement(box, q, i, j)
            r2 = norm2(d)
            r2 < 1e-24 && continue
            r = sqrt(r2)
            inv_r_dudr = k_cg * (r - br0[p]) / r
            wx, wy, wz = _fractional_vec(box, d.x, d.y, d.z)
            _accum_outer!(G, inv_r_dudr * d.x, inv_r_dudr * d.y, inv_r_dudr * d.z, wx, wy, wz)
        end
    end

    # 2b. Zone 1 C1 force-shifted (:lfs) LJ pairs
    eps = f.cg_epsilon
    if eps != 0.0 && !isempty(f.cg_lj_i)
        sig = f.cg_sigma
        rc = f.cg_rcut
        rc2 = rc * rc
        duc = Factors.lj_dudr(eps, sig, rc)
        li = f.cg_lj_i
        lj = f.cg_lj_j
        @inbounds for p in eachindex(li)
            i = li[p]
            j = lj[p]
            d = unwrap_displacement(box, q, i, j)
            r2 = norm2(d)
            (r2 >= rc2 || r2 < 1e-24) && continue
            r = sqrt(r2)
            inv_r_dudr = (Factors.lj_dudr(eps, sig, r) - duc) / r
            wx, wy, wz = _fractional_vec(box, d.x, d.y, d.z)
            _accum_outer!(G, inv_r_dudr * d.x, inv_r_dudr * d.y, inv_r_dudr * d.z, wx, wy, wz)
        end
    end

    # 3. Zone 2 Rigid-Scaffold centroid-relative harmonic summary tether
    k_rig = f.k_rigid
    ridx = f.rigid_indices
    n_rig = length(ridx)
    if k_rig != 0.0 && n_rig > 0
        cx = 0.0; cy = 0.0; cz = 0.0
        @inbounds for m in 1:n_rig
            idx = ridx[m]
            cx += q[1, idx]
            cy += q[2, idx]
            cz += q[3, idx]
        end
        inv_n = 1.0 / n_rig
        cx *= inv_n; cy *= inv_n; cz *= inv_n
        ref_rel = f.rigid_ref_rel
        @inbounds for m in 1:n_rig
            idx = ridx[m]
            vx = q[1, idx] - cx
            vy = q[2, idx] - cy
            vz = q[3, idx] - cz
            gx = k_rig * (vx - ref_rel[1, m])
            gy = k_rig * (vy - ref_rel[2, m])
            gz = k_rig * (vz - ref_rel[3, m])
            wx, wy, wz = _fractional_vec(box, vx, vy, vz)
            _accum_outer!(G, gx, gy, gz, wx, wy, wz)
        end
    end

    return G
end

function accumulate_box_derivative!(
    G::AbstractMatrix{Float64},
    f::AdaptiveMultiscaleFactor,
    q::AbstractMatrix{Float64},
    box::AbstractBox
)
    ctx = Factors.Context(box, Types.REDUCED, size(q, 2))
    return accumulate_box_derivative!(G, f, q, ctx)
end

if isdefined(parentmodule(@__MODULE__), :ResolvedSystem)
    import ..ResolvedSystem as _RS
    _RS.box_derivative_support(f::AdaptiveMultiscaleFactor) =
        f.aa_factor === nothing ? nothing : _RS.box_derivative_support(f.aa_factor)
    _RS.accumulate_box_derivative!(G::Matrix{Float64}, f::AdaptiveMultiscaleFactor,
                                    q::AbstractMatrix{Float64}, ctx::Factors.Context) =
        accumulate_box_derivative!(G, f, q, ctx)
end

# ---------------------------------------------------------------------------
# 4. O(N) Linked-Cell Beutler Soft-Core LJ Clash Relaxation
# ---------------------------------------------------------------------------

"""
    ClashRelaxationResult

Diagnostic summary returned by `softcore_relax_clashes!`.
"""
struct ClashRelaxationResult
    initial_energy::Float64
    final_energy::Float64
    initial_max_force::Float64
    final_max_force::Float64
    steps_run::Int
    converged::Bool
end

@inline function _beutler_lj_raw(r::Float64, sigma::Float64, eps::Float64, lam::Float64, alpha::Float64)
    s6 = sigma^6
    s12 = s6 * s6
    r2 = r * r
    r6 = r2 * r2 * r2
    S = alpha * (1.0 - lam) * s6 + r6
    invS = 1.0 / S
    invS2 = invS * invS
    invS3 = invS2 * invS
    u = 4.0 * eps * lam * (s12 * invS2 - s6 * invS)
    if r == 0.0
        dudr = 0.0
    else
        r5 = r * r2 * r2
        dudr = -24.0 * eps * lam * r5 * s6 * (2.0 * s6 - S) * invS3
    end
    return (u = u, dudr = dudr)
end

function _eval_softcore_linked!(
    F::Matrix{Float64},
    q::AbstractMatrix{Float64},
    sigma::Vector{Float64},
    epsilon::Vector{Float64},
    box::AbstractBox,
    rcut::Float64,
    alpha_sc::Float64,
    lambda::Float64
)
    fill!(F, 0.0)
    total_u = 0.0
    n_clashes = 0

    _foreach_neighbor_pair(q, box, rcut) do i, j, d, r2
        sig_ij = 0.5 * (sigma[i] + sigma[j])
        eps_ij = sqrt(epsilon[i] * epsilon[j])
        (eps_ij == 0.0 || sig_ij <= 0.0) && return
        r = sqrt(r2)
        if r < 0.7 * sig_ij
            n_clashes += 1
        end
        raw = _beutler_lj_raw(r, sig_ij, eps_ij, lambda, alpha_sc)
        raw_c = _beutler_lj_raw(rcut, sig_ij, eps_ij, lambda, alpha_sc)
        u_lfs = raw.u - raw_c.u - (r - rcut) * raw_c.dudr
        total_u += u_lfs
        if r > 0.0
            c = -(raw.dudr - raw_c.dudr) / r
            gx = c * d.x
            gy = c * d.y
            gz = c * d.z
            @inbounds begin
                F[1, i] += gx; F[2, i] += gy; F[3, i] += gz
                F[1, j] -= gx; F[2, j] -= gy; F[3, j] -= gz
            end
        end
    end

    max_f2 = 0.0
    @inbounds for i in axes(F, 2)
        f2 = F[1, i] * F[1, i] + F[2, i] * F[2, i] + F[3, i] * F[3, i]
        if f2 > max_f2
            max_f2 = f2
        end
    end
    return (energy = total_u, max_force = sqrt(max_f2), n_clashes = n_clashes)
end

"""
    softcore_relax_clashes!(q, sigma, epsilon, box; rcut=1.0, alpha_sc=0.5, lambda=0.5,
                            max_steps=50, step_size=0.002, max_disp=0.02, force_tol=500.0)
        -> ClashRelaxationResult

Relax severe atomic overlaps in-place using a Beutler soft-core Lennard-Jones (`:lfs`)
adaptive steepest-descent minimizer backed by `O(N)` linked cells.

Uses normalized steepest descent (`Δq_i = h * F_i / max_j ||F_j||` with `h <= max_disp`)
and adaptive backtracking (`1.5x` growth on energy decrease, `0.5x` shrinkage on increase),
ensuring rapid, finite-force relaxation even on the flat top of the Beutler soft-core barrier.
"""
function softcore_relax_clashes!(
    q::AbstractMatrix{Float64},
    sigma::AbstractVector{<:Real},
    epsilon::AbstractVector{<:Real},
    box::AbstractBox;
    rcut::Real = 1.0,
    alpha_sc::Real = 0.5,
    lambda::Real = 0.5,
    max_steps::Int = 50,
    step_size::Real = 0.002,
    max_disp::Real = 0.02,
    force_tol::Real = 500.0
)
    size(q, 1) == 3 || throw(ArgumentError("q must be 3 x N, got $(size(q))"))
    N = size(q, 2)
    length(sigma) == N || throw(ArgumentError("sigma length $(length(sigma)) != N=$N"))
    length(epsilon) == N || throw(ArgumentError("epsilon length $(length(epsilon)) != N=$N"))
    (isfinite(rcut) && rcut > 0.0) || throw(ArgumentError("rcut must be finite and > 0"))
    (isfinite(alpha_sc) && alpha_sc >= 0.0) || throw(ArgumentError("alpha_sc must be finite and >= 0"))
    (isfinite(lambda) && 0.0 < lambda <= 1.0) || throw(ArgumentError("lambda must be in (0, 1]"))
    max_steps >= 0 || throw(ArgumentError("max_steps must be >= 0"))
    (isfinite(step_size) && step_size > 0.0) || throw(ArgumentError("step_size must be finite and > 0"))
    (isfinite(max_disp) && max_disp > 0.0) || throw(ArgumentError("max_disp must be finite and > 0"))
    (isfinite(force_tol) && force_tol >= 0.0) || throw(ArgumentError("force_tol must be finite and >= 0"))

    sig_v = sigma isa Vector{Float64} ? sigma : Vector{Float64}(sigma)
    eps_v = epsilon isa Vector{Float64} ? epsilon : Vector{Float64}(epsilon)
    rc = Float64(rcut)
    asc = Float64(alpha_sc)
    lam = Float64(lambda)
    ftol = Float64(force_tol)
    h_max = Float64(max_disp)
    h = min(Float64(step_size), h_max)

    F = Matrix{Float64}(undef, 3, N)
    F_trial = Matrix{Float64}(undef, 3, N)
    q_trial = Matrix{Float64}(undef, 3, N)

    cur = _eval_softcore_linked!(F, q, sig_v, eps_v, box, rc, asc, lam)
    init_e = cur.energy
    init_fmax = cur.max_force

    if cur.max_force <= ftol && cur.n_clashes == 0
        return ClashRelaxationResult(init_e, init_e, init_fmax, init_fmax, 0, true)
    end

    steps_run = 0
    converged = false

    for step in 1:max_steps
        fmax = cur.max_force
        if fmax == 0.0
            converged = (cur.n_clashes == 0)
            break
        end

        steps_run = step
        accepted = false
        for _bt in 1:12
            scale = h / fmax
            @inbounds for j in 1:N
                q_trial[1, j] = q[1, j] + scale * F[1, j]
                q_trial[2, j] = q[2, j] + scale * F[2, j]
                q_trial[3, j] = q[3, j] + scale * F[3, j]
            end
            trial = _eval_softcore_linked!(F_trial, q_trial, sig_v, eps_v, box, rc, asc, lam)
            if isfinite(trial.energy) && trial.energy <= cur.energy
                copyto!(q, q_trial)
                copyto!(F, F_trial)
                cur = trial
                h = min(h * 1.5, h_max)
                accepted = true
                break
            else
                h *= 0.5
                h < 1e-12 && break
            end
        end

        if cur.max_force <= ftol && cur.n_clashes == 0
            converged = true
            break
        end
        if !accepted
            break
        end
    end

    return ClashRelaxationResult(init_e, cur.energy, init_fmax, cur.max_force, steps_run, converged)
end

# ---------------------------------------------------------------------------
# 5. DirtyTileDeltaTracker (Sparse 32-atom Morton Tile Delta Tracking)
# ---------------------------------------------------------------------------

"""
    DirtyTileDeltaTracker(q, threshold; box=NoBox(), tile_size=Neighbors.TILE_SIZE)

Tracks per-atom displacements relative to reference positions and emits the sorted
sparse list of dirty 32-atom Morton tile indices whenever any atom in a tile
exceeds `threshold`.
"""
mutable struct DirtyTileDeltaTracker
    threshold::Float64
    tile_size::Int
    ref_positions::Matrix{Float64}
    morton_perm::Vector{Int32}
    atom_to_tile::Vector{Int32}
    n_tiles::Int
    dirty_flags::BitVector
    dirty_tiles::Vector{Int32}

    function DirtyTileDeltaTracker(
        q::AbstractMatrix{<:Real},
        threshold::Real;
        box::AbstractBox = NoBox(),
        tile_size::Int = Neighbors.TILE_SIZE
    )
        size(q, 1) == 3 || throw(ArgumentError("q must be 3 x N, got $(size(q))"))
        (isfinite(threshold) && threshold >= 0.0) ||
            throw(ArgumentError("threshold must be finite and >= 0, got $threshold"))
        tile_size > 0 || throw(ArgumentError("tile_size must be positive, got $tile_size"))

        N = size(q, 2)
        ref = Matrix{Float64}(q)
        perm = Neighbors.morton_order(ref, box)
        n_tiles = cld(N, tile_size)
        atom_to_tile = Vector{Int32}(undef, N)
        @inbounds for slot in 1:N
            atom_to_tile[perm[slot]] = Int32(div(slot - 1, tile_size) + 1)
        end
        return new(
            Float64(threshold),
            tile_size,
            ref,
            perm,
            atom_to_tile,
            n_tiles,
            falses(n_tiles),
            Int32[]
        )
    end
end

"""
    update_dirty_tiles!(tracker::DirtyTileDeltaTracker, q; box=NoBox(), commit=false) -> Vector{Int32}

Compare `q` against `tracker.ref_positions` (MIC-aware under `box`) and return the
sorted vector of 1-based Morton tile indices containing at least one atom displaced
by strictly more than `tracker.threshold`. If `commit=true`, updates `ref_positions`
for all atoms in the emitted dirty tiles.
"""
function update_dirty_tiles!(
    tracker::DirtyTileDeltaTracker,
    q::AbstractMatrix{<:Real};
    box::AbstractBox = NoBox(),
    commit::Bool = false
)
    size(q) == size(tracker.ref_positions) ||
        throw(ArgumentError("q size $(size(q)) != ref_positions size $(size(tracker.ref_positions))"))
    N = size(q, 2)
    ref = tracker.ref_positions
    a2t = tracker.atom_to_tile
    flags = tracker.dirty_flags
    fill!(flags, false)
    thresh2 = tracker.threshold * tracker.threshold

    @inbounds for i in 1:N
        dx = Float64(q[1, i]) - ref[1, i]
        dy = Float64(q[2, i]) - ref[2, i]
        dz = Float64(q[3, i]) - ref[3, i]
        d = Neighbors.minimum_image(box, dx, dy, dz)
        if norm2(d) > thresh2
            flags[a2t[i]] = true
        end
    end

    out = tracker.dirty_tiles
    empty!(out)
    @inbounds for t in 1:tracker.n_tiles
        if flags[t]
            push!(out, Int32(t))
        end
    end

    if commit && !isempty(out)
        commit_tiles!(tracker, q, out)
    end

    return copy(out)
end

"""
    commit_tiles!(tracker::DirtyTileDeltaTracker, q, tiles) -> tracker

Update `tracker.ref_positions` for all atoms belonging to the given 1-based Morton tile indices.
"""
function commit_tiles!(
    tracker::DirtyTileDeltaTracker,
    q::AbstractMatrix{<:Real},
    tiles::AbstractVector{<:Integer}
)
    size(q) == size(tracker.ref_positions) ||
        throw(ArgumentError("q size $(size(q)) != ref_positions size $(size(tracker.ref_positions))"))
    N = size(q, 2)
    ts = tracker.tile_size
    perm = tracker.morton_perm
    ref = tracker.ref_positions
    @inbounds for t in tiles
        1 <= t <= tracker.n_tiles || throw(BoundsError(1:tracker.n_tiles, t))
        s = (Int(t) - 1) * ts + 1
        e = min(N, Int(t) * ts)
        for slot in s:e
            idx = perm[slot]
            ref[1, idx] = Float64(q[1, idx])
            ref[2, idx] = Float64(q[2, idx])
            ref[3, idx] = Float64(q[3, idx])
        end
    end
    return tracker
end

"""
    rebuild_morton!(tracker::DirtyTileDeltaTracker, q; box=NoBox()) -> tracker

Recompute the 3D Morton ordering and tile mapping from `q` and reset reference positions.
"""
function rebuild_morton!(
    tracker::DirtyTileDeltaTracker,
    q::AbstractMatrix{<:Real};
    box::AbstractBox = NoBox()
)
    size(q) == size(tracker.ref_positions) ||
        throw(ArgumentError("q size $(size(q)) != ref_positions size $(size(tracker.ref_positions))"))
    N = size(q, 2)
    @inbounds for i in 1:N
        tracker.ref_positions[1, i] = Float64(q[1, i])
        tracker.ref_positions[2, i] = Float64(q[2, i])
        tracker.ref_positions[3, i] = Float64(q[3, i])
    end
    perm = Neighbors.morton_order(tracker.ref_positions, box)
    tracker.morton_perm = perm
    ts = tracker.tile_size
    @inbounds for slot in 1:N
        tracker.atom_to_tile[perm[slot]] = Int32(div(slot - 1, ts) + 1)
    end
    fill!(tracker.dirty_flags, false)
    empty!(tracker.dirty_tiles)
    return tracker
end

"""
    tile_atoms(tracker::DirtyTileDeltaTracker, tile_idx::Integer) -> SubArray{Int32}

Return a view of the atom indices belonging to 1-based Morton tile `tile_idx`.
"""
function tile_atoms(tracker::DirtyTileDeltaTracker, tile_idx::Integer)
    1 <= tile_idx <= tracker.n_tiles || throw(BoundsError(1:tracker.n_tiles, tile_idx))
    N = length(tracker.morton_perm)
    s = (Int(tile_idx) - 1) * tracker.tile_size + 1
    e = min(N, Int(tile_idx) * tracker.tile_size)
    return view(tracker.morton_perm, s:e)
end

end # module Multiscale
