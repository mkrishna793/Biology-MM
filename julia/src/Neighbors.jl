"""
Neighbour search for the *spatial* interaction graph.

This is separate from `Types.ChemicalTopology` on purpose (FRAMEWORK.md §3): the
chemical bond graph is static, the nonbonded pair set is a geometric predicate
that must be recomputed as atoms move.

Supported: orthorhombic cells via a cell list, triclinic cells via a lattice-correct
fractional cell list with exact minimum-image convention, and the non-periodic
case via brute-force or k-d scan.

`NeighborList` (also aliased as `VerletList`) stores, for each atom, the atoms
that are within `rc + skin`. The `skin` is a *deliberate* source of buffer:
pairs between `rc` and `rc + skin` are present and contribute zero energy
(the factor handles the cutoff), but a pair that moves from beyond `rc + skin`
to inside `rc` between rebuilds would be missed.

`needs_rebuild!` tracks single-atom displacements AND box deformation (Δh)
since the build so the driver can rebuild before any pair can cross from
outside `rc + skin` to inside `rc` unnoticed.

`PairCache` is the production form of the same idea: a persistent *flat*,
lexicographically sorted `(i, j)` array filled by the same cell traversal.
`refresh_pair_cache!` rebuilds exactly when the same conservative bound is
violated, and the sorted superset makes a cached energy/force evaluation
bit-identical to a from-scratch rebuild at the same geometry.
"""
module Neighbors

using LinearAlgebra
using ..Types
using ..Types: AbstractBox, NoBox, OrthorhombicBox, TriclinicBox, box_volume, box_lengths,
               UnsupportedGeometry, InvalidState, SVector3, norm2

export NeighborList, VerletList, build_neighbors, needs_rebuild!, CellList,
       TriclinicCellList, neighbor_pairs, minimum_image, triclinic_mic,
       PairCache, num_pairs, search_radius, fill_pair_cache!, refresh_pair_cache!,
       pair_cache_stale, pair_cache_diagnostics, pair_cache_displacement,
       TILE_SIZE, morton_codes, morton_order, TilePairCache, num_tile_pairs,
       fill_tile_cache!, refresh_tile_cache!, tile_cache_stale, tile_cache_diagnostics

"""Maximum system size for explicitly materializing a tuple pair vector."""
const MAX_MATERIALIZED_PAIR_ATOMS = 100_000

"""
`lists[j]` = atoms within `rcut + skin` of atom `j`. Pairs are enumerated once via
`j < k` so no pair is evaluated twice.
`build_box` records the simulation box at the time of the build, allowing `needs_rebuild!`
to detect both particle excursion and box deformation.
"""
struct NeighborList
    lists::Vector{Vector{Int}}
    rcut::Float64
    skin::Float64
    build_positions::Matrix{Float64}
    n_rebuilds::Int
    build_box::AbstractBox
end

# 5-argument constructor for backwards compatibility with legacy tests
NeighborList(lists, rcut, skin, build_pos, n_reb) =
    NeighborList(lists, rcut, skin, build_pos, n_reb, NoBox())

"""Verlet list alias for NeighborList."""
const VerletList = NeighborList

# ---------------------------------------------------------------------------
# exact triclinic minimum-image convention (MIC)
# ---------------------------------------------------------------------------

"""
    triclinic_mic(box::TriclinicBox, dx::Float64, dy::Float64, dz::Float64) -> SVector3

Exact minimum-image displacement for a triclinic box with lattice vectors in columns of `box.H`.
Unlike naive componentwise rounding of fractional coordinates (which chooses the wrong image
for skewed non-orthogonal boxes), this algorithm searches the 27 integer lattice shifts in
{-1, 0, 1}³ around the initial rounded fractional point, and verifies Cauchy-Schwarz
bounding spheres to guarantee the globally shortest vector without heap allocation.
"""
@inline function triclinic_mic(box::TriclinicBox, dx::Float64, dy::Float64, dz::Float64)
    # Fractional coordinates f = Hinv * d
    f1 = box.Hinv[1, 1] * dx + box.Hinv[1, 2] * dy + box.Hinv[1, 3] * dz
    f2 = box.Hinv[2, 1] * dx + box.Hinv[2, 2] * dy + box.Hinv[2, 3] * dz
    f3 = box.Hinv[3, 1] * dx + box.Hinv[3, 2] * dy + box.Hinv[3, 3] * dz

    # Initial candidate from rounded fractional coordinates
    n01 = round(Int, f1)
    n02 = round(Int, f2)
    n03 = round(Int, f3)

    rx0 = dx - (box.H[1, 1] * n01 + box.H[1, 2] * n02 + box.H[1, 3] * n03)
    ry0 = dy - (box.H[2, 1] * n01 + box.H[2, 2] * n02 + box.H[2, 3] * n03)
    rz0 = dz - (box.H[3, 1] * n01 + box.H[3, 2] * n02 + box.H[3, 3] * n03)

    best2 = rx0 * rx0 + ry0 * ry0 + rz0 * rz0
    best_rx = rx0
    best_ry = ry0
    best_rz = rz0

    # Search 27 integer lattice shifts in {-1, 0, 1} around n0
    for dc in -1:1
        nc = n03 + dc
        H13_nc = box.H[1, 3] * nc
        H23_nc = box.H[2, 3] * nc
        H33_nc = box.H[3, 3] * nc
        for db in -1:1
            nb = n02 + db
            H12_nb = box.H[1, 2] * nb + H13_nc
            H22_nb = box.H[2, 2] * nb + H23_nc
            H32_nb = box.H[3, 2] * nb + H33_nc
            for da in -1:1
                (da == 0 && db == 0 && dc == 0) && continue
                na = n01 + da
                rx = dx - (box.H[1, 1] * na + H12_nb)
                ry = dy - (box.H[2, 1] * na + H22_nb)
                rz = dz - (box.H[3, 1] * na + H32_nb)
                r2 = rx * rx + ry * ry + rz * rz
                if r2 < best2
                    best2 = r2
                    best_rx = rx
                    best_ry = ry
                    best_rz = rz
                end
            end
        end
    end

    # Cauchy-Schwarz bound check: any image satisfying ‖d - H n‖ < sqrt(best2)
    # must have |f_k - n_k| <= ‖(Hinv)_k‖ * sqrt(best2).
    # If the Cauchy-Schwarz range exceeds {-1, 0, 1} around n0 (which can happen
    # for extreme skewness), exhaustively search the outer range.
    best_dist = sqrt(best2)
    bnd1 = sqrt(box.Hinv[1, 1]^2 + box.Hinv[1, 2]^2 + box.Hinv[1, 3]^2) * best_dist
    bnd2 = sqrt(box.Hinv[2, 1]^2 + box.Hinv[2, 2]^2 + box.Hinv[2, 3]^2) * best_dist
    bnd3 = sqrt(box.Hinv[3, 1]^2 + box.Hinv[3, 2]^2 + box.Hinv[3, 3]^2) * best_dist

    r1_min = ceil(Int, f1 - bnd1); r1_max = floor(Int, f1 + bnd1)
    r2_min = ceil(Int, f2 - bnd2); r2_max = floor(Int, f2 + bnd2)
    r3_min = ceil(Int, f3 - bnd3); r3_max = floor(Int, f3 + bnd3)

    if (r1_min < n01 - 1 || r1_max > n01 + 1 ||
        r2_min < n02 - 1 || r2_max > n02 + 1 ||
        r3_min < n03 - 1 || r3_max > n03 + 1)
        for nc in r3_min:r3_max
            H13_nc = box.H[1, 3] * nc
            H23_nc = box.H[2, 3] * nc
            H33_nc = box.H[3, 3] * nc
            for nb in r2_min:r2_max
                H12_nb = box.H[1, 2] * nb + H13_nc
                H22_nb = box.H[2, 2] * nb + H23_nc
                H32_nb = box.H[3, 2] * nb + H33_nc
                for na in r1_min:r1_max
                    # Skip the inner 3x3x3 cube already searched
                    if (abs(na - n01) <= 1 && abs(nb - n02) <= 1 && abs(nc - n03) <= 1)
                        continue
                    end
                    rx = dx - (box.H[1, 1] * na + H12_nb)
                    ry = dy - (box.H[2, 1] * na + H22_nb)
                    rz = dz - (box.H[3, 1] * na + H32_nb)
                    r2 = rx * rx + ry * ry + rz * rz
                    if r2 < best2
                        best2 = r2
                        best_rx = rx
                        best_ry = ry
                        best_rz = rz
                    end
                end
            end
        end
    end

    return SVector3(best_rx, best_ry, best_rz)
end

@inline function triclinic_mic(box::TriclinicBox, d::AbstractVector{<:Real})
    return triclinic_mic(box, Float64(d[1]), Float64(d[2]), Float64(d[3]))
end

"""
    minimum_image(box, dx, dy, dz)
    minimum_image(box, d)
    minimum_image(box, q, i, j)

Return the minimum-image displacement vector as `SVector3`.
"""
@inline function minimum_image(box::TriclinicBox, dx::Real, dy::Real, dz::Real)
    return triclinic_mic(box, Float64(dx), Float64(dy), Float64(dz))
end

@inline function minimum_image(box::TriclinicBox, d::AbstractVector{<:Real})
    return triclinic_mic(box, Float64(d[1]), Float64(d[2]), Float64(d[3]))
end

@inline function minimum_image(box::OrthorhombicBox, dx::Real, dy::Real, dz::Real)
    L = box.L
    x = Float64(dx) - L[1] * round(Float64(dx) / L[1])
    y = Float64(dy) - L[2] * round(Float64(dy) / L[2])
    z = Float64(dz) - L[3] * round(Float64(dz) / L[3])
    return SVector3(x, y, z)
end

@inline function minimum_image(box::OrthorhombicBox, d::AbstractVector{<:Real})
    return minimum_image(box, d[1], d[2], d[3])
end

@inline function minimum_image(box::NoBox, dx::Real, dy::Real, dz::Real)
    return SVector3(Float64(dx), Float64(dy), Float64(dz))
end

@inline function minimum_image(box::NoBox, d::AbstractVector{<:Real})
    return SVector3(Float64(d[1]), Float64(d[2]), Float64(d[3]))
end

@inline function _displacement(box::TriclinicBox, q::AbstractMatrix{Float64}, k::Int, j::Int)
    dx = q[1, k] - q[1, j]
    dy = q[2, k] - q[2, j]
    dz = q[3, k] - q[3, j]
    return triclinic_mic(box, dx, dy, dz)
end

@inline function _displacement(box::AbstractBox, q::AbstractMatrix{Float64}, k::Int, j::Int)
    return Types.unwrap_displacement(box, q, k, j)
end

@inline function minimum_image(box::AbstractBox, q::AbstractMatrix{Float64}, i::Int, j::Int)
    return _displacement(box, q, i, j)
end

# ---------------------------------------------------------------------------
# cell lists
# ---------------------------------------------------------------------------

"Cell-list buckets for an orthorhombic box."
struct CellList
    nc::NTuple{3,Int}
    cell_of::Vector{Int}
    members::Vector{Vector{Int}}
    L::NTuple{3,Float64}
    cell_size::NTuple{3,Float64}
end

"Cell-list buckets for a triclinic box using fractional coordinate binning."
struct TriclinicCellList
    nc::NTuple{3,Int}
    cell_of::Vector{Int}
    members::Vector{Vector{Int}}
end

function _copy_box(b::NoBox)
    return NoBox()
end
function _copy_box(b::OrthorhombicBox)
    return OrthorhombicBox(b.L)
end
function _copy_box(b::TriclinicBox)
    return TriclinicBox(copy(b.H))
end
function _copy_box(b::AbstractBox)
    return b
end

function build_neighbors(q::AbstractMatrix{Float64}, box::AbstractBox,
                         rcut::Float64, skin::Float64;
                         min_cell::Float64 = 0.0, n_rebuilds::Int = 0)
    N = size(q, 2)
    N <= MAX_MATERIALIZED_PAIR_ATOMS ||
        throw(ArgumentError("build_neighbors materializes per-atom Vector{Vector} " *
                            "storage and is disabled above N=$MAX_MATERIALIZED_PAIR_ATOMS " *
                            "(N=$N); use a streamed cell traversal"))
    rsearch = rcut + skin
    (isfinite(rcut) && rcut > 0 && isfinite(skin) && skin >= 0) ||
        throw(ArgumentError("rcut must be finite and positive and skin finite and non-negative"))
    if box isa OrthorhombicBox || box isa TriclinicBox
        widths = box isa OrthorhombicBox ? box.L : box.widths
        2rcut < minimum(widths) ||
            throw(ArgumentError("rcut must be less than half the minimum perpendicular box width"))
        2rsearch < minimum(widths) ||
            throw(ArgumentError("rcut + skin must be less than half the minimum perpendicular box width"))
    end
    if box isa AbstractBox && !(box isa NoBox) && !(box isa OrthorhombicBox) && !(box isa TriclinicBox)
        throw(UnsupportedGeometry("neighbor search does not support $(typeof(box))"))
    end
    lists = [Int[] for _ in 1:N]
    if box isa OrthorhombicBox
        cl = make_cell_list(q, box, rsearch)
        _fill_from_cells!(lists, q, box, cl, rsearch)
    elseif box isa TriclinicBox
        cl = make_triclinic_cell_list(q, box, rsearch)
        _fill_from_triclinic_cells!(lists, q, box, cl, rsearch)
    else
        _fill_bruteforce!(lists, q, box, rsearch)
    end
    build_box = _copy_box(box)
    return NeighborList(lists, rcut, skin, copy(q), n_rebuilds, build_box)
end

function make_cell_list(q::AbstractMatrix{Float64}, box::OrthorhombicBox, rsearch::Float64)
    L = box.L
    nc = ntuple(d -> max(1, floor(Int, L[d] / rsearch)), 3)
    # A cell must be at least rsearch wide for the 27-cell stencil to be valid.
    # When the box is too small for that, fall back to nc = 1 (the stencil then
    # covers the whole box, which is correct but is brute force).
    cs = ntuple(d -> L[d] / nc[d], 3)
    cell_of = Vector{Int}(undef, size(q, 2))
    members = [Int[] for _ in 1:(nc[1] * nc[2] * nc[3])]
    @inbounds for j in axes(q, 2)
        c1 = _bucket(q[1, j], L[1], nc[1])
        c2 = _bucket(q[2, j], L[2], nc[2])
        c3 = _bucket(q[3, j], L[3], nc[3])
        id = (c3 - 1) * nc[1] * nc[2] + (c2 - 1) * nc[1] + c1
        cell_of[j] = id
        push!(members[id], j)
    end
    return CellList(nc, cell_of, members, L, cs)
end

@inline function _bucket(x::Float64, L::Float64, n::Int)
    # wrap into [0, L) first so atoms outside the primary cell bucket correctly
    xx = x - L * floor(x / L)
    return min(n, max(1, floor(Int, xx / L * n) + 1))
end

function make_triclinic_cell_list(q::AbstractMatrix{Float64}, box::TriclinicBox, rsearch::Float64)
    # The cell must be at least rsearch wide in each perpendicular direction for the
    # 27-cell stencil in fractional coordinates to cover all pairs within rsearch.
    nc = ntuple(d -> max(1, floor(Int, box.widths[d] / rsearch)), 3)
    N = size(q, 2)
    cell_of = Vector{Int}(undef, N)
    ncells = nc[1] * nc[2] * nc[3]
    members = [Int[] for _ in 1:ncells]

    @inbounds for j in 1:N
        # Fractional coordinates s = Hinv * q
        s1 = box.Hinv[1, 1] * q[1, j] + box.Hinv[1, 2] * q[2, j] + box.Hinv[1, 3] * q[3, j]
        s2 = box.Hinv[2, 1] * q[1, j] + box.Hinv[2, 2] * q[2, j] + box.Hinv[2, 3] * q[3, j]
        s3 = box.Hinv[3, 1] * q[1, j] + box.Hinv[3, 2] * q[2, j] + box.Hinv[3, 3] * q[3, j]

        # Wrap into [0, 1)
        f1 = s1 - floor(s1)
        f2 = s2 - floor(s2)
        f3 = s3 - floor(s3)

        c1 = min(nc[1], max(1, floor(Int, f1 * nc[1]) + 1))
        c2 = min(nc[2], max(1, floor(Int, f2 * nc[2]) + 1))
        c3 = min(nc[3], max(1, floor(Int, f3 * nc[3]) + 1))

        id = (c3 - 1) * nc[1] * nc[2] + (c2 - 1) * nc[1] + c1
        cell_of[j] = id
        push!(members[id], j)
    end
    return TriclinicCellList(nc, cell_of, members)
end

"""
Distinct target cells reachable from a cell with a radius-1 stencil.

The 27-entry `(-1:1)³` offset stencil is only 27 distinct cells when every axis
has ≥ 3 cells. With 2 cells per axis, `mod1(c+1,2)` and `mod1(c-1,2)` are the
*same* cell, so a naive triple loop visits that cell twice — and every pair inside
it is then counted twice. Dedup must therefore be on the **resulting cell index**,
not on the offset triple.

This was a real bug caught by NIST validation: config 2 (box 8, rc 3 → 2 cells per
axis) and config 4 failed while config 1 and 3 (3 cells) passed, which is exactly
the signature of a cell-count-dependent over-count.

Deduplicating cells is necessary and sufficient. For `nc < 3` the distinct set
covers all cells, which is correct (and reduces to brute force in that dimension,
the honest behaviour when a cell is wider than the search radius).
"""
function _neighbor_cells(c1::Int, c2::Int, c3::Int, nc::NTuple{3,Int})
    seen = Set{Int}()
    out = Int[]
    for dz in -1:1, dy in -1:1, dx in -1:1
        d1 = mod1(c1 + dx, nc[1])
        d2 = mod1(c2 + dy, nc[2])
        d3 = mod1(c3 + dz, nc[3])
        id = (d3 - 1) * nc[1] * nc[2] + (d2 - 1) * nc[1] + d1
        id in seen && continue
        push!(seen, id)
        push!(out, id)
    end
    return out
end

function _fill_from_cells!(lists, q, box::OrthorhombicBox, cl::CellList, rsearch::Float64)
    nc = cl.nc
    r2cut = rsearch^2
    L = cl.L
    @inbounds for j in axes(q, 2)
        c1 = _bucket(q[1, j], L[1], nc[1])
        c2 = _bucket(q[2, j], L[2], nc[2])
        c3 = _bucket(q[3, j], L[3], nc[3])
        for id in _neighbor_cells(c1, c2, c3, nc)
            for k in cl.members[id]
                k <= j && continue
                dd = _displacement(box, q, k, j)
                if norm2(dd) <= r2cut
                    push!(lists[j], k)
                end
            end
        end
    end
    return lists
end

function _fill_from_triclinic_cells!(lists, q, box::TriclinicBox, cl::TriclinicCellList, rsearch::Float64)
    nc = cl.nc
    r2cut = rsearch^2
    N = size(q, 2)
    @inbounds for j in 1:N
        cid = cl.cell_of[j]
        c1 = mod1(cid, nc[1])
        rem1 = div(cid - c1, nc[1])
        c2 = mod1(rem1 + 1, nc[2])
        c3 = div(rem1 + 1 - c2, nc[2]) + 1

        for id in _neighbor_cells(c1, c2, c3, nc)
            for k in cl.members[id]
                k <= j && continue
                dd = _displacement(box, q, k, j)
                if norm2(dd) <= r2cut
                    push!(lists[j], k)
                end
            end
        end
    end
    return lists
end

function _fill_bruteforce!(lists, q, box, rsearch::Float64)
    N = size(q, 2)
    r2cut = rsearch^2
    @inbounds for j in 1:N-1
        for k in j+1:N
            dd = _displacement(box, q, k, j)
            if norm2(dd) <= r2cut
                push!(lists[j], k)
            end
        end
    end
    return lists
end

# ---------------------------------------------------------------------------
# Verlet rebuild criteria: particle excursion + box deformation (Δh)
# ---------------------------------------------------------------------------

"""
Compute the maximum displacement shift ‖Δh n‖ across all 26 integer image shifts
`n ∈ {-1, 0, 1}³ \\ {0}` caused by box deformation between `b0` and `b1`.
"""
function _max_lattice_deformation(b0::AbstractBox, b1::AbstractBox)::Float64
    if b0 isa NoBox && b1 isa NoBox
        return 0.0
    elseif b0 isa NoBox || b1 isa NoBox
        return Inf
    elseif b0 isa OrthorhombicBox && b1 isa OrthorhombicBox
        dL1 = b1.L[1] - b0.L[1]
        dL2 = b1.L[2] - b0.L[2]
        dL3 = b1.L[3] - b0.L[3]
        return sqrt(dL1 * dL1 + dL2 * dL2 + dL3 * dL3)
    else
        # Extract 3x3 lattice matrices
        H0 = b0 isa TriclinicBox ? b0.H : diagm(0 => [b0.L[1], b0.L[2], b0.L[3]])
        H1 = b1 isa TriclinicBox ? b1.H : diagm(0 => [b1.L[1], b1.L[2], b1.L[3]])
        dH11 = H1[1, 1] - H0[1, 1]; dH12 = H1[1, 2] - H0[1, 2]; dH13 = H1[1, 3] - H0[1, 3]
        dH21 = H1[2, 1] - H0[2, 1]; dH22 = H1[2, 2] - H0[2, 2]; dH23 = H1[2, 3] - H0[2, 3]
        dH31 = H1[3, 1] - H0[3, 1]; dH32 = H1[3, 2] - H0[3, 2]; dH33 = H1[3, 3] - H0[3, 3]

        max_dh = 0.0
        for n3 in -1:1, n2 in -1:1, n1 in -1:1
            (n1 == 0 && n2 == 0 && n3 == 0) && continue
            v1 = dH11 * n1 + dH12 * n2 + dH13 * n3
            v2 = dH21 * n1 + dH22 * n2 + dH23 * n3
            v3 = dH31 * n1 + dH32 * n2 + dH33 * n3
            norm_v = sqrt(v1 * v1 + v2 * v2 + v3 * v3)
            if norm_v > max_dh
                max_dh = norm_v
            end
        end
        return max_dh
    end
end

"""
    needs_rebuild!(nl::NeighborList, q::AbstractMatrix{Float64}, [box::AbstractBox])

Verlet rebuild criterion: returns `true` if any particle displacement OR box deformation
exceeds the skin safety envelope.

Specifically:
- Box deformation creates image shifts bounded by `dh_max = max_{n ∈ {-1,0,1}³} ‖Δh n‖`.
- Particle displacement reduces the distance between any pair by at most `2 * max_j ‖Δq_j‖`.
- Together, `2 * max_j ‖Δq_j‖ + dh_max <= skin` guarantees that NO pair outside the
  cutoff sphere `rcut + skin` can enter `rcut` unnoticed.
- Therefore, if `dh_max >= skin` or any atom displacement exceeds `(skin - dh_max) / 2`,
  a rebuild is strictly required.
"""
function needs_rebuild!(nl::NeighborList, q::AbstractMatrix{Float64}, box::AbstractBox = nl.build_box)
    dh_max = _max_lattice_deformation(nl.build_box, box)
    if !isfinite(dh_max) || dh_max >= nl.skin
        return true
    end

    # Check if box shrunk below minimum perpendicular width limit
    if box isa OrthorhombicBox || box isa TriclinicBox
        w = box isa OrthorhombicBox ? box.L : box.widths
        if 2.0 * nl.rcut >= minimum(w)
            return true
        end
    end

    thresh = 0.5 * (nl.skin - dh_max)
    th2 = thresh * thresh
    @inbounds for j in axes(q, 2)
        dx = q[1, j] - nl.build_positions[1, j]
        dy = q[2, j] - nl.build_positions[2, j]
        dz = q[3, j] - nl.build_positions[3, j]
        (dx * dx + dy * dy + dz * dz) > th2 && return true
    end
    return false
end

needs_rebuild!(nl::NeighborList, q::AbstractMatrix{Float64}, H::AbstractMatrix{<:Real}) =
    needs_rebuild!(nl, q, TriclinicBox(H))

needs_rebuild!(nl::NeighborList, q::AbstractMatrix{Float64}, L::Union{NTuple{3,Float64}, AbstractVector{<:Real}}) =
    needs_rebuild!(nl, q, OrthorhombicBox(L))

# ---------------------------------------------------------------------------
# persistent flat pair cache (the production neighbour list)
# ---------------------------------------------------------------------------

"""
`PairCache` — a persistent, position-cached neighbour list in flat pair-major form.

`pairs_i[k] < pairs_j[k]`, and the array is sorted lexicographically by
`(pairs_i, pairs_j)`. That canonical order is what makes a cached evaluation
*bitwise* identical to a from-scratch one at the same geometry: a pair stored
only because of the skin contributes exactly `+0.0` to the energy and `+0.0` to
both force components (the pair kernel returns before touching the accumulator
when `r > rcut`), and inserting an exact `+0.0` term anywhere in a sum leaves
every partial sum bit-identical. So the cached pair array is a superset whose
extra terms are additive identities, enumerated in the same relative order as
the rebuild would enumerate the subset.

Rebuild criterion for the persistent cache:

* Let `d0 = rcut + skin` and let the list have been built at `(q0, h0)`.
* For a pair that was *not* stored, `r0 > d0`.
* Particle motion changes either endpoint by at most `max_j ‖q_j - q0_j‖`, so
  the pair distance can shrink by at most `2 max_j ‖Δq_j‖`.
* For a changed lattice, let `A = h₁ h₀⁻¹`, `s = σ_min(A)`, and let `δ` be the
  largest current minimum-image residual from the affine prediction `A q₀`.
  Then every image satisfies `r₁ ≥ s r₀ - 2δ`, so reuse is safe when
  `s (rcut + skin) - 2δ ≥ rcut`. This handles skew boxes and arbitrary
  unwrapped coordinates; whole-box winding has zero periodic residual.

The cache uses this affine/singular-value bound plus the minimum-width stencil
guard. It rebuilds only when the resulting slack is negative, rather than on
every lattice change.
"""
mutable struct PairCache
    pairs_i::Vector{Int32}
    pairs_j::Vector{Int32}
    rcut::Float64
    skin::Float64
    build_positions::Matrix{Float64}
    build_box::AbstractBox
    n_rebuilds::Int
    n_reuses::Int
end

"Number of stored pairs (including the skin buffer)."
num_pairs(pc::PairCache) = length(pc.pairs_i)

"Search radius used to fill the cache: `rcut + skin`."
search_radius(pc::PairCache) = pc.rcut + pc.skin

function _validate_pair_cache_geometry(q::AbstractMatrix{Float64}, box::AbstractBox,
                                      rcut::Float64, skin::Float64)
    size(q, 1) == 3 || throw(ArgumentError("q must be 3 x N, got $(size(q))"))
    (isfinite(rcut) && rcut > 0) ||
        throw(ArgumentError("rcut must be finite and positive, got $rcut"))
    (isfinite(skin) && skin >= 0) ||
        throw(ArgumentError("skin must be finite and non-negative, got $skin"))
    if box isa OrthorhombicBox || box isa TriclinicBox
        w = box isa OrthorhombicBox ? box.L : box.widths
        2.0 * rcut < minimum(w) ||
            throw(ArgumentError("rcut must be less than half the minimum perpendicular " *
                                "box width"))
        2.0 * (rcut + skin) < minimum(w) ||
            throw(ArgumentError("rcut + skin must be less than half the minimum " *
                                "perpendicular box width"))
    elseif !(box isa NoBox)
        throw(UnsupportedGeometry("pair cache does not support $(typeof(box))"))
    end
    return nothing
end

"Maximum `‖q_j - q0_j‖` since the build (0 for an empty cache)."
function pair_cache_displacement(pc::PairCache, q::AbstractMatrix{Float64})
    m = 0.0
    b = pc.build_positions
    @inbounds for j in axes(q, 2)
        dx = q[1, j] - b[1, j]
        dy = q[2, j] - b[2, j]
        dz = q[3, j] - b[3, j]
        d2 = dx * dx + dy * dy + dz * dz
        d2 > m * m && (m = sqrt(d2))
    end
    return m
end

"""
    pair_cache_diagnostics(pc, q, [box]) -> (; max_displacement, box_deformation,
                                              slack, stale, reason)

Expose the exact quantities the rebuild test compares against `skin`, so a test
(or a log) can show the bound is applied and not approximated.
"""
function _pair_cache_geometry_bound(pc::PairCache, q::AbstractMatrix{Float64},
                                    box::AbstractBox)
    b0 = pc.build_box
    b0 isa NoBox && box isa NoBox && return (1.0, pair_cache_displacement(pc, q))
    ((b0 isa OrthorhombicBox && box isa OrthorhombicBox) ||
     (b0 isa TriclinicBox && box isa TriclinicBox)) || return (0.0, Inf)
    if b0 isa OrthorhombicBox && box isa OrthorhombicBox && b0.L == box.L
        L = box.L
        delta = 0.0
        @inbounds for j in axes(q, 2)
            dx = q[1, j] - pc.build_positions[1, j]
            dy = q[2, j] - pc.build_positions[2, j]
            dz = q[3, j] - pc.build_positions[3, j]
            dx -= L[1] * round(dx / L[1])
            dy -= L[2] * round(dy / L[2])
            dz -= L[3] * round(dz / L[3])
            delta = max(delta, sqrt(dx * dx + dy * dy + dz * dz))
        end
        return 1.0, delta
    end
    H0 = b0 isa OrthorhombicBox ? diagm(0 => collect(b0.L)) : b0.H
    H1 = box isa OrthorhombicBox ? diagm(0 => collect(box.L)) : box.H
    A = H1 / H0
    scale = minimum(svdvals(A))
    delta = 0.0
    @inbounds for j in axes(q, 2)
        b = pc.build_positions
        dx = q[1, j] - (A[1, 1] * b[1, j] + A[1, 2] * b[2, j] + A[1, 3] * b[3, j])
        dy = q[2, j] - (A[2, 1] * b[1, j] + A[2, 2] * b[2, j] + A[2, 3] * b[3, j])
        dz = q[3, j] - (A[3, 1] * b[1, j] + A[3, 2] * b[2, j] + A[3, 3] * b[3, j])
        if box isa OrthorhombicBox
            L = box.L
            dx -= L[1] * round(dx / L[1])
            dy -= L[2] * round(dy / L[2])
            dz -= L[3] * round(dz / L[3])
            delta = max(delta, sqrt(dx * dx + dy * dy + dz * dz))
        else
            d = triclinic_mic(box, dx, dy, dz)
            delta = max(delta, sqrt(norm2(d)))
        end
    end
    return scale, delta
end

function pair_cache_diagnostics(pc::PairCache, q::AbstractMatrix{Float64},
                                box::AbstractBox = pc.build_box)
    size(q) == size(pc.build_positions) ||
        throw(ArgumentError("q must be 3 x $(size(pc.build_positions, 2)) for this pair cache"))
    scale, delta = _pair_cache_geometry_bound(pc, q, box)
    disp = pair_cache_displacement(pc, q)
    width_ok = if box isa OrthorhombicBox || box isa TriclinicBox
        w = box isa OrthorhombicBox ? box.L : box.widths
        2.0 * search_radius(pc) < minimum(w)
    else
        true
    end
    slack = scale * search_radius(pc) - 2.0 * delta - pc.rcut
    reason = !isfinite(scale) || !isfinite(delta) ? :box_type_changed :
             !width_ok ? :box_too_small :
             slack < 0.0 ? :displacement : :none
    deformation = (1.0 - scale) * search_radius(pc) + 2.0 * delta
    return (max_displacement = disp, affine_displacement = delta,
            lattice_scale = scale, box_deformation = deformation, slack = slack,
            stale = reason !== :none, reason = reason)
end

"""
    pair_cache_stale(pc, q, [box]) -> Bool

`true` when the cached list can no longer be proven complete for `(q, box)`.
"""
pair_cache_stale(pc::PairCache, q::AbstractMatrix{Float64},
                 box::AbstractBox = pc.build_box) =
    pair_cache_diagnostics(pc, q, box).stale

# --- construction -----------------------------------------------------------

function _append_pairs_from_cells!(pc::PairCache, q, box::OrthorhombicBox,
                                   cl::CellList, rsearch::Float64)
    nc = cl.nc
    L = cl.L
    r2cut = rsearch^2
    pi, pj = pc.pairs_i, pc.pairs_j
    @inbounds for j in axes(q, 2)
        c1 = _bucket(q[1, j], L[1], nc[1])
        c2 = _bucket(q[2, j], L[2], nc[2])
        c3 = _bucket(q[3, j], L[3], nc[3])
        for id in _neighbor_cells(c1, c2, c3, nc)
            for k in cl.members[id]
                k <= j && continue
                dd = _displacement(box, q, k, j)
                if norm2(dd) <= r2cut
                    push!(pi, Int32(j))
                    push!(pj, Int32(k))
                end
            end
        end
    end
    return nothing
end

function _append_pairs_from_cells!(pc::PairCache, q, box::TriclinicBox,
                                   cl::TriclinicCellList, rsearch::Float64)
    nc = cl.nc
    r2cut = rsearch^2
    pi, pj = pc.pairs_i, pc.pairs_j
    @inbounds for j in axes(q, 2)
        cid = cl.cell_of[j]
        c1 = mod1(cid, nc[1])
        rem1 = div(cid - c1, nc[1])
        c2 = mod1(rem1 + 1, nc[2])
        c3 = div(rem1 + 1 - c2, nc[2]) + 1
        for id in _neighbor_cells(c1, c2, c3, nc)
            for k in cl.members[id]
                k <= j && continue
                dd = _displacement(box, q, k, j)
                if norm2(dd) <= r2cut
                    push!(pi, Int32(j))
                    push!(pj, Int32(k))
                end
            end
        end
    end
    return nothing
end

function _append_pairs_bruteforce!(pc::PairCache, q, box::AbstractBox, rsearch::Float64)
    N = size(q, 2)
    r2cut = rsearch^2
    pi, pj = pc.pairs_i, pc.pairs_j
    @inbounds for j in 1:N-1
        for k in j+1:N
            dd = _displacement(box, q, k, j)
            if norm2(dd) <= r2cut
                push!(pi, Int32(j))
                push!(pj, Int32(k))
            end
        end
    end
    return nothing
end

# Pairs are appended grouped by ascending `j`, so each block is contiguous and
# only its `k` values need sorting. Sorting at build time is what makes the
# order a function of the pair *set* alone, independent of which cells the
# atoms happened to occupy when the list was built.
function _sort_pair_blocks!(pc::PairCache)
    pj = pc.pairs_j
    n = length(pj)
    n == 0 && return nothing
    i0 = 1
    last_i = pc.pairs_i[1]
    @inbounds for k in 2:n
        cur = pc.pairs_i[k]
        if cur != last_i
            if k - 1 > i0
                sort!(view(pj, i0:(k - 1)))
            end
            i0 = k
            last_i = cur
        end
    end
    if n > i0
        sort!(view(pj, i0:n))
    end
    return nothing
end

"""
    fill_pair_cache!(pc, q, box) -> pc

Rebuild `pc` in place from the current geometry using the streamed cell
traversal (cell list + 27-cell stencil) and the exact minimum-image convention.
The box and the search radius are validated first, so a failed rebuild leaves
the previous pair array intact and the object unchanged.
"""
function fill_pair_cache!(pc::PairCache, q::AbstractMatrix{Float64}, box::AbstractBox)
    _validate_pair_cache_geometry(q, box, pc.rcut, pc.skin)
    size(q, 2) == size(pc.build_positions, 2) ||
        throw(ArgumentError("q has $(size(q, 2)) atoms but the cache was built for " *
                            "$(size(pc.build_positions, 2))"))
    rsearch = search_radius(pc)
    empty!(pc.pairs_i)
    empty!(pc.pairs_j)
    if box isa OrthorhombicBox
        cl = make_cell_list(q, box, rsearch)
        _append_pairs_from_cells!(pc, q, box, cl, rsearch)
    elseif box isa TriclinicBox
        cl = make_triclinic_cell_list(q, box, rsearch)
        _append_pairs_from_cells!(pc, q, box, cl, rsearch)
    else
        _append_pairs_bruteforce!(pc, q, box, rsearch)
    end
    _sort_pair_blocks!(pc)
    copyto!(pc.build_positions, q)
    pc.build_box = _copy_box(box)
    pc.n_rebuilds += 1
    return pc
end

"""
    PairCache(q, box, rcut, skin) -> PairCache

Build a persistent neighbour list for pairs within `rcut + skin`. Sorted
lexicographically by `(i, j)`; see the type docstring for the completeness
bound.
"""
function PairCache(q::AbstractMatrix{Float64}, box::AbstractBox,
                   rcut::Float64, skin::Float64)
    _validate_pair_cache_geometry(q, box, rcut, skin)
    N = size(q, 2)
    pc = PairCache(Int32[], Int32[], Float64(rcut), Float64(skin), copy(q),
                   _copy_box(box), 0, 0)
    return fill_pair_cache!(pc, q, box)
end

"""
    refresh_pair_cache!(pc, q, [box]) -> Bool

Rebuild only when [`pair_cache_stale`](@ref) says the list is no longer provably
complete; otherwise count a reuse. Returns `true` when a rebuild happened.
"""
function refresh_pair_cache!(pc::PairCache, q::AbstractMatrix{Float64},
                             box::AbstractBox = pc.build_box)
    if pair_cache_stale(pc, q, box)
        fill_pair_cache!(pc, q, box)
        return true
    end
    pc.n_reuses += 1
    return false
end

# ---------------------------------------------------------------------------
# Morton (Z-order) spatial curve & 32-atom TilePairCache (O(N/32) cluster list)
# ---------------------------------------------------------------------------

"""Number of atoms per spatial tile cluster."""
const TILE_SIZE = 32

@inline function _expand_bits_21(v::UInt64)
    x = v & 0x00000000001fffff
    x = (x | (x << 32)) & 0x001f00000000ffff
    x = (x | (x << 16)) & 0x001f0000ff0000ff
    x = (x | (x << 8))  & 0x100f00f00f00f00f
    x = (x | (x << 4))  & 0x10c30c30c30c30c3
    x = (x | (x << 2))  & 0x1249249249249249
    return x
end

@inline function _morton3d(ix::UInt64, iy::UInt64, iz::UInt64)
    return _expand_bits_21(ix) | (_expand_bits_21(iy) << 1) | (_expand_bits_21(iz) << 2)
end

"""
    morton_codes(q, box) -> Vector{UInt64}

Compute 63-bit 3D Morton (Z-order) spatial keys for `3 × N` coordinates `q`.
For periodic boxes, fractional coordinates are wrapped into `[0, 1)`; for `NoBox`,
coordinates are normalized to the bounding box of `q`.
"""
function morton_codes(q::AbstractMatrix{<:Real}, box::AbstractBox)
    size(q, 1) == 3 || throw(ArgumentError("q must be 3 x N, got $(size(q))"))
    N = size(q, 2)
    codes = Vector{UInt64}(undef, N)
    N == 0 && return codes
    scale = Float64((1 << 21) - 1)
    if box isa OrthorhombicBox
        L = box.L
        @inbounds for j in 1:N
            fx = (Float64(q[1, j]) / L[1]); fx -= floor(fx)
            fy = (Float64(q[2, j]) / L[2]); fy -= floor(fy)
            fz = (Float64(q[3, j]) / L[3]); fz -= floor(fz)
            ix = UInt64(clamp(floor(Int, fx * scale), 0, (1 << 21) - 1))
            iy = UInt64(clamp(floor(Int, fy * scale), 0, (1 << 21) - 1))
            iz = UInt64(clamp(floor(Int, fz * scale), 0, (1 << 21) - 1))
            codes[j] = _morton3d(ix, iy, iz)
        end
    elseif box isa TriclinicBox
        Hinv = box.Hinv
        @inbounds for j in 1:N
            q1 = Float64(q[1, j]); q2 = Float64(q[2, j]); q3 = Float64(q[3, j])
            fx = Hinv[1, 1]*q1 + Hinv[1, 2]*q2 + Hinv[1, 3]*q3; fx -= floor(fx)
            fy = Hinv[2, 1]*q1 + Hinv[2, 2]*q2 + Hinv[2, 3]*q3; fy -= floor(fy)
            fz = Hinv[3, 1]*q1 + Hinv[3, 2]*q2 + Hinv[3, 3]*q3; fz -= floor(fz)
            ix = UInt64(clamp(floor(Int, fx * scale), 0, (1 << 21) - 1))
            iy = UInt64(clamp(floor(Int, fy * scale), 0, (1 << 21) - 1))
            iz = UInt64(clamp(floor(Int, fz * scale), 0, (1 << 21) - 1))
            codes[j] = _morton3d(ix, iy, iz)
        end
    else
        xmin = minimum(view(q, 1, :)); xmax = maximum(view(q, 1, :))
        ymin = minimum(view(q, 2, :)); ymax = maximum(view(q, 2, :))
        zmin = minimum(view(q, 3, :)); zmax = maximum(view(q, 3, :))
        dx = max(xmax - xmin, 1e-12)
        dy = max(ymax - ymin, 1e-12)
        dz = max(zmax - zmin, 1e-12)
        @inbounds for j in 1:N
            fx = clamp((Float64(q[1, j]) - xmin) / dx, 0.0, 1.0)
            fy = clamp((Float64(q[2, j]) - ymin) / dy, 0.0, 1.0)
            fz = clamp((Float64(q[3, j]) - zmin) / dz, 0.0, 1.0)
            ix = UInt64(clamp(floor(Int, fx * scale), 0, (1 << 21) - 1))
            iy = UInt64(clamp(floor(Int, fy * scale), 0, (1 << 21) - 1))
            iz = UInt64(clamp(floor(Int, fz * scale), 0, (1 << 21) - 1))
            codes[j] = _morton3d(ix, iy, iz)
        end
    end
    return codes
end

"""
    morton_order(q, box) -> Vector{Int32}

Return the `Int32` permutation of `1:N` sorted along the 3D Morton (Z-order) space-filling curve.
"""
function morton_order(q::AbstractMatrix{<:Real}, box::AbstractBox)
    codes = morton_codes(q, box)
    perm = Vector{Int32}(undef, length(codes))
    @inbounds for i in eachindex(perm)
        perm[i] = Int32(i)
    end
    sort!(perm; by = idx -> (codes[idx], idx))
    return perm
end

"""
`TilePairCache` — 32-atom spatial cluster (tile) neighbour list with Morton Z-order locality.

Unlike `PairCache` (which materialises every interacting atom pair `(i, j)` and is
capped at `MAX_MATERIALIZED_PAIR_ATOMS = 100_000`), `TilePairCache` groups atoms into
contiguous 32-atom tiles along the Morton curve (`order::Vector{Int32}`), computes tight
axis-aligned bounding boxes (AABBs) for each tile, and stores only interacting tile pairs
`(ti, tj)` (`ti <= tj`) whose bounding boxes lie within `rcut + skin`.
Storage scales as `O(N / 32)` tiles instead of `O(N * z)` atom pairs, enabling 100k–1M+
atom streaming without pair-list memory blowup.
"""
mutable struct TilePairCache
    tile_size::Int
    n_atoms::Int
    n_tiles::Int
    order::Vector{Int32}
    inv_order::Vector{Int32}
    tile_starts::Vector{Int32}
    tile_ends::Vector{Int32}
    bbox_min::Matrix{Float64}      # 3 x n_tiles
    bbox_max::Matrix{Float64}      # 3 x n_tiles
    tile_pairs_i::Vector{Int32}
    tile_pairs_j::Vector{Int32}
    rcut::Float64
    skin::Float64
    build_positions::Matrix{Float64}
    build_box::AbstractBox
    n_rebuilds::Int
    n_reuses::Int
end

num_tile_pairs(tc::TilePairCache) = length(tc.tile_pairs_i)
search_radius(tc::TilePairCache) = tc.rcut + tc.skin

function Base.show(io::IO, tc::TilePairCache)
    print(io, "TilePairCache(N=", tc.n_atoms, ", tiles=", tc.n_tiles,
          ", tile_pairs=", num_tile_pairs(tc), ", rcut=", tc.rcut,
          ", skin=", tc.skin, ", rebuilds=", tc.n_rebuilds,
          ", reuses=", tc.n_reuses, ")")
end

function _tile_cache_geometry_bound(tc::TilePairCache, q::AbstractMatrix{Float64},
                                    box::AbstractBox)
    b0 = tc.build_box
    if b0 isa NoBox && box isa NoBox
        m = 0.0
        b = tc.build_positions
        @inbounds for j in axes(q, 2)
            dx = q[1, j] - b[1, j]; dy = q[2, j] - b[2, j]; dz = q[3, j] - b[3, j]
            d2 = dx * dx + dy * dy + dz * dz
            d2 > m * m && (m = sqrt(d2))
        end
        return (1.0, m)
    end
    ((b0 isa OrthorhombicBox && box isa OrthorhombicBox) ||
     (b0 isa TriclinicBox && box isa TriclinicBox)) || return (0.0, Inf)
    if b0 isa OrthorhombicBox && box isa OrthorhombicBox && b0.L == box.L
        L = box.L
        delta = 0.0
        @inbounds for j in axes(q, 2)
            dx = q[1, j] - tc.build_positions[1, j]
            dy = q[2, j] - tc.build_positions[2, j]
            dz = q[3, j] - tc.build_positions[3, j]
            dx -= L[1] * round(dx / L[1])
            dy -= L[2] * round(dy / L[2])
            dz -= L[3] * round(dz / L[3])
            delta = max(delta, sqrt(dx * dx + dy * dy + dz * dz))
        end
        return 1.0, delta
    end
    H0 = b0 isa OrthorhombicBox ? diagm(0 => collect(b0.L)) : b0.H
    H1 = box isa OrthorhombicBox ? diagm(0 => collect(box.L)) : box.H
    A = H1 / H0
    scale = minimum(svdvals(A))
    delta = 0.0
    @inbounds for j in axes(q, 2)
        b = tc.build_positions
        dx = q[1, j] - (A[1, 1] * b[1, j] + A[1, 2] * b[2, j] + A[1, 3] * b[3, j])
        dy = q[2, j] - (A[2, 1] * b[1, j] + A[2, 2] * b[2, j] + A[2, 3] * b[3, j])
        dz = q[3, j] - (A[3, 1] * b[1, j] + A[3, 2] * b[2, j] + A[3, 3] * b[3, j])
        if box isa OrthorhombicBox
            L = box.L
            dx -= L[1] * round(dx / L[1])
            dy -= L[2] * round(dy / L[2])
            dz -= L[3] * round(dz / L[3])
            delta = max(delta, sqrt(dx * dx + dy * dy + dz * dz))
        else
            d = triclinic_mic(box, dx, dy, dz)
            delta = max(delta, sqrt(norm2(d)))
        end
    end
    return scale, delta
end

function tile_cache_stale(tc::TilePairCache, q::AbstractMatrix{Float64},
                          box::AbstractBox = tc.build_box)
    size(q) == size(tc.build_positions) ||
        throw(ArgumentError("q shape $(size(q)) does not match cache shape $(size(tc.build_positions))"))
    all(isfinite, q) || throw(ArgumentError("non-finite coordinate in q"))
    if box isa OrthorhombicBox || box isa TriclinicBox
        w = box isa OrthorhombicBox ? box.L : box.widths
        2.0 * (tc.rcut + tc.skin) >= minimum(w) && return true
    end
    scale, delta = _tile_cache_geometry_bound(tc, q, box)
    (!isfinite(scale) || !isfinite(delta)) && return true
    slack = scale * search_radius(tc) - 2.0 * delta - tc.rcut
    return slack < 0.0
end

function tile_cache_diagnostics(tc::TilePairCache, q::AbstractMatrix{Float64} = tc.build_positions,
                                box::AbstractBox = tc.build_box)
    scale, max_disp = _tile_cache_geometry_bound(tc, q, box)
    dh_max = (1.0 - scale) * search_radius(tc)
    bytes = sizeof(tc.order) + sizeof(tc.inv_order) + sizeof(tc.tile_starts) +
            sizeof(tc.tile_ends) + sizeof(tc.bbox_min) + sizeof(tc.bbox_max) +
            sizeof(tc.tile_pairs_i) + sizeof(tc.tile_pairs_j) + sizeof(tc.build_positions)
    return (
        n_atoms = tc.n_atoms,
        n_tiles = tc.n_tiles,
        n_tile_pairs = num_tile_pairs(tc),
        tile_size = tc.tile_size,
        rcut = tc.rcut,
        skin = tc.skin,
        search_radius = search_radius(tc),
        max_displacement = max_disp,
        lattice_deformation = dh_max,
        stale = tile_cache_stale(tc, q, box),
        n_rebuilds = tc.n_rebuilds,
        n_reuses = tc.n_reuses,
        rebuilds = tc.n_rebuilds,
        reuses = tc.n_reuses,
        memory_bytes = bytes,
    )
end

@inline function _aabb_min_dist2_ortho(bmin_i, bmax_i, bmin_j, bmax_j, L::NTuple{3,Float64})
    d2 = 0.0
    @inbounds for d in 1:3
        span_i = bmax_i[d] - bmin_i[d]
        span_j = bmax_j[d] - bmin_j[d]
        # If either tile spans >= L[d]/2, single-center MIC could underestimate wrap-around
        if span_i + span_j >= 0.5 * L[d]
            continue
        end
        ci = 0.5 * (bmin_i[d] + bmax_i[d])
        cj = 0.5 * (bmin_j[d] + bmax_j[d])
        hi = 0.5 * span_i
        hj = 0.5 * span_j
        dc = abs(ci - cj - L[d] * round((ci - cj) / L[d]))
        gap = max(0.0, dc - (hi + hj))
        d2 += gap * gap
    end
    return d2
end

@inline function _aabb_min_dist2_nobox(bmin_i, bmax_i, bmin_j, bmax_j)
    d2 = 0.0
    @inbounds for d in 1:3
        ci = 0.5 * (bmin_i[d] + bmax_i[d])
        cj = 0.5 * (bmin_j[d] + bmax_j[d])
        hi = 0.5 * (bmax_i[d] - bmin_i[d])
        hj = 0.5 * (bmax_j[d] - bmin_j[d])
        gap = max(0.0, abs(ci - cj) - (hi + hj))
        d2 += gap * gap
    end
    return d2
end

function fill_tile_cache!(tc::TilePairCache, q::AbstractMatrix{Float64}, box::AbstractBox)
    size(q) == (3, tc.n_atoms) ||
        throw(ArgumentError("q must be (3, $(tc.n_atoms)), got $(size(q))"))
    all(isfinite, q) || throw(ArgumentError("non-finite coordinate in q"))
    (isfinite(tc.rcut) && tc.rcut > 0 && isfinite(tc.skin) && tc.skin >= 0) ||
        throw(ArgumentError("rcut must be positive finite and skin non-negative finite"))
    rsearch = search_radius(tc)
    if box isa OrthorhombicBox || box isa TriclinicBox
        w = box isa OrthorhombicBox ? box.L : box.widths
        2.0 * tc.rcut < minimum(w) ||
            throw(ArgumentError("rcut=$(tc.rcut) must be < min(width)/2"))
        2.0 * rsearch < minimum(w) ||
            throw(ArgumentError("rcut + skin = $rsearch must be < min(width)/2"))
    elseif !(box isa NoBox)
        throw(UnsupportedGeometry("TilePairCache does not support $(typeof(box))"))
    end

    N = tc.n_atoms
    ord = morton_order(q, box)
    copyto!(tc.order, ord)
    @inbounds for k in 1:N
        tc.inv_order[ord[k]] = Int32(k)
    end

    ntiles = tc.n_tiles
    tsize = tc.tile_size
    empty!(tc.tile_pairs_i)
    empty!(tc.tile_pairs_j)

    # Compute tile AABBs: wrap each atom into the primary box [0, L) first,
    # then unwrap relative to the tile's first atom a0 only if the tile straddles
    # a periodic boundary by < L[d]/2. If any atom in the tile is >= L[d]/4 away
    # from a0 along axis d (e.g., when the Morton Z-curve jumps between octants),
    # expand the tile's axis d bounds to [0, L[d]] so no pair along d is ever pruned!
    @inbounds for t in 1:ntiles
        s = (t - 1) * tsize + 1
        e = min(N, t * tsize)
        tc.tile_starts[t] = Int32(s)
        tc.tile_ends[t] = Int32(e)
        a0 = Int(ord[s])
        x0 = q[1, a0]; y0 = q[2, a0]; z0 = q[3, a0]
        if box isa OrthorhombicBox
            L = box.L
            x0 -= L[1] * floor(x0 / L[1])
            y0 -= L[2] * floor(y0 / L[2])
            z0 -= L[3] * floor(z0 / L[3])
        elseif box isa TriclinicBox
            f0 = box.Hinv * [x0, y0, z0]
            w0 = box.H * (f0 .- floor.(f0))
            x0, y0, z0 = w0[1], w0[2], w0[3]
        end
        xmin = x0; xmax = x0
        ymin = y0; ymax = y0
        zmin = z0; zmax = z0
        wide_x = false; wide_y = false; wide_z = false
        for idx in (s + 1):e
            aj = Int(ord[idx])
            if box isa OrthorhombicBox
                L = box.L
                dx = q[1, aj] - q[1, a0]; dx -= L[1] * round(dx / L[1])
                dy = q[2, aj] - q[2, a0]; dy -= L[2] * round(dy / L[2])
                dz = q[3, aj] - q[3, a0]; dz -= L[3] * round(dz / L[3])
                abs(dx) >= 0.25 * L[1] && (wide_x = true)
                abs(dy) >= 0.25 * L[2] && (wide_y = true)
                abs(dz) >= 0.25 * L[3] && (wide_z = true)
                xj = x0 + dx; yj = y0 + dy; zj = z0 + dz
            elseif box isa TriclinicBox
                d = triclinic_mic(box, q[1, aj] - q[1, a0], q[2, aj] - q[2, a0], q[3, aj] - q[3, a0])
                xj = x0 + d.x; yj = y0 + d.y; zj = z0 + d.z
            else
                xj = q[1, aj]; yj = q[2, aj]; zj = q[3, aj]
            end
            xj < xmin && (xmin = xj); xj > xmax && (xmax = xj)
            yj < ymin && (ymin = yj); yj > ymax && (ymax = yj)
            zj < zmin && (zmin = zj); zj > zmax && (zmax = zj)
        end
        if box isa OrthorhombicBox
            L = box.L
            if wide_x || (xmax - xmin) >= 0.25 * L[1]
                xmin = 0.0; xmax = L[1]
            end
            if wide_y || (ymax - ymin) >= 0.25 * L[2]
                ymin = 0.0; ymax = L[2]
            end
            if wide_z || (zmax - zmin) >= 0.25 * L[3]
                zmin = 0.0; zmax = L[3]
            end
        end
        tc.bbox_min[1, t] = xmin; tc.bbox_min[2, t] = ymin; tc.bbox_min[3, t] = zmin
        tc.bbox_max[1, t] = xmax; tc.bbox_max[2, t] = ymax; tc.bbox_max[3, t] = zmax
    end

    # Find all potentially interacting tile pairs (ti <= tj) within rsearch
    r2_cut = rsearch * rsearch
    if box isa OrthorhombicBox
        L = box.L
        # Even at N = 108,000 atoms (ntiles = 3,375), scanning 5.6M tile AABB pairs
        # in pure Float64 takes ~8 ms and is 100% immune to coarse-cell boundary wraps
        # when individual Morton tiles span across periodic sub-cubes.
        @inbounds for ti in 1:ntiles
            push!(tc.tile_pairs_i, Int32(ti))
            push!(tc.tile_pairs_j, Int32(ti))
            bmin_i = view(tc.bbox_min, :, ti); bmax_i = view(tc.bbox_max, :, ti)
            for tj in (ti + 1):ntiles
                if _aabb_min_dist2_ortho(bmin_i, bmax_i, view(tc.bbox_min, :, tj), view(tc.bbox_max, :, tj), L) <= r2_cut
                    push!(tc.tile_pairs_i, Int32(ti))
                    push!(tc.tile_pairs_j, Int32(tj))
                end
            end
        end
    elseif box isa TriclinicBox
        tb = box::TriclinicBox
        @inbounds for ti in 1:ntiles
            push!(tc.tile_pairs_i, Int32(ti))
            push!(tc.tile_pairs_j, Int32(ti))
            ci1 = 0.5 * (tc.bbox_min[1, ti] + tc.bbox_max[1, ti])
            ci2 = 0.5 * (tc.bbox_min[2, ti] + tc.bbox_max[2, ti])
            ci3 = 0.5 * (tc.bbox_min[3, ti] + tc.bbox_max[3, ti])
            ri = 0.5 * sqrt((tc.bbox_max[1, ti] - tc.bbox_min[1, ti])^2 +
                            (tc.bbox_max[2, ti] - tc.bbox_min[2, ti])^2 +
                            (tc.bbox_max[3, ti] - tc.bbox_min[3, ti])^2)
            for tj in (ti + 1):ntiles
                cj1 = 0.5 * (tc.bbox_min[1, tj] + tc.bbox_max[1, tj])
                cj2 = 0.5 * (tc.bbox_min[2, tj] + tc.bbox_max[2, tj])
                cj3 = 0.5 * (tc.bbox_min[3, tj] + tc.bbox_max[3, tj])
                rj = 0.5 * sqrt((tc.bbox_max[1, tj] - tc.bbox_min[1, tj])^2 +
                                (tc.bbox_max[2, tj] - tc.bbox_min[2, tj])^2 +
                                (tc.bbox_max[3, tj] - tc.bbox_min[3, tj])^2)
                d = triclinic_mic(tb, ci1 - cj1, ci2 - cj2, ci3 - cj3)
                if norm2(d) <= (rsearch + ri + rj)^2
                    push!(tc.tile_pairs_i, Int32(ti))
                    push!(tc.tile_pairs_j, Int32(tj))
                end
            end
        end
    else
        @inbounds for ti in 1:ntiles
            push!(tc.tile_pairs_i, Int32(ti))
            push!(tc.tile_pairs_j, Int32(ti))
            bmin_i = view(tc.bbox_min, :, ti); bmax_i = view(tc.bbox_max, :, ti)
            for tj in (ti + 1):ntiles
                if _aabb_min_dist2_nobox(bmin_i, bmax_i, view(tc.bbox_min, :, tj), view(tc.bbox_max, :, tj)) <= r2_cut
                    push!(tc.tile_pairs_i, Int32(ti))
                    push!(tc.tile_pairs_j, Int32(tj))
                end
            end
        end
    end

    # Sort tile pairs canonically by (ti, tj)
    np = length(tc.tile_pairs_i)
    if np > 1
        p = sortperm(1:np; by = k -> (tc.tile_pairs_i[k], tc.tile_pairs_j[k]))
        tc.tile_pairs_i .= tc.tile_pairs_i[p]
        tc.tile_pairs_j .= tc.tile_pairs_j[p]
    end

    copyto!(tc.build_positions, q)
    tc.build_box = _copy_box(box)
    tc.n_rebuilds += 1
    return tc
end

function TilePairCache(q::AbstractMatrix{Float64}, box::AbstractBox,
                       rcut::Float64, skin::Float64; tile_size::Int = TILE_SIZE)
    tile_size > 0 || throw(ArgumentError("tile_size must be positive"))
    size(q, 1) == 3 || throw(ArgumentError("q must be 3 x N"))
    N = size(q, 2)
    ntiles = cld(N, tile_size)
    tc = TilePairCache(
        tile_size, N, ntiles,
        zeros(Int32, N), zeros(Int32, N),
        zeros(Int32, ntiles), zeros(Int32, ntiles),
        zeros(Float64, 3, ntiles), zeros(Float64, 3, ntiles),
        Int32[], Int32[],
        Float64(rcut), Float64(skin),
        copy(q), _copy_box(box), 0, 0
    )
    return fill_tile_cache!(tc, q, box)
end

function refresh_tile_cache!(tc::TilePairCache, q::AbstractMatrix{Float64},
                             box::AbstractBox = tc.build_box)
    if tile_cache_stale(tc, q, box)
        fill_tile_cache!(tc, q, box)
        return true
    end
    tc.n_reuses += 1
    return false
end

# ---------------------------------------------------------------------------
# pair iteration
# ---------------------------------------------------------------------------

"Iterate `(j, k)` pairs with `j < k` exactly once."
function neighbor_pairs(nl::NeighborList)
    n = length(nl.lists)
    n <= MAX_MATERIALIZED_PAIR_ATOMS ||
        throw(ArgumentError("neighbor_pairs materializes O(Nz) tuple objects and is " *
                            "disabled above N=$MAX_MATERIALIZED_PAIR_ATOMS (N=$n); " *
                            "stream neighbors directly instead"))
    out = Tuple{Int,Int}[]
    for j in eachindex(nl.lists)
        for k in nl.lists[j]
            k > j && push!(out, (j, k))
        end
    end
    return out
end

end # module Neighbors
