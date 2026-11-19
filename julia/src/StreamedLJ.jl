"""
`StreamedLJ` — a one-pass Lennard-Jones energy/force evaluator.

The streaming core is deliberately small and standalone: it depends only on
`..Types` for the orthorhombic box and minimum-image displacement convention.
It does not materialise a pair list and does not build `Vector{Vector}` cell
buckets. An optional `StreamedLJFactor` adapter at the bottom of this file
implements the sibling `Factors.AbstractFactor` protocol so that the same
streamed kernel can be handed to `Dynamics.HarmonicSystem`; the core workspace
API remains independent of the factor protocol.

## Cutoff conventions

For each unordered pair `(i, j)` with `r <= rcut`, using Lorentz–Berthelot
mixing

    eps_ij = sqrt(eps_i * eps_j)
    sig_ij = (sig_i + sig_j) / 2

the raw Lennard-Jones potential is

    U_ij(r) = 4 eps_ij [ (sig_ij/r)^12 - (sig_ij/r)^6 ].

Two cutoff modes are implemented. They are *different potentials*, not
cosmetic choices, exactly as in `Factors.LennardJonesPair`:

* `:lfs` (default, production) — linear-force-shifted:

      U_lfs(r) = U(r) - U(rc) - U'(rc) (r - rc)
      dU_lfs/dr = U'(r) - U'(rc)

  so `U_lfs(rc) = 0` and `U_lfs'(rc) = 0`. The energy and the force are both
  continuous at the cutoff (`C^1`); the force is the exact derivative of the
  energy. This is the only mode that is safe for production dynamics.

* `:shift` (legacy, opt-in) — energy-shifted:

      U_shift(r) = U(r) - U(rc)

  so the energy is continuous (`C^0`) but the force is not: `U_shift'(rc) != 0`.
  It is retained because existing validation fixtures and NIST-style references
  use it. `FORCE_CONSISTENT_AT_CUTOFF` is `true` for the default `:lfs` mode;
  callers that need `:shift` must ask for it explicitly.

## Pair topology: exclusions and scaled pairs

A `PairTopology` supplies per-pair weights `w_ij` without creating one object
per pair. It stores only two sorted vectors of packed canonical keys
(`UInt64(i) << 32 | UInt64(j)`, with `i < j`) and one parallel weight vector,
so lookup is an allocation-free binary search and memory is O(P) integers.

The weight multiplies the *entire* effective pair potential, including the
linear-force-shift correction:

    U_eff(r) = w_ij * U_lfs(r)          (or w_ij * U_shift(r))
    F_eff     = -dU_eff/dr * (q_i-q_j)/r

Supported topology and exact semantics:

* **1-2 and 1-3 exclusions** — pass the pair as `excluded` (or with weight
  `0.0` in `scaled`). The pair is skipped before any geometry is computed, so
  it contributes neither energy, force, parameter gradient, nor to the
  returned pair count.
* **1-4 scaling** — pass `(i, j, lambda)` in `scaled` (or `(i,j) => lambda`).
  `lambda` is a finite non-negative number applied to the full LFS/shift
  energy and force; `0.5` is the usual AMBER/OPLS 1-4 LJ scale, `1.0` is the
  CHARMM convention. A zero scale is normalised to an exclusion.
* **No automatic derivation from `ChemicalTopology`.** This module refuses to
  guess which bonded path is 1-2, 1-3, or 1-4; the caller must supply the exact
  canonical pair sets. This refusal is deliberate: hidden topology inference is
  how exclusions silently go wrong.
* Invalid topology is refused with `ArgumentError`: non-integer or
  non-positive indices, `i == j`, indices above `2^32 - 1`, a pair listed
  twice, a pair that is both excluded and scaled, a negative or non-finite
  scale, or a topology whose largest atom index exceeds `N` at evaluation time.
* **Unsupported geometries** are refused, not approximated: only
  `OrthorhombicBox` is accepted (`NoBox` and triclinic cells raise
  `ArgumentError`), and `rcut < min(L)/2` is required so that the
  minimum-image/cell stencil is valid.

The topology is a plain read-only description of the pair set; it owns no
per-atom state and is reusable across workspaces, boxes, and evaluations. Do
not mutate its internal vectors after construction: build a new topology
instead.

## Parameter derivatives

`evaluate_with_gradients!` returns the exact analytic per-atom derivatives of
the *effective* energy (weights and cutoff corrections included) with respect
to the input `epsilon` and `sigma` vectors:

    dU/deps_i = sum_j w_ij * U_eff_pair / (2 eps_i)
    dU/dsig_i = sum_j 0.5 * w_ij * dU_eff_pair/dsig_ij

These are the quantities a force-field parameter optimiser needs; they are
compatible with the Lorentz–Berthelot mixing rule because the chain rule is
applied to `eps_ij = sqrt(eps_i eps_j)` and `sig_ij = (sig_i + sig_j)/2`.
The gradient path requires every `eps_i > 0` (the square-root mixing rule is
not differentiable at zero) and refuses other values before touching any
output array.

## Complexity and storage

The workspace stores O(N + Ncells) reusable arrays: `head` (one `Int32` per
cell), `next` (one `Int32` per atom), a `3 x N` force matrix, and two `N`
gradient accumulators. Every `evaluate!` rebuilds the cell linked list from
scratch; there is no neighbour-list cache. At fixed density and cutoff,
`Ncells` grows linearly with N and the work is asymptotically O(N). Pairs are
streamed directly inside the cell traversal: no `(i, j)` tuple array, no pair
energy array, and no `Vector{Vector}` is ever formed.

Opt-in `threaded=true` assigns every unordered cell pair to its lower-ID cell
exactly once. Each worker accumulates both endpoints in a reusable sparse buffer
containing only touched atoms. Buffers are reduced in fixed worker order and
first-touch atom order, while per-cell energy/count are reduced in increasing
cell order. This uses O(threads * touched_atoms) additional force storage.
Reusable pair buffers do not allocate after warmup, but the `Threads.@threads`
scheduler itself allocates a small amount per threaded evaluation. Results are
repeatable for a fixed worker count; floating-point grouping means
cross-thread-count results are not promised bitwise-identical.
Parameter-gradient evaluations always use the serial kernel.

All input validation happens before any output array is modified, so an
invalid call leaves `ws.forces` (and the gradient arrays) byte-for-byte
unchanged.
"""
module StreamedLJ

using ..Types
using ..Types: AbstractBox, OrthorhombicBox, TriclinicBox, SVector3, box_lengths

# `StreamedLJFactor` implements the sibling `Factors` protocol. FoundationCore
# includes `Factors` before this file, so `..Factors` is normally available. A
# standalone host that only supplies `Types` (as `test/test_streamed_lj.jl`
# does) must still dispatch on the *same* `AbstractFactor`/`evaluate!` objects —
# a private copy of the protocol would be invisible to `Dynamics`. Resolve the
# already-loaded module explicitly instead of guessing.
const _FACTORS_MODULE = let
    p = parentmodule(@__MODULE__)
    if isdefined(p, :Factors)
        getfield(p, :Factors)
    else
        gp = parentmodule(p)
        if isdefined(gp, :FoundationCore)
            fc = getfield(gp, :FoundationCore)
            isdefined(fc, :Factors) || error(
                "StreamedLJ: FoundationCore is loaded but does not define `Factors`")
            getfield(fc, :Factors)
        else
            error("StreamedLJ: could not resolve the `Factors` module. Include " *
                  "this file in a module that defines `Factors` (as " *
                  "FoundationCore does) or after loading FoundationCore.")
        end
    end
end
const Factors = _FACTORS_MODULE

# The persistent neighbour list (`PairCache`) lives in the sibling `Neighbors`
# module, which FoundationCore includes before this file. Resolve it the same way
# `Factors` is resolved so a standalone host that only supplies `Types` still
# gets the *same* cache object rather than a private copy.
const _NEIGHBORS_MODULE = let
    p = parentmodule(@__MODULE__)
    if isdefined(p, :Neighbors)
        getfield(p, :Neighbors)
    else
        gp = parentmodule(p)
        if isdefined(gp, :FoundationCore)
            fc = getfield(gp, :FoundationCore)
            isdefined(fc, :Neighbors) || error(
                "StreamedLJ: FoundationCore is loaded but does not define `Neighbors`")
            getfield(fc, :Neighbors)
        else
            error("StreamedLJ: could not resolve the `Neighbors` module. Include " *
                  "this file in a module that defines `Neighbors` (as " *
                  "FoundationCore does) or after loading FoundationCore.")
        end
    end
end
const Neighbors = _NEIGHBORS_MODULE

export LJWorkspace, PairTopology, evaluate!, evaluate_with_gradients!,
       estimate_memory_bytes, pair_weight,
       CUTOFF_MODE, DEFAULT_CUTOFF_MODE, SUPPORTED_CUTOFF_MODES,
       FORCE_CONSISTENT_AT_CUTOFF, FIXED_POINT_SCALE, INV_FIXED_POINT_SCALE,
       StreamedLJFactor,
       streamed_energy, streamed_parameter_gradients!,
       box_derivative, box_derivative!, evaluate_with_box_derivative!,
       streamed_box_derivative, streamed_box_derivative!

"Scale factor `2^40` for deterministic 64-bit integer (`Int64`) force accumulation."
const FIXED_POINT_SCALE = Float64(Int64(1) << 40)
const INV_FIXED_POINT_SCALE = 1.0 / FIXED_POINT_SCALE

"Cutoff conventions implemented here."
const SUPPORTED_CUTOFF_MODES = (:lfs, :shift, :trunc)

"The production default: linear-force-shifted, `C^1` at the cutoff."
const DEFAULT_CUTOFF_MODE = :lfs

"Default cutoff mode (`:lfs`); `:shift` remains available as an explicit opt-in."
const CUTOFF_MODE = DEFAULT_CUTOFF_MODE

"The default (`:lfs`) potential is force-consistent at the cutoff."
const FORCE_CONSISTENT_AT_CUTOFF = true

# The 14 half-stencil cells: self plus one representative of each ± pair in the
# 3x3x3 stencil. This is valid only when every grid dimension is at least 3,
# where the offsets are distinct modulo the periodic grid.
const _HALF_OFFSETS = (
    (0, 0, 0), (1, 0, 0),
    (-1, 1, 0), (0, 1, 0), (1, 1, 0),
    (-1, -1, 1), (0, -1, 1), (1, -1, 1),
    (-1, 0, 1), (0, 0, 1), (1, 0, 1),
    (-1, 1, 1), (0, 1, 1), (1, 1, 1),
)

# Full 3x3x3 stencil. Used when a grid dimension is 1 or 2, where a half-stencil
# would count a pair of periodic cells twice. Target cell IDs are de-duplicated
# per source cell (a small fixed scratch array, never a per-particle Set), and
# cross-cell pairs are filtered by the global atom order j > i.
const _FULL_OFFSETS = Tuple((dx, dy, dz)
                            for dz in -1:1 for dy in -1:1 for dx in -1:1)

# ---------------------------------------------------------------------------
# compact pair topology
# ---------------------------------------------------------------------------

"""
Compact, read-only description of pair weights.

`excluded` and `scaled_keys` are sorted vectors of packed canonical pair keys
(`i < j`); `scaled_weights` is parallel to `scaled_keys`. Memory is O(P)
`UInt64`/`Float64` entries — never one object per pair.

Construct with explicit pair specifications:

    PairTopology(excluded = [(1, 2), (2, 3)],          # 1-2 / 1-3 exclusions
                  scaled   = [(1, 4, 0.5)])            # 1-4 LJ scale

`excluded` entries are `(i, j)`; `scaled` entries are `(i, j, lambda)` or
`(i, j) => lambda`, and `scaled` may also be a `Dict{Tuple{Int,Int},Float64}`.
Indices are canonicalised, so `(i, j)` and `(j, i)` name the same pair.
Duplicate pairs, excluded/scaled overlaps, non-positive indices, `i == j`,
negative/non-finite scales, and indices above `2^32 - 1` are refused with
`ArgumentError`. A zero scale is stored as an exclusion.
"""
struct PairTopology
    excluded::Vector{UInt64}
    scaled_keys::Vector{UInt64}
    scaled_weights::Vector{Float64}
    max_atom::Int
end

@inline function _pack_pair(i::Int, j::Int)
    a, b = i < j ? (i, j) : (j, i)
    return (UInt64(a) << 32) | UInt64(b)
end

@inline function _unpack_pair(key::UInt64)
    return (Int(key >> 32), Int(key & UInt64(0xffffffff)))
end

function _pair_indices(spec, label::AbstractString)
    if spec isa Pair
        i, j = first(spec), last(spec)
    elseif spec isa Tuple && length(spec) == 2
        i, j = spec[1], spec[2]
    elseif spec isa AbstractVector && length(spec) == 2
        i, j = spec[1], spec[2]
    else
        throw(ArgumentError("$label entry must be an (i, j) pair, got $(repr(spec))"))
    end
    (i isa Integer && j isa Integer) ||
        throw(ArgumentError("$label indices must be integers, got $(repr(spec))"))
    i == j && throw(ArgumentError("$label pair must have distinct atoms, got $(repr(spec))"))
    (1 <= i <= typemax(UInt32) && 1 <= j <= typemax(UInt32)) ||
        throw(ArgumentError("$label indices must lie in 1:$(typemax(UInt32)), got $(repr(spec))"))
    ii, jj = Int(i), Int(j)
    return ii < jj ? (ii, jj) : (jj, ii)
end

function _scaled_entry(item)
    if item isa Pair
        i, j = _pair_indices(first(item), "scaled")
        w = last(item)
    elseif item isa Tuple && length(item) == 3
        i, j = _pair_indices((item[1], item[2]), "scaled")
        w = item[3]
    elseif item isa AbstractVector && length(item) == 3
        i, j = _pair_indices((item[1], item[2]), "scaled")
        w = item[3]
    else
        throw(ArgumentError("scaled entry must be (i, j, weight) or (i, j) => weight, " *
                            "got $(repr(item))"))
    end
    (w isa Real && isfinite(w) && w >= 0) ||
        throw(ArgumentError("scaled-pair weight must be finite and >= 0, got $(repr(w))"))
    return _pack_pair(i, j), Float64(w)
end

_excluded_entries(x::AbstractDict) = keys(x)
_excluded_entries(x) = x

_scaled_entries(x::AbstractDict) = ((k => v) for (k, v) in x)
_scaled_entries(x) = x

function _sorted_unique_check!(v::Vector{UInt64}, label::AbstractString)
    @inbounds for k in 2:length(v)
        if v[k] == v[k - 1]
            i, j = _unpack_pair(v[k])
            throw(ArgumentError("duplicate $label pair ($i, $j)"))
        end
    end
    return nothing
end

function _build_pair_topology(excluded, scaled)
    ex = UInt64[]
    max_atom = 0
    for spec in _excluded_entries(excluded)
        i, j = _pair_indices(spec, "excluded")
        push!(ex, _pack_pair(i, j))
        max_atom = max(max_atom, i, j)
    end

    sk = UInt64[]
    sw = Float64[]
    for item in _scaled_entries(scaled)
        key, w = _scaled_entry(item)
        i, j = _unpack_pair(key)
        max_atom = max(max_atom, i, j)
        if w == 0.0
            push!(ex, key)          # a zero scale is exactly an exclusion
        else
            push!(sk, key)
            push!(sw, w)
        end
    end

    sort!(ex)
    _sorted_unique_check!(ex, "excluded")

    perm = sortperm(sk)
    sk .= sk[perm]
    sw .= sw[perm]
    _sorted_unique_check!(sk, "scaled")

    for key in sk
        if _sorted_has(ex, key)
            i, j = _unpack_pair(key)
            throw(ArgumentError("pair ($i, $j) is both excluded and scaled; " *
                                "remove one of the two entries"))
        end
    end

    return PairTopology(ex, sk, sw, max_atom)
end

"""
    PairTopology(excluded = (), scaled = ()) -> PairTopology

Build a validated compact pair topology. See [`PairTopology`](@ref) for the
accepted entry forms and the exact validation rules.
"""
function PairTopology(; excluded = (), scaled = ())
    return _build_pair_topology(excluded, scaled)
end

PairTopology(excluded, scaled) = _build_pair_topology(excluded, scaled)

@inline function _sorted_has(v::Vector{UInt64}, key::UInt64)
    pos = searchsortedfirst(v, key)
    if pos <= length(v)
        @inbounds if v[pos] == key
            return true
        end
    end
    return false
end

@inline function _topology_weight(topo::PairTopology, key::UInt64)
    if !isempty(topo.excluded) && _sorted_has(topo.excluded, key)
        return 0.0
    end
    if !isempty(topo.scaled_keys)
        pos = searchsortedfirst(topo.scaled_keys, key)
        if pos <= length(topo.scaled_keys)
            @inbounds if topo.scaled_keys[pos] == key
                return topo.scaled_weights[pos]
            end
        end
    end
    return 1.0
end

@inline _topology_weight(::Nothing, ::UInt64) = 1.0

@inline function _pair_weight_raw(topo, i::Int, j::Int)
    topo === nothing && return 1.0
    return _topology_weight(topo, _pack_pair(i, j))
end

"""
    pair_weight(topology, i, j) -> Float64

Canonical pair weight: `0.0` when `(i, j)` is excluded, the stored scale when
it is a scaled pair, otherwise `1.0`. `(i, j)` and `(j, i)` return the same
value. With `topology === nothing` the weight is always `1.0`.
"""
function pair_weight(topo::PairTopology, i::Integer, j::Integer)
    i == j && throw(ArgumentError("pair_weight needs distinct atoms, got ($i, $j)"))
    (1 <= i <= typemax(UInt32) && 1 <= j <= typemax(UInt32)) ||
        throw(ArgumentError("pair indices must lie in 1:$(typemax(UInt32)), got ($i, $j)"))
    return _topology_weight(topo, _pack_pair(Int(i), Int(j)))
end

pair_weight(::Nothing, i::Integer, j::Integer) = 1.0

function Base.show(io::IO, t::PairTopology)
    print(io, "PairTopology(excluded=", length(t.excluded),
          ", scaled=", length(t.scaled_keys), ", max_atom=", t.max_atom, ")")
    return nothing
end

# ---------------------------------------------------------------------------
# workspace
# ---------------------------------------------------------------------------

"""
Reusable storage for streaming Lennard-Jones evaluations.

Fields are intentionally flat. `head`/`next` form a cell linked list; `forces`
is the `3 x N` force accumulator. `grad_epsilon` and `grad_sigma` are the
per-atom parameter-gradient accumulators written only by
`evaluate_with_gradients!`. The workspace owns these arrays and `evaluate!`
overwrites them, so callers that need to keep a result must copy it. Concurrent
evaluations must use separate workspaces; threaded tasks within one evaluation
are race-free but a workspace is not safe for simultaneous calls.
"""
mutable struct SparseThreadForces
    atom_index::Dict{Int,Int}
    atoms::Vector{Int}
    values::Vector{Float64}
    active::BitVector
end

SparseThreadForces() = SparseThreadForces(Dict{Int,Int}(), Int[], Float64[], BitVector())

mutable struct LJWorkspace
    N::Int
    head::Vector{Int32}
    next::Vector{Int32}
    forces::Matrix{Float64}
    scratch::Vector{Int}
    grad_epsilon::Vector{Float64}
    grad_sigma::Vector{Float64}
    cell_energy::Vector{Float64}
    cell_count::Vector{Int}
    thread_forces::Vector{SparseThreadForces}
    box_derivative::Matrix{Float64}
end

LJWorkspace(N, head, next, forces, scratch, grad_epsilon, grad_sigma,
            cell_energy, cell_count, thread_forces) =
    LJWorkspace(N, head, next, forces, scratch, grad_epsilon, grad_sigma,
                cell_energy, cell_count, thread_forces, zeros(Float64, 3, 3))

"""
    LJWorkspace(q, box, rcut; skin=0)

Validate the initial geometry and allocate the reusable O(N + Ncells) arrays.
`q`, `epsilon`, and `sigma` are supplied again to `evaluate!`; this constructor
only fixes the initial size so that ordinary repeated calls do not allocate.
"""
function LJWorkspace(q::AbstractMatrix{<:Real}, box::AbstractBox, rcut::Real;
                     skin::Real = 0)
    _validate_positions(q)
    rc, sk = _validate_cutoff(box, rcut, skin)
    N = size(q, 2)
    _validate_capacity(N)
    widths = box isa OrthorhombicBox ? box.L : box.widths
    nc = _grid_nc(widths, rc + sk)
    ncell = _checked_ncells(nc)
    return LJWorkspace(N, zeros(Int32, ncell), Vector{Int32}(undef, N),
                       zeros(Float64, 3, N), Vector{Int}(undef, 27),
                       zeros(Float64, N), zeros(Float64, N),
                       Vector{Float64}(undef, ncell), Vector{Int}(undef, ncell),
                       SparseThreadForces[], zeros(Float64, 3, 3))
end

# ---------------------------------------------------------------------------
# public evaluation entry points
# ---------------------------------------------------------------------------

"""
    evaluate!(ws, q, epsilon, sigma, box; rcut, skin=0.0,
              mode=DEFAULT_CUTOFF_MODE, topology=nothing, threaded=false)
        -> (energy, forces, pair_count)

Evaluate the selected LJ energy and forces for every unordered pair once. The
cell list is rebuilt on every call. Input validation (including `mode` and
`topology`) happens **before** any force array is touched, so an invalid call
leaves `ws.forces` unchanged.

`mode` is `:lfs` (default, `C^1` force-consistent) or `:shift` (legacy `C^0`).
`topology` is `nothing` or a [`PairTopology`](@ref); excluded pairs contribute
nothing, scaled pairs contribute their weight times the full effective
potential.

Returns the total energy, the workspace-owned `3 x N` force matrix (overwritten
by the next call), and the number of *interacting* pairs inside `rcut`
(excluded pairs are not counted).
Set `threaded=true` to enable deterministic per-cell parallel force evaluation
when Julia was started with more than one thread. A single-thread Julia process
uses the original serial kernel regardless of this flag.
"""
@inline _box_widths(b::OrthorhombicBox) = b.L
@inline _box_widths(b::Types.TriclinicBox) = b.widths

function evaluate!(ws::LJWorkspace, q::AbstractMatrix{<:Real},
                   epsilon::AbstractVector{<:Real},
                   sigma::AbstractVector{<:Real},
                   box::B; rcut::Real, skin::Real = 0.0,
                   mode = DEFAULT_CUTOFF_MODE, topology = nothing,
                   threaded::Bool = false,
                   fixed_point::Bool = false,
                   cache::Union{Nothing,Neighbors.PairCache,Neighbors.TilePairCache} = nothing) where {B <: AbstractBox}
    energy, forces, pair_count, _, _, _ =
        _evaluate!(ws, q, epsilon, sigma, box, rcut, skin, mode, topology,
                   false, false, threaded, nothing, cache, fixed_point)
    return energy, forces, pair_count
end

"""
    evaluate_with_gradients!(ws, q, epsilon, sigma, box; rcut, skin=0.0,
                              mode=DEFAULT_CUTOFF_MODE, topology=nothing,
                              threaded=false)
        -> (energy, forces, pair_count, dU_depsilon, dU_dsigma)

Same evaluation as [`evaluate!`](@ref), but also returns the exact analytic
per-atom parameter gradients of the *effective* energy (weights and cutoff
corrections included). The two gradient vectors are workspace-owned and are
overwritten by the next gradient call.

The gradient path requires `epsilon[i] > 0` for every atom because
`eps_ij = sqrt(eps_i eps_j)` is not differentiable at zero; this is validated
before any output array is modified.
`threaded=true` explicitly falls back to the serial kernel: gradients are
updated by both endpoints and cannot share the force-only cell ownership rule.
"""
function evaluate_with_gradients!(ws::LJWorkspace, q::AbstractMatrix{<:Real},
                                  epsilon::AbstractVector{<:Real},
                                  sigma::AbstractVector{<:Real},
                                  box::B; rcut::Real, skin::Real = 0.0,
                                  mode = DEFAULT_CUTOFF_MODE, topology = nothing,
                                  threaded::Bool = false,
                                  cache::Union{Nothing,Neighbors.PairCache,Neighbors.TilePairCache} = nothing) where {B <: AbstractBox}
    energy, forces, pair_count, ge, gs, _ =
        _evaluate!(ws, q, epsilon, sigma, box, rcut, skin, mode, topology,
                   true, false, threaded, nothing, cache, false)
    return energy, forces, pair_count, ge, gs
end

"""
    evaluate_with_box_derivative!(ws, q, epsilon, sigma, box; rcut, skin=0.0,
                                  mode=DEFAULT_CUTOFF_MODE, topology=nothing,
                                  threaded=false)
        -> (energy, forces, pair_count, box_derivative)

Evaluate the LJ energy, Cartesian forces, and exact analytic box derivative
∂U/∂h|_s at fixed fractional coordinates. The returned 3x3 matrix is workspace-owned
and overwritten on the next box-derivative call.
"""
function evaluate_with_box_derivative!(ws::LJWorkspace, q::AbstractMatrix{<:Real},
                                       epsilon::AbstractVector{<:Real},
                                       sigma::AbstractVector{<:Real},
                                       box::B; rcut::Real, skin::Real = 0.0,
                                       mode = DEFAULT_CUTOFF_MODE, topology = nothing,
                                       threaded::Bool = false,
                                       cache::Union{Nothing,Neighbors.PairCache,Neighbors.TilePairCache} = nothing) where {B <: AbstractBox}
    energy, forces, pair_count, _, _, G =
        _evaluate!(ws, q, epsilon, sigma, box, rcut, skin, mode, topology,
                   false, true, threaded, nothing, cache, false)
    return energy, forces, pair_count, G
end

"""
    box_derivative!(G::AbstractMatrix{Float64}, ws::LJWorkspace, q, epsilon, sigma, box;
                    rcut, skin=0.0, mode=DEFAULT_CUTOFF_MODE, topology=nothing,
                    threaded=false) -> Matrix{Float64}

Accumulate the exact analytic box derivative ∂U/∂h|_s at fixed fractional coordinates
into caller-supplied 3x3 matrix `G`.
"""
function box_derivative!(G::AbstractMatrix{Float64}, ws::LJWorkspace,
                         q::AbstractMatrix{<:Real},
                         epsilon::AbstractVector{<:Real},
                         sigma::AbstractVector{<:Real},
                         box::B; rcut::Real, skin::Real = 0.0,
                         mode = DEFAULT_CUTOFF_MODE, topology = nothing,
                         threaded::Bool = false,
                         cache::Union{Nothing,Neighbors.PairCache,Neighbors.TilePairCache} = nothing) where {B <: AbstractBox}
    size(G) == (3, 3) || throw(ArgumentError("G must be 3x3, got $(size(G))"))
    _evaluate!(ws, q, epsilon, sigma, box, rcut, skin, mode, topology,
               false, true, threaded, G, cache, false)
    return G
end

"""
    box_derivative(ws::LJWorkspace, q, epsilon, sigma, box; kwargs...) -> Matrix{Float64}

Compute the exact analytic box derivative ∂U/∂h|_s at fixed fractional coordinates
as a new 3x3 matrix.
"""
function box_derivative(ws::LJWorkspace, q::AbstractMatrix{<:Real},
                        epsilon::AbstractVector{<:Real},
                        sigma::AbstractVector{<:Real},
                        box::AbstractBox; kwargs...)
    G = zeros(Float64, 3, 3)
    box_derivative!(G, ws, q, epsilon, sigma, box; kwargs...)
    return G
end

function _evaluate!(ws::LJWorkspace, q::AbstractMatrix{<:Real},
                    epsilon::AbstractVector{<:Real},
                    sigma::AbstractVector{<:Real},
                    box::B, rcut::Real, skin::Real,
                    mode, topology, want_grads::Bool, want_box_deriv::Bool,
                    threaded::Bool, G_out::Union{Nothing,AbstractMatrix{Float64}} = nothing,
                    cache::Union{Nothing,Neighbors.PairCache,Neighbors.TilePairCache} = nothing,
                    fixed_point::Bool = false) where {B <: AbstractBox}
    # All validation precedes any mutation of ws.forces / gradient arrays / G.
    _validate_positions(q)
    rc, sk = _validate_cutoff(box, rcut, skin)
    N = size(q, 2)
    _validate_capacity(N)
    _validate_parameters(epsilon, sigma, N)
    _validate_mode(mode)
    _validate_topology(topology, N)
    want_grads && _validate_gradient_parameters(epsilon)
    cache !== nothing && _validate_cache(cache, q, box, rc)

    nc = _grid_nc(_box_widths(box), rc + sk)
    ncell = _checked_ncells(nc)
    _ensure_workspace!(ws, N, ncell)

    fill!(ws.forces, 0.0)

    if want_grads
        fill!(ws.grad_epsilon, 0.0)
        fill!(ws.grad_sigma, 0.0)
        ge = ws.grad_epsilon
        gs = ws.grad_sigma
    else
        ge = nothing
        gs = nothing
    end

    if want_box_deriv
        G = G_out !== nothing ? G_out : ws.box_derivative
        fill!(G, 0.0)
    else
        G = nothing
    end

    rc2 = rc * rc
    mode_sym = Symbol(mode)

    if cache isa Neighbors.TilePairCache
        Neighbors.refresh_tile_cache!(cache, Matrix{Float64}(q), box)
        if !want_grads && G === nothing && (fixed_point || (threaded && Threads.nthreads() > 1))
            energy, pair_count = _stream_tiled_fixed_point!(ws, cache, q, epsilon, sigma,
                                                            box, rc, rc2, mode_sym, topology, threaded)
        else
            energy, pair_count = _stream_tiled!(ws, cache, q, epsilon, sigma, box,
                                                rc, rc2, mode_sym, topology, ge, gs, G)
        end
        return energy, ws.forces, pair_count, ge, gs, G
    elseif cache isa Neighbors.PairCache
        # Cached Verlet path: the persistent pair array replaces both the cell
        # linked-list construction and the stencil traversal. `refresh!` is the
        # only work the step pays beyond the pair forces themselves.
        Neighbors.refresh_pair_cache!(cache, q, box)
        if !want_grads && G === nothing && fixed_point
            energy, pair_count = _stream_cached_fixed_point!(ws, cache, q, epsilon, sigma,
                                                             box, rc, rc2, mode_sym, topology, threaded)
        elseif threaded && !want_grads && G === nothing && Threads.nthreads() > 1
            energy, pair_count = _stream_cached_threaded!(ws, cache, q, epsilon, sigma,
                                                          box, rc, rc2, mode_sym, topology)
        else
            energy, pair_count = _stream_cached!(ws, cache, q, epsilon, sigma, box,
                                                 rc, rc2, mode_sym, topology, ge, gs, G)
        end
        return energy, ws.forces, pair_count, ge, gs, G
    end

    fill!(ws.head, Int32(0))

    # Build the cell linked list. Inserting atoms in increasing index order makes
    # each cell chain strictly descending in atom index, which lets the self-cell
    # traversal enumerate i > j pairs without any index comparisons.
    @inbounds for j in 1:N
        c = _cell_id(q, box, nc, j)
        ws.next[j] = ws.head[c]
        ws.head[c] = Int32(j)
    end

    if threaded && !want_grads && G === nothing && Threads.nthreads() > 1
        energy, pair_count = _stream_threaded!(ws, q, epsilon, sigma, box, nc,
                                               rc, rc2, mode_sym, topology)
    elseif nc[1] >= 3 && nc[2] >= 3 && nc[3] >= 3
        energy, pair_count = _stream_half!(ws, q, epsilon, sigma, box, nc, rc, rc2,
                                           mode_sym, topology, ge, gs, G)
    else
        energy, pair_count = _stream_full!(ws, q, epsilon, sigma, box, nc, rc, rc2,
                                           mode_sym, topology, ge, gs, G)
    end
    return energy, ws.forces, pair_count, ge, gs, G
end

# A cache is only usable when it was built for the same atom count, the same
# cutoff and a search radius that still fits the minimum-image stencil of the
# current box. The box *lattice may change* between builds (that is what the
# deformation term of the rebuild criterion is for), but the search radius must
# stay below half the minimum perpendicular width.
function _validate_cache(cache::Neighbors.TilePairCache, q::AbstractMatrix{<:Real},
                         box::AbstractBox, rc::Float64)
    size(q, 2) == cache.n_atoms ||
        throw(ArgumentError("the tile cache was built for $(cache.n_atoms) atoms " *
                            "but q has $(size(q, 2))"))
    cache.rcut == rc ||
        throw(ArgumentError("the tile cache was built for rcut=$(cache.rcut) " *
                            "but this evaluation uses rcut=$rc"))
    if box isa OrthorhombicBox || box isa Types.TriclinicBox
        w = box isa OrthorhombicBox ? box.L : box.widths
        2.0 * Neighbors.search_radius(cache) < minimum(w) ||
            throw(ArgumentError("the tile cache search radius " *
                                "rcut+skin=$(Neighbors.search_radius(cache)) is not below " *
                                "half the minimum box width $(0.5 * minimum(w))"))
    end
    return nothing
end

function _validate_cache(cache::Neighbors.PairCache, q::AbstractMatrix{<:Real},
                         box::AbstractBox, rc::Float64)
    size(q, 2) == size(cache.build_positions, 2) ||
        throw(ArgumentError("the neighbour cache was built for " *
                            "$(size(cache.build_positions, 2)) atoms but q has " *
                            "$(size(q, 2))"))
    cache.rcut == rc ||
        throw(ArgumentError("the neighbour cache was built for rcut=$(cache.rcut) " *
                            "but this evaluation uses rcut=$rc"))
    if box isa OrthorhombicBox || box isa Types.TriclinicBox
        w = box isa OrthorhombicBox ? box.L : box.widths
        2.0 * Neighbors.search_radius(cache) < minimum(w) ||
            throw(ArgumentError("the neighbour cache search radius " *
                                "rcut+skin=$(Neighbors.search_radius(cache)) is not below " *
                                "half the minimum box width $(0.5 * minimum(w))"))
    end
    return nothing
end

# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

function _validate_positions(q::AbstractMatrix{<:Real})
    size(q, 1) == 3 ||
        throw(ArgumentError("q must be 3 x N (columns are atoms), got $(size(q))"))
    @inbounds for j in axes(q, 2), d in 1:3
        isfinite(q[d, j]) ||
            throw(ArgumentError("q[$d,$j] is not finite ($(q[d, j]))"))
    end
    return nothing
end

function _validate_cutoff(box::AbstractBox, rcut::Real, skin::Real)
    (box isa OrthorhombicBox || box isa Types.TriclinicBox) ||
        throw(ArgumentError("StreamedLJ requires an orthorhombic or triclinic periodic box; got $(typeof(box))"))
    rc = Float64(rcut)
    sk = Float64(skin)
    (isfinite(rc) && rc > 0) ||
        throw(ArgumentError("rcut must be finite and > 0, got $rcut"))
    (isfinite(sk) && sk >= 0) ||
        throw(ArgumentError("skin must be finite and >= 0, got $skin"))
    widths = box isa OrthorhombicBox ? box.L : box.widths
    minW = min(widths[1], widths[2], widths[3])
    rc < 0.5 * minW ||
        throw(ArgumentError("rcut=$rc must be < min(L)/2=$(0.5 * minW); " *
                            "the minimum-image/cell stencil is not valid otherwise"))
    return rc, sk
end

function _validate_parameters(epsilon::AbstractVector{<:Real},
                              sigma::AbstractVector{<:Real}, N::Int)
    length(epsilon) == N ||
        throw(ArgumentError("epsilon has length $(length(epsilon)); expected N=$N"))
    length(sigma) == N ||
        throw(ArgumentError("sigma has length $(length(sigma)); expected N=$N"))
    @inbounds for i in 1:N
        e = epsilon[i]
        s = sigma[i]
        (isfinite(e) && e >= 0) ||
            throw(ArgumentError("epsilon[$i] must be finite and >= 0, got $e"))
        # A zero-size atom is physical only when it has no dispersion energy
        # (AMBER/TIP3P hydrogens). The pair energy is exactly zero then, so no
        # fake positive sigma is needed; a positive epsilon with zero sigma is
        # still nonsense and refused.
        (isfinite(s) && s > 0 || (s == 0 && e == 0)) ||
            throw(ArgumentError("sigma[$i] must be finite and > 0, or exactly 0 " *
                                "when epsilon[$i] == 0; got sigma=$s epsilon=$e"))
    end
    return nothing
end

function _validate_gradient_parameters(epsilon::AbstractVector{<:Real})
    @inbounds for i in eachindex(epsilon)
        epsilon[i] > 0 ||
            throw(ArgumentError("parameter gradients require epsilon[$i] > 0 " *
                                "(sqrt(eps_i*eps_j) is not differentiable at 0); " *
                                "got $(epsilon[i])"))
    end
    return nothing
end

function _validate_capacity(N::Int)
    N <= typemax(Int32) ||
        throw(ArgumentError("N=$N exceeds the Int32 linked-list capacity " *
                            "($(typemax(Int32)))"))
    return nothing
end

function _validate_mode(mode)
    throw(ArgumentError("mode must be one of $(SUPPORTED_CUTOFF_MODES), got $(repr(mode))"))
end

function _validate_mode(mode::Symbol)
    mode in SUPPORTED_CUTOFF_MODES ||
        throw(ArgumentError("mode must be one of $(SUPPORTED_CUTOFF_MODES), " *
                            "got $(repr(mode))"))
    return nothing
end

_validate_topology(::Nothing, N::Int) = nothing

function _validate_topology(topo::PairTopology, N::Int)
    topo.max_atom <= N ||
        throw(ArgumentError("topology references atom $(topo.max_atom), but the " *
                            "system has N=$N; the topology and coordinate set do " *
                            "not match"))
    return nothing
end

function _validate_topology(topology, ::Int)
    throw(ArgumentError("topology must be `nothing` or a `PairTopology`, got " *
                        "$(typeof(topology))"))
end

# ---------------------------------------------------------------------------
# grid / workspace
# ---------------------------------------------------------------------------

@inline function _grid_nc(widths::NTuple{3,Float64}, rsearch::Float64)
    rsearch > 0 ||
        throw(ArgumentError("rcut + skin must be > 0, got $rsearch"))
    return (max(1, floor(Int, widths[1] / rsearch)),
            max(1, floor(Int, widths[2] / rsearch)),
            max(1, floor(Int, widths[3] / rsearch)))
end

function _checked_ncells(nc::NTuple{3,Int})
    n = 1
    for d in 1:3
        n > typemax(Int) ÷ nc[d] &&
            throw(ArgumentError("cell grid $(nc) is too large to index"))
        n *= nc[d]
    end
    return n
end

function _ensure_workspace!(ws::LJWorkspace, N::Int, ncell::Int)
    if ws.N != N
        ws.N = N
        ws.forces = zeros(Float64, 3, N)
        ws.next = Vector{Int32}(undef, N)
        ws.grad_epsilon = zeros(Float64, N)
        ws.grad_sigma = zeros(Float64, N)
    end
    if length(ws.head) != ncell
        ws.head = zeros(Int32, ncell)
    end
    if length(ws.cell_energy) != ncell
        ws.cell_energy = Vector{Float64}(undef, ncell)
        ws.cell_count = Vector{Int}(undef, ncell)
    end
    if length(ws.scratch) < 27
        ws.scratch = Vector{Int}(undef, 27)
    end
    nthreads = Threads.maxthreadid()
    if length(ws.thread_forces) != nthreads
        ws.thread_forces = [SparseThreadForces() for _ in 1:nthreads]
    end
    if size(ws.box_derivative) != (3, 3)
        ws.box_derivative = zeros(Float64, 3, 3)
    end
    return nothing
end

@inline function _bucket(x::Float64, L::Float64, n::Int)
    # Wrap into [0, L) first so atoms outside the primary cell still bucket
    # correctly; the minimum-image displacement is applied later to every pair.
    xx = x - L * floor(x / L)
    return min(n, max(1, floor(Int, xx / L * n) + 1))
end

@inline function _cell_id(q::AbstractMatrix{<:Real}, box::OrthorhombicBox,
                          nc::NTuple{3,Int}, j::Int)
    L = box.L
    c1 = _bucket(Float64(q[1, j]), L[1], nc[1])
    c2 = _bucket(Float64(q[2, j]), L[2], nc[2])
    c3 = _bucket(Float64(q[3, j]), L[3], nc[3])
    return (c3 - 1) * nc[1] * nc[2] + (c2 - 1) * nc[1] + c1
end

@inline function _cell_id(q::AbstractMatrix{<:Real}, box::Types.TriclinicBox,
                          nc::NTuple{3,Int}, j::Int)
    Hinv = box.Hinv
    q1 = Float64(q[1, j]); q2 = Float64(q[2, j]); q3 = Float64(q[3, j])
    f1 = Hinv[1, 1] * q1 + Hinv[1, 2] * q2 + Hinv[1, 3] * q3
    f2 = Hinv[2, 1] * q1 + Hinv[2, 2] * q2 + Hinv[2, 3] * q3
    f3 = Hinv[3, 1] * q1 + Hinv[3, 2] * q2 + Hinv[3, 3] * q3
    s1 = f1 - floor(f1)
    s2 = f2 - floor(f2)
    s3 = f3 - floor(f3)
    c1 = min(nc[1], max(1, floor(Int, s1 * nc[1]) + 1))
    c2 = min(nc[2], max(1, floor(Int, s2 * nc[2]) + 1))
    c3 = min(nc[3], max(1, floor(Int, s3 * nc[3]) + 1))
    return (c3 - 1) * nc[1] * nc[2] + (c2 - 1) * nc[1] + c1
end

# ---------------------------------------------------------------------------
# streamed pair kernels
# ---------------------------------------------------------------------------

# Each unordered pair is visited by its lower-index cell, or once within a
# cell. The thread-local force matrix receives both endpoint contributions.
@inline function _sparse_add!(buffer::SparseThreadForces, atom::Int,
                              x::Float64, y::Float64, z::Float64)
    index = get(buffer.atom_index, atom, 0)
    if index == 0
        index = length(buffer.active) + 1
        required = 3 * index
        if required > length(buffer.values)
            resize!(buffer.values, max(required, 2 * length(buffer.values)))
        end
        buffer.atom_index[atom] = index
        push!(buffer.active, false)
        buffer.active[index] = true
        push!(buffer.atoms, atom)
        offset = 3 * (index - 1)
        @inbounds begin
            buffer.values[offset + 1] = 0.0
            buffer.values[offset + 2] = 0.0
            buffer.values[offset + 3] = 0.0
        end
    elseif !buffer.active[index]
        buffer.active[index] = true
        push!(buffer.atoms, atom)
        offset = 3 * (index - 1)
        @inbounds begin
            buffer.values[offset + 1] = 0.0
            buffer.values[offset + 2] = 0.0
            buffer.values[offset + 3] = 0.0
        end
    end
    offset = 3 * (index - 1)
    @inbounds begin
        buffer.values[offset + 1] += x
        buffer.values[offset + 2] += y
        buffer.values[offset + 3] += z
    end
    return nothing
end

@inline function _pair_displacement(box::OrthorhombicBox,
                                    dx0::Float64, dy0::Float64, dz0::Float64)
    L = box.L
    dx = dx0 - L[1] * round(dx0 / L[1])
    dy = dy0 - L[2] * round(dy0 / L[2])
    dz = dz0 - L[3] * round(dz0 / L[3])
    r2 = dx * dx + dy * dy + dz * dz
    return dx, dy, dz, r2
end

@inline function _unwrap_displacement(box::Types.TriclinicBox,
                                      dx::Float64, dy::Float64, dz::Float64)
    Hinv = box.Hinv
    H = box.H
    f1 = Hinv[1, 1] * dx + Hinv[1, 2] * dy + Hinv[1, 3] * dz
    f2 = Hinv[2, 1] * dx + Hinv[2, 2] * dy + Hinv[2, 3] * dz
    f3 = Hinv[3, 1] * dx + Hinv[3, 2] * dy + Hinv[3, 3] * dz

    n0_1 = round(Int, f1)
    n0_2 = round(Int, f2)
    n0_3 = round(Int, f3)

    best_x = dx - (H[1, 1] * n0_1 + H[1, 2] * n0_2 + H[1, 3] * n0_3)
    best_y = dy - (H[2, 1] * n0_1 + H[2, 2] * n0_2 + H[2, 3] * n0_3)
    best_z = dz - (H[3, 1] * n0_1 + H[3, 2] * n0_2 + H[3, 3] * n0_3)
    best2 = best_x * best_x + best_y * best_y + best_z * best_z

    sbest = sqrt(best2)
    b1 = sbest / box.widths[1]
    b2 = sbest / box.widths[2]
    b3 = sbest / box.widths[3]

    r1_min = ceil(Int, f1 - b1); r1_max = floor(Int, f1 + b1)
    r2_min = ceil(Int, f2 - b2); r2_max = floor(Int, f2 + b2)
    r3_min = ceil(Int, f3 - b3); r3_max = floor(Int, f3 + b3)

    if r1_min == r1_max && r2_min == r2_max && r3_min == r3_max
        return best_x, best_y, best_z, best2
    end

    for a in r1_min:r1_max, b in r2_min:r2_max, c in r3_min:r3_max
        rx = dx - (H[1, 1] * a + H[1, 2] * b + H[1, 3] * c)
        ry = dy - (H[2, 1] * a + H[2, 2] * b + H[2, 3] * c)
        rz = dz - (H[3, 1] * a + H[3, 2] * b + H[3, 3] * c)
        r2 = rx * rx + ry * ry + rz * rz
        if r2 < best2
            best_x, best_y, best_z = rx, ry, rz
            best2 = r2
        end
    end
    return best_x, best_y, best_z, best2
end

@inline function _pair_displacement(box::Types.TriclinicBox,
                                    dx0::Float64, dy0::Float64, dz0::Float64)
    return _unwrap_displacement(box, dx0, dy0, dz0)
end

@inline function _fractional_displacement(box::OrthorhombicBox,
                                          dx::Float64, dy::Float64, dz::Float64)
    L = box.L
    return dx / L[1], dy / L[2], dz / L[3]
end

@inline function _fractional_displacement(box::Types.TriclinicBox,
                                          dx::Float64, dy::Float64, dz::Float64)
    Hinv = box.Hinv
    wx = Hinv[1, 1] * dx + Hinv[1, 2] * dy + Hinv[1, 3] * dz
    wy = Hinv[2, 1] * dx + Hinv[2, 2] * dy + Hinv[2, 3] * dz
    wz = Hinv[3, 1] * dx + Hinv[3, 2] * dy + Hinv[3, 3] * dz
    return wx, wy, wz
end

@inline function _pair_owned!(F::SparseThreadForces, q::AbstractMatrix{<:Real},
                              epsilon::AbstractVector{<:Real},
                              sigma::AbstractVector{<:Real},
                              box::AbstractBox, i::Int, j::Int,
                              rc::Float64, rc2::Float64, mode::Symbol, topo)
    w = _pair_weight_raw(topo, i, j)
    w == 0.0 && return (0.0, 0)

    dx0 = Float64(q[1, i]) - Float64(q[1, j])
    dy0 = Float64(q[2, i]) - Float64(q[2, j])
    dz0 = Float64(q[3, i]) - Float64(q[3, j])
    dx, dy, dz, r2 = _pair_displacement(box, dx0, dy0, dz0)
    r2 > rc2 && return (0.0, 0)
    r2 < 1e-24 && return (Inf, 1)

    r = sqrt(r2)
    eij = sqrt(Float64(epsilon[i]) * Float64(epsilon[j]))
    sij = 0.5 * (Float64(sigma[i]) + Float64(sigma[j]))
    sij2 = sij * sij
    s6 = (sij2 / r2)^3
    s12 = s6 * s6
    s6c = (sij2 / rc2)^3
    s12c = s6c * s6c
    # `:trunc` is the unshifted 12-6 energy used by AMBER/CHARMM/OpenMM
    # reference evaluations; it is discontinuous at rc and is therefore
    # energy-only for dynamics (`force_consistent` stays false).
    u = mode === :trunc ? 4.0 * eij * (s12 - s6) :
                          4.0 * eij * ((s12 - s6) - (s12c - s6c))
    dudr = 24.0 * eij * (s6 - 2.0 * s12) / r
    if mode === :lfs
        duc = 24.0 * eij * (s6c - 2.0 * s12c) / rc
        u -= duc * (r - rc)
        dudr -= duc
    end
    c = -(w * dudr) / r
    _sparse_add!(F, i, c * dx, c * dy, c * dz)
    _sparse_add!(F, j, -c * dx, -c * dy, -c * dz)
    return (w * u, 1)
end

function _stream_threaded!(ws::LJWorkspace, q::AbstractMatrix{<:Real},
                           epsilon::AbstractVector{<:Real},
                           sigma::AbstractVector{<:Real},
                           box::AbstractBox, nc::NTuple{3,Int},
                           rc::Float64, rc2::Float64, mode::Symbol, topo)
    nc1, nc2, nc3 = nc
    ncell = length(ws.head)
    cell_energy = ws.cell_energy
    cell_count = ws.cell_count
    head, nxt, F = ws.head, ws.next, ws.forces
    for buffer in ws.thread_forces
        @inbounds for atom in buffer.atoms
            buffer.active[buffer.atom_index[atom]] = false
        end
        empty!(buffer.atoms)
    end
    Threads.@threads :static for cell in 1:ncell
        thread_force = ws.thread_forces[Threads.threadid()]
        c1 = (cell - 1) % nc1 + 1
        c2 = ((cell - 1) ÷ nc1) % nc2 + 1
        c3 = (cell - 1) ÷ (nc1 * nc2) + 1
        e = 0.0
        count = 0
        # Canonicalize aliased stencil targets, then let the lower cell ID own
        # every cross-cell pair. Same-cell pairs use i > j.
        @inbounds for dz in -1:1, dy in -1:1, dx in -1:1
            (nc1 == 1 && dx != 0 || nc1 == 2 && dx == -1 ||
             nc2 == 1 && dy != 0 || nc2 == 2 && dy == -1 ||
             nc3 == 1 && dz != 0 || nc3 == 2 && dz == -1) && continue
            t1 = mod1(c1 + dx, nc1)
            t2 = mod1(c2 + dy, nc2)
            t3 = mod1(c3 + dz, nc3)
            target = (t3 - 1) * nc1 * nc2 + (t2 - 1) * nc1 + t1
            target < cell && continue
            i = Int(head[cell])
            while i != 0
                j = Int(head[target])
                while j != 0
                    if (target != cell || i > j)
                        de, dn = _pair_owned!(thread_force, q, epsilon, sigma, box, i, j,
                                              rc, rc2, mode, topo)
                        e += de
                        count += dn
                    end
                    j = Int(nxt[j])
                end
                i = Int(nxt[i])
            end
        end
        cell_energy[cell] = e
        cell_count[cell] = count
    end
    energy = 0.0
    pair_count = 0
    @inbounds for tid in eachindex(ws.thread_forces)
        buffer = ws.thread_forces[tid]
        for index in eachindex(buffer.atoms)
            atom = buffer.atoms[index]
            slot = buffer.atom_index[atom]
            offset = 3 * (slot - 1)
            for d in 1:3
                F[d, atom] += buffer.values[offset + d]
            end
        end
    end
    @inbounds for cell in 1:ncell
        energy += cell_energy[cell]
        pair_count += cell_count[cell]
    end
    return energy, pair_count
end

# ---------------------------------------------------------------------------
# cached-pair kernels
# ---------------------------------------------------------------------------

# Serial traversal of the persistent pair array. The array is sorted by
# `(i, j)`, and pairs inside `rcut + skin` but outside `rcut` return
# `(0.0, false)` without touching any accumulator, so this sum is bitwise
# identical to a rebuild at the same geometry (see `Neighbors.PairCache`).
function _stream_cached!(ws::LJWorkspace, cache::Neighbors.PairCache,
                         q::AbstractMatrix{<:Real},
                         epsilon::AbstractVector{<:Real},
                         sigma::AbstractVector{<:Real},
                         box::AbstractBox,
                         rc::Float64, rc2::Float64, mode::Symbol, topo,
                         ge, gs, G)
    F = ws.forces
    pi = cache.pairs_i
    pj = cache.pairs_j
    energy = 0.0
    pair_count = 0
    @inbounds for k in eachindex(pi)
        u, hit = _pair!(F, q, epsilon, sigma, box, Int(pi[k]), Int(pj[k]),
                        rc, rc2, mode, topo, ge, gs, G)
        energy += u
        pair_count += (hit ? 1 : 0)
    end
    return energy, pair_count
end

# Deterministic parallel traversal: contiguous pair chunks are assigned statically
# to `Threads.@threads :static`, each chunk accumulates into its own sparse force
# buffer and its own energy slot, and the merge order (chunk index, then atom
# insertion order) does not depend on scheduling. Repeated runs at a fixed thread
# count are therefore bit-for-bit reproducible; changing the thread count changes
# only the summation grouping (O(eps) differences).
function _stream_cached_threaded!(ws::LJWorkspace, cache::Neighbors.PairCache,
                                  q::AbstractMatrix{<:Real},
                                  epsilon::AbstractVector{<:Real},
                                  sigma::AbstractVector{<:Real},
                                  box::AbstractBox,
                                  rc::Float64, rc2::Float64, mode::Symbol, topo)
    pi, pj = cache.pairs_i, cache.pairs_j
    np = length(pi)
    F = ws.forces
    for buffer in ws.thread_forces
        @inbounds for atom in buffer.atoms
            buffer.active[buffer.atom_index[atom]] = false
        end
        empty!(buffer.atoms)
    end
    nt = min(Threads.nthreads(), max(np, 1))
    if length(ws.cell_energy) < nt
        resize!(ws.cell_energy, nt)
        resize!(ws.cell_count, nt)
    end
    chunk_energy = ws.cell_energy
    chunk_count = ws.cell_count
    chunk = cld(np, nt)
    Threads.@threads :static for t in 1:nt
        lo = (t - 1) * chunk + 1
        hi = min(t * chunk, np)
        thread_force = ws.thread_forces[Threads.threadid()]
        e = 0.0
        count = 0
        @inbounds for k in lo:hi
            u, hit = _pair_owned!(thread_force, q, epsilon, sigma, box,
                                  Int(pi[k]), Int(pj[k]), rc, rc2, mode, topo)
            e += u
            count += hit
        end
        chunk_energy[t] = e
        chunk_count[t] = count
    end
    energy = 0.0
    pair_count = 0
    @inbounds for t in 1:nt
        energy += chunk_energy[t]
        pair_count += chunk_count[t]
    end
    @inbounds for tid in eachindex(ws.thread_forces)
        buffer = ws.thread_forces[tid]
        for index in eachindex(buffer.atoms)
            atom = buffer.atoms[index]
            slot = buffer.atom_index[atom]
            offset = 3 * (slot - 1)
            for d in 1:3
                F[d, atom] += buffer.values[offset + d]
            end
        end
    end
    return energy, pair_count
end

# ---------------------------------------------------------------------------
# 64-bit Fixed-Point (Int64) Force & Tiled Cluster Kernels
# ---------------------------------------------------------------------------

@inline function _float_to_fixed(x::Float64)::Int64
    y = x * FIXED_POINT_SCALE
    if y >= 4.611686018427388e18
        return Int64(4611686018427387904)
    elseif y <= -4.611686018427388e18
        return Int64(-4611686018427387904)
    end
    return round(Int64, y)
end

@inline function _pair_fixed_point!(F_fixed::AbstractMatrix{Int64}, q::AbstractMatrix{<:Real},
                                    epsilon::AbstractVector{<:Real},
                                    sigma::AbstractVector{<:Real},
                                    box::AbstractBox, i::Int, j::Int,
                                    rc::Float64, rc2::Float64, mode::Symbol, topo)
    # Canonicalize (i, j) so that floating-point subtraction q[:, i] - q[:, j]
    # and force rounding are identical regardless of tile visit order!
    if i > j
        i, j = j, i
    end
    w = _pair_weight_raw(topo, i, j)
    w == 0.0 && return (0.0, 0)

    dx0 = Float64(q[1, i]) - Float64(q[1, j])
    dy0 = Float64(q[2, i]) - Float64(q[2, j])
    dz0 = Float64(q[3, i]) - Float64(q[3, j])
    dx, dy, dz, r2 = _pair_displacement(box, dx0, dy0, dz0)
    r2 > rc2 && return (0.0, 0)
    r2 < 1e-24 && return (Inf, 1)

    r = sqrt(r2)
    eij = sqrt(Float64(epsilon[i]) * Float64(epsilon[j]))
    sij = 0.5 * (Float64(sigma[i]) + Float64(sigma[j]))
    sij2 = sij * sij
    s6 = (sij2 / r2)^3
    s12 = s6 * s6
    s6c = (sij2 / rc2)^3
    s12c = s6c * s6c
    u = mode === :trunc ? 4.0 * eij * (s12 - s6) :
                          4.0 * eij * ((s12 - s6) - (s12c - s6c))
    dudr = 24.0 * eij * (s6 - 2.0 * s12) / r
    if mode === :lfs
        duc = 24.0 * eij * (s6c - 2.0 * s12c) / rc
        u -= duc * (r - rc)
        dudr -= duc
    end
    c = -(w * dudr) / r
    fx_i = _float_to_fixed(c * dx)
    fy_i = _float_to_fixed(c * dy)
    fz_i = _float_to_fixed(c * dz)
    @inbounds begin
        F_fixed[1, i] += fx_i; F_fixed[2, i] += fy_i; F_fixed[3, i] += fz_i
        F_fixed[1, j] -= fx_i; F_fixed[2, j] -= fy_i; F_fixed[3, j] -= fz_i
    end
    return (w * u, 1)
end

@inline function _pair_fixed_point_sparse!(buffer::SparseThreadForces, q::AbstractMatrix{<:Real},
                                           epsilon::AbstractVector{<:Real},
                                           sigma::AbstractVector{<:Real},
                                           box::AbstractBox, i::Int, j::Int,
                                           rc::Float64, rc2::Float64, mode::Symbol, topo)
    if i > j
        i, j = j, i
    end
    w = _pair_weight_raw(topo, i, j)
    w == 0.0 && return (0.0, 0)

    dx0 = Float64(q[1, i]) - Float64(q[1, j])
    dy0 = Float64(q[2, i]) - Float64(q[2, j])
    dz0 = Float64(q[3, i]) - Float64(q[3, j])
    dx, dy, dz, r2 = _pair_displacement(box, dx0, dy0, dz0)
    r2 > rc2 && return (0.0, 0)
    r2 < 1e-24 && return (Inf, 1)

    r = sqrt(r2)
    eij = sqrt(Float64(epsilon[i]) * Float64(epsilon[j]))
    sij = 0.5 * (Float64(sigma[i]) + Float64(sigma[j]))
    sij2 = sij * sij
    s6 = (sij2 / r2)^3
    s12 = s6 * s6
    s6c = (sij2 / rc2)^3
    s12c = s6c * s6c
    u = mode === :trunc ? 4.0 * eij * (s12 - s6) :
                          4.0 * eij * ((s12 - s6) - (s12c - s6c))
    dudr = 24.0 * eij * (s6 - 2.0 * s12) / r
    if mode === :lfs
        duc = 24.0 * eij * (s6c - 2.0 * s12c) / rc
        u -= duc * (r - rc)
        dudr -= duc
    end
    c = -(w * dudr) / r
    # Store exact integer-rounded values as Float64 in SparseThreadForces (Float64 has 53-bit exact integer mantissa)
    fx = Float64(_float_to_fixed(c * dx))
    fy = Float64(_float_to_fixed(c * dy))
    fz = Float64(_float_to_fixed(c * dz))
    _sparse_add!(buffer, i, fx, fy, fz)
    _sparse_add!(buffer, j, -fx, -fy, -fz)
    return (w * u, 1)
end

function _stream_cached_fixed_point!(ws::LJWorkspace, cache::Neighbors.PairCache,
                                     q::AbstractMatrix{<:Real},
                                     epsilon::AbstractVector{<:Real},
                                     sigma::AbstractVector{<:Real},
                                     box::AbstractBox,
                                     rc::Float64, rc2::Float64, mode::Symbol, topo,
                                     threaded::Bool)
    pi, pj = cache.pairs_i, cache.pairs_j
    np = length(pi)
    N = size(q, 2)
    F = ws.forces
    F_fixed = zeros(Int64, 3, N)
    if threaded && Threads.nthreads() > 1 && np > 0
        for buffer in ws.thread_forces
            @inbounds for atom in buffer.atoms
                buffer.active[buffer.atom_index[atom]] = false
            end
            empty!(buffer.atoms)
        end
        nt = min(Threads.nthreads(), max(np, 1))
        if length(ws.cell_energy) < nt
            resize!(ws.cell_energy, nt)
            resize!(ws.cell_count, nt)
        end
        chunk_energy = ws.cell_energy
        chunk_count = ws.cell_count
        chunk = cld(np, nt)
        Threads.@threads :static for t in 1:nt
            lo = (t - 1) * chunk + 1
            hi = min(t * chunk, np)
            thread_force = ws.thread_forces[Threads.threadid()]
            e_fx = Int64(0)
            count = 0
            @inbounds for k in lo:hi
                u, hit = _pair_fixed_point_sparse!(thread_force, q, epsilon, sigma, box,
                                                   Int(pi[k]), Int(pj[k]), rc, rc2, mode, topo)
                e_fx += _float_to_fixed(u)
                count += hit
            end
            chunk_energy[t] = Float64(e_fx)
            chunk_count[t] = count
        end
        energy_fx = Int64(0)
        pair_count = 0
        @inbounds for t in 1:nt
            energy_fx += Int64(chunk_energy[t])
            pair_count += chunk_count[t]
        end
        energy = Float64(energy_fx) * INV_FIXED_POINT_SCALE
        @inbounds for tid in eachindex(ws.thread_forces)
            buffer = ws.thread_forces[tid]
            for index in eachindex(buffer.atoms)
                atom = buffer.atoms[index]
                slot = buffer.atom_index[atom]
                offset = 3 * (slot - 1)
                for d in 1:3
                    F_fixed[d, atom] += Int64(buffer.values[offset + d])
                end
            end
        end
    else
        energy_fx = Int64(0)
        pair_count = 0
        @inbounds for k in 1:np
            u, hit = _pair_fixed_point!(F_fixed, q, epsilon, sigma, box,
                                        Int(pi[k]), Int(pj[k]), rc, rc2, mode, topo)
            energy_fx += _float_to_fixed(u)
            pair_count += hit
        end
        energy = Float64(energy_fx) * INV_FIXED_POINT_SCALE
    end
    @inbounds for j in 1:N, d in 1:3
        F[d, j] = Float64(F_fixed[d, j]) * INV_FIXED_POINT_SCALE
    end
    return energy, pair_count
end

function _stream_tiled!(ws::LJWorkspace, tc::Neighbors.TilePairCache,
                        q::AbstractMatrix{<:Real},
                        epsilon::AbstractVector{<:Real},
                        sigma::AbstractVector{<:Real},
                        box::AbstractBox,
                        rc::Float64, rc2::Float64, mode::Symbol, topo,
                        ge, gs, G)
    F = ws.forces
    tpi = tc.tile_pairs_i
    tpj = tc.tile_pairs_j
    ord = tc.order
    tstarts = tc.tile_starts
    tends = tc.tile_ends
    energy = 0.0
    pair_count = 0
    @inbounds for k in eachindex(tpi)
        ti = Int(tpi[k])
        tj = Int(tpj[k])
        si, ei = Int(tstarts[ti]), Int(tends[ti])
        if ti == tj
            for a in si:(ei - 1)
                i = Int(ord[a])
                for b in (a + 1):ei
                    j = Int(ord[b])
                    ii, jj = i < j ? (i, j) : (j, i)
                    u, hit = _pair!(F, q, epsilon, sigma, box, ii, jj, rc, rc2, mode, topo, ge, gs, G)
                    energy += u
                    pair_count += (hit ? 1 : 0)
                end
            end
        else
            sj, ej = Int(tstarts[tj]), Int(tends[tj])
            for a in si:ei
                i = Int(ord[a])
                for b in sj:ej
                    j = Int(ord[b])
                    ii, jj = i < j ? (i, j) : (j, i)
                    u, hit = _pair!(F, q, epsilon, sigma, box, ii, jj, rc, rc2, mode, topo, ge, gs, G)
                    energy += u
                    pair_count += (hit ? 1 : 0)
                end
            end
        end
    end
    return energy, pair_count
end

function _stream_tiled_fixed_point!(ws::LJWorkspace, tc::Neighbors.TilePairCache,
                                    q::AbstractMatrix{<:Real},
                                    epsilon::AbstractVector{<:Real},
                                    sigma::AbstractVector{<:Real},
                                    box::AbstractBox,
                                    rc::Float64, rc2::Float64, mode::Symbol, topo,
                                    threaded::Bool)
    tpi = tc.tile_pairs_i
    tpj = tc.tile_pairs_j
    ord = tc.order
    tstarts = tc.tile_starts
    tends = tc.tile_ends
    ntp = length(tpi)
    N = size(q, 2)
    F = ws.forces
    F_fixed = zeros(Int64, 3, N)

    if length(ws.cell_energy) < ntp
        resize!(ws.cell_energy, ntp)
        resize!(ws.cell_count, ntp)
    end
    tile_energy = ws.cell_energy
    tile_count = ws.cell_count

    if threaded && Threads.nthreads() > 1 && ntp > 0
        for buffer in ws.thread_forces
            @inbounds for atom in buffer.atoms
                buffer.active[buffer.atom_index[atom]] = false
            end
            empty!(buffer.atoms)
        end
        Threads.@threads :static for k in 1:ntp
            thread_force = ws.thread_forces[Threads.threadid()]
            ti = Int(tpi[k])
            tj = Int(tpj[k])
            si, ei = Int(tstarts[ti]), Int(tends[ti])
            e = 0.0
            cnt = 0
            if ti == tj
                @inbounds for a in si:(ei - 1)
                    i = Int(ord[a])
                    for b in (a + 1):ei
                        j = Int(ord[b])
                        u, hit = _pair_fixed_point_sparse!(thread_force, q, epsilon, sigma, box,
                                                           i, j, rc, rc2, mode, topo)
                        e += u
                        cnt += hit
                    end
                end
            else
                sj, ej = Int(tstarts[tj]), Int(tends[tj])
                @inbounds for a in si:ei
                    i = Int(ord[a])
                    for b in sj:ej
                        j = Int(ord[b])
                        u, hit = _pair_fixed_point_sparse!(thread_force, q, epsilon, sigma, box,
                                                           i, j, rc, rc2, mode, topo)
                        e += u
                        cnt += hit
                    end
                end
            end
            tile_energy[k] = e
            tile_count[k] = cnt
        end
        @inbounds for tid in eachindex(ws.thread_forces)
            buffer = ws.thread_forces[tid]
            for index in eachindex(buffer.atoms)
                atom = buffer.atoms[index]
                slot = buffer.atom_index[atom]
                offset = 3 * (slot - 1)
                for d in 1:3
                    F_fixed[d, atom] += Int64(buffer.values[offset + d])
                end
            end
        end
    else
        @inbounds for k in 1:ntp
            ti = Int(tpi[k])
            tj = Int(tpj[k])
            si, ei = Int(tstarts[ti]), Int(tends[ti])
            e = 0.0
            cnt = 0
            if ti == tj
                for a in si:(ei - 1)
                    i = Int(ord[a])
                    for b in (a + 1):ei
                        j = Int(ord[b])
                        u, hit = _pair_fixed_point!(F_fixed, q, epsilon, sigma, box,
                                                    i, j, rc, rc2, mode, topo)
                        e += u
                        cnt += hit
                    end
                end
            else
                sj, ej = Int(tstarts[tj]), Int(tends[tj])
                for a in si:ei
                    i = Int(ord[a])
                    for b in sj:ej
                        j = Int(ord[b])
                        u, hit = _pair_fixed_point!(F_fixed, q, epsilon, sigma, box,
                                                    i, j, rc, rc2, mode, topo)
                        e += u
                        cnt += hit
                    end
                end
            end
            tile_energy[k] = e
            tile_count[k] = cnt
        end
    end

    # Sum tile energies in canonical tile-pair order (1:ntp) so energy is bit-identical across thread counts!
    energy = 0.0
    pair_count = 0
    @inbounds for k in 1:ntp
        energy += tile_energy[k]
        pair_count += tile_count[k]
    end
    @inbounds for j in 1:N, d in 1:3
        F[d, j] = Float64(F_fixed[d, j]) * INV_FIXED_POINT_SCALE
    end
    return energy, pair_count
end

@inline function _pair!(F::Matrix{Float64}, q::AbstractMatrix{<:Real},
                        epsilon::AbstractVector{<:Real},
                        sigma::AbstractVector{<:Real},
                        box::AbstractBox, i::Int, j::Int,
                        rc::Float64, rc2::Float64, mode::Symbol,
                        topo, grad_epsilon, grad_sigma, G)
    # An excluded pair is skipped before any geometry is computed. This also
    # keeps excluded overlapping pairs out of the coincident-atom Inf guard.
    w = _pair_weight_raw(topo, i, j)
    w == 0.0 && return (0.0, false)

    dx0 = Float64(q[1, i]) - Float64(q[1, j])
    dy0 = Float64(q[2, i]) - Float64(q[2, j])
    dz0 = Float64(q[3, i]) - Float64(q[3, j])
    dx, dy, dz, r2 = _pair_displacement(box, dx0, dy0, dz0)
    r2 > rc2 && return (0.0, false)
    r2 < 1e-24 && return (Inf, true)  # coincident atoms: no finite force

    r = sqrt(r2)
    eij = sqrt(Float64(epsilon[i]) * Float64(epsilon[j]))
    sij = 0.5 * (Float64(sigma[i]) + Float64(sigma[j]))
    sij2 = sij * sij

    s6 = (sij2 / r2)^3
    s12 = s6 * s6
    s6c = (sij2 / rc2)^3
    s12c = s6c * s6c

    # Raw LJ energy and dU/dr at r. `:trunc` is unshifted (reference parity).
    u = mode === :trunc ? 4.0 * eij * (s12 - s6) :
                          4.0 * eij * ((s12 - s6) - (s12c - s6c))
    dudr = 24.0 * eij * (s6 - 2.0 * s12) / r
    if mode === :lfs
        # Linear-force shift: subtract U(rc) and the tangent at rc. This leaves
        # u(rc)=0 and dudr(rc)=0, so both energy and force are C1 at the cutoff.
        duc = 24.0 * eij * (s6c - 2.0 * s12c) / rc
        u -= duc * (r - rc)
        dudr -= duc
    end

    u *= w
    c = -(w * dudr) / r
    cdx = c * dx; cdy = c * dy; cdz = c * dz
    F[1, i] += cdx
    F[2, i] += cdy
    F[3, i] += cdz
    F[1, j] -= cdx
    F[2, j] -= cdy
    F[3, j] -= cdz

    if grad_epsilon !== nothing
        # U_eff is linear in eps_ij = sqrt(eps_i eps_j), so
        # dU/deps_i = U_eff / (2 eps_i). epsilon[i] > 0 is validated up front.
        grad_epsilon[i] += u / (2.0 * Float64(epsilon[i]))
        grad_epsilon[j] += u / (2.0 * Float64(epsilon[j]))

        # dU_eff/dsig_ij, including the linear-force-shift correction.
        dsig = (4.0 * eij / sij) * ((12.0 * s12 - 6.0 * s6) -
                                    (12.0 * s12c - 6.0 * s6c))
        if mode === :lfs
            dsig -= (r - rc) * ((24.0 * eij / (sij * rc)) *
                                (6.0 * s6c - 24.0 * s12c))
        end
        dsig *= w
        # sig_ij = (sig_i + sig_j)/2, so each atom gets half the pair derivative.
        grad_sigma[i] += 0.5 * dsig
        grad_sigma[j] += 0.5 * dsig
    end

    if G !== nothing
        wx, wy, wz = _fractional_displacement(box, dx, dy, dz)
        gx = -cdx; gy = -cdy; gz = -cdz
        G[1, 1] += gx * wx; G[1, 2] += gx * wy; G[1, 3] += gx * wz
        G[2, 1] += gy * wx; G[2, 2] += gy * wy; G[2, 3] += gy * wz
        G[3, 1] += gz * wx; G[3, 2] += gz * wy; G[3, 3] += gz * wz
    end

    return (u, true)
end

# Same cell: the linked list is strictly descending in atom index, so starting
# each inner walk at next[i] enumerates i > j exactly once.
@inline function _same_cell_pairs!(F::Matrix{Float64}, q::AbstractMatrix{<:Real},
                                   epsilon::AbstractVector{<:Real},
                                   sigma::AbstractVector{<:Real},
                                   box::AbstractBox,
                                   head::Vector{Int32}, nxt::Vector{Int32},
                                   c::Int, rc::Float64, rc2::Float64,
                                   mode::Symbol, topo, ge, gs, G)
    energy = 0.0
    np = 0
    i = Int(head[c])
    while i != 0
        j = Int(nxt[i])
        while j != 0
            u, hit = _pair!(F, q, epsilon, sigma, box, i, j, rc, rc2, mode, topo, ge, gs, G)
            energy += u
            np += (hit ? 1 : 0)
            j = Int(nxt[j])
        end
        i = Int(nxt[i])
    end
    return energy, np
end

# Distinct cells in the half-stencil: the cell pair is already unique, so every
# atom in source c is paired with every atom in target t.
@inline function _cross_all_pairs!(F::Matrix{Float64}, q::AbstractMatrix{<:Real},
                                   epsilon::AbstractVector{<:Real},
                                   sigma::AbstractVector{<:Real},
                                   box::AbstractBox,
                                   head::Vector{Int32}, nxt::Vector{Int32},
                                   c::Int, t::Int, rc::Float64, rc2::Float64,
                                   mode::Symbol, topo, ge, gs, G)
    energy = 0.0
    np = 0
    i = Int(head[c])
    while i != 0
        j = Int(head[t])
        while j != 0
            u, hit = _pair!(F, q, epsilon, sigma, box, i, j, rc, rc2, mode, topo, ge, gs, G)
            energy += u
            np += (hit ? 1 : 0)
            j = Int(nxt[j])
        end
        i = Int(nxt[i])
    end
    return energy, np
end

# Distinct cells in the full stencil (an nc<3 degeneracy): process a pair only
# from the cell of the smaller atom index, so the two source directions cannot
# double count.
@inline function _cross_ordered_pairs!(F::Matrix{Float64}, q::AbstractMatrix{<:Real},
                                       epsilon::AbstractVector{<:Real},
                                       sigma::AbstractVector{<:Real},
                                       box::AbstractBox,
                                       head::Vector{Int32}, nxt::Vector{Int32},
                                       c::Int, t::Int, rc::Float64, rc2::Float64,
                                       mode::Symbol, topo, ge, gs, G)
    energy = 0.0
    np = 0
    i = Int(head[c])
    while i != 0
        j = Int(head[t])
        while j != 0
            if j > i
                u, hit = _pair!(F, q, epsilon, sigma, box, i, j, rc, rc2,
                                mode, topo, ge, gs, G)
                energy += u
                np += (hit ? 1 : 0)
            end
            j = Int(nxt[j])
        end
        i = Int(nxt[i])
    end
    return energy, np
end

function _stream_half!(ws::LJWorkspace, q::AbstractMatrix{<:Real},
                       epsilon::AbstractVector{<:Real},
                       sigma::AbstractVector{<:Real},
                       box::AbstractBox, nc::NTuple{3,Int},
                       rc::Float64, rc2::Float64, mode::Symbol, topo, ge, gs, G)
    F = ws.forces
    head = ws.head
    nxt = ws.next
    nc1, nc2, nc3 = nc
    energy = 0.0
    pair_count = 0
    @inbounds for c3 in 1:nc3, c2 in 1:nc2, c1 in 1:nc1
        c = (c3 - 1) * nc1 * nc2 + (c2 - 1) * nc1 + c1
        for off in _HALF_OFFSETS
            t1 = mod1(c1 + off[1], nc1)
            t2 = mod1(c2 + off[2], nc2)
            t3 = mod1(c3 + off[3], nc3)
            t = (t3 - 1) * nc1 * nc2 + (t2 - 1) * nc1 + t1
            if t == c
                de, dn = _same_cell_pairs!(F, q, epsilon, sigma, box, head, nxt, c,
                                           rc, rc2, mode, topo, ge, gs, G)
            else
                de, dn = _cross_all_pairs!(F, q, epsilon, sigma, box, head, nxt, c, t,
                                           rc, rc2, mode, topo, ge, gs, G)
            end
            energy += de
            pair_count += dn
        end
    end
    return energy, pair_count
end

function _stream_full!(ws::LJWorkspace, q::AbstractMatrix{<:Real},
                       epsilon::AbstractVector{<:Real},
                       sigma::AbstractVector{<:Real},
                       box::AbstractBox, nc::NTuple{3,Int},
                       rc::Float64, rc2::Float64, mode::Symbol, topo, ge, gs, G)
    F = ws.forces
    head = ws.head
    nxt = ws.next
    scratch = ws.scratch
    nc1, nc2, nc3 = nc
    energy = 0.0
    pair_count = 0
    @inbounds for c3 in 1:nc3, c2 in 1:nc2, c1 in 1:nc1
        c = (c3 - 1) * nc1 * nc2 + (c2 - 1) * nc1 + c1
        nd = 0
        for off in _FULL_OFFSETS
            t1 = mod1(c1 + off[1], nc1)
            t2 = mod1(c2 + off[2], nc2)
            t3 = mod1(c3 + off[3], nc3)
            t = (t3 - 1) * nc1 * nc2 + (t2 - 1) * nc1 + t1

            dup = false
            for s in 1:nd
                if scratch[s] == t
                    dup = true
                    break
                end
            end
            dup && continue
            nd += 1
            scratch[nd] = t

            if t == c
                de, dn = _same_cell_pairs!(F, q, epsilon, sigma, box, head, nxt, c,
                                           rc, rc2, mode, topo, ge, gs, G)
            else
                de, dn = _cross_ordered_pairs!(F, q, epsilon, sigma, box, head, nxt,
                                               c, t, rc, rc2, mode, topo, ge, gs, G)
            end
            energy += de
            pair_count += dn
        end
    end
    return energy, pair_count
end

# ---------------------------------------------------------------------------
# factor-protocol adapter
# ---------------------------------------------------------------------------

"""
    StreamedLJFactor(q, epsilon, sigma, box; rcut, skin = 0.0,
                     mode = DEFAULT_CUTOFF_MODE, topology = nothing,
                     threaded = false, cache = false, cache_skin = nothing)

A `Factors.AbstractFactor` front end for the streamed kernel.

All atoms are in the read-support (`factor_support_size == N`,
`is_local == false`), and `evaluate!` adds its scattered force to the supplied
accumulator, so it composes with bonded factors without double counting. The
factor owns a reusable [`LJWorkspace`](@ref), a *copy* of the per-atom
`epsilon`/`sigma` vectors, and a validated [`PairTopology`](@ref). `q` fixes
`N` and the initial cell grid; later calls may pass any configuration with the
same `N` and the same box lengths.

`threaded=true` opts into the deterministic per-call force threading path.
`cache=true` opts into a persistent `Neighbors.PairCache`; set `cache_skin` to a
positive buffer length, or leave it `nothing` to use the calculated default.
Caching is off by default. The factor's Dynamics context remains fixed to its
construction box; PairCache's changing-box safety bound does not make this
factor an NPT/barostat interface. Set `cache=false` to disable caching at
construction; changing `f.cache` later does not create or remove `f.pair_cache`.

Validation happens before any output is touched. The constructor rejects bad
lengths, non-positive `sigma`, negative/non-finite `epsilon`, unsupported
cutoff modes, non-`PairTopology` topologies, cutoffs at or above `min(L)/2`,
negative skins, and non-orthorhombic boxes. Each `evaluate!` revalidates the
factor state and the context (atom count, box lengths, mode, topology, cutoff),
so mutating a field cannot bypass the checks.

`force_consistent` is `true` only for `mode = :lfs`. A `:shift` factor is still
usable for energy evaluation, but `Dynamics.check_force_model` refuses to
integrate it — the shift force is not the derivative of the energy at `rc`.

The accumulated force is exactly additive: `evaluate!(F, f, q, ctx)` returns the
factor energy and does `F += -∂U/∂q`. `F` must be `3 x N` and must not alias
`q` or the factor's internal workspace matrix; both cases are refused with
`ArgumentError` rather than silently zeroing or corrupting prior contributions.
"""
mutable struct StreamedLJFactor <: Factors.AbstractFactor
    N::Int
    epsilon::Vector{Float64}
    sigma::Vector{Float64}
    box::AbstractBox
    rcut::Float64
    skin::Float64
    mode::Symbol
    topology::Union{Nothing,PairTopology}
    ws::LJWorkspace
    threaded::Bool
    cache::Bool
    cache_skin::Float64
    pair_cache::Union{Nothing,Neighbors.PairCache,Neighbors.TilePairCache}
    fixed_point::Bool
end

StreamedLJFactor(N::Int, epsilon::Vector{Float64}, sigma::Vector{Float64},
                 box::AbstractBox, rcut::Float64, skin::Float64, mode::Symbol,
                 topology::Union{Nothing,PairTopology}, ws::LJWorkspace,
                 threaded::Bool, cache::Bool, cache_skin::Float64,
                 pair_cache::Union{Nothing,Neighbors.PairCache,Neighbors.TilePairCache}) =
    StreamedLJFactor(N, epsilon, sigma, box, rcut, skin, mode, topology, ws,
                     threaded, cache, cache_skin, pair_cache, false)

"""
Largest neighbour-list skin that still keeps `rc + skin` inside the minimum-image
stencil of `box`: `0.9 * (min(widths)/2 - rcut)`, capped at `0.3 * rcut`.

The cap is the usual skin/cutoff ratio; the box term is the hard constraint,
because the cell stencil is only guaranteed to find every pair when the search
radius is below half the minimum perpendicular width. Returns `0.0` when the box
leaves no room, in which case the persistent list cannot amortize a rebuild and
the factor falls back to a per-call cell list.
"""
function default_pair_cache_skin(box::AbstractBox, rcut::Float64;
                                 ratio::Float64 = 0.3)
    (box isa OrthorhombicBox || box isa Types.TriclinicBox) || return 0.0
    w = box isa OrthorhombicBox ? box.L : box.widths
    headroom = 0.5 * minimum(w) - rcut
    headroom > 0.0 || return 0.0
    s = min(ratio * rcut, 0.9 * headroom)
    return s > 0.0 ? s : 0.0
end

function StreamedLJFactor(q::AbstractMatrix{<:Real},
                          epsilon::AbstractVector{<:Real},
                          sigma::AbstractVector{<:Real},
                          box::AbstractBox;
                          rcut::Real, skin::Real = 0.0,
                          mode = DEFAULT_CUTOFF_MODE,
                          topology = nothing,
                          threaded::Bool = false,
                          cache::Bool = false,
                          tile_cache::Bool = false,
                          cache_skin::Union{Nothing,Real} = nothing,
                          fixed_point::Bool = false)
    _validate_positions(q)
    N = size(q, 2)
    length(epsilon) == N ||
        throw(ArgumentError("epsilon has length $(length(epsilon)); expected N=$N"))
    length(sigma) == N ||
        throw(ArgumentError("sigma has length $(length(sigma)); expected N=$N"))
    _validate_parameters(epsilon, sigma, N)
    _validate_mode(mode)
    _validate_topology(topology, N)
    ws = LJWorkspace(q, box, rcut; skin = skin)
    use_cache = cache || tile_cache
    cs = use_cache ? (cache_skin === nothing ? default_pair_cache_skin(box, Float64(rcut)) :
                      Float64(cache_skin)) : 0.0
    (isfinite(cs) && cs >= 0.0) ||
        throw(ArgumentError("cache_skin must be finite and >= 0, got $cache_skin"))
    pc = nothing
    if use_cache && cs > 0.0
        w = box isa OrthorhombicBox ? box.L : box.widths
        2.0 * (Float64(rcut) + cs) < minimum(w) ||
            throw(ArgumentError("cache_skin=$cs makes rcut+cache_skin exceed half the " *
                                "minimum box width; pass a smaller cache_skin or " *
                                "cache=false"))
        if tile_cache || N > Neighbors.MAX_MATERIALIZED_PAIR_ATOMS
            pc = Neighbors.TilePairCache(Matrix{Float64}(q), box, Float64(rcut), cs)
        else
            pc = Neighbors.PairCache(Matrix{Float64}(q), box, Float64(rcut), cs)
        end
    end
    return StreamedLJFactor(N, collect(Float64, epsilon), collect(Float64, sigma),
                            box, Float64(rcut), Float64(skin), mode, topology, ws,
                            threaded, use_cache, cs, pc, fixed_point)
end

"""
    StreamedLJFactor(N, epsilon, sigma, box; rcut, skin = 0.0,
                     mode = DEFAULT_CUTOFF_MODE, topology = nothing)

Atom-count constructor for callers that do not have a configuration yet. It
allocates the workspace for `N` atoms and then behaves exactly like the
`q`-based constructor; the first `evaluate!` supplies the coordinates.
"""
function StreamedLJFactor(N::Integer, epsilon::AbstractVector{<:Real},
                          sigma::AbstractVector{<:Real}, box::AbstractBox;
                          kwargs...)
    N >= 0 || throw(ArgumentError("N must be >= 0, got $N"))
    _validate_capacity(Int(N))
    return StreamedLJFactor(zeros(Float64, 3, Int(N)), epsilon, sigma, box; kwargs...)
end

Factors.factor_support_size(f::StreamedLJFactor) = f.N
Factors.read_support(f::StreamedLJFactor) = Tuple(1:f.N)
Factors.is_local(::StreamedLJFactor) = false
Factors.force_consistent(f::StreamedLJFactor) = f.mode === :lfs

function Base.show(io::IO, f::StreamedLJFactor)
    print(io, "StreamedLJFactor(N=", f.N, ", mode=", repr(f.mode),
          ", rcut=", f.rcut, ", skin=", f.skin,
          ", topology=", f.topology === nothing ? "nothing" : "PairTopology",
          ", threaded=", f.threaded,
          ", cache=", f.pair_cache === nothing ? "off" : "skin=$(f.cache_skin)")
    return nothing
end

# Validate the static factor state and the dynamic context before any array is
# mutated. Returns the context box for the kernel call.
function _validate_factor_context(f::StreamedLJFactor, q::AbstractMatrix{<:Real},
                                  ctx::Factors.Context)
    size(q, 1) == 3 ||
        throw(ArgumentError("q must be 3 x N, got $(size(q))"))
    size(q, 2) == f.N ||
        throw(ArgumentError("q has $(size(q, 2)) atoms but the factor was built " *
                            "for N=$(f.N)"))
    ctx.N == f.N ||
        throw(ArgumentError("context N=$(ctx.N) does not match factor N=$(f.N)"))
    box = ctx.box
    (box isa OrthorhombicBox || box isa Types.TriclinicBox) ||
        throw(ArgumentError("StreamedLJFactor requires an orthorhombic or triclinic periodic " *
                            "context box; got $(typeof(box))"))
    if box isa OrthorhombicBox
        (f.box isa OrthorhombicBox && box.L == f.box.L) ||
            throw(ArgumentError("context box lengths $(box.L) differ from the " *
                                "factor's box $(box_lengths(f.box)); rebuild the factor for a " *
                                "different box"))
    else
        (f.box isa Types.TriclinicBox && box.H == f.box.H) ||
            throw(ArgumentError("context box lattice differs from the " *
                                "factor's box; rebuild the factor for a different box"))
    end
    _validate_cutoff(box, f.rcut, f.skin)
    _validate_mode(f.mode)
    _validate_topology(f.topology, f.N)
    _validate_parameters(f.epsilon, f.sigma, f.N)
    return box
end

"""
    evaluate!(F, f::StreamedLJFactor, q, ctx) -> U

Factor-protocol entry point. Returns the streamed LJ energy and **adds** the
factor's force to `F` (sign convention `F = -∂U/∂q`). Every other factor's
contributions in `F` are preserved; `F` must be `3 x N` and must not be the
factor's internal workspace matrix.
"""
function Factors.evaluate!(F::AbstractMatrix{Float64}, f::StreamedLJFactor,
                           q::AbstractMatrix{Float64}, ctx::Factors.Context)
    size(F) == (3, f.N) ||
        throw(ArgumentError("F must be 3 x N = (3, $(f.N)), got $(size(F))"))
    Base.mightalias(F, f.ws.forces) &&
        throw(ArgumentError("the force accumulator aliases the factor's internal " *
                            "workspace; pass a separate 3 x N matrix"))
    Base.mightalias(F, q) &&
        throw(ArgumentError("the force accumulator must not alias the coordinate " *
                            "matrix q"))
    Base.mightalias(q, f.ws.forces) &&
        throw(ArgumentError("q aliases the factor's internal force workspace; " *
                            "pass a separate coordinate matrix"))
    box = _validate_factor_context(f, q, ctx)
    energy, _, _ = evaluate!(f.ws, q, f.epsilon, f.sigma, box;
                              rcut = f.rcut, skin = f.skin, mode = f.mode,
                              topology = f.topology, threaded = f.threaded,
                              cache = f.pair_cache, fixed_point = f.fixed_point)
    @inbounds for j in 1:f.N, d in 1:3
        F[d, j] += f.ws.forces[d, j]
    end
    return energy
end

"""
    streamed_energy(f, q, ctx; epsilon = f.epsilon, sigma = f.sigma) -> U

Evaluate the factor's effective potential without touching a caller force
accumulator. The optional `epsilon`/`sigma` overrides change only the mixing
parameters: `rcut`, `skin`, `mode`, the box, and the explicit `PairTopology` are
exactly the factor's. This is the energy used by the parameter-gradient finite
difference checks.
"""
function streamed_energy(f::StreamedLJFactor, q::AbstractMatrix{<:Real},
                         ctx::Factors.Context;
                         epsilon::AbstractVector{<:Real} = f.epsilon,
                         sigma::AbstractVector{<:Real} = f.sigma)
    box = _validate_factor_context(f, q, ctx)
    _validate_parameters(epsilon, sigma, f.N)
    Base.mightalias(q, f.ws.forces) &&
        throw(ArgumentError("q aliases the factor's internal force workspace; " *
                            "pass a separate coordinate matrix"))
    U, _, _ = evaluate!(f.ws, q, epsilon, sigma, box;
                        rcut = f.rcut, skin = f.skin, mode = f.mode,
                        topology = f.topology, cache = f.pair_cache)
    return U
end

"""
    streamed_parameter_gradients!(f, q, ctx; epsilon = f.epsilon,
                                  sigma = f.sigma) -> (U, dU_depsilon, dU_dsigma)

    streamed_parameter_gradients!(dEpsilon, dSigma, f, q, ctx; kwargs...) -> U

Dedicated parameter-gradient evaluator for a [`StreamedLJFactor`](@ref). It
returns the exact analytic per-atom derivatives of the *same* effective
potential and topology as [`streamed_energy`](@ref) / `evaluate!`: the two
gradient vectors are copies, never aliases of the workspace accumulators.

`ParameterGradients.evaluate_parameter_gradients!` intentionally does not
support this factor (it only knows the five local pair/bond factor types); this
method is the explicit, separately validated API instead of silently omitting a
factor from a joint gradient.
"""
function streamed_parameter_gradients!(f::StreamedLJFactor,
                                       q::AbstractMatrix{<:Real},
                                       ctx::Factors.Context;
                                       epsilon::AbstractVector{<:Real} = f.epsilon,
                                       sigma::AbstractVector{<:Real} = f.sigma)
    box = _validate_factor_context(f, q, ctx)
    f.mode === :trunc && throw(ArgumentError(
        "streamed_parameter_gradients! does not implement the unshifted :trunc " *
        "reference mode; use :lfs or :shift"))
    _validate_parameters(epsilon, sigma, f.N)
    Base.mightalias(q, f.ws.forces) &&
        throw(ArgumentError("q aliases the factor's internal force workspace; " *
                            "pass a separate coordinate matrix"))
    energy, _, _, ge, gs = evaluate_with_gradients!(
        f.ws, q, epsilon, sigma, box;
        rcut = f.rcut, skin = f.skin, mode = f.mode, topology = f.topology)
    return energy, copy(ge), copy(gs)
end

function streamed_parameter_gradients!(dEpsilon::AbstractVector{<:Real},
                                       dSigma::AbstractVector{<:Real},
                                       f::StreamedLJFactor,
                                       q::AbstractMatrix{<:Real},
                                       ctx::Factors.Context; kwargs...)
    length(dEpsilon) == f.N ||
        throw(ArgumentError("dEpsilon has length $(length(dEpsilon)); expected N=$(f.N)"))
    length(dSigma) == f.N ||
        throw(ArgumentError("dSigma has length $(length(dSigma)); expected N=$(f.N)"))
    energy, ge, gs = streamed_parameter_gradients!(f, q, ctx; kwargs...)
    dEpsilon .= ge
    dSigma .= gs
    return energy
end

"""
    evaluate!(F, f::StreamedLJFactor, q, ctx) -> U

Convenience forwarding method for callers that hold the `StreamedLJ` module
rather than `Factors`. It is exactly `Factors.evaluate!`, so the factor's
additive-force and validation semantics are unchanged.
"""
evaluate!(F::AbstractMatrix{Float64}, f::StreamedLJFactor,
          q::AbstractMatrix{Float64}, ctx::Factors.Context) =
    Factors.evaluate!(F, f, q, ctx)

"""
    evaluate_with_gradients!(f::StreamedLJFactor, q, ctx; epsilon, sigma)
        -> (U, dU_depsilon, dU_dsigma)

Convenience forwarding method to [`streamed_parameter_gradients!`](@ref) for
callers that hold the `StreamedLJ` module. The returned vectors are copies.
"""
evaluate_with_gradients!(f::StreamedLJFactor, q::AbstractMatrix{<:Real},
                         ctx::Factors.Context; kwargs...) =
    streamed_parameter_gradients!(f, q, ctx; kwargs...)

"""
    streamed_box_derivative(f::StreamedLJFactor, q::AbstractMatrix{<:Real}, ctx::Factors.Context) -> Matrix{Float64}
    streamed_box_derivative(f::StreamedLJFactor, q::AbstractMatrix{<:Real}) -> Matrix{Float64}

Analytic box derivative ∂U/∂h|_s for a [`StreamedLJFactor`](@ref) at fixed fractional
coordinates. Returns a new 3x3 matrix. Requires force-consistent mode `:lfs`.
"""
function streamed_box_derivative(f::StreamedLJFactor, q::AbstractMatrix{<:Real},
                                ctx::Factors.Context)
    box = _validate_factor_context(f, q, ctx)
    f.mode === :lfs || throw(ArgumentError(
        "streamed_box_derivative requires force-consistent mode=:lfs; got $(repr(f.mode))"))
    G = zeros(Float64, 3, 3)
    streamed_box_derivative!(G, f, q, ctx)
    return G
end

function streamed_box_derivative(f::StreamedLJFactor, q::AbstractMatrix{<:Real})
    ctx = Factors.Context(f.box, Factors.REDUCED, f.N)
    return streamed_box_derivative(f, q, ctx)
end

"""
    streamed_box_derivative!(G::AbstractMatrix{Float64}, f::StreamedLJFactor,
                             q::AbstractMatrix{<:Real}, ctx::Factors.Context) -> Matrix{Float64}
    streamed_box_derivative!(G::AbstractMatrix{Float64}, f::StreamedLJFactor,
                             q::AbstractMatrix{<:Real}) -> Matrix{Float64}

In-place analytic box derivative ∂U/∂h|_s for a [`StreamedLJFactor`](@ref) at fixed
fractional coordinates, accumulated into supplied 3x3 matrix `G`.
"""
function streamed_box_derivative!(G::AbstractMatrix{Float64}, f::StreamedLJFactor,
                                  q::AbstractMatrix{<:Real}, ctx::Factors.Context)
    box = _validate_factor_context(f, q, ctx)
    f.mode === :lfs || throw(ArgumentError(
        "streamed_box_derivative! requires force-consistent mode=:lfs; got $(repr(f.mode))"))
    box_derivative!(G, f.ws, q, f.epsilon, f.sigma, box;
                    rcut = f.rcut, skin = f.skin, mode = f.mode,
                    topology = f.topology, threaded = f.threaded,
                    cache = f.pair_cache)
    return G
end

function streamed_box_derivative!(G::AbstractMatrix{Float64}, f::StreamedLJFactor,
                                  q::AbstractMatrix{<:Real})
    ctx = Factors.Context(f.box, Factors.REDUCED, f.N)
    return streamed_box_derivative!(G, f, q, ctx)
end

"""
    evaluate_with_box_derivative!(f::StreamedLJFactor, q::AbstractMatrix{<:Real},
                                  ctx::Factors.Context)
        -> (energy, forces, pair_count, box_derivative)

Convenience evaluator returning the LJ energy, forces, interacting pair count,
and exact analytic box derivative ∂U/∂h|_s for a [`StreamedLJFactor`](@ref).
"""
function evaluate_with_box_derivative!(f::StreamedLJFactor, q::AbstractMatrix{<:Real},
                                       ctx::Factors.Context)
    box = _validate_factor_context(f, q, ctx)
    f.mode === :lfs || throw(ArgumentError(
        "evaluate_with_box_derivative! requires mode=:lfs; got $(repr(f.mode))"))
    return evaluate_with_box_derivative!(f.ws, q, f.epsilon, f.sigma, box;
                                         rcut = f.rcut, skin = f.skin, mode = f.mode,
                                         topology = f.topology, threaded = f.threaded,
                                         cache = f.pair_cache)
end

function evaluate_with_box_derivative!(f::StreamedLJFactor, q::AbstractMatrix{<:Real})
    ctx = Factors.Context(f.box, Factors.REDUCED, f.N)
    return evaluate_with_box_derivative!(f, q, ctx)
end

# ---------------------------------------------------------------------------
# memory calculator
# ---------------------------------------------------------------------------

"""
    estimate_memory_bytes(N, box, rcut; skin=0.0, overhead=0.30, gradients=false)
        -> Int

Conservative estimate of the bytes an evaluation needs: coordinates,
forces, per-atom epsilon/sigma, the `next` array, the cell `head` array, and the
27-entry scratch array. Reusable per-cell energy/count reduction arrays are
included. With `gradients=true`, the two per-atom gradient accumulators are
included.
A 30% overhead is included by default. This allocates nothing large and is
intended for a preflight check before a 10-million-atom run.
"""
function estimate_memory_bytes(N::Integer, box::AbstractBox, rcut::Real;
                               skin::Real = 0.0, overhead::Real = 0.30,
                               gradients::Bool = false,
                               cache::Symbol = :none,
                               threads::Integer = 1)
    N >= 0 || throw(ArgumentError("N must be >= 0, got $N"))
    threads >= 1 || throw(ArgumentError("threads must be >= 1, got $threads"))
    (cache === :none || cache === :pair || cache === :tile) ||
        throw(ArgumentError("cache must be :none, :pair, or :tile; got $(repr(cache))"))
    rc, sk = _validate_cutoff(box, rcut, skin)
    _validate_capacity(Int(N))
    widths = box isa OrthorhombicBox ? box.L : box.widths
    nc = _grid_nc(widths, rc + sk)
    ncell = _checked_ncells(nc)
    n = Int(N)
    # 3N coords + 3N forces + N epsilon + N sigma + N next + ncell head + scratch
    base = (6 * 8 + 2 * 8 + 4) * n + 16 * ncell + 27 * 8
    gradients && (base += 2 * 8 * n)
    if threads > 1
        base += Int(threads) * 3 * 8 * n
    end
    if cache === :pair
        vol = box isa OrthorhombicBox ? (widths[1] * widths[2] * widths[3]) : Types.box_volume(box)
        rho = vol > 0.0 ? n / vol : 0.0
        est_pairs = ceil(Int, 0.5 * n * (4.0 / 3.0) * pi * (rc + sk)^3 * rho)
        base += 24 * n + 8 * max(n, est_pairs)
    elseif cache === :tile
        ntiles = cld(n, Neighbors.TILE_SIZE)
        vol = box isa OrthorhombicBox ? (widths[1] * widths[2] * widths[3]) : Types.box_volume(box)
        rho = vol > 0.0 ? n / vol : 0.0
        atoms_per_cutoff = max(1.0, (4.0 / 3.0) * pi * (rc + sk)^3 * rho)
        tiles_per_tile = ceil(Int, atoms_per_cutoff / Neighbors.TILE_SIZE) + 14
        est_tile_pairs = ntiles * tiles_per_tile
        # order (4N) + inv_order (4N) + q_sorted (24N) + q_ref (24N) + tile bboxes (48*ntiles) + tile pairs (8*est_tile_pairs)
        base += 56 * n + 48 * ntiles + 8 * est_tile_pairs
    end
    return ceil(Int, base * (1.0 + Float64(overhead)))
end

end # module StreamedLJ
