# Core type system: units, typed state, periodic box, chemical topology.
#
# Design rule (from FRAMEWORK.md §2): a molecular state is a typed
# record, not five loose arrays. Every field that changes behaviour is present
# and every unit is explicit.
#
# Periodic geometry includes orthorhombic and triclinic boxes; neighbour
# stencils may impose stricter geometry requirements.

module Types

using LinearAlgebra

export Units, REDUCED, SIReduced, AtomSet, OrthorhombicBox, TriclinicBox, NoBox, AbstractBox,
       triclinic_guard, ChemicalTopology, Bond, Angle, Dihedral, Improper,
       system_of, box_volume, box_lengths, wrap!, unwrap_displacement,
       UnsupportedGeometry, InvalidState,
       UnsupportedChemistry, InvalidGeometry, NumericalFailure

# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------

"""
A unit system fixes the meaning of every number in the state. Energies are in
`energy` units, lengths in `length`, masses in `mass`, charges in `charge`.
`coulomb_prefactor` converts `z_i z_j / r` into an energy in these units.

`REDUCED` is Lennard-Jones reduced units (sigma = epsilon = mass = 1), which is
the convention the NIST reference data in `validation/` uses. `SIReduced` maps
to SI so that a physical force field can be attached later without changing the
integrator or any energy code.
"""
struct Units
    name::String
    length::Float64          # metres per length unit
    energy::Float64          # joules per energy unit
    mass::Float64            # kilograms per mass unit
    charge::Float64          # coulombs per charge unit
    coulomb_prefactor::Float64  # energy*length/charge^2, in these units
    boltzmann::Float64       # energy per temperature unit
end

const REDUCED = Units("reduced (sigma=epsilon=m=1; k_B=1)",
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

# CODATA 2018: e = 1.602176634e-19 C, 1/(4 pi eps0) = 8.9875517923e9 J m / C^2
const _COULOMB_SI = 8.9875517923e9
const _E_CHARGE = 1.602176634e-19

struct InvalidState <: Exception
    msg::String
end
Base.showerror(io::IO, e::InvalidState) = print(io, "InvalidState: ", e.msg)

struct UnsupportedGeometry <: Exception
    msg::String
end
Base.showerror(io::IO, e::UnsupportedGeometry) = print(io, "UnsupportedGeometry: ", e.msg)

"""
Exception raised when an unsupported chemical element, factor type,
topological feature, or force-field functional form is encountered.
"""
struct UnsupportedChemistry <: Exception
    msg::String
end
Base.showerror(io::IO, e::UnsupportedChemistry) = print(io, "UnsupportedChemistry: ", e.msg)

"""
Exception raised when a geometric configuration, box matrix, or coordinate
specification is unphysical, degenerate, or violates geometric invariants.
"""
struct InvalidGeometry <: Exception
    msg::String
end
Base.showerror(io::IO, e::InvalidGeometry) = print(io, "InvalidGeometry: ", e.msg)

"""
Exception raised when a numerical computation fails (e.g. non-finite energy/forces,
diverged pressure, non-convergent iterative solver, unstable matrix inversion).
"""
struct NumericalFailure <: Exception
    msg::String
end
Base.showerror(io::IO, e::NumericalFailure) = print(io, "NumericalFailure: ", e.msg)

# ---------------------------------------------------------------------------
# boxes
# ---------------------------------------------------------------------------

abstract type AbstractBox end

"Non-periodic system."
struct NoBox <: AbstractBox end

"""
Orthorhombic periodic box with side lengths `L` (all angles 90 degrees).
"""
struct OrthorhombicBox <: AbstractBox
    L::NTuple{3,Float64}
    function OrthorhombicBox(L::NTuple{3,Float64})
        all(l -> l > 0 && isfinite(l), L) ||
            throw(InvalidGeometry("box lengths must be finite and positive, got $L"))
        new(L)
    end
end

"""Triclinic periodic box; columns of `H` are the three lattice vectors."""
struct TriclinicBox <: AbstractBox
    H::Matrix{Float64}
    Hinv::Matrix{Float64}
    widths::NTuple{3,Float64}
    function TriclinicBox(H::AbstractMatrix{<:Real})
        size(H) == (3, 3) || throw(InvalidGeometry("cell matrix must be 3x3, got $(size(H))"))
        M = Matrix{Float64}(H)
        all(isfinite, M) || throw(InvalidGeometry("cell matrix must be finite"))
        detM = det(M)
        isfinite(detM) && detM > 0 || throw(InvalidGeometry("triclinic cell must have finite positive volume"))
        invM = inv(M)
        widths = ntuple(i -> inv(norm(view(invM, i, :))), 3)
        all(w -> isfinite(w) && w > 0, widths) || throw(InvalidGeometry("triclinic cell has invalid perpendicular widths"))
        new(M, invM, widths)
    end
end
OrthorhombicBox(L::AbstractVector{<:Real}) =
    OrthorhombicBox((Float64(L[1]), Float64(L[2]), Float64(L[3])))

box_lengths(::NoBox) = nothing
box_lengths(b::OrthorhombicBox) = b.L
box_lengths(b::TriclinicBox) = ntuple(i -> norm(view(b.H, :, i)), 3)
box_volume(::NoBox) = throw(UnsupportedGeometry("no box: volume undefined"))
box_volume(b::OrthorhombicBox) = b.L[1] * b.L[2] * b.L[3]
box_volume(b::TriclinicBox) = det(b.H)

"""Guard APIs that only accept orthorhombic matrices; use `TriclinicBox` for triclinic MIC."""
function triclinic_guard(H::AbstractMatrix{<:Real}; tol::Float64 = 1e-12)
    size(H) == (3, 3) || throw(InvalidState("cell matrix must be 3x3, got $(size(H))"))
    off = maximum(abs, H .- Diagonal(diag(H)))
    if off <= tol
        return OrthorhombicBox((Float64(H[1, 1]), Float64(H[2, 2]), Float64(H[3, 3])))
    end
    throw(UnsupportedGeometry(
        "triclinic cells are unsupported by this guarded API; TriclinicBox provides " *
        "an exact nearest-image displacement, but downstream periodic consumers " *
        "may not implement a lattice-correct stencil"))
end

# ---------------------------------------------------------------------------
# atom set
# ---------------------------------------------------------------------------

"""
Typed atom state. Positions and momenta are `3 x N` so that per-atom columns are
contiguous for the force scatter. Masses and charges are `N`-vectors.

Invariants enforced by the constructor: finite positions/momenta, strictly
positive finite masses, finite charges, consistent shapes. A system at a
singular configuration (coincident atoms) is valid *state* but not a valid
*sample*; that is checked by `validation/prepare.jl`, not here, because the
correct repair is a policy decision.
"""
struct AtomSet
    q::Matrix{Float64}   # 3 x N positions
    p::Matrix{Float64}   # 3 x N momenta
    m::Vector{Float64}   # N masses
    z::Vector{Float64}   # N charges
    names::Vector{Symbol}
    function AtomSet(q::AbstractMatrix{<:Real}, p::AbstractMatrix{<:Real},
                     m::AbstractVector{<:Real}, z::AbstractVector{<:Real},
                     names::AbstractVector{Symbol})
        q = Matrix{Float64}(q); p = Matrix{Float64}(p)
        m = Vector{Float64}(m); z = Vector{Float64}(z)
        N = size(q, 2)
        size(q, 1) == 3 || throw(InvalidState("positions must be 3 x N, got $(size(q))"))
        size(p) == size(q) || throw(InvalidState("momenta must match positions, got $(size(p)) vs $(size(q))"))
        length(m) == N || throw(InvalidState("mass vector length $(length(m)) != N=$N"))
        length(z) == N || throw(InvalidState("charge vector length $(length(z)) != N=$N"))
        length(names) == N || throw(InvalidState("names length $(length(names)) != N=$N"))
        all(isfinite, q) || throw(InvalidState("non-finite coordinate in positions"))
        all(isfinite, p) || throw(InvalidState("non-finite coordinate in momenta"))
        all(x -> isfinite(x) && x > 0, m) ||
            throw(InvalidState("masses must be finite and strictly positive"))
        all(isfinite, z) || throw(InvalidState("non-finite charge"))
        return new(q, p, m, z, names)
    end
end

AtomSet(q, p, m, z) = AtomSet(q, p, m, z, fill(:X, size(q, 2)))

Base.show(io::IO, a::AtomSet) =
    print(io, "AtomSet(N=$(length(a.m)), Σm=$(round(sum(a.m), digits=4)), " *
              "Σq=$(round(sum(a.z), digits=6)))")

# ---------------------------------------------------------------------------
# chemical topology: a hypergraph, NOT the spatial grouping
# ---------------------------------------------------------------------------

abstract type AbstractChemicalFactor end
struct Bond <: AbstractChemicalFactor
    i::Int; j::Int; k::Float64; r0::Float64
end
struct Angle <: AbstractChemicalFactor
    i::Int; j::Int; k::Int; ktheta::Float64; theta0::Float64
end
struct Dihedral <: AbstractChemicalFactor
    i::Int; j::Int; k::Int; l::Int; kphi::Float64; n::Int; phi0::Float64
end
struct Improper <: AbstractChemicalFactor
    i::Int; j::Int; k::Int; l::Int; kphi::Float64; phi0::Float64
end

"""
Chemical topology. `exclusions` is a set of unordered index pairs that must not
receive nonbonded interactions; `scales` maps excluded pairs to (lj, coulomb)
multipliers for 1-4 scaling.

This is deliberately separate from the *spatial* grouping used for interaction
summaries. Conflating them is the classic error: a chemical bond is not a
spatial cluster, and two atoms bonded to each other can be far apart in
space only if the geometry is unphysical.
"""
struct ChemicalTopology
    bonds::Vector{Bond}
    angles::Vector{Angle}
    dihedrals::Vector{Dihedral}
    impropers::Vector{Improper}
    exclusions::Set{Tuple{Int,Int}}
    scales::Dict{Tuple{Int,Int},NTuple{2,Float64}}
end

ChemicalTopology(; bonds = Bond[], angles = Angle[], dihedrals = Dihedral[],
                 impropers = Improper[], exclusions = Set{Tuple{Int,Int}}(),
                 scales = Dict{Tuple{Int,Int},NTuple{2,Float64}}()) =
    ChemicalTopology(bonds, angles, dihedrals, impropers, exclusions, scales)

function ChemicalTopology(factors::AbstractVector{<:AbstractChemicalFactor})
    ChemicalTopology(; bonds = Bond[f for f in factors if f isa Bond],
                     angles = Angle[f for f in factors if f isa Angle],
                     dihedrals = Dihedral[f for f in factors if f isa Dihedral],
                     impropers = Improper[f for f in factors if f isa Improper])
end

key(i::Int, j::Int) = i < j ? (i, j) : (j, i)

"True when the pair must not receive a nonbonded interaction."
haskey_excluded(t::ChemicalTopology, i::Int, j::Int) = key(i, j) in t.exclusions

"Nonbonded (lj, coulomb) scale for a pair; 1.0 unless excluded or 1-4 scaled."
function pair_scale(t::ChemicalTopology, i::Int, j::Int)
    k = key(i, j)
    if haskey(t.scales, k)
        return t.scales[k]
    elseif k in t.exclusions
        return (0.0, 0.0)
    else
        return (1.0, 1.0)
    end
end

# ---------------------------------------------------------------------------
# system
# ---------------------------------------------------------------------------

"""
A complete problem statement: typed atom state, box, topology, unit system.

`temperature` is the target thermodynamic temperature in the unit system's
temperature unit; it is metadata for samplers, never a force.
"""
struct System
    atoms::AtomSet
    box::AbstractBox
    topology::ChemicalTopology
    units::Units
    temperature::Float64
end

function System(atoms::AtomSet, box::AbstractBox = NoBox(),
                topology::ChemicalTopology = ChemicalTopology();
                units::Units = REDUCED, temperature::Float64 = 0.0)
    temperature >= 0 || throw(InvalidState("temperature must be non-negative"))
    return System(atoms, box, topology, units, temperature)
end

system_of(atoms::AtomSet; kwargs...) = System(atoms; kwargs...)

Base.show(io::IO, s::System) =
    print(io, "System(", s.atoms, ", box=",
              s.box isa NoBox ? "none" : s.box isa OrthorhombicBox ? string(s.box.L) : string(s.box.H),
              ", ", length(s.topology.bonds), " bonds)")

# ---------------------------------------------------------------------------
# periodic helpers
# ---------------------------------------------------------------------------

"Wrap all positions into the primary cell."
function wrap!(s::System)
    s.box isa NoBox && return s
    q = s.atoms.q
    if s.box isa OrthorhombicBox
        L = s.box.L
        @inbounds for j in axes(q, 2), d in 1:3
            q[d, j] -= L[d] * floor(q[d, j] / L[d])
        end
    else
        box = s.box::TriclinicBox
        @inbounds for j in axes(q, 2)
            f = box.Hinv * view(q, :, j)
            q[:, j] .= box.H * (f .- floor.(f))
        end
    end
    return s
end

@inline function unwrap_displacement(box::TriclinicBox, q::AbstractMatrix{Float64}, i::Int, j::Int)
    dx = q[1, i] - q[1, j]; dy = q[2, i] - q[2, j]; dz = q[3, i] - q[3, j]
    f1 = box.Hinv[1, 1] * dx + box.Hinv[1, 2] * dy + box.Hinv[1, 3] * dz
    f2 = box.Hinv[2, 1] * dx + box.Hinv[2, 2] * dy + box.Hinv[2, 3] * dz
    f3 = box.Hinv[3, 1] * dx + box.Hinv[3, 2] * dy + box.Hinv[3, 3] * dz
    n01 = round(Int, f1); n02 = round(Int, f2); n03 = round(Int, f3)
    best_x = dx - (box.H[1, 1] * n01 + box.H[1, 2] * n02 + box.H[1, 3] * n03)
    best_y = dy - (box.H[2, 1] * n01 + box.H[2, 2] * n02 + box.H[2, 3] * n03)
    best_z = dz - (box.H[3, 1] * n01 + box.H[3, 2] * n02 + box.H[3, 3] * n03)
    best2 = best_x * best_x + best_y * best_y + best_z * best_z
    b0 = sqrt(best2)
    b1 = sqrt(box.Hinv[1, 1]^2 + box.Hinv[1, 2]^2 + box.Hinv[1, 3]^2) * b0
    b2 = sqrt(box.Hinv[2, 1]^2 + box.Hinv[2, 2]^2 + box.Hinv[2, 3]^2) * b0
    b3 = sqrt(box.Hinv[3, 1]^2 + box.Hinv[3, 2]^2 + box.Hinv[3, 3]^2) * b0
    for c in ceil(Int, f3 - b3):floor(Int, f3 + b3)
        for b in ceil(Int, f2 - b2):floor(Int, f2 + b2)
            for a in ceil(Int, f1 - b1):floor(Int, f1 + b1)
                rx = dx - (box.H[1, 1] * a + box.H[1, 2] * b + box.H[1, 3] * c)
                ry = dy - (box.H[2, 1] * a + box.H[2, 2] * b + box.H[2, 3] * c)
                rz = dz - (box.H[3, 1] * a + box.H[3, 2] * b + box.H[3, 3] * c)
                r2 = rx * rx + ry * ry + rz * rz
                if r2 < best2
                    best_x, best_y, best_z, best2 = rx, ry, rz, r2
                end
            end
        end
    end
    return SVector3(best_x, best_y, best_z)
end

"""
Minimum-image displacement `q_i - q_j` for an orthorhombic box.

Computed as `d - L*round(d/L)`, which is the exact nearest image for an
orthorhombic cell when `|d_d| <= L_d`; it is extended to work for any `d` by the
same formula because `round` selects the nearest lattice translation.
"""
@inline function unwrap_displacement(box::NoBox, q::AbstractMatrix{Float64}, i::Int, j::Int)
    return SVector3(q[1, i] - q[1, j], q[2, i] - q[2, j], q[3, i] - q[3, j])
end

struct SVector3
    x::Float64; y::Float64; z::Float64
end
Base.:(-)(a::SVector3, b::SVector3) = SVector3(a.x - b.x, a.y - b.y, a.z - b.z)
norm2(a::SVector3) = a.x * a.x + a.y * a.y + a.z * a.z

@inline function unwrap_displacement(box::OrthorhombicBox, q::AbstractMatrix{Float64}, i::Int, j::Int)
    L = box.L
    dx = q[1, i] - q[1, j]
    dy = q[2, i] - q[2, j]
    dz = q[3, i] - q[3, j]
    dx -= L[1] * round(dx / L[1])
    dy -= L[2] * round(dy / L[2])
    dz -= L[3] * round(dz / L[3])
    return SVector3(dx, dy, dz)
end

end # module Types
