"""
Shared, deterministic fixtures for the verification lanes.

Everything here is a *fixture*: a six-atom bonded chain, a rigid-water
constraint network, and a two-site volume-move energy. The lanes consume the
engine's public APIs (`Factors`, `Constraints`, `MonteCarloBarostat`) and never
reach into engine internals by hand.

The generic potential below (`chain_energy`) is an independent reimplementation
of the same three bonded laws. It is written generically so that TaylorSeries
and IntervalArithmetic numbers can flow through it. It is pinned to the engine
by `Float64` energy parity in `lane_derivatives.jl`; only after that parity
holds is its Taylor expansion a valid derivative reference.
"""

using FoundationCore
using LinearAlgebra: dot, norm, cross

const FactorLib = FoundationCore.Factors
const TypesLib = FoundationCore.Types
const ConstraintLib = FoundationCore.Constraints

# `MonteCarloBarostat` is deliberately *not* included by `FoundationCore`
# ("standalone adapter and is not wired into FoundationCore"). Including it in a
# host module that exposes `Types` satisfies its `using ..Types` and gives the
# lanes the exact engine API instead of a copy.
module BarostatHost
using FoundationCore
using FoundationCore: Types
include(joinpath(pkgdir(FoundationCore), "src", "MonteCarloBarostat.jl"))
end
const BarostatLib = BarostatHost.MonteCarloBarostat

# ---------------------------------------------------------------------------
# bonded chain fixture
# ---------------------------------------------------------------------------

struct ChainParameters
    bond_k::Float64
    bond_r0::Float64
    angle_k::Float64
    angle_theta0::Float64
    dihedral_k::Float64
    dihedral_n::Int
    dihedral_phi0::Float64
end

const CHAIN = ChainParameters(25.0, 1.0, 8.0, 1.9, 1.5, 3, 0.3)

"Six non-collinear atoms; every dihedral of the chain is well defined."
function chain_positions()
    return [0.00  0.96  1.85  2.50  3.42  4.30;
            0.00  0.28  0.90  1.70  1.50  2.20;
            0.00  0.10  0.05  0.60  0.35  1.00]
end

function chain_factors(p::ChainParameters = CHAIN; na::Int = 6)
    factors = FactorLib.AbstractFactor[]
    for i in 1:(na - 1)
        push!(factors, FactorLib.HarmonicBond(i, i + 1, p.bond_k, p.bond_r0))
    end
    for i in 1:(na - 2)
        push!(factors, FactorLib.HarmonicAngle(i, i + 1, i + 2, p.angle_k, p.angle_theta0))
    end
    for i in 1:(na - 3)
        push!(factors, FactorLib.PeriodicDihedral(
            i, i + 1, i + 2, i + 3, p.dihedral_k, p.dihedral_n, p.dihedral_phi0))
    end
    return factors
end

chain_context(na::Int = 6) = FactorLib.Context(TypesLib.REDUCED, na)

"Engine energy and forces `(U, F)` with `F = -dU/dq`."
function chain_energy_forces(q::AbstractMatrix{Float64}, p::ChainParameters = CHAIN)
    factors = chain_factors(p; na = size(q, 2))
    return FactorLib.total_energy_forces(factors, q, chain_context(size(q, 2)))
end

# ---------------------------------------------------------------------------
# generic reimplementation of the same bonded laws
# ---------------------------------------------------------------------------

@inline _vec(q::AbstractMatrix, i::Int) = view(q, :, i)

function _cross3(a, b)
    return [a[2] * b[3] - a[3] * b[2],
            a[3] * b[1] - a[1] * b[3],
            a[1] * b[2] - a[2] * b[1]]
end

function _dot3(a, b)
    return a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
end

function bond_energy(q::AbstractMatrix, i::Int, j::Int, p::ChainParameters)
    d = _vec(q, j) .- _vec(q, i)
    r2 = _dot3(d, d)
    r = sqrt(r2)
    return 0.5 * p.bond_k * (r - p.bond_r0)^2
end

function angle_energy(q::AbstractMatrix, i::Int, j::Int, k::Int, p::ChainParameters)
    u1 = _vec(q, i) .- _vec(q, j)
    u2 = _vec(q, k) .- _vec(q, j)
    r1 = sqrt(_dot3(u1, u1))
    r2 = sqrt(_dot3(u2, u2))
    ct = _dot3(u1, u2) / (r1 * r2)   # callers keep |ct| strictly below 1
    th = acos(ct)
    return 0.5 * p.angle_k * (th - p.angle_theta0)^2
end

function dihedral_energy(q::AbstractMatrix, i::Int, j::Int, k::Int, l::Int,
                         p::ChainParameters)
    b1 = _vec(q, j) .- _vec(q, i)
    b2 = _vec(q, k) .- _vec(q, j)
    b3 = _vec(q, l) .- _vec(q, k)
    n1 = _cross3(b1, b2)
    n2 = _cross3(b2, b3)
    lb2 = sqrt(_dot3(b2, b2))
    phi = atan(_dot3(_cross3(n1, n2), b2) / lb2, _dot3(n1, n2))
    return p.dihedral_k * (1.0 + cos(p.dihedral_n * phi - p.dihedral_phi0))
end

"""
    chain_energy(q, p = CHAIN) -> scalar

Sum of the same bond + angle + dihedral laws as `chain_factors`, evaluated in
whatever arithmetic `eltype(q)` provides (`Float64`, `Taylor1`, `Interval`).
"""
function chain_energy(q::AbstractMatrix, p::ChainParameters = CHAIN)
    na = size(q, 2)
    u = zero(eltype(q))
    for i in 1:(na - 1)
        u += bond_energy(q, i, i + 1, p)
    end
    for i in 1:(na - 2)
        u += angle_energy(q, i, i + 1, i + 2, p)
    end
    for i in 1:(na - 3)
        u += dihedral_energy(q, i, i + 1, i + 2, i + 3, p)
    end
    return u
end

# ---------------------------------------------------------------------------
# rigid water fixture
# ---------------------------------------------------------------------------

"Rigid TIP3P-like geometry: O-H 0.9572, H-O-H 104.52 degrees."
function water_geometry()
    oh = 0.9572
    theta = 104.52 * pi / 180
    h1 = [oh, 0.0, 0.0]
    h2 = [oh * cos(theta), oh * sin(theta), 0.0]
    q = [0.0 h1[1] h2[1];
         0.0 h1[2] h2[2];
         0.0 h1[3] h2[3]]
    return q, oh
end

"Constraint set for the rigid-water fixture with exact internal distances."
function water_constraints()
    q, oh = water_geometry()
    hh = norm(q[:, 2] - q[:, 3])
    return ConstraintLib.CoupledConstraintSet(3, [(1, 2, oh), (1, 3, oh), (2, 3, hh)]), q
end

# ---------------------------------------------------------------------------
# barostat fixture
# ---------------------------------------------------------------------------

"Configurational energy callback for the volume-move fixture."
harmonic_trap_energy(q::AbstractMatrix{<:Real}, box) = 0.5 * sum(abs2, q)
