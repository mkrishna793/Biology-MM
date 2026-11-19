"""
Model systems for the molecular-HMC / parallel-tempering experiment.

Everything here is built from `Factors.AbstractFactor` objects and wrapped in a
`Dynamics.HarmonicSystem`, so the samplers under test are the *same* code path
that velocity Verlet and BAOAB use. Nothing in this file changes the library.

`SeparationDoubleWell` is the one coarse-grained ingredient: it is an external
potential on the distance between the centres of mass of two clusters. It is
labelled as a model potential, not a physical force field.
"""
module SamplingFactorModels

using FoundationCore
using FoundationCore: Types, Factors, Dynamics, Sampling
using Random, Statistics, LinearAlgebra
import FoundationCore.Factors: evaluate!, read_support, factor_support_size

export lj_fluid_system, lj_fluid_energy, baoab_run, two_cluster_system,
       SeparationDoubleWell, separation_series, separation_crossings,
       walker_round_trips, cluster_separation, lattice_positions

# ---------------------------------------------------------------------------
# homogeneous Lennard-Jones fluid, orthorhombic PBC
# ---------------------------------------------------------------------------

"""
    lattice_positions(N, L) -> (q, a)

Simple cubic lattice inside an `L`-cubed box, one atom per site, with the
sublattice spacing `a` chosen so `N` sites fill the box. Used only as a starting
configuration; the equilibrium run melts it.
"""
function lattice_positions(N::Int, L::Float64)
    nside = ceil(Int, cbrt(N))
    a = L / nside
    q = zeros(3, N)
    k = 1
    for i in 1:nside, j in 1:nside, l in 1:nside
        k > N && break
        q[:, k] = [(i - 0.5) * a, (j - 0.5) * a, (l - 0.5) * a]
        k += 1
    end
    return q, a
end

"""
    lj_fluid_system(N, rho; rc = 2.5, seed = 3) -> (system, q0, box_length)

All-pairs Lennard-Jones fluid in reduced units (`ε = σ = m = 1`), with the
force-consistent `:lfs` cutoff required by the integrator gate. All-pairs (not
neighbour-list) construction is deliberate: it is the brute-force reference, so
a factor-list mistake cannot hide behind a cell list. `O(N²)` is fine at N = 32.
"""
function lj_fluid_system(N::Int, rho::Float64; rc::Float64 = 2.5, seed::Int = 3,
                         jitter::Float64 = 0.02)
    L = (N / rho)^(1 / 3)
    box = Types.OrthorhombicBox((L, L, L))
    ctx = Factors.Context(box, Types.REDUCED, N)
    q, a = lattice_positions(N, L)
    rng = MersenneTwister(seed)
    q .+= jitter * a .* randn(rng, 3, N)
    factors = Factors.AbstractFactor[]
    for i in 1:N, j in (i + 1):N
        push!(factors, Factors.LennardJonesPair(i, j, 1.0, 1.0; mode = :lfs, rc = rc))
    end
    return Dynamics.HarmonicSystem(factors, ones(N), ctx), q, L
end

lj_fluid_energy(s, q) = Dynamics.potential_energy(s, q)

# ---------------------------------------------------------------------------
# BAOAB reference (the trusted sampler for this comparison)
# ---------------------------------------------------------------------------

"""
    baoab_run(s, q0, T, dt, nsteps; gamma = 1.0, seed = 5) -> (U_series, q_end)

One BAOAB Langevin chain. `U_series` is the potential energy *after* each step
(the same convention as `hmc_trajectory`). Uses `Dynamics.langevin_baoab_step!`,
i.e. the exact Ornstein–Uhlenbeck momentum update, so its invariant measure is
the canonical one up to the usual `O(dt²)` discretisation error.
"""
function baoab_run(s::Dynamics.HarmonicSystem, q0, T::Float64, dt::Float64,
                   nsteps::Int; gamma::Float64 = 1.0, seed::Int = 5)
    n = Dynamics.N(s)
    q = Matrix{Float64}(q0)
    p = zeros(3, n)
    F = zeros(3, n)
    Dynamics.forces!(s, q, F)
    rng = MersenneTwister(seed)
    beta = Sampling.molecular_beta(s, T)
    Us = zeros(nsteps)
    for i in 1:nsteps
        Us[i] = Dynamics.langevin_baoab_step!(q, p, F, s, dt, gamma, beta, rng)
    end
    return Us, q
end

# ---------------------------------------------------------------------------
# two-cluster barrier model
# ---------------------------------------------------------------------------

"""
    SeparationDoubleWell(group_a, group_b, a, rmid, d)

Coarse-grained external potential on the distance `r` between the centres of
mass of two equal-mass atom groups:

    U(r) = a (r - rmid)² - d²)²

with minima `U = 0` at `r = rmid ± d` and a barrier of height `a` at `r = rmid`.
Forces are `-dU/dr` along the COM separation, split equally over the atoms of
each group (the COM map has derivative `1/K`).

This is a **model** potential chosen to make a symmetric two-state barrier with
a controlled height in reduced units. It is not a physical inter-cluster force
field; the clusters themselves are real bonded/LJ clusters.
"""
struct SeparationDoubleWell <: Factors.AbstractFactor
    group_a::Vector{Int}
    group_b::Vector{Int}
    a::Float64
    rmid::Float64
    d::Float64
end

factor_support_size(f::SeparationDoubleWell) = length(f.group_a) + length(f.group_b)
read_support(f::SeparationDoubleWell) = (f.group_a..., f.group_b...)

function _com(q, group)
    c = zeros(3)
    for i in group
        @inbounds for m in 1:3
            c[m] += q[m, i]
        end
    end
    return c ./ length(group)
end

function evaluate!(F::AbstractMatrix{Float64}, f::SeparationDoubleWell,
                   q::AbstractMatrix{Float64}, ctx)
    ca = _com(q, f.group_a)
    cb = _com(q, f.group_b)
    d = cb .- ca
    r = norm(d)
    r < 1e-12 && return Inf
    s = r - f.rmid
    u = f.a * (s^2 - f.d^2)^2
    dudr = 4.0 * f.a * s * (s^2 - f.d^2)
    # force on COM_b = -dU/dr * d/r; equal split over the group atoms
    c = -dudr / r / length(f.group_a)
    @inbounds for m in 1:3
        g = c * d[m]
        for i in f.group_a
            F[m, i] -= g
        end
        for i in f.group_b
            F[m, i] += g
        end
    end
    return u
end

"""
    two_cluster_system(; a = 8.0, rmid = 2.5, d = 0.9) -> (system, q0)

Two three-atom clusters. Each cluster is held by three harmonic bonds
(`k = 200`, `r0 = 1`) and internal Lennard-Jones pairs, so it is a genuine
molecular cluster. The only inter-cluster interaction is the coarse-grained
`SeparationDoubleWell`; no LJ cross term is added, so the barrier height is
exactly `a` and the two wells (`r = rmid ± d`) are exactly degenerate.

The initial configuration sits in the inner well (`r = rmid - d`).
"""
function two_cluster_system(; a::Float64 = 8.0, rmid::Float64 = 2.5,
                            d::Float64 = 0.9, k::Float64 = 200.0, r0::Float64 = 1.0)
    ctx = Factors.Context(Types.REDUCED, 6)
    # equilateral triangle of side r0 (centre of mass at the origin)
    tri = [0.0 1.0 0.5; 0.0 0.0 sqrt(3) / 2; 0.0 0.0 0.0]
    ca = [0.3, 0.2, 0.0]
    cb = ca .+ [rmid - d, 0.0, 0.0]
    q0 = zeros(3, 6)
    q0[:, 1:3] = tri .+ ca
    q0[:, 4:6] = tri .+ cb
    factors = Factors.AbstractFactor[]
    for g in (1:3, 4:6)
        for (i, j) in ((1, 2), (1, 3), (2, 3))
            push!(factors, Factors.HarmonicBond(g[i], g[j], k, r0))
            push!(factors, Factors.LennardJonesPair(g[i], g[j], 1.0, 1.0; mode = :none))
        end
    end
    push!(factors, SeparationDoubleWell(collect(1:3), collect(4:6), a, rmid, d))
    s = Dynamics.HarmonicSystem(factors, ones(6), ctx)
    return s, q0
end

"Distance between the two cluster centres of mass for a 3 x 6 configuration."
function cluster_separation(q::AbstractMatrix)
    return norm(_com(q, 1:3) .- _com(q, 4:6))
end
separation_series(samples) = [cluster_separation(q) for q in samples]

"Number of times `r` crosses `rstar` (a proxy for barrier crossings)."
function separation_crossings(r::AbstractVector{<:Real}, rstar::Real)
    c = 0
    prev = r[1] > rstar
    for v in r
        cur = v > rstar
        if cur != prev
            c += 1
            prev = cur
        end
    end
    return c
end

"""
    walker_round_trips(replica_index, walker) -> Int

How many times `walker` visits the hottest temperature slot and returns to the
coldest one. `replica_index[s, r]` is the initial walker label in slot `r` at
sweep `s` (as returned by `Sampling.replica_exchange`). This measures actual
exchange of configurations through the ladder, not merely local swaps.
"""
function walker_round_trips(replica_index::AbstractMatrix{Int}, walker::Int)
    ns, R = size(replica_index)
    slot = [findfirst(==(walker), view(replica_index, s, :)) for s in 1:ns]
    any(isnothing, slot) && error("walker $walker not found on every sweep")
    trips = 0
    at_top = false
    for s in slot
        if s == R
            at_top = true
        elseif s == 1 && at_top
            trips += 1
            at_top = false
        end
    end
    return trips
end

end # module
