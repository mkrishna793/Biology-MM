"""
Enhanced equilibrium sampling: HMC and parallel tempering, with acceptance
correctness measured against an analytically known target.

## Scope, stated before any result

This module samples **equilibrium** distributions `π(q) ∝ exp(-βU(q))`. It does
not generate physical trajectories, and no claim here says anything about
protein folding, kinetics, or rates. What is demonstrated is that a sampler which
is *constructed* to be exact (reversible, volume-preserving proposal + full-H
Metropolis correction) actually reproduces a known target to within measured
statistical error on a small analytic benchmark.

## Two backends, one algorithm

The original closures (`hmc_trajectory(q0, U, gradU, m, ...)`) are kept unchanged
for backward compatibility and remain the way the analytic targets are tested.
The **molecular backend** (`hmc_step!`, `hmc_trajectory(s::HarmonicSystem, ...)`,
`replica_exchange(s::HarmonicSystem, ...)`) takes the *real* factor energy field:

* `U(q)` comes from `Dynamics.forces!`, which evaluates every
  `Factors.AbstractFactor` and scatters `-∂U/∂q` in one pass (the same code path
  used by velocity Verlet and BAOAB), and
* the proposal is leapfrog on that system, so the target is exactly
  `exp(-βU(q)) dq` for any force-consistent factor set. The molecular LJ
  experiment below exercises the periodic `LennardJonesPair(mode = :lfs)`
  path; `PMEFactor` and `StreamedLJFactor` are force-consistent and therefore
  admissible, but HMC with those backends was **not** run here.

Nothing in this module knows about constraints. A `HarmonicSystem` does not
carry them, so a caller who has a constrained model and calls the molecular
sampler without passing `constraints` would silently sample the *unconstrained*
distribution. To make that impossible to do by accident, the molecular entry
points accept a `constraints` keyword and **refuse** any non-`nothing` value with
`SamplingConstraintError`; see the docstring of that type for why a correct
constrained HMC is not just "project at the end of a step".

## Two momenta, never conflated

MD uses the physical momentum from `Types.AtomSet`. HMC draws an **auxiliary**
momentum `p ~ N(0, M)` at the start of each trajectory and discards it. The same
field name is used for both in most code; here the HMC momentum is local to the
trajectory.

## The proposal

Velocity Verlet (leapfrog) is symplectic, hence volume preserving, and is an
involution up to momentum negation: `T(neg ∘ T(z)) = z`. We use the convention

    T(z) = (q', -p')     after L leapfrog steps

i.e. the momentum is negated at the end of the proposal. That makes `T` an exact
involution, which is what reversibility of the Metropolis proposal requires.
`test_sampling.jl` verifies the involution to machine precision rather than
assuming it.

## Collective-variable proposals

A proposal that moves only a subspace (a collective variable, a dihedral, a
reaction coordinate) is **not** automatically volume preserving in the full
configuration space. If the map has a nontrivial Jacobian, or the mass metric
restricts the kinetic term, the acceptance ratio must include
`log|det(∂z'/∂z)|` (and reversibility must be re-established). A hidden
projection that drops that term samples the wrong distribution. No such proposal
is implemented here; this is a documented extension point, not a solved problem.
"""
module Sampling

using ..Types
using ..Factors
using ..Dynamics
using ..Dynamics: N
using LinearAlgebra
using Random
using Printf
using Statistics

export hmc_trajectory, HMCResult, replica_exchange, ParallelTemperingResult,
       GaussianTarget, DoubleWell, exact_doublewell_z, doublewell_moments_exact,
       ks_distance, integrated_autocorrelation, effective_sample_size,
       acceptance_rate, swap_acceptance, barrier_crossings, round_trip_rate,
       # molecular (factor-energy) backend
       hmc_step!, hmc_proposal, MolecularHMCResult, MolecularPTResult,
       SamplingConstraintError, sampling_constraints_supported,
       molecular_beta, maxwell_momenta!, flatten_samples,
       swap_log_ratio, swap_accept_probability, correlated_mean,
       CorrelatedEstimate, ess_per_force_eval,
       # blocking & autocorrelation-aware observable summaries
       block_means, block_standard_error, block_variance_se,
       BlockedEstimate, blocked_mean, flyvbjerg_petersen_se,
       ObservableSummary, summarize_observable

# ---------------------------------------------------------------------------
# leapfrog proposal
# ---------------------------------------------------------------------------

"""
`L` leapfrog steps of size `dt` for `U` with diagonal mass `m`.

Returns the new `(q, p)` with the **momentum negated** so the whole map is an
involution. `gradU` is a closure returning `∇U(q)`.
"""
function leapfrog(q::Vector{Float64}, p::Vector{Float64}, gradU, m::Vector{Float64},
                  dt::Float64, L::Int)
    n = length(q)
    q = copy(q); p = copy(p)
    g = gradU(q)
    @inbounds for i in 1:n
        p[i] -= 0.5 * dt * g[i]
    end
    for step in 1:L
        @inbounds for i in 1:n
            q[i] += dt * p[i] / m[i]
        end
        g = gradU(q)
        spec = step == L ? 0.5 : 1.0
        @inbounds for i in 1:n
            p[i] -= spec * dt * g[i]
        end
    end
    # negate so T is an involution: T(T(q,p)) = (q,p)
    @inbounds for i in 1:n
        p[i] = -p[i]
    end
    return q, p
end

kinetic(p::Vector{Float64}, m::Vector{Float64}) = 0.5 * sum(p[i]^2 / m[i] for i in eachindex(p))

# ---------------------------------------------------------------------------
# HMC
# ---------------------------------------------------------------------------

struct HMCResult
    samples::Matrix{Float64}    # dim x nsamples
    accept::Vector{Bool}
    dH::Vector{Float64}
    acceptance_rate::Float64
end

"""
One HMC chain.

`nsteps` Metropolis-corrected trajectories starting from `q0`. Auxiliary momenta
are drawn from `N(0, M)`; acceptance uses the **full** Hamiltonian
`H = K + U`, so an inexact integrator is corrected rather than silently
biased. The target is exactly `exp(-βU)` as long as the proposal is reversible
and volume preserving — both verified in the test suite.

`U` must be a closure `q -> energy` and `gradU` a closure `q -> gradient`.
"""
function hmc_trajectory(q0::Vector{Float64}, U, gradU, m::Vector{Float64},
                        dt::Float64, L::Int, nsteps::Int, beta::Float64,
                        rng::AbstractRNG)
    n = length(q0)
    q = copy(q0)
    samples = Matrix{Float64}(undef, n, nsteps)
    accept = falses(nsteps)
    dHs = zeros(nsteps)
    Ucur = U(q)
    sig = sqrt.(m ./ beta)
    for s in 1:nsteps
        p = sig .* randn(rng, n)
        H0 = Ucur + kinetic(p, m)
        qp, pp = leapfrog(q, p, gradU, m, dt, L)
        Up = U(qp)
        if !isfinite(Up)
            samples[:, s] = q
            dHs[s] = -Inf
            continue
        end
        H1 = Up + kinetic(pp, m)
        dH = H1 - H0
        dHs[s] = dH
        if log(rand(rng)) < -beta * dH
            q = qp
            Ucur = Up
            accept[s] = true
        end
        samples[:, s] = q
    end
    return HMCResult(samples, accept, dHs, mean(accept))
end

# ---------------------------------------------------------------------------
# replica exchange / parallel tempering
# ---------------------------------------------------------------------------

struct ParallelTemperingResult
    samples::Vector{Matrix{Float64}}   # per replica, dim x nsamples
    beta::Vector{Float64}
    swap_attempts::Vector{Int}         # per adjacent pair
    swap_accepts::Vector{Int}
    replica_energy::Matrix{Float64}    # nsteps x nreplicas
end

"""
Parallel tempering over a ladder of inverse temperatures.

Each replica runs `hmc_trajectory`-style dynamics at its own `beta_r`; after a
block of steps, adjacent pairs `(r, r+1)` are proposed for exchange. The joint
target is `Π_r exp(-β_r U(q_r))`, so the swap acceptance is

    min(1, exp((β_r - β_{r+1}) (U(q_r) - U(q_{r+1}))))

which is the standard ratio and is *derived*, not tuned. Adjacent swaps are the
only ones proposed, which keeps the acceptance reasonable for a smooth ladder.

Returns per-replica samples and the per-pair swap acceptance counts.
"""
function replica_exchange(q0::Vector{Float64}, U, gradU, m::Vector{Float64},
                          betas::Vector{Float64}, dt::Float64, L::Int,
                          nsteps::Int, swap_every::Int, rng::AbstractRNG)
    R = length(betas)
    n = length(q0)
    qs = [copy(q0) for _ in 1:R]
    curU = [U(q) for q in qs]
    samples = [Matrix{Float64}(undef, n, nsteps) for _ in 1:R]
    n_att = zeros(Int, R - 1)
    n_acc = zeros(Int, R - 1)
    Ehist = zeros(nsteps, R)
    curU = [U(qs[r]) for r in 1:R]
    for s in 1:nsteps
        # one HMC move per replica
        for r in 1:R
            b = betas[r]
            sig = sqrt.(m ./ b)
            p = sig .* randn(rng, n)
            H0 = curU[r] + kinetic(p, m)
            qp, pp = leapfrog(qs[r], p, gradU, m, dt, L)
            Up = U(qp)
            if isfinite(Up)
                dH = Up + kinetic(pp, m) - H0
                if log(rand(rng)) < -b * dH
                    qs[r] = qp
                    curU[r] = Up
                end
            end
            samples[r][:, s] = qs[r]
        end
        Ehist[s, :] = curU
        # swap proposals
        if swap_every > 0 && s % swap_every == 0
            for r in 1:(R - 1)
                n_att[r] += 1
                arg = (betas[r] - betas[r+1]) * (curU[r] - curU[r+1])
                if log(rand(rng)) < arg
                    qs[r], qs[r+1] = qs[r+1], qs[r]
                    curU[r], curU[r+1] = curU[r+1], curU[r]
                    n_acc[r] += 1
                end
            end
        end
    end
    return ParallelTemperingResult(samples, copy(betas), n_att, n_acc, Ehist)
end

# ---------------------------------------------------------------------------
# analytic test targets
# ---------------------------------------------------------------------------

"`U(x) = 1/2 k (x - μ)²`; target is Gaussian with variance `1/(βk)`."
struct GaussianTarget
    k::Float64
    μ::Float64
end
(t::GaussianTarget)(x::Vector{Float64}) = 0.5 * t.k * sum(abs2, x .- t.μ)
grad(t::GaussianTarget, x::Vector{Float64}) = t.k .* (x .- t.μ)

"""
Symmetric double well `U(x) = a (x² - b)²`, minima at `±√b`, barrier height `a b²`.

This is the benchmark for barrier traversal: at low temperature the two wells are
separated by a barrier that a short HMC trajectory cannot cross, so a single
replica is trapped and its empirical distribution is wrong for the true
two-well target. Replica exchange restores traversal.
"""
struct DoubleWell
    a::Float64
    b::Float64
end
(d::DoubleWell)(x::Vector{Float64}) = d.a * sum(t -> (t^2 - d.b)^2, x)
grad(d::DoubleWell, x::Vector{Float64}) = [4.0 * d.a * (t^2 - d.b) * t for t in x]

"Normalising constant `Z = ∫ exp(-βU(x)) dx` for the 1-D double well, by quadrature."
function exact_doublewell_z(d::DoubleWell, beta::Float64; lo = -6.0, hi = 6.0, n = 400_001)
    xs = range(lo, hi, length = n)
    h = step(xs)
    s = 0.0
    for (i, x) in enumerate(xs)
        w = (i == 1 || i == n) ? 1.0 : (isodd(i) ? 4.0 : 2.0)
        s += w * exp(-beta * d([x]))
    end
    return s * h / 3.0
end

"Exact first two moments of the 1-D double well by quadrature (same grid)."
function doublewell_moments_exact(d::DoubleWell, beta::Float64;
                                  lo = -6.0, hi = 6.0, n = 400_001)
    xs = range(lo, hi, length = n)
    h = step(xs)
    s0 = 0.0; s1 = 0.0; s2 = 0.0
    for (i, x) in enumerate(xs)
        w = (i == 1 || i == n) ? 1.0 : (isodd(i) ? 4.0 : 2.0)
        p = w * exp(-beta * d([x]))
        s0 += p; s1 += p * x; s2 += p * x * x
    end
    s0 *= h / 3.0; s1 *= h / 3.0; s2 *= h / 3.0
    μ = s1 / s0
    var = s2 / s0 - μ^2
    return (mean = μ, var = var)
end

"Exact CDF of the 1-D double well on a grid, normalised by quadrature."
function doublewell_cdf_exact(d::DoubleWell, beta::Float64, xs::AbstractVector{Float64};
                              lo = -6.0, hi = 6.0, n = 400_001)
    g = collect(range(lo, hi, length = n))
    h = step(range(lo, hi, length = n))
    dens = Float64[]
    for (i, x) in enumerate(g)
        w = (i == 1 || i == n) ? 1.0 : (isodd(i) ? 4.0 : 2.0)
        push!(dens, w * exp(-beta * d([x])))
    end
    tot = sum(dens) * h / 3.0
    cdf = Float64[]
    for x in xs
        acc = 0.0
        for (i, xg) in enumerate(g)
            xg > x && break
            w = (i == 1 || i == n) ? 1.0 : (isodd(i) ? 4.0 : 2.0)
            acc += w * exp(-beta * d([xg]))
        end
        push!(cdf, acc * h / 3.0 / tot)
    end
    return cdf
end

# ---------------------------------------------------------------------------
# distributional tests
# ---------------------------------------------------------------------------

"Two-sample Kolmogorov–Smirnov distance between samples and a reference CDF."
function ks_distance(samples::AbstractVector{Float64}, refcdf::Function)
    xs = sort(samples)
    n = length(xs)
    d = 0.0
    for (i, x) in enumerate(xs)
        F = refcdf(x)
        d = max(d, abs(F - (i - 1) / n), abs(F - i / n))
    end
    return d
end

"KS critical value at level `alpha` (asymptotic); `alpha = 0.05` is conventional."
ks_critical(n::Int, alpha::Float64 = 0.05) = sqrt(-0.5 * log(alpha / 2)) / sqrt(n)

"""
Integrated autocorrelation time by the initial-positive-sequence estimator.

`τ = 1 + 2 Σ_{t=1}^{W} ρ_t` truncated at the first non-positive `ρ_t`. This is a
standard, deliberately conservative estimator; it is not exact and its own
uncertainty grows with `τ`, which is why the ESS is reported with the run length
rather than as a standalone number.
"""
function integrated_autocorrelation(x::AbstractVector{Float64}; maxlag::Int = -1)
    n = length(x)
    n >= 2 || throw(ArgumentError("need at least two samples"))
    all(isfinite, x) || throw(ArgumentError("samples must be finite"))
    μ = mean(x)
    v = sum(abs2, x .- μ) / n
    v <= 0 && return 1.0
    W = maxlag < 0 ? min(n - 1, 10 * max(1, n ÷ 100)) : maxlag
    tau = 1.0
    for t in 1:W
        c = sum((x[i] - μ) * (x[i+t] - μ) for i in 1:(n - t)) / (n * v)
        c <= 0 && break
        tau += 2.0 * c
    end
    return tau
end

integrated_autocorrelation(x::AbstractVector{<:Real}; maxlag::Int = -1) =
    integrated_autocorrelation(collect(Float64, x); maxlag = maxlag)

effective_sample_size(x::AbstractVector{Float64}; kwargs...) =
    length(x) / integrated_autocorrelation(x; kwargs...)

effective_sample_size(x::AbstractVector{<:Real}; kwargs...) =
    effective_sample_size(collect(Float64, x); kwargs...)

acceptance_rate(r::HMCResult) = r.acceptance_rate

swap_acceptance(p::ParallelTemperingResult) =
    [p.swap_attempts[i] == 0 ? NaN : p.swap_accepts[i] / p.swap_attempts[i]
     for i in eachindex(p.swap_attempts)]

"""
Count sign changes of `x` across zero, a proxy for barrier crossings in the
symmetric double well. Reported per replica.
"""
function barrier_crossings(x::AbstractVector{Float64})
    c = 0
    prev = sign(x[1])
    for v in x
        s = sign(v)
        if s != 0 && prev != 0 && s != prev
            c += 1
            prev = s
        end
    end
    return c
end

"""
Round-trip rate: how often a walker starting in one well visits the other and
returns. This is the quantity that actually measures whether tempering is
producing *equilibrium* exchange between wells, rather than occasional one-way
escapes (which would leave the low-temperature replicas unequilibrated).
"""
function round_trip_rate(x::AbstractVector{Float64})
    s = [sign(v) for v in x if v != 0]
    isempty(s) && return 0
    trips = 0
    side = s[1]
    visited_other = false
    for v in s
        if v != side
            if !visited_other
                visited_other = true
            else
                trips += 1
                visited_other = false
            end
            side = v
        end
    end
    return trips
end

# ===========================================================================
# molecular backend: HMC on a real factor energy field
# ===========================================================================

"""
    SamplingConstraintError(reason)

Thrown by the molecular sampler whenever a caller supplies constraints.

A constrained HMC proposal must preserve the *constrained* measure, not the
unconstrained one. Projecting positions onto the manifold at the end of a
leapfrog step and then applying the usual `exp(-βΔH)` acceptance is **not**
correct: the projection is not a flow of a Hamiltonian on the manifold, the
proposal is not the time reversal of itself (the momentum correction done by
RATTLE changes the map's Jacobian), and the measure on the manifold is
`sqrt(det G(q)) dσ` in local coordinates, which neither the unconstrained
Lebesgue measure nor the projected momentum update accounts for. A measure-
preserving constrained integrator is a different algorithm (e.g. geodesic
integration on the manifold, or a Gibbs-style move that samples the constrained
momentum subspace exactly) and is not implemented here.

`Dynamics` *does* implement RATTLE (`rattle_project!`) for deterministic
trajectories, where sampling a canonical measure is not the goal. Using it inside
a Metropolis sampler would produce samples from the wrong distribution with no
diagnostic that says so, so this module refuses loudly instead.

`sampling_constraints_supported === false` records the status as a value a test
can assert on, rather than as prose.
"""
struct SamplingConstraintError <: Exception
    reason::String
end
Base.showerror(io::IO, e::SamplingConstraintError) =
    print(io, "Sampling refuses a constrained HMC problem: ", e.reason)

"Sampling-level constraints are **not** supported; constrained HMC is refused."
const sampling_constraints_supported = false

function _refuse_constraints(constraints)
    constraints === nothing && return nothing
    throw(SamplingConstraintError(
        "constraints were supplied. The molecule-level sampler has no " *
        "measure-preserving constrained proposal; a projected leapfrog step " *
        "would target the unconstrained distribution. Remove the constraints " *
        "keyword only if the model really is unconstrained (the 3N-dimensional " *
        "Lebesgue measure), or use a constrained algorithm that is not this one."))
end

"""
    molecular_beta(s::HarmonicSystem, T) -> β

Inverse temperature in the system's own unit system, `β = 1/(k_B T)`.
`REDUCED` has `k_B = 1`.
"""
function molecular_beta(s::HarmonicSystem, T::Float64)
    isfinite(T) && T > 0 || throw(ArgumentError("temperature must be positive and finite, got $T"))
    return 1.0 / (T * s.ctx.units.boltzmann)
end

"""
    maxwell_momenta!(p, s, T, rng)

Fill `p` with `N(0, M k_B T)` — the mass-weighted Maxwell distribution at
temperature `T`. This is the HMC auxiliary momentum; it is *not* a physical
momentum from a trajectory.
"""
function maxwell_momenta!(p::Matrix{Float64}, s::HarmonicSystem, T::Float64,
                          rng::AbstractRNG)
    size(p) == (3, N(s)) || throw(ArgumentError("momentum buffer has the wrong shape"))
    σ = sqrt.(s.m .* (T * s.ctx.units.boltzmann))
    @inbounds for j in 1:N(s), d in 1:3
        p[d, j] = σ[j] * randn(rng)
    end
    return p
end

"""
    _leapfrog_molecular!(q, p, s, dt, L, F; U0 = nothing) -> U

One velocity-Verlet (leapfrog) sweep of `L` steps on the real factor energy.
`F` is a `3 x N` scratch buffer. Force evaluations: `L + 1` when `U0 === nothing`,
`L` when the caller already has `U0` in `F`.

No momentum negation is performed here; `hmc_proposal` applies it so the map is
an exact involution (see the module docstring).
"""
function _leapfrog_molecular!(q::Matrix{Float64}, p::Matrix{Float64},
                              s::HarmonicSystem, dt::Float64, L::Int,
                              F::Matrix{Float64}; U0 = nothing)
    n = N(s)
    U = U0 === nothing ? forces!(s, q, F) : U0
    # F = -∇U, so the kick is p += (dt/2) F  (equivalently p -= (dt/2) ∇U)
    @inbounds for j in 1:n, d in 1:3
        p[d, j] += 0.5 * dt * F[d, j]
    end
    for step in 1:L
        @inbounds for j in 1:n, d in 1:3
            q[d, j] += dt * p[d, j] / s.m[j]
        end
        U = forces!(s, q, F)
        spec = step == L ? 0.5 : 1.0
        @inbounds for j in 1:n, d in 1:3
            p[d, j] += spec * dt * F[d, j]
        end
    end
    return U
end

"""
    hmc_proposal(q, p, s::HarmonicSystem, dt, L) -> (q′, p′)

The Metropolis proposal `T = R ∘ Φ_L`, where `Φ_L` is `L` leapfrog steps on the
factor energy and `R` negates momentum. `T` is an exact involution and preserves
phase-space volume, which is what makes the acceptance below exact. Exposed so
the test suite can verify both properties on the *molecular* code path rather
than assuming they carry over from the closure path.
"""
function hmc_proposal(q::AbstractMatrix{Float64}, p::AbstractMatrix{Float64},
                      s::HarmonicSystem, dt::Float64, L::Int)
    size(q) == (3, N(s)) || throw(ArgumentError("q must be 3 x N"))
    size(p) == size(q) || throw(ArgumentError("p must have the same shape as q"))
    dt > 0 && isfinite(dt) || throw(ArgumentError("dt must be positive and finite"))
    L >= 1 || throw(ArgumentError("L must be at least 1"))
    check_force_model(s)
    qn = Matrix{Float64}(q)
    pn = Matrix{Float64}(p)
    F = zeros(3, N(s))
    _leapfrog_molecular!(qn, pn, s, dt, L, F)
    @inbounds for j in 1:N(s), d in 1:3
        pn[d, j] = -pn[d, j]
    end
    return qn, pn
end

"""
    hmc_step!(q, p, s::HarmonicSystem, dt, nsteps; T = 1.0, rng, constraints = nothing,
              F = zeros(3, N(s))) -> (accepted, dH, U, K, H, force_evals)

One Metropolis-corrected HMC move on the real factor energy field, in place.

`p` is the auxiliary momentum and must already have been drawn from
`maxwell_momenta!` (or any `N(0, M k_B T)`). On acceptance `q` becomes the new
configuration and `p` the negated final momentum; on rejection both are restored
bit-for-bit. The acceptance uses the **full** Hamiltonian
`H = 0.5 pᵀ M⁻¹ p + U(q)`, so integrator error is corrected, not ignored.

Returns a named tuple with at least `accepted` and `dH` (the requested return
values) plus the energy diagnostics and the force-evaluation count (`L + 1`,
independent of acceptance).

`constraints` must be `nothing`; anything else throws `SamplingConstraintError`
instead of silently sampling the unconstrained target.
"""
function hmc_step!(q::Matrix{Float64}, p::Matrix{Float64}, s::HarmonicSystem,
                   dt::Float64, nsteps::Int;
                   T::Float64 = 1.0,
                   rng::AbstractRNG = Random.default_rng(),
                   constraints = nothing,
                   F::Matrix{Float64} = zeros(3, N(s)))
    _refuse_constraints(constraints)
    size(q) == (3, N(s)) || throw(ArgumentError("q must be 3 x N with N=$(N(s))"))
    size(p) == size(q) || throw(ArgumentError("p must have the same shape as q"))
    size(F) == size(q) || throw(ArgumentError("F scratch buffer has the wrong shape"))
    dt > 0 && isfinite(dt) || throw(ArgumentError("dt must be positive and finite"))
    nsteps >= 1 || throw(ArgumentError("nsteps must be at least 1"))
    check_force_model(s)
    beta = molecular_beta(s, T)

    q0 = copy(q)
    p0 = copy(p)
    K0 = kinetic_energy(s, p0)
    U0 = forces!(s, q0, F)
    H0 = U0 + K0

    U1 = _leapfrog_molecular!(q, p, s, dt, nsteps, F; U0 = U0)
    @inbounds for j in 1:N(s), d in 1:3
        p[d, j] = -p[d, j]
    end
    K1 = kinetic_energy(s, p)
    H1 = U1 + K1
    dH = H1 - H0

    accepted = false
    if isfinite(U1) && isfinite(dH)
        # log(rand) < -β dH  ⇔  rand < exp(-β dH), without overflow
        accepted = log(rand(rng)) < -beta * dH
    else
        dH = Inf
    end

    if accepted
        return (accepted = true, dH = dH, U = U1, K = K1, H = H1,
                force_evals = nsteps + 1)
    else
        copyto!(q, q0)
        copyto!(p, p0)
        return (accepted = false, dH = dH, U = U0, K = K0, H = H0,
                force_evals = nsteps + 1)
    end
end

# ---------------------------------------------------------------------------
# molecular trajectory
# ---------------------------------------------------------------------------

struct MolecularHMCResult
    samples::Vector{Matrix{Float64}}   # each 3 x N
    U::Vector{Float64}                 # potential energy at each stored sample
    K::Vector{Float64}                 # auxiliary kinetic energy at each stored sample
    H::Vector{Float64}                 # U + K
    dH::Vector{Float64}                # proposed energy error of each stored move
    accepted::BitVector
    acceptance_rate::Float64
    force_evals::Int
    temperature::Float64
    beta::Float64
end

function _as_positions(q0, s::HarmonicSystem)
    if q0 isa AbstractMatrix
        size(q0) == (3, N(s)) || throw(ArgumentError("q0 must be 3 x N with N=$(N(s))"))
        return Matrix{Float64}(q0)
    else
        length(q0) == 3 * N(s) || throw(ArgumentError("q0 vector must have length 3N=$(3N(s))"))
        return reshape(collect(Float64, q0), 3, N(s))
    end
end

"""
    hmc_trajectory(s::HarmonicSystem, q0, nsamples; T = 1.0, dt, L, rng,
                   nwarmup = 0, thin = 1, constraints = nothing)

HMC on the real factor energy field: `nsamples` stored configurations, each the
result of `thin` Metropolis-corrected trajectories of `L` leapfrog steps.

The chain targets `exp(-βU(q))` with `β = 1/(k_B T)` in the system's units,
because the proposal is reversible, volume preserving, and corrected with the
full Hamiltonian. `nwarmup` moves are discarded before the first stored sample.

Per stored sample the result carries `U`, `K`, `H`, `dH` and `accepted`, plus the
total force-evaluation count so cost per effective sample can be reported
honestly (`ess_per_force_eval`).

`constraints` must be `nothing`; see `SamplingConstraintError`.
"""
function hmc_trajectory(s::HarmonicSystem, q0, nsamples::Int;
                        T::Float64 = 1.0, dt::Float64 = 0.002, L::Int = 20,
                        rng::AbstractRNG = Random.default_rng(),
                        nwarmup::Int = 0, thin::Int = 1, constraints = nothing)
    _refuse_constraints(constraints)
    nsamples >= 1 || throw(ArgumentError("nsamples must be at least 1"))
    nwarmup >= 0 || throw(ArgumentError("nwarmup must be non-negative"))
    thin >= 1 || throw(ArgumentError("thin must be at least 1"))
    q = _as_positions(q0, s)
    n = N(s)
    p = zeros(3, n)
    F = zeros(3, n)
    samples = Vector{Matrix{Float64}}(undef, nsamples)
    Us = zeros(nsamples); Ks = zeros(nsamples); Hs = zeros(nsamples)
    dHs = zeros(nsamples); acc = falses(nsamples)
    beta = molecular_beta(s, T)
    evals = 0
    # warmup never stores, so the same loop is used with a negative store index
    total = nwarmup + nsamples * thin
    for move in 1:total
        maxwell_momenta!(p, s, T, rng)
        r = hmc_step!(q, p, s, dt, L; T = T, rng = rng, F = F)
        evals += r.force_evals
        if move > nwarmup && (move - nwarmup) % thin == 0
            k = (move - nwarmup) ÷ thin
            samples[k] = copy(q)
            Us[k] = r.U; Ks[k] = r.K; Hs[k] = r.H
            dHs[k] = r.dH; acc[k] = r.accepted
        end
    end
    return MolecularHMCResult(samples, Us, Ks, Hs, dHs, acc, mean(acc), evals, T, beta)
end

"Stack the samples of a molecular result into a `3N x nsamples` matrix."
flatten_samples(r::MolecularHMCResult) = reduce(hcat, (vec(q) for q in r.samples))

acceptance_rate(r::MolecularHMCResult) = r.acceptance_rate

# ---------------------------------------------------------------------------
# molecular replica exchange
# ---------------------------------------------------------------------------

struct MolecularPTResult
    samples::Vector{Matrix{Float64}}   # per temperature slot, 3N x nsamples
    temperatures::Vector{Float64}
    beta::Vector{Float64}
    energy::Matrix{Float64}            # nsamples x nreplicas (potential energy)
    acceptance::Vector{Float64}        # per temperature slot
    swap_attempts::Vector{Int}         # per adjacent pair
    swap_accepts::Vector{Int}
    swap_u_low::Matrix{Float64}        # attempts x (R-1): U at the colder slot
    swap_u_high::Matrix{Float64}       # attempts x (R-1): U at the hotter slot
    replica_index::Matrix{Int}         # nsamples x nreplicas: which walker is where
    force_evals::Int
end

"Return replica `r`'s samples (`3N x nsamples`)."
flatten_samples(p::MolecularPTResult, replica::Int) = p.samples[replica]

"""
    replica_exchange(s::HarmonicSystem, q0, nsamples; temperatures, dt, L, rng,
                     swap_every = 1, nwarmup = 0, constraints = nothing)

Parallel tempering over replicas of the *same* molecular system at different
temperatures. Each replica runs the molecular HMC of `hmc_step!`; every
`swap_every` sweeps, adjacent temperature pairs `(r, r+1)` are proposed for
exchange.

The joint target is `Π_r exp(-β_r U(q_r))`, so exchanging configurations gives
the acceptance

    min(1, exp((β_r - β_{r+1}) (U_r - U_{r+1})))

which is implemented once (`swap_log_ratio` / `swap_accept_probability`) and
used by both the sampler and the two-state test.

Returns per-temperature-slot samples, per-replica potential-energy traces, and
per-adjacent-pair swap attempt/accept counts. `replica_index[s, r]` records which
initial walker occupies slot `r` at sweep `s`, so walker round trips can be
measured rather than guessed. `acceptance` is averaged over all sweeps,
including warmup (the production-only estimate differs only by `nwarmup`).

`constraints` must be `nothing`; see `SamplingConstraintError`.
"""
function replica_exchange(s::HarmonicSystem, q0, nsamples::Int;
                          temperatures::Vector{Float64},
                          dt::Float64 = 0.002, L::Int = 20,
                          rng::AbstractRNG = Random.default_rng(),
                          swap_every::Int = 1, nwarmup::Int = 0,
                          constraints = nothing)
    _refuse_constraints(constraints)
    R = length(temperatures)
    R >= 2 || throw(ArgumentError("parallel tempering needs at least two temperatures"))
    all(t -> isfinite(t) && t > 0, temperatures) ||
        throw(ArgumentError("temperatures must be positive and finite"))
    issorted(temperatures) || throw(ArgumentError("temperatures must be increasing"))
    nsamples >= 1 || throw(ArgumentError("nsamples must be at least 1"))
    swap_every >= 1 || throw(ArgumentError("swap_every must be at least 1"))
    nwarmup >= 0 || throw(ArgumentError("nwarmup must be non-negative"))
    check_force_model(s)
    n = N(s)
    betas = [molecular_beta(s, T) for T in temperatures]
    qs = [_as_positions(q0, s) for _ in 1:R]
    Us = zeros(R)
    Fs = [zeros(3, n) for _ in 1:R]
    # initial energies (recomputed by the first hmc_step! call anyway, this is
    # only so the swap block has a valid value before any move happens)
    for r in 1:R
        Us[r] = potential_energy(s, qs[r])
    end
    p = zeros(3, n)
    samples = [Matrix{Float64}(undef, 3n, nsamples) for _ in 1:R]
    energy = zeros(nsamples, R)
    acc = zeros(R)
    n_att = zeros(Int, R - 1)
    n_acc = zeros(Int, R - 1)
    perm = collect(1:R)
    replica_index = Matrix{Int}(undef, nsamples, R)
    total = nwarmup + nsamples
    max_swap = total ÷ swap_every
    swap_u_low = zeros(max_swap, R - 1)
    swap_u_high = zeros(max_swap, R - 1)
    swap_seen = zeros(Int, R - 1)
    evals = R   # the initial per-replica energy evaluation above
    for sweep in 1:total
        for r in 1:R
            maxwell_momenta!(p, s, temperatures[r], rng)
            res = hmc_step!(qs[r], p, s, dt, L; T = temperatures[r], rng = rng,
                            F = Fs[r])
            Us[r] = res.U
            evals += res.force_evals
            acc[r] += res.accepted ? 1.0 : 0.0
        end
        if sweep > nwarmup
            k = sweep - nwarmup
            for r in 1:R
                samples[r][:, k] = vec(qs[r])
                energy[k, r] = Us[r]
                replica_index[k, r] = perm[r]
            end
        end
        if sweep % swap_every == 0
            # sequential adjacent swaps; the same schedule for every pair keeps
            # the statistics per pair well defined
            for r in 1:(R - 1)
                n_att[r] += 1
                swap_seen[r] += 1
                swap_u_low[swap_seen[r], r] = Us[r]
                swap_u_high[swap_seen[r], r] = Us[r + 1]
                arg = swap_log_ratio(betas[r], betas[r + 1], Us[r], Us[r + 1])
                if log(rand(rng)) < arg
                    qs[r], qs[r + 1] = qs[r + 1], qs[r]
                    Us[r], Us[r + 1] = Us[r + 1], Us[r]
                    perm[r], perm[r + 1] = perm[r + 1], perm[r]
                    n_acc[r] += 1
                end
            end
        end
    end
    acceptance = [acc[r] / total for r in 1:R]
    return MolecularPTResult(samples, copy(temperatures), betas, energy, acceptance,
                             n_att, n_acc, swap_u_low, swap_u_high,
                             replica_index, evals)
end

swap_acceptance(p::MolecularPTResult) =
    [p.swap_attempts[i] == 0 ? NaN : p.swap_accepts[i] / p.swap_attempts[i]
     for i in eachindex(p.swap_attempts)]

"""
    swap_log_ratio(βi, βj, Ui, Uj)

Log of the parallel-tempering exchange ratio for adjacent replicas at inverse
temperatures `βi < βj` with potential energies `Ui`, `Uj`:

    (βi - βj) (Ui - Uj)

Derived from the joint target `exp(-βi Ui) exp(-βj Uj)`; not tuned.
"""
swap_log_ratio(βi::Real, βj::Real, Ui::Real, Uj::Real) = (βi - βj) * (Ui - Uj)

"Acceptance probability `min(1, exp(swap_log_ratio(...)))`, overflow-safe."
function swap_accept_probability(βi::Real, βj::Real, Ui::Real, Uj::Real)
    x = swap_log_ratio(βi, βj, Ui, Uj)
    return x >= 0 ? 1.0 : exp(x)
end

# ---------------------------------------------------------------------------
# correlated-sample statistics (shared by the molecular experiments)
# ---------------------------------------------------------------------------

"""
    CorrelatedEstimate

A mean with an autocorrelation-corrected standard error: `se = sqrt(var/n_eff)`
with `n_eff = n/τ` from `integrated_autocorrelation`. The estimator of `τ` is
itself noisy (that noise grows with `τ`), so `se` is a lower bound on the true
uncertainty in the heavy-correlation regime; experiments report `τ` alongside it
so this is visible.
"""
struct CorrelatedEstimate
    mean::Float64
    se::Float64
    tau::Float64
    n_eff::Float64
    n::Int
end

function correlated_mean(x::AbstractVector{<:Real}; maxlag::Int = -1)
    n = length(x)
    n >= 2 || throw(ArgumentError("need at least two samples"))
    τ = integrated_autocorrelation(collect(Float64, x); maxlag = maxlag)
    n_eff = n / τ
    se = sqrt(var(x) / n_eff)
    return CorrelatedEstimate(mean(x), se, τ, n_eff, n)
end

"""
    ess_per_force_eval(x; force_evals) -> (ess, per_eval)

Effective sample size of the series `x` divided by the number of force
evaluations spent. HMC spends `L + 1` evaluations per move regardless of
acceptance, so this is the fair cost measure against a Langevin run, which
spends one per step.
"""
function ess_per_force_eval(x::AbstractVector{<:Real}; force_evals::Integer)
    force_evals > 0 || throw(ArgumentError("force_evals must be positive"))
    ess = effective_sample_size(collect(Float64, x))
    return (ess = ess, per_eval = ess / force_evals)
end

# ---------------------------------------------------------------------------
# block averaging and autocorrelation-aware observable summaries
# ---------------------------------------------------------------------------

"""
    block_means(x, n_blocks = 20) -> Vector{Float64}

Partition `x` into `n_blocks` contiguous blocks of nearly equal size and return
the mean of each block. Requires `2 <= n_blocks <= length(x)` and finite values.
"""
function block_means(x::AbstractVector{<:Real}, n_blocks::Integer = 20)
    n = length(x)
    nb = Int(n_blocks)
    2 <= nb <= n || throw(ArgumentError("n_blocks=$nb must satisfy 2 <= n_blocks <= length(x)=$n"))
    all(isfinite, x) || throw(ArgumentError("samples must be finite"))
    edges = round.(Int, range(1, n + 1; length = nb + 1))
    return Float64[mean(@view x[edges[i]:(edges[i + 1] - 1)]) for i in 1:nb]
end

"""
    block_standard_error(x; n_blocks = 20) -> Float64

Non-overlapping block-averaged standard error of the mean `std(block_means) / sqrt(n_blocks)`.
"""
function block_standard_error(x::AbstractVector{<:Real}; n_blocks::Integer = 20)
    bm = block_means(x, n_blocks)
    return std(bm) / sqrt(length(bm))
end

"""
    block_variance_se(x; n_blocks = 20) -> Float64

Block-averaged standard error of the sample variance across `n_blocks` contiguous
blocks (each block must contain at least 2 samples).
"""
function block_variance_se(x::AbstractVector{<:Real}; n_blocks::Integer = 20)
    n = length(x)
    nb = Int(n_blocks)
    2 <= nb <= n ÷ 2 ||
        throw(ArgumentError("n_blocks=$nb requires at least 2 samples per block (n=$n)"))
    all(isfinite, x) || throw(ArgumentError("samples must be finite"))
    edges = round.(Int, range(1, n + 1; length = nb + 1))
    bv = Float64[var(@view x[edges[i]:(edges[i + 1] - 1)]) for i in 1:nb]
    return std(bv) / sqrt(nb)
end

struct BlockedEstimate
    mean::Float64
    std::Float64
    se::Float64
    n_blocks::Int
    block_size::Int
    n::Int
end

"""
    blocked_mean(x; n_blocks = 20) -> BlockedEstimate

Compute mean, sample standard deviation, and block-averaged standard error.
"""
function blocked_mean(x::AbstractVector{<:Real}; n_blocks::Integer = 20)
    n = length(x)
    nb = min(Int(n_blocks), n)
    nb >= 2 || throw(ArgumentError("need at least 2 samples for blocked_mean"))
    bm = block_means(x, nb)
    se = std(bm) / sqrt(nb)
    return BlockedEstimate(Float64(mean(x)), Float64(std(x)), se, nb, n ÷ nb, n)
end

"""
    flyvbjerg_petersen_se(x; min_blocks = 8)

Iterative pair-halving Flyvbjerg–Petersen blocking analysis. Repeatedly averages
adjacent pairs `(x[2k-1] + x[2k])/2` while at least `min_blocks` blocks remain,
recording `se_k = std(b_k)/sqrt(n_k)` and its uncertainty `se_k / sqrt(2*(n_k - 1))`.
Returns `(se, se_max, ses, err_ses, block_sizes)`, where `se` is the plateau
estimate (first level where `se` does not grow beyond `se_k + err_k`, falling back
to `se_max` if still rising).
"""
function flyvbjerg_petersen_se(x::AbstractVector{<:Real}; min_blocks::Integer = 8)
    n = length(x)
    mb = Int(min_blocks)
    mb >= 2 || throw(ArgumentError("min_blocks must be >= 2"))
    n >= mb || throw(ArgumentError("need at least min_blocks=$mb samples, got $n"))
    all(isfinite, x) || throw(ArgumentError("samples must be finite"))

    cur = collect(Float64, x)
    ses = Float64[]
    err_ses = Float64[]
    block_sizes = Int[]
    bsize = 1
    while length(cur) >= mb
        nk = length(cur)
        se_k = std(cur) / sqrt(nk)
        err_k = se_k / sqrt(2.0 * (nk - 1))
        push!(ses, se_k)
        push!(err_ses, err_k)
        push!(block_sizes, bsize)
        nk ÷ 2 >= mb || break
        m = nk ÷ 2
        nxt = Vector{Float64}(undef, m)
        @inbounds for i in 1:m
            nxt[i] = 0.5 * (cur[2i - 1] + cur[2i])
        end
        cur = nxt
        bsize *= 2
    end
    se_max = maximum(ses)
    se_plateau = se_max
    # Plateau heuristic: first index k where the remaining maximum does not exceed se[k] + err[k]
    for k in eachindex(ses)
        if maximum(@view ses[k:end]) <= ses[k] + err_ses[k]
            se_plateau = ses[k]
            break
        end
    end
    return (se = se_plateau, se_max = se_max, ses = ses, err_ses = err_ses,
            block_sizes = block_sizes)
end

struct ObservableSummary
    mean::Float64
    std::Float64
    var::Float64
    tau::Float64
    ess::Float64
    se_ess::Float64
    se_block::Float64
    se_fp::Float64
    se::Float64
    n::Int
end

"""
    summarize_observable(x; n_blocks = 20, maxlag = -1, min_blocks = 8) -> ObservableSummary

Autocorrelation-aware summary combining ESS (`integrated_autocorrelation`), fixed
non-overlapping block averaging (`block_standard_error`), and Flyvbjerg–Petersen
halving (`flyvbjerg_petersen_se`). `se = max(se_ess, se_block, se_fp)` provides a
conservative standard error across estimators.
"""
function summarize_observable(x::AbstractVector{<:Real};
                              n_blocks::Integer = 20,
                              maxlag::Int = -1,
                              min_blocks::Integer = 8)
    n = length(x)
    n >= max(2, Int(min_blocks)) ||
        throw(ArgumentError("need at least $(max(2, Int(min_blocks))) samples, got $n"))
    xv = collect(Float64, x)
    all(isfinite, xv) || throw(ArgumentError("samples must be finite"))
    μ = mean(xv)
    v = var(xv)
    s = sqrt(v)
    τ = integrated_autocorrelation(xv; maxlag = maxlag)
    ess = n / τ
    se_ess = sqrt(v / ess)
    nb = min(Int(n_blocks), n)
    se_block = block_standard_error(xv; n_blocks = nb)
    fp = flyvbjerg_petersen_se(xv; min_blocks = min_blocks)
    se_fp = fp.se
    return ObservableSummary(μ, s, v, τ, ess, se_ess, se_block, se_fp,
                             max(se_ess, se_block, se_fp), n)
end

end # module Sampling
