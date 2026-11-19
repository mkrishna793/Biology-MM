"""
Hamiltonian dynamics with an explicit numerical-accuracy layer.

`H(q,p) = 1/2 pᵀ M⁻¹ p + U(q)`.

## What is and is not claimed

Velocity Verlet is symplectic and time-reversible for a separable Hamiltonian. It
does **not** conserve energy exactly: it conserves a *shadow* Hamiltonian to
`O(dt²)`, so the energy error is bounded and oscillatory rather than secular in
integrable regimes, and random-walks inside that band in chaotic ones. Everything
in this module is written to keep that distinction visible — `energy_drift_report`
returns a drift *slope with a standard error* precisely so that "bounded
oscillation" and "slow leak" are distinguishable by a number rather than by eye.

## Why there is no naive adaptive time step

The tempting "if the local error is large, shrink dt" controller **breaks both**
properties that make Verlet worth using: a state-dependent `dt(q,p)` is neither
symplectic nor time-reversible, so the shadow-Hamiltonian argument fails and the
energy can drift even though every individual step looked accurate. This module
therefore refuses that design:

* `select_dt` performs **offline validation** — trial runs at a ladder of fixed
  `dt` values — and returns a single fixed `dt` with the measured evidence.
* `step_doubling_defect` is a *local heuristic estimate*, not a certified bound.
* Nothing here adapts `dt` during production integration.

A rigorous adaptive symplectic scheme exists in the literature (e.g. reversible
time-transformed / regularised integrators), but it is a different algorithm and
is not implemented here.
"""
module Dynamics

using ..Types
using ..Types: System
using ..Factors
using ..Constraints
using ..Settle
using LinearAlgebra
using Random
using Printf

export State, HarmonicSystem, kinetic_energy, potential_energy, total_energy,
       forces!, velocity_verlet_step!, run_trajectory, TrajectoryRecord,
       step_doubling_defect, EnergyDriftReport, energy_drift_report, select_dt,
       StepSelection, ErrorBudget, kahan_sum, neumaier_sum,
       langevin_baoab_step!, rattle_project!, constraints_supported,
       ConstraintConvergenceError, EnergySeries, force_model_supported, check_force_model,
       FrameWriter, write_frame!, read_xyz_frames,
       INTEGRATOR_STAGE_KINDS, IntegratorStage, IntegratorSpec,
       integrator_spec, stage_classification, is_reversible, is_stochastic,
       DynamicsCheckpoint, checkpoint, restore!,
       rattle_residuals, rattle_project_with_status!

# ---------------------------------------------------------------------------
# integrator state machine & reversible / stochastic stage classification
# ---------------------------------------------------------------------------

"""
Canonical stage kinds for the integrator state machine:
- `:force_evaluation`     — evaluate `-∂U/∂q` (`:reversible`)
- `:half_kick`            — momentum update `p ← p + (dt/2) F` (`:reversible`)
- `:position_drift`       — Cartesian drift `q ← q + h p / m` (`:reversible`)
- `:rigid_body_drift`     — symmetric principal-axis rigid drift (`:reversible`)
- `:position_constraints` — holonomic position manifold projection (`:reversible`)
- `:velocity_constraints` — mass-metric tangent velocity projection (`:reversible`)
- `:thermostat_ou`        — exact Ornstein–Uhlenbeck momentum update (`:stochastic`)
- `:diagnostics`          — energy/residual/momentum sampling (`:diagnostic`)
"""
const INTEGRATOR_STAGE_KINDS = (
    :force_evaluation,
    :half_kick,
    :position_drift,
    :rigid_body_drift,
    :position_constraints,
    :velocity_constraints,
    :thermostat_ou,
    :diagnostics,
)

"Classify an individual integrator stage kind as `:reversible`, `:stochastic`, or `:diagnostic`."
function stage_classification(stage::Symbol)
    stage === :thermostat_ou && return :stochastic
    stage === :diagnostics && return :diagnostic
    stage in INTEGRATOR_STAGE_KINDS && return :reversible
    throw(ArgumentError("unknown integrator stage kind: :$stage"))
end

"Single stage in an integrator state machine."
struct IntegratorStage
    name::Symbol
    classification::Symbol
    reversible::Bool
    stochastic::Bool
    description::String
    function IntegratorStage(name::Symbol, description::String = "")
        cls = stage_classification(name)
        new(name, cls, cls === :reversible, cls === :stochastic, description)
    end
end

"Complete stage specification and reversibility classification for an integrator."
struct IntegratorSpec
    name::Symbol
    classification::Symbol
    stages::Vector{IntegratorStage}
    function IntegratorSpec(name::Symbol, stages::Vector{IntegratorStage})
        any_stoch = any(s -> s.stochastic, stages)
        cls = any_stoch ? :stochastic : :reversible
        new(name, cls, stages)
    end
end

is_reversible(s::IntegratorStage) = s.reversible
is_stochastic(s::IntegratorStage) = s.stochastic
is_reversible(spec::IntegratorSpec) = spec.classification === :reversible
is_stochastic(spec::IntegratorSpec) = spec.classification === :stochastic

"""
    integrator_spec(name::Symbol) -> IntegratorSpec

Return the explicit stage machine specification and reversible/stochastic
classification for a supported `Dynamics` integrator:
`:velocity_verlet`, `:rattle_verlet`, `:baoab`, or `:constrained_baoab`.
"""
function integrator_spec(name::Symbol)
    if name === :velocity_verlet
        return IntegratorSpec(:velocity_verlet, [
            IntegratorStage(:force_evaluation, "ensure F = -∂U/∂q at current q if stale"),
            IntegratorStage(:half_kick, "p ← p + (dt/2) F"),
            IntegratorStage(:position_drift, "q ← q + dt p/m"),
            IntegratorStage(:force_evaluation, "F ← -∂U/∂q at updated q"),
            IntegratorStage(:half_kick, "p ← p + (dt/2) F"),
            IntegratorStage(:diagnostics, "optional trajectory/energy recording"),
        ])
    elseif name === :rattle_verlet
        return IntegratorSpec(:rattle_verlet, [
            IntegratorStage(:force_evaluation, "ensure F = -∂U/∂q at current q if stale"),
            IntegratorStage(:half_kick, "p ← p + (dt/2) F"),
            IntegratorStage(:position_drift, "q_trial ← q + dt p/m"),
            IntegratorStage(:position_constraints, "SHAKE/RATTLE position projection + momentum impulse m Δq / dt"),
            IntegratorStage(:force_evaluation, "F ← -∂U/∂q at constrained q"),
            IntegratorStage(:half_kick, "p ← p + (dt/2) F"),
            IntegratorStage(:velocity_constraints, "RATTLE mass-metric tangent velocity projection"),
            IntegratorStage(:diagnostics, "optional trajectory/energy/residual recording"),
        ])
    elseif name === :baoab
        return IntegratorSpec(:baoab, [
            IntegratorStage(:half_kick, "B: p ← p + (dt/2) F"),
            IntegratorStage(:position_drift, "A: q ← q + (dt/2) p/m"),
            IntegratorStage(:thermostat_ou, "O: exact Ornstein–Uhlenbeck momentum update"),
            IntegratorStage(:position_drift, "A: q ← q + (dt/2) p/m"),
            IntegratorStage(:force_evaluation, "F ← -∂U/∂q at updated q"),
            IntegratorStage(:half_kick, "B: p ← p + (dt/2) F"),
            IntegratorStage(:diagnostics, "optional trajectory/energy recording"),
        ])
    elseif name === :constrained_baoab
        return IntegratorSpec(:constrained_baoab, [
            IntegratorStage(:half_kick, "B: p ← p + (dt/2) F"),
            IntegratorStage(:position_drift, "A: q ← q + (dt/2) p/m"),
            IntegratorStage(:thermostat_ou, "O: exact Ornstein–Uhlenbeck momentum update"),
            IntegratorStage(:position_drift, "A: q ← q + (dt/2) p/m"),
            IntegratorStage(:position_constraints, "coupled mass-metric position projection"),
            IntegratorStage(:velocity_constraints, "coupled mass-metric tangent velocity projection"),
            IntegratorStage(:force_evaluation, "F ← -∂U/∂q at constrained q"),
            IntegratorStage(:half_kick, "B: p ← p + (dt/2) F"),
            IntegratorStage(:velocity_constraints, "final RATTLE tangent velocity projection"),
            IntegratorStage(:diagnostics, "optional trajectory/energy/residual recording"),
        ])
    else
        throw(ArgumentError("unknown integrator :$name; expected :velocity_verlet, :rattle_verlet, :baoab, or :constrained_baoab"))
    end
end

# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------

"""
Dynamic state. `q` and `p` are `3 x N`. Position is kept unwrapped: wrapping is a
*display* operation (`Types.wrap!`), not a dynamics one, because wrapping mid-run
would make the trajectory non-differentiable in the stored coordinates and break
the reversibility test for no physical benefit.

The force buffer `F` and the `force_valid` flag are part of the state because
velocity Verlet *requires* the force at the current `q` at the start of a step.
An earlier version of this code allocated a zeroed `F` inside the convenience
wrapper, which silently turned Verlet into a first-order drift-kick scheme —
symplectic, so its energy still looked bounded, which is exactly why the bug was
invisible in an energy plot and only showed up as a reversibility failure. The
flag makes that class of mistake impossible: `F` is recomputed whenever `q` has
moved without it.
"""
mutable struct State
    q::Matrix{Float64}
    p::Matrix{Float64}
    F::Matrix{Float64}
    force_valid::Bool
end
function State(q::AbstractMatrix{<:Real}, p::AbstractMatrix{<:Real})
    q = Matrix{Float64}(q); p = Matrix{Float64}(p)
    size(q) == size(p) || error("q and p must have the same shape")
    return State(q, p, zeros(size(q)), false)
end
function Base.copy(s::State)
    return State(copy(s.q), copy(s.p), copy(s.F), s.force_valid)
end
"Explicitly mark the cached force stale (call after mutating `q`)."
invalidate!(s::State) = (s.force_valid = false; s)

"""
Exact deterministic snapshot of dynamic state `(q, p, F, force_valid)`, step
counter, simulation time, and optional RNG state for bit-exact replay.
"""
struct DynamicsCheckpoint
    state::State
    step::Int
    t::Float64
    rng::Union{Nothing,AbstractRNG}
end

"""
    checkpoint(st::State; step=0, t=0.0, rng=nothing) -> DynamicsCheckpoint

Capture an exact deep-copied snapshot of `st` (and `copy(rng)` if provided).
"""
function checkpoint(st::State; step::Integer = 0, t::Real = 0.0,
                    rng::Union{Nothing,AbstractRNG} = nothing)
    return DynamicsCheckpoint(copy(st), Int(step), Float64(t),
                              rng === nothing ? nothing : copy(rng))
end

"""
    restore!(st::State, cp::DynamicsCheckpoint; rng=nothing) -> State

Restore `st` (and `rng` if both `rng` and `cp.rng` are present) to the exact
checkpointed state.
"""
function restore!(st::State, cp::DynamicsCheckpoint;
                  rng::Union{Nothing,AbstractRNG} = nothing)
    size(st.q) == size(cp.state.q) && size(st.p) == size(cp.state.p) ||
        throw(ArgumentError("checkpoint shape mismatch"))
    copyto!(st.q, cp.state.q)
    copyto!(st.p, cp.state.p)
    copyto!(st.F, cp.state.F)
    st.force_valid = cp.state.force_valid
    if rng !== nothing && cp.rng !== nothing
        copy!(rng, cp.rng)
    end
    return st
end

# ---------------------------------------------------------------------------
# system
# ---------------------------------------------------------------------------

"""
A separable Hamiltonian system: factors for `U`, masses for the kinetic metric.

`H(q,p) = Σ p_j²/(2 m_j) + U(q)`. The mass matrix is diagonal; a full metric
`M` is a documented extension point (`kinetic_energy` is the only place it
enters).
"""
struct HarmonicSystem
    factors::Vector{AbstractFactor}
    m::Vector{Float64}
    ctx::Context
    function HarmonicSystem(factors, m::AbstractVector{<:Real}, ctx::Context)
        m = Vector{Float64}(m)
        length(m) == ctx.N || error("mass count $(length(m)) != N=$(ctx.N)")
        all(x -> isfinite(x) && x > 0, m) || error("masses must be positive and finite")
        new(collect(AbstractFactor, factors), m, ctx)
    end
end

function HarmonicSystem(v::System, factors)
    HarmonicSystem(factors, v.atoms.m, Context(v.box, v.units, size(v.atoms.q, 2)))
end

N(s::HarmonicSystem) = s.ctx.N

kinetic_energy(s::HarmonicSystem, p::AbstractMatrix{Float64}) =
    0.5 * sum(@inbounds(p[d, j]^2 / s.m[j]) for j in 1:N(s) for d in 1:3)

potential_energy(s::HarmonicSystem, q::AbstractMatrix{Float64}) =
    total_energy_forces(s.factors, q, s.ctx)[1]

total_energy(s::HarmonicSystem, q, p) = kinetic_energy(s, p) + potential_energy(s, q)

"""
Fill `F` with `-∂U/∂q`, return `U`.

`total_energy_forces` returns `(U, F)`; this wrapper exists so integrators read
`U = forces!(...)` without unpacking a tuple every step.
"""
function forces!(s::HarmonicSystem, q::AbstractMatrix{Float64}, F::Matrix{Float64})
    U, _ = total_energy_forces(s.factors, q, s.ctx; F = F)
    return U
end

# ---------------------------------------------------------------------------
# force-model gate: refuse non-conservative force fields before evolving
# ---------------------------------------------------------------------------

"""
    force_model_supported(s::HarmonicSystem) -> Bool

`true` when every factor's scattered force is the exact derivative of the energy
it returns (`Factors.force_consistent`).

This is the structural precondition for *evolution*, not for energy evaluation: a
factor such as `LennardJonesPair(mode = :trunc)` still has a well-defined energy,
so constructing a `HarmonicSystem` and calling `potential_energy` on it is allowed.
Integrating it is not, because a trajectory would then have no conserved shadow
Hamiltonian: the force can be discontinuous at `rc` while the reported energy
jumps, so the integrator would silently leak or inject energy.
"""
force_model_supported(s::HarmonicSystem) = all(force_consistent, s.factors)

function _force_model_name(f::AbstractFactor)
    name = string(nameof(typeof(f)))
    # Report the cutoff convention when the factor has one; it is the actionable
    # detail (`mode = :lfs` is the supported production choice).
    if hasfield(typeof(f), :mode)
        return name * "(mode = :" * string(getfield(f, :mode)) * ")"
    end
    return name
end

"""
    check_force_model(s::HarmonicSystem)

Throw an `ErrorException` unless every factor in `s` is force-consistent.

Every function in this module that moves `(q, p)` calls this **before** touching
the state, so a refused integration leaves the state bit-for-bit unchanged. The
check is a per-call scan of the factor list (not a cached flag) because
`HarmonicSystem.factors` is a mutable vector: a factor with a non-conservative
cutoff could otherwise be appended after construction and slip past the gate.
Energy-only entry points (`potential_energy`, `total_energy`, `forces!`) are not
gated; inspecting an inconsistent model is legitimate, evolving it is not.
"""
function check_force_model(s::HarmonicSystem)
    bad = AbstractFactor[f for f in s.factors if !force_consistent(f)]
    isempty(bad) && return nothing
    offenders = join(unique(_force_model_name.(bad)), ", ")
    error("refusing to integrate a system containing non-force-consistent " *
          "factor(s): $offenders. Their force is not the derivative of their " *
          "energy across the cutoff, so a step that crosses it has no conserved " *
          "shadow Hamiltonian and the run silently drifts. Use a C^1 cutoff " *
          "such as LennardJonesPair(mode = :lfs, rc = ...); :trunc and :shift " *
          "are for energy evaluation only.")
end

# ---------------------------------------------------------------------------
# compensated summation
# ---------------------------------------------------------------------------

"""
Kahan compensated summation. Reduces the `O(n ε)` growth of naive summation to
`O(ε)` for the *sum*, at the cost of one extra subtraction per term.

This matters for energies: a total energy is a small difference of large
per-factor contributions, so the naive sum can lose most of its significant
digits precisely where drift is being measured.
"""
function kahan_sum(xs)
    s = 0.0
    c = 0.0
    for x in xs
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    end
    return s
end

"Neumaier variant: also correct when a term is larger than the running sum."
function neumaier_sum(xs)
    s = 0.0
    c = 0.0
    for x in xs
        t = s + x
        if abs(s) >= abs(x)
            c += (s - t) + x
        else
            c += (x - t) + s
        end
        s = t
    end
    return s + c
end

# ---------------------------------------------------------------------------
# velocity verlet (explicit state-machine stages)
# ---------------------------------------------------------------------------

@inline _emit_stage(hook, stage::Symbol) = (hook === nothing || hook(stage); nothing)

@inline function _stage_half_kick!(p::Matrix{Float64}, F::Matrix{Float64},
                                   dt::Float64, n::Int, hook = nothing)
    _emit_stage(hook, :half_kick)
    h = 0.5 * dt
    @inbounds for j in 1:n, d in 1:3
        p[d, j] += h * F[d, j]
    end
    return nothing
end

@inline function _stage_drift!(q::Matrix{Float64}, p::Matrix{Float64},
                               m::Vector{Float64}, h::Float64, n::Int, hook = nothing)
    _emit_stage(hook, :position_drift)
    @inbounds for j in 1:n, d in 1:3
        q[d, j] += h * p[d, j] / m[j]
    end
    return nothing
end

@inline function _stage_force_eval!(q::Matrix{Float64}, F::Matrix{Float64},
                                    s::HarmonicSystem, hook = nothing)
    _emit_stage(hook, :force_evaluation)
    return forces!(s, q, F)
end

@inline function _stage_thermostat_ou!(p::Matrix{Float64}, m::Vector{Float64},
                                      dt::Float64, gamma::Float64, beta::Float64,
                                      n::Int, rng::AbstractRNG, hook = nothing)
    _emit_stage(hook, :thermostat_ou)
    c1 = exp(-gamma * dt)
    c2 = sqrt((1.0 - c1 * c1) / beta)
    @inbounds for j in 1:n, d in 1:3
        p[d, j] = c1 * p[d, j] + c2 * sqrt(m[j]) * randn(rng)
    end
    return nothing
end

"""
One velocity-Verlet step, in place on `st`.

Stages (`integrator_spec(:velocity_verlet)` or `:rattle_verlet`, all `:reversible`):
1. `:force_evaluation` (if `force_valid == false`)
2. `:half_kick` (`p ← p + (dt/2) F(q)`)
3. `:position_drift` (`q ← q + dt p/m`)
4. `:position_constraints` (if `constraints !== nothing`, plus momentum impulse `m Δq / dt`)
5. `:force_evaluation` (`F ← -∂U/∂q`)
6. `:half_kick` (`p ← p + (dt/2) F(q)`)
7. `:velocity_constraints` (if `constraints !== nothing`)

Symplectic and time-reversible. The force at the *current* `q` must be known at
entry; if the cached force is stale it is recomputed (a correctness requirement,
not an optimization). Each step costs exactly one new force evaluation.
"""
function velocity_verlet_step!(st::State, s::HarmonicSystem, dt::Float64;
                               constraints = nothing, constraint_tolerance::Float64 = 1e-12,
                               constraint_max_iterations::Int = 100,
                               stage_hook = nothing)
    check_force_model(s)
    (isfinite(dt) && dt != 0.0) || throw(ArgumentError("dt must be finite and non-zero"))
    n = N(s)
    if !st.force_valid
        _stage_force_eval!(st.q, st.F, s, stage_hook)
        st.force_valid = true
    end
    _stage_half_kick!(st.p, st.F, dt, n, stage_hook)
    _stage_drift!(st.q, st.p, s.m, dt, n, stage_hook)
    if constraints !== nothing
        _emit_stage(stage_hook, :position_constraints)
        q_before_drift = st.q .- dt .* st.p ./ reshape(s.m, 1, :)
        q_unconstrained = copy(st.q)
        _project_positions!(st.q, s, constraints, constraint_tolerance,
                            constraint_max_iterations; reference = q_before_drift)
        @inbounds for j in 1:n, d in 1:3
            st.p[d, j] += s.m[j] * (st.q[d, j] - q_unconstrained[d, j]) / dt
        end
    end
    _stage_force_eval!(st.q, st.F, s, stage_hook)
    _stage_half_kick!(st.p, st.F, dt, n, stage_hook)
    if constraints !== nothing
        _emit_stage(stage_hook, :velocity_constraints)
        _project_velocities!(st.q, st.p, s, constraints,
                             constraint_tolerance,
                             constraint_max_iterations)
    end
    st.force_valid = true
    return nothing
end

"""
Lower-level step on explicit `(q, p, F)` buffers.

Kept for callers that already manage the buffers themselves (e.g. the trajectory
runner, which reuses `F` across steps). `F` must hold `-∂U/∂q` at `q` on entry and
is left holding the value at the new `q`.
"""
function velocity_verlet_step!(q::Matrix{Float64}, p::Matrix{Float64},
                               F::Matrix{Float64}, s::HarmonicSystem, dt::Float64;
                               stage_hook = nothing)
    check_force_model(s)
    (isfinite(dt) && dt != 0.0) || throw(ArgumentError("dt must be finite and non-zero"))
    n = N(s)
    _stage_half_kick!(p, F, dt, n, stage_hook)
    _stage_drift!(q, p, s.m, dt, n, stage_hook)
    U = _stage_force_eval!(q, F, s, stage_hook)
    _stage_half_kick!(p, F, dt, n, stage_hook)
    return U
end

# ---------------------------------------------------------------------------
# trajectory recording
# ---------------------------------------------------------------------------

struct TrajectoryRecord
    step::Int
    t::Float64
    U::Float64
    K::Float64
    E::Float64
end

"Energy history with the compensated total kept separately from the naive one."
struct EnergySeries
    naive::Vector{Float64}
    compensated::Vector{Float64}
end
Base.length(e::EnergySeries) = length(e.naive)

"""Buffered XYZ trajectory writer with a JSONL metadata sidecar.

Coordinates are written in XYZ (Å convention is caller-defined); each corresponding
JSONL record stores step, time, and orthorhombic box lengths. XYZ itself has no
standard unit-cell field. `labels` defaults to `X` and must have one label per atom.
"""
mutable struct FrameWriter
    xyz::IOStream
    metadata::IOStream
    labels::Vector{String}
    n::Int
    closed::Bool
end

function FrameWriter(path::AbstractString, n::Integer; labels=fill("X", n))
    n > 0 || throw(ArgumentError("frame atom count must be positive"))
    length(labels) == n || throw(ArgumentError("one label is required per atom"))
    xyzpath = endswith(lowercase(path), ".xyz") ? String(path) : String(path) * ".xyz"
    mkpath(dirname(abspath(xyzpath)))
    return FrameWriter(open(xyzpath, "w"), open(xyzpath * ".jsonl", "w"),
                       String.(labels), Int(n), false)
end

"""Append one XYZ frame and its JSONL step/time/box record."""
function write_frame!(w::FrameWriter, q::AbstractMatrix, box, step::Integer, time::Real)
    w.closed && throw(ArgumentError("frame writer is closed"))
    size(q) == (3, w.n) || throw(ArgumentError("coordinate shape does not match writer"))
    L = box isa Types.OrthorhombicBox ? collect(box.L) : Float64.(box)
    length(L) == 3 && all(x -> isfinite(x) && x > 0, L) ||
        throw(ArgumentError("box must contain three finite positive lengths"))
    println(w.xyz, w.n)
    println(w.xyz, "step=", step, " time=", repr(Float64(time)))
    @inbounds for i in 1:w.n
        println(w.xyz, w.labels[i], ' ', repr(Float64(q[1,i])), ' ',
                repr(Float64(q[2,i])), ' ', repr(Float64(q[3,i])))
    end
    println(w.metadata, "{\"step\":", step, ",\"time\":", repr(Float64(time)),
            ",\"box\":[", repr(L[1]), ',', repr(L[2]), ',', repr(L[3]), "]}")
    return w
end

function Base.close(w::FrameWriter)
    w.closed && return nothing
    close(w.xyz); close(w.metadata); w.closed = true
    return nothing
end

"""Read XYZ frames as `(labels, coordinates)` tuples (coordinates are 3×N)."""
function read_xyz_frames(path::AbstractString)
    frames = Tuple{Vector{String},Matrix{Float64}}[]
    open(path, "r") do io
        while !eof(io)
            nline = readline(io); isempty(strip(nline)) && continue
            n = parse(Int, strip(nline)); readline(io)
            labels = Vector{String}(undef, n); q = Matrix{Float64}(undef, 3, n)
            for i in 1:n
                fields = split(strip(readline(io)))
                length(fields) == 4 || throw(ArgumentError("malformed XYZ atom row"))
                labels[i] = fields[1]
                q[:,i] .= parse.(Float64, fields[2:4])
            end
            push!(frames, (labels, q))
        end
    end
    return frames
end

"""
Integrate `nsteps` steps of fixed `dt`, recording energies.

`compensated = true` recomputes the total energy with `neumaier_sum` over the
per-factor contributions, which is the honest number to use when measuring drift.
"""
function run_trajectory(st::State, s::HarmonicSystem, dt::Float64, nsteps::Int;
                        record_every::Int = 1, compensated::Bool = true,
                        constraints = nothing, constraint_tolerance::Float64 = 1e-12,
                        constraint_max_iterations::Int = 100,
                        frame_writer::Union{Nothing,FrameWriter}=nothing,
                        frame_every::Int=record_every,
                        frame_box=s.ctx.box)
    check_force_model(s)
    record_every > 0 || throw(ArgumentError("record_every must be positive"))
    frame_every > 0 || throw(ArgumentError("frame_every must be positive"))
    frame_writer !== nothing && frame_writer.n != N(s) &&
        throw(ArgumentError("frame writer atom count differs from system"))
    F = st.F
    U = forces!(s, st.q, F)
    st.force_valid = true
    recs = TrajectoryRecord[]
    Es_naive = Float64[]
    Es_comp = Float64[]
    t = 0.0
    for step in 1:nsteps
        if constraints === nothing
            U = velocity_verlet_step!(st.q, st.p, F, s, dt)
        else
            # State overload owns the constraint projections and force cache.
            velocity_verlet_step!(st, s, dt; constraints, constraint_tolerance,
                                  constraint_max_iterations)
            U = potential_energy(s, st.q)
            F = st.F
        end
        t += dt
        frame_writer !== nothing && step % frame_every == 0 &&
            write_frame!(frame_writer, st.q, frame_box, step, t)
        if step % record_every == 0
            K = kinetic_energy(s, st.p)
            E_naive = K + U
            E_comp = compensated ? _compensated_energy(s, st.q, st.p) : E_naive
            push!(recs, TrajectoryRecord(step, t, U, K, E_comp))
            push!(Es_naive, E_naive)
            push!(Es_comp, E_comp)
        end
    end
    return recs, EnergySeries(Es_naive, Es_comp)
end

@inline q_of(st::State) = st.q

"Total energy with per-factor contributions summed by Neumaier summation."
function _compensated_energy(s::HarmonicSystem, q::Matrix{Float64}, p::Matrix{Float64})
    F = zeros(3, N(s))
    contribs = Float64[]
    for f in s.factors
        push!(contribs, evaluate!(F, f, q, s.ctx))
    end
    U = neumaier_sum(contribs)
    return kinetic_energy(s, p) + U
end

# ---------------------------------------------------------------------------
# energy drift diagnostics
# ---------------------------------------------------------------------------

"""
Quantitative drift report.

`slope` is the ordinary least-squares slope of `E(t)` against `t`, `slope_stderr`
its naive standard error, and `predicted_drift = |slope| * T` the change the
fitted line attributes to the whole run.

The **decision statistic** is `predicted_drift / band`:

* for a bounded oscillation the fitted slope is ~0, so this ratio is ~0;
* for a genuine linear leak the deviation band *is* the leak, so the ratio is ~2.

`band = 2·max|E - E₀|` is twice the peak excursion, which for a symmetric
oscillation equals the peak-to-peak amplitude.

**Why not just `slope / stderr`.** An earlier version of this function did exactly
that, and it was wrong in a way worth recording: a symplectic run is a *deterministic*
oscillation with no stochastic noise, so the residual sum of squares is pure
roundoff and `stderr` is microscopic. A vanishing slope then still divides by a
vanishing error and produces a huge ratio, flagging a perfectly good trajectory as
drifting. The ratio is reported as `ratio` for reference, but the verdict uses
`predicted_drift` against the physically meaningful scale.
"""
struct EnergyDriftReport
    E0::Float64
    Efinal::Float64
    max_abs_dev::Float64
    band::Float64
    slope::Float64
    slope_stderr::Float64
    ratio::Float64
    predicted_drift::Float64
    drift_fraction::Float64
    n::Int
    bounded::Bool
end

"Verdict threshold on `predicted_drift / band`. 2.0 is a pure linear leak."
const DRIFT_FRACTION_MAX = 0.5

function energy_drift_report(t::AbstractVector{Float64}, E::AbstractVector{Float64};
                             drift_fraction_max::Float64 = DRIFT_FRACTION_MAX)
    n = length(E)
    n >= 3 || error("need at least 3 samples for a drift estimate")
    E0 = E[1]
    Efinal = E[end]
    Emax = maximum(abs, E .- E0)
    tb = sum(t) / n
    Eb = sum(E) / n
    sxx = sum((t[i] - tb)^2 for i in 1:n)
    sxy = sum((t[i] - tb) * (E[i] - Eb) for i in 1:n)
    slope = sxx == 0 ? 0.0 : sxy / sxx
    resid = [E[i] - (Eb + slope * (t[i] - tb)) for i in 1:n]
    s2 = sum(abs2, resid) / max(n - 2, 1)
    se = sxx == 0 ? Inf : sqrt(s2 / sxx)
    ratio = se == 0 ? (slope == 0 ? 0.0 : Inf) : slope / se
    T = t[end] - t[1]
    predicted = abs(slope) * T
    band = 2 * Emax
    frac = band == 0 ? 0.0 : predicted / band
    return EnergyDriftReport(E0, Efinal, Emax, band, slope, se, ratio,
                             predicted, frac, n, frac <= drift_fraction_max)
end

energy_drift_report(recs::Vector{TrajectoryRecord}; kwargs...) =
    energy_drift_report([r.t for r in recs], [r.E for r in recs]; kwargs...)

energy_drift_report(recs::Vector{TrajectoryRecord}, es::EnergySeries; kwargs...) =
    energy_drift_report([r.t for r in recs], es.compensated; kwargs...)

# ---------------------------------------------------------------------------
# step-doubling defect (heuristic estimate, not a certified bound)
# ---------------------------------------------------------------------------

"""
Symmetric step-doubling defect estimate.

From the same `(q, p)`, integrate one step of `dt` and two steps of `dt/2`; the
discrepancy in the resulting state is the local step-doubling defect.

**This is a heuristic estimate of local truncation error.** It is not a
certified bound on the global trajectory error, and it is not a Lyapunov
exponent. It is used only to compare candidate fixed `dt` values offline. For a
second-order integrator the defect scales as `dt³` per step, which the test suite
checks by fitting the exponent.
"""
function step_doubling_defect(st::State, s::HarmonicSystem, dt::Float64)
    check_force_model(s)
    # one step of dt
    a = copy(st)
    velocity_verlet_step!(a, s, dt)
    # two steps of dt/2
    b = copy(st)
    velocity_verlet_step!(b, s, 0.5 * dt)
    velocity_verlet_step!(b, s, 0.5 * dt)
    dq = maximum(abs, a.q .- b.q)
    dp = maximum(abs, a.p .- b.p)
    return max(dq, dp)
end

# ---------------------------------------------------------------------------
# offline step selection (validation run, NOT an in-run adaptive controller)
# ---------------------------------------------------------------------------

struct StepSelection
    dt::Float64
    defect::Float64
    drift_fraction::Float64
    accepted::Bool
    table::Vector{NamedTuple{(:dt, :defect, :drift_fraction, :bounded),Tuple{Float64,Float64,Float64,Bool}}}
end

"""
Choose a single fixed `dt` for production by running short trial trajectories at
a ladder of candidate values.

Criteria (both must hold):
* the step-doubling defect is below `max_defect`;
* the fitted drift over the run is below `drift_fraction_max` of the energy
  oscillation band (default 0.5; a pure linear leak scores 2.0).

Returns the **largest** `dt` passing both, together with the full table so the
choice can be audited. This is deliberately an *offline* procedure: the returned
`dt` is a constant, so the production integrator stays symplectic and reversible.
"""
function select_dt(st::State, s::HarmonicSystem, dts::AbstractVector{Float64};
                   nsteps::Int = 2000, max_defect::Float64 = 1e-6,
                   drift_fraction_max::Float64 = DRIFT_FRACTION_MAX)
    check_force_model(s)
    rows = NamedTuple{(:dt, :defect, :drift_fraction, :bounded),Tuple{Float64,Float64,Float64,Bool}}[]
    best = nothing
    for dt in sort(collect(dts))
        test = copy(st)
        recs, es = run_trajectory(test, s, dt, nsteps)
        rep = energy_drift_report(recs, es; drift_fraction_max = drift_fraction_max)
        defect = step_doubling_defect(st, s, dt)
        ok = defect <= max_defect && rep.bounded
        push!(rows, (dt = dt, defect = defect, drift_fraction = rep.drift_fraction,
                     bounded = rep.bounded))
        if ok
            best = (dt = dt, defect = defect, frac = rep.drift_fraction)
        end
    end
    if best === nothing
        return StepSelection(NaN, NaN, NaN, false, rows)
    end
    return StepSelection(best.dt, best.defect, best.frac, true, rows)
end

# ---------------------------------------------------------------------------
# error budget: three kinds, never conflated
# ---------------------------------------------------------------------------

"""
Named error budget. `kind` is `:exact`, `:estimate`, or `:proved_bound` — the
three are different epistemic categories and are stored per field rather than
blended into one number.

Nothing in this struct is populated by a certificate except where a Lean theorem
supplies one; the field comments say which.
"""
struct BudgetEntry
    value::Float64
    kind::Symbol
    note::String
end

struct ErrorBudget
    model::BudgetEntry       # force-field / functional-form error
    spatial::BudgetEntry     # cutoff, neighbour skin, neighbour staleness
    time::BudgetEntry        # integrator truncation
    constraint::BudgetEntry  # projection error; currently none supported
    arithmetic::BudgetEntry  # floating-point accumulation
end

"""
Build a budget from measurements. This is honest bookkeeping, not a certificate:
`model` is `:estimate` because no certified physical bound exists (FRAMEWORK.md
§6 'Bounded error' is a research target), `spatial` is `:estimate`, and `time`
is `:estimate` from step doubling. `arithmetic` is `:estimate` unless the
compensated sum is provably exact.
"""
function ErrorBudget(; model = NaN, spatial = NaN, time = NaN,
                     constraint = NaN, arithmetic = NaN)
    BudgetEntry(model, :estimate, "no certified physical bound available") |>
        (e -> ErrorBudget(e,
            BudgetEntry(spatial, :estimate, "cutoff + neighbour skin/staleness"),
            BudgetEntry(time, :estimate, "step-doubling defect (local heuristic)"),
            BudgetEntry(constraint, :exact, "no constraints in this backend"),
            BudgetEntry(arithmetic, :estimate, "compensated (Neumaier) summation")))
end

# ---------------------------------------------------------------------------
# constraints
# ---------------------------------------------------------------------------

"""Raised when iterative mass-metric constraint projection misses its tolerance."""
struct ConstraintConvergenceError <: Exception
    stage::Symbol
    residual::Float64
    iterations::Int
    tolerance::Float64
end
struct UnsupportedConstraintTopology <: Exception
    reason::String
end
Base.showerror(io::IO, e::UnsupportedConstraintTopology) =
    print(io, "unsupported RATTLE constraint topology: ", e.reason)
export UnsupportedConstraintTopology
Base.showerror(io::IO, e::ConstraintConvergenceError) =
    print(io, "RATTLE ", e.stage, " projection did not converge after ", e.iterations,
          " iterations (residual ", e.residual, ", tolerance ", e.tolerance, ")")

constraints_supported() = true

"""
Accept either the legacy disjoint-pair form or a `Constraints.CoupledConstraintSet`.

The legacy vector-of-triples still goes through `_validated_constraints`, which
refuses shared atoms. A `CoupledConstraintSet` bypasses that refusal because the
coupled Newton solver in `Constraints` handles shared-atom components (including
rigid water). Returning the set keeps the BAOAB path able to use the general
solver while leaving existing callers untouched.
"""
function _validated_coupled_constraints(n::Int, constraints, tol, max_iterations)
    constraints isa Constraints.CoupledConstraintSet && return constraints
    throw(ArgumentError("BAOAB constraints must be a Constraints.CoupledConstraintSet; " *
                        "build one with CoupledConstraintSet(natoms, [(i,j,d),...])"))
end

function _validated_constraints(s::HarmonicSystem, constraints, tol, max_iterations)
    isfinite(tol) && tol > 0 || throw(ArgumentError("constraint tolerance must be positive and finite"))
    max_iterations > 0 || throw(ArgumentError("constraint max_iterations must be positive"))
    out = Tuple{Int,Int,Float64}[]
    for c in constraints
        length(c) == 3 || throw(ArgumentError("each distance constraint must be (i, j, distance)"))
        i, j, d = Int(c[1]), Int(c[2]), Float64(c[3])
        1 <= i <= N(s) && 1 <= j <= N(s) && i != j || throw(ArgumentError("constraint atom indices are invalid"))
        isfinite(d) && d > 0 || throw(ArgumentError("constraint distance must be positive and finite"))
        push!(out, (i, j, d))
    end
    isempty(out) && throw(ArgumentError("at least one distance constraint is required"))
    seen = Set{Int}()
    for (i, j, _) in out
        (i in seen || j in seen) && throw(UnsupportedConstraintTopology(
            "only disjoint independent distance pairs are supported; shared-atom/coupled networks must be refused"))
        push!(seen, i); push!(seen, j)
    end
    return out
end

@inline _constraint_residual(q, c) = abs(sum(abs2, view(q, :, c[2]) .- view(q, :, c[1])) - c[3]^2)
@inline _tangent_residual(q, p, m, c) = abs(dot(view(q, :, c[2]) .- view(q, :, c[1]),
                                                view(p, :, c[2]) ./ m[c[2]] .- view(p, :, c[1]) ./ m[c[1]]))

"""
    rattle_residuals(st::State, s::HarmonicSystem, constraints) -> (pos_residual, tan_residual)

Return maximum squared-distance residual and maximum velocity-tangent residual
for a disjoint-pair RATTLE constraint list.
"""
function rattle_residuals(st::State, s::HarmonicSystem, constraints)
    if constraints isa Constraints.CoupledConstraintSet
        pos_res = maximum(abs, Constraints.constraint_residuals(constraints, st.q))
        tan_res = maximum(abs, Constraints.tangent_residuals(constraints, st.q, st.p, s.m))
        return pos_res, tan_res
    end
    cs = _validated_constraints(s, constraints, 1e-12, 1)
    pos_res = maximum(_constraint_residual(st.q, c) for c in cs)
    tan_res = maximum(_tangent_residual(st.q, st.p, s.m, c) for c in cs)
    return pos_res, tan_res
end

function _is_all_water_triangles(cs::Constraints.CoupledConstraintSet, m::AbstractVector{Float64})
    isempty(cs.components) && return false
    for comp in cs.components
        length(comp.atoms) == 3 && length(comp.edges) == 3 || return false
        a1, a2, a3 = comp.atoms[1], comp.atoms[2], comp.atoms[3]
        m1, m2, m3 = m[a1], m[a2], m[a3]
        if m1 > m2 && m1 > m3
            abs(m2 - m3) <= 1e-12 * max(m2, m3) || return false
        elseif m2 > m1 && m2 > m3
            abs(m1 - m3) <= 1e-12 * max(m1, m3) || return false
        elseif m3 > m1 && m3 > m2
            abs(m1 - m2) <= 1e-12 * max(m1, m2) || return false
        else
            return false
        end
    end
    return true
end

function _project_positions!(q, s, constraints::Constraints.CoupledConstraintSet, tol, max_iterations;
                             reference = nothing)
    all(isfinite, q) || throw(ArgumentError("positions must be finite"))
    if reference !== nothing
        all(isfinite, reference) || throw(ArgumentError("reference positions must be finite"))
        if !Base.mightalias(reference, q) && _is_all_water_triangles(constraints, s.m)
            # Try analytic SETTLE when reference already satisfies constraints; fallback to Newton if reference was unconstrained
            ref_res = maximum(abs, Constraints.constraint_residuals(constraints, reference))
            if ref_res <= max(1e-6, 100 * tol)
                return Settle.settle_positions!(reference, q, s.m, constraints; box = s.ctx.box)
            end
        end
    end
    res = Constraints.solve_positions!(q, s.m, constraints; tolerance = tol, max_iterations = max_iterations)
    res.converged || throw(ConstraintConvergenceError(:position, res.position_residual, res.iterations, tol))
    return res.position_residual
end

function _project_positions!(q, s, constraints, tol, max_iterations; reference = copy(q))
    cs = _validated_constraints(s, constraints, tol, max_iterations)
    all(isfinite, q) || throw(ArgumentError("positions must be finite"))
    all(isfinite, reference) || throw(ArgumentError("reference positions must be finite"))
    q0 = copy(q)
    residual = Inf
    try
        for iteration in 1:max_iterations
            residual = 0.0
            for (i, j, d) in cs
                r = view(q, :, j) .- view(q, :, i)
                r0 = view(reference, :, j) .- view(reference, :, i)
                dot(r0, r0) > 0 || throw(ArgumentError("coincident atoms cannot satisfy a distance constraint"))
                c = dot(r, r) - d^2
                residual = max(residual, abs(c))
                abs(c) <= tol && continue
                denom = 2 * (1 / s.m[i] + 1 / s.m[j]) * dot(r, r0)
                (isfinite(denom) && abs(denom) > eps(Float64)) ||
                    throw(ArgumentError("singular SHAKE position correction"))
                λ = -c / denom
                @views q[:, i] .-= λ / s.m[i] .* r0
                @views q[:, j] .+= λ / s.m[j] .* r0
            end
            residual = maximum(_constraint_residual(q, c) for c in cs)
            isfinite(residual) || throw(ConstraintConvergenceError(:position, residual, iteration, tol))
            residual <= tol && return residual
        end
        throw(ConstraintConvergenceError(:position, residual, max_iterations, tol))
    catch
        copyto!(q, q0)
        rethrow()
    end
end

function _project_velocities!(q, p, s, constraints::Constraints.CoupledConstraintSet, tol, max_iterations)
    all(isfinite, q) || throw(ArgumentError("positions must be finite"))
    all(isfinite, p) || throw(ArgumentError("momenta must be finite"))
    if _is_all_water_triangles(constraints, s.m)
        return Settle.project_momenta!(q, p, s.m, constraints; box = s.ctx.box)
    end
    res = Constraints.solve_velocities!(q, p, s.m, constraints; tolerance = tol, max_iterations = max_iterations)
    res.converged || throw(ConstraintConvergenceError(:velocity, res.tangent_residual, res.iterations, tol))
    return res.tangent_residual
end

function _project_velocities!(q, p, s, constraints, tol, max_iterations)
    cs = _validated_constraints(s, constraints, tol, max_iterations)
    all(isfinite, q) || throw(ArgumentError("positions must be finite"))
    all(isfinite, p) || throw(ArgumentError("momenta must be finite"))
    p0 = copy(p)
    residual = Inf
    try
        for iteration in 1:max_iterations
            residual = 0.0
            for (i, j, d) in cs
                r = view(q, :, j) .- view(q, :, i)
                r2 = sum(abs2, r)
                (isfinite(r2) && r2 > eps(Float64)^2) ||
                    throw(ArgumentError("singular RATTLE velocity projection (coincident atoms $i, $j)"))
                vrel = view(p, :, j) ./ s.m[j] .- view(p, :, i) ./ s.m[i]
                dotrv = dot(r, vrel)
                isfinite(dotrv) || throw(ConstraintConvergenceError(:velocity, NaN, iteration, tol))
                residual = max(residual, abs(dotrv))
                abs(dotrv) <= tol && continue
                λ = -dotrv / (2 * (1 / s.m[i] + 1 / s.m[j]) * r2)
                @views p[:, i] .-= 2λ .* r
                @views p[:, j] .+= 2λ .* r
            end
            residual = maximum(_tangent_residual(q, p, s.m, c) for c in cs)
            isfinite(residual) || throw(ConstraintConvergenceError(:velocity, residual, iteration, tol))
            residual <= tol && return residual
        end
        throw(ConstraintConvergenceError(:velocity, residual, max_iterations, tol))
    catch
        copyto!(p, p0)
        rethrow()
    end
end

"Project positions onto pair distances and momenta onto the mass-metric tangent space."
function rattle_project!(st::State, s::HarmonicSystem, constraints;
                         tolerance::Float64 = 1e-12, max_iterations::Int = 100)
    q0 = copy(st.q)
    p0 = copy(st.p)
    try
        _project_positions!(st.q, s, constraints, tolerance, max_iterations)
        _project_velocities!(st.q, st.p, s, constraints, tolerance, max_iterations)
        st.force_valid = false
        return st
    catch
        copyto!(st.q, q0)
        copyto!(st.p, p0)
        rethrow()
    end
end

"""
    rattle_project_with_status!(st, s, constraints; tolerance=1e-12, max_iterations=100)
        -> (converged, pos_residual, tan_residual, reason)

Audited disjoint-pair RATTLE projection returning explicit residuals and
convergence status. Restores `st` and returns `converged=false` on convergence failure.
"""
function rattle_project_with_status!(st::State, s::HarmonicSystem, constraints;
                                     tolerance::Float64 = 1e-12, max_iterations::Int = 100)
    q0 = copy(st.q)
    p0 = copy(st.p)
    try
        rp = _project_positions!(st.q, s, constraints, tolerance, max_iterations)
        rt = _project_velocities!(st.q, st.p, s, constraints, tolerance, max_iterations)
        st.force_valid = false
        return (converged = true, position_residual = rp, tangent_residual = rt, reason = "")
    catch err
        copyto!(st.q, q0)
        copyto!(st.p, p0)
        if err isa ConstraintConvergenceError || err isa UnsupportedConstraintTopology
            rp, rt = try
                rattle_residuals(st, s, constraints)
            catch
                (NaN, NaN)
            end
            return (converged = false, position_residual = rp, tangent_residual = rt,
                    reason = sprint(showerror, err))
        end
        rethrow()
    end
end

# ---------------------------------------------------------------------------
# Langevin: a separate stochastic layer, not another force
# ---------------------------------------------------------------------------

"""
BAOAB Langevin splitting (stages from `integrator_spec(:baoab)` or `:constrained_baoab`).

B: half kick (`:reversible`), A: half drift (`:reversible`), O: Ornstein–Uhlenbeck
momentum update (`:stochastic`), A: half drift (`:reversible`), optional constraint
projection (`:reversible`), force evaluation (`:reversible`), B: half kick (`:reversible`).
The `O` step is exact for the momentum OU process, so the scheme has a well-defined
canonical invariant measure at temperature `T`; naive Euler–Maruyama noise injected
into a Verlet update does not.

`rng` is passed explicitly so runs are reproducible.
"""
function langevin_baoab_step!(q::Matrix{Float64}, p::Matrix{Float64},
                              F::Matrix{Float64}, s::HarmonicSystem, dt::Float64,
                              gamma::Float64, beta::Float64, rng::AbstractRNG;
                              constraints = nothing,
                              constraint_tolerance::Float64 = 1e-12,
                              constraint_max_iterations::Int = 100,
                              stage_hook = nothing)
    check_force_model(s)
    (isfinite(dt) && dt > 0.0) || throw(ArgumentError("dt must be positive and finite"))
    (isfinite(gamma) && gamma >= 0.0) || throw(ArgumentError("gamma must be non-negative and finite"))
    (isfinite(beta) && beta > 0.0) || throw(ArgumentError("beta must be positive and finite"))
    n = N(s)
    cs = constraints === nothing ? nothing :
         _validated_coupled_constraints(n, constraints, constraint_tolerance,
                                        constraint_max_iterations)
    # B: half kick
    _stage_half_kick!(p, F, dt, n, stage_hook)
    # A: half drift
    _stage_drift!(q, p, s.m, 0.5 * dt, n, stage_hook)
    # O: exact OU update of momentum (stochastic stage)
    _stage_thermostat_ou!(p, s.m, dt, gamma, beta, n, rng, stage_hook)
    # A: half drift
    _stage_drift!(q, p, s.m, 0.5 * dt, n, stage_hook)
    # Constraint projection (position + tangent), then the new force.
    #
    # The O step injects independent noise into every momentum component, which
    # generally violates the constraint tangent condition. Projecting the momenta
    # onto the tangent space restores it, and the position projection restores
    # the constraint itself. Both are exact projections in the mass metric, so
    # the constrained invariant measure is preserved; a partial or one-sided
    # correction would bias the ensemble.
    if cs !== nothing
        _emit_stage(stage_hook, :position_constraints)
        Constraints.project_positions!(q, s.m, cs; tolerance = constraint_tolerance,
                                       max_iterations = constraint_max_iterations)
        _emit_stage(stage_hook, :velocity_constraints)
        Constraints.project_velocities!(q, p, s.m, cs; tolerance = constraint_tolerance,
                                        max_iterations = constraint_max_iterations)
    end
    # force at new position
    U = _stage_force_eval!(q, F, s, stage_hook)
    # B: half kick
    _stage_half_kick!(p, F, dt, n, stage_hook)
    # The final kick can reintroduce a tangent violation (force components along
    # the constraints); project once more so the state handed back is consistent.
    if cs !== nothing
        _emit_stage(stage_hook, :velocity_constraints)
        Constraints.project_velocities!(q, p, s.m, cs; tolerance = constraint_tolerance,
                                        max_iterations = constraint_max_iterations)
    end
    return U
end

end # module Dynamics
