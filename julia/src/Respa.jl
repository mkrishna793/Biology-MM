"""
Nested reversible RESPA splitting for an explicit partition of conservative factors.

`step!(state, integrator, dt; m)` applies a slow half-kick, `m` fast
velocity-Verlet substeps of size `dt/m`, then a slow half-kick. For fixed
partitions this is a symmetric composition and has global error `O(dt^2)`;
it can still exhibit impulse resonance at outer steps near fast-mode periods,
even when both component Verlet steps are individually stable. Each step
evaluates the fast factors `2m` times and the slow factors twice; endpoint force
values are not reused. Constraints are explicitly unsupported.

This is an opt-in module. `RespaIntegrator(system, fast_indices)` places every
other system factor in the slow group. Indices refer to `system.factors`.
"""
module Respa

using ..Factors
using ..Constraints
using ..Settle
using ..Dynamics
using ..Dynamics: State, HarmonicSystem, force_model_supported

export RespaIntegrator, step!

"Cached factor partition and reusable force accumulators for nested RESPA."
mutable struct RespaIntegrator
    system::HarmonicSystem
    fast_indices::Vector{Int}
    slow_indices::Vector{Int}
    fast_force::Matrix{Float64}
    slow_force::Matrix{Float64}
    q_ref::Matrix{Float64}
    q_pred::Matrix{Float64}
    last_energy::Float64
end

function RespaIntegrator(system::HarmonicSystem, fast_indices)
    fast = sort!(unique!(Int.(collect(fast_indices))))
    all(i -> 1 <= i <= length(system.factors), fast) ||
        throw(ArgumentError("fast factor indices must refer to system.factors"))
    slow = setdiff(collect(eachindex(system.factors)), fast)
    n = length(system.m)
    return RespaIntegrator(system, fast, slow, zeros(3, n), zeros(3, n),
                           zeros(3, n), zeros(3, n), 0.0)
end

function _group_forces!(buf, factors, indices, q, ctx)
    fill!(buf, 0.0)
    energy = 0.0
    @inbounds for i in indices
        energy += evaluate!(buf, factors[i], q, ctx)
    end
    return energy
end

@inline function _kick!(p, F, h)
    @inbounds @simd for i in eachindex(p, F)
        p[i] += h * F[i]
    end
end

@inline function _drift!(q, p, masses, h)
    @inbounds for j in eachindex(masses), d in 1:3
        q[d, j] += h * p[d, j] / masses[j]
    end
end

function _apply_position_constraints!(q::Matrix{Float64}, p::Matrix{Float64},
                                      q_ref::Matrix{Float64}, q_pred::Matrix{Float64},
                                      s::HarmonicSystem, h::Float64,
                                      cs::Constraints.CoupledConstraintSet;
                                      tolerance::Float64 = 1e-12,
                                      max_iterations::Int = 100)
    copyto!(q_pred, q)
    # Try analytic SETTLE when every component is a 3-atom 3-edge water triangle;
    # fall back to coupled Newton projection otherwise.
    used_settle = false
    if !isempty(cs.components) && all(c -> length(c.atoms) == 3 && length(c.edges) == 3, cs.components)
        try
            Settle.settle_positions!(q_ref, q, s.m, cs; box = s.ctx.box)
            used_settle = true
        catch
            copyto!(q, q_pred)
            used_settle = false
        end
    end
    if !used_settle
        Constraints.project_positions!(q, s.m, cs; tolerance = tolerance,
                                       max_iterations = max_iterations)
    end
    @inbounds for j in eachindex(s.m), d in 1:3
        p[d, j] += s.m[j] * (q[d, j] - q_pred[d, j]) / h
    end
    return nothing
end

function _apply_position_constraints!(q::Matrix{Float64}, p::Matrix{Float64},
                                      q_ref::Matrix{Float64}, q_pred::Matrix{Float64},
                                      s::HarmonicSystem, h::Float64,
                                      cs::AbstractVector;
                                      tolerance::Float64 = 1e-12,
                                      max_iterations::Int = 100)
    copyto!(q_pred, q)
    Dynamics._project_positions!(q, s, cs, tolerance, max_iterations; reference = q_ref)
    @inbounds for j in eachindex(s.m), d in 1:3
        p[d, j] += s.m[j] * (q[d, j] - q_pred[d, j]) / h
    end
    return nothing
end

function _apply_velocity_constraints!(q::Matrix{Float64}, p::Matrix{Float64},
                                      s::HarmonicSystem,
                                      cs::Constraints.CoupledConstraintSet;
                                      tolerance::Float64 = 1e-12,
                                      max_iterations::Int = 100)
    Constraints.project_velocities!(q, p, s.m, cs; tolerance = tolerance,
                                    max_iterations = max_iterations)
    return nothing
end

function _apply_velocity_constraints!(q::Matrix{Float64}, p::Matrix{Float64},
                                      s::HarmonicSystem,
                                      cs::AbstractVector;
                                      tolerance::Float64 = 1e-12,
                                      max_iterations::Int = 100)
    Dynamics._project_velocities!(q, p, s, cs, tolerance, max_iterations)
    return nothing
end

"""
    step!(state, integrator, dt; m=1, constraints=nothing)

Advance one outer step in place. A positive integer `m` selects the number of
fast Verlet substeps. Supports `constraints = nothing`, a
`Constraints.CoupledConstraintSet`, or a disjoint `(i, j, d)` constraint vector
applied on the inner fast substeps; arbitrary symbol/unknown constraint specs
are refused with `ArgumentError`.
"""
function step!(st::State, r::RespaIntegrator, dt::Real; m::Integer = 1,
               constraints = nothing, constraint_tolerance::Float64 = 1e-12,
               constraint_max_iterations::Int = 100)
    (constraints === nothing || constraints isa Constraints.CoupledConstraintSet ||
     constraints isa AbstractVector) ||
        throw(ArgumentError("RESPA constraints are unsupported for $(typeof(constraints)); pass nothing, CoupledConstraintSet, or disjoint (i, j, d) triples"))
    dt = Float64(dt)
    (isfinite(dt) && dt != 0) || throw(ArgumentError("dt must be finite and non-zero"))
    m > 0 || throw(ArgumentError("m must be a positive integer"))
    s = r.system
    size(st.q) == (3, length(s.m)) && size(st.p) == size(st.q) ||
        throw(ArgumentError("state must have shape (3, N) matching system"))
    force_model_supported(s) || throw(ArgumentError("RESPA requires force-consistent factors"))
    if constraints isa AbstractVector
        Dynamics._validated_constraints(s, constraints, constraint_tolerance, constraint_max_iterations)
    end
    _group_forces!(r.slow_force, s.factors, r.slow_indices, st.q, s.ctx)
    _kick!(st.p, r.slow_force, 0.5dt)
    if constraints !== nothing
        _apply_velocity_constraints!(st.q, st.p, s, constraints;
                                     tolerance = constraint_tolerance,
                                     max_iterations = constraint_max_iterations)
    end
    h = dt / m
    u_fast = 0.0
    for _ in 1:m
        _group_forces!(r.fast_force, s.factors, r.fast_indices, st.q, s.ctx)
        _kick!(st.p, r.fast_force, 0.5h)
        if constraints === nothing
            _drift!(st.q, st.p, s.m, h)
        else
            copyto!(r.q_ref, st.q)
            _drift!(st.q, st.p, s.m, h)
            _apply_position_constraints!(st.q, st.p, r.q_ref, r.q_pred, s, h, constraints;
                                         tolerance = constraint_tolerance,
                                         max_iterations = constraint_max_iterations)
        end
        u_fast = _group_forces!(r.fast_force, s.factors, r.fast_indices, st.q, s.ctx)
        _kick!(st.p, r.fast_force, 0.5h)
        if constraints !== nothing
            _apply_velocity_constraints!(st.q, st.p, s, constraints;
                                         tolerance = constraint_tolerance,
                                         max_iterations = constraint_max_iterations)
        end
    end
    u_slow = _group_forces!(r.slow_force, s.factors, r.slow_indices, st.q, s.ctx)
    _kick!(st.p, r.slow_force, 0.5dt)
    if constraints !== nothing
        _apply_velocity_constraints!(st.q, st.p, s, constraints;
                                     tolerance = constraint_tolerance,
                                     max_iterations = constraint_max_iterations)
    end
    @inbounds @simd for k in eachindex(st.F, r.fast_force, r.slow_force)
        st.F[k] = r.fast_force[k] + r.slow_force[k]
    end
    st.force_valid = true
    r.last_energy = u_fast + u_slow
    return nothing
end

end # module Respa
