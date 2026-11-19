"""
Symplectic Balanced-Ternary Fixed-Point Time-Reversible Integrator & Adjoint Engine.

Standard IEEE-754 `Float64` Velocity Verlet fails exact time-reversibility
because floating-point addition is non-associative: `(q + v*dt) - v*dt != q`.

In `SymplecticTernary`, we encode phase space `(Q, P)` as `Int128` fixed-point
balanced-ternary lattice coordinates:
- `q_scale = 3.0^26 ≈ 2.541865828329e12` trits/nm (exact sub-femtometer resolution)
- `p_scale = 3.0^24 ≈ 2.82429536481e11` trits/(nm·amu/ps)

Because each Kick and Drift is a pure integer shear on ℤ^{6N}:
    P += ΔP(Q)
    Q += ΔQ(P)
using symmetric sign-preserving rounding:
    round_symmetric(x::Float64)::Int128 = x >= 0.0 ? round(Int128, x, RoundNearestTiesAway) : -round(Int128, -x, RoundNearestTiesAway)

For any deterministic force field `F(q)`, integer subtraction `-=` undoes `+=`
bit-for-bit identically:
    Φ_{-Δt}(Φ_{+Δt}(Q, P)) ≡ (Q, P)
with ZERO bit error across arbitrary step counts without intermediate frame storage (O(1) memory).
"""
module SymplecticTernary

using ..Types
using ..Factors
using ..Dynamics
using LinearAlgebra

# Provide alias Types.SystemState = Types.System if not already defined
if !isdefined(Types, :SystemState)
    @eval Types const SystemState = System
end
const SystemState = Types.SystemState

export SymplecticTernaryState,
       to_system_state,
       to_dynamics_state,
       round_symmetric,
       symplectic_ternary_step!,
       simulate_symplectic_ternary!,
       reversible_trajectory_replay_accum,
       kinetic_energy,
       potential_energy,
       total_energy,
       SystemState

# ---------------------------------------------------------------------------
# Symmetric rounding on balanced ternary lattice
# ---------------------------------------------------------------------------

"""
    round_symmetric(x::Float64)::Int128

Symmetric odd rounding (`RoundNearestTiesAway`) ensuring `round_symmetric(-x) == -round_symmetric(x)`
identically for all `x`, including exact `.5` ties.
"""
@inline function round_symmetric(x::Float64)::Int128
    return x >= 0.0 ? round(Int128, x, RoundNearestTiesAway) : -round(Int128, -x, RoundNearestTiesAway)
end
@inline round_symmetric(x::Real)::Int128 = round_symmetric(Float64(x))

# ---------------------------------------------------------------------------
# Symplectic ternary state
# ---------------------------------------------------------------------------

"""
    mutable struct SymplecticTernaryState

Fixed-point discrete phase space state in balanced-ternary representation.
Coordinates `Q` and momenta `P` are 3 x N `Int128` lattice positions.
"""
mutable struct SymplecticTernaryState
    Q::Matrix{Int128}
    P::Matrix{Int128}
    q_scale::Float64
    p_scale::Float64
    masses::Vector{Float64}
    box::Types.AbstractBox
    step_index::Int
    template::Union{Nothing, Types.System}
    _q_buf::Matrix{Float64}
    _F_buf::Matrix{Float64}

    function SymplecticTernaryState(Q::AbstractMatrix{<:Integer},
                                   P::AbstractMatrix{<:Integer},
                                   q_scale::Real,
                                   p_scale::Real,
                                   masses::AbstractVector{<:Real},
                                   box::Types.AbstractBox = Types.NoBox(),
                                   step_index::Integer = 0,
                                   template::Union{Nothing, Types.System} = nothing)
        size(Q) == size(P) || throw(ArgumentError("Q and P shapes must match, got $(size(Q)) vs $(size(P))"))
        size(Q, 1) == 3 || throw(ArgumentError("Q must be 3 x N, got $(size(Q))"))
        length(masses) == size(Q, 2) || throw(ArgumentError("masses length must match atom count, got $(length(masses)) vs $(size(Q, 2))"))
        q_scale > 0 || throw(ArgumentError("q_scale must be positive"))
        p_scale > 0 || throw(ArgumentError("p_scale must be positive"))
        all(m -> m > 0 && isfinite(m), masses) || throw(ArgumentError("masses must be positive and finite"))
        Q_mat = Matrix{Int128}(Q)
        P_mat = Matrix{Int128}(P)
        m_vec = Vector{Float64}(masses)
        _q = zeros(Float64, size(Q_mat))
        _F = zeros(Float64, size(Q_mat))
        new(Q_mat, P_mat, Float64(q_scale), Float64(p_scale), m_vec, box, Int(step_index), template, _q, _F)
    end
end

"""
    SymplecticTernaryState(sys::Types.SystemState; q_scale=3.0^26, p_scale=3.0^24)

Construct a discrete balanced-ternary phase state from a typed `SystemState`.
"""
function SymplecticTernaryState(sys::Types.System;
                               q_scale::Float64 = 3.0^26,
                               p_scale::Float64 = 3.0^24)
    Q = round_symmetric.(sys.atoms.q .* q_scale)
    P = round_symmetric.(sys.atoms.p .* p_scale)
    return SymplecticTernaryState(Q, P, q_scale, p_scale, copy(sys.atoms.m), sys.box, 0, sys)
end

function SymplecticTernaryState(q::AbstractMatrix{<:Real},
                               p::AbstractMatrix{<:Real},
                               masses::AbstractVector{<:Real},
                               box::Types.AbstractBox = Types.NoBox();
                               q_scale::Float64 = 3.0^26,
                               p_scale::Float64 = 3.0^24,
                               step_index::Integer = 0)
    Q = round_symmetric.(Float64.(q) .* q_scale)
    P = round_symmetric.(Float64.(p) .* p_scale)
    return SymplecticTernaryState(Q, P, q_scale, p_scale, masses, box, step_index, nothing)
end

function SymplecticTernaryState(sys::Dynamics.State,
                               masses::AbstractVector{<:Real},
                               box::Types.AbstractBox = Types.NoBox();
                               q_scale::Float64 = 3.0^26,
                               p_scale::Float64 = 3.0^24,
                               step_index::Integer = 0)
    return SymplecticTernaryState(sys.q, sys.p, masses, box;
                                 q_scale = q_scale, p_scale = p_scale, step_index = step_index)
end

function Base.copy(st::SymplecticTernaryState)
    return SymplecticTernaryState(
        copy(st.Q), copy(st.P),
        st.q_scale, st.p_scale,
        copy(st.masses), st.box,
        st.step_index, st.template
    )
end

"""
    to_system_state(st::SymplecticTernaryState)::Types.SystemState

Convert discrete ternary lattice state back into a typed continuous `SystemState`.
"""
function to_system_state(st::SymplecticTernaryState)::Types.System
    q = Float64.(st.Q) ./ st.q_scale
    p = Float64.(st.P) ./ st.p_scale
    N = size(q, 2)
    if st.template !== nothing
        atoms = Types.AtomSet(q, p, copy(st.masses), copy(st.template.atoms.z), copy(st.template.atoms.names))
        return Types.System(atoms, st.box, st.template.topology;
                            units = st.template.units,
                            temperature = st.template.temperature)
    else
        atoms = Types.AtomSet(q, p, copy(st.masses), zeros(N), fill(:X, N))
        return Types.System(atoms, st.box, Types.ChemicalTopology();
                            units = Types.REDUCED,
                            temperature = 0.0)
    end
end

"""
    to_dynamics_state(st::SymplecticTernaryState)::Dynamics.State

Convert discrete ternary state to a `Dynamics.State`.
"""
function to_dynamics_state(st::SymplecticTernaryState)::Dynamics.State
    q = Float64.(st.Q) ./ st.q_scale
    p = Float64.(st.P) ./ st.p_scale
    return Dynamics.State(q, p)
end

# ---------------------------------------------------------------------------
# Force dispatcher helper
# ---------------------------------------------------------------------------

function _eval_forces!(F_buf::Matrix{Float64}, force_source, q::Matrix{Float64}, st::SymplecticTernaryState)
    fill!(F_buf, 0.0)
    if force_source isa Dynamics.HarmonicSystem
        return Dynamics.forces!(force_source, q, F_buf)
    elseif force_source isa AbstractVector{<:Factors.AbstractFactor} ||
           (force_source isa AbstractVector && !isempty(force_source) && first(force_source) isa Factors.AbstractFactor)
        units = st.template !== nothing ? st.template.units : Types.REDUCED
        ctx = Factors.Context(st.box, units, size(q, 2))
        U, _ = Factors.total_energy_forces(force_source, q, ctx; F = F_buf)
        return U
    elseif force_source isa Factors.AbstractFactor
        units = st.template !== nothing ? st.template.units : Types.REDUCED
        ctx = Factors.Context(st.box, units, size(q, 2))
        U = Factors.evaluate!(F_buf, force_source, q, ctx)
        return U
    elseif force_source isa Function
        if applicable(force_source, q, F_buf)
            res = force_source(q, F_buf)
            return res isa Real ? Float64(res) : 0.0
        else
            res = force_source(q)
            if res isa Tuple && length(res) == 2
                U, F_ret = res
                copyto!(F_buf, F_ret)
                return Float64(U)
            elseif res isa AbstractMatrix
                copyto!(F_buf, res)
                return 0.0
            else
                error("callable force source must take (q, F) or return (U, F) or F")
            end
        end
    else
        error("unsupported force source type: $(typeof(force_source))")
    end
end

# ---------------------------------------------------------------------------
# Energy diagnostics
# ---------------------------------------------------------------------------

"""
    kinetic_energy(st::SymplecticTernaryState)::Float64

Kinetic energy T = 0.5 Σ (P_j / p_scale)^2 / m_j.
"""
function kinetic_energy(st::SymplecticTernaryState)::Float64
    N = length(st.masses)
    ps = st.p_scale
    E_kin = 0.0
    @inbounds for j in 1:N
        inv_2m = 0.5 / st.masses[j]
        for d in 1:3
            p_val = Float64(st.P[d, j]) / ps
            E_kin += inv_2m * (p_val * p_val)
        end
    end
    return E_kin
end

"""
    potential_energy(st::SymplecticTernaryState, factors)::Float64

Potential energy U(Q / q_scale).
"""
function potential_energy(st::SymplecticTernaryState, factors)::Float64
    N = length(st.masses)
    for j in 1:N, d in 1:3
        st._q_buf[d, j] = Float64(st.Q[d, j]) / st.q_scale
    end
    return _eval_forces!(st._F_buf, factors, st._q_buf, st)
end

"""
    total_energy(st::SymplecticTernaryState, factors)::Float64

Total energy H = T(P) + U(Q).
"""
function total_energy(st::SymplecticTernaryState, factors)::Float64
    return kinetic_energy(st) + potential_energy(st, factors)
end

# ---------------------------------------------------------------------------
# Symplectic ternary time step (Forward and Exact Reverse)
# ---------------------------------------------------------------------------

"""
    symplectic_ternary_step!(st::SymplecticTernaryState, factors, dt::Float64; direction::Int=+1)

Evolve `st` by one time step `dt`.
When `direction == +1` (or `dt > 0`), executes Forward KDK on the integer lattice:
  1. P += round_symmetric(0.5 * dt * p_scale * F(Q_n / q_scale))
  2. Q += round_symmetric((dt * q_scale / p_scale) * (P_half / m))
  3. P += round_symmetric(0.5 * dt * p_scale * F(Q_{n+1} / q_scale))

When `direction == -1` (or `dt < 0`), executes Exact Reverse KDK:
  1. P -= round_symmetric(0.5 * dt * p_scale * F(Q_{n+1} / q_scale))
  2. Q -= round_symmetric((dt * q_scale / p_scale) * (P_half / m))
  3. P -= round_symmetric(0.5 * dt * p_scale * F(Q_n / q_scale))

Because integer subtraction `-=` exactly inverts `+=`, the reverse step is bit-for-bit
identical to reversing time, producing ZERO bit error regardless of step count.
"""
function symplectic_ternary_step!(st::SymplecticTernaryState, factors, dt::Float64; direction::Int = +1)
    (isfinite(dt) && dt != 0.0) || throw(ArgumentError("dt must be finite and non-zero"))
    eff_dir = dt < 0 ? -direction : direction
    (eff_dir == 1 || eff_dir == -1) || throw(ArgumentError("direction must be +1 or -1"))
    step_dt = abs(dt)

    N = length(st.masses)
    q_scale = st.q_scale
    p_scale = st.p_scale
    half_dt_p = 0.5 * step_dt * p_scale
    drift_scale = step_dt * q_scale / p_scale

    if eff_dir == 1
        # Forward KDK:
        # 1. Kick half step at Q_n
        for j in 1:N, d in 1:3
            st._q_buf[d, j] = Float64(st.Q[d, j]) / q_scale
        end
        _eval_forces!(st._F_buf, factors, st._q_buf, st)
        for j in 1:N, d in 1:3
            ΔP1 = round_symmetric(half_dt_p * st._F_buf[d, j])
            st.P[d, j] += ΔP1
        end

        # 2. Drift full step at P_half
        for j in 1:N
            inv_m = 1.0 / st.masses[j]
            for d in 1:3
                v_drift = drift_scale * (Float64(st.P[d, j]) * inv_m)
                ΔQ = round_symmetric(v_drift)
                st.Q[d, j] += ΔQ
            end
        end

        # 3. Kick half step at Q_{n+1}
        for j in 1:N, d in 1:3
            st._q_buf[d, j] = Float64(st.Q[d, j]) / q_scale
        end
        _eval_forces!(st._F_buf, factors, st._q_buf, st)
        for j in 1:N, d in 1:3
            ΔP2 = round_symmetric(half_dt_p * st._F_buf[d, j])
            st.P[d, j] += ΔP2
        end
        st.step_index += 1
    else
        # Exact Reverse KDK:
        # 1. Reverse kick 2 at Q_{n+1}
        for j in 1:N, d in 1:3
            st._q_buf[d, j] = Float64(st.Q[d, j]) / q_scale
        end
        _eval_forces!(st._F_buf, factors, st._q_buf, st)
        for j in 1:N, d in 1:3
            ΔP2 = round_symmetric(half_dt_p * st._F_buf[d, j])
            st.P[d, j] -= ΔP2
        end

        # 2. Reverse drift at P_half
        for j in 1:N
            inv_m = 1.0 / st.masses[j]
            for d in 1:3
                v_drift = drift_scale * (Float64(st.P[d, j]) * inv_m)
                ΔQ = round_symmetric(v_drift)
                st.Q[d, j] -= ΔQ
            end
        end

        # 3. Reverse kick 1 at Q_n
        for j in 1:N, d in 1:3
            st._q_buf[d, j] = Float64(st.Q[d, j]) / q_scale
        end
        _eval_forces!(st._F_buf, factors, st._q_buf, st)
        for j in 1:N, d in 1:3
            ΔP1 = round_symmetric(half_dt_p * st._F_buf[d, j])
            st.P[d, j] -= ΔP1
        end
        st.step_index -= 1
    end
    return st
end

"""
    simulate_symplectic_ternary!(st::SymplecticTernaryState, factors, dt::Float64, n_steps::Int; direction::Int=+1)

Evolve `st` for `n_steps` steps using the symplectic ternary integrator.
"""
function simulate_symplectic_ternary!(st::SymplecticTernaryState, factors, dt::Float64, n_steps::Int; direction::Int = +1)
    for _ in 1:n_steps
        symplectic_ternary_step!(st, factors, dt; direction = direction)
    end
    return st
end

# ---------------------------------------------------------------------------
# Reversible trajectory replay with O(1) memory
# ---------------------------------------------------------------------------

"""
    reversible_trajectory_replay_accum(st0::SymplecticTernaryState, factors, dt::Float64, n_steps::Int, obs_fn::Function)

Run `n_steps` forward with O(1) trajectory storage (zero intermediate frames saved to DRAM/disk).
Then step `n_steps` backward (`direction = -1`), evaluating `obs_fn(step_idx, st)` on every
reconstructed frame `step_idx = n_steps-1, ..., 0`.
Finally, asserts `st.Q == st0.Q && st.P == st0.P` with 0 bit error and returns the accumulated observable.
"""
function reversible_trajectory_replay_accum(st0::SymplecticTernaryState, factors, dt::Float64, n_steps::Int, obs_fn::Function)
    st = copy(st0)
    # Forward pass: O(1) memory, no stored frames
    for _ in 1:n_steps
        symplectic_ternary_step!(st, factors, dt; direction = +1)
    end

    # Backward pass: reconstruct frames n_steps-1 down to 0
    accum = nothing
    for step_idx in (n_steps - 1):-1:0
        symplectic_ternary_step!(st, factors, dt; direction = -1)
        val = applicable(obs_fn, step_idx, st) ? obs_fn(step_idx, st) : obs_fn(st)
        if val !== nothing
            if accum === nothing
                accum = deepcopy(val)
            else
                accum += val
            end
        end
    end

    # Confirm exact bit-reversibility at st0
    if !(st.Q == st0.Q && st.P == st0.P)
        error("Exact reversibility check failed in reversible_trajectory_replay_accum: final state does not match st0!")
    end
    return accum
end

end # module SymplecticTernary
