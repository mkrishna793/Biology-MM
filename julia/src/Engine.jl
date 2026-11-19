"""
Engine — Unified Top-Level Scientific Simulation Engine Facade for Biology MM + Trium-XBT.

Unites:
- `Types`, `Factors`, `ResolvedSystem`, `SystemContract` (validated topology & force protocol)
- `TriumXBT` (balanced-ternary carry-free arithmetic & ARM64 NEON kernels)
- `TriumGPU` (adaptive multi-backend Metal / Vulkan / SYCL GPU dispatch & hardware auto-tuner)
- `Dynamics`, `Respa`, `SymplecticTernary` (Velocity Verlet, Langevin BAOAB, r-RESPA,
  and bit-exact time-reversible `Int128` Symplectic Ternary integration)

into a single ergonomic facade:
`create_engine -> engine_info -> evaluate_step! -> run_steps!`.
"""
module Engine

using LinearAlgebra
using Random

using ..Types
using ..Factors
using ..Neighbors
using ..Dynamics
using ..Respa
import ..ResolvedSystem
const _RS = ResolvedSystem
using ..SystemContract
using ..TriumXBT
using ..TriumGPU
using ..SymplecticTernary
using ..ReactiveNano
using ..CustomPhysics
using ..NEMD
using ..EnhancedSampling

export SimulationEngine,
       create_engine,
       engine_info,
       evaluate_step!,
       run_steps!,
       verify_engine_conservation

const _VALID_ENGINE_BACKENDS = (:auto, :cpu, :gpu, :metal, :vulkan, :sycl)
const _VALID_INTEGRATORS = (:verlet, :baoab, :respa, :symplectic_ternary)

"""
    SimulationEngine

Configured molecular-mechanics execution engine binding a validated `system`,
conservative `factors`, target `backend`, auto-tuned `gpu_profile`, and `integrator`.
"""
struct SimulationEngine
    system::Union{_RS.ResolvedSystem, Types.System}
    factors::Vector{Factors.AbstractFactor}
    backend::Symbol
    gpu_profile::TriumGPU.AdaptiveGPUProfile
    integrator::Symbol
    dt_ps::Float64
    temperature::Float64
    gamma_ps_inv::Float64
    fast_indices::Vector{Int}
    respa_m::Int
    rng::AbstractRNG
end

@inline function Base.getproperty(eng::SimulationEngine, sym::Symbol)
    if sym === :active_backend
        return _resolved_active_backend(getfield(eng, :backend), getfield(eng, :gpu_profile))
    end
    return getfield(eng, sym)
end

Base.propertynames(::SimulationEngine, private::Bool = false) =
    (fieldnames(SimulationEngine)..., :active_backend)

@inline function _system_masses(sys::_RS.ResolvedSystem)::Vector{Float64}
    return sys.atoms.masses
end

@inline function _system_masses(sys::Types.System)::Vector{Float64}
    return sys.atoms.m
end

@inline function _system_box(sys::Union{_RS.ResolvedSystem, Types.System})::Types.AbstractBox
    return sys.box
end

@inline function _system_units(sys::Union{_RS.ResolvedSystem, Types.System})::Types.Units
    return sys.units
end

@inline function _system_natoms(sys::Union{_RS.ResolvedSystem, Types.System})::Int
    return length(_system_masses(sys))
end

@inline function _resolved_active_backend(requested::Symbol, profile::TriumGPU.AdaptiveGPUProfile)::Symbol
    requested === :cpu && return :cpu
    profile.has_gpu || return :cpu
    if requested === :auto || requested === :gpu
        return profile.backend
    end
    return TriumGPU.has_gpu_bridge(requested) ? requested : :cpu
end

function _default_fast_indices(factors::Vector{Factors.AbstractFactor})::Vector{Int}
    fast = Int[]
    for (idx, f) in enumerate(factors)
        if f isa Factors.HarmonicBond || f isa Factors.HarmonicAngle ||
           f isa Factors.PeriodicDihedral || f isa ReactiveNano.MorseBondFactor
            push!(fast, idx)
        end
    end
    return fast
end

"""
    create_engine(sys, factors = nothing;
                  backend::Symbol = :auto,
                  integrator::Symbol = :verlet,
                  dt_ps::Real = 0.002,
                  temperature::Union{Nothing, Real} = nothing,
                  gamma_ps_inv::Real = 1.0,
                  fast_indices = nothing,
                  respa_m::Integer = 2,
                  rng::AbstractRNG = MersenneTwister(42)) -> SimulationEngine

Probes available GPU/CPU hardware, constructs an `AdaptiveGPUProfile`, validates
the system contract and force-consistency of all factors, and returns a ready
`SimulationEngine`.
"""
function create_engine(
    sys::Union{_RS.ResolvedSystem, Types.System},
    factors = nothing;
    backend::Symbol = :auto,
    integrator::Symbol = :verlet,
    dt_ps::Real = 0.002,
    temperature::Union{Nothing, Real} = nothing,
    gamma_ps_inv::Real = 1.0,
    fast_indices = nothing,
    respa_m::Integer = 2,
    rng::AbstractRNG = MersenneTwister(42)
)::SimulationEngine
    backend in _VALID_ENGINE_BACKENDS ||
        throw(ArgumentError("unsupported backend $(repr(backend)); expected one of $_VALID_ENGINE_BACKENDS"))
    integrator in _VALID_INTEGRATORS ||
        throw(ArgumentError("unsupported integrator $(repr(integrator)); expected one of $_VALID_INTEGRATORS"))

    dt = Float64(dt_ps)
    (isfinite(dt) && dt > 0.0) || throw(ArgumentError("dt_ps must be positive and finite, got $dt_ps"))
    gamma = Float64(gamma_ps_inv)
    (isfinite(gamma) && gamma >= 0.0) || throw(ArgumentError("gamma_ps_inv must be non-negative and finite"))
    m_respa = Int(respa_m)
    m_respa >= 1 || throw(ArgumentError("respa_m must be >= 1, got $m_respa"))

    if sys isa _RS.ResolvedSystem
        _RS.validate_resolved!(sys)
        if !isempty(sys.atoms.virtual_sites)
            throw(_RS.UnsupportedCapability(
                :virtual_sites,
                "virtual sites are declared ($(length(sys.atoms.virtual_sites))) and cannot be executed directly"
            ))
        end
    end

    masses = _system_masses(sys)
    n = length(masses)
    n >= 1 || throw(ArgumentError("system must contain at least 1 atom"))
    all(m -> isfinite(m) && m > 0.0, masses) ||
        throw(ArgumentError("all atom masses must be positive and finite"))

    fac_vec = if factors === nothing
        sys isa _RS.ResolvedSystem ? copy(sys.factors) : Factors.AbstractFactor[]
    else
        collect(Factors.AbstractFactor, factors)
    end

    bad_factors = Factors.AbstractFactor[f for f in fac_vec if !Factors.force_consistent(f)]
    if !isempty(bad_factors)
        names = join(unique(string.(nameof.(typeof.(bad_factors)))), ", ")
        error("refusing to build SimulationEngine with non-force-consistent factor(s): $names")
    end

    for f in fac_vec
        if Factors.is_local(f)
            for idx in Factors.read_support(f)
                (1 <= idx <= n) ||
                    throw(ArgumentError("factor $(nameof(typeof(f))) references atom $idx out of 1:$n"))
            end
        elseif Factors.factor_support_size(f) != n
            throw(ArgumentError("nonlocal factor $(nameof(typeof(f))) support size $(Factors.factor_support_size(f)) != $n"))
        end
    end

    fast_idx = fast_indices === nothing ? _default_fast_indices(fac_vec) : sort!(unique!(Int.(collect(fast_indices))))
    all(i -> 1 <= i <= length(fac_vec), fast_idx) ||
        throw(ArgumentError("fast_indices must refer to valid indices in 1:$(length(fac_vec))"))

    temp = if temperature !== nothing
        Float64(temperature)
    elseif sys isa Types.System
        sys.temperature > 0.0 ? sys.temperature : 300.0
    else
        300.0
    end
    (isfinite(temp) && temp > 0.0) || throw(ArgumentError("temperature must be positive and finite"))

    probe_target = (backend === :auto || backend === :gpu) ? :auto : backend
    profile = TriumGPU.probe_gpu_device(; backend = probe_target)

    return SimulationEngine(
        sys, fac_vec, backend, profile, integrator, dt,
        temp, gamma, fast_idx, m_respa, rng
    )
end

"""
    engine_info(eng::SimulationEngine) -> NamedTuple

Returns a diagnostic summary of the configured engine, active hardware backend,
GPU device profile, SIMD width, unified memory flag, crossover threshold,
and active factors.
"""
function engine_info(eng::SimulationEngine)::NamedTuple
    resolved = _resolved_active_backend(eng.backend, eng.gpu_profile)
    p = eng.gpu_profile
    return (
        backend = eng.backend,
        resolved_backend = resolved,
        gpu_device = p.device_name,
        has_gpu = p.has_gpu,
        simd_width = p.simd_width,
        unified_memory = p.unified_memory,
        auto_workgroup_size = p.auto_workgroup_size,
        crossover_pairs = p.auto_gpu_crossover_pairs,
        integrator = eng.integrator,
        dt_ps = eng.dt_ps,
        n_atoms = _system_natoms(eng.system),
        n_factors = length(eng.factors),
        active_factors = String[string(nameof(typeof(f))) for f in eng.factors],
    )
end

function _evaluate_factor_subset!(
    eng::SimulationEngine,
    F::AbstractMatrix{Float64},
    q::AbstractMatrix{Float64},
    box::Types.AbstractBox,
    indices;
    box_deriv::Union{Nothing, AbstractMatrix{Float64}, AbstractVector{Float64}} = nothing
)::Float64
    n = _system_natoms(eng.system)
    size(q) == (3, n) || throw(ArgumentError("q must have shape (3, $n), got $(size(q))"))
    size(F) == (3, n) || throw(ArgumentError("F must have shape (3, $n), got $(size(F))"))

    fill!(F, 0.0)
    if box_deriv !== nothing
        if box_deriv isa AbstractMatrix
            size(box_deriv) == (3, 3) || throw(DimensionMismatch("box_deriv must be 3x3"))
        elseif box_deriv isa AbstractVector
            length(box_deriv) == 3 || throw(DimensionMismatch("box_deriv vector must have length 3"))
        end
        fill!(box_deriv, 0.0)
    end

    ctx = Factors.Context(box, _system_units(eng.system), n)
    u_total = 0.0

    G_mat = box_deriv === nothing ? nothing :
            (box_deriv isa Matrix{Float64} ? box_deriv : zeros(Float64, 3, 3))

    for i in indices
        f = eng.factors[i]
        if f isa TriumXBT.TriumTiledNonbondedFactor
            u_total += TriumGPU.eval_lj_dsf_gpu!(
                F, box_deriv, q, box, f;
                backend = eng.backend, profile = eng.gpu_profile
            )
        elseif f isa TriumGPU.TriumGPUNonbondedFactor
            sub_backend = eng.backend === :auto ? f.backend : eng.backend
            u_total += TriumGPU.eval_lj_dsf_gpu!(
                F, box_deriv, q, box, f.inner;
                backend = sub_backend, profile = eng.gpu_profile
            )
        else
            u_total += Factors.evaluate!(F, f, q, ctx)
            if G_mat !== nothing
                if box_deriv isa Matrix{Float64}
                    SystemContract._accumulate_box_derivative!(G_mat, f, q, ctx, Symbol("factor_", i))
                elseif box_deriv isa AbstractMatrix
                    fill!(G_mat, 0.0)
                    SystemContract._accumulate_box_derivative!(G_mat, f, q, ctx, Symbol("factor_", i))
                    box_deriv .+= G_mat
                else
                    fill!(G_mat, 0.0)
                    SystemContract._accumulate_box_derivative!(G_mat, f, q, ctx, Symbol("factor_", i))
                    box_deriv[1] += G_mat[1, 1]
                    box_deriv[2] += G_mat[2, 2]
                    box_deriv[3] += G_mat[3, 3]
                end
            end
        end
    end

    return u_total
end

"""
    evaluate_step!(eng::SimulationEngine, F, q, box; box_deriv=nothing) -> Float64

Evaluates all factors in `eng` at positions `q` and periodic `box`, overwriting `F` with
`-∂U/∂q` (and `box_deriv` with `∂U/∂h` when supplied), routing nonbonded factors through
`TriumGPU` (`:auto`, `:gpu`, `:metal`, `:vulkan`, `:sycl`, or `:cpu`). Returns total potential energy `U`.
"""
function evaluate_step!(
    eng::SimulationEngine,
    F::AbstractMatrix{Float64},
    q::AbstractMatrix{Float64},
    box::Types.AbstractBox;
    box_deriv::Union{Nothing, AbstractMatrix{Float64}, AbstractVector{Float64}} = nothing
)::Float64
    return _evaluate_factor_subset!(eng, F, q, box, eachindex(eng.factors); box_deriv = box_deriv)
end

function _extract_batched_gpu_factors(eng::SimulationEngine)
    nb_fac::Union{Nothing, TriumXBT.TriumTiledNonbondedFactor} = nothing
    bonds = Factors.HarmonicBond[]
    angles = Factors.HarmonicAngle[]
    for f in eng.factors
        Factors.force_consistent(f) || return (false, nothing, bonds, angles)
        if f isa TriumXBT.TriumTiledNonbondedFactor
            nb_fac === nothing || return (false, nothing, bonds, angles)
            nb_fac = f
        elseif f isa TriumGPU.TriumGPUNonbondedFactor
            nb_fac === nothing || return (false, nothing, bonds, angles)
            sub_backend = eng.backend === :auto ? f.backend : eng.backend
            sub_backend === :cpu && return (false, nothing, bonds, angles)
            nb_fac = f.inner
        elseif f isa Factors.HarmonicBond
            push!(bonds, f)
        elseif f isa Factors.HarmonicAngle
            push!(angles, f)
        else
            return (false, nothing, bonds, angles)
        end
    end
    nb_fac === nothing && return (false, nothing, bonds, angles)
    (!nb_fac.use_carry_free &&
     (nb_fac.alpha > 0.0 || nb_fac.coulomb_prefactor == 0.0 || all(iszero, nb_fac.charges))) ||
        return (false, nothing, bonds, angles)
    return (true, nb_fac, bonds, angles)
end

@inline function _engine_force_model_supported(
    eng::SimulationEngine,
    masses::Vector{Float64},
    box::Types.AbstractBox
)::Bool
    hs = Dynamics.HarmonicSystem(
        eng.factors, masses,
        Factors.Context(box, _system_units(eng.system), length(masses))
    )
    return Dynamics.force_model_supported(hs)
end

function _run_verlet!(eng::SimulationEngine, state::Dynamics.State, n_steps::Int)
    n = _system_natoms(eng.system)
    m = _system_masses(eng.system)
    box = _system_box(eng.system)
    dt = eng.dt_ps
    half_dt = 0.5 * dt

    active = eng.active_backend
    if active in (:metal, :vulkan, :sycl) && n_steps >= 10 && TriumGPU.has_gpu_md_batch(active)
        ok_batch, nb_fac, bonds, angles = _extract_batched_gpu_factors(eng)
        if ok_batch && nb_fac !== nothing && _engine_force_model_supported(eng, m, box)
            inv_mass = Vector{Float64}(undef, n)
            @inbounds for j in 1:n
                inv_mass[j] = 1.0 / m[j]
            end
            TriumGPU.run_batched_gpu_steps!(
                state.q, state.p, inv_mass, box,
                nb_fac, bonds, angles, dt, n_steps;
                F_out = state.F,
                backend = active
            )
            state.force_valid = true
            return state
        end
    end

    if !state.force_valid
        evaluate_step!(eng, state.F, state.q, box)
        state.force_valid = true
    end

    for _ in 1:n_steps
        @inbounds for j in 1:n, d in 1:3
            state.p[d, j] += half_dt * state.F[d, j]
        end
        @inbounds for j in 1:n
            inv_m = 1.0 / m[j]
            for d in 1:3
                state.q[d, j] += dt * state.p[d, j] * inv_m
            end
        end
        evaluate_step!(eng, state.F, state.q, box)
        @inbounds for j in 1:n, d in 1:3
            state.p[d, j] += half_dt * state.F[d, j]
        end
    end
    state.force_valid = true
    return state
end

function _run_baoab!(eng::SimulationEngine, state::Dynamics.State, n_steps::Int)
    n = _system_natoms(eng.system)
    m = _system_masses(eng.system)
    box = _system_box(eng.system)
    units = _system_units(eng.system)
    dt = eng.dt_ps
    half_dt = 0.5 * dt
    beta = 1.0 / (units.boltzmann * eng.temperature)
    c1 = exp(-eng.gamma_ps_inv * dt)
    c2 = sqrt((1.0 - c1 * c1) / beta)

    if !state.force_valid
        evaluate_step!(eng, state.F, state.q, box)
        state.force_valid = true
    end

    for _ in 1:n_steps
        @inbounds for j in 1:n, d in 1:3
            state.p[d, j] += half_dt * state.F[d, j]
        end
        @inbounds for j in 1:n
            inv_m = 1.0 / m[j]
            for d in 1:3
                state.q[d, j] += half_dt * state.p[d, j] * inv_m
            end
        end
        @inbounds for j in 1:n
            sm = sqrt(m[j])
            for d in 1:3
                state.p[d, j] = c1 * state.p[d, j] + c2 * sm * randn(eng.rng)
            end
        end
        @inbounds for j in 1:n
            inv_m = 1.0 / m[j]
            for d in 1:3
                state.q[d, j] += half_dt * state.p[d, j] * inv_m
            end
        end
        evaluate_step!(eng, state.F, state.q, box)
        @inbounds for j in 1:n, d in 1:3
            state.p[d, j] += half_dt * state.F[d, j]
        end
    end
    state.force_valid = true
    return state
end

function _run_respa!(eng::SimulationEngine, state::Dynamics.State, n_steps::Int)
    n = _system_natoms(eng.system)
    m = _system_masses(eng.system)
    box = _system_box(eng.system)
    dt = eng.dt_ps
    m_sub = eng.respa_m
    dt_fast = dt / m_sub
    half_dt_outer = 0.5 * dt
    half_dt_fast = 0.5 * dt_fast

    fast_idx = eng.fast_indices
    slow_idx = setdiff(collect(eachindex(eng.factors)), fast_idx)
    F_fast = zeros(Float64, 3, n)
    F_slow = zeros(Float64, 3, n)

    for _ in 1:n_steps
        _evaluate_factor_subset!(eng, F_slow, state.q, box, slow_idx)
        @inbounds for j in 1:n, d in 1:3
            state.p[d, j] += half_dt_outer * F_slow[d, j]
        end

        _evaluate_factor_subset!(eng, F_fast, state.q, box, fast_idx)
        for _ in 1:m_sub
            @inbounds for j in 1:n, d in 1:3
                state.p[d, j] += half_dt_fast * F_fast[d, j]
            end
            @inbounds for j in 1:n
                inv_m = 1.0 / m[j]
                for d in 1:3
                    state.q[d, j] += dt_fast * state.p[d, j] * inv_m
                end
            end
            _evaluate_factor_subset!(eng, F_fast, state.q, box, fast_idx)
            @inbounds for j in 1:n, d in 1:3
                state.p[d, j] += half_dt_fast * F_fast[d, j]
            end
        end

        _evaluate_factor_subset!(eng, F_slow, state.q, box, slow_idx)
        @inbounds for j in 1:n, d in 1:3
            state.p[d, j] += half_dt_outer * F_slow[d, j]
        end
    end

    @inbounds for j in 1:n, d in 1:3
        state.F[d, j] = F_fast[d, j] + F_slow[d, j]
    end
    state.force_valid = true
    return state
end

"""
    run_steps!(eng::SimulationEngine, state, n_steps::Int)

Advances `state` (`Dynamics.State` or `SymplecticTernary.SymplecticTernaryState`) by
`n_steps` steps using `eng.integrator` (`:verlet`, `:baoab`, `:respa`, or `:symplectic_ternary`),
routing all nonbonded evaluations through the configured `TriumGPU`/`TriumXBT` backend.
"""
function run_steps!(eng::SimulationEngine, state::Dynamics.State, n_steps::Int)
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative, got $n_steps"))
    n = _system_natoms(eng.system)
    size(state.q) == (3, n) && size(state.p) == (3, n) && size(state.F) == (3, n) ||
        throw(ArgumentError("state shapes must match (3, $n)"))
    n_steps == 0 && return state

    if eng.integrator === :verlet
        return _run_verlet!(eng, state, n_steps)
    elseif eng.integrator === :baoab
        return _run_baoab!(eng, state, n_steps)
    elseif eng.integrator === :respa
        return _run_respa!(eng, state, n_steps)
    elseif eng.integrator === :symplectic_ternary
        masses = _system_masses(eng.system)
        box = _system_box(eng.system)
        st_tern = SymplecticTernary.SymplecticTernaryState(state, masses, box)
        run_steps!(eng, st_tern, n_steps)
        @inbounds for j in 1:n, d in 1:3
            state.q[d, j] = Float64(st_tern.Q[d, j]) / st_tern.q_scale
            state.p[d, j] = Float64(st_tern.P[d, j]) / st_tern.p_scale
        end
        evaluate_step!(eng, state.F, state.q, box)
        state.force_valid = true
        return state
    else
        throw(ArgumentError("unsupported integrator $(repr(eng.integrator))"))
    end
end

function run_steps!(eng::SimulationEngine, state::SymplecticTernary.SymplecticTernaryState, n_steps::Int)
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative, got $n_steps"))
    n = _system_natoms(eng.system)
    length(state.masses) == n ||
        throw(ArgumentError("SymplecticTernaryState atom count $(length(state.masses)) != $n"))
    n_steps == 0 && return state

    eng.integrator === :symplectic_ternary ||
        throw(ArgumentError("SymplecticTernaryState requires integrator=:symplectic_ternary, got $(repr(eng.integrator))"))

    box = state.box isa Types.NoBox && !(_system_box(eng.system) isa Types.NoBox) ?
          _system_box(eng.system) : state.box
    force_fn = (q_mat, F_buf) -> evaluate_step!(eng, F_buf, q_mat, box)
    SymplecticTernary.simulate_symplectic_ternary!(state, force_fn, eng.dt_ps, n_steps)
    return state
end

"""
    verify_engine_conservation(eng::SimulationEngine, state0::Dynamics.State; n_steps::Int = 200) -> NamedTuple

Runs `n_steps` on a deep copy of `state0` using `eng`, measures wall-clock throughput,
total energy drift, and total linear momentum drift, and returns a telemetry `NamedTuple`:
`(n_steps, elapsed_s, ms_per_step, ns_per_day_2fs, initial_energy, final_energy,
  abs_energy_drift, rel_energy_drift, momentum_drift_norm, backend)`.
"""
function verify_engine_conservation(
    eng::SimulationEngine,
    state0::Dynamics.State;
    n_steps::Int = 200
)
    n_steps >= 1 || throw(ArgumentError("n_steps must be >= 1, got $n_steps"))
    n = _system_natoms(eng.system)
    size(state0.q) == (3, n) && size(state0.p) == (3, n) && size(state0.F) == (3, n) ||
        throw(ArgumentError("state0 shapes must match (3, $n)"))

    st = deepcopy(state0)
    m = _system_masses(eng.system)
    box = _system_box(eng.system)
    F_tmp = zeros(Float64, 3, n)

    PE0 = evaluate_step!(eng, F_tmp, st.q, box)
    KE0 = 0.5 * sum(@inbounds(st.p[d, j]^2 / m[j]) for j in 1:n for d in 1:3)
    E0 = PE0 + KE0
    P0 = vec(sum(st.p, dims = 2))
    if !st.force_valid
        copyto!(st.F, F_tmp)
        st.force_valid = true
    end

    t0 = time_ns()
    run_steps!(eng, st, n_steps)
    elapsed_s = (time_ns() - t0) * 1e-9

    PE1 = evaluate_step!(eng, F_tmp, st.q, box)
    KE1 = 0.5 * sum(@inbounds(st.p[d, j]^2 / m[j]) for j in 1:n for d in 1:3)
    E1 = PE1 + KE1
    P1 = vec(sum(st.p, dims = 2))

    ms_per_step = (elapsed_s * 1e3) / n_steps
    ns_per_day_2fs = TriumXBT.effective_ns_per_day(max(ms_per_step, 1e-12); dt_fs = 2.0)
    abs_energy_drift = abs(E1 - E0)
    rel_energy_drift = abs_energy_drift / max(abs(E0), 1.0)
    momentum_drift_norm = norm(P1 - P0)

    return (
        n_steps = n_steps,
        elapsed_s = elapsed_s,
        ms_per_step = ms_per_step,
        ns_per_day_2fs = ns_per_day_2fs,
        initial_energy = E0,
        final_energy = E1,
        abs_energy_drift = abs_energy_drift,
        rel_energy_drift = rel_energy_drift,
        momentum_drift_norm = momentum_drift_norm,
        backend = eng.active_backend,
    )
end

end # module Engine
