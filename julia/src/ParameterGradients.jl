"""
Joint force and parameter derivatives for the factor protocol.

`evaluate_parameter_gradients!` returns the same energy and force as
`Factors.total_energy_forces` for the *same* factor list and configuration, and
additionally fills `∂U/∂p` for every supported scalar parameter at fixed `q`.
The force is not re-derived here: each factor's existing `evaluate!` is called,
so `F = -∂U/∂q` is exactly the evaluator's force (including cutoff modes,
`lj_scale`, `coulomb_scale`, and the Coulomb prefactor). Only the parameter
chain rule is new.

## Gradient keys

Keys are stable, explicit symbols formed from a factor-kind prefix and the
factor's 1-based position in the `factors` iteration order:

| Factor | Keys |
| --- | --- |
| `HarmonicBond(i,j,k,r0)` | `:bond_k_<i>`, `:bond_r0_<i>` |
| `HarmonicAngle(i,j,k,kθ,θ0)` | `:angle_ktheta_<i>`, `:angle_theta0_<i>` |
| `PeriodicDihedral(i,j,k,l,kφ,n,φ0)` | `:dihedral_kphi_<i>`, `:dihedral_phi0_<i>` |
| `LennardJonesPair(i,j,ε,σ; mode, rc)` | `:lj_epsilon_<i>`, `:lj_sigma_<i>` |
| `CoulombPair(i,j,qi,qj)` | `:coulomb_qi_<i>`, `:coulomb_qj_<i>` |

`<i>` is the factor index, not an atom index. The torsion multiplicity `n` and
all non-parameter fields are held fixed.

## Shared parameters

`parameter_ids` maps `(factor_index, local_name)` to a shared symbol, where
`local_name` is one of `:k`, `:r0`, `:ktheta`, `:theta0`, `:kphi`, `:phi0`,
`:epsilon`, `:sigma`, `:qi`, `:qj`. Contributions from every factor mapped to
the same id are summed into that one key. The default is an empty mapping, so
each factor gets the per-factor keys above. A mapping is ignored for a
`(factor_index, local_name)` pair that the factor does not own.

## Parameter value overrides

`parameters::AbstractDict{Symbol,Float64}` supplies authoritative values used
for *both* the energy evaluation and the derivative. Lookup order for a local
parameter is:

1. its shared id in `parameter_ids`, if mapped and present in `parameters`;
2. its per-factor key (`:bond_k_3`, ...), if present in `parameters`;
3. the value already stored in the factor struct.

This lets a caller drive shared parameters without rebuilding the factor list,
and it is also what the finite-difference tests perturb. The default empty dict
means "use the factor fields as-is".

## Cutoffs

Cutoffs are fixed parameters, not differentiated. For `LennardJonesPair` the
derivative follows the configured mode exactly:

* `:none` and `:trunc`: derivative of the plain 12-6 form inside `rc`;
* `:shift`: includes `-∂U(rc)/∂p`;
* `:lfs`: includes `-∂U(rc)/∂p - ∂U'(rc)/∂p · (r - rc)`;
* any cutoff mode with `r > rc`: energy is the evaluator's exact `0.0`, so both
  derivatives are written as `0.0`.

No cutoff is invented or rescaled. `:trunc` keeps its documented discontinuity;
this module differentiates its energy, not a smoothed variant of it.

## Singular and degenerate configurations

If a factor's `evaluate!` returns a nonfinite energy (coincident atoms,
collinear angle/dihedral, zero normals), no derivative entries are added for
that factor: the total energy is nonfinite and that factor's parameter
derivative is undefined. Finite factors in the same list still contribute.
For a singular factor the existing evaluator also scatters no force.

## Factor coverage

Only the five factor types in the key table are supported. `BipartiteHarmonicGroup`
and any future factor are rejected with `ArgumentError` rather than silently
omitting derivatives.
"""
module ParameterGradients

using ..Factors
using ..Types

using ..Factors: evaluate!, Context, AbstractFactor,
    HarmonicBond, HarmonicAngle, PeriodicDihedral, LennardJonesPair, CoulombPair,
    dihedral_angle_grad, _dot
using ..Types: unwrap_displacement, norm2

export ParameterGradientWorkspace, evaluate_parameter_gradients!

"""
Reusable state for `evaluate_parameter_gradients!`.

Fields:

* `F` — the `3 x N` force accumulator, zeroed on every call;
* `gradient_scratch` — per-factor derivative buffer, reused between calls;
* `parameter_ids` — default shared-id mapping for the workspace overload.

The caller still owns the output `gradients` dictionary; it is emptied at the
start of each call, so stale keys from a previous, differently shaped factor
list cannot survive.
"""
mutable struct ParameterGradientWorkspace
    F::Matrix{Float64}
    gradient_scratch::Dict{Symbol,Float64}
    parameter_ids::Dict{Tuple{Int,Symbol},Symbol}
end

"""
    ParameterGradientWorkspace(N; parameter_ids = Dict())

Workspace for a system with `N` atoms.
"""
function ParameterGradientWorkspace(N::Integer;
                                    parameter_ids = Dict{Tuple{Int,Symbol},Symbol}())
    N >= 0 || throw(ArgumentError("atom count must be nonnegative, got $N"))
    return ParameterGradientWorkspace(zeros(3, Int(N)),
                                      Dict{Symbol,Float64}(),
                                      Dict{Tuple{Int,Symbol},Symbol}(parameter_ids))
end

"Workspace sized from an existing `Context`."
ParameterGradientWorkspace(ctx::Context; kwargs...) =
    ParameterGradientWorkspace(ctx.N; kwargs...)

# ---------------------------------------------------------------------------
# supported factors and key construction
# ---------------------------------------------------------------------------

const _SUPPORTED = Union{HarmonicBond,HarmonicAngle,PeriodicDihedral,
                         LennardJonesPair,CoulombPair}

function _check_supported(f::AbstractFactor)
    f isa _SUPPORTED || throw(ArgumentError(
        "ParameterGradients has no parameter derivative for $(typeof(f)); " *
        "supported factors are HarmonicBond, HarmonicAngle, PeriodicDihedral, " *
        "LennardJonesPair, CoulombPair"))
    return nothing
end

const _KEY_PREFIX = Dict{Tuple{DataType,Symbol},String}(
    (HarmonicBond,     :k)      => "bond_k",
    (HarmonicBond,     :r0)     => "bond_r0",
    (HarmonicAngle,    :ktheta) => "angle_ktheta",
    (HarmonicAngle,    :theta0) => "angle_theta0",
    (PeriodicDihedral, :kphi)   => "dihedral_kphi",
    (PeriodicDihedral, :phi0)   => "dihedral_phi0",
    (LennardJonesPair, :epsilon) => "lj_epsilon",
    (LennardJonesPair, :sigma)   => "lj_sigma",
    (CoulombPair,      :qi)      => "coulomb_qi",
    (CoulombPair,      :qj)      => "coulomb_qj",
)

"Per-factor key for `pname` on factor at position `idx` (1-based)."
function _factor_key(::Type{F}, idx::Int, pname::Symbol) where {F<:AbstractFactor}
    prefix = _KEY_PREFIX[(F, pname)]
    return Symbol(prefix, "_", idx)
end

"Key after applying the optional shared-id mapping."
function _resolved_key(f::AbstractFactor, idx::Int, pname::Symbol, ids)
    return get(ids, (idx, pname), _factor_key(typeof(f), idx, pname))
end

# ---------------------------------------------------------------------------
# parameter overrides: build an equivalent factor with effective values
# ---------------------------------------------------------------------------

function _param_value(f::AbstractFactor, idx::Int, pname::Symbol, default::Float64,
                      parameters, ids)
    shared = get(ids, (idx, pname), nothing)
    if shared !== nothing && haskey(parameters, shared)
        return parameters[shared]
    end
    per_factor = _factor_key(typeof(f), idx, pname)
    return get(parameters, per_factor, default)
end

function _effective_factor(f::HarmonicBond, idx, parameters, ids)
    k  = _param_value(f, idx, :k,  f.k,  parameters, ids)
    r0 = _param_value(f, idx, :r0, f.r0, parameters, ids)
    return HarmonicBond(f.i, f.j, k, r0)
end

function _effective_factor(f::HarmonicAngle, idx, parameters, ids)
    ktheta = _param_value(f, idx, :ktheta, f.ktheta, parameters, ids)
    theta0 = _param_value(f, idx, :theta0, f.theta0, parameters, ids)
    return HarmonicAngle(f.i, f.j, f.k, ktheta, theta0)
end

function _effective_factor(f::PeriodicDihedral, idx, parameters, ids)
    kphi = _param_value(f, idx, :kphi, f.kphi, parameters, ids)
    phi0 = _param_value(f, idx, :phi0, f.phi0, parameters, ids)
    return PeriodicDihedral(f.i, f.j, f.k, f.l, kphi, f.n, phi0)
end

function _effective_factor(f::LennardJonesPair, idx, parameters, ids)
    ε = _param_value(f, idx, :epsilon, f.ε, parameters, ids)
    σ = _param_value(f, idx, :sigma,   f.σ, parameters, ids)
    return LennardJonesPair(f.i, f.j, ε, σ, f.mode, f.rc, f.lj_scale)
end

function _effective_factor(f::CoulombPair, idx, parameters, ids)
    qi = _param_value(f, idx, :qi, f.qi, parameters, ids)
    qj = _param_value(f, idx, :qj, f.qj, parameters, ids)
    return CoulombPair(f.i, f.j, qi, qj, f.coulomb_scale)
end

# ---------------------------------------------------------------------------
# analytic parameter derivatives of the existing local energy forms
# ---------------------------------------------------------------------------

@inline function _contribute!(out::Dict{Symbol,Float64}, f::AbstractFactor,
                              idx::Int, pname::Symbol, value::Float64, ids)
    key = _resolved_key(f, idx, pname, ids)
    out[key] = get(out, key, 0.0) + value
    return nothing
end

function _parameter_derivatives!(out::Dict{Symbol,Float64}, f::HarmonicBond,
                                 idx::Int, q, ctx, ids)
    d = unwrap_displacement(ctx.box, q, f.i, f.j)
    r2 = norm2(d)
    r2 < 1e-24 && return out
    r = sqrt(r2)
    dr = r - f.r0
    # U = (k/2)(r-r0)^2
    _contribute!(out, f, idx, :k,  0.5 * dr * dr, ids)
    _contribute!(out, f, idx, :r0, -f.k * dr, ids)
    return out
end

function _parameter_derivatives!(out::Dict{Symbol,Float64}, f::HarmonicAngle,
                                 idx::Int, q, ctx, ids)
    u1 = unwrap_displacement(ctx.box, q, f.i, f.j)
    u2 = unwrap_displacement(ctx.box, q, f.k, f.j)
    r1 = sqrt(norm2(u1)); r2 = sqrt(norm2(u2))
    (r1 < 1e-12 || r2 < 1e-12) && return out
    ct = clamp(_dot(u1, u2) / (r1 * r2), -1.0, 1.0)
    th = acos(ct)
    dth = th - f.theta0
    abs(sin(th)) < 1e-12 && return out
    # U = (kθ/2)(θ-θ0)^2
    _contribute!(out, f, idx, :ktheta, 0.5 * dth * dth, ids)
    _contribute!(out, f, idx, :theta0, -f.ktheta * dth, ids)
    return out
end

function _parameter_derivatives!(out::Dict{Symbol,Float64}, f::PeriodicDihedral,
                                 idx::Int, q, ctx, ids)
    t = dihedral_angle_grad(q, ctx.box, f.i, f.j, f.k, f.l)
    t === nothing && return out
    phi = t[1]
    arg = f.n * phi - f.phi0
    # U = kφ [1 + cos(nφ - φ0)]
    _contribute!(out, f, idx, :kphi, 1.0 + cos(arg), ids)
    _contribute!(out, f, idx, :phi0, f.kphi * sin(arg), ids)
    return out
end

# ∂/∂ε and ∂/∂σ of the plain 12-6 form 4ε[(σ/r)^12 - (σ/r)^6], at fixed r.
@inline function _lj_depsilon(ε::Float64, σ::Float64, r::Float64)
    s6 = (σ / r)^6
    return 4.0 * (s6 * s6 - s6)
end

@inline function _lj_dsigma(ε::Float64, σ::Float64, r::Float64)
    s6 = (σ / r)^6
    return 24.0 * ε * (2.0 * s6 - 1.0) * s6 / σ
end

# ∂/∂ε and ∂/∂σ of dU/dr = 24ε[(σ/r)^6 - 2(σ/r)^12]/r.
@inline function _lj_dudr_depsilon(ε::Float64, σ::Float64, r::Float64)
    s6 = (σ / r)^6
    return 24.0 * (s6 - 2.0 * s6 * s6) / r
end

@inline function _lj_dudr_dsigma(ε::Float64, σ::Float64, r::Float64)
    s6 = (σ / r)^6
    return 24.0 * ε * (1.0 - 4.0 * s6) * 6.0 * s6 / (σ * r)
end

function _parameter_derivatives!(out::Dict{Symbol,Float64}, f::LennardJonesPair,
                                 idx::Int, q, ctx, ids)
    d = unwrap_displacement(ctx.box, q, f.i, f.j)
    r2 = norm2(d)
    r2 < 1e-24 && return out
    r = sqrt(r2)

    # The evaluator returns exactly 0.0 beyond a fixed cutoff; both partials
    # are then exactly zero too (rc itself is not differentiated).
    if f.mode !== :none && r > f.rc
        _contribute!(out, f, idx, :epsilon, 0.0, ids)
        _contribute!(out, f, idx, :sigma,   0.0, ids)
        return out
    end

    dε = _lj_depsilon(f.ε, f.σ, r)
    dσ = _lj_dsigma(f.ε, f.σ, r)

    if f.mode === :lfs
        dε -= _lj_depsilon(f.ε, f.σ, f.rc)
        dσ -= _lj_dsigma(f.ε, f.σ, f.rc)
        dr = r - f.rc
        dε -= _lj_dudr_depsilon(f.ε, f.σ, f.rc) * dr
        dσ -= _lj_dudr_dsigma(f.ε, f.σ, f.rc) * dr
    elseif f.mode === :shift
        dε -= _lj_depsilon(f.ε, f.σ, f.rc)
        dσ -= _lj_dsigma(f.ε, f.σ, f.rc)
    elseif f.mode !== :trunc && f.mode !== :none
        error("unknown LennardJonesPair mode $(f.mode)")
    end

    _contribute!(out, f, idx, :epsilon, dε * f.lj_scale, ids)
    _contribute!(out, f, idx, :sigma,   dσ * f.lj_scale, ids)
    return out
end

function _parameter_derivatives!(out::Dict{Symbol,Float64}, f::CoulombPair,
                                 idx::Int, q, ctx, ids)
    d = unwrap_displacement(ctx.box, q, f.i, f.j)
    r2 = norm2(d)
    r2 < 1e-24 && return out
    r = sqrt(r2)
    pref = ctx.units.coulomb_prefactor * f.coulomb_scale
    # U = pref·qi·qj/r
    _contribute!(out, f, idx, :qi, pref * f.qj / r, ids)
    _contribute!(out, f, idx, :qj, pref * f.qi / r, ids)
    return out
end

# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

"""
    evaluate_parameter_gradients!(F, gradients, factors, q, ctx;
                                  parameters = Dict{Symbol,Float64}(),
                                  parameter_ids = Dict{Tuple{Int,Symbol},Symbol}())

Zero `F`, empty `gradients`, then for every factor in `factors` (in order):

1. build an equivalent factor with any `parameters` overrides applied;
2. call the factor's existing `evaluate!`, accumulating `F` and the energy;
3. if the local energy is finite, add its analytic `∂U/∂p` contributions to
   `gradients`, resolving shared ids through `parameter_ids`.

Returns the total energy. `F` is exactly the force from
`total_energy_forces` on the same effective factor list (same factor order,
same evaluator), not a re-derived approximation. On a singular/degenerate
factor the energy is nonfinite and no derivative keys are added for that factor.

The caller-owned `gradients` dictionary is cleared first, so it can be reused
across calls. See the module docstring for keys, overrides, cutoffs, and the
singular behaviour.
"""
function evaluate_parameter_gradients!(F::AbstractMatrix{Float64},
                                       gradients::Dict{Symbol,Float64},
                                       factors, q::AbstractMatrix{Float64},
                                       ctx::Context;
                                       parameters = Dict{Symbol,Float64}(),
                                       parameter_ids = Dict{Tuple{Int,Symbol},Symbol}(),
                                       _gradient_scratch::Union{Nothing,Dict{Symbol,Float64}} = nothing)
    fill!(F, 0.0)
    empty!(gradients)

    U = 0.0
    scratch = _gradient_scratch
    for (idx, f) in enumerate(factors)
        _check_supported(f)
        fe = _effective_factor(f, idx, parameters, parameter_ids)
        u = evaluate!(F, fe, q, ctx)
        U += u
        # A nonfinite local energy has no derivative; the evaluator has also
        # scattered no force for this factor. Other factors still contribute.
        isfinite(u) || continue

        if scratch === nothing
            buf = Dict{Symbol,Float64}()
        else
            empty!(scratch)
            buf = scratch
        end
        _parameter_derivatives!(buf, fe, idx, q, ctx, parameter_ids)
        for (key, value) in buf
            gradients[key] = get(gradients, key, 0.0) + value
        end
    end
    return U
end

"""
    evaluate_parameter_gradients!(ws, gradients, factors, q, ctx; parameters = Dict())

Workspace overload: uses `ws.F` as the force accumulator, `ws.gradient_scratch`
as the reusable per-factor buffer, and `ws.parameter_ids` as the shared-id
mapping.
"""
function evaluate_parameter_gradients!(ws::ParameterGradientWorkspace,
                                       gradients::Dict{Symbol,Float64},
                                       factors, q::AbstractMatrix{Float64},
                                       ctx::Context;
                                       parameters = Dict{Symbol,Float64}())
    return evaluate_parameter_gradients!(ws.F, gradients, factors, q, ctx;
                                         parameters = parameters,
                                         parameter_ids = ws.parameter_ids,
                                         _gradient_scratch = ws.gradient_scratch)
end

end # module ParameterGradients
