"""
Equilibrium (fixed-measure) parameter sensitivities from correlated samples.

## What this module estimates

For a fixed inverse temperature `β`, a fixed configuration domain, and a
parameter-independent measure, the equilibrium free-energy derivative is

    ∂A/∂θ = ⟨∂U/∂θ⟩_β ,

where `⟨·⟩_β` is the average over `π(q) ∝ exp(-βU(q; θ))`.  This module
estimates the *mean parameter gradient* from samples and attaches an
autocorrelation-aware standard error.  It does **not** estimate a finite free
energy difference: a finite `Δθ` requires integrating the sensitivity over that
path, and the module never claims otherwise.  It also assumes the measure and
domain do not depend on `θ`; parameter-dependent restraints, Jacobians, or
integration limits are outside its scope.

The gradient function must return the fixed-configuration derivative
`∂U/∂θ` (for example, the values produced by
`ParameterGradients.evaluate_parameter_gradients!`).  The sample mean of those
vectors is the estimator of `∂A/∂θ`.

## Input forms

    estimate_sensitivity(grad_samples; min_ess = 100)
    estimate_sensitivity(samples_matrix, gradient_fn; min_ess = 100)

* `grad_samples::AbstractVector{<:AbstractDict}` is a sequence of per-sample
  dictionaries whose keys are parameter ids (`Symbol`) and whose values are
  `∂U/∂θ` at that sample.
* `samples_matrix` has one **configuration per column** (`size(samples, 2)` is
  the sample count), and `gradient_fn` is called as `gradient_fn(view(samples,
  :, s))`, returning such a dictionary.

Every sample must expose the same key set.  A missing key is an error rather
than an implicit zero: silently treating an absent derivative as zero would make
a broken evaluator look like an exact sensitivity.

## Autocorrelation and uncertainty

The module uses the *biased* (normalization `1/n`) autocovariance

    γ(t) = (1/n) Σ_{i=1}^{n-t} (x_i - x̄)(x_{i+t} - x̄),    ρ(t) = γ(t)/γ(0).

The unbiased `1/(n-t)` estimator is deliberately not used: it does not
guarantee a positive-semidefinite autocovariance sequence, and individual
`ρ(t)` can exceed one.  The `1/n` bias is `O(t/n)` and is small for the
truncation windows used here.

The integrated autocorrelation time is

    τ_int = 1/2 + Σ_{t≥1} ρ(t),

so an independent chain has `τ_int = 1/2` and the effective sample size is

    ESS = n / (2 τ_int).

This is the `1/2 + Σ_t ρ(t)` convention.  `Sampling.integrated_autocorrelation`
returns `1 + 2 Σ_t ρ(t) = 2 τ_int` for the lags it retains, so its returned
`tau` is twice this module's `tau_int`; do not mix the two numbers when
comparing diagnostics.

Truncation uses Geyer's **initial positive sequence with a monotone envelope**:
pair the correlations as `Γ_t = ρ(2t) + ρ(2t+1)` for `t = 0, 1, ...`; stop at
the first `Γ_t <= 0`; then replace the retained sequence by its nonincreasing
envelope `Γ_t ← min(Γ_t, Γ_{t-1})`.  The estimate is

    τ_int = -1/2 + Σ_t Γ_t,   floored at 1/2.

The floor makes `ESS <= n`; it is conservative for a negatively correlated
chain and keeps the reported effective size from exceeding the number of draws.
The default maximum lag is `min(n-1, 10*max(1, n÷100))`; a user `maxlag` can
only shorten it.  This truncation is a standard estimator, not a theorem about
a particular chain: a single run's ESS has its own sampling error, and an
alternating or resonant chain can defeat initial-positive truncation (as
`Sampling.integrated_autocorrelation` also documents).

For each parameter component the standard error of the mean is

    SE = sqrt(γ(0) / ESS).

Components are estimated **separately**, each with its own `τ_int` and `ESS`; a
single vector covariance matrix is not reported.  If components have different
autocorrelation times, a covariance built from one common `τ` would be wrong,
so the module does not pretend to provide one.  When a charge-tangent projection
is applied, it is applied to each sample *before* these marginal estimates, so
the projected component's variance includes the within-sample cross-component
covariance induced by the projection.  Cross-leg covariance is still not
estimated (see `binding_sensitivity`).

## Status, never silent certainty

`SensitivityResult.status` is `:ok` only when every component has
`ESS >= min_ess`.  Otherwise it is `:low_ess`, and the offending keys are listed
in `low_ess`.  A low-ESS result still reports its mean and SE, but the status is
part of the result so a caller cannot mistake a short correlated run for a
precise one.  `min_ess` defaults to 100 in `estimate_sensitivity` and is
propagated to both legs by `binding_sensitivity`.

## Binding differences: exactly what is and is not subtracted

    binding_sensitivity(complex_samples, ligand_samples, grad_fn;
                        corrections = Dict(),
                        charge_tangent_groups = [])

returns the component-wise vector

    mean complex - mean ligand + correction.

It does **not** compute a physical binding affinity or a standard-state free
energy.  It performs only the subtraction the caller asks for.  Receptor
cancellation is valid only when both legs use the same fixed receptor
configuration and the same parameter definitions; this API cannot check that
physical precondition, so it does not assert it.  The caller must supply any
standard-state, restraint, or finite-size correction explicitly through
`corrections`; the default is empty.  The two legs are assumed to be
independent ensembles, so the difference SE is the quadrature sum
`sqrt(SE_c² + SE_l²)`.  Corrections are treated as exact constants (zero
uncertainty); if they have their own uncertainty, propagate it outside.

`corrections` keys must be parameter ids already present in both legs.  If
`charge_tangent_groups` is nonempty, the correction vector is projected by the
same tangent operator before being added, so the reported difference itself
lies in the tangent.  This order is explicit because projecting only part of a
sum would otherwise leave a misleading non-tangent result.

## Charge tangents

`project_charge_tangent!` / `project_charge_tangent` take groups of gradient ids
(for example `[:coulomb_qi_1, :coulomb_qj_1, :coulomb_qi_2, ...]`) and subtract
the group mean from every member:

    g_i ← g_i - (1/|G|) Σ_{j∈G} g_j.

That is the orthogonal projection onto the sum-zero (charge-transfer tangent)
subspace.  Every group must be nonempty, contain no duplicate ids, and every id
must be present.  Groups must be disjoint: applying overlapping projections
sequentially would silently destroy the first constraint.  This projection is a
linear algebraic constraint, not a charge-transfer model, and it does not add
missing physics.

## Factor-protocol adapter

    factor_gradient_function(factors, ctx; parameter_ids, parameters)

returns a closure `q -> Dict{Symbol,Float64}` that calls
`ParameterGradients.evaluate_parameter_gradients!` with the supplied factor list,
context, shared-id map, and parameter overrides.  The closure accepts either a
`3 × ctx.N` configuration matrix or a length-`3ctx.N` vector.  It allocates a
fresh force buffer and gradient dictionary per call; it is reentrant, but it is
not a zero-allocation workspace.

## Limits, stated once

* Fixed `β`; parameter-independent measure and domain only.
* No finite-`Δθ` free-energy change and no thermodynamic integration.
* No covariance matrix across parameters; per-component marginal errors only.
* No cross-correlation between the complex and ligand legs.
* No check that a receptor, standard state, or restraint is physically shared.
* No convergence proof.  A large ESS measures noise, not bias, and a trapped
  chain can have a large ESS while sampling the wrong distribution.
"""
module EquilibriumSensitivities

using Statistics: mean
using ..Types
using ..Factors
using ..ParameterGradients

export SensitivityResult, BindingSensitivityResult,
       estimate_sensitivity, binding_sensitivity,
       factor_gradient_function, project_charge_tangent!,
       project_charge_tangent, integrated_autocorrelation_time,
       effective_sample_size

# ---------------------------------------------------------------------------
# result types
# ---------------------------------------------------------------------------

"""
Result of `estimate_sensitivity`.

Fields are dictionaries keyed by parameter id:

* `means`   : sample mean of `∂U/∂θ`;
* `se`      : autocorrelation-adjusted standard error of that mean;
* `ess`     : effective sample size for that component;
* `tau_int` : integrated autocorrelation time, `τ_int = 1/2 + Σ ρ(t)`;
* `n_samples`, `min_ess`: the run length and the requested ESS floor;
* `status`  : `:ok` or `:low_ess`;
* `low_ess` : keys with `ess < min_ess` (empty when `status == :ok`).
"""
struct SensitivityResult
    means::Dict{Symbol,Float64}
    se::Dict{Symbol,Float64}
    ess::Dict{Symbol,Float64}
    tau_int::Dict{Symbol,Float64}
    n_samples::Int
    min_ess::Float64
    status::Symbol
    low_ess::Vector{Symbol}
end

"""
Result of `binding_sensitivity`.

`means` is `mean complex - mean ligand + correction` (after the optional
charge-tangent projection), and `se` is the component-wise quadrature sum of
the two leg standard errors.  `complex` and `ligand` retain the full
`SensitivityResult`s, including their own ESS and status values.
`corrections` is the effective correction vector that was added (after tangent
projection, when one was requested).  `status` is `:low_ess` if either leg has
any component below `min_ess`.
"""
struct BindingSensitivityResult
    means::Dict{Symbol,Float64}
    se::Dict{Symbol,Float64}
    complex::SensitivityResult
    ligand::SensitivityResult
    corrections::Dict{Symbol,Float64}
    charge_tangent_groups::Vector{Vector{Symbol}}
    status::Symbol
    low_ess::Vector{Symbol}
end

# ---------------------------------------------------------------------------
# autocorrelation estimator
# ---------------------------------------------------------------------------

@inline function _lag_rho(x::AbstractVector{Float64}, μ::Float64,
                          γ0::Float64, lag::Int, n::Int)
    s = 0.0
    @inbounds for i in 1:(n - lag)
        s += (x[i] - μ) * (x[i + lag] - μ)
    end
    return s / (n * γ0)
end

"""
    _series_stats(x) -> (mean, tau_int, ess, se)

Biased-autocovariance, initial-positive/monotone integrated autocorrelation
estimate.  `x` must be a `Vector{Float64}` of length at least two.
"""
function _series_stats(x::Vector{Float64}; maxlag::Int = -1)
    n = length(x)
    n >= 2 || throw(ArgumentError("need at least two samples; got $n"))
    μ = sum(x) / n
    γ0 = 0.0
    @inbounds for i in 1:n
        d = x[i] - μ
        γ0 += d * d
    end
    γ0 /= n

    if γ0 == 0.0
        # Every sampled value is identical.  The sample mean is exact for the
        # sampled values, but the chain carries no fluctuation information.
        # ESS = n is the algebraic value for a constant series (τ_int = 1/2).
        return (mean = μ, tau_int = 0.5, ess = Float64(n), se = 0.0)
    end

    W = maxlag < 0 ? min(n - 1, 10 * max(1, n ÷ 100)) : min(maxlag, n - 1)
    W = max(W, 0)

    # Initial positive sequence in pairs Γ_t = ρ(2t) + ρ(2t+1), followed by a
    # nonincreasing envelope.  Stop at the first nonpositive pair.
    sumΓ = 0.0
    prevΓ = Inf
    anypair = false
    t = 0
    while 2 * t + 1 <= W
        r1 = t == 0 ? 1.0 : _lag_rho(x, μ, γ0, 2 * t, n)
        r2 = _lag_rho(x, μ, γ0, 2 * t + 1, n)
        Γ = r1 + r2
        Γ <= 0.0 && break
        if anypair
            Γ = min(Γ, prevΓ)
        end
        sumΓ += Γ
        prevΓ = Γ
        anypair = true
        t += 1
    end

    tau_int = anypair ? (-0.5 + sumΓ) : 0.5
    tau_int = max(tau_int, 0.5)
    ess = n / (2 * tau_int)
    se = sqrt(γ0 / ess)
    return (mean = μ, tau_int = tau_int, ess = ess, se = se)
end

"""
    integrated_autocorrelation_time(x; maxlag = -1) -> (tau_int, ess)

Return the integrated autocorrelation time and effective sample size for the
scalar series `x`, using the biased autocovariance and the initial-positive,
monotone estimator documented in the module.  `τ_int` is floored at `1/2`, so
`ess = length(x)/(2τ_int) <= length(x)`.
"""
function integrated_autocorrelation_time(x::AbstractVector{<:Real};
                                        maxlag::Int = -1)
    st = _series_stats(Float64.(x); maxlag = maxlag)
    return (tau_int = st.tau_int, ess = st.ess)
end

"""
    effective_sample_size(x; maxlag = -1)

`length(x)/(2*integrated_autocorrelation_time(x))`, capped at `length(x)` by
the `τ_int >= 1/2` floor.
"""
function effective_sample_size(x::AbstractVector{<:Real}; maxlag::Int = -1)
    return integrated_autocorrelation_time(x; maxlag = maxlag).ess
end

# ---------------------------------------------------------------------------
# gradient-sample normalisation
# ---------------------------------------------------------------------------

function _copy_gradient(g, sample_index::Int)
    g isa AbstractDict ||
        throw(ArgumentError("gradient sample $sample_index is not an AbstractDict; got $(typeof(g))"))
    d = Dict{Symbol,Float64}()
    for (k, v) in g
        k isa Symbol ||
            throw(ArgumentError("gradient sample $sample_index has non-Symbol key $(repr(k)); parameter ids must be Symbols"))
        v isa Real ||
            throw(ArgumentError("gradient sample $sample_index key $k has non-real value $(repr(v))"))
        fv = Float64(v)
        isfinite(fv) ||
            throw(ArgumentError("gradient sample $sample_index key $k is nonfinite ($fv)"))
        d[k] = fv
    end
    return d
end

function _gradient_dicts(samples::AbstractVector{<:AbstractDict}, ::Any)
    isempty(samples) &&
        throw(ArgumentError("gradient sample vector is empty"))
    out = Vector{Dict{Symbol,Float64}}(undef, length(samples))
    for (i, g) in enumerate(samples)
        out[i] = _copy_gradient(g, i)
    end
    return out
end

function _gradient_dicts(samples::AbstractMatrix, gradient_fn)
    gradient_fn === nothing &&
        throw(ArgumentError("gradient_fn is required when samples is a matrix"))
    n = size(samples, 2)
    n >= 1 || throw(ArgumentError("samples matrix must have at least one column"))
    out = Vector{Dict{Symbol,Float64}}(undef, n)
    for s in 1:n
        g = gradient_fn(view(samples, :, s))
        out[s] = _copy_gradient(g, s)
    end
    return out
end

# ---------------------------------------------------------------------------
# sensitivity estimation
# ---------------------------------------------------------------------------

function _estimate_from_dicts(dicts::Vector{Dict{Symbol,Float64}}; min_ess::Real)
    n = length(dicts)
    n >= 2 || throw(ArgumentError("need at least two samples; got $n"))
    min_ess_value = Float64(min_ess)
    isfinite(min_ess_value) && min_ess_value >= 0.0 ||
        throw(ArgumentError("min_ess must be finite and nonnegative; got $min_ess"))

    keyorder = sort!(collect(keys(dicts[1])))
    isempty(keyorder) &&
        throw(ArgumentError("gradient samples contain no parameter keys"))
    for (i, d) in enumerate(dicts)
        length(d) == length(keyorder) ||
            throw(ArgumentError("gradient sample $i has $(length(d)) keys; expected $(length(keyorder))"))
        for k in keyorder
            haskey(d, k) ||
                throw(ArgumentError("gradient sample $i is missing parameter key $k; all samples must expose the same key set"))
        end
    end

    means = Dict{Symbol,Float64}()
    ses = Dict{Symbol,Float64}()
    esss = Dict{Symbol,Float64}()
    taus = Dict{Symbol,Float64}()
    low = Symbol[]
    x = Vector{Float64}(undef, n)
    for k in keyorder
        @inbounds for i in 1:n
            x[i] = dicts[i][k]
        end
        st = _series_stats(x)
        means[k] = st.mean
        ses[k] = st.se
        esss[k] = st.ess
        taus[k] = st.tau_int
        st.ess < min_ess_value && push!(low, k)
    end
    status = isempty(low) ? :ok : :low_ess
    return SensitivityResult(means, ses, esss, taus, n, min_ess_value, status, low)
end

"""
    estimate_sensitivity(grad_samples; min_ess = 100)

Estimate the equilibrium fixed-measure parameter sensitivity from a vector of
per-sample gradient dictionaries.  See the module documentation for the
estimator, the biased-autocovariance convention, the initial-positive/monotone
truncation, `ESS = n/(2τ_int)`, and the per-component standard error
`SE = sqrt(γ(0)/ESS)`.
"""
function estimate_sensitivity(grad_samples::AbstractVector{<:AbstractDict};
                              min_ess::Real = 100)
    dicts = _gradient_dicts(grad_samples, nothing)
    return _estimate_from_dicts(dicts; min_ess = min_ess)
end

"""
    estimate_sensitivity(samples_matrix, gradient_fn; min_ess = 100)

As above, but `samples_matrix` has one configuration per column and
`gradient_fn(view(samples_matrix, :, s))` returns the per-sample gradient
dictionary.  A typical `gradient_fn` is a closure built by
`factor_gradient_function`, optionally wrapped to map a reduced coordinate to a
full `3 × N` configuration.
"""
function estimate_sensitivity(samples::AbstractMatrix, gradient_fn;
                              min_ess::Real = 100)
    dicts = _gradient_dicts(samples, gradient_fn)
    return _estimate_from_dicts(dicts; min_ess = min_ess)
end

# ---------------------------------------------------------------------------
# charge tangent projection
# ---------------------------------------------------------------------------

function _normalise_groups(groups)
    groups === nothing && return Vector{Vector{Symbol}}()
    if groups isa AbstractVector{Symbol}
        return [collect(groups)]
    elseif groups isa AbstractVector
        out = Vector{Vector{Symbol}}()
        for (gi, g) in enumerate(groups)
            g isa AbstractVector ||
                throw(ArgumentError("charge_tangent group $gi is not a vector of Symbols"))
            isempty(g) &&
                throw(ArgumentError("charge_tangent group $gi is empty; a group must contain at least one id"))
            gg = Symbol[]
            for k in g
                k isa Symbol ||
                    throw(ArgumentError("charge_tangent group $gi contains non-Symbol id $(repr(k))"))
                k in gg &&
                    throw(ArgumentError("charge_tangent group $gi contains duplicate id $k"))
                push!(gg, k)
            end
            push!(out, gg)
        end
        seen = Set{Symbol}()
        for (gi, gg) in enumerate(out)
            for k in gg
                k in seen &&
                    throw(ArgumentError("charge_tangent id $k appears in more than one group; groups must be disjoint"))
                push!(seen, k)
            end
        end
        return out
    end
    throw(ArgumentError("charge_tangent_groups must be a vector of Symbol vectors; got $(typeof(groups))"))
end

function _project_group_means!(grads::AbstractDict{Symbol,Float64},
                               groups::Vector{Vector{Symbol}})
    # Validate every group before mutating anything: a malformed later group
    # must not leave earlier groups half-projected in the caller's dictionary.
    for (gi, group) in enumerate(groups)
        isempty(group) &&
            throw(ArgumentError("charge_tangent group $gi is empty"))
        missing_keys = Symbol[]
        for k in group
            haskey(grads, k) || push!(missing_keys, k)
        end
        isempty(missing_keys) ||
            throw(ArgumentError("charge_tangent group $gi is missing gradient id(s) $(missing_keys)"))
    end
    for group in groups
        m = 0.0
        for k in group
            m += grads[k]
        end
        m /= length(group)
        for k in group
            grads[k] -= m
        end
    end
    return grads
end

"""
    project_charge_tangent!(grads, charge_tangent_groups)

Project the gradient dictionary `grads` in place onto the sum-zero tangent for
each supplied group: every member is replaced by `g_i - mean(group)`.  Groups
must be nonempty, disjoint, duplicate-free, and every id must be present in
`grads`.  An error is thrown otherwise; no missing id is treated as zero.
"""
function project_charge_tangent!(grads::AbstractDict{Symbol,Float64},
                                 charge_tangent_groups)
    groups = _normalise_groups(charge_tangent_groups)
    return _project_group_means!(grads, groups)
end

"""
    project_charge_tangent(grads, charge_tangent_groups)

Non-mutating variant of `project_charge_tangent!`; returns a new dictionary.
"""
function project_charge_tangent(grads::AbstractDict{Symbol,Float64},
                                charge_tangent_groups)
    return project_charge_tangent!(copy(grads), charge_tangent_groups)
end

# ---------------------------------------------------------------------------
# factor-protocol adapter
# ---------------------------------------------------------------------------

"""
    factor_gradient_function(factors, ctx; parameter_ids = ..., parameters = ...)

Return a closure `q -> Dict{Symbol,Float64}` that evaluates
`ParameterGradients.evaluate_parameter_gradients!` on `factors` at `q` with the
captured shared-id map and parameter overrides.  `q` may be a `3 × ctx.N`
matrix or a length-`3ctx.N` vector.  The returned dictionary is freshly
allocated on every call.

The `parameter_ids` and `parameters` dictionaries are copied when the closure is
built, so later mutation of the caller's dictionaries does not silently change
the estimator's parameterization.  The factor list itself is captured by
reference, matching the rest of the package's factor protocol.
"""
function factor_gradient_function(factors, ctx;
                                  parameter_ids = Dict{Tuple{Int,Symbol},Symbol}(),
                                  parameters = Dict{Symbol,Float64}())
    ids = Dict{Tuple{Int,Symbol},Symbol}(parameter_ids)
    params = Dict{Symbol,Float64}(parameters)
    n = ctx.N
    return function (q)
        local qm::Matrix{Float64}
        if q isa AbstractMatrix
            size(q) == (3, n) ||
                throw(ArgumentError("factor gradient expects a 3×$n configuration; got $(size(q))"))
            qm = Matrix{Float64}(q)
        elseif q isa AbstractVector
            length(q) == 3 * n ||
                throw(ArgumentError("factor gradient expects a length-$(3n) vector; got $(length(q))"))
            qm = reshape(Float64.(q), 3, n)
        else
            throw(ArgumentError("factor gradient expects a configuration matrix or vector; got $(typeof(q))"))
        end
        Fbuf = zeros(3, n)
        grads = Dict{Symbol,Float64}()
        evaluate_parameter_gradients!(Fbuf, grads, factors, qm, ctx;
                                      parameters = params,
                                      parameter_ids = ids)
        return grads
    end
end

# ---------------------------------------------------------------------------
# binding differences
# ---------------------------------------------------------------------------

function _select_leg_function(grad_fn, which::Int)
    grad_fn === nothing && return nothing
    if grad_fn isa Tuple
        length(grad_fn) == 2 ||
            throw(ArgumentError("grad_fn tuple must have exactly two entries: (complex_fn, ligand_fn)"))
        return grad_fn[which]
    end
    return grad_fn
end

_leg_dicts(samples::AbstractVector{<:AbstractDict}, gradient_fn) =
    _gradient_dicts(samples, nothing)

_leg_dicts(samples::AbstractMatrix, gradient_fn) =
    _gradient_dicts(samples, gradient_fn)

function _leg_dicts(samples, gradient_fn)
    throw(ArgumentError("samples must be a matrix (one configuration per column) or a vector of gradient dictionaries; got $(typeof(samples))"))
end

"""
    binding_sensitivity(complex_samples, ligand_samples, grad_fn = nothing;
                        corrections = Dict(),
                        charge_tangent_groups = [],
                        min_ess = 100)

Compute `mean complex - mean ligand + correction` with per-component,
autocorrelation-adjusted uncertainties.  `complex_samples` and
`ligand_samples` may each be a matrix of configurations (with a gradient
function) or a vector of precomputed gradient dictionaries.  If a matrix is
used, `grad_fn` may be one function shared by both legs or a tuple
`(complex_fn, ligand_fn)` when the two legs have different configuration
shapes.

Both legs must expose exactly the same parameter ids.  Corrections must use ids
already present in that shared set.  The two legs are treated as independent, so
`SE(difference) = sqrt(SE_complex² + SE_ligand²)`; corrections are exact
constants.  This routine performs only the requested subtraction.  It does not
assert receptor identity, does not add a standard-state correction, and does not
compute a physical binding free energy.
"""
function binding_sensitivity(complex_samples, ligand_samples, grad_fn = nothing;
                             corrections = Dict{Symbol,Float64}(),
                             charge_tangent_groups = Vector{Vector{Symbol}}(),
                             min_ess::Real = 100)
    groups = _normalise_groups(charge_tangent_groups)
    cdicts = _leg_dicts(complex_samples, _select_leg_function(grad_fn, 1))
    ldicts = _leg_dicts(ligand_samples, _select_leg_function(grad_fn, 2))
    if !isempty(groups)
        for d in cdicts
            _project_group_means!(d, groups)
        end
        for d in ldicts
            _project_group_means!(d, groups)
        end
    end
    c = _estimate_from_dicts(cdicts; min_ess = min_ess)
    l = _estimate_from_dicts(ldicts; min_ess = min_ess)

    ckeys = sort!(collect(keys(c.means)))
    lkeys = sort!(collect(keys(l.means)))
    ckeys == lkeys ||
        throw(ArgumentError("complex and ligand gradient key sets differ; shared parameter ids must match exactly (complex=$(ckeys), ligand=$(lkeys))"))

    corr = Dict{Symbol,Float64}()
    for (k, v) in corrections
        k isa Symbol ||
            throw(ArgumentError("correction key $(repr(k)) is not a Symbol"))
        v isa Real ||
            throw(ArgumentError("correction $k has non-real value $(repr(v))"))
        haskey(c.means, k) ||
            throw(ArgumentError("correction key $k is not present in the gradient key set $ckeys"))
        fv = Float64(v)
        isfinite(fv) ||
            throw(ArgumentError("correction $k is nonfinite ($fv)"))
        corr[k] = fv
    end

    if !isempty(groups)
        corr_full = Dict{Symbol,Float64}()
        for k in ckeys
            corr_full[k] = get(corr, k, 0.0)
        end
        _project_group_means!(corr_full, groups)
        corr = corr_full
    end

    means = Dict{Symbol,Float64}()
    ses = Dict{Symbol,Float64}()
    for k in ckeys
        means[k] = c.means[k] - l.means[k] + get(corr, k, 0.0)
        ses[k] = hypot(c.se[k], l.se[k])
    end
    low = sort!(unique(vcat(c.low_ess, l.low_ess)))
    status = isempty(low) ? :ok : :low_ess
    return BindingSensitivityResult(means, ses, c, l, corr, copy(groups),
                                    status, low)
end

end # module EquilibriumSensitivities
