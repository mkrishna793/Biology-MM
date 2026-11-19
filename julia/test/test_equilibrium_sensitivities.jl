# Tests for EquilibriumSensitivities: correlated-sample means, ESS, standard
# errors, fixed-measure harmonic derivative, binding differences, and charge
# tangent projection.
#
# Policy: every statistical claim is compared with an independently known value
# (closed form, direct estimator, or a separately computed dictionary route),
# and every low-ESS or malformed-input case is required to raise or flag, never
# to return silent certainty.
#
# Standalone: `EquilibriumSensitivities` is not yet included by package glue, so
# this file loads it against FoundationCore's Types/Factors/ParameterGradients,
# exactly like `test_parameter_gradients.jl` does.

using FoundationCore
using FoundationCore.Types
using FoundationCore.Factors
using FoundationCore.ParameterGradients
using Random, Statistics, Printf

module _EQHost
    using ..FoundationCore
    if !isdefined(FoundationCore, :EquilibriumSensitivities)
        include(joinpath(@__DIR__, "..", "src", "EquilibriumSensitivities.jl"))
    end
end

const ES = isdefined(FoundationCore, :EquilibriumSensitivities) ?
           FoundationCore.EquilibriumSensitivities : _EQHost.EquilibriumSensitivities
const T = FoundationCore.Types
const F = FoundationCore.Factors
const PG = FoundationCore.ParameterGradients

function expect_argument_error(f, label)
    ok = false
    try
        f()
    catch e
        ok = e isa ArgumentError
        ok || println("  unexpected exception for ", label, ": ", typeof(e), " ", e)
    end
    @assert ok "$label: expected ArgumentError"
    return true
end

function ar1_series(rng::AbstractRNG, n::Int, phi::Float64)
    x = Vector{Float64}(undef, n)
    x[1] = randn(rng)
    factor = sqrt(1.0 - phi^2)
    for i in 2:n
        x[i] = phi * x[i - 1] + factor * randn(rng)
    end
    return x
end

function q_of_bond(x::Real)
    q = zeros(3, 2)
    q[1, 2] = x
    return q
end

# ---------------------------------------------------------------------------
# 1. Independent samples: ESS ~ n, standard error ~ iid formula.
# ---------------------------------------------------------------------------
function test_iid_ess_and_mean()
    n = 20_000
    rng = MersenneTwister(7)
    samples = reshape(randn(rng, n), 1, n)
    res = ES.estimate_sensitivity(samples, x -> Dict(:g => x[1]); min_ess = 100)
    naive_se = std(vec(samples)) / sqrt(n)

    @printf("  iid: mean=%+.5f se=%.5f (iid %.5f) ess=%.0f/%.0f tau=%.4f status=%s\n",
            res.means[:g], res.se[:g], naive_se, res.ess[:g], n,
            res.tau_int[:g], res.status)
    @assert res.status == :ok
    @assert 0.90 * n <= res.ess[:g] <= n + 1e-9
    @assert 0.45 <= res.tau_int[:g] <= 0.60
    @assert abs(res.means[:g]) < 5 * res.se[:g]
    @assert 0.8 <= res.se[:g] / naive_se <= 1.2
    return true
end

# ---------------------------------------------------------------------------
# 2. AR(1): the estimator must see correlation and reduce ESS, with the
#    theoretical ESS/N = (1-phi)/(1+phi) as the reference.
# ---------------------------------------------------------------------------
function test_ar1_ess()
    n = 20_000
    phi = 0.8
    theory_ess = n * (1 - phi) / (1 + phi)
    seeds = (11, 12, 13, 14, 15)
    ratios = Float64[]
    first_res = nothing
    first_naive = 0.0
    for (i, seed) in enumerate(seeds)
        rng = MersenneTwister(seed)
        x = ar1_series(rng, n, phi)
        res = ES.estimate_sensitivity([Dict(:g => x[j]) for j in 1:n];
                                      min_ess = 100)
        naive_se = std(x) / sqrt(n)
        push!(ratios, res.ess[:g] / theory_ess)
        if i == 1
            first_res = res
            first_naive = naive_se
            tau = ES.integrated_autocorrelation_time(x)
            @assert tau.ess == res.ess[:g]
        end
        @assert res.ess[:g] < n / 2
        @assert abs(res.ess[:g] / theory_ess - 1) < 0.35
        @assert res.se[:g] > 1.5 * naive_se
        @assert res.status == :ok
    end

    @printf("  AR(1) phi=%.2f, %d seeds: ess/theory in [%.3f, %.3f]\n",
            phi, length(seeds), minimum(ratios), maximum(ratios))
    @printf("    seed %d: ess=%.0f (theory %.0f) tau=%.3f se=%.5f naive=%.5f\n",
            seeds[1], first_res.ess[:g], theory_ess, first_res.tau_int[:g],
            first_res.se[:g], first_naive)
    return true
end

# ---------------------------------------------------------------------------
# 3. Low ESS must be an explicit status, not a quiet number.
# ---------------------------------------------------------------------------
function test_low_ess_status()
    n = 2_000
    phi = 0.97
    rng = MersenneTwister(13)
    x = ar1_series(rng, n, phi)
    grads = [Dict(:g => x[i]) for i in 1:n]
    res = ES.estimate_sensitivity(grads; min_ess = 100)
    res_relaxed = ES.estimate_sensitivity(grads; min_ess = 1)

    @printf("  low ESS: ess=%.1f (min_ess=100) status=%s low=%s; relaxed=%s\n",
            res.ess[:g], res.status, res.low_ess, res_relaxed.status)
    @assert res.ess[:g] < 100
    @assert res.status == :low_ess
    @assert :g in res.low_ess
    @assert isfinite(res.se[:g]) && res.se[:g] > 0
    @assert res_relaxed.status == :ok
    return true
end

# ---------------------------------------------------------------------------
# 4. Exact harmonic free-energy derivative: closed form + central difference;
#    Monte Carlo gradient mean from 50k deterministic independent Gaussian draws.
# ---------------------------------------------------------------------------
function test_exact_harmonic_sensitivity()
    beta = 1.0
    k0 = 4.0
    n = 50_000
    sigma = sqrt(1.0 / (beta * k0))
    rng = MersenneTwister(3)
    x = sigma .* randn(rng, n)

    # A(k) = -1/beta log sqrt(2*pi/(beta*k)); additive constants do not matter.
    A(k) = -log(sqrt(2.0 * pi / (beta * k))) / beta
    exact_derivative = 1.0 / (2.0 * beta * k0)
    h = 1e-6
    fd_derivative = (A(k0 + h) - A(k0 - h)) / (2h)
    fd_rel = abs(fd_derivative - exact_derivative) /
             max(abs(exact_derivative), 1.0)

    res = ES.estimate_sensitivity(reshape(x, 1, n),
                                  z -> Dict(:bond_k => 0.5 * z[1]^2);
                                  min_ess = 100)
    exact_se = sqrt(sigma^4 / (2n))   # Var(x^2/2) = sigma^4/2 for a Gaussian
    zscore = (res.means[:bond_k] - exact_derivative) / res.se[:bond_k]

    @printf("  harmonic: dA/dk exact=%.8f fd=%.8f rel=%.2e\n",
            exact_derivative, fd_derivative, fd_rel)
    @printf("            MC mean=%.8f se=%.8f (exact se %.8f) z=%+.2f ess=%.0f\n",
            res.means[:bond_k], res.se[:bond_k], exact_se, zscore, res.ess[:bond_k])
    @assert fd_rel < 1e-8
    @assert abs(res.means[:bond_k] - exact_derivative) < 5 * res.se[:bond_k]
    @assert 0.8 <= res.se[:bond_k] / exact_se <= 1.2
    @assert res.status == :ok
    return true
end

# ---------------------------------------------------------------------------
# 5. Factor-protocol adapter: same values as the direct evaluator, analytic
#    bond derivatives, vector/matrix input, and matrix-vs-dict estimator paths.
# ---------------------------------------------------------------------------
function test_factor_gradient_function()
    ctx = F.Context(T.REDUCED, 2)
    factors = [F.HarmonicBond(1, 2, 3.0, 0.4)]
    ids = Dict{Tuple{Int,Symbol},Symbol}((1, :k) => :bond_k,
                                          (1, :r0) => :bond_r0)
    params = Dict{Symbol,Float64}(:bond_k => 4.0, :bond_r0 => 0.4)
    fg = ES.factor_gradient_function(factors, ctx;
                                     parameter_ids = ids, parameters = params)

    q = q_of_bond(1.3)
    g = fg(q)
    @assert haskey(g, :bond_k) && haskey(g, :bond_r0)
    @assert abs(g[:bond_k] - 0.5 * (1.3 - 0.4)^2) < 1e-14
    @assert abs(g[:bond_r0] - (-4.0 * (1.3 - 0.4))) < 1e-14

    Fv = zeros(3, 2)
    direct = Dict{Symbol,Float64}()
    U = PG.evaluate_parameter_gradients!(Fv, direct, factors, q, ctx;
                                          parameters = params, parameter_ids = ids)
    @assert abs(U - 0.5 * 4.0 * (1.3 - 0.4)^2) < 1e-14
    @assert direct == g
    @assert fg(vec(q)) == g

    expect_argument_error(() -> fg(zeros(3, 3)), "factor gradient wrong matrix size")
    expect_argument_error(() -> fg(zeros(5)), "factor gradient wrong vector size")

    # The matrix path calls gradient_fn per column; the dict path gets exactly
    # the same dictionaries.  Both must produce identical statistics.
    n = 3_000
    rng = MersenneTwister(31)
    xs = 1.3 .+ 0.4 .* randn(rng, n)
    samples = reshape(xs, 1, n)
    matrix_result = ES.estimate_sensitivity(samples,
                                            z -> fg(q_of_bond(z[1]));
                                            min_ess = 0)
    dicts = [fg(q_of_bond(xs[i])) for i in 1:n]
    dict_result = ES.estimate_sensitivity(dicts; min_ess = 0)
    for key in (:bond_k, :bond_r0)
        @assert matrix_result.means[key] == dict_result.means[key]
        @assert matrix_result.se[key] == dict_result.se[key]
        @assert matrix_result.ess[key] == dict_result.ess[key]
        @assert matrix_result.tau_int[key] == dict_result.tau_int[key]
    end

    @printf("  factor closure: direct == closure; matrix path == dict path (%d samples)\n", n)
    return true
end

# ---------------------------------------------------------------------------
# 6. Binding difference: exact arithmetic relation, quadrature SE, corrections,
#    and explicit rejection of mismatched keys.
# ---------------------------------------------------------------------------
function test_binding_difference()
    n = 5_000
    rng = MersenneTwister(21)
    c = [Dict(:p => 2.0 + randn(rng), :q => -0.5 + 0.3 * randn(rng))
         for _ in 1:n]
    l = [Dict(:p => 1.0 + randn(rng), :q => 0.3 + 0.2 * randn(rng))
         for _ in 1:n]
    rc = ES.estimate_sensitivity(c; min_ess = 100)
    rl = ES.estimate_sensitivity(l; min_ess = 100)
    corrections = Dict{Symbol,Float64}(:p => 0.25)
    rb = ES.binding_sensitivity(c, l; corrections = corrections, min_ess = 100)

    @assert rb.means[:p] ≈ rc.means[:p] - rl.means[:p] + 0.25
    @assert rb.means[:q] ≈ rc.means[:q] - rl.means[:q]
    @assert rb.se[:p] ≈ hypot(rc.se[:p], rl.se[:p])
    @assert rb.se[:q] ≈ hypot(rc.se[:q], rl.se[:q])
    @assert rb.corrections[:p] == 0.25
    @assert rb.status == :ok
    @printf("  binding: delta_p=%.5f se=%.5f (legs %.5f, %.5f)\n",
            rb.means[:p], rb.se[:p], rc.se[:p], rl.se[:p])

    bad_keys = [Dict(:p => 1.0) for _ in 1:n]
    expect_argument_error(() -> ES.binding_sensitivity(c, bad_keys),
                          "binding key-set mismatch")
    expect_argument_error(() -> ES.binding_sensitivity(c, l;
                                                       corrections = Dict(:nope => 1.0)),
                          "binding correction key absent")
    return true
end

function test_binding_tuple_gradient_functions()
    n = 3_000
    rng = MersenneTwister(22)
    cs = reshape(randn(rng, n), 1, n)
    ls = reshape(randn(rng, n), 1, n)
    gc = z -> Dict(:p => z[1]^2)
    gl = z -> Dict(:p => 0.5 * z[1]^2)

    tuple_result = ES.binding_sensitivity(cs, ls, (gc, gl); min_ess = 0)
    cdicts = [gc(cs[:, i]) for i in 1:n]
    ldicts = [gl(ls[:, i]) for i in 1:n]
    dict_result = ES.binding_sensitivity(cdicts, ldicts; min_ess = 0)

    @assert tuple_result.means[:p] == dict_result.means[:p]
    @assert tuple_result.se[:p] == dict_result.se[:p]
    @printf("  binding tuple grad_fn: %.6f +- %.6f\n",
            tuple_result.means[:p], tuple_result.se[:p])
    return true
end

# ---------------------------------------------------------------------------
# 7. Charge tangent: non-mutating and mutating forms, exact sum-zero result,
#    and explicit rejection of malformed groups.
# ---------------------------------------------------------------------------
function test_charge_tangent()
    g = Dict{Symbol,Float64}(:q1 => 1.0, :q2 => 2.0, :q3 => 4.0, :other => 9.0)
    original = copy(g)
    p = ES.project_charge_tangent(g, [[:q1, :q2, :q3]])
    @assert g == original
    @assert abs(p[:q1] + p[:q2] + p[:q3]) < 1e-14
    @assert p[:other] == 9.0
    @assert abs(p[:q1] - (1.0 - 7.0 / 3.0)) < 1e-14

    inplace = copy(g)
    ES.project_charge_tangent!(inplace, [:q1, :q2, :q3])
    @assert abs(inplace[:q1] + inplace[:q2] + inplace[:q3]) < 1e-14

    expect_argument_error(() -> ES.project_charge_tangent(g, [Symbol[]]),
                          "empty tangent group")
    expect_argument_error(() -> ES.project_charge_tangent(g, [[:q1, :missing]]),
                          "missing tangent id")
    expect_argument_error(() -> ES.project_charge_tangent(g, [[:q1, :q1]]),
                          "duplicate tangent id")
    expect_argument_error(() -> ES.project_charge_tangent(g, [[:q1, :q2], [:q2, :q3]]),
                          "overlapping tangent groups")

    # Same projection inside a binding difference, including correction.
    n = 2_000
    rng = MersenneTwister(41)
    c = [Dict(:q1 => 1.0 + 0.1randn(rng), :q2 => 2.0 + 0.1randn(rng),
              :q3 => 3.0 + 0.1randn(rng), :other => randn(rng)) for _ in 1:n]
    l = [Dict(:q1 => 0.5 + 0.1randn(rng), :q2 => 1.0 + 0.1randn(rng),
              :q3 => 1.5 + 0.1randn(rng), :other => randn(rng)) for _ in 1:n]
    rb = ES.binding_sensitivity(c, l;
                                corrections = Dict(:q1 => 0.9),
                                charge_tangent_groups = [[:q1, :q2, :q3]],
                                min_ess = 0)
    tangent_sum = rb.means[:q1] + rb.means[:q2] + rb.means[:q3]
    corr_sum = rb.corrections[:q1] + rb.corrections[:q2] + rb.corrections[:q3]

    @printf("  tangent: sum after projection=%.2e; correction sum=%.2e\n",
            tangent_sum, corr_sum)
    @assert abs(tangent_sum) < 1e-12
    @assert abs(corr_sum) < 1e-12
    @assert rb.means[:other] ≈ rb.complex.means[:other] - rb.ligand.means[:other]
    return true
end

# ---------------------------------------------------------------------------
# 8. Malformed sample input is rejected rather than guessed at.
# ---------------------------------------------------------------------------
function test_input_validation()
    expect_argument_error(() -> ES.estimate_sensitivity(Dict{Symbol,Float64}[]),
                          "empty gradient sample vector")
    expect_argument_error(() -> ES.estimate_sensitivity([Dict(:g => 1.0)]),
                          "single-sample gradient vector")
    expect_argument_error(() -> ES.estimate_sensitivity(zeros(1, 0),
                                                        x -> Dict(:g => x[1])),
                          "zero-column sample matrix")
    expect_argument_error(() -> ES.estimate_sensitivity(
                              [Dict(:g => 1.0), Dict(:g => 2.0, :h => 3.0)]),
                          "inconsistent gradient keys")
    expect_argument_error(() -> ES.estimate_sensitivity(
                              [Dict(:g => 1.0), Dict(:g => NaN)]),
                          "nonfinite gradient value")
    expect_argument_error(() -> ES.estimate_sensitivity(
                              [Dict("g" => 1.0), Dict("g" => 2.0)]),
                          "non-Symbol gradient key")
    println("  malformed sample input: explicit ArgumentError in all cases")
    return true
end

function _run_all()
    tests = [
        ("iid_ess_and_mean", test_iid_ess_and_mean),
        ("ar1_ess", test_ar1_ess),
        ("low_ess_status", test_low_ess_status),
        ("exact_harmonic_sensitivity", test_exact_harmonic_sensitivity),
        ("factor_gradient_function", test_factor_gradient_function),
        ("binding_difference", test_binding_difference),
        ("binding_tuple_gradient_functions", test_binding_tuple_gradient_functions),
        ("charge_tangent", test_charge_tangent),
        ("input_validation", test_input_validation),
    ]
    for (name, fn) in tests
        println("TEST ", name)
        fn()
    end
    println("")
    println("ALL EQUILIBRIUM-SENSITIVITY TESTS PASSED")
end

_run_all()
