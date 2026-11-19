"""
Lane C — synthetic MBAR.

Four harmonic states `U_k(x) = k x^2 / 2` with `k = 1, 2, 4, 8` and `beta = 1`
have exact reduced free energies `f_k = -log Z_k = -0.5 log(2 pi / k)`, so
`Delta f_{k,0} = 0.5 log k` with no approximation and no reference data.

The lane samples each state from its exact Gaussian (so MBAR's sampling
assumption is satisfied by construction), solves the MBAR fixed-point equation
with a log-sum-exp implementation, and reports:

* `max_abs_err` against the analytic difference,
* a nonparametric bootstrap standard error (resampling within each state),
* the fixed-point iteration count.

It also writes the reduced-potential matrix `u_kn`, the sample counts `N_k`, and
the Julia estimate to `VERIFICATION_ARTIFACTS` so `python/mbar_pymbar.py` can
solve the *same* dataset with pymbar and compare both solvers directly.
"""

using Random
using Statistics
using Printf

struct MbarMetrics
    delta_f_est::Vector{Float64}
    delta_f_exact::Vector{Float64}
    max_abs_err::Float64
    max_bootstrap_se::Float64
    iterations::Int
    n_samples::Int
    artifacts_dir::String
end

const MBAR_K = [1.0, 2.0, 4.0, 8.0]
const MBAR_N = 2000
const MBAR_SEED = 20261112

"Reduced potentials `u[k, n] = beta U_k(x_n)` and per-state counts."
function harmonic_dataset(kvals = MBAR_K, n::Int = MBAR_N; seed::Int = MBAR_SEED)
    rng = MersenneTwister(seed)
    K = length(kvals)
    x = Float64[]
    Nk = fill(n, K)
    for k in kvals
        append!(x, randn(rng, n) ./ sqrt(k))
    end
    u = Matrix{Float64}(undef, K, length(x))
    for k in 1:K, idx in 1:length(x)
        u[k, idx] = 0.5 * kvals[k] * x[idx]^2
    end
    return u, Nk, x
end

"Log-sum-exp of a finite-length iterable of reals."
function logsumexp(values)
    m = -Inf
    for v in values
        m = max(m, v)
    end
    isfinite(m) || return m
    s = 0.0
    for v in values
        s += exp(v - m)
    end
    return m + log(s)
end

"""
    mbar_free_energies(u, Nk; tol, maxiter) -> (f, iterations)

Self-consistent MBAR solution
`f_k = -log sum_n exp(-u_kn) / sum_j N_j exp(f_j - u_jn)`
with `f_1 = 0` gauge and log-sum-exp throughout.
"""
function mbar_free_energies(u::AbstractMatrix{Float64}, Nk::AbstractVector{<:Integer};
                            tol::Float64 = 1e-12, maxiter::Int = 20000)
    K, Ntot = size(u)
    length(Nk) == K || throw(ArgumentError("Nk must have one entry per state"))
    logN = log.(Float64.(Nk))
    f = zeros(K)
    denom = Vector{Float64}(undef, Ntot)
    for iter in 1:maxiter
        for n in 1:Ntot
            m = -Inf
            for j in 1:K
                m = max(m, logN[j] + f[j] - u[j, n])
            end
            s = 0.0
            for j in 1:K
                s += exp(logN[j] + f[j] - u[j, n] - m)
            end
            denom[n] = m + log(s)
        end
        fnew = Vector{Float64}(undef, K)
        for k in 1:K
            m = -Inf
            for n in 1:Ntot
                m = max(m, -u[k, n] - denom[n])
            end
            s = 0.0
            for n in 1:Ntot
                s += exp(-u[k, n] - denom[n] - m)
            end
            fnew[k] = -(m + log(s))
        end
        fnew .-= fnew[1]
        delta = maximum(abs, fnew .- f)
        f = fnew
        delta < tol && return f, iter
    end
    error("MBAR fixed point did not converge in $maxiter iterations")
end

analytic_delta_f(kvals = MBAR_K) = 0.5 .* log.(kvals)

"Nonparametric bootstrap standard error of `Delta f_{k,0}`."
function bootstrap_se(u::AbstractMatrix{Float64}, Nk::AbstractVector{<:Integer},
                     reps::Int = 100; seed::Int = MBAR_SEED + 1)
    K = size(u, 1)
    offsets = cumsum([0; collect(Nk)])
    rng = MersenneTwister(seed)
    draws = zeros(reps, K)
    for r in 1:reps
        idx = Int[]
        for k in 1:K
            state_idx = (offsets[k] + 1):offsets[k + 1]
            append!(idx, rand(rng, state_idx, Nk[k]))
        end
        fr, _ = mbar_free_energies(u[:, idx], Nk)
        draws[r, :] = fr .- fr[1]
    end
    return [std(view(draws, :, k)) for k in 1:K]
end

"Write the shared dataset so pymbar can solve exactly the same problem."
function write_artifacts(dir::String, u, Nk, f)
    mkpath(dir)
    open(joinpath(dir, "u_kn.csv"), "w") do io
        for k in 1:size(u, 1)
            println(io, join((@sprintf("%.17g", u[k, n]) for n in 1:size(u, 2)), ","))
        end
    end
    open(joinpath(dir, "n_k.csv"), "w") do io
        println(io, join(Nk, ","))
    end
    open(joinpath(dir, "f_julia.csv"), "w") do io
        println(io, join((@sprintf("%.17g", x) for x in f), ","))
    end
    open(joinpath(dir, "k_values.csv"), "w") do io
        println(io, join(MBAR_K, ","))
    end
    open(joinpath(dir, "meta.txt"), "w") do io
        println(io, "k_values=", join(MBAR_K, ","))
        println(io, "n_per_state=", MBAR_N)
        println(io, "seed=", MBAR_SEED)
        println(io, "beta=1")
        println(io, "potential=0.5*k*x^2")
    end
    return dir
end

function artifacts_dir()
    dir = get(ENV, "VERIFICATION_ARTIFACTS",
              joinpath(homedir(), ".cache", "biology-mm",
                       "verification-artifacts"))
    return mkpath(dir)
end

function run_mbar()
    u, Nk, _ = harmonic_dataset()
    f, iterations = mbar_free_energies(u, Nk)
    est = f .- f[1]
    exact = analytic_delta_f()
    se = bootstrap_se(u, Nk, 100)
    dir = write_artifacts(artifacts_dir(), u, Nk, f)
    return MbarMetrics(est, exact, maximum(abs, est .- exact), maximum(se),
                       iterations, sum(Nk), dir)
end
