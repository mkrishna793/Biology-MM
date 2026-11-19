"""Autocorrelation-aware statistics for the NPT validation experiment.

Every uncertainty quoted by `run.jl` is either an integrated-autocorrelation-time
(IAT) standard error or a block-average standard error. Naive `std/sqrt(n)` is
reported only as a diagnostic, because consecutive volume samples from a
Metropolis random walk are correlated and the naive error understates the true
one.

The IAT uses the standard Sokal window: `tau = 1/2 + sum_k rho_k`, truncated at
the smallest `K >= c*tau(K)` with `c = 5`. The unbiased autocovariance is
computed by FFT. `n_eff = n / (2*tau)`, and `se = std / sqrt(n_eff)`.
"""
module NPTAnalysis

using Statistics
using FFTW
using Printf

export tau_int, n_effective, mean_stderr, naive_stderr, block_means,
       block_stderr, block_variances, block_var_stderr, two_sample_ks,
       zscore, half_mean_agreement, linear_slope

"""
    tau_int(x; c = 5.0)

Integrated autocorrelation time of `x` in units of samples. Returns `0.5` for
white noise. Uses the initial-positive Sokal window with the automatic cutoff
`K >= c*tau(K)`.
"""
function tau_int(x::AbstractVector{<:Real}; c::Float64 = 5.0)
    n = length(x)
    n < 16 && return 0.5
    d = x .- mean(x)
    nfft = 2^ceil(Int, log2(2n))
    dp = zeros(Float64, nfft)
    dp[1:n] .= d
    F = fft(dp)
    ac = real(ifft(abs2.(F)))[1:n]
    ac ./= (n .- (0:n-1))          # unbiased autocovariance
    ac[1] <= 0 && return 0.5
    rho = ac ./ ac[1]
    tau = 0.5
    for k in 1:n-1
        r = rho[k+1]
        r <= 0 && break
        tau += r
        k >= c * tau && break
    end
    return max(tau, 0.5)            # tau >= 1/2 by construction
end

n_effective(x) = max(1.0, length(x) / (2 * tau_int(x)))

"Autocorrelation-corrected standard error of the mean."
mean_stderr(x) = std(x) / sqrt(n_effective(x))

"Naive `std/sqrt(n)`; reported only to show how much correlation inflates the error."
naive_stderr(x) = std(x) / sqrt(length(x))

"Split `x` into `nb` contiguous, nearly equal blocks and return their means."
function block_means(x::AbstractVector{<:Real}, nb::Integer)
    n = length(x)
    nb >= 1 || throw(ArgumentError("nb must be positive"))
    nb <= n || throw(ArgumentError("nb=$nb exceeds sample count $n"))
    edges = round.(Int, range(1, n + 1; length = nb + 1))
    return [mean(@view x[edges[i]:edges[i+1]-1]) for i in 1:nb]
end

"Standard error of the mean from `nb` block means (blocks must exceed the correlation time)."
block_stderr(x, nb::Integer) = std(block_means(x, nb)) / sqrt(nb)

"Per-block sample variances."
function block_variances(x::AbstractVector{<:Real}, nb::Integer)
    n = length(x)
    edges = round.(Int, range(1, n + 1; length = nb + 1))
    return [var(@view x[edges[i]:edges[i+1]-1]) for i in 1:nb]
end

"Standard error of the variance from block variances."
block_var_stderr(x, nb::Integer) = std(block_variances(x, nb)) / sqrt(nb)

"""
    two_sample_ks(a, b)

Two-sample Kolmogorov–Smirnov statistic. The 5% critical value is
`1.36*sqrt((na+nb)/(na*nb))`; a value below it is consistent with the same
distribution.
"""
function two_sample_ks(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    sa, sb = sort(a), sort(b)
    na, nb = length(sa), length(sb)
    i = j = 1
    d = 0.0
    while i <= na || j <= nb
        if j > nb || (i <= na && sa[i] <= sb[j])
            v = sa[i]; i += 1
        else
            v = sb[j]; j += 1
        end
        while i <= na && sa[i] == v; i += 1; end
        while j <= nb && sb[j] == v; j += 1; end
        d = max(d, abs(i - 1) / na - abs(j - 1) / nb)
    end
    return d
end

"Standardised deviation `(value - reference) / se`."
zscore(value, reference, se) = abs(value - reference) / se

"""
    half_mean_agreement(x)

Compare the mean of the first and second halves. Returns
`(first, second, difference, combined_se, z)`.
"""
function half_mean_agreement(x::AbstractVector{<:Real})
    n = length(x)
    h = div(n, 2)
    a, b = @view(x[1:h]), @view(x[h+1:2h])
    sa, sb = mean_stderr(a), mean_stderr(b)
    diff = mean(a) - mean(b)
    se = sqrt(sa^2 + sb^2)
    return (first = mean(a), second = mean(b), difference = diff,
            se = se, z = abs(diff) / se)
end

"Least-squares slope of `y` against its index, with its standard error."
function linear_slope(y::AbstractVector{<:Real})
    n = length(y)
    x = collect(1.0:n)
    xm, ym = mean(x), mean(y)
    sxx = sum((x .- xm) .^ 2)
    slope = sum((x .- xm) .* (y .- ym)) / sxx
    intercept = ym - slope * xm
    resid = y .- (intercept .+ slope .* x)
    dof = max(n - 2, 1)
    s2 = sum(abs2, resid) / dof
    se = sqrt(s2 / sxx)
    return (slope = slope, se = se, z = abs(slope) / se)
end

end # module NPTAnalysis
