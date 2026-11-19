"""
`EnhancedSampling` — collective variables, umbrella sampling, well-tempered
metadynamics, log-space WHAM unbiasing, and replica-exchange MD (REMD).

All collective variables implement exact analytic Cartesian gradients (`∇_q s`)
and exact fixed-fractional box derivatives (`∂s/∂h` at fixed fractional
coordinates `s = h⁻¹ q`) across `NoBox`, `OrthorhombicBox`, and `TriclinicBox`.
`UmbrellaFactor` and `MetadynamicsBias` subtype `Factors.AbstractFactor` and
integrate directly with `Dynamics` (Velocity Verlet / BAOAB Langevin) and
`SystemContract.box_derivative`.
"""
module EnhancedSampling

using LinearAlgebra
using Random
using Statistics
using ..Types
using ..Factors
using ..Dynamics
import ..ResolvedSystem

export AbstractCV,
       cv_value_and_grad!, cv_box_derivative!,
       is_periodic, cv_period, cv_diff, verify_cv_gradient,
       DistanceCV, CoordinationCV, DihedralCV, ComDistanceCV, CustomCV,
       UmbrellaFactor, MetadynamicsBias,
       bias_potential, deposit_hill!, reconstruct_free_energy,
       unbias_umbrella_wham, remd_exchange_attempt!

# ---------------------------------------------------------------------------
# Internal periodic / fractional-coordinate helpers
# ---------------------------------------------------------------------------

@inline function _frac_disp(box::Types.OrthorhombicBox, d::Types.SVector3)
    L = box.L
    return (d.x / L[1], d.y / L[2], d.z / L[3])
end

@inline function _frac_disp(box::Types.TriclinicBox, d::Types.SVector3)
    M = box.Hinv
    return (M[1, 1] * d.x + M[1, 2] * d.y + M[1, 3] * d.z,
            M[2, 1] * d.x + M[2, 2] * d.y + M[2, 3] * d.z,
            M[3, 1] * d.x + M[3, 2] * d.y + M[3, 3] * d.z)
end

@inline function _frac_disp(::Types.NoBox, ::Types.SVector3)
    throw(Types.UnsupportedGeometry(
        "box derivative requires a periodic box (OrthorhombicBox or TriclinicBox), got NoBox"))
end

@inline _vadd(a::NTuple{3,Float64}, b::NTuple{3,Float64}) =
    (a[1] + b[1], a[2] + b[2], a[3] + b[3])

@inline function _outer_add!(G::Matrix{Float64}, c::Float64,
                             d::Types.SVector3, w::NTuple{3,Float64})
    gx = c * d.x; gy = c * d.y; gz = c * d.z
    @inbounds begin
        G[1, 1] += gx * w[1]; G[1, 2] += gx * w[2]; G[1, 3] += gx * w[3]
        G[2, 1] += gy * w[1]; G[2, 2] += gy * w[2]; G[2, 3] += gy * w[3]
        G[3, 1] += gz * w[1]; G[3, 2] += gz * w[2]; G[3, 3] += gz * w[3]
    end
    return G
end

@inline function _check_q_and_grad(grad::AbstractMatrix{Float64}, q::AbstractMatrix{Float64})
    size(q, 1) == 3 || throw(ArgumentError("positions q must be 3 x N, got $(size(q))"))
    size(grad) == size(q) ||
        throw(ArgumentError("grad shape $(size(grad)) must match q shape $(size(q))"))
    return size(q, 2)
end

@inline function _check_box_mat(G::Matrix{Float64})
    size(G) == (3, 3) || throw(ArgumentError("box derivative G must be 3 x 3, got $(size(G))"))
    return nothing
end

# ---------------------------------------------------------------------------
# Collective Variable Protocol
# ---------------------------------------------------------------------------

"""
    AbstractCV

Abstract supertype for collective variables `s(q, box)`.

Subtypes implement:
- [`cv_value_and_grad!`](@ref)
- [`cv_box_derivative!`](@ref)
- [`is_periodic`](@ref) and [`cv_period`](@ref) (optional, defaults to non-periodic)
"""
abstract type AbstractCV end

"""
    cv_value_and_grad!(grad::AbstractMatrix{Float64}, cv::AbstractCV,
                       q::AbstractMatrix{Float64}, box::Types.AbstractBox) -> Float64

Zero `grad`, compute the collective variable value `s(q, box)` and write its
Cartesian gradient `∇_q s` into `grad`, returning `s`.
"""
function cv_value_and_grad! end

"""
    cv_box_derivative!(G::Matrix{Float64}, cv::AbstractCV, dUds::Float64,
                       q::AbstractMatrix{Float64}, box::Types.AbstractBox) -> Matrix{Float64}

Accumulate `G[a, b] += dUds * ∂s/∂h[a, b]` at fixed fractional coordinates into `G`.
"""
function cv_box_derivative! end

"True if the collective variable lives on a periodic domain of period `cv_period(cv)`."
is_periodic(::AbstractCV) = false

"Period of a periodic collective variable (0.0 for non-periodic CVs)."
cv_period(::AbstractCV) = 0.0

"""
    cv_diff(cv::AbstractCV, s1::Real, s2::Real) -> Float64

Minimum-image cyclic difference `s1 - s2` when `is_periodic(cv)` is `true`,
otherwise plain `Float64(s1) - Float64(s2)`.
"""
@inline function cv_diff(cv::AbstractCV, s1::Real, s2::Real)
    d = Float64(s1) - Float64(s2)
    if is_periodic(cv)
        P = cv_period(cv)
        (isfinite(P) && P > 0.0) || throw(ArgumentError("periodic CV must have positive finite period, got $P"))
        d -= P * round(d / P)
    end
    return d
end

# ---------------------------------------------------------------------------
# Concrete CVs
# ---------------------------------------------------------------------------

"""
    DistanceCV(i::Int, j::Int)

Minimum-image Euclidean distance `r = ||unwrap_displacement(box, q, i, j)||`.
"""
struct DistanceCV <: AbstractCV
    i::Int
    j::Int
    function DistanceCV(i::Int, j::Int)
        (i >= 1 && j >= 1 && i != j) ||
            throw(ArgumentError("DistanceCV requires distinct positive atom indices, got ($i, $j)"))
        return new(i, j)
    end
end

function cv_value_and_grad!(grad::AbstractMatrix{Float64}, cv::DistanceCV,
                            q::AbstractMatrix{Float64}, box::Types.AbstractBox)
    N = _check_q_and_grad(grad, q)
    (cv.i <= N && cv.j <= N) ||
        throw(BoundsError(q, (1, max(cv.i, cv.j))))
    fill!(grad, 0.0)
    d = Types.unwrap_displacement(box, q, cv.i, cv.j)
    r2 = Types.norm2(d)
    r = sqrt(r2)
    r < 1e-12 && throw(Types.InvalidGeometry("DistanceCV: coincident atoms $(cv.i) and $(cv.j)"))
    inv_r = 1.0 / r
    @inbounds begin
        grad[1, cv.i] = inv_r * d.x
        grad[2, cv.i] = inv_r * d.y
        grad[3, cv.i] = inv_r * d.z
        grad[1, cv.j] = -inv_r * d.x
        grad[2, cv.j] = -inv_r * d.y
        grad[3, cv.j] = -inv_r * d.z
    end
    return r
end

function cv_box_derivative!(G::Matrix{Float64}, cv::DistanceCV, dUds::Float64,
                            q::AbstractMatrix{Float64}, box::Types.AbstractBox)
    _check_box_mat(G)
    size(q, 1) == 3 || throw(ArgumentError("positions q must be 3 x N"))
    d = Types.unwrap_displacement(box, q, cv.i, cv.j)
    r = sqrt(Types.norm2(d))
    r < 1e-12 && throw(Types.InvalidGeometry("DistanceCV: coincident atoms $(cv.i) and $(cv.j)"))
    w = _frac_disp(box, d)
    _outer_add!(G, dUds / r, d, w)
    return G
end

"""
    CoordinationCV(group_a::Vector{Int}, group_b::Vector{Int}, r0::Float64; n::Int=6, m::Int=12)

Smooth rational switching-function coordination number:

    S = Σ_{i ∈ A, j ∈ B, i ≠ j} (1 - (r_ij / r0)^n) / (1 - (r_ij / r0)^m)

with `1 ≤ n < m`. Uses exact finite geometric-series cancellation
`(1 + x + … + x^(n-1)) / (1 + x + … + x^(m-1))` and quotient-rule derivative
for `|x - 1| < 1e-4` to eliminate the removable singularity at `r = r0`.
"""
struct CoordinationCV <: AbstractCV
    group_a::Vector{Int}
    group_b::Vector{Int}
    r0::Float64
    n::Int
    m::Int
    function CoordinationCV(group_a::AbstractVector{<:Integer},
                            group_b::AbstractVector{<:Integer},
                            r0::Real; n::Int = 6, m::Int = 12)
        ga = Int.(collect(group_a))
        gb = Int.(collect(group_b))
        r0f = Float64(r0)
        (!isempty(ga) && !isempty(gb)) ||
            throw(ArgumentError("CoordinationCV requires non-empty groups"))
        all(>=(1), ga) && all(>=(1), gb) ||
            throw(ArgumentError("CoordinationCV atom indices must be >= 1"))
        (isfinite(r0f) && r0f > 0.0) ||
            throw(ArgumentError("CoordinationCV r0 must be finite and positive, got $r0"))
        (n >= 1 && m > n) ||
            throw(ArgumentError("CoordinationCV requires 1 <= n < m, got n=$n, m=$m"))
        return new(ga, gb, r0f, n, m)
    end
end

@inline function _switching_val_and_deriv(x::Float64, n::Int, m::Int)
    if abs(x - 1.0) < 1e-4
        # Exact removable-singularity cancellation:
        # (1 - x^n) / (1 - x^m) = P_n(x) / Q_m(x), P_n(x) = Σ_{k=0}^{n-1} x^k
        s_sum = 1.0
        ds_sum = 0.0
        xk_prev = 1.0
        pn = 1.0
        dpn = 0.0
        @inbounds for k in 1:(m - 1)
            ds_sum += k * xk_prev
            xk = xk_prev * x
            s_sum += xk
            xk_prev = xk
            if k == n - 1
                pn = s_sum
                dpn = ds_sum
            end
        end
        qm = s_sum
        dqm = ds_sum
        val = pn / qm
        dval_dx = (dpn * qm - pn * dqm) / (qm * qm)
        return val, dval_dx
    elseif x == 0.0
        return 1.0, (n == 1 ? -1.0 : 0.0)
    elseif x > 1.0 && m * log(x) > 500.0
        lval = (n - m) * log(x)
        lval < -745.0 && return 0.0, 0.0
        val = exp(lval)
        dval_dx = (n - m) * exp(lval - log(x))
        return val, dval_dx
    else
        xm1 = x^(n - 1)
        xn = xm1 * x
        ym1 = x^(m - 1)
        ym = ym1 * x
        num = 1.0 - xn
        den = 1.0 - ym
        val = num / den
        dval_dx = (m * ym1 * num - n * xm1 * den) / (den * den)
        return val, dval_dx
    end
end

function cv_value_and_grad!(grad::AbstractMatrix{Float64}, cv::CoordinationCV,
                            q::AbstractMatrix{Float64}, box::Types.AbstractBox)
    N = _check_q_and_grad(grad, q)
    (maximum(cv.group_a) <= N && maximum(cv.group_b) <= N) ||
        throw(BoundsError(q, (1, max(maximum(cv.group_a), maximum(cv.group_b)))))
    fill!(grad, 0.0)
    inv_r0 = 1.0 / cv.r0
    S = 0.0
    @inbounds for i in cv.group_a, j in cv.group_b
        i == j && continue
        d = Types.unwrap_displacement(box, q, i, j)
        r = sqrt(Types.norm2(d))
        x = r * inv_r0
        val, dval_dx = _switching_val_and_deriv(x, cv.n, cv.m)
        S += val
        if r < 1e-12
            cv.n == 1 && throw(Types.InvalidGeometry("CoordinationCV with n=1 has cusp at r=0 for ($i, $j)"))
            continue
        end
        c = (dval_dx * inv_r0) / r
        gx = c * d.x; gy = c * d.y; gz = c * d.z
        grad[1, i] += gx; grad[2, i] += gy; grad[3, i] += gz
        grad[1, j] -= gx; grad[2, j] -= gy; grad[3, j] -= gz
    end
    return S
end

function cv_box_derivative!(G::Matrix{Float64}, cv::CoordinationCV, dUds::Float64,
                            q::AbstractMatrix{Float64}, box::Types.AbstractBox)
    _check_box_mat(G)
    size(q, 1) == 3 || throw(ArgumentError("positions q must be 3 x N"))
    inv_r0 = 1.0 / cv.r0
    @inbounds for i in cv.group_a, j in cv.group_b
        i == j && continue
        d = Types.unwrap_displacement(box, q, i, j)
        r = sqrt(Types.norm2(d))
        x = r * inv_r0
        _, dval_dx = _switching_val_and_deriv(x, cv.n, cv.m)
        if r < 1e-12
            cv.n == 1 && throw(Types.InvalidGeometry("CoordinationCV with n=1 has cusp at r=0 for ($i, $j)"))
            continue
        end
        w = _frac_disp(box, d)
        c = dUds * (dval_dx * inv_r0) / r
        _outer_add!(G, c, d, w)
    end
    return G
end

"""
    DihedralCV(i::Int, j::Int, k::Int, l::Int)

Proper torsion angle `φ ∈ (-π, π]` around bond `j-k`, wrapping
`Factors.dihedral_angle_grad`. Periodic with `cv_period = 2π`.
"""
struct DihedralCV <: AbstractCV
    i::Int
    j::Int
    k::Int
    l::Int
    function DihedralCV(i::Int, j::Int, k::Int, l::Int)
        (i >= 1 && j >= 1 && k >= 1 && l >= 1) ||
            throw(ArgumentError("DihedralCV atom indices must be >= 1"))
        (i != j && i != k && i != l && j != k && j != l && k != l) ||
            throw(ArgumentError("DihedralCV requires 4 distinct atom indices, got ($i, $j, $k, $l)"))
        return new(i, j, k, l)
    end
end

is_periodic(::DihedralCV) = true
cv_period(::DihedralCV) = 2.0 * π

function cv_value_and_grad!(grad::AbstractMatrix{Float64}, cv::DihedralCV,
                            q::AbstractMatrix{Float64}, box::Types.AbstractBox)
    N = _check_q_and_grad(grad, q)
    max(cv.i, cv.j, cv.k, cv.l) <= N ||
        throw(BoundsError(q, (1, max(cv.i, cv.j, cv.k, cv.l))))
    fill!(grad, 0.0)
    t = Factors.dihedral_angle_grad(q, box, cv.i, cv.j, cv.k, cv.l)
    t === nothing && throw(Types.InvalidGeometry(
        "DihedralCV: degenerate (collinear) dihedral geometry for atoms ($(cv.i), $(cv.j), $(cv.k), $(cv.l))"))
    phi, (gi, gj, gk, gl) = t
    @inbounds begin
        grad[1, cv.i] = gi.x; grad[2, cv.i] = gi.y; grad[3, cv.i] = gi.z
        grad[1, cv.j] = gj.x; grad[2, cv.j] = gj.y; grad[3, cv.j] = gj.z
        grad[1, cv.k] = gk.x; grad[2, cv.k] = gk.y; grad[3, cv.k] = gk.z
        grad[1, cv.l] = gl.x; grad[2, cv.l] = gl.y; grad[3, cv.l] = gl.z
    end
    return phi
end

function cv_box_derivative!(G::Matrix{Float64}, cv::DihedralCV, dUds::Float64,
                            q::AbstractMatrix{Float64}, box::Types.AbstractBox)
    _check_box_mat(G)
    size(q, 1) == 3 || throw(ArgumentError("positions q must be 3 x N"))
    t = Factors.dihedral_angle_grad(q, box, cv.i, cv.j, cv.k, cv.l)
    if t === nothing
        G .= Inf
        return G
    end
    _, (_, gj, gk, gl) = t
    w1 = _frac_disp(box, Types.unwrap_displacement(box, q, cv.j, cv.i))
    w2 = _frac_disp(box, Types.unwrap_displacement(box, q, cv.k, cv.j))
    w3 = _frac_disp(box, Types.unwrap_displacement(box, q, cv.l, cv.k))
    s_k = _vadd(w1, w2)
    s_l = _vadd(s_k, w3)
    _outer_add!(G, dUds, gj, w1)
    _outer_add!(G, dUds, gk, s_k)
    _outer_add!(G, dUds, gl, s_l)
    return G
end

"""
    ComDistanceCV(group_a::Vector{Int}, group_b::Vector{Int}, masses::Vector{Float64})

Minimum-image distance between centers of mass of `group_a` and `group_b`.
Each group is unwrapped contiguously relative to its anchor (`group_a[1]` and
`group_b[1]`) before forming the COMs `R_a, R_b`, followed by minimum-image
displacement `d = unwrap_displacement(box, hcat(R_a, R_b), 1, 2)`.
"""
struct ComDistanceCV <: AbstractCV
    group_a::Vector{Int}
    group_b::Vector{Int}
    masses::Vector{Float64}
    function ComDistanceCV(group_a::AbstractVector{<:Integer},
                           group_b::AbstractVector{<:Integer},
                           masses::AbstractVector{<:Real})
        ga = Int.(collect(group_a))
        gb = Int.(collect(group_b))
        ms = Float64.(collect(masses))
        (!isempty(ga) && !isempty(gb)) ||
            throw(ArgumentError("ComDistanceCV requires non-empty groups"))
        all(m -> isfinite(m) && m > 0.0, ms) ||
            throw(ArgumentError("ComDistanceCV masses must be finite and positive"))
        Nm = length(ms)
        all(i -> 1 <= i <= Nm, ga) && all(j -> 1 <= j <= Nm, gb) ||
            throw(ArgumentError("ComDistanceCV atom indices must lie in 1:$Nm"))
        return new(ga, gb, ms)
    end
end

function _com_disp(cv::ComDistanceCV, q::AbstractMatrix{Float64}, box::Types.AbstractBox)
    a0 = cv.group_a[1]
    Ma = 0.0
    dx_a = 0.0; dy_a = 0.0; dz_a = 0.0
    @inbounds for i in cv.group_a
        mi = cv.masses[i]
        Ma += mi
        if i != a0
            u = Types.unwrap_displacement(box, q, i, a0)
            dx_a += mi * u.x; dy_a += mi * u.y; dz_a += mi * u.z
        end
    end
    @inbounds Ra = [q[1, a0] + dx_a / Ma, q[2, a0] + dy_a / Ma, q[3, a0] + dz_a / Ma]

    b0 = cv.group_b[1]
    Mb = 0.0
    dx_b = 0.0; dy_b = 0.0; dz_b = 0.0
    @inbounds for j in cv.group_b
        mj = cv.masses[j]
        Mb += mj
        if j != b0
            u = Types.unwrap_displacement(box, q, j, b0)
            dx_b += mj * u.x; dy_b += mj * u.y; dz_b += mj * u.z
        end
    end
    @inbounds Rb = [q[1, b0] + dx_b / Mb, q[2, b0] + dy_b / Mb, q[3, b0] + dz_b / Mb]

    R_ab = hcat(Ra, Rb)
    d = Types.unwrap_displacement(box, R_ab, 1, 2)
    return d, Ma, Mb
end

function cv_value_and_grad!(grad::AbstractMatrix{Float64}, cv::ComDistanceCV,
                            q::AbstractMatrix{Float64}, box::Types.AbstractBox)
    N = _check_q_and_grad(grad, q)
    max(maximum(cv.group_a), maximum(cv.group_b)) <= N ||
        throw(BoundsError(q, (1, max(maximum(cv.group_a), maximum(cv.group_b)))))
    fill!(grad, 0.0)
    d, Ma, Mb = _com_disp(cv, q, box)
    r = sqrt(Types.norm2(d))
    r < 1e-12 && throw(Types.InvalidGeometry("ComDistanceCV: coincident group centers of mass"))
    inv_r = 1.0 / r
    @inbounds for i in cv.group_a
        w = (cv.masses[i] / Ma) * inv_r
        grad[1, i] += w * d.x
        grad[2, i] += w * d.y
        grad[3, i] += w * d.z
    end
    @inbounds for j in cv.group_b
        w = (cv.masses[j] / Mb) * inv_r
        grad[1, j] -= w * d.x
        grad[2, j] -= w * d.y
        grad[3, j] -= w * d.z
    end
    return r
end

function cv_box_derivative!(G::Matrix{Float64}, cv::ComDistanceCV, dUds::Float64,
                            q::AbstractMatrix{Float64}, box::Types.AbstractBox)
    _check_box_mat(G)
    size(q, 1) == 3 || throw(ArgumentError("positions q must be 3 x N"))
    d, _, _ = _com_disp(cv, q, box)
    r = sqrt(Types.norm2(d))
    r < 1e-12 && throw(Types.InvalidGeometry("ComDistanceCV: coincident group centers of mass"))
    w = _frac_disp(box, d)
    _outer_add!(G, dUds / r, d, w)
    return G
end

"""
    CustomCV(val_fn, grad_fn, support::Vector{Int};
             box_deriv_fn=nothing, periodic::Bool=false, period::Float64=0.0)

User-defined collective variable wrapping `val_fn(q, box) -> Float64` and
`grad_fn(grad, q, box)` (or `grad_fn(q, box) -> Matrix{Float64}`).
"""
struct CustomCV{F1,F2,F3} <: AbstractCV
    val_fn::F1
    grad_fn::F2
    support::Vector{Int}
    box_deriv_fn::F3
    periodic::Bool
    period::Float64
    function CustomCV(val_fn::F1, grad_fn::F2, support::AbstractVector{<:Integer};
                      box_deriv_fn::F3 = nothing,
                      periodic::Bool = false, period::Real = 0.0) where {F1,F2,F3}
        sup = Int.(collect(support))
        all(>=(1), sup) || throw(ArgumentError("CustomCV support indices must be >= 1"))
        pf = Float64(period)
        if periodic
            (isfinite(pf) && pf > 0.0) ||
                throw(ArgumentError("periodic CustomCV requires finite positive period, got $period"))
        end
        return new{F1,F2,F3}(val_fn, grad_fn, sup, box_deriv_fn, periodic, pf)
    end
end

is_periodic(cv::CustomCV) = cv.periodic
cv_period(cv::CustomCV) = cv.period

function cv_value_and_grad!(grad::AbstractMatrix{Float64}, cv::CustomCV,
                            q::AbstractMatrix{Float64}, box::Types.AbstractBox)
    _check_q_and_grad(grad, q)
    fill!(grad, 0.0)
    val = Float64(cv.val_fn(q, box))
    if applicable(cv.grad_fn, grad, q, box)
        cv.grad_fn(grad, q, box)
    elseif applicable(cv.grad_fn, q, box)
        g = cv.grad_fn(q, box)
        size(g) == size(grad) ||
            throw(DimensionMismatch("CustomCV grad_fn returned size $(size(g)), expected $(size(grad))"))
        grad .= g
    else
        throw(MethodError(cv.grad_fn, (grad, q, box)))
    end
    return val
end

function cv_box_derivative!(G::Matrix{Float64}, cv::CustomCV, dUds::Float64,
                            q::AbstractMatrix{Float64}, box::Types.AbstractBox)
    _check_box_mat(G)
    cv.box_deriv_fn === nothing && throw(Types.UnsupportedGeometry(
        "CustomCV has no box_deriv_fn implemented"))
    box isa Types.NoBox && throw(Types.UnsupportedGeometry(
        "box derivative requires a periodic box, got NoBox"))
    fn = cv.box_deriv_fn
    if applicable(fn, G, dUds, q, box)
        fn(G, dUds, q, box)
    elseif applicable(fn, q, box)
        dG = fn(q, box)
        size(dG) == (3, 3) || throw(DimensionMismatch("box_deriv_fn must return 3x3 matrix"))
        G .+= dUds .* dG
    elseif applicable(fn, dUds, q, box)
        dG = fn(dUds, q, box)
        size(dG) == (3, 3) || throw(DimensionMismatch("box_deriv_fn must return 3x3 matrix"))
        G .+= dG
    else
        throw(MethodError(fn, (G, dUds, q, box)))
    end
    return G
end

"""
    verify_cv_gradient(cv::AbstractCV, q::AbstractMatrix{Float64}, box::Types.AbstractBox;
                       h::Float64=1e-6, tol::Float64=1e-7)

Check analytic `cv_value_and_grad!` against central finite differences using
`cv_diff` (handling periodic CVs cleanly). Returns a NamedTuple
`(passed, max_error, analytic, fd)`.
"""
function verify_cv_gradient(cv::AbstractCV, q::AbstractMatrix{Float64}, box::Types.AbstractBox;
                            h::Float64 = 1e-6, tol::Float64 = 1e-7)
    (isfinite(h) && h > 0.0) || throw(ArgumentError("h must be finite and positive"))
    (isfinite(tol) && tol > 0.0) || throw(ArgumentError("tol must be finite and positive"))
    N = size(q, 2)
    analytic = zeros(Float64, 3, N)
    cv_value_and_grad!(analytic, cv, q, box)

    fd = zeros(Float64, 3, N)
    q_work = copy(q)
    g_tmp = zeros(Float64, 3, N)
    @inbounds for j in 1:N, d in 1:3
        orig = q_work[d, j]
        q_work[d, j] = orig + h
        sp = cv_value_and_grad!(g_tmp, cv, q_work, box)
        q_work[d, j] = orig - h
        sm = cv_value_and_grad!(g_tmp, cv, q_work, box)
        q_work[d, j] = orig
        fd[d, j] = cv_diff(cv, sp, sm) / (2.0 * h)
    end
    max_err = maximum(abs, analytic .- fd)
    return (passed = max_err <= tol, max_error = max_err, analytic = analytic, fd = fd)
end

# ---------------------------------------------------------------------------
# Biased Factors (<: Factors.AbstractFactor)
# ---------------------------------------------------------------------------

"""
    UmbrellaFactor(cv::AbstractCV, k::Float64, s0::Float64, n_atoms::Int)

Harmonic umbrella restraint `U = 0.5 * k * cv_diff(cv, s, s0)^2`.
"""
struct UmbrellaFactor{C<:AbstractCV} <: Factors.AbstractFactor
    cv::C
    k::Float64
    s0::Float64
    n_atoms::Int
    function UmbrellaFactor(cv::C, k::Real, s0::Real, n_atoms::Int) where {C<:AbstractCV}
        kf = Float64(k); s0f = Float64(s0)
        (isfinite(kf) && kf >= 0.0) || throw(ArgumentError("UmbrellaFactor k must be finite and >= 0"))
        isfinite(s0f) || throw(ArgumentError("UmbrellaFactor s0 must be finite"))
        n_atoms >= 1 || throw(ArgumentError("UmbrellaFactor n_atoms must be >= 1"))
        return new{C}(cv, kf, s0f, n_atoms)
    end
end

Factors.factor_support_size(f::UmbrellaFactor) = f.n_atoms
Factors.read_support(f::UmbrellaFactor) = Tuple(1:f.n_atoms)
Factors.is_local(::UmbrellaFactor) = false
Factors.force_consistent(::UmbrellaFactor) = true

function Factors.evaluate!(F::AbstractMatrix{Float64}, f::UmbrellaFactor,
                           q::AbstractMatrix{Float64}, ctx::Factors.Context)
    grad = zeros(Float64, 3, size(q, 2))
    s = cv_value_and_grad!(grad, f.cv, q, ctx.box)
    ds = cv_diff(f.cv, s, f.s0)
    U = 0.5 * f.k * ds * ds
    dUds = f.k * ds
    @inbounds for j in axes(q, 2), d in 1:3
        F[d, j] -= dUds * grad[d, j]
    end
    return U
end

function ResolvedSystem.accumulate_box_derivative!(G::Matrix{Float64}, f::UmbrellaFactor,
                                                   q::AbstractMatrix{Float64},
                                                   ctx::Factors.Context)
    grad = zeros(Float64, 3, size(q, 2))
    s = cv_value_and_grad!(grad, f.cv, q, ctx.box)
    ds = cv_diff(f.cv, s, f.s0)
    dUds = f.k * ds
    cv_box_derivative!(G, f.cv, dUds, q, ctx.box)
    return G
end

function ResolvedSystem.box_derivative_support(f::UmbrellaFactor)
    if f.cv isa CustomCV && f.cv.box_deriv_fn === nothing
        return "CustomCV has no box_deriv_fn"
    end
    return nothing
end

"""
    MetadynamicsBias(cv, n_atoms; height=1.0, sigma=0.1, bias_factor=Inf,
                     temperature=1.0, kB=1.0, centers=Float64[],
                     heights=Float64[], sigmas=Float64[])

Well-Tempered (or standard, when `bias_factor = Inf`) Gaussian metadynamics
history-dependent bias potential along `cv`.

When the deposited hill archive `(centers, heights, sigmas)` is fixed between
hill depositions, `force_consistent` is `true` and Verlet / BAOAB integrate
with exact energy-consistent forces `-dV/ds * ∇_q s`.
"""
struct MetadynamicsBias{C<:AbstractCV} <: Factors.AbstractFactor
    cv::C
    n_atoms::Int
    height::Float64
    sigma::Float64
    bias_factor::Float64
    temperature::Float64
    kB::Float64
    centers::Vector{Float64}
    heights::Vector{Float64}
    sigmas::Vector{Float64}
    function MetadynamicsBias(cv::C, n_atoms::Int, height::Real, sigma::Real,
                              bias_factor::Real, temperature::Real, kB::Real,
                              centers::AbstractVector{<:Real},
                              heights::AbstractVector{<:Real},
                              sigmas::AbstractVector{<:Real}) where {C<:AbstractCV}
        n_atoms >= 1 || throw(ArgumentError("MetadynamicsBias n_atoms must be >= 1"))
        hf = Float64(height); sf = Float64(sigma)
        gf = Float64(bias_factor); Tf = Float64(temperature); kBf = Float64(kB)
        (isfinite(hf) && hf >= 0.0) || throw(ArgumentError("height must be finite and >= 0"))
        (isfinite(sf) && sf > 0.0) || throw(ArgumentError("sigma must be finite and > 0"))
        (gf > 1.0 && !isnan(gf)) || throw(ArgumentError("bias_factor must be > 1.0 (or Inf)"))
        (isfinite(Tf) && Tf > 0.0) || throw(ArgumentError("temperature must be finite and > 0"))
        (isfinite(kBf) && kBf > 0.0) || throw(ArgumentError("kB must be finite and > 0"))
        cs = Float64.(collect(centers))
        hs = Float64.(collect(heights))
        ss = Float64.(collect(sigmas))
        (length(cs) == length(hs) == length(ss)) ||
            throw(ArgumentError("centers, heights, and sigmas must have identical length"))
        all(isfinite, cs) || throw(ArgumentError("hill centers must be finite"))
        all(isfinite, hs) || throw(ArgumentError("hill heights must be finite"))
        all(x -> isfinite(x) && x > 0.0, ss) || throw(ArgumentError("hill sigmas must be finite and > 0"))
        return new{C}(cv, n_atoms, hf, sf, gf, Tf, kBf, cs, hs, ss)
    end
end

function MetadynamicsBias(cv::AbstractCV, n_atoms::Int;
                          height::Real = 1.0,
                          sigma::Real = 0.1,
                          bias_factor::Real = Inf,
                          temperature::Real = 1.0,
                          kB::Real = 1.0,
                          centers::AbstractVector{<:Real} = Float64[],
                          heights::AbstractVector{<:Real} = Float64[],
                          sigmas::AbstractVector{<:Real} = Float64[])
    return MetadynamicsBias(cv, n_atoms, height, sigma, bias_factor, temperature, kB,
                            centers, heights, sigmas)
end

Factors.factor_support_size(f::MetadynamicsBias) = f.n_atoms
Factors.read_support(f::MetadynamicsBias) = Tuple(1:f.n_atoms)
Factors.is_local(::MetadynamicsBias) = false
Factors.force_consistent(::MetadynamicsBias) = true

@inline function _bias_val_and_deriv(bias::MetadynamicsBias, s::Float64)
    V = 0.0
    dVds = 0.0
    @inbounds for k in eachindex(bias.centers)
        ds = cv_diff(bias.cv, s, bias.centers[k])
        sig = bias.sigmas[k]
        inv_s2 = 1.0 / (sig * sig)
        g = bias.heights[k] * exp(-0.5 * ds * ds * inv_s2)
        V += g
        dVds -= g * ds * inv_s2
    end
    return V, dVds
end

"Evaluate the deposited metadynamics bias `V(s)` at collective variable value `s`."
function bias_potential(bias::MetadynamicsBias, s::Real)
    sf = Float64(s)
    V = 0.0
    @inbounds for k in eachindex(bias.centers)
        ds = cv_diff(bias.cv, sf, bias.centers[k])
        sig = bias.sigmas[k]
        V += bias.heights[k] * exp(-0.5 * (ds * ds) / (sig * sig))
    end
    return V
end

"""
    deposit_hill!(bias::MetadynamicsBias, s::Real) -> Float64
    deposit_hill!(bias::MetadynamicsBias, q::AbstractMatrix{Float64}, box::Types.AbstractBox) -> Float64

Deposit a Gaussian hill at the current CV value `s`, using Well-Tempered
rescaling `w = w0 * exp(-V(s) / (kB * T * (γ - 1)))` (or `w0` when
`isinf(bias_factor)`). Returns the deposited hill height `w`.
"""
function deposit_hill!(bias::MetadynamicsBias, s::Real)
    sf = Float64(s)
    V = bias_potential(bias, sf)
    w = if isinf(bias.bias_factor)
        bias.height
    else
        dT = bias.kB * bias.temperature * (bias.bias_factor - 1.0)
        bias.height * exp(-V / dT)
    end
    push!(bias.centers, sf)
    push!(bias.heights, w)
    push!(bias.sigmas, bias.sigma)
    return w
end

function deposit_hill!(bias::MetadynamicsBias, q::AbstractMatrix{Float64}, box::Types.AbstractBox)
    grad = zeros(Float64, 3, size(q, 2))
    s = cv_value_and_grad!(grad, bias.cv, q, box)
    return deposit_hill!(bias, s)
end

function Factors.evaluate!(F::AbstractMatrix{Float64}, f::MetadynamicsBias,
                           q::AbstractMatrix{Float64}, ctx::Factors.Context)
    isempty(f.centers) && return 0.0
    grad = zeros(Float64, 3, size(q, 2))
    s = cv_value_and_grad!(grad, f.cv, q, ctx.box)
    V, dVds = _bias_val_and_deriv(f, s)
    @inbounds for j in axes(q, 2), d in 1:3
        F[d, j] -= dVds * grad[d, j]
    end
    return V
end

function ResolvedSystem.accumulate_box_derivative!(G::Matrix{Float64}, f::MetadynamicsBias,
                                                   q::AbstractMatrix{Float64},
                                                   ctx::Factors.Context)
    isempty(f.centers) && return G
    grad = zeros(Float64, 3, size(q, 2))
    s = cv_value_and_grad!(grad, f.cv, q, ctx.box)
    _, dVds = _bias_val_and_deriv(f, s)
    cv_box_derivative!(G, f.cv, dVds, q, ctx.box)
    return G
end

function ResolvedSystem.box_derivative_support(f::MetadynamicsBias)
    if f.cv isa CustomCV && f.cv.box_deriv_fn === nothing
        return "CustomCV has no box_deriv_fn"
    end
    return nothing
end

"""
    reconstruct_free_energy(bias::MetadynamicsBias, s_grid::AbstractVector{Float64}) -> Vector{Float64}

Reconstruct the free-energy profile along `s_grid` from accumulated hills:
`F(s) = -(γ / (γ - 1)) * V(s)` (or `-V(s)` for `isinf(bias_factor)`),
shifted so `minimum(F) == 0.0`.
"""
function reconstruct_free_energy(bias::MetadynamicsBias, s_grid::AbstractVector{<:Real})
    isempty(s_grid) && throw(ArgumentError("s_grid must be non-empty"))
    scale = isinf(bias.bias_factor) ? -1.0 : -(bias.bias_factor / (bias.bias_factor - 1.0))
    F = Float64[scale * bias_potential(bias, Float64(s)) for s in s_grid]
    F .-= minimum(F)
    return F
end

# ---------------------------------------------------------------------------
# Log-space Unbinned WHAM Unbiasing
# ---------------------------------------------------------------------------

@inline function _scalar_diff(s1::Float64, s2::Float64, periodic::Bool, period::Float64)
    d = s1 - s2
    if periodic
        d -= period * round(d / period)
    end
    return d
end

@inline function _logsumexp_vec(v::AbstractVector{Float64})
    m = maximum(v)
    isfinite(m) || return m
    s = 0.0
    @inbounds for x in v
        s += exp(x - m)
    end
    return m + log(s)
end

"""
    unbias_umbrella_wham(centers::Vector{Float64}, ks::Vector{Float64},
                         samples::Vector{Vector{Float64}}, beta::Float64;
                         s_grid::AbstractVector{Float64},
                         max_iter::Int=10_000, tol::Float64=1e-9,
                         periodic::Bool=false, period::Float64=2pi)

Solve the exact unbinned log-space self-consistent WHAM / MBAR equations across
umbrella windows `i = 1..K` with bias `0.5 * ks[i] * d(s, centers[i])^2`, then
evaluate a smooth log-space weighted Gaussian KDE PMF on `s_grid` shifted so
`minimum(pmf) == 0.0`.

Returns `(s_grid, pmf, log_Z, weights, converged, iterations)`.
"""
function unbias_umbrella_wham(centers::AbstractVector{<:Real},
                              ks::AbstractVector{<:Real},
                              samples::AbstractVector{<:AbstractVector{<:Real}},
                              beta::Real;
                              s_grid::AbstractVector{<:Real},
                              max_iter::Int = 10_000,
                              tol::Float64 = 1e-9,
                              periodic::Bool = false,
                              period::Real = 2.0 * π)
    K = length(centers)
    K >= 1 || throw(ArgumentError("WHAM requires at least 1 umbrella window"))
    length(ks) == K && length(samples) == K ||
        throw(ArgumentError("centers, ks, and samples must have equal length K=$K"))
    bf = Float64(beta)
    (isfinite(bf) && bf > 0.0) || throw(ArgumentError("beta must be finite and positive"))
    !isempty(s_grid) || throw(ArgumentError("s_grid must be non-empty"))
    max_iter >= 1 || throw(ArgumentError("max_iter must be >= 1"))
    (isfinite(tol) && tol > 0.0) || throw(ArgumentError("tol must be finite and positive"))
    pf = Float64(period)
    if periodic
        (isfinite(pf) && pf > 0.0) || throw(ArgumentError("period must be finite and positive"))
    end

    cs = Float64.(collect(centers))
    k_vec = Float64.(collect(ks))
    all(isfinite, cs) || throw(ArgumentError("centers must be finite"))
    all(x -> isfinite(x) && x >= 0.0, k_vec) || throw(ArgumentError("spring constants ks must be finite and >= 0"))

    counts = Int[length(s) for s in samples]
    all(>(0), counts) || throw(ArgumentError("every umbrella window must have at least 1 sample"))
    N_total = sum(counts)
    log_N = Float64[log(Float64(c)) for c in counts]

    all_s = Vector{Float64}(undef, N_total)
    idx = 1
    for win in samples
        for v in win
            fv = Float64(v)
            isfinite(fv) || throw(ArgumentError("samples must be finite"))
            all_s[idx] = fv
            idx += 1
        end
    end

    # Precompute -beta * u_i(s_n) for all windows i and samples n
    neg_beta_u = Matrix{Float64}(undef, K, N_total)
    @inbounds for n in 1:N_total
        sn = all_s[n]
        for i in 1:K
            d = _scalar_diff(sn, cs[i], periodic, pf)
            neg_beta_u[i, n] = -0.5 * bf * k_vec[i] * d * d
        end
    end

    log_Z = zeros(Float64, K)
    log_Z_new = zeros(Float64, K)
    log_w = Vector{Float64}(undef, N_total)
    col_buf = Vector{Float64}(undef, K)
    row_buf = Vector{Float64}(undef, N_total)

    converged = false
    iterations = 0

    for iter in 1:max_iter
        iterations = iter
        # Step 1: log w_n = -logsumexp_j(log N_j - log Z_j - beta * u_j(s_n))
        @inbounds for n in 1:N_total
            for j in 1:K
                col_buf[j] = log_N[j] - log_Z[j] + neg_beta_u[j, n]
            end
            log_w[n] = -_logsumexp_vec(col_buf)
        end
        lw_norm = _logsumexp_vec(log_w)
        log_w .-= lw_norm

        # Step 2: log Z_i = logsumexp_n(log w_n - beta * u_i(s_n))
        @inbounds for i in 1:K
            for n in 1:N_total
                row_buf[n] = log_w[n] + neg_beta_u[i, n]
            end
            log_Z_new[i] = _logsumexp_vec(row_buf)
        end
        z0 = log_Z_new[1]
        log_Z_new .-= z0

        max_diff = 0.0
        @inbounds for i in 1:K
            diff_i = abs(log_Z_new[i] - log_Z[i])
            diff_i > max_diff && (max_diff = diff_i)
            log_Z[i] = log_Z_new[i]
        end

        if max_diff < tol
            converged = true
            break
        end
    end

    # Recompute exact normalized sample weights consistent with final log_Z
    @inbounds for n in 1:N_total
        for j in 1:K
            col_buf[j] = log_N[j] - log_Z[j] + neg_beta_u[j, n]
        end
        log_w[n] = -_logsumexp_vec(col_buf)
    end
    log_w .-= _logsumexp_vec(log_w)
    weights = exp.(log_w)

    # Smooth log-space weighted Gaussian KDE onto s_grid
    s_vec = Float64.(collect(s_grid))
    h_grid = if length(s_vec) > 1
        diffs = filter(>(0.0), diff(sort(s_vec)))
        isempty(diffs) ? 0.05 : median(diffs)
    else
        0.05
    end
    span = periodic ? pf : max(maximum(s_vec) - minimum(s_vec), h_grid)
    h = max(0.8 * h_grid, 0.5 * span / sqrt(max(N_total, 100)))
    inv_2h2 = 0.5 / (h * h)

    pmf = Vector{Float64}(undef, length(s_vec))
    @inbounds for g in eachindex(s_vec)
        sg = s_vec[g]
        for n in 1:N_total
            d = _scalar_diff(sg, all_s[n], periodic, pf)
            row_buf[n] = log_w[n] - d * d * inv_2h2
        end
        pmf[g] = -_logsumexp_vec(row_buf) / bf
    end
    pmf .-= minimum(pmf)

    return (s_grid = s_vec, pmf = pmf, log_Z = log_Z, weights = weights,
            converged = converged, iterations = iterations)
end

# ---------------------------------------------------------------------------
# Replica Exchange Molecular Dynamics (REMD)
# ---------------------------------------------------------------------------

"""
    remd_exchange_attempt!(state_a::Dynamics.State, state_b::Dynamics.State,
                           U_a::Float64, U_b::Float64,
                           beta_a::Float64, beta_b::Float64,
                           rng::AbstractRNG) -> Bool

Attempt a temperature Replica Exchange (REMD) swap between two `Dynamics.State`
replicas at inverse temperatures `beta_a, beta_b` with potential energies `U_a, U_b`.

Metropolis acceptance probability:

    P_acc = min(1, exp((beta_a - beta_b) * (U_a - U_b)))

When accepted:
1. Coordinates `state_a.q` and `state_b.q` are swapped in-place.
2. Momenta are swapped and temperature-rescaled:
   `p_a_new = sqrt(beta_b / beta_a) * p_b_old`,
   `p_b_new = sqrt(beta_a / beta_b) * p_a_old`.
3. `force_valid` is marked `false` on both states.

Returns `true` if accepted, `false` otherwise.
"""
function remd_exchange_attempt!(state_a::Dynamics.State, state_b::Dynamics.State,
                                U_a::Real, U_b::Real,
                                beta_a::Real, beta_b::Real,
                                rng::AbstractRNG)
    ua = Float64(U_a); ub = Float64(U_b)
    ba = Float64(beta_a); bb = Float64(beta_b)
    (isfinite(ua) && isfinite(ub)) || throw(ArgumentError("energies U_a, U_b must be finite"))
    (isfinite(ba) && ba > 0.0 && isfinite(bb) && bb > 0.0) ||
        throw(ArgumentError("inverse temperatures beta_a, beta_b must be finite and positive"))
    size(state_a.q) == size(state_b.q) && size(state_a.p) == size(state_b.p) ||
        throw(DimensionMismatch("replica states must have matching shapes"))

    delta = (ba - bb) * (ua - ub)
    accepted = delta >= 0.0 || rand(rng) < exp(delta)
    accepted || return false

    scale_b_to_a = sqrt(bb / ba)
    scale_a_to_b = sqrt(ba / bb)
    @inbounds for idx in eachindex(state_a.q)
        qa = state_a.q[idx]
        state_a.q[idx] = state_b.q[idx]
        state_b.q[idx] = qa

        pa = state_a.p[idx]
        pb = state_b.p[idx]
        state_a.p[idx] = scale_b_to_a * pb
        state_b.p[idx] = scale_a_to_b * pa
    end
    state_a.force_valid = false
    state_b.force_valid = false
    return true
end

end # module EnhancedSampling
