"""
Radix-3 27-Point Ternary Multilevel Summation Method (Ternary-MSM) Electrostatics.

Zero-FFT, O(N) periodic long-range electrostatics evaluated natively on
3^k × 3^k × 3^k balanced ternary grids using 27-point balanced ternary tryte
stencils d ∈ {-1, 0, +1}³.

Key features:
1. 27-point balanced ternary tryte indexing: bijective bijection {-1, 0, +1}³ <-> 1:27
   with exact opposite direction symmetry tryte_index(-dx, -dy, -dz) == 28 - tryte_index(dx, dy, dz).
2. C²-smooth Hardy softening kernel γ(ρ) = 15/8 - 5/4 ρ² + 3/8 ρ⁴ with strict C² continuity
   at cutoff r = r_cut and smooth non-singular evaluation at r = 0 without square roots.
3. Zero-FFT multilevel radix-3 hierarchy across levels M_l = 3^(L-l) with exact 27-point
   restriction and prolongation operators R_27, P_27 = R_27ᵀ.
4. Conservative energy U(q, H) with exact analytical Cartesian forces F = -∂U/∂q
   matching central finite differences to < 1e-9.
5. Exact analytical 9-component periodic box derivative G_αβ = ∂U/∂H_αβ|_s at fixed
   fractional coordinates matching central finite differences to < 1e-7 on both
   orthorhombic and skew triclinic cells.
6. Support for PairTopology exclusions (s = 0) and 1-4 scaling, neutral and net-charged
   systems (uniform neutralizing background).
7. Allocation-free execution on warm evaluations via preallocated `TernaryMSMPlan`.
"""
module TernaryMSM

using LinearAlgebra
using Statistics

const _PARENT = parentmodule(@__MODULE__)
const _HAS_FACTOR_PROTOCOL = isdefined(_PARENT, :Factors) && isdefined(_PARENT, :Types)
if _HAS_FACTOR_PROTOCOL
    const Factors = getfield(_PARENT, :Factors)
    const Types = getfield(_PARENT, :Types)
end
const _HAS_NEIGHBORS = isdefined(_PARENT, :Neighbors)
const Neighbors = _HAS_NEIGHBORS ? getfield(_PARENT, :Neighbors) : nothing
const _HAS_ELECTROSTATICS = isdefined(_PARENT, :Electrostatics)
const Electrostatics = _HAS_ELECTROSTATICS ? getfield(_PARENT, :Electrostatics) : nothing

export tryte_index, tryte_coords, radix3_stencil_weights,
       gamma_softening, gamma, dgamma_softening, d2gamma_softening,
       gamma_sos_softening, dgamma_sos_softening, d2gamma_sos_softening,
       sos_spectral_positivity_certificate,
       g_short, dg_short, g_long, dg_long,
       CirculantWeighingMatrix, cw_matrix, verify_cw_orthogonality,
       cw_transform!, cw_inverse_transform!, cw_convolve_1d!,
       radix3_compress_1d, radix3_compress_3d!,
       PairCorrections, TernaryMSMPlan,
       evaluate_ternary_msm, ternary_msm_energy,
       ternary_msm_energy_forces, ternary_msm_energy_forces!,
       ternary_msm_box_derivative, ternary_msm_box_derivative!,
       ternary_msm_energy_forces_box_derivative, ternary_msm_energy_forces_box_derivative!,
       TernaryMSMCoulombFactor,
       msm_factor_energy, msm_factor_box_derivative, msm_factor_box_derivative!,
       msm_factor_energy_forces_box_derivative!

# ---------------------------------------------------------------------------
# 1. 27-Point Balanced Ternary Tryte Indexing
# ---------------------------------------------------------------------------

"""
    tryte_index(dx::Int, dy::Int, dz::Int) -> Int

Map a 3-trit tryte displacement (dx, dy, dz) ∈ {-1, 0, +1}³ to a 1-based index in 1:27.
The 0-based indexing is (dx + 1) + 3*(dy + 1) + 9*(dz + 1) ∈ 0:26.
Opposite direction satisfies: `tryte_index(-dx, -dy, -dz) == 28 - tryte_index(dx, dy, dz)`.
"""
@inline function tryte_index(dx::Integer, dy::Integer, dz::Integer)
    (-1 <= dx <= 1 && -1 <= dy <= 1 && -1 <= dz <= 1) ||
        throw(ArgumentError("trit values must be in {-1, 0, 1}, got ($dx, $dy, $dz)"))
    return (Int(dx) + 1) + 3 * (Int(dy) + 1) + 9 * (Int(dz) + 1) + 1
end

"""
    tryte_coords(idx::Int) -> (Int, Int, Int)

Inverse mapping from 1-based index 1:27 to 3-trit displacement (dx, dy, dz) ∈ {-1, 0, +1}³.
"""
@inline function tryte_coords(idx::Integer)
    1 <= idx <= 27 || throw(ArgumentError("tryte index must be in 1:27, got $idx"))
    idx0 = Int(idx) - 1
    dx = mod(idx0, 3) - 1
    dy = mod(idx0 ÷ 3, 3) - 1
    dz = mod(idx0 ÷ 9, 3) - 1
    return (dx, dy, dz)
end

"""
    radix3_stencil_weights(order::Integer=4) -> (NTuple{3, Float64}, Vector{Float64})

Return 1D ternary restriction weights and full 27-point 3D tensor-product stencil weights.
For cubic B-spline (order=4): 1D weights are (1/6, 4/6, 1/6).
For linear (order=2): 1D weights are (1/4, 2/4, 1/4).
For nearest/flat (order=1): 1D weights are (1/3, 1/3, 1/3).
"""
function radix3_stencil_weights(order::Integer=4)
    w1d = order == 4 ? (1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0) :
          order == 2 ? (0.25, 0.5, 0.25) :
          order == 1 ? (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0) :
          throw(ArgumentError("unsupported radix-3 stencil order: $order (use 1, 2, or 4)"))
    w27 = zeros(Float64, 27)
    for dz in -1:1, dy in -1:1, dx in -1:1
        idx = tryte_index(dx, dy, dz)
        w27[idx] = w1d[dx + 2] * w1d[dy + 2] * w1d[dz + 2]
    end
    return w1d, w27
end

# ---------------------------------------------------------------------------
# 2. C²-Smooth Hardy Softening Functions
# ---------------------------------------------------------------------------

"""
    gamma_softening(rho::Real) -> Float64

C²-smooth softening function γ(ρ) for dimensionless distance ρ = r / r_cut:
- For ρ <= 1: γ(ρ) = 15/8 - 5/4 ρ² + 3/8 ρ⁴
- For ρ > 1:  γ(ρ) = 1 / ρ

At ρ = 1: γ(1) = 1, γ'(1) = -1, γ''(1) = 2, strictly matching 1/ρ up to 2nd derivative.
At ρ = 0: γ(0) = 15/8, γ'(0) = 0, γ''(0) = -5/2.
"""
@inline function gamma_softening(rho::Real)
    r = Float64(rho)
    if r <= 1.0
        u = r * r
        return 15.0 / 8.0 - 1.25 * u + 0.375 * u * u
    else
        return 1.0 / r
    end
end

const gamma = gamma_softening

"""
    dgamma_softening(rho::Real) -> Float64

First derivative dγ/dρ.
"""
@inline function dgamma_softening(rho::Real)
    r = Float64(rho)
    if r <= 1.0
        return -2.5 * r + 1.5 * r * r * r
    else
        return -1.0 / (r * r)
    end
end

"""
    d2gamma_softening(rho::Real) -> Float64

Second derivative d²γ/dρ².
"""
@inline function d2gamma_softening(rho::Real)
    r = Float64(rho)
    if r <= 1.0
        return -2.5 + 4.5 * r * r
    else
        return 2.0 / (r * r * r)
    end
end

@inline function _dgamma_over_rho(rho::Float64)
    if rho <= 1.0
        return -2.5 + 1.5 * rho * rho
    else
        return -1.0 / (rho * rho * rho)
    end
end

# ---------------------------------------------------------------------------
# 2b. Jellium / Cohn-Elkies SOS C⁴-Smooth Radial Smearing
# ---------------------------------------------------------------------------

"""
    gamma_sos_softening(rho::Real) -> Float64

C⁴-smooth sum-of-squares (SOS) radial smearing softening kernel γ_SOS(ρ) for ρ = r / r_cut,
induced by the nonnegative radial charge density u(ρ) = (315 / (64π)) (1 - ρ²)³ on [0, 1]:
- For ρ <= 1: γ_SOS(ρ) = 315/128 - (105/32)ρ² + (189/64)ρ⁴ - (45/32)ρ⁶ + (35/128)ρ⁸
- For ρ > 1:  γ_SOS(ρ) = 1 / ρ

Matches 1/ρ continuously through the 4th derivative at ρ = 1, with γ_SOS(0) = 315/128.
"""
@inline function gamma_sos_softening(rho::Real)
    r = Float64(rho)
    if r <= 1.0
        u = r * r
        return 315.0 / 128.0 + u * (-105.0 / 32.0 + u * (189.0 / 64.0 + u * (-45.0 / 32.0 + u * (35.0 / 128.0))))
    else
        return 1.0 / r
    end
end

"""
    dgamma_sos_softening(rho::Real) -> Float64

First derivative dγ_SOS/dρ.
"""
@inline function dgamma_sos_softening(rho::Real)
    r = Float64(rho)
    if r <= 1.0
        u = r * r
        return r * (-105.0 / 16.0 + u * (189.0 / 16.0 + u * (-135.0 / 16.0 + u * (35.0 / 16.0))))
    else
        return -1.0 / (r * r)
    end
end

"""
    d2gamma_sos_softening(rho::Real) -> Float64

Second derivative d²γ_SOS/dρ².
"""
@inline function d2gamma_sos_softening(rho::Real)
    r = Float64(rho)
    if r <= 1.0
        u = r * r
        return -105.0 / 16.0 + u * (567.0 / 16.0 + u * (-675.0 / 16.0 + u * (245.0 / 16.0)))
    else
        return 2.0 / (r * r * r)
    end
end

@inline function _d3gamma_sos_softening(rho::Float64)
    if rho <= 1.0
        u = rho * rho
        return rho * (1134.0 / 16.0 + u * (-2700.0 / 16.0 + u * (1470.0 / 16.0)))
    else
        return -6.0 / (rho^4)
    end
end

@inline function _d4gamma_sos_softening(rho::Float64)
    if rho <= 1.0
        u = rho * rho
        return 1134.0 / 16.0 + u * (-8100.0 / 16.0 + u * (7350.0 / 16.0))
    else
        return 24.0 / (rho^5)
    end
end

@inline function _dgamma_sos_over_rho(rho::Float64)
    if rho <= 1.0
        u = rho * rho
        return -105.0 / 16.0 + u * (189.0 / 16.0 + u * (-135.0 / 16.0 + u * (35.0 / 16.0)))
    else
        return -1.0 / (rho * rho * rho)
    end
end

"""
    sos_spectral_positivity_certificate(n_samples::Int=64)

Verify the Jellium / Cohn-Elkies SOS radial smearing certificate for
u(ρ) = (315 / (64π)) (1 - ρ²)³ on [0, 1] and C⁴ continuity of γ_SOS at ρ = 1.
Returns `(density_min, integral_4pi, continuity_c4_err, self_coeff, certified)`.
"""
function sos_spectral_positivity_certificate(n_samples::Int=64)
    n_samples >= 4 || throw(ArgumentError("n_samples must be >= 4, got $n_samples"))
    # Golub-Welsch Gauss-Legendre quadrature mapped to [0, 1]
    subdiag = [k / sqrt(4.0 * k * k - 1.0) for k in 1:(n_samples - 1)]
    eig = eigen(SymTridiagonal(zeros(n_samples), subdiag))
    nodes = 0.5 .* (eig.values .+ 1.0)
    weights = vec(eig.vectors[1, :] .^ 2)

    u_rad(rho) = (315.0 / (64.0 * pi)) * (1.0 - rho * rho)^3

    density_min = min(u_rad(0.0), u_rad(1.0))
    integral_4pi = 0.0
    poisson_err = 0.0
    for i in 1:n_samples
        rho = nodes[i]
        ui = u_rad(rho)
        density_min = min(density_min, ui)
        integral_4pi += 4.0 * pi * weights[i] * (rho * rho) * ui
        lhs = -(d2gamma_sos_softening(rho) + 2.0 * _dgamma_sos_over_rho(rho))
        rhs = 4.0 * pi * ui
        poisson_err = max(poisson_err, abs(lhs - rhs))
    end

    c0_err = abs(gamma_sos_softening(1.0) - 1.0)
    c1_err = abs(dgamma_sos_softening(1.0) - (-1.0))
    c2_err = abs(d2gamma_sos_softening(1.0) - 2.0)
    c3_err = abs(_d3gamma_sos_softening(1.0) - (-6.0))
    c4_err = abs(_d4gamma_sos_softening(1.0) - 24.0)
    continuity_c4_err = max(c0_err, c1_err, c2_err, c3_err, c4_err)

    self_coeff = 0.5 * gamma_sos_softening(0.0)
    certified = (density_min >= 0.0) &&
                (abs(integral_4pi - 1.0) < 1e-10) &&
                (continuity_c4_err < 1e-12) &&
                (poisson_err < 1e-12) &&
                (abs(self_coeff - 315.0 / 256.0) < 1e-14)
    return (density_min=density_min,
            integral_4pi=integral_4pi,
            continuity_c4_err=continuity_c4_err,
            self_coeff=self_coeff,
            certified=certified)
end

@inline function _validate_softening(softening::Symbol)
    softening in (:hardy_c2, :sos_c4) ||
        throw(ArgumentError("softening must be :hardy_c2 or :sos_c4, got $softening"))
    return softening === :sos_c4
end

@inline function _gamma_kernel(rho::Float64, use_sos::Bool)
    return use_sos ? gamma_sos_softening(rho) : gamma_softening(rho)
end

@inline function _dgamma_kernel(rho::Float64, use_sos::Bool)
    return use_sos ? dgamma_sos_softening(rho) : dgamma_softening(rho)
end

@inline function _dgamma_kernel_over_rho(rho::Float64, use_sos::Bool)
    return use_sos ? _dgamma_sos_over_rho(rho) : _dgamma_over_rho(rho)
end

"""
    g_short(r::Real, rcut::Real, softening::Symbol=:hardy_c2) -> Float64

Short-range real-space potential kernel:
- For r < rcut: 1/r - (1/rcut) * γ(r/rcut)
- For r >= rcut: 0.0

Supports `softening = :hardy_c2` (C² Hardy) or `:sos_c4` (C⁴ Jellium SOS).
"""
@inline function g_short(r::Real, rcut::Real, softening::Symbol=:hardy_c2)
    use_sos = _validate_softening(softening)
    return _g_short(Float64(r), Float64(rcut), use_sos)
end

@inline function _g_short(rf::Float64, rc::Float64, use_sos::Bool)
    rf >= rc && return 0.0
    rf <= 0.0 && throw(ArgumentError("distance must be positive"))
    return 1.0 / rf - (1.0 / rc) * _gamma_kernel(rf / rc, use_sos)
end

"""
    dg_short(r::Real, rcut::Real, softening::Symbol=:hardy_c2) -> Float64

Derivative of short-range real-space kernel dg_short/dr. Vanishes at r = rcut.
"""
@inline function dg_short(r::Real, rcut::Real, softening::Symbol=:hardy_c2)
    use_sos = _validate_softening(softening)
    rf = Float64(r)
    rc = Float64(rcut)
    rf >= rc && return 0.0
    rf <= 0.0 && throw(ArgumentError("distance must be positive"))
    return -1.0 / (rf * rf) - (1.0 / (rc * rc)) * _dgamma_kernel(rf / rc, use_sos)
end

@inline function _dg_short_over_r(r::Float64, rcut::Float64, use_sos::Bool=false)
    r >= rcut && return 0.0
    r <= 0.0 && throw(ArgumentError("distance must be positive"))
    inv_r = 1.0 / r
    inv_r3 = inv_r * inv_r * inv_r
    rho = r / rcut
    dgam_div_r = (1.0 / (rcut * rcut * rcut)) * _dgamma_kernel_over_rho(rho, use_sos)
    return -inv_r3 - dgam_div_r
end

@inline function _short_kernel_fast(r2::Float64, inv_rc::Float64, inv_rc2::Float64, inv_rc3::Float64, use_sos::Bool)
    r = sqrt(r2)
    inv_r2 = 1.0 / r2
    inv_r = inv_r2 * r
    inv_r3 = inv_r * inv_r2
    u = r2 * inv_rc2
    if !use_sos
        gam = 1.875 + u * (-1.25 + 0.375 * u)
        dgam_over_rho = -2.5 + 1.5 * u
    else
        gam = (315.0 / 128.0) + u * ((-105.0 / 32.0) + u * ((189.0 / 64.0) + u * ((-45.0 / 32.0) + u * (35.0 / 128.0))))
        dgam_over_rho = (-105.0 / 16.0) + u * ((189.0 / 16.0) + u * ((-135.0 / 16.0) + u * (35.0 / 16.0)))
    end
    gs = inv_r - inv_rc * gam
    dgs_over_r = -inv_r3 - inv_rc3 * dgam_over_rho
    return gs, dgs_over_r
end

"""
    g_long(r::Real, rcut::Real, softening::Symbol=:hardy_c2) -> Float64

Long-range grid potential kernel:
- For r < rcut: (1/rcut) * γ(r/rcut)
- For r >= rcut: 1/r
"""
@inline function g_long(r::Real, rcut::Real, softening::Symbol=:hardy_c2)
    use_sos = _validate_softening(softening)
    return _g_long(Float64(r), Float64(rcut), use_sos)
end

@inline function _g_long(rf::Float64, rc::Float64, use_sos::Bool)
    if rf < rc
        return (1.0 / rc) * _gamma_kernel(rf / rc, use_sos)
    else
        return 1.0 / rf
    end
end

"""
    dg_long(r::Real, rcut::Real, softening::Symbol=:hardy_c2) -> Float64

Derivative of long-range grid kernel dg_long/dr.
"""
@inline function dg_long(r::Real, rcut::Real, softening::Symbol=:hardy_c2)
    use_sos = _validate_softening(softening)
    rf = Float64(r)
    rc = Float64(rcut)
    if rf < rc
        return (1.0 / (rc * rc)) * _dgamma_kernel(rf / rc, use_sos)
    else
        return -1.0 / (rf * rf)
    end
end

@inline function _dg_long_over_r(r::Float64, rcut::Float64, use_sos::Bool=false)
    if r < rcut
        rho = r / rcut
        return (1.0 / (rcut * rcut * rcut)) * _dgamma_kernel_over_rho(rho, use_sos)
    else
        return -1.0 / (r * r * r)
    end
end

# ---------------------------------------------------------------------------
# 2c. Circulant Weighing Matrices CW(n, k) & Radix-3 3-Compression
# ---------------------------------------------------------------------------

"""
    CirculantWeighingMatrix

Circulant weighing matrix CW(n, k) with entries in {-1, 0, +1} satisfying W * Wᵀ = k * Iₙ.
Allows multiplication-free forward transforms via pure additions/subtractions.
"""
struct CirculantWeighingMatrix
    n::Int
    k::Int
    first_row::Vector{Int8}
    W::Matrix{Int8}
end

const _CW_FIRST_ROWS = Dict{Tuple{Int,Int},Vector{Int8}}(
    (3, 1)   => Int8[1, 0, 0],
    (4, 4)   => Int8[-1, 1, 1, 1],
    (6, 4)   => Int8[-1, 1, 0, 1, 1, 0],
    (7, 4)   => Int8[-1, 1, 1, 0, 1, 0, 0],
    (13, 9)  => Int8[-1, -1, 1, 1, -1, 1, 0, 1, 0, 1, 1, 0, 0],
    (14, 4)  => Int8[1, 0, 0, -1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    (21, 16) => Int8[-1, -1, -1, -1, -1, 1, -1, 0, -1, 0, 1, -1, -1, 1, 0, 0, -1, 1, 0, 1, 1],
)

"""
    cw_matrix(n::Integer, k::Integer) -> CirculantWeighingMatrix

Construct a verified canonical circulant weighing matrix CW(n, k) with entries in {-1, 0, +1}.
Supported `(n, k)` pairs include `(3, 1)`, `(4, 4)`, `(6, 4)`, `(7, 4)`, `(13, 9)`, `(14, 4)`, and `(21, 16)`.
"""
function cw_matrix(n::Integer, k::Integer)::CirculantWeighingMatrix
    ni = Int(n)
    ki = Int(k)
    row = get(_CW_FIRST_ROWS, (ni, ki), nothing)
    row === nothing && throw(ArgumentError("unsupported circulant weighing matrix CW($ni, $ki)"))
    W = Matrix{Int8}(undef, ni, ni)
    @inbounds for j in 1:ni, i in 1:ni
        W[i, j] = row[mod(j - i, ni) + 1]
    end
    return CirculantWeighingMatrix(ni, ki, copy(row), W)
end

"""
    verify_cw_orthogonality(cw::CirculantWeighingMatrix) -> Bool

Verify that `cw` has entries in {-1, 0, +1}, circulant row structure, and `W * Wᵀ == k * I`.
"""
function verify_cw_orthogonality(cw::CirculantWeighingMatrix)::Bool
    n = cw.n
    k = cw.k
    (n > 0 && k > 0) || return false
    length(cw.first_row) == n || return false
    size(cw.W) == (n, n) || return false
    for v in cw.first_row
        (v == Int8(-1) || v == Int8(0) || v == Int8(1)) || return false
    end
    for j in 1:n, i in 1:n
        v = cw.W[i, j]
        (v == Int8(-1) || v == Int8(0) || v == Int8(1)) || return false
        v == cw.first_row[mod(j - i, n) + 1] || return false
    end
    Wi = Int.(cw.W)
    return Wi * transpose(Wi) == k * Matrix{Int}(I, n, n)
end

"""
    cw_transform!(y::AbstractVector{Float64}, cw::CirculantWeighingMatrix, x::AbstractVector{Float64})

Compute `y = W * x` using pure additions and subtractions (zero multiplications by weights).
"""
function cw_transform!(y::AbstractVector{Float64}, cw::CirculantWeighingMatrix, x::AbstractVector{Float64})
    n = cw.n
    (length(y) == n && length(x) == n) ||
        throw(ArgumentError("vector lengths ($(length(y)), $(length(x))) must equal n=$n"))
    Base.mightalias(y, x) && throw(ArgumentError("output y must not alias input x"))
    W = cw.W
    @inbounds for i in 1:n
        acc = 0.0
        for j in 1:n
            w = W[i, j]
            if w == Int8(1)
                acc += x[j]
            elseif w == Int8(-1)
                acc -= x[j]
            end
        end
        y[i] = acc
    end
    return y
end

"""
    cw_inverse_transform!(x::AbstractVector{Float64}, cw::CirculantWeighingMatrix, y::AbstractVector{Float64})

Compute `x = (1 / k) * Wᵀ * y` using pure additions/subtractions for `Wᵀ * y` followed by `1/k` scaling.
"""
function cw_inverse_transform!(x::AbstractVector{Float64}, cw::CirculantWeighingMatrix, y::AbstractVector{Float64})
    n = cw.n
    (length(x) == n && length(y) == n) ||
        throw(ArgumentError("vector lengths ($(length(x)), $(length(y))) must equal n=$n"))
    Base.mightalias(x, y) && throw(ArgumentError("output x must not alias input y"))
    W = cw.W
    inv_k = 1.0 / Float64(cw.k)
    @inbounds for j in 1:n
        acc = 0.0
        for i in 1:n
            w = W[i, j]
            if w == Int8(1)
                acc += y[i]
            elseif w == Int8(-1)
                acc -= y[i]
            end
        end
        x[j] = acc * inv_k
    end
    return x
end

"""
    _build_hartley_1d(M::Int) -> Matrix{Float64}

Construct the real, symmetric, involutory orthonormal Discrete Hartley matrix
`H[a+1, b+1] = (cos(2πab/M) + sin(2πab/M)) / sqrt(M)` satisfying `H == H'` and `H^2 == I`.
"""
function _build_hartley_1d(M::Int)::Matrix{Float64}
    H = Matrix{Float64}(undef, M, M)
    inv_sqrt_M = 1.0 / sqrt(Float64(M))
    two_pi_over_M = 2.0 * Float64(pi) / Float64(M)
    @inbounds for b in 0:(M - 1)
        for a in 0:(M - 1)
            s, c = sincos(two_pi_over_M * Float64(a * b))
            H[a + 1, b + 1] = inv_sqrt_M * (c + s)
        end
    end
    return H
end

"""
    _dht3!(out::Array{Float64,3}, src::Array{Float64,3}, H1d::Matrix{Float64}, tmp::Array{Float64,3})

Apply the exact 3D real orthonormal Discrete Hartley Transform `H_3` in O(M^4) operations
with zero complex arithmetic and zero heap allocations (`out !== tmp`, `src !== tmp`; `out === src` allowed).
"""
function _dht3!(out::Array{Float64,3}, src::Array{Float64,3}, H1d::Matrix{Float64}, tmp::Array{Float64,3})
    M = size(H1d, 1)
    # Mode 1: src -> tmp (stride-1 along axis 1)
    @inbounds for k in 1:M, j in 1:M
        for i in 1:M
            acc = 0.0
            @simd for p in 1:M
                acc += H1d[p, i] * src[p, j, k]
            end
            tmp[i, j, k] = acc
        end
    end
    # Mode 2: tmp -> out (stride-1 along axis 1)
    @inbounds for k in 1:M
        for j in 1:M
            h1 = H1d[1, j]
            @simd for i in 1:M
                out[i, j, k] = h1 * tmp[i, 1, k]
            end
            for p in 2:M
                hp = H1d[p, j]
                @simd for i in 1:M
                    out[i, j, k] += hp * tmp[i, p, k]
                end
            end
        end
    end
    # Mode 3: out -> tmp (stride-1 along axis 1)
    @inbounds for k in 1:M
        h1 = H1d[1, k]
        for j in 1:M
            @simd for i in 1:M
                tmp[i, j, k] = h1 * out[i, j, 1]
            end
        end
        for p in 2:M
            hp = H1d[p, k]
            for j in 1:M
                @simd for i in 1:M
                    tmp[i, j, k] += hp * out[i, j, p]
                end
            end
        end
    end
    # 3D cas(a+b+c) butterfly: tmp -> out
    @inbounds for k3 in 1:M
        nk3 = k3 == 1 ? 1 : (M + 2 - k3)
        for k2 in 1:M
            nk2 = k2 == 1 ? 1 : (M + 2 - k2)
            for k1 in 1:M
                nk1 = k1 == 1 ? 1 : (M + 2 - k1)
                out[k1, k2, k3] = 0.5 * (tmp[k1, k2, nk3] + tmp[k1, nk2, k3] +
                                         tmp[nk1, k2, k3] - tmp[nk1, nk2, nk3])
            end
        end
    end
    return out
end

"""
    cw_convolve_1d!(out::AbstractVector{Float64}, kernel::AbstractVector{<:Real},
                    x::AbstractVector{<:Real}, cw::CirculantWeighingMatrix) -> AbstractVector{Float64}

Compute exact 1D circular / zero-padded convolution `(kernel * x)[m] = ∑_j kernel[(m - j) mod n] * x[j]`
using multiplication-free `cw_transform!` and `cw_inverse_transform!` combined with real-only
orthonormal Hartley spectral convolution (zero complex twiddles).
Inputs shorter than `cw.n` are zero-padded to length `cw.n`.
"""
function cw_convolve_1d!(out::AbstractVector{Float64},
                         kernel::AbstractVector{<:Real},
                         x::AbstractVector{<:Real},
                         cw::CirculantWeighingMatrix)
    n = cw.n
    length(out) == n ||
        throw(ArgumentError("output length $(length(out)) must equal cw.n=$n"))
    (1 <= length(kernel) <= n && 1 <= length(x) <= n) ||
        throw(ArgumentError("kernel length $(length(kernel)) and x length $(length(x)) must be in 1:$n"))

    k_pad = zeros(Float64, n)
    x_pad = zeros(Float64, n)
    @inbounds for i in eachindex(kernel)
        k_pad[i] = Float64(kernel[i])
    end
    @inbounds for i in eachindex(x)
        x_pad[i] = Float64(x[i])
    end

    x_cw = Vector{Float64}(undef, n)
    cw_transform!(x_cw, cw, x_pad)

    H1 = _build_hartley_1d(n)
    k_hat = H1 * k_pad
    y_hat = H1 * x_cw

    sqrt_n = sqrt(Float64(n))
    z_hat = Vector{Float64}(undef, n)
    @inbounds for a in 1:n
        na = a == 1 ? 1 : (n + 2 - a)
        ke = 0.5 * (k_hat[a] + k_hat[na])
        ko = 0.5 * (k_hat[a] - k_hat[na])
        z_hat[a] = sqrt_n * (ke * y_hat[a] + ko * y_hat[na])
    end

    z_cw = H1 * z_hat
    cw_inverse_transform!(out, cw, z_cw)
    return out
end

"""
    radix3_compress_1d(x::AbstractVector{Float64}) -> Vector{Float64}

Radix-3 3-compression of a 1D vector of length `3M` into length `M`:
`c[i] = x[i] + x[i + M] + x[i + 2M]`, preserving exact net sum.
"""
function radix3_compress_1d(x::AbstractVector{Float64})::Vector{Float64}
    N = length(x)
    (N > 0 && mod(N, 3) == 0) ||
        throw(ArgumentError("length(x) must be a positive multiple of 3, got $N"))
    M = N ÷ 3
    c = Vector{Float64}(undef, M)
    @inbounds for i in 1:M
        c[i] = x[i] + x[i + M] + x[i + 2 * M]
    end
    return c
end

"""
    radix3_compress_3d!(c_out::Array{Float64,3}, grid_in::Array{Float64,3}) -> Array{Float64,3}

Radix-3 3-compression on a 3D grid of size `(3M₁, 3M₂, 3M₃)` into `c_out` of size `(M₁, M₂, M₃)`:
`c_out[i, j, k] = ∑_{a,b,c ∈ {0,1,2}} grid_in[i + a*M₁, j + b*M₂, k + c*M₃]`,
preserving exact net grid sum.
"""
function radix3_compress_3d!(c_out::Array{Float64,3}, grid_in::Array{Float64,3})::Array{Float64,3}
    M1, M2, M3 = size(c_out)
    (M1 > 0 && M2 > 0 && M3 > 0) || throw(ArgumentError("c_out dimensions must be positive"))
    size(grid_in) == (3 * M1, 3 * M2, 3 * M3) ||
        throw(ArgumentError("grid_in size $(size(grid_in)) must equal (3*$M1, 3*$M2, 3*$M3)"))
    Base.mightalias(c_out, grid_in) && throw(ArgumentError("c_out must not alias grid_in"))
    @inbounds for k in 1:M3, j in 1:M2, i in 1:M1
        acc = 0.0
        for c in 0:2
            kc = k + c * M3
            for b in 0:2
                jb = j + b * M2
                for a in 0:2
                    acc += grid_in[i + a * M1, jb, kc]
                end
            end
        end
        c_out[i, j, k] = acc
    end
    return c_out
end

# ---------------------------------------------------------------------------
# 3. Pair Corrections
# ---------------------------------------------------------------------------

"""
    PairCorrections(; excluded=[(i,j), ...], scaled=[(i,j,s), ...])

Point-Coulomb pair corrections for excluded and 1-4 scaled pairs.
Correction energy is: `(s - 1) * prefactor * q_i * q_j / r`.
`excluded` corresponds to scale `s = 0.0`.
"""
struct PairCorrections
    weights::Dict{Tuple{Int,Int},Float64}
    function PairCorrections(; excluded=Tuple{Int,Int}[], scaled=Tuple{Int,Int,Float64}[])
        w = Dict{Tuple{Int,Int},Float64}()
        for p in excluded
            length(p) == 2 || throw(ArgumentError("excluded entries must be (i,j)"))
            i, j = _canonical_pair(p[1], p[2])
            haskey(w, (i, j)) && throw(ArgumentError("duplicate pair correction"))
            w[(i, j)] = 0.0
        end
        for p in scaled
            length(p) == 3 || throw(ArgumentError("scaled entries must be (i,j,scale)"))
            i, j = _canonical_pair(p[1], p[2])
            s = Float64(p[3])
            (isfinite(s) && s >= 0.0) || throw(ArgumentError("scale must be finite and nonnegative"))
            haskey(w, (i, j)) && throw(ArgumentError("duplicate pair correction"))
            w[(i, j)] = s
        end
        new(w)
    end
end

const _EMPTY_CORRECTIONS = PairCorrections()

@inline function _canonical_pair(i, j)
    (i isa Integer && j isa Integer && i > 0 && j > 0 && i != j) ||
        throw(ArgumentError("pair indices must be distinct positive integers"))
    return min(Int(i), Int(j)), max(Int(i), Int(j))
end

function _parse_corrections(corr)
    corr === nothing && return _EMPTY_CORRECTIONS
    corr isa PairCorrections && return corr
    if hasproperty(corr, :weights) && corr.weights isa Dict{Tuple{Int,Int},Float64}
        return PairCorrections(; scaled=[(i, j, s) for ((i, j), s) in corr.weights])
    end
    if _HAS_FACTOR_PROTOCOL && corr isa Types.ChemicalTopology
        w = Tuple{Int,Int,Float64}[]
        for p in corr.exclusions
            push!(w, (p[1], p[2], 0.0))
        end
        for (p, s) in corr.scales
            push!(w, (p[1], p[2], s[2]))
        end
        return PairCorrections(; scaled=w)
    end
    throw(ArgumentError("unsupported pair corrections type: $(typeof(corr))"))
end

# ---------------------------------------------------------------------------
# 4. Box Representation & Stencils
# ---------------------------------------------------------------------------

function _parse_box(box)
    if _HAS_FACTOR_PROTOCOL && box isa Types.OrthorhombicBox
        L = box.L
        H = [L[1] 0.0 0.0; 0.0 L[2] 0.0; 0.0 0.0 L[3]]
        Hinv = [1.0/L[1] 0.0 0.0; 0.0 1.0/L[2] 0.0; 0.0 0.0 1.0/L[3]]
        return (is_triclinic=false, H=H, Hinv=Hinv, L=L, widths=L, volume=L[1]*L[2]*L[3], box_obj=box)
    elseif _HAS_FACTOR_PROTOCOL && box isa Types.TriclinicBox
        H = copy(box.H)
        Hinv = copy(box.Hinv)
        widths = box.widths
        vol = det(H)
        return (is_triclinic=true, H=H, Hinv=Hinv, L=ntuple(i -> norm(view(H, :, i)), 3),
                widths=widths, volume=vol, box_obj=box)
    elseif box isa AbstractVector && length(box) == 3
        all(x -> isfinite(x) && x > 0.0, box) || throw(ArgumentError("box lengths must be positive and finite"))
        L = (Float64(box[1]), Float64(box[2]), Float64(box[3]))
        H = [L[1] 0.0 0.0; 0.0 L[2] 0.0; 0.0 0.0 L[3]]
        Hinv = [1.0/L[1] 0.0 0.0; 0.0 1.0/L[2] 0.0; 0.0 0.0 1.0/L[3]]
        box_obj = _HAS_FACTOR_PROTOCOL ? Types.OrthorhombicBox(L) : nothing
        return (is_triclinic=false, H=H, Hinv=Hinv, L=L, widths=L, volume=L[1]*L[2]*L[3], box_obj=box_obj)
    elseif box isa AbstractMatrix && size(box) == (3, 3)
        H = Matrix{Float64}(box)
        all(isfinite, H) || throw(ArgumentError("cell matrix must be finite"))
        vol = det(H)
        (isfinite(vol) && vol > 0.0) || throw(ArgumentError("cell volume must be positive"))
        Hinv = inv(H)
        widths = ntuple(i -> inv(norm(view(Hinv, i, :))), 3)
        is_ortho = maximum(abs, H .- Diagonal(diag(H))) <= 1e-12
        box_obj = _HAS_FACTOR_PROTOCOL ? (is_ortho ? Types.OrthorhombicBox((H[1,1], H[2,2], H[3,3])) : Types.TriclinicBox(H)) : nothing
        return (is_triclinic=!is_ortho, H=H, Hinv=Hinv, L=ntuple(i -> norm(view(H, :, i)), 3),
                widths=widths, volume=vol, box_obj=box_obj)
    elseif hasproperty(box, :H) && hasproperty(box, :Hinv)
        H = Matrix{Float64}(box.H)
        Hinv = Matrix{Float64}(box.Hinv)
        vol = det(H)
        widths = ntuple(i -> inv(norm(view(Hinv, i, :))), 3)
        return (is_triclinic=true, H=H, Hinv=Hinv, L=ntuple(i -> norm(view(H, :, i)), 3),
                widths=widths, volume=vol, box_obj=box)
    elseif hasproperty(box, :L) && length(box.L) == 3
        L = (Float64(box.L[1]), Float64(box.L[2]), Float64(box.L[3]))
        H = [L[1] 0.0 0.0; 0.0 L[2] 0.0; 0.0 0.0 L[3]]
        Hinv = [1.0/L[1] 0.0 0.0; 0.0 1.0/L[2] 0.0; 0.0 0.0 1.0/L[3]]
        return (is_triclinic=false, H=H, Hinv=Hinv, L=L, widths=L, volume=L[1]*L[2]*L[3], box_obj=box)
    else
        throw(ArgumentError("unsupported periodic box type: $(typeof(box))"))
    end
end

@inline function _bspline_p4(t::Float64)
    om = 1.0 - t
    t2 = t * t
    om2 = om * om
    w1 = (1.0 / 6.0) * om2 * om
    w4 = (1.0 / 6.0) * t2 * t
    w2 = (1.0 / 6.0) * (4.0 - 6.0 * t2 + 3.0 * t2 * t)
    w3 = 1.0 - w1 - w2 - w4
    dw1 = -0.5 * om2
    dw4 = 0.5 * t2
    dw2 = -2.0 * t + 1.5 * t2
    dw3 = -(dw1 + dw2 + dw4)
    return (w1, w2, w3, w4), (dw1, dw2, dw3, dw4)
end

@inline function _bspline_nodes_weights(u::Float64, M0::Int, M0_f::Float64)
    uf = floor(u)
    t = u - uf
    ws, dws = _bspline_p4(t)
    n1 = mod(Int(uf) - 1, M0) + 1
    n2 = n1 == M0 ? 1 : n1 + 1
    n3 = n2 == M0 ? 1 : n2 + 1
    n4 = n3 == M0 ? 1 : n3 + 1
    dws_scaled = (dws[1] * M0_f, dws[2] * M0_f, dws[3] * M0_f, dws[4] * M0_f)
    return (n1, n2, n3, n4), ws, dws_scaled
end

@inline function _min_image_ortho(dx::Float64, dy::Float64, dz::Float64,
                                  L1::Float64, L2::Float64, L3::Float64,
                                  invL1::Float64, invL2::Float64, invL3::Float64)
    rx = dx - L1 * round(dx * invL1)
    ry = dy - L2 * round(dy * invL2)
    rz = dz - L3 * round(dz * invL3)
    r2 = rx * rx + ry * ry + rz * rz
    return rx, ry, rz, r2
end

@inline function _min_image(dx::Float64, dy::Float64, dz::Float64,
                            H::Matrix{Float64}, Hinv::Matrix{Float64}, is_tri::Bool)
    if !is_tri
        L1 = H[1, 1]; L2 = H[2, 2]; L3 = H[3, 3]
        invL1 = Hinv[1, 1]; invL2 = Hinv[2, 2]; invL3 = Hinv[3, 3]
        rx = dx - L1 * round(dx * invL1)
        ry = dy - L2 * round(dy * invL2)
        rz = dz - L3 * round(dz * invL3)
        w1 = rx * invL1; w2 = ry * invL2; w3 = rz * invL3
        r2 = rx * rx + ry * ry + rz * rz
        return (rx, ry, rz), (w1, w2, w3), r2
    else
        f1 = Hinv[1, 1] * dx + Hinv[1, 2] * dy + Hinv[1, 3] * dz
        f2 = Hinv[2, 1] * dx + Hinv[2, 2] * dy + Hinv[2, 3] * dz
        f3 = Hinv[3, 1] * dx + Hinv[3, 2] * dy + Hinv[3, 3] * dz
        n1 = round(f1); n2 = round(f2); n3 = round(f3)
        best_rx = dx - (H[1, 1] * n1 + H[1, 2] * n2 + H[1, 3] * n3)
        best_ry = dy - (H[2, 1] * n1 + H[2, 2] * n2 + H[2, 3] * n3)
        best_rz = dz - (H[3, 1] * n1 + H[3, 2] * n2 + H[3, 3] * n3)
        best_w1 = f1 - n1; best_w2 = f2 - n2; best_w3 = f3 - n3
        best_r2 = best_rx * best_rx + best_ry * best_ry + best_rz * best_rz
        for s3 in -1:1, s2 in -1:1, s1 in -1:1
            (s1 == 0 && s2 == 0 && s3 == 0) && continue
            m1 = n1 + s1; m2 = n2 + s2; m3 = n3 + s3
            rx = dx - (H[1, 1] * m1 + H[1, 2] * m2 + H[1, 3] * m3)
            ry = dy - (H[2, 1] * m1 + H[2, 2] * m2 + H[2, 3] * m3)
            rz = dz - (H[3, 1] * m1 + H[3, 2] * m2 + H[3, 3] * m3)
            r2 = rx * rx + ry * ry + rz * rz
            if r2 < best_r2
                best_rx, best_ry, best_rz = rx, ry, rz
                best_w1, best_w2, best_w3 = f1 - m1, f2 - m2, f3 - m3
                best_r2 = r2
            end
        end
        return (best_rx, best_ry, best_rz), (best_w1, best_w2, best_w3), best_r2
    end
end

@inline function _inner_grid_product(A::Array{Float64,3}, B::Array{Float64,3})
    s = 0.0
    @inbounds for i in eachindex(A)
        s += A[i] * B[i]
    end
    return s
end

@inline function _sum_abs2(v::AbstractVector{Float64})
    s = 0.0
    @inbounds for x in v
        s += x * x
    end
    return s
end

# ---------------------------------------------------------------------------
# 5. TernaryMSMPlan Architecture
# ---------------------------------------------------------------------------

"""
    TernaryMSMPlan

Preallocated reusable execution plan for Zero-FFT Radix-3 Ternary Multilevel
Summation electrostatics on periodic boxes.
"""
mutable struct TernaryMSMPlan
    N::Int
    H::Matrix{Float64}
    Hinv::Matrix{Float64}
    volume::Float64
    widths::NTuple{3,Float64}
    is_triclinic::Bool
    box_obj::Any

    rcut::Float64
    prefactor::Float64
    order::Int
    mode::Symbol
    stencil_radius::Int
    softening::Symbol

    # Hierarchical grid levels: dims[l] = M_l
    levels::Int
    dims::Vector{Int}
    w1d::NTuple{3,Float64}
    w27::Vector{Float64}

    # Atom scratch buffers
    xbuf::Matrix{Float64}
    qqbuf::Vector{Float64}
    nodes::Array{Int,3}
    weights::Array{Float64,3}
    dweights::Array{Float64,3}

    # Grid charge and potential pyramids
    Q::Vector{Array{Float64,3}}
    Phi_dir::Vector{Array{Float64,3}}
    Phi::Vector{Array{Float64,3}}

    # Coarsest level Green's function cache & O(M_L^4) real Hartley diagonalization
    coarse_K::Array{Float64,3}
    coarse_DK::Array{Float64,5}
    coarse_G::Array{Float64,3}
    coarse_DG::Array{Float64,5}
    coarse_H1d::Matrix{Float64}
    coarse_G_hat::Array{Float64,3}
    coarse_DG_hat::Array{Float64,5}
    coarse_buf1::Array{Float64,3}
    coarse_buf2::Array{Float64,3}
    last_coarse_H::Matrix{Float64}
    coarse_valid::Base.RefValue{Bool}
    coarse_dg_valid::Base.RefValue{Bool}
    last_coarse_softening::Base.RefValue{Symbol}

    # Linked cell list for real space
    cell_n::NTuple{3,Int}
    cell_counts::Vector{Int}
    cell_offsets::Vector{Int}
    cell_id::Vector{Int}
    perm::Vector{Int}
    cell_pairs::Vector{Tuple{Int,Int}}
    sxbuf::Matrix{Float64}
    sqbuf::Vector{Float64}
    sfbuf::Matrix{Float64}
end

function TernaryMSMPlan(q::AbstractMatrix{<:Real}, box;
                        rcut::Real=5.0,
                        grid_n::Union{Nothing,Integer,NTuple{3,Integer}}=nothing,
                        levels::Union{Nothing,Integer}=nothing,
                        order::Integer=4,
                        prefactor::Real=1.0,
                        mode::Symbol=:multilevel,
                        stencil_radius::Integer=1,
                        softening::Symbol=:hardy_c2)
    size(q, 1) == 3 || throw(ArgumentError("positions q must be 3×N, got $(size(q))"))
    N = size(q, 2)
    N >= 0 || throw(ArgumentError("particle count must be nonnegative"))
    rc = Float64(rcut)
    p = Float64(prefactor)
    order == 4 || throw(ArgumentError("only cubic B-spline order 4 is supported, got $order"))
    mode in (:multilevel, :direct) || throw(ArgumentError("mode must be :multilevel or :direct, got $mode"))
    stencil_radius >= 1 || throw(ArgumentError("stencil_radius must be >= 1"))
    _validate_softening(softening)

    pbox = _parse_box(box)
    min_w = minimum(pbox.widths)
    (rc > 0.0 && rc < min_w / 2.0) ||
        throw(ArgumentError("rcut must satisfy 0 < rcut < min(widths)/2 = $(min_w/2), got $rc"))

    # Determine finest radix-3 grid dimension M0
    M0 = if grid_n === nothing
        target = min_w / 1.5
        pow3 = 9
        while pow3 < target && pow3 < 81
            pow3 *= 3
        end
        max(9, pow3)
    elseif grid_n isa Integer
        Int(grid_n)
    else
        Int(grid_n[1])
    end
    M0 >= 3 || throw(ArgumentError("grid_n must be >= 3"))

    # Determine level count
    L = if levels === nothing
        levs = 1
        cur = M0
        while cur > 3 && mod(cur, 3) == 0
            cur ÷= 3
            levs += 1
        end
        levs
    else
        Int(levels)
    end
    L >= 1 || throw(ArgumentError("levels must be >= 1"))

    pow_factor = 3^(L - 1)
    mod(M0, pow_factor) == 0 ||
        throw(ArgumentError("finest grid $M0 must be divisible by 3^(levels-1) = $pow_factor"))

    dims = Int[M0 ÷ (3^l) for l in 0:(L - 1)]
    w1d, w27 = radix3_stencil_weights(order)

    # Scratch arrays
    xbuf = zeros(Float64, 3, N)
    qqbuf = zeros(Float64, N)
    nodes = Array{Int}(undef, 4, 3, N)
    weights = Array{Float64}(undef, 4, 3, N)
    dweights = Array{Float64}(undef, 4, 3, N)

    Q = [zeros(dims[l], dims[l], dims[l]) for l in 1:L]
    Phi_dir = [zeros(dims[l], dims[l], dims[l]) for l in 1:L]
    Phi = [zeros(dims[l], dims[l], dims[l]) for l in 1:L]

    coarse_M = dims[end]
    coarse_K = zeros(Float64, coarse_M, coarse_M, coarse_M)
    coarse_DK = zeros(Float64, 3, 3, coarse_M, coarse_M, coarse_M)
    coarse_G = zeros(Float64, coarse_M, coarse_M, coarse_M)
    coarse_DG = zeros(Float64, 3, 3, coarse_M, coarse_M, coarse_M)
    coarse_H1d = _build_hartley_1d(coarse_M)
    coarse_G_hat = zeros(Float64, coarse_M, coarse_M, coarse_M)
    coarse_DG_hat = zeros(Float64, 3, 3, coarse_M, coarse_M, coarse_M)
    coarse_buf1 = zeros(Float64, coarse_M, coarse_M, coarse_M)
    coarse_buf2 = zeros(Float64, coarse_M, coarse_M, coarse_M)

    # Linked cell structures
    cn = ntuple(d -> max(3, floor(Int, pbox.widths[d] / rc)), 3)
    ncells = prod(cn)
    cell_counts = zeros(Int, ncells)
    cell_offsets = zeros(Int, ncells + 1)
    cell_id = zeros(Int, N)
    perm = zeros(Int, N)
    sxbuf = zeros(Float64, 3, N)
    sqbuf = zeros(Float64, N)
    sfbuf = zeros(Float64, 3, N)

    cell_pairs = Tuple{Int,Int}[]
    nc1, nc2, nc3 = cn
    for c3 in 0:(nc3 - 1), c2 in 0:(nc2 - 1), c1 in 0:(nc1 - 1)
        c_idx = c1 + nc1 * c2 + nc1 * nc2 * c3 + 1
        for d3 in 0:1, d2 in (d3 == 0 ? 0 : -1):1, d1 in (d3 == 0 && d2 == 0 ? 0 : -1):1
            t1 = mod(c1 + d1, nc1)
            t2 = mod(c2 + d2, nc2)
            t3 = mod(c3 + d3, nc3)
            t_idx = t1 + nc1 * t2 + nc1 * nc2 * t3 + 1
            if c_idx <= t_idx
                push!(cell_pairs, (c_idx, t_idx))
            end
        end
    end

    plan = TernaryMSMPlan(N, copy(pbox.H), copy(pbox.Hinv), pbox.volume, pbox.widths,
                          pbox.is_triclinic, pbox.box_obj, rc, p, order, mode,
                          Int(stencil_radius), softening, L, dims, w1d, w27,
                          xbuf, qqbuf, nodes, weights, dweights,
                          Q, Phi_dir, Phi,
                          coarse_K, coarse_DK, coarse_G, coarse_DG,
                          coarse_H1d, coarse_G_hat, coarse_DG_hat, coarse_buf1, coarse_buf2,
                          fill(NaN, 3, 3), Ref(false), Ref(false), Ref(:none),
                          cn, cell_counts, cell_offsets, cell_id, perm, cell_pairs,
                          sxbuf, sqbuf, sfbuf)
    return plan
end

# ---------------------------------------------------------------------------
# 6. Evaluation Core
# ---------------------------------------------------------------------------

function _update_box_if_needed!(plan::TernaryMSMPlan, H::Matrix{Float64})
    if plan.H != H
        plan.H .= H
        plan.Hinv .= inv(H)
        plan.volume = det(H)
        plan.widths = ntuple(i -> inv(norm(view(plan.Hinv, i, :))), 3)
        plan.coarse_valid[] = false
        plan.coarse_dg_valid[] = false
    end
    return nothing
end

function _rebuild_cells!(plan::TernaryMSMPlan, x::AbstractMatrix{Float64}, qq::AbstractVector{Float64})
    N = plan.N
    N <= 0 && return nothing
    cn = plan.cell_n
    nc1, nc2, nc3 = cn
    nc12 = nc1 * nc2
    counts = plan.cell_counts
    offsets = plan.cell_offsets
    cell_id = plan.cell_id
    perm = plan.perm
    sx = plan.sxbuf
    sq = plan.sqbuf
    Hinv = plan.Hinv

    fill!(counts, 0)
    @inbounds for i in 1:N
        s1 = Hinv[1, 1] * x[1, i] + Hinv[1, 2] * x[2, i] + Hinv[1, 3] * x[3, i]
        s2 = Hinv[2, 1] * x[1, i] + Hinv[2, 2] * x[2, i] + Hinv[2, 3] * x[3, i]
        s3 = Hinv[3, 1] * x[1, i] + Hinv[3, 2] * x[2, i] + Hinv[3, 3] * x[3, i]
        c1 = mod(floor(Int, s1 * nc1), nc1)
        c2 = mod(floor(Int, s2 * nc2), nc2)
        c3 = mod(floor(Int, s3 * nc3), nc3)
        cid = c1 + nc1 * c2 + nc12 * c3 + 1
        cell_id[i] = cid
        counts[cid] += 1
    end

    offsets[1] = 0
    @inbounds for c in 1:length(counts)
        offsets[c + 1] = offsets[c] + counts[c]
    end

    fill!(counts, 0)
    @inbounds for i in 1:N
        cid = cell_id[i]
        pos = offsets[cid] + counts[cid] + 1
        perm[pos] = i
        sx[1, pos] = x[1, i]
        sx[2, pos] = x[2, i]
        sx[3, pos] = x[3, i]
        sq[pos] = qq[i]
        counts[cid] += 1
    end
    return nothing
end

function _anterpolate!(plan::TernaryMSMPlan, x::AbstractMatrix{Float64}, qq::AbstractVector{Float64})
    N = plan.N
    M0 = plan.dims[1]
    M0_f = Float64(M0)
    stride2 = M0
    stride3 = M0 * M0
    Hinv = plan.Hinv
    is_tri = plan.is_triclinic
    nodes = plan.nodes
    weights = plan.weights
    dweights = plan.dweights
    Q0 = plan.Q[1]
    fill!(Q0, 0.0)

    @inbounds begin
        inv_h11 = M0_f * Hinv[1, 1]; inv_h12 = M0_f * Hinv[1, 2]; inv_h13 = M0_f * Hinv[1, 3]
        inv_h21 = M0_f * Hinv[2, 1]; inv_h22 = M0_f * Hinv[2, 2]; inv_h23 = M0_f * Hinv[2, 3]
        inv_h31 = M0_f * Hinv[3, 1]; inv_h32 = M0_f * Hinv[3, 2]; inv_h33 = M0_f * Hinv[3, 3]
    end

    @inbounds for i in 1:N
        qi = qq[i]
        qi == 0.0 && continue
        x1 = x[1, i]; x2 = x[2, i]; x3 = x[3, i]
        if !is_tri
            u1 = inv_h11 * x1
            u2 = inv_h22 * x2
            u3 = inv_h33 * x3
        else
            u1 = inv_h11 * x1 + inv_h12 * x2 + inv_h13 * x3
            u2 = inv_h21 * x1 + inv_h22 * x2 + inv_h23 * x3
            u3 = inv_h31 * x1 + inv_h32 * x2 + inv_h33 * x3
        end

        n1s, w1s, dw1s = _bspline_nodes_weights(u1, M0, M0_f)
        n2s, w2s, dw2s = _bspline_nodes_weights(u2, M0, M0_f)
        n3s, w3s, dw3s = _bspline_nodes_weights(u3, M0, M0_f)

        for k in 1:4
            nodes[k, 1, i] = n1s[k]; weights[k, 1, i] = w1s[k]; dweights[k, 1, i] = dw1s[k]
            nodes[k, 2, i] = n2s[k]; weights[k, 2, i] = w2s[k]; dweights[k, 2, i] = dw2s[k]
            nodes[k, 3, i] = n3s[k]; weights[k, 3, i] = w3s[k]; dweights[k, 3, i] = dw3s[k]
        end

        n1a, n1b, n1c, n1d = n1s
        w1a, w1b, w1c, w1d = w1s
        for k3 in 1:4
            b3 = (n3s[k3] - 1) * stride3
            qw3 = qi * w3s[k3]
            for k2 in 1:4
                base = b3 + (n2s[k2] - 1) * stride2
                qw23 = qw3 * w2s[k2]
                Q0[base + n1a] += qw23 * w1a
                Q0[base + n1b] += qw23 * w1b
                Q0[base + n1c] += qw23 * w1c
                Q0[base + n1d] += qw23 * w1d
            end
        end
    end
    return nothing
end

function _restrict_hierarchy!(plan::TernaryMSMPlan)
    L = plan.levels
    L <= 1 && return nothing
    w1d = plan.w1d
    dims = plan.dims
    Q = plan.Q

    @inbounds for l in 1:(L - 1)
        Mf = dims[l]
        Mc = dims[l + 1]
        Qf = Q[l]
        Qc = Q[l + 1]
        fill!(Qc, 0.0)
        for nc in 1:Mc
            cc = 3 * (nc - 1)
            mcs = (nc == 1 ? Mf : cc, cc + 1, cc + 2)
            for nb in 1:Mc
                cb = 3 * (nb - 1)
                mbs = (nb == 1 ? Mf : cb, cb + 1, cb + 2)
                for na in 1:Mc
                    ca = 3 * (na - 1)
                    mas = (na == 1 ? Mf : ca, ca + 1, ca + 2)
                    val = 0.0
                    for kz in 1:3
                        wz = w1d[kz]
                        mc = mcs[kz]
                        for ky in 1:3
                            wyz = wz * w1d[ky]
                            mb = mbs[ky]
                            val += wyz * (w1d[1] * Qf[mas[1], mb, mc] +
                                          w1d[2] * Qf[mas[2], mb, mc] +
                                          w1d[3] * Qf[mas[3], mb, mc])
                        end
                    end
                    Qc[na, nb, nc] = val
                end
            end
        end
    end
    return nothing
end

function _prolong_hierarchy!(plan::TernaryMSMPlan)
    L = plan.levels
    L <= 1 && return nothing
    w1d = plan.w1d
    dims = plan.dims
    Phi = plan.Phi

    @inbounds for l in (L - 1):-1:1
        Mf = dims[l]
        Mc = dims[l + 1]
        Phif = Phi[l]
        Phic = Phi[l + 1]
        for nc in 1:Mc
            cc = 3 * (nc - 1)
            mcs = (nc == 1 ? Mf : cc, cc + 1, cc + 2)
            for nb in 1:Mc
                cb = 3 * (nb - 1)
                mbs = (nb == 1 ? Mf : cb, cb + 1, cb + 2)
                for na in 1:Mc
                    val = Phic[na, nb, nc]
                    val == 0.0 && continue
                    ca = 3 * (na - 1)
                    mas = (na == 1 ? Mf : ca, ca + 1, ca + 2)
                    for kz in 1:3
                        wz = w1d[kz] * val
                        mc = mcs[kz]
                        for ky in 1:3
                            wyz = wz * w1d[ky]
                            mb = mbs[ky]
                            Phif[mas[1], mb, mc] += wyz * w1d[1]
                            Phif[mas[2], mb, mc] += wyz * w1d[2]
                            Phif[mas[3], mb, mc] += wyz * w1d[3]
                        end
                    end
                end
            end
        end
    end
    return nothing
end

@inline function _eval_short_pair!(sf::Matrix{Float64},
                                   sq::Vector{Float64},
                                   sx::Matrix{Float64},
                                   H::Matrix{Float64},
                                   Hinv::Matrix{Float64},
                                   L1::Float64, L2::Float64, L3::Float64,
                                   invL1::Float64, invL2::Float64, invL3::Float64,
                                   xi::Float64, yi::Float64, zi::Float64,
                                   qi::Float64,
                                   sj::Int,
                                   rc2::Float64, inv_rc::Float64, inv_rc2::Float64, inv_rc3::Float64,
                                   use_sos::Bool,
                                   U_short::Float64,
                                   fxi::Float64, fyi::Float64, fzi::Float64,
                                   g11::Float64, g12::Float64, g13::Float64,
                                   g21::Float64, g22::Float64, g23::Float64,
                                   g31::Float64, g32::Float64, g33::Float64,
                                   ::Val{NEED_F}, ::Val{NEED_G}, ::Val{IS_TRI}) where {NEED_F, NEED_G, IS_TRI}
    @inbounds begin
        qsj = sq[sj]
        qsj == 0.0 && return U_short, fxi, fyi, fzi, g11, g12, g13, g21, g22, g23, g31, g32, g33
        dx = xi - sx[1, sj]
        dy = yi - sx[2, sj]
        dz = zi - sx[3, sj]
        if !IS_TRI
            rx, ry, rz, r2 = _min_image_ortho(dx, dy, dz, L1, L2, L3, invL1, invL2, invL3)
            if r2 < rc2
                r2 > 0.0 || throw(ArgumentError("coincident charges are singular"))
                qij = qi * qsj
                gs, dgs_over_r = _short_kernel_fast(r2, inv_rc, inv_rc2, inv_rc3, use_sos)
                U_short += qij * gs
                cff = qij * dgs_over_r
                if NEED_F
                    fx = cff * rx; fy = cff * ry; fz = cff * rz
                    fxi -= fx; fyi -= fy; fzi -= fz
                    sf[1, sj] += fx; sf[2, sj] += fy; sf[3, sj] += fz
                end
                if NEED_G
                    w1 = rx * invL1; w2 = ry * invL2; w3 = rz * invL3
                    crx = cff * rx; cry = cff * ry; crz = cff * rz
                    g11 += crx * w1; g12 += crx * w2; g13 += crx * w3
                    g21 += cry * w1; g22 += cry * w2; g23 += cry * w3
                    g31 += crz * w1; g32 += crz * w2; g33 += crz * w3
                end
            end
        else
            (rx, ry, rz), (w1, w2, w3), r2 = _min_image(dx, dy, dz, H, Hinv, true)
            if r2 < rc2
                r2 > 0.0 || throw(ArgumentError("coincident charges are singular"))
                qij = qi * qsj
                gs, dgs_over_r = _short_kernel_fast(r2, inv_rc, inv_rc2, inv_rc3, use_sos)
                U_short += qij * gs
                cff = qij * dgs_over_r
                if NEED_F
                    fx = cff * rx; fy = cff * ry; fz = cff * rz
                    fxi -= fx; fyi -= fy; fzi -= fz
                    sf[1, sj] += fx; sf[2, sj] += fy; sf[3, sj] += fz
                end
                if NEED_G
                    crx = cff * rx; cry = cff * ry; crz = cff * rz
                    g11 += crx * w1; g12 += crx * w2; g13 += crx * w3
                    g21 += cry * w1; g22 += cry * w2; g23 += cry * w3
                    g31 += crz * w1; g32 += crz * w2; g33 += crz * w3
                end
            end
        end
    end
    return U_short, fxi, fyi, fzi, g11, g12, g13, g21, g22, g23, g31, g32, g33
end

function _short_range_cells!(sf::Matrix{Float64},
                             cell_pairs::Vector{Tuple{Int,Int}},
                             offsets::Vector{Int},
                             sx::Matrix{Float64},
                             sq::Vector{Float64},
                             H::Matrix{Float64},
                             Hinv::Matrix{Float64},
                             rc2::Float64, inv_rc::Float64, inv_rc2::Float64, inv_rc3::Float64,
                             p::Float64, use_sos::Bool,
                             ::Val{NEED_F}, ::Val{NEED_G}, ::Val{IS_TRI}) where {NEED_F, NEED_G, IS_TRI}
    U_short = 0.0
    g11 = 0.0; g12 = 0.0; g13 = 0.0
    g21 = 0.0; g22 = 0.0; g23 = 0.0
    g31 = 0.0; g32 = 0.0; g33 = 0.0
    @inbounds begin
        L1 = H[1, 1]; L2 = H[2, 2]; L3 = H[3, 3]
        invL1 = Hinv[1, 1]; invL2 = Hinv[2, 2]; invL3 = Hinv[3, 3]
    end

    @inbounds for (c, t) in cell_pairs
        i0 = offsets[c] + 1; i1 = offsets[c + 1]
        i0 > i1 && continue
        if c == t
            for si in i0:(i1 - 1)
                qi = p * sq[si]
                qi == 0.0 && continue
                xi = sx[1, si]; yi = sx[2, si]; zi = sx[3, si]
                fxi = 0.0; fyi = 0.0; fzi = 0.0
                for sj in (si + 1):i1
                    U_short, fxi, fyi, fzi, g11, g12, g13, g21, g22, g23, g31, g32, g33 =
                        _eval_short_pair!(sf, sq, sx, H, Hinv, L1, L2, L3, invL1, invL2, invL3,
                                          xi, yi, zi, qi, sj,
                                          rc2, inv_rc, inv_rc2, inv_rc3, use_sos,
                                          U_short, fxi, fyi, fzi,
                                          g11, g12, g13, g21, g22, g23, g31, g32, g33,
                                          Val(NEED_F), Val(NEED_G), Val(IS_TRI))
                end
                if NEED_F
                    sf[1, si] += fxi; sf[2, si] += fyi; sf[3, si] += fzi
                end
            end
        else
            j0 = offsets[t] + 1; j1 = offsets[t + 1]
            j0 > j1 && continue
            for si in i0:i1
                qi = p * sq[si]
                qi == 0.0 && continue
                xi = sx[1, si]; yi = sx[2, si]; zi = sx[3, si]
                fxi = 0.0; fyi = 0.0; fzi = 0.0
                for sj in j0:j1
                    U_short, fxi, fyi, fzi, g11, g12, g13, g21, g22, g23, g31, g32, g33 =
                        _eval_short_pair!(sf, sq, sx, H, Hinv, L1, L2, L3, invL1, invL2, invL3,
                                          xi, yi, zi, qi, sj,
                                          rc2, inv_rc, inv_rc2, inv_rc3, use_sos,
                                          U_short, fxi, fyi, fzi,
                                          g11, g12, g13, g21, g22, g23, g31, g32, g33,
                                          Val(NEED_F), Val(NEED_G), Val(IS_TRI))
                end
                if NEED_F
                    sf[1, si] += fxi; sf[2, si] += fyi; sf[3, si] += fzi
                end
            end
        end
    end
    return U_short, g11, g12, g13, g21, g22, g23, g31, g32, g33
end

function _coarse_convolve_direct!(Phil::Array{Float64,3},
                                  Ql::Array{Float64,3},
                                  cG::Array{Float64,3},
                                  cDG::Array{Float64,5},
                                  Ml::Int, p::Float64, compute_G::Bool)
    fill!(Phil, 0.0)
    stride2 = Ml
    stride3 = Ml * Ml
    g11 = 0.0; g12 = 0.0; g13 = 0.0
    g21 = 0.0; g22 = 0.0; g23 = 0.0
    g31 = 0.0; g32 = 0.0; g33 = 0.0
    @inbounds for m3 in 1:Ml, m2 in 1:Ml, m1 in 1:Ml
        qm = Ql[m1, m2, m3]
        qm == 0.0 && continue
        if !compute_G
            for n3 in 1:Ml
                d3 = m3 >= n3 ? (m3 - n3) : (m3 - n3 + Ml)
                b3_g = d3 * stride3
                b3_n = (n3 - 1) * stride3
                for n2 in 1:Ml
                    d2 = m2 >= n2 ? (m2 - n2) : (m2 - n2 + Ml)
                    base_g = b3_g + d2 * stride2
                    base_n = b3_n + (n2 - 1) * stride2
                    for n1 in 1:m1
                        Phil[base_n + n1] += cG[base_g + (m1 - n1 + 1)] * qm
                    end
                    for n1 in (m1 + 1):Ml
                        Phil[base_n + n1] += cG[base_g + (m1 - n1 + Ml + 1)] * qm
                    end
                end
            end
        else
            half_pqm = 0.5 * p * qm
            for n3 in 1:Ml
                d3 = m3 >= n3 ? (m3 - n3) : (m3 - n3 + Ml)
                b3_g = d3 * stride3
                b3_n = (n3 - 1) * stride3
                for n2 in 1:Ml
                    d2 = m2 >= n2 ? (m2 - n2) : (m2 - n2 + Ml)
                    base_g = b3_g + d2 * stride2
                    base_n = b3_n + (n2 - 1) * stride2
                    for n1 in 1:m1
                        d1 = m1 - n1 + 1
                        idx_g = base_g + d1
                        idx_n = base_n + n1
                        Phil[idx_n] += cG[idx_g] * qm
                        qn = Ql[idx_n]
                        if qn != 0.0
                            c = half_pqm * qn
                            off = 9 * (idx_g - 1)
                            g11 += c * cDG[off + 1]; g21 += c * cDG[off + 2]; g31 += c * cDG[off + 3]
                            g12 += c * cDG[off + 4]; g22 += c * cDG[off + 5]; g32 += c * cDG[off + 6]
                            g13 += c * cDG[off + 7]; g23 += c * cDG[off + 8]; g33 += c * cDG[off + 9]
                        end
                    end
                    for n1 in (m1 + 1):Ml
                        d1 = m1 - n1 + Ml + 1
                        idx_g = base_g + d1
                        idx_n = base_n + n1
                        Phil[idx_n] += cG[idx_g] * qm
                        qn = Ql[idx_n]
                        if qn != 0.0
                            c = half_pqm * qn
                            off = 9 * (idx_g - 1)
                            g11 += c * cDG[off + 1]; g21 += c * cDG[off + 2]; g31 += c * cDG[off + 3]
                            g12 += c * cDG[off + 4]; g22 += c * cDG[off + 5]; g32 += c * cDG[off + 6]
                            g13 += c * cDG[off + 7]; g23 += c * cDG[off + 8]; g33 += c * cDG[off + 9]
                        end
                    end
                end
            end
        end
    end
    return g11, g12, g13, g21, g22, g23, g31, g32, g33
end

function _coarse_convolve_hartley!(Phil::Array{Float64,3},
                                   Ql::Array{Float64,3},
                                   cG_hat::Array{Float64,3},
                                   cDG_hat::Array{Float64,5},
                                   H1d::Matrix{Float64},
                                   buf1::Array{Float64,3},
                                   buf2::Array{Float64,3},
                                   p::Float64, compute_G::Bool)
    _dht3!(buf1, Ql, H1d, buf2)
    g11 = 0.0; g12 = 0.0; g13 = 0.0
    g21 = 0.0; g22 = 0.0; g23 = 0.0
    g31 = 0.0; g32 = 0.0; g33 = 0.0
    if !compute_G
        @inbounds @simd for idx in eachindex(buf1)
            buf1[idx] *= cG_hat[idx]
        end
    else
        half_p = 0.5 * p
        @inbounds for idx in eachindex(buf1)
            qh = buf1[idx]
            buf1[idx] = cG_hat[idx] * qh
            c = half_p * qh * qh
            off = 9 * (idx - 1)
            g11 += c * cDG_hat[off + 1]; g21 += c * cDG_hat[off + 2]; g31 += c * cDG_hat[off + 3]
            g12 += c * cDG_hat[off + 4]; g22 += c * cDG_hat[off + 5]; g32 += c * cDG_hat[off + 6]
            g13 += c * cDG_hat[off + 7]; g23 += c * cDG_hat[off + 8]; g33 += c * cDG_hat[off + 9]
        end
    end
    _dht3!(Phil, buf1, H1d, buf2)
    return g11, g12, g13, g21, g22, g23, g31, g32, g33
end

function _msm_core!(F::Union{Nothing,AbstractMatrix{Float64}},
                    G::Union{Nothing,AbstractMatrix{Float64}},
                    plan::TernaryMSMPlan,
                    q::AbstractMatrix{<:Real},
                    charges::AbstractVector{<:Real};
                    corrections::PairCorrections=_EMPTY_CORRECTIONS,
                    accumulate_F::Bool=false,
                    accumulate_G::Bool=false)
    N = plan.N
    size(q, 1) == 3 && size(q, 2) == N ||
        throw(ArgumentError("positions q must be 3×$N, got $(size(q))"))
    length(charges) == N || throw(ArgumentError("charges length must match N=$N"))
    all(isfinite, q) || throw(ArgumentError("positions must be finite"))
    all(isfinite, charges) || throw(ArgumentError("charges must be finite"))

    copyto!(plan.qqbuf, charges)
    copyto!(plan.xbuf, q)
    qq = plan.qqbuf
    x = plan.xbuf

    if F !== nothing
        size(F) == (3, N) || throw(ArgumentError("force accumulator F must be 3×$N"))
        Base.mightalias(F, q) && throw(ArgumentError("force accumulator must not alias q"))
        accumulate_F || fill!(F, 0.0)
    end
    if G !== nothing
        size(G) == (3, 3) || throw(ArgumentError("box derivative G must be 3×3"))
        Base.mightalias(G, q) && throw(ArgumentError("box derivative accumulator must not alias q"))
        accumulate_G || fill!(G, 0.0)
    end

    H = plan.H
    Hinv = plan.Hinv
    is_tri = plan.is_triclinic
    rc = plan.rcut
    rc2 = rc * rc
    inv_rc = 1.0 / rc
    inv_rc2 = inv_rc * inv_rc
    inv_rc3 = inv_rc * inv_rc2
    p = plan.prefactor
    L_levels = plan.levels
    dims = plan.dims
    stencil_radius = plan.stencil_radius
    use_sos = _validate_softening(plan.softening)

    # 1. Real-space linked cell traversal
    _rebuild_cells!(plan, x, qq)
    offsets = plan.cell_offsets
    sx = plan.sxbuf
    sq = plan.sqbuf
    sf = plan.sfbuf
    perm = plan.perm
    fill!(sf, 0.0)

    U_short, g11, g12, g13, g21, g22, g23, g31, g32, g33 = if !is_tri
        _short_range_cells!(sf, plan.cell_pairs, offsets, sx, sq, H, Hinv,
                            rc2, inv_rc, inv_rc2, inv_rc3, p, use_sos,
                            Val(F !== nothing), Val(G !== nothing), Val(false))
    else
        _short_range_cells!(sf, plan.cell_pairs, offsets, sx, sq, H, Hinv,
                            rc2, inv_rc, inv_rc2, inv_rc3, p, use_sos,
                            Val(F !== nothing), Val(G !== nothing), Val(true))
    end

    if F !== nothing
        @inbounds for s in 1:N
            idx = perm[s]
            F[1, idx] += sf[1, s]
            F[2, idx] += sf[2, s]
            F[3, idx] += sf[3, s]
        end
    end

    # 2. Multilevel Radix-3 Grid Computation
    _anterpolate!(plan, x, qq)
    _restrict_hierarchy!(plan)

    Q = plan.Q
    Phi_dir = plan.Phi_dir
    Phi = plan.Phi

    for l in 1:L_levels
        fill!(Phi_dir[l], 0.0)
    end

    for l in 0:(L_levels - 1)
        Ml = dims[l + 1]
        Ql = Q[l + 1]
        Phil = Phi_dir[l + 1]
        al = rc * (3.0^l)
        al1 = rc * (3.0^(l + 1))
        is_top = (l == L_levels - 1)
        inv_Ml = 1.0 / Ml

        if !is_top
            stride2 = Ml
            stride3 = Ml * Ml
            @inbounds for dz in -stencil_radius:stencil_radius, dy in -stencil_radius:stencil_radius, dx in -stencil_radius:stencil_radius
                s1 = dx * inv_Ml; s2 = dy * inv_Ml; s3 = dz * inv_Ml
                rx = H[1, 1] * s1 + H[1, 2] * s2 + H[1, 3] * s3
                ry = H[2, 1] * s1 + H[2, 2] * s2 + H[2, 3] * s3
                rz = H[3, 1] * s1 + H[3, 2] * s2 + H[3, 3] * s3
                r = sqrt(rx * rx + ry * ry + rz * rz)
                kval = _g_long(r, al, use_sos) - _g_long(r, al1, use_sos)
                dkl_over_r = r > 0.0 ? (_dg_long_over_r(r, al, use_sos) - _dg_long_over_r(r, al1, use_sos)) : 0.0
                shift1 = mod(dx, Ml)
                n1_split = Ml - shift1

                if G === nothing || r == 0.0
                    for n3 in 1:Ml
                        m3 = mod(n3 - 1 + dz, Ml) + 1
                        b3_m = (m3 - 1) * stride3
                        b3_n = (n3 - 1) * stride3
                        for n2 in 1:Ml
                            m2 = mod(n2 - 1 + dy, Ml) + 1
                            base_m = b3_m + (m2 - 1) * stride2
                            base_n = b3_n + (n2 - 1) * stride2
                            for n1 in 1:n1_split
                                qm = Ql[base_m + n1 + shift1]
                                qm != 0.0 && (Phil[base_n + n1] += kval * qm)
                            end
                            for n1 in (n1_split + 1):Ml
                                qm = Ql[base_m + n1 + shift1 - Ml]
                                qm != 0.0 && (Phil[base_n + n1] += kval * qm)
                            end
                        end
                    end
                else
                    g_loc = 0.0
                    for n3 in 1:Ml
                        m3 = mod(n3 - 1 + dz, Ml) + 1
                        b3_m = (m3 - 1) * stride3
                        b3_n = (n3 - 1) * stride3
                        for n2 in 1:Ml
                            m2 = mod(n2 - 1 + dy, Ml) + 1
                            base_m = b3_m + (m2 - 1) * stride2
                            base_n = b3_n + (n2 - 1) * stride2
                            for n1 in 1:n1_split
                                qm = Ql[base_m + n1 + shift1]
                                if qm != 0.0
                                    Phil[base_n + n1] += kval * qm
                                    g_loc += Ql[base_n + n1] * qm
                                end
                            end
                            for n1 in (n1_split + 1):Ml
                                qm = Ql[base_m + n1 + shift1 - Ml]
                                if qm != 0.0
                                    Phil[base_n + n1] += kval * qm
                                    g_loc += Ql[base_n + n1] * qm
                                end
                            end
                        end
                    end
                    if g_loc != 0.0
                        c = 0.5 * p * dkl_over_r * g_loc
                        g11 += c * rx * s1; g12 += c * rx * s2; g13 += c * rx * s3
                        g21 += c * ry * s1; g22 += c * ry * s2; g23 += c * ry * s3
                        g31 += c * rz * s1; g32 += c * rz * s2; g33 += c * rz * s3
                    end
                end
            end
        else
            # Coarsest level periodic lattice with neutralizing background subtraction
            # Cached by box matrix H & softening kernel for zero-recompute fixed-box MD steps,
            # diagonalized in O(M_L^4) via real orthonormal Discrete Hartley Transform.
            cK = plan.coarse_K
            cDK = plan.coarse_DK
            cG = plan.coarse_G
            cDG = plan.coarse_DG
            cH1d = plan.coarse_H1d
            cG_hat = plan.coarse_G_hat
            cDG_hat = plan.coarse_DG_hat
            cbuf1 = plan.coarse_buf1
            cbuf2 = plan.coarse_buf2
            scale_hat = Float64(Ml) * sqrt(Float64(Ml))

            box_changed = !(plan.coarse_valid[] && plan.last_coarse_H == H &&
                            plan.last_coarse_softening[] === plan.softening)
            if box_changed
                @inbounds for d3 in 0:(Ml - 1), d2 in 0:(Ml - 1), d1 in 0:(Ml - 1)
                    s1 = (d1 - Ml * round(d1 * inv_Ml)) * inv_Ml
                    s2 = (d2 - Ml * round(d2 * inv_Ml)) * inv_Ml
                    s3 = (d3 - Ml * round(d3 * inv_Ml)) * inv_Ml
                    rx = H[1, 1] * s1 + H[1, 2] * s2 + H[1, 3] * s3
                    ry = H[2, 1] * s1 + H[2, 2] * s2 + H[2, 3] * s3
                    rz = H[3, 1] * s1 + H[3, 2] * s2 + H[3, 3] * s3
                    r = sqrt(rx * rx + ry * ry + rz * rz)
                    cK[d1 + 1, d2 + 1, d3 + 1] = _g_long(r, al, use_sos)
                end

                K_sum = 0.0
                @inbounds for i in eachindex(cK)
                    K_sum += cK[i]
                end
                K_mean = K_sum / length(cK)
                @inbounds for i in eachindex(cG)
                    cG[i] = cK[i] - K_mean
                end

                _dht3!(cG_hat, cG, cH1d, cbuf2)
                @inbounds @simd for i in eachindex(cG_hat)
                    cG_hat[i] *= scale_hat
                end

                copyto!(plan.last_coarse_H, H)
                plan.last_coarse_softening[] = plan.softening
                plan.coarse_valid[] = true
                plan.coarse_dg_valid[] = false
            end

            if G !== nothing && !plan.coarse_dg_valid[]
                @inbounds for d3 in 0:(Ml - 1), d2 in 0:(Ml - 1), d1 in 0:(Ml - 1)
                    s1 = (d1 - Ml * round(d1 * inv_Ml)) * inv_Ml
                    s2 = (d2 - Ml * round(d2 * inv_Ml)) * inv_Ml
                    s3 = (d3 - Ml * round(d3 * inv_Ml)) * inv_Ml
                    rx = H[1, 1] * s1 + H[1, 2] * s2 + H[1, 3] * s3
                    ry = H[2, 1] * s1 + H[2, 2] * s2 + H[2, 3] * s3
                    rz = H[3, 1] * s1 + H[3, 2] * s2 + H[3, 3] * s3
                    r = sqrt(rx * rx + ry * ry + rz * rz)
                    if r > 0.0
                        dgl = _dg_long_over_r(r, al, use_sos)
                        for a in 1:3, b in 1:3
                            r_comp = a == 1 ? rx : (a == 2 ? ry : rz)
                            w_comp = b == 1 ? s1 : (b == 2 ? s2 : s3)
                            cDK[a, b, d1 + 1, d2 + 1, d3 + 1] = dgl * r_comp * w_comp
                        end
                    else
                        for a in 1:3, b in 1:3
                            cDK[a, b, d1 + 1, d2 + 1, d3 + 1] = 0.0
                        end
                    end
                end
                inv_vol_coarse = inv_Ml * inv_Ml * inv_Ml
                n_coarse = Ml * Ml * Ml
                @inbounds for b in 1:3, a in 1:3
                    dk_sum = 0.0
                    for d3 in 1:Ml, d2 in 1:Ml, d1 in 1:Ml
                        dk_sum += cDK[a, b, d1, d2, d3]
                    end
                    dk_mean = dk_sum * inv_vol_coarse
                    for d3 in 1:Ml, d2 in 1:Ml, d1 in 1:Ml
                        val = cDK[a, b, d1, d2, d3] - dk_mean
                        cDG[a, b, d1, d2, d3] = val
                        cbuf1[d1, d2, d3] = val
                    end
                    _dht3!(cbuf1, cbuf1, cH1d, cbuf2)
                    ab_off = a + 3 * (b - 1)
                    for idx in 1:n_coarse
                        cDG_hat[ab_off + 9 * (idx - 1)] = scale_hat * cbuf1[idx]
                    end
                end
                plan.coarse_dg_valid[] = true
            end

            dg11, dg12, dg13, dg21, dg22, dg23, dg31, dg32, dg33 =
                _coarse_convolve_hartley!(Phil, Ql, cG_hat, cDG_hat, cH1d, cbuf1, cbuf2, p, G !== nothing)
            if G !== nothing
                g11 += dg11; g12 += dg12; g13 += dg13
                g21 += dg21; g22 += dg22; g23 += dg23
                g31 += dg31; g32 += dg32; g33 += dg33
            end
        end
    end

    for l in 1:L_levels
        copyto!(Phi[l], Phi_dir[l])
    end
    _prolong_hierarchy!(plan)

    U_grid = 0.5 * p * _inner_grid_product(Q[1], Phi[1])

    # 3. Force interpolation from Level 0 potential Phi[1]
    if F !== nothing
        nodes = plan.nodes
        weights = plan.weights
        dweights = plan.dweights
        Phi0 = Phi[1]
        M0 = dims[1]
        stride2 = M0
        stride3 = M0 * M0
        @inbounds begin
            h11 = Hinv[1, 1]; h21 = Hinv[2, 1]; h31 = Hinv[3, 1]
            h12 = Hinv[1, 2]; h22 = Hinv[2, 2]; h32 = Hinv[3, 2]
            h13 = Hinv[1, 3]; h23 = Hinv[2, 3]; h33 = Hinv[3, 3]
        end
        @inbounds for i in 1:N
            qi = qq[i]
            qi == 0.0 && continue
            n1a = nodes[1, 1, i]; n1b = nodes[2, 1, i]; n1c = nodes[3, 1, i]; n1d = nodes[4, 1, i]
            w1a = weights[1, 1, i]; w1b = weights[2, 1, i]; w1c = weights[3, 1, i]; w1d = weights[4, 1, i]
            dw1a = dweights[1, 1, i]; dw1b = dweights[2, 1, i]; dw1c = dweights[3, 1, i]; dw1d = dweights[4, 1, i]
            g1 = 0.0; g2 = 0.0; g3 = 0.0
            for k3 in 1:4
                b3 = (nodes[k3, 3, i] - 1) * stride3
                w3 = weights[k3, 3, i]
                dw3 = dweights[k3, 3, i]
                for k2 in 1:4
                    base = b3 + (nodes[k2, 2, i] - 1) * stride2
                    w2 = weights[k2, 2, i]
                    dw2 = dweights[k2, 2, i]
                    v1 = Phi0[base + n1a]
                    v2 = Phi0[base + n1b]
                    v3 = Phi0[base + n1c]
                    v4 = Phi0[base + n1d]
                    acc_w1 = v1 * w1a + v2 * w1b + v3 * w1c + v4 * w1d
                    acc_dw1 = v1 * dw1a + v2 * dw1b + v3 * dw1c + v4 * dw1d
                    g1 += acc_dw1 * (w2 * w3)
                    g2 += acc_w1 * (dw2 * w3)
                    g3 += acc_w1 * (w2 * dw3)
                end
            end
            gr1 = h11 * g1 + h21 * g2 + h31 * g3
            gr2 = h12 * g1 + h22 * g2 + h32 * g3
            gr3 = h13 * g1 + h23 * g2 + h33 * g3
            pqi = p * qi
            F[1, i] -= pqi * gr1
            F[2, i] -= pqi * gr2
            F[3, i] -= pqi * gr3
        end
    end

    # 4. Self-energy correction
    self_coeff = use_sos ? (315.0 / 256.0) : (15.0 / 16.0)
    U_self = p * (self_coeff * inv_rc) * _sum_abs2(qq)
    U_total = U_short + U_grid - U_self

    # 5. Point-Coulomb Pair Corrections (exclusions & 1-4 scaling)
    if !isempty(corrections.weights)
        @inbounds for ((i, j), s) in corrections.weights
            (rx, ry, rz), (w1, w2, w3), r2 = _min_image(
                x[1, i] - x[1, j], x[2, i] - x[2, j], x[3, i] - x[3, j], H, Hinv, is_tri)
            r2 > 0.0 || throw(ArgumentError("coincident corrected charges are singular"))
            r = sqrt(r2)
            inv_r2 = 1.0 / r2
            inv_r = inv_r2 * r
            c = p * (s - 1.0) * qq[i] * qq[j]
            U_total += c * inv_r
            cff = c * inv_r * inv_r2
            if F !== nothing
                F[1, i] += cff * rx; F[2, i] += cff * ry; F[3, i] += cff * rz
                F[1, j] -= cff * rx; F[2, j] -= cff * ry; F[3, j] -= cff * rz
            end
            if G !== nothing
                crx = cff * rx; cry = cff * ry; crz = cff * rz
                g11 -= crx * w1; g12 -= crx * w2; g13 -= crx * w3
                g21 -= cry * w1; g22 -= cry * w2; g23 -= cry * w3
                g31 -= crz * w1; g32 -= crz * w2; g33 -= crz * w3
            end
        end
    end

    if G !== nothing
        @inbounds begin
            G[1, 1] += g11; G[1, 2] += g12; G[1, 3] += g13
            G[2, 1] += g21; G[2, 2] += g22; G[2, 3] += g23
            G[3, 1] += g31; G[3, 2] += g32; G[3, 3] += g33
        end
    end

    return U_total
end

# ---------------------------------------------------------------------------
# 7. Public Evaluators
# ---------------------------------------------------------------------------

"""
    ternary_msm_energy(plan, q, charges; corrections=PairCorrections()) -> Float64

Evaluate Zero-FFT Radix-3 Ternary-MSM electrostatics energy.
"""
function ternary_msm_energy(plan::TernaryMSMPlan, q::AbstractMatrix, charges::AbstractVector;
                            corrections::PairCorrections=_EMPTY_CORRECTIONS)
    return _msm_core!(nothing, nothing, plan, q, charges; corrections=corrections)
end

"""
    ternary_msm_energy_forces!(F, plan, q, charges; corrections=PairCorrections(), accumulate_F=false) -> Float64

Fill `F` (3×N) with exact conservative forces `F = -∇U` and return energy.
"""
function ternary_msm_energy_forces!(F::AbstractMatrix{Float64}, plan::TernaryMSMPlan,
                                    q::AbstractMatrix, charges::AbstractVector;
                                    corrections::PairCorrections=_EMPTY_CORRECTIONS,
                                    accumulate_F::Bool=false)
    return _msm_core!(F, nothing, plan, q, charges; corrections=corrections, accumulate_F=accumulate_F)
end

"""
    ternary_msm_energy_forces(plan, q, charges; corrections=PairCorrections()) -> (Float64, Matrix{Float64})

Convenience wrapper returning `(energy, forces)`.
"""
function ternary_msm_energy_forces(plan::TernaryMSMPlan, q::AbstractMatrix, charges::AbstractVector;
                                  corrections::PairCorrections=_EMPTY_CORRECTIONS)
    F = zeros(Float64, 3, plan.N)
    U = ternary_msm_energy_forces!(F, plan, q, charges; corrections=corrections)
    return U, F
end

"""
    ternary_msm_box_derivative!(G, plan, q, charges; corrections=PairCorrections(), accumulate=false) -> Matrix{Float64}

Write or accumulate exact 9-component periodic box derivative `G_αβ = ∂U/∂H_αβ|_s` into `G` (3×3).
"""
function ternary_msm_box_derivative!(G::AbstractMatrix{Float64}, plan::TernaryMSMPlan,
                                     q::AbstractMatrix, charges::AbstractVector;
                                     corrections::PairCorrections=_EMPTY_CORRECTIONS,
                                     accumulate::Bool=false)
    _msm_core!(nothing, G, plan, q, charges; corrections=corrections, accumulate_G=accumulate)
    return G
end

"""
    ternary_msm_box_derivative(plan, q, charges; corrections=PairCorrections()) -> Matrix{Float64}

Compute exact 9-component periodic box derivative `G_αβ = ∂U/∂H_αβ|_s` as a 3×3 matrix.
"""
function ternary_msm_box_derivative(plan::TernaryMSMPlan, q::AbstractMatrix, charges::AbstractVector;
                                    corrections::PairCorrections=_EMPTY_CORRECTIONS)
    G = zeros(Float64, 3, 3)
    ternary_msm_box_derivative!(G, plan, q, charges; corrections=corrections, accumulate=false)
    return G
end

"""
    ternary_msm_energy_forces_box_derivative!(F, G, plan, q, charges; corrections=PairCorrections(), accumulate_F=false, accumulate_G=false) -> Float64

Evaluate energy, forces, and 9-component box derivative together in a single pass.
"""
function ternary_msm_energy_forces_box_derivative!(F::AbstractMatrix{Float64},
                                                  G::AbstractMatrix{Float64},
                                                  plan::TernaryMSMPlan,
                                                  q::AbstractMatrix,
                                                  charges::AbstractVector;
                                                  corrections::PairCorrections=_EMPTY_CORRECTIONS,
                                                  accumulate_F::Bool=false,
                                                  accumulate_G::Bool=false)
    return _msm_core!(F, G, plan, q, charges;
                      corrections=corrections, accumulate_F=accumulate_F, accumulate_G=accumulate_G)
end

"""
    ternary_msm_energy_forces_box_derivative(plan, q, charges; corrections=PairCorrections()) -> (Float64, Matrix{Float64}, Matrix{Float64})

Convenience wrapper returning `(energy, forces, box_derivative)`.
"""
function ternary_msm_energy_forces_box_derivative(plan::TernaryMSMPlan, q::AbstractMatrix, charges::AbstractVector;
                                                 corrections::PairCorrections=_EMPTY_CORRECTIONS)
    F = zeros(Float64, 3, plan.N)
    G = zeros(Float64, 3, 3)
    U = ternary_msm_energy_forces_box_derivative!(F, G, plan, q, charges; corrections=corrections)
    return U, F, G
end

"""
    evaluate_ternary_msm(q, charges, box; rcut=5.0, grid_n=nothing, levels=nothing, order=4, prefactor=1.0, corrections=PairCorrections()) -> (Float64, Matrix{Float64})

One-shot evaluation of Zero-FFT Radix-3 Ternary-MSM electrostatics returning `(energy, forces)`.
"""
function evaluate_ternary_msm(q::AbstractMatrix, charges::AbstractVector, box;
                              rcut::Real=5.0,
                              grid_n=nothing,
                              levels=nothing,
                              order::Integer=4,
                              prefactor::Real=1.0,
                              corrections::PairCorrections=_EMPTY_CORRECTIONS,
                              softening::Symbol=:hardy_c2)
    plan = TernaryMSMPlan(q, box; rcut=rcut, grid_n=grid_n, levels=levels,
                          order=order, prefactor=prefactor, softening=softening)
    return ternary_msm_energy_forces(plan, q, charges; corrections=corrections)
end

# ---------------------------------------------------------------------------
# 8. Factors.AbstractFactor Adapter
# ---------------------------------------------------------------------------

if _HAS_FACTOR_PROTOCOL
    """
        TernaryMSMCoulombFactor <: Factors.AbstractFactor

    Nonlocal factor implementing Zero-FFT Radix-3 27-point Ternary Multilevel Summation
    Electrostatics for the modular `FoundationCore` compositional factor protocol.
    """
    mutable struct TernaryMSMCoulombFactor{B<:Union{Types.OrthorhombicBox, Types.TriclinicBox}, PL<:TernaryMSMPlan} <: Factors.AbstractFactor
        N::Int
        charges::Vector{Float64}
        box::B
        plan::PL
        corrections::PairCorrections
    end

    function TernaryMSMCoulombFactor(q::AbstractMatrix{<:Real},
                                    charges::AbstractVector{<:Real},
                                    box::Union{Types.OrthorhombicBox, Types.TriclinicBox};
                                    rcut::Real=5.0,
                                    grid_n=nothing,
                                    levels=nothing,
                                    order::Integer=4,
                                    prefactor::Real=1.0,
                                    corrections::PairCorrections=_EMPTY_CORRECTIONS,
                                    mode::Symbol=:multilevel,
                                    stencil_radius::Integer=1,
                                    softening::Symbol=:hardy_c2)
        size(q, 1) == 3 || throw(ArgumentError("positions q must be 3×N"))
        N = size(q, 2)
        length(charges) == N || throw(ArgumentError("charges count must match positions"))
        plan = TernaryMSMPlan(q, box; rcut=rcut, grid_n=grid_n, levels=levels,
                              order=order, prefactor=prefactor, mode=mode,
                              stencil_radius=stencil_radius, softening=softening)
        all(i <= N && j <= N for ((i, j), _) in corrections.weights) ||
            throw(ArgumentError("correction index exceeds particle count"))
        return TernaryMSMCoulombFactor(N, Float64.(charges), box, plan, corrections)
    end

    function TernaryMSMCoulombFactor(q::AbstractMatrix{<:Real}, charges::AbstractVector{<:Real},
                                    box::AbstractMatrix{<:Real}; kwargs...)
        return TernaryMSMCoulombFactor(q, charges, Types.TriclinicBox(box); kwargs...)
    end

    function TernaryMSMCoulombFactor(q::AbstractMatrix{<:Real}, charges::AbstractVector{<:Real},
                                    box::AbstractVector{<:Real}; kwargs...)
        length(box) == 3 || throw(ArgumentError("box vector must have 3 lengths"))
        return TernaryMSMCoulombFactor(q, charges, Types.OrthorhombicBox((box[1], box[2], box[3])); kwargs...)
    end

    Factors.factor_support_size(f::TernaryMSMCoulombFactor) = f.N
    Factors.read_support(f::TernaryMSMCoulombFactor) = Tuple(1:f.N)
    Factors.is_local(::TernaryMSMCoulombFactor) = false
    Factors.force_consistent(::TernaryMSMCoulombFactor) = true

    function _validate_factor(f::TernaryMSMCoulombFactor, q::AbstractMatrix, ctx::Factors.Context)
        size(q) == (3, f.N) || throw(ArgumentError("q must be 3×$(f.N)"))
        ctx.N == f.N || throw(ArgumentError("context atom count $(ctx.N) != $(f.N)"))
        if f.box isa Types.OrthorhombicBox
            ctx.box isa Types.OrthorhombicBox || throw(Types.UnsupportedGeometry("context box must be OrthorhombicBox"))
            ctx.box.L == f.box.L || throw(ArgumentError("context box lengths differ from factor box"))
        else
            ctx.box isa Types.TriclinicBox || throw(Types.UnsupportedGeometry("context box must be TriclinicBox"))
            ctx.box.H == f.box.H || throw(ArgumentError("context box matrix differs from factor box"))
        end
        f.plan.N == f.N || throw(ArgumentError("plan atom count differs from factor atom count"))
        length(f.charges) == f.N || throw(ArgumentError("charges count differs from factor atom count"))
        return nothing
    end

    function Factors.evaluate!(F::AbstractMatrix{Float64}, f::TernaryMSMCoulombFactor,
                               q::AbstractMatrix{Float64}, ctx::Factors.Context)
        _validate_factor(f, q, ctx)
        return _msm_core!(F, nothing, f.plan, q, f.charges;
                          corrections=f.corrections, accumulate_F=true)
    end

    function msm_factor_energy(f::TernaryMSMCoulombFactor, q::AbstractMatrix{<:Real}, ctx::Factors.Context;
                               charges::AbstractVector{<:Real}=f.charges)
        _validate_factor(f, q, ctx)
        return ternary_msm_energy(f.plan, q, charges; corrections=f.corrections)
    end

    function msm_factor_box_derivative!(G::AbstractMatrix{Float64}, f::TernaryMSMCoulombFactor,
                                        q::AbstractMatrix{<:Real}, ctx::Factors.Context;
                                        charges::AbstractVector{<:Real}=f.charges,
                                        accumulate::Bool=false)
        _validate_factor(f, q, ctx)
        return ternary_msm_box_derivative!(G, f.plan, q, charges;
                                           corrections=f.corrections, accumulate=accumulate)
    end

    function msm_factor_box_derivative(f::TernaryMSMCoulombFactor, q::AbstractMatrix{<:Real},
                                       ctx::Factors.Context;
                                       charges::AbstractVector{<:Real}=f.charges)
        G = zeros(Float64, 3, 3)
        return msm_factor_box_derivative!(G, f, q, ctx; charges=charges, accumulate=false)
    end

    function msm_factor_energy_forces_box_derivative!(F::AbstractMatrix{Float64},
                                                      G::AbstractMatrix{Float64},
                                                      f::TernaryMSMCoulombFactor,
                                                      q::AbstractMatrix{<:Real},
                                                      ctx::Factors.Context;
                                                      charges::AbstractVector{<:Real}=f.charges,
                                                      accumulate_F::Bool=false,
                                                      accumulate_G::Bool=false)
        _validate_factor(f, q, ctx)
        return ternary_msm_energy_forces_box_derivative!(F, G, f.plan, q, charges;
                                                        corrections=f.corrections,
                                                        accumulate_F=accumulate_F,
                                                        accumulate_G=accumulate_G)
    end
end

end # module TernaryMSM
