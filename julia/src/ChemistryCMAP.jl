"""
Periodic CMAP (backbone cross-map) correction: a pure-Julia port of OpenMM's
`CMAPTorsionForce`.

## Convention

`grid` is an `n × n` matrix; `grid[i,j]` is the energy (kJ/mol) of the map at

    φ_i = (i-1)Δ ,  ψ_j = (j-1)Δ ,  Δ = 2π/n.

That is the *flat* OpenMM map order `i + n*j` (first index fastest), which is
what `amber_file_parser.py` produces after its AMBER permutation

    flat[i + n*j] = prmtop[n*((j+n÷2) mod n) + ((i+n÷2) mod n)]      (0-based),

so the importer must hand rows = φ and columns = ψ in that order.

For the 8 atoms `(a1,a2,a3,a4,b1,b2,b3,b4)` the two arguments are the torsions

    φ = torsion(a1,a2,a3,a4) ,  ψ = torsion(b1,b2,b3,b4)

using the **same signed atan2 convention as `Factors.dihedral_angle_grad`**
(chirality `sign(det[b1,b2,b3])`). OpenMM's `ReferenceCMAPTorsionIxn` measures
the torsion from `(a2-a1, a2-a3, a4-a3)`; the extractor verified numerically
against Reference/CPU/OpenCL OpenMM with an analytic `cos(φ)+0.5 sin(ψ)` map
that the OpenMM map is evaluated at the **same** signed angle as
`Factors.dihedral_angle_grad`, not its negative. The extractor therefore
applies only the AMBER half-grid permutation and no axis reversal. A sign error
here is detectable by the analytic-map test.

## Energy and forces

The energy is the bicubic Hermite patch built from the periodic cubic spline of
the map along both axes — exactly `CMAPTorsionForceImpl::calcMapDerivatives`
(periodic spline second derivatives, nodal slopes, mixed derivative, the 16×16
`wt` coefficient table) and the patch polynomial in
`ReferenceCMAPTorsionIxn::calculateOneIxn`:

* the angle is wrapped with `fmod(angle + 2π, 2π)`,
* the patch is `s = min(floor(φ/Δ), n-1)` (likewise `t` for ψ),
* `E = Σ_{i,j} C[i,j] da^i db^j` with `da = φ/Δ - s`, `db = ψ/Δ - t`.

Forces use the chain rule, not OpenMM's `forceFactors` algebra:

    F = -∂E/∂φ ∇φ - ∂E/∂ψ ∇ψ,

with `∇φ` the four-tuple returned by `dihedral_angle_grad`. The map is locally
the identity on the wrapped angle, so wrapping changes the patch index only, not
the derivative.

`evaluate!` returns `Inf` at a degenerate (collinear) geometry, where
`dihedral_angle_grad` has no gradient, and scatters nothing.
"""
module ChemistryCMAP

using ..Factors

export CMAPCorrection, cmap_energy_grad

const _TWO_PI = 2.0 * pi

# Coefficient table from CMAPTorsionForceImpl.cpp / SplineFitter.cpp, indexed
# as _WT[k+1 + 16m] for k (4*aPower + bPower) and m (4*derivativeKind + corner).
# It is exactly the tensor-product bicubic Hermite basis
# (value/∂a/∂b/∂a∂b) at the corners (s,t), (s+1,t), (s+1,t+1), (s,t+1).
const _WT = Int[
     1, 0, -3, 2, 0, 0, 0, 0, -3, 0, 9, -6, 2, 0, -6, 4,
     0, 0, 0, 0, 0, 0, 0, 0, 3, 0, -9, 6, -2, 0, 6, -4,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, -6, 0, 0, -6, 4,
     0, 0, 3, -2, 0, 0, 0, 0, 0, 0, -9, 6, 0, 0, 6, -4,
     0, 0, 0, 0, 1, 0, -3, 2, -2, 0, 6, -4, 1, 0, -3, 2,
     0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 3, -2, 1, 0, -3, 2,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 2, 0, 0, 3, -2,
     0, 0, 0, 0, 0, 0, 3, -2, 0, 0, -6, 4, 0, 0, 3, -2,
     0, 1, -2, 1, 0, 0, 0, 0, 0, -3, 6, -3, 0, 2, -4, 2,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -6, 3, 0, -2, 4, -2,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, 2, -2,
     0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 3, -3, 0, 0, -2, 2,
     0, 0, 0, 0, 0, 1, -2, 1, 0, -2, 4, -2, 0, 1, -2, 1,
     0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 1, -2, 1,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, -1, 1,
     0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 2, -2, 0, 0, -1, 1,
]

# ---------------------------------------------------------------------------
# periodic cubic spline (port of SplineFitter::createPeriodicSpline)
# ---------------------------------------------------------------------------

function _solve_tridiagonal!(sol::Vector{Float64}, a::Vector{Float64},
                             b::Vector{Float64}, c::Vector{Float64},
                             rhs::Vector{Float64})
    m = length(a)
    gamma = Vector{Float64}(undef, m)
    @inbounds begin
        sol[1] = rhs[1] / b[1]
        beta = b[1]
        for i in 2:m
            gamma[i] = c[i-1] / beta
            beta = b[i] - a[i] * gamma[i]
            sol[i] = (rhs[i] - a[i] * sol[i-1]) / beta
        end
        for i in (m-1):-1:1
            sol[i] -= gamma[i+1] * sol[i+1]
        end
    end
    return sol
end

"""
Second derivatives of the periodic cubic spline through `y[1:n]` at the knots
`x[k] = (k-1)Δ`, with `y[n+1] = y[1]` and `x[n+1] = 2π`. Writes `m2[1:n+1]`
(`m2[n+1] = m2[1]`), which is the convention of OpenMM's `deriv` array: the
cyclic tridiagonal system is solved by Sherman–Morrison exactly as
`SplineFitter::createPeriodicSpline` does.
"""
function _periodic_second_derivatives!(m2::Vector{Float64}, y::Vector{Float64},
                                       x::Vector{Float64})
    N = length(x)
    n = N - 1
    a = Vector{Float64}(undef, n)
    b = Vector{Float64}(undef, n)
    c = Vector{Float64}(undef, n)
    rhs = Vector{Float64}(undef, n)
    @inbounds begin
        a[1] = x[N] - x[N-1]
        b[1] = 2.0 * ((x[2] - x[1]) + (x[N] - x[N-1]))
        c[1] = x[2] - x[1]
        rhs[1] = 6.0 * ((y[2] - y[1]) / (x[2] - x[1]) -
                        (y[N] - y[N-1]) / (x[N] - x[N-1]))
        for i in 2:n
            a[i] = x[i] - x[i-1]
            b[i] = 2.0 * (x[i+1] - x[i-1])
            c[i] = x[i+1] - x[i]
            rhs[i] = 6.0 * ((y[i+1] - y[i]) / (x[i+1] - x[i]) -
                            (y[i] - y[i-1]) / (x[i] - x[i-1]))
        end
    end
    beta = a[1]
    alpha = c[n]
    gamma = -b[1]
    b[1] -= gamma
    b[n] -= alpha * beta / gamma
    _solve_tridiagonal!(m2, a, b, c, rhs)
    u = zeros(n)
    z = zeros(n)
    u[1] = gamma
    u[n] = alpha
    _solve_tridiagonal!(z, a, b, c, u)
    scale = (m2[1] + beta * m2[n] / gamma) / (1.0 + z[1] + beta * z[n] / gamma)
    @inbounds for i in 1:n
        m2[i] -= scale * z[i]
    end
    m2[N] = m2[1]
    return m2
end

"""
Spline slope at each knot, evaluated on the right-hand segment — the value
`SplineFitter::evaluateSplineDerivative` returns at a knot. With uniform
spacing `Δ` and second derivatives `m2` this is

    (y[j+1] - y[j])/Δ + (-2 m2[j] - m2[j+1]) Δ/6 .
"""
function _knot_slopes!(s::Vector{Float64}, y::Vector{Float64},
                       m2::Vector{Float64}, delta::Float64, n::Int)
    @inbounds for j in 1:n
        s[j] = (y[j+1] - y[j]) / delta + (-2.0 * m2[j] - m2[j+1]) * delta / 6.0
    end
    return s
end

# ---------------------------------------------------------------------------
# coefficient construction (port of CMAPTorsionForceImpl::calcMapDerivatives)
# ---------------------------------------------------------------------------

"""
Build the `coeffs[4,4,n,n]` Hermite patch coefficients.

`coeffs[i+1,j+1,s+1,t+1]` is the coefficient of `da^i db^j` on patch `(s,t)`,
where `da = φ/Δ - s`, `db = ψ/Δ - t`. The patch data are the four corner
energies, the ∂/∂φ and ∂/∂ψ nodal slopes times `Δ`, and the mixed derivative
times `Δ²`, combined with OpenMM's `wt` table.
"""
function _cmap_coefficients(grid::Matrix{Float64}, n::Int)
    delta = _TWO_PI / n
    x = Vector{Float64}(undef, n + 1)
    @inbounds for k in 1:n+1
        x[k] = (k - 1) * delta
    end
    m2 = Vector{Float64}(undef, n + 1)
    col = Vector{Float64}(undef, n + 1)
    slope = Vector{Float64}(undef, n)
    d1 = Matrix{Float64}(undef, n, n)    # ∂E/∂φ at the grid nodes
    d2 = Matrix{Float64}(undef, n, n)    # ∂E/∂ψ at the grid nodes
    d12 = Matrix{Float64}(undef, n, n)   # ∂²E/∂φ∂ψ at the grid nodes

    # ∂/∂φ: one spline per column (fixed ψ).
    @inbounds for j in 1:n
        for i in 1:n
            col[i] = grid[i, j]
        end
        col[n+1] = grid[1, j]
        _periodic_second_derivatives!(m2, col, x)
        _knot_slopes!(slope, col, m2, delta, n)
        for i in 1:n
            d1[i, j] = slope[i]
        end
    end

    # ∂/∂ψ: one spline per row (fixed φ).
    @inbounds for i in 1:n
        for j in 1:n
            col[j] = grid[i, j]
        end
        col[n+1] = grid[i, 1]
        _periodic_second_derivatives!(m2, col, x)
        _knot_slopes!(slope, col, m2, delta, n)
        for j in 1:n
            d2[i, j] = slope[j]
        end
    end

    # Mixed derivative: spline of the ψ-derivative field along φ.
    @inbounds for j in 1:n
        for i in 1:n
            col[i] = d2[i, j]
        end
        col[n+1] = d2[1, j]
        _periodic_second_derivatives!(m2, col, x)
        _knot_slopes!(slope, col, m2, delta, n)
        for i in 1:n
            d12[i, j] = slope[i]
        end
    end

    coeffs = Array{Float64,4}(undef, 4, 4, n, n)
    rhs = Vector{Float64}(undef, 16)
    e = Vector{Float64}(undef, 4)
    e1 = Vector{Float64}(undef, 4)
    e2 = Vector{Float64}(undef, 4)
    e12 = Vector{Float64}(undef, 4)
    @inbounds for t in 1:n, s in 1:n
        sp = s == n ? 1 : s + 1
        tp = t == n ? 1 : t + 1
        e[1] = grid[s, t];  e1[1] = d1[s, t]
        e[2] = grid[sp, t]; e1[2] = d1[sp, t]
        e[3] = grid[sp, tp]; e1[3] = d1[sp, tp]
        e[4] = grid[s, tp]; e1[4] = d1[s, tp]
        e2[1] = d2[s, t]; e2[2] = d2[sp, t]
        e2[3] = d2[sp, tp]; e2[4] = d2[s, tp]
        e12[1] = d12[s, t]; e12[2] = d12[sp, t]
        e12[3] = d12[sp, tp]; e12[4] = d12[s, tp]
        for q in 1:4
            rhs[q] = e[q]
            rhs[q+4] = e1[q] * delta
            rhs[q+8] = e2[q] * delta
            rhs[q+12] = e12[q] * delta * delta
        end
        for k in 0:15
            acc = 0.0
            for m in 0:15
                acc += _WT[k+1+16*m] * rhs[m+1]
            end
            coeffs[(k >> 2) + 1, (k & 3) + 1, s, t] = acc
        end
    end
    return coeffs
end

# ---------------------------------------------------------------------------
# factor
# ---------------------------------------------------------------------------

"""
Periodic CMAP correction for one quadruple pair, `E(φ,ψ)` from a tabulated map.

The spline derivatives and the `4 × 4 × n × n` patch coefficients are computed
once in the constructor, so `evaluate!` only wraps the two angles, selects a
patch and evaluates four Horner polynomials (`cmap_energy_grad` allocates
nothing; the two `dihedral_angle_grad` calls cost the boxed union return that
every existing torsion factor pays).
"""
struct CMAPCorrection <: Factors.AbstractFactor
    atoms::NTuple{8,Int}
    grid::Matrix{Float64}
    coeffs::Array{Float64,4}   # [4, 4, n, n]

    function CMAPCorrection(atoms, grid::AbstractMatrix{<:Real})
        length(atoms) == 8 ||
            throw(ArgumentError("CMAPCorrection needs exactly 8 atom indices, got $(length(atoms))"))
        idx = ntuple(i -> Int(atoms[i]), 8)
        all(>(0), idx) ||
            throw(ArgumentError("CMAPCorrection atom indices must be positive, got $idx"))
        n = size(grid, 1)
        size(grid) == (n, n) ||
            throw(ArgumentError("CMAP grid must be n x n, got $(size(grid))"))
        n >= 3 ||
            throw(ArgumentError("CMAP grid must have at least 3 points per axis, got $n"))
        G = Matrix{Float64}(grid)
        all(isfinite, G) || throw(ArgumentError("CMAP grid must be finite"))
        return new(idx, G, _cmap_coefficients(G, n))
    end
end

Factors.factor_support_size(f::CMAPCorrection) = length(unique(f.atoms))
# The eight-slot definition is (a,b,c,d,b,c,d,e); its actual read-support is the
# five distinct atoms. Reporting the unique support keeps the Phase-1 resolved
# rule "a support does not repeat an atom" meaningful.
Factors.read_support(f::CMAPCorrection) = Tuple(unique(f.atoms))

"""
`(E, ∂E/∂φ, ∂E/∂ψ)` for the map at `(φ, ψ)`, both angles wrapped into `[0, 2π)`
with `fmod(angle + 2π, 2π)` and the patch selected by `s = min(floor(φ/Δ), n-1)`.
Allocation-free.
"""
@inline function cmap_energy_grad(f::CMAPCorrection, phi::Float64, psi::Float64)
    n = size(f.grid, 1)
    delta = _TWO_PI / n
    pw = mod(phi + _TWO_PI, _TWO_PI)
    qw = mod(psi + _TWO_PI, _TWO_PI)
    si = min(Int(floor(pw / delta)), n - 1) + 1
    ti = min(Int(floor(qw / delta)), n - 1) + 1
    da = pw / delta - (si - 1)
    db = qw / delta - (ti - 1)
    c = f.coeffs
    @inbounds begin
        c11 = c[1, 1, si, ti]; c12 = c[1, 2, si, ti]
        c13 = c[1, 3, si, ti]; c14 = c[1, 4, si, ti]
        c21 = c[2, 1, si, ti]; c22 = c[2, 2, si, ti]
        c23 = c[2, 3, si, ti]; c24 = c[2, 4, si, ti]
        c31 = c[3, 1, si, ti]; c32 = c[3, 2, si, ti]
        c33 = c[3, 3, si, ti]; c34 = c[3, 4, si, ti]
        c41 = c[4, 1, si, ti]; c42 = c[4, 2, si, ti]
        c43 = c[4, 3, si, ti]; c44 = c[4, 4, si, ti]
        # E = Σ_i P_i(db) da^(i-1), ∂E/∂da = Σ_i (i-1) P_i(db) da^(i-2)
        p1 = ((c14 * db + c13) * db + c12) * db + c11
        p2 = ((c24 * db + c23) * db + c22) * db + c21
        p3 = ((c34 * db + c33) * db + c32) * db + c31
        p4 = ((c44 * db + c43) * db + c42) * db + c41
        energy = ((p4 * da + p3) * da + p2) * da + p1
        dda = (3.0 * p4 * da + 2.0 * p3) * da + p2
        # Same along ψ: Q_j(da) = Σ_i c_ij da^(i-1).
        q1 = ((c41 * da + c31) * da + c21) * da + c11
        q2 = ((c42 * da + c32) * da + c22) * da + c12
        q3 = ((c43 * da + c33) * da + c23) * da + c13
        q4 = ((c44 * da + c34) * da + c24) * da + c14
        ddb = (3.0 * q4 * db + 2.0 * q3) * db + q2
    end
    return energy, dda / delta, ddb / delta
end

function Factors.evaluate!(F::AbstractMatrix{Float64}, f::CMAPCorrection,
                           q::AbstractMatrix{Float64}, ctx::Factors.Context)
    a1, a2, a3, a4, b1, b2, b3, b4 = f.atoms
    ta = Factors.dihedral_angle_grad(q, ctx.box, a1, a2, a3, a4)
    ta === nothing && return Inf
    tb = Factors.dihedral_angle_grad(q, ctx.box, b1, b2, b3, b4)
    tb === nothing && return Inf
    phi, (g1, g2, g3, g4) = ta
    psi, (h1, h2, h3, h4) = tb
    u, dphi, dpsi = cmap_energy_grad(f, phi, psi)
    # F = -∂E/∂φ ∇φ - ∂E/∂ψ ∇ψ
    ma = -dphi
    mb = -dpsi
    @inbounds for m in 1:3
        F[m, a1] += ma * Factors._comp(g1, m)
        F[m, a2] += ma * Factors._comp(g2, m)
        F[m, a3] += ma * Factors._comp(g3, m)
        F[m, a4] += ma * Factors._comp(g4, m)
        F[m, b1] += mb * Factors._comp(h1, m)
        F[m, b2] += mb * Factors._comp(h2, m)
        F[m, b3] += mb * Factors._comp(h3, m)
        F[m, b4] += mb * Factors._comp(h4, m)
    end
    return u
end

end # module ChemistryCMAP
