"""Direct reference Ewald evaluator for orthorhombic periodic point charges.

This is an O(N^2) real-space and O(N*K) reciprocal-space evaluator, not PME.
Coordinates may be outside the primary cell; pair and reciprocal phases are
periodic. Energy units use the caller-supplied Coulomb prefactor.

For each canonical pair, `scaled` multiplies only the correction relative to
the full Ewald result: the evaluator adds `(scale-1)*prefactor*q[i]*q[j]/r`
at the minimum-image distance. `excluded` is scale zero. This is the standard
point-Coulomb pair correction convention, not a replacement Ewald model or an
automatic inference from bonded topology.
"""
module Electrostatics

using LinearAlgebra

const _PARENT = parentmodule(@__MODULE__)
const _HAS_FACTOR_PROTOCOL = isdefined(_PARENT, :Factors) && isdefined(_PARENT, :Types)
if _HAS_FACTOR_PROTOCOL
    const Factors = getfield(_PARENT, :Factors)
    const Types = getfield(_PARENT, :Types)
end
const _HAS_NEIGHBORS = isdefined(_PARENT, :Neighbors)
const Neighbors = _HAS_NEIGHBORS ? getfield(_PARENT, :Neighbors) : nothing
const _HAS_RESOLVED = isdefined(_PARENT, :ResolvedSystem)
const _RS = _HAS_RESOLVED ? getfield(_PARENT, :ResolvedSystem) : nothing

_erfc(x::Float64) = ccall(:erfc, Cdouble, (Cdouble,), x)

export evaluate_ewald, PairCorrections, EwaldFactor, ewald_energy,
       ewald_charge_gradient!, coulomb_pair_box_derivative,
       coulomb_pair_box_derivative!, coulomb_box_derivative,
       coulomb_pair_energy, DSFCoulombFactor, dsf_box_derivative,
       dsf_box_derivative!, FIXED_POINT_SCALE

const FIXED_POINT_SCALE = Float64(Int64(1) << 40)

struct PairCorrections
    weights::Dict{Tuple{Int,Int},Float64}
    function PairCorrections(; excluded=Tuple{Int,Int}[], scaled=Tuple{Int,Int,Float64}[])
        w = Dict{Tuple{Int,Int},Float64}()
        for p in excluded
            length(p) == 2 || throw(ArgumentError("excluded entries must be (i,j)"))
            i,j = _canonical(p[1], p[2])
            haskey(w,(i,j)) && throw(ArgumentError("duplicate pair correction"))
            w[(i,j)] = 0.0
        end
        for p in scaled
            length(p) == 3 || throw(ArgumentError("scaled entries must be (i,j,weight)"))
            i,j = _canonical(p[1], p[2]); s=Float64(p[3])
            isfinite(s) && s >= 0 || throw(ArgumentError("scale must be finite and nonnegative"))
            haskey(w,(i,j)) && throw(ArgumentError("duplicate pair correction"))
            w[(i,j)] = s
        end
        new(w)
    end
end

function _canonical(i,j)
    i isa Integer && j isa Integer && i > 0 && j > 0 && i != j ||
        throw(ArgumentError("pair indices must be distinct positive integers"))
    return min(Int(i),Int(j)), max(Int(i),Int(j))
end

"""Evaluate neutralizing-background Ewald energy and forces.

Arguments: `q` is 3×N Cartesian coordinates, `charges` has length N, `box`
contains three positive orthorhombic lengths. `rcut` must be less than half
the shortest box length. Reciprocal vectors are `2π*(nx/Lx,ny/Ly,nz/Lz)` for
each integer triplet in `-kmax:kmax`, excluding zero. Returns `(energy, forces)`;
 forces are `-∇U` with shape 3×N. Charges need not sum to zero: the uniform
 neutralizing-background term `-π Q²/(2 α² V)` is included.
"""
function evaluate_ewald(q::AbstractMatrix, charges::AbstractVector, box::AbstractVector;
                        alpha::Real, rcut::Real, kmax::Integer,
                        prefactor::Real=1.0, corrections::PairCorrections=PairCorrections())
    size(q,1)==3 || throw(ArgumentError("q must have shape 3×N"))
    n=size(q,2); length(charges)==n || throw(ArgumentError("charge count must match coordinates"))
    length(box)==3 || throw(ArgumentError("box must contain three lengths"))
    all(isfinite,q) && all(isfinite,charges) && all(isfinite,box) || throw(ArgumentError("inputs must be finite"))
    L=Float64.(box); all(>(0),L) || throw(ArgumentError("box lengths must be positive"))
    twopi = 2pi
    a=Float64(alpha); rc=Float64(rcut); p=Float64(prefactor)
    isfinite(a) && a>0 || throw(ArgumentError("alpha must be finite and positive"))
    isfinite(rc) && rc>0 && rc < minimum(L)/2 || throw(ArgumentError("require 0 < rcut < min(box)/2"))
    kmax >= 0 || throw(ArgumentError("kmax must be nonnegative"))
    isfinite(p) || throw(ArgumentError("prefactor must be finite"))
    all(i<=n && j<=n for (i,j) in keys(corrections.weights)) || throw(ArgumentError("correction index exceeds particle count"))
    qq=Float64.(charges); x=Float64.(q); V=prod(L); Q=sum(qq)
    E=0.0; F=zeros(Float64,3,n); rc2=rc^2
    # Minimum-image real-space pair sum. Cutoff is strict, as usual for Ewald.
    @inbounds for i in 1:n-1, j in i+1:n
        d1=x[1,i]-x[1,j]; d2=x[2,i]-x[2,j]; d3=x[3,i]-x[3,j]
        d1 -= L[1] * round(d1 / L[1])
        d2 -= L[2] * round(d2 / L[2])
        d3 -= L[3] * round(d3 / L[3])
        r2=d1*d1+d2*d2+d3*d3; r2==0 && throw(ArgumentError("coincident charges are singular"))
        r2 < rc2 || continue
        r=sqrt(r2); z=a*r; er=_erfc(z); c=p*qq[i]*qq[j]
        E += c*er/r
        dudr=c*(-2a/sqrt(pi)*exp(-z*z)/r-er/r2)
        scale = dudr / r
        F[1,i] -= scale*d1; F[2,i] -= scale*d2; F[3,i] -= scale*d3
        F[1,j] += scale*d1; F[2,j] += scale*d2; F[3,j] += scale*d3
    end
    # Reciprocal sum, full integer cube excluding k=0.
    @inbounds for nx in -kmax:kmax, ny in -kmax:kmax, nz in -kmax:kmax
        nx==0 && ny==0 && nz==0 && continue
        kx=twopi*nx/L[1]; ky=twopi*ny/L[2]; kz=twopi*nz/L[3]
        k2=kx*kx+ky*ky+kz*kz
        coeff=p*(2pi/V)*exp(-k2/(4a*a))/k2
        C=0.0; S=0.0
        for i in 1:n
            phase=kx*x[1,i]+ky*x[2,i]+kz*x[3,i]
            C+=qq[i]*cos(phase); S+=qq[i]*sin(phase)
        end
        E+=coeff*(C*C+S*S)
        for i in 1:n
            phase=kx*x[1,i]+ky*x[2,i]+kz*x[3,i]
            g=2coeff*qq[i]*(C*sin(phase)-S*cos(phase))
            F[1,i]+=g*kx; F[2,i]+=g*ky; F[3,i]+=g*kz
        end
    end
    E -= p*a/sqrt(pi)*sum(abs2,qq)
    E -= p*pi*Q^2/(2a*a*V)
    @inbounds for ((i,j),s) in corrections.weights
        d1=x[1,i]-x[1,j]; d2=x[2,i]-x[2,j]; d3=x[3,i]-x[3,j]
        d1 -= L[1]*round(d1/L[1]); d2 -= L[2]*round(d2/L[2]); d3 -= L[3]*round(d3/L[3])
        r2=d1*d1+d2*d2+d3*d3
        r2>0 || throw(ArgumentError("coincident corrected charges are singular"))
        r=sqrt(r2); c=p*(s-1)*qq[i]*qq[j]
        E+=c/r
        scale=c/r^3
        F[1,i]+=scale*d1; F[2,i]+=scale*d2; F[3,i]+=scale*d3
        F[1,j]-=scale*d1; F[2,j]-=scale*d2; F[3,j]-=scale*d3
    end
    return E,F
end

if _HAS_FACTOR_PROTOCOL
"""
    EwaldFactor(q, charges, box; alpha, rcut, kmax, prefactor=1,
                corrections=PairCorrections())

Nonlocal `Factors.AbstractFactor` adapter for direct orthorhombic Ewald. It
shares `evaluate_ewald` with the standalone evaluator and is intended for small
reference systems (at most 10,000 charges and `kmax <= 16`; direct O(N^2 + N(2kmax+1)^3)
work, not PME). The factor fixes atom count and box lengths. Only a validated
orthorhombic periodic context is accepted. `force_consistent` is true because
the direct evaluator's returned force differentiates its exact energy,
including pair corrections.
"""
mutable struct EwaldFactor <: Factors.AbstractFactor
    N::Int
    charges::Vector{Float64}
    box::Types.OrthorhombicBox
    alpha::Float64
    rcut::Float64
    kmax::Int
    prefactor::Float64
    corrections::PairCorrections
end

function EwaldFactor(q::AbstractMatrix{<:Real}, charges::AbstractVector{<:Real},
                     box::Types.OrthorhombicBox; alpha::Real, rcut::Real,
                     kmax::Integer, prefactor::Real=1.0,
                     corrections::PairCorrections=PairCorrections())
    size(q, 1) == 3 || throw(ArgumentError("q must have shape 3×N"))
    n = size(q, 2)
    n <= 10_000 || throw(ArgumentError("direct Ewald adapter supports at most 10000 charges; use a PME implementation for larger systems"))
    kmax <= 16 || throw(ArgumentError("direct Ewald adapter requires kmax <= 16"))
    evaluate_ewald(q, charges, collect(box.L); alpha=alpha, rcut=rcut,
                   kmax=kmax, prefactor=prefactor, corrections=corrections)
    return EwaldFactor(n, collect(Float64, charges), box, Float64(alpha),
                       Float64(rcut), Int(kmax), Float64(prefactor), corrections)
end

function EwaldFactor(::AbstractMatrix{<:Real}, ::AbstractVector{<:Real}, box;
                     kwargs...)
    box isa Types.TriclinicBox && throw(Types.UnsupportedGeometry(
        "EwaldFactor requires an orthorhombic periodic box; got $(typeof(box))"))
    throw(ArgumentError("EwaldFactor requires an orthorhombic periodic box; got $(typeof(box))"))
end

Factors.factor_support_size(f::EwaldFactor) = f.N
Factors.read_support(f::EwaldFactor) = Tuple(1:f.N)
Factors.is_local(::EwaldFactor) = false
Factors.force_consistent(::EwaldFactor) = true

function _validate_factor(f::EwaldFactor, q::AbstractMatrix, ctx::Factors.Context)
    size(q) == (3, f.N) || throw(ArgumentError("q must be 3×$(f.N)"))
    ctx.N == f.N || throw(ArgumentError("context atom count does not match Ewald factor"))
    ctx.box isa Types.OrthorhombicBox || throw(ArgumentError("EwaldFactor requires an orthorhombic periodic context box"))
    ctx.box.L == f.box.L || throw(ArgumentError("context box differs from Ewald factor box"))
    length(f.charges) == f.N || throw(ArgumentError("charge count no longer matches Ewald factor"))
    f.N <= 10_000 && f.kmax <= 16 || throw(ArgumentError("EwaldFactor size or reciprocal cutoff exceeds supported direct-evaluation limits"))
    return nothing
end

function Factors.evaluate!(F::AbstractMatrix{Float64}, f::EwaldFactor,
                           q::AbstractMatrix{Float64}, ctx::Factors.Context)
    size(F) == (3, f.N) || throw(ArgumentError("F must be 3×$(f.N)"))
    Base.mightalias(F, q) && throw(ArgumentError("force accumulator must not alias q"))
    _validate_factor(f, q, ctx)
    U, ef = evaluate_ewald(q, f.charges, collect(f.box.L); alpha=f.alpha,
                           rcut=f.rcut, kmax=f.kmax, prefactor=f.prefactor,
                           corrections=f.corrections)
    F .+= ef
    return U
end

"Evaluate the effective Ewald potential, optionally with replacement charges."
function ewald_energy(f::EwaldFactor, q::AbstractMatrix{<:Real}, ctx::Factors.Context;
                      charges::AbstractVector{<:Real}=f.charges)
    _validate_factor(f, q, ctx)
    return evaluate_ewald(q, charges, collect(f.box.L); alpha=f.alpha,
                          rcut=f.rcut, kmax=f.kmax, prefactor=f.prefactor,
                          corrections=f.corrections)[1]
end

"""Fill `out` with the analytic derivative of `ewald_energy` by charge."""
function ewald_charge_gradient!(out::AbstractVector{Float64}, f::EwaldFactor,
                                q::AbstractMatrix{<:Real}, ctx::Factors.Context;
                                charges::AbstractVector{<:Real}=f.charges)
    _validate_factor(f, q, ctx)
    length(charges) == f.N || throw(ArgumentError("charge count must match factor atom count"))
    length(out) == f.N || throw(ArgumentError("gradient length must match factor atom count"))
    all(isfinite, charges) || throw(ArgumentError("charges must be finite"))
    evaluate_ewald(q, charges, collect(f.box.L); alpha=f.alpha, rcut=f.rcut,
                   kmax=f.kmax, prefactor=f.prefactor, corrections=f.corrections)
    qq = Float64.(charges); x = Float64.(q); L = collect(f.box.L)
    twopi = 2pi
    fill!(out, 0.0)
    @inbounds for i in 1:f.N-1, j in i+1:f.N
        d1=x[1,i]-x[1,j]; d2=x[2,i]-x[2,j]; d3=x[3,i]-x[3,j]
        d1 -= L[1]*round(d1/L[1]); d2 -= L[2]*round(d2/L[2]); d3 -= L[3]*round(d3/L[3])
        r = sqrt(d1*d1+d2*d2+d3*d3)
        if r < f.rcut
            v = f.prefactor * _erfc(f.alpha*r) / r
            out[i] += qq[j]*v; out[j] += qq[i]*v
        end
    end
    V = prod(L); a = f.alpha; p = f.prefactor; Q = sum(qq)
    @inbounds for nx in -f.kmax:f.kmax, ny in -f.kmax:f.kmax, nz in -f.kmax:f.kmax
        nx == 0 && ny == 0 && nz == 0 && continue
        kx=twopi*nx/L[1]; ky=twopi*ny/L[2]; kz=twopi*nz/L[3]
        k2=kx*kx+ky*ky+kz*kz
        coeff = p*(twopi/V)*exp(-k2/(4a*a))/k2
        C = 0.0; S = 0.0
        for i in 1:f.N
            phase=kx*x[1,i]+ky*x[2,i]+kz*x[3,i]
            C += qq[i]*cos(phase); S += qq[i]*sin(phase)
        end
        for i in 1:f.N
            phase=kx*x[1,i]+ky*x[2,i]+kz*x[3,i]
            out[i] += 2coeff*(C*cos(phase) + S*sin(phase))
        end
    end
    out .-= (2p*a/sqrt(pi)) .* qq
    out .-= p*pi*Q/(a*a*V)
    @inbounds for ((i,j),s) in f.corrections.weights
        d1=x[1,i]-x[1,j]; d2=x[2,i]-x[2,j]; d3=x[3,i]-x[3,j]
        d1 -= L[1]*round(d1/L[1]); d2 -= L[2]*round(d2/L[2]); d3 -= L[3]*round(d3/L[3])
        v = p*(s-1)/sqrt(d1*d1+d2*d2+d3*d3)
        out[i] += v*qq[j]; out[j] += v*qq[i]
    end
    return out
end
end # _HAS_FACTOR_PROTOCOL

# ---------------------------------------------------------------------------
# Coulomb pair box derivative: ∂U/∂h|_s
#
# U = prefactor * coulomb_scale * qi * qj / r
# r = |d| = |h * w| where w is the fractional displacement held fixed.
# ∂U/∂h[a,b] = (dU/dr / r) * d[a] * w[b]
#            = - (prefactor * coulomb_scale * qi * qj / r^3) * d[a] * w[b]
# ---------------------------------------------------------------------------

_is_nobox(::Nothing) = true
_is_nobox(box) = (typeof(box) === Symbol ? box === :NoBox :
                  (nameof(typeof(box)) === :NoBox ||
                   (_HAS_FACTOR_PROTOCOL && box isa Types.NoBox)))

_is_orthorhombic(box::AbstractVector{<:Real}) = length(box) == 3
_is_orthorhombic(box::NTuple{3,<:Real}) = true
_is_orthorhombic(box) = (hasproperty(box, :L) && length(box.L) == 3) ||
                        nameof(typeof(box)) === :OrthorhombicBox

_ortho_lengths(box::AbstractVector{<:Real}) = (Float64(box[1]), Float64(box[2]), Float64(box[3]))
_ortho_lengths(box::NTuple{3,<:Real}) = (Float64(box[1]), Float64(box[2]), Float64(box[3]))
_ortho_lengths(box) = (Float64(box.L[1]), Float64(box.L[2]), Float64(box.L[3]))

_is_triclinic(box::AbstractMatrix{<:Real}) = size(box) == (3, 3)
_is_triclinic(box) = (hasproperty(box, :H) && hasproperty(box, :Hinv)) ||
                     nameof(typeof(box)) === :TriclinicBox

function _triclinic_matrices(box::AbstractMatrix{<:Real})
    H = Matrix{Float64}(box)
    Hinv = inv(H)
    return H, Hinv
end
function _triclinic_matrices(box)
    H = Matrix{Float64}(box.H)
    Hinv = Matrix{Float64}(box.Hinv)
    return H, Hinv
end

function _displacement_and_frac(q::AbstractMatrix{<:Real}, i::Integer, j::Integer, box)
    dx = Float64(q[1, i] - q[1, j])
    dy = Float64(q[2, i] - q[2, j])
    dz = Float64(q[3, i] - q[3, j])

    if _is_orthorhombic(box)
        L = _ortho_lengths(box)
        all(isfinite, L) && all(>(0.0), L) || throw(ArgumentError("orthorhombic box lengths must be positive and finite"))
        mx = dx - L[1] * round(dx / L[1])
        my = dy - L[2] * round(dy / L[2])
        mz = dz - L[3] * round(dz / L[3])
        d = (mx, my, mz)
        w = (mx / L[1], my / L[2], mz / L[3])
        return d, w
    elseif _is_triclinic(box)
        H, Hinv = _triclinic_matrices(box)
        all(isfinite, H) && all(isfinite, Hinv) || throw(ArgumentError("triclinic box matrices must be finite"))
        delta = [dx, dy, dz]
        f = Hinv * delta
        n0 = round.(Int, f)
        best = delta - H * n0
        best2 = sum(abs2, best)
        for a in -2:2, b in -2:2, c in -2:2
            r_img = delta - H * [a, b, c]
            r2_img = sum(abs2, r_img)
            if r2_img < best2
                best = r_img
                best2 = r2_img
            end
        end
        w_vec = Hinv * best
        d = (best[1], best[2], best[3])
        w = (w_vec[1], w_vec[2], w_vec[3])
        return d, w
    else
        throw(ArgumentError("unsupported box type: $(typeof(box))"))
    end
end

"""
    coulomb_pair_box_derivative(q, i, j, qi, qj, box; prefactor=1.0, coulomb_scale=1.0) -> Matrix{Float64}
    coulomb_pair_box_derivative(f, q, box; prefactor=1.0) -> Matrix{Float64}
    coulomb_pair_box_derivative(q, charges, i, j, box; prefactor=1.0, coulomb_scale=1.0) -> Matrix{Float64}

Compute analytic periodic box derivative `G[a,b] = ∂U/∂h[a,b]` at fixed fractional coordinates `s`
for a point-Coulomb pair interaction between sites `i` and `j`.

Supports:
- `OrthorhombicBox` (or 3-element box length vector): `w = d ./ L`
- `TriclinicBox` (or 3x3 lattice matrix `H` with columns as cell vectors): `w = Hinv * d`
- `NoBox`: throws `ArgumentError` (box derivative is undefined without lattice vectors).
"""
function coulomb_pair_box_derivative(q::AbstractMatrix{<:Real}, i::Integer, j::Integer,
                                     qi::Real, qj::Real, box;
                                     prefactor::Real=1.0, coulomb_scale::Real=1.0)
    G = zeros(Float64, 3, 3)
    coulomb_pair_box_derivative!(G, q, i, j, qi, qj, box;
                                 prefactor=prefactor, coulomb_scale=coulomb_scale)
    return G
end

function coulomb_pair_box_derivative(q::AbstractMatrix{<:Real}, charges::AbstractVector{<:Real},
                                     i::Integer, j::Integer, box;
                                     prefactor::Real=1.0, coulomb_scale::Real=1.0)
    length(charges) >= max(i, j) || throw(ArgumentError("charges length must cover pair indices"))
    return coulomb_pair_box_derivative(q, i, j, charges[i], charges[j], box;
                                       prefactor=prefactor, coulomb_scale=coulomb_scale)
end

function coulomb_pair_box_derivative(f, q::AbstractMatrix{<:Real}, box; prefactor::Real=1.0)
    scale = hasproperty(f, :coulomb_scale) ? Float64(f.coulomb_scale) : 1.0
    return coulomb_pair_box_derivative(q, Int(f.i), Int(f.j), Float64(f.qi), Float64(f.qj), box;
                                       prefactor=prefactor, coulomb_scale=scale)
end

const coulomb_box_derivative = coulomb_pair_box_derivative

"""
    coulomb_pair_box_derivative!(G, q, i, j, qi, qj, box; prefactor=1.0, coulomb_scale=1.0)

In-place accumulation of the analytic periodic box derivative `G[a,b] += ∂U/∂h[a,b]`.
"""
function coulomb_pair_box_derivative!(G::AbstractMatrix{Float64},
                                      q::AbstractMatrix{<:Real}, i::Integer, j::Integer,
                                      qi::Real, qj::Real, box;
                                      prefactor::Real=1.0, coulomb_scale::Real=1.0)
    size(G) == (3, 3) || throw(DimensionMismatch("G must be 3x3"))
    size(q, 1) == 3 || throw(ArgumentError("q must have shape 3xN"))
    n = size(q, 2)
    1 <= i <= n && 1 <= j <= n || throw(ArgumentError("indices $i, $j out of bounds (1:$n)"))
    i != j || throw(ArgumentError("pair indices must be distinct positive integers"))
    all(isfinite, q) || throw(ArgumentError("coordinates must be finite"))
    isfinite(qi) && isfinite(qj) || throw(ArgumentError("charges must be finite"))
    isfinite(prefactor) && isfinite(coulomb_scale) || throw(ArgumentError("prefactor and scale must be finite"))

    if _is_nobox(box)
        throw(ArgumentError("no periodic box: box derivative ∂U/∂h is undefined for NoBox"))
    end

    d, w = _displacement_and_frac(q, i, j, box)
    r2 = d[1]^2 + d[2]^2 + d[3]^2
    r2 > 1e-24 || throw(ArgumentError("coincident charges are singular"))
    r = sqrt(r2)

    coeff = -Float64(prefactor) * Float64(coulomb_scale) * Float64(qi) * Float64(qj) / (r2 * r)
    @inbounds for b in 1:3, a in 1:3
        G[a, b] += coeff * d[a] * w[b]
    end
    return G
end

"""
    coulomb_pair_energy(q, i, j, qi, qj, box; prefactor=1.0, coulomb_scale=1.0) -> Float64
    coulomb_pair_energy(f, q, box; prefactor=1.0) -> Float64

Evaluate minimum-image Coulomb pair potential energy `U = prefactor * coulomb_scale * qi * qj / r_mic`.
"""
function coulomb_pair_energy(q::AbstractMatrix{<:Real}, i::Integer, j::Integer,
                             qi::Real, qj::Real, box;
                             prefactor::Real=1.0, coulomb_scale::Real=1.0)
    if _is_nobox(box)
        dx = Float64(q[1, i] - q[1, j])
        dy = Float64(q[2, i] - q[2, j])
        dz = Float64(q[3, i] - q[3, j])
        r2 = dx*dx + dy*dy + dz*dz
        r2 > 1e-24 || throw(ArgumentError("coincident charges are singular"))
        return Float64(prefactor) * Float64(coulomb_scale) * Float64(qi) * Float64(qj) / sqrt(r2)
    end
    d, _ = _displacement_and_frac(q, i, j, box)
    r2 = d[1]^2 + d[2]^2 + d[3]^2
    r2 > 1e-24 || throw(ArgumentError("coincident charges are singular"))
    return Float64(prefactor) * Float64(coulomb_scale) * Float64(qi) * Float64(qj) / sqrt(r2)
end

function coulomb_pair_energy(f, q::AbstractMatrix{<:Real}, box; prefactor::Real=1.0)
    scale = hasproperty(f, :coulomb_scale) ? Float64(f.coulomb_scale) : 1.0
    return coulomb_pair_energy(q, Int(f.i), Int(f.j), Float64(f.qi), Float64(f.qj), box;
                               prefactor=prefactor, coulomb_scale=scale)
end

if _HAS_FACTOR_PROTOCOL
    function coulomb_pair_box_derivative(f::Factors.CoulombPair, q::AbstractMatrix{Float64}, ctx::Factors.Context)
        return coulomb_pair_box_derivative(f, q, ctx.box; prefactor=ctx.units.coulomb_prefactor)
    end
    function coulomb_pair_energy(f::Factors.CoulombPair, q::AbstractMatrix{Float64}, ctx::Factors.Context)
        return coulomb_pair_energy(f, q, ctx.box; prefactor=ctx.units.coulomb_prefactor)
    end

    const TileCacheType = _HAS_NEIGHBORS ? Union{Nothing, Neighbors.TilePairCache} : Any

    """
        DSFCoulombFactor <: Factors.AbstractFactor

    Implements Fennell & Gezelter (J. Chem. Phys. 124, 234104, 2006) C¹-continuous
    Damped Shifted Force (DSF) periodic electrostatics.
    """
    struct DSFCoulombFactor <: Factors.AbstractFactor
        charges::Vector{Float64}
        alpha::Float64
        rcut::Float64
        prefactor::Float64
        corrections::PairCorrections
        include_self::Bool
        tile_cache::TileCacheType
        fixed_point::Bool
    end

    function DSFCoulombFactor(charges::AbstractVector{<:Real};
                              alpha::Real = 0.2,
                              rcut::Real = 10.0,
                              prefactor::Real = 1.0,
                              corrections::PairCorrections = PairCorrections(),
                              include_self::Bool = false,
                              tile_cache::TileCacheType = nothing,
                              fixed_point::Bool = false)
        all(isfinite, charges) || throw(ArgumentError("charges must be finite"))
        a = Float64(alpha)
        rc = Float64(rcut)
        p = Float64(prefactor)
        isfinite(a) && a >= 0 || throw(ArgumentError("alpha must be finite and nonnegative"))
        isfinite(rc) && rc > 0 || throw(ArgumentError("rcut must be finite and positive"))
        isfinite(p) || throw(ArgumentError("prefactor must be finite"))
        return DSFCoulombFactor(collect(Float64, charges), a, rc, p, corrections,
                                include_self, tile_cache, fixed_point)
    end

    Factors.factor_support_size(f::DSFCoulombFactor) = length(f.charges)
    Factors.read_support(f::DSFCoulombFactor) = Tuple(1:length(f.charges))
    Factors.is_local(::DSFCoulombFactor) = false
    Factors.force_consistent(::DSFCoulombFactor) = true

    function Factors.evaluate!(F::AbstractMatrix{Float64}, f::DSFCoulombFactor,
                               q::AbstractMatrix{Float64}, ctx)
        box = hasproperty(ctx, :box) ? ctx.box : ctx
        n = length(f.charges)
        size(q, 1) == 3 && size(q, 2) == n ||
            throw(ArgumentError("q must have shape 3 x $n"))
        size(F) == size(q) ||
            throw(ArgumentError("F must have shape 3 x $n"))
        Base.mightalias(F, q) && throw(ArgumentError("force accumulator must not alias q"))

        if _is_orthorhombic(box)
            L = _ortho_lengths(box)
            all(isfinite, L) && all(>(0.0), L) || throw(ArgumentError("orthorhombic box lengths must be positive and finite"))
            2.0 * f.rcut < minimum(L) || throw(ArgumentError("2 * rcut must be less than minimum box length"))
        elseif _is_triclinic(box)
            min_w = hasproperty(box, :widths) ? minimum(box.widths) : begin
                _, Hinv = _triclinic_matrices(box)
                minimum(ntuple(i -> inv(norm(view(Hinv, i, :))), 3))
            end
            2.0 * f.rcut < min_w || throw(ArgumentError("2 * rcut must be less than minimum box width"))
        elseif !_is_nobox(box)
            throw(ArgumentError("unsupported box type: $(typeof(box))"))
        end

        rc = f.rcut
        rc2 = rc * rc
        alpha = f.alpha
        pref = f.prefactor
        two_alpha_sqrt_pi = 2.0 * alpha / sqrt(pi)

        erfc_rc = _erfc(alpha * rc)
        exp_rc = exp(-alpha * alpha * rc2)
        Arc = erfc_rc / rc2 + two_alpha_sqrt_pi * exp_rc / rc
        Brc = erfc_rc / rc + Arc * rc

        u_total = 0.0
        if f.include_self
            u_total += -pref * (erfc_rc / (2.0 * rc) + alpha / sqrt(pi)) * sum(abs2, f.charges)
        end

        tc = f.tile_cache
        if tc !== nothing && _HAS_NEIGHBORS && tc isa Neighbors.TilePairCache
            Neighbors.refresh_tile_cache!(tc, q, box)
            tpi = tc.tile_pairs_i
            tpj = tc.tile_pairs_j
            ord = tc.order
            tstarts = tc.tile_starts
            tends = tc.tile_ends

            if f.fixed_point
                F_int = zeros(Int64, 3, n)
                @inbounds for k in eachindex(tpi)
                    ti = Int(tpi[k]); tj = Int(tpj[k])
                    si, ei = Int(tstarts[ti]), Int(tends[ti])
                    if ti == tj
                        for a in si:(ei - 1)
                            i = Int(ord[a])
                            for b in (a + 1):ei
                                j = Int(ord[b])
                                ii, jj = i < j ? (i, j) : (j, i)
                                scale = get(f.corrections.weights, (ii, jj), 1.0)
                                scale == 0.0 && continue
                                qiqj = f.charges[ii] * f.charges[jj]
                                qiqj == 0.0 && continue
                                d = Types.unwrap_displacement(box, q, ii, jj)
                                r2 = Types.norm2(d)
                                r2 >= rc2 && continue
                                r2 < 1e-24 && return Inf
                                r = sqrt(r2)
                                z = alpha * r
                                er = _erfc(z)
                                exp_z = exp(-z * z)
                                u_pair = scale * pref * qiqj * (er / r - Brc + Arc * r)
                                coeff = scale * pref * qiqj * (er / r2 + two_alpha_sqrt_pi * exp_z / r - Arc) / r
                                u_total += u_pair
                                gx = coeff * d.x; gy = coeff * d.y; gz = coeff * d.z
                                fx_int = round(Int64, gx * FIXED_POINT_SCALE)
                                fy_int = round(Int64, gy * FIXED_POINT_SCALE)
                                fz_int = round(Int64, gz * FIXED_POINT_SCALE)
                                F_int[1, ii] += fx_int; F_int[2, ii] += fy_int; F_int[3, ii] += fz_int
                                F_int[1, jj] -= fx_int; F_int[2, jj] -= fy_int; F_int[3, jj] -= fz_int
                            end
                        end
                    else
                        sj, ej = Int(tstarts[tj]), Int(tends[tj])
                        for a in si:ei
                            i = Int(ord[a])
                            for b in sj:ej
                                j = Int(ord[b])
                                ii, jj = i < j ? (i, j) : (j, i)
                                scale = get(f.corrections.weights, (ii, jj), 1.0)
                                scale == 0.0 && continue
                                qiqj = f.charges[ii] * f.charges[jj]
                                qiqj == 0.0 && continue
                                d = Types.unwrap_displacement(box, q, ii, jj)
                                r2 = Types.norm2(d)
                                r2 >= rc2 && continue
                                r2 < 1e-24 && return Inf
                                r = sqrt(r2)
                                z = alpha * r
                                er = _erfc(z)
                                exp_z = exp(-z * z)
                                u_pair = scale * pref * qiqj * (er / r - Brc + Arc * r)
                                coeff = scale * pref * qiqj * (er / r2 + two_alpha_sqrt_pi * exp_z / r - Arc) / r
                                u_total += u_pair
                                gx = coeff * d.x; gy = coeff * d.y; gz = coeff * d.z
                                fx_int = round(Int64, gx * FIXED_POINT_SCALE)
                                fy_int = round(Int64, gy * FIXED_POINT_SCALE)
                                fz_int = round(Int64, gz * FIXED_POINT_SCALE)
                                F_int[1, ii] += fx_int; F_int[2, ii] += fy_int; F_int[3, ii] += fz_int
                                F_int[1, jj] -= fx_int; F_int[2, jj] -= fy_int; F_int[3, jj] -= fz_int
                            end
                        end
                    end
                end
                inv_scale = 1.0 / FIXED_POINT_SCALE
                @inbounds for a in 1:n, m in 1:3
                    F[m, a] += Float64(F_int[m, a]) * inv_scale
                end
            else
                @inbounds for k in eachindex(tpi)
                    ti = Int(tpi[k]); tj = Int(tpj[k])
                    si, ei = Int(tstarts[ti]), Int(tends[ti])
                    if ti == tj
                        for a in si:(ei - 1)
                            i = Int(ord[a])
                            for b in (a + 1):ei
                                j = Int(ord[b])
                                ii, jj = i < j ? (i, j) : (j, i)
                                scale = get(f.corrections.weights, (ii, jj), 1.0)
                                scale == 0.0 && continue
                                qiqj = f.charges[ii] * f.charges[jj]
                                qiqj == 0.0 && continue
                                d = Types.unwrap_displacement(box, q, ii, jj)
                                r2 = Types.norm2(d)
                                r2 >= rc2 && continue
                                r2 < 1e-24 && return Inf
                                r = sqrt(r2)
                                z = alpha * r
                                er = _erfc(z)
                                exp_z = exp(-z * z)
                                u_pair = scale * pref * qiqj * (er / r - Brc + Arc * r)
                                coeff = scale * pref * qiqj * (er / r2 + two_alpha_sqrt_pi * exp_z / r - Arc) / r
                                u_total += u_pair
                                gx = coeff * d.x; gy = coeff * d.y; gz = coeff * d.z
                                F[1, ii] += gx; F[2, ii] += gy; F[3, ii] += gz
                                F[1, jj] -= gx; F[2, jj] -= gy; F[3, jj] -= gz
                        end
                    end
                else
                    sj, ej = Int(tstarts[tj]), Int(tends[tj])
                    for a in si:ei
                        i = Int(ord[a])
                        for b in sj:ej
                            j = Int(ord[b])
                            ii, jj = i < j ? (i, j) : (j, i)
                            scale = get(f.corrections.weights, (ii, jj), 1.0)
                            scale == 0.0 && continue
                            qiqj = f.charges[ii] * f.charges[jj]
                            qiqj == 0.0 && continue
                            d = Types.unwrap_displacement(box, q, ii, jj)
                            r2 = Types.norm2(d)
                            r2 >= rc2 && continue
                            r2 < 1e-24 && return Inf
                            r = sqrt(r2)
                            z = alpha * r
                            er = _erfc(z)
                            exp_z = exp(-z * z)
                            u_pair = scale * pref * qiqj * (er / r - Brc + Arc * r)
                            coeff = scale * pref * qiqj * (er / r2 + two_alpha_sqrt_pi * exp_z / r - Arc) / r
                            u_total += u_pair
                            gx = coeff * d.x; gy = coeff * d.y; gz = coeff * d.z
                            F[1, ii] += gx; F[2, ii] += gy; F[3, ii] += gz
                            F[1, jj] -= gx; F[2, jj] -= gy; F[3, jj] -= gz
                        end
                    end
                end
            end
        end
    else
        if f.fixed_point
            F_int = zeros(Int64, 3, n)
            @inbounds for i in 1:n-1
                qi = f.charges[i]
                for j in i+1:n
                    qj = f.charges[j]
                    scale = get(f.corrections.weights, (i, j), 1.0)
                    scale == 0.0 && continue
                    qiqj = qi * qj
                    qiqj == 0.0 && continue
                    d = Types.unwrap_displacement(box, q, i, j)
                    r2 = Types.norm2(d)
                    r2 >= rc2 && continue
                    r2 < 1e-24 && return Inf
                    r = sqrt(r2)
                    z = alpha * r
                    er = _erfc(z)
                    exp_z = exp(-z * z)
                    u_pair = scale * pref * qiqj * (er / r - Brc + Arc * r)
                    coeff = scale * pref * qiqj * (er / r2 + two_alpha_sqrt_pi * exp_z / r - Arc) / r
                    u_total += u_pair
                    gx = coeff * d.x; gy = coeff * d.y; gz = coeff * d.z
                    fx_int = round(Int64, gx * FIXED_POINT_SCALE)
                    fy_int = round(Int64, gy * FIXED_POINT_SCALE)
                    fz_int = round(Int64, gz * FIXED_POINT_SCALE)
                    F_int[1, i] += fx_int; F_int[2, i] += fy_int; F_int[3, i] += fz_int
                    F_int[1, j] -= fx_int; F_int[2, j] -= fy_int; F_int[3, j] -= fz_int
                end
            end
            inv_scale = 1.0 / FIXED_POINT_SCALE
            @inbounds for a in 1:n, m in 1:3
                F[m, a] += Float64(F_int[m, a]) * inv_scale
            end
        else
            @inbounds for i in 1:n-1
                qi = f.charges[i]
                for j in i+1:n
                    qj = f.charges[j]
                    scale = get(f.corrections.weights, (i, j), 1.0)
                    scale == 0.0 && continue
                    qiqj = qi * qj
                    qiqj == 0.0 && continue
                    d = Types.unwrap_displacement(box, q, i, j)
                    r2 = Types.norm2(d)
                    r2 >= rc2 && continue
                    r2 < 1e-24 && return Inf
                    r = sqrt(r2)
                    z = alpha * r
                    er = _erfc(z)
                    exp_z = exp(-z * z)
                    u_pair = scale * pref * qiqj * (er / r - Brc + Arc * r)
                    coeff = scale * pref * qiqj * (er / r2 + two_alpha_sqrt_pi * exp_z / r - Arc) / r
                    u_total += u_pair
                    gx = coeff * d.x; gy = coeff * d.y; gz = coeff * d.z
                    F[1, i] += gx; F[2, i] += gy; F[3, i] += gz
                    F[1, j] -= gx; F[2, j] -= gy; F[3, j] -= gz
                end
            end
        end
    end
    return u_total
end

    Factors.evaluate!(f::DSFCoulombFactor, q::AbstractMatrix{Float64}, F::AbstractMatrix{Float64}, ctx) =
        Factors.evaluate!(F, f, q, ctx)

    function _dsf_box_derivative_impl!(G::AbstractMatrix{Float64}, f::DSFCoulombFactor,
                                       q::AbstractMatrix{Float64}, box)
        size(G) == (3, 3) || throw(DimensionMismatch("G must be 3x3"))
        n = length(f.charges)
        size(q, 1) == 3 && size(q, 2) == n || throw(ArgumentError("q must have shape 3 x $n"))

        if _is_nobox(box)
            throw(ArgumentError("no periodic box: box derivative ∂U/∂h is undefined for NoBox"))
        end

        if _is_orthorhombic(box)
            L = _ortho_lengths(box)
            all(isfinite, L) && all(>(0.0), L) || throw(ArgumentError("orthorhombic box lengths must be positive and finite"))
            2.0 * f.rcut < minimum(L) || throw(ArgumentError("2 * rcut must be less than minimum box length"))
        elseif _is_triclinic(box)
            min_w = hasproperty(box, :widths) ? minimum(box.widths) : begin
                _, Hinv = _triclinic_matrices(box)
                minimum(ntuple(i -> inv(norm(view(Hinv, i, :))), 3))
            end
            2.0 * f.rcut < min_w || throw(ArgumentError("2 * rcut must be less than minimum box width"))
        else
            throw(ArgumentError("unsupported box type: $(typeof(box))"))
        end

        rc = f.rcut
        rc2 = rc * rc
        alpha = f.alpha
        pref = f.prefactor
        two_alpha_sqrt_pi = 2.0 * alpha / sqrt(pi)

        erfc_rc = _erfc(alpha * rc)
        exp_rc = exp(-alpha * alpha * rc2)
        Arc = erfc_rc / rc2 + two_alpha_sqrt_pi * exp_rc / rc

        @inbounds for i in 1:n-1
            qi = f.charges[i]
            for j in i+1:n
                qj = f.charges[j]
                scale = get(f.corrections.weights, (i, j), 1.0)
                scale == 0.0 && continue
                qiqj = qi * qj
                qiqj == 0.0 && continue

                d, w = _displacement_and_frac(q, i, j, box)
                r2 = d[1]^2 + d[2]^2 + d[3]^2
                r2 >= rc2 && continue
                if r2 < 1e-24
                    G .= Inf
                    return G
                end
                r = sqrt(r2)
                z = alpha * r
                er = _erfc(z)
                exp_z = exp(-z * z)
                coeff = -scale * pref * qiqj * (er / r2 + two_alpha_sqrt_pi * exp_z / r - Arc) / r
                for b in 1:3, a in 1:3
                    G[a, b] += coeff * d[a] * w[b]
                end
            end
        end
        return G
    end

    dsf_box_derivative!(G::AbstractMatrix{Float64}, f::DSFCoulombFactor, q::AbstractMatrix{Float64}, box::Types.OrthorhombicBox) =
        _dsf_box_derivative_impl!(G, f, q, box)
    dsf_box_derivative!(G::AbstractMatrix{Float64}, f::DSFCoulombFactor, q::AbstractMatrix{Float64}, box::Types.TriclinicBox) =
        _dsf_box_derivative_impl!(G, f, q, box)
    dsf_box_derivative!(G::AbstractMatrix{Float64}, f::DSFCoulombFactor, q::AbstractMatrix{Float64}, box) =
        _dsf_box_derivative_impl!(G, f, q, box)

    function dsf_box_derivative(f::DSFCoulombFactor, q::AbstractMatrix{Float64}, box)
        G = zeros(Float64, 3, 3)
        dsf_box_derivative!(G, f, q, box)
        return G
    end

    if _HAS_RESOLVED
        _RS.accumulate_box_derivative!(G::Matrix{Float64}, f::DSFCoulombFactor,
                                        q::AbstractMatrix{Float64}, ctx::Factors.Context) =
            dsf_box_derivative!(G, f, q, ctx.box)
        _RS.box_derivative_support(::DSFCoulombFactor) = nothing
    end
end

end # module
