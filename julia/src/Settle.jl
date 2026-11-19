#= OpenMM source license notice for the adapted analytic position equations:

Copyright (c) 2013 Stanford University and the Authors.
Author: Peter Eastman.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE. =#
"""
Analytic SETTLE position correction for isolated rigid three-site waters.

The position equations follow the five-step analytic construction in
OpenMM's `ReferenceSETTLEAlgorithm.cpp` (Peter Eastman, Stanford University,
2013), distributed under the OpenMM MIT-style license; this implementation is
an independent Julia translation. The correction takes old coordinates and
unconstrained predicted coordinates, preserves their mass center, and solves
the rigid geometry with closed-form rotations, not iterative projection.

Coordinates must describe each constrained molecule intact and unwrapped;
the reference coordinates must already satisfy the three target distances.
This module performs direct coordinate differences and does not apply periodic
image corrections.

`settle!` is a projection convenience, not a complete integrator: it does not
apply the position correction as a momentum impulse. `position_multipliers`
returns multipliers from that impulse separately.

`project_velocities!` applies the mass-metric tangent correction at the
constrained coordinates. Its returned edge multipliers use `Δv_i = -∇c_i λ_i/m_i`
for `c=|x_j-x_i|²-d²`; the corresponding impulse virial is
`-2 Σ λ_i |r_i|²`. Divide impulses by dt for force multipliers, or by dt²
for position-projection impulse multipliers when using `Constraints`' virial.
"""
module Settle

using LinearAlgebra
import ..Types
import ..Constraints

export settle_positions!, position_multipliers, project_velocities!, project_momenta!, settle!, settle_step!

const C = Constraints
const T = Types

function _check(q, qpred, masses, cs)
    size(q) == (3, length(masses)) && size(qpred) == size(q) ||
        throw(ArgumentError("coordinates must be 3×natoms and match masses"))
    all(isfinite, q) && all(isfinite, qpred) || throw(ArgumentError("coordinates must be finite"))
    all(x -> isfinite(x) && x > 0, masses) || throw(ArgumentError("masses must be positive and finite"))
    length(cs.components) > 0 || throw(ArgumentError("constraint set is empty"))
    for comp in cs.components
        length(comp.atoms) == 3 && length(comp.edges) == 3 ||
            throw(ArgumentError("SETTLE accepts only three-atom, three-edge components"))
    end
end

@inline function _resolve_water_component(comp::C.ConstraintComponent, masses::AbstractVector{Float64})
    length(comp.atoms) == 3 && length(comp.edges) == 3 ||
        throw(ArgumentError("SETTLE needs three constraints per component"))
    ids = comp.atoms
    a1, a2, a3 = ids[1], ids[2], ids[3]
    m1, m2, m3 = masses[a1], masses[a2], masses[a3]
    # Identify heavy center (oxygen) and two equal-mass light sites (hydrogens)
    local i::Int, h1::Int, h2::Int
    if m1 > m2 && m1 > m3
        i, h1, h2 = a1, a2, a3
    elseif m2 > m1 && m2 > m3
        i, h1, h2 = a2, a1, a3
    elseif m3 > m1 && m3 > m2
        i, h1, h2 = a3, a1, a2
    else
        throw(ArgumentError("SETTLE requires one heavy center and two equal light sites"))
    end
    mh1, mh2 = masses[h1], masses[h2]
    abs(mh1 - mh2) <= 1e-12 * max(mh1, mh2) ||
        throw(ArgumentError("SETTLE requires equal hydrogen masses"))

    r1 = -1.0; r2 = -1.0; dHH = -1.0
    n_oh = 0; n_hh = 0
    for (ea, eb, d) in comp.edges
        u, v = ids[ea], ids[eb]
        if u == i || v == i
            n_oh += 1
            other = (u == i) ? v : u
            if other == h1
                r1 = d
            elseif other == h2
                r2 = d
            end
        elseif (u == h1 && v == h2) || (u == h2 && v == h1)
            n_hh += 1
            dHH = d
            # Match original edge order for (h1, h2)
            h1, h2 = u, v
        end
    end
    n_oh == 2 && r1 > 0 && r2 > 0 || throw(ArgumentError("water requires two center-site constraints"))
    n_hh == 1 && dHH > 0 || throw(ArgumentError("invalid water constraint topology"))
    abs(r1 - r2) <= 1e-10 * max(r1, r2) || throw(ArgumentError("SETTLE requires equal O-H distances"))
    return i, h1, h2, r1, dHH
end

@inline function _mic_vec(box::T.AbstractBox, q::AbstractMatrix{Float64}, j::Int, i::Int)
    # Displacement q[:, j] - q[:, i] under minimum-image convention
    if box isa T.NoBox
        return @inbounds (q[1, j] - q[1, i], q[2, j] - q[2, i], q[3, j] - q[3, i])
    else
        d = T.unwrap_displacement(box, q, j, i)
        return (d.x, d.y, d.z)
    end
end

@inline _cross3(a::NTuple{3,Float64}, b::NTuple{3,Float64}) = (
    a[2] * b[3] - a[3] * b[2],
    a[3] * b[1] - a[1] * b[3],
    a[1] * b[2] - a[2] * b[1]
)
@inline _dot3(a::NTuple{3,Float64}, b::NTuple{3,Float64}) = a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
@inline _norm3(a::NTuple{3,Float64}) = sqrt(_dot3(a, a))

"Apply SETTLE to predicted coordinates in place; return maximum squared-distance residual."
function settle_positions!(q::AbstractMatrix{Float64}, qpred::AbstractMatrix{Float64},
                           masses::AbstractVector{Float64}, cs::C.CoupledConstraintSet;
                           box::T.AbstractBox = T.NoBox())
    _check(q, qpred, masses, cs)
    Base.mightalias(q, qpred) && throw(ArgumentError("old and predicted coordinates must not alias"))
    max_res = 0.0
    for comp in cs.components
        i, h1, h2, r1, dHH = _resolve_water_component(comp, masses)

        # Unwrap reference and step displacements using minimum-image convention if box is periodic.
        a0 = @inbounds (q[1, i], q[2, i], q[3, i])
        b0 = _mic_vec(box, q, h1, i)
        c0 = _mic_vec(box, q, h2, i)

        # Step displacements qpred - q (under MIC if periodic wrap happened between q and qpred, or direct)
        if box isa T.NoBox
            xp0 = @inbounds (qpred[1, i] - q[1, i], qpred[2, i] - q[2, i], qpred[3, i] - q[3, i])
            xp1 = @inbounds (qpred[1, h1] - q[1, h1], qpred[2, h1] - q[2, h1], qpred[3, h1] - q[3, h1])
            xp2 = @inbounds (qpred[1, h2] - q[1, h2], qpred[2, h2] - q[2, h2], qpred[3, h2] - q[3, h2])
        elseif box isa T.OrthorhombicBox
            L = box.L
            @inbounds begin
                dx0 = qpred[1, i] - q[1, i]; dy0 = qpred[2, i] - q[2, i]; dz0 = qpred[3, i] - q[3, i]
                xp0 = (dx0 - L[1]*round(dx0/L[1]), dy0 - L[2]*round(dy0/L[2]), dz0 - L[3]*round(dz0/L[3]))
                dx1 = qpred[1, h1] - q[1, h1]; dy1 = qpred[2, h1] - q[2, h1]; dz1 = qpred[3, h1] - q[3, h1]
                xp1 = (dx1 - L[1]*round(dx1/L[1]), dy1 - L[2]*round(dy1/L[2]), dz1 - L[3]*round(dz1/L[3]))
                dx2 = qpred[1, h2] - q[1, h2]; dy2 = qpred[2, h2] - q[2, h2]; dz2 = qpred[3, h2] - q[3, h2]
                xp2 = (dx2 - L[1]*round(dx2/L[1]), dy2 - L[2]*round(dy2/L[2]), dz2 - L[3]*round(dz2/L[3]))
            end
        else
            tb = box::T.TriclinicBox
            d0 = [qpred[1, i] - q[1, i], qpred[2, i] - q[2, i], qpred[3, i] - q[3, i]]
            d0 .-= tb.H * round.(Int, tb.Hinv * d0)
            xp0 = (d0[1], d0[2], d0[3])
            d1 = [qpred[1, h1] - q[1, h1], qpred[2, h1] - q[2, h1], qpred[3, h1] - q[3, h1]]
            d1 .-= tb.H * round.(Int, tb.Hinv * d1)
            xp1 = (d1[1], d1[2], d1[3])
            d2 = [qpred[1, h2] - q[1, h2], qpred[2, h2] - q[2, h2], qpred[3, h2] - q[3, h2]]
            d2 .-= tb.H * round.(Int, tb.Hinv * d2)
            xp2 = (d2[1], d2[2], d2[3])
        end

        m0, m1, m2 = masses[i], masses[h1], masses[h2]
        mt = m0 + m1 + m2
        cm = (
            (m0 * xp0[1] + m1 * (b0[1] + xp1[1]) + m2 * (c0[1] + xp2[1])) / mt,
            (m0 * xp0[2] + m1 * (b0[2] + xp1[2]) + m2 * (c0[2] + xp2[2])) / mt,
            (m0 * xp0[3] + m1 * (b0[3] + xp1[3]) + m2 * (c0[3] + xp2[3])) / mt
        )
        xa = (xp0[1] - cm[1], xp0[2] - cm[2], xp0[3] - cm[3])
        xb = (b0[1] + xp1[1] - cm[1], b0[2] + xp1[2] - cm[2], b0[3] + xp1[3] - cm[3])
        xc = (c0[1] + xp2[1] - cm[1], c0[2] + xp2[2] - cm[2], c0[3] + xp2[3] - cm[3])

        az = _cross3(b0, c0)
        ax = _cross3(xa, az)
        ay = _cross3(az, ax)
        lx, ly, lz = _norm3(ax), _norm3(ay), _norm3(az)
        scale = max(r1^2, _dot3(xb, xb), _dot3(xc, xc), eps(Float64))
        minimum((lx, ly, lz)) > 1e-10 * scale || throw(ArgumentError("SETTLE refused singular predicted geometry"))

        ex = (ax[1]/lx, ax[2]/lx, ax[3]/lx)
        ey = (ay[1]/ly, ay[2]/ly, ay[3]/ly)
        ez = (az[1]/lz, az[2]/lz, az[3]/lz)

        # In the original formulation, R = hcat(ex, ey, ez), so columns of R are ex, ey, ez
        # and rows of R' are ex, ey, ez:
        b0d = (_dot3(ex, b0), _dot3(ey, b0), _dot3(ez, b0))
        c0d = (_dot3(ex, c0), _dot3(ey, c0), _dot3(ez, c0))
        zad = _dot3(ez, xa)
        xb1d = (_dot3(ex, xb), _dot3(ey, xb), _dot3(ez, xb))
        xc1d = (_dot3(ex, xc), _dot3(ey, xc), _dot3(ez, xc))

        rc = dHH / 2
        rb = sqrt(r1^2 - rc^2)
        ra = rb * (m1 + m2) / mt
        rb -= ra
        abs(ra) > 1e-14 || throw(ArgumentError("invalid SETTLE water geometry"))
        sphi = zad / ra
        abs(sphi) < 1 || throw(ArgumentError("predicted SETTLE step outside analytic domain"))
        cphi = sqrt(1 - sphi^2)
        abs(cphi) > 1e-12 || throw(ArgumentError("predicted SETTLE step outside analytic domain"))
        spsi = (xb1d[3] - xc1d[3]) / (2 * rc * cphi)
        abs(spsi) < 1 || throw(ArgumentError("predicted SETTLE step outside analytic domain"))
        cpsi = sqrt(1 - spsi^2)

        yA = ra * cphi
        xB = -rc * cpsi
        yB = -rb * cphi - rc * spsi * sphi
        yC = -rb * cphi + rc * spsi * sphi
        hh2 = 4 * xB^2 + (yB - yC)^2 + (xb1d[3] - xc1d[3])^2
        rad = 4 * xB^2 - hh2 + dHH^2
        rad >= -1e-12 * max(dHH^2, 1.0) || throw(ArgumentError("predicted SETTLE step outside analytic domain"))
        xB -= (2 * xB + sqrt(max(0.0, rad))) / 2

        alpha = xB * (b0d[1] - c0d[1]) + b0d[2] * yB + c0d[2] * yC
        beta  = xB * (c0d[2] - b0d[2]) + b0d[1] * yB + c0d[1] * yC
        gamma = b0d[1] * xb1d[2] - xb1d[1] * b0d[2] + c0d[1] * xc1d[2] - xc1d[1] * c0d[2]
        den = alpha^2 + beta^2
        den > 1e-24 * scale^2 || throw(ArgumentError("SETTLE rotational solve is singular"))
        discr = den - gamma^2
        discr >= -1e-12 * den || throw(ArgumentError("predicted SETTLE step outside analytic domain"))
        st = (alpha * gamma - beta * sqrt(max(0.0, discr))) / den
        abs(st) <= 1 + 1e-12 || throw(ArgumentError("predicted SETTLE rotation is invalid"))
        st = clamp(st, -1.0, 1.0)
        ct = sqrt(1 - st^2)

        a3 = (-yA * st, yA * ct, zad)
        b3 = (xB * ct - yB * st, xB * st + yB * ct, xb1d[3])
        c3 = (-xB * ct - yC * st, -xB * st + yC * ct, xc1d[3])

        xyzA = (ex[1]*a3[1] + ey[1]*a3[2] + ez[1]*a3[3],
                ex[2]*a3[1] + ey[2]*a3[2] + ez[2]*a3[3],
                ex[3]*a3[1] + ey[3]*a3[2] + ez[3]*a3[3])
        xyzB = (ex[1]*b3[1] + ey[1]*b3[2] + ez[1]*b3[3],
                ex[2]*b3[1] + ey[2]*b3[2] + ez[2]*b3[3],
                ex[3]*b3[1] + ey[3]*b3[2] + ez[3]*b3[3])
        xyzC = (ex[1]*c3[1] + ey[1]*c3[2] + ez[1]*c3[3],
                ex[2]*c3[1] + ey[2]*c3[2] + ez[2]*c3[3],
                ex[3]*c3[1] + ey[3]*c3[2] + ez[3]*c3[3])

        # Preserve periodic image offset of hydrogens relative to oxygen if reference was wrapped
        if box isa T.NoBox
            @inbounds for d in 1:3
                qpred[d, i]  = a0[d] + cm[d] + xyzA[d]
                qpred[d, h1] = a0[d] + cm[d] + xyzB[d]
                qpred[d, h2] = a0[d] + cm[d] + xyzC[d]
            end
        else
            shift1 = @inbounds (q[1, h1] - (a0[1] + b0[1]), q[2, h1] - (a0[2] + b0[2]), q[3, h1] - (a0[3] + b0[3]))
            shift2 = @inbounds (q[1, h2] - (a0[1] + c0[1]), q[2, h2] - (a0[2] + c0[2]), q[3, h2] - (a0[3] + c0[3]))
            @inbounds for d in 1:3
                qpred[d, i]  = a0[d] + cm[d] + xyzA[d]
                qpred[d, h1] = a0[d] + cm[d] + xyzB[d] + shift1[d]
                qpred[d, h2] = a0[d] + cm[d] + xyzC[d] + shift2[d]
            end
        end

        r_oh1 = _dot3((xyzB[1] - xyzA[1], xyzB[2] - xyzA[2], xyzB[3] - xyzA[3]),
                      (xyzB[1] - xyzA[1], xyzB[2] - xyzA[2], xyzB[3] - xyzA[3])) - r1^2
        r_oh2 = _dot3((xyzC[1] - xyzA[1], xyzC[2] - xyzA[2], xyzC[3] - xyzA[3]),
                      (xyzC[1] - xyzA[1], xyzC[2] - xyzA[2], xyzC[3] - xyzA[3])) - r1^2
        r_hh  = _dot3((xyzC[1] - xyzB[1], xyzC[2] - xyzB[2], xyzC[3] - xyzB[3]),
                      (xyzC[1] - xyzB[1], xyzC[2] - xyzB[2], xyzC[3] - xyzB[3])) - dHH^2
        max_res = max(max_res, abs(r_oh1), abs(r_oh2), abs(r_hh))
    end
    return max_res
end

"""
    position_multipliers(qreference, qpredicted, qcorrected, masses, constraints, dt; box=Types.NoBox())

Return per-component Lagrange multipliers for the coordinate impulse
`Δp = M(qcorrected-qpredicted)/dt`, using gradients at `qreference`. These are
impulse multipliers (momentum units divided by squared-distance units), not
static constraint forces and not a pressure estimator. The linearized impulse
virial is `-2Σλ|r_reference|²`; divide λ by `dt` for force-multiplier units.

The returned multipliers reproduce the complete coordinate impulse in the
old-geometry constraint-normal span to floating-point precision. Callers may
also form the complete impulse directly as `M*(qcorrected-qpredicted)/dt`.
"""
function position_multipliers(qreference::AbstractMatrix{Float64}, qpredicted::AbstractMatrix{Float64},
                              qcorrected::AbstractMatrix{Float64},
                              masses::AbstractVector{Float64}, cs::C.CoupledConstraintSet, dt::Real;
                              box::T.AbstractBox = T.NoBox())
    dt > 0 && isfinite(dt) || throw(ArgumentError("dt must be positive and finite"))
    size(qreference)==size(qpredicted)==size(qcorrected) && size(qreference,1)==3 && size(qreference,2)==length(masses) ||
        throw(ArgumentError("coordinate/mass shape mismatch"))
    all(isfinite,qreference) && all(isfinite,qpredicted) && all(isfinite,qcorrected) && all(x->isfinite(x)&&x>0,masses) ||
        throw(ArgumentError("coordinates and masses must be finite; masses positive"))
    result=Vector{Vector{Float64}}(undef, length(cs.components))
    G=zeros(3,9); A=zeros(3,3); rhs=zeros(3)
    for (cidx, comp) in enumerate(cs.components)
        length(comp.atoms)==3 && length(comp.edges)==3 || throw(ArgumentError("SETTLE accepts rigid triangles only"))
        ids=comp.atoms; fill!(G, 0.0); fill!(A, 0.0); fill!(rhs, 0.0)
        for (k,(a,b,d)) in enumerate(comp.edges)
            i,j=ids[a],ids[b]
            r_tup = _mic_vec(box, qreference, j, i)
            r = [r_tup[1], r_tup[2], r_tup[3]]
            G[k,3(a-1)+1:3a] .= -2r; G[k,3(b-1)+1:3b] .= 2r
            rhs[k]=-2dot(r, (qcorrected[:,j]-qpredicted[:,j])-(qcorrected[:,i]-qpredicted[:,i]))/dt
        end
        for atom in 1:3
            cols=3(atom-1)+1:3atom; A .+= (G[:,cols]*G[:,cols]')/masses[ids[atom]]
        end
        # λ solves G M^-1 G' λ = -G M^-1 Δp, with Δp=MΔq/dt.
        result[cidx] = Symmetric(A)\rhs
    end
    return result
end

"Project velocities in place onto the rigid tangent space; return edge multipliers."
function project_velocities!(q::AbstractMatrix{Float64}, v::AbstractMatrix{Float64},
                              masses::AbstractVector{Float64}, cs::C.CoupledConstraintSet;
                              box::T.AbstractBox = T.NoBox())
    size(q)==size(v) && size(q,1)==3 && size(q,2)==length(masses) || throw(ArgumentError("shape mismatch"))
    all(isfinite,q) && all(isfinite,v) && all(x->isfinite(x)&&x>0,masses) ||
        throw(ArgumentError("coordinates and velocities must be finite; masses positive and finite"))
    lambdas=Vector{Vector{Float64}}(undef, length(cs.components))
    G=zeros(3,9); rhs=zeros(3); A=zeros(3,3)
    for (cidx, comp) in enumerate(cs.components)
        length(comp.atoms)==3 && length(comp.edges)==3 || throw(ArgumentError("SETTLE accepts rigid triangles only"))
        ids=comp.atoms; fill!(G, 0.0); fill!(rhs, 0.0); fill!(A, 0.0)
        for (k,(a,b,d)) in enumerate(comp.edges)
            i,j=ids[a],ids[b]
            r_tup = _mic_vec(box, q, j, i)
            r = [r_tup[1], r_tup[2], r_tup[3]]
            G[k,3(a-1)+1:3a] .= -2r
            G[k,3(b-1)+1:3b] .= 2r
            rhs[k]=2dot(r,v[:,j]-v[:,i])
        end
        for atom in 1:3
            cols=3(atom-1)+1:3atom
            A .+= (G[:,cols]*G[:,cols]')/masses[ids[atom]]
        end
        lam=Symmetric(A)\rhs
        for atom in 1:3
            cols=3(atom-1)+1:3atom
            v[:,ids[atom]] .-= (G[:,cols]'*lam)/masses[ids[atom]]
        end
        lambdas[cidx] = lam
    end
    return lambdas
end

"Project momenta `p` in place onto the rigid tangent space at `q` with zero per-component heap allocations."
function project_momenta!(q::AbstractMatrix{Float64}, p::AbstractMatrix{Float64},
                          masses::AbstractVector{Float64}, cs::C.CoupledConstraintSet;
                          box::T.AbstractBox = T.NoBox())
    size(q)==size(p) && size(q,1)==3 && size(q,2)==length(masses) || throw(ArgumentError("shape mismatch"))
    all(isfinite,q) && all(isfinite,p) && all(x->isfinite(x)&&x>0,masses) ||
        throw(ArgumentError("coordinates and momenta must be finite; masses positive and finite"))
    max_tan = 0.0
    for comp in cs.components
        length(comp.atoms)==3 && length(comp.edges)==3 || throw(ArgumentError("SETTLE accepts rigid triangles only"))
        ids = comp.atoms
        a1, a2, a3 = ids[1], ids[2], ids[3]
        inv_m = (1.0 / masses[a1], 1.0 / masses[a2], 1.0 / masses[a3])

        e1 = comp.edges[1]; e2 = comp.edges[2]; e3 = comp.edges[3]
        r1 = _mic_vec(box, q, ids[e1[2]], ids[e1[1]])
        r2 = _mic_vec(box, q, ids[e2[2]], ids[e2[1]])
        r3 = _mic_vec(box, q, ids[e3[2]], ids[e3[1]])
        rs = (r1, r2, r3)
        es = (e1, e2, e3)

        # Build 3x3 symmetric Gram matrix A_kl = sum_atom (g_k,atom . g_l,atom) / m_atom
        # and rhs_k = 2 * r_k . (v_b - v_a)
        rhs1 = 0.0; rhs2 = 0.0; rhs3 = 0.0
        @inbounds for k in 1:3
            ea, eb, _ = es[k]
            ia, ib = ids[ea], ids[eb]
            ima, imb = inv_m[ea], inv_m[eb]
            dv = (p[1, ib]*imb - p[1, ia]*ima, p[2, ib]*imb - p[2, ia]*ima, p[3, ib]*imb - p[3, ia]*ima)
            val = 2.0 * _dot3(rs[k], dv)
            if k == 1; rhs1 = val; elseif k == 2; rhs2 = val; else; rhs3 = val; end
        end

        # Helper to compute entry A_{k,l}
        @inline function _A_entry(k::Int, l::Int)
            ea_k, eb_k, _ = es[k]
            ea_l, eb_l, _ = es[l]
            rk = rs[k]; rl = rs[l]
            dot_kl = 4.0 * _dot3(rk, rl)
            acc = 0.0
            for atom in 1:3
                sk = (atom == eb_k ? 1.0 : (atom == ea_k ? -1.0 : 0.0))
                sl = (atom == eb_l ? 1.0 : (atom == ea_l ? -1.0 : 0.0))
                acc += (sk * sl) * dot_kl * inv_m[atom]
            end
            return acc
        end
        a11 = _A_entry(1, 1); a12 = _A_entry(1, 2); a13 = _A_entry(1, 3)
        a22 = _A_entry(2, 2); a23 = _A_entry(2, 3); a33 = _A_entry(3, 3)

        # Solve 3x3 linear system A * lam = rhs via Cramer's rule / cofactor inverse
        c11 = a22*a33 - a23*a23
        c12 = a13*a23 - a12*a33
        c13 = a12*a23 - a13*a22
        c22 = a11*a33 - a13*a13
        c23 = a12*a13 - a11*a23
        c33 = a11*a22 - a12*a12
        detA = a11*c11 + a12*c12 + a13*c13
        abs(detA) > 1e-30 || throw(ArgumentError("SETTLE tangent projection singular"))
        inv_det = 1.0 / detA
        lam1 = (c11*rhs1 + c12*rhs2 + c13*rhs3) * inv_det
        lam2 = (c12*rhs1 + c22*rhs2 + c23*rhs3) * inv_det
        lam3 = (c13*rhs1 + c23*rhs2 + c33*rhs3) * inv_det
        lams = (lam1, lam2, lam3)

        # Apply impulse p[:, ids[atom]] -= G[:, cols]' * lam
        @inbounds for k in 1:3
            ea, eb, _ = es[k]
            ia, ib = ids[ea], ids[eb]
            rk = rs[k]
            lk = 2.0 * lams[k]
            p[1, ia] += lk * rk[1]; p[2, ia] += lk * rk[2]; p[3, ia] += lk * rk[3]
            p[1, ib] -= lk * rk[1]; p[2, ib] -= lk * rk[2]; p[3, ib] -= lk * rk[3]
        end

        @inbounds for k in 1:3
            ea, eb, _ = es[k]
            ia, ib = ids[ea], ids[eb]
            ima, imb = inv_m[ea], inv_m[eb]
            dv = (p[1, ib]*imb - p[1, ia]*ima, p[2, ib]*imb - p[2, ia]*ima, p[3, ib]*imb - p[3, ia]*ima)
            max_tan = max(max_tan, abs(_dot3(rs[k], dv)))
        end
    end
    return max_tan
end

"""
    settle_step!(q_ref, q_curr, p, masses, cs, dt; box=Types.NoBox())

Complete symplectic SETTLE position + momentum-impulse update for a step from `q_ref`
to unconstrained predicted coordinates `q_curr` (modified in place to constrained positions),
updating `p` with the constraint coordinate impulse `M * (q_constrained - q_predicted) / dt`.
Does not project end-of-step velocities (call `project_momenta!` after the second half-kick).
"""
function settle_step!(q_ref::AbstractMatrix{Float64}, q_curr::AbstractMatrix{Float64},
                      p::AbstractMatrix{Float64}, masses::AbstractVector{Float64},
                      cs::C.CoupledConstraintSet, dt::Real;
                      box::T.AbstractBox = T.NoBox())
    dt > 0 && isfinite(dt) || throw(ArgumentError("dt must be positive and finite"))
    _check(q_ref, q_curr, masses, cs)
    size(p) == size(q_curr) || throw(ArgumentError("momenta must match coordinate dimensions"))
    inv_dt = 1.0 / Float64(dt)
    # Save unconstrained predicted positions per component and apply momentum impulse in place
    for comp in cs.components
        ids = comp.atoms
        i1, i2, i3 = ids[1], ids[2], ids[3]
        @inbounds begin
            u1x, u1y, u1z = q_curr[1, i1], q_curr[2, i1], q_curr[3, i1]
            u2x, u2y, u2z = q_curr[1, i2], q_curr[2, i2], q_curr[3, i2]
            u3x, u3y, u3z = q_curr[1, i3], q_curr[2, i3], q_curr[3, i3]
        end
        single_cs = C.CoupledConstraintSet([comp], cs.dim)
        settle_positions!(q_ref, q_curr, masses, single_cs; box=box)
        @inbounds begin
            m1, m2, m3 = masses[i1] * inv_dt, masses[i2] * inv_dt, masses[i3] * inv_dt
            p[1, i1] += m1 * (q_curr[1, i1] - u1x); p[2, i1] += m1 * (q_curr[2, i1] - u1y); p[3, i1] += m1 * (q_curr[3, i1] - u1z)
            p[1, i2] += m2 * (q_curr[1, i2] - u2x); p[2, i2] += m2 * (q_curr[2, i2] - u2y); p[3, i2] += m2 * (q_curr[3, i2] - u2z)
            p[1, i3] += m3 * (q_curr[1, i3] - u3x); p[2, i3] += m3 * (q_curr[2, i3] - u3y); p[3, i3] += m3 * (q_curr[3, i3] - u3z)
        end
    end
    return nothing
end

"Position correction and independent tangent projection; not a complete integrator. Returns residual and tangent multipliers."
function settle!(q::AbstractMatrix{Float64}, qpred::AbstractMatrix{Float64}, v::AbstractMatrix{Float64},
                 masses::AbstractVector{Float64}, cs::C.CoupledConstraintSet;
                 box::T.AbstractBox = T.NoBox())
    r=settle_positions!(q,qpred,masses,cs; box=box)
    λ=project_velocities!(qpred,v,masses,cs; box=box)
    return (position_residual=r, velocity_multipliers=λ)
end

end
