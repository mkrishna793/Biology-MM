"""
Matrix-free induced point-dipole solve for a deliberately narrow model.

The implemented model is non-periodic point charges and isotropic polarizabilities,
with no damping. Permanent fields and dipole coupling use Coulomb prefactor `k_e`.
The polarizabilities must use units consistent with that prefactor, coordinates,
and charges, so `1/alpha` and `k_e*q/r^2` have compatible operator/field units.
For displacement `d = r_i-r_j`, `E_i += k_e q_j d/r^3`,
`T_ij = k_e(3dd'/r^5-I/r^3)`, and `A_ij=-T_ij`; the diagonal is `I/alpha_i`.
Zero-alpha sites are removed from the unknown vector but remain charge sources.
The direct operator has O(N) storage and O(N^2) work per matvec. It is not PME,
not an accelerated polarization backend, and supplies no nuclear forces; callers
must not use it as a Dynamics potential.

PCG reports true residual and actual iterations. Small systems get an eigenvalue
SPD certificate; larger systems are marked `:not_certified`, since positive
curvature along the Krylov search directions alone is not a global SPD proof.
No energy or usable dipole vector is returned on nonconvergence/negative curvature.
"""
module PolarizationOperator

using LinearAlgebra
using Random
using ..Types: AbstractBox, NoBox, OrthorhombicBox, TriclinicBox, unwrap_displacement

export DipoleOperator, DipoleSolveResult, build_operator, solve_dipoles!,
       induced_energy, induced_forces, analytic_induced_forces, require_converged, dense_reference,
       apply_dipole_operator!, solve_induced_dipoles,
       TernaryDipolePreconditioner, ternary_compression_ratio, build_ternary_preconditioner,
       ternary_bitplane_matvec!, apply_ternary_neumann_precond!, solve_induced_dipoles_ternary_pcg,
       gray_code_bitplane_sweep, QuasiCyclicBlockPreconditioner, build_qc_block_preconditioner,
       apply_qc_ternary_precond!, certify_dipole_spd_luttinger_tisza,
       cw_orthogonal_noise, apply_cw_block_filter!

struct DipoleOperator
    q::Matrix{Float64}
    charges::Vector{Float64}
    alpha::Vector{Float64}
    active::Vector{Int}
    inverse_alpha::Vector{Float64}
    box::AbstractBox
    prefactor::Float64
    rcut::Float64
    field::Vector{Float64}
    thole::Bool
    a_thole::Float64
    periodic::Bool
end

DipoleOperator(q, charges, alpha, active, inverse_alpha, box, prefactor, rcut, field) =
    DipoleOperator(q, charges, alpha, active, inverse_alpha, box, prefactor, rcut, field, false, 1.0, false)

DipoleOperator(q, charges, alpha, active, inverse_alpha, box, prefactor, rcut, field, thole, a_thole) =
    DipoleOperator(q, charges, alpha, active, inverse_alpha, box, prefactor, rcut, field, thole, a_thole, false)

struct DipoleSolveResult
    mu::Union{Nothing,Vector{Float64}}
    residual::Float64
    iterations::Int
    status::Symbol
    spd::Symbol
    min_eigenvalue::Float64
end

function _displacement(box::NoBox, q, i, j)
    return q[1,i]-q[1,j], q[2,i]-q[2,j], q[3,i]-q[3,j]
end
function _displacement(box::OrthorhombicBox, q, i, j)
    L=box.L
    dx=q[1,i]-q[1,j]; dy=q[2,i]-q[2,j]; dz=q[3,i]-q[3,j]
    return dx-L[1]*round(dx/L[1]), dy-L[2]*round(dy/L[2]), dz-L[3]*round(dz/L[3])
end
function _displacement(box::TriclinicBox, q, i, j)
    d = unwrap_displacement(box, q, i, j)
    return d.x, d.y, d.z
end

"""Build the non-periodic or periodic point-dipole operator.

`alpha` units must be consistent with `coulomb_prefactor`, the coordinate units,
and charge units; in particular the diagonal `1/alpha` must have the same units
as the dipole-coupling tensor formed with `coulomb_prefactor`.
"""
function build_operator(q::AbstractMatrix{<:Real}, charges::AbstractVector{<:Real},
                        alpha::AbstractVector{<:Real}, box::AbstractBox=NoBox();
                        coulomb_prefactor::Real=1.0,
                        thole::Bool=false, a_thole::Real=1.0,
                        periodic::Bool=false)
    size(q,1)==3 || throw(ArgumentError("q must be 3xN"))
    n=size(q,2)
    length(charges)==n && length(alpha)==n || throw(ArgumentError("q/charge/alpha size mismatch"))
    all(isfinite,q) || throw(ArgumentError("coordinates must be finite"))
    all(isfinite,charges) || throw(ArgumentError("charges must be finite"))
    all(x->isfinite(x)&&x>=0,alpha) || throw(ArgumentError("alpha must be finite and >= 0"))
    isfinite(coulomb_prefactor) && coulomb_prefactor>0 || throw(ArgumentError("coulomb_prefactor must be finite and > 0"))
    if !periodic
        box isa NoBox || throw(ArgumentError("this operator currently supports NoBox only; periodic electrostatics requires Ewald/PME"))
    else
        (box isa NoBox || box isa OrthorhombicBox || box isa TriclinicBox) ||
            throw(ArgumentError("unsupported periodic box type $(typeof(box))"))
    end
    ath = Float64(a_thole)
    ath > 0 || throw(ArgumentError("Thole damping radius must be positive"))
    qa=Matrix{Float64}(q); za=Vector{Float64}(charges); aa=Vector{Float64}(alpha)
    all(isfinite,qa) && all(isfinite,za) && all(isfinite,aa) || throw(ArgumentError("coordinates, charges, and alpha must be representable as Float64"))
    all(k->alpha[k]==0 || aa[k]>0,eachindex(aa)) || throw(ArgumentError("positive alpha must be representable as a positive Float64"))
    active=findall(>(0.0),aa)
    inva=1.0 ./ aa[active]
    all(isfinite,inva) || throw(ArgumentError("inverse alpha must be finite"))
    m=length(active); E=zeros(3m); pre=Float64(coulomb_prefactor)
    # Validate all pair geometries that enter the permanent field or dipole tensor.
    for i in 1:n-1, j in i+1:n
        dx,dy,dz=_displacement(box,qa,i,j)
        r2=dx*dx+dy*dy+dz*dz
        isfinite(r2) && r2>1e-24 || throw(ArgumentError("invalid or singular pair geometry for sites $i and $j"))
    end
    @inbounds for (ai,i) in enumerate(active), j in 1:n
        i==j && continue
        dx,dy,dz=_displacement(box,qa,i,j)
        r2=dx*dx+dy*dy+dz*dz; c=pre*za[j]/(r2*sqrt(r2)); b=3(ai-1)
        E[b+1]+=c*dx; E[b+2]+=c*dy; E[b+3]+=c*dz
    end
    all(isfinite,E) || throw(ArgumentError("permanent electric field must be finite"))
    return DipoleOperator(qa,za,aa,active,inva,box,pre,Inf,E,thole,ath,periodic)
end

Base.size(A::DipoleOperator)=(3length(A.active),3length(A.active))
Base.size(A::DipoleOperator,d::Integer)=size(A)[d]

function LinearAlgebra.mul!(y::AbstractVector{Float64}, A::DipoleOperator, x::AbstractVector{Float64})
    m=length(A.active); n=3m
    length(x)==n && length(y)==n || throw(DimensionMismatch("operator dimension is $n"))
    @inbounds for a in 1:m, d in 1:3
        y[3(a-1)+d]=A.inverse_alpha[a]*x[3(a-1)+d]
    end
    @inbounds for a in 1:m-1, b in a+1:m
        i=A.active[a]; j=A.active[b]
        dx,dy,dz=_displacement(A.box,A.q,i,j)
        r2=dx*dx+dy*dy+dz*dz; r=sqrt(r2); r3=r2*r; r5=r3*r2
        g = 1.0
        if A.thole
            aij = (max(A.alpha[i], 1e-30) * max(A.alpha[j], 1e-30))^(1 / 6)
            u = r / (A.a_thole > 0 ? max(A.a_thole, aij) : aij)
            g = 1.0 - exp(-2.1304 * u^3)
        end
        # A_ab=-T_ab = I/r^3 - 3 dd'/r^5; symmetric opposite block.
        vj=(x[3(b-1)+1],x[3(b-1)+2],x[3(b-1)+3])
        vi=(x[3(a-1)+1],x[3(a-1)+2],x[3(a-1)+3])
        dj=dx*vj[1]+dy*vj[2]+dz*vj[3]
        di=dx*vi[1]+dy*vi[2]+dz*vi[3]
        c=g*A.prefactor/r3; t=3*g*A.prefactor/r5
        y[3(a-1)+1]+=c*vj[1]-t*dx*dj; y[3(a-1)+2]+=c*vj[2]-t*dy*dj; y[3(a-1)+3]+=c*vj[3]-t*dz*dj
        y[3(b-1)+1]+=c*vi[1]-t*dx*di; y[3(b-1)+2]+=c*vi[2]-t*dy*di; y[3(b-1)+3]+=c*vi[3]-t*dz*di
    end
    return y
end

function dense_reference(A::DipoleOperator)
    n=size(A,1); M=zeros(n,n)
    for j in 1:n
        x=zeros(n); x[j]=1; y=zeros(n); mul!(y,A,x); M[:,j]=y
    end
    return M
end

"""Solve `A*mu = field` by guarded preconditioned conjugate gradients.

`tol` is a finite, positive relative tolerance representable as `Float64`.
"""
function solve_dipoles!(A::DipoleOperator; tol::Real=1e-10, maxiter::Integer=max(1,4size(A,1)),
                        certify_sites::Integer=64, lt_certify::Bool=true)
    tol64=Float64(tol)
    isfinite(tol64)&&tol64>0 || throw(ArgumentError("tol must be finite, positive, and representable as Float64"))
    maxiter>=0 || throw(ArgumentError("maxiter must be nonnegative"))
    n=size(A,1); n==0 && return DipoleSolveResult(Float64[],0.0,0,:converged,:certified,Inf)
    E=A.field
    all(isfinite,E) || throw(ArgumentError("right-hand side field must be finite"))
    normE=norm(E); threshold=tol64*normE
    isfinite(threshold) || throw(ArgumentError("tol and right-hand side produce a non-finite convergence threshold"))
    M=n<=3certify_sites ? dense_reference(A) : nothing
    mineig=M===nothing ? NaN : minimum(eigvals(Symmetric(M)))
    if M!==nothing && !(mineig>0)
        return DipoleSolveResult(nothing,Inf,0,:indefinite,:indefinite,mineig)
    end
    if M===nothing
        if lt_certify
            lt_spd, lt_bound = certify_dipole_spd_luttinger_tisza(A)
            spd = lt_spd
            mineig = lt_spd === :certified ? lt_bound : NaN
        else
            spd = :not_certified
        end
    else
        spd = :certified
    end
    x=zeros(n); r=copy(E); z=similar(r)
    # Jacobi preconditioner is exact for the self blocks.
    @inbounds for i in 1:n; z[i]=r[i]/A.inverse_alpha[(i-1)÷3+1]; end
    p=copy(z); rz=dot(r,z)
    if norm(r)<=threshold || normE==0.0
        return DipoleSolveResult(x,norm(r)/max(normE,eps()),0,:converged,spd,mineig)
    end
    y=zeros(n)
    for it in 1:Int(maxiter)
        mul!(y,A,p); curvature=dot(p,y)
        if !(isfinite(curvature)&&curvature>0)
            return DipoleSolveResult(nothing,Inf,it-1,:indefinite,:indefinite,mineig)
        end
        step=rz/curvature; @. x=x+step*p; @. r=r-step*y
        if norm(r)<=threshold
            mul!(y,A,x); rel=norm(y-E)/max(normE,eps())
            if rel<=tol64
                return DipoleSolveResult(x,rel,it,:converged,spd,mineig)
            end
        end
        @inbounds for i in 1:n; z[i]=r[i]/A.inverse_alpha[(i-1)÷3+1]; end
        rznew=dot(r,z); beta=rznew/rz; @. p=z+beta*p; rz=rznew
    end
    mul!(y,A,x); rel=norm(y-E)/max(normE,eps())
    return DipoleSolveResult(nothing,rel,Int(maxiter),:not_converged,spd,mineig)
end

function require_converged(r::DipoleSolveResult)
    r.status===:converged || throw(ErrorException("induced dipole solve unusable: status=$(r.status), residual=$(r.residual), iterations=$(r.iterations)"))
    return r.mu::Vector{Float64}
end

function induced_energy(A::DipoleOperator, r::DipoleSolveResult)
    mu=require_converged(r); y=zeros(length(mu)); mul!(y,A,mu)
    return 0.5dot(mu,y)-dot(mu,A.field)
end

"""
    analytic_induced_forces(A::DipoleOperator, r::DipoleSolveResult) -> Matrix{Float64}

Nuclear forces `-dU/dq` holding `μ` fixed at the stationary point:
    U = ½ μᵀ A(q) μ − μᵀ E(q)

Evaluated analytically via the Hellmann-Feynman / envelope gradient in O(N²) time.
"""
function analytic_induced_forces(A::DipoleOperator, r::DipoleSolveResult)
    mu = require_converged(r)
    n_sites = size(A.q, 2)
    F = zeros(Float64, 3, n_sites)
    pre = A.prefactor

    # Expand active mu into full 3 x n_sites array
    M = zeros(Float64, 3, n_sites)
    for (a, i) in enumerate(A.active)
        M[1, i] = mu[3(a-1)+1]
        M[2, i] = mu[3(a-1)+2]
        M[3, i] = mu[3(a-1)+3]
    end

    for i in 1:n_sites-1
        mi = (M[1, i], M[2, i], M[3, i])
        is_i_active = A.alpha[i] > 0
        for j in i+1:n_sites
            mj = (M[1, j], M[2, j], M[3, j])
            is_j_active = A.alpha[j] > 0

            # If neither site is active, no induced dipoles at i or j, so no force on pair
            (!is_i_active && !is_j_active) && continue

            dx, dy, dz = _displacement(A.box, A.q, i, j)
            r2 = dx*dx + dy*dy + dz*dz
            r2 < 1e-24 && continue
            r_dist = sqrt(r2)

            # 1. Field term contribution
            # p = z_j * μ_i - z_i * μ_j
            px = A.charges[j] * mi[1] - A.charges[i] * mj[1]
            py = A.charges[j] * mi[2] - A.charges[i] * mj[2]
            pz = A.charges[j] * mi[3] - A.charges[i] * mj[3]
            p_dot_r = px * dx + py * dy + pz * dz

            c_field = pre / (r2 * r_dist)
            t_field = 3.0 * pre / (r2 * r2 * r_dist)

            fx_field = c_field * px - t_field * p_dot_r * dx
            fy_field = c_field * py - t_field * p_dot_r * dy
            fz_field = c_field * pz - t_field * p_dot_r * dz

            # 2. Dipole-dipole term contribution (only non-zero if both sites active)
            fx_dip = 0.0; fy_dip = 0.0; fz_dip = 0.0
            if is_i_active && is_j_active
                g = 1.0
                gprime = 0.0
                if A.thole
                    aij = (max(A.alpha[i], 1e-30) * max(A.alpha[j], 1e-30))^(1 / 6)
                    a_eff = A.a_thole > 0 ? max(A.a_thole, aij) : aij
                    u = r_dist / a_eff
                    s_val = 2.1304 * u^3
                    g = 1.0 - exp(-s_val)
                    gprime = (3.0 * s_val / r_dist) * exp(-s_val)
                end

                d1 = mi[1] * dx + mi[2] * dy + mi[3] * dz
                d2 = mj[1] * dx + mj[2] * dy + mj[3] * dz
                d12 = mi[1] * mj[1] + mi[2] * mj[2] + mi[3] * mj[3]

                C1 = -3.0 * pre * g / (r2 * r2 * r_dist)
                C1prime = 3.0 * pre * (5.0 * g - r_dist * gprime) / (r2 * r2 * r2)
                C2prime = -pre * (3.0 * g - r_dist * gprime) / (r2 * r2)

                scalar_factor = (C1prime * d1 * d2 + C2prime * d12) / r_dist

                fx_dip = -(scalar_factor * dx + C1 * (d2 * mi[1] + d1 * mj[1]))
                fy_dip = -(scalar_factor * dy + C1 * (d2 * mi[2] + d1 * mj[2]))
                fz_dip = -(scalar_factor * dz + C1 * (d2 * mi[3] + d1 * mj[3]))
            end

            f_tot_x = fx_field + fx_dip
            f_tot_y = fy_field + fy_dip
            f_tot_z = fz_field + fz_dip

            F[1, i] += f_tot_x; F[2, i] += f_tot_y; F[3, i] += f_tot_z
            F[1, j] -= f_tot_x; F[2, j] -= f_tot_y; F[3, j] -= f_tot_z
        end
    end
    return F
end

"""
    induced_forces(A::DipoleOperator, r::DipoleSolveResult; h::Float64=1e-6, analytic::Bool=false) -> Matrix{Float64}

Nuclear forces `-dU/dq` holding `μ` fixed at the stationary point:
    U = ½ μᵀ A(q) μ − μᵀ E(q)

When `analytic = true`, computes exact forces via `analytic_induced_forces`.
Otherwise evaluates via central finite differences of the quadratic form at fixed `μ`.
"""
function induced_forces(A::DipoleOperator, r::DipoleSolveResult; h::Float64=1e-6, analytic::Bool=false)
    if analytic
        return analytic_induced_forces(A, r)
    end
    mu = require_converged(r)
    n_sites = size(A.q, 2)
    F = zeros(Float64, 3, n_sites)
    for j in 1:n_sites, d in 1:3
        qp = copy(A.q); qp[d, j] += h
        qm = copy(A.q); qm[d, j] -= h
        Ap = build_operator(qp, A.charges, A.alpha, A.box;
                            coulomb_prefactor=A.prefactor,
                            thole=A.thole, a_thole=A.a_thole,
                            periodic=A.periodic)
        Am = build_operator(qm, A.charges, A.alpha, A.box;
                            coulomb_prefactor=A.prefactor,
                            thole=A.thole, a_thole=A.a_thole,
                            periodic=A.periodic)
        yp = zeros(length(mu)); mul!(yp, Ap, mu)
        ym = zeros(length(mu)); mul!(ym, Am, mu)
        Up = 0.5 * dot(mu, yp) - dot(mu, Ap.field)
        Um = 0.5 * dot(mu, ym) - dot(mu, Am.field)
        F[d, j] = -(Up - Um) / (2.0 * h)
    end
    return F
end

# Iteration / destructuring interface for DipoleSolveResult:
# enables `(mu, stats) = result` syntax while retaining `result.mu`, `result.iterations`, etc.
Base.iterate(r::DipoleSolveResult) = (r.mu, 1)
Base.iterate(r::DipoleSolveResult, state) = state == 1 ? (r, 2) : nothing
Base.length(r::DipoleSolveResult) = 2
Base.indexed_iterate(r::DipoleSolveResult, i::Int, state=1) = i == 1 ? (r.mu, 2) : (r, 3)

"""
    apply_dipole_operator!(y::AbstractVector{Float64}, A::DipoleOperator, x::AbstractVector{Float64})

Matrix-free application of the dipole operator `y = A * x`.
"""
apply_dipole_operator!(y::AbstractVector{Float64}, A::DipoleOperator, x::AbstractVector{Float64}) = mul!(y, A, x)

"""
    TernaryDipolePreconditioner

Dual-rail ternary MMSE preconditioner for polarizable dipole systems.
Represents the off-diagonal dipole coupling matrix `T = A - diag(A)` using `num_planes`
dual-rail bitplanes (`W_P`, `W_N`) and greedy MMSE row scales.
Includes transposed stride-1 bitplane buffers (`W_P_t`, `W_N_t`) and preallocated
quantization scratch buffers (`x_P`, `x_N`, `x_scales`, `x_rem`) for zero-allocation matvecs.
"""
mutable struct TernaryDipolePreconditioner
    n_atoms::Int
    num_planes::Int
    n_words::Int
    inv_diag::Vector{Float64}
    W_P::Array{UInt64, 3}
    W_N::Array{UInt64, 3}
    row_scales::Matrix{Float64}
    W_P_t::Array{UInt64, 3}
    W_N_t::Array{UInt64, 3}
    x_P::Matrix{UInt64}
    x_N::Matrix{UInt64}
    x_scales::Vector{Float64}
    x_rem::Vector{Float64}
end

function TernaryDipolePreconditioner(n_atoms::Int, num_planes::Int, n_words::Int,
                                      inv_diag::Vector{Float64},
                                      W_P::Array{UInt64, 3}, W_N::Array{UInt64, 3},
                                      row_scales::Matrix{Float64})
    M = 3 * n_atoms
    W_P_t = Array{UInt64, 3}(undef, n_words, M, num_planes)
    W_N_t = Array{UInt64, 3}(undef, n_words, M, num_planes)
    @inbounds for p in 1:num_planes, w in 1:n_words, i in 1:M
        W_P_t[w, i, p] = W_P[i, w, p]
        W_N_t[w, i, p] = W_N[i, w, p]
    end
    kx_cap = max(8, num_planes)
    x_P = zeros(UInt64, n_words, kx_cap)
    x_N = zeros(UInt64, n_words, kx_cap)
    x_scales = zeros(Float64, kx_cap)
    x_rem = zeros(Float64, M)
    return TernaryDipolePreconditioner(n_atoms, num_planes, n_words, inv_diag, W_P, W_N, row_scales,
                                        W_P_t, W_N_t, x_P, x_N, x_scales, x_rem)
end

"""
    ternary_compression_ratio(prec::TernaryDipolePreconditioner) -> Float64

Ratio of dense Float64 matrix bytes (`M * M * 8`) to compressed bitplane representation
`sizeof(W_P) + sizeof(W_N) + sizeof(row_scales) + sizeof(inv_diag)`.
"""
function ternary_compression_ratio(prec::TernaryDipolePreconditioner)::Float64
    M = 3 * prec.n_atoms
    dense_bytes = Float64(M * M * sizeof(Float64))
    comp_bytes = Float64(sizeof(prec.W_P) + sizeof(prec.W_N) + sizeof(prec.row_scales) + sizeof(prec.inv_diag))
    comp_bytes == 0.0 && return Inf
    return dense_bytes / comp_bytes
end

"""
    build_ternary_preconditioner(op::DipoleOperator; num_planes::Integer=3) -> TernaryDipolePreconditioner

Constructs a dual-rail ternary MMSE preconditioner for `op` in O(M^2) time without
materializing or storing a full dense Float64 matrix. Assembles the 3x3 dipole coupling
blocks analytically per active site pair `(a, b)` and decomposes each row across
`p = 1:num_planes` using greedy residual MMSE ternary quantization.
"""
function build_ternary_preconditioner(op::DipoleOperator; num_planes::Integer=3)::TernaryDipolePreconditioner
    return _build_ternary_preconditioner(op; num_planes=num_planes, exclude_block_sites=0)
end

function _build_ternary_preconditioner(op::DipoleOperator; num_planes::Integer=3,
                                       exclude_block_sites::Integer=0)::TernaryDipolePreconditioner
    m = length(op.active)
    M = 3m
    num_p = max(1, Int(num_planes))
    n_words = cld(M, 64)
    eblock = Int(exclude_block_sites)

    inv_diag = Vector{Float64}(undef, M)
    alpha_16 = Vector{Float64}(undef, m)
    @inbounds for a in 1:m
        id = 1.0 / op.inverse_alpha[a]
        base = 3 * (a - 1)
        inv_diag[base + 1] = id
        inv_diag[base + 2] = id
        inv_diag[base + 3] = id
        alpha_16[a] = op.thole ? max(op.alpha[op.active[a]], 1e-30)^(1 / 6) : 0.0
    end

    W_P = zeros(UInt64, M, n_words, num_p)
    W_N = zeros(UInt64, M, n_words, num_p)
    W_P_t = zeros(UInt64, n_words, M, num_p)
    W_N_t = zeros(UInt64, n_words, M, num_p)
    row_scales = zeros(Float64, M, num_p)
    kx_cap = max(8, num_p)
    x_P = zeros(UInt64, n_words, kx_cap)
    x_N = zeros(UInt64, n_words, kx_cap)
    x_scales = zeros(Float64, kx_cap)
    x_rem = zeros(Float64, M)
    M == 0 && return TernaryDipolePreconditioner(m, num_p, n_words, inv_diag, W_P, W_N, row_scales,
                                                 W_P_t, W_N_t, x_P, x_N, x_scales, x_rem)

    rows_buf = zeros(Float64, M, 3)
    r_rem = Vector{Float64}(undef, M)
    invM = 0.7 / M

    @inbounds for a in 1:m
        isite = op.active[a]
        ai_base = 3 * (a - 1)
        block_a = eblock > 0 ? (a - 1) ÷ eblock : -1
        a16 = alpha_16[a]

        rows_buf[ai_base + 1, 1] = 0.0; rows_buf[ai_base + 2, 1] = 0.0; rows_buf[ai_base + 3, 1] = 0.0
        rows_buf[ai_base + 1, 2] = 0.0; rows_buf[ai_base + 2, 2] = 0.0; rows_buf[ai_base + 3, 2] = 0.0
        rows_buf[ai_base + 1, 3] = 0.0; rows_buf[ai_base + 2, 3] = 0.0; rows_buf[ai_base + 3, 3] = 0.0

        for b in 1:m
            b == a && continue
            jb = 3 * (b - 1)
            if eblock > 0 && ((b - 1) ÷ eblock == block_a)
                rows_buf[jb + 1, 1] = 0.0; rows_buf[jb + 2, 1] = 0.0; rows_buf[jb + 3, 1] = 0.0
                rows_buf[jb + 1, 2] = 0.0; rows_buf[jb + 2, 2] = 0.0; rows_buf[jb + 3, 2] = 0.0
                rows_buf[jb + 1, 3] = 0.0; rows_buf[jb + 2, 3] = 0.0; rows_buf[jb + 3, 3] = 0.0
                continue
            end
            jsite = op.active[b]
            dx, dy, dz = _displacement(op.box, op.q, isite, jsite)
            r2 = dx * dx + dy * dy + dz * dz
            if r2 < 1e-24
                rows_buf[jb + 1, 1] = 0.0; rows_buf[jb + 2, 1] = 0.0; rows_buf[jb + 3, 1] = 0.0
                rows_buf[jb + 1, 2] = 0.0; rows_buf[jb + 2, 2] = 0.0; rows_buf[jb + 3, 2] = 0.0
                rows_buf[jb + 1, 3] = 0.0; rows_buf[jb + 2, 3] = 0.0; rows_buf[jb + 3, 3] = 0.0
                continue
            end
            r = sqrt(r2); r3 = r2 * r; r5 = r3 * r2
            g = 1.0
            if op.thole
                aij = a16 * alpha_16[b]
                u = r / (op.a_thole > 0 ? max(op.a_thole, aij) : aij)
                g = 1.0 - exp(-2.1304 * u * u * u)
            end
            c = g * op.prefactor / r3
            t = 3.0 * g * op.prefactor / r5
            tdx = t * dx; tdy = t * dy; tdz = t * dz
            rows_buf[jb + 1, 1] = c - tdx * dx
            rows_buf[jb + 2, 1] = -tdx * dy
            rows_buf[jb + 3, 1] = -tdx * dz
            rows_buf[jb + 1, 2] = -tdy * dx
            rows_buf[jb + 2, 2] = c - tdy * dy
            rows_buf[jb + 3, 2] = -tdy * dz
            rows_buf[jb + 1, 3] = -tdz * dx
            rows_buf[jb + 2, 3] = -tdz * dy
            rows_buf[jb + 3, 3] = c - tdz * dz
        end

        for d in 1:3
            i = ai_base + d
            copyto!(r_rem, 1, rows_buf, (d - 1) * M + 1, M)
            for p in 1:num_p
                sum_abs = 0.0
                @simd for j in 1:M
                    sum_abs += abs(r_rem[j])
                end
                tau = invM * sum_abs
                (tau <= 0.0 || !isfinite(tau)) && break

                cnt = 0
                num = 0.0
                for w in 1:n_words
                    j0 = (w - 1) << 6
                    jmax = min(64, M - j0)
                    wp = UInt64(0)
                    wn = UInt64(0)
                    for bit in 0:(jmax - 1)
                        rj = r_rem[j0 + bit + 1]
                        if abs(rj) > tau
                            mask = UInt64(1) << bit
                            if rj > 0.0
                                wp |= mask
                                num += rj
                            else
                                wn |= mask
                                num -= rj
                            end
                            cnt += 1
                        end
                    end
                    W_P[i, w, p] = wp
                    W_N[i, w, p] = wn
                    W_P_t[w, i, p] = wp
                    W_N_t[w, i, p] = wn
                end
                cnt == 0 && break

                alpha_scale = num / cnt
                row_scales[i, p] = alpha_scale
                for w in 1:n_words
                    j0 = (w - 1) << 6
                    jmax = min(64, M - j0)
                    wp = W_P_t[w, i, p]
                    wn = W_N_t[w, i, p]
                    for bit in 0:(jmax - 1)
                        mask = UInt64(1) << bit
                        pos = (wp & mask) != 0
                        neg = (wn & mask) != 0
                        r_rem[j0 + bit + 1] -= ifelse(pos, alpha_scale, ifelse(neg, -alpha_scale, 0.0))
                    end
                end
            end
        end
    end

    return TernaryDipolePreconditioner(m, num_p, n_words, inv_diag, W_P, W_N, row_scales,
                                        W_P_t, W_N_t, x_P, x_N, x_scales, x_rem)
end

"""
    ternary_bitplane_matvec!(y::AbstractVector{Float64}, prec::TernaryDipolePreconditioner, x::AbstractVector{Float64}; Kx::Integer=3)

Quantizes input vector `x` into `Kx` dual-rail ternary bitplanes using preallocated scratch
buffers and computes `y = T_ternary * x` via a 4-word unrolled bitwise POPCNT kernel with
zero heap allocations.
"""
function ternary_bitplane_matvec!(y::AbstractVector{Float64}, prec::TernaryDipolePreconditioner,
                                  x::AbstractVector{Float64}; Kx::Integer=3)
    M = 3 * prec.n_atoms
    length(x) == M && length(y) == M || throw(DimensionMismatch("dimension mismatch in ternary_bitplane_matvec!"))
    @inbounds for i in 1:M
        y[i] = 0.0
    end
    M == 0 && return y
    n_words = prec.n_words
    kx = max(1, Int(Kx))

    if kx > size(prec.x_P, 2)
        prec.x_P = zeros(UInt64, n_words, kx)
        prec.x_N = zeros(UInt64, n_words, kx)
        prec.x_scales = zeros(Float64, kx)
    end
    if length(prec.x_rem) != M
        prec.x_rem = Vector{Float64}(undef, M)
    end

    x_P = prec.x_P
    x_N = prec.x_N
    x_scales = prec.x_scales
    x_rem = prec.x_rem
    @inbounds for k in 1:kx
        x_scales[k] = 0.0
        for w in 1:n_words
            x_P[w, k] = UInt64(0)
            x_N[w, k] = UInt64(0)
        end
    end
    @inbounds for j in 1:M
        x_rem[j] = x[j]
    end

    invM = 0.7 / M
    @inbounds for k in 1:kx
        sum_abs = 0.0
        @simd for j in 1:M
            sum_abs += abs(x_rem[j])
        end
        tau = invM * sum_abs
        (tau <= 0.0 || !isfinite(tau)) && break

        cnt = 0
        num = 0.0
        for w in 1:n_words
            j0 = (w - 1) << 6
            jmax = min(64, M - j0)
            xp_w = UInt64(0)
            xn_w = UInt64(0)
            for bit in 0:(jmax - 1)
                xj = x_rem[j0 + bit + 1]
                if abs(xj) > tau
                    mask = UInt64(1) << bit
                    if xj > 0.0
                        xp_w |= mask
                        num += xj
                    else
                        xn_w |= mask
                        num -= xj
                    end
                    cnt += 1
                end
            end
            x_P[w, k] = xp_w
            x_N[w, k] = xn_w
        end
        cnt == 0 && break

        scale = num / cnt
        x_scales[k] = scale
        for w in 1:n_words
            j0 = (w - 1) << 6
            jmax = min(64, M - j0)
            xp_w = x_P[w, k]
            xn_w = x_N[w, k]
            for bit in 0:(jmax - 1)
                mask = UInt64(1) << bit
                pos = (xp_w & mask) != 0
                neg = (xn_w & mask) != 0
                x_rem[j0 + bit + 1] -= ifelse(pos, scale, ifelse(neg, -scale, 0.0))
            end
        end
    end

    W_P_t = prec.W_P_t
    W_N_t = prec.W_N_t
    row_scales = prec.row_scales
    nw4 = n_words & ~3

    @inbounds for p in 1:prec.num_planes
        for k in 1:kx
            sk = x_scales[k]
            sk == 0.0 && continue
            for i in 1:M
                sp = row_scales[i, p]
                sp == 0.0 && continue
                s = sp * sk
                acc0 = 0
                acc1 = 0
                acc2 = 0
                acc3 = 0
                w = 1
                while w <= nw4
                    wp0 = W_P_t[w, i, p];     wn0 = W_N_t[w, i, p]
                    xp0 = x_P[w, k];          xn0 = x_N[w, k]
                    acc0 += Int(count_ones((wp0 & xp0) | (wn0 & xn0))) - Int(count_ones((wp0 & xn0) | (wn0 & xp0)))

                    wp1 = W_P_t[w + 1, i, p]; wn1 = W_N_t[w + 1, i, p]
                    xp1 = x_P[w + 1, k];      xn1 = x_N[w + 1, k]
                    acc1 += Int(count_ones((wp1 & xp1) | (wn1 & xn1))) - Int(count_ones((wp1 & xn1) | (wn1 & xp1)))

                    wp2 = W_P_t[w + 2, i, p]; wn2 = W_N_t[w + 2, i, p]
                    xp2 = x_P[w + 2, k];      xn2 = x_N[w + 2, k]
                    acc2 += Int(count_ones((wp2 & xp2) | (wn2 & xn2))) - Int(count_ones((wp2 & xn2) | (wn2 & xp2)))

                    wp3 = W_P_t[w + 3, i, p]; wn3 = W_N_t[w + 3, i, p]
                    xp3 = x_P[w + 3, k];      xn3 = x_N[w + 3, k]
                    acc3 += Int(count_ones((wp3 & xp3) | (wn3 & xn3))) - Int(count_ones((wp3 & xn3) | (wn3 & xp3)))

                    w += 4
                end
                rem_bits = 0
                while w <= n_words
                    wp = W_P_t[w, i, p]; wn = W_N_t[w, i, p]
                    xp = x_P[w, k];      xn = x_N[w, k]
                    rem_bits += Int(count_ones((wp & xp) | (wn & xn))) - Int(count_ones((wp & xn) | (wn & xp)))
                    w += 1
                end
                y[i] += s * ((acc0 + acc1) + (acc2 + acc3) + rem_bits)
            end
        end
    end
    return y
end

"""
    gray_code_bitplane_sweep(prec::TernaryDipolePreconditioner, base_P::Vector{UInt64}, base_N::Vector{UInt64}, flip_sites::AbstractVector{Int}; objective::Symbol=:min, on_step=nothing) -> (best_norm::Float64, best_gray::UInt64, n_evals::Int)

Sweeps all `2^length(flip_sites)` Gray-code combinations of `flip_sites` starting from
`(base_P, base_N)` using `bit_pos = trailing_zeros(i) + 1` (the `ctzll` pattern from
GSI-R-0017 `mdfull.c`). Flips one site bit per step and updates `y = T_ternary * x`
incrementally in `O(M * num_planes)` time per step.
"""
function gray_code_bitplane_sweep(prec::TernaryDipolePreconditioner,
                                    base_P::Vector{UInt64}, base_N::Vector{UInt64},
                                    flip_sites::AbstractVector{Int};
                                    objective::Symbol=:min,
                                    on_step=nothing)::Tuple{Float64, UInt64, Int}
    M = 3 * prec.n_atoms
    n_words = prec.n_words
    num_p = prec.num_planes
    length(base_P) == n_words && length(base_N) == n_words ||
        throw(DimensionMismatch("base_P and base_N must have length n_words=$n_words"))
    objective in (:min, :max) || throw(ArgumentError("objective must be :min or :max"))
    K = length(flip_sites)
    0 <= K <= 62 || throw(ArgumentError("flip_sites length must be in 0:62, got $K"))
    for s in flip_sites
        1 <= s <= M || throw(ArgumentError("flip_sites entry $s out of bounds 1:$M"))
    end
    length(unique(flip_sites)) == K || throw(ArgumentError("flip_sites must not contain duplicates"))

    cur_P = copy(base_P)
    cur_N = copy(base_N)
    y = zeros(Float64, M)

    W_P_t = prec.W_P_t
    W_N_t = prec.W_N_t
    W_P = prec.W_P
    W_N = prec.W_N
    row_scales = prec.row_scales

    # Exact base matvec at Gray state 0
    @inbounds for p in 1:num_p, i in 1:M
        sp = row_scales[i, p]
        sp == 0.0 && continue
        dot_bits = 0
        for w in 1:n_words
            wp = W_P_t[w, i, p]; wn = W_N_t[w, i, p]
            xp = cur_P[w];       xn = cur_N[w]
            dot_bits += Int(count_ones((wp & xp) | (wn & xn))) - Int(count_ones((wp & xn) | (wn & xp)))
        end
        y[i] += sp * dot_bits
    end

    is_signed = Vector{Bool}(undef, K)
    @inbounds for k in 1:K
        col = flip_sites[k]
        w = ((col - 1) >>> 6) + 1
        mask = UInt64(1) << ((col - 1) & 63)
        is_signed[k] = ((base_P[w] | base_N[w]) & mask) != 0
    end

    base_nonzero = any(!iszero, base_P) || any(!iszero, base_N)
    on_step !== nothing && on_step(0, UInt64(0), y, cur_P, cur_N)

    if base_nonzero || K == 0
        best_norm = norm(y)
        best_gray = UInt64(0)
        n_evals = 1
    else
        best_norm = objective === :min ? Inf : 0.0
        best_gray = UInt64(0)
        n_evals = 0
    end

    n_states = UInt64(1) << K
    @inbounds for i in UInt64(1):(n_states - UInt64(1))
        bit_pos = trailing_zeros(i) + 1
        col = flip_sites[bit_pos]
        w = ((col - 1) >>> 6) + 1
        mask = UInt64(1) << ((col - 1) & 63)

        if is_signed[bit_pos]
            was_pos = (cur_P[w] & mask) != 0
            cur_P[w] ⊻= mask
            cur_N[w] ⊻= mask
            delta = was_pos ? -2.0 : 2.0
        else
            was_pos = (cur_P[w] & mask) != 0
            cur_P[w] ⊻= mask
            delta = was_pos ? -1.0 : 1.0
        end

        for p in 1:num_p
            for row in 1:M
                wp_on = (W_P[row, w, p] & mask) != 0
                wn_on = (W_N[row, w, p] & mask) != 0
                sgn = ifelse(wp_on, 1.0, ifelse(wn_on, -1.0, 0.0))
                y[row] += delta * row_scales[row, p] * sgn
            end
        end

        gray = i ⊻ (i >> 1)
        cur_norm = norm(y)
        if (objective === :min && cur_norm < best_norm) || (objective === :max && cur_norm > best_norm)
            best_norm = cur_norm
            best_gray = gray
        end
        n_evals += 1
        on_step !== nothing && on_step(Int(i), gray, y, cur_P, cur_N)
    end

    return (best_norm, best_gray, n_evals)
end

"""
    apply_ternary_neumann_precond!(z::AbstractVector{Float64}, prec::TernaryDipolePreconditioner, r::AbstractVector{Float64}, tmp::AbstractVector{Float64})

Applies 1-step Neumann preconditioning `M⁻¹ r = D⁻¹(r - T_ternary(D⁻¹ r))` with an SPD safety guard.
If `dot(r, z) <= 0`, smoothly falls back to Jacobi preconditioning `D⁻¹ r`.
"""
function apply_ternary_neumann_precond!(z::AbstractVector{Float64}, prec::TernaryDipolePreconditioner,
                                        r::AbstractVector{Float64}, tmp::AbstractVector{Float64})
    M = 3 * prec.n_atoms
    length(z) == M && length(r) == M && length(tmp) == M || throw(DimensionMismatch("dimension mismatch in apply_ternary_neumann_precond!"))
    M == 0 && return z

    @inbounds for i in 1:M
        tmp[i] = prec.inv_diag[i] * r[i]
    end
    ternary_bitplane_matvec!(z, prec, tmp)
    @inbounds for i in 1:M
        z[i] = prec.inv_diag[i] * (r[i] - z[i])
    end

    rz = dot(r, z)
    rz_diag = dot(r, tmp)
    if !(isfinite(rz) && rz > 1e-12 * rz_diag)
        @inbounds for i in 1:M
            z[i] = tmp[i]
        end
    end
    return z
end

"""
    QuasiCyclicBlockPreconditioner

Spatial / Quasi-Cyclic block preconditioner (GSI-R-0010 / GSI-R-0017 `pinv` pattern)
pairing exact local dense `(3k × 3k)` block inverses `inv(Symmetric(A_bb))` with
far-field dual-rail ternary bitplane preconditioning.
"""
struct QuasiCyclicBlockPreconditioner
    n_atoms::Int
    block_sites::Int
    num_blocks::Int
    block_ranges::Vector{UnitRange{Int}}
    block_invs::Vector{Matrix{Float64}}
    far_ternary::TernaryDipolePreconditioner
    block_scratch::Vector{Float64}
end

"""
    build_qc_block_preconditioner(op::DipoleOperator; block_sites::Integer=3, num_planes::Integer=3) -> QuasiCyclicBlockPreconditioner

Groups consecutive `block_sites` active sites into local dense blocks `A_bb` of size `(3k × 3k)`,
computes explicit `inv(Symmetric(A_bb))` (with `pinv` fallback if near-singular), and pairs them
with far-field ternary preconditioning for the inter-block couplings.
"""
function build_qc_block_preconditioner(op::DipoleOperator; block_sites::Integer=3,
                                       num_planes::Integer=3)::QuasiCyclicBlockPreconditioner
    bs = Int(block_sites)
    bs >= 1 || throw(ArgumentError("block_sites must be >= 1, got $bs"))
    m = length(op.active)
    num_blocks = cld(m, bs)
    block_ranges = Vector{UnitRange{Int}}(undef, num_blocks)
    block_invs = Vector{Matrix{Float64}}(undef, num_blocks)

    @inbounds for b in 1:num_blocks
        s_start = (b - 1) * bs + 1
        s_end = min(m, b * bs)
        k = s_end - s_start + 1
        dim_b = 3 * k
        block_ranges[b] = (3 * (s_start - 1) + 1):(3 * s_end)

        A_bb = zeros(Float64, dim_b, dim_b)
        for u in 1:k
            a = s_start + u - 1
            isite = op.active[a]
            ia = op.inverse_alpha[a]
            ub = 3 * (u - 1)
            A_bb[ub + 1, ub + 1] = ia
            A_bb[ub + 2, ub + 2] = ia
            A_bb[ub + 3, ub + 3] = ia

            for v in (u + 1):k
                b_idx = s_start + v - 1
                jsite = op.active[b_idx]
                dx, dy, dz = _displacement(op.box, op.q, isite, jsite)
                r2 = dx * dx + dy * dy + dz * dz
                r2 < 1e-24 && continue
                r = sqrt(r2); r3 = r2 * r; r5 = r3 * r2
                g = 1.0
                if op.thole
                    aij = (max(op.alpha[isite], 1e-30) * max(op.alpha[jsite], 1e-30))^(1 / 6)
                    u_th = r / (op.a_thole > 0 ? max(op.a_thole, aij) : aij)
                    g = 1.0 - exp(-2.1304 * u_th * u_th * u_th)
                end
                c = g * op.prefactor / r3
                t = 3.0 * g * op.prefactor / r5
                vb = 3 * (v - 1)
                tdx = t * dx; tdy = t * dy; tdz = t * dz
                b11 = c - tdx * dx; b12 = -tdx * dy;    b13 = -tdx * dz
                b21 = -tdy * dx;    b22 = c - tdy * dy; b23 = -tdy * dz
                b31 = -tdz * dx;    b32 = -tdz * dy;    b33 = c - tdz * dz

                A_bb[ub + 1, vb + 1] = b11; A_bb[ub + 1, vb + 2] = b12; A_bb[ub + 1, vb + 3] = b13
                A_bb[ub + 2, vb + 1] = b21; A_bb[ub + 2, vb + 2] = b22; A_bb[ub + 2, vb + 3] = b23
                A_bb[ub + 3, vb + 1] = b31; A_bb[ub + 3, vb + 2] = b32; A_bb[ub + 3, vb + 3] = b33

                A_bb[vb + 1, ub + 1] = b11; A_bb[vb + 2, ub + 1] = b12; A_bb[vb + 3, ub + 1] = b13
                A_bb[vb + 1, ub + 2] = b21; A_bb[vb + 2, ub + 2] = b22; A_bb[vb + 3, ub + 2] = b23
                A_bb[vb + 1, ub + 3] = b31; A_bb[vb + 2, ub + 3] = b32; A_bb[vb + 3, ub + 3] = b33
            end
        end

        S_bb = Symmetric(A_bb)
        evals = eigvals(S_bb)
        if minimum(evals) > 1e-12
            block_invs[b] = Matrix(inv(S_bb))
        else
            block_invs[b] = Matrix(pinv(S_bb))
        end
    end

    far_ternary = _build_ternary_preconditioner(op; num_planes=num_planes, exclude_block_sites=bs)
    block_scratch = zeros(Float64, 3 * bs)
    return QuasiCyclicBlockPreconditioner(m, bs, num_blocks, block_ranges, block_invs, far_ternary, block_scratch)
end

"""
    apply_qc_ternary_precond!(z::AbstractVector{Float64}, prec::QuasiCyclicBlockPreconditioner, r::AbstractVector{Float64}, tmp::AbstractVector{Float64})

Applies hybrid quasi-cyclic block + far-field ternary 1-step Neumann preconditioning:
`M⁻¹ r = D_block⁻¹ (r - T_far_ternary (D_block⁻¹ r))`
with zero heap allocations and an SPD safety guard falling back to `D_block⁻¹ r`.
"""
function apply_qc_ternary_precond!(z::AbstractVector{Float64}, prec::QuasiCyclicBlockPreconditioner,
                                   r::AbstractVector{Float64}, tmp::AbstractVector{Float64})
    M = 3 * prec.n_atoms
    length(z) == M && length(r) == M && length(tmp) == M ||
        throw(DimensionMismatch("dimension mismatch in apply_qc_ternary_precond!"))
    M == 0 && return z

    # Step 1: tmp = D_block^{-1} * r
    @inbounds for b in 1:prec.num_blocks
        rng = prec.block_ranges[b]
        off = first(rng) - 1
        dim_b = length(rng)
        inv_b = prec.block_invs[b]
        for ii in 1:dim_b
            acc = 0.0
            for jj in 1:dim_b
                acc += inv_b[ii, jj] * r[off + jj]
            end
            tmp[off + ii] = acc
        end
    end

    # Step 2: z = T_far_ternary * tmp
    ternary_bitplane_matvec!(z, prec.far_ternary, tmp)

    # Step 3: z = D_block^{-1} * (r - z)
    scratch = prec.block_scratch
    @inbounds for b in 1:prec.num_blocks
        rng = prec.block_ranges[b]
        off = first(rng) - 1
        dim_b = length(rng)
        inv_b = prec.block_invs[b]
        for jj in 1:dim_b
            scratch[jj] = r[off + jj] - z[off + jj]
        end
        for ii in 1:dim_b
            acc = 0.0
            for jj in 1:dim_b
                acc += inv_b[ii, jj] * scratch[jj]
            end
            z[off + ii] = acc
        end
    end

    # SPD safety guard
    rz = dot(r, z)
    rz_block = dot(r, tmp)
    if !(isfinite(rz) && rz > 1e-12 * rz_block)
        @inbounds for i in 1:M
            z[i] = tmp[i]
        end
    end
    return z
end

"""
    certify_dipole_spd_luttinger_tisza(op::DipoleOperator) -> Tuple{Symbol, Float64}

Exact O(N^2) Bloch Luttinger-Tisza / block-Gershgorin spectral positivity certificate.
Uses the exact spectral norm `||A_ab||_2 = 2 * g_ab * prefactor / r_ab^3` of each 3x3 dipole
coupling block and computes:
    lambda_LT = min_a (1/alpha_a - sum_{b != a} ||A_ab||_2).
Returns `(:certified, lambda_LT)` when `lambda_LT > 0` (mathematically guaranteeing `A` is
strictly SPD), or `(:not_certified, lambda_LT)` conservatively otherwise.
"""
function certify_dipole_spd_luttinger_tisza(op::DipoleOperator)::Tuple{Symbol, Float64}
    m = length(op.active)
    m == 0 && return (:certified, Inf)

    row_sums = zeros(Float64, m)
    alpha_16 = Vector{Float64}(undef, m)
    @inbounds for a in 1:m
        alpha_16[a] = op.thole ? max(op.alpha[op.active[a]], 1e-30)^(1 / 6) : 0.0
    end

    @inbounds for a in 1:(m - 1)
        isite = op.active[a]
        a16 = alpha_16[a]
        for b in (a + 1):m
            jsite = op.active[b]
            dx, dy, dz = _displacement(op.box, op.q, isite, jsite)
            r2 = dx * dx + dy * dy + dz * dz
            (isfinite(r2) && r2 >= 1e-24) || return (:not_certified, -Inf)
            r = sqrt(r2)
            r3 = r2 * r
            g = 1.0
            if op.thole
                aij = a16 * alpha_16[b]
                u = r / (op.a_thole > 0 ? max(op.a_thole, aij) : aij)
                g = 1.0 - exp(-2.1304 * u * u * u)
            end
            norm_ab = 2.0 * g * op.prefactor / r3
            row_sums[a] += norm_ab
            row_sums[b] += norm_ab
        end
    end

    lambda_lt = Inf
    @inbounds for a in 1:m
        margin = op.inverse_alpha[a] - row_sums[a]
        margin < lambda_lt && (lambda_lt = margin)
    end
    if isfinite(lambda_lt) && lambda_lt > 0.0
        return (:certified, lambda_lt)
    end
    return (:not_certified, lambda_lt)
end

function _extract_field(op::DipoleOperator, E_field::Union{Nothing, AbstractVecOrMat{<:Real}})
    if E_field === nothing
        return copy(op.field)
    elseif E_field isa AbstractMatrix
        size(E_field, 1) == 3 || throw(ArgumentError("E_field matrix must have 3 rows"))
        n_sites = size(op.q, 2)
        m = length(op.active)
        if size(E_field, 2) == n_sites
            E = zeros(Float64, 3m)
            @inbounds for (ai, i) in enumerate(op.active)
                E[3*(ai-1)+1] = Float64(E_field[1, i])
                E[3*(ai-1)+2] = Float64(E_field[2, i])
                E[3*(ai-1)+3] = Float64(E_field[3, i])
            end
            return E
        elseif size(E_field, 2) == m
            E = zeros(Float64, 3m)
            @inbounds for ai in 1:m
                E[3*(ai-1)+1] = Float64(E_field[1, ai])
                E[3*(ai-1)+2] = Float64(E_field[2, ai])
                E[3*(ai-1)+3] = Float64(E_field[3, ai])
            end
            return E
        else
            throw(ArgumentError("E_field columns must match total sites ($(n_sites)) or active sites ($(m))"))
        end
    elseif E_field isa AbstractVector
        m = length(op.active)
        n_sites = size(op.q, 2)
        if length(E_field) == 3m
            return Vector{Float64}(E_field)
        elseif length(E_field) == 3n_sites
            E = zeros(Float64, 3m)
            @inbounds for (ai, i) in enumerate(op.active)
                E[3*(ai-1)+1] = Float64(E_field[3*(i-1)+1])
                E[3*(ai-1)+2] = Float64(E_field[3*(i-1)+2])
                E[3*(ai-1)+3] = Float64(E_field[3*(i-1)+3])
            end
            return E
        else
            throw(ArgumentError("E_field length must match 3*active ($(3m)) or 3*sites ($(3n_sites))"))
        end
    end
end

@inline _apply_pcg_precond!(z, P::TernaryDipolePreconditioner, r, tmp) =
    apply_ternary_neumann_precond!(z, P, r, tmp)
@inline _apply_pcg_precond!(z, P::QuasiCyclicBlockPreconditioner, r, tmp) =
    apply_qc_ternary_precond!(z, P, r, tmp)

"""
    solve_induced_dipoles_ternary_pcg(op::DipoleOperator, [E_field]; tol=1e-10, max_iter=200, num_planes=3, prec=nothing, certify_sites=64, precond_mode=:ternary, block_sites=3, lt_certify=true) -> DipoleSolveResult

Solve induced dipole equations using exact Float64 outer PCG accelerated by either a
dual-rail ternary MMSE preconditioner (`precond_mode=:ternary`) or a quasi-cyclic block +
far-field ternary preconditioner (`precond_mode=:qc_hybrid`).
"""
function solve_induced_dipoles_ternary_pcg(op::DipoleOperator, E_field::Union{Nothing, AbstractVecOrMat{<:Real}}=nothing;
                                          tol::Real=1e-10, max_iter::Integer=200, num_planes::Integer=3,
                                          prec::Union{Nothing, TernaryDipolePreconditioner, QuasiCyclicBlockPreconditioner}=nothing,
                                          certify_sites::Integer=64,
                                          precond_mode::Symbol=:ternary,
                                          block_sites::Integer=3,
                                          lt_certify::Bool=true)
    tol64 = Float64(tol)
    isfinite(tol64) && tol64 > 0 || throw(ArgumentError("tol must be finite, positive, and representable as Float64"))
    max_iter >= 0 || throw(ArgumentError("max_iter must be nonnegative"))
    precond_mode in (:ternary, :qc_hybrid) ||
        throw(ArgumentError("precond_mode must be :ternary or :qc_hybrid, got :$precond_mode"))
    n = size(op, 1)
    n == 0 && return DipoleSolveResult(Float64[], 0.0, 0, :converged, :certified, Inf)

    E = _extract_field(op, E_field)
    all(isfinite, E) || throw(ArgumentError("right-hand side field must be finite"))
    normE = norm(E)

    M_dense = n <= 3certify_sites ? dense_reference(op) : nothing
    mineig = M_dense === nothing ? NaN : minimum(eigvals(Symmetric(M_dense)))
    if M_dense !== nothing && !(mineig > 0)
        return DipoleSolveResult(nothing, Inf, 0, :indefinite, :indefinite, mineig)
    end
    if M_dense === nothing
        if lt_certify
            lt_spd, lt_bound = certify_dipole_spd_luttinger_tisza(op)
            spd = lt_spd
            mineig = lt_spd === :certified ? lt_bound : NaN
        else
            spd = :not_certified
        end
    else
        spd = :certified
    end

    P = if prec !== nothing
        prec
    elseif precond_mode === :qc_hybrid
        build_qc_block_preconditioner(op; block_sites=block_sites, num_planes=num_planes)
    else
        build_ternary_preconditioner(op; num_planes=num_planes)
    end

    x = zeros(Float64, n)
    r = copy(E)
    threshold = tol64 * normE
    if norm(r) <= threshold || normE == 0.0
        return DipoleSolveResult(x, norm(r) / max(normE, eps()), 0, :converged, spd, mineig)
    end

    tmp = zeros(Float64, n)
    z = zeros(Float64, n)
    _apply_pcg_precond!(z, P, r, tmp)
    p = copy(z)
    rz = dot(r, z)
    y = zeros(Float64, n)

    for it in 1:Int(max_iter)
        apply_dipole_operator!(y, op, p)
        curvature = dot(p, y)
        if !(isfinite(curvature) && curvature > 0.0)
            return DipoleSolveResult(nothing, Inf, it - 1, :indefinite, :indefinite, mineig)
        end
        step = rz / curvature
        @. x = x + step * p
        @. r = r - step * y

        if norm(r) <= threshold
            apply_dipole_operator!(y, op, x)
            rel = norm(y - E) / max(normE, eps())
            if rel <= tol64
                return DipoleSolveResult(x, rel, it, :converged, spd, mineig)
            end
        end

        _apply_pcg_precond!(z, P, r, tmp)
        rznew = dot(r, z)
        beta = rznew / rz
        @. p = z + beta * p
        rz = rznew
    end

    apply_dipole_operator!(y, op, x)
    rel = norm(y - E) / max(normE, eps())
    if rel <= tol64
        return DipoleSolveResult(x, rel, Int(max_iter), :converged, spd, mineig)
    end
    return DipoleSolveResult(nothing, rel, Int(max_iter), :not_converged, spd, mineig)
end

"""
    solve_induced_dipoles(op::DipoleOperator, [E_field]; tol=1e-10, max_iter=max(1, 4*size(op, 1)), certify_sites=64, lt_certify=true) -> DipoleSolveResult

Solve induced dipole equations using standard Jacobi-preconditioned CG.
"""
function solve_induced_dipoles(op::DipoleOperator, E_field::Union{Nothing, AbstractVecOrMat{<:Real}}=nothing;
                               tol::Real=1e-10, max_iter::Integer=max(1, 4*size(op, 1)),
                               certify_sites::Integer=64, lt_certify::Bool=true)
    if E_field === nothing
        return solve_dipoles!(op; tol=tol, maxiter=max_iter, certify_sites=certify_sites, lt_certify=lt_certify)
    else
        E = _extract_field(op, E_field)
        op_with_field = DipoleOperator(op.q, op.charges, op.alpha, op.active, op.inverse_alpha,
                                       op.box, op.prefactor, op.rcut, E, op.thole, op.a_thole, op.periodic)
        return solve_dipoles!(op_with_field; tol=tol, maxiter=max_iter, certify_sites=certify_sites, lt_certify=lt_certify)
    end
end

# ---------------------------------------------------------------------------
# Circulant Weighing CW(n, k) Orthogonal Noise & Multiplication-Free Block Filter
# ---------------------------------------------------------------------------

const _PO_CW_TABLE = Dict{Int, Tuple{Int, Vector{Int8}}}(
    3  => (1,  Int8[1, 0, 0]),
    4  => (4,  Int8[-1, 1, 1, 1]),
    6  => (4,  Int8[-1, 1, 0, 1, 1, 0]),
    7  => (4,  Int8[-1, 1, 1, 0, 1, 0, 0]),
    13 => (9,  Int8[-1, -1, 1, 1, -1, 1, 0, 1, 0, 1, 1, 0, 0]),
    14 => (4,  Int8[1, 0, 0, -1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]),
    21 => (16, Int8[-1, -1, -1, -1, -1, 1, -1, 0, -1, 0, 1, -1, -1, 1, 0, 0, -1, 1, 0, 1, 1]),
)

function _cw_params(n_block::Int)
    entry = get(_PO_CW_TABLE, n_block, nothing)
    entry === nothing && throw(ArgumentError("unsupported CW block size n_block=$n_block; supported: 3, 4, 6, 7, 13, 14, 21"))
    return entry
end

"""
    cw_orthogonal_noise(rng::AbstractRNG, n_cols::Int; n_block::Int=13) -> Matrix{Float64}
    cw_orthogonal_noise(n_cols::Int; n_block::Int=13) -> Matrix{Float64}

Generate an `n_block × n_cols` matrix of structured orthogonal random probe vectors
using canonical Circulant Weighing matrices `CW(n_block, k)` (default `CW(13, 9)`, or
`CW(7, 4)`) with strictly ternary `{-1, 0, +1}` weights scaled so each full block
satisfies exact row orthogonality `Z_blk * Z_blk' == n_block * I` (`W * W' / k == I`)
and exact isotropic covariance `E[z * z'] == I`, reducing trace-estimation variance
compared to independent standard normal samples.
"""
function cw_orthogonal_noise(rng::AbstractRNG, n_cols::Int; n_block::Int=13)::Matrix{Float64}
    n_cols > 0 || throw(ArgumentError("n_cols must be positive, got $n_cols"))
    k, row = _cw_params(n_block)
    scale = sqrt(Float64(n_block) / Float64(k))
    n_blocks = cld(n_cols, n_block)
    Z = Matrix{Float64}(undef, n_block, n_cols)
    r1 = Vector{Float64}(undef, n_block)
    r2 = Vector{Float64}(undef, n_block)
    perm = Vector{Int}(undef, n_block)

    @inbounds for b in 0:(n_blocks - 1)
        for i in 1:n_block
            r1[i] = rand(rng, Bool) ? 1.0 : -1.0
            r2[i] = rand(rng, Bool) ? 1.0 : -1.0
        end
        randperm!(rng, perm)
        c0 = b * n_block
        ncols_b = min(n_block, n_cols - c0)
        for c in 1:ncols_b
            j = perm[c]
            s_col = scale * r2[c]
            for i in 1:n_block
                w = row[mod(j - i, n_block) + 1]
                if w == Int8(1)
                    Z[i, c0 + c] = r1[i] * s_col
                elseif w == Int8(-1)
                    Z[i, c0 + c] = -r1[i] * s_col
                else
                    Z[i, c0 + c] = 0.0
                end
            end
        end
    end
    return Z
end

cw_orthogonal_noise(n_cols::Int; n_block::Int=13)::Matrix{Float64} =
    cw_orthogonal_noise(Random.default_rng(), n_cols; n_block=n_block)

@inline function _cw_block_apply!(dst::AbstractVector{Float64}, dst_off::Int, dst_len::Int,
                                  src::AbstractVector{Float64}, src_off::Int,
                                  W::Matrix{Int8}, n_block::Int,
                                  transpose_w::Bool, scale::Float64)
    @inbounds for i in 1:dst_len
        acc = 0.0
        for j in 1:n_block
            w = transpose_w ? W[j, i] : W[i, j]
            if w == Int8(1)
                acc += src[src_off + j]
            elseif w == Int8(-1)
                acc -= src[src_off + j]
            end
        end
        dst[dst_off + i] = scale == 1.0 ? acc : acc * scale
    end
    return nothing
end

"""
    apply_cw_block_filter!(y::AbstractVector{Float64}, x::AbstractVector{Float64};
                           n_block::Int=13, mode::Symbol=:roundtrip) -> AbstractVector{Float64}

Apply a fast multiplication-free Circulant Weighing `CW(n_block, k)` orthogonal block
filter across `x` using strictly additions/subtractions for `W * v` and `W' * z`
followed by final `1 / k` scaling:
- `mode === :roundtrip` (default): applies `(1/k) * W' * (W * x_blk)` per block (`y ≈ x` to `< 1e-14`).
- `mode === :forward`: applies `W * x_blk` per block.
- `mode === :inverse`: applies `(1/k) * W' * x_blk` per block.
"""
function apply_cw_block_filter!(y::AbstractVector{Float64},
                                x::AbstractVector{Float64};
                                n_block::Int=13,
                                mode::Symbol=:roundtrip)
    N = length(x)
    length(y) == N || throw(ArgumentError("output length $(length(y)) must match input length $N"))
    Base.mightalias(y, x) && throw(ArgumentError("output y must not alias input x"))
    mode in (:roundtrip, :forward, :inverse) ||
        throw(ArgumentError("mode must be :roundtrip, :forward, or :inverse, got :$mode"))

    k, row = _cw_params(n_block)
    inv_k = 1.0 / Float64(k)
    N == 0 && return y

    W = Matrix{Int8}(undef, n_block, n_block)
    @inbounds for j in 1:n_block, i in 1:n_block
        W[i, j] = row[mod(j - i, n_block) + 1]
    end
    z = Vector{Float64}(undef, n_block)
    xbuf = Vector{Float64}(undef, n_block)

    offset = 0
    @inbounds while offset < N
        len = min(n_block, N - offset)
        src_vec = x
        src_off = offset
        if len < n_block
            for j in 1:len
                xbuf[j] = x[offset + j]
            end
            for j in (len + 1):n_block
                xbuf[j] = 0.0
            end
            src_vec = xbuf
            src_off = 0
        end

        if mode === :roundtrip
            _cw_block_apply!(z, 0, n_block, src_vec, src_off, W, n_block, false, 1.0)
            _cw_block_apply!(y, offset, len, z, 0, W, n_block, true, inv_k)
        elseif mode === :forward
            _cw_block_apply!(y, offset, len, src_vec, src_off, W, n_block, false, 1.0)
        else
            _cw_block_apply!(y, offset, len, src_vec, src_off, W, n_block, true, inv_k)
        end
        offset += n_block
    end
    return y
end

end
