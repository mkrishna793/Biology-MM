using LinearAlgebra, Random, Printf

include(joinpath(@__DIR__, "..", "..", "src", "FoundationCore.jl"))
using .FoundationCore.PolarizationOperator

const JULIA = VERSION

function lattice(n; spacing=2.0, alpha=0.002, seed=41)
    rng = MersenneTwister(seed)
    m = ceil(Int, n^(1/3))
    q = Matrix{Float64}(undef, 3, n)
    for i in 1:n
        ix = (i-1) % m
        iy = ((i-1) ÷ m) % m
        iz = (i-1) ÷ (m*m)
        q[:,i] = spacing .* (ix, iy, iz) .+ 0.03 .* randn(rng, 3)
    end
    z = randn(rng, n)
    a = fill(alpha, n)
    return q, z, a
end

function dense_oracle(A)
    n = size(A, 1)
    n <= 600 || error("dense oracle limited to at most 200 atoms")
    D = zeros(n, n)
    for j in 1:n
        e = zeros(n); e[j] = 1
        y = zeros(n); mul!(y, A, e); D[:,j] = y
    end
    D
end

function pcg(A, b; x0=zeros(length(b)), Mdiag=ones(length(b)), tol=1e-9, maxiter=500)
    x = copy(x0); r = b - A*x; z = r ./ Mdiag; p = copy(z)
    rz = dot(r,z); hist = [norm(r)/max(norm(b), eps())]
    for it in 1:maxiter
        Ap = A*p; den = dot(p,Ap)
        den > 0 || return x, it-1, hist, false
        step = rz/den; x .+= step .* p; r .-= step .* Ap
        push!(hist, norm(r)/max(norm(b), eps()))
        if hist[end] <= tol
            return x, it, hist, true
        end
        z .= r ./ Mdiag; rznew = dot(r,z); p .= z .+ (rznew/rz).*p; rz=rznew
    end
    x, maxiter, hist, false
end

function oracle_and_solvers()
    q,z,a = lattice(24; alpha=0.0002)
    A = build_operator(q,z,a)
    D = dense_oracle(A)
    rng = MersenneTwister(9); x=randn(rng,size(A,1)); y=zeros(length(x)); mul!(y,A,x)
    relop = norm(y-D*x)/norm(D*x)
    mu_ref = D \ A.field
    mu = solve_dipoles!(A; tol=1e-11, certify_sites=100)
    @printf("ORACLE N=24 op_rel=%.3e solve_rel=%.3e residual=%.3e iterations=%d spd=%s\n",
        relop, norm(require_converged(mu)-mu_ref)/norm(mu_ref), mu.residual, mu.iterations, string(mu.spd))
    return relop
end

function warm_preconditioner()
    q,z,a = lattice(90; alpha=0.0003)
    A = build_operator(q,z,a)
    D = dense_oracle(A)
    b=A.field
    diag=repeat(A.inverse_alpha, inner=3)
    _,itd,_,okd=pcg(D,b; Mdiag=diag,tol=1e-9)
    mu,iti,_,oki=pcg(D,b; x0=0.98 .* (D\b) .+ 0.02 .* randn(MersenneTwister(2),length(b)), Mdiag=diag,tol=1e-9)
    @printf("PCG N=90 diagonal_cold=%d warm_perturbed=%d cold_ok=%s warm_ok=%s true_resid=%.3e\n",itd,iti,string(okd),string(oki),norm(D*mu-b)/norm(b))
    # Exact 3x3 site block Jacobi, retaining only the diagonal site blocks.
    blocks=[inv(D[3i-2:3i,3i-2:3i]) for i in 1:90]
    function blockpcg(b)
        x=zeros(length(b)); r=copy(b)
        apply(v) = vcat((blocks[i] * v[3i-2:3i] for i in 1:90)...)
        z=apply(r); p=copy(z); rz=dot(r,z)
        for it in 1:500
            Ap=D*p; st=rz/dot(p,Ap); x .+= st.*p; r .-= st.*Ap
            norm(r)/norm(b) <= 1e-9 && return it,norm(r)/norm(b)
            z=apply(r); rz2=dot(r,z); p .= z .+ (rz2/rz).*p; rz=rz2
        end
        return 500,norm(r)/norm(b)
    end
    ib,rb=blockpcg(b)
    @printf("PCG N=90 block_jacobi=%d residual=%.3e\n",ib,rb)
end

function adjoint_checks()
    # A(theta)=A0+theta*B, E(theta)=E0+theta*f. Analytic envelope derivative
    # and adjoint derivative are compared to fully re-solved central differences.
    q,z,a=lattice(5; alpha=0.00015)
    A0=build_operator(q,z,a); D0=dense_oracle(A0); E0=A0.field
    rng=MersenneTwister(12); R=randn(rng,size(D0)...); B=0.5*(R+R')*1e-3; f=randn(rng,length(E0))*1e-3
    theta=0.17; h=1e-5
    function eval(theta)
        D=D0+theta*B; E=E0+theta*f; mu=D\E
        return 0.5dot(mu,D*mu)-dot(mu,E), mu, D, E
    end
    U,mu,D,E=eval(theta)
    fd=(eval(theta+h)[1]-eval(theta-h)[1])/(2h)
    env=0.5dot(mu,B*mu)-dot(mu,f)
    # Observable g(mu)=c'mu + 0.2/2 ||mu||^2, derivative by one adjoint solve.
    c=randn(rng,length(mu)); g(v)=dot(c,v)+0.1dot(v,v)
    gradmu=c+0.2mu; lambda=D'\gradmu
    dmu=f-B*mu
    adj=dot(lambda,dmu)
    fdg=(g(eval(theta+h)[2])-g(eval(theta-h)[2]))/(2h)
    @printf("ADJOINT envelope_abs=%.3e envelope_rel=%.3e observable_abs=%.3e observable_rel=%.3e\n",abs(fd-env),abs(fd-env)/max(abs(fd),eps()),abs(fdg-adj),abs(fdg-adj)/max(abs(fdg),eps()))
end

function timing(n)
    q,z,a=lattice(n; alpha=1e-5)
    tbuild=@elapsed A=build_operator(q,z,a)
    x=randn(MersenneTwister(1),size(A,1)); y=zeros(length(x)); mul!(y,A,x)
    # Warmed single-thread matvec; do not run dense reference for this scale.
    mul!(y,A,x)
    t=@elapsed mul!(y,A,x)
    @printf("DIRECT n=%d build_s=%.6f matvec_s=%.6f threads=%d checksum=%.8e\n",n,tbuild,t,Threads.nthreads(),sum(y))
end

if abspath(PROGRAM_FILE) == @__FILE__
    BLAS.set_num_threads(1)
    if length(ARGS)>0 && ARGS[1]=="--bench"
        timing(parse(Int,ARGS[2]))
    else
        println("Julia ", JULIA, " threads=", Threads.nthreads(), " BLAS threads=", BLAS.get_num_threads())
        oracle_and_solvers(); warm_preconditioner(); adjoint_checks()
    end
end
