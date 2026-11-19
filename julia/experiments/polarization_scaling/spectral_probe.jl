using LinearAlgebra, Random, Printf, FFTW, AbstractFFTs, Statistics

const SQRTPI = sqrt(pi)
erfc_std(x::Float64) = ccall(:erfc, Cdouble, (Cdouble,), x)

"""Direct Gaussian Ewald dipole field tensor, spherical vacuum boundary."""
function direct_ewald(q, mu; box, alpha, rcut, kmax, vacuum=true)
    n=size(q,2); V=prod(box); E=zeros(3,n)
    # Real-space Hessian of erfc(alpha*r)/r, including periodic images.
    nmax=ceil.(Int, rcut ./ box)
    for i in 1:n, j in 1:n, ix in -nmax[1]:nmax[1], iy in -nmax[2]:nmax[2], iz in -nmax[3]:nmax[3]
        i==j && ix==0 && iy==0 && iz==0 && continue
        d=q[:,i]-q[:,j]+box.*[ix,iy,iz]; r2=dot(d,d); r=sqrt(r2)
        r<=rcut || continue
        er=erfc_std(alpha*r); ex=exp(-(alpha*r)^2)
        cI=er/r^3 + 2alpha/SQRTPI*ex/r^2
        cR=3er/r^5 + (6alpha/r^4 + 4alpha^3/r^2)/SQRTPI*ex
        E[:,i] .+= cR.*d.*dot(d,mu[:,j]) .- cI.*mu[:,j]
    end
    # Reciprocal Hessian of the long-range Gaussian Coulomb kernel.
    for nx in -kmax:kmax, ny in -kmax:kmax, nz in -kmax:kmax
        nx==0 && ny==0 && nz==0 && continue
        k=2pi .* [nx/box[1],ny/box[2],nz/box[3]]; k2=dot(k,k)
        c=-4pi/V*exp(-k2/(4alpha^2))/k2
        for i in 1:n
            phase=dot(k,q[:,i]); s=0.0
            for j in 1:n; s+=dot(k,mu[:,j])*cos(phase-dot(k,q[:,j])); end
            E[:,i] .+= c.*k.*s
        end
    end
    # Gaussian dipole self tensor; remove the i=j contribution of the smooth part.
    E .+= (4alpha^3/(3SQRTPI)).*mu
    # Vacuum spherical surface: U_s=2pi |M|^2/(3V), so E_s=-dU/dmu.
    if vacuum
        E .-= (4pi/(3V)).*sum(mu,dims=2)
    end
    E
end

function ngp_fft(q, mu; box, dims, alpha=0.55, kmax=typemax(Int), vacuum=true)
    n=size(q,2); rho=zeros(ComplexF64, dims...,3); index=Vector{NTuple{3,Int}}(undef,n)
    for j in 1:n
        ij=ntuple(d->mod(floor(Int,q[d,j]/box[d]*dims[d]),dims[d])+1,3); index[j]=ij
        for d in 1:3; rho[ij...,d]+=mu[d,j]; end
    end
    khat=[ [i<=dims[d]÷2 ? i-1 : i-1-dims[d] for i in 1:dims[d]] for d in 1:3 ]
    for d in 1:3; rho[:,:, :,d] .= fft(rho[:,:,:,d]); end
    for i in 1:dims[1], j in 1:dims[2], k in 1:dims[3]
        kv=2pi .* [khat[1][i]/box[1],khat[2][j]/box[2],khat[3][k]/box[3]]; k2=dot(kv,kv)
        mode=max(abs(khat[1][i]),abs(khat[2][j]),abs(khat[3][k]))
        if k2==0 || mode>kmax
            rho[i,j,k,:].=0
        else
            c=-4pi/prod(box)*exp(-k2/(4alpha^2))/k2*prod(dims)
            v=ComplexF64[rho[i,j,k,1],rho[i,j,k,2],rho[i,j,k,3]]
            rho[i,j,k,1:3] .= c .* kv .* dot(kv,v)
        end
    end
    for d in 1:3; rho[:,:,:,d] .= ifft(rho[:,:,:,d]); end
    out=zeros(3,n)
    for j in 1:n, d in 1:3; out[d,j]=real(rho[index[j]...,d]); end
    vacuum && (out .-= (4pi/(3prod(box))).*sum(mu,dims=2))
    out
end

function reciprocal_reference(q,mu;box,alpha,kmax)
    n=size(q,2); E=zeros(3,n); V=prod(box)
    for nx in -kmax:kmax, ny in -kmax:kmax, nz in -kmax:kmax
        nx==0 && ny==0 && nz==0 && continue
        kv=2pi .* [nx/box[1],ny/box[2],nz/box[3]]; k2=dot(kv,kv)
        c=-4pi/V*exp(-k2/(4alpha^2))/k2
        for i in 1:n
            phase=dot(kv,q[:,i]); s=0.0
            for j in 1:n; s+=dot(kv,mu[:,j])*cos(phase-dot(kv,q[:,j])); end
            E[:,i] .+= c.*kv.*s
        end
    end
    E
end

function random_fixture(n, L; seed=11)
    rng=MersenneTwister(seed); q=rand(rng,3,n).*L; mu=randn(rng,3,n)
    q,mu
end

function relerr(x,y); norm(x-y)/max(norm(y),eps()); end

function surface_test()
    L=[8.,8.,8.]; q=[1. 4.;2. 5.;3. 6.]; mu=[1. 1.;0. 0.;0. 0.]
    full=direct_ewald(q,mu;box=L,alpha=.4,rcut=12.,kmax=8,vacuum=true)
    foil=direct_ewald(q,mu;box=L,alpha=.4,rcut=12.,kmax=8,vacuum=false)
    expected=-(4pi/(3prod(L))).*sum(mu,dims=2)
    @assert norm((full-foil)-repeat(expected,1,2)) < 1e-12
    @assert norm(full-foil)>1e-3
    @printf("SURFACE M=%s V=%.1f dE_per_site=%.9e omission_delta=%.9e PASS\n",string(vec(sum(mu,dims=2))),prod(L),dot(vec(sum(mu,dims=2)),vec(sum(mu,dims=2)))*2pi/(3prod(L)),norm(full-foil))
end

function apply_A_fft(x,q,alpha;box,dims)
    mu=reshape(x,3,:)
    # T is the reciprocal electric-field operator; A = alpha^-1 I - T.
    y=-ngp_fft(q,mu;box,dims,alpha=.55,kmax=typemax(Int),vacuum=true)
    y .+= mu ./ reshape(alpha,1,:)
    vec(y)
end

function pcg_fixture()
    n=64; L=fill((n/0.8)^(1/3),3); q,_=random_fixture(n,L;seed=77)
    dims=(16,16,16); rng=MersenneTwister(83); spread=0.05 .* randn(rng,n)
    # Set a close-to-instability but SPD polarizability scale from a measured
    # spectral radius, making mutual coupling substantial and reproducible.
    probe=fill(1.0,n); K=zeros(3n,3n)
    for j in 1:3n
        e=zeros(3n); e[j]=1
        K[:,j]=apply_A_fft(e,q,ones(n);box=L,dims=dims)-e
    end
    rho=maximum(abs,eigvals(Symmetric(K)))
    alpha=0.72/rho .* exp.(spread)
    A=zeros(3n,3n)
    for j in 1:3n
        e=zeros(3n); e[j]=1; A[:,j]=apply_A_fft(e,q,alpha;box=L,dims=dims)
    end
    A=Symmetric((A+A')/2); b=randn(rng,3n); xref=A\b
    x=zeros(3n); r=b-copy(A*x); z=copy(r); p=copy(r); rz=dot(r,z); it=0
    for k in 1:1000
        Ap=apply_A_fft(p,q,alpha;box=L,dims=dims); den=dot(p,Ap); den>0 || error("nonpositive PCG curvature")
        st=rz/den; x .+= st.*p; r .-= st.*Ap; it=k
        norm(r)/norm(b)<1e-9 && break
        rz2=dot(r,r); p .= r .+ (rz2/rz).*p; rz=rz2
    end
    @printf("PCG_FFT n=%d density=0.8 rhoK=%.6e alpha_mean=%.6e alpha_cv=%.3f iterations=%d residual=%.6e dense_rel=%.6e min_eig=%.6e\n",n,rho,mean(alpha),std(alpha)/mean(alpha),it,norm(b-A*x)/norm(b),norm(x-xref)/norm(xref),minimum(eigvals(A)))
end

function error_curve()
    L=[12.,12.,12.]; q,mu=random_fixture(32,L)
    for dims in ((8,8,8),(12,12,12),(16,16,16),(24,24,24))
        ref=reciprocal_reference(q,mu;box=L,alpha=.55,kmax=9)
        got=ngp_fft(q,mu;box=L,dims=dims,alpha=.55,kmax=9,vacuum=false)
        @printf("ERROR reciprocal_only dims=%s h=%.3f field_rel=%.6e energy_rel=%.6e\n",string(dims),L[1]/dims[1],relerr(got,ref),abs(dot(mu,got)-dot(mu,ref))/max(abs(dot(mu,ref)),eps()))
        total=direct_ewald(q,mu;box=L,alpha=.55,rcut=9.,kmax=9)
        @printf("TOTAL_EWALD_NGP dims=%s field_rel=%.6e (missing real+self; total error does not converge with grid)\n",string(dims),relerr(got,total))
    end
    for a in (.40,.55,.70), km in (6,9,12)
        r=reciprocal_reference(q,mu;box=L,alpha=a,kmax=km)
        f=ngp_fft(q,mu;box=L,dims=(24,24,24),alpha=a,kmax=km,vacuum=false)
        @printf("PARAM alpha=%.2f kmax=%d grid24_field_rel=%.6e energy_rel=%.6e recip_norm=%.6e\n",a,km,relerr(f,r),abs(dot(mu,f)-dot(mu,r))/max(abs(dot(mu,r)),eps()),norm(r))
    end
    println("DIRECT_EWALD_CONVERGENCE; compare field to alpha=.55, rcut=12, kmax=12")
    tight=direct_ewald(q,mu;box=L,alpha=.55,rcut=12.,kmax=12)
    for (a,rc,km) in ((.40,16.,8),(.50,13.,10),(.55,12.,12),(.70,10.,16))
        v=direct_ewald(q,mu;box=L,alpha=a,rcut=rc,kmax=km)
        @printf("EWALD alpha=%.2f rcut=%.1f kmax=%d field_rel=%.6e energy_rel=%.6e\n",a,rc,km,relerr(v,tight),abs(dot(mu,v)-dot(mu,tight))/max(abs(dot(mu,tight)),eps()))
    end
end

function scaling()
    for n in (1000,10000,100000,1000000)
        # Fixed density; padded grid dimensions round each side up to powers of two.
        L=fill((n/0.05)^(1/3),3); q,mu=random_fixture(n,L)
        side=2^ceil(Int,log2(max(8,ceil(Int,n^(1/3)*1.25)))); dims=(side,side,side)
        gc=@elapsed begin
            y=ngp_fft(q,mu;box=L,dims=dims)
        end
        ngp_fft(q[:,1:min(n,16)],mu[:,1:min(n,16)];box=L,dims=dims) # warm FFT sizes
        samples=Float64[]
        for _ in 1:3
            push!(samples,@elapsed y=ngp_fft(q,mu;box=L,dims=dims))
        end
        @printf("FFT n=%d dims=%s matvec_ms=%.3f samples_ms=%s setup_s=%.3f checksum=%.8e\n",n,string(dims),1e3median(samples),string(round.(1e3 .* samples;digits=3)),gc,sum(y))
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    BLAS.set_num_threads(1); FFTW.set_num_threads(1)
    println("JULIA ",VERSION," FFTW ",pkgversion(FFTW)," AbstractFFTs ",pkgversion(AbstractFFTs)," threads=",Threads.nthreads()," FFTW_threads=1 BLAS_threads=",BLAS.get_num_threads())
    surface_test(); error_curve(); pcg_fixture(); scaling()
end
