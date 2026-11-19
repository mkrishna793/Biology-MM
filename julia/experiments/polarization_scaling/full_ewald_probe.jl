using LinearAlgebra, Random, Printf, FFTW, Statistics

const SQRTPI = sqrt(pi)
erfc_std(x::Float64) = ccall(:erfc, Cdouble, (Cdouble,), x)

function direct_ewald(q, mu; box, alpha, rcut, kmax, vacuum=true)
    n=size(q,2); V=prod(box); E=zeros(3,n); nmax=ceil.(Int,rcut ./ box)
    for i in 1:n, j in 1:n, ix in -nmax[1]:nmax[1], iy in -nmax[2]:nmax[2], iz in -nmax[3]:nmax[3]
        i==j && ix==0 && iy==0 && iz==0 && continue
        d=q[:,i]-q[:,j]+box.*[ix,iy,iz]; r2=dot(d,d); r=sqrt(r2)
        r<=rcut || continue
        er=erfc_std(alpha*r); ex=exp(-(alpha*r)^2)
        cI=er/r^3 + 2alpha/SQRTPI*ex/r^2
        cR=3er/r^5 + (6alpha/r^4 + 4alpha^3/r^2)/SQRTPI*ex
        E[:,i] .+= cR.*d.*dot(d,mu[:,j]) .- cI.*mu[:,j]
    end
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
    E .+= (4alpha^3/(3SQRTPI)).*mu
    vacuum && (E .-= (4pi/(3V)).*sum(mu,dims=2))
    E
end

"""Centered cardinal B-spline of even order, support [-p/2,p/2]."""
function cardinal_bspline(p::Int, x::Float64)
    abs(x) >= p/2 && return 0.0
    z=x+p/2; s=0.0
    for k in 0:p
        t=z-k
        t > 0 && (s += (-1)^k * binomial(p,k) * t^(p-1))
    end
    s/factorial(p-1)
end

function spline_stencil(u::Float64, p::Int, ngrid::Int)
    firstnode=floor(Int,u)-p÷2+1
    nodes=Vector{Int}(undef,p); weights=Vector{Float64}(undef,p)
    for a in 1:p
        node=firstnode+a-1
        nodes[a]=mod(node,ngrid)+1
        weights[a]=cardinal_bspline(p,u-node)
    end
    weights ./= sum(weights)
    nodes,weights
end

function reciprocal_fft_before(q,mu; box,dims,alpha,kmax=typemax(Int),order=4)
    order in (4,6) || error("spline order must be 4 or 6")
    n=size(q,2); rho=zeros(ComplexF64,dims...,3); stencils=Vector{Any}(undef,n)
    for a in 1:n
        st=ntuple(d->spline_stencil(q[d,a]/box[d]*dims[d],order,dims[d]),3); stencils[a]=st
        for i in 1:order,j in 1:order,k in 1:order,c in 1:3
            rho[st[1][1][i],st[2][1][j],st[3][1][k],c] += mu[c,a]*st[1][2][i]*st[2][2][j]*st[3][2][k]
        end
    end
    for c in 1:3; rho[:,:,:,c] .= fft(rho[:,:,:,c]); end
    for i in 1:dims[1],j in 1:dims[2],k in 1:dims[3]
        ns=(i<=dims[1]÷2 ? i-1 : i-1-dims[1],j<=dims[2]÷2 ? j-1 : j-1-dims[2],k<=dims[3]÷2 ? k-1 : k-1-dims[3])
        # Nyquist planes are omitted to keep the ik derivative unambiguous and Hermitian.
        any(d->iseven(dims[d]) && abs(ns[d])==dims[d]÷2,1:3) && (rho[i,j,k,:].=0; continue)
        kv=2pi .* [ns[d]/box[d] for d in 1:3]; k2=dot(kv,kv)
        if k2==0 || maximum(abs,ns)>kmax
            rho[i,j,k,:].=0; continue
        end
        window=prod(sinc(ns[d]/dims[d])^order for d in 1:3)
        coeff=(-4pi/prod(box))*exp(-k2/(4alpha^2))/k2*prod(dims)/(window^2)
        v=ComplexF64[rho[i,j,k,1],rho[i,j,k,2],rho[i,j,k,3]]
        rho[i,j,k,1:3] .= coeff .* kv .* dot(kv,v)
    end
    for c in 1:3; rho[:,:,:,c] .= ifft(rho[:,:,:,c]); end
    out=zeros(3,n)
    for a in 1:n
        st=stencils[a]
        for i in 1:order,j in 1:order,k in 1:order,c in 1:3
            out[c,a] += real(rho[st[1][1][i],st[2][1][j],st[3][1][k],c])*st[1][2][i]*st[2][2][j]*st[3][2][k]
        end
    end
    out
end

function real_cell_field_before(q,mu;box,alpha,rcut)
    all(rcut .< box./2) || error("real-space cutoff must be below L/2")
    n=size(q,2); nc=max.(1,floor.(Int,box./rcut)); ncell=prod(nc)
    head=zeros(Int,ncell); next=zeros(Int,n); cellof=zeros(Int,n)
    linear(c)=c[1]+nc[1]*(c[2]-1)+nc[1]*nc[2]*(c[3]-1)
    for a in 1:n
        c=ntuple(d->clamp(floor(Int,q[d,a]/box[d]*nc[d]),0,nc[d]-1)+1,3); id=linear(c)
        cellof[a]=id; next[a]=head[id]; head[id]=a
    end
    E=zeros(3,n)
    for i in 1:n
        ci=ntuple(d->mod(floor(Int,q[d,i]/box[d]*nc[d]),nc[d])+1,3)
        cells=Int[]
        if minimum(nc)>=3
            for dx in -1:1,dy in -1:1,dz in -1:1
                push!(cells,linear(ntuple(d->mod(ci[d]+(dx,dy,dz)[d]-1,nc[d])+1,3)))
            end
        else
            append!(cells,1:ncell)
        end
        for cid in unique(cells)
            j=head[cid]
            while j != 0
                if i != j
                    d=q[:,i]-q[:,j]; d .-= box.*round.(d./box); r2=dot(d,d); r=sqrt(r2)
                    if 0 < r <= rcut
                        er=erfc_std(alpha*r); ex=exp(-(alpha*r)^2)
                        ci0=er/r^3+2alpha/SQRTPI*ex/r^2
                        cr=3er/r^5+(6alpha/r^4+4alpha^3/r^2)/SQRTPI*ex
                        E[:,i] .+= cr.*d.*dot(d,mu[:,j]).-ci0.*mu[:,j]
                    end
                end
                j=next[j]
            end
        end
    end
    E
end

# Reusable structures for repeated operator applications at fixed coordinates.
struct Stencil{P}
    nodes::NTuple{3,NTuple{P,Int}}
    weights::NTuple{3,NTuple{P,Float64}}
end

mutable struct EwaldWorkspace{P}
    nc::NTuple{3,Int}
    cellof::Vector{Int}
    head::Vector{Int}
    next::Vector{Int}
    cellpairs::Vector{Tuple{Int,Int}}
    stencils::Vector{Stencil{P}}
    rho::NTuple{3,Array{ComplexF64,3}}
    fftplans::NTuple{3,Any}
    ifftplans::NTuple{3,Any}
    out::Matrix{Float64}
    realbuf::Matrix{Float64}
end

function make_workspace(q; box, rcut, dims, order=4)
    order in (4,6) || error("spline order must be 4 or 6")
    n=size(q,2); nc=ntuple(d->max(1,ceil(Int,box[d]/rcut)),3)
    radius=ntuple(d->ceil(Int,rcut/(box[d]/nc[d]))+1,3)
    cellof=zeros(Int,n); head=zeros(prod(nc)); next=zeros(Int,n)
    for a in 1:n
        c=ntuple(d->clamp(floor(Int,q[d,a]/box[d]*nc[d]),0,nc[d]-1)+1,3)
        id=c[1]+nc[1]*(c[2]-1)+nc[1]*nc[2]*(c[3]-1)
        cellof[a]=id
        next[a]=head[id]; head[id]=a
    end
    cellpairs=Tuple{Int,Int}[]; seen=zeros(Int,prod(nc))
    for c in 1:prod(nc)
        c1=(c-1)%nc[1]+1; c2=((c-1)÷nc[1])%nc[2]+1; c3=(c-1)÷(nc[1]*nc[2])+1
        for dz in -radius[3]:radius[3],dy in -radius[2]:radius[2],dx in -radius[1]:radius[1]
            t1=mod(c1+dx-1,nc[1])+1; t2=mod(c2+dy-1,nc[2])+1; t3=mod(c3+dz-1,nc[3])+1
            t=t1+nc[1]*(t2-1)+nc[1]*nc[2]*(t3-1)
            (t<c || seen[t]==c) && continue
            seen[t]=c
            dx=min(abs(c1-t1),nc[1]-abs(c1-t1)); dy=min(abs(c2-t2),nc[2]-abs(c2-t2)); dz=min(abs(c3-t3),nc[3]-abs(c3-t3))
            gx=max(0,(dx-1)*box[1]/nc[1]); gy=max(0,(dy-1)*box[2]/nc[2]); gz=max(0,(dz-1)*box[3]/nc[3])
            gx^2+gy^2+gz^2<=rcut^2 && push!(cellpairs,(c,t))
        end
    end
    st=Vector{Stencil{order}}(undef,n)
    for a in 1:n
        nodes=ntuple(d->begin
            u=q[d,a]/box[d]*dims[d]; first=floor(Int,u)-order÷2+1
            ntuple(k->mod(first+k-1,dims[d])+1,order)
        end,3)
        weights=ntuple(d->begin
            u=q[d,a]/box[d]*dims[d]; first=floor(Int,u)-order÷2+1
            w=ntuple(k->cardinal_bspline(order,u-(first+k-1)),order)
            z=sum(w); ntuple(k->w[k]/z,order)
        end,3)
        st[a]=Stencil{order}(nodes,weights)
    end
    grids=ntuple(_->zeros(ComplexF64,dims),3)
    fp=ntuple(c->plan_fft!(grids[c]; flags=FFTW.ESTIMATE),3)
    ip=ntuple(c->plan_ifft!(grids[c]; flags=FFTW.ESTIMATE),3)
    EwaldWorkspace{order}(nc,cellof,head,next,cellpairs,st,grids,fp,ip,zeros(3,n),zeros(3,n))
end

@inline function real_cell_field!(E,q,mu,ws::EwaldWorkspace;box,alpha,rcut)
    fill!(E,0); nc=ws.nc; n=size(q,2)
    for i in 1:n
        ci1=mod(floor(Int,q[1,i]/box[1]*nc[1]),nc[1])+1
        ci2=mod(floor(Int,q[2,i]/box[2]*nc[2]),nc[2])+1
        ci3=mod(floor(Int,q[3,i]/box[3]*nc[3]),nc[3])+1
        for dx in -1:1, dy in -1:1, dz in -1:1
            c1=mod(ci1+dx-1,nc[1])+1; c2=mod(ci2+dy-1,nc[2])+1; c3=mod(ci3+dz-1,nc[3])+1
            if minimum(nc)<3
                duplicate=false
                for px in -1:1, py in -1:1, pz in -1:1
                    isless((px,py,pz),(dx,dy,dz)) || continue
                    if mod(ci1+px-1,nc[1])+1==c1 && mod(ci2+py-1,nc[2])+1==c2 && mod(ci3+pz-1,nc[3])+1==c3
                        duplicate=true; break
                    end
                end
                duplicate && continue
            end
            cid=c1+nc[1]*(c2-1)+nc[1]*nc[2]*(c3-1)
            j=ws.head[cid]
            while j!=0
                if i!=j
                    x=q[1,i]-q[1,j]; y=q[2,i]-q[2,j]; z=q[3,i]-q[3,j]
                    x-=box[1]*round(x/box[1]); y-=box[2]*round(y/box[2]); z-=box[3]*round(z/box[3])
                    r2=x*x+y*y+z*z
                    if 0<r2<=rcut^2
                        r=sqrt(r2); er=erfc_std(alpha*r); ex=exp(-(alpha*r)^2)
                        ci=er/r^3+2alpha/SQRTPI*ex/r^2
                        cr=3er/r^5+(6alpha/r^4+4alpha^3/r^2)/SQRTPI*ex
                        dm=x*mu[1,j]+y*mu[2,j]+z*mu[3,j]
                        E[1,i]+=cr*x*dm-ci*mu[1,j]; E[2,i]+=cr*y*dm-ci*mu[2,j]; E[3,i]+=cr*z*dm-ci*mu[3,j]
                    end
                end
                j=ws.next[j]
            end
        end
    end
    E
end

@inline function real_cell_field_pairs!(E,q,mu,ws::EwaldWorkspace;box,alpha,rcut)
    fill!(E,0); rc2=rcut^2
    for (c,t) in ws.cellpairs
        i=ws.head[c]
        while i!=0
            j=ws.head[t]
            while j!=0
                if c!=t || i<j
                    x=q[1,i]-q[1,j]; y=q[2,i]-q[2,j]; z=q[3,i]-q[3,j]
                    x-=box[1]*round(x/box[1]); y-=box[2]*round(y/box[2]); z-=box[3]*round(z/box[3])
                    r2=x*x+y*y+z*z
                    if 0<r2<=rc2
                        r=sqrt(r2); er=erfc_std(alpha*r); ex=exp(-(alpha*r)^2)
                        ci=er/r^3+2alpha/SQRTPI*ex/r^2
                        cr=3er/r^5+(6alpha/r^4+4alpha^3/r^2)*ex/SQRTPI
                        dmi=x*mu[1,i]+y*mu[2,i]+z*mu[3,i]
                        dmj=x*mu[1,j]+y*mu[2,j]+z*mu[3,j]
                        E[1,i]+=cr*x*dmj-ci*mu[1,j]; E[2,i]+=cr*y*dmj-ci*mu[2,j]; E[3,i]+=cr*z*dmj-ci*mu[3,j]
                        E[1,j]+=cr*x*dmi-ci*mu[1,i]; E[2,j]+=cr*y*dmi-ci*mu[2,i]; E[3,j]+=cr*z*dmi-ci*mu[3,i]
                    end
                end
                j=ws.next[j]
            end
            i=ws.next[i]
        end
    end
    E
end

@inline function reciprocal_fft!(out,q,mu,ws::EwaldWorkspace{P};box,alpha,kmax) where P
    fill!(ws.rho[1],0); fill!(ws.rho[2],0); fill!(ws.rho[3],0)
    for a in axes(q,2)
        st=ws.stencils[a]
        @inbounds for i in 1:P,j in 1:P,k in 1:P,c in 1:3
            ws.rho[c][st.nodes[1][i],st.nodes[2][j],st.nodes[3][k]] += mu[c,a]*st.weights[1][i]*st.weights[2][j]*st.weights[3][k]
        end
    end
    for c in 1:3; mul!(ws.rho[c],ws.fftplans[c],ws.rho[c]); end
    nxg,nyg,nzg=size(ws.rho[1]); V=prod(box)
    for i in 1:nxg,j in 1:nyg,k in 1:nzg
        nx=i<=nxg÷2 ? i-1 : i-1-nxg; ny=j<=nyg÷2 ? j-1 : j-1-nyg; nz=k<=nzg÷2 ? k-1 : k-1-nzg
        ((iseven(nxg)&&abs(nx)==nxg÷2)||(iseven(nyg)&&abs(ny)==nyg÷2)||(iseven(nzg)&&abs(nz)==nzg÷2)) && (ws.rho[1][i,j,k]=0;ws.rho[2][i,j,k]=0;ws.rho[3][i,j,k]=0;continue)
        if nx==0&&ny==0&&nz==0 || max(abs(nx),abs(ny),abs(nz))>kmax
            ws.rho[1][i,j,k]=0;ws.rho[2][i,j,k]=0;ws.rho[3][i,j,k]=0; continue
        end
        kx=2pi*nx/box[1]; ky=2pi*ny/box[2]; kz=2pi*nz/box[3]; k2=kx*kx+ky*ky+kz*kz
        win=(sinc(nx/nxg)*sinc(ny/nyg)*sinc(nz/nzg))^P
        coeff=-4pi/V*exp(-k2/(4alpha^2))/k2*(nxg*nyg*nzg)/win^2
        dotk=kx*ws.rho[1][i,j,k]+ky*ws.rho[2][i,j,k]+kz*ws.rho[3][i,j,k]
        ws.rho[1][i,j,k]=coeff*kx*dotk; ws.rho[2][i,j,k]=coeff*ky*dotk; ws.rho[3][i,j,k]=coeff*kz*dotk
    end
    for c in 1:3; mul!(ws.rho[c],ws.ifftplans[c],ws.rho[c]); end
    fill!(out,0)
    for a in axes(q,2)
        st=ws.stencils[a]
        @inbounds for i in 1:P,j in 1:P,k in 1:P,c in 1:3
            out[c,a]+=real(ws.rho[c][st.nodes[1][i],st.nodes[2][j],st.nodes[3][k]])*st.weights[1][i]*st.weights[2][j]*st.weights[3][k]
        end
    end
    out
end

function optimized_field(q,mu,ws;box,alpha,rcut,kmax,vacuum=true)
    real_cell_field_pairs!(ws.realbuf,q,mu,ws;box,alpha,rcut)
    reciprocal_fft!(ws.out,q,mu,ws;box,alpha,kmax)
    ws.out .+= ws.realbuf
    ws.out .+= (4alpha^3/(3SQRTPI)).*mu
    vacuum && (ws.out .-= (4pi/(3prod(box))).*sum(mu,dims=2))
    ws.out
end

function reciprocal_fft(q,mu;box,dims,alpha,kmax=typemax(Int),order=4)
    ws=make_workspace(q;box,rcut=minimum(box)/2,dims,order)
    reciprocal_fft!(ws.out,q,mu,ws;box,alpha,kmax)
end
function real_cell_field(q,mu;box,alpha,rcut)
    ws=make_workspace(q;box,rcut,dims=(4,4,4),order=4)
    real_cell_field!(ws.out,q,mu,ws;box,alpha,rcut)
end

function full_field(q,mu;box,alpha,rcut,kmax,dims,order=4,vacuum=true)
    ws=make_workspace(q;box,rcut,dims,order)
    optimized_field(q,mu,ws;box,alpha,rcut,kmax,vacuum)
end

relative(x,y)=norm(x-y)/max(norm(y),eps())
function fixture(n,box;seed=41)
    rng=MersenneTwister(seed); rand(rng,3,n).*box,randn(rng,3,n)
end
function compare_fixture(n)
    box=[12.0,12.0,12.0]; q,mu=fixture(n,box;seed=100+n)
    for (alpha,rcut,kmax) in ((0.40,5.0,8),(0.55,5.5,9),(0.70,5.9,12))
        ref=direct_ewald(q,mu;box,alpha,rcut,kmax)
        for order in (4,6), side in (12,18,24)
            got=full_field(q,mu;box,alpha,rcut,kmax,dims=(side,side,side),order)
            println("COMPARE N=$n p=$order grid=$side alpha=$alpha rcut=$rcut kmax=$kmax field_rel=$(relative(got,ref)) energy_rel=$(abs(dot(mu,got)-dot(mu,ref))/max(abs(dot(mu,ref)),eps()))")
        end
    end
end
function symmetry_test()
    box=[12.0,12,12]; q,x=fixture(32,box;seed=212); _,y=fixture(32,box;seed=213)
    for p in (4,6)
        fx=full_field(q,x;box,alpha=.55,rcut=5.5,kmax=9,dims=(24,24,24),order=p)
        fy=full_field(q,y;box,alpha=.55,rcut=5.5,kmax=9,dims=(24,24,24),order=p)
        println("SYMMETRY p=$p bilinear_rel=$(abs(dot(x,fy)-dot(y,fx))/max(abs(dot(x,fy)),abs(dot(y,fx)),eps()))")
    end
end
function scaling()
    for n in (1000,10000,100000)
        box=fill((n/0.8)^(1/3),3); q,mu=fixture(n,box;seed=300+n); side=max(16,2^ceil(Int,log2(ceil(Int,n^(1/3)*1.2))))
        dims=(side,side,side); f=full_field(q,mu;box,alpha=.55,rcut=min(8.,minimum(box)/2-0.01),kmax=6,dims,order=4)
        elapsed=@elapsed f=full_field(q,mu;box,alpha=.55,rcut=min(8.,minimum(box)/2-0.01),kmax=6,dims,order=4)
        println("SCALE N=$n grid=$side matvec_s=$elapsed checksum=$(sum(f))")
    end
end
function pcg_fixture()
    n=1000; box=fill((n/0.8)^(1/3),3); q,b=fixture(n,box;seed=501)
    dims=(16,16,16); alpha_site=fill(1e-5,n); rcut=min(8.,minimum(box)/2-0.01)
    apply(v)=vec(reshape(v,3,n)./reshape(alpha_site,1,n) .-
        full_field(q,reshape(v,3,n);box,alpha=.55,rcut,kmax=6,dims,order=4))
    x=zeros(3n); r=copy(vec(b)); p=copy(r); rr=dot(r,r); it=0
    for k in 1:20
        Ap=apply(p); den=dot(p,Ap); den>0 || break
        step=rr/den; x .+= step.*p; r .-= step.*Ap; it=k
        rrnew=dot(r,r); sqrt(rrnew)/norm(vec(b))<1e-8 && break
        p .= r .+ (rrnew/rr).*p; rr=rrnew
    end
    println("PCG N=$n iterations=$it relative_residual=$(norm(apply(x)-vec(b))/norm(vec(b)))")
end

function performance_experiment()
    BLAS.set_num_threads(1); FFTW.set_num_threads(1)
    box=[12.0,12.0,12.0]; alpha=.55; rcut=5.5; dims=(24,24,24); kmax=9; order=4
    for n in (1000,10000,100000)
        boxn=fill((n/0.8)^(1/3),3); rcn=min(rcut,minimum(boxn)/2-0.01)
        side=max(16,2^ceil(Int,log2(ceil(Int,n^(1/3)*1.2)))); dimsn=(side,side,side)
        q,mu=fixture(n,boxn;seed=700+n)
        ws=make_workspace(q;box=boxn,rcut=rcn,dims=dimsn,order)
        rafter()=real_cell_field_pairs!(ws.realbuf,q,mu,ws;box=boxn,alpha,rcut=rcn)
        fafter()=reciprocal_fft!(ws.out,q,mu,ws;box=boxn,alpha,kmax)
        rafter(); fafter()
        ar=@allocated rafter(); af=@allocated fafter()
        realtimes=[@elapsed(rafter()) for _ in 1:3]; ffttimes=[@elapsed(fafter()) for _ in 1:3]
        if n<=10000
            rbefore()=real_cell_field_before(q,mu;box=boxn,alpha,rcut=rcn)
            fbefore()=reciprocal_fft_before(q,mu;box=boxn,dims=dimsn,alpha,kmax,order)
            rb=@allocated rbefore(); fb=@allocated fbefore()
            rbt=@elapsed rbefore(); fbt=@elapsed fbefore()
            println("PERF N=$n dims=$dimsn real_before_alloc=$rb real_after_alloc=$ar real_before_s=$rbt real_after_median_s=$(median(realtimes)) reciprocal_before_alloc=$fb reciprocal_after_alloc=$af reciprocal_before_s=$fbt reciprocal_after_median_s=$(median(ffttimes))")
        else
            println("PERF N=$n dims=$dimsn real_after_alloc=$ar real_after_median_s=$(median(realtimes)) reciprocal_after_alloc=$af reciprocal_after_median_s=$(median(ffttimes))")
        end
    end
    q,mu=fixture(64,box;seed=764); ws=make_workspace(q;box,rcut,dims,order)
    x=mu; y=randn(MersenneTwister(766),3,64)
    fx=copy(optimized_field(q,x,ws;box,alpha,rcut,kmax)); fy=copy(optimized_field(q,y,ws;box,alpha,rcut,kmax))
    q24,m24=fixture(24,box;seed=724); ws24=make_workspace(q24;box,rcut,dims,order)
    got=optimized_field(q24,m24,ws24;box,alpha,rcut,kmax)
    ref=full_field(q24,m24;box,alpha,rcut,kmax,dims,order)
    println("ACCURACY symmetry_rel=$(abs(dot(x,fy)-dot(y,fx))/max(abs(dot(x,fy)),abs(dot(y,fx)),eps())) direct_n24_rel=$(relative(got,ref))")
    println("PCG scaling intentionally not launched: this experimental operator is O(N^2) per matvec and the current fixture fixes iteration count at 20; N=1e5 implies about 5e9 pair visits per matvec before convergence checks. Use subprocess RSS/preflight before such runs.")
end
function main()
    BLAS.set_num_threads(1); FFTW.set_num_threads(1)
    println("JULIA $(VERSION) FFTW $(pkgversion(FFTW)) threads=$(Threads.nthreads())")
    symmetry_test(); compare_fixture(32); compare_fixture(64); pcg_fixture(); scaling()
end

if abspath(PROGRAM_FILE)==@__FILE__
    get(ENV,"EWALD_PERF","0")=="1" ? performance_experiment() : main()
end
