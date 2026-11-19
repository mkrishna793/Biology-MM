using Test, LinearAlgebra, Random
using FoundationCore

if !isdefined(FoundationCore, :Settle)
    Base.include(FoundationCore, joinpath(@__DIR__, "../src/Settle.jl"))
end
const S=FoundationCore.Settle
const C=FoundationCore.Constraints
const O=FoundationCore.Types

function water_fixture()
    d=0.9572; a=deg2rad(104.52); hh=2d*sin(a/2)
    q=zeros(3,3); q[:,2]=[d,0,0]; q[:,3]=[d*cos(a),d*sin(a),0]
    m=[15.9994,1.008,1.008]
    cs=C.CoupledConstraintSet(3,[(1,2,d),(1,3,d),(2,3,hh)])
    q,m,cs
end

function test_settle_geometry_conservation_and_reference()
    q,m,cs=water_fixture(); rng=MersenneTwister(87)
    for scale in (1e-4,1e-3,2e-3)
        for rep in 1:20
            pred=q .+ scale.*randn(rng,3,3)
            com0=vec(sum(pred.*reshape(m,1,:),dims=2))./sum(m)
            # Coupled Newton result is an independent constraint-surface reference.
            ref=copy(pred); C.project_positions!(ref,m,cs;tolerance=1e-13)
            out=copy(pred); residual=S.settle_positions!(q,out,m,cs)
            @test maximum(abs,C.constraint_residuals(cs,out)) < 2e-11
            @test norm(vec(sum(out.*reshape(m,1,:),dims=2))./sum(m)-com0)<2e-13
            # Both maps land on the target geometry, but need not choose the same orientation.
            @test maximum(abs,C.constraint_residuals(cs,ref)) < 2e-11
            @test isfinite(residual)
            @test maximum(abs,norm(out[:,j]-out[:,i])-d for (i,j,d) in ((1,2,0.9572),(1,3,0.9572),(2,3,2*0.9572*sin(deg2rad(104.52)/2)))) < 2e-11
        end
    end
end

function test_settle_identity_rotations_and_multiple_waters()
    q,m,cs=water_fixture()
    out=copy(q); @test S.settle_positions!(q,out,m,cs)<1e-14
    for axis in ([1.0,0,0],[0.0,1,0],[1.0,2,3])
        u=axis/norm(axis); θ=0.71
        K=[0.0 -u[3] u[2]; u[3] 0.0 -u[1]; -u[2] u[1] 0.0]
        R=I*cos(θ)+(1-cos(θ))*(u*u')+sin(θ)*K
        qr=R*q .+ reshape([0.3,-0.2,0.7],3,1)
        pred=qr .+ 2e-4.*randn(MersenneTwister(11),3,3)
        @test S.settle_positions!(qr,pred,m,cs)<1e-10
        @test maximum(abs,C.constraint_residuals(cs,pred))<2e-11
    end
    q2=zeros(3,6); q2[:,1:3]=q; q2[:,4:6]=q .+ reshape([2.0,-1.0,0.5],3,1)
    m2=vcat(m,m); cs2=C.CoupledConstraintSet(6,[(1,2,0.9572),(1,3,0.9572),(2,3,2*0.9572*sin(deg2rad(104.52)/2)),(4,5,0.9572),(4,6,0.9572),(5,6,2*0.9572*sin(deg2rad(104.52)/2))])
    pred2=q2 .+ 1e-4.*randn(MersenneTwister(12),3,6)
    @test S.settle_positions!(q2,pred2,m2,cs2)<1e-9
end

function test_settle_momentum_angular_momentum_and_tangent()
    q,m,cs=water_fixture(); rng=MersenneTwister(91)
    p=randn(rng,3,3); P0=vec(sum(p,dims=2)); com=vec(sum(q.*reshape(m,1,:),dims=2))./sum(m)
    L0=sum((cross(q[:,i]-com,p[:,i]) for i in 1:3))
    v=p./reshape(m,1,:)
    λ=S.project_velocities!(q,v,m,cs)
    pnew=v.*reshape(m,1,:)
    @test norm(vec(sum(pnew,dims=2))-P0)<1e-12
    @test norm(sum((cross(q[:,i]-com,pnew[:,i]) for i in 1:3))-L0)<1e-12
    @test maximum(abs,C.tangent_residuals(cs,q,pnew,m))<1e-11
    @test length(λ)==1 && length(λ[1])==3
end

function rattle_step(q,p,m,cs,dt; algorithm=:settle)
    qold=copy(q); qu=q .+ dt.*(p./reshape(m,1,:)); qnew=copy(qu)
    if algorithm===:settle
        S.settle_positions!(qold,qnew,m,cs)
        λ=S.position_multipliers(qold,qu,qnew,m,cs,dt)
        dp=zeros(size(p)); virial=0.0
        for (compidx,comp) in enumerate(cs.components), (k,(a,b,d)) in enumerate(comp.edges)
            ids=comp.atoms; i,j=ids[a],ids[b]; r=qold[:,j]-qold[:,i]
            impulse=2λ[compidx][k].*r
            dp[:,i] .+= impulse; dp[:,j] .-= impulse
            virial -= 2λ[compidx][k]*dot(r,r)
        end
        dp_expected=reshape(m,1,:).*(qnew-qu)./dt
        @test norm(dp-dp_expected) < 1e-9*norm(dp_expected)+1e-11
        directvirial=sum(dot(qold[:,i],dp_expected[:,i]) for i in axes(qold,2))
        directtensor=sum(qold[:,i]*dp_expected[:,i]' for i in axes(qold,2))
        edgetensor=zeros(3,3)
        for (compidx,comp) in enumerate(cs.components), (k,(a,b,d)) in enumerate(comp.edges)
            i,j=comp.atoms[a],comp.atoms[b]; r=qold[:,j]-qold[:,i]
            edgetensor .+= -2λ[compidx][k].*(r*r')
        end
        @test isapprox(virial,directvirial;atol=1e-11,rtol=1e-9)
        @test norm(edgetensor-directtensor) < 1e-9*norm(directtensor)+1e-11
        p .+= dp_expected
    else
        C.project_positions!(qnew,m,cs;tolerance=1e-13)
        p .+= reshape(m,1,:).*(qnew-qu)./dt
    end
    v=p./reshape(m,1,:)
    if algorithm===:settle
        λv=S.project_velocities!(qnew,v,m,cs)
        p .= v.*reshape(m,1,:)
    else
        C.project_velocities!(qnew,p,m,cs;tolerance=1e-13)
        v=p./reshape(m,1,:)
    end
    return qnew,v.*reshape(m,1,:)
end

function free_rotor_error(q,p,m,cs,dt,n,algorithm)
    qn,pn=copy(q),copy(p)
    for _ in 1:n; qn,pn=rattle_step(qn,pn,m,cs,dt;algorithm); end
    qn,pn
end

function test_settle_verlet_reversal_and_dt_order()
    q0,m,cs=water_fixture(); p0=[0.0 0.01 -0.01; 0.0 0.015 -0.015; 0.01 -0.005 -0.005]
    C.project_velocities!(q0,p0,m,cs;tolerance=1e-13)
    for alg in (:settle,:newton)
        dt=0.0005; n=1
        qf,pf=free_rotor_error(q0,p0,m,cs,dt,n,alg)
        qb,pb=free_rotor_error(qf,-pf,m,cs,dt,n,alg)
        @test norm(qb-q0)<1e-9
        @test norm(pb+p0)<1e-9
        @test maximum(abs,C.constraint_residuals(cs,qf))<2e-11
        @test norm(vec(sum(pf,dims=2))-vec(sum(p0,dims=2)))<2e-11
        com=vec(sum(q0.*reshape(m,1,:),dims=2))./sum(m)
        L0=sum((cross(q0[:,i]-com,p0[:,i]) for i in 1:3))
        Lf=sum((cross(qf[:,i]-com,pf[:,i]) for i in 1:3))
        @test norm(Lf-L0)<1e-9
    end
    # Fixed total duration and a fine SETTLE reference expose the
    # convergence order instead of only checking that errors are small.
    p_ref=[0.0 1.0 -1.0; 0.0 1.5 -1.5; 1.0 -0.5 -0.5]
    C.project_velocities!(q0,p_ref,m,cs;tolerance=1e-13)
    qforward,pforward=free_rotor_error(q0,p_ref,m,cs,0.004,20,:settle)
    qback,pback=free_rotor_error(qforward,-pforward,m,cs,0.004,20,:settle)
    @test norm(qback-q0)<1e-9
    @test norm(pback+p_ref)<1e-9
    Lreference=sum(cross(q0[:,i],p_ref[:,i]) for i in 1:3)
    Lforward=sum(cross(qforward[:,i],pforward[:,i]) for i in 1:3)
    @test norm(Lforward-Lreference)<1e-9
    T=0.08; href=0.00025
    qref,pref=free_rotor_error(q0,p_ref,m,cs,href,round(Int,T/href),:settle)
    errs=Float64[]
    for h in (0.004,0.002,0.001)
        qh,ph=free_rotor_error(q0,p_ref,m,cs,h,round(Int,T/h),:settle)
        push!(errs,norm(qh-qref)+norm(ph-pref))
    end
    println("  fixed-time SETTLE errors=$(errs), ratios=$(errs[1]/errs[2]), $(errs[2]/errs[3])")
    @test errs[1] > errs[2] > errs[3]
    @test 3.2 < errs[1]/errs[2] < 4.8
    @test 3.2 < errs[2]/errs[3] < 4.8
end

function test_settle_refusals()
    q,m,cs=water_fixture(); pred=copy(q); pred[1,2]+=100
    @test_throws ArgumentError S.settle_positions!(q,pred,m,cs)
end

function test_settle_pbc_and_waterbox_integration()
    q0, m, cs = water_fixture()
    L = (5.0, 5.0, 5.0)
    obox = FoundationCore.Types.OrthorhombicBox(L)
    H = [5.0 0.8 0.3; 0.0 4.8 0.5; 0.0 0.0 5.2]
    tbox = FoundationCore.Types.TriclinicBox(H)

    for box in (obox, tbox)
        # Shift H1 across +x boundary and H2 across -y boundary so the water straddles PBC
        q_pbc = copy(q0)
        shift1 = box isa FoundationCore.Types.OrthorhombicBox ? [L[1], 0.0, 0.0] : H[:, 1]
        shift2 = box isa FoundationCore.Types.OrthorhombicBox ? [0.0, -L[2], 0.0] : -H[:, 2]
        q_pbc[:, 2] .+= shift1
        q_pbc[:, 3] .+= shift2

        p0 = [0.01 -0.02 0.01; -0.01 0.03 -0.02; 0.02 -0.01 -0.01]
        S.project_momenta!(q_pbc, p0, m, cs; box = box)
        dt = 0.002
        pred = q_pbc .+ dt .* (p0 ./ reshape(m, 1, :))
        q_corr = copy(pred)
        S.settle_positions!(q_pbc, q_corr, m, cs; box = box)
        λ = S.position_multipliers(q_pbc, pred, q_corr, m, cs, dt; box = box)
        @test length(λ) == 1 && all(isfinite, λ[1])
        @test maximum(abs, C.constraint_residuals(cs, q_corr, box)) < 1e-12

        # Check settle_step! with box
        q_step = copy(pred)
        p_step = copy(p0)
        S.settle_step!(q_pbc, q_step, p_step, m, cs, dt; box = box)
        @test maximum(abs, C.constraint_residuals(cs, q_step, box)) < 1e-12
    end

    # Check Dynamics.velocity_verlet_step! and RespaIntegrator with CoupledConstraintSet
    f_harm = FoundationCore.Factors.HarmonicBond(1, 2, 0.5, 0.9572)
    ctx_o = FoundationCore.Factors.Context(obox, FoundationCore.Types.REDUCED, 3)
    sys = FoundationCore.Dynamics.HarmonicSystem([f_harm], m, ctx_o)
    p_dyn = [0.0 0.05 -0.05; 0.02 -0.01 -0.01; -0.01 0.02 -0.01]
    st = FoundationCore.Dynamics.State(copy(q0), copy(p_dyn))
    FoundationCore.Dynamics.rattle_project!(st, sys, cs)
    pos_err, vel_err = FoundationCore.Dynamics.rattle_residuals(st, sys, cs)
    @test pos_err < 1e-12
    @test vel_err < 1e-12
    for _ in 1:10
        FoundationCore.Dynamics.velocity_verlet_step!(st, sys, 0.001; constraints = cs)
    end
    pos_err2, vel_err2 = FoundationCore.Dynamics.rattle_residuals(st, sys, cs)
    @test pos_err2 < 1e-11
    @test vel_err2 < 1e-11

    respa = FoundationCore.Respa.RespaIntegrator(sys, [1])
    st_r = FoundationCore.Dynamics.State(copy(q0), copy(p_dyn))
    FoundationCore.Dynamics.rattle_project!(st_r, sys, cs)
    for _ in 1:5
        FoundationCore.Respa.step!(st_r, respa, 0.002; m = 2, constraints = cs)
    end
    pos_err_r, vel_err_r = FoundationCore.Dynamics.rattle_residuals(st_r, sys, cs)
    @test pos_err_r < 1e-11
    @test vel_err_r < 1e-11

    # Check WaterBox constraint_solver = :settle vs :principal_axis
    isdefined(FoundationCore, :WaterBox) ||
        Base.include(FoundationCore, joinpath(@__DIR__, "..", "src", "WaterBox.jl"))
    WB = FoundationCore.WaterBox
    w = WB.tip3p_water(8; seed = 42)
    field = WB.water_field(w)
    tr1 = WB.WaterTrajectory(w, field; temperature = 300.0, seed = 42)
    tr2 = WB.WaterTrajectory(w, field; temperature = 300.0, seed = 42)
    res1 = WB.run_nve!(tr1, 5; dt = 0.001, sample_every = 1, constraint_solver = :principal_axis)
    res2 = WB.run_nve!(tr2, 5; dt = 0.001, sample_every = 1, constraint_solver = :settle)
    @test res2.diagnostics.max_bond_error < 1e-12
    @test isapprox(res1.samples[end].E, res2.samples[end].E; rtol = 1e-4)
end

@testset "analytic SETTLE" begin
    test_settle_geometry_conservation_and_reference()
    test_settle_identity_rotations_and_multiple_waters()
    test_settle_momentum_angular_momentum_and_tangent()
    test_settle_verlet_reversal_and_dt_order()
    test_settle_refusals()
    test_settle_pbc_and_waterbox_integration()
end
