using FoundationCore
using FoundationCore.Types
using FoundationCore.Factors
using FoundationCore.Dynamics
using FoundationCore.StreamedLJ
using LinearAlgebra
using Test

const T = FoundationCore.Types
const FA = FoundationCore.Factors
const DY = FoundationCore.Dynamics
const SL = FoundationCore.StreamedLJ
const RP = isdefined(FoundationCore, :Respa) ? FoundationCore.Respa : include("../src/Respa.jl")

mutable struct CountedBond <: FA.AbstractFactor
    bond::FA.HarmonicBond
    calls::Base.RefValue{Int}
end
FA.evaluate!(F::AbstractMatrix{Float64}, f::CountedBond, q::AbstractMatrix{Float64}, ctx::FA.Context) =
    (f.calls[] += 1; FA.evaluate!(F, f.bond, q, ctx))

function oscillator(kfast=1.0, kslow=0.5)
    q = zeros(3, 2); q[1, 1] = -0.5; q[1, 2] = 0.5
    p = zeros(3, 2)
    fs = [FA.HarmonicBond(1, 2, kfast, 0.0), FA.HarmonicBond(1, 2, kslow, 0.0)]
    s = DY.HarmonicSystem(fs, [1.0, 1.0], FA.Context(T.REDUCED, 2))
    DY.State(q, p), s
end

@testset "nested reversible RESPA" begin
    st, sys = oscillator()
    ref = copy(st); verlet = RP.RespaIntegrator(sys, [1])
    DY.velocity_verlet_step!(ref, sys, 0.01)
    RP.step!(st, verlet, 0.01; m=1)
    @test st.q ≈ ref.q atol=1e-14
    @test st.p ≈ ref.p atol=1e-14

    # Composition through the factor protocol: bonded fast force plus streamed
    # periodic LJ slow force, compared with the full additive force evaluation.
    box = T.OrthorhombicBox([12.0, 12.0, 12.0])
    qmix = zeros(3, 3); qmix[1, :] .= [1.0, 2.0, 4.0]
    bond = FA.HarmonicBond(1, 2, 8.0, 0.9)
    lj = SL.StreamedLJFactor(qmix, fill(0.2, 3), fill(1.0, 3), box; rcut=4.0)
    mixsys = DY.HarmonicSystem([bond, lj], ones(3), FA.Context(box, T.REDUCED, 3))
    mixstate = DY.State(qmix, [0.0 0.0 0.0; 0.1 -0.1 0.0; 0.0 0.0 0.0])
    mixref = copy(mixstate)
    DY.velocity_verlet_step!(mixref, mixsys, 0.001)
    RP.step!(mixstate, RP.RespaIntegrator(mixsys, [1]), 0.001; m=1)
    @test mixstate.q ≈ mixref.q atol=1e-13
    @test mixstate.p ≈ mixref.p atol=1e-13

    fc, sc = Ref(0), Ref(0)
    q = zeros(3, 2); q[1, 2] = 1.0
    factors = [CountedBond(FA.HarmonicBond(1, 2, 1.0, 0.0), fc),
               CountedBond(FA.HarmonicBond(1, 2, 0.5, 0.0), sc)]
    counted = DY.HarmonicSystem(factors, [1.0, 1.0], FA.Context(T.REDUCED, 2))
    RP.step!(DY.State(q, zeros(3,2)), RP.RespaIntegrator(counted, [1]), 0.01; m=4)
    @test fc[] == 8
    @test sc[] == 2
    # The scalar harmonic model verifies reversibility and error scaling independently.
    st, sys = oscillator()
    r = RP.RespaIntegrator(sys, [1]); q0=copy(st.q); p0=copy(st.p)
    st.p[1,1] = 0.2; st.p[1,2] = -0.2; p0=copy(st.p)
    for _ in 1:500; RP.step!(st, r, 0.01; m=3); end
    for _ in 1:500; RP.step!(st, r, -0.01; m=3); end
    @test st.q ≈ q0 atol=1e-11
    @test st.p ≈ p0 atol=1e-11

    function err(dt)
        x, s = oscillator()
        rr = RP.RespaIntegrator(s, [1])
        for _ in 1:round(Int, 1.0/dt); RP.step!(x, rr, dt; m=2); end
        omega = sqrt(3.0)
        exactq = cos(omega)
        exactp = -omega*sin(omega)
        hypot(x.q[1,2]-x.q[1,1]-exactq, x.p[1,2]-x.p[1,1]-exactp)
    end
    @test err(0.01) < 0.3err(0.02)

    # Compare one outer RESPA step with two cached-force full-system Verlet
    # steps at dt/2. Then check trapezoidal slow-force work against the actual
    # slow-potential change over a short trajectory.
    function work_and_energy(dt)
        x, s = oscillator(10.0, 1.0)
        x.p[1,1] = 0.2; x.p[1,2] = -0.2
        rr = RP.RespaIntegrator(s, [1]); E0 = DY.total_energy(s, x.q, x.p)
        max_slow_work_defect = 0.0
        for _ in 1:round(Int, 0.2/dt)
            qbefore = copy(x.q); pbefore = copy(x.p)
            RP.step!(x, rr, dt; m=5)
            # Discrete slow kick work versus its potential-energy change.
            F0 = zeros(3,2); F1 = zeros(3,2)
            u0 = FA.total_energy_forces([s.factors[2]], qbefore, s.ctx; F=F0)[1]
            u1 = FA.total_energy_forces([s.factors[2]], x.q, s.ctx; F=F1)[1]
            slow_work = sum((F0 .+ F1) .* (x.q .- qbefore)) / 2
            max_slow_work_defect = max(max_slow_work_defect, abs(slow_work - (u0-u1)))
        end
        return abs(DY.total_energy(s, x.q, x.p)-E0), max_slow_work_defect
    end
    function matched_work_error(dt, use_respa)
        fast_calls, slow_calls = Ref(0), Ref(0)
        qmatch = zeros(3,2); qmatch[1,1]=-0.5; qmatch[1,2]=0.5
        fs = [CountedBond(FA.HarmonicBond(1,2,10.0,0.0),fast_calls),
              CountedBond(FA.HarmonicBond(1,2,1.0,0.0),slow_calls)]
        s = DY.HarmonicSystem(fs,[1.0,1.0],FA.Context(T.REDUCED,2))
        x = DY.State(qmatch,zeros(3,2)); x.p[1,1]=0.2; x.p[1,2]=-0.2
        E0 = DY.total_energy(s,x.q,x.p)
        fast_calls[]=0; slow_calls[]=0
        if use_respa
            rr=RP.RespaIntegrator(s,[1]); RP.step!(x,rr,dt;m=2)
        else
            for _ in 1:2; DY.velocity_verlet_step!(x,s,dt/2); end
        end
        nf, ns = fast_calls[], slow_calls[]
        return abs(DY.total_energy(s,x.q,x.p)-E0), nf, ns
    end
    respa_defect, respa_fast_calls, respa_slow_calls = matched_work_error(0.004, true)
    verlet_defect, verlet_fast_calls, verlet_slow_calls = matched_work_error(0.004, false)
    @test (respa_fast_calls,respa_slow_calls) == (4,2)
    @test (verlet_fast_calls,verlet_slow_calls) == (3,3)
    @test respa_fast_calls + respa_slow_calls == verlet_fast_calls + verlet_slow_calls == 6
    @test isfinite(respa_defect) && isfinite(verlet_defect)
    println("  matched force work: RESPA energy_defect=$(respa_defect), VV energy_defect=$(verlet_defect); factor evaluations=$(respa_fast_calls+respa_slow_calls) each")
    _, w_coarse = work_and_energy(0.004)
    _, w_fine = work_and_energy(0.002)
    @test w_fine < 1e-12 && w_coarse < 1e-12

    # Impulse resonance from the composed 2x2 map: set ω_fast=1, dt*ω_fast=1.3,
    # dt*ω_slow=1.8, m=2 (h*ω_fast=0.65). Both standalone VV maps are stable,
    # but the RESPA transfer-matrix trace is below -2.
    qres = zeros(3, 2); qres[1, 1] = -0.5; qres[1, 2] = 0.5
    dtres=1.3; wf=1.0; ws=1.8/dtres
    fres = [FA.HarmonicBond(1, 2, wf^2/2, 0.0), FA.HarmonicBond(1, 2, ws^2/2, 0.0)]
    sres = DY.HarmonicSystem(fres, [1.0, 1.0], FA.Context(T.REDUCED, 2))
    unstable = DY.State(qres, zeros(3, 2)); rr = RP.RespaIntegrator(sres, [1])
    @test dtres*wf < 2 && dtres*ws < 2 && (dtres/2)*wf < 2
    # Matrix order is (relative position, relative velocity); each VV map is
    # [[a,h],[-h*w^2*(1-h^2*w^2/4),a]], bracketed by slow half-kick maps.
    function mmul(A,B)
        [sum(A[i,k]*B[k,j] for k in 1:2) for i in 1:2, j in 1:2]
    end
    function vvmap(w,h)
        a=1-h^2*w^2/2
        [a h; -h*w^2*(1-h^2*w^2/4) a]
    end
    slowkick=[1.0 0.0; -dtres*ws^2/2 1.0]
    composed=mmul(slowkick, mmul(mmul(vvmap(wf,dtres/2),vvmap(wf,dtres/2)),slowkick))
    @test abs(tr(composed)) > 2
    for _ in 1:40; RP.step!(unstable, rr, dtres; m=2); end
    @test norm(unstable.q) > 1e3

    # Stable split: bounded total-energy band and actual full-step cost after warmup.
    stable, ss = oscillator(10.0, 0.5); stable.p[1,1]=0.2; stable.p[1,2]=-0.2
    sr = RP.RespaIntegrator(ss, [1]); energies = Float64[]
    E0 = DY.total_energy(ss, stable.q, stable.p)
    for i in 1:5000
        RP.step!(stable, sr, 0.01; m=4)
        i % 20 == 0 && push!(energies, DY.total_energy(ss, stable.q, stable.p))
    end
    band = maximum(abs.(energies .- E0))
    @test band < 0.02
    RP.step!(stable, sr, 0.01; m=4) # warmup
    elapsed = @elapsed for _ in 1:1000; RP.step!(stable, sr, 0.01; m=4); end
    allocated = @allocated for _ in 1:1000; RP.step!(stable, sr, 0.01; m=4); end
    println("  RESPA cost: N=2 threads=$(Threads.nthreads()) m=4 steps=1000 elapsed=$(elapsed)s allocated=$(allocated) B; energy_band=$(band)")

    cst, csys = oscillator()
    @test_throws ArgumentError RP.step!(cst, RP.RespaIntegrator(csys, [1]), 0.01; constraints=:rigid)

    # Reject malformed Cartesian state shapes before any force evaluation or
    # momentum mutation; Dynamics.State itself intentionally permits them.
    badsys = RP.RespaIntegrator(last(oscillator()), [1])
    for badq in (zeros(2, 2), zeros(4, 2))
        bad = DY.State(badq, zeros(size(badq)))
        p_before = copy(bad.p)
        @test_throws ArgumentError RP.step!(bad, badsys, 0.01; m=2)
        @test bad.p == p_before
    end
end
