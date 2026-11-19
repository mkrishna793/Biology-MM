using FoundationCore, LinearAlgebra, Test
const WB = FoundationCore.WaterBox
const D = FoundationCore.Dynamics

@testset "WaterBox explicit neighbor cache" begin
    w = WB.tip3p_water(8; box=(2.0,2.0,2.0), seed=31)
    a = WB.water_field(w; rcut=0.8)
    b = WB.water_field(w; rcut=0.8, neighbor_cache=true, neighbor_cache_skin=0.15)
    @test a.lj.pair_cache === nothing
    @test b.lj.pair_cache !== nothing
    ta = WB.WaterTrajectory(w,a; temperature=30.0, seed=19)
    tb = WB.WaterTrajectory(w,b; temperature=30.0, seed=19)
    for step in 1:12
        WB.run_nve!(ta,1; dt=0.0001,mutate=true)
        WB.run_nve!(tb,1; dt=0.0001,mutate=true)
        @test isapprox(ta.st.q,tb.st.q; atol=2e-13,rtol=2e-13)
        @test isapprox(ta.st.p,tb.st.p; atol=2e-11,rtol=2e-11)
        fresh = WB.water_field(w; rcut=0.8, neighbor_cache=true,neighbor_cache_skin=0.15)
        ff = zeros(size(tb.st.q))
        cached_force = similar(ff)
        D.forces!(fresh.system,tb.st.q,ff)
        D.forces!(b.system,tb.st.q,cached_force)
        @test isapprox(ff,cached_force; atol=2e-9,rtol=2e-10)
    end
    cache = b.lj.pair_cache
    @test cache.n_reuses > 0
    @test cache.n_rebuilds >= 1
    q = copy(tb.st.q)
    q[:,1:3] .+= [0.3,0.0,0.0]
    before = cache.n_rebuilds
    fc = zeros(size(q)); fu = similar(fc)
    D.forces!(b.system,q,fc)
    D.forces!(a.system,q,fu)
    @test cache.n_rebuilds > before
    @test isapprox(fc,fu; atol=2e-9,rtol=2e-10)
end
