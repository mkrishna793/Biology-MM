using FoundationCore
using Test

const NVEWater = FoundationCore.WaterBox

@testset "NVE diagnostics are independent of frame sampling" begin
    w = NVEWater.tip3p_water(8; seed = 11)
    tr = NVEWater.WaterTrajectory(w, NVEWater.water_field(w); seed = 11)
    NVEWater.relax!(tr; max_steps = 100)
    NVEWater.rethermalize!(tr)
    q0, p0 = copy(tr.st.q), copy(tr.st.p)
    dense = NVEWater.run_nve!(tr, 100; dt = 0.001, sample_every = 1)
    sparse = NVEWater.run_nve!(tr, 100; dt = 0.001, sample_every = 100)
    @test length(dense.samples) == 100
    @test length(sparse.samples) == 1
    for name in fieldnames(NVEWater.NVEDiagnostics)
        @test getfield(dense.diagnostics, name) == getfield(sparse.diagnostics, name)
    end
    @test tr.st.q == q0
    @test tr.st.p == p0
    @test tr.step == 0
    @test tr.time_ps == 0.0
end
