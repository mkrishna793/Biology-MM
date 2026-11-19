using FoundationCore
using Test

const WF = FoundationCore.Workflow
const TY = FoundationCore.Types
const DY = FoundationCore.Dynamics

@testset "checkpoint previous-generation recovery" begin
    mktempdir() do dir
        box = TY.OrthorhombicBox((10.0, 10.0, 10.0))
        manifest = WF.build_manifest(dt=0.01, n_atoms=1, box=box)
        path = joinpath(dir, "state.json")
        function checkpoint(step)
            st = DY.State(fill(Float64(step), 3, 1), zeros(3,1))
            WF.make_checkpoint(stage=:produce, step=step, time=step*0.01, manifest=manifest, state=st, box=box)
        end
        WF.write_checkpoint_atomic(path, checkpoint(1))
        WF.write_checkpoint_atomic(path, checkpoint(2))
        @test isfile(path * ".prev")
        write(path, "broken")
        @test WF.read_checkpoint(path).step == 1
        next = DY.State(fill(3.0, 3, 1), zeros(3,1))
        WF.write_checkpoint_atomic(path, WF.make_checkpoint(stage=:produce, step=3, time=0.03,
            manifest=manifest, state=next, box=box))
        @test WF.read_checkpoint(path).step == 3
        @test_throws WF.CheckpointMismatchError WF.read_checkpoint(path; expected_manifest_digest="wrong")
        write(path, "broken")
        @test_throws WF.CheckpointVersionError WF.read_checkpoint(path; expected_version=999)
    end
end
