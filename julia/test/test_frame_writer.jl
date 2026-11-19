using FoundationCore
using Test
using LinearAlgebra

const DY = FoundationCore.Dynamics
const FA = FoundationCore.Factors
const TY = FoundationCore.Types
const WF = FoundationCore.Workflow

mktempdir() do dir
    q0 = [0.0 1.0; 0.0 0.0; 0.0 0.0]
    p0 = [0.0 0.0; 0.0 0.1; 0.0 -0.1]
    ctx = FA.Context(TY.REDUCED, 2)
    sys = DY.HarmonicSystem([FA.HarmonicBond(1, 2, 1.0, 1.0)], [1.0, 1.0], ctx)
    plain = DY.State(q0, p0)
    saved = DY.State(q0, p0)
    reference = DY.State(q0, p0)
    xyz = joinpath(dir, "frames.xyz")
    writer = DY.FrameWriter(xyz, 2; labels=["O", "H"])
    box = [3.0, 4.0, 5.0]
    DY.run_trajectory(saved, sys, 0.01, 10; frame_writer=writer, frame_every=3,
                      frame_box=box, compensated=false)
    close(writer)
    DY.run_trajectory(plain, sys, 0.01, 10; compensated=false)
    snapshots = Matrix{Float64}[]
    for step in 1:9
        DY.velocity_verlet_step!(reference, sys, 0.01)
        step % 3 == 0 && push!(snapshots, copy(reference.q))
    end
    @test saved.q == plain.q
    frames = DY.read_xyz_frames(xyz)
    @test length(frames) == 3
    @test frames[1][1] == ["O", "H"]
    for i in eachindex(frames)
        @test reinterpret(UInt64, vec(frames[i][2])) == reinterpret(UInt64, vec(snapshots[i]))
    end
    rows = readlines(xyz * ".jsonl")
    @test length(rows) == 3
    @test occursin("\"step\":3", rows[1])
    @test occursin("\"time\":0.03", rows[1])
    @test occursin("\"box\":[3.0,4.0,5.0]", rows[1])
    @test occursin("\"step\":9", rows[3])

    # Workflow.TrajectoryWriter + read_trajectory: orthorhombic, triclinic, and NoBox metadata
    tri_path = joinpath(dir, "workflow_tri.xyz")
    H1 = [10.0 1.5 0.0; 0.0 10.0 0.5; 0.0 0.0 10.0]
    H2 = [10.5 1.5 0.0; 0.0 10.2 0.5; 0.0 0.0 10.1]
    tw = WF.TrajectoryWriter(tri_path, 2; labels=["O", "H"], manifest_digest="abc123")
    WF.write_trajectory_frame!(tw, q0, TY.TriclinicBox(H1), 1, 0.01)
    WF.write_trajectory_frame!(tw, q0 .+ 0.2, TY.TriclinicBox(H2), 2, 0.02)
    close(tw)
    tdata = WF.read_trajectory(tri_path)
    @test length(tdata.frames) == 2
    @test tdata.steps == [1, 2]
    @test tdata.times == [0.01, 0.02]
    @test tdata.boxes[1] isa TY.TriclinicBox
    @test tdata.boxes[1].H == H1
    @test tdata.boxes[2].H == H2
    @test tdata.manifest_digests == ["abc123", "abc123"]
    @test reinterpret(UInt64, vec(tdata.frames[1])) == reinterpret(UInt64, vec(q0))

    # NoBox round-trip
    nb_path = joinpath(dir, "workflow_nobox.xyz")
    tw_nb = WF.TrajectoryWriter(nb_path, 2; labels=["O", "H"])
    WF.write_trajectory_frame!(tw_nb, q0, TY.NoBox(), 1, 0.01)
    WF.write_trajectory_frame!(tw_nb, q0 .+ 0.1, TY.NoBox(), 2, 0.02)
    close(tw_nb)
    nb_data = WF.read_trajectory(nb_path)
    @test all(b -> b isa TY.NoBox, nb_data.boxes)

    # Corrupt sidecar length mismatch must throw TrajectoryError
    open(nb_path * ".jsonl", "w") do io
        println(io, "{\"step\":1,\"time\":0.01,\"box_kind\":\"none\"}")
    end
    @test_throws WF.TrajectoryError WF.read_trajectory(nb_path)
end
println("frame_writer: XYZ round-trip, sidecar metadata/box (ortho/triclinic/none), stride, and bit-identical dynamics PASS")
