Base.include(@__MODULE__, joinpath(@__DIR__, "..", "src", "TrajectoryFormat.jl"))
using .TrajectoryFormat, Test, LinearAlgebra

positions = [0.1 1.2; 0.2 2.3; 0.3 3.4]
box = [10.0 1.5 -0.2; 0.0 9.0 0.7; 0.0 0.0 8.0]
io = IOBuffer()
write_extxyz_frame(io; positions, species=["O", "H"], time=2.5,
                   box_matrix=box, length_unit="angstrom", time_unit="fs")
text = String(take!(io))
@test occursin("Lattice=\"10.0 0.0 0.0 1.5 9.0 0.0 -0.2 0.7 8.0\"", text)
frame = only(read_extxyz_frames(IOBuffer(text)))
@test frame.positions == positions
@test frame.species == ["O", "H"]
@test frame.time == 2.5
@test frame.box_matrix == box
@test frame.pbc == ["T", "T", "T"]
@test frame.length_unit == "angstrom" && frame.time_unit == "fs"

# Multi-frame stream round-trip (periodic + non-periodic frames)
io_multi = IOBuffer()
write_extxyz_frame(io_multi; positions=positions, species=["O", "H"], time=0.0,
                   box_matrix=box, length_unit="nm", time_unit="ps")
write_extxyz_frame(io_multi; positions=positions .+ 0.5, species=["O", "H"], time=1.0,
                   box_matrix=nothing, length_unit="nm", time_unit="ps")
frames_multi = read_extxyz_frames(IOBuffer(String(take!(io_multi))))
@test length(frames_multi) == 2
@test frames_multi[1].box_matrix == box && frames_multi[1].pbc == ["T", "T", "T"]
@test frames_multi[2].box_matrix === nothing && frames_multi[2].pbc == ["F", "F", "F"]
@test frames_multi[2].positions ≈ positions .+ 0.5

@test_throws ArgumentError write_extxyz_frame(IOBuffer(); positions=[0.0 NaN; 0 0; 0 0], species=["H", "H"], time=0, length_unit="nm", time_unit="ps")
@test_throws ArgumentError write_extxyz_frame(IOBuffer(); positions, species=["H"], time=0, length_unit="nm", time_unit="ps")
@test_throws ArgumentError write_extxyz_frame(IOBuffer(); positions, species=["O", "H"], time=Inf, length_unit="nm", time_unit="ps")
@test_throws ArgumentError write_extxyz_frame(IOBuffer(); positions, species=["O", "H"], time=0, length_unit="nm\" pbc=\"F F F", time_unit="ps")
@test_throws ArgumentError write_extxyz_frame(IOBuffer(); positions, species=["O", "H"], time=0, box_matrix=zeros(3, 3), length_unit="nm", time_unit="ps")
@test_throws ArgumentError write_extxyz_frame(IOBuffer(); positions=ones(3, 0), species=String[], time=0, length_unit="nm", time_unit="ps")
for malformed in (
    "0\nProperties=species:S:1:pos:R:3 Time=0 time_unit=\"ps\" length_unit=\"nm\" pbc=\"F F F\"\n",
    "1\nProperties=species:S:1:pos:R:3 Time=NaN time_unit=\"ps\" length_unit=\"nm\" pbc=\"F F F\"\nH 0 0 0\n",
    "1\nProperties=species:S:1:pos:R:3 Time=0 time_unit=\"ps\" length_unit=\"nm\" pbc=\"T T T\"\nH 0 0 0\n",
    "1\nLattice=\"0 0 0 0 0 0 0 0 0\" Properties=species:S:1:pos:R:3 Time=0 time_unit=\"ps\" length_unit=\"nm\" pbc=\"T T T\"\nH 0 0 0\n",
    "1\nProperties=species:S:1:pos:R:3 Time=0 time_unit=\"ps\" length_unit=\"nm\" pbc=\"F F F\"\nH NaN 0 0\n",
    "1\nLattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3 Time=0 time_unit=\"ps\" length_unit=\"nm\" pbc=\"F F F\"\nH 0 0 0\n",
)
    @test_throws ArgumentError read_extxyz_frames(IOBuffer(malformed))
end
println("trajectory_format_extxyz: 22 assertions passed (round-trip, multi-frame, metadata validation, and malformed-frame refusal)")
