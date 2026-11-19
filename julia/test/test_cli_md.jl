using Test
using JSON

const CLI = joinpath(@__DIR__, "..", "bin", "md.jl")
const JULIA = get(ENV, "JULIA", first(Base.julia_cmd()))
const PROJECT = dirname(@__DIR__)

function invoke_cli(spec)
    path = joinpath(dirname(spec), "spec.json")
    write(path, JSON.json(spec))
    cmd = `$JULIA --project=$PROJECT $CLI $path`
    out = read(pipeline(cmd, stderr=Pipe()), String)
    return out
end

@testset "md CLI end-to-end and typed refusals" begin
    help = read(`$JULIA --project=$PROJECT $CLI --help`, String)
    @test occursin("validate -> prepare -> minimize -> equilibrate -> produce -> analyze", help)
    mktempdir() do dir
        good = Dict("schema_version"=>1, "coordinates"=>[[1.0,1,1],[2.3,1,1]], "masses"=>[1,1], "box"=>[20,20,20],
            "ensemble"=>"nve", "workdir"=>"run", "factors"=>[Dict("type"=>"harmonic_bond","i"=>1,"j"=>2,"k"=>1.0,"r0"=>1.3)],
            "workflow"=>Dict("run_id"=>"cli-e2e","dt"=>0.001,"temperature"=>0.0,"minimize_max_steps"=>0,"equilibrate_steps"=>0,
                "produce_steps"=>100,"frame_every"=>5,"checkpoint_every"=>0),
            "analysis"=>Dict("min_frames"=>10,"lag_window"=>[0.005,0.025],"min_fit_points"=>5,"min_ess"=>1.0,"unwrap"=>false))
        specpath = joinpath(dir,"spec.json")
        write(specpath, JSON.json(good))
        cmd = `$JULIA --project=$PROJECT $CLI $specpath`
        output = read(cmd, String)
        result = JSON.parse(output)
        run_dir = joinpath(dir,"run")
        @test result["stages"] == ["validate","prepare","minimize","equilibrate","produce","analyze"]
        @test result["frames"] == 20
        traj = read(joinpath(run_dir,"trajectory.xyz"), String)
        @test occursin("Lattice=", traj) && occursin("Time=", traj)
        records = JSON.parse.(filter(!isempty, split(read(joinpath(run_dir,"trajectory.jsonl"),String),'\n')))
        @test length(records) == 20
        @test all(r -> r["volume"] == 8000 && r["pressure"] === nothing && r["temperature"] isa Real && haskey(r,"component_energies"), records)
        @test all(r -> r["elapsed_seconds"] >= 0 && (r["steps_per_second"] === nothing || r["steps_per_second"] >= 0), records)
        meta = JSON.parse.(filter(!isempty, split(read(joinpath(run_dir,"trajectory.xyz.jsonl"),String),'\n')))
        @test length(meta) == 20 && meta[end]["time"] ≈ 0.1
        @test occursin("length_unit=\"reduced_length\"", traj)
        @test occursin("time_unit=\"reduced_time\"", traj)
        @test meta[end]["box_matrix"] == [[20,0,0],[0,20,0],[0,0,20]]
        @test isfile(joinpath(run_dir,"checkpoints","ckpt_latest.json"))

        cases = [
            (merge(copy(good), Dict("box"=>[0,20,20])), "InvalidBox"),
            (merge(copy(good), Dict("factors"=>[Dict("type"=>"harmonic_bond","i"=>1,"j"=>2,"k"=>1.0)])), "MissingParameter"),
            (merge(copy(good), Dict("ensemble"=>"npt")), "UnsupportedEnsemble"),
            (merge(copy(good), Dict("pdb"=>"input.pdb")), "UnsupportedChemistry"),
        ]
        for (bad, code) in cases
            bad["workdir"] = "bad-" * code
            p = joinpath(dir,"bad.json")
            write(p, JSON.json(bad))
            errpipe = Pipe()
            failure = run(pipeline(`$JULIA --project=$PROJECT $CLI $p`, stdout=devnull, stderr=errpipe), wait=false)
            close(errpipe.in)
            err = read(errpipe, String)
            wait(failure)
            @test !success(failure)
            @test occursin(code * ":", err)
        end
    end
end
