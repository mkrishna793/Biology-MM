using FoundationCore, JSON, Test
module ImportedCLI
include(joinpath(@__DIR__, "..", "bin", "md.jl"))
end

@testset "prepared AMBER CLI execution" begin
    repo = normpath(joinpath(@__DIR__, "..", ".."))
    python = let p = get(ENV, "CHEM_PYTHON", "")
        !isempty(p) ? p : something(Sys.which("python3"), Sys.which("python"), "python3")
    end
    mktempdir() do dir
        parm = joinpath(repo,"wrappers","python","tests","systems","DNA_mbondi3.prmtop")
        crd = joinpath(repo,"wrappers","python","tests","systems","DNA_mbondi3.inpcrd")
        normalized = joinpath(dir,"dna.json")
        extractor = joinpath(repo,"chemistry","tools","extract_amber.py")
        run(pipeline(`$python $extractor --parm $parm --crd $crd --out $normalized --force-field "DNA integration test" --repo-root $repo`; stdout=devnull))
        lines = readlines(crd)
        n = parse(Int,split(lines[2])[1])
        vals = parse.(Float64,split(join(lines[3:end]," ")))[1:3n] ./ 10
        coordinates = [vals[3i-2:3i] for i in 1:n]
        doc = Dict("schema_version"=>1,"normalized_amber"=>"dna.json",
            "coordinates"=>coordinates,"units"=>"nm,kJ/mol,amu,ps",
            "protocol"=>"direct-nocutoff-flexible","ensemble"=>"nve","workdir"=>"run",
            "workflow"=>Dict("temperature"=>0.0,"dt"=>1e-4,"minimize_max_steps"=>0,
                "equilibrate_steps"=>0,"produce_steps"=>4,"frame_every"=>1,"contact_tol"=>0.0),
            "analysis"=>Dict("min_frames"=>4,"min_fit_points"=>2,"fit_window"=>[0,1],"unwrap"=>false,"min_ess"=>1))
        spec = joinpath(dir,"spec.json")
        write(spec,JSON.json(doc))
        ImportedCLI.run_spec(spec)
        manifest = JSON.parsefile(joinpath(dir,"run","manifest.json"))
        @test haskey(manifest["resolved_parameters"],"normalized_amber")
        @test isfile(joinpath(dir,"run","trajectory.xyz"))

        # Test resume_from in CLI spec
        doc_resume = copy(doc)
        doc_resume["resume_from"] = Dict("checkpoint" => "run/checkpoints/ckpt_latest.json",
                                         "manifest" => "run/manifest.json")
        spec_resume = joinpath(dir, "spec_resume.json")
        write(spec_resume, JSON.json(doc_resume))
        ImportedCLI.run_spec(spec_resume)
        @test isfile(joinpath(dir, "run", "checkpoints", "ckpt_latest.json"))

        doc["units"] = "reduced"
        write(spec,JSON.json(doc))
        @test_throws ImportedCLI.CLIInputError ImportedCLI.run_spec(spec)
    end
end
