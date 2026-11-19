include("ProductionEngineering.jl")
using .ProductionEngineering

const PE = ProductionEngineering
snapshot = PE.Checkpoint.SimulationSnapshot(
    q=[1.0 2.0; 3.0 4.0], p=zeros(2, 2), velocities=zeros(2, 2), box=(10.0, 10.0, 10.0),
    topology_id="topology-sha256", parameter_id="parameters-sha256",
    rng=PE.Determinism.CounterRNG(1234), integrator=(kind=:verlet, dt=0.002, phase=0),
    neighbor_state=(list=[(1, 2)], rebuild_counter=7, reference_q=[1.0 2.0; 3.0 4.0]),
    step=42, seeds=Dict("initial"=>1234), engine_version="demo-1")
path = joinpath(@__DIR__, "demo.checkpoint")
digest = PE.Checkpoint.write_checkpoint(path, snapshot)
restored = PE.Checkpoint.read_checkpoint(path; expected_topology_id="topology-sha256",
    expected_parameter_id="parameters-sha256", expected_engine_version="demo-1", expected_box=snapshot.box)
manifest = PE.Provenance.provenance_manifest(seed=1234,
    package_versions=Dict("Julia"=>string(VERSION), "FoundationCore"=>"0.1.0"),
    box=snapshot.box, cutoff_conventions=(mode=:none, cutoff=nothing),
    thread_count=Threads.nthreads(), model_parameters=(potential=:harmonic, k=2.0))
prov = PE.Provenance.provenance_digest(manifest)
println("Julia ", VERSION, " | checkpoint SHA-256 ", digest, " | provenance SHA-256 ", prov,
        " | restored step ", restored.step)
rm(path)
