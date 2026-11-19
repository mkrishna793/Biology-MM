using Test
using SHA
using Serialization
include(joinpath(@__DIR__, "..", "ProductionEngineering.jl"))
using .ProductionEngineering

const PE = ProductionEngineering
const E = PE.ProductionErrors

function fixture()
    PE.Checkpoint.SimulationSnapshot(q=[1.0, 2.0], p=[3.0, 4.0], velocities=[0.1, 0.2],
        box=(8.0, 9.0, 10.0), topology_id="topo-a", parameter_id="params-a",
        rng=PE.Determinism.CounterRNG(44, 2), integrator=(kind=:langevin, phase=1, dt=0.001),
        neighbor_state=(pairs=[(1,2)], rebuild_counter=12, reference_q=[1.0,2.0]),
        step=999, seeds=Dict("init"=>UInt64(44), "thermostat"=>UInt64(81)), engine_version="engine-a")
end

function advance(s)
    next_step = s.step + 1
    noise = PE.Determinism.random_float(s.rng, next_step)
    increment = PE.Determinism.canonical_sum((noise, s.integrator.dt, s.neighbor_state.rebuild_counter * 1e-6))
    q = s.q .+ increment
    p = s.p .- increment
    PE.Checkpoint.SimulationSnapshot(q=q, p=p, velocities=s.velocities .+ increment,
        box=s.box, topology_id=s.topology_id, parameter_id=s.parameter_id, rng=s.rng,
        integrator=(kind=s.integrator.kind, phase=s.integrator.phase + 1, dt=s.integrator.dt),
        neighbor_state=(pairs=s.neighbor_state.pairs, rebuild_counter=s.neighbor_state.rebuild_counter,
                        reference_q=s.neighbor_state.reference_q),
        step=next_step, seeds=s.seeds, engine_version=s.engine_version)
end

function advance_n(s, n)
    for _ in 1:n
        s = advance(s)
    end
    s
end

function same_snapshot(a, b)
    a.q == b.q && a.p == b.p && a.velocities == b.velocities && a.box == b.box &&
    a.topology_id == b.topology_id && a.parameter_id == b.parameter_id &&
    a.rng.seed == b.rng.seed && a.rng.stream == b.rng.stream && a.integrator == b.integrator &&
    a.neighbor_state == b.neighbor_state && a.step == b.step && a.seeds == b.seeds &&
    a.engine_version == b.engine_version
end

@testset "production engineering" begin
    path = joinpath(@__DIR__, "continuation-threads-$(Threads.nthreads()).chk")
    s = fixture()
    PE.Checkpoint.write_checkpoint(path, s)
    r = PE.Checkpoint.read_checkpoint(path; expected_topology_id="topo-a", expected_parameter_id="params-a",
        expected_engine_version="engine-a", expected_box=s.box)
    @test r.q == s.q && r.p == s.p && r.velocities == s.velocities
    @test r.topology_id == s.topology_id && r.parameter_id == s.parameter_id
    @test r.rng.seed == s.rng.seed && r.integrator == s.integrator && r.neighbor_state == s.neighbor_state
    @test r.step == s.step && r.seeds == s.seeds && r.engine_version == s.engine_version

    start = fixture()
    uninterrupted = advance_n(start, 10)
    checkpoint = advance_n(start, 4)
    PE.Checkpoint.write_checkpoint(path, checkpoint)
    restored = PE.Checkpoint.read_checkpoint(path; expected_topology_id="topo-a", expected_parameter_id="params-a")
    restarted = advance_n(restored, 6)
    @test same_snapshot(restarted, uninterrupted)
    @test reinterpret(UInt64, restarted.q) == reinterpret(UInt64, uninterrupted.q)
    @test reinterpret(UInt64, restarted.p) == reinterpret(UInt64, uninterrupted.p)
    @test reinterpret(UInt64, restarted.velocities) == reinterpret(UInt64, uninterrupted.velocities)

    PE.Checkpoint.write_checkpoint(path, s)
    @test_throws E.CheckpointIncompatibility PE.Checkpoint.read_checkpoint(path; expected_topology_id="other")
    @test_throws E.CheckpointIncompatibility PE.Checkpoint.read_checkpoint(path; expected_parameter_id="other")
    encoded = read(path)
    payload_start = 8 + 8 + 32 + 1
    envelope = deserialize(IOBuffer(encoded[payload_start:end]))
    io = IOBuffer()
    serialize(io, (version=PE.Checkpoint.CHECKPOINT_VERSION + 1, snapshot=envelope.snapshot))
    unsupported_payload = take!(io)
    unsupported_digest = SHA.sha256(unsupported_payload)
    replacement = IOBuffer()
    write(replacement, encoded[1:8])
    write(replacement, hton(UInt64(length(unsupported_payload))))
    write(replacement, unsupported_digest)
    write(replacement, unsupported_payload)
    write(path, take!(replacement))
    @test_throws E.UnsupportedCheckpointVersion PE.Checkpoint.read_checkpoint(path)
    PE.Checkpoint.write_checkpoint(path, s)
    bytes = read(path); bytes[end] ⊻= 0x01; write(path, bytes)
    @test_throws E.CheckpointCorruption PE.Checkpoint.read_checkpoint(path)
    PE.Checkpoint.write_checkpoint(path, s)

    rng = PE.Determinism.CounterRNG(9001, 3)
    serial_draws = [PE.Determinism.random_u64(rng, i) for i in 0:1000]
    threaded_draws = Vector{UInt64}(undef, length(serial_draws))
    Threads.@threads for i in eachindex(threaded_draws)
        threaded_draws[i] = PE.Determinism.random_u64(rng, i - 1)
    end
    @test serial_draws == threaded_draws
    println("thread fixture SHA-256 ", bytes2hex(SHA.sha256(reinterpret(UInt8, threaded_draws))))
    @test PE.Determinism.random_u64(rng, 9) != PE.Determinism.random_u64(rng, 10)
    values = [(-1.0)^i / i for i in 1:1001]
    @test PE.Determinism.canonical_sum(values) === PE.Determinism.canonical_sum(copy(values))

    model = Dict("k"=>2.0, "kind"=>:harmonic, "parameters"=>[2.0, 1.5])
    manifest = PE.Provenance.provenance_manifest(seed=44, package_versions=Dict("Julia"=>string(VERSION), "FoundationCore"=>"0.1.0"),
        box=(8.0, 9.0, 10.0), cutoff_conventions=(mode=:lfs, cutoff=2.5), thread_count=Threads.nthreads(), model_parameters=model)
    @test manifest.seed == 44
    @test manifest.package_versions["Julia"] == string(VERSION)
    @test manifest.box == (8.0, 9.0, 10.0)
    @test manifest.cutoff_conventions == (mode=:lfs, cutoff=2.5)
    @test manifest.thread_count == Threads.nthreads()
    @test manifest.model_parameter_hash == bytes2hex(SHA.sha256(PE.Provenance.canonical_bytes(model)))
    @test PE.Provenance.provenance_digest(manifest) != PE.Provenance.provenance_digest(PE.Provenance.provenance_manifest(
        seed=45, package_versions=manifest.package_versions, box=manifest.box,
        cutoff_conventions=manifest.cutoff_conventions, thread_count=manifest.thread_count, model_parameters=model))
    @test PE.Provenance.provenance_digest(manifest) != PE.Provenance.provenance_digest(PE.Provenance.provenance_manifest(
        seed=44, package_versions=manifest.package_versions, box=manifest.box,
        cutoff_conventions=manifest.cutoff_conventions, thread_count=manifest.thread_count, model_parameters=Dict("k"=>3.0)))

    taxonomy = (E.InvalidInput, E.SingularGeometry, E.UnstableOperator,
                E.NonConvergence, E.UnsupportedFeature, E.ResourceRefusal)
    @test length(unique(taxonomy)) == 6
    @test all(T("x") isa E.ProductionError for T in taxonomy)
end

println("Julia ", VERSION, "; threads=", Threads.nthreads(), "; production engineering tests PASS")
