module Checkpoint

using Serialization
using SHA
using ..ProductionErrors

export SimulationSnapshot, write_checkpoint, read_checkpoint, CHECKPOINT_VERSION

const CHECKPOINT_VERSION = 1
const MAGIC = Vector{UInt8}(codeunits("FPECHK01"))

"""Complete restart record. Caller-owned arrays/metadata are copied on construction."""
struct SimulationSnapshot{Q,P,V,B,T,A,N}
    q::Q
    p::P
    velocities::V
    box::B
    topology_id::String
    parameter_id::String
    rng::A
    integrator::T
    neighbor_state::N
    step::Int64
    seeds::Dict{String,UInt64}
    engine_version::String
end

function SimulationSnapshot(; q, p, velocities, box, topology_id, parameter_id, rng,
                             integrator, neighbor_state, step, seeds, engine_version)
    step >= 0 || throw(InvalidCheckpoint("step must be nonnegative"))
    SimulationSnapshot(deepcopy(q), deepcopy(p), deepcopy(velocities), deepcopy(box),
        String(topology_id), String(parameter_id), deepcopy(rng), deepcopy(integrator),
        deepcopy(neighbor_state), Int64(step), Dict(String(k)=>UInt64(v) for (k,v) in seeds), String(engine_version))
end

function _payload(snapshot)
    io = IOBuffer(); serialize(io, (version=CHECKPOINT_VERSION, snapshot=snapshot)); take!(io)
end

function write_checkpoint(path::AbstractString, snapshot::SimulationSnapshot)
    payload = _payload(snapshot)
    digest = SHA.sha256(payload)
    open(path, "w") do io
        write(io, MAGIC); write(io, hton(UInt64(length(payload)))); write(io, digest); write(io, payload)
    end
    bytes2hex(digest)
end

function read_checkpoint(path::AbstractString; expected_topology_id=nothing,
                         expected_parameter_id=nothing, expected_engine_version=nothing,
                         expected_box=nothing)
    bytes = read(path)
    minimum = length(MAGIC) + 8 + 32
    length(bytes) >= minimum || throw(CheckpointCorruption("truncated envelope"))
    bytes[1:length(MAGIC)] == MAGIC || throw(CheckpointCorruption("bad magic"))
    len = ntoh(reinterpret(UInt64, bytes[length(MAGIC)+1:length(MAGIC)+8])[1])
    len == length(bytes) - minimum || throw(CheckpointCorruption("payload length mismatch"))
    digest_start = length(MAGIC) + 9
    digest = bytes[digest_start:digest_start+31]
    payload = bytes[minimum+1:end]
    SHA.sha256(payload) == digest || throw(CheckpointCorruption("SHA-256 mismatch"))
    envelope = try
        deserialize(IOBuffer(payload))
    catch e
        throw(CheckpointCorruption("payload cannot be decoded ($(typeof(e)))"))
    end
    hasproperty(envelope, :version) && hasproperty(envelope, :snapshot) || throw(CheckpointCorruption("invalid payload structure"))
    envelope.version == CHECKPOINT_VERSION || throw(UnsupportedCheckpointVersion(Int(envelope.version), CHECKPOINT_VERSION))
    s = envelope.snapshot
    s isa SimulationSnapshot || throw(CheckpointCorruption("payload does not contain a simulation snapshot"))
    expected_topology_id === nothing || s.topology_id == expected_topology_id || throw(CheckpointIncompatibility("topology ID differs"))
    expected_parameter_id === nothing || s.parameter_id == expected_parameter_id || throw(CheckpointIncompatibility("parameter ID differs"))
    expected_engine_version === nothing || s.engine_version == expected_engine_version || throw(CheckpointIncompatibility("engine version differs"))
    expected_box === nothing || isequal(s.box, expected_box) || throw(CheckpointIncompatibility("box differs"))
    deepcopy(s)
end

end
