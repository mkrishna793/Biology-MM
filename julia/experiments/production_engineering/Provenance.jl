module Provenance

using SHA

export canonical_bytes, ProvenanceManifest, provenance_manifest, provenance_digest

function _encode(io::IO, x::Nothing)
    write(io, "n;")
end
function _encode(io::IO, x::Bool)
    write(io, x ? "b1;" : "b0;")
end
function _encode(io::IO, x::Integer)
    write(io, "i", string(typeof(x)), ":", string(x), ";")
end
function _encode(io::IO, x::AbstractFloat)
    isfinite(x) || throw(ArgumentError("canonical provenance rejects non-finite floats"))
    write(io, "f", string(typeof(x)), ":", bitstring(x), ";")
end
function _encode(io::IO, x::AbstractString)
    b = codeunits(x); write(io, "s", string(length(b)), ":", b, ";")
end
function _encode(io::IO, x::Symbol)
    write(io, "y"); _encode(io, String(x))
end
function _encode(io::IO, x::Pair)
    write(io, "p{"); _encode(io, first(x)); _encode(io, last(x)); write(io, "}")
end
function _encode(io::IO, x::NamedTuple)
    write(io, "t{"); _encode(io, collect(pairs(x))); write(io, "}")
end
function _encode(io::IO, x::AbstractDict)
    write(io, "d", string(length(x)), "{")
    entries = collect(pairs(x))
    sort!(entries; by=p -> String(p.first))
    foreach(p -> _encode(io, p), entries)
    write(io, "}")
end
function _encode(io::IO, x::Tuple)
    write(io, "u", string(length(x)), "["); foreach(v -> _encode(io, v), x); write(io, "]")
end
function _encode(io::IO, x::AbstractArray)
    write(io, "a", string(eltype(x)), ":", join(size(x), ","), "[")
    foreach(v -> _encode(io, v), x); write(io, "]")
end
function _encode(io::IO, x)
    throw(ArgumentError("unsupported canonical provenance type: $(typeof(x))"))
end

function canonical_bytes(x)
    io = IOBuffer(); _encode(io, x); take!(io)
end

"""Named reproducibility metadata; the model hash is computed from model parameters."""
struct ProvenanceManifest{B,C}
    seed::UInt64
    package_versions::Dict{String,String}
    box::B
    cutoff_conventions::C
    thread_count::Int
    model_parameter_hash::String
end

function provenance_manifest(; seed, package_versions, box, cutoff_conventions,
                             thread_count, model_parameters)
    thread_count > 0 || throw(ArgumentError("thread_count must be positive"))
    model_hash = bytes2hex(SHA.sha256(canonical_bytes(model_parameters)))
    ProvenanceManifest(UInt64(seed), Dict(String(k)=>String(v) for (k,v) in package_versions),
                       deepcopy(box), deepcopy(cutoff_conventions), Int(thread_count), model_hash)
end

"""SHA-256 over the explicitly named reproducibility manifest fields."""
function provenance_digest(manifest::ProvenanceManifest)
    document = (schema=:production_provenance_v1,
                seed=manifest.seed,
                package_versions=manifest.package_versions,
                box=manifest.box,
                cutoff_conventions=manifest.cutoff_conventions,
                thread_count=manifest.thread_count,
                model_parameter_hash=manifest.model_parameter_hash)
    bytes2hex(SHA.sha256(canonical_bytes(document)))
end

end
