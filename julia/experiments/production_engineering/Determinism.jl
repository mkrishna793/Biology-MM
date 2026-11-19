module Determinism

using SHA
using ..ProductionErrors: DeterminismError

export CounterRNG, random_u64, random_float, canonical_sum

"""Counter-based stream; a draw depends only on seed, stream, and explicit index."""
struct CounterRNG
    seed::UInt64
    stream::UInt64
    function CounterRNG(seed::Integer, stream::Integer=0)
        new(UInt64(seed), UInt64(stream))
    end
end

function random_u64(rng::CounterRNG, index::Integer)
    index >= 0 || throw(DeterminismError("random index must be nonnegative"))
    io = IOBuffer()
    write(io, codeunits("FoundationProductionEngineering/counter-v1\0"))
    write(io, hton(rng.seed)); write(io, hton(rng.stream)); write(io, hton(UInt64(index)))
    digest = SHA.sha256(take!(io))
    foldl((x, b) -> (x << 8) | UInt64(b), digest[1:8]; init=UInt64(0))
end

random_float(rng::CounterRNG, index::Integer) = Float64(random_u64(rng, index) >> 11) * 0x1.0p-53

"""Fixed balanced pairwise sum. Input order and floating-point type are part of the contract."""
function canonical_sum(values)
    isempty(values) && return 0.0
    work = collect(values)
    all(x -> x isa Real && isfinite(x), work) || throw(DeterminismError("canonical_sum accepts finite real values only"))
    while length(work) > 1
        next = Vector{promote_type(eltype(work), Float64)}(undef, cld(length(work), 2))
        j = 1
        for i in 1:2:length(work)
            next[j] = i == length(work) ? work[i] : work[i] + work[i+1]
            j += 1
        end
        work = next
    end
    work[1]
end

end
