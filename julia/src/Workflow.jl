"""
Operable, recoverable, and auditable simulation workflow (`validate -> prepare ->
minimize -> equilibrate -> produce -> analyze`).

## Guarantees and explicit boundaries

* **Immutable `RunManifest`**: fixes resolved parameters, model versions,
  units, periodic box matrix/lengths, floating-point precision, algorithm, `dt`,
  seed, and software version, sealed with SHA-256 digests (`parameter_digest`,
  `manifest_digest`).
* **Atomic versioned checkpoints**: `write_checkpoint_atomic` writes to a
  same-filesystem temporary file, flushes, and replaces the target path via
  atomic rename. Failed or interrupted writes never corrupt an existing
  checkpoint.
* **Explicit restart guarantees**: every checkpoint records
  `restart_guarantee` (`:bitwise` or `:statistical`). `:bitwise` checkpoints
  serialize IEEE-754 `Float64` bit patterns for state (`q`, `p`, `F`), box
  geometry, thermostat/barostat state, and exact RNG state (`Xoshiro` or
  `MersenneTwister`). `:statistical` checkpoints are explicitly rejected when
  `require_bitwise = true`.
* **Trajectory metadata**: `TrajectoryWriter` and `read_trajectory` pair XYZ
  frames with a JSONL metadata sidecar carrying `step`, `time`, `box_kind`,
  `box_lengths`, 3x3 `box_matrix`, `units`, and `manifest_digest`.
* **Strict `analyze` guards**: `analyze` refuses insufficient frame counts,
  non-monotonic or non-finite timestamps, negative/inverted/short MSD fit
  windows, ambiguous half-box periodic jumps, and insufficient ESS.
"""
module Workflow

using ..Types
using ..Factors
using ..Constraints
using ..Dynamics
using ..Sampling
using ..Prepare
using ..MonteCarloBarostat
using ..Respa
using JSON
using LinearAlgebra
using Printf
using Random
using Statistics

using ..TrajectoryFormat: write_extxyz_frame, read_extxyz_frames

export CHECKPOINT_FORMAT_VERSION,
       CHECKPOINT_MAGIC,
       MANIFEST_VERSION,
       WorkflowError,
       ValidationError,
       ManifestError,
       CheckpointError,
       CheckpointCorruptionError,
       CheckpointVersionError,
       CheckpointMismatchError,
       TrajectoryError,
       AnalysisGuardError,
       TransactionError,
       sha256_bytes,
       sha256_hex,
       RunManifest,
       build_manifest,
       verify_manifest,
       manifest_resolved_parameters,
       manifest_model_versions,
       manifest_box_matrix,
       write_manifest_atomic,
       read_manifest,
       ThermostatState,
       BarostatState,
       WorkflowCheckpoint,
       encode_rng_state,
       decode_rng_state,
       make_checkpoint,
       restore_state,
       write_checkpoint_atomic,
       read_checkpoint,
       TrajectoryWriter,
       TrajectoryData,
       write_trajectory_frame!,
       read_trajectory,
       AnalysisConfig,
       AnalysisResult,
       unwrap_frames,
       analyze,
       WorkflowConfig,
       StageSpec,
       StageRecord,
       WorkflowResult,
       validate_stage,
       prepare_stage,
       minimize_stage,
       equilibrate_stage,
       produce_stage,
       analyze_stage,
       run_workflow,
       resume_workflow

const MANIFEST_VERSION = "1.0.0"
const CHECKPOINT_FORMAT_VERSION = 1
const CHECKPOINT_MAGIC = "FCCHK_V1"
const WORKFLOW_SOFTWARE_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------

abstract type WorkflowError <: Exception end

struct ValidationError <: WorkflowError
    field::Union{Nothing,Symbol}
    msg::String
    ValidationError(field::Union{Nothing,Symbol}, msg::AbstractString) = new(field, String(msg))
end
ValidationError(msg::AbstractString) = ValidationError(nothing, msg)
struct ManifestError <: WorkflowError
    msg::String
end
abstract type CheckpointError <: WorkflowError end
struct CheckpointCorruptionError <: CheckpointError
    msg::String
end
struct CheckpointVersionError <: CheckpointError
    found::Int
    expected::Int
end
struct CheckpointMismatchError <: CheckpointError
    msg::String
end
struct TrajectoryError <: WorkflowError
    msg::String
end
struct AnalysisGuardError <: WorkflowError
    msg::String
end
struct TransactionError <: WorkflowError
    stage::Symbol
    msg::String
end

Base.showerror(io::IO, e::ValidationError) =
    e.field === nothing ? print(io, "ValidationError: ", e.msg) : print(io, "ValidationError[", e.field, "]: ", e.msg)
Base.showerror(io::IO, e::ManifestError) = print(io, "ManifestError: ", e.msg)
Base.showerror(io::IO, e::CheckpointCorruptionError) = print(io, "CheckpointCorruptionError: ", e.msg)
Base.showerror(io::IO, e::CheckpointVersionError) =
    print(io, "CheckpointVersionError: unsupported checkpoint format version $(e.found) (expected $(e.expected))")
Base.showerror(io::IO, e::CheckpointMismatchError) = print(io, "CheckpointMismatchError: ", e.msg)
Base.showerror(io::IO, e::TrajectoryError) = print(io, "TrajectoryError: ", e.msg)
Base.showerror(io::IO, e::AnalysisGuardError) = print(io, "AnalysisGuardError: ", e.msg)
Base.showerror(io::IO, e::TransactionError) = print(io, "TransactionError[", e.stage, "]: ", e.msg)

# ---------------------------------------------------------------------------
# Pure-Julia FIPS 180-4 SHA-256 and canonical encoding
# ---------------------------------------------------------------------------

const _SHA256_K = UInt32[
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]

@inline _rotr32(x::UInt32, n::Int) = (x >>> n) | (x << (32 - n))

"""Compute the SHA-256 digest (32 bytes) of `data` using FIPS 180-4."""
function sha256_bytes(data::AbstractVector{UInt8})
    len = length(data)
    bitlen = UInt64(len) * UInt64(8)
    rem = (len + 1 + 8) % 64
    pad_zeros = rem == 0 ? 0 : 64 - rem
    padded = Vector{UInt8}(undef, len + 1 + pad_zeros + 8)
    copyto!(padded, 1, data, 1, len)
    padded[len + 1] = 0x80
    for i in 1:pad_zeros
        padded[len + 1 + i] = 0x00
    end
    for b in 0:7
        padded[end - b] = UInt8((bitlen >> (8 * b)) & 0xff)
    end

    H = UInt32[
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ]
    w = Vector{UInt32}(undef, 64)

    for offset in 1:64:length(padded)
        @inbounds for i in 0:15
            j = offset + 4 * i
            w[i + 1] = (UInt32(padded[j]) << 24) |
                       (UInt32(padded[j + 1]) << 16) |
                       (UInt32(padded[j + 2]) << 8) |
                       UInt32(padded[j + 3])
        end
        @inbounds for i in 17:64
            s0 = _rotr32(w[i - 15], 7) ⊻ _rotr32(w[i - 15], 18) ⊻ (w[i - 15] >>> 3)
            s1 = _rotr32(w[i - 2], 17) ⊻ _rotr32(w[i - 2], 19) ⊻ (w[i - 2] >>> 10)
            w[i] = w[i - 16] + s0 + w[i - 7] + s1
        end
        a, b, c, d, e, f, g, h = H
        @inbounds for i in 1:64
            S1 = _rotr32(e, 6) ⊻ _rotr32(e, 11) ⊻ _rotr32(e, 25)
            ch = (e & f) ⊻ ((~e) & g)
            temp1 = h + S1 + ch + _SHA256_K[i] + w[i]
            S0 = _rotr32(a, 2) ⊻ _rotr32(a, 13) ⊻ _rotr32(a, 22)
            maj = (a & b) ⊻ (a & c) ⊻ (b & c)
            temp2 = S0 + maj
            h = g; g = f; f = e; e = d + temp1
            d = c; c = b; b = a; a = temp1 + temp2
        end
        H[1] += a; H[2] += b; H[3] += c; H[4] += d
        H[5] += e; H[6] += f; H[7] += g; H[8] += h
    end

    out = Vector{UInt8}(undef, 32)
    @inbounds for i in 1:8
        v = H[i]
        out[4i - 3] = UInt8((v >> 24) & 0xff)
        out[4i - 2] = UInt8((v >> 16) & 0xff)
        out[4i - 1] = UInt8((v >> 8) & 0xff)
        out[4i]     = UInt8(v & 0xff)
    end
    return out
end

sha256_hex(data::AbstractVector{UInt8}) = bytes2hex(sha256_bytes(data))
sha256_hex(s::AbstractString) = sha256_hex(Vector{UInt8}(codeunits(String(s))))

function _canonical_encode(io::IO, ::Nothing)
    write(io, "n;")
end
function _canonical_encode(io::IO, x::Bool)
    write(io, x ? "b1;" : "b0;")
end
function _canonical_encode(io::IO, x::Integer)
    write(io, "i:", string(Int64(x)), ";")
end
function _canonical_encode(io::IO, x::AbstractFloat)
    xf = Float64(x)
    isfinite(xf) || throw(ManifestError("canonical provenance rejects non-finite floats: $x"))
    # Normalize -0.0 bit pattern deterministically only if needed; preserve exact IEEE-754 bits
    write(io, "f64:", bitstring(xf), ";")
end
function _canonical_encode(io::IO, x::AbstractString)
    b = codeunits(String(x))
    write(io, "s", string(length(b)), ":")
    write(io, b)
    write(io, ";")
end
function _canonical_encode(io::IO, x::Symbol)
    write(io, "y:")
    _canonical_encode(io, String(x))
end
function _canonical_encode(io::IO, x::Tuple)
    write(io, "u", string(length(x)), "[")
    for v in x
        _canonical_encode(io, v)
    end
    write(io, "]")
end
function _canonical_encode(io::IO, x::AbstractArray)
    write(io, "a:", join(size(x), ","), "[")
    for v in x
        _canonical_encode(io, v)
    end
    write(io, "]")
end
function _canonical_encode(io::IO, x::AbstractDict)
    entries = [(String(k), v) for (k, v) in pairs(x)]
    sort!(entries; by = first)
    write(io, "d", string(length(entries)), "{")
    for (k, v) in entries
        _canonical_encode(io, k)
        _canonical_encode(io, v)
    end
    write(io, "}")
end
function _canonical_encode(io::IO, x::NamedTuple)
    entries = [(String(k), v) for (k, v) in pairs(x)]
    sort!(entries; by = first)
    write(io, "nt", string(length(entries)), "{")
    for (k, v) in entries
        _canonical_encode(io, k)
        _canonical_encode(io, v)
    end
    write(io, "}")
end
function _canonical_encode(io::IO, x)
    throw(ManifestError("unsupported canonical encoding type $(typeof(x))"))
end

function canonical_bytes(x)
    io = IOBuffer()
    _canonical_encode(io, x)
    return take!(io)
end

# ---------------------------------------------------------------------------
# Atomic file write helper
# ---------------------------------------------------------------------------

function _write_bytes_atomic(path::AbstractString, bytes::Vector{UInt8};
                             before_rename_hook::Union{Nothing,Function} = nothing)
    dest = abspath(String(path))
    dir = dirname(dest)
    mkpath(dir)
    tmp, io = mktemp(dir)
    try
        write(io, bytes)
        flush(io)
        ccall(:fsync, Cint, (Cint,), Base.fd(io)) == 0 || throw(SystemError("fsync", Libc.errno()))
        close(io)
        before_rename_hook !== nothing && before_rename_hook(tmp)
        mv(tmp, dest; force = true)
        dirfd = ccall(:open, Cint, (Cstring, Cint), dir, 0)
        if dirfd >= 0
            try
                ccall(:fsync, Cint, (Cint,), dirfd)
            finally
                ccall(:close, Cint, (Cint,), dirfd)
            end
        end
    catch
        try
            isopen(io) && close(io)
        catch
        end
        rm(tmp; force = true)
        rethrow()
    end
    return dest
end

# ---------------------------------------------------------------------------
# IEEE-754 bitstring helpers
# ---------------------------------------------------------------------------

@inline _float_to_hex(x::Float64) = string(reinterpret(UInt64, x); base = 16, pad = 16)

function _hex_to_float(s::AbstractString)
    length(s) == 16 || throw(CheckpointCorruptionError("invalid IEEE-754 hex word length: $s"))
    u = try
        parse(UInt64, s; base = 16)
    catch
        throw(CheckpointCorruptionError("invalid IEEE-754 hex word: $s"))
    end
    x = reinterpret(Float64, u)
    isfinite(x) || throw(CheckpointCorruptionError("non-finite float decoded from checkpoint"))
    return x
end

function _matrix_to_bits(M::AbstractMatrix{Float64})
    return [_float_to_hex(x) for x in vec(M)]
end

function _bits_to_matrix(bits::AbstractVector, nrows::Int, ncols::Int)
    length(bits) == nrows * ncols ||
        throw(CheckpointCorruptionError("expected $(nrows * ncols) matrix entries, got $(length(bits))"))
    v = Vector{Float64}(undef, nrows * ncols)
    @inbounds for i in eachindex(bits)
        v[i] = _hex_to_float(String(bits[i]))
    end
    return reshape(v, nrows, ncols)
end

# ---------------------------------------------------------------------------
# Box encoding helpers
# ---------------------------------------------------------------------------

function _box_summary(box::Types.AbstractBox)
    if box isa Types.NoBox
        return (:none, nothing, nothing)
    elseif box isa Types.OrthorhombicBox
        L = box.L
        H = [L[1] 0.0 0.0; 0.0 L[2] 0.0; 0.0 0.0 L[3]]
        return (:orthorhombic, H, L)
    elseif box isa Types.TriclinicBox
        return (:triclinic, copy(box.H), Types.box_lengths(box))
    else
        throw(ValidationError("unsupported box type $(typeof(box))"))
    end
end

function _encode_box_dict(box::Types.AbstractBox)
    if box isa Types.NoBox
        return Dict{String,Any}("kind" => "none")
    elseif box isa Types.OrthorhombicBox
        return Dict{String,Any}(
            "kind" => "orthorhombic",
            "lengths" => collect(box.L),
            "lengths_bits" => [_float_to_hex(x) for x in box.L],
        )
    elseif box isa Types.TriclinicBox
        return Dict{String,Any}(
            "kind" => "triclinic",
            "matrix" => [[box.H[r, c] for c in 1:3] for r in 1:3],
            "matrix_bits" => _matrix_to_bits(box.H),
        )
    else
        throw(CheckpointError("unsupported box type $(typeof(box))"))
    end
end

function _decode_box_dict(d::AbstractDict)
    kind = get(d, "kind", "")
    if kind == "none"
        return Types.NoBox()
    elseif kind == "orthorhombic"
        if haskey(d, "lengths_bits")
            bits = d["lengths_bits"]
            length(bits) == 3 || throw(CheckpointCorruptionError("orthorhombic box requires 3 lengths"))
            return Types.OrthorhombicBox((_hex_to_float(String(bits[1])),
                                          _hex_to_float(String(bits[2])),
                                          _hex_to_float(String(bits[3]))))
        else
            Ls = d["lengths"]
            length(Ls) == 3 || throw(CheckpointCorruptionError("orthorhombic box requires 3 lengths"))
            return Types.OrthorhombicBox((Float64(Ls[1]), Float64(Ls[2]), Float64(Ls[3])))
        end
    elseif kind == "triclinic"
        if haskey(d, "matrix_bits")
            H = _bits_to_matrix(d["matrix_bits"], 3, 3)
            return Types.TriclinicBox(H)
        else
            rows = d["matrix"]
            length(rows) == 3 || throw(CheckpointCorruptionError("triclinic matrix must be 3x3"))
            H = zeros(3, 3)
            for r in 1:3, c in 1:3
                H[r, c] = Float64(rows[r][c])
            end
            return Types.TriclinicBox(H)
        end
    else
        throw(CheckpointCorruptionError("unknown box kind: $kind"))
    end
end

# ---------------------------------------------------------------------------
# Immutable RunManifest
# ---------------------------------------------------------------------------

"""
Immutable run provenance manifest fixing resolved parameters, model versions,
units, box geometry, precision, integration algorithm, timestep, seed, and
software version, sealed with SHA-256 digests.
"""
struct RunManifest
    manifest_version::String
    run_id::String
    software_version::String
    julia_version::String
    model_versions::Dict{String,String}
    resolved_parameters::Dict{String,Any}
    parameter_digest::String
    units::String
    box_kind::Symbol
    box_matrix::Union{Nothing,Matrix{Float64}}
    box_lengths::Union{Nothing,NTuple{3,Float64}}
    precision::String
    algorithm::Symbol
    dt::Float64
    seed::UInt64
    n_atoms::Int
    temperature::Float64
    pressure::Union{Nothing,Float64}
    gamma::Union{Nothing,Float64}
    constraints_summary::String
    manifest_digest::String
end

const SUPPORTED_ALGORITHMS = (:velocity_verlet, :respa_nve, :langevin_baoab, :npt_baoab_mc, :hmc)

function _normalize_json_value(v)
    if v isa AbstractDict
        return Dict{String,Any}(String(k) => _normalize_json_value(val) for (k, val) in pairs(v))
    elseif v isa AbstractArray
        return [_normalize_json_value(x) for x in v]
    elseif v isa Tuple
        return [_normalize_json_value(x) for x in v]
    elseif v isa Symbol
        return String(v)
    elseif v isa AbstractFloat
        xf = Float64(v)
        isfinite(xf) || throw(ManifestError("non-finite float in manifest parameters: $v"))
        return xf
    elseif v isa Bool || v isa AbstractString || v === nothing
        return v isa AbstractString ? String(v) : v
    elseif v isa Integer
        return Int64(v)
    else
        throw(ManifestError("unsupported parameter value type in RunManifest: $(typeof(v))"))
    end
end

function _compute_manifest_digest(; manifest_version::String, run_id::String,
                                  software_version::String, julia_version::String,
                                  model_versions::Dict{String,String},
                                  parameter_digest::String, units::String,
                                  box_kind::Symbol,
                                  box_matrix::Union{Nothing,Matrix{Float64}},
                                  box_lengths::Union{Nothing,NTuple{3,Float64}},
                                  precision::String, algorithm::Symbol,
                                  dt::Float64, seed::UInt64, n_atoms::Int,
                                  temperature::Float64,
                                  pressure::Union{Nothing,Float64},
                                  gamma::Union{Nothing,Float64},
                                  constraints_summary::String)
    doc = (
        manifest_version = manifest_version,
        run_id = run_id,
        software_version = software_version,
        julia_version = julia_version,
        model_versions = model_versions,
        parameter_digest = parameter_digest,
        units = units,
        box_kind = String(box_kind),
        box_matrix = box_matrix === nothing ? nothing : vec(box_matrix),
        box_lengths = box_lengths === nothing ? nothing : collect(box_lengths),
        precision = precision,
        algorithm = String(algorithm),
        dt = dt,
        seed = Int64(bitcast_u64_to_i64(seed)),
        n_atoms = n_atoms,
        temperature = temperature,
        pressure = pressure,
        gamma = gamma,
        constraints_summary = constraints_summary,
    )
    return sha256_hex(canonical_bytes(doc))
end

@inline bitcast_u64_to_i64(u::UInt64) = reinterpret(Int64, u)

function build_manifest(; run_id::AbstractString = "run-001",
                        software_version::AbstractString = WORKFLOW_SOFTWARE_VERSION,
                        julia_version::AbstractString = string(VERSION),
                        model_versions::AbstractDict = Dict("force_field" => "FoundationCore-0.1.0"),
                        resolved_parameters::AbstractDict = Dict{String,Any}(),
                        units::Union{Types.Units,AbstractString} = Types.REDUCED,
                        box::Types.AbstractBox = Types.NoBox(),
                        precision::AbstractString = "Float64",
                        algorithm::Symbol = :velocity_verlet,
                        dt::Real,
                        seed::Integer = 0x0000000000000001,
                        n_atoms::Integer,
                        temperature::Real = 0.0,
                        pressure::Union{Nothing,Real} = nothing,
                        gamma::Union{Nothing,Real} = nothing,
                        constraints_summary::AbstractString = "none")
    !isempty(strip(String(run_id))) || throw(ManifestError("run_id must be non-empty"))
    String(precision) == "Float64" ||
        throw(ManifestError("unsupported precision $(precision); only Float64 is implemented"))
    algorithm in SUPPORTED_ALGORITHMS ||
        throw(ManifestError("unsupported algorithm :$algorithm; expected one of $SUPPORTED_ALGORITHMS"))
    dt64 = Float64(dt)
    isfinite(dt64) && dt64 > 0 || throw(ManifestError("dt must be positive and finite, got $dt"))
    nat = Int(n_atoms)
    nat > 0 || throw(ManifestError("n_atoms must be positive, got $n_atoms"))
    temp64 = Float64(temperature)
    isfinite(temp64) && temp64 >= 0 ||
        throw(ManifestError("temperature must be finite and non-negative, got $temperature"))
    pres64 = pressure === nothing ? nothing : Float64(pressure)
    pres64 === nothing || isfinite(pres64) ||
        throw(ManifestError("pressure must be finite, got $pressure"))
    gamma64 = gamma === nothing ? nothing : Float64(gamma)
    gamma64 === nothing || (isfinite(gamma64) && gamma64 >= 0) ||
        throw(ManifestError("gamma must be finite and non-negative, got $gamma"))
    seed >= 0 || throw(ManifestError("seed must be non-negative, got $seed"))

    mv = Dict{String,String}(String(k) => String(v) for (k, v) in pairs(model_versions))
    rp = Dict{String,Any}(String(k) => _normalize_json_value(v) for (k, v) in pairs(resolved_parameters))
    p_digest = sha256_hex(canonical_bytes(rp))
    units_name = units isa Types.Units ? units.name : String(units)
    !isempty(strip(units_name)) || throw(ManifestError("units must be non-empty"))
    bkind, bmat, blen = _box_summary(box)
    seed64 = UInt64(seed)

    m_digest = _compute_manifest_digest(;
        manifest_version = MANIFEST_VERSION,
        run_id = String(run_id),
        software_version = String(software_version),
        julia_version = String(julia_version),
        model_versions = mv,
        parameter_digest = p_digest,
        units = units_name,
        box_kind = bkind,
        box_matrix = bmat,
        box_lengths = blen,
        precision = String(precision),
        algorithm = algorithm,
        dt = dt64,
        seed = seed64,
        n_atoms = nat,
        temperature = temp64,
        pressure = pres64,
        gamma = gamma64,
        constraints_summary = String(constraints_summary),
    )

    return RunManifest(
        MANIFEST_VERSION, String(run_id), String(software_version), String(julia_version),
        mv, rp, p_digest, units_name, bkind, bmat, blen, String(precision),
        algorithm, dt64, seed64, nat, temp64, pres64, gamma64,
        String(constraints_summary), m_digest,
    )
end

manifest_resolved_parameters(m::RunManifest) = deepcopy(m.resolved_parameters)
manifest_model_versions(m::RunManifest) = deepcopy(m.model_versions)
manifest_box_matrix(m::RunManifest) = m.box_matrix === nothing ? nothing : copy(m.box_matrix)

"""Recompute `parameter_digest` and `manifest_digest` and verify integrity."""
function verify_manifest(m::RunManifest)
    m.manifest_version == MANIFEST_VERSION ||
        throw(ManifestError("unsupported manifest version $(m.manifest_version)"))
    recomputed_param = sha256_hex(canonical_bytes(m.resolved_parameters))
    recomputed_param == m.parameter_digest ||
        throw(ManifestError("parameter_digest mismatch in RunManifest"))
    recomputed_manifest = _compute_manifest_digest(;
        manifest_version = m.manifest_version,
        run_id = m.run_id,
        software_version = m.software_version,
        julia_version = m.julia_version,
        model_versions = m.model_versions,
        parameter_digest = m.parameter_digest,
        units = m.units,
        box_kind = m.box_kind,
        box_matrix = m.box_matrix,
        box_lengths = m.box_lengths,
        precision = m.precision,
        algorithm = m.algorithm,
        dt = m.dt,
        seed = m.seed,
        n_atoms = m.n_atoms,
        temperature = m.temperature,
        pressure = m.pressure,
        gamma = m.gamma,
        constraints_summary = m.constraints_summary,
    )
    recomputed_manifest == m.manifest_digest ||
        throw(ManifestError("manifest_digest mismatch in RunManifest"))
    return true
end

function _manifest_to_dict(m::RunManifest)
    box_obj = if m.box_kind === :none
        Dict{String,Any}("kind" => "none")
    elseif m.box_kind === :orthorhombic
        Dict{String,Any}(
            "kind" => "orthorhombic",
            "lengths" => collect(m.box_lengths),
            "lengths_bits" => [_float_to_hex(x) for x in m.box_lengths],
        )
    else
        Dict{String,Any}(
            "kind" => "triclinic",
            "matrix" => [[m.box_matrix[r, c] for c in 1:3] for r in 1:3],
            "matrix_bits" => _matrix_to_bits(m.box_matrix),
        )
    end
    return Dict{String,Any}(
        "manifest_version" => m.manifest_version,
        "run_id" => m.run_id,
        "software_version" => m.software_version,
        "julia_version" => m.julia_version,
        "model_versions" => m.model_versions,
        "resolved_parameters" => m.resolved_parameters,
        "parameter_digest" => m.parameter_digest,
        "units" => m.units,
        "box" => box_obj,
        "precision" => m.precision,
        "algorithm" => String(m.algorithm),
        "dt" => m.dt,
        "dt_bits" => _float_to_hex(m.dt),
        "seed" => string(m.seed; base = 16, pad = 16),
        "n_atoms" => m.n_atoms,
        "temperature" => m.temperature,
        "temperature_bits" => _float_to_hex(m.temperature),
        "pressure" => m.pressure,
        "pressure_bits" => m.pressure === nothing ? nothing : _float_to_hex(m.pressure),
        "gamma" => m.gamma,
        "gamma_bits" => m.gamma === nothing ? nothing : _float_to_hex(m.gamma),
        "constraints_summary" => m.constraints_summary,
        "manifest_digest" => m.manifest_digest,
    )
end

function write_manifest_atomic(path::AbstractString, m::RunManifest;
                               before_rename_hook::Union{Nothing,Function} = nothing)
    verify_manifest(m)
    bytes = Vector{UInt8}(codeunits(JSON.json(_manifest_to_dict(m))))
    return _write_bytes_atomic(path, bytes; before_rename_hook = before_rename_hook)
end

function _decode_manifest_float(d::AbstractDict, key::String)
    bits_key = key * "_bits"
    if haskey(d, bits_key) && d[bits_key] !== nothing
        v_bits = _hex_to_float(String(d[bits_key]))
        if haskey(d, key) && d[key] !== nothing
            v_dec = Float64(d[key])
            v_dec == v_bits ||
                throw(ManifestError("inconsistent $key ($v_dec) and $bits_key ($v_bits) in manifest"))
        end
        return v_bits
    end
    return Float64(d[key])
end

function read_manifest(path::AbstractString)
    isfile(path) || throw(ManifestError("manifest file does not exist: $path"))
    d = try
        JSON.parse(read(path, String))
    catch e
        throw(ManifestError("failed to parse manifest JSON: $(sprint(showerror, e))"))
    end
    d isa AbstractDict || throw(ManifestError("manifest root must be a JSON object"))
    ver = String(get(d, "manifest_version", ""))
    ver == MANIFEST_VERSION || throw(ManifestError("unsupported manifest version: $ver"))

    box = _decode_box_dict(d["box"])
    bkind, bmat, blen = _box_summary(box)
    mv = Dict{String,String}(String(k) => String(v) for (k, v) in pairs(d["model_versions"]))
    rp = Dict{String,Any}(String(k) => _normalize_json_value(v) for (k, v) in pairs(d["resolved_parameters"]))
    dt64 = _decode_manifest_float(d, "dt")
    temp64 = _decode_manifest_float(d, "temperature")
    pres64 = d["pressure"] === nothing ? nothing : _decode_manifest_float(d, "pressure")
    gamma64 = d["gamma"] === nothing ? nothing : _decode_manifest_float(d, "gamma")
    seed64 = parse(UInt64, String(d["seed"]); base = 16)

    m = RunManifest(
        ver,
        String(d["run_id"]),
        String(d["software_version"]),
        String(d["julia_version"]),
        mv,
        rp,
        String(d["parameter_digest"]),
        String(d["units"]),
        bkind,
        bmat,
        blen,
        String(d["precision"]),
        Symbol(d["algorithm"]),
        dt64,
        seed64,
        Int(d["n_atoms"]),
        temp64,
        pres64,
        gamma64,
        String(d["constraints_summary"]),
        String(d["manifest_digest"]),
    )
    verify_manifest(m)
    return m
end

# ---------------------------------------------------------------------------
# Thermostat, Barostat, and RNG state for checkpoints
# ---------------------------------------------------------------------------

struct ThermostatState
    kind::Symbol       # :none, :langevin_baoab
    temperature::Float64
    gamma::Float64
    beta::Float64
    steps::Int
    extra::Dict{String,Float64}
end

ThermostatState(; kind::Symbol = :none, temperature::Real = 0.0,
                gamma::Real = 0.0, beta::Real = 0.0, steps::Integer = 0,
                extra::AbstractDict = Dict{String,Float64}()) =
    ThermostatState(kind, Float64(temperature), Float64(gamma), Float64(beta),
                    Int(steps), Dict{String,Float64}(String(k) => Float64(v) for (k, v) in pairs(extra)))

struct BarostatState
    kind::Symbol       # :none, :mc_volume
    pressure::Float64
    beta::Float64
    max_delta_log_volume::Float64
    attempts::Int
    accepts::Int
    last_delta_log_volume::Float64
end

BarostatState(; kind::Symbol = :none, pressure::Real = 0.0, beta::Real = 0.0,
              max_delta_log_volume::Real = 0.0, attempts::Integer = 0,
              accepts::Integer = 0, last_delta_log_volume::Real = 0.0) =
    BarostatState(kind, Float64(pressure), Float64(beta),
                  Float64(max_delta_log_volume), Int(attempts), Int(accepts),
                  Float64(last_delta_log_volume))

function encode_rng_state(rng::Random.Xoshiro)
    return Dict{String,Any}(
        "type" => "Xoshiro",
        "words" => [
            string(rng.s0; base = 16, pad = 16),
            string(rng.s1; base = 16, pad = 16),
            string(rng.s2; base = 16, pad = 16),
            string(rng.s3; base = 16, pad = 16),
            string(rng.s4; base = 16, pad = 16),
        ],
    )
end

function encode_rng_state(rng::Random.MersenneTwister)
    seed_vals = rng.seed isa Integer ? Int64[Int64(rng.seed)] : Int64[Int64(x) for x in rng.seed]
    return Dict{String,Any}(
        "type" => "MersenneTwister",
        "seed" => seed_vals,
        "seed_is_scalar" => (rng.seed isa Integer),
        "state_val" => String[string(reinterpret(UInt32, Int32(x)); base = 16, pad = 8) for x in rng.state.val],
        "vals_bits" => String[_float_to_hex(x) for x in rng.vals],
        "ints" => String[string(x; base = 16, pad = 32) for x in rng.ints],
        "idxF" => Int64(rng.idxF),
        "idxI" => Int64(rng.idxI),
        "adv" => Int64(rng.adv),
        "adv_jump" => string(rng.adv_jump; base = 16),
        "adv_vals" => Int64(rng.adv_vals),
        "adv_ints" => Int64(rng.adv_ints),
    )
end

function encode_rng_state(rng::Random.AbstractRNG)
    throw(CheckpointMismatchError("unsupported RNG type for bitwise checkpointing: $(typeof(rng))"))
end

function decode_rng_state(d::AbstractDict)
    rtype = String(get(d, "type", ""))
    if rtype == "Xoshiro"
        words = d["words"]
        length(words) == 5 || throw(CheckpointCorruptionError("Xoshiro state requires 5 words"))
        s = ntuple(i -> parse(UInt64, String(words[i]); base = 16), 5)
        return Random.Xoshiro(s[1], s[2], s[3], s[4], s[5])
    elseif rtype == "MersenneTwister"
        seed_arr = d["seed"]
        rng = get(d, "seed_is_scalar", true) ?
              Random.MersenneTwister(Int(seed_arr[1])) :
              Random.MersenneTwister(UInt32[UInt32(x) for x in seed_arr])
        sv = d["state_val"]
        length(sv) == length(rng.state.val) ||
            throw(CheckpointCorruptionError("MersenneTwister state_val length mismatch"))
        for i in eachindex(sv)
            rng.state.val[i] = reinterpret(Int32, parse(UInt32, String(sv[i]); base = 16))
        end
        vb = d["vals_bits"]
        length(vb) == length(rng.vals) ||
            throw(CheckpointCorruptionError("MersenneTwister vals length mismatch"))
        for i in eachindex(vb)
            rng.vals[i] = reinterpret(Float64, parse(UInt64, String(vb[i]); base = 16))
        end
        ib = d["ints"]
        length(ib) == length(rng.ints) ||
            throw(CheckpointCorruptionError("MersenneTwister ints length mismatch"))
        for i in eachindex(ib)
            rng.ints[i] = parse(UInt128, String(ib[i]); base = 16)
        end
        rng.idxF = Int(d["idxF"])
        rng.idxI = Int(d["idxI"])
        rng.adv = Int(d["adv"])
        rng.adv_jump = parse(BigInt, String(d["adv_jump"]); base = 16)
        rng.adv_vals = Int(d["adv_vals"])
        rng.adv_ints = Int(d["adv_ints"])
        return rng
    else
        throw(CheckpointCorruptionError("unknown RNG type in checkpoint: $rtype"))
    end
end

# ---------------------------------------------------------------------------
# Atomic versioned checkpoints
# ---------------------------------------------------------------------------

struct WorkflowCheckpoint
    format_version::Int
    stage::Symbol
    step::Int
    time::Float64
    manifest_digest::String
    restart_guarantee::Symbol    # :bitwise or :statistical
    q::Matrix{Float64}
    p::Matrix{Float64}
    F::Matrix{Float64}
    force_valid::Bool
    box::Types.AbstractBox
    rng_state::Union{Nothing,Dict{String,Any}}
    thermostat_state::ThermostatState
    barostat_state::BarostatState
    metadata::Dict{String,Any}
    payload_sha256::String
end

function _validate_checkpoint_fields(; format_version::Int, stage::Symbol, step::Int,
                                     time::Float64, restart_guarantee::Symbol,
                                     q::Matrix{Float64}, p::Matrix{Float64},
                                     F::Matrix{Float64},
                                     thermostat_state::ThermostatState,
                                     barostat_state::BarostatState)
    format_version == CHECKPOINT_FORMAT_VERSION ||
        throw(CheckpointVersionError(format_version, CHECKPOINT_FORMAT_VERSION))
    step >= 0 || throw(CheckpointMismatchError("checkpoint step must be non-negative, got $step"))
    isfinite(time) && time >= 0 ||
        throw(CheckpointMismatchError("checkpoint time must be finite and non-negative, got $time"))
    restart_guarantee in (:bitwise, :statistical) ||
        throw(CheckpointMismatchError("restart_guarantee must be :bitwise or :statistical, got :$restart_guarantee"))
    size(q, 1) == 3 && size(p) == size(q) && size(F) == size(q) ||
        throw(CheckpointMismatchError("q, p, F must have identical 3xN shape"))
    all(isfinite, q) && all(isfinite, p) && all(isfinite, F) ||
        throw(CheckpointMismatchError("checkpoint state contains non-finite coordinates, momenta, or forces"))
    isfinite(thermostat_state.temperature) && isfinite(thermostat_state.gamma) && isfinite(thermostat_state.beta) ||
        throw(CheckpointMismatchError("thermostat state contains non-finite numbers"))
    isfinite(barostat_state.pressure) && isfinite(barostat_state.beta) &&
        isfinite(barostat_state.max_delta_log_volume) && isfinite(barostat_state.last_delta_log_volume) ||
        throw(CheckpointMismatchError("barostat state contains non-finite numbers"))
    return true
end

function _checkpoint_payload_dict(stage::Symbol, step::Int, time::Float64,
                                  manifest_digest::String, restart_guarantee::Symbol,
                                  q::Matrix{Float64}, p::Matrix{Float64},
                                  F::Matrix{Float64}, force_valid::Bool,
                                  box::Types.AbstractBox,
                                  rng_state::Union{Nothing,Dict{String,Any}},
                                  th::ThermostatState, ba::BarostatState,
                                  metadata::Dict{String,Any})
    nat = size(q, 2)
    return Dict{String,Any}(
        "format_version" => CHECKPOINT_FORMAT_VERSION,
        "stage" => String(stage),
        "step" => step,
        "time" => time,
        "time_bits" => _float_to_hex(time),
        "manifest_digest" => manifest_digest,
        "restart_guarantee" => String(restart_guarantee),
        "n_atoms" => nat,
        "q_bits" => _matrix_to_bits(q),
        "p_bits" => _matrix_to_bits(p),
        "F_bits" => _matrix_to_bits(F),
        "force_valid" => force_valid,
        "box" => _encode_box_dict(box),
        "rng_state" => rng_state,
        "thermostat_state" => Dict{String,Any}(
            "kind" => String(th.kind),
            "temperature_bits" => _float_to_hex(th.temperature),
            "gamma_bits" => _float_to_hex(th.gamma),
            "beta_bits" => _float_to_hex(th.beta),
            "steps" => th.steps,
            "extra_bits" => Dict(k => _float_to_hex(v) for (k, v) in th.extra),
        ),
        "barostat_state" => Dict{String,Any}(
            "kind" => String(ba.kind),
            "pressure_bits" => _float_to_hex(ba.pressure),
            "beta_bits" => _float_to_hex(ba.beta),
            "max_delta_log_volume_bits" => _float_to_hex(ba.max_delta_log_volume),
            "attempts" => ba.attempts,
            "accepts" => ba.accepts,
            "last_delta_log_volume_bits" => _float_to_hex(ba.last_delta_log_volume),
        ),
        "metadata" => metadata,
    )
end

function make_checkpoint(; stage::Symbol,
                         step::Integer,
                         time::Real,
                         manifest::Union{RunManifest,AbstractString},
                         state::Dynamics.State,
                         box::Types.AbstractBox,
                         rng::Union{Nothing,Random.AbstractRNG} = nothing,
                         thermostat_state::ThermostatState = ThermostatState(),
                         barostat_state::BarostatState = BarostatState(),
                         restart_guarantee::Symbol = :bitwise,
                         metadata::AbstractDict = Dict{String,Any}())
    mdigest = manifest isa RunManifest ? manifest.manifest_digest : String(manifest)
    rng_dict = rng === nothing ? nothing :
               (restart_guarantee === :bitwise ? encode_rng_state(rng) :
                try
                    encode_rng_state(rng)
                catch
                    Dict{String,Any}("type" => string(typeof(rng)), "statistical_only" => true)
                end)
    if restart_guarantee === :bitwise && rng === nothing &&
       (thermostat_state.kind !== :none || barostat_state.kind !== :none)
        throw(CheckpointMismatchError("bitwise restart with thermostat/barostat requires an explicit RNG"))
    end
    q = copy(state.q); p = copy(state.p); F = copy(state.F)
    st_step = Int(step); st_time = Float64(time)
    meta = Dict{String,Any}(String(k) => _normalize_json_value(v) for (k, v) in pairs(metadata))
    _validate_checkpoint_fields(;
        format_version = CHECKPOINT_FORMAT_VERSION, stage = stage, step = st_step,
        time = st_time, restart_guarantee = restart_guarantee,
        q = q, p = p, F = F, thermostat_state = thermostat_state, barostat_state = barostat_state)
    payload = _checkpoint_payload_dict(stage, st_step, st_time, mdigest, restart_guarantee,
                                       q, p, F, state.force_valid, box, rng_dict,
                                       thermostat_state, barostat_state, meta)
    digest = sha256_hex(canonical_bytes(payload))
    return WorkflowCheckpoint(
        CHECKPOINT_FORMAT_VERSION, stage, st_step, st_time, mdigest, restart_guarantee,
        q, p, F, state.force_valid, box, deepcopy(rng_dict),
        thermostat_state, barostat_state, meta, digest,
    )
end

"""
    write_checkpoint_atomic(path, ckpt; before_rename_hook = nothing)

Atomically write a versioned checkpoint using a same-directory temp file + rename.
If writing or `before_rename_hook` fails, the temporary file is removed and any
existing file at `path` remains untouched.
"""
function write_checkpoint_atomic(path::AbstractString, ckpt::WorkflowCheckpoint;
                                 before_rename_hook::Union{Nothing,Function} = nothing)
    _validate_checkpoint_fields(;
        format_version = ckpt.format_version, stage = ckpt.stage, step = ckpt.step,
        time = ckpt.time, restart_guarantee = ckpt.restart_guarantee,
        q = ckpt.q, p = ckpt.p, F = ckpt.F,
        thermostat_state = ckpt.thermostat_state, barostat_state = ckpt.barostat_state)
    payload = _checkpoint_payload_dict(
        ckpt.stage, ckpt.step, ckpt.time, ckpt.manifest_digest, ckpt.restart_guarantee,
        ckpt.q, ckpt.p, ckpt.F, ckpt.force_valid, ckpt.box, ckpt.rng_state,
        ckpt.thermostat_state, ckpt.barostat_state, ckpt.metadata)
    digest = sha256_hex(canonical_bytes(payload))
    digest == ckpt.payload_sha256 ||
        throw(CheckpointCorruptionError("checkpoint payload_sha256 does not match contents"))
    envelope = Dict{String,Any}(
        "magic" => CHECKPOINT_MAGIC,
        "format_version" => ckpt.format_version,
        "payload_sha256" => digest,
        "payload" => payload,
    )
    bytes = Vector{UInt8}(codeunits(JSON.json(envelope)))
    dest = abspath(String(path))
    prev = dest * ".prev"
    had_primary = isfile(dest)
    if had_primary
        try
            _read_checkpoint_file(dest; expected_manifest_digest = nothing,
                                  expected_version = CHECKPOINT_FORMAT_VERSION, require_bitwise = false)
            _write_bytes_atomic(prev, read(dest); before_rename_hook = before_rename_hook)
        catch err
            if !(err isa CheckpointCorruptionError) || !isfile(prev)
                rethrow()
            end
            _read_checkpoint_file(prev; expected_manifest_digest = nothing,
                                  expected_version = CHECKPOINT_FORMAT_VERSION, require_bitwise = false)
        end
    end
    _write_bytes_atomic(dest, bytes; before_rename_hook = had_primary ? nothing : before_rename_hook)
    return (path = dest, sha256 = digest, bytes = length(bytes),
            format_version = ckpt.format_version, restart_guarantee = ckpt.restart_guarantee)
end

"""
    read_checkpoint(path; expected_manifest_digest=nothing, expected_version=CHECKPOINT_FORMAT_VERSION, require_bitwise=false)

Read and cryptographically verify a versioned checkpoint before returning deep
copies of state, box, RNG, thermostat, and barostat data.
"""
function _read_checkpoint_file(path::AbstractString;
                         expected_manifest_digest::Union{Nothing,AbstractString} = nothing,
                         expected_version::Int = CHECKPOINT_FORMAT_VERSION,
                         require_bitwise::Bool = false)
    isfile(path) || throw(CheckpointCorruptionError("checkpoint file not found: $path"))
    env = try
        JSON.parse(read(path, String))
    catch e
        throw(CheckpointCorruptionError("corrupt checkpoint JSON: $(sprint(showerror, e))"))
    end
    env isa AbstractDict || throw(CheckpointCorruptionError("checkpoint root must be a JSON object"))
    String(get(env, "magic", "")) == CHECKPOINT_MAGIC ||
        throw(CheckpointCorruptionError("invalid checkpoint magic header"))
    haskey(env, "format_version") && haskey(env, "payload") && haskey(env, "payload_sha256") ||
        throw(CheckpointCorruptionError("missing required checkpoint envelope fields"))
    ver = Int(env["format_version"])
    ver == expected_version || throw(CheckpointVersionError(ver, expected_version))

    p = env["payload"]
    p isa AbstractDict || throw(CheckpointCorruptionError("checkpoint payload must be a JSON object"))
    Int(get(p, "format_version", -1)) == ver ||
        throw(CheckpointCorruptionError("envelope/payload format_version mismatch"))

    stored_sha = String(env["payload_sha256"])
    recomputed_sha = try
        sha256_hex(canonical_bytes(_normalize_json_value(p)))
    catch e
        throw(CheckpointCorruptionError("failed canonical checkpoint hash check: $(sprint(showerror, e))"))
    end
    stored_sha == recomputed_sha ||
        throw(CheckpointCorruptionError("checkpoint SHA-256 mismatch (corrupted or tampered file)"))

    mdigest = String(p["manifest_digest"])
    if expected_manifest_digest !== nothing && mdigest != String(expected_manifest_digest)
        throw(CheckpointMismatchError("checkpoint manifest_digest $mdigest != expected $expected_manifest_digest"))
    end
    guarantee = Symbol(p["restart_guarantee"])
    guarantee in (:bitwise, :statistical) ||
        throw(CheckpointCorruptionError("invalid restart_guarantee: $guarantee"))
    if require_bitwise && guarantee !== :bitwise
        throw(CheckpointMismatchError("checkpoint only provides :$guarantee restart, but require_bitwise=true"))
    end

    nat = Int(p["n_atoms"])
    nat > 0 || throw(CheckpointCorruptionError("checkpoint n_atoms must be positive"))
    q = _bits_to_matrix(p["q_bits"], 3, nat)
    p_mat = _bits_to_matrix(p["p_bits"], 3, nat)
    F = _bits_to_matrix(p["F_bits"], 3, nat)
    force_valid = Bool(p["force_valid"])
    box = _decode_box_dict(p["box"])
    time_val = _hex_to_float(String(p["time_bits"]))
    step_val = Int(p["step"])

    thd = p["thermostat_state"]
    th = ThermostatState(
        Symbol(thd["kind"]),
        _hex_to_float(String(thd["temperature_bits"])),
        _hex_to_float(String(thd["gamma_bits"])),
        _hex_to_float(String(thd["beta_bits"])),
        Int(thd["steps"]),
        Dict{String,Float64}(String(k) => _hex_to_float(String(v)) for (k, v) in pairs(thd["extra_bits"])),
    )

    bad = p["barostat_state"]
    ba = BarostatState(
        Symbol(bad["kind"]),
        _hex_to_float(String(bad["pressure_bits"])),
        _hex_to_float(String(bad["beta_bits"])),
        _hex_to_float(String(bad["max_delta_log_volume_bits"])),
        Int(bad["attempts"]),
        Int(bad["accepts"]),
        _hex_to_float(String(bad["last_delta_log_volume_bits"])),
    )

    rng_raw = p["rng_state"] === nothing ? nothing :
              Dict{String,Any}(String(k) => _normalize_json_value(v) for (k, v) in pairs(p["rng_state"]))
    if require_bitwise && rng_raw !== nothing && get(rng_raw, "statistical_only", false) == true
        throw(CheckpointMismatchError("checkpoint RNG state is marked statistical_only"))
    end

    meta = Dict{String,Any}(String(k) => _normalize_json_value(v) for (k, v) in pairs(p["metadata"]))

    return WorkflowCheckpoint(
        ver, Symbol(p["stage"]), step_val, time_val, mdigest, guarantee,
        q, p_mat, F, force_valid, box, rng_raw, th, ba, meta, stored_sha,
    )
end

function read_checkpoint(path::AbstractString; expected_manifest_digest::Union{Nothing,AbstractString} = nothing,
                         expected_version::Int = CHECKPOINT_FORMAT_VERSION, require_bitwise::Bool = false)
    try
        return _read_checkpoint_file(path; expected_manifest_digest, expected_version, require_bitwise)
    catch err
        err isa CheckpointCorruptionError || rethrow()
        prev = String(path) * ".prev"
        isfile(prev) || rethrow()
        return _read_checkpoint_file(prev; expected_manifest_digest, expected_version, require_bitwise)
    end
end

"""Reconstruct `(Dynamics.State, Types.AbstractBox, rng)` from a verified checkpoint."""
function restore_state(ckpt::WorkflowCheckpoint)
    st = Dynamics.State(copy(ckpt.q), copy(ckpt.p), copy(ckpt.F), ckpt.force_valid)
    rng = if ckpt.rng_state === nothing || get(ckpt.rng_state, "statistical_only", false) == true
        nothing
    else
        decode_rng_state(ckpt.rng_state)
    end
    return st, ckpt.box, rng
end

# ---------------------------------------------------------------------------
# Trajectory metadata & reader/writer
# ---------------------------------------------------------------------------

mutable struct TrajectoryWriter
    xyz_path::String
    meta_path::String
    xyz::IOStream
    metadata::IOStream
    labels::Vector{String}
    n::Int
    units::String
    manifest_digest::String
    closed::Bool
end

function TrajectoryWriter(path::AbstractString, n::Integer;
                          labels = fill("X", n),
                          units::Union{Types.Units,AbstractString} = Types.REDUCED,
                          manifest_digest::AbstractString = "")
    n > 0 || throw(TrajectoryError("frame atom count must be positive"))
    length(labels) == n || throw(TrajectoryError("one label is required per atom"))
    xyzpath = endswith(lowercase(String(path)), ".xyz") ? String(path) : String(path) * ".xyz"
    metapath = xyzpath * ".jsonl"
    mkpath(dirname(abspath(xyzpath)))
    uname = units isa Types.Units ? units.name : String(units)
    return TrajectoryWriter(
        xyzpath, metapath,
        open(xyzpath, "w"), open(metapath, "w"),
        String.(labels), Int(n), uname, String(manifest_digest), false,
    )
end

function write_trajectory_frame!(w::TrajectoryWriter, q::AbstractMatrix{<:Real},
                                 box::Types.AbstractBox, step::Integer, time::Real)
    w.closed && throw(TrajectoryError("trajectory writer is closed"))
    size(q) == (3, w.n) || throw(TrajectoryError("coordinate shape $(size(q)) != (3, $(w.n))"))
    all(isfinite, q) || throw(TrajectoryError("cannot write non-finite coordinates"))
    t = Float64(time)
    isfinite(t) && t >= 0 || throw(TrajectoryError("frame time must be finite and non-negative, got $time"))
    st = Int(step)
    st >= 0 || throw(TrajectoryError("frame step must be non-negative, got $step"))

    bkind, bmat, blen = _box_summary(box)
    length_label = w.units == Types.REDUCED.name ? "reduced_length" : "internal_length_unit"
    time_label = w.units == Types.REDUCED.name ? "reduced_time" : "internal_time_unit"
    write_extxyz_frame(w.xyz; positions=q, species=w.labels, time=t,
                       box_matrix=bmat, length_unit=length_label, time_unit=time_label)
    rec = Dict{String,Any}(
        "step" => st,
        "time" => t,
        "box_kind" => String(bkind),
        "box_lengths" => blen === nothing ? nothing : collect(blen),
        "box_matrix" => bmat === nothing ? nothing : [[bmat[r, c] for c in 1:3] for r in 1:3],
        "units" => w.units,
        "manifest_digest" => w.manifest_digest,
    )
    println(w.metadata, JSON.json(rec))
    return w
end

function Base.close(w::TrajectoryWriter)
    w.closed && return nothing
    flush(w.xyz); flush(w.metadata)
    close(w.xyz); close(w.metadata)
    w.closed = true
    return nothing
end

struct TrajectoryData
    labels::Vector{String}
    frames::Vector{Matrix{Float64}}
    steps::Vector{Int}
    times::Vector{Float64}
    boxes::Vector{Types.AbstractBox}
    units::Vector{String}
    manifest_digests::Vector{String}
end

function _parse_metadata_box(rec::AbstractDict)
    bkind = haskey(rec, "box_kind") ? String(rec["box_kind"]) :
            (haskey(rec, "box") && rec["box"] !== nothing ? "orthorhombic" : "none")
    if bkind == "none"
        return Types.NoBox()
    elseif bkind == "orthorhombic"
        Ls = haskey(rec, "box_lengths") && rec["box_lengths"] !== nothing ?
             rec["box_lengths"] : get(rec, "box", nothing)
        Ls !== nothing && length(Ls) == 3 ||
            throw(TrajectoryError("orthorhombic trajectory metadata missing 3 box lengths"))
        box = Types.OrthorhombicBox((Float64(Ls[1]), Float64(Ls[2]), Float64(Ls[3])))
        if haskey(rec, "box_matrix") && rec["box_matrix"] !== nothing
            bm = rec["box_matrix"]
            length(bm) == 3 || throw(TrajectoryError("box_matrix must be 3x3"))
            H = [Float64(bm[r][c]) for r in 1:3, c in 1:3]
            isapprox(H, Diagonal(collect(box.L)); atol = 1e-12, rtol = 1e-12) ||
                throw(TrajectoryError("inconsistent box_lengths and box_matrix in trajectory metadata"))
        end
        return box
    elseif bkind == "triclinic"
        bm = get(rec, "box_matrix", nothing)
        bm !== nothing && length(bm) == 3 ||
            throw(TrajectoryError("triclinic trajectory metadata missing 3x3 box_matrix"))
        H = zeros(3, 3)
        for r in 1:3
            length(bm[r]) == 3 || throw(TrajectoryError("box_matrix row $r must have 3 entries"))
            for c in 1:3
                H[r, c] = Float64(bm[r][c])
            end
        end
        return Types.TriclinicBox(H)
    else
        throw(TrajectoryError("unknown box_kind in trajectory metadata: $bkind"))
    end
end

"""Read an XYZ trajectory paired with its `.jsonl` step/time/box metadata sidecar."""
function read_trajectory(path::AbstractString)
    xyzpath = endswith(lowercase(String(path)), ".xyz") ? String(path) : String(path) * ".xyz"
    metapath = xyzpath * ".jsonl"
    isfile(xyzpath) || throw(TrajectoryError("XYZ trajectory not found: $xyzpath"))
    isfile(metapath) || throw(TrajectoryError("trajectory metadata sidecar not found: $metapath"))

    first_two = open(xyzpath, "r") do io
        l1 = eof(io) ? "" : readline(io)
        l2 = eof(io) ? "" : readline(io)
        (l1, l2)
    end
    use_extxyz = occursin("Properties=species:S:1:pos:R:3", first_two[2])
    labels = String[]
    frames = Matrix{Float64}[]
    if use_extxyz
        ext_frames = try
            open(read_extxyz_frames, xyzpath, "r")
        catch e
            throw(TrajectoryError("invalid extxyz trajectory: $(sprint(showerror, e))"))
        end
        !isempty(ext_frames) || throw(TrajectoryError("trajectory contains zero frames"))
        labels = ext_frames[1].species
        nat = length(labels)
        for (i, ef) in enumerate(ext_frames)
            length(ef.species) == nat && size(ef.positions) == (3, nat) ||
                throw(TrajectoryError("frame $i atom count mismatch"))
            all(isfinite, ef.positions) || throw(TrajectoryError("frame $i contains non-finite coordinates"))
            push!(frames, ef.positions)
        end
    else
        raw_frames = Dynamics.read_xyz_frames(xyzpath)
        !isempty(raw_frames) || throw(TrajectoryError("trajectory contains zero frames"))
        labels = raw_frames[1][1]
        nat = length(labels)
        for (i, (lbls, q)) in enumerate(raw_frames)
            length(lbls) == nat && size(q) == (3, nat) ||
                throw(TrajectoryError("frame $i atom count mismatch"))
            all(isfinite, q) || throw(TrajectoryError("frame $i contains non-finite coordinates"))
            push!(frames, q)
        end
    end

    lines = [l for l in eachline(metapath) if !isempty(strip(l))]
    length(lines) == length(frames) ||
        throw(TrajectoryError("metadata record count ($(length(lines))) != frame count ($(length(frames)))"))

    steps = Int[]
    times = Float64[]
    boxes = Types.AbstractBox[]
    units_vec = String[]
    digests = String[]
    for (idx, line) in enumerate(lines)
        rec = try
            JSON.parse(line)
        catch e
            throw(TrajectoryError("malformed metadata JSON at line $idx: $(sprint(showerror, e))"))
        end
        rec isa AbstractDict || throw(TrajectoryError("metadata line $idx must be a JSON object"))
        haskey(rec, "step") && haskey(rec, "time") ||
            throw(TrajectoryError("metadata line $idx missing required step/time fields"))
        st = Int(rec["step"])
        t = Float64(rec["time"])
        isfinite(t) && t >= 0 || throw(TrajectoryError("invalid frame time $t at line $idx"))
        if !isempty(times) && t <= times[end]
            throw(TrajectoryError("non-increasing frame time at line $idx: $t <= $(times[end])"))
        end
        push!(steps, st)
        push!(times, t)
        push!(boxes, _parse_metadata_box(rec))
        push!(units_vec, String(get(rec, "units", "")))
        push!(digests, String(get(rec, "manifest_digest", "")))
    end

    return TrajectoryData(labels, frames, steps, times, boxes, units_vec, digests)
end

# ---------------------------------------------------------------------------
# Strict analysis guards & MSD / diffusion / observable analyzer
# ---------------------------------------------------------------------------

struct AnalysisConfig
    min_frames::Int
    fit_window::Tuple{Float64,Float64}
    lag_window::Union{Nothing,Tuple{Float64,Float64}}
    min_fit_points::Int
    min_time_span::Float64
    unwrap::Bool
    max_step_half_box_fraction::Float64
    min_ess::Float64
    dt_tol::Float64
    refuse_degenerate_series::Bool
end

function AnalysisConfig(; min_frames::Integer = 20,
                        fit_window::Tuple{Real,Real} = (0.10, 0.30),
                        lag_window::Union{Nothing,Tuple{Real,Real}} = nothing,
                        min_fit_points::Integer = 5,
                        min_time_span::Real = 0.0,
                        unwrap::Bool = true,
                        max_step_half_box_fraction::Real = 0.5,
                        min_ess::Real = 5.0,
                        dt_tol::Real = 1e-6,
                        refuse_degenerate_series::Bool = false)
    mf = Int(min_frames)
    mf >= 4 || throw(AnalysisGuardError("min_frames must be >= 4, got $mf"))
    fw = (Float64(fit_window[1]), Float64(fit_window[2]))
    lw = lag_window === nothing ? nothing : (Float64(lag_window[1]), Float64(lag_window[2]))
    mfp = Int(min_fit_points)
    mfp >= 2 || throw(AnalysisGuardError("min_fit_points must be >= 2, got $mfp"))
    mts = Float64(min_time_span)
    isfinite(mts) && mts >= 0 || throw(AnalysisGuardError("min_time_span must be finite and >= 0"))
    mhb = Float64(max_step_half_box_fraction)
    isfinite(mhb) && 0 < mhb <= 0.5 ||
        throw(AnalysisGuardError("max_step_half_box_fraction must be in (0, 0.5], got $mhb"))
    mess = Float64(min_ess)
    isfinite(mess) && mess > 0 || throw(AnalysisGuardError("min_ess must be finite and > 0"))
    dtol = Float64(dt_tol)
    isfinite(dtol) && dtol >= 0 || throw(AnalysisGuardError("dt_tol must be finite and >= 0"))

    return AnalysisConfig(mf, fw, lw, mfp, mts, unwrap, mhb, mess, dtol, refuse_degenerate_series)
end

struct AnalysisResult
    n_frames::Int
    n_atoms::Int
    total_time::Float64
    dt_frame::Float64
    msd_lags::Vector{Float64}
    msd_values::Vector{Float64}
    fit_lag_indices::UnitRange{Int}
    fit_slope::Float64
    fit_intercept::Float64
    diffusion_coefficient::Float64
    fit_r2::Float64
    max_step_half_box_fraction::Float64
    unwrapped::Bool
    energy_summary::Union{Nothing,NamedTuple}
    warnings::Vector{String}
    caveat::String
end

function _validate_fit_window(config::AnalysisConfig, lags::Vector{Float64})
    n_lags = length(lags)
    if config.lag_window !== nothing
        lo, hi = config.lag_window
        (isfinite(lo) && isfinite(hi)) ||
            throw(AnalysisGuardError("lag_window bounds must be finite"))
        (lo > 0.0 && hi > 0.0) ||
            throw(AnalysisGuardError("lag_window bounds must be positive, got ($lo, $hi)"))
        hi > lo ||
            throw(AnalysisGuardError("lag_window upper bound must exceed lower bound, got ($lo, $hi)"))
        tol = 16eps(max(abs(lo), abs(hi), abs(lags[1]), abs(lags[end]), 1.0))
        lo >= lags[1] - tol && hi <= lags[end] + tol ||
            throw(AnalysisGuardError("lag_window ($lo, $hi) lies outside available lag range [$(lags[1]), $(lags[end])]"))
        idx_lo = max(1, searchsortedfirst(lags, lo - tol))
        idx_hi = min(n_lags, searchsortedlast(lags, hi + tol))
        idx_hi - idx_lo + 1 >= config.min_fit_points ||
            throw(AnalysisGuardError("short MSD fit window: only $(max(0, idx_hi - idx_lo + 1)) points in [$lo, $hi], require >= $(config.min_fit_points)"))
        return idx_lo:idx_hi
    else
        f_lo, f_hi = config.fit_window
        (isfinite(f_lo) && isfinite(f_hi)) ||
            throw(AnalysisGuardError("fit_window bounds must be finite"))
        (f_lo >= 0.0 && f_hi > 0.0) ||
            throw(AnalysisGuardError("fit_window cannot be negative or non-positive, got ($f_lo, $f_hi)"))
        f_hi > f_lo ||
            throw(AnalysisGuardError("fit_window upper fraction must exceed lower fraction, got ($f_lo, $f_hi)"))
        f_hi <= 1.0 ||
            throw(AnalysisGuardError("fit_window upper fraction must be <= 1.0, got $f_hi"))
        idx_lo = max(1, ceil(Int, f_lo * n_lags))
        idx_hi = min(n_lags, floor(Int, f_hi * n_lags))
        count = idx_hi - idx_lo + 1
        count >= config.min_fit_points ||
            throw(AnalysisGuardError("short MSD fit window ($f_lo, $f_hi): $count lag points < min_fit_points ($(config.min_fit_points))"))
        return idx_lo:idx_hi
    end
end

"""
    unwrap_frames(frames, boxes; max_half_box_fraction = 0.5)

Unwrap periodic trajectory frames by integrating frame-to-frame minimum-image
displacements. Refuses `NoBox` and refuses ambiguous displacements exceeding
`max_half_box_fraction` of the half box width (`0.5` in fractional coordinates).
"""
function unwrap_frames(frames::AbstractVector{<:AbstractMatrix{Float64}},
                       boxes::AbstractVector{<:Types.AbstractBox};
                       max_half_box_fraction::Float64 = 0.5)
    nframes = length(frames)
    nframes >= 2 || throw(AnalysisGuardError("need at least 2 frames to unwrap"))
    length(boxes) == nframes || throw(AnalysisGuardError("box list length must match frame count"))
    nat = size(frames[1], 2)
    unwrapped = [copy(frames[1])]
    max_frac = 0.0

    for k in 2:nframes
        b = boxes[k]
        b isa Types.NoBox &&
            throw(AnalysisGuardError("cannot unwrap periodic trajectory when frame $k has NoBox; pass unwrap=false for non-periodic coordinates"))
        size(frames[k]) == (3, nat) ||
            throw(AnalysisGuardError("frame $k shape mismatch"))
        prev = frames[k - 1]
        curr = frames[k]
        prev_u = unwrapped[k - 1]
        next_u = Matrix{Float64}(undef, 3, nat)
        if b isa Types.OrthorhombicBox
            L = b.L
            @inbounds for j in 1:nat
                for d in 1:3
                    raw = curr[d, j] - prev[d, j]
                    s = raw / L[d]
                    step_s = s - round(s)
                    frac_half = abs(step_s) / 0.5
                    frac_half > max_frac && (max_frac = frac_half)
                    abs(step_s) >= max_half_box_fraction &&
                        throw(AnalysisGuardError("ambiguous periodic unwrap at frame $k atom $j axis $d: fractional step |$(round(step_s, digits=4))| >= $max_half_box_fraction"))
                    next_u[d, j] = prev_u[d, j] + step_s * L[d]
                end
            end
        elseif b isa Types.TriclinicBox
            H = b.H
            Hinv = b.Hinv
            @inbounds for j in 1:nat
                dr = curr[:, j] .- prev[:, j]
                ds = Hinv * dr
                step_s = ds .- round.(ds)
                for d in 1:3
                    frac_half = abs(step_s[d]) / 0.5
                    frac_half > max_frac && (max_frac = frac_half)
                    abs(step_s[d]) >= max_half_box_fraction &&
                        throw(AnalysisGuardError("ambiguous periodic unwrap at frame $k atom $j axis $d: fractional step |$(round(step_s[d], digits=4))| >= $max_half_box_fraction"))
                end
                next_u[:, j] = prev_u[:, j] .+ H * step_s
            end
        else
            throw(AnalysisGuardError("unsupported box type $(typeof(b))"))
        end
        push!(unwrapped, next_u)
    end
    return unwrapped, max_frac
end

function _compute_all_origin_msd(frames::AbstractVector{<:AbstractMatrix{Float64}},
                                 dt_frame::Float64)
    nframes = length(frames)
    nat = size(frames[1], 2)
    n_lags = nframes - 1
    lags = [k * dt_frame for k in 1:n_lags]
    msd = Vector{Float64}(undef, n_lags)
    @inbounds for k in 1:n_lags
        total = 0.0
        n_origins = nframes - k
        for t in 1:n_origins
            q0 = frames[t]
            q1 = frames[t + k]
            for j in 1:nat
                dx = q1[1, j] - q0[1, j]
                dy = q1[2, j] - q0[2, j]
                dz = q1[3, j] - q0[3, j]
                total += dx * dx + dy * dy + dz * dz
            end
        end
        msd[k] = total / (n_origins * nat)
    end
    return lags, msd
end

function _ols_fit(x::AbstractVector{Float64}, y::AbstractVector{Float64})
    n = length(x)
    n >= 2 || throw(AnalysisGuardError("OLS fit requires at least 2 points"))
    mx = mean(x); my = mean(y)
    sxx = sum(abs2, x .- mx)
    sxx > 0 || throw(AnalysisGuardError("degenerate lag span in MSD fit"))
    sxy = sum((x[i] - mx) * (y[i] - my) for i in 1:n)
    slope = sxy / sxx
    intercept = my - slope * mx
    sst = sum(abs2, y .- my)
    ssr = sum(abs2, y[i] - (slope * x[i] + intercept) for i in 1:n)
    r2 = sst == 0.0 ? (ssr == 0.0 ? 1.0 : 0.0) : 1.0 - ssr / sst
    return slope, intercept, r2
end

"""
    analyze(traj::TrajectoryData; config = AnalysisConfig(), energies = nothing)
    analyze(frames, times, boxes; config = AnalysisConfig(), energies = nothing)

Run guarded trajectory analysis (MSD, Einstein self-diffusion slope, and optional
correlated energy diagnostics). Refuses insufficient frames, non-monotonic or
non-uniform timestamps, negative/short/invalid fit windows, ambiguous periodic
jumps, and insufficient ESS.
"""
function analyze(frames::AbstractVector{<:AbstractMatrix{Float64}},
                 times::AbstractVector{<:Real},
                 boxes::AbstractVector{<:Types.AbstractBox};
                 config::AnalysisConfig = AnalysisConfig(),
                 energies::Union{Nothing,AbstractVector{<:Real}} = nothing)
    nframes = length(frames)
    nframes >= config.min_frames ||
        throw(AnalysisGuardError("insufficient trajectory sampling: $nframes frames < min_frames=$(config.min_frames)"))
    length(times) == nframes ||
        throw(AnalysisGuardError("times length ($(length(times))) != frame count ($nframes)"))
    length(boxes) == nframes ||
        throw(AnalysisGuardError("boxes length ($(length(boxes))) != frame count ($nframes)"))

    nat = size(frames[1], 2)
    nat > 0 || throw(AnalysisGuardError("trajectory has zero atoms"))
    for (i, q) in enumerate(frames)
        size(q) == (3, nat) || throw(AnalysisGuardError("frame $i shape mismatch"))
        all(isfinite, q) || throw(AnalysisGuardError("frame $i contains non-finite coordinates"))
    end

    ts = Float64.(times)
    all(isfinite, ts) || throw(AnalysisGuardError("non-finite timestamp in trajectory"))
    dts = diff(ts)
    all(d -> d > 0, dts) || throw(AnalysisGuardError("timestamps must be strictly increasing"))
    dt_frame = mean(dts)
    max_dt_dev = maximum(abs.(dts .- dt_frame)) / dt_frame
    max_dt_dev <= config.dt_tol ||
        throw(AnalysisGuardError("frame time spacing is non-uniform (max relative deviation $(round(max_dt_dev, sigdigits=3)) > dt_tol=$(config.dt_tol))"))

    total_time = ts[end] - ts[1]
    total_time >= config.min_time_span ||
        throw(AnalysisGuardError("insufficient total trajectory duration $total_time < min_time_span=$(config.min_time_span)"))

    unwrapped_frames, max_frac = if config.unwrap
        unwrap_frames(frames, boxes; max_half_box_fraction = config.max_step_half_box_fraction)
    else
        ([copy(f) for f in frames], 0.0)
    end

    lags, msd = _compute_all_origin_msd(unwrapped_frames, dt_frame)
    all(isfinite, msd) || throw(AnalysisGuardError("non-finite MSD values encountered"))
    fit_idx = _validate_fit_window(config, lags)
    slope, intercept, r2 = _ols_fit(view(lags, fit_idx), view(msd, fit_idx))
    D = slope / 6.0

    warnings = String[]
    D < 0 && push!(warnings, "fitted MSD slope is negative (D=$D); trajectory may be trapped or under-sampled")

    esum = nothing
    if energies !== nothing
        ev = Float64.(energies)
        length(ev) == nframes ||
            throw(AnalysisGuardError("energies length ($(length(ev))) != frame count ($nframes)"))
        all(isfinite, ev) || throw(AnalysisGuardError("non-finite values in energy series"))
        v_e = var(ev)
        if config.refuse_degenerate_series && v_e == 0.0
            throw(AnalysisGuardError("energy series has zero variance (degenerate/stuck trajectory)"))
        end
        est = Sampling.correlated_mean(ev)
        est.n_eff >= config.min_ess ||
            throw(AnalysisGuardError("insufficient effective sample size for energy series: ESS=$(round(est.n_eff, digits=2)) < min_ess=$(config.min_ess)"))
        drift = Dynamics.energy_drift_report(ts, ev)
        esum = (
            mean = est.mean,
            se = est.se,
            tau_int = est.tau,
            ess = est.n_eff,
            n = est.n,
            max_abs_dev = drift.max_abs_dev,
            slope = drift.slope,
        )
    end

    caveat = "Finite-time all-origin MSD fit over lag indices $(first(fit_idx)):$(last(fit_idx)); " *
             "no hydrodynamic finite-size correction or asymptotic Einstein-regime proof is claimed."

    return AnalysisResult(
        nframes, nat, total_time, dt_frame, lags, msd, fit_idx,
        slope, intercept, D, r2, max_frac, config.unwrap, esum, warnings, caveat,
    )
end

function analyze(traj::TrajectoryData;
                 config::AnalysisConfig = AnalysisConfig(),
                 energies::Union{Nothing,AbstractVector{<:Real}} = nothing)
    return analyze(traj.frames, traj.times, traj.boxes; config = config, energies = energies)
end

# ---------------------------------------------------------------------------
# Transactional Workflow Pipeline:
# validate -> prepare -> minimize -> equilibrate -> produce -> analyze
# ---------------------------------------------------------------------------

struct WorkflowConfig
    run_id::String
    algorithm::Symbol
    fast_factors::Vector{Any}
    respa_m::Int
    dt::Float64
    seed::UInt64
    temperature::Float64
    pressure::Union{Nothing,Float64}
    gamma::Union{Nothing,Float64}
    max_delta_log_volume::Float64
    barostat_every::Int
    minimize_max_steps::Int
    minimize_grad_tol::Float64
    minimize_step_size::Float64
    equilibrate_steps::Int
    produce_steps::Int
    frame_every::Int
    checkpoint_every::Int
    restart_guarantee::Symbol
    contact_tol::Float64
    analysis::AnalysisConfig
    model_versions::Dict{String,String}
    resolved_parameters::Dict{String,Any}
    molecules::Union{Nothing,Vector{Vector{Int}}}
    barostat_coupling::Symbol
    surface_tension::Float64
end

function WorkflowConfig(; run_id::AbstractString = "workflow-001",
                         algorithm::Symbol = :langevin_baoab,
                         fast_factors = Any[],
                         respa_m::Integer = 1,
                        dt::Real = 0.005,
                        seed::Integer = 0x0000000000000001,
                        temperature::Real = 1.0,
                        pressure::Union{Nothing,Real} = nothing,
                        gamma::Union{Nothing,Real} = 1.0,
                        max_delta_log_volume::Real = 0.05,
                        barostat_every::Integer = 10,
                        minimize_max_steps::Integer = 50,
                        minimize_grad_tol::Real = 1e-3,
                        minimize_step_size::Real = 1e-3,
                        equilibrate_steps::Integer = 100,
                        produce_steps::Integer = 400,
                        frame_every::Integer = 10,
                        checkpoint_every::Integer = 100,
                        restart_guarantee::Symbol = :bitwise,
                        contact_tol::Real = 0.5,
                        analysis::AnalysisConfig = AnalysisConfig(),
                        model_versions::AbstractDict = Dict("force_field" => "FoundationCore-0.1.0"),
                        resolved_parameters::AbstractDict = Dict{String,Any}(),
                        molecules = nothing,
                        barostat_coupling::Symbol = :isotropic,
                        surface_tension::Real = 0.0)
    algorithm in SUPPORTED_ALGORITHMS ||
        throw(ValidationError("unsupported algorithm :$algorithm"))
    respa_m > 0 || throw(ValidationError("respa_m must be a positive integer"))
    algorithm === :respa_nve && (pressure !== nothing || temperature != 0 || gamma !== nothing) &&
        throw(ValidationError("RESPA requires NVE settings (temperature=0, pressure=nothing, gamma=nothing)"))
    dt64 = Float64(dt); isfinite(dt64) && dt64 > 0 || throw(ValidationError("dt must be positive and finite"))
    temp64 = Float64(temperature); isfinite(temp64) && temp64 >= 0 || throw(ValidationError("temperature must be finite and >= 0"))
    pres64 = pressure === nothing ? nothing : Float64(pressure)
    pres64 === nothing || isfinite(pres64) || throw(ValidationError("pressure must be finite"))
    gamma64 = gamma === nothing ? nothing : Float64(gamma)
    gamma64 === nothing || (isfinite(gamma64) && gamma64 >= 0) || throw(ValidationError("gamma must be finite and >= 0"))
    algorithm === :npt_baoab_mc && pres64 === nothing &&
        throw(ValidationError("algorithm :npt_baoab_mc requires an explicit finite pressure"))
    restart_guarantee in (:bitwise, :statistical) ||
        throw(ValidationError("restart_guarantee must be :bitwise or :statistical"))
    Int(minimize_max_steps) >= 0 || throw(ValidationError("minimize_max_steps must be >= 0"))
    Int(equilibrate_steps) >= 0 || throw(ValidationError("equilibrate_steps must be >= 0"))
    Int(produce_steps) > 0 || throw(ValidationError("produce_steps must be > 0"))
    Int(frame_every) > 0 || throw(ValidationError("frame_every must be > 0"))
    Int(checkpoint_every) >= 0 || throw(ValidationError("checkpoint_every must be >= 0"))
    Int(barostat_every) > 0 || throw(ValidationError("barostat_every must be > 0"))
    Float64(contact_tol) >= 0 || throw(ValidationError("contact_tol must be >= 0"))

    (barostat_coupling === :isotropic || barostat_coupling === :semi_isotropic_xy || barostat_coupling === :anisotropic) ||
        throw(ValidationError("barostat_coupling must be :isotropic, :semi_isotropic_xy, or :anisotropic"))
    st64 = Float64(surface_tension)
    isfinite(st64) || throw(ValidationError("surface_tension must be finite"))
    mols = molecules === nothing ? nothing : Vector{Int}[collect(Int, m) for m in molecules]

    return WorkflowConfig(
        String(run_id), algorithm, Any[fast_factors...], Int(respa_m), dt64, UInt64(seed), temp64, pres64, gamma64,
        Float64(max_delta_log_volume), Int(barostat_every),
        Int(minimize_max_steps), Float64(minimize_grad_tol), Float64(minimize_step_size),
        Int(equilibrate_steps), Int(produce_steps), Int(frame_every), Int(checkpoint_every),
        restart_guarantee, Float64(contact_tol), analysis,
        Dict{String,String}(String(k) => String(v) for (k, v) in pairs(model_versions)),
        Dict{String,Any}(String(k) => _normalize_json_value(v) for (k, v) in pairs(resolved_parameters)),
        mols, barostat_coupling, st64
    )
end

struct StageRecord
    stage::Symbol
    status::Symbol
    duration_s::Float64
    detail::String
    checkpoint_path::Union{Nothing,String}
end

struct WorkflowResult
    manifest::RunManifest
    stages::Vector{StageRecord}
    final_state::Dynamics.State
    final_box::Types.AbstractBox
    final_checkpoint_path::Union{Nothing,String}
    trajectory_path::String
    analysis::AnalysisResult
    committed::Bool
end

function _min_pair_distance(q::AbstractMatrix{Float64})
    nat = size(q, 2)
    nat < 2 && return Inf
    mind = Inf
    @inbounds for i in 1:nat-1, j in i+1:nat
        dx = q[1, i] - q[1, j]
        dy = q[2, i] - q[2, j]
        dz = q[3, i] - q[3, j]
        d = sqrt(dx * dx + dy * dy + dz * dz)
        d < mind && (mind = d)
    end
    return mind
end

"""Stage 1 (`validate`): check structure/system, force-model consistency, and finite initial forces."""
function validate_stage(sys::Types.System, factors, config::WorkflowConfig;
                        constraints = nothing)
    nat = size(sys.atoms.q, 2)
    nat > 0 || throw(ValidationError("system has zero atoms"))
    if config.contact_tol > 0 && nat >= 2
        dmin = _min_pair_distance(sys.atoms.q)
        dmin >= config.contact_tol ||
            throw(ValidationError("close contact detected ($dmin < contact_tol=$(config.contact_tol))"))
    end
    if config.algorithm === :npt_baoab_mc
        sys.box isa Types.OrthorhombicBox ||
            throw(ValidationError("NPT MC barostat requires an OrthorhombicBox"))
        if constraints !== nothing && config.molecules === nothing
            throw(ValidationError(:barostat_constraints,
                "stage validate: :npt_baoab_mc with active holonomic constraints requires config.molecules (e.g. MonteCarloBarostat.infer_molecules) for molecular-COM scaling"))
        end
    end
    hsys = Dynamics.HarmonicSystem(sys, factors)
    if config.algorithm === :respa_nve
        (constraints === nothing || constraints isa Constraints.CoupledConstraintSet ||
         constraints isa AbstractVector) ||
            throw(ValidationError("RESPA refuses unsupported constraint specification $(typeof(constraints))"))
        fast = _resolve_fast_factors(hsys.factors, config.fast_factors)
        isempty(fast) && throw(ValidationError("RESPA requires at least one fast factor"))
        CF = parentmodule(@__MODULE__).ChemistryFactors
        CM = parentmodule(@__MODULE__).ChemistryCMAP
        all(f -> f isa Factors.HarmonicBond || f isa Factors.HarmonicAngle ||
                 f isa Factors.PeriodicDihedral || f isa Factors.LennardJonesPair ||
                 f isa Factors.CoulombPair || f isa Factors.BipartiteHarmonicGroup ||
                 f isa CF.MultiPeriodicDihedral || f isa CF.HarmonicImproper ||
                 f isa CF.UreyBradley || f isa CF.PairTableNonbonded ||
                 f isa CF.PairC64Table || f isa CM.CMAPCorrection ||
                 f isa parentmodule(@__MODULE__).StreamedLJ.StreamedLJFactor ||
                 f isa parentmodule(@__MODULE__).Electrostatics.EwaldFactor ||
                 f isa parentmodule(@__MODULE__).PME.PMEFactor, hsys.factors) ||
            throw(ValidationError("RESPA does not support one or more force models in this system"))
    end
    try
        Dynamics.check_force_model(hsys)
    catch e
        throw(ValidationError("force model failed conservative consistency check: $(sprint(showerror, e))"))
    end
    F = zeros(3, nat)
    U = try
        Dynamics.forces!(hsys, sys.atoms.q, F)
    catch e
        throw(ValidationError("initial force evaluation failed: $(sprint(showerror, e))"))
    end
    (isfinite(U) && all(isfinite, F)) ||
        throw(ValidationError("initial energy or forces are non-finite (U=$U)"))
    return (system = sys, harmonic_system = hsys, initial_U = U, constraints = constraints)
end

struct StageSpec
    name::Symbol
    config::WorkflowConfig
end

function validate_stage(stage::StageSpec, hsys::Any, box::Types.AbstractBox; constraints = nothing)
    cfg = stage.config
    if cfg.algorithm === :npt_baoab_mc && constraints !== nothing && cfg.molecules === nothing
        throw(ValidationError(:barostat_constraints,
            "stage $(stage.name): :npt_baoab_mc with active holonomic constraints requires config.molecules (e.g. MonteCarloBarostat.infer_molecules) for molecular-COM scaling"))
    end
    return true
end

function _resolve_fast_factors(factors, requested)
    indices = Int[]
    for item in requested
        if item isa Integer
            push!(indices, Int(item))
        elseif item isa AbstractString || item isa Symbol
            label = String(item)
            matches = findall(i -> label == "$(i):$(nameof(typeof(factors[i])))" ||
                                   label == String(nameof(typeof(factors[i]))), eachindex(factors))
            length(matches) == 1 || throw(ValidationError("fast factor name '$label' is missing or ambiguous"))
            push!(indices, only(matches))
        else
            throw(ValidationError("fast_factors entries must be indices or names"))
        end
    end
    length(unique(indices)) == length(indices) || throw(ValidationError("fast_factors contains duplicates"))
    all(i -> 1 <= i <= length(factors), indices) || throw(ValidationError("fast factor index is out of range"))
    return sort!(indices)
end

function validate_stage(pdb_path::AbstractString, policy::Prepare.PreparationPolicy,
                        factor_builder::Function, config::WorkflowConfig;
                        units::Types.Units = Types.REDUCED, constraints = nothing)
    isfile(pdb_path) || throw(ValidationError("structure file not found: $pdb_path"))
    rep = Prepare.diagnose_structure(pdb_path; contact_tol = policy.close_contact_tol)
    rep.n_atoms > 0 || throw(ValidationError("PDB file contains zero atoms: $pdb_path"))
    rep.n_nonfinite == 0 || throw(ValidationError("PDB file contains $(rep.n_nonfinite) non-finite coordinates"))
    return (pdb_path = String(pdb_path), policy = policy, factor_builder = factor_builder,
            units = units, report = rep, constraints = constraints)
end

"""Stage 2 (`prepare`): construct fresh state, `HarmonicSystem`, RNG, and immutable `RunManifest`."""
function prepare_stage(validated::NamedTuple, config::WorkflowConfig)
    sys, factors, prep_actions = if haskey(validated, :pdb_path)
        p = Prepare.prepare(validated.pdb_path, validated.policy;
                            units = validated.units, temperature = config.temperature)
        facs = validated.factor_builder(p)
        (p.system, facs, p.actions)
    else
        (validated.system, validated.harmonic_system.factors, Prepare.RepairAction[])
    end

    hsys = Dynamics.HarmonicSystem(sys, factors)
    try
        Dynamics.check_force_model(hsys)
    catch e
        throw(ValidationError("force model failed conservative consistency check: $(sprint(showerror, e))"))
    end
    st = Dynamics.State(copy(sys.atoms.q), copy(sys.atoms.p))
    U = Dynamics.forces!(hsys, st.q, st.F)
    st.force_valid = true
    (isfinite(U) && all(isfinite, st.F)) ||
        throw(ValidationError("prepared energy/forces are non-finite"))

    constraints_desc = validated.constraints === nothing ? "none" : string(typeof(validated.constraints))
    rp = deepcopy(config.resolved_parameters)
    if config.algorithm === :respa_nve
        fast = _resolve_fast_factors(hsys.factors, config.fast_factors)
        rp["integrator"] = Dict("algorithm" => "respa_nve",
                                "fast_factors" => ["$(i):$(nameof(typeof(hsys.factors[i])))" for i in fast],
                                "respa_m" => config.respa_m)
    end
    manifest = build_manifest(;
        run_id = config.run_id,
        model_versions = config.model_versions,
        resolved_parameters = rp,
        units = sys.units,
        box = sys.box,
        algorithm = config.algorithm,
        dt = config.dt,
        seed = config.seed,
        n_atoms = size(st.q, 2),
        temperature = config.temperature,
        pressure = config.pressure,
        gamma = config.gamma,
        constraints_summary = constraints_desc,
    )
    rng = Random.Xoshiro(config.seed)
    return (
        system = sys,
        harmonic_system = hsys,
        state = st,
        box = sys.box,
        manifest = manifest,
        rng = rng,
        constraints = validated.constraints,
        preparation_actions = prep_actions,
    )
end

"""Stage 3 (`minimize`): deterministic backtracking steepest-descent energy minimization."""
function minimize_stage(state::Dynamics.State, hsys::Dynamics.HarmonicSystem,
                        config::WorkflowConfig; constraints = nothing)
    q = copy(state.q)
    F = zeros(3, size(q, 2))
    if constraints !== nothing
        tmp = Dynamics.State(q, zeros(size(q)))
        Dynamics.rattle_project!(tmp, hsys, constraints)
        q .= tmp.q
    end
    U = Dynamics.forces!(hsys, q, F)
    (isfinite(U) && all(isfinite, F)) ||
        throw(TransactionError(:minimize, "non-finite initial energy/forces"))

    step_size = config.minimize_step_size
    iters = 0
    for it in 1:config.minimize_max_steps
        gmax = maximum(abs, F)
        gmax <= config.minimize_grad_tol && break
        fnorm = norm(F)
        fnorm == 0.0 && break
        dir = F ./ fnorm
        accepted = false
        alpha = step_size
        for _ in 1:20
            q_cand = q .+ alpha .* dir
            if constraints !== nothing
                tmp = Dynamics.State(q_cand, zeros(size(q_cand)))
                Dynamics.rattle_project!(tmp, hsys, constraints)
                q_cand .= tmp.q
            end
            F_cand = zeros(3, size(q, 2))
            U_cand = Dynamics.forces!(hsys, q_cand, F_cand)
            if isfinite(U_cand) && all(isfinite, F_cand) && U_cand <= U
                q .= q_cand
                F .= F_cand
                U = U_cand
                accepted = true
                break
            end
            alpha *= 0.5
        end
        accepted || break
        iters = it
    end

    out = Dynamics.State(q, copy(state.p), F, true)
    return (state = out, potential_energy = U, max_force = maximum(abs, F), iterations = iters)
end

function _rebuild_hamiltonian_for_box(hsys::Dynamics.HarmonicSystem, box_new::Types.AbstractBox)
    new_factors = map(hsys.factors) do fac
        if fac isa parentmodule(@__MODULE__).StreamedLJ.StreamedLJFactor
            SLJ = parentmodule(@__MODULE__).StreamedLJ
            new_ws = SLJ.LJWorkspace(zeros(3, fac.N), box_new, fac.rcut; skin = fac.skin)
            new_pc = nothing
            if fac.pair_cache !== nothing
                cs = fac.cache_skin > 0.0 ? min(fac.cache_skin, SLJ.default_pair_cache_skin(box_new, fac.rcut)) : 0.0
                if cs > 0.0
                    if fac.pair_cache isa parentmodule(@__MODULE__).Neighbors.TilePairCache || fac.N > parentmodule(@__MODULE__).Neighbors.MAX_MATERIALIZED_PAIR_ATOMS
                        new_pc = parentmodule(@__MODULE__).Neighbors.TilePairCache(zeros(3, fac.N), box_new, fac.rcut, cs)
                    else
                        new_pc = parentmodule(@__MODULE__).Neighbors.PairCache(zeros(3, fac.N), box_new, fac.rcut, cs)
                    end
                end
            end
            SLJ.StreamedLJFactor(fac.N, fac.epsilon, fac.sigma, box_new, fac.rcut, fac.skin, fac.mode,
                                 fac.topology, new_ws, fac.threaded, fac.cache, fac.cache_skin, new_pc, fac.fixed_point)
        elseif fac isa parentmodule(@__MODULE__).PME.PMEFactor
            PME_mod = parentmodule(@__MODULE__).PME
            box_arg = box_new isa Types.OrthorhombicBox ? collect(box_new.L) : box_new
            new_plan = PME_mod.PMEPlan(zeros(3, fac.N), box_arg; alpha = fac.plan.alpha, rcut = fac.plan.rcut,
                                       grid = fac.plan.dims, order = fac.plan.order, prefactor = fac.plan.prefactor)
            PME_mod.PMEFactor(fac.N, fac.charges, box_new, new_plan, fac.corrections)
        elseif fac isa parentmodule(@__MODULE__).Electrostatics.EwaldFactor
            Elec = parentmodule(@__MODULE__).Electrostatics
            Elec.EwaldFactor(fac.N, fac.charges, box_new, fac.alpha, fac.rcut, fac.kmax, fac.prefactor, fac.corrections)
        else
            fac
        end
    end
    new_ctx = Factors.Context(box_new, hsys.ctx.units, hsys.ctx.N)
    return Dynamics.HarmonicSystem(new_factors, hsys.m, new_ctx)
end

_rebuild_hamiltonian_for_box(hsys::Any, box_new::Types.AbstractBox) =
    _rebuild_hamiltonian_for_box(hsys::Dynamics.HarmonicSystem, box_new)

function _step_engine!(st::Dynamics.State, hsys::Dynamics.HarmonicSystem,
                       box::Types.AbstractBox, config::WorkflowConfig,
                       rng::Random.AbstractRNG, th::ThermostatState, ba::BarostatState,
                       step_idx::Int; constraints = nothing,
                       respa_integrator::Union{Nothing,Respa.RespaIntegrator} = nothing)
    if config.algorithm === :velocity_verlet
        st.force_valid || (Dynamics.forces!(hsys, st.q, st.F); st.force_valid = true)
        U = if constraints === nothing
            u_val = Dynamics.velocity_verlet_step!(st.q, st.p, st.F, hsys, config.dt)
            st.force_valid = true
            u_val
        else
            Dynamics.velocity_verlet_step!(st, hsys, config.dt; constraints = constraints)
            Dynamics.potential_energy(hsys, st.q)
        end
        return U, hsys, box, th, ba
    elseif config.algorithm === :respa_nve
        (constraints === nothing || constraints isa Constraints.CoupledConstraintSet ||
         constraints isa AbstractVector) ||
            throw(TransactionError(:produce, "RESPA refuses unsupported constraint specification $(typeof(constraints))"))
        integrator = respa_integrator === nothing ?
                     Respa.RespaIntegrator(hsys, _resolve_fast_factors(hsys.factors, config.fast_factors)) :
                     respa_integrator
        Respa.step!(st, integrator, config.dt; m = config.respa_m, constraints = constraints)
        U = integrator.last_energy
        return U, hsys, box, th, ba
    elseif config.algorithm === :langevin_baoab
        gamma = config.gamma === nothing ? 1.0 : config.gamma
        beta = Sampling.molecular_beta(hsys, config.temperature)
        st.force_valid || (Dynamics.forces!(hsys, st.q, st.F); st.force_valid = true)
        U = Dynamics.langevin_baoab_step!(st.q, st.p, st.F, hsys, config.dt, gamma, beta, rng;
                                          constraints = constraints)
        st.force_valid = true
        th2 = ThermostatState(kind = :langevin_baoab, temperature = config.temperature,
                              gamma = gamma, beta = beta, steps = th.steps + 1, extra = th.extra)
        return U, hsys, box, th2, ba
    elseif config.algorithm === :npt_baoab_mc
        gamma = config.gamma === nothing ? 1.0 : config.gamma
        beta = Sampling.molecular_beta(hsys, config.temperature)
        st.force_valid || (Dynamics.forces!(hsys, st.q, st.F); st.force_valid = true)
        U = Dynamics.langevin_baoab_step!(st.q, st.p, st.F, hsys, config.dt, gamma, beta, rng;
                                          constraints = constraints)
        st.force_valid = true
        th2 = ThermostatState(kind = :langevin_baoab, temperature = config.temperature,
                              gamma = gamma, beta = beta, steps = th.steps + 1, extra = th.extra)
        ba2 = ba
        box2 = box
        hsys2 = hsys
        if step_idx % config.barostat_every == 0
            barostat = MonteCarloBarostat.VolumeMoveBarostat(
                beta, config.pressure, config.max_delta_log_volume;
                molecules = config.molecules,
                masses = hsys.m,
                constraints = constraints,
                coupling = config.barostat_coupling,
                surface_tension = config.surface_tension)
            energy_fn = (q_try, b_try) -> begin
                hsys_try = _rebuild_hamiltonian_for_box(hsys, b_try)
                Factors.total_energy_forces(hsys_try.factors, q_try, hsys_try.ctx)[1]
            end
            res = MonteCarloBarostat.step!(barostat, st.q, box, energy_fn, rng)
            if res.accepted
                box2 = res.box
                hsys2 = _rebuild_hamiltonian_for_box(hsys, box2)
                U = Dynamics.forces!(hsys2, st.q, st.F)
                st.force_valid = true
            end
            ba2 = BarostatState(
                kind = :mc_volume, pressure = config.pressure, beta = beta,
                max_delta_log_volume = config.max_delta_log_volume,
                attempts = ba.attempts + 1,
                accepts = ba.accepts + (res.accepted ? 1 : 0),
                last_delta_log_volume = res.delta_log_volume,
            )
        end
        return U, hsys2, box2, th2, ba2
    else
        throw(TransactionError(:equilibrate, "algorithm :$(config.algorithm) is not executable in MD loop"))
    end
end

"""Stage 4 (`equilibrate`): thermalize momenta (if needed) and evolve `equilibrate_steps`."""
function equilibrate_stage(state::Dynamics.State, hsys::Dynamics.HarmonicSystem,
                           manifest::RunManifest, rng::Random.AbstractRNG,
                           config::WorkflowConfig;
                           constraints = nothing,
                           checkpoint_dir::Union{Nothing,AbstractString} = nothing)
    st = copy(state)
    box = hsys.ctx.box
    cur_hsys = hsys
    if config.temperature > 0 && config.algorithm in (:langevin_baoab, :npt_baoab_mc) && all(iszero, st.p)
        Sampling.maxwell_momenta!(st.p, cur_hsys, config.temperature, rng)
        if constraints !== nothing
            Dynamics.rattle_project!(st, cur_hsys, constraints)
        end
    end
    th = config.algorithm in (:langevin_baoab, :npt_baoab_mc) ?
         ThermostatState(kind = :langevin_baoab, temperature = config.temperature,
                         gamma = config.gamma === nothing ? 1.0 : config.gamma,
                         beta = config.temperature > 0 ? Sampling.molecular_beta(cur_hsys, config.temperature) : 0.0) :
         ThermostatState()
    ba = config.algorithm === :npt_baoab_mc ?
         BarostatState(kind = :mc_volume, pressure = config.pressure,
                       beta = Sampling.molecular_beta(cur_hsys, config.temperature),
                       max_delta_log_volume = config.max_delta_log_volume) :
         BarostatState()

    respa_int = config.algorithm === :respa_nve ?
                Respa.RespaIntegrator(cur_hsys, _resolve_fast_factors(cur_hsys.factors, config.fast_factors)) :
                nothing
    last_ckpt = nothing
    for step in 1:config.equilibrate_steps
        U, cur_hsys, box, th, ba = _step_engine!(st, cur_hsys, box, config, rng, th, ba, step;
                                                 constraints = constraints,
                                                 respa_integrator = respa_int)
        (isfinite(U) && all(isfinite, st.q) && all(isfinite, st.p) && all(isfinite, st.F)) ||
            throw(TransactionError(:equilibrate, "non-finite state at equilibration step $step"))
        if checkpoint_dir !== nothing && config.checkpoint_every > 0 && step % config.checkpoint_every == 0
            ckpt = make_checkpoint(
                stage = :equilibrate, step = step, time = step * config.dt,
                manifest = manifest, state = st, box = box, rng = rng,
                thermostat_state = th, barostat_state = ba,
                restart_guarantee = config.restart_guarantee)
            cpath = joinpath(checkpoint_dir, @sprintf("ckpt_equilibrate_%08d.json", step))
            write_checkpoint_atomic(cpath, ckpt)
            last_ckpt = cpath
        end
    end
    return (state = st, harmonic_system = cur_hsys, box = box, rng = rng,
            thermostat_state = th, barostat_state = ba, checkpoint_path = last_ckpt)
end

"""Stage 5 (`produce`): evolve production steps, write trajectory metadata, and checkpoint atomically."""
function produce_stage(state::Dynamics.State, hsys::Dynamics.HarmonicSystem,
                       manifest::RunManifest, rng::Random.AbstractRNG,
                       config::WorkflowConfig;
                       trajectory_path::AbstractString,
                       labels::Vector{String} = fill("X", size(state.q, 2)),
                       constraints = nothing,
                       checkpoint_dir::Union{Nothing,AbstractString} = nothing,
                       thermostat_state::ThermostatState = ThermostatState(),
                       barostat_state::BarostatState = BarostatState(),
                       start_step::Int = 0,
                       start_time::Float64 = 0.0)
    st = copy(state)
    cur_hsys = hsys
    box = hsys.ctx.box
    th = thermostat_state
    ba = barostat_state
    t = start_time

    frames = Matrix{Float64}[]
    times = Float64[]
    boxes = Types.AbstractBox[]
    energies = Float64[]
    last_ckpt = nothing
    log_path = replace(endswith(lowercase(String(trajectory_path)), ".xyz") ? String(trajectory_path) : String(trajectory_path) * ".xyz", r"\.xyz$" => ".jsonl")
    logio = open(log_path, "w")
    started = time_ns()
    steps_done = 0

    writer = TrajectoryWriter(trajectory_path, size(st.q, 2);
                              labels = labels, units = cur_hsys.ctx.units,
                              manifest_digest = manifest.manifest_digest)
    respa_int = config.algorithm === :respa_nve ?
                Respa.RespaIntegrator(cur_hsys, _resolve_fast_factors(cur_hsys.factors, config.fast_factors)) :
                nothing
    comp_force_buf = zeros(3, size(st.q, 2))
    try
        for rel_step in 1:config.produce_steps
            step = start_step + rel_step
            U, cur_hsys, box, th, ba = _step_engine!(st, cur_hsys, box, config, rng, th, ba, step;
                                                     constraints = constraints,
                                                     respa_integrator = respa_int)
            t += config.dt
            K = Dynamics.kinetic_energy(cur_hsys, st.p)
            E = K + U
            (isfinite(E) && all(isfinite, st.q) && all(isfinite, st.p) && all(isfinite, st.F)) ||
                throw(TransactionError(:produce, "non-finite state at production step $step"))

            if rel_step % config.frame_every == 0
                write_trajectory_frame!(writer, st.q, box, step, t)
                push!(frames, copy(st.q))
                push!(times, t)
                push!(boxes, box)
                push!(energies, E)
                elapsed = (time_ns() - started) / 1.0e9
                dof = 3size(st.q, 2)
                components = Dict{String,Any}()
                for (idx, factor) in enumerate(cur_hsys.factors)
                    fill!(comp_force_buf, 0.0)
                    components["$(idx):$(nameof(typeof(factor)))"] =
                        Factors.evaluate!(comp_force_buf, factor, st.q, cur_hsys.ctx)
                end
                rec = Dict{String,Any}(
                    "step" => step, "time" => t, "potential_energy" => U,
                    "kinetic_energy" => K, "total_energy" => E,
                    "component_energies" => components,
                    "temperature" => constraints === nothing ? 2K / (dof * cur_hsys.ctx.units.boltzmann) : nothing,
                    "volume" => (box isa Types.NoBox ? nothing : Types.box_volume(box)),
                    "pressure" => nothing, "constraint_error" => nothing,
                    "elapsed_seconds" => elapsed,
                    "steps_per_second" => elapsed > 0 ? rel_step / elapsed : nothing)
                println(logio, JSON.json(rec))
                flush(logio)
            end
            steps_done += 1

            if checkpoint_dir !== nothing && config.checkpoint_every > 0 && rel_step % config.checkpoint_every == 0
                ckpt = make_checkpoint(
                    stage = :produce, step = step, time = t,
                    manifest = manifest, state = st, box = box, rng = rng,
                    thermostat_state = th, barostat_state = ba,
                    restart_guarantee = config.restart_guarantee)
                cpath = joinpath(checkpoint_dir, @sprintf("ckpt_produce_%08d.json", step))
                write_checkpoint_atomic(cpath, ckpt)
                last_ckpt = cpath
            end
        end
    finally
        close(writer)
        close(logio)
    end

    if checkpoint_dir !== nothing
        final_step = start_step + config.produce_steps
        ckpt = make_checkpoint(
            stage = :produce, step = final_step, time = t,
            manifest = manifest, state = st, box = box, rng = rng,
            thermostat_state = th, barostat_state = ba,
            restart_guarantee = config.restart_guarantee)
        cpath = joinpath(checkpoint_dir, "ckpt_latest.json")
        write_checkpoint_atomic(cpath, ckpt)
        last_ckpt = cpath
    end

    return (state = st, harmonic_system = cur_hsys, box = box, rng = rng,
            thermostat_state = th, barostat_state = ba,
            trajectory_path = writer.xyz_path,
            log_path = log_path,
            frames = frames, times = times, boxes = boxes, energies = energies,
            checkpoint_path = last_ckpt, final_step = start_step + config.produce_steps,
            final_time = t)
end

"""Stage 6 (`analyze`): verify trajectory file round-trip and execute guarded analysis."""
function analyze_stage(trajectory_path::AbstractString, energies::Union{Nothing,AbstractVector{<:Real}},
                       config::WorkflowConfig)
    traj = read_trajectory(trajectory_path)
    return analyze(traj; config = config.analysis, energies = energies)
end

# ---------------------------------------------------------------------------
# Transactional end-to-end orchestrator and checkpoint resume
# ---------------------------------------------------------------------------

function _run_stage(fn::Function, stage::Symbol, stages::Vector{StageRecord};
                    checkpoint_path::Union{Nothing,String} = nothing,
                    detail::String = "ok")
    t0 = time()
    try
        val = fn()
        push!(stages, StageRecord(stage, :completed, time() - t0, detail, checkpoint_path))
        return val
    catch e
        push!(stages, StageRecord(stage, :failed, time() - t0, sprint(showerror, e), nothing))
        rethrow()
    end
end

function _commit_staging_dir(staging_dir::AbstractString, workdir::AbstractString)
    mkpath(workdir)
    ckpt_dest = joinpath(workdir, "checkpoints")
    mkpath(ckpt_dest)
    staging_ckpt = joinpath(staging_dir, "checkpoints")
    if isdir(staging_ckpt)
        for fname in readdir(staging_ckpt)
            src = joinpath(staging_ckpt, fname)
            isfile(src) || continue
            bytes = read(src)
            _write_bytes_atomic(joinpath(ckpt_dest, fname), bytes)
        end
    end
    for fname in ("trajectory.xyz", "trajectory.xyz.jsonl", "trajectory.jsonl", "manifest.json", "analysis.json")
        src = joinpath(staging_dir, fname)
        isfile(src) || continue
        _write_bytes_atomic(joinpath(workdir, fname), read(src))
    end
    return nothing
end

function _write_analysis_atomic(path::AbstractString, res::AnalysisResult)
    doc = Dict{String,Any}(
        "n_frames" => res.n_frames,
        "n_atoms" => res.n_atoms,
        "total_time" => res.total_time,
        "dt_frame" => res.dt_frame,
        "fit_lag_indices" => [first(res.fit_lag_indices), last(res.fit_lag_indices)],
        "fit_slope" => res.fit_slope,
        "fit_intercept" => res.fit_intercept,
        "diffusion_coefficient" => res.diffusion_coefficient,
        "fit_r2" => res.fit_r2,
        "max_step_half_box_fraction" => res.max_step_half_box_fraction,
        "unwrapped" => res.unwrapped,
        "warnings" => res.warnings,
        "caveat" => res.caveat,
    )
    return _write_bytes_atomic(path, Vector{UInt8}(codeunits(JSON.json(doc))))
end

function _execute_workflow_stages(validate_fn::Function, config::WorkflowConfig,
                                  workdir::AbstractString)
    wdir = abspath(String(workdir))
    mkpath(wdir)
    staging = mktempdir(wdir; prefix = ".staging_")
    ckpt_dir = joinpath(staging, "checkpoints")
    mkpath(ckpt_dir)
    stages = StageRecord[]

    try
        val = _run_stage(validate_fn, :validate, stages)
        prep = _run_stage(:prepare, stages) do
            prepare_stage(val, config)
        end
        write_manifest_atomic(joinpath(staging, "manifest.json"), prep.manifest)

        min_res = _run_stage(:minimize, stages) do
            minimize_stage(prep.state, prep.harmonic_system, config; constraints = prep.constraints)
        end
        eq_res = _run_stage(:equilibrate, stages) do
            equilibrate_stage(min_res.state, prep.harmonic_system, prep.manifest,
                              prep.rng, config;
                              constraints = prep.constraints, checkpoint_dir = ckpt_dir)
        end
        labels = [String(nm) for nm in prep.system.atoms.names]
        prod_res = _run_stage(:produce, stages) do
            produce_stage(eq_res.state, eq_res.harmonic_system, prep.manifest,
                          eq_res.rng, config;
                          trajectory_path = joinpath(staging, "trajectory.xyz"),
                          labels = labels,
                          constraints = prep.constraints,
                          checkpoint_dir = ckpt_dir,
                          thermostat_state = eq_res.thermostat_state,
                          barostat_state = eq_res.barostat_state)
        end
        ana_res = _run_stage(:analyze, stages) do
            analyze_stage(prod_res.trajectory_path, prod_res.energies, config)
        end
        _write_analysis_atomic(joinpath(staging, "analysis.json"), ana_res)
        _commit_staging_dir(staging, wdir)

        final_ckpt = prod_res.checkpoint_path === nothing ? nothing :
                     joinpath(wdir, "checkpoints", basename(prod_res.checkpoint_path))
        return WorkflowResult(
            prep.manifest, stages, prod_res.state, prod_res.box,
            final_ckpt, joinpath(wdir, "trajectory.xyz"), ana_res, true,
        )
    finally
        rm(staging; recursive = true, force = true)
    end
end

"""
    run_workflow(sys::Types.System, factors, config::WorkflowConfig; workdir, constraints = nothing)

Execute the full transactional pipeline `validate -> prepare -> minimize ->
equilibrate -> produce -> analyze`. All outputs are staged in a temporary
subdirectory under `workdir` and committed atomically only after every stage
succeeds; if any stage fails, `workdir` receives no partial manifest or
completed output bundle.
"""
function run_workflow(sys::Types.System, factors, config::WorkflowConfig;
                      workdir::AbstractString, constraints = nothing)
    return _execute_workflow_stages(config, workdir) do
        validate_stage(sys, factors, config; constraints = constraints)
    end
end

"""
    run_workflow(pdb_path, policy, factor_builder, config; workdir, units=REDUCED, constraints=nothing)

Transactional workflow entry point starting from a raw PDB file and explicit
`PreparationPolicy`.
"""
function run_workflow(pdb_path::AbstractString, policy::Prepare.PreparationPolicy,
                      factor_builder::Function, config::WorkflowConfig;
                      workdir::AbstractString,
                      units::Types.Units = Types.REDUCED,
                      constraints = nothing)
    return _execute_workflow_stages(config, workdir) do
        validate_stage(pdb_path, policy, factor_builder, config;
                       units = units, constraints = constraints)
    end
end

"""
    resume_workflow(checkpoint_path, manifest_path, factors, masses, units, config; ...)

Resume production from an atomic checkpoint and verified `RunManifest`, checking
manifest digest compatibility and restart guarantee (`require_bitwise`).
"""
function resume_workflow(checkpoint_path::AbstractString,
                         manifest_path::AbstractString,
                         factors,
                         masses::AbstractVector{<:Real},
                         units::Types.Units,
                         config::WorkflowConfig;
                         trajectory_path::AbstractString,
                         labels::Vector{String} = fill("X", length(masses)),
                         constraints = nothing,
                         checkpoint_dir::Union{Nothing,AbstractString} = nothing,
                         require_bitwise::Bool = (config.restart_guarantee === :bitwise))
    manifest = read_manifest(manifest_path)
    config.algorithm == manifest.algorithm ||
        throw(CheckpointMismatchError("resume config algorithm :$(config.algorithm) != manifest algorithm :$(manifest.algorithm)"))
    isapprox(config.dt, manifest.dt; atol = 1e-15, rtol = 0.0) ||
        throw(CheckpointMismatchError("resume config dt $(config.dt) != manifest dt $(manifest.dt)"))
    String(units.name) == manifest.units ||
        throw(CheckpointMismatchError("resume units $(units.name) != manifest units $(manifest.units)"))
    ckpt = read_checkpoint(checkpoint_path;
                           expected_manifest_digest = manifest.manifest_digest,
                           require_bitwise = require_bitwise)
    st, box, rng = restore_state(ckpt)
    rng_eff = rng === nothing ? Random.Xoshiro(config.seed) : rng
    hsys = Dynamics.HarmonicSystem(factors, masses, Factors.Context(box, units, length(masses)))
    hsys = _rebuild_hamiltonian_for_box(hsys, box)
    Dynamics.check_force_model(hsys)
    return produce_stage(
        st, hsys, manifest, rng_eff, config;
        trajectory_path = trajectory_path,
        labels = labels,
        constraints = constraints,
        checkpoint_dir = checkpoint_dir,
        thermostat_state = ckpt.thermostat_state,
        barostat_state = ckpt.barostat_state,
        start_step = ckpt.step,
        start_time = ckpt.time,
    )
end

end # module Workflow
