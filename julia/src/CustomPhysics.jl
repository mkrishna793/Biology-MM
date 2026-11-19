"""
`CustomPhysics` — extensible parameter-encoding engine, analytical & spline
radial potential kernels, user-defined pair/external/many-body factors,
potential registry, and finite-difference gradient verification.
"""
module CustomPhysics

using LinearAlgebra
using ..Types
using ..Factors

const _HAS_RESOLVED_SYSTEM = isdefined(parentmodule(@__MODULE__), :ResolvedSystem)
if _HAS_RESOLVED_SYSTEM
    import ..ResolvedSystem
end

export MD_UNITS, EV_ANGSTROM_UNITS,
       ParameterSpec, ParameterEncoder,
       validate_parameters, encode_parameters, decode_parameters,
       encoding_jacobian_diag, int_to_balanced_trits, balanced_trits_to_int,
       encoder_to_dict, encoder_from_dict,
       CubicSplineTable, eval_spline, eval_radial_kernel,
       CustomPairFactor, CustomExternalFactor, CustomManyBodyFactor,
       POTENTIAL_REGISTRY, register_potential!, registered_potentials,
       build_custom_factor, verify_custom_factor_gradients

# ---------------------------------------------------------------------------
# Unit presets
# ---------------------------------------------------------------------------

const MD_UNITS = Types.Units(
    "MD (nm, kJ/mol, amu, e, ps)",
    1e-9,
    1000.0 / 6.02214076e23,
    1.66053906660e-27,
    1.602176634e-19,
    138.93545764438198,
    0.00831446261815324,
)

const EV_ANGSTROM_UNITS = Types.Units(
    "Metal (A, eV, amu, e, fs)",
    1e-10,
    1.602176634e-19,
    1.66053906660e-27,
    1.602176634e-19,
    14.39964547842567,
    8.617333262e-5,
)

# ---------------------------------------------------------------------------
# Helpers for fractional coordinates and box-derivative outer products
# ---------------------------------------------------------------------------

@inline function _frac_disp(b::Types.OrthorhombicBox, d::Types.SVector3)
    L = b.L
    return (d.x / L[1], d.y / L[2], d.z / L[3])
end

@inline function _frac_disp(b::Types.TriclinicBox, d::Types.SVector3)
    M = b.Hinv
    return (M[1, 1] * d.x + M[1, 2] * d.y + M[1, 3] * d.z,
            M[2, 1] * d.x + M[2, 2] * d.y + M[2, 3] * d.z,
            M[3, 1] * d.x + M[3, 2] * d.y + M[3, 3] * d.z)
end

@inline function _frac_disp(::Types.NoBox, ::Types.SVector3)
    throw(Types.UnsupportedGeometry("no periodic box: box derivative requires lattice vectors"))
end

@inline function _outer_add!(G::Matrix{Float64}, c::Float64,
                             dx::Float64, dy::Float64, dz::Float64,
                             wx::Float64, wy::Float64, wz::Float64)
    gx = c * dx; gy = c * dy; gz = c * dz
    @inbounds begin
        G[1, 1] += gx * wx; G[1, 2] += gx * wy; G[1, 3] += gx * wz
        G[2, 1] += gy * wx; G[2, 2] += gy * wy; G[2, 3] += gy * wz
        G[3, 1] += gz * wx; G[3, 2] += gz * wy; G[3, 3] += gz * wz
    end
    return G
end

@inline _outer_add!(G::Matrix{Float64}, c::Float64, d::Types.SVector3, w::NTuple{3,Float64}) =
    _outer_add!(G, c, d.x, d.y, d.z, w[1], w[2], w[3])

# ---------------------------------------------------------------------------
# Deep Parameter Encoding Engine
# ---------------------------------------------------------------------------

const _VALID_ENCODINGS = (:identity, :log, :logit, :angle, :ternary_quantized)

"""
    ParameterSpec(name, default; unit="", lower=-Inf, upper=Inf, encoding=:identity, trit_scale=1.0)

Specification for one physical parameter, including domain bounds and smooth
unconstrained reparameterization (`:identity`, `:log`, `:logit`, `:angle`,
or `:ternary_quantized`).
"""
struct ParameterSpec
    name::Symbol
    default::Float64
    unit::String
    lower::Float64
    upper::Float64
    encoding::Symbol
    trit_scale::Float64
    function ParameterSpec(name::Symbol, default::Real;
                           unit::String = "",
                           lower::Real = -Inf,
                           upper::Real = Inf,
                           encoding::Symbol = :identity,
                           trit_scale::Real = 1.0)
        encoding in _VALID_ENCODINGS ||
            throw(ArgumentError("unsupported encoding :$encoding; expected one of $_VALID_ENCODINGS"))
        def = Float64(default)
        lo = Float64(lower)
        hi = Float64(upper)
        ts = Float64(trit_scale)
        isfinite(def) || throw(ArgumentError("default for :$name must be finite, got $default"))
        (!isnan(lo) && !isnan(hi) && lo <= hi) ||
            throw(ArgumentError("invalid bounds [$lo, $hi] for :$name"))
        (lo <= def <= hi) ||
            throw(ArgumentError("default $def for :$name outside bounds [$lo, $hi]"))
        (isfinite(ts) && ts > 0.0) ||
            throw(ArgumentError("trit_scale for :$name must be finite and > 0, got $trit_scale"))
        if encoding === :log
            lo >= 0.0 || throw(ArgumentError(":log encoding for :$name requires lower >= 0, got $lo"))
            def > lo || throw(ArgumentError(":log encoding for :$name requires default > lower ($lo), got $def"))
        elseif encoding === :logit
            (isfinite(lo) && isfinite(hi) && lo < hi) ||
                throw(ArgumentError(":logit encoding for :$name requires finite lower < upper, got [$lo, $hi]"))
            (lo < def < hi) ||
                throw(ArgumentError(":logit encoding for :$name requires lower < default < upper"))
        end
        return new(name, def, unit, lo, hi, encoding, ts)
    end
end

"""
    ParameterEncoder(specs::Vector{ParameterSpec})

Bijective (or quantized) mapping between physical parameter dictionaries and
unconstrained optimization coordinates `z::Vector{Float64}`.
"""
struct ParameterEncoder
    specs::Vector{ParameterSpec}
    index_map::Dict{Symbol,Int}
    function ParameterEncoder(specs::AbstractVector{ParameterSpec})
        vec = collect(ParameterSpec, specs)
        imap = Dict{Symbol,Int}()
        for (i, s) in enumerate(vec)
            haskey(imap, s.name) && throw(ArgumentError("duplicate parameter spec :$(s.name)"))
            imap[s.name] = i
        end
        return new(vec, imap)
    end
end

"""Convert an integer to balanced ternary digits in `{-1, 0, +1}` (LSB first)."""
function int_to_balanced_trits(n::Int)::Vector{Int}
    n == 0 && return [0]
    trits = Int[]
    v = n
    while v != 0
        rem = mod(v, 3)
        if rem == 2
            rem = -1
            v = div(v + 1, 3)
        else
            v = div(v - rem, 3)
        end
        push!(trits, rem)
    end
    return trits
end

"""Convert balanced ternary digits in `{-1, 0, +1}` (LSB first) back to `Int`."""
function balanced_trits_to_int(trits::AbstractVector{<:Integer})::Int
    s = 0
    pow3 = 1
    for t in trits
        (t == -1 || t == 0 || t == 1) || throw(ArgumentError("invalid balanced trit $t; expected -1, 0, or 1"))
        s += Int(t) * pow3
        pow3 *= 3
    end
    return s
end

function _get_param_value(params::AbstractDict, name::Symbol)
    if haskey(params, name)
        return Float64(params[name])
    elseif keytype(params) <: AbstractString && haskey(params, String(name))
        return Float64(params[String(name)])
    end
    throw(ArgumentError("missing required parameter :$name"))
end

"""
    validate_parameters(enc::ParameterEncoder, params::AbstractDict) -> Bool

Validate finiteness and domain bounds of all parameters specified in `enc`.
Throws `ArgumentError` if any parameter is missing, non-finite, or out of bounds.
"""
function validate_parameters(enc::ParameterEncoder, params::AbstractDict)::Bool
    for s in enc.specs
        x = _get_param_value(params, s.name)
        isfinite(x) || throw(ArgumentError("parameter :$(s.name) must be finite, got $x"))
        if s.encoding === :log
            (x > s.lower && x <= s.upper) ||
                throw(ArgumentError("parameter :$(s.name)=$x out of :log domain ($(s.lower), $(s.upper)]"))
        elseif s.encoding === :logit
            (s.lower < x < s.upper) ||
                throw(ArgumentError("parameter :$(s.name)=$x out of :logit open interval ($(s.lower), $(s.upper))"))
        else
            (s.lower <= x <= s.upper) ||
                throw(ArgumentError("parameter :$(s.name)=$x outside bounds [$(s.lower), $(s.upper)]"))
        end
    end
    return true
end

@inline function _sigmoid(z::Float64)::Float64
    if z >= 0.0
        return 1.0 / (1.0 + exp(-z))
    else
        ez = exp(z)
        return ez / (1.0 + ez)
    end
end

"""
    encode_parameters(enc::ParameterEncoder, params::AbstractDict) -> Vector{Float64}

Map physical parameters to unconstrained latent coordinates `z`.
"""
function encode_parameters(enc::ParameterEncoder, params::AbstractDict)::Vector{Float64}
    validate_parameters(enc, params)
    n = length(enc.specs)
    z = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        s = enc.specs[i]
        x = _get_param_value(params, s.name)
        if s.encoding === :identity
            z[i] = x
        elseif s.encoding === :log
            z[i] = log(x - s.lower)
        elseif s.encoding === :logit
            u = (x - s.lower) / (s.upper - s.lower)
            z[i] = log(u / (1.0 - u))
        elseif s.encoding === :angle
            z[i] = rem2pi(x, RoundNearest)
        elseif s.encoding === :ternary_quantized
            q_idx = round(Int, x / s.trit_scale)
            # Round-trip check via balanced ternary representation
            trits = int_to_balanced_trits(q_idx)
            z[i] = Float64(balanced_trits_to_int(trits))
        end
    end
    return z
end

"""
    decode_parameters(enc::ParameterEncoder, z::AbstractVector{<:Real}) -> Dict{Symbol,Float64}

Map unconstrained latent vector `z` back to physical parameter dictionary.
"""
function decode_parameters(enc::ParameterEncoder, z::AbstractVector{<:Real})::Dict{Symbol,Float64}
    n = length(enc.specs)
    length(z) == n || throw(ArgumentError("expected latent vector of length $n, got $(length(z))"))
    out = Dict{Symbol,Float64}()
    sizehint!(out, n)
    @inbounds for i in 1:n
        zi = Float64(z[i])
        isfinite(zi) || throw(ArgumentError("latent coordinate z[$i] must be finite, got $zi"))
        s = enc.specs[i]
        if s.encoding === :identity
            out[s.name] = zi
        elseif s.encoding === :log
            out[s.name] = s.lower + exp(zi)
        elseif s.encoding === :logit
            sig = _sigmoid(zi)
            out[s.name] = s.lower + (s.upper - s.lower) * sig
        elseif s.encoding === :angle
            out[s.name] = rem2pi(zi, RoundNearest)
        elseif s.encoding === :ternary_quantized
            out[s.name] = round(zi) * s.trit_scale
        end
    end
    return out
end

"""
    encoding_jacobian_diag(enc::ParameterEncoder, z::AbstractVector{<:Real}) -> Vector{Float64}

Diagonal of `∂x/∂z` for the decoding map `z -> x`.
"""
function encoding_jacobian_diag(enc::ParameterEncoder, z::AbstractVector{<:Real})::Vector{Float64}
    n = length(enc.specs)
    length(z) == n || throw(ArgumentError("expected latent vector of length $n, got $(length(z))"))
    jac = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        zi = Float64(z[i])
        isfinite(zi) || throw(ArgumentError("latent coordinate z[$i] must be finite, got $zi"))
        s = enc.specs[i]
        if s.encoding === :identity
            jac[i] = 1.0
        elseif s.encoding === :log
            jac[i] = exp(zi)
        elseif s.encoding === :logit
            sig = _sigmoid(zi)
            jac[i] = (s.upper - s.lower) * sig * (1.0 - sig)
        elseif s.encoding === :angle
            jac[i] = 1.0
        elseif s.encoding === :ternary_quantized
            jac[i] = 0.0
        end
    end
    return jac
end

function _serialize_bound(v::Float64)
    isfinite(v) && return v
    return v > 0 ? "Inf" : "-Inf"
end

function _deserialize_bound(v, default_val::Float64)::Float64
    v === nothing && return default_val
    if v isa Real
        return Float64(v)
    elseif v isa AbstractString
        s = lowercase(strip(v))
        (s == "inf" || s == "+inf" || s == "infinity") && return Inf
        (s == "-inf" || s == "-infinity") && return -Inf
        return parse(Float64, s)
    end
    throw(ArgumentError("cannot parse bound value $v"))
end

"""Serialize `ParameterEncoder` to a JSON-compatible dictionary."""
function encoder_to_dict(enc::ParameterEncoder)::Dict{String,Any}
    specs_list = Vector{Dict{String,Any}}(undef, length(enc.specs))
    for (i, s) in enumerate(enc.specs)
        specs_list[i] = Dict{String,Any}(
            "name" => String(s.name),
            "default" => s.default,
            "unit" => s.unit,
            "lower" => _serialize_bound(s.lower),
            "upper" => _serialize_bound(s.upper),
            "encoding" => String(s.encoding),
            "trit_scale" => s.trit_scale,
        )
    end
    return Dict{String,Any}(
        "schema" => "parameter_encoder_v1",
        "specs" => specs_list,
    )
end

@inline _dict_get(d::AbstractDict, key::String, default) =
    haskey(d, key) ? d[key] : (haskey(d, Symbol(key)) ? d[Symbol(key)] : default)

"""Reconstruct `ParameterEncoder` from a dictionary created by `encoder_to_dict`."""
function encoder_from_dict(d::AbstractDict)::ParameterEncoder
    raw_specs = _dict_get(d, "specs", nothing)
    raw_specs === nothing && throw(ArgumentError("missing 'specs' field in encoder dictionary"))
    specs = ParameterSpec[]
    sizehint!(specs, length(raw_specs))
    for item in raw_specs
        name = Symbol(_dict_get(item, "name", ""))
        def = Float64(_dict_get(item, "default", 0.0))
        unit = String(_dict_get(item, "unit", ""))
        lo = _deserialize_bound(_dict_get(item, "lower", -Inf), -Inf)
        hi = _deserialize_bound(_dict_get(item, "upper", Inf), Inf)
        enc_sym = Symbol(_dict_get(item, "encoding", "identity"))
        ts = Float64(_dict_get(item, "trit_scale", 1.0))
        push!(specs, ParameterSpec(name, def; unit = unit, lower = lo, upper = hi,
                                   encoding = enc_sym, trit_scale = ts))
    end
    return ParameterEncoder(specs)
end

# ---------------------------------------------------------------------------
# Natural Cubic Spline & Analytical Radial Kernel Library
# ---------------------------------------------------------------------------

"""
    CubicSplineTable(r_grid, u_grid)

Exact `C²` natural cubic spline interpolator over strictly increasing knots
`r_grid`. Extrapolates `C¹`-linearly outside `[r_grid[1], r_grid[end]]`.
"""
struct CubicSplineTable
    r_grid::Vector{Float64}
    a::Vector{Float64}
    b::Vector{Float64}
    c::Vector{Float64}
    d::Vector{Float64}
    u_end::Float64
    dudr_end::Float64
    function CubicSplineTable(r_in::AbstractVector{<:Real}, u_in::AbstractVector{<:Real})
        n = length(r_in)
        n >= 2 || throw(ArgumentError("CubicSplineTable requires at least 2 grid points"))
        length(u_in) == n || throw(ArgumentError("r_grid and u_grid must have the same length"))
        r = Float64.(collect(r_in))
        u = Float64.(collect(u_in))
        (all(isfinite, r) && all(isfinite, u)) ||
            throw(ArgumentError("spline grid coordinates and values must be finite"))
        for i in 1:(n - 1)
            r[i + 1] > r[i] || throw(ArgumentError("r_grid must be strictly increasing at index $i"))
        end

        h = Vector{Float64}(undef, n - 1)
        @inbounds for i in 1:(n - 1)
            h[i] = r[i + 1] - r[i]
        end

        c_full = zeros(Float64, n)
        if n > 2
            # Thomas algorithm for natural cubic spline tridiagonal system
            m = n - 2
            dl = Vector{Float64}(undef, m)
            dm = Vector{Float64}(undef, m)
            du = Vector{Float64}(undef, m)
            rhs = Vector{Float64}(undef, m)
            @inbounds for k in 1:m
                i = k + 1
                dl[k] = h[i - 1]
                dm[k] = 2.0 * (h[i - 1] + h[i])
                du[k] = h[i]
                rhs[k] = 3.0 * ((u[i + 1] - u[i]) / h[i] - (u[i] - u[i - 1]) / h[i - 1])
            end
            @inbounds for k in 2:m
                w = dl[k] / dm[k - 1]
                dm[k] -= w * du[k - 1]
                rhs[k] -= w * rhs[k - 1]
            end
            @inbounds c_full[n - 1] = rhs[m] / dm[m]
            @inbounds for k in (m - 1):-1:1
                c_full[k + 1] = (rhs[k] - du[k] * c_full[k + 2]) / dm[k]
            end
        end

        a = u[1:(n - 1)]
        b = Vector{Float64}(undef, n - 1)
        c = c_full[1:(n - 1)]
        d = Vector{Float64}(undef, n - 1)
        @inbounds for i in 1:(n - 1)
            hi = h[i]
            b[i] = (u[i + 1] - u[i]) / hi - hi * (2.0 * c_full[i] + c_full[i + 1]) / 3.0
            d[i] = (c_full[i + 1] - c_full[i]) / (3.0 * hi)
        end
        hlast = h[end]
        u_end = u[end]
        dudr_end = b[end] + 2.0 * c[end] * hlast + 3.0 * d[end] * hlast * hlast
        return new(r, a, b, c, d, u_end, dudr_end)
    end
end

"""
    eval_spline(tab::CubicSplineTable, r::Float64) -> (u::Float64, dudr::Float64)

Evaluate spline energy and radial derivative `dU/dr`.
"""
@inline function eval_spline(tab::CubicSplineTable, r::Float64)::Tuple{Float64,Float64}
    r0 = tab.r_grid[1]
    if r <= r0
        dr = r - r0
        return (tab.a[1] + tab.b[1] * dr, tab.b[1])
    end
    rend = tab.r_grid[end]
    if r >= rend
        dr = r - rend
        return (tab.u_end + tab.dudr_end * dr, tab.dudr_end)
    end
    idx = clamp(searchsortedlast(tab.r_grid, r), 1, length(tab.a))
    t = r - tab.r_grid[idx]
    @inbounds begin
        ai = tab.a[idx]; bi = tab.b[idx]; ci = tab.c[idx]; di = tab.d[idx]
        u = ai + t * (bi + t * (ci + t * di))
        dudr = bi + t * (2.0 * ci + 3.0 * di * t)
    end
    return (u, dudr)
end

"""
    eval_radial_kernel(kind::Symbol, r::Float64, p::AbstractDict{Symbol,Float64}; spline=nothing) -> (u, dudr)

Evaluate one of the built-in radial kernels:
`:morse`, `:buckingham`, `:yukawa`, `:mie`, `:fene`, `:wca`, `:tabulated_spline`.
"""
function eval_radial_kernel(kind::Symbol, r::Float64, p::AbstractDict{Symbol,Float64};
                            spline::Union{Nothing,CubicSplineTable} = nothing)::Tuple{Float64,Float64}
    if kind === :morse
        De = p[:De]; a = p[:a]; r0 = p[:r0]
        e = exp(-a * (r - r0))
        om = 1.0 - e
        u = De * om * om
        dudr = 2.0 * De * a * om * e
        return (u, dudr)
    elseif kind === :buckingham
        A = p[:A]; B = p[:B]; C = p[:C]
        eb = exp(-B * r)
        r6 = r^6
        u = A * eb - C / r6
        dudr = -A * B * eb + 6.0 * C / (r6 * r)
        return (u, dudr)
    elseif kind === :yukawa
        A = p[:A]; kappa = p[:kappa]
        ek = exp(-kappa * r)
        u = A * ek / r
        dudr = -A * ek * (kappa * r + 1.0) / (r * r)
        return (u, dudr)
    elseif kind === :mie
        eps = p[:epsilon]; sig = p[:sigma]; m = p[:m]; n = p[:n]
        (m > n && n > 0.0) || throw(ArgumentError("Mie potential requires m > n > 0, got m=$m, n=$n"))
        pref = (m / (m - n)) * (m / n)^(n / (m - n)) * eps
        sr = sig / r
        srm = sr^m
        srn = sr^n
        u = pref * (srm - srn)
        dudr = pref * (-m * srm + n * srn) / r
        return (u, dudr)
    elseif kind === :fene
        k = p[:k]; R0 = p[:R0]
        r >= R0 && return (Inf, 0.0)
        ratio2 = (r / R0)^2
        one_minus = 1.0 - ratio2
        u = -0.5 * k * (R0 * R0) * log(one_minus)
        dudr = k * r / one_minus
        return (u, dudr)
    elseif kind === :wca
        eps = p[:epsilon]; sig = p[:sigma]
        r_wca = 2.0^(1.0 / 6.0) * sig
        r >= r_wca && return (0.0, 0.0)
        s6 = (sig / r)^6
        u = 4.0 * eps * (s6 * s6 - s6) + eps
        dudr = 24.0 * eps * (s6 - 2.0 * s6 * s6) / r
        return (u, dudr)
    elseif kind === :tabulated_spline
        spline === nothing && throw(ArgumentError("spline::CubicSplineTable must be provided for :tabulated_spline"))
        return eval_spline(spline, r)
    else
        throw(ArgumentError("unknown radial kernel :$kind"))
    end
end

# ---------------------------------------------------------------------------
# Custom Factors
# ---------------------------------------------------------------------------

const _VALID_PAIR_MODES = (:none, :lfs, :shift, :trunc)
const _BUILTIN_KERNELS = (:morse, :buckingham, :yukawa, :mie, :fene, :wca, :tabulated_spline)

"""
    CustomPairFactor

Pairwise factor supporting built-in analytical/spline kernels (`kind`) or
custom callables `(energy_fn(r, p), dudr_fn(r, p))`, with `:none` or `:lfs`
C¹ force-consistent cutoff.
"""
struct CustomPairFactor{EF,DF} <: Factors.AbstractFactor
    kind::Symbol
    energy_fn::EF
    dudr_fn::DF
    spline::Union{Nothing,CubicSplineTable}
    params::Dict{Symbol,Float64}
    pairs::Vector{Tuple{Int,Int}}
    rc::Float64
    mode::Symbol
    n_atoms::Int
    uc::Float64
    duc::Float64
end

function _make_pair_factor(kind::Symbol, energy_fn, dudr_fn,
                           spline::Union{Nothing,CubicSplineTable},
                           params::AbstractDict{Symbol,<:Real},
                           pairs::AbstractVector{Tuple{Int,Int}};
                           rc::Real = Inf,
                           mode::Symbol = :none,
                           n_atoms::Union{Nothing,Int} = nothing)
    mode in _VALID_PAIR_MODES || throw(ArgumentError("unsupported pair mode :$mode"))
    pdict = Dict{Symbol,Float64}(k => Float64(v) for (k, v) in params)
    pvec = collect(Tuple{Int,Int}, pairs)
    max_idx = isempty(pvec) ? 0 : maximum(p -> max(p[1], p[2]), pvec)
    for (i, j) in pvec
        (i >= 1 && j >= 1 && i != j) ||
            throw(ArgumentError("invalid pair ($i, $j): indices must be >= 1 and distinct"))
    end
    nat = n_atoms === nothing ? max_idx : Int(n_atoms)
    nat >= max_idx || throw(ArgumentError("n_atoms=$nat smaller than max pair index $max_idx"))
    rc_f = Float64(rc)
    if mode !== :none
        (isfinite(rc_f) && rc_f > 0.0) || throw(ArgumentError("mode :$mode requires finite rc > 0, got $rc"))
    end
    uc = 0.0
    duc = 0.0
    if mode === :lfs || mode === :shift
        if kind === :custom
            uc = Float64(energy_fn(rc_f, pdict))
            duc = Float64(dudr_fn(rc_f, pdict))
        else
            uc, duc = eval_radial_kernel(kind, rc_f, pdict; spline = spline)
        end
        (isfinite(uc) && isfinite(duc)) ||
            throw(ArgumentError("cutoff evaluation at rc=$rc_f produced non-finite value ($uc, $duc)"))
    end
    return CustomPairFactor(kind, energy_fn, dudr_fn, spline, pdict, pvec, rc_f, mode, nat, uc, duc)
end

function CustomPairFactor(kind::Symbol,
                          params::AbstractDict{Symbol,<:Real},
                          pairs::AbstractVector{Tuple{Int,Int}};
                          rc::Real = Inf,
                          mode::Symbol = :none,
                          n_atoms::Union{Nothing,Int} = nothing,
                          spline::Union{Nothing,CubicSplineTable} = nothing)
    kind in _BUILTIN_KERNELS || throw(ArgumentError("unknown built-in radial kernel :$kind"))
    return _make_pair_factor(kind, nothing, nothing, spline, params, pairs;
                             rc = rc, mode = mode, n_atoms = n_atoms)
end

function CustomPairFactor(energy_fn, dudr_fn,
                          params::AbstractDict{Symbol,<:Real},
                          pairs::AbstractVector{Tuple{Int,Int}};
                          rc::Real = Inf,
                          mode::Symbol = :none,
                          n_atoms::Union{Nothing,Int} = nothing,
                          kind::Symbol = :custom)
    return _make_pair_factor(kind, energy_fn, dudr_fn, nothing, params, pairs;
                             rc = rc, mode = mode, n_atoms = n_atoms)
end

Factors.factor_support_size(f::CustomPairFactor) = f.n_atoms
Factors.read_support(f::CustomPairFactor) = Tuple(1:f.n_atoms)
Factors.is_local(::CustomPairFactor) = false
Factors.force_consistent(f::CustomPairFactor) = f.mode in (:none, :lfs)

@inline function _eval_pair_ud(f::CustomPairFactor, r::Float64)::Tuple{Float64,Float64}
    if f.kind === :custom
        return (Float64(f.energy_fn(r, f.params)), Float64(f.dudr_fn(r, f.params)))
    else
        return eval_radial_kernel(f.kind, r, f.params; spline = f.spline)
    end
end

function Factors.evaluate!(F::AbstractMatrix{Float64}, f::CustomPairFactor,
                           q::AbstractMatrix{Float64}, ctx::Factors.Context)
    U_tot = 0.0
    has_cut = f.mode !== :none
    @inbounds for (i, j) in f.pairs
        d = Types.unwrap_displacement(ctx.box, q, i, j)
        r2 = Types.norm2(d)
        r2 < 1e-24 && return Inf
        r = sqrt(r2)
        if has_cut && r >= f.rc
            continue
        end
        u, dudr = _eval_pair_ud(f, r)
        isfinite(u) || return Inf
        if f.mode === :lfs
            u = u - f.uc - f.duc * (r - f.rc)
            dudr = dudr - f.duc
        elseif f.mode === :shift
            u = u - f.uc
        end
        U_tot += u
        c = -dudr / r
        F[1, i] += c * d.x; F[2, i] += c * d.y; F[3, i] += c * d.z
        F[1, j] -= c * d.x; F[2, j] -= c * d.y; F[3, j] -= c * d.z
    end
    return U_tot
end

function accumulate_box_derivative!(G::Matrix{Float64}, f::CustomPairFactor,
                                    q::AbstractMatrix{Float64}, ctx::Factors.Context)
    ctx.box isa Types.NoBox && throw(Types.UnsupportedGeometry("box derivative requires a periodic box"))
    Factors.force_consistent(f) ||
        throw(ArgumentError("CustomPairFactor mode :$(f.mode) is not force-consistent"))
    has_cut = f.mode !== :none
    @inbounds for (i, j) in f.pairs
        d = Types.unwrap_displacement(ctx.box, q, i, j)
        r2 = Types.norm2(d)
        if r2 < 1e-24
            G .= Inf
            return G
        end
        r = sqrt(r2)
        if has_cut && r >= f.rc
            continue
        end
        u, dudr = _eval_pair_ud(f, r)
        if !isfinite(u) || !isfinite(dudr)
            G .= Inf
            return G
        end
        if f.mode === :lfs
            dudr -= f.duc
        end
        w = _frac_disp(ctx.box, d)
        _outer_add!(G, dudr / r, d, w)
    end
    return G
end

"""
    CustomExternalFactor(energy_fn, grad_fn, params, atom_indices)

Per-atom external potential wrapping `energy_fn(x, y, z, i, p) -> Float64` and
`grad_fn(x, y, z, i, p) -> (gx, gy, gz)` (where force is `-grad`).
"""
struct CustomExternalFactor{EF,GF} <: Factors.AbstractFactor
    energy_fn::EF
    grad_fn::GF
    params::Dict{Symbol,Float64}
    atom_indices::Vector{Int}
    function CustomExternalFactor(energy_fn::EF, grad_fn::GF,
                                  params::AbstractDict{Symbol,<:Real},
                                  atom_indices::AbstractVector{<:Integer}) where {EF,GF}
        pdict = Dict{Symbol,Float64}(k => Float64(v) for (k, v) in params)
        idx = Int.(collect(atom_indices))
        all(i -> i >= 1, idx) || throw(ArgumentError("atom_indices must be >= 1"))
        return new{EF,GF}(energy_fn, grad_fn, pdict, idx)
    end
end

Factors.factor_support_size(f::CustomExternalFactor) = length(f.atom_indices)
Factors.read_support(f::CustomExternalFactor) = Tuple(f.atom_indices)
Factors.is_local(::CustomExternalFactor) = true
Factors.force_consistent(::CustomExternalFactor) = true

function Factors.evaluate!(F::AbstractMatrix{Float64}, f::CustomExternalFactor,
                           q::AbstractMatrix{Float64}, ctx::Factors.Context)
    U = 0.0
    @inbounds for i in f.atom_indices
        x = q[1, i]; y = q[2, i]; z = q[3, i]
        U += Float64(f.energy_fn(x, y, z, i, f.params))
        g = f.grad_fn(x, y, z, i, f.params)
        F[1, i] -= Float64(g[1])
        F[2, i] -= Float64(g[2])
        F[3, i] -= Float64(g[3])
    end
    return U
end

function accumulate_box_derivative!(G::Matrix{Float64}, f::CustomExternalFactor,
                                    q::AbstractMatrix{Float64}, ctx::Factors.Context)
    ctx.box isa Types.NoBox && throw(Types.UnsupportedGeometry("box derivative requires a periodic box"))
    @inbounds for i in f.atom_indices
        x = q[1, i]; y = q[2, i]; z = q[3, i]
        g = f.grad_fn(x, y, z, i, f.params)
        w = _frac_disp(ctx.box, Types.SVector3(x, y, z))
        _outer_add!(G, 1.0, Float64(g[1]), Float64(g[2]), Float64(g[3]), w[1], w[2], w[3])
    end
    return G
end

"""
    CustomManyBodyFactor(eval_fn, params, n_atoms)

Many-body factor wrapping `eval_fn(q, box, params) -> (U, F, G_or_nothing)` where
`F = -∂U/∂q` is accumulated additively into `F_out`.
"""
struct CustomManyBodyFactor{EF} <: Factors.AbstractFactor
    eval_fn::EF
    params::Dict{Symbol,Float64}
    n_atoms::Int
    function CustomManyBodyFactor(eval_fn::EF, params::AbstractDict{Symbol,<:Real}, n_atoms::Int) where {EF}
        n_atoms >= 1 || throw(ArgumentError("n_atoms must be >= 1"))
        pdict = Dict{Symbol,Float64}(k => Float64(v) for (k, v) in params)
        return new{EF}(eval_fn, pdict, n_atoms)
    end
end

Factors.factor_support_size(f::CustomManyBodyFactor) = f.n_atoms
Factors.read_support(f::CustomManyBodyFactor) = Tuple(1:f.n_atoms)
Factors.is_local(::CustomManyBodyFactor) = false
Factors.force_consistent(::CustomManyBodyFactor) = true

function Factors.evaluate!(F::AbstractMatrix{Float64}, f::CustomManyBodyFactor,
                           q::AbstractMatrix{Float64}, ctx::Factors.Context)
    res = f.eval_fn(q, ctx.box, f.params)
    U = Float64(res[1])
    F_local = res[2]
    size(F_local) == size(F) ||
        throw(DimensionMismatch("CustomManyBodyFactor returned force array of size $(size(F_local)), expected $(size(F))"))
    F .+= F_local
    return U
end

function accumulate_box_derivative!(G::Matrix{Float64}, f::CustomManyBodyFactor,
                                    q::AbstractMatrix{Float64}, ctx::Factors.Context)
    ctx.box isa Types.NoBox && throw(Types.UnsupportedGeometry("box derivative requires a periodic box"))
    res = f.eval_fn(q, ctx.box, f.params)
    if length(res) >= 3 && res[3] !== nothing
        G_local = res[3]
        size(G_local) == (3, 3) || throw(DimensionMismatch("expected 3x3 box derivative G, got $(size(G_local))"))
        G .+= G_local
        return G
    end
    throw(ArgumentError("CustomManyBodyFactor eval_fn did not provide box derivative G"))
end

if _HAS_RESOLVED_SYSTEM
    ResolvedSystem.box_derivative_support(f::CustomPairFactor) =
        Factors.force_consistent(f) ? nothing : "mode :$(f.mode) is not force-consistent"
    ResolvedSystem.accumulate_box_derivative!(G::Matrix{Float64}, f::CustomPairFactor,
                                              q::AbstractMatrix{Float64}, ctx::Factors.Context) =
        accumulate_box_derivative!(G, f, q, ctx)

    ResolvedSystem.box_derivative_support(::CustomExternalFactor) = nothing
    ResolvedSystem.accumulate_box_derivative!(G::Matrix{Float64}, f::CustomExternalFactor,
                                              q::AbstractMatrix{Float64}, ctx::Factors.Context) =
        accumulate_box_derivative!(G, f, q, ctx)

    ResolvedSystem.box_derivative_support(::CustomManyBodyFactor) = nothing
    ResolvedSystem.accumulate_box_derivative!(G::Matrix{Float64}, f::CustomManyBodyFactor,
                                              q::AbstractMatrix{Float64}, ctx::Factors.Context) =
        accumulate_box_derivative!(G, f, q, ctx)
end

# ---------------------------------------------------------------------------
# Potential Registry
# ---------------------------------------------------------------------------

const POTENTIAL_REGISTRY = Dict{Symbol,Function}()

for k in _BUILTIN_KERNELS
    let kind = k
        POTENTIAL_REGISTRY[kind] = (params, pairs; kwargs...) -> CustomPairFactor(kind, params, pairs; kwargs...)
    end
end

"""Register a named custom potential builder `builder(params, args...; kwargs...)`."""
function register_potential!(name::Symbol, builder::Function; overwrite::Bool = true)
    (!overwrite && haskey(POTENTIAL_REGISTRY, name)) &&
        throw(ArgumentError("potential :$name is already registered"))
    POTENTIAL_REGISTRY[name] = builder
    return name
end

"""Sorted list of registered potential names."""
registered_potentials()::Vector{Symbol} = sort!(collect(keys(POTENTIAL_REGISTRY)))

"""Construct a registered potential factor by name."""
function build_custom_factor(name::Symbol, params::AbstractDict, args...; kwargs...)
    haskey(POTENTIAL_REGISTRY, name) ||
        throw(ArgumentError("unknown potential :$name; registered = $(registered_potentials())"))
    return POTENTIAL_REGISTRY[name](params, args...; kwargs...)
end

# ---------------------------------------------------------------------------
# Finite-difference gradient & box-derivative verification
# ---------------------------------------------------------------------------

function _box_to_matrix(box::Types.OrthorhombicBox)::Matrix{Float64}
    return [box.L[1] 0.0 0.0; 0.0 box.L[2] 0.0; 0.0 0.0 box.L[3]]
end
_box_to_matrix(box::Types.TriclinicBox)::Matrix{Float64} = copy(box.H)

function _matrix_to_box(H::Matrix{Float64}, ::Types.OrthorhombicBox, a::Int, b::Int)
    if a == b
        return Types.OrthorhombicBox((H[1, 1], H[2, 2], H[3, 3]))
    end
    return Types.TriclinicBox(H)
end
_matrix_to_box(H::Matrix{Float64}, ::Types.TriclinicBox, ::Int, ::Int) = Types.TriclinicBox(H)

"""
    verify_custom_factor_gradients(factor, q, box, units; h=1e-6, force_tol=1e-6, box_tol=1e-6, order=2)

Verify analytical forces (`-∂U/∂q`) and, when `box` is periodic, all 9 entries
of fixed-fractional box derivative (`∂U/∂h`) against central finite differences
(`order=2` or `order=4` 4-point stencil). Checks both absolute and relative error
tolerances and returns a diagnostic `NamedTuple`.
"""
function verify_custom_factor_gradients(factor::Factors.AbstractFactor,
                                        q::AbstractMatrix{Float64},
                                        box::Types.AbstractBox,
                                        units::Types.Units;
                                        h::Float64 = 1e-6,
                                        force_tol::Float64 = 1e-6,
                                        box_tol::Float64 = 1e-6,
                                        order::Int = 2)
    (order == 2 || order == 4) || throw(ArgumentError("FD order must be 2 or 4, got $order"))
    N = size(q, 2)
    ctx = Factors.Context(box, units, N)
    U0, F = Factors.total_energy_forces([factor], q, ctx)

    F_fd = zeros(Float64, 3, N)
    qp = copy(q)
    qm = copy(q)
    for j in 1:N, d in 1:3
        if order == 2
            qp[d, j] = q[d, j] + h
            qm[d, j] = q[d, j] - h
            up, _ = Factors.total_energy_forces([factor], qp, ctx)
            um, _ = Factors.total_energy_forces([factor], qm, ctx)
            F_fd[d, j] = -(up - um) / (2.0 * h)
        else
            qp[d, j] = q[d, j] + h;       up1, _ = Factors.total_energy_forces([factor], qp, ctx)
            qm[d, j] = q[d, j] - h;       um1, _ = Factors.total_energy_forces([factor], qm, ctx)
            qp[d, j] = q[d, j] + 2.0 * h; up2, _ = Factors.total_energy_forces([factor], qp, ctx)
            qm[d, j] = q[d, j] - 2.0 * h; um2, _ = Factors.total_energy_forces([factor], qm, ctx)
            F_fd[d, j] = -(-up2 + 8.0 * up1 - 8.0 * um1 + um2) / (12.0 * h)
        end
        qp[d, j] = q[d, j]
        qm[d, j] = q[d, j]
    end
    max_force_err = maximum(abs, F .- F_fd)
    rel_force_err = max_force_err / max(maximum(abs, F), 1.0)

    max_box_err = 0.0
    rel_box_err = 0.0
    G = zeros(Float64, 3, 3)
    G_fd = zeros(Float64, 3, 3)
    if !(box isa Types.NoBox)
        if _HAS_RESOLVED_SYSTEM && hasmethod(ResolvedSystem.accumulate_box_derivative!,
                                             Tuple{Matrix{Float64}, typeof(factor), Matrix{Float64}, Factors.Context})
            ResolvedSystem.accumulate_box_derivative!(G, factor, q, ctx)
        else
            accumulate_box_derivative!(G, factor, q, ctx)
        end
        H0 = _box_to_matrix(box)
        s = H0 \ q
        for a in 1:3, b in 1:3
            if order == 2
                Hp = copy(H0); Hp[a, b] += h
                Hm = copy(H0); Hm[a, b] -= h
                bp = _matrix_to_box(Hp, box, a, b)
                bm = _matrix_to_box(Hm, box, a, b)
                up, _ = Factors.total_energy_forces([factor], Hp * s, Factors.Context(bp, units, N))
                um, _ = Factors.total_energy_forces([factor], Hm * s, Factors.Context(bm, units, N))
                G_fd[a, b] = (up - um) / (2.0 * h)
            else
                Hp1 = copy(H0); Hp1[a, b] += h
                Hm1 = copy(H0); Hm1[a, b] -= h
                Hp2 = copy(H0); Hp2[a, b] += 2.0 * h
                Hm2 = copy(H0); Hm2[a, b] -= 2.0 * h
                up1, _ = Factors.total_energy_forces([factor], Hp1 * s, Factors.Context(_matrix_to_box(Hp1, box, a, b), units, N))
                um1, _ = Factors.total_energy_forces([factor], Hm1 * s, Factors.Context(_matrix_to_box(Hm1, box, a, b), units, N))
                up2, _ = Factors.total_energy_forces([factor], Hp2 * s, Factors.Context(_matrix_to_box(Hp2, box, a, b), units, N))
                um2, _ = Factors.total_energy_forces([factor], Hm2 * s, Factors.Context(_matrix_to_box(Hm2, box, a, b), units, N))
                G_fd[a, b] = (-up2 + 8.0 * up1 - 8.0 * um1 + um2) / (12.0 * h)
            end
        end
        max_box_err = maximum(abs, G .- G_fd)
        rel_box_err = max_box_err / max(maximum(abs, G), 1.0)
    end

    passed = isfinite(U0) && isfinite(max_force_err) && isfinite(max_box_err) &&
             min(max_force_err, rel_force_err) <= force_tol &&
             min(max_box_err, rel_box_err) <= box_tol
    return (
        passed = passed,
        max_force_err = max_force_err,
        rel_force_err = rel_force_err,
        max_box_err = max_box_err,
        rel_box_err = rel_box_err,
        U0 = U0,
        F = F,
        F_fd = F_fd,
        G = G,
        G_fd = G_fd,
    )
end

end # module CustomPhysics
