"""
TriumXBT — Balanced Ternary Architecture (XBT) Core, Carry-Free Force Accumulator,
and Fast Tiled Nonbonded Worker for Molecular Mechanics.

Implements:
1. Dual-Rail Balanced Ternary Arithmetic (`TritWord64`):
   - 0-cycle rail-swap negation
   - Bitwise BTHA (Half-Adder) and BTFA (Full-Adder / 3-to-2 compressor)
   - O(1) carry_free_add
   - Bounded ripple resolve (fast path) and 6-stage Kogge-Stone parallel prefix resolve
   - Encode/decode between Int64 and TritWord64
2. Carry-Free 3D Force Accumulator (`CarryFreeForceBuffer`):
   - Dual-rail (S, C) storage for (x, y, z) per atom
   - Permutation-invariant 0-ULP pair force scattering
   - Bit-exact assimilation into Cartesian force matrix
3. Native Bridge Loader (`libtrium_mm_bridge.dylib`):
   - Graceful detection with pure-Julia fallback
   - High-precision radix-3 range-reduced exponential (`xbt_exp_radix3`)
   - Cubic-convergence Householder inverse square root (`xbt_rsqrt_fast`)
   - High-accuracy complementary error function (`xbt_erfc_fast`)
4. TriumTiledNonbondedFactor <: Factors.AbstractFactor:
   - Unified Lennard-Jones (LFS / trunc) + Fennell–Gezelter DSF Coulomb
   - Orthorhombic and Triclinic box support with exact MIC
   - 1-4 scaling and exclusion handling via PairTopology
   - TilePairCache and PairCache support
   - Analytic Cartesian forces and 3-entry orthorhombic / 9-entry triclinic box derivatives
"""
module TriumXBT

using LinearAlgebra
const Libdl = Base.Libc.Libdl

const _PARENT = parentmodule(@__MODULE__)

# Resolve dependencies from parent module (FoundationCore)
using ..Types
using ..Factors
using ..Neighbors
using ..StreamedLJ
using ..Electrostatics

const _HAS_RESOLVED = isdefined(_PARENT, :ResolvedSystem)
const _RS = _HAS_RESOLVED ? getfield(_PARENT, :ResolvedSystem) : nothing

# Ensure extension points exist in Factors
if !isdefined(Factors, :accumulate_box_derivative!)
    @eval Factors function accumulate_box_derivative! end
end
if !isdefined(Factors, :energy_and_forces!)
    @eval Factors function energy_and_forces! end
end

export TritWord64, encode_tritword64, decode_tritword64,
       btha, btfa, kogge_stone_add64, carry_free_add,
       resolve_bounded_ripple, resolve_parallel,
       CarryFreeForceBuffer, reset!, scatter_pair_force!, resolve_forces!,
       has_native_bridge, init_bridge!,
       xbt_exp_radix3, xbt_rsqrt_fast, xbt_erfc_fast,
       xbt_csa_accumulate_array, eval_lj_dsf_pairs!,
       TriumTiledNonbondedFactor, energy_and_forces!,
       accumulate_box_derivative!, accumulate_box_derivative,
       DifferenceSetTileSchedule, schedule_disjoint_tile_orbits,
       verify_disjoint_orbits,
       TriumCompiledVerletPlan, compiled_verlet_num_pairs,
       eval_lj_dsf_direct_into!, clear_verlet_plan_cache!,
       TriumMultiRateConfig, effective_ns_per_day

# ==============================================================================
# 1. Dual-Rail Balanced Ternary Arithmetic
# ==============================================================================

"""
    TritWord64(P::UInt64, N::UInt64)

Register-width bit-sliced 64-trit planar storage.
- `P`: Positive rail mask (1 where trit is +1)
- `N`: Negative rail mask (1 where trit is -1)
- Invariant: `(P & N) == 0` (positive and negative rails must be disjoint).
"""
struct TritWord64
    P::UInt64
    N::UInt64
    function TritWord64(P::UInt64, N::UInt64)
        @assert (P & N) == 0 "Dual-rail invariant violated: P and N must be disjoint!"
        new(P, N)
    end
end

TritWord64() = TritWord64(UInt64(0), UInt64(0))
TritWord64(P::Integer, N::Integer) = TritWord64(UInt64(P), UInt64(N))

Base.:(==)(a::TritWord64, b::TritWord64) = (a.P == b.P) && (a.N == b.N)
Base.iszero(w::TritWord64) = (w.P == 0) && (w.N == 0)

# 0-Cycle Negation: compile-time rail swap
Base.:-(w::TritWord64) = TritWord64(w.N, w.P)

"""
    encode_tritword64(val::Int64, width::Int=40) -> TritWord64

Encodes an integer into a canonical balanced ternary `TritWord64`.
Each trit k ∈ {-1, 0, 1} satisfies val = sum_{k=0}^{width-1} trit_k * 3^k.
"""
function encode_tritword64(val::Int64, width::Int = 40)::TritWord64
    P = UInt64(0)
    N = UInt64(0)
    q = val
    w_lim = clamp(width, 1, 64)
    for k in 0:(w_lim - 1)
        q == 0 && break
        rem = q % 3
        q = div(q, 3)
        if rem == 2
            rem = -1
            q += 1
        elseif rem == -2
            rem = 1
            q -= 1
        end
        if rem == 1
            P |= (UInt64(1) << k)
        elseif rem == -1
            N |= (UInt64(1) << k)
        end
    end
    return TritWord64(P, N)
end

"""
    decode_tritword64(w::TritWord64, width::Int=40) -> Int64

Decodes a balanced ternary `TritWord64` back into an `Int64`.
"""
function decode_tritword64(w::TritWord64, width::Int = 40)::Int64
    val = Int128(0)
    p3 = Int128(1)
    w_lim = clamp(width, 1, 64)
    for k in 0:(w_lim - 1)
        pk = (w.P >> k) & UInt64(1)
        nk = (w.N >> k) & UInt64(1)
        if pk == 1
            val += p3
        elseif nk == 1
            val -= p3
        end
        if k < w_lim - 1
            p3 *= Int128(3)
        end
    end
    return Int64(val)
end

"""
    btha(a::TritWord64, b::TritWord64) -> (sum::TritWord64, carry::TritWord64)

Exact Balanced Ternary Half-Adder (BTHA) across 64 trits in parallel.
Matches `ternary_xbt_foundation.hpp` / `ternary_core.hpp`.
"""
@inline function btha(a::TritWord64, b::TritWord64)::Tuple{TritWord64, TritWord64}
    PC = a.P & b.P
    NC = a.N & b.N
    PS = ((a.P ⊻ b.P) & ~(a.N | b.N)) | NC
    NS = ((a.N ⊻ b.N) & ~(a.P | b.P)) | PC
    return (TritWord64(PS, NS), TritWord64(PC, NC))
end

"""
    btfa(a::TritWord64, b::TritWord64, d::TritWord64) -> (sum::TritWord64, carry::TritWord64)

Exact Balanced Ternary Full-Adder (BTFA / 3-to-2 Compressor).
Compresses 3 input words (a, b, d) into (sum, carry) without carry ripple.
"""
@inline function btfa(a::TritWord64, b::TritWord64, d::TritWord64)::Tuple{TritWord64, TritWord64}
    s1, c1 = btha(a, b)
    s2, c2 = btha(s1, d)
    PC = (c1.P | c2.P) & ~(c1.N | c2.N)
    NC = (c1.N | c2.N) & ~(c1.P | c2.P)
    return (s2, TritWord64(PC, NC))
end

"""
    carry_free_add(S::TritWord64, C::TritWord64, X::TritWord64) -> (S_new::TritWord64, C_new::TritWord64)

One O(1) accumulation step of the symplectic-ternary carry-free accumulator.
val(S_new) + 3 * val(C_new) = val(S) + 3 * val(C) + val(X).
"""
@inline function carry_free_add(S::TritWord64, C::TritWord64, X::TritWord64)::Tuple{TritWord64, TritWord64}
    C_shifted = TritWord64(C.P << 1, C.N << 1)
    return btfa(S, C_shifted, X)
end

"""
    kogge_stone_add64(a::TritWord64, b::TritWord64, cin::Int8 = Int8(0)) -> (sum::TritWord64, cout_msb::Int8)

Branchless 6-Step Kogge-Stone Parallel Prefix Balanced Ternary Adder across 64 trits.
Uses the closed 7-state monotone carry transfer semigroup.
"""
function kogge_stone_add64(a::TritWord64, b::TritWord64, cin::Int8 = Int8(0))::Tuple{TritWord64, Int8}
    st1_sum, st1_carry = btha(a, b)
    f0_p = st1_carry.P
    f0_n = st1_carry.N
    d_p  = st1_sum.P
    d_n  = st1_sum.N

    fp_p = (a.P | b.P) & ~(a.N | b.N)
    fp_n = UInt64(0)
    fm_p = UInt64(0)
    fm_n = (a.N | b.N) & ~(a.P | b.P)

    s1_p = d_p
    s1_n = d_n
    if cin != 0
        bit0_d = TritWord64(d_p & UInt64(1), d_n & UInt64(1))
        bit0_cin = TritWord64(cin > 0 ? UInt64(1) : UInt64(0), cin < 0 ? UInt64(1) : UInt64(0))
        bit0_ha_sum, bit0_ha_c = btha(bit0_d, bit0_cin)
        f0_p = (f0_p | bit0_ha_c.P) & ~(f0_n | bit0_ha_c.N)
        f0_n = (st1_carry.N | bit0_ha_c.N) & ~(st1_carry.P | bit0_ha_c.P)
        s1_p = (d_p & ~UInt64(1)) | bit0_ha_sum.P
        s1_n = (d_n & ~UInt64(1)) | bit0_ha_sum.N
    end

    for shift in (1, 2, 4, 8, 16)
        l_fm_p = fm_p << shift
        l_fm_n = fm_n << shift
        l_f0_p = f0_p << shift
        l_f0_n = f0_n << shift
        l_fp_p = fp_p << shift
        l_fp_n = fp_n << shift

        n_fm_p = fm_p | (f0_p & ~l_fm_n) | (fp_p & l_fm_p)
        n_fm_n = fp_n | (f0_n & ~l_fm_p) | (fm_n & l_fm_n)

        n_f0_p = fm_p | (f0_p & ~l_f0_n) | (fp_p & l_f0_p)
        n_f0_n = fp_n | (f0_n & ~l_f0_p) | (fm_n & l_f0_n)

        n_fp_p = fm_p | (f0_p & ~l_fp_n) | (fp_p & l_fp_p)
        n_fp_n = fp_n | (f0_n & ~l_fp_p) | (fm_n & l_fp_n)

        fm_p = n_fm_p; fm_n = n_fm_n
        f0_p = n_f0_p; f0_n = n_f0_n
        fp_p = n_fp_p; fp_n = n_fp_n
    end

    # Stage 5 (shift = 32): cout
    l_f0_p = f0_p << 32
    l_f0_n = f0_n << 32
    cout_p = fm_p | (f0_p & ~l_f0_n) | (fp_p & l_f0_p)
    cout_n = fp_n | (f0_n & ~l_f0_p) | (fm_n & l_f0_n)

    cout_msb = Int8(((cout_p >> 63) & UInt64(1)) - ((cout_n >> 63) & UInt64(1)))
    st2_sum, _ = btha(TritWord64(s1_p, s1_n), TritWord64(cout_p << 1, cout_n << 1))
    return (st2_sum, cout_msb)
end

"""
    resolve_bounded_ripple(S::TritWord64, C::TritWord64, width::Int=40) -> (TritWord64, Bool)

Fast data-dependent bounded ripple assimilation.
Resolves (S, C) to a single canonical TritWord64. Returns `(canonical_sum, overflow)`.
"""
function resolve_bounded_ripple(S::TritWord64, C::TritWord64, width::Int = 40)::Tuple{TritWord64, Bool}
    s = S
    c = C
    shift_limit = min(width - 1, 63)
    overflow = ((c.P >> shift_limit) & UInt64(1)) != 0 || ((c.N >> shift_limit) & UInt64(1)) != 0

    for iter in 1:64
        if c.P == 0 && c.N == 0
            return (s, overflow)
        end
        if ((c.P >> shift_limit) & UInt64(1)) != 0 || ((c.N >> shift_limit) & UInt64(1)) != 0
            overflow = true
        end
        sh = TritWord64(c.P << 1, c.N << 1)
        ha_s, ha_c = btha(s, sh)
        s = ha_s
        c = ha_c
    end
    return (s, true) # failed to converge within bound
end

"""
    resolve_parallel(S::TritWord64, C::TritWord64, width::Int=40) -> (TritWord64, Bool)

Fixed 6-stage Kogge-Stone parallel-prefix restore.
Resolves redundant (S, C) to canonical form with O(log w) latency.
"""
function resolve_parallel(S::TritWord64, C::TritWord64, width::Int = 40)::Tuple{TritWord64, Bool}
    shift_limit = min(width - 1, 63)
    overflow = ((C.P >> shift_limit) & UInt64(1)) != 0 || ((C.N >> shift_limit) & UInt64(1)) != 0
    C_shifted = TritWord64(C.P << 1, C.N << 1)
    res, cout = kogge_stone_add64(S, C_shifted)
    if cout != 0
        overflow = true
    end
    return (res, overflow)
end

# ==============================================================================
# 2. Carry-Free 3D Force Accumulator Buffer
# ==============================================================================

"""
    CarryFreeForceBuffer(n_atoms::Int; scale::Float64 = Float64(Int64(1) << 28), width::Int = 40)

Carry-free force accumulator storing redundant (S, C) state across all 3 Cartesian axes.
Guarantees strict bitwise permutation invariance (0 ULP difference) during accumulation.
"""
mutable struct CarryFreeForceBuffer
    Sx::Vector{TritWord64}
    Cx::Vector{TritWord64}
    Sy::Vector{TritWord64}
    Cy::Vector{TritWord64}
    Sz::Vector{TritWord64}
    Cz::Vector{TritWord64}
    scale::Float64
    inv_scale::Float64
    width::Int

    function CarryFreeForceBuffer(n_atoms::Int;
                                  scale::Float64 = Float64(Int64(1) << 28),
                                  width::Int = 40)
        z = TritWord64(UInt64(0), UInt64(0))
        new(fill(z, n_atoms), fill(z, n_atoms),
            fill(z, n_atoms), fill(z, n_atoms),
            fill(z, n_atoms), fill(z, n_atoms),
            scale, 1.0 / scale, width)
    end
end

"""
    reset!(buf::CarryFreeForceBuffer) -> buf

Resets all redundant (S, C) vectors to 0.
"""
function reset!(buf::CarryFreeForceBuffer)
    z = TritWord64(UInt64(0), UInt64(0))
    fill!(buf.Sx, z); fill!(buf.Cx, z)
    fill!(buf.Sy, z); fill!(buf.Cy, z)
    fill!(buf.Sz, z); fill!(buf.Cz, z)
    return buf
end

"""
    scatter_pair_force!(buf::CarryFreeForceBuffer, i::Int, j::Int, fx::Float64, fy::Float64, fz::Float64)

Encodes `(fx, fy, fz)` into `TritWord64`, adds `+w` to atom `i` via `carry_free_add`,
and adds `-w` (0-cycle rail swap!) to atom `j` via `carry_free_add`.
"""
@inline function scatter_pair_force!(buf::CarryFreeForceBuffer, i::Int, j::Int,
                                     fx::Float64, fy::Float64, fz::Float64)
    sc = buf.scale
    w = buf.width

    ix = round(Int64, fx * sc)
    iy = round(Int64, fy * sc)
    iz = round(Int64, fz * sc)

    wx = encode_tritword64(ix, w)
    wy = encode_tritword64(iy, w)
    wz = encode_tritword64(iz, w)

    # +w for atom i
    @inbounds begin
        buf.Sx[i], buf.Cx[i] = carry_free_add(buf.Sx[i], buf.Cx[i], wx)
        buf.Sy[i], buf.Cy[i] = carry_free_add(buf.Sy[i], buf.Cy[i], wy)
        buf.Sz[i], buf.Cz[i] = carry_free_add(buf.Sz[i], buf.Cz[i], wz)

        # -w for atom j (0-cycle rail swap!)
        buf.Sx[j], buf.Cx[j] = carry_free_add(buf.Sx[j], buf.Cx[j], -wx)
        buf.Sy[j], buf.Cy[j] = carry_free_add(buf.Sy[j], buf.Cy[j], -wy)
        buf.Sz[j], buf.Cz[j] = carry_free_add(buf.Sz[j], buf.Cz[j], -wz)
    end
    return nothing
end

"""
    resolve_forces!(F::AbstractMatrix{Float64}, buf::CarryFreeForceBuffer; accumulate::Bool = true)

Resolves redundant (S, C) vectors into canonical Float64 Cartesian forces in `F`.
"""
function resolve_forces!(F::AbstractMatrix{Float64}, buf::CarryFreeForceBuffer;
                         accumulate::Bool = true)
    n = length(buf.Sx)
    size(F, 1) == 3 && size(F, 2) >= n ||
        throw(DimensionMismatch("F must be 3 x $(size(F, 2)), expected >= $n"))
    inv_sc = buf.inv_scale
    w = buf.width

    @inbounds for i in 1:n
        rx, _ = resolve_bounded_ripple(buf.Sx[i], buf.Cx[i], w)
        ry, _ = resolve_bounded_ripple(buf.Sy[i], buf.Cy[i], w)
        rz, _ = resolve_bounded_ripple(buf.Sz[i], buf.Cz[i], w)

        fx = Float64(decode_tritword64(rx, w)) * inv_sc
        fy = Float64(decode_tritword64(ry, w)) * inv_sc
        fz = Float64(decode_tritword64(rz, w)) * inv_sc

        if accumulate
            F[1, i] += fx
            F[2, i] += fy
            F[3, i] += fz
        else
            F[1, i] = fx
            F[2, i] = fy
            F[3, i] = fz
        end
    end
    return F
end

# ==============================================================================
# 3. Native C++ Bridge Loader & Fast Math Fallbacks
# ==============================================================================

function _candidate_bridge_paths()::Vector{String}
    ext = Libdl.dlext
    fnames = ("libtrium_mm_bridge." * ext, "trium_mm_bridge." * ext)
    paths = String[]
    env_dir = strip(get(ENV, "BIOLOGY_MM_LIB_DIR", ""))
    if !isempty(env_dir)
        for f in fnames
            push!(paths, joinpath(expanduser(env_dir), f))
        end
    end
    for dir in (
        joinpath(@__DIR__, "..", "lib"),
        joinpath(@__DIR__, "..", "..", "native", "build"),
    )
        for f in fnames
            push!(paths, joinpath(dir, f))
        end
    end
    for f in fnames
        push!(paths, f)
    end
    return paths
end

const _BRIDGE_LOCK = ReentrantLock()
const _NATIVE_BRIDGE_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
const _NATIVE_BRIDGE_PATH = Ref{String}("")

const _SYM_EVAL_PAIRS_MT     = Ref{Ptr{Cvoid}}(C_NULL)
const _SYM_EVAL_TILES_MT     = Ref{Ptr{Cvoid}}(C_NULL)
const _SYM_SCHED_ORBITS      = Ref{Ptr{Cvoid}}(C_NULL)
const _SYM_MATVEC_MULTIPLANE = Ref{Ptr{Cvoid}}(C_NULL)
const _SYM_VERLET_CREATE     = Ref{Ptr{Cvoid}}(C_NULL)
const _SYM_VERLET_EVAL_MT    = Ref{Ptr{Cvoid}}(C_NULL)
const _SYM_VERLET_DESTROY    = Ref{Ptr{Cvoid}}(C_NULL)
const _SYM_VERLET_NUM_PAIRS  = Ref{Ptr{Cvoid}}(C_NULL)

@inline function _dlsym_safe(h::Ptr{Cvoid}, sym::Symbol)::Ptr{Cvoid}
    h == C_NULL && return C_NULL
    p = Libdl.dlsym(h, sym; throw_error = false)
    return p === nothing ? C_NULL : p
end

function _load_bridge_symbols!(h::Ptr{Cvoid})
    _SYM_EVAL_PAIRS_MT[]     = _dlsym_safe(h, :xbt_eval_lj_dsf_pairs_mt)
    _SYM_EVAL_TILES_MT[]     = _dlsym_safe(h, :xbt_eval_lj_dsf_tiles_mt)
    _SYM_SCHED_ORBITS[]      = _dlsym_safe(h, :xbt_schedule_disjoint_tile_orbits)
    _SYM_MATVEC_MULTIPLANE[] = _dlsym_safe(h, :xbt_ternary_matvec_multiplane)
    _SYM_VERLET_CREATE[]     = _dlsym_safe(h, :xbt_verlet_plan_create)
    _SYM_VERLET_EVAL_MT[]    = _dlsym_safe(h, :xbt_verlet_plan_eval_mt)
    _SYM_VERLET_DESTROY[]    = _dlsym_safe(h, :xbt_verlet_plan_destroy)
    _SYM_VERLET_NUM_PAIRS[]  = _dlsym_safe(h, :xbt_verlet_plan_num_pairs)
    return nothing
end

"""
    init_bridge!(path::Union{String, Nothing} = nothing) -> Bool

Attempts to load the native C++ TriumXBT acceleration bridge shared library and resolves
all optional high-throughput kernel symbols gracefully.
"""
function init_bridge!(path::Union{String, Nothing} = nothing)::Bool
    lock(_BRIDGE_LOCK)
    try
        if path !== nothing
            if path == basename(path) || isfile(path)
                h = Libdl.dlopen_e(path)
                if h != C_NULL
                    _NATIVE_BRIDGE_HANDLE[] = h
                    _NATIVE_BRIDGE_PATH[] = path
                    _load_bridge_symbols!(h)
                    return true
                end
            end
            return false
        end
        for p in _candidate_bridge_paths()
            if p == basename(p) || isfile(p)
                h = Libdl.dlopen_e(p)
                if h != C_NULL
                    _NATIVE_BRIDGE_HANDLE[] = h
                    _NATIVE_BRIDGE_PATH[] = p
                    _load_bridge_symbols!(h)
                    return true
                end
            end
        end
        return false
    finally
        unlock(_BRIDGE_LOCK)
    end
end

"""
    has_native_bridge() -> Bool

Returns `true` if the native C++ bridge shared library is loaded and active.
"""
function has_native_bridge()::Bool
    if _NATIVE_BRIDGE_HANDLE[] == C_NULL
        init_bridge!()
    end
    return _NATIVE_BRIDGE_HANDLE[] != C_NULL
end

# Fast math constants for pure-Julia fallback
const _LN3       = 1.09861228866810969139524523692252570464749055782275
const _INV_LN3   = 0.91023922662683739361424016575626764282384607855666
const _CBRT3     = 1.44224957030740838232163831078010958839186925349935
const _INV_CBRT3 = 0.69336127435063470484335227428596433944529357221827

"""
    xbt_exp_radix3(x::Float64) -> Float64

Fast Radix-3 range-reduced exponential: e^x = 3^k * e^r.
Evaluates e^r via a degree-7 Horner polynomial on [-ln(3)/6, +ln(3)/6].
Dispatches to native C++ bridge if loaded, otherwise uses verified Julia implementation.
"""
function xbt_exp_radix3(x::Float64)::Float64
    if has_native_bridge()
        sym = Libdl.dlsym_e(_NATIVE_BRIDGE_HANDLE[], :xbt_exp_radix3)
        if sym != C_NULL
            return ccall(sym, Cdouble, (Cdouble,), x)
        end
    end
    # Pure-Julia verified radix-3 implementation
    if !isfinite(x) || x > 700.0 || x < -700.0
        return exp(x)
    end
    u = round(x * (3.0 * _INV_LN3))
    u_int = Int64(u)
    r = x - u * (_LN3 / 3.0)
    p = 1.0 + r * (1.0 + r * (0.5 + r * ((1.0 / 6.0) + r * ((1.0 / 24.0) +
        r * ((1.0 / 120.0) + r * ((1.0 / 720.0) + r * (1.0 / 5040.0)))))))
    k = div(u_int, 3)
    rem_k = u_int % 3
    if rem_k == 2
        rem_k = -1
        k += 1
    elseif rem_k == -2
        rem_k = 1
        k -= 1
    end
    if rem_k == 1
        p *= _CBRT3
    elseif rem_k == -1
        p *= _INV_CBRT3
    end
    return p * (3.0^k)
end

"""
    xbt_rsqrt_fast(x::Float64) -> Float64

3rd-Order Cubic Householder Inverse Square Root (1 / sqrt(x)).
Dispatches to native C++ bridge if loaded, otherwise uses exact 1.0 / sqrt(x).
"""
function xbt_rsqrt_fast(x::Float64)::Float64
    if has_native_bridge()
        sym = Libdl.dlsym_e(_NATIVE_BRIDGE_HANDLE[], :xbt_rsqrt_fast)
        if sym != C_NULL
            return ccall(sym, Cdouble, (Cdouble,), x)
        end
    end
    return 1.0 / sqrt(x)
end

"""
    xbt_erfc_fast(x::Float64) -> Float64

High-accuracy complementary error function.
Dispatches to native C++ bridge if loaded, otherwise uses libm erfc.
"""
function xbt_erfc_fast(x::Float64)::Float64
    if has_native_bridge()
        sym = Libdl.dlsym_e(_NATIVE_BRIDGE_HANDLE[], :xbt_erfc_fast)
        if sym != C_NULL
            return ccall(sym, Cdouble, (Cdouble,), x)
        end
    end
    return ccall(:erfc, Cdouble, (Cdouble,), x)
end

"""
    xbt_csa_accumulate_array(words::Vector{TritWord64}) -> (S::TritWord64, C::TritWord64)

Accumulates an array of TritWord64 values into redundant (S, C) state.
"""
function xbt_csa_accumulate_array(words::Vector{TritWord64})::Tuple{TritWord64, TritWord64}
    S = TritWord64(UInt64(0), UInt64(0))
    C = TritWord64(UInt64(0), UInt64(0))
    for w in words
        S, C = carry_free_add(S, C, w)
    end
    return (S, C)
end

# ==============================================================================
# 3B. Difference-Set Disjoint Tile Orbit Scheduler (GSI-R-0004 / GSI-R-0012)
# ==============================================================================

"""
    DifferenceSetTileSchedule

Conflict-free CSR schedule partitioning interacting tile pairs `(tile_i[k], tile_j[k])`
into `n_orbits` disjoint orbits such that no two tile pairs inside the same orbit share
a tile index. Enables lock-free, race-free multi-threaded execution.
"""
struct DifferenceSetTileSchedule
    n_tiles::Int
    n_orbits::Int
    orbit_ids::Vector{Int32}
    orbit_offsets::Vector{Int32}
    orbit_perm::Vector{Int32}
end

function _schedule_disjoint_tile_orbits_julia!(
    orbit_ids::Vector{Int32},
    n_tiles::Int,
    tile_i::Vector{Int32},
    tile_j::Vector{Int32}
)::Int
    np = length(tile_i)
    words_per_tile = 4
    tile_mask = zeros(UInt64, n_tiles * words_per_tile)
    n_orbits = 0

    @inbounds for k in 1:np
        ti = Int(tile_i[k])
        tj = Int(tile_j[k])
        (1 <= ti <= n_tiles && 1 <= tj <= n_tiles) ||
            throw(ArgumentError("tile index ($ti, $tj) out of bounds 1:$n_tiles"))
        u_base = (ti - 1) * words_per_tile
        v_base = (tj - 1) * words_per_tile

        assigned = 0
        for w in 1:words_per_tile
            avail = ~(tile_mask[u_base + w] | tile_mask[v_base + w])
            if avail != UInt64(0)
                bit = trailing_zeros(avail)
                bit_mask = UInt64(1) << bit
                tile_mask[u_base + w] |= bit_mask
                tile_mask[v_base + w] |= bit_mask
                assigned = (w - 1) * 64 + bit + 1
                break
            end
        end

        if assigned == 0
            old_w = words_per_tile
            new_w = old_w * 2
            grown = zeros(UInt64, n_tiles * new_w)
            for t in 1:n_tiles
                src_off = (t - 1) * old_w
                dst_off = (t - 1) * new_w
                for w in 1:old_w
                    grown[dst_off + w] = tile_mask[src_off + w]
                end
            end
            tile_mask = grown
            words_per_tile = new_w
            tile_mask[(ti - 1) * words_per_tile + old_w + 1] |= UInt64(1)
            tile_mask[(tj - 1) * words_per_tile + old_w + 1] |= UInt64(1)
            assigned = old_w * 64 + 1
        end

        orbit_ids[k] = Int32(assigned)
        if assigned > n_orbits
            n_orbits = assigned
        end
    end
    return n_orbits
end

"""
    schedule_disjoint_tile_orbits(n_tiles, tile_i, tile_j) -> DifferenceSetTileSchedule

Partitions tile pairs `(tile_i[k], tile_j[k])` into strictly tile-disjoint orbits and
builds a 1-based CSR permutation (`orbit_offsets`, `orbit_perm`) grouped by orbit.
Dispatches to `xbt_schedule_disjoint_tile_orbits` when the native bridge is available,
otherwise uses the verified bitset greedy fallback.
"""
function schedule_disjoint_tile_orbits(
    n_tiles::Integer,
    tile_i::AbstractVector{<:Integer},
    tile_j::AbstractVector{<:Integer}
)::DifferenceSetTileSchedule
    nt = Int(n_tiles)
    nt >= 0 || throw(ArgumentError("n_tiles must be non-negative, got $nt"))
    np = length(tile_i)
    length(tile_j) == np ||
        throw(ArgumentError("tile_i and tile_j must have equal lengths ($np vs $(length(tile_j)))"))

    if np == 0
        return DifferenceSetTileSchedule(nt, 0, Int32[], Int32[1], Int32[])
    end

    ti_v = tile_i isa Vector{Int32} ? tile_i : Int32.(tile_i)
    tj_v = tile_j isa Vector{Int32} ? tile_j : Int32.(tile_j)
    orbit_ids = Vector{Int32}(undef, np)

    n_orbits = 0
    if has_native_bridge() && _SYM_SCHED_ORBITS[] != C_NULL
        n_orb = GC.@preserve ti_v tj_v orbit_ids ccall(
            _SYM_SCHED_ORBITS[], Int32,
            (Csize_t, Csize_t, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}),
            Csize_t(nt), Csize_t(np), pointer(ti_v), pointer(tj_v), pointer(orbit_ids)
        )
        n_orb < 0 && throw(ArgumentError("invalid tile index in schedule_disjoint_tile_orbits"))
        @inbounds for k in 1:np
            orbit_ids[k] += Int32(1)
        end
        n_orbits = Int(n_orb)
    else
        n_orbits = _schedule_disjoint_tile_orbits_julia!(orbit_ids, nt, ti_v, tj_v)
    end

    counts = zeros(Int32, n_orbits)
    @inbounds for k in 1:np
        counts[orbit_ids[k]] += Int32(1)
    end

    orbit_offsets = Vector{Int32}(undef, n_orbits + 1)
    orbit_offsets[1] = Int32(1)
    @inbounds for o in 1:n_orbits
        orbit_offsets[o + 1] = orbit_offsets[o] + counts[o]
    end

    cursor = copy(orbit_offsets)
    orbit_perm = Vector{Int32}(undef, np)
    @inbounds for k in 1:np
        o = orbit_ids[k]
        pos = cursor[o]
        orbit_perm[pos] = Int32(k)
        cursor[o] = pos + Int32(1)
    end

    return DifferenceSetTileSchedule(nt, n_orbits, orbit_ids, orbit_offsets, orbit_perm)
end

"""
    verify_disjoint_orbits(sched::DifferenceSetTileSchedule, tile_i, tile_j) -> Bool

Verifies that `sched` is a valid CSR permutation of `1:length(tile_i)` and that every
tile index appears at most once within each orbit.
"""
function verify_disjoint_orbits(
    sched::DifferenceSetTileSchedule,
    tile_i::AbstractVector{<:Integer},
    tile_j::AbstractVector{<:Integer}
)::Bool
    np = length(tile_i)
    length(tile_j) == np || return false
    length(sched.orbit_ids) == np || return false
    length(sched.orbit_perm) == np || return false
    sched.n_orbits >= 0 || return false
    length(sched.orbit_offsets) == sched.n_orbits + 1 || return false

    if np == 0
        return sched.n_orbits == 0 && sched.orbit_offsets[1] == Int32(1)
    end

    sched.orbit_offsets[1] == Int32(1) || return false
    sched.orbit_offsets[end] == Int32(np + 1) || return false

    seen_pair = falses(np)
    tile_stamp = zeros(Int32, sched.n_tiles)

    @inbounds for o in 1:sched.n_orbits
        s = Int(sched.orbit_offsets[o])
        e = Int(sched.orbit_offsets[o + 1]) - 1
        s <= e + 1 || return false
        o32 = Int32(o)
        for idx in s:e
            k = Int(sched.orbit_perm[idx])
            (1 <= k <= np && !seen_pair[k]) || return false
            seen_pair[k] = true
            sched.orbit_ids[k] == o32 || return false

            ti = Int(tile_i[k])
            tj = Int(tile_j[k])
            (1 <= ti <= sched.n_tiles && 1 <= tj <= sched.n_tiles) || return false
            tile_stamp[ti] == o32 && return false
            tile_stamp[ti] = o32
            if tj != ti
                tile_stamp[tj] == o32 && return false
                tile_stamp[tj] = o32
            end
        end
    end

    return all(seen_pair)
end

# ==============================================================================
# 4. TriumTiledNonbondedFactor & Unified Pair Evaluator
# ==============================================================================

"""
    TriumTiledNonbondedFactor <: Factors.AbstractFactor

Unified Lennard-Jones (LFS or trunc) + Fennell–Gezelter DSF Coulomb nonbonded factor.
Supports:
- Orthorhombic and Triclinic periodic cells with exact MIC
- 1-4 scaling and exclusions via `PairTopology`
- Direct pair iteration, `PairCache`, or `TilePairCache`
- `use_carry_free = true` (0-ULP permutation-invariant accumulation)
- `use_native_bridge = true` (accelerated C++ path when available)
- `use_fast_math = true` (radix-3 exponential & fast rsqrt/erfc)
"""
struct TriumTiledNonbondedFactor <: Factors.AbstractFactor
    sigma::Vector{Float64}
    epsilon::Vector{Float64}
    charges::Vector{Float64}
    rcut::Float64
    alpha::Float64
    coulomb_prefactor::Float64
    lj_mode::Symbol
    lj14_scale::Float64
    coul14_scale::Float64
    topology::Union{Nothing, StreamedLJ.PairTopology}
    cache::Union{Nothing, Neighbors.PairCache, Neighbors.TilePairCache}
    use_carry_free::Bool
    use_native_bridge::Bool
    use_fast_math::Bool
    include_self::Bool

    function TriumTiledNonbondedFactor(
        sigma::AbstractVector{<:Real},
        epsilon::AbstractVector{<:Real},
        charges::AbstractVector{<:Real};
        rcut::Real = 1.0,
        alpha::Real = 0.25,
        coulomb_prefactor::Real = 1.0,
        lj_mode::Symbol = :lfs,
        lj14_scale::Real = 0.5,
        coul14_scale::Real = 0.8333333333333334,
        topology::Union{Nothing, StreamedLJ.PairTopology} = nothing,
        cache::Union{Nothing, Neighbors.PairCache, Neighbors.TilePairCache} = nothing,
        use_carry_free::Bool = false,
        use_native_bridge::Bool = true,
        use_fast_math::Bool = false,
        include_self::Bool = false
    )
        n = length(charges)
        length(sigma) == n && length(epsilon) == n ||
            throw(ArgumentError("sigma, epsilon, charges must have identical lengths"))
        all(isfinite, sigma) && all(>(0.0), sigma) ||
            throw(ArgumentError("sigma values must be positive and finite"))
        all(isfinite, epsilon) && all(>=(0.0), epsilon) ||
            throw(ArgumentError("epsilon values must be non-negative and finite"))
        all(isfinite, charges) ||
            throw(ArgumentError("charges must be finite"))
        rc = Float64(rcut)
        rc > 0.0 && isfinite(rc) ||
            throw(ArgumentError("rcut must be positive and finite"))
        a = Float64(alpha)
        a >= 0.0 && isfinite(a) ||
            throw(ArgumentError("alpha must be non-negative and finite"))
        (lj_mode === :lfs || lj_mode === :trunc) ||
            throw(ArgumentError("lj_mode must be :lfs or :trunc, got $(repr(lj_mode))"))

        new(collect(Float64, sigma), collect(Float64, epsilon), collect(Float64, charges),
            rc, a, Float64(coulomb_prefactor), lj_mode, Float64(lj14_scale),
            Float64(coul14_scale), topology, cache, use_carry_free,
            use_native_bridge, use_fast_math, include_self)
    end
end

Factors.factor_support_size(fac::TriumTiledNonbondedFactor) = length(fac.charges)
Factors.read_support(fac::TriumTiledNonbondedFactor) = Tuple(1:length(fac.charges))
Factors.is_local(::TriumTiledNonbondedFactor) = false
Factors.force_consistent(fac::TriumTiledNonbondedFactor) = fac.lj_mode === :lfs

# Topology scale helper
@inline function _pair_scales(fac::TriumTiledNonbondedFactor, i::Int, j::Int)::Tuple{Float64, Float64}
    topo = fac.topology
    topo === nothing && return (1.0, 1.0)
    ii, jj = i < j ? (i, j) : (j, i)
    key = (UInt64(ii) << 32) | UInt64(jj)
    if !isempty(topo.excluded)
        pos = searchsortedfirst(topo.excluded, key)
        if pos <= length(topo.excluded) && topo.excluded[pos] == key
            return (0.0, 0.0)
        end
    end
    if !isempty(topo.scaled_keys)
        pos = searchsortedfirst(topo.scaled_keys, key)
        if pos <= length(topo.scaled_keys) && topo.scaled_keys[pos] == key
            w = topo.scaled_weights[pos]
            lj_sc = (w != 1.0) ? w : fac.lj14_scale
            return (lj_sc, fac.coul14_scale)
        end
    end
    return (1.0, 1.0)
end

# Fractional coordinate helper for box derivatives
@inline function _frac_displacement(box::Types.OrthorhombicBox, d::Types.SVector3)::NTuple{3,Float64}
    L = box.L
    return (d.x / L[1], d.y / L[2], d.z / L[3])
end

@inline function _frac_displacement(box::Types.TriclinicBox, d::Types.SVector3)::NTuple{3,Float64}
    M = box.Hinv
    return (M[1, 1]*d.x + M[1, 2]*d.y + M[1, 3]*d.z,
            M[2, 1]*d.x + M[2, 2]*d.y + M[2, 3]*d.z,
            M[3, 1]*d.x + M[3, 2]*d.y + M[3, 3]*d.z)
end

@inline _frac_displacement(::Types.NoBox, ::Types.SVector3) =
    throw(ArgumentError("no periodic box: box derivative ∂U/∂h is undefined for NoBox"))

# Reusable thread-safe scratch buffers for native C++ bridge dispatch
mutable struct TriumScratchBuffers
    forces::Matrix{Float64}
    virial9::Vector{Float64}
    H::Matrix{Float64}
    q_buf::Matrix{Float64}
    pair_i0::Vector{Int32}
    pair_j0::Vector{Int32}
    scale_lj::Vector{Float64}
    scale_coul::Vector{Float64}
end

function TriumScratchBuffers()
    TriumScratchBuffers(
        Matrix{Float64}(undef, 3, 0),
        zeros(Float64, 9),
        zeros(Float64, 3, 3),
        Matrix{Float64}(undef, 3, 0),
        Int32[],
        Int32[],
        Float64[],
        Float64[],
    )
end

const _SCRATCH_LOCK = ReentrantLock()
const _SCRATCH_POOL = TriumScratchBuffers[]

@inline function _acquire_scratch()::TriumScratchBuffers
    lock(_SCRATCH_LOCK)
    try
        return isempty(_SCRATCH_POOL) ? TriumScratchBuffers() : pop!(_SCRATCH_POOL)
    finally
        unlock(_SCRATCH_LOCK)
    end
end

@inline function _release_scratch(buf::TriumScratchBuffers)
    lock(_SCRATCH_LOCK)
    try
        push!(_SCRATCH_POOL, buf)
    finally
        unlock(_SCRATCH_LOCK)
    end
    return nothing
end

@inline function _ensure_forces_buf!(buf::TriumScratchBuffers, n::Int)::Matrix{Float64}
    if size(buf.forces, 2) < n
        buf.forces = Matrix{Float64}(undef, 3, n)
    end
    return buf.forces
end

@inline function _ensure_contiguous_q(buf::TriumScratchBuffers, q::AbstractMatrix{Float64}, n::Int)::Matrix{Float64}
    if q isa Matrix{Float64}
        return q
    end
    if size(buf.q_buf, 2) < n
        buf.q_buf = Matrix{Float64}(undef, 3, n)
    end
    @inbounds for j in 1:n
        buf.q_buf[1, j] = q[1, j]
        buf.q_buf[2, j] = q[2, j]
        buf.q_buf[3, j] = q[3, j]
    end
    return buf.q_buf
end

@inline function _ensure_pair_bufs!(buf::TriumScratchBuffers, np::Int, need_scales::Bool)
    if length(buf.pair_i0) < np
        resize!(buf.pair_i0, np)
        resize!(buf.pair_j0, np)
    end
    if need_scales && length(buf.scale_lj) < np
        resize!(buf.scale_lj, np)
        resize!(buf.scale_coul, np)
    end
    return nothing
end

@inline function _accumulate_forces!(F::AbstractMatrix{Float64}, f_buf::Matrix{Float64}, n::Int)
    @inbounds for i in 1:n
        F[1, i] += f_buf[1, i]
        F[2, i] += f_buf[2, i]
        F[3, i] += f_buf[3, i]
    end
    return nothing
end

@inline function _accumulate_virial!(box_deriv, v_buf::Vector{Float64})
    if box_deriv isa AbstractMatrix
        @inbounds for b in 1:3, a in 1:3
            box_deriv[a, b] += v_buf[a + 3 * (b - 1)]
        end
    elseif box_deriv isa AbstractVector
        @inbounds begin
            box_deriv[1] += v_buf[1]
            box_deriv[2] += v_buf[5]
            box_deriv[3] += v_buf[9]
        end
    end
    return nothing
end

@inline function _prepare_box_matrices!(scratch::TriumScratchBuffers, box::Types.AbstractBox)
    if box isa Types.OrthorhombicBox
        L = box.L
        H = scratch.H
        @inbounds begin
            H[1, 1] = L[1]; H[2, 1] = 0.0; H[3, 1] = 0.0
            H[1, 2] = 0.0; H[2, 2] = L[2]; H[3, 2] = 0.0
            H[1, 3] = 0.0; H[2, 3] = 0.0; H[3, 3] = L[3]
        end
        return (Cint(0), H, nothing)
    elseif box isa Types.TriclinicBox
        return (Cint(1), box.H, box.Hinv)
    else
        return (Cint(0), nothing, nothing)
    end
end

function _eval_tiles_native!(
    F::AbstractMatrix{Float64},
    box_deriv::Union{Nothing, AbstractMatrix{Float64}, AbstractVector{Float64}},
    q::AbstractMatrix{Float64},
    box::Types.AbstractBox,
    fac::TriumTiledNonbondedFactor,
    tc::Neighbors.TilePairCache,
    is_lfs::Bool
)::Float64
    n = length(fac.charges)
    n_tp = length(tc.tile_pairs_i)
    n_tp == 0 && return 0.0

    scratch = _acquire_scratch()
    try
        q_mat = _ensure_contiguous_q(scratch, q, n)
        f_buf = _ensure_forces_buf!(scratch, n)
        v_buf = scratch.virial9
        need_vir = (box_deriv !== nothing) && !(box isa Types.NoBox)
        vir_ptr = need_vir ? pointer(v_buf) : Ptr{Cdouble}(C_NULL)

        is_tri, H_mat, Hinv_mat = _prepare_box_matrices!(scratch, box)
        H_ptr = H_mat === nothing ? Ptr{Cdouble}(C_NULL) : pointer(H_mat)
        Hinv_ptr = Hinv_mat === nothing ? Ptr{Cdouble}(C_NULL) : pointer(Hinv_mat)

        topo = fac.topology
        excl_vec = topo === nothing ? UInt64[] : topo.excluded
        sk_vec = topo === nothing ? UInt64[] : topo.scaled_keys
        sw_vec = topo === nothing ? Float64[] : topo.scaled_weights
        n_excl = Csize_t(length(excl_vec))
        n_scaled = Csize_t(length(sk_vec))
        excl_ptr = n_excl == 0 ? Ptr{UInt64}(C_NULL) : pointer(excl_vec)
        sk_ptr = n_scaled == 0 ? Ptr{UInt64}(C_NULL) : pointer(sk_vec)
        sw_ptr = n_scaled == 0 ? Ptr{Cdouble}(C_NULL) : pointer(sw_vec)

        sig = fac.sigma
        eps = fac.epsilon
        chg = fac.charges
        tpi = tc.tile_pairs_i
        tpj = tc.tile_pairs_j
        ord = tc.order
        tstarts = tc.tile_starts
        tends = tc.tile_ends

        u_pair = GC.@preserve q_mat sig eps chg H_mat Hinv_mat tpi tpj ord tstarts tends excl_vec sk_vec sw_vec f_buf v_buf ccall(
            _SYM_EVAL_TILES_MT[], Cdouble,
            (Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
             Ptr{Cdouble}, Ptr{Cdouble}, Cint,
             Csize_t, Ptr{Int32}, Ptr{Int32},
             Ptr{Int32}, Ptr{Int32}, Ptr{Int32},
             Csize_t, Ptr{UInt64}, Csize_t, Ptr{UInt64}, Ptr{Cdouble},
             Cdouble, Cdouble,
             Cdouble, Cdouble, Cdouble,
             Cint, Cint, Cint,
             Ptr{Cdouble}, Ptr{Int64}, Ptr{Cdouble}),
            Csize_t(n), pointer(q_mat), pointer(sig), pointer(eps), pointer(chg),
            H_ptr, Hinv_ptr, is_tri,
            Csize_t(n_tp), pointer(tpi), pointer(tpj),
            pointer(ord), pointer(tstarts), pointer(tends),
            n_excl, excl_ptr, n_scaled, sk_ptr, sw_ptr,
            Cdouble(fac.lj14_scale), Cdouble(fac.coul14_scale),
            Cdouble(fac.rcut), Cdouble(fac.alpha), Cdouble(fac.coulomb_prefactor),
            Cint(is_lfs), Cint(fac.use_fast_math), Cint(0),
            pointer(f_buf), Ptr{Int64}(C_NULL), vir_ptr
        )

        _accumulate_forces!(F, f_buf, n)
        if need_vir
            _accumulate_virial!(box_deriv, v_buf)
        end
        return u_pair
    finally
        _release_scratch(scratch)
    end
end

# ==============================================================================
# 4B. Compiled CSR Verlet Plan Caching & Zero-Copy Direct Evaluator
# ==============================================================================

"""
    TriumCompiledVerletPlan

Owns a native C++ compiled CSR Verlet neighbor plan (`xbt_verlet_plan_*`) bound to a
`(TriumTiledNonbondedFactor, TilePairCache)` pair. Automatically frees its native C++
plan on garbage collection via `xbt_verlet_plan_destroy`.
"""
mutable struct TriumCompiledVerletPlan
    ptr::Ptr{Cvoid}
    last_rebuilds::Int
    n_atoms::Int
    rcut::Float64
    alpha::Float64
    coulomb_prefactor::Float64
    lj_mode::Symbol
    lj14_scale::Float64
    coul14_scale::Float64
    is_triclinic::Bool
    num_pairs::Int

    function TriumCompiledVerletPlan(
        ptr::Ptr{Cvoid},
        last_rebuilds::Int,
        n_atoms::Int,
        rcut::Float64,
        alpha::Float64,
        coulomb_prefactor::Float64,
        lj_mode::Symbol,
        lj14_scale::Float64,
        coul14_scale::Float64,
        is_triclinic::Bool,
        num_pairs::Int
    )
        plan = new(
            ptr, last_rebuilds, n_atoms, rcut, alpha, coulomb_prefactor,
            lj_mode, lj14_scale, coul14_scale, is_triclinic, num_pairs
        )
        finalizer(p -> begin
            if p.ptr != C_NULL && _SYM_VERLET_DESTROY[] != C_NULL
                ccall(_SYM_VERLET_DESTROY[], Cvoid, (Ptr{Cvoid},), p.ptr)
                p.ptr = C_NULL
            end
        end, plan)
        return plan
    end
end

const _VERLET_CACHE_LOCK = ReentrantLock()
const _VERLET_PLAN_CACHE = Dict{Tuple{UInt64, UInt64}, TriumCompiledVerletPlan}()
const _VERLET_PLAN_ORDER = Tuple{UInt64, UInt64}[]
const _MAX_VERLET_PLAN_CACHE = 64

"""
    clear_verlet_plan_cache!()

Evicts all cached `TriumCompiledVerletPlan` entries from the thread-safe plan cache.
"""
function clear_verlet_plan_cache!()
    lock(_VERLET_CACHE_LOCK)
    try
        empty!(_VERLET_PLAN_CACHE)
        empty!(_VERLET_PLAN_ORDER)
    finally
        unlock(_VERLET_CACHE_LOCK)
    end
    return nothing
end

@inline function _verlet_plan_valid(
    plan::TriumCompiledVerletPlan,
    tc::Neighbors.TilePairCache,
    fac::TriumTiledNonbondedFactor,
    n::Int,
    is_tri::Bool
)::Bool
    return plan.ptr != C_NULL &&
           plan.last_rebuilds == tc.n_rebuilds &&
           plan.n_atoms == n &&
           plan.rcut == fac.rcut &&
           plan.alpha == fac.alpha &&
           plan.coulomb_prefactor == fac.coulomb_prefactor &&
           plan.lj_mode === fac.lj_mode &&
           plan.lj14_scale == fac.lj14_scale &&
           plan.coul14_scale == fac.coul14_scale &&
           plan.is_triclinic == is_tri
end

function _get_or_create_verlet_plan!(
    q_mat::Matrix{Float64},
    H_ptr::Ptr{Cdouble},
    Hinv_ptr::Ptr{Cdouble},
    is_tri::Cint,
    fac::TriumTiledNonbondedFactor,
    tc::Neighbors.TilePairCache
)::Union{Nothing, TriumCompiledVerletPlan}
    n = length(fac.charges)
    n_tp = length(tc.tile_pairs_i)
    (n == 0 || n_tp == 0 || _SYM_VERLET_CREATE[] == C_NULL) && return nothing

    is_tri_bool = (is_tri != Cint(0))
    key = (objectid(fac), objectid(tc))

    lock(_VERLET_CACHE_LOCK)
    try
        existing = get(_VERLET_PLAN_CACHE, key, nothing)
        if existing !== nothing && _verlet_plan_valid(existing, tc, fac, n, is_tri_bool)
            return existing
        end

        topo = fac.topology
        excl_vec = topo === nothing ? UInt64[] : topo.excluded
        sk_vec = topo === nothing ? UInt64[] : topo.scaled_keys
        sw_vec = topo === nothing ? Float64[] : topo.scaled_weights
        n_excl = Csize_t(length(excl_vec))
        n_scaled = Csize_t(length(sk_vec))
        excl_ptr = n_excl == 0 ? Ptr{UInt64}(C_NULL) : pointer(excl_vec)
        sk_ptr = n_scaled == 0 ? Ptr{UInt64}(C_NULL) : pointer(sk_vec)
        sw_ptr = n_scaled == 0 ? Ptr{Cdouble}(C_NULL) : pointer(sw_vec)

        sig = fac.sigma
        eps = fac.epsilon
        chg = fac.charges
        tpi = tc.tile_pairs_i
        tpj = tc.tile_pairs_j
        ord = tc.order
        tstarts = tc.tile_starts
        tends = tc.tile_ends
        is_lfs = (fac.lj_mode === :lfs)

        plan_ptr = GC.@preserve q_mat sig eps chg tpi tpj ord tstarts tends excl_vec sk_vec sw_vec ccall(
            _SYM_VERLET_CREATE[], Ptr{Cvoid},
            (Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
             Ptr{Cdouble}, Ptr{Cdouble}, Cint,
             Csize_t, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32},
             Csize_t, Ptr{UInt64}, Csize_t, Ptr{UInt64}, Ptr{Cdouble},
             Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Cint),
            Csize_t(n), pointer(q_mat), pointer(sig), pointer(eps), pointer(chg),
            H_ptr, Hinv_ptr, is_tri,
            Csize_t(n_tp), pointer(tpi), pointer(tpj), pointer(ord), pointer(tstarts), pointer(tends),
            n_excl, excl_ptr, n_scaled, sk_ptr, sw_ptr,
            Cdouble(fac.lj14_scale), Cdouble(fac.coul14_scale),
            Cdouble(fac.rcut), Cdouble(tc.skin), Cdouble(fac.alpha), Cdouble(fac.coulomb_prefactor),
            Cint(is_lfs)
        )
        plan_ptr == C_NULL && return nothing

        npairs = _SYM_VERLET_NUM_PAIRS[] != C_NULL ?
            Int(ccall(_SYM_VERLET_NUM_PAIRS[], Csize_t, (Ptr{Cvoid},), plan_ptr)) : 0

        plan = TriumCompiledVerletPlan(
            plan_ptr, tc.n_rebuilds, n,
            fac.rcut, fac.alpha, fac.coulomb_prefactor,
            fac.lj_mode, fac.lj14_scale, fac.coul14_scale,
            is_tri_bool, npairs
        )

        if !haskey(_VERLET_PLAN_CACHE, key)
            while length(_VERLET_PLAN_CACHE) >= _MAX_VERLET_PLAN_CACHE && !isempty(_VERLET_PLAN_ORDER)
                evict_key = popfirst!(_VERLET_PLAN_ORDER)
                delete!(_VERLET_PLAN_CACHE, evict_key)
            end
            push!(_VERLET_PLAN_ORDER, key)
        end
        _VERLET_PLAN_CACHE[key] = plan
        return plan
    finally
        unlock(_VERLET_CACHE_LOCK)
    end
end

function _eval_verlet_plan_native!(
    F::AbstractMatrix{Float64},
    box_deriv::Union{Nothing, AbstractMatrix{Float64}, AbstractVector{Float64}},
    q::AbstractMatrix{Float64},
    box::Types.AbstractBox,
    fac::TriumTiledNonbondedFactor,
    tc::Neighbors.TilePairCache,
    is_lfs::Bool
)::Float64
    n = length(fac.charges)
    n_tp = length(tc.tile_pairs_i)
    (n == 0 || n_tp == 0) && return 0.0

    scratch = _acquire_scratch()
    try
        q_mat = _ensure_contiguous_q(scratch, q, n)
        f_buf = _ensure_forces_buf!(scratch, n)
        v_buf = scratch.virial9
        need_vir = (box_deriv !== nothing) && !(box isa Types.NoBox)
        vir_ptr = need_vir ? pointer(v_buf) : Ptr{Cdouble}(C_NULL)

        is_tri, H_mat, Hinv_mat = _prepare_box_matrices!(scratch, box)
        H_ptr = H_mat === nothing ? Ptr{Cdouble}(C_NULL) : pointer(H_mat)
        Hinv_ptr = Hinv_mat === nothing ? Ptr{Cdouble}(C_NULL) : pointer(Hinv_mat)

        plan = GC.@preserve H_mat Hinv_mat _get_or_create_verlet_plan!(
            q_mat, H_ptr, Hinv_ptr, is_tri, fac, tc
        )
        if plan === nothing
            return _eval_tiles_native!(F, box_deriv, q, box, fac, tc, is_lfs)
        end

        u_pair = GC.@preserve plan q_mat H_mat Hinv_mat f_buf v_buf ccall(
            _SYM_VERLET_EVAL_MT[], Cdouble,
            (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
             Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Int64}, Ptr{Cdouble}),
            plan.ptr, pointer(q_mat), H_ptr, Hinv_ptr,
            is_tri, Cint(fac.use_fast_math), Cint(0),
            pointer(f_buf), Ptr{Int64}(C_NULL), vir_ptr
        )

        _accumulate_forces!(F, f_buf, n)
        if need_vir
            _accumulate_virial!(box_deriv, v_buf)
        end
        return u_pair
    finally
        _release_scratch(scratch)
    end
end

"""
    compiled_verlet_num_pairs(fac::TriumTiledNonbondedFactor, q, box) -> Int

Ensures a compiled CSR Verlet plan exists for `(fac, fac.cache)` at `(q, box)` and returns
the pair count reported by `xbt_verlet_plan_num_pairs`. Returns `0` when the native
Verlet plan ABI is unavailable or `fac.cache` is not a `TilePairCache`.
"""
function compiled_verlet_num_pairs(
    fac::TriumTiledNonbondedFactor,
    q::AbstractMatrix{Float64},
    box::Types.AbstractBox
)::Int
    n = length(fac.charges)
    size(q, 1) == 3 && size(q, 2) == n ||
        throw(ArgumentError("q must have shape 3 x $n"))

    tc = fac.cache
    (tc isa Neighbors.TilePairCache && has_native_bridge() &&
     _SYM_VERLET_CREATE[] != C_NULL && _SYM_VERLET_NUM_PAIRS[] != C_NULL) || return 0

    Neighbors.refresh_tile_cache!(tc, q, box)
    (n == 0 || isempty(tc.tile_pairs_i)) && return 0

    scratch = _acquire_scratch()
    try
        q_mat = _ensure_contiguous_q(scratch, q, n)
        is_tri, H_mat, Hinv_mat = _prepare_box_matrices!(scratch, box)
        H_ptr = H_mat === nothing ? Ptr{Cdouble}(C_NULL) : pointer(H_mat)
        Hinv_ptr = Hinv_mat === nothing ? Ptr{Cdouble}(C_NULL) : pointer(Hinv_mat)
        plan = GC.@preserve H_mat Hinv_mat _get_or_create_verlet_plan!(
            q_mat, H_ptr, Hinv_ptr, is_tri, fac, tc
        )
        plan === nothing && return 0
        return Int(GC.@preserve plan ccall(_SYM_VERLET_NUM_PAIRS[], Csize_t, (Ptr{Cvoid},), plan.ptr))
    finally
        _release_scratch(scratch)
    end
end

function _eval_verlet_direct_core!(
    F::Matrix{Float64},
    box_deriv::Union{Nothing, AbstractMatrix{Float64}, AbstractVector{Float64}},
    q::Matrix{Float64},
    box::Types.AbstractBox,
    H_ptr::Ptr{Cdouble},
    Hinv_ptr::Ptr{Cdouble},
    is_tri::Cint,
    fac::TriumTiledNonbondedFactor,
    tc::Neighbors.TilePairCache,
    u_self::Float64
)::Float64
    plan = _get_or_create_verlet_plan!(q, H_ptr, Hinv_ptr, is_tri, fac, tc)
    if plan === nothing
        fill!(F, 0.0)
        box_deriv !== nothing && fill!(box_deriv, 0.0)
        return eval_lj_dsf_pairs!(F, box_deriv, q, box, fac)
    end

    if box_deriv === nothing
        u_pair = GC.@preserve plan q F ccall(
            _SYM_VERLET_EVAL_MT[], Cdouble,
            (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
             Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Int64}, Ptr{Cdouble}),
            plan.ptr, pointer(q), H_ptr, Hinv_ptr,
            is_tri, Cint(fac.use_fast_math), Cint(0),
            pointer(F), Ptr{Int64}(C_NULL), Ptr{Cdouble}(C_NULL)
        )
        return u_self + u_pair
    elseif box_deriv isa Matrix{Float64}
        u_pair = GC.@preserve plan q F box_deriv ccall(
            _SYM_VERLET_EVAL_MT[], Cdouble,
            (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
             Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Int64}, Ptr{Cdouble}),
            plan.ptr, pointer(q), H_ptr, Hinv_ptr,
            is_tri, Cint(fac.use_fast_math), Cint(0),
            pointer(F), Ptr{Int64}(C_NULL), pointer(box_deriv)
        )
        return u_self + u_pair
    else
        scratch = _acquire_scratch()
        try
            v_buf = scratch.virial9
            u_pair = GC.@preserve plan q F v_buf ccall(
                _SYM_VERLET_EVAL_MT[], Cdouble,
                (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
                 Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Int64}, Ptr{Cdouble}),
                plan.ptr, pointer(q), H_ptr, Hinv_ptr,
                is_tri, Cint(fac.use_fast_math), Cint(0),
                pointer(F), Ptr{Int64}(C_NULL), pointer(v_buf)
            )
            fill!(box_deriv, 0.0)
            _accumulate_virial!(box_deriv, v_buf)
            return u_self + u_pair
        finally
            _release_scratch(scratch)
        end
    end
end

"""
    eval_lj_dsf_direct_into!(F::Matrix{Float64}, box_deriv, q::Matrix{Float64},
                             box::Types.AbstractBox, fac::TriumTiledNonbondedFactor) -> Float64

Zero-copy single-pass nonbonded evaluation that writes Cartesian forces directly into `F`
(overwriting `F`, and overwriting `box_deriv` when provided) using the compiled CSR
Verlet plan (`xbt_verlet_plan_eval_mt`). When `box_deriv === nothing`, passes `C_NULL`
for the virial pointer with zero scratch allocation. Falls back to `eval_lj_dsf_pairs!`
when the compiled Verlet plan path is unavailable.
"""
function eval_lj_dsf_direct_into!(
    F::Matrix{Float64},
    box_deriv::Union{Nothing, AbstractMatrix{Float64}, AbstractVector{Float64}},
    q::Matrix{Float64},
    box::Types.AbstractBox,
    fac::TriumTiledNonbondedFactor
)::Float64
    n = length(fac.charges)
    size(q, 1) == 3 && size(q, 2) == n ||
        throw(ArgumentError("q must have shape 3 x $n"))
    size(F, 1) == 3 && size(F, 2) == n ||
        throw(ArgumentError("F must have shape 3 x $n"))

    if box isa Types.OrthorhombicBox
        2.0 * fac.rcut < minimum(box.L) ||
            throw(ArgumentError("2 * rcut must be less than minimum box length"))
    elseif box isa Types.TriclinicBox
        2.0 * fac.rcut < minimum(box.widths) ||
            throw(ArgumentError("2 * rcut must be less than minimum box width"))
    elseif box isa Types.NoBox
        box_deriv !== nothing &&
            throw(ArgumentError("no periodic box: box derivative ∂U/∂h is undefined for NoBox"))
    end

    if box_deriv isa AbstractMatrix
        size(box_deriv) == (3, 3) || throw(DimensionMismatch("box_deriv must be 3x3"))
    elseif box_deriv isa AbstractVector
        length(box_deriv) == 3 || throw(DimensionMismatch("box_deriv vector must have length 3"))
    end

    tc = fac.cache
    use_native = fac.use_native_bridge && !fac.use_carry_free && has_native_bridge() &&
                 (fac.alpha > 0.0 || fac.coulomb_prefactor == 0.0 || all(iszero, fac.charges))

    if tc isa Neighbors.TilePairCache && use_native &&
       _SYM_VERLET_CREATE[] != C_NULL && _SYM_VERLET_EVAL_MT[] != C_NULL
        u_self = 0.0
        if fac.include_self
            rc = fac.rcut
            alpha = fac.alpha
            erfc_rc = fac.use_fast_math ? xbt_erfc_fast(alpha * rc) : ccall(:erfc, Cdouble, (Cdouble,), alpha * rc)
            u_self = -fac.coulomb_prefactor * (erfc_rc / (2.0 * rc) + alpha / sqrt(pi)) * sum(abs2, fac.charges)
        end

        Neighbors.refresh_tile_cache!(tc, q, box)
        if n == 0 || isempty(tc.tile_pairs_i)
            fill!(F, 0.0)
            box_deriv !== nothing && fill!(box_deriv, 0.0)
            return u_self
        end

        if box isa Types.OrthorhombicBox
            L = box.L
            H_ref = Ref((L[1], 0.0, 0.0, 0.0, L[2], 0.0, 0.0, 0.0, L[3]))
            GC.@preserve H_ref begin
                H_ptr = Base.unsafe_convert(Ptr{Cdouble}, H_ref)
                return _eval_verlet_direct_core!(F, box_deriv, q, box, H_ptr, Ptr{Cdouble}(C_NULL), Cint(0), fac, tc, u_self)
            end
        elseif box isa Types.TriclinicBox
            H_mat = box.H
            Hinv_mat = box.Hinv
            GC.@preserve H_mat Hinv_mat begin
                return _eval_verlet_direct_core!(F, box_deriv, q, box, pointer(H_mat), pointer(Hinv_mat), Cint(1), fac, tc, u_self)
            end
        else
            return _eval_verlet_direct_core!(F, box_deriv, q, box, Ptr{Cdouble}(C_NULL), Ptr{Cdouble}(C_NULL), Cint(0), fac, tc, u_self)
        end
    end

    fill!(F, 0.0)
    box_deriv !== nothing && fill!(box_deriv, 0.0)
    return eval_lj_dsf_pairs!(F, box_deriv, q, box, fac)
end

# ==============================================================================
# 4C. Multi-Rate r-RESPA & HMR Throughput Helpers
# ==============================================================================

"""
    TriumMultiRateConfig(; dt_fs=2.0, hmr_factor=1.0, respa_outer_stride=1)

Configuration describing effective physical time advancement per outer nonbonded evaluation
under Hydrogen Mass Repartitioning (`hmr_factor`) and r-RESPA outer striding (`respa_outer_stride`).
"""
struct TriumMultiRateConfig
    dt_fs::Float64
    hmr_factor::Float64
    respa_outer_stride::Int

    function TriumMultiRateConfig(dt_fs::Real, hmr_factor::Real, respa_outer_stride::Integer)
        dt = Float64(dt_fs)
        hmr = Float64(hmr_factor)
        stride = Int(respa_outer_stride)
        dt > 0.0 && isfinite(dt) || throw(ArgumentError("dt_fs must be positive and finite"))
        hmr > 0.0 && isfinite(hmr) || throw(ArgumentError("hmr_factor must be positive and finite"))
        stride >= 1 || throw(ArgumentError("respa_outer_stride must be >= 1"))
        new(dt, hmr, stride)
    end
end

TriumMultiRateConfig(; dt_fs::Real = 2.0, hmr_factor::Real = 1.0, respa_outer_stride::Integer = 1) =
    TriumMultiRateConfig(dt_fs, hmr_factor, respa_outer_stride)

"""
    effective_ns_per_day(eval_ms::Real; dt_fs=2.0, hmr_factor=1.0, respa_outer_stride=1) -> Float64

Computes effective simulation throughput in ns/day from per-evaluation wall time `eval_ms` (ms):
`(86.4 * dt_fs * hmr_factor * respa_outer_stride) / eval_ms`.
"""
function effective_ns_per_day(
    eval_ms::Real;
    dt_fs::Real = 2.0,
    hmr_factor::Real = 1.0,
    respa_outer_stride::Integer = 1
)::Float64
    ms = Float64(eval_ms)
    dt = Float64(dt_fs)
    hmr = Float64(hmr_factor)
    stride = Int(respa_outer_stride)
    ms > 0.0 && isfinite(ms) || throw(ArgumentError("eval_ms must be positive and finite"))
    dt > 0.0 && isfinite(dt) || throw(ArgumentError("dt_fs must be positive and finite"))
    hmr > 0.0 && isfinite(hmr) || throw(ArgumentError("hmr_factor must be positive and finite"))
    stride >= 1 || throw(ArgumentError("respa_outer_stride must be >= 1"))
    return (86.4 * dt * hmr * stride) / ms
end

effective_ns_per_day(eval_ms::Real, cfg::TriumMultiRateConfig)::Float64 =
    effective_ns_per_day(eval_ms; dt_fs = cfg.dt_fs, hmr_factor = cfg.hmr_factor,
                         respa_outer_stride = cfg.respa_outer_stride)

function _eval_pairs_native!(
    F::AbstractMatrix{Float64},
    box_deriv::Union{Nothing, AbstractMatrix{Float64}, AbstractVector{Float64}},
    q::AbstractMatrix{Float64},
    box::Types.AbstractBox,
    fac::TriumTiledNonbondedFactor,
    tc::Neighbors.PairCache,
    is_lfs::Bool
)::Float64
    n = length(fac.charges)
    np = length(tc.pairs_i)
    np == 0 && return 0.0

    scratch = _acquire_scratch()
    try
        q_mat = _ensure_contiguous_q(scratch, q, n)
        f_buf = _ensure_forces_buf!(scratch, n)
        has_topo = fac.topology !== nothing
        _ensure_pair_bufs!(scratch, np, has_topo)

        pairs_i = tc.pairs_i
        pairs_j = tc.pairs_j
        pi0 = scratch.pair_i0
        pj0 = scratch.pair_j0
        slj = scratch.scale_lj
        scoul = scratch.scale_coul

        if has_topo
            @inbounds for k in 1:np
                ii = Int(pairs_i[k])
                jj = Int(pairs_j[k])
                pi0[k] = Int32(ii - 1)
                pj0[k] = Int32(jj - 1)
                w_lj, w_c = _pair_scales(fac, ii, jj)
                slj[k] = w_lj
                scoul[k] = w_c
            end
        else
            @inbounds for k in 1:np
                pi0[k] = pairs_i[k] - Int32(1)
                pj0[k] = pairs_j[k] - Int32(1)
            end
        end

        v_buf = scratch.virial9
        need_vir = (box_deriv !== nothing) && !(box isa Types.NoBox)
        vir_ptr = need_vir ? pointer(v_buf) : Ptr{Cdouble}(C_NULL)

        is_tri, H_mat, Hinv_mat = _prepare_box_matrices!(scratch, box)
        H_ptr = H_mat === nothing ? Ptr{Cdouble}(C_NULL) : pointer(H_mat)
        Hinv_ptr = Hinv_mat === nothing ? Ptr{Cdouble}(C_NULL) : pointer(Hinv_mat)

        slj_ptr = has_topo ? pointer(slj) : Ptr{Cdouble}(C_NULL)
        scoul_ptr = has_topo ? pointer(scoul) : Ptr{Cdouble}(C_NULL)

        sig = fac.sigma
        eps = fac.epsilon
        chg = fac.charges

        u_pair = GC.@preserve q_mat sig eps chg H_mat Hinv_mat pi0 pj0 slj scoul f_buf v_buf ccall(
            _SYM_EVAL_PAIRS_MT[], Cdouble,
            (Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
             Ptr{Cdouble}, Ptr{Cdouble}, Cint,
             Csize_t, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble},
             Cdouble, Cdouble, Cdouble,
             Cint, Cint,
             Ptr{Cdouble}, Ptr{Int64}, Ptr{Cdouble}, Cint),
            Csize_t(n), pointer(q_mat), pointer(sig), pointer(eps), pointer(chg),
            H_ptr, Hinv_ptr, is_tri,
            Csize_t(np), pointer(pi0), pointer(pj0), slj_ptr, scoul_ptr,
            Cdouble(fac.rcut), Cdouble(fac.alpha), Cdouble(fac.coulomb_prefactor),
            Cint(is_lfs), Cint(fac.use_fast_math),
            pointer(f_buf), Ptr{Int64}(C_NULL), vir_ptr, Cint(0)
        )

        _accumulate_forces!(F, f_buf, n)
        if need_vir
            _accumulate_virial!(box_deriv, v_buf)
        end
        return u_pair
    finally
        _release_scratch(scratch)
    end
end

@inline function _eval_pair_terms(
    fac::TriumTiledNonbondedFactor,
    box::Types.AbstractBox,
    q::AbstractMatrix{Float64},
    i::Int, j::Int,
    rc::Float64, rc2::Float64,
    alpha::Float64, pref::Float64,
    two_alpha_sqrt_pi::Float64,
    Arc::Float64, Brc::Float64,
    is_lfs::Bool
)
    lj_scale, coul_scale = _pair_scales(fac, i, j)
    (lj_scale == 0.0 && coul_scale == 0.0) && return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false, false)

    d = Types.unwrap_displacement(box, q, i, j)
    r2 = Types.norm2(d)
    r2 >= rc2 && return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false, false)
    r2 < 1e-24 && return (Inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false, true)

    r = sqrt(r2)

    u_lj = 0.0
    coeff_lj = 0.0
    if lj_scale > 0.0
        sig = 0.5 * (fac.sigma[i] + fac.sigma[j])
        eps = sqrt(fac.epsilon[i] * fac.epsilon[j])
        if eps > 0.0
            sr = sig / r
            sr2 = sr * sr
            sr6 = sr2 * sr2 * sr2
            sr12 = sr6 * sr6
            u_raw = 4.0 * eps * (sr12 - sr6)
            dudr_raw = 24.0 * eps * (sr6 - 2.0 * sr12) / r

            if is_lfs
                src = sig / rc
                src2 = src * src
                src6 = src2 * src2 * src2
                src12 = src6 * src6
                u_rc = 4.0 * eps * (src12 - src6)
                duc = 24.0 * eps * (src6 - 2.0 * src12) / rc
                u_lj = (u_raw - u_rc - duc * (r - rc)) * lj_scale
                dudr_lj = (dudr_raw - duc) * lj_scale
            else
                u_lj = u_raw * lj_scale
                dudr_lj = dudr_raw * lj_scale
            end
            coeff_lj = -dudr_lj / r
        end
    end

    u_coul = 0.0
    coeff_coul = 0.0
    if coul_scale > 0.0
        qiqj = fac.charges[i] * fac.charges[j]
        if qiqj != 0.0
            z = alpha * r
            er = fac.use_fast_math ? xbt_erfc_fast(z) : ccall(:erfc, Cdouble, (Cdouble,), z)
            exp_z = fac.use_fast_math ? xbt_exp_radix3(-z * z) : exp(-z * z)

            u_coul = coul_scale * pref * qiqj * (er / r - Brc + Arc * r)
            coeff_coul = coul_scale * pref * qiqj * (er / r2 + two_alpha_sqrt_pi * exp_z / r - Arc) / r
        end
    end

    coeff = coeff_lj + coeff_coul
    gx = coeff * d.x
    gy = coeff * d.y
    gz = coeff * d.z
    return (u_lj + u_coul, gx, gy, gz, d.x, d.y, d.z, coeff, true, false)
end

function _eval_tile_pair_range_julia!(
    F::AbstractMatrix{Float64},
    force_buf::Union{Nothing, CarryFreeForceBuffer},
    q::AbstractMatrix{Float64},
    box::Types.AbstractBox,
    fac::TriumTiledNonbondedFactor,
    tpi::Vector{Int32}, tpj::Vector{Int32},
    ord::Vector{Int32}, tstarts::Vector{Int32}, tends::Vector{Int32},
    indices,
    need_vir::Bool,
    rc::Float64, rc2::Float64,
    alpha::Float64, pref::Float64,
    two_alpha_sqrt_pi::Float64,
    Arc::Float64, Brc::Float64,
    is_lfs::Bool
)
    u_loc = 0.0
    v11 = 0.0; v21 = 0.0; v31 = 0.0
    v12 = 0.0; v22 = 0.0; v32 = 0.0
    v13 = 0.0; v23 = 0.0; v33 = 0.0
    use_cf = force_buf !== nothing

    @inbounds for k in indices
        ti = Int(tpi[k]); tj = Int(tpj[k])
        si, ei = Int(tstarts[ti]), Int(tends[ti])
        if ti == tj
            for a in si:(ei - 1)
                i = Int(ord[a])
                fxi = 0.0; fyi = 0.0; fzi = 0.0
                for b in (a + 1):ei
                    j = Int(ord[b])
                    u_p, gx, gy, gz, dx, dy, dz, coeff, active, overlap = _eval_pair_terms(
                        fac, box, q, i, j, rc, rc2, alpha, pref, two_alpha_sqrt_pi, Arc, Brc, is_lfs
                    )
                    overlap && return (Inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    active || continue
                    u_loc += u_p
                    if use_cf
                        scatter_pair_force!(force_buf, i, j, gx, gy, gz)
                    else
                        fxi += gx; fyi += gy; fzi += gz
                        F[1, j] -= gx; F[2, j] -= gy; F[3, j] -= gz
                    end
                    if need_vir
                        w = _frac_displacement(box, Types.SVector3(dx, dy, dz))
                        vx = -coeff * dx; vy = -coeff * dy; vz = -coeff * dz
                        v11 += vx * w[1]; v21 += vy * w[1]; v31 += vz * w[1]
                        v12 += vx * w[2]; v22 += vy * w[2]; v32 += vz * w[2]
                        v13 += vx * w[3]; v23 += vy * w[3]; v33 += vz * w[3]
                    end
                end
                if !use_cf
                    F[1, i] += fxi; F[2, i] += fyi; F[3, i] += fzi
                end
            end
        else
            sj, ej = Int(tstarts[tj]), Int(tends[tj])
            for a in si:ei
                i = Int(ord[a])
                fxi = 0.0; fyi = 0.0; fzi = 0.0
                for b in sj:ej
                    j = Int(ord[b])
                    u_p, gx, gy, gz, dx, dy, dz, coeff, active, overlap = _eval_pair_terms(
                        fac, box, q, i, j, rc, rc2, alpha, pref, two_alpha_sqrt_pi, Arc, Brc, is_lfs
                    )
                    overlap && return (Inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    active || continue
                    u_loc += u_p
                    if use_cf
                        scatter_pair_force!(force_buf, i, j, gx, gy, gz)
                    else
                        fxi += gx; fyi += gy; fzi += gz
                        F[1, j] -= gx; F[2, j] -= gy; F[3, j] -= gz
                    end
                    if need_vir
                        w = _frac_displacement(box, Types.SVector3(dx, dy, dz))
                        vx = -coeff * dx; vy = -coeff * dy; vz = -coeff * dz
                        v11 += vx * w[1]; v21 += vy * w[1]; v31 += vz * w[1]
                        v12 += vx * w[2]; v22 += vy * w[2]; v32 += vz * w[2]
                        v13 += vx * w[3]; v23 += vy * w[3]; v33 += vz * w[3]
                    end
                end
                if !use_cf
                    F[1, i] += fxi; F[2, i] += fyi; F[3, i] += fzi
                end
            end
        end
    end
    return (u_loc, v11, v21, v31, v12, v22, v32, v13, v23, v33)
end

@inline function _accum_virial_tuple!(box_deriv, vt)
    box_deriv === nothing && return nothing
    if box_deriv isa AbstractMatrix
        @inbounds begin
            box_deriv[1, 1] += vt[2]; box_deriv[2, 1] += vt[3]; box_deriv[3, 1] += vt[4]
            box_deriv[1, 2] += vt[5]; box_deriv[2, 2] += vt[6]; box_deriv[3, 2] += vt[7]
            box_deriv[1, 3] += vt[8]; box_deriv[2, 3] += vt[9]; box_deriv[3, 3] += vt[10]
        end
    elseif box_deriv isa AbstractVector
        @inbounds begin
            box_deriv[1] += vt[2]
            box_deriv[2] += vt[6]
            box_deriv[3] += vt[10]
        end
    end
    return nothing
end

"""
    eval_lj_dsf_pairs!(F, box_deriv, q, box, fac::TriumTiledNonbondedFactor) -> Float64

Core pair evaluation loop computing energy, scattering forces, and accumulating box derivatives.
"""
function eval_lj_dsf_pairs!(F::AbstractMatrix{Float64},
                            box_deriv::Union{Nothing, AbstractMatrix{Float64}, AbstractVector{Float64}},
                            q::AbstractMatrix{Float64},
                            box::Types.AbstractBox,
                            fac::TriumTiledNonbondedFactor)::Float64
    n = length(fac.charges)
    size(q, 1) == 3 && size(q, 2) == n ||
        throw(ArgumentError("q must have shape 3 x $n"))
    size(F, 1) == 3 && size(F, 2) == n ||
        throw(ArgumentError("F must have shape 3 x $n"))

    # Validate box size vs cutoff
    if box isa Types.OrthorhombicBox
        2.0 * fac.rcut < minimum(box.L) ||
            throw(ArgumentError("2 * rcut must be less than minimum box length"))
    elseif box isa Types.TriclinicBox
        min_w = minimum(box.widths)
        2.0 * fac.rcut < min_w ||
            throw(ArgumentError("2 * rcut must be less than minimum box width"))
    elseif box isa Types.NoBox
        box_deriv !== nothing &&
            throw(ArgumentError("no periodic box: box derivative ∂U/∂h is undefined for NoBox"))
    end

    rc = fac.rcut
    rc2 = rc * rc
    alpha = fac.alpha
    pref = fac.coulomb_prefactor
    two_alpha_sqrt_pi = 2.0 * alpha / sqrt(pi)

    erfc_rc = fac.use_fast_math ? xbt_erfc_fast(alpha * rc) : ccall(:erfc, Cdouble, (Cdouble,), alpha * rc)
    exp_rc = fac.use_fast_math ? xbt_exp_radix3(-alpha * alpha * rc2) : exp(-alpha * alpha * rc2)
    Arc = erfc_rc / rc2 + two_alpha_sqrt_pi * exp_rc / rc
    Brc = erfc_rc / rc + Arc * rc

    is_lfs = fac.lj_mode === :lfs

    u_total = 0.0
    if fac.include_self
        u_total += -pref * (erfc_rc / (2.0 * rc) + alpha / sqrt(pi)) * sum(abs2, fac.charges)
    end

    tc = fac.cache
    use_native = fac.use_native_bridge && !fac.use_carry_free && has_native_bridge() &&
                 (alpha > 0.0 || pref == 0.0 || all(iszero, fac.charges))

    # Path A: Native C++ Multi-Threaded Compiled CSR Verlet Plan / Tile-Cache Kernel
    if tc isa Neighbors.TilePairCache
        Neighbors.refresh_tile_cache!(tc, q, box)
        if use_native
            if _SYM_VERLET_CREATE[] != C_NULL && _SYM_VERLET_EVAL_MT[] != C_NULL
                return u_total + _eval_verlet_plan_native!(F, box_deriv, q, box, fac, tc, is_lfs)
            elseif _SYM_EVAL_TILES_MT[] != C_NULL
                return u_total + _eval_tiles_native!(F, box_deriv, q, box, fac, tc, is_lfs)
            end
        end

        # Path C: Pure-Julia TilePairCache (Orbit-MT when Threads.nthreads() > 1)
        force_buf = fac.use_carry_free ? CarryFreeForceBuffer(n) : nothing
        tpi = tc.tile_pairs_i
        tpj = tc.tile_pairs_j
        ord = tc.order
        tstarts = tc.tile_starts
        tends = tc.tile_ends
        need_vir = (box_deriv !== nothing)

        if !fac.use_carry_free && Threads.nthreads() > 1 && length(tpi) >= 8
            sched = schedule_disjoint_tile_orbits(tc.n_tiles, tpi, tpj)
            nt_avail = Threads.nthreads()
            @inbounds for o in 1:sched.n_orbits
                s = Int(sched.orbit_offsets[o])
                e = Int(sched.orbit_offsets[o + 1]) - 1
                cnt = e - s + 1
                if cnt >= 4
                    nt = min(nt_avail, cnt)
                    tasks = Vector{Task}(undef, nt)
                    for c in 1:nt
                        c_start = s + fld((c - 1) * cnt, nt)
                        c_end = s + fld(c * cnt, nt) - 1
                        perm_slice = view(sched.orbit_perm, c_start:c_end)
                        tasks[c] = Threads.@spawn _eval_tile_pair_range_julia!(
                            F, nothing, q, box, fac, tpi, tpj, ord, tstarts, tends,
                            perm_slice, need_vir, rc, rc2, alpha, pref, two_alpha_sqrt_pi, Arc, Brc, is_lfs
                        )
                    end
                    for c in 1:nt
                        res = fetch(tasks[c])
                        res[1] == Inf && return Inf
                        u_total += res[1]
                        need_vir && _accum_virial_tuple!(box_deriv, res)
                    end
                else
                    res = _eval_tile_pair_range_julia!(
                        F, nothing, q, box, fac, tpi, tpj, ord, tstarts, tends,
                        view(sched.orbit_perm, s:e), need_vir, rc, rc2, alpha, pref, two_alpha_sqrt_pi, Arc, Brc, is_lfs
                    )
                    res[1] == Inf && return Inf
                    u_total += res[1]
                    need_vir && _accum_virial_tuple!(box_deriv, res)
                end
            end
        else
            res = _eval_tile_pair_range_julia!(
                F, force_buf, q, box, fac, tpi, tpj, ord, tstarts, tends,
                eachindex(tpi), need_vir, rc, rc2, alpha, pref, two_alpha_sqrt_pi, Arc, Brc, is_lfs
            )
            res[1] == Inf && return Inf
            u_total += res[1]
            need_vir && _accum_virial_tuple!(box_deriv, res)
            if fac.use_carry_free
                resolve_forces!(F, force_buf; accumulate = true)
            end
        end
        return u_total
    elseif tc isa Neighbors.PairCache
        Neighbors.refresh_pair_cache!(tc, q, box)
        # Path B: Native C++ Multi-Threaded Pair-Cache Kernel
        if use_native && _SYM_EVAL_PAIRS_MT[] != C_NULL
            return u_total + _eval_pairs_native!(F, box_deriv, q, box, fac, tc, is_lfs)
        end
    end

    # Pure-Julia fallback for PairCache and Direct All-Pairs
    force_buf = fac.use_carry_free ? CarryFreeForceBuffer(n) : nothing

    @inline function process_pair(i::Int, j::Int)
        u_p, gx, gy, gz, dx, dy, dz, coeff, active, overlap = _eval_pair_terms(
            fac, box, q, i, j, rc, rc2, alpha, pref, two_alpha_sqrt_pi, Arc, Brc, is_lfs
        )
        overlap && return Inf
        active || return 0.0

        if fac.use_carry_free
            scatter_pair_force!(force_buf, i, j, gx, gy, gz)
        else
            F[1, i] += gx; F[2, i] += gy; F[3, i] += gz
            F[1, j] -= gx; F[2, j] -= gy; F[3, j] -= gz
        end

        if box_deriv !== nothing
            w = _frac_displacement(box, Types.SVector3(dx, dy, dz))
            if box_deriv isa AbstractMatrix
                dv = (dx, dy, dz)
                for b in 1:3, a in 1:3
                    box_deriv[a, b] += (-coeff * dv[a]) * w[b]
                end
            elseif box_deriv isa AbstractVector
                box_deriv[1] += (-coeff * dx) * w[1]
                box_deriv[2] += (-coeff * dy) * w[2]
                box_deriv[3] += (-coeff * dz) * w[3]
            end
        end

        return u_p
    end

    if tc isa Neighbors.PairCache
        pairs_i = tc.pairs_i
        pairs_j = tc.pairs_j
        @inbounds for k in eachindex(pairs_i)
            i = Int(pairs_i[k])
            j = Int(pairs_j[k])
            u_p = process_pair(i, j)
            u_p == Inf && return Inf
            u_total += u_p
        end
    else
        # Direct all-pairs traversal
        @inbounds for i in 1:(n - 1)
            for j in (i + 1):n
                u_p = process_pair(i, j)
                u_p == Inf && return Inf
                u_total += u_p
            end
        end
    end

    if fac.use_carry_free
        resolve_forces!(F, force_buf; accumulate = true)
    end

    return u_total
end

# Factor evaluation interface
function Factors.evaluate!(F::AbstractMatrix{Float64}, fac::TriumTiledNonbondedFactor,
                           q::AbstractMatrix{Float64}, ctx::Factors.Context)
    return eval_lj_dsf_pairs!(F, nothing, q, ctx.box, fac)
end

function Factors.evaluate!(F::AbstractMatrix{Float64}, fac::TriumTiledNonbondedFactor,
                           q::AbstractMatrix{Float64}, box::Types.AbstractBox)
    return eval_lj_dsf_pairs!(F, nothing, q, box, fac)
end

"""
    energy_and_forces!(F, box_deriv, sys, fac::TriumTiledNonbondedFactor) -> Float64

Computes the energy, scatters forces into `F`, and accumulates box derivatives into `box_deriv`.
Cleanly dispatches on `Types.System`, `Types.AtomSet`, or raw coordinate matrices.
"""
function energy_and_forces!(F::AbstractMatrix{Float64},
                            box_deriv::Union{Nothing, AbstractMatrix{Float64}, AbstractVector{Float64}},
                            sys::Types.System,
                            fac::TriumTiledNonbondedFactor)::Float64
    return eval_lj_dsf_pairs!(F, box_deriv, sys.atoms.q, sys.box, fac)
end

function energy_and_forces!(F::AbstractMatrix{Float64},
                            box_deriv::Union{Nothing, AbstractMatrix{Float64}, AbstractVector{Float64}},
                            q::AbstractMatrix{Float64},
                            box::Types.AbstractBox,
                            fac::TriumTiledNonbondedFactor)::Float64
    return eval_lj_dsf_pairs!(F, box_deriv, q, box, fac)
end

function energy_and_forces!(F::AbstractMatrix{Float64},
                            box_deriv::Union{Nothing, AbstractMatrix{Float64}, AbstractVector{Float64}},
                            q::AbstractMatrix{Float64},
                            ctx::Factors.Context,
                            fac::TriumTiledNonbondedFactor)::Float64
    return eval_lj_dsf_pairs!(F, box_deriv, q, ctx.box, fac)
end

# Forwarding methods for Factors / RS compatibility
Factors.energy_and_forces!(F::AbstractMatrix{Float64}, box_deriv, sys, fac::TriumTiledNonbondedFactor) =
    energy_and_forces!(F, box_deriv, sys, fac)

"""
    accumulate_box_derivative!(G::AbstractMatrix{Float64}, fac::TriumTiledNonbondedFactor, q, box) -> G

Accumulates the exact analytic box derivative ∂U/∂h into caller-supplied 3x3 matrix `G`.
"""
function accumulate_box_derivative!(G::AbstractMatrix{Float64},
                                    fac::TriumTiledNonbondedFactor,
                                    q::AbstractMatrix{Float64},
                                    box::Types.AbstractBox)
    size(G) == (3, 3) || throw(DimensionMismatch("G must be 3x3"))
    F_scratch = zeros(Float64, size(q))
    eval_lj_dsf_pairs!(F_scratch, G, q, box, fac)
    return G
end

function accumulate_box_derivative!(G::AbstractMatrix{Float64},
                                    fac::TriumTiledNonbondedFactor,
                                    q::AbstractMatrix{Float64},
                                    ctx::Factors.Context)
    return accumulate_box_derivative!(G, fac, q, ctx.box)
end

Factors.accumulate_box_derivative!(G::AbstractMatrix{Float64},
                                   fac::TriumTiledNonbondedFactor,
                                   q::AbstractMatrix{Float64},
                                   ctx::Factors.Context) =
    accumulate_box_derivative!(G, fac, q, ctx.box)

function accumulate_box_derivative(fac::TriumTiledNonbondedFactor,
                                   q::AbstractMatrix{Float64},
                                   box::Types.AbstractBox)
    G = zeros(Float64, 3, 3)
    accumulate_box_derivative!(G, fac, q, box)
    return G
end

# ResolvedSystem extension if loaded
if _HAS_RESOLVED
    _RS.accumulate_box_derivative!(G::Matrix{Float64}, fac::TriumTiledNonbondedFactor,
                                   q::AbstractMatrix{Float64}, ctx::Factors.Context) =
        accumulate_box_derivative!(G, fac, q, ctx.box)
    _RS.box_derivative_support(::TriumTiledNonbondedFactor) = nothing
end

end # module TriumXBT
