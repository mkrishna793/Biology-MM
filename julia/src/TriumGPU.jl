"""
TriumGPU — Universal Multi-Backend GPU Engine & Adaptive Hardware Auto-Tuner
for Molecular Mechanics and Balanced-Ternary Acceleration.

Supports dynamic probing and zero-crash fallback across:
1. Apple Metal (`libtrium_gpu_bridge.dylib`) — unified zero-copy memory
2. Vulkan Compute (`libtrium_gpu_vulkan.dylib`) — portable cross-vendor GPU
3. SYCL / oneAPI (`libtrium_gpu_sycl.dylib`) — heterogeneous GPU/accelerator
4. CPU / ARM64 NEON (`TriumXBT`) — multi-threaded deterministic fallback

Includes:
- `AdaptiveGPUProfile` with hardware-tuned workgroup size and dynamic
  CPU/GPU pair-count crossover (`~150_000` for unified memory, `~600_000` for discrete PCIe).
- Cached GPU compiled CSR Verlet plans (`xbt_gpu_verlet_plan_*`) with exact
  invalidation on `tc.n_rebuilds`, box geometry type, and factor parameters.
- `TriumGPUNonbondedFactor <: Factors.AbstractFactor` and `eval_lj_dsf_gpu!`.
- `gpu_ternary_matvec!` (dual-rail POPCNT ternary matrix-vector product).
- `gpu_msm_stencil27!` (27-point 3D periodic radix-3 MSM stencil convolution).
"""
module TriumGPU

using LinearAlgebra
const Libdl = Base.Libc.Libdl

const _PARENT = parentmodule(@__MODULE__)

using ..Types
using ..Factors
using ..Neighbors
using ..Dynamics
using ..TriumXBT

const _HAS_RESOLVED = isdefined(_PARENT, :ResolvedSystem)
const _RS = _HAS_RESOLVED ? getfield(_PARENT, :ResolvedSystem) : nothing

if !isdefined(Factors, :accumulate_box_derivative!)
    @eval Factors function accumulate_box_derivative! end
end
if !isdefined(Factors, :energy_and_forces!)
    @eval Factors function energy_and_forces! end
end

export AdaptiveGPUProfile,
       TriumGPUNonbondedFactor,
       probe_gpu_device,
       available_gpu_backends,
       init_gpu_bridge!,
       has_gpu_bridge,
       has_gpu_md_batch,
       eval_lj_dsf_gpu!,
       run_batched_gpu_steps!,
       gpu_verlet_num_pairs,
       gpu_ternary_matvec!,
       gpu_msm_stencil27!,
       clear_gpu_plan_cache!

# ==============================================================================
# 1. Adaptive GPU Hardware Profile & Auto-Tuner
# ==============================================================================

const _VALID_GPU_BACKENDS = (:metal, :vulkan, :sycl, :cpu)

"""
    AdaptiveGPUProfile

Hardware capability and auto-tuning profile for GPU/CPU dispatch:
- `backend`: `:metal`, `:vulkan`, `:sycl`, or `:cpu`
- `device_name`: human-readable device identifier
- `has_gpu`: `true` when an accelerated GPU backend bridge is active
- `unified_memory`: `true` for zero-copy SoC memory (e.g. Apple Silicon Metal)
- `max_threads_per_tg`: hardware threadgroup / workgroup limit
- `simd_width`: warp / subgroup / SIMD width (`32` Apple/NVIDIA, `64` AMD, `1` CPU)
- `vram_bytes`: available GPU memory (or host memory on unified-memory / CPU)
- `auto_workgroup_size`: tuned workgroup size from `max_threads_per_tg` and `simd_width`
- `auto_gpu_crossover_pairs`: pair threshold above which `:auto` offloads to GPU
  (`~150_000` for unified memory, `~600_000` for discrete PCIe GPUs)
"""
struct AdaptiveGPUProfile
    backend::Symbol
    device_name::String
    has_gpu::Bool
    unified_memory::Bool
    max_threads_per_tg::Int
    simd_width::Int
    vram_bytes::UInt64
    auto_workgroup_size::Int
    auto_gpu_crossover_pairs::Int

    function AdaptiveGPUProfile(
        backend::Symbol,
        device_name::AbstractString,
        has_gpu::Bool,
        unified_memory::Bool,
        max_threads_per_tg::Integer,
        simd_width::Integer,
        vram_bytes::Integer,
        auto_workgroup_size::Integer,
        auto_gpu_crossover_pairs::Integer
    )
        backend in _VALID_GPU_BACKENDS ||
            throw(ArgumentError("unsupported GPU backend $(repr(backend)); expected one of $_VALID_GPU_BACKENDS"))
        mtg = Int(max_threads_per_tg)
        sw = Int(simd_width)
        wg = Int(auto_workgroup_size)
        xover = Int(auto_gpu_crossover_pairs)
        mtg >= 1 || throw(ArgumentError("max_threads_per_tg must be >= 1, got $mtg"))
        sw >= 1 || throw(ArgumentError("simd_width must be >= 1, got $sw"))
        wg >= 1 || throw(ArgumentError("auto_workgroup_size must be >= 1, got $wg"))
        xover >= 0 || throw(ArgumentError("auto_gpu_crossover_pairs must be >= 0, got $xover"))
        vram_bytes >= 0 || throw(ArgumentError("vram_bytes must be non-negative"))
        new(backend, String(device_name), has_gpu, unified_memory, mtg, sw,
            UInt64(vram_bytes), wg, xover)
    end
end

@inline function _compute_auto_workgroup_size(has_gpu::Bool, max_threads_per_tg::Int, simd_width::Int)::Int
    if !has_gpu
        return min(64, max(1, max_threads_per_tg))
    end
    sw = max(1, simd_width)
    mtg = max(sw, max_threads_per_tg)
    target = clamp(4 * sw, sw, mtg)
    aligned = (target ÷ sw) * sw
    return max(sw, aligned)
end

@inline function _compute_auto_crossover_pairs(has_gpu::Bool, unified_memory::Bool)::Int
    has_gpu || return typemax(Int)
    return unified_memory ? 150_000 : 600_000
end

"""
    AdaptiveGPUProfile(; backend, device_name, has_gpu, unified_memory,
                       max_threads_per_tg=1024, simd_width=32, vram_bytes=0)

Constructs an `AdaptiveGPUProfile` with automatically tuned `auto_workgroup_size`
and `auto_gpu_crossover_pairs`.
"""
function AdaptiveGPUProfile(;
    backend::Symbol = :cpu,
    device_name::AbstractString = backend === :cpu ? "CPU (TriumXBT)" : "GPU ($backend)",
    has_gpu::Bool = (backend !== :cpu),
    unified_memory::Bool = (backend === :metal || !has_gpu),
    max_threads_per_tg::Integer = has_gpu ? 1024 : max(1, Threads.nthreads()),
    simd_width::Integer = has_gpu ? 32 : 1,
    vram_bytes::Integer = UInt64(Sys.total_memory()),
    auto_workgroup_size::Union{Nothing, Integer} = nothing,
    auto_gpu_crossover_pairs::Union{Nothing, Integer} = nothing
)
    sw = has_gpu ? max(1, Int(simd_width)) : 1
    mtg = max(sw, Int(max_threads_per_tg))
    wg = auto_workgroup_size === nothing ?
         _compute_auto_workgroup_size(has_gpu, mtg, sw) :
         Int(auto_workgroup_size)
    xover = auto_gpu_crossover_pairs === nothing ?
            _compute_auto_crossover_pairs(has_gpu, unified_memory) :
            Int(auto_gpu_crossover_pairs)
    return AdaptiveGPUProfile(
        backend, device_name, has_gpu, unified_memory,
        mtg, sw, vram_bytes, wg, xover
    )
end

# ==============================================================================
# 2. Dynamic Multi-Backend Bridge Loader
# ==============================================================================

const _GPU_BRIDGE_ORDER = (:metal, :vulkan, :sycl)

function _gpu_lib_stem(backend::Symbol)::String
    if backend === :metal
        return "trium_gpu_bridge"
    elseif backend === :vulkan
        return "trium_gpu_vulkan"
    elseif backend === :sycl
        return "trium_gpu_sycl"
    end
    throw(ArgumentError("no GPU bridge library associated with backend $(repr(backend))"))
end

function _gpu_lib_filename(backend::Symbol)::String
    return "lib" * _gpu_lib_stem(backend) * "." * Libdl.dlext
end

function _candidate_gpu_lib_paths(backend::Symbol)::Vector{String}
    ext = Libdl.dlext
    stem = _gpu_lib_stem(backend)
    fnames = ("lib" * stem * "." * ext, stem * "." * ext)
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

mutable struct _GpuBackendHandle
    backend::Symbol
    handle::Ptr{Cvoid}
    path::String
    sym_probe_device::Ptr{Cvoid}
    sym_verlet_create::Ptr{Cvoid}
    sym_verlet_eval::Ptr{Cvoid}
    sym_verlet_destroy::Ptr{Cvoid}
    sym_verlet_num_pairs::Ptr{Cvoid}
    sym_ternary_matvec::Ptr{Cvoid}
    sym_msm_stencil27::Ptr{Cvoid}
    sym_is_available::Ptr{Cvoid}
    sym_unified_memory::Ptr{Cvoid}
    sym_max_threads::Ptr{Cvoid}
    sym_simd_width::Ptr{Cvoid}
    sym_vram_bytes::Ptr{Cvoid}
    md_batch_run::Ptr{Cvoid}
end

function _GpuBackendHandle(backend::Symbol)
    _GpuBackendHandle(
        backend, C_NULL, "",
        C_NULL,
        C_NULL, C_NULL, C_NULL, C_NULL,
        C_NULL, C_NULL,
        C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
        C_NULL
    )
end

const _GPU_BRIDGE_LOCK = ReentrantLock()
const _GPU_BACKENDS = Dict{Symbol, _GpuBackendHandle}(
    :metal  => _GpuBackendHandle(:metal),
    :vulkan => _GpuBackendHandle(:vulkan),
    :sycl   => _GpuBackendHandle(:sycl),
)
const _GPU_ACTIVE_BACKEND = Ref{Symbol}(:cpu)
const _GPU_PROBED_ONCE = Ref{Bool}(false)

@inline function _dlsym_safe(h::Ptr{Cvoid}, sym::Symbol)::Ptr{Cvoid}
    h == C_NULL && return C_NULL
    p = Libdl.dlsym(h, sym; throw_error = false)
    return p === nothing ? C_NULL : p
end

function _populate_backend_symbols!(st::_GpuBackendHandle, h::Ptr{Cvoid}, path::String)
    st.handle = h
    st.path = path
    st.sym_probe_device     = _dlsym_safe(h, :xbt_gpu_probe_device)
    st.sym_verlet_create    = _dlsym_safe(h, :xbt_gpu_verlet_plan_create)
    st.sym_verlet_eval      = _dlsym_safe(h, :xbt_gpu_verlet_plan_eval)
    st.sym_verlet_destroy   = _dlsym_safe(h, :xbt_gpu_verlet_plan_destroy)
    st.sym_verlet_num_pairs = _dlsym_safe(h, :xbt_gpu_verlet_plan_num_pairs)
    st.sym_ternary_matvec   = _dlsym_safe(h, :xbt_gpu_ternary_matvec_dualrail)
    st.sym_msm_stencil27    = _dlsym_safe(h, :xbt_gpu_msm_stencil27)
    avail_sym = _dlsym_safe(h, :xbt_gpu_is_available)
    if avail_sym == C_NULL && st.backend === :vulkan
        avail_sym = _dlsym_safe(h, :xbt_vulkan_available)
    end
    st.sym_is_available     = avail_sym
    st.sym_unified_memory   = _dlsym_safe(h, :xbt_gpu_unified_memory)
    st.sym_max_threads      = _dlsym_safe(h, :xbt_gpu_max_threads_per_tg)
    st.sym_simd_width       = _dlsym_safe(h, :xbt_gpu_simd_width)
    st.sym_vram_bytes       = _dlsym_safe(h, :xbt_gpu_vram_bytes)
    st.md_batch_run         = _dlsym_safe(h, :xbt_gpu_md_batch_run)
    return st
end

function _infer_backend_from_path(path::AbstractString)::Symbol
    lp = lowercase(basename(String(path)))
    occursin("vulkan", lp) && return :vulkan
    occursin("sycl", lp) && return :sycl
    return :metal
end

@inline function _backend_hw_available(st::_GpuBackendHandle)::Bool
    st.handle == C_NULL && return false
    if st.sym_is_available != C_NULL
        return ccall(st.sym_is_available, Cint, ()) != Cint(0)
    end
    return true
end

function _clear_backend_handle!(st::_GpuBackendHandle)
    st.handle = C_NULL
    st.path = ""
    st.sym_probe_device = C_NULL
    st.sym_verlet_create = C_NULL
    st.sym_verlet_eval = C_NULL
    st.sym_verlet_destroy = C_NULL
    st.sym_verlet_num_pairs = C_NULL
    st.sym_ternary_matvec = C_NULL
    st.sym_msm_stencil27 = C_NULL
    st.sym_is_available = C_NULL
    st.sym_unified_memory = C_NULL
    st.sym_max_threads = C_NULL
    st.sym_simd_width = C_NULL
    st.sym_vram_bytes = C_NULL
    st.md_batch_run = C_NULL
    return st
end

function _try_open_backend!(backend::Symbol)::Bool
    st = _GPU_BACKENDS[backend]
    if st.handle != C_NULL
        return _backend_hw_available(st)
    end
    for p in _candidate_gpu_lib_paths(backend)
        if p == basename(p) || isfile(p)
            h = Libdl.dlopen_e(p)
            if h != C_NULL
                _populate_backend_symbols!(st, h, String(p))
                if _backend_hw_available(st)
                    return true
                end
                _clear_backend_handle!(st)
                Libdl.dlclose(h)
            end
        end
    end
    return false
end

"""
    init_gpu_bridge!(path::Union{Nothing, AbstractString} = nothing; backend::Symbol = :auto) -> Bool

Loads one or more native GPU acceleration bridges (`libtrium_gpu_bridge.dylib` for Metal,
`libtrium_gpu_vulkan.dylib` for Vulkan, `libtrium_gpu_sycl.dylib` for SYCL).
Returns `true` if a GPU bridge is loaded and active, or `false` when falling back to `:cpu`.
"""
function init_gpu_bridge!(path::Union{Nothing, AbstractString} = nothing; backend::Symbol = :auto)::Bool
    lock(_GPU_BRIDGE_LOCK)
    try
        if path !== nothing
            if isfile(path)
                h = Libdl.dlopen_e(String(path))
                if h != C_NULL
                    b = backend === :auto ? _infer_backend_from_path(path) : backend
                    haskey(_GPU_BACKENDS, b) ||
                        throw(ArgumentError("unsupported GPU backend $(repr(b))"))
                    _populate_backend_symbols!(_GPU_BACKENDS[b], h, String(path))
                    _GPU_ACTIVE_BACKEND[] = b
                    _GPU_PROBED_ONCE[] = true
                    return true
                end
            end
            _GPU_PROBED_ONCE[] = true
            return false
        end

        if backend === :cpu
            _GPU_ACTIVE_BACKEND[] = :cpu
            _GPU_PROBED_ONCE[] = true
            return false
        elseif backend === :auto || backend === :gpu
            _GPU_PROBED_ONCE[] = true
            chosen = :cpu
            for b in _GPU_BRIDGE_ORDER
                if _try_open_backend!(b) && chosen === :cpu
                    chosen = b
                end
            end
            _GPU_ACTIVE_BACKEND[] = chosen
            return chosen !== :cpu
        elseif haskey(_GPU_BACKENDS, backend)
            _GPU_PROBED_ONCE[] = true
            if _try_open_backend!(backend)
                _GPU_ACTIVE_BACKEND[] = backend
                return true
            end
            _GPU_ACTIVE_BACKEND[] = :cpu
            return false
        else
            throw(ArgumentError("unknown backend $(repr(backend))"))
        end
    finally
        unlock(_GPU_BRIDGE_LOCK)
    end
end

"""
    has_gpu_bridge(backend::Symbol = :auto) -> Bool

Returns `true` if a GPU bridge library is loaded (for the specified `backend` or any active GPU backend).
"""
function has_gpu_bridge(backend::Symbol = :auto)::Bool
    lock(_GPU_BRIDGE_LOCK)
    try
        if !_GPU_PROBED_ONCE[]
            init_gpu_bridge!(nothing; backend = :auto)
        end
        if backend === :auto || backend === :gpu
            return _GPU_ACTIVE_BACKEND[] !== :cpu && _backend_hw_available(_GPU_BACKENDS[_GPU_ACTIVE_BACKEND[]])
        elseif backend === :cpu
            return false
        elseif haskey(_GPU_BACKENDS, backend)
            st = _GPU_BACKENDS[backend]
            return _backend_hw_available(st)
        else
            return false
        end
    finally
        unlock(_GPU_BRIDGE_LOCK)
    end
end

"""
    has_gpu_md_batch(backend::Symbol = :auto) -> Bool

Returns `true` if a GPU bridge library is loaded for `backend` (or the active GPU backend)
and exports both the compiled Verlet plan ABI and `xbt_gpu_md_batch_run`.
"""
function has_gpu_md_batch(backend::Symbol = :auto)::Bool
    lock(_GPU_BRIDGE_LOCK)
    try
        if !_GPU_PROBED_ONCE[]
            init_gpu_bridge!(nothing; backend = :auto)
        end
        target = if backend === :auto || backend === :gpu
            _GPU_ACTIVE_BACKEND[]
        else
            backend
        end
        (target === :cpu || !haskey(_GPU_BACKENDS, target)) && return false
        st = _GPU_BACKENDS[target]
        return _backend_hw_available(st) && _gpu_verlet_abi_ready(target) && st.md_batch_run != C_NULL
    finally
        unlock(_GPU_BRIDGE_LOCK)
    end
end

"""
    available_gpu_backends() -> Vector{Symbol}

Probes all supported GPU bridge libraries and returns available backends in priority order,
always ending with `:cpu` (`TriumXBT` fallback).
"""
function available_gpu_backends()::Vector{Symbol}
    lock(_GPU_BRIDGE_LOCK)
    try
        avail = Symbol[]
        for b in _GPU_BRIDGE_ORDER
            if _try_open_backend!(b)
                push!(avail, b)
            end
        end
        push!(avail, :cpu)
        _GPU_PROBED_ONCE[] = true
        if _GPU_ACTIVE_BACKEND[] === :cpu && length(avail) > 1
            _GPU_ACTIVE_BACKEND[] = avail[1]
        end
        return avail
    finally
        unlock(_GPU_BRIDGE_LOCK)
    end
end

function _macos_cpu_brand()::String
    !Sys.isapple() && return string(Sys.ARCH)
    try
        s = strip(read(`sysctl -n machdep.cpu.brand_string`, String))
        return isempty(s) ? "Apple Silicon" : s
    catch
        return "Apple Silicon"
    end
end

function _build_cpu_profile()::AdaptiveGPUProfile
    nt = max(1, Threads.nthreads())
    brand = _macos_cpu_brand()
    arch_tag = Sys.ARCH === :aarch64 ? "ARM64 NEON" : string(Sys.ARCH)
    return AdaptiveGPUProfile(
        backend = :cpu,
        device_name = "$(brand) CPU ($(arch_tag), $(nt) threads)",
        has_gpu = false,
        unified_memory = true,
        max_threads_per_tg = nt,
        simd_width = 1,
        vram_bytes = UInt64(Sys.total_memory()),
        auto_workgroup_size = min(64, nt),
        auto_gpu_crossover_pairs = typemax(Int)
    )
end

function _probe_loaded_backend(st::_GpuBackendHandle)::AdaptiveGPUProfile
    if st.sym_probe_device != C_NULL
        name_buf = zeros(UInt8, 256)
        backend_id_ref = Ref{Int32}(0)
        unified_ref = Ref{Int32}(0)
        mtg_ref = Ref{Int32}(0)
        sw_ref = Ref{Int32}(0)
        vram_ref = Ref{UInt64}(0)
        ok = GC.@preserve name_buf backend_id_ref unified_ref mtg_ref sw_ref vram_ref ccall(
            st.sym_probe_device, Int32,
            (Ptr{UInt8}, Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{UInt64}),
            pointer(name_buf), Int32(length(name_buf)),
            backend_id_ref, unified_ref, mtg_ref, sw_ref, vram_ref
        )
        ok != Int32(0) || return _build_cpu_profile()
        nul = findfirst(iszero, name_buf)
        dev_name = nul === nothing ? String(name_buf) : String(name_buf[1:(nul - 1)])
        isempty(dev_name) && (dev_name = "GPU ($(st.backend))")
        return AdaptiveGPUProfile(
            backend = st.backend,
            device_name = dev_name,
            has_gpu = true,
            unified_memory = unified_ref[] != Int32(0),
            max_threads_per_tg = mtg_ref[] > 0 ? Int(mtg_ref[]) : 1024,
            simd_width = sw_ref[] > 0 ? Int(sw_ref[]) : 32,
            vram_bytes = vram_ref[]
        )
    end

    is_apple_arm = Sys.isapple() && Sys.ARCH === :aarch64
    gpu_avail = true
    if st.sym_is_available != C_NULL
        gpu_avail = ccall(st.sym_is_available, Cint, ()) != Cint(0)
    end
    gpu_avail || return _build_cpu_profile()

    unified = st.backend === :metal ? true : is_apple_arm
    if st.sym_unified_memory != C_NULL
        unified = ccall(st.sym_unified_memory, Cint, ()) != Cint(0)
    end

    mtg = st.backend === :metal ? 1024 : 256
    if st.sym_max_threads != C_NULL
        val = Int(ccall(st.sym_max_threads, Cint, ()))
        if val > 0
            mtg = val
        end
    end

    sw = (st.backend === :metal || is_apple_arm) ? 32 : 64
    if st.sym_simd_width != C_NULL
        val = Int(ccall(st.sym_simd_width, Cint, ()))
        if val > 0
            sw = val
        end
    end

    vram = UInt64(Sys.total_memory())
    if st.sym_vram_bytes != C_NULL
        val = ccall(st.sym_vram_bytes, UInt64, ())
        if val > 0
            vram = val
        end
    end

    brand = is_apple_arm ? _macos_cpu_brand() : uppercase(string(st.backend))
    dev_name = st.backend === :metal ? "$(brand) GPU (Metal)" :
               st.backend === :vulkan ? "$(brand) GPU (Vulkan)" :
               "$(brand) GPU (SYCL)"

    return AdaptiveGPUProfile(
        backend = st.backend,
        device_name = dev_name,
        has_gpu = true,
        unified_memory = unified,
        max_threads_per_tg = mtg,
        simd_width = sw,
        vram_bytes = vram
    )
end

"""
    probe_gpu_device(; backend::Symbol = :auto) -> AdaptiveGPUProfile

Probes available GPU hardware and bridge libraries (`libtrium_gpu_bridge.dylib`,
`libtrium_gpu_vulkan.dylib`, `libtrium_gpu_sycl.dylib`), returning an auto-tuned
`AdaptiveGPUProfile`. Gracefully falls back to `:cpu` (`TriumXBT` NEON) when no GPU
bridge is available.
"""
function probe_gpu_device(; backend::Symbol = :auto)::AdaptiveGPUProfile
    backend in (:auto, :gpu, :metal, :vulkan, :sycl, :cpu) ||
        throw(ArgumentError("unsupported backend $(repr(backend))"))

    backend === :cpu && return _build_cpu_profile()

    lock(_GPU_BRIDGE_LOCK)
    try
        if !_GPU_PROBED_ONCE[]
            init_gpu_bridge!(nothing; backend = :auto)
        end

        target = if backend === :auto || backend === :gpu
            _GPU_ACTIVE_BACKEND[]
        else
            _try_open_backend!(backend) ? backend : :cpu
        end

        if target === :cpu || !haskey(_GPU_BACKENDS, target)
            return _build_cpu_profile()
        end
        st = _GPU_BACKENDS[target]
        st.handle == C_NULL && return _build_cpu_profile()
        return _probe_loaded_backend(st)
    finally
        unlock(_GPU_BRIDGE_LOCK)
    end
end

# ==============================================================================
# 3. GPU Compiled CSR Verlet Plan Cache & Nonbonded Evaluator
# ==============================================================================

mutable struct TriumGPUVerletPlan
    ptr::Ptr{Cvoid}
    backend::Symbol
    destroy_sym::Ptr{Cvoid}
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

    function TriumGPUVerletPlan(
        ptr::Ptr{Cvoid},
        backend::Symbol,
        destroy_sym::Ptr{Cvoid},
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
            ptr, backend, destroy_sym, last_rebuilds, n_atoms,
            rcut, alpha, coulomb_prefactor, lj_mode, lj14_scale, coul14_scale,
            is_triclinic, num_pairs
        )
        finalizer(p -> begin
            if p.ptr != C_NULL && p.destroy_sym != C_NULL
                ccall(p.destroy_sym, Cvoid, (Ptr{Cvoid},), p.ptr)
                p.ptr = C_NULL
            end
        end, plan)
        return plan
    end
end

const _GPU_PLAN_CACHE_LOCK = ReentrantLock()
const _GPU_PLAN_CACHE = Dict{Tuple{Symbol, UInt64, UInt64}, TriumGPUVerletPlan}()
const _GPU_PLAN_ORDER = Tuple{Symbol, UInt64, UInt64}[]
const _MAX_GPU_PLAN_CACHE = 64

"""
    clear_gpu_plan_cache!()

Evicts all cached GPU compiled CSR Verlet plans and clears the underlying CPU plan cache.
"""
function clear_gpu_plan_cache!()
    lock(_GPU_PLAN_CACHE_LOCK)
    try
        empty!(_GPU_PLAN_CACHE)
        empty!(_GPU_PLAN_ORDER)
    finally
        unlock(_GPU_PLAN_CACHE_LOCK)
    end
    TriumXBT.clear_verlet_plan_cache!()
    return nothing
end

@inline function _gpu_plan_valid(
    plan::TriumGPUVerletPlan,
    backend::Symbol,
    tc::Neighbors.TilePairCache,
    fac::TriumXBT.TriumTiledNonbondedFactor,
    n::Int,
    is_tri::Bool
)::Bool
    return plan.ptr != C_NULL &&
           plan.backend === backend &&
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

@inline function _gpu_verlet_abi_ready(backend::Symbol)::Bool
    haskey(_GPU_BACKENDS, backend) || return false
    st = _GPU_BACKENDS[backend]
    return st.handle != C_NULL &&
           st.sym_verlet_create != C_NULL &&
           st.sym_verlet_eval != C_NULL &&
           st.sym_verlet_destroy != C_NULL
end

function _get_or_create_gpu_plan!(
    backend::Symbol,
    q_mat::Matrix{Float64},
    H_ptr::Ptr{Cdouble},
    Hinv_ptr::Ptr{Cdouble},
    is_tri::Cint,
    fac::TriumXBT.TriumTiledNonbondedFactor,
    tc::Neighbors.TilePairCache
)::Union{Nothing, TriumGPUVerletPlan}
    _gpu_verlet_abi_ready(backend) || return nothing
    st = _GPU_BACKENDS[backend]
    n = length(fac.charges)
    n_tp = length(tc.tile_pairs_i)
    (n == 0 || n_tp == 0) && return nothing

    is_tri_bool = (is_tri != Cint(0))
    key = (backend, objectid(fac), objectid(tc))

    lock(_GPU_PLAN_CACHE_LOCK)
    try
        existing = get(_GPU_PLAN_CACHE, key, nothing)
        if existing !== nothing && _gpu_plan_valid(existing, backend, tc, fac, n, is_tri_bool)
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
            st.sym_verlet_create, Ptr{Cvoid},
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

        npairs = st.sym_verlet_num_pairs != C_NULL ?
            Int(ccall(st.sym_verlet_num_pairs, Csize_t, (Ptr{Cvoid},), plan_ptr)) : 0

        plan = TriumGPUVerletPlan(
            plan_ptr, backend, st.sym_verlet_destroy, tc.n_rebuilds, n,
            fac.rcut, fac.alpha, fac.coulomb_prefactor,
            fac.lj_mode, fac.lj14_scale, fac.coul14_scale,
            is_tri_bool, npairs
        )

        if !haskey(_GPU_PLAN_CACHE, key)
            while length(_GPU_PLAN_CACHE) >= _MAX_GPU_PLAN_CACHE && !isempty(_GPU_PLAN_ORDER)
                evict_key = popfirst!(_GPU_PLAN_ORDER)
                delete!(_GPU_PLAN_CACHE, evict_key)
            end
            push!(_GPU_PLAN_ORDER, key)
        end
        _GPU_PLAN_CACHE[key] = plan
        return plan
    finally
        unlock(_GPU_PLAN_CACHE_LOCK)
    end
end

@inline function _estimate_active_pairs(
    fac::TriumXBT.TriumTiledNonbondedFactor,
    q::AbstractMatrix{Float64},
    box::Types.AbstractBox
)::Int
    n = length(fac.charges)
    n <= 1 && return 0
    tc = fac.cache
    if tc isa Neighbors.TilePairCache
        Neighbors.refresh_tile_cache!(tc, q, box)
        # Upper-bounded estimate from interacting tile pairs (each tile <= TILE_SIZE=32 atoms)
        return min((n * (n - 1)) ÷ 2, length(tc.tile_pairs_i) * 256)
    elseif tc isa Neighbors.PairCache
        Neighbors.refresh_pair_cache!(tc, q, box)
        return length(tc.pairs_i)
    else
        return (n * (n - 1)) ÷ 2
    end
end

@inline function _resolve_nonbonded_backend(
    requested::Symbol,
    profile::AdaptiveGPUProfile,
    fac::TriumXBT.TriumTiledNonbondedFactor,
    est_pairs::Int
)::Symbol
    requested in (:auto, :cpu, :gpu, :metal, :vulkan, :sycl) ||
        throw(ArgumentError("unsupported backend $(repr(requested))"))

    requested === :cpu && return :cpu
    fac.use_carry_free && return :cpu
    (fac.alpha > 0.0 || fac.coulomb_prefactor == 0.0 || all(iszero, fac.charges)) || return :cpu

    if requested === :auto
        profile.has_gpu || return :cpu
        est_pairs >= profile.auto_gpu_crossover_pairs || return :cpu
        has_gpu_bridge(profile.backend) && _gpu_verlet_abi_ready(profile.backend) && return profile.backend
        return :cpu
    end

    target = requested === :gpu ? profile.backend : requested
    if target !== :cpu && has_gpu_bridge(target) && _gpu_verlet_abi_ready(target)
        return target
    end
    return :cpu
end

function _eval_verlet_plan_gpu!(
    backend::Symbol,
    F::AbstractMatrix{Float64},
    box_deriv::Union{Nothing, AbstractMatrix{Float64}, AbstractVector{Float64}},
    q::AbstractMatrix{Float64},
    box::Types.AbstractBox,
    fac::TriumXBT.TriumTiledNonbondedFactor,
    tc::Neighbors.TilePairCache
)::Union{Nothing, Float64}
    st = _GPU_BACKENDS[backend]
    n = length(fac.charges)
    n_tp = length(tc.tile_pairs_i)
    (n == 0 || n_tp == 0) && return 0.0

    scratch = TriumXBT._acquire_scratch()
    try
        q_mat = TriumXBT._ensure_contiguous_q(scratch, q, n)
        f_buf = TriumXBT._ensure_forces_buf!(scratch, n)
        v_buf = scratch.virial9
        need_vir = (box_deriv !== nothing) && !(box isa Types.NoBox)
        vir_ptr = need_vir ? pointer(v_buf) : Ptr{Cdouble}(C_NULL)

        is_tri, H_mat, Hinv_mat = TriumXBT._prepare_box_matrices!(scratch, box)
        H_ptr = H_mat === nothing ? Ptr{Cdouble}(C_NULL) : pointer(H_mat)
        Hinv_ptr = Hinv_mat === nothing ? Ptr{Cdouble}(C_NULL) : pointer(Hinv_mat)

        plan = GC.@preserve H_mat Hinv_mat _get_or_create_gpu_plan!(
            backend, q_mat, H_ptr, Hinv_ptr, is_tri, fac, tc
        )
        plan === nothing && return nothing

        u_pair = GC.@preserve plan q_mat H_mat Hinv_mat f_buf v_buf ccall(
            st.sym_verlet_eval, Cdouble,
            (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
             Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}),
            plan.ptr, pointer(q_mat), H_ptr, Hinv_ptr,
            is_tri, Cint(need_vir ? 1 : 0),
            pointer(f_buf), vir_ptr
        )

        TriumXBT._accumulate_forces!(F, f_buf, n)
        if need_vir
            TriumXBT._accumulate_virial!(box_deriv, v_buf)
        end

        return _compute_self_energy(fac) + u_pair
    finally
        TriumXBT._release_scratch(scratch)
    end
end

@inline function _compute_self_energy(fac::TriumXBT.TriumTiledNonbondedFactor)::Float64
    fac.include_self || return 0.0
    rc = fac.rcut
    alpha = fac.alpha
    erfc_rc = fac.use_fast_math ? TriumXBT.xbt_erfc_fast(alpha * rc) :
              ccall(:erfc, Cdouble, (Cdouble,), alpha * rc)
    return -fac.coulomb_prefactor * (erfc_rc / (2.0 * rc) + alpha / sqrt(pi)) * sum(abs2, fac.charges)
end

"""
    eval_lj_dsf_gpu!(F, box_deriv, q, box, fac::TriumXBT.TriumTiledNonbondedFactor;
                     backend::Symbol = :auto,
                     profile::AdaptiveGPUProfile = probe_gpu_device()) -> Float64

Evaluates unified LJ + Fennell-Gezelter DSF nonbonded energy, forces (accumulated into `F`),
and optional analytic box derivatives (accumulated into `box_deriv`).

In `:auto` mode, inspects `profile` and active pair count against `profile.auto_gpu_crossover_pairs`
to dispatch between GPU (`:metal`, `:vulkan`, `:sycl`) and CPU/NEON (`TriumXBT`).
"""
function eval_lj_dsf_gpu!(
    F::AbstractMatrix{Float64},
    box_deriv::Union{Nothing, AbstractMatrix{Float64}, AbstractVector{Float64}},
    q::AbstractMatrix{Float64},
    box::Types.AbstractBox,
    fac::TriumXBT.TriumTiledNonbondedFactor;
    backend::Symbol = :auto,
    profile::AdaptiveGPUProfile = probe_gpu_device()
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

    est_pairs = _estimate_active_pairs(fac, q, box)
    target = _resolve_nonbonded_backend(backend, profile, fac, est_pairs)

    if target !== :cpu && fac.cache isa Neighbors.TilePairCache
        u_gpu = _eval_verlet_plan_gpu!(target, F, box_deriv, q, box, fac, fac.cache)
        if u_gpu !== nothing
            return u_gpu
        end
    end

    return TriumXBT.eval_lj_dsf_pairs!(F, box_deriv, q, box, fac)
end

"""
    gpu_verlet_num_pairs(fac::TriumXBT.TriumTiledNonbondedFactor, q, box; backend=:auto) -> Int

Returns the number of active pairs in the compiled GPU CSR Verlet plan (when a GPU backend
is active) or delegates to `TriumXBT.compiled_verlet_num_pairs`.
"""
function gpu_verlet_num_pairs(
    fac::TriumXBT.TriumTiledNonbondedFactor,
    q::AbstractMatrix{Float64},
    box::Types.AbstractBox;
    backend::Symbol = :auto
)::Int
    n = length(fac.charges)
    size(q, 1) == 3 && size(q, 2) == n ||
        throw(ArgumentError("q must have shape 3 x $n"))

    tc = fac.cache
    if tc isa Neighbors.TilePairCache
        prof = probe_gpu_device(; backend = backend === :cpu ? :cpu : :auto)
        target = backend === :cpu ? :cpu : (backend === :auto || backend === :gpu ? prof.backend : backend)
        if target !== :cpu && _gpu_verlet_abi_ready(target)
            Neighbors.refresh_tile_cache!(tc, q, box)
            (n == 0 || isempty(tc.tile_pairs_i)) && return 0
            scratch = TriumXBT._acquire_scratch()
            try
                q_mat = TriumXBT._ensure_contiguous_q(scratch, q, n)
                is_tri, H_mat, Hinv_mat = TriumXBT._prepare_box_matrices!(scratch, box)
                H_ptr = H_mat === nothing ? Ptr{Cdouble}(C_NULL) : pointer(H_mat)
                Hinv_ptr = Hinv_mat === nothing ? Ptr{Cdouble}(C_NULL) : pointer(Hinv_mat)
                plan = GC.@preserve H_mat Hinv_mat _get_or_create_gpu_plan!(
                    target, q_mat, H_ptr, Hinv_ptr, is_tri, fac, tc
                )
                plan !== nothing && return plan.num_pairs
            finally
                TriumXBT._release_scratch(scratch)
            end
        end
    end

    return TriumXBT.compiled_verlet_num_pairs(fac, q, box)
end

# ==============================================================================
# 4. TriumGPUNonbondedFactor <: Factors.AbstractFactor
# ==============================================================================

"""
    TriumGPUNonbondedFactor <: Factors.AbstractFactor

Factor wrapper around `TriumXBT.TriumTiledNonbondedFactor` that routes evaluations
through `TriumGPU.eval_lj_dsf_gpu!` using the adaptive hardware profile.
"""
struct TriumGPUNonbondedFactor <: Factors.AbstractFactor
    inner::TriumXBT.TriumTiledNonbondedFactor
    backend::Symbol
    profile::AdaptiveGPUProfile

    function TriumGPUNonbondedFactor(
        inner::TriumXBT.TriumTiledNonbondedFactor;
        backend::Symbol = :auto,
        profile::AdaptiveGPUProfile = probe_gpu_device()
    )
        backend in (:auto, :cpu, :gpu, :metal, :vulkan, :sycl) ||
            throw(ArgumentError("unsupported backend $(repr(backend))"))
        new(inner, backend, profile)
    end
end

function TriumGPUNonbondedFactor(
    sigma::AbstractVector{<:Real},
    epsilon::AbstractVector{<:Real},
    charges::AbstractVector{<:Real};
    backend::Symbol = :auto,
    profile::AdaptiveGPUProfile = probe_gpu_device(),
    kwargs...
)
    inner = TriumXBT.TriumTiledNonbondedFactor(sigma, epsilon, charges; kwargs...)
    return TriumGPUNonbondedFactor(inner; backend = backend, profile = profile)
end

Factors.factor_support_size(fac::TriumGPUNonbondedFactor) = Factors.factor_support_size(fac.inner)
Factors.read_support(fac::TriumGPUNonbondedFactor) = Factors.read_support(fac.inner)
Factors.is_local(::TriumGPUNonbondedFactor) = false
Factors.force_consistent(fac::TriumGPUNonbondedFactor) = Factors.force_consistent(fac.inner)

function Factors.evaluate!(
    F::AbstractMatrix{Float64},
    fac::TriumGPUNonbondedFactor,
    q::AbstractMatrix{Float64},
    ctx::Factors.Context
)
    return eval_lj_dsf_gpu!(F, nothing, q, ctx.box, fac.inner;
                            backend = fac.backend, profile = fac.profile)
end

function Factors.evaluate!(
    F::AbstractMatrix{Float64},
    fac::TriumGPUNonbondedFactor,
    q::AbstractMatrix{Float64},
    box::Types.AbstractBox
)
    return eval_lj_dsf_gpu!(F, nothing, q, box, fac.inner;
                            backend = fac.backend, profile = fac.profile)
end

function Factors.energy_and_forces!(
    F::AbstractMatrix{Float64},
    box_deriv::Union{Nothing, AbstractMatrix{Float64}, AbstractVector{Float64}},
    sys::Types.System,
    fac::TriumGPUNonbondedFactor
)::Float64
    return eval_lj_dsf_gpu!(F, box_deriv, sys.atoms.q, sys.box, fac.inner;
                            backend = fac.backend, profile = fac.profile)
end

function Factors.energy_and_forces!(
    F::AbstractMatrix{Float64},
    box_deriv::Union{Nothing, AbstractMatrix{Float64}, AbstractVector{Float64}},
    q::AbstractMatrix{Float64},
    box::Types.AbstractBox,
    fac::TriumGPUNonbondedFactor
)::Float64
    return eval_lj_dsf_gpu!(F, box_deriv, q, box, fac.inner;
                            backend = fac.backend, profile = fac.profile)
end

function Factors.accumulate_box_derivative!(
    G::AbstractMatrix{Float64},
    fac::TriumGPUNonbondedFactor,
    q::AbstractMatrix{Float64},
    ctx::Factors.Context
)
    size(G) == (3, 3) || throw(DimensionMismatch("G must be 3x3"))
    F_scratch = zeros(Float64, size(q))
    eval_lj_dsf_gpu!(F_scratch, G, q, ctx.box, fac.inner;
                     backend = fac.backend, profile = fac.profile)
    return G
end

if _HAS_RESOLVED
    _RS.accumulate_box_derivative!(
        G::Matrix{Float64},
        fac::TriumGPUNonbondedFactor,
        q::AbstractMatrix{Float64},
        ctx::Factors.Context
    ) = Factors.accumulate_box_derivative!(G, fac, q, ctx)
    _RS.box_derivative_support(::TriumGPUNonbondedFactor) = nothing
end

# ==============================================================================
# 5. Dual-Rail Ternary MatVec & 27-Point MSM Stencil Evaluators
# ==============================================================================

function _cpu_ternary_matvec_dualrail!(
    y::Vector{Float64},
    W_P::Matrix{UInt64},
    W_N::Matrix{UInt64},
    row_scales::Union{Nothing, Vector{Float64}},
    x_P::Vector{UInt64},
    x_N::Vector{UInt64},
    x_scale::Float64
)
    M, K_words = size(W_P)
    if TriumXBT.has_native_bridge()
        sym = Libdl.dlsym(TriumXBT._NATIVE_BRIDGE_HANDLE[], :xbt_ternary_matvec_dualrail; throw_error = false)
        if sym !== nothing && sym != C_NULL
            rs_ptr = row_scales === nothing ? Ptr{Cdouble}(C_NULL) : pointer(row_scales)
            GC.@preserve y W_P W_N row_scales x_P x_N ccall(
                sym, Cvoid,
                (Csize_t, Csize_t, Ptr{UInt64}, Ptr{UInt64}, Ptr{Cdouble},
                 Ptr{UInt64}, Ptr{UInt64}, Cdouble, Ptr{Cdouble}),
                Csize_t(M), Csize_t(K_words), pointer(W_P), pointer(W_N),
                rs_ptr, pointer(x_P), pointer(x_N), Cdouble(x_scale), pointer(y)
            )
            return y
        end
    end

    @inbounds for m in 1:M
        acc = Int64(0)
        for k in 1:K_words
            wp = W_P[m, k]; wn = W_N[m, k]
            xp = x_P[k]; xn = x_N[k]
            pos = count_ones((wp & xp) | (wn & xn))
            neg = count_ones((wp & xn) | (wn & xp))
            acc += Int64(pos) - Int64(neg)
        end
        r_scale = row_scales === nothing ? 1.0 : row_scales[m]
        y[m] = r_scale * x_scale * Float64(acc)
    end
    return y
end

"""
    gpu_ternary_matvec!(y, W_P, W_N, row_scales, x_P, x_N, x_scale; backend=:auto)

Bit-sliced dual-rail balanced ternary matrix-vector product:
`y[m] = (row_scales[m]) * x_scale * <W[m, :], x>` where `W` and `x` are represented
via disjoint positive/negative `UInt64` bitplanes (`(P & N) == 0`).
Dispatches to `xbt_gpu_ternary_matvec_dualrail` when an active GPU bridge exports it,
otherwise runs CPU `xbt_ternary_matvec_dualrail` (NEON POPCNT) or exact Julia fallback.
"""
function gpu_ternary_matvec!(
    y::AbstractVector{Float64},
    W_P::AbstractMatrix{UInt64},
    W_N::AbstractMatrix{UInt64},
    row_scales::Union{Nothing, AbstractVector{Float64}},
    x_P::AbstractVector{UInt64},
    x_N::AbstractVector{UInt64},
    x_scale::Real;
    backend::Symbol = :auto
)
    M, K_words = size(W_P)
    size(W_N) == (M, K_words) ||
        throw(DimensionMismatch("W_N shape $(size(W_N)) must match W_P shape $(size(W_P))"))
    length(y) == M ||
        throw(DimensionMismatch("output vector y length $(length(y)) must match M=$M"))
    length(x_P) == K_words && length(x_N) == K_words ||
        throw(DimensionMismatch("x_P and x_N lengths must match K_words=$K_words"))
    if row_scales !== nothing && length(row_scales) != M
        throw(DimensionMismatch("row_scales length $(length(row_scales)) must match M=$M"))
    end

    y_vec = y isa Vector{Float64} ? y : collect(Float64, y)
    wp_mat = W_P isa Matrix{UInt64} ? W_P : Matrix{UInt64}(W_P)
    wn_mat = W_N isa Matrix{UInt64} ? W_N : Matrix{UInt64}(W_N)
    xp_vec = x_P isa Vector{UInt64} ? x_P : collect(UInt64, x_P)
    xn_vec = x_N isa Vector{UInt64} ? x_N : collect(UInt64, x_N)
    rs_vec = row_scales === nothing ? nothing :
             (row_scales isa Vector{Float64} ? row_scales : collect(Float64, row_scales))
    xs = Float64(x_scale)

    if M == 0
        return y
    end

    used_gpu = false
    if backend !== :cpu
        prof = probe_gpu_device(; backend = (backend === :auto || backend === :gpu) ? :auto : backend)
        target = (backend === :auto || backend === :gpu) ? prof.backend : backend
        if target !== :cpu && haskey(_GPU_BACKENDS, target)
            st = _GPU_BACKENDS[target]
            if st.handle != C_NULL && st.sym_ternary_matvec != C_NULL
                rs_ptr = rs_vec === nothing ? Ptr{Cdouble}(C_NULL) : pointer(rs_vec)
                GC.@preserve y_vec wp_mat wn_mat rs_vec xp_vec xn_vec ccall(
                    st.sym_ternary_matvec, Cvoid,
                    (Csize_t, Csize_t, Ptr{UInt64}, Ptr{UInt64}, Ptr{Cdouble},
                     Ptr{UInt64}, Ptr{UInt64}, Cdouble, Ptr{Cdouble}),
                    Csize_t(M), Csize_t(K_words), pointer(wp_mat), pointer(wn_mat),
                    rs_ptr, pointer(xp_vec), pointer(xn_vec), Cdouble(xs), pointer(y_vec)
                )
                used_gpu = true
            end
        end
    end

    if !used_gpu
        _cpu_ternary_matvec_dualrail!(y_vec, wp_mat, wn_mat, rs_vec, xp_vec, xn_vec, xs)
    end

    if !(y isa Vector{Float64})
        copyto!(y, y_vec)
    end
    return y
end

function _cpu_msm_stencil27!(
    Phi_out::Array{Float64, 3},
    Q_grid::Array{Float64, 3},
    stencil_k27::Vector{Float64},
    prefactor::Float64
)::Float64
    Mx, My, Mz = size(Q_grid)
    fill!(Phi_out, 0.0)
    @inbounds for dz in -1:1, dy in -1:1, dx in -1:1
        idx = (dx + 1) + 3 * (dy + 1) + 9 * (dz + 1) + 1
        kval = stencil_k27[idx]
        kval == 0.0 && continue
        for n3 in 1:Mz
            m3 = mod(n3 - 1 + dz, Mz) + 1
            for n2 in 1:My
                m2 = mod(n2 - 1 + dy, My) + 1
                for n1 in 1:Mx
                    m1 = mod(n1 - 1 + dx, Mx) + 1
                    Phi_out[n1, n2, n3] += kval * Q_grid[m1, m2, m3]
                end
            end
        end
    end
    sum_qp = 0.0
    @inbounds for i in eachindex(Q_grid, Phi_out)
        sum_qp += Q_grid[i] * Phi_out[i]
    end
    return 0.5 * prefactor * sum_qp
end

"""
    gpu_msm_stencil27!(Phi_out, Q_grid, stencil_k27, prefactor; backend=:auto) -> Float64

Applies a 27-point periodic 3D ternary stencil convolution (`d ∈ {-1, 0, +1}³`,
tryte index `(dx+1) + 3*(dy+1) + 9*(dz+1) + 1`) to 3D charge grid `Q_grid`,
writing the convolved potential grid into `Phi_out` and returning `0.5 * prefactor * ∑ Q ⋅ Φ`.
Dispatches to `xbt_gpu_msm_stencil27` when an active GPU bridge exports it,
otherwise runs the verified CPU stencil kernel.
"""
function gpu_msm_stencil27!(
    Phi_out::AbstractArray{Float64, 3},
    Q_grid::AbstractArray{Float64, 3},
    stencil_k27::AbstractVector{<:Real},
    prefactor::Real;
    backend::Symbol = :auto
)::Float64
    size(Phi_out) == size(Q_grid) ||
        throw(DimensionMismatch("Phi_out size $(size(Phi_out)) must match Q_grid size $(size(Q_grid))"))
    length(stencil_k27) == 27 ||
        throw(DimensionMismatch("stencil_k27 must have length 27, got $(length(stencil_k27))"))

    Mx, My, Mz = size(Q_grid)
    phi_arr = Phi_out isa Array{Float64, 3} ? Phi_out : Array{Float64, 3}(undef, Mx, My, Mz)
    q_arr = Q_grid isa Array{Float64, 3} ? Q_grid : Array{Float64, 3}(Q_grid)
    k_vec = stencil_k27 isa Vector{Float64} ? stencil_k27 : Float64.(collect(stencil_k27))
    pref = Float64(prefactor)

    if Mx == 0 || My == 0 || Mz == 0
        return 0.0
    end

    used_gpu = false
    u_ret = 0.0
    if backend !== :cpu && Mx == My && My == Mz
        prof = probe_gpu_device(; backend = (backend === :auto || backend === :gpu) ? :auto : backend)
        target = (backend === :auto || backend === :gpu) ? prof.backend : backend
        if target !== :cpu && haskey(_GPU_BACKENDS, target)
            st = _GPU_BACKENDS[target]
            if st.handle != C_NULL && st.sym_msm_stencil27 != C_NULL
                u_ret = GC.@preserve phi_arr q_arr k_vec ccall(
                    st.sym_msm_stencil27, Cdouble,
                    (Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Cdouble, Ptr{Cdouble}),
                    Csize_t(Mx), pointer(q_arr), pointer(k_vec), Cdouble(pref), pointer(phi_arr)
                )
                used_gpu = true
            end
        end
    end

    if !used_gpu
        u_ret = _cpu_msm_stencil27!(phi_arr, q_arr, k_vec, pref)
    end

    if !(Phi_out isa Array{Float64, 3})
        copyto!(Phi_out, phi_arr)
    end
    return u_ret
end

# ==============================================================================
# 6. Batched GPU-Resident Velocity Verlet MD Runner
# ==============================================================================

function _extract_bond_arrays(bonds::Vector{Factors.HarmonicBond}, n::Int)
    nb = length(bonds)
    bi0 = Vector{Int32}(undef, nb)
    bj0 = Vector{Int32}(undef, nb)
    bk = Vector{Float64}(undef, nb)
    br0 = Vector{Float64}(undef, nb)
    @inbounds for idx in 1:nb
        b = bonds[idx]
        (1 <= b.i <= n && 1 <= b.j <= n) ||
            throw(ArgumentError("HarmonicBond atom index ($(b.i), $(b.j)) out of 1:$n"))
        bi0[idx] = Int32(b.i - 1)
        bj0[idx] = Int32(b.j - 1)
        bk[idx] = Float64(b.k)
        br0[idx] = Float64(b.r0)
    end
    return (bi0, bj0, bk, br0)
end

function _extract_angle_arrays(angles::Vector{Factors.HarmonicAngle}, n::Int)
    na = length(angles)
    ai0 = Vector{Int32}(undef, na)
    aj0 = Vector{Int32}(undef, na)
    ak0 = Vector{Int32}(undef, na)
    akth = Vector{Float64}(undef, na)
    ath0 = Vector{Float64}(undef, na)
    @inbounds for idx in 1:na
        a = angles[idx]
        (1 <= a.i <= n && 1 <= a.j <= n && 1 <= a.k <= n) ||
            throw(ArgumentError("HarmonicAngle atom index ($(a.i), $(a.j), $(a.k)) out of 1:$n"))
        ai0[idx] = Int32(a.i - 1)
        aj0[idx] = Int32(a.j - 1)
        ak0[idx] = Int32(a.k - 1)
        akth[idx] = Float64(a.ktheta)
        ath0[idx] = Float64(a.theta0)
    end
    return (ai0, aj0, ak0, akth, ath0)
end

function _cpu_batched_verlet_fallback!(
    q::Matrix{Float64},
    p::Matrix{Float64},
    inv_mass::Vector{Float64},
    box::Types.AbstractBox,
    fac::TriumXBT.TriumTiledNonbondedFactor,
    bonds::Vector{Factors.HarmonicBond},
    angles::Vector{Factors.HarmonicAngle},
    dt::Float64,
    n_steps::Int,
    F_out::Union{Nothing, AbstractMatrix{Float64}}
)
    n = length(inv_mass)
    masses = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        masses[j] = 1.0 / inv_mass[j]
    end
    all_factors = Factors.AbstractFactor[]
    sizehint!(all_factors, length(bonds) + length(angles) + 1)
    append!(all_factors, bonds)
    append!(all_factors, angles)
    push!(all_factors, fac)

    hs = Dynamics.HarmonicSystem(all_factors, masses, Factors.Context(box, Types.REDUCED, n))
    F_buf = F_out isa Matrix{Float64} ? F_out : zeros(Float64, 3, n)
    pe = Dynamics.forces!(hs, q, F_buf)
    for _ in 1:n_steps
        pe = Dynamics.velocity_verlet_step!(q, p, F_buf, hs, dt)
    end
    ke = Dynamics.kinetic_energy(hs, p)
    if F_out !== nothing && !(F_out isa Matrix{Float64})
        copyto!(F_out, F_buf)
    end
    return (total_energy = pe + ke, potential_energy = pe, kinetic_energy = ke, used_gpu = false)
end

"""
    run_batched_gpu_steps!(q, p, inv_mass, box, fac, bonds, angles, dt, n_steps;
                           F_out=nothing, backend=:auto)
        -> NamedTuple{(:total_energy, :potential_energy, :kinetic_energy, :used_gpu)}

Executes `n_steps` of Velocity Verlet integration in-place on `(q, p)` for a system
consisting of a `TriumTiledNonbondedFactor` plus `HarmonicBond` and `HarmonicAngle` factors.
When a GPU bridge exporting `xbt_gpu_md_batch_run` is active and `backend !== :cpu`,
all `n_steps` execute GPU-resident without intermediate CPU-GPU roundtrips; otherwise
falls back cleanly to CPU `Dynamics.velocity_verlet_step!`.
"""
function run_batched_gpu_steps!(
    q::Matrix{Float64},
    p::Matrix{Float64},
    inv_mass::Vector{Float64},
    box::Types.AbstractBox,
    fac::TriumXBT.TriumTiledNonbondedFactor,
    bonds::Vector{Factors.HarmonicBond},
    angles::Vector{Factors.HarmonicAngle},
    dt::Float64,
    n_steps::Int;
    F_out::Union{Nothing, AbstractMatrix{Float64}} = nothing,
    backend::Symbol = :auto
)
    backend in (:auto, :cpu, :gpu, :metal, :vulkan, :sycl) ||
        throw(ArgumentError("unsupported backend $(repr(backend))"))
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative, got $n_steps"))
    (isfinite(dt) && dt != 0.0) || throw(ArgumentError("dt must be finite and non-zero, got $dt"))
    Factors.force_consistent(fac) ||
        error("refusing batched MD with non-force-consistent nonbonded factor (lj_mode=$(repr(fac.lj_mode)))")

    n = length(fac.charges)
    size(q) == (3, n) || throw(ArgumentError("q must have shape (3, $n), got $(size(q))"))
    size(p) == (3, n) || throw(ArgumentError("p must have shape (3, $n), got $(size(p))"))
    length(inv_mass) == n || throw(ArgumentError("inv_mass must have length $n, got $(length(inv_mass))"))
    all(x -> isfinite(x) && x > 0.0, inv_mass) ||
        throw(ArgumentError("inv_mass values must be positive and finite"))
    if F_out !== nothing
        size(F_out) == (3, n) || throw(ArgumentError("F_out must have shape (3, $n), got $(size(F_out))"))
    end

    if box isa Types.OrthorhombicBox
        2.0 * fac.rcut < minimum(box.L) ||
            throw(ArgumentError("2 * rcut must be less than minimum box length"))
    elseif box isa Types.TriclinicBox
        2.0 * fac.rcut < minimum(box.widths) ||
            throw(ArgumentError("2 * rcut must be less than minimum box width"))
    end

    bi0, bj0, bk, br0 = _extract_bond_arrays(bonds, n)
    ai0, aj0, ak0, akth, ath0 = _extract_angle_arrays(angles, n)

    if backend !== :cpu && n_steps >= 1 && !fac.use_carry_free &&
       (fac.alpha > 0.0 || fac.coulomb_prefactor == 0.0 || all(iszero, fac.charges))
        prof = probe_gpu_device(; backend = (backend === :auto || backend === :gpu) ? :auto : backend)
        target = (backend === :auto || backend === :gpu) ? prof.backend : backend
        if target !== :cpu && has_gpu_md_batch(target)
            tc = fac.cache isa Neighbors.TilePairCache ?
                 fac.cache :
                 Neighbors.TilePairCache(q, box, fac.rcut, max(0.15, 0.1 * fac.rcut))
            Neighbors.refresh_tile_cache!(tc, q, box)
            if n > 0 && !isempty(tc.tile_pairs_i)
                st = _GPU_BACKENDS[target]
                scratch = TriumXBT._acquire_scratch()
                try
                    is_tri, H_mat, Hinv_mat = TriumXBT._prepare_box_matrices!(scratch, box)
                    H_ptr = H_mat === nothing ? Ptr{Cdouble}(C_NULL) : pointer(H_mat)
                    Hinv_ptr = Hinv_mat === nothing ? Ptr{Cdouble}(C_NULL) : pointer(Hinv_mat)
                    plan = GC.@preserve H_mat Hinv_mat _get_or_create_gpu_plan!(
                        target, q, H_ptr, Hinv_ptr, is_tri, fac, tc
                    )
                    if plan !== nothing
                        nb = length(bonds)
                        na = length(angles)

                        bi0_ptr = nb == 0 ? Ptr{Int32}(C_NULL) : pointer(bi0)
                        bj0_ptr = nb == 0 ? Ptr{Int32}(C_NULL) : pointer(bj0)
                        bk_ptr  = nb == 0 ? Ptr{Cdouble}(C_NULL) : pointer(bk)
                        br0_ptr = nb == 0 ? Ptr{Cdouble}(C_NULL) : pointer(br0)

                        ai0_ptr  = na == 0 ? Ptr{Int32}(C_NULL) : pointer(ai0)
                        aj0_ptr  = na == 0 ? Ptr{Int32}(C_NULL) : pointer(aj0)
                        ak0_ptr  = na == 0 ? Ptr{Int32}(C_NULL) : pointer(ak0)
                        akth_ptr = na == 0 ? Ptr{Cdouble}(C_NULL) : pointer(akth)
                        ath0_ptr = na == 0 ? Ptr{Cdouble}(C_NULL) : pointer(ath0)

                        f_buf = F_out === nothing ? nothing :
                                (F_out isa Matrix{Float64} ? F_out : Matrix{Float64}(undef, 3, n))
                        f_ptr = f_buf === nothing ? Ptr{Cdouble}(C_NULL) : pointer(f_buf)
                        pe_ref = Ref{Cdouble}(0.0)
                        ke_ref = Ref{Cdouble}(0.0)

                        E_raw = GC.@preserve plan q p inv_mass H_mat Hinv_mat bi0 bj0 bk br0 ai0 aj0 ak0 akth ath0 f_buf pe_ref ke_ref ccall(
                            st.md_batch_run, Cdouble,
                            (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint,
                             Csize_t, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble},
                             Csize_t, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble},
                             Cdouble, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
                            plan.ptr, pointer(q), pointer(p), pointer(inv_mass), H_ptr, Hinv_ptr, is_tri,
                            Csize_t(nb), bi0_ptr, bj0_ptr, bk_ptr, br0_ptr,
                            Csize_t(na), ai0_ptr, aj0_ptr, ak0_ptr, akth_ptr, ath0_ptr,
                            Cdouble(dt), Cint(n_steps), f_ptr, pe_ref, ke_ref
                        )

                        if F_out !== nothing && !(F_out isa Matrix{Float64})
                            copyto!(F_out, f_buf)
                        end

                        u_self = _compute_self_energy(fac)
                        pe = Float64(pe_ref[]) + u_self
                        ke = Float64(ke_ref[])
                        E = Float64(E_raw) + u_self
                        return (total_energy = E, potential_energy = pe, kinetic_energy = ke, used_gpu = true)
                    end
                finally
                    TriumXBT._release_scratch(scratch)
                end
            end
        end
    end

    return _cpu_batched_verlet_fallback!(q, p, inv_mass, box, fac, bonds, angles, dt, n_steps, F_out)
end

end # module TriumGPU
