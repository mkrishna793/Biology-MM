#!/usr/bin/env julia
# StreamedLJ benchmark: one Lennard-Jones energy+force evaluation pass.
#
# What this measures
# ------------------
# A single call to `StreamedLJ.evaluate!` on a reproducible, fixed-density
# random configuration in an orthorhombic periodic box. Each pass rebuilds the
# cell linked list and streams every unordered pair once; the timed work is
# energy + forces + pair counting.
#
# What this does NOT measure or claim
# -----------------------------------
# * This is NOT molecular dynamics, NOT an integrator, NOT polarization, and NOT
#   a production force-consistent potential. The potential is the shifted LJ
#   `:shift` mode (C0 at the cutoff); forces are the derivative of the
#   unshifted LJ form, which is the same inside the cutoff but is not C1 there.
# * The configuration is random, not equilibrated. Large LJ energies/checksums
#   indicate random close pairs, not a physical free energy.
# * No tolerance is asserted. The numbers are measurements on this machine.
#
# Usage
# -----
#     julia --project=julia julia/bench/streamed_lj_10m.jl
#     julia --project=julia julia/bench/streamed_lj_10m.jl --n 1000000
#     julia --project=julia julia/bench/streamed_lj_10m.jl --n 10m --density 0.8 --rcut 2.5
#
# With no `--n`, the 1e3, 1e4, 1e5, and 1e6 levels run in this process, then the
# 10-million-atom level runs in an isolated child process. `--n` runs exactly one
# level in the current process. The child is only started after its own
# preflight; if it is skipped, it allocates no coordinate or force arrays.

using Printf, Random
using FoundationCore

module StreamedLJHost
    using FoundationCore: Types
    include(joinpath(@__DIR__, "..", "src", "StreamedLJ.jl"))
end

const SL = StreamedLJHost.StreamedLJ
const T = FoundationCore.Types
const RP = FoundationCore.ResourceProbe

const DEFAULT_DENSITY = 0.8
const DEFAULT_RCUT = 2.5
const DEFAULT_SKIN = 0.0
const DEFAULT_SEED = 20261112
const SMALL_LEVELS = (1_000, 10_000, 100_000, 1_000_000)
const BIG_N = 10_000_000
const MIN_FREE_BYTES = 1.5e9  # 1.5 GB, conservative

function parse_count(s::AbstractString)
    t = lowercase(replace(strip(s), "_" => ""))
    mult = 1.0
    if endswith(t, "k")
        mult = 1e3
        t = chop(t)
    elseif endswith(t, "m")
        mult = 1e6
        t = chop(t)
    elseif endswith(t, "b")
        mult = 1e9
        t = chop(t)
    end
    f = tryparse(Float64, t)
    (f !== nothing && isfinite(f) && f > 0) ||
        error("invalid count '$s'; use an integer, scientific notation, or k/m/b suffix")
    n = round(Int, f * mult)
    n > 0 || error("count '$s' rounds to zero")
    return n
end

function parse_args(args)
    n_arg = nothing
    density = DEFAULT_DENSITY
    rcut = DEFAULT_RCUT
    skin = DEFAULT_SKIN
    seed = DEFAULT_SEED
    skip_isolated = false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--n"
            i == length(args) && error("--n needs a value")
            n_arg = parse_count(args[i + 1])
            i += 2
        elseif startswith(a, "--n=")
            n_arg = parse_count(split(a, "="; limit = 2)[2])
            i += 1
        elseif a == "--density"
            i == length(args) && error("--density needs a value")
            density = parse(Float64, args[i + 1])
            i += 2
        elseif startswith(a, "--density=")
            density = parse(Float64, split(a, "="; limit = 2)[2])
            i += 1
        elseif a == "--rcut"
            i == length(args) && error("--rcut needs a value")
            rcut = parse(Float64, args[i + 1])
            i += 2
        elseif startswith(a, "--rcut=")
            rcut = parse(Float64, split(a, "="; limit = 2)[2])
            i += 1
        elseif a == "--skin"
            i == length(args) && error("--skin needs a value")
            skin = parse(Float64, args[i + 1])
            i += 2
        elseif startswith(a, "--skin=")
            skin = parse(Float64, split(a, "="; limit = 2)[2])
            i += 1
        elseif a == "--seed"
            i == length(args) && error("--seed needs a value")
            seed = parse(Int, args[i + 1])
            i += 2
        elseif startswith(a, "--seed=")
            seed = parse(Int, split(a, "="; limit = 2)[2])
            i += 1
        elseif a == "--no-isolated"
            skip_isolated = true
            i += 1
        elseif a in ("-h", "--help")
            print_usage()
            exit(0)
        else
            error("unknown argument '$a' (use --help)")
        end
    end
    (density > 0 && isfinite(density)) || error("--density must be finite and > 0")
    (rcut > 0 && isfinite(rcut)) || error("--rcut must be finite and > 0")
    (skin >= 0 && isfinite(skin)) || error("--skin must be finite and >= 0")
    return (; n = n_arg, density, rcut, skin, seed, skip_isolated)
end

function print_usage()
    println("""
Usage: streamed_lj_10m.jl [options]

  --n <count>       run one size in this process (e.g. 1000, 1e6, 10m)
  --density <rho>   number density (default $DEFAULT_DENSITY)
  --rcut <rc>       single LJ cutoff (default $DEFAULT_RCUT)
  --skin <s>        cell-search skin; cutoff stays rcut (default $DEFAULT_SKIN)
  --seed <int>      RNG seed for the random configuration (default $DEFAULT_SEED)
  --no-isolated     with no --n, do not spawn the isolated 10m child
  -h, --help        show this message

The benchmark measures ONE evaluation pass (cell rebuild + energy + forces),
not MD and not polarization. The potential is shifted LJ (:shift, C0), not the
production force-consistent :lfs mode.
""")
end

function print_header(opts)
    println("=" ^ 78)
    println("StreamedLJ benchmark: one evaluation pass (cell rebuild + energy + forces)")
    println("NOT MD, NOT polarization, NOT a production force-consistent potential.")
    println("Potential: shifted LJ (:shift, C0); Lorentz-Berthelot mixing; orthorhombic PBC.")
    println("Units: reduced (sigma=epsilon=1); energies/forces are reduced LJ units.")
    println("Julia $(VERSION); Sys.WORD_SIZE=$(Sys.WORD_SIZE); Threads.nthreads()=$(Threads.nthreads())")
    @printf("defaults: density=%.6g rcut=%.6g skin=%.6g seed=%d\n",
            opts.density, opts.rcut, opts.skin, opts.seed)
    if opts.n === nothing
        println("levels: ", join(SMALL_LEVELS, ", "), ", then isolated ", BIG_N)
    else
        println("single level: ", opts.n)
    end
    println("=" ^ 78)
end

function make_box(n::Int, density::Float64)
    n > 0 || error("n must be positive")
    L = (Float64(n) / density)^(1.0 / 3.0)
    return T.OrthorhombicBox((L, L, L)), L
end

function make_system(n::Int, L::Float64, seed::Int)
    q = Matrix{Float64}(undef, 3, n)
    rng = MersenneTwister(seed)
    rand!(rng, q)
    q .*= L
    epsilon = fill(1.0, n)
    sigma = fill(1.0, n)
    return q, epsilon, sigma
end

function preflight(n::Int, box::T.OrthorhombicBox, rcut::Float64, skin::Float64)
    est = SL.estimate_memory_bytes(n, box, rcut; skin = skin)
    # The 1.5 GB availability floor is the safety gate for the 10m allocation.
    # Smaller levels still require their working set to fit in 25% of available RAM,
    # so a tiny/short benchmark remains possible on a busy machine.
    strict = n > 1_000_000
    pf = RP.memory_preflight(est; min_available_bytes = strict ? ceil(Int, MIN_FREE_BYTES) : 0)
    return (; pf..., strict)
end

function print_preflight(n::Int, L::Float64, rcut::Float64, skin::Float64, pf)
    @printf("n=%d  L=%.6f  density=%.6g  rcut=%.6g  skin=%.6g\n",
            n, L, n / L^3, rcut, skin)
    @printf("  preflight: estimated_working_set=%.1f MiB (includes 30%% overhead)\n",
            pf.est / 2.0^20)
    @printf("             available=%.1f MiB (%s) of total=%.1f MiB; hard limit=%.1f MiB\n",
            pf.available / 2.0^20, string(pf.source), pf.total / 2.0^20, pf.limit / 2.0^20)
    if pf.strict
        @printf("             25%% available=%s reserve(>=256 MiB)=%s available>=1.5 GB=%s -> %s\n",
                pf.fraction_ok ? "yes" : "no", pf.reserve_ok ? "yes" : "no",
                pf.floor_ok ? "yes" : "no",
                pf.ok ? "PASS" : "SKIP")
    else
        @printf("             25%% available=%s reserve(>=256 MiB)=%s; 10m-only floor not applied -> %s\n",
                pf.fraction_ok ? "yes" : "no", pf.reserve_ok ? "yes" : "no",
                pf.ok ? "PASS" : "SKIP")
    end
    return pf.ok
end

function compile_warmup(rcut::Float64, skin::Float64)
    # A small system on the same code path so the 10m wall time excludes JIT.
    L = max(10.0, 2.2 * rcut)
    n = 64
    box = T.OrthorhombicBox((L, L, L))
    q, epsilon, sigma = make_system(n, L, 7)
    ws = SL.LJWorkspace(q, box, rcut; skin = skin)
    SL.evaluate!(ws, q, epsilon, sigma, box; rcut = rcut, skin = skin)
    return nothing
end

function run_level(n::Int, opts)
    box, L = make_box(n, opts.density)
    rss_before = Sys.maxrss()
    pf = preflight(n, box, opts.rcut, opts.skin)
    print_preflight(n, L, opts.rcut, opts.skin, pf) || return false

    # Allocation of the dataset happens only after the preflight passes.
    q, epsilon, sigma = make_system(n, L, opts.seed)
    ws = SL.LJWorkspace(q, box, opts.rcut; skin = opts.skin)

    energy = 0.0
    forces = ws.forces
    npairs = 0
    wall = 0.0
    alloc_bytes = -1

    if n <= 1_000_000
        # Warm up on this size, measure allocations after warm-up, then time one
        # clean pass.
        SL.evaluate!(ws, q, epsilon, sigma, box; rcut = opts.rcut, skin = opts.skin)
        alloc_bytes = Base.@allocated SL.evaluate!(ws, q, epsilon, sigma, box;
                                                   rcut = opts.rcut, skin = opts.skin)
        t0 = time_ns()
        energy, forces, npairs = SL.evaluate!(ws, q, epsilon, sigma, box;
                                              rcut = opts.rcut, skin = opts.skin)
        wall = (time_ns() - t0) / 1e9
    else
        # Compile on a tiny system, then make the single timed pass. Do not run
        # an extra @allocated pass at 10m: the measurement is one evaluation.
        compile_warmup(opts.rcut, opts.skin)
        t0 = time_ns()
        energy, forces, npairs = SL.evaluate!(ws, q, epsilon, sigma, box;
                                              rcut = opts.rcut, skin = opts.skin)
        wall = (time_ns() - t0) / 1e9
    end

    checksum = energy + sum(abs, forces)
    finite = isfinite(energy) && isfinite(checksum) && all(isfinite, forces)
    steps_per_s = inv(wall)
    @printf("  pairs=%d  energy=%.12e  checksum=%.12e\n", npairs, energy, checksum)
    @printf("  wall=%.6f s  throughput=%.6e atom-steps/s  allocations=%s  pairs/atom=%.6f finite=%s\n",
            wall,
            n * steps_per_s,
            alloc_bytes >= 0 ? string(alloc_bytes) * " B" : "not measured (large path)",
            npairs / n, finite ? "yes" : "NO")
    @printf("  process_peak_rss=%.1f MiB (before level=%.1f MiB; Sys.maxrss)\n",
            Sys.maxrss() / 2.0^20, rss_before / 2.0^20)
    return finite
end

function spawn_isolated(n::Int, opts)
    script = @__FILE__
    active = Base.active_project()
    projdir = active === nothing ? normpath(joinpath(@__DIR__, "..")) : dirname(active)
    cmd = `$(Base.julia_cmd()) --project=$projdir --startup-file=no --threads=1 $script --n $n --density $(opts.density) --rcut $(opts.rcut) --skin $(opts.skin) --seed $(opts.seed)`
    println("")
    println("--- isolated process: n=$n ---")
    println("command: ", cmd)
    GC.gc()
    run(cmd)
    return nothing
end

function main()
    opts = parse_args(ARGS)
    print_header(opts)
    if opts.n !== nothing
        run_level(opts.n, opts)
        return 0
    end
    for n in SMALL_LEVELS
        run_level(n, opts)
        println("")
    end
    if opts.skip_isolated
        println("Skipping isolated 10m child (--no-isolated).")
    else
        spawn_isolated(BIG_N, opts)
    end
    return 0
end

exit(main())
