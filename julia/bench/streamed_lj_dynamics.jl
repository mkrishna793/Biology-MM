#!/usr/bin/env julia
# End-to-end streamed Lennard-Jones molecular dynamics benchmark.
#
# What this measures
# ------------------
# A short, bounded trajectory of the *factor* stack used in production:
# `StreamedLJFactor` (mode = :lfs) inside a `Dynamics.HarmonicSystem`, advanced
# by `Dynamics.velocity_verlet_step!`. Every step rebuilds the cell list, streams
# every pair once for energy + forces, and does the Verlet half-kicks and drift.
#
# The staged sizes are N = 1e5, 1e6, 1e7. Initial positions are a deterministic
# simple-cubic lattice with seeded jitter, in an orthorhombic periodic box at
# density 0.8, so runs are reproducible on any machine.
#
# What this is NOT
# ----------------
# * Not equilibrated MD, not a physical free energy, not a diffusion measurement.
#   Three steps from a lattice at rest only exercise the full step path.
# * Not a parallel benchmark: FoundationCore is single-threaded here.
# * No tolerance is asserted. Timings are one machine, one run, short enough to
#   be dominated by cache/LLC behavior at these sizes.
# * Runs are short, unrelaxed probes; no physical observable is inferred.
#
# Safety
# ------
# Before allocating q, p, F, or the factor workspace for a level, the script
# estimates the full Dynamics working set and compares it against available memory.
# If the check fails, the level is skipped with an explicit reason and *nothing*
# large is allocated. The estimate is deliberately conservative: it starts from
# `StreamedLJ.estimate_memory_bytes` and adds the state/workspace arrays that
# StreamedLJ's own estimate does not know about, then applies 30% overhead.
#
# Usage
# -----
#     julia --project=julia julia/bench/streamed_lj_dynamics.jl
#     julia --project=julia julia/bench/streamed_lj_dynamics.jl --n 10000
#     julia --project=julia julia/bench/streamed_lj_dynamics.jl --steps 2
#
# `--n` runs one stage in this process; with no `--n`, all three stages run.

using Printf, Random
using FoundationCore

const T = FoundationCore.Types
const FA = FoundationCore.Factors
const DY = FoundationCore.Dynamics
const SL = FoundationCore.StreamedLJ
const RP = FoundationCore.ResourceProbe

const LEVELS = (100_000, 1_000_000, 10_000_000)
const MAX_N = 10_000_000
const DEFAULT_DENSITY = 0.8
const DEFAULT_RCUT = 2.5
const DEFAULT_SKIN = 0.0
const DEFAULT_SEED = 20261112
const JITTER_FRAC = 0.02
const DEFAULT_DT = 0.002
const DEFAULT_STEPS = 3
const MIB = 2.0^20

function parse_args(args)
    n = nothing
    steps = DEFAULT_STEPS
    threaded = false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--n"
            i == length(args) && error("--n needs a value")
            n = parse(Int, replace(args[i + 1], "_" => ""))
            i += 2
        elseif startswith(a, "--n=")
            n = parse(Int, replace(split(a, "="; limit = 2)[2], "_" => ""))
            i += 1
        elseif a == "--steps"
            i == length(args) && error("--steps needs a value")
            steps = parse(Int, args[i + 1])
            i += 2
        elseif startswith(a, "--steps=")
            steps = parse(Int, split(a, "="; limit = 2)[2])
            i += 1
        elseif a == "--threaded"
            threaded = true
            i += 1
        elseif a in ("-h", "--help")
            print_usage()
            exit(0)
        else
            error("unknown argument '$a' (use --help)")
        end
    end
    n === nothing || (1 <= n <= MAX_N) ||
        error("--n must be in 1:$MAX_N")
    steps >= 1 || error("--steps must be >= 1")
    return (; n, steps, threaded)
end

function print_usage()
    println("""
Usage: streamed_lj_dynamics.jl [options]

  --n <int>      run one stage (1..$MAX_N) instead of all of $(join(LEVELS, ", "))
  --steps <int>  timed velocity-Verlet steps (default $DEFAULT_STEPS)
  --threaded     opt in to deterministic threaded force evaluation; choose worker count with JULIA_NUM_THREADS or --threads=N
  -h, --help     show this message

Fixed for reproducibility: density=$DEFAULT_DENSITY, rcut=$DEFAULT_RCUT,
skin=$DEFAULT_SKIN, seed=$DEFAULT_SEED, mode=:lfs, dt=$DEFAULT_DT.
Exit status is 0 when every attempted stage either ran or was safely skipped.
""")
end

function print_header(opts)
    println("=" ^ 78)
    println("StreamedLJFactor + Dynamics end-to-end trajectory benchmark")
    println("Potential: linear-force-shifted LJ (:lfs), rc=$DEFAULT_RCUT, skin=$DEFAULT_SKIN")
    println("Integrator: velocity Verlet, dt=$DEFAULT_DT, 1 warmup step + $(opts.steps) timed steps")
    println("Box: orthorhombic periodic; targets density=$DEFAULT_DENSITY; seeded lattice + jitter")
    println("Julia $(VERSION); Sys.WORD_SIZE=$(Sys.WORD_SIZE); Threads.nthreads()=$(Threads.nthreads())")
    println("force path: ", opts.threaded ? "threaded deterministic" : "serial")
    println("thread sweep: invoke with --threads=1,2,4,8 and --threaded")
    println("=" ^ 78)
end

"Largest integer whose cube is <= n, then one more if it cannot hold n sites."
function lattice_side(n::Int)
    ns = 1
    while ns^3 < n
        ns += 1
    end
    return ns
end

"""
    make_lattice(n, L, seed) -> (q, ns, spacing)

Place `n` atoms on `n` distinct sites of an `ns^3` simple-cubic lattice inside a
box of side `L`, with `ns = ceil(cbrt(n))`. Sites are sampled with a fixed stride
so that the `ns^3 - n` vacancies are spread uniformly instead of leaving a slab
void. Each coordinate gets a seeded jitter of ±`JITTER_FRAC * spacing`.
"""
function make_lattice(n::Int, L::Float64, seed::Int)
    ns = lattice_side(n)
    total = ns^3
    spacing = L / ns
    a = JITTER_FRAC * spacing
    q = Matrix{Float64}(undef, 3, n)
    rng = MersenneTwister(seed)
    @inbounds for k in 0:n-1
        site = (k * total) ÷ n            # strictly increasing: total >= n
        i = site % ns
        j = (site ÷ ns) % ns
        l = site ÷ (ns * ns)
        q[1, k + 1] = (i + 0.5) * spacing + a * (2.0 * rand(rng) - 1.0)
        q[2, k + 1] = (j + 0.5) * spacing + a * (2.0 * rand(rng) - 1.0)
        q[3, k + 1] = (l + 0.5) * spacing + a * (2.0 * rand(rng) - 1.0)
    end
    return q, ns, spacing
end

"""
Conservative working-set estimate for the Dynamics run, in bytes.

`StreamedLJ.estimate_memory_bytes` accounts for coordinates, force accumulator,
epsilon/sigma, the linked list, and cells (with 30% overhead). A Dynamics run
additionally holds the state's own q/p/F, the factor's copies of epsilon/sigma,
and the workspace gradient accumulators; `80N` bytes/atom covers those, also
with 30% overhead.
"""
function estimate_dynamics_bytes(n::Int, box::T.OrthorhombicBox, rcut::Float64,
                                 skin::Float64)
    builtin = SL.estimate_memory_bytes(n, box, rcut; skin = skin)
    extra = (2 * 3 + 2 + 2) * 8 * n
    return builtin + ceil(Int, 1.30 * extra), builtin
end

"""
    preflight(n, box, rcut, skin) -> NamedTuple

Decide, using only scalar arithmetic, whether the level may allocate. No q/p/F
or workspace array exists yet when this runs.
"""
function preflight(n::Int, box::T.OrthorhombicBox, rcut::Float64, skin::Float64)
    est, builtin = estimate_dynamics_bytes(n, box, rcut, skin)
    pf = RP.memory_preflight(est; total_fraction = true)
    return (; pf..., builtin, ok = n <= MAX_N && pf.ok)
end

function print_preflight(n::Int, L::Float64, pf; density::Float64,
                         rcut::Float64, skin::Float64)
    @printf("  geometry: L=%.6f density=%.6f rcut=%.3f skin=%.3f (rc < L/2: %s)\n",
            L, n / L^3, rcut, skin, rcut < L / 2 ? "yes" : "no")
    @printf("  preflight: est=%.2f MiB (StreamedLJ base %.2f MiB)\n",
            pf.est / MIB, pf.builtin / MIB)
    @printf("             available=%.1f MiB (%s) / total=%.1f MiB; hard limit=%.1f MiB\n",
            pf.available / MIB, string(pf.source), pf.total / MIB, pf.limit / MIB)
    @printf("             gates: 25%%available=%s reserve(>=%.0f MiB)=%s 25%%total=%s nsites=%s\n",
            pf.fraction_ok ? "ok" : "NO", pf.reserve / MIB,
            pf.reserve_ok ? "ok" : "NO", pf.total_ok ? "ok" : "NO",
            n <= MAX_N ? "ok" : "NO")
    if !pf.ok
        println("  SKIP: memory preflight failed; no q/p/workspace allocated for n=$n.")
    end
    return pf.ok
end

function run_level(n::Int, opts)
    L = (n / DEFAULT_DENSITY)^(1.0 / 3.0)
    box = T.OrthorhombicBox((L, L, L))

    println("")
    @printf("--- n=%d ---\n", n)
    if !(DEFAULT_RCUT < L / 2)
        @printf("  SKIP: rcut=%.3f is not < min(L)/2=%.6f at this density.\n",
                DEFAULT_RCUT, L / 2)
        return false
    end

    rss_before = Sys.maxrss()
    pf = preflight(n, box, DEFAULT_RCUT, DEFAULT_SKIN)
    print_preflight(n, L, pf; density = DEFAULT_DENSITY, rcut = DEFAULT_RCUT,
                    skin = DEFAULT_SKIN) || return false

    # --- allocation starts here, only after the preflight passed -------------
    seed = DEFAULT_SEED + n
    q, ns, spacing = make_lattice(n, L, seed)
    epsilon = ones(Float64, n)
    sigma = ones(Float64, n)

    fac = SL.StreamedLJFactor(q, epsilon, sigma, box;
                              rcut = DEFAULT_RCUT, skin = DEFAULT_SKIN,
                              mode = :lfs, threaded = opts.threaded)
    ctx = FA.Context(box, T.REDUCED, n)
    sys = DY.HarmonicSystem([fac], ones(Float64, n), ctx)
    st = DY.State(q, zeros(Float64, 3, n))

    # One untimed step: validates the setup, proves the force model is
    # conservative, and removes first-call compilation from the measurement.
    DY.velocity_verlet_step!(st, sys, DEFAULT_DT)
    DY.force_model_supported(sys) ||
        error("StreamedLJFactor(:lfs) must be force-consistent before timing")

    alloc_step = Base.@allocated DY.velocity_verlet_step!(st, sys, DEFAULT_DT)

    GC.gc()
    t0 = time_ns()
    for _ in 1:opts.steps
        DY.velocity_verlet_step!(st, sys, DEFAULT_DT)
    end
    wall = (time_ns() - t0) / 1e9

    U = DY.potential_energy(sys, st.q)
    K = DY.kinetic_energy(sys, st.p)
    finite = all(isfinite, st.q) && all(isfinite, st.p) && all(isfinite, st.F) &&
             isfinite(U) && isfinite(K)
    E = U + K
    steps_per_s = opts.steps / wall
    atom_steps_per_s = n * steps_per_s
    rss_after = Sys.maxrss()

    @printf("  lattice: %d^3 sites, spacing=%.6f, jitter=±%.6f; used %d sites\n",
            ns, spacing, JITTER_FRAC * spacing, n)
    @printf("  state: U=%.12e K=%.12e E=%.12e finite=%s\n", U, K, E, finite)
    @printf("  time: %.6f s for %d steps (%.3f ms/step)\n",
            wall, opts.steps, 1e3 * wall / opts.steps)
    @printf("  throughput: %.3f steps/s = %.6e atom-steps/s (N=%d)\n",
            steps_per_s, atom_steps_per_s, n)
    @printf("  allocations: %d B/step after 1 warmup step\n", alloc_step)
    @printf("  rss: peak=%.1f MiB (level start=%.1f MiB; Sys.maxrss, process-wide)\n",
            rss_after / MIB, rss_before / MIB)
    finite || println("  WARNING: non-finite state; this run is not a valid measurement.")
    return finite
end

function main()
    opts = parse_args(ARGS)
    print_header(opts)
    levels = opts.n === nothing ? LEVELS : (opts.n,)
    results = Bool[]
    for n in levels
        push!(results, run_level(n, opts))
    end
    println("")
    @printf("summary: %d/%d stages ran to completion; others skipped safely or non-finite\n",
            count(identity, results), length(results))
    return 0
end

exit(main())
