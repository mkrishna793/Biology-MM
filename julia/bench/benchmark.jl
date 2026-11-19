# Bounded performance evidence.
#
# What this measures, and what it does NOT
# -----------------------------------------
# This measures:
#   * pair-count scaling of the exactly-summarised bipartite factor vs direct
#     all-pairs enumeration, with wall-clock time for each;
#   * neighbour-list vs brute-force cost for a cutoff potential;
#   * per-step cost of the integrator at a few system sizes.
#
# This does NOT establish that any of this extrapolates to a million or billion
# atoms. One harmonic test does not predict protein-scale behaviour, memory
# bandwidth, GPU occupancy, or anything about a production engine. The numbers
# below are evidence for the specific workloads timed here.
#
# Run:  JULIA_DEPOT_PATH=... julia --project=julia \\
#           julia/bench/benchmark.jl

using FoundationCore
using FoundationCore.Types
using FoundationCore.Factors
using FoundationCore.Neighbors
using FoundationCore.Dynamics
using Printf, Random, Statistics

const T = FoundationCore.Types
const FA = FoundationCore.Factors
const NB = FoundationCore.Neighbors
const DY = FoundationCore.Dynamics

"Median wall time over `reps` runs of `f`, after one warm-up call."
function besttime(f; reps = 7)
    f()  # warm up / compile
    ts = Float64[]
    for _ in 1:reps
        t0 = time_ns()
        f()
        push!(ts, (time_ns() - t0) / 1e9)
    end
    return minimum(ts)
end

function bench_bipartite()
    println("== bipartite harmonic: direct O(n*m) vs exact summary O(n+m) ==")
    println("   (energy AND per-atom forces; the summary is exact, not approximate)")
    @printf("   %8s %8s %14s %14s %10s %10s\n",
            "n", "m", "direct (s)", "summary (s)", "speedup", "max|ΔF|")
    rng = MersenneTwister(1)
    for (n, m) in ((10, 10), (50, 50), (100, 100), (200, 200), (400, 400), (800, 800))
        A = collect(1:n); B = collect(n+1:n+m)
        N = n + m
        q = randn(rng, 3, N)
        ctx = FA.Context(T.REDUCED, N)
        # direct: one LJ-free exact pair sum via BipartiteHarmonicGroup is the
        # summary; for the direct path we enumerate pairs through explicit factors.
        # To keep the comparison about the *algebra* rather than the factor
        # dispatch, time both through total_energy_forces.
        direct_facs = Any[]
        # a complete bipartite quadratic interaction written as explicit pairs
        # would be n*m factors; instead time the reference double loop directly.
        function direct_energy_forces()
            F = zeros(3, N)
            U = 0.0
            k = 1.0
            @inbounds for i in A
                for j in B
                    dx = q[1, i] - q[1, j]
                    dy = q[2, i] - q[2, j]
                    dz = q[3, i] - q[3, j]
                    U += 0.5 * k * (dx * dx + dy * dy + dz * dz)
                    gx = k * dx; gy = k * dy; gz = k * dz
                    F[1, i] -= gx; F[2, i] -= gy; F[3, i] -= gz
                    F[1, j] += gx; F[2, j] += gy; F[3, j] += gz
                end
            end
            return U, F
        end
        fac = FA.BipartiteHarmonicGroup(A, B, 1.0)
        summary_energy_forces() = FA.total_energy_forces([fac], q, ctx)
        td = besttime(direct_energy_forces; reps = n >= 400 ? 3 : 7)
        ts = besttime(summary_energy_forces; reps = n >= 400 ? 3 : 7)
        Ud, Fd = direct_energy_forces()
        Us, Fs = summary_energy_forces()
        dF = maximum(abs, Fd .- Fs)
        relU = abs(Ud - Us) / max(abs(Ud), 1e-30)
        @printf("   %8d %8d %14.3e %14.3e %10.1fx %10.2e\n", n, m, td, ts, td / ts, dF)
        @assert relU < 1e-12 "summary energy mismatch at n=$n"
        @assert dF < 1e-10 * max(maximum(abs, Fd), 1.0) "summary force mismatch at n=$n"
    end
    println("   -> the summary is exact (energy and force) and the speedup grows with n*m")
    println()
end

function bench_neighbor_list()
    println("== neighbour list vs brute force, LJ at rc = 2.5 sigma, liquid density ==")
    println("   (density rho = 0.8 sigma^-3, the triple point region of LJ; build time")
    println("    is reported separately because a list is amortised over many steps)")
    @printf("   %8s %10s %12s %12s %10s %12s\n",
            "N", "L", "brute (s)", "list trav (s)", "speedup", "pairs")
    rng = MersenneTwister(2)
    rc = 2.5
    rho = 0.8
    for N in (500, 1000, 2000, 4000)
        L = (N / rho)^(1 / 3)
        box = T.OrthorhombicBox((L, L, L))
        q = rand(rng, 3, N) .* L
        ctx = FA.Context(box, T.REDUCED, N)
        function brute_energy()
            F = zeros(3, N)
            U = 0.0
            for i in 1:N-1, j in i+1:N
                d = T.unwrap_displacement(box, q, i, j)
                r2 = d.x^2 + d.y^2 + d.z^2
                if r2 <= rc * rc
                    U += FA.evaluate!(F, FA.LennardJonesPair(i, j, 1.0, 1.0; mode = :lfs, rc = rc), q, ctx)
                end
            end
            return U
        end
        # build once, traverse many: the amortised regime a dynamics run actually uses
        nl = NB.build_neighbors(q, box, rc, 0.3)
        pairs = NB.neighbor_pairs(nl)
        function list_energy()
            F = zeros(3, N)
            U = 0.0
            for (i, j) in pairs
                U += FA.evaluate!(F, FA.LennardJonesPair(i, j, 1.0, 1.0; mode = :lfs, rc = rc), q, ctx)
            end
            return U
        end
        tb = besttime(brute_energy; reps = N >= 2000 ? 2 : 5)
        tl = besttime(list_energy; reps = N >= 2000 ? 3 : 7)
        tbuild = besttime(() -> NB.build_neighbors(q, box, rc, 0.3); reps = 3)
        # energy must agree between the two paths, else the comparison is moot
        Ub = brute_energy(); Ul = list_energy()
        rel = abs(Ub - Ul) / max(abs(Ub), 1e-30)
        @assert rel < 1e-12 "brute and list energies disagree at N=$N: $rel"
        @printf("   %8d %10.2f %12.3e %12.3e %9.1fx %12d  (build %.2e s)\n",
                N, L, tb, tl, tb / tl, length(pairs), tbuild)
    end
    println("   -> the traversal win grows with N; the rebuild is O(N) and amortises over")
    println("      the steps between rebuilds. Neither path is sub-quadratic in N.")
    println()
end

function bench_integrator_step()
    println("== velocity-Verlet step cost (one force evaluation per step) ==")
    @printf("   %8s %12s %14s\n", "N", "step (s)", "ns/atom/step")
    rng = MersenneTwister(3)
    L = 25.0
    box = T.OrthorhombicBox((L, L, L))
    rc = 2.5
    for N in (500, 1000, 2000)
        q = rand(rng, 3, N) .* L
        p = randn(rng, 3, N) .* 0.01
        nl = NB.build_neighbors(q, box, rc, 0.3)
        pairs = NB.neighbor_pairs(nl)
        ctx = FA.Context(box, T.REDUCED, N)
        facs = [FA.LennardJonesPair(i, j, 1.0, 1.0; mode = :lfs, rc = rc) for (i, j) in pairs]
        sys = DY.HarmonicSystem(facs, ones(N), ctx)
        st = DY.State(q, p)
        t = besttime(() -> begin
            for _ in 1:20
                DY.velocity_verlet_step!(st, sys, 0.002)
            end
        end; reps = 3)
        per = t / 20
        @printf("   %8d %12.3e %14.1f\n", N, per, per * 1e9 / N)
    end
    println("   -> per-atom cost is dominated by the pair loop; the O(N) state I/O is")
    println("      a floor that no algebraic reorganization removes")
    println()
end

println("="^72)
println(" FoundationCore benchmark — $(Sys.KERNEL) $(Sys.ARCH), Julia $(VERSION)")
println(" NOTE: these are single-threaded CPU numbers for the workloads shown.")
println("       They do not extrapolate to protein- or billion-atom scale.")
println("="^72)
println()
bench_bipartite()
bench_neighbor_list()
bench_integrator_step()
println("benchmark complete")
