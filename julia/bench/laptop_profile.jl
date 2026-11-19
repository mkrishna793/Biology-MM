#!/usr/bin/env julia
# Step-by-step CPU profiler, memory budget, and sustained MD trajectory benchmark.
#
# What this measures (MEASURED) and what it projects (EXTRAPOLATED)
# ------------------------------------------------------------------
# Measured on this host (single-process CPU):
#   * Per-stage CPU breakdown on solvated TIP3P water boxes (N = 768..5184 atoms):
#       - Neighbor list build & skin-check reuse (`Neighbors.build_neighbors` / `needs_rebuild!`)
#       - Streamed LJ (`StreamedLJ.evaluate!` with `LJWorkspace`)
#       - PME stages (`_prepare_stencils!`, spread+FFT+brfft `_reciprocal_potential!`,
#         gather + real-space via `PME.pme_energy_forces!`)
#       - Constraints (`Constraints.project_positions!` + `project_velocities!` with `ProjectionWorkspace`)
#       - Velocity-Verlet integration (`O(N)` kicks + drift)
#   * Sustained trajectory wall time, ms/step, ns/day (at dt = 2 fs), per-step
#     allocations, and process peak RSS (`ResourceProbe.peak_rss_bytes()`).
#   * Deterministic memory budget (`ResourceProbe.md_memory_budget`) compared
#     against `Base.summarysize` of the live workspaces.
#
# Extrapolated (explicitly labeled EXTRAPOLATED):
#   * Linear O(N) projections for N = 10k, 50k, 100k, 1M atoms from the largest
#     measured system (N = 5184), paired with exact `md_memory_budget` and
#     `ResourceProbe.memory_preflight` feasibility on this machine.
#
# Usage:
#   JULIA_DEPOT_PATH=... julia --project=. bench/laptop_profile.jl

using Printf, Random, Statistics
using FoundationCore

const T = FoundationCore.Types
const NB = FoundationCore.Neighbors
const SL = FoundationCore.StreamedLJ
const PME = FoundationCore.PME
const CS = FoundationCore.Constraints
const WB = FoundationCore.WaterBox
const RP = FoundationCore.ResourceProbe

const DT_PS = 0.002       # 2 fs timestep
const DT_NS = DT_PS * 1e-3
const MIB = 2.0^20

function min_time(f::F; reps::Int = 5) where {F}
    f()
    f()
    best = Inf
    for _ in 1:reps
        t0 = time_ns()
        f()
        dt = (time_ns() - t0) / 1e9
        dt < best && (best = dt)
    end
    return best
end

function alloc_bytes(f::F) where {F}
    f()
    return @allocated f()
end

function setup_water_system(nmol::Int; seed::Int = 1)
    w = WB.tip3p_water(nmol; density = 1.0, seed = seed)
    N = size(w.positions, 2)
    L = w.box.L
    rc_lj = WB.default_rcut(L)
    rc_pme = WB.default_pme_rcut(L)
    alpha = WB.PME_ALPHA_TARGET / rc_pme
    grid = WB.default_grid(L, 4; alpha = alpha)

    q = copy(w.positions)
    p = zeros(Float64, 3, N)
    rng = MersenneTwister(seed + 100)
    for j in 1:N, d in 1:3
        p[d, j] = sqrt(w.masses[j] * WB.KB * 300.0) * randn(rng)
    end
    CS.project_velocities!(q, p, w.masses, w.constraints; tolerance = 1e-10)

    lj_ws = SL.LJWorkspace(q, w.box, rc_lj; skin = 0.0)
    pme_plan = PME.PMEPlan(q, collect(L); alpha = alpha, rcut = rc_pme,
                           grid = grid, order = 4, prefactor = WB.COULOMB_PREFACTOR)
    cs_ws = CS.ProjectionWorkspace(w.constraints)
    skin = min(0.15, 0.5 * (0.49 * minimum(L) - rc_lj))
    nl = NB.build_neighbors(q, w.box, rc_lj, skin)
    F = zeros(3, N)
    F_pme = zeros(3, N)
    q_ref = copy(q)

    return (; w, N, L, rc_lj, rc_pme, alpha, grid, skin,
            q, p, F, F_pme, q_ref, lj_ws, pme_plan, cs_ws, nl)
end

@inline function verlet_kick!(p::Matrix{Float64}, F::Matrix{Float64}, half_dt::Float64)
    @inbounds for i in eachindex(p)
        p[i] += half_dt * F[i]
    end
end

@inline function verlet_drift!(q::Matrix{Float64}, p::Matrix{Float64},
                               m::Vector{Float64}, dt::Float64)
    n = size(q, 2)
    @inbounds for j in 1:n
        inv_m = dt / m[j]
        q[1, j] += inv_m * p[1, j]
        q[2, j] += inv_m * p[2, j]
        q[3, j] += inv_m * p[3, j]
    end
end

function profile_breakdown(nmol::Int)
    sys = setup_water_system(nmol)
    w = sys.w
    N = sys.N
    q = sys.q
    p = sys.p
    F = sys.F
    F_pme = sys.F_pme
    lj_ws = sys.lj_ws
    plan = sys.pme_plan
    cs_ws = sys.cs_ws
    reps = N <= 2000 ? 7 : 4

    # 1. Neighbor build & skin check
    t_nb_build = min_time(() -> NB.build_neighbors(q, w.box, sys.rc_lj, sys.skin); reps = 3)
    t_nb_check = min_time(() -> NB.needs_rebuild!(sys.nl, q, w.box); reps = reps)
    a_nb_check = alloc_bytes(() -> NB.needs_rebuild!(sys.nl, q, w.box))

    # 2. Streamed LJ
    eval_lj!() = SL.evaluate!(lj_ws, q, w.epsilon, w.sigma, w.box;
                              rcut = sys.rc_lj, mode = :lfs, topology = w.lj_topology)
    t_lj = min_time(eval_lj!; reps = reps)
    a_lj = alloc_bytes(eval_lj!)

    # 3. PME breakdown
    eval_pme!() = PME.pme_energy_forces!(F_pme, plan, q, w.charges;
                                         corrections = w.pme_corrections)
    t_pme_total = min_time(eval_pme!; reps = reps)
    a_pme = alloc_bytes(eval_pme!)

    copyto!(plan.xbuf, q)
    copyto!(plan.qqbuf, w.charges)
    t_pme_stencils = min_time(() -> PME._prepare_stencils!(plan, plan.xbuf); reps = reps)
    t_pme_recip_fft = min_time(() -> PME._reciprocal_potential!(plan, plan.qqbuf, nothing); reps = reps)
    t_pme_real_gather = max(0.0, t_pme_total - t_pme_stencils - t_pme_recip_fft)

    # 4. Constraints (RATTLE position + velocity projection on perturbed step)
    q_tmp = copy(q)
    p_tmp = copy(p)
    q_ref = copy(q)
    function step_constraints!()
        copyto!(q_tmp, q_ref)
        verlet_drift!(q_tmp, p_tmp, w.masses, DT_PS)
        CS.project_positions!(q_tmp, w.masses, w.constraints; tolerance = 1e-10)
        CS.project_velocities!(q_tmp, p_tmp, w.masses, w.constraints; tolerance = 1e-10)
    end
    t_cs = min_time(step_constraints!; reps = reps)
    a_cs = alloc_bytes(step_constraints!)

    # 5. Integration (O(N) kicks + drift + force accumulate)
    function step_integ!()
        verlet_kick!(p_tmp, F, 0.5 * DT_PS)
        verlet_drift!(q_tmp, p_tmp, w.masses, DT_PS)
        @. F = lj_ws.forces + F_pme
        verlet_kick!(p_tmp, F, 0.5 * DT_PS)
    end
    t_int = min_time(step_integ!; reps = reps)
    a_int = alloc_bytes(step_integ!)

    # Live summarysize vs ResourceProbe budget
    nl_pairs = length(NB.neighbor_pairs(sys.nl))
    live_bytes = Base.summarysize(lj_ws) + Base.summarysize(plan) +
                 Base.summarysize(cs_ws) + Base.summarysize(sys.nl) +
                 sizeof(q) + sizeof(p) + sizeof(F) + sizeof(F_pme)
    budget = RP.md_memory_budget(atoms = N,
                                 neighbor_capacity = nl_pairs,
                                 dims = sys.grid,
                                 order = 4,
                                 lj_cells = length(lj_ws.head),
                                 pme_cells = prod(plan.cell_n),
                                 constraints = length(w.constraints.components) * 3,
                                 overhead = 0.25)

    return (; nmol, N, grid = sys.grid, nl_pairs,
            t_nb_build, t_nb_check, a_nb_check,
            t_lj, a_lj,
            t_pme_total, a_pme, t_pme_stencils, t_pme_recip_fft, t_pme_real_gather,
            t_cs, a_cs,
            t_int, a_int,
            live_bytes, budget)
end

function run_sustained_trajectory(nmol::Int, nsteps::Int)
    sys = setup_water_system(nmol; seed = 7)
    w = sys.w
    q = sys.q
    p = sys.p
    F = sys.F
    F_pme = sys.F_pme
    q_ref = sys.q_ref
    lj_ws = sys.lj_ws
    plan = sys.pme_plan
    cs_ws = sys.cs_ws
    nl = sys.nl
    m = w.masses
    cs = w.constraints
    eps = w.epsilon
    sig = w.sigma
    box = w.box
    rc_lj = sys.rc_lj
    skin = sys.skin
    topo = w.lj_topology
    charges = w.charges
    corr = w.pme_corrections
    half_dt = 0.5 * DT_PS

    function total_forces!()
        _, flj, _ = SL.evaluate!(lj_ws, q, eps, sig, box;
                                 rcut = rc_lj, mode = :lfs, topology = topo)
        PME.pme_energy_forces!(F_pme, plan, q, charges; corrections = corr)
        @inbounds for i in eachindex(F)
            F[i] = flj[i] + F_pme[i]
        end
    end

    function one_step!(rebuilds::Base.RefValue{Int})
        if NB.needs_rebuild!(nl, q, box)
            nl = NB.build_neighbors(q, box, rc_lj, skin)
            rebuilds[] += 1
        end
        verlet_kick!(p, F, half_dt)
        verlet_drift!(q, p, m, DT_PS)
        CS.project_positions!(q, m, cs; tolerance = 1e-10)
        total_forces!()
        verlet_kick!(p, F, half_dt)
        CS.project_velocities!(q, p, m, cs; tolerance = 1e-10)
    end

    total_forces!()
    rebuilds = Ref(0)
    # Warm-up 2 steps
    one_step!(rebuilds)
    one_step!(rebuilds)
    rebuilds[] = 0

    GC.gc(false)
    t0 = time_ns()
    bytes_alloc = @allocated begin
        for _ in 1:nsteps
            one_step!(rebuilds)
        end
    end
    wall_s = (time_ns() - t0) / 1e9
    ms_per_step = 1000.0 * wall_s / nsteps
    ns_per_day = (nsteps * DT_NS) / wall_s * 86400.0
    return (; nmol, N = sys.N, nsteps, wall_s, ms_per_step, ns_per_day,
            alloc_per_step = bytes_alloc / nsteps,
            rebuilds = rebuilds[],
            peak_rss_mib = RP.peak_rss_bytes() / MIB)
end

function main()
    avail = RP.available_memory()
    println("=" ^ 84)
    println(" FoundationCore Laptop Performance & Memory Profile (MEASURED + EXTRAPOLATED)")
    println(" Host: $(Sys.KERNEL) $(Sys.ARCH), Julia $(VERSION), CPU threads=$(Sys.CPU_THREADS), active=$(Threads.nthreads())")
    @printf(" RAM: total=%.2f GiB, available=%.2f GiB (%s), initial peak RSS=%.1f MiB\n",
            Sys.total_memory() / 2^30, avail.available / 2^30, string(avail.source),
            RP.peak_rss_bytes() / MIB)
    println(" Workload: TIP3P solvated water (StreamedLJ :lfs + PME order=4 + RATTLE + Verlet, dt=2 fs)")
    println("=" ^ 84)
    println()

    # 1. Step-by-step CPU breakdown
    println("[MEASURED] Step-by-step CPU breakdown (ms/step and share):")
    @printf("  %6s %11s | %7s %7s | %7s | %22s | %7s | %7s | %8s\n",
            "N", "PME grid", "NBbld", "NBchk", "LJ",
            "PME(st/fft/g+r = tot)", "Constr", "Integ", "Sum(ms)")
    profiles = []
    for nmol in (256, 512, 1024, 1728)
        pr = profile_breakdown(nmol)
        push!(profiles, pr)
        pme_str = @sprintf("%.2f/%.2f/%.2f=%.2f",
                           pr.t_pme_stencils * 1e3,
                           pr.t_pme_recip_fft * 1e3,
                           pr.t_pme_real_gather * 1e3,
                           pr.t_pme_total * 1e3)
        tot_ms = (pr.t_nb_check + pr.t_lj + pr.t_pme_total + pr.t_cs + pr.t_int) * 1e3
        @printf("  %6d %11s | %7.3f %7.4f | %7.2f | %22s | %7.3f | %7.4f | %8.2f\n",
                pr.N, string(pr.grid),
                pr.t_nb_build * 1e3, pr.t_nb_check * 1e3,
                pr.t_lj * 1e3, pme_str,
                pr.t_cs * 1e3, pr.t_int * 1e3, tot_ms)
    end
    println("  Warm allocations (B/call) at N=$(profiles[end].N): ",
            "NBchk=$(profiles[end].a_nb_check), LJ=$(profiles[end].a_lj), ",
            "PME=$(profiles[end].a_pme), Constr=$(profiles[end].a_cs), Integ=$(profiles[end].a_int)")
    println()

    # 2. Sustained trajectory timings
    println("[MEASURED] Sustained trajectory (dt = 2 fs, full LJ+PME+RATTLE+Verlet+NB check):")
    @printf("  %6s %6s %9s %10s %10s %12s %9s %10s\n",
            "N", "steps", "wall(s)", "ms/step", "ns/day", "alloc/step", "rebuilds", "peakRSS")
    runs = []
    for (nmol, nsteps) in ((256, 200), (512, 100), (1024, 50), (1728, 30))
        r = run_sustained_trajectory(nmol, nsteps)
        push!(runs, r)
        @printf("  %6d %6d %9.3f %10.2f %10.3f %10.0f B %9d %7.1f MiB\n",
                r.N, r.nsteps, r.wall_s, r.ms_per_step, r.ns_per_day,
                r.alloc_per_step, r.rebuilds, r.peak_rss_mib)
    end
    println()

    # 3. Memory budget table (measured summarysize vs ResourceProbe.md_memory_budget)
    println("[MEASURED] Memory budget vs live summarysize (atoms, neighbor capacity, PME mesh):")
    @printf("  %6s %9s %11s | %9s %9s %9s | %10s %10s\n",
            "N", "NB pairs", "mesh", "LJ(MiB)", "PME(MiB)", "NB(MiB)", "Budget+25%", "Live(MiB)")
    for pr in profiles
        b = pr.budget
        @printf("  %6d %9d %11s | %9.2f %9.2f %9.2f | %10.2f %10.2f\n",
                pr.N, pr.nl_pairs, string(pr.grid),
                b.lj_bytes / MIB, b.pme_bytes / MIB, b.neighbor_bytes / MIB,
                b.total_bytes / MIB, pr.live_bytes / MIB)
    end
    println()

    # 4. Scaling extrapolation (explicitly marked EXTRAPOLATED)
    ref = runs[end]
    ms_per_atom = ref.ms_per_step / ref.N
    println("[EXTRAPOLATED] Linear O(N) scaling projection from measured N=$(ref.N) ($(round(ref.ms_per_step, digits=2)) ms/step):")
    @printf("  %10s %11s %10s %10s %12s %12s\n",
            "N", "mesh(est)", "ms/step*", "ns/day*", "Budget(MiB)", "Preflight")
    for N_ext in (10_000, 50_000, 100_000, 1_000_000)
        L_ext = (N_ext / 0.1)^(1 / 3)
        g_ext = nextpow(2, max(12, ceil(Int, 3.0 * 3.5 * (L_ext / 10.0))))
        dims_ext = (g_ext, g_ext, g_ext)
        nb_cap = 150 * N_ext
        b_ext = RP.md_memory_budget(atoms = N_ext,
                                    neighbor_capacity = nb_cap,
                                    dims = dims_ext,
                                    order = 4,
                                    lj_cells = max(1, fld(N_ext, 16)),
                                    pme_cells = max(1, fld(N_ext, 32)),
                                    constraints = N_ext)
        pf = RP.memory_preflight(b_ext.total_bytes; probe = avail)
        ms_ext = ms_per_atom * N_ext
        ns_day_ext = (DT_NS / (ms_ext / 1e3)) * 86400.0
        @printf("  %10d %11s %10.1f %10.4f %12.1f %12s\n",
                N_ext, string(dims_ext), ms_ext, ns_day_ext,
                b_ext.total_bytes / MIB, pf.ok ? "PASS" : "SKIP_MEM")
    end
    println("  * EXTRAPOLATED values assume linear O(N) cost from N=$(ref.N) and are NOT measured.")
end

main()
