# PME cost-versus-N evidence: timings, allocations, fitted exponent, and the
# crossover against direct Ewald.
#
# What this measures, and what it does NOT
# ----------------------------------------
# This measures, for one fixed physical state point (number density 0.1 sites/Å³,
# alpha = 0.35 Å⁻¹, rcut = 8 Å, order 4, grid spacing 0.8 Å, unit prefactor):
#
#   * wall time of one energy+force evaluation (median of several runs after
#     warm-up) and allocations per call for a reused plan;
#   * peak RSS of the whole level in a *fresh* Julia process;
#   * the fitted cost exponent in N with a least-squares standard error;
#   * the direct-Ewald cost curve at the same state point and the crossover N;
#   * the error of the coarse grid against a grid-refined evaluation, versus N.
#
# It does NOT establish: performance on any other density, cutoff, spline
# order, grid spacing, or cell shape; behaviour on GPU or >1 core; comparison
# with a production engine; or that the fitted exponent extrapolates beyond the
# largest N measured. Single-node, single-threaded, FFTW via its own threads.
#
# Usage
# -----
#   JULIA_DEPOT_PATH=... julia --project=julia \
#       julia/bench/pme_scaling.jl                # full sweep
#   ... bench/pme_scaling.jl --level=10000                   # one level, JSON
#   ... bench/pme_scaling.jl --crossover
#   ... bench/pme_scaling.jl --accuracy
#   ... bench/pme_scaling.jl --all --out=experiments/pme_scaling/results
#
# Every heavy level runs in its own subprocess so "peak RSS" is attributable to
# that N and a crashed/refused level cannot poison the rest.

using Printf, Random, Statistics

const HERE = @__DIR__
const REPO = normpath(joinpath(HERE, ".."))

module PMEHost
    include(joinpath(normpath(joinpath(@__DIR__, "..")), "src", "PME.jl"))
end
const PME = PMEHost.PME

module EWaldHost
    include(joinpath(normpath(joinpath(@__DIR__, "..")), "src", "Electrostatics.jl"))
end
const EW = EWaldHost.Electrostatics

# ---------------------------------------------------------------------------
# State point (one point for every N: cost versus N at fixed physics)
# ---------------------------------------------------------------------------

const RHO = 0.1          # sites / Å³
const ALPHA = 0.35       # Å⁻¹
const RCUT = 8.0         # Å      (alpha*rcut = 2.8 -> erfc(2.8) = 7.5e-5)
const HGRID = 0.8        # Å      grid spacing held fixed while N grows
const HACC = 1.0         # Å      coarser fixed spacing for the error-vs-N table
const ORDER = 4
const PREFACTOR = 1.0

# Levels for the timing series. 1e6 is attempted only if the memory/load
# preflight passes comfortably.
const LEVELS = (1_000, 10_000, 100_000)
const BIG_LEVEL = 1_000_000

box_length(N) = (N / RHO)^(1 / 3)

"Odd grid dimension: no Nyquist plane, so the influence function has no
ambiguous half-mode and the error is a clean function of grid spacing."
function grid_for(L::Float64; h::Float64=HGRID, order::Int=ORDER)
    n = max(2 * order, round(Int, L / h))
    isodd(n) || (n += 1)
    return (n, n, n)
end

"Reproducible, exactly neutral, uniform-density system in a cubic box."
function make_system(N::Int; seed::Int=1, rho::Float64=RHO)
    rng = MersenneTwister(seed)
    L = (N / rho)^(1 / 3)
    q = Matrix{Float64}(undef, 3, N)
    rand!(rng, q)
    q .*= L
    charges = rand(rng, N) .- 0.5
    charges .-= sum(charges) / N
    return q, charges, (L, L, L)
end

# ---------------------------------------------------------------------------
# Memory / load preflight
# ---------------------------------------------------------------------------

function footprint_estimate(N::Int, dims::NTuple{3,Int}, order::Int)
    ngrid = prod(dims)
    # complex work array, float influence, one FFTW scratch of the same size,
    # spline nodes/weights/dweights, coordinate+charge scratch, linked cells
    grid = ngrid * 16 + ngrid * 8 + 2 * ngrid * 16
    per_atom = order * 3 * 8 * 3 + 3 * 8 + 8 + 8
    cells = 16 * max(prod(dims) ÷ 8, N)
    return grid + N * per_atom + cells
end

"""
Reclaimable memory: on macOS `Sys.free_memory()` counts only truly free pages,
which is a poor guide because the OS uses spare RAM as cache. Add the inactive
pages (`vm_stat`) when available; fall back to the free count elsewhere.
"""
function available_memory()
    free = Sys.free_memory()
    if Sys.isapple()
        try
            out = read(`vm_stat`, String)
            page = match(r"page size of (\d+) bytes", out)
            page === nothing && return free
            psz = parse(Int, page.captures[1])
            inactive = match(r"Pages inactive:\s+(\d+)", out)
            inactive === nothing && return free
            return free + parse(Int, inactive.captures[1]) * psz
        catch
            return free
        end
    end
    return free
end

"""
Refuse a level only for memory safety: estimate below 10% of RAM and below half
of currently reclaimable memory. A 1-minute load average at or above the CPU
count does not refuse a required measurement (the timing spread is reported);
it only refuses the *optional* 10⁶ level, so a loaded machine is not made worse.
"""
function preflight(N::Int, dims::NTuple{3,Int}, order::Int)
    est = footprint_estimate(N, dims, order)
    total = Sys.total_memory()
    avail = available_memory()
    load1 = Sys.loadavg()[1]
    ncpu = Sys.CPU_THREADS
    safe = est < 0.10 * total && est < 0.5 * avail
    loaded = load1 >= ncpu
    reason = if est >= 0.10 * total
        "estimated footprint $(round(est/2^30, digits=2)) GiB exceeds 10% of RAM"
    elseif est >= 0.5 * avail
        "estimated footprint $(round(est/2^30, digits=2)) GiB exceeds half of reclaimable $(round(avail/2^30, digits=2)) GiB"
    elseif loaded
        "1-min load average $(round(load1, digits=2)) >= $(ncpu) CPUs (measurement will be noisy)"
    else
        "ok"
    end
    return (ok=safe, safe=safe, loaded=loaded, reason=reason,
            estimate_bytes=est, total_bytes=total, available_bytes=avail,
            load1=load1, ncpu=ncpu)
end

# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

"Median, minimum, and maximum of `reps` timed calls after two warm-ups."
function timed(f; reps::Int)
    f()
    f()
    ts = Float64[]
    for _ in 1:reps
        t0 = time_ns()
        f()
        push!(ts, (time_ns() - t0) / 1e9)
    end
    return (median=median(ts), min=minimum(ts), max=maximum(ts), reps=reps)
end

"Allocations of one warm call, measured through a function barrier."
function allocs_per_call(f::F) where {F}
    f()
    return @allocated f()
end

struct Level
    N::Int
    dims::NTuple{3,Int}
    nc::NTuple{3,Int}
    n_cell_pairs::Int
    build_s::Float64
    eval::NamedTuple
    energy_only_s::Float64
    alloc_bytes::Int
    rss_before::Int
    rss_after::Int
    estimate_bytes::Int
end

function run_level(N::Int; reps::Int=0, seed::Int=1)
    q, charges, box = make_system(N; seed=seed)
    dims = grid_for(box[1])
    pf = preflight(N, dims, ORDER)
    pf.ok || return (refused=true, reason=pf.reason, preflight=pf, N=N, dims=dims)

    rss_before = Sys.maxrss()
    t0 = time_ns()
    plan = PME.PMEPlan(q, box; alpha=ALPHA, rcut=RCUT, grid=dims, order=ORDER,
                       prefactor=PREFACTOR)
    build_s = (time_ns() - t0) / 1e9

    F = zeros(3, N)
    n = reps > 0 ? reps : (N <= 10_000 ? 9 : N <= 100_000 ? 5 : 3)
    eval = timed(() -> PME.pme_energy_forces!(F, plan, q, charges); reps=n)
    energy_only = timed(() -> PME.pme_energy(plan, q, charges); reps=max(3, n ÷ 2))
    a = allocs_per_call(() -> PME.pme_energy_forces!(F, plan, q, charges))

    level = Level(N, dims, plan.cell_n, length(plan.cell_pairs), build_s, eval,
                  energy_only.median, a, rss_before, Sys.maxrss(), pf.estimate_bytes)
    return (refused=false, reason=pf.reason, preflight=pf, level=level)
end

# ---------------------------------------------------------------------------
# Fitted exponent
# ---------------------------------------------------------------------------

"""
Least-squares slope of log(time) against log(N) with its standard error.
Requires at least three positive points, i.e. at least one residual degree of
freedom.
"""
function fit_exponent(levels::Vector{Level})
    pts = [(log(l.N), log(l.eval.median)) for l in levels if l.eval.median > 0]
    length(pts) >= 3 || return (slope=NaN, se=NaN, dof=0, r2=NaN)
    n = length(pts)
    mx = mean(first.(pts)); my = mean(last.(pts))
    sxx = sum((x - mx)^2 for (x, _) in pts)
    sxy = sum((x - mx) * (y - my) for (x, y) in pts)
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_res = sum((y - (intercept + slope * x))^2 for (x, y) in pts)
    ss_tot = sum((y - my)^2 for (_, y) in pts)
    dof = n - 2
    se = dof > 0 ? sqrt(ss_res / dof / sxx) : NaN
    r2 = ss_tot > 0 ? 1 - ss_res / ss_tot : 1.0
    # two-point segment exponents make the local slope visible
    seg = [log(levels[i+1].eval.median / levels[i].eval.median) /
           log(levels[i+1].N / levels[i].N) for i in 1:n-1]
    return (slope=slope, se=se, dof=dof, r2=r2, segments=seg)
end

# ---------------------------------------------------------------------------
# Direct-Ewald crossover
# ---------------------------------------------------------------------------

"Reciprocal cutoff for a 1e-4 per-mode truncation at alpha: k_max = 2α√ln(1e4).
Using the tighter 1e-6 tolerance only makes direct Ewald more expensive, so the
reported crossover is conservative."
function ewald_kmax(L::Float64; alpha::Float64=ALPHA)
    kmaxf = 2 * alpha * sqrt(log(1e4)) / (2 * pi) * L
    return max(4, ceil(Int, kmaxf))
end

"""
Allocation-free direct Ewald with the same convention as
`Electrostatics.evaluate_ewald`, used only as a control for how much of the
shipped evaluator's cost is its per-mode allocation pattern (`k = 2π*[...]` and
`x[:,i]` slices inside the k loops) rather than the O(N² + N·K³) algebra.
Validated against the shipped evaluator in `mode_crossover`.
"""
function lean_ewald(q::Matrix{Float64}, charges::Vector{Float64},
                    L::NTuple{3,Float64}; alpha::Float64, rcut::Float64,
                    kmax::Int, prefactor::Float64=1.0)
    n = size(q, 2); V = L[1] * L[2] * L[3]
    a = alpha; rc = rcut; p = prefactor; rc2 = rc * rc
    qq = charges
    F = zeros(3, n)
    E = 0.0
    @inbounds for i in 1:n-1, j in i+1:n
        dx = q[1, i] - q[1, j]; dy = q[2, i] - q[2, j]; dz = q[3, i] - q[3, j]
        dx -= L[1] * round(dx / L[1]); dy -= L[2] * round(dy / L[2]); dz -= L[3] * round(dz / L[3])
        r2 = dx * dx + dy * dy + dz * dz
        r2 < rc2 || continue
        r = sqrt(r2); z = a * r
        er = ccall(:erfc, Cdouble, (Cdouble,), z)
        c = p * qq[i] * qq[j]
        E += c * er / r
        dudr = c * (-2a / sqrt(pi) * exp(-z * z) / r - er / r2)
        f = dudr / r
        F[1, i] -= f * dx; F[2, i] -= f * dy; F[3, i] -= f * dz
        F[1, j] += f * dx; F[2, j] += f * dy; F[3, j] += f * dz
    end
    cbuf = Vector{Float64}(undef, n); sbuf = Vector{Float64}(undef, n)
    twopi = 2 * pi; norm = 2 * pi / V
    for nx in -kmax:kmax, ny in -kmax:kmax, nz in -kmax:kmax
        (nx == 0 && ny == 0 && nz == 0) && continue
        kx = twopi * nx / L[1]; ky = twopi * ny / L[2]; kz = twopi * nz / L[3]
        k2 = kx * kx + ky * ky + kz * kz
        coeff = p * norm * exp(-k2 / (4a * a)) / k2
        C = 0.0; S = 0.0
        @inbounds for i in 1:n
            ph = kx * q[1, i] + ky * q[2, i] + kz * q[3, i]
            cb = cos(ph); sb = sin(ph)
            cbuf[i] = cb; sbuf[i] = sb
            C += qq[i] * cb; S += qq[i] * sb
        end
        E += coeff * (C * C + S * S)
        @inbounds for i in 1:n
            g = 2 * coeff * qq[i] * (C * sbuf[i] - S * cbuf[i])
            F[1, i] += g * kx; F[2, i] += g * ky; F[3, i] += g * kz
        end
    end
    Q = sum(qq)
    E -= p * a / sqrt(pi) * sum(abs2, qq)
    E -= p * pi * Q * Q / (2a * a * V)
    return E, F
end

struct CrossPoint
    N::Int
    ewald_s::Float64
    pme_s::Float64
    lean_s::Float64
    pme_build_s::Float64
    kmax::Int
    dE_rel::Float64
    lean_dE_rel::Float64
end

"""
Time direct Ewald (as shipped, and allocation-free), and PME at the same state
point. All three use the same alpha, rcut, and prefactor, hence the same
real-space splitting; direct Ewald's reciprocal sum is truncated at
`ewald_kmax`. Relative energy differences are reported so the reader can see
whether the comparison is at matched accuracy.
"""
function crossover_point(N::Int)
    q, charges, box = make_system(N; seed=2)
    L = box[1]
    kmax = ewald_kmax(L)
    dims = grid_for(L)
    t0 = time_ns()
    plan = PME.PMEPlan(q, box; alpha=ALPHA, rcut=RCUT, grid=dims, order=ORDER)
    build_s = (time_ns() - t0) / 1e9
    F = zeros(3, N)
    pme = timed(() -> PME.pme_energy_forces!(F, plan, q, charges);
                reps=N <= 2200 ? 5 : N <= 4600 ? 3 : 2)
    Ee, _ = EW.evaluate_ewald(q, charges, collect(box); alpha=ALPHA, rcut=RCUT,
                              kmax=kmax, prefactor=PREFACTOR)
    ew = timed(() -> EW.evaluate_ewald(q, charges, collect(box); alpha=ALPHA,
                                       rcut=RCUT, kmax=kmax,
                                       prefactor=PREFACTOR);
               reps=N <= 2200 ? 5 : N <= 4600 ? 3 : 2)
    El, _ = lean_ewald(q, charges, box; alpha=ALPHA, rcut=RCUT, kmax=kmax,
                       prefactor=PREFACTOR)
    lean = timed(() -> lean_ewald(q, charges, box; alpha=ALPHA, rcut=RCUT,
                                  kmax=kmax, prefactor=PREFACTOR);
                 reps=N <= 2200 ? 5 : N <= 4600 ? 3 : 2)
    Up = PME.pme_energy_forces!(F, plan, q, charges)
    # the lean control must reproduce the shipped evaluator, or it is not a control
    @assert abs(El - Ee) / max(abs(Ee), 1.0) < 1e-12
    return CrossPoint(N, ew.median, pme.median, lean.median, build_s, kmax,
                      abs(Ee - Up) / max(abs(Ee), 1.0),
                      abs(El - Up) / max(abs(El), 1.0))
end

# ---------------------------------------------------------------------------
# Accuracy versus N at fixed grid spacing
# ---------------------------------------------------------------------------

struct AccPoint
    N::Int
    coarse_dims::NTuple{3,Int}
    fine_dims::NTuple{3,Int}
    dE_rel::Float64
    dE_sem::Float64
    dF_rms_rel::Float64
    dF_sem::Float64
    ewald_dE_rel::Float64
    ewald_dF_rms_rel::Float64
    ewald_kmax::Int
end

"""
At a fixed grid spacing, compare the coarse grid with a twice-refined grid and
(with direct Ewald where affordable) with an independent reference. Averaged
over `seeds` independent configurations, so grid-mode accidents do not drive
the table.
"""
function accuracy_point(N::Int; seeds::Int=5)
    relE = Float64[]; relF = Float64[]
    eE = Float64[]; eF = Float64[]
    cdims = grid_for(box_length(N); h=HACC)
    fdims = grid_for(box_length(N); h=HACC / 2)
    kmax = 0
    for s in 1:seeds
        q, charges, box = make_system(N; seed=s)
        pc = PME.PMEPlan(q, box; alpha=ALPHA, rcut=RCUT, grid=cdims, order=ORDER)
        pf = PME.PMEPlan(q, box; alpha=ALPHA, rcut=RCUT, grid=fdims, order=ORDER)
        Uc, Fc = PME.pme_energy_forces(pc, q, charges)
        Uf, Ff = PME.pme_energy_forces(pf, q, charges)
        push!(relE, abs(Uc - Uf) / max(abs(Uf), 1.0))
        push!(relF, sqrt(mean(abs2, Fc - Ff)) / max(sqrt(mean(abs2, Ff)), 1e-30))
        if N <= 10_000   # direct Ewald is affordable only here
            kmax = ewald_kmax(box[1])
            Ue, Fe = EW.evaluate_ewald(q, charges, collect(box); alpha=ALPHA,
                                       rcut=RCUT, kmax=kmax, prefactor=PREFACTOR)
            push!(eE, abs(Uc - Ue) / max(abs(Ue), 1.0))
            push!(eF, sqrt(mean(abs2, Fc - Fe)) / max(sqrt(mean(abs2, Fe)), 1e-30))
        end
    end
    sem(v) = length(v) > 1 ? std(v) / sqrt(length(v)) : NaN
    return AccPoint(N, cdims, fdims, mean(relE), sem(relE), mean(relF), sem(relF),
                    isempty(eE) ? NaN : mean(eE), isempty(eF) ? NaN : mean(eF), kmax)
end

# ---------------------------------------------------------------------------
# Small JSON writer (no external dependency)
# ---------------------------------------------------------------------------

fmt(x::Float64) = isfinite(x) ? @sprintf("%.6g", x) : "null"
jobj(pairs::Vector{Tuple{String,String}}) =
    "{" * join(["\"$(k)\":$(v)" for (k, v) in pairs], ",") * "}"

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

function print_level_table(levels::Vector{Level})
    println("== PME energy+force cost versus N (density $(RHO)/Å³, alpha=$(ALPHA), rcut=$(RCUT) Å, order=$(ORDER), h=$(HGRID) Å) ==")
    @printf("%9s %12s %6s %8s %10s %10s %12s %10s %10s\n",
            "N", "grid", "cells", "cellpairs", "median (s)", "min (s)", "alloc (B/call)",
            "RSS (MiB)", "dRSS (MiB)")
    for l in levels
        @printf("%9d %12s %6s %8d %10.4f %10.4f %12d %10.1f %10.1f\n",
                l.N, string(l.dims), string(l.nc), l.n_cell_pairs,
                l.eval.median, l.eval.min, l.alloc_bytes, l.rss_after / 2^20,
                (l.rss_after - l.rss_before) / 2^20)
    end
    fit = fit_exponent(levels)
    @printf("\nfitted exponent d log(t) / d log(N) = %.3f +/- %.3f (R²=%.4f, %d points)\n",
            fit.slope, fit.se, fit.r2, length(levels))
    if hasproperty(fit, :segments)
        for (i, s) in enumerate(fit.segments)
            @printf("   segment %d->%d: %.3f\n", levels[i].N, levels[i+1].N, s)
        end
    end
    println("plan build time (s): ", [l.build_s for l in levels])
    return fit
end

function mode_level(N::Int)
    r = run_level(N)
    if r.refused
        println(jobj([("N", string(N)), ("refused", "true"),
                      ("reason", "\"$(r.reason)\"")]))
        return 1
    end
    l = r.level
    println(jobj([
        ("N", string(l.N)),
        ("grid", "\"$(l.dims)\""),
        ("cells", "\"$(l.nc)\""),
        ("cell_pairs", string(l.n_cell_pairs)),
        ("build_s", fmt(l.build_s)),
        ("median_s", fmt(l.eval.median)),
        ("min_s", fmt(l.eval.min)),
        ("max_s", fmt(l.eval.max)),
        ("energy_only_s", fmt(l.energy_only_s)),
        ("alloc_bytes", string(l.alloc_bytes)),
        ("rss_before", string(l.rss_before)),
        ("rss_after", string(l.rss_after)),
        ("estimate_bytes", string(l.estimate_bytes)),
        ("reps", string(l.eval.reps)),
        ("refused", "false"),
    ]))
    return 0
end

function read_result(line::AbstractString)
    # tiny parser for the flat objects we emit (values are numbers, booleans or
    # quoted strings). Comma splitting must respect quotes, because the grid and
    # cell dimensions are printed as "(a, b, c)".
    body = strip(line, ['{', '}', '\n'])
    parts = String[]
    cur = IOBuffer()
    inq = false
    for ch in body
        if ch == '"'
            inq = !inq
            print(cur, ch)
        elseif ch == ',' && !inq
            push!(parts, String(take!(cur)))
        else
            print(cur, ch)
        end
    end
    push!(parts, String(take!(cur)))
    out = Dict{String,String}()
    for kv in parts
        k, v = split(kv, ':', limit=2)
        out[strip(k, ['"', ' '])] = strip(v)
    end
    return out
end

function parse_level(d::Dict{String,String})
    get(d, "refused", "false") == "true" && return nothing
    Level(parse(Int, d["N"]),
          Tuple(parse.(Int, split(strip(d["grid"], ['"', '(', ')']), ", "))),
          Tuple(parse.(Int, split(strip(d["cells"], ['"', '(', ')']), ", "))),
          parse(Int, d["cell_pairs"]),
          parse(Float64, d["build_s"]),
          (median=parse(Float64, d["median_s"]), min=parse(Float64, d["min_s"]),
           max=parse(Float64, d["max_s"]), reps=parse(Int, d["reps"])),
          parse(Float64, d["energy_only_s"]),
          parse(Int, d["alloc_bytes"]),
          parse(Int, d["rss_before"]),
          parse(Int, d["rss_after"]),
          parse(Int, d["estimate_bytes"]))
end

julia_exe() = joinpath(Sys.BINDIR, "julia")
project_flag() = ["--project=$(REPO)"]

function run_child(N::Int)
    cmd = Cmd([julia_exe(), project_flag()..., @__FILE__, "--level=$(N)"])
    out = read(cmd, String)
    for line in split(out, '\n')
        startswith(line, "{") && return read_result(line)
    end
    return Dict{String,String}("refused" => "true", "reason" => "\"no output\"")
end

function mode_sweep(; big::Bool=false)
    levels = Level[]
    println("== fresh subprocess per level ==")
    for N in (big ? (LEVELS..., BIG_LEVEL) : LEVELS)
        dims = grid_for(box_length(N))
        pf = preflight(N, dims, ORDER)
        @printf("N=%7d preflight: est=%.2f GiB reclaimable=%.2f GiB load1=%.2f/%.0f -> %s\n",
                N, pf.estimate_bytes / 2^30, pf.available_bytes / 2^30,
                pf.load1, pf.ncpu, pf.ok ? "run" : "skip")
        if !pf.safe
            println("N=$(N): refused by preflight — $(pf.reason)")
            continue
        elseif N == BIG_LEVEL && pf.loaded
            println("N=$(N): optional level skipped — $(pf.reason)")
            continue
        end
        d = run_child(N)
        if get(d, "refused", "false") == "true"
            println("N=$(N): refused — $(get(d, "reason", "?"))")
            continue
        end
        l = parse_level(d)
        l === nothing && (println("N=$(N): refused"); continue)
        push!(levels, l)
        @printf("N=%7d grid=%s  median=%.4f s  min=%.4f s  alloc=%d B  RSS=%.1f MiB (+%.1f)\n",
                l.N, l.dims, l.eval.median, l.eval.min, l.alloc_bytes,
                l.rss_after / 2^20, (l.rss_after - l.rss_before) / 2^20)
    end
    sort!(levels, by=l -> l.N)
    fit = print_level_table(levels)
    return levels, fit
end

function mode_crossover()
    println("== PME vs direct Ewald at matched alpha/rcut (kmax for 1e-4 per-mode) ==")
    @printf("%8s %6s %12s %12s %12s %10s %10s %10s\n", "N", "kmax", "ewald (s)",
            "ewald lean", "pme (s)", "e/pme", "lean/pme", "|dE|/|E|")
    pts = CrossPoint[]
    for N in (500, 700, 1_000, 1_500, 2_200, 3_200, 4_600, 6_800, 10_000)
        p = crossover_point(N)
        push!(pts, p)
        @printf("%8d %6d %12.4f %12.4f %12.4f %10.2f %10.2f %10.2e\n",
                p.N, p.kmax, p.ewald_s, p.lean_s, p.pme_s,
                p.ewald_s / p.pme_s, p.lean_s / p.pme_s, p.dE_rel)
    end
    # Crossing of log(cost_ewald / cost_pme) through zero, interpolated in log N.
    function crossing(pts, cost)
        for i in 1:length(pts)-1
            ri = cost(pts[i]) / pts[i].pme_s
            rj = cost(pts[i+1]) / pts[i+1].pme_s
            if ri >= 1 && rj < 1
                t = log(ri) / (log(ri) - log(rj))
                return exp(log(pts[i].N) + t * (log(pts[i+1].N) - log(pts[i].N)))
            end
        end
        return NaN
    end
    for (label, cost) in (("shipped direct Ewald", p -> p.ewald_s),
                          ("allocation-free direct Ewald", p -> p.lean_s))
        cross = crossing(pts, cost)
        if isnan(cross)
            @printf("\n%s: still cheaper than PME at N=%d; crossover is below the measured range.\n",
                    label, pts[end].N)
        else
            @printf("\n%s: crossover N ≈ %.0f (PME cheaper above it)\n", label, cross)
        end
    end
    @printf("smallest measured N is %d (box %.1f Å must exceed 2*rcut = %.1f Å).\n",
            pts[1].N, box_length(pts[1].N), 2 * RCUT)
    println("direct Ewald cost here is O(N²) real space + O(N·kmax³) reciprocal; the EwaldFactor guard (n ≤ 10,000, kmax ≤ 16) is not involved.")
    return pts
end

"""
Scale-consistent small-N state: `rcut = 0.45·L`, `alpha = 3.2/rcut`, and grid
spacing `h = 0.8·0.35/alpha`, so that `alpha·rcut` (real-space truncation) and
`alpha·h` (grid error) are the same as in the main series while the box stays
big enough for the cutoff. Used only to locate the crossover below N=500, where
a fixed 8 Å cutoff no longer fits in a box of twice that width.
"""
function scaled_state(N::Int)
    L = box_length(N)
    rcut = 0.45 * L
    alpha = 3.2 / rcut
    h = HGRID * ALPHA / alpha
    n = max(2 * ORDER + 1, round(Int, L / h))
    isodd(n) || (n += 1)
    dims = (n, n, n)
    kmax = max(4, ceil(Int, 2 * alpha * sqrt(log(1e4)) / (2 * pi) * L))
    return (L=L, rcut=rcut, alpha=alpha, dims=dims, kmax=kmax)
end

function crossover_scaled(N::Int)
    q, charges, box = make_system(N; seed=2)
    st = scaled_state(N)
    t0 = time_ns()
    plan = PME.PMEPlan(q, box; alpha=st.alpha, rcut=st.rcut, grid=st.dims,
                       order=ORDER)
    build_s = (time_ns() - t0) / 1e9
    F = zeros(3, N)
    pme = timed(() -> PME.pme_energy_forces!(F, plan, q, charges); reps=9)
    Ee, _ = EW.evaluate_ewald(q, charges, collect(box); alpha=st.alpha,
                              rcut=st.rcut, kmax=st.kmax, prefactor=PREFACTOR)
    ew = timed(() -> EW.evaluate_ewald(q, charges, collect(box); alpha=st.alpha,
                                       rcut=st.rcut, kmax=st.kmax,
                                       prefactor=PREFACTOR); reps=5)
    El, _ = lean_ewald(q, charges, box; alpha=st.alpha, rcut=st.rcut,
                       kmax=st.kmax, prefactor=PREFACTOR)
    lean = timed(() -> lean_ewald(q, charges, box; alpha=st.alpha, rcut=st.rcut,
                                  kmax=st.kmax, prefactor=PREFACTOR); reps=9)
    Up = PME.pme_energy_forces!(F, plan, q, charges)
    @assert abs(El - Ee) / max(abs(Ee), 1.0) < 1e-12
    return CrossPoint(N, ew.median, pme.median, lean.median, build_s, st.kmax,
                      abs(Ee - Up) / max(abs(Ee), 1.0),
                      abs(El - Up) / max(abs(El), 1.0))
end

function mode_crossover_small()
    println("== crossover below N=500 with scale-consistent alpha/rcut (rcut=0.45L, alpha=3.2/rcut, alpha*h fixed) ==")
    @printf("%8s %8s %8s %8s %12s %12s %12s %10s %10s\n", "N", "rcut", "alpha",
            "kmax", "ewald (s)", "ewald lean", "pme (s)", "e/pme", "lean/pme")
    pts = CrossPoint[]
    for N in (25, 50, 100, 150, 200, 300, 400, 500)
        p = crossover_scaled(N)
        push!(pts, p)
        st = scaled_state(N)
        @printf("%8d %8.2f %8.3f %8d %12.4f %12.4f %12.4f %10.2f %10.2f\n",
                p.N, st.rcut, st.alpha, p.kmax, p.ewald_s, p.lean_s, p.pme_s,
                p.ewald_s / p.pme_s, p.lean_s / p.pme_s)
    end
    for (label, cost) in (("shipped direct Ewald", p -> p.ewald_s),
                          ("allocation-free direct Ewald", p -> p.lean_s))
        cross = NaN
        for i in 1:length(pts)-1
            ri = cost(pts[i]) / pts[i].pme_s
            rj = cost(pts[i+1]) / pts[i+1].pme_s
            if ri >= 1 && rj < 1
                t = log(ri) / (log(ri) - log(rj))
                cross = exp(log(pts[i].N) + t * (log(pts[i+1].N) - log(pts[i].N)))
            end
        end
        isnan(cross) ? @printf("\n%s: no crossing in %d..%d\n", label, pts[1].N, pts[end].N) :
                       @printf("\n%s: crossover N ≈ %.0f (PME cheaper above it)\n", label, cross)
    end
    return pts
end

function mode_accuracy()
    println("== error versus N at fixed alpha/order/spacing (h=$(HACC) Å coarse, h/2 refined; 5 seeds, mean ± sem) ==")
    @printf("%8s %10s %10s %16s %16s %12s %12s %6s\n", "N", "coarse", "fine",
            "grid dE/E", "grid rms dF", "Ewald dE/E", "Ewald rms dF", "kmax")
    pts = AccPoint[]
    for N in (1_000, 3_000, 10_000, 30_000, 100_000)
        p = accuracy_point(N)
        push!(pts, p)
        @printf("%8d %10s %10s %8.2e±%7.2e %8.2e±%7.2e %12.2e %12.2e %6d\n",
                p.N, string(p.coarse_dims), string(p.fine_dims), p.dE_rel, p.dE_sem,
                p.dF_rms_rel, p.dF_sem, p.ewald_dE_rel, p.ewald_dF_rms_rel, p.ewald_kmax)
    end
    return pts
end

function main(args)
    outdir = nothing
    big = false
    repeat = 1
    mode = :sweep
    for a in args
        if startswith(a, "--level=")
            return mode_level(parse(Int, split(a, "=")[2]))
        elseif a == "--crossover"
            mode = :crossover
        elseif a == "--crossover-small"
            mode = :crossover_small
        elseif a == "--accuracy"
            mode = :accuracy
        elseif a == "--all"
            mode = :all
        elseif a == "--big"
            big = true
        elseif startswith(a, "--repeat=")
            repeat = parse(Int, split(a, "=")[2])
        elseif startswith(a, "--out=")
            outdir = split(a, "=")[2]
        end
    end
    if outdir !== nothing
        mkpath(outdir)
        open(joinpath(outdir, "sweep.txt"), "w") do io
            redirect_stdout(io) do
                levels, fit = mode_sweep(; big=big)
                open(joinpath(outdir, "levels.json"), "w") do j
                    print(j, "[")
                    for (i, l) in enumerate(levels)
                        i > 1 && print(j, ",")
                        print(j, jobj([("N", string(l.N)),
                                       ("grid", "\"$(l.dims)\""),
                                       ("median_s", fmt(l.eval.median)),
                                       ("min_s", fmt(l.eval.min)),
                                       ("alloc_bytes", string(l.alloc_bytes)),
                                       ("rss_after", string(l.rss_after)),
                                       ("rss_delta", string(l.rss_after - l.rss_before)),
                                       ("build_s", fmt(l.build_s))]))
                    end
                    print(j, "]")
                end
                @printf("\nexponent %.3f +/- %.3f\n", fit.slope, fit.se)
            end
        end
        if repeat > 1
            open(joinpath(outdir, "runs.txt"), "w") do io
                redirect_stdout(io) do
                    fits = Float64[]
                    times = Dict{Int,Vector{Float64}}()
                    for r in 1:repeat
                        println("--- sweep run $(r)/$(repeat) ---")
                        lv, ft = mode_sweep(; big=false)
                        push!(fits, ft.slope)
                        for l in lv
                            push!(get!(times, l.N, Float64[]), l.eval.median)
                        end
                    end
                    println("\n== across-run spread (same machine, same state point) ==")
                    @printf("%9s %14s %14s %14s %10s\n", "N", "median of runs", "min", "max", "spread")
                    for N in sort(collect(keys(times)))
                        v = times[N]
                        @printf("%9d %14.4f %14.4f %14.4f %9.2fx\n", N, median(v),
                                minimum(v), maximum(v), maximum(v) / minimum(v))
                    end
                    @printf("\nexponent per run: %s   median %.3f\n",
                            join([@sprintf("%.3f", f) for f in fits], ", "), median(fits))
                end
            end
        end
        for (name, f, file) in (("crossover", mode_crossover, "crossover.txt"),
                                ("crossover_small", mode_crossover_small, "crossover_small.txt"),
                                ("accuracy", mode_accuracy, "accuracy.txt"))
            open(joinpath(outdir, file), "w") do io
                redirect_stdout(io) do
                    f()
                end
            end
        end
        println("wrote results under $(outdir)")
        return 0
    end
    mode === :crossover && (mode_crossover(); return 0)
    mode === :crossover_small && (mode_crossover_small(); return 0)
    mode === :accuracy && (mode_accuracy(); return 0)
    mode_sweep(; big=big)
    mode === :all && (mode_crossover(); mode_crossover_small(); mode_accuracy())
    return 0
end

exit(main(ARGS))
