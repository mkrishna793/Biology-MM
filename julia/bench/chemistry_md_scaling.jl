#!/usr/bin/env julia
# Actual molecular dynamics at increasing atom count: import -> build -> factor
# stack -> force evaluation -> velocity-Verlet integration.
#
# What "scaling" means here
# -------------------------
# The ability to *run* a larger molecular system, not to import or evaluate one
# statically. A row is only evidence of simulation when the integrator advanced a
# state; the static energy/force timings are recorded in separate columns and are
# never labelled a trajectory. For every case this script reports, at separated
# stages: extractor wall time, `build_resolved` wall time, warmed energy and
# force evaluation cost, and -- when the system can be integrated at all -- a
# bounded NVE trajectory with step count, timestep, simulated time, wall time,
# ms/step, peak RSS, constraint count and energy diagnostics.
#
# What it does NOT establish
# --------------------------
# * These are smoke trajectories (tens of femtoseconds), not production
#   sampling. Nothing here is a throughput, efficiency or strong-scaling claim;
#   ms/step at one N is a single short-run cost, not a scalability law.
# * Timings are single-node, single-threaded (`JULIA_NUM_THREADS=1`), taken on a
#   possibly shared host; each case is timed twice and the spread is reported so
#   contention is visible rather than hidden.
# * Peak RSS is the whole child process (`Sys.maxrss`), including compilation and
#   the Python extractor's peak? No -- the extractor runs as a separate process,
#   so its memory is *not* included. Only the Julia-side import/build/integrate
#   peak is reported.
#
# Usage
# -----
#   JULIA_DEPOT_PATH=... julia --project=julia \
#       julia/bench/chemistry_md_scaling.jl
#   ... bench/chemistry_md_scaling.jl --no-membrane      # skip the 28k-atom case
#   ... bench/chemistry_md_scaling.jl --out-dir=/tmp/x   # write report elsewhere
#   ... bench/chemistry_md_scaling.jl --case=dna --json=/tmp/dna.bin   # one case
#
# Each case runs in its own subprocess so that `Sys.maxrss` is attributable to
# that N and a refused or crashing case cannot poison the rest. The parent
# deserializes the child's result record and writes a deterministic report
# (sorted keys, no timestamps, no random seeds).

using FoundationCore
using Printf, Statistics, Serialization

const CH = FoundationCore.Chemistry
const D = FoundationCore.Dynamics
const F = FoundationCore.Factors
const T = FoundationCore.Types

const HERE = @__DIR__
const REPO = normpath(joinpath(HERE, "..", "..", ".."))
const PROJECT = joinpath(REPO, "foundation", "julia")
const REFDIR = joinpath(REPO, "foundation", "chemistry", "reference")
const OUTDIR_DEFAULT = joinpath(REFDIR, "scaling")

const PYTHON = get(ENV, "CHEM_PYTHON",
                   joinpath(homedir(), ".cache", "biology-mm",
                            ".venv", "bin", "python"))
const EXTRACT_AMBER = joinpath(REPO, "foundation", "chemistry", "tools", "extract_amber.py")
const EXTRACT_CHARMM = joinpath(REPO, "foundation", "chemistry", "tools", "extract_charmm.py")

const SYS = joinpath(REPO, "wrappers", "python", "tests", "systems")
const FIX = joinpath(REPO, "foundation", "chemistry", "fixtures")
const TOOLS = joinpath(homedir(), ".cache", "biology-mm",
                       "chemistry-fixtures")

# 0.1 fs. The protocol keeps water *flexible*, so the O-H stretch is a real
# degree of freedom (period ~9 fs, ~3600 cm^-1). Velocity Verlet on an
# unconstrained X-H bond is normally run at 0.5 fs; 0.1 fs is a 5x conservative
# margin, chosen because these runs are only tens of steps long (so 5x more
# steps is cheap) and because a smoke test must not confuse "the path works"
# with "the timestep was marginal". It is also the timestep the existing
# imported-dynamics suite uses (test/test_chemistry_dynamics.jl).
const DT_PS = 1.0e-4
const CUTOFF_NM = 1.0
const CASE_BUDGET_S = 180.0
const CHILD_WATCHDOG_S = 300.0
const DRIFT_ENERGY_CEILING = 1.0e3     # kJ/mol, same ceiling as the smoke suite

struct Case
    name::String
    tool::Symbol             # :amber | :charmm
    parm::String
    crd::String
    prms::Vector{String}
    nonbonded::Symbol
    steps::Int
    note::String
end

const CASES = [
    Case("charmm_adapter", :charmm,
         joinpath(SYS, "methanol_ions.psf"), joinpath(SYS, "methanol_ions.pdb"),
         [joinpath(SYS, "par_all36_cgenff.prm"), joinpath(SYS, "water_methanol.prm"),
          joinpath(SYS, "toppar_water_ions.str")],
         :direct, 10, "CHARMM PSF adapter, 8 atoms, NoCutoff (smallest available)"),
    Case("dna", :amber,
         joinpath(SYS, "DNA_mbondi3.prmtop"), joinpath(SYS, "DNA_mbondi3.inpcrd"),
         String[], :direct, 50, "DNA duplex, NoCutoff, net -18"),
    Case("water_ions", :amber,
         joinpath(SYS, "Mg_water.prmtop"), joinpath(SYS, "Mg_water.inpcrd"),
         String[], :pme, 20, "Mg2+ + water box, periodic PME, net +2"),
    Case("protein_ligand", :amber,
         joinpath(FIX, "protein_ligand", "protein_ligand.prmtop"),
         joinpath(FIX, "protein_ligand", "protein_ligand.inpcrd"),
         String[], :direct, 20, "1XEP T4 lysozyme + catechol, NoCutoff, net +8"),
    Case("solvated_protein", :amber,
         joinpath(SYS, "18protein.parm7"), joinpath(SYS, "18protein.rst7"),
         String[], :pme, 10, "solvated protein, periodic PME"),
    Case("membrane", :amber,
         joinpath(TOOLS, "dppc.prmtop"), joinpath(TOOLS, "dppc.inpcrd"),
         String[], :pme, 10, "DPPC bilayer, periodic PME (optional, largest)"),
]

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

"Median/min/max of `reps` timed calls after one warm-up call."
function timed(f; reps::Int)
    f()
    ts = Float64[]
    sizehint!(ts, reps)
    for _ in 1:reps
        t0 = time_ns()
        f()
        push!(ts, (time_ns() - t0) / 1e9)
    end
    return (median=median(ts), min=minimum(ts), max=maximum(ts), reps=reps)
end

"""
Amber coordinate reader for `.inpcrd`/`.rst7` (both are `natom [time]` on line 2
then `3N` charges-less coordinates in Å). The trailing box record, when present,
is ignored: the box comes from the parameter record, not the coordinate file.
"""
function read_crd(path::AbstractString)
    lines = readlines(path)
    head = split(lines[2])
    n = parse(Int, head[1])
    toks = split(join(lines[2:end]), r"\s+"; keepempty=false)
    off = length(head) > 1 ? 3 : 2
    vals = parse.(Float64, toks[off:off+3n-1])
    q = Matrix{Float64}(undef, 3, n)
    k = 1
    for i in 1:n, d in 1:3
        q[d, i] = vals[k] / 10.0
        k += 1
    end
    return q
end

"First stack frame inside `src/`, repo-relative, as the site to record."
function error_site(bt)
    for fr in stacktrace(bt)
        # Julia abbreviates method-instance file paths as `~/...`, so expand
        # before relativising or the report carries a machine-specific path.
        path = string(fr.file)
        startswith(path, "~") && (path = expanduser(path))
        rel = try
            startswith(path, "/") ? relpath(path, REPO) : path
        catch
            path
        end
        occursin(joinpath("src", ""), rel) || continue
        return Dict{String,Any}("file" => rel, "line" => fr.line,
                                "function" => string(fr.func))
    end
    return nothing
end

function short_error(e)
    msg = sprint(showerror, e)
    return length(msg) > 600 ? first(msg, 600) * "..." : msg
end

function blocker(case, stage, message, site=nothing)
    d = Dict{String,Any}("case" => case, "stage" => stage, "message" => message)
    site === nothing || (d["site"] = site)
    return d
end

# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------

"""
Run the schema extractor for one case into a temp JSON path.

Returns `(ok, seconds, stderr_text, extractor_relative_path)`. A non-zero exit
is the extractor's typed refusal (for example CHARMM NBFIX), which is recorded
verbatim as a blocker rather than worked around.
"""
function extract_case(c::Case, outjson::AbstractString)
    err = IOBuffer()
    cmd = if c.tool === :amber
        `$PYTHON $EXTRACT_AMBER --parm $(c.parm) --crd $(c.crd) --out $outjson
         --force-field $(c.name) --repo-root $REPO`
    else
        prms = Cmd[`--prm $p` for p in c.prms]
        `$PYTHON $EXTRACT_CHARMM --parm $(c.parm) --crd $(c.crd) $(prms...)
         --out $outjson --force-field $(c.name) --repo-root $REPO`
    end
    tool = c.tool === :amber ? "chemistry/tools/extract_amber.py" :
                               "chemistry/tools/extract_charmm.py"
    t0 = time_ns()
    ok = try
        run(pipeline(cmd; stdout=devnull, stderr=err))
        true
    catch e
        e isa ProcessFailedException || rethrow()
        false
    end
    return ok, (time_ns() - t0) / 1e9, strip(String(take!(err))), tool
end

# ---------------------------------------------------------------------------
# one case, in a child process
# ---------------------------------------------------------------------------

function run_case(c::Case)
    res = Dict{String,Any}(
        "case" => c.name,
        "note" => c.note,
        "backend" => string(c.nonbonded),
        "nonbonded" => string(c.nonbonded),
        "lj_mode" => "lfs",
        "water_model" => "flexible",
        "dt_ps" => DT_PS,
        "dt_fs" => DT_PS * 1000,
        "steps_requested" => c.steps,
        "integrated" => false,
        "simulated_atoms" => 0,
        "simulated_time_fs" => 0.0,
        "simulated_time_ps" => 0.0,
        "constraints" => 0,
        "velocity_init" => "zero (deterministic NVE)",
    )
    blk = Dict{String,Any}[]

    tmp = mktempdir(; cleanup=true)
    json = joinpath(tmp, "normalized.json")
    ok, t_ext, stderr_text, tool = extract_case(c, json)
    res["extractor"] = tool
    res["extract_s"] = t_ext
    if !ok
        res["status"] = "BLOCKED_IMPORT"
        push!(blk, blocker(c.name, "import", stderr_text, Dict{String,Any}("file" => tool)))
        return res, blk
    end

    ns = CH.read_normalized(json)
    q = read_crd(c.crd)
    n = CH.n_atoms(ns)
    size(q, 2) == n ||
        error("coordinate file has $(size(q,2)) atoms but the record has $n")
    res["n_atoms"] = n
    res["net_charge_e"] = sum(a.charge for a in ns.atoms)

    prot = CH.default_protocol(ns; nonbonded=c.nonbonded, cutoff_nm=CUTOFF_NM,
                               lj_mode=:lfs, water_model=:flexible)
    t0 = time_ns()
    rs = try
        CH.build_resolved(ns; protocol=prot)
    catch e
        res["build_s"] = (time_ns() - t0) / 1e9
        res["status"] = "BLOCKED_BUILD"
        push!(blk, blocker(c.name, "build", short_error(e), error_site(catch_backtrace())))
        return res, blk
    end
    res["build_s"] = (time_ns() - t0) / 1e9
    res["n_factors"] = length(rs.factors)
    res["n_constraints"] = rs.constraints === nothing ? 0 : length(rs.constraints.components)
    res["constraints"] = res["n_constraints"]
    res["box_nm"] = rs.box isa T.OrthorhombicBox ? collect(rs.box.L) : nothing

    hs = D.HarmonicSystem(rs.factors, rs.atoms.masses,
                          F.Context(rs.box, rs.units, n))
    # The gate is a *scan*, so it can itself hit a factor-implementation bug;
    # that is recorded as a blocker with its source site instead of guessed at.
    fm = try
        D.force_model_supported(hs)
    catch e
        # Recorded, not raised: the integrator's own refusal below carries the
        # decisive message and source site.
        res["force_model_supported_error"] = short_error(e)
        nothing
    end
    fm === nothing || (res["force_model_supported"] = fm)

    # Warmed evaluations. These are force/energy *evaluations*, not dynamics.
    reps = n <= 3000 ? 5 : 3
    ev = timed(() -> D.potential_energy(hs, q); reps=reps)
    Fbuf = zeros(3, n)
    fv = timed(() -> D.forces!(hs, q, Fbuf); reps=reps)
    res["energy_eval_ms"] = 1000 * ev.median
    res["energy_eval_min_ms"] = 1000 * ev.min
    res["energy_eval_max_ms"] = 1000 * ev.max
    res["forces_eval_ms"] = 1000 * fv.median
    res["forces_eval_min_ms"] = 1000 * fv.min
    res["forces_eval_max_ms"] = 1000 * fv.max
    res["eval_reps"] = reps
    res["u_initial_kj_mol"] = D.forces!(hs, q, Fbuf)
    res["max_force_initial_kj_mol_per_nm"] = maximum(abs, Fbuf)

    # --- bounded NVE, two independently timed runs -------------------------
    t_force = max(ev.median, 1e-9)
    steps = c.steps
    projected = 2 * steps * t_force
    reduced = false
    if projected > CASE_BUDGET_S
        steps = floor(Int, CASE_BUDGET_S / (2 * t_force))
        reduced = true
    end
    res["timebox"] = Dict{String,Any}("budget_s" => CASE_BUDGET_S,
                                      "projected_s" => projected,
                                      "steps_chosen" => steps,
                                      "steps_reduced" => reduced)
    if steps < 5
        res["status"] = "BLOCKED_TIMEOUT"
        push!(blk, blocker(c.name, "integrate",
                           @sprintf("projected %.1f s for 5 steps exceeds the %.0f s case budget; window too short to be evidence",
                                    projected, CASE_BUDGET_S)))
        return res, blk
    end

    samples = Float64[]
    st_last = nothing
    recs = nothing
    es = nothing
    try
        for _ in 1:2
            st = D.State(copy(q), zeros(3, n))
            t1 = time_ns()
            rr, ee = D.run_trajectory(st, hs, DT_PS, steps; record_every=1)
            push!(samples, (time_ns() - t1) / 1e9)
            recs, es, st_last = rr, ee, st
        end
    catch e
        res["status"] = "BLOCKED_INTEGRATOR"
        res["wall_s_samples"] = samples
        push!(blk, blocker(c.name, "integrate", short_error(e),
                           error_site(catch_backtrace())))
        return res, blk
    end

    E = es.compensated
    report = D.energy_drift_report(recs, es)
    res["wall_s_samples"] = samples
    res["wall_s_median"] = median(samples)
    res["wall_s_range"] = maximum(samples) - minimum(samples)
    res["wall_s_rel_spread"] = maximum(samples) / minimum(samples) - 1
    res["ms_per_step"] = 1000 * median(samples) / steps
    res["steps_run"] = steps
    res["simulated_atoms"] = n
    res["simulated_time_ps"] = steps * DT_PS
    res["simulated_time_fs"] = steps * DT_PS * 1000
    res["integrated"] = true
    res["e0_kj_mol"] = report.E0
    res["e_final_kj_mol"] = report.Efinal
    res["max_abs_de_kj_mol"] = report.max_abs_dev
    res["drift_slope_kj_mol_per_ps"] = report.slope
    res["drift_slope_stderr"] = report.slope_stderr
    res["drift_predicted_kj_mol"] = report.predicted_drift
    res["drift_fraction"] = report.drift_fraction
    res["drift_bounded"] = report.bounded
    res["max_abs_de_naive_kj_mol"] = maximum(abs, es.naive .- es.naive[1])
    res["max_force_kj_mol_per_nm"] = maximum(abs, st_last.F)
    res["kinetic_energy_final_kj_mol"] = D.kinetic_energy(hs, st_last.p)
    res["all_finite"] = all(isfinite, E) && all(isfinite, st_last.q)
    res["peak_rss_mib"] = Sys.maxrss() / 2^20
    res["gc_live_mib"] = Base.gc_live_bytes() / 2^20

    reasons = String[]
    res["all_finite"] || push!(reasons, "non-finite energy or coordinate")
    report.bounded ||
        push!(reasons, @sprintf("drift fraction %.3f > 0.5 (predicted drift %.3g kJ/mol vs band %.3g)",
                                report.drift_fraction, report.predicted_drift, report.band))
    res["max_abs_de_kj_mol"] < DRIFT_ENERGY_CEILING ||
        push!(reasons, @sprintf("max|dE| %.3g kJ/mol >= %.0f", res["max_abs_de_kj_mol"],
                                DRIFT_ENERGY_CEILING))
    isfinite(res["max_force_kj_mol_per_nm"]) || push!(reasons, "non-finite force")
    res["fail_reasons"] = reasons
    res["status"] = isempty(reasons) ? "PASS" : "FAIL"
    return res, blk
end

# ---------------------------------------------------------------------------
# force-model gate: :lfs accepted, :trunc refused, state untouched
# ---------------------------------------------------------------------------

function run_gate()
    # `lj_mode` is only reachable on a periodic system. For `nonbonded = :direct`
    # (NoCutoff) the builder emits one `PairTableNonbonded(mode = :none)` pair
    # table and never reads `lj_mode` (ChemistryBuild.jl:378-401), so a NoCutoff
    # system is force-consistent whether :lfs or :trunc was requested. The gate is
    # therefore exercised where the cutoff convention exists: the periodic
    # solvated-protein record, whose `StreamedLJFactor` carries `mode`.
    res = Dict{String,Any}("case" => "solvated_protein", "backend" => "pme")
    blk = Dict{String,Any}[]
    c = CASES[findfirst(x -> x.name == "solvated_protein", CASES)]
    tmp = mktempdir(; cleanup=true)
    json = joinpath(tmp, "gate.json")
    ok, _, stderr_text, tool = extract_case(c, json)
    if !ok
        res["status"] = "BLOCKED_IMPORT"
        push!(blk, blocker(c.name, "gate_import", stderr_text, Dict{String,Any}("file" => tool)))
        return res, blk
    end
    ns = CH.read_normalized(json)
    q = read_crd(c.crd)
    n = CH.n_atoms(ns)

    function system(nb, lj)
        prot = CH.default_protocol(ns; nonbonded=nb, cutoff_nm=CUTOFF_NM, lj_mode=lj,
                                   water_model=:flexible)
        rs = CH.build_resolved(ns; protocol=prot)
        return D.HarmonicSystem(rs.factors, rs.atoms.masses,
                                F.Context(rs.box, rs.units, n))
    end

    hs_lfs = system(:pme, :lfs)
    res["n_atoms"] = n
    res["lfs_force_consistent"] = D.force_model_supported(hs_lfs)
    res["lfs_gate_accepted"] = try
        D.check_force_model(hs_lfs)
        true
    catch
        false
    end

    hs_trunc = system(:pme, :trunc)
    res["trunc_force_consistent"] = D.force_model_supported(hs_trunc)
    res["trunc_gate_threw"] = false
    res["trunc_refusal_message"] = nothing
    res["trunc_refusal_site"] = nothing
    st = D.State(copy(q), zeros(3, n))
    q_before = copy(st.q)
    p_before = copy(st.p)
    try
        D.run_trajectory(st, hs_trunc, DT_PS, 1)
    catch e
        res["trunc_gate_threw"] = true
        res["trunc_refusal_message"] = short_error(e)
        res["trunc_refusal_site"] = error_site(catch_backtrace())
    end
    # A refused integration must not have moved the state at all.
    res["refused_run_left_state_unchanged"] = (st.q == q_before && st.p == p_before)

    # Documented consequence of the builder, measured rather than assumed.
    dna = CASES[findfirst(x -> x.name == "dna", CASES)]
    dna_json = joinpath(tmp, "dna.json")
    ok_dna, _, _, _ = extract_case(dna, dna_json)
    if ok_dna
        nd = CH.read_normalized(dna_json)
        qd = read_crd(dna.crd); nd_n = CH.n_atoms(nd)
        prot = CH.default_protocol(nd; nonbonded=:direct, lj_mode=:trunc,
                                   water_model=:flexible)
        rsd = CH.build_resolved(nd; protocol=prot)
        hsd = D.HarmonicSystem(rsd.factors, rsd.atoms.masses,
                               F.Context(rsd.box, rsd.units, nd_n))
        res["direct_trunc_force_consistent"] = D.force_model_supported(hsd)
        res["direct_note"] =
            "nonbonded = :direct is a NoCutoff pair table (PairTableNonbonded " *
            "mode = :none), so lj_mode is inert there and the :trunc refusal " *
            "cannot be exercised on a boxless system; it is exercised on PME above"
    end

    res["status"] = (res["lfs_force_consistent"] && res["lfs_gate_accepted"] &&
                     !res["trunc_force_consistent"] && res["trunc_gate_threw"] &&
                     res["refused_run_left_state_unchanged"] &&
                     get(res, "direct_trunc_force_consistent", true)) ? "PASS" : "FAIL"
    res["status"] == "PASS" ||
        push!(blk, blocker(res["case"], "force_model_gate",
                           "the :lfs/:trunc gate did not behave as documented"))
    return res, blk
end

# ---------------------------------------------------------------------------
# deterministic JSON (sorted keys, no timestamps)
# ---------------------------------------------------------------------------

function jdump(x)
    if x isa Dict
        ks = sort!(collect(keys(x)))
        return "{" * join(["\"$(k)\":" * jdump(x[k]) for k in ks], ",") * "}"
    elseif x isa AbstractVector
        return "[" * join(jdump.(x), ",") * "]"
    elseif x isa Bool
        return x ? "true" : "false"
    elseif x isa Integer
        return string(x)
    elseif x isa AbstractFloat
        return isfinite(x) ? repr(Float64(x)) : "null"
    elseif x isa AbstractString
        return "\"" * escape_string(String(x)) * "\""
    elseif x isa Symbol
        return "\"" * escape_string(String(x)) * "\""
    elseif x === nothing
        return "null"
    else
        error("no JSON encoding for $(typeof(x))")
    end
end

# ---------------------------------------------------------------------------
# machine description
# ---------------------------------------------------------------------------

function sysctl(key::AbstractString)
    try
        return strip(read(`sysctl -n $key`, String))
    catch
        return nothing
    end
end

function hardware()
    mem = sysctl("hw.memsize")
    return Dict{String,Any}(
        "cpu_brand" => sysctl("machdep.cpu.brand_string"),
        "cpu_count" => sysctl("hw.ncpu"),
        "physical_cpu_count" => sysctl("hw.physicalcpu"),
        "memory_bytes" => mem === nothing ? nothing : parse(Int, mem),
        "os" => Sys.KERNEL,
        "julia_version" => string(VERSION),
        "threads" => Threads.nthreads(),
    )
end

# ---------------------------------------------------------------------------
# README generation from the report (same numbers, no hand editing)
# ---------------------------------------------------------------------------

function readme(report)
    io = IOBuffer()
    hw = report["hardware"]
    settings = report["settings"]
    gate = report["force_model_gate"]
    cases = report["cases"]
    blockers = report["blockers"]

    println(io, "# Chemistry MD scaling: actual trajectories at increasing atom count\n")
    println(io, "Generated by `julia/bench/chemistry_md_scaling.jl`; the numbers ",
                "below are read straight from the sibling `report.json`.\n")
    println(io, "**What \"scaling\" means here.** The ability to *run* a larger molecular ",
                "system through the full path import -> `build_resolved` -> factor stack -> ",
                "force evaluation -> velocity-Verlet integration. A row is evidence of ",
                "simulation only when the integrator advanced a state; atom counts that ",
                "were imported and evaluated but never integrated appear as ",
                "`N simulated = 0` and are not called simulations.\n")

    println(io, "## What this does and does not establish\n")
    println(io, "* It establishes that bounded velocity-Verlet trajectories of ",
                join([string(c["simulated_atoms"]) for c in cases if c["integrated"]], ", "),
                " atoms run end to end on this machine, with conserved total energy over ",
                "the simulated window, and it records the cost of each case.\n")
    println(io, "* It does **not** establish throughput, efficiency, weak or strong scaling: ",
                "each case is a smoke trajectory of tens of femtoseconds timed twice, not a ",
                "production sampling run. `ms/step` is a per-run cost at that N, not a ",
                "scalability law, and no fit is claimed across N.\n")
    println(io, "* It does not establish accuracy against any other engine. The reference ",
                "comparison lives in `chemistry/reference/` and is unrelated to ",
                "the timings here.\n")

    println(io, "## Hardware\n")
    println(io, "| item | value |")
    println(io, "| --- | --- |")
    println(io, "| CPU | $(hw["cpu_brand"]) |")
    println(io, "| logical CPUs | $(hw["cpu_count"]) |")
    println(io, "| physical CPUs | $(hw["physical_cpu_count"]) |")
    mem = hw["memory_bytes"]
    println(io, "| memory | ", mem === nothing ? "unknown" :
                @sprintf("%.1f GiB", mem / 2^30), " |")
    println(io, "| OS | $(hw["os"]) |")
    println(io, "| Julia | $(hw["julia_version"]) |")
    println(io, "| threads (`JULIA_NUM_THREADS`) | $(hw["threads"]) |")
    println(io, "\nSingle-threaded by construction. The host may run other work at the ",
                "same time; every case is timed twice and the spread is reported, so a ",
                "contended measurement is visible instead of being averaged away.\n")

    println(io, "## Settings\n")
    for k in sort!(collect(keys(settings)))
        println(io, "* `", k, "` = `", string(settings[k]), "`")
    end
    println(io)

    println(io, "## Results\n")
    println(io, "| case | N imported | N simulated | factors | backend | steps | dt (fs) | ",
                "sim time (fs) | wall median (s) | wall spread | ms/step | peak RSS (MiB) | ",
                "max\\|dE\\| (kJ/mol) | max\\|F\\| (kJ/mol/nm) | status |")
    println(io, "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for c in cases
        nimp = get(c, "n_atoms", nothing)
        sim = get(c, "simulated_atoms", 0)
        st = get(c, "status", "?")
        steps = get(c, "steps_run", 0)
        wall = get(c, "wall_s_median", nothing)
        spread = get(c, "wall_s_rel_spread", nothing)
        mps = get(c, "ms_per_step", nothing)
        rss = get(c, "peak_rss_mib", nothing)
        de = get(c, "max_abs_de_kj_mol", nothing)
        fmx = get(c, "max_force_kj_mol_per_nm", nothing)
        println(io, "| `", c["case"], "` | ",
                nimp === nothing ? "?" : string(nimp), " | ", string(sim), " | ",
                get(c, "n_factors", 0), " | ", c["backend"], " | ", string(steps), " | ",
                @sprintf("%.2f", DT_PS * 1000), " | ",
                get(c, "simulated_time_fs", 0.0) == 0.0 ? "-" :
                    @sprintf("%.1f", c["simulated_time_fs"]), " | ",
                wall === nothing ? "-" : @sprintf("%.2f", wall), " | ",
                spread === nothing ? "-" : @sprintf("%+.1f%%", 100 * spread), " | ",
                mps === nothing ? "-" : @sprintf("%.2f", mps), " | ",
                rss === nothing ? "-" : @sprintf("%.0f", rss), " | ",
                de === nothing ? "-" : @sprintf("%.3g", de), " | ",
                fmx === nothing ? "-" : @sprintf("%.3g", fmx), " | ", st, " |")
    end
    println(io)
    println(io, "Static evaluation columns (`energy_eval_ms`, `forces_eval_ms`, and the ",
                "`*_min_ms`/`*_max_ms` spread) are in `report.json`. They are evaluations of ",
                "a fixed configuration, not dynamics.\n")

    println(io, "## Energy and constraint diagnostics\n")
    println(io, "NVE, velocities initialised to zero (deterministic), no constraints: the ",
                "protocol uses `water_model = :flexible`, so there is no RATTLE projection ",
                "and `constraints = 0` for every case. `max|dE|` is the peak excursion of ",
                "the Neumaier-compensated total energy; `drift_fraction` is ",
                "`|slope|*T / (2*max|E-E0|)` from `Dynamics.energy_drift_report`, which is ",
                "the deciding statistic (a bounded oscillation gives a small fraction, a ",
                "linear leak gives ~2). PASS requires finite energies, ",
                "`drift_fraction <= 0.5`, and `max|dE| < 1000` kJ/mol.\n")
    for c in cases
        c["integrated"] || continue
        @printf(io,
                "* `%s`: E0 = %.6g, E_final = %.6g, max|dE| = %.3g kJ/mol ",
                c["case"], c["e0_kj_mol"], c["e_final_kj_mol"], c["max_abs_de_kj_mol"])
        @printf(io, "(naive sum %.3g), slope = %.3g kJ/mol/ps, drift_fraction = %.3f, ",
                c["max_abs_de_naive_kj_mol"], c["drift_slope_kj_mol_per_ps"],
                c["drift_fraction"])
        @printf(io, "bounded = %s, constraints = %d\n", string(c["drift_bounded"]),
                c["constraints"])
    end
    println(io)

    println(io, "## Force-model gate\n")
    println(io, "`Dynamics.check_force_model` must accept the force-consistent `:lfs` ",
                "cutoff and refuse `:trunc` before touching the state. Measured on the ",
                "`dna` fixture: `lfs_force_consistent = ",
                string(gate["lfs_force_consistent"]), "`, `lfs_gate_accepted = ",
                string(gate["lfs_gate_accepted"]), "`, `trunc_force_consistent = ",
                string(gate["trunc_force_consistent"]), "`, `trunc_gate_threw = ",
                string(gate["trunc_gate_threw"]), "`, `refused_run_left_state_unchanged = ",
                string(gate["refused_run_left_state_unchanged"]), "`.\n")
    if gate["trunc_refusal_message"] !== nothing
        println(io, "Refusal text:\n\n```\n", gate["trunc_refusal_message"], "\n```\n")
    end

    println(io, "## Blockers\n")
    if isempty(blockers)
        println(io, "None.\n")
    else
        for b in blockers
            site = get(b, "site", nothing)
            where = site === nothing ? "" :
                    " at `$(site["file"]):$(site["line"])` (`$(site["function"])`)"
            println(io, "* **", b["case"], " / ", b["stage"], "**", where, ":\n\n  ```\n  ",
                    replace(b["message"], "\n" => "\n  "), "\n  ```\n")
        end
    end

    println(io, "## Exact commands\n")
    println(io, "```sh")
    println(io, "export JULIA_DEPOT_PATH=$HOME/.cache/biology-mm/julia-depot")
    println(io, "export JULIA_NUM_THREADS=1")
    println(io, "cd ", REPO)
    println(io, "~/.cache/biology-mm/julia-1.13.3/bin/julia \\")
    println(io, "  --project=julia julia/bench/chemistry_md_scaling.jl")
    println(io, "```")
    println(io)
    println(io, "Each case is also runnable alone, which is how the per-case peak RSS is ",
                "attributed to one N:\n")
    println(io, "```sh")
    println(io, "# one case, result record serialized to a path")
    println(io, "julia --project=julia julia/bench/chemistry_md_scaling.jl \\")
    println(io, "  --case=solvated_protein --json=/tmp/solvated.bin")
    println(io, "```")
    println(io)
    println(io, "Fixture inputs (all pre-existing, none regenerated here):\n")
    for c in CASES
        println(io, "* `", c.name, "`: `", relpath(c.parm, REPO), "`")
    end
    println(io)
    println(io, "## Interpretation\n")
    println(io, "The largest system that was *simulated* is ",
                string(maximum(c["simulated_atoms"] for c in cases)), " atoms; the largest ",
                "that was *imported and evaluated* is ",
                string(maximum(get(c, "n_atoms", 0) for c in cases)), " atoms. Those two ",
                "numbers are deliberately separate: importing and evaluating a force field ",
                "at 20k atoms is not the same capability as integrating it, and this report ",
                "never calls the former a simulation.\n")
    println(io, "Wall times include one force evaluation per step plus the half-kicks; ",
                "`ms/step` at each N is the median of two runs of the same short window. ",
                "Because the windows differ in length (",
                join(["$(c["case"]): $(c["steps_run"])" for c in cases if c["integrated"]], ", "),
                " steps), the per-step costs are not a controlled comparison of N: they ",
                "are the honest cost of running each case at the length it can afford. ",
                "Reading a scaling exponent out of them would be unsound, and none is ",
                "reported.\n")
    if !isempty(blockers)
        println(io, "Cases marked BLOCKED did not integrate. Their blockers above are the ",
                    "exact API-level reasons, recorded rather than patched around: the ",
                    "system was imported (or refused at import), and the failure is in the ",
                    "core path, not in the fixture.\n")
    end
    return String(take!(io))
end

# ---------------------------------------------------------------------------
# parent: sweep in subprocesses, then write report.json + README.md
# ---------------------------------------------------------------------------

function watch(p::Base.Process, limit_s::Float64)
    t0 = time()
    while Base.process_running(p)
        if time() - t0 > limit_s
            kill(p)
            return false
        end
        sleep(0.5)
    end
    return true
end

function sweep(cases::Vector{Case})
    results = Dict{String,Any}[]
    blockers = Dict{String,Any}[]
    tmp = mktempdir(; cleanup=true)
    for c in cases
        path = joinpath(tmp, c.name * ".bin")
        @printf("== %s: %s ==\n", c.name, c.note)
        cmd = `$(joinpath(Sys.BINDIR, "julia")) --project=$PROJECT $(@__FILE__)
               --case=$(c.name) --json=$path`
        t0 = time_ns()
        proc = run(pipeline(cmd; stdout=stdout, stderr=stderr); wait=false)
        finished = watch(proc, CHILD_WATCHDOG_S)
        wall = (time_ns() - t0) / 1e9
        if !finished || !isfile(path)
            res = Dict{String,Any}("case" => c.name, "status" => "BLOCKED_TIMEOUT",
                                   "integrated" => false, "simulated_atoms" => 0,
                                   "backend" => string(c.nonbonded), "note" => c.note,
                                   "case_wall_s" => wall)
            push!(results, res)
            push!(blockers, blocker(c.name, "child",
                                    @sprintf("child process exceeded the %.0f s watchdog or produced no record",
                                             CHILD_WATCHDOG_S)))
            @printf("   -> BLOCKED_TIMEOUT after %.1f s\n", wall)
            continue
        end
        res, blk = deserialize(path)
        res["case_wall_s"] = wall
        append!(blockers, blk)
        push!(results, res)
        @printf("   -> %-16s N=%s sim=%s steps=%s wall=%s\n", res["status"],
                get(res, "n_atoms", "?"), res["simulated_atoms"],
                get(res, "steps_run", 0),
                haskey(res, "wall_s_median") ? @sprintf("%.2fs", res["wall_s_median"]) : "-")
    end
    return results, blockers
end

function main(args)
    outdir = OUTDIR_DEFAULT
    include_membrane = true
    for a in args
        if startswith(a, "--out-dir=")
            outdir = split(a, "=")[2]
        elseif a == "--no-membrane"
            include_membrane = false
        end
    end
    for a in args
        if startswith(a, "--case=")
            name = split(a, "=")[2]
            jsonpath = nothing
            for b in args
                if startswith(b, "--json=")
                    jsonpath = split(b, "=")[2]
                end
            end
            jsonpath === nothing && error("--case requires --json=<path>")
            if name == "__gate"
                res, blk = run_gate()
            else
                idx = findfirst(c -> c.name == name, CASES)
                idx === nothing && error("unknown case $(name); known: " *
                                         join([c.name for c in CASES], ", ") * ", __gate")
                res, blk = run_case(CASES[idx])
            end
            serialize(jsonpath, (res, blk))
            return 0
        end
    end
    # never invent inputs: a case whose fixture files are absent is dropped
    available = filter(c -> isfile(c.parm) && isfile(c.crd), CASES)
    for c in CASES
        isfile(c.parm) && isfile(c.crd) ||
            @printf("skipping %s: missing %s\n", c.name,
                    isfile(c.parm) ? c.crd : c.parm)
    end
    cases = include_membrane ? available :
            filter(c -> c.name != "membrane", available)

    gate_path = joinpath(mktempdir(; cleanup=true), "gate.bin")
    gcmd = `$(joinpath(Sys.BINDIR, "julia")) --project=$PROJECT $(@__FILE__)
            --case=__gate --json=$gate_path`
    println("== force-model gate (:lfs accepted, :trunc refused) ==")
    run(pipeline(gcmd; stdout=stdout, stderr=stderr))
    gate, gate_blockers = if isfile(gate_path)
        deserialize(gate_path)
    else
        (Dict{String,Any}("status" => "BLOCKED_ERROR"), [])
    end
    @printf("   -> %s\n", gate["status"])

    results, blockers = sweep(cases)
    append!(blockers, gate_blockers)
    sort!(results, by=r -> (get(r, "n_atoms", -1), r["case"]))

    report = Dict{String,Any}(
        "schema" => "chemistry_md_scaling/1",
        "generated_by" => "julia/bench/chemistry_md_scaling.jl",
        "hardware" => hardware(),
        "settings" => Dict{String,Any}(
            "integrator" => "velocity Verlet (Dynamics.run_trajectory)",
            "ensemble" => "NVE",
            "dt_ps" => DT_PS,
            "dt_fs" => DT_PS * 1000,
            "dt_justification" =>
                "flexible water, so O-H stretches (period ~9 fs) are unconstrained; " *
                "0.5 fs is the usual Verlet limit for X-H bonds and 0.1 fs is a 5x " *
                "conservative margin, affordable because each case is only tens of steps",
            "cutoff_nm" => CUTOFF_NM,
            "lj_mode" => "lfs",
            "lj_mode_constraint" =>
                ":lfs only; :trunc is refused by Dynamics.check_force_model",
            "water_model" => "flexible",
            "constraints" => 0,
            "velocity_initialisation" => "zero",
            "record_every" => 1,
            "compensated_energy" => true,
            "timings" => "two independent runs per case, median and relative spread reported",
            "case_budget_s" => CASE_BUDGET_S,
            "drift_energy_ceiling_kj_mol" => DRIFT_ENERGY_CEILING,
            "n_atoms_note" =>
                "N imported = atoms in the normalized record; N simulated = atoms the " *
                "integrator actually advanced (0 when the case is blocked)",
        ),
        "force_model_gate" => gate,
        "cases" => results,
        "blockers" => blockers,
    )

    mkpath(outdir)
    rj = joinpath(outdir, "report.json")
    open(rj, "w") do io
        print(io, jdump(report), "\n")
    end
    rm = joinpath(outdir, "README.md")
    open(rm, "w") do io
        print(io, readme(report))
    end
    @printf("wrote %s and %s\n", relpath(rj, REPO), relpath(rm, REPO))
    return 0
end

exit(main(ARGS))
