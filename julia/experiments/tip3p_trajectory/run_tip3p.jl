#!/usr/bin/env julia
#
# Rigid TIP3P water box: relaxation -> NVT equilibration -> NVT production.
#
# Run one tier per process (so `Sys.maxrss()` is a clean peak-RSS measurement):
#
#   cd julia
#   JULIA_DEPOT_PATH=~/.cache/biology-mm/julia-depot \
#   ~/.cache/biology-mm/julia-1.13.3/bin/julia --project=. \
#       experiments/tip3p_trajectory/run_tip3p.jl --waters 200 \
#       --equil-ps 20 --prod-ps 20 --dt-fs 2 --gamma 1 \
#       --chunk-ps 5 --temp-gate warn \
#       --nve-ladder 0.5,1.0,2.0 --outdir experiments/tip3p_trajectory/runs/w200_corrected
#
# Order of operations, and why: relaxation -> equilibration -> TEMPERATURE GATE
# -> production -> checkpoint replay -> NVE check -> dt ladder -> report. The
# gate (last half of equilibration, IAT-corrected SE, 3 sigma or 3 K) is
# evaluated *before* production, so a run whose thermostat never settled cannot
# be presented as a trusted production window; `--temp-gate abort` refuses to
# produce, the default `warn` records the outcome in report.txt/report.json and
# continues. The gate only ever sets `temperature_trusted`; `rdf_diffusion_trusted`
# and `equilibrium_claim` stay false because no independent equilibration
# criteria are implemented here.
#
# `--chunk-ps` splits production into restartable chunks; each boundary writes
# `checkpoint.bin` (previous kept as `checkpoint.bin.prev`) and the final chunk
# is replayed from it to check exact restart (`max|dq|`, `max|dp|`). The
# production bookkeeping lives in this runner, not in `WaterBox.production!`:
# frames, COM stream, MSD, RDF and the diffusion fit use the module's own
# helpers, but the COM is mass-weighted per molecule (the module historically
# divided by `N*m_O`; that denominator was fixed in `src/WaterBox.jl` at
# 20:43 on 2026-11-12, after the w200_corrected run started).
#
# Every number written to the report is measured in this process. Nothing here
# extrapolates or assumes a target temperature is reached.

using FoundationCore
using Printf
using Random
using Serialization
using Statistics

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
isdefined(FoundationCore, :WaterBox) ||
    Base.include(FoundationCore, joinpath(ROOT, "src", "WaterBox.jl"))
const WB = FoundationCore.WaterBox
const D = FoundationCore.Dynamics

function parse_args(args)
    opts = Dict{String,String}(
        "waters" => "216",
        "equil-ps" => "20.0",
        "prod-ps" => "100.0",
        "dt-fs" => "1.0",
        "gamma" => "5.0",
        "temp" => "300.0",
        "seed" => "11",
        "density" => "1.0",
        "rcut" => "",
        "pme-rcut" => "",
        "relax-steps" => "1500",
        "frame-ps" => "0.5",
        "sample-every" => "10",
        "rdf-frames" => "40",
        "rdf-rmax" => "1.2",
        "nve-steps" => "1000",
        "nve-dt-fs" => "1.0",
        "nve-ladder" => "0.5,1.0,2.0",
        "nve-ladder-ps" => "1.0",
        "chunk-ps" => "0.0",
        "checkpoint" => "1",
        "replay-check" => "1",
        "temp-gate" => "warn",
        "outdir" => joinpath("experiments", "tip3p_trajectory", "runs", "default"),
        "label" => "",
    )
    i = 1
    while i <= length(args)
        a = args[i]
        startswith(a, "--") || error("unexpected argument $a")
        key = a[3:end]
        i += 1
        i <= length(args) || error("missing value for --$key")
        opts[key] = args[i]
        i += 1
    end
    return opts
end

nf(opts, k) = parse(Float64, opts[k])
ni(opts, k) = parse(Int, opts[k])
maybe_f(opts, k) = isempty(opts[k]) ? nothing : parse(Float64, opts[k])

# ---------------------------------------------------------------------------
# production bookkeeping + checkpoints
# ---------------------------------------------------------------------------
#
# `WaterBox.production!` divides the molecular centre of mass by
# `sum(m_O over molecules)` instead of by the molecule mass, so its COM — and
# therefore its MSD, diffusion coefficient and `com_unwrapped.csv` — is too
# small by the factor `N * m_O / M_mol`. That is an engine-side defect in a file
# this experiment does not own (`src/WaterBox.jl`), so this runner does the
# production bookkeeping itself, driving the module's own BAOAB step
# (`WB.integrate!`) and reusing its frame writer, RDF accumulator and diffusion
# fit. The energy samples, frame grid and RDF normalisation are identical to
# `production!`; the COM stream is the mass-weighted molecular COM.

"Molecular centres of mass (3 x n, nm) from unwrapped coordinates."
function com_matrix(q::AbstractMatrix, w)
    m = w.masses
    out = Matrix{Float64}(undef, 3, length(w.molecules))
    for (i, (o, h1, h2)) in enumerate(w.molecules)
        M = m[o] + m[h1] + m[h2]
        for d in 1:3
            out[d, i] = (m[o] * q[d, o] + m[h1] * q[d, h1] + m[h2] * q[d, h2]) / M
        end
    end
    return out
end

"""
Restartable integrator + bookkeeping state. `Serialization` round-trips the
`MersenneTwister` state exactly, so a replay from a checkpoint is bit-for-bit
the same trajectory.
"""
struct Checkpoint
    q::Matrix{Float64}
    p::Matrix{Float64}
    rng::Random.MersenneTwister
    step::Int
    time_ps::Float64
    temperature::Float64
    com0::Union{Nothing,Matrix{Float64}}
    msd_t::Vector{Float64}
    msd_v::Vector{Float64}
    rdf_counts::Vector{Int}
    rdf_done::Int
    nframes::Int
end

function write_checkpoint(path::AbstractString, ck::Checkpoint)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        serialize(io, ck)
    end
    mv(tmp, path; force = true)
    return path
end

read_checkpoint(path::AbstractString) = open(deserialize, path)

"""
    production_loop!(traj; ...) -> (ProductionResult, last_chunk_steps)

Chunked production window built on `WB.integrate!`. Every chunk boundary writes
`checkpoint.bin` (the previous one is kept as `checkpoint.bin.prev`), so a long
run is restartable and the final chunk can be replayed for an exactness check.
"""
function production_loop!(traj::WB.WaterTrajectory; ps::Real, dt::Real,
                          temperature::Real, gamma::Real,
                          sample_every::Integer, frame_writer, frame_every::Integer,
                          rdf_frames::Integer, rdf_every::Integer,
                          rmax::Real = 1.2, nbins::Integer = 300,
                          com_io::Union{Nothing,IO} = nothing,
                          chunk_ps::Real = 0.0,
                          ckpt_path::Union{Nothing,String} = nothing)
    w = traj.water
    n = length(w.molecules)
    dtf = Float64(dt)
    total_steps = max(1, round(Int, Float64(ps) / dtf))
    # An RDF bin at r is only meaningful if its whole spherical shell is inside
    # the minimum-image sphere of radius min(L)/2; beyond that the orthorhombic
    # minimum image and the shell normalisation are both invalid.
    rmax_req = Float64(rmax)
    rmax = min(rmax_req, 0.5 * minimum(w.box.L))
    if rmax < rmax_req
        @printf("RDF cutoff clamped: requested %.3f nm -> %.3f nm (= min(L)/2, NVT box)\n",
                rmax_req, rmax)
    end
    traj.temperature = Float64(temperature)
    nb = Int(nbins)
    dr = Float64(rmax) / nb
    counts = zeros(Int, nb)
    rdf_done = 0
    nframes = 0
    samples = WB.Sample[]
    msd_t = Float64[]
    msd_v = Float64[]
    com0 = nothing
    wall = 0.0
    chunk_steps = Float64(chunk_ps) > 0 ?
        max(Int(sample_every), round(Int, Float64(chunk_ps) / dtf)) : total_steps
    chunk_steps = (chunk_steps ÷ Int(sample_every)) * Int(sample_every)
    chunk_steps = max(chunk_steps, Int(sample_every))

    function sample_hook(tr, _sample)
        if tr.step % frame_every == 0
            nframes += 1
            if frame_writer !== nothing
                D.write_frame!(frame_writer, WB.molecule_frame(tr.st.q, w), w.box,
                               tr.step, tr.time_ps)
            end
            com = com_matrix(tr.st.q, w)
            if com0 === nothing
                com0 = copy(com)
            end
            msd = 0.0
            @inbounds for i in 1:n
                msd += (com[1, i] - com0[1, i])^2 + (com[2, i] - com0[2, i])^2 +
                       (com[3, i] - com0[3, i])^2
            end
            push!(msd_t, tr.time_ps)
            push!(msd_v, msd / n)
            if com_io !== nothing
                for i in 1:n
                    @printf(com_io, "%.6f,%d,%.10f,%.10f,%.10f\n", tr.time_ps, i,
                            com[1, i], com[2, i], com[3, i])
                end
            end
            if rdf_done < rdf_frames && (nframes % max(1, Int(rdf_every)) == 0)
                WB._accumulate_oo_rdf!(counts, tr.st.q, w, Float64(rmax), dr)
                rdf_done += 1
            end
        end
        return nothing
    end

    budget = total_steps
    last_steps = 0
    while budget > 0
        nst = min(chunk_steps, budget)
        nst = (nst ÷ Int(sample_every)) * Int(sample_every)
        if nst == 0
            nst = budget          # final sub-sample remainder
        end
        t0 = time()
        smp = WB.integrate!(traj, nst; dt = dtf, temperature = temperature,
                            gamma = gamma, sample_every = Int(sample_every),
                            on_sample = sample_hook)
        wall += time() - t0
        append!(samples, smp)
        budget -= nst
        last_steps = nst
        if ckpt_path !== nothing
            ck = Checkpoint(copy(traj.st.q), copy(traj.st.p), copy(traj.rng),
                            traj.step, traj.time_ps, traj.temperature,
                            com0 === nothing ? nothing : copy(com0),
                            copy(msd_t), copy(msd_v), copy(counts), rdf_done, nframes)
            prev = ckpt_path * ".prev"
            isfile(ckpt_path) && mv(ckpt_path, prev; force = true)
            write_checkpoint(ckpt_path, ck)
        end
    end

    if !isempty(msd_t)
        t0 = msd_t[1]
        msd_t = msd_t .- t0      # report the production window from 0 ps
    end
    r = WB._rdf_axis(Float64(rmax), nb)
    g = WB._rdf_g(counts, r, n, prod(w.box.L), max(rdf_done, 1))
    D_, Dse, win = WB.diffusion_from_msd(msd_t, msd_v)
    prodres = WB.ProductionResult(samples, total_steps, total_steps * dtf, dtf, wall,
                                  msd_t, msd_v, r, g, rdf_done, D_, Dse, win)
    return prodres, last_steps
end

"Max |r_OH - r0| over all molecules (nm) for a configuration."
function worst_bond_error(q::AbstractMatrix, w)
    r0 = WB.TIP3P_PARAMETERS.r_OH
    worst = 0.0
    for (o, h1, h2) in w.molecules
        worst = max(worst,
                    abs(sqrt(sum(abs2, q[:, h1] .- q[:, o])) - r0),
                    abs(sqrt(sum(abs2, q[:, h2] .- q[:, o])) - r0))
    end
    return worst
end

"""
Replay the final production chunk from `checkpoint.bin.prev` and compare with the
actual final state. Returns a NamedTuple (dq, dp, step_mismatch, time_mismatch,
bond_err, ok).
"""
function replay_check(prev_path::AbstractString, final_state, water, field;
                      dt::Real, temperature::Real, gamma::Real, nsteps::Integer)
    ck = read_checkpoint(prev_path)
    tr = WB.WaterTrajectory(water, field; temperature = temperature, seed = 1)
    tr.st.q .= ck.q
    tr.st.p .= ck.p
    tr.rng = ck.rng
    tr.step = ck.step
    tr.time_ps = ck.time_ps
    tr.temperature = ck.temperature
    tr.st.force_valid = false
    WB.integrate!(tr, nsteps; dt = dt, temperature = temperature, gamma = gamma,
                  sample_every = nsteps)
    dq = maximum(abs, tr.st.q .- final_state.q)
    dp = maximum(abs, tr.st.p .- final_state.p)
    step_mismatch = tr.step - final_state.step
    time_mismatch = tr.time_ps - final_state.time_ps
    bond_err = worst_bond_error(ck.q, water)
    ok = dq == 0.0 && dp == 0.0 && step_mismatch == 0 && time_mismatch == 0.0
    return (dq = dq, dp = dp, step_mismatch = step_mismatch,
            time_mismatch = time_mismatch, bond_err = bond_err, ok = ok)
end

function main()
    opts = parse_args(ARGS)
    n = ni(opts, "waters")
    dt = nf(opts, "dt-fs") / 1000.0
    temp = nf(opts, "temp")
    gamma = nf(opts, "gamma")
    seed = ni(opts, "seed")
    outdir = abspath(opts["outdir"])
    mkpath(outdir)
    label = isempty(opts["label"]) ? basename(outdir) : opts["label"]

    @printf("=== TIP3P NVT: %d molecules, T=%.1f K, dt=%.3f fs, gamma=%.1f/ps ===\n",
            n, temp, dt * 1000, gamma)

    t_build = @elapsed water = WB.tip3p_water(n; density = nf(opts, "density"), seed = seed)
    field = WB.water_field(water; rcut = maybe_f(opts, "rcut"),
                           pme_rcut = maybe_f(opts, "pme-rcut"))
    @printf("built %d molecules in %.2f s: L=(%.4f, %.4f, %.4f) nm, LJ rcut=%.3f nm, PME rcut=%.3f nm, grid=%dx%dx%d order %d, alpha=%.2f 1/nm\n",
            n, t_build, water.box.L..., field.rcut, field.pme_rcut,
            field.grid..., field.order, field.alpha)

    traj = WB.WaterTrajectory(water, field; temperature = temp, seed = seed)

    # ---- relaxation -------------------------------------------------------
    relax = WB.relax!(traj; max_steps = ni(opts, "relax-steps"))
    @printf("relaxation: %d accepted steps, U %.1f -> %.1f kJ/mol (%.2f per molecule), max|F_tangent| %.1f, converged=%s\n",
            relax.steps, relax.u_initial, relax.u_final, relax.u_final / n,
            relax.max_tangent_force_final, relax.converged)

    # ---- equilibration ----------------------------------------------------
    equil = WB.equilibrate!(traj; ps = nf(opts, "equil-ps"), dt = dt,
                            temperature = temp, gamma = gamma,
                            sample_every = ni(opts, "sample-every"))
    @printf("equilibration: %.2f ps, T = %.2f +/- %.2f K, wall %.1f s\n",
            equil.ps, equil.temp_mean, equil.temp_se, equil.wall_s)

    # ---- temperature gate: evaluated BEFORE production ---------------------
    # The thermostat must have settled before the production window is trusted.
    # `--temp-gate abort` refuses to produce; the default `warn` records the
    # decision and continues, so a bounded diagnostic still yields artefacts.
    eq_tail = equil.samples[(length(equil.samples) ÷ 2 + 1):end]
    eq_tail_T, eq_tail_se, eq_tail_g = WB.mean_with_error([s.T_inst for s in eq_tail])
    temp_reached = abs(eq_tail_T - temp) <= max(3 * eq_tail_se, 3.0)
    @printf("temperature gate (pre-production, last %.2f ps of equilibration): T = %.2f +/- %.2f K (g = %.2f); target %.1f K reached at 3 sigma: %s\n",
            equil.ps / 2, eq_tail_T, eq_tail_se, eq_tail_g, temp, temp_reached)
    if !temp_reached
        if opts["temp-gate"] == "abort"
            error("equilibration temperature $(round(eq_tail_T, digits=2)) +/- $(round(eq_tail_se, digits=2)) K did not reach $(temp) K at 3 sigma; refusing production (--temp-gate abort)")
        else
            @printf("  WARNING: production continues (--temp-gate warn); RDF/diffusion are marked as not trusted in the report.\n")
        end
    end

    # ---- production -------------------------------------------------------
    frame_ps = nf(opts, "frame-ps")
    sample_every = ni(opts, "sample-every")
    frame_every = max(1, round(Int, frame_ps / dt))
    frame_every % sample_every == 0 ||
        error("--frame-ps must be a multiple of sample-every*dt = $(sample_every * dt) ps")
    trajfile = joinpath(outdir, "traj.xyz")
    writer = D.FrameWriter(trajfile, 3n; labels = repeat(["O", "H", "H"], n))
    comio = open(joinpath(outdir, "com_unwrapped.csv"), "w")
    println(comio, "time_ps,molecule,x_nm,y_nm,z_nm")
    ckpt_path = ni(opts, "checkpoint") == 1 ? joinpath(outdir, "checkpoint.bin") : nothing
    prod, last_chunk_steps = production_loop!(traj; ps = nf(opts, "prod-ps"), dt = dt,
                          temperature = temp, gamma = gamma,
                          sample_every = sample_every,
                          frame_writer = writer, frame_every = frame_every,
                          rdf_frames = ni(opts, "rdf-frames"), rdf_every = 1,
                          rmax = nf(opts, "rdf-rmax"),
                          com_io = comio, chunk_ps = nf(opts, "chunk-ps"),
                          ckpt_path = ckpt_path)
    close(writer)
    close(comio)
    @printf("production: %d steps = %.4f ns, wall %.1f s, %.2f ns/day, %d frames, %d MSD/RDF samples\n",
            prod.steps, prod.ps / 1000, prod.wall_s,
            (prod.ps / 1000) / (prod.wall_s / 86400), length(prod.msd_time_ps),
            prod.rdf_frames)
    rdf_rmax_requested = nf(opts, "rdf-rmax")
    rdf_rmax_valid = min(rdf_rmax_requested, 0.5 * minimum(water.box.L))

    # ---- restart / checkpoint replay check ---------------------------------
    restart = nothing
    if ckpt_path !== nothing && ni(opts, "replay-check") == 1 &&
       isfile(ckpt_path * ".prev")
        final_state = (q = copy(traj.st.q), p = copy(traj.st.p), step = traj.step,
                       time_ps = traj.time_ps)
        restart = replay_check(ckpt_path * ".prev", final_state, water, field;
                               dt = dt, temperature = temp, gamma = gamma,
                               nsteps = last_chunk_steps)
        @printf("restart check: replayed final %d-step chunk from checkpoint (.prev): max|dq| = %.3g, max|dp| = %.3g, step/time match = %s, checkpoint bond error %.2e nm\n",
                last_chunk_steps, restart.dq, restart.dp,
                (restart.step_mismatch == 0 && restart.time_mismatch == 0.0), restart.bond_err)
    end

    # ---- NVE shadow-energy check -----------------------------------------
    nve_band, nve_slope, nve_se, nve_steps =
        WB.nve_energy_band(traj; nsteps = ni(opts, "nve-steps"),
                           dt = nf(opts, "nve-dt-fs") / 1000.0)

    # ---- timestep ladder: same simulated time at each dt -------------------
    ladder = NamedTuple{(:dt_fs, :dt_ps, :steps, :sim_ps, :band, :slope, :se),Tuple{Float64,Float64,Int,Float64,Float64,Float64,Float64}}[]
    if !isempty(opts["nve-ladder"])
        ladder_ps = nf(opts, "nve-ladder-ps")
        for tok in split(opts["nve-ladder"], ",")
            isempty(strip(tok)) && continue
            dt_fs = parse(Float64, tok)
            nst = max(10, round(Int, ladder_ps / (dt_fs / 1000.0)))
            b, s, e, n_used = WB.nve_energy_band(traj; nsteps = nst,
                                                 dt = dt_fs / 1000.0)
            push!(ladder, (dt_fs = dt_fs, dt_ps = dt_fs / 1000.0, steps = n_used,
                           sim_ps = n_used * dt_fs / 1000.0, band = b, slope = s,
                           se = e))
            @printf("NVE ladder: dt=%.2f fs, %d steps = %.2f ps: band %.4g kJ/mol, slope %.4g +/- %.4g kJ/mol/ps\n",
                    dt_fs, n_used, n_used * dt_fs / 1000.0, b, s, e)
        end
    end

    rep = WB.summarize_run(; label = label, water = water, field = field,
                           relax = relax, equil = equil, prod = prod,
                           nve = (band = nve_band, slope = nve_slope,
                                  se = nve_se, steps = nve_steps))

    # ---- write artefacts --------------------------------------------------
    open(joinpath(outdir, "series.csv"), "w") do io
        println(io, "step,time_ps,U_kJ_mol,K_kJ_mol,E_kJ_mol,T_K")
        for s in prod.samples
            @printf(io, "%d,%.6f,%.10f,%.10f,%.10f,%.6f\n", s.step, s.time_ps, s.U,
                    s.K, s.E, s.T_inst)
        end
    end
    open(joinpath(outdir, "equil_series.csv"), "w") do io
        println(io, "step,time_ps,U_kJ_mol,K_kJ_mol,E_kJ_mol,T_K")
        for s in equil.samples
            @printf(io, "%d,%.6f,%.10f,%.10f,%.10f,%.6f\n", s.step, s.time_ps, s.U,
                    s.K, s.E, s.T_inst)
        end
    end
    open(joinpath(outdir, "msd_oo.csv"), "w") do io
        println(io, "time_ps,msd_nm2")
        for k in eachindex(prod.msd_time_ps)
            @printf(io, "%.6f,%.10f\n", prod.msd_time_ps[k] - prod.msd_time_ps[1],
                    prod.msd_nm2[k])
        end
    end
    open(joinpath(outdir, "rdf_oo.csv"), "w") do io
        println(io, "r_nm,g_oo")
        for k in eachindex(prod.rdf_r)
            @printf(io, "%.6f,%.10f\n", prod.rdf_r[k], prod.rdf_g[k])
        end
    end
    open(joinpath(outdir, "nve_ladder.csv"), "w") do io
        println(io, "dt_fs,dt_ps,steps,sim_ps,band_kJ_mol,slope_kJ_mol_per_ps,slope_se_kJ_mol_per_ps")
        for r in ladder
            @printf(io, "%.3f,%.6f,%d,%.6f,%.10g,%.10g,%.10g\n",
                    r.dt_fs, r.dt_ps, r.steps, r.sim_ps, r.band, r.slope, r.se)
        end
    end
    open(joinpath(outdir, "report.txt"), "w") do io
        WB.print_report(io, rep)
        println(io)
        @printf(io, "  build         %.2f s (one process, %d molecules)\n", t_build, n)
        @printf(io, "  notes         dt=%.3f fs, gamma=%.1f/ps, seed=%d, frames every %.2f ps\n",
                dt * 1000, gamma, seed, frame_ps)
        @printf(io, "  equilibration tail  %.2f ps: T = %.2f +/- %.2f K (IAT g = %.2f, %d samples)\n",
                equil.ps / 2, eq_tail_T, eq_tail_se, eq_tail_g, length(eq_tail))
        @printf(io, "  temperature gate  evaluated BEFORE production: requested %.1f K; tail within 3 sigma (or 3 K): %s\n",
                temp, temp_reached)
        @printf(io, "  trust         temperature_trusted=%s; rdf_diffusion_trusted=false, equilibrium_claim=false\n",
                temp_reached)
        @printf(io, "                (the temperature gate alone never sets RDF/D trust; no independent equilibration criteria are implemented)\n")
        if !temp_reached
            @printf(io, "  caveat        temperature gate failed; this window is not equilibrated\n")
        end
        if restart !== nothing
            @printf(io, "  restart       replayed final %d-step chunk from checkpoint.bin.prev\n",
                    last_chunk_steps)
            @printf(io, "                max|dq| = %.3g nm, max|dp| = %.3g amu nm/ps, step/time match = %s\n",
                    restart.dq, restart.dp,
                    restart.step_mismatch == 0 && restart.time_mismatch == 0.0)
            @printf(io, "                checkpoint bond error %.2e nm (r_OH)\n", restart.bond_err)
        end
        @printf(io, "  COM stream    mass-weighted molecular COM (production! divides by N*m_O)\n")
        @printf(io, "  RDF sampling  rmax requested %.3f nm, valid %.3f nm (= min(L)/2); g(r) beyond the valid range is not reported\n",
                rdf_rmax_requested, rdf_rmax_valid)
        @printf(io, "  performance   %.3f ms/step, %.1f steps/s (production loop incl. frame/COM/RDF I/O)\n",
                1000 * prod.wall_s / prod.steps, prod.steps / prod.wall_s)
        for r in ladder
            @printf(io, "  NVE ladder    dt=%.2f fs, %d steps = %.2f ps: band %.4g kJ/mol, slope %.4g +/- %.4g kJ/mol/ps\n",
                    r.dt_fs, r.steps, r.sim_ps, r.band, r.slope, r.se)
        end
    end
    open(joinpath(outdir, "report.json"), "w") do io
        kv = [
            ("label", "\"$(rep.label)\""),
            ("molecules", string(rep.molecules)),
            ("atoms", string(rep.atoms)),
            ("box_nm", "[$(rep.box_nm[1]),$(rep.box_nm[2]),$(rep.box_nm[3])]"),
            ("volume_nm3", string(rep.volume_nm3)),
            ("density_g_cm3", string(rep.density)),
            ("lj_rcut_nm", string(rep.rcut)),
            ("pme_rcut_nm", string(rep.pme_rcut)),
            ("alpha_per_nm", string(rep.alpha)),
            ("grid", "[$(rep.grid[1]),$(rep.grid[2]),$(rep.grid[3])]"),
            ("order", string(rep.order)),
            ("relax_steps", string(rep.relax_steps)),
            ("relax_u_initial", string(rep.relax_u_initial)),
            ("relax_u_final", string(rep.relax_u_final)),
            ("equil_ps", string(rep.equil_ps)),
            ("equil_temp_mean_K", string(rep.equil_temp)),
            ("equil_temp_se_K", string(rep.equil_temp_se)),
            ("prod_steps", string(rep.prod_steps)),
            ("prod_ns", string(rep.prod_ns)),
            ("prod_wall_s", string(rep.prod_wall_s)),
            ("ns_per_day", string(rep.ns_per_day)),
            ("peak_rss_mib", string(rep.peak_rss_mib)),
            ("drift_slope_kJ_mol_per_ps", string(rep.drift_slope)),
            ("drift_slope_se_kJ_mol_per_ps", string(rep.drift_se)),
            ("drift_total_kJ_mol", string(rep.drift_window_kj)),
            ("temp_mean_K", string(rep.temp_mean)),
            ("temp_se_K", string(rep.temp_se)),
            ("temp_iat_g", string(rep.temp_iat)),
            ("n_samples", string(rep.n_samples)),
            ("potential_mean_kJ_mol", string(rep.potential_mean)),
            ("potential_per_molecule_kJ_mol", string(rep.potential_per_molecule)),
            ("nve_band_kJ_mol", string(rep.nve_band_kj)),
            ("nve_slope_kJ_mol_per_ps", string(rep.nve_slope)),
            ("nve_steps", string(rep.nve_steps)),
            ("diffusion_nm2_per_ps", string(rep.diffusion)),
            ("diffusion_se_nm2_per_ps", string(rep.diffusion_se)),
            ("msd_fit_from_ps", string(rep.msd_window[1])),
            ("msd_fit_to_ps", string(rep.msd_window[2])),
            ("rdf_frames", string(rep.rdf_frames)),
            ("rdf_first_peak_nm", string(rep.first_peak_nm)),
            ("rdf_first_min_nm", string(rep.first_min_nm)),
            ("dt_ps", string(dt)),
            ("gamma_per_ps", string(gamma)),
            ("seed", string(seed)),
            ("equil_tail_temp_mean_K", string(eq_tail_T)),
            ("equil_tail_temp_se_K", string(eq_tail_se)),
            ("equil_tail_iat_g", string(eq_tail_g)),
            ("target_temp_reached", string(temp_reached)),
            ("temp_gate_stage", "\"pre-production\""),
            ("temperature_trusted", string(temp_reached)),
            ("rdf_diffusion_trusted", "false"),
            ("equilibrium_claim", "false"),
            ("nve_dt_ps", string(nf(opts, "nve-dt-fs") / 1000.0)),
            ("chunk_ps", string(nf(opts, "chunk-ps"))),
            ("checkpoint_written", string(ckpt_path !== nothing)),
            ("restart_max_dq_nm", restart === nothing ? "null" : string(restart.dq)),
            ("restart_max_dp", restart === nothing ? "null" : string(restart.dp)),
            ("restart_ok", restart === nothing ? "null" : string(restart.ok)),
            ("com_stream", "\"mass-weighted molecular COM\""),
            ("rdf_rmax_requested_nm", string(rdf_rmax_requested)),
            ("rdf_rmax_valid_nm", string(rdf_rmax_valid)),
            ("rdf_tail_valid", "false"),
            ("ms_per_step", string(1000 * prod.wall_s / prod.steps)),
            ("steps_per_s", string(prod.steps / prod.wall_s)),
            ("nve_ladder", "[" * join(["{\"dt_fs\":$(r.dt_fs),\"steps\":$(r.steps),\"sim_ps\":$(r.sim_ps),\"band_kJ_mol\":$(r.band),\"slope_kJ_mol_per_ps\":$(r.slope),\"slope_se_kJ_mol_per_ps\":$(r.se)}" for r in ladder], ",") * "]"),
        ]
        println(io, "{")
        for (k, (name, val)) in enumerate(kv)
            println(io, "  \"", name, "\": ", val, k == length(kv) ? "" : ",")
        end
        println(io, "}")
    end

    println()
    WB.print_report(stdout, rep)
    println()
    @printf("artefacts in %s\n", outdir)
end

main()
