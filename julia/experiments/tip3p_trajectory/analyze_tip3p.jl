#!/usr/bin/env julia
#
# Independent re-analysis of a saved TIP3P trajectory. Reads ONLY the artefact
# files (traj.xyz + traj.xyz.jsonl, the report.json for the reference density)
# and recomputes:
#
#   * molecule integrity (every frame is 3N atoms, every O-H at r_OH)
#   * the O-O radial distribution function from the frame coordinates
#   * the molecular centre-of-mass mean squared displacement and the diffusion
#     coefficient, after unwrapping each molecule across the periodic boundary
#   * the density implied by the box and the molecule count
#
# It deliberately does not import `WaterBox`, so it is a genuine second opinion:
# if the in-run analysis and this disagree, one of them is wrong.
#
#   julia --project=. experiments/tip3p_trajectory/analyze_tip3p.jl <run-dir>
#
using Printf
using Statistics

const AVOGADRO = 6.02214076e23
const M_WATER = 18.0154        # g/mol
const R_OH = 0.09572           # nm

"Parse an XYZ frame file written by `Dynamics.FrameWriter`."
function read_xyz(path)
    frames = Vector{Matrix{Float64}}()
    labels = String[]
    open(path) do io
        while !eof(io)
            line = readline(io)
            isempty(strip(line)) && continue
            n = parse(Int, strip(line))
            readline(io)                      # step/time comment
            q = Matrix{Float64}(undef, 3, n)
            lab = Vector{String}(undef, n)
            for i in 1:n
                f = split(strip(readline(io)))
                length(f) >= 4 || error("malformed atom line $i in $path")
                lab[i] = f[1]
                q[:, i] .= parse.(Float64, f[2:4])
            end
            if isempty(labels)
                labels = lab
            elseif lab != labels
                error("atom labels/order change between frames in $path")
            end
            push!(frames, q)
        end
    end
    isempty(frames) && error("no frames in $path")
    return frames, labels
end

"Parse the JSONL sidecar: returns (steps, times, box vectors)."
function read_sidecar(path)
    steps = Int[]
    times = Float64[]
    boxes = Vector{NTuple{3,Float64}}()
    for line in eachline(path)
        isempty(strip(line)) && continue
        m = match(r"\"step\":(\d+).*\"time\":([-0-9.eE+]+).*\"box\":\[([-0-9.eE+]+),([-0-9.eE+]+),([-0-9.eE+]+)\]", line)
        m === nothing && continue
        push!(steps, parse(Int, m.captures[1]))
        push!(times, parse(Float64, m.captures[2]))
        push!(boxes, (parse(Float64, m.captures[3]), parse(Float64, m.captures[4]),
                      parse(Float64, m.captures[5])))
    end
    return steps, times, boxes
end

"Molecular COM positions from a frame, assuming atoms (3i-2, 3i-1, 3i)."
function com_positions(q, mO, mH)
    n = size(q, 2) ÷ 3
    com = Matrix{Float64}(undef, 3, n)
    mt = mO + 2mH
    for i in 1:n
        o = 3i - 2
        for d in 1:3
            com[d, i] = (mO * q[d, o] + mH * q[d, o + 1] + mH * q[d, o + 2]) / mt
        end
    end
    return com
end

"Unwrap a COM trajectory molecule-by-molecule under the orthorhombic minimum image."
function unwrap_coms(coms, L)
    out = [copy(c) for c in coms]
    for k in 2:length(out)
        prev = out[k - 1]
        cur = out[k]
        for i in axes(cur, 2), d in 1:3
            dx = cur[d, i] - prev[d, i]
            cur[d, i] = prev[d, i] + (dx - L[d] * round(dx / L[d]))
        end
    end
    return out
end

function rdf_oo(frames, L, rmax, nbins)
    dr = rmax / nbins
    counts = zeros(Int, nbins)
    n = size(frames[1], 2) ÷ 3
    for q in frames
        for b in 2:n, a in 1:(b - 1)
            oa, ob = 3a - 2, 3b - 2
            dx = q[1, oa] - q[1, ob]
            dy = q[2, oa] - q[2, ob]
            dz = q[3, oa] - q[3, ob]
            dx -= L[1] * round(dx / L[1])
            dy -= L[2] * round(dy / L[2])
            dz -= L[3] * round(dz / L[3])
            r2 = dx * dx + dy * dy + dz * dz
            r2 > rmax^2 && continue
            k = floor(Int, sqrt(r2) / dr) + 1
            k <= nbins && (counts[k] += 1)
        end
    end
    V = prod(L)
    npairs = 0.5 * n * (n - 1) * length(frames)
    g = zeros(Float64, nbins)
    r = zeros(Float64, nbins)
    for k in 1:nbins
        r[k] = (k - 0.5) * dr
        shell = (4 / 3) * π * ((k * dr)^3 - ((k - 1) * dr)^3) / V
        g[k] = counts[k] / (npairs * shell)
    end
    return r, g
end

"OLS slope over the second half of (t, y)."
function slope_second_half(t, y)
    lo = max(1, length(t) ÷ 2)
    tt = t[lo:end]; yy = y[lo:end]
    tb = mean(tt); yb = mean(yy)
    sxx = sum(abs2, tt .- tb)
    b = sum((tt .- tb) .* (yy .- yb)) / sxx
    resid = yy .- (yb .+ b .* (tt .- tb))
    se = sqrt(sum(abs2, resid) / max(length(tt) - 2, 1) / sxx)
    return b, se, (tt[1], tt[end])
end

function main()
    isempty(ARGS) && (println("usage: analyze_tip3p.jl <run-dir>"); exit(1))
    dir = ARGS[1]
    xyz = joinpath(dir, "traj.xyz")
    isfile(xyz) || (println("no traj.xyz in $dir"); exit(1))
    frames, labels = read_xyz(xyz)
    steps, times, boxes = read_sidecar(xyz * ".jsonl")
    @assert length(frames) == length(times) "frame/sidecar mismatch"
    @assert length(boxes) == length(frames) "frame/box-record mismatch"

    # ---- saved-frame structural assertions (cheap, and they matter) --------
    n_atoms = size(frames[1], 2)
    n_atoms % 3 == 0 || error("atom count $n_atoms is not 3 per rigid water")
    n = n_atoms ÷ 3
    for (k, q) in enumerate(frames)
        size(q, 2) == n_atoms ||
            error("frame $k has $(size(q, 2)) atoms; expected $n_atoms")
    end
    expected_labels = repeat(["O", "H", "H"], n)
    labels == expected_labels ||
        error("atom labels/order are not O,H,H per molecule (got $(labels[1:min(end, 6)])...)")
    L = boxes[1]
    for (k, b) in enumerate(boxes)
        maximum(abs.(collect(b) .- collect(L))) <= 1e-9 ||
            error("box changes at frame $k ($(b) vs $(L)); this analysis assumes NVT")
    end
    all(diff(times) .> 0) || error("frame times are not strictly increasing")
    V = prod(L)
    rho = n * M_WATER / (AVOGADRO * V * 1e-21)

    println("run directory : ", dir)
    @printf("frames        : %d, atoms/frame %d (= %d molecules; labels O,H,H verified every frame)\n",
            length(frames), n_atoms, n)
    @printf("box           : %.4f x %.4f x %.4f nm, V = %.4f nm^3 (constant, NVT)\n", L..., V)
    @printf("density       : %.5f g/cm^3 (from box and molecule count)\n", rho)
    @printf("time span     : %.3f .. %.3f ps (%.3f ps covered)\n",
            times[1], times[end], times[end] - times[1])
    if length(times) > 2
        dtf = diff(times)
        @printf("frame spacing : %.4f ps (min %.4f, max %.4f)\n",
                mean(dtf), minimum(dtf), maximum(dtf))
    end

    # ---- molecule integrity ------------------------------------------------
    worst_oh = 0.0
    for q in frames
        for i in 1:n
            o = 3i - 2
            worst_oh = max(worst_oh,
                           abs(sqrt(sum(abs2, q[:, o + 1] .- q[:, o])) - R_OH),
                           abs(sqrt(sum(abs2, q[:, o + 2] .- q[:, o])) - R_OH))
        end
    end
    @printf("O-H integrity : max |r_OH - %.5f| = %.3e nm (all frames)\n", R_OH, worst_oh)

    # ---- RDF ---------------------------------------------------------------
    # The orthorhombic minimum image and the spherical-shell normalisation are
    # only valid inside a sphere of radius min(L)/2. Beyond that a bin is
    # double-counted and its shell volume is outside the cell, so the cutoff is
    # clamped; the tail is reported as not valid rather than silently plotted.
    rmax_requested = 1.0
    rmax_valid = min(rmax_requested, 0.5 * minimum(L))
    if rmax_valid < rmax_requested
        @printf("RDF cutoff    : requested %.3f nm clamped to %.3f nm (= min(L)/2); g(r) beyond that is not valid\n",
                rmax_requested, rmax_valid)
    end
    r, g = rdf_oo(frames, L, rmax_valid, 200)
    ipk = argmax(g[10:end]) + 9
    @printf("O-O RDF       : first peak g = %.3f at r = %.3f nm (valid range 0-%.3f nm)\n",
            g[ipk], r[ipk], rmax_valid)
    open(joinpath(dir, "rdf_oo_recheck.csv"), "w") do io
        println(io, "r_nm,g_oo")
        for k in eachindex(r)
            @printf(io, "%.6f,%.10f\n", r[k], g[k])
        end
    end

    # ---- MSD and diffusion -------------------------------------------------
    # Preferred source: the exact unwrapped COM stream written by the run. A
    # wrapped XYZ cannot be unwrapped reliably at coarse frame spacing (one
    # picosecond of water diffusion already moves a molecule further than
    # L/2), so that path is only a fallback and is flagged.
    comcsv = joinpath(dir, "com_unwrapped.csv")
    if isfile(comcsv)
        times_c = Float64[]
        coms_c = Matrix{Float64}[]
        tmap = Dict{Float64,Int}()
        for line in eachline(comcsv)
            startswith(line, "time_ps") && continue
            isempty(strip(line)) && continue
            f = split(line, ",")
            tp = parse(Float64, f[1])
            idx = get(tmap, tp, 0)
            if idx == 0
                push!(times_c, tp)
                push!(coms_c, zeros(3, n))
                idx = length(coms_c)
                tmap[tp] = idx
            end
            mi = parse(Int, f[2])
            coms_c[idx][1, mi] = parse(Float64, f[3])
            coms_c[idx][2, mi] = parse(Float64, f[4])
            coms_c[idx][3, mi] = parse(Float64, f[5])
        end
        perm = sortperm(times_c)
        coms = coms_c[perm]
        t = times_c[perm]
        println("COM source    : com_unwrapped.csv (exact, never wrapped)")
    else
        t = times .- times[1]
        coms = [com_positions(q, 15.9994, 1.008) for q in frames]
        coms = unwrap_coms(coms, L)
        println("COM source    : wrapped XYZ, minimum-image unwrapping (valid only")
        println("                when one frame moves a molecule less than L/2)")
    end
    t = t .- t[1]
    ref = coms[1]
    msd = [mean((c[1, i] - ref[1, i])^2 + (c[2, i] - ref[2, i])^2 +
                (c[3, i] - ref[3, i])^2 for i in 1:n) for c in coms]
    b, bse, win = slope_second_half(t, msd)
    @printf("diffusion     : D = %.4f +/- %.4f nm^2/ns (MSD fit %.2f-%.2f ps, %d frames)\n",
            b / 6 * 1000, bse / 6 * 1000, win[1], win[2], length(t))
    open(joinpath(dir, "msd_recheck.csv"), "w") do io
        println(io, "time_ps,msd_nm2")
        for k in eachindex(t)
            @printf(io, "%.6f,%.10f\n", t[k], msd[k])
        end
    end

    println()
    println("compare with report.txt/report.json in the same directory; the two were")
    println("produced by different code paths (in-run accumulators vs this re-read).")
end

main()
