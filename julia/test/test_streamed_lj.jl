# Tests for the standalone streamed Lennard-Jones backend.
#
# Run directly:
#
#     julia --project=julia julia/test/test_streamed_lj.jl
#
# The suite intentionally uses only `@assert` (like the other FoundationCore
# suites) so it can also be included from another runner without depending on a
# test framework. It is included in run_all.jl under its own isolated module.
#
# Coverage: independent O(N^2) parity for :lfs and :shift (with and without
# exclusions/scaled 1-4 pairs), the existing factor protocol as a second
# reference, finite-difference forces, finite-difference parameter gradients,
# cutoff C1 continuity, wrapping/cell-edge handling, periodic-grid degeneracies,
# workspace reuse, and validation that never mutates forces on failure.
# Threaded parity/allocation checks run in child Julia processes at multiple counts.

using FoundationCore
using FoundationCore.Types
using FoundationCore.Factors
using Random
using LinearAlgebra: norm

# Include the standalone module under a host that supplies the already-loaded
# FoundationCore.Types binding. `module StreamedLJ` uses `..Types`, so this is
# exactly the include pattern a parent module (FoundationCore) would use.
module StreamedLJHost
    using FoundationCore: Types
    include(joinpath(@__DIR__, "..", "src", "StreamedLJ.jl"))
end

const SL = StreamedLJHost.StreamedLJ
const T = FoundationCore.Types
const FA = FoundationCore.Factors

# ---------------------------------------------------------------------------
# independent references
# ---------------------------------------------------------------------------

# Weight lookup in the *test* reference deliberately does not use StreamedLJ's
# PairTopology: it is a plain dictionary on canonical (i, j) tuples, so the
# compact topology encoding is checked against an independent representation.
function test_weight(weights, i::Int, j::Int)
    weights === nothing && return 1.0
    return get(weights, (min(i, j), max(i, j)), 1.0)
end

"""
Explicit O(N^2) LJ pair sum. `mode` is `:lfs`, `:shift`, or `:trunc`; `weights` maps
canonical `(i, j)` to a pair weight (missing entries mean weight 1).
"""
function brute_force_lj(q::AbstractMatrix{Float64}, epsilon::Vector{Float64},
                        sigma::Vector{Float64}, box::T.AbstractBox,
                        rc::Float64; mode::Symbol = :lfs, weights = nothing)
    N = size(q, 2)
    F = zeros(Float64, 3, N)
    U = 0.0
    npairs = 0
    rc2 = rc * rc
    @inbounds for i in 1:N-1
        for j in i+1:N
            w = test_weight(weights, i, j)
            w == 0.0 && continue
            d = T.unwrap_displacement(box, q, i, j)
            r2 = T.norm2(d)
            r2 > rc2 && continue
            r = sqrt(r2)
            eij = sqrt(epsilon[i] * epsilon[j])
            sij = 0.5 * (sigma[i] + sigma[j])
            s6 = (sij / r)^6
            s12 = s6 * s6
            s6c = (sij / rc)^6
            s12c = s6c * s6c
            u = 4.0 * eij * (s12 - s6)
            dudr = 24.0 * eij * (s6 - 2.0 * s12) / r
            if mode === :lfs
                uc = 4.0 * eij * (s12c - s6c)
                duc = 24.0 * eij * (s6c - 2.0 * s12c) / rc
                u = u - uc - duc * (r - rc)
                dudr = dudr - duc
            elseif mode === :shift
                u = u - 4.0 * eij * (s12c - s6c)
            else
                @assert mode === :trunc "reference supports :lfs, :shift and :trunc"
            end
            u *= w
            c = -w * dudr / r
            F[1, i] += c * d.x
            F[2, i] += c * d.y
            F[3, i] += c * d.z
            F[1, j] -= c * d.x
            F[2, j] -= c * d.y
            F[3, j] -= c * d.z
            U += u
            npairs += 1
        end
    end
    return U, F, npairs
end

"""
Explicit O(N^2) reference box derivative ∂U/∂h|_s at fixed fractional coordinates.
"""
function brute_force_lj_box_derivative(q::AbstractMatrix{Float64}, epsilon::Vector{Float64},
                                       sigma::Vector{Float64}, box::T.AbstractBox,
                                       rc::Float64; mode::Symbol = :lfs, weights = nothing)
    N = size(q, 2)
    G = zeros(Float64, 3, 3)
    rc2 = rc * rc
    for i in 1:N-1
        for j in i+1:N
            w = test_weight(weights, i, j)
            w == 0.0 && continue
            d = T.unwrap_displacement(box, q, i, j)
            r2 = T.norm2(d)
            r2 > rc2 && continue
            r = sqrt(r2)
            eij = sqrt(epsilon[i] * epsilon[j])
            sij = 0.5 * (sigma[i] + sigma[j])
            s6 = (sij / r)^6
            s12 = s6 * s6
            s6c = (sij / rc)^6
            s12c = s6c * s6c
            dudr = 24.0 * eij * (s6 - 2.0 * s12) / r
            if mode === :lfs
                duc = 24.0 * eij * (s6c - 2.0 * s12c) / rc
                dudr -= duc
            end
            c_box = (w * dudr) / r
            if box isa T.OrthorhombicBox
                wx = d.x / box.L[1]; wy = d.y / box.L[2]; wz = d.z / box.L[3]
            else
                M = box.Hinv
                wx = M[1, 1] * d.x + M[1, 2] * d.y + M[1, 3] * d.z
                wy = M[2, 1] * d.x + M[2, 2] * d.y + M[2, 3] * d.z
                wz = M[3, 1] * d.x + M[3, 2] * d.y + M[3, 3] * d.z
            end
            gx = c_box * d.x; gy = c_box * d.y; gz = c_box * d.z
            G[1, 1] += gx * wx; G[1, 2] += gx * wy; G[1, 3] += gx * wz
            G[2, 1] += gy * wx; G[2, 2] += gy * wy; G[2, 3] += gy * wz
            G[3, 1] += gz * wx; G[3, 2] += gz * wy; G[3, 3] += gz * wz
        end
    end
    return G
end

"Reference built from the existing factor protocol (one LennardJonesPair per pair)."
function factors_lj(q::AbstractMatrix{Float64}, epsilon::Vector{Float64},
                    sigma::Vector{Float64}, box::T.OrthorhombicBox, rc::Float64;
                    mode::Symbol = :lfs, weights = nothing)
    N = size(q, 2)
    factors = FA.LennardJonesPair[]
    for i in 1:N-1, j in i+1:N
        w = test_weight(weights, i, j)
        w == 0.0 && continue
        eij = sqrt(epsilon[i] * epsilon[j])
        sij = 0.5 * (sigma[i] + sigma[j])
        push!(factors, FA.LennardJonesPair(i, j, eij, sij;
                                           mode = mode, rc = rc, lj_scale = w))
    end
    ctx = FA.Context(box, T.REDUCED, N)
    return FA.total_energy_forces(factors, q, ctx)
end

function relative_scale(v::AbstractArray)
    return max(maximum(abs, v), 1.0)
end

function check_array(a::AbstractArray, b::AbstractArray, tol::Float64, label::String)
    err = maximum(abs, a .- b)
    scale = relative_scale(b)
    @assert err <= tol * scale "$label: max|Δ|=$err exceeds $(tol)×scale=$scale"
    return err
end

function expect_argument_error(f::Function, label::String)
    ok = false
    try
        f()
    catch e
        if e isa ArgumentError
            ok = true
        else
            println("  wrong error type for $label: ", typeof(e), " — ", sprint(showerror, e))
        end
    end
    @assert ok "$label should throw ArgumentError"
    return nothing
end

function random_q(rng, N::Int, L)
    q = rand(rng, 3, N)
    q[1, :] .*= L[1]
    q[2, :] .*= L[2]
    q[3, :] .*= L[3]
    return q
end

# 1-2/1-3 exclusion list plus 1-4 scaled list, and the independent
# canonical-key dictionary used by the brute-force reference. When `q`, `box`
# and `rc` are supplied, the *closest* pairs are selected so the topology is
# exercised by pairs that are actually inside the cutoff.
function topology_fixture(rng, N::Int; nex::Int = 6, nsc::Int = 5,
                          q = nothing, box = nothing, rc = nothing)
    pairs = Tuple{Int,Int}[]
    for i in 1:N-1, j in i+1:N
        push!(pairs, (i, j))
    end
    nex + nsc <= length(pairs) || error("fixture too small")
    if q !== nothing && box !== nothing && rc !== nothing
        sort!(pairs; by = p -> T.norm2(T.unwrap_displacement(box, q, p[1], p[2])))
    else
        shuffle!(rng, pairs)
    end
    excluded = [pairs[k] for k in 1:nex]
    scaled = [(pairs[nex + k][1], pairs[nex + k][2], 0.5) for k in 1:nsc]

    weights = Dict{Tuple{Int,Int},Float64}()
    for (i, j) in excluded
        weights[(i, j)] = 0.0
    end
    for (i, j, w) in scaled
        weights[(i, j)] = w
    end
    topo = SL.PairTopology(excluded, scaled)

    if q !== nothing && box !== nothing && rc !== nothing
        rc2 = rc * rc
        inside = 0
        for (i, j) in keys(weights)
            T.norm2(T.unwrap_displacement(box, q, i, j)) <= rc2 && (inside += 1)
        end
        inside > 0 || error("topology fixture has no pair inside the cutoff")
    end
    return topo, weights, excluded, scaled
end

# ---------------------------------------------------------------------------
# parity
# ---------------------------------------------------------------------------

function test_parity_lfs_shift_and_topology()
    rng = MersenneTwister(4)
    N = 30
    box = T.OrthorhombicBox((10.0, 11.0, 12.0))
    rc = 2.4
    q = random_q(rng, N, box.L)
    epsilon = 0.5 .+ rand(rng, N)
    sigma = 0.8 .+ 0.4 .* rand(rng, N)
    topo, weights, _, _ = topology_fixture(rng, N; q = q, box = box, rc = rc)

    ws = SL.LJWorkspace(q, box, rc)
    for mode in (:lfs, :shift, :trunc)
        U, F, np = SL.evaluate!(ws, q, epsilon, sigma, box; rcut = rc, mode = mode)
        Ub, Fb, npb = brute_force_lj(q, epsilon, sigma, box, rc; mode = mode)
        Uf, Ff = factors_lj(q, epsilon, sigma, box, rc; mode = mode)
        @assert F === ws.forces
        @assert np == npb "mode=$mode pair count $(np) != brute-force $(npb)"
        e1 = check_array([U], [Ub], 1e-11, "$mode energy vs brute force")
        e2 = check_array(F, Fb, 1e-11, "$mode forces vs brute force")
        e3 = check_array([U], [Uf], 1e-11, "$mode energy vs factor protocol")
        e4 = check_array(F, Ff, 1e-11, "$mode forces vs factor protocol")
        println("  $mode no topology: N=$N pairs=$np ",
                "energy_err=$e1 force_err=$e2 factor_err=$((e3, e4))")

        Ut, Ft, npt = SL.evaluate!(ws, q, epsilon, sigma, box;
                                   rcut = rc, mode = mode, topology = topo)
        Ubt, Fbt, npbt = brute_force_lj(q, epsilon, sigma, box, rc;
                                        mode = mode, weights = weights)
        Uft, Fft = factors_lj(q, epsilon, sigma, box, rc;
                              mode = mode, weights = weights)
        @assert npt == npbt "mode=$mode topology pair count $(npt) != $(npbt)"
        @assert npt <= np
        @assert Ut != U || npt != np "topology fixture had no measurable effect"
        et1 = check_array([Ut], [Ubt], 1e-11, "$mode topology energy vs brute force")
        et2 = check_array(Ft, Fbt, 1e-11, "$mode topology forces vs brute force")
        et3 = check_array([Ut], [Uft], 1e-11, "$mode topology energy vs factors")
        et4 = check_array(Ft, Fft, 1e-11, "$mode topology forces vs factors")
        println("  $mode with topology: pairs=$npt ",
                "energy_err=$et1 force_err=$et2 factor_err=$((et3, et4))")
    end

    # The production defaults are force-consistent LFS; :shift remains opt-in.
    @assert SL.CUTOFF_MODE === :lfs
    @assert SL.DEFAULT_CUTOFF_MODE === :lfs
    @assert SL.FORCE_CONSISTENT_AT_CUTOFF === true
    @assert Set(SL.SUPPORTED_CUTOFF_MODES) == Set((:lfs, :shift, :trunc))

    # Zero-size/zero-energy atoms (AMBER/TIP3P hydrogens) are physical: the
    # pair energy is exactly zero, so no fake positive sigma is required.
    qz = [0.0 0.9; 0.0 0.0; 0.0 0.0]
    Uz, Fz, npz = SL.evaluate!(SL.LJWorkspace(qz, box, rc), qz, [0.0, 1.0],
                               [0.0, 0.31], box; rcut = rc, mode = :lfs)
    @assert Uz == 0.0 && npz == 1
    @assert all(Fz .== 0.0)
    return true
end

function test_threaded_force_path()
    rng = MersenneTwister(892)
    N = 64
    box = T.OrthorhombicBox((9.0, 10.0, 11.0))
    rc = 2.3
    # A lightly jittered lattice gives many cutoff pairs but avoids extremely
    # close encounters, so the absolute force comparison is meaningful.
    q = zeros(3, N)
    for k in 0:N-1
        q[:, k+1] .= ((k % 4 + 0.5) * box.L[1] / 4,
                       ((k ÷ 4) % 4 + 0.5) * box.L[2] / 4,
                       (k ÷ 16 + 0.5) * box.L[3] / 4)
    end
    q .+= 0.15 .* (rand(rng, 3, N) .- 0.5)
    epsilon = 0.6 .+ 0.3 .* rand(rng, N)
    sigma = 0.8 .+ 0.2 .* rand(rng, N)
    topo, weights, _, _ = topology_fixture(rng, N; q = q, box = box, rc = rc)
    ws = SL.LJWorkspace(q, box, rc)
    serial_result = SL.evaluate!(ws, q, epsilon, sigma, box; rcut = rc,
                                 skin = 0.0, mode = :lfs)
    signatures = String[]

    for skin in (0.0, 2.4, 8.0), mode in (:lfs, :shift), use_topo in (false, true)
        tp = use_topo ? topo : nothing
        wt = use_topo ? weights : nothing
        Ub, Fb, npb = brute_force_lj(q, epsilon, sigma, box, rc;
                                     mode = mode, weights = wt)
        U, F, np = SL.evaluate!(ws, q, epsilon, sigma, box;
                               rcut = rc, skin = skin, mode = mode,
                               topology = tp, threaded = true)
        @assert F === ws.forces
        @assert np == npb "threaded count differs for skin=$skin mode=$mode"
        check_array([U], [Ub], 1e-11, "threaded energy skin=$skin mode=$mode")
        check_array(F, Fb, 1e-11, "threaded force skin=$skin mode=$mode")
        if skin == 0.0 && mode === :lfs && !use_topo
            @assert np == serial_result[3]
            check_array([U], [serial_result[1]], 1e-11, "serial/threaded energy")
            serial_thread_error = check_array(F, serial_result[2], 1e-11,
                                              "serial/threaded forces")
            println("  serial/threaded max force error: $serial_thread_error")
        end
        # Reduction order is fixed for a given thread count; cross-count force
        # comparison is numeric because per-thread grouping changes rounding.
        result = (U, np, copy(F))
        U2, F2, np2 = SL.evaluate!(ws, q, epsilon, sigma, box;
                                  rcut = rc, skin = skin, mode = mode,
                                  topology = tp, threaded = true)
        @assert isequal((U2, np2, F2), result)
        push!(signatures, string(bitstring(U), ':', np, ':',
                                 join(bitstring.(vec(result[3])), ',')))
    end

    # Multiple atoms per cell, mixed dimensions of size 2/3, and positions
    # outside the primary box exercise same-cell and periodic cross-cell walks.
    dense_box = T.OrthorhombicBox((6.0, 7.0, 8.0))
    dense_rc = 2.6
    dense_q = random_q(rng, 32, dense_box.L)
    dense_q[:, 1:3] .+= 6.0
    dense_eps = 0.6 .+ 0.3 .* rand(rng, 32)
    dense_sig = 0.8 .+ 0.2 .* rand(rng, 32)
    dense_ws = SL.LJWorkspace(dense_q, dense_box, dense_rc)
    for mode in (:lfs, :shift)
        Ub, Fb, npb = brute_force_lj(dense_q, dense_eps, dense_sig,
                                     dense_box, dense_rc; mode = mode)
        U, F, np = SL.evaluate!(dense_ws, dense_q, dense_eps, dense_sig,
                                dense_box; rcut = dense_rc, mode = mode,
                                threaded = true)
        @assert np == npb
        check_array([U], [Ub], 1e-11, "threaded dense $mode energy")
        check_array(F, Fb, 1e-11, "threaded dense $mode force")
        push!(signatures, string(bitstring(U), ':', np, ':',
                                 join(bitstring.(vec(F)), ',')))
    end

    # Warm the topology-enabled path before reporting scheduler/force-buffer
    # allocation. Allocation is diagnostic only, not a physics correctness gate.
    _, _, measured_pairs = SL.evaluate!(ws, q, epsilon, sigma, box;
                                        rcut=rc, topology=topo, threaded=true)
    allocated = @allocated SL.evaluate!(ws, q, epsilon, sigma, box;
                                         rcut=rc, topology=topo, threaded=true)
    println("  warmed threaded allocations (diagnostic): $allocated bytes; " *
            "N=$N pairs=$measured_pairs; serial/threaded numeric parity checked")

    # Gradients deliberately use the original serial kernel, even when the
    # force-only threaded path is enabled by the caller.
    U0, F0, np0, ge0, gs0 = SL.evaluate_with_gradients!(
        ws, q, epsilon, sigma, box; rcut = rc, topology = topo)
    result = (U0, np0, copy(F0), copy(ge0), copy(gs0))
    U1, F1, np1, ge1, gs1 = SL.evaluate_with_gradients!(
        ws, q, epsilon, sigma, box; rcut = rc, topology = topo, threaded = true)
    @assert isequal((U1, np1, F1, ge1, gs1), result)
    println("  threaded reference parity at $(Threads.nthreads()) Julia threads")
    return "threads=$(Threads.nthreads()), repeated=$(length(signatures))"
end

function test_thread_counts()
    # Fork the focused fixture at 1, 2, 4, and 8 threads. Each process checks
    # fixed-configuration repeatability; each threaded result is independently
    # compared with the serial reference using bounded numerical error.
    signatures = String[]
    for nt in (1, 2, 4, 8)
        cmd = `$(Base.julia_cmd()) --startup-file=no --threads=$nt --project=$(joinpath(@__DIR__, "..")) $(@__FILE__)`
        output = read(addenv(cmd, "STREAMED_LJ_THREADS_CHILD" => "1"), String)
        marker = "THREAD_SIGNATURE "
        pos = findfirst(marker, output)
        @assert pos !== nothing "threaded Julia child $nt did not produce signature: $output"
        push!(signatures, strip(output[last(pos)+1:end]))
    end
    if Threads.nthreads() in (1, 2, 4, 8)
        child_index = findfirst(==(Threads.nthreads()), (1, 2, 4, 8))
        @assert test_threaded_force_path() == signatures[child_index]
    end
    return true
end

function test_serial_allocation_parity()
    rng = MersenneTwister(122)
    box = T.OrthorhombicBox((9.0, 10.0, 11.0))
    q = random_q(rng, 40, box.L)
    epsilon = fill(0.8, 40)
    sigma = fill(0.9, 40)
    ws = SL.LJWorkspace(q, box, 2.3)
    U0, F0, np0 = SL.evaluate!(ws, q, epsilon, sigma, box; rcut = 2.3)
    serial = (U0, copy(F0), np0)
    U1, F1, np1 = SL.evaluate!(ws, q, epsilon, sigma, box;
                              rcut = 2.3, threaded = false)
    @assert isequal(serial, (U1, F1, np1))
    # Both calls compile to the same serial path; opt-in threading must not
    # impose extra per-evaluation storage on the default call.
    original = @allocated SL.evaluate!(ws, q, epsilon, sigma, box; rcut = 2.3)
    explicit = @allocated SL.evaluate!(ws, q, epsilon, sigma, box;
                                       rcut = 2.3, threaded = false)
    @assert original <= explicit + 64 "default serial path allocated more: $original vs $explicit"
    println("  serial hot-path allocations: default=$original explicit_false=$explicit")
    return true
end

function test_explicit_1_2_1_3_exclusion_and_1_4_scaling()
    # Distances: 1-2 = 2.0, 1-3 = 2.0, 1-4 = 2.5, 2-3 = 2.828, 2-4/3-4 > rc.
    L = 20.0
    rc = 3.0
    box = T.OrthorhombicBox((L, L, L))
    q = zeros(Float64, 3, 4)
    q[1, 2] = 2.0
    q[2, 3] = 2.0
    q[3, 4] = 2.5
    epsilon = [1.0, 0.8, 1.2, 0.9]
    sigma = [1.0, 1.1, 0.95, 1.05]

    excluded = [(1, 2), (1, 3)]   # 1-2 and 1-3 exclusions
    scaled = [(1, 4, 0.5)]        # 1-4 LJ scale
    topo = SL.PairTopology(excluded = excluded, scaled = scaled)
    weights = Dict{Tuple{Int,Int},Float64}((1, 2) => 0.0, (1, 3) => 0.0,
                                           (1, 4) => 0.5)

    @assert SL.pair_weight(topo, 2, 1) == 0.0
    @assert SL.pair_weight(topo, 3, 1) == 0.0
    @assert SL.pair_weight(topo, 4, 1) == 0.5
    @assert SL.pair_weight(topo, 2, 3) == 1.0
    @assert SL.pair_weight(nothing, 2, 1) == 1.0

    ws = SL.LJWorkspace(q, box, rc)
    for mode in (:lfs, :shift, :trunc)
        U, F, np = SL.evaluate!(ws, q, epsilon, sigma, box;
                                rcut = rc, mode = mode, topology = topo)
        Ub, Fb, npb = brute_force_lj(q, epsilon, sigma, box, rc;
                                     mode = mode, weights = weights)
        @assert np == npb == 2 "1-2/1-3 exclusion pair count: $(np) / $(npb)"
        check_array([U], [Ub], 1e-12, "$mode 1-4 scaled energy")
        check_array(F, Fb, 1e-12, "$mode 1-4 scaled forces")
        println("  $mode 1-2/1-3 excluded + 1-4 scaled: pairs=$np energy=$U")
    end

    # A wholly excluded two-atom system contributes nothing at all.
    q2 = zeros(Float64, 3, 2)
    q2[1, 2] = 2.0
    topo2 = SL.PairTopology(excluded = [(2, 1)])
    @assert SL.pair_weight(topo2, 1, 2) == 0.0
    for mode in (:lfs, :shift)
        U2, F2, np2 = SL.evaluate!(ws, q2, epsilon[1:2], sigma[1:2], box;
                                   rcut = rc, mode = mode, topology = topo2)
        @assert U2 == 0.0
        @assert all(==(0.0), F2)
        @assert np2 == 0
    end

    # A zero scale is normalised to an exclusion, and Dict input is accepted.
    topo3 = SL.PairTopology(scaled = Dict((2, 3) => 0.0))
    @assert SL.pair_weight(topo3, 3, 2) == 0.0
    topo4 = SL.PairTopology(scaled = Dict((2, 3) => 0.25))
    @assert SL.pair_weight(topo4, 3, 2) == 0.25
    println("  1-2/1-3 exclusion and 1-4 scaling semantics verified")
    return true
end

# ---------------------------------------------------------------------------
# derivatives
# ---------------------------------------------------------------------------

function test_finite_difference_forces()
    rng = MersenneTwister(6)
    N = 12
    box = T.OrthorhombicBox((8.0, 9.0, 10.0))
    rc = 3.5
    q = random_q(rng, N, box.L)
    epsilon = 0.6 .+ rand(rng, N)
    sigma = 0.85 .+ 0.3 .* rand(rng, N)
    topo, weights, _, _ = topology_fixture(rng, N; nex = 3, nsc = 2,
                                          q = q, box = box, rc = rc)
    ws = SL.LJWorkspace(q, box, rc)

    for mode in (:lfs, :shift)
        U0, F, np = SL.evaluate!(ws, q, epsilon, sigma, box;
                                 rcut = rc, mode = mode, topology = topo)
        @assert np > 0 "finite-difference fixture must have interacting pairs"
        h = 1e-6
        G = zeros(Float64, 3, N)
        qp = copy(q)
        qm = copy(q)
        for j in 1:N, d in 1:3
            qp[d, j] += h
            qm[d, j] -= h
            Up = brute_force_lj(qp, epsilon, sigma, box, rc;
                                mode = mode, weights = weights)[1]
            Um = brute_force_lj(qm, epsilon, sigma, box, rc;
                                mode = mode, weights = weights)[1]
            G[d, j] = (Up - Um) / (2h)
            qp[d, j] -= h
            qm[d, j] += h
        end
        err = maximum(abs, F .+ G) / max(maximum(abs, F), 1.0)
        @assert err < 1e-5 "$mode finite-difference force error $err"
        @assert isfinite(U0)
        println("  $mode finite-difference forces: pairs=$np normalized_err=$err")
    end
    return true
end

function test_parameter_gradients()
    rng = MersenneTwister(17)
    N = 12
    box = T.OrthorhombicBox((8.0, 9.0, 10.0))
    rc = 3.5
    q = random_q(rng, N, box.L)
    epsilon = 0.7 .+ rand(rng, N)
    sigma = 0.85 .+ 0.3 .* rand(rng, N)
    topo, weights, _, _ = topology_fixture(rng, N; nex = 3, nsc = 2,
                                          q = q, box = box, rc = rc)
    ws = SL.LJWorkspace(q, box, rc)

    for mode in (:lfs, :shift)
        U, F, np, ge, gs = SL.evaluate_with_gradients!(
            ws, q, epsilon, sigma, box; rcut = rc, mode = mode, topology = topo)
        U2, F2, np2 = SL.evaluate!(ws, q, epsilon, sigma, box;
                                   rcut = rc, mode = mode, topology = topo)
        @assert (U, np) == (U2, np2)
        @assert F == F2
        @assert length(ge) == N && length(gs) == N
        @assert ge === ws.grad_epsilon && gs === ws.grad_sigma

        h = 1e-6
        eps_fd = zeros(Float64, N)
        sig_fd = zeros(Float64, N)
        for k in 1:N
            ep = copy(epsilon); ep[k] += h
            em = copy(epsilon); em[k] -= h
            eps_fd[k] = (brute_force_lj(q, ep, sigma, box, rc;
                                        mode = mode, weights = weights)[1] -
                         brute_force_lj(q, em, sigma, box, rc;
                                        mode = mode, weights = weights)[1]) / (2h)

            sp = copy(sigma); sp[k] += h
            sm = copy(sigma); sm[k] -= h
            sig_fd[k] = (brute_force_lj(q, epsilon, sp, box, rc;
                                        mode = mode, weights = weights)[1] -
                         brute_force_lj(q, epsilon, sm, box, rc;
                                        mode = mode, weights = weights)[1]) / (2h)
        end
        ee = check_array(ge, eps_fd, 1e-5, "$mode dU/depsilon")
        es = check_array(gs, sig_fd, 1e-5, "$mode dU/dsigma")
        println("  $mode parameter gradients: pairs=$np ",
                "eps_err=$ee sigma_err=$es")
    end
    return true
end

# ---------------------------------------------------------------------------
# cutoff behaviour
# ---------------------------------------------------------------------------

function test_cutoff_continuity_and_wrapping()
    L = 20.0
    rc = 2.5
    box = T.OrthorhombicBox((L, L, L))
    epsilon = [1.2, 0.8]
    sigma = [1.0, 1.1]

    function two_atom_energy_force(d::Float64, mode::Symbol)
        q = zeros(Float64, 3, 2)
        q[1, 2] = d
        ws = SL.LJWorkspace(q, box, rc)
        return SL.evaluate!(ws, q, epsilon, sigma, box; rcut = rc, mode = mode)
    end

    delta = 1e-6
    for mode in (:lfs, :shift)
        Uin, Fin, npin = two_atom_energy_force(rc - delta, mode)
        Uat, Fat, npat = two_atom_energy_force(rc, mode)
        Uout, Fout, npout = two_atom_energy_force(rc + delta, mode)
        @assert npin == 1 && npat == 1 && npout == 0
        @assert Uat == 0.0
        mode === :lfs && @assert all(==(0.0), Fat)
        @assert Uout == 0.0 && all(==(0.0), Fout)
        @assert abs(Uin) < (mode === :lfs ? 1e-9 : 1e-7) "$mode energy is not continuous at rcut: $Uin"
        if mode === :lfs
            # C1: the force also vanishes linearly at the cutoff.
            @assert maximum(abs, Fin) < 1e-3 "LFS force is not continuous at rcut"
            Ush, Fsh, _ = two_atom_energy_force(rc - delta, :shift)
            @assert maximum(abs, Fsh) > 10 * maximum(abs, Fin) "LFS force should be far smaller than the discontinuous :shift force"
            println("  LFS cutoff continuity: U(rc-δ)=$(Uin) |F|=$(maximum(abs, Fin)); ",
                    ":shift |F|=$(maximum(abs, Fsh))")
        else
            # C0 only: the energy is continuous but the force is not.
            @assert maximum(abs, Fin) > 1e-3 "the fixture should expose the :shift force jump"
            println("  :shift cutoff continuity: U(rc-δ)=$(Uin) |F|=$(maximum(abs, Fin))")
        end
        # Independent reference agrees right up to the cutoff.
        q = zeros(Float64, 3, 2); q[1, 2] = rc - delta
        Ub, Fb, npb = brute_force_lj(q, epsilon, sigma, box, rc; mode = mode)
        @assert npb == 1
        check_array([Uin], [Ub], 1e-12, "$mode near-cutoff energy")
        check_array(Fin, Fb, 1e-12, "$mode near-cutoff forces")
    end

    # Wrapping: translating atoms by exact box vectors must not change anything.
    rng = MersenneTwister(23)
    N = 14
    Lw = 8.0
    boxw = T.OrthorhombicBox((Lw, Lw, Lw))
    q = random_q(rng, N, boxw.L)
    epsilon = 0.9 .+ 0.2 .* rand(rng, N)
    sigma = 0.9 .+ 0.2 .* rand(rng, N)
    topo, weights, _, _ = topology_fixture(rng, N; nex = 2, nsc = 2,
                                          q = q, box = boxw, rc = rc)
    ws = SL.LJWorkspace(q, boxw, rc)
    for mode in (:lfs, :shift, :trunc)
        U, F, np = SL.evaluate!(ws, q, epsilon, sigma, boxw;
                                rcut = rc, mode = mode, topology = topo)
        shifts = rand(rng, -2:2, 3, N)
        qs = copy(q)
        for j in 1:N, d in 1:3
            qs[d, j] += shifts[d, j] * boxw.L[d]
        end
        Us, Fs, nps = SL.evaluate!(ws, qs, epsilon, sigma, boxw;
                                   rcut = rc, mode = mode, topology = topo)
        @assert np == nps
        @assert np > 0 "wrapping fixture must have interacting pairs"
        check_array([U], [Us], 1e-12, "$mode wrapped energy")
        check_array(F, Fs, 1e-12, "$mode wrapped forces")
        println("  $mode wrapping: pairs=$np energy_delta=$(abs(U - Us))")
    end

    # Atoms exactly on cell boundaries (including x == L) must bucket correctly.
    Lb = 8.0
    rcb = 2.0                     # nc = 4 per axis, cell width 2.0
    boxb = T.OrthorhombicBox((Lb, Lb, Lb))
    qb = zeros(Float64, 3, 8)
    for k in 0:7
        qb[1, k + 1] = 2.0 * (k % 4)      # 0, 2, 4, 6, 0, ...
        qb[2, k + 1] = (k >= 4 ? Lb : 0.0) # exactly L on the top boundary
        qb[3, k + 1] = 0.5 * k
    end
    epsb = [0.8 + 0.1 * k for k in 1:8]
    sigb = [0.95 + 0.02 * k for k in 1:8]
    wsb = SL.LJWorkspace(qb, boxb, rcb)
    for mode in (:lfs, :shift)
        Ub, Fb, npb = SL.evaluate!(wsb, qb, epsb, sigb, boxb;
                                   rcut = rcb, mode = mode)
        Ur, Fr, npr = brute_force_lj(qb, epsb, sigb, boxb, rcb; mode = mode)
        @assert npb == npr
        check_array([Ub], [Ur], 1e-11, "$mode boundary-cell energy")
        check_array(Fb, Fr, 1e-11, "$mode boundary-cell forces")
    end
    println("  cell-boundary and wrapping fixtures verified")
    return true
end

function test_orthorhombic_cutoff_safety()
    box = T.OrthorhombicBox((8.0, 10.0, 12.0))
    rng = MersenneTwister(29)
    N = 10
    q = random_q(rng, N, box.L)
    epsilon = fill(1.0, N)
    sigma = fill(1.0, N)
    ws = SL.LJWorkspace(q, box, 3.9)   # min(L)/2 = 4.0, so 3.9 is valid
    for mode in (:lfs, :shift, :trunc)
        U, F, np = SL.evaluate!(ws, q, epsilon, sigma, box; rcut = 3.9, mode = mode)
        Ub, Fb, npb = brute_force_lj(q, epsilon, sigma, box, 3.9; mode = mode)
        @assert np == npb
        check_array([U], [Ub], 1e-11, "$mode orthorhombic energy")
        check_array(F, Fb, 1e-11, "$mode orthorhombic forces")
    end
    for bad in (4.0, 4.1, 6.0)
        expect_argument_error(() -> SL.evaluate!(ws, q, epsilon, sigma, box;
                                                 rcut = bad, mode = :lfs),
                              "rcut=$bad >= min(L)/2")
    end
    expect_argument_error(() -> SL.LJWorkspace(q, box, 4.0),
                          "constructor rcut >= min(L)/2")
    println("  orthorhombic rc < min(L)/2 safety verified")
    return true
end

# ---------------------------------------------------------------------------
# periodic-grid degeneracies and workspace reuse
# ---------------------------------------------------------------------------

function test_periodic_grid_degeneracies()
    rng = MersenneTwister(11)
    # nc = 2 per axis: the half-stencil would double count cell pairs; the
    # full-stencil + j>i fallback must still be exact.
    L = 7.0
    rc = 2.4
    @assert floor(Int, L / rc) == 2
    N = 18
    q = random_q(rng, N, (L, L, L))
    epsilon = fill(1.0, N)
    sigma = fill(1.0, N)
    box = T.OrthorhombicBox((L, L, L))
    topo, weights, _, _ = topology_fixture(rng, N; nex = 2, nsc = 2,
                                          q = q, box = box, rc = rc)

    ws = SL.LJWorkspace(q, box, rc)
    for mode in (:lfs, :shift, :trunc)
        U, F, np = SL.evaluate!(ws, q, epsilon, sigma, box;
                                rcut = rc, mode = mode, topology = topo)
        Ub, Fb, npb = brute_force_lj(q, epsilon, sigma, box, rc;
                                     mode = mode, weights = weights)
        @assert np == npb "nc=2 mode=$mode pair count $(np) != $(npb)"
        check_array([U], [Ub], 1e-11, "nc=2 $mode energy")
        check_array(F, Fb, 1e-11, "nc=2 $mode forces")
        Ut, Ft, npt = SL.evaluate!(ws, q, epsilon, sigma, box;
                                   rcut=rc, mode=mode, topology=topo, threaded=true)
        @assert npt == npb "threaded nc=2 $mode pair count"
        check_array([Ut], [Ub], 1e-11, "threaded nc=2 $mode energy")
        check_array(Ft, Fb, 1e-11, "threaded nc=2 $mode forces")
        println("  nc=2 $mode: pairs=$np")
    end

    # A skin large enough to force a one-cell grid (nc = 1) is also handled,
    # though it degenerates to an O(N^2) scan of one cell.
    L1 = 5.0
    rc1 = 2.0
    skin1 = 4.0  # rsearch = 6 > L1
    @assert floor(Int, L1 / (rc1 + skin1)) == 0
    N1 = 12
    q1 = random_q(rng, N1, (L1, L1, L1))
    eps1 = fill(0.9, N1)
    sig1 = fill(1.1, N1)
    box1 = T.OrthorhombicBox((L1, L1, L1))
    ws1 = SL.LJWorkspace(q1, box1, rc1; skin = skin1)
    for mode in (:lfs, :shift)
        U1, F1, np1 = SL.evaluate!(ws1, q1, eps1, sig1, box1;
                                   rcut = rc1, skin = skin1, mode = mode)
        Ub1, Fb1, npb1 = brute_force_lj(q1, eps1, sig1, box1, rc1; mode = mode)
        @assert np1 == npb1 "nc=1 mode=$mode pair count $(np1) != $(npb1)"
        check_array([U1], [Ub1], 1e-11, "nc=1 $mode energy")
        check_array(F1, Fb1, 1e-11, "nc=1 $mode forces")
        U1t, F1t, n1t = SL.evaluate!(ws1, q1, eps1, sig1, box1;
                                     rcut=rc1, skin=skin1, mode=mode, threaded=true)
        @assert n1t == npb1 "threaded nc=1 $mode pair count"
        check_array([U1t], [Ub1], 1e-11, "threaded nc=1 $mode energy")
        check_array(F1t, Fb1, 1e-11, "threaded nc=1 $mode forces")
        println("  nc=1 $mode: pairs=$np1")
    end
    return true
end

function test_workspace_reuse_and_skin()
    rng = MersenneTwister(31)
    N = 16
    L = 8.0
    rc = 1.8
    skin = 0.2  # rsearch = 2.0 -> nc=4 per axis for L=8
    q = random_q(rng, N, (L, L, L))
    epsilon = 0.7 .+ 0.3 .* rand(rng, N)
    sigma = 0.9 .+ 0.2 .* rand(rng, N)
    box = T.OrthorhombicBox((L, L, L))
    topo, weights, _, _ = topology_fixture(rng, N; nex = 2, nsc = 2,
                                          q = q, box = box, rc = rc)

    ws = SL.LJWorkspace(q, box, rc; skin = skin)
    for mode in (:lfs, :shift)
        U1, F1, np1 = SL.evaluate!(ws, q, epsilon, sigma, box;
                                   rcut = rc, skin = skin, mode = mode,
                                   topology = topo)
        U2, F2, np2 = SL.evaluate!(ws, q, epsilon, sigma, box;
                                   rcut = rc, skin = skin, mode = mode,
                                   topology = topo)
        @assert np1 == np2
        @assert U1 == U2
        @assert F1 == F2
        Ub, Fb, npb = brute_force_lj(q, epsilon, sigma, box, rc;
                                     mode = mode, weights = weights)
        @assert np1 == npb "skin mode=$mode pair count $(np1) != $(npb)"
        check_array([U1], [Ub], 1e-11, "skin $mode energy")
        check_array(F1, Fb, 1e-11, "skin $mode forces")

        # Gradient determinism on the same reusable workspace.
        _, _, _, ge1, gs1 = SL.evaluate_with_gradients!(
            ws, q, epsilon, sigma, box; rcut = rc, skin = skin, mode = mode,
            topology = topo)
        _, _, _, ge2, gs2 = SL.evaluate_with_gradients!(
            ws, q, epsilon, sigma, box; rcut = rc, skin = skin, mode = mode,
            topology = topo)
        @assert ge1 == ge2 && gs1 == gs2
        println("  $mode workspace reuse + skin: pairs=$np1 deterministic")
    end

    # Changing N must resize every workspace-owned array, including gradients.
    M = 7
    qm = random_q(rng, M, (L, L, L))
    em = 0.8 .+ 0.2 .* rand(rng, M)
    sm = 0.9 .+ 0.2 .* rand(rng, M)
    Um, Fm, npm = SL.evaluate!(ws, qm, em, sm, box; rcut = rc, mode = :lfs)
    @assert size(ws.forces) == (3, M)
    @assert length(ws.grad_epsilon) == M && length(ws.grad_sigma) == M
    Ubm, Fbm, npbm = brute_force_lj(qm, em, sm, box, rc; mode = :lfs)
    @assert npm == npbm
    check_array([Um], [Ubm], 1e-11, "resized workspace energy")
    check_array(Fm, Fbm, 1e-11, "resized workspace forces")
    println("  workspace resized from N=$N to N=$M")
    return true
end

# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

function test_invalid_inputs_do_not_mutate_forces()
    rng = MersenneTwister(21)
    N = 10
    L = 6.0
    rc = 2.0
    q = random_q(rng, N, (L, L, L))
    epsilon = fill(1.0, N)
    sigma = fill(1.0, N)
    box = T.OrthorhombicBox((L, L, L))
    topo = SL.PairTopology(excluded = [(1, 2)], scaled = [(3, 4, 0.5)])

    ws = SL.LJWorkspace(q, box, rc)
    SL.evaluate!(ws, q, epsilon, sigma, box; rcut = rc, mode = :lfs, topology = topo)
    F_before = copy(ws.forces)
    U_before = sum(F_before)
    grad_before = copy(ws.grad_epsilon)
    gsig_before = copy(ws.grad_sigma)

    bad_eps = copy(epsilon); bad_eps[3] = -1.0
    expect_argument_error(() -> SL.evaluate!(ws, q, bad_eps, sigma, box;
                                             rcut = rc, mode = :lfs),
                          "negative epsilon")
    bad_sig = copy(sigma); bad_sig[1] = 0.0
    expect_argument_error(() -> SL.evaluate!(ws, q, epsilon, bad_sig, box;
                                             rcut = rc, mode = :lfs),
                          "zero sigma")
    bad_len = ones(N - 1)
    expect_argument_error(() -> SL.evaluate!(ws, q, bad_len, sigma, box;
                                             rcut = rc, mode = :lfs),
                          "short epsilon")
    bad_q = copy(q); bad_q[2, 4] = NaN
    expect_argument_error(() -> SL.evaluate!(ws, bad_q, epsilon, sigma, box;
                                             rcut = rc, mode = :lfs),
                          "NaN coordinate")
    expect_argument_error(() -> SL.evaluate!(ws, q, epsilon, sigma, T.NoBox();
                                             rcut = rc, mode = :lfs),
                          "NoBox")
    expect_argument_error(() -> SL.evaluate!(ws, q, epsilon, sigma, box;
                                             rcut = 0.5 * L, mode = :lfs),
                          "rcut >= min(L)/2")
    expect_argument_error(() -> SL.evaluate!(ws, q, epsilon, sigma, box;
                                             rcut = rc, skin = -1.0, mode = :lfs),
                          "negative skin")
    for mode in (:none, "lfs", :bogus)
        expect_argument_error(() -> SL.evaluate!(ws, q, epsilon, sigma, box;
                                                 rcut = rc, mode = mode),
                              "unsupported mode $(repr(mode))")
    end

    # A topology built for a larger coordinate set must be refused, not ignored.
    bad_topo = SL.PairTopology(excluded = [(1, N + 3)])
    expect_argument_error(() -> SL.evaluate!(ws, q, epsilon, sigma, box;
                                             rcut = rc, mode = :lfs,
                                             topology = bad_topo),
                          "topology atom index beyond N")
    expect_argument_error(() -> SL.evaluate!(ws, q, epsilon, sigma, box;
                                             rcut = rc, mode = :lfs,
                                             topology = Dict((1, 2) => 0.5)),
                          "topology is not a PairTopology")

    # Gradient path refuses epsilon == 0 and leaves both forces and gradients alone.
    zero_eps = copy(epsilon); zero_eps[2] = 0.0
    expect_argument_error(() -> SL.evaluate_with_gradients!(ws, q, zero_eps, sigma,
                                                            box; rcut = rc,
                                                            mode = :lfs,
                                                            topology = topo),
                          "zero epsilon in gradient path")

    @assert ws.forces == F_before "invalid inputs mutated the force array"
    @assert sum(ws.forces) == U_before
    @assert ws.grad_epsilon == grad_before
    @assert ws.grad_sigma == gsig_before
    println("  invalid inputs rejected before force/parameter-gradient mutation")
    return true
end

function test_topology_validation_errors()
    expect_argument_error(() -> SL.PairTopology(excluded = [(1, 1)]),
                          "i == j")
    expect_argument_error(() -> SL.PairTopology(excluded = [(1, 2), (2, 1)]),
                          "duplicate canonical exclusion")
    expect_argument_error(() -> SL.PairTopology(scaled = [(1, 2, 0.5), (2, 1, 0.5)]),
                          "duplicate canonical scaled pair")
    expect_argument_error(() -> SL.PairTopology(excluded = [(1, 2)],
                                                scaled = [(2, 1, 0.5)]),
                          "pair both excluded and scaled")
    expect_argument_error(() -> SL.PairTopology(scaled = [(1, 2, -0.1)]),
                          "negative scale")
    expect_argument_error(() -> SL.PairTopology(scaled = [(1, 2, NaN)]),
                          "NaN scale")
    expect_argument_error(() -> SL.PairTopology(scaled = [(1, 2, Inf)]),
                          "infinite scale")
    expect_argument_error(() -> SL.PairTopology(scaled = [(1, 2, "half")]),
                          "non-numeric scale")
    expect_argument_error(() -> SL.PairTopology(excluded = [(0, 2)]),
                          "zero index")
    expect_argument_error(() -> SL.PairTopology(excluded = [(-1, 2)]),
                          "negative index")
    expect_argument_error(() -> SL.PairTopology(excluded = [(1, 2 .^ 33)]),
                          "index above UInt32")
    expect_argument_error(() -> SL.PairTopology(excluded = [(1, 2, 3)]),
                          "three-tuple in excluded")
    expect_argument_error(() -> SL.PairTopology(excluded = [42]),
                          "non-pair excluded entry")

    t = SL.PairTopology()
    @assert SL.pair_weight(t, 1, 2) == 1.0
    @assert length(t.excluded) == 0 && length(t.scaled_keys) == 0
    expect_argument_error(() -> SL.pair_weight(t, 2, 2), "pair_weight i == j")
    expect_argument_error(() -> SL.pair_weight(t, 0, 2), "pair_weight zero index")
    @assert occursin("PairTopology", sprint(show, t))
    println("  compact topology validation errors verified")
    return true
end

function test_memory_estimator()
    box = T.OrthorhombicBox((20.0, 20.0, 20.0))
    est = SL.estimate_memory_bytes(1000, box, 2.5; skin = 0.0)
    estg = SL.estimate_memory_bytes(1000, box, 2.5; skin = 0.0, gradients = true)
    @assert est isa Int
    @assert est > 68 * 1000 "estimate must include coords/forces/arrays and overhead"
    @assert estg > est + 2 * 8 * 1000
    println("  memory estimate for N=1000: $est bytes ($estg with gradients)")
    return true
end

# ---------------------------------------------------------------------------
# TriclinicBox tests: parity, box derivatives, cutoff continuity, wrapping
# ---------------------------------------------------------------------------

function test_triclinic_parity_and_stencils()
    rng = MersenneTwister(42)
    H = [8.5  1.8  0.9;
         0.0  8.0  1.4;
         0.0  0.0  9.5]
    box = T.TriclinicBox(H)
    N = 25
    S = 0.1 .+ 0.8 .* rand(rng, 3, N)
    q = H * S
    epsilon = 0.5 .+ rand(rng, N)
    sigma = 0.8 .+ 0.4 .* rand(rng, N)
    rc = 3.2

    # Topology: 1-2 and 1-3 exclusions, 1-4 scaling
    excluded = [(1, 2), (2, 3), (4, 5)]
    scaled = [(1, 4, 0.5), (3, 6, 0.4)]
    topo = SL.PairTopology(excluded = excluded, scaled = scaled)
    weights = Dict{Tuple{Int,Int},Float64}((1, 2) => 0.0, (2, 3) => 0.0, (4, 5) => 0.0,
                                           (1, 4) => 0.5, (3, 6) => 0.4)

    ws = SL.LJWorkspace(q, box, rc)
    for mode in (:lfs, :shift, :trunc)
        # Without topology
        U, F, np = SL.evaluate!(ws, q, epsilon, sigma, box; rcut = rc, mode = mode)
        Ub, Fb, npb = brute_force_lj(q, epsilon, sigma, box, rc; mode = mode)
        @assert np == npb "triclinic $mode pair count mismatch: $np vs $npb"
        check_array([U], [Ub], 1e-12, "triclinic $mode energy (no topology)")
        check_array(F, Fb, 1e-12, "triclinic $mode forces (no topology)")

        # With topology
        Ut, Ft, npt = SL.evaluate!(ws, q, epsilon, sigma, box; rcut = rc, mode = mode, topology = topo)
        Ubt, Fbt, npbt = brute_force_lj(q, epsilon, sigma, box, rc; mode = mode, weights = weights)
        @assert npt == npbt "triclinic $mode with topology pair count mismatch: $npt vs $npbt"
        check_array([Ut], [Ubt], 1e-12, "triclinic $mode energy (with topology)")
        check_array(Ft, Fbt, 1e-12, "triclinic $mode forces (with topology)")
    end

    # Test small skew box triggering _stream_full! (nc < 3)
    H_small = [5.5  1.2  0.7;
               0.0  5.0  1.0;
               0.0  0.0  6.0]
    box_small = T.TriclinicBox(H_small)
    N_s = 15
    q_s = H_small * (0.05 .+ 0.9 .* rand(rng, 3, N_s))
    eps_s = fill(1.0, N_s)
    sig_s = fill(1.0, N_s)
    rc_s = 2.2
    ws_s = SL.LJWorkspace(q_s, box_small, rc_s)
    for mode in (:lfs, :shift, :trunc)
        U, F, np = SL.evaluate!(ws_s, q_s, eps_s, sig_s, box_small; rcut = rc_s, mode = mode, topology = topo)
        Ub, Fb, npb = brute_force_lj(q_s, eps_s, sig_s, box_small, rc_s; mode = mode, weights = weights)
        @assert np == npb "small triclinic $mode pair count: $np vs $npb"
        check_array([U], [Ub], 1e-12, "small triclinic $mode energy")
        check_array(F, Fb, 1e-12, "small triclinic $mode forces")
    end

    println("  triclinic parity and full/half stencils verified across all modes")
    return true
end

function test_triclinic_box_derivative_fd()
    H = [6.0  1.0  0.7;
         0.0  5.5  0.9;
         0.0  0.0  6.5]
    box = T.TriclinicBox(H)
    N = 8
    S = zeros(3, N)
    idx = 1
    for x in (0.3, 0.7), y in (0.3, 0.7), z in (0.3, 0.7)
        S[:, idx] .= [x, y, z]
        idx += 1
    end
    q = H * S
    epsilon = fill(1.0, N)
    sigma = fill(1.0, N)
    rc = 2.6

    excluded = [(1, 2)]
    scaled = [(1, 3, 0.5)]
    topo = SL.PairTopology(excluded = excluded, scaled = scaled)
    weights = Dict{Tuple{Int,Int},Float64}((1, 2) => 0.0, (1, 3) => 0.5)

    ws = SL.LJWorkspace(q, box, rc)

    # 1. Analytic box derivative vs brute-force reference
    G = SL.box_derivative(ws, q, epsilon, sigma, box; rcut = rc, mode = :lfs, topology = topo)
    G_ref = brute_force_lj_box_derivative(q, epsilon, sigma, box, rc; mode = :lfs, weights = weights)
    check_array(G, G_ref, 1e-12, "triclinic analytic G vs brute force G")

    # 2. evaluate_with_box_derivative! agreement
    U, F, np, G2 = SL.evaluate_with_box_derivative!(ws, q, epsilon, sigma, box;
                                                   rcut = rc, mode = :lfs, topology = topo)
    @assert isequal(G, G2) "evaluate_with_box_derivative! matrix does not match box_derivative"

    # 3. Central finite differences ∂U/∂h|_s across all 9 matrix entries
    steps = (1e-3, 1e-4, 1e-5)
    errs = Float64[]
    for eps_h in steps
        max_err = 0.0
        for a in 1:3, b in 1:3
            Hp = copy(H); Hp[a, b] += eps_h
            Hm = copy(H); Hm[a, b] -= eps_h
            box_p = T.TriclinicBox(Hp)
            box_m = T.TriclinicBox(Hm)
            qp = Hp * S
            qm = Hm * S
            Up, _, _ = SL.evaluate!(ws, qp, epsilon, sigma, box_p; rcut = rc, mode = :lfs, topology = topo)
            Um, _, _ = SL.evaluate!(ws, qm, epsilon, sigma, box_m; rcut = rc, mode = :lfs, topology = topo)
            g_fd = (Up - Um) / (2.0 * eps_h)
            max_err = max(max_err, abs(G[a, b] - g_fd))
        end
        push!(errs, max_err)
    end

    # Truncation error falls quadratically (~100x per 10x step reduction) down to machine cancellation
    @assert errs[2] < errs[1] / 10 "triclinic box derivative FD not converging: $errs"
    @assert errs[2] < 1e-8 && errs[3] < 1e-10 "triclinic box derivative FD errors: $errs"
    println("  triclinic box derivative ∂U/∂h|_s verified against central FD: max_err=$(errs[3]) at 1e-5")
    return true
end

function test_triclinic_near_cutoff()
    H = [8.0  1.5  0.8;
         0.0  7.5  1.2;
         0.0  0.0  9.0]
    box = T.TriclinicBox(H)
    rc = 3.0
    epsilon = [1.0, 1.0]
    sigma = [1.0, 1.0]

    # Place two atoms near cutoff along the first lattice direction
    delta = 1e-5
    r_target = rc - delta
    s1 = [0.2, 0.3, 0.4]
    # Unit vector along H[:, 1]
    u1 = H[:, 1] ./ norm(H[:, 1])
    q1 = H * s1
    q2 = q1 .+ r_target .* u1
    q = hcat(q1, q2)

    ws = SL.LJWorkspace(q, box, rc)
    U_in, F_in, np_in, G_in = SL.evaluate_with_box_derivative!(ws, q, epsilon, sigma, box;
                                                              rcut = rc, mode = :lfs)
    @assert np_in == 1 "near-cutoff inside pair must be counted"
    # In :lfs mode, U is O(delta^2), forces and G are O(delta)
    @assert abs(U_in) < 1e-8 "near-cutoff LFS energy must vanish quadratically: $U_in"
    @assert maximum(abs, F_in) < 1e-3 "near-cutoff LFS forces must vanish linearly: $(maximum(abs, F_in))"
    @assert maximum(abs, G_in) < 1e-3 "near-cutoff LFS box derivative must vanish linearly: $(maximum(abs, G_in))"

    # Finite difference of box derivative near cutoff
    S = box.Hinv * q
    eps_h = 1e-5
    for a in 1:3, b in 1:3
        Hp = copy(H); Hp[a, b] += eps_h
        Hm = copy(H); Hm[a, b] -= eps_h
        Up = SL.evaluate!(ws, Hp * S, epsilon, sigma, T.TriclinicBox(Hp); rcut = rc, mode = :lfs)[1]
        Um = SL.evaluate!(ws, Hm * S, epsilon, sigma, T.TriclinicBox(Hm); rcut = rc, mode = :lfs)[1]
        g_fd = (Up - Um) / (2.0 * eps_h)
        @assert abs(G_in[a, b] - g_fd) < 1e-7 "near-cutoff box derivative FD mismatch at ($a,$b)"
    end

    # Place pair just outside cutoff: r = rc + delta
    q2_out = q1 .+ (rc + delta) .* u1
    q_out = hcat(q1, q2_out)
    U_out, F_out, np_out, G_out = SL.evaluate_with_box_derivative!(ws, q_out, epsilon, sigma, box;
                                                                  rcut = rc, mode = :lfs)
    @assert np_out == 0 "near-cutoff outside pair must not interact"
    @assert U_out == 0.0 && all(==(0.0), F_out) && all(==(0.0), G_out)

    println("  triclinic near-cutoff C1 continuity verified for energy, forces, and box derivative")
    return true
end

function test_triclinic_force_fd()
    rng = MersenneTwister(55)
    H = [7.0  1.2  0.7;
         0.0  6.5  1.0;
         0.0  0.0  7.5]
    box = T.TriclinicBox(H)
    N = 10
    q = H * (0.15 .+ 0.7 .* rand(rng, 3, N))
    epsilon = 0.7 .+ 0.4 .* rand(rng, N)
    sigma = 0.9 .+ 0.2 .* rand(rng, N)
    rc = 2.8

    excluded = [(1, 2)]
    scaled = [(1, 3, 0.5)]
    topo = SL.PairTopology(excluded = excluded, scaled = scaled)

    ws = SL.LJWorkspace(q, box, rc)
    U, F, _ = SL.evaluate!(ws, q, epsilon, sigma, box; rcut = rc, mode = :lfs, topology = topo)

    # Central FD forces
    h_coord = 1e-6
    F_fd = zeros(3, N)
    qp = copy(q); qm = copy(q)
    for j in 1:N, a in 1:3
        qp[a, j] += h_coord
        qm[a, j] -= h_coord
        Up, _, _ = SL.evaluate!(ws, qp, epsilon, sigma, box; rcut = rc, mode = :lfs, topology = topo)
        Um, _, _ = SL.evaluate!(ws, qm, epsilon, sigma, box; rcut = rc, mode = :lfs, topology = topo)
        F_fd[a, j] = -(Up - Um) / (2.0 * h_coord)
        qp[a, j] -= h_coord
        qm[a, j] += h_coord
    end

    check_array(F, F_fd, 1e-5, "triclinic Cartesian forces vs central FD")
    println("  triclinic Cartesian forces verified against central FD")
    return true
end

function test_triclinic_wrapping_and_safety()
    H = [7.0  1.4  0.6;
         0.0  6.8  1.1;
         0.0  0.0  8.0]
    box = T.TriclinicBox(H)
    rng = MersenneTwister(71)
    N = 8
    S = 0.2 .+ 0.6 .* rand(rng, 3, N)
    q = H * S
    epsilon = fill(1.0, N)
    sigma = fill(1.0, N)
    rc = 2.5
    ws = SL.LJWorkspace(q, box, rc)

    U0, F0, np0, G0 = SL.evaluate_with_box_derivative!(ws, q, epsilon, sigma, box; rcut = rc, mode = :lfs)

    # Shift several atoms by arbitrary integer lattice vectors
    q_wrapped = copy(q)
    shifts = ([1, 0, 0], [-1, 2, 0], [0, -1, 1], [2, -1, -2])
    for j in 1:N
        shift_vec = H * shifts[mod1(j, length(shifts))]
        q_wrapped[:, j] .+= shift_vec
    end

    U_w, F_w, np_w, G_w = SL.evaluate_with_box_derivative!(ws, q_wrapped, epsilon, sigma, box;
                                                          rcut = rc, mode = :lfs)
    @assert np_w == np0 "wrapping changed pair count: $np_w vs $np0"
    check_array([U_w], [U0], 1e-12, "triclinic wrapped energy")
    check_array(F_w, F0, 1e-12, "triclinic wrapped forces")
    check_array(G_w, G0, 1e-12, "triclinic wrapped box derivative")

    # Cutoff safety: rcut < 0.5 * min(widths)
    minW = minimum(box.widths)
    # Valid cutoff
    ws_safe = SL.LJWorkspace(q, box, 0.49 * minW)
    @assert ws_safe isa SL.LJWorkspace
    # Invalid cutoff
    expect_argument_error(() -> SL.LJWorkspace(q, box, 0.501 * minW),
                          "rcut >= min(widths)/2")
    expect_argument_error(() -> SL.evaluate!(ws, q, epsilon, sigma, box; rcut = 0.501 * minW),
                          "evaluate! rcut >= min(widths)/2")

    println("  triclinic periodic wrapping and cutoff safety verified")
    return true
end

function test_triclinic_streamed_factor()
    H = [7.5  1.2  0.8;
         0.0  7.0  1.0;
         0.0  0.0  8.0]
    box = T.TriclinicBox(H)
    rng = MersenneTwister(88)
    N = 10
    q = H * (0.2 .+ 0.6 .* rand(rng, 3, N))
    epsilon = 0.8 .+ 0.4 .* rand(rng, N)
    sigma = 1.0 .+ 0.2 .* rand(rng, N)
    rc = 2.7
    topo = SL.PairTopology(excluded = [(1, 2)], scaled = [(1, 3, 0.5)])

    fac = SL.StreamedLJFactor(q, epsilon, sigma, box; rcut = rc, mode = :lfs, topology = topo)
    ctx = FA.Context(box, FA.REDUCED, N)

    F_buf = zeros(Float64, 3, N)
    U_fac = FA.evaluate!(F_buf, fac, q, ctx)
    U_ref, F_ref, _, G_ref = SL.evaluate_with_box_derivative!(fac.ws, q, epsilon, sigma, box;
                                                             rcut = rc, mode = :lfs, topology = topo)
    check_array([U_fac], [U_ref], 1e-14, "StreamedLJFactor triclinic energy")
    check_array(F_buf, F_ref, 1e-14, "StreamedLJFactor triclinic forces")

    # streamed_box_derivative
    G_fac = SL.streamed_box_derivative(fac, q, ctx)
    check_array(G_fac, G_ref, 1e-14, "StreamedLJFactor triclinic box derivative")

    # streamed_parameter_gradients!
    U_grad, dEps, dSig = SL.streamed_parameter_gradients!(fac, q, ctx)
    @assert isfinite(U_grad) && all(isfinite, dEps) && all(isfinite, dSig)

    # Context box mismatch check
    H_other = [7.6 1.2 0.8; 0.0 7.0 1.0; 0.0 0.0 8.0]
    ctx_bad = FA.Context(T.TriclinicBox(H_other), FA.REDUCED, N)
    expect_argument_error(() -> FA.evaluate!(F_buf, fac, q, ctx_bad), "mismatched context lattice")

    println("  StreamedLJFactor triclinic evaluation and box derivative verified")
    return true
end

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

function main()
    println("=" ^ 72)
    println("StreamedLJ tests")
    println("=" ^ 72)
    test_parity_lfs_shift_and_topology()
    test_thread_counts()
    test_serial_allocation_parity()
    test_explicit_1_2_1_3_exclusion_and_1_4_scaling()
    test_finite_difference_forces()
    test_parameter_gradients()
    test_cutoff_continuity_and_wrapping()
    test_orthorhombic_cutoff_safety()
    test_periodic_grid_degeneracies()
    test_workspace_reuse_and_skin()
    test_invalid_inputs_do_not_mutate_forces()
    test_topology_validation_errors()
    test_memory_estimator()
    test_triclinic_parity_and_stencils()
    test_triclinic_box_derivative_fd()
    test_triclinic_near_cutoff()
    test_triclinic_force_fd()
    test_triclinic_wrapping_and_safety()
    test_triclinic_streamed_factor()
    println("STREAMED LJ TESTS PASSED")
    return true
end

if get(ENV, "STREAMED_LJ_THREADS_CHILD", "") == "1"
    println("THREAD_SIGNATURE ", test_threaded_force_path())
else
    main()
end
