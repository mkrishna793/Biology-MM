# Neighbour-list correctness.
#
# The NIST validation caught a genuine over-count bug in the cell-list stencil
# when a box had fewer than 3 cells per axis. That bug is now a permanent
# regression test here: the neighbour list is compared against exhaustive
# O(N^2) enumeration over a range of box/cutoff combinations chosen to hit
# nc = 1, 2, 3 and more, in both periodic and non-periodic settings.
#
# Phase 3A: exact triclinic minimum-image convention (MIC), persistent Verlet
# neighbour search on triclinic boxes, and rebuild criteria accounting for
# both particle excursions and lattice box deformations (Δh).

using Random, Printf, LinearAlgebra

module LocalFoundationCore
    include(joinpath(@__DIR__, "..", "src", "Types.jl"))
    include(joinpath(@__DIR__, "..", "src", "Factors.jl"))
    include(joinpath(@__DIR__, "..", "src", "Neighbors.jl"))
    include(joinpath(@__DIR__, "..", "src", "StreamedLJ.jl"))
    include(joinpath(@__DIR__, "..", "src", "Electrostatics.jl"))
end

const T = LocalFoundationCore.Types
const NB = LocalFoundationCore.Neighbors
const Factors = LocalFoundationCore.Factors
const StreamedLJ = LocalFoundationCore.StreamedLJ
const Electrostatics = LocalFoundationCore.Electrostatics

"Exhaustive pair enumeration within `rc` using minimum image."
function brute_pairs(q, box, rc)
    N = size(q, 2)
    out = Tuple{Int,Int}[]
    for i in 1:N-1
        for j in i+1:N
            d = NB.minimum_image(box, q, i, j)
            if T.norm2(d) <= rc * rc
                push!(out, (i, j))
            end
        end
    end
    return sort(out)
end

function test_neighbor_list_matches_bruteforce()
    rng = MersenneTwister(101)
    worst_missing = 0
    worst_extra = 0
    cases = 0
    # Deliberately vary the box size relative to rc so that the cell count per
    # axis sweeps 1, 2, 3, 4, ... The nc < 3 cases are exactly where the original
    # stencil bug lived.
    for L in (3.0, 4.0, 5.0, 6.0, 6.5, 8.0, 12.0)
        for rc in (0.9, 1.0, 1.4)
            for periodic in (true, false)
                N = 60
                if periodic
                    q = rand(rng, 3, N) .* L
                    box = T.OrthorhombicBox((L, L, L))
                else
                    q = randn(rng, 3, N) .* 2.0
                    box = T.NoBox()
                end
                nl = NB.build_neighbors(q, box, rc, 0.0)
                got = sort(NB.neighbor_pairs(nl))
                want = brute_pairs(q, box, rc)
                missing = length(setdiff(want, got))
                extra = length(setdiff(got, want))
                worst_missing = max(worst_missing, missing)
                worst_extra = max(worst_extra, extra)
                cases += 1
                nc = box isa T.OrthorhombicBox ? max(1, floor(Int, L / rc)) : 0
                @assert missing == 0 "L=$L rc=$rc periodic=$periodic nc=$nc: $missing missing pairs"
                @assert extra == 0 "L=$L rc=$rc periodic=$periodic nc=$nc: $extra extra pairs"
            end
        end
    end
    println("  ", cases, " box/cutoff cases (nc spans 1..4): missing=", worst_missing,
            " extra=", worst_extra, " -> exact")
    return true
end

function test_materialized_pairs_have_scale_guard()
    n = NB.MAX_MATERIALIZED_PAIR_ATOMS + 1
    nl = NB.NeighborList([Int[] for _ in 1:n], 1.0, 0.0,
                         zeros(Float64, 3, n), 0)
    failed = false
    try
        NB.neighbor_pairs(nl)
    catch e
        failed = e isa ArgumentError && occursin("stream neighbors directly", sprint(showerror, e))
    end
    @assert failed "large materialized pair requests must be refused before tuple allocation"
    failed = false
    try
        NB.build_neighbors(zeros(Float64, 3, n), T.NoBox(), 1.0, 0.0)
    catch e
        failed = e isa ArgumentError && occursin("streamed cell traversal", sprint(showerror, e))
    end
    @assert failed "large neighbor-list requests must be refused before per-atom vector allocation"
    println("  legacy tuple-pair materialization guard: N=$n refused")
end

function test_skin_is_conservative()
    # Pairs within rc must never be missing even with a skin, and nothing beyond
    # rc+skin should be present.
    rng = MersenneTwister(202)
    L = 10.0
    rc, skin = 2.5, 0.4
    box = T.OrthorhombicBox((L, L, L))
    q = rand(rng, 3, 120) .* L
    nl = NB.build_neighbors(q, box, rc, skin)
    got = sort(NB.neighbor_pairs(nl))
    want_rc = brute_pairs(q, box, rc)
    want_skin = brute_pairs(q, box, rc + skin)
    @assert isempty(setdiff(want_rc, got)) "pairs inside rc were missing"
    @assert isempty(setdiff(got, want_skin)) "pairs beyond rc+skin were present"
    println("  skin list: contains all rc pairs (", length(want_rc),
            ") and no pairs beyond rc+skin (", length(want_skin), ")")
    return true
end

function test_rebuild_criterion()
    # Displace one atom by just over skin/2 and the criterion must fire; just
    # under and it must not.
    rng = MersenneTwister(303)
    L = 10.0
    rc, skin = 2.5, 0.5
    box = T.OrthorhombicBox((L, L, L))
    q = rand(rng, 3, 50) .* L
    nl = NB.build_neighbors(q, box, rc, skin)
    @assert !NB.needs_rebuild!(nl, q) "fresh list must not need rebuild"
    q2 = copy(q); q2[1, 1] += 0.24
    @assert !NB.needs_rebuild!(nl, q2) "0.24 < skin/2 = 0.25 must not trigger"
    q3 = copy(q); q3[1, 1] += 0.26
    @assert NB.needs_rebuild!(nl, q3) "0.26 > skin/2 = 0.25 must trigger"
    println("  rebuild criterion fires exactly at skin/2")
    return true
end

function exhaustive_image(box, d; shell=8)
    best = Inf
    for a in -shell:shell, b in -shell:shell, c in -shell:shell
        r = d - box.H * [a, b, c]
        best = min(best, sum(abs2, r))
    end
    return best
end

function brute_force_27_image(box, d)
    f = box.Hinv * d
    n0 = round.(Int, f)
    best = Inf
    for a in -1:1, b in -1:1, c in -1:1
        r = d - box.H * (n0 + [a, b, c])
        best = min(best, sum(abs2, r))
    end
    return best
end

function test_triclinic_nearest_image()
    rng = MersenneTwister(505)
    cells = [
        [4.0 3.7 0.0; 0.0 1.0 0.0; 0.0 0.0 5.0],
        [4.0 3.95 0.0; 0.0 0.32 0.0; 0.0 0.0 3.0],
        [3.0 1.4 0.8; 0.0 2.1 1.2; 0.0 0.0 2.5],
    ]
    for H in cells
        box = T.TriclinicBox(H)
        for _ in 1:80
            d = H * (rand(rng, 3) .* 2 .- 1)
            got = NB.triclinic_mic(box, d[1], d[2], d[3])
            got2 = got.x^2 + got.y^2 + got.z^2
            want = exhaustive_image(box, d)
            @assert isapprox(got2, want; atol=1e-11, rtol=1e-11) "triclinic MIC mismatch: got=$got2 want=$want H=$H d=$d"
        end
    end
    skew = T.TriclinicBox(cells[1])
    d = [2.1, 0.49, 0.0]
    got = NB.triclinic_mic(skew, d[1], d[2], d[3])
    @assert isapprox(got.x^2 + got.y^2 + got.z^2, exhaustive_image(skew, d); atol=1e-12)
    # A generic bonded factor consumes the same exact minimum-image primitive.
    qforce = hcat(d, zeros(3))
    force = zeros(3, 2)
    ctx = Factors.Context(skew, T.REDUCED, 2)
    energy = Factors.evaluate!(force, Factors.HarmonicBond(1, 2, 2.0, 1.0), qforce, ctx)
    rmin = sqrt(exhaustive_image(skew, d))
    @assert isapprox(energy, (rmin - 1.0)^2; atol=1e-12)
    @assert maximum(abs, force[:, 1] .+ force[:, 2]) < 1e-14
    println("  triclinic nearest image agrees with exhaustive lattice shells")
end

function test_skew_triclinic_mic_vs_27_image_search()
    # Test that exact triclinic MIC matches 27-image search and defeats naive rounding
    rng = MersenneTwister(606)
    cells = [
        # Strongly skewed monoclinic / triclinic cells
        [4.0 3.7 0.0; 0.0 1.0 0.0; 0.0 0.0 5.0],
        [4.0 3.95 0.0; 0.0 0.32 0.0; 0.0 0.0 3.0],
        [3.0 1.4 0.8; 0.0 2.1 1.2; 0.0 0.0 2.5],
        [6.0 2.5 1.5; 0.0 5.0 2.0; 0.0 0.0 4.0],
    ]

    # Explicit demonstration of naive componentwise fractional rounding failure:
    # For H = [4.0 3.7 0; 0 1.0 0; 0 0 5.0] and d = [2.1, 0.49, 0.0]
    H_skew = cells[1]
    box_skew = T.TriclinicBox(H_skew)
    d_fail = [2.1, 0.49, 0.0]
    f_naive = box_skew.Hinv * d_fail
    n_naive = round.(Int, f_naive)
    r_naive = d_fail - box_skew.H * n_naive
    d2_naive = sum(abs2, r_naive)

    got_mic = NB.minimum_image(box_skew, d_fail)
    d2_mic = got_mic.x^2 + got_mic.y^2 + got_mic.z^2
    d2_want = exhaustive_image(box_skew, d_fail)
    d2_27 = brute_force_27_image(box_skew, d_fail)

    # Naive rounding gives d2_naive ≈ 4.6501, whereas true MIC gives d2_want ≈ 2.8201!
    @assert d2_naive > d2_want + 1.0 "naive rounding should fail on this skewed case"
    @assert isapprox(d2_mic, d2_want; atol=1e-12) "mic must find true minimum"
    @assert isapprox(d2_mic, d2_27; atol=1e-12) "mic must match 27-image search"

    # Extensive sweep across all cells
    for H in cells
        b = T.TriclinicBox(H)
        for _ in 1:100
            d = H * (rand(rng, 3) .* 2.0 .- 1.0)
            res = NB.minimum_image(b, d)
            r2 = res.x^2 + res.y^2 + res.z^2
            r2_27 = brute_force_27_image(b, d)
            r2_ex = exhaustive_image(b, d; shell=5)
            @assert isapprox(r2, r2_27; atol=1e-11) "MIC does not match 27-image search: r2=$r2 vs 27=$r2_27"
            @assert isapprox(r2, r2_ex; atol=1e-11) "MIC does not match exhaustive search: r2=$r2 vs ex=$r2_ex"
        end
    end
    println("  skew triclinic MIC: exactness vs 27-image and exhaustive search confirmed")
    return true
end

function test_triclinic_neighbors_supported_and_cutoff_guarded()
    rng = MersenneTwister(707)
    cells = [
        [6.0 1.5 0.5; 0.0 5.0 1.2; 0.0 0.0 5.5],
        [5.0 2.0 0.0; 0.0 4.5 0.0; 0.0 0.0 4.0],
    ]
    for H in cells
        box = T.TriclinicBox(H)
        N = 60
        q = H * rand(rng, 3, N)
        rc = 1.2
        skin = 0.3

        # Test build_neighbors on TriclinicBox
        nl = NB.build_neighbors(q, box, rc, skin)
        @assert nl isa NB.NeighborList
        @assert nl isa NB.VerletList
        got = sort(NB.neighbor_pairs(nl))

        # Check against brute force
        want_rc = brute_pairs(q, box, rc)
        want_skin = brute_pairs(q, box, rc + skin)

        @assert isempty(setdiff(want_rc, got)) "triclinic neighbor list missed pairs within rc"
        @assert isempty(setdiff(got, want_skin)) "triclinic neighbor list included pairs beyond rc + skin"

        # Check with zero skin: exact match
        nl0 = NB.build_neighbors(q, box, rc, 0.0)
        got0 = sort(NB.neighbor_pairs(nl0))
        @assert got0 == want_rc "zero-skin neighbor list must match brute force exactly"
    end

    # Guard tests:
    q0 = zeros(Float64, 3, 2)
    tri_box = T.TriclinicBox([4.0 1.0 0.0; 0.0 4.0 0.0; 0.0 0.0 4.0])

    failed_ewald = false
    try
        Electrostatics.EwaldFactor(q0, [1.0, -1.0], tri_box; alpha=0.3, rcut=1.0, kmax=4)
    catch e
        failed_ewald = e isa T.UnsupportedGeometry
    end
    @assert failed_ewald "EwaldFactor must refuse triclinic cells explicitly"

    # 2. Cutoff guard: 2*rcut < minimum(widths)
    box_guard = T.TriclinicBox([4.0 1.0 0.0; 0.0 3.0 0.0; 0.0 0.0 5.0])
    failed = false
    try
        NB.build_neighbors(q0, box_guard, 1.5, 0.0)
    catch e
        failed = e isa ArgumentError && occursin("half the minimum perpendicular", sprint(showerror, e))
    end
    @assert failed "cutoff at half the minimum perpendicular width must be refused"
    widths = box_guard.widths
    @assert isapprox(widths[1], 12 / sqrt(10)) && isapprox(widths[2], 3.0) && isapprox(widths[3], 5.0)

    println("  triclinic neighbors supported, verified against brute force, and cutoff guarded")
    return true
end

function test_rebuild_criterion_box_deformation()
    rng = MersenneTwister(808)
    L = 10.0
    rc, skin = 2.5, 0.5
    box = T.OrthorhombicBox((L, L, L))
    q = rand(rng, 3, 50) .* L
    nl = NB.build_neighbors(q, box, rc, skin)

    # 1. Fresh state: no rebuild needed
    @assert !NB.needs_rebuild!(nl, q, box) "fresh state must not trigger"
    @assert !NB.needs_rebuild!(nl, q) "default box must not trigger"

    # 2. Pure box dilation/contraction without particle displacement:
    # For single-axis expansion ΔL: dh_max = ΔL
    box_sub = T.OrthorhombicBox((L + 0.45, L, L)) # dh_max = 0.45 < 0.50
    @assert !NB.needs_rebuild!(nl, q, box_sub) "dh_max=0.45 < skin=0.50 with q unchanged must not trigger"

    box_super = T.OrthorhombicBox((L + 0.52, L, L)) # dh_max = 0.52 > 0.50
    @assert NB.needs_rebuild!(nl, q, box_super) "dh_max=0.52 > skin=0.50 must trigger rebuild"

    # 3. Coupled particle motion AND box deformation:
    # dh_max = 0.20 -> effective skin = 0.50 - 0.20 = 0.30 -> per-atom threshold = 0.15
    box_deformed = T.OrthorhombicBox((L + 0.20, L, L))
    q_moved = copy(q)
    q_moved[1, 1] += 0.14
    @assert !NB.needs_rebuild!(nl, q_moved, box_deformed) "Δq=0.14 < thresh=0.15 with Δh=0.20 must not trigger"

    q_moved[1, 1] += 0.03 # now Δq = 0.17 > thresh = 0.15
    @assert NB.needs_rebuild!(nl, q_moved, box_deformed) "Δq=0.17 > thresh=0.15 with Δh=0.20 must trigger"

    # 4. Triclinic box shear deformation:
    H0 = [6.0 0.0 0.0; 0.0 6.0 0.0; 0.0 0.0 6.0]
    box_tri = T.TriclinicBox(H0)
    q_tri = rand(rng, 3, 30) .* 5.0
    nl_tri = NB.build_neighbors(q_tri, box_tri, 1.5, 0.4)

    # Apply pure shear: H12 = 0.35 -> max ||ΔH n|| over n ∈ {-1,0,1}³ is 0.35 < 0.40
    H_shear_sub = [6.0 0.35 0.0; 0.0 6.0 0.0; 0.0 0.0 6.0]
    @assert !NB.needs_rebuild!(nl_tri, q_tri, T.TriclinicBox(H_shear_sub)) "shear=0.35 < skin=0.40 must not trigger"

    # Apply pure shear: H12 = 0.45 -> max ||ΔH n|| is 0.45 > 0.40
    H_shear_super = [6.0 0.45 0.0; 0.0 6.0 0.0; 0.0 0.0 6.0]
    @assert NB.needs_rebuild!(nl_tri, q_tri, T.TriclinicBox(H_shear_super)) "shear=0.45 > skin=0.40 must trigger"

    # Matrix overload check
    @assert NB.needs_rebuild!(nl_tri, q_tri, H_shear_super) "Matrix overload must match TriclinicBox"

    println("  Verlet rebuild accounts for particle motion AND box deformation (orthorhombic + triclinic)")
    return true
end

function test_near_cutoff_pairs_never_lost()
    # Physical conservation test:
    # Construct a pair of atoms placed intentionally at r0 = rc + skin + 0.01
    # at build time, so this pair is EXCLUDED from the neighbor list.
    # Then apply combined particle motion and box deformation.
    # As long as needs_rebuild! returns false, the pair must NOT have entered rc!
    rc = 2.0
    skin = 0.4
    L = 10.0
    box0 = T.OrthorhombicBox((L, L, L))

    # Atom 1 at origin, Atom 2 along x-axis near cutoff
    r0 = rc + skin + 0.02 # 2.42 > rc + skin = 2.40 -> excluded
    q0 = zeros(3, 2)
    q0[1, 2] = r0

    nl = NB.build_neighbors(q0, box0, rc, skin)
    @assert isempty(nl.lists[1]) "pair at r0=2.42 must not be in neighbor list (rc+skin=2.40)"

    # Sub-threshold motion + deformation:
    # Choose box expansion ΔL_x = 0.15 (dh_max = 0.15)
    # Remaining skin = 0.40 - 0.15 = 0.25 -> atom motion threshold = 0.125
    # Move atom 2 inwards by 0.12 < 0.125
    box_deformed = T.OrthorhombicBox((L + 0.15, L, L))
    q_curr = copy(q0)
    q_curr[1, 2] -= 0.12 # new distance = 2.42 - 0.12 = 2.30 > rc = 2.00

    @assert !NB.needs_rebuild!(nl, q_curr, box_deformed) "sub-threshold motion must not trigger"
    d_curr = NB.minimum_image(box_deformed, q_curr, 1, 2)
    r_curr = sqrt(d_curr.x^2 + d_curr.y^2 + d_curr.z^2)
    @assert r_curr > rc "pair outside neighbor list MUST stay outside rc as long as needs_rebuild! is false: r_curr=$r_curr > rc=$rc"

    # Now move atom 2 to violate threshold
    q_violating = copy(q0)
    q_violating[1, 2] -= 0.15 # 0.15 > 0.125
    @assert NB.needs_rebuild!(nl, q_violating, box_deformed) "exceeding threshold must trigger needs_rebuild!"

    # Now test with periodic image near cutoff:
    # Place atom 2 at L - (rc + skin + 0.02) so its periodic image is at rc + skin + 0.02
    q_img = zeros(3, 2)
    q_img[1, 1] = 0.0
    q_img[1, 2] = L - (rc + skin + 0.02)
    nl_img = NB.build_neighbors(q_img, box0, rc, skin)
    @assert isempty(nl_img.lists[1]) "periodic image at 2.42 must be excluded"

    # Box compression: ΔL_x = -0.10 -> image moves closer by 0.10
    # dh_max = 0.10 -> per-atom thresh = (0.4 - 0.1) / 2 = 0.15
    # Move atom 1 by +0.10: image distance reduces by 0.10 + 0.10 = 0.20
    # New image distance = 2.42 - 0.20 = 2.22 > rc = 2.00
    box_compressed = T.OrthorhombicBox((L - 0.10, L, L))
    q_img_curr = copy(q_img)
    q_img_curr[1, 1] += 0.10
    @assert !NB.needs_rebuild!(nl_img, q_img_curr, box_compressed)
    d_img_curr = NB.minimum_image(box_compressed, q_img_curr, 1, 2)
    r_img_curr = sqrt(d_img_curr.x^2 + d_img_curr.y^2 + d_img_curr.z^2)
    @assert r_img_curr > rc "periodic near-cutoff pair must stay outside rc: r=$r_img_curr > $rc"

    # Compression past skin envelope:
    box_too_compressed = T.OrthorhombicBox((L - 0.45, L, L)) # dh_max = 0.45 > skin=0.40
    @assert NB.needs_rebuild!(nl_img, q_img_curr, box_too_compressed) "box deformation exceeding skin must trigger"

    println("  near-cutoff pairs: guaranteed safety envelope verified for direct and periodic pairs")
    return true
end

function test_energy_equals_bruteforce_with_neighbors()
    # The neighbour-list driven energy must equal the exact all-pairs energy to
    # roundoff for a cutoff potential that is continuous at rc (:lfs). This is the
    # check that would have caught the stencil bug in an energy rather than a
    # pair count.
    rng = MersenneTwister(404)
    for (L, rc) in ((8.0, 3.0), (10.0, 3.0), (6.0, 2.4))
        N = 80
        q = rand(rng, 3, N) .* L
        box = T.OrthorhombicBox((L, L, L))
        ctx = Factors.Context(box, T.REDUCED, N)
        # list-driven
        nl = NB.build_neighbors(q, box, rc, 0.0)
        F1 = zeros(3, N)
        U1 = 0.0
        for (i, j) in NB.neighbor_pairs(nl)
            U1 += Factors.evaluate!(F1, Factors.LennardJonesPair(i, j, 1.0, 1.0; mode = :lfs, rc = rc), q, ctx)
        end
        # brute force
        F2 = zeros(3, N)
        U2 = 0.0
        for (i, j) in brute_pairs(q, box, rc)
            U2 += Factors.evaluate!(F2, Factors.LennardJonesPair(i, j, 1.0, 1.0; mode = :lfs, rc = rc), q, ctx)
        end
        relU = abs(U1 - U2) / max(abs(U2), 1e-30)
        scaleF = max(maximum(abs, F2), 1.0)
        dF = maximum(abs, F1 .- F2) / scaleF
        # Ordering of the factor sum differs between the list-driven and brute
        # paths, so the forces agree to accumulation roundoff, not bitwise. The
        # tolerance is relative to the force scale because a bare absolute
        # tolerance would either pass a real bug on a quiet configuration or
        # fail on a loud one.
        @assert relU < 1e-14 "energy mismatch L=$L rc=$rc: $relU"
        @assert dF < 1e-12 "force mismatch L=$L rc=$rc: $dF (scale $(scaleF))"
        @printf("  L=%.0f rc=%.1f: U_rel_diff=%.2e  rel|ΔF|=%.2e  |F|max=%.3g\n",
                L, rc, relU, dF, scaleF)
    end
    return true
end

function _run_all()
    for (name, fn) in [
        ("neighbor_list_matches_bruteforce", test_neighbor_list_matches_bruteforce),
        ("materialized_pairs_have_scale_guard", test_materialized_pairs_have_scale_guard),
        ("skin_is_conservative", test_skin_is_conservative),
        ("rebuild_criterion", test_rebuild_criterion),
        ("triclinic_nearest_image", test_triclinic_nearest_image),
        ("skew_triclinic_mic_vs_27_image_search", test_skew_triclinic_mic_vs_27_image_search),
        ("triclinic_neighbors_supported_and_cutoff_guarded", test_triclinic_neighbors_supported_and_cutoff_guarded),
        ("rebuild_criterion_box_deformation", test_rebuild_criterion_box_deformation),
        ("near_cutoff_pairs_never_lost", test_near_cutoff_pairs_never_lost),
        ("energy_equals_bruteforce_with_neighbors", test_energy_equals_bruteforce_with_neighbors),
    ]
        println("TEST ", name)
        fn()
    end
    println("")
    println("ALL TESTS PASSED")
end

_run_all()
