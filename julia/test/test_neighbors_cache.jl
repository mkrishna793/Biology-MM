using FoundationCore
using FoundationCore.Types
using FoundationCore.Factors

const NB = FoundationCore.Neighbors
const SL = FoundationCore.StreamedLJ

function cache_fixture(box, q; rcut=2.0, skin=0.5)
    NB.PairCache(Matrix{Float64}(q), box, rcut, skin)
end

function test_cache_reuse_and_deterministic_force_parity()
    box = Types.OrthorhombicBox((12.0, 12.0, 12.0))
    q = [1.0 2.2 8.0 9.2; 1.0 1.0 8.0 8.0; 1.0 1.0 8.0 8.0]
    epsilon = ones(4)
    sigma = ones(4)
    rc, skin = 2.0, 0.5
    cache = cache_fixture(box, q; rcut=rc, skin=skin)
    ws = SL.LJWorkspace(q, box, rc)
    U0, F0raw, n0 = SL.evaluate!(ws, q, epsilon, sigma, box; rcut=rc, cache=cache)
    F0 = copy(F0raw)
    @assert cache.n_rebuilds == 1
    @assert !NB.refresh_pair_cache!(cache, q, box)
    @assert cache.n_rebuilds == 1 && cache.n_reuses == 2
    U1, F1raw, n1 = SL.evaluate!(ws, q, epsilon, sigma, box; rcut=rc, cache=cache)
    F1 = copy(F1raw)
    rebuilt = cache_fixture(box, q; rcut=rc, skin=skin)
    U2, F2raw, n2 = SL.evaluate!(ws, q, epsilon, sigma, box; rcut=rc, cache=rebuilt)
    F2 = copy(F2raw)
    @assert (U0, n0, F0) == (U1, n1, F1) == (U2, n2, F2)
    qmove = copy(q); qmove[1, 1] += 0.03
    Uc, Fc, nc = SL.evaluate!(ws, qmove, epsilon, sigma, box; rcut=rc, cache=cache)
    Ufresh, Ffresh, nfresh = SL.evaluate!(ws, qmove, epsilon, sigma, box; rcut=rc)
    @assert (Uc, nc, Fc) == (Ufresh, nfresh, Ffresh)

    # A pair starts outside rc+skin, then enters rc. The displacement bound
    # must force rebuilding before evaluating that geometry.
    q0 = [1.0 3.6; 1.0 1.0; 1.0 1.0]
    c2 = cache_fixture(box, q0; rcut=rc, skin=skin)
    q1 = copy(q0); q1[1, 2] = 2.9
    @assert NB.pair_cache_stale(c2, q1, box)
    @assert NB.refresh_pair_cache!(c2, q1, box)
    @assert c2.n_rebuilds == 2
    @assert any(k -> c2.pairs_i[k] == 1 && c2.pairs_j[k] == 2,
                eachindex(c2.pairs_i))
    uenter, fenter, nenter = SL.evaluate!(SL.LJWorkspace(q1, box, rc), q1,
                                          ones(2), ones(2), box;
                                          rcut=rc, cache=c2)
    @assert nenter == 1 && uenter != 0.0 && any(!iszero, fenter)

    # A pair already inside the skin may cross rc while both atoms remain
    # below the half-skin displacement threshold; no rebuild is needed.
    qskin0 = [1.0 3.3; 1.0 1.0; 1.0 1.0]
    cskin = cache_fixture(box, qskin0; rcut=rc, skin=skin)
    qskin1 = copy(qskin0); qskin1[1, 1] += 0.2; qskin1[1, 2] -= 0.2
    @assert !NB.refresh_pair_cache!(cskin, qskin1, box)
    @assert cskin.n_rebuilds == 1
    _, fskin, nskin = SL.evaluate!(SL.LJWorkspace(qskin1, box, rc), qskin1,
                                   ones(2), ones(2), box; rcut=rc, cache=cskin)
    @assert nskin == 1 && any(!iszero, fskin)

    # A whole-box crossing is safe (may rebuild conservatively) and preserves
    # the exact minimum-image force result.
    qwrap = copy(q); qwrap[1, 1] += box.L[1]
    U3, F3raw, n3 = SL.evaluate!(ws, qwrap, epsilon, sigma, box; rcut=rc, cache=cache)
    F3 = copy(F3raw)
    ws_fresh = SL.LJWorkspace(qwrap, box, rc)
    Ub, Fbraw, nb = SL.evaluate!(ws_fresh, qwrap, epsilon, sigma, box; rcut=rc)
    Fb = copy(Fbraw)
    @assert n3 == nb && isapprox(U3, Ub; rtol=1e-14, atol=1e-14)
    @assert isapprox(F3, Fb; rtol=1e-13, atol=1e-13)
    println("  PairCache serial parity/reuse: pairs=$(NB.num_pairs(cache)) rebuilds=$(cache.n_rebuilds) reuses=$(cache.n_reuses)")
end

function test_box_changes_rebuild_conservatively()
    q = [0.2 9.7; 0.0 0.0; 0.0 0.0]
    ortho = Types.OrthorhombicBox((10.0, 10.0, 10.0))
    co = cache_fixture(ortho, q)
    smaller = Types.OrthorhombicBox((9.9, 10.0, 10.0))
    @assert !NB.pair_cache_stale(co, q, smaller)
    @assert !NB.refresh_pair_cache!(co, q, smaller)
    q_unwrapped = copy(q); q_unwrapped[1, 2] += 1000.0
    @assert !NB.pair_cache_stale(co, q_unwrapped, smaller)
    @assert !NB.refresh_pair_cache!(co, q_unwrapped, smaller)

    H = [10.0 1.0 0.0; 0.0 10.0 0.0; 0.0 0.0 10.0]
    tri = Types.TriclinicBox(H)
    qt = H * [0.02 0.97; 0.0 0.0; 0.0 0.0]
    ct = cache_fixture(tri, qt)
    H2 = copy(H); H2[1, 2] += 0.01
    @assert !NB.pair_cache_stale(ct, qt, Types.TriclinicBox(H2))
    @assert !NB.refresh_pair_cache!(ct, qt, Types.TriclinicBox(H2))
    H3 = copy(H); H3[1, 1] -= 1.0
    @assert NB.pair_cache_stale(ct, qt, Types.TriclinicBox(H3))
    @assert NB.refresh_pair_cache!(ct, qt, Types.TriclinicBox(H3))
    println("  PairCache small box deformations reuse; unsafe orthorhombic/triclinic bounds rebuild")
end

function test_no_box_motion_bound()
    q0 = [0.0 2.6; 0.0 0.0; 0.0 0.0]
    q1 = copy(q0); q1[1, 2] = 1.9
    c = cache_fixture(Types.NoBox(), q0; rcut=2.0, skin=0.5)
    @assert NB.pair_cache_stale(c, q1, Types.NoBox())
    @assert NB.refresh_pair_cache!(c, q1, Types.NoBox())
    @assert any(k -> c.pairs_i[k] == 1 && c.pairs_j[k] == 2, eachindex(c.pairs_i))
    println("  NoBox PairCache displacement bound catches entering pair")
end

function test_factor_cache_is_opt_in_and_integrated()
    box = Types.OrthorhombicBox((12.0, 12.0, 12.0))
    q = [1.0 2.2 8.0 9.2; 1.0 1.0 8.0 8.0; 1.0 1.0 8.0 8.0]
    eps = ones(4); sig = ones(4)
    ctx = Factors.Context(box, Types.REDUCED, 4)
    plain = SL.StreamedLJFactor(q, eps, sig, box; rcut=2.0)
    cached = SL.StreamedLJFactor(q, eps, sig, box; rcut=2.0,
                                 cache=true, cache_skin=0.5)
    @assert plain.pair_cache === nothing
    @assert cached.pair_cache !== nothing
    Fp = zeros(3, 4); Fc = zeros(3, 4)
    Up = Factors.evaluate!(Fp, plain, q, ctx)
    Uc = Factors.evaluate!(Fc, cached, q, ctx)
    @assert Up == Uc && Fp == Fc
    Fp_snapshot = copy(Fp); Fc_snapshot = copy(Fc)
    @assert cached.pair_cache.n_rebuilds == 1
    qnext = copy(q); qnext[1, 1] += 0.05
    fill!(Fc, 0.0)
    Factors.evaluate!(Fc, cached, qnext, ctx)
    @assert cached.pair_cache.n_reuses == 2

    topo = SL.PairTopology(excluded=[(1, 2)], scaled=[(3, 4, 0.35)])
    plain_topo = SL.StreamedLJFactor(q, eps, sig, box; rcut=2.0,
                                     topology=topo, cache=false)
    cache_topo = SL.StreamedLJFactor(q, eps, sig, box; rcut=2.0,
                                     topology=topo, cache=true, cache_skin=0.5)
    Fp .= 0.0; Fc .= 0.0
    Up = Factors.evaluate!(Fp, plain_topo, q, ctx)
    Uc = Factors.evaluate!(Fc, cache_topo, q, ctx)
    @assert Up == Uc && Fp == Fc "cached topology energy/force parity failed"
    @assert Fp_snapshot == Fc_snapshot
    println("  StreamedLJFactor cache opt-in and Dynamics factor protocol verified")
end

test_cache_reuse_and_deterministic_force_parity()
test_box_changes_rebuild_conservatively()
test_no_box_motion_bound()
test_factor_cache_is_opt_in_and_integrated()
println("ALL CACHE TESTS PASSED")
