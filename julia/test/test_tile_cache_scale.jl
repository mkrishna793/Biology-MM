# Phase 2 Scale & Determinism Tests:
#   - Morton (Z-order) 3D space-filling curve spatial locality
#   - 32-atom TilePairCache cluster-pair list vs PairCache and cell list (orthorhombic & triclinic)
#   - Deterministic 64-bit fixed-point (Int64) force accumulation bit-identical across thread counts
#   - 100k+ atom scale test (N = 108,000 > MAX_MATERIALIZED_PAIR_ATOMS = 100,000) with TilePairCache

using FoundationCore
using Random
using LinearAlgebra
using Test

const T   = FoundationCore.Types
const F   = FoundationCore.Factors
const NB  = FoundationCore.Neighbors
const SLJ = FoundationCore.StreamedLJ

@testset "Phase 2: Morton order, TilePairCache, Fixed-Point determinism, and 100k+ scale" begin
    rng = MersenneTwister(20261114)

    # 1. Morton ordering and spatial locality
    @testset "Morton Z-order spatial locality" begin
        N = 1024
        box = T.OrthorhombicBox((20.0, 20.0, 20.0))
        q = zeros(Float64, 3, N)
        for i in 1:N
            q[1, i] = rand(rng) * 20.0
            q[2, i] = rand(rng) * 20.0
            q[3, i] = rand(rng) * 20.0
        end
        codes = NB.morton_codes(q, box)
        @test length(codes) == N
        order = NB.morton_order(q, box)
        @test sort(Int.(order)) == collect(1:N)

        # Consecutive atoms in Morton order should have significantly smaller mean
        # intra-tile bounding-box diameter than random ordering.
        function mean_tile_span(q_mat, perm)
            ntiles = cld(size(q_mat, 2), NB.TILE_SIZE)
            total_span = 0.0
            for t in 1:ntiles
                lo = (t - 1) * NB.TILE_SIZE + 1
                hi = min(t * NB.TILE_SIZE, size(q_mat, 2))
                xmin = ymin = zmin = Inf
                xmax = ymax = zmax = -Inf
                for idx in lo:hi
                    a = perm[idx]
                    xmin = min(xmin, q_mat[1, a]); xmax = max(xmax, q_mat[1, a])
                    ymin = min(ymin, q_mat[2, a]); ymax = max(ymax, q_mat[2, a])
                    zmin = min(zmin, q_mat[3, a]); zmax = max(zmax, q_mat[3, a])
                end
                total_span += sqrt((xmax - xmin)^2 + (ymax - ymin)^2 + (zmax - zmin)^2)
            end
            return total_span / ntiles
        end
        span_random = mean_tile_span(q, collect(1:N))
        span_morton = mean_tile_span(q, order)
        @test span_morton < 0.65 * span_random
        println("  Morton tile mean diameter=$(round(span_morton, digits=3)) vs random=$(round(span_random, digits=3))")
    end

    # 2. TilePairCache exactness across Orthorhombic & Triclinic boxes +PairTopology
    @testset "TilePairCache exact parity (orthorhombic & triclinic)" begin
        N = 160
        eps = 0.08 .+ 0.04 .* rand(rng, N)
        sig = 0.90 .+ 0.20 .* rand(rng, N)
        rcut = 2.6
        skin = 0.4

        # Build a non-trivial PairTopology with 1-2/1-3 exclusions and 1-4 scaled pairs
        exclusions = Tuple{Int,Int}[(i, i + 1) for i in 1:2:(N - 1)]
        scaled = Tuple{Int,Int,Float64}[(i, i + 2, 0.5) for i in 1:3:(N - 2)]
        topo = SLJ.PairTopology(; excluded = exclusions, scaled = scaled)

        obox = T.OrthorhombicBox((9.0, 9.5, 8.8))
        H_tric = [9.2 1.1 0.5;
                  0.0 8.9 0.7;
                  0.0 0.0 9.4]
        tbox = T.TriclinicBox(H_tric)

        for (label, box) in (("orthorhombic", obox), ("triclinic", tbox))
            # Place atoms on a jittered 6x6x5 lattice so minimum separation > 0.8 nm
            q = zeros(Float64, 3, N)
            idx = 0
            for iz in 0:4, iy in 0:5, ix in 0:5
                idx >= N && break
                idx += 1
                s1 = (ix + 0.5 + 0.15 * (rand(rng) - 0.5)) / 6.0
                s2 = (iy + 0.5 + 0.15 * (rand(rng) - 0.5)) / 6.0
                s3 = (iz + 0.5 + 0.15 * (rand(rng) - 0.5)) / 5.0
                if box isa T.OrthorhombicBox
                    q[1, idx] = s1 * box.L[1]
                    q[2, idx] = s2 * box.L[2]
                    q[3, idx] = s3 * box.L[3]
                else
                    q[:, idx] = box.H * [s1, s2, s3]
                end
            end

            ws_cell = SLJ.LJWorkspace(q, box, rcut)
            ws_pair = SLJ.LJWorkspace(q, box, rcut)
            ws_tile = SLJ.LJWorkspace(q, box, rcut)
            pc = NB.PairCache(q, box, rcut, skin)
            tc = NB.TilePairCache(q, box, rcut, skin)

            for mode in (:lfs, :shift, :trunc), topo_arg in (nothing, topo), thr in (false, true)
                U_cell, F_cell, np_cell = SLJ.evaluate!(ws_cell, q, eps, sig, box;
                                                        rcut = rcut, mode = mode, topology = topo_arg,
                                                        threaded = thr)
                U_pair, F_pair, np_pair = SLJ.evaluate!(ws_pair, q, eps, sig, box;
                                                        rcut = rcut, mode = mode, topology = topo_arg,
                                                        threaded = thr, cache = pc)
                U_tile, F_tile, np_tile = SLJ.evaluate!(ws_tile, q, eps, sig, box;
                                                        rcut = rcut, mode = mode, topology = topo_arg,
                                                        threaded = thr, cache = tc)
                @test np_tile == np_cell == np_pair
                @test isapprox(U_tile, U_cell; rtol = 1e-13, atol = 1e-13)
                @test maximum(abs, F_tile .- F_cell) <= 1e-11 * max(1.0, maximum(abs, F_cell))
            end

            # Verify TilePairCache skin reuse and automatic rebuild on large displacement
            diag1 = NB.tile_cache_diagnostics(tc)
            @test diag1.rebuilds == 1
            @test diag1.reuses >= 1
            q_small = q .+ 0.02 .* (rand(rng, 3, N) .- 0.5)
            U_s1, F_s1, np_s1 = SLJ.evaluate!(ws_cell, q_small, eps, sig, box; rcut = rcut, mode = :lfs, topology = topo)
            U_s2, F_s2, np_s2 = SLJ.evaluate!(ws_tile, q_small, eps, sig, box; rcut = rcut, mode = :lfs, topology = topo, cache = tc)
            @test np_s1 == np_s2
            @test isapprox(U_s1, U_s2; rtol = 1e-13, atol = 1e-13)
            @test maximum(abs, F_s1 .- F_s2) <= 1e-11 * max(1.0, maximum(abs, F_s1))
            @test NB.tile_cache_diagnostics(tc).rebuilds == 1

            q_big = q .+ 0.8 .* (rand(rng, 3, N) .- 0.5)
            U_b1, F_b1, np_b1 = SLJ.evaluate!(ws_cell, q_big, eps, sig, box; rcut = rcut, mode = :lfs, topology = topo)
            U_b2, F_b2, np_b2 = SLJ.evaluate!(ws_tile, q_big, eps, sig, box; rcut = rcut, mode = :lfs, topology = topo, cache = tc)
            @test np_b1 == np_b2
            @test isapprox(U_b1, U_b2; rtol = 1e-13, atol = 1e-13)
            @test maximum(abs, F_b1 .- F_b2) <= 1e-11 * max(1.0, maximum(abs, F_b1))
            @test NB.tile_cache_diagnostics(tc).rebuilds == 2

            # Verify StreamedLJFactor with tile_cache = true and fixed_point = true
            f_tile = SLJ.StreamedLJFactor(q, eps, sig, box; rcut = rcut, mode = :lfs,
                                          topology = topo, tile_cache = true, fixed_point = true)
            @test f_tile.pair_cache isa NB.TilePairCache
            ctx = F.Context(box, T.REDUCED, N)
            F_fac = zeros(Float64, 3, N)
            U_fac = F.evaluate!(F_fac, f_tile, q, ctx)
            U_ref, F_ref, _ = SLJ.evaluate!(ws_cell, q, eps, sig, box; rcut = rcut, mode = :lfs, topology = topo)
            @test isapprox(U_fac, U_ref; rtol = 1e-11, atol = 1e-11)
            @test maximum(abs, F_fac .- F_ref) <= 1e-9 * max(1.0, maximum(abs, F_ref))
            G_fac = SLJ.streamed_box_derivative(f_tile, q, ctx)
            G_ref = zeros(Float64, 3, 3)
            SLJ.box_derivative!(G_ref, ws_cell, q, eps, sig, box; rcut = rcut, mode = :lfs, topology = topo)
            @test maximum(abs, G_fac .- G_ref) <= 1e-12 * max(1.0, maximum(abs, G_ref))
            println("  $label TilePairCache + StreamedLJFactor parity verified (n_tiles=$(tc.n_tiles), tile_pairs=$(NB.num_tile_pairs(tc)))")
        end
    end

    # 3. Deterministic 64-bit Fixed-Point force accumulation (bit-identical serial vs threaded)
    @testset "Deterministic 64-bit fixed-point force & energy bit-exactness" begin
        N = 320
        box = T.OrthorhombicBox((12.0, 12.0, 12.0))
        q = zeros(Float64, 3, N)
        idx = 0
        for iz in 0:6, iy in 0:6, ix in 0:6
            idx >= N && break
            idx += 1
            q[1, idx] = (ix + 0.5 + 0.2 * (rand(rng) - 0.5)) * (12.0 / 7.0)
            q[2, idx] = (iy + 0.5 + 0.2 * (rand(rng) - 0.5)) * (12.0 / 7.0)
            q[3, idx] = (iz + 0.5 + 0.2 * (rand(rng) - 0.5)) * (12.0 / 7.0)
        end
        eps = 0.05 .+ 0.05 .* rand(rng, N)
        sig = 0.85 .+ 0.25 .* rand(rng, N)
        rcut = 2.8
        skin = 0.4

        pc = NB.PairCache(q, box, rcut, skin)
        tc = NB.TilePairCache(q, box, rcut, skin)

        for (cname, cache_obj) in (("PairCache", pc), ("TilePairCache", tc))
            ws1 = SLJ.LJWorkspace(q, box, rcut)
            ws2 = SLJ.LJWorkspace(q, box, rcut)
            U1, F1, np1 = SLJ.evaluate!(ws1, q, eps, sig, box;
                                        rcut = rcut, mode = :lfs, threaded = false,
                                        cache = cache_obj, fixed_point = true)
            U2, F2, np2 = SLJ.evaluate!(ws2, q, eps, sig, box;
                                        rcut = rcut, mode = :lfs, threaded = true,
                                        cache = cache_obj, fixed_point = true)
            @test np1 == np2
            # Bitwise exact equality between serial and threaded fixed-point!
            @test U1 === U2
            @test F1 == F2
            # And accurate to ~1e-11 relative to Float64 evaluation
            U_fp64, F_fp64, _ = SLJ.evaluate!(ws1, q, eps, sig, box;
                                              rcut = rcut, mode = :lfs, threaded = false,
                                              cache = cache_obj, fixed_point = false)
            @test isapprox(U1, U_fp64; rtol = 1e-11, atol = 1e-11)
            @test maximum(abs, F1 .- F_fp64) <= 1e-9 * max(1.0, maximum(abs, F_fp64))
            println("  $cname fixed-point bit-exactness verified: U_serial === U_threaded ($(U1 === U2)), max|F_fp - F_f64|=$(maximum(abs, F1 .- F_fp64))")
        end
    end

    # 4. 100k+ atom scale test (N = 108,000 > MAX_MATERIALIZED_PAIR_ATOMS = 100,000)
    @testset "100k+ atom TilePairCache scale & memory footprint (N = 108,000)" begin
        # 30 x 30 x 30 unit cells of 4-atom FCC lattice = 108,000 atoms
        nx = ny = nz = 30
        a = 1.55
        N = 4 * nx * ny * nz
        @test N == 108_000
        @test N > NB.MAX_MATERIALIZED_PAIR_ATOMS

        L = (nx * a, ny * a, nz * a)
        box = T.OrthorhombicBox(L)
        q = Matrix{Float64}(undef, 3, N)
        basis = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5))
        idx = 0
        for iz in 0:(nz - 1), iy in 0:(ny - 1), ix in 0:(nx - 1), (bx, by, bz) in basis
            idx += 1
            q[1, idx] = (ix + bx) * a
            q[2, idx] = (iy + by) * a
            q[3, idx] = (iz + bz) * a
        end
        eps = fill(0.05, N)
        sig = fill(0.95, N)
        rcut = 1.4
        skin = 0.25

        # Legacy materialized neighbor_pairs must refuse N > 100,000, while TilePairCache succeeds!
        nl_108k = NB.NeighborList([Int[] for _ in 1:N], rcut, skin, copy(q), 1, box)
        @test_throws ArgumentError NB.neighbor_pairs(nl_108k)

        tc = NB.TilePairCache(q, box, rcut, skin)
        @test tc.n_tiles == cld(N, NB.TILE_SIZE)
        @test NB.num_tile_pairs(tc) > 0

        # StreamedLJFactor with cache=true automatically selects TilePairCache for N > 100,000
        f_108k = SLJ.StreamedLJFactor(q, eps, sig, box; rcut = rcut, mode = :lfs,
                                      cache = true, cache_skin = skin, threaded = true,
                                      fixed_point = true)
        @test f_108k.pair_cache isa NB.TilePairCache

        ws = SLJ.LJWorkspace(q, box, rcut)
        t0 = time_ns()
        U, F_mat, npairs = SLJ.evaluate!(ws, q, eps, sig, box;
                                         rcut = rcut, mode = :lfs, threaded = true,
                                         cache = tc, fixed_point = true)
        elapsed_ms = (time_ns() - t0) / 1e6
        @test isfinite(U)
        @test npairs == 6 * N # each FCC atom has 12 nearest neighbors at a/sqrt(2) ≈ 1.096 < 1.4
        # Force on every ideal FCC lattice atom must vanish by symmetry!
        @test maximum(abs, F_mat) < 1e-11

        mem_none = SLJ.estimate_memory_bytes(N, box, rcut; cache = :none, threads = Threads.nthreads())
        mem_tile = SLJ.estimate_memory_bytes(N, box, rcut; skin = skin, cache = :tile, threads = Threads.nthreads())
        mem_pair = SLJ.estimate_memory_bytes(N, box, rcut; skin = skin, cache = :pair, threads = Threads.nthreads())
        @test mem_tile > mem_none
        println("  108,000-atom FCC TilePairCache: n_tiles=$(tc.n_tiles), tile_pairs=$(NB.num_tile_pairs(tc)), interacting_pairs=$npairs, U=$(round(U, digits=2)), max|F|=$(maximum(abs, F_mat)), eval_time=$(round(elapsed_ms, digits=1)) ms, est_mem_tile=$(round(mem_tile / 1e6, digits=2)) MB")
    end
end
