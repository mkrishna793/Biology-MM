# Unit and integration test suite for TriumXBT:
# 1. Dual-Rail Balanced Ternary Arithmetic (TritWord64, encode/decode, 0-cycle rail swap negation, BTHA & BTFA truth tables)
# 2. Symplectic Carry-Free Accumulator (carry_free_add, resolve_bounded_ripple, resolve_parallel, 0-ULP permutation invariance)
# 3. Native C++ Bridge Loader & Fast Math Accuracy (xbt_exp_radix3, xbt_rsqrt_fast, xbt_erfc_fast)
# 4. TriumTiledNonbondedFactor Parity vs StreamedLJ + DSFCoulomb (< 1e-10)
# 5. Central Finite Differences (Cartesian Forces < 1e-9, 3-entry orthorhombic & 9-entry triclinic box derivatives < 1e-8)

using Test
using Random
using LinearAlgebra
using FoundationCore

if !isdefined(FoundationCore, :TriumXBT)
    @eval FoundationCore begin
        include(joinpath(@__DIR__, "..", "src", "TriumXBT.jl"))
    end
end
const TXBT = FoundationCore.TriumXBT

using .TXBT
using FoundationCore.Types
using FoundationCore.Factors
using FoundationCore.Neighbors
using FoundationCore.StreamedLJ
using FoundationCore.Electrostatics

@testset "TriumXBT Suite" begin

    # ==========================================================================
    # 1. Dual-Rail Balanced Ternary Arithmetic (TritWord64)
    # ==========================================================================
    @testset "1. TritWord64 & Arithmetic Truth Tables" begin
        # 1.1 Encode/decode roundtrip
        test_integers = Int64[
            0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 13, -13, 42, -42,
            123456789, -987654321, 10^12, -10^12, 10^15, -10^15,
            (3^38 - 1) ÷ 2, -((3^38 - 1) ÷ 2)
        ]
        for v in test_integers
            w = encode_tritword64(v, 40)
            @test (w.P & w.N) == 0
            v_dec = decode_tritword64(w, 40)
            @test v == v_dec
        end

        # 1.2 0-Cycle rail-swap negation
        for v in [1, -1, 42, -42, 1234567, -7654321]
            w = encode_tritword64(v, 40)
            neg_w = -w
            @test neg_w.P == w.N
            @test neg_w.N == w.P
            @test decode_tritword64(neg_w, 40) == -v
        end

        # 1.3 Disjoint rail invariant enforcement
        @test_throws AssertionError TritWord64(UInt64(1), UInt64(1))

        # 1.4 BTHA exhaustive 3x3 truth-table check
        val_trit(w::TritWord64) = Int64(w.P & UInt64(1)) - Int64(w.N & UInt64(1))
        to_trit(v::Int) = TritWord64(v == 1 ? UInt64(1) : UInt64(0), v == -1 ? UInt64(1) : UInt64(0))

        for a in -1:1, b in -1:1
            wa = to_trit(a); wb = to_trit(b)
            s, c = btha(wa, wb)
            @test (s.P & s.N) == 0
            @test (c.P & c.N) == 0
            @test val_trit(s) + 3 * val_trit(c) == a + b
        end

        # 1.5 BTFA exhaustive 3x3x3 truth-table check
        for a in -1:1, b in -1:1, d in -1:1
            wa = to_trit(a); wb = to_trit(b); wd = to_trit(d)
            s, c = btfa(wa, wb, wd)
            @test (s.P & s.N) == 0
            @test (c.P & c.N) == 0
            @test val_trit(s) + 3 * val_trit(c) == a + b + d
        end
    end

    # ==========================================================================
    # 2. Carry-Free Accumulation, Resolvers & 0-ULP Permutation Invariance
    # ==========================================================================
    @testset "2. Carry-Free Accumulator & Permutation Invariance" begin
        # 2.1 1,000 random terms vs exact Int64 sum (ripple & parallel)
        Random.seed!(42)
        S = TritWord64(UInt64(0), UInt64(0))
        C = TritWord64(UInt64(0), UInt64(0))
        exact_sum = Int64(0)
        for _ in 1:1000
            term = rand(-1_000_000:1_000_000)
            exact_sum += term
            X = encode_tritword64(term, 40)
            S, C = carry_free_add(S, C, X)
        end

        r_rip, ovf_rip = resolve_bounded_ripple(S, C, 40)
        r_par, ovf_par = resolve_parallel(S, C, 40)

        @test !ovf_rip
        @test !ovf_par
        @test r_rip == r_par
        @test decode_tritword64(r_rip, 40) == exact_sum
        @test decode_tritword64(r_par, 40) == exact_sum

        # 2.2 CarryFreeForceBuffer 0-ULP Permutation Invariance
        n_atoms = 12
        buf1 = CarryFreeForceBuffer(n_atoms; width = 40)
        buf2 = CarryFreeForceBuffer(n_atoms; width = 40)

        # Generate 100 random pair forces
        pairs_data = [
            (rand(1:n_atoms), rand(1:n_atoms), randn() * 50.0, randn() * 50.0, randn() * 50.0)
            for _ in 1:100
        ]
        # Ensure i != j
        pairs_clean = filter(p -> p[1] != p[2], pairs_data)

        # Scatter in order 1
        reset!(buf1)
        for (i, j, fx, fy, fz) in pairs_clean
            scatter_pair_force!(buf1, i, j, fx, fy, fz)
        end
        F1 = zeros(Float64, 3, n_atoms)
        resolve_forces!(F1, buf1)

        # Scatter in randomly permuted order
        p_order = randperm(length(pairs_clean))
        reset!(buf2)
        for idx in p_order
            i, j, fx, fy, fz = pairs_clean[idx]
            scatter_pair_force!(buf2, i, j, fx, fy, fz)
        end
        F2 = zeros(Float64, 3, n_atoms)
        resolve_forces!(F2, buf2)

        # Strict 0-ULP bitwise equality
        @test F1 == F2
        @test maximum(abs, F1 - F2) == 0.0
    end

    # ==========================================================================
    # 3. Native Bridge Detection & Fast Math Accuracy
    # ==========================================================================
    @testset "3. Native Bridge Detection & Fast Math Accuracy" begin
        # Detection
        bridge_active = has_native_bridge()
        @test bridge_active isa Bool

        # Radix-3 exponential vs standard exp
        for x in [-4.5, -2.0, -0.5, 0.0, 0.1, 1.5, 3.2, 5.0]
            val_xbt = xbt_exp_radix3(x)
            val_std = exp(x)
            rel_err = abs(val_xbt - val_std) / val_std
            @test rel_err < 1e-10
        end

        # Cubic Householder rsqrt vs 1 / sqrt(x)
        for x in [0.05, 0.5, 1.0, 2.7, 10.0, 36.0, 100.0]
            val_rsqrt = xbt_rsqrt_fast(x)
            val_exact = 1.0 / sqrt(x)
            rel_err = abs(val_rsqrt - val_exact) / val_exact
            @test rel_err < 1e-14
        end

        # High-accuracy erfc (Abramowitz & Stegun A&S 7.1.26 absolute error < 1.5e-7)
        for x in [0.0, 0.1, 0.5, 1.0, 2.0, 3.0]
            val_erfc = xbt_erfc_fast(x)
            val_exact = ccall(:erfc, Cdouble, (Cdouble,), x)
            abs_err = abs(val_erfc - val_exact)
            @test abs_err < 2e-7
        end

        # CSA array accumulator
        words = [encode_tritword64(rand(-100_000:100_000), 40) for _ in 1:30]
        S_arr, C_arr = xbt_csa_accumulate_array(words)
        res_arr, ovf_arr = resolve_bounded_ripple(S_arr, C_arr, 40)
        @test !ovf_arr
        exact_arr_sum = sum(decode_tritword64(w, 40) for w in words)
        @test decode_tritword64(res_arr, 40) == exact_arr_sum
    end

    # ==========================================================================
    # 4. TriumTiledNonbondedFactor Parity vs StreamedLJ + DSFCoulomb (< 1e-10)
    # ==========================================================================
    @testset "4. Parity against StreamedLJ + DSFCoulomb" begin
        N = 6
        L = (4.0, 4.0, 4.0)
        box = OrthorhombicBox(L)
        q = [0.5 1.2 2.1 0.8 1.9 3.1;
             0.6 1.8 0.4 2.5 3.0 1.1;
             0.7 0.9 2.8 1.4 0.2 2.3]
        sigma = [0.30, 0.32, 0.28, 0.31, 0.29, 0.33]
        epsilon = [0.50, 0.60, 0.45, 0.55, 0.52, 0.48]
        charges = [0.40, -0.40, 0.30, -0.50, 0.20, -0.10]
        rcut = 1.8
        alpha = 0.22
        pref = 138.935456

        # Exclusions and 1-4 scaling
        topo = StreamedLJ.PairTopology(
            excluded = [(1, 2), (3, 4)],
            scaled = [(1, 3, 0.5), (2, 4, 0.5)]
        )
        corrections = Electrostatics.PairCorrections(
            excluded = [(1, 2), (3, 4)],
            scaled = [(1, 3, 0.8333333333333334), (2, 4, 0.8333333333333334)]
        )

        ctx = Factors.Context(box, Factors.REDUCED, N)

        # 4.1 Independent Reference: StreamedLJFactor + DSFCoulombFactor
        lj_fac = StreamedLJ.StreamedLJFactor(N, epsilon, sigma, box;
                                            rcut = rcut, mode = :lfs, topology = topo)
        F_lj = zeros(Float64, 3, N)
        u_lj = Factors.evaluate!(F_lj, lj_fac, q, ctx)
        G_lj = zeros(Float64, 3, 3)
        StreamedLJ.streamed_box_derivative!(G_lj, lj_fac, q, ctx)

        dsf_fac = Electrostatics.DSFCoulombFactor(charges;
                                                alpha = alpha, rcut = rcut,
                                                prefactor = pref, corrections = corrections,
                                                include_self = false)
        F_dsf = zeros(Float64, 3, N)
        u_dsf = Factors.evaluate!(F_dsf, dsf_fac, q, ctx)
        G_dsf = zeros(Float64, 3, 3)
        Electrostatics.dsf_box_derivative!(G_dsf, dsf_fac, q, box)

        u_ref = u_lj + u_dsf
        F_ref = F_lj + F_dsf
        G_ref = G_lj + G_dsf

        # 4.2 TriumTiledNonbondedFactor (Direct All-Pairs)
        txbt_fac = TriumTiledNonbondedFactor(
            sigma, epsilon, charges;
            rcut = rcut, alpha = alpha, coulomb_prefactor = pref,
            lj_mode = :lfs, lj14_scale = 0.5, coul14_scale = 0.8333333333333334,
            topology = topo, use_fast_math = false, use_carry_free = false
        )

        F_txbt = zeros(Float64, 3, N)
        G_txbt = zeros(Float64, 3, 3)
        u_txbt = eval_lj_dsf_pairs!(F_txbt, G_txbt, q, box, txbt_fac)

        @test abs(u_txbt - u_ref) < 1e-10
        @test maximum(abs, F_txbt - F_ref) < 1e-10
        @test maximum(abs, G_txbt - G_ref) < 1e-10

        # Factors.evaluate! protocol check
        F_proto = zeros(Float64, 3, N)
        u_proto = Factors.evaluate!(F_proto, txbt_fac, q, ctx)
        @test abs(u_proto - u_ref) < 1e-10
        @test maximum(abs, F_proto - F_ref) < 1e-10

        # 4.3 Cached Execution with TilePairCache and PairCache
        tile_cache = Neighbors.TilePairCache(q, box, rcut, 0.1)
        txbt_tiled = TriumTiledNonbondedFactor(
            sigma, epsilon, charges;
            rcut = rcut, alpha = alpha, coulomb_prefactor = pref,
            lj_mode = :lfs, lj14_scale = 0.5, coul14_scale = 0.8333333333333334,
            topology = topo, cache = tile_cache, use_fast_math = false, use_carry_free = false
        )
        F_tiled = zeros(Float64, 3, N)
        G_tiled = zeros(Float64, 3, 3)
        u_tiled = eval_lj_dsf_pairs!(F_tiled, G_tiled, q, box, txbt_tiled)

        @test abs(u_tiled - u_ref) < 1e-10
        @test maximum(abs, F_tiled - F_ref) < 1e-10
        @test maximum(abs, G_tiled - G_ref) < 1e-10

        pair_cache = Neighbors.PairCache(q, box, rcut, 0.1)
        txbt_paircached = TriumTiledNonbondedFactor(
            sigma, epsilon, charges;
            rcut = rcut, alpha = alpha, coulomb_prefactor = pref,
            lj_mode = :lfs, lj14_scale = 0.5, coul14_scale = 0.8333333333333334,
            topology = topo, cache = pair_cache, use_fast_math = false, use_carry_free = false
        )
        F_pc = zeros(Float64, 3, N)
        G_pc = zeros(Float64, 3, 3)
        u_pc = eval_lj_dsf_pairs!(F_pc, G_pc, q, box, txbt_paircached)

        @test abs(u_pc - u_ref) < 1e-10
        @test maximum(abs, F_pc - F_ref) < 1e-10
        @test maximum(abs, G_pc - G_ref) < 1e-10

        # 4.4 CarryFreeForceBuffer Mode
        txbt_cf = TriumTiledNonbondedFactor(
            sigma, epsilon, charges;
            rcut = rcut, alpha = alpha, coulomb_prefactor = pref,
            lj_mode = :lfs, lj14_scale = 0.5, coul14_scale = 0.8333333333333334,
            topology = topo, use_fast_math = false, use_carry_free = true
        )
        F_cf = zeros(Float64, 3, N)
        u_cf = eval_lj_dsf_pairs!(F_cf, nothing, q, box, txbt_cf)
        @test abs(u_cf - u_ref) < 1e-10
        # Quantization bound with scale 2^28
        @test maximum(abs, F_cf - F_ref) < 1e-6

        # 4.5 energy_and_forces! and Box Derivative API Signatures
        sys = Types.System(Types.AtomSet(q, zeros(3, N), fill(1.0, N), charges), box)
        F_sys = zeros(Float64, 3, N)
        G_sys = zeros(Float64, 3, 3)
        u_sys = energy_and_forces!(F_sys, G_sys, sys, txbt_fac)
        @test abs(u_sys - u_ref) < 1e-10
        @test maximum(abs, F_sys - F_ref) < 1e-10
        @test maximum(abs, G_sys - G_ref) < 1e-10

        G_acc = zeros(Float64, 3, 3)
        accumulate_box_derivative!(G_acc, txbt_fac, q, box)
        @test maximum(abs, G_acc - G_ref) < 1e-10

        G_acc2 = accumulate_box_derivative(txbt_fac, q, box)
        @test maximum(abs, G_acc2 - G_ref) < 1e-10

        G_ctx = zeros(Float64, 3, 3)
        Factors.accumulate_box_derivative!(G_ctx, txbt_fac, q, ctx)
        @test maximum(abs, G_ctx - G_ref) < 1e-10

        # Factor contract checks
        @test Factors.factor_support_size(txbt_fac) == N
        @test Factors.is_local(txbt_fac) == false
        @test Factors.force_consistent(txbt_fac) == true

        txbt_trunc = TriumTiledNonbondedFactor(
            sigma, epsilon, charges;
            rcut = rcut, lj_mode = :trunc
        )
        @test Factors.force_consistent(txbt_trunc) == false

        # Guards and refusals
        @test_throws ArgumentError accumulate_box_derivative!(zeros(3, 3), txbt_fac, q, Types.NoBox())
        @test_throws ArgumentError eval_lj_dsf_pairs!(zeros(3, N), nothing, q, OrthorhombicBox((3.0, 3.0, 3.0)), txbt_fac) # 2*rcut = 3.6 > 3.0
    end

    # ==========================================================================
    # 5. Central Finite Differences (Forces < 1e-9, Box Derivatives < 1e-8)
    # ==========================================================================
    @testset "5. Finite Difference Gradient Verification" begin
        N = 5
        charges = [0.35, -0.45, 0.25, -0.30, 0.15]
        sigma = [0.30, 0.32, 0.28, 0.31, 0.29]
        epsilon = [0.50, 0.60, 0.45, 0.55, 0.52]
        rcut = 1.7
        alpha = 0.25
        pref = 1.0

        txbt_lfs = TriumTiledNonbondedFactor(
            sigma, epsilon, charges;
            rcut = rcut, alpha = alpha, coulomb_prefactor = pref,
            lj_mode = :lfs, use_fast_math = false, use_carry_free = false
        )

        # 5.1 Central FD Cartesian Forces (F = -∂U/∂q)
        L = (4.0, 4.2, 4.5)
        box_ortho = OrthorhombicBox(L)
        q0 = [0.6 1.4 2.2 0.9 2.0;
              0.7 1.9 0.5 2.6 3.1;
              0.8 1.0 2.9 1.5 0.3]

        F_an = zeros(Float64, 3, N)
        u0 = eval_lj_dsf_pairs!(F_an, nothing, q0, box_ortho, txbt_lfs)

        h_fd = 1e-6
        max_f_err = 0.0
        for i in 1:N, coord in 1:3
            qp = copy(q0); qp[coord, i] += h_fd
            qm = copy(q0); qm[coord, i] -= h_fd
            up = eval_lj_dsf_pairs!(zeros(3, N), nothing, qp, box_ortho, txbt_lfs)
            um = eval_lj_dsf_pairs!(zeros(3, N), nothing, qm, box_ortho, txbt_lfs)
            f_fd = (um - up) / (2.0 * h_fd) # -∂U/∂q
            err = abs(F_an[coord, i] - f_fd)
            max_f_err = max(max_f_err, err)
        end
        @test max_f_err < 1e-9

        # 5.2 Central FD 3-entry Orthorhombic Box Derivatives
        G_ortho_vec = zeros(Float64, 3)
        eval_lj_dsf_pairs!(zeros(3, N), G_ortho_vec, q0, box_ortho, txbt_lfs)

        S_frac_ortho = [q0[1, :] ./ L[1], q0[2, :] ./ L[2], q0[3, :] ./ L[3]]
        max_g_ortho_err = 0.0
        for d in 1:3
            L_p = collect(L); L_p[d] += h_fd
            L_m = collect(L); L_m[d] -= h_fd
            box_p = OrthorhombicBox((L_p[1], L_p[2], L_p[3]))
            box_m = OrthorhombicBox((L_m[1], L_m[2], L_m[3]))
            qp = [S_frac_ortho[1] .* L_p[1] S_frac_ortho[2] .* L_p[2] S_frac_ortho[3] .* L_p[3]]'
            qm = [S_frac_ortho[1] .* L_m[1] S_frac_ortho[2] .* L_m[2] S_frac_ortho[3] .* L_m[3]]'
            up = eval_lj_dsf_pairs!(zeros(3, N), nothing, Matrix(qp), box_p, txbt_lfs)
            um = eval_lj_dsf_pairs!(zeros(3, N), nothing, Matrix(qm), box_m, txbt_lfs)
            g_fd = (up - um) / (2.0 * h_fd)
            err = abs(G_ortho_vec[d] - g_fd)
            max_g_ortho_err = max(max_g_ortho_err, err)
        end
        @test max_g_ortho_err < 1e-8

        # 5.3 Central FD 9-entry Triclinic Box Derivatives
        H0 = [4.0 0.2 0.1;
              0.0 4.2 0.3;
              0.0 0.0 4.1]
        box_tri = TriclinicBox(H0)
        S_frac_tri = [0.15 0.55 0.25 0.75 0.40;
                      0.20 0.60 0.30 0.80 0.35;
                      0.10 0.50 0.40 0.70 0.60]
        q_tri = H0 * S_frac_tri

        G_tri_an = zeros(Float64, 3, 3)
        eval_lj_dsf_pairs!(zeros(3, N), G_tri_an, q_tri, box_tri, txbt_lfs)

        max_g_tri_err = 0.0
        for a in 1:3, b in 1:3
            Eab = zeros(3, 3); Eab[a, b] = h_fd
            Hp = H0 + Eab; Hm = H0 - Eab
            box_p = TriclinicBox(Hp); box_m = TriclinicBox(Hm)
            qp = Hp * S_frac_tri; qm = Hm * S_frac_tri
            up = eval_lj_dsf_pairs!(zeros(3, N), nothing, qp, box_p, txbt_lfs)
            um = eval_lj_dsf_pairs!(zeros(3, N), nothing, qm, box_m, txbt_lfs)
            g_fd = (up - um) / (2.0 * h_fd)
            err = abs(G_tri_an[a, b] - g_fd)
            max_g_tri_err = max(max_g_tri_err, err)
        end
        @test max_g_tri_err < 1e-8
    end

    # ==========================================================================
    # 6. Difference-Set Disjoint Tile Orbit Scheduler
    # ==========================================================================
    @testset "6. Difference-Set Disjoint Tile Orbit Scheduler" begin
        # 6.1 Empty graph
        sched_empty = schedule_disjoint_tile_orbits(5, Int32[], Int32[])
        @test sched_empty.n_tiles == 5
        @test sched_empty.n_orbits == 0
        @test sched_empty.orbit_offsets == Int32[1]
        @test isempty(sched_empty.orbit_perm)
        @test verify_disjoint_orbits(sched_empty, Int32[], Int32[])

        # 6.2 Synthetic dense tile graph (all pairs 1 <= ti <= tj <= 10)
        ti_syn = Int32[]
        tj_syn = Int32[]
        for i in 1:10, j in i:10
            push!(ti_syn, Int32(i))
            push!(tj_syn, Int32(j))
        end
        sched_syn = schedule_disjoint_tile_orbits(10, ti_syn, tj_syn)
        @test sched_syn.n_orbits > 0
        @test length(sched_syn.orbit_perm) == length(ti_syn)
        @test sched_syn.orbit_offsets[1] == Int32(1)
        @test sched_syn.orbit_offsets[end] == Int32(length(ti_syn) + 1)
        @test verify_disjoint_orbits(sched_syn, ti_syn, tj_syn)

        # Pure-Julia fallback scheduler check
        orb_jl = Vector{Int32}(undef, length(ti_syn))
        n_orb_jl = TXBT._schedule_disjoint_tile_orbits_julia!(orb_jl, 10, ti_syn, tj_syn)
        @test n_orb_jl == sched_syn.n_orbits
        @test orb_jl == sched_syn.orbit_ids

        # Negative checks for verify_disjoint_orbits
        bad_ids = copy(sched_syn.orbit_ids)
        bad_ids .= Int32(1)
        sched_bad = DifferenceSetTileSchedule(10, sched_syn.n_orbits, bad_ids,
                                              sched_syn.orbit_offsets, sched_syn.orbit_perm)
        @test !verify_disjoint_orbits(sched_bad, ti_syn, tj_syn)

        # 6.3 Morton TilePairCache graph (N = 640 -> 20 tiles, > 128 tile pairs)
        rng_m = MersenneTwister(1234)
        N_m = 640
        L_m = (7.0, 7.0, 7.0)
        box_m = OrthorhombicBox(L_m)
        q_m = rand(rng_m, 3, N_m) .* 7.0
        tc_m = Neighbors.TilePairCache(q_m, box_m, 2.8, 0.2)
        @test length(tc_m.tile_pairs_i) >= 128
        sched_m = schedule_disjoint_tile_orbits(tc_m.n_tiles, tc_m.tile_pairs_i, tc_m.tile_pairs_j)
        @test verify_disjoint_orbits(sched_m, tc_m.tile_pairs_i, tc_m.tile_pairs_j)
    end

    # ==========================================================================
    # 7. Native C++ TilePairCache & PairCache MT Parity, Accumulation & Determinism
    # ==========================================================================
    @testset "7. Native C++ TilePairCache & PairCache MT Parity" begin
        @test init_bridge!()
        @test has_native_bridge()
        @test TXBT._SYM_EVAL_PAIRS_MT[] != C_NULL
        @test TXBT._SYM_EVAL_TILES_MT[] != C_NULL
        @test TXBT._SYM_SCHED_ORBITS[] != C_NULL
        @test TXBT._SYM_MATVEC_MULTIPLANE[] != C_NULL

        rng = MersenneTwister(202611)
        N = 640
        rcut = 2.8
        alpha = 0.25
        pref = 138.935456

        # Place particles on a jittered grid so minimum inter-particle distance > 0.35 (well above r^2 >= 1e-24 & finite LJ)
        grid_pts = Matrix{Float64}(undef, 3, N)
        idx = 1
        for ix in 0:8, iy in 0:8, iz in 0:7
            idx > N && break
            grid_pts[1, idx] = (ix + 0.2 + 0.5 * rand(rng)) * (7.2 / 9.0)
            grid_pts[2, idx] = (iy + 0.2 + 0.5 * rand(rng)) * (7.2 / 9.0)
            grid_pts[3, idx] = (iz + 0.2 + 0.5 * rand(rng)) * (7.2 / 8.0)
            idx += 1
        end

        sigma = 0.28 .+ 0.04 .* rand(rng, N)
        epsilon = 0.40 .+ 0.20 .* rand(rng, N)
        charges = 0.4 .* (rand(rng, N) .- 0.5)
        charges .-= sum(charges) / N

        # Bonded exclusions and 1-4 scaled pairs
        excl_pairs = [(i, i + 1) for i in 1:2:(N - 1)]
        scaled_pairs = [(i, i + 2, 0.5) for i in 1:4:(N - 2)]
        topo = StreamedLJ.PairTopology(excluded = excl_pairs, scaled = scaled_pairs)

        boxes = [
            ("OrthorhombicBox", OrthorhombicBox((7.2, 7.2, 7.2)), grid_pts),
            ("TriclinicBox", TriclinicBox([7.4 0.35 0.15; 0.0 7.3 0.25; 0.0 0.0 7.2]), grid_pts),
        ]

        for (label, box, q_pos) in boxes
            @testset "$label" begin
                tc_tile = Neighbors.TilePairCache(q_pos, box, rcut, 0.2)
                tc_pair = Neighbors.PairCache(q_pos, box, rcut, 0.2)
                @test length(tc_tile.tile_pairs_i) >= 128
                @test length(tc_pair.pairs_i) >= 8192

                fac_tile_cpp = TriumTiledNonbondedFactor(
                    sigma, epsilon, charges;
                    rcut = rcut, alpha = alpha, coulomb_prefactor = pref,
                    lj_mode = :lfs, lj14_scale = 0.5, coul14_scale = 0.8333333333333334,
                    topology = topo, cache = tc_tile,
                    use_native_bridge = true, use_fast_math = false, include_self = true
                )
                fac_tile_jl = TriumTiledNonbondedFactor(
                    sigma, epsilon, charges;
                    rcut = rcut, alpha = alpha, coulomb_prefactor = pref,
                    lj_mode = :lfs, lj14_scale = 0.5, coul14_scale = 0.8333333333333334,
                    topology = topo, cache = tc_tile,
                    use_native_bridge = false, use_fast_math = false, include_self = true
                )
                fac_pair_cpp = TriumTiledNonbondedFactor(
                    sigma, epsilon, charges;
                    rcut = rcut, alpha = alpha, coulomb_prefactor = pref,
                    lj_mode = :lfs, lj14_scale = 0.5, coul14_scale = 0.8333333333333334,
                    topology = topo, cache = tc_pair,
                    use_native_bridge = true, use_fast_math = false, include_self = true
                )
                fac_pair_jl = TriumTiledNonbondedFactor(
                    sigma, epsilon, charges;
                    rcut = rcut, alpha = alpha, coulomb_prefactor = pref,
                    lj_mode = :lfs, lj14_scale = 0.5, coul14_scale = 0.8333333333333334,
                    topology = topo, cache = tc_pair,
                    use_native_bridge = false, use_fast_math = false, include_self = true
                )

                F_t_cpp = zeros(Float64, 3, N); G_t_cpp = zeros(Float64, 3, 3)
                F_t_jl  = zeros(Float64, 3, N); G_t_jl  = zeros(Float64, 3, 3)
                F_p_cpp = zeros(Float64, 3, N); G_p_cpp = zeros(Float64, 3, 3)
                F_p_jl  = zeros(Float64, 3, N); G_p_jl  = zeros(Float64, 3, 3)

                u_t_cpp = eval_lj_dsf_pairs!(F_t_cpp, G_t_cpp, q_pos, box, fac_tile_cpp)
                u_t_jl  = eval_lj_dsf_pairs!(F_t_jl,  G_t_jl,  q_pos, box, fac_tile_jl)
                u_p_cpp = eval_lj_dsf_pairs!(F_p_cpp, G_p_cpp, q_pos, box, fac_pair_cpp)
                u_p_jl  = eval_lj_dsf_pairs!(F_p_jl,  G_p_jl,  q_pos, box, fac_pair_jl)

                @test abs(u_t_cpp - u_t_jl) < 1e-10
                @test maximum(abs, F_t_cpp - F_t_jl) < 1e-10
                @test maximum(abs, G_t_cpp - G_t_jl) < 1e-8

                @test abs(u_p_cpp - u_p_jl) < 1e-10
                @test maximum(abs, F_p_cpp - F_p_jl) < 1e-10
                @test maximum(abs, G_p_cpp - G_p_jl) < 1e-8

                @test abs(u_t_cpp - u_p_cpp) < 1e-10
                @test maximum(abs, F_t_cpp - F_p_cpp) < 1e-10
                @test maximum(abs, G_t_cpp - G_p_cpp) < 1e-8

                if box isa OrthorhombicBox
                    Gv_t_cpp = zeros(Float64, 3)
                    Gv_t_jl = zeros(Float64, 3)
                    eval_lj_dsf_pairs!(zeros(Float64, 3, N), Gv_t_cpp, q_pos, box, fac_tile_cpp)
                    eval_lj_dsf_pairs!(zeros(Float64, 3, N), Gv_t_jl,  q_pos, box, fac_tile_jl)
                    @test maximum(abs, Gv_t_cpp - Gv_t_jl) < 1e-8
                    @test maximum(abs, Gv_t_cpp - diag(G_t_cpp)) < 1e-12
                end

                # Non-zero initial F & box_deriv accumulation parity
                F_init = randn(rng, 3, N)
                G_init = randn(rng, 3, 3)
                F_acc_cpp = copy(F_init); G_acc_cpp = copy(G_init)
                F_acc_jl  = copy(F_init); G_acc_jl  = copy(G_init)
                u_acc_cpp = eval_lj_dsf_pairs!(F_acc_cpp, G_acc_cpp, q_pos, box, fac_tile_cpp)
                u_acc_jl  = eval_lj_dsf_pairs!(F_acc_jl,  G_acc_jl,  q_pos, box, fac_tile_jl)
                @test abs(u_acc_cpp - u_acc_jl) < 1e-10
                @test maximum(abs, F_acc_cpp - F_acc_jl) < 1e-10
                @test maximum(abs, G_acc_cpp - G_acc_jl) < 1e-8
                @test maximum(abs, (F_acc_cpp - F_init) - F_t_cpp) < 1e-11

                F_p_acc = copy(F_init); G_p_acc = copy(G_init)
                u_p_acc = eval_lj_dsf_pairs!(F_p_acc, G_p_acc, q_pos, box, fac_pair_cpp)
                @test abs(u_p_acc - u_p_cpp) < 1e-10
                @test maximum(abs, (F_p_acc - F_init) - F_p_cpp) < 1e-11
                @test maximum(abs, (G_p_acc - G_init) - G_p_cpp) < 1e-11

                # Strict determinism check (eval 1 === eval 2)
                F_det1 = zeros(Float64, 3, N); G_det1 = zeros(Float64, 3, 3)
                F_det2 = zeros(Float64, 3, N); G_det2 = zeros(Float64, 3, 3)
                u_det1 = eval_lj_dsf_pairs!(F_det1, G_det1, q_pos, box, fac_tile_cpp)
                u_det2 = eval_lj_dsf_pairs!(F_det2, G_det2, q_pos, box, fac_tile_cpp)
                @test u_det1 === u_det2
                @test all(F_det1 .=== F_det2)
                @test all(G_det1 .=== G_det2)

                Fp_det1 = zeros(Float64, 3, N); Gp_det1 = zeros(Float64, 3, 3)
                Fp_det2 = zeros(Float64, 3, N); Gp_det2 = zeros(Float64, 3, 3)
                up_det1 = eval_lj_dsf_pairs!(Fp_det1, Gp_det1, q_pos, box, fac_pair_cpp)
                up_det2 = eval_lj_dsf_pairs!(Fp_det2, Gp_det2, q_pos, box, fac_pair_cpp)
                @test up_det1 === up_det2
                @test all(Fp_det1 .=== Fp_det2)
                @test all(Gp_det1 .=== Gp_det2)
            end
        end
    end

    # ==========================================================================
    # 8. Compiled CSR Verlet Plan Caching, Zero-Copy Direct Eval & Multi-Rate
    # ==========================================================================
    @testset "8. Compiled CSR Verlet Plan, Zero-Copy Eval & Multi-Rate" begin
        # 8.1 TriumMultiRateConfig & effective_ns_per_day unit math
        cfg_def = TriumMultiRateConfig()
        @test cfg_def.dt_fs == 2.0
        @test cfg_def.hmr_factor == 1.0
        @test cfg_def.respa_outer_stride == 1
        @test effective_ns_per_day(1.0) ≈ 172.8
        @test effective_ns_per_day(1.0, cfg_def) ≈ 172.8

        cfg_mr = TriumMultiRateConfig(dt_fs = 2.5, hmr_factor = 2.0, respa_outer_stride = 2)
        @test effective_ns_per_day(0.5, cfg_mr) ≈ (86.4 * 2.5 * 2.0 * 2) / 0.5
        @test effective_ns_per_day(2.0; dt_fs = 2.0, hmr_factor = 2.0, respa_outer_stride = 3) ≈
              (86.4 * 2.0 * 2.0 * 3) / 2.0
        @test_throws ArgumentError effective_ns_per_day(0.0)
        @test_throws ArgumentError TriumMultiRateConfig(dt_fs = 2.0, respa_outer_stride = 0)

        # 8.2 Direct-into evaluation, fast-math NEON accuracy, and Verlet cache rebuild
        clear_verlet_plan_cache!()
        rng = MersenneTwister(202618)
        N = 320
        rcut = 2.6
        skin = 0.25
        alpha = 0.25
        pref = 138.935456

        box_ortho = OrthorhombicBox((6.8, 6.8, 6.8))
        q_pos = Matrix{Float64}(undef, 3, N)
        idx = 1
        for ix in 0:6, iy in 0:6, iz in 0:6
            idx > N && break
            q_pos[1, idx] = (ix + 0.25 + 0.4 * rand(rng)) * (6.8 / 7.0)
            q_pos[2, idx] = (iy + 0.25 + 0.4 * rand(rng)) * (6.8 / 7.0)
            q_pos[3, idx] = (iz + 0.25 + 0.4 * rand(rng)) * (6.8 / 7.0)
            idx += 1
        end

        sigma = 0.28 .+ 0.04 .* rand(rng, N)
        epsilon = 0.40 .+ 0.20 .* rand(rng, N)
        charges = 0.4 .* (rand(rng, N) .- 0.5)
        charges .-= sum(charges) / N
        topo = StreamedLJ.PairTopology(
            excluded = [(i, i + 1) for i in 1:2:(N - 1)],
            scaled = [(i, i + 2, 0.5) for i in 1:4:(N - 2)]
        )

        tc = Neighbors.TilePairCache(q_pos, box_ortho, rcut, skin)
        fac_exact = TriumTiledNonbondedFactor(
            sigma, epsilon, charges;
            rcut = rcut, alpha = alpha, coulomb_prefactor = pref,
            lj_mode = :lfs, lj14_scale = 0.5, coul14_scale = 0.8333333333333334,
            topology = topo, cache = tc,
            use_native_bridge = true, use_fast_math = false, include_self = true
        )
        fac_fast = TriumTiledNonbondedFactor(
            sigma, epsilon, charges;
            rcut = rcut, alpha = alpha, coulomb_prefactor = pref,
            lj_mode = :lfs, lj14_scale = 0.5, coul14_scale = 0.8333333333333334,
            topology = topo, cache = tc,
            use_native_bridge = true, use_fast_math = true, include_self = true
        )
        fac_jl = TriumTiledNonbondedFactor(
            sigma, epsilon, charges;
            rcut = rcut, alpha = alpha, coulomb_prefactor = pref,
            lj_mode = :lfs, lj14_scale = 0.5, coul14_scale = 0.8333333333333334,
            topology = topo, cache = tc,
            use_native_bridge = false, use_fast_math = false, include_self = true
        )

        F_jl = zeros(Float64, 3, N); G_jl = zeros(Float64, 3, 3)
        u_jl = eval_lj_dsf_pairs!(F_jl, G_jl, q_pos, box_ortho, fac_jl)

        # Pre-fill with sentinel values to verify direct_into! overwrites F & G
        F_dir = fill(999.0, 3, N); G_dir = fill(999.0, 3, 3)
        u_dir = eval_lj_dsf_direct_into!(F_dir, G_dir, q_pos, box_ortho, fac_exact)
        @test abs(u_dir - u_jl) < 1e-10
        @test maximum(abs, F_dir - F_jl) < 1e-10
        @test maximum(abs, G_dir - G_jl) < 1e-8

        F_dir_novir = fill(-777.0, 3, N)
        u_dir_novir = eval_lj_dsf_direct_into!(F_dir_novir, nothing, q_pos, box_ortho, fac_exact)
        @test u_dir_novir === u_dir
        @test all(F_dir_novir .=== F_dir)

        # NoBox support & error guard
        tc_nb = Neighbors.TilePairCache(q_pos, Types.NoBox(), rcut, skin)
        fac_nb = TriumTiledNonbondedFactor(
            sigma, epsilon, charges;
            rcut = rcut, alpha = alpha, coulomb_prefactor = pref,
            topology = topo, cache = tc_nb, use_native_bridge = true, include_self = false
        )
        fac_nb_jl = TriumTiledNonbondedFactor(
            sigma, epsilon, charges;
            rcut = rcut, alpha = alpha, coulomb_prefactor = pref,
            topology = topo, cache = tc_nb, use_native_bridge = false, include_self = false
        )
        F_nb = zeros(Float64, 3, N); F_nb_jl = zeros(Float64, 3, N)
        u_nb = eval_lj_dsf_direct_into!(F_nb, nothing, q_pos, Types.NoBox(), fac_nb)
        u_nb_jl = eval_lj_dsf_pairs!(F_nb_jl, nothing, q_pos, Types.NoBox(), fac_nb_jl)
        @test abs(u_nb - u_nb_jl) < 1e-10
        @test maximum(abs, F_nb - F_nb_jl) < 1e-10
        @test_throws ArgumentError eval_lj_dsf_direct_into!(F_nb, zeros(3, 3), q_pos, Types.NoBox(), fac_nb)

        # Fast-math NEON accuracy vs exact (< 1e-5 relative force & energy error)
        F_fast = zeros(Float64, 3, N)
        u_fast = eval_lj_dsf_direct_into!(F_fast, nothing, q_pos, box_ortho, fac_fast)
        @test abs(u_fast - u_jl) / max(1.0, abs(u_jl)) < 1e-5
        @test maximum(abs, F_fast - F_jl) / max(1.0, maximum(abs, F_jl)) < 1e-5

        # Native Verlet CSR Plan ABI checks (live when libtrium_mm_bridge exports xbt_verlet_plan_*)
        has_verlet_abi = TXBT._SYM_VERLET_CREATE[] != C_NULL &&
                         TXBT._SYM_VERLET_EVAL_MT[] != C_NULL &&
                         TXBT._SYM_VERLET_DESTROY[] != C_NULL &&
                         TXBT._SYM_VERLET_NUM_PAIRS[] != C_NULL

        if has_verlet_abi
            np0 = compiled_verlet_num_pairs(fac_exact, q_pos, box_ortho)
            @test np0 > 0

            # Small displacement (< skin/2): plan must be reused without rebuilding tc
            rebuilds_0 = tc.n_rebuilds
            q_small = copy(q_pos)
            q_small[1, 1] += 0.05
            F_small = zeros(Float64, 3, N)
            F_small_jl = zeros(Float64, 3, N)
            u_small = eval_lj_dsf_direct_into!(F_small, nothing, q_small, box_ortho, fac_exact)
            u_small_jl = eval_lj_dsf_pairs!(F_small_jl, nothing, q_small, box_ortho, fac_jl)
            @test tc.n_rebuilds == rebuilds_0
            @test abs(u_small - u_small_jl) < 1e-10
            @test maximum(abs, F_small - F_small_jl) < 1e-10

            # Large displacement (> skin): tc rebuilds and compiled Verlet plan dynamically invalidates/rebuilds
            q_big = copy(q_pos)
            q_big[1, 1] += 0.45
            # Ensure moved atom stays safely separated from all others
            min_d = minimum(sqrt(Types.norm2(Types.unwrap_displacement(box_ortho, q_big, 1, j))) for j in 2:N)
            if min_d < 0.35
                q_big[2, 1] += 0.35
            end
            F_big = zeros(Float64, 3, N)
            F_big_jl = zeros(Float64, 3, N)
            u_big = eval_lj_dsf_direct_into!(F_big, nothing, q_big, box_ortho, fac_exact)
            u_big_jl = eval_lj_dsf_pairs!(F_big_jl, nothing, q_big, box_ortho, fac_jl)
            @test tc.n_rebuilds == rebuilds_0 + 1
            np1 = compiled_verlet_num_pairs(fac_exact, q_big, box_ortho)
            @test np1 > 0
            @test abs(u_big - u_big_jl) < 1e-10
            @test maximum(abs, F_big - F_big_jl) < 1e-10

            # Triclinic direct_into! check
            H_tri = [7.0 0.3 0.15; 0.0 6.9 0.25; 0.0 0.0 6.8]
            box_tri = TriclinicBox(H_tri)
            tc_tri = Neighbors.TilePairCache(q_pos, box_tri, rcut, skin)
            fac_tri = TriumTiledNonbondedFactor(
                sigma, epsilon, charges;
                rcut = rcut, alpha = alpha, coulomb_prefactor = pref,
                topology = topo, cache = tc_tri, use_native_bridge = true, include_self = true
            )
            fac_tri_jl = TriumTiledNonbondedFactor(
                sigma, epsilon, charges;
                rcut = rcut, alpha = alpha, coulomb_prefactor = pref,
                topology = topo, cache = tc_tri, use_native_bridge = false, include_self = true
            )
            @test compiled_verlet_num_pairs(fac_tri, q_pos, box_tri) > 0
            F_tri = zeros(Float64, 3, N); G_tri = zeros(Float64, 3, 3)
            F_tri_jl = zeros(Float64, 3, N); G_tri_jl = zeros(Float64, 3, 3)
            u_tri = eval_lj_dsf_direct_into!(F_tri, G_tri, q_pos, box_tri, fac_tri)
            u_tri_jl = eval_lj_dsf_pairs!(F_tri_jl, G_tri_jl, q_pos, box_tri, fac_tri_jl)
            @test abs(u_tri - u_tri_jl) < 1e-10
            @test maximum(abs, F_tri - F_tri_jl) < 1e-10
            @test maximum(abs, G_tri - G_tri_jl) < 1e-8
        else
            @test compiled_verlet_num_pairs(fac_exact, q_pos, box_ortho) == 0
        end

        clear_verlet_plan_cache!()
    end
end

println("All TriumXBT tests passed!")
