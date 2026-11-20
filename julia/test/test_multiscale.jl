# Unit tests for Multiscale module:
# 1. SymmetryAssemblySpec & expand_symmetry_assembly (atom count, affine math, masses/charges)
# 2. assign_multiscale_zones & zone_counts on NoBox and OrthorhombicBox (MIC wrapping)
# 3. AdaptiveMultiscaleFactor: central finite-difference forces (< 1e-6 rel err),
#    Newton's 3rd law momentum conservation, and periodic box derivative consistency
# 4. softcore_relax_clashes! on clashing atomic cluster (r = 0.02 nm): finite energy/forces
#    and large energy reduction
# 5. DirtyTileDeltaTracker sparse 32-atom Morton tile delta filtering

using Test
using Random
using LinearAlgebra

if !isdefined(Main, :FoundationCore)
    try
        @eval using FoundationCore
    catch
        @eval module FoundationCore
            include(joinpath(@__DIR__, "../src/Types.jl"))
            include(joinpath(@__DIR__, "../src/Factors.jl"))
            include(joinpath(@__DIR__, "../src/Neighbors.jl"))
            include(joinpath(@__DIR__, "../src/ParameterGradients.jl"))
            include(joinpath(@__DIR__, "../src/EquilibriumSensitivities.jl"))
            include(joinpath(@__DIR__, "../src/Alchemical.jl"))
            include(joinpath(@__DIR__, "../src/Multiscale.jl"))
        end
    end
end

if !isdefined(FoundationCore, :Multiscale)
    Base.include(FoundationCore, joinpath(@__DIR__, "../src/Multiscale.jl"))
end

using .FoundationCore
using .FoundationCore.Types
using .FoundationCore.Factors
using .FoundationCore.Neighbors
using .FoundationCore.Alchemical
using .FoundationCore.Multiscale

@testset "Multiscale 3-Zone & Clash Relaxation" begin

    @testset "1. Symmetry Assembly Expansion" begin
        asu_pos = [
            1.0  2.0  0.5;
            0.0  1.5 -0.5;
            0.0  0.0  1.0
        ]
        asu_masses = [12.011, 14.007, 15.999]
        asu_charges = [0.25, -0.50, 0.25]

        # K = 4 transforms: identity, 90-deg Z rotation + translation, 180-deg Z rotation, translation
        T1 = Matrix{Float64}(I, 4, 4)
        T2 = [
            0.0 -1.0  0.0  5.0;
            1.0  0.0  0.0 -2.0;
            0.0  0.0  1.0  3.0;
            0.0  0.0  0.0  1.0
        ]
        T3 = [
            -1.0  0.0  0.0  0.0;
             0.0 -1.0  0.0  0.0;
             0.0  0.0  1.0  1.5;
             0.0  0.0  0.0  1.0
        ]
        T4 = [
            1.0  0.0  0.0 -4.0;
            0.0  1.0  0.0  6.0;
            0.0  0.0  1.0 -1.0;
            0.0  0.0  0.0  1.0
        ]

        spec = SymmetryAssemblySpec(asu_pos, asu_masses, asu_charges, [T1, T2, T3, T4])
        asm = expand_symmetry_assembly(spec)

        M = size(asu_pos, 2)
        K = 4
        @test size(asm.positions) == (3, K * M)
        @test length(asm.masses) == K * M
        @test length(asm.charges) == K * M

        # Verify exact affine transformation for every copy and atom
        transforms = [T1, T2, T3, T4]
        for k in 1:K
            Tk = transforms[k]
            R = Tk[1:3, 1:3]
            t = Tk[1:3, 4]
            for m in 1:M
                idx = (k - 1) * M + m
                expected = R * asu_pos[:, m] + t
                @test isapprox(asm.positions[:, idx], expected; atol = 1e-14)
                @test asm.masses[idx] == asu_masses[m]
                @test asm.charges[idx] == asu_charges[m]
            end
        end

        # Validation guards
        @test_throws ArgumentError SymmetryAssemblySpec(zeros(2, 3), asu_masses, asu_charges, [T1])
        @test_throws ArgumentError SymmetryAssemblySpec(asu_pos, [1.0, 2.0], asu_charges, [T1])
        @test_throws ArgumentError SymmetryAssemblySpec(asu_pos, [-1.0, 1.0, 1.0], asu_charges, [T1])
        @test_throws ArgumentError SymmetryAssemblySpec(asu_pos, asu_masses, asu_charges, Matrix{Float64}[])
        @test_throws ArgumentError SymmetryAssemblySpec(asu_pos, asu_masses, asu_charges, [Matrix{Float64}(I, 3, 3)])
    end

    @testset "2. Multiscale Zone Partitioning (NoBox & OrthorhombicBox MIC)" begin
        # 6 atoms placed at known distances from origin (0, 0, 0)
        q_nobox = [
            0.0  0.4  0.8  1.2  2.5  9.8;
            0.0  0.0  0.0  0.0  0.0  0.0;
            0.0  0.0  0.0  0.0  0.0  0.0
        ]
        part_nb = assign_multiscale_zones(q_nobox, (0.0, 0.0, 0.0), 0.5, 1.5; box = NoBox())
        @test part_nb.zones == UInt8[
            ZONE_ALL_ATOM,
            ZONE_ALL_ATOM,
            ZONE_COARSE_GRAINED,
            ZONE_COARSE_GRAINED,
            ZONE_RIGID_SCAFFOLD,
            ZONE_RIGID_SCAFFOLD
        ]
        counts_nb = zone_counts(part_nb)
        @test counts_nb.all_atom == 2
        @test counts_nb.coarse_grained == 2
        @test counts_nb.rigid_scaffold == 2

        # OrthorhombicBox of length 10.0: atom 6 at x = 9.8 has MIC distance 0.2 from x = 0.0 -> ZONE_ALL_ATOM!
        # Also test atom at x = 8.8 (MIC distance 1.2 -> ZONE_COARSE_GRAINED)
        q_pbc = [
            0.1  9.8  0.9  8.8  5.0  4.2;
            0.0  0.1  0.0  0.0  0.0  0.0;
            0.0  0.0  0.0  0.0  0.0  0.0
        ]
        box = OrthorhombicBox((10.0, 10.0, 10.0))
        part_pbc = assign_multiscale_zones(q_pbc, (0.0, 0.0, 0.0), 0.5, 1.5; box = box)
        @test part_pbc.zones[1] == ZONE_ALL_ATOM
        @test part_pbc.zones[2] == ZONE_ALL_ATOM       # sqrt(0.2^2 + 0.1^2) = 0.2236 <= 0.5
        @test part_pbc.zones[3] == ZONE_COARSE_GRAINED # 0.9 in (0.5, 1.5]
        @test part_pbc.zones[4] == ZONE_COARSE_GRAINED # MIC dist = 1.2 in (0.5, 1.5]
        @test part_pbc.zones[5] == ZONE_RIGID_SCAFFOLD # 5.0 > 1.5
        @test part_pbc.zones[6] == ZONE_RIGID_SCAFFOLD # 4.2 > 1.5
        counts_pbc = zone_counts(part_pbc)
        @test counts_pbc == (all_atom = 2, coarse_grained = 2, rigid_scaffold = 2)
    end

    @testset "3. AdaptiveMultiscaleFactor Forces, Momentum & Box Derivative" begin
        rng = MersenneTwister(42)
        # 12-atom 3-zone geometry:
        #   Zone 0 (atoms 1..4): r <= 0.6 around origin
        #   Zone 1 (atoms 5..8): 0.6 < r <= 1.4
        #   Zone 2 (atoms 9..12): r > 1.4
        q_ref = [
            -0.25  0.25  0.00  0.00  -0.95  0.95  0.00  0.00  -2.00  2.00  0.00  0.00;
             0.00  0.00 -0.25  0.25   0.00  0.00 -0.95  0.95   0.00  0.00 -2.00  2.00;
             0.10 -0.10  0.15 -0.15   0.20 -0.20  0.10 -0.10   0.30 -0.30  0.25 -0.25
        ]
        # Perturb slightly away from reference so all elastic/scaffold/LJ terms have non-zero forces
        q = q_ref .+ 0.06 .* randn(rng, 3, 12)

        part = assign_multiscale_zones(q_ref, (0.0, 0.0, 0.0), 0.6, 1.4; box = NoBox())
        c = zone_counts(part)
        @test c.all_atom == 4
        @test c.coarse_grained == 4
        @test c.rigid_scaffold == 4

        # Inner All-Atom factor covering Zone 0 + Zone 0-1 boundary pairs
        aa_i = Int[]
        aa_j = Int[]
        for i in 1:4, j in (i+1):8
            push!(aa_i, i)
            push!(aa_j, j)
        end
        n_aa_pairs = length(aa_i)
        aa_fac = AlchemicalNonbondedFactor(
            aa_i, aa_j,
            fill(0.32, n_aa_pairs),
            fill(0.6, n_aa_pairs),
            zeros(Float64, n_aa_pairs);
            lambda_lj = 1.0, lambda_elec = 0.0,
            mode = :lfs, rcut = 2.0
        )

        # Test on NoBox
        ms_nb = AdaptiveMultiscaleFactor(
            aa_fac, part, q_ref;
            box = NoBox(),
            cg_k = 12.0, cg_enm_cutoff = 1.6,
            cg_sigma = 0.35, cg_epsilon = 0.4, cg_rcut = 1.8,
            k_rigid = 35.0
        )
        ctx_nb = Factors.Context(NoBox(), Types.REDUCED, 12)
        F = zeros(Float64, 3, 12)
        u0 = Factors.evaluate!(F, ms_nb, q, ctx_nb)
        @test isfinite(u0)
        @test norm(F) > 1.0

        # Newton's 3rd law: total force must vanish to machine precision
        net_force = vec(sum(F, dims = 2))
        @test norm(net_force) < 1e-11

        # Central finite-difference force verification (< 1e-6 relative error)
        eps_fd = 1e-6
        F_fd = zeros(Float64, 3, 12)
        F_tmp = zeros(Float64, 3, 12)
        q_work = copy(q)
        for j in 1:12, d in 1:3
            orig = q_work[d, j]
            q_work[d, j] = orig + eps_fd
            fill!(F_tmp, 0.0)
            up = Factors.evaluate!(F_tmp, ms_nb, q_work, ctx_nb)
            q_work[d, j] = orig - eps_fd
            fill!(F_tmp, 0.0)
            um = Factors.evaluate!(F_tmp, ms_nb, q_work, ctx_nb)
            q_work[d, j] = orig
            F_fd[d, j] = -(up - um) / (2.0 * eps_fd)
        end
        rel_err = norm(F .- F_fd) / max(norm(F_fd), 1e-12)
        @test rel_err < 1e-6

        # Test periodic box derivative ∂U/∂H on OrthorhombicBox against fixed-fractional FD
        box = OrthorhombicBox((8.0, 8.0, 8.0))
        q_ref_pbc = q_ref .+ 4.0
        q_pbc = q .+ 4.0
        part_pbc = assign_multiscale_zones(q_ref_pbc, (4.0, 4.0, 4.0), 0.6, 1.4; box = box)
        ms_pbc = AdaptiveMultiscaleFactor(
            aa_fac, part_pbc, q_ref_pbc;
            box = box,
            cg_k = 12.0, cg_enm_cutoff = 1.6,
            cg_sigma = 0.35, cg_epsilon = 0.4, cg_rcut = 1.8,
            k_rigid = 35.0
        )
        ctx_pbc = Factors.Context(box, Types.REDUCED, 12)
        G = zeros(Float64, 3, 3)
        accumulate_box_derivative!(G, ms_pbc, q_pbc, ctx_pbc)

        # Diagonal fixed-fractional finite difference dU/dL_a
        s_frac = copy(q_pbc)
        for a in 1:3
            s_frac[a, :] ./= box.L[a]
        end
        eps_h = 1e-6
        for a in 1:3
            Lp = ntuple(k -> k == a ? box.L[k] + eps_h : box.L[k], 3)
            Lm = ntuple(k -> k == a ? box.L[k] - eps_h : box.L[k], 3)
            qp = copy(s_frac); qm = copy(s_frac)
            for k in 1:3
                qp[k, :] .*= Lp[k]
                qm[k, :] .*= Lm[k]
            end
            fill!(F_tmp, 0.0)
            up = Factors.evaluate!(F_tmp, ms_pbc, qp, Factors.Context(OrthorhombicBox(Lp), Types.REDUCED, 12))
            fill!(F_tmp, 0.0)
            um = Factors.evaluate!(F_tmp, ms_pbc, qm, Factors.Context(OrthorhombicBox(Lm), Types.REDUCED, 12))
            fd_aa = (up - um) / (2.0 * eps_h)
            @test isapprox(G[a, a], fd_aa; rtol = 1e-5, atol = 1e-6)
        end
    end

    @testset "4. Soft-Core LJ Clash Relaxation (r = 0.02 nm)" begin
        # Tightly clashing 8-atom cluster with ~0.02 nm spacing
        rng = MersenneTwister(123)
        N = 8
        q_clash = zeros(Float64, 3, N)
        for i in 1:N
            angle = 2.0 * pi * (i - 1) / N
            q_clash[1, i] = 0.01 * cos(angle) + 0.002 * randn(rng)
            q_clash[2, i] = 0.01 * sin(angle) + 0.002 * randn(rng)
            q_clash[3, i] = 0.005 * ((i % 2 == 0) ? 1.0 : -1.0)
        end
        # Guarantee an explicit r = 0.02 nm pair
        q_clash[:, 1] .= (0.0, 0.0, 0.0)
        q_clash[:, 2] .= (0.02, 0.0, 0.0)

        sigma = fill(0.34, N)
        epsilon = fill(1.0, N)

        res = softcore_relax_clashes!(
            q_clash, sigma, epsilon, NoBox();
            rcut = 1.0, alpha_sc = 0.5, lambda = 0.5,
            max_steps = 50, step_size = 0.002, max_disp = 0.02, force_tol = 500.0
        )

        @test isfinite(res.initial_energy)
        @test isfinite(res.final_energy)
        @test isfinite(res.initial_max_force)
        @test isfinite(res.final_max_force)
        @test res.initial_energy > 50.0
        @test res.final_energy < 0.25 * res.initial_energy
        @test res.steps_run > 0

        # Verify periodic OrthorhombicBox relaxation as well
        q_pbc_clash = [
            0.01  9.99  0.01  0.03;
            0.00  0.00  0.02  0.00;
            0.00  0.00  0.00  0.02
        ] # atoms 1 and 2 are 0.02 nm apart across PBC boundary (L = 10.0)
        box = OrthorhombicBox((10.0, 10.0, 10.0))
        res_pbc = softcore_relax_clashes!(
            q_pbc_clash, fill(0.34, 4), fill(1.0, 4), box;
            rcut = 1.0, alpha_sc = 0.5, lambda = 0.5,
            max_steps = 50, step_size = 0.002, max_disp = 0.02, force_tol = 500.0
        )
        @test isfinite(res_pbc.initial_energy) && isfinite(res_pbc.final_energy)
        @test res_pbc.final_energy < 0.25 * res_pbc.initial_energy
        @test res_pbc.converged
    end

    @testset "5. DirtyTileDeltaTracker Sparse Filtering" begin
        rng = MersenneTwister(99)
        N = 100  # ceil(100 / 32) = 4 Morton tiles
        q0 = 10.0 .* rand(rng, 3, N)
        tracker = DirtyTileDeltaTracker(q0, 0.1; box = NoBox())
        @test tracker.tile_size == Neighbors.TILE_SIZE
        @test tracker.n_tiles == cld(N, 32)

        # Identical coordinates -> no dirty tiles
        @test isempty(update_dirty_tiles!(tracker, q0))

        # Sub-threshold jitter -> still no dirty tiles
        q_jitter = copy(q0)
        q_jitter .+= 0.02
        @test isempty(update_dirty_tiles!(tracker, q_jitter))

        # Displace a single atom beyond threshold -> exactly its 1 Morton tile is dirty
        q_move = copy(q0)
        target_atom = 17
        expected_tile = tracker.atom_to_tile[target_atom]
        q_move[1, target_atom] += 0.25
        dirty = update_dirty_tiles!(tracker, q_move; commit = false)
        @test dirty == Int32[expected_tile]
        @test target_atom in tile_atoms(tracker, expected_tile)

        # With commit=true, dirty tile is emitted and then cleared on next identical check
        dirty_commit = update_dirty_tiles!(tracker, q_move; commit = true)
        @test dirty_commit == Int32[expected_tile]
        @test isempty(update_dirty_tiles!(tracker, q_move; commit = false))

        # Periodic MIC awareness: shifting an atom by an exact box length L is 0 MIC displacement
        box = OrthorhombicBox((10.0, 10.0, 10.0))
        tracker_pbc = DirtyTileDeltaTracker(q0, 0.1; box = box)
        q_wrapped = copy(q0)
        q_wrapped[1, 5] += 10.0
        @test isempty(update_dirty_tiles!(tracker_pbc, q_wrapped; box = box))
    end

end
