using Test
using Random
using LinearAlgebra
using FoundationCore

if !isdefined(FoundationCore, :TriumGPU)
    @eval FoundationCore include(joinpath(@__DIR__, "..", "src", "TriumGPU.jl"))
end
if !isdefined(FoundationCore, :Engine)
    @eval FoundationCore include(joinpath(@__DIR__, "..", "src", "Engine.jl"))
end

const TGPU = FoundationCore.TriumGPU
const ENG  = FoundationCore.Engine

using .TGPU
using .ENG
using FoundationCore.Types
using FoundationCore.Factors
using FoundationCore.Neighbors
using FoundationCore.StreamedLJ
using FoundationCore.Dynamics
import FoundationCore.ResolvedSystem as RS
using FoundationCore.TriumXBT
using FoundationCore.TernaryMSM
using FoundationCore.SymplecticTernary

@testset "TriumGPU & Unified Engine Suite" begin

    # ==========================================================================
    # 1. Hardware Probing & AdaptiveGPUProfile Auto-Tuner
    # ==========================================================================
    @testset "1. Hardware Probe & AdaptiveGPUProfile" begin
        backends = available_gpu_backends()
        @test :cpu in backends
        @test backends[end] === :cpu

        has_b = has_gpu_bridge()
        @test has_b isa Bool
        @test init_gpu_bridge!(nothing; backend = :cpu) == false
        init_gpu_bridge!(nothing; backend = :auto)

        prof = probe_gpu_device()
        @test prof isa AdaptiveGPUProfile
        @test prof.backend in (:metal, :vulkan, :sycl, :cpu)
        @test !isempty(prof.device_name)
        @test prof.max_threads_per_tg >= 1
        @test prof.simd_width >= 1
        @test prof.auto_workgroup_size >= 1
        @test prof.auto_gpu_crossover_pairs >= 0

        # Synthetic Apple Metal profile (unified memory -> lower crossover ~150_000)
        metal_prof = AdaptiveGPUProfile(
            backend = :metal,
            device_name = "Apple Silicon GPU",
            has_gpu = true,
            unified_memory = true,
            max_threads_per_tg = 1024,
            simd_width = 32,
            vram_bytes = UInt64(16) * 1024^3
        )
        @test metal_prof.unified_memory == true
        @test metal_prof.simd_width == 32
        @test metal_prof.auto_workgroup_size % 32 == 0
        @test 32 <= metal_prof.auto_workgroup_size <= 1024
        @test metal_prof.auto_gpu_crossover_pairs == 150_000

        # Synthetic discrete PCIe Vulkan/AMD profile (non-unified -> higher crossover ~600_000)
        pcie_prof = AdaptiveGPUProfile(
            backend = :vulkan,
            device_name = "Discrete AMD GPU",
            has_gpu = true,
            unified_memory = false,
            max_threads_per_tg = 256,
            simd_width = 64,
            vram_bytes = UInt64(8) * 1024^3
        )
        @test pcie_prof.unified_memory == false
        @test pcie_prof.simd_width == 64
        @test pcie_prof.auto_workgroup_size == 256
        @test pcie_prof.auto_gpu_crossover_pairs == 600_000

        # CPU fallback profile
        cpu_prof = probe_gpu_device(; backend = :cpu)
        @test cpu_prof.backend === :cpu
        @test cpu_prof.has_gpu == false
        @test cpu_prof.simd_width == 1
        @test cpu_prof.auto_gpu_crossover_pairs == typemax(Int)

        # Crossover routing threshold check
        dummy_fac = TriumTiledNonbondedFactor([0.3, 0.3], [0.5, 0.5], [0.1, -0.1]; rcut = 1.0)
        @test TGPU._resolve_nonbonded_backend(:auto, metal_prof, dummy_fac, 10_000) === :cpu
        if has_gpu_bridge(:metal)
            @test TGPU._resolve_nonbonded_backend(:auto, metal_prof, dummy_fac, 200_000) === :metal
            @test TGPU._resolve_nonbonded_backend(:auto, pcie_prof, dummy_fac, 200_000) === :cpu
        end
    end

    # ==========================================================================
    # 2. Nonbonded GPU Evaluator Parity & Central Finite Differences
    # ==========================================================================
    @testset "2. eval_lj_dsf_gpu! Parity (Orthorhombic, Triclinic, Topology)" begin
        clear_gpu_plan_cache!()
        rng = MersenneTwister(202604)
        N = 48
        rcut = 1.9
        skin = 0.25
        alpha = 0.24
        pref = 138.935456

        box_ortho = OrthorhombicBox((4.8, 5.0, 5.2))
        H_tri = [5.1 0.35 0.15; 0.0 5.0 0.25; 0.0 0.0 4.9]
        box_tri = TriclinicBox(H_tri)

        # Well-separated initial lattice positions
        q = Matrix{Float64}(undef, 3, N)
        idx = 1
        for ix in 0:3, iy in 0:3, iz in 0:2
            q[1, idx] = (ix + 0.3 + 0.3 * rand(rng)) * (4.6 / 4.0)
            q[2, idx] = (iy + 0.3 + 0.3 * rand(rng)) * (4.6 / 4.0)
            q[3, idx] = (iz + 0.3 + 0.3 * rand(rng)) * (4.6 / 3.0)
            idx += 1
        end

        sigma = 0.28 .+ 0.05 .* rand(rng, N)
        epsilon = 0.35 .+ 0.25 .* rand(rng, N)
        charges = 0.45 .* (rand(rng, N) .- 0.5)
        charges .-= sum(charges) / N

        topo = StreamedLJ.PairTopology(
            excluded = [(i, i + 1) for i in 1:2:(N - 1)],
            scaled = [(i, i + 2, 0.5) for i in 1:4:(N - 2)]
        )

        for (box_label, box) in (("OrthorhombicBox", box_ortho), ("TriclinicBox", box_tri))
            tc = Neighbors.TilePairCache(q, box, rcut, skin)
            fac_cpu = TriumTiledNonbondedFactor(
                sigma, epsilon, charges;
                rcut = rcut, alpha = alpha, coulomb_prefactor = pref,
                lj_mode = :lfs, lj14_scale = 0.5, coul14_scale = 0.8333333333333334,
                topology = topo, cache = tc,
                use_native_bridge = true, use_fast_math = false, include_self = true
            )

            F_ref = zeros(Float64, 3, N)
            G_ref = zeros(Float64, 3, 3)
            u_ref = eval_lj_dsf_pairs!(F_ref, G_ref, q, box, fac_cpu)

            for req_backend in (:auto, :cpu, :metal, :gpu)
                F_g = zeros(Float64, 3, N)
                G_g = zeros(Float64, 3, 3)
                u_g = eval_lj_dsf_gpu!(F_g, G_g, q, box, fac_cpu; backend = req_backend)

                rel_u = abs(u_g - u_ref) / max(1.0, abs(u_ref))
                rel_F = maximum(abs, F_g - F_ref) / max(1.0, maximum(abs, F_ref))
                rel_G = maximum(abs, G_g - G_ref) / max(1.0, maximum(abs, G_ref))

                @test rel_u < 1e-4
                @test rel_F < 1e-4
                @test rel_G < 1e-4
            end

            # Non-zero initial accumulation parity
            F_init = randn(rng, 3, N)
            G_init = randn(rng, 3, 3)
            F_acc = copy(F_init)
            G_acc = copy(G_init)
            u_acc = eval_lj_dsf_gpu!(F_acc, G_acc, q, box, fac_cpu; backend = :auto)
            @test abs(u_acc - u_ref) / max(1.0, abs(u_ref)) < 1e-10
            @test maximum(abs, (F_acc - F_init) - F_ref) < 1e-10
            @test maximum(abs, (G_acc - G_init) - G_ref) < 1e-8

            # Central finite-difference check of forces and box derivative
            h_fd = 1e-6
            for atom in (1, 17, N), dim in 1:3
                qp = copy(q); qp[dim, atom] += h_fd
                qm = copy(q); qm[dim, atom] -= h_fd
                up = eval_lj_dsf_gpu!(zeros(3, N), nothing, qp, box, fac_cpu; backend = :auto)
                um = eval_lj_dsf_gpu!(zeros(3, N), nothing, qm, box, fac_cpu; backend = :auto)
                f_fd = -(up - um) / (2.0 * h_fd)
                @test abs(F_ref[dim, atom] - f_fd) < 1e-5
            end

            if box isa OrthorhombicBox
                Gv = zeros(Float64, 3)
                eval_lj_dsf_gpu!(zeros(3, N), Gv, q, box, fac_cpu; backend = :auto)
                @test maximum(abs, Gv - diag(G_ref)) < 1e-10
            end

            # gpu_verlet_num_pairs and rebuild invalidation check
            np = gpu_verlet_num_pairs(fac_cpu, q, box; backend = :gpu)
            @test np >= 0
            if box isa OrthorhombicBox
                reb0 = tc.n_rebuilds
                q_moved = copy(q)
                q_moved[1, 1] += 0.45
                min_d = minimum(sqrt(Types.norm2(Types.unwrap_displacement(box, q_moved, 1, j))) for j in 2:N)
                if min_d < 0.35
                    q_moved[2, 1] += 0.35
                end
                F_mov = zeros(Float64, 3, N)
                F_mov_ref = zeros(Float64, 3, N)
                u_mov = eval_lj_dsf_gpu!(F_mov, nothing, q_moved, box, fac_cpu; backend = :gpu)
                u_mov_ref = eval_lj_dsf_pairs!(F_mov_ref, nothing, q_moved, box, fac_cpu)
                @test tc.n_rebuilds == reb0 + 1
                @test abs(u_mov - u_mov_ref) / max(1.0, abs(u_mov_ref)) < 1e-4
                @test maximum(abs, F_mov - F_mov_ref) / max(1.0, maximum(abs, F_mov_ref)) < 1e-4
            end

            # TriumGPUNonbondedFactor protocol check
            gfac = TriumGPUNonbondedFactor(fac_cpu; backend = :auto)
            @test Factors.factor_support_size(gfac) == N
            @test Factors.is_local(gfac) == false
            @test Factors.force_consistent(gfac) == true
            ctx = Factors.Context(box, Types.REDUCED, N)
            F_proto = zeros(Float64, 3, N)
            u_proto = Factors.evaluate!(F_proto, gfac, q, ctx)
            @test abs(u_proto - u_ref) / max(1.0, abs(u_ref)) < 1e-10
            @test maximum(abs, F_proto - F_ref) < 1e-10
            G_proto = zeros(Float64, 3, 3)
            Factors.accumulate_box_derivative!(G_proto, gfac, q, ctx)
            @test maximum(abs, G_proto - G_ref) < 1e-8
        end

        clear_gpu_plan_cache!()
    end

    # ==========================================================================
    # 3. Dual-Rail Ternary MatVec Parity (0.0 Exact Integer Parity)
    # ==========================================================================
    @testset "3. gpu_ternary_matvec! Exact Integer Parity" begin
        rng = MersenneTwister(9911)
        M = 25
        K_words = 7
        W_P = zeros(UInt64, M, K_words)
        W_N = zeros(UInt64, M, K_words)
        for k in 1:K_words, m in 1:M
            a = rand(rng, UInt64)
            b = rand(rng, UInt64)
            W_P[m, k] = a & ~b
            W_N[m, k] = b & ~a
            @assert (W_P[m, k] & W_N[m, k]) == 0
        end
        x_P = zeros(UInt64, K_words)
        x_N = zeros(UInt64, K_words)
        for k in 1:K_words
            a = rand(rng, UInt64)
            b = rand(rng, UInt64)
            x_P[k] = a & ~b
            x_N[k] = b & ~a
            @assert (x_P[k] & x_N[k]) == 0
        end

        # Reference exact integer dot products
        acc_ref = zeros(Int64, M)
        for m in 1:M
            s = Int64(0)
            for k in 1:K_words
                for bit in 0:63
                    mask = UInt64(1) << bit
                    wt = ((W_P[m, k] & mask) != 0 ? 1 : 0) - ((W_N[m, k] & mask) != 0 ? 1 : 0)
                    xt = ((x_P[k] & mask) != 0 ? 1 : 0) - ((x_N[k] & mask) != 0 ? 1 : 0)
                    s += Int64(wt * xt)
                end
            end
            acc_ref[m] = s
        end

        # Exact integer parity (unit scales -> exact 0.0 difference)
        y_int = zeros(Float64, M)
        for b in (:auto, :cpu, :metal, :gpu)
            fill!(y_int, NaN)
            gpu_ternary_matvec!(y_int, W_P, W_N, nothing, x_P, x_N, 1.0; backend = b)
            @test maximum(abs, y_int .- Float64.(acc_ref)) == 0.0
        end

        # With explicit row_scales and x_scale
        row_scales = rand(rng, M) .* 2.5 .+ 0.25
        x_scale = 0.75
        y_scaled = zeros(Float64, M)
        gpu_ternary_matvec!(y_scaled, W_P, W_N, row_scales, x_P, x_N, x_scale; backend = :auto)
        y_ref = row_scales .* x_scale .* Float64.(acc_ref)
        @test maximum(abs, y_scaled .- y_ref) < 1e-12
    end

    # ==========================================================================
    # 4. 27-Point MSM Stencil Convolution Parity
    # ==========================================================================
    @testset "4. gpu_msm_stencil27! Periodic Convolution Parity" begin
        rng = MersenneTwister(7722)
        Mgrid = 9
        Q_grid = randn(rng, Mgrid, Mgrid, Mgrid)
        _, w27 = TernaryMSM.radix3_stencil_weights(4)
        stencil_k27 = w27 .+ 0.05 .* rand(rng, 27)
        prefactor = 1.75

        # Independent explicit reference using TernaryMSM.tryte_coords
        Phi_ref = zeros(Float64, Mgrid, Mgrid, Mgrid)
        for idx in 1:27
            dx, dy, dz = TernaryMSM.tryte_coords(idx)
            kval = stencil_k27[idx]
            for n3 in 1:Mgrid, n2 in 1:Mgrid, n1 in 1:Mgrid
                m1 = mod1(n1 + dx, Mgrid)
                m2 = mod1(n2 + dy, Mgrid)
                m3 = mod1(n3 + dz, Mgrid)
                Phi_ref[n1, n2, n3] += kval * Q_grid[m1, m2, m3]
            end
        end
        E_ref = 0.5 * prefactor * sum(Q_grid .* Phi_ref)

        for b in (:auto, :cpu, :metal, :gpu)
            Phi_out = fill(-99.0, Mgrid, Mgrid, Mgrid)
            E_out = gpu_msm_stencil27!(Phi_out, Q_grid, stencil_k27, prefactor; backend = b)
            @test maximum(abs, Phi_out .- Phi_ref) < 1e-6
            @test abs(E_out - E_ref) < 1e-5
        end
    end

    # ==========================================================================
    # 5. Unified SimulationEngine Facade
    # ==========================================================================
    @testset "5. SimulationEngine (create_engine, engine_info, evaluate_step!, run_steps!)" begin
        rng = MersenneTwister(5512)
        N = 16
        box = OrthorhombicBox((4.5, 4.5, 4.5))
        q = Matrix{Float64}(undef, 3, N)
        idx = 1
        for ix in 0:3, iy in 0:1, iz in 0:1
            q[1, idx] = 0.6 + ix * 0.95
            q[2, idx] = 0.8 + iy * 1.6
            q[3, idx] = 0.8 + iz * 1.6
            idx += 1
        end
        p = 0.05 .* randn(rng, 3, N)
        p .-= sum(p, dims = 2) ./ N
        masses = fill(12.0, N)
        charges = [isodd(i) ? 0.25 : -0.25 for i in 1:N]
        sigma = fill(0.30, N)
        epsilon = fill(0.45, N)

        topo = StreamedLJ.PairTopology(excluded = [(1, 2), (3, 4)])
        tc = Neighbors.TilePairCache(q, box, 1.8, 0.2)
        nb_fac = TriumTiledNonbondedFactor(
            sigma, epsilon, charges;
            rcut = 1.8, alpha = 0.22, coulomb_prefactor = 138.935456,
            lj_mode = :lfs, topology = topo, cache = tc, include_self = true
        )
        bond_fac1 = Factors.HarmonicBond(1, 2, 500.0, 0.95)
        bond_fac2 = Factors.HarmonicBond(3, 4, 500.0, 0.95)
        factors = Factors.AbstractFactor[bond_fac1, bond_fac2, nb_fac]

        atoms = AtomSet(q, p, masses, charges, fill(:Ar, N))
        sys = Types.System(atoms, box, ChemicalTopology(); units = Types.REDUCED, temperature = 300.0)

        # Contract refusals
        bad_nb = TriumTiledNonbondedFactor(sigma, epsilon, charges; rcut = 1.8, lj_mode = :trunc)
        @test_throws ErrorException create_engine(sys, Factors.AbstractFactor[bad_nb])

        rs_vs = RS.ResolvedSystem(
            masses = [12.0, 12.0, 0.0],
            charges = [0.0, 0.0, 0.0],
            box = box,
            virtual_sites = [RS.VirtualSite(:linear, 3, [1, 2], [0.5, 0.5])]
        )
        @test_throws RS.UnsupportedCapability create_engine(rs_vs)

        # Evaluate parity across backends (:auto, :cpu, :metal, :gpu)
        F_ref = zeros(Float64, 3, N)
        G_ref = zeros(Float64, 3, 3)
        eng_cpu = create_engine(sys, factors; backend = :cpu, integrator = :verlet, dt_ps = 0.001)
        u_cpu = evaluate_step!(eng_cpu, F_ref, q, box; box_deriv = G_ref)

        for b in (:auto, :metal, :gpu)
            eng_b = create_engine(sys, factors; backend = b, integrator = :verlet, dt_ps = 0.001)
            info = engine_info(eng_b)
            @test info.backend === b
            @test info.resolved_backend in (:metal, :vulkan, :sycl, :cpu)
            @test info.n_atoms == N
            @test info.n_factors == 3
            @test info.integrator === :verlet
            @test info.dt_ps == 0.001
            @test info.simd_width >= 1
            @test info.crossover_pairs >= 0
            @test "TriumTiledNonbondedFactor" in info.active_factors

            F_b = zeros(Float64, 3, N)
            G_b = zeros(Float64, 3, 3)
            u_b = evaluate_step!(eng_b, F_b, q, box; box_deriv = G_b)
            @test abs(u_b - u_cpu) / max(1.0, abs(u_cpu)) < 1e-4
            @test maximum(abs, F_b - F_ref) / max(1.0, maximum(abs, F_ref)) < 1e-4
            @test maximum(abs, G_b - G_ref) / max(1.0, maximum(abs, G_ref)) < 1e-4
        end

        # Helper to compute total energy
        function _total_e(eng, st)
            F_tmp = zeros(Float64, 3, N)
            u = evaluate_step!(eng, F_tmp, st.q, box)
            k = 0.5 * sum(st.p[d, j]^2 / masses[j] for j in 1:N, d in 1:3)
            return u + k
        end

        # 5.1 Velocity Verlet NVE conservation across :auto and :cpu
        for b in (:auto, :cpu)
            eng_v = create_engine(sys, factors; backend = b, integrator = :verlet, dt_ps = 0.0005)
            st_v = Dynamics.State(copy(q), copy(p))
            E0 = _total_e(eng_v, st_v)
            run_steps!(eng_v, st_v, 20)
            E1 = _total_e(eng_v, st_v)
            @test isfinite(E1)
            @test abs(E1 - E0) / max(1.0, abs(E0)) < 1e-3
        end

        # 5.2 Langevin BAOAB
        eng_baoab = create_engine(sys, factors; backend = :auto, integrator = :baoab,
                                  dt_ps = 0.0005, temperature = 300.0, gamma_ps_inv = 1.0)
        st_b = Dynamics.State(copy(q), copy(p))
        run_steps!(eng_baoab, st_b, 15)
        @test all(isfinite, st_b.q)
        @test all(isfinite, st_b.p)
        @test st_b.force_valid == true

        # 5.3 Reversible r-RESPA
        eng_respa = create_engine(sys, factors; backend = :auto, integrator = :respa,
                                  dt_ps = 0.0008, respa_m = 2)
        st_r = Dynamics.State(copy(q), copy(p))
        E0_r = _total_e(eng_respa, st_r)
        run_steps!(eng_respa, st_r, 15)
        E1_r = _total_e(eng_respa, st_r)
        @test isfinite(E1_r)
        @test abs(E1_r - E0_r) / max(1.0, abs(E0_r)) < 1e-3

        # 5.4 SymplecticTernary via ResolvedSystem
        rs = RS.ResolvedSystem(
            masses = masses,
            charges = charges,
            box = box,
            factors = factors,
            units = Types.REDUCED,
            provenance = RS.Provenance(source = "test_trium_gpu", force_field = "test")
        )
        eng_st = create_engine(rs; backend = :auto, integrator = :symplectic_ternary, dt_ps = 0.0005)
        st_tern = SymplecticTernaryState(copy(q), copy(p), masses, box)
        run_steps!(eng_st, st_tern, 10)
        @test st_tern.step_index == 10
        # Exact reverse step recovers bit-exact initial state
        force_fn = (qm, Fm) -> evaluate_step!(eng_st, Fm, qm, box)
        st_rev = copy(st_tern)
        simulate_symplectic_ternary!(st_rev, force_fn, eng_st.dt_ps, 10; direction = -1)
        st_init = SymplecticTernaryState(copy(q), copy(p), masses, box)
        @test st_rev.Q == st_init.Q
        @test st_rev.P == st_init.P
    end

    # ==========================================================================
    # 6. Batched GPU MD (run_batched_gpu_steps! vs CPU velocity_verlet_step!)
    # ==========================================================================
    @testset "6. Batched GPU MD (run_batched_gpu_steps! & CPU Reference Parity)" begin
        clear_gpu_plan_cache!()
        rng = MersenneTwister(202611)
        n_mol = 8
        N = 3 * n_mol
        box = OrthorhombicBox((6.0, 6.0, 6.0))
        q0 = Matrix{Float64}(undef, 3, N)
        bonds = Factors.HarmonicBond[]
        angles = Factors.HarmonicAngle[]
        excl_pairs = Tuple{Int, Int}[]

        theta_eq = 1.85
        r_eq = 0.95
        mol_idx = 0
        for ix in 0:1, iy in 0:1, iz in 0:1
            mol_idx += 1
            a = 3 * (mol_idx - 1) + 1
            b = 3 * (mol_idx - 1) + 2
            c = 3 * (mol_idx - 1) + 3
            cx = 1.2 + ix * 3.0
            cy = 1.2 + iy * 3.0
            cz = 1.2 + iz * 3.0
            q0[:, b] = [cx, cy, cz]
            q0[:, a] = [cx + 0.97, cy, cz]
            q0[:, c] = [cx + 0.93 * cos(theta_eq + 0.05), cy + 0.93 * sin(theta_eq + 0.05), cz]

            push!(bonds, Factors.HarmonicBond(a, b, 450.0, r_eq))
            push!(bonds, Factors.HarmonicBond(b, c, 450.0, r_eq))
            push!(angles, Factors.HarmonicAngle(a, b, c, 60.0, theta_eq))
            push!(excl_pairs, (a, b), (b, c), (a, c))
        end

        p0 = 0.04 .* randn(rng, 3, N)
        p0 .-= sum(p0, dims = 2) ./ N
        masses = fill(12.0, N)
        inv_mass = fill(1.0 / 12.0, N)
        charges = [mod1(i, 3) == 2 ? -0.4 : 0.2 for i in 1:N]
        sigma = fill(0.30, N)
        epsilon = fill(0.40, N)

        topo = StreamedLJ.PairTopology(excluded = excl_pairs)
        tc = Neighbors.TilePairCache(q0, box, 2.0, 0.25)
        nb_fac = TriumTiledNonbondedFactor(
            sigma, epsilon, charges;
            rcut = 2.0, alpha = 0.22, coulomb_prefactor = 138.935456,
            lj_mode = :lfs, topology = topo, cache = tc, include_self = true
        )

        dt = 0.0005
        n_steps = 200

        # Exact CPU reference using Dynamics.velocity_verlet_step!
        all_facs = Factors.AbstractFactor[bonds..., angles..., nb_fac]
        hs = Dynamics.HarmonicSystem(all_facs, masses, Factors.Context(box, Types.REDUCED, N))
        q_ref = copy(q0)
        p_ref = copy(p0)
        F_ref = zeros(Float64, 3, N)
        pe0_ref = Dynamics.forces!(hs, q_ref, F_ref)
        E0_ref = pe0_ref + Dynamics.kinetic_energy(hs, p_ref)
        pe1_ref = pe0_ref
        for _ in 1:n_steps
            pe1_ref = Dynamics.velocity_verlet_step!(q_ref, p_ref, F_ref, hs, dt)
        end
        ke1_ref = Dynamics.kinetic_energy(hs, p_ref)
        E1_ref = pe1_ref + ke1_ref

        # Explicit :cpu fallback parity (must match Dynamics.velocity_verlet_step! exactly)
        q_cpu = copy(q0)
        p_cpu = copy(p0)
        F_cpu = zeros(Float64, 3, N)
        res_cpu = run_batched_gpu_steps!(
            q_cpu, p_cpu, inv_mass, box, nb_fac, bonds, angles, dt, n_steps;
            F_out = F_cpu, backend = :cpu
        )
        @test res_cpu.used_gpu == false
        @test abs(res_cpu.total_energy - E1_ref) < 1e-12
        @test abs(res_cpu.potential_energy - pe1_ref) < 1e-12
        @test abs(res_cpu.kinetic_energy - ke1_ref) < 1e-12
        @test maximum(abs, q_cpu - q_ref) < 1e-12
        @test maximum(abs, p_cpu - p_ref) < 1e-12
        @test maximum(abs, F_cpu - F_ref) < 1e-12

        # :auto dispatch (uses GPU batch when xbt_gpu_md_batch_run is loaded, else CPU fallback)
        q_auto = copy(q0)
        p_auto = copy(p0)
        F_auto = zeros(Float64, 3, N)
        res_auto = run_batched_gpu_steps!(
            q_auto, p_auto, inv_mass, box, nb_fac, bonds, angles, dt, n_steps;
            F_out = F_auto, backend = :auto
        )
        @test res_auto.used_gpu == has_gpu_md_batch(:auto)
        @test isfinite(res_auto.total_energy)
        @test abs((res_auto.potential_energy + res_auto.kinetic_energy) - res_auto.total_energy) < 1e-10
        @test abs(res_auto.total_energy - E0_ref) / max(abs(E0_ref), 1.0) < 1e-4
        @test maximum(abs, q_auto - q_ref) < 1e-3
        @test maximum(abs, p_auto - p_ref) < 1e-3
        @test maximum(abs, F_auto - F_ref) / max(1.0, maximum(abs, F_ref)) < 1e-3

        # 0-step & F_out=nothing smoke checks + bounds check
        q_z = copy(q0); p_z = copy(p0)
        res_z = run_batched_gpu_steps!(q_z, p_z, inv_mass, box, nb_fac, bonds, angles, dt, 0; F_out = nothing, backend = :auto)
        @test res_z.used_gpu == false
        @test abs(res_z.total_energy - E0_ref) < 1e-12
        @test_throws ArgumentError run_batched_gpu_steps!(
            copy(q0), copy(p0), inv_mass, box, nb_fac,
            [Factors.HarmonicBond(1, N + 1, 100.0, 1.0)], angles, dt, 10
        )

        # ======================================================================
        # 7. SimulationEngine Batched :verlet & verify_engine_conservation
        # ======================================================================
        @testset "7. SimulationEngine Batched :verlet & verify_engine_conservation" begin
            atoms = AtomSet(copy(q0), copy(p0), masses, charges, fill(:Ar, N))
            sys = Types.System(atoms, box, ChemicalTopology(); units = Types.REDUCED, temperature = 300.0)

            eng_cpu = create_engine(sys, all_facs; backend = :cpu, integrator = :verlet, dt_ps = dt)
            eng_auto = create_engine(sys, all_facs; backend = :auto, integrator = :verlet, dt_ps = dt)
            @test eng_cpu.active_backend === :cpu
            @test eng_auto.active_backend in (:metal, :vulkan, :sycl, :cpu)

            st_cpu = Dynamics.State(copy(q0), copy(p0))
            st_auto = Dynamics.State(copy(q0), copy(p0))
            run_steps!(eng_cpu, st_cpu, n_steps)
            run_steps!(eng_auto, st_auto, n_steps)

            @test st_cpu.force_valid == true
            @test st_auto.force_valid == true
            @test maximum(abs, st_cpu.q - q_ref) < 1e-12
            @test maximum(abs, st_cpu.p - p_ref) < 1e-12
            @test maximum(abs, st_auto.q - st_cpu.q) < 1e-3
            @test maximum(abs, st_auto.p - st_cpu.p) < 1e-3
            @test maximum(abs, st_auto.F - st_cpu.F) / max(1.0, maximum(abs, st_cpu.F)) < 1e-3

            st0 = Dynamics.State(copy(q0), copy(p0))
            q0_snap = copy(st0.q)
            p0_snap = copy(st0.p)

            tel_cpu = verify_engine_conservation(eng_cpu, st0; n_steps = n_steps)
            tel_auto = verify_engine_conservation(eng_auto, st0; n_steps = n_steps)

            # Deep-copy guarantee: st0 must remain untouched
            @test st0.q == q0_snap
            @test st0.p == p0_snap

            for (eng, tel) in ((eng_cpu, tel_cpu), (eng_auto, tel_auto))
                @test tel.n_steps == n_steps
                @test tel.elapsed_s > 0.0
                @test tel.ms_per_step > 0.0
                @test tel.ns_per_day_2fs > 0.0
                @test isfinite(tel.initial_energy)
                @test isfinite(tel.final_energy)
                @test tel.abs_energy_drift >= 0.0
                @test tel.rel_energy_drift < 1e-4
                @test tel.momentum_drift_norm < 1e-3
                @test tel.backend === eng.active_backend
            end

            # Mixed factor (PeriodicDihedral) routes cleanly via existing per-step path
            dihed = Factors.PeriodicDihedral(1, 2, 3, 4, 2.0, 2, 0.0)
            mixed_facs = Factors.AbstractFactor[all_facs..., dihed]
            eng_mix = create_engine(sys, mixed_facs; backend = :auto, integrator = :verlet, dt_ps = dt)
            tel_mix = verify_engine_conservation(eng_mix, st0; n_steps = 20)
            @test tel_mix.n_steps == 20
            @test tel_mix.rel_energy_drift < 1e-3
        end

        clear_gpu_plan_cache!()
    end
end

println("All TriumGPU & Engine tests passed!")
