# Unit tests for Alchemical module:
# 1. Soft-core singularity avoidance at r = 0.0
# 2. Cartesian force adjoint consistency (central FD < 1e-9) across NoBox, OrthorhombicBox, TriclinicBox (:none and :lfs)
# 3. Box derivative consistency (central FD < 1e-7) on OrthorhombicBox and TriclinicBox (:none and :lfs)
# 4. Lambda parameter and schedule derivatives consistency (central FD < 1e-9) (:none and :lfs)
# 5. Exact analytical shifted-harmonic benchmark for TI, FEP, BAR, and MBAR (< 2.5 SE recovery)

using Test
using Random
using LinearAlgebra
using Statistics

# Load FoundationCore or fallback to standalone module
if !isdefined(Main, :FoundationCore)
    try
        @eval using FoundationCore
    catch e
        @info "Loading standalone FoundationCore modules for test_alchemical"
        @eval module FoundationCore
            include(joinpath(@__DIR__, "../src/Types.jl"))
            include(joinpath(@__DIR__, "../src/Factors.jl"))
            include(joinpath(@__DIR__, "../src/ParameterGradients.jl"))
            include(joinpath(@__DIR__, "../src/EquilibriumSensitivities.jl"))
        end
    end
end

if !isdefined(FoundationCore, :Alchemical)
    Base.include(FoundationCore, joinpath(@__DIR__, "../src/Alchemical.jl"))
end

using .FoundationCore
using .FoundationCore.Types
using .FoundationCore.Factors
using .FoundationCore.Alchemical

@testset "Alchemical Free Energy Engine" begin

    @testset "Alchemical Coupling Schedules" begin
        # Schedule construction and bounds
        s_sim = AlchemicalSchedule(:simultaneous)
        @test s_sim.mode === :simultaneous
        @test s_sim.lambda_split == 0.5

        s_stage = AlchemicalSchedule(:staged_decouple; lambda_split = 0.6)
        @test s_stage.mode === :staged_decouple
        @test s_stage.lambda_split == 0.6

        @test_throws ArgumentError AlchemicalSchedule(:invalid_mode)
        @test_throws ArgumentError AlchemicalSchedule(:simultaneous; lambda_split = -0.1)
        @test_throws ArgumentError AlchemicalSchedule(:simultaneous; lambda_split = 1.0)

        # Simultaneous schedule evaluations
        st0 = schedule_state(s_sim, 0.0)
        @test st0.lambda_lj == 0.0 && st0.lambda_elec == 0.0
        st1 = schedule_state(s_sim, 1.0)
        @test st1.lambda_lj == 1.0 && st1.lambda_elec == 1.0
        st_mid = evaluate_schedule(s_sim, 0.4)
        @test st_mid.lambda_lj == 0.4 && st_mid.lambda_elec == 0.4
        d_sim = schedule_derivatives(s_sim, 0.4)
        @test d_sim.dlambda_lj == 1.0 && d_sim.dlambda_elec == 1.0

        # Staged decouple schedule: electrostatics turns off first on [0, s], LJ turns off on [s, 1]
        s_dec = AlchemicalSchedule(:staged_decouple; lambda_split = 0.5)
        st_dec0 = schedule_state(s_dec, 0.0)
        @test st_dec0.lambda_lj == 1.0 && st_dec0.lambda_elec == 1.0

        st_dec_stage1 = schedule_state(s_dec, 0.25)
        @test st_dec_stage1.lambda_lj == 1.0
        @test isapprox(st_dec_stage1.lambda_elec, 0.5; atol = 1e-12)

        st_dec_split = schedule_state(s_dec, 0.5)
        @test st_dec_split.lambda_lj == 1.0
        @test st_dec_split.lambda_elec == 0.0

        st_dec_stage2 = schedule_state(s_dec, 0.75)
        @test isapprox(st_dec_stage2.lambda_lj, 0.5; atol = 1e-12)
        @test st_dec_stage2.lambda_elec == 0.0

        st_dec1 = schedule_state(s_dec, 1.0)
        @test st_dec1.lambda_lj == 0.0 && st_dec1.lambda_elec == 0.0

        # Staged decouple derivatives
        d_s1 = schedule_derivatives(s_dec, 0.2)
        @test d_s1.dlambda_lj == 0.0 && isapprox(d_s1.dlambda_elec, -2.0; atol = 1e-12)
        d_s2 = schedule_derivatives(s_dec, 0.8)
        @test isapprox(d_s2.dlambda_lj, -2.0; atol = 1e-12) && d_s2.dlambda_elec == 0.0

        # AlchemicalState validation
        @test_throws ArgumentError AlchemicalState(-0.1, 0.5)
        @test_throws ArgumentError AlchemicalState(0.5, 1.1)
    end

    @testset "Constructors and with_lambdas" begin
        # Direct pair constructor
        f_direct = AlchemicalNonbondedFactor(
            [1, 2], [2, 3], [1.2, 1.5], [0.8, 1.1], [-0.5, 0.3];
            lj_scale = [1.0, 0.5], elec_scale = [1.0, 0.5],
            lambda_lj = 0.7, lambda_elec = 0.4,
            alpha_lj = 0.5, alpha_elec = 0.5,
            coulomb_prefactor = 138.935, mode = :none
        )
        @test length(f_direct.i) == 2
        @test f_direct.lambda_lj == 0.7
        @test f_direct.lambda_elec == 0.4

        # with_lambdas non-allocating copy
        f_updated = with_lambdas(f_direct, 0.9, 0.1)
        @test f_updated.lambda_lj == 0.9
        @test f_updated.lambda_elec == 0.1
        @test f_updated.sigma === f_direct.sigma # array sharing without reallocation

        # Atom-subset convenience constructor
        n_atoms = 5
        alch_atoms = [1, 2] # 2 alchemical ligand atoms, 3 environment atoms
        sigma_atom = [1.0, 1.1, 1.2, 1.3, 1.4]
        epsilon_atom = [0.5, 0.6, 0.7, 0.8, 0.9]
        charges = [0.2, -0.3, 0.1, -0.1, 0.0]
        exc = Set([(1, 3)]) # atom 1 and 3 excluded
        sc14 = Dict((1, 4) => (0.5, 0.8333))

        f_subset = AlchemicalNonbondedFactor(
            n_atoms, alch_atoms, sigma_atom, epsilon_atom, charges;
            excluded_pairs = exc, scale_14_pairs = sc14,
            include_intra_alchemical = true,
            lambda_lj = 1.0, lambda_elec = 1.0,
            mode = :lfs, rcut = 2.5
        )
        # Expected pairs:
        # ligand 1 with env (2 atoms: 4, 5, since 3 is excluded)
        # ligand 2 with env (3 atoms: 3, 4, 5)
        # intra-ligand (1, 2)
        # total = 2 + 3 + 1 = 6 pairs
        @test length(f_subset.i) == 6
        @test Factors.is_local(f_subset) == false
        @test Factors.force_consistent(f_subset) == true
        @test Factors.factor_support_size(f_subset) == 5
    end

    @testset "1. Soft-Core Singularity Avoidance at r = 0.0" begin
        # When two particles are coincident (r = 0.0), soft-core potential must be finite,
        # non-NaN, non-Inf, and have exactly zero Cartesian force!
        q_coincident = [1.0 1.0; 2.0 2.0; 3.0 3.0] # 2 atoms at identical positions
        f_zero_none = AlchemicalNonbondedFactor(
            [1], [2], [1.2], [2.0], [-1.0]; # strong attractive opposite charges!
            lambda_lj = 0.5, lambda_elec = 0.5,
            alpha_lj = 0.5, alpha_elec = 0.5,
            coulomb_prefactor = 138.935, mode = :none
        )
        F_buf = zeros(Float64, 3, 2)
        u_none = Factors.evaluate!(F_buf, f_zero_none, q_coincident, Types.NoBox())
        @test isfinite(u_none)
        @test !isnan(u_none)
        @test !isinf(u_none)
        @test all(abs.(F_buf) .== 0.0) # exact zero force at r = 0

        # Test derivatives at r = 0
        der_none = alchemical_derivatives(f_zero_none, q_coincident, Types.NoBox())
        @test isfinite(der_none.dU_dlambda_lj)
        @test isfinite(der_none.dU_dlambda_elec)

        # Test in :lfs mode at r = 0
        f_zero_lfs = AlchemicalNonbondedFactor(
            [1], [2], [1.2], [2.0], [-1.0];
            lambda_lj = 0.5, lambda_elec = 0.5,
            alpha_lj = 0.5, alpha_elec = 0.5,
            coulomb_prefactor = 138.935, mode = :lfs, rcut = 2.5
        )
        F_lfs = zeros(Float64, 3, 2)
        u_lfs = Factors.evaluate!(F_lfs, f_zero_lfs, q_coincident, Types.NoBox())
        @test isfinite(u_lfs)
        @test !isnan(u_lfs)
        @test !isinf(u_lfs)
        @test all(abs.(F_lfs) .== 0.0)
    end

    @testset "2. Central Finite-Difference Cartesian Forces (< 1e-9)" begin
        # 4 atoms with varying parameters and positions
        q = [
            1.0  2.5  1.2  3.4;
            1.2  3.1  2.8  1.5;
            0.5  0.8  2.2  3.0
        ]
        sigma = [1.2, 1.0, 1.4, 1.1]
        eps = [0.8, 1.2, 0.9, 1.5]
        charges = [0.5, -0.4, 0.3, -0.6]

        boxes = (
            ("NoBox", Types.NoBox()),
            ("OrthorhombicBox", Types.OrthorhombicBox((10.0, 11.0, 12.0))),
            ("TriclinicBox", Types.TriclinicBox([9.0 1.2 0.8; 0.0 8.5 1.0; 0.0 0.0 9.5]))
        )

        modes = (:none, :lfs)

        for mode in modes
            rcut = mode === :lfs ? 2.8 : Inf
            f = AlchemicalNonbondedFactor(
                4, [1, 2], sigma, eps, charges;
                lambda_lj = 0.65, lambda_elec = 0.45,
                alpha_lj = 0.5, alpha_elec = 0.5,
                coulomb_prefactor = 1.0,
                mode = mode, rcut = rcut
            )

            for (bname, box) in boxes
                # Analytic forces
                F_analytic = zeros(Float64, 3, 4)
                U0 = Factors.evaluate!(F_analytic, f, q, box)

                # Central finite-difference forces
                h = 1e-6
                F_fd = zeros(Float64, 3, 4)
                F_dummy = zeros(Float64, 3, 4)

                for a in 1:4
                    for d in 1:3
                        qp = copy(q); qp[d, a] += h
                        qm = copy(q); qm[d, a] -= h
                        fill!(F_dummy, 0.0)
                        Up = Factors.evaluate!(F_dummy, f, qp, box)
                        fill!(F_dummy, 0.0)
                        Um = Factors.evaluate!(F_dummy, f, qm, box)
                        F_fd[d, a] = -(Up - Um) / (2.0 * h)
                    end
                end

                max_err = maximum(abs.(F_analytic .- F_fd))
                @test max_err < 1e-9
            end
        end
    end

    @testset "3. Central Finite-Difference Box Derivatives (< 1e-7)" begin
        # 4 atoms in periodic boxes
        sigma = [1.1, 1.2, 1.0, 1.3]
        eps = [1.0, 0.8, 1.2, 0.9]
        charges = [0.4, -0.5, 0.2, -0.3]

        for mode in (:none, :lfs)
            rcut = mode === :lfs ? 2.5 : Inf
            f = AlchemicalNonbondedFactor(
                4, [1, 2], sigma, eps, charges;
                lambda_lj = 0.7, lambda_elec = 0.6,
                alpha_lj = 0.5, alpha_elec = 0.5,
                coulomb_prefactor = 1.0,
                mode = mode, rcut = rcut
            )

            # (a) OrthorhombicBox box derivatives
            L0 = (8.0, 9.0, 10.0)
            box_ortho = Types.OrthorhombicBox(L0)
            # Fractional coordinates s in [0.2, 0.8]
            S = [
                0.25 0.75 0.30 0.65;
                0.30 0.20 0.70 0.80;
                0.40 0.60 0.25 0.75
            ]
            q0 = S .* [L0[1], L0[2], L0[3]]

            G_ortho = alchemical_box_derivative(f, q0, box_ortho)
            h = 1e-5

            for d in 1:3
                Lp = [L0[1], L0[2], L0[3]]; Lp[d] += h
                Lm = [L0[1], L0[2], L0[3]]; Lm[d] -= h
                box_p = Types.OrthorhombicBox(Lp)
                box_m = Types.OrthorhombicBox(Lm)
                qp = S .* Lp
                qm = S .* Lm
                F_dummy = zeros(Float64, 3, 4)
                Up = Factors.evaluate!(F_dummy, f, qp, box_p)
                fill!(F_dummy, 0.0)
                Um = Factors.evaluate!(F_dummy, f, qm, box_m)
                fd_G_dd = (Up - Um) / (2.0 * h)
                @test isapprox(G_ortho[d, d], fd_G_dd; atol = 1e-7)
            end

            # (b) TriclinicBox box derivatives across all 9 matrix entries
            H0 = [
                8.0  1.0  0.7;
                0.0  7.5  0.9;
                0.0  0.0  8.5
            ]
            box_tri = Types.TriclinicBox(H0)
            q_tri = H0 * S
            G_tri = alchemical_box_derivative(f, q_tri, box_tri)

            for a in 1:3, b in 1:3
                Hp = copy(H0); Hp[a, b] += h
                Hm = copy(H0); Hm[a, b] -= h
                box_p = Types.TriclinicBox(Hp)
                box_m = Types.TriclinicBox(Hm)
                qp = Hp * S
                qm = Hm * S
                F_dummy = zeros(Float64, 3, 4)
                Up = Factors.evaluate!(F_dummy, f, qp, box_p)
                fill!(F_dummy, 0.0)
                Um = Factors.evaluate!(F_dummy, f, qm, box_m)
                fd_G_ab = (Up - Um) / (2.0 * h)
                @test isapprox(G_tri[a, b], fd_G_ab; atol = 1e-7)
            end
        end
    end

    @testset "4. Central Finite-Difference Lambda Derivatives (< 1e-9)" begin
        q = [
            1.0  2.2  1.5;
            1.1  2.8  3.0;
            0.5  1.2  2.0
        ]
        sigma = [1.2, 1.1, 1.3]
        eps = [1.0, 1.2, 0.8]
        charges = [0.5, -0.3, 0.2]

        h = 1e-6
        for mode in (:none, :lfs)
            rcut = mode === :lfs ? 2.5 : Inf
            f0 = AlchemicalNonbondedFactor(
                3, [1], sigma, eps, charges;
                lambda_lj = 0.6, lambda_elec = 0.4,
                alpha_lj = 0.5, alpha_elec = 0.5,
                coulomb_prefactor = 1.0,
                mode = mode, rcut = rcut
            )

            res = alchemical_derivatives(f0, q, Types.NoBox())

            # (a) dU / dlambda_lj
            f_lj_p = with_lambdas(f0, 0.6 + h, 0.4)
            f_lj_m = with_lambdas(f0, 0.6 - h, 0.4)
            Up_lj = alchemical_derivatives(f_lj_p, q, Types.NoBox()).energy
            Um_lj = alchemical_derivatives(f_lj_m, q, Types.NoBox()).energy
            fd_dU_dlj = (Up_lj - Um_lj) / (2.0 * h)
            @test isapprox(res.dU_dlambda_lj, fd_dU_dlj; atol = 1e-9)

            # (b) dU / dlambda_elec
            f_el_p = with_lambdas(f0, 0.6, 0.4 + h)
            f_el_m = with_lambdas(f0, 0.6, 0.4 - h)
            Up_el = alchemical_derivatives(f_el_p, q, Types.NoBox()).energy
            Um_el = alchemical_derivatives(f_el_m, q, Types.NoBox()).energy
            fd_dU_delec = (Up_el - Um_el) / (2.0 * h)
            @test isapprox(res.dU_dlambda_elec, fd_dU_delec; atol = 1e-9)

            # (c) schedule_derivative (:simultaneous)
            sched_sim = AlchemicalSchedule(:simultaneous)
            lam_test = 0.45
            sd_sim = schedule_derivative(f0, q, sched_sim, lam_test)
            st_p = schedule_state(sched_sim, lam_test + h)
            st_m = schedule_state(sched_sim, lam_test - h)
            Up_sim = alchemical_derivatives(with_lambdas(f0, st_p.lambda_lj, st_p.lambda_elec), q).energy
            Um_sim = alchemical_derivatives(with_lambdas(f0, st_m.lambda_lj, st_m.lambda_elec), q).energy
            fd_dU_dlam_sim = (Up_sim - Um_sim) / (2.0 * h)
            @test isapprox(sd_sim.dU_dlambda, fd_dU_dlam_sim; atol = 1e-9)

            # (d) schedule_derivative (:staged_decouple) in both stages
            sched_stg = AlchemicalSchedule(:staged_decouple; lambda_split = 0.5)
            for lam_stg in (0.25, 0.75)
                sd_stg = schedule_derivative(f0, q, sched_stg, lam_stg)
                st_p = schedule_state(sched_stg, lam_stg + h)
                st_m = schedule_state(sched_stg, lam_stg - h)
                Up_stg = alchemical_derivatives(with_lambdas(f0, st_p.lambda_lj, st_p.lambda_elec), q).energy
                Um_stg = alchemical_derivatives(with_lambdas(f0, st_m.lambda_lj, st_m.lambda_elec), q).energy
                fd_dU_dlam_stg = (Up_stg - Um_stg) / (2.0 * h)
                @test isapprox(sd_stg.dU_dlambda, fd_dU_dlam_stg; atol = 1e-9)
            end
        end
    end

    @testset "5. Shifted-Harmonic Analytical Benchmark (TI, FEP, BAR, MBAR)" begin
        # Analytical system:
        # U_λ(x, y) = 0.5 * k * (x - λ a)^2 + 0.5 * (1 + λ) * k2 * y^2
        # Partition function:
        # Z(λ) = sqrt(2π / (β k)) * sqrt(2π / (β (1+λ) k2)) ∝ 1 / sqrt(1 + λ)
        # Analytical free energy difference:
        # ΔF = F(1) - F(0) = (1 / 2β) * log(2)
        Random.seed!(2026)
        beta = 1.0
        k = 2.0
        a = 0.4
        k2 = 2.0
        exact_dF = 0.5 * log(2.0) / beta # ≈ 0.34657359

        # Generate samples from multiple lambda windows
        lambdas = collect(range(0.0, 1.0, length = 11))
        K = length(lambdas)
        N_samples = 15000
        N_k = fill(N_samples, K)

        samples_x = [Float64[] for _ in 1:K]
        samples_y = [Float64[] for _ in 1:K]
        dU_dlam_samples = [Float64[] for _ in 1:K]

        for (idx, lam) in enumerate(lambdas)
            # Sample from canonical distribution at state λ
            sx = lam * a .+ randn(N_samples) ./ sqrt(beta * k)
            sy = randn(N_samples) ./ sqrt(beta * (1.0 + lam) * k2)
            samples_x[idx] = sx
            samples_y[idx] = sy

            # dU/dλ = -k*a*(x - λ a) + 0.5*k2*y^2
            du_dlam = -k * a .* (sx .- lam * a) .+ 0.5 * k2 .* sy.^2
            dU_dlam_samples[idx] = du_dlam
        end

        # -----------------------------------------------------------
        # 1. Thermodynamic Integration (TI)
        # -----------------------------------------------------------
        ti_res = estimate_ti(lambdas, dU_dlam_samples)
        @test isapprox(ti_res.lambdas, lambdas)
        err_ti = abs(ti_res.delta_f - exact_dF)
        @test err_ti <= 2.5 * ti_res.se

        # -----------------------------------------------------------
        # 2. Free Energy Perturbation (FEP) on sub-window [0, 0.2]
        # -----------------------------------------------------------
        # Analytical ΔF(0 -> 0.2) = 0.5 * log(1.2) / beta
        exact_dF_fep = 0.5 * log(1.2) / beta
        x0 = samples_x[1]
        y0 = samples_y[1]
        lam_target = 0.2
        # ΔU = U_target(x0, y0) - U_0(x0, y0)
        dU_fep = 0.5 * k .* ((x0 .- lam_target * a).^2 .- x0.^2) .+ 0.5 * lam_target * k2 .* y0.^2
        fep_res = estimate_fep(dU_fep, beta)
        err_fep = abs(fep_res.delta_f - exact_dF_fep)
        @test err_fep <= 2.5 * fep_res.se
        @test fep_res.ess > 100.0

        # -----------------------------------------------------------
        # 3. Bennett Acceptance Ratio (BAR)
        # -----------------------------------------------------------
        # Forward work from state 0 to state 1
        x1 = samples_x[end]
        y1 = samples_y[end]
        w_fwd = 0.5 * k .* ((x0 .- a).^2 .- x0.^2) .+ 0.5 * k2 .* y0.^2
        # Reverse work from state 1 to state 0
        w_rev = 0.5 * k .* (x1.^2 .- (x1 .- a).^2) .- 0.5 * k2 .* y1.^2

        bar_res = estimate_bar(w_fwd, w_rev, beta)
        @test bar_res.converged == true
        @test 0.0 < bar_res.overlap <= 1.0
        err_bar = abs(bar_res.delta_f - exact_dF)
        @test err_bar <= 2.5 * bar_res.se

        # -----------------------------------------------------------
        # 4. Multistate Bennett Acceptance Ratio (MBAR)
        # -----------------------------------------------------------
        N_tot = K * N_samples
        u_kn = zeros(Float64, K, N_tot)
        all_x = vcat(samples_x...)
        all_y = vcat(samples_y...)

        for i in 1:K
            lam_i = lambdas[i]
            for n in 1:N_tot
                xn = all_x[n]
                yn = all_y[n]
                u_val = 0.5 * k * (xn - lam_i * a)^2 + 0.5 * (1.0 + lam_i) * k2 * yn^2
                u_kn[i, n] = beta * u_val
            end
        end

        mbar_res = estimate_mbar(u_kn, N_k)
        @test mbar_res.converged == true
        @test mbar_res.se_delta_f > 0.0
        @test size(mbar_res.overlap_matrix) == (K, K)
        @test all(diag(mbar_res.overlap_matrix) .> 0.0)

        # In MBAR, delta_f is in units of kBT (dimensionless)
        dF_mbar_phys = mbar_res.delta_f / beta
        se_mbar_phys = mbar_res.se_delta_f / beta
        err_mbar = abs(dF_mbar_phys - exact_dF)
        @test err_mbar <= 2.5 * se_mbar_phys
    end

end
