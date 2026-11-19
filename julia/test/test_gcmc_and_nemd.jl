# Focused unit & physics tests for Phase 9B: GrandCanonical & NEMD.

using Test
using LinearAlgebra
using Random
using Statistics
using FoundationCore

if !isdefined(FoundationCore, :GrandCanonical)
    @eval FoundationCore include(joinpath(@__DIR__, "..", "src", "GrandCanonical.jl"))
end
if !isdefined(FoundationCore, :NEMD)
    @eval FoundationCore include(joinpath(@__DIR__, "..", "src", "NEMD.jl"))
end

const GC = FoundationCore.GrandCanonical
const NM = FoundationCore.NEMD
const T = FoundationCore.Types
const F = FoundationCore.Factors
const D = FoundationCore.Dynamics
const RS = FoundationCore.ResolvedSystem
const TX = FoundationCore.TriumXBT

function fd_forces(fac::F.AbstractFactor, q::Matrix{Float64}, ctx::F.Context; h::Float64 = 1e-6)
    N = size(q, 2)
    F_fd = zeros(Float64, 3, N)
    F_tmp = zeros(Float64, 3, N)
    q_pert = copy(q)
    for j in 1:N, d in 1:3
        q_pert[d, j] = q[d, j] + h
        fill!(F_tmp, 0.0)
        up = F.evaluate!(F_tmp, fac, q_pert, ctx)
        q_pert[d, j] = q[d, j] - h
        fill!(F_tmp, 0.0)
        um = F.evaluate!(F_tmp, fac, q_pert, ctx)
        q_pert[d, j] = q[d, j]
        F_fd[d, j] = -(up - um) / (2.0 * h)
    end
    return F_fd
end

function fd_box_derivative(fac::F.AbstractFactor, q::Matrix{Float64}, box::T.AbstractBox,
                           units::T.Units; h::Float64 = 1e-6)
    H0 = box isa T.OrthorhombicBox ? Matrix(Diagonal([box.L[1], box.L[2], box.L[3]])) :
         Matrix{Float64}((box::T.TriclinicBox).H)
    s = H0 \ q
    N = size(q, 2)
    F_tmp = zeros(Float64, 3, N)
    G_fd = zeros(Float64, 3, 3)
    for b in 1:3, a in 1:3
        Hp = copy(H0); Hp[a, b] += h
        Hm = copy(H0); Hm[a, b] -= h
        ctxp = F.Context(T.TriclinicBox(Hp), units, N)
        ctxm = F.Context(T.TriclinicBox(Hm), units, N)
        fill!(F_tmp, 0.0)
        up = F.evaluate!(F_tmp, fac, Hp * s, ctxp)
        fill!(F_tmp, 0.0)
        um = F.evaluate!(F_tmp, fac, Hm * s, ctxm)
        G_fd[a, b] = (up - um) / (2.0 * h)
    end
    return G_fd
end

@testset "GrandCanonical & NEMD Suite" begin

    # ==========================================================================
    # 1. GCMC Ideal Gas (Single-Species & Multi-Species Detailed Balance)
    # ==========================================================================
    @testset "1. GCMC Ideal Gas Analytical Ensemble" begin
        box = T.OrthorhombicBox((5.0, 5.0, 5.0))
        V = T.box_volume(box)
        temp = 1.5
        units = T.REDUCED
        beta = 1.0 / (units.boltzmann * temp)
        lam = GC.thermal_de_broglie(1.0, temp, units)
        @test isapprox(lam, 1.0 / sqrt(2.0 * pi * temp); rtol = 1e-12)

        # Single-species ideal gas: <N> = (V / Λ^3) exp(β μ)
        n_target = 12.0
        mu = log(n_target * lam^3 / V) / beta
        sp = GC.GCMCSpecies(:id, 1.0, mu, lam, 1.0, 0.0)
        sampler = GC.GrandCanonicalSampler(sp, box, temp; units = units, rc = 2.0, mode = :lfs)
        res = GC.run_gcmc!(sampler, 25_000; rng = MersenneTwister(101), burnin = 5_000)

        se = res.std_loading / sqrt(length(res.n_history) / 20.0)
        @test abs(res.mean_loading - n_target) < max(3.0 * se, 0.5)
        @test isapprox(res.mean_loading, n_target; rtol = 0.08)
        @test all(==(0.0), res.energy_history)

        # Multi-species ideal gas (verifies exact multi-species insert/delete/swap detailed balance)
        n1_target = 8.0
        n2_target = 5.0
        lam1 = 0.35
        lam2 = 0.50
        mu1 = log(n1_target * lam1^3 / V) / beta
        mu2 = log(n2_target * lam2^3 / V) / beta
        sp1 = GC.GCMCSpecies(:A, 1.0, mu1, lam1, 1.0, 0.0)
        sp2 = GC.GCMCSpecies(:B, 2.0, mu2, lam2, 1.0, 0.0)
        msampler = GC.GrandCanonicalSampler([sp1, sp2], box, temp; units = units)
        mres = GC.run_gcmc!(msampler, 40_000; rng = MersenneTwister(202),
                            weights = (0.35, 0.35, 0.15, 0.15), burnin = 8_000)
        mean_n1 = mean(@view mres.species_counts[1, :])
        mean_n2 = mean(@view mres.species_counts[2, :])
        @test isapprox(mean_n1, n1_target; rtol = 0.08)
        @test isapprox(mean_n2, n2_target; rtol = 0.08)
    end

    # ==========================================================================
    # 2. Farkas Steric Screen & Dense LJ + Host Nanopore GCMC
    # ==========================================================================
    @testset "2. Farkas Steric Screen & Dense Host-Guest LJ GCMC" begin
        # Direct equivalence between farkas_steric_screen and exact MIC distance
        box_ortho = T.OrthorhombicBox((6.0, 6.0, 6.0))
        H_tri = [6.0 1.2 0.3; 0.0 5.8 0.7; 0.0 0.0 6.2]
        box_tri = T.TriclinicBox(H_tri)
        rng = MersenneTwister(303)
        q_ex = rand(rng, 3, 25) .* 6.0
        r_min = 0.75

        for box in (box_ortho, box_tri)
            for _ in 1:200
                p = (rand(rng) * 6.0, rand(rng) * 6.0, rand(rng) * 6.0)
                cert = GC.farkas_steric_screen(p, q_ex, box, r_min)
                # Brute-force MIC reference
                q_comb = hcat(reshape(collect(p), 3, 1), q_ex)
                min_r2_ref = Inf
                for j in 1:size(q_ex, 2)
                    d2 = T.norm2(T.unwrap_displacement(box, q_comb, 1, j + 1))
                    if d2 < min_r2_ref
                        min_r2_ref = d2
                    end
                end
                overlap_ref = min_r2_ref < r_min^2
                @test cert.feasible == !overlap_ref
                if cert.feasible
                    @test cert.clash_atom == 0
                    @test cert.reason === :feasible
                    @test isapprox(cert.min_dist_sq, min_r2_ref; atol = 1e-12)
                else
                    @test 1 <= cert.clash_atom <= size(q_ex, 2)
                    @test cert.reason in (:cell_clash, :steric_clash)
                    @test cert.min_dist_sq < r_min^2
                end
            end
        end

        # active_mask test
        mask = fill(false, size(q_ex, 2))
        p_on = (q_ex[1, 1] + 0.05, q_ex[2, 1], q_ex[3, 1])
        @test !GC.farkas_steric_screen(p_on, q_ex, box_ortho, r_min).feasible
        @test GC.farkas_steric_screen(p_on, q_ex, box_ortho, r_min; active_mask = mask).feasible

        # Dense LJ + host framework atoms GCMC: monotonic loading with chemical potential
        box_pore = T.OrthorhombicBox((6.0, 6.0, 6.0))
        q_host = [0.5 0.5 5.5 5.5;
                  1.5 4.5 1.5 4.5;
                  3.0 3.0 3.0 3.0]
        sigma_host = fill(1.0, 4)
        eps_host = fill(1.2, 4)
        temp = 1.2
        lam = 0.4
        mus = [-5.0, -2.0, 0.5]
        loadings = Float64[]
        total_steric_rej = 0

        for mu in mus
            sp = GC.GCMCSpecies(:Ar, 1.0, mu, lam, 1.0, 1.0)
            samp = GC.GrandCanonicalSampler(sp, box_pore, temp; rc = 2.5, mode = :lfs,
                                            hard_core_frac = 0.5,
                                            q_host = q_host, sigma_host = sigma_host,
                                            epsilon_host = eps_host)
            res = GC.run_gcmc!(samp, 8_000; rng = MersenneTwister(404), burnin = 2_000)
            push!(loadings, res.mean_loading)
            total_steric_rej += samp.steric_rejections
            # Verify final configuration has zero hard-core overlaps
            Ng = size(samp.q_guest, 2)
            for i in 1:Ng
                pi = (samp.q_guest[1, i], samp.q_guest[2, i], samp.q_guest[3, i])
                @test GC.farkas_steric_screen(pi, q_host, box_pore, 0.5).feasible
                mask_i = Bool[j != i for j in 1:Ng]
                @test GC.farkas_steric_screen(pi, samp.q_guest, box_pore, 0.5; active_mask = mask_i).feasible
            end
        end
        @test total_steric_rej > 0
        @test loadings[1] < loadings[2] < loadings[3]
    end

    # ==========================================================================
    # 3. Widom Excess Chemical Potential
    # ==========================================================================
    @testset "3. Widom Test-Particle Insertion" begin
        box = T.OrthorhombicBox((6.0, 6.0, 6.0))
        q_host = zeros(Float64, 3, 0)
        q_guest = [1.0 3.0 5.0 2.0 4.0;
                   1.0 3.0 5.0 4.0 2.0;
                   1.0 3.0 5.0 3.0 3.0]
        sp_id = GC.GCMCSpecies(:id, 1.0, 0.0, 0.4, 1.0, 0.0)
        mu_id, se_id = GC.widom_excess_chemical_potential(
            q_host, q_guest, fill(1.0, 5), zeros(5), box, sp_id, 1.0; n_trials = 1000
        )
        @test abs(mu_id) < 1e-12
        @test se_id == 0.0

        # Dense purely repulsive / hard-core-like fluid at high T => positive excess μ
        rng = MersenneTwister(505)
        q_dense = rand(rng, 3, 60) .* 5.0
        box_d = T.OrthorhombicBox((5.0, 5.0, 5.0))
        sp_rep = GC.GCMCSpecies(:LJ, 1.0, 0.0, 0.4, 1.0, 1.0)
        mu_ex, se_ex = GC.widom_excess_chemical_potential(
            q_host, q_dense, fill(1.0, 60), fill(1.0, 60), box_d, sp_rep, 4.0;
            n_trials = 3000, rc = 2.5, mode = :lfs, rng = rng
        )
        @test isfinite(mu_ex) && mu_ex > 0.0
        @test isfinite(se_ex) && se_ex > 0.0
    end

    # ==========================================================================
    # 4. TernaryLatticeGas (TritWord64)
    # ==========================================================================
    @testset "4. TernaryLatticeGas (TritWord64)" begin
        # 8 x 8 x 2 = 128 sites => 2 TritWord64 words (tests multi-word indexing)
        temp = 1.2
        beta = 1.0 / temp
        epsA = -0.3; epsB = 0.2
        muA = 0.5; muB = 0.1
        lg = GC.TernaryLatticeGas(8, 8, 2; eps_A = epsA, eps_B = epsB,
                                  mu_A = muA, mu_B = muB, temperature = temp)
        @test length(lg.words) == 2

        res = GC.run_lattice_gcmc!(lg, 600; burnin = 150, rng = MersenneTwister(606))
        wA = exp(beta * (muA - epsA))
        wB = exp(beta * (muB - epsB))
        Z = 1.0 + wA + wB
        thetaA_exact = wA / Z
        thetaB_exact = wB / Z
        sel_exact = exp(-beta * (epsA - epsB))

        @test isapprox(res.theta_A, thetaA_exact; atol = 0.025)
        @test isapprox(res.theta_B, thetaB_exact; atol = 0.025)
        @test isapprox(res.selectivity, sel_exact; rtol = 0.15)
        for w in lg.words
            @test (w.P & w.N) == UInt64(0)
        end

        # Pore-blocked & interacting lattice gas
        blocked = [1, 2, 3, 65, 66]
        lg_int = GC.TernaryLatticeGas(8, 8, 2; J_AA = -0.25, J_BB = 0.1, J_AB = 0.4,
                                      eps_A = -0.5, eps_B = 0.0, mu_A = 0.2, mu_B = 0.2,
                                      temperature = 1.0, blocked_sites = blocked)
        res_int = GC.run_lattice_gcmc!(lg_int, 400; burnin = 100, rng = MersenneTwister(707))
        for b in blocked
            @test GC.get_site(lg_int, b) == Int8(0)
        end
        for w in lg_int.words
            @test (w.P & w.N) == UInt64(0)
        end
        @test res_int.theta_A > res_int.theta_B
        @test res_int.selectivity > 1.0
    end

    # ==========================================================================
    # 5. ConstantPHSampler (Henderson-Hasselbalch)
    # ==========================================================================
    @testset "5. ConstantPHSampler Henderson-Hasselbalch" begin
        pKa = 4.5
        temp = 300.0
        for pH in (pKa - 1.0, pKa, pKa + 1.0)
            site = GC.TitrationSite([1], pKa, [0.0], [-1.0], true)
            cph = GC.ConstantPHSampler([site], pH, temp, T.REDUCED)
            res = GC.run_constant_ph!(cph, nothing, 20_000; burnin = 2_000, rng = MersenneTwister(808))
            f_exact = 1.0 / (1.0 + 10.0^(pH - pKa))
            @test isapprox(res.protonation_fractions[1], f_exact; atol = 0.025)
        end

        # Coupled electrostatic stabilization of deprotonated state shifts fraction down
        site_elec = GC.TitrationSite([1], 4.5, [0.0], [-1.0], true)
        cph_elec = GC.ConstantPHSampler([site_elec], 4.5, 1.0, T.REDUCED)
        # U(q) = 1.5 * q[1] => deprotonating (q: 0 -> -1) lowers U by -1.5
        res_elec = GC.run_constant_ph!(cph_elec, q -> 1.5 * q[1], 10_000;
                                       burnin = 1_000, rng = MersenneTwister(809))
        @test res_elec.protonation_fractions[1] < 0.3
    end

    # ==========================================================================
    # 6. NEMD Factors: FD Forces, 9-Entry Box Derivatives, and Langevin Drift
    # ==========================================================================
    @testset "6. NEMD Factors, Box Derivatives & Langevin Drift" begin
        rng = MersenneTwister(909)
        N = 6
        q = [1.0 2.5 4.0 1.2 3.3 2.0;
             1.5 2.0 3.5 4.2 1.1 2.5;
             0.3 2.5 4.7 1.2 3.8 2.5]
        box_ortho = T.OrthorhombicBox((5.0, 5.5, 6.0))
        box_tri = T.TriclinicBox([5.5 0.8 0.4; 0.0 5.2 0.6; 0.0 0.0 6.0])

        factors_to_test = [
            NM.ExternalElectricFieldFactor((0.4, -0.25, 0.7), [1.0, -1.0, 0.5, -0.5, 1.2, -1.2]),
            NM.NanofluidicBodyForceFactor((0.15, -0.1, 0.35), [1, 3, 5], N),
            NM.WallRestraintFactor(:harmonic_flat_bottom, N; axis = 3, lo = 0.8, hi = 4.2, k = 3.5),
            NM.WallRestraintFactor(:lj93, N; axis = 3, lo = -0.2, hi = 5.2, epsilon = 0.8, sigma = 0.7, rc = 1.8),
            NM.WallRestraintFactor(:cylindrical_pore, N; axis = 3, center = (2.2, 2.2), r_max = 1.5, k = 4.0),
            NM.WallRestraintFactor(:cylindrical_pore, [1, 2, 4], N; axis = 3, center = (2.2, 2.2),
                                   r_max = 2.8, k = 0.0, epsilon = 0.6, sigma = 0.5, rc = 1.5),
        ]

        for fac in factors_to_test
            @test F.force_consistent(fac)
            @test RS.box_derivative_support(fac) === nothing

            for box in (box_ortho, box_tri)
                ctx = F.Context(box, T.REDUCED, N)
                F_an = zeros(Float64, 3, N)
                U = F.evaluate!(F_an, fac, q, ctx)
                @test isfinite(U)
                F_fd = fd_forces(fac, q, ctx)
                @test maximum(abs, F_an .- F_fd) < 1e-8

                G_rs = zeros(Float64, 3, 3)
                RS.accumulate_box_derivative!(G_rs, fac, q, ctx)
                G_fd = fd_box_derivative(fac, q, box, T.REDUCED)
                @test maximum(abs, G_rs .- G_fd) < 1e-7

                G_fac = zeros(Float64, 3, 3)
                F.accumulate_box_derivative!(G_fac, fac, q, ctx)
                @test isapprox(G_rs, G_fac; atol = 1e-14)
            end

            @test_throws ArgumentError F.accumulate_box_derivative!(
                zeros(3, 3), fac, q, F.Context(T.NoBox(), T.REDUCED, N)
            )
        end

        # C^1 continuity of :lj93 across rc
        w93 = NM.WallRestraintFactor(:lj93, 1; axis = 3, lo = 0.0, hi = Inf,
                                     epsilon = 1.0, sigma = 1.0, rc = 2.0)
        ctx1 = F.Context(T.OrthorhombicBox((5.0, 5.0, 5.0)), T.REDUCED, 1)
        F_near = zeros(3, 1)
        u_near = F.evaluate!(F_near, w93, reshape([0.0, 0.0, 2.0 - 1e-7], 3, 1), ctx1)
        @test abs(u_near) < 1e-12
        @test abs(F_near[3, 1]) < 1e-5

        # Langevin BAOAB + ExternalElectricFieldFactor steady-state drift velocity v_d = qE/(m*gamma)
        N_ions = 8
        masses = fill(2.0, N_ions)
        charges = [i <= 4 ? 1.0 : -1.0 for i in 1:N_ions]
        E_field = (0.0, 0.0, 2.0)
        efac = NM.ExternalElectricFieldFactor(E_field, charges)
        box_drift = T.OrthorhombicBox((10.0, 10.0, 10.0))
        hsys = D.HarmonicSystem([efac], masses, F.Context(box_drift, T.REDUCED, N_ions))
        st = D.State(zeros(3, N_ions), zeros(3, N_ions))
        D.forces!(hsys, st.q, st.F)
        st.force_valid = true

        dt = 0.02
        gamma = 1.5
        beta = 4.0
        rng_baoab = MersenneTwister(1010)
        for _ in 1:2000
            D.langevin_baoab_step!(st.q, st.p, st.F, hsys, dt, gamma, beta, rng_baoab)
        end
        v_hist = Matrix{Float64}[]
        for _ in 1:10000
            D.langevin_baoab_step!(st.q, st.p, st.F, hsys, dt, gamma, beta, rng_baoab)
            push!(v_hist, st.p ./ masses')
        end
        vd_pos = mean(v[3, i] for v in v_hist for i in 1:4)
        vd_neg = mean(v[3, i] for v in v_hist for i in 5:8)
        vd_exact = 1.0 * E_field[3] / (2.0 * gamma)
        @test isapprox(vd_pos, vd_exact; rtol = 0.04)
        @test isapprox(vd_neg, -vd_exact; rtol = 0.04)

        sigma_est = NM.estimate_ionic_conductivity(v_hist, charges, E_field, T.box_volume(box_drift))
        sigma_exact = sum(q^2 / (m * gamma) for (q, m) in zip(charges, masses)) / T.box_volume(box_drift)
        @test isapprox(sigma_est, sigma_exact; rtol = 0.04)

        # Mass flux helper exact check
        q0 = zeros(3, 2); q1 = [0.0 0.0; 0.0 0.0; 3.0 1.5]
        @test isapprox(NM.estimate_mass_flux(q0, q1, 2.0, 4.0, [2.0, 4.0], 3),
                       (2.0 * 3.0 + 4.0 * 1.5) / (2.0 * 4.0); rtol = 1e-14)
    end

    # ==========================================================================
    # 7. Müller-Plathe Reverse-NEMD (Shear & Thermal)
    # ==========================================================================
    @testset "7. Müller-Plathe Reverse-NEMD Conservation & Gradient" begin
        box = T.OrthorhombicBox((8.0, 8.0, 10.0))
        n_slabs = 10
        rng = MersenneTwister(1111)

        # Equal and unequal masses conservation test for both shear & thermal swaps
        for masses in (fill(1.0, 80), 0.5 .+ 2.5 .* rand(rng, 80))
            q = zeros(3, 80)
            q[1, :] .= rand(rng, 80) .* 8.0
            q[2, :] .= rand(rng, 80) .* 8.0
            # Populate all slabs uniformly along z
            for i in 1:80
                q[3, i] = ((i - 1) % n_slabs + 0.5) * (10.0 / n_slabs)
            end
            p = randn(rng, 3, 80) .* sqrt.(masses)'
            st_shear = D.State(copy(q), copy(p))
            P0 = vec(sum(st_shear.p; dims = 2))
            K0 = sum(0.5 * sum(st_shear.p[:, i].^2) / masses[i] for i in 1:80)

            dp_tot = 0.0
            for _ in 1:30
                dp_tot += NM.muller_plathe_shear_step!(st_shear, masses, box;
                                                       n_slabs = n_slabs, flow_axis = 1, grad_axis = 3)
            end
            P1 = vec(sum(st_shear.p; dims = 2))
            K1 = sum(0.5 * sum(st_shear.p[:, i].^2) / masses[i] for i in 1:80)
            @test dp_tot > 0.0
            @test norm(P1 .- P0) < 1e-13
            @test abs(K1 - K0) < 1e-13

            # Verify expected shear direction: slab 1 vx < mid_slab vx
            slab1_vx = mean(st_shear.p[1, i] / masses[i] for i in 1:80 if floor(Int, st_shear.q[3, i]) == 0)
            mid_vx = mean(st_shear.p[1, i] / masses[i] for i in 1:80 if floor(Int, st_shear.q[3, i]) == 5)
            @test mid_vx > slab1_vx
            eta = NM.estimate_shear_viscosity(dp_tot, 10.0, 64.0, (mid_vx - slab1_vx) / 5.0)
            @test isfinite(eta) && eta > 0.0

            # Thermal swap conservation & temperature gradient direction
            st_therm = D.State(copy(q), copy(p))
            P0t = vec(sum(st_therm.p; dims = 2))
            K0t = sum(0.5 * sum(st_therm.p[:, i].^2) / masses[i] for i in 1:80)
            dK_tot = 0.0
            for _ in 1:30
                dK_tot += NM.muller_plathe_thermal_step!(st_therm, masses, box;
                                                         n_slabs = n_slabs, grad_axis = 3)
            end
            P1t = vec(sum(st_therm.p; dims = 2))
            K1t = sum(0.5 * sum(st_therm.p[:, i].^2) / masses[i] for i in 1:80)
            @test dK_tot > 0.0
            @test norm(P1t .- P0t) < 1e-13
            @test abs(K1t - K0t) < 1e-13

            slab1_ke = mean(0.5 * sum(st_therm.p[:, i].^2) / masses[i] for i in 1:80 if floor(Int, st_therm.q[3, i]) == 0)
            mid_ke = mean(0.5 * sum(st_therm.p[:, i].^2) / masses[i] for i in 1:80 if floor(Int, st_therm.q[3, i]) == 5)
            @test mid_ke > slab1_ke
            kappa = NM.estimate_thermal_conductivity(dK_tot, 10.0, 64.0, (mid_ke - slab1_ke) / 5.0)
            @test isfinite(kappa) && kappa > 0.0
        end
    end
end
