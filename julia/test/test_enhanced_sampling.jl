using Test
using LinearAlgebra
using Random
using Statistics
using Printf

using FoundationCore
if !isdefined(FoundationCore, :EnhancedSampling)
    @eval FoundationCore begin
        include(joinpath(@__DIR__, "..", "src", "EnhancedSampling.jl"))
    end
end
const ES = FoundationCore.EnhancedSampling

using .ES
using FoundationCore.Types
using FoundationCore.Factors
using FoundationCore.Dynamics
import FoundationCore.ResolvedSystem as RS
using FoundationCore.SystemContract

# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

"1D/3D double-well potential factor: U(q) = 4 (x^2 - 1)^2 + 0.5 * k_perp * (y^2 + z^2) on atom 1."
struct DoubleWellTestFactor <: Factors.AbstractFactor
    k_perp::Float64
end
Factors.factor_support_size(::DoubleWellTestFactor) = 1
Factors.read_support(::DoubleWellTestFactor) = (1,)
Factors.is_local(::DoubleWellTestFactor) = true
Factors.force_consistent(::DoubleWellTestFactor) = true

function Factors.evaluate!(F::AbstractMatrix{Float64}, f::DoubleWellTestFactor,
                           q::AbstractMatrix{Float64}, ::Factors.Context)
    x = q[1, 1]; y = q[2, 1]; z = q[3, 1]
    xm = x * x - 1.0
    u = 4.0 * xm * xm + 0.5 * f.k_perp * (y * y + z * z)
    @inbounds begin
        F[1, 1] -= 16.0 * x * xm
        F[2, 1] -= f.k_perp * y
        F[3, 1] -= f.k_perp * z
    end
    return u
end

"1D/3D periodic cosine potential factor along x (period 2π): U = k_phi*(1 + cos(2*x)) + 0.5*k_perp*(y^2+z^2)."
struct PeriodicCosineTestFactor <: Factors.AbstractFactor
    k_phi::Float64
    k_perp::Float64
end
Factors.factor_support_size(::PeriodicCosineTestFactor) = 1
Factors.read_support(::PeriodicCosineTestFactor) = (1,)
Factors.is_local(::PeriodicCosineTestFactor) = true
Factors.force_consistent(::PeriodicCosineTestFactor) = true

function Factors.evaluate!(F::AbstractMatrix{Float64}, f::PeriodicCosineTestFactor,
                           q::AbstractMatrix{Float64}, ::Factors.Context)
    x = q[1, 1]; y = q[2, 1]; z = q[3, 1]
    u = f.k_phi * (1.0 + cos(2.0 * x)) + 0.5 * f.k_perp * (y * y + z * z)
    @inbounds begin
        F[1, 1] -= -2.0 * f.k_phi * sin(2.0 * x)
        F[2, 1] -= f.k_perp * y
        F[3, 1] -= f.k_perp * z
    end
    return u
end

function _box_mat(box::Types.OrthorhombicBox)
    return [box.L[1] 0.0 0.0; 0.0 box.L[2] 0.0; 0.0 0.0 box.L[3]]
end
_box_mat(box::Types.TriclinicBox) = copy(box.H)

function _fd_cv_box_derivative(cv::AbstractCV, dUds::Float64,
                                q::Matrix{Float64}, box::Types.AbstractBox;
                                eps::Float64 = 1e-6)
    H = _box_mat(box)
    S = inv(H) * q
    G_fd = zeros(3, 3)
    g_tmp = zeros(3, size(q, 2))
    for a in 1:3, b in 1:3
        Hp = copy(H); Hp[a, b] += eps
        Hm = copy(H); Hm[a, b] -= eps
        sp = cv_value_and_grad!(g_tmp, cv, Hp * S, Types.TriclinicBox(Hp))
        sm = cv_value_and_grad!(g_tmp, cv, Hm * S, Types.TriclinicBox(Hm))
        G_fd[a, b] = dUds * cv_diff(cv, sp, sm) / (2.0 * eps)
    end
    return G_fd
end

function _fd_factor_forces(f::Factors.AbstractFactor, q::Matrix{Float64},
                           box::Types.AbstractBox; eps::Float64 = 1e-6)
    N = size(q, 2)
    ctx = Factors.Context(box, Types.REDUCED, N)
    F_fd = zeros(3, N)
    q_work = copy(q)
    F_tmp = zeros(3, N)
    for j in 1:N, d in 1:3
        orig = q_work[d, j]
        q_work[d, j] = orig + eps
        fill!(F_tmp, 0.0)
        up = Factors.evaluate!(F_tmp, f, q_work, ctx)
        q_work[d, j] = orig - eps
        fill!(F_tmp, 0.0)
        um = Factors.evaluate!(F_tmp, f, q_work, ctx)
        q_work[d, j] = orig
        F_fd[d, j] = -(up - um) / (2.0 * eps)
    end
    return F_fd
end

function _fd_factor_box_derivative(rs::RS.ResolvedSystem, q::Matrix{Float64};
                                    eps::Float64 = 1e-5)
    H = _box_mat(rs.box)
    S = inv(H) * q
    G_fd = zeros(3, 3)
    for a in 1:3, b in 1:3
        Hp = copy(H); Hp[a, b] += eps
        Hm = copy(H); Hm[a, b] -= eps
        rsp = RS.restart_system(rs; box = Types.TriclinicBox(Hp))
        rsm = RS.restart_system(rs; box = Types.TriclinicBox(Hm))
        G_fd[a, b] = (SystemContract.energy(rsp, Hp * S) -
                      SystemContract.energy(rsm, Hm * S)) / (2.0 * eps)
    end
    return G_fd
end

# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

@testset "EnhancedSampling Suite" begin

    # =========================================================================
    # 1. Collective Variables: Analytic Gradients & Box Derivatives
    # =========================================================================
    @testset "1. Collective Variables (5 CVs across NoBox, Orthorhombic, Triclinic)" begin
        nobox = Types.NoBox()
        obox = Types.OrthorhombicBox((6.0, 6.5, 7.0))
        H_tri = [6.2 0.7 0.3;
                 0.0 6.5 0.5;
                 0.0 0.0 6.8]
        tbox = Types.TriclinicBox(H_tri)
        boxes = (nobox, obox, tbox)

        # 6-atom 3D geometry including wrapped periodic images across box walls
        q = [0.2  1.3  2.1  2.8  5.9  0.4;
             0.3  1.1  0.5  1.4  6.1  0.6;
             0.1  0.7  1.5  1.1  0.2  6.5]
        masses = [12.0, 14.0, 16.0, 12.0, 1.0, 16.0]

        dist_cv = DistanceCV(1, 5)  # spans periodic boundary in obox/tbox
        coord_cv = CoordinationCV([1, 2], [3, 4, 5], 1.8; n = 6, m = 12)
        dihed_cv = DihedralCV(1, 2, 3, 4)
        com_cv = ComDistanceCV([1, 5], [3, 4, 6], masses)
        custom_cv = CustomCV(
            (pos, box) -> begin
                d = Types.unwrap_displacement(box, pos, 2, 1)
                0.5 * Types.norm2(d)
            end,
            (grad, pos, box) -> begin
                d = Types.unwrap_displacement(box, pos, 2, 1)
                grad[1, 2] = d.x; grad[2, 2] = d.y; grad[3, 2] = d.z
                grad[1, 1] = -d.x; grad[2, 1] = -d.y; grad[3, 1] = -d.z
            end,
            [1, 2];
            box_deriv_fn = (G, dUds, pos, box) -> begin
                d = Types.unwrap_displacement(box, pos, 2, 1)
                H = _box_mat(box)
                w = inv(H) * [d.x, d.y, d.z]
                G .+= dUds .* ([d.x, d.y, d.z] * transpose(w))
            end,
            periodic = false,
        )

        @test !is_periodic(dist_cv) && cv_period(dist_cv) == 0.0
        @test !is_periodic(coord_cv)
        @test is_periodic(dihed_cv) && isapprox(cv_period(dihed_cv), 2π; atol = 1e-15)
        @test !is_periodic(com_cv)
        @test isapprox(cv_diff(dihed_cv, 3.1, -3.1), 6.2 - 2π; atol = 1e-12)

        for box in boxes
            for cv in (dist_cv, coord_cv, dihed_cv, com_cv, custom_cv)
                res = verify_cv_gradient(cv, q, box; h = 1e-6, tol = 1e-8)
                @test res.passed
                @test res.max_error < 1e-8
            end

            if !(box isa Types.NoBox)
                for cv in (dist_cv, coord_cv, dihed_cv, com_cv, custom_cv)
                    G = zeros(3, 3)
                    dUds = 1.75
                    cv_box_derivative!(G, cv, dUds, q, box)
                    G_fd = _fd_cv_box_derivative(cv, dUds, q, box; eps = 1e-6)
                    err = maximum(abs, G .- G_fd)
                    @test err < 1e-8
                end
            else
                G = zeros(3, 3)
                @test_throws Types.UnsupportedGeometry cv_box_derivative!(G, dist_cv, 1.0, q, box)
            end
        end

        # Removable singularity at r == r0 for CoordinationCV
        r0 = 2.0
        coord_sing = CoordinationCV([1], [2], r0; n = 6, m = 12)
        g_sing = zeros(3, 2)
        for delta in (-1e-5, -1e-9, 0.0, 1e-9, 1e-5)
            q_sing = [0.0 (r0 * (1.0 + delta)); 0.0 0.0; 0.0 0.0]
            val = cv_value_and_grad!(g_sing, coord_sing, q_sing, nobox)
            @test isfinite(val) && all(isfinite, g_sing)
            # Exact value at x=1 is n/m = 6/12 = 0.5; derivative wrt r is n*(n-m)/(2*m*r0) = -1.5/r0
            if delta == 0.0
                @test isapprox(val, 0.5; atol = 1e-14)
                @test isapprox(g_sing[1, 2], -1.5 / r0; atol = 1e-14)
                @test isapprox(g_sing[1, 1], 1.5 / r0; atol = 1e-14)
            end
            res_sing = verify_cv_gradient(coord_sing, q_sing, nobox; h = 1e-6, tol = 1e-9)
            @test res_sing.passed
        end

        # Verify CustomCV two-arg grad_fn support and verify_cv_gradient near dihedral branch cut ±π
        q_branch = [0.0  1.0  1.0  0.0;
                    1.0  0.0  0.0 -1.0;
                    0.0  0.0  1.5  1.5 + 1e-8]
        res_branch = verify_cv_gradient(dihed_cv, q_branch, nobox; h = 1e-6, tol = 1e-8)
        @test res_branch.passed
    end

    # =========================================================================
    # 2. Biased Factors: FD Forces, 9-Entry Box Derivatives & 200-Step NVE
    # =========================================================================
    @testset "2. UmbrellaFactor & MetadynamicsBias (FD, Box Deriv, 200-step NVE)" begin
        obox = Types.OrthorhombicBox((6.0, 6.5, 7.0))
        H_tri = [6.2 0.7 0.3;
                 0.0 6.5 0.5;
                 0.0 0.0 6.8]
        tbox = Types.TriclinicBox(H_tri)

        q = [0.4  1.8  2.6  3.3;
             0.5  1.4  0.8  1.7;
             0.2  0.9  1.7  1.2]
        N = size(q, 2)
        masses = [12.0, 14.0, 16.0, 12.0]

        for cv in (DistanceCV(1, 2),
                   CoordinationCV([1, 2], [3, 4], 2.0; n = 6, m = 12),
                   DihedralCV(1, 2, 3, 4),
                   ComDistanceCV([1, 2], [3, 4], masses))

            s_init = cv_value_and_grad!(zeros(3, N), cv, q, tbox)
            umb = UmbrellaFactor(cv, 25.0, s_init - 0.35, N)
            @test Factors.factor_support_size(umb) == N
            @test Factors.read_support(umb) == Tuple(1:N)
            @test !Factors.is_local(umb)
            @test Factors.force_consistent(umb)

            meta = MetadynamicsBias(cv, N; height = 0.8, sigma = 0.35,
                                    bias_factor = 6.0, temperature = 1.0, kB = 1.0)
            deposit_hill!(meta, s_init - 0.15)
            deposit_hill!(meta, s_init + 0.10)
            deposit_hill!(meta, s_init + 0.28)
            @test length(meta.centers) == 3
            @test meta.heights[2] < meta.heights[1]  # well-tempered damping

            for box in (obox, tbox)
                for fac in (umb, meta)
                    ctx = Factors.Context(box, Types.REDUCED, N)
                    F_ana = zeros(3, N)
                    Factors.evaluate!(F_ana, fac, q, ctx)
                    F_fd = _fd_factor_forces(fac, q, box; eps = 1e-6)
                    @test maximum(abs, F_ana .- F_fd) < 1e-8

                    rs = RS.ResolvedSystem(
                        masses = masses, charges = zeros(N), box = box,
                        factors = Factors.AbstractFactor[fac],
                        component_names = [:bias],
                        declared = [:energy, :cartesian_forces, :component_energies, :box_derivative],
                    )
                    @test SystemContract.supports_box_derivative(fac, box)
                    G_ana = SystemContract.box_derivative(rs, q)
                    G_fd = _fd_factor_box_derivative(rs, q; eps = 1e-5)
                    @test maximum(abs, G_ana .- G_fd) < 1e-7

                    # Fractional derivative check via SystemContract
                    hmat = _box_mat(box)
                    dUds_sc = SystemContract.fractional_derivative(rs, q)
                    dUds_ana = -(transpose(hmat) * F_ana)
                    @test maximum(abs, dUds_sc .- dUds_ana) < 1e-8
                end
            end
        end

        # CustomCV without box_deriv_fn refuses SystemContract.box_derivative cleanly
        cv_nobd = CustomCV((pos, box) -> pos[1, 1], (g, pos, box) -> (g[1, 1] = 1.0), [1])
        umb_nobd = UmbrellaFactor(cv_nobd, 10.0, 0.0, N)
        rs_nobd = RS.ResolvedSystem(
            masses = masses, charges = zeros(N), box = obox,
            factors = Factors.AbstractFactor[umb_nobd],
            component_names = [:bias],
        )
        @test !SystemContract.supports_box_derivative(umb_nobd, obox)
        @test_throws SystemContract.MissingBoxDerivative SystemContract.box_derivative(rs_nobd, q)

        # 200-step NVE energy conservation for UmbrellaFactor and fixed-hill MetadynamicsBias
        bond12 = Factors.HarmonicBond(1, 2, 10.0, 1.5)
        bond23 = Factors.HarmonicBond(2, 3, 10.0, 1.4)
        bond34 = Factors.HarmonicBond(3, 4, 10.0, 1.4)

        umb_nve = UmbrellaFactor(DistanceCV(1, 4), 8.0, 2.5, N)
        meta_nve = MetadynamicsBias(DihedralCV(1, 2, 3, 4), N;
                                    height = 0.5, sigma = 0.4, bias_factor = 5.0)
        deposit_hill!(meta_nve, 0.2)
        deposit_hill!(meta_nve, -0.3)

        p0 = [0.25 -0.15  0.10 -0.20;
              -0.10 0.20 -0.15  0.05;
              0.15 -0.05 -0.20  0.10]

        for (label, bias_fac) in (("Umbrella NVE", umb_nve), ("Metadynamics NVE", meta_nve))
            sys = Dynamics.HarmonicSystem(
                Factors.AbstractFactor[bond12, bond23, bond34, bias_fac],
                masses,
                Factors.Context(tbox, Types.REDUCED, N),
            )
            st = Dynamics.State(copy(q), copy(p0))
            E0 = Dynamics.total_energy(sys, st.q, st.p)
            recs, es = Dynamics.run_trajectory(st, sys, 1e-3, 200)
            max_rel_drift = maximum(abs.(es.compensated .- E0)) / abs(E0)
            @printf("  %s 200-step max|dE/E0| = %.3e\n", label, max_rel_drift)
            @test max_rel_drift < 1e-6
        end
    end

    # =========================================================================
    # 3. Double-Well PMF: 9-Window Umbrella + WHAM & Well-Tempered Metadynamics
    # =========================================================================
    @testset "3. Double-Well 1D/3D PMF (Umbrella+WHAM & Well-Tempered MetaD)" begin
        dw_factor = DoubleWellTestFactor(20.0)
        x_cv = CustomCV(
            (pos, box) -> pos[1, 1],
            (grad, pos, box) -> (grad[1, 1] = 1.0),
            [1],
        )
        exact_pmf(x) = 4.0 * (x^2 - 1.0)^2

        # 3a. 9-window Umbrella Sampling + Langevin BAOAB + unbias_umbrella_wham
        centers = collect(range(-1.2, 1.2; length = 9))
        ks = fill(28.0, 9)
        beta = 1.0
        dt = 0.01
        gamma = 2.0
        samples = Vector{Vector{Float64}}(undef, 9)
        rng = MersenneTwister(202603)

        for (w_idx, s0) in enumerate(centers)
            umb = UmbrellaFactor(x_cv, ks[w_idx], s0, 1)
            sys = Dynamics.HarmonicSystem(
                Factors.AbstractFactor[dw_factor, umb],
                [1.0],
                Factors.Context(Types.NoBox(), Types.REDUCED, 1),
            )
            st = Dynamics.State([s0; 0.0; 0.0][:, :], zeros(3, 1))
            Dynamics.forces!(sys, st.q, st.F)
            # Equilibrate
            for _ in 1:2000
                Dynamics.langevin_baoab_step!(st.q, st.p, st.F, sys, dt, gamma, beta, rng)
            end
            # Production sampling
            win_s = Float64[]
            sizehint!(win_s, 3000)
            for step in 1:15000
                Dynamics.langevin_baoab_step!(st.q, st.p, st.F, sys, dt, gamma, beta, rng)
                if step % 5 == 0
                    push!(win_s, st.q[1, 1])
                end
            end
            samples[w_idx] = win_s
        end

        s_grid = collect(range(-1.3, 1.3; length = 53))
        wham = unbias_umbrella_wham(centers, ks, samples, beta; s_grid = s_grid, tol = 1e-10)
        @test wham.converged
        @test isapprox(sum(wham.weights), 1.0; atol = 1e-10)
        @test wham.log_Z[1] == 0.0
        @test isapprox(minimum(wham.pmf), 0.0; atol = 1e-12)

        pmf_exact = exact_pmf.(s_grid)
        pmf_exact .-= minimum(pmf_exact)
        rmse = sqrt(mean((wham.pmf .- pmf_exact) .^ 2))
        idx_zero = argmin(abs.(s_grid))
        barrier_wham = wham.pmf[idx_zero] - minimum(wham.pmf)
        barrier_err_wham = abs(barrier_wham - 4.0)
        @printf("  WHAM double-well RMSE=%.4f, barrier=%.4f (err=%.4f, iters=%d)\n",
                rmse, barrier_wham, barrier_err_wham, wham.iterations)
        @test rmse < 0.25
        @test barrier_err_wham < 0.35

        # 3b. Periodic WHAM check on U(x) = 1.5 * (1 + cos(2x)) over [-pi, pi)
        p_fac = PeriodicCosineTestFactor(1.5, 20.0)
        p_centers = collect(range(-π + π / 12, π - π / 12; length = 12))
        p_ks = fill(20.0, 12)
        p_samples = Vector{Vector{Float64}}(undef, 12)
        rng_p = MersenneTwister(9911)
        p_cv = CustomCV((pos, box) -> pos[1, 1], (grad, pos, box) -> (grad[1, 1] = 1.0), [1];
                        periodic = true, period = 2π)
        for (idx, s0) in enumerate(p_centers)
            umb = UmbrellaFactor(p_cv, p_ks[idx], s0, 1)
            sys = Dynamics.HarmonicSystem(
                Factors.AbstractFactor[p_fac, umb],
                [1.0],
                Factors.Context(Types.NoBox(), Types.REDUCED, 1),
            )
            st = Dynamics.State([s0; 0.0; 0.0][:, :], zeros(3, 1))
            Dynamics.forces!(sys, st.q, st.F)
            for _ in 1:1500
                Dynamics.langevin_baoab_step!(st.q, st.p, st.F, sys, dt, gamma, beta, rng_p)
            end
            ws = Float64[]
            for step in 1:12000
                Dynamics.langevin_baoab_step!(st.q, st.p, st.F, sys, dt, gamma, beta, rng_p)
                if step % 5 == 0
                    push!(ws, mod2pi(st.q[1, 1] + π) - π)
                end
            end
            p_samples[idx] = ws
        end
        p_grid = collect(range(-π, π - 0.1; length = 48))
        wham_p = unbias_umbrella_wham(p_centers, p_ks, p_samples, beta;
                                      s_grid = p_grid, periodic = true, period = 2π)
        @test wham_p.converged
        p_exact = [1.5 * (1.0 + cos(2.0 * x)) for x in p_grid]
        p_exact .-= minimum(p_exact)
        p_rmse = sqrt(mean((wham_p.pmf .- p_exact) .^ 2))
        @printf("  Periodic WHAM RMSE=%.4f (iters=%d)\n", p_rmse, wham_p.iterations)
        @test p_rmse < 0.25

        # 3c. Well-Tempered Metadynamics drives repeated barrier crossings & reconstructs PMF
        meta_dw = MetadynamicsBias(x_cv, 1;
                                   height = 0.25, sigma = 0.12,
                                   bias_factor = 6.0, temperature = 1.0, kB = 1.0)
        sys_meta = Dynamics.HarmonicSystem(
            Factors.AbstractFactor[dw_factor, meta_dw],
            [1.0],
            Factors.Context(Types.NoBox(), Types.REDUCED, 1),
        )
        st_meta = Dynamics.State([-1.0; 0.0; 0.0][:, :], zeros(3, 1))
        Dynamics.forces!(sys_meta, st_meta.q, st_meta.F)
        rng_m = MersenneTwister(4242)

        n_steps_meta = 120_000
        deposit_every = 150
        crossings = 0
        prev_sign = sign(st_meta.q[1, 1])

        for step in 1:n_steps_meta
            Dynamics.langevin_baoab_step!(st_meta.q, st_meta.p, st_meta.F,
                                          sys_meta, 0.01, 2.0, 1.0, rng_m)
            cur_sign = sign(st_meta.q[1, 1])
            if cur_sign != 0.0 && cur_sign != prev_sign
                crossings += 1
                prev_sign = cur_sign
            end
            if step % deposit_every == 0
                deposit_hill!(meta_dw, st_meta.q, Types.NoBox())
                Dynamics.forces!(sys_meta, st_meta.q, st_meta.F)
            end
        end

        F_meta = reconstruct_free_energy(meta_dw, s_grid)
        @test isapprox(minimum(F_meta), 0.0; atol = 1e-12)
        barrier_meta = F_meta[idx_zero] - minimum(F_meta)
        barrier_err_meta = abs(barrier_meta - 4.0)
        # Symmetry check between left well (-1) and right well (+1)
        idx_left = argmin(abs.(s_grid .+ 1.0))
        idx_right = argmin(abs.(s_grid .- 1.0))
        well_asym = abs(F_meta[idx_left] - F_meta[idx_right])
        @printf("  WT-MetaD crossings=%d, barrier=%.4f (err=%.4f), well_asym=%.4f\n",
                crossings, barrier_meta, barrier_err_meta, well_asym)
        @test crossings >= 10
        @test barrier_err_meta < 0.6
        @test well_asym < 0.6
    end

    # =========================================================================
    # 4. REMD Exchange Criterion & Kinetic Temperature Scaling
    # =========================================================================
    @testset "4. REMD Exchange Acceptance Ratio & Temperature Scaling" begin
        rng = MersenneTwister(777)

        # Analytical acceptance check for delta < 0: P_acc = exp((beta_a - beta_b) * (U_a - U_b))
        beta_a = 2.0   # T_a = 0.5
        beta_b = 1.0   # T_b = 1.0
        U_a = 1.0
        U_b = 2.2      # delta = (2.0 - 1.0) * (1.0 - 2.2) = -1.2 => P_acc = exp(-1.2)
        expected_p = exp((beta_a - beta_b) * (U_a - U_b))

        n_trials = 20_000
        n_acc = 0
        for _ in 1:n_trials
            sa = Dynamics.State([1.0; 0.0; 0.0][:, :], [0.5; 0.0; 0.0][:, :])
            sb = Dynamics.State([2.0; 0.0; 0.0][:, :], [1.0; 0.0; 0.0][:, :])
            sa.force_valid = true
            sb.force_valid = true
            if remd_exchange_attempt!(sa, sb, U_a, U_b, beta_a, beta_b, rng)
                n_acc += 1
            end
        end
        emp_p = n_acc / n_trials
        se = sqrt(expected_p * (1.0 - expected_p) / n_trials)
        @printf("  REMD acceptance: emp=%.4f vs exact=%.4f (SE=%.4f)\n",
                emp_p, expected_p, se)
        @test abs(emp_p - expected_p) < 4.0 * se

        # Always-accept case: cold replica has higher energy (delta > 0)
        sa = Dynamics.State([1.0 2.0; 0.0 0.0; 0.0 0.0], [0.3 -0.4; 0.1 0.2; -0.1 0.5])
        sb = Dynamics.State([5.0 6.0; 1.0 1.0; 2.0 2.0], [0.8 0.9; -0.3 0.4; 0.2 -0.6])
        sa.force_valid = true
        sb.force_valid = true
        q_a0 = copy(sa.q); q_b0 = copy(sb.q)
        p_a0 = copy(sa.p); p_b0 = copy(sb.p)

        @test remd_exchange_attempt!(sa, sb, 5.0, 1.0, 2.0, 1.0, rng) == true
        @test sa.q == q_b0 && sb.q == q_a0
        @test !sa.force_valid && !sb.force_valid
        @test isapprox(sa.p, sqrt(1.0 / 2.0) * p_b0; atol = 1e-14)
        @test isapprox(sb.p, sqrt(2.0 / 1.0) * p_a0; atol = 1e-14)

        # Ensemble kinetic temperature preservation across swaps
        n_samples = 5000
        masses = [2.0, 3.0]
        T_a_samples = Float64[]
        T_b_samples = Float64[]
        for _ in 1:n_samples
            pa = hcat(sqrt(masses[1] / beta_a) .* randn(rng, 3),
                      sqrt(masses[2] / beta_a) .* randn(rng, 3))
            pb = hcat(sqrt(masses[1] / beta_b) .* randn(rng, 3),
                      sqrt(masses[2] / beta_b) .* randn(rng, 3))
            st_a = Dynamics.State(zeros(3, 2), pa)
            st_b = Dynamics.State(ones(3, 2), pb)
            # Force accept (delta >= 0)
            remd_exchange_attempt!(st_a, st_b, 2.0, 1.0, beta_a, beta_b, rng)
            dof = 6
            K_a = 0.5 * sum(st_a.p[d, j]^2 / masses[j] for d in 1:3, j in 1:2)
            K_b = 0.5 * sum(st_b.p[d, j]^2 / masses[j] for d in 1:3, j in 1:2)
            push!(T_a_samples, 2.0 * K_a / dof)
            push!(T_b_samples, 2.0 * K_b / dof)
        end
        @test isapprox(mean(T_a_samples), 1.0 / beta_a; rtol = 0.05)
        @test isapprox(mean(T_b_samples), 1.0 / beta_b; rtol = 0.05)
    end
end
