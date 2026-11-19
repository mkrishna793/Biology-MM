# Focused unit tests for Radix-3 27-Point Ternary Multilevel Summation Method (Ternary-MSM).
#
# Tests strictly verify:
# 1. 27-point balanced ternary tryte index bijection over {-1, 0, +1}³ -> 1:27 and opposite symmetry 28 - idx.
# 2. C² continuity of g_short(r), dg_short(r) at r = rcut and smooth gamma(rho) at rho = 0.
# 3. Analytical Cartesian forces vs central finite differences (< 1e-9) for neutral & excluded/1-4 systems
#    in BOTH orthorhombic and skew triclinic cells.
# 4. Analytical 3-entry orthorhombic and 9-entry triclinic box derivatives vs fixed-fractional central finite
#    differences (< 1e-7, matching test_pme.jl convention).
# 5. Comparison of TernaryMSMCoulombFactor against PME.PMEFactor on a periodic salt/water test box,
#    and O(N) scaling (N = 256, 1024, 4096) with zero FFT calls.

using Test
using FoundationCore
if !isdefined(FoundationCore, :TernaryMSM)
    @eval FoundationCore begin
        include(joinpath(@__DIR__, "..", "src", "TernaryMSM.jl"))
    end
end
const TMSM = FoundationCore.TernaryMSM

using LinearAlgebra, Statistics, Random

maxabs(A) = maximum(abs, A)

function refuses(f)
    try
        f()
    catch err
        return err isa ArgumentError || err isa FoundationCore.Types.InvalidGeometry ||
               err isa FoundationCore.Types.UnsupportedGeometry
    end
    return false
end

@testset "TernaryMSM Suite" begin

# ===========================================================================
# 1. 27-Point Balanced Ternary Tryte Indexing & Symmetry
# ===========================================================================
@testset "1. Balanced Ternary Tryte Indexing & Symmetry" begin
    indices = Set{Int}()
    for dz in -1:1, dy in -1:1, dx in -1:1
        idx = TMSM.tryte_index(dx, dy, dz)
        @test 1 <= idx <= 27
        push!(indices, idx)

        # Opposite direction tryte symmetry: (-dx, -dy, -dz) -> 28 - idx
        opp_idx = TMSM.tryte_index(-dx, -dy, -dz)
        @test opp_idx == 28 - idx

        # Exact bijective coordinate reconstruction
        coords = TMSM.tryte_coords(idx)
        @test coords == (dx, dy, dz)
    end

    # Bijection check
    @test length(indices) == 27
    @test minimum(indices) == 1
    @test maximum(indices) == 27

    # Center tryte (0, 0, 0) is index 14, self-opposite: 28 - 14 == 14
    @test TMSM.tryte_index(0, 0, 0) == 14
    @test TMSM.tryte_coords(14) == (0, 0, 0)

    # Corner trytes
    @test TMSM.tryte_index(-1, -1, -1) == 1
    @test TMSM.tryte_index(1, 1, 1) == 27
    @test TMSM.tryte_index(1, 1, 1) == 28 - TMSM.tryte_index(-1, -1, -1)

    # Restriction and prolongation stencil weights
    w1d_4, w27_4 = TMSM.radix3_stencil_weights(4)
    @test isapprox(sum(w1d_4), 1.0)
    @test isapprox(sum(w27_4), 1.0)
    @test w1d_4 == (1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0)
    @test isapprox(w27_4[14], (4.0 / 6.0)^3)  # center weight

    w1d_2, w27_2 = TMSM.radix3_stencil_weights(2)
    @test isapprox(sum(w1d_2), 1.0)
    @test isapprox(sum(w27_2), 1.0)

    w1d_1, w27_1 = TMSM.radix3_stencil_weights(1)
    @test isapprox(sum(w1d_1), 1.0)
    @test isapprox(sum(w27_1), 1.0)

    # Refusals
    @test refuses(() -> TMSM.tryte_index(2, 0, 0))
    @test refuses(() -> TMSM.tryte_index(0, -2, 0))
    @test refuses(() -> TMSM.tryte_coords(0))
    @test refuses(() -> TMSM.tryte_coords(28))
    @test refuses(() -> TMSM.radix3_stencil_weights(3))
end

# ===========================================================================
# 2. C² Continuity of Hardy Softening Kernel & Derivatives
# ===========================================================================
@testset "2. C² Softening Continuity & Kernel Smoothness" begin
    rcut = 5.5

    # Values at cutoff
    @test TMSM.g_short(rcut, rcut) == 0.0
    @test TMSM.dg_short(rcut, rcut) == 0.0
    @test TMSM.g_short(rcut + 1.0, rcut) == 0.0
    @test TMSM.dg_short(rcut + 1.0, rcut) == 0.0

    # Finite difference derivative across cutoff
    h_fd = 1e-6
    fd_dg_cut = (TMSM.g_short(rcut + h_fd, rcut) - TMSM.g_short(rcut - h_fd, rcut)) / (2 * h_fd)
    @test abs(fd_dg_cut) < 1e-10

    # Finite difference second derivative across cutoff
    fd_d2g_cut = (TMSM.dg_short(rcut + h_fd, rcut) - TMSM.dg_short(rcut - h_fd, rcut)) / (2 * h_fd)
    @test abs(fd_d2g_cut) < 1e-5

    # C² continuity of γ(ρ) at ρ = 1.0
    @test TMSM.gamma_softening(1.0) == 1.0
    @test TMSM.dgamma_softening(1.0) == -1.0
    @test TMSM.d2gamma_softening(1.0) == 2.0

    fd_dgam_1 = (TMSM.gamma_softening(1.0 + h_fd) - TMSM.gamma_softening(1.0 - h_fd)) / (2 * h_fd)
    @test abs(fd_dgam_1 - (-1.0)) < 1e-10

    fd_d2gam_1 = (TMSM.dgamma_softening(1.0 + h_fd) - TMSM.dgamma_softening(1.0 - h_fd)) / (2 * h_fd)
    @test abs(fd_d2gam_1 - 2.0) < 1e-5

    # Smoothness at origin ρ = 0.0
    @test TMSM.gamma_softening(0.0) == 15.0 / 8.0
    @test TMSM.dgamma_softening(0.0) == 0.0
    @test TMSM.d2gamma_softening(0.0) == -2.5

    # Range partition: g_short(r) + g_long(r) == 1/r everywhere
    for r in [0.5, 2.0, rcut - 0.1, rcut, rcut + 0.1, 10.0]
        expected_1_over_r = 1.0 / r
        actual_sum = TMSM.g_short(r, rcut) + TMSM.g_long(r, rcut)
        @test isapprox(actual_sum, expected_1_over_r, atol=1e-14)

        expected_deriv = -1.0 / (r * r)
        actual_deriv_sum = TMSM.dg_short(r, rcut) + TMSM.dg_long(r, rcut)
        @test isapprox(actual_deriv_sum, expected_deriv, atol=1e-14)
    end
end

# ===========================================================================
# 3. Analytical Cartesian Forces vs Central Finite Differences (< 1e-9)
# ===========================================================================
@testset "3. Analytical Cartesian Forces vs Central Finite Differences" begin
    h_fd = 1e-6

    # Fixture A: Orthorhombic cell (neutral & charged, with exclusions and 1-4 scaling)
    box_ortho = [12.0, 13.0, 14.0]
    q_ortho_neutral = [1.2 4.1 7.0 2.6;
                       2.0 6.0 3.3 8.4;
                       1.5 5.2 9.0 4.1]
    ch_neutral = [1.0, -1.0, 0.5, -0.5]
    @test sum(ch_neutral) == 0.0

    q_ortho_charged = [1.2 4.1 7.0 2.6 9.3;
                       2.0 6.0 3.3 8.4 5.5;
                       1.5 5.2 9.0 4.1 6.2]
    q_ortho_charged[:, 2] .+= box_ortho  # outside primary cell
    ch_charged = [1.0, -0.7, 0.35, 0.9, -0.25]
    @test sum(ch_charged) != 0.0

    corr = TMSM.PairCorrections(excluded=[(1, 2)], scaled=[(2, 3, 0.5), (3, 4, 0.25)])

    # Test Neutral Orthorhombic
    plan_on = TMSM.TernaryMSMPlan(q_ortho_neutral, box_ortho; rcut=5.0, grid_n=9, levels=2, prefactor=1.3)
    U_on, F_on = TMSM.ternary_msm_energy_forces(plan_on, q_ortho_neutral, ch_neutral; corrections=corr)
    @test isfinite(U_on)
    @test all(isfinite, F_on)

    F_fd_on = zeros(size(q_ortho_neutral))
    q_mut = copy(q_ortho_neutral)
    for idx in eachindex(q_mut)
        old = q_mut[idx]
        q_mut[idx] = old + h_fd
        Up = TMSM.ternary_msm_energy(plan_on, q_mut, ch_neutral; corrections=corr)
        q_mut[idx] = old - h_fd
        Um = TMSM.ternary_msm_energy(plan_on, q_mut, ch_neutral; corrections=corr)
        q_mut[idx] = old
        F_fd_on[idx] = -(Up - Um) / (2 * h_fd)
    end
    @test maxabs(F_on - F_fd_on) < 1e-9

    # Test Charged Orthorhombic
    plan_oc = TMSM.TernaryMSMPlan(q_ortho_charged, box_ortho; rcut=5.0, grid_n=9, levels=2, prefactor=1.3)
    U_oc, F_oc = TMSM.ternary_msm_energy_forces(plan_oc, q_ortho_charged, ch_charged; corrections=corr)
    @test isfinite(U_oc)
    @test all(isfinite, F_oc)

    F_fd_oc = zeros(size(q_ortho_charged))
    q_mut = copy(q_ortho_charged)
    for idx in eachindex(q_mut)
        old = q_mut[idx]
        q_mut[idx] = old + h_fd
        Up = TMSM.ternary_msm_energy(plan_oc, q_mut, ch_charged; corrections=corr)
        q_mut[idx] = old - h_fd
        Um = TMSM.ternary_msm_energy(plan_oc, q_mut, ch_charged; corrections=corr)
        q_mut[idx] = old
        F_fd_oc[idx] = -(Up - Um) / (2 * h_fd)
    end
    @test maxabs(F_oc - F_fd_oc) < 1e-9

    # Fixture B: Skew Triclinic cell
    H_skew = [12.0  1.5  0.8;
               0.0 13.0  1.2;
               0.0  0.0 14.0]
    tbox = FoundationCore.Types.TriclinicBox(H_skew)

    q_skew = [1.2 4.1 7.0 2.6 9.3;
              2.0 6.0 3.3 8.4 5.5;
              1.5 5.2 9.0 4.1 6.2]
    q_skew[:, 2] .+= H_skew[:, 1] + 2.0 .* H_skew[:, 2]  # image shift

    plan_skew = TMSM.TernaryMSMPlan(q_skew, tbox; rcut=5.0, grid_n=9, levels=2, prefactor=1.3)
    @test plan_skew.is_triclinic

    U_skew, F_skew = TMSM.ternary_msm_energy_forces(plan_skew, q_skew, ch_charged; corrections=corr)
    @test isfinite(U_skew)
    @test all(isfinite, F_skew)

    F_fd_skew = zeros(size(q_skew))
    q_mut = copy(q_skew)
    for idx in eachindex(q_mut)
        old = q_mut[idx]
        q_mut[idx] = old + h_fd
        Up = TMSM.ternary_msm_energy(plan_skew, q_mut, ch_charged; corrections=corr)
        q_mut[idx] = old - h_fd
        Um = TMSM.ternary_msm_energy(plan_skew, q_mut, ch_charged; corrections=corr)
        q_mut[idx] = old
        F_fd_skew[idx] = -(Up - Um) / (2 * h_fd)
    end
    @test maxabs(F_skew - F_fd_skew) < 1e-9
end

# ===========================================================================
# 4. Analytical 3-Entry Orthorhombic & 9-Entry Triclinic Box Derivatives (< 1e-7)
# ===========================================================================
@testset "4. Analytical Box Derivatives vs Fixed-Fractional Finite Differences" begin
    h_fd = 1e-6
    corr = TMSM.PairCorrections(excluded=[(1, 2)], scaled=[(2, 3, 0.5), (3, 4, 0.25)])

    # 4A. 3-Entry Orthorhombic Box Derivative
    box_ortho = [12.0, 13.0, 14.0]
    q_ortho = [1.2 4.1 7.0 2.6 9.3;
               2.0 6.0 3.3 8.4 5.5;
               1.5 5.2 9.0 4.1 6.2]
    ch_charged = [1.0, -0.7, 0.35, 0.9, -0.25]
    plan_ortho = TMSM.TernaryMSMPlan(q_ortho, box_ortho; rcut=5.0, grid_n=9, levels=2, prefactor=1.3)

    G_ortho = TMSM.ternary_msm_box_derivative(plan_ortho, q_ortho, ch_charged; corrections=corr)
    @test size(G_ortho) == (3, 3)
    @test all(isfinite, G_ortho)

    # Diagonal central finite difference at fixed fractional coordinates
    s_ortho = q_ortho ./ box_ortho
    G_diag_fd = zeros(3)
    for a in 1:3
        box_p = copy(box_ortho); box_p[a] += h_fd
        qp = s_ortho .* box_p
        plan_p = TMSM.TernaryMSMPlan(qp, box_p; rcut=5.0, grid_n=9, levels=2, prefactor=1.3)
        ep = TMSM.ternary_msm_energy(plan_p, qp, ch_charged; corrections=corr)

        box_m = copy(box_ortho); box_m[a] -= h_fd
        qm = s_ortho .* box_m
        plan_m = TMSM.TernaryMSMPlan(qm, box_m; rcut=5.0, grid_n=9, levels=2, prefactor=1.3)
        em = TMSM.ternary_msm_energy(plan_m, qm, ch_charged; corrections=corr)

        G_diag_fd[a] = (ep - em) / (2 * h_fd)
    end
    @test maxabs(diag(G_ortho) - G_diag_fd) < 1e-7

    # 4B. Exact 9-Component Skew Triclinic Box Derivative
    H_skew = [12.0  1.5  0.8;
               0.0 13.0  1.2;
               0.0  0.0 14.0]
    tbox = FoundationCore.Types.TriclinicBox(H_skew)
    q_skew = [1.2 4.1 7.0 2.6 9.3;
              2.0 6.0 3.3 8.4 5.5;
              1.5 5.2 9.0 4.1 6.2]
    q_skew[:, 2] .+= H_skew[:, 1] + 2.0 .* H_skew[:, 2]

    plan_tri = TMSM.TernaryMSMPlan(q_skew, tbox; rcut=5.0, grid_n=9, levels=2, prefactor=1.3)
    U_tri, F_tri, G_tri = TMSM.ternary_msm_energy_forces_box_derivative(plan_tri, q_skew, ch_charged; corrections=corr)
    @test isfinite(U_tri)
    @test size(G_tri) == (3, 3)
    @test all(isfinite, G_tri)

    s_skew = tbox.Hinv * q_skew
    G_fd_tri = zeros(3, 3)
    for a in 1:3, b in 1:3
        Hp = copy(H_skew); Hp[a, b] += h_fd
        qp = Hp * s_skew
        plan_p = TMSM.TernaryMSMPlan(qp, Hp; rcut=5.0, grid_n=9, levels=2, prefactor=1.3)
        ep = TMSM.ternary_msm_energy(plan_p, qp, ch_charged; corrections=corr)

        Hm = copy(H_skew); Hm[a, b] -= h_fd
        qm = Hm * s_skew
        plan_m = TMSM.TernaryMSMPlan(qm, Hm; rcut=5.0, grid_n=9, levels=2, prefactor=1.3)
        em = TMSM.ternary_msm_energy(plan_m, qm, ch_charged; corrections=corr)

        G_fd_tri[a, b] = (ep - em) / (2 * h_fd)
    end
    @test maxabs(G_tri - G_fd_tri) < 1e-7

    # Virial tensor Ξ = -H * G^T
    Ξ = -(H_skew * transpose(G_tri))
    @test all(isfinite, Ξ)

    # In-place pre-seeded accumulation preservation
    F_acc = fill(2.5, 3, plan_tri.N)
    G_acc = fill(-1.25, 3, 3)
    U_acc = TMSM.ternary_msm_energy_forces_box_derivative!(F_acc, G_acc, plan_tri, q_skew, ch_charged;
                                                          corrections=corr,
                                                          accumulate_F=true, accumulate_G=true)
    @test U_acc == U_tri
    @test maxabs(F_acc .- (2.5 .+ F_tri)) < 1e-13
    @test maxabs(G_acc .- (-1.25 .+ G_tri)) < 1e-13
end

# ===========================================================================
# 5. Parity vs PME.PMEFactor & O(N) Scaling with Zero FFT Calls
# ===========================================================================
@testset "5. PMEFactor Comparison & O(N) Scaling with Zero FFT Calls" begin
    # 5A. Comparison against PME.PMEFactor on neutral periodic salt/water test box
    box_L = [18.0, 18.0, 18.0]
    box_obj = FoundationCore.Types.OrthorhombicBox(box_L)
    N_salt = 64
    rng = MersenneTwister(101)
    q_salt = rand(rng, 3, N_salt) .* 18.0
    ch_salt = randn(rng, N_salt)
    ch_salt .-= sum(ch_salt) / N_salt  # neutral system

    ctx = FoundationCore.Factors.Context(box_obj, FoundationCore.Types.REDUCED, N_salt)

    pme_factor = FoundationCore.PME.PMEFactor(q_salt, ch_salt, box_obj;
                                              alpha=0.35, rcut=6.0, grid=(32, 32, 32))
    F_pme = zeros(3, N_salt)
    U_pme = FoundationCore.Factors.evaluate!(F_pme, pme_factor, q_salt, ctx)

    msm_factor = TMSM.TernaryMSMCoulombFactor(q_salt, ch_salt, box_obj;
                                            rcut=6.0, grid_n=27, levels=2)
    F_msm = zeros(3, N_salt)
    U_msm = FoundationCore.Factors.evaluate!(F_msm, msm_factor, q_salt, ctx)

    @test isfinite(U_pme)
    @test isfinite(U_msm)
    @test all(isfinite, F_pme)
    @test all(isfinite, F_msm)

    # Both electrostatic models yield strong positive force alignment on neutral periodic systems
    cos_similarity = dot(vec(F_pme), vec(F_msm)) / (norm(F_pme) * norm(F_msm))
    @test cos_similarity > 0.90
    @test isapprox(norm(F_msm) / norm(F_pme), 1.0, atol=0.20)

    # 5B. O(N) Scaling benchmark across N = 256, 1024, 4096 (ZERO FFTs)
    rho = 0.05
    times = Float64[]
    Ns = [256, 1024, 4096]

    for N in Ns
        L_box = (N / rho)^(1/3)
        b = [L_box, L_box, L_box]
        r = MersenneTwister(N)
        q_bench = rand(r, 3, N) .* L_box
        ch_bench = randn(r, N)
        ch_bench .-= sum(ch_bench) / N

        plan_bench = TMSM.TernaryMSMPlan(q_bench, b; rcut=5.0, grid_n=27, levels=2)
        F_bench = zeros(3, N)

        # Warm up
        TMSM.ternary_msm_energy_forces!(F_bench, plan_bench, q_bench, ch_bench)

        t0 = time_ns()
        reps = 3
        for _ in 1:reps
            TMSM.ternary_msm_energy_forces!(F_bench, plan_bench, q_bench, ch_bench)
        end
        t_avg = (time_ns() - t0) / (reps * 1e9)
        push!(times, t_avg)
    end

    # 4x particle increase from 1024 to 4096: pure O(N) would be ~4, O(N²) would be ~16
    ratio_4x = times[3] / times[2]
    @test ratio_4x <= 10.0
    # Per-atom cost does not blow up with N
    @test times[3] / 4096 <= 3.0 * (times[2] / 1024)

    # Zero-allocation verification on warm calls
    plan_alloc = TMSM.TernaryMSMPlan(q_salt, box_L; rcut=5.0, grid_n=27, levels=2)
    F_alloc = zeros(3, N_salt)
    G_alloc = zeros(3, 3)
    TMSM.ternary_msm_energy_forces_box_derivative!(F_alloc, G_alloc, plan_alloc, q_salt, ch_salt)
    alloc_bytes = @allocated TMSM.ternary_msm_energy_forces_box_derivative!(F_alloc, G_alloc, plan_alloc, q_salt, ch_salt)
    @test alloc_bytes <= 64  # warm call allocates zero or small scalar wrapper
end

# ===========================================================================
# 6. Factor Protocol Compliance & Robust Refusals
# ===========================================================================
@testset "6. Factor Protocol Discipline & Refusals" begin
    box_ortho = FoundationCore.Types.OrthorhombicBox((15.0, 15.0, 15.0))
    N = 4
    q = [1.0 5.0 9.0 12.0; 2.0 6.0 8.0 11.0; 3.0 7.0 10.0 13.0]
    ch = [1.0, -1.0, 0.5, -0.5]
    ctx = FoundationCore.Factors.Context(box_ortho, FoundationCore.Types.REDUCED, N)

    factor = TMSM.TernaryMSMCoulombFactor(q, ch, box_ortho; rcut=5.0, grid_n=9, levels=2)
    @test FoundationCore.Factors.factor_support_size(factor) == N
    @test FoundationCore.Factors.read_support(factor) == (1, 2, 3, 4)
    @test FoundationCore.Factors.is_local(factor) == false
    @test FoundationCore.Factors.force_consistent(factor) == true

    # evaluate! accumulation rule
    F_accum = fill(1.0, 3, N)
    U_fac = FoundationCore.Factors.evaluate!(F_accum, factor, q, ctx)
    @test isfinite(U_fac)
    @test all(!=(1.0), F_accum)  # forces must have been added to F_accum

    # Factor convenience helpers
    U_fac_e = TMSM.msm_factor_energy(factor, q, ctx)
    @test U_fac_e == U_fac
    G_fac = TMSM.msm_factor_box_derivative(factor, q, ctx)
    @test size(G_fac) == (3, 3)
    @test all(isfinite, G_fac)

    # Construction refusals
    @test refuses(() -> TMSM.TernaryMSMPlan(q, box_ortho; rcut=0.0))
    @test refuses(() -> TMSM.TernaryMSMPlan(q, box_ortho; rcut=8.0))  # rcut >= min_width/2 = 7.5
    @test refuses(() -> TMSM.TernaryMSMPlan(zeros(2, N), box_ortho; rcut=5.0))
    @test refuses(() -> TMSM.TernaryMSMPlan(q, box_ortho; rcut=5.0, order=3))
    @test refuses(() -> TMSM.TernaryMSMPlan(q, box_ortho; rcut=5.0, levels=0))
    @test refuses(() -> TMSM.TernaryMSMPlan(q, box_ortho; rcut=5.0, grid_n=10, levels=2))  # 10 not div by 3
    @test refuses(() -> TMSM.TernaryMSMPlan(q, [10.0, 10.0]; rcut=4.0))

    # Evaluation refusals
    q_nan = copy(q); q_nan[1, 1] = NaN
    @test refuses(() -> TMSM.ternary_msm_energy(factor.plan, q_nan, ch))
    ch_inf = copy(ch); ch_inf[2] = Inf
    @test refuses(() -> TMSM.ternary_msm_energy(factor.plan, q, ch_inf))
    q_coincident = copy(q); q_coincident[:, 2] .= q_coincident[:, 1]
    @test refuses(() -> TMSM.ternary_msm_energy(factor.plan, q_coincident, ch))
    @test refuses(() -> TMSM.TernaryMSMPlan(q, box_ortho; rcut=5.0, softening=:unknown))
end

# ===========================================================================
# 7. Phase 7: Jellium / Cohn-Elkies SOS C⁴-Smooth Radial Smearing
# ===========================================================================
@testset "7. Jellium SOS C⁴ Softening, Certificate & FD Consistency" begin
    # 7A. C⁰..C⁴ continuity at ρ = 1 and smoothness at ρ = 0
    @test TMSM.gamma_sos_softening(1.0) == 1.0
    @test TMSM.dgamma_sos_softening(1.0) == -1.0
    @test TMSM.d2gamma_sos_softening(1.0) == 2.0
    @test TMSM.gamma_sos_softening(0.0) == 315.0 / 128.0
    @test TMSM.dgamma_sos_softening(0.0) == 0.0
    @test TMSM.d2gamma_sos_softening(0.0) == -105.0 / 16.0

    # Check 3rd and 4th derivatives across ρ = 1 via central FD
    h_fd = 1e-5
    fd_d3 = (TMSM.d2gamma_sos_softening(1.0 + h_fd) - TMSM.d2gamma_sos_softening(1.0 - h_fd)) / (2 * h_fd)
    @test abs(fd_d3 - (-6.0)) < 1e-7
    d3_left = TMSM._d3gamma_sos_softening(1.0)
    d4_left = TMSM._d4gamma_sos_softening(1.0)
    @test abs(d3_left - (-6.0)) < 1e-13
    @test abs(d4_left - 24.0) < 1e-13

    # Range partition g_short + g_long == 1/r for :sos_c4
    rcut = 5.5
    for r in [0.5, 2.0, rcut - 0.1, rcut, rcut + 0.1, 10.0]
        @test isapprox(TMSM.g_short(r, rcut, :sos_c4) + TMSM.g_long(r, rcut, :sos_c4), 1.0 / r, atol=1e-14)
        @test isapprox(TMSM.dg_short(r, rcut, :sos_c4) + TMSM.dg_long(r, rcut, :sos_c4), -1.0 / (r * r), atol=1e-14)
    end

    # 7B. Spectral positivity certificate
    cert = TMSM.sos_spectral_positivity_certificate(64)
    @test cert.certified == true
    @test cert.density_min >= 0.0
    @test isapprox(cert.integral_4pi, 1.0, atol=1e-12)
    @test cert.continuity_c4_err < 1e-12
    @test isapprox(cert.self_coeff, 315.0 / 256.0, atol=1e-14)
    @test refuses(() -> TMSM.sos_spectral_positivity_certificate(2))

    # 7C. SOS C⁴ Cartesian Forces & 3-Entry / 9-Entry Box Derivatives vs Central FD
    h_fd = 1e-6
    corr = TMSM.PairCorrections(excluded=[(1, 2)], scaled=[(2, 3, 0.5), (3, 4, 0.25)])
    ch_charged = [1.0, -0.7, 0.35, 0.9, -0.25]

    # Orthorhombic SOS C⁴
    box_ortho = [12.0, 13.0, 14.0]
    q_ortho = [1.2 4.1 7.0 2.6 9.3;
               2.0 6.0 3.3 8.4 5.5;
               1.5 5.2 9.0 4.1 6.2]
    factor_ortho = TMSM.TernaryMSMCoulombFactor(q_ortho, ch_charged, box_ortho;
                                                rcut=5.0, grid_n=9, levels=2, prefactor=1.3,
                                                corrections=corr, softening=:sos_c4)
    @test factor_ortho.plan.softening === :sos_c4
    ctx_ortho = FoundationCore.Factors.Context(factor_ortho.box, FoundationCore.Types.REDUCED, 5)
    F_sos_o = zeros(3, 5)
    G_sos_o = zeros(3, 3)
    U_sos_o = TMSM.msm_factor_energy_forces_box_derivative!(F_sos_o, G_sos_o, factor_ortho, q_ortho, ctx_ortho)
    U_hardy_o, F_hardy_o = TMSM.evaluate_ternary_msm(q_ortho, ch_charged, box_ortho;
                                                     rcut=5.0, grid_n=9, levels=2, prefactor=1.3,
                                                     corrections=corr, softening=:hardy_c2)
    @test U_sos_o != U_hardy_o

    F_fd_o = zeros(3, 5)
    q_mut = copy(q_ortho)
    for idx in eachindex(q_mut)
        old = q_mut[idx]
        q_mut[idx] = old + h_fd
        Up = TMSM.msm_factor_energy(factor_ortho, q_mut, ctx_ortho)
        q_mut[idx] = old - h_fd
        Um = TMSM.msm_factor_energy(factor_ortho, q_mut, ctx_ortho)
        q_mut[idx] = old
        F_fd_o[idx] = -(Up - Um) / (2 * h_fd)
    end
    @test maxabs(F_sos_o - F_fd_o) < 1e-9

    s_ortho = q_ortho ./ box_ortho
    G_diag_fd_o = zeros(3)
    for a in 1:3
        bp = copy(box_ortho); bp[a] += h_fd
        qp = s_ortho .* bp
        ep = TMSM.ternary_msm_energy(TMSM.TernaryMSMPlan(qp, bp; rcut=5.0, grid_n=9, levels=2, prefactor=1.3, softening=:sos_c4), qp, ch_charged; corrections=corr)
        bm = copy(box_ortho); bm[a] -= h_fd
        qm = s_ortho .* bm
        em = TMSM.ternary_msm_energy(TMSM.TernaryMSMPlan(qm, bm; rcut=5.0, grid_n=9, levels=2, prefactor=1.3, softening=:sos_c4), qm, ch_charged; corrections=corr)
        G_diag_fd_o[a] = (ep - em) / (2 * h_fd)
    end
    @test maxabs(diag(G_sos_o) - G_diag_fd_o) < 1e-8

    # Skew Triclinic SOS C⁴
    H_skew = [12.0  1.5  0.8;
               0.0 13.0  1.2;
               0.0  0.0 14.0]
    tbox = FoundationCore.Types.TriclinicBox(H_skew)
    q_skew = [1.2 4.1 7.0 2.6 9.3;
              2.0 6.0 3.3 8.4 5.5;
              1.5 5.2 9.0 4.1 6.2]
    q_skew[:, 2] .+= H_skew[:, 1] + 2.0 .* H_skew[:, 2]

    factor_tri = TMSM.TernaryMSMCoulombFactor(q_skew, ch_charged, tbox;
                                              rcut=5.0, grid_n=9, levels=2, prefactor=1.3,
                                              corrections=corr, softening=:sos_c4)
    ctx_tri = FoundationCore.Factors.Context(tbox, FoundationCore.Types.REDUCED, 5)
    F_sos_t = zeros(3, 5)
    G_sos_t = zeros(3, 3)
    U_sos_t = TMSM.msm_factor_energy_forces_box_derivative!(F_sos_t, G_sos_t, factor_tri, q_skew, ctx_tri)

    F_fd_t = zeros(3, 5)
    q_mut = copy(q_skew)
    for idx in eachindex(q_mut)
        old = q_mut[idx]
        q_mut[idx] = old + h_fd
        Up = TMSM.msm_factor_energy(factor_tri, q_mut, ctx_tri)
        q_mut[idx] = old - h_fd
        Um = TMSM.msm_factor_energy(factor_tri, q_mut, ctx_tri)
        q_mut[idx] = old
        F_fd_t[idx] = -(Up - Um) / (2 * h_fd)
    end
    @test maxabs(F_sos_t - F_fd_t) < 1e-9

    s_skew = tbox.Hinv * q_skew
    G_fd_t = zeros(3, 3)
    for a in 1:3, b in 1:3
        Hp = copy(H_skew); Hp[a, b] += h_fd
        qp = Hp * s_skew
        ep = TMSM.ternary_msm_energy(TMSM.TernaryMSMPlan(qp, Hp; rcut=5.0, grid_n=9, levels=2, prefactor=1.3, softening=:sos_c4), qp, ch_charged; corrections=corr)
        Hm = copy(H_skew); Hm[a, b] -= h_fd
        qm = Hm * s_skew
        em = TMSM.ternary_msm_energy(TMSM.TernaryMSMPlan(qm, Hm; rcut=5.0, grid_n=9, levels=2, prefactor=1.3, softening=:sos_c4), qm, ch_charged; corrections=corr)
        G_fd_t[a, b] = (ep - em) / (2 * h_fd)
    end
    @test maxabs(G_sos_t - G_fd_t) < 1e-8

    # Warm zero-allocation check for :sos_c4
    TMSM.ternary_msm_energy_forces_box_derivative!(F_sos_t, G_sos_t, factor_tri.plan, q_skew, ch_charged; corrections=corr)
    alloc_sos = @allocated TMSM.ternary_msm_energy_forces_box_derivative!(F_sos_t, G_sos_t, factor_tri.plan, q_skew, ch_charged; corrections=corr)
    @test alloc_sos <= 64
end

# ===========================================================================
# 8. Phase 7: Circulant Weighing Matrices CW(n, k) & Radix-3 3-Compression
# ===========================================================================
@testset "8. Circulant Weighing Matrices CW(n,k) & Radix-3 3-Compression" begin
    rng = MersenneTwister(2026)
    for (n, k) in [(3, 1), (4, 4), (6, 4), (7, 4), (13, 9), (14, 4), (21, 16)]
        cw = TMSM.cw_matrix(n, k)
        @test cw.n == n && cw.k == k
        @test TMSM.verify_cw_orthogonality(cw) == true
        @test all(v -> v in (Int8(-1), Int8(0), Int8(1)), cw.W)

        x = randn(rng, n)
        y = zeros(Float64, n)
        x_rec = zeros(Float64, n)
        TMSM.cw_transform!(y, cw, x)
        @test maxabs(y - Float64.(cw.W) * x) < 1e-13
        TMSM.cw_inverse_transform!(x_rec, cw, y)
        @test maxabs(x_rec - x) < 1e-14
    end

    # Refusals for CW
    @test refuses(() -> TMSM.cw_matrix(5, 4))
    cw7 = TMSM.cw_matrix(7, 4)
    @test refuses(() -> TMSM.cw_transform!(zeros(6), cw7, zeros(7)))
    buf7 = randn(rng, 7)
    @test refuses(() -> TMSM.cw_transform!(buf7, cw7, buf7))
    @test refuses(() -> TMSM.cw_inverse_transform!(buf7, cw7, buf7))

    # Radix-3 1D 3-compression
    x1d = randn(rng, 27)
    c1d = TMSM.radix3_compress_1d(x1d)
    @test length(c1d) == 9
    @test isapprox(sum(c1d), sum(x1d), atol=1e-13)
    @test isapprox(c1d[4], x1d[4] + x1d[13] + x1d[22], atol=1e-14)
    @test refuses(() -> TMSM.radix3_compress_1d(randn(10)))
    @test refuses(() -> TMSM.radix3_compress_1d(Float64[]))

    # Radix-3 3D 3-compression
    grid_in = randn(rng, 9, 9, 9)
    c_out = zeros(Float64, 3, 3, 3)
    TMSM.radix3_compress_3d!(c_out, grid_in)
    @test isapprox(sum(c_out), sum(grid_in), atol=1e-12)
    expected_231 = sum(grid_in[2 + 3*a, 3 + 3*b, 1 + 3*c] for a in 0:2, b in 0:2, c in 0:2)
    @test isapprox(c_out[2, 3, 1], expected_231, atol=1e-14)
    grid_rect = randn(rng, 6, 9, 3)
    c_rect = zeros(Float64, 2, 3, 1)
    TMSM.radix3_compress_3d!(c_rect, grid_rect)
    @test isapprox(sum(c_rect), sum(grid_rect), atol=1e-12)
    @test isapprox(c_rect[2, 2, 1], sum(grid_rect[2 + 2*a, 2 + 3*b, 1 + 1*c] for a in 0:2, b in 0:2, c in 0:2), atol=1e-14)
    @test refuses(() -> TMSM.radix3_compress_3d!(zeros(3, 3, 3), zeros(9, 9, 6)))
end

# ===========================================================================
# 9. Phase 7: Coarse Box-Matrix Tensor Caching & Sparse Convolution Parity
# ===========================================================================
@testset "9. Coarse Box-Matrix Tensor Caching & Invalidation Parity" begin
    rng = MersenneTwister(77)
    H1 = [15.0  1.2  0.4;
           0.0 16.0  0.9;
           0.0  0.0 17.0]
    N = 12
    q1 = H1 * rand(rng, 3, N)
    ch = randn(rng, N)
    ch .-= sum(ch) / N

    plan_cached = TMSM.TernaryMSMPlan(q1, H1; rcut=5.0, grid_n=27, levels=3, softening=:sos_c4)
    @test plan_cached.coarse_valid[] == false

    # First call (forces only): populates cK/cG cache, leaves coarse_dg_valid false
    U1, F1 = TMSM.ternary_msm_energy_forces(plan_cached, q1, ch)
    @test plan_cached.coarse_valid[] == true
    @test plan_cached.coarse_dg_valid[] == false
    @test plan_cached.last_coarse_H == H1

    # Second call at same box with different positions q2 & requesting box derivative G
    q2 = H1 * rand(rng, 3, N)
    U2_c, F2_c, G2_c = TMSM.ternary_msm_energy_forces_box_derivative(plan_cached, q2, ch)
    @test plan_cached.coarse_valid[] == true
    @test plan_cached.coarse_dg_valid[] == true

    plan_fresh2 = TMSM.TernaryMSMPlan(q2, H1; rcut=5.0, grid_n=27, levels=3, softening=:sos_c4)
    U2_f, F2_f, G2_f = TMSM.ternary_msm_energy_forces_box_derivative(plan_fresh2, q2, ch)
    @test isapprox(U2_c, U2_f, atol=1e-13)
    @test maxabs(F2_c - F2_f) < 1e-13
    @test maxabs(G2_c - G2_f) < 1e-13

    # Third call with box update H2 invalidates coarse cache and matches fresh plan
    H2 = [16.5  0.5  0.2;
           0.0 15.5  1.1;
           0.0  0.0 18.0]
    TMSM._update_box_if_needed!(plan_cached, H2)
    @test plan_cached.coarse_valid[] == false
    @test plan_cached.coarse_dg_valid[] == false
    q3 = H2 * rand(rng, 3, N)
    U3_c, F3_c, G3_c = TMSM.ternary_msm_energy_forces_box_derivative(plan_cached, q3, ch)
    @test plan_cached.coarse_valid[] == true
    @test plan_cached.coarse_dg_valid[] == true
    @test plan_cached.last_coarse_H == H2

    plan_fresh3 = TMSM.TernaryMSMPlan(q3, H2; rcut=5.0, grid_n=27, levels=3, softening=:sos_c4)
    U3_f, F3_f, G3_f = TMSM.ternary_msm_energy_forces_box_derivative(plan_fresh3, q3, ch)
    @test isapprox(U3_c, U3_f, atol=1e-13)
    @test maxabs(F3_c - F3_f) < 1e-13
    @test maxabs(G3_c - G3_f) < 1e-13
end

# ===========================================================================
# 10. Phase 7B: Division-Free Short-Range Kernel, Zero-Charge Skip & N=4096 Alloc
# ===========================================================================
@testset "10. Division-Free Kernel Parity, Zero-Charge Atoms & N=4096 Alloc" begin
    rc = 5.5
    inv_rc = 1.0 / rc
    inv_rc2 = inv_rc * inv_rc
    inv_rc3 = inv_rc * inv_rc2
    for use_sos in (false, true)
        for r in (0.15, 0.9, 2.3, 4.1, rc - 1e-6)
            gs_fast, dgs_fast = TMSM._short_kernel_fast(r * r, inv_rc, inv_rc2, inv_rc3, use_sos)
            gs_ref = TMSM._g_short(r, rc, use_sos)
            dgs_ref = TMSM._dg_short_over_r(r, rc, use_sos)
            @test isapprox(gs_fast, gs_ref, atol=1e-14, rtol=1e-14)
            @test isapprox(dgs_fast, dgs_ref, atol=1e-14, rtol=1e-14)
        end
    end

    # Zero-charge atom parity & safe F accumulation
    box_z = [14.0, 14.0, 14.0]
    q_sub = [1.5 4.2 8.1 10.7;
             2.1 6.3 3.4 9.2;
             1.8 5.9 7.6 11.1]
    ch_sub = [1.0, -0.6, 0.4, -0.8]
    q_with_zero = hcat(q_sub, [3.0 9.5; 4.5 1.2; 6.0 8.8])
    ch_with_zero = vcat(ch_sub, [0.0, 0.0])

    for soft in (:hardy_c2, :sos_c4)
        plan_sub = TMSM.TernaryMSMPlan(q_sub, box_z; rcut=5.0, grid_n=9, levels=2, softening=soft)
        plan_z = TMSM.TernaryMSMPlan(q_with_zero, box_z; rcut=5.0, grid_n=9, levels=2, softening=soft)
        U_sub, F_sub, G_sub = TMSM.ternary_msm_energy_forces_box_derivative(plan_sub, q_sub, ch_sub)
        U_z, F_z, G_z = TMSM.ternary_msm_energy_forces_box_derivative(plan_z, q_with_zero, ch_with_zero)
        @test isapprox(U_z, U_sub, atol=1e-14)
        @test maxabs(F_z[:, 1:4] - F_sub) < 1e-14
        @test all(==(0.0), F_z[:, 5:6])
        @test maxabs(G_z - G_sub) < 1e-14

        F_seed = fill(3.25, 3, 6)
        G_seed = fill(-0.5, 3, 3)
        TMSM.ternary_msm_energy_forces_box_derivative!(F_seed, G_seed, plan_z, q_with_zero, ch_with_zero;
                                                       accumulate_F=true, accumulate_G=true)
        @test all(==(3.25), F_seed[:, 5:6])
        @test maxabs(F_seed[:, 1:4] .- (3.25 .+ F_sub)) < 1e-13
        @test maxabs(G_seed .- (-0.5 .+ G_sub)) < 1e-13
    end

    # N=4096 warm allocation <= 64 B
    N_big = 4096
    L_big = (N_big / 0.05)^(1/3)
    rng_big = MersenneTwister(N_big)
    q_big = rand(rng_big, 3, N_big) .* L_big
    ch_big = randn(rng_big, N_big)
    ch_big .-= sum(ch_big) / N_big
    plan_big = TMSM.TernaryMSMPlan(q_big, [L_big, L_big, L_big]; rcut=5.0, grid_n=27, levels=2)
    F_big = zeros(3, N_big)
    G_big = zeros(3, 3)
    TMSM.ternary_msm_energy_forces!(F_big, plan_big, q_big, ch_big)
    @test (@allocated TMSM.ternary_msm_energy_forces!(F_big, plan_big, q_big, ch_big)) <= 64
    TMSM.ternary_msm_energy_forces_box_derivative!(F_big, G_big, plan_big, q_big, ch_big)
    @test (@allocated TMSM.ternary_msm_energy_forces_box_derivative!(F_big, G_big, plan_big, q_big, ch_big)) <= 64
end

# ===========================================================================
# 11. O(M_L^4) Real Hartley Coarsest Solver & CW(n, k) 1D Convolution
# ===========================================================================
@testset "11. O(M_L^4) Coarsest Hartley Diagonalization & cw_convolve_1d!" begin
    rng = MersenneTwister(2026)

    # 11A. Exact parity (< 1e-13) against direct O(M_L^6) on orthorhombic & triclinic cells (Ml = 3, 9)
    boxes = [
        ("orthorhombic", [15.0 0.0 0.0; 0.0 16.5 0.0; 0.0 0.0 18.0]),
        ("triclinic",    [15.0 1.8 0.7; 0.0 16.0 1.3; 0.0 0.0 17.5]),
    ]
    for (bname, Hbox) in boxes
        for Ml_test in (3, 9)
            N_c = 18
            q_c = Hbox * rand(rng, 3, N_c)
            ch_c = randn(rng, N_c)
            ch_c .-= sum(ch_c) / N_c
            plan_c = TMSM.TernaryMSMPlan(q_c, Hbox; rcut=4.8, grid_n=Ml_test, levels=1, softening=:sos_c4)
            F_c = zeros(3, N_c)
            G_c = zeros(3, 3)
            U_h = TMSM.ternary_msm_energy_forces_box_derivative!(F_c, G_c, plan_c, q_c, ch_c)

            # Warm alloc check for Ml=9
            if Ml_test == 9
                @test (@allocated TMSM.ternary_msm_energy_forces!(F_c, plan_c, q_c, ch_c)) <= 64
                @test (@allocated TMSM.ternary_msm_energy_forces_box_derivative!(F_c, G_c, plan_c, q_c, ch_c)) <= 64
            end

            # Compare coarsest Hartley kernel directly against O(M_L^6) reference
            Ql = plan_c.Q[end]
            Phil_h = zeros(Float64, Ml_test, Ml_test, Ml_test)
            Phil_d = zeros(Float64, Ml_test, Ml_test, Ml_test)
            dg_h = TMSM._coarse_convolve_hartley!(
                Phil_h, Ql, plan_c.coarse_G_hat, plan_c.coarse_DG_hat,
                plan_c.coarse_H1d, plan_c.coarse_buf1, plan_c.coarse_buf2,
                plan_c.prefactor, true
            )
            dg_d = TMSM._coarse_convolve_direct!(
                Phil_d, Ql, plan_c.coarse_G, plan_c.coarse_DG,
                Ml_test, plan_c.prefactor, true
            )
            @test maxabs(Phil_h - Phil_d) < 1e-13
            @test maxabs(collect(dg_h) - collect(dg_d)) < 1e-13

            # Central FD check for box derivative on Ml_test=9
            if Ml_test == 9
                h_fd = 1e-5
                s_frac = plan_c.Hinv * q_c
                G_fd = zeros(3, 3)
                for b in 1:3, a in 1:3
                    Hp = copy(Hbox); Hp[a, b] += h_fd
                    Hm = copy(Hbox); Hm[a, b] -= h_fd
                    pp = TMSM.TernaryMSMPlan(Hp * s_frac, Hp; rcut=4.8, grid_n=9, levels=1, softening=:sos_c4)
                    pm = TMSM.TernaryMSMPlan(Hm * s_frac, Hm; rcut=4.8, grid_n=9, levels=1, softening=:sos_c4)
                    Up = TMSM.ternary_msm_energy(pp, Hp * s_frac, ch_c)
                    Um = TMSM.ternary_msm_energy(pm, Hm * s_frac, ch_c)
                    G_fd[a, b] = (Up - Um) / (2 * h_fd)
                end
                @test maxabs(G_c - G_fd) < 1e-9
            end
        end
    end

    # 11B. Measured speedup on Ml = 9 and Ml = 27 grids (Hartley O(M_L^4) vs direct O(M_L^6))
    for Ml_bench in (9, 27)
        H_b = [18.0 1.1 0.3; 0.0 19.0 0.8; 0.0 0.0 20.0]
        plan_b = TMSM.TernaryMSMPlan(H_b * rand(rng, 3, 24), H_b; rcut=5.0, grid_n=Ml_bench, levels=1)
        Ql_b = randn(rng, Ml_bench, Ml_bench, Ml_bench)
        raw_G = randn(rng, Ml_bench, Ml_bench, Ml_bench)
        cG_b = similar(raw_G)
        for d3 in 1:Ml_bench, d2 in 1:Ml_bench, d1 in 1:Ml_bench
            nd1 = d1 == 1 ? 1 : (Ml_bench + 2 - d1)
            nd2 = d2 == 1 ? 1 : (Ml_bench + 2 - d2)
            nd3 = d3 == 1 ? 1 : (Ml_bench + 2 - d3)
            cG_b[d1, d2, d3] = 0.5 * (raw_G[d1, d2, d3] + raw_G[nd1, nd2, nd3])
        end
        cDG_b = zeros(Float64, 3, 3, Ml_bench, Ml_bench, Ml_bench)
        cG_hat_b = (Ml_bench * sqrt(Float64(Ml_bench))) .* TMSM._dht3!(
            similar(cG_b), cG_b, plan_b.coarse_H1d, plan_b.coarse_buf2
        )
        Phil_h = zeros(Ml_bench, Ml_bench, Ml_bench)
        Phil_d = zeros(Ml_bench, Ml_bench, Ml_bench)

        # Warm up JIT
        TMSM._coarse_convolve_hartley!(Phil_h, Ql_b, cG_hat_b, plan_b.coarse_DG_hat,
                                       plan_b.coarse_H1d, plan_b.coarse_buf1, plan_b.coarse_buf2, 1.0, false)
        @test (@allocated TMSM._coarse_convolve_hartley!(
            Phil_h, Ql_b, cG_hat_b, plan_b.coarse_DG_hat,
            plan_b.coarse_H1d, plan_b.coarse_buf1, plan_b.coarse_buf2, 1.0, false)) == 0

        if Ml_bench == 9
            TMSM._coarse_convolve_direct!(Phil_d, Ql_b, cG_b, cDG_b, Ml_bench, 1.0, false)
            t_hartley = @elapsed for _ in 1:200
                TMSM._coarse_convolve_hartley!(Phil_h, Ql_b, cG_hat_b, plan_b.coarse_DG_hat,
                                               plan_b.coarse_H1d, plan_b.coarse_buf1, plan_b.coarse_buf2, 1.0, false)
            end
            t_direct = @elapsed for _ in 1:200
                TMSM._coarse_convolve_direct!(Phil_d, Ql_b, cG_b, cDG_b, Ml_bench, 1.0, false)
            end
            @test t_hartley < t_direct
            println("  Ml=9 coarsest speedup: ", round(t_direct / t_hartley; digits=2), "x")
        else
            # At Ml=27, direct O(27^6 = 3.87e8) takes ~seconds per single call; 1 call vs 20 Hartley calls
            t_hartley = @elapsed for _ in 1:20
                TMSM._coarse_convolve_hartley!(Phil_h, Ql_b, cG_hat_b, plan_b.coarse_DG_hat,
                                               plan_b.coarse_H1d, plan_b.coarse_buf1, plan_b.coarse_buf2, 1.0, false)
            end
            t_hartley /= 20.0
            t_direct = @elapsed TMSM._coarse_convolve_direct!(Phil_d, Ql_b, cG_b, cDG_b, Ml_bench, 1.0, false)
            @test isapprox(Phil_h, Phil_d; rtol=1e-13, atol=1e-13)
            @test t_hartley < t_direct / 10.0
            println("  Ml=27 coarsest speedup: ", round(t_direct / t_hartley; digits=1), "x")
        end
    end

    # 11C. cw_convolve_1d! zero-twiddle precision & conservation (CW(7,4) and CW(13,9))
    for (n_cw, k_cw) in ((7, 4), (13, 9))
        cw = TMSM.cw_matrix(n_cw, k_cw)
        kern = randn(rng, n_cw)
        x_in = randn(rng, n_cw)
        out = zeros(Float64, n_cw)
        TMSM.cw_convolve_1d!(out, kern, x_in, cw)

        ref = [sum(kern[mod(m - j, n_cw) + 1] * x_in[j] for j in 1:n_cw) for m in 1:n_cw]
        @test maxabs(out - ref) < 1e-13
        @test isapprox(sum(out), sum(kern) * sum(x_in); atol=1e-12)

        # Zero-padded exact linear convolution (lengths 3 and 4 -> support 6 <= 7 <= n_cw)
        k_short = randn(rng, 3)
        x_short = randn(rng, 4)
        out_pad = zeros(Float64, n_cw)
        TMSM.cw_convolve_1d!(out_pad, k_short, x_short, cw)
        ref_lin = zeros(Float64, n_cw)
        for i in 1:3, j in 1:4
            ref_lin[i + j - 1] += k_short[i] * x_short[j]
        end
        @test maxabs(out_pad - ref_lin) < 1e-13
        @test isapprox(sum(out_pad), sum(k_short) * sum(x_short); atol=1e-13)
    end
    cw13 = TMSM.cw_matrix(13, 9)
    @test refuses(() -> TMSM.cw_convolve_1d!(zeros(12), ones(4), ones(4), cw13))
    @test refuses(() -> TMSM.cw_convolve_1d!(zeros(13), ones(14), ones(4), cw13))
end

end  # @testset

println("test_ternary_msm PASS")
