using Test
using LinearAlgebra
using Random
using FoundationCore

const _MOD_ROOT = if isdefined(Main, :_DEV_PHASE9_ROOT)
    Main._DEV_PHASE9_ROOT
elseif isdefined(FoundationCore, :CustomPhysics) && isdefined(FoundationCore, :ReactiveNano)
    FoundationCore
else
    m = Module(:_DevPhase9)
    @eval m begin
        using FoundationCore
        const Types = FoundationCore.Types
        const Factors = FoundationCore.Factors
        const Neighbors = FoundationCore.Neighbors
        const ResolvedSystem = FoundationCore.ResolvedSystem
        Base.include(@__MODULE__, $(joinpath(@__DIR__, "..", "src", "CustomPhysics.jl")))
        Base.include(@__MODULE__, $(joinpath(@__DIR__, "..", "src", "ReactiveNano.jl")))
    end
    m
end

const CP = _MOD_ROOT.CustomPhysics
const RN = _MOD_ROOT.ReactiveNano
const T = FoundationCore.Types
const F = FoundationCore.Factors
const D = FoundationCore.Dynamics
const RS = FoundationCore.ResolvedSystem
const SC = FoundationCore.SystemContract

function _skew_box(L::Float64 = 1.6)
    H = [L      0.22*L  0.08*L;
         0.0    0.96*L  0.15*L;
         0.0    0.0     0.98*L]
    return T.TriclinicBox(H)
end

function _nve_relative_drift(factors::Vector{<:F.AbstractFactor}, q0::Matrix{Float64},
                             masses::Vector{Float64}, box::T.AbstractBox,
                             units::T.Units, dt::Float64, nsteps::Int;
                             p0::Union{Nothing,Matrix{Float64}} = nothing,
                             seed::Int = 42)
    N = size(q0, 2)
    rng = MersenneTwister(seed)
    p = p0 === nothing ? 0.05 .* randn(rng, 3, N) .* sqrt.(reshape(masses, 1, N)) : copy(p0)
    # Remove net COM drift
    p .-= sum(p, dims = 2) ./ N
    ctx = F.Context(box, units, N)
    sys = D.HarmonicSystem(factors, masses, ctx)
    E0 = D.total_energy(sys, q0, p)
    st = D.State(copy(q0), p)
    _, series = D.run_trajectory(st, sys, dt, nsteps; compensated = true)
    Es = series.compensated
    max_dev = maximum(abs.(Es .- E0))
    rel_drift = max_dev / max(abs(E0), 1e-12)
    return (E0 = E0, max_dev = max_dev, rel_drift = rel_drift, Es = Es)
end

function _check_factor_fd(factor::F.AbstractFactor, q::Matrix{Float64},
                          box::T.AbstractBox, units::T.Units;
                          h::Float64 = 1e-5, force_tol::Float64 = 1e-8,
                          box_tol::Float64 = 1e-7, order::Int = 4,
                          check_abs::Bool = false)
    res = CP.verify_custom_factor_gradients(factor, q, box, units;
                                            h = h, force_tol = force_tol, box_tol = box_tol,
                                            order = order)
    if check_abs
        @test res.max_force_err <= force_tol
        if !(box isa T.NoBox)
            @test res.max_box_err <= box_tol
        end
    else
        @test min(res.max_force_err, res.rel_force_err) <= force_tol
        if !(box isa T.NoBox)
            @test min(res.max_box_err, res.rel_box_err) <= box_tol
        end
    end
    @test res.passed
    return res
end

@testset "Phase 9A: CustomPhysics & ReactiveNano Suite" begin

# ===========================================================================
# 1. Parameter Encoding Engine
# ===========================================================================
@testset "1. ParameterEncoder (5 encodings, Jacobian FD, JSON dict, errors)" begin
    specs = [
        CP.ParameterSpec(:shift, 1.25; unit = "nm", lower = -5.0, upper = 5.0, encoding = :identity),
        CP.ParameterSpec(:sigma, 0.34; unit = "nm", lower = 0.0, encoding = :log),
        CP.ParameterSpec(:shifted_eps, 2.5; unit = "kJ/mol", lower = 1.0, encoding = :log),
        CP.ParameterSpec(:mixing, 0.35; lower = 0.0, upper = 1.0, encoding = :logit),
        CP.ParameterSpec(:phase, 2.4; unit = "rad", encoding = :angle),
        CP.ParameterSpec(:charge_q, -0.75; lower = -2.0, upper = 2.0, encoding = :ternary_quantized, trit_scale = 0.25),
    ]
    enc = CP.ParameterEncoder(specs)

    params = Dict{Symbol,Float64}(
        :shift => -1.7,
        :sigma => 0.34,
        :shifted_eps => 2.5,
        :mixing => 0.35,
        :phase => 5.0, # outside [-pi, pi], wraps under :angle
        :charge_q => -0.75,
    )
    @test CP.validate_parameters(enc, params)
    z = CP.encode_parameters(enc, params)
    dec = CP.decode_parameters(enc, z)

    @test isapprox(dec[:shift], -1.7; atol = 1e-12)
    @test isapprox(dec[:sigma], 0.34; atol = 1e-12)
    @test isapprox(dec[:shifted_eps], 2.5; atol = 1e-12)
    @test isapprox(dec[:mixing], 0.35; atol = 1e-12)
    @test isapprox(dec[:phase], rem2pi(5.0, RoundNearest); atol = 1e-12)
    @test isapprox(dec[:charge_q], -0.75; atol = 1e-12)

    # Balanced ternary round-trip across [-100, 100]
    for k in -100:100
        trits = CP.int_to_balanced_trits(k)
        @test all(t -> t in (-1, 0, 1), trits)
        @test CP.balanced_trits_to_int(trits) == k
    end

    # Finite-difference Jacobian verification (< 1e-8)
    z_test = [0.3, -0.8, 0.4, -0.7, 0.9, 2.0]
    jac = CP.encoding_jacobian_diag(enc, z_test)
    h = 1e-6
    for i in 1:length(specs)
        if specs[i].encoding === :ternary_quantized
            @test jac[i] == 0.0
            continue
        end
        zp = copy(z_test); zp[i] += h
        zm = copy(z_test); zm[i] -= h
        xp = CP.decode_parameters(enc, zp)[specs[i].name]
        xm = CP.decode_parameters(enc, zm)[specs[i].name]
        fd = (xp - xm) / (2.0 * h)
        @test abs(jac[i] - fd) < 1e-8
    end

    # Dict / JSON round-trip
    d_enc = CP.encoder_to_dict(enc)
    enc2 = CP.encoder_from_dict(d_enc)
    z2 = CP.encode_parameters(enc2, params)
    @test isapprox(z, z2; atol = 1e-14)
    @test enc2.specs[5].lower == -Inf && enc2.specs[5].upper == Inf

    # Invalid inputs & constructor checks
    @test_throws ArgumentError CP.ParameterSpec(:bad, 1.0; encoding = :unknown)
    @test_throws ArgumentError CP.ParameterSpec(:bad_log, 1.0; lower = -1.0, encoding = :log)
    @test_throws ArgumentError CP.ParameterSpec(:bad_logit, 0.5; lower = 0.0, upper = Inf, encoding = :logit)
    @test_throws ArgumentError CP.ParameterSpec(:bad_trit, 0.0; trit_scale = 0.0)
    @test_throws ArgumentError CP.ParameterSpec(:oob, 10.0; lower = 0.0, upper = 1.0)
    @test_throws ArgumentError CP.ParameterEncoder([specs[1], specs[1]])

    bad_oob = copy(params); bad_oob[:shift] = 6.0
    @test_throws ArgumentError CP.validate_parameters(enc, bad_oob)
    bad_nan = copy(params); bad_nan[:sigma] = NaN
    @test_throws ArgumentError CP.validate_parameters(enc, bad_nan)
    bad_log0 = copy(params); bad_log0[:sigma] = 0.0
    @test_throws ArgumentError CP.validate_parameters(enc, bad_log0)
    bad_logit1 = copy(params); bad_logit1[:mixing] = 1.0
    @test_throws ArgumentError CP.validate_parameters(enc, bad_logit1)
    bad_missing = delete!(copy(params), :phase)
    @test_throws ArgumentError CP.validate_parameters(enc, bad_missing)
end

# ===========================================================================
# 2. Cubic Spline & All 7 CustomPairFactor Kernels + External + ManyBody
# ===========================================================================
@testset "2. CubicSplineTable, 7 Radial Kernels, External & ManyBody Factors" begin
    # Exact reproduction of cubic polynomial
    r_grid = collect(range(0.1, 1.2; length = 25))
    poly(r) = 2.0 * r^3 - 3.5 * r^2 + 1.2 * r - 0.4
    dpoly(r) = 6.0 * r^2 - 7.0 * r + 1.2
    # Wait: natural cubic spline has S''=0 at endpoints, so test C^2 smoothness and FD on smooth potential
    u_grid = exp.(-3.0 .* (r_grid .- 0.25)) .- 0.5 .* exp.(-1.5 .* (r_grid .- 0.25))
    spline = CP.CubicSplineTable(r_grid, u_grid)

    # C^2 continuity across interior knots
    for k in 2:(length(r_grid) - 1)
        rk = r_grid[k]
        ul, dul = CP.eval_spline(spline, rk - 1e-9)
        ur, dur = CP.eval_spline(spline, rk + 1e-9)
        @test isapprox(ul, ur; atol = 1e-7)
        @test isapprox(dul, dur; atol = 1e-6)
    end
    # FD check on spline derivative
    for r_eval in (0.15, 0.44, 0.91)
        _, dudr = CP.eval_spline(spline, r_eval)
        u_p, _ = CP.eval_spline(spline, r_eval + 1e-6)
        u_m, _ = CP.eval_spline(spline, r_eval - 1e-6)
        @test abs(dudr - (u_p - u_m) / 2e-6) < 1e-8
    end

    # 4-atom test geometry inside periodic boxes
    rng = MersenneTwister(11)
    q = [0.25  0.56  0.35  0.82;
         0.30  0.34  0.66  0.71;
         0.40  0.45  0.42  0.76] .+ 0.02 .* randn(rng, 3, 4)
    pairs = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    ortho = T.OrthorhombicBox((1.8, 1.8, 1.8))
    tric = _skew_box(1.8)

    kernel_configs = [
        (:morse, Dict(:De => 250.0, :a => 12.0, :r0 => 0.30), :lfs, 0.85, nothing),
        (:buckingham, Dict(:A => 8000.0, :B => 14.0, :C => 0.05), :lfs, 0.85, nothing),
        (:yukawa, Dict(:A => 45.0, :kappa => 3.2), :lfs, 0.85, nothing),
        (:mie, Dict(:epsilon => 1.5, :sigma => 0.28, :m => 9.0, :n => 6.0), :lfs, 0.85, nothing),
        (:fene, Dict(:k => 300.0, :R0 => 1.2), :none, Inf, nothing),
        (:wca, Dict(:epsilon => 2.0, :sigma => 0.40), :none, Inf, nothing),
        (:tabulated_spline, Dict{Symbol,Float64}(), :lfs, 0.95, spline),
    ]

    for (kind, params, mode, rc, sp) in kernel_configs
        fac = CP.CustomPairFactor(kind, params, pairs; rc = rc, mode = mode, n_atoms = 4, spline = sp)
        @test F.force_consistent(fac)
        @test RS.box_derivative_support(fac) === nothing
        _check_factor_fd(fac, q, ortho, CP.MD_UNITS; h = 1e-5, force_tol = 1e-7, box_tol = 1e-6, order = 4)
        _check_factor_fd(fac, q, tric, CP.MD_UNITS; h = 1e-5, force_tol = 1e-7, box_tol = 1e-6, order = 4)
    end

    # Callable CustomPairFactor (:custom)
    e_fn(r, p) = p[:k] * (r - p[:r0])^2
    de_fn(r, p) = 2.0 * p[:k] * (r - p[:r0])
    fac_custom = CP.CustomPairFactor(e_fn, de_fn, Dict(:k => 120.0, :r0 => 0.32), pairs;
                                     rc = 0.85, mode = :lfs, n_atoms = 4)
    _check_factor_fd(fac_custom, q, ortho, CP.MD_UNITS; h = 1e-5, force_tol = 1e-7, box_tol = 1e-6, order = 4, check_abs = true)
    _check_factor_fd(fac_custom, q, tric, CP.MD_UNITS; h = 1e-5, force_tol = 1e-7, box_tol = 1e-6, order = 4, check_abs = true)

    # CustomExternalFactor
    ext_e(x, y, z, i, p) = p[:kx] * (x - 0.5)^2 + p[:ky] * sin(2.0 * y) + p[:kz] * (z - 0.4)^2
    ext_g(x, y, z, i, p) = (2.0 * p[:kx] * (x - 0.5), 2.0 * p[:ky] * cos(2.0 * y), 2.0 * p[:kz] * (z - 0.4))
    fac_ext = CP.CustomExternalFactor(ext_e, ext_g, Dict(:kx => 15.0, :ky => 8.0, :kz => 11.0), [1, 3, 4])
    _check_factor_fd(fac_ext, q, ortho, CP.MD_UNITS; h = 1e-5, force_tol = 1e-7, box_tol = 1e-6, order = 4, check_abs = true)
    _check_factor_fd(fac_ext, q, tric, CP.MD_UNITS; h = 1e-5, force_tol = 1e-7, box_tol = 1e-6, order = 4, check_abs = true)

    # CustomManyBodyFactor (centroid spring using minimum-image from atom 1)
    function mb_eval(q_in, box_in, p)
        N_in = size(q_in, 2)
        F_mb = zeros(3, N_in)
        G_mb = box_in isa T.NoBox ? nothing : zeros(3, 3)
        k_mb = p[:k]
        U_mb = 0.0
        for i in 2:N_in
            d = T.unwrap_displacement(box_in, q_in, i, 1) # q_i - q_1
            r2 = T.norm2(d)
            U_mb += 0.5 * k_mb * r2
            F_mb[1, i] -= k_mb * d.x; F_mb[2, i] -= k_mb * d.y; F_mb[3, i] -= k_mb * d.z
            F_mb[1, 1] += k_mb * d.x; F_mb[2, 1] += k_mb * d.y; F_mb[3, 1] += k_mb * d.z
            if G_mb !== nothing
                w = CP._frac_disp(box_in, d)
                CP._outer_add!(G_mb, k_mb, d, w)
            end
        end
        return (U_mb, F_mb, G_mb)
    end
    fac_mb = CP.CustomManyBodyFactor(mb_eval, Dict(:k => 25.0), 4)
    # Verify additive scatter does NOT overwrite pre-existing forces
    F_acc = ones(3, 4) .* 7.0
    F.evaluate!(F_acc, fac_mb, q, F.Context(ortho, CP.MD_UNITS, 4))
    _, F_only = F.total_energy_forces([fac_mb], q, F.Context(ortho, CP.MD_UNITS, 4))
    @test isapprox(F_acc, F_only .+ 7.0; atol = 1e-13)
    _check_factor_fd(fac_mb, q, ortho, CP.MD_UNITS; h = 1e-5, force_tol = 1e-7, box_tol = 1e-6, order = 4, check_abs = true)
    _check_factor_fd(fac_mb, q, tric, CP.MD_UNITS; h = 1e-5, force_tol = 1e-7, box_tol = 1e-6, order = 4, check_abs = true)

    # Registry & SystemContract.box_derivative check
    regs = CP.registered_potentials()
    for k in (:morse, :buckingham, :yukawa, :mie, :fene, :wca, :tabulated_spline)
        @test k in regs
    end
    CP.register_potential!(:test_custom_morse, (p, prs; kwargs...) ->
        CP.CustomPairFactor(:morse, p, prs; kwargs...))
    @test :test_custom_morse in CP.registered_potentials()
    built = CP.build_custom_factor(:test_custom_morse, Dict(:De => 200.0, :a => 10.0, :r0 => 0.3), pairs;
                                   rc = 0.85, mode = :lfs, n_atoms = 4)
    @test built isa CP.CustomPairFactor

    # SystemContract.box_derivative parity
    sys_t = T.System(T.AtomSet(q, zeros(3, 4), fill(12.0, 4), zeros(4)), tric, T.ChemicalTopology(), CP.MD_UNITS, 300.0)
    rs = SC.resolved_from_system(sys_t, F.AbstractFactor[built])
    G_sc = SC.box_derivative(rs, q)
    G_dir = zeros(3, 3)
    RS.accumulate_box_derivative!(G_dir, built, q, F.Context(tric, CP.MD_UNITS, 4))
    @test isapprox(G_sc, G_dir; atol = 1e-13)

    # 200-step Velocity Verlet NVE conservation for CustomPairFactor(:morse)
    fac_nve = CP.CustomPairFactor(:morse, Dict(:De => 350.0, :a => 15.0, :r0 => 0.28), pairs;
                                     rc = 0.85, mode = :lfs, n_atoms = 4)
    nve = _nve_relative_drift([fac_nve], q, fill(12.011, 4), ortho, CP.MD_UNITS, 0.00003, 200)
    @test nve.rel_drift < 1e-5
end

# ===========================================================================
# 3. Tersoff Potential (C graphene/CNT, Si diamond, lambda3, dimer zeta=0, strain, NVE)
# ===========================================================================
@testset "3. TersoffFactor (C & Si, dimer zeta=0, ortho & tric FD, CNT strain, NVE)" begin
    tc = RN.tersoff_carbon(; units = :md)
    ts = RN.tersoff_silicon(; units = :md)
    tc_ev = RN.tersoff_carbon(; units = :ev)
    @test tc.R + tc.D ≈ 0.21
    @test ts.R + ts.D ≈ 0.30

    # Isolated dimer edge case: zeta = 0, n < 1 must be completely finite and exact
    q_dimer = [0.5  0.5 + 0.145;
               0.5  0.5;
               0.5  0.5]
    box_dimer = T.OrthorhombicBox((2.0, 2.0, 2.0))
    f_dimer = RN.TersoffFactor(2, tc)
    res_dimer = _check_factor_fd(f_dimer, q_dimer, box_dimer, CP.MD_UNITS; h = 1e-5, force_tol = 1e-8, box_tol = 1e-7, order = 4)
    @test isfinite(res_dimer.U0) && res_dimer.U0 < 0.0

    # Graphene perturbed sheet (orthorhombic & triclinic)
    gr = RN.build_graphene_sheet(2, 2; bond_length_nm = 0.142, vacuum_z_nm = 1.5)
    N_gr = size(gr.q, 2)
    rng = MersenneTwister(101)
    q_gr = gr.q .+ 0.006 .* randn(rng, 3, N_gr)
    f_gr = RN.TersoffFactor(N_gr, tc)
    _check_factor_fd(f_gr, q_gr, gr.box, CP.MD_UNITS; h = 1e-6, force_tol = 1e-8, box_tol = 1e-7, order = 4)

    # Also verify Metal (eV, A) units (relative < 1e-8, absolute < 5e-8)
    f_gr_ev = RN.TersoffFactor(N_gr, tc_ev)
    box_gr_ev = T.OrthorhombicBox((gr.box.L[1] * 10.0, gr.box.L[2] * 10.0, gr.box.L[3] * 10.0))
    res_ev = _check_factor_fd(f_gr_ev, q_gr .* 10.0, box_gr_ev, CP.EV_ANGSTROM_UNITS;
                              h = 1e-5, force_tol = 1e-8, box_tol = 1e-7, order = 4)
    @test res_ev.max_force_err < 5e-8
    @test res_ev.max_box_err < 1e-7

    # Skew triclinic graphene test box
    Lgr = gr.box.L
    H_gr_skew = [Lgr[1]  0.12    0.05;
                 0.0     Lgr[2]  0.07;
                 0.0     0.0     Lgr[3]]
    tric_gr = T.TriclinicBox(H_gr_skew)
    _check_factor_fd(f_gr, q_gr, tric_gr, CP.MD_UNITS; h = 1e-6, force_tol = 1e-8, box_tol = 1e-7, order = 4)

    # (5,5) CNT force FD & box derivative FD
    cnt = RN.build_cnt(5, 5, 2)
    N_cnt = size(cnt.q, 2)
    q_cnt = cnt.q .+ 0.005 .* randn(rng, 3, N_cnt)
    f_cnt = RN.TersoffFactor(N_cnt, tc)
    _check_factor_fd(f_cnt, q_cnt, cnt.box, CP.MD_UNITS; h = 1e-6, force_tol = 1e-8, box_tol = 1e-7, order = 4)

    # Tersoff Si on perturbed 1x1x1 diamond Si (orthorhombic & triclinic)
    dia = RN.build_diamond_lattice(1, 1, 1; a_nm = 0.5431, element = :Si)
    N_dia = size(dia.q, 2)
    q_dia = dia.q .+ 0.008 .* randn(rng, 3, N_dia)
    f_si = RN.TersoffFactor(N_dia, ts)
    _check_factor_fd(f_si, q_dia, dia.box, CP.MD_UNITS; h = 1e-6, force_tol = 1e-8, box_tol = 1e-7, order = 4)
    a_si = 0.5431
    tric_si = T.TriclinicBox([a_si 0.05 0.02; 0.0 0.98*a_si 0.04; 0.0 0.0 1.02*a_si])
    _check_factor_fd(f_si, q_dia, tric_si, CP.MD_UNITS; h = 1e-6, force_tol = 1e-8, box_tol = 1e-7, order = 4)

    # Nonzero lambda3 test on 4-atom cluster
    tc_l3 = RN.TersoffParams(tc.A, tc.B, tc.lambda1, tc.lambda2, tc.beta, tc.n,
                             tc.c, tc.d, tc.h, 5.0, tc.R, tc.D)
    q_cl = [0.50 0.64 0.43 0.56;
            0.50 0.52 0.62 0.41;
            0.50 0.54 0.53 0.61]
    _check_factor_fd(RN.TersoffFactor(4, tc_l3), q_cl, T.OrthorhombicBox((2.0, 2.0, 2.0)),
                     CP.MD_UNITS; h = 1e-6, force_tol = 1e-8, box_tol = 1e-7, order = 4)

    # Axial CNT strain test (positive Young's modulus: d^2U/deps^2 > 0)
    cnt3 = RN.build_cnt(5, 5, 3)
    N_cnt3 = size(cnt3.q, 2)
    f_cnt3 = RN.TersoffFactor(N_cnt3, tc)
    function cnt_energy_at_strain(eps_z)
        L0 = cnt3.box.L
        Lz_new = L0[3] * (1.0 + eps_z)
        box_s = T.OrthorhombicBox((L0[1], L0[2], Lz_new))
        qs = copy(cnt3.q)
        qs[3, :] .*= (1.0 + eps_z)
        u, _ = F.total_energy_forces([f_cnt3], qs, F.Context(box_s, CP.MD_UNITS, N_cnt3))
        return u
    end
    eps0 = 0.025
    deps = 1e-3
    U_m = cnt_energy_at_strain(eps0 - deps)
    U_0 = cnt_energy_at_strain(eps0)
    U_p = cnt_energy_at_strain(eps0 + deps)
    d2U_deps2 = (U_p - 2.0 * U_0 + U_m) / (deps * deps)
    @test d2U_deps2 > 0.0

    # 200-step Velocity Verlet NVE energy conservation on graphene
    nve_gr = _nve_relative_drift([f_gr], q_gr, gr.masses, gr.box, CP.MD_UNITS, 0.00005, 200)
    @test nve_gr.rel_drift < 1e-5
end

# ===========================================================================
# 4. Stillinger-Weber Si, MorseBondFactor, and MLIPFactor (SymmetryFunctionMLIP)
# ===========================================================================
@testset "4. StillingerWeber, MorseBond, and SymmetryFunctionMLIP" begin
    rng = MersenneTwister(202)

    # Stillinger-Weber Si on 2x2x2 diamond lattice (N=64, tests cell-list O(N) path!)
    sw_p = RN.stillinger_weber_silicon(; units = :md)
    sw_ev = RN.stillinger_weber_silicon(; units = :ev)
    dia2 = RN.build_diamond_lattice(2, 2, 2; a_nm = 0.5431, element = :Si)
    N_dia2 = size(dia2.q, 2)
    q_dia2 = dia2.q .+ 0.01 .* randn(rng, 3, N_dia2)
    f_sw = RN.StillingerWeberFactor(N_dia2, sw_p)
    _check_factor_fd(f_sw, q_dia2, dia2.box, CP.MD_UNITS; h = 1e-5, force_tol = 1e-8, box_tol = 1e-7, order = 4)

    L2 = dia2.box.L
    tric_dia2 = T.TriclinicBox([L2[1] 0.08 0.03; 0.0 0.97*L2[2] 0.06; 0.0 0.0 1.03*L2[3]])
    _check_factor_fd(f_sw, q_dia2, tric_dia2, CP.MD_UNITS; h = 1e-5, force_tol = 1e-8, box_tol = 1e-7, order = 4)

    # Also absolute FD check in Metal (eV, A) units
    f_sw_ev = RN.StillingerWeberFactor(N_dia2, sw_ev)
    _check_factor_fd(f_sw_ev, q_dia2 .* 10.0, T.OrthorhombicBox((L2[1]*10.0, L2[2]*10.0, L2[3]*10.0)),
                     CP.EV_ANGSTROM_UNITS; h = 1e-4, force_tol = 1e-8, box_tol = 1e-7, order = 4, check_abs = true)

    # SW 200-step NVE energy conservation
    nve_sw = _nve_relative_drift([f_sw], q_dia2, dia2.masses, dia2.box, CP.MD_UNITS, 0.0002, 200)
    @test nve_sw.rel_drift < 1e-5

    # MorseBondFactor
    q_m = [0.25  0.42  0.70;
           0.30  0.35  0.75;
           0.40  0.44  0.80]
    f_morse = RN.MorseBondFactor(1, 2, 420.0, 18.0, 0.15)
    _check_factor_fd(f_morse, q_m, T.OrthorhombicBox((1.5, 1.5, 1.5)), CP.MD_UNITS;
                     h = 1e-5, force_tol = 1e-8, box_tol = 1e-7, order = 4, check_abs = true)
    _check_factor_fd(f_morse, q_m, _skew_box(1.5), CP.MD_UNITS;
                     h = 1e-5, force_tol = 1e-8, box_tol = 1e-7, order = 4, check_abs = true)
    nve_m = _nve_relative_drift([f_morse], q_m[:, 1:2], [12.0, 12.0],
                                T.OrthorhombicBox((1.5, 1.5, 1.5)), CP.MD_UNITS, 0.00002, 200)
    @test nve_m.rel_drift < 1e-5

    # SymmetryFunctionMLIP & MLIPFactor
    mlip_model = RN.SymmetryFunctionMLIP(0.45;
                                         g2_eta = [6.0, 18.0],
                                         g2_rs = [0.20, 0.30],
                                         g4_eta = [2.5, 3.0],
                                         g4_zeta = [1.0, 2.0],
                                         g4_lambda = [1.0, -1.0],
                                         weights = [12.0, -8.0, 5.0, -3.5],
                                         bias = -10.0)
    q_mlip = [0.30  0.51  0.35  0.58  0.44;
              0.30  0.32  0.52  0.54  0.41;
              0.30  0.35  0.33  0.36  0.55] .+ 0.01 .* randn(rng, 3, 5)
    f_mlip = RN.MLIPFactor(5, 0.45, mlip_model)
    _check_factor_fd(f_mlip, q_mlip, T.OrthorhombicBox((1.8, 1.8, 1.8)), CP.MD_UNITS;
                     h = 1e-5, force_tol = 1e-8, box_tol = 1e-7, order = 4, check_abs = true)
    _check_factor_fd(f_mlip, q_mlip, _skew_box(1.8), CP.MD_UNITS;
                     h = 1e-5, force_tol = 1e-8, box_tol = 1e-7, order = 4, check_abs = true)

    # Verify MLIPFactor accumulates additively without overwriting F
    F_pre = ones(3, 5) .* 3.5
    F.evaluate!(F_pre, f_mlip, q_mlip, F.Context(T.OrthorhombicBox((1.8, 1.8, 1.8)), CP.MD_UNITS, 5))
    _, F_mlip_only = F.total_energy_forces([f_mlip], q_mlip, F.Context(T.OrthorhombicBox((1.8, 1.8, 1.8)), CP.MD_UNITS, 5))
    @test isapprox(F_pre, F_mlip_only .+ 3.5; atol = 1e-13)

    # MLIP 200-step NVE conservation
    nve_mlip = _nve_relative_drift([f_mlip], q_mlip, fill(12.011, 5),
                                   T.OrthorhombicBox((1.8, 1.8, 1.8)), CP.MD_UNITS, 0.0002, 200)
    @test nve_mlip.rel_drift < 1e-5
end

# ===========================================================================
# 5. Nanostructure Builders Geometric Verification
# ===========================================================================
@testset "5. Nanostructure Builders (Graphene, CNTs, Nanopore, Diamond)" begin
    # Graphene sheet
    gr = RN.build_graphene_sheet(3, 2; bond_length_nm = 0.142, vacuum_z_nm = 3.0)
    N_gr = size(gr.q, 2)
    @test N_gr == 24
    for i in 1:N_gr
        bonds = Float64[]
        for j in 1:N_gr
            i == j && continue
            d = sqrt(T.norm2(T.unwrap_displacement(gr.box, gr.q, j, i)))
            d < 0.18 && push!(bonds, d)
        end
        @test length(bonds) == 3
        @test all(d -> abs(d - 0.142) < 1e-9, bonds)
    end

    # (5,5) armchair & (6,0) zigzag CNTs including periodic z-boundary
    for (n, m, expected_angle) in ((5, 5, pi / 6), (6, 0, 0.0))
        cnt = RN.build_cnt(n, m, 2; bond_length_nm = 0.142, pad_xy_nm = 2.0)
        N_c = size(cnt.q, 2)
        @test N_c == (n == 5 ? 40 : 48)
        @test isapprox(cnt.chiral_angle_rad, expected_angle; atol = 1e-12)
        @test cnt.radius_nm > 0.2
        all_bonds = Float64[]
        for i in 1:N_c
            bonds = Float64[]
            for j in 1:N_c
                i == j && continue
                d = sqrt(T.norm2(T.unwrap_displacement(cnt.box, cnt.q, j, i)))
                d < 0.18 && push!(bonds, d)
            end
            @test length(bonds) == 3
            append!(all_bonds, bonds)
        end
        # Rolled cylinder chord distance is within 1.5e-3 nm of planar 0.142 nm
        @test all(d -> abs(d - 0.142) < 1.5e-3, all_bonds)
    end

    # Nanopore membrane clearance
    np = RN.build_nanopore_membrane(5, 4; pore_radius_nm = 0.45, bond_length_nm = 0.142)
    cx, cy = np.center_xy
    for i in 1:size(np.q, 2)
        r_xy = hypot(np.q[1, i] - cx, np.q[2, i] - cy)
        @test r_xy >= 0.45 - 1e-12
    end
    @test size(np.q, 2) < 4 * 5 * 4

    # Diamond lattice 4-fold coordination
    dia = RN.build_diamond_lattice(2, 2, 2; a_nm = 0.5431, element = :Si)
    N_d = size(dia.q, 2)
    @test N_d == 64
    r_nn = sqrt(3.0) / 4.0 * 0.5431
    for i in 1:N_d
        bonds = Float64[]
        for j in 1:N_d
            i == j && continue
            d = sqrt(T.norm2(T.unwrap_displacement(dia.box, dia.q, j, i)))
            d < 0.28 && push!(bonds, d)
        end
        @test length(bonds) == 4
        @test all(d -> abs(d - r_nn) < 1e-9, bonds)
    end
end

end # testset
