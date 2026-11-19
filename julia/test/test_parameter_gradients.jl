# Tests for ParameterGradients: ∂U/∂p at fixed q by analytic chain rule,
# checked against central finite differences of the *same* energy evaluator.
#
# Policy (same as the rest of the suite): every claim measures a number and
# compares it to an independently justified tolerance. Finite-difference tests
# use h = 1e-6 and require |analytic - FD| <= 1e-5 * max(|analytic|, |FD|, 1).

using FoundationCore
using FoundationCore.Types
using FoundationCore.Factors

# FoundationCore does not yet include ParameterGradients. This harness loads the
# new module against the package's Factors/Types now, and defers to
# FoundationCore.ParameterGradients once it is included there by the package.
module _PGHost
    using ..FoundationCore
    if !isdefined(FoundationCore, :ParameterGradients)
        include(joinpath(@__DIR__, "..", "src", "ParameterGradients.jl"))
    end
end
const PG = isdefined(FoundationCore, :ParameterGradients) ?
           FoundationCore.ParameterGradients : _PGHost.ParameterGradients

const T = FoundationCore.Types
const F = FoundationCore.Factors

const NO_IDS = Dict{Tuple{Int,Symbol},Symbol}()

function base_parameters(factors)
    params = Dict{Symbol,Float64}()
    for (idx, f) in enumerate(factors)
        if f isa F.HarmonicBond
            params[Symbol("bond_k_", idx)] = f.k
            params[Symbol("bond_r0_", idx)] = f.r0
        elseif f isa F.HarmonicAngle
            params[Symbol("angle_ktheta_", idx)] = f.ktheta
            params[Symbol("angle_theta0_", idx)] = f.theta0
        elseif f isa F.PeriodicDihedral
            params[Symbol("dihedral_kphi_", idx)] = f.kphi
            params[Symbol("dihedral_phi0_", idx)] = f.phi0
        elseif f isa F.LennardJonesPair
            params[Symbol("lj_epsilon_", idx)] = f.ε
            params[Symbol("lj_sigma_", idx)] = f.σ
        elseif f isa F.CoulombPair
            params[Symbol("coulomb_qi_", idx)] = f.qi
            params[Symbol("coulomb_qj_", idx)] = f.qj
        else
            error("test helper: unsupported factor $(typeof(f))")
        end
    end
    return params
end

function energy_grad_force(factors, q, ctx; parameters = Dict{Symbol,Float64}(),
                           parameter_ids = NO_IDS)
    Fv = zeros(3, size(q, 2))
    grads = Dict{Symbol,Float64}()
    U = PG.evaluate_parameter_gradients!(Fv, grads, factors, q, ctx;
                                         parameters = parameters,
                                         parameter_ids = parameter_ids)
    return U, grads, Fv
end

function fd_parameter(factors, q, ctx, key, parameters, ids = NO_IDS; h = 1e-6)
    pp = copy(parameters); pm = copy(parameters)
    pp[key] = parameters[key] + h
    pm[key] = parameters[key] - h
    up = energy_grad_force(factors, q, ctx; parameters = pp, parameter_ids = ids)[1]
    um = energy_grad_force(factors, q, ctx; parameters = pm, parameter_ids = ids)[1]
    return (up - um) / (2h)
end

function check_parameter(label, factors, q, ctx, key, parameters, ids = NO_IDS)
    U, grads, Fv = energy_grad_force(factors, q, ctx;
                                    parameters = parameters, parameter_ids = ids)
    @assert haskey(grads, key) "$label: missing gradient key $key"
    analytic = grads[key]
    fd = fd_parameter(factors, q, ctx, key, parameters, ids)
    err = abs(analytic - fd)
    scale = max(abs(analytic), abs(fd), 1.0)
    @assert err <= 1e-5 * scale "$label $key: analytic=$analytic fd=$fd err=$err"
    println("  ", rpad(label, 22), " ", rpad(string(key), 20),
            " ∂U/∂p=", round(analytic, sigdigits = 8),
            "  rel err=", round(err / scale, sigdigits = 3))
    return U, grads, Fv
end

function three_atom_angle_config()
    # i=(0,0,0), j=(1,0,0), k=(0,1,0): a valid 90-degree angle.
    return [0.0 1.0 0.0;
            0.0 0.0 1.0;
            0.0 0.0 0.0]
end

function four_atom_dihedral_config()
    # Non-collinear, non-planar, all support atoms finite.
    return [0.0 1.0 1.0 1.0;
            0.0 0.0 1.0 1.0;
            0.0 0.0 0.0 1.0]
end

function two_atom_config(r::Float64)
    return [0.0 r 0.0;
            0.0 0.0 0.0;
            0.0 0.0 0.0]
end

function test_bond_parameter_gradients()
    q = two_atom_config(1.25)
    ctx = F.Context(T.REDUCED, 2)
    factors = [F.HarmonicBond(1, 2, 3.1, 0.95)]
    params = base_parameters(factors)

    U, grads, _ = check_parameter("bond", factors, q, ctx, :bond_k_1, params)
    check_parameter("bond", factors, q, ctx, :bond_r0_1, params)
    # U = (k/2)(r-r0)^2 with the configured values
    r = 1.25
    @assert abs(U - 0.5 * 3.1 * (r - 0.95)^2) < 1e-14
    @assert abs(grads[:bond_k_1] - 0.5 * (r - 0.95)^2) < 1e-14
    @assert abs(grads[:bond_r0_1] - (-3.1 * (r - 0.95))) < 1e-14
    println("  bond: exact closed forms agree with FD")
    return true
end

function test_angle_parameter_gradients()
    q = three_atom_angle_config()
    ctx = F.Context(T.REDUCED, 3)
    factors = [F.HarmonicAngle(1, 2, 3, 2.4, 1.9)]
    params = base_parameters(factors)

    check_parameter("angle", factors, q, ctx, :angle_ktheta_1, params)
    check_parameter("angle", factors, q, ctx, :angle_theta0_1, params)
    println("  angle: ktheta and theta0 derivatives match FD")
    return true
end

function test_dihedral_parameter_gradients()
    q = four_atom_dihedral_config()
    ctx = F.Context(T.REDUCED, 4)
    factors = [F.PeriodicDihedral(1, 2, 3, 4, 0.9, 3, 0.4)]
    params = base_parameters(factors)

    check_parameter("dihedral", factors, q, ctx, :dihedral_kphi_1, params)
    check_parameter("dihedral", factors, q, ctx, :dihedral_phi0_1, params)
    println("  dihedral: kphi and phi0 derivatives match FD")
    return true
end

function test_lj_parameter_gradients_all_modes()
    q = two_atom_config(1.3)
    ctx = F.Context(T.REDUCED, 2)
    rc = 2.5
    # r = 1.3 is strictly inside every cutoff; no FD straddles rc.
    for mode in (:none, :trunc, :shift, :lfs)
        factors = [F.LennardJonesPair(1, 2, 1.2, 0.9; mode = mode, rc = rc)]
        params = base_parameters(factors)
        check_parameter("LJ $mode", factors, q, ctx, :lj_epsilon_1, params)
        check_parameter("LJ $mode", factors, q, ctx, :lj_sigma_1, params)
    end
    # lj_scale multiplies both energy and parameter derivatives.
    factors_scaled = [F.LennardJonesPair(1, 2, 1.2, 0.9;
                                         mode = :lfs, rc = rc, lj_scale = 1.5)]
    params_scaled = base_parameters(factors_scaled)
    check_parameter("LJ lfs scaled", factors_scaled, q, ctx, :lj_epsilon_1, params_scaled)
    check_parameter("LJ lfs scaled", factors_scaled, q, ctx, :lj_sigma_1, params_scaled)
    println("  LJ: epsilon/sigma derivatives match FD in all four modes")
    return true
end

function test_coulomb_parameter_gradients()
    q = two_atom_config(1.4)
    ctx = F.Context(T.REDUCED, 2)
    factors = [F.CoulombPair(1, 2, 0.5, -0.7; coulomb_scale = 1.0)]
    params = base_parameters(factors)
    check_parameter("coulomb", factors, q, ctx, :coulomb_qi_1, params)
    check_parameter("coulomb", factors, q, ctx, :coulomb_qj_1, params)

    # A zero charge has a well-defined derivative for the *other* charge.
    qz = [F.CoulombPair(1, 2, 0.0, -0.7; coulomb_scale = 2.0)]
    pz = base_parameters(qz)
    Uz, gz, _ = energy_grad_force(qz, q, ctx; parameters = pz)
    check_parameter("coulomb qi=0", qz, q, ctx, :coulomb_qi_1, pz)
    check_parameter("coulomb qi=0", qz, q, ctx, :coulomb_qj_1, pz)
    @assert gz[:coulomb_qj_1] == 0.0
    @assert abs(gz[:coulomb_qi_1] - 2.0 * -0.7 / 1.4) < 1e-14
    println("  coulomb: qi/qj derivatives match FD, including qi = 0")
    return true
end

function test_shared_id_accumulation()
    q = [0.0 1.2 0.0 1.5;
         0.0 0.0 0.0 0.0;
         0.0 0.0 0.0 0.0]
    ctx = F.Context(T.REDUCED, 4)
    factors = [F.HarmonicBond(1, 2, 2.0, 1.0),
               F.HarmonicBond(3, 4, 3.5, 1.1)]
    ids = Dict{Tuple{Int,Symbol},Symbol}(
        (1, :k) => :kb_shared,
        (2, :k) => :kb_shared,
        (1, :r0) => :r0_shared,
        (2, :r0) => :r0_shared,
    )
    params = Dict{Symbol,Float64}(:kb_shared => 2.2, :r0_shared => 1.05)

    U, grads, _ = energy_grad_force(factors, q, ctx;
                                    parameters = params, parameter_ids = ids)
    @assert haskey(grads, :kb_shared) "shared k id missing"
    @assert haskey(grads, :r0_shared) "shared r0 id missing"
    @assert !haskey(grads, :bond_k_1) && !haskey(grads, :bond_k_2)
    @assert !haskey(grads, :bond_r0_1) && !haskey(grads, :bond_r0_2)

    r1, r2 = 1.2, 1.5
    expected_k = 0.5 * ((r1 - 1.05)^2 + (r2 - 1.05)^2)
    expected_r0 = -2.2 * ((r1 - 1.05) + (r2 - 1.05))
    expected_U = 0.5 * 2.2 * ((r1 - 1.05)^2 + (r2 - 1.05)^2)
    @assert abs(grads[:kb_shared] - expected_k) < 1e-13
    @assert abs(grads[:r0_shared] - expected_r0) < 1e-13
    @assert abs(U - expected_U) < 1e-13

    check_parameter("shared", factors, q, ctx, :kb_shared, params, ids)
    check_parameter("shared", factors, q, ctx, :r0_shared, params, ids)
    println("  shared ids: contributions accumulate into one key; per-factor keys absent")
    return true
end

function test_parameter_overrides_match_effective_factors()
    q = [0.0 1.2 0.0 1.5 0.0 1.3;
         0.0 0.0 0.0 0.0 0.0 0.0;
         0.0 0.0 0.0 0.0 0.0 0.0]
    ctx = F.Context(T.REDUCED, 6)
    bond = F.HarmonicBond(1, 2, 2.0, 1.0)
    lj = F.LennardJonesPair(3, 4, 1.1, 0.95; mode = :lfs, rc = 2.5)
    coul = F.CoulombPair(5, 6, 0.4, -0.6)
    factors = [bond, lj, coul]

    # No overrides: energy and force are exactly the existing evaluator's.
    U0, g0, F0 = energy_grad_force(factors, q, ctx)
    Uref, Fref = total_energy_forces(factors, q, ctx)
    @assert U0 == Uref "unmodified energy must match total_energy_forces exactly"
    @assert F0 == Fref "unmodified force must match total_energy_forces exactly"

    # Explicit overrides: the function must evaluate the same effective factors.
    params = Dict{Symbol,Float64}(:bond_k_1 => 4.2,
                                  :lj_sigma_2 => 1.05,
                                  :coulomb_qj_3 => -0.9)
    U1, g1, F1 = energy_grad_force(factors, q, ctx; parameters = params)
    effective = [F.HarmonicBond(1, 2, 4.2, 1.0),
                 F.LennardJonesPair(3, 4, 1.1, 1.05; mode = :lfs, rc = 2.5),
                 F.CoulombPair(5, 6, 0.4, -0.9)]
    Ueff, Feff = total_energy_forces(effective, q, ctx)
    @assert U1 == Ueff "override energy must match effective-factor energy exactly"
    @assert F1 == Feff "override force must match effective-factor force exactly"
    check_parameter("override", factors, q, ctx, :bond_k_1, base_parameters(factors))
    check_parameter("override", factors, q, ctx, :lj_sigma_2, params)
    check_parameter("override", factors, q, ctx, :coulomb_qj_3, params)
    println("  parameter overrides: energy/F equal effective factor list; derivatives match FD")
    return true
end

function test_fixed_cutoff_semantics()
    ctx = F.Context(T.REDUCED, 2)
    rc = 2.5

    # Beyond a fixed cutoff the evaluator returns exactly 0.0; both parameter
    # partials are then exactly 0.0 and no rc gradient key exists.
    q_out = two_atom_config(3.0)
    for mode in (:trunc, :shift, :lfs)
        factors = [F.LennardJonesPair(1, 2, 1.0, 1.0; mode = mode, rc = rc)]
        params = base_parameters(factors)
        U, grads, Fv = energy_grad_force(factors, q_out, ctx; parameters = params)
        @assert U == 0.0 "$mode beyond cutoff energy must be exactly zero"
        @assert grads[:lj_epsilon_1] == 0.0
        @assert grads[:lj_sigma_1] == 0.0
        @assert maximum(abs, Fv) == 0.0
        @assert !haskey(grads, :lj_rc_1) "rc is fixed and must not be differentiated"
    end

    # Inside the cutoff, :lfs includes its cutoff corrections; verify by FD and
    # by comparing with the plain 12-6 derivative (they must differ here).
    q_in = two_atom_config(1.4)
    factors = [F.LennardJonesPair(1, 2, 1.0, 1.0; mode = :lfs, rc = rc)]
    params = base_parameters(factors)
    U, grads, _ = check_parameter("cutoff lfs", factors, q_in, ctx, :lj_epsilon_1, params)
    s6 = (1.0 / 1.4)^6
    plain = 4.0 * (s6 * s6 - s6)
    @assert abs(grads[:lj_epsilon_1] - plain) > 1e-6
    println("  fixed cutoffs: zero beyond rc, :lfs corrections retained inside")
    return true
end

function test_singular_configurations_have_no_derivative()
    ctx2 = F.Context(T.REDUCED, 2)
    ctx3 = F.Context(T.REDUCED, 3)
    ctx4 = F.Context(T.REDUCED, 4)

    overlap = zeros(3, 2)
    collinear3 = [0.0 1.0 2.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
    collinear4 = [0.0 1.0 2.0 3.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]

    cases = [
        ("bond overlap", [F.HarmonicBond(1, 2, 1.0, 1.0)], overlap, ctx2,
         (:bond_k_1, :bond_r0_1)),
        ("angle collinear", [F.HarmonicAngle(1, 2, 3, 1.0, 1.0)], collinear3, ctx3,
         (:angle_ktheta_1, :angle_theta0_1)),
        ("dihedral collinear", [F.PeriodicDihedral(1, 2, 3, 4, 1.0, 2, 0.0)],
         collinear4, ctx4, (:dihedral_kphi_1, :dihedral_phi0_1)),
        ("LJ overlap", [F.LennardJonesPair(1, 2, 1.0, 1.0)], overlap, ctx2,
         (:lj_epsilon_1, :lj_sigma_1)),
        ("Coulomb overlap", [F.CoulombPair(1, 2, 1.0, -1.0)], overlap, ctx2,
         (:coulomb_qi_1, :coulomb_qj_1)),
    ]
    for (label, factors, q, ctx, keys) in cases
        U, grads, Fv = energy_grad_force(factors, q, ctx)
        @assert !isfinite(U) "$label: singular energy must be nonfinite, got $U"
        for key in keys
            @assert !haskey(grads, key) "$label: derivative $key must be absent"
        end
        # The existing evaluator also scatters no force at a singular geometry.
        Uref, Fref = total_energy_forces(factors, q, ctx)
        @assert Fv == Fref
        @assert !isfinite(Uref)
    end
    println("  singular configurations: nonfinite U, no derivative keys, F matches evaluator")
    return true
end

function test_workspace_reuse()
    q = two_atom_config(1.25)
    ctx = F.Context(T.REDUCED, 2)
    ws = PG.ParameterGradientWorkspace(ctx)
    @assert size(ws.F) == (3, 2)

    grads = Dict{Symbol,Float64}(:stale => 99.0)
    bond = [F.HarmonicBond(1, 2, 3.1, 0.95)]
    U1 = PG.evaluate_parameter_gradients!(ws, grads, bond, q, ctx)
    @assert !haskey(grads, :stale) "gradient dict must be cleared for reuse"
    @assert haskey(grads, :bond_k_1)
    @assert ws.F == total_energy_forces(bond, q, ctx)[2]

    # Second call with a different factor list: no keys from the first survive.
    coul = [F.CoulombPair(1, 2, 0.5, -0.7)]
    U2 = PG.evaluate_parameter_gradients!(ws, grads, coul, q, ctx)
    @assert !haskey(grads, :bond_k_1)
    @assert haskey(grads, :coulomb_qi_1)
    @assert ws.F == total_energy_forces(coul, q, ctx)[2]
    @assert U2 == total_energy_forces(coul, q, ctx)[1]

    # Workspace-held shared ids are applied by the workspace overload.
    ws2 = PG.ParameterGradientWorkspace(ctx;
                                        parameter_ids = Dict{Tuple{Int,Symbol},Symbol}(
                                            (1, :k) => :k_ws))
    grads2 = Dict{Symbol,Float64}()
    params = Dict{Symbol,Float64}(:k_ws => 2.5)
    U3 = PG.evaluate_parameter_gradients!(ws2, grads2, bond, q, ctx; parameters = params)
    @assert haskey(grads2, :k_ws) && !haskey(grads2, :bond_k_1)
    @assert abs(U3 - 0.5 * 2.5 * (1.25 - 0.95)^2) < 1e-14
    println("  workspace: F/gradient buffers reused and cleared; shared ids honored")
    return true
end

function test_unsupported_factor_is_rejected()
    q = [0.0 1.0; 0.0 0.0; 0.0 0.0]
    ctx = F.Context(T.REDUCED, 2)
    factors = [F.BipartiteHarmonicGroup([1], [2], 1.0)]
    ok = false
    try
        energy_grad_force(factors, q, ctx)
    catch e
        ok = e isa ArgumentError
    end
    @assert ok "BipartiteHarmonicGroup must be rejected explicitly"
    println("  unsupported factor: explicit ArgumentError")
    return true
end

function _run_all()
    tests = [
        ("bond_parameter_gradients", test_bond_parameter_gradients),
        ("angle_parameter_gradients", test_angle_parameter_gradients),
        ("dihedral_parameter_gradients", test_dihedral_parameter_gradients),
        ("lj_parameter_gradients", test_lj_parameter_gradients_all_modes),
        ("coulomb_parameter_gradients", test_coulomb_parameter_gradients),
        ("shared_id_accumulation", test_shared_id_accumulation),
        ("parameter_overrides", test_parameter_overrides_match_effective_factors),
        ("fixed_cutoff_semantics", test_fixed_cutoff_semantics),
        ("singular_configurations", test_singular_configurations_have_no_derivative),
        ("workspace_reuse", test_workspace_reuse),
        ("unsupported_factor", test_unsupported_factor_is_rejected),
    ]
    for (name, fn) in tests
        println("TEST ", name)
        fn()
    end
    println("")
    println("ALL PARAMETER-GRADIENT TESTS PASSED")
end

_run_all()
