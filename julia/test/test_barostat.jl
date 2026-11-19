"""Focused checks for the isotropic NPT Monte-Carlo barostat and pressure math.

Run:
    cd julia
    JULIA_DEPOT_PATH=... julia --project=. test/test_barostat.jl
"""

using FoundationCore
using FoundationCore.MonteCarloBarostat
using LinearAlgebra, Printf, Random, Statistics, Test

const T = FoundationCore.Types
const Dyn = FoundationCore.Dynamics
const Fac = FoundationCore.Factors
const Con = FoundationCore.Constraints
const RS = FoundationCore.ResolvedSystem
const SC = FoundationCore.SystemContract
const SA = FoundationCore.Sampling
const SLJ = FoundationCore.StreamedLJ
const PME = FoundationCore.PME
const WB = FoundationCore.WaterBox

const ZERO_ENERGY = (q, box) -> 0.0
const ZERO_VIRIAL = (q, box) -> 0.0

# ---------------------------------------------------------------------------
# 1. analytic proposal / Jacobian algebra & exact detailed balance
# ---------------------------------------------------------------------------

function test_log_volume_detailed_balance()
    beta, pressure, delta_u = 1.7, 0.23, -0.41
    v0, dlnv, n = 12.0, 0.08, 7
    v1 = v0 * exp(dlnv)

    comp = log_acceptance_components(beta, pressure, delta_u, v0, v1, n;
                                     proposal = :log_volume)
    @test isapprox(comp.energy_term, -beta * delta_u; atol = 1e-14)
    @test isapprox(comp.pv_term, -beta * pressure * (v1 - v0); atol = 1e-14)
    @test isapprox(comp.jacobian_term, n * dlnv; atol = 1e-14)
    @test isapprox(comp.proposal_term, dlnv; atol = 1e-14)
    @test isapprox(comp.total, -beta * (delta_u + pressure * (v1 - v0)) + (n + 1) * dlnv;
                   atol = 1e-14)

    loga = log_acceptance(beta, pressure, delta_u, v0, v1, n)
    @test isapprox(loga, comp.total; atol = 1e-14)
    @test isapprox(propose_log_volume(propose_log_volume(log(v0), dlnv), -dlnv),
                   log(v0); atol = 1e-14)
    # The Jacobian is N+1, not N, for log-volume proposals.
    @test !isapprox(loga, -beta * (delta_u + pressure * (v1 - v0)) + n * dlnv;
                    atol = 1e-6)

    # Linear-volume proposal: q(V'->V)/q(V->V') = 1, so only N_mol * log(V1/V0).
    comp_lin = log_acceptance_components(beta, pressure, delta_u, v0, v1, n;
                                         proposal = :linear_volume)
    @test comp_lin.proposal_term == 0.0
    @test isapprox(comp_lin.total, -beta * (delta_u + pressure * (v1 - v0)) + n * dlnv;
                   atol = 1e-14)
    @test log_proposal_ratio(:linear_volume, v0, -0.5) == -Inf

    # Exact detailed balance check in (q, V) space for non-zero potential energy:
    # pi(q0, V0) * q(V0 -> V1) * alpha(0 -> 1) == pi(q1, V1) * q(V1 -> V0) * alpha(1 -> 0)
    # where moving from q0 to q1 scales COMs by s = (V1/V0)^(1/3) (Jacobian V1/V0 per unit).
    U(q, box) = 0.35 * sum(abs2, q) + 0.1 * T.box_volume(box)^2
    rng = MersenneTwister(19)
    mols = [(1, 2, 3), (4, 5, 6)]
    masses = [16.0, 1.0, 1.0, 16.0, 1.0, 1.0]
    q0 = randn(rng, 3, 6)
    box0 = T.OrthorhombicBox((3.0, 3.0, 3.0))
    for prop in (:log_volume, :linear_volume)
        dln = 0.07
        s = exp(dln / 3)
        box1 = T.OrthorhombicBox(ntuple(i -> box0.L[i] * s, 3))
        V0, V1 = T.box_volume(box0), T.box_volume(box1)
        q1 = similar(q0)
        n_units = scale_coordinates!(q1, q0, s; groups = [collect(m) for m in mols],
                                     masses = masses)
        @test n_units == 2
        dU_fwd = U(q1, box1) - U(q0, box0)
        dU_rev = U(q0, box0) - U(q1, box1)
        la_fwd = log_acceptance(beta, pressure, dU_fwd, V0, V1, n_units; proposal = prop)
        la_rev = log_acceptance(beta, pressure, dU_rev, V1, V0, n_units; proposal = prop)
        acc_fwd = min(1.0, exp(la_fwd))
        acc_rev = min(1.0, exp(la_rev))
        # Target ratio pi(q1, V1) * |dq1/dq0| / pi(q0, V0) in V-space:
        # |dq1/dq0| = (V1/V0)^n_units; proposal ratio q(V1->V0)/q(V0->V1) = exp(log_proposal_ratio)
        q_ratio = exp(log_proposal_ratio(prop, V0, V1))
        lhs = acc_fwd
        rhs = exp(-beta * (dU_fwd + pressure * (V1 - V0))) * (V1 / V0)^n_units * q_ratio * acc_rev
        @test isapprox(lhs, rhs; rtol = 1e-13)
    end
    return true
end

# ---------------------------------------------------------------------------
# 2. scaling, proposal reversal, molecular COM preservation
# ---------------------------------------------------------------------------

function test_scaling_and_reversal()
    rng = MersenneTwister(77)
    n = 6
    q0 = rand(rng, 3, n) .+ 0.5
    L0 = 5.0
    box0 = T.OrthorhombicBox((L0, L0, L0))
    barostat = VolumeMoveBarostat(1.0, 0.0, 0.3)  # P = 0

    scaled_energy(q, box) = sum(sin.(2pi .* (q ./ collect(box.L))))

    found_atomic = false
    for seed in 1:200
        q = copy(q0)
        box = box0
        r = step!(barostat, q, box, scaled_energy, MersenneTwister(seed))
        box = r.box
        if r.accepted && r.delta_log_volume > 0
            @test r.n_translational_units == n
            @test isapprox(scaled_coordinates(q, box),
                           scaled_coordinates(q0, box0); atol = 1e-12)
            @test isapprox(r.new_energy, r.old_energy; atol = 1e-10)
            s_inv = exp(-r.delta_log_volume / 3)
            q_back = q .* s_inv
            box_back = T.OrthorhombicBox(ntuple(i -> box.L[i] * s_inv, 3))
            @test isapprox(q_back, q0; atol = 1e-12)
            @test isapprox(T.box_volume(box_back), T.box_volume(box0); rtol = 1e-12)
            found_atomic = true
            break
        end
    end
    @test found_atomic

    # Molecular COM scaling on rigid TIP3P waters: internal bond lengths and
    # angles must stay invariant to machine precision, while COMs scale by s.
    w = WB.tip3p_water(8; density = 1.0, seed = 5)
    q_w0 = copy(w.positions)
    box_w0 = w.box
    b_mol = VolumeMoveBarostat(1.0, 0.0, 0.25;
                               molecules = w.molecules,
                               constraints = w.constraints,
                               masses = w.masses)
    com0 = molecular_centers_of_mass(q_w0, w.molecules, w.masses)
    bond_len(qq, i, j) = norm(qq[:, j] .- qq[:, i])
    found_mol = false
    for seed in 1:200
        q = copy(q_w0)
        r = step!(b_mol, q, box_w0, ZERO_ENERGY, MersenneTwister(seed))
        if r.accepted && abs(r.delta_log_volume) > 0.02
            @test r.n_translational_units == length(w.molecules)
            @test isapprox(r.log_jacobian, length(w.molecules) * r.delta_log_volume; atol = 1e-14)
            s = exp(r.delta_log_volume / 3)
            com1 = molecular_centers_of_mass(q, w.molecules, w.masses)
            @test isapprox(com1, com0 .* s; atol = 1e-13)
            # Intramolecular geometry unchanged
            for (o, h1, h2) in w.molecules
                @test isapprox(bond_len(q, o, h1), WB.TIP3P_PARAMETERS.r_OH; atol = 1e-12)
                @test isapprox(bond_len(q, o, h2), WB.TIP3P_PARAMETERS.r_OH; atol = 1e-12)
                @test isapprox(bond_len(q, h1, h2), WB.R_HH; atol = 1e-12)
            end
            @test maximum(abs, Con.constraint_residuals(w.constraints, q)) < 1e-12
            # Reversal
            q_back = similar(q)
            scale_coordinates!(q_back, q, 1.0 / s;
                               groups = [collect(m) for m in w.molecules],
                               masses = w.masses)
            @test isapprox(q_back, q_w0; atol = 1e-12)
            # wrap_molecules! preserves bond lengths and shifts only by integer box lengths
            q_wrap = copy(q) .+ 1.7
            wrap_molecules!(q_wrap, r.box, w.molecules, w.masses)
            @test maximum(abs, Con.constraint_residuals(w.constraints, q_wrap)) < 1e-12
            com_wrap = molecular_centers_of_mass(q_wrap, w.molecules, w.masses)
            for d in 1:3
                @test all(x -> 0.0 <= x < r.box.L[d], com_wrap[d, :])
            end
            found_mol = true
            break
        end
    end
    @test found_mol
    return true
end

# ---------------------------------------------------------------------------
# 3. seeded replay determinism
# ---------------------------------------------------------------------------

function test_seeded_replay()
    function trace(seed)
        rng = MersenneTwister(seed)
        q = rand(rng, 3, 5)
        box = T.OrthorhombicBox((2.0, 2.0, 2.0))
        b = VolumeMoveBarostat(1.0, 0.3, 0.2)
        volumes = Float64[T.box_volume(box)]
        for _ in 1:800
            r = step!(b, q, box, ZERO_ENERGY, rng)
            box = r.box
            push!(volumes, T.box_volume(box))
        end
        return (volumes = volumes, q = copy(q), L = box.L)
    end
    a, b, c = trace(931), trace(931), trace(932)
    @test a.volumes == b.volumes
    @test a.q == b.q
    @test a.L == b.L
    @test a.volumes != c.volumes
    @test length(unique(a.volumes)) > 10
    return true
end

# ---------------------------------------------------------------------------
# 4. ideal-gas analytic volume distribution (atomic & rigid-molecule)
# ---------------------------------------------------------------------------

function _make_rigid_trimer_positions(nmol::Int, L::Float64; seed::Int = 1)
    rng = MersenneTwister(seed)
    q = zeros(3, 3 * nmol)
    m = repeat([16.0, 1.0, 1.0], nmol)
    triples = Tuple{Int,Int,Float64}[]
    mols = NTuple{3,Int}[]
    r_oh = 0.1
    ang = deg2rad(104.52)
    r_hh = 2 * r_oh * sin(ang / 2)
    for k in 0:(nmol - 1)
        o, h1, h2 = 3k + 1, 3k + 2, 3k + 3
        push!(mols, (o, h1, h2))
        push!(triples, (o, h1, r_oh), (o, h2, r_oh), (h1, h2, r_hh))
        com = rand(rng, 3) .* L
        q[:, o] .= com
        q[:, h1] .= com .+ [r_oh, 0.0, 0.0]
        q[:, h2] .= com .+ [r_oh * cos(ang), r_oh * sin(ang), 0.0]
    end
    cs = Con.CoupledConstraintSet(3 * nmol, triples)
    return q, m, mols, cs
end

function test_ideal_gas_analytic()
    # (a) Atomic ideal gas: N = 8
    n = 8
    beta, pressure, dlnv = 1.0, 1.0, 0.25
    exact_mean = (n + 1) / (beta * pressure)          # 9.0
    exact_var = (n + 1) / (beta * pressure)^2         # 9.0

    rng = MersenneTwister(11)
    L = exact_mean^(1 / 3)
    box = T.OrthorhombicBox((L, L, L))
    q = zeros(3, n)
    b = VolumeMoveBarostat(beta, pressure, dlnv)
    for _ in 1:2000
        box = step!(b, q, box, ZERO_ENERGY, rng).box
    end
    volumes = Float64[]
    for i in 1:40000
        box = step!(b, q, box, ZERO_ENERGY, rng).box
        i % 2 == 0 && push!(volumes, T.box_volume(box))
    end
    sv = SA.summarize_observable(volumes; n_blocks = 20)
    @test abs(sv.mean - exact_mean) <= 4 * sv.se
    @test abs(sv.var - exact_var) <= 4 * SA.block_variance_se(volumes; n_blocks = 20)
    eos = n ./ (beta .* volumes)
    seos = SA.summarize_observable(eos; n_blocks = 20)
    @test abs(seos.mean - pressure) <= 4 * seos.se

    # (b) Rigid-molecule ideal gas: N_mol = 6 rigid 3-atom molecules (18 atoms, 18 constraints).
    # Exact distribution is Gamma(N_mol + 1, 1/(beta P)) -> mean = 7.0 (NOT 19.0!).
    nmol = 6
    exact_mol_mean = (nmol + 1) / (beta * pressure)   # 7.0
    exact_mol_var = (nmol + 1) / (beta * pressure)^2  # 7.0
    for (prop, step_sz) in ((:log_volume, 0.30), (:linear_volume, 2.5))
        q_m, m_m, mols, cs = _make_rigid_trimer_positions(nmol, exact_mol_mean^(1 / 3); seed = 7)
        box_m = T.OrthorhombicBox((exact_mol_mean^(1 / 3), exact_mol_mean^(1 / 3), exact_mol_mean^(1 / 3)))
        b_m = VolumeMoveBarostat(beta, pressure, step_sz;
                                 molecules = mols, constraints = cs, masses = m_m,
                                 proposal = prop)
        rng_m = MersenneTwister(prop === :log_volume ? 31 : 32)
        for _ in 1:3000
            box_m = step!(b_m, q_m, box_m, ZERO_ENERGY, rng_m).box
        end
        vols_m = Float64[]
        for i in 1:40000
            box_m = step!(b_m, q_m, box_m, ZERO_ENERGY, rng_m).box
            i % 2 == 0 && push!(vols_m, T.box_volume(box_m))
        end
        sm = SA.summarize_observable(vols_m; n_blocks = 20)
        seos_m = SA.summarize_observable(nmol ./ (beta .* vols_m); n_blocks = 20)
        @printf("  rigid ideal gas (%s, N_mol=%d): <V>=%.3f±%.3f (exact %.1f)  var=%.3f (exact %.1f)  <N_mol/(βV)>=%.3f±%.3f\n",
                string(prop), nmol, sm.mean, sm.se, exact_mol_mean, sm.var, exact_mol_var,
                seos_m.mean, seos_m.se)
        @test abs(sm.mean - exact_mol_mean) <= 4 * sm.se
        @test abs(sm.var - exact_mol_var) <= 4 * SA.block_variance_se(vols_m; n_blocks = 20)
        @test abs(seos_m.mean - pressure) <= 4 * seos_m.se
        @test maximum(abs, Con.constraint_residuals(cs, q_m)) < 1e-12
    end

    # (c) Rigid-molecule ideal gas with dynamics (6 DOFs/molecule): check
    # dynamical constraint virial cancels rotational KE and <P_inst> == P_target.
    q_d, m_d, mols_d, cs_d = _make_rigid_trimer_positions(nmol, exact_mol_mean^(1 / 3); seed = 9)
    rng_d = MersenneTwister(41)
    p_d = randn(rng_d, 3, 3 * nmol) .* sqrt.(reshape(m_d, 1, :) ./ beta)
    Con.project_velocities!(q_d, p_d, m_d, cs_d; tolerance = 1e-12)
    box_d = T.OrthorhombicBox((exact_mol_mean^(1 / 3), exact_mol_mean^(1 / 3), exact_mol_mean^(1 / 3)))
    zero_f = (qq, bx) -> zeros(3, size(qq, 2))
    model_dyn = NPTModel(ZERO_ENERGY; virial = ZERO_VIRIAL, forces = zero_f,
                         masses = m_d, constraints = cs_d, molecules = mols_d,
                         pressure_mode = :dynamical)
    model_com = NPTModel(ZERO_ENERGY; virial = ZERO_VIRIAL, forces = zero_f,
                         masses = m_d, constraints = cs_d, molecules = mols_d,
                         pressure_mode = :molecular_com)
    baro_d = NPTBarostat(beta, pressure, 0.25, model_dyn)
    max_mode_diff = 0.0
    ps_d = Float64[]; vs_d = Float64[]
    ξ = zeros(3, 3 * nmol)
    c1 = exp(-5.0 * 0.005); c2 = sqrt((1.0 - c1^2) / beta)
    for sweep in 1:14000
        for _ in 1:2
            WB._drift!(q_d, p_d, m_d, mols_d, 0.0025)
            for j in 1:(3 * nmol), d in 1:3
                ξ[d, j] = sqrt(m_d[j]) * randn(rng_d)
            end
            Con.project_velocities!(q_d, ξ, m_d, cs_d; tolerance = 1e-12)
            p_d .= c1 .* p_d .+ c2 .* ξ
            WB._drift!(q_d, p_d, m_d, mols_d, 0.0025)
        end
        r = npt_step!(baro_d, q_d, p_d, box_d, rng_d)
        box_d = r.box
        wrap_molecules!(q_d, box_d, mols_d, m_d)
        if sweep > 1000 && sweep % 2 == 0
            pc_dyn = pressure_components(model_dyn, q_d, p_d, box_d)
            pc_com = pressure_components(model_com, q_d, p_d, box_d)
            max_mode_diff = max(max_mode_diff, abs(pc_dyn.total - pc_com.total))
            push!(ps_d, pc_dyn.total)
            push!(vs_d, T.box_volume(box_d))
        end
    end
    @test max_mode_diff < 1e-12
    sp = SA.summarize_observable(ps_d; n_blocks = 20)
    sv_d = SA.summarize_observable(vs_d; n_blocks = 20)
    @printf("  rigid ideal gas hybrid: <P>=%.4f±%.4f (target %.1f)  <V>=%.3f±%.3f (exact %.1f)  |P_dyn-P_com|=%.2e\n",
            sp.mean, sp.se, pressure, sv_d.mean, sv_d.se, exact_mol_mean, max_mode_diff)
    @test abs(sp.mean - pressure) <= 4 * sp.se
    @test abs(sv_d.mean - exact_mol_mean) <= 4 * sv_d.se
    return true
end

function test_jacobian_control()
    n, beta, pressure = 8, 1.0, 1.0
    exact_mean = (n + 1) / (beta * pressure)
    function control(seed, dlnv)
        rng = MersenneTwister(seed)
        V = exact_mean
        out = Float64[]
        for i in 1:20000
            d = (2rand(rng) - 1) * dlnv
            newV = V * exp(d)
            newV > 0 || break
            log_ratio = log_acceptance(beta, pressure, 0.0, V, newV, n) - (n + 1) * d
            if log(rand(rng)) < min(0.0, log_ratio)
                V = newV
            end
            i > 1000 && i % 2 == 0 && push!(out, V)
        end
        return out
    end
    c_big = control(11, 0.30)
    c_small = control(11, 0.10)
    @test all(isfinite, c_big) && all(>(0), c_big)
    @test mean(c_big) < 0.5 * exact_mean
    @test mean(c_big) < mean(c_small)
    @test mean(c_big) / exact_mean < 0.1

    # Wrong-DOF control on rigid trimers: using N_atoms = 18 instead of N_mol = 6
    # gives <V> = 19 instead of 7.
    nmol = 6
    natoms = 18
    rng = MersenneTwister(55)
    V = 7.0
    wrong_vols = Float64[]
    for i in 1:20000
        d = (2rand(rng) - 1) * 0.25
        newV = V * exp(d)
        log_ratio = log_acceptance(beta, pressure, 0.0, V, newV, natoms)
        if log(rand(rng)) < min(0.0, log_ratio)
            V = newV
        end
        i > 2000 && i % 2 == 0 && push!(wrong_vols, V)
    end
    @test abs(mean(wrong_vols) - (natoms + 1)) < 1.0
    @test mean(wrong_vols) > 2.0 * (nmol + 1)
    return true
end

# ---------------------------------------------------------------------------
# 5. solvable harmonic crystal detailed balance (atomic, rigid-molecule, HarmonicBond)
# ---------------------------------------------------------------------------

function test_harmonic_crystal_detailed_balance()
    # (a) Atomic 3D Einstein crystal with quadratic bulk-volume term:
    #     U(q, h) = 0.5 * K_bulk * (V - V0)^2 / V0 + 0.5 * k * sum_i |q_i - h s0_i|^2
    # Exact marginal in V: Gaussian around V0 * (1 - P / K_bulk), var = V0 / (beta * K_bulk)
    n = 8
    beta = 1.0
    pressure = 1.5
    K_bulk = 25.0
    V0 = 10.0
    k_spring = 40.0
    s0 = Matrix{Float64}(undef, 3, n)
    idx = 0
    for ix in 0:1, iy in 0:1, iz in 0:1
        idx += 1
        s0[:, idx] .= ((ix + 0.5) / 2, (iy + 0.5) / 2, (iz + 0.5) / 2)
    end

    function crystal_energy(q, box)
        V = T.box_volume(box)
        L = collect(box.L)
        u = 0.5 * K_bulk * (V - V0)^2 / V0
        @inbounds for j in 1:n, d in 1:3
            dq = q[d, j] - L[d] * s0[d, j]
            u += 0.5 * k_spring * dq * dq
        end
        return u
    end
    function crystal_forces(q, box)
        L = collect(box.L)
        F = Matrix{Float64}(undef, 3, n)
        @inbounds for j in 1:n, d in 1:3
            F[d, j] = -k_spring * (q[d, j] - L[d] * s0[d, j])
        end
        return F
    end
    function crystal_box_deriv(q, box)
        # G = ∂U/∂h|_s = K_bulk * (V - V0)/V0 * V * h^{-T} + k * Σ (q - h s0)(s - s0)^T
        V = T.box_volume(box)
        L = collect(box.L)
        G = zeros(3, 3)
        dU_dV = K_bulk * (V - V0) / V0
        for a in 1:3
            G[a, a] = dU_dV * (V / L[a])
            for j in 1:n
                sa = q[a, j] / L[a]
                G[a, a] += k_spring * (q[a, j] - L[a] * s0[a, j]) * (sa - s0[a, j])
            end
        end
        return G
    end

    # Finite-difference verification of crystal_box_deriv at fixed fractional s = q ./ L
    q_test = (V0^(1 / 3)) .* s0 .+ 0.03 .* randn(MersenneTwister(1), 3, n)
    box_test = T.OrthorhombicBox((V0^(1 / 3), V0^(1 / 3), V0^(1 / 3)))
    G_an = crystal_box_deriv(q_test, box_test)
    s_frac = scaled_coordinates(q_test, box_test)
    eps_fd = 1e-6
    for a in 1:3
        Lp = collect(box_test.L); Lp[a] += eps_fd
        Lm = collect(box_test.L); Lm[a] -= eps_fd
        g_fd = (crystal_energy(s_frac .* Lp, T.OrthorhombicBox(Lp)) -
                crystal_energy(s_frac .* Lm, T.OrthorhombicBox(Lm))) / (2 * eps_fd)
        @test isapprox(G_an[a, a], g_fd; rtol = 1e-8)
    end

    exact_V_mean = V0 * (1.0 - pressure / K_bulk)  # 9.4
    exact_V_var = V0 / (beta * K_bulk)             # 0.4
    exact_U_harm = 1.5 * n / beta                  # 12.0

    rng = MersenneTwister(88)
    L_init = exact_V_mean^(1 / 3)
    box = T.OrthorhombicBox((L_init, L_init, L_init))
    q = L_init .* s0 .+ 0.05 .* randn(rng, 3, n)
    p = randn(rng, 3, n) ./ sqrt(beta)
    model = NPTModel(crystal_energy; box_derivative = crystal_box_deriv,
                     forces = crystal_forces, masses = 1.0)
    baro = NPTBarostat(beta, pressure, 0.08, model)

    # BAOAB on the crystal potential + MC volume moves
    dt = 0.02
    gamma = 5.0
    c1 = exp(-gamma * dt)
    c2 = sqrt((1.0 - c1^2) / beta)
    Vs = Float64[]; Ps = Float64[]; Uh = Float64[]; Ts = Float64[]
    for sweep in 1:16000
        F = crystal_forces(q, box)
        for _ in 1:3
            p .+= (0.5 * dt) .* F
            q .+= (0.5 * dt) .* p
            p .= c1 .* p .+ c2 .* randn(rng, 3, n)
            q .+= (0.5 * dt) .* p
            F = crystal_forces(q, box)
            p .+= (0.5 * dt) .* F
        end
        r = npt_step!(baro, q, p, box, rng)
        box = r.box
        if sweep > 1500 && sweep % 2 == 0
            V = T.box_volume(box)
            push!(Vs, V)
            push!(Ps, r.pressure_new)
            L = collect(box.L)
            push!(Uh, 0.5 * k_spring * sum(abs2, q .- L .* s0))
            push!(Ts, sum(abs2, p) / (3 * n))
        end
    end
    sv = SA.summarize_observable(Vs; n_blocks = 20)
    sp = SA.summarize_observable(Ps; n_blocks = 20)
    su = SA.summarize_observable(Uh; n_blocks = 20)
    st = SA.summarize_observable(Ts; n_blocks = 20)
    @printf("  atomic harmonic crystal: <V>=%.4f±%.4f (%.2f)  var=%.4f (%.2f)  <Uh>=%.3f±%.3f (%.1f)  <P>=%.4f±%.4f (%.2f)\n",
            sv.mean, sv.se, exact_V_mean, sv.var, exact_V_var, su.mean, su.se, exact_U_harm,
            sp.mean, sp.se, pressure)
    @test abs(sv.mean - exact_V_mean) <= 4 * sv.se
    @test abs(sv.var - exact_V_var) <= 4 * SA.block_variance_se(Vs; n_blocks = 20)
    @test abs(su.mean - exact_U_harm) <= 4 * su.se
    @test abs(st.mean - 1.0 / beta) <= 4 * st.se
    @test abs(sp.mean - pressure) <= 4 * sp.se

    # (b) Rigid-molecule harmonic crystal: 8 rigid trimers (24 atoms, 24 constraints),
    # COMs tethered to lattice sites s0 with bulk term U_bulk(V).
    # Exact <V> is STILL exact_V_mean (9.4) because N_mol = 8 (NOT N_atoms = 24!).
    nmol = 8
    q_r, m_r, mols_r, cs_r = _make_rigid_trimer_positions(nmol, L_init; seed = 3)
    # Place COMs at lattice sites
    com_init = molecular_centers_of_mass(q_r, mols_r, m_r)
    for (alpha, mol) in enumerate(mols_r)
        shift = L_init .* s0[:, alpha] .- com_init[:, alpha]
        for i in mol
            q_r[:, i] .+= shift
        end
    end

    function mol_crystal_energy(q, box)
        V = T.box_volume(box)
        L = collect(box.L)
        com = molecular_centers_of_mass(q, mols_r, m_r)
        u = 0.5 * K_bulk * (V - V0)^2 / V0
        @inbounds for alpha in 1:nmol, d in 1:3
            dq = com[d, alpha] - L[d] * s0[d, alpha]
            u += 0.5 * k_spring * dq * dq
        end
        return u
    end
    function mol_crystal_forces(q, box)
        L = collect(box.L)
        com = molecular_centers_of_mass(q, mols_r, m_r)
        F = zeros(3, 3 * nmol)
        @inbounds for (alpha, mol) in enumerate(mols_r)
            M = sum(m_r[i] for i in mol)
            for i in mol, d in 1:3
                F[d, i] = -k_spring * (com[d, alpha] - L[d] * s0[d, alpha]) * (m_r[i] / M)
            end
        end
        return F
    end
    function mol_crystal_box_deriv(q, box)
        V = T.box_volume(box)
        L = collect(box.L)
        com = molecular_centers_of_mass(q, mols_r, m_r)
        G = zeros(3, 3)
        dU_dV = K_bulk * (V - V0) / V0
        for a in 1:3
            G[a, a] = dU_dV * (V / L[a])
            for (alpha, mol) in enumerate(mols_r)
                M = sum(m_r[i] for i in mol)
                dU_dR = k_spring * (com[a, alpha] - L[a] * s0[a, alpha])
                for i in mol
                    G[a, a] += dU_dR * (m_r[i] / M) * (q[a, i] / L[a])
                end
                G[a, a] -= dU_dR * s0[a, alpha]
            end
        end
        return G
    end

    rng_r = MersenneTwister(99)
    p_r = randn(rng_r, 3, 3 * nmol) .* sqrt.(reshape(m_r, 1, :) ./ beta)
    Con.project_velocities!(q_r, p_r, m_r, cs_r; tolerance = 1e-12)
    box_r = T.OrthorhombicBox((L_init, L_init, L_init))
    model_rd = NPTModel(mol_crystal_energy; box_derivative = mol_crystal_box_deriv,
                        forces = mol_crystal_forces, masses = m_r,
                        constraints = cs_r, molecules = mols_r,
                        pressure_mode = :dynamical)
    model_rc = NPTModel(mol_crystal_energy; box_derivative = mol_crystal_box_deriv,
                        forces = mol_crystal_forces, masses = m_r,
                        constraints = cs_r, molecules = mols_r,
                        pressure_mode = :molecular_com)
    baro_r = NPTBarostat(beta, pressure, 0.08, model_rd)
    ξr = zeros(3, 3 * nmol)
    Vs_r = Float64[]; Ps_r = Float64[]; Ts_r = Float64[]
    max_diff_r = 0.0
    dt_r = 0.005; c1_r = exp(-5.0 * dt_r); c2_r = sqrt((1.0 - c1_r^2) / beta)
    for sweep in 1:14000
        for _ in 1:3
            Fext = mol_crystal_forces(q_r, box_r)
            λ = Con.constraint_multipliers(q_r, m_r, cs_r, Fext)
            Ftan = Fext .+ Con.constraint_forces(q_r, m_r, cs_r, λ)
            p_r .+= (0.5 * dt_r) .* Ftan
            Con.project_velocities!(q_r, p_r, m_r, cs_r; tolerance = 1e-12)
            WB._drift!(q_r, p_r, m_r, mols_r, 0.5 * dt_r)
            for j in 1:(3 * nmol), d in 1:3
                ξr[d, j] = sqrt(m_r[j]) * randn(rng_r)
            end
            Con.project_velocities!(q_r, ξr, m_r, cs_r; tolerance = 1e-12)
            p_r .= c1_r .* p_r .+ c2_r .* ξr
            WB._drift!(q_r, p_r, m_r, mols_r, 0.5 * dt_r)
            Fext = mol_crystal_forces(q_r, box_r)
            λ = Con.constraint_multipliers(q_r, m_r, cs_r, Fext)
            Ftan = Fext .+ Con.constraint_forces(q_r, m_r, cs_r, λ)
            p_r .+= (0.5 * dt_r) .* Ftan
            Con.project_velocities!(q_r, p_r, m_r, cs_r; tolerance = 1e-12)
        end
        r = npt_step!(baro_r, q_r, p_r, box_r, rng_r)
        box_r = r.box
        if sweep > 1500 && sweep % 2 == 0
            pc_d = pressure_components(model_rd, q_r, p_r, box_r)
            pc_c = pressure_components(model_rc, q_r, p_r, box_r)
            max_diff_r = max(max_diff_r, abs(pc_d.total - pc_c.total))
            push!(Vs_r, T.box_volume(box_r))
            push!(Ps_r, pc_d.total)
            K_tot = 0.5 * sum(p_r[d, j]^2 / m_r[j] for j in 1:(3 * nmol) for d in 1:3)
            push!(Ts_r, 2.0 * K_tot / (6 * nmol))
        end
    end
    @test max_diff_r < 1e-12
    svr = SA.summarize_observable(Vs_r; n_blocks = 20)
    spr = SA.summarize_observable(Ps_r; n_blocks = 20)
    str = SA.summarize_observable(Ts_r; n_blocks = 20)
    @printf("  rigid harmonic crystal: <V>=%.4f±%.4f (%.2f)  var=%.4f (%.2f)  <T>=%.4f±%.4f (%.1f)  <P>=%.4f±%.4f (%.2f)\n",
            svr.mean, svr.se, exact_V_mean, svr.var, exact_V_var,
            str.mean, str.se, 1.0 / beta, spr.mean, spr.se, pressure)
    @test abs(svr.mean - exact_V_mean) <= 4 * svr.se
    @test abs(svr.var - exact_V_var) <= 4 * SA.block_variance_se(Vs_r; n_blocks = 20)
    @test abs(str.mean - 1.0 / beta) <= 4 * str.se
    @test abs(spr.mean - pressure) <= 4 * spr.se

    # (c) Periodic Factors.HarmonicBond 3x3x3 lattice using SystemContract.box_derivative
    nx = 3; nb = nx^3
    r0_lat = 1.5
    L_lat = nx * r0_lat
    q_lat = Matrix{Float64}(undef, 3, nb)
    id_of(ix, iy, iz) = (mod(iz, nx) * nx + mod(iy, nx)) * nx + mod(ix, nx) + 1
    bonds = Fac.AbstractFactor[]
    for ix in 0:(nx - 1), iy in 0:(nx - 1), iz in 0:(nx - 1)
        i = id_of(ix, iy, iz)
        q_lat[:, i] .= ((ix + 0.5) * r0_lat, (iy + 0.5) * r0_lat, (iz + 0.5) * r0_lat)
        push!(bonds, Fac.HarmonicBond(i, id_of(ix + 1, iy, iz), 30.0, r0_lat))
        push!(bonds, Fac.HarmonicBond(i, id_of(ix, iy + 1, iz), 30.0, r0_lat))
        push!(bonds, Fac.HarmonicBond(i, id_of(ix, iy, iz + 1), 30.0, r0_lat))
    end
    make_rs(bx) = RS.ResolvedSystem(; masses = ones(nb), charges = zeros(nb),
                                    names = fill(:Ar, nb), box = bx,
                                    topology = T.ChemicalTopology(),
                                    factors = bonds, units = T.REDUCED)
    rs0 = make_rs(T.OrthorhombicBox((L_lat, L_lat, L_lat)))
    # Check virial_from_box_derivative vs -dU/da|_s finite difference on perturbed lattice
    rng_l = MersenneTwister(12)
    q_l = q_lat .+ 0.05 .* randn(rng_l, 3, nb)
    box_l = T.OrthorhombicBox((L_lat * 0.98, L_lat * 0.98, L_lat * 0.98))
    G_sc = SC.box_derivative(make_rs(box_l), q_l)
    W_sc = virial_from_box_derivative(G_sc, box_l)
    eps_s = 1e-6
    U_sc(a) = SC.energy(make_rs(T.OrthorhombicBox(ntuple(d -> box_l.L[d] * a, 3))), q_l .* a)
    W_fd = -(U_sc(1 + eps_s) - U_sc(1 - eps_s)) / (2 * eps_s)
    @test isapprox(W_sc, W_fd; rtol = 1e-8)

    # Short NPT run on the HarmonicBond lattice at P = 0.5, beta = 2.0
    beta_l = 2.0; P_l = 0.5
    p_l = randn(rng_l, 3, nb) ./ sqrt(beta_l)
    model_l = NPTModel((qq, bx) -> SC.energy(make_rs(bx), qq);
                       box_derivative = (qq, bx) -> SC.box_derivative(make_rs(bx), qq),
                       masses = 1.0)
    baro_l = NPTBarostat(beta_l, P_l, 0.05, model_l)
    Ps_l = Float64[]; Ts_l = Float64[]
    dt_l = 0.015; c1_l = exp(-5.0 * dt_l); c2_l = sqrt((1.0 - c1_l^2) / beta_l)
    for sweep in 1:8000
        rs_cur = make_rs(box_l)
        sys_cur = Dyn.HarmonicSystem(bonds, ones(nb), Fac.Context(box_l, T.REDUCED, nb))
        F_l = zeros(3, nb)
        Dyn.forces!(sys_cur, q_l, F_l)
        for _ in 1:3
            p_l .+= (0.5 * dt_l) .* F_l
            q_l .+= (0.5 * dt_l) .* p_l
            p_l .= c1_l .* p_l .+ c2_l .* randn(rng_l, 3, nb)
            q_l .+= (0.5 * dt_l) .* p_l
            Dyn.forces!(sys_cur, q_l, F_l)
            p_l .+= (0.5 * dt_l) .* F_l
        end
        r = npt_step!(baro_l, q_l, p_l, box_l, rng_l)
        box_l = r.box
        if sweep > 1000 && sweep % 2 == 0
            push!(Ps_l, r.pressure_new)
            push!(Ts_l, sum(abs2, p_l) / (3 * nb))
        end
    end
    spl = SA.summarize_observable(Ps_l; n_blocks = 20)
    stl = SA.summarize_observable(Ts_l; n_blocks = 20)
    @printf("  HarmonicBond lattice: <P>=%.4f±%.4f (target %.2f)  <T>=%.4f±%.4f (target %.2f)\n",
            spl.mean, spl.se, P_l, stl.mean, stl.se, 1.0 / beta_l)
    @test abs(spl.mean - P_l) <= 4 * spl.se
    @test abs(stl.mean - 1.0 / beta_l) <= 4 * stl.se
    return true
end

# ---------------------------------------------------------------------------
# 6. pressure decomposition & virial consistency on interacting rigid water
# ---------------------------------------------------------------------------

function test_pressure_decomposition_and_virial()
    w = WB.tip3p_water(27; density = 1.0, seed = 4)
    rc = 0.44
    a = WB.PME_ALPHA_TARGET / rc
    g = (24, 24, 24)
    q = copy(w.positions)
    m = w.masses
    cs = w.constraints
    N = size(q, 2)
    ws = SLJ.LJWorkspace(q, w.box, rc)

    function water_U(qq, bx)
        u_lj, _, _ = SLJ.evaluate!(ws, qq, w.epsilon, w.sigma, bx;
                                   rcut = rc, mode = :lfs, topology = w.lj_topology)
        plan = PME.PMEPlan(qq, collect(bx.L); alpha = a, rcut = rc, grid = g,
                           order = 6, prefactor = WB.COULOMB_PREFACTOR)
        u_pme = PME.pme_energy(plan, qq, w.charges; corrections = w.pme_corrections)
        return u_lj + u_pme
    end
    function water_F(qq, bx)
        _, f_lj, _ = SLJ.evaluate!(ws, qq, w.epsilon, w.sigma, bx;
                                   rcut = rc, mode = :lfs, topology = w.lj_topology)
        plan = PME.PMEPlan(qq, collect(bx.L); alpha = a, rcut = rc, grid = g,
                           order = 6, prefactor = WB.COULOMB_PREFACTOR)
        F_pme = zeros(3, N)
        PME.pme_energy_forces!(F_pme, plan, qq, w.charges; corrections = w.pme_corrections)
        return f_lj .+ F_pme
    end
    function water_G(qq, bx)
        _, _, _, g_lj = SLJ.evaluate_with_box_derivative!(ws, qq, w.epsilon, w.sigma, bx;
                                                          rcut = rc, mode = :lfs,
                                                          topology = w.lj_topology)
        plan = PME.PMEPlan(qq, collect(bx.L); alpha = a, rcut = rc, grid = g,
                           order = 6, prefactor = WB.COULOMB_PREFACTOR)
        g_pme = PME.pme_box_derivative(plan, qq, w.charges; corrections = w.pme_corrections)
        return g_lj .+ g_pme
    end

    # Finite difference of COM-scaled energy vs W_int + W_c^static
    G0 = water_G(q, w.box)
    W_int = virial_from_box_derivative(G0, w.box)
    F0 = water_F(q, w.box)
    λ_stat = static_constraint_multipliers(q, m, cs, F0)
    W_c_stat = Con.constraint_virial(q, m, cs, λ_stat)

    eps_s = 1e-6
    groups = [collect(mol) for mol in w.molecules]
    qp = similar(q); qm = similar(q)
    scale_coordinates!(qp, q, 1.0 + eps_s; groups = groups, masses = m)
    scale_coordinates!(qm, q, 1.0 - eps_s; groups = groups, masses = m)
    box_p = T.OrthorhombicBox(ntuple(d -> w.box.L[d] * (1.0 + eps_s), 3))
    box_m = T.OrthorhombicBox(ntuple(d -> w.box.L[d] * (1.0 - eps_s), 3))
    dU_da_fd = (water_U(qp, box_p) - water_U(qm, box_m)) / (2 * eps_s)
    @test isapprox(-(W_int + W_c_stat), dU_da_fd; rtol = 1e-6)

    # At zero momenta, dynamical multipliers equal static multipliers
    λ_dyn0 = dynamical_constraint_multipliers(q, zeros(3, N), m, cs, F0)
    @test isapprox(λ_dyn0, λ_stat; rtol = 1e-13)

    # At non-zero constrained momenta, :dynamical and :molecular_com pressure
    # components must produce identical total pressure while decomposing
    # kinetic, interaction, and constraint pieces distinctly.
    rng = MersenneTwister(17)
    p = randn(rng, 3, N) .* sqrt.(reshape(m .* WB.KB * 300.0, 1, :))
    Con.project_velocities!(q, p, m, cs; tolerance = 1e-12)

    m_dyn = NPTModel(water_U; box_derivative = water_G, forces = water_F,
                     masses = m, constraints = cs, molecules = w.molecules,
                     pressure_mode = :dynamical)
    m_com = NPTModel(water_U; box_derivative = water_G, forces = water_F,
                     masses = m, constraints = cs, molecules = w.molecules,
                     pressure_mode = :molecular_com)
    pc_dyn = pressure_components(m_dyn, q, p, w.box)
    pc_com = pressure_components(m_com, q, p, w.box)
    @test isapprox(pc_dyn.kinetic + pc_dyn.interaction + pc_dyn.constraint, pc_dyn.total; atol = 1e-14)
    @test isapprox(pc_com.kinetic + pc_com.interaction + pc_com.constraint, pc_com.total; atol = 1e-14)
    @test isapprox(pc_dyn.interaction, pc_com.interaction; atol = 1e-14)
    @test isapprox(pc_dyn.total, pc_com.total; rtol = 1e-12)
    @test abs(pc_dyn.constraint - pc_com.constraint) > 1e-2  # centrifugal term is non-trivial!
    @printf("  water virial check: dU/da_fd=%.4f  -(W_int+W_c)=%.4f  |P_dyn-P_com|=%.2e\n",
            dU_da_fd, -(W_int + W_c_stat), abs(pc_dyn.total - pc_com.total))
    return true
end

# ---------------------------------------------------------------------------
# 7. typed refusals: constraints, box geometry, missing virial/forces/masses
# ---------------------------------------------------------------------------

function test_refusals()
    # Atomic scaling with constraints -> ConstraintIncompatibilityError
    @test_throws ConstraintIncompatibilityError VolumeMoveBarostat(
        1.0, 1.0, 0.1; constraints = [(1, 2, 1.0)])
    @test_throws ConstraintIncompatibilityError VolumeMoveBarostat(
        1.0, 1.0, 0.1; constraints = [(1, 2, 1.0)], molecules = [(1, 2)],
        masses = [1.0, 1.0], scale_mode = :atomic)
    model = NPTModel(ZERO_ENERGY; virial = ZERO_VIRIAL,
                     forces = (q, b) -> zeros(3, size(q, 2)), masses = 1.0)
    @test_throws ConstraintIncompatibilityError NPTBarostat(
        1.0, 1.0, 0.1, model; constraints = [(1, 2, 1.0)])

    # Inter-molecular constraint -> ConstraintIncompatibilityError
    @test_throws ConstraintIncompatibilityError VolumeMoveBarostat(
        1.0, 1.0, 0.1;
        molecules = [(1, 2), (3, 4)],
        constraints = [(2, 3, 1.0)],
        masses = ones(4))
    @test_throws ConstraintIncompatibilityError validate_molecular_constraints(
        4, [(1, 2)], [(2, 3, 1.0)])
    # Overlapping or empty molecules -> ConstraintIncompatibilityError
    @test_throws ConstraintIncompatibilityError VolumeMoveBarostat(
        1.0, 1.0, 0.1; molecules = [(1, 2), (2, 3)], masses = ones(3))
    @test_throws ConstraintIncompatibilityError VolumeMoveBarostat(
        1.0, 1.0, 0.1; molecules = [Int[]], masses = ones(2))

    # Multi-atom molecules without masses -> MissingMassError
    @test_throws MissingMassError VolumeMoveBarostat(
        1.0, 1.0, 0.1; molecules = [(1, 2)], constraints = [(1, 2, 1.0)])

    # True constraint violation -> ConstraintIncompatibilityError during step!
    q_broken = [0.0 0.5; 0.0 0.0; 0.0 0.0]
    b_mol = VolumeMoveBarostat(1.0, 1.0, 0.1;
                               molecules = [(1, 2)],
                               constraints = [(1, 2, 0.1)],
                               masses = [1.0, 1.0])
    @test_throws ConstraintIncompatibilityError step!(
        b_mol, q_broken, T.OrthorhombicBox((2.0, 2.0, 2.0)), ZERO_ENERGY, MersenneTwister(1))

    # Molecule straddling PBC is unwrapped and does NOT trigger false violation:
    q_straddle = [0.0 1.9; 0.0 0.0; 0.0 0.0]
    r_straddle = step!(b_mol, q_straddle, T.OrthorhombicBox((2.0, 2.0, 2.0)), ZERO_ENERGY, MersenneTwister(1))
    @test r_straddle.accepted !== nothing

    # Unsupported geometry: TriclinicBox and NoBox
    H = [2.0 0.4 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0]
    tri = T.TriclinicBox(H)
    @test_throws T.UnsupportedGeometry VolumeMoveBarostat(1.0, 1.0, 0.1; box = tri)
    @test_throws T.UnsupportedGeometry VolumeMoveBarostat(1.0, 1.0, 0.1; box = T.NoBox())
    @test_throws T.UnsupportedGeometry VolumeMoveBarostat(
        1.0, 1.0, 0.1; molecules = [(1, 2)], constraints = [(1, 2, 1.0)],
        masses = ones(2), box = tri)
    b_ok = VolumeMoveBarostat(1.0, 1.0, 0.1)
    @test_throws T.UnsupportedGeometry step!(b_ok, zeros(3, 1), T.NoBox(),
                                             ZERO_ENERGY, MersenneTwister(1))
    @test_throws T.UnsupportedGeometry step!(b_ok, zeros(3, 1), tri,
                                             ZERO_ENERGY, MersenneTwister(1))
    @test_throws T.UnsupportedGeometry virial_from_box_derivative(zeros(3, 3), tri)
    @test_throws T.UnsupportedGeometry configurational_pressure(
        model, zeros(3, 1), zeros(3, 1), tri)

    # Missing virial / forces / masses
    n = 4
    q = zeros(3, n); p = zeros(3, n)
    box = T.OrthorhombicBox((2.0, 2.0, 2.0))
    no_virial = NPTModel(ZERO_ENERGY; virial = nothing, masses = 1.0)
    @test_throws MissingVirialError NPTBarostat(1.0, 1.0, 0.1, no_virial)
    @test_throws MissingVirialError configurational_pressure(no_virial, q, p, box)
    no_mass = NPTModel(ZERO_ENERGY; virial = ZERO_VIRIAL, masses = nothing)
    @test_throws MissingMassError NPTBarostat(1.0, 1.0, 0.1, no_mass)
    no_forces_con = NPTModel(ZERO_ENERGY; virial = ZERO_VIRIAL, masses = 1.0,
                             constraints = [(1, 2, 1.0)], molecules = [(1, 2)])
    @test_throws MissingVirialError NPTBarostat(1.0, 1.0, 0.1, no_forces_con)
    @test_throws MissingVirialError pressure_components(no_forces_con, q, p, box)
    @test_throws ArgumentError NPTModel(ZERO_ENERGY; virial = ZERO_VIRIAL,
                                        box_derivative = (qq, bb) -> zeros(3, 3))

    # Empty constraint containers are accepted
    @test VolumeMoveBarostat(1.0, 1.0, 0.1; constraints = ()) isa VolumeMoveBarostat
    @test VolumeMoveBarostat(1.0, 1.0, 0.1; constraints = nothing) isa VolumeMoveBarostat
    return true
end

# ---------------------------------------------------------------------------
# 8. monatomic hybrid dynamics-loop composition
# ---------------------------------------------------------------------------

function test_hybrid_pressure_mean()
    n = 16
    beta, pressure, dlnv = 1.0, 1.0, 0.20
    rng = MersenneTwister(2025)
    L = ((n + 1) / (beta * pressure))^(1 / 3)
    box = T.OrthorhombicBox((L, L, L))
    q = rand(rng, 3, n) .* L
    p = randn(rng, 3, n) ./ sqrt(beta)
    F = zeros(3, n)
    system = Dyn.HarmonicSystem(Fac.AbstractFactor[], ones(n),
                                Fac.Context(T.NoBox(), T.REDUCED, n))
    model = NPTModel(ZERO_ENERGY; virial = ZERO_VIRIAL, masses = 1.0)
    barostat = NPTBarostat(beta, pressure, dlnv, model)

    pressures = Float64[]
    volumes = Float64[]
    kinetics = Float64[]
    for i in 1:32000
        for _ in 1:3
            Dyn.langevin_baoab_step!(q, p, F, system, 0.002, 20.0, beta, rng)
        end
        r = npt_step!(barostat, q, p, box, rng)
        box = r.box
        for d in 1:3
            @views q[d, :] .= mod.(q[d, :], box.L[d])
        end
        if i > 2000 && i % 2 == 0
            push!(pressures, r.pressure_new)
            push!(volumes, T.box_volume(box))
            push!(kinetics, 0.5 * sum(abs2, p))
        end
    end
    sp = SA.summarize_observable(pressures; n_blocks = 20)
    sv = SA.summarize_observable(volumes; n_blocks = 20)
    sk = SA.summarize_observable(kinetics; n_blocks = 20)
    @test abs(sp.mean - pressure) <= 4 * sp.se
    @test abs(sv.mean - (n + 1) / (beta * pressure)) <= 4 * sv.se
    @test abs(sk.mean - 3n / (2beta)) <= 4 * sk.se
    return true
end

# ---------------------------------------------------------------------------
# 9. interacting rigid molecular LJ fluid NPT (high-ESS volume/density/EOS)
# ---------------------------------------------------------------------------

function test_interacting_rigid_molecular_lj_npt()
    # 64 rigid trimers (192 atoms, 192 constraints) on a 4x4x4 lattice; O-site LJ
    # interaction (epsilon=1, sigma=1, rcut=1.65, :lfs) in reduced units.
    nx = 4
    nmol = nx^3
    L0 = 5.6
    spacing = L0 / nx
    r_oh = 0.28
    ang = deg2rad(104.52)
    r_hh = 2 * r_oh * sin(ang / 2)
    m = repeat([16.0, 1.0, 1.0], nmol)
    eps = repeat([1.0, 0.0, 0.0], nmol)
    sig = repeat([1.0, 1.0, 1.0], nmol)
    q = zeros(3, 3 * nmol)
    mols = NTuple{3,Int}[]
    triples = Tuple{Int,Int,Float64}[]
    excl = Tuple{Int,Int}[]
    k = 0
    for ix in 0:(nx - 1), iy in 0:(nx - 1), iz in 0:(nx - 1)
        o, h1, h2 = 3k + 1, 3k + 2, 3k + 3
        push!(mols, (o, h1, h2))
        push!(triples, (o, h1, r_oh), (o, h2, r_oh), (h1, h2, r_hh))
        push!(excl, (o, h1), (o, h2), (h1, h2))
        com = [(ix + 0.5) * spacing, (iy + 0.5) * spacing, (iz + 0.5) * spacing]
        q[:, o] .= com
        q[:, h1] .= com .+ [r_oh, 0.0, 0.0]
        q[:, h2] .= com .+ [r_oh * cos(ang), r_oh * sin(ang), 0.0]
        k += 1
    end
    cs = Con.CoupledConstraintSet(3 * nmol, triples)
    topo = SLJ.PairTopology(excluded = excl)
    rcut = 1.65
    box = T.OrthorhombicBox((L0, L0, L0))
    ws = SLJ.LJWorkspace(q, box, rcut)

    lj_U(qq, bx) = SLJ.evaluate!(ws, qq, eps, sig, bx; rcut = rcut, mode = :lfs, topology = topo)[1]
    lj_F(qq, bx) = copy(SLJ.evaluate!(ws, qq, eps, sig, bx; rcut = rcut, mode = :lfs, topology = topo)[2])
    lj_G(qq, bx) = copy(SLJ.evaluate_with_box_derivative!(ws, qq, eps, sig, bx;
                                                          rcut = rcut, mode = :lfs, topology = topo)[4])

    T_target = 1.5
    beta = 1.0 / T_target
    rng = MersenneTwister(314)
    p = randn(rng, 3, 3 * nmol) .* sqrt.(reshape(m .* T_target, 1, :))
    Con.project_velocities!(q, p, m, cs; tolerance = 1e-12)
    ξ = zeros(3, 3 * nmol)
    dof = 6 * nmol

    function run_at_pressure(P_target, q_init, p_init, box_init)
        qq = copy(q_init); pp = copy(p_init); bx = box_init
        model = NPTModel(lj_U; box_derivative = lj_G, forces = lj_F,
                         masses = m, constraints = cs, molecules = mols)
        baro = NPTBarostat(beta, P_target, 0.06, model)
        dt = 0.004; gamma = 2.0
        c1 = exp(-gamma * dt); c2 = sqrt((1.0 - c1^2) / beta)
        Vs = Float64[]; rhos = Float64[]; Us = Float64[]; Ts = Float64[]
        Ps = Float64[]; Pkins = Float64[]; Pints = Float64[]; Pcons = Float64[]
        accs = Bool[]
        for sweep in 1:4500
            for _ in 1:3
                Fext = lj_F(qq, bx)
                λ = Con.constraint_multipliers(qq, m, cs, Fext)
                Ftan = Fext .+ Con.constraint_forces(qq, m, cs, λ)
                pp .+= (0.5 * dt) .* Ftan
                Con.project_velocities!(qq, pp, m, cs; tolerance = 1e-12)
                WB._drift!(qq, pp, m, mols, 0.5 * dt)
                for j in 1:(3 * nmol), d in 1:3
                    ξ[d, j] = sqrt(m[j]) * randn(rng)
                end
                Con.project_velocities!(qq, ξ, m, cs; tolerance = 1e-12)
                pp .= c1 .* pp .+ c2 .* ξ
                WB._drift!(qq, pp, m, mols, 0.5 * dt)
                Fext = lj_F(qq, bx)
                λ = Con.constraint_multipliers(qq, m, cs, Fext)
                Ftan = Fext .+ Con.constraint_forces(qq, m, cs, λ)
                pp .+= (0.5 * dt) .* Ftan
                Con.project_velocities!(qq, pp, m, cs; tolerance = 1e-12)
            end
            wrap_molecules!(qq, bx, mols, m)
            r = npt_step!(baro, qq, pp, bx, rng)
            bx = r.box
            push!(accs, r.accepted)
            if sweep > 1000
                V = T.box_volume(bx)
                push!(Vs, V)
                push!(rhos, nmol / V)
                push!(Us, r.accepted ? r.new_energy : r.old_energy)
                K = 0.5 * sum(pp[d, j]^2 / m[j] for j in 1:(3 * nmol) for d in 1:3)
                push!(Ts, 2.0 * K / dof)
                pc = r.pressure_components_new
                push!(Ps, pc.total)
                push!(Pkins, pc.kinetic); push!(Pints, pc.interaction); push!(Pcons, pc.constraint)
            end
        end
        return (q = qq, p = pp, box = bx, acc = mean(accs),
                V = SA.summarize_observable(Vs),
                rho = SA.summarize_observable(rhos),
                U = SA.summarize_observable(Us),
                T = SA.summarize_observable(Ts),
                P = SA.summarize_observable(Ps),
                Pkin = SA.summarize_observable(Pkins),
                Pint = SA.summarize_observable(Pints),
                Pcon = SA.summarize_observable(Pcons))
    end

    r_low = run_at_pressure(0.8, q, p, box)
    r_high = run_at_pressure(2.0, r_low.q, r_low.p, r_low.box)
    for (lbl, res, P_t) in (("P=0.8", r_low, 0.8), ("P=2.0", r_high, 2.0))
        @printf("  rigid LJ (%s, acc=%.2f): V=%.2f±%.2f (ess=%.0f)  rho=%.4f±%.4f  U=%.2f±%.2f  T=%.3f±%.3f  P=%.4f±%.4f [kin=%.3f int=%.3f con=%.3f]\n",
                lbl, res.acc, res.V.mean, res.V.se, res.V.ess, res.rho.mean, res.rho.se,
                res.U.mean, res.U.se, res.T.mean, res.T.se, res.P.mean, res.P.se,
                res.Pkin.mean, res.Pint.mean, res.Pcon.mean)
        @test abs(res.T.mean - T_target) <= 4 * res.T.se
        @test abs(res.P.mean - P_t) <= 4 * res.P.se
        @test res.V.ess > 20
        @test isfinite(res.U.mean) && isfinite(res.rho.mean)
    end
    # Compression response: higher pressure gives higher density and smaller volume
    @test r_high.rho.mean > r_low.rho.mean + 4 * hypot(r_high.rho.se, r_low.rho.se)
    @test r_high.V.mean < r_low.V.mean
    return true
end

# ---------------------------------------------------------------------------
# 10. interacting rigid TIP3P water NPT validation (WaterBox + PME + StreamedLJ)
# ---------------------------------------------------------------------------

function test_interacting_tip3p_water_npt()
    w = WB.tip3p_water(64; density = 1.0, seed = 6)
    rc = 0.54
    a = WB.PME_ALPHA_TARGET / rc
    g = (24, 24, 24)
    order = 6
    T_target = 300.0
    beta = 1.0 / (WB.KB * T_target)
    P_bar = 1.0
    P_int = internal_pressure_from_bar(P_bar, WB.TIP3P_UNITS)

    q = copy(w.positions)
    m = w.masses
    cs = w.constraints
    mols = w.molecules
    N = size(q, 2)
    rng = MersenneTwister(123)
    p = zeros(3, N)
    for j in 1:N, d in 1:3
        p[d, j] = sqrt(m[j] * WB.KB * T_target) * randn(rng)
    end
    Con.project_velocities!(q, p, m, cs; tolerance = 1e-12)

    box = w.box
    ws = SLJ.LJWorkspace(q, box, rc)
    plan_cache = Dict{NTuple{3,Float64}, PME.PMEPlan}()
    get_plan(bx) = get!(plan_cache, bx.L) do
        PME.PMEPlan(q, collect(bx.L); alpha = a, rcut = rc, grid = g,
                    order = order, prefactor = WB.COULOMB_PREFACTOR)
    end
    F_buf = zeros(3, N)
    G_pme = zeros(3, 3)

    function energy_fn(qq, bx)
        u_lj, _, _ = SLJ.evaluate!(ws, qq, w.epsilon, w.sigma, bx;
                                   rcut = rc, mode = :lfs, topology = w.lj_topology)
        u_pme = PME.pme_energy(get_plan(bx), qq, w.charges; corrections = w.pme_corrections)
        return u_lj + u_pme
    end
    function forces_fn(qq, bx)
        fill!(F_buf, 0.0)
        _, f_lj, _ = SLJ.evaluate!(ws, qq, w.epsilon, w.sigma, bx;
                                   rcut = rc, mode = :lfs, topology = w.lj_topology)
        PME.pme_energy_forces!(F_buf, get_plan(bx), qq, w.charges; corrections = w.pme_corrections)
        F_buf .+= f_lj
        return copy(F_buf)
    end
    function box_deriv_fn(qq, bx)
        _, _, _, glj = SLJ.evaluate_with_box_derivative!(ws, qq, w.epsilon, w.sigma, bx;
                                                         rcut = rc, mode = :lfs,
                                                         topology = w.lj_topology)
        PME.pme_box_derivative!(G_pme, get_plan(bx), qq, w.charges; corrections = w.pme_corrections)
        return glj .+ G_pme
    end

    model = NPTModel(energy_fn; box_derivative = box_deriv_fn, forces = forces_fn,
                     masses = m, constraints = cs, molecules = mols)
    barostat = NPTBarostat(beta, P_int, 0.04, model)

    ξ = zeros(3, N)
    F_ext = zeros(3, N)
    F_tan = zeros(3, N)
    function update_tan_forces!(qq, bx)
        fill!(F_ext, 0.0)
        u_lj, f_lj, _ = SLJ.evaluate!(ws, qq, w.epsilon, w.sigma, bx;
                                      rcut = rc, mode = :lfs, topology = w.lj_topology)
        u_pme = PME.pme_energy_forces!(F_ext, get_plan(bx), qq, w.charges;
                                       corrections = w.pme_corrections)
        F_ext .+= f_lj
        λ = Con.constraint_multipliers(qq, m, cs, F_ext)
        Fc = Con.constraint_forces(qq, m, cs, λ)
        F_tan .= F_ext .+ Fc
        return u_lj + u_pme
    end
    function baoab_step!(qq, pp, bx, dt, gamma)
        update_tan_forces!(qq, bx)
        pp .+= (0.5 * dt) .* F_tan
        Con.project_velocities!(qq, pp, m, cs; tolerance = 1e-12)
        WB._drift!(qq, pp, m, mols, 0.5 * dt)
        for j in 1:N, d in 1:3
            ξ[d, j] = sqrt(m[j]) * randn(rng)
        end
        Con.project_velocities!(qq, ξ, m, cs; tolerance = 1e-12)
        c1 = exp(-gamma * dt)
        c2 = sqrt((1.0 - c1 * c1) / beta)
        pp .= c1 .* pp .+ c2 .* ξ
        WB._drift!(qq, pp, m, mols, 0.5 * dt)
        U = update_tan_forces!(qq, bx)
        pp .+= (0.5 * dt) .* F_tan
        Con.project_velocities!(qq, pp, m, cs; tolerance = 1e-12)
        return U
    end

    # Thermalize / relax lattice
    for _ in 1:1000
        baoab_step!(q, p, box, 0.002, 10.0)
    end

    Vs = Float64[]; rhos = Float64[]; Us = Float64[]; Hs = Float64[]
    Ts = Float64[]; Ps = Float64[]
    accs = Bool[]
    max_con_res = 0.0
    dof = 6 * length(mols)
    mass_total_g = sum(m) / WB.AVOGADRO
    for sweep in 1:1200
        for _ in 1:4
            baoab_step!(q, p, box, 0.002, 2.0)
        end
        wrap_molecules!(q, box, mols, m)
        res = npt_step!(barostat, q, p, box, rng)
        box = res.box
        push!(accs, res.accepted)
        max_con_res = max(max_con_res, maximum(abs, Con.constraint_residuals(cs, q)))
        if sweep > 300
            V = T.box_volume(box)
            U_cur = res.accepted ? res.new_energy : res.old_energy
            rho = mass_total_g / (V * 1e-21)
            K = 0.5 * sum(p[d, j]^2 / m[j] for j in 1:N for d in 1:3)
            T_inst = 2.0 * K / (dof * WB.KB)
            push!(Vs, V); push!(rhos, rho); push!(Us, U_cur); push!(Hs, U_cur + P_int * V)
            push!(Ts, T_inst)
            push!(Ps, bar_from_internal_pressure(res.pressure_new, WB.TIP3P_UNITS))
        end
        length(plan_cache) > 4 && empty!(plan_cache)
    end

    sv = SA.summarize_observable(Vs; n_blocks = 20)
    sr = SA.summarize_observable(rhos; n_blocks = 20)
    su = SA.summarize_observable(Us; n_blocks = 20)
    sh = SA.summarize_observable(Hs; n_blocks = 20)
    st = SA.summarize_observable(Ts; n_blocks = 20)
    sp = SA.summarize_observable(Ps; n_blocks = 20)

    @printf("  TIP3P NPT (64 mols, acc=%.3f, max_con=%.2e):\n", mean(accs), max_con_res)
    @printf("    V   = %.4f ± %.4f nm^3   (std=%.4f, tau=%.1f, ess=%.1f)\n",
            sv.mean, sv.se, sv.std, sv.tau, sv.ess)
    @printf("    rho = %.4f ± %.4f g/cm^3 (std=%.4f, tau=%.1f, ess=%.1f)\n",
            sr.mean, sr.se, sr.std, sr.tau, sr.ess)
    @printf("    U   = %.2f ± %.2f kJ/mol   (std=%.2f, tau=%.1f, ess=%.1f)\n",
            su.mean, su.se, su.std, su.tau, su.ess)
    @printf("    H   = %.2f ± %.2f kJ/mol   (std=%.2f, tau=%.1f, ess=%.1f)\n",
            sh.mean, sh.se, sh.std, sh.tau, sh.ess)
    @printf("    T   = %.2f ± %.2f K        (target 300.0, tau=%.1f, ess=%.1f)\n",
            st.mean, st.se, st.tau, st.ess)
    @printf("    P   = %.1f ± %.1f bar      (target 1.0, tau=%.1f, ess=%.1f)\n",
            sp.mean, sp.se, sp.tau, sp.ess)

    @test max_con_res < 1e-10
    @test 0.15 < mean(accs) < 0.85
    @test abs(st.mean - T_target) <= 4 * st.se
    @test abs(sp.mean - P_bar) <= 4 * sp.se
    @test 0.75 < sr.mean < 1.15
    @test su.mean < -30.0 * length(mols)
    @test sv.ess > 5.0 && sp.ess > 50.0
    return true
end

# ---------------------------------------------------------------------------
# Phase 4B: infer_molecules, semi-isotropic, anisotropic, PBC unwrapping
# ---------------------------------------------------------------------------

function test_infer_molecules()
    # 1. Water + ion + diatomic
    bonds = [(1, 2), (1, 3)]
    constraints = [(5, 6, 1.2)]
    mols = infer_molecules(6; bonds = bonds, constraints = constraints)
    @test mols == [[1, 2, 3], [4], [5, 6]]

    # 2. CoupledConstraintSet
    cs = Con.CoupledConstraintSet(6, [(1, 2, 0.9572), (1, 3, 0.9572), (5, 6, 1.1)])
    mols_cs = infer_molecules(6; constraints = cs)
    @test mols_cs == [[1, 2, 3], [4], [5, 6]]

    # 3. Only singletons
    @test infer_molecules(4) == [[1], [2], [3], [4]]

    # 4. Ring / cycle topology
    @test infer_molecules(3; bonds = [(1, 2), (2, 3), (3, 1)]) == [[1, 2, 3]]
    return true
end

function test_semi_isotropic_and_anisotropic_moves()
    rng = MersenneTwister(42)
    box0 = T.OrthorhombicBox((4.0, 4.0, 5.0))
    gamma_st = 0.05
    P = 0.02
    beta = 1.5
    q0 = rand(rng, 3, 6)

    # 1. Semi-isotropic XY with surface tension
    b_semi = VolumeMoveBarostat(beta, P, 0.08;
                                coupling = :semi_isotropic_xy,
                                surface_tension = gamma_st)
    @test b_semi.coupling === :semi_isotropic_xy
    @test b_semi.surface_tension == gamma_st

    U(q, b) = 0.08 * sum(abs2, q) + 0.02 * T.box_volume(b)
    for trial in 1:20
        q = copy(q0)
        r = step!(b_semi, q, box0, U, rng)
        if r.accepted
            # Check Lx and Ly scale by exact same factor (semi-isotropic xy)
            sx = r.box.L[1] / box0.L[1]
            sy = r.box.L[2] / box0.L[2]
            @test isapprox(sx, sy; atol = 1e-13)
            @test isapprox((r.box.L[1] * r.box.L[2]) - (box0.L[1] * box0.L[2]), r.delta_area; atol = 1e-12)
        end

        # Check exact log w identity:
        # log w = -β [ ΔU + P ΔV - γ ΔA_xy ] + N_dof * log(V'/V)
        dU = r.new_energy - r.old_energy
        dV = r.new_volume - r.old_volume
        dA = r.delta_area
        n_dof = r.n_translational_units + 1  # :log_volume proposal
        expected_log_w = -beta * (dU + P * dV - gamma_st * dA) + n_dof * log(r.new_volume / r.old_volume)
        @test isapprox(r.log_acceptance, expected_log_w; atol = 1e-12)
        @test isapprox(r.surface_tension_work, beta * gamma_st * dA; atol = 1e-12)
    end

    # 2. Anisotropic
    b_aniso = VolumeMoveBarostat(beta, P, 0.08; coupling = :anisotropic)
    @test b_aniso.coupling === :anisotropic
    for trial in 1:20
        q = copy(q0)
        r = step!(b_aniso, q, box0, U, rng)
        dU = r.new_energy - r.old_energy
        dV = r.new_volume - r.old_volume
        n_dof = r.n_translational_units + 1
        expected_log_w = -beta * (dU + P * dV) + n_dof * log(r.new_volume / r.old_volume)
        @test isapprox(r.log_acceptance, expected_log_w; atol = 1e-12)
    end
    return true
end

function test_pbc_straddling_rigid_molecules()
    # 1. OrthorhombicBox with dimer straddling periodic boundary
    box = T.OrthorhombicBox((3.0, 3.0, 3.0))
    q = [0.05 2.95; 1.5 1.5; 1.5 1.5]
    d_bond = 0.10
    constraints = [(1, 2, d_bond)]
    cs = Con.CoupledConstraintSet(2, constraints)
    mols = [[1, 2]]
    masses = [1.0, 1.0]

    b_ortho = VolumeMoveBarostat(1.0, 0.05, 0.05;
                                 molecules = mols,
                                 constraints = constraints,
                                 masses = masses)
    U_zero = (qq, bb) -> 0.0

    # Test scaling directly with scale_anisotropic_coordinates!
    q_out = similar(q)
    scale_anisotropic_coordinates!(q_out, q, box, 1.08, 1.08, 1.08;
                                   molecules = mols, masses = masses)
    res_scaled = Con.constraint_residuals(cs, q_out, box)
    @test maximum(abs, res_scaled) < 1e-12

    # Test step! (trial move with PBC unwrapping)
    for seed in 1:10
        q_test = copy(q)
        r = step!(b_ortho, q_test, box, U_zero, MersenneTwister(seed))
        cur_box = r.accepted ? r.box : box
        res = Con.constraint_residuals(cs, q_test, cur_box)
        @test maximum(abs, res) < 1e-12
    end

    # 2. TriclinicBox with allow_triclinic = true
    H = [3.0 0.4 0.0; 0.0 3.0 0.0; 0.0 0.0 3.0]
    box_tri = T.TriclinicBox(H)
    q_tri = Matrix{Float64}(undef, 3, 2)
    q_tri[:, 1] = [0.05, 0.05, 0.05]
    q_tri[:, 2] = [0.05, 0.05, 0.05] .+ H[:, 1] .- [0.10, 0.0, 0.0]
    cs_tri = Con.CoupledConstraintSet(2, [(1, 2, 0.10)])
    @test maximum(abs, Con.constraint_residuals(cs_tri, q_tri, box_tri)) < 1e-12

    b_tri = VolumeMoveBarostat(1.0, 0.05, 0.05;
                               molecules = mols,
                               constraints = [(1, 2, 0.10)],
                               masses = masses,
                               allow_triclinic = true)
    for seed in 1:10
        q_test = copy(q_tri)
        r = step!(b_tri, q_test, box_tri, U_zero, MersenneTwister(seed))
        cur_box = r.accepted ? r.box : box_tri
        res = Con.constraint_residuals(cs_tri, q_test, cur_box)
        @test maximum(abs, res) < 1e-12
    end
    return true
end

# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

for (name, test) in [
    ("log_volume_detailed_balance", test_log_volume_detailed_balance),
    ("scaling_and_reversal", test_scaling_and_reversal),
    ("seeded_replay", test_seeded_replay),
    ("ideal_gas_analytic", test_ideal_gas_analytic),
    ("jacobian_control", test_jacobian_control),
    ("harmonic_crystal_detailed_balance", test_harmonic_crystal_detailed_balance),
    ("pressure_decomposition_and_virial", test_pressure_decomposition_and_virial),
    ("constraint_and_box_refusals", test_refusals),
    ("hybrid_pressure_mean", test_hybrid_pressure_mean),
    ("interacting_rigid_molecular_lj_npt", test_interacting_rigid_molecular_lj_npt),
    ("interacting_tip3p_water_npt", test_interacting_tip3p_water_npt),
    ("infer_molecules", test_infer_molecules),
    ("semi_isotropic_and_anisotropic_moves", test_semi_isotropic_and_anisotropic_moves),
    ("pbc_straddling_rigid_molecules", test_pbc_straddling_rigid_molecules),
]
    println("TEST ", name)
    test()
end
println("ALL BAROSTAT TESTS PASSED")
