using Test
using LinearAlgebra
using Random
using Statistics
using FoundationCore

const T  = FoundationCore.Types
const F  = FoundationCore.Factors
const D  = FoundationCore.Dynamics
const RS = FoundationCore.ResolvedSystem
const SC = FoundationCore.SystemContract
const CP = FoundationCore.CustomPhysics
const RN = FoundationCore.ReactiveNano
const GC = FoundationCore.GrandCanonical
const NM = FoundationCore.NEMD
const ES = FoundationCore.EnhancedSampling
const EN = FoundationCore.Engine

function _fd_box_deriv_engine(eng::EN.SimulationEngine, q::Matrix{Float64},
                              box::T.AbstractBox; h::Float64 = 1e-6)
    H0 = box isa T.OrthorhombicBox ? Matrix(Diagonal([box.L[1], box.L[2], box.L[3]])) :
         Matrix{Float64}((box::T.TriclinicBox).H)
    s = H0 \ q
    N = size(q, 2)
    F_tmp = zeros(Float64, 3, N)
    G_fd = zeros(3, 3)
    for b in 1:3, a in 1:3
        Hp = copy(H0); Hp[a, b] += h
        Hm = copy(H0); Hm[a, b] -= h
        up = EN.evaluate_step!(eng, F_tmp, Hp * s, T.TriclinicBox(Hp))
        um = EN.evaluate_step!(eng, F_tmp, Hm * s, T.TriclinicBox(Hm))
        G_fd[a, b] = (up - um) / (2.0 * h)
    end
    return G_fd
end

@testset "Phase 9 Nanotech & Custom Physics E2E via SimulationEngine" begin

    # =========================================================================
    # 1. (5,5) Armchair CNT + TersoffFactor through SimulationEngine
    # =========================================================================
    @testset "1. (5,5) CNT + TersoffFactor (Verlet, BAOAB, SymplecticTernary, Stress, Strain)" begin
        cnt = RN.build_cnt(5, 5, 2)
        n = size(cnt.q, 2)
        @test n == 40

        rng = MersenneTwister(2026)
        q0 = cnt.q .+ 0.0015 .* randn(rng, 3, n)
        p0 = 0.08 .* randn(rng, 3, n)
        p0 .-= sum(p0, dims = 2) ./ n

        atoms = T.AtomSet(q0, p0, cnt.masses, zeros(n), cnt.names)
        sys = T.System(atoms, cnt.box, T.ChemicalTopology(), CP.MD_UNITS, 300.0)
        fac = RN.TersoffFactor(n, RN.tersoff_carbon(; units = :md))

        eng_v = EN.create_engine(sys, [fac]; backend = :cpu, integrator = :verlet, dt_ps = 0.0002)
        info = EN.engine_info(eng_v)
        @test info.n_atoms == 40
        @test info.active_factors == ["TersoffFactor"]

        # Exact 9-entry matrix & 3-entry vector box_deriv parity vs central FD
        F_buf = zeros(3, n)
        G_mat = zeros(3, 3)
        G_vec = zeros(3)
        U_mat = EN.evaluate_step!(eng_v, F_buf, q0, cnt.box; box_deriv = G_mat)
        U_vec = EN.evaluate_step!(eng_v, F_buf, q0, cnt.box; box_deriv = G_vec)
        @test isfinite(U_mat) && U_mat ≈ U_vec
        @test isapprox(G_vec, [G_mat[1, 1], G_mat[2, 2], G_mat[3, 3]]; atol = 1e-12)

        G_fd = _fd_box_deriv_engine(eng_v, q0, cnt.box; h = 1e-6)
        @test maximum(abs, G_mat .- G_fd) / max(1.0, maximum(abs, G_fd)) < 1e-7

        # 200-step NVE energy conservation via SimulationEngine
        st0 = D.State(copy(q0), copy(p0))
        cons = EN.verify_engine_conservation(eng_v, st0; n_steps = 200)
        @test cons.rel_energy_drift < 1e-5
        @test cons.momentum_drift_norm < 1e-9

        # Axial tensile stretch beyond Tersoff equilibrium (~0.145 nm): sigma_zz > 0 under elongation
        eps_z = 0.04
        L0 = cnt.box.L
        box_stretch = T.OrthorhombicBox((L0[1], L0[2], L0[3] * (1.0 + eps_z)))
        q_stretch = copy(cnt.q)
        q_stretch[3, :] .*= (1.0 + eps_z)
        G_stretch = zeros(3, 3)
        EN.evaluate_step!(eng_v, F_buf, q_stretch, box_stretch; box_deriv = G_stretch)
        V_stretch = T.box_volume(box_stretch)
        # Cauchy stress sigma_zz = (1/V) * (G * h')[3,3] > 0 under tension
        sigma_zz = (G_stretch[3, 3] * box_stretch.L[3]) / V_stretch
        @test sigma_zz > 0.0

        # Also check BAOAB and SymplecticTernary via SimulationEngine on graphene
        gr = RN.build_graphene_sheet(2, 2)
        ng = size(gr.q, 2)
        qg = gr.q .+ 0.002 .* randn(rng, 3, ng)
        pg = 0.05 .* randn(rng, 3, ng)
        pg .-= sum(pg, dims = 2) ./ ng
        sys_gr = T.System(T.AtomSet(qg, pg, gr.masses, zeros(ng), gr.names),
                          gr.box, T.ChemicalTopology(), CP.MD_UNITS, 300.0)
        fac_gr = RN.TersoffFactor(ng, RN.tersoff_carbon(; units = :md))

        eng_b = EN.create_engine(sys_gr, [fac_gr]; backend = :cpu, integrator = :baoab,
                                 dt_ps = 0.0005, temperature = 300.0, gamma_ps_inv = 2.0)
        st_b = D.State(copy(qg), copy(pg))
        EN.run_steps!(eng_b, st_b, 50)
        @test all(isfinite, st_b.q) && all(isfinite, st_b.p) && st_b.force_valid

        eng_st = EN.create_engine(sys_gr, [fac_gr]; backend = :cpu,
                                  integrator = :symplectic_ternary, dt_ps = 0.0002)
        st_t = D.State(copy(qg), copy(pg))
        cons_t = EN.verify_engine_conservation(eng_st, st_t; n_steps = 100)
        @test cons_t.rel_energy_drift < 1e-4
    end

    # =========================================================================
    # 2. Silicon Diamond (StillingerWeber) + ParameterEncoder + RESPA Morse/CustomPair
    # =========================================================================
    @testset "2. StillingerWeber Si + ParameterEncoder + r-RESPA (Morse fast / CustomPair slow)" begin
        rng = MersenneTwister(303)
        dia = RN.build_diamond_lattice(2, 2, 2; a_nm = 0.5431, element = :Si)
        nd = size(dia.q, 2)
        qd = dia.q .+ 0.004 .* randn(rng, 3, nd)
        pd = 0.06 .* randn(rng, 3, nd)
        pd .-= sum(pd, dims = 2) ./ nd

        sys_si = T.System(T.AtomSet(qd, pd, dia.masses, zeros(nd), dia.names),
                          dia.box, T.ChemicalTopology(), CP.MD_UNITS, 300.0)
        fac_sw = RN.StillingerWeberFactor(nd, RN.stillinger_weber_silicon(; units = :md))
        eng_si = EN.create_engine(sys_si, [fac_sw]; backend = :cpu, integrator = :verlet, dt_ps = 0.0005)
        cons_si = EN.verify_engine_conservation(eng_si, D.State(copy(qd), copy(pd)); n_steps = 200)
        @test cons_si.rel_energy_drift < 1e-5

        # ParameterEncoder -> CustomPairFactor(:morse) + MorseBondFactor under :respa
        enc = CP.ParameterEncoder([
            CP.ParameterSpec(:De, 120.0; lower = 0.0, encoding = :log),
            CP.ParameterSpec(:a, 8.0; lower = 0.0, encoding = :log),
            CP.ParameterSpec(:r0, 0.25; lower = 0.05, upper = 1.0, encoding = :logit),
        ])
        z_vec = CP.encode_parameters(enc, Dict(:De => 120.0, :a => 8.0, :r0 => 0.25))
        p_dec = CP.decode_parameters(enc, z_vec)

        q4 = [0.30  0.55  0.80  1.05;
              0.40  0.42  0.75  0.77;
              0.50  0.50  0.50  0.50]
        p4 = zeros(3, 4)
        p4[1, 1] = 0.05; p4[1, 2] = -0.05
        box4 = T.OrthorhombicBox((2.5, 2.5, 2.5))
        m4 = fill(12.011, 4)
        sys4 = T.System(T.AtomSet(q4, p4, m4, zeros(4)), box4, T.ChemicalTopology(), CP.MD_UNITS, 300.0)

        f_morse_fast = RN.MorseBondFactor(1, 2, 250.0, 10.0, 0.25)
        f_cp_slow = CP.CustomPairFactor(:morse, p_dec, [(1, 3), (2, 4), (3, 4)];
                                        rc = 1.1, mode = :lfs, n_atoms = 4)
        eng_respa = EN.create_engine(sys4, [f_morse_fast, f_cp_slow];
                                     backend = :cpu, integrator = :respa,
                                     dt_ps = 0.0004, respa_m = 4)
        @test eng_respa.fast_indices == [1]  # MorseBondFactor auto-classified as fast

        cons_respa = EN.verify_engine_conservation(eng_respa, D.State(copy(q4), copy(p4)); n_steps = 200)
        @test cons_respa.rel_energy_drift < 1e-5

        # SystemContract.box_derivative and virial parity vs SimulationEngine.evaluate_step!
        rs4 = SC.resolved_from_system(sys4, F.AbstractFactor[f_morse_fast, f_cp_slow])
        G_sc = SC.box_derivative(rs4, q4)
        Xi_sc = SC.virial(rs4, q4)
        F_e = zeros(3, 4)
        G_e = zeros(3, 3)
        EN.evaluate_step!(eng_respa, F_e, q4, box4; box_deriv = G_e)
        @test isapprox(G_e, G_sc; atol = 1e-12)
        hmat = Diagonal(collect(box4.L))
        @test isapprox(-(hmat * transpose(G_e)), Xi_sc; atol = 1e-12)
    end

    # =========================================================================
    # 3. Nanofluidic CNT Pore + GCMC Loading + NEMD Electric-Field Drift
    # =========================================================================
    @testset "3. Nanofluidic Cylindrical Pore + GCMC + BAOAB E-Field Drift" begin
        rng = MersenneTwister(404)
        box_pore = T.OrthorhombicBox((4.0, 4.0, 6.0))
        cx, cy = 2.0, 2.0
        r_pore = 1.0

        # Host ring representing CNT wall atoms at r = r_pore + 0.35
        host_cols = Vector{Vector{Float64}}()
        for iz in 0.5:1.0:5.5
            for k in 0:7
                th = 2.0 * pi * k / 8.0
                push!(host_cols, [cx + 1.35 * cos(th), cy + 1.35 * sin(th), iz])
            end
        end
        q_host = hcat(host_cols...)
        n_host = size(q_host, 2)

        sp_ion = GC.GCMCSpecies(:Ion, 1.0, 0.5, 0.3, 0.4, 0.2, 1.0)
        samp = GC.GrandCanonicalSampler(sp_ion, box_pore, 1.0;
                                        units = T.REDUCED, rc = 1.5, mode = :lfs,
                                        hard_core_frac = 0.6,
                                        q_host = q_host,
                                        sigma_host = fill(0.4, n_host),
                                        epsilon_host = fill(0.2, n_host))

        # Place deterministic feasible guests inside the cylindrical pore using farkas_steric_screen
        q_guests = Matrix{Float64}(undef, 3, 0)
        for z_try in (1.0, 2.5, 4.0, 5.0)
            p_try = (cx + 0.15 * cos(z_try), cy + 0.15 * sin(z_try), z_try)
            cert_h = GC.farkas_steric_screen(p_try, q_host, box_pore, 0.5)
            cert_g = GC.farkas_steric_screen(p_try, q_guests, box_pore, 0.5)
            @test cert_h.feasible && cert_g.feasible
            q_guests = hcat(q_guests, collect(p_try))
        end

        # Also run GCMC sampling to confirm GCMC loading inside the pore environment
        gres = GC.run_gcmc!(samp, 1500; rng = rng, burnin = 300)
        @test isfinite(gres.mean_loading) && gres.mean_loading >= 0.0

        # Combine pore guests in SimulationEngine with cylindrical wall + E-field + body force
        n_ions = size(q_guests, 2)
        charges = [1.0, 1.0, 1.0, 1.0]
        masses = fill(1.0, n_ions)
        sys_nemd = T.System(T.AtomSet(copy(q_guests), zeros(3, n_ions), masses, charges),
                            box_pore, T.ChemicalTopology(), T.REDUCED, 1.0)

        f_wall = NM.WallRestraintFactor(:cylindrical_pore, collect(1:n_ions), n_ions;
                                        axis = 3, center = (cx, cy), r_max = 0.65, k = 15.0)
        f_efield = NM.ExternalElectricFieldFactor((0.0, 0.0, 1.2), charges, collect(1:n_ions), n_ions)
        f_body = NM.NanofluidicBodyForceFactor((0.0, 0.0, 0.3), collect(1:n_ions), n_ions)

        eng_nemd = EN.create_engine(sys_nemd, [f_wall, f_efield, f_body];
                                    backend = :cpu, integrator = :baoab,
                                    dt_ps = 0.005, temperature = 1.0, gamma_ps_inv = 1.5,
                                    rng = MersenneTwister(55))

        # Check exact force & 9-entry box derivative consistency across all NEMD factors
        F_n = zeros(3, n_ions)
        G_n = zeros(3, 3)
        EN.evaluate_step!(eng_nemd, F_n, q_guests, box_pore; box_deriv = G_n)
        G_fd_n = _fd_box_deriv_engine(eng_nemd, q_guests, box_pore; h = 1e-6)
        @test maximum(abs, G_n .- G_fd_n) < 1e-6

        st_n = D.State(copy(q_guests), zeros(3, n_ions))
        vel_hist = Matrix{Float64}[]
        for _ in 1:80
            EN.run_steps!(eng_nemd, st_n, 2)
            push!(vel_hist, st_n.p ./ reshape(masses, 1, n_ions))
        end
        # Positive axial drift & ionic conductivity along +z
        sigma_ion = NM.estimate_ionic_conductivity(vel_hist[end-20:end], charges,
                                                   (0.0, 0.0, 1.2), T.box_volume(box_pore))
        @test sigma_ion > 0.0
        j_mass = NM.estimate_mass_flux(q_guests, st_n.q, 160 * 0.005, pi * 0.65^2, masses, 3)
        @test j_mass > 0.0
    end

    # =========================================================================
    # 4. Enhanced Sampling (MetadynamicsBias & UmbrellaFactor) + All Phase 9 Factors Box Deriv
    # =========================================================================
    @testset "4. MetadynamicsBias & UmbrellaFactor + All Phase 9 Factors via SimulationEngine" begin
        q = [0.2  1.1  1.9  2.7;
             0.3  0.8  1.5  2.1;
             0.4  0.9  1.6  2.3]
        N = size(q, 2)
        box = T.OrthorhombicBox((4.0, 4.0, 4.0))
        tric = T.TriclinicBox([4.1 0.3 0.1; 0.0 4.0 0.2; 0.0 0.0 4.2])
        masses = fill(12.0, N)
        sys = T.System(T.AtomSet(copy(q), zeros(3, N), masses, zeros(N)),
                       box, T.ChemicalTopology(), T.REDUCED, 1.0)

        cv = ES.DistanceCV(1, 2)
        s0 = ES.cv_value_and_grad!(zeros(3, N), cv, q, box)
        meta = ES.MetadynamicsBias(cv, N; height = 0.5, sigma = 0.3, bias_factor = 5.0)
        ES.deposit_hill!(meta, s0 - 0.1)
        ES.deposit_hill!(meta, s0 + 0.15)
        umb = ES.UmbrellaFactor(ES.CoordinationCV([1, 2], [3, 4], 1.5), 10.0, 0.8, N)

        eng_es = EN.create_engine(sys, [meta, umb]; backend = :cpu, integrator = :verlet, dt_ps = 0.001)
        F_es = zeros(3, N)
        G_es = zeros(3, 3)
        EN.evaluate_step!(eng_es, F_es, q, box; box_deriv = G_es)
        G_es_fd = _fd_box_deriv_engine(eng_es, q, box; h = 1e-6)
        @test maximum(abs, G_es .- G_es_fd) < 1e-7

        p0 = [0.05 -0.05 0.02 -0.02;
              0.01  0.02 -0.02 -0.01;
             -0.01  0.01  0.02 -0.02]
        cons_es = EN.verify_engine_conservation(eng_es, D.State(copy(q), p0); n_steps = 200)
        @test cons_es.rel_energy_drift < 1e-5

        # Smoke-check evaluate_step! box_deriv (Matrix & Vector) for remaining Phase 9 factors
        f_ext = CP.CustomExternalFactor(
            (x, y, z, i, p) -> p[:k] * (x^2 + y^2 + z^2),
            (x, y, z, i, p) -> (2.0 * p[:k] * x, 2.0 * p[:k] * y, 2.0 * p[:k] * z),
            Dict(:k => 1.5), [1, 2],
        )
        f_mb = CP.CustomManyBodyFactor(
            (qm, bm, p) -> begin
                d = T.unwrap_displacement(bm, qm, 2, 1)
                Fm = zeros(3, size(qm, 2))
                Fm[1, 2] = -p[:k] * d.x; Fm[2, 2] = -p[:k] * d.y; Fm[3, 2] = -p[:k] * d.z
                Fm[1, 1] = p[:k] * d.x; Fm[2, 1] = p[:k] * d.y; Fm[3, 1] = p[:k] * d.z
                Gm = zeros(3, 3)
                w = CP._frac_disp(bm, d)
                CP._outer_add!(Gm, p[:k], d, w)
                return (0.5 * p[:k] * T.norm2(d), Fm, Gm)
            end,
            Dict(:k => 2.0), N,
        )
        mlip_model = RN.SymmetryFunctionMLIP(2.0; weights = [0.2, 0.1, 0.05, 0.05])
        f_mlip = RN.MLIPFactor(N, 2.0, mlip_model)

        for fac in (f_ext, f_mb, f_mlip)
            eng_f = EN.create_engine(sys, [fac]; backend = :cpu, integrator = :verlet, dt_ps = 0.001)
            Gm = zeros(3, 3)
            Gv = zeros(3)
            EN.evaluate_step!(eng_f, F_es, q, tric; box_deriv = Gm)
            EN.evaluate_step!(eng_f, F_es, q, tric; box_deriv = Gv)
            @test isapprox(Gv, [Gm[1, 1], Gm[2, 2], Gm[3, 3]]; atol = 1e-12)
            Gfd = _fd_box_deriv_engine(eng_f, q, tric; h = 1e-6)
            @test maximum(abs, Gm .- Gfd) < 1e-6
        end
    end

end
