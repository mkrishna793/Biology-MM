# Production End-to-End GPU & CPU Simulation Suite (7 Complete Scientific Workflows)
#
# Covers:
#   1. GPU-resident batched NVE simulation on official biomolecules (water_ions 2,164 & solvated_protein 21,536)
#   2. 2-level symplectic r-RESPA integration & O(dt^2) energy band scaling
#   3. Extended bit-exact reversible SymplecticTernary MD + O(1)-memory adjoint replay (dna 628 & water_ions 2,164)
#   4. Zero-FFT TernaryMSM 2.0 (O(M_L^4) Hartley coarsest solver + :sos_c4 + CW(13,9)) 100-step NVE
#   5. Triclinic skew-cell polarizable PCG (:qc_hybrid) + analytic forces + CW orthogonal noise + 9-entry virial
#   6. Semi-isotropic lipid membrane NPT (membrane 28,160) & soft-core alchemical MBAR (protein_ligand 2,615)
#   7. Adaptive multi-backend routing (:auto, :cpu, :metal, Vulkan/SYCL probes) & 108k-atom scale parity

using FoundationCore
using JSON
using LinearAlgebra
using Printf
using Random
using Test

const CH   = FoundationCore.Chemistry
const CF   = FoundationCore.ChemistryFactors
const T    = FoundationCore.Types
const F    = FoundationCore.Factors
const D    = FoundationCore.Dynamics
const RP   = FoundationCore.Respa
const NB   = FoundationCore.Neighbors
const SLJ  = FoundationCore.StreamedLJ
const SC   = FoundationCore.SystemContract
const MCB  = FoundationCore.MonteCarloBarostat
const ALC  = FoundationCore.Alchemical
const TXBT = FoundationCore.TriumXBT
const TGPU = FoundationCore.TriumGPU
const ENG  = FoundationCore.Engine
const TMSM = FoundationCore.TernaryMSM
const SYMT = FoundationCore.SymplecticTernary
const PO   = FoundationCore.PolarizationOperator

const FOUNDATION = normpath(joinpath(@__DIR__, "..", ".."))
const REPO       = FOUNDATION
const CACHE      = get(ENV, "CHEM_FIXTURE_CACHE",
                       joinpath(homedir(), ".cache", "biology-mm", "chemistry-fixtures"))
const PYTHON     = let p = get(ENV, "CHEM_PYTHON", "")
    !isempty(p) ? p : something(Sys.which("python3"), Sys.which("python"), "python3")
end

function read_amber_coordinates(path::AbstractString)
    lines = readlines(path)
    head = split(lines[2])
    n = parse(Int, head[1])
    toks = split(join(lines[2:end]), r"\s+"; keepempty = false)
    offset = length(head) > 1 ? 3 : 2
    vals = parse.(Float64, toks[offset:end])
    length(vals) >= 3n || error("$path: expected $(3n) coordinates, got $(length(vals))")
    q = Matrix{Float64}(undef, 3, n)
    k = 1
    for i in 1:n, d in 1:3
        q[d, i] = vals[k] / 10.0
        k += 1
    end
    return q
end

function load_official_fixture(id::String)
    manifest = JSON.parsefile(joinpath(FOUNDATION, "chemistry", "fixtures", "manifest.json"))
    fixtures = Dict(String(fx["id"]) => fx for fx in manifest["fixtures"])
    fx = fixtures[id]
    if String(fx["status"]) == "generated"
        parm = joinpath(CACHE, "dppc.prmtop")
        crd  = joinpath(CACHE, "dppc.inpcrd")
    else
        parm = joinpath(REPO, String(fx["parameter_file"]))
        crd  = joinpath(REPO, String(fx["coordinate_file"]))
    end
    norm_path = joinpath(CACHE, "$id.json")
    if !isfile(norm_path)
        mkpath(CACHE)
        tool = joinpath(FOUNDATION, "chemistry", "tools", "extract_amber.py")
        run(pipeline(`$PYTHON $tool --parm $parm --crd $crd --out $norm_path --force-field $(fx["force_field"]) --repo-root $REPO`;
                     stdout = devnull, stderr = devnull))
    end
    ns = CH.read_normalized(norm_path)
    q0 = read_amber_coordinates(crd)
    return (ns = ns, q0 = q0)
end

function build_txbt_bonded_system(ns::CH.NormalizedSystem, q0::Matrix{Float64};
                                  rcut::Float64, skin::Float64 = 0.15, alpha::Float64 = 2.2)
    n = CH.n_atoms(ns)
    box = T.OrthorhombicBox(ns.box.lengths_nm)
    masses = Float64[a.mass for a in ns.atoms]
    inv_mass = 1.0 ./ masses
    sigma = Float64[a.sigma_nm for a in ns.atoms]
    epsilon = Float64[a.epsilon_kj for a in ns.atoms]
    charges = Float64[a.charge for a in ns.atoms]
    topo = SLJ.PairTopology(
        excluded = [(e[1], e[2]) for e in ns.exclusions],
        scaled = Tuple{Int, Int, Float64}[(p.i, p.j, p.lj_scale) for p in ns.pair14]
    )
    tc = NB.TilePairCache(q0, box, rcut, skin)
    nb_fac = TXBT.TriumTiledNonbondedFactor(
        sigma, epsilon, charges;
        rcut = rcut, alpha = alpha, coulomb_prefactor = CH.CHEM_UNITS.coulomb_prefactor,
        lj_mode = :lfs, topology = topo, cache = tc,
        use_native_bridge = true, use_fast_math = false, use_carry_free = false
    )
    bonds = F.HarmonicBond[F.HarmonicBond(b.i, b.j, b.k, b.r0_nm) for b in ns.bonds]
    angles = F.HarmonicAngle[F.HarmonicAngle(a.i, a.j, a.k, a.k_theta, a.theta0_rad) for a in ns.angles]
    all_facs = F.AbstractFactor[bonds..., angles..., nb_fac]
    sys = T.System(T.AtomSet(copy(q0), zeros(3, n), masses, charges, fill(:X, n)),
                   box; units = CH.CHEM_UNITS, temperature = 300.0)
    return (sys = sys, box = box, masses = masses, inv_mass = inv_mass,
            nb_fac = nb_fac, bonds = bonds, angles = angles, factors = all_facs)
end

@testset "Production End-to-End GPU & CPU Simulation Suite (7 Workflows)" begin
    TGPU.clear_gpu_plan_cache!()

    # ==========================================================================
    # E2E 1: GPU-Resident Batched NVE on water_ions (2,164) & solvated_protein (21,536)
    # ==========================================================================
    @testset "E2E 1: GPU-Resident Batched NVE on Official Biomolecules" begin
        dt = 0.0001  # 0.1 fs for unconstrained O-H stability on raw coordinates
        n_steps = 200
        has_batch = TGPU.has_gpu_md_batch(:auto)

        for (sys_id, rcut, expected_n) in [("water_ions", 0.8, 2164), ("solvated_protein", 0.9, 21536)]
            fx = load_official_fixture(sys_id)
            n = CH.n_atoms(fx.ns)
            @test n == expected_n
            built = build_txbt_bonded_system(fx.ns, fx.q0; rcut = rcut)

            # Direct batched GPU call vs CPU Dynamics.velocity_verlet_step!
            q_gpu = copy(fx.q0)
            p_gpu = zeros(Float64, 3, n)
            F_gpu = zeros(Float64, 3, n)
            res0 = TGPU.run_batched_gpu_steps!(
                copy(fx.q0), zeros(Float64, 3, n), built.inv_mass, built.box,
                built.nb_fac, built.bonds, built.angles, dt, 0; backend = :auto
            )
            res_gpu = TGPU.run_batched_gpu_steps!(
                q_gpu, p_gpu, built.inv_mass, built.box,
                built.nb_fac, built.bonds, built.angles, dt, n_steps;
                F_out = F_gpu, backend = :auto
            )

            @test res_gpu.used_gpu == has_batch
            @test isfinite(res_gpu.total_energy)
            @test all(isfinite, q_gpu) && all(isfinite, p_gpu) && all(isfinite, F_gpu)
            @test !any(isnan, q_gpu) && !any(isnan, F_gpu)
            rel_drift_gpu = abs(res_gpu.total_energy - res0.total_energy) / max(1.0, abs(res0.total_energy))
            @test rel_drift_gpu < 5e-4

            # CPU Dynamics reference parity (200 steps on water_ions, 5 steps on 21.5k solvated_protein)
            n_cpu_ref = sys_id == "water_ions" ? n_steps : 5
            hs_cpu = D.HarmonicSystem(built.factors, built.masses,
                                      F.Context(built.box, CH.CHEM_UNITS, n))
            q_cpu = copy(fx.q0)
            p_cpu = zeros(Float64, 3, n)
            F_cpu = zeros(Float64, 3, n)
            D.forces!(hs_cpu, q_cpu, F_cpu)
            pe_cpu = 0.0
            for _ in 1:n_cpu_ref
                pe_cpu = D.velocity_verlet_step!(q_cpu, p_cpu, F_cpu, hs_cpu, dt)
            end
            E_cpu = pe_cpu + D.kinetic_energy(hs_cpu, p_cpu)
            if n_cpu_ref == n_steps
                @test abs(res_gpu.total_energy - E_cpu) / max(1.0, abs(E_cpu)) < 1e-4
                @test maximum(abs, q_gpu .- q_cpu) < 1e-3
                @test maximum(abs, F_gpu .- F_cpu) / max(1.0, maximum(abs, F_cpu)) < 1e-3
            else
                q_gpu_cmp = copy(fx.q0); p_gpu_cmp = zeros(Float64, 3, n); F_gpu_cmp = zeros(Float64, 3, n)
                res_cmp = TGPU.run_batched_gpu_steps!(
                    q_gpu_cmp, p_gpu_cmp, built.inv_mass, built.box,
                    built.nb_fac, built.bonds, built.angles, dt, n_cpu_ref;
                    F_out = F_gpu_cmp, backend = :auto
                )
                @test abs(res_cmp.total_energy - E_cpu) / max(1.0, abs(E_cpu)) < 1e-4
                @test maximum(abs, q_gpu_cmp .- q_cpu) < 1e-3
                @test maximum(abs, F_gpu_cmp .- F_cpu) / max(1.0, maximum(abs, F_cpu)) < 1e-3
            end

            # Engine.create_engine + Engine.verify_engine_conservation
            eng = ENG.create_engine(built.sys, built.factors;
                                    backend = :auto, integrator = :verlet, dt_ps = dt)
            st0 = D.State(copy(fx.q0), zeros(Float64, 3, n))
            tel = ENG.verify_engine_conservation(eng, st0; n_steps = n_steps)
            @test tel.n_steps == n_steps
            @test isfinite(tel.initial_energy) && isfinite(tel.final_energy)
            @test tel.rel_energy_drift < 5e-4
            @test tel.momentum_drift_norm < 1e-3
            @printf("  [E2E 1] %-17s N=%5d (%s): ms/step=%.3f (%.1f ns/d @ 2fs), |ΔE/E0|=%.2e\n",
                    sys_id, n, string(tel.backend), tel.ms_per_step, tel.ns_per_day_2fs, tel.rel_energy_drift)
        end
    end

    # ==========================================================================
    # E2E 2: 2-Level Symplectic r-RESPA & O(dt^2) Energy Band Scaling
    # ==========================================================================
    @testset "E2E 2: 2-Level Symplectic r-RESPA & O(dt^2) Scaling" begin
        rng = MersenneTwister(202602)
        n_mol = 8
        N = 3 * n_mol
        box = T.OrthorhombicBox((5.0, 5.0, 5.0))
        q0 = zeros(Float64, 3, N)
        bonds = F.HarmonicBond[]
        angles = F.HarmonicAngle[]
        excl = Tuple{Int, Int}[]
        mol = 0
        for ix in 0:1, iy in 0:1, iz in 0:1
            mol += 1
            a, b, c = 3*(mol-1)+1, 3*(mol-1)+2, 3*(mol-1)+3
            cx, cy, cz = 1.0 + 2.2*ix, 1.0 + 2.2*iy, 1.0 + 2.2*iz
            q0[:, b] = [cx, cy, cz]
            q0[:, a] = [cx + 0.31, cy, cz]
            q0[:, c] = [cx + 0.31*cos(1.85), cy + 0.31*sin(1.85), cz]
            push!(bonds, F.HarmonicBond(a, b, 1200.0, 0.30))
            push!(bonds, F.HarmonicBond(b, c, 1200.0, 0.30))
            push!(angles, F.HarmonicAngle(a, b, c, 80.0, 1.85))
            push!(excl, (a, b), (b, c), (a, c))
        end
        p0 = 0.04 .* randn(rng, 3, N)
        p0 .-= sum(p0, dims = 2) ./ N
        masses = fill(12.0, N)
        charges = [mod1(i, 3) == 2 ? -0.4 : 0.2 for i in 1:N]
        tc = NB.TilePairCache(q0, box, 1.8, 0.2)
        nb = TXBT.TriumTiledNonbondedFactor(
            fill(0.28, N), fill(0.35, N), charges;
            rcut = 1.8, alpha = 0.22, coulomb_prefactor = 138.935456,
            lj_mode = :lfs, topology = SLJ.PairTopology(excluded = excl),
            cache = tc, include_self = true
        )
        facs = F.AbstractFactor[bonds..., angles..., nb]
        fast_idx = collect(1:(length(bonds) + length(angles)))
        hs = D.HarmonicSystem(facs, masses, F.Context(box, T.REDUCED, N))

        function run_respa_band(dt_outer::Float64, n_steps::Int, m_sub::Int)
            st = D.State(copy(q0), copy(p0))
            integ = RP.RespaIntegrator(hs, fast_idx)
            Es = Float64[D.total_energy(hs, st.q, st.p)]
            for _ in 1:n_steps
                RP.step!(st, integ, dt_outer; m = m_sub)
                push!(Es, D.total_energy(hs, st.q, st.p))
            end
            return (band = maximum(Es) - minimum(Es),
                    rel_drift = abs(Es[end] - Es[1]) / max(1.0, abs(Es[1])),
                    Es = Es)
        end

        r1 = run_respa_band(0.001, 100, 4)
        r2 = run_respa_band(0.0005, 200, 4)
        ratio = r1.band / r2.band
        @test all(isfinite, r1.Es) && all(isfinite, r2.Es)
        @test r1.rel_drift < 5e-4
        @test r2.rel_drift < 2e-4
        @test ratio > 3.0  # O(dt^2) -> ~4x contraction

        # Also verify Engine :respa matches
        sys_r = T.System(T.AtomSet(copy(q0), copy(p0), masses, charges, fill(:Ar, N)),
                         box; units = T.REDUCED, temperature = 300.0)
        eng_r = ENG.create_engine(sys_r, facs; backend = :auto, integrator = :respa,
                                  dt_ps = 0.0005, fast_indices = fast_idx, respa_m = 4)
        tel_r = ENG.verify_engine_conservation(eng_r, D.State(copy(q0), copy(p0)); n_steps = 40)
        @test tel_r.rel_energy_drift < 5e-4
        @printf("  [E2E 2] r-RESPA O(dt^2) band: dt=1fs -> %.4e, dt=0.5fs -> %.4e (%.2fx contraction)\n",
                r1.band, r2.band, ratio)
    end

    # ==========================================================================
    # E2E 3: Extended Bit-Exact Reversible SymplecticTernary + O(1) Replay
    # ==========================================================================
    @testset "E2E 3: Bit-Exact Reversible SymplecticTernary & O(1) Replay" begin
        dt = 2.5e-5
        for (sys_id, n_rev, n_rep) in [("dna", 100, 10), ("water_ions", 50, 5)]
            fx = load_official_fixture(sys_id)
            n = CH.n_atoms(fx.ns)
            is_per = fx.ns.box.kind !== :none
            prot = CH.default_protocol(fx.ns; nonbonded = is_per ? :pme : :direct,
                                      cutoff_nm = is_per ? 0.9 : 1.0,
                                      lj_mode = :lfs, water_model = :flexible)
            rs = CF.with_lfs_c64(CH.build_resolved(fx.ns; protocol = prot))

            p0 = zeros(Float64, 3, n)
            p0[1, 1] = 0.01
            sys = T.System(T.AtomSet(copy(fx.q0), copy(p0), rs.atoms.masses, zeros(n), fill(:X, n)),
                           rs.box; units = rs.units)
            st0 = SYMT.SymplecticTernaryState(sys)
            st = copy(st0)

            SYMT.simulate_symplectic_ternary!(st, rs.factors, dt, n_rev; direction = +1)
            @test st.step_index == n_rev
            @test st.Q != st0.Q && st.P != st0.P

            SYMT.simulate_symplectic_ternary!(st, rs.factors, dt, n_rev; direction = -1)
            @test st.step_index == 0
            @test st.Q == st0.Q && st.P == st0.P
            @test maximum(abs, st.Q .- st0.Q) == 0
            @test maximum(abs, st.P .- st0.P) == 0

            # O(1) replay observable match against explicitly stored forward frames (0.0 exact)
            st_fwd = copy(st0)
            stored_ke = Vector{Float64}(undef, n_rep)
            for k in 1:n_rep
                stored_ke[k] = SYMT.kinetic_energy(st_fwd)
                SYMT.symplectic_ternary_step!(st_fwd, rs.factors, dt; direction = +1)
            end
            obs_stored_rev = 0.0
            for k in n_rep:-1:1
                obs_stored_rev = k == n_rep ? stored_ke[k] : (obs_stored_rev + stored_ke[k])
            end
            obs_replay = SYMT.reversible_trajectory_replay_accum(
                st0, rs.factors, dt, n_rep, (_, s) -> SYMT.kinetic_energy(s)
            )
            @test abs(obs_replay - obs_stored_rev) == 0.0
            @printf("  [E2E 3] %-10s N=%4d (%d fwd + %d bwd): 0 trit error, O(1) replay Δ=%.1e\n",
                    sys_id, n, n_rev, n_rev, abs(obs_replay - obs_stored_rev))
        end
    end

    # ==========================================================================
    # E2E 4: Zero-FFT TernaryMSM 2.0 (O(M_L^4) Hartley + :sos_c4 + CW(13,9))
    # ==========================================================================
    @testset "E2E 4: Zero-FFT TernaryMSM 2.0 100-Step NVE & CW(13,9)" begin
        rng = MersenneTwister(202604)
        L = 15.0
        box = T.OrthorhombicBox((L, L, L))
        N = 24
        q0 = zeros(Float64, 3, N)
        bonds = F.HarmonicBond[]
        excl = Tuple{Int, Int}[]
        for m in 1:(N ÷ 2)
            i1, i2 = 2m - 1, 2m
            cx = 1.5 + 3.2 * ((m - 1) % 4)
            cy = 2.0 + 3.5 * (((m - 1) ÷ 4) % 3)
            cz = 3.0 + 4.0 * ((m - 1) ÷ 12)
            q0[:, i1] = [cx, cy, cz]
            q0[:, i2] = [cx + 0.31, cy + 0.02, cz]
            push!(bonds, F.HarmonicBond(i1, i2, 800.0, 0.30))
            push!(excl, (i1, i2))
        end
        chg = [isodd(i) ? 0.45 : -0.45 for i in 1:N]
        corr = TMSM.PairCorrections(excluded = excl)
        msm = TMSM.TernaryMSMCoulombFactor(
            q0, chg, box;
            rcut = 3.0, grid_n = 27, levels = 2,
            prefactor = CH.CHEM_UNITS.coulomb_prefactor,
            corrections = corr, softening = :sos_c4
        )
        lj = SLJ.StreamedLJFactor(
            N, fill(0.35, N), fill(0.28, N), box;
            rcut = 3.0, mode = :lfs, topology = SLJ.PairTopology(excluded = excl)
        )
        facs = F.AbstractFactor[bonds..., lj, msm]
        hs = D.HarmonicSystem(facs, fill(12.0, N), F.Context(box, CH.CHEM_UNITS, N))
        p0 = 0.03 .* randn(rng, 3, N)
        p0 .-= sum(p0, dims = 2) ./ N
        st = D.State(copy(q0), copy(p0))
        recs, ser = D.run_trajectory(st, hs, 0.0005, 100; record_every = 10)
        @test length(recs) == 10
        @test all(isfinite, ser.compensated)
        E0 = ser.compensated[1]
        rel_msm = maximum(abs, ser.compensated .- E0) / max(1.0, abs(E0))
        @test rel_msm < 1e-3

        # Verify coarsest O(M_L^4) Hartley matches O(M_L^6) direct reference
        pl = msm.plan
        Ml = pl.dims[end]
        @test Ml == 9
        Phil_h = zeros(Float64, Ml, Ml, Ml)
        Phil_d = zeros(Float64, Ml, Ml, Ml)
        TMSM._coarse_convolve_hartley!(Phil_h, pl.Q[end], pl.coarse_G_hat, pl.coarse_DG_hat,
                                       pl.coarse_H1d, pl.coarse_buf1, pl.coarse_buf2, pl.prefactor, false)
        TMSM._coarse_convolve_direct!(Phil_d, pl.Q[end], pl.coarse_G, pl.coarse_DG, Ml, pl.prefactor, false)
        @test maximum(abs, Phil_h .- Phil_d) < 1e-13

        # Verify cw_convolve_1d! exactness with CW(13, 9)
        cw13 = TMSM.cw_matrix(13, 9)
        kern = randn(rng, 13)
        x_in = randn(rng, 13)
        out = zeros(Float64, 13)
        TMSM.cw_convolve_1d!(out, kern, x_in, cw13)
        ref = [sum(kern[mod(m - j, 13) + 1] * x_in[j] for j in 1:13) for m in 1:13]
        @test maximum(abs, out .- ref) < 1e-13
        @printf("  [E2E 4] TernaryMSM 2.0 (:sos_c4, Ml=%d Hartley) 100-step NVE: |ΔE/E0|=%.2e\n",
                Ml, rel_msm)
    end

    # ==========================================================================
    # E2E 5: Triclinic Skew-Cell Polarizable + Virial + Orthogonal CW Noise
    # ==========================================================================
    @testset "E2E 5: Triclinic Skew-Cell Polarizable, Virial & CW Noise" begin
        rng = MersenneTwister(202605)
        H = [4.0 0.6 0.2; 0.0 4.2 0.5; 0.0 0.0 4.5]
        tbox = T.TriclinicBox(H)
        N = 12
        s_frac = [0.15 0.55 0.85 0.20 0.60 0.80 0.30 0.70 0.10 0.45 0.75 0.90;
                  0.20 0.25 0.30 0.55 0.60 0.65 0.85 0.80 0.40 0.15 0.45 0.75;
                  0.15 0.40 0.70 0.25 0.55 0.85 0.35 0.65 0.80 0.20 0.50 0.88]
        q_tri = H * s_frac
        chg_tri = [isodd(i) ? 0.35 : -0.35 for i in 1:N]
        alpha_tri = fill(0.00025, N)

        op_tri = PO.build_operator(q_tri, chg_tri, alpha_tri, tbox;
                                   coulomb_prefactor = 138.935456,
                                   thole = true, a_thole = 0.2, periodic = true)
        cert, lam_lt = PO.certify_dipole_spd_luttinger_tisza(op_tri)
        @test cert === :certified && lam_lt > 0.0

        res_std = PO.solve_induced_dipoles(op_tri; tol = 1e-12)
        res_qc = PO.solve_induced_dipoles_ternary_pcg(op_tri; tol = 1e-12, precond_mode = :qc_hybrid, block_sites = 3)
        @test res_std.status === :converged && res_qc.status === :converged
        @test norm(res_qc.mu .- res_std.mu) < 1e-10

        F_ana = PO.analytic_induced_forces(op_tri, res_qc)
        F_fd = PO.induced_forces(op_tri, res_qc; h = 1e-6, analytic = false)
        @test maximum(abs, F_ana .- F_fd) < 1e-8

        # CW orthogonal noise & block filter
        Z = PO.cw_orthogonal_noise(rng, 26; n_block = 13)
        @test size(Z) == (13, 26)
        Z_blk1 = Z[:, 1:13]
        @test maximum(abs, Z_blk1 * Z_blk1' .- 13.0 * Matrix{Float64}(I, 13, 13)) < 1e-12
        x_flt = randn(rng, 26)
        y_flt = similar(x_flt)
        PO.apply_cw_block_filter!(y_flt, x_flt; n_block = 13, mode = :roundtrip)
        @test maximum(abs, y_flt .- x_flt) < 1e-13

        # 9-entry triclinic virial / box_derivative vs central FD on TriumTiledNonbondedFactor
        nb_tri = TXBT.TriumTiledNonbondedFactor(
            fill(0.32, N), fill(0.45, N), chg_tri;
            rcut = 1.6, alpha = 0.25, coulomb_prefactor = 138.935456,
            lj_mode = :lfs, use_native_bridge = true
        )
        F_tri = zeros(Float64, 3, N)
        G_tri = zeros(Float64, 3, 3)
        TXBT.eval_lj_dsf_pairs!(F_tri, G_tri, q_tri, tbox, nb_tri)
        h_fd = 1e-6
        G_fd = zeros(Float64, 3, 3)
        for b in 1:3, a in 1:3
            Hp = copy(H); Hp[a, b] += h_fd
            Hm = copy(H); Hm[a, b] -= h_fd
            up = TXBT.eval_lj_dsf_pairs!(zeros(3, N), nothing, Hp * s_frac, T.TriclinicBox(Hp), nb_tri)
            um = TXBT.eval_lj_dsf_pairs!(zeros(3, N), nothing, Hm * s_frac, T.TriclinicBox(Hm), nb_tri)
            G_fd[a, b] = (up - um) / (2 * h_fd)
        end
        max_g_err = maximum(abs, G_tri .- G_fd)
        @test max_g_err < 1e-8
        @printf("  [E2E 5] Triclinic polarizable LT=%s (λ=%.1f), F_FD=%.2e, 9-entry G_FD=%.2e\n",
                string(cert), lam_lt, maximum(abs, F_ana .- F_fd), max_g_err)
    end

    # ==========================================================================
    # E2E 6: Membrane Semi-Isotropic NPT (28,160) & Alchemical MBAR (2,615)
    # ==========================================================================
    @testset "E2E 6: Membrane Semi-Isotropic NPT & Alchemical MBAR" begin
        # 6A. Membrane (28,160 atoms) semi-isotropic XY barostat (when external DPPC fixture exists)
        d_before = 0.0
        d_after = 0.0
        has_membrane = isfile(joinpath(CACHE, "dppc.prmtop")) && isfile(joinpath(CACHE, "dppc.inpcrd"))
        if has_membrane
            fx_mem = load_official_fixture("membrane")
            n_mem = CH.n_atoms(fx_mem.ns)
            @test n_mem == 28160
            box_mem = T.OrthorhombicBox(fx_mem.ns.box.lengths_nm)
            masses_mem = Float64[a.mass for a in fx_mem.ns.atoms]
            bonds_mem = [(b.i, b.j) for b in fx_mem.ns.bonds]
            mols_mem = MCB.infer_molecules(n_mem; bonds = bonds_mem)
            @test length(mols_mem) > 100

            baro = MCB.VolumeMoveBarostat(1.0, 0.0, 0.005;
                                          molecules = mols_mem, masses = masses_mem,
                                          box = box_mem, coupling = :semi_isotropic_xy,
                                          surface_tension = 0.0)
            q_prop = copy(fx_mem.q0)
            prop = MCB.propose_volume_move(baro, q_prop, box_mem,
                                           (_, b_new) -> (b_new == box_mem ? 0.0 : -1000.0),
                                           MersenneTwister(601))
            @test prop.accepted
            @test isapprox(prop.box.L[1] / box_mem.L[1], prop.box.L[2] / box_mem.L[2]; atol = 1e-14)
            @test !isapprox(prop.box.L[1] / box_mem.L[1], prop.box.L[3] / box_mem.L[3]; atol = 1e-6)
            b1 = bonds_mem[1]
            d_before = sqrt(T.norm2(T.unwrap_displacement(box_mem, fx_mem.q0, b1[1], b1[2])))
            d_after  = sqrt(T.norm2(T.unwrap_displacement(prop.box, q_prop, b1[1], b1[2])))
            @test abs(d_before - d_after) < 1e-10
        end

        # 6B. Protein-ligand (2,615 atoms) 3-window soft-core alchemical TI + MBAR
        fx_pl = load_official_fixture("protein_ligand")
        n_pl = CH.n_atoms(fx_pl.ns)
        @test n_pl == 2615
        bonds_pl = [(b.i, b.j) for b in fx_pl.ns.bonds]
        mols_pl = MCB.infer_molecules(n_pl; bonds = bonds_pl)
        lig_idx = findfirst(m -> 8 <= length(m) <= 30, mols_pl)
        @test lig_idx !== nothing
        lig_atoms = mols_pl[lig_idx]

        excl_pl = Set{Tuple{Int, Int}}((min(e[1], e[2]), max(e[1], e[2])) for e in fx_pl.ns.exclusions)
        alch = ALC.AlchemicalNonbondedFactor(
            n_pl, lig_atoms,
            Float64[a.sigma_nm for a in fx_pl.ns.atoms],
            Float64[a.epsilon_kj for a in fx_pl.ns.atoms],
            Float64[a.charge for a in fx_pl.ns.atoms];
            excluded_pairs = excl_pl, lambda_lj = 0.6, lambda_elec = 0.4,
            alpha_lj = 0.5, alpha_elec = 0.5,
            coulomb_prefactor = CH.CHEM_UNITS.coulomb_prefactor,
            mode = :lfs, rcut = 1.0
        )
        sched = ALC.AlchemicalSchedule(:staged_decouple; lambda_split = 0.5)
        lambdas = [0.0, 0.5, 1.0]
        K = length(lambdas)
        n_samp = 8
        rng_pl = MersenneTwister(602)
        configs = [fx_pl.q0 .+ 0.002 .* randn(rng_pl, size(fx_pl.q0)) for _ in 1:(K * n_samp)]
        dU_samp = [Float64[] for _ in 1:K]
        u_kn = zeros(Float64, K, K * n_samp)
        beta = 1.0 / 2.479
        for (k, lam) in enumerate(lambdas)
            st_l = ALC.schedule_state(sched, lam)
            fk = ALC.with_lambdas(alch, st_l.lambda_lj, st_l.lambda_elec)
            for idx in ((k - 1) * n_samp + 1):(k * n_samp)
                push!(dU_samp[k], ALC.schedule_derivative(fk, configs[idx], sched, lam; box = T.NoBox()).dU_dlambda)
            end
            for n_idx in 1:(K * n_samp)
                u_kn[k, n_idx] = beta * ALC.alchemical_derivatives(fk, configs[n_idx], T.NoBox()).energy
            end
        end
        ti_res = ALC.estimate_ti(lambdas, dU_samp)
        mbar_res = ALC.estimate_mbar(u_kn, fill(n_samp, K))
        @test isfinite(ti_res.delta_f) && isfinite(ti_res.se)
        @test mbar_res.converged && isfinite(mbar_res.delta_f) && isfinite(mbar_res.se_delta_f)

        # 6C. Engine validation & telemetry + refusal gate on :trunc
        fx_eng = has_membrane ? fx_mem : load_official_fixture("water_ions")
        built_eng = build_txbt_bonded_system(fx_eng.ns, fx_eng.q0; rcut = 0.9)
        eng_mem = ENG.create_engine(built_eng.sys, built_eng.factors; backend = :auto)
        info_mem = ENG.engine_info(eng_mem)
        @test info_mem.n_atoms == CH.n_atoms(fx_eng.ns)
        @test info_mem.resolved_backend in (:metal, :vulkan, :sycl, :cpu)
        bad_nb = TXBT.TriumTiledNonbondedFactor(
            fill(0.3, 4), fill(0.1, 4), zeros(4); rcut = 1.0, lj_mode = :trunc
        )
        bad_sys = T.System(T.AtomSet(zeros(3, 4), zeros(3, 4), ones(4), zeros(4), fill(:Ar, 4)),
                           T.OrthorhombicBox((4.0, 4.0, 4.0)))
        @test_throws ErrorException ENG.create_engine(bad_sys, F.AbstractFactor[bad_nb])
        @printf("  [E2E 6] membrane NPT bond Δ=%.2e, protein_ligand MBAR ΔF=%.2f±%.2f kJ/mol\n",
                abs(d_before - d_after), mbar_res.delta_f, mbar_res.se_delta_f)
    end

    # ==========================================================================
    # E2E 7: Multi-Backend Routing (:auto/:cpu/:metal/Vulkan/SYCL) & 108k Parity
    # ==========================================================================
    @testset "E2E 7: Adaptive Multi-Backend Routing & 108k-Atom Scale Parity" begin
        prof = TGPU.probe_gpu_device()
        avail = TGPU.available_gpu_backends()
        @test prof isa TGPU.AdaptiveGPUProfile
        @test :cpu in avail
        @test TGPU.has_gpu_bridge(:vulkan) isa Bool
        @test TGPU.has_gpu_bridge(:sycl) isa Bool

        # Small system (< crossover) across :auto, :cpu, :metal, :vulkan, :sycl
        rng = MersenneTwister(202607)
        N_sm = 16
        box_sm = T.OrthorhombicBox((4.5, 4.5, 4.5))
        q_sm = Matrix{Float64}(undef, 3, N_sm)
        k = 1
        for ix in 0:3, iy in 0:1, iz in 0:1
            q_sm[:, k] = [0.6 + ix * 0.95, 0.8 + iy * 1.6, 0.8 + iz * 1.6]
            k += 1
        end
        chg_sm = [isodd(i) ? 0.25 : -0.25 for i in 1:N_sm]
        nb_sm = TXBT.TriumTiledNonbondedFactor(
            fill(0.30, N_sm), fill(0.45, N_sm), chg_sm;
            rcut = 1.8, alpha = 0.22, coulomb_prefactor = 138.935456,
            lj_mode = :lfs, cache = NB.TilePairCache(q_sm, box_sm, 1.8, 0.2),
            include_self = true
        )
        sys_sm = T.System(T.AtomSet(q_sm, zeros(3, N_sm), fill(12.0, N_sm), chg_sm, fill(:Ar, N_sm)),
                          box_sm; units = T.REDUCED)
        F_ref_sm = zeros(Float64, 3, N_sm)
        eng_cpu_sm = ENG.create_engine(sys_sm, F.AbstractFactor[nb_sm]; backend = :cpu)
        u_ref_sm = ENG.evaluate_step!(eng_cpu_sm, F_ref_sm, q_sm, box_sm)
        for req in (:auto, :cpu, :metal, :vulkan, :sycl)
            eng_b = ENG.create_engine(sys_sm, F.AbstractFactor[nb_sm]; backend = req)
            @test eng_b.active_backend in (:metal, :vulkan, :sycl, :cpu)
            if !TGPU.has_gpu_bridge(req) && req in (:vulkan, :sycl)
                @test eng_b.active_backend === :cpu
            end
            F_b = zeros(Float64, 3, N_sm)
            u_b = ENG.evaluate_step!(eng_b, F_b, q_sm, box_sm)
            @test abs(u_b - u_ref_sm) / max(1.0, abs(u_ref_sm)) < 1e-4
            @test maximum(abs, F_b .- F_ref_sm) / max(1.0, maximum(abs, F_ref_sm)) < 1e-4
        end

        # 108,000-atom FCC Lattice (:auto vs :cpu vs :metal)
        nx = ny = nz = 30
        a = 1.55
        N_108k = 4 * nx * ny * nz
        @test N_108k == 108_000
        box_108k = T.OrthorhombicBox((nx * a, ny * a, nz * a))
        q_108k = Matrix{Float64}(undef, 3, N_108k)
        basis = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5))
        idx = 0
        for iz in 0:(nz - 1), iy in 0:(ny - 1), ix in 0:(nx - 1), (bx, by, bz) in basis
            idx += 1
            q_108k[1, idx] = (ix + bx) * a
            q_108k[2, idx] = (iy + by) * a
            q_108k[3, idx] = (iz + bz) * a
        end
        chg_108k = [isodd(i) ? 0.5 : -0.5 for i in 1:N_108k]
        tc_108k = NB.TilePairCache(q_108k, box_108k, 1.4, 0.25)
        nb_108k = TXBT.TriumTiledNonbondedFactor(
            fill(0.95, N_108k), fill(0.05, N_108k), chg_108k;
            rcut = 1.4, alpha = 0.2, coulomb_prefactor = 138.93545764,
            lj_mode = :lfs, cache = tc_108k
        )
        sys_108k = T.System(T.AtomSet(q_108k, zeros(3, N_108k), fill(39.948, N_108k), chg_108k, fill(:Ar, N_108k)),
                            box_108k; units = T.REDUCED)
        state_mb = (sizeof(q_108k) + sizeof(sys_108k.atoms.p) + sizeof(chg_108k)) / (1024^2)
        @test state_mb < 6.5  # ~6.0 MB state RAM

        F_cpu_108k = zeros(Float64, 3, N_108k)
        eng_cpu_108k = ENG.create_engine(sys_108k, F.AbstractFactor[nb_108k]; backend = :cpu)
        u_cpu_108k = ENG.evaluate_step!(eng_cpu_108k, F_cpu_108k, q_108k, box_108k)

        for req in (:auto, :metal)
            eng_g = ENG.create_engine(sys_108k, F.AbstractFactor[nb_108k]; backend = req)
            F_g = zeros(Float64, 3, N_108k)
            u_g = ENG.evaluate_step!(eng_g, F_g, q_108k, box_108k)
            rel_f = norm(F_g .- F_cpu_108k) / max(sqrt(Float64(length(F_cpu_108k))), norm(F_cpu_108k))
            rel_u = abs(u_g - u_cpu_108k) / max(1.0, abs(u_cpu_108k))
            @test rel_f < 2e-4
            @test rel_u < 1e-4
            if req === :auto
                @printf("  [E2E 7] 108k FCC (:auto->:%s): rel|dF|=%.2e, rel|dU|=%.2e, state=%.2f MB\n",
                        string(eng_g.active_backend), rel_f, rel_u, state_mb)
            end
        end
    end

    TGPU.clear_gpu_plan_cache!()
end

println("test_production_e2e_gpu_cpu PASS")
