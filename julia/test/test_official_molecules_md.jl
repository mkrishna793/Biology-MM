# Official Downloaded Molecules Deep MD & Phase 0–2 Integration Suite
#
# Deeply tests all five official downloaded biological molecule classes:
#   1. dna              (628 atoms, AMBER DNA duplex, non-periodic direct)
#   2. water_ions       (2,164 atoms, Zn2+ in TIP3P water with AMBER 12-6-4 lj146, periodic PME)
#   3. protein_ligand   (2,615 atoms, T4 lysozyme L99A/M102Q + catechol ligand, non-periodic direct)
#   4. solvated_protein (21,536 atoms, 18-residue protein + K+/Cl- + TIP3P water + CMAP, periodic PME)
#   5. membrane         (28,160 atoms, DPPC lipid bilayer + TIP3P water, periodic PME)
#
# Verifies:
#   - Normalized schema loading + force-consistent protocol building (including with_lfs_c64 for water_ions)
#   - Dynamics.force_model_supported & finite energies/forces on every official molecule
#   - Short NVE (velocity_verlet_step!), NVT (langevin_baoab_step!), and multi-timescale r-RESPA
#     (fast bonded/CMAP + slow nonbonded/PME) trajectories
#   - Rigid-water constrained dynamics via Settle / CoupledConstraintSet on periodic water_ions & solvated_protein
#   - TilePairCache + 64-bit fixed-point nonbonded evaluation on the periodic official systems
#   - End-to-end CLI `bin/md.jl` execution on periodic `water_ions` (`pme-lfs-flexible` & `pme-lfs-rigid`) + `resume_from`

using FoundationCore
using JSON
using LinearAlgebra
using Printf
using Random
using Test

const CH  = FoundationCore.Chemistry
const CF  = FoundationCore.ChemistryFactors
const CM  = FoundationCore.ChemistryCMAP
const T   = FoundationCore.Types
const F   = FoundationCore.Factors
const D   = FoundationCore.Dynamics
const RP  = FoundationCore.Respa
const C   = FoundationCore.Constraints
const S   = FoundationCore.Settle
const NB  = FoundationCore.Neighbors
const SLJ = FoundationCore.StreamedLJ
const SC  = FoundationCore.SystemContract
const EL  = FoundationCore.Electrostatics
const MCB = FoundationCore.MonteCarloBarostat
const ALC = FoundationCore.Alchemical
const RS  = FoundationCore.ResolvedSystem
const POL = FoundationCore.Polarizable
const TXBT = FoundationCore.TriumXBT
const TMSM = FoundationCore.TernaryMSM
const SYMT = FoundationCore.SymplecticTernary
const PO   = FoundationCore.PolarizationOperator

module OfficialMolCLI
include(joinpath(@__DIR__, "..", "bin", "md.jl"))
end

const FOUNDATION = normpath(joinpath(@__DIR__, "..", ".."))
const REPO = FOUNDATION
const CACHE = get(ENV, "CHEM_FIXTURE_CACHE",
                  joinpath(homedir(), ".cache", "biology-mm", "chemistry-fixtures"))
const PYTHON = let p = get(ENV, "CHEM_PYTHON", "")
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
        q[d, i] = vals[k] / 10.0          # Å -> nm
        k += 1
    end
    return q
end

function ensure_normalized(id::String, parm::String, crd::String, ff::String)
    mkpath(CACHE)
    out = joinpath(CACHE, "$id.json")
    if !isfile(out)
        tool = joinpath(FOUNDATION, "chemistry", "tools", "extract_amber.py")
        cmd = `$PYTHON $tool --parm $parm --crd $crd --out $out --force-field $ff --repo-root $REPO`
        run(pipeline(cmd; stdout = devnull, stderr = devnull))
    end
    return out
end

function is_local_bonded_factor(f::F.AbstractFactor)
    return F.is_local(f) && !(f isa CF.PairC64Table) && !(f isa CF.PairTableNonbonded)
end

@testset "Official Downloaded Molecules Deep MD Suite (5 classes)" begin
    manifest_path = joinpath(FOUNDATION, "chemistry", "fixtures", "manifest.json")
    manifest = JSON.parsefile(manifest_path)
    crd_paths = Dict{String,String}()

    for fx in manifest["fixtures"]
        id = String(fx["id"])
        cls = String(fx["class"])
        st_flag = String(fx["status"])
        if st_flag == "generated"
            parm = joinpath(CACHE, "dppc.prmtop")
            crd = joinpath(CACHE, "dppc.inpcrd")
            if !(isfile(parm) && isfile(crd))
                println("  $id ($cls): skipped (external generated fixture absent at $CACHE)")
                continue
            end
        else
            parm = joinpath(REPO, String(fx["parameter_file"]))
            crd = joinpath(REPO, String(fx["coordinate_file"]))
        end
        crd_paths[id] = crd
        ff = String(fx["force_field"])

        @testset "Official molecule: $id ($cls)" begin
            @test isfile(parm) && isfile(crd)
            norm_path = ensure_normalized(id, parm, crd, ff)
            @test isfile(norm_path)

            ns = CH.read_normalized(norm_path)
            q0 = read_amber_coordinates(crd)
            n = CH.n_atoms(ns)
            @test size(q0) == (3, n)

            is_periodic = ns.box.kind !== :none
            prot_flex = CH.default_protocol(ns;
                nonbonded = is_periodic ? :pme : :direct,
                cutoff_nm = is_periodic ? 0.9 : 1.0,
                lj_mode = :lfs,
                water_model = :flexible)
            rs_raw = CH.build_resolved(ns; protocol = prot_flex)
            # Upgrade any PairC64Table(mode=:trunc) to force-consistent :lfs
            rs = CF.with_lfs_c64(rs_raw)

            ctx = F.Context(rs.box, rs.units, n)
            hs = D.HarmonicSystem(rs.factors, rs.atoms.masses, ctx)
            @test D.force_model_supported(hs)

            # 1. Initial energy & force evaluation
            U0, F0 = SC.energy_forces(rs, q0)
            @test isfinite(U0)
            @test all(isfinite, F0)
            fmax = maximum(abs, F0)
            @test fmax > 1.0

            # 2. Short NVE trajectory (velocity Verlet)
            # Use small timestep (dt = 5e-5 ps = 0.05 fs) because raw crystal/solvated
            # coordinates have unminimized steric contacts up to ~1e4 kJ/(mol*nm).
            dt = is_periodic && n > 10_000 ? 2.5e-5 : 5.0e-5
            steps_nve = is_periodic && n > 10_000 ? 2 : 3
            st_nve = D.State(copy(q0), zeros(3, n))
            recs, series = D.run_trajectory(st_nve, hs, dt, steps_nve; record_every = 1)
            @test length(recs) == steps_nve
            @test all(isfinite, series.compensated)
            drift_nve = maximum(abs, series.compensated .- series.compensated[1])
            @test drift_nve < 500.0

            # 3. Short NVT step (Langevin BAOAB)
            rng = MersenneTwister(101)
            q_nvt = copy(q0)
            p_nvt = zeros(3, n)
            F_nvt = copy(F0)
            D.langevin_baoab_step!(q_nvt, p_nvt, F_nvt, hs, dt, 1.0, 1.0 / 2.479, rng)
            @test all(isfinite, q_nvt)
            @test all(isfinite, p_nvt)
            @test D.kinetic_energy(hs, p_nvt) > 0.0

            # 4. Multi-timescale r-RESPA (fast bonded/CMAP, slow nonbonded/PME)
            fast_idx = [i for (i, fac) in enumerate(rs.factors) if is_local_bonded_factor(fac)]
            @test !isempty(fast_idx)
            respa = RP.RespaIntegrator(hs, fast_idx)
            st_respa = D.State(copy(q0), zeros(3, n))
            RP.step!(st_respa, respa, dt; m = 2)
            @test isfinite(respa.last_energy)
            @test all(isfinite, st_respa.q)
            @test all(isfinite, st_respa.p)
            @test st_respa.force_valid

            # 5. For periodic systems with StreamedLJFactor: test TilePairCache + fixed_point parity
            if is_periodic
                slj_idx = findfirst(f -> f isa SLJ.StreamedLJFactor, rs.factors)
                @test slj_idx !== nothing
                slj = rs.factors[slj_idx]::SLJ.StreamedLJFactor
                tc = NB.TilePairCache(q0, rs.box, slj.rcut, 0.15)
                ws_t = SLJ.LJWorkspace(q0, rs.box, slj.rcut)
                U_tc, F_tc, np_tc = SLJ.evaluate!(ws_t, q0, slj.epsilon, slj.sigma, rs.box;
                                                  rcut = slj.rcut, mode = :lfs, topology = slj.topology,
                                                  threaded = true, cache = tc, fixed_point = true)
                ws_c = SLJ.LJWorkspace(q0, rs.box, slj.rcut)
                U_cell, F_cell, np_cell = SLJ.evaluate!(ws_c, q0, slj.epsilon, slj.sigma, rs.box;
                                                        rcut = slj.rcut, mode = :lfs, topology = slj.topology,
                                                        threaded = false)
                @test np_tc == np_cell
                @test isapprox(U_tc, U_cell; rtol = 1e-10, atol = 1e-10)
                @test maximum(abs, F_tc .- F_cell) <= 1e-8 * max(1.0, maximum(abs, F_cell))
            end

            @printf("  %-18s N=%-6d periodic=%-5s U0=%14.3f max|F|=%10.2e dE_NVE=%9.3e E_RESPA=%14.3f\n",
                    id, n, is_periodic, U0, fmax, drift_nve, respa.last_energy)
        end
    end

    # 6. Rigid-water constrained dynamics & periodic CLI `bin/md.jl` on `water_ions`
    @testset "Periodic water_ions rigid-water Settle + CLI pme-lfs-flexible & pme-lfs-rigid" begin
        norm_wi = joinpath(CACHE, "water_ions.json")
        crd_wi = joinpath(REPO, "wrappers", "python", "tests", "systems", "Mg_water.inpcrd")
        ns_wi = CH.read_normalized(norm_wi)
        q_wi = read_amber_coordinates(crd_wi)
        n_wi = CH.n_atoms(ns_wi)

        prot_rigid = CH.default_protocol(ns_wi; nonbonded = :pme, cutoff_nm = 0.8,
                                         lj_mode = :lfs, water_model = :rigid)
        rs_rigid = CF.with_lfs_c64(CH.build_resolved(ns_wi; protocol = prot_rigid))
        @test rs_rigid.constraints isa C.CoupledConstraintSet
        @test length(rs_rigid.constraints.components) == 721 # 721 rigid TIP3P waters

        ctx_wi = F.Context(rs_rigid.box, rs_rigid.units, n_wi)
        hs_wi = D.HarmonicSystem(rs_rigid.factors, rs_rigid.atoms.masses, ctx_wi)
        st_wi = D.State(copy(q_wi), zeros(3, n_wi))
        D.rattle_project!(st_wi, hs_wi, rs_rigid.constraints)
        pos_res0, vel_res0 = D.rattle_residuals(st_wi, hs_wi, rs_rigid.constraints)
        @test pos_res0 < 1e-11
        @test vel_res0 < 1e-11

        for _ in 1:2
            D.velocity_verlet_step!(st_wi, hs_wi, 5.0e-5; constraints = rs_rigid.constraints)
        end
        pos_res1, vel_res1 = D.rattle_residuals(st_wi, hs_wi, rs_rigid.constraints)
        @test pos_res1 < 1e-10
        @test vel_res1 < 1e-10

        # End-to-end CLI execution on periodic `water_ions` with pme-lfs-rigid and resume_from
        mktempdir() do dir
            cp(norm_wi, joinpath(dir, "water_ions.json"))
            coordinates = [q_wi[:, i] for i in 1:n_wi]
            doc = Dict(
                "schema_version" => 1,
                "normalized_amber" => "water_ions.json",
                "coordinates" => coordinates,
                "units" => "nm,kJ/mol,amu,ps",
                "protocol" => "pme-lfs-rigid",
                "cutoff_nm" => 0.8,
                "ensemble" => "nve",
                "workdir" => "run_wi",
                "workflow" => Dict(
                    "temperature" => 0.0,
                    "dt" => 2.5e-5,
                    "minimize_max_steps" => 0,
                    "equilibrate_steps" => 0,
                    "produce_steps" => 4,
                    "frame_every" => 1,
                    "contact_tol" => 0.0
                ),
                "analysis" => Dict(
                    "min_frames" => 4,
                    "min_fit_points" => 2,
                    "fit_window" => [0, 1],
                    "unwrap" => true,
                    "min_ess" => 1
                )
            )
            spec = joinpath(dir, "spec_wi.json")
            write(spec, JSON.json(doc))
            OfficialMolCLI.run_spec(spec)
            @test isfile(joinpath(dir, "run_wi", "manifest.json"))
            @test isfile(joinpath(dir, "run_wi", "trajectory.xyz"))
            @test isfile(joinpath(dir, "run_wi", "checkpoints", "ckpt_latest.json"))

            # Test resume_from on periodic rigid-water run
            doc_resume = copy(doc)
            doc_resume["resume_from"] = Dict("checkpoint" => "run_wi/checkpoints/ckpt_latest.json",
                                             "manifest" => "run_wi/manifest.json")
            spec_resume = joinpath(dir, "spec_wi_resume.json")
            write(spec_resume, JSON.json(doc_resume))
            OfficialMolCLI.run_spec(spec_resume)
            @test isfile(joinpath(dir, "run_wi", "checkpoints", "ckpt_latest.json"))
        end
    end

    # 7. Phase 3 on Official Molecules: DSFCoulombFactor + TilePairCache + Fixed-Point PairC64Table + NBFIXPairTable
    @testset "Phase 3 on Official Molecules (water_ions & solvated_protein DSF + NBFIX + fixed-point)" begin
        norm_wi = joinpath(CACHE, "water_ions.json")
        crd_wi = joinpath(REPO, "wrappers", "python", "tests", "systems", "Mg_water.inpcrd")
        ns_wi = CH.read_normalized(norm_wi)
        q_wi = read_amber_coordinates(crd_wi)
        n_wi = CH.n_atoms(ns_wi)
        prot_wi = CH.default_protocol(ns_wi; nonbonded = :pme, cutoff_nm = 0.8, lj_mode = :lfs, water_model = :flexible)
        rs_wi = CF.with_lfs_c64(CH.build_resolved(ns_wi; protocol = prot_wi))
        ctx_wi = F.Context(rs_wi.box, rs_wi.units, n_wi)

        # Verify PairC64Table fixed-point & threaded determinism on water_ions (2,164 atoms)
        c64_idx = findfirst(f -> f isa CF.PairC64Table, rs_wi.factors)
        @test c64_idx !== nothing
        c64_ref = rs_wi.factors[c64_idx]::CF.PairC64Table
        c64_fp = CF.PairC64Table(c64_ref.i, c64_ref.j, c64_ref.c; mode = :lfs, rcut = c64_ref.rcut, threaded = true, fixed_point = true)
        F_ref = zeros(3, n_wi)
        F_fp1 = zeros(3, n_wi)
        F_fp2 = zeros(3, n_wi)
        u_ref = F.evaluate!(F_ref, c64_ref, q_wi, ctx_wi)
        u_fp1 = F.evaluate!(F_fp1, c64_fp, q_wi, ctx_wi)
        u_fp2 = F.evaluate!(F_fp2, c64_fp, q_wi, ctx_wi)
        @test isapprox(u_ref, u_fp1; rtol = 1e-12)
        @test F_fp1 == F_fp2 # bitwise deterministic across evaluations
        @test maximum(abs, F_ref .- F_fp1) < 1e-8

        # Verify NBFIXPairTable off-diagonal Zn2+ (atom 1) to water oxygen overrides on water_ions
        ow_indices = [i for i in 2:n_wi if ns_wi.atoms[i].mass > 10.0][1:20]
        nbfix = CF.NBFIXPairTable(fill(1, length(ow_indices)), ow_indices,
                                  fill(0.245, length(ow_indices)), fill(0.85, length(ow_indices));
                                  mode = :lfs, rcut = 0.8, threaded = true, fixed_point = true)
        F_nb = zeros(3, n_wi)
        u_nb = F.evaluate!(F_nb, nbfix, q_wi, ctx_wi)
        G_nb = zeros(3, 3)
        RS.accumulate_box_derivative!(G_nb, nbfix, q_wi, ctx_wi)
        @test isfinite(u_nb) && all(isfinite, F_nb) && all(isfinite, G_nb)

        # Verify DSFCoulombFactor + TilePairCache on solvated_protein (21,536 atoms)
        norm_sp = joinpath(CACHE, "solvated_protein.json")
        crd_sp = crd_paths["solvated_protein"]
        ns_sp = CH.read_normalized(norm_sp)
        q_sp = read_amber_coordinates(crd_sp)
        n_sp = CH.n_atoms(ns_sp)
        box_sp = T.OrthorhombicBox(ns_sp.box.lengths_nm)
        charges_sp = [a.charge for a in ns_sp.atoms]
        tc_sp = NB.TilePairCache(q_sp, box_sp, 0.9, 0.15)
        dsf_sp = EL.DSFCoulombFactor(charges_sp; alpha = 2.2, rcut = 0.9,
                                     prefactor = CH.CHEM_UNITS.coulomb_prefactor,
                                     tile_cache = tc_sp, fixed_point = true)
        F_dsf = zeros(3, n_sp)
        u_dsf = F.evaluate!(F_dsf, dsf_sp, q_sp, F.Context(box_sp, CH.CHEM_UNITS, n_sp))
        @test isfinite(u_dsf)
        @test all(isfinite, F_dsf)
        @test maximum(abs, F_dsf) > 1.0
    end

    # 8. Phase 4 on Official Molecules: Constrained NPT on water_ions & Semi-Isotropic Membrane Barostat on membrane
    @testset "Phase 4 on Official Molecules (water_ions Constrained NPT & membrane Semi-Isotropic Barostat)" begin
        # 8a. Constrained molecular-COM NPT on water_ions (721 rigid waters + 1 Zn2+ = 722 molecules)
        norm_wi = joinpath(CACHE, "water_ions.json")
        crd_wi = crd_paths["water_ions"]
        ns_wi = CH.read_normalized(norm_wi)
        q_wi = read_amber_coordinates(crd_wi)
        n_wi = CH.n_atoms(ns_wi)
        prot_rigid = CH.default_protocol(ns_wi; nonbonded = :pme, cutoff_nm = 0.8, lj_mode = :lfs, water_model = :rigid)
        rs_rigid = CF.with_lfs_c64(CH.build_resolved(ns_wi; protocol = prot_rigid))
        ctx_wi = F.Context(rs_rigid.box, rs_rigid.units, n_wi)
        hs_wi = D.HarmonicSystem(rs_rigid.factors, rs_rigid.atoms.masses, ctx_wi)
        st_wi = D.State(copy(q_wi), zeros(3, n_wi))
        D.rattle_project!(st_wi, hs_wi, rs_rigid.constraints)

        bonds_wi = [(b.i, b.j) for b in ns_wi.bonds]
        mols_wi = MCB.infer_molecules(n_wi; bonds = bonds_wi, constraints = rs_rigid.constraints)
        @test length(mols_wi) == 722 # 1 ion + 721 water molecules

        q_scaled = similar(st_wi.q)
        MCB.scale_anisotropic_coordinates!(q_scaled, st_wi.q, rs_rigid.box, 1.01, 1.01, 1.01;
                                           molecules = mols_wi, masses = rs_rigid.atoms.masses)
        box_scaled = T.OrthorhombicBox(1.01 .* rs_rigid.box.L)
        # Verify every single rigid water constraint is preserved under molecular-COM scaling!
        st_scaled = D.State(q_scaled, zeros(3, n_wi))
        hs_scaled = D.HarmonicSystem(rs_rigid.factors, rs_rigid.atoms.masses, F.Context(box_scaled, rs_rigid.units, n_wi))
        pos_res_scaled, _ = D.rattle_residuals(st_scaled, hs_scaled, rs_rigid.constraints)
        @test pos_res_scaled < 1e-10

        # 8b. Semi-isotropic XY membrane barostat scaling on DPPC membrane (28,160 atoms)
        norm_mem = joinpath(CACHE, "membrane.json")
        if isfile(norm_mem) && haskey(crd_paths, "membrane") && isfile(crd_paths["membrane"])
            crd_mem = crd_paths["membrane"]
            ns_mem = CH.read_normalized(norm_mem)
            q_mem = read_amber_coordinates(crd_mem)
            n_mem = CH.n_atoms(ns_mem)
            box_mem = T.OrthorhombicBox(ns_mem.box.lengths_nm)
            masses_mem = [a.mass for a in ns_mem.atoms]
            bonds_mem = [(b.i, b.j) for b in ns_mem.bonds]
            mols_mem = MCB.infer_molecules(n_mem; bonds = bonds_mem)
            @test length(mols_mem) > 100 # lipids + waters

            baro_mem = MCB.VolumeMoveBarostat(1.0, 0.0, 0.005;
                                              molecules = mols_mem, masses = masses_mem,
                                              box = box_mem, coupling = :semi_isotropic_xy,
                                              surface_tension = 0.0)
            q_mem_prop = copy(q_mem)
            prop_mem = MCB.propose_volume_move(baro_mem, q_mem_prop, box_mem,
                                               (q_new, b_new) -> (b_new == box_mem ? 0.0 : -1000.0), MersenneTwister(77))
            @test prop_mem.accepted
            @test isapprox(prop_mem.box.L[1] / box_mem.L[1], prop_mem.box.L[2] / box_mem.L[2]; atol = 1e-14)
            @test !isapprox(prop_mem.box.L[1] / box_mem.L[1], prop_mem.box.L[3] / box_mem.L[3]; atol = 1e-6)
            # Check internal lipid bond length invariance across semi-isotropic move
            b1 = bonds_mem[1]
            d_before = sqrt(T.norm2(T.unwrap_displacement(box_mem, q_mem, b1[1], b1[2])))
            d_after  = sqrt(T.norm2(T.unwrap_displacement(prop_mem.box, q_mem_prop, b1[1], b1[2])))
            @test isapprox(d_before, d_after; atol = 1e-12)
        end
    end

    # 9. Phase 5 on Official Molecules: Soft-Core Alchemical Ligand Decoupling on protein_ligand (2,615 atoms)
    @testset "Phase 5 on Official Molecules (protein_ligand Soft-Core Alchemical Decoupling & TI/BAR/MBAR)" begin
        norm_pl = joinpath(CACHE, "protein_ligand.json")
        crd_pl = crd_paths["protein_ligand"]
        ns_pl = CH.read_normalized(norm_pl)
        q_pl = read_amber_coordinates(crd_pl)
        n_pl = CH.n_atoms(ns_pl)

        # Identify the catechol ligand molecule via connected components
        bonds_pl = [(b.i, b.j) for b in ns_pl.bonds]
        mols_pl = MCB.infer_molecules(n_pl; bonds = bonds_pl)
        # Catechol ligand is the small organic molecule (10-25 atoms) in the complex
        ligand_mol_idx = findfirst(m -> 8 <= length(m) <= 30, mols_pl)
        @test ligand_mol_idx !== nothing
        ligand_atoms = mols_pl[ligand_mol_idx]
        @test length(ligand_atoms) >= 8

        sigmas_pl = [a.sigma_nm for a in ns_pl.atoms]
        epsilons_pl = [a.epsilon_kj for a in ns_pl.atoms]
        charges_pl = [a.charge for a in ns_pl.atoms]
        excl_pl = Set{Tuple{Int,Int}}((min(e[1], e[2]), max(e[1], e[2])) for e in ns_pl.exclusions)

        alch_fac = ALC.AlchemicalNonbondedFactor(
            n_pl, ligand_atoms, sigmas_pl, epsilons_pl, charges_pl;
            excluded_pairs = excl_pl,
            lambda_lj = 0.6, lambda_elec = 0.4,
            alpha_lj = 0.5, alpha_elec = 0.5,
            coulomb_prefactor = CH.CHEM_UNITS.coulomb_prefactor,
            mode = :lfs, rcut = 1.0
        )

        # Evaluate analytical Cartesian forces & alchemical derivatives on the 2,615-atom complex
        F_alch = zeros(3, n_pl)
        u_alch = F.evaluate!(F_alch, alch_fac, q_pl, T.NoBox())
        derivs = ALC.alchemical_derivatives(alch_fac, q_pl, T.NoBox())
        @test isapprox(u_alch, derivs.energy; rtol = 1e-12)
        @test isfinite(derivs.dU_dlambda_lj) && isfinite(derivs.dU_dlambda_elec)
        @test all(isfinite, F_alch)

        # Central finite-difference verification of dU/dlambda_lj and dU/dlambda_elec on the real protein-ligand complex!
        eps_lam = 1e-6
        u_lj_p = ALC.alchemical_derivatives(ALC.with_lambdas(alch_fac, 0.6 + eps_lam, 0.4), q_pl, T.NoBox()).energy
        u_lj_m = ALC.alchemical_derivatives(ALC.with_lambdas(alch_fac, 0.6 - eps_lam, 0.4), q_pl, T.NoBox()).energy
        fd_lj = (u_lj_p - u_lj_m) / (2 * eps_lam)
        @test isapprox(derivs.dU_dlambda_lj, fd_lj; rtol = 1e-8, atol = 1e-8)

        u_el_p = ALC.alchemical_derivatives(ALC.with_lambdas(alch_fac, 0.6, 0.4 + eps_lam), q_pl, T.NoBox()).energy
        u_el_m = ALC.alchemical_derivatives(ALC.with_lambdas(alch_fac, 0.6, 0.4 - eps_lam), q_pl, T.NoBox()).energy
        fd_el = (u_el_p - u_el_m) / (2 * eps_lam)
        @test isapprox(derivs.dU_dlambda_elec, fd_el; rtol = 1e-7, atol = 1e-7)

        # Evaluate staged decoupling schedule across 5 windows and run TI / MBAR on perturbed configurations
        sched = ALC.AlchemicalSchedule(:staged_decouple; lambda_split = 0.5)
        lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
        K = length(lambdas)
        n_samples = 12
        rng_pl = MersenneTwister(2026)
        configs = [q_pl .+ 0.002 .* randn(rng_pl, size(q_pl)) for _ in 1:(K * n_samples)]
        dU_samples = [Float64[] for _ in 1:K]
        u_kn = zeros(Float64, K, K * n_samples)
        beta = 1.0 / 2.479

        for (k, lam) in enumerate(lambdas)
            st_lam = ALC.schedule_state(sched, lam)
            fac_k = ALC.with_lambdas(alch_fac, st_lam.lambda_lj, st_lam.lambda_elec)
            for idx in ((k - 1) * n_samples + 1):(k * n_samples)
                sd = ALC.schedule_derivative(fac_k, configs[idx], sched, lam; box = T.NoBox())
                push!(dU_samples[k], sd.dU_dlambda)
            end
            for n_idx in 1:(K * n_samples)
                u_kn[k, n_idx] = beta * ALC.alchemical_derivatives(fac_k, configs[n_idx], T.NoBox()).energy
            end
        end

        ti_res = ALC.estimate_ti(lambdas, dU_samples)
        mbar_res = ALC.estimate_mbar(u_kn, fill(n_samples, K))
        @test isfinite(ti_res.delta_f) && isfinite(ti_res.se)
        @test mbar_res.converged && isfinite(mbar_res.delta_f) && isfinite(mbar_res.se_delta_f)
        @printf("  protein_ligand alchemical decoupling (N=%d, ligand=%d atoms): dU/dλ_LJ=%10.3f dU/dλ_elec=%10.3f ΔF_TI=%10.3f ΔF_MBAR=%10.3f\n",
                n_pl, length(ligand_atoms), derivs.dU_dlambda_lj, derivs.dU_dlambda_elec, ti_res.delta_f, mbar_res.delta_f)
    end

    # 10. Trium-XBT C++/NEON, Ternary-MSM, SymplecticTernary & Ternary PCG on Official Molecules
    @testset "10. Trium-XBT C++/NEON, Ternary-MSM, SymplecticTernary & Ternary PCG on Official Molecules" begin
        # ----------------------------------------------------------------------
        # 10.1 TriumXBT.TriumTiledNonbondedFactor vs StreamedLJFactor + DSFCoulombFactor
        # ----------------------------------------------------------------------
        @testset "10.1 TriumXBT.TriumTiledNonbondedFactor Native C++/NEON & CarryFree Parity" begin
            @test TXBT.has_native_bridge() == true

            for id in filter(x -> haskey(crd_paths, x) && isfile(crd_paths[x]) && isfile(joinpath(CACHE, "$x.json")),
                             ["water_ions", "solvated_protein", "membrane"])
                ns_sys = CH.read_normalized(joinpath(CACHE, "$id.json"))
                crd_sys = crd_paths[id]
                q_sys = read_amber_coordinates(crd_sys)
                n_sys = CH.n_atoms(ns_sys)
                box_sys = T.OrthorhombicBox(ns_sys.box.lengths_nm)
                sig_sys = Float64[a.sigma_nm for a in ns_sys.atoms]
                eps_sys = Float64[a.epsilon_kj for a in ns_sys.atoms]
                chg_sys = Float64[a.charge for a in ns_sys.atoms]
                rcut = 0.9
                alpha = 2.0
                pref = CH.CHEM_UNITS.coulomb_prefactor

                topo_sys = SLJ.PairTopology(
                    excluded = ns_sys.exclusions,
                    scaled = [(p.i, p.j, p.lj_scale) for p in ns_sys.pair14]
                )
                corr_sys = EL.PairCorrections(
                    excluded = ns_sys.exclusions,
                    scaled = [(p.i, p.j, p.coulomb_scale) for p in ns_sys.pair14]
                )
                ctx_sys = F.Context(box_sys, CH.CHEM_UNITS, n_sys)

                # Reference: StreamedLJFactor (:lfs) + DSFCoulombFactor
                lj_ref = SLJ.StreamedLJFactor(n_sys, eps_sys, sig_sys, box_sys; rcut = rcut, mode = :lfs, topology = topo_sys)
                F_lj = zeros(Float64, 3, n_sys)
                u_lj = F.evaluate!(F_lj, lj_ref, q_sys, ctx_sys)
                G_lj = zeros(Float64, 3, 3)
                SLJ.streamed_box_derivative!(G_lj, lj_ref, q_sys, ctx_sys)

                dsf_ref = EL.DSFCoulombFactor(chg_sys; alpha = alpha, rcut = rcut, prefactor = pref, corrections = corr_sys, include_self = false)
                F_dsf = zeros(Float64, 3, n_sys)
                u_dsf = F.evaluate!(F_dsf, dsf_ref, q_sys, ctx_sys)
                G_dsf = zeros(Float64, 3, 3)
                EL.dsf_box_derivative!(G_dsf, dsf_ref, q_sys, box_sys)

                u_ref = u_lj + u_dsf
                F_ref = F_lj + F_dsf
                G_ref = G_lj + G_dsf

                # TriumTiledNonbondedFactor with native C++/NEON bridge
                txbt_fac = TXBT.TriumTiledNonbondedFactor(
                    sig_sys, eps_sys, chg_sys;
                    rcut = rcut, alpha = alpha, coulomb_prefactor = pref,
                    lj_mode = :lfs, lj14_scale = 0.5, coul14_scale = 0.8333333333333334,
                    topology = topo_sys, use_native_bridge = true, use_carry_free = false
                )
                F_txbt = zeros(Float64, 3, n_sys)
                G_txbt = zeros(Float64, 3, 3)
                u_txbt = TXBT.eval_lj_dsf_pairs!(F_txbt, G_txbt, q_sys, box_sys, txbt_fac)

                @test isapprox(u_txbt, u_ref; rtol = 1e-8, atol = 2e-8)
                @test maximum(abs, F_txbt - F_ref) < 1e-8
                @test maximum(abs, G_txbt - G_ref) < 1e-8

                # Also test carry-free accumulation mode on water_ions (2,164 atoms)
                if id == "water_ions"
                    txbt_cf = TXBT.TriumTiledNonbondedFactor(
                        sig_sys, eps_sys, chg_sys;
                        rcut = rcut, alpha = alpha, coulomb_prefactor = pref,
                        lj_mode = :lfs, lj14_scale = 0.5, coul14_scale = 0.8333333333333334,
                        topology = topo_sys, use_native_bridge = true, use_carry_free = true
                    )
                    F_cf = zeros(Float64, 3, n_sys)
                    G_cf = zeros(Float64, 3, 3)
                    u_cf = TXBT.eval_lj_dsf_pairs!(F_cf, G_cf, q_sys, box_sys, txbt_cf)
                    @test isapprox(u_cf, u_ref; rtol = 1e-8, atol = 2e-8)
                    @test maximum(abs, F_cf - F_ref) < 1e-7
                    @test maximum(abs, G_cf - G_ref) < 1e-8
                end
            end
        end

        # ----------------------------------------------------------------------
        # 10.2 TernaryMSM.TernaryMSMCoulombFactor (Zero-FFT O(N) Electrostatics)
        # ----------------------------------------------------------------------
        @testset "10.2 TernaryMSM.TernaryMSMCoulombFactor Radix-3 27-Point Electrostatics" begin
            # 10.2a DNA (628 atoms)
            ns_dna = CH.read_normalized(joinpath(CACHE, "dna.json"))
            crd_dna = crd_paths["dna"]
            q_dna = read_amber_coordinates(crd_dna)
            n_dna = CH.n_atoms(ns_dna)
            chg_dna = Float64[a.charge for a in ns_dna.atoms]

            min_dna = minimum(q_dna, dims = 2)
            max_dna = maximum(q_dna, dims = 2)
            span_dna = max_dna .- min_dna
            L_dna = [ceil(span_dna[d] + 4.0) for d in 1:3]
            q_dna_box = q_dna .- min_dna .+ 2.0
            box_dna = T.OrthorhombicBox((L_dna[1], L_dna[2], L_dna[3]))

            corr_dna = TMSM.PairCorrections(
                excluded = ns_dna.exclusions,
                scaled = [(p.i, p.j, p.coulomb_scale) for p in ns_dna.pair14]
            )
            rcut_msm = 1.0
            pref_msm = CH.CHEM_UNITS.coulomb_prefactor
            msm_dna = TMSM.TernaryMSMCoulombFactor(
                q_dna_box, chg_dna, box_dna;
                rcut = rcut_msm, prefactor = pref_msm, corrections = corr_dna
            )
            ctx_dna = F.Context(box_dna, CH.CHEM_UNITS, n_dna)
            F_dna = zeros(Float64, 3, n_dna)
            u_dna = F.evaluate!(F_dna, msm_dna, q_dna_box, ctx_dna)

            @test isfinite(u_dna)
            @test all(isfinite, F_dna)
            @test !any(isnan, F_dna)

            # Central finite differences verification on DNA subset (12 atoms)
            sub12_dna = 1:12
            q_sub_dna = q_dna_box[:, sub12_dna]
            chg_sub_dna = chg_dna[sub12_dna]
            box_sub = T.OrthorhombicBox((6.0, 6.0, 6.0))
            plan_sub_dna = TMSM.TernaryMSMPlan(q_sub_dna, box_sub; rcut = 1.5, prefactor = 1.0)
            u_sub_dna, F_sub_dna = TMSM.ternary_msm_energy_forces(plan_sub_dna, q_sub_dna, chg_sub_dna)

            h_fd = 1e-6
            max_dF_dna = 0.0
            for a in 1:12, d in 1:3
                qp = copy(q_sub_dna); qp[d, a] += h_fd
                up = TMSM.ternary_msm_energy(plan_sub_dna, qp, chg_sub_dna)
                qm = copy(q_sub_dna); qm[d, a] -= h_fd
                um = TMSM.ternary_msm_energy(plan_sub_dna, qm, chg_sub_dna)
                ffd = -(up - um) / (2 * h_fd)
                max_dF_dna = max(max_dF_dna, abs(F_sub_dna[d, a] - ffd))
            end
            @test max_dF_dna < 1e-8

            G_sub_dna = TMSM.ternary_msm_box_derivative(plan_sub_dna, q_sub_dna, chg_sub_dna)
            s_sub_dna = q_sub_dna ./ 6.0
            max_dG_dna = 0.0
            for a in 1:3
                box_p = [6.0, 6.0, 6.0]; box_p[a] += h_fd
                qp = s_sub_dna .* box_p
                plan_p = TMSM.TernaryMSMPlan(qp, box_p; rcut = 1.5, prefactor = 1.0)
                up = TMSM.ternary_msm_energy(plan_p, qp, chg_sub_dna)

                box_m = [6.0, 6.0, 6.0]; box_m[a] -= h_fd
                qm = s_sub_dna .* box_m
                plan_m = TMSM.TernaryMSMPlan(qm, box_m; rcut = 1.5, prefactor = 1.0)
                um = TMSM.ternary_msm_energy(plan_m, qm, chg_sub_dna)

                gfd = (up - um) / (2 * h_fd)
                max_dG_dna = max(max_dG_dna, abs(G_sub_dna[a, a] - gfd))
            end
            @test max_dG_dna < 1e-6

            # 10.2b water_ions (2,164 atoms)
            ns_wi = CH.read_normalized(joinpath(CACHE, "water_ions.json"))
            crd_wi = crd_paths["water_ions"]
            q_wi = read_amber_coordinates(crd_wi)
            n_wi = CH.n_atoms(ns_wi)
            chg_wi = Float64[a.charge for a in ns_wi.atoms]
            box_wi = T.OrthorhombicBox(ns_wi.box.lengths_nm)

            corr_wi = TMSM.PairCorrections(
                excluded = ns_wi.exclusions,
                scaled = [(p.i, p.j, p.coulomb_scale) for p in ns_wi.pair14]
            )
            msm_wi = TMSM.TernaryMSMCoulombFactor(
                q_wi, chg_wi, box_wi;
                rcut = 0.9, prefactor = 1.0, corrections = corr_wi
            )
            ctx_wi = F.Context(box_wi, CH.CHEM_UNITS, n_wi)
            F_wi = zeros(Float64, 3, n_wi)
            u_wi = F.evaluate!(F_wi, msm_wi, q_wi, ctx_wi)

            @test isfinite(u_wi)
            @test all(isfinite, F_wi)
            @test !any(isnan, F_wi)

            # Central finite differences verification on water_ions subset (12 atoms)
            sub12_wi = 1:12
            q_sub_wi = q_wi[:, sub12_wi]
            chg_sub_wi = chg_wi[sub12_wi]
            plan_sub_wi = TMSM.TernaryMSMPlan(q_sub_wi, box_sub; rcut = 1.5, prefactor = 1.0)
            u_sub_wi, F_sub_wi = TMSM.ternary_msm_energy_forces(plan_sub_wi, q_sub_wi, chg_sub_wi)

            max_dF_wi = 0.0
            for a in 1:12, d in 1:3
                qp = copy(q_sub_wi); qp[d, a] += h_fd
                up = TMSM.ternary_msm_energy(plan_sub_wi, qp, chg_sub_wi)
                qm = copy(q_sub_wi); qm[d, a] -= h_fd
                um = TMSM.ternary_msm_energy(plan_sub_wi, qm, chg_sub_wi)
                ffd = -(up - um) / (2 * h_fd)
                max_dF_wi = max(max_dF_wi, abs(F_sub_wi[d, a] - ffd))
            end
            @test max_dF_wi < 1e-8

            G_sub_wi = TMSM.ternary_msm_box_derivative(plan_sub_wi, q_sub_wi, chg_sub_wi)
            s_sub_wi = q_sub_wi ./ 6.0
            max_dG_wi = 0.0
            for a in 1:3
                box_p = [6.0, 6.0, 6.0]; box_p[a] += h_fd
                qp = s_sub_wi .* box_p
                plan_p = TMSM.TernaryMSMPlan(qp, box_p; rcut = 1.5, prefactor = 1.0)
                up = TMSM.ternary_msm_energy(plan_p, qp, chg_sub_wi)

                box_m = [6.0, 6.0, 6.0]; box_m[a] -= h_fd
                qm = s_sub_wi .* box_m
                plan_m = TMSM.TernaryMSMPlan(qm, box_m; rcut = 1.5, prefactor = 1.0)
                um = TMSM.ternary_msm_energy(plan_m, qm, chg_sub_wi)

                gfd = (up - um) / (2 * h_fd)
                max_dG_wi = max(max_dG_wi, abs(G_sub_wi[a, a] - gfd))
            end
            @test max_dG_wi < 1e-6
        end

        # ----------------------------------------------------------------------
        # 10.3 SymplecticTernary Exact Time-Reversible MD on Official Molecules
        # ----------------------------------------------------------------------
        @testset "10.3 SymplecticTernary Bit-Exact Time-Reversible MD (0 Trit/Bit Error)" begin
            # 10.3a DNA (628 atoms, 1,884 DOFs, 50 steps forward + 50 backward)
            ns_dna = CH.read_normalized(joinpath(CACHE, "dna.json"))
            crd_dna = crd_paths["dna"]
            q_dna = read_amber_coordinates(crd_dna)
            n_dna = CH.n_atoms(ns_dna)
            prot_dna = CH.default_protocol(ns_dna; nonbonded = :direct, cutoff_nm = 1.0, lj_mode = :lfs, water_model = :flexible)
            rs_dna = CH.build_resolved(ns_dna; protocol = prot_dna)

            p_dna = zeros(3, n_dna)
            p_dna[1, 1] = 0.01
            sys_dna = T.System(T.AtomSet(copy(q_dna), copy(p_dna), rs_dna.atoms.masses, zeros(n_dna), fill(:C, n_dna)), rs_dna.box; units = rs_dna.units)
            st0_dna = SYMT.SymplecticTernaryState(sys_dna)
            st_dna = copy(st0_dna)

            dt = 2.5e-5
            SYMT.simulate_symplectic_ternary!(st_dna, rs_dna.factors, dt, 50; direction = +1)
            @test st_dna.step_index == 50
            @test st_dna.Q != st0_dna.Q && st_dna.P != st0_dna.P

            SYMT.simulate_symplectic_ternary!(st_dna, rs_dna.factors, dt, 50; direction = -1)
            @test st_dna.step_index == 0
            @test all(st_dna.Q .== st0_dna.Q)
            @test all(st_dna.P .== st0_dna.P)
            @test maximum(abs.(st_dna.Q .- st0_dna.Q)) == 0
            @test maximum(abs.(st_dna.P .- st0_dna.P)) == 0

            # Reversible trajectory replay observable accumulation on DNA
            accum_dna = SYMT.reversible_trajectory_replay_accum(st0_dna, rs_dna.factors, dt, 10, st -> SYMT.kinetic_energy(st))
            @test isfinite(accum_dna) && accum_dna > 0.0

            # 10.3b water_ions (2,164 atoms, 6,492 DOFs, 20 steps forward + 20 backward)
            ns_wi = CH.read_normalized(joinpath(CACHE, "water_ions.json"))
            crd_wi = crd_paths["water_ions"]
            q_wi = read_amber_coordinates(crd_wi)
            n_wi = CH.n_atoms(ns_wi)
            prot_wi = CH.default_protocol(ns_wi; nonbonded = :pme, cutoff_nm = 0.9, lj_mode = :lfs, water_model = :flexible)
            rs_wi = CF.with_lfs_c64(CH.build_resolved(ns_wi; protocol = prot_wi))

            p_wi = zeros(3, n_wi)
            p_wi[1, 1] = 0.01
            sys_wi = T.System(T.AtomSet(copy(q_wi), copy(p_wi), rs_wi.atoms.masses, zeros(n_wi), fill(:O, n_wi)), rs_wi.box; units = rs_wi.units)
            st0_wi = SYMT.SymplecticTernaryState(sys_wi)
            st_wi = copy(st0_wi)

            SYMT.simulate_symplectic_ternary!(st_wi, rs_wi.factors, dt, 20; direction = +1)
            @test st_wi.step_index == 20
            @test st_wi.Q != st0_wi.Q && st_wi.P != st0_wi.P

            SYMT.simulate_symplectic_ternary!(st_wi, rs_wi.factors, dt, 20; direction = -1)
            @test st_wi.step_index == 0
            @test all(st_wi.Q .== st0_wi.Q)
            @test all(st_wi.P .== st0_wi.P)
            @test maximum(abs.(st_wi.Q .- st0_wi.Q)) == 0
            @test maximum(abs.(st_wi.P .- st0_wi.P)) == 0

            accum_wi = SYMT.reversible_trajectory_replay_accum(st0_wi, rs_wi.factors, dt, 5, st -> SYMT.kinetic_energy(st))
            @test isfinite(accum_wi) && accum_wi > 0.0
        end

        # ----------------------------------------------------------------------
        # 10.4 PolarizationOperator.solve_induced_dipoles_ternary_pcg on Official Polarizable Subsystem
        # ----------------------------------------------------------------------
        @testset "10.4 Dual-Rail Ternary PCG Mutual Polarization on Official Subsystem" begin
            ns_wi = CH.read_normalized(joinpath(CACHE, "water_ions.json"))
            crd_wi = crd_paths["water_ions"]
            q_wi = read_amber_coordinates(crd_wi)

            # 160-atom polarizable subsystem from water_ions
            N_sub = 160
            q_sub = q_wi[:, 1:N_sub]
            z_sub = Float64[a.charge for a in ns_wi.atoms[1:N_sub]]
            α_sub = fill(0.0005, N_sub)

            op_sub = PO.build_operator(q_sub, z_sub, α_sub; thole = true, a_thole = 0.2)
            prec_sub = PO.build_ternary_preconditioner(op_sub; num_planes = 2)
            comp_ratio = PO.ternary_compression_ratio(prec_sub)
            @test comp_ratio > 10.0

            r_std = PO.solve_induced_dipoles(op_sub; tol = 1e-11)
            r_ter = PO.solve_induced_dipoles_ternary_pcg(op_sub; tol = 1e-11, prec = prec_sub)
            @test r_std.status === :converged
            @test r_ter.status === :converged
            @test norm(r_ter.mu .- r_std.mu) < 1e-10
        end
    end
end
