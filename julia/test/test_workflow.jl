using FoundationCore
using Test
using LinearAlgebra
using Printf
using Random
using Statistics

const TY = FoundationCore.Types
const FA = FoundationCore.Factors
const DY = FoundationCore.Dynamics
const SA = FoundationCore.Sampling
const PR = FoundationCore.Prepare
const MB = FoundationCore.MonteCarloBarostat
const WF = FoundationCore.Workflow
const CS = FoundationCore.Constraints
const SLJ = FoundationCore.StreamedLJ

function _write_clean_pdb(path::AbstractString)
    open(path, "w") do io
        println(io, "CRYST1   12.000   12.000   12.000  90.00  90.00  90.00 P 1           1")
        coords = [
            (1, "C1", 2.0, 2.0, 2.0),
            (2, "C2", 3.5, 2.0, 2.0),
            (3, "C3", 5.0, 2.0, 2.0),
            (4, "C4", 2.0, 4.0, 2.0),
        ]
        for (idx, nm, x, y, z) in coords
            println(io, @sprintf("ATOM  %5d %-4s ALA A%4d    %8.3f%8.3f%8.3f  1.00  0.00           C  ",
                                 idx, nm, 1, x, y, z))
        end
        println(io, "END")
    end
    return path
end

@testset "SHA-256 NIST vectors" begin
    @test WF.sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    @test WF.sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    @test WF.sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
end

@testset "RunManifest immutability and tamper detection" begin
    mktempdir() do dir
        box = TY.OrthorhombicBox((10.0, 11.0, 12.0))
        params = Dict("lj_eps" => 1.0, "lj_sigma" => 1.0, "rcut" => 3.0)
        m = WF.build_manifest(
            run_id = "manifest-test",
            model_versions = Dict("lj" => "1.0.0"),
            resolved_parameters = params,
            units = TY.REDUCED,
            box = box,
            precision = "Float64",
            algorithm = :langevin_baoab,
            dt = 0.005,
            seed = 12345,
            n_atoms = 4,
            temperature = 1.2,
            gamma = 1.0,
        )
        @test WF.verify_manifest(m)
        # Mutating returned copies does not mutate the manifest
        p_copy = WF.manifest_resolved_parameters(m)
        p_copy["lj_eps"] = 99.0
        @test WF.manifest_resolved_parameters(m)["lj_eps"] == 1.0
        @test WF.verify_manifest(m)

        mpath = joinpath(dir, "manifest.json")
        WF.write_manifest_atomic(mpath, m)
        m2 = WF.read_manifest(mpath)
        @test m2.manifest_digest == m.manifest_digest
        @test m2.parameter_digest == m.parameter_digest
        @test m2.dt == m.dt
        @test m2.seed == m.seed
        @test m2.box_lengths == (10.0, 11.0, 12.0)

        # Tampering with JSON file fails verification on read
        tampered = replace(read(mpath, String), "\"dt\":0.005" => "\"dt\":0.01")
        write(mpath, tampered)
        @test_throws WF.ManifestError WF.read_manifest(mpath)

        # Invalid manifest fields are refused
        @test_throws WF.ManifestError WF.build_manifest(dt = 0.0, n_atoms = 4)
        @test_throws WF.ManifestError WF.build_manifest(dt = 0.01, n_atoms = 4, precision = "Float32")
        @test_throws WF.ManifestError WF.build_manifest(dt = 0.01, n_atoms = 4, algorithm = :unknown)
    end
end

@testset "Atomic versioned checkpoints and interruption safety" begin
    mktempdir() do dir
        box = TY.OrthorhombicBox((12.0, 12.0, 12.0))
        m = WF.build_manifest(dt = 0.005, seed = 42, n_atoms = 2,
                              box = box, algorithm = :langevin_baoab, temperature = 1.0, gamma = 1.0)
        q1 = [1.0 2.0; 0.0 0.5; -0.25 0.25]
        p1 = [0.1 -0.1; 0.2 -0.2; 0.0 0.0]
        F1 = [-0.5 0.5; 0.0 0.0; 0.1 -0.1]
        st1 = DY.State(q1, p1, F1, true)
        rng1 = Random.Xoshiro(42)
        rand(rng1, 7)

        ckpt1 = WF.make_checkpoint(
            stage = :produce, step = 10, time = 0.05, manifest = m, state = st1,
            box = box, rng = rng1,
            thermostat_state = WF.ThermostatState(kind = :langevin_baoab, temperature = 1.0, gamma = 1.0, beta = 1.0, steps = 10),
            barostat_state = WF.BarostatState(kind = :mc_volume, pressure = 0.1, beta = 1.0, max_delta_log_volume = 0.05, attempts = 2, accepts = 1),
            restart_guarantee = :bitwise,
        )
        cpath = joinpath(dir, "ckpt.json")
        WF.write_checkpoint_atomic(cpath, ckpt1)

        # Interruption safety: simulate crash before rename when overwriting with ckpt2
        st2 = DY.State(q1 .+ 1.0, p1 .* 2.0, F1, true)
        ckpt2 = WF.make_checkpoint(
            stage = :produce, step = 20, time = 0.10, manifest = m, state = st2,
            box = box, rng = rng1, restart_guarantee = :bitwise,
        )
        @test_throws ErrorException WF.write_checkpoint_atomic(
            cpath, ckpt2; before_rename_hook = _ -> error("simulated power loss"))

        # No leftover temp files; original ckpt1 remains intact
        @test readdir(dir) == ["ckpt.json"]
        loaded1 = WF.read_checkpoint(cpath; expected_manifest_digest = m.manifest_digest, require_bitwise = true)
        @test loaded1.step == 10
        @test reinterpret(UInt64, vec(loaded1.q)) == reinterpret(UInt64, vec(q1))
        @test reinterpret(UInt64, vec(loaded1.p)) == reinterpret(UInt64, vec(p1))
        @test loaded1.thermostat_state.steps == 10
        @test loaded1.barostat_state.attempts == 2
        @test loaded1.barostat_state.accepts == 1

        # Statistical-only guarantee refused when require_bitwise=true
        stat_ckpt = WF.make_checkpoint(
            stage = :produce, step = 10, time = 0.05, manifest = m, state = st1,
            box = box, rng = rng1, restart_guarantee = :statistical,
        )
        spath = joinpath(dir, "stat_ckpt.json")
        WF.write_checkpoint_atomic(spath, stat_ckpt)
        @test_throws WF.CheckpointMismatchError WF.read_checkpoint(spath; require_bitwise = true)
        stat_loaded = WF.read_checkpoint(spath; require_bitwise = false)
        @test stat_loaded.restart_guarantee === :statistical

        # Corruption, version mismatch, and manifest mismatch checks
        @test_throws WF.CheckpointMismatchError WF.read_checkpoint(cpath; expected_manifest_digest = "deadbeef")
        @test_throws WF.CheckpointVersionError WF.read_checkpoint(cpath; expected_version = 999)
        bad_json = replace(read(cpath, String), "\"step\":10" => "\"step\":11")
        write(cpath, bad_json)
        @test_throws WF.CheckpointCorruptionError WF.read_checkpoint(cpath)

        # MersenneTwister RNG exact state round-trip
        mt = Random.MersenneTwister(99)
        rand(mt, 13); randn(mt, 5)
        mt_ckpt = WF.make_checkpoint(stage = :produce, step = 1, time = 0.005, manifest = m,
                                     state = st1, box = box, rng = mt, restart_guarantee = :bitwise)
        mt_path = joinpath(dir, "mt_ckpt.json")
        WF.write_checkpoint_atomic(mt_path, mt_ckpt)
        _, _, mt_restored = WF.restore_state(WF.read_checkpoint(mt_path; require_bitwise = true))
        @test [rand(mt) for _ in 1:10] == [rand(mt_restored) for _ in 1:10]
        @test [randn(mt) for _ in 1:10] == [randn(mt_restored) for _ in 1:10]
    end
end

@testset "Bitwise restart continuation (NVE, NVT BAOAB, NPT MC)" begin
    mktempdir() do dir
        box = TY.OrthorhombicBox((10.0, 10.0, 10.0))
        q0 = [1.0 2.1; 1.0 1.0; 1.0 1.0]
        p0 = [0.2 -0.2; 0.1 -0.1; 0.0 0.0]
        at = TY.AtomSet(q0, p0, [1.0, 1.0], [0.0, 0.0], [:A, :B])
        sys = TY.System(at, box; units = TY.REDUCED, temperature = 1.0)
        facs = [FA.HarmonicBond(1, 2, 2.0, 1.0),
                FA.LennardJonesPair(1, 2, 0.2, 0.9; mode = :lfs, rc = 4.0)]
        hsys = DY.HarmonicSystem(sys, facs)

        for (alg, pres) in ((:velocity_verlet, nothing), (:langevin_baoab, nothing), (:npt_baoab_mc, 0.05))
            cfg_full = WF.WorkflowConfig(
                run_id = "cont-$alg", algorithm = alg, dt = 0.005, seed = 777,
                temperature = 1.0, pressure = pres, gamma = 1.5,
                barostat_every = 2, equilibrate_steps = 0, produce_steps = 20,
                frame_every = 5, checkpoint_every = 10,
            )
            cfg_part = WF.WorkflowConfig(
                run_id = "cont-$alg", algorithm = alg, dt = 0.005, seed = 777,
                temperature = 1.0, pressure = pres, gamma = 1.5,
                barostat_every = 2, equilibrate_steps = 0, produce_steps = 10,
                frame_every = 5, checkpoint_every = 10,
            )
            m = WF.build_manifest(run_id = cfg_full.run_id, units = TY.REDUCED, box = box,
                                  algorithm = alg, dt = cfg_full.dt, seed = cfg_full.seed,
                                  n_atoms = 2, temperature = 1.0, pressure = pres, gamma = 1.5)
            mpath = joinpath(dir, "$(alg)_manifest.json")
            WF.write_manifest_atomic(mpath, m)

            full_res = WF.produce_stage(
                DY.State(copy(q0), copy(p0)), hsys, m, Random.Xoshiro(777), cfg_full;
                trajectory_path = joinpath(dir, "$(alg)_full.xyz"),
                checkpoint_dir = joinpath(dir, "$(alg)_full_ckpts"),
            )
            part1_res = WF.produce_stage(
                DY.State(copy(q0), copy(p0)), hsys, m, Random.Xoshiro(777), cfg_part;
                trajectory_path = joinpath(dir, "$(alg)_part1.xyz"),
                checkpoint_dir = joinpath(dir, "$(alg)_part_ckpts"),
            )
            resumed = WF.resume_workflow(
                joinpath(dir, "$(alg)_part_ckpts", "ckpt_produce_00000010.json"),
                mpath, facs, [1.0, 1.0], TY.REDUCED, cfg_part;
                trajectory_path = joinpath(dir, "$(alg)_part2.xyz"),
                checkpoint_dir = joinpath(dir, "$(alg)_part_ckpts2"),
                require_bitwise = true,
            )

            @test reinterpret(UInt64, vec(resumed.state.q)) == reinterpret(UInt64, vec(full_res.state.q))
            @test reinterpret(UInt64, vec(resumed.state.p)) == reinterpret(UInt64, vec(full_res.state.p))
            @test reinterpret(UInt64, vec(resumed.state.F)) == reinterpret(UInt64, vec(full_res.state.F))
            @test resumed.box == full_res.box
            @test resumed.thermostat_state.steps == full_res.thermostat_state.steps
            @test resumed.barostat_state.attempts == full_res.barostat_state.attempts
            @test resumed.barostat_state.accepts == full_res.barostat_state.accepts
            @test rand(resumed.rng) === rand(full_res.rng)
        end
    end
end

@testset "Transactional pipeline & rollback" begin
    mktempdir() do dir
        pdb_path = _write_clean_pdb(joinpath(dir, "clean.pdb"))
        pol = PR.PreparationPolicy(require_box = true)
        builder = p -> begin
            nat = size(p.system.atoms.q, 2)
            FA.AbstractFactor[
                FA.HarmonicBond(1, 2, 5.0, 1.5),
                FA.HarmonicBond(2, 3, 5.0, 1.5),
                FA.HarmonicBond(1, 4, 5.0, 2.0),
            ]
        end
        cfg = WF.WorkflowConfig(
            run_id = "e2e-pdb",
            algorithm = :langevin_baoab,
            dt = 0.01,
            seed = 2026,
            temperature = 1.0,
            gamma = 5.0,
            minimize_max_steps = 20,
            equilibrate_steps = 100,
            produce_steps = 1000,
            frame_every = 10,
            checkpoint_every = 500,
            analysis = WF.AnalysisConfig(min_frames = 20, fit_window = (0.10, 0.40), min_fit_points = 5),
        )
        wdir = joinpath(dir, "good_run")
        res = WF.run_workflow(pdb_path, pol, builder, cfg; workdir = wdir)
        @test res.committed
        @test [s.stage for s in res.stages] == [:validate, :prepare, :minimize, :equilibrate, :produce, :analyze]
        @test all(s -> s.status === :completed, res.stages)
        @test isfile(joinpath(wdir, "manifest.json"))
        @test isfile(joinpath(wdir, "trajectory.xyz"))
        @test isfile(joinpath(wdir, "trajectory.xyz.jsonl"))
        @test isfile(joinpath(wdir, "analysis.json"))
        @test isfile(joinpath(wdir, "checkpoints", "ckpt_latest.json"))
        @test isempty(filter(n -> startswith(n, ".staging_"), readdir(wdir)))

        # Failure in analyze (insufficient frames) must rollback: no manifest/trajectory/analysis committed
        fail_dir = joinpath(dir, "fail_run")
        bad_cfg = WF.WorkflowConfig(
            run_id = "fail-short",
            algorithm = :velocity_verlet,
            dt = 0.005,
            produce_steps = 30,
            frame_every = 10, # only 3 frames < min_frames=20
            analysis = WF.AnalysisConfig(min_frames = 20),
        )
        @test_throws WF.AnalysisGuardError WF.run_workflow(pdb_path, pol, builder, bad_cfg; workdir = fail_dir)
        @test !isfile(joinpath(fail_dir, "manifest.json"))
        @test !isfile(joinpath(fail_dir, "trajectory.xyz"))
        @test !isfile(joinpath(fail_dir, "analysis.json"))
        @test isempty(filter(n -> startswith(n, ".staging_"), readdir(fail_dir)))

        # Failure in validate (non-conservative :trunc LJ factor) refused cleanly
        bad_builder = _ -> FA.AbstractFactor[FA.LennardJonesPair(1, 2, 1.0, 1.0; mode = :trunc, rc = 3.0)]
        val_fail_dir = joinpath(dir, "val_fail")
        @test_throws WF.ValidationError WF.run_workflow(pdb_path, pol, bad_builder, cfg; workdir = val_fail_dir)
        @test !isfile(joinpath(val_fail_dir, "manifest.json"))
    end
end

@testset "Strict analyze guards and synthetic Brownian/ballistic MSD" begin
    # 1. Synthetic 3D Brownian random walk with known D = 0.25 in OrthorhombicBox
    rng = Random.Xoshiro(314159)
    D_true = 0.25
    dt_f = 0.01
    nframes = 300
    nat = 80
    L = 12.0
    box = TY.OrthorhombicBox((L, L, L))
    q_unwrap = zeros(3, nat)
    frames_wrapped = Matrix{Float64}[]
    boxes = TY.AbstractBox[]
    times = Float64[]
    sigma = sqrt(2 * D_true * dt_f)
    for k in 1:nframes
        if k > 1
            q_unwrap .+= sigma .* randn(rng, 3, nat)
        end
        q_w = mod.(q_unwrap, L)
        push!(frames_wrapped, q_w)
        push!(boxes, box)
        push!(times, (k - 1) * dt_f)
    end
    ana = WF.analyze(frames_wrapped, times, boxes;
                     config = WF.AnalysisConfig(min_frames = 50, fit_window = (0.10, 0.35), min_fit_points = 10))
    @test abs(ana.diffusion_coefficient - D_true) / D_true < 0.15
    @test ana.fit_r2 > 0.98
    @test ana.unwrapped

    # 2. Exact constant-velocity ballistic MSD in TriclinicBox: r(t) = v*t -> MSD(tau) = |v|^2 * tau^2
    H = [15.0 2.0 0.0; 0.0 15.0 1.0; 0.0 0.0 15.0]
    tbox = TY.TriclinicBox(H)
    v = [0.3, -0.2, 0.1]
    tframes = Matrix{Float64}[]
    tboxes = fill(tbox, 40)
    ttimes = collect(0.0:0.1:3.9)
    for t in ttimes
        q_abs = repeat(v .* t, 1, 5)
        f = tbox.Hinv * q_abs
        q_w = tbox.H * (f .- floor.(f))
        push!(tframes, q_w)
    end
    t_ana = WF.analyze(tframes, ttimes, tboxes;
                       config = WF.AnalysisConfig(min_frames = 20, fit_window = (0.10, 0.50), min_fit_points = 5))
    @test isapprox(t_ana.msd_values[10], sum(abs2, v) * (t_ana.msd_lags[10])^2; atol = 1e-12)

    # 3. Guard refusals
    @test_throws WF.AnalysisGuardError WF.analyze(frames_wrapped[1:5], times[1:5], boxes[1:5])
    bad_times = copy(times); bad_times[10] = bad_times[9]
    @test_throws WF.AnalysisGuardError WF.analyze(frames_wrapped, bad_times, boxes)
    @test_throws WF.AnalysisGuardError WF.analyze(frames_wrapped, times, boxes;
        config = WF.AnalysisConfig(fit_window = (-0.1, 0.3)))
    @test_throws WF.AnalysisGuardError WF.analyze(frames_wrapped, times, boxes;
        config = WF.AnalysisConfig(fit_window = (0.4, 0.2)))
    @test_throws WF.AnalysisGuardError WF.analyze(frames_wrapped, times, boxes;
        config = WF.AnalysisConfig(fit_window = (0.20, 0.202), min_fit_points = 5))
    @test_throws WF.AnalysisGuardError WF.analyze(frames_wrapped, times, boxes;
        config = WF.AnalysisConfig(lag_window = (-0.5, 0.5)))

    # Exact lag_window index selection (no off-by-one expansion)
    lags_test = collect(0.1:0.1:1.0)
    idx_range = WF._validate_fit_window(WF.AnalysisConfig(lag_window = (0.3, 0.5), min_fit_points = 3), lags_test)
    @test idx_range == 3:5

    # Ambiguous >= half-box jump during periodic unwrap (exact half-box 0.5*L and custom threshold)
    jump_frames = [zeros(3, 2) for _ in 1:25]
    jump_frames[10][1, 1] = 0.50 * L
    @test_throws WF.AnalysisGuardError WF.analyze(jump_frames, collect(0.0:0.1:2.4), fill(box, 25))
    jump_frames[10][1, 1] = 0.46 * L
    @test_throws WF.AnalysisGuardError WF.analyze(jump_frames, collect(0.0:0.1:2.4), fill(box, 25);
        config = WF.AnalysisConfig(max_step_half_box_fraction = 0.45))

    # Insufficient ESS on slowly varying scalar series
    slow_e = [sin(0.02 * i) for i in 1:30]
    @test_throws WF.AnalysisGuardError WF.analyze(
        tframes[1:30], ttimes[1:30], tboxes[1:30];
        config = WF.AnalysisConfig(min_frames = 20, min_ess = 10.0),
        energies = slow_e)
end

@testset "Constrained NPT BAOAB MC workflow with inferred molecules & StreamedLJFactor" begin
    mktempdir() do dir
        box0 = TY.OrthorhombicBox((10.0, 10.0, 10.0))
        d_bond = 1.2
        q0 = [
            1.0 (1.0 + d_bond) 6.0 (6.0 + d_bond);
            2.0 2.0            5.0 5.0;
            2.0 2.0            5.0 5.0
        ]
        p0 = [
            0.1 -0.1  0.2 -0.2;
            0.0  0.0 -0.1  0.1;
            0.1 -0.1  0.0  0.0
        ]
        m = [1.0, 1.0, 1.0, 1.0]
        con_pairs = [(1, 2, d_bond), (3, 4, d_bond)]
        cs = CS.CoupledConstraintSet(4, con_pairs)
        CS.project_velocities!(q0, p0, m, cs)

        # 1. Infer molecules using MB.infer_molecules
        inferred = MB.infer_molecules(4; constraints = con_pairs)
        @test inferred == [[1, 2], [3, 4]]

        eps = [0.2, 0.2, 0.2, 0.2]
        sig = [1.0, 1.0, 1.0, 1.0]
        rcut = 3.0
        lj_fac = SLJ.StreamedLJFactor(q0, eps, sig, box0; rcut = rcut, mode = :lfs)

        at = TY.AtomSet(q0, p0, m, zeros(4), [:A, :A, :B, :B])
        sys = TY.System(at, box0; units = TY.REDUCED, temperature = 1.0)
        hsys = DY.HarmonicSystem(sys, [lj_fac])

        # 2. Refusal check when constraints are active but molecules is nothing
        cfg_refusal = WF.WorkflowConfig(
            run_id = "test-refuse", algorithm = :npt_baoab_mc, dt = 0.005, seed = 42,
            temperature = 1.0, pressure = 0.05, gamma = 1.0,
            equilibrate_steps = 0, produce_steps = 10,
            molecules = nothing
        )
        @test_throws WF.ValidationError WF.validate_stage(sys, [lj_fac], cfg_refusal; constraints = cs)
        stage_spec = WF.StageSpec(:produce, cfg_refusal)
        @test_throws WF.ValidationError WF.validate_stage(stage_spec, hsys, box0; constraints = cs)

        # 3. Successful constrained NPT simulation with inferred molecules and bitwise restart
        cfg_full = WF.WorkflowConfig(
            run_id = "npt-con-full", algorithm = :npt_baoab_mc, dt = 0.005, seed = 1234,
            temperature = 1.0, pressure = 0.05, gamma = 1.0,
            barostat_every = 2, equilibrate_steps = 0, produce_steps = 20,
            frame_every = 5, checkpoint_every = 10,
            molecules = inferred, barostat_coupling = :isotropic
        )
        cfg_part = WF.WorkflowConfig(
            run_id = "npt-con-full", algorithm = :npt_baoab_mc, dt = 0.005, seed = 1234,
            temperature = 1.0, pressure = 0.05, gamma = 1.0,
            barostat_every = 2, equilibrate_steps = 0, produce_steps = 10,
            frame_every = 5, checkpoint_every = 10,
            molecules = inferred, barostat_coupling = :isotropic
        )
        man = WF.build_manifest(
            run_id = cfg_full.run_id, units = TY.REDUCED, box = box0,
            algorithm = :npt_baoab_mc, dt = cfg_full.dt, seed = cfg_full.seed,
            n_atoms = 4, temperature = 1.0, pressure = 0.05, gamma = 1.0
        )
        man_path = joinpath(dir, "man.json")
        WF.write_manifest_atomic(man_path, man)

        full_res = WF.produce_stage(
            DY.State(copy(q0), copy(p0)), hsys, man, Random.Xoshiro(1234), cfg_full;
            constraints = cs,
            trajectory_path = joinpath(dir, "full.xyz"),
            checkpoint_dir = joinpath(dir, "full_ckpts"),
        )

        # Zero constraint drift on final coordinates
        res_final = CS.constraint_residuals(cs, full_res.state.q, full_res.box)
        @test maximum(abs, res_final) < 1e-10

        # Part 1 run
        part1_res = WF.produce_stage(
            DY.State(copy(q0), copy(p0)), hsys, man, Random.Xoshiro(1234), cfg_part;
            constraints = cs,
            trajectory_path = joinpath(dir, "part1.xyz"),
            checkpoint_dir = joinpath(dir, "part_ckpts"),
        )
        res_part1 = CS.constraint_residuals(cs, part1_res.state.q, part1_res.box)
        @test maximum(abs, res_part1) < 1e-10

        # Deterministic checkpoint restart from step 10
        ckpt_10 = joinpath(dir, "part_ckpts", "ckpt_produce_00000010.json")
        resumed = WF.resume_workflow(
            ckpt_10, man_path, [lj_fac], m, TY.REDUCED, cfg_part;
            constraints = cs,
            trajectory_path = joinpath(dir, "part2.xyz"),
            checkpoint_dir = joinpath(dir, "part_ckpts2"),
            require_bitwise = true,
        )

        # Verify exact bitwise match with the uninterrupted run
        @test reinterpret(UInt64, vec(resumed.state.q)) == reinterpret(UInt64, vec(full_res.state.q))
        @test reinterpret(UInt64, vec(resumed.state.p)) == reinterpret(UInt64, vec(full_res.state.p))
        @test reinterpret(UInt64, vec(resumed.state.F)) == reinterpret(UInt64, vec(full_res.state.F))
        @test resumed.box == full_res.box
        @test resumed.thermostat_state.steps == full_res.thermostat_state.steps
        @test resumed.barostat_state.attempts == full_res.barostat_state.attempts
        @test resumed.barostat_state.accepts == full_res.barostat_state.accepts
        @test rand(resumed.rng) === rand(full_res.rng)
    end
end
