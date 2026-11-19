using FoundationCore
using Test
using LinearAlgebra

const WFR = FoundationCore.Workflow
const TYR = FoundationCore.Types
const FAR = FoundationCore.Factors
const DYR = FoundationCore.Dynamics

@testset "Workflow RESPA NVE" begin
    q = [0.0 1.4; 0.0 0.0; 0.0 0.0]
    p = [0.1 -0.1; 0.0 0.0; 0.0 0.0]
    sys = TYR.System(TYR.AtomSet(q, p, [1.0, 1.0], [0.0, 0.0], [:A, :B]), TYR.NoBox(); units=TYR.REDUCED)
    factors = [FAR.HarmonicBond(1, 2, 2.0, 1.0), FAR.HarmonicBond(1, 2, 0.2, 1.2)]
    cfg = WFR.WorkflowConfig(algorithm=:respa_nve, fast_factors=["1:HarmonicBond"], respa_m=3,
                             temperature=0.0, gamma=nothing, dt=0.001, equilibrate_steps=0,
                              produce_steps=4, frame_every=1, checkpoint_every=2,
                               analysis=WFR.AnalysisConfig(min_frames=4, min_fit_points=2, fit_window=(0.0, 1.0), unwrap=false, min_ess=1.0))
    @test_throws WFR.ValidationError WFR.validate_stage(sys, factors, cfg; constraints=:unsupported)
    bad = WFR.WorkflowConfig(algorithm=:respa_nve, fast_factors=[99], temperature=0.0,
                             gamma=nothing, equilibrate_steps=0, produce_steps=2, frame_every=1)
    @test_throws WFR.ValidationError WFR.validate_stage(sys, factors, bad)
    @test_throws WFR.ValidationError WFR.WorkflowConfig(algorithm=:respa_nve, fast_factors=[1], temperature=1.0, gamma=nothing)
    @test_throws WFR.ValidationError WFR._resolve_fast_factors(factors, [1,1])
    @test_throws WFR.ValidationError WFR._resolve_fast_factors(factors, ["HarmonicBond"])

    mktempdir() do dir
        result = WFR.run_workflow(sys, factors, cfg; workdir=dir)
        @test result.manifest.algorithm == :respa_nve
        @test result.manifest.resolved_parameters["integrator"]["respa_m"] == 3
        @test result.manifest.resolved_parameters["integrator"]["fast_factors"] == ["1:HarmonicBond"]
        cp = WFR.read_checkpoint(result.final_checkpoint_path; expected_manifest_digest=result.manifest.manifest_digest)
        @test cp.step == 4
        @test isfile(result.trajectory_path)
        other = WFR.WorkflowConfig(algorithm=:respa_nve, fast_factors=[1], respa_m=2,
            temperature=0.0, gamma=nothing, dt=cfg.dt)
        changed = WFR.prepare_stage(WFR.validate_stage(sys,factors,other),other)
        @test changed.manifest.manifest_digest != result.manifest.manifest_digest

        # Symmetric integration is reversible when momenta are negated after forward evolution.
        valid = WFR.validate_stage(sys, factors, cfg)
        prep = WFR.prepare_stage(valid, cfg)
        forward = WFR.produce_stage(prep.state, prep.harmonic_system, prep.manifest, prep.rng, cfg;
                                    trajectory_path=joinpath(dir, "reverse.xyz"))
        back = DYR.State(copy(forward.state.q), -copy(forward.state.p))
        hfast = FoundationCore.Respa.RespaIntegrator(prep.harmonic_system, [1])
        for _ in 1:cfg.produce_steps
            FoundationCore.Respa.step!(back, hfast, cfg.dt; m=cfg.respa_m)
        end
        @test norm(back.q - prep.state.q) < 1e-10
        @test norm(back.p + prep.state.p) < 1e-10
    end
end
