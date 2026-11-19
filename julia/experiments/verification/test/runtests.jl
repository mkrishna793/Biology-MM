#!/usr/bin/env julia
"""
Assertion suite for the three lanes.

Run with the isolated environment:

    julia --project=\$HOME/.cache/biology-mm/verification-env test/runtests.jl

Every threshold below is a property of the measured quantity, not a golden
number copied from a previous run: derivative errors are compared against
double-precision limits, interval checks are containment predicates, and the
MBAR error is compared with its own bootstrap standard error.
"""

using Test
using Printf

include(joinpath(@__DIR__, "..", "common.jl"))
include(joinpath(@__DIR__, "..", "lane_derivatives.jl"))
include(joinpath(@__DIR__, "..", "lane_intervals.jl"))
include(joinpath(@__DIR__, "..", "lane_mbar.jl"))

@testset "verification lanes" begin
    @testset "derivatives" begin
        d = run_derivatives()
        @printf("derivatives: parity=%.3e taylor=%.3e enzyme=%.3e fd=%.3e tangent=%.3e\n",
                d.energy_parity, d.taylor_force_err, d.enzyme_force_err,
                d.fd_direction_err, d.tangent_derivative)
        @test d.energy_parity < 1e-12
        @test d.taylor_force_err < 1e-9
        @test d.enzyme_force_err < 1e-9
        @test d.fd_direction_err < 1e-6
        @test d.tangent_derivative < 1e-10
    end

    @testset "intervals" begin
        b = run_intervals()
        @printf("intervals: width=%.3e samples=%d/%d sym=%.3e step=%.3e\n",
                b.chain_energy_width, b.chain_energy_contained,
                b.chain_energy_samples, b.barostat_symmetry_err,
                b.barostat_step_consistency_err)
        @test b.chain_energy_contained == b.chain_energy_samples
        @test b.chain_energy_width > 0.0
        @test isfinite(b.chain_energy_width)
        @test b.bond_force_contained
        @test b.constraint_contained
        @test b.constraint_residual_contained
        @test b.decorations_ok
        @test b.barostat_contained
        @test b.barostat_symmetry_err < 1e-12
        @test b.barostat_step_consistency_err < 1e-12
    end

    @testset "mbar" begin
        m = run_mbar()
        @printf("mbar: iterations=%d samples=%d max_err=%.3e max_se=%.3e\n",
                m.iterations, m.n_samples, m.max_abs_err, m.max_bootstrap_se)
        @test m.iterations < 20000
        @test m.max_bootstrap_se < 0.1
        @test m.max_abs_err < 4.0 * m.max_bootstrap_se + 1e-6
        @test maximum(abs, m.delta_f_est .- m.delta_f_exact) == m.max_abs_err
    end
end
