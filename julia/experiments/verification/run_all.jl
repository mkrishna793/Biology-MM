#!/usr/bin/env julia
"""
Run the three verification lanes and print their measured quantities.

    julia --project=<isolated-env> run_all.jl

Lane A requires TaylorSeries + Enzyme, lane B IntervalArithmetic, lane C only
stdlibs (plus pymbar for `python/mbar_pymbar.py`). Assertions live in
`test/runtests.jl`; this script is the human-readable report.
"""

include("common.jl")
include("lane_derivatives.jl")
include("lane_intervals.jl")
include("lane_mbar.jl")

using Printf

function main()
    println("Julia ", VERSION, " | threads ", Threads.nthreads())
    println("FoundationCore ", FoundationCore.VERSION, " from ", pkgdir(FoundationCore))

    d = run_derivatives()
    @printf("lane_derivatives : energy_parity=%.3e taylor_err=%.3e enzyme_err=%.3e fd_err=%.3e tangent=%.3e\n",
            d.energy_parity, d.taylor_force_err, d.enzyme_force_err,
            d.fd_direction_err, d.tangent_derivative)

    b = run_intervals()
    @printf("lane_intervals   : energy_width=%.3e contained=%d/%d bond_force=%s constraint=%s/%s barostat=%s decorations=%s\n",
            b.chain_energy_width, b.chain_energy_contained, b.chain_energy_samples,
            b.bond_force_contained, b.constraint_contained,
            b.constraint_residual_contained, b.barostat_contained, b.decorations_ok)
    @printf("lane_intervals   : barostat_symmetry=%.3e barostat_step=%.3e bond_force_width=%.3e\n",
            b.barostat_symmetry_err, b.barostat_step_consistency_err, b.bond_force_width)

    m = run_mbar()
    @printf("lane_mbar        : iterations=%d samples=%d max_abs_err=%.3e max_bootstrap_se=%.3e\n",
            m.iterations, m.n_samples, m.max_abs_err, m.max_bootstrap_se)
    @printf("lane_mbar        : Delta_f_est   = %s\n", join((@sprintf("%.5f", v) for v in m.delta_f_est), " "))
    @printf("lane_mbar        : Delta_f_exact = %s\n", join((@sprintf("%.5f", v) for v in m.delta_f_exact), " "))
    println("lane_mbar        : artifacts in ", m.artifacts_dir)
    return nothing
end

main()
