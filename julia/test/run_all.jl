# Single-command test runner for the whole FoundationCore suite.
#
#     julia --project=. test/run_all.jl
#
# Every suite is executed in ONE Julia process, each inside its own fresh
# module:
#
#     m = Module(Symbol("Suite_", name))
#     Base.include(m, path)
#
# This is not cosmetic. Each suite is written to run standalone: it does its
# own `using FoundationCore` and defines top-level aliases such as
# `const T = FoundationCore.Types`. Running two of them in the same module
# would be an immediate error (cannot `const`-redefine `T`), and same-named
# helper functions would silently shadow each other. A fresh module per suite
# gives every file the namespace it assumes, and `using FoundationCore` still
# resolves inside it because the package is loaded from the active project.
#
# A suite PASSES iff its include did not throw — the suites are assertion
# driven (`@assert`), so an uncaught throw is exactly a failed check. The
# process exits nonzero if any suite failed, so this is usable in CI.

using Printf

const TEST_DIR = @__DIR__

# Run order: cheap deterministic suites first, cross-module integration last.
const SUITES = [
    "test_factors",
    "test_resolved_system",
    "test_system_contract",
    "test_neighbors",
    "test_neighbors_cache",
    "test_dynamics",
    "test_respa",
    "test_frame_writer",
    "test_sampling",
    "test_polarizable",
    "test_prepare",
    "test_structure",
    "test_chemistry_schema",
    "test_chemistry_factors",
    "test_chemistry_cmap",
    "test_chemistry_build",
    "test_chemistry_diagnostics",
    "test_chemistry_mutations",
    "test_chemistry_fixtures",
    "test_chemistry_dynamics",
    "test_shadow",
    "test_integration",
    "test_parameter_gradients",
    "test_streamed_lj",
    "test_streamed_lj_dynamics",
    "test_threading",
    "test_resource_probe",
    "test_equilibrium_sensitivities",
    "test_polarization_operator",
    "test_electrostatics",
    "test_electrostatics_dynamics",
    "test_pme",
    "test_pme_dynamics",
    "test_constraints",
    "test_settle",
    "test_barostat",
    "test_water_box",
    "test_waterbox_cache",
    "test_nve_sampling",
    "test_workflow",
    "test_workflow_respa",
    "test_cli_md",
    "test_cli_imported",
    "test_workflow_recovery",
    "test_trajectory_format_extxyz",
    "test_tile_cache_scale",
    "test_alchemical",
    "test_multiscale",
    "test_trium_xbt",
    "test_ternary_msm",
    "test_symplectic_ternary",
    "test_trium_gpu",
    "test_official_molecules_md",
    "test_production_e2e_gpu_cpu",
    "test_custom_and_reactive",
    "test_gcmc_and_nemd",
    "test_enhanced_sampling",
    "test_phase9_nanotech_e2e",
]

"""
Run one suite file inside a fresh isolated module. Returns
`(name, status, seconds)` with status in {"PASS", "FAIL", "MISSING"}.
"""
function run_suite(name::AbstractString)
    path = joinpath(TEST_DIR, name * ".jl")
    isfile(path) || return (String(name), "MISSING", 0.0)
    m = Module(Symbol("Suite_", name))
    t0 = time()
    status = "PASS"
    err_msg = ""
    try
        Base.include(m, path)
    catch e
        status = "FAIL"
        err_msg = sprint(showerror, e, catch_backtrace())
    end
    secs = time() - t0
    if status === "FAIL"
        println("")
        println("!!! SUITE $name FAILED — error (stack truncated to 12 lines):")
        lines = split(rstrip(err_msg, '\n'), '\n')
        for l in lines[1:min(12, length(lines))]
            println("!!!   ", l)
        end
    end
    return (String(name), status, secs)
end

function main()
    println("=" ^ 78)
    println("FoundationCore test runner: ", length(SUITES),
            " suites, one Julia process, isolated module per suite")
    println("=" ^ 78)
    results = Tuple{String, String, Float64}[]
    for name in SUITES
        println("")
        println(">>> BEGIN SUITE ", name)
        println("-" ^ 78)
        r = run_suite(name)
        println("-" ^ 78)
        @printf(">>> END SUITE %-16s %s  (%.2fs)\n", r[1], r[2], r[3])
        push!(results, r)
    end

    println("")
    println("=" ^ 78)
    println("SUMMARY")
    println("=" ^ 78)
    @printf("%-24s %-8s %10s\n", "SUITE", "STATUS", "TIME(s)")
    println("-" ^ 78)
    npass = 0
    nfail = 0
    total_secs = 0.0
    for (name, status, secs) in results
        @printf("%-24s %-8s %10.2f\n", name, status, secs)
        total_secs += secs
        if status == "PASS"
            npass += 1
        else
            nfail += 1
        end
    end
    println("-" ^ 78)
    @printf("TOTAL: %d suites, %d passed, %d failed   (wall %.1fs)\n",
            length(results), npass, nfail, total_secs)
    if nfail == 0
        println("ALL SUITES PASSED")
        exit(0)
    else
        println("SUITES FAILED: ", nfail)
        exit(1)
    end
end

main()
