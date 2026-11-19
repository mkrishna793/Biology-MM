# Deterministic macOS and portable resource probes, and hard memory gates.
# Run directly with `julia --project=julia julia/test/test_resource_probe.jl`.

using Random
using FoundationCore

const RP = FoundationCore.ResourceProbe
const PME = FoundationCore.PME

const VM_STAT = """
Mach Virtual Memory Statistics: (page size of 16,384 bytes)
Pages free:                            1,234.
Pages active:                          99999.
Pages inactive:                        2,048.
Pages speculative:                       512.
Pages wired down:                      12345.
"""

function main()
    bytes = (1234 + 2048 + 512) * 16384
    @assert RP.parse_vm_stat_available(VM_STAT) == bytes
    @assert RP.parse_vm_stat_available(replace(VM_STAT, "16,384" => "4096")) ==
            (1234 + 2048 + 512) * 4096
    @assert RP.parse_vm_stat_available(replace(VM_STAT, "Pages inactive:" => "Missing:")) === nothing
    @assert RP.parse_vm_stat_available(replace(VM_STAT, "Mach Virtual Memory Statistics: (page size of 16,384 bytes)" => "no page size")) === nothing
    @assert RP.parse_vm_stat_available(replace(VM_STAT, "Pages free:" => "Pages free: NaN\nPages free:")) === nothing
    @assert RP.parse_vm_stat_available(VM_STAT * "Pages free: 10.\n") === nothing
    @assert RP.parse_vm_stat_available(replace(VM_STAT, "1,234." => "-1.")) === nothing
    @assert RP.parse_vm_stat_available(replace(VM_STAT, "1,234." => "9223372036854775807.")) === nothing
    @assert RP.parse_vm_stat_available(replace(VM_STAT, "16,384" => "0")) === nothing

    mac = RP.available_memory(platform = :macos, vm_stat_output = VM_STAT,
                              portable_bytes = 10^12)
    @assert mac == (; available = bytes, source = :vm_stat)
    failed = RP.available_memory(platform = :macos, vm_stat_output = "invalid",
                                 portable_bytes = 10^12)
    @assert failed == (; available = 0, source = :vm_stat_unavailable)
    @assert RP.available_memory(platform = :portable, portable_bytes = 12345) ==
            (; available = 12345, source = :sys_free_memory)

    reserve = RP.MIN_RESERVE_BYTES
    fake(available) = (; available, source = :test)
    pf = RP.memory_preflight(100; probe = fake(reserve + 100), total_bytes = 1_000_000)
    @assert pf.ok && pf.limit == 100 && pf.available - pf.est == reserve
    @assert !RP.memory_preflight(101; probe = fake(reserve + 100), total_bytes = 1_000_000).ok
    @assert !RP.memory_preflight(1; probe = fake(reserve), total_bytes = 1_000_000).ok
    @assert !RP.memory_preflight(1; probe = fake(0), total_bytes = 1_000_000).ok
    @assert !RP.memory_preflight(0; probe = fake(4 * reserve), total_bytes = 1_000_000).ok
    @assert RP.memory_preflight(reserve; probe = fake(4 * reserve)).ok
    @assert !RP.memory_preflight(reserve + 1; probe = fake(4 * reserve)).ok
    pf = RP.memory_preflight(250; probe = fake(1000), total_bytes = 1_000_000)
    @assert !pf.ok && pf.fraction_limit == 250 && !pf.reserve_ok
    pf = RP.memory_preflight(100; probe = fake(2 * reserve), total_bytes = 399,
                              total_fraction = true)
    @assert !pf.ok && pf.total_limit == 99 && pf.limit == 99
    @assert RP.memory_preflight(99; probe = fake(2 * reserve), total_bytes = 399,
                                total_fraction = true).ok

    floor = 1_500_000_000
    @assert !RP.memory_preflight(1; probe = fake(floor - 1),
                                 min_available_bytes = floor).ok
    @assert RP.memory_preflight(1; probe = fake(floor),
                                min_available_bytes = floor).ok
    @assert !RP.memory_preflight(1; probe = failed).ok
    @assert try
        RP.memory_preflight(1; probe = fake(floor), reserve_bytes = reserve - 1)
        false
    catch e
        e isa ArgumentError
    end

    # Peak RSS probe must return a positive integer
    rss = RP.peak_rss_bytes()
    @assert rss isa Int && rss > 0

    # PME memory budget: monotonicity, invalid-input refusals, and live summarysize match
    b_small = RP.pme_memory_bytes(1_000, (32, 32, 32), 4, 64)
    b_more_atoms = RP.pme_memory_bytes(2_000, (32, 32, 32), 4, 64)
    b_bigger_mesh = RP.pme_memory_bytes(1_000, (64, 64, 64), 4, 64)
    b_order6 = RP.pme_memory_bytes(1_000, (32, 32, 32), 6, 64)
    @assert b_more_atoms > b_small
    @assert b_bigger_mesh > b_small
    @assert b_order6 > b_small
    @assert try; RP.pme_memory_bytes(-1, (32, 32, 32), 4, 64); false; catch e; e isa ArgumentError; end
    @assert try; RP.pme_memory_bytes(10, (32, 32, 32), 5, 64); false; catch e; e isa ArgumentError; end
    @assert try; RP.pme_memory_bytes(10, (32, 32, 32), 4, 0); false; catch e; e isa ArgumentError; end

    rng = MersenneTwister(42)
    N_check = 1_000
    L_check = (N_check / 0.1)^(1 / 3)
    q_check = rand(rng, 3, N_check) .* L_check
    ch_check = rand(rng, N_check) .- 0.5; ch_check .-= sum(ch_check) / N_check
    plan_check = PME.PMEPlan(q_check, (L_check, L_check, L_check);
                             alpha = 0.35, rcut = 8.0, grid = (32, 32, 32), order = 4)
    est_plan = RP.pme_memory_bytes(plan_check.N, plan_check.dims, plan_check.order,
                                   prod(plan_check.cell_n);
                                   n_cell_pairs = length(plan_check.cell_pairs))
    actual_plan = Base.summarysize(plan_check)
    @assert 0.8 * actual_plan <= est_plan <= 1.25 * actual_plan
    F_check = zeros(3, N_check)
    PME.pme_energy_forces!(F_check, plan_check, q_check, ch_check)
    @assert (@allocated PME.pme_energy_forces!(F_check, plan_check, q_check, ch_check)) == 0

    # MD memory budget: component sum, overhead, scaling in atoms/neighbors/mesh
    mb1 = RP.md_memory_budget(atoms = 1_000, neighbor_capacity = 100_000,
                              dims = (32, 32, 32), order = 4,
                              lj_cells = 125, pme_cells = 125, constraints = 333,
                              overhead = 0.25)
    @assert mb1.subtotal_bytes == mb1.state_bytes + mb1.neighbor_bytes +
                                  mb1.lj_bytes + mb1.pme_bytes + mb1.constraint_bytes
    @assert mb1.total_bytes == ceil(Int, mb1.subtotal_bytes * 1.25)
    mb_more_nb = RP.md_memory_budget(atoms = 1_000, neighbor_capacity = 200_000,
                                     dims = (32, 32, 32), order = 4,
                                     lj_cells = 125, pme_cells = 125, constraints = 333)
    @assert mb_more_nb.neighbor_bytes > mb1.neighbor_bytes
    @assert mb_more_nb.total_bytes > mb1.total_bytes
    mb_no_mesh = RP.md_memory_budget(atoms = 1_000, neighbor_capacity = 100_000,
                                     dims = (0, 0, 0))
    @assert mb_no_mesh.pme_bytes == 0
    @assert RP.memory_preflight(mb1.total_bytes; probe = fake(16 * mb1.total_bytes + reserve)).ok
    @assert !RP.memory_preflight(mb1.total_bytes; probe = fake(mb1.total_bytes)).ok

    println("ResourceProbe: vm_stat parsing, fail-closed probe, portable fallback, reserve/limit/floor, memory budget & zero-alloc PME PASS")
end

main()
