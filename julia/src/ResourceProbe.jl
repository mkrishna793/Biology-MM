module ResourceProbe

export MIN_RESERVE_BYTES, parse_vm_stat_available, available_memory, memory_preflight,
       peak_rss_bytes, pme_memory_bytes, md_memory_budget

const MIN_RESERVE_BYTES = 256 * 2^20

"""
    parse_vm_stat_available(output) -> Union{Int,Nothing}

Parse macOS `vm_stat` page size and the free, inactive, and speculative page
counts. Missing/malformed counters or overflowing arithmetic fail closed.
"""
function parse_vm_stat_available(output::AbstractString)
    page_size = nothing
    counts = Dict{String,Int}()
    for line in eachline(IOBuffer(output))
        header = match(r"\bpage\s+size\s+(?:of|:)\s*([0-9][0-9,]*)\s+bytes\b"i, line)
        if header !== nothing
            page_size === nothing || return nothing
            page_size = tryparse(Int, replace(header[1], "," => ""))
            (page_size !== nothing && page_size > 0) || return nothing
        end
        key = match(r"^\s*Pages\s+(free|inactive|speculative)\s*:"i, line)
        key === nothing && continue
        value = match(r"^\s*Pages\s+(free|inactive|speculative)\s*:\s*([0-9][0-9,]*)\.?\s*$"i, line)
        value === nothing && return nothing
        name = lowercase(value[1])
        haskey(counts, name) && return nothing
        parsed = tryparse(Int, replace(value[2], "," => ""))
        parsed === nothing && return nothing
        counts[name] = parsed
    end
    (page_size !== nothing && length(counts) == 3) || return nothing
    try
        pages = Base.Checked.checked_add(counts["free"], counts["inactive"])
        pages = Base.Checked.checked_add(pages, counts["speculative"])
        return Base.Checked.checked_mul(pages, page_size)
    catch e
        e isa OverflowError || rethrow()
        return nothing
    end
end

"""
    available_memory(; platform, vm_stat_output, portable_bytes)

On macOS use `vm_stat` available pages, never `Sys.free_memory()` as an
implicit fallback: failed probes report zero and prevent allocation. On other
platforms explicitly fall back to `Sys.free_memory()`. Optional inputs allow
deterministic tests without invoking the host's memory probe.
"""
function available_memory(; platform::Symbol = Sys.isapple() ? :macos : :portable,
                          vm_stat_output::Union{Nothing,AbstractString} = nothing,
                          portable_bytes::Union{Nothing,Integer} = nothing)
    if platform === :macos
        output = vm_stat_output
        if output === nothing
            output = try
                read(`vm_stat`, String)
            catch
                nothing
            end
        end
        available = output === nothing ? nothing : parse_vm_stat_available(output)
        return (; available = something(available, 0),
                source = available === nothing ? :vm_stat_unavailable : :vm_stat)
    elseif platform === :portable
        bytes = portable_bytes === nothing ? Sys.free_memory() : portable_bytes
        return (; available = max(bytes, 0), source = :sys_free_memory)
    end
    throw(ArgumentError("platform must be :macos or :portable"))
end

"""
    memory_preflight(estimate; probe, reserve_bytes, min_available_bytes,
                     total_bytes, total_fraction)

Allow allocation only when the positive estimate fits within 25% of available
memory AND leaves at least 256 MiB untouched. Optional total-memory fraction
and availability floor preserve benchmark-specific stricter policies.
"""
function memory_preflight(estimate::Integer;
                          probe = available_memory(),
                          reserve_bytes::Integer = MIN_RESERVE_BYTES,
                          min_available_bytes::Integer = 0,
                          total_bytes::Integer = Sys.total_memory(),
                          total_fraction::Bool = false)
    reserve_bytes >= MIN_RESERVE_BYTES ||
        throw(ArgumentError("reserve_bytes must be at least 256 MiB"))
    min_available_bytes >= 0 || throw(ArgumentError("min_available_bytes must be nonnegative"))
    available = probe.available
    available >= 0 || throw(ArgumentError("available bytes must be nonnegative"))
    fraction_limit = div(available, 4)
    reserve_limit = available >= reserve_bytes ? available - reserve_bytes : 0
    total_limit = total_bytes > 0 ? div(total_bytes, 4) : 0
    limit = min(fraction_limit, reserve_limit,
                total_fraction ? total_limit : typemax(Int))
    fraction_ok = estimate > 0 && available > 0 && estimate <= fraction_limit
    reserve_ok = estimate > 0 && available >= reserve_bytes && estimate <= reserve_limit
    total_ok = !total_fraction || (total_bytes > 0 && estimate <= total_limit)
    floor_ok = available >= min_available_bytes
    ok = fraction_ok && reserve_ok && total_ok && floor_ok
    return (; est = estimate, available, source = probe.source, total = total_bytes,
            fraction_limit, reserve_limit, total_limit, limit,
            reserve = reserve_bytes, min_available = min_available_bytes,
            fraction_ok, reserve_ok, total_ok, floor_ok, ok)
end

"""
    peak_rss_bytes() -> Int

Return the process peak resident set size in bytes (`Sys.maxrss()`).
"""
peak_rss_bytes() = Int(Sys.maxrss())

"""
    pme_memory_bytes(atoms, dims, order, ncell; n_cell_pairs=nothing) -> Int

Deterministic working-set byte estimate for a `PME.PMEPlan` with `atoms` sites,
3D FFT grid `dims`, B-spline `order` (4 or 6), and `ncell` real-space cells.
Accounts for the real grid (`Float64`), half-complex rFFT grid (`ComplexF64`),
influence grid (`Float64`), per-atom spline/cell/scratch buffers, and cell-pair
topology.
"""
function pme_memory_bytes(atoms::Integer, dims::NTuple{3,Integer},
                          order::Integer = 4, ncell::Integer = 1;
                          n_cell_pairs::Union{Nothing,Integer} = nothing)
    atoms >= 0 || throw(ArgumentError("atoms must be nonnegative"))
    all(>=(0), dims) || throw(ArgumentError("grid dimensions must be nonnegative"))
    (order == 4 || order == 6) || throw(ArgumentError("order must be 4 or 6"))
    ncell >= 1 || throw(ArgumentError("ncell must be >= 1"))
    n_cell_pairs === nothing || n_cell_pairs >= 0 ||
        throw(ArgumentError("n_cell_pairs must be nonnegative"))

    n = Int(atoms)
    n1, n2, n3 = Int(dims[1]), Int(dims[2]), Int(dims[3])
    p = Int(order)
    nc = Int(ncell)

    ngrid = n1 * n2 * n3
    nk = (n1 ÷ 2 + 1) * n2 * n3
    grid_bytes = 8 * ngrid + 16 * nk + 8 * nk

    # Per-atom: nodes (P*3*8) + weights+dweights (2*P*3*8) +
    # xbuf/sxbuf/sfbuf (9*8) + qqbuf/sqbuf (2*8) + cell_id/perm (2*8)
    per_atom_bytes = (3 * p * 3 + 13) * 8
    atom_bytes = n * per_atom_bytes

    npairs = n_cell_pairs === nothing ? min(nc * 64, nc * (nc + 1) ÷ 2) : Int(n_cell_pairs)
    cell_bytes = 16 * nc + 8 + 16 * npairs
    return grid_bytes + atom_bytes + cell_bytes
end

"""
    md_memory_budget(; atoms, neighbor_capacity=0, dims=(0,0,0), order=4,
                       lj_cells=0, pme_cells=1, constraints=0, overhead=0.25)

Compute a structured, deterministic MD memory budget (in bytes) as a function of
`atoms`, `neighbor_capacity` (total neighbor-pair entries), PME mesh `dims`,
B-spline `order`, cell counts, and `constraints`. Returns component byte counts,
`subtotal_bytes`, and `total_bytes` (with `overhead` headroom).
"""
function md_memory_budget(; atoms::Integer,
                          neighbor_capacity::Integer = 0,
                          dims::NTuple{3,Integer} = (0, 0, 0),
                          order::Integer = 4,
                          lj_cells::Integer = 0,
                          pme_cells::Integer = 1,
                          constraints::Integer = 0,
                          overhead::Real = 0.25)
    atoms >= 0 || throw(ArgumentError("atoms must be nonnegative"))
    neighbor_capacity >= 0 || throw(ArgumentError("neighbor_capacity must be nonnegative"))
    all(>=(0), dims) || throw(ArgumentError("mesh dimensions must be nonnegative"))
    (order == 4 || order == 6) || throw(ArgumentError("order must be 4 or 6"))
    lj_cells >= 0 || throw(ArgumentError("lj_cells must be nonnegative"))
    pme_cells >= 1 || throw(ArgumentError("pme_cells must be >= 1"))
    constraints >= 0 || throw(ArgumentError("constraints must be nonnegative"))
    overhead >= 0 || throw(ArgumentError("overhead must be nonnegative"))

    n = Int(atoms)
    # State: q (3N), p (3N), F (3N), masses (N), charges (N) -> 11*8*N
    state_bytes = 11 * 8 * n
    # Neighbors: pair entries (2 Int64 per pair = 16B) + per-atom offsets/ref coords
    neighbor_bytes = 16 * Int(neighbor_capacity) + 5 * 8 * n
    # StreamedLJ workspace: coords/forces/eps/sig/next/head
    lj_bytes = (6 * 8 + 2 * 8 + 4) * n + 16 * Int(lj_cells)
    # PME workspace
    pme_bytes = dims == (0, 0, 0) ? 0 :
        pme_memory_bytes(n, dims, order, pme_cells)
    # Constraints: coupled triples + projection scratch
    constraint_bytes = 152 * Int(constraints)

    subtotal_bytes = state_bytes + neighbor_bytes + lj_bytes + pme_bytes + constraint_bytes
    total_bytes = ceil(Int, subtotal_bytes * (1.0 + Float64(overhead)))
    return (; atoms = n,
              neighbor_capacity = Int(neighbor_capacity),
              dims = (Int(dims[1]), Int(dims[2]), Int(dims[3])),
              order = Int(order),
              state_bytes, neighbor_bytes, lj_bytes, pme_bytes, constraint_bytes,
              subtotal_bytes, total_bytes, overhead = Float64(overhead))
end

end # module ResourceProbe
