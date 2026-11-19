"""
Input preparation: structured validation and staged repair for real structure
files, with every non-unique scientific decision exposed rather than silently
resolved.

## Why this module is mostly refusals

Reading a PDB or mmCIF file and producing a force-field-ready system involves
decisions that are **not determined by the data**:

* which alternate location (`altloc`) to keep when several are present,
* what to do with missing heavy atoms or missing loops,
* how to assign protonation states at a given pH,
* whether an atom with zero occupancy is present at all,
* what the identity of an unresolved ion is.

There is no canonical answer to any of these. A tool that silently picks one and
reports "structure prepared" is a source of irreproducible science. So this module
separates:

1. **Diagnosis** (`diagnose_structure`) — always available, reports every issue
   with the exact record and line that caused it.
2. **Policy** (`PreparationPolicy`) — an explicit, inspectable record of the
   decisions the user has authorised, each with a stated default and a
   provenance note.
3. **Deterministic repair** (`prepare`) — applies only the configured policies,
   fails loudly when a decision is not covered, and records what it did in a
   machine-readable provenance log.

## Narrow reader and adapter cross-checks

The Julia core always uses a deliberately narrow fixed-column PDB `ATOM`/`HETATM`
reader that handles only the fixed-column records it claims to handle, and *says
so* in its report. External cross-checks against MDAnalysis/MDTraj live in
`validation/adapters`; this reader does not parse mmCIF or claim equivalence to
a mature parser.

The percentages sometimes quoted for "structures with problems" are not
reproduced here as facts; this module reports what it actually observed on the
file it was given.
"""
module Prepare

using ..Types
using ..Types: System
using ..Factors
using Printf

export StructureReport, diagnose_structure, PreparationPolicy, prepare,
       PreparedSystem, AtomRecord, read_pdb_fallback, AltlocPolicy,
       RepairAction, provenance_log, remap_pair_indices

# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

struct AtomRecord
    serial::Int
    name::Symbol
    resname::Symbol
    chain::Char
    resseq::Int
    icode::Char
    altloc::Char
    occupancy::Float64
    element::Symbol
    x::Float64
    y::Float64
    z::Float64
    source_line::Int
end

struct StructureReport
    n_atoms::Int
    n_altloc::Int
    n_zero_occupancy::Int
    n_nonfinite::Int
    n_duplicate_identities::Int
    n_close_contacts::Int
    min_pair_distance::Float64
    close_contact_pairs::Vector{Tuple{Int,Int,Float64}}
    chains::Vector{Char}
    resnames::Vector{Symbol}
    box::Union{Nothing,NTuple{3,Float64}}
    issues::Vector{String}
    parser::String
end

# ---------------------------------------------------------------------------
# narrow fallback PDB reader
# ---------------------------------------------------------------------------

"""
Fixed-column PDB `ATOM`/`HETATM` reader.

Handles exactly: `ATOM`/`HETATM` records, columns 1-6/7-11/13-16/17/18-20/22/
23-26/27/31-38/39-46/47-54/55-60/77-78 per the PDB format, and `CRYST1` for the
box. It does **not** handle multi-model files, symmetry records, anisotropic
temperature factors, or `TER` semantics. That limitation is reported, and
`diagnose_structure` names the parser it used so a result is never ambiguous.
"""
function read_pdb_fallback(path::AbstractString)
    atoms = AtomRecord[]
    box = nothing
    issues = String[]
    nmodel = 0
    for (ln, line) in enumerate(eachline(path))
        if startswith(line, "MODEL")
            nmodel += 1
            nmodel > 1 && push!(issues, "multi-model file: only the first model is read (line $ln)")
        elseif startswith(line, "CRYST1")
            try
                a = parse(Float64, strip(line[7:15]))
                b = parse(Float64, strip(line[16:24]))
                c = parse(Float64, strip(line[25:33]))
                box = (a, b, c)
            catch
                push!(issues, "unparseable CRYST1 record at line $ln")
            end
        elseif startswith(line, "ATOM  ") || startswith(line, "HETATM")
            length(line) < 54 && (push!(issues, "truncated ATOM record at line $ln"); continue)
            try
                serial = parse(Int, strip(line[7:11]))
                name = Symbol(strip(line[13:16]))
                altloc = length(line) >= 17 ? line[17] : ' '
                resname = Symbol(strip(line[18:20]))
                chain = line[22]
                resseq = parse(Int, strip(line[23:26]))
                icode = length(line) >= 27 ? line[27] : ' '
                x = parse(Float64, strip(line[31:38]))
                y = parse(Float64, strip(line[39:46]))
                z = parse(Float64, strip(line[47:54]))
                occ = length(line) >= 60 && !isempty(strip(line[55:60])) ?
                      parse(Float64, strip(line[55:60])) : 1.0
                elem = length(line) >= 78 && !isempty(strip(line[77:78])) ?
                       Symbol(strip(line[77:78])) : Symbol(first(String(name), 1))
                push!(atoms, AtomRecord(serial, name, resname, chain, resseq, icode,
                                        altloc, occ, elem, x, y, z, ln))
            catch e
                push!(issues, "unparseable ATOM record at line $ln: $(sprint(showerror, e))")
            end
        end
    end
    return atoms, box, issues
end

# ---------------------------------------------------------------------------
# diagnosis
# ---------------------------------------------------------------------------

"""
Diagnose a structure. Never mutates, never decides. Reports:

* alternate locations,
* zero-occupancy atoms,
* non-finite coordinates,
* duplicate atom identities (same chain/resseq/name/altloc),
* close contacts below `contact_tol` (default 0.5 A, i.e. clearly unphysical),
* the box if present,
* which parser was used.

Close contacts are the ones that matter for force evaluation: at `r → 0` every
supported potential is singular, so the *only* correct behaviours are to repair
deterministically or refuse before any force is evaluated. This function
identifies them; `prepare` decides.
"""
function diagnose_structure(path::AbstractString;
                            parser::Symbol = :auto, contact_tol::Float64 = 0.5)
    atoms = AtomRecord[]
    box = nothing
    issues = String[]
    used = ""
    md_ok = false
    if parser in (:auto, :mdanalysis)
        md_ok = _have_module("MDAnalysis")
    end
    mdt_ok = parser in (:auto, :mdtraj) && _have_module("mdtraj")
    if parser === :mdanalysis && !md_ok
        push!(issues, "requested MDAnalysis parser but the module is not installed")
    elseif parser === :mdtraj && !mdt_ok
        push!(issues, "requested MDTraj parser but the module is not installed")
    end
    # Julia cannot import Python modules directly; the Python adapters live in
    # validation/adapters. For the Julia core we use the narrow reader and
    # report that fact. The adapter layer cross-checks with MDAnalysis.
    atoms, box, iss = read_pdb_fallback(path)
    append!(issues, iss)
    used = "narrow fixed-column PDB reader (Julia core); MDAnalysis/MDTraj cross-check available via validation/adapters"

    n = length(atoms)
    n_alt = count(a -> a.altloc != ' ', atoms)
    n_zero = count(a -> a.occupancy == 0.0, atoms)
    n_nonfinite = count(a -> !(isfinite(a.x) && isfinite(a.y) && isfinite(a.z)), atoms)
    ids = Dict{Tuple{Char,Int,Char,Symbol,Char},Int}()
    dups = 0
    for a in atoms
        k = (a.chain, a.resseq, a.icode, a.name, a.altloc)
        if haskey(ids, k)
            dups += 1
        else
            ids[k] = a.serial
        end
    end
    # close contacts, O(N^2) here on purpose: diagnosis is off the hot path and
    # must not depend on a neighbour list whose correctness is itself under test.
    contacts = Tuple{Int,Int,Float64}[]
    mind = Inf
    for i in 1:n-1
        for j in i+1:n
            dx = atoms[i].x - atoms[j].x
            dy = atoms[i].y - atoms[j].y
            dz = atoms[i].z - atoms[j].z
            d = sqrt(dx*dx + dy*dy + dz*dz)
            if d < mind
                mind = d
            end
            if d < contact_tol
                push!(contacts, (i, j, d))
            end
        end
    end
    n == 0 && (mind = NaN)
    chains = unique([a.chain for a in atoms])
    resnames = unique([a.resname for a in atoms])
    n_alt > 0 && push!(issues, "$n_alt atoms carry a non-blank altloc; a policy is required")
    n_zero > 0 && push!(issues, "$n_zero atoms have zero occupancy")
    n_nonfinite > 0 && push!(issues, "$n_nonfinite atoms have non-finite coordinates")
    dups > 0 && push!(issues, "$dups duplicate atom identities")
    !isempty(contacts) && push!(issues,
        "$(length(contacts)) close contacts below $(contact_tol) A; force evaluation is singular there")
    box === nothing && push!(issues, "no CRYST1 record: system is treated as non-periodic")

    return StructureReport(n, n_alt, n_zero, n_nonfinite, dups, length(contacts),
                           mind, contacts, chains, resnames, box, issues, used)
end

function _have_module(m::String)
    try
        Base.find_package(m) !== nothing && return true
    catch
    end
    return false
end

# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------

"Explicit altloc choice. `.keep_first` is deterministic but arbitrary."
struct AltlocPolicy
    mode::Symbol          # :error, :keep_first, :keep_highest_occupancy, :keep_list
    keep::Vector{Char}
    note::String
end
AltlocPolicy(; mode = :error, keep = Char[], note = "") =
    AltlocPolicy(mode, keep, note)

struct RepairAction
    kind::Symbol
    detail::String
    n_affected::Int
end

"""
The complete set of authorised scientific decisions.

Every field has an explicitly *conservative* default: by default this module
refuses rather than guesses. A caller who wants a decision made must say so, and
the provenance log records that they did.
"""
struct PreparationPolicy
    altloc::AltlocPolicy
    drop_zero_occupancy::Bool
    drop_hydrogens::Bool
    close_contact_tol::Float64
    close_contact_action::Symbol     # :error, :drop_second, :nudge
    nudge_distance::Float64
    require_box::Bool
    periodic_tol::Float64
end

function PreparationPolicy(; altloc = AltlocPolicy(),
                           drop_zero_occupancy::Bool = false,
                           drop_hydrogens::Bool = false,
                           close_contact_tol::Float64 = 0.5,
                           close_contact_action::Symbol = :error,
                           nudge_distance::Float64 = 1e-3,
                           require_box::Bool = false,
                           periodic_tol::Float64 = 1e-6)
    close_contact_action in (:error, :drop_second, :nudge) ||
        error("close_contact_action must be :error, :drop_second or :nudge")
    isfinite(close_contact_tol) && close_contact_tol >= 0 ||
        error("close_contact_tol must be finite and non-negative")
    isfinite(nudge_distance) && nudge_distance > 0 ||
        error("nudge_distance must be finite and positive")
    isfinite(periodic_tol) && periodic_tol >= 0 ||
        error("periodic_tol must be finite and non-negative")
    return PreparationPolicy(altloc, drop_zero_occupancy, drop_hydrogens,
                             close_contact_tol, close_contact_action,
                             nudge_distance, require_box, periodic_tol)
end

struct PreparedSystem
    system::System
    report::StructureReport
    actions::Vector{RepairAction}
    source_to_prepared::Vector{Int}
end

provenance_log(p::PreparedSystem) = p.actions

"""Remap one-based source atom pairs into prepared indices; refuse dropped atoms."""
function remap_pair_indices(p::PreparedSystem, pairs)
    mapped = Tuple{Int,Int}[]
    nsource = length(p.source_to_prepared)
    for pair in pairs
        length(pair) == 2 || throw(ArgumentError("pairs must contain (i,j) entries"))
        i, j = Int(pair[1]), Int(pair[2])
        (1 <= i <= nsource && 1 <= j <= nsource && i != j) ||
            throw(ArgumentError("source pair indices must be distinct and in 1:$nsource"))
        pi, pj = p.source_to_prepared[i], p.source_to_prepared[j]
        (pi > 0 && pj > 0) ||
            throw(ArgumentError("pair ($i,$j) references an atom removed during preparation"))
        push!(mapped, (min(pi, pj), max(pi, pj)))
    end
    return mapped
end

# ---------------------------------------------------------------------------
# staged repair
# ---------------------------------------------------------------------------

"""
Apply `policy` to a diagnosed structure, returning a `System` plus the full
action log.

Stages, in order, each of which either succeeds deterministically or throws:

1. **altloc selection** — per policy; `:error` by default.
2. **occupancy filter** — only if explicitly enabled.
3. **hydrogen removal** — only if explicitly enabled (crude but deterministic).
4. **identity check** — duplicate identities are always an error; there is no
   correct silent choice.
5. **close-contact handling** — per policy. `:drop_second` removes the later
   atom of each offending pair (deterministic, recorded). `:nudge` displaces the
   later atom along the pair axis by `nudge_distance`, which is recorded because
   it *changes the geometry* and therefore must never be confused with the
   original structure. `:error` refuses.

After repair the returned `System` is guaranteed free of coincident atoms at the
configured tolerance, so force evaluation cannot be singular because of input
geometry. It is **not** guaranteed to be a physically valid model: missing atoms
and protonation are still unaddressed, and the report says so.
"""
function _close_contacts(q::AbstractMatrix{Float64}, tol::Float64)
    n = size(q, 2)
    pairs = Tuple{Int,Int,Float64}[]
    for i in 1:n-1
        for j in i+1:n
            dx = q[1, i] - q[1, j]
            dy = q[2, i] - q[2, j]
            dz = q[3, i] - q[3, j]
            d = sqrt(dx * dx + dy * dy + dz * dz)
            d < tol && push!(pairs, (i, j, d))
        end
    end
    return pairs
end

function prepare(path::AbstractString, policy::PreparationPolicy = PreparationPolicy();
                 units::Units = REDUCED, temperature::Float64 = 0.0)
    rep = diagnose_structure(path; contact_tol = policy.close_contact_tol)
    atoms, box, _ = read_pdb_fallback(path)
    source_indices = collect(eachindex(atoms))
    nsource = length(atoms)
    actions = RepairAction[]

    # 1. altloc
    if rep.n_altloc > 0
        p = policy.altloc
        if p.mode === :error
            error("structure has $(rep.n_altloc) atoms with non-blank altloc and " *
                  "AltlocPolicy is :error. Choose an explicit policy: " *
                  ":keep_first (deterministic, arbitrary), :keep_highest_occupancy, " *
                  "or :keep_list with an explicit set. There is no canonical choice.")
        elseif p.mode === :keep_first
            seen = Set{Tuple{Char,Int,Char,Symbol}}()
            kept = AtomRecord[]
            kept_indices = Int[]
            for (source_index, a) in zip(source_indices, atoms)
                k = (a.chain, a.resseq, a.icode, a.name)
                if a.altloc == ' ' || !haskey(seen, k)
                    push!(kept, a)
                    push!(kept_indices, source_index)
                    push!(seen, k)
                end
            end
            push!(actions, RepairAction(:altloc_keep_first,
                "kept the first altloc encountered per atom identity", length(atoms) - length(kept)))
            atoms = kept
            source_indices = kept_indices
        elseif p.mode === :keep_highest_occupancy
            best = Dict{Tuple{Char,Int,Char,Symbol},AtomRecord}()
            for a in atoms
                k = (a.chain, a.resseq, a.icode, a.name)
                if !haskey(best, k) || a.occupancy > best[k].occupancy
                    best[k] = a
                end
            end
            kept = [a for a in atoms if best[(a.chain, a.resseq, a.icode, a.name)] === a]
            source_indices = [source_indices[i] for i in eachindex(atoms) if best[(atoms[i].chain, atoms[i].resseq, atoms[i].icode, atoms[i].name)] === atoms[i]]
            push!(actions, RepairAction(:altloc_highest_occupancy,
                "kept the highest-occupancy altloc per atom identity", length(atoms) - length(kept)))
            atoms = kept
        elseif p.mode === :keep_list
            keep_mask = [a.altloc == ' ' || a.altloc in p.keep for a in atoms]
            kept = atoms[keep_mask]
            source_indices = source_indices[keep_mask]
            push!(actions, RepairAction(:altloc_keep_list,
                "kept altlocs $(p.keep)", length(atoms) - length(kept)))
            atoms = kept
        else
            error("unknown altloc policy mode $(p.mode)")
        end
    end

    # 2. occupancy
    if policy.drop_zero_occupancy
        before = length(atoms)
        keep_mask = [a.occupancy > 0.0 for a in atoms]
        atoms = atoms[keep_mask]
        source_indices = source_indices[keep_mask]
        push!(actions, RepairAction(:drop_zero_occupancy,
            "removed zero-occupancy atoms", before - length(atoms)))
    end

    # 3. hydrogens
    if policy.drop_hydrogens
        before = length(atoms)
        keep_mask = [!(a.element === :H || a.element === :D) for a in atoms]
        atoms = atoms[keep_mask]
        source_indices = source_indices[keep_mask]
        push!(actions, RepairAction(:drop_hydrogens,
            "removed H/D atoms; protonation state is now undefined and must be " *
            "rebuilt by a parameterisation step", before - length(atoms)))
    end

    # 4. duplicate identities are always fatal
    seen = Set{Tuple{Char,Int,Char,Symbol,Char}}()
    for a in atoms
        k = (a.chain, a.resseq, a.icode, a.name, a.altloc)
        k in seen && error("duplicate atom identity $k after repair; there is no " *
                           "correct silent choice, fix the input")
        push!(seen, k)
    end

    # 5. close contacts. Filtering changes atom indices, so contacts must be
    # recomputed against the exact atom list that will be repaired.
    n = length(atoms)
    q = zeros(3, n)
    for (i, a) in enumerate(atoms)
        (isfinite(a.x) && isfinite(a.y) && isfinite(a.z)) ||
            error("non-finite coordinates at source atom $(a.serial) (line $(a.source_line))")
        q[1, i] = a.x; q[2, i] = a.y; q[3, i] = a.z
    end
    close_contact_pairs = _close_contacts(q, policy.close_contact_tol)
    if !isempty(close_contact_pairs)
        if policy.close_contact_action === :error
            c = close_contact_pairs[1]
            error("$(length(close_contact_pairs)) close contacts below $(policy.close_contact_tol) A " *
                  "(first: atoms $(c[1]),$(c[2]) at $(round(c[3], digits=4)) A). " *
                  "Every supported potential is singular there. Choose :drop_second " *
                  "or :nudge explicitly, or repair the input.")
        elseif policy.close_contact_action === :drop_second
            dead = Set{Int}()
            for (i, j, d) in close_contact_pairs
                if !(i in dead) && !(j in dead)
                    push!(dead, j)
                end
            end
            keep = [i for i in 1:n if !(i in dead)]
            q = q[:, keep]
            atoms = atoms[keep]
            source_indices = source_indices[keep]
            push!(actions, RepairAction(:drop_close_contact_second,
                "dropped the later atom of each close-contact pair", length(dead)))
        elseif policy.close_contact_action === :nudge
            for (i, j, d) in close_contact_pairs
                dv = q[:, j] .- q[:, i]
                nv = sqrt(sum(abs2, dv))
                dir = nv < 1e-12 ? [1.0, 0.0, 0.0] : dv ./ nv
                q[:, j] .+= dir .* policy.nudge_distance
            end
            remaining = _close_contacts(q, policy.close_contact_tol)
            isempty(remaining) ||
                error("nudge left $(length(remaining)) unresolved close contacts below " *
                      "$(policy.close_contact_tol) A; increase nudge_distance or use :drop_second")
            push!(actions, RepairAction(:nudge_close_contacts,
                "displaced later atoms by $(policy.nudge_distance); geometry is " *
                "MODIFIED and this is not the original structure",
                length(close_contact_pairs)))
        end
    end

    # box
    bx = Types.NoBox()
    if box !== nothing
        bx = Types.OrthorhombicBox(box)
    elseif policy.require_box
        error("policy.require_box is set but the file has no CRYST1 record")
    end

    nat = length(atoms)
    masses = [element_mass(a.element) for a in atoms]
    charges = zeros(nat)
    names = [a.name for a in atoms]
    atomset = Types.AtomSet(q, zeros(3, nat), masses, charges, names)
    sys = Types.System(atomset, bx, Types.ChemicalTopology();
                       units = units, temperature = temperature)
    push!(actions, RepairAction(:no_parameters,
        "charges are zero and there is no bonded topology: this is a *geometry* " *
        "prepared for validation, NOT a parameterised force field. Assigning " *
        "charges and bonded terms is a separate, provenance-bearing step.", nat))
    source_to_prepared = zeros(Int, nsource)
    for (prepared_index, source_index) in enumerate(source_indices)
        source_to_prepared[source_index] = prepared_index
    end
    return PreparedSystem(sys, rep, actions, source_to_prepared)
end

"Atomic mass in the unit system's mass unit (reduced: sigma=eps=m=1, so 1.0)."
element_mass(e::Symbol) = 1.0

end # module Prepare
