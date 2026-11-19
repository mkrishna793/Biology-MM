# Preparation/consistency audit for imported parameterized systems and for raw
# structure files.
#
# The audit *reports*; it never repairs and never fills in missing parameters.
# Every record carries 1-based atom indices (and residue indices where the
# notion applies) so a caller can go back to the exact record in the input.
#
# Two entry points:
#
# * `audit_parameterized(ns, q; protocol, bond_tolerance_nm)` checks a normalized
#   `mdparams` record against a coordinate set and the execution protocol.
# * `audit_raw_structure(path; kwargs...)` delegates to `Prepare.diagnose_structure`
#   and maps the findings it actually made. Missing atoms, unknown residues,
#   protonation ambiguity and ligand parameters are reported only when `Prepare`
#   reports them; the audit never invents them.

using ..Prepare: diagnose_structure, read_pdb_fallback

function _file_sha256(path::AbstractString)
    isfile(path) || return ""
    try
        sha_mod = Base.require(Base.PkgId(Base.UUID("ea8e919c-243c-51af-8825-aaa63cd721ce"), "SHA"))
        return bytes2hex(sha_mod.sha256(read(path)))
    catch
        try
            return String(split(readchomp(`shasum -a 256 $path`))[1])
        catch
            return ""
        end
    end
end

const _AUDIT_ION_RESIDUES = Set(["WAT", "HOH", "TP3", "TIP3", "NA", "CL", "K",
                                 "CA", "MG", "ZN", "SOD", "CLA", "POT"])
const _AUDIT_BENIGN_ZERO_LJ = Set(["H", "HE", "D"])
const _AUDIT_CHAIN_BREAK_NM = 0.25

const _AUDIT_STANDARD_AMINO_ACIDS = Set([
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "HID", "HIE", "HIP",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "ASH", "GLH", "LYN", "CYX", "CYM"
])
const _AUDIT_STANDARD_NUCLEIC_ACIDS = Set([
    "DA", "DC", "DG", "DT", "A", "C", "G", "U", "RA", "RC", "RG", "RU"
])
const _AUDIT_STANDARD_CAPS = Set(["ACE", "NME", "NHE"])

const _KNOWN_STANDARD_RESIDUES_SYM = Set([
    :ALA, :ARG, :ASN, :ASP, :CYS, :GLN, :GLU, :GLY, :HIS, :HID, :HIE, :HIP,
    :ILE, :LEU, :LYS, :MET, :PHE, :PRO, :SER, :THR, :TRP, :TYR, :VAL,
    :ASH, :GLH, :LYN, :CYX, :CYM,
    :DA, :DC, :DG, :DT, :A, :C, :G, :U, :RA, :RC, :RG, :RU,
    :ACE, :NME, :NHE,
    :WAT, :HOH, :TP3, :TIP3, :NA, :CL, :K, :CA, :MG, :ZN, :SOD, :CLA, :POT
])

struct AuditReport
    diagnostics::Vector{DiagnosticRecord}
    provable_errors::Vector{DiagnosticRecord}
    warnings::Vector{DiagnosticRecord}
    choices::Vector{DiagnosticRecord}
    counts::NamedTuple
    provenance::Union{ImportProvenance,Nothing}
    source_file::String
    parameter_version::String
    mapping_decisions::Vector{String}
    hashes::Dict{String,String}
end
AuditReport(diags, errors, warnings, choices, counts) =
    AuditReport(diags, errors, warnings, choices, counts, nothing, "", "", String[], Dict{String,String}())

struct RawAuditReport
    diagnostics::Vector{DiagnosticRecord}
    provable_errors::Vector{DiagnosticRecord}
    warnings::Vector{DiagnosticRecord}
    counts::NamedTuple
    source_file::String
    parameter_version::String
    mapping_decisions::Vector{String}
    hashes::Dict{String,String}
end
RawAuditReport(diags, errors, warnings, counts) =
    RawAuditReport(diags, errors, warnings, counts, "", "", String[], Dict{String,String}())

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

_fmt(x::Real, digits::Int = 6) = string(round(Float64(x), digits = digits))

_pair(i::Int, j::Int) = i < j ? (i, j) : (j, i)

function _pair_distance(q::AbstractMatrix{<:Real}, i::Int, j::Int)
    dx = Float64(q[1, i]) - Float64(q[1, j])
    dy = Float64(q[2, i]) - Float64(q[2, j])
    dz = Float64(q[3, i]) - Float64(q[3, j])
    return sqrt(dx * dx + dy * dy + dz * dz)
end

_residue(ns::NormalizedSystem, i::Int) = ns.atoms[i].residue_index

function _residues(ns::NormalizedSystem, atoms)
    out = Int[]
    for i in atoms
        r = _residue(ns, i)
        r in out || push!(out, r)
    end
    return out
end

_diag(code::Symbol, severity::Symbol, message::AbstractString,
      atoms::Vector{Int} = Int[], residues::Vector{Int} = Int[]) =
    DiagnosticRecord(code, severity, String(message), sort(atoms), unique(residues))

"Deterministic order: code, first atom, message, then stable tie-breakers."
function _audit_sort!(diags::Vector{DiagnosticRecord})
    sort!(diags; by = d -> (String(d.code),
                            isempty(d.atoms) ? 0 : first(d.atoms),
                            d.message,
                            String(d.severity),
                            Tuple(d.atoms),
                            Tuple(d.residues)))
    return diags
end

# ---------------------------------------------------------------------------
# parameterized audit
# ---------------------------------------------------------------------------

"""
    audit_parameterized(ns, q; protocol = default_protocol(ns),
                        bond_tolerance_nm = 0.05) -> AuditReport

Audit a normalized system against coordinates `q` (3 × N, nm). The result
partitions records into `provable_errors`, `warnings` and `choices`, and keeps
all of them in `diagnostics` in deterministic order.

Nothing is repaired: a bad shape returns a single `:shape` error, and a
non-finite coordinate returns `:nonfinite`; both stop the geometric checks
because every subsequent distance would be undefined.
"""
function audit_parameterized(ns::NormalizedSystem, q::AbstractMatrix{<:Real};
                             protocol = default_protocol(ns),
                             bond_tolerance_nm::Real = 0.05)
    N = length(ns.atoms)
    diags = DiagnosticRecord[]

    # --- shape -------------------------------------------------------------
    if size(q, 1) != 3 || size(q, 2) != N
        push!(diags, _diag(:shape, :error,
            "coordinate array has shape $(size(q)); expected (3, $N)"))
        return _finish_parameterized(ns, diags)
    end

    # --- non-finite coordinates -------------------------------------------
    bad = Int[]
    for i in 1:N
        (isfinite(q[1, i]) && isfinite(q[2, i]) && isfinite(q[3, i])) || push!(bad, i)
    end
    if !isempty(bad)
        push!(diags, _diag(:nonfinite, :error,
            "atoms $(bad) have non-finite coordinates; no distance is defined",
            bad, _residues(ns, bad)))
        return _finish_parameterized(ns, diags)
    end

    tol = Float64(bond_tolerance_nm)

    # --- coincident bonded atoms ------------------------------------------
    # Every pair that the topology declares as bonded (bond, Urey-Bradley or
    # distance constraint) must have distinct positions: at r → 0 every
    # supported potential is singular.
    seen_pairs = Set{Tuple{Int,Int}}()
    coincident = Tuple{Int,Int}[]
    function check_coincident(i::Int, j::Int)
        key = _pair(i, j)
        key in seen_pairs && return nothing
        push!(seen_pairs, key)
        if _pair_distance(q, i, j) < 1e-6
            push!(coincident, key)
        end
        return nothing
    end
    for b in ns.bonds
        check_coincident(b.i, b.j)
    end
    for u in ns.urey_bradleys
        check_coincident(u.i, u.j)
    end
    for c in ns.constraints
        check_coincident(c.i, c.j)
    end
    for (i, j) in coincident
        push!(diags, _diag(:coincident_atoms, :error,
            "bonded atoms $i and $j are coincident (r < 1e-6 nm); force evaluation is singular",
            [i, j], _residues(ns, [i, j])))
    end

    # --- bond deviations ---------------------------------------------------
    for b in ns.bonds
        r = _pair_distance(q, b.i, b.j)
        r < 1e-6 && continue          # already reported as :coincident_atoms
        dev = abs(r - b.r0_nm)
        if dev > tol
            push!(diags, _diag(:bond_deviation, :warning,
                "bond ($(b.i),$(b.j)) measures $(_fmt(r, 4)) nm, expected " *
                "$(_fmt(b.r0_nm, 4)) nm (deviation $(_fmt(dev, 4)) nm > " *
                "$(_fmt(tol, 4)) nm)",
                [b.i, b.j], _residues(ns, [b.i, b.j])))
        end
    end

    # --- singular angles ---------------------------------------------------
    for a in ns.angles
        i, j, k = a.i, a.j, a.k
        ux = Float64(q[1, i]) - Float64(q[1, j])
        uy = Float64(q[2, i]) - Float64(q[2, j])
        uz = Float64(q[3, i]) - Float64(q[3, j])
        vx = Float64(q[1, k]) - Float64(q[1, j])
        vy = Float64(q[2, k]) - Float64(q[2, j])
        vz = Float64(q[3, k]) - Float64(q[3, j])
        cx = uy * vz - uz * vy
        cy = uz * vx - ux * vz
        cz = ux * vy - uy * vx
        nu = sqrt(ux * ux + uy * uy + uz * uz)
        nv = sqrt(vx * vx + vy * vy + vz * vz)
        sin_theta = (nu == 0.0 || nv == 0.0) ? 0.0 :
                    sqrt(cx * cx + cy * cy + cz * cz) / (nu * nv)
        if sin_theta < 1e-8
            push!(diags, _diag(:singular_angle, :error,
                "angle ($i,$j,$k) has sin(theta) = $(_fmt(sin_theta, 3)) < 1e-8; " *
                "the angle coordinate is singular",
                [i, j, k], _residues(ns, [i, j, k])))
        end
    end

    # --- box consistency with protocol ------------------------------------
    if protocol.nonbonded === :pme
        if ns.box.kind === :none
            push!(diags, _diag(:box_conflict, :error,
                "protocol specifies PME nonbonded method but box is :none; periodic box is required"))
        elseif ns.box.kind === :orthorhombic
            cutoff = Float64(protocol.cutoff_nm)
            if any(l <= 2.0 * cutoff for l in ns.box.lengths_nm)
                push!(diags, _diag(:box_conflict, :error,
                    "orthorhombic box dimensions $(ns.box.lengths_nm) nm are <= 2*cutoff ($(2.0 * cutoff) nm); minimum-image convention violated"))
            end
        end
    end

    # --- 1-4 pairs beyond the nonbonded cutoff -----------------------------
    if protocol.nonbonded === :pme
        cutoff = Float64(protocol.cutoff_nm)
        for p in ns.pair14
            d = _pair_distance(q, p.i, p.j)
            if d >= cutoff
                push!(diags, _diag(:pair14_outside_cutoff, :error,
                    "1-4 pair ($(p.i),$(p.j)) measures $(_fmt(d, 4)) nm, at or beyond " *
                    "the $(_fmt(cutoff, 4)) nm cutoff; the StreamedLJ cutoff drops " *
                    "the 1-4 LJ term",
                    [p.i, p.j], _residues(ns, [p.i, p.j])))
            end
        end
    end

    # --- net charge and background ----------------------------------------
    qsum = sum(a.charge for a in ns.atoms)
    push!(diags, _diag(:net_charge, :info,
        "sum of atom charges = $(_fmt(qsum)) e (declared $(_fmt(ns.net_charge)) e); " *
        "charges are reported, never rescaled"))
    if protocol.background === :uniform
        push!(diags, _diag(:background_choice, :info,
            "protocol.background = :uniform: PME applies a neutralizing background; " *
            "imported charges are unchanged"))
    elseif protocol.background === :none
        push!(diags, _diag(:background_choice, :info,
            "protocol.background = :none: non-periodic direct evaluation uses no " *
            "neutralizing background; a nonzero net charge stays uncompensated"))
    else
        push!(diags, _diag(:background_choice, :info,
            "protocol.background = $(repr(protocol.background)); background " *
            "treatment is not one of the two documented choices"))
    end

    # --- preparation / input diagnostics surfaced from the extractor / preparation ---
    for d in ns.input_diagnostics
        code = d.code
        if code in (:box_mismatch, :inconsistent_box, :box_conflict)
            push!(diags, _diag(:box_conflict, d.severity, d.message,
                               copy(d.atoms), copy(d.residues)))
        elseif code in (:protonation_ambiguity, :uncertain_protonation)
            push!(diags, _diag(:uncertain_protonation, d.severity, d.message,
                               copy(d.atoms), copy(d.residues)))
            push!(diags, _diag(:protonation_ambiguity, d.severity, d.message,
                               copy(d.atoms), copy(d.residues)))
        elseif code in (:ligand_parameters, :absent_ligand_parameters)
            push!(diags, _diag(:absent_ligand_parameters, d.severity, d.message,
                               copy(d.atoms), copy(d.residues)))
            push!(diags, _diag(:ligand_parameters, d.severity, d.message,
                               copy(d.atoms), copy(d.residues)))
        elseif code in (:missing_atoms, :missing_atom)
            push!(diags, _diag(:missing_atoms, d.severity, d.message,
                               copy(d.atoms), copy(d.residues)))
        else
            push!(diags, _diag(code, d.severity, d.message,
                               copy(d.atoms), copy(d.residues)))
        end
    end

    # --- zero LJ parameters -------------------------------------------------
    # An atom participates in a nonbonded pair unless every other atom is
    # excluded from it. This is O(total exclusions), not O(N²).
    excluded = [Set{Int}() for _ in 1:N]
    for (i, j) in ns.exclusions
        push!(excluded[i], j)
        push!(excluded[j], i)
    end
    for (i, a) in enumerate(ns.atoms)
        a.sigma_nm == 0.0 && a.epsilon_kj == 0.0 || continue
        participates = N > 1 && length(excluded[i]) < N - 1
        participates || continue
        benign = uppercase(a.element) in _AUDIT_BENIGN_ZERO_LJ
        sev = benign ? :info : :warning
        push!(diags, _diag(:zero_lj_parameters, sev,
            "atom $i ($(a.name), $(a.residue_name) $(a.residue_index)) has " *
            "sigma_nm = 0 and epsilon_kj = 0 while participating in nonbonded " *
            "pairs; no parameters were fabricated" *
            (benign ? " (zero LJ is normal for this element)" : ""),
            [i], [a.residue_index]))
    end

    # --- polymer C–N chain breaks -----------------------------------------
    res_order = Int[]
    res_atoms = Dict{Int,Vector{Int}}()
    for (i, a) in enumerate(ns.atoms)
        if !haskey(res_atoms, a.residue_index)
            res_atoms[a.residue_index] = Int[]
            push!(res_order, a.residue_index)
        end
        push!(res_atoms[a.residue_index], i)
    end
    for k in 1:(length(res_order) - 1)
        r = res_order[k]
        r2 = res_order[k + 1]
        r2 == r + 1 || continue
        atoms_r = res_atoms[r]
        atoms_r2 = res_atoms[r2]
        (length(atoms_r) == 1 || length(atoms_r2) == 1) && continue
        name_r = uppercase(ns.atoms[first(atoms_r)].residue_name)
        name_r2 = uppercase(ns.atoms[first(atoms_r2)].residue_name)
        (name_r in _AUDIT_ION_RESIDUES || name_r2 in _AUDIT_ION_RESIDUES) && continue
        ic = findfirst(i -> ns.atoms[i].name == "C", atoms_r)
        inn = findfirst(i -> ns.atoms[i].name == "N", atoms_r2)
        if ic !== nothing && inn !== nothing
            ci = atoms_r[ic]
            ni = atoms_r2[inn]
            d = _pair_distance(q, ci, ni)
            if d > _AUDIT_CHAIN_BREAK_NM
                push!(diags, _diag(:chain_break, :warning,
                    "C$(r) (atom $ci) to N$(r2) (atom $ni) is $(_fmt(d, 4)) nm, " *
                    "above the $(_AUDIT_CHAIN_BREAK_NM) nm peptide-bond threshold",
                    [ci, ni], [r, r2]))
            end
        end
        # Nucleic acid phosphodiester chain breaks
        io3 = findfirst(i -> ns.atoms[i].name in ("O3'", "O3*"), atoms_r)
        inn_p = findfirst(i -> ns.atoms[i].name == "P", atoms_r2)
        if io3 !== nothing && inn_p !== nothing
            o3i = atoms_r[io3]
            pi = atoms_r2[inn_p]
            d = _pair_distance(q, o3i, pi)
            if d > _AUDIT_CHAIN_BREAK_NM
                push!(diags, _diag(:chain_break, :warning,
                    "O3'$(r) (atom $o3i) to P$(r2) (atom $pi) is $(_fmt(d, 4)) nm, " *
                    "above the $(_AUDIT_CHAIN_BREAK_NM) nm phosphodiester threshold",
                    [o3i, pi], [r, r2]))
            end
        end
    end

    # --- declared virtual sites -------------------------------------------
    for (k, v) in enumerate(ns.virtual_sites)
        push!(diags, _diag(:virtual_site_execution, :info,
            "virtual site $k at atom $(v.site) ($(v.kind)) is represented " *
            "structurally; execution is refused by ResolvedSystem",
            [v.site], [ns.atoms[v.site].residue_index]))
    end

    # --- CMAP coverage ----------------------------------------------------
    for (k, c) in enumerate(ns.cmaps)
        atoms = unique(collect(c.indices))
        push!(diags, _diag(:cmap_present, :info,
            "CMAP term $k: $(c.resolution)x$(c.resolution) map over atoms $(atoms) " *
            "is built as a ChemistryCMAP.CMAPCorrection",
            atoms, _residues(ns, atoms)))
    end

    return _finish_parameterized(ns, diags)
end

function _finish_parameterized(ns::NormalizedSystem,
                               diags::Vector{DiagnosticRecord})
    _audit_sort!(diags)
    errors = [d for d in diags if d.severity === :error]
    warnings = [d for d in diags if d.severity === :warning]
    choices = [d for d in diags if d.severity === :info]
    counts = (
        n_atoms = length(ns.atoms),
        n_errors = length(errors),
        n_warnings = length(warnings),
        n_choices = length(choices),
        n_bond_deviations = count(d -> d.code === :bond_deviation, diags),
        n_chain_breaks = count(d -> d.code === :chain_break, diags),
        n_pair14_outside_cutoff = count(d -> d.code === :pair14_outside_cutoff, diags),
        n_missing_atoms = count(d -> d.code === :missing_atoms, diags),
        n_box_conflicts = count(d -> d.code === :box_conflict, diags),
        n_uncertain_protonation = count(d -> d.code in (:uncertain_protonation, :protonation_ambiguity), diags),
        n_absent_ligand_parameters = count(d -> d.code in (:absent_ligand_parameters, :ligand_parameters), diags),
        n_altloc = count(d -> d.code === :altloc, diags),
    )
    prov = ns.provenance
    src_file = prov.source
    param_ver = !isempty(prov.generator_version) ? prov.generator_version : prov.force_field
    mapping_decisions = String[]
    !isempty(prov.notes) && push!(mapping_decisions, prov.notes)
    for d in ns.input_diagnostics
        if d.code === :mapping_decision || occursin("mapping", lowercase(d.message))
            push!(mapping_decisions, d.message)
        end
    end
    hashes = Dict{String,String}()
    !isempty(prov.source_sha256) && (hashes["source_sha256"] = prov.source_sha256)
    !isempty(prov.coordinate_sha256) && (hashes["coordinate_sha256"] = prov.coordinate_sha256)

    return AuditReport(diags, errors, warnings, choices, counts, prov,
                       src_file, param_ver, mapping_decisions, hashes)
end

# ---------------------------------------------------------------------------
# raw structure audit
# ---------------------------------------------------------------------------

"""
    audit_raw_structure(path; kwargs...) -> RawAuditReport

Delegate to `Prepare.diagnose_structure(path; kwargs...)` and translate its
findings into `DiagnosticRecord`s. Only findings that `Prepare` actually made
are emitted: this function never assigns charges, LJ parameters, residue
identities or protonation states.

`Prepare.StructureReport` carries a flat `issues` list, so severity is assigned
by whether the finding blocks force evaluation (non-finite coordinates, duplicate
identities, close contacts) or is an unresolved *choice* (altloc, zero
occupancy). Coordinates in `Prepare` are Angstroms, and messages say so.
"""
function audit_raw_structure(path::AbstractString; kwargs...)
    rep = diagnose_structure(path; kwargs...)
    atoms, _box, _issues = read_pdb_fallback(path)
    diags = DiagnosticRecord[]

    if rep.n_altloc > 0
        idx = Int[i for (i, a) in enumerate(atoms) if a.altloc != ' ']
        push!(diags, _diag(:altloc, :warning,
            "$(rep.n_altloc) atoms carry a non-blank altloc; an explicit altloc " *
            "policy is required before preparation",
            idx, unique([atoms[i].resseq for i in idx])))
    end

    if rep.n_nonfinite > 0
        idx = Int[i for (i, a) in enumerate(atoms)
                  if !(isfinite(a.x) && isfinite(a.y) && isfinite(a.z))]
        push!(diags, _diag(:nonfinite, :error,
            "$(rep.n_nonfinite) atoms have non-finite coordinates; no force can be evaluated",
            idx, unique([atoms[i].resseq for i in idx])))
    end

    if rep.n_duplicate_identities > 0
        seen = Set{Tuple{Char,Int,Char,Symbol,Char}}()
        idx = Int[]
        for (i, a) in enumerate(atoms)
            key = (a.chain, a.resseq, a.icode, a.name, a.altloc)
            if key in seen
                push!(idx, i)
            else
                push!(seen, key)
            end
        end
        push!(diags, _diag(:duplicate_atom, :error,
            "$(rep.n_duplicate_identities) duplicate atom identities " *
            "(chain, resseq, icode, name, altloc); there is no correct silent choice",
            idx, unique([atoms[i].resseq for i in idx])))
    end

    if rep.n_zero_occupancy > 0
        idx = Int[i for (i, a) in enumerate(atoms) if a.occupancy == 0.0]
        push!(diags, _diag(:zero_occupancy, :warning,
            "$(rep.n_zero_occupancy) atoms have zero occupancy; presence is not " *
            "determined by the file",
            idx, unique([atoms[i].resseq for i in idx])))
    end

    for (i, j, d) in rep.close_contact_pairs
        ri = atoms[i].resseq
        rj = atoms[j].resseq
        push!(diags, _diag(:coincident_atoms, :error,
            "atoms $i and $j are $(_fmt(d, 4)) A apart, below the close-contact " *
            "tolerance; every supported potential is singular there",
            [i, j], ri == rj ? [ri] : [ri, rj]))
    end

    for issue in rep.issues
        lower_issue = lowercase(issue)
        if startswith(issue, "truncated") || startswith(issue, "unparseable")
            push!(diags, _diag(:structure_parse, :error, issue))
        elseif startswith(issue, "multi-model")
            push!(diags, _diag(:multi_model, :warning, issue))
        elseif startswith(issue, "requested ")
            push!(diags, _diag(:parser_choice, :warning, issue))
        elseif occursin("missing atom", lower_issue) || occursin("missing heavy atom", lower_issue)
            push!(diags, _diag(:missing_atoms, :error, issue))
        elseif occursin("protonation", lower_issue)
            push!(diags, _diag(:uncertain_protonation, :warning, issue))
            push!(diags, _diag(:protonation_ambiguity, :warning, issue))
        elseif occursin("ligand parameter", lower_issue) || occursin("absent ligand", lower_issue)
            push!(diags, _diag(:absent_ligand_parameters, :warning, issue))
            push!(diags, _diag(:ligand_parameters, :warning, issue))
        elseif occursin("chain break", lower_issue)
            push!(diags, _diag(:chain_break, :warning, issue))
        elseif occursin("box mismatch", lower_issue) || occursin("inconsistent box", lower_issue)
            push!(diags, _diag(:box_conflict, :warning, issue))
        end
    end

    # Group atoms by (chain, resseq)
    res_order = Tuple{Char,Int}[]
    res_atoms = Dict{Tuple{Char,Int},Vector{Int}}()
    for (i, a) in enumerate(atoms)
        k = (a.chain, a.resseq)
        if !haskey(res_atoms, k)
            res_atoms[k] = Int[]
            push!(res_order, k)
        end
        push!(res_atoms[k], i)
    end

    # Polymer chain breaks in raw structure (C to next N > 2.5 A)
    for k in 1:(length(res_order) - 1)
        ch1, r1 = res_order[k]
        ch2, r2 = res_order[k + 1]
        (ch1 == ch2 && r2 == r1 + 1) || continue
        atoms_r1 = res_atoms[(ch1, r1)]
        atoms_r2 = res_atoms[(ch2, r2)]
        ic = findfirst(i -> atoms[i].name === :C, atoms_r1)
        inn = findfirst(i -> atoms[i].name === :N, atoms_r2)
        (ic === nothing || inn === nothing) && continue
        ci = atoms_r1[ic]
        ni = atoms_r2[inn]
        dx = atoms[ci].x - atoms[ni].x
        dy = atoms[ci].y - atoms[ni].y
        dz = atoms[ci].z - atoms[ni].z
        d = sqrt(dx * dx + dy * dy + dz * dz)
        if d > 2.5
            push!(diags, _diag(:chain_break, :warning,
                "chain break between residue $r1 and $r2 (C-N distance $(_fmt(d, 4)) A > 2.5 A)",
                [ci, ni], [r1, r2]))
        end
    end

    # Missing backbone atoms in standard residues
    seen_missing = Set{Tuple{Char,Int}}()
    for (ch, r) in res_order
        atoms_r = res_atoms[(ch, r)]
        rname = atoms[first(atoms_r)].resname
        if rname in (:ALA, :ARG, :ASN, :ASP, :CYS, :GLN, :GLU, :GLY, :HIS, :HID, :HIE, :HIP,
                     :ILE, :LEU, :LYS, :MET, :PHE, :PRO, :SER, :THR, :TRP, :TYR, :VAL)
            names = Set(atoms[i].name for i in atoms_r)
            missing_bb = [b for b in (:N, :CA, :C) if !(b in names)]
            if !isempty(missing_bb) && !((ch, r) in seen_missing)
                push!(seen_missing, (ch, r))
                push!(diags, _diag(:missing_atoms, :error,
                    "residue $(rname) $r (chain $ch) is missing backbone atom(s): $(join(missing_bb, ", ")); no atoms were silently fabricated",
                    atoms_r, [r]))
            end
        end
    end

    # Uncertain protonation (HIS residue without explicit HID/HIE/HIP)
    seen_his = Set{Tuple{Char,Int}}()
    for (i, a) in enumerate(atoms)
        if a.resname === :HIS && !((a.chain, a.resseq) in seen_his)
            push!(seen_his, (a.chain, a.resseq))
            idx = [j for (j, at) in enumerate(atoms) if at.chain == a.chain && at.resseq == a.resseq]
            push!(diags, _diag(:uncertain_protonation, :warning,
                "residue HIS $(a.resseq) (chain $(a.chain)) has ambiguous protonation state (HID/HIE/HIP unspecified)",
                idx, [a.resseq]))
            push!(diags, _diag(:protonation_ambiguity, :warning,
                "residue HIS $(a.resseq) (chain $(a.chain)) has ambiguous protonation state (HID/HIE/HIP unspecified)",
                idx, [a.resseq]))
        end
    end

    # Absent ligand parameters (non-standard residues in raw structure)
    seen_lig = Set{Tuple{Char,Int}}()
    for (i, a) in enumerate(atoms)
        if !(a.resname in _KNOWN_STANDARD_RESIDUES_SYM) && !((a.chain, a.resseq) in seen_lig)
            push!(seen_lig, (a.chain, a.resseq))
            idx = [j for (j, at) in enumerate(atoms) if at.chain == a.chain && at.resseq == a.resseq]
            push!(diags, _diag(:absent_ligand_parameters, :warning,
                "ligand residue $(a.resname) $(a.resseq) (chain $(a.chain)) has absent force-field parameters; parameters must not be silently invented",
                idx, [a.resseq]))
            push!(diags, _diag(:ligand_parameters, :warning,
                "ligand residue $(a.resname) $(a.resseq) (chain $(a.chain)) has absent force-field parameters; parameters must not be silently invented",
                idx, [a.resseq]))
        end
    end

    _audit_sort!(diags)
    errors = [d for d in diags if d.severity === :error]
    warnings = [d for d in diags if d.severity === :warning]
    counts = (
        n_atoms = rep.n_atoms,
        n_errors = length(errors),
        n_warnings = length(warnings),
        n_choices = length(diags) - length(errors) - length(warnings),
        n_altloc = rep.n_altloc,
        n_close_contacts = rep.n_close_contacts,
        n_zero_occupancy = rep.n_zero_occupancy,
        n_duplicate_identities = rep.n_duplicate_identities,
        n_nonfinite = rep.n_nonfinite,
    )
    src_file = String(path)
    param_ver = ""
    mapping_decisions = String[]
    for issue in rep.issues
        if occursin("mapping", lowercase(issue)) || startswith(issue, "requested ")
            push!(mapping_decisions, issue)
        end
    end
    hashes = Dict{String,String}()
    h = _file_sha256(path)
    !isempty(h) && (hashes["source_sha256"] = h)

    return RawAuditReport(diags, errors, warnings, counts,
                          src_file, param_ver, mapping_decisions, hashes)
end
