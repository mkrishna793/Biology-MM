# Diagnostics audit tests.
#
# Each fixture isolates one provable finding so the test asserts *which* record
# the audit must produce, with which atoms, and with which severity. The counts
# are checked against the partitioned vectors so a record can never be silently
# dropped or duplicated in the summary.

using FoundationCore
using FoundationCore.Chemistry
using Printf

const C = FoundationCore.Chemistry
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

const PROV = C.ImportProvenance("test fixture", "AMBER test topology", "", "", "", "", "", "")

"One atom with explicit zero/nonzero LJ parameters."
function atom(name, charge, element, res, resname;
              sigma_nm = 0.3, epsilon_kj = 0.5, mass = 12.0)
    return C.NormalizedAtom(name, mass, charge, element, res, resname,
                            sigma_nm, epsilon_kj)
end

"""
Six atoms across two residues with one bond (1-2), one angle (1-2-3) and one
1-4 pair (1-4). Residue 1's C and residue 2's N are 0.133 nm apart, so the
ideal geometry has no chain break. Atom 6 is a hydrogen with zero LJ
parameters, which is normal and therefore informational.
"""
function base_atoms()
    return [
        atom("N", 0.5, "N", 1, "ALA"; sigma_nm = 0.325, epsilon_kj = 0.71, mass = 14.007),
        atom("CA", 0.0, "C", 1, "ALA"; sigma_nm = 0.34, epsilon_kj = 0.46),
        atom("C", 0.0, "C", 1, "ALA"; sigma_nm = 0.34, epsilon_kj = 0.36),
        atom("N", -0.2, "N", 2, "ALA"; sigma_nm = 0.325, epsilon_kj = 0.71, mass = 14.007),
        atom("CA", 0.0, "C", 2, "ALA"; sigma_nm = 0.34, epsilon_kj = 0.46),
        atom("H", 0.0, "H", 2, "ALA"; sigma_nm = 0.0, epsilon_kj = 0.0, mass = 1.008),
    ]
end

function base_ns(; atoms = base_atoms(), bonds = [C.BondTerm(1, 2, 418400.0, 0.145)],
                 angles = [C.AngleTerm(1, 2, 3, 418.4, pi / 2)],
                 pair14 = [C.Pair14Term(1, 4, 0.8333333333333334, 0.5)],
                 exclusions = [(2, 3)],
                 cmaps = C.CmapTerm[],
                 virtual_sites = C.VirtualSiteTerm[],
                 input_diagnostics = C.DiagnosticRecord[])
    return C.NormalizedSystem(
        v"1.0.0", :amber, C.CHEM_UNITS_NAME, PROV, atoms,
        C.NormalizedBox(:orthorhombic, (4.0, 4.0, 4.0), zeros(3, 3)),
        bonds, angles, C.UreyBradleyTerm[], C.DihedralTerm[], C.ImproperTerm[],
        cmaps, exclusions, pair14, C.C64Term[], virtual_sites, C.ConstraintTerm[],
        input_diagnostics, sum(a.charge for a in atoms))
end

function base_q()
    q = zeros(3, 6)
    q[:, 1] = [0.5, 0.5, 0.5]
    q[:, 2] = [0.645, 0.5, 0.5]      # 0.145 nm from atom 1
    q[:, 3] = [0.645, 0.652, 0.5]    # right angle at atom 2
    q[:, 4] = [0.778, 0.652, 0.5]    # 0.133 nm from atom 3
    q[:, 5] = [0.923, 0.652, 0.5]
    q[:, 6] = [0.923, 0.752, 0.5]
    return q
end

pme_protocol(; cutoff_nm = 1.0) =
    C.SimulationProtocol(nonbonded = :pme, cutoff_nm = cutoff_nm,
                         water_model = :flexible, background = :uniform)

direct_protocol() =
    C.SimulationProtocol(nonbonded = :direct, cutoff_nm = 1.0,
                         water_model = :flexible, background = :none)

# ---------------------------------------------------------------------------
# assertions shared by every report
# ---------------------------------------------------------------------------

function check_partition(rep)
    @assert length(rep.diagnostics) ==
            length(rep.provable_errors) + length(rep.warnings) + length(rep.choices)
    @assert rep.counts.n_errors == length(rep.provable_errors)
    @assert rep.counts.n_warnings == length(rep.warnings)
    @assert rep.counts.n_choices == length(rep.choices)
    @assert all(d -> d.severity === :error, rep.provable_errors)
    @assert all(d -> d.severity === :warning, rep.warnings)
    @assert all(d -> d.severity === :info, rep.choices)
    @assert rep.counts.n_bond_deviations ==
            count(d -> d.code === :bond_deviation, rep.diagnostics)
    @assert rep.counts.n_chain_breaks ==
            count(d -> d.code === :chain_break, rep.diagnostics)
    @assert rep.counts.n_pair14_outside_cutoff ==
            count(d -> d.code === :pair14_outside_cutoff, rep.diagnostics)
    keys = [(String(d.code), isempty(d.atoms) ? 0 : first(d.atoms), d.message)
            for d in rep.diagnostics]
    @assert issorted(keys) "diagnostics must be sorted by (code, first atom, message)"
    return true
end

function has_code(rep, code; severity = nothing, atoms = nothing)
    for d in rep.diagnostics
        d.code === code || continue
        severity !== nothing && d.severity !== severity && continue
        atoms !== nothing && d.atoms != atoms && continue
        return true
    end
    return false
end

# ---------------------------------------------------------------------------
# parameterized audit
# ---------------------------------------------------------------------------

function test_ideal_geometry_has_no_errors()
    rep = C.audit_parameterized(base_ns(), base_q(); protocol = pme_protocol())
    @printf("  ideal: errors=%d warnings=%d choices=%d\n",
            rep.counts.n_errors, rep.counts.n_warnings, rep.counts.n_choices)
    @assert rep.counts.n_atoms == 6
    @assert isempty(rep.provable_errors)
    @assert isempty(rep.warnings)
    @assert has_code(rep, :net_charge; severity = :info)
    @assert has_code(rep, :background_choice; severity = :info)
    @assert has_code(rep, :zero_lj_parameters; severity = :info, atoms = [6])
    @assert !has_code(rep, :bond_deviation)
    @assert !has_code(rep, :pair14_outside_cutoff)
    check_partition(rep)
    return true
end

function test_bond_deviation_is_a_warning()
    q = base_q()
    q[1, 2] += 0.12                    # 0.145 -> 0.265 nm
    rep = C.audit_parameterized(base_ns(), q; protocol = pme_protocol())
    @assert has_code(rep, :bond_deviation; severity = :warning, atoms = [1, 2])
    @assert rep.counts.n_bond_deviations == 1
    @assert isempty(rep.provable_errors)
    check_partition(rep)
    return true
end

function test_coincident_bonded_atoms_are_an_error()
    q = base_q()
    q[:, 2] = q[:, 1]
    rep = C.audit_parameterized(base_ns(), q; protocol = pme_protocol())
    @assert has_code(rep, :coincident_atoms; severity = :error, atoms = [1, 2])
    @assert !has_code(rep, :bond_deviation, atoms = [1, 2])
    check_partition(rep)
    return true
end

function test_pair14_outside_cutoff()
    q = base_q()
    q[:, 4] = [1.8, 0.5, 0.5]          # 1.30 nm from atom 1
    rep = C.audit_parameterized(base_ns(), q; protocol = pme_protocol(cutoff_nm = 1.0))
    @assert has_code(rep, :pair14_outside_cutoff; severity = :error, atoms = [1, 4])
    @assert rep.counts.n_pair14_outside_cutoff == 1
    d = only(filter(x -> x.code === :pair14_outside_cutoff, rep.diagnostics))
    @assert occursin("1.3", d.message)
    check_partition(rep)
    return true
end

function test_direct_protocol_has_no_cutoff_check()
    q = base_q()
    q[:, 4] = [1.8, 0.5, 0.5]
    rep = C.audit_parameterized(base_ns(), q; protocol = direct_protocol())
    @assert !has_code(rep, :pair14_outside_cutoff)
    bg = only(filter(d -> d.code === :background_choice, rep.diagnostics))
    @assert bg.severity === :info
    @assert occursin("no neutralizing background", bg.message)
    nc = only(filter(d -> d.code === :net_charge, rep.diagnostics))
    @assert occursin("0.3", nc.message)
    check_partition(rep)
    return true
end

function test_box_mismatch_is_preserved()
    diag = C.DiagnosticRecord(:box_mismatch, :warning,
                              "CRYST1 lengths differ from the coordinate header",
                              [2, 3], [1])
    rep = C.audit_parameterized(base_ns(; input_diagnostics = [diag]), base_q();
                                protocol = pme_protocol())
    rec = only(filter(d -> d.code === :box_conflict, rep.diagnostics))
    @assert rec.severity === :warning
    @assert rec.message == diag.message
    @assert rec.atoms == [2, 3]
    @assert rec.residues == [1]
    @assert any(d -> d.code === :box_conflict, rep.warnings)
    check_partition(rep)
    return true
end

function test_zero_lj_heavy_atom_warns()
    atoms = base_atoms()
    atoms[5] = atom("CA", 0.0, "C", 2, "ALA"; sigma_nm = 0.0, epsilon_kj = 0.0)
    rep = C.audit_parameterized(base_ns(; atoms = atoms), base_q();
                                protocol = pme_protocol())
    @assert has_code(rep, :zero_lj_parameters; severity = :warning, atoms = [5])
    @assert has_code(rep, :zero_lj_parameters; severity = :info, atoms = [6])
    check_partition(rep)
    return true
end

function test_chain_break_fixture()
    atoms = [
        atom("N", 0.0, "N", 1, "ALA"; sigma_nm = 0.325, epsilon_kj = 0.71, mass = 14.007),
        atom("CA", 0.0, "C", 1, "ALA"; sigma_nm = 0.34, epsilon_kj = 0.46),
        atom("C", 0.0, "C", 1, "ALA"; sigma_nm = 0.34, epsilon_kj = 0.36),
        atom("N", 0.0, "N", 2, "ALA"; sigma_nm = 0.325, epsilon_kj = 0.71, mass = 14.007),
        atom("CA", 0.0, "C", 2, "ALA"; sigma_nm = 0.34, epsilon_kj = 0.46),
        # water residues: C/N names present but must never count as polymer
        atom("C", 0.0, "C", 3, "WAT"; sigma_nm = 0.34, epsilon_kj = 0.36),
        atom("N", 0.0, "N", 4, "WAT"; sigma_nm = 0.325, epsilon_kj = 0.71, mass = 14.007),
        # single-atom residues are skipped
        atom("C", 0.0, "C", 5, "ALA"; sigma_nm = 0.34, epsilon_kj = 0.36),
        atom("N", 0.0, "N", 6, "ALA"; sigma_nm = 0.325, epsilon_kj = 0.71, mass = 14.007),
    ]
    q = zeros(3, length(atoms))
    q[:, 1] = [0.0, 0.0, 0.0]
    q[:, 2] = [0.145, 0.0, 0.0]
    q[:, 3] = [0.145, 0.152, 0.0]
    q[:, 4] = [0.745, 0.152, 0.0]       # 0.600 nm from residue 1's C
    q[:, 5] = [0.89, 0.152, 0.0]
    q[:, 6] = [3.0, 0.0, 0.0]
    q[:, 7] = [4.0, 0.0, 0.0]
    q[:, 8] = [6.0, 0.0, 0.0]
    q[:, 9] = [7.0, 0.0, 0.0]
    ns = base_ns(; atoms = atoms, bonds = C.BondTerm[], angles = C.AngleTerm[],
                 pair14 = C.Pair14Term[], exclusions = Tuple{Int,Int}[])
    rep = C.audit_parameterized(ns, q; protocol = pme_protocol())
    @assert rep.counts.n_chain_breaks == 1
    br = only(filter(d -> d.code === :chain_break, rep.diagnostics))
    @assert br.severity === :warning
    @assert br.atoms == [3, 4]
    @assert br.residues == [1, 2]
    @assert occursin("0.6", br.message)
    check_partition(rep)
    return true
end

function test_cmap_and_virtual_site()
    ns = base_ns(; cmaps = [C.CmapTerm((1, 2, 3, 4, 5, 6, 2, 3), 2, zeros(4))],
                 virtual_sites = [C.VirtualSiteTerm(:linear, 6, [1, 2], [0.5, 0.5])])
    rep = C.audit_parameterized(ns, base_q(); protocol = pme_protocol())
    @assert has_code(rep, :cmap_present; severity = :info)
    @assert has_code(rep, :virtual_site_execution; severity = :info, atoms = [6])
    @assert !any(d -> d.code === :virtual_site_execution && d.severity === :error,
                 rep.diagnostics)
    check_partition(rep)
    return true
end

function test_shape_and_nonfinite()
    ns = base_ns()
    rep = C.audit_parameterized(ns, zeros(3, 5); protocol = pme_protocol())
    @assert rep.counts.n_errors == 1
    @assert has_code(rep, :shape; severity = :error)
    @assert rep.counts.n_atoms == 6
    check_partition(rep)

    q = base_q()
    q[2, 3] = NaN
    rep = C.audit_parameterized(ns, q; protocol = pme_protocol())
    @assert has_code(rep, :nonfinite; severity = :error, atoms = [3])
    @assert rep.counts.n_errors == 1
    check_partition(rep)
    return true
end

# ---------------------------------------------------------------------------
# raw structure audit
# ---------------------------------------------------------------------------

function test_raw_altlocs()
    path = joinpath(REPO_ROOT, "wrappers", "python", "tests", "systems", "altlocs.pdb")
    if !isfile(path)
        println("  altlocs.pdb not present; skipping")
        return true
    end
    rep = C.audit_raw_structure(path)
    @printf("  altlocs.pdb: atoms=%d errors=%d warnings=%d altloc=%d\n",
            rep.counts.n_atoms, rep.counts.n_errors, rep.counts.n_warnings,
            rep.counts.n_altloc)
    @assert rep.counts.n_atoms == 57
    @assert rep.counts.n_altloc == 57
    @assert has_code(rep, :altloc; severity = :warning)
    # The raw audit must not invent parameters or structural findings that
    # Prepare did not make.
    for code in (:ligand_parameters, :missing_atoms, :unknown_residue,
                 :protonation_ambiguity)
        @assert !has_code(rep, code) "raw audit fabricated $code"
    end
    @assert rep.counts.n_errors == length(rep.provable_errors)
    @assert rep.counts.n_warnings == length(rep.warnings)
    @assert length(rep.diagnostics) ==
            length(rep.provable_errors) + length(rep.warnings) + rep.counts.n_choices
    return true
end

function test_raw_clean_structure()
    path = joinpath(REPO_ROOT, "wrappers", "python", "tests", "systems",
                    "alanine-dipeptide-implicit.pdb")
    if !isfile(path)
        println("  alanine-dipeptide-implicit.pdb not present; skipping")
        return true
    end
    rep = C.audit_raw_structure(path)
    @printf("  clean.pdb: atoms=%d errors=%d warnings=%d\n",
            rep.counts.n_atoms, rep.counts.n_errors, rep.counts.n_warnings)
    @assert rep.counts.n_atoms == 22
    @assert isempty(rep.provable_errors)
    @assert !has_code(rep, :altloc)
    @assert !has_code(rep, :ligand_parameters)
    @assert !has_code(rep, :missing_atoms)
    return true
end

# ---------------------------------------------------------------------------

function test_missing_atoms_reported()
    # 1. Parameterized system with missing atoms surfaced from preparation
    diag = C.DiagnosticRecord(:missing_atoms, :error,
                              "residue 2 missing sidechain atoms CG, CD", [4, 5], [2])
    rep = C.audit_parameterized(base_ns(; input_diagnostics = [diag]), base_q();
                                protocol = pme_protocol())
    @assert has_code(rep, :missing_atoms; severity = :error, atoms = [4, 5])
    @assert rep.counts.n_missing_atoms == 1
    check_partition(rep)

    # 2. Raw structure with missing backbone atom C in ALA residue
    tmppath = tempname() * ".pdb"
    open(tmppath, "w") do io
        println(io, "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N")
        println(io, "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C")
        println(io, "END")
    end
    try
        raw_rep = C.audit_raw_structure(tmppath)
        @assert has_code(raw_rep, :missing_atoms; severity = :error)
        d = only(filter(x -> x.code === :missing_atoms, raw_rep.diagnostics))
        @assert occursin("missing backbone atom", d.message)
        @assert d.atoms == [1, 2]
        @assert d.residues == [1]
    finally
        rm(tmppath; force = true)
    end
    return true
end

function test_chain_breaks_reported()
    # 1. Nucleic acid phosphodiester chain break
    atoms = [
        atom("P", 0.0, "P", 1, "DA"; sigma_nm = 0.35, epsilon_kj = 0.5),
        atom("O3'", 0.0, "O", 1, "DA"; sigma_nm = 0.30, epsilon_kj = 0.5),
        atom("P", 0.0, "P", 2, "DA"; sigma_nm = 0.35, epsilon_kj = 0.5),
        atom("O3'", 0.0, "O", 2, "DA"; sigma_nm = 0.30, epsilon_kj = 0.5),
    ]
    q = zeros(3, 4)
    q[:, 1] = [0.0, 0.0, 0.0]
    q[:, 2] = [0.15, 0.0, 0.0]
    q[:, 3] = [0.75, 0.0, 0.0]  # 0.60 nm from atom 2 (O3') to atom 3 (P) > 0.25 nm threshold
    q[:, 4] = [0.90, 0.0, 0.0]
    ns = base_ns(; atoms = atoms, bonds = C.BondTerm[], angles = C.AngleTerm[],
                 pair14 = C.Pair14Term[], exclusions = Tuple{Int,Int}[])
    rep = C.audit_parameterized(ns, q; protocol = pme_protocol())
    @assert has_code(rep, :chain_break; severity = :warning, atoms = [2, 3])
    @assert rep.counts.n_chain_breaks == 1
    check_partition(rep)

    # 2. Raw structure chain break (distance between C1 and N2 > 2.5 A)
    tmppath = tempname() * ".pdb"
    open(tmppath, "w") do io
        println(io, "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N")
        println(io, "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C")
        println(io, "ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C")
        println(io, "ATOM      4  N   ALA A   2       8.000   0.000   0.000  1.00  0.00           N")
        println(io, "ATOM      5  CA  ALA A   2       9.458   0.000   0.000  1.00  0.00           C")
        println(io, "ATOM      6  C   ALA A   2      10.000   0.000   0.000  1.00  0.00           C")
        println(io, "END")
    end
    try
        raw_rep = C.audit_raw_structure(tmppath)
        @assert has_code(raw_rep, :chain_break; severity = :warning)
        br = only(filter(x -> x.code === :chain_break, raw_rep.diagnostics))
        @assert occursin("chain break", br.message)
        @assert br.atoms == [3, 4]
        @assert br.residues == [1, 2]
    finally
        rm(tmppath; force = true)
    end
    return true
end

function test_uncertain_protonation_reported()
    # 1. Parameterized system with uncertain protonation diagnostic
    diag = C.DiagnosticRecord(:uncertain_protonation, :warning,
                              "residue 3 HIS protonation state is uncertain (pKa near pH 7)",
                              [3], [1])
    rep = C.audit_parameterized(base_ns(; input_diagnostics = [diag]), base_q();
                                protocol = pme_protocol())
    @assert has_code(rep, :uncertain_protonation; severity = :warning)
    @assert has_code(rep, :protonation_ambiguity; severity = :warning)
    @assert rep.counts.n_uncertain_protonation >= 1
    check_partition(rep)

    # 2. Raw structure with ambiguous HIS residue
    tmppath = tempname() * ".pdb"
    open(tmppath, "w") do io
        println(io, "ATOM      1  N   HIS A   1       0.000   0.000   0.000  1.00  0.00           N")
        println(io, "ATOM      2  CA  HIS A   1       1.458   0.000   0.000  1.00  0.00           C")
        println(io, "ATOM      3  C   HIS A   1       2.000   0.000   0.000  1.00  0.00           C")
        println(io, "END")
    end
    try
        raw_rep = C.audit_raw_structure(tmppath)
        @assert has_code(raw_rep, :uncertain_protonation; severity = :warning)
        @assert has_code(raw_rep, :protonation_ambiguity; severity = :warning)
        d = only(filter(x -> x.code === :uncertain_protonation, raw_rep.diagnostics))
        @assert occursin("ambiguous protonation state", d.message)
        @assert d.residues == [1]
    finally
        rm(tmppath; force = true)
    end
    return true
end

function test_absent_ligand_parameters_reported()
    # 1. Parameterized system with absent ligand parameters
    diag = C.DiagnosticRecord(:absent_ligand_parameters, :warning,
                              "ligand residue UNL 3 lacks torsion parameters", [5], [3])
    rep = C.audit_parameterized(base_ns(; input_diagnostics = [diag]), base_q();
                                protocol = pme_protocol())
    @assert has_code(rep, :absent_ligand_parameters; severity = :warning)
    @assert has_code(rep, :ligand_parameters; severity = :warning)
    @assert rep.counts.n_absent_ligand_parameters >= 1
    check_partition(rep)

    # 2. Raw structure with ligand residue UNL
    tmppath = tempname() * ".pdb"
    open(tmppath, "w") do io
        println(io, "HETATM    1  C1  UNL A   1       0.000   0.000   0.000  1.00  0.00           C")
        println(io, "HETATM    2  C2  UNL A   1       1.400   0.000   0.000  1.00  0.00           C")
        println(io, "END")
    end
    try
        raw_rep = C.audit_raw_structure(tmppath)
        @assert has_code(raw_rep, :absent_ligand_parameters; severity = :warning)
        @assert has_code(raw_rep, :ligand_parameters; severity = :warning)
        d = only(filter(x -> x.code === :absent_ligand_parameters, raw_rep.diagnostics))
        @assert occursin("absent force-field parameters", d.message)
        @assert d.residues == [1]
    finally
        rm(tmppath; force = true)
    end
    return true
end

function test_inconsistent_periodic_boxes_reported()
    # 1. PME protocol requires periodic box, but box is :none
    ns_nobox = C.NormalizedSystem(
        v"1.0.0", :amber, C.CHEM_UNITS_NAME, PROV, base_atoms(),
        C.NormalizedBox(:none, (0.0, 0.0, 0.0), zeros(3, 3)),
        [C.BondTerm(1, 2, 418400.0, 0.145)], [C.AngleTerm(1, 2, 3, 418.4, pi / 2)],
        C.UreyBradleyTerm[], C.DihedralTerm[], C.ImproperTerm[],
        C.CmapTerm[], [(2, 3)], [C.Pair14Term(1, 4, 0.8333333333333334, 0.5)],
        C.C64Term[], C.VirtualSiteTerm[], C.ConstraintTerm[],
        C.DiagnosticRecord[], sum(a.charge for a in base_atoms()))
    rep = C.audit_parameterized(ns_nobox, base_q(); protocol = pme_protocol())
    @assert has_code(rep, :box_conflict; severity = :error)
    @assert rep.counts.n_box_conflicts >= 1
    d = only(filter(x -> x.code === :box_conflict, rep.diagnostics))
    @assert occursin("box is :none", d.message)
    check_partition(rep)

    # 2. PME protocol with box lengths <= 2*cutoff
    ns_smallbox = C.NormalizedSystem(
        v"1.0.0", :amber, C.CHEM_UNITS_NAME, PROV, base_atoms(),
        C.NormalizedBox(:orthorhombic, (1.8, 1.8, 1.8), zeros(3, 3)),
        [C.BondTerm(1, 2, 418400.0, 0.145)], [C.AngleTerm(1, 2, 3, 418.4, pi / 2)],
        C.UreyBradleyTerm[], C.DihedralTerm[], C.ImproperTerm[],
        C.CmapTerm[], [(2, 3)], [C.Pair14Term(1, 4, 0.8333333333333334, 0.5)],
        C.C64Term[], C.VirtualSiteTerm[], C.ConstraintTerm[],
        C.DiagnosticRecord[], sum(a.charge for a in base_atoms()))
    rep_small = C.audit_parameterized(ns_smallbox, base_q(); protocol = pme_protocol(cutoff_nm = 1.0))
    @assert has_code(rep_small, :box_conflict; severity = :error)
    d2 = only(filter(x -> x.code === :box_conflict, rep_small.diagnostics))
    @assert occursin("minimum-image convention violated", d2.message)
    check_partition(rep_small)
    return true
end

function test_provenance_and_metadata_preservation()
    # 1. Parameterized report preserves source file, parameter version, mapping decisions, hashes
    prov = C.ImportProvenance(
        "fixtures/solvated_protein.parm7",
        "AMBER ff14SB",
        "a"^64,
        "fixtures/solvated_protein.rst7",
        "b"^64,
        "parmed",
        "4.3.1",
        "assigned HID to HIS 30; mapped ring atoms"
    )
    mapping_diag = C.DiagnosticRecord(:mapping_decision, :info,
                                      "applied ring-order mapping C1<-C6", Int[], Int[])
    ns = C.NormalizedSystem(
        v"1.0.0", :amber, C.CHEM_UNITS_NAME, prov, base_atoms(),
        C.NormalizedBox(:orthorhombic, (4.0, 4.0, 4.0), zeros(3, 3)),
        [C.BondTerm(1, 2, 418400.0, 0.145)], [C.AngleTerm(1, 2, 3, 418.4, pi / 2)],
        C.UreyBradleyTerm[], C.DihedralTerm[], C.ImproperTerm[],
        C.CmapTerm[], [(2, 3)], [C.Pair14Term(1, 4, 0.8333333333333334, 0.5)],
        C.C64Term[], C.VirtualSiteTerm[], C.ConstraintTerm[],
        [mapping_diag], sum(a.charge for a in base_atoms()))
    rep = C.audit_parameterized(ns, base_q(); protocol = pme_protocol())

    @assert rep.source_file == "fixtures/solvated_protein.parm7"
    @assert rep.parameter_version == "4.3.1"
    @assert rep.hashes["source_sha256"] == "a"^64
    @assert rep.hashes["coordinate_sha256"] == "b"^64
    @assert any(m -> occursin("assigned HID", m), rep.mapping_decisions)
    @assert any(m -> occursin("ring-order mapping", m), rep.mapping_decisions)
    @assert rep.provenance === prov
    check_partition(rep)

    # 2. Raw structure report preserves source file and computed hash
    path = joinpath(REPO_ROOT, "wrappers", "python", "tests", "systems",
                    "alanine-dipeptide-implicit.pdb")
    if isfile(path)
        raw_rep = C.audit_raw_structure(path)
        @assert raw_rep.source_file == path
        @assert haskey(raw_rep.hashes, "source_sha256")
        @assert length(raw_rep.hashes["source_sha256"]) == 64
        @assert all(c -> c in "0123456789abcdef", raw_rep.hashes["source_sha256"])
    end
    return true
end

function test_never_silently_invent()
    # Clean parameterized system: none of these must be fabricated
    rep = C.audit_parameterized(base_ns(), base_q(); protocol = pme_protocol())
    for code in (:missing_atoms, :chain_break, :box_conflict,
                 :uncertain_protonation, :protonation_ambiguity,
                 :absent_ligand_parameters, :ligand_parameters, :altloc)
        @assert !has_code(rep, code) "clean parameterized system fabricated $code"
    end

    # Clean raw structure: none of these must be fabricated
    path = joinpath(REPO_ROOT, "wrappers", "python", "tests", "systems",
                    "alanine-dipeptide-implicit.pdb")
    if isfile(path)
        raw_rep = C.audit_raw_structure(path)
        for code in (:missing_atoms, :chain_break, :box_conflict,
                     :uncertain_protonation, :protonation_ambiguity,
                     :absent_ligand_parameters, :ligand_parameters, :altloc)
            @assert !has_code(raw_rep, code) "clean raw structure fabricated $code"
        end
    end
    return true
end

function _run_all()
    tests = [
        ("ideal_geometry", test_ideal_geometry_has_no_errors),
        ("bond_deviation", test_bond_deviation_is_a_warning),
        ("coincident_atoms", test_coincident_bonded_atoms_are_an_error),
        ("pair14_cutoff", test_pair14_outside_cutoff),
        ("direct_no_cutoff", test_direct_protocol_has_no_cutoff_check),
        ("box_mismatch", test_box_mismatch_is_preserved),
        ("zero_lj", test_zero_lj_heavy_atom_warns),
        ("chain_break", test_chain_break_fixture),
        ("cmap_virtual_site", test_cmap_and_virtual_site),
        ("shape_nonfinite", test_shape_and_nonfinite),
        ("raw_altlocs", test_raw_altlocs),
        ("raw_clean", test_raw_clean_structure),
        ("missing_atoms", test_missing_atoms_reported),
        ("chain_breaks", test_chain_breaks_reported),
        ("uncertain_protonation", test_uncertain_protonation_reported),
        ("absent_ligand_parameters", test_absent_ligand_parameters_reported),
        ("inconsistent_boxes", test_inconsistent_periodic_boxes_reported),
        ("provenance_preservation", test_provenance_and_metadata_preservation),
        ("never_silently_invent", test_never_silently_invent),
    ]
    for (name, fn) in tests
        println("TEST ", name)
        fn()
    end
    println("")
    println("ALL TESTS PASSED")
end

_run_all()
