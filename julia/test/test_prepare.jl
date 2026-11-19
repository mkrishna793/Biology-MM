# Structure preparation tests.
#
# The behaviour under test is mostly *refusal*: the module must not silently
# resolve a scientific ambiguity. Each test constructs a fixture with a specific
# defect and asserts the specific either/or outcome: a deterministic repair that
# is logged, or an actionable error.

using FoundationCore
using FoundationCore.Types
using FoundationCore.Prepare
using FoundationCore.Factors
using FoundationCore.StreamedLJ
using Printf

const T = FoundationCore.Types
const PR = FoundationCore.Prepare

const FIXTURE_DIR = mktempdir()

"""
Fixed-column PDB line writer. Deliberately explicit about columns so the fixture
is readable and so a column-offset bug in the reader shows up as a wrong field
rather than as a mysterious parse failure.
"""
function pdb_line(; serial, name, resname = "ALA", chain = 'A', resseq = 1,
                  icode = ' ', altloc = ' ', occupancy = 1.0, element = "C",
                  x = 0.0, y = 0.0, z = 0.0)
    rec = "ATOM  "
    @sprintf("%-80s", string(
        rec,
        lpad(serial, 5),
        " ",
        rpad(name, 4),
        altloc,
        rpad(resname, 3),
        " ",
        chain,
        lpad(resseq, 4),
        icode,
        "   ",
        @sprintf("%8.3f%8.3f%8.3f", x, y, z),
        @sprintf("%6.2f", occupancy),
        "      ",
        rpad(element, 2),
        "  "))[1:80]
end

function write_pdb(path, lines; cryst = nothing)
    open(path, "w") do io
        if cryst !== nothing
            println(io, @sprintf("CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1           1",
                                 cryst[1], cryst[2], cryst[3], 90.0, 90.0, 90.0))
        end
        for l in lines
            println(io, l)
        end
        println(io, "END")
    end
    return path
end

"Deterministic, sane fixture: 4 atoms in a chain, well separated, with a box."
function clean_fixture()
    lines = [
        pdb_line(serial = 1, name = "N", x = 0.0, y = 0.0, z = 0.0, element = "N"),
        pdb_line(serial = 2, name = "CA", x = 1.5, y = 0.0, z = 0.0, element = "C"),
        pdb_line(serial = 3, name = "C", x = 2.8, y = 0.0, z = 0.0, element = "C"),
        pdb_line(serial = 4, name = "O", x = 3.9, y = 0.0, z = 0.0, element = "O"),
    ]
    return write_pdb(joinpath(FIXTURE_DIR, "clean.pdb"), lines; cryst = (20.0, 20.0, 20.0))
end

function test_clean_fixture_prepares()
    path = clean_fixture()
    rep = PR.diagnose_structure(path)
    @printf("  clean: N=%d altloc=%d zeroocc=%d close=%d min_dist=%.3f box=%s\n",
            rep.n_atoms, rep.n_altloc, rep.n_zero_occupancy, rep.n_close_contacts,
            rep.min_pair_distance, rep.box)
    @assert rep.n_atoms == 4
    @assert rep.n_altloc == 0
    @assert rep.n_close_contacts == 0
    @assert rep.box == (20.0, 20.0, 20.0)
    p = PR.prepare(path)
    @assert p.system.atoms.q |> size |> x -> x == (3, 4)
    @assert p.system.box isa T.OrthorhombicBox
    # the provenance log must always record that no parameters were assigned
    @assert any(a -> a.kind === :no_parameters, p.actions)
    @printf("  prepared: %s  actions=%s\n", string(p.system),
            string([a.kind for a in p.actions]))
    return true
end

function test_altloc_is_refused_by_default()
    lines = [
        pdb_line(serial = 1, name = "CA", altloc = 'A', occupancy = 0.6, x = 0.0, element = "C"),
        pdb_line(serial = 2, name = "CA", altloc = 'B', occupancy = 0.4, x = 1.2, element = "C"),
        pdb_line(serial = 3, name = "N", x = 2.0, element = "N"),
    ]
    path = write_pdb(joinpath(FIXTURE_DIR, "altloc.pdb"), lines)
    rep = PR.diagnose_structure(path)
    @assert rep.n_altloc == 2
    ok = false
    msg = ""
    try
        PR.prepare(path)
    catch e
        ok = true
        msg = sprint(showerror, e)
    end
    @assert ok "default policy must refuse altlocs"
    @assert occursin("altloc", msg)
    println("  default: altloc present -> refused with a message naming the choice")

    # an explicit policy makes it deterministic AND records which one was used
    pol = PR.PreparationPolicy(altloc = PR.AltlocPolicy(mode = :keep_highest_occupancy))
    p = PR.prepare(path, pol)
    @assert size(p.system.atoms.q, 2) == 2   # kept one CA + N
    @assert any(a -> a.kind === :altloc_highest_occupancy, p.actions)
    println("  explicit policy: kept highest occupancy, action logged")
    return true
end

function test_coincident_atoms_refused_before_force()
    lines = [
        pdb_line(serial = 1, name = "CA", x = 0.0, y = 0.0, z = 0.0),
        pdb_line(serial = 2, name = "CB", x = 0.0, y = 0.0, z = 0.0),  # exact overlap
        pdb_line(serial = 3, name = "N", x = 2.0, y = 0.0, z = 0.0, element = "N"),
    ]
    path = write_pdb(joinpath(FIXTURE_DIR, "coincident.pdb"), lines)
    rep = PR.diagnose_structure(path)
    @assert rep.n_close_contacts >= 1
    @assert rep.min_pair_distance == 0.0
    ok = false
    msg = ""
    try
        PR.prepare(path)
    catch e
        ok = true
        msg = sprint(showerror, e)
    end
    @assert ok "coincident atoms must be refused before force evaluation"
    @assert occursin("singular", msg)
    println("  coincident atoms: refused before any force is evaluated")

    # deterministic explicit repair
    pol = PR.PreparationPolicy(close_contact_action = :drop_second)
    p = PR.prepare(path, pol)
    @assert size(p.system.atoms.q, 2) == 2
    @assert any(a -> a.kind === :drop_close_contact_second, p.actions)
    # and the repaired geometry must actually be free of the contact
    rep2 = PR.diagnose_structure(path; contact_tol = 0.5)
    q = p.system.atoms.q
    dmin = Inf
    for i in 1:size(q, 2)-1, j in i+1:size(q, 2)
        dmin = min(dmin, sqrt(sum(abs2, q[:, i] .- q[:, j])))
    end
    @assert dmin >= 0.5 "repair left a close contact: dmin=$dmin"
    @printf("  explicit :drop_second -> %d atoms, min distance now %.3f\n",
            size(q, 2), dmin)
    return true
end

function test_close_contact_indices_follow_hydrogen_filter()
    lines = [
        pdb_line(serial = 1, name = "H", x = -2.0, element = "H"),
        pdb_line(serial = 2, name = "CA", x = 0.0, element = "C"),
        pdb_line(serial = 3, name = "CB", x = 0.1, element = "C"),
        pdb_line(serial = 4, name = "N", x = 2.0, element = "N"),
    ]
    path = write_pdb(joinpath(FIXTURE_DIR, "filtered_contact.pdb"), lines)
    policy = PR.PreparationPolicy(drop_hydrogens = true,
                                  close_contact_action = :drop_second)
    p = PR.prepare(path, policy)

    # After H removal CA/CB are indices 1/2, so :drop_second must remove CB,
    # not N (which was index 4 in the original structure).
    @assert size(p.system.atoms.q, 2) == 2
    @assert p.system.atoms.names == [:CA, :N]
    @assert p.system.atoms.q[1, :] == [0.0, 2.0]
    @assert any(a -> a.kind === :drop_hydrogens && a.n_affected == 1, p.actions)
    @assert any(a -> a.kind === :drop_close_contact_second && a.n_affected == 1, p.actions)
    mapped = PR.remap_pair_indices(p, [(2, 4)])
    @assert mapped == [(1, 2)]
    qref = [0.0 2.0; 0.0 0.0; 0.0 0.0]
    f_mapped = StreamedLJ.PairTopology(excluded=mapped)
    f_reference = StreamedLJ.PairTopology(excluded=[(1, 2)])
    boxref = T.OrthorhombicBox((8.0, 8.0, 8.0))
    ws_mapped = StreamedLJ.LJWorkspace(qref, boxref, 3.0)
    ws_reference = StreamedLJ.LJWorkspace(qref, boxref, 3.0)
    epsilon = [1.0, 1.0]
    sigma = [1.0, 1.0]
    um, fm, _ = StreamedLJ.evaluate!(ws_mapped, qref, epsilon, sigma, boxref;
                                     rcut=3.0, mode=:lfs, topology=f_mapped)
    ur, fr, _ = StreamedLJ.evaluate!(ws_reference, qref, epsilon, sigma, boxref;
                                     rcut=3.0, mode=:lfs, topology=f_reference)
    @assert um == ur == 0.0
    @assert fm == fr == zeros(3, 2)
    threw = false
    try
        PR.remap_pair_indices(p, [(2, 3)])
    catch e
        threw = e isa ArgumentError
    end
    @assert threw "topology pairs touching removed atoms must be refused"
    println("  filtered contact indices: removed CB; retained CA and N at original coordinates")
    println("  source pair remap (CA,N) -> (1,2); explicit exclusion energy/force parity and dropped-atom refusal passed")
    return true
end

function test_nudge_is_logged_as_geometry_change()
    lines = [
        pdb_line(serial = 1, name = "CA", x = 0.0, y = 0.0, z = 0.0),
        pdb_line(serial = 2, name = "CB", x = 0.01, y = 0.0, z = 0.0),
        pdb_line(serial = 3, name = "N", x = 2.0, y = 0.0, z = 0.0, element = "N"),
    ]
    path = write_pdb(joinpath(FIXTURE_DIR, "nearcoincident.pdb"), lines)
    pol = PR.PreparationPolicy(close_contact_action = :nudge, nudge_distance = 0.9)
    p = PR.prepare(path, pol)
    act = [a for a in p.actions if a.kind === :nudge_close_contacts]
    @assert length(act) == 1
    # the log must say the geometry was modified — this is the honesty requirement
    @assert occursin("MODIFIED", act[1].detail)
    @printf("  nudge logged as a geometry change: \"%s\"\n", act[1].detail)

    # default nudge_distance (1e-3) is smaller than close_contact_tol (0.5): must refuse
    # rather than silently returning an unresolved close contact.
    bad_nudge = PR.PreparationPolicy(close_contact_action = :nudge)
    ok = false
    msg = ""
    try
        PR.prepare(path, bad_nudge)
    catch e
        ok = true
        msg = sprint(showerror, e)
    end
    @assert ok "unresolved post-nudge close contact must be refused"
    @assert occursin("unresolved close contacts", msg)
    println("  insufficient nudge refused rather than leaving close contact")
    return true
end

function test_nonfinite_coordinates_diagnosed_and_refused()
    bad_line = pdb_line(serial = 1, name = "CA", x = 0.0, y = 0.0, z = 0.0)
    nan_line = replace(bad_line, "   0.000   0.000   0.000" => "     NaN   0.000   0.000")
    path = write_pdb(joinpath(FIXTURE_DIR, "nonfinite.pdb"), [
        nan_line,
        pdb_line(serial = 2, name = "N", x = 2.0, element = "N"),
    ])
    rep = PR.diagnose_structure(path)
    @assert rep.n_nonfinite == 1
    @assert any(occursin("non-finite", s) for s in rep.issues)
    ok = false
    try
        PR.prepare(path)
    catch e
        ok = true
    end
    @assert ok "non-finite coordinates must be refused in prepare"
    println("  non-finite coordinates: diagnosed and refused")
    return true
end

function test_missing_box_policy()
    lines = [pdb_line(serial = 1, name = "CA", x = 0.0), pdb_line(serial = 2, name = "N", x = 2.0, element = "N")]
    path = write_pdb(joinpath(FIXTURE_DIR, "nobox.pdb"), lines)
    rep = PR.diagnose_structure(path)
    @assert rep.box === nothing
    @assert any(occursin("no CRYST1", s) for s in rep.issues)
    p = PR.prepare(path)  # default allows non-periodic
    @assert p.system.box isa T.NoBox
    ok = false
    try
        PR.prepare(path, PR.PreparationPolicy(require_box = true))
    catch e
        ok = true
    end
    @assert ok "require_box policy must fail when there is no CRYST1"
    println("  missing box: allowed as non-periodic by default, refused when required")
    return true
end

function test_duplicate_identity_always_fatal()
    lines = [
        pdb_line(serial = 1, name = "CA", resseq = 1, x = 0.0),
        pdb_line(serial = 2, name = "CA", resseq = 1, x = 2.0),  # same identity
    ]
    path = write_pdb(joinpath(FIXTURE_DIR, "dup.pdb"), lines)
    ok = false
    msg = ""
    try
        PR.prepare(path)
    catch e
        ok = true
        msg = sprint(showerror, e)
    end
    @assert ok "duplicate identity must always be fatal"
    @assert occursin("duplicate", msg)
    println("  duplicate identity: always fatal, no silent choice")
    return true
end

function test_report_names_its_parser()
    path = clean_fixture()
    rep = PR.diagnose_structure(path)
    @assert occursin("narrow", rep.parser)
    @printf("  parser provenance recorded: \"%s\"\n", first(split(rep.parser, ";")))
    return true
end

function _run_all()
    for (name, fn) in [
        ("clean_fixture", test_clean_fixture_prepares),
        ("altloc_refused", test_altloc_is_refused_by_default),
        ("coincident_refused", test_coincident_atoms_refused_before_force),
        ("filtered_contact_indices", test_close_contact_indices_follow_hydrogen_filter),
        ("nudge_logged", test_nudge_is_logged_as_geometry_change),
        ("missing_box", test_missing_box_policy),
        ("duplicate_identity", test_duplicate_identity_always_fatal),
        ("parser_provenance", test_report_names_its_parser),
        ("nonfinite_coordinates", test_nonfinite_coordinates_diagnosed_and_refused),
    ]
        println("TEST ", name)
        fn()
    end
    println("")
    println("ALL TESTS PASSED")
end

_run_all()
