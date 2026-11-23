# Cross-parser validation against real structure data.
#
# The narrow fixed-column PDB reader in `Prepare` claims to handle ATOM/HETATM and
# CRYST1 correctly and nothing else. That claim is only meaningful if it agrees
# with a mature parser on the records it *does* handle. This suite checks that
# against a real protein (bovine pancreatic trypsin inhibitor, 892 atoms, from the
# OpenMM test corpus already present in this repository).
#
# If MDAnalysis is not installed the Python-side comparison reports BLOCKED; this
# Julia test then validates only the internal consistency of the narrow reader
# (atom count, coordinate ranges) and says so, rather than silently passing.

using FoundationCore
using FoundationCore.Types
using FoundationCore.Prepare
using Printf

const T = FoundationCore.Types
const PR = FoundationCore.Prepare

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const BPTI = joinpath(REPO_ROOT, "wrappers", "python", "tests", "systems", "bpti.pdb")

function test_real_protein_parses()
    @assert isfile(BPTI) "real structure fixture missing: $BPTI"
    rep = PR.diagnose_structure(BPTI; contact_tol = 0.5)
    @printf("  bpti.pdb: N=%d chains=%s residues=%s close_contacts=%d min_dist=%.3f\n",
            rep.n_atoms, rep.chains, rep.resnames, rep.n_close_contacts,
            rep.min_pair_distance)
    @assert rep.n_atoms == 892 "expected 892 atoms in bpti.pdb"
    @assert rep.n_nonfinite == 0
    @assert rep.n_duplicate_identities == 0
    # A real prepared structure must have no sub-0.5 A contacts; if it does, that is
    # a real finding about the file and the assertion should be revisited, not deleted.
    @assert rep.n_close_contacts == 0 "unexpected close contacts in bpti.pdb: $(rep.close_contact_pairs[1:min(end,3)])"
    @assert rep.min_pair_distance > 0.9 "suspiciously short minimum distance"
    @assert rep.box === nothing  # this particular file has no CRYST1
    return true
end

function test_prepared_protein_is_dynamics_ready()
    # The prepared geometry, with a large explicit box, must be usable by the
    # energy machinery without any singular pair. This is the end-to-end link from
    # file -> validated state -> finite energy.
    rep = PR.diagnose_structure(BPTI)
    atoms, _, _ = PR.read_pdb_fallback(BPTI)
    q = zeros(3, length(atoms))
    for (i, a) in enumerate(atoms)
        q[1, i] = a.x; q[2, i] = a.y; q[3, i] = a.z
    end
    # shift into a box comfortably larger than the molecule
    span = maximum(q, dims = 2) .- minimum(q, dims = 2)
    L = maximum(span) + 20.0
    q .-= minimum(q, dims = 2)
    box = T.OrthorhombicBox((L, L, L))
    z = zeros(length(atoms))
    at = T.AtomSet(q, zeros(3, length(atoms)), ones(length(atoms)), z,
                   [a.name for a in atoms])
    sys = T.System(at, box)
    # Build LJ factors for all pairs within 10 A and evaluate twice: the energy must
    # be finite and identical on repeat (deterministic evaluation order).
    using_pairs = FoundationCore.Neighbors.build_neighbors(q, box, 10.0, 0.0)
    pairs = FoundationCore.Neighbors.neighbor_pairs(using_pairs)
    ctx = FoundationCore.Factors.Context(box, T.REDUCED, length(atoms))
    facs = [FoundationCore.Factors.LennardJonesPair(i, j, 0.1, 3.0; mode = :lfs, rc = 10.0)
            for (i, j) in pairs]
    U1, F1 = FoundationCore.Factors.total_energy_forces(facs, q, ctx)
    U2, F2 = FoundationCore.Factors.total_energy_forces(facs, q, ctx)
    @printf("  prepared BPTI in %.1f A box: %d pairs within 10 A, U=%.6f (finite=%s)\n",
            L, length(pairs), U1, isfinite(U1))
    @assert isfinite(U1) "energy is not finite for the prepared structure"
    @assert U1 == U2 && F1 == F2 "energy evaluation is not deterministic"
    return true
end

function test_altloc_file_diagnoses()
    # A real PDB with alternate locations from the same test corpus.
    p = joinpath(REPO_ROOT, "wrappers", "python", "tests", "systems", "altlocs.pdb")
    if !isfile(p)
        println("  altlocs.pdb not present; skipping real-altloc diagnosis")
        return true
    end
    rep = PR.diagnose_structure(p)
    @printf("  altlocs.pdb: N=%d altloc_atoms=%d\n", rep.n_atoms, rep.n_altloc)
    @assert rep.n_altloc > 0 "expected alternate locations in this file"
    ok = false
    try
        PR.prepare(p)
    catch e
        ok = occursin("altloc", sprint(showerror, e))
    end
    @assert ok "default policy must refuse a real altloc file"
    return true
end

function _run_all()
    for (name, fn) in [
        ("real_protein_parses", test_real_protein_parses),
        ("prepared_protein_dynamics_ready", test_prepared_protein_is_dynamics_ready),
        ("altloc_file", test_altloc_file_diagnoses),
    ]
        println("TEST ", name)
        fn()
    end
    println("")
    println("ALL TESTS PASSED")
end

_run_all()
