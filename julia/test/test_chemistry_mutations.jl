# Mutation controls for the parameter -> resolved-system mapping.
#
# A mapping test that cannot fail is not evidence. Each check perturbs one
# normalized input field (bond prefactor, 1-4 scale, torsion atom order, charge
# factor) and requires a measurable, analytically predicted change in the own
# engine's energy/forces.

using FoundationCore
const C = FoundationCore.Chemistry
const SC = FoundationCore.SystemContract
const F = FoundationCore.Factors
const T = FoundationCore.Types

function ns_fixture(; bond_k = 1000.0, lj14 = 0.5, charges = [0.4, -0.2, 0.1, -0.3],
                    dihedral_order = (1, 2, 3, 4))
    atoms = [
        C.NormalizedAtom("C", 12.011, charges[1], "C", 1, "LIG", 0.34, 0.36),
        C.NormalizedAtom("H", 1.008, charges[2], "H", 1, "LIG", 0.24, 0.08),
        C.NormalizedAtom("O", 15.999, charges[3], "O", 2, "LIG", 0.30, 0.64),
        C.NormalizedAtom("N", 14.007, charges[4], "N", 2, "LIG", 0.32, 0.20),
    ]
    i, j, k, l = dihedral_order
    return C.NormalizedSystem(C.CHEM_SCHEMA_VERSION, :amber, C.CHEM_UNITS_NAME,
        C.ImportProvenance("mutation.json", "mutation-control force field",
                           "c"^64, "", "", "test", "1.0.0", ""),
        atoms, C.NormalizedBox(:orthorhombic, (3.0, 3.0, 3.0), zeros(3, 3)),
        [C.BondTerm(1, 2, bond_k, 0.109)],
        [C.AngleTerm(2, 1, 3, 100.0, 1.9)],
        C.UreyBradleyTerm[],
        [C.DihedralTerm(i, j, k, l, [C.TorsionTerm(1.0, 2, 0.2)])],
        [C.ImproperTerm(1, 2, 3, 4, 10.0, 0.0)],
        C.CmapTerm[], [(1, 3), (2, 3)],
        [C.Pair14Term(1, 4, 1 / 1.2, lj14)],
        C.C64Term[], C.VirtualSiteTerm[], C.ConstraintTerm[], C.DiagnosticRecord[],
        sum(charges))
end

function build(ns)
    prot = C.default_protocol(ns; nonbonded = :pme, cutoff_nm = 1.0, lj_mode = :trunc,
                              pme_grid = (24, 24, 24))
    return C.build_resolved(ns; protocol = prot)
end

function main()
    println("="^72)
    println("Chemistry mutation controls")
    println("="^72)
    q = [0.0 0.150 0.290 0.160;
         0.0 0.020 0.090 0.190;
         0.0 0.040 0.010 0.020]

    base = build(ns_fixture())
    U0, F0 = SC.energy_forces(base, q)

    # 1. Bond prefactor doubling: the harmonic bond energy must change by
    # exactly 0.5*k*(r-r0)^2 (the removable part is linear in k).
    mut = build(ns_fixture(; bond_k = 2000.0))
    U1, F1 = SC.energy_forces(mut, q)
    r = sqrt(T.norm2(T.unwrap_displacement(base.box, q, 1, 2)))
    expected = 0.5 * 1000.0 * (r - 0.109)^2
    got = U1 - U0
    @assert abs(got - expected) <= 1e-9 * max(1.0, abs(expected)) + 1e-12
    @assert maximum(abs, F1 .- F0) > 1e-6 "bond prefactor mutation was invisible"
    println("  bond k 1000->2000: dU=$got expected=$expected")

    # 2. 1-4 LJ scale 0.5 -> 1.0: only that scaled pair may change, by the
    # analytic pair energy at the actual distance.
    mut = build(ns_fixture(; lj14 = 1.0))
    U2, F2 = SC.energy_forces(mut, q)
    eps = sqrt(0.36 * 0.20)
    sig = 0.5 * (0.34 + 0.32)
    r14 = sqrt(T.norm2(T.unwrap_displacement(base.box, q, 1, 4)))
    lj = 4.0 * eps * ((sig / r14)^12 - (sig / r14)^6)
    got2 = U2 - U0
    @assert r14 < 1.0 "1-4 pair must be inside the cutoff for this control"
    @assert abs(got2 - 0.5 * lj) <= 1e-9 * max(1.0, abs(lj)) + 1e-12
    @assert maximum(abs, F2 .- F0) > 1e-9 "1-4 scale mutation was invisible"
    println("  LJ 1-4 scale 0.5->1.0: dU=$got2 expected=$(0.5 * lj)")

    # 3. Torsion atom order: reverse the first two atoms, which changes the
    # signed angle. The energy and forces must change; an index mix-up cannot be
    # silently absorbed.
    mut = build(ns_fixture(; dihedral_order = (2, 1, 3, 4)))
    U3, F3 = SC.energy_forces(mut, q)
    @assert abs(U3 - U0) > 1e-8 "dihedral atom-order mutation was invisible"
    @assert maximum(abs, F3 .- F0) > 1e-8
    println("  dihedral order (1,2,3,4)->(2,1,3,4): |dU|=$(abs(U3 - U0))")

    # 4. Charge conversion: scale one charge by 10. PME and the reciprocal
    # energy must respond; a kcal/mol-vs-kJ/mol or e-vs-C mix-up would not be
    # hidden by the self term.
    mut = build(ns_fixture(; charges = [4.0, -0.2, 0.1, -0.3]))
    U4, F4 = SC.energy_forces(mut, q)
    @assert abs(U4 - U0) > 1.0 "charge mutation was invisible"
    @assert maximum(abs, F4 .- F0) > 1e-3
    println("  charge 0.4->4.0: |dU|=$(abs(U4 - U0))")

    # 5. Component addressing must select the intended factor, not merely sum
    # everything. A wrong component name must throw.
    c = SC.component_energy(base, q, :bond_1)
    fb = F.HarmonicBond(1, 2, 1000.0, 0.109)
    direct = F.evaluate!(zeros(3, 4), fb, q, F.Context(base.box, C.CHEM_UNITS, 4))
    @assert abs(c - direct) <= 1e-12 * max(1.0, abs(direct))
    threw = false
    try
        SC.component_energy(base, q, :no_such_component)
    catch
        threw = true
    end
    @assert threw
    println("  component :bond_1 direct parity: $c")

    println("CHEMISTRY MUTATION CONTROLS PASSED")
    return true
end

main()
