# Builder suite: normalized record -> `ResolvedSystem`.
#
# Run standalone:
#
#     cd julia
#     julia --project=. test/test_chemistry_build.jl
#
# Assertion-driven and independent of any parameter file: every fixture is built
# directly from the public schema structs, so the suite pins the builder's wiring
# (factor order, names, exclusions, 1-4 scales, constraints, protocol refusals)
# rather than the importer. The parity checks evaluate the *same* factor classes
# through `Factors.total_energy_forces`, so a mis-wired component name or a
# dropped exclusion changes a number here.

using FoundationCore

const C  = FoundationCore.Chemistry
const T  = FoundationCore.Types
const F  = FoundationCore.Factors
const RS = FoundationCore.ResolvedSystem
const SC = FoundationCore.SystemContract
const CF = FoundationCore.ChemistryFactors
const SL = FoundationCore.StreamedLJ
const PM = FoundationCore.PME
const CN = FoundationCore.Constraints

function caught_throw(f::Function)
    try
        f()
    catch e
        return e
    end
    return nothing
end

"Catch a `ChemistryError` with `code`, optionally require message substrings."
function assert_chem_error(code::Symbol, f::Function, needles::Vector{String} = String[])
    e = caught_throw(f)
    e isa C.ChemistryError || error("expected ChemistryError[$code], got $(repr(e))")
    @assert e.code === code "expected error code $code, got $(e.code): $(e.message)"
    text = e.path * ": " * e.message
    for needle in needles
        @assert occursin(needle, text) "message for $code must mention " *
                                       "$(repr(needle)); got: $text"
    end
    return e
end

# ---------------------------------------------------------------------------
# fixtures: 6-atom mixed organic-like system, everything explicit
# ---------------------------------------------------------------------------

const BOND_R0 = (0.152, 0.145, 0.150, 0.141, 0.096)

function six_atoms()
    return [
        C.NormalizedAtom("C1", 12.011, 0.40, "C", 1, "LIG", 0.34, 0.36),
        C.NormalizedAtom("C2", 12.011, -0.20, "C", 1, "LIG", 0.34, 0.36),
        C.NormalizedAtom("N1", 14.007, 0.10, "N", 1, "LIG", 0.32, 0.20),
        C.NormalizedAtom("C3", 12.011, -0.30, "C", 1, "LIG", 0.34, 0.36),
        C.NormalizedAtom("O1", 15.999, 0.20, "O", 2, "LIG", 0.30, 0.64),
        C.NormalizedAtom("H1", 1.008, -0.20, "H", 2, "LIG", 0.24, 0.08),
    ]
end

const SIX_Q = [0.10 0.26 0.39 0.55 0.68 0.74;
               0.12 0.15 0.31 0.28 0.44 0.41;
               0.20 0.22 0.19 0.24 0.16 0.30]

"6 atoms, 5 bonds, 4 angles, 1 Urey-Bradley, one 2-term dihedral, 1 improper,
5 exclusions and one 1-4 pair. Every nonbonded rule is exercised."
function six_system(; box = C.NormalizedBox(:orthorhombic, (3.0, 3.2, 3.4), zeros(3, 3)),
                    atoms = six_atoms(),
                    bonds = [C.BondTerm(1, 2, 300_000.0, BOND_R0[1]),
                             C.BondTerm(2, 3, 320_000.0, BOND_R0[2]),
                             C.BondTerm(3, 4, 280_000.0, BOND_R0[3]),
                             C.BondTerm(4, 5, 400_000.0, BOND_R0[4]),
                             C.BondTerm(5, 6, 450_000.0, BOND_R0[5])],
                    angles = [C.AngleTerm(1, 2, 3, 600.0, 1.95),
                              C.AngleTerm(2, 3, 4, 500.0, 2.05),
                              C.AngleTerm(3, 4, 5, 700.0, 1.90),
                              C.AngleTerm(4, 5, 6, 550.0, 1.85)],
                    urey_bradleys = [C.UreyBradleyTerm(1, 3, 80_000.0, 0.25)],
                    dihedrals = [C.DihedralTerm(1, 2, 3, 4,
                                                [C.TorsionTerm(1.2, 2, 0.4),
                                                 C.TorsionTerm(0.5, 3, pi)])],
                    impropers = [C.ImproperTerm(2, 3, 4, 6, 40.0, -0.3)],
                    exclusions = [(1, 2), (2, 3), (3, 4), (1, 3), (2, 4)],
                    pair14 = [C.Pair14Term(1, 4, 1 / 1.2, 0.5)],
                    virtual_sites = C.VirtualSiteTerm[],
                    constraints = C.ConstraintTerm[],
                    net_charge = 0.0)
    ns = C.NormalizedSystem(v"1.0.0", :amber, C.CHEM_UNITS_NAME,
                            C.ImportProvenance("fixture-six.json", "unit-test force field",
                                               "a"^64, "", "", "test", "1.0.0", ""),
                            atoms, box, bonds, angles, urey_bradleys, dihedrals,
                            impropers, C.CmapTerm[], exclusions, pair14, C.C64Term[],
                            virtual_sites, constraints, C.DiagnosticRecord[], net_charge)
    return C.validate_normalized!(ns)
end

no_box_six() = six_system(; box = C.NormalizedBox())

"Manual factor list for the six-atom periodic fixture, assembled independently of
the builder. Used to prove the builder wired the documented factors."
function manual_factors(ns::C.NormalizedSystem, protocol::C.SimulationProtocol)
    box = T.OrthorhombicBox(ns.box.lengths_nm)
    N = length(ns.atoms)
    charges = [a.charge for a in ns.atoms]
    factors = F.AbstractFactor[
        F.HarmonicBond(b.i, b.j, b.k, b.r0_nm) for b in ns.bonds]
    append!(factors, [F.HarmonicAngle(a.i, a.j, a.k, a.k_theta, a.theta0_rad)
                      for a in ns.angles])
    append!(factors, [CF.UreyBradley(u.i, u.j, u.k, u.r0_nm) for u in ns.urey_bradleys])
    append!(factors, [CF.MultiPeriodicDihedral(d.i, d.j, d.k, d.l,
                                              [t.k for t in d.terms],
                                              [t.n for t in d.terms],
                                              [t.phase_rad for t in d.terms])
                      for d in ns.dihedrals if length(d.terms) > 1])
    append!(factors, [F.PeriodicDihedral(d.i, d.j, d.k, d.l, d.terms[1].k,
                                         d.terms[1].n, d.terms[1].phase_rad)
                      for d in ns.dihedrals if length(d.terms) == 1])
    append!(factors, [CF.HarmonicImproper(t.i, t.j, t.k, t.l, t.k_phi, t.phase_rad)
                      for t in ns.impropers])
    push!(factors, SL.StreamedLJFactor(N, [a.epsilon_kj for a in ns.atoms],
                                       [a.sigma_nm for a in ns.atoms], box;
                                       rcut = protocol.cutoff_nm, mode = protocol.lj_mode,
                                       topology = SL.PairTopology(
                                           excluded = ns.exclusions,
                                           scaled = [(p.i, p.j, p.lj_scale)
                                                     for p in ns.pair14])))
    push!(factors, PM.PMEFactor(zeros(3, N), charges, box;
                                alpha = protocol.pme_alpha, rcut = protocol.cutoff_nm,
                                grid = protocol.pme_grid, order = protocol.pme_order,
                                prefactor = C.CHEM_COULOMB_PREFACTOR,
                                corrections = PM.PairCorrections(
                                    excluded = ns.exclusions,
                                    scaled = [(p.i, p.j, p.coulomb_scale)
                                              for p in ns.pair14])))
    return factors
end

function maxabs(A)
    isempty(A) && return 0.0
    return maximum(abs, A)
end

# ---------------------------------------------------------------------------
# 1. protocol validation
# ---------------------------------------------------------------------------

println("="^72)
println("Chemistry builder tests")
println("="^72)

ns = six_system()
p_def = C.default_protocol(ns)
@assert p_def.nonbonded === :pme
@assert p_def.cutoff_nm == 1.0
@assert p_def.lj_mode === :trunc
@assert p_def.pme_alpha == 3.5          # nm^-1, not the Å^-1 value 0.35
@assert p_def.pme_order == 4
@assert p_def.water_model === :flexible
@assert p_def.background === :uniform
@assert p_def.temperature_k == 298.15
@assert p_def.pme_grid === nothing
@assert C.default_protocol(no_box_six()).nonbonded === :direct

tric = C.NormalizedBox(:triclinic, (0.0, 0.0, 0.0),
                       [3.0 0.5 0.2; 0.0 3.1 0.3; 0.0 0.0 3.2])
ns_tric = six_system(; box = tric)
@assert C.default_protocol(ns_tric).nonbonded === :pme   # returned; build refuses

assert_chem_error(:protocol, () -> C.SimulationProtocol(nonbonded = :ewald),
                  ["nonbonded", ":pme"])
assert_chem_error(:protocol, () -> C.SimulationProtocol(lj_mode = :shift),
                  ["lj_mode", ":trunc"])
assert_chem_error(:protocol, () -> C.SimulationProtocol(water_model = :tip4p),
                  ["water_model", ":rigid"])
assert_chem_error(:protocol, () -> C.SimulationProtocol(background = :jellium),
                  ["background", ":uniform"])
assert_chem_error(:protocol, () -> C.SimulationProtocol(cutoff_nm = 0.0),
                  ["cutoff_nm", "0 <"])
assert_chem_error(:protocol, () -> C.SimulationProtocol(cutoff_nm = Inf),
                  ["cutoff_nm"])
assert_chem_error(:protocol, () -> C.SimulationProtocol(pme_order = 5),
                  ["pme_order", "4 or 6"])
assert_chem_error(:protocol, () -> C.SimulationProtocol(pme_alpha = 0.0),
                  ["pme_alpha", "> 0"])
assert_chem_error(:protocol, () -> C.SimulationProtocol(pme_grid = (8, 8)),
                  ["pme_grid", "three"])
assert_chem_error(:protocol, () -> C.SimulationProtocol(pme_grid = (8, 8, 4)),
                  ["pme_grid", "2*pme_order"])
assert_chem_error(:protocol, () -> C.SimulationProtocol(temperature_k = -1.0),
                  ["temperature_k"])
# the :pme cutoff bound is a protocol error, not a backend ArgumentError
assert_chem_error(:protocol, () -> C.default_protocol(ns; cutoff_nm = 1.6),
                  ["cutoff_nm", "min(L)/2"])
assert_chem_error(:protocol, () -> C.default_protocol(ns_tric; cutoff_nm = 1.6),
                  ["cutoff_nm"])
assert_chem_error(:protocol, () -> C.build_resolved(ns;
                    protocol = C.SimulationProtocol(cutoff_nm = 1.5)),
                  ["cutoff_nm", "min(L)/2"])
@assert C.SimulationProtocol(nonbonded = :none).nonbonded === :none
println("  protocol: defaults + 12 typed :protocol refusals hold")
println("  min(L)/2 = $(0.5 * minimum(ns.box.lengths_nm)) nm for L = $(ns.box.lengths_nm)")

# ---------------------------------------------------------------------------
# 2. periodic build: factor order, names, energy, components
# ---------------------------------------------------------------------------

rs = C.build_resolved(ns; protocol = p_def)
@assert rs isa RS.ResolvedSystem
@assert RS.n_atoms(rs) == 6
@assert rs.box isa T.OrthorhombicBox
@assert rs.box.L == ns.box.lengths_nm
@assert rs.units === C.CHEM_UNITS
@assert rs.declared == Set([:energy, :cartesian_forces, :component_energies])
@assert rs.provenance.source == ns.provenance.source
@assert rs.provenance.force_field == ns.provenance.force_field

expected_names = [:bond_1, :bond_2, :bond_3, :bond_4, :bond_5,
                  :angle_1, :angle_2, :angle_3, :angle_4,
                  :urey_1, :dihedral_1, :improper_1, :lj, :pme]
@assert rs.component_names == expected_names
@assert length(rs.factors) == length(expected_names)
@assert rs.factors[1] isa F.HarmonicBond
@assert rs.factors[6] isa F.HarmonicAngle
@assert rs.factors[10] isa CF.UreyBradley
@assert rs.factors[11] isa CF.MultiPeriodicDihedral   # two terms, not two factors
@assert length(rs.factors[11].ks) == 2
@assert rs.factors[12] isa CF.HarmonicImproper
@assert rs.factors[13] isa SL.StreamedLJFactor
@assert rs.factors[14] isa PM.PMEFactor
@assert rs.constraints === nothing
@assert RS.validate_resolved!(rs) === rs

U, Fp = SC.energy_forces(rs, SIX_Q)
@assert isfinite(U)
@assert all(isfinite, Fp)
ce = SC.component_energies(rs, SIX_Q)
@assert collect(keys(ce)) == rs.component_names
@assert sum(values(ce)) == U "component sum must reproduce the total exactly"
@assert SC.energy(rs, SIX_Q) == U
@assert SC.component_energy(rs, SIX_Q, :lj) == ce.lj
@assert SC.component_energy(rs, SIX_Q, :pme) == ce.pme

# restrictions on :trunc still hold, while :lfs + PME supports full box_derivative
@assert !(:box_derivative in RS.engine_capabilities(rs))
e_refuse = caught_throw(() -> SC.box_derivative(rs, SIX_Q))
@assert e_refuse isa SC.MissingBoxDerivative
# per-component: :trunc LJ has no analytic stress, PME supports box_derivative
e_lj = caught_throw(() -> SC.box_derivative_component(rs, SIX_Q, :lj))
@assert e_lj isa SC.MissingBoxDerivative && e_lj.component === :lj
rs_lfs = C.build_resolved(ns; protocol = C.default_protocol(ns; lj_mode = :lfs))
@assert :box_derivative in RS.engine_capabilities(rs_lfs)
G_lfs_all = SC.box_derivative(rs_lfs, SIX_Q)
@assert size(G_lfs_all) == (3, 3) && all(isfinite, G_lfs_all)
# All bonded/local chemistry factors, :lfs StreamedLJ, and PME support per-component box derivatives.
for comp_name in (:bond_1, :angle_1, :urey_1, :dihedral_1, :improper_1, :lj, :pme)
    Gc = SC.box_derivative_component(rs_lfs, SIX_Q, comp_name)
    @assert size(Gc) == (3, 3) && all(isfinite, Gc) "component $comp_name box derivative failed"
end
println("  periodic :pme  U = $U  (14 components, sum matches exactly)")
println("  max|F| = $(maxabs(Fp)); pme component = $(ce.pme); lj component = $(ce.lj)")

# ---------------------------------------------------------------------------
# 2b. zero-LJ atoms: sigma = epsilon = 0 is an exactly zero contribution
# ---------------------------------------------------------------------------

hz_atoms = [C.NormalizedAtom("H1", 1.008, 0.0, "H", 1, "WAT", 0.0, 0.0),
            C.NormalizedAtom("O", 15.999, 0.0, "O", 1, "WAT", 0.31506, 0.6364)]
hz_ns = C.NormalizedSystem(v"1.0.0", :amber, C.CHEM_UNITS_NAME,
                           C.ImportProvenance("fixture-zero-lj.json",
                                              "unit-test force field", "e"^64, "", "",
                                              "test", "1.0.0", ""),
                           hz_atoms,
                           C.NormalizedBox(:orthorhombic, (3.0, 3.0, 3.0), zeros(3, 3)),
                           C.BondTerm[], C.AngleTerm[], C.UreyBradleyTerm[],
                           C.DihedralTerm[], C.ImproperTerm[], C.CmapTerm[],
                           NTuple{2,Int}[], C.Pair14Term[], C.C64Term[], C.VirtualSiteTerm[],
                           C.ConstraintTerm[], C.DiagnosticRecord[], 0.0)
C.validate_normalized!(hz_ns)
hz_rs = C.build_resolved(hz_ns)
hz_lj = hz_rs.factors[findfirst(f -> f isa SL.StreamedLJFactor, hz_rs.factors)]
@assert hz_lj.sigma[1] == 0.0 && hz_lj.epsilon[1] == 0.0
qz = [0.30 0.80; 0.40 1.10; 0.50 0.90]
Fz = zeros(3, 2)
@assert F.evaluate!(Fz, hz_lj, qz, F.Context(hz_rs.box, hz_rs.units, 2)) == 0.0
@assert maxabs(Fz) == 0.0
@assert SC.component_energy(hz_rs, qz, :lj) == 0.0
println("  zero-LJ hydrogen: sigma = epsilon = 0 builds, LJ energy and force exactly 0")

# ... but a zero size with a nonzero well is a typed parameter error
bad_atoms = [C.NormalizedAtom("H1", 1.008, 0.0, "H", 1, "WAT", 0.0, 0.05),
             C.NormalizedAtom("O", 15.999, 0.0, "O", 1, "WAT", 0.31506, 0.6364)]
bad_ns = C.NormalizedSystem(v"1.0.0", :amber, C.CHEM_UNITS_NAME,
                            C.ImportProvenance("fixture-bad-lj.json",
                                               "unit-test force field", "f"^64, "", "",
                                               "test", "1.0.0", ""),
                            bad_atoms, hz_ns.box, C.BondTerm[], C.AngleTerm[],
                            C.UreyBradleyTerm[], C.DihedralTerm[], C.ImproperTerm[],
                            C.CmapTerm[], NTuple{2,Int}[], C.Pair14Term[], C.C64Term[],
                            C.VirtualSiteTerm[], C.ConstraintTerm[],
                            C.DiagnosticRecord[], 0.0)
assert_chem_error(:parameter, () -> C.build_resolved(bad_ns),
                  ["atoms[1].sigma_nm", "sigma_nm = 0", "epsilon_kj"])

# ---------------------------------------------------------------------------
# 3. wiring parity: same factors, both routes, independent assembly
# ---------------------------------------------------------------------------

manual = manual_factors(ns, p_def)
ctx = F.Context(rs.box, rs.units, 6)
Um, Fm = F.total_energy_forces(manual, SIX_Q, ctx)
dU = abs(Um - U)
dF = maxabs(Fm .- Fp)
@assert dU <= 1e-12 "builder/manual energy mismatch: $dU"
@assert dF <= 1e-12 "builder/manual force mismatch: $dF"

# the builder's exclusions and 1-4 scales reach the LJ kernel, not just the record
lj = rs.factors[13]
@assert lj.rcut == 1.0
@assert lj.mode === :trunc
@assert lj.topology !== nothing
@assert SL.pair_weight(lj.topology, 1, 2) == 0.0        # excluded
@assert SL.pair_weight(lj.topology, 1, 3) == 0.0        # excluded
@assert SL.pair_weight(lj.topology, 1, 4) == 0.5        # 1-4 LJ scale
@assert SL.pair_weight(lj.topology, 2, 5) == 1.0        # untouched pair
@assert rs.factors[14].corrections.weights[(1, 2)] == 0.0
@assert isapprox(rs.factors[14].corrections.weights[(1, 4)], 1 / 1.2; rtol = 0.0)

# topology is metadata: every imported bond is listed even under a rigid choice
@assert length(rs.topology.bonds) == length(ns.bonds)
@assert length(rs.topology.angles) == length(ns.angles)
@assert length(rs.topology.dihedrals) == 1
@assert rs.topology.dihedrals[1].kphi == 1.2            # first term, documented
@assert rs.topology.dihedrals[1].n == 2
@assert rs.topology.exclusions == Set([(1, 2), (2, 3), (3, 4), (1, 3), (2, 4)])
@assert rs.topology.scales[(1, 4)] == (0.5, 1 / 1.2)
@assert T.pair_scale(rs.topology, 1, 4) == (0.5, 1 / 1.2)
println("  wiring parity: dU = $dU, max|dF| = $dF (builder vs independent assembly)")

# ---------------------------------------------------------------------------
# 4. direct/NoBox: explicit pair table, exclusion, 1-4 scale, FD force check
# ---------------------------------------------------------------------------

ns_direct = no_box_six()
p_direct = C.default_protocol(ns_direct)
@assert p_direct.nonbonded === :direct
rsd = C.build_resolved(ns_direct; protocol = p_direct)
@assert rsd.box isa T.NoBox
@assert rsd.component_names[end] === :nonbonded
@assert count(f -> f isa CF.PairTableNonbonded, rsd.factors) == 1

tbl = rsd.factors[end]
@assert tbl.mode === :none
npairs = 6 * 5 ÷ 2 - length(ns_direct.exclusions)
@assert length(tbl.i) == npairs
@assert length(tbl.i) == 10 && npairs == 10
@assert !any(p -> tbl.i[p] == 1 && tbl.j[p] == 2, eachindex(tbl.i))
@assert !any(p -> tbl.i[p] == 1 && tbl.j[p] == 3, eachindex(tbl.i))
p14 = findfirst(p -> tbl.i[p] == 1 && tbl.j[p] == 4, eachindex(tbl.i))
@assert p14 !== nothing
@assert tbl.lj_scale[p14] == 0.5
@assert isapprox(tbl.coul_scale[p14], 1 / 1.2; rtol = 0.0)
plain = findfirst(p -> tbl.i[p] == 2 && tbl.j[p] == 5, eachindex(tbl.i))
@assert plain !== nothing
@assert tbl.lj_scale[plain] == 1.0 && tbl.coul_scale[plain] == 1.0
@assert tbl.sigma[p14] == 0.5 * (0.34 + 0.34)
@assert tbl.epsilon[p14] == sqrt(0.36 * 0.36)
@assert tbl.qiqj[p14] == 0.40 * -0.30

Ud, Fd = SC.energy_forces(rsd, SIX_Q)
@assert isfinite(Ud) && all(isfinite, Fd)

# explicit LennardJonesPair + CoulombPair sum over the same rows
explicit = F.AbstractFactor[]
for p in eachindex(tbl.i)
    push!(explicit, F.LennardJonesPair(tbl.i[p], tbl.j[p], tbl.epsilon[p],
                                       tbl.sigma[p]; mode = :none,
                                       lj_scale = tbl.lj_scale[p]))
    push!(explicit, F.CoulombPair(tbl.i[p], tbl.j[p],
                                  ns_direct.atoms[tbl.i[p]].charge,
                                  ns_direct.atoms[tbl.j[p]].charge;
                                  coulomb_scale = tbl.coul_scale[p]))
end
bond_only = [f for f in rsd.factors if f isa F.HarmonicBond ||
                                       f isa F.HarmonicAngle ||
                                       f isa CF.UreyBradley ||
                                       f isa CF.MultiPeriodicDihedral ||
                                       f isa CF.HarmonicImproper]
Ue, Fe = F.total_energy_forces(vcat(bond_only, explicit), SIX_Q,
                               F.Context(T.NoBox(), rsd.units, 6))
df_pair = maxabs(Fe .- Fd)
@assert abs(Ue - Ud) <= 1e-12 "direct table vs explicit pair sum: $(abs(Ue - Ud))"
# The force sum uses a different accumulation order (one factor per pair instead of
# one table), so it is compared relative to the force scale, not bitwise.
@assert df_pair <= 1e-12 * max(1.0, maxabs(Fd)) "force mismatch $df_pair"

# central-difference force check at h = 1e-6 on the whole boxless system
function fd_force_error(rs, q0; h = 1e-6)
    F0 = SC.forces(rs, q0)
    q = copy(q0)
    err = 0.0
    n = 0
    for j in axes(q, 2), a in 1:3
        up = q[a, j]
        q[a, j] = up + h
        uplus = SC.energy(rs, q)
        q[a, j] = up - h
        uminus = SC.energy(rs, q)
        q[a, j] = up
        err = max(err, abs(-(uplus - uminus) / (2h) - F0[a, j]))
        n += 1
    end
    return err, n
end

h = 1e-6
fd_err, neval = fd_force_error(rsd, SIX_Q; h = h)
scale = max(1.0, maxabs(Fd))
@assert fd_err <= 1e-7 * scale "FD force error $fd_err exceeds 1e-7 * $scale"
@assert ce.dihedral_1 != 0.0
dihedral_u = SC.component_energy(rsd, SIX_Q, :dihedral_1)
@assert dihedral_u > 0.0 && isfinite(dihedral_u)
@assert SC.component_energy(rsd, SIX_Q, :improper_1) != 0.0
@assert SC.component_energy(rsd, SIX_Q, :urey_1) != 0.0
println("  direct :none   U = $Ud over $npairs explicit pairs (exclusion and 1-4 scaled)")
println("  direct vs explicit Pair sum: dU = $(abs(Ue - Ud)), max|dF| = $(maxabs(Fe .- Fd))")
println("  FD at h=$h: max|F_fd - F| = $fd_err over $neval components; tolerance " *
        "$(1e-7 * scale)")

# ---------------------------------------------------------------------------
# 5. triclinic PME build + FD stress + typed refusals
# ---------------------------------------------------------------------------

let
    p_tric_lfs = C.default_protocol(ns_tric; lj_mode = :lfs)
    rs_tric_pme = C.build_resolved(ns_tric; protocol = p_tric_lfs)
    @assert rs_tric_pme.box isa T.TriclinicBox
    @assert rs_tric_pme.box.H == tric.vectors_nm
    @assert :box_derivative in RS.engine_capabilities(rs_tric_pme)
    U_tp, F_tp = SC.energy_forces(rs_tric_pme, SIX_Q)
    @assert isfinite(U_tp) && all(isfinite, F_tp)
    G_tp = SC.box_derivative(rs_tric_pme, SIX_Q)
    S_tp = SC.fractional_coordinates(rs_tric_pme, SIX_Q)
    eps_tp = 1e-6
    max_tp_fd = 0.0
    for a in 1:3, b in 1:3
        Eab = zeros(3, 3); Eab[a, b] = eps_tp
        Hp = tric.vectors_nm .+ Eab
        Hm = tric.vectors_nm .- Eab
        rsp = RS.restart_system(rs_tric_pme; box = T.TriclinicBox(Hp))
        rsm = RS.restart_system(rs_tric_pme; box = T.TriclinicBox(Hm))
        up = SC.energy(rsp, Hp * S_tp)
        um = SC.energy(rsm, Hm * S_tp)
        max_tp_fd = max(max_tp_fd, abs(G_tp[a, b] - (up - um) / (2 * eps_tp)))
    end
    @assert max_tp_fd <= 1e-6 * max(1.0, maximum(abs, G_tp)) "triclinic PME+bonded FD err=$(max_tp_fd)"
    println("  triclinic :pme build: U=$U_tp, 9-entry box_derivative FD_err=$max_tp_fd")
end

# Small triclinic cell with cutoff >= 0.5 * min(width) is refused as :protocol
tric_small = C.NormalizedBox(:triclinic, (0.0, 0.0, 0.0),
                             [2.0 0.0 0.0; 0.3 2.0 0.0; 0.1 0.2 2.0])
ns_tric_small = six_system(; box = tric_small)
assert_chem_error(:protocol, () -> C.default_protocol(ns_tric_small), ["cutoff_nm", "triclinic widths"])
assert_chem_error(:protocol, () -> C.build_resolved(ns_tric_small; protocol = C.SimulationProtocol(nonbonded = :pme, cutoff_nm = 1.0)), ["cutoff_nm"])

# CMAP: schema-valid record (8 atoms, 4x4 grid) refused by the builder
cmap_atoms = vcat(six_atoms(),
                  [C.NormalizedAtom("C4", 12.011, 0.0, "C", 3, "LIG", 0.34, 0.36),
                   C.NormalizedAtom("H2", 1.008, 0.0, "H", 3, "LIG", 0.24, 0.08)])
ns_cmap = C.NormalizedSystem(v"1.0.0", :charmm, C.CHEM_UNITS_NAME,
                             C.ImportProvenance("fixture-cmap.json", "unit-test force field",
                                                "b"^64, "", "", "test", "1.0.0", ""),
                             cmap_atoms,
                             C.NormalizedBox(:orthorhombic, (3.0, 3.2, 3.4), zeros(3, 3)),
                             C.BondTerm[], C.AngleTerm[], C.UreyBradleyTerm[],
                             C.DihedralTerm[], C.ImproperTerm[],
                             [C.CmapTerm(ntuple(i -> i, 8), 4, zeros(16))],
                             NTuple{2,Int}[], C.Pair14Term[], C.C64Term[], C.VirtualSiteTerm[],
                             C.ConstraintTerm[], C.DiagnosticRecord[], 0.0)
C.validate_normalized!(ns_cmap)
rs_cmap = C.build_resolved(ns_cmap)
@assert any(f -> f isa FoundationCore.ChemistryCMAP.CMAPCorrection, rs_cmap.factors)
@assert any(==(:cmap_1), rs_cmap.component_names)
println("  cmaps -> CMAPCorrection factor (zero map energy)")

assert_chem_error(:protocol, () -> C.build_resolved(ns;
                    protocol = C.SimulationProtocol(nonbonded = :direct)),
                  ["nonbonded = :direct", "orthorhombic"])
assert_chem_error(:protocol, () -> C.build_resolved(ns_direct;
                    protocol = C.SimulationProtocol(nonbonded = :pme)),
                  ["nonbonded = :pme", "orthorhombic"])

# virtual sites: represented, not executed
vs_ns = C.NormalizedSystem(v"1.0.0", :amber, C.CHEM_UNITS_NAME,
                           C.ImportProvenance("fixture-vs.json", "unit-test force field",
                                              "c"^64, "", "", "test", "1.0.0", ""),
                           [C.NormalizedAtom("O", 15.999, -0.8, "O", 1, "WAT", 0.30, 0.64),
                            C.NormalizedAtom("H1", 1.008, 0.4, "H", 1, "WAT", 0.24, 0.08),
                            C.NormalizedAtom("H2", 1.008, 0.4, "H", 1, "WAT", 0.24, 0.08),
                            C.NormalizedAtom("M", 0.0, 0.0, "O", 1, "WAT", 0.24, 0.08)],
                           C.NormalizedBox(:orthorhombic, (3.0, 3.0, 3.0), zeros(3, 3)),
                           [C.BondTerm(1, 2, 450_000.0, 0.09572),
                            C.BondTerm(1, 3, 450_000.0, 0.09572)],
                           [C.AngleTerm(2, 1, 3, 550.0, 1.8242)],
                           C.UreyBradleyTerm[], C.DihedralTerm[], C.ImproperTerm[],
                           C.CmapTerm[], NTuple{2,Int}[], C.Pair14Term[], C.C64Term[],
                           [C.VirtualSiteTerm(:linear, 4, [2, 3], [0.5, 0.5])],
                           C.ConstraintTerm[], C.DiagnosticRecord[], 0.0)
C.validate_normalized!(vs_ns)
rsvs = C.build_resolved(vs_ns; protocol = C.default_protocol(vs_ns; water_model = :rigid))
@assert RS.n_atoms(rsvs) == 4
@assert RS.virtual_site_indices(rsvs) == Set([4])
@assert rsvs.atoms.masses[4] == 0.0
@assert length(rsvs.atoms.virtual_sites) == 1
@assert rsvs.atoms.virtual_sites[1].kind === :linear
@assert rsvs.atoms.virtual_sites[1].parents == [2, 3]
@assert rsvs.constraints !== nothing       # rigid water bonds became constraints
ev = caught_throw(() -> SC.energy(rsvs, zeros(3, 4)))
@assert ev isa RS.UnsupportedCapability && ev.capability === :virtual_sites
println("  virtual site: record preserved (site 4, mass 0), SC.energy refuses " *
        ":virtual_sites")

# ---------------------------------------------------------------------------
# 6. water_model: flexible keeps every bond, rigid omits only water bonds
# ---------------------------------------------------------------------------

wats = [C.NormalizedAtom("O", 15.999, -0.83, "O", 1, "WAT", 0.31506, 0.6364),
        C.NormalizedAtom("H1", 1.008, 0.415, "H", 1, "WAT", 0.0, 0.0),
        C.NormalizedAtom("H2", 1.008, 0.415, "H", 1, "WAT", 0.0, 0.0),
        C.NormalizedAtom("C1", 12.011, -0.10, "C", 2, "LIG", 0.34, 0.36),
        C.NormalizedAtom("H3", 1.008, 0.10, "H", 2, "LIG", 0.24, 0.08),
        C.NormalizedAtom("C2", 12.011, 0.00, "C", 2, "LIG", 0.34, 0.36)]
wbox = C.NormalizedBox(:orthorhombic, (3.0, 3.0, 3.0), zeros(3, 3))
wbonds = [C.BondTerm(1, 2, 450_000.0, 0.09572),
          C.BondTerm(1, 3, 450_000.0, 0.09572),
          C.BondTerm(4, 5, 300_000.0, 0.109),
          C.BondTerm(4, 6, 300_000.0, 0.152)]
wangs = [C.AngleTerm(2, 1, 3, 550.0, 1.8242), C.AngleTerm(5, 4, 6, 500.0, 1.9)]
wnet = sum(a.charge for a in wats)
w_ns = C.NormalizedSystem(v"1.0.0", :amber, C.CHEM_UNITS_NAME,
                          C.ImportProvenance("fixture-water.json", "unit-test force field",
                                             "d"^64, "", "", "test", "1.0.0", ""),
                          wats, wbox, wbonds, wangs, C.UreyBradleyTerm[],
                          C.DihedralTerm[], C.ImproperTerm[], C.CmapTerm[],
                          [(1, 2), (1, 3)], C.Pair14Term[], C.C64Term[], C.VirtualSiteTerm[],
                          C.ConstraintTerm[], C.DiagnosticRecord[], wnet)
C.validate_normalized!(w_ns)
rs_flex = C.build_resolved(w_ns)   # default :flexible
rs_rigid = C.build_resolved(w_ns; protocol = C.default_protocol(w_ns; water_model = :rigid))
flex_bonds = [f for f in rs_flex.factors if f isa F.HarmonicBond]
rigid_bonds = [f for f in rs_rigid.factors if f isa F.HarmonicBond]
@assert length(flex_bonds) == 4
@assert rs_flex.constraints === nothing
@assert length(rigid_bonds) == 2
@assert all(b -> !(b.i in (1, 2, 3) && b.j in (1, 2, 3)), rigid_bonds)
@assert Set((b.i, b.j) for b in rigid_bonds) == Set([(4, 5), (4, 6)])
@assert rs_rigid.component_names == [:bond_1, :bond_2, :angle_1, :angle_2, :lj, :pme]
@assert rs_rigid.constraints isa CN.CoupledConstraintSet
@assert CN.natoms(rs_rigid.constraints) == 3
@assert length(rs_rigid.constraints.components) == 1
comp = rs_rigid.constraints.components[1]
@assert comp.atoms == [1, 2, 3]
@assert sort([d for (_, _, d) in comp.edges]) == [0.09572, 0.09572]
@assert rs_flex.component_names[1:4] == [:bond_1, :bond_2, :bond_3, :bond_4]
println("  water: flexible $(length(flex_bonds)) bonds / no constraints; " *
        "rigid $(length(rigid_bonds)) bonds / $(length(comp.edges)) constraints at " *
        "$(sort([d for (_, _, d) in comp.edges])) nm")

# ---------------------------------------------------------------------------
# 7. build report
# ---------------------------------------------------------------------------

rep = C.build_report(ns)
@assert rep.rs isa RS.ResolvedSystem
@assert rep.protocol.nonbonded === :pme && rep.protocol.cutoff_nm == 1.0
@assert rep.summary.n_atoms == 6
@assert rep.summary.n_pair14 == 1
@assert rep.factor_counts.total == length(rep.rs.factors)
@assert rep.factor_counts.bonds == 5
@assert rep.factor_counts.angles == 4
@assert rep.factor_counts.urey_bradleys == 1
@assert rep.factor_counts.dihedrals == 1
@assert rep.factor_counts.impropers == 1
@assert rep.factor_counts.lj == 1
@assert rep.factor_counts.pme == 1
@assert rep.factor_counts.nonbonded == 0
@assert length(rep.warnings) == 1
@assert occursin(":trunc", rep.warnings[1]) && occursin("integrator", rep.warnings[1])
@assert SC.energy(rep.rs, SIX_Q) == U       # same factor objects, same numbers

rep_lfs = C.build_report(ns; protocol = C.default_protocol(ns; lj_mode = :lfs,
                                                           background = :none))
@assert length(rep_lfs.warnings) == 1
@assert occursin("background", rep_lfs.warnings[1])
@assert rep_lfs.rs.factors[13].mode === :lfs
@assert SC.energy(rep_lfs.rs, SIX_Q) != U   # a different potential, not a relabel
rep_direct = C.build_report(no_box_six())
@assert rep_direct.factor_counts.nonbonded == 1
@assert rep_direct.factor_counts.lj == 0 && rep_direct.factor_counts.pme == 0
@assert rep_direct.summary.box_kind === :none
println("  report: $(rep.factor_counts.total) factors, $(length(rep.warnings)) " *
        "warning(s) for :trunc; lfs/background report has " *
        "$(length(rep_lfs.warnings)) warning(s)")

# a triclinic record produces a valid build report under default :pme
rep_tric_pme = C.build_report(ns_tric)
@assert rep_tric_pme.rs.box isa T.TriclinicBox
@assert rep_tric_pme.factor_counts.pme == 1 && rep_tric_pme.factor_counts.lj == 1

# ---------------------------------------------------------------------------
# 8. Phase 1-3 seams: triclinic local build (:none), FD stress, schema seams
# ---------------------------------------------------------------------------

# Triclinic bonded/local build with explicit nonbonded = :none preserves the
# full 3x3 lattice matrix and exposes analytic box_derivative across all 9 entries.
let
    p_tric_none = C.default_protocol(ns_tric; nonbonded = :none)
    rs_tric = C.build_resolved(ns_tric; protocol = p_tric_none)
    @assert rs_tric.box isa T.TriclinicBox
    @assert rs_tric.box.H == tric.vectors_nm
    @assert :box_derivative in RS.engine_capabilities(rs_tric)
    U_tr, F_tr = SC.energy_forces(rs_tric, SIX_Q)
    ce_tr = SC.component_energies(rs_tric, SIX_Q)
    @assert isapprox(sum(values(ce_tr)), U_tr; rtol = 1e-13)
    G_tr = SC.box_derivative(rs_tric, SIX_Q)
    S_tr = SC.fractional_coordinates(rs_tric, SIX_Q)
    eps_tr = 1e-6
    max_tr_fd = 0.0
    for a in 1:3, b in 1:3
        Eab = zeros(3, 3); Eab[a, b] = eps_tr
        Hp = tric.vectors_nm .+ Eab
        Hm = tric.vectors_nm .- Eab
        rsp = RS.restart_system(rs_tric; box = T.TriclinicBox(Hp))
        rsm = RS.restart_system(rs_tric; box = T.TriclinicBox(Hm))
        up = SC.energy(rsp, Hp * S_tr)
        um = SC.energy(rsm, Hm * S_tr)
        max_tr_fd = max(max_tr_fd, abs(G_tr[a, b] - (up - um) / (2 * eps_tr)))
    end
    @assert max_tr_fd <= 1e-7 * max(1.0, maximum(abs, G_tr))
    rep_tric = C.build_report(ns_tric; protocol = p_tric_none)
    @assert any(w -> occursin("nonbonded = :none", w), rep_tric.warnings)
    println("  triclinic :none local build: U=$U_tr, box_derivative FD_err=$max_tr_fd")
end

# Schema/builder seam checks:
# (a) CMAP resolution = 2 passes validate_normalized! but must raise typed ChemistryError in build_resolved
ns_cmap2 = C.NormalizedSystem(v"1.0.0", :charmm, C.CHEM_UNITS_NAME,
    C.ImportProvenance("fixture-cmap2.json", "unit-test force field",
                       "b"^64, "", "", "test", "1.0.0", ""),
    cmap_atoms, C.NormalizedBox(:orthorhombic, (3.0, 3.2, 3.4), zeros(3, 3)),
    C.BondTerm[], C.AngleTerm[], C.UreyBradleyTerm[], C.DihedralTerm[],
    C.ImproperTerm[], [C.CmapTerm(ntuple(i -> i, 8), 2, zeros(4))],
    NTuple{2,Int}[], C.Pair14Term[], C.C64Term[], C.VirtualSiteTerm[],
    C.ConstraintTerm[], C.DiagnosticRecord[], 0.0)
C.validate_normalized!(ns_cmap2)
assert_chem_error(:parameter, () -> C.build_resolved(ns_cmap2), ["cmaps[1].resolution", ">= 3"])

# (b) Zero sigma with nonzero epsilon must be refused in :direct as well as :pme
bad_direct = C.NormalizedSystem(v"1.0.0", :amber, C.CHEM_UNITS_NAME,
    bad_ns.provenance, bad_atoms, C.NormalizedBox(),
    C.BondTerm[], C.AngleTerm[], C.UreyBradleyTerm[], C.DihedralTerm[],
    C.ImproperTerm[], C.CmapTerm[], NTuple{2,Int}[], C.Pair14Term[],
    C.C64Term[], C.VirtualSiteTerm[], C.ConstraintTerm[], C.DiagnosticRecord[], 0.0)
assert_chem_error(:parameter, () -> C.build_resolved(bad_direct), ["atoms[1].sigma_nm"])

# (c) Reversed duplicate exclusions (1,2) and (2,1) raise typed ChemistryError(:duplicate)
dup_excl_ns = six_system(; exclusions = [(1, 2), (2, 1), (2, 3), (3, 4), (1, 3), (2, 4)])
assert_chem_error(:duplicate, () -> C.build_resolved(dup_excl_ns), ["exclusions"])

# (d) Conflicting constraint distances between rigid water bond and declared ConstraintTerm
conflict_c_ns = C.NormalizedSystem(v"1.0.0", :amber, C.CHEM_UNITS_NAME,
    w_ns.provenance, wats, wbox, wbonds, wangs, C.UreyBradleyTerm[],
    C.DihedralTerm[], C.ImproperTerm[], C.CmapTerm[], [(1, 2), (1, 3)],
    C.Pair14Term[], C.C64Term[], C.VirtualSiteTerm[],
    [C.ConstraintTerm(1, 2, 0.120)], C.DiagnosticRecord[], wnet)
assert_chem_error(:parameter, () -> C.build_resolved(conflict_c_ns;
    protocol = C.default_protocol(conflict_c_ns; water_model = :rigid)),
    ["constraints", "0.09572", "0.12"])

# (e) factor_counts includes cmaps and lj146; lj146 warning in :pme + :lfs
rep_cmap = C.build_report(ns_cmap)
@assert rep_cmap.factor_counts.cmaps == 1
@assert rep_cmap.factor_counts.lj146 == 0
ns_146 = C.NormalizedSystem(v"1.0.0", :amber, C.CHEM_UNITS_NAME,
    ns.provenance, ns.atoms, ns.box, ns.bonds, ns.angles, ns.urey_bradleys,
    ns.dihedrals, ns.impropers, ns.cmaps, ns.exclusions, ns.pair14,
    [C.C64Term(1, 5, 0.05)], ns.virtual_sites, ns.constraints,
    ns.input_diagnostics, ns.net_charge)
rep_146 = C.build_report(ns_146; protocol = C.default_protocol(ns_146; lj_mode = :lfs))
@assert rep_146.factor_counts.lj146 == 1
@assert !F.force_consistent(rep_146.rs.factors[end])
@assert any(w -> occursin("lj146", w) && occursin(":trunc", w), rep_146.warnings)
println("  schema/build seams (CMAP res, :direct zero-sigma, reversed excl, constraint conflict, factor_counts): OK")

println("test_chemistry_build.jl: all builder checks passed")
