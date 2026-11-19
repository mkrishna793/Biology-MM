# Resolution-contract suite: `ResolvedSystem` (specification) and the parts of
# `SystemContract` that gate execution of a resolved record.
#
# Run standalone:
#
#     cd julia
#     julia --project=. test/test_resolved_system.jl
#
# This file is assertion-driven and runs entirely at include time (no `main`),
# matching the other FoundationCore suites. It is *not* a mirror of the
# implementation: every failure test asserts the symbolic error code AND that the
# message names the offending atom / pair / factor, because a resolution failure
# is a force-field transcription error and the index is the actionable part.
#
# No numerical tolerances are needed here (there is no energy evaluation except
# one exact zero at a bond's equilibrium length); the only floating-point
# comparisons are exact ownership/round-trip checks.

using FoundationCore

const T = FoundationCore.Types
const F = FoundationCore.Factors
const RS = FoundationCore.ResolvedSystem
const SC = FoundationCore.SystemContract
const SL = FoundationCore.StreamedLJ
const PM = FoundationCore.PME

# The constraint set has no exported alias in the interface; qualify it locally.
const CN = FoundationCore.Constraints

# ---------------------------------------------------------------------------
# helpers: catch a throw, return it, and assert on code/capability + message
# ---------------------------------------------------------------------------

function caught_throw(f::Function)
    try
        f()
    catch e
        return e
    end
    return nothing
end

"Assert that `f()` throws `ResolvedSystemError` with `code`; optionally require
message substrings so the diagnostic stays actionable."
function assert_rs_error(code::Symbol, f::Function, needles::Vector{String} = String[])
    e = caught_throw(f)
    e isa RS.ResolvedSystemError ||
        error("expected ResolvedSystemError[$code], got $(repr(e))")
    @assert e.code === code "expected error code $code, got $(e.code): $(e.msg)"
    for needle in needles
        @assert occursin(needle, e.msg) "message for $code must mention " *
                                        "$(repr(needle)); got: $(e.msg)"
    end
    return e
end

"Assert that `f()` throws `UnsupportedCapability` with `capability`."
function assert_unsupported(capability::Symbol, f::Function,
                            needles::Vector{String} = String[])
    e = caught_throw(f)
    e isa RS.UnsupportedCapability ||
        error("expected UnsupportedCapability[$(capability)], got $(repr(e))")
    @assert e.capability === capability "expected capability $capability, got " *
                                        "$(e.capability): $(e.reason)"
    for needle in needles
        @assert occursin(needle, e.reason) "refusal for $capability must mention " *
                                           "$(repr(needle)); got: $(e.reason)"
    end
    return e
end

const test_ff = RS.Provenance(source = "fixture/model.pdb", force_field = "test-ff")
const ljbox = T.OrthorhombicBox((8.0, 8.0, 8.0))

# ---------------------------------------------------------------------------
# 1. valid construction: schema, default names, counts, capabilities
# ---------------------------------------------------------------------------

bond3 = F.HarmonicBond(1, 2, 500.0, 1.09)
angle3 = F.HarmonicAngle(1, 2, 3, 40.0, 1.91)
lj3 = F.LennardJonesPair(1, 3, 0.2, 3.1; mode = :lfs, rc = 3.0)

rs_local = RS.ResolvedSystem(;
    masses = [12.0, 1.008, 1.008],
    charges = [0.2, -0.1, -0.1],
    box = T.OrthorhombicBox((4.0, 5.0, 6.0)),
    factors = [bond3, angle3, lj3],
    provenance = test_ff,
)

@assert RS.schema_version(rs_local) == RS.RESOLVED_SCHEMA_VERSION
@assert rs_local.schema == v"1.0.0"
@assert RS.n_atoms(rs_local) == 3
@assert rs_local.atoms.names == [:X, :X, :X]            # default names
@assert rs_local.component_names == [:component_1, :component_2, :component_3]
@assert rs_local.declared == Set([:energy, :cartesian_forces, :component_energies])
@assert RS.factor_support_atoms(bond3) == [1, 2]
@assert RS.virtual_site_indices(rs_local) == Set{Int}()
@assert RS.validate_resolved!(rs_local) === rs_local

caps = RS.engine_capabilities(rs_local)
@assert caps isa Set{Symbol}
@assert all(c in caps for c in (:energy, :cartesian_forces, :component_energies))
@assert :box_derivative in caps            # local factors + orthorhombic box
@assert occursin("ResolvedSystem(", sprint(show, rs_local))
# `print` renders a Symbol without its leading colon, so the header is
# "ResolvedSystemError[shape]" (the interface spells the format this way).
@assert startswith(sprint(showerror, RS.ResolvedSystemError(:shape, "x")),
                   "ResolvedSystemError[shape]")

# The file container is copied, the factor objects are shared, and names accept
# Strings as well as Symbols (one Symbol per atom is what is stored).
f_names = [F.HarmonicBond(1, 2, 100.0, 1.0)]
rs_names = RS.ResolvedSystem(; masses = [1.0, 1.0], charges = [0.0, 0.0],
                             names = ["Ow", "H1"], factors = f_names)
@assert rs_names.atoms.names == [:Ow, :H1]
@assert rs_names.factors !== f_names
@assert rs_names.factors[1] === f_names[1]

# A NoBox system cannot supply a lattice derivative, so the capability is absent.
rs_nobox = RS.ResolvedSystem(; masses = [1.0, 1.0], charges = [0.0, 0.0],
                             factors = [F.HarmonicBond(1, 2, 100.0, 1.0)])
@assert !(:box_derivative in RS.engine_capabilities(rs_nobox))

# A triclinic cell with local factors only still supports the box derivative;
# the periodic backends are the things that refuse it (tested in section 6).
tribox = T.TriclinicBox([6.0 0.5 0.0; 0.0 6.0 0.3; 0.0 0.0 6.0])
rs_tri_local = RS.ResolvedSystem(; masses = [1.0, 1.0], charges = [0.0, 0.0],
                                 box = tribox,
                                 factors = [F.HarmonicBond(1, 2, 100.0, 1.0)])
@assert :box_derivative in RS.engine_capabilities(rs_tri_local)

# ---------------------------------------------------------------------------
# 2. schema
# ---------------------------------------------------------------------------

assert_rs_error(:schema, () -> RS.ResolvedSystem(; masses = [1.0], charges = [0.0],
                                                 schema = v"2.0.0"),
                ["1.0.0", "not supported"])

assert_rs_error(:schema, () -> RS.ResolvedSystem(; masses = [1.0], charges = [0.0],
                                                 schema = "1.0.0"),
                ["VersionNumber"])

# ---------------------------------------------------------------------------
# 3. shape / mass / charge / units / provenance
# ---------------------------------------------------------------------------

assert_rs_error(:shape, () -> RS.ResolvedSystem(; masses = [1.0, 2.0],
                                                charges = [0.0]),
                ["charges", "1", "masses", "2"])

assert_rs_error(:shape, () -> RS.ResolvedSystem(; masses = Float64[],
                                                charges = Float64[]),
                ["at least one atom"])

assert_rs_error(:shape, () -> RS.ResolvedSystem(; masses = [1.0], charges = [0.0],
                                                names = [:A, :B]),
                ["names has length 2", "masses has length 1"])

assert_rs_error(:mass, () -> RS.ResolvedSystem(; masses = [NaN], charges = [0.0]),
                ["mass[1]", "not finite"])

assert_rs_error(:mass, () -> RS.ResolvedSystem(; masses = [1.0, 0.0],
                                               charges = [0.0, 0.0]),
                ["atom 2", "VirtualSite"])

assert_rs_error(:charge, () -> RS.ResolvedSystem(; masses = [1.0],
                                                 charges = [Inf]),
                ["charge[1]", "not finite"])

bad_units = T.Units("broken", 1.0, 1.0, 1.0, 1.0, 1.0, NaN)
assert_rs_error(:units, () -> RS.ResolvedSystem(; masses = [1.0], charges = [0.0],
                                                units = bad_units),
                ["units.boltzmann", "strictly positive"])

# A zero Coulomb prefactor is only an error when a Coulomb factor exists, because
# otherwise every Coulomb energy would silently be zero.
zero_coulomb_units = T.Units("zero-coulomb", 1.0, 1.0, 1.0, 1.0, 0.0, 1.0)
assert_rs_error(:units, () -> RS.ResolvedSystem(;
                        masses = [1.0, 1.0], charges = [0.0, 0.0],
                        units = zero_coulomb_units,
                        factors = [F.CoulombPair(1, 2, 0.0, 0.0)]),
                ["coulomb_prefactor", "CoulombPair"])

assert_rs_error(:provenance, () -> RS.ResolvedSystem(;
                        masses = [1.0], charges = [0.0],
                        provenance = RS.Provenance(source = "   ",
                                                   force_field = "ff")),
                ["provenance.source", "not auditable"])

# ---------------------------------------------------------------------------
# 4. component names, factor indices, duplicate interactions, capabilities
# ---------------------------------------------------------------------------

assert_rs_error(:component_name, () -> RS.ResolvedSystem(;
                        masses = [1.0, 1.0, 1.0], charges = zeros(3),
                        factors = [F.HarmonicBond(1, 2, 100.0, 1.0),
                                   F.HarmonicBond(2, 3, 100.0, 1.0)],
                        component_names = [:stretch, :stretch]),
                [":stretch", "factors 1 and 2"])

assert_rs_error(:component_name, () -> RS.ResolvedSystem(;
                        masses = [1.0, 1.0, 1.0], charges = zeros(3),
                        factors = [F.HarmonicBond(1, 2, 100.0, 1.0),
                                   F.HarmonicBond(2, 3, 100.0, 1.0)],
                        component_names = [:stretch]),
                ["component_names has 1 entries", "2 factors"])

assert_rs_error(:factor_index, () -> RS.ResolvedSystem(;
                        masses = [1.0, 1.0], charges = zeros(2),
                        factors = [F.HarmonicBond(1, 7, 100.0, 1.0)]),
                ["factor 1", "atom 7", "1:2"])

# (1,2) and (2,1) are the same physical pair: sorting the support makes the
# reversed spelling collide instead of evaluating twice.
assert_rs_error(:duplicate_interaction, () -> RS.ResolvedSystem(;
                        masses = [1.0, 1.0], charges = zeros(2),
                        factors = [F.HarmonicBond(1, 2, 100.0, 1.0),
                                   F.HarmonicBond(2, 1, 200.0, 1.0)]),
                ["factors 1 and 2", "(1, 2)", "counted twice"])

assert_rs_error(:unknown_capability, () -> RS.ResolvedSystem(;
                        masses = [1.0], charges = [0.0],
                        declared = [:energy, :pressure]),
                [":pressure", "box_derivative"])

# ---------------------------------------------------------------------------
# 5. charge mismatch: per-atom charge data duplicated in a pair factor
# ---------------------------------------------------------------------------

assert_rs_error(:charge_mismatch, () -> RS.ResolvedSystem(;
                        masses = [1.0, 1.0], charges = [0.5, -0.5],
                        factors = [F.CoulombPair(1, 2, 0.4, -0.5)]),
                ["CoulombPair", "atom 1", "0.4"])

# Positive control: a consistent pair factor resolves.
rs_coulomb = RS.ResolvedSystem(; masses = [1.0, 1.0], charges = [0.5, -0.5],
                               factors = [F.CoulombPair(1, 2, 0.5, -0.5)])
@assert length(rs_coulomb.factors) == 1

# ---------------------------------------------------------------------------
# 6. streamed-LJ topology agreement and periodic-backend geometry
# ---------------------------------------------------------------------------

eps3 = [1.0, 1.0, 1.0]
sig3 = [1.0, 1.0, 1.0]

topo_excl = T.ChemicalTopology(exclusions = Set([(1, 2)]))
ftopo_excl = SL.PairTopology(excluded = [(1, 2)])
f_lj_excl = SL.StreamedLJFactor(3, eps3, sig3, ljbox; rcut = 3.0,
                                topology = ftopo_excl)
rs_lj_excl = RS.ResolvedSystem(; masses = [1.0, 1.0, 1.0], charges = zeros(3),
                               box = ljbox, topology = topo_excl,
                               factors = [f_lj_excl],
                               provenance = RS.Provenance(source = "lj/fixture",
                                                          force_field = "lj"))
@assert :box_derivative in RS.engine_capabilities(rs_lj_excl)   # :lfs default mode

# Silently dropped exclusion: ChemicalTopology excludes the pair but the factor's
# explicit topology still applies weight 0.5.
topo_scaled = T.ChemicalTopology(scales = Dict((1, 2) => (0.5, 0.5)))
ftopo_scaled = SL.PairTopology(scaled = [(1, 2, 0.5)])
f_lj_scaled = SL.StreamedLJFactor(3, eps3, sig3, ljbox; rcut = 3.0,
                                  topology = ftopo_scaled)
assert_rs_error(:conflicting_pair_rules, () -> RS.ResolvedSystem(;
                        masses = [1.0, 1.0, 1.0], charges = zeros(3), box = ljbox,
                        topology = topo_excl, factors = [f_lj_scaled],
                        provenance = test_ff),
                ["(1, 2)", "0.0", "0.5"])

# Silently applied exclusion: the topology scales the pair but the factor
# excludes it.
assert_rs_error(:conflicting_pair_rules, () -> RS.ResolvedSystem(;
                        masses = [1.0, 1.0, 1.0], charges = zeros(3), box = ljbox,
                        topology = topo_scaled, factors = [f_lj_excl],
                        provenance = test_ff),
                ["(1, 2)", "0.5", "0.0"])

# Positive control: the two spellings agree on the 1-4 scale.
rs_lj_scaled = RS.ResolvedSystem(; masses = [1.0, 1.0, 1.0], charges = zeros(3),
                                 box = ljbox, topology = topo_scaled,
                                 factors = [f_lj_scaled], provenance = test_ff)
@assert length(rs_lj_scaled.factors) == 1

# An orthorhombic-built backend paired with a triclinic cell is refused with :box_mismatch,
# whereas a triclinic-built StreamedLJFactor resolves cleanly on TriclinicBox.
f_lj_ortho = SL.StreamedLJFactor(3, eps3, sig3, ljbox; rcut = 3.0)
assert_rs_error(:box_mismatch, () -> RS.ResolvedSystem(;
                        masses = [1.0, 1.0, 1.0], charges = zeros(3), box = tribox,
                        factors = [f_lj_ortho], provenance = test_ff),
                ["StreamedLJFactor", "OrthorhombicBox", "TriclinicBox"])

f_lj_tri = SL.StreamedLJFactor(3, eps3, sig3, tribox; rcut = 2.0)
rs_lj_tri = RS.ResolvedSystem(; masses = [1.0, 1.0, 1.0], charges = zeros(3),
                              box = tribox, factors = [f_lj_tri],
                              declared = [:energy, :cartesian_forces, :component_energies, :box_derivative],
                              provenance = test_ff)
@assert :box_derivative in RS.engine_capabilities(rs_lj_tri)

# Mutating rcut past 0.5 * min(tribox.widths) is caught by validate_resolved!
f_lj_tri.rcut = 0.5 * minimum(tribox.widths) + 0.1
assert_rs_error(:unsupported_geometry, () -> RS.validate_resolved!(rs_lj_tri),
                ["StreamedLJFactor", "perpendicular width"])
f_lj_tri.rcut = 2.0
@assert RS.validate_resolved!(rs_lj_tri) === rs_lj_tri

# ---------------------------------------------------------------------------
# 7. PME: charge agreement, box_derivative support, triclinic & Ewald refusal
# ---------------------------------------------------------------------------

# 4 atoms, net charge +1, small cell. `rcut < min(L)/2` is required strictly, so
# L = 8.0 with rcut = 3.0 (not L = 6.0, which PMEPlan refuses).
qpme = [0.5 2.0 3.5 5.0;
        0.5 2.5 4.0 1.5;
        0.5 1.5 2.5 3.5]
zpme = [0.5, 0.5, 0.5, -0.5]
pme_box = T.OrthorhombicBox((8.0, 8.0, 8.0))
f_pme = PM.PMEFactor(qpme, zpme, pme_box; alpha = 0.35, rcut = 3.0,
                     grid = (8, 8, 8), order = 4)
rs_pme = RS.ResolvedSystem(; masses = fill(1.0, 4), charges = zpme, box = pme_box,
                           factors = [f_pme],
                           declared = [:energy, :cartesian_forces, :component_energies, :box_derivative],
                           provenance = RS.Provenance(source = "pme/fixture",
                                                      force_field = "pme"))

caps_pme = RS.engine_capabilities(rs_pme)
@assert all(c in caps_pme for c in (:energy, :cartesian_forces, :component_energies, :box_derivative))

# Triclinic PMEFactor resolves cleanly (both from TriclinicBox and 3x3 matrix) and supports box_derivative
tri_pme_box = T.TriclinicBox([8.0 0.8 0.3; 0.0 8.2 0.4; 0.0 0.0 8.5])
f_pme_tri = PM.PMEFactor(qpme, zpme, tri_pme_box; alpha = 0.35, rcut = 3.0,
                         grid = (16, 16, 16), order = 4)
f_pme_tri_mat = PM.PMEFactor(qpme, zpme, tri_pme_box.H; alpha = 0.35, rcut = 3.0,
                             grid = (16, 16, 16), order = 4)
@assert f_pme_tri_mat.box isa T.TriclinicBox
rs_pme_tri = RS.ResolvedSystem(; masses = fill(1.0, 4), charges = zpme, box = tri_pme_box,
                               factors = [f_pme_tri],
                               declared = [:energy, :cartesian_forces, :component_energies, :box_derivative],
                               provenance = test_ff)
@assert :box_derivative in RS.engine_capabilities(rs_pme_tri)

# restart_system rebuilds triclinic periodic systems (StreamedLJFactor + PMEFactor) cleanly
tri_pme_box2 = T.TriclinicBox([8.5 0.9 0.35; 0.0 8.6 0.45; 0.0 0.0 8.9])
f_lj_pme_tri = SL.StreamedLJFactor(4, fill(0.2, 4), fill(1.0, 4), tri_pme_box; rcut = 2.5, mode = :lfs)
rs_tri_combo = RS.ResolvedSystem(; masses = fill(1.0, 4), charges = zpme, box = tri_pme_box,
                                 factors = F.AbstractFactor[f_lj_pme_tri, f_pme_tri],
                                 component_names = [:lj, :pme],
                                 declared = [:energy, :cartesian_forces, :component_energies, :box_derivative],
                                 provenance = test_ff)
rs_tri_restarted = RS.restart_system(rs_tri_combo; box = tri_pme_box2, provenance_notes = "triclinic NPT")
@assert rs_tri_restarted.box.H == tri_pme_box2.H
@assert rs_tri_restarted.factors[1] !== f_lj_pme_tri
@assert rs_tri_restarted.factors[2] !== f_pme_tri
@assert rs_tri_restarted.factors[2].box.H == tri_pme_box2.H
@assert RS.validate_resolved!(rs_tri_restarted) === rs_tri_restarted

# Direct EwaldFactor still refuses :box_derivative and TriclinicBox
f_ew = FoundationCore.Electrostatics.EwaldFactor(qpme, zpme, pme_box; alpha = 0.35, rcut = 3.0, kmax = 4)
rs_ew = RS.ResolvedSystem(; masses = fill(1.0, 4), charges = zpme, box = pme_box,
                          factors = [f_ew], provenance = test_ff)
@assert !(:box_derivative in RS.engine_capabilities(rs_ew))
assert_rs_error(:overdeclared_capability, () -> RS.ResolvedSystem(;
                        masses = fill(1.0, 4), charges = zpme, box = pme_box,
                        factors = [f_ew], provenance = test_ff,
                        declared = [:energy, :box_derivative]),
                ["box_derivative", "EwaldFactor"])
assert_rs_error(:unsupported_geometry, () -> RS.ResolvedSystem(;
                        masses = fill(1.0, 4), charges = zpme, box = tri_pme_box,
                        factors = [f_ew], provenance = test_ff),
                ["EwaldFactor", "orthorhombic-only"])

assert_rs_error(:charge_mismatch, () -> RS.ResolvedSystem(;
                        masses = fill(1.0, 4),
                        charges = [0.5, 0.5, 0.5, -0.4], box = pme_box,
                        factors = [f_pme], provenance = test_ff),
                ["PMEFactor", "charge 4", "atom 4", "-0.4"])

assert_rs_error(:box_mismatch, () -> RS.ResolvedSystem(;
                        masses = fill(1.0, 4), charges = zpme,
                        box = T.OrthorhombicBox((9.0, 9.0, 9.0)),
                        factors = [f_pme], provenance = test_ff),
                ["(8.0, 8.0, 8.0)", "(9.0, 9.0, 9.0)", "rebuild"])

# ---------------------------------------------------------------------------
# 8. virtual sites: representable, not executable
# ---------------------------------------------------------------------------

vs = RS.VirtualSite(:linear, 3, [1, 2], [0.5, 0.5])
rs_vs = RS.ResolvedSystem(;
    masses = [15.999, 1.008, 0.0],
    charges = [-0.834, 0.417, 0.417],
    factors = [F.HarmonicBond(1, 2, 100.0, 1.0)],
    virtual_sites = [vs],
    provenance = RS.Provenance(source = "water/tip4p-like", force_field = "test-ff"),
)
@assert RS.virtual_site_indices(rs_vs) == Set([3])
@assert RS.n_atoms(rs_vs) == 3                     # sites occupy a slot
@assert rs_vs.atoms.virtual_sites[1] !== vs
@assert rs_vs.atoms.virtual_sites[1].parents !== vs.parents
@assert rs_vs.atoms.virtual_sites[1].coefficients == vs.coefficients

q_vs = zeros(3, 3)
p_vs = zeros(3, 3)
assert_unsupported(:virtual_sites, () -> SC.energy(rs_vs, q_vs),
                   ["virtual site", "execution is not implemented"])
assert_unsupported(:virtual_sites, () -> SC.forces(rs_vs, q_vs),
                   ["execution is not implemented"])
assert_unsupported(:virtual_sites, () -> SC.to_atomset(rs_vs, q_vs, p_vs),
                   ["representation only"])

# A massless atom without a declaration is refused at resolution instead.
assert_rs_error(:mass, () -> RS.ResolvedSystem(; masses = [1.0, 0.0],
                                               charges = [0.0, 0.0]),
                ["atom 2", "must be declared"])

# Every virtual-site rule reports its own site and repair.
assert_rs_error(:virtual_site, () -> RS.ResolvedSystem(;
                        masses = [1.0, 1.0, 0.0], charges = zeros(3),
                        virtual_sites = [RS.VirtualSite(:tip4p, 3, [1, 2],
                                                        [0.5, 0.5])]),
                [":tip4p", "supported kinds"])

assert_rs_error(:virtual_site, () -> RS.ResolvedSystem(;
                        masses = [1.0, 1.0, 0.0], charges = zeros(3),
                        virtual_sites = [RS.VirtualSite(:linear, 3, [1, 2],
                                                        [0.6, 0.6])]),
                ["sum to 1.2", "not 1"])

assert_rs_error(:virtual_site, () -> RS.ResolvedSystem(;
                        masses = [1.0, 1.0, 0.0], charges = zeros(3),
                        virtual_sites = [RS.VirtualSite(:linear, 3, [1, 2],
                                                        [0.5, 0.5]),
                                         RS.VirtualSite(:linear, 3, [1, 2],
                                                        [0.5, 0.5])]),
                ["more than one virtual site"])

assert_rs_error(:mass, () -> RS.ResolvedSystem(;
                        masses = [1.0, 1.0, 1.0], charges = zeros(3),
                        virtual_sites = [RS.VirtualSite(:average, 3, [1, 2],
                                                        [0.5, 0.5])]),
                ["atom 3", "must be massless"])

# ---------------------------------------------------------------------------
# 9. constraints: out-of-range index, massless site, owned rebuild
# ---------------------------------------------------------------------------

cs_valid = CN.CoupledConstraintSet(2, [(1, 2, 1.0)])
rs_constrained = RS.ResolvedSystem(;
    masses = [1.0, 1.0], charges = zeros(2),
    factors = [F.HarmonicBond(1, 2, 100.0, 1.0)], constraints = cs_valid)

@assert rs_constrained.constraints !== cs_valid
@assert rs_constrained.constraints.components[1] !== cs_valid.components[1]
@assert rs_constrained.constraints.components[1].atoms == [1, 2]

# The constraint set is an owned copy: mutating the caller's component is not
# visible through the resolved record.
cs_valid.components[1].atoms[1] = 2
@assert rs_constrained.constraints.components[1].atoms == [1, 2]
@assert RS.validate_resolved!(rs_constrained) === rs_constrained

bad_cs = CN.CoupledConstraintSet([CN.ConstraintComponent([1, 5], [(1, 2, 1.0)])], 3)
assert_rs_error(:constraint_index, () -> RS.ResolvedSystem(;
                        masses = [1.0, 1.0], charges = zeros(2),
                        constraints = bad_cs),
                ["spans atom 5", "2 atoms"])

assert_rs_error(:constraint_mass, () -> RS.ResolvedSystem(;
                        masses = [1.0, 1.0, 0.0], charges = zeros(3),
                        factors = [F.HarmonicBond(1, 2, 100.0, 1.0)],
                        virtual_sites = [RS.VirtualSite(:average, 3, [1, 2],
                                                        [0.5, 0.5])],
                        constraints = CN.CoupledConstraintSet(3, [(1, 3, 1.0)])),
                ["atom 3", "massless virtual site"])

# ---------------------------------------------------------------------------
# 10. ownership, shallow copy, and post-construction corruption
# ---------------------------------------------------------------------------

m_in = [12.0, 1.0]
z_in = [0.2, -0.2]
n_in = [:C, :H]
rs_own = RS.ResolvedSystem(; masses = m_in, charges = z_in, names = n_in,
                           factors = [F.HarmonicBond(1, 2, 300.0, 1.09)])

@assert rs_own.atoms.masses !== m_in
@assert rs_own.atoms.charges !== z_in
@assert rs_own.atoms.names !== n_in

push!(m_in, 99.0)
z_in[1] = 9.0
n_in[1] = :ZZ
@assert rs_own.atoms.masses == [12.0, 1.0]
@assert rs_own.atoms.charges == [0.2, -0.2]
@assert rs_own.atoms.names == [:C, :H]

rs_shallow = copy(rs_own)
@assert rs_shallow.atoms.masses === rs_own.atoms.masses   # documented shallow copy
@assert rs_shallow.atoms.charges === rs_own.atoms.charges
@assert rs_shallow.factors !== rs_own.factors
@assert rs_shallow.factors[1] === rs_own.factors[1]       # factor objects shared

# The shared array is the documented hazard: a corrupted mass is visible through
# the shallow copy and is exactly what `validate_resolved!` is for.
rs_own.atoms.masses[1] = -1.0
@assert rs_shallow.atoms.masses[1] == -1.0
assert_rs_error(:mass, () -> RS.validate_resolved!(rs_own),
                ["atom 1", "-1.0", "strictly positive"])
rs_own.atoms.masses[1] = 12.0
@assert RS.validate_resolved!(rs_own) === rs_own

# Replacing a factor in the owned container with an out-of-range one is a
# post-construction corruption that resolution alone cannot see.
rs_own.factors[1] = F.HarmonicBond(1, 42, 300.0, 1.09)
assert_rs_error(:factor_index, () -> RS.validate_resolved!(rs_own),
                ["factor 1", "atom 42", "1:2"])

# A resized owned array is a shape violation, not a crash.
push!(rs_own.atoms.charges, 0.0)
assert_rs_error(:shape, () -> RS.validate_resolved!(rs_own),
                ["charges has length 3", "masses has length 2"])
pop!(rs_own.atoms.charges)

# Restore rs_own.factors[1] to valid factor after corruption test
rs_own.factors[1] = F.HarmonicBond(1, 2, 300.0, 1.09)
@assert RS.validate_resolved!(rs_own) === rs_own

# ---------------------------------------------------------------------------
# 11. restart semantics & immutability contract (10-point specification)
# ---------------------------------------------------------------------------

# Verify restart semantics coverage of all 10 items
sem = RS.restart_semantics()
@assert sem.topology === :invariant
@assert sem.masses === :invariant
@assert sem.parameters === :invariant
@assert sem.exclusions === :invariant
@assert sem.one_four_rules === :invariant
@assert sem.constraints === :invariant
@assert sem.virtual_sites === :invariant
@assert sem.box === :stateful_rebuild_if_changed
@assert sem.provenance === :lineage_appended
@assert sem.units === :invariant

# ResolvedSystem immutability check
@assert isimmutable(rs_own)
@assert isimmutable(rs_own.atoms)
@assert isimmutable(rs_own.provenance)

# Test restart_system with invariant box (NVT/NVE continuation)
rs_restart_nvt = RS.restart_system(rs_own; provenance_notes = "checkpoint step 5000")
@assert rs_restart_nvt.schema == rs_own.schema
@assert rs_restart_nvt.atoms.masses == rs_own.atoms.masses
@assert rs_restart_nvt.atoms.charges == rs_own.atoms.charges
@assert rs_restart_nvt.atoms.names == rs_own.atoms.names
@assert rs_restart_nvt.box == rs_own.box
@assert rs_restart_nvt.units == rs_own.units
@assert occursin("step 5000", rs_restart_nvt.provenance.notes)

# Test restart_system with box update (NPT barostat volume change)
# Periodic factors must be rebuilt for the new box
new_ljbox = T.OrthorhombicBox((8.5, 8.5, 8.5))
rs_restart_npt = RS.restart_system(rs_lj_excl; box = new_ljbox, provenance_notes = "NPT restart step 10000")
@assert rs_restart_npt.box == new_ljbox
@assert rs_restart_npt.factors[1] isa SL.StreamedLJFactor
@assert rs_restart_npt.factors[1].box == new_ljbox
@assert rs_restart_npt.factors[1] !== rs_lj_excl.factors[1]  # factor rebuilt!
@assert occursin("step 10000", rs_restart_npt.provenance.notes)
@assert RS.validate_resolved!(rs_restart_npt) === rs_restart_npt

# Test validate_restart_state helper
q_valid = [1.0 2.0; 0.0 0.0; 0.0 0.0]
p_valid = [0.1 -0.1; 0.2 -0.2; 0.0 0.0]
@assert RS.validate_restart_state(rs_own; q = q_valid, p = p_valid)

# Invalid restart states
# Non-finite coordinates throw InvalidGeometry
q_nan = copy(q_valid); q_nan[1, 1] = NaN
let err = caught_throw(() -> RS.validate_restart_state(rs_own; q = q_nan))
    @assert err isa RS.InvalidGeometry
end

# Non-finite momenta throw InvalidGeometry
p_inf = copy(p_valid); p_inf[2, 2] = Inf
let err = caught_throw(() -> RS.validate_restart_state(rs_own; p = p_inf))
    @assert err isa RS.InvalidGeometry
end

# Overlapping atoms throw InvalidGeometry
q_overlap = [1.0 1.0; 0.0 0.0; 0.0 0.0]
let err = caught_throw(() -> RS.validate_restart_state(rs_own; q = q_overlap, min_distance = 1e-3))
    @assert err isa RS.InvalidGeometry
    @assert occursin("overlapping atoms", err.msg)
end

# Constraint violation throws InvalidGeometry
q_violated_constraint = [0.0 1.5; 0.0 0.0; 0.0 0.0] # target distance is 1.0
let err = caught_throw(() -> RS.validate_restart_state(rs_constrained; q = q_violated_constraint))
    @assert err isa RS.InvalidGeometry
    @assert occursin("violates constraint", err.msg)
end

# ---------------------------------------------------------------------------
# 12. distinct typed error types: UnsupportedChemistry, InvalidGeometry, NumericalFailure
# ---------------------------------------------------------------------------

# 12a. UnsupportedChemistry
# Negative bond force constant
let err = caught_throw(() -> RS.validate_chemistry(T.ChemicalTopology(), [F.HarmonicBond(1, 2, -50.0, 1.0)]))
    @assert err isa RS.UnsupportedChemistry
    @assert occursin("invalid force constant", err.msg)
end

# Non-positive bond equilibrium distance
let err = caught_throw(() -> RS.validate_chemistry(T.ChemicalTopology(), [F.HarmonicBond(1, 2, 100.0, -1.0)]))
    @assert err isa RS.UnsupportedChemistry
    @assert occursin("non-positive equilibrium distance", err.msg)
end

# Angle out of range [0, pi]
let err = caught_throw(() -> RS.validate_chemistry(T.ChemicalTopology(), [F.HarmonicAngle(1, 2, 3, 50.0, 4.5)]))
    @assert err isa RS.UnsupportedChemistry
    @assert occursin("outside [0, π]", err.msg)
end

# Non-factor object
let err = caught_throw(() -> RS.validate_chemistry(T.ChemicalTopology(), ["not_a_factor"]))
    @assert err isa RS.UnsupportedChemistry
    @assert occursin("not an AbstractFactor", err.msg)
end

# 12b. InvalidGeometry
# Box with negative side length
let err = caught_throw(() -> T.OrthorhombicBox((-1.0, 5.0, 5.0)))
    @assert err isa RS.InvalidGeometry
    @assert occursin("box lengths must be finite and positive", err.msg)
end

# Triclinic box with negative determinant
let err = caught_throw(() -> T.TriclinicBox([-1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]))
    @assert err isa RS.InvalidGeometry
    @assert occursin("finite positive volume", err.msg)
end

# validate_geometry with coincident atoms
let err = caught_throw(() -> RS.validate_geometry(rs_own, [0.0 0.0; 0.0 0.0; 0.0 0.0]))
    @assert err isa RS.InvalidGeometry
    @assert occursin("coincident/overlapping", err.msg)
end

# 12c. NumericalFailure
# NaN in energy
let err = caught_throw(() -> SC.check_numerical_stability(NaN, zeros(3, 2)))
    @assert err isa RS.NumericalFailure
    @assert occursin("non-finite value", err.msg)
end

# Inf in forces
let err = caught_throw(() -> SC.check_numerical_stability(1.0, fill(Inf, 3, 2)))
    @assert err isa RS.NumericalFailure
    @assert occursin("non-finite values", err.msg)
end

# Positions with NaN to SystemContract
let err = caught_throw(() -> SC.energy(rs_local, fill(NaN, 3, 3)))
    @assert err isa RS.InvalidGeometry
end

# Momenta with NaN to SystemContract
let err = caught_throw(() -> SC.kinetic_energy(rs_local, fill(NaN, 3, 3)))
    @assert err isa RS.InvalidGeometry
end

# ---------------------------------------------------------------------------
# 13. clean delegation & extension of box_derivative for custom/extensible factors
# ---------------------------------------------------------------------------

struct CustomBoxFactor <: F.AbstractFactor
    i::Int
    j::Int
    k::Float64
    r0::Float64
end
F.read_support(f::CustomBoxFactor) = (f.i, f.j)
F.factor_support_size(::CustomBoxFactor) = 2

function F.evaluate!(Fout::AbstractMatrix{Float64}, f::CustomBoxFactor,
                     q::AbstractMatrix{Float64}, ctx::F.Context)
    d = T.unwrap_displacement(ctx.box, q, f.i, f.j)
    r = sqrt(T.norm2(d))
    dr = r - f.r0
    u = 0.5 * f.k * dr * dr
    c = -f.k * dr / r
    for m in 1:3
        g = c * (m == 1 ? d.x : m == 2 ? d.y : d.z)
        Fout[m, f.i] += g
        Fout[m, f.j] -= g
    end
    return u
end

# Before implementing accumulate_box_derivative!, box_derivative_support refuses it
f_custom = CustomBoxFactor(1, 2, 200.0, 1.5)
@assert RS.box_derivative_support(f_custom) !== nothing
@assert occursin("no analytic box derivative implemented", RS.box_derivative_support(f_custom))

# Now implement accumulate_box_derivative! extension point for CustomBoxFactor
function RS.accumulate_box_derivative!(G::Matrix{Float64}, f::CustomBoxFactor,
                                      q::AbstractMatrix{Float64}, ctx::F.Context)
    d = T.unwrap_displacement(ctx.box, q, f.i, f.j)
    r = sqrt(T.norm2(d))
    dr = r - f.r0
    c = f.k * dr / r
    L = ctx.box.L
    w = (d.x / L[1], d.y / L[2], d.z / L[3])
    # G += c * d * w'
    G[1, 1] += c * d.x * w[1]; G[1, 2] += c * d.x * w[2]; G[1, 3] += c * d.x * w[3]
    G[2, 1] += c * d.y * w[1]; G[2, 2] += c * d.y * w[2]; G[2, 3] += c * d.y * w[3]
    G[3, 1] += c * d.z * w[1]; G[3, 2] += c * d.z * w[2]; G[3, 3] += c * d.z * w[3]
    return G
end

# Now box_derivative_support cleanly recognizes the implementation!
@assert RS.box_derivative_support(f_custom) === nothing
c_box = T.OrthorhombicBox((10.0, 10.0, 10.0))
@assert SC.supports_box_derivative(f_custom, c_box)

# Construct a resolved system with the custom factor declaring :box_derivative
rs_custom = RS.ResolvedSystem(;
    masses = [12.0, 12.0],
    charges = [0.0, 0.0],
    box = c_box,
    factors = [f_custom],
    declared = [:energy, :cartesian_forces, :component_energies, :box_derivative],
    provenance = test_ff
)
@assert :box_derivative in RS.engine_capabilities(rs_custom)

q_cust = [2.0 4.0; 3.0 3.0; 1.0 1.0]

# Compute analytic box derivative via delegation
G_cust = SC.box_derivative(rs_custom, q_cust)
@assert all(isfinite, G_cust)

# Verify component box derivative delegation
G_comp = SC.box_derivative_component(rs_custom, q_cust, :component_1)
@assert G_comp == G_cust

# Verify against central finite difference of energy with fixed fractional coordinates
s_cust = SC.fractional_coordinates(rs_custom, q_cust)
errors = Float64[]
eps_steps = [1e-3, 1e-4, 1e-5]
for eps in eps_steps
    G_fd = zeros(3, 3)
    for a in 1:3
        # Perturb diagonal box length in direction a
        L_plus = collect(c_box.L)
        L_plus[a] += eps
        box_p = T.OrthorhombicBox((L_plus[1], L_plus[2], L_plus[3]))
        q_p = [s_cust[d, j] * box_p.L[d] for d in 1:3, j in 1:2]
        rs_p = RS.ResolvedSystem(; masses = [12.0, 12.0], charges = [0.0, 0.0],
                                 box = box_p, factors = [f_custom], provenance = test_ff)
        u_plus = SC.energy(rs_p, q_p)

        L_minus = collect(c_box.L)
        L_minus[a] -= eps
        box_m = T.OrthorhombicBox((L_minus[1], L_minus[2], L_minus[3]))
        q_m = [s_cust[d, j] * box_m.L[d] for d in 1:3, j in 1:2]
        rs_m = RS.ResolvedSystem(; masses = [12.0, 12.0], charges = [0.0, 0.0],
                                 box = box_m, factors = [f_custom], provenance = test_ff)
        u_minus = SC.energy(rs_m, q_m)

        G_fd[a, a] = (u_plus - u_minus) / (2.0 * eps)
    end
    # Compare diagonal elements
    push!(errors, maximum(abs.([G_cust[i, i] - G_fd[i, i] for i in 1:3])))
end
# Analytical accuracy verification against central finite difference
@assert all(e < 1e-8 for e in errors) "expected sub-1e-8 agreement with FD, got $(errors)"

# Virial and affine pressure check for delegated box derivative
vir = SC.virial(rs_custom, q_cust)
@assert all(isfinite, vir)
p_aff = SC.affine_pressure(rs_custom, q_cust)
@assert isfinite(p_aff)

println("  delegated box derivative FD errors: eps=1e-3 => $(errors[1]), eps=1e-4 => $(errors[2]), eps=1e-5 => $(errors[3])")

println("test_resolved_system.jl: all resolution-contract checks passed")
