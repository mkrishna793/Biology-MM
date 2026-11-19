# Schema and unit-contract suite for the Phase-2 importer.
#
# These checks do not need a parameter file: they pin the versioned JSON
# contract, the unit constants, and every documented validation code.

using FoundationCore
const C = FoundationCore.Chemistry
const T = FoundationCore.Types

function sample_system(; kwargs...)
    atoms = [
        C.NormalizedAtom("C", 12.011, -0.2, "C", 1, "LIG", 0.34, 0.36),
        C.NormalizedAtom("H", 1.008, 0.1, "H", 1, "LIG", 0.24, 0.08),
        C.NormalizedAtom("O", 15.999, -0.3, "O", 2, "LIG", 0.30, 0.64),
        C.NormalizedAtom("N", 14.007, 0.4, "N", 2, "LIG", 0.32, 0.20),
    ]
    defaults = Dict{Symbol,Any}(
        :schema => C.CHEM_SCHEMA_VERSION,
        :family => :amber,
        :units => C.CHEM_UNITS_NAME,
        :provenance => C.ImportProvenance("fixture.json", "unit-test force field",
                                          "a"^64, "", "", "test", "1.0.0", ""),
        :atoms => atoms,
        :box => C.NormalizedBox(:orthorhombic, (4.0, 4.0, 4.0), zeros(3, 3)),
        :bonds => [C.BondTerm(1, 2, C.CHEM_BOND_KCAL_A2_TO_KJ_NM2 * 300.0, 0.109)],
        :angles => [C.AngleTerm(2, 1, 3, 418.4 * 40.0, 1.9)],
        :urey_bradleys => C.UreyBradleyTerm[],
        :dihedrals => [C.DihedralTerm(1, 2, 3, 4,
                                      [C.TorsionTerm(1.0, 2, 0.0),
                                       C.TorsionTerm(0.5, 3, pi)])],
        :impropers => C.ImproperTerm[],
        :cmaps => C.CmapTerm[],
        :exclusions => [(1, 3), (2, 3)],
        :pair14 => [C.Pair14Term(1, 4, 1 / 1.2, 0.5)],
        :lj146 => [C.C64Term(2, 4, 0.01)],
        :virtual_sites => C.VirtualSiteTerm[],
        :constraints => C.ConstraintTerm[],
        :input_diagnostics => C.DiagnosticRecord[],
        :net_charge => 0.0,
    )
    merge!(defaults, Dict(kwargs))
    return C.NormalizedSystem(defaults[:schema], defaults[:family], defaults[:units],
                              defaults[:provenance], defaults[:atoms], defaults[:box],
                              defaults[:bonds], defaults[:angles], defaults[:urey_bradleys],
                              defaults[:dihedrals], defaults[:impropers], defaults[:cmaps],
                              defaults[:exclusions], defaults[:pair14], defaults[:lj146],
                              defaults[:virtual_sites], defaults[:constraints],
                              defaults[:input_diagnostics], defaults[:net_charge])
end

function expect_code(code::Symbol, f::Function, label::String)
    ok = false
    try
        f()
    catch e
        if e isa C.ChemistryError && e.code === code
            ok = true
        else
            rethrow()
        end
    end
    ok || error("$label: expected ChemistryError[$code]")
    return true
end

function main()
    println("="^72)
    println("Chemistry schema tests")
    println("="^72)

    # Unit constants: the imported unit record must be exactly the nm/kJ/mol/e/amu
    # system the JSON contract names, not REDUCED with a re-labelled name.
    @assert isapprox(C.CHEM_COULOMB_PREFACTOR, 138.9354576449822; rtol = 1e-14)
    @assert C.CHEM_UNITS.length == 1e-9
    @assert C.CHEM_UNITS.energy == 1000.0 / 6.02214076e23
    @assert C.CHEM_UNITS.charge == 1.602176634e-19
    @assert C.CHEM_UNITS.boltzmann == 0.00831446261815324
    @assert isapprox(C.CHEM_BOND_KCAL_A2_TO_KJ_NM2, 418.4; rtol = 1e-15)
    println("  unit record: prefactor=$(C.CHEM_COULOMB_PREFACTOR) bond_scale=$(C.CHEM_BOND_KCAL_A2_TO_KJ_NM2)")

    ns = sample_system()
    C.validate_normalized!(ns)
    s = C.normalization_summary(ns)
    @assert s.n_atoms == 4
    @assert s.n_dihedral_quadruples == 1
    @assert s.n_dihedral_terms == 2
    @assert s.n_pair14 == 1
    @assert s.n_lj146 == 1
    @assert s.n_exclusions == 2
    @assert s.n_residues == 2

    mktempdir() do dir
        path = joinpath(dir, "sample.json")
        C.write_normalized(ns, path)
        first_bytes = read(path)
        C.write_normalized(ns, path)
        @assert read(path) == first_bytes "writer is not deterministic"

        back = C.read_normalized(path)
        @assert back.schema == ns.schema
        @assert back.family == ns.family
        @assert back.units == ns.units
        @assert back.provenance.source == ns.provenance.source
        @assert back.provenance.source_sha256 == ns.provenance.source_sha256
        @assert [a.name for a in back.atoms] == [a.name for a in ns.atoms]
        @assert [a.charge for a in back.atoms] == [a.charge for a in ns.atoms]
        @assert length(back.dihedrals) == 1
        @assert length(back.dihedrals[1].terms) == 2
        @assert back.dihedrals[1].terms[2].n == 3
        @assert back.pair14[1].lj_scale == 0.5
        @assert length(back.lj146) == 1 && back.lj146[1].c == 0.01
        @assert C.normalization_summary(back).n_bonds == s.n_bonds

        # Unknown keys are refused rather than ignored.
        obj = FoundationCore.Chemistry.JSON.parsefile(path; dicttype = Dict{String,Any})
        obj["bogus"] = 1
        badpath = joinpath(dir, "unknown.json")
        open(badpath, "w") do io
            FoundationCore.Chemistry.JSON.print(io, obj)
        end
        expect_code(:json, () -> C.read_normalized(badpath), "unknown key")

        # Wrong unit system is a hard refusal: no silent Å/kcal mixing.
        obj2 = FoundationCore.Chemistry.JSON.parsefile(path; dicttype = Dict{String,Any})
        obj2["units"] = "reduced"
        b2 = joinpath(dir, "units.json")
        open(b2, "w") do io
            FoundationCore.Chemistry.JSON.print(io, obj2)
        end
        expect_code(:units, () -> C.read_normalized(b2), "wrong units")
    end

    # Validation codes that protect against silent double counting / loss.
    expect_code(:pair, () -> C.validate_normalized!(sample_system(;
        exclusions = [(1, 4)], pair14 = [C.Pair14Term(1, 4, 1 / 1.2, 0.5)])),
        "1-4 pair also excluded")
    expect_code(:duplicate, () -> C.validate_normalized!(sample_system(;
        bonds = [C.BondTerm(1, 2, 1.0, 0.1), C.BondTerm(1, 2, 2.0, 0.1)])),
        "duplicate bond")
    expect_code(:index, () -> C.validate_normalized!(sample_system(;
        bonds = [C.BondTerm(1, 9, 1.0, 0.1)])), "out-of-range atom")
    expect_code(:charge, () -> C.validate_normalized!(sample_system(; net_charge = 1.0)),
        "net charge mismatch")
    expect_code(:units, () -> C.validate_normalized!(sample_system(; units = "kcal")),
        "bad units field")
    expect_code(:parameter, () -> C.validate_normalized!(sample_system(;
        angles = [C.AngleTerm(2, 1, 3, 1.0, 4.0)])), "angle outside (0,pi]")
    expect_code(:duplicate, () -> C.validate_normalized!(sample_system(;
        dihedrals = [C.DihedralTerm(1, 2, 3, 4, [C.TorsionTerm(1.0, 2, 0.0)]),
                     C.DihedralTerm(1, 2, 3, 4, [C.TorsionTerm(2.0, 1, 0.0)])])),
        "duplicate torsion quadruple")
    expect_code(:shape, () -> C.validate_normalized!(sample_system(; atoms = C.NormalizedAtom[])),
        "empty atoms")
    expect_code(:atom, () -> C.validate_normalized!(sample_system(;
        atoms = [C.NormalizedAtom("C", 0.0, 0.0, "C", 1, "LIG", 0.3, 0.2),
                 C.NormalizedAtom("H", 1.0, 0.0, "H", 1, "LIG", 0.2, 0.1),
                 C.NormalizedAtom("O", 1.0, 0.0, "O", 2, "LIG", 0.2, 0.1),
                 C.NormalizedAtom("N", 1.0, 0.0, "N", 2, "LIG", 0.2, 0.1)])),
        "zero mass without a virtual site")

    # A virtual site is structurally accepted and must have three parents for
    # `:planar`; execution is a later ResolvedSystem refusal, not an import error.
    vs = C.NormalizedSystem(C.CHEM_SCHEMA_VERSION, :amber, C.CHEM_UNITS_NAME,
                            C.ImportProvenance("x", "ff", "b"^64, "", "", "t", "1", ""),
                            [C.NormalizedAtom("O", 16.0, -0.8, "O", 1, "WAT", 0.3, 0.6),
                             C.NormalizedAtom("H", 1.0, 0.4, "H", 1, "WAT", 0.2, 0.1),
                             C.NormalizedAtom("H", 1.0, 0.4, "H", 1, "WAT", 0.2, 0.1),
                             C.NormalizedAtom("M", 0.0, 0.0, "X", 1, "WAT", 0.0, 0.0)],
                            C.NormalizedBox(:none, (0.0, 0.0, 0.0), zeros(3, 3)),
                            C.BondTerm[], C.AngleTerm[], C.UreyBradleyTerm[],
                            C.DihedralTerm[], C.ImproperTerm[], C.CmapTerm[],
                            [(1, 2), (1, 3), (2, 3)], C.Pair14Term[], C.C64Term[],
                            [C.VirtualSiteTerm(:planar, 4, [1, 2, 3], [0.5, 0.25, 0.25])],
                            C.ConstraintTerm[], C.DiagnosticRecord[], 0.0)
    C.validate_normalized!(vs)
    println("  valid record, deterministic round trip, refusal codes: OK")

    println("CHEMISTRY SCHEMA TESTS PASSED")
    return true
end

main()
