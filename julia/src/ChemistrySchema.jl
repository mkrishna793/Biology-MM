# Normalized parameter-file schema v1.0.0 (AMBER/CHARMM -> foundation units).
#
# This file defines the import record only. It contains no engine factors and no
# OpenMM objects: the builder (`ChemistryBuild.jl`) is the single place where a
# normalized record becomes a `ResolvedSystem`. See `chemistry/
# INTERFACE.md` for the normative JSON contract.

const CHEM_SCHEMA_VERSION = v"1.0.0"
const CHEM_UNITS_NAME = "nm-kj-mol-e-amu"
const CHEM_KCAL_TO_KJ = 4.184                 # exact by definition
const CHEM_ANGSTROM_TO_NM = 0.1
const CHEM_BOND_KCAL_A2_TO_KJ_NM2 = CHEM_KCAL_TO_KJ / CHEM_ANGSTROM_TO_NM^2
const CHEM_AMU_KG = 1.66053906660e-27
const CHEM_ELEMENTARY_CHARGE_C = 1.602176634e-19
const CHEM_AVOGADRO = 6.02214076e23
const CHEM_COULOMB_PREFACTOR =
    8.9875517923e9 * CHEM_ELEMENTARY_CHARGE_C^2 * CHEM_AVOGADRO * 1e9 / 1000.0
# 0.00831446261815324 kJ/mol/K; written explicitly to avoid a misleading formula.
const CHEM_BOLTZMANN = 0.00831446261815324

"""
Foundation unit record for imported AMBER/CHARMM systems: nm, kJ/mol, e, amu.
One `Types.Units` record is used by every imported system, so a factor cannot
silently mix Å with nm or kcal with kJ.
"""
const CHEM_UNITS = Types.Units(
    "import (nm, kJ/mol, e, amu)",
    1e-9,                                       # metres per nm
    1000.0 / CHEM_AVOGADRO,                     # joules per kJ/mol
    CHEM_AMU_KG,                                # kg per amu
    CHEM_ELEMENTARY_CHARGE_C,                   # C per e
    CHEM_COULOMB_PREFACTOR,                     # kJ/mol * nm / e^2
    CHEM_BOLTZMANN,
)

"Typed import/normalization refusal. `path` locates the JSON field when known."
struct ChemistryError <: Exception
    code::Symbol
    message::String
    path::String
end
ChemistryError(code::Symbol, message::AbstractString) =
    ChemistryError(code, String(message), "")
Base.showerror(io::IO, e::ChemistryError) =
    print(io, "ChemistryError[", e.code, "]",
          isempty(e.path) ? "" : " at " * e.path, ": ", e.message)

_chem_error(code::Symbol, path::AbstractString, msg::AbstractString) =
    throw(ChemistryError(code, String(msg), String(path)))

# ---------------------------------------------------------------------------
# types
# ---------------------------------------------------------------------------

struct NormalizedAtom
    name::String
    mass::Float64
    charge::Float64
    element::String
    residue_index::Int
    residue_name::String
    sigma_nm::Float64
    epsilon_kj::Float64
end

struct NormalizedBox
    kind::Symbol                   # :none, :orthorhombic, :triclinic
    lengths_nm::NTuple{3,Float64}
    vectors_nm::Matrix{Float64}    # 3x3, columns are lattice vectors
end
NormalizedBox() = NormalizedBox(:none, (0.0, 0.0, 0.0), zeros(3, 3))

struct BondTerm
    i::Int; j::Int; k::Float64; r0_nm::Float64
end
struct AngleTerm
    i::Int; j::Int; k::Int; k_theta::Float64; theta0_rad::Float64
end
struct UreyBradleyTerm
    i::Int; j::Int; k::Float64; r0_nm::Float64
end
struct TorsionTerm
    k::Float64; n::Int; phase_rad::Float64
end
struct DihedralTerm
    i::Int; j::Int; k::Int; l::Int
    terms::Vector{TorsionTerm}
end
struct ImproperTerm
    i::Int; j::Int; k::Int; l::Int; k_phi::Float64; phase_rad::Float64
end
struct CmapTerm
    indices::NTuple{8,Int}
    resolution::Int
    grid::Vector{Float64}
end
struct Pair14Term
    i::Int; j::Int; coulomb_scale::Float64; lj_scale::Float64
end
struct C64Term
    i::Int; j::Int; c::Float64         # kJ/mol * nm^4, U = -C/r^4
end
struct VirtualSiteTerm
    kind::Symbol
    site::Int
    parents::Vector{Int}
    coefficients::Vector{Float64}
end
struct ConstraintTerm
    i::Int; j::Int; distance_nm::Float64
end

struct ImportProvenance
    source::String
    force_field::String
    source_sha256::String
    coordinate_source::String
    coordinate_sha256::String
    generator::String
    generator_version::String
    notes::String
end

struct DiagnosticRecord
    code::Symbol
    severity::Symbol              # :error, :warning, :info
    message::String
    atoms::Vector{Int}
    residues::Vector{Int}
end

struct NormalizedSystem
    schema::VersionNumber
    family::Symbol
    units::String
    provenance::ImportProvenance
    atoms::Vector{NormalizedAtom}
    box::NormalizedBox
    bonds::Vector{BondTerm}
    angles::Vector{AngleTerm}
    urey_bradleys::Vector{UreyBradleyTerm}
    dihedrals::Vector{DihedralTerm}
    impropers::Vector{ImproperTerm}
    cmaps::Vector{CmapTerm}
    exclusions::Vector{NTuple{2,Int}}
    pair14::Vector{Pair14Term}
    lj146::Vector{C64Term}
    virtual_sites::Vector{VirtualSiteTerm}
    constraints::Vector{ConstraintTerm}
    input_diagnostics::Vector{DiagnosticRecord}
    net_charge::Float64
end

n_atoms(ns::NormalizedSystem) = length(ns.atoms)

# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

const _TOP_KEYS = (:schema, :family, :units, :provenance, :atoms, :box, :bonds,
                   :angles, :urey_bradleys, :dihedrals, :impropers, :cmaps,
                   :exclusions, :pair14, :virtual_sites, :constraints,
                   :input_diagnostics, :net_charge, :lj146)

function _obj(x, path::AbstractString)
    x isa AbstractDict || _chem_error(:json, path, "expected a JSON object")
    return x
end
function _arr(x, path::AbstractString)
    x isa AbstractVector || _chem_error(:json, path, "expected a JSON array")
    return x
end
function _str(x, path::AbstractString; nonempty::Bool = false)
    x isa AbstractString || _chem_error(:json, path, "expected a string")
    s = String(x)
    nonempty && isempty(strip(s)) && _chem_error(:json, path, "must be non-empty")
    return s
end
function _int(x, path::AbstractString; min::Int = typemin(Int))
    (x isa Integer) || _chem_error(:json, path, "expected an integer")
    v = Int(x)
    v >= min || _chem_error(:json, path, "must be >= $min")
    return v
end
function _num(x, path::AbstractString; positive::Bool = false, nonnegative::Bool = false)
    (x isa Real) || _chem_error(:json, path, "expected a number")
    v = Float64(x)
    isfinite(v) || _chem_error(:parameter, path, "must be finite")
    positive && !(v > 0) && _chem_error(:parameter, path, "must be > 0")
    nonnegative && v < 0 && _chem_error(:parameter, path, "must be >= 0")
    return v
end
function _key(d::AbstractDict, key::Symbol, path::AbstractString)
    haskey(d, String(key)) || _chem_error(:json, path, "missing required key '$key'")
    return d[String(key)]
end
function _opt(d::AbstractDict, key::Symbol, default, path::AbstractString)
    haskey(d, String(key)) || return default
    return d[String(key)]
end
function _check_keys(d::AbstractDict, allowed::Tuple, path::AbstractString)
    for k in keys(d)
        ks = Symbol(k)
        ks in allowed || _chem_error(:json, path, "unknown key '$(k)'")
    end
end

# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

function _parse_atom(x, k::Int)
    d = _obj(x, "atoms[$k]")
    _check_keys(d, (:name, :mass, :charge, :element, :residue_index,
                    :residue_name, :sigma_nm, :epsilon_kj), "atoms[$k]")
    NormalizedAtom(
        _str(_key(d, :name, "atoms[$k]"), "atoms[$k].name"; nonempty = true),
        _num(_key(d, :mass, "atoms[$k]"), "atoms[$k].mass"; positive = true),
        _num(_key(d, :charge, "atoms[$k]"), "atoms[$k].charge"),
        _str(_key(d, :element, "atoms[$k]"), "atoms[$k].element"; nonempty = true),
        _int(_key(d, :residue_index, "atoms[$k]"), "atoms[$k].residue_index"; min = 1),
        _str(_key(d, :residue_name, "atoms[$k]"), "atoms[$k].residue_name"; nonempty = true),
        _num(_key(d, :sigma_nm, "atoms[$k]"), "atoms[$k].sigma_nm"; nonnegative = true),
        _num(_key(d, :epsilon_kj, "atoms[$k]"), "atoms[$k].epsilon_kj"; nonnegative = true),
    )
end

function _parse_box(x)
    d = _obj(x, "box")
    kind = Symbol(_str(_key(d, :kind, "box"), "box.kind"; nonempty = true))
    if kind === :none
        _check_keys(d, (:kind,), "box")
        return NormalizedBox()
    elseif kind === :orthorhombic
        _check_keys(d, (:kind, :lengths_nm), "box")
        L = _arr(_key(d, :lengths_nm, "box"), "box.lengths_nm")
        length(L) == 3 || _chem_error(:box, "box.lengths_nm", "must have three entries")
        Lv = ntuple(i -> _num(L[i], "box.lengths_nm[$i]"; positive = true), 3)
        return NormalizedBox(:orthorhombic, Lv, zeros(3, 3))
    elseif kind === :triclinic
        _check_keys(d, (:kind, :vectors_nm), "box")
        V = _arr(_key(d, :vectors_nm, "box"), "box.vectors_nm")
        length(V) == 3 || _chem_error(:box, "box.vectors_nm", "must have three rows")
        M = zeros(3, 3)
        for a in 1:3
            row = _arr(V[a], "box.vectors_nm[$a]")
            length(row) == 3 || _chem_error(:box, "box.vectors_nm[$a]", "must have three entries")
            for b in 1:3
                M[a, b] = _num(row[b], "box.vectors_nm[$a][$b]")
            end
        end
        return NormalizedBox(:triclinic, (0.0, 0.0, 0.0), M)
    end
    _chem_error(:box, "box.kind", "must be one of none/orthorhombic/triclinic")
end

function _parse_terms(x, label::AbstractString, keyorder::Tuple)
    arr = _arr(x, label)
    for (k, item) in enumerate(arr)
        _obj(item, "$label[$k]")
        _check_keys(item, keyorder, "$label[$k]")
    end
    return arr
end

function _parse_provenance(x)
    d = _obj(x, "provenance")
    allowed = (:source, :force_field, :source_sha256, :coordinate_source,
               :coordinate_sha256, :generator, :generator_version, :notes)
    _check_keys(d, allowed, "provenance")
    return ImportProvenance(
        _str(_key(d, :source, "provenance"), "provenance.source"; nonempty = true),
        _str(_key(d, :force_field, "provenance"), "provenance.force_field"; nonempty = true),
        _str(_key(d, :source_sha256, "provenance"), "provenance.source_sha256"),
        _str(_opt(d, :coordinate_source, "", "provenance"), "provenance.coordinate_source"),
        _str(_opt(d, :coordinate_sha256, "", "provenance"), "provenance.coordinate_sha256"),
        _str(_opt(d, :generator, "", "provenance"), "provenance.generator"),
        _str(_opt(d, :generator_version, "", "provenance"), "provenance.generator_version"),
        _str(_opt(d, :notes, "", "provenance"), "provenance.notes"),
    )
end

function _parse_diagnostic(x, path::AbstractString)
    d = _obj(x, path)
    _check_keys(d, (:code, :severity, :message, :atoms, :residues), path)
    atoms = Int[_int(v, "$path.atoms[$i]"; min = 1) for (i, v) in
                enumerate(_arr(_opt(d, :atoms, Any[], path), "$path.atoms"))]
    residues = Int[_int(v, "$path.residues[$i]"; min = 1) for (i, v) in
                   enumerate(_arr(_opt(d, :residues, Any[], path), "$path.residues"))]
    sev = Symbol(_str(_key(d, :severity, path), "$path.severity"))
    sev in (:error, :warning, :info) || _chem_error(:json, "$path.severity",
        "must be error/warning/info")
    return DiagnosticRecord(Symbol(_str(_key(d, :code, path), "$path.code")), sev,
                            _str(_key(d, :message, path), "$path.message"), atoms, residues)
end

function _parse_json(root)
    d = _obj(root, "<root>")
    for k in keys(d)
        Symbol(k) in _TOP_KEYS || _chem_error(:json, "<root>", "unknown top-level key '$k'")
    end
    for k in _TOP_KEYS
        haskey(d, String(k)) || _chem_error(:json, "<root>", "missing top-level key '$k'")
    end
    schema = VersionNumber(_str(d["schema"], "schema"))
    family = Symbol(_str(d["family"], "family"))
    units = _str(d["units"], "units")
    prov = _parse_provenance(d["provenance"])
    atoms = [_parse_atom(a, k) for (k, a) in enumerate(_arr(d["atoms"], "atoms"))]
    box = _parse_box(d["box"])
    bonds = [BondTerm(_int(_key(b, :i, "bonds[$k]"), "bonds[$k].i"; min = 1),
                      _int(_key(b, :j, "bonds[$k]"), "bonds[$k].j"; min = 1),
                      _num(_key(b, :k, "bonds[$k]"), "bonds[$k].k"),
                      _num(_key(b, :r0_nm, "bonds[$k]"), "bonds[$k].r0_nm"; positive = true))
             for (k, b) in enumerate(_parse_terms(d["bonds"], "bonds", (:i, :j, :k, :r0_nm)))]
    angles = [AngleTerm(_int(_key(a, :i, "angles[$k]"), "angles[$k].i"; min = 1),
                        _int(_key(a, :j, "angles[$k]"), "angles[$k].j"; min = 1),
                        _int(_key(a, :k, "angles[$k]"), "angles[$k].k"; min = 1),
                        _num(_key(a, :k_theta, "angles[$k]"), "angles[$k].k_theta"),
                        _num(_key(a, :theta0_rad, "angles[$k]"), "angles[$k].theta0_rad"))
              for (k, a) in enumerate(_parse_terms(d["angles"], "angles",
                                                   (:i, :j, :k, :k_theta, :theta0_rad)))]
    ureys = [UreyBradleyTerm(_int(_key(u, :i, "urey_bradleys[$k]"), "urey_bradleys[$k].i"; min = 1),
                             _int(_key(u, :j, "urey_bradleys[$k]"), "urey_bradleys[$k].j"; min = 1),
                             _num(_key(u, :k, "urey_bradleys[$k]"), "urey_bradleys[$k].k"),
                             _num(_key(u, :r0_nm, "urey_bradleys[$k]"), "urey_bradleys[$k].r0_nm"; positive = true))
             for (k, u) in enumerate(_parse_terms(d["urey_bradleys"], "urey_bradleys",
                                                  (:i, :j, :k, :r0_nm)))]
    dihedrals = DihedralTerm[]
    for (k, x) in enumerate(_parse_terms(d["dihedrals"], "dihedrals", (:i, :j, :k, :l, :terms)))
        path = "dihedrals[$k]"
        terms = TorsionTerm[]
        for (m, t) in enumerate(_parse_terms(x["terms"], "$path.terms", (:k, :n, :phase_rad)))
            push!(terms, TorsionTerm(_num(t["k"], "$path.terms[$m].k"),
                                     _int(t["n"], "$path.terms[$m].n"; min = 1),
                                     _num(t["phase_rad"], "$path.terms[$m].phase_rad")))
        end
        isempty(terms) && _chem_error(:parameter, "$path.terms", "must not be empty")
        push!(dihedrals, DihedralTerm(_int(x["i"], "$path.i"; min = 1),
                                      _int(x["j"], "$path.j"; min = 1),
                                      _int(x["k"], "$path.k"; min = 1),
                                      _int(x["l"], "$path.l"; min = 1), terms))
    end
    impropers = [ImproperTerm(_int(_key(x, :i, "impropers[$k]"), "impropers[$k].i"; min = 1),
                              _int(_key(x, :j, "impropers[$k]"), "impropers[$k].j"; min = 1),
                              _int(_key(x, :k, "impropers[$k]"), "impropers[$k].k"; min = 1),
                              _int(_key(x, :l, "impropers[$k]"), "impropers[$k].l"; min = 1),
                              _num(_key(x, :k_phi, "impropers[$k]"), "impropers[$k].k_phi"),
                              _num(_key(x, :phase_rad, "impropers[$k]"), "impropers[$k].phase_rad"))
                 for (k, x) in enumerate(_parse_terms(d["impropers"], "impropers",
                                                      (:i, :j, :k, :l, :k_phi, :phase_rad)))]
    cmaps = CmapTerm[]
    for (k, x) in enumerate(_parse_terms(d["cmaps"], "cmaps", (:i, :resolution, :grid)))
        path = "cmaps[$k]"
        idx = _arr(x["i"], "$path.i")
        length(idx) == 8 || _chem_error(:parameter, "$path.i", "needs eight atom indices")
        tupleidx = ntuple(m -> _int(idx[m], "$path.i[$m]"; min = 1), 8)
        grid = Float64[_num(g, "$path.grid[$m]") for (m, g) in enumerate(_arr(x["grid"], "$path.grid"))]
        push!(cmaps, CmapTerm(tupleidx, _int(x["resolution"], "$path.resolution"; min = 2), grid))
    end
    exclusions = NTuple{2,Int}[]
    for (k, p) in enumerate(_arr(d["exclusions"], "exclusions"))
        path = "exclusions[$k]"
        length(p) == 2 || _chem_error(:parameter, path, "must be an (i,j) pair")
        i = _int(p[1], "$path[1]"; min = 1); j = _int(p[2], "$path[2]"; min = 1)
        i == j && _chem_error(:parameter, path, "pair must reference distinct atoms")
        push!(exclusions, i < j ? (i, j) : (j, i))
    end
    pair14 = [Pair14Term(_int(_key(p, :i, "pair14[$k]"), "pair14[$k].i"; min = 1),
                         _int(_key(p, :j, "pair14[$k]"), "pair14[$k].j"; min = 1),
                         _num(_key(p, :coulomb_scale, "pair14[$k]"), "pair14[$k].coulomb_scale"; nonnegative = true),
                         _num(_key(p, :lj_scale, "pair14[$k]"), "pair14[$k].lj_scale"; nonnegative = true))
              for (k, p) in enumerate(_parse_terms(d["pair14"], "pair14",
                                                   (:i, :j, :coulomb_scale, :lj_scale)))]
    lj146 = [C64Term(_int(_key(x, :i, "lj146[$k]"), "lj146[$k].i"; min = 1),
                     _int(_key(x, :j, "lj146[$k]"), "lj146[$k].j"; min = 1),
                     _num(_key(x, :c, "lj146[$k]"), "lj146[$k].c"))
              for (k, x) in enumerate(_parse_terms(d["lj146"], "lj146", (:i, :j, :c)))]
    vss = VirtualSiteTerm[]
    for (k, x) in enumerate(_parse_terms(d["virtual_sites"], "virtual_sites",
                                         (:kind, :site, :parents, :coefficients)))
        path = "virtual_sites[$k]"
        parents = Int[_int(v, "$path.parents[$m]"; min = 1) for (m, v) in
                      enumerate(_arr(x["parents"], "$path.parents"))]
        coefs = Float64[_num(v, "$path.coefficients[$m]") for (m, v) in
                        enumerate(_arr(x["coefficients"], "$path.coefficients"))]
        push!(vss, VirtualSiteTerm(Symbol(_str(x["kind"], "$path.kind"; nonempty = true)),
                                   _int(x["site"], "$path.site"; min = 1), parents, coefs))
    end
    constraints = ConstraintTerm[]
    for (k, x) in enumerate(_parse_terms(d["constraints"], "constraints", (:i, :j, :distance_nm)))
        path = "constraints[$k]"
        push!(constraints, ConstraintTerm(_int(x["i"], "$path.i"; min = 1),
                                          _int(x["j"], "$path.j"; min = 1),
                                          _num(x["distance_nm"], "$path.distance_nm"; positive = true)))
    end
    diags = [DiagnosticRecord(Symbol(_str(_key(x, :code, "input_diagnostics[$k]"), "input_diagnostics[$k].code")),
                              :info, "", Int[], Int[])
             for (k, x) in enumerate(_arr(d["input_diagnostics"], "input_diagnostics")) if false]
    diags = [_parse_diagnostic(x, "input_diagnostics[$k]")
             for (k, x) in enumerate(_arr(d["input_diagnostics"], "input_diagnostics"))]
    net = _num(d["net_charge"], "net_charge")
    return NormalizedSystem(schema, family, units, prov, atoms, box, bonds, angles,
                            ureys, dihedrals, impropers, cmaps, exclusions, pair14,
                            lj146, vss, constraints, diags, net)
end

"""
    read_normalized(path) -> NormalizedSystem

Read and validate a `mdparams` v1.0.0 JSON record. Parsing is strict: unknown
keys, wrong units, non-finite numbers, out-of-range indices and inconsistent
pairs are typed `ChemistryError`s, never warnings.
"""
function read_normalized(path::AbstractString)
    isfile(path) || _chem_error(:io, "", "no such file: $path")
    root = try
        JSON.parsefile(path; dicttype = Dict{String,Any})
    catch e
        e isa ChemistryError && rethrow()
        _chem_error(:json, "", "cannot parse JSON: $(sprint(showerror, e))")
    end
    ns = _parse_json(root)
    validate_normalized!(ns)
    return ns
end

# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

function _check_index(i::Int, N::Int, path::AbstractString)
    1 <= i <= N || _chem_error(:index, path, "atom index $i is outside 1:$N")
    return nothing
end

function _check_support(idx, N::Int, path::AbstractString)
    length(unique(idx)) == length(idx) ||
        _chem_error(:parameter, path, "support repeats an atom index")
    for i in idx
        _check_index(i, N, path)
    end
    return nothing
end

function _check_duplicates(pairs, path::AbstractString)
    seen = Set{Any}()
    for p in pairs
        p in seen && _chem_error(:duplicate, path, "duplicate term $p")
        push!(seen, p)
    end
    return nothing
end

"""
    validate_normalized!(ns) -> ns

Schema/consistency validation. This is the import gate, separate from
`ResolvedSystem` validation: it knows about 1-4 lists, multi-term torsions,
CHARMM Urey-Bradley/CMAP records, and the fixed v1.0.0 unit system.
"""
function validate_normalized!(ns::NormalizedSystem)
    ns.schema.major == CHEM_SCHEMA_VERSION.major ||
        _chem_error(:schema, "schema", "unsupported major version $(ns.schema); expected $(CHEM_SCHEMA_VERSION)")
    ns.family in (:amber, :charmm) ||
        _chem_error(:family, "family", "must be amber or charmm, got $(ns.family)")
    ns.units == CHEM_UNITS_NAME ||
        _chem_error(:units, "units", "v1.0.0 requires '$CHEM_UNITS_NAME', got '$(ns.units)'")
    prov = ns.provenance
    isempty(strip(prov.source)) && _chem_error(:provenance, "provenance.source", "must be non-empty")
    isempty(strip(prov.force_field)) && _chem_error(:provenance, "provenance.force_field", "must be non-empty")
    if !isempty(prov.source_sha256)
        occursin(r"^[0-9a-f]{64}$", prov.source_sha256) ||
            _chem_error(:provenance, "provenance.source_sha256", "must be 64 lowercase hex characters")
    end
    if !isempty(prov.coordinate_sha256)
        occursin(r"^[0-9a-f]{64}$", prov.coordinate_sha256) ||
            _chem_error(:provenance, "provenance.coordinate_sha256", "must be 64 lowercase hex characters")
    end
    N = length(ns.atoms)
    N >= 1 || _chem_error(:shape, "atoms", "at least one atom is required")
    last_res = 0
    for (k, a) in enumerate(ns.atoms)
        path = "atoms[$k]"
        isempty(strip(a.name)) && _chem_error(:atom, "$path.name", "must be non-empty")
        isempty(strip(a.element)) && _chem_error(:atom, "$path.element", "must be non-empty")
        a.mass >= 0 && isfinite(a.mass) || _chem_error(:atom, "$path.mass", "must be finite and >= 0 (zero only for a declared virtual site)")
        isfinite(a.charge) || _chem_error(:atom, "$path.charge", "must be finite")
        a.sigma_nm >= 0 && isfinite(a.sigma_nm) || _chem_error(:atom, "$path.sigma_nm", "must be finite and >= 0")
        a.epsilon_kj >= 0 && isfinite(a.epsilon_kj) || _chem_error(:atom, "$path.epsilon_kj", "must be finite and >= 0")
        a.residue_index >= last_res ||
            _chem_error(:atom, "$path.residue_index", "residue indices must be non-decreasing in atom order")
        last_res = a.residue_index
    end
    # box
    if ns.box.kind === :orthorhombic
        all(l -> isfinite(l) && l > 0, ns.box.lengths_nm) ||
            _chem_error(:box, "box.lengths_nm", "lengths must be finite and positive")
    elseif ns.box.kind === :triclinic
        M = ns.box.vectors_nm
        all(isfinite, M) || _chem_error(:box, "box.vectors_nm", "entries must be finite")
        det(M) > 0 || _chem_error(:box, "box.vectors_nm", "cell must have positive volume")
    elseif ns.box.kind !== :none
        _chem_error(:box, "box.kind", "unknown box kind $(ns.box.kind)")
    end
    # bonded terms
    _check_duplicates(((t.i, t.j) for t in ns.bonds), "bonds")
    for (k, t) in enumerate(ns.bonds)
        _check_support((t.i, t.j), N, "bonds[$k]")
        t.r0_nm > 0 && isfinite(t.r0_nm) || _chem_error(:parameter, "bonds[$k].r0_nm", "must be > 0")
        isfinite(t.k) || _chem_error(:parameter, "bonds[$k].k", "must be finite")
    end
    _check_duplicates(((t.i, t.j, t.k) for t in ns.angles), "angles")
    for (k, t) in enumerate(ns.angles)
        _check_support((t.i, t.j, t.k), N, "angles[$k]")
        isfinite(t.k) || _chem_error(:parameter, "angles[$k].k", "must be finite")
        isfinite(t.theta0_rad) || _chem_error(:parameter, "angles[$k].theta0_rad", "must be finite")
        (0 < t.theta0_rad <= pi) || _chem_error(:parameter, "angles[$k].theta0_rad", "must lie in (0, pi]")
    end
    _check_duplicates(((t.i, t.j) for t in ns.urey_bradleys), "urey_bradleys")
    for (k, t) in enumerate(ns.urey_bradleys)
        _check_support((t.i, t.j), N, "urey_bradleys[$k]")
        t.r0_nm > 0 && isfinite(t.r0_nm) || _chem_error(:parameter, "urey_bradleys[$k].r0_nm", "must be > 0")
        isfinite(t.k) || _chem_error(:parameter, "urey_bradleys[$k].k", "must be finite")
    end
    _check_duplicates(((t.i, t.j, t.k, t.l) for t in ns.dihedrals), "dihedrals")
    for (k, t) in enumerate(ns.dihedrals)
        _check_support((t.i, t.j, t.k, t.l), N, "dihedrals[$k]")
        isempty(t.terms) && _chem_error(:parameter, "dihedrals[$k].terms", "must not be empty")
        for (m, q) in enumerate(t.terms)
            q.n >= 1 || _chem_error(:parameter, "dihedrals[$k].terms[$m].n", "must be >= 1")
            isfinite(q.k) && isfinite(q.phase_rad) ||
                _chem_error(:parameter, "dihedrals[$k].terms[$m]", "k and phase must be finite")
        end
    end
    _check_duplicates(((t.i, t.j, t.k, t.l) for t in ns.impropers), "impropers")
    for (k, t) in enumerate(ns.impropers)
        _check_support((t.i, t.j, t.k, t.l), N, "impropers[$k]")
        isfinite(t.k) && isfinite(t.phase_rad) ||
            _chem_error(:parameter, "impropers[$k]", "k and phase must be finite")
    end
    for (k, t) in enumerate(ns.cmaps)
        # A CMAP torsion is (a,b,c,d,b,c,d,e): five distinct atoms in an
        # eight-slot list, so the general "no repeated index" support rule does
        # not apply. Only range validity is required here.
        for (m, i) in enumerate(t.indices)
            _check_index(i, N, "cmaps[$k].i[$m]")
        end
        t.resolution >= 2 || _chem_error(:parameter, "cmaps[$k].resolution", "must be >= 2")
        length(t.grid) == t.resolution^2 ||
            _chem_error(:parameter, "cmaps[$k].grid", "expected $(t.resolution^2) values, got $(length(t.grid))")
        all(isfinite, t.grid) || _chem_error(:parameter, "cmaps[$k].grid", "must be finite")
    end
    # pair rules
    _check_duplicates(ns.exclusions, "exclusions")
    excl = Set(ns.exclusions)
    _check_duplicates(((t.i < t.j ? (t.i, t.j) : (t.j, t.i)) for t in ns.pair14), "pair14")
    p14 = Set{Tuple{Int,Int}}()
    for (k, t) in enumerate(ns.pair14)
        _check_support((t.i, t.j), N, "pair14[$k]")
        key = t.i < t.j ? (t.i, t.j) : (t.j, t.i)
        key in excl && _chem_error(:pair, "pair14[$k]",
            "pair $(key) is both an exclusion and a 1-4 interaction; remove it from exclusions")
        (0 <= t.coulomb_scale <= 1) && (0 <= t.lj_scale <= 1) ||
            _chem_error(:pair, "pair14[$k]", "scales must lie in [0, 1]")
        push!(p14, key)
    end
    _check_duplicates(((min(t.i, t.j), max(t.i, t.j)) for t in ns.lj146), "lj146")
    lj146pairs = Set{Tuple{Int,Int}}()
    for (k, t) in enumerate(ns.lj146)
        i, j = min(t.i, t.j), max(t.i, t.j)
        _check_support((i, j), N, "lj146[$k]")
        isfinite(t.c) || _chem_error(:parameter, "lj146[$k].c", "must be finite")
        i in excl && _chem_error(:pair, "lj146[$k]",
            "pair ($i, $j) is an exclusion; the 1-6-4 term must not reintroduce it")
        (i, j) in p14 && _chem_error(:pair, "lj146[$k]",
            "pair ($i, $j) is a 1-4 pair; 1-6-4 terms are not applied to exceptions")
        push!(lj146pairs, (i, j))
    end
    # virtual sites
    seen_sites = Set{Int}()
    for (k, v) in enumerate(ns.virtual_sites)
        path = "virtual_sites[$k]"
        v.kind in (:linear, :planar, :average) ||
            _chem_error(:parameter, "$path.kind", "linear/planar/average required")
        _check_index(v.site, N, "$path.site")
        v.site in seen_sites && _chem_error(:duplicate, "$path.site", "site declared twice")
        push!(seen_sites, v.site)
        npar = v.kind === :linear ? 2 : v.kind === :planar ? 3 : 2
        v.kind === :average && length(v.parents) < 2 &&
            _chem_error(:parameter, "$path.parents", "average sites need at least two parents")
        v.kind !== :average && length(v.parents) != npar &&
            _chem_error(:parameter, "$path.parents", "expected $npar parents for $(v.kind)")
        length(v.coefficients) == length(v.parents) ||
            _chem_error(:parameter, "$path.coefficients", "one coefficient per parent")
        _check_support(v.parents, N, "$path.parents")
        v.site in v.parents && _chem_error(:parameter, "$path.parents", "site may not be its own parent")
        all(isfinite, v.coefficients) || _chem_error(:parameter, "$path.coefficients", "must be finite")
        abs(sum(v.coefficients) - 1.0) <= 1e-12 ||
            _chem_error(:parameter, "$path.coefficients", "coefficients must sum to 1")
        ns.atoms[v.site].mass == 0.0 ||
            _chem_error(:atom, "atoms[$(v.site)].mass", "a virtual site must be massless")
    end
    for (k, a) in enumerate(ns.atoms)
        a.mass == 0.0 && !(k in seen_sites) &&
            _chem_error(:atom, "atoms[$k].mass", "massless atom is not a declared virtual site")
    end
    # constraints
    _check_duplicates(((min(t.i, t.j), max(t.i, t.j)) for t in ns.constraints), "constraints")
    for (k, t) in enumerate(ns.constraints)
        _check_support((t.i, t.j), N, "constraints[$k]")
        t.distance_nm > 0 && isfinite(t.distance_nm) ||
            _chem_error(:parameter, "constraints[$k].distance_nm", "must be > 0")
    end
    # net charge
    qsum = sum(a.charge for a in ns.atoms)
    abs(qsum - ns.net_charge) <= 1e-6 ||
        _chem_error(:charge, "net_charge", "declared $(ns.net_charge) but atom charges sum to $(qsum)")
    for (k, d) in enumerate(ns.input_diagnostics)
        for i in d.atoms
            _check_index(i, N, "input_diagnostics[$k].atoms")
        end
    end
    return ns
end

# ---------------------------------------------------------------------------
# deterministic writer + summary
# ---------------------------------------------------------------------------

function _atom_json(a::NormalizedAtom)
    Dict{String,Any}("name" => a.name, "mass" => a.mass, "charge" => a.charge,
                     "element" => a.element, "residue_index" => a.residue_index,
                     "residue_name" => a.residue_name, "sigma_nm" => a.sigma_nm,
                     "epsilon_kj" => a.epsilon_kj)
end

function _box_json(b::NormalizedBox)
    if b.kind === :orthorhombic
        return Dict{String,Any}("kind" => "orthorhombic", "lengths_nm" => collect(b.lengths_nm))
    elseif b.kind === :triclinic
        return Dict{String,Any}("kind" => "triclinic",
                                "vectors_nm" => [collect(b.vectors_nm[i, :]) for i in 1:3])
    end
    return Dict{String,Any}("kind" => "none")
end

function _diag_json(d::DiagnosticRecord)
    Dict{String,Any}("code" => String(d.code), "severity" => String(d.severity),
                     "message" => d.message, "atoms" => d.atoms, "residues" => d.residues)
end

function _normalized_json(ns::NormalizedSystem)
    prov = ns.provenance
    return Dict{String,Any}(
        "schema" => string(ns.schema),
        "family" => String(ns.family),
        "units" => ns.units,
        "provenance" => Dict{String,Any}(
            "source" => prov.source, "force_field" => prov.force_field,
            "source_sha256" => prov.source_sha256,
            "coordinate_source" => prov.coordinate_source,
            "coordinate_sha256" => prov.coordinate_sha256,
            "generator" => prov.generator, "generator_version" => prov.generator_version,
            "notes" => prov.notes),
        "atoms" => [_atom_json(a) for a in ns.atoms],
        "box" => _box_json(ns.box),
        "bonds" => [Dict{String,Any}("i" => t.i, "j" => t.j, "k" => t.k,
                                     "r0_nm" => t.r0_nm) for t in ns.bonds],
        "angles" => [Dict{String,Any}("i" => t.i, "j" => t.j, "k" => t.k,
                                      "k_theta" => t.k_theta, "theta0_rad" => t.theta0_rad)
                     for t in ns.angles],
        "urey_bradleys" => [Dict{String,Any}("i" => t.i, "j" => t.j, "k" => t.k,
                                             "r0_nm" => t.r0_nm) for t in ns.urey_bradleys],
        "dihedrals" => [Dict{String,Any}("i" => t.i, "j" => t.j, "k" => t.k, "l" => t.l,
                                         "terms" => [Dict{String,Any}("k" => q.k,
                                                                      "n" => q.n,
                                                                      "phase_rad" => q.phase_rad)
                                                     for q in t.terms])
                        for t in ns.dihedrals],
        "impropers" => [Dict{String,Any}("i" => t.i, "j" => t.j, "k" => t.k, "l" => t.l,
                                         "k_phi" => t.k_phi, "phase_rad" => t.phase_rad)
                        for t in ns.impropers],
        "cmaps" => [Dict{String,Any}("i" => collect(t.indices),
                                     "resolution" => t.resolution,
                                     "grid" => t.grid) for t in ns.cmaps],
        "exclusions" => [collect(p) for p in ns.exclusions],
        "pair14" => [Dict{String,Any}("i" => t.i, "j" => t.j,
                                      "coulomb_scale" => t.coulomb_scale,
                                      "lj_scale" => t.lj_scale) for t in ns.pair14],
        "lj146" => [Dict{String,Any}("i" => t.i, "j" => t.j, "c" => t.c)
                    for t in ns.lj146],
        "virtual_sites" => [Dict{String,Any}("kind" => String(v.kind), "site" => v.site,
                                             "parents" => v.parents,
                                             "coefficients" => v.coefficients)
                            for v in ns.virtual_sites],
        "constraints" => [Dict{String,Any}("i" => t.i, "j" => t.j,
                                           "distance_nm" => t.distance_nm)
                          for t in ns.constraints],
        "input_diagnostics" => [_diag_json(d) for d in ns.input_diagnostics],
        "net_charge" => ns.net_charge,
    )
end

"""
    write_normalized(ns, path) -> path

Deterministic JSON serialization (sorted keys, two-space indent, trailing
newline). Read -> write -> read is a semantic no-op, including atom order and
provenance.
"""
function write_normalized(ns::NormalizedSystem, path::AbstractString)
    validate_normalized!(ns)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        JSON.print(io, _normalized_json(ns), 2)
        println(io)
    end
    return path
end

"Counts and provenance-free facts, used by build reports and tests."
function normalization_summary(ns::NormalizedSystem)
    validate_normalized!(ns)
    return (
        schema = ns.schema,
        family = ns.family,
        n_atoms = length(ns.atoms),
        n_residues = isempty(ns.atoms) ? 0 : maximum(a.residue_index for a in ns.atoms),
        n_bonds = length(ns.bonds),
        n_angles = length(ns.angles),
        n_urey_bradleys = length(ns.urey_bradleys),
        n_dihedral_quadruples = length(ns.dihedrals),
        n_dihedral_terms = sum((length(t.terms) for t in ns.dihedrals); init = 0),
        n_impropers = length(ns.impropers),
        n_cmaps = length(ns.cmaps),
        n_exclusions = length(ns.exclusions),
        n_pair14 = length(ns.pair14),
        n_lj146 = length(ns.lj146),
        n_virtual_sites = length(ns.virtual_sites),
        n_constraints = length(ns.constraints),
        net_charge = ns.net_charge,
        box_kind = ns.box.kind,
    )
end
