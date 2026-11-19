# Builder: normalized record -> `ResolvedSystem`.
#
# This is the single place where an imported parameter file becomes an energy
# function. Two rules shape every line below:
#
# * Nothing is inferred. Exclusions, 1-4 scales, constraints and water rigidity
#   come from the record and the explicit `SimulationProtocol`; `ChemicalTopology`
#   is metadata only and is never a source of energy terms.
# * Nothing is dropped silently. Every term v1.0.0 cannot execute is a typed
#   `ChemistryError`, and the one term that is *represented but not executed*
#   (virtual sites) is handed to `ResolvedSystem`, whose own rules refuse
#   execution at the `SystemContract` boundary.
#
# Units are `CHEM_UNITS` (nm, kJ/mol, e, amu). The builder performs no
# rescaling: `k` is copied verbatim, so the normalized record must already carry
# the coefficient of the target factor's documented potential form
# (`Factors.HarmonicBond` uses `U = (k/2)(r-r0)²`,
# `ChemistryFactors.UreyBradley` uses `U = k(r13-r0)²`). Converting a source
# file's `strmk`/`anglk` into that convention belongs to the extractor, the only
# layer that knows which file convention applies.

"""
Residue names treated as water by `water_model = :rigid`.

A bond is omitted only when **both** of its atoms are in one of these residues;
no other bond is ever omitted, so a rigid-water choice cannot silently rigidify
a solute.
"""
const CHEM_WATER_RESIDUES = Set([:WAT, :HOH, :TP4, :TP5, :T4E, :H2O])

# `Chemistry` does `using ..ResolvedSystem`, so in this module the exported struct
# name `ResolvedSystem` shadows the module name. Resolve the module explicitly
# instead of guessing which binding wins (the same issue `StreamedLJ` documents).
const _RESOLVED = parentmodule(@__MODULE__).ResolvedSystem

"""
Protocol choices are explicit and separate from the imported topology. They are
never inferred silently.

* `nonbonded = :pme` for periodic (orthorhombic or triclinic) systems,
  `:direct` for boxless explicit pair tables, `:none` for bonded/local-only
  builds.
* `lj_mode = :trunc` reproduces unshifted 12-6 reference energies; `:lfs` is
  the C¹ force-consistent mode for dynamics. `:trunc` is refused by the
  integrator, exactly as Phase 1 documents for discontinuous potentials.
* `water_model = :flexible` keeps every bond in the parameter file (the
  reference-comparison choice); `:rigid` omits water bond energies and records
  rigid-water distance constraints.

`pme_alpha` is the Ewald splitting parameter in **inverse nanometres** (this
record's length unit), not inverse Ångström: 3.5 nm⁻¹ is the nm/kJ counterpart
of the 0.35 Å⁻¹ value quoted by Å-based examples, and it pairs with
`cutoff_nm = 1.0` (`alpha*rcut = 3.5`, `erfc(3.5) ≈ 7e-7`). `pme_grid = nothing`
lets `PMEPlan` size the mesh (about three grid points per `1/alpha`, rounded up
to a power of two).

Every field is validated in the constructor; an impossible protocol is a
`ChemistryError(:protocol, ...)` before any factor is built.
"""
struct SimulationProtocol
    nonbonded::Symbol
    cutoff_nm::Float64
    lj_mode::Symbol
    pme_alpha::Float64
    pme_grid::Union{Nothing,NTuple{3,Int}}
    pme_order::Int
    water_model::Symbol
    background::Symbol
    temperature_k::Float64

    function SimulationProtocol(nonbonded::Symbol, cutoff_nm::Real, lj_mode::Symbol,
                                pme_alpha::Real,
                                pme_grid::Union{Nothing,NTuple{3,Int}},
                                pme_order::Integer, water_model::Symbol,
                                background::Symbol, temperature_k::Real)
        _validate_protocol(nonbonded, Float64(cutoff_nm), lj_mode, Float64(pme_alpha),
                           pme_grid, Int(pme_order), water_model, background,
                           Float64(temperature_k))
        return new(nonbonded, Float64(cutoff_nm), lj_mode, Float64(pme_alpha),
                   pme_grid, Int(pme_order), water_model, background,
                   Float64(temperature_k))
    end
end

function _validate_protocol(nonbonded::Symbol, cutoff_nm::Float64, lj_mode::Symbol,
                            pme_alpha::Float64,
                            pme_grid::Union{Nothing,NTuple{3,Int}}, pme_order::Int,
                            water_model::Symbol, background::Symbol,
                            temperature_k::Float64)
    nonbonded in (:pme, :direct, :none) ||
        _chem_error(:protocol, "nonbonded", "must be :pme (periodic), :direct " *
                                           "(NoBox explicit pair table), or :none " *
                                           "(bonded/local only), got $(repr(nonbonded))")
    lj_mode in (:trunc, :lfs) ||
        _chem_error(:protocol, "lj_mode", "must be :trunc (reference parity) or " *
                                          ":lfs (force-consistent), got $(repr(lj_mode))")
    water_model in (:flexible, :rigid) ||
        _chem_error(:protocol, "water_model", "must be :flexible or :rigid, got " *
                                              "$(repr(water_model))")
    background in (:uniform, :none) ||
        _chem_error(:protocol, "background", "must be :uniform or :none, got " *
                                             "$(repr(background))")
    (isfinite(cutoff_nm) && cutoff_nm > 0) ||
        _chem_error(:protocol, "cutoff_nm", "must be finite and satisfy 0 < " *
                                            "cutoff_nm < Inf, got $cutoff_nm")
    pme_order in (4, 6) ||
        _chem_error(:protocol, "pme_order", "PME spline order must be 4 or 6, got $pme_order")
    (isfinite(pme_alpha) && pme_alpha > 0) ||
        _chem_error(:protocol, "pme_alpha", "must be finite and > 0, got $pme_alpha")
    if pme_grid !== nothing
        all(g -> g >= 2 * pme_order, pme_grid) ||
            _chem_error(:protocol, "pme_grid", "each grid dimension must be an " *
                                               "integer >= 2*pme_order = $(2 * pme_order), " *
                                               "got $(Tuple(pme_grid))")
    end
    (isfinite(temperature_k) && temperature_k > 0) ||
        _chem_error(:protocol, "temperature_k", "must be finite and > 0, got $temperature_k")
    return nothing
end

function _protocol_grid(pme_grid)
    pme_grid === nothing && return nothing
    (pme_grid isa Tuple || pme_grid isa AbstractVector) && length(pme_grid) == 3 ||
        _chem_error(:protocol, "pme_grid", "must be nothing or three grid dimensions, " *
                                           "got $(repr(pme_grid))")
    all(g -> g isa Integer, pme_grid) ||
        _chem_error(:protocol, "pme_grid", "grid dimensions must be integers, got " *
                                           "$(repr(pme_grid))")
    return ntuple(d -> Int(pme_grid[d]), 3)
end

function SimulationProtocol(; nonbonded = :pme, cutoff_nm = 1.0, lj_mode = :trunc,
                            pme_alpha = 3.5, pme_grid = nothing, pme_order = 4,
                            water_model = :flexible, background = :uniform,
                            temperature_k = 298.15)
    return SimulationProtocol(Symbol(nonbonded), cutoff_nm, Symbol(lj_mode), pme_alpha,
                              _protocol_grid(pme_grid), pme_order, Symbol(water_model),
                              Symbol(background), temperature_k)
end

Base.show(io::IO, p::SimulationProtocol) =
    print(io, "SimulationProtocol(nonbonded=", p.nonbonded, ", cutoff_nm=", p.cutoff_nm,
          ", lj_mode=", p.lj_mode, ", pme_alpha=", p.pme_alpha,
          ", pme_grid=", p.pme_grid === nothing ? "auto" : p.pme_grid,
          ", pme_order=", p.pme_order, ", water_model=", p.water_model,
          ", background=", p.background, ", temperature_k=", p.temperature_k, ")")

# ---------------------------------------------------------------------------
# default protocol
# ---------------------------------------------------------------------------

"For a `:pme` protocol a cutoff must fit inside one minimum image of the cell."
function _check_periodic_cutoff(ns::NormalizedSystem, protocol::SimulationProtocol)
    protocol.nonbonded === :pme || return nothing
    if ns.box.kind === :orthorhombic
        lo = minimum(ns.box.lengths_nm)
        (0 < protocol.cutoff_nm < 0.5 * lo) ||
            _chem_error(:protocol, "cutoff_nm",
                        ":pme requires 0 < cutoff_nm < min(L)/2 = $(0.5 * lo) nm for box " *
                        "lengths $(ns.box.lengths_nm); got $(protocol.cutoff_nm). The " *
                        "minimum-image/cell stencil is not valid otherwise — lower " *
                        "cutoff_nm or use a larger box.")
    elseif ns.box.kind === :triclinic
        M = ns.box.vectors_nm
        (all(isfinite, M) && det(M) > 0) ||
            _chem_error(:geometry, "box",
                        "triclinic cell matrix must be finite with positive determinant")
        widths = Types.TriclinicBox(M).widths
        lo = minimum(widths)
        (0 < protocol.cutoff_nm < 0.5 * lo) ||
            _chem_error(:protocol, "cutoff_nm",
                        ":pme requires 0 < cutoff_nm < min(width)/2 = $(0.5 * lo) nm for " *
                        "triclinic widths $(widths); got $(protocol.cutoff_nm). The " *
                        "minimum-image/cell stencil is not valid otherwise — lower " *
                        "cutoff_nm or use a larger box.")
    end
    return nothing
end

"""
    default_protocol(ns; kwargs...) -> SimulationProtocol

Choose the nonbonded model from the record's box and leave every other choice at
its documented default: orthorhombic or triclinic -> `:pme`, `NoBox` -> `:direct`.
Defaults are `cutoff_nm = 1.0`, `lj_mode = :trunc` (reference parity),
`pme_alpha = 3.5` nm⁻¹ (see [`SimulationProtocol`](@ref)), flexible water,
uniform background, 298.15 K.

Any keyword accepted by [`SimulationProtocol`](@ref) overrides the default; the
`:pme` cutoff bound `0 < cutoff_nm < min(width)/2` is checked here so a bad
default fails before any factor is built.
"""
function default_protocol(ns::NormalizedSystem; nonbonded = nothing, kwargs...)
    kind = ns.box.kind
    nb = nonbonded === nothing ? (kind === :none ? :direct : :pme) : Symbol(nonbonded)
    protocol = SimulationProtocol(; nonbonded = nb, kwargs...)
    if kind !== :none || nonbonded !== nothing || !isempty(kwargs)
        _check_periodic_cutoff(ns, protocol)
    end
    return protocol
end

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

@inline _canonical_pair(i::Integer, j::Integer) = i < j ? (Int(i), Int(j)) : (Int(j), Int(i))

function _atom_names(ns::NormalizedSystem)
    return [Symbol(a.name) for a in ns.atoms]
end

function _water_mask(ns::NormalizedSystem)
    return [Symbol(a.residue_name) in CHEM_WATER_RESIDUES for a in ns.atoms]
end

function _provenance_notes(ns::NormalizedSystem, protocol::SimulationProtocol,
                           n_factors::Int)
    nores = isempty(ns.atoms) ? 0 : maximum(a.residue_index for a in ns.atoms)
    return string("normalized mdparams v", ns.schema, " (", ns.family, "): ",
                  length(ns.atoms), " atoms, ", nores, " residues; protocol ",
                  protocol.nonbonded, " cutoff=", protocol.cutoff_nm, " nm lj_mode=",
                  protocol.lj_mode, " water=", protocol.water_model, " background=",
                  protocol.background, "; ", n_factors, " factors")
end

# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------

"""
    build_resolved(ns; protocol = default_protocol(ns)) -> ResolvedSystem

Turn one validated normalized record into a `ResolvedSystem`.

Factor order is fixed and deterministic: bonds, angles, Urey-Bradleys,
dihedrals, impropers, then `lj` + `pme` (periodic) or `nonbonded` (direct).
Component names are `bond_1...`, `angle_1...`, `urey_1...`, `dihedral_1...`,
`improper_1...` numbered over the *emitted* factors, plus `lj`/`pme` or
`nonbonded`.

Typed refusals, never silent drops:

* `:unsupported_term` — a CMAP record whose flat grid length does not match
  its resolution, or a virtual-site execution request (the latter is refused by
  `ResolvedSystem`, not here). Valid CMAP grids are built as
  `ChemistryCMAP.CMAPCorrection` factors; the schema's flat order is
  `grid[s + resolution*(t-1)]`.
* `:protocol` — a nonbonded model that does not match the box (`:pme` without an
  orthorhombic or triclinic periodic box, `:direct` with a box) or a cutoff
  outside `(0, min(width)/2)`.

Virtual sites are **passed through**, not refused: `ResolvedSystem` represents
the declaration (mass-0 site, parent coefficients) and `SystemContract` refuses
execution. A bond, angle or pair that reads a virtual-site slot is built
verbatim, so the record is inspectable rather than half-built.

Constraints come from the protocol and the record: `:rigid` turns every water
O–H bond into a `ConstraintComponent` carrying the parameter file's bond length,
and the record's own `constraints` are added in both water models (deduplicated
by pair, water bond first). `:flexible` adds no constraint of its own.

`units` is [`CHEM_UNITS`](@ref) and `declared` is exactly
`[:energy, :cartesian_forces, :component_energies]`; `provenance` carries the
record's source/force field plus a summary of the protocol used.
"""
function build_resolved(ns::NormalizedSystem;
                        protocol::SimulationProtocol = default_protocol(ns))
    validate_normalized!(ns)
    _check_periodic_cutoff(ns, protocol)

    seen_excl = Set{Tuple{Int,Int}}()
    for (k, pair) in enumerate(ns.exclusions)
        key = _canonical_pair(pair[1], pair[2])
        key in seen_excl && _chem_error(:duplicate, "exclusions[$k]",
                                        "exclusions[$k] = $pair duplicates canonical " *
                                        "exclusion pair $key")
        push!(seen_excl, key)
    end

    N = length(ns.atoms)
    masses = Float64[a.mass for a in ns.atoms]
    charges = Float64[a.charge for a in ns.atoms]
    names = _atom_names(ns)

    box = if protocol.nonbonded === :pme
        if ns.box.kind === :orthorhombic
            Types.OrthorhombicBox(ns.box.lengths_nm)
        elseif ns.box.kind === :triclinic
            Types.TriclinicBox(ns.box.vectors_nm)
        else
            _chem_error(:protocol, "box",
                        "nonbonded = :pme needs an orthorhombic or triclinic periodic box, " *
                        "but the record's box is $(ns.box.kind); use nonbonded = :direct for a " *
                        "boxless explicit pair table")
        end
    elseif protocol.nonbonded === :direct
        ns.box.kind === :none ||
            _chem_error(:protocol, "box",
                        "nonbonded = :direct is the NoBox explicit pair table (no " *
                        "cutoff, no minimum image), but the record's box is " *
                        "$(ns.box.kind); use nonbonded = :pme for a periodic cell")
        Types.NoBox()
    else
        # nonbonded === :none permits local bonded/CMAP evaluation on any box kind.
        if ns.box.kind === :none
            Types.NoBox()
        elseif ns.box.kind === :orthorhombic
            Types.OrthorhombicBox(ns.box.lengths_nm)
        else
            Types.TriclinicBox(ns.box.vectors_nm)
        end
    end

    factors = Factors.AbstractFactor[]
    component_names = Symbol[]
    sizehint!(factors, length(ns.bonds) + length(ns.angles) + length(ns.urey_bradleys) +
                      length(ns.dihedrals) + length(ns.impropers) +
                      length(ns.cmaps) + 2)

    # Bonded terms, in the one documented order. Rigid water removes a bond from
    # the *energy* and records its parameter-file length as a constraint; the
    # topology metadata below still lists every imported bond.
    water = _water_mask(ns)
    water_constraints = Tuple{Int,Int,Float64}[]
    n_bond = 0
    for b in ns.bonds
        if protocol.water_model === :rigid && water[b.i] && water[b.j]
            push!(water_constraints, (b.i, b.j, b.r0_nm))
            continue
        end
        n_bond += 1
        push!(factors, Factors.HarmonicBond(b.i, b.j, b.k, b.r0_nm))
        push!(component_names, Symbol("bond_", n_bond))
    end
    for (n, a) in enumerate(ns.angles)
        push!(factors, Factors.HarmonicAngle(a.i, a.j, a.k, a.k_theta, a.theta0_rad))
        push!(component_names, Symbol("angle_", n))
    end
    for (n, u) in enumerate(ns.urey_bradleys)
        push!(factors, ChemistryFactors.UreyBradley(u.i, u.j, u.k, u.r0_nm))
        push!(component_names, Symbol("urey_", n))
    end
    for (n, d) in enumerate(ns.dihedrals)
        if length(d.terms) == 1
            t = d.terms[1]
            push!(factors, Factors.PeriodicDihedral(d.i, d.j, d.k, d.l, t.k, t.n,
                                                    t.phase_rad))
        else
            push!(factors, ChemistryFactors.MultiPeriodicDihedral(
                d.i, d.j, d.k, d.l,
                [t.k for t in d.terms], [t.n for t in d.terms],
                [t.phase_rad for t in d.terms]))
        end
        push!(component_names, Symbol("dihedral_", n))
    end
    for (n, t) in enumerate(ns.impropers)
        push!(factors, ChemistryFactors.HarmonicImproper(t.i, t.j, t.k, t.l, t.k_phi,
                                                         t.phase_rad))
        push!(component_names, Symbol("improper_", n))
    end
    for (n, t) in enumerate(ns.cmaps)
        res = t.resolution
        res >= 3 || _chem_error(:parameter, "cmaps[$n].resolution",
                                "CMAP bicubic spline requires resolution >= 3, got $res")
        grid = reshape(t.grid, res, res)   # flat order grid[s + res*(t-1)]
        push!(factors, ChemistryCMAP.CMAPCorrection(t.indices, grid))
        push!(component_names, Symbol("cmap_", n))
    end

    if protocol.nonbonded in (:pme, :direct)
        for (k, a) in enumerate(ns.atoms)
            # A zero-size, zero-well atom (e.g. a TIP3P hydrogen) is a legitimate
            # exactly-zero LJ contribution. A zero size with a nonzero well is not
            # representable by the mixing rule and must not be approximated.
            (a.sigma_nm > 0 || (a.sigma_nm == 0 && a.epsilon_kj == 0)) ||
                _chem_error(:parameter, "atoms[$k].sigma_nm",
                            "atom $k ($(a.name)) has sigma_nm = 0 with epsilon_kj = " *
                            "$(a.epsilon_kj); a zero LJ size is only meaningful when " *
                            "the well depth is also 0 (an exactly zero LJ atom). Give " *
                            "it a positive sigma or set epsilon_kj to 0.")
        end
    end

    if protocol.nonbonded === :pme
        epsilon = Float64[a.epsilon_kj for a in ns.atoms]
        sigma = Float64[a.sigma_nm for a in ns.atoms]
        pair_topology = StreamedLJ.PairTopology(
            excluded = ns.exclusions,
            scaled = [(p.i, p.j, p.lj_scale) for p in ns.pair14])
        # The factor is built from N alone; PMEPlan/StreamedLJFactor store no
        # positions, so the zero matrix is a shape, not a geometry.
        push!(factors, StreamedLJ.StreamedLJFactor(N, epsilon, sigma, box;
                                                   rcut = protocol.cutoff_nm,
                                                   mode = protocol.lj_mode,
                                                   topology = pair_topology))
        push!(component_names, :lj)
        push!(factors, PME.PMEFactor(zeros(3, N), charges, box;
                                     alpha = protocol.pme_alpha,
                                     rcut = protocol.cutoff_nm,
                                     grid = protocol.pme_grid,
                                     order = protocol.pme_order,
                                     prefactor = CHEM_COULOMB_PREFACTOR,
                                     corrections = PME.PairCorrections(
                                         excluded = ns.exclusions,
                                         scaled = [(p.i, p.j, p.coulomb_scale)
                                                   for p in ns.pair14])))
        push!(component_names, :pme)
    elseif protocol.nonbonded === :direct
        excluded = Set(_canonical_pair(i, j) for (i, j) in ns.exclusions)
        scales = Dict(_canonical_pair(p.i, p.j) => (p.lj_scale, p.coulomb_scale)
                      for p in ns.pair14)
        ii = Int[]; jj = Int[]; sig = Float64[]; eps = Float64[]
        qq = Float64[]; ljs = Float64[]; cs = Float64[]
        for i in 1:(N - 1), j in (i + 1):N
            (i, j) in excluded && continue
            sc = get(scales, (i, j), (1.0, 1.0))
            push!(ii, i); push!(jj, j)
            push!(sig, 0.5 * (ns.atoms[i].sigma_nm + ns.atoms[j].sigma_nm))
            push!(eps, sqrt(ns.atoms[i].epsilon_kj * ns.atoms[j].epsilon_kj))
            push!(qq, ns.atoms[i].charge * ns.atoms[j].charge)
            push!(ljs, sc[1]); push!(cs, sc[2])
        end
        isempty(ii) && _chem_error(:shape, "nonbonded",
                                   "a :direct pair table needs at least one " *
                                   "non-excluded pair, but atom count $N and " *
                                   "$(length(ns.exclusions)) exclusion(s) leave none; " *
                                   "a single-atom boxless system has no nonbonded " *
                                   "energy to tabulate")
        push!(factors, ChemistryFactors.PairTableNonbonded(ii, jj, sig, eps, qq,
                                                           ljs, cs; mode = :none))
        push!(component_names, :nonbonded)
    end

    # AMBER 1-6-4 dispersion is a separate pairwise term (-C/r^4), not a change
    # to sigma/epsilon. It uses the same cutoff convention as the reference
    # (unshifted :trunc for PME, no cutoff for the direct table).
    if !isempty(ns.lj146) && protocol.nonbonded !== :none
        ci = Int[getfield(t, :i) for t in ns.lj146]
        cj = Int[getfield(t, :j) for t in ns.lj146]
        cc = Float64[t.c for t in ns.lj146]
        push!(factors, ChemistryFactors.PairC64Table(ci, cj, cc;
                    mode = protocol.nonbonded === :pme ? :trunc : :none,
                    rcut = protocol.cutoff_nm))
        push!(component_names, :lj146)
    end

    topology = Types.ChemicalTopology(
        bonds = [Types.Bond(b.i, b.j, b.k, b.r0_nm) for b in ns.bonds],
        angles = [Types.Angle(a.i, a.j, a.k, a.k_theta, a.theta0_rad) for a in ns.angles],
        # Metadata only: the first torsion term of each quadruple, which is the
        # complete term set only when the record lists one multiplicity.
        dihedrals = [Types.Dihedral(d.i, d.j, d.k, d.l, d.terms[1].k, d.terms[1].n,
                                    d.terms[1].phase_rad) for d in ns.dihedrals],
        impropers = [Types.Improper(t.i, t.j, t.k, t.l, t.k_phi, t.phase_rad)
                     for t in ns.impropers],
        exclusions = Set(_canonical_pair(i, j) for (i, j) in ns.exclusions),
        scales = Dict(_canonical_pair(p.i, p.j) => (p.lj_scale, p.coulomb_scale)
                      for p in ns.pair14))

    constraints = _build_constraints(N, protocol, water_constraints, ns.constraints)
    virtual_sites = [_RESOLVED.VirtualSite(v.kind, v.site, copy(v.parents),
                                                copy(v.coefficients))
                     for v in ns.virtual_sites]
    provenance = _RESOLVED.Provenance(
        source = ns.provenance.source,
        force_field = ns.provenance.force_field,
        notes = _provenance_notes(ns, protocol, length(factors)))

    return _RESOLVED.ResolvedSystem(;
        masses = masses, charges = charges, names = names, box = box,
        topology = topology, factors = factors, component_names = component_names,
        constraints = constraints, virtual_sites = virtual_sites, units = CHEM_UNITS,
        provenance = provenance,
        declared = [:energy, :cartesian_forces, :component_energies])
end

"""
Constraint set for a build: rigid-water bond lengths (from the parameter file) plus
the record's declared `constraints`, deduplicated by canonical pair.

The water triple wins a duplicate: both name the same imported file, and the bond
length is the one the omitted `HarmonicBond` would have used.
"""
function _build_constraints(N::Int, protocol::SimulationProtocol,
                            water_triples::Vector{Tuple{Int,Int,Float64}},
                            declared::Vector{ConstraintTerm})
    triples = Tuple{Int,Int,Float64}[]
    seen = Dict{Tuple{Int,Int},Float64}()
    if protocol.water_model === :rigid
        for (i, j, d) in water_triples
            key = _canonical_pair(i, j)
            if haskey(seen, key)
                isapprox(seen[key], d; atol = 1e-9, rtol = 1e-9) ||
                    _chem_error(:parameter, "constraints",
                                "conflicting constraint distances for pair $key: " *
                                "$(seen[key]) nm vs $d nm")
                continue
            end
            seen[key] = d
            push!(triples, (i, j, d))
        end
    end
    for c in declared
        key = _canonical_pair(c.i, c.j)
        if haskey(seen, key)
            isapprox(seen[key], c.distance_nm; atol = 1e-9, rtol = 1e-9) ||
                _chem_error(:parameter, "constraints",
                            "conflicting constraint distances for pair $key: " *
                            "$(seen[key]) nm vs $(c.distance_nm) nm")
            continue
        end
        seen[key] = c.distance_nm
        push!(triples, (c.i, c.j, c.distance_nm))
    end
    isempty(triples) && return nothing
    return try
        Constraints.CoupledConstraintSet(N, triples)
    catch e
        e isa ArgumentError || rethrow()
        _chem_error(:parameter, "constraints",
                    "the constraint set is not realizable: $(sprint(showerror, e))")
    end
end

# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

function _factor_counts(rs::_RESOLVED.ResolvedSystem)
    f = rs.factors
    return (;
        bonds = count(x -> x isa Factors.HarmonicBond, f),
        angles = count(x -> x isa Factors.HarmonicAngle, f),
        urey_bradleys = count(x -> x isa ChemistryFactors.UreyBradley, f),
        dihedrals = count(x -> x isa Factors.PeriodicDihedral ||
                                x isa ChemistryFactors.MultiPeriodicDihedral, f),
        impropers = count(x -> x isa ChemistryFactors.HarmonicImproper, f),
        cmaps = count(x -> x isa ChemistryCMAP.CMAPCorrection, f),
        lj = count(x -> x isa StreamedLJ.StreamedLJFactor, f),
        pme = count(x -> x isa PME.PMEFactor, f),
        nonbonded = count(x -> x isa ChemistryFactors.PairTableNonbonded, f),
        lj146 = count(x -> x isa ChemistryFactors.PairC64Table, f),
        total = length(f))
end

"""
    build_report(ns; protocol = default_protocol(ns)) -> NamedTuple

`(; rs, protocol, summary = normalization_summary(ns), factor_counts, warnings)`.
`rs` is the built [`ResolvedSystem`](@ref), so the report is a description of a
record that actually resolved — it carries no state and evaluates nothing.

`warnings` are strings the audit surfaces, not errors:

* `lj_mode = :trunc` is discontinuous at the cutoff, so `Dynamics` refuses it as
  a force model; it exists for reference-energy parity only.
* `background = :none` is a declaration, not a behaviour change: PME always
  includes its uniform neutralizing background term, and the builder never
  rescales charges.

An invalid periodic cutoff or nonbonded/box mismatch throws `ChemistryError(:protocol, ...)`.
"""
function build_report(ns::NormalizedSystem;
                      protocol::SimulationProtocol = default_protocol(ns))
    rs = build_resolved(ns; protocol = protocol)
    warnings = String[]
    if protocol.nonbonded === :pme && protocol.lj_mode === :trunc
        push!(warnings,
              "lj_mode = :trunc reproduces the unshifted 12-6 reference energy; it " *
              "is discontinuous at the cutoff, so the integrator refuses it " *
              "(LennardJonesPair/StreamedLJ force_consistent = false). Use " *
              "lj_mode = :lfs for dynamics.")
    end
    if protocol.nonbonded === :pme && !isempty(ns.lj146)
        push!(warnings,
              "lj146 terms under nonbonded = :pme use PairC64Table(mode = :trunc), " *
              "which is discontinuous at the cutoff (force_consistent = false) and " *
              "does not implement box_derivative; dynamics and box_derivative are " *
              "refused until a smooth/PME C4/C64 backend is used.")
    end
    if protocol.nonbonded === :pme && protocol.background === :none
        push!(warnings,
              "background = :none is a declaration only: the PME backend always " *
              "includes the uniform neutralizing background term, so a net charge is " *
              "still screened by the solver and the charges are never rescaled.")
    end
    if protocol.nonbonded === :none
        push!(warnings,
              "nonbonded = :none omits all nonbonded (LJ, PME, and lj146) factors; " *
              "only local bonded, Urey-Bradley, dihedral, improper, CMAP, and " *
              "constraint terms are evaluated.")
    end
    return (; rs = rs, protocol = protocol, summary = normalization_summary(ns),
            factor_counts = _factor_counts(rs), warnings = warnings)
end
