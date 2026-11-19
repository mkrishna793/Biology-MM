# Resolved system: the static, validated specification of a molecular problem.
#
# `ResolvedSystem` is the record a factor list, a box, a topology, a constraint set
# and a unit system are resolved into before any energy is evaluated. It is the
# boundary between "what problem is this?" and "run it": the force field is fixed
# here, the state (positions, momenta, integrator drift) is supplied per call by
# `SystemContract`.
#
# The public construction path is the keyword constructor, which owns its data and
# refuses an inconsistent record with `ResolvedSystemError` carrying a symbolic
# code. Construction is the one place where the rules are checked cheaply;
# `validate_resolved!` re-runs them on an existing record so that post-construction
# mutation of the owned arrays cannot silently change the model.
#
# The capability predicates here answer a deliberately narrow question — what can
# this *resolved specification* evaluate — and never what a future backend might
# do. `box_derivative_support` names the reason a factor has no analytic stress, so
# a pressure consumer fails at the point of use instead of substituting zeros or a
# finite difference.
#
# This prose is a comment rather than a module docstring on purpose: the module and
# the record share the name `ResolvedSystem`, and Julia's docsystem resolves a
# module docstring's name inside the module it just created, where `ResolvedSystem`
# now denotes the struct. Attaching a module docstring there fails to define. The
# record's docstring is the authoritative documentation.
module ResolvedSystem

using ..Types
using ..Types: UnsupportedChemistry, InvalidGeometry, NumericalFailure
using ..Factors
using ..Constraints
using ..StreamedLJ
using ..Electrostatics
using ..PME

export RESOLVED_SCHEMA_VERSION, ResolvedSystem, ResolvedAtoms, Provenance, VirtualSite,
       ResolvedSystemError, UnsupportedCapability, factor_support_atoms,
       engine_capabilities, box_derivative_support, validate_resolved!, n_atoms,
       virtual_site_indices, schema_version,
       restart_semantics, restart_system, validate_restart_state,
       validate_chemistry, validate_geometry,
       accumulate_box_derivative!, box_derivative,
       UnsupportedChemistry, InvalidGeometry, NumericalFailure

"""
Schema version of the `ResolvedSystem` record itself.

`v"1.0.0"` means: these fields, these units conventions, and these validation
rules. A *major* mismatch is refused because the meaning of a field changed; a
minor/patch difference is accepted because it may only add capability. This is
the schema of the specification, not of a checkpoint — see the `ResolvedSystem`
docstring for why restart state is not stored here at all.
"""
const RESOLVED_SCHEMA_VERSION = v"1.0.0"

"Deduplicated at construction; unknown capabilities are refused, not ignored."
const _CAPABILITIES = (:energy, :cartesian_forces, :component_energies, :box_derivative, :virtual_sites)

const _VIRTUAL_SITE_KINDS = (:linear, :planar, :average)

# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

"""
A resolved record violates the schema contract.

`code` is the machine-readable classification (`:schema`, `:shape`, `:mass`,
`:charge`, `:units`, `:provenance`, `:virtual_site`, `:component_name`,
`:factor_index`, `:duplicate_interaction`, `:duplicate_backend`,
`:unsupported_geometry`, `:box_mismatch`, `:charge_mismatch`,
`:conflicting_pair_rules`, `:constraint_index`, `:constraint_mass`,
`:unknown_capability`, `:overdeclared_capability`); `msg` names the offending
field, indices and the repair, because a resolution failure is almost always a
force-field transcription error and the index is the useful part.
"""
struct ResolvedSystemError <: Exception
    code::Symbol
    msg::String
end
Base.showerror(io::IO, e::ResolvedSystemError) =
    print(io, "ResolvedSystemError[", e.code, "]: ", e.msg)

"""
A capability the specification declares or the caller requests, which this engine
does not implement.

Raised at *use* (evaluation, coordinate transforms, stress) rather than at
resolution, because representing a system is not the same as being able to
execute it: a virtual-site model is exactly such a case.
"""
struct UnsupportedCapability <: Exception
    capability::Symbol
    reason::String
end
Base.showerror(io::IO, e::UnsupportedCapability) =
    print(io, "UnsupportedCapability[", e.capability, "]: ", e.reason)

_throw_error(code::Symbol, msg::AbstractString) =
    throw(ResolvedSystemError(code, String(msg)))

# ---------------------------------------------------------------------------
# record types
# ---------------------------------------------------------------------------

"""
Provenance of a resolved specification: where the numbers came from.

`source` (file/PDB/model) and `force_field` (parameter set name) must be nonempty:
an unresolved-provenance system cannot be audited, and silently defaulting to
"unspecified" would make two physically different models indistinguishable in a
run log. `notes` is free text (conversions, substitutions, known omissions).
"""
struct Provenance
    source::String
    force_field::String
    notes::String
end
Provenance(; source = "unspecified", force_field = "unspecified", notes = "") =
    Provenance(String(source), String(force_field), String(notes))

"""
A virtual site: atom `site` is not an independent degree of freedom but is placed
from `parents` with the linear weights `coefficients`.

This generic `site = Σ_p c_p · parent_p` form covers the common cases — `:linear`
(exactly 2 parents), `:planar` (exactly 3 parents) and `:average` (>= 2 parents,
equal or weighted). It deliberately does *not* encode an out-of-plane distance or
a local-frame construction (TIP4P-style sites built from a normal vector): those
need a reference geometry and a different datum, and a coefficient table cannot
be silently reinterpreted as one. A site kind outside this list is refused.

`coefficients` must sum to 1 (atol `1e-12`) so that a parent coinciding with the
site reproduces its position exactly (`c_p = 1`). The mass of `site` must be
exactly 0: a massless coordinate is what makes "this position is determined, not
integrated" true rather than an approximation.
"""
struct VirtualSite
    kind::Symbol          # :linear (2 parents), :planar (3), :average (>=2)
    site::Int
    parents::Vector{Int}
    coefficients::Vector{Float64}
end

Base.show(io::IO, v::VirtualSite) =
    print(io, "VirtualSite(", v.kind, ", site=", v.site, ", parents=", v.parents, ")")

"""
Owned per-atom data of a resolved system.

The keyword constructor of [`ResolvedSystem`](@ref) copies these vectors, so the
caller's arrays and the record do not alias. `virtual_sites` is a *declaration*
list: it changes what the atom means (a determined position, zero mass) and does
not make the apparent site an executable degree of freedom.
"""
struct ResolvedAtoms
    masses::Vector{Float64}     # owned copies
    charges::Vector{Float64}
    names::Vector{Symbol}
    virtual_sites::Vector{VirtualSite}
end

"""
    ResolvedSystem(; masses, charges, names = nothing, box = Types.NoBox(),
                     topology = Types.ChemicalTopology(),
                     factors = Factors.AbstractFactor[], component_names = nothing,
                     constraints = nothing, virtual_sites = VirtualSite[],
                     units = Types.REDUCED, provenance = Provenance(),
                     declared = [:energy, :cartesian_forces, :component_energies],
                     schema = RESOLVED_SCHEMA_VERSION)

The static specification of a molecular problem: per-atom masses/charges/names, an
ordered factor list with one name per factor, an optional coupled constraint set,
a box, a chemical topology, a unit system, provenance, and a capability
declaration. It is what an engine is asked to evaluate; it is not a trajectory.

`names` defaults to `fill(:X, N)` and `component_names` to
`[Symbol("component_", i) for i in 1:length(factors)]`. Construction validates the
record (rules `2`–`14` of the interface contract, plus the schema check) and
throws [`ResolvedSystemError`](@ref) on the first violation; `validate_resolved!`
re-runs the same rules on an existing record.

# Schema versioning

`schema` pins the *meaning* of the fields, not the numbers: field set, unit
conventions, validation rules. `RESOLVED_SCHEMA_VERSION` is `v"1.0.0"`. A different
major version is refused (`:schema`) rather than interpreted optimistically —
reading a v2 record as v1 would mis-assign a field and silently change the model.
Minor/patch differences are accepted because they may only add capability.

# Specification versus restart state

This record carries **no** positions, momenta, integrator state, step counter,
thermostat/barostat state, or accumulated time. All of that is restart state: it
is per-trajectory, changes every step, and belongs in a checkpoint record that
references this specification (by schema + provenance + a hash of the factor
list). Keeping it out means the specification can be compared, cached and
re-resolved without carrying an accidental half-updated state, and it makes
"the same model, a different trajectory" expressible.

# Explicit restart semantics (10-point contract)

On restart (resuming a trajectory from a checkpoint), the system components
follow explicit deterministic rules:
1. `topology`: INVARIANT (immutable). Chemical topology (bonds, angles, dihedrals,
   impropers) is preserved bit-identical across restarts; changing topology produces
   an incompatible restart.
2. `masses`: INVARIANT (immutable). Per-atom masses are preserved bit-identical;
   altering masses would invalidate kinetic energy and thermostat calibration.
3. `parameters`: INVARIANT (immutable). Force field parameters (force constants,
   equilibrium geometries, LJ parameters, partial charges) remain identical.
4. `exclusions`: INVARIANT (immutable). Nonbonded exclusion lists derived from
   topology are preserved unchanged.
5. `1-4 rules`: INVARIANT (immutable). 1-4 scaling factors are preserved unchanged.
6. `constraints`: INVARIANT (immutable). Target constraint distances and constraint
   graph connectivity are preserved; restart coordinates/momenta are validated/projected
   against them (`validate_restart_state`).
7. `virtual sites`: INVARIANT (immutable). Virtual site parent indices and linear
   coefficients are preserved; virtual site coordinates are reconstructed deterministically.
8. `box`: STATE-DEPENDENT. In NVE/NVT ensembles, the box matrix is preserved unchanged.
   In NPT ensembles with barostat volume moves, the box matrix updates from the checkpoint,
   and any periodic backend factors whose stencils cache cell geometry (`StreamedLJFactor`,
   `PMEFactor`) are rebuilt for the new box (`restart_system`).
9. `provenance`: LINEAGE-EXTENDED. Model source and force field name are preserved;
   continuation lineage (parent checkpoint, step index, timestamp) is appended to notes.
10. `units`: INVARIANT (immutable). Unit system conventions are preserved bit-identical.

Use [`restart_semantics`](@ref) to programmatically query this contract,
[`restart_system`](@ref) to construct a restart-ready specification, and
[`validate_restart_state`](@ref) to validate checkpoint coordinates and momenta.

# Ownership, shallow immutability and aliasing

The keyword constructor **copies** `masses`, `charges`, `names`,
`component_names` and `virtual_sites` (including each site's `parents` and
`coefficients`), copies the `factors` container, and copies `declared`. The
constraint set is rebuilt into an owned copy (new component objects with copied
`atoms`/`edges` vectors). The field types are concrete `Vector`s, so a field can
be replaced but the *arrays* are only as immutable as their contents: mutating
`rs.atoms.masses[1]` in place is possible, and is precisely why
`validate_resolved!` exists.

The factor *objects* are shared, never deep-copied: a `StreamedLJFactor` owns a
cell-list workspace and a `PMEFactor` owns FFT plans, so copying them would either
be very expensive or produce a factor whose cached geometry disagrees with its
copy. Sharing is correct because factors are configuration-agnostic apart from
their fixed box and atom count, which resolution checks.

`copy(rs)` is a **shallow** copy by design: it shares `atoms.masses`,
`atoms.charges`, `atoms.names` and the factor objects, and copies the containers
(`virtual_sites` vector and each site's arrays, `component_names`, `declared`, the
factor and constraint-component containers). Hence
`copy(rs).atoms.masses === rs.atoms.masses`, and an in-place mutation of that
array is visible through both. That is documented, not accidental: a deep copy
would hide exactly the aliasing that a caller's reuse of a mass vector depends on.
Use `validate_resolved!` after mutating shared state.

# Virtual sites are representable, not executable

A massless atom with a `VirtualSite` declaration is a legitimate specification
(this is how a TIP4P/4P-type water model is written down): it resolves, and its
geometry is inspectable. Execution is refused explicitly — `SystemContract.energy`
and `to_atomset` throw `UnsupportedCapability(:virtual_sites, ...)` — because
integrating or reporting forces on a determined, massless coordinate would invent
a degree of freedom. A massless atom *without* a declaration is refused at
resolution (`:mass`).

# Triclinic geometry

`TriclinicBox` is accepted for local factors, `StreamedLJ.StreamedLJFactor`, and
`PME.PMEFactor` (with `0 < rcut < 0.5 * min(box.widths)`). Only the direct-Ewald
backend `Electrostatics.EwaldFactor` remains orthorhombic-only: resolution refuses
it on `TriclinicBox` with `:unsupported_geometry`. A factor built for a different
box is refused with `:box_mismatch`.

# Declared versus engine capabilities

`declared` is the caller's claim of what this problem needs; `engine_capabilities`
is what the factor list can actually deliver. The two are checked against each
other at construction: a declared capability the engine cannot provide is refused
(`:overdeclared_capability`) with the concrete per-capability reason, so a
pressure/stress consumer learns at resolution rather than receiving a zero, a
finite-difference surrogate, or a different physics model.

# Shallow record plus explicit validation

`validate_resolved!` works on the existing record, so it catches corruption of
the owned arrays and of mutable factors after construction; it returns the record
on success and throws on the first violation.
"""
struct ResolvedSystem
    schema::VersionNumber
    atoms::ResolvedAtoms
    box::Types.AbstractBox
    topology::Types.ChemicalTopology
    factors::Vector{Factors.AbstractFactor}
    component_names::Vector{Symbol}
    constraints::Union{Nothing,Constraints.CoupledConstraintSet}
    units::Types.Units
    provenance::Provenance
    declared::Set{Symbol}
end

Base.show(io::IO, rs::ResolvedSystem) =
    print(io, "ResolvedSystem(schema=", rs.schema, ", N=", n_atoms(rs),
          ", factors=", length(rs.factors), ", box=", _box_label(rs.box),
          ", declared=", sort!(collect(rs.declared)), ")")

_box_label(::Types.NoBox) = "none"
_box_label(b::Types.OrthorhombicBox) = string(b.L)
_box_label(b::Types.TriclinicBox) = "triclinic"

# ---------------------------------------------------------------------------
# construction: owned copies of caller data
# ---------------------------------------------------------------------------

function _copy_real_vector(x, what::AbstractString, code::Symbol)
    x isa AbstractVector ||
        _throw_error(code, "$what must be a vector of real numbers, got $(typeof(x))")
    out = Vector{Float64}(undef, length(x))
    for (k, v) in enumerate(x)
        v isa Real ||
            _throw_error(code, "$what[$k] must be a real number, got $(repr(v))")
        out[k] = Float64(v)
    end
    return out
end

_as_symbol(x::Symbol) = x
_as_symbol(x::AbstractString) = Symbol(x)
_as_symbol(x) = nothing

function _copy_names(names, N::Int)
    names isa AbstractVector ||
        _throw_error(:shape, "names must be a vector of Symbols, got $(typeof(names))")
    length(names) == N ||
        _throw_error(:shape, "names has length $(length(names)) but masses has " *
                             "length $N; supply one name per atom or leave names unset")
    out = Vector{Symbol}(undef, N)
    for (k, n) in enumerate(names)
        s = _as_symbol(n)
        s === nothing &&
            _throw_error(:shape, "names[$k] must be a Symbol (or a String), got " *
                                 "$(repr(n)); one name per atom is required")
        out[k] = s
    end
    return out
end

function _copy_component_names(names)
    names isa AbstractVector ||
        _throw_error(:component_name, "component_names must be a vector of Symbols, " *
                                      "got $(typeof(names))")
    out = Vector{Symbol}(undef, length(names))
    for (k, n) in enumerate(names)
        s = _as_symbol(n)
        s === nothing &&
            _throw_error(:component_name, "component_names[$k] must be a Symbol (or a " *
                                          "String), got $(repr(n))")
        out[k] = s
    end
    return out
end

function _copy_factors(factors)
    factors isa AbstractVector ||
        _throw_error(:factor_index, "factors must be a vector of " *
                                    "Factors.AbstractFactor, got $(typeof(factors))")
    out = Vector{Factors.AbstractFactor}(undef, length(factors))
    for (k, f) in enumerate(factors)
        f isa Factors.AbstractFactor ||
            _throw_error(:factor_index, "factors[$k] is $(typeof(f)); each entry must " *
                                        "be a Factors.AbstractFactor")
        out[k] = f
    end
    return out
end

_copy_virtual_site(v::VirtualSite) =
    VirtualSite(v.kind, v.site, copy(v.parents), copy(v.coefficients))

function _copy_virtual_sites(sites)
    sites isa AbstractVector ||
        _throw_error(:virtual_site, "virtual_sites must be a vector of VirtualSite, " *
                                    "got $(typeof(sites))")
    out = VirtualSite[]
    sizehint!(out, length(sites))
    for (k, vs) in enumerate(sites)
        vs isa VirtualSite ||
            _throw_error(:virtual_site, "virtual_sites[$k] is $(typeof(vs)); expected " *
                                        "a VirtualSite(kind, site, parents, coefficients)")
        push!(out, _copy_virtual_site(vs))
    end
    return out
end

_copy_constraints(::Nothing) = nothing

function _copy_constraints(cs::Constraints.CoupledConstraintSet)
    comps = [Constraints.ConstraintComponent(copy(c.atoms), copy(c.edges))
             for c in cs.components]
    return Constraints.CoupledConstraintSet(comps, cs.dim)
end

_copy_constraints(other) =
    _throw_error(:constraint_index, "constraints must be nothing or a " *
                                    "Constraints.CoupledConstraintSet, got $(typeof(other))")

function _copy_declared(declared)
    out = Set{Symbol}()
    for cap in declared
        cap isa Symbol ||
            _throw_error(:unknown_capability, "declared capabilities must be Symbols, " *
                                              "got $(typeof(cap)) ($(repr(cap)))")
        push!(out, cap)
    end
    return out
end

function ResolvedSystem(; masses, charges, names = nothing, box = Types.NoBox(),
                        topology = Types.ChemicalTopology(),
                        factors = Factors.AbstractFactor[], component_names = nothing,
                        constraints = nothing, virtual_sites = VirtualSite[],
                        units = Types.REDUCED, provenance = Provenance(),
                        declared = [:energy, :cartesian_forces, :component_energies],
                        schema = RESOLVED_SCHEMA_VERSION)
    schema isa VersionNumber ||
        _throw_error(:schema, "schema must be a VersionNumber such as v\"1.0.0\"; got " *
                              "$(repr(schema))")
    schema.major == RESOLVED_SCHEMA_VERSION.major ||
        _throw_error(:schema, "schema $(schema) is not supported: this build " *
                              "implements $(RESOLVED_SCHEMA_VERSION) and accepts " *
                              "major version $(RESOLVED_SCHEMA_VERSION.major) only. " *
                              "A major change means field meanings changed, so the " *
                              "record must be re-resolved, not reinterpreted.")

    masses_v = _copy_real_vector(masses, "masses", :mass)
    charges_v = _copy_real_vector(charges, "charges", :charge)
    N = length(masses_v)
    names_v = names === nothing ? fill(:X, N) : _copy_names(names, N)
    factors_v = _copy_factors(factors)
    comp_names_v = component_names === nothing ?
        [Symbol("component_", i) for i in 1:length(factors_v)] :
        _copy_component_names(component_names)

    atoms = ResolvedAtoms(masses_v, charges_v, names_v, _copy_virtual_sites(virtual_sites))
    rs = ResolvedSystem(schema, atoms, box, topology, factors_v, comp_names_v,
                        _copy_constraints(constraints), units, provenance,
                        _copy_declared(declared))
    return validate_resolved!(rs)
end

"""
Shallow copy: shares the atom arrays and the factor objects, copies the
containers. `copy(rs).atoms.masses === rs.atoms.masses` holds, and an in-place
mutation of a shared array is visible through both copies. Does not revalidate —
call [`validate_resolved!`](@ref) on the copy after mutating shared state.
"""
function Base.copy(rs::ResolvedSystem)
    atoms = ResolvedAtoms(rs.atoms.masses, rs.atoms.charges, rs.atoms.names,
                          [_copy_virtual_site(v) for v in rs.atoms.virtual_sites])
    cs = rs.constraints === nothing ? nothing :
         Constraints.CoupledConstraintSet(copy(rs.constraints.components),
                                          rs.constraints.dim)
    return ResolvedSystem(rs.schema, atoms, rs.box, rs.topology, copy(rs.factors),
                          copy(rs.component_names), cs, rs.units,
                          Provenance(rs.provenance.source, rs.provenance.force_field,
                                     rs.provenance.notes),
                          copy(rs.declared))
end

# ---------------------------------------------------------------------------
# accessors
# ---------------------------------------------------------------------------

"Number of atom slots in the resolved system, including declared virtual sites."
n_atoms(rs::ResolvedSystem) = length(rs.atoms.masses)

"Atom indices that carry a `VirtualSite` declaration."
function virtual_site_indices(rs::ResolvedSystem)
    out = Set{Int}()
    for v in rs.atoms.virtual_sites
        push!(out, v.site)
    end
    return out
end

"Schema version of the record (see `RESOLVED_SCHEMA_VERSION`)."
schema_version(rs::ResolvedSystem) = rs.schema

"The static read-support of one factor, as a plain `Vector{Int}`."
factor_support_atoms(f::Factors.AbstractFactor) = collect(Int, Factors.read_support(f))

"Concise label for a component, tolerating a not-yet-validated name vector."
_component_label(rs::ResolvedSystem, k::Int) =
    k <= length(rs.component_names) ? string(rs.component_names[k]) : "component $k"

# ---------------------------------------------------------------------------
# capability predicates
# ---------------------------------------------------------------------------

"""
    accumulate_box_derivative!(G, f, q, ctx)

Extension point: accumulate the fixed-fractional box derivative `G[a,b] += ∂U_f/∂h[a,b]`
for a single factor `f`. Factors may implement this method (or `box_derivative(f, q, ctx)`)
to participate in `SystemContract.box_derivative`.
"""
function accumulate_box_derivative! end

"""
    box_derivative(f::Factors.AbstractFactor, q, ctx) -> Matrix{Float64}

Extension point: compute the 3x3 fixed-fractional box derivative for factor `f`.
"""
function box_derivative end

function _has_factor_box_derivative(f::Factors.AbstractFactor)
    T = typeof(f)
    if hasmethod(accumulate_box_derivative!, Tuple{Matrix{Float64}, T, Matrix{Float64}, Factors.Context}) ||
       hasmethod(accumulate_box_derivative!, Tuple{Matrix{Float64}, T, AbstractMatrix{Float64}, Factors.Context})
        m = which(accumulate_box_derivative!, Tuple{Matrix{Float64}, T, Matrix{Float64}, Factors.Context})
        p = m.sig.parameters[3]
        return p !== Factors.AbstractFactor && p !== Any
    end
    if hasmethod(box_derivative, Tuple{T, Matrix{Float64}, Factors.Context}) ||
       hasmethod(box_derivative, Tuple{T, AbstractMatrix{Float64}, Factors.Context})
        m = which(box_derivative, Tuple{T, Matrix{Float64}, Factors.Context})
        p = m.sig.parameters[2]
        return p !== Factors.AbstractFactor && p !== Any
    end
    return false
end

"""
    box_derivative_support(f) -> Union{Nothing,String}

`nothing` when `f` has an analytic derivative with respect to the box matrix `h`
at fixed fractional coordinates; otherwise a reason string.

A reason string is a refusal, not a promise: `SystemContract.box_derivative`
turns it into `MissingBoxDerivative` instead of returning a zero matrix or a
finite-difference surrogate, either of which would be a different observable.
"""
function box_derivative_support(f::Factors.AbstractFactor)
    _has_factor_box_derivative(f) && return nothing
    return "no analytic box derivative implemented for $(nameof(typeof(f)))"
end
box_derivative_support(::Factors.HarmonicBond) = nothing
box_derivative_support(::Factors.HarmonicAngle) = nothing
box_derivative_support(::Factors.PeriodicDihedral) = nothing
box_derivative_support(::Factors.BipartiteHarmonicGroup) = nothing
box_derivative_support(f::Factors.LennardJonesPair) =
    f.mode in (:none, :lfs) ? nothing : "mode $(f.mode) is not force-consistent at the cutoff"

function box_derivative_support(f::Factors.CoulombPair)
    _has_factor_box_derivative(f) && return nothing
    return "CoulombPair is non-periodic only; no periodic box derivative exists"
end

function box_derivative_support(f::StreamedLJ.StreamedLJFactor)
    f.mode === :lfs || return "StreamedLJ mode $(f.mode) is not C1/force-consistent"
    return nothing
end

box_derivative_support(::PME.PMEFactor) = nothing

function box_derivative_support(f::Electrostatics.EwaldFactor)
    _has_factor_box_derivative(f) && return nothing
    return "direct-Ewald box derivative (stress) is not implemented"
end

"Periodic backends whose stencil is only valid on an orthorhombic cell."
_requires_orthorhombic(::Factors.AbstractFactor) = false
_requires_orthorhombic(::Electrostatics.EwaldFactor) = true

"""
Reason the whole system has no analytic box derivative, or `nothing` when every
factor and the box support one. Single source of truth for both
[`engine_capabilities`](@ref) and the `:overdeclared_capability` message.
"""
function _box_derivative_refusal(rs::ResolvedSystem)
    rs.box isa Types.NoBox &&
        return "no periodic box, so there is no lattice matrix h to differentiate"
    if !(rs.box isa Types.OrthorhombicBox) && !(rs.box isa Types.TriclinicBox)
        return "the box $(nameof(typeof(rs.box))) provides no lattice matrix for a " *
               "box derivative"
    end
    if rs.box isa Types.TriclinicBox
        for (k, f) in enumerate(rs.factors)
            _requires_orthorhombic(f) || continue
            return "component $(_component_label(rs, k)) ($(nameof(typeof(f)))) is an " *
                   "orthorhombic-only periodic backend and refuses triclinic cells"
        end
    end
    for (k, f) in enumerate(rs.factors)
        reason = box_derivative_support(f)
        reason === nothing && continue
        return "component $(_component_label(rs, k)) ($(nameof(typeof(f)))): $reason"
    end
    return nothing
end

"""
    engine_capabilities(rs) -> Set{Symbol}

What this resolved factor list can deliver: always `(:energy,
:cartesian_forces, :component_energies)`, plus `:box_derivative` only when the box
is periodic, every factor has an analytic box derivative, and no orthorhombic-only
backend is paired with a triclinic cell.

This is the engine side of the declared-versus-engine check; it says nothing
about whether the *caller* needs those capabilities.
"""
function engine_capabilities(rs::ResolvedSystem)
    caps = Set{Symbol}((:energy, :cartesian_forces, :component_energies))
    _box_derivative_refusal(rs) === nothing && push!(caps, :box_derivative)
    if !isempty(rs.atoms.virtual_sites) && (:virtual_sites in rs.declared)
        push!(caps, :virtual_sites)
    end
    return caps
end

# ---------------------------------------------------------------------------
# validation: rules 2-14, re-runnable on an existing record
# ---------------------------------------------------------------------------

"""
    validate_resolved!(rs) -> rs

Re-run the resolution rules on an existing record and return it, or throw
[`ResolvedSystemError`](@ref) on the first violation. The constructor calls this
before returning, so a system that exists has passed it at least once; calling it
again is the supported way to detect post-construction mutation of the owned
arrays or of a mutable factor (the factors are shared, not copied).

The schema rule is a constructor-time check because `schema` is a non-mutable
field; everything else (shape, masses, charges, units, provenance, virtual sites,
component names, factor support, duplicate interactions, backend families,
periodic consistency, constraints, declared capabilities) is re-checked here. The
massless-site rule is enforced in the mass check, so a non-massless virtual-site
atom reports code `:mass`.
"""
function validate_resolved!(rs::ResolvedSystem)
    N = _check_shape(rs)
    _check_masses(rs, N)
    _check_charges(rs)
    _check_units(rs)
    _check_provenance(rs)
    _check_virtual_sites(rs, N)
    _check_component_names(rs)
    supports = _check_factor_supports(rs, N)
    _check_duplicate_interactions(rs, supports)
    _check_backend_families(rs)
    _check_periodic_consistency(rs, N)
    _check_constraints(rs, N)
    _check_declared(rs)
    return rs
end

# rule 2
function _check_shape(rs::ResolvedSystem)
    N = length(rs.atoms.masses)
    N >= 1 ||
        _throw_error(:shape, "a resolved system needs at least one atom, got N=0; " *
                             "pass masses and charges for at least one atom")
    nc = length(rs.atoms.charges)
    nc == N ||
        _throw_error(:shape, "charges has length $nc but masses has length $N; both " *
                             "must describe the same N atoms")
    nn = length(rs.atoms.names)
    nn == N ||
        _throw_error(:shape, "names has length $nn but masses has length $N; supply " *
                             "one Symbol per atom")
    return N
end

# rule 3 (including the mass-0 requirement for declared virtual sites)
function _check_masses(rs::ResolvedSystem, N::Int)
    sites = virtual_site_indices(rs)
    for k in 1:N
        m = rs.atoms.masses[k]
        isfinite(m) ||
            _throw_error(:mass, "mass[$k] = $m is not finite; every mass must be a " *
                                "finite number in the system's mass unit")
        if k in sites
            m == 0.0 ||
                _throw_error(:mass, "atom $k is the site of a virtual-site " *
                                    "declaration and must be massless (mass 0), got " *
                                    "$m; a placed site is not an integrated degree of " *
                                    "freedom, so it carries no kinetic energy")
        else
            m > 0 ||
                _throw_error(:mass, "atom $k has mass $m; every atom without a " *
                                    "virtual-site declaration needs a strictly positive " *
                                    "mass (a deliberately massless site must be declared " *
                                    "as a VirtualSite)")
        end
    end
    return nothing
end

# rule 4
function _check_charges(rs::ResolvedSystem)
    for k in eachindex(rs.atoms.charges)
        z = rs.atoms.charges[k]
        isfinite(z) ||
            _throw_error(:charge, "charge[$k] = $z is not finite; a non-finite charge " *
                                  "poisons every Coulomb term")
    end
    return nothing
end

# rule 5
function _check_units(rs::ResolvedSystem)
    u = rs.units
    for (name, v) in (("length", u.length), ("energy", u.energy), ("mass", u.mass),
                      ("charge", u.charge), ("boltzmann", u.boltzmann))
        (isfinite(v) && v > 0) ||
            _throw_error(:units, "units.$name = $v; every unit scale must be finite " *
                                 "and strictly positive, otherwise energies, masses " *
                                 "and temperatures are not comparable quantities")
    end
    isfinite(u.coulomb_prefactor) ||
        _throw_error(:units, "units.coulomb_prefactor = $(u.coulomb_prefactor) is not " *
                             "finite")
    if u.coulomb_prefactor <= 0 &&
       any(f -> f isa Factors.CoulombPair, rs.factors)
        _throw_error(:units, "this system contains a CoulombPair but " *
                             "units.coulomb_prefactor = $(u.coulomb_prefactor) is not " *
                             "positive; every Coulomb energy would be zero or " *
                             "sign-flipped. Set the prefactor the charge unit implies " *
                             "(REDUCED uses 1.0, SI uses 1/(4*pi*eps0)).")
    end
    return nothing
end

# rule 6
function _check_provenance(rs::ResolvedSystem)
    isempty(strip(rs.provenance.source)) &&
        _throw_error(:provenance, "provenance.source is empty; name the file, model " *
                                  "or experiment the coordinates and parameters came " *
                                  "from (a run log that cannot identify its source is " *
                                  "not auditable)")
    isempty(strip(rs.provenance.force_field)) &&
        _throw_error(:provenance, "provenance.force_field is empty; name the " *
                                  "parameter set (two physical models differ only in " *
                                  "their parameters, so this is not optional metadata)")
    return nothing
end

# rule 7
function _check_virtual_sites(rs::ResolvedSystem, N::Int)
    seen = Set{Int}()
    for v in rs.atoms.virtual_sites
        tag = "virtual site at atom $(v.site)"
        v.kind in _VIRTUAL_SITE_KINDS ||
            _throw_error(:virtual_site, "$tag has kind $(repr(v.kind)); supported " *
                                        "kinds are :linear (2 parents), :planar " *
                                        "(3 parents) and :average (>= 2 parents). " *
                                        "A kind needing a normal vector or an " *
                                        "out-of-plane distance is not representable " *
                                        "by a coefficient table and is refused.")
        1 <= v.site <= N ||
            _throw_error(:virtual_site, "$tag is outside 1:$N; the site index must " *
                                        "name an atom slot of this system")
        v.site in seen &&
            _throw_error(:virtual_site, "atom $(v.site) carries more than one virtual " *
                                        "site; this record represents one site per " *
                                        "atom slot")
        push!(seen, v.site)

        np = length(v.parents)
        if v.kind === :linear
            np == 2 ||
                _throw_error(:virtual_site, "$tag is :linear, which interpolates " *
                                            "exactly two parents; got $np parents " *
                                            "$(v.parents)")
        elseif v.kind === :planar
            np == 3 ||
                _throw_error(:virtual_site, "$tag is :planar, which spans exactly " *
                                            "three parents; got $np parents $(v.parents)")
        else
            np >= 2 ||
                _throw_error(:virtual_site, "$tag is :average, which needs at least " *
                                            "two parents; got $np parent(s)")
        end
        for (k, p) in enumerate(v.parents)
            1 <= p <= N ||
                _throw_error(:virtual_site, "$tag parent $k = $p is outside 1:$N")
            p == v.site &&
                _throw_error(:virtual_site, "$tag lists itself as parent $k; a site " *
                                            "cannot be interpolated from itself")
        end
        length(unique(v.parents)) == np ||
            _throw_error(:virtual_site, "$tag repeats a parent index in $(v.parents); " *
                                        "the coefficients of a repeated parent are not " *
                                        "identifiable, so the declaration is ambiguous")
        length(v.coefficients) == np ||
            _throw_error(:virtual_site, "$tag has $(np) parents but " *
                                        "$(length(v.coefficients)) coefficients; one " *
                                        "weight per parent is required")
        all(isfinite, v.coefficients) ||
            _throw_error(:virtual_site, "$tag has a non-finite coefficient in " *
                                        "$(v.coefficients)")
        s = sum(v.coefficients)
        isapprox(s, 1.0; atol = 1e-12, rtol = 0.0) ||
            _throw_error(:virtual_site, "$tag coefficients sum to $s, not 1 (atol " *
                                        "1e-12); the interpolation must reproduce a " *
                                        "parent position when it sits on that parent")
    end
    return nothing
end

# rule 8
function _check_component_names(rs::ResolvedSystem)
    nf = length(rs.factors)
    nc = length(rs.component_names)
    nc == nf ||
        _throw_error(:component_name, "component_names has $nc entries but there are " *
                                      "$nf factors; component_energies is keyed by " *
                                      "name, so every factor needs exactly one name")
    first_at = Dict{Symbol,Int}()
    for (k, name) in enumerate(rs.component_names)
        if haskey(first_at, name)
            _throw_error(:component_name, "component name $(repr(name)) is used by " *
                                          "factors $(first_at[name]) and $k; " *
                                          "component energies are keyed by name, so " *
                                          "duplicate names would overwrite a component")
        end
        first_at[name] = k
    end
    return nothing
end

# rule 9
function _check_factor_supports(rs::ResolvedSystem, N::Int)
    supports = Vector{Vector{Int}}(undef, length(rs.factors))
    for (k, f) in enumerate(rs.factors)
        type_name = nameof(typeof(f))
        hasmethod(Factors.read_support, Tuple{typeof(f)}) ||
            _throw_error(:factor_index, "factor $k ($type_name) does not implement " *
                                        "Factors.read_support, so its read-support is " *
                                        "unknown and overlapping interactions cannot be " *
                                        "detected; implement Factors.read_support(f) " *
                                        "returning the atom indices it reads")
        sup = factor_support_atoms(f)
        isempty(sup) &&
            _throw_error(:factor_index, "factor $k ($type_name) has an empty " *
                                        "read-support; a factor must read at least one " *
                                        "atom")
        for a in sup
            1 <= a <= N ||
                _throw_error(:factor_index, "factor $k ($type_name) reads atom $a, " *
                                            "outside 1:$N; a factor index outside the " *
                                            "atom set would write outside the force " *
                                            "accumulator")
        end
        length(unique(sup)) == length(sup) ||
            _throw_error(:factor_index, "factor $k ($type_name) repeats an atom in " *
                                        "its read-support $(sup); the repeated atom " *
                                        "would be counted twice in the local gradient")
        supports[k] = sup
    end
    return supports
end

# rule 10: the same interaction must not be evaluated twice. Sorting the support
# makes (1,2) and (2,1) the same key for pair factors, which is the whole point:
# a reversed duplicate is the same physical pair.
function _check_duplicate_interactions(rs::ResolvedSystem, supports::Vector{Vector{Int}})
    seen = Dict{Tuple{DataType,Tuple{Vararg{Int}}},Int}()
    for (k, f) in enumerate(rs.factors)
        # Pair factors are symmetric in their two atoms, so (1,2) and (2,1) are the
        # same interaction. Three- and four-atom factors are not: the vertex/axis
        # roles are ordered, so (1,2,3) and (2,1,3) are different angles.
        support = supports[k]
        key = (typeof(f), length(support) == 2 ? Tuple(sort(support)) : Tuple(support))
        if haskey(seen, key)
            _throw_error(:duplicate_interaction, "factors $(seen[key]) and $k both " *
                                                 "evaluate $(nameof(typeof(f))) over " *
                                                 "atoms $(key[2]); the interaction " *
                                                 "would be counted twice. Merge or " *
                                                 "remove one of them.")
        end
        seen[key] = k
    end
    return nothing
end

# rule 11: a shared backend may appear at most once. PME and direct Ewald are one
# physical model (the same periodic Coulomb sum), so they are one family even
# though they are different types.
_backend_family(::Factors.AbstractFactor) = :local
_backend_family(::Factors.BipartiteHarmonicGroup) = :bipartite_harmonic
_backend_family(::StreamedLJ.StreamedLJFactor) = :streamed_lj
_backend_family(::PME.PMEFactor) = :long_range_electrostatics
_backend_family(::Electrostatics.EwaldFactor) = :long_range_electrostatics

function _family_label(fam::Symbol)
    fam === :streamed_lj && return "streamed-LJ"
    fam === :bipartite_harmonic && return "bipartite-harmonic"
    fam === :long_range_electrostatics && return "long-range electrostatic"
    return string(fam)
end

function _check_backend_families(rs::ResolvedSystem)
    owner = Dict{Symbol,Int}()
    for (k, f) in enumerate(rs.factors)
        fam = _backend_family(f)
        fam === :local && continue
        if haskey(owner, fam)
            prev = owner[fam]
            _throw_error(:duplicate_backend, "factors $prev " *
                                             "($(nameof(typeof(rs.factors[prev])))) " *
                                             "and $k ($(nameof(typeof(f)))) both drive " *
                                             "the $(_family_label(fam)) backend; at most " *
                                             "one factor per backend family is " *
                                             "supported (PME and direct Ewald are the " *
                                             "same long-range electrostatic family)")
        end
        owner[fam] = k
    end
    return nothing
end

# rule 12
function _check_periodic_consistency(rs::ResolvedSystem, N::Int)
    for (k, f) in enumerate(rs.factors)
        if f isa StreamedLJ.StreamedLJFactor
            _check_periodic_backend(rs, k, f)
        elseif f isa PME.PMEFactor
            _check_periodic_backend(rs, k, f)
            _check_backend_charges(rs, k, f)
        elseif f isa Electrostatics.EwaldFactor
            _check_periodic_backend(rs, k, f)
            _check_backend_charges(rs, k, f)
        elseif f isa Factors.CoulombPair
            _check_coulomb_pair_charges(rs, k, f)
        end
    end
    _check_pair_topologies(rs, N)
    return nothing
end

function _check_periodic_backend(rs::ResolvedSystem, k::Int, f)
    label = _component_label(rs, k)
    type_name = nameof(typeof(f))
    if f isa Electrostatics.EwaldFactor
        rs.box isa Types.OrthorhombicBox ||
            _throw_error(:unsupported_geometry, "component $label ($type_name) is an " *
                                                "orthorhombic-only periodic backend, but " *
                                                "the system box is $(typeof(rs.box)). Its " *
                                                "stencil is not lattice-correct for a " *
                                                "skewed cell, so it is refused instead of " *
                                                "being run on a geometry it was not " *
                                                "derived for.")
        rs.box.L == f.box.L ||
            _throw_error(:box_mismatch, "component $label ($type_name) was built for box " *
                                        "lengths $(f.box.L) but the system box is " *
                                        "$(rs.box.L); the cached cell grid, cutoff and " *
                                        "pair topology are only valid for the box they " *
                                        "were built from, so rebuild the factor")
        return nothing
    end

    (rs.box isa Types.OrthorhombicBox || rs.box isa Types.TriclinicBox) ||
        _throw_error(:unsupported_geometry, "component $label ($type_name) requires a " *
                                            "periodic OrthorhombicBox or TriclinicBox, but " *
                                            "the system box is $(typeof(rs.box)).")

    if rs.box isa Types.OrthorhombicBox
        f.box isa Types.OrthorhombicBox ||
            _throw_error(:box_mismatch, "component $label ($type_name) was built for a " *
                                        "$(typeof(f.box)) box but the system box is " *
                                        "OrthorhombicBox $(rs.box.L); rebuild the factor")
        rs.box.L == f.box.L ||
            _throw_error(:box_mismatch, "component $label ($type_name) was built for box " *
                                        "lengths $(f.box.L) but the system box is " *
                                        "$(rs.box.L); the cached cell grid, cutoff and " *
                                        "pair topology are only valid for the box they " *
                                        "were built from, so rebuild the factor")
        min_w = minimum(rs.box.L)
    else
        f.box isa Types.TriclinicBox ||
            _throw_error(:box_mismatch, "component $label ($type_name) was built for a " *
                                        "$(typeof(f.box)) box but the system box is " *
                                        "TriclinicBox; rebuild the factor")
        rs.box.H == f.box.H ||
            _throw_error(:box_mismatch, "component $label ($type_name) was built for a " *
                                        "different triclinic lattice matrix than the " *
                                        "system box; the cached cell grid, cutoff and " *
                                        "pair topology are only valid for the box they " *
                                        "were built from, so rebuild the factor")
        min_w = minimum(rs.box.widths)
    end

    rc = f isa PME.PMEFactor ? f.plan.rcut : f.rcut
    (isfinite(rc) && rc > 0 && rc < 0.5 * min_w) ||
        _throw_error(:unsupported_geometry, "component $label ($type_name) cutoff " *
                                            "rcut=$rc must satisfy 0 < rcut < 0.5 * min " *
                                            "perpendicular width = $(0.5 * min_w)")
    return nothing
end

function _check_backend_charges(rs::ResolvedSystem, k::Int, f)
    label = _component_label(rs, k)
    type_name = nameof(typeof(f))
    nz = length(f.charges)
    nz == length(rs.atoms.charges) ||
        _throw_error(:charge_mismatch, "component $label ($type_name) holds $nz " *
                                       "charges but the system has " *
                                       "$(length(rs.atoms.charges)) atoms; a backend " *
                                       "built for a different atom count would " *
                                       "evaluate a different system")
    for a in eachindex(f.charges)
        isapprox(f.charges[a], rs.atoms.charges[a]; rtol = 1e-12, atol = 1e-15) ||
            _throw_error(:charge_mismatch, "component $label ($type_name) carries " *
                                           "charge $(a) = $(f.charges[a]) but the " *
                                           "resolved system charges atom $a with " *
                                           "$(rs.atoms.charges[a]); the backend would " *
                                           "evaluate electrostatics for a different " *
                                           "charge distribution")
    end
    return nothing
end

function _check_coulomb_pair_charges(rs::ResolvedSystem, k::Int, f::Factors.CoulombPair)
    label = _component_label(rs, k)
    for (atom, q) in ((f.i, f.qi), (f.j, f.qj))
        isapprox(q, rs.atoms.charges[atom]; rtol = 1e-12, atol = 1e-15) ||
            _throw_error(:charge_mismatch, "component $label (CoulombPair) stores " *
                                           "q$(atom == f.i ? 'i' : 'j') = $q for " *
                                           "atom $atom, but the resolved system " *
                                           "charges that atom with " *
                                           "$(rs.atoms.charges[atom]); the pair " *
                                           "factor duplicates per-atom charge data, " *
                                           "so the two must agree")
    end
    return nothing
end

# Rule 12, pair-rule agreement: the chemical topology and the streamed kernel's
# explicit PairTopology are two spellings of one set of exclusions/1-4 scales.
# Comparing only the union of mentioned pairs refuses both directions of error:
# an exclusion kept only in ChemicalTopology would be silently applied by the
# streamed kernel, and one kept only in PairTopology would be silently dropped.
@inline _unpack_pair(key::UInt64) = (Int(key >> 32), Int(key & 0xffffffff))

function _check_pair_topologies(rs::ResolvedSystem, N::Int)
    lj = nothing
    for f in rs.factors
        f isa StreamedLJ.StreamedLJFactor && (lj = f)
    end
    lj === nothing && return nothing

    topo = rs.topology
    pairs = Set{Tuple{Int,Int}}()
    for (i, j) in topo.exclusions
        push!(pairs, (min(i, j), max(i, j)))
    end
    for k in keys(topo.scales)
        push!(pairs, (min(k[1], k[2]), max(k[1], k[2])))
    end
    ftopo = lj.topology
    if ftopo !== nothing
        for key in ftopo.excluded
            push!(pairs, _unpack_pair(key))
        end
        for key in ftopo.scaled_keys
            push!(pairs, _unpack_pair(key))
        end
        ftopo.max_atom <= N ||
            _throw_error(:factor_index, "the StreamedLJ PairTopology references atom " *
                                        "$(ftopo.max_atom) but the system has $N atoms")
    end

    for (i, j) in pairs
        (1 <= i <= N && 1 <= j <= N) ||
            _throw_error(:factor_index, "topology pair ($i, $j) refers to atoms " *
                                        "outside 1:$N; the pair rules name atoms of " *
                                        "another system")
        i == j &&
            _throw_error(:factor_index, "topology pair ($i, $i) names one atom twice")
        sys_w = Types.pair_scale(topo, i, j)[1]
        fac_w = StreamedLJ.pair_weight(ftopo, i, j)
        isapprox(sys_w, fac_w; rtol = 1e-12, atol = 1e-15) ||
            _throw_error(:conflicting_pair_rules, "pair ($i, $j) has LJ weight " *
                                                  "$sys_w in ChemicalTopology but " *
                                                  "$fac_w in the StreamedLJ " *
                                                  "PairTopology; the streamed kernel " *
                                                  "would apply a different " *
                                                  "exclusion/1-4 scaling than the " *
                                                  "topology declares, so the two " *
                                                  "spellings must agree")
    end
    return nothing
end

# rule 13
function _check_constraints(rs::ResolvedSystem, N::Int)
    cs = rs.constraints
    cs === nothing && return nothing
    Constraints.natoms(cs) <= N ||
        _throw_error(:constraint_index, "the constraint set spans atom " *
                                        "$(Constraints.natoms(cs)) but the system has " *
                                        "$N atoms")
    for (c, comp) in enumerate(cs.components)
        for a in comp.atoms
            1 <= a <= N ||
                _throw_error(:constraint_index, "constraint component $c references " *
                                                "atom $a, outside 1:$N")
        end
        for (li, lj_, d) in comp.edges
            (1 <= li <= length(comp.atoms) && 1 <= lj_ <= length(comp.atoms)) ||
                _throw_error(:constraint_index, "constraint component $c has an edge " *
                                                "($li, $lj_) that does not index its " *
                                                "own atom list of length " *
                                                "$(length(comp.atoms))")
            li == lj_ &&
                _throw_error(:constraint_index, "constraint component $c has an edge " *
                                                "($li, $lj_) naming one atom twice")
            (isfinite(d) && d > 0) ||
                _throw_error(:constraint_index, "constraint component $c has target " *
                                                "distance $d; a constrained distance " *
                                                "must be finite and positive")
        end
    end
    sites = virtual_site_indices(rs)
    for comp in cs.components, a in comp.atoms
        a in sites &&
            _throw_error(:constraint_mass, "atom $a is a massless virtual site but " *
                                           "the constraint set constrains it; the " *
                                           "mass metric goes as 1/m, which is " *
                                           "undefined for a placed site")
    end
    return nothing
end

# rule 14
function _capability_refusal(rs::ResolvedSystem, cap::Symbol)
    cap === :box_derivative && return something(_box_derivative_refusal(rs),
                                               "box derivative is available")
    if cap === :virtual_sites
        isempty(rs.atoms.virtual_sites) && return "system declares :virtual_sites but contains no virtual sites"
        return "virtual sites require :virtual_sites in declared capabilities"
    end
    return "capability $(repr(cap)) is not implemented by this engine"
end

function _check_declared(rs::ResolvedSystem)
    for cap in rs.declared
        cap in _CAPABILITIES ||
            _throw_error(:unknown_capability, "declared capability $(repr(cap)) is not " *
                                              "one of $(_CAPABILITIES); an unknown " *
                                              "capability can be neither honoured nor " *
                                              "refused individually")
    end
    engine = engine_capabilities(rs)
    missing = sort!(collect(setdiff(rs.declared, engine)))
    isempty(missing) && return nothing
    details = join(["$(cap): $(_capability_refusal(rs, cap))" for cap in missing], "; ")
    _throw_error(:overdeclared_capability, "the system declares " *
                                           "$(join(String[string(cap) for cap in missing], ", ")) " *
                                           "but the resolved engine cannot provide it " *
                                           "($details). Remove the declaration or " *
                                           "drop/repair the factor that prevents it.")
end

# ---------------------------------------------------------------------------
# restart semantics & helpers
# ---------------------------------------------------------------------------

"""
    restart_semantics() -> NamedTuple

Programmatic specification of the deterministic restart contract across all 10
structural and parameter facets of a `ResolvedSystem`:
1. `topology`: :invariant (chemical bonds/angles/dihedrals/impropers preserved bit-identical)
2. `masses`: :invariant (per-atom masses preserved bit-identical)
3. `parameters`: :invariant (force-field parameters fixed across continuation)
4. `exclusions`: :invariant (nonbonded exclusions fixed)
5. `one_four_rules`: :invariant (1-4 scaling factors fixed)
6. `constraints`: :invariant (constraint distances & connectivity preserved)
7. `virtual_sites`: :invariant (site definitions & weights preserved)
8. `box`: :stateful_rebuild_if_changed (preserved in NVE/NVT; updated from checkpoint in NPT with backend factor rebuild)
9. `provenance`: :lineage_appended (preserves source/forcefield, appends restart notes)
10. `units`: :invariant (unit system conventions preserved bit-identical)
"""
function restart_semantics()
    return (
        topology = :invariant,
        masses = :invariant,
        parameters = :invariant,
        exclusions = :invariant,
        one_four_rules = :invariant,
        constraints = :invariant,
        virtual_sites = :invariant,
        box = :stateful_rebuild_if_changed,
        provenance = :lineage_appended,
        units = :invariant,
    )
end

"""
    restart_system(rs::ResolvedSystem; box = nothing, provenance_notes = nothing,
                   check_compatibility = true) -> ResolvedSystem

Produce a valid `ResolvedSystem` specification for restarting a simulation.
Topology, masses, parameters, exclusions, 1-4 rules, constraints, virtual sites,
and units are preserved bit-identical.
If `box` is supplied and differs from `rs.box` (e.g. from an NPT barostat checkpoint),
factors whose cached geometry depends on box lengths (`StreamedLJFactor`, `PMEFactor`,
`EwaldFactor`) are automatically rebuilt for the new box.
`provenance_notes` (if provided) is appended to the provenance notes to record
continuation lineage.
"""
function restart_system(rs::ResolvedSystem;
                        box::Union{Nothing,Types.AbstractBox} = nothing,
                        provenance_notes::Union{Nothing,AbstractString} = nothing,
                        check_compatibility::Bool = true)
    target_box = box === nothing ? rs.box : box
    box_changed = target_box != rs.box

    new_factors = Factors.AbstractFactor[]
    sizehint!(new_factors, length(rs.factors))

    for f in rs.factors
        if box_changed
            if f isa StreamedLJ.StreamedLJFactor
                (target_box isa Types.OrthorhombicBox || target_box isa Types.TriclinicBox) ||
                    _throw_error(:unsupported_geometry,
                        "restart box is $(typeof(target_box)), but StreamedLJFactor requires OrthorhombicBox or TriclinicBox")
                push!(new_factors, StreamedLJ.StreamedLJFactor(
                    f.N, f.epsilon, f.sigma, target_box;
                    rcut = f.rcut, skin = f.skin, mode = f.mode,
                    topology = f.topology, threaded = f.threaded
                ))
            elseif f isa PME.PMEFactor
                (target_box isa Types.OrthorhombicBox || target_box isa Types.TriclinicBox) ||
                    _throw_error(:unsupported_geometry,
                        "restart box is $(typeof(target_box)), but PMEFactor requires OrthorhombicBox or TriclinicBox")
                push!(new_factors, PME.PMEFactor(
                    zeros(Float64, 3, f.N), f.charges, target_box;
                    alpha = f.plan.alpha, rcut = f.plan.rcut, grid = f.plan.dims,
                    order = f.plan.order, prefactor = f.plan.prefactor,
                    corrections = f.corrections
                ))
            elseif f isa Electrostatics.EwaldFactor
                target_box isa Types.OrthorhombicBox ||
                    _throw_error(:unsupported_geometry,
                        "restart box is $(typeof(target_box)), but EwaldFactor requires OrthorhombicBox")
                push!(new_factors, Electrostatics.EwaldFactor(
                    zeros(Float64, 3, f.N), f.charges, target_box;
                    alpha = f.alpha, rcut = f.rcut, kmax = f.kmax,
                    prefactor = f.prefactor, corrections = f.corrections
                ))
            else
                push!(new_factors, f)
            end
        else
            push!(new_factors, f)
        end
    end

    new_notes = rs.provenance.notes
    if provenance_notes !== nothing && !isempty(strip(provenance_notes))
        new_notes = isempty(strip(new_notes)) ? String(provenance_notes) : (new_notes * "; " * String(provenance_notes))
    elseif box_changed
        box_note = "restart with updated box: $(_box_label(rs.box)) -> $(_box_label(target_box))"
        new_notes = isempty(strip(new_notes)) ? box_note : (new_notes * "; " * box_note)
    end
    prov = Provenance(rs.provenance.source, rs.provenance.force_field, new_notes)

    restarted = ResolvedSystem(
        rs.schema,
        copy(rs).atoms,
        target_box,
        rs.topology,
        new_factors,
        copy(rs.component_names),
        rs.constraints === nothing ? nothing : _copy_constraints(rs.constraints),
        rs.units,
        prov,
        copy(rs.declared)
    )

    if check_compatibility
        validate_resolved!(restarted)
    end
    return restarted
end

"""
    validate_restart_state(rs::ResolvedSystem; q = nothing, p = nothing, box = nothing,
                           min_distance::Float64 = 1e-6, constraint_tol::Float64 = 1e-4)

Validate restart state coordinates `q`, momenta `p`, and optional `box` against
`rs` specification invariants:
- verifies dimensions and finiteness
- verifies particle separations (throws `InvalidGeometry` on overlap < `min_distance`)
- verifies distance constraint satisfaction within `constraint_tol`
"""
function validate_restart_state(rs::ResolvedSystem;
                                q::Union{Nothing,AbstractMatrix{Float64}} = nothing,
                                p::Union{Nothing,AbstractMatrix{Float64}} = nothing,
                                box::Union{Nothing,Types.AbstractBox} = nothing,
                                min_distance::Float64 = 1e-6,
                                constraint_tol::Float64 = 1e-4)
    N = n_atoms(rs)
    if box !== nothing
        if box isa Types.OrthorhombicBox
            all(l -> l > 0 && isfinite(l), box.L) ||
                throw(InvalidGeometry("restart box lengths must be positive and finite, got $(box.L)"))
        elseif box isa Types.TriclinicBox
            det(box.H) > 0 ||
                throw(InvalidGeometry("restart triclinic box must have positive volume"))
        end
    end

    cur_box = box === nothing ? rs.box : box
    if q !== nothing
        size(q) == (3, N) ||
            throw(ArgumentError("restart positions must be 3 x $N, got $(size(q))"))
        all(isfinite, q) ||
            throw(InvalidGeometry("restart positions contain non-finite coordinates"))

        if min_distance > 0 && N > 1
            for i in 1:N-1, j in (i+1):N
                d = Types.unwrap_displacement(cur_box, q, i, j)
                r = sqrt(Types.norm2(d))
                if r < min_distance
                    throw(InvalidGeometry("restart positions have overlapping atoms $i and $j: distance $r < min_distance $min_distance"))
                end
            end
        end

        if rs.constraints !== nothing
            for comp in rs.constraints.components
                for (li, lj_, target_d) in comp.edges
                    ai = comp.atoms[li]
                    aj = comp.atoms[lj_]
                    d = Types.unwrap_displacement(cur_box, q, ai, aj)
                    dist = sqrt(Types.norm2(d))
                    if abs(dist - target_d) > constraint_tol
                        throw(InvalidGeometry("restart configuration violates constraint between atoms $ai and $aj: target $target_d vs actual $dist"))
                    end
                end
            end
        end
    end

    if p !== nothing
        size(p) == (3, N) ||
            throw(ArgumentError("restart momenta must be 3 x $N, got $(size(p))"))
        all(isfinite, p) ||
            throw(InvalidGeometry("restart momenta contain non-finite coordinates"))
    end

    return true
end

"""
    validate_chemistry(rs::ResolvedSystem)
    validate_chemistry(topology::Types.ChemicalTopology, factors::AbstractVector)

Validate that chemical topology and parameters conform to physical force field chemistry.
Throws `UnsupportedChemistry` if an invalid force constant, non-positive bond length,
or unsupported interaction is encountered.
"""
function validate_chemistry(rs::ResolvedSystem)
    validate_chemistry(rs.topology, rs.factors)
    return rs
end

function validate_chemistry(topology::Types.ChemicalTopology, factors::AbstractVector)
    for (k, f) in enumerate(factors)
        f isa Factors.AbstractFactor ||
            throw(UnsupportedChemistry("factor $k ($(typeof(f))) is not an AbstractFactor"))
        if f isa Factors.HarmonicBond
            (f.k >= 0 && isfinite(f.k)) ||
                throw(UnsupportedChemistry("factor $k (HarmonicBond $(f.i)-$(f.j)) has invalid force constant k=$(f.k)"))
            (f.r0 > 0 && isfinite(f.r0)) ||
                throw(UnsupportedChemistry("factor $k (HarmonicBond $(f.i)-$(f.j)) has non-positive equilibrium distance r0=$(f.r0)"))
        elseif f isa Factors.HarmonicAngle
            (f.ktheta >= 0 && isfinite(f.ktheta)) ||
                throw(UnsupportedChemistry("factor $k (HarmonicAngle $(f.i)-$(f.j)-$(f.k)) has invalid force constant ktheta=$(f.ktheta)"))
            (0 <= f.theta0 <= pi) ||
                throw(UnsupportedChemistry("factor $k (HarmonicAngle $(f.i)-$(f.j)-$(f.k)) has equilibrium angle theta0=$(f.theta0) outside [0, π]"))
        elseif f isa Factors.PeriodicDihedral
            (isfinite(f.kphi) && isfinite(f.phi0)) ||
                throw(UnsupportedChemistry("factor $k (PeriodicDihedral $(f.i)-$(f.j)-$(f.k)-$(f.l)) has non-finite parameter"))
            f.n >= 0 ||
                throw(UnsupportedChemistry("factor $k (PeriodicDihedral $(f.i)-$(f.j)-$(f.k)-$(f.l)) has negative periodicity n=$(f.n)"))
        end
    end
    return true
end

"""
    validate_geometry(rs::ResolvedSystem, q::AbstractMatrix{Float64}; min_distance::Float64 = 1e-12)

Verify that geometry `q` is non-degenerate and physically valid for `rs`.
Throws `InvalidGeometry` on non-finite coordinates or coincident atom pairs (< min_distance).
"""
function validate_geometry(rs::ResolvedSystem, q::AbstractMatrix{Float64}; min_distance::Float64 = 1e-12)
    N = n_atoms(rs)
    size(q) == (3, N) ||
        throw(ArgumentError("positions must be 3 x $N, got $(size(q))"))
    all(isfinite, q) ||
        throw(InvalidGeometry("coordinates contain non-finite values"))
    for i in 1:N-1, j in (i+1):N
        d = Types.unwrap_displacement(rs.box, q, i, j)
        if Types.norm2(d) < min_distance * min_distance
            throw(InvalidGeometry("atoms $i and $j are coincident/overlapping (distance < $min_distance)"))
        end
    end
    return true
end

end # module ResolvedSystem
