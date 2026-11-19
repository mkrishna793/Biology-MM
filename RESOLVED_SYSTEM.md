# Resolved system: Phase-1 representation and factor contract

Schema `v1.0.0` (`ResolvedSystem.RESOLVED_SCHEMA_VERSION`). Companion to
`FRAMEWORK.md` (design) and `julia/src/ResolvedSystem.jl` +
`julia/src/SystemContract.jl` (implementation). Measured evidence is recorded in
`validation/STATUS.md`; this document states the contract and the gates, not
results.

Labels used throughout: **implemented** (code exists and is exercised),
**refused** (a typed error, by design), **Phase 3** (not derived, not tested,
not claimable).

---

## 1. Purpose and Phase-1 boundary

A `ResolvedSystem` is a *coherent, versioned specification of one energy
function*: typed atoms (masses, charges, names), a box, a topology record, an
ordered list of energy factors with stable component names, an optional
constraint set, a unit system, provenance, and a declared capability set.
`SystemContract` turns that specification into energy, forces, component
energies, a box derivative, a virial and a pressure, with every unsupported
request refused by name.

The split matters: `ResolvedSystem` says *what the problem is*; the caller
supplies positions and momenta at evaluation time. There is no position array in
the struct, so "resolved" can never be mistaken for "already evaluated", and no
schema field is a dynamics result.

**In scope (Phase 1).**

1. The versioned representation and its validation rules, with named error codes.
2. The factor contract: `U(q) = Σ_α U_α(q|_{S_α})`, plus the derived quantities
   `∂U/∂s`, `∂U/∂h`, virial and pressure for the factor types that implement an
   analytic box derivative.
3. Adapters from `Types.System` and to `Types.AtomSet`.

**Explicitly out of scope for the `ResolvedSystem` core itself.** `ResolvedSystem`
still parses no file format and infers no energy term from topology; the caller
assembles factor lists. Phase 2 adds a separate importer/builder layer
(`FoundationCore.Chemistry`, see [`CHEMISTRY.md`](CHEMISTRY.md)) that produces a
`ResolvedSystem` from an external AMBER/CHARMM parameter file; that layer is not
a capability of the schema and changes no default here. No universal
triclinic force support (triclinic geometry is implemented for local factors and
MIC; `StreamedLJ`, `PME` and `Ewald` refuse triclinic cells). No production
barostat (the exported MC barostat is validated in the ideal-gas limit only, and
interacting pressure is refused without a virial provider). No performance
rewrite (the box-derivative path is direct and correctness-first; it is not the
production kernel). No PME/Ewald box derivative or stress.

**No field is a capability claim.** Concretely: a non-`nothing` `constraints`
does not imply constraint-virial-aware pressure; a populated `topology` does not
imply excluded nonbonded pairs (each backend infers nothing — see §4); a
`TriclinicBox` does not imply triclinic nonbonded support; non-empty
`virtual_sites` does not imply virtual-site dynamics; `units` does not imply a
physical force field; and `declared` is a caller promise, checked against the
engine set, never a source of capability (§6).

---

## 2. Geometry, wrapping and units

### 2.1 Periodic geometry

Cartesian positions and fractional coordinates are related by

```
r_i = h s_i ,        q = h s            (h ∈ R^{3×3}, s ∈ R^{3×N})
```

where **the columns of `h` are the lattice vectors**. `Types.OrthorhombicBox(L)`
maps to `h = Diagonal(L)`; `Types.TriclinicBox(H)` uses `H` directly (validated
finite, `det H > 0`, positive perpendicular widths); `Types.NoBox()` has no `h`.

Box volume is `V = det h` (`Types.box_volume`), i.e. `L₁L₂L₃` for the
orthorhombic case.

### 2.2 Wrapping and images

* Coordinates **may lie outside the primary cell**. Every factor that reads a
  pair uses `Types.unwrap_displacement(box, q, i, j)`, which returns
  `d = q_i - q_j - h n` for an integer image `n`: no image for `NoBox`, plain
  rounding for orthorhombic cells, a bounded search seeded at the rounded image
  for triclinic cells. The resolved layer never re-wraps positions.
* `Types.wrap!` is available for output/diagnostics. It changes no energy of a
  minimum-image factor, so wrapping is a presentation choice, not part of the
  specification.
* Minimum image is a *choice of representative*, not a physical statement. The
  periodic model is a cutoff model: pair factors must satisfy
  `rcut < min(L)/2` (enforced by the periodic backends), and the truncated sum is
  not the infinite lattice sum. `LennardJonesPair` with `mode = :none` means "no
  cutoff, one pair term"; it is meaningful for non-periodic fixtures, not as a
  periodic lattice sum.

Affine strain is `h ↦ (I + ε) h` at fixed `s`; all distances, including
intramolecular ones, scale. That is the path the box derivative follows.

### 2.3 `NoBox` behaviour (typed refusals)

| Call | `NoBox` result |
| --- | --- |
| `energy`, `forces`, `energy_forces`, `evaluate_system` | defined (local factors only) |
| `component_energies` | defined |
| `fractional_coordinates` | `UnsupportedCapability(:fractional_coordinates, …)` |
| `fractional_derivative` | refuses through the same fractional-coordinate path |
| `box_derivative`, `virial`, `affine_pressure`, `pressure` | `MissingBoxDerivative(:system, "no periodic box; …")` |

Fractional coordinates and box derivatives are **unavailable**, not zero. No
fallback substitutes Cartesian coordinates or a numerical derivative.

### 2.4 Units

`Types.Units` has exactly seven fields; every number in the state is read in
these units.

| Field | Meaning |
| --- | --- |
| `name::String` | human label |
| `length::Float64` | metres per length unit |
| `energy::Float64` | joules per energy unit |
| `mass::Float64` | kilograms per mass unit |
| `charge::Float64` | coulombs per charge unit |
| `coulomb_prefactor::Float64` | `energy·length/charge²` in these units |
| `boltzmann::Float64` | energy per temperature unit |

Consequences: energies (including `U`, `Ξ`, `W`, `W_c`) are in `units.energy`;
`g_i = ∂U/∂s_i` is energy per fractional coordinate (dimensionless coordinate →
energy); force is `units.energy / units.length`; `G = ∂U/∂h` is
`units.energy / units.length`; pressure is `units.energy / units.length³`.
`coulomb_prefactor` converts `z_i z_j / r` into an energy; a non-positive value
reverses or removes the interaction sign, so validation rejects
`coulomb_prefactor ≤ 0` when any `CoulombPair` is present. PME and Ewald take
their own `prefactor` argument (finite-checked at construction); the resolved
layer does not cross-check it against `units.coulomb_prefactor` (§8, A7).

`REDUCED` (σ = ε = m = 1, k_B = 1) is the only predefined system.
`Types` currently *names* `SIReduced` in its export list and `Units` docstring but
does **not** define it in this tree; an SI unit record must be constructed
explicitly. The record has no time field; a time unit, where one is needed,
follows from `√(mass·length²/energy)`.

---

## 3. Derivatives and sign conventions

With `U(s, h)` the total energy at fixed fractional coordinates `s` and box `h`:

| Symbol | Definition | Size / units |
| --- | --- | --- |
| `U` | total energy | scalar, `energy` |
| `g` | `g_i = ∂U/∂s_i` | `3×N`, `energy` |
| `F` | `F_i = -h^{-T} g_i` | `3×N`, `energy/length` |
| `G` | `G[a,b] = ∂U/∂h[a,b]` at fixed `s` | `3×3`, `energy/length` |
| `Ξ` | `Ξ = -h Gᵀ = Σ_i r_i F_iᵀ` | `3×3`, `energy` |
| `W` | `W = tr Ξ = Σ_i r_i · F_i` | scalar, `energy` |
| `P_aff` | `-tr(G hᵀ)/(3V) = tr(Ξ)/(3V)` | scalar, `energy/length³` |
| `P_mech` | `(2K + tr Ξ + W_c)/(3V)`, `K = ½ Σ p_i²/m_i` | scalar, `energy/length³` |

Sign conventions, stated once and implemented everywhere: `F = -∂U/∂q` (the
existing factor convention), `energy_forces` returns `(U, F)`, and Cartesian
forces follow from the fractional gradient by `F = -h^{-T} g`. The inverse
transpose is the whole content of the chain rule: `g_i = -hᵀ F_i`.

**Per-atom box derivative.** Because `r_i = h s_i` with `s` held fixed,

```
∂U/∂h[a,b] = Σ_i (∂U/∂r_i[a]) s_i[b] = -Σ_i F_i[a] s_i[b] ,
G = -Σ_i F_i s_iᵀ .
```

This is why `G` needs no finite differences and no second factor sweep: the
forces already contain the information.

**Image correction is exact, not approximate.** For a pair with
`d = h w`, `w = s_i - s_j - n`, the integer image `n` is piecewise constant in
`h`, so at fixed `s` the pair contributes
`G += (dU/dr)(1/r) d wᵀ` — the same `w` that appears in the displacement. No
"image term" is dropped, and `w` is computed from the same
`Types.unwrap_displacement` call the force path uses. The identity
`tr(Ξ) = -tr(h Gᵀ) = -tr(G hᵀ)` fixes the sign in `P_aff`; `P_aff = tr(Ξ)/(3V)`
equals `-tr(G hᵀ)/(3V)` and reduces to `W/(3V)`.

**Domain of validity.** `G` is the derivative of the *implemented* cutoff
energy. It is undefined on minimum-image tie sets (where `n` jumps); the energy is
continuous there, the derivative is not. FD fixtures stay away from ties.

**Difference between pressures.** `P_aff` is the unconstrained affine value: it
scales every vector, intramolecular bonds included. A rigid-molecule pressure
that scales only centres of mass leaves the internal geometry fixed and therefore
differs from `P_aff` by exactly the constrained intramolecular work. Constraint
forces do no work on the constraint manifold but do contribute to the virial
because the box couples to all coordinates: for
`c_k = |x_j - x_i|² - d_k²`, `W_c = Σ_i r_i·F^c_i = -2 Σ_k λ_k |r_k|²`
(`Constraints.constraint_virial`). `Constraints` documents and tests the identity
`-dU/ds = W_ext + W_c` along the projected scaling path `q(s) = Π(s q)`.

**The constraint virial is separate and required, never optional.**
`pressure(rs, q, p; extra_virial = W_c)`; `extra_virial` is where a caller adds
it. `pressure` with the default `extra_virial = 0.0` is the unconstrained
mechanical value and must not be reported as the pressure of a constrained
system. Multipliers must come from `Constraints.constraint_multipliers` (the
static, box-derivative-validated route) in force units — not from a projection
impulse.

**What is not claimed.** `Σ_i r_i F_iᵀ` is the configurational virial of the
implemented local factors. It is **not** the box derivative of a periodic
Ewald/PME energy: the reciprocal-space stress, the `1/V` volume terms, the
self/background volume dependence and the tin-foil boundary term are not derived
here. PME/Ewald stress is **Phase 3**; `box_derivative`, `virial`,
`affine_pressure` and `pressure` refuse a system containing those factors
(§6). No result in this repository should be read as a periodic electrostatic
pressure.

---

## 4. Boundary conditions and electrostatics

PME (`PME.pme_evaluate` / `PMEFactor`) and direct Ewald
(`Electrostatics.evaluate_ewald` / `EwaldFactor`) share one convention:

```
U = prefactor · (U_real + U_recip + U_self + U_background + U_pair)
```

* **Uniform neutralizing background, always present.** `U_background` is
  included unconditionally, so charges need not sum to zero and a net charge is
  not a validation error. It must be *declared* by the caller: the resolved
  system stores the charges it is given and never neutralises, rescales or
  recentres them.
* **Tin-foil (conducting) boundary.** No surface/dipole term is added. This is
  not a slab, vacuum or finite-dipole correction, and the resolved layer adds
  none.
* **Self term included** (`-α/√π Σ q_i²`); forces are the exact negative gradient
  of the discrete energy, not an `i k` spectral derivative.
* **Explicit pair corrections only.** `PairCorrections(; excluded=[(i,j), …],
  scaled=[(i,j,s), …])` adds `(s-1)·prefactor·q_i q_j / r` at the minimum-image
  distance for each listed pair; `excluded` is `s = 0`. This is the standard
  point-Coulomb correction, identical in both modules. **No bonded topology is
  inferred** — supplying `ChemicalTopology` does not switch corrections on.
* **Orthorhombic only.** Both backends require an orthorhombic box;
  `0 < rcut < min(L)/2` (PME's linked-cell traversal assumes it). PME spline
  order is 4 or 6; the default grid targets ~3 grid points per `1/α` and rounds up
  to a power of two.

Cutoff conventions are **different potentials**, not cosmetics:

| Mode | Continuity | Force is `-dU/dr`? | Status |
| --- | --- | --- | --- |
| `:none` | n/a (no cutoff) | yes | implemented; one pair term, not a lattice sum |
| `:lfs` | `C¹` | yes, at and beyond `rc` | implemented, force-consistent, default for `StreamedLJ` |
| `:shift` | `C⁰` | no at `rc` | implemented as energy-only; refused by the integrator gate and by `box_derivative`/`virial`/`pressure` |
| `:trunc` | discontinuous at `rc` | no | retained only for NIST reference data; `force_consistent` false; refused by the integrator |

The resolved layer **changes no default**. It copies factor records verbatim,
reads their parameters during validation and never rewrites them; the defaults
are exactly the documented constructor defaults (`units = REDUCED`,
`box = NoBox()`, `names = fill(:X, N)`, `component_names = component_1, …`,
`StreamedLJ` mode `:lfs`, PME order 4, `LennardJonesPair` mode `:none` with
`rc = Inf`). A missing box is refused by the capability check, never filled in
silently. Cutoff validation is **not** uniform: the periodic backends enforce
`0 < rcut < min(L)/2` at construction, while `LennardJonesPair` does not check
`rc` at all — a cutoff mode with `rc = Inf` is a caller error, not a documented
degenerate case (§8, A7).

---

## 5. Resolved-system representation

### 5.1 Schema `v1.0.0`

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | `VersionNumber` | must share `RESOLVED_SCHEMA_VERSION.major` |
| `atoms` | `ResolvedAtoms` | `masses`, `charges`, `names`, `virtual_sites` |
| `box` | `Types.AbstractBox` | `NoBox`, `OrthorhombicBox`, `TriclinicBox` |
| `topology` | `Types.ChemicalTopology` | bonds/angles/dihedrals/impropers, `exclusions`, `scales` — a record, never a source of energy |
| `factors` | `Vector{Factors.AbstractFactor}` | the energy terms, in evaluation order |
| `component_names` | `Vector{Symbol}` | one stable name per factor, parallel to `factors` |
| `constraints` | `Union{Nothing, Constraints.CoupledConstraintSet}` | constrained submanifold, or none |
| `units` | `Types.Units` | §2.4 |
| `provenance` | `Provenance` | `source`, `force_field`, `notes` |
| `declared` | `Set{Symbol}` | caller's minimum capability promise, checked at construction |

Construction is keyword-only: `ResolvedSystem(; masses, charges, names = nothing,
box = Types.NoBox(), topology = Types.ChemicalTopology(),
factors = Factors.AbstractFactor[], component_names = nothing, constraints =
nothing, virtual_sites = VirtualSite[], units = Types.REDUCED, provenance =
Provenance(), declared = [:energy, :cartesian_forces, :component_energies],
schema = RESOLVED_SCHEMA_VERSION)`.

`VirtualSite(kind, site, parents, coefficients)` accepts `:linear` (2 parents),
`:planar` (3) and `:average` (≥ 2); coefficients must sum to 1 within `1e-12`.

### 5.2 Validation

`validate_resolved!` runs rules 2–14 on an existing struct (catches
post-construction mutation); the constructor runs rules 1–14 and ends with the
same call. Every refusal is `ResolvedSystemError(code, msg)`.

| # | Rule | Code |
| --- | --- | --- |
| 1 | `schema` is a `VersionNumber` with matching major | `:schema` |
| 2 | `N = length(masses) ≥ 1`; `charges`, `names` have length `N` | `:shape` |
| 3 | masses finite; non-virtual atoms strictly `> 0`; declared virtual-site atoms exactly `0` | `:mass` |
| 4 | charges finite | `:charge` |
| 5 | `length`, `energy`, `mass`, `charge`, `boltzmann` finite and `> 0`; `coulomb_prefactor` finite, and `> 0` if any `CoulombPair` exists | `:units` |
| 6 | `provenance.source`, `provenance.force_field` non-empty | `:provenance` |
| 7 | virtual-site kind/parent count/indices/distinctness/coefficient length, finiteness, `Σcoef ≈ 1` (`atol = 1e-12`), unique site, zero mass at the site | `:virtual_site` |
| 8 | `component_names` length equals `factors` length, all `Symbol`, no duplicates | `:component_name` |
| 9 | every factor's support non-empty, indices in `1:N`, no repeated index | `:factor_index` |
| 10 | no duplicate interaction key `(type, sorted support)`, pairs compared as sorted pairs | `:duplicate_interaction` |
| 11 | at most one factor per backend family: `:long_range_electrostatics` (PME and Ewald are **one** family), `:streamed_lj`, `:bipartite_harmonic` | `:duplicate_backend` |
| 12a | `StreamedLJFactor`, `PMEFactor`, `EwaldFactor` require an orthorhombic box | `:unsupported_geometry` |
| 12b | periodic factor box lengths equal `box.L` | `:box_mismatch` |
| 12c | factor charges equal the system charges (`rtol 1e-12`, `atol 1e-15`) | `:charge_mismatch` |
| 12d | every pair mentioned in `ChemicalTopology` exclusions/scales **or** in the `StreamedLJFactor` packed topology has the same LJ weight (`Types.pair_scale` vs `StreamedLJ.pair_weight`) | `:conflicting_pair_rules` |
| 13a | `Constraints.natoms(cs) ≤ N` | `:constraint_index` |
| 13b | no constrained atom is a massless virtual site | `:constraint_mass` |
| 14a | `declared ⊆ (:energy, :cartesian_forces, :component_energies, :box_derivative)` | `:unknown_capability` |
| 14b | every declared capability is in `engine_capabilities(rs)`, with the concrete reason per missing capability | `:overdeclared_capability` |

Rule 12d is the anti-silent-loss rule: an exclusion applied by the `LJ` backend
but absent from `ChemicalTopology` (or the reverse) is refused rather than
reconciled. Rule 11 means "one long-range solver", so `PMEFactor` **and**
`EwaldFactor` in one system is `:duplicate_backend`.

### 5.3 Ownership and aliasing

The struct is immutable; that is **shallow**. Arrays inside it are mutable and
may be shared.

| Object | Constructor behaviour | `copy(rs)` behaviour |
| --- | --- | --- |
| `masses`, `charges`, `names`, `component_names` | copied | containers copied, **arrays shared** |
| `virtual_sites` (and each `parents`/`coefficients`) | copied | shared |
| `factors` container | copied; **factor objects shared** | shared |
| `constraints` | rebuilt as an owned copy (`copy(atoms)`, `copy(edges)` per component) | shared |
| `topology` | stored by reference (`exclusions`, `scales`, bond vectors are shared) | shared |
| `box` | stored by value; a `TriclinicBox` shares its `H`/`Hinv` matrices | shared |
| `units`, `provenance` | immutable value structs | shared |

Tested property: mutating the `masses` vector passed to the constructor does not
change `rs.atoms.masses`, and `copy(rs).atoms.masses === rs.atoms.masses`.
Consequence: a `ResolvedSystem` must be treated as read-only. Mutating a shared
factor (e.g. `StreamedLJFactor.rcut`, `PMEFactor.charges`) is visible through
every system that references it; `validate_resolved!` catches structural
corruption (bad indices, duplicate backends, box or charge mismatch) but cannot
police every parameter edit, and it cannot make a shared factor thread-safe
(`PMEPlan` is explicitly not thread-safe).

### 5.4 Specification versus restart state

| Quantity | Home | Why |
| --- | --- | --- |
| masses, charges, factor parameters, box, topology, constraints, units, provenance, `declared` | `ResolvedSystem` (specification) | defines the energy function and its algebra |
| positions `q`, momenta `p` | caller | configuration, re-evaluated per call |
| integrator (timestep, step index), thermostat/barostat state, RNG seed and stream | caller | integration state, not part of the energy |
| `Types.System.temperature` | caller | sampler metadata; **not** carried into `ResolvedSystem` |
| constraint multipliers | caller | derived per configuration (`Constraints.constraint_multipliers`) |

A full checkpoint format (specification + restart state + RNG + versioned
serializer) is a later phase. `ResolvedSystem` is a fixed-box specification: a
barostat volume move changes `h`, and because the periodic factors validate
their own box against the context, the factors must be rebuilt for the new box
rather than the box edited in place.

### 5.5 Virtual sites

Massless sites are **representable** — rule 3 requires mass exactly `0` at a
declared site, and rule 7 checks the geometry of the definition — so a force
field's virtual-site *description* can be carried and validated. Execution is
**refused**: `SystemContract.energy`, `energy_forces`, `forces` and `to_atomset`
raise `UnsupportedCapability(:virtual_sites, "…representation only; execution is
not implemented")`. No force redistribution, mass-metric handling, charge
redistribution or integrator coupling is implemented, and none is invented here.
A massless atom without a declared `VirtualSite` is refused (`:mass`).

---

## 6. Capability-aware API

### 6.1 Functions

| Function | Signature | Returns / notes |
| --- | --- | --- |
| `energy` | `energy(rs, q)` | `U::Float64` |
| `forces` | `forces(rs, q)` | `3×N`, `F = -∂U/∂q` |
| `energy_forces` | `energy_forces(rs, q; F = nothing)` | `(U, F)`; optional accumulator |
| `evaluate_system` | `evaluate_system(rs, q; F = nothing)` | alias of `energy_forces`, identical values; no second code path |
| `component_energies` | `component_energies(rs, q)` | `NamedTuple` keyed by `component_names`; each factor evaluated into a private zeroed accumulator, so summing in factor order reproduces `energy` exactly and components are independent |
| `component_energy` | `component_energy(rs, q, name)` | one component |
| `box_derivative` | `box_derivative(rs, q)` | `G::Matrix` (3×3), `∂U/∂h` at fixed `s`; requires a periodic box and every component supported |
| `box_derivative_component` | `box_derivative_component(rs, q, name)` | 3×3 for one component |
| `virial` | `virial(rs, q)` | `Ξ = -h Gᵀ`; calls `box_derivative` |
| `affine_pressure` | `affine_pressure(rs, q)` | `-tr(G hᵀ)/(3V)` |
| `pressure` | `pressure(rs, q, p; extra_virial = 0.0)` | `(2K + tr Ξ + extra_virial)/(3V)`; pass `W_c` |
| `kinetic_energy` | `kinetic_energy(rs, p)` | `½ Σ p²/m` |
| `fractional_coordinates` | `fractional_coordinates(rs, q)` | `q ./ L` or `H⁻¹q`; refuses `NoBox` |
| `cartesian_coordinates` | `cartesian_coordinates(rs, s)` | `h s` |
| `fractional_derivative` | `fractional_derivative(rs, q; h = 1e-6)` | `3×N` `∂U/∂s` by central differences; a diagnostic, `O(N)` energy calls, never used by `forces` or `box_derivative` |
| `resolved_context` | `resolved_context(rs)` | `Factors.Context(rs.box, rs.units, n_atoms(rs))` |
| `resolved_from_system` | `resolved_from_system(s::Types.System, factors; component_names = nothing, constraints = nothing, virtual_sites = VirtualSite[], provenance = Provenance(), declared = [:energy, :cartesian_forces, :component_energies], schema = RESOLVED_SCHEMA_VERSION)` | masses/charges/names/box/topology/units from `s`; factors must be supplied — the adapter cannot infer energy terms from topology |
| `to_atomset` | `to_atomset(rs, q, p)` | `Types.AtomSet`; refuses virtual sites |
| `supports_box_derivative` | `supports_box_derivative(f, box)` | predicate combining the factor support and the geometry guard |

`box_derivative` and everything built on it (virial, pressures) are all-or-nothing:
one unsupported component refuses the total. No component is silently zeroed and
none is replaced by finite differences.

`component_energies` zeroes a `3×N` accumulator per factor, so it is a diagnostic
path (cost `O(#factors·N)` array work), not an allocation-free force path.

### 6.2 Per-factor capability

| Factor | energy / forces / components | `box_derivative` | Reason when refused |
| --- | --- | --- | --- |
| `HarmonicBond` | supported | supported (orthorhombic, triclinic) | — |
| `HarmonicAngle` | supported | supported (orthorhombic, triclinic) | — |
| `PeriodicDihedral` | supported | supported (orthorhombic, triclinic) | — |
| `BipartiteHarmonicGroup` | supported | supported (orthorhombic, triclinic) | — |
| `LennardJonesPair` | supported | supported for `:none`, `:lfs` | `:shift`/`:trunc`: mode is not force-consistent at the cutoff |
| `CoulombPair` | supported (`NoBox` only) | refused | non-periodic only; no periodic box derivative exists |
| `StreamedLJFactor` | supported (orthorhombic) | supported for `:lfs` | `:shift`: not `C¹`/force-consistent |
| `PMEFactor` | supported (orthorhombic) | refused | Phase 3: stress not derived or tested |
| `EwaldFactor` | supported (orthorhombic) | refused | Phase 3: stress not derived or tested |

`engine_capabilities(rs)` always contains `:energy`, `:cartesian_forces`,
`:component_energies`; it adds `:box_derivative` iff the box is orthorhombic or
triclinic, every factor's `box_derivative_support` is `nothing`, and no factor
requires orthorhombic while the box is triclinic.

Two timing details matter. `CoulombPair` in a periodic box is refused by the
factor at **evaluation** time (`Types.UnsupportedGeometry`), not at
construction: a factor record can be inspected before a box is attached, so the
validation rules do not reject it. And a per-factor `box_derivative` support does
not substitute for a periodic box — with `NoBox`, `box_derivative_component`
still throws `MissingBoxDerivative` for `CoulombPair`, and the total refuses at
the system level.

### 6.3 Declared versus engine capability

* `declared` is the caller's **minimum** promise about what downstream code may
  use. It may be a strict subset of the engine set (under-declaration is legal).
* `engine_capabilities` is **computed** from the factor list and box, and is
  never upgraded by declaration.
* Over-declaration is a **construction error** (`:overdeclared_capability`) with
  the concrete per-capability reason, so a capability cannot be claimed into
  existence.
* Capability sets describe the factor layer. They do not override the
  virtual-site execution refusal (§5.5), which takes precedence at every
  evaluation entry point, and they do not imply that any specified physics has
  been validated — see §8.

---

## 7. Examples

All snippets run from the repository root against the package project:

```sh
~/.cache/biology-mm/julia-1.13.3/bin/julia --project=julia
```

(Set `JULIA_DEPOT_PATH=~/.cache/biology-mm/julia-depot` if the
package environment is not already resolved.) The snippets form a single session
and are meant to be run in order: 7.1 defines the `RS`/`SC`/`T`/`F` aliases and
`rs`.

### 7.1 Build a resolved system

```julia
using FoundationCore
const RS = FoundationCore.ResolvedSystem
const SC = FoundationCore.SystemContract
const T  = FoundationCore.Types
const F  = FoundationCore.Factors

box = T.OrthorhombicBox((4.0, 5.0, 6.0))
rs = RS.ResolvedSystem(;
    masses = [12.0, 1.0, 1.0],
    charges = [0.0, 0.1, -0.1],
    names = [:C, :H1, :H2],
    box = box,
    factors = [F.HarmonicBond(1, 2, 1000.0, 1.1),
               F.HarmonicAngle(2, 1, 3, 100.0, 1.9)],
    component_names = [:stretch, :bend],
    units = T.REDUCED,
    provenance = RS.Provenance(; source = "doc example",
                                 force_field = "bond + angle toy model"))

RS.n_atoms(rs)                     # 3
RS.schema_version(rs)              # v"1.0.0"
RS.engine_capabilities(rs)         # includes :box_derivative (orthorhombic, local factors)
RS.validate_resolved!(rs)          # re-checks rules 2–14; returns normally
```

### 7.2 Energy, forces and components

```julia
q = [0.0 1.1 1.6;
     0.0 0.2 -0.3;
     0.0 0.1 0.2]                  # 3×3 Cartesian coordinates

U  = SC.energy(rs, q)
Fc = SC.forces(rs, q)              # F = -∂U/∂q
ce = SC.component_energies(rs, q)  # (stretch = …, bend = …)

@assert U ≈ ce.stretch + ce.bend
@assert SC.energy_forces(rs, q)[1] ≈ U
@assert SC.component_energy(rs, q, :stretch) ≈ ce.stretch
@assert SC.evaluate_system(rs, q)[1] ≈ U    # alias, same values
```

### 7.3 Box derivative, virial and pressure (exact analytic check)

```julia
h = [4.0 0.0 0.0;
     0.0 5.0 0.0;
     0.0 0.0 6.0]                  # columns are the lattice vectors
s = [0.0 0.3;
     0.0 0.0;
     0.0 0.0]                      # fractional coordinates
q2 = h * s                          # r = (0,0,0) and (1.2,0,0)

rs2 = RS.ResolvedSystem(;
    masses = [1.0, 1.0], charges = [0.0, 0.0],
    box = T.OrthorhombicBox((4.0, 5.0, 6.0)),
    factors = [F.HarmonicBond(1, 2, 100.0, 1.0)],
    component_names = [:bond],
    provenance = RS.Provenance(; source = "doc example"),
    declared = [:energy, :cartesian_forces, :component_energies, :box_derivative])

G   = SC.box_derivative(rs2, q2)     # ∂U/∂h at fixed s
Xi  = SC.virial(rs2, q2)             # Ξ = -h Gᵀ
Paf = SC.affine_pressure(rs2, q2)
p   = zeros(3, 2)
Pmech = SC.pressure(rs2, q2, p)      # p = 0 so 2K = 0; no constraints here

# r = 1.2, dr = 0.2, k dr / r = 16.666…, w = d./L = (0.3, 0, 0)
@assert G[1, 1] ≈ 6.0                              # = -Σ F_i[1] s_i[1]
@assert Xi[1, 1] ≈ -24.0                           # = Σ r_i[1] F_i[1], F_2x = -20
@assert Paf ≈ -24.0 / (3 * 4.0 * 5.0 * 6.0)        # = -1/15
@assert Pmech ≈ Paf
@assert SC.fractional_derivative(rs2, q2)[1, 2] ≈ 80.0  # g_2x = -h₁₁ F_2x = 4·20
```

The last line is the chain rule in numbers: with `h` diagonal and
`F_2x = -20`, `g_2x = -h₁₁ F_2x = 4·20 = 80`, which is also
`∂U/∂s_2x = k(r - r₀)·h₁₁`. `fractional_derivative` is a central-difference
estimate; this energy is quadratic in `s`, so the difference is exact up to
roundoff. It is a consistency check on the convention, not an adjoint proof.

### 7.4 Typed refusal for PME (no stress in Phase 1)

```julia
L  = (20.0, 20.0, 20.0)
bx = T.OrthorhombicBox(L)
q3 = [0.0 3.0 0.0;
      0.0 0.0 3.0;
      0.0 0.0 0.0]
ch = [1.0, 0.5, -0.5]               # net +1: the uniform background is always on

pme = FoundationCore.PME.PMEFactor(q3, ch, bx;
                                   alpha = 0.35, rcut = 8.0,
                                   grid = (16, 16, 16), order = 4)

# Declaring stress is a construction error, not a warning.
try
    RS.ResolvedSystem(; masses = ones(3), charges = ch, box = bx,
        factors = [pme], component_names = [:pme],
        declared = [:energy, :cartesian_forces, :component_energies, :box_derivative])
catch e
    @assert e isa RS.ResolvedSystemError && e.code === :overdeclared_capability
    println(e)          # ResolvedSystemError[overdeclared_capability]: …
end

rs3 = RS.ResolvedSystem(; masses = ones(3), charges = ch, box = bx,
    factors = [pme], component_names = [:pme],
    declared = [:energy, :cartesian_forces, :component_energies])

@assert !(:box_derivative in RS.engine_capabilities(rs3))
U3 = SC.energy(rs3, q3)             # supported: PME energy/forces

try
    SC.box_derivative(rs3, q3)
catch e
    @assert e isa SC.MissingBoxDerivative && e.component === :pme
    println(e)          # MissingBoxDerivative[pme]: PME box derivative (stress) is
                        # a Phase 3 capability; not derived or tested here
end
```

`virial`, `affine_pressure` and `pressure` refuse through the same
`MissingBoxDerivative`; a local bond in the same system still has its own
`box_derivative_component`.

### 7.5 Virtual-site refusal

```julia
vs = RS.VirtualSite(:linear, 3, [1, 2], [0.5, 0.5])
rs4 = RS.ResolvedSystem(; masses = [1.0, 1.0, 0.0], charges = zeros(3),
    factors = [F.HarmonicBond(1, 2, 100.0, 1.0)], virtual_sites = [vs],
    provenance = RS.Provenance(; source = "doc example"))

@assert RS.virtual_site_indices(rs4) == Set([3])   # representation accepted
try
    SC.energy(rs4, [0.0 1.0 0.5; 0.0 0.0 0.0; 0.0 0.0 0.0])
catch e
    @assert e isa RS.UnsupportedCapability && e.capability === :virtual_sites
end
```

---

## 8. Assumptions and future interface boundaries

Assumptions carried by the contract:

* **A1.** Positions are finite; a coincident-atom configuration is state but not
  a valid sample. Degenerate geometry returns `Inf` from the local factors (as
  they already do), and preparation/repair is out of scope.
* **A2.** Cutoff models are physical only with `rcut < min(L)/2`; the periodic
  backends enforce that at construction. The resolved layer adds no global check
  for local pair factors, so a too-large cutoff on a `LennardJonesPair` is a
  caller error.
* **A3.** `fractional_derivative` on `NoBox` refuses via the
  `:fractional_coordinates` capability; it never substitutes Cartesian
  differences, and `NoBox` force/energy evaluation never touches fractional
  coordinates.
* **A4.** The box derivative is exact for the implemented local energies away
  from minimum-image ties, and is not the derivative of any long-range solver.
* **A5.** A `ResolvedSystem` is read-only (shallow immutability, §5.3); shared
  factor objects are not thread-safe (`PMEPlan` is documented as such).
* **A6.** `declared` bounds what a consumer may rely on, but does not certify
  that the behaviour has been validated for the intended application.
* **A7.** Validation checks structure and cross-consistency, not every parameter
  convention: `LennardJonesPair` does not reject a non-finite `rc` for cutoff
  modes, and a PME/Ewald `prefactor` is not compared with
  `units.coulomb_prefactor`. Both are caller responsibilities in Phase 1; they
  are recorded here rather than silently repaired, and no validation rule in §5.2
  claims otherwise.

Future boundaries, each a separate deliverable with its own evidence:

* **Phase 3 — PME/Ewald stress.** Reciprocal-space box derivative including the
  `1/V` terms, self/background volume dependence and the tin-foil convention;
  FD-checked against `pme_energy`. Until then, every electrostatic pressure is
  refused.
* **Phase 3 — triclinic nonbonded backends.** A lattice-correct stencil for
  `StreamedLJ`/PME/Ewald in a `TriclinicBox`; today those constructors refuse
  triclinic cells while local factors and MIC already work there.
* **Phase 3 — virtual sites.** Force/charge redistribution, mass-metric handling
  and integrator coupling; the schema already carries the definition.
* **Phase 3 — full checkpoint.** Versioned serializer for specification +
  restart state (positions, momenta, integrator/RNG/barostat state) with
  round-trip tests, distinct from this specification record.
* **Phase 3 — barostat pressure coupling.** Vector/tensor pressure with the
  constraint virial included by construction and factors rebuilt per box; the
  current MC barostat is isotropic, ideal-gas-validated, and refuses interacting
  pressure without a virial provider.

---

## 9. Phase-1 completion gates and measured evidence

A gate is closed only by a passing run of its command in this repository, not by
the presence of code.

| Gate | Requirement |
| --- | --- |
| G1 Representation | schema `v1.0.0` constructs and validates; all 14 rules produce exactly the documented codes; ownership/aliasing properties hold |
| G2 Local derivatives | `forces` matches central FD; `fractional_derivative` matches `-h^{-T} g`; analytic `G` matches the closed form and fixed-fractional FD; `virial = -h Gᵀ = Σ r Fᵀ`; `affine_pressure` matches isotropic-scaling FD |
| G3 Periodic nonbonded | resolved `StreamedLJ` parity with the direct factor; exclusions/scales exact; `:lfs` continuity at `rc`; `:shift` refuses `box_derivative`/`virial`/`pressure` |
| G4 Long range | PME resolved energy/forces parity with `pme_energy_forces`; stress refusal names the PME component; local components in the same system still differentiate |
| G5 Adapters | `resolved_from_system` route equals the direct factor route bit-for-bit, and one velocity-Verlet step agrees; `to_atomset` round-trips masses/charges |
| G6 Regression | full suite runner and `verify.sh` pass |

Evidence commands (exact):

```sh
export JULIA_DEPOT_PATH=~/.cache/biology-mm/julia-depot
JULIA=~/.cache/biology-mm/julia-1.13.3/bin/julia
cd julia
$JULIA --project=. test/test_resolved_system.jl     # G1
$JULIA --project=. test/test_system_contract.jl     # G2–G5
$JULIA --project=. test/run_all.jl                  # G6 (all suites)
cd ../.. && sh verify.sh                 # G6 (proofs + materials + NIST)
```

### Measured evidence

The test suites print their measured numbers (convergence ratios, maximum
errors); those values are transcribed here after the final verification run. The
`Measured` column is filled by the orchestrator from that run only.

| Evidence | Gate | Measured | Status |
| --- | --- | --- | --- |
| Constructor validation suite (`test_resolved_system.jl`) | G1 | 50 assertions / ~120 executed checks: every documented code fires; ownership alias properties hold | PASS |
| Force vs central FD at `h ∈ (1e-4, 1e-6, 1e-8)` | G2 | max abs error `1.61e-7 / 3.61e-9 / 4.53e-7` (the last is the `ε|U|/h` roundoff floor, asserted as such) | PASS |
| Fractional chain rule `F ≈ -h^{-T} g` | G2 | max abs error `2.15e-8 / 2.27e-10 / 4.69e-10` at `h = 1e-5/1e-6/1e-7`; MIC energy `0.07695` vs unwrapped `430.55` across the boundary | PASS |
| Box-derivative FD convergence, `ε ∈ (1e-3 … 1e-6)` | G2 | bond `G` errors `8.97e-8 → 8.84e-10 → 2.10e-10`; angle `1.85e-8 → 2.14e-10`; dihedral `1.68e-10 → 1.28e-12`; LJ `9.98e-9 → 9.95e-11`; streamed-LJ `3.30e-7 → 3.30e-9 → 2.33e-11`; triclinic bond `1.63e-7 → 1.60e-9 → 3.54e-10` | PASS |
| Exact analytic `G` at `1e-12` | G2 | max abs error `4.44e-16`; dihedral cumulative-offset form `5.74e-12` vs non-cumulative form `7.75e-3` (the typo is detectable, not cosmetic) | PASS |
| Streamed-LJ resolved parity (`1e-12`) and FD (`< 1e-7`) | G3 | parity `dU = dF = 0`; single-pair `U = -0.459325160148`, half `= -0.229662580074`, exclusion `= 0`; `:lfs` cutoff `U(rc+1e-6)=0`, `U(rc-1e-6)=-5.18e-13`; force `|dF|=3.64e-9`; `:shift` refuses | PASS |
| PME refusal: `MissingBoxDerivative` naming `:pme`; `:overdeclared_capability` | G4 | resolved PME parity `dU=dF=0` at `U=-0.211116442`; every total-stress entry point refuses with `:pme`; a local bond beside PME still gives FD error `1.60e-9` | PASS |
| Adapter parity: equal force arrays and one equal Verlet step | G5 | `max|Δq| = max|Δp| = max|ΔF| = 0.0` (bit-identical); `to_atomset` round-trips masses/charges/names | PASS |
| Virial and affine pressure | G2 | `max|Ξ - (-hGᵀ)| = 0`, `max|Ξ - Σ rFᵀ| = 4.89e-15`; `P_aff` vs isotropic-scaling FD `6.65e-9 → 6.64e-11 → 7.49e-13`; component sum and superposition error `0.0` | PASS |
| Full `verify.sh` | G6 | independent verifier run; see `validation/STATUS.md` §9 | see STATUS |

The behavioural claims above are exactly the measured checks. They do **not**
establish bulk simulation accuracy, production NPT, or a scientifically
validated force field; those remain later-phase work.
