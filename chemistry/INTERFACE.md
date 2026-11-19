# Phase-2 normalized chemistry interface (internal contract)

This file is the coordination contract for the Phase-2 implementation. It is
normative for every module under `chemistry/` and
`julia/src/Chemistry*.jl`. Units are fixed for schema `v1.0.0`:
**length nm, energy kJ/mol, mass amu, charge elementary charge e, angle rad**.
Temperature is kelvin (only metadata). There is no per-atom epsilon/sigma
mixing choice: Lorentz–Berthelot is the only model in schema v1.0.0.

The engine force convention is unchanged: `F = -dU/dq`.

---

## 1. Normalized JSON (`mdparams` v1.0.0)

Top-level object, keys exactly as listed (unknown keys are an error):

```json
{
  "schema": "1.0.0",
  "family": "amber",
  "units": "nm-kj-mol-e-amu",
  "provenance": {
    "source": "wrappers/python/tests/systems/Mg_water.prmtop",
    "force_field": "AMBER prmtop; title from file",
    "source_sha256": "<64 hex>",
    "coordinate_source": "wrappers/python/tests/systems/Mg_water.inpcrd",
    "coordinate_sha256": "<64 hex>",
    "generator": "chemistry/tools/extract_amber.py",
    "generator_version": "1.0.0",
    "notes": "free text (may be empty)"
  },
  "atoms": [
    {"name":"O","mass":15.9994,"charge":-0.834,"element":"O",
     "residue_index":1,"residue_name":"WAT",
     "sigma_nm":0.315061,"epsilon_kj":0.636386}
  ],
  "box": {"kind":"orthorhombic","lengths_nm":[3.2,3.1,3.0]},
  "bonds": [{"i":1,"j":2,"k":418400.0,"r0_nm":0.09572}],
  "angles": [{"i":1,"j":2,"k":3,"k_theta":418.4,"theta0_rad":1.8242}],
  "urey_bradleys": [{"i":1,"j":3,"k":41840.0,"r0_nm":0.25}],
  "dihedrals": [
    {"i":1,"j":2,"k":3,"l":4,
     "terms":[{"k":0.5,"n":2,"phase_rad":3.141592653589793}]}
  ],
  "impropers": [{"i":1,"j":2,"k":3,"l":4,"k_phi":418.4,"phase_rad":0.0}],
  "cmaps": [{"i":[1,2,3,4,5,6,7,8],"resolution":24,"grid":[0.0]}],
  "exclusions": [[1,3],[2,3]],
  "pair14": [{"i":1,"j":4,"coulomb_scale":0.8333333333333334,"lj_scale":0.5}],
  "lj146": [{"i":1,"j":2,"c":0.0556}],
  "virtual_sites": [
    {"kind":"linear","site":5,"parents":[1,2],"coefficients":[0.5,0.5]}
  ],
  "constraints": [{"i":1,"j":2,"distance_nm":0.09572}],
  "input_diagnostics": [
    {"code":"box_mismatch","severity":"warning","message":"...",
     "atoms":[],"residues":[]}
  ],
  "net_charge": 2.0
}
```

Rules:

* Atom order is **the parameter-file order**; it is never re-sorted. `atoms[i]`
  is 1-based atom `i`. All term indices are 1-based.
* `scheme/family` is `amber` or `charmm`. `units` must be exactly
  `"nm-kj-mol-e-amu"`.
* `box.kind` is `none | orthorhombic | triclinic`. Triclinic uses
  `vectors_nm` (3 rows, columns are lattice vectors). No other key is present.
* `dihedrals` is one entry per **quadruple**, with **all multiplicities** in
  `terms`; `n >= 1`. `k` is the coefficient of
  `k*(1 + cos(n*phi - phase))` (Amber `Vn/2`, CHARMM `K`). Never collapse terms.
* `impropers` is harmonic: `k*(wrap(phi - phase))^2`, `k` in kJ/mol/rad^2.
  `phase` is radians. `phi` is the same signed torsion angle used by
  `Factors.PeriodicDihedral` (verified by FD tests).
* `exclusions` contains **only** pairs with zero nonbonded interaction after
  1-4 pairs are removed. `pair14` is the explicit 1-4 set; a pair may not appear
  in both. `coulomb_scale`, `lj_scale` are multipliers on the *full* pair
  interaction (Amber `1/SCEE`, `1/SCNB`; CHARMM 1.0).
* `cmaps` carry periodic bicubic maps with 8 slots `(a,b,c,d,b,c,d,e)`,
  `resolution n`, and a flat `grid` of length `n^2` in the order
  `grid[s + n*(t-1)]` (s is the φ index). The builder emits one
  `ChemistryCMAP.CMAPCorrection` per map. The extractor verified against
  OpenMM that the signed torsion convention matches `Factors.dihedral_angle_grad`;
  no axis reversal is applied.
* `lj146` is the AMBER 1-6-4 list: each entry contributes `-C/r^4` (C in
  kJ/mol·nm⁴) and must not overlap an exclusion or 1-4 pair. It is a separate
  pair term, never a σ/ε rescaling.
* Virtual sites are carried structurally; execution is refused by
  `ResolvedSystem`/`SystemContract`, exactly as in Phase 1.
* `input_diagnostics` are extractor findings (e.g. `box_mismatch`), not engine
  results. They are preserved and surfaced by the audit.
* `net_charge = sum(charges)` is informational; the difference from the sum must
  be <= 1e-6. It is never used to rescale charges.
* No NaN/Inf anywhere; JSON must be finite. Serialization is deterministic:
  `json.dump(..., sort_keys=True, separators=(',', ':'), allow_nan=False)` plus
  a trailing newline.

## 2. Julia schema API (`FoundationCore.Chemistry`)

```julia
Chemistry.read_normalized(path) -> NormalizedSystem
Chemistry.validate_normalized!(ns) -> ns        # typed ChemistryError
Chemistry.write_normalized(ns, path) -> path    # deterministic, round trip
Chemistry.normalization_summary(ns) -> NamedTuple # counts, net charge, family
```

Types (fields are exactly these names):

```julia
NormalizedAtom(name, mass, charge, element, residue_index, residue_name,
               sigma_nm, epsilon_kj)
NormalizedBox(kind::Symbol, lengths_nm::NTuple{3,Float64},
              vectors_nm::Matrix{Float64})   # vectors ignored unless triclinic
BondTerm(i,j,k,r0_nm)
AngleTerm(i,j,k,k,theta0_rad)
UreyBradleyTerm(i,j,k,r0_nm)
TorsionTerm(k,n,phase_rad)
DihedralTerm(i,j,k,l,terms::Vector{TorsionTerm})
ImproperTerm(i,j,k,l,k,phase_rad)
CmapTerm(indices::NTuple{8,Int}, resolution::Int, grid::Vector{Float64})
Pair14Term(i,j,coulomb_scale,lj_scale)
VirtualSiteTerm(kind::Symbol, site::Int, parents::Vector{Int},
                coefficients::Vector{Float64})
ConstraintTerm(i,j,distance_nm)
ImportProvenance(source, force_field, source_sha256, coordinate_source,
                 coordinate_sha256, generator, generator_version, notes)
DiagnosticRecord(code,severity,message,atoms,residues)
NormalizedSystem(schema, family, units, provenance, atoms, box, bonds, angles,
                 urey_bradleys, dihedrals, impropers, cmaps, exclusions,
                 pair14, virtual_sites, constraints, input_diagnostics,
                 net_charge)
```

`ChemistryError(code::Symbol, message::String, path::String)` with codes
`:io`, `:json`, `:schema`, `:units`, `:family`, `:shape`, `:atom`, `:index`,
`:duplicate`, `:box`, `:parameter`, `:provenance`, `:charge`, `:pair`,
`:unsupported_term`, `:protocol`, `:geometry`.

## 3. Builder API (`FoundationCore.Chemistry`)

```julia
SimulationProtocol(; nonbonded = :pme,      # :pme (periodic), :direct (NoBox)
                   cutoff_nm = 1.0,
                   lj_mode = :trunc,        # :trunc (reference parity) or :lfs
                   pme_alpha = 3.5,        # nm^-1 (rcut 1.0 nm; do not use the Å^-1 value)
                   pme_grid = nothing,
                   pme_order = 4,
                   water_model = :flexible, # :flexible | :rigid
                   background = :uniform,   # :uniform | :none
                   temperature_k = 298.15)
Chemistry.default_protocol(ns; kwargs...) -> SimulationProtocol
Chemistry.build_resolved(ns; protocol = Chemistry.default_protocol(ns)) -> ResolvedSystem
```

Builder guarantees:

* Emits `HarmonicBond`, `HarmonicAngle`, `ChemistryFactors.UreyBradley`,
  `PeriodicDihedral` (one term) or `ChemistryFactors.MultiPeriodicDihedral`
  (several terms), `ChemistryFactors.HarmonicImproper`.
* Periodic: one `StreamedLJ.StreamedLJFactor` + one `PME.PMEFactor`, with
  exclusions and 1-4 scales in both. `PMEFactor` prefactor is
  `units.coulomb_prefactor`; `PairCorrections` uses `excluded` (0) and
  `scaled` (1/coulomb 1-4 scale).
* Non-periodic: one `ChemistryFactors.PairTableNonbonded` with all pairs not in
  `exclusions`; 1-4 pairs carry the scale. No cutoff unless protocol asks.
* `ChemicalTopology` carries bonds/angles/dihedrals/impropers as metadata,
  `exclusions`, and `scales` with `(lj_scale, coulomb_scale)`.
* Triclinic geometry is retained by the normalized record but `build_resolved`
  refuses nonbonded execution with `ChemistryError(:geometry, ...)` (Phase 3).
* NBFIX, virtual sites requiring execution, and unsupported terms are typed
  refusals, never dropped. CMAP is built as a `ChemistryCMAP.CMAPCorrection`;
  1-6-4 `-C/r^4` is built as `PairC64Table`.
* `component_names` are unique and stable: `bond_1`, `angle_1`, `urey_1`,
  `dihedral_1`, `improper_1`, `lj`, `pme` (or `nonbonded`).
* The returned `ResolvedSystem` declares only
  `[:energy, :cartesian_forces, :component_energies]`.

## 4. Diagnostics API

```julia
Chemistry.audit_parameterized(ns, q; protocol = Chemistry.default_protocol(ns),
                              bond_tolerance_nm = 0.05) -> AuditReport
Chemistry.audit_raw_structure(path; kwargs...) -> RawAuditReport
```

`AuditReport` has `diagnostics::Vector{DiagnosticRecord}`, `provable_errors`,
`warnings`, `choices` (all vectors of `DiagnosticRecord`), plus
`counts::NamedTuple`. Checks: finite/shape, coincident atoms, bond deviation,
singular angles, 1-4 pair outside the nonbonded cutoff, net-charge/background
choice, box conflicts surfaced from `input_diagnostics`, zero LJ parameters,
and polymer C–N chain breaks. Raw PDB diagnosis delegates to `Prepare` and maps
its findings; it never fabricates parameters.

## 5. Five-class fixture manifest

`chemistry/fixtures/manifest.json` lists each class with `id`,
`class`, `family`, `parameter_file`, `coordinate_file`, `force_field`,
`status` (`ready`/`blocked`), and for blocked entries a `blocker` string.
Paths are relative to the repository root. Optional generated fixtures name a
`generator` instead of a parameter file.

The five ready classes are `water_ions`, `solvated_protein`, `dna`, `membrane`
(generated into the external cache) and `protein_ligand` (1XEP T4 lysozyme +
catechol, 2615-atom solute, `NoCutoff`). The 1XEP fixture's provenance, MIT
license, pose alignment and derivation are in
`protein_ligand/PROVENANCE.json`. A fixture is counted only with a complete
matching parameter/coordinate pair and an independently generated reference.

Reference artifacts live in `chemistry/reference/`:
`<id>.json` metadata and `<id>.f32` little-endian `Float32` forces for each
configuration in sequence. Per-term audit artifacts live in
`chemistry/reference/term_audit/`. The periodic reference uses
`ewaldErrorTolerance = 1e-6` and disables both the `NonbondedForce` dispersion
correction and the `CustomNonbondedForce` 1-6-4 long-range correction; the
protocol itself is regression-tested in `tools/test_reference_protocol.py`.
Coordinates are regenerated from the source coordinate file by the shared
`perturb` rule:

```
q[0,i] += 0.02 * config * sin(0.37*(i+1) + 1.1)
q[1,i] += 0.02 * config * sin(0.37*(i+1) + 2.2)
q[2,i] += 0.02 * config * sin(0.37*(i+1) + 3.3)
```

with i 0-based, for `config >= 1`; `config == 0` is unperturbed. Config 1
uses amplitude 0.02 nm, config 2 uses 0.04 nm, so the distorted configurations
are distinct. Python and
Julia implementations must use double precision for the sine argument and
addition.

## 6. Ownership / parallel work

* Orchestrator owns `FoundationCore.jl`, `Project.toml`, `Manifest.toml`,
  `test/run_all.jl`, `README.md`, `validation/STATUS.md`, `RESOLVED_SYSTEM.md`,
  `CHEMISTRY.md`, this file, and the fixture manifest.
* `ChemistryFactors.jl` + `test_chemistry_factors.jl`: factor worker.
* `ChemistrySchema.jl`: schema/parser/validation (orchestrator).
* `ChemistryBuild.jl` + `test_chemistry_build.jl`: builder worker.
* `ChemistryDiagnostics.jl` + `test_chemistry_diagnostics.jl`: diagnostics
  worker.
* `chemistry/tools/*.py`: extractor/reference workers.
* `test_chemistry_fixtures.jl`: orchestrator/verifier after artifacts exist.
* `protein_ligand/`: ligand fixture, provenance and its verification tool;
  `term_audit.py`/`exact_reference.py`: per-term/independent-oracle audit;
  `register_fixture.py` + `EXTENDING.md`: extension workflow.
* Registrations, the schema and `STATUS.md` are reserved for the orchestrator;
  workers add new files and report diffs rather than editing shared state.
