# Phase 2: parameter-file import, resolved systems, and chemistry-class evidence

Companion to `README.md`, `RESOLVED_SYSTEM.md` and `validation/STATUS.md`.
Schema `v1.0.0` (`FoundationCore.Chemistry.CHEM_SCHEMA_VERSION`). This document
states what is implemented, the exact commands, the unit conversions, the five
fixture classes, and the release boundary. Measured numbers are transcribed only
from a completed run; unrun checks are labelled **BLOCKED**, never PASS.

Labels: **implemented** (code exists and is exercised by a suite), **refused**
(typed error by design), **blocked** (missing external input), **Phase 3** (not
claimed).

---

## 1. Supported input paths and terms

| Family | Input | Terms imported | Status |
| --- | --- | --- | --- |
| AMBER | `.prmtop`/`.parm7` + `.inpcrd`/`.rst7` | bonds, angles, all proper-torsion multiplicities, harmonic impropers, periodic CMAP, 1-2/1-3 exclusions, explicit 1-4 Coulomb/LJ scales, per-atom LJ (Lorentz–Berthelot), 1-6-4 `-C/r^4`, box | implemented |
| CHARMM | `.psf` + parameter files (+ optional PDB/CRD coordinates) | bonds, angles, Urey–Bradley, multi-term dihedrals, harmonic impropers, exclusions/1-4, per-atom LJ | adapter implemented for resolved atom types; CMAP maps are emitted, NBFIX is a typed refusal, and no CHARMM reference comparison has been run (structural only) |
| Either | periodic orthorhombic | `StreamedLJ` + scalar-charge `PME` | implemented |
| Either | no box | explicit `PairTableNonbonded` (all pairs, no cutoff) | implemented |
| Either | triclinic box | geometry retained in the normalized record; PME periodic path | implemented (PME path) |
| Either | CMAP | periodic bicubic map (`ChemistryCMAP.CMAPCorrection`), analytic forces | implemented |
| Either | NBFIX / off-diagonal LJ | detected | **refused** (not silently diagonalized) |
| AMBER | 1-6-4 `LENNARD_JONES_CCOEF` | separate `PairC64Table` (`-C/r^4`) | implemented |
| AMBER | TIP4P/5P virtual sites | carried structurally | representation only; execution refused by Phase 1 |
| Both | 10-12 (HBOND) terms | detected | **refused** |

The parser is ParmEd (a mature format reader); the physics comes from our own
Julia factors. OpenMM is used only to generate independent reference numbers,
never as the executing backend.

## 2. Dependencies and reproducibility

* Julia 1.13.3, project `julia`; `JSON.jl` 1.13.4 handles chemistry records; `FFTW` 1.10.0
  is also a declared package dependency for PME.
* Python 3.12 in `~/.cache/biology-mm/.venv` with
  `parmed==4.3.1` and `openmm==8.6.1`. ParmEd is used offline to
  read parameter files; OpenMM only in `chemistry/tools/make_reference.py`.
* Fixture parameters are the parameter files already vendored in this checkout
  (`wrappers/python/tests/systems/`, `wrappers/python/openmm/app/data/`). Their
  provenance and SHA-256 are recorded in the normalized JSON and reference JSON.
  The membrane fixture is generated from OpenMM's app-data `DPPC.pdb` and
  `amber19/lipid21.xml`; the 12 MB prmtop is kept in the external cache, not in
  the repository.
* No network access is required to rerun the committed checks; only the
  optional reference regeneration needs the Python environment. Exact commands:

  ```sh
  PY=~/.cache/biology-mm/.venv/bin/python
  $PY chemistry/tools/extract_amber.py \
      --parm wrappers/python/tests/systems/18protein.parm7 \
      --crd  wrappers/python/tests/systems/18protein.rst7 \
      --out /tmp/protein.json --force-field "AMBER prmtop; 18protein test set"
  $PY chemistry/tools/make_reference.py --fixture all --force
  $PY chemistry/tools/make_generated_fixtures.py --force
  $PY -m pytest -q chemistry/tools/test_extractors.py
  ```

  `extract_charmm.py` takes `--parm PSF --prm PARM1 --prm PARM2 ... [--crd]`.

## 3. Units and conversions

All normalized numbers are **nm, kJ/mol, e, amu, radians**. One
`Types.Units` record (`Chemistry.CHEM_UNITS`) is attached to every built
system; factors never hard-code a Coulomb constant.

| Quantity | Native | Imported |
| --- | --- | --- |
| length | Å | nm (`×0.1`) |
| energy | kcal/mol | kJ/mol (`×4.184`) |
| bond/UB `k` | kcal mol⁻¹ Å⁻² | kJ mol⁻¹ nm⁻² (`×418.4`) |
| angle/improper `k` | kcal mol⁻¹ rad⁻² | kJ mol⁻¹ rad⁻² (`×4.184`) |
| proper torsion | `Vn` kcal/mol | `k = Vn/2` in `k(1+cos(nφ−φ0))` |
| CHARMM UB | `k(r−r0)²` | same form, no `1/2` |
| charge | e | e |
| Coulomb prefactor | — | `138.9354576449822 kJ mol⁻¹ nm e⁻²` |
| mass | amu | amu |
| temperature | K | K (metadata) |

AMBER 1-4: `coulomb_scale = 1/SCEE`, `lj_scale = 1/SCNB`; CHARMM: 1.0.
Exclusions are the parameter file's explicit list **minus** the 1-4 pairs, so a
pair is never both excluded and scaled (validation enforces this).

## 4. API example (no Julia source edits)

```julia
using FoundationCore
const Ch = FoundationCore.Chemistry
const SC = FoundationCore.SystemContract

ns = Ch.read_normalized("/path/to/normalized.json")   # strict v1.0.0 record
prot = Ch.default_protocol(ns; nonbonded = :pme, cutoff_nm = 1.0, lj_mode = :lfs)
rs = Ch.build_resolved(ns; protocol = prot)           # ResolvedSystem
q = zeros(3, Ch.n_atoms(ns)); q .= your_coordinates    # 3×N, nm
U, F = SC.energy_forces(rs, q)                         # kJ/mol, kJ/mol/nm
```

`lj_mode = :lfs` is the force-consistent choice for dynamics; `:trunc`
reproduces the unshifted 12-6 reference model and is refused by the integrator,
exactly as Phase 1 documents. `:pme` refuses a cutoff outside
`0 < rcut < min(L)/2`; `:direct` refuses a periodic box. Neither a populated
topology nor a box constitutes a capability claim.

## 5. Diagnostics

`Ch.audit_parameterized(ns, q; protocol=...)` returns typed records with atom
and residue references for coincident bonded atoms, bond-length deviations,
singular angles, 1-4 pairs outside the nonbonded cutoff, net-charge/background
choice, zero LJ parameters, preserved box conflicts, and polymer C–N chain
breaks. `Ch.audit_raw_structure(path)` maps `Prepare.diagnose_structure`'s
findings without repairing anything. A prepared parameterized topology is
authoritative for its protonation state; raw-structure ambiguity is not
projected onto it.

## 6. Five-class status

Exact commands:

```sh
export JULIA_DEPOT_PATH=~/.cache/biology-mm/julia-depot
JULIA=~/.cache/biology-mm/julia-1.13.3/bin/julia
cd julia
$JULIA --project=. test/test_chemistry_schema.jl
$JULIA --project=. test/test_chemistry_factors.jl
$JULIA --project=. test/test_chemistry_cmap.jl
$JULIA --project=. test/test_chemistry_build.jl
$JULIA --project=. test/test_chemistry_diagnostics.jl
$JULIA --project=. test/test_chemistry_mutations.jl
$JULIA --project=. test/test_chemistry_fixtures.jl
```

| Class | Fixture | Loads + own forces | Independent reference | Status |
| --- | --- | --- | --- | --- |
| water + ions | `Mg_water.prmtop` (2164 atoms; ZN²⁺ + TIP3P) | yes | OpenMM PME | PASS |
| solvated protein | `18protein.parm7` (21536 atoms) | yes | OpenMM PME + 16 CMAP maps | PASS |
| DNA | `DNA_mbondi3.prmtop` (628 atoms, no box) | yes | OpenMM NoCutoff | PASS |
| lipid membrane | generated DPPC bilayer (28160 atoms) | yes | OpenMM PME | PASS |
| protein–ligand | 1XEP T4 lysozyme + catechol (2615-atom solute) | yes | OpenMM NoCutoff | PASS |

Protein–ligand fixture: PDB 1XEP T4 lysozyme L99A/M102Q + a 14-atom catechol
ligand (`UNL`), ff14SB protein + OpenForceField ligand parameters, net charge
+8, solute only, no box. The parameters/coordinates come from the MIT-licensed
`ljmartin/mmpbsa_from_openmm` preparation at commit `48664084`; the exact source
file SHA-256 values, the solute-stripping/derivation steps and the 1XEP pose
verification (protein heavy atoms bit-identical; catechol = CAQ altloc A within
0.013 nm) are in `chemistry/fixtures/protein_ligand/PROVENANCE.json`. The pose
is the deposited crystal pose, not a constructed pose. The earlier 4TVP-DMJ PSF
and T4-lysozyme/p-xylene PDB remain unusable as complete
coordinate/topology/parameter triples.

### Measured checks (final run)

| Fixture | Configs | max \|dU\| (kJ/mol) | max relative energy | max force error (kJ mol⁻¹ nm⁻¹) | max RMS force rel. |
| --- | ---: | ---: | ---: | ---: | ---: |
| water + ions | 3 | `1.39e-2` | `1.86e-6` | `4.70` | `3.19e-5` |
| solvated protein | 3 | `1.10e-1` | `2.02e-6` | `3.73` | `2.96e-5` |
| DNA | 3 | `5.37e-1` | `6.73e-7` | `179.9` (on `5.1e5` max force) | `2.04e-6` |
| lipid membrane | 3 | `1.90e-1` | `8.64e-8` | `8.39` | `1.17e-4` |
| protein–ligand | 3 | `1.70e-1` | `3.85e-7` | `2.10` | `1.94e-6` |

Reference protocol: PME with `cutoff 1.0 nm`, `ewaldErrorTolerance 1e-6`,
grid converged by both codes, no `NonbondedForce` dispersion correction, no
`CustomNonbondedForce` 1-6-4 long-range correction, uniform neutralizing
background, flexible water (constraint protocol separate). DNA and the
protein–ligand solute are non-periodic explicit all-pair (`NoCutoff`)
references. The numbers are static energy/force comparisons, not trajectories.

**Two reference-side defects were found and corrected in this pass** (they made
the own-engine error look larger, not smaller):

1. OpenMM enables a long-range tail correction on the AMBER 1-6-4
   `CustomNonbondedForce` by default. The water+ions reference was therefore
   15.4 kJ/mol more negative than the truncated `-C/r^4` model; with the
   correction off, the 1-6-4 term agrees to `9e-6` kJ/mol and the total water
   residual falls from 16.1 to 0.014 kJ/mol.
2. The reference's `ewaldErrorTolerance = 5e-4` left a 5.7 kJ/mol PME error in
   the solvated-protein reference and 6.2 kJ/mol in the membrane reference.
   Re-running the same reference at 1e-6 converges to the values the engine
   already computed (agreement 0.1 kJ/mol); the 1e-6 protocol is now pinned by
   `tools/test_reference_protocol.py`.

The DNA `maxF = 179.9` is a single-atom absolute difference on a `5.1e5`
force scale (`rmsF = 2.0e-6`); it is the large bonded force at the most
distorted configuration, not a nonbonded mapping error. The per-term audit in
`chemistry/reference/term_audit/` reports each force class separately so a
total-energy cancellation cannot hide a nonbonded defect.

The ligand gate also has mutation controls: a +0.10 e ligand charge, a doubled
ligand bond prefactor and a ligand 1-4 LJ scale changed to 1.0 produce
`|dU| = 48.3 / 0.245 / 17.4 kJ/mol` respectively, so the ligand parameters are
actually exercised rather than shadowed by the protein.

Final full verification (frozen tree): `sh verify.sh`, exit **0**,
all Julia suites, wall time recorded in `validation/STATUS.md`; Lean/Rocq/
experiment/NIST/Molly checks pass. The chemistry gate is **PASS (5/5)** for the
five classes, with the explicit boundaries in §7.

## 7. Release boundary

Phase 2 is about **parameter correctness and potential-energy/force fidelity**
for imported files. It does not claim: equilibrated trajectories, throughput,
production interacting-system NPT, virtual-site execution, NBFIX, ligand
parameterization, protonation prediction, or binding affinities. Separate current
targeted tests pass PME stress/triclinic derivatives and scoped interacting
LJ/TIP3P NPT checks; see `validation/STATUS.md` §11. Direct-Ewald stress remains
refused. Static energy/force agreement is not a dynamics result.
