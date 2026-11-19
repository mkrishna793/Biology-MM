# Phase-2 chemistry fixtures and independent OpenMM references

This directory holds the five-class fixture manifest and the provenance record
for the reference artifacts in `../reference/`. The references are produced by
OpenMM loading the *parameter file itself* (`AmberPrmtopFile`, or
`CharmmPsfFile` + `CharmmParameterSet`), never a normalized intermediate from
this repository. They are an independent oracle, not engine output.

## Fixture set

| id | class | input | N | configs |
| --- | --- | --- | ---: | ---: |
| `water_ions` | water + ions | `wrappers/python/tests/systems/Mg_water.prmtop` + `.inpcrd` (atom is ZN²⁺; AMBER 1-6-4 CCOEF present) | 2164 | 3 |
| `solvated_protein` | solvated protein | `wrappers/python/tests/systems/18protein.parm7` + `.rst7` | 21536 | 3 |
| `dna` | DNA | `wrappers/python/tests/systems/DNA_mbondi3.prmtop` + `.inpcrd` | 628 | 3 |
| `membrane` | lipid membrane | generated external `dppc.prmtop` + `dppc.inpcrd` | 28160 | 3 |
| `charmm_adapter` | CHARMM adapter (not a five-class fixture) | `methanol_ions.psf`/`.pdb` + CGenFF/water/ions parameters | 8 | 3 |
| `protein_ligand` | protein–ligand complex | `chemistry/fixtures/protein_ligand/protein_ligand.prmtop` + `.inpcrd` (1XEP T4 lysozyme + catechol, solute only) | 2615 | 3 |

`protein_ligand` is a real, licensed complex: PDB 1XEP T4 lysozyme with a
catechol ligand (residue `UNL`, 14 atoms), prepared with ff14SB protein and an
OpenForceField-parameterized ligand. The source preparation and MIT license are
recorded in `protein_ligand/PROVENANCE.json`; the source is
`ljmartin/mmpbsa_from_openmm` at commit `48664084`, and the pose is the
deposited 1XEP crystal pose, not a constructed pose. The fixture is solute-only
(waters/ions stripped) and has no box, so the reference is an exact `NoCutoff`
comparison. An earlier draft's attribution to a different tutorial repository
was wrong and is explicitly rejected in the provenance record. The original
4TVP-DMJ PSF and T4-lysozyme/p-xylene PDB remain unusable as complete
coordinate/topology/parameter triples and are not counted.

The membrane prmtop is ~12 MB and is **never committed**. It lives in the
external cache (`~/.cache/biology-mm/chemistry-fixtures` by
default, overridable with `CHEM_FIXTURE_CACHE` for the Julia test).

## Regeneration (exact commands)

From the repository root:

```sh
PY=~/.cache/biology-mm/.venv/bin/python
CACHE=~/.cache/biology-mm/chemistry-fixtures

# 1. Membrane parameter file (external cache; verifies with AmberPrmtopFile)
$PY chemistry/tools/make_generated_fixtures.py --out "$CACHE" --force

# 2. Independent OpenMM references (all five; use --fixture <id> for one)
$PY chemistry/tools/make_reference.py --fixture all --force

# 3. Consumer (orchestrator-owned; needs the extractors)
cd julia
JULIA_DEPOT_PATH=~/.cache/biology-mm/julia-depot \
  ~/.cache/biology-mm/julia-1.13.3/bin/julia --project=. \
  test/test_chemistry_fixtures.jl
```

Re-running step 2 is deterministic on the same machine/platform. Without
`--force` an existing artifact is reused (and re-checked) rather than rebuilt.

## OpenMM protocol

* Periodic fixtures: PME, `nonbondedCutoff = 1.0 nm`,
  `ewaldErrorTolerance = 1e-6`, `constraints=None`, `rigidWater=False`,
  `removeCMMotion=False`, `hydrogenMass=None`, dispersion correction disabled on
  every `NonbondedForce`, and `setUseLongRangeCorrection(False)` on every
  `CustomNonbondedForce` (the AMBER 1-6-4 term). The 1e-6 tolerance and the
  1-6-4 tail-correction choice are regression-tested in
  `../tools/test_reference_protocol.py`; at the earlier 5e-4 tolerance the
  reference itself carried a 5–6 kJ/mol PME error.
* `dna` (prmtop has no box), `protein_ligand`, and `charmm_adapter`: `NoCutoff`,
  same flags. The
  CHARMM PDB carries a ~1.2 nm box, too small for a 1.0 nm PME cutoff; it is
  recorded but not used.
* The **prmtop box is authoritative** and is applied with
  `Context.setPeriodicBoxVectors(...)`. The coordinate-file box is recorded only
  (`coordinate_box_nm`) and never used. `Mg_water.prmtop` and
  `Mg_water.inpcrd` disagree; that difference is a fixture diagnostic, not a
  reason to override the topology.
* Platform `CPU`, `Threads=1`, `DeterministicForces=true`.

## Artifact encoding

`<id>.json` (deterministic, sorted keys, trailing newline) records: schema,
fixture/class, source paths + SHA-256, generator, OpenMM version, platform,
protocol, `periodic`, `box_nm`, `coordinate_box_nm`, `n_atoms`, `n_configs`,
the perturbation rule, the force-group map, per-config energies and force
summaries, and the force-file name/encoding.

Term split: every bonded force is force group 0, every nonbonded force
(`NonbondedForce`, and `CustomNonbondedForce` when present, e.g. the AMBER 1/r⁴
term or CHARMM NBFIX) is group 1. `bonded_kj_mol + nonbonded_kj_mol` equals
`energy_kj_mol` to 1e-6 relative; any other assigned group is listed in
`force_groups`. `force_norm` is the Frobenius norm, `force_absmax` the largest
component magnitude (kJ mol⁻¹ nm⁻¹).

`<id>.f32`: little-endian `Float32`, config-major (config 0 first), and within a
config the flat C-order sequence of the OpenMM `(N,3)` force array:
`x1,y1,z1,x2,y2,z2,…`. Julia reads it as `reshape(flat, 3, N)`; equivalently it
is `forces.reshape(3,N).astype('<f4').tobytes()` on the `(N,3)` array. File
length is exactly `n_configs * 3 * N * 4` bytes.

Configurations: config 0 is the source coordinates; every config ≥ 1 applies the
`INTERFACE.md` rule to **all three** components, with 0-based `i`, double
precision, in nm:

```
q[d,i] += 0.02 * config * sin(0.37*(i+1) + 1.1*(d+1))      d = 0,1,2
```

Config 1 has amplitude 0.02 nm, config 2 amplitude 0.04 nm, so every stored
config is distinct. All five references keep configs 0, 1 and 2. No coordinates
are stored. Generators assert `max|F| > 1e-3 kJ mol⁻¹ nm⁻¹` and that each
perturbed config changes the energy by more than `1e-3 kJ/mol`.

Measured artifact sizes: `water_ions.f32` 77 904 B, `solvated_protein.f32`
775 296 B, `dna.f32` 22 608 B, `membrane.f32` 1 013 760 B, `charmm_adapter.f32`
288 B, `protein_ligand.f32` 94 140 B; JSON 3.1–4.0 kB each; total reference data
≈ 2.00 MB (2 004 186 B excluding `capabilities.json`). The ligand prmtop/inpcrd/pdb fixture is a separate
1.94 MB in the repository; it is small enough to keep offline-reproducible.

## Provenance and licensing

* OpenMM 8.6.1 and ParmEd 4.3.1 are MIT-licensed; used only as reference/oracle
  and export tooling.
* The 1XEP protein–ligand fixture is derived from the MIT-licensed
  `ljmartin/mmpbsa_from_openmm` preparation at commit `48664084`, with ff14SB
  protein and OpenForceField ligand parameters. The source commit, file hashes
  and derivation are in `protein_ligand/PROVENANCE.json`; PDB 1XEP is public
  structure data. The fixture is redistributed as a small derived solute
  prmtop/inpcrd/pdb and is used only as a test input.
* The AMBER prmtop/inpcrd and CHARMM psf/prm/pdb inputs are third-party
  force-field test data **already vendored in this checkout** and are used here
  as test inputs; they are not redistributed by these tools.
* The membrane system is built from OpenMM's MIT-licensed
  `openmm/app/data/DPPC.pdb` with the `amber19/lipid21.xml` + `amber19/tip3p.xml`
  templates; only the generated external prmtop/inpcrd pair is written, and only
  to the cache.
