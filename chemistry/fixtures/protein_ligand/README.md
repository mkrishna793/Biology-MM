# `protein_ligand` — T4 lysozyme L99A/M102Q + catechol (PDB 1XEP)

A single solute-only protein–ligand parameter set used as the Phase-2 AMBER
importer fixture. It exists so the engine can be compared against OpenMM on a
real ff14SB + OpenFF complex with no solvent, no ions and no periodic box.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `protein_ligand.prmtop` | 1 640 178 | `4e9c9333c25987fa0a5a1ca710db3ed2244079a4103af18d9b22298062046d1c` |
| `protein_ligand.inpcrd` | 95 470 | `bda9d2e1311aca352314d9f1ffef1ab12dfa3e87d0392722d5b4a956ef4a799f` |
| `protein_ligand.pdb` | 206 679 | `719f7f5c41b7d1847d512b08d659c4c955e4badeae1c8e70e6f0bf1e88feb1fe` |

Full machine-readable record: [`PROVENANCE.json`](PROVENANCE.json).

## Contents

* 2615 atoms, 163 residues: 2601 protein atoms (162 residues, 1XEP residues
  1–162, i.e. MET1…LYS162 of T4 lysozyme including the L99A/M102Q sites; ParmEd
  numbers the same residues 0–161) plus one `UNL` residue of 14 atoms.
* `UNL` is catechol (benzene-1,2-diol): C6H6O2, one six-membered all-carbon ring,
  two C–O bonds, four aromatic C–H and two O–H. It is ParmEd residue index 162
  and atoms 2601–2614 (0-based), i.e. the ligand is written *after* the protein.
* Net charge +8.0 e (`7.999999926` before rounding); the ligand itself sums to
  0.0 e.
* No box: no `%FLAG BOX_DIMENSIONS` in the prmtop and no box in the inpcrd or
  PDB. The system is solute-only — no waters, no Na⁺/Cl⁻.
* 2635 bonds; bond lengths span 0.1006–0.1817 nm, median 0.1215 nm.
* Minimum ligand-heavy ↔ protein-heavy distance 0.2609 nm: the closest pair is
  the catechol hydroxyl oxygen `O1` and `OE1` of GLN 102 (the M102Q affinity
  site; ParmEd residue number 101), i.e. a hydrogen-bond contact rather than a
  clash, as in the deposited structure.
* `protein_ligand.pdb` is a sidecar for inspection and for the importer's
  structure reader: 2615 `ATOM` records in prmtop atom order, residue names and
  atom names identical to the prmtop, coordinates equal to the inpcrd to the
  0.001 nm PDB print precision.

## Provenance (short form)

The coordinates **are** the 1XEP crystal pose. The protein heavy atoms are
bit-identical to 1XEP chain A (1290/1290 atoms, deviation 0.000 Å), and the
catechol matches deposited `CAQ` altloc A to 0.0075 Å RMSD (max 0.0133 Å) after
mapping the parameterization's ring order onto the deposited atom names. No
rigid-body transform, docking, hand placement or post-hoc minimization was
applied to the pose.

Parameters: AMBER **ff14SB** for the protein; the catechol `UNL` is
OpenForceField-parameterized. This is checked, not assumed: 159 of the 162
protein residues have partial charges byte-identical to OpenMM's
`amber14/protein.ff14SB.xml` templates (max deviation 2.2e-9 e); the three
exclusions are the two terminal residues (cap atoms `H2`/`H3` and `OXT`) and
`HIS 30`, whose protonation template OpenMM names `HID`/`HIE`/`HIP`. Note that
the prmtop's `AMBER_ATOM_TYPE` strings are element-class labels (`N1`, `H1`,
`C1`, …) with 21 nonbonded type indices sharing 14 labels — the layout of an
OpenMM System exported through ParmEd, not a `tleap` prmtop. Neither the fixture
nor its recorded upstream material states the exact Open Force Field Toolkit /
force-field version used for the ligand — that is a real provenance gap, and the
reference is therefore defined by these committed parameters, not by re-running
a ligand parameterization.

**Source and attribution.** The fixture is derived from the MIT-licensed
`https://github.com/ljmartin/mmpbsa_from_openmm` example at commit
`48664084b88e7cf4f731e5ab4082a7edd8198345` (2020-10-21), from
`openmm_simulation/complex_plus_water.parm7` (parent solvated 1XEP + catechol
topology) and `openmm_simulation/complex_coords.pdb`. Exact source-file SHA-256
values, the MIT license hash and the strip/remove-box derivation steps are in
`PROVENANCE.json`. The parent prmtop has 18 461 atoms (protein + `UNL` catechol +
HOH/Na/Cl, triclinic 60° box); this fixture strips the 15 846 solvent/ion atoms
and the box, leaving 2 615 atoms. The prmtop here carries a local `%VERSION`
stamp because it is a ParmEd export of that pinned source, not a byte copy.

An earlier integration draft wrongly attributed the fixture to
`freeenergylab/amber_tutorials`; that repository's `MDforIL2/1PY2` system is
IL2 + SP4206 (27 569 atoms / 8 625 residues), has no matching
`complex_coords.pdb`, and is not this system. The wrong claim is recorded and
rejected in `PROVENANCE.json` → `attribution_correction`. The verified fact is
the structural one: this is the deposited 1XEP pose.

## Why not `4TVP`-DMJ or T4-lysozyme-L99A-*p*-xylene?

Both were considered for the same slot and rejected earlier in
`CHEMISTRY.md` (protein–ligand blocker note), for input-completeness
reasons rather than pose reasons:

* **T4-lysozyme-L99A-*p*-xylene** — the only file present is an implicit-solvent
  PDB whose ligand residue is `TMP`, which has no matching template in the
  shipped Amber parameter files. There is no prmtop/psf carrying its charges, so
  the complex cannot be loaded without parameterizing the ligand (antechamber /
  GAFF / OpenFF are not installed and are deliberately not fetched). An
  unparameterized ligand is not a force-field fixture.
* **`4TVP`-DMJ** — the CHARMM system (`4TVP-dmj_wat-ion.psf`) has **no
  coordinate file**, so its 1-4/DMJ geometry cannot be reconstructed from the
  checkout. DMJ is also a stereochemically dense iminosugar whose protonation
  and ring puckering make a from-scratch ligand parameterization a separate
  validation project, not a fixture.

A hand-placed or docked pose was ruled out as well: it would not be a realistic
complex, so it would not test the importer against anything physical.

## The config-0 check (exact protocol and numbers)

The phase-2 protocol is the one in `chemistry/INTERFACE.md` and
`chemistry/fixtures/README.md`:

* OpenMM reads the parameter file itself: `AmberPrmtopFile` +
  `AmberInpcrdFile` (never a normalized intermediate produced by this repo).
* `createSystem(nonbondedMethod=NoCutoff, constraints=None, rigidWater=False,
  removeCMMotion=False, hydrogenMass=None)`; dispersion correction disabled on
  every `NonbondedForce`.
* Force groups: 0 = bonded (`HarmonicBondForce`, `HarmonicAngleForce`,
  `PeriodicTorsionForce`), 1 = `NonbondedForce` + `CustomNonbondedForce`.
* Platform `CPU`, `Threads=1`, `DeterministicForces=true`.
* Config 0 is the source coordinates. Configs 1 and 2 apply
  `q[d,i] += 0.02*config*sin(0.37*(i+1) + 1.1*(d+1))` nm (config 0 unperturbed);
  only energies/forces are stored, never coordinates.

Measured, config 0 (committed reference
`chemistry/reference/protein_ligand.json`, `n_atoms` 2615,
`n_configs` 3, `periodic` false, `box_nm` null):

| quantity | OpenMM 8.6.1 (double) | engine (Float32) | delta |
| --- | ---: | ---: | ---: |
| potential energy (kJ/mol) | `42780.460312603434` | `42780.4667` | `0.0064` |
| max &#124;force&#124; (kJ mol⁻¹ nm⁻¹) | `13751.46569680516` | `13751.4659` | `0.0002` |

Bonded/nonbonded split of the reference energy: `65477.770469547504` +
`−22697.31015694407` kJ/mol; Frobenius force norm `226763.18302746926`
kJ mol⁻¹ nm⁻¹. The stored Float32 force array
(`reference/protein_ligand.f32`, 94 140 B = 3 configs × 3 × 2615 × 4 B) agrees
with the double recomputation to at most `4.9e-4` kJ mol⁻¹ nm⁻¹ in config 0, so
the JSON and the `.f32` are the same oracle.

Verification gate (offline, reads only the committed files):

```sh
PY=~/.cache/biology-mm/.venv/bin/python
$PY chemistry/tools/verify_protein_ligand.py            # JSON report, exit 0
$PY -m pytest -q chemistry/tools/test_verify_protein_ligand.py
```

The verifier recomputes config 0 and checks it against the reference with
tolerances of **0.05 kJ/mol** (energy) and **0.05 kJ mol⁻¹ nm⁻¹** (max force),
and additionally checks: file hashes vs the reference record, 2615 atoms, one
`UNL` of 14 atoms (C6H6O2), net charge +8.0 ± 0.01, no NaN/Inf, prmtop↔inpcrd↔PDB
atom order, bond-length median 0.09–0.15 nm and max < 0.20 nm, ligand–protein
minimum heavy-atom separation 0.14–0.35 nm, the intact catechol bond graph
(one ring, 14 bonds, cycle rank 1, two divalent oxygens, six monovalent
hydrogens), and ff14SB partial charges on every templateable protein residue.

## What this fixture is *not*

* Not a trajectory, MD run, or ensemble: there are no velocities, no box and no
  time series. Only static energies/forces at three coordinate sets.
* Not a binding-affinity or free-energy result. Catechol is a small fragment
  ligand of the L99A/M102Q cavity; no ΔG, no alchemical estimate and no
  experimental comparison is claimed here.
* Not a solvated or membrane system and not a protonation-state study: the
  preparation's protonation is whatever ff14SB assigns by default (net +8 e).
* Not reproduced from the cited upstream repository — see the attribution
  caveat above and `PROVENANCE.json["discrepancies"]`. Do not cite
  `freeenergylab/amber_tutorials` as the source of these bytes.
