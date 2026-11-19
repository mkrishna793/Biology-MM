# Lane 4 — bonded-term convention closure

Additive lane. Nothing outside this directory is modified, and nothing here
changes `FoundationCore`, its `Project.toml`, or any other experiment.

| File | Role |
| --- | --- |
| `check.jl` | the measurement and its assertions (single entry point) |
| `run.sh` | wrapper: frozen references by default, `--refresh` reruns both stages |
| `reference/extract_reference.py` | stage 1: per-term parameters from InterMol + ParmEd (InterMol env) |
| `reference/terms.tsv` | **frozen** per-term parameter/geometry table |
| `reference/openmm_terms.py` | stage 2: builds standard OpenMM bonded forces per term and records energy + force |
| `reference/openmm_terms.tsv` | **frozen** OpenMM 8.6.1 per-term energies, forces and bonded totals |
| `reference/provenance.txt` | versions, hashes, URLs fetched, and what was refused |

Question. Our engine has exactly three bonded forms:

```
HarmonicBond(i,j,k,r0)          U = (k/2) (r - r0)^2
HarmonicAngle(i,j,k,k_th,th0)   U = (k_th/2) (theta - theta0)^2
PeriodicDihedral(i,j,k,l,k_phi,n,phi0)
                                U = k_phi [1 + cos(n*phi - phi0)]
```

Every other code writes "the same" physics with a different `1/2`, different
units, and the phase in a different place. This lane pins those differences to
an external source and reports what is, and is not, closed.

---

## 1. External source

**NIST SRSW was fetched and is not usable for per-term values.** The TraPPE-UA
ethane/n-octane/n-butane pages were retrieved on 2026-11-12 (URLs, page
structure and one verbatim ethane row are recorded in
`reference/provenance.txt`). They publish simulation metadata and
liquid–vapour coexistence tables — `rho_vap`, `rho_liq`, `p_sat`, `u_vap`,
`u_liq`, `ln z` — and cite Martin & Siepmann, *J. Phys. Chem. B* **102**, 2569
(1998) for the force field. They publish **no** bond, angle or torsion
parameters. Quoting a TraPPE bonded parameter from these pages would be
fabrication, so none is quoted.

**The convention gap is closed against a third-party engine, OpenMM 8.6.1**, in
two stages, over two GROMACS topologies shipped in InterMol's own test corpus
(same 9-atom OPLS-AA ethanol, `funct 1` bonds/angles, `funct 1` and `funct 9`
proper dihedrals):

* **Stage 1 — parameters (InterMol env, Python 3.10).** InterMol 0.1.2
  `GromacsParser` and ParmEd 4.3.1 `load_file` parse the same `.top`
  independently; the extractor refuses to write a row unless the two agree
  after unit conversion, so a silent drift in one parser cannot enter the
  table. `parmed.geometry.dihedral` supplies an independent dihedral **angle**
  implementation for the sign check.
* **Stage 2 — energies and forces (OpenMM env at
  `~/.cache/biology-mm/.venv/bin/python`, Python 3.12.15,
  OpenMM 8.6.1, platform `Reference`, double precision).**
  `reference/openmm_terms.py` takes the **ParmEd** columns, converts them with
  the declared mapping in §2, and builds one standard OpenMM
  `HarmonicBondForce`, `HarmonicAngleForce` or `PeriodicTorsionForce` per
  term. It then records the per-term energy, the per-term force vector, and
  the all-bonded total for the frozen geometry. `check.jl` compares our engine
  against those numbers term by term.

So the comparison is against a real external evaluator, not against a second
copy of the same formula. The documented closed forms in `check.jl` remain as a
*form* check (they exercise the `n` and `φ0` dependence over a φ grid and the
cis/trans sign); they are no longer the only evidence, and §5 lists them
separately from the OpenMM comparison.

## 2. Convention mapping (the deliverable)

| Term | GROMACS/ParmEd reference | Our factor | Mapping |
| --- | --- | --- | --- |
| bond | `funct 1`: `V = ½ k_b (r-b0)²`, kJ mol⁻¹ nm⁻²; ParmEd `V = k_p (r-req)²`, kcal mol⁻¹ Å⁻² | `½ k (r-r0)²` | `k = k_b`, `r0 = b0` — **same k, same ½**; ParmEd needs `k_b = 836.8 · k_p` (4.184 × 100 because of Å² and kcal, ×2 because ParmEd drops the ½) |
| angle | `funct 1`: `V = ½ k_θ (θ-θ0)²`, kJ mol⁻¹ rad⁻²; ParmEd `V = k_p (θ-θ0)²` | `½ k_θ (θ-θ0)²` | `k_θ = k_θ`, `θ0` same; ParmEd needs `k_θ = 8.368 · k_p` (4.184, ×2 for the missing ½; no length factor) |
| torsion | `funct 1/9`: `V = k_φ [1 + cos(nφ - φ_s)]`, φ_s in degrees; ParmEd `V = φ_k [1 + cos(per·φ - phase)]` | `k_φ [1 + cos(nφ - φ0)]` | `k_φ = k_φ` (**no ½**, exactly as we implement it), `n = multiplicity = per`, `φ0 = φ_s = phase` converted to radians. `4.184 · φ_k[kcal] = k_φ[kJ]` |
| φ itself | IUPAC: φ = 0 cis, ±180° trans; `cos φ = n̂1·n̂2`, `sin φ = (n̂1×n̂2)·b̂2` with `b1 = r_j-r_i`, `b2 = r_k-r_j`, `b3 = r_l-r_k` | `atan2((n1×n2)·b̂2, n1·n2)` with the same `b1,b2,b3` | identical definition, including sign — verified against ParmEd and against an engineered cis/trans fixture |

Two traps this table makes explicit: (a) a torsion has **no** ½ in either
convention, so copying the bond/angle halving onto `k_φ` is a 2× error; (b) the
phase is a *single* angle in the `n`-th harmonic — `cos(nφ - φ_s)` — not
"rotate φ by φ_s first", `cos(n(φ-φ_s))`.

## 3. Tolerances (fixed before measurement)

`check.jl` declares them as constants at the top of the file:

```
TOL_ENERGY = 1e-10      kJ/mol absolute, engine vs reference formula
TOL_PHI    = 1e-10      rad, engine φ vs ParmEd geometry.dihedral
TOL_FD     = 1e-6       relative, engine force vs central difference of engine U
TOL_COLUMN = 1e-9       relative, GROMACS k vs ParmEd k through 836.8/8.368/4.184
TOL_OMM_E  = 1e-8       kJ/mol, engine vs OpenMM per-term and total energy
TOL_OMM_F  = 1e-9       relative, engine vs OpenMM per-term and total force
DISCRIMINATION_MIN = 1.0 kJ/mol  a wrong convention must exceed this
```

`TOL_OMM_*` are deliberately looser than double-precision agreement because both
sides exchange numbers through decimal text (`terms.tsv`, `openmm_terms.tsv`).

`TOL_COLUMN` is what makes the frozen parameter table self-checking: the
GROMACS-unit and ParmEd-unit columns of `terms.tsv` come from different parsers,
so a drift in either one is caught even where an energy comparison that reads
`terms.tsv` on both sides could not see it.

The primary reference energies come from **OpenMM's own kernels** on standard
bonded forces built from the ParmEd columns. Separately, the documented closed
forms are re-evaluated in `check.jl` (ParmEd's `k(Δ)²` in kcal/mol/Å² converted
to kJ/mol/nm²; the documented `k[1+cos(nφ-φ_s)]` with `cos nφ`, `sin nφ` from a
Chebyshev recurrence). Neither route calls the engine's kernels twice.

---

## 4. Per-term reference values (frozen table)

Case A = `dihedral9_vacuum`, case B = `dihedral1_vacuum`; both use the same
9-atom geometry.  Units: `k` in the GROMACS convention (kJ mol⁻¹ nm⁻² for
bonds, kJ mol⁻¹ rad⁻² for angles, kJ mol⁻¹ for torsions); `k_p` is ParmEd's
stored value in kcal mol⁻¹ Å⁻², kcal mol⁻¹ rad⁻², kcal mol⁻¹.

`terms.tsv` therefore holds **2 × 9** tagged `ATOM` rows (18 rows, one block per
case, same coordinates); `check.jl` asserts 9 unique atoms per case with indices
1…9 before using a geometry.  The matching OpenMM numbers for every row below
live in `openmm_terms.tsv` (same term keys), and agree with the printed engine
values to ≤ 1.4e-13 kJ/mol.

### Bonds (8 terms, identical in A and B)

| i-j | k (GROMACS) | k_p (ParmEd) | r0 [nm] | U_engine at geometry [kJ/mol] |
| --- | --- | --- | --- | --- |
| 1-2 | 284512 | 340 | 0.1090 | 0.00478132 |
| 1-3 | 284512 | 340 | 0.1090 | 0.02326675 |
| 1-4 | 267776 | 320 | 0.1410 | 0.14761403 |
| 1-6 | 224262.4 | 268 | 0.1529 | 1.00097619 |
| 4-5 | 462750.4 | 553 | 0.0945 | 0.03028746 |
| 6-7 | 284512 | 340 | 0.1090 | 0.00119633 |
| 6-8 | 284512 | 340 | 0.1090 | 0.06960433 |
| 6-9 | 284512 | 340 | 0.1090 | 0.03285666 |

Check `k = 836.8 · k_p` holds exactly for every row: 340×836.8 = 284512,
320×836.8 = 267776, 268×836.8 = 224262.4, 553×836.8 = 462750.4.

### Angles (13 terms, identical in A and B)

| i-j-k | k (GROMACS) | k_p (ParmEd) | θ0 [deg] |
| --- | --- | --- | --- |
| 1-4-5 | 460.24 | 55 | 108.5 |
| 1-6-7 | 313.8 | 37.5 | 110.7 |
| 1-6-8 | 313.8 | 37.5 | 110.7 |
| 1-6-9 | 313.8 | 37.5 | 110.7 |
| 2-1-3 | 276.144 | 33 | 107.8 |
| 2-1-4 | 292.88 | 35 | 109.5 |
| 2-1-6 | 313.8 | 37.5 | 110.7 |
| 3-1-4 | 292.88 | 35 | 109.5 |
| 3-1-6 | 313.8 | 37.5 | 110.7 |
| 4-1-6 | 418.4 | 50 | 109.5 |
| 7-6-8 | 276.144 | 33 | 107.8 |
| 7-6-9 | 276.144 | 33 | 107.8 |
| 8-6-9 | 276.144 | 33 | 107.8 |

`k = 8.368 · k_p` holds exactly (55×8.368 = 460.24, 37.5×8.368 = 313.8, …).

### Torsions, case A (14 terms; multiplicity 1,2,3,4 and φ_s = 0°, 60°, 180°)

| i-j-k-l | k_φ [kJ/mol] | k_p [kcal/mol] | n | φ_s [deg] | U_engine at geometry [kJ/mol] |
| --- | --- | --- | --- | --- | --- |
| 2-1-4-5 | 100 | 23.900574 | 1 | 180 | 58.62338069 |
| 2-1-4-5 | 100 | 23.900574 | 2 | 0 | 34.24049250 |
| 2-1-6-7 | 100 | 23.900574 | 3 | 180 | 145.58988502 |
| 2-1-6-8 | 100 | 23.900574 | 3 | 180 | 151.76183828 |
| 2-1-6-9 | 100 | 23.900574 | 3 | 180 | 187.73362272 |
| 3-1-4-5 | 100 | 23.900574 | 1 | 180 | 199.92920516 |
| 3-1-4-5 | 100 | 23.900574 | 2 | 0 | 199.71692088 |
| 3-1-6-7 | 100 | 23.900574 | 3 | 180 | 165.37550864 |
| 3-1-6-8 | 100 | 23.900574 | 3 | 180 | 170.55645471 |
| 3-1-6-9 | 100 | 23.900574 | 3 | 180 | 196.60396727 |
| 4-1-6-7 | 100 | 23.900574 | 4 | 0 | 0.00035393 |
| 4-1-6-8 | 100 | 23.900574 | 4 | 0 | 141.87015736 |
| 4-1-6-9 | 250 | 59.751434 | 3 | 180 | 495.47776664 |
| 6-1-4-5 | 100 | 23.900574 | 4 | 60 | 128.45418200 |

### Torsions, case B (11 terms; all n = 3, φ_s = 60°)

| i-j-k-l | k_φ [kJ/mol] | n | φ_s [deg] | U_engine at geometry [kJ/mol] |
| --- | --- | --- | --- | --- |
| 2-1-4-5 | 1.4 | 3 | 60 | 0.38153593 |
| 2-1-6-7 | 1.4 | 3 | 60 | 2.15997676 |
| 2-1-6-8 | 1.4 | 3 | 60 | 2.07504107 |
| 2-1-6-9 | 1.4 | 3 | 60 | 1.36768358 |
| 3-1-4-5 | 1.4 | 3 | 60 | 0.84103944 |
| 3-1-6-7 | 1.4 | 3 | 60 | 1.85982896 |
| 3-1-6-8 | 1.4 | 3 | 60 | 1.76529202 |
| 3-1-6-9 | 1.4 | 3 | 60 | 1.03705805 |
| 4-1-6-7 | 1.4 | 3 | 60 | 1.76504434 |
| 4-1-6-9 | 2.9 | 3 | 60 | 1.95175815 |
| 6-1-4-5 | 1.4 | 3 | 60 | 0.49820070 |

---

## 5. Measured results (all pass)

ΔU in the OpenMM rows is signed `engine − OpenMM`.

Two independent kinds of comparison are listed. **OpenMM** rows are a real
external engine evaluating standard `HarmonicBondForce` / `HarmonicAngleForce` /
`PeriodicTorsionForce` terms built from the ParmEd-parsed parameters.
**Closed-form** rows re-evaluate the documented GROMACS/ParmEd formulas from
their vector definitions (`cos φ = n̂1·n̂2`, `sin φ = (n̂1×n̂2)·b̂2`, Chebyshev
recurrence for `cos nφ`, `sin nφ`) — a *form* check that exercises the `n` and
`φ0` dependence over a 13-point φ grid without calling the engine's
`dihedral_angle_grad`.

| Check | Measured | Gate |
| --- | --- | --- |
| A k columns obey 836.8/8.368/4.184 | max rel dev = 1.42e-16 | 1e-9 |
| B k columns obey 836.8/8.368/4.184 | max rel dev = 1.26e-16 | 1e-9 |
| A bonds, engine vs **OpenMM** per term | max abs ΔU = 2.76e-15 kJ/mol | 1e-8 |
| A angles, engine vs **OpenMM** per term | max abs ΔU = 1.09e-14 kJ/mol | 1e-8 |
| A torsions, engine vs **OpenMM** per term | max abs ΔU = 1.42e-13 kJ/mol | 1e-8 |
| A per-term forces, engine vs **OpenMM** | max rel ΔF = 1.67e-13 | 1e-9 |
| A all-bonded total vs **OpenMM** | ΔU = +4.55e-13 kJ/mol, rel ΔF = 1.54e-15 | 1e-8 / 1e-9 |
| B torsions, engine vs **OpenMM** per term | max abs ΔU = 5.11e-15 kJ/mol | 1e-8 |
| B per-term forces, engine vs **OpenMM** | max rel ΔF = 1.51e-13 | 1e-9 |
| B all-bonded total vs **OpenMM** | ΔU = -1.42e-14 kJ/mol, rel ΔF = 5.49e-15 | 1e-8 / 1e-9 |
| A bonds, engine vs ParmEd closed form | max abs ΔU = 1.84e-14 kJ/mol | 1e-10 |
| A angles, engine vs ParmEd closed form | max abs ΔU = 1.09e-14 kJ/mol | 1e-10 |
| A torsions, documented form × 13 φ values | max abs ΔU = 2.27e-13 kJ/mol | 1e-10 |
| A φ vs ParmEd `geometry.dihedral` | max abs Δφ = 7.60e-13 rad | 1e-10 |
| A torsion force vs central difference | max rel err = 1.32e-07 | 1e-6 |
| B bonds / angles (ParmEd closed form) | same 1.84e-14 / 1.09e-14 kJ/mol | 1e-10 |
| B torsions, documented form × 13 φ values | max abs ΔU = 5.77e-15 kJ/mol | 1e-10 |
| B φ vs ParmEd | max abs Δφ = 7.60e-13 rad | 1e-10 |
| B torsion force vs central difference | max rel err = 3.27e-10 | 1e-6 |
| φ sign convention (engineered cis/trans) | cis = 0.000000000000°, trans = 180.000000000000° | 1e-10 rad |

Totals at the reference geometry (kJ/mol):

| Case | bond | angle | torsion | total (engine) | total (ParmEd form) | total (OpenMM) | Δ engine − OpenMM (signed) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 1.3105830779 | 20.1174336865 | 2275.9337358063 | 2297.3617525707 | 2297.3617525707 | 2297.3617525707 | +4.55e-13 |
| B | 1.3105830779 | 20.1174336865 | 15.7024590054 | 37.1304757698 | 37.1304757698 | 37.1304757698 | -1.42e-14 |

### The check has teeth

Each deliberately wrong convention is evaluated on the same 13-point φ grid per
term; each deviation exceeds 1 kJ/mol (gates above are 1e-8 kJ/mol or tighter),
so a mapping error could not hide inside the agreement above:

| Wrong convention | max abs ΔU |
| --- | --- |
| `k_φ/2` instead of `k_φ` (bond-style ½ copied onto the torsion) | 247.739 kJ/mol |
| `φ0 = φ_s + π` (phase shifted by a half turn) | 490.956 kJ/mol |
| `n → 1` (multiplicity dropped) | 383.964 kJ/mol |
| InterMol `fc/phi` read as `k[1+cos(n(φ-φ_s))]` (phase inside the harmonic) | 194.500 kJ/mol |

---

## 6. What this establishes, and what it does not

**Established (verified by running `./run.sh`, output above):**

1. **A third-party engine agrees, per term, on energy and force.** OpenMM 8.6.1
   (`Reference` platform) evaluates the same terms built from the ParmEd-parsed
   parameters: 8 bonds and 13 angles in each case, and 25 torsion terms in total
   (14 in case A, 11 in case B). Per-term agreement is
   ≤ 1.42e-13 kJ/mol in energy and ≤ 1.67e-13 relative in force; the
   all-bonded totals agree to +4.55e-13 kJ/mol (case A) and -1.42e-14 kJ/mol
   (case B), signed engine − OpenMM. The mapping is `k_bond = k_GROMACS` (with the ½),
   `k_angle = k_GROMACS` (with the ½), `k_φ = k_GROMACS` (**no ½**),
   `n = multiplicity`, `φ0 = φ_s`.
2. The `1/2` sits exactly where the table says: bond and angle carry it, the
   torsion does not; the affine relations `k = 836.8 k_p` (bond),
   `k = 8.368 k_p` (angle), `k = 4.184 k_p` (torsion) hold for every term to
   1.4e-16 relative — machine exact.
3. Our dihedral angle φ equals ParmEd's independent implementation to
   7.6e-13 rad, *including sign*, and is 0° at cis / 180° at trans on an
   engineered fixture — the IUPAC convention OpenMM's `PeriodicTorsionForce`
   and the GROMACS formula assume. (If the sign were flipped the torsion
   comparison would fail by O(10–100) kJ/mol, not by 1e-13.)
4. The torsion force scattered by `evaluate!` equals `-dU/dq` of the same
   factor to 1.3e-7 relative, and equals OpenMM's analytic force to 1.7e-13
   relative.

**Negative controls (the gates bite).** Run in scratch copies outside the
repository, exit status 1 in both cases:

* Mutating the **parameter table** (`terms.tsv`, ParmEd column of bond 1-4
  raised 1 %): `FAIL ... k columns ... max rel dev=1.00e-02` and
  `FAIL ... bonds ... max|dU|=1.48e-03 kJ/mol`. The frozen `openmm_terms.tsv`
  is a *separate* artifact, so this mutation cannot move engine and OpenMM
  together — the OpenMM numbers stay at the true values.
* Mutating the **OpenMM artifact** (`openmm_terms.tsv`: torsion 4-1-6-9 energy
  +1 %, one bond 1-2 force component +1 kJ/mol/nm):
  `FAIL ... OpenMM per-term energies ... max|dU|=4.95e+00 kJ/mol` and
  `FAIL ... OpenMM per-term forces ... max rel |dF|=2.11e-02`, in both cases.

So neither side can drift silently, and the "agreement" in §5 is not a
tautology: the two artifacts are produced by different stages and the engine
sits between them.

**Not established — do not infer these:**

1. *Not a TraPPE result.* No SRSW/TraPPE bonded parameter is used or claimed;
   see `reference/provenance.txt` for what the SRSW pages contain.
2. *One static geometry, three bonded force types, no dynamics.* OpenMM is used
   as an evaluator of `HarmonicBondForce`, `HarmonicAngleForce` and
   `PeriodicTorsionForce` only, at the single frozen `.gro` geometry of one
   molecule (plus a manufactured φ grid for the form check). No trajectory, no
   integrator, no thermodynamic or structural observable, and no GROMACS/LAMMPS
   binary is involved (none is installed). "Agreement with OpenMM" is therefore
   agreement on these term formulas and parameters, not an engine-equivalence
   claim for any simulation.
3. *Bonded terms only — the nonbonded bookkeeping is out of scope.* These
   topologies scale 1-4 pairs by OPLS `fudgeLJ = fudgeQQ = 0.5`. Our engine has
   no explicit fudge parameter, but the same effect **is** reachable by scaling
   the parameters: `ε → 0.5 ε` in `LennardJonesPair`, and `q → √0.5 q` (or one
   charge by 0.5) in `CoulombPair`. That scaling is *not* exercised here: the
   comparison is bonded-only, so no total ethanol energy including nonbonded
   terms is claimed.
4. *Not a general torsion object.* Our engine has one harmonic per
   `PeriodicDihedral` and no constant/offset term. GROMACS `funct 9` composes
   several terms per atom quadruple; we model those as several factors (e.g.
   two for `HC-CT-OH-HO`: n=1, φ_s=180 and n=2, φ_s=0). `funct 3`
   (Ryckaert–Bellemans) and `funct 5` (Fourier) torsions exist in the same test
   corpus and are **not** expressible as a single `PeriodicDihedral`; they were
   excluded rather than approximated.
5. *Not an InterMol energy validation.* InterMol stores GROMACS proper
   dihedrals as `fc0 = k, fc_n = k, phi = φ_s`. Read with the phase inside the
   harmonic, `k[1+cos(n(φ-φ_s))]`, that representation reproduces the GROMACS
   term only when `n·φ_s ≡ φ_s (mod 2π)`: for `n=4, φ_s=60°` and `n=3, φ_s=60°`
   it deviates by up to 194.5 kJ/mol on the φ grid. The stored `fc0..fc6` are
   kept in `terms.tsv` as provenance only, and are not used as the reference.

## 7. Reproduce

```sh
cd julia/experiments/bonded_verification
./run.sh                 # frozen tables, ~10 s
./run.sh --refresh       # stage 1 (InterMol/ParmEd) + stage 2 (OpenMM)
```

Overrides: `JULIA_BIN`, `JULIA_DEPOT_PATH` (default
`~/.cache/biology-mm/julia-depot`), `INTERMOL_PYTHON`
(default `~/.cache/biology-mm/intermol-verify/bin/python`),
`OPENMM_PYTHON` (default `~/.cache/biology-mm/.venv/bin/python`),
`INTERMOL_TESTS` (test-corpus root for `--refresh`).

Exit status is nonzero if any gate above fails.
