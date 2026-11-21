# Biology-MM v2.0 Open-Science Quickstart (Bio, Chemistry & Physics)

Biology-MM v2.0 is a 100% open-science simulation and method-development platform with **zero closed-source dependencies** and **no PyMOL lock-in**. Trajectories export directly in standard binary **CHARMM/NAMD `.dcd`**, **GROMACS `.xtc`** (XDR + 3DF compressed), and **`.pdb`** formats for immediate analysis in `MDTraj`, `MDAnalysis`, `VMD`, `GROMACS`, and `OpenMM`.

## 1. Run the Published Open Physics Validation Suite

```bash
PYTHONPATH="python:studio/backend" python3 validation/run_open_validation.py
```

Generates `validation/OPEN_VALIDATION_REPORT.md` and `validation/open_validation_results.json` verifying:
- NVE symplectic energy conservation on periodic LJ Argon (`rel. fluctuation ~ 3.1e-6`)
- Bulk TIP3P water NPT density (`~0.984 g/cm³`) and first `g_OO(r)` peak (`~2.75 Å`)
- Alanine dipeptide Well-Tempered Metadynamics + analytical CV VJP gradient check (`~1.2e-9` error)
- Cross-domain Chemistry (Morse bond oscillator) & Physics (Yukawa screened-Coulomb plasma)
- Standard `.dcd` / `.xtc` / `.pdb` binary round-trip interoperability

## 2. Modify Any Physics Parameter & Export Standard `.dcd` / `.xtc`

```python
from biology_mm import (
    OpenScienceEngine,
    PhysicsParameters,
    build_tip3p_water_box,
)

top = build_tip3p_water_box(n_side=4)
params = PhysicsParameters(
    integrator="baoab_npt",     # "nve_verlet" | "baoab_nvt" | "baoab_npt"
    pair_style="lj_lfs",        # "lj_lfs" | "lj_shifted" | "morse" | "buckingham" | "yukawa"
    dt_ps=0.0005,
    temperature_K=300.0,
    pressure_bar=1.0,
    rcut_nm=0.60,
    efield_z=0.0,
)
engine = OpenScienceEngine(top, params)
engine.set_parameters(temperature_K=310.0)  # live parameter override

report = engine.run(
    n_steps=200,
    report_interval=10,
    dcd_path="water.dcd",
    xtc_path="water.xtc",
    pdb_path="water.pdb",
)
```

Load directly in standard open analysis tools:
```python
# MDTraj
import mdtraj as md
traj = md.load("water.xtc", top="water.pdb")

# MDAnalysis
import MDAnalysis as mda
u = mda.Universe("water.pdb", "water.dcd")
```

## 3. Enhanced Sampling (PLUMED-Style CVs, Umbrella, Steered MD, Metadynamics)

```python
from biology_mm import OpenScienceEngine, PLUMEDParser, PhysicsParameters, build_alanine_dipeptide

plumed_cfg = """
phi: TORSION ATOMS=2,4,5,7
d1: DISTANCE ATOMS=1,10
umb: RESTRAINT ARG=phi AT=-1.2 KAPPA=150.0
metad: METAD ARG=phi HEIGHT=1.2 SIGMA=0.3 BIASFACTOR=10.0 PACE=20
"""
suite = PLUMEDParser.from_text(plumed_cfg)
eng = OpenScienceEngine(build_alanine_dipeptide(), PhysicsParameters(dt_ps=0.001), enhanced_sampling=suite)
rep = eng.run(n_steps=500, report_interval=10)
```

## 4. Custom User Potential Callback (Physics / Chemistry Research)

```python
import numpy as np
from biology_mm import OpenScienceEngine, PhysicsParameters, build_morse_diatomic

def custom_anharmonic_trap(positions_nm, box_nm, params):
    k = 100.0
    u = 0.5 * k * float(np.sum(positions_nm ** 2))
    f = -k * positions_nm
    return u, f

eng = OpenScienceEngine(
    build_morse_diatomic(),
    PhysicsParameters(integrator="nve_verlet", pair_style="morse", custom_potential=custom_anharmonic_trap),
)
```

## 5. OpenFF-Sage Parameterization, AMBER `.prmtop`/`.inpcrd` & AlphaFold `pLDDT` Focus Zone

```python
from biology_mm import parameterize_openff_sage, export_amber_prmtop_inpcrd, parse_alphafold_plddt, compute_plddt_focus_zone

pset = parameterize_openff_sage(positions_nm, elements=["C", "O", "H"])
prmtop_text, inpcrd_text = export_amber_prmtop_inpcrd(pset)
```

## 6. Studio REST API Endpoints (v2.0 Open-Science)

- `GET /api/sessions/{sid}/export/dcd` — Download standard CHARMM/NAMD `.dcd` trajectory
- `GET /api/sessions/{sid}/export/xtc` — Download standard GROMACS `.xtc` compressed trajectory
- `GET /api/sessions/{sid}/export/pdb` — Download companion `.pdb` topology
- `POST /api/sessions/{sid}/enhanced-sampling` — Attach PLUMED-style CVs & bias potentials
- `POST /api/sessions/{sid}/physics-params` — Override live temperature, `dt`, `rcut`, `efield_z`, `sigma_scale`, `epsilon_scale`, `charge_scale`
