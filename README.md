# Biology-MM — Open-Science Atomic Simulation & Interactive 3D Workbench

[![Release](https://img.shields.io/badge/release-v2.6.0-blue.svg)](https://github.com/mkrishna793/Biology-MM/releases/tag/v2.6.0)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Formal Proofs](https://img.shields.io/badge/Lean%204%20%2B%20Rocq-0%20sorryAx-purple.svg)](lean/)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-HECBioSim%20%7C%20MPI%20GROMACS%20%7C%20LAMMPS-orange.svg)](validation/COMPETITIVE_BENCHMARK_REPORT.md)

**Biology-MM** is an open-science, formally grounded molecular dynamics and materials physics platform built to **democratize multi-million-atom atomic simulation**. Instead of requiring multi-GPU supercomputers or closed commercial toolchains, Biology-MM lets scientists, students, and researchers simulate **biology, chemistry, and condensed-matter physics** atom-by-atom on a single laptop — and watch every atom, bond, force vector, and electrostatic field evolve live in the browser.

> **Production-Ready & Community-Driven:** Biology-MM v2.6.0 is production-ready across our native Metal/CPU kernels, Python SDK, and WebGL2 Studio. If you encounter any issue in the UI or in a simulation, please [open a GitHub Issue](https://github.com/mkrishna793/Biology-MM/issues) with your structure/parameters so we can fix it promptly.

---

## Quick Start: Clone & Launch the Web Studio in 30 Seconds

### Prerequisites
- **Python 3.10+** (`numpy`, `scipy`, `fastapi`, `uvicorn`, `websockets`, `pydantic`)
- **macOS Apple Silicon (Metal GPU)** or **Linux / x86_64 (CPU / NEON)**
- **Node.js 20+ & `pnpm`** if building the browser 3D UI locally (`studio/**/dist/` is gitignored; the Python SDK and REST/WebSocket backend run without Node).

### 1. Clone & Start the Live Web UI
```bash
git clone https://github.com/mkrishna793/Biology-MM.git
cd Biology-MM

# Create a virtual environment and install backend dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r studio/backend/requirements.txt
pip install -e python

# Build the 3D frontend bundle once (optional if running Python SDK / API only)
(cd studio/frontend/biology-mm-studio && pnpm install && pnpm run build)

# Launch Biology-MM Studio (serves API + WebGL2 3D UI at http://127.0.0.1:8765)
cd studio
./start_studio.sh
```

Then open **`http://127.0.0.1:8765`** in your browser:
1. **Load any structure:** Upload `.cif`, `.mmcif`, `.pdb`, `.gro`, `.xyz`, `.data` (LAMMPS), or `.prmtop`/`.inpcrd` (AMBER), fetch any **AlphaFold DB** entry by UniProt ID directly in the **Prep** tab (e.g., `P69905`, `P00533`), or fetch any PDB/EMDB entry in **Download Structure** (`1CRN`, `8C6J`, `3J3Y`, `5J7V`) and click **Simulate Active Mol\* Structure**.
2. **Scale interactively (`Scale` tab):** Adjust the **3-Zone Multiscale Focus Sphere** (`AA Rad` / `CG Rad`) to focus up to **65,536 All-Atom (`AA`)** atoms on the **Apple Metal GPU** while simulating millions of surrounding atoms in **`Tryte 27+5`** tiles.
3. **Run & Steer (`Run` tab):** Press **Play**, steer **Temperature (K)** or **E-Field**, switch live coloring (`Force Magnitude`, `Kinetic Temp`, `Charge Potential`, `Multiscale Zone`), click any atom to inspect or tug it with **Steered MD (`Pull +5 Å`)**, and export `.mvsx` scenes or standard `.dcd`, `.xtc`, and `.pdb` trajectories directly from the **Run** tab (or via the REST API and Python SDK).

---

## Why Biology-MM? What It Does on a Single Laptop That Struggles on Clusters

Traditional engines (`OpenMM`, `GROMACS`, `LAMMPS`, `NAMD`) store full `float64` coordinates, velocities, forces, and dense neighbor lists (`~150–180+ bytes/atom` plus FFT PME grids). On a 24 GB laptop, a 12M–40M-atom system hits memory exhaustion or swaps to disk.

Biology-MM solves this with four architectural innovations:

1. **27+5 Trium Tryte Tiles (`44–52 B/atom`):** Reorders atoms along a 3D Radix-3 Peano space-filling curve into 27-atom cache-coherent tiles with 5 metadata lanes (`centroids`, `bounding radii & net charge`, `Harmonic.lean residual moments & dipoles`, `tile energies`, and `dirty/zone flags`).
   - Simulates **12,000,000 atoms (`MPI GROMACS benchPEP`)** in **`505.55 MB`** (`2.06%` of 24 GB laptop RAM — **`28.9×` smaller** than GROMACS/OpenMM's `~14.6 GB`).
   - Simulates **2,116,800 atoms (`3J3Y` HIV-1 Capsid)** interactively on a laptop in **`89 MB`** with **`65,529` Metal GPU All-Atom (`AA`)** + **`2.05M` `CG`** coupled tiles and **`1.37e-7` Shadow-$H$ drift**.
2. **Symmetry-Compact Multiscale Engine (`0.05 B/atom`):** For viral capsids and crystalline assemblies (`40,000,000` atoms), stores only the asymmetric unit + $4\times 4$ transforms (`1.82 MB` total RAM — **`3,519×` memory reduction** vs. `7,340 MB` float64).
3. **Apple Metal GPU-Resident Batched Verlet (`3,484 ns/day` @ 2 fs | `6,969 ns/day` @ 4 fs HMR on a 216-atom unbonded LJ fluid micro-benchmark):** Executes 1,000+ symplectic Velocity-Verlet steps inside a single Metal command buffer with zero Python/CPU overhead (`0.0496 ms/step`).
4. **Formally Verified Mechanics (`0 sorryAx`):** Exact centroid/residual-moment factorization (`Harmonic.lean`) and Velocity-Verlet Shadow Hamiltonian invariants (`Shadow.lean`) are machine-checked in **Lean 4** and **Rocq**, with **HMC Metropolis Shadow-$H$ guards** preventing energy blowups during interactive steering.
5. **AlphaFold DB (`pLDDT` + `PAE`) & ATLAS Flexibility Pipeline (`v2.6.0`):** Fetches predicted structures and `PAE JSON` directly by UniProt ID, partitions rigid domains vs. flexible inter-domain hinges (`cluster_pae_domains`), automatically focuses the Metal All-Atom zone on low-`pLDDT` loops/hinges, and applies smooth confidence-weighted restraints with active-zone C¹-smooth DSF electrostatics.

---

## Head-to-Head Production Benchmarks

All benchmarks below were executed directly on an **Apple M5 Laptop (`24.0 GB Unified RAM`, `macOS arm64`, Metal GPU)** using authentic multi-species topologies (protein `[C,N,O,S,H]`, POPC lipid bilayer `[P,N,C,O,H]`, ribosomal rRNA `[P,O,N,C,H]`, TIP3P water `[O,H1,H2]`, counterions `[Na+,Cl-,K+,Mg2+]`, MOFs, gold nanoparticles, and crystals).

Reproduce everything on your machine:
```bash
PYTHONPATH="python:studio/backend" python3 validation/run_competitive_benchmarks.py
```

### 1. HECBioSim & Max Planck Institute (MPI) GROMACS Benchmarks
| Benchmark Suite / Workload | Atoms ($N$) | Biology-MM RAM | GROMACS / OpenMM Ref RAM | RAM Reduction | Bytes / Atom | Laptop RSS | % of 24 GB RAM | Step Time (`ms/step`) | Speed (`MATOM-steps/s`) | Throughput (`ns/day` @ 2 fs) | Shadow-$H$ Drift | Status |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **HECBioSim Crambin in Water (`1CRN`)** | `20,004` | **`0.99 MB`** | `225.0 MB` | **`227.7×`** | `51.8 B` | `622.3 MB` | `0.004%` | `102.7 ms` | `0.195 M/s` | `1.68 ns/day` | `5.06e-4` | **bounded** |
| **HECBioSim / Amber-CHARMM DHFR** | `23,558` | **`1.17 MB`** | `265.0 MB` | **`227.4×`** | `51.9 B` | `632.1 MB` | `0.005%` | `100.4 ms` | `0.235 M/s` | `1.72 ns/day` | `2.35e-4` | **bounded** |
| **MPI GROMACS `benchMEM`** (Protein + POPC Bilayer + Water) | `81,743` | **`3.85 MB`** | `480.0 MB` | **`124.7×`** | `49.4 B` | `634.7 MB` | `0.016%` | `101.7 ms` | `0.80 M/s` | `1.70 ns/day` | `9.05e-4` | **bounded** |
| **MPI GROMACS `benchRIB`** (70S Ribosome + rRNA + $\text{Mg}^{2+}$) | `2,000,004` | **`84.52 MB`** | `2,850.0 MB` | **`33.7×`** | `44.3 B` | `1,042.5 MB` | `0.344%` | `248.5 ms` | **`8.05 M/s`** | `0.70 ns/day` | `9.44e-5` | **bounded** |
| **MPI GROMACS `benchPEP`** (12M-Atom Supercomputer Box) | `12,000,000` | **`505.55 MB`** | `14,600.0 MB` | **`28.9×`** | `44.2 B` | `2,764.4 MB` | **`2.057%`** | `1,196.8 ms` | **`10.03 M/s`** | `0.14 ns/day` | `1.12e-6` | **bounded** |

> **Benchmark Scope & Limitations:** In `benchRIB` (2M atoms) and `benchPEP` (12M atoms), Biology-MM runs its **Tryte 27+5 multiscale** representation (an active All-Atom focus zone coupled to tile-level far-field multipoles/elastic terms, without full-system PME electrostatics or full AMBER/CHARMM dihedral evaluation across all 12M atoms). The `GROMACS / OpenMM Ref RAM` column reflects full-system PME all-atom MD; RAM reduction and step timings above therefore measure multiscale interactive laptop execution rather than an identical full-system PME workload.

### 2. Official LAMMPS Benchmark Suite (`docs.lammps.org/Speed_bench.html`) & Metal GPU Peak
| Workload | Atoms ($N$) | Bonds | Charged | Biology-MM RAM | LAMMPS Ref RAM | Step Time (`ms/step`) | `Timesteps/s` | `MATOM-steps/s` | Throughput (`ns/day` @ 2 fs) | Shadow-$H$ Drift | Status |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **LAMMPS `in.lj`** (LJ Argon Fluid, $\rho^*=0.8442$) | `32,000` | `0` | `False` | **`1.52 MB`** | `118.5 MB` | `17.20 ms` | `58.2/s` | `1.86 M/s` | `10.05 ns/day` | `1.19e-6` | **bounded** |
| **LAMMPS `in.chain`** (320 Polymer Chains) | `32,000` | `31,680` | `False` | **`1.76 MB`** | `118.5 MB` | `17.30 ms` | `57.8/s` | `1.85 M/s` | `9.99 ns/day` | `4.48e-6` | **bounded** |
| **LAMMPS `in.eam`** (FCC Copper Crystal) | `32,000` | `0` | `False` | **`1.53 MB`** | `118.5 MB` | `79.36 ms` | `12.6/s` | `0.40 M/s` | `2.18 ns/day` | `9.18e-6` | **bounded** |
| **LAMMPS `in.rhodo`** (Rhodopsin GPCR in Lipid) | `32,000` | `12,000` | `True` | **`1.63 MB`** | `118.5 MB` | `52.75 ms` | `19.0/s` | `0.61 M/s` | `3.28 ns/day` | `1.46e-4` | **bounded** |
| **LAMMPS `in.ters`** (Tersoff CNT Tensile) | `320` | `480` | `False` | **`0.09 MB`** | `42.0 MB` | `3.17 ms` | `315.5/s` | `0.10 M/s` | `54.5 ns/day` | `4.23e-4` | **bounded** |
| **Apple Metal GPU Batched Verlet** (216-atom LJ micro-bench) | `216` | `0` | `False` | **`0.08 MB`** | `48.0 MB` | **`0.0496 ms`** | **`20,163.8/s`** | **`4.36 M/s`** | **`3,484.3 ns/day`** *(6,969 @ 4fs HMR)* | `0.1019` | **bounded** |

### 3. Real Chemistry & Condensed-Matter Crystals (`Float64-Full Metal` & `Tryte 27+5`)
| Structure | Engine Mode | Atoms | Intra-Cell Bonds | Periodic Box (Å) | In-Box | Max $\|F\|$ | Step Time (`ms/step`) | `ns/day` (@ 2 fs) | Status |
| :--- | :--- | ---: | ---: | :--- | :--- | ---: | ---: | ---: | :--- |
| **UiO-66 Zr-MOF** (`COD 4512074`) | `float64-full` / `tryte-all-atom` | `440` | `441` / `399` | `20.75 × 20.75 × 20.75` | `True` | `1329.5` / `474.6` | `1.21 ms` / `0.96 ms` | **`142.6` / `179.4 ns/day`** | **bounded** |
| **$\text{Au}_{102}(p\text{-MBA})_{44}$ Nanoparticle** | `float64-full` / `tryte-all-atom` | `762` | `948` / `948` | `54.01 × 53.50 × 53.62` | `True` | `1141.0` / `408.4` | `1.89 ms` / `1.30 ms` | **`91.2` / `133.1 ns/day`** | **bounded** |
| **$\text{Bi}_2\text{Se}_3$ Topological Insulator** (`mp-541837`) | `float64-full` / `tryte-all-atom` | `240` | `411` / `410` | `16.55 × 14.33 × 28.64` | `True` | `1041.1` / `142.3` | `1.00 ms` / `0.44 ms` | **`172.9` / `393.1 ns/day`** | **bounded** |

### 4. Physics Accuracy vs. OpenMM / GROMACS / Experimental Baselines
- **NVE Symplectic Energy Conservation ($C^1$ LFS Argon Fluid):**
  - $\Delta t = 0.5\text{ fs}$: $\sigma(E)/|\langle E\rangle| = \mathbf{5.81 \times 10^{-7}}$
  - $\Delta t = 1.0\text{ fs}$: $\sigma(E)/|\langle E\rangle| = \mathbf{1.90 \times 10^{-6}}$
  - $\Delta t = 2.0\text{ fs}$: $\sigma(E)/|\langle E\rangle| = \mathbf{5.43 \times 10^{-6}}$ *(OpenMM / GROMACS double-precision NVE ceiling: `< 1.0e-3`)*
- **Bulk TIP3P Water NPT ($300\text{ K}, 1\text{ bar}$):**
  - Equilibrium Density: **`0.9842 g/cm³`** *(OpenMM 8.1 TIP3P: `0.9860 g/cm³`, Experimental Water: `0.9970 g/cm³`)*
  - $g_{OO}(r)$ First Solvation Peak: **`2.75 Å`** *(GROMACS 2024 Reference: `2.780 Å`)*
- **Enhanced Sampling Analytical VJP Gradient Error:** **`1.16 × 10⁻⁹`** vs. central finite differences.

See [`validation/COMPETITIVE_BENCHMARK_REPORT.md`](validation/COMPETITIVE_BENCHMARK_REPORT.md) and [`validation/OPEN_VALIDATION_REPORT.md`](validation/OPEN_VALIDATION_REPORT.md) for full details.

---

## Custom Simulations & Interactive Code Lab Guide

### A. Inside the Browser **Code Lab (`Lab` Tab)**
Every live session in Biology-MM Studio exposes `session`, `active_structure`, and `biology_mm` directly in the sandboxed Python **Code Lab**. Any changes you make immediately rebuild the Metal GPU plan and update the 3D Mol\* viewport:

```python
# 1. Steer or modify the currently running simulation in-place
session.steer(temperature=420.0, efield_z=0.25)
session.rcut = 0.92
session.sigma = session.sigma * 1.02
session.step(20)
```

```python
# 2. Or launch a custom OpenScienceEngine workflow directly into the 3D UI
from biology_mm import OpenScienceEngine, PhysicsParameters, build_alanine_dipeptide

params = PhysicsParameters(
    integrator="baoab_nvt",
    pair_style="lj_lfs",   # also supports: morse, buckingham, yukawa
    dt_ps=0.001,
    temperature_K=310.0,
    rcut_nm=0.95,
)
eng = OpenScienceEngine(build_alanine_dipeptide(), params)
eng.run(n_steps=50)
```

### B. Standalone Python SDK (`OpenScienceEngine` + Custom Potentials + PLUMED + `.dcd`/`.xtc`)
```python
from biology_mm import (
    OpenScienceEngine, PhysicsParameters, PLUMEDParser,
    build_tip3p_water_box, parameterize_openff_sage
)

top = build_tip3p_water_box(n_side=4)
plumed = PLUMEDParser.from_text("""
d1: DISTANCE ATOMS=1,4
METAD ARG=d1 PACE=20 HEIGHT=1.2 SIGMA=0.05 BIASFACTOR=8.0
""")

def custom_wall(pos_nm, box_nm, params):
    # Optional custom Python/NumPy force field hook -> (energy_kj_mol, forces_kj_mol_nm)
    z = pos_nm[:, 2]
    f = pos_nm * 0.0
    f[:, 2] = -10.0 * (z - 1.0)
    return float(5.0 * ((z - 1.0) ** 2).sum()), f

params = PhysicsParameters(integrator="baoab_npt", temperature_K=300.0, custom_potential=custom_wall)
eng = OpenScienceEngine(top, params, enhanced_sampling=plumed)
report = eng.run(n_steps=500, dcd_path="traj.dcd", xtc_path="traj.xtc", pdb_path="top.pdb")
```

For full Python SDK tutorials, see [`OPEN_SCIENCE_QUICKSTART.md`](OPEN_SCIENCE_QUICKSTART.md).

---

## Repository Structure

| Path | Contents |
| :--- | :--- |
| `studio/` | **Biology-MM Studio**: FastAPI + WebSocket `TriumStream` backend (`studio/backend/`) and Mol\*-powered WebGL2 frontend (`studio/frontend/`), launched via `studio/start_studio.sh` |
| `python/biology_mm/` | **Python Open-Science SDK**: `OpenScienceEngine`, `openff_amber_bridge.py`, `enhanced_sampling.py` (PLUMED CVs/Metadynamics/Umbrella/SMD), `trajectory_io.py` (`.dcd`/`.xtc`/`.pdb`), and native Metal/CPU `ctypes` bindings |
| `julia/` | `FoundationCore` Julia core (`TriumXBT`, `TriumGPU`, `TernaryMSM`, `ReactiveNano`, `GrandCanonical`, `NEMD`, `EnhancedSampling`, 57 test suites) |
| `lean/` & `rocq/` | **Lean 4** (`Harmonic.lean`, `Shadow.lean`, `TernarySymplectic.lean`, `TernaryWeighing.lean` — 0 `sorryAx`) & **Rocq** (`Core.v`) formal proofs |
| `validation/` | `run_competitive_benchmarks.py` (HECBioSim, MPI GROMACS, LAMMPS, crystals), `run_open_validation.py`, and capability/status reports |

---

## Acknowledgments

- **[Mol\* (MolStar) & MolViewSpec](https://molstar.org/):** Biology-MM Studio's 3D molecular rendering, crystallographic unit cell visualization, representation presets, and interactive selection UI are built on top of the open-source **Mol\*** and **MolViewSpec** projects. We deeply appreciate the Mol\* contributors for creating a world-class WebGL2/WebGPU molecular visualization engine.
- **Open-Science Benchmarks & Standards:** We thank the **HECBioSim** consortium, the **Max Planck Institute for Multidisciplinary Sciences (GROMACS `benchMEM`/`benchRIB`/`benchPEP`)**, **LAMMPS (`Speed_bench`)**, **OpenMM**, **OpenFF**, **COD**, and the **Materials Project** for open community benchmarks and structures.

---

## Running Verification Tests
```bash
# Python Open-Science & Studio End-to-End Tests
PYTHONPATH="python:studio/backend" pytest studio/backend/tests/test_alphafold_v26.py studio/backend/tests/test_open_science_v2.py studio/backend/tests/test_tryte_tiles_and_full_atom.py studio/backend/tests/test_backend_api.py -q

# AlphaFold PAE/pLDDT & ATLAS-Style Flexibility Validation
PYTHONPATH="python:studio/backend" python3 validation/run_atlas_validation.py

# Full Multi-Domain Competitive & Accuracy Benchmarks
PYTHONPATH="python:studio/backend" python3 validation/run_competitive_benchmarks.py
```

---

## License & Open-Science Terms

Biology-MM is licensed under the **[GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE)**.

- **For individuals, students, researchers, and scientists:** You are free to download, run, study, modify, and extend Biology-MM end-to-end for your research and personal work at no cost.
- **Strong Copyleft Open-Source Protection:** Any distributed modifications or network-hosted services built on Biology-MM must also be released as open source under **AGPL-3.0**, ensuring the platform and community improvements remain open rather than being enclosed into proprietary closed-source commercial software.
- **Commercial / Proprietary Use:** Organizations that wish to integrate Biology-MM into closed-source commercial products or hosted services without AGPL-3.0 source-disclosure obligations should contact the author for a separate commercial license.
