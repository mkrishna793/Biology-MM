# Biology MM v1.0 — Formal Foundation & Trium-XBT Simulation Core

`Biology MM` (`FoundationCore` v1.0.0) + `Trium-XBT` / `TriumGPU` is a modular molecular-mechanics and nanotechnology simulation engine that is explicitly specified, formally proved in Lean 4 & Rocq where exact discrete algebraic structure exists, and cross-checked against analytical finite differences, NIST reference data, and external engines. It is **not** a drop-in replacement for OpenMM and states its verified scope and refusal gates explicitly; see [`validation/STATUS.md`](validation/STATUS.md) and [`validation/CAPABILITY_MATRIX.md`](validation/CAPABILITY_MATRIX.md) for exact pass/fail evidence.

---

## Repository Structure

| Path | Contents |
| --- | --- |
| `lean/` | Lean 4 proofs (`leanprover/lean4:v4.35.0`): 13 classical harmonic & shadow-H theorems + `TernarySymplectic.lean` & `TernaryWeighing.lean` (23 weighing/Farkas theorems), standard axioms only, 0 `sorryAx` |
| `rocq/Core.v` | Rocq proof: pairwise force cancellation & momentum conservation over `Q` |
| `experiment/` | Python algebraic demonstrator: exact rational + float agreement |
| `julia/` | `FoundationCore` v1.0.0 Julia core package: typed state, bonded/nonbonded factors, reactive nanotech potentials, custom physics, GCMC, NEMD, enhanced sampling, Trium-XBT C++/NEON & TriumGPU Metal 4 engines, Zero-FFT `TernaryMSM`, `SymplecticTernary`, `Alchemical`, `WaterBox`, and transactional `Workflow` |
| `julia/bin/md.jl` | Headless JSON CLI runner (`--backend <auto\|cpu\|gpu\|metal\|vulkan\|sycl>`, `--info`) |
| `julia/test/` | **57 Julia test suites** + single-command runner (`julia --project=. test/run_all.jl`) |
| `python/biology_mm/` | Modular Python SDK v1.0.0 (`native.py`, `parameters.py`, `builders.py`, `simulation.py`, `cli.py`) with `pytest` suite & 3 runnable examples |
| `examples/` | 5 runnable Julia end-to-end examples (`01`–`05`, including `05_nanotech_custom_physics.jl`) with numeric gates |
| `validation/` | NIST LJ data checks, OpenMM/Molly.jl/MDAnalysis adapters, `CAPABILITY_MATRIX.md`, and `STATUS.md` |
| `FRAMEWORK.md` | Design document: label scheme (PROVED / TARGET / HYPOTHESIS) throughout |
| `INSTALL.md` | Reproducible environment setup |
| `verify.sh` | Single-command verification script (`--core` and full modes) |

---

## Python SDK Quickstart (`python/`)

The `biology_mm` v1.0.0 Python package provides direct `ctypes` bindings to the native Trium-XBT CPU/NEON and Apple M5 Metal 4 GPU bridges (`GPUVerletPlan`, `run_batched_gpu_steps`, `gpu_ternary_matvec`, `gpu_msm_stencil27`, `csa_accumulate`), unconstrained parameter transforms (`ParameterSpec`, `ParameterEncoder`), nanostructure builders (`NanostructureBuilder`, `SystemBuilder`), and Julia CLI orchestration (`Simulation`, `SimulationConfig`):

```sh
cd python
PYTHONPATH=. python -m biology_mm info --json
PYTHONPATH=. pytest -q tests/test_biology_mm_sdk.py
```

```python
from biology_mm import (
    CustomPotential,
    NanostructureBuilder,
    ParameterEncoder,
    ParameterSpec,
    Simulation,
    SimulationConfig,
)

cnt = NanostructureBuilder.carbon_nanotube(n=5, m=5, n_cells=2, bond_length_nm=0.142)
enc = ParameterEncoder([
    ParameterSpec("A", default=45.0, lower=0.0, encoding="log"),
    ParameterSpec("mix", default=0.4, lower=0.0, upper=1.0, encoding="logit"),
])
z = enc.encode([45.0, 0.4])

cfg = SimulationConfig(ensemble="nve", integrator="velocity_verlet", dt=0.0005, produce_steps=25)
result = Simulation(cnt, potentials=[CustomPotential.tersoff(element="C", units="md")], config=cfg).run()
```

Runnable SDK examples in `python/examples/`:
- `01_cnt_nanomechanics.py` — (5,5) armchair CNT construction and NVE workflow
- `02_custom_physics_law.py` — `ParameterEncoder`, callable-to-cubic-spline potential tabulation, and `md.jl` execution
- `03_gpu_native_accelerated.py` — Apple Metal 4 `GPUVerletPlan`, GPU-resident batched MD, dual-rail ternary MatVec, and 27-point MSM stencil

---

## Formal Theorems (all 0 `sorryAx`, 0 `native_decide`)

### Harmonic Factorization and Forces (Lean)

| Theorem | Statement | File |
| --- | --- | --- |
| `pairSum_decomp` | `Σ‖q_i-q_j‖² = nm‖a-b‖² + m·resid(A) + n·resid(B)` for centroids | `Harmonic.lean` |
| `forceOnA_eq` | `F_i = -k·m·(q_i - b)` per atom, O(1) given the centroid | `Forces.lean` |
| `forceOnA_injective` | distinct atoms receive distinct forces when `k·m ≠ 0` | `Forces.lean` |
| `total_force_zero` | total internal force vanishes (Newton III) | `Forces.lean` |
| `harmEnergy_pert` | `U(q+tv) = U(q) - t⟨F,v⟩ + t²U(v)` — `F = -∇U` algebraically | `Consistency.lean` |
| `pairSum_decomp_general` | the same sum for arbitrary reference points, with three named correction terms | `General.lean` |

### Velocity-Verlet Shadow Hamiltonian (Lean)

| Theorem | Statement | File |
| --- | --- | --- |
| `shadow_invariant` | `H̃(q',p') = H̃(q,p)` exactly, for every step size | `Shadow.lean` |
| `verlet_energy_defect` | `E(q',p') = H̃(q',p') + (h²ω⁴/8)·q'²` | `Shadow.lean` |
| `energy_is_shadow_plus_defect` | energy = initial shadow + defect at new position | `Shadow.lean` |
| `shadow_coeff_pos` | shadow coefficient > 0 for `0 < ω²` and `h²ω² < 4` (stability limit) | `Shadow.lean` |
| `shadow_energy_band` | on the same orbit: `E₁ - E₂ = (h²ω⁴/8)(q₁² - q₂²)` | `Shadow.lean` |
| `energy_band_abs` | one-sided bound `E₁ - E₂ ≤ (h²ω⁴/8)(q₁² + q₂²)` on the same orbit | `Shadow.lean` |
| `energy_after_step_eq` | single-step energy = initial shadow value + defect at the new position | `Shadow.lean` |

### Balanced Ternary Symplectic & Weighing/Farkas Proofs (Lean)

| Module | Proved Theorems (Audited in `verify.sh`) |
| --- | --- |
| `TernarySymplectic.lean` | `trit_neg_involutive`, `dual_rail_neg_involutive`, `btha_conservation`, `btfa_conservation`, `carry_free_step_conservation`, `verletKDK_exact_reversible`, `verletKDK_n_steps_reversible`, `hamiltonian_energy_conservation` |
| `TernaryWeighing.lean` | 23 theorems including `cw4_valid`, `cw7_valid`, `cw13_valid`, `cw4_orthogonal`, `cw7_orthogonal`, `cw13_orthogonal`, `weighing_left_inverse`, `weighing_parseval_energy`, `radix3_compression_sum_eq`, `farkas_refutation_sound`, `amo_steric_exclusion_sound` |

### Rocq: Pairwise Momentum Conservation

`rocq/Core.v` proves that two forces summing to zero (Newton III) preserve total momentum over `Q`. Compiled independently with `coqc`; see `verify.sh`.

---

## Trium-XBT CPU/NEON & Apple M5 Metal 4 GPU (`TriumXBT`, `TriumGPU`, `Engine`)

- **`TriumXBT.jl`**: Compiled CSR Verlet plans (`TriumCompiledVerletPlan`, `eval_lj_dsf_direct_into!`) and 32-atom Morton tile traversal (`TriumTiledNonbondedFactor`) merging LJ (`:lfs`) and DSF Coulomb across multi-threaded C++ and ARM64 NEON kernels, with `DifferenceSetTileSchedule` disjoint orbit scheduling, 0-cycle rail-swap negation, carry-free `TritWord64` accumulation, and exact orthorhombic/triclinic box derivatives.
- **`TriumGPU.jl`**: Adaptive multi-backend GPU dispatch (`:auto`, `:metal`, `:vulkan`, `:sycl`, `:cpu`) with native Apple M5 Metal 4 library (`libtrium_gpu_bridge.dylib`), GPU-resident batched Velocity Verlet (`run_batched_gpu_steps!`), `gpu_ternary_matvec!`, and `gpu_msm_stencil27!`.
- **`Engine.jl`**: Unified `SimulationEngine` (`create_engine`, `evaluate_step!`, `run_steps!`, `verify_engine_conservation`) routing `verlet`, `baoab`, `respa`, and `symplectic_ternary` across CPU/NEON and GPU backends, including Phase 9 reactive/custom/NEMD/enhanced-sampling factors.

---

## Nanotechnology, Custom Physics & Enhanced Sampling

| Module | File | Capability |
| --- | --- | --- |
| `CustomPhysics` | `julia/src/CustomPhysics.jl` | `ParameterSpec`, `ParameterEncoder` (`:identity`, `:log`, `:logit`, `:angle`, `:ternary_quantized`), `CubicSplineTable`, 7 radial kernels (`:morse`, `:buckingham`, `:yukawa`, `:mie`, `:fene`, `:wca`, `:tabulated_spline`), `CustomPairFactor`, `CustomExternalFactor`, `CustomManyBodyFactor`, `POTENTIAL_REGISTRY` |
| `ReactiveNano` | `julia/src/ReactiveNano.jl` | `TersoffFactor` (`tersoff_carbon`, `tersoff_silicon`), `StillingerWeberFactor` (`stillinger_weber_silicon`), `MorseBondFactor`, `SymmetryFunctionMLIP` / `MLIPFactor`, `build_graphene_sheet`, `build_cnt`, `build_nanopore_membrane`, `build_diamond_lattice` (with exact forces & 9-entry triclinic box derivatives) |
| `GrandCanonical` | `julia/src/GrandCanonical.jl` | Multi-species `muVT` `GrandCanonicalSampler`, `widom_excess_chemical_potential`, `farkas_steric_screen` (`GSI-R-0013` AMO hard-core screen), `TritWord64` `TernaryLatticeGas`, and `ConstantPHSampler` |
| `NEMD` | `julia/src/NEMD.jl` | `ExternalElectricFieldFactor`, `NanofluidicBodyForceFactor`, `WallRestraintFactor` (`:harmonic_flat_bottom`, `:lj93`, `:cylindrical_pore`), and Müller–Plathe RNEMD (`muller_plathe_shear_step!`, `muller_plathe_thermal_step!`, conductivity/viscosity/flux estimators) |
| `EnhancedSampling` | `julia/src/EnhancedSampling.jl` | `DistanceCV`, `CoordinationCV`, `DihedralCV`, `ComDistanceCV`, `CustomCV`, `UmbrellaFactor`, `MetadynamicsBias` (well-tempered), log-space WHAM (`unbias_umbrella_wham`), and `remd_exchange_attempt!` |

---

## CLI (`julia/bin/md.jl`)

```sh
julia --project=julia julia/bin/md.jl --info
julia --project=julia julia/bin/md.jl --backend auto SPEC.json
```

Supports `units` (`"reduced"`, `"nm,kJ/mol,amu,ps"` / `"md"`, `"A,eV,amu,fs"` / `"ev_angstrom"`), explicit bonded/LJ/imported AMBER factors, and Phase 9 factor specs (`morse_bond`, `tersoff`, `stillinger_weber`, `custom_pair`, `tabulated_spline`, `external_efield`, `wall_restraint`, `umbrella`).

---

## Running Everything

```sh
sh verify.sh --core   # proofs + experiment + 57 Julia suites + -t 4 threading + chemistry & SDK pytest
sh verify.sh          # full run including NIST LJ and optional third-party cross-checks
```

Single-command Julia suite runner only:

```sh
cd julia
export JULIA_DEPOT_PATH=~/.cache/biology-mm/julia-depot
~/.cache/biology-mm/julia-1.13.3/bin/julia --project=. test/run_all.jl
```

---

## Core Module Inventory (`julia/src/`)

| Module | File | Capability |
| --- | --- | --- |
| `Types` | `julia/src/Types.jl` | Typed state (`AtomSet`, `System`, `OrthorhombicBox`, `TriclinicBox`, `NoBox`, `ChemicalTopology`) |
| `Factors` | `julia/src/Factors.jl` | Composable local energy maps (`HarmonicBond`, `HarmonicAngle`, `PeriodicDihedral`, `LennardJonesPair`, `CoulombPair`, `BipartiteHarmonicGroup`) and force adjoints |
| `Neighbors` | `julia/src/Neighbors.jl` | Orthorhombic/triclinic cell-list search and Verlet rebuild checks |
| `Dynamics` | `julia/src/Dynamics.jl` | Velocity Verlet, shadow-H diagnostics, BAOAB Langevin, numerical error controller |
| `Respa` | `julia/src/Respa.jl` | 2-level reversible r-RESPA integrator with coupled constraints |
| `Sampling` | `julia/src/Sampling.jl` | HMC + parallel tempering, analytic double-well benchmark, ESS estimation |
| `Polarizable` | `julia/src/Polarizable.jl` | Induced-dipole solver `Aμ=E`, Thole damping, exact Hellmann–Feynman nuclear forces |
| `PolarizationOperator` | `julia/src/PolarizationOperator.jl` | Matrix-free induced dipoles, dual-rail ternary bitplane & `:qc_hybrid` PCG, Bloch Luttinger–Tisza SPD certificate, `cw_orthogonal_noise`, `apply_cw_block_filter!` |
| `Prepare` | `julia/src/Prepare.jl` | Structure diagnosis, staged deterministic repair, provenance log |
| `ResourceProbe` | `julia/src/ResourceProbe.jl` | Resource estimates and measured memory/load preflight |
| `ParameterGradients` | `julia/src/ParameterGradients.jl` | Analytic fixed-state gradients for supported bonded, LJ, and Coulomb parameters |
| `EquilibriumSensitivities` | `julia/src/EquilibriumSensitivities.jl` | Fixed-measure equilibrium gradient averages, autocorrelation-aware ESS/error |
| `StreamedLJ` | `julia/src/StreamedLJ.jl` | Periodic per-atom Lorentz–Berthelot LJ with `PairCache`/`TilePairCache`, 64-bit fixed-point determinism, and `StreamedLJFactor` |
| `Electrostatics` | `julia/src/Electrostatics.jl` | Direct reference orthorhombic Ewald and `DSFCoulombFactor` |
| `PME` | `julia/src/PME.jl` | Scalar-charge PME with cardinal B-spline mesh, cached FFTW plans, forces/charge gradients/box derivatives, `PMEFactor` |
| `TernaryMSM` | `julia/src/TernaryMSM.jl` | Zero-FFT radix-3 27-point electrostatics (`:hardy_c2`, `:sos_c4`), $O(M_L^4)$ Hartley coarsest solver, `CirculantWeighingMatrix` |
| `SymplecticTernary` | `julia/src/SymplecticTernary.jl` | Discrete time-reversible balanced-ternary KDK Verlet and $O(1)$-memory adjoint replay |
| `Alchemical` | `julia/src/Alchemical.jl` | Soft-core `AlchemicalNonbondedFactor` with `TI`, `FEP`, `BAR`, and `MBAR` estimators |
| `ResolvedSystem` | `julia/src/ResolvedSystem.jl` | Schema `v1.0.0` validated specification with 14 typed construction/revalidation rules |
| `SystemContract` | `julia/src/SystemContract.jl` | Capability-aware `energy`/`forces`/`component_energies`, analytic box derivative `G = ∂U/∂h`, virial `Ξ = -hGᵀ`, affine/mechanical pressure |
| `Chemistry` | `julia/src/Chemistry*.jl` | AMBER prmtop/inpcrd + CHARMM PSF/PRM import, CMAP, `PairC64Table`, `NBFIXPairTable`, and `ChemistryBuild` |
| `Constraints` / `Settle` | `julia/src/Constraints.jl`, `Settle.jl` | Disjoint-pair RATTLE, coupled rigid-molecule projection, and analytical periodic SETTLE |
| `MonteCarloBarostat` | `julia/src/MonteCarloBarostat.jl` | Isotropic, semi-isotropic, and anisotropic MC barostat with `infer_molecules` |
| `WaterBox` | `julia/src/WaterBox.jl` | Composed rigid TIP3P water box (`:principal_axis` / `:settle`), constrained BAOAB, MSD analysis |
| `Workflow` | `julia/src/Workflow.jl` | Transactional 6-stage workflow runner, SHA-256 `RunManifest`, atomic checkpoints, extended-XYZ/JSONL I/O |
| `TriumXBT` / `TriumGPU` / `Engine` | `julia/src/TriumXBT.jl`, `TriumGPU.jl`, `Engine.jl` | Native C++/NEON CSR Verlet & Morton tiles, Metal 4 / Vulkan / SYCL GPU dispatch, unified `SimulationEngine` |
| Phase 9 Modules | `CustomPhysics.jl`, `ReactiveNano.jl`, `GrandCanonical.jl`, `NEMD.jl`, `EnhancedSampling.jl` | Custom kernels & parameter encoding, Tersoff/SW/Morse/MLIP & nanostructures, μVT GCMC & lattice gas, NEMD/RNEMD, CVs/Umbrella/Metadynamics/WHAM/REMD |

---

## Scope and Explicit Refusals

Stated plainly, so they are not discovered by accident:

- **Not an OpenMM replacement**: OpenMM cross-checks cover matched LJ, neutral Ewald/PME, explicit TIP3P, and static energy/force parity on five AMBER classes (`water_ions`, `solvated_protein`, `dna`, `membrane`, `protein_ligand`, including 1XEP). Long multi-ns biomolecular convergence remains outside fast CI gates.
- **Fast multipole method (FMM)**: not implemented.
- **Resolved-system stress refusals**: `NoBox` refuses `box_derivative`/`virial`/`pressure` (`UnsupportedCapability` / `MissingBoxDerivative`). Direct `EwaldFactor` refuses stress and `TriclinicBox`. Discontinuous `:trunc` and C⁰ `:shift` LJ modes are refused by `SystemContract` stress and `Dynamics.force_model_supported`.
- **Parameter gradients & sensitivities**: analytic gradients cover the five local factor types and dedicated `StreamedLJ` / `CustomPhysics` evaluators; nonlocal `BipartiteHarmonicGroup` parameter gradients are refused.
- **Scope of Lean/Rocq proofs**: Lean 4 theorems prove exact rational (`Harmonic`–`Shadow`) and discrete integer/balanced-ternary/Farkas identities (`TernarySymplectic`, `TernaryWeighing`), and Rocq proves Newton III momentum conservation over `Q`. Floating-point trajectories are verified numerically via finite differences and conservation gates rather than float-level formal proofs.
- **Platform GPU execution**: Apple Metal 4 (`libtrium_gpu_bridge.dylib`) is natively executed and verified on macOS arm64; Vulkan Compute and SYCL 2020 sources compile on their target toolchains and automatically fall back to `TriumXBT` CPU/NEON when unavailable.

For detailed open problems and release gates, see [`FRAMEWORK.md`](FRAMEWORK.md) and [`validation/STATUS.md`](validation/STATUS.md).
