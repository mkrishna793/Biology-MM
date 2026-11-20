"""
`FoundationCore` — a modular molecular-simulation core.

This package is the executable counterpart to the proofs in
`lean/` and the design document `FRAMEWORK.md`. It is
deliberately narrow: every behaviour it claims is implemented, tested against an
independent reference, and labelled with what it does *not* establish.

## What is supported

| Capability | Status | Evidence |
| --- | --- | --- |
| Typed state, orthorhombic periodic box | implemented | `test_factors.jl` (typed-state validation) |
| Bonded factors (bond/angle/dihedral) | implemented | finite-difference adjoint tests |
| Lennard-Jones with 4 cutoff conventions | implemented | NIST reference data |
| Plain Coulomb (non-periodic / minimum image) | implemented | finite-difference tests |
| Exactly summarised bipartite harmonic group | implemented | Lean proof + exact rational test |
| Neighbour list + Verlet rebuild criterion | implemented | brute-force cross-check |
| Velocity Verlet, energy-drift diagnostics | implemented | reversibility + dt² tests |
| BAOAB Langevin | implemented | canonical Gaussian moment test |
| HMC + parallel tempering | implemented | analytic double-well KS test |
| Induced-dipole polarization (`Aμ = E`) | implemented | residual + envelope-force test |
| Local factor parameter gradients | implemented | analytic chain rule + central finite differences for all five supported factor types |
| Equilibrium parameter sensitivities | implemented for fixed-measure ensembles | closed-form harmonic check, correlated-sample ESS/error, two-leg subtraction and charge-tangent tests |
| Matrix-free induced point dipoles | narrow direct solver | dense-reference parity, guarded PCG convergence; O(N²) matvec, small-system SPD certificate only |
| Streamed periodic LJ | implemented as a narrow factor | linked-cell one-pass parity vs brute force; `:lfs`/`:shift`; explicit `PairTopology`; `StreamedLJFactor` integrated with velocity Verlet in `test_streamed_lj_dynamics.jl` |
| Structure diagnosis / staged preparation | implemented | deterministic fixture |
| Distance constraints | disjoint-pair RATTLE and coupled rigid triangles | constraint suite; WaterBox uses symmetric principal-axis free rotation with tangent kicks/noise |
| Triclinic geometry | nearest-image, generic pair factors, StreamedLJ, and PME | skew-cell exhaustive-image, force, and 9-component box-derivative tests |
| Long-range electrostatics (direct Ewald + scalar PME) | orthorhombic + triclinic implementation | PME factor is integrated with velocity Verlet and RESPA; parity/force-consistency/stress suites |
| Resolved-system schema `v1.0.0` | implemented | constructor/revalidation suite; typed errors, ownership and capability checks |
| Analytic box derivative / virial / affine pressure | implemented for local factors, streamed LJ `:lfs`, and PME | exact analytic check and multi-step fixed-fractional finite differences |
| Parameter-file import / resolved builder | AMBER prmtop + CHARMM PSF adapter; CMAP, 1-6-4, 1-4 scales, typed NBFIX refusal | five-class fixture suite; all five classes pass independent OpenMM comparison |
| Transactional workflow, manifest, atomic checkpoints & guarded analysis | implemented | `test_workflow.jl`, `test_frame_writer.jl` (SHA-256 manifest/checkpoint round-trip, bitwise continuation, interruption safety, strict MSD guards) |
| Analytic SETTLE, reversible r-RESPA & extxyz trajectory I/O | implemented | `test_settle.jl`, `test_respa.jl`, `test_workflow_respa.jl`, `test_trajectory_format_extxyz.jl` |
| CustomPhysics (`ParameterEncoder`, `CubicSplineTable`, 7 radial kernels, external/many-body factors) | implemented | `test_custom_and_reactive.jl` (4th-order FD forces & 9-entry triclinic stress) |
| ReactiveNano (Tersoff C/Si, Stillinger–Weber Si, `MorseBondFactor`, `MLIPFactor`, CNT/graphene/nanopore/diamond builders) | implemented | `test_custom_and_reactive.jl`, `test_phase9_nanotech_e2e.jl` |
| GrandCanonical (μVT GCMC, Farkas AMO steric screen, Widom insertion, `TernaryLatticeGas`, constant-pH MC) | implemented | `test_gcmc_and_nemd.jl` (ideal-gas analytical ensemble, H-H titration, lattice selectivity) |
| NEMD & RNEMD (`ExternalElectricFieldFactor`, `NanofluidicBodyForceFactor`, `WallRestraintFactor`, Müller–Plathe swaps) | implemented | `test_gcmc_and_nemd.jl`, `test_phase9_nanotech_e2e.jl` |
| EnhancedSampling (`DistanceCV`, `CoordinationCV`, `DihedralCV`, `ComDistanceCV`, `CustomCV`, `UmbrellaFactor`, `MetadynamicsBias`, WHAM, REMD) | implemented | `test_enhanced_sampling.jl`, `test_phase9_nanotech_e2e.jl` |
| FMM / fast multipole | **not implemented** | documented |

## What is not claimed

* Not a replacement for OpenMM. No head-to-head benchmark exists here.
* No FMM backend; PME is a scalar-charge backend, not a complete polarizable multipole PME force field.
* The exported MC barostat is validated in the ideal-gas NPT limit and small-system virial checks.
* StreamedLJ is a periodic (orthorhombic and triclinic), homogeneous per-atom Lorentz–Berthelot LJ backend. It accepts explicit `PairTopology` exclusions and scaled 1-4 pairs, optional persistent `PairCache` / `TilePairCache` Verlet lists, and deterministic multi-threaded evaluation. Its default `:lfs` mode is force-consistent and integrated with `Dynamics` and `Respa`; `:shift` is energy-only and refused by the integrator gate.
* Parameter gradients do not cover the nonlocal bipartite factor or parameter-dependent cutoffs; `StreamedLJFactor` uses its dedicated `streamed_parameter_gradients!` evaluator instead of the general `ParameterGradients` path. Equilibrium sensitivities average fixed-state derivatives for fixed-measure ensembles; they do not integrate finite parameter paths or establish physical binding free energies.
* The matrix-free point-dipole operator is non-periodic and direct O(N²) per matvec; no PME, damping, nuclear forces, or large-system SPD certificate is provided. Nonconverged solves cannot be used for energy/force evaluation.
* The O(nm)→O(n+m) reduction applies to *pair evaluations of one quadratic
  model*, not to per-atom reads/writes, which remain Ω(N).
* "Exact" means exact in rational arithmetic for the proved model; in float64 it
  means roundoff-limited.
* CDC is design guidance; no category is formalised in a proof assistant here.
"""
module FoundationCore

include("Types.jl")
include("Factors.jl")
include("Neighbors.jl")
include("Constraints.jl")
include("Settle.jl")
include("Dynamics.jl")
include("Respa.jl")
include("Sampling.jl")
include("Prepare.jl")
include("Polarizable.jl")
include("ParameterGradients.jl")
include("StreamedLJ.jl")
include("ResourceProbe.jl")
include("EquilibriumSensitivities.jl")
include("PolarizationOperator.jl")
include("Electrostatics.jl")
include("PME.jl")
include("ResolvedSystem.jl")
include("SystemContract.jl")
include("TriumXBT.jl")
include("TriumGPU.jl")
include("TernaryMSM.jl")
include("ChemistryFactors.jl")
include("ChemistryCMAP.jl")
include("Chemistry.jl")
include("MonteCarloBarostat.jl")
include("WaterBox.jl")
include("TrajectoryFormat.jl")
include("Workflow.jl")
include("Alchemical.jl")
include("Multiscale.jl")
include("SymplecticTernary.jl")
include("CustomPhysics.jl")
include("ReactiveNano.jl")
include("GrandCanonical.jl")
include("NEMD.jl")
include("EnhancedSampling.jl")
include("Engine.jl")

using .Types
using .Factors
using .Neighbors
using .Constraints
using .Settle
using .Dynamics
using .Respa
using .Sampling
using .Prepare
using .Polarizable
using .ParameterGradients
using .StreamedLJ
using .ResourceProbe
using .EquilibriumSensitivities
using .PolarizationOperator
using .Electrostatics
using .PME
using .ResolvedSystem
using .SystemContract
using .ChemistryFactors
using .ChemistryCMAP
using .Chemistry
using .MonteCarloBarostat
using .WaterBox
using .TrajectoryFormat
using .Workflow
using .Alchemical
using .TriumXBT
using .TriumGPU
using .TernaryMSM
using .Multiscale
using .SymplecticTernary
using .CustomPhysics
using .ReactiveNano
using .GrandCanonical
using .NEMD
using .EnhancedSampling
using .Engine

export Types, Factors, Neighbors, Constraints, Settle, Dynamics, Respa, Sampling, Prepare, Polarizable,
       ParameterGradients, StreamedLJ, ResourceProbe, EquilibriumSensitivities,
       PolarizationOperator, Electrostatics, PME, ResolvedSystem, SystemContract,
       ChemistryFactors, ChemistryCMAP, Chemistry, MonteCarloBarostat, WaterBox,
       TrajectoryFormat, Workflow, Alchemical,
       TriumXBT, TriumGPU, TernaryMSM, Multiscale, SymplecticTernary,
       CustomPhysics, ReactiveNano, GrandCanonical, NEMD, EnhancedSampling, Engine

const VERSION = v"1.0.0"

end # module FoundationCore
