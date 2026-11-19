# Polarization Scaling Spike

**Status: bounded research spike, not an accelerated implementation.** The existing direct non-periodic operator was measured, its dense parity and small-system derivatives were checked, and the requested hierarchical and periodic algorithms were not implemented in this spike. The negative scaling result is the main finding: the direct operator is plainly not viable at 100k sites.

## Model and conventions

The fixed-geometry energy is

```text
U(q, μ) = U_perm(q) + 1/2 μᵀ A(q) μ - μᵀ E(q),
A_ii = I/α_i,  A_ij = -T_ij,
T_ij = k_e (3 rhat rhatᵀ - I)/r³.
```

Thus the off-diagonal block is `k_e (I/r³ - 3 ddᵀ/r⁵)`, where `d = r_i-r_j`; it is symmetric under exchange of sites. `∂U/∂μ=Aμ-E`, so the unique minimum satisfies `Aμ=E` only when `A` is SPD. `U_perm` is independent of the minimization here. The tested production operator is undamped, non-periodic, isotropic, removes zero-alpha unknowns, and takes a caller-supplied consistent Coulomb prefactor. It does not include Thole damping; the separate `Polarizable` model's exponential damping must not be silently conflated with this operator.

The dense oracle in `run.jl` is assembled columnwise through the production `mul!` for small N; consequently parity with it is a structural consistency check, not an independent derivation. The numerical finite-difference checks use independently formed dense perturbations of the operator. All production source was read-only.

## What was measured

Machine: macOS arm64, 27 GiB physical memory (`hw.memsize`), Julia 1.13.3, Julia threads 1, BLAS threads 1. Julia depot was the configured project depot. The benchmark generated deterministic jittered cubic lattices with `MersenneTwister`; matvec timings are one warmed call and include no repeated-sample statistics. The free+inactive+speculative preflight reported 10,208, 10,197, and 9,714 MiB before the 1k, 10k, and 100k fresh subprocesses, respectively. The 100k attempt to impose a 2 GiB virtual-memory limit was rejected by zsh/macOS; therefore no hard RSS limit or peak RSS measurement is claimed.

| N | setup/build (s) | one warmed direct matvec (s) | matvec (ms) | status |
|---:|---:|---:|---:|---|
| 1,000 | 0.002711 | 0.002474 | 2.474 | measured |
| 10,000 | 0.313069 | 0.283875 | 283.875 | measured |
| 100,000 | 102.531237 | 49.417538 | 49,417.538 | measured; fresh process |

Two-point endpoint fit gives `t ∝ N^2.15`; the adjacent intervals give exponents 2.06 (1k→10k) and 2.24 (10k→100k). This is consistent with the explicit all-pairs loop and is not evidence for an asymptotic accelerated operator. Build also validates all pair geometries directly and is quadratic. No 1e5 solve was run because each matvec already costs about 49 s; no 1e5–1e6 extrapolation is presented as a measurement. Peak RSS was not captured.

| Check | Result |
|---|---:|
| Operator vs dense oracle, N=24, relative matvec error | `3.679e-16` |
| PCG vs dense solve, N=24, relative dipole error | `3.676e-13` |
| Production PCG, N=24, true relative residual / iterations | `3.675e-13` / 3 |
| Dense diagonal-PCG N=90, cold / perturbed warm iterations | 3 / 3 |
| Dense 3x3 site-block Jacobi N=90 iterations / residual | 3 / `1.447e-12` |
| Envelope gradient central-FD absolute / relative discrepancy | `2.954e-14` / `2.482e-7` |
| μ-dependent observable one-adjoint central-FD absolute / relative discrepancy | `1.639e-15` / `1.843e-9` |

The fixture is deliberately weakly coupled (`α=2e-4` at N=24 and `3e-4` at N=90) and is very well conditioned. Three iterations across these tiny tests say nothing about iterations vs N or strong coupling. Warm start saved zero iterations in this test. A conditioning sweep, realistic chemistry, solver timings, and a nontrivial perturbation study remain unmeasured.

## Requested algorithms not established

**Range split and hierarchy: not implemented.** There is no cell-list near operator, octree, admissibility traversal, multipole expansion, expansion-order sweep, theta sweep, or hierarchical error-vs-dense table. Consequently there is no chosen admissibility/error point. Fitting a low-rank Taylor expansion for the tensor kernel requires a carefully differentiated expansion and symmetric pair-block treatment; the classic `a_A+a_B < theta R` criterion alone is not a correctness result. The direct data show why an accelerated replacement is needed, but do not establish its speed or accuracy.

**Periodicity: not implemented or validated.** Minimum-image dipole sums are not a periodic Ewald method. No reciprocal dipole contribution, direct dipole Ewald comparison, or surface-term test was run. For a spherical macroscopic boundary with dielectric `ε_out`, the standard dipolar surface energy in Gaussian Ewald convention is `2π |M|² / ((2ε_out+1)V)`; vacuum gives coefficient `2π/(3V)`, while conducting/tin-foil boundary sets this term to zero. The corresponding field is the derivative of this energy and its Hessian contributes to the induced-dipole operator. A test with nonzero total cell dipole distinguishes vacuum from tin-foil and fails if the selected vacuum term is omitted. This spike selects no boundary condition because it implements neither periodic method. Any integration must select and test one boundary explicitly; omitting the term silently changes the boundary physics.

**Preconditioning and PCG: partial prototype only.** The existing production Jacobi-PCG and small dense SPD check were exercised. A separate dense diagnostic used scalar diagonal and exact 3x3 site-block Jacobi; this is not the requested neighbor-block preconditioner and is not a scalable matvec. No N-dependent/conditioning iteration curve, 100k solve, or realistic residual/time benchmark exists.

**Adjoints: small algebraic numerical check only.** For a parameterized symmetric system `A(θ)μ(θ)=E(θ)`, stationarity gives the envelope derivative `dU*/dθ = 1/2 μᵀ A_θ μ - μᵀ E_θ` (plus `∂U_perm/∂θ`). For an observable `O(μ,θ)`, solve `Aᵀλ=∂O/∂μ`; then `dO/dθ = ∂O/∂θ + λᵀ(E_θ-A_θ μ)`. The experiment compared these formulas to fully re-solved central finite differences on a five-site dense system; values are in the table. These checks do not validate nuclear-force derivatives of the spatial kernel.

For exact equilibrium marginalization over Gaussian induced dipoles, integrating `μ` contributes `(1/(2β)) log det A(θ)` to the effective potential, up to constants. Its derivative is `(1/(2β)) Tr(A⁻¹ A_θ)`. Hutchinson estimation uses random probes and solves with A, but the trace estimator's variance and solve error must be controlled. Omitting this determinant samples the minimized-energy/Born-Oppenheimer target, not the exactly marginalized distribution; these are different distributions.

**Extended Lagrangian and Drude.** Extended-Lagrangian induced dipoles carry auxiliary momenta and require a fictitious mass plus a stable inner integration; this can replace a tight self-consistent solve with a few force iterations, but introduces timestep, thermostat, reversibility/energy-conservation, and adiabatic-separation gates. Drude oscillators make polarization explicit as additional charged particles with stiff springs; the fast mode forces a small timestep or a validated dual thermostat/extended-system treatment, with roughly three extra coordinates per polarizable site. Recommendation: prototype a validated periodic reciprocal/real-space operator and symmetric preconditioner first, since both alternatives still need correct periodic forces and units. Then compare extended-Lagrangian against converged PCG on a small matched model before considering Drude integration.

## Integration plan and acceptance gates

1. Freeze conventions: unit system, Thole pair damping convention, permanent field, exclusions, active sites, and explicit periodic boundary condition. Pass hand-derived two-site sign/unit tests and independent small dense matrix/energy checks.
2. Implement a direct periodic induced-dipole Ewald reference in an isolated module. Verify energy-field/operator derivatives by finite differences; compare real and reciprocal convergence against tighter sums; include a nonzero-cell-dipole surface-term test for both selected boundary and a deliberately omitted-term negative control.
3. Implement cell-list near field plus a symmetric hierarchical far field. At each expansion order and admissibility theta, publish operator relative/absolute error and symmetry defect against dense systems up to a feasible small N; test force/energy consistency if forces are exposed. Require error below a stated application tolerance before speed claims.
4. Add diagonal and 3x3 block preconditioners, then a neighbor-block preconditioner only if measured convergence justifies its setup/storage cost. Check true residual, `μ` agreement, SPD/curvature behavior, and iterations over both N and alpha/geometry conditioning.
5. Run each scale in a fresh process after available-memory preflight. Record peak RSS from a supported process monitor, warm and cold timings, repeated matvec/solve samples, hierarchy rebuild cost, and error against the reference. Gates at 1k/10k/100k; do not attempt 1e6 until 100k RSS is <2 GiB and accuracy/cost gates pass.
6. Integrate into `src` only after all preceding gates, with independent tests and explicit failure on nonconvergence or non-SPD operators. Validate relaxed nuclear forces and parameter gradients by finite differences. For HMC, state whether the target is minimized-energy or includes the log determinant.

No production source change is proposed by this spike, so there is no source diff to apply.

## Recommendation

The proposed **Ewald-split + hierarchical far field + PCG** architecture is a plausible route in principle, but this spike provides no evidence that it is viable at 1e5–1e6. The measured current direct path costs 49.4 s per matvec at 100k, before the many PCG iterations, and its setup costs 102.5 s. The remaining bottleneck is unambiguously the all-pairs operator/build; periodic reciprocal acceleration, hierarchy accuracy, preconditioner quality, memory, and force consistency are all open. Integrate nothing from this experiment into production yet. Next prototype should be a small periodic dipole Ewald reference with explicit surface convention, then compare a symmetric hierarchical operator to it and to dense results before scaling.

## Bounded follow-up probe (2026-11-12)

This follow-up is intentionally bounded by the available Julia project dependencies and concurrent production work. It adds only `julia/experiments/polarization_scaling/factored_probe.jl`; it does not modify production sources. The project does not declare FFTW, and no FFT/Ewald dependency was installed. Therefore the spectral factorization, a cell-list near field, full hierarchy, and N=1e3/1e4/1e5 factored benchmarks were **not implemented**. The prior direct measurements above remain the only full-operator scale results.

### Probe methods and results

The probe forms one dense 144-by-144 off-diagonal tensor block for two randomly jittered compact 48-site clusters, centers separated by 6 length units. It computes a dense SVD, then measures truncated Frobenius error and the `r*(m+n)` projection/expansion work proxy. The admissibility radii use the enclosing cube bound `a=sqrt(3)*0.25=0.433013`; thus `a_A+a_B < 0.5 R` holds for this block. This checks the proposed separable factorization on one geometry, not a cluster tree or operator-wide error.

| Rank | Relative block Frobenius error | Projection/expansion proxy | Dense block entries |
|---:|---:|---:|---:|
| 3 | 1.143283e-3 | 864 | 20,736 |
| 6 | 6.049471e-4 | 1,728 | 20,736 |
| 12 | 6.454541e-7 | 3,456 | 20,736 |
| 24 | 1.195882e-12 | 6,912 | 20,736 |
| 48 | 1.427206e-15 | 13,824 | 20,736 |

Rank 12 is the best measured compromise in this single block, but the proxy is not a wall-time measurement and the dense SVD is not how a scalable factor would be constructed. No theta sweep, expansion-order sweep, hierarchy traversal, near-field split, full dense-oracle application comparison, or N exponent was measured. Accordingly the low-rank exponent is **not available**; presenting this block as an O(Nr) measured full-operator result would be incorrect.

The same runner checks a vacuum spherical-boundary surface contribution in the Gaussian Ewald convention. For `M=(1.25,-0.5,0.75)`, `V=1000`, it computes `U_s=2π|M|²/(3V)=4.974188368184e-3`; the negative control sets the term to zero and fails the equality assertion. Output reports `boundary=vacuum ... omission_detected=true`. This is a formula-level regression test, not an Ewald sum or reciprocal-operator test. Tin-foil has zero macroscopic surface term by convention; it is not tested by a periodic calculation here.

The exact reproducible command was:

```sh
~/.cache/biology-mm/julia-1.13.3/bin/julia --project=julia julia/experiments/polarization_scaling/factored_probe.jl
```

Observed output: Julia 1.13.3, Julia threads 1, BLAS threads 1; the table values above; `SURFACE boundary=vacuum M2=2.37500000 V=1000.0 U_surface=4.974188368184e-03 omission_detected=true tin_foil=0`. `/usr/bin/time -l` on a fresh process reported maximum resident set size 306,626,560 bytes (peak memory footprint 247,661,648 bytes), wall 1.88 s. This RSS includes Julia startup and this tiny diagnostic only; it says nothing about the missing scalable algorithms. Available-memory preflight and isolated 1k/10k/100k benchmark processes were not run for this probe.

### Required results not established

| Requirement | Status / evidence |
|---|---|
| Direct full operator | **MEASURED previously:** endpoint exponent 2.15; 49.42 s/matvec at N=100k; no solve at that size. |
| Separable low-rank blocks | **MEASURED narrowly:** one 48x48-site cluster block, table above. Full hierarchy error/cost curve and exponent unavailable. |
| Periodic spectral/FFT factorization | **NOT IMPLEMENTED:** FFTW absent from declared project dependencies; no `D(k)`, periodic dipole Ewald reference, crossover, or log-det mode sum. |
| Hard aqueous-like coupled fixture; diagonal and neighbor-block PCG; conditioning and N iteration curve; cold/warm MD perturbations | **NOT MEASURED.** The earlier weak fixture's zero warm-start saving remains explicitly nonrepresentative. |
| Hard-fixture envelope force and one-adjoint finite-difference checks; error versus SCF residual | **NOT MEASURED** beyond the earlier small algebraic check in the table above. |
| Exact marginal log-det sampling demonstration | **NOT IMPLEMENTED.** The Gaussian identity in the model section remains mathematical framing only; no distribution-shift experiment or approximate factor determinant was measured. |
| Full-system factored matvec/solve timings and peak RSS at 1k/10k/100k | **NOT MEASURED.** Only the tiny diagnostic's fresh-process RSS is reported above. |

No factorization exponent, crossover N, hard-fixture iteration count, adjoint hard-fixture error, or HMC marginal shift can be honestly reported from this run. The requested claim that factored polarization reaches 1e5–1e6 is therefore **not established**. The only new measured accuracy point is a small low-rank block; the only surface-term test is the formula-level vacuum regression above. The immediate bottleneck remains absence of a validated periodic dipole reference and an operator-level factorization, not the SVD rank of one block.

### Revised integration gates and source-diff status

There is no proposed source integration diff: without a periodic reference, full-operator accuracy and symmetry results, hard-fixture convergence, or scale/resource measurements, a src change would be premature. In particular, do not copy the vacuum surface formula into a production solver without deriving the reciprocal operator, signs, units, and force derivative consistently.

1. Add an isolated periodic dipole Ewald reference with a named boundary condition, explicit surface energy/operator, and a nonzero-total-dipole test whose deliberately omitted-term control fails. Compare matrix, energy, and coordinate derivatives to independent finite differences.
2. Against that reference and small dense matrices, implement symmetric near/far application; publish full-operator relative error, symmetry defect, and wall-time by N, theta, and rank/order. Do not accept a block-only SVD table as this gate.
3. Add realistic aqueous-like configurations and report min eigenvalue, true residual, diagonal and neighbor-block PCG iterations, conditioning, and cold/warm perturbation results. Refuse forces/energy on non-SPD or non-converged states.
4. Validate envelope forces, one-solve observable adjoints, and their error versus residual. Separately validate exact log-det corrections and a finite-dimensional marginal distribution shift before describing HMC semantics.
5. Benchmark each implementation in fresh, sequential processes at 1k/10k/100k, with available-memory preflight, repeated timings, `ru_maxrss`, and reference error. Fit exponents with uncertainty; attempt 1e6 only after 100k accuracy and <2 GiB RSS gates pass.
6. Only after all gates pass, prepare a unified src integration diff plus production tests. At present this report contains **no src diff**, because there is no safe, supported production API or algorithm to propose.

Verification note: the initial required `sh verify.sh --core` preflight passed all 17 Julia suites. The final rerun after this documentation/probe-only change passed proofs and 15/17 suites; `test_factors` failed `triclinic cell must raise UnsupportedGeometry` and `test_dynamics` failed `maximum(abs, reverse.q .- q) < 1e-8`. Both suites passed in the initial run. These failures are outside the two writable paths and were not modified or diagnosed further; with other work active on this tree, their cause is not established. The probe command passed on its final rerun.

## Full-Ewald spline follow-up (2026-11-12)

**Verdict: small-system full-Ewald accuracy demonstrated; complete scalable solve and integration readiness are not demonstrated.** The experiment-only file `julia/experiments/polarization_scaling/full_ewald_probe.jl` adds a linked-cell real term and order-4/order-6 cardinal B-spline assignment around the previously measured reciprocal tensor. It does not modify production sources. The run used Julia 1.13.3, FFTW 1.10.0, one Julia thread, one FFTW thread, and one BLAS thread. FFTW remains isolated in the experiment environment.

### Operator, energy, and boundary convention

The operator field is the real-space Hessian of `erfc(αr)/r`, plus the Gaussian-screened reciprocal dipole Hessian, Gaussian dipole self correction `+(4α³/(3√π)) μ`, and (for spherical vacuum) field `-(4π/(3V)) Σμ`. The code's reciprocal field is negative semidefinite; the induced linear-system operator is `A = α_site⁻¹ I - E_operator`. The energy convention for a dipole interaction is `-½ μᵀ E_operator μ`, with the vacuum surface energy `2π|M|²/(3V)` included consistently. Tin-foil sets the macroscopic surface term to zero. The previously recorded surface negative control remains the explicit measured boundary test; this new probe compares both conventions only through that same surface delta, not through separate force-field validation.

**[PROVEN, algebraic identity]** The reciprocal `ik` tensor and vacuum surface field are symmetric linear maps. **[MEASURED]** The implemented total field passed a bilinear symmetry check at relative defects `6.17e-16` (order 4) and `1.66e-14` (order 6). This tests discrete symmetry, not positive definiteness for all geometries.

The assignment uses tensor-product centered cardinal B-splines, normalized per axis, spreads each dipole component to `p³` mesh points and gathers with the identical weights. The reciprocal influence is divided by the squared assignment window `Π_d sinc(n_d/N_d)^(2p)`. Reciprocal differentiation is spectral `ik`; Nyquist planes are omitted to preserve an unambiguous Hermitian derivative. Orders 4 and 6 were implemented. **[MEASURED]** On the tested random fixtures, the discrete total-field result agreed with the direct Ewald oracle as shown below. Energy error is relative error in `μᵀE`; field and energy have different cancellation behavior and both are reported.

### Error surface against direct Ewald

The direct reference is the existing explicit real-image and reciprocal-integer sum, plus the same analytic self and selected surface terms. The N=64 fixture has a 12³ box and seed 164. Each `(α, rcut, kmax)` row is compared against its direct oracle at those same parameters. Values are relative errors; this is a numerical error surface, not a statistical confidence interval.

| α | rcut | kmax | spline | 12³ grid field / energy | 18³ grid field / energy | 24³ grid field / energy |
|---:|---:|---:|---:|---:|---:|---:|
| 0.40 | 5.0 | 8 | 4 | 9.46e-5 / 3.37e-4 | 1.16e-5 / 1.12e-4 | 3.43e-6 / 1.00e-5 |
| 0.40 | 5.0 | 8 | 6 | 8.22e-6 / 4.04e-5 | 3.15e-7 / 3.17e-6 | 4.71e-8 / 2.51e-7 |
| 0.55 | 5.5 | 9 | 4 | 9.53e-4 / 2.12e-3 | 9.75e-5 / 1.21e-3 | 3.03e-5 / 1.43e-4 |
| 0.55 | 5.5 | 9 | 6 | 2.10e-4 / 6.24e-4 | 6.39e-6 / 7.62e-5 | 9.06e-7 / 4.18e-6 |
| 0.70 | 5.9 | 12 | 4 | 5.15e-3 / 4.37e-2 | 5.73e-4 / 6.24e-3 | 1.72e-4 / 6.49e-4 |
| 0.70 | 5.9 | 12 | 6 | 2.18e-3 / 6.27e-2 | 8.31e-5 / 7.64e-4 | 9.80e-6 / 2.53e-5 |

An additional refinement at α=.55, rcut=5.5, kmax=9 gives the following order-6 N=64 curve:

| Grid | Relative field error | Relative energy error |
|---:|---:|---:|
| 24³ | 9.06e-7 | 4.18e-6 |
| 32³ | 1.07e-7 | 5.78e-7 |
| 40³ | 3.12e-8 | 1.96e-7 |

For comparison, the old NGP reciprocal-only error at 24³ was 1.04e-1 against the direct reciprocal sum, while the total Ewald error was about 9.52e-1 because real and self terms were missing. The spline result is therefore a major measured improvement and includes all four terms, but the tables cover only 32/64-site random fixtures and selected cutoffs. They do not establish uniform application-level tolerance across configurations or parameter gradients. The direct Ewald convergence controls recorded in the prior section remain separate and are near roundoff for their tested settings.

### Real-space implementation and self/surface checks

**[MEASURED]** `real_cell_field` uses flat `head`/`next` linked-cell arrays, a 27-cell periodic stencil when there are at least three cells per dimension, deduplicated visited cells, and an all-cell fallback for smaller cell counts. It applies the minimum-image real term to ordered site pairs. Full results include the analytic self correction and either vacuum or tin-foil macroscopic boundary field. The reciprocal/self/surface implementation matches the oracle's convention; the small-system field and energy comparisons above are the acceptance evidence. No separate exhaustive image-vs-cell-list parity sweep or large-configuration self-term isolation sweep was run.

### PCG and complete-operator scaling

The available PCG measurement is **not the requested hard fixture**. It used N=1,000, density 0.8, site polarizability `1e-5`, rcut about 5.38, and a 16³ mesh. The diagonal dominated: PCG reached relative residual `3.39e-9` in 4 iterations. There are no valid results at N=10,000 or 100,000 for iterations, final residual, or dense-reference solution error; no strong-coupling fixture was solved. Consequently iteration growth with N and preconditioner adequacy remain **NOT MEASURED**.

**[MEASURED, preliminary complete-operator timing]** A warmed single application, including cell construction, real term, spline spreading/gathering and FFT, took 1.025 s at N=1,000 (16³ grid) and 18.553 s at N=10,000 (32³ grid), with density 0.8. The two-point endpoint exponent is 1.258, but this is not a reliable asymptotic fit: the implementation allocates per target and builds temporary cell vectors, only two N values were timed, and no 100,000-site run was attempted. The N=10,000 full process took 60.18 s wall including Julia startup, oracle comparisons, PCG, and both timings. `/usr/bin/time -l` reported maximum RSS 453,902,336 bytes (peak footprint 353,879,192 bytes) for that fresh process. This stays below 4 GiB, but is not a 100k memory result.

There is no complete-operator N=100,000 matvec or solve timing, and there is no 1,000/10,000/100,000 dense-vs-spline comparison at matched accuracy. The historical direct dense N=100,000 matvec was 49.42 s at a different fixture and accuracy; it cannot be used to claim a matched-accuracy crossover. An accurate small-system point does not make the present slow 10k probe production-scalable.

### Parameter derivatives, diff, and integration gates

For a charge-source reciprocal Ewald term, the charge gradient can be expressed through the reciprocal structure factors and the influence function; an assignment-based PME implementation must differentiate the same deconvolved spread/gather energy, not merely reuse the dipole `ik` field. The dipole experiment has no charge-energy API and no coordinate, charge, or spline-window derivative tests. **[NOT IMPLEMENTED / NOT MEASURED]** Parameter-gradient consistency, force consistency, reciprocal charge gradients, PCG at N=10k/100k, and a hard fixture remain open.

No production unified diff is proposed in the experiment directory. Although one small-fixture accuracy point passes a tight numerical field target, the production API, units, damping/exclusion semantics, charge gradients, coordinate forces, full solve behavior, and 100k scale/resource gates are not validated. A speculative source diff would not be an integration-ready patch. Required gates remain: independent oracle field and energy tolerances across seeds and Ewald parameters; symmetry and SPD/curvature refusal; true PCG residual and dense-reference dipoles; force/charge/parameter finite differences including reciprocal assignment; both explicit boundary conventions; and fresh-process matched-accuracy timings/RSS at N=1k/10k/100k. The integration verdict is **NOT READY**.

Reproducibility command:

```sh
JULIA_DEPOT_PATH="$HOME/.cache/biology-mm/julia-depot" \
  "$HOME/.cache/biology-mm/julia-1.13.3/bin/julia" \
  --project=julia/experiments/polarization_scaling \
  julia/experiments/polarization_scaling/full_ewald_probe.jl
```

### Verification and evidence labels

**[MEASURED]** The above command completed with symmetry and direct-oracle comparisons, one N=1,000 PCG solve, and N=1,000/10,000 complete matvec timings. **[MEASURED]** A separate order-6 24³/32³/40³ refinement command produced the values above. **[NOT MEASURED]** N=100,000 complete matvec/solve/RSS; hard coupled PCG iterations; matched-accuracy dense crossover; broad seed/error statistics; charge/force gradient checks. **[ASSUMED]** The spline-window deconvolution and `ik` method are the intended PME discretization; their stated implementation is not a substitute for those missing derivative and integration tests.

## Workspace optimization follow-up (2026-11-12)

**Verdict: allocation pathology fixed and cell-pair ownership corrected, but physical real-space cost and solve gates still fail. Integration remains NOT READY.** `full_ewald_probe.jl` retains allocating before routines for reproducible comparisons and adds reusable `EwaldWorkspace{P}` state: cached linked-cell membership, concretely typed `Stencil{P}` assignment data, FFTW 1.10.0 plans, and reusable grid/output buffers. Spread and gather each visit exactly `P^3` mesh points per site (local stencil), not every grid point. FFT work is `O(G log G)` for G grid points. Geometry and assignment stencils are cached and must be rebuilt if coordinates change; moving-geometry updates are not implemented.

### Bottleneck measurements

**[MEASURED, original baseline, N=10,000, density 0.8, 32³ grid, one Julia/FFTW/BLAS thread]** The original real-space call allocated `11,124,035,072` bytes and took `3.479 s`; reciprocal spline/FFT allocated `2,338,563,408` bytes and took `13.698 s`. The real term created temporary coordinate slices and cell vectors inside loops. The reciprocal path rebuilt `Vector{Any}` spline stencils, grid arrays, and FFTs each call. The prior complete-operator measurement was 18.553 s in its own run.

**[MEASURED, after, median of three warmed calls, density 0.8]**

| N | grid | real before: alloc / s | real after: alloc / s | reciprocal before: alloc / s | reciprocal after: alloc / s |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 16³ | 593,842,592 B / 0.112 s | 32 B / 0.0283 s | 284,602,432 B / 0.838 s | 128 B / 0.000474 s |
| 10,000 | 32³ | 11,124,742,976 B / 3.810 s | 32 B / 0.473 s | 2,338,179,664 B / 8.783 s | 128 B / 0.0129 s |

The post-edit before values use retained allocating routines and a changed runner/fixture, so only the initial profiler row is the direct original baseline. **[DIAGNOSIS, MEASURED]** That intermediate workspace used `floor(L/rc)` and two cells per axis at N=10,000, then scanned distinct cells per target. It introduced an avoidable O(N²) candidate traversal.

**[MEASURED]** Optimized reciprocal-vs-allocating reciprocal relative discrepancies were `3.01e-16`, `3.09e-16`, and `3.23e-16` at N=128/512/2048 on the compact correctness fixture. The optimized total field returned exact parity to its wrapper on N=24 (`0.0` relative discrepancy); bilinear symmetry defects in the full accuracy runner were `3.70e-15` (order 4) and `1.20e-14` (order 6). These are fixture-specific parity checks, not broad accuracy or moving-geometry cache validation.

**[MEASURED]** An intermediate fresh process for N=1,000/10,000/100,000 reported maximum RSS `532,250,624` bytes; its real timings (28.3 ms, 473 ms, 8.42 s) are superseded by the corrected ownership kernel below.

### Hard solve and scaling status

**[NOT MEASURED]** The hard-coupling PCG fixture remains unrun. The only full-operator solve is the earlier weak, diagonal-dominated N=1,000 fixture: 4 iterations and true relative residual `3.39e-9`. No strong-coupling fixture, N=10,000 or N=100,000 solve, diagonal-vs-neighbor-block preconditioner comparison, or error against an independent dense reference was completed. The old fixed 20-step diagnostic is not a convergence result. Iterations vs N, full solve time, and matched-accuracy crossover against dense remain unavailable; no full-operator exponent with uncertainty can be reported.

### Corrected cell ownership and remaining profile

**[MEASURED]** The corrected workspace uses `nc[d]=ceil(L[d]/rcut)` (no clamp to 2), so cells are at most `rcut` wide. The stencil radius is `ceil(rcut/cell_width)+1`, with periodic targets deduplicated and cell pairs whose minimum box gap exceeds `rcut` rejected at workspace construction. Each unordered cell pair belongs to the lower cell ID; same-cell pairs use `i<j`. Every accepted physical dipole pair is evaluated once and applied to both endpoints. At density 0.8, `L` / `nc` per axis are N=1,000: 10.772 / 2; N=10,000: 23.208 / 5; N=100,000: 50.000 / 10. No per-target all-cell scan remains. The 1k case has only two cells per axis, so its broad candidate stencil is expected and deduplicated, rather than clamped or handled quadratically per target.

**[MEASURED, corrected cached component run, three warmed samples, fresh process]**

| N | grid | real before alloc / time | real after alloc / median time | reciprocal after alloc / median time |
|---:|---:|---:|---:|---:|
| 1,000 | 16³ | 593,841,040 B / 0.183 s | 32 B / 23.3 ms | 128 B / 0.989 ms |
| 10,000 | 32³ | 11,124,742,976 B / 5.927 s | 32 B / 374 ms | 128 B / 23.0 ms |
| 100,000 | 64³ | not run (allocating baseline unsafe) | 32 B / 4.924 s | 128 B / 247 ms |

The original N=10,000 profile was 11.12 GB / 3.479 s for real space and 2.34 GB / 13.698 s for reciprocal. The corrected run's allocating baseline is from the retained routine and current process, not the exact original binary. Corrected fresh-process maximum RSS was `579,256,320` bytes (peak footprint `477,316,368` bytes), Julia/FFTW/BLAS one thread. Optimized-vs-wrapper total field error at N=24 was `0.0`; symmetry defects were `3.54e-16` (order 4) and `1.33e-14` (order 6). The test also confirms the assignment is local: order P loops over exactly P³ grid points per site, so order 6 is 216 points/site (not 512; the 8-point-per-axis estimate in the request does not match this implementation's P=6 support).

**[MEASURED]** Pair ownership and allocation pathologies are fixed, but exact real-space time remains 374 ms at N=10k and 4.924 s at N=100k. This does not meet the millisecond/tens-of-milliseconds targets. The remaining cost is the number and cost of accepted physical interactions: density 0.8 and radius 5.5 imply about `0.8*(4π/3)*5.5^3 ≈ 558` sites in the cutoff sphere per target asymptotically, with additional periodic crowding in the N=10k box. This scalar Julia kernel achieves only several million pair evaluations/second in the measured run. A lower physical cutoff or approximation changes the operator and needs a separate accuracy study; neither is claimed here.

**[NOT MEASURED]** Hard-coupling PCG at N=1k/10k/100k; diagonal vs neighbor-block iterations; dense-reference solution error; complete cached matvec and solve scaling exponent/uncertainty; matched-accuracy dense crossover; charge gradients through the same assignment/deconvolution. Only the weak N=1k four-iteration solve is available. No production unified diff is supplied. Integration remains **NOT READY**.

## Spectral reciprocal dipole experiment (2026-11-12)

**Verdict: measured reciprocal-space acceleration, not a validated periodic polarization operator. Do not integrate.** The new isolated experiment files are `julia/experiments/polarization_scaling/spectral_probe.jl`, `Project.toml`, and `Manifest.toml`; no production source is changed. The first two are experiment-only. FFTW was absent from both the configured Julia depot and package environment; FFTW was installed into this experiment project only (FFTW 1.10.0, AbstractFFTs 1.5.0). Machine/runtime: macOS arm64, Julia 1.13.3, Julia threads 1, FFTW threads 1, BLAS threads 1.

### Derivation and conventions

**[PROVEN, continuum identity]** In Gaussian units, periodic Coulomb Green function has reciprocal coefficient `4π/(V k²) exp(-k²/(4α²))`. A dipole source has Fourier charge `-i k·μ`; taking the field gradient gives `T(k)=-(4π/V) exp(-k²/(4α²)) kkᵀ/k²`. Since this code's linear-system operator is `A=α_site⁻¹ I - T`, its reciprocal contribution is

```text
A_recip(k) = (4π/V) exp(-k²/(4α_Ewald²)) kkᵀ/k², k != 0.
```

The requested un-damped expression `D(k)=4π kkᵀ/(V k²)` is the α_Ewald→∞ factor (or the unsplit continuum Fourier kernel), not by itself a converged periodic Ewald operator. Here the direct reference uses the Gaussian Ewald screening factor; the reciprocal FFT truncates at integer `kmax`. For `k=0`, the conducting/tin-foil convention sets the mode to zero. This is a rank-one projector per mode, hence reciprocal work is three forward FFTs, one 3×3 tensor-vector multiply per mode, and three inverse FFTs, `O(G log G)` for G grid points.

**[PROVEN, boundary identity]** The macroscopic spherical-boundary dipole energy in Gaussian units is `U_surface=2π|M|²/((2 ε_out+1)V)`. Spherical vacuum (`ε_out=1`) therefore has `U_s=2π|M|²/(3V)` and field contribution `E_s=-∂U_s/∂μ_i=-(4π/(3V))M` for every site; the operator has `A_s=+(4π/(3V)) 11ᵀ ⊗ I`. Tin-foil (`ε_out→∞`) has zero. The reciprocal `k=0` mode is therefore zero only after the macroscopic boundary contribution has been handled separately; this experiment explicitly adds the vacuum surface field and uses zero for tin-foil.

**[PROVEN, Gaussian self identity]** The smooth Gaussian dipole field has a same-site Hessian `-(4α_Ewald³/(3√π)) I`; excluding the singular i=j real-space term requires adding `+(4α_Ewald³/(3√π)) μ_i` to the field (equivalently subtracting that term from A). `direct_ewald` includes this self correction. Energy is computed as `-½ Σ μ_i·E_i` for a pure dipole interaction; the surface energy is separately reported. `k_e=1` in this Gaussian-unit experiment; production's unit/prefactor and damping conventions are not asserted equivalent.

### Reference, negative control, and measured errors

**[MEASURED]** `direct_ewald` sums the real tensor
`T_real = (3 erfc(αr)/r⁵ + (6α/r⁴+4α³/r²)e^{-α²r²}/√π) rrᵀ - (erfc(αr)/r³ + 2αe^{-α²r²}/(√πr²))I`
over periodic images through `rcut`, the reciprocal tensor through `kmax`, self, and spherical-vacuum surface field. Error controls are explicit. At N=32, box 12³, against `(α,rcut,kmax)=(.55,12,12)`, parameter sets `(.40,16,8)`, `(.50,13,10)`, and `(.70,10,16)` differed in field by `6.72e-15`, `4.21e-15`, and `7.91e-15`, respectively; induced interaction energy relative differences were `1.89e-14`, `1.26e-15`, and `1.11e-14`. Reciprocal-tail changes were negligible by kmax 9 at α=.55; truncation error is dominated by real-space truncation in the lower-alpha test.

**[MEASURED, NGP reciprocal-only error; not total operator accuracy]** The mesh spreads each dipole to the nearest grid point (NGP), applies FFTW's normalized inverse transform with explicit grid-count scaling, then gathers the same grid point. Box 12³, N=32, α=.55, kmax=9:

| Grid | h | reciprocal field relative error | reciprocal energy relative error | total direct Ewald field error |
|---:|---:|---:|---:|---:|
| 8³ | 1.500 | 3.028e-1 | 7.835e-2 | 9.715e-1 |
| 12³ | 1.000 | 2.088e-1 | 2.094e-2 | 9.607e-1 |
| 16³ | 0.750 | 1.626e-1 | 2.672e-2 | 9.533e-1 |
| 24³ | 0.500 | 1.040e-1 | 1.697e-2 | 9.519e-1 |

The relative errors compare the FFT's **screened reciprocal part** to an independent direct reciprocal sum, and separately compare the reciprocal-only result to the complete Ewald result. NGP's grid error remains 10.4% in field at the finest tested grid; the omitted real-space and self terms leave about 95.2% field error against the full reference. This is a negative result, not a PME accuracy claim.

**[MEASURED, α/kmax sweep, 24³ grid]** At α=.40, .55, .70, reciprocal field relative errors were `.11354`, `.10401`, `.09679`; energy errors `.03021`, `.01697`, `.00862`. At each α, kmax 6, 9, 12 gave essentially the same error by 9 (the mesh error dominates). **[MEASURED]** Surface regression with two parallel unit dipoles, box 8³, `M=(2,0,0)`: vacuum energy `1.636246174e-2`, field-vector omission delta `2.314001530e-2`; assertion confirms full-vacuum minus tin-foil field equals `-(4π/(3V))M` at both sites. Test passes and fails if the term is omitted.

### Scaling and PCG

**[MEASURED, reciprocal-only, one run set]** Fixed density 0.05 sites/unit³ and grid side chosen as next power of two above `1.25 N^(1/3)`. Reported matvec is the median of three warmed runs; samples in milliseconds:

| N | grid | matvec ms (samples) |
|---:|---:|---:|
| 1,000 | 16³ | 0.498 (0.498, 0.497, 0.536) |
| 10,000 | 32³ | 6.923 (6.487, 6.923, 12.310) |
| 100,000 | 64³ | 87.420 (99.187, 75.282, 87.420) |
| 1,000,000 | 128³ | 819.859 (819.859, 796.137, 829.136) |

Log-log fit across four points: exponent 1.075, standard error 0.028, nominal 95% OLS interval [0.956, 1.194] (df=2; interval is descriptive with only four sizes and timing noise). Endpoint exponent is 1.072. Dense direct reference timings from above are 2.474 ms, 283.875 ms, 49,417.538 ms at N=1e3/1e4/1e5. Observed FFT timings are already lower at N=1e3, so observed crossover is at or below 1e3 for these non-equivalent kernels. This comparison is cost-only: accuracy gate fails, and FFT setup/interpolation correction/real-space work are not included. Do not infer a production crossover.

Fresh subprocess `/usr/bin/time -l` for all levels plus the small PCG diagnostic (including Julia startup) reported maximum resident set size 1,109,803,008 bytes and peak footprint 1,047,250,504 bytes. The 1e6 reciprocal-only measurement fits under the 4 GiB limit. This is not a complete PME memory estimate.

**[MEASURED, limited PCG diagnostic]** N=64 random periodic sites at density .8, 5.5% alpha coefficient of variation, selected alpha scale so diagonal dominates the measured spectral radius by factor 1/0.72. FFT-only dense matrix minimum eigenvalue `4.331265`; PCG converged in 11 iterations to true relative residual `6.038e-10`, dense-solve relative dipole difference `6.050e-10`. This checks a small dense solve against the same incomplete FFT operator only. It does not test direct-Ewald agreement, realistic chemical Thole coupling, or iteration growth with N. **[NOT MEASURED]** No hard-fixture large-N solve/iteration curve is available; do not extrapolate 11 iterations.

### Integration plan and source diff

**[ASSUMED / proposed stages, not production-ready]** A viable src integration would: (1) freeze units, damping, exclusions and boundary; (2) implement a consistent charge/dipole assignment window and deconvolution (NGP is too inaccurate); (3) add the matched Ewald real-space neighbor sum and analytic self and surface terms; (4) validate energy-field-operator derivatives and symmetry against the direct oracle; (5) test density-realistic PCG and iterations, forces, and memory in fresh processes; (6) only then implement an opt-in backend with explicit `:vacuum`/`:tin_foil` selection. No src unified diff is proposed: the current experiment fails the total-field accuracy gate and omits the real-space part. Applying a diff now would be unsafe. **[MEASURED verdict]** 1e5 and 1e6 reciprocal FFT work runs quickly and within RSS, but this spectral experiment does **not** establish acceptable full periodic polarization accuracy at either scale. Do not integrate.
