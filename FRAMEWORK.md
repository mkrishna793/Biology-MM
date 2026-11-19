# Framework design target

This document records a narrow molecular-simulation implementation and its
design targets. Measured results are recorded in `validation/STATUS.md`;
they are short benchmarks, not production performance claims. Proved statements
are cited by theorem name from `lean/`; everything else is marked as
target, hypothesis, or classical estimate. The distinction is kept throughout.

Labels used below:

- **[PROVED]** a machine-checked theorem here, exact rational arithmetic.
- **[TARGET]** a design goal, not yet built.
- **[HYPOTHESIS]** a claim we believe but have not established.
- **[PROPOSITION, proof sketched]** a statement with an argument, not formalized.

## 1. Scope and non-goals

This framework specifies, in typed and testable form, the compositional parts of
a molecular-dynamics core: how state is represented, how energy factors over
interacting subsets of atoms, how forces derive from those factors, and when group
summaries are exact versus only error-bounded approximations.

It is **not** a claim of sublinear simulation. Reading N atoms, writing N forces,
and updating N positions and velocities is Ω(N) and no algebraic reorganization
avoids it. Only pair and field evaluations can be reduced; per-atom work cannot.
This recurs wherever a reduction appears: the usual failure mode is moving Θ(N)
work out of the accounting.

It is **not** a replacement for OpenMM. OpenMM is mature, fast, and validated.
This work specifies interfaces and invariants a backend must satisfy, and may
eventually inform a kernel; replacing a backend would require benchmarking on
real systems, and no such benchmark exists.

It is also **not** a general theory of physics. Exactly one interaction model has
been proved: a complete-bipartite quadratic harmonic interaction over finite,
nonperiodic groups in exact rational arithmetic. It is useful as the cleanest
case of an exact group summary, so it calibrates every later approximation, but
it is one model.

## 2. State and configuration space

A molecular state is a typed record, not five loose arrays. The README's scope
corrections bind: box vectors, constraints, exclusions, virtual sites, masses,
parameters, integrator state, and precision metadata all affect behavior.

Minimal state for N atoms:

| Field | Type | Notes |
| --- | --- | --- |
| positions `q` | `R^{3N}` or `(T^3)^N` | torus when periodic |
| momenta `p` | `R^{3N}` | physical momenta for MD |
| masses `m` | `R^N_{>0}` | define the kinetic metric |
| charges `z` | `R^N` | with an explicit unit system and prefactor |
| box `H` | `GL(3)` upper-triangular | triclinic allowed |
| constraints | index set + target values | defines a submanifold |
| parameters | typed per-factor records | not a flat blob |

**Periodic case.** A periodic configuration lives in `(T^3)^N`, with
`T^3 = R^3 / (L Z^3)`. A displacement `q_i - q_j` needs a lift, and minimum
image picks one by fiat, not canonically. In particular, **subtracting centroids
is not well-defined on a torus**: the centroid depends on representative choices,
and shifting one atom by a lattice vector moves it by `L/N` while changing nothing
physical. Any "let `a` be the centroid" statement is about `R^{3N}`, not
`(T^3)^N`. Hence the proved result here is nonperiodic, and a torus version needs
different invariants (§11), not a transcription.

**Constraints.** Holonomic constraints `σ(q) = 0` define a submanifold `M ⊂
R^{3N}`. A force from the unconstrained potential generally has a normal
component; only its projection onto `T_q M` is physical. With `A = ∂σ/∂q` the
projected force is `P F` with `P = I − A^T (A M^{-1} A^T)^{-1} A M^{-1}`, and
multipliers solve `A M^{-1} A^T λ = A M^{-1} F`. Projection and update remain
Θ(N). Constraint forces do no virtual work: `F_c = A^T λ` is normal to `M`, so
`<F_c, v> = 0` for every `v ∈ T_q M`.

## 3. Interaction structure

The potential is a sum over **factors**. There are three kinds:

1. **Fixed molecular graph** — bonds, angles, dihedrals, impropers, fixed
   exclusions. Static topology, compiled once.
2. **Dynamic spatial graph** — nonbonded pairs within a cutoff. This is a
   geometric predicate over positions, recomputed as atoms move.
3. **Hyperedges** — many-body factors that are not decomposable into pairs
   without loss (polarizable sites, three-body terms, coarse-grained beads).

Formally this is a factor graph between atoms and factors. Each factor `α` has a
**read-support** `S_α ⊆ {1..N}` and a local energy `U_α : R^{3|S_α|} → R`; the
total is a sum of pullbacks:

```
U(q) = Σ_α U_α( q|_{S_α} )
```

**Support propagation.** The gradient has support exactly where the factors are:

```
supp(∂U/∂q_i) ⊆ ⋃ { S_α : i ∈ S_α }
```

so `∂U/∂q_i` sums over factors reading atom `i`, each nonzero only on `S_α`.
This is where compositionality is literally true: it fixes the sparsity of the
force array and Hessian before any numerics. A factor reading `k` atoms
contributes a `3k × 3k` Hessian block. This is a structural fact, not a cost
reduction.

## 4. Differentiation: what is actually generic

**Cartesian Differential Categories as a design target [TARGET].** A CDC supplies
a derivative combinator `D` with a chain rule, linearity in its second argument,
and compatibility with products. Modeled abstractly, this would give a
correct-by-construction chain rule, so composed energies differentiate without
hand-written adjoints; backend-independent derivative code; and formal guarantees
that the two agree.

**What a CDC does not buy is speed.** A CDC is a *law about equality*: it
constrains which expressions denote the same derivative. It says nothing about
operation counts, memory traffic, or the constant factors that dominate real
kernels. A category-theoretic chain rule can make a derivative correct and
composable; it cannot make it cheap. Any claim that a CDC "reduces complexity"
is a category error, and this document makes no such claim.

**The concrete facts.** For a scalar energy `U` with evaluation cost `C_U`:

- **Force = one reverse-mode sweep (VJP).** `F = −∇U` costs `O(C_U)`, typically
  2–4× a forward evaluation but the same asymptotic order. This is the workhorse
  of the design.
- **Hessian-vector product = JVP of the VJP.** `H v = ∇(q ↦ <∇U(q), v>)`, cost
  `O(C_U)`, with no `N × N` object formed.
- **Dense Jacobian.** `∂F/∂q` explicitly costs `Θ(N)` VJP sweeps and `Θ(N^2)`
  memory; it is not affordable on realistic systems. Use matrix-free HVP.
- **Symmetry.** For conservative `C^2` potentials `∂F/∂q = −Hess U` is symmetric,
  which halves work in symmetric-aware factorizations and is a cheap correctness
  check in tests.

| Primitive | Direction | Cost | Output size |
| --- | --- | --- | --- |
| JVP `(∂U/∂q) v` | forward | `O(C_U)` | 1 scalar |
| VJP `(∂U/∂q)^T w` | reverse | `O(C_U)` | `3N` vector |
| HVP `H v` | forward-over-reverse | `O(C_U)` | `3N` vector |
| dense Jacobian | `N` sweeps | `Θ(N · C_U)` | `Θ(N^2)` |

The table's asymmetry is the practical argument for reverse mode: one VJP yields
the entire force field an integrator needs each step.

## 5. Exact factorization vs approximation

This is the heart of the document; the distinction it draws must never be
blurred.

### 5.1 The exact condition

Let a pair kernel be `f(x, y)` on `R^3 × R^3`. A group summary is a fixed-size
record `(σ_1, …, σ_r)` from one pass over a group, from which the group–group
interaction is recovered exactly.

**[PROPOSITION, proof sketched]** `f` admits an exact summary of size `r` per
group, with `O(rN)` work and `O(r)` memory per group, **iff** `f` has finite rank:

```
f(x, y) = Σ_{α=1}^{r} φ_α(x) ψ_α(y)
```

*Sketch.* Given that form,
`Σ_{i∈A} Σ_{j∈B} f(x_i, y_j) = Σ_α (Σ_i φ_α(x_i)) (Σ_j ψ_α(y_j))`: accumulate
`r` moments per group, combine in `O(r^2)`. Conversely, recoverability from `r`
statistics for *all* configurations forces the kernel into the span of `r`
products; no bookkeeping beats the rank of the kernel.

For **degree-`d` polynomial kernels in the pair displacement**, moments up to
order `d` suffice: the monomials of degree `≤ d` in `x` and in `y` form a finite
basis, and the kernel is a finite sum of their products. The number of moments
grows like `O(d^3)` in three dimensions and is independent of `N`.

This says nothing about whether the physical potential has finite rank; Coulomb
and Lennard-Jones do not.

### 5.2 The one case we have proved

Quadratic bipartite harmonic `U = (k/2) Σ_{i∈A} Σ_{j∈B} ||q_i − q_j||^2` has
finite rank; its summary is centroid plus residual. **[PROVED]**:

- `Foundation.pairSum_decomp` (`Foundation/Harmonic.lean`): with centroids `a`,
  `b` (division-free: `Σ_i (q_i − a) = 0`),
  `pairSum A B = |A||B| ||a − b||^2 + |B| resid A a + |A| resid B b`.
- `Foundation.forceOnA_eq` (`Foundation/Forces.lean`): `F_i = −k|B| (q_i − b)`,
  `O(1)` per atom given `b`.
- `Foundation.total_force_zero` (`Foundation/Forces.lean`): the two groups' total
  forces are equal and opposite.
- `Foundation.harmEnergy_pert` (`Foundation/Consistency.lean`): the exact identity
  `U(q+tv) = U(q) − t<F, v> + t² U(v)`, so `F = −∇U` algebraically, not by finite
  differences.

Crucially, `F_i` is still a *distinct* force for every atom. The summary collapses
the summation, not the output; the per-atom write is Ω(N) and stands.

### 5.3 What does not factorize

| Potential | Exact finite-rank summary? | Practical method | Exactness |
| --- | --- | --- | --- |
| Bipartite quadratic harmonic | Yes: 1 centroid + 1 residual scalar per group | centroid + residual closed form | **[PROVED]** exact |
| Coulomb `1/r`, nonperiodic | No | multipole / FMM / tree code | approximation, error `O((R/d)^{L+1})` |
| Coulomb `1/r`, periodic | No | Ewald / PME | approximation; minimum image is invalid |
| Lennard-Jones 12-6 | No | neighbor list + cutoff | approximation; no exact summary |
| Harmonic bond, `r_0 ≠ 0` | No | explicit pair loop | exact pair evaluation only |
| Harmonic bond, `r_0 = 0` | Yes | reduces to quadratic case | exact |
| Any cutoff potential | No — the pair set is a predicate | neighbor list | exact given the set, no moment summary |

**Coulomb.** `1/r` is not finite rank. FMM is a design target, not an implemented
backend in this tree. Multipole/FMM are tolerance-controlled: for
source radius `R`, separation `d`, and order `L`, the standard error estimate is
`O((R/d)^{L+1})` (classical, not proved here). FMM lowers far-field cost to
`O(N log N)` or `O(N)` but does not make the potential finite rank; it makes the
error controllable. Periodic sums are conditionally convergent and
order-dependent, so minimum image is ill-posed; Ewald/PME splits into real- and
reciprocal-space parts at `O(N log N)`. The scalar-charge PME implementation in
`julia/src/PME.jl` uses a cardinal B-spline mesh and FFT for reciprocal energy,
with analytic derivatives of the discrete mesh energy. Its algorithmic cost is
`O(N + M log M)` for mesh size `M`; no measured crossover or production scaling
claim is made until benchmark and OpenMM validation gates pass.

**Lennard-Jones.** `12-6` is non-polynomial in distance, with no finite exact
summary. The experiment's negative control applies the centroid bookkeeping to LJ
and fails to reproduce the pair energy, relative error of order 1 — the numerical
counterpart of the rank argument.

**Harmonic bond with `r_0 ≠ 0`.** `(k/2)(r − r_0)^2` with `r = ||x_i − x_j||` is
not quadratic in coordinates, because of the square root. Only `r_0 = 0` collapses
to the quadratic case: a "harmonic" bond is quadratic in *scalar separation*, not
in positions.

**Cutoffs.** The pair set within a cutoff is a geometric predicate on positions,
not a moment. No finite summary reproduces it for all configurations; neighbor
lists compute it directly.

### 5.4 The honest template for approximation

`Foundation.pairSum_decomp_general` (`Foundation/General.lean`) is **[PROVED]**
and states the decomposition for *arbitrary* reference points `a`, `b`:

```
pairSum A B = |A||B| ||a − b||^2 + |B| resid A a + |A| resid B b
              + 2|B| <a − b, SA> − 2|A| <a − b, SB> − 2<SA, SB>
```

where `SA = Σ_i (q_i − a)` and `SB = Σ_j (r_j − b)` are the residual sums. When
`a`, `b` are centroids, `SA = SB = 0` and the three correction terms vanish,
recovering the compact form exactly.

This is the template for every approximation here: an exact identity plus a named
correction whose magnitude can be bounded. Dropping a hypothesis costs exactly the
correction you can measure. "Approximate with certified error" keeps the identity
and bounds the correction; "exact factorization" means the correction is
*identically* zero. The two are never conflated.

## 6. Summaries, adaptivity, and multiscale

A group summary is a renormalization step: partition atoms into groups, summarize
each, evaluate group–group interactions, then **expand** back to per-atom forces.
Expansion is essential: without it atoms in a group would share a force, which is
wrong; with it every atom keeps its distinct force while only the group–group
evaluation is summarized. Expansion is Θ(N) per step and not optional.

**Adaptive partitioning [TARGET].** Groups should follow spatial clustering, not
a static atom index. As atoms diffuse the partition must be rebuilt, a
neighbor-search-shaped problem with `Θ(N)` amortized cost.

**Singularities.** Near `r → 0`, summaries degrade and error estimates blow up.
The partitioner must detect close approaches and fall back to direct near-field
pair evaluation — a correctness requirement, not an optimization.

**Cutoff switching.** A truncated potential must switch off smoothly. If the
switching function is only `C^0`, the force — its derivative — is discontinuous at
the cutoff. It must be at least `C^1`, and the force must be the exact derivative
of the switched energy, switching term included; inconsistency produces silent
drift. `harmEnergy_pert` is the model for such a consistency proof.

**Backreaction.** Summaries must react back on the atoms that produced them. A
summary computed and then discarded is a diagnostic, not a force field; expansion
implements the backreaction.

**Conservation.** If the grouped field is a true gradient, momentum is conserved
by the same argument as `total_force_zero`: internal forces are pairwise equal and
opposite in sum. If it is not a gradient — e.g. from an inconsistent
approximation — conservation is not automatic and must be checked separately.

**Bounded error [TARGET].** A useful guarantee is an a posteriori estimate: given
a configuration and partition, produce `ε` with `|U − Ũ| ≤ ε` and a bound on
`|F − F̃|`. A certified bound of this kind is a research target, not a current
result; none exists here.

## 7. Algorithms and cost model

Far field via tree codes or FMM: `O(N log N)` or `O(N)`; periodic electrostatics
via PME: `O(N log N)`; near field exact. Memory is
`O(N + interactions + summaries)`, explicitly **not** an `N × N` pair matrix.

| Operation | Cost | Reducible? |
| --- | --- | --- |
| Read `N` positions | `Θ(N)` | No |
| Write `N` forces | `Θ(N)` | No |
| Update `N` positions and velocities | `Θ(N)` | No |
| All-pairs energy/force | `Θ(N^2)` | Yes, to `O(N log N)` / `O(N)` far field |
| Neighbor search | `Θ(N)` amortized with cell lists | Only relative to `N^2` |
| Dense Hessian | `Θ(N^2)` memory | No; use matrix-free HVP |
| Reconstruct per-atom forces from summaries | `Ω(N)` | No |

The far-field reduction requires structure: finite-rank or tolerance-controlled
expansion. Whether that structure wins at realistic sizes is a [HYPOTHESIS]. It
does not remove the `Ω(N)` reconstruction of per-atom forces or the per-step state
read/write: a step that is `O(N)` in particle count is still `Ω(N)`; the win is in
the constant and pair-count exponent, never the per-atom floor.

## 8. Dynamics, constraints, thermostats

**Integrators.** Velocity Verlet is symplectic and time-reversible for separable
Hamiltonians. It conserves a *shadow* Hamiltonian to `O(dt^2)`, so energy does not
drift secularly but oscillates within that band; in chaotic regimes it random-walks
inside the band. Exact energy conservation is false.

**Constraints.** SHAKE, RATTLE, and LINCS are distinct algorithms with different
assumptions. SHAKE iterates positions; RATTLE also corrects velocities and is the
reversible, symplectic-on-the-manifold variant a geometric integrator needs; LINCS
is a linearized iterative scheme with different accuracy and stability. They must
be specified and verified separately, as the README requires. Constraint forces do
no virtual work (§2).

What is implemented here is narrower: disjoint-pair RATTLE inside velocity Verlet,
and a coupled Newton projection in the mass metric for rigid-molecule triangles
(`Constraints.jl`). The coupled projection is compared against OpenMM's constraint
machinery on displaced TIP3P water triangles; both land on the same constraint
surface (targets satisfied, matching bond lengths, H-O-H angle and mass-weighted
RMSD), but the raw coordinates differ by a rigid rotation of up to `0.17°` about the
centre of mass. That is a surface-level, not iterate-level, comparison: no
specialised SETTLE/SHAKE/LINCS kernel is claimed. The constraint virial is
*defined* by the static multipliers, and the identity
`-dU/ds|_{s=1} = W_ext + W_c` along an affine-scale-then-reproject path is verified
against finite differences to `2.64e-10` relative. That calibrates the
constraint-virial bookkeeping; it is not a validated pressure, because no
force-field factor exposes a virial (see §11 and `validation/STATUS.md`).

**Thermostats.** Langevin dynamics is a separate stochastic layer with its own
invariant measure, not another deterministic force. Use a splitting scheme such as
BAOAB, which preserves the configurational distribution to higher order; naive
Euler–Maruyama does not, and noise injected into an unspecified Verlet update has
no defined invariant measure.

## 9. HMC for equilibrium sampling

**Two momenta, never conflated.** MD uses the physical momentum `p` from the
state. HMC introduces an *auxiliary* momentum, resampled from `N(0, M)` at the
start of each trajectory, and integrates a fictitious Hamiltonian. These are
different objects, and mixing them in code or prose is a recurring error.

HMC step: resample `p ~ N(0, M)`, integrate the fictitious Hamiltonian with a
symplectic integrator for `L` steps, then accept with

```
α = min(1, exp(−β ΔH)),   ΔH = H(q', p') − H(q, p)
```

using the **full** Hamiltonian, kinetic plus potential. Acceptance is
distributional: the chain satisfies detailed balance with respect to `exp(−β U)`.
The trajectory is not a physical trajectory; it is auxiliary dynamics chosen for
sampling efficiency. **HMC does not generate exact physical trajectories, and its
exactness is distributional, not dynamical.**

Requirements: reversibility and volume preservation of the proposal. Symplecticity
is *sufficient* for good acceptance and for volume preservation, but *not
necessary* for exactness; any reversible, volume-preserving integrator gives a
valid sampler.

**Approximate potentials.** If the proposal uses `Ũ` but the accept step uses the
true `U`, the sampler stays exact for `exp(−βU)`: Metropolis absorbs the model
error at the cost of a lower acceptance rate. If the accept step itself uses `Ũ`,
the chain targets `exp(−βŨ)`, which is **biased** — it samples the wrong
distribution, and no trajectory length fixes it.

**Delayed acceptance** is rigorous only with a proven bound `|U − Ũ| ≤ ε`, giving
a lower bound on acceptance probability; without one the savings are unjustified.

**The central tension.** The full-`H` correction can erase the savings from a
cheap approximate force. If `Ũ` lowers force cost by a factor `c` but raises `ΔH`
enough to cut acceptance by more than `c`, the effective sampling rate is *worse*.
§12 treats this as a falsification criterion. Collective-variable proposals need
their own Jacobian and reversibility treatment; a proposal in a lower-dimensional
space is not automatically volume-preserving in the full space.

## 10. Proved results in this repository

All are in `lean/`, exact rational arithmetic, with no `sorry`,
`axiom`, or `admit`. `verify.sh` audits this: it prints the axiom dependency of
each theorem and fails if `sorryAx` appears.

| Theorem | File | Statement | Establishes | Does not establish |
| --- | --- | --- | --- | --- |
| `Foundation.pairSum_decomp` | `Foundation/Harmonic.lean` | `Σ‖q_i−q_j‖² = n m ‖a−b‖² + m·resid(A,a) + n·resid(B,b)` for centroids `a`,`b` | exact bipartite harmonic decomposition | anything about non-quadratic or periodic kernels |
| `Foundation.forceOnA_eq` | `Foundation/Forces.lean` | `F_i = −k m (q_i − b)` | exact per-atom force from a summary | that forces are cheap to write |
| `Foundation.forceOnA_injective` | `Foundation/Forces.lean` | `k m ≠ 0 → F_i = F_j → q_i = q_j` | the summary does not merge distinct atoms' forces | anything about numerics near `k m = 0` |
| `Foundation.total_force_zero` | `Foundation/Forces.lean` | `totalForceA + totalForceB = 0` | internal force cancellation | momentum conservation under constraints or approximations |
| `Foundation.harmEnergy_pert` | `Foundation/Consistency.lean` | `U(q+tv) = U(q) − t<F,v> + t² U(v)` | `F = −∇U` exactly, as an identity | that the identity survives a cutoff or a switch |
| `Foundation.pairSum_decomp_general` | `Foundation/General.lean` | decomposition with three residual-sum corrections | cost of dropping the centroid hypothesis | a bound on those corrections for a physical system |

`Foundation.pairSum_decomp_from_general` re-derives the compact form from the
general one, so the dependency is machine-checked rather than asserted:
the compact form is exact *because* `SA = SB = 0`, not by coincidence.

`Foundation/Witness.lean` guards against vacuity. A theorem with unsatisfiable
hypotheses proves nothing, so it exhibits concrete rational groups satisfying
the centroid hypotheses and discharges them, applying `pairSum_decomp` and
`forceOnA_eq` to produce genuine equations on a real finite configuration.
(Concrete numerals are not evaluated there: this bare Lean core has no
`norm_num`, and `decide` cannot reduce `Rat` numeral equality. Non-vacuity is
established structurally, which is what matters.)

Scope: one quadratic all-to-all model, exact rational arithmetic, finite
nonperiodic groups. A calibration point, not a force field; it fixes the meaning
of "exact factorization" for the rest of the document.

## 11. Open problems and next milestones

In dependency order.

1. **Torus version.** The centroid is undefined on `(T^3)^N`. A periodic analogue
   needs invariants equivariant under lattice translation — for example sums of
   periodic functions, or a reciprocal-space decomposition. What replaces `resid`
   is not obvious; this is a mathematical question, not a port.
2. **Error-bounded multipole.** A certified a posteriori bound on `|U − Ũ|` and
   `|F − F̃|` for a multipole truncation, with runtime-evaluable constants.
3. **A `C^1` cutoff with proved force/energy consistency,** in the style of
   `harmEnergy_pert`: switched energy and analytic gradient must agree
   identically, switching term included.
4. **Delayed-acceptance HMC with a real bound.** Requires item 2 as input.
5. **Jacobian-corrected constrained proposals.** A proposal moving on the
   constraint manifold with the correct volume element.
6. **Benchmarking against OpenMM on a real system.** No benchmark exists; until
   one does, no performance claim in this document is empirical.
7. **Per-factor virial and interacting-system NPT.** The ideal-gas barostat kernel
   is validated against the exact `Gamma(N+1, 1/(βP))` volume law, but every
   interacting model is refused because the factors do not expose `W = Σ r·F`.
   Exposing a consistent virial for bonded, LJ, and PME terms — and showing that
   the pressure estimator covers every factor in the energy — is prerequisite to
   any non-ideal NPT claim. The constraint-virial identity is a partial step only.
8. **A converged molecular trajectory.** The registered `WaterBox` composition
   has numerical checks for composed forces, angular momentum, reversibility,
   second-order rigid drift and kinetic temperature. The old 146.59 K run used
   a dissipative projection map; the current principal-axis Hamiltonian splitting
   fixes that failure. The PME split now makes real-space cutoff energy jumps
   negligible (`alpha*rcut=4.5`). A separate COM mass-denominator bug suppressed
   the old diffusion estimate by `3.17e-5` at 200 waters. Synthetic estimator
   checks pass, but physical bulk observables require longer equilibrated runs,
   replication, and finite-size analysis. See `validation/STATUS.md` for actual
   completed trajectory duration and measured limits.

## 12. What would falsify this program

The program should be narrowed or abandoned if any of the following holds.

**The constant dominates.** If the far-field constant is large enough that
realistic system sizes see no wall-clock win over a well-tuned direct kernel, the
asymptotic advantage is academic. Asymptotics are not performance, and a crossover
beyond accessible `N` is a failure.

**Certified bounds are too loose.** If the a posteriori bounds are so pessimistic
that they force prohibitively small groups or large truncation orders, the
certified-approximation route produces no usable speedup. A bound that is valid
but vacuous is a negative result.

**The acceptance rate collapses.** If approximate-potential HMC loses more
sampling efficiency through `ΔH` than it gains through cheaper forces, the
full-`H` correction has erased the benefit. This is the sharpest test because it
is internal: it can be measured without reference to any competitor.

**The exact case does not generalize.** If the only exactly factorizable potential
is the quadratic one, and every physical term needs tolerance-controlled
approximation with expensive certification, then "exact factorization" is a single
worked example rather than a design principle. That would not make it false, but
it would make it much less valuable.

**The structural claims are already known.** If the factor-graph and
support-propagation view adds nothing over standard neighbor-list and exclusion
machinery in production codes, then §3 is exposition rather than contribution.
This is a real risk and should be assessed honestly against existing
implementations.

**No backend ever benefits.** If the interfaces prove unimplementable at
production speed — because typed state, constraint projection, or summary
expansion costs more than the algebraic savings — the framework remains a
specification exercise. That is a legitimate outcome, but it should be stated as
such, not dressed up as a speedup.
