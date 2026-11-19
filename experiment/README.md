# Group-harmonic experiment

An algebraic demonstrator for one finite, nonperiodic, two-group **complete-bipartite
quadratic** interaction. Group A has `n` sites at `q_i`, group B has `m` sites at
`q_j`, all in R^3, with stiffness `k >= 0`:

    U = (k/2) * sum_{i in A} sum_{j in B} ||q_i - q_j||^2

With centroids `a = mean(A)`, `b = mean(B)` and residuals `da_i = q_i - a`,
`db_j = q_j - b`, the exact identity holds:

    U = (k/2) * [ n*m*||a - b||^2 + m*sum_i ||da_i||^2 + n*sum_j ||db_j||^2 ]

and the exact forces are

    F_i = -k*m*(q_i - b)   (i in A),      F_j = -k*n*(q_j - a)   (j in B),

so `sum_i F_i + sum_j F_j = 0` exactly.

## What is measured

* `direct_energy_forces`: a genuine O(n*m) enumeration over an `(n, m, 3)` pair
  array; it never forms a centroid.
* `summarized_energy_forces`: the O(n+m) centroid/residual closed form.
* Float comparison of energy and forces over several `(n, m)`, `k` and geometries
  (Gaussian, far-shifted, coincident centroids, highly stretched), reporting
  max absolute / relative differences and total-force residuals.
* `operation_counts`: counted algebraic work terms (pairs vs. summaries), not
  measured wall time.
* Central-difference check of the summarized gradient `dU/dx` against `-F`.
* Exact-rational (`fractions.Fraction`) cross-check of direct vs summarized with
  zero tolerance, including `k = 0` and `k = 7/3`.
* A negative control: the same bookkeeping applied to Lennard-Jones does not
  reproduce the pair energy.

## How this relates to the Lean proofs

The identity and the force formula above are not merely checked numerically.
They are proved in exact rational arithmetic in `lean/`:

| Experiment claim | Lean theorem |
| --- | --- |
| energy decomposition | `Foundation.pairSum_decomp` (`Foundation/Harmonic.lean`) |
| `F_i = -k*m*(q_i - b)` | `Foundation.forceOnA_eq` (`Foundation/Forces.lean`) |
| total force is zero | `Foundation.total_force_zero` (`Foundation/Forces.lean`) |
| `F = -grad U` | `Foundation.harmEnergy_pert` (`Foundation/Consistency.lean`) |
| what the centroid hypothesis buys | `Foundation.pairSum_decomp_general` (`Foundation/General.lean`) |

The numerical run and the proof are independent derivations of the same
statement: the proof is exact, while the experiment is bound by floating-point
roundoff and therefore also reports the exact-rational path, which agrees with
zero tolerance.

## How to run

```
python3 experiment/test_group_harmonic.py
python3 experiment/group_harmonic.py
```

Both commands exit 0 on success; the test script prints `ALL TESTS PASSED`.

## Scope

This is an algebraic demonstrator for **one quadratic all-to-all model**. It is
**not** a realistic Lennard-Jones, cutoff, or electrostatic force field, and the
closed form does not generalize to those (see the LJ negative control). The
O(n*m) -> O(n+m) reduction is purely in **pair evaluations**; all N atom state
reads/writes and the output force array remain Omega(N). Floating-point results
are roundoff-limited, while the rational path is exact.
