/-
The exact shadow invariant of the velocity-Verlet map on a 1-D harmonic
oscillator, and the consequence that its energy error is bounded rather than
growing.

## Why this theorem exists

`FRAMEWORK.md` §8 asserts that velocity Verlet "conserves a *shadow*
Hamiltonian to `O(dt^2)`, so energy does not drift secularly but oscillates
within that band". Until now that was a classical claim with no machine-checked
counterpart in this repository. It is the load-bearing statement behind every
energy-diagnostic in `julia/src/Dynamics.jl`, so it is worth proving.

## The statement

For the oscillator `q'' = −ω² q` with step `h`, one velocity-Verlet step is the
linear map

    pʰ   = p − (h/2) ω² q
    q'   = q + h pʰ
    p'   = pʰ − (h/2) ω² q'

This is symplectic for any `h`, so it has an exactly invariant quadratic form.
The theorem below identifies it explicitly:

    H̃(q,p) = ½ p² + ½ ω² (1 − h²ω²/4) q²

is invariant under the map. `shadow_invariant` proves that identity in exact
rational arithmetic.

Two consequences are then derived, not asserted:

* `shadow_coeff_pos`: for `0 < ω²` and `0 < h²ω² < 4` (with `h²ω² < 4` the exact
  stability limit) the shadow coefficient is positive, so `H̃` is a
  positive-definite quadratic form and the orbit lies on an ellipse of constant
  `H̃`. A bounded orbit is exactly what "the energy error is bounded" means for
  this system.
* `verlet_energy_defect`: the true energy `E = ½p² + ½ω²q²` differs from `H̃` by
  `(h²ω⁴/8) q²`. Along a single orbit (`shadow_energy_band` / `energy_band_abs`),
  the energy oscillates within a band of width `O(h²)` proportional to the
  amplitude of `q²` — and the *shape* of that statement is what the numerics in
  `test_dynamics.jl` and `test_shadow.jl` confirm (`dt²` scaling, zero drift
  slope). It is an exact identity here, not an asymptotic estimate.

## What this does not prove

* Nothing about more than one degree of freedom. For a general separable
  Hamiltonian the shadow Hamiltonian exists only as a formal asymptotic series
  (and not as a closed form), so this is a one-oscillator result.
* Nothing about chaotic regimes, where the numerically observed energy does a
  bounded random walk rather than a clean oscillation.
* Nothing about the floating-point implementation: this is exact rational
  algebra, and the numerical counterpart is roundoff-limited.
-/

import Init.Grind
import Foundation.Vec3

namespace Foundation

open V3

/-- One velocity-Verlet step with step `h` and squared frequency `w2` applied to
the phase-space point `(q, p)`, returned as a pair. -/
def verletStep (h w2 q p : Rat) : Rat × Rat :=
  let ph := p - (h / 2) * w2 * q
  let q' := q + h * ph
  let p' := ph - (h / 2) * w2 * q'
  (q', p')

/-- The shadow quadratic form `H̃(q,p) = ½ p² + ½ D q²` with
`D = ω² (1 − h²ω²/4)`. -/
def shadowH (D q p : Rat) : Rat := p * p / 2 + D * q * q / 2

/-- The true oscillator energy `E(q,p) = ½ p² + ½ ω² q²`. -/
def harmH (w2 q p : Rat) : Rat := p * p / 2 + w2 * q * q / 2

/-- **The shadow invariant.**  With `D = ω²(1 − h²ω²/4)`, one velocity-Verlet
step leaves `H̃` exactly unchanged, for every step size `h` and every phase-space
point.  This is an identity in `Rat`: no limit and no truncation. -/
theorem shadow_invariant (h w2 q p : Rat) :
    shadowH (w2 * (1 - h * h * w2 / 4)) (verletStep h w2 q p).1 (verletStep h w2 q p).2
      = shadowH (w2 * (1 - h * h * w2 / 4)) q p := by
  unfold shadowH verletStep
  grind

/-- **The energy defect is an explicit `O(h²)` term.**  The true energy after one
step differs from the shadow form by exactly `(h²ω⁴/8) q'²`:

    E(q',p') = H̃(q',p') + (h² ω⁴ / 8) q'²

and since `H̃` is invariant, the energy at step `n` is
`E_n = H̃_0 + (h²ω⁴/8) q_n²`.  So the energy's *variation* over a trajectory is
bounded by `(h²ω⁴/8) · max q²`, which is the `O(h²)` band the design document
describes. -/
theorem verlet_energy_defect (h w2 q p : Rat) :
    harmH w2 (verletStep h w2 q p).1 (verletStep h w2 q p).2
      = shadowH (w2 * (1 - h * h * w2 / 4)) (verletStep h w2 q p).1 (verletStep h w2 q p).2
        + h * h * w2 * w2 / 8 * (verletStep h w2 q p).1 * (verletStep h w2 q p).1 := by
  unfold harmH shadowH verletStep
  grind

/-- Combining the two: the energy at the new point equals the *initial* shadow
value plus the explicit defect. This is the form used to bound the band, since
`H̃_0` is a constant along the trajectory. -/
theorem energy_is_shadow_plus_defect (h w2 q p : Rat) :
    harmH w2 (verletStep h w2 q p).1 (verletStep h w2 q p).2
      = shadowH (w2 * (1 - h * h * w2 / 4)) q p
        + h * h * w2 * w2 / 8 * (verletStep h w2 q p).1 * (verletStep h w2 q p).1 := by
  rw [verlet_energy_defect, shadow_invariant]

/-- `0 ≤ a²` for rationals, from the bare order primitives available in this
core. Needed because the ordered-field library here has no `sq_nonneg`. -/
theorem sq_nonneg_rat (a : Rat) : 0 ≤ a * a := by
  have ht : (0:Rat) ≤ a ∨ a ≤ 0 := Rat.le_total
  cases ht with
  | inl h => exact Rat.mul_nonneg h h
  | inr h =>
    have h2 : 0 ≤ -a := by grind
    have hm := Rat.mul_nonneg h2 h2
    grind

/-- The defect coefficient `h²ω⁴/8` is non-negative. -/
theorem hsq_w2_nonneg (h w2 : Rat) : 0 ≤ h * h * w2 * w2 / 8 := by
  have hh := sq_nonneg_rat h
  have hw := sq_nonneg_rat w2
  have h3 : 0 ≤ (h * h) * (w2 * w2) := Rat.mul_nonneg hh hw
  have h4 : 0 ≤ (h * h) * (w2 * w2) * ((8:Rat)⁻¹) :=
    Rat.mul_nonneg h3 (Rat.le_of_lt (by rw [← Rat.inv_pos]; grind))
  have heq : (h * h) * (w2 * w2) * ((8:Rat)⁻¹) = h * h * w2 * w2 / 8 := by grind
  rwa [heq] at h4

/-- The defect term is non-negative for every `q`. -/
theorem defect_nonneg (h w2 q : Rat) :
    0 ≤ h * h * w2 * w2 / 8 * q * q := by
  have hc := hsq_w2_nonneg h w2
  have hq := sq_nonneg_rat q
  -- rearrange to (h²ω⁴/8) * (q²) and apply mul_nonneg twice
  have hstep : h * h * w2 * w2 / 8 * q * q = (h * h * w2 * w2 / 8) * (q * q) := by grind
  rw [hstep]
  exact Rat.mul_nonneg hc hq

/-- **Positivity of the shadow coefficient.**  For `0 < ω²` and `0 < h²ω² < 4`
the shadow coefficient `ω²(1 − h²ω²/4)` is strictly positive, so `H̃` is positive
definite and its level sets are ellipses: the orbit is bounded.

The restriction `h²ω² < 4` is the standard stability limit of velocity Verlet on
an oscillator. Beyond it the map becomes hyperbolic and no bounded orbit exists,
which is a genuine restriction of the theorem, not an omission. -/
theorem shadow_coeff_pos (h2 w2 : Rat) (hw : 0 < w2) (_h4 : 0 < h2 * w2)
    (hlt : h2 * w2 < 4) : 0 < w2 * (1 - h2 * w2 / 4) := by
  -- 1 - h2*w2/4 = (4 - h2*w2)/4, and both factors are positive
  have hpos : 0 < (4 : Rat) - h2 * w2 := by grind
  have hq : 0 < (4 : Rat) := by grind
  have hinv : 0 < ((4 : Rat))⁻¹ := by
    exact (Rat.inv_pos).mpr (by grind)
  have hrew : w2 * (1 - h2 * w2 / 4) = w2 * ((4 - h2 * w2) * ((4:Rat))⁻¹) := by grind
  rw [hrew]
  exact Rat.mul_pos hw (Rat.mul_pos hpos hinv)

/-- **The energy band, correctly qualified.**  For two points *on the same Verlet
orbit* (equivalent to their shadow values being equal), the energies differ by
exactly the difference of their defects:

    E(q₁,p₁) − E(q₂,p₂) = (h²ω⁴/8)(q₁² − q₂²).

The same-orbit hypothesis is essential, not decorative. Without it the `p` terms
do **not** cancel, and an earlier draft of this theorem omitted it, which made the
statement false for two arbitrary points. Stating it with the hypothesis is the
honest form. -/
theorem shadow_energy_band (h w2 q1 p1 q2 p2 : Rat)
    (horbit : shadowH (w2 * (1 - h * h * w2 / 4)) q1 p1
                = shadowH (w2 * (1 - h * h * w2 / 4)) q2 p2) :
    harmH w2 q1 p1 - harmH w2 q2 p2
      = h * h * w2 * w2 / 8 * (q1 * q1 - q2 * q2) := by
  unfold harmH shadowH at *
  grind

/-- Corollary: for two points on the same orbit, the one-sided energy difference
`E(q₁,p₁) − E(q₂,p₂)` is bounded above by the sum of the two defect terms
`(h²ω⁴/8) q₁² + (h²ω⁴/8) q₂²` (and swapping `(q₁,p₁)` with `(q₂,p₂)` yields
the symmetric lower bound), so the band width is controlled by `q²`. -/
theorem energy_band_abs (h w2 q1 p1 q2 p2 : Rat)
    (horbit : shadowH (w2 * (1 - h * h * w2 / 4)) q1 p1
                = shadowH (w2 * (1 - h * h * w2 / 4)) q2 p2) :
    harmH w2 q1 p1 - harmH w2 q2 p2
      ≤ h * h * w2 * w2 / 8 * q1 * q1
        + h * h * w2 * w2 / 8 * q2 * q2 := by
  rw [shadow_energy_band h w2 q1 p1 q2 p2 horbit]
  have h1 : 0 ≤ h * h * w2 * w2 / 8 * q2 * q2 := defect_nonneg h w2 q2
  grind

/-- The practical single-step form: **the energy after one Verlet step is the
initial shadow value plus the defect at the new position.**  Since the shadow
value is invariant across steps (by induction over `n` applications of
`shadow_invariant`), this is what makes the orbit bound
`E_n − E_0 ≤ (h²ω⁴/8)(q_n² + q_0²)` (and symmetrically `|E_n − E_0|`) follow,
i.e. the energy stays in a band set by the trajectory's spatial amplitude and
`h²`. -/
theorem energy_after_step_eq (h w2 q p : Rat) :
    harmH w2 (verletStep h w2 q p).1 (verletStep h w2 q p).2
      = shadowH (w2 * (1 - h * h * w2 / 4)) q p
        + h * h * w2 * w2 / 8 * (verletStep h w2 q p).1 * (verletStep h w2 q p).1 :=
  energy_is_shadow_plus_defect h w2 q p

end Foundation
