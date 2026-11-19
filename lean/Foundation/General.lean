/-
The general centroid-free decomposition, and the exact role of the
centroid hypothesis.

`pairSum_decomp_general` holds for ARBITRARY reference points `a`, `b`
and carries three correction terms driven by the residual sums
`sum_i (q_i - a)` and `sum_j (r_j - b)`.

`pairSum_decomp_from_general` specialises it: setting those sums to zero
kills all three corrections and recovers the compact form of
`Harmonic.lean`.  Keeping both statements makes the dependency explicit --
the compact form is exact BECAUSE `a`, `b` are centroids, not because the
algebra happens to be approximately true.
-/

import Init.Grind
import Foundation.Vec3
import Foundation.Harmonic
import Foundation.Forces

namespace Foundation

open V3

/-! ### The general identity, without centroid hypotheses -/

/-- `sum_x <u, f x> = <u, sum_x f x>`. -/
theorem ssum_dot_vsum (l : List V3) (u : V3) (f : V3 → V3) :
    ssum (l.map (fun x => V3.dot u (f x))) = V3.dot u (vsum (l.map f)) := by
  induction l with
  | nil => simp only [List.map_nil, vsum_nil, ssum_nil, V3.dot_zero_right]
  | cons x xs ih =>
    simp only [List.map_cons, vsum_cons, ssum_cons, ih]
    rw [V3.dot_add_right]

/-- `sum_x <f x, u> = <sum_x f x, u>`. -/
theorem ssum_vsum_dot (l : List V3) (u : V3) (f : V3 → V3) :
    ssum (l.map (fun x => V3.dot (f x) u)) = V3.dot (vsum (l.map f)) u := by
  induction l with
  | nil => simp only [List.map_nil, vsum_nil, ssum_nil, V3.dot_zero_left]
  | cons x xs ih =>
    simp only [List.map_cons, vsum_cons, ssum_cons, ih]
    rw [V3.dot_add_left]

/-- Inner sum over `B` expanded about references `(a, b)`, keeping the
residual-term corrections. -/
theorem inner_decomp_general (B : List V3) (a b q : V3) :
    ssum (B.map (fun r => V3.normSq (V3.sub q r))) =
      (B.length : Rat) * V3.normSq (V3.sub a b)
        + (B.length : Rat) * V3.normSq (V3.sub q a)
        + resid B b
        + 2 * (B.length : Rat) * V3.dot (V3.sub a b) (V3.sub q a)
        - 2 * V3.dot (V3.sub a b) (vsum (B.map (fun r => V3.sub r b)))
        - 2 * V3.dot (V3.sub q a) (vsum (B.map (fun r => V3.sub r b))) := by
  rw [ssum_map_congr B
    (fun r => V3.normSq (V3.sub q r))
    (fun r => V3.normSq (V3.sub a b) + V3.normSq (V3.sub q a) + V3.normSq (V3.sub r b)
          + 2 * V3.dot (V3.sub a b) (V3.sub q a)
          - 2 * V3.dot (V3.sub a b) (V3.sub r b)
          - 2 * V3.dot (V3.sub q a) (V3.sub r b))
    (fun r => V3.pointwise_decomp q r a b)]
  rw [ssum_map6]
  have h1 : ssum (B.map (fun _ => V3.normSq (V3.sub a b))) =
      (B.length : Rat) * V3.normSq (V3.sub a b) :=
    ssum_map_const B _ _ (fun _ => rfl)
  have h2 : ssum (B.map (fun _ => V3.normSq (V3.sub q a))) =
      (B.length : Rat) * V3.normSq (V3.sub q a) :=
    ssum_map_const B _ _ (fun _ => rfl)
  have h4 : ssum (B.map (fun _ => 2 * V3.dot (V3.sub a b) (V3.sub q a))) =
      (B.length : Rat) * (2 * V3.dot (V3.sub a b) (V3.sub q a)) :=
    ssum_map_const B _ _ (fun _ => rfl)
  have h5 : ssum (B.map (fun r => 2 * V3.dot (V3.sub a b) (V3.sub r b))) =
      2 * V3.dot (V3.sub a b) (vsum (B.map (fun r => V3.sub r b))) := by
    rw [ssum_map_mul_left, ssum_dot_vsum]
  have h6 : ssum (B.map (fun r => 2 * V3.dot (V3.sub q a) (V3.sub r b))) =
      2 * V3.dot (V3.sub q a) (vsum (B.map (fun r => V3.sub r b))) := by
    rw [ssum_map_mul_left, ssum_dot_vsum]
  simp only [resid, h1, h2, h4, h5, h6]
  grind

/-- **General decomposition.**  For arbitrary reference points `a`, `b` the
bipartite sum carries three correction terms driven by the residual sums
`SA = sum_i (q_i - a)` and `SB = sum_j (r_j - b)`.  Setting `SA = SB = 0`
(equivalently, `a` and `b` the centroids) kills all three and recovers
`pairSum_decomp`. -/
theorem pairSum_decomp_general (A B : List V3) (a b : V3) :
    pairSum A B =
      (A.length : Rat) * (B.length : Rat) * V3.normSq (V3.sub a b)
        + (B.length : Rat) * resid A a
        + (A.length : Rat) * resid B b
        + 2 * (B.length : Rat) * V3.dot (V3.sub a b) (vsum (A.map (fun q => V3.sub q a)))
        - 2 * (A.length : Rat) * V3.dot (V3.sub a b) (vsum (B.map (fun r => V3.sub r b)))
        - 2 * V3.dot (vsum (A.map (fun q => V3.sub q a))) (vsum (B.map (fun r => V3.sub r b))) := by
  rw [pairSum]
  rw [ssum_map_congr A
    (fun q => ssum (B.map (fun r => V3.normSq (V3.sub q r))))
    (fun q => (B.length : Rat) * V3.normSq (V3.sub a b)
        + (B.length : Rat) * V3.normSq (V3.sub q a)
        + resid B b
        + 2 * (B.length : Rat) * V3.dot (V3.sub a b) (V3.sub q a)
        - 2 * V3.dot (V3.sub a b) (vsum (B.map (fun r => V3.sub r b)))
        - 2 * V3.dot (V3.sub q a) (vsum (B.map (fun r => V3.sub r b))))
    (fun q => inner_decomp_general B a b q)]
  rw [ssum_map6]
  have h1 : ssum (A.map (fun _ => (B.length : Rat) * V3.normSq (V3.sub a b))) =
      (A.length : Rat) * ((B.length : Rat) * V3.normSq (V3.sub a b)) :=
    ssum_map_const A _ _ (fun _ => rfl)
  have h2 : ssum (A.map (fun q => (B.length : Rat) * V3.normSq (V3.sub q a))) =
      (B.length : Rat) * resid A a := by
    rw [ssum_map_mul_left]; rfl
  have h3 : ssum (A.map (fun _ => resid B b)) = (A.length : Rat) * resid B b :=
    ssum_map_const A _ _ (fun _ => rfl)
  have h4 : ssum (A.map (fun q => 2 * (B.length : Rat) * V3.dot (V3.sub a b) (V3.sub q a))) =
      2 * (B.length : Rat) * V3.dot (V3.sub a b) (vsum (A.map (fun q => V3.sub q a))) := by
    rw [ssum_map_mul_left, ssum_dot_vsum]
  have h5 : ssum (A.map (fun _ => 2 * V3.dot (V3.sub a b) (vsum (B.map (fun r => V3.sub r b))))) =
      (A.length : Rat) * (2 * V3.dot (V3.sub a b) (vsum (B.map (fun r => V3.sub r b)))) :=
    ssum_map_const A _ _ (fun _ => rfl)
  have h6 : ssum (A.map (fun q => 2 * V3.dot (V3.sub q a) (vsum (B.map (fun r => V3.sub r b))))) =
      2 * V3.dot (vsum (A.map (fun q => V3.sub q a))) (vsum (B.map (fun r => V3.sub r b))) := by
    rw [ssum_map_mul_left, ssum_vsum_dot]
  simp only [h1, h2, h3, h4, h5, h6]
  grind

/-- **Corollary.**  Under the centroid hypotheses the three correction terms
vanish and the compact exact decomposition is recovered.  This re-derives
`pairSum_decomp` from the general identity, exhibiting exactly what the
centroid hypothesis buys. -/
theorem pairSum_decomp_from_general (A B : List V3) (a b : V3)
    (hA : vsum (A.map (fun q => V3.sub q a)) = V3.zero)
    (hB : vsum (B.map (fun r => V3.sub r b)) = V3.zero) :
    pairSum A B =
      (A.length : Rat) * (B.length : Rat) * V3.normSq (V3.sub a b)
        + (B.length : Rat) * resid A a
        + (A.length : Rat) * resid B b := by
  rw [pairSum_decomp_general A B a b, hA, hB, V3.dot_zero_right, V3.dot_zero_left]
  grind

end Foundation
