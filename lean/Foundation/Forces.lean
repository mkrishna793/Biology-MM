/-
Per-atom forces of the bipartite harmonic interaction, and the exact
cancellation of the total internal force.

The main result is `forceOnA_eq`: the O(n*m) force accumulation
`F_i = -k * sum_j (q_i - q_j)` collapses to `-k*m*(q_i - b)`, an
O(1) expression per atom given the centroid `b`.  Note that each atom
still gets its OWN force -- the summary collapses the summation, not
the per-atom output.  `total_force_zero` then shows the two groups'
total forces are equal and opposite.

This is a statement about exact rational arithmetic, not about
floating-point roundoff.
-/

import Init.Grind
import Foundation.Vec3
import Foundation.Harmonic

namespace Foundation

open V3

/-! ### Extra vector-sum lemmas needed for forces -/

theorem vsum_map_congr (l : List V3) (f g : V3 → V3) (h : ∀ x, f x = g x) :
    vsum (l.map f) = vsum (l.map g) := by
  induction l with
  | nil => rfl
  | cons x xs ih => simp only [List.map_cons, vsum_cons, h x, ih]

theorem sub_zero' (u : V3) : V3.sub u V3.zero = u := by
  apply V3.ext <;> simp only [V3.sub, V3.zero] <;> grind

theorem zero_sub (u : V3) : V3.sub V3.zero u = V3.neg u := by
  apply V3.ext <;> simp only [V3.sub, V3.neg, V3.zero] <;> grind

theorem neg_sub' (u v : V3) : V3.neg (V3.sub u v) = V3.sub v u := V3.neg_sub u v

theorem add_smul (c d : Rat) (u : V3) :
    V3.add (V3.smul c u) (V3.smul d u) = V3.smul (c + d) u := by
  apply V3.ext <;> simp only [V3.add, V3.smul] <;> grind

theorem neg_smul_eq (c : Rat) (u : V3) : V3.neg (V3.smul c u) = V3.smul (-c) u := by
  apply V3.ext <;> simp only [V3.neg, V3.smul] <;> grind

/-- `q - r = (q - b) - (r - b)`. -/
theorem sub_sub_sub (q r b : V3) :
    V3.sub q r = V3.sub (V3.sub q b) (V3.sub r b) := by
  apply V3.ext <;> simp only [V3.sub] <;> grind

/-- Componentwise linearity of `vsum` under subtraction. -/
theorem vsum_map_sub (l : List V3) (f g : V3 → V3) :
    vsum (l.map (fun x => V3.sub (f x) (g x))) =
      V3.sub (vsum (l.map f)) (vsum (l.map g)) := by
  induction l with
  | nil => simp only [List.map_nil, vsum_nil, V3.sub_self]
  | cons x xs ih =>
    simp only [List.map_cons, vsum_cons, ih]
    apply V3.ext <;> simp only [V3.add, V3.sub] <;> grind

/-! ### Force accumulation -/

/-- The un-scaled force accumulation on a site `q` against group `B`:
`sum_{r in B} (q - r)`. -/
def forceSum (B : List V3) (q : V3) : V3 :=
  vsum (B.map (fun r => V3.sub q r))

/-- `sum_{r in B} (q - r) = m * (q - b)` when `b` is the centroid of `B`. -/
theorem forceSum_eq (B : List V3) (q b : V3)
    (hB : vsum (B.map (fun r => V3.sub r b)) = V3.zero) :
    forceSum B q = V3.smul (B.length : Rat) (V3.sub q b) := by
  have hcongr : vsum (B.map (fun r => V3.sub q r))
      = vsum (B.map (fun r => V3.sub (V3.sub q b) (V3.sub r b))) :=
    vsum_map_congr B (fun r => V3.sub q r)
      (fun r => V3.sub (V3.sub q b) (V3.sub r b)) (fun r => sub_sub_sub q r b)
  rw [forceSum, hcongr, vsum_map_sub, vsum_map_const, hB, sub_zero']

/-- The force on an atom of group `A` with stiffness `k`, from group `B`. -/
def forceOnA (B : List V3) (k : Rat) (q : V3) : V3 :=
  V3.smul (-k) (forceSum B q)

/-- Exact per-atom force: `F_i = -k*m*(q_i - b)`. -/
theorem forceOnA_eq (B : List V3) (k : Rat) (q b : V3)
    (hB : vsum (B.map (fun r => V3.sub r b)) = V3.zero) :
    forceOnA B k q = V3.smul (-(k * (B.length : Rat))) (V3.sub q b) := by
  rw [forceOnA, forceSum_eq B q b hB, V3.smul_smul]
  congr 1
  grind

/-! ### Total force vanishes -/

/-- Sum of the forces on all of group `A` against group `B`. -/
def totalForceA (A B : List V3) (k : Rat) : V3 :=
  vsum (A.map (fun q => forceOnA B k q))

/-- Sum of the forces on all of group `B` against group `A`. -/
def totalForceB (A B : List V3) (k : Rat) : V3 :=
  vsum (B.map (fun r => forceOnA A k r))

/-- Sum over `A` of `q - b` is `n*(a - b)`. -/
theorem vsum_sub_ref (A : List V3) (a b : V3)
    (hA : vsum (A.map (fun q => V3.sub q a)) = V3.zero) :
    vsum (A.map (fun q => V3.sub q b)) = V3.smul (A.length : Rat) (V3.sub a b) := by
  have hcongr : vsum (A.map (fun q => V3.sub q b))
      = vsum (A.map (fun q => V3.add (V3.sub q a) (V3.sub a b))) :=
    vsum_map_congr A (fun q => V3.sub q b)
      (fun q => V3.add (V3.sub q a) (V3.sub a b)) (fun q => V3.sub_repr q a b)
  rw [hcongr, vsum_map_add, hA, V3.zero_add, vsum_map_const]

theorem totalForceA_eq (A B : List V3) (a b : V3) (k : Rat)
    (hA : vsum (A.map (fun q => V3.sub q a)) = V3.zero)
    (hB : vsum (B.map (fun r => V3.sub r b)) = V3.zero) :
    totalForceA A B k =
      V3.smul (-(k * (A.length : Rat) * (B.length : Rat))) (V3.sub a b) := by
  have hcongr : vsum (A.map (fun q => forceOnA B k q))
      = vsum (A.map (fun q => V3.smul (-(k * (B.length : Rat))) (V3.sub q b))) :=
    vsum_map_congr A (fun q => forceOnA B k q)
      (fun q => V3.smul (-(k * (B.length : Rat))) (V3.sub q b))
      (fun q => forceOnA_eq B k q b hB)
  rw [totalForceA, hcongr, vsum_map_smul, vsum_sub_ref A a b hA, V3.smul_smul]
  congr 1
  grind

theorem totalForceB_eq (A B : List V3) (a b : V3) (k : Rat)
    (hA : vsum (A.map (fun q => V3.sub q a)) = V3.zero)
    (hB : vsum (B.map (fun r => V3.sub r b)) = V3.zero) :
    totalForceB A B k =
      V3.smul (k * (A.length : Rat) * (B.length : Rat)) (V3.sub a b) := by
  have hcongr : vsum (B.map (fun r => forceOnA A k r))
      = vsum (B.map (fun r => V3.smul (-(k * (A.length : Rat))) (V3.sub r a))) :=
    vsum_map_congr B (fun r => forceOnA A k r)
      (fun r => V3.smul (-(k * (A.length : Rat))) (V3.sub r a))
      (fun r => forceOnA_eq A k r a hA)
  rw [totalForceB, hcongr, vsum_map_smul, vsum_sub_ref B b a hB, V3.smul_smul]
  rw [show V3.sub b a = V3.neg (V3.sub a b) from (V3.neg_sub a b).symm, V3.smul_neg]
  rw [neg_smul_eq]
  congr 1
  grind

/-- Total internal force vanishes. -/
theorem total_force_zero (A B : List V3) (a b : V3) (k : Rat)
    (hA : vsum (A.map (fun q => V3.sub q a)) = V3.zero)
    (hB : vsum (B.map (fun r => V3.sub r b)) = V3.zero) :
    V3.add (totalForceA A B k) (totalForceB A B k) = V3.zero := by
  rw [totalForceA_eq A B a b k hA hB, totalForceB_eq A B a b k hA hB]
  rw [add_smul]
  rw [show -(k * (A.length : Rat) * (B.length : Rat))
        + k * (A.length : Rat) * (B.length : Rat) = 0 by grind]
  rw [V3.zero_smul]

/-! ### The summary preserves every atom's distinct force -/

/-- Scalar cancellation in the rationals. -/
theorem mul_left_cancel_rat (c u v : Rat) (hc : c ≠ 0) (h : c * u = c * v) : u = v := by
  have := congrArg (fun x => c⁻¹ * x) h
  simpa only [← Rat.mul_assoc, Rat.inv_mul_cancel c hc, Rat.one_mul] using this

/-- **Distinct outputs are preserved.**  When `k*m != 0`, two atoms of group `A`
with the same computed force necessarily have the same position.  Contrapositive:
atoms at different positions receive different forces.  So replacing the
`O(n*m)` pair accumulation by the `O(1)`-per-atom expression does not merge,
average, or lose any atom's distinct force. -/
theorem forceOnA_injective (B : List V3) (k : Rat) (b : V3)
    (hB : vsum (B.map (fun r => V3.sub r b)) = V3.zero)
    (hk : k * (B.length : Rat) ≠ 0) :
    ∀ q₁ q₂ : V3, forceOnA B k q₁ = forceOnA B k q₂ → q₁ = q₂ := by
  intro q₁ q₂ h
  rw [forceOnA_eq B k q₁ b hB, forceOnA_eq B k q₂ b hB] at h
  have hc : (-(k * (B.length : Rat))) ≠ 0 := by
    intro hz
    apply hk
    grind
  have hsub : V3.sub q₁ b = V3.sub q₂ b := by
    apply V3.ext
    · exact mul_left_cancel_rat _ _ _ hc (congrArg (fun v : V3 => v.x) h)
    · exact mul_left_cancel_rat _ _ _ hc (congrArg (fun v : V3 => v.y) h)
    · exact mul_left_cancel_rat _ _ _ hc (congrArg (fun v : V3 => v.z) h)
  calc q₁ = V3.add (V3.sub q₁ b) b := (V3.add_sub_cancel q₁ b).symm
    _ = V3.add (V3.sub q₂ b) b := by rw [hsub]
    _ = q₂ := V3.add_sub_cancel q₂ b

end Foundation
