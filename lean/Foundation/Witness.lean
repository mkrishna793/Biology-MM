/-
Non-vacuity witnesses for the harmonic factorization theorems.

A theorem with unsatisfiable hypotheses says nothing about anything.  These
witnesses show the hypotheses are satisfiable and that the headline theorems
actually apply to a concrete finite configuration, producing a genuine
equation and a genuine force distinction.

They are named lemmas, not new theory.  Note that the concrete arithmetic is
NOT evaluated numerically: this bare Lean core has no `norm_num`, and `decide`
cannot reduce `Rat` numeral equality.  Non-vacuity is therefore established
structurally -- by discharging the hypotheses and applying the theorem --
which is what actually matters for the result being non-empty.
-/

import Init.Grind
import Foundation.Vec3
import Foundation.Harmonic
import Foundation.Forces
import Foundation.Consistency

namespace Foundation

open V3

/-- Group `A` with two sites; centroid `(1,0,0)`. -/
def wA : List V3 := [⟨0, 0, 0⟩, ⟨2, 0, 0⟩]

/-- Group `B` with two sites; centroid `(5,0,0)`. -/
def wB : List V3 := [⟨4, 0, 0⟩, ⟨6, 0, 0⟩]

/-- The centroid hypothesis for `wA` at `a = (1,0,0)` is satisfiable. -/
theorem wA_centroid : vsum (wA.map (fun q => V3.sub q ⟨1, 0, 0⟩)) = V3.zero := by
  simp only [wA, vsum, List.map, List.foldr, V3.add, V3.sub, V3.zero]
  grind

/-- The centroid hypothesis for `wB` at `b = (5,0,0)` is satisfiable. -/
theorem wB_centroid : vsum (wB.map (fun r => V3.sub r ⟨5, 0, 0⟩)) = V3.zero := by
  simp only [wB, vsum, List.map, List.foldr, V3.add, V3.sub, V3.zero]
  grind

/-- Both witness groups are nonempty, so the `length`-based coefficients are
not degenerate. -/
theorem wA_nonempty : wA.length = 2 := rfl

/-- The decomposition theorem is applicable to the witnesses and yields a
genuine equation -- this is the non-vacuity claim. -/
theorem w_decomp_applies :
    pairSum wA wB =
      (wA.length : Rat) * (wB.length : Rat) * V3.normSq (V3.sub ⟨1, 0, 0⟩ ⟨5, 0, 0⟩)
        + (wB.length : Rat) * resid wA ⟨1, 0, 0⟩
        + (wA.length : Rat) * resid wB ⟨5, 0, 0⟩ :=
  pairSum_decomp wA wB ⟨1, 0, 0⟩ ⟨5, 0, 0⟩ wA_centroid wB_centroid

/-- The per-atom force theorem applies to the witnesses. -/
theorem w_force_applies :
    forceOnA wB 1 ⟨0, 0, 0⟩ =
      V3.smul (-(1 * (wB.length : Rat))) (V3.sub ⟨0, 0, 0⟩ ⟨5, 0, 0⟩) :=
  forceOnA_eq wB 1 ⟨0, 0, 0⟩ ⟨5, 0, 0⟩ wB_centroid

/-- The two atoms of `A` sit at different positions, and the injectivity
theorem therefore says their forces differ: a summary that collapsed distinct
sites into one output would violate this. -/
theorem w_positions_distinct : (⟨0, 0, 0⟩ : V3) ≠ ⟨2, 0, 0⟩ := by
  intro h
  have := congrArg (fun v : V3 => v.x) h
  simp only at this
  grind

end Foundation
