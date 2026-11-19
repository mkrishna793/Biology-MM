/-
Exact energy/force consistency for the bipartite harmonic interaction.

`harmEnergy_pert` is the identity

    U(q + t*v) = U(q) - t*(sum_i <F_i, v_i> + sum_j <F_j, w_j>) + t^2*U(v)

for a simultaneous displacement of every site, where `F` is the per-atom
force of `Forces.lean` and `U(v)` is the same harmonic form evaluated on
the displacement fields.  The linear coefficient is exactly the force
pairing, so `F = -grad U` holds as an algebraic identity in `Rat`; the
quadratic coefficient is the curvature term.

There is no limit, no finite difference and no truncation here.  This is
the formal counterpart of the numerical finite-difference check in
`experiment/`, and it is strictly stronger: an identity, not
a measured agreement.
-/

import Init.Grind
import Foundation.Vec3
import Foundation.Harmonic
import Foundation.Forces

namespace Foundation

open V3

/-! ### Helpers -/

theorem ssum_map_map (l : List V3) (g : V3 → V3) (F : V3 → Rat) :
    ssum ((l.map g).map F) = ssum (l.map (fun q => F (g q))) := by
  rw [List.map_map]; rfl

theorem ssum_map_zero (l : List V3) (f : V3 → Rat) (h : ∀ x, f x = 0) :
    ssum (l.map f) = 0 := by
  rw [ssum_map_const l f 0 h]
  grind

theorem ssum_map3 (l : List V3) (f1 f2 f3 : V3 → Rat) :
    ssum (l.map (fun x => f1 x + f2 x + f3 x)) =
      ssum (l.map f1) + ssum (l.map f2) + ssum (l.map f3) := by
  induction l with
  | nil => simp only [List.map_nil, ssum_nil]; grind
  | cons x xs ih => simp only [List.map_cons, ssum_cons, ih]; grind

/-- Interchange of the two finite sums over groups `A` and `B`. -/
theorem ssum_swap (A B : List V3) (f : V3 → V3 → Rat) :
    ssum (A.map (fun q => ssum (B.map (fun r => f q r)))) =
      ssum (B.map (fun r => ssum (A.map (fun q => f q r)))) := by
  induction A with
  | nil =>
    simp only [List.map_nil, ssum_nil]
    exact (ssum_map_zero B (fun r => 0) (fun r => rfl)).symm
  | cons q qs ih =>
    simp only [List.map_cons, ssum_cons]
    rw [ssum_map_add, ih]

theorem dot_smul_left' (c : Rat) (u v : V3) : V3.dot (V3.smul c u) v = c * V3.dot u v := by
  cases u; cases v; simp only [V3.dot, V3.smul]; grind

theorem dot_sub_right' (u v w : V3) : V3.dot u (V3.sub v w) = V3.dot u v - V3.dot u w := by
  cases u; cases v; cases w; simp only [V3.dot, V3.sub]; grind

theorem dot_neg_left' (u v : V3) : V3.dot (V3.neg u) v = -V3.dot u v := by
  cases u; cases v; simp only [V3.dot, V3.neg]; grind

theorem sub_swap' (q r : V3) : V3.sub q r = V3.neg (V3.sub r q) :=
  (V3.neg_sub r q).symm

/-- `sum_{q in A} (q - r) = -forceSum A r`. -/
theorem vsum_sub_ref_neg (A : List V3) (r : V3) :
    vsum (A.map (fun q => V3.sub q r)) = V3.neg (forceSum A r) := by
  have hc : vsum (A.map (fun q => V3.sub q r))
      = vsum (A.map (fun q => V3.neg (V3.sub r q))) :=
    vsum_map_congr A (fun q => V3.sub q r) (fun q => V3.neg (V3.sub r q))
      (fun q => sub_swap' q r)
  rw [hc, vsum_map_neg]
  rfl

/-- Pointwise exact second-order expansion of a perturbed squared distance. -/
theorem normSq_pert (q r v w : V3) (t : Rat) :
    V3.normSq (V3.sub (V3.add q (V3.smul t v)) (V3.add r (V3.smul t w)))
      = V3.normSq (V3.sub q r)
        + 2 * t * V3.dot (V3.sub q r) (V3.sub v w)
        + t * t * V3.normSq (V3.sub v w) := by
  cases q; cases r; cases v; cases w
  simp only [V3.normSq, V3.dot, V3.sub, V3.add, V3.smul]
  grind

theorem inner_pert (B : List V3) (q v : V3) (vB : V3 → V3) (t : Rat) :
    ssum (B.map (fun r =>
        V3.normSq (V3.sub (V3.add q (V3.smul t v)) (V3.add r (V3.smul t (vB r))))))
      = ssum (B.map (fun r => V3.normSq (V3.sub q r)))
        + 2 * t * ssum (B.map (fun r => V3.dot (V3.sub q r) (V3.sub v (vB r))))
        + t * t * ssum (B.map (fun r => V3.normSq (V3.sub v (vB r)))) := by
  rw [ssum_map_congr B _ _ (fun r => normSq_pert q r v (vB r) t)]
  rw [ssum_map3]
  rw [ssum_map_mul_left (2 * t) B (fun r => V3.dot (V3.sub q r) (V3.sub v (vB r)))]
  rw [ssum_map_mul_left (t * t) B (fun r => V3.normSq (V3.sub v (vB r)))]

/-- The quadratic form of the displacement fields. -/
def pairSumF (A B : List V3) (vA vB : V3 → V3) : Rat :=
  ssum (A.map (fun q => ssum (B.map (fun r => V3.normSq (V3.sub (vA q) (vB r))))))

/-- Bridge: `pairSum` of mapped groups is the explicit double sum. -/
theorem pairSum_map (A B : List V3) (f g : V3 → V3) :
    pairSum (A.map f) (B.map g) =
      ssum (A.map (fun q => ssum (B.map (fun r => V3.normSq (V3.sub (f q) (g r)))))) := by
  rw [pairSum, ssum_map_map]
  apply ssum_map_congr
  intro q
  rw [ssum_map_map]

/-- Splitting the cross term into the two force sums. -/
theorem cross_decomp (A B : List V3) (vA vB : V3 → V3) :
    ssum (A.map (fun q => ssum (B.map (fun r => V3.dot (V3.sub q r) (V3.sub (vA q) (vB r))))))
      = ssum (A.map (fun q => V3.dot (forceSum B q) (vA q)))
        + ssum (B.map (fun r => V3.dot (forceSum A r) (vB r))) := by
  -- split every inner dot into its two halves
  have hsplit : ssum (A.map (fun q => ssum (B.map (fun r => V3.dot (V3.sub q r) (V3.sub (vA q) (vB r))))))
      = ssum (A.map (fun q => ssum (B.map (fun r => V3.dot (V3.sub q r) (vA q)))))
        - ssum (A.map (fun q => ssum (B.map (fun r => V3.dot (V3.sub q r) (vB r))))) := by
    rw [ssum_map_congr A
      (fun q => ssum (B.map (fun r => V3.dot (V3.sub q r) (V3.sub (vA q) (vB r)))))
      (fun q => ssum (B.map (fun r => V3.dot (V3.sub q r) (vA q)))
              - ssum (B.map (fun r => V3.dot (V3.sub q r) (vB r))))]
    · exact ssum_map_sub A _ _
    · intro q
      rw [ssum_map_congr B
        (fun r => V3.dot (V3.sub q r) (V3.sub (vA q) (vB r)))
        (fun r => V3.dot (V3.sub q r) (vA q) - V3.dot (V3.sub q r) (vB r))
        (fun r => dot_sub_right' _ _ _)]
      exact ssum_map_sub B _ _
  have hA : ssum (A.map (fun q => ssum (B.map (fun r => V3.dot (V3.sub q r) (vA q)))))
      = ssum (A.map (fun q => V3.dot (forceSum B q) (vA q))) := by
    apply ssum_map_congr A _
    intro q
    have hd := dot_vsum_left (B.map (fun r => V3.sub q r)) (vA q)
    rw [List.map_map] at hd
    exact hd.symm
  have hB : ssum (A.map (fun q => ssum (B.map (fun r => V3.dot (V3.sub q r) (vB r)))))
      = -ssum (B.map (fun r => V3.dot (forceSum A r) (vB r))) := by
    rw [ssum_swap A B (fun q r => V3.dot (V3.sub q r) (vB r))]
    have hneg : ssum (B.map (fun r => ssum (A.map (fun q => V3.dot (V3.sub q r) (vB r)))))
        = ssum (B.map (fun r => -V3.dot (forceSum A r) (vB r))) := by
      apply ssum_map_congr B _
      intro r
      have hd := dot_vsum_left (A.map (fun q => V3.sub q r)) (vB r)
      rw [List.map_map] at hd
      rw [vsum_sub_ref_neg] at hd
      rw [dot_neg_left'] at hd
      exact hd.symm
    rw [hneg, ssum_map_neg]
  rw [hsplit, hA, hB]
  grind

/-- Exact perturbation identity for the unweighted pair sum. -/
theorem pairSum_pert (A B : List V3) (vA vB : V3 → V3) (t : Rat) :
    pairSum (A.map (fun q => V3.add q (V3.smul t (vA q))))
            (B.map (fun r => V3.add r (V3.smul t (vB r))))
      = pairSum A B
        + 2 * t * (ssum (A.map (fun q => V3.dot (forceSum B q) (vA q)))
                 + ssum (B.map (fun r => V3.dot (forceSum A r) (vB r))))
        + t * t * pairSumF A B vA vB := by
  rw [pairSum_map]
  rw [ssum_map_congr A
    (fun q => ssum (B.map (fun r =>
      V3.normSq (V3.sub (V3.add q (V3.smul t (vA q)))
                       (V3.add r (V3.smul t (vB r)))))))
    (fun q => ssum (B.map (fun r => V3.normSq (V3.sub q r)))
            + 2 * t * ssum (B.map (fun r => V3.dot (V3.sub q r) (V3.sub (vA q) (vB r))))
            + t * t * ssum (B.map (fun r => V3.normSq (V3.sub (vA q) (vB r)))))
    (fun q => inner_pert B q (vA q) vB t)]
  rw [ssum_map3]
  rw [ssum_map_mul_left (2 * t) A
        (fun q => ssum (B.map (fun r => V3.dot (V3.sub q r) (V3.sub (vA q) (vB r)))))]
  rw [ssum_map_mul_left (t * t) A
        (fun q => ssum (B.map (fun r => V3.normSq (V3.sub (vA q) (vB r)))))]
  rw [cross_decomp, pairSumF, ← pairSum]

/-- The bipartite harmonic energy `(k/2) * sum_{i,j} ||q_i - q_j||^2`. -/
def harmEnergy (A B : List V3) (k : Rat) : Rat := (k / 2) * pairSum A B

theorem dot_forceOnA (B : List V3) (k : Rat) (q v : V3) :
    V3.dot (forceOnA B k q) v = -(k * V3.dot (forceSum B q) v) := by
  rw [forceOnA, dot_smul_left']
  grind

/-- **Exact energy/force consistency.**  Displacing site `i` by `t * vA q_i` and
site `j` by `t * vB r_j` changes the energy by exactly

  `-t * (sum_i <F_i, v_i> + sum_j <F'_j, w_j>) + t^2 * U(v)`,

where `F_i = forceOnA` and `F'_j = forceOnA A k r_j` are the per-atom forces.
The linear coefficient is exactly the force pairing, so `F` is the negative
gradient of `U`; the quadratic coefficient is the same harmonic form evaluated
on the displacement fields.  This is an identity in `Q` -- no limit, no finite
difference, no approximation. -/
theorem harmEnergy_pert (A B : List V3) (vA vB : V3 → V3) (k t : Rat) :
    harmEnergy (A.map (fun q => V3.add q (V3.smul t (vA q))))
               (B.map (fun r => V3.add r (V3.smul t (vB r)))) k
      = harmEnergy A B k
        - t * (ssum (A.map (fun q => V3.dot (forceOnA B k q) (vA q)))
             + ssum (B.map (fun r => V3.dot (forceOnA A k r) (vB r))))
        + t * t * harmEnergy (A.map vA) (B.map vB) k := by
  rw [harmEnergy, harmEnergy, harmEnergy]
  rw [pairSum_pert]
  rw [show pairSumF A B vA vB = pairSum (A.map vA) (B.map vB) from
    (pairSum_map A B vA vB).symm]
  have hA : ssum (A.map (fun q => V3.dot (forceOnA B k q) (vA q)))
      = -(k * ssum (A.map (fun q => V3.dot (forceSum B q) (vA q)))) := by
    rw [ssum_map_congr A _ _ (fun q => dot_forceOnA B k q (vA q))]
    rw [ssum_map_congr A
      (fun q => -(k * V3.dot (forceSum B q) (vA q)))
      (fun q => (-k) * V3.dot (forceSum B q) (vA q)) (fun q => by grind)]
    rw [ssum_map_mul_left (-k) A (fun q => V3.dot (forceSum B q) (vA q))]
    grind
  have hB : ssum (B.map (fun r => V3.dot (forceOnA A k r) (vB r)))
      = -(k * ssum (B.map (fun r => V3.dot (forceSum A r) (vB r)))) := by
    rw [ssum_map_congr B _ _ (fun r => dot_forceOnA A k r (vB r))]
    rw [ssum_map_congr B
      (fun r => -(k * V3.dot (forceSum A r) (vB r)))
      (fun r => (-k) * V3.dot (forceSum A r) (vB r)) (fun r => by grind)]
    rw [ssum_map_mul_left (-k) B (fun r => V3.dot (forceSum A r) (vB r))]
    grind
  rw [hA, hB]
  grind

end Foundation
