/-
Finite sums over groups of 3-vectors, and the exact centroid/residual
factorization of the complete-bipartite harmonic interaction.

Model: `U = (k/2) * sum_{i in A} sum_{j in B} ||q_i - q_j||^2`.

`pairSum_decomp` is the exact identity

    sum_{i,j} ||q_i - q_j||^2
      = n*m*||a-b||^2 + m*sum_i ||q_i - a||^2 + n*sum_j ||q_j - b||^2 ,

valid whenever `a`, `b` are the centroids, expressed here without
division as `sum_i (q_i - a) = 0` and `sum_j (q_j - b) = 0`.

So the O(n*m) pair count collapses to an O(n+m) reduction.  That is a
statement about the number of pair terms, not about the cost of a whole
simulation step: reading the N coordinates and writing the N forces is
still Omega(N), and `General.lean` shows what is lost for arbitrary
(non-centroid) reference points.
-/

import Init.Grind
import Foundation.Vec3

namespace Foundation

open V3

/-! ### Extra scalar-sum and vector-sum lemmas -/

theorem ssum_map_neg (l : List V3) (f : V3 → Rat) :
    ssum (l.map (fun x => -f x)) = -ssum (l.map f) := by
  induction l with
  | nil => simp only [List.map_nil, ssum_nil]; grind
  | cons x xs ih => simp only [List.map_cons, ssum_cons, ih]; grind

theorem ssum_map_sub (l : List V3) (f g : V3 → Rat) :
    ssum (l.map (fun x => f x - g x)) = ssum (l.map f) - ssum (l.map g) := by
  induction l with
  | nil => simp only [List.map_nil, ssum_nil]; grind
  | cons x xs ih => simp only [List.map_cons, ssum_cons, ih]; grind

/-- Split a four-term summand. -/
theorem ssum_map4 (l : List V3) (f1 f2 f3 f4 : V3 → Rat) :
    ssum (l.map (fun x => f1 x + f2 x + f3 x + f4 x)) =
      ssum (l.map f1) + ssum (l.map f2) + ssum (l.map f3) + ssum (l.map f4) := by
  induction l with
  | nil => simp only [List.map_nil, ssum_nil]; grind
  | cons x xs ih => simp only [List.map_cons, ssum_cons, ih]; grind

/-- Split a six-term summand. -/
theorem ssum_map6 (l : List V3) (f1 f2 f3 f4 f5 f6 : V3 → Rat) :
    ssum (l.map (fun x => f1 x + f2 x + f3 x + f4 x - f5 x - f6 x)) =
      ssum (l.map f1) + ssum (l.map f2) + ssum (l.map f3) + ssum (l.map f4)
        - ssum (l.map f5) - ssum (l.map f6) := by
  induction l with
  | nil => simp only [List.map_nil, ssum_nil]; grind
  | cons x xs ih => simp only [List.map_cons, ssum_cons, ih]; grind

theorem smul_sub (c : Rat) (u v : V3) : V3.smul c (V3.sub u v) = V3.sub (V3.smul c u) (V3.smul c v) := by
  apply V3.ext <;> simp only [V3.sub, V3.smul] <;> grind

theorem smul_neg' (c : Rat) (u : V3) : V3.smul c (V3.neg u) = V3.neg (V3.smul c u) := V3.smul_neg c u

theorem neg_zero' : V3.neg V3.zero = V3.zero := by
  apply V3.ext <;> simp only [V3.neg, V3.zero] <;> grind

theorem sub_eq_neg_sub (u v : V3) : V3.sub u v = V3.neg (V3.sub v u) := V3.neg_sub v u ▸ rfl

theorem sub_antisym (u v : V3) : V3.sub u v = V3.neg (V3.sub v u) := by
  rw [V3.neg_sub]

theorem vsum_map_neg (l : List V3) (f : V3 → V3) :
    vsum (l.map (fun x => V3.neg (f x))) = V3.neg (vsum (l.map f)) := by
  induction l with
  | nil => simp only [List.map_nil, vsum_nil, neg_zero']
  | cons x xs ih =>
    simp only [List.map_cons, vsum_cons, ih]
    apply V3.ext <;> simp only [V3.add, V3.neg] <;> grind

/-- The residual sums vanish exactly when `vsum l = n • a`. -/
theorem res_zero_of_centroid (l : List V3) (a : V3)
    (h : vsum l = V3.smul (l.length : Rat) a) :
    vsum (l.map (fun x => V3.sub x a)) = V3.zero := by
  rw [vsum_map_sub_const, h, V3.sub_self]

/-! ### The double sum -/

/-- `pairSum A B = sum_i sum_j ||q_i - q_j||^2`. -/
def pairSum (A B : List V3) : Rat :=
  ssum (A.map (fun q => ssum (B.map (fun r => V3.normSq (V3.sub q r)))))

/-- `resid A a = sum_i ||q_i - a||^2`. -/
def resid (A : List V3) (a : V3) : Rat :=
  ssum (A.map (fun q => V3.normSq (V3.sub q a)))

theorem ssum_map_congr (l : List V3) (f g : V3 → Rat) (h : ∀ x, f x = g x) :
    ssum (l.map f) = ssum (l.map g) := by
  induction l with
  | nil => rfl
  | cons x xs ih => simp only [List.map_cons, ssum_cons, h x, ih]

/-- Inner sum over `B`, expanded about the reference pair `(a, b)`. -/
theorem inner_decomp (B : List V3) (a b q : V3)
    (hB : vsum (B.map (fun r => V3.sub r b)) = V3.zero) :
    ssum (B.map (fun r => V3.normSq (V3.sub q r))) =
      (B.length : Rat) * V3.normSq (V3.sub a b)
        + (B.length : Rat) * V3.normSq (V3.sub q a)
        + resid B b
        + 2 * (B.length : Rat) * V3.dot (V3.sub a b) (V3.sub q a) := by
  rw [ssum_map_congr B
    (fun r => V3.normSq (V3.sub q r))
    (fun r => V3.normSq (V3.sub a b) + V3.normSq (V3.sub q a) + V3.normSq (V3.sub r b)
          + 2 * V3.dot (V3.sub a b) (V3.sub q a)
          - 2 * V3.dot (V3.sub a b) (V3.sub r b)
          - 2 * V3.dot (V3.sub q a) (V3.sub r b))
    (fun r => V3.pointwise_decomp q r a b)]
  rw [ssum_map6]
  -- the four surviving pieces
  have h1 : ssum (B.map (fun _ => V3.normSq (V3.sub a b))) =
      (B.length : Rat) * V3.normSq (V3.sub a b) :=
    ssum_map_const B (fun _ => V3.normSq (V3.sub a b)) _ (fun _ => rfl)
  have h2 : ssum (B.map (fun _ => V3.normSq (V3.sub q a))) =
      (B.length : Rat) * V3.normSq (V3.sub q a) :=
    ssum_map_const B (fun _ => V3.normSq (V3.sub q a)) _ (fun _ => rfl)
  have h4 : ssum (B.map (fun _ => 2 * V3.dot (V3.sub a b) (V3.sub q a))) =
      (B.length : Rat) * (2 * V3.dot (V3.sub a b) (V3.sub q a)) :=
    ssum_map_const B (fun _ => 2 * V3.dot (V3.sub a b) (V3.sub q a)) _ (fun _ => rfl)
  -- the two cross terms vanish
  have h5 : ssum (B.map (fun r => 2 * V3.dot (V3.sub a b) (V3.sub r b))) = 0 := by
    have hm : ssum (B.map (fun r => V3.dot (V3.sub a b) (V3.sub r b))) = 0 := by
      have hd : V3.dot (V3.sub a b) (vsum (B.map (fun r => V3.sub r b))) =
          ssum ((B.map (fun r => V3.sub r b)).map (fun v => V3.dot (V3.sub a b) v)) :=
        dot_vsum_right _ _
      rw [hB, V3.dot_zero_right] at hd
      rw [List.map_map] at hd
      exact hd.symm
    rw [ssum_map_mul_left 2 B (fun r => V3.dot (V3.sub a b) (V3.sub r b)), hm]
    grind
  have h6 : ssum (B.map (fun r => 2 * V3.dot (V3.sub q a) (V3.sub r b))) = 0 := by
    have hm : ssum (B.map (fun r => V3.dot (V3.sub q a) (V3.sub r b))) = 0 := by
      have hd : V3.dot (V3.sub q a) (vsum (B.map (fun r => V3.sub r b))) =
          ssum ((B.map (fun r => V3.sub r b)).map (fun v => V3.dot (V3.sub q a) v)) :=
        dot_vsum_right _ _
      rw [hB, V3.dot_zero_right] at hd
      rw [List.map_map] at hd
      exact hd.symm
    rw [ssum_map_mul_left 2 B (fun r => V3.dot (V3.sub q a) (V3.sub r b)), hm]
    grind
  simp only [resid, h1, h2, h4, h5, h6]
  grind

theorem pairSum_decomp (A B : List V3) (a b : V3)
    (hA : vsum (A.map (fun q => V3.sub q a)) = V3.zero)
    (hB : vsum (B.map (fun r => V3.sub r b)) = V3.zero) :
    pairSum A B =
      (A.length : Rat) * (B.length : Rat) * V3.normSq (V3.sub a b)
        + (B.length : Rat) * resid A a
        + (A.length : Rat) * resid B b := by
  rw [pairSum]
  rw [ssum_map_congr A
    (fun q => ssum (B.map (fun r => V3.normSq (V3.sub q r))))
    (fun q => (B.length : Rat) * V3.normSq (V3.sub a b)
        + (B.length : Rat) * V3.normSq (V3.sub q a)
        + resid B b
        + 2 * (B.length : Rat) * V3.dot (V3.sub a b) (V3.sub q a))
    (fun q => inner_decomp B a b q hB)]
  rw [ssum_map4]
  -- constant term
  have h1 : ssum (A.map (fun _ => (B.length : Rat) * V3.normSq (V3.sub a b))) =
      (A.length : Rat) * ((B.length : Rat) * V3.normSq (V3.sub a b)) :=
    ssum_map_const A _ _ (fun _ => rfl)
  have h2 : ssum (A.map (fun q => (B.length : Rat) * V3.normSq (V3.sub q a))) =
      (B.length : Rat) * resid A a := by
    rw [ssum_map_mul_left]
    rfl
  have h3 : ssum (A.map (fun _ => resid B b)) =
      (A.length : Rat) * resid B b :=
    ssum_map_const A _ _ (fun _ => rfl)
  have h4 : ssum (A.map (fun q => 2 * (B.length : Rat) * V3.dot (V3.sub a b) (V3.sub q a))) = 0 := by
    have hm : ssum (A.map (fun q => V3.dot (V3.sub a b) (V3.sub q a))) = 0 := by
      have hd : V3.dot (V3.sub a b) (vsum (A.map (fun q => V3.sub q a))) =
          ssum ((A.map (fun q => V3.sub q a)).map (fun v => V3.dot (V3.sub a b) v)) :=
        dot_vsum_right _ _
      rw [hA, V3.dot_zero_right] at hd
      rw [List.map_map] at hd
      exact hd.symm
    have hc : ssum (A.map (fun q => 2 * (B.length : Rat) * V3.dot (V3.sub a b) (V3.sub q a)))
        = (2 * (B.length : Rat)) * ssum (A.map (fun q => V3.dot (V3.sub a b) (V3.sub q a))) :=
      ssum_map_mul_left _ _ _
    rw [hc, hm]
    grind
  rw [h1, h2, h3, h4]
  grind

end Foundation
