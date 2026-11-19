/-
Rational 3-vectors and the bilinear algebra used by the harmonic
factorization proofs.

Everything here is exact arithmetic in `Rat`.  A `V3` is a triple of
`Rat` coordinates; `dot`, `normSq`, `add`, `sub`, `smul` are componentwise.
The key lemma is `V3.pointwise_decomp`, the second-order expansion of
`||q - r||^2` about two reference points, which is what makes the
centroid decomposition exact rather than approximate.
-/

import Init.Grind

namespace Foundation

structure V3 where
  x : Rat
  y : Rat
  z : Rat

namespace V3

theorem ext {u v : V3} (hx : u.x = v.x) (hy : u.y = v.y) (hz : u.z = v.z) : u = v := by
  cases u; cases v; simp_all

def zero : V3 := ⟨0, 0, 0⟩
def dot (u v : V3) : Rat := u.x * v.x + u.y * v.y + u.z * v.z
def normSq (u : V3) : Rat := dot u u
def add (u v : V3) : V3 := ⟨u.x + v.x, u.y + v.y, u.z + v.z⟩
def neg (u : V3) : V3 := ⟨-u.x, -u.y, -u.z⟩
def sub (u v : V3) : V3 := ⟨u.x - v.x, u.y - v.y, u.z - v.z⟩
def smul (c : Rat) (u : V3) : V3 := ⟨c * u.x, c * u.y, c * u.z⟩

theorem dot_comm (u v : V3) : dot u v = dot v u := by
  cases u; cases v; simp only [dot]; grind

theorem dot_add_right (u v w : V3) : dot u (add v w) = dot u v + dot u w := by
  cases u; cases v; cases w; simp only [dot, add]; grind

theorem dot_add_left (u v w : V3) : dot (add u v) w = dot u w + dot v w := by
  cases u; cases v; cases w; simp only [dot, add]; grind

theorem dot_zero_right (u : V3) : dot u zero = 0 := by
  cases u; simp only [dot, zero]; grind

theorem dot_zero_left (u : V3) : dot zero u = 0 := by
  cases u; simp only [dot, zero]; grind

theorem normSq_sub (u v : V3) :
    normSq (sub u v) = normSq u - 2 * dot u v + normSq v := by
  cases u; cases v; simp only [normSq, dot, sub]; grind

theorem add_zero (u : V3) : add u zero = u := by
  cases u; simp only [add, zero]; grind

theorem zero_add (u : V3) : add zero u = u := by
  cases u; simp only [add, zero]; grind

theorem add_neg (u : V3) : add u (neg u) = zero := by
  apply ext <;> simp only [add, neg, zero] <;> grind

theorem sub_repr (q a b : V3) : sub q b = add (sub q a) (sub a b) := by
  cases q; cases a; cases b; simp only [sub, add]; grind

theorem neg_sub (u v : V3) : neg (sub u v) = sub v u := by
  apply ext <;> simp only [neg, sub] <;> grind

theorem add_comm4 (a b c d : V3) :
    add (add a b) (add c d) = add (add a c) (add b d) := by
  apply ext <;> simp only [add] <;> grind

theorem smul_add (c : Rat) (u v : V3) : smul c (add u v) = add (smul c u) (smul c v) := by
  apply ext <;> simp only [add, smul] <;> grind

theorem smul_smul (c d : Rat) (u : V3) : smul c (smul d u) = smul (c * d) u := by
  cases u; simp only [smul]; grind

theorem smul_zero (c : Rat) : smul c zero = zero := by
  apply ext <;> simp only [smul, zero] <;> grind

theorem smul_eq_zero_iff (u : V3) : smul 0 u = zero := by
  apply ext <;> simp only [smul, zero] <;> grind

theorem zero_smul (u : V3) : smul 0 u = zero := smul_eq_zero_iff u

theorem smul_neg (c : Rat) (u : V3) : smul c (neg u) = neg (smul c u) := by
  apply ext <;> simp only [smul, neg] <;> grind

/-- `(u - v) + v = u`. -/
theorem add_sub_cancel (u v : V3) : add (sub u v) v = u := by
  apply ext <;> simp only [add, sub] <;> grind

theorem sub_self (u : V3) : sub u u = zero := by
  apply ext <;> simp only [sub, zero] <;> grind

/-- Crux: expansion of `||q - r||^2` about reference points `a`, `b`. -/
theorem pointwise_decomp (q r a b : V3) :
    normSq (sub q r) =
      normSq (sub a b) + normSq (sub q a) + normSq (sub r b)
        + 2 * dot (sub a b) (sub q a)
        - 2 * dot (sub a b) (sub r b)
        - 2 * dot (sub q a) (sub r b) := by
  cases q; cases r; cases a; cases b
  simp only [normSq, dot, sub]; grind

end V3

open V3

def ssum (l : List Rat) : Rat := l.foldr (· + ·) 0
def vsum (l : List V3) : V3 := l.foldr V3.add V3.zero

@[simp] theorem ssum_nil : ssum [] = 0 := rfl
@[simp] theorem ssum_cons (c : Rat) (l : List Rat) : ssum (c :: l) = c + ssum l := rfl
@[simp] theorem vsum_nil : vsum [] = V3.zero := rfl
@[simp] theorem vsum_cons (u : V3) (l : List V3) : vsum (u :: l) = V3.add u (vsum l) := rfl

theorem ssum_map_add (l : List V3) (f g : V3 → Rat) :
    ssum (l.map (fun x => f x + g x)) = ssum (l.map f) + ssum (l.map g) := by
  induction l with
  | nil => simp only [List.map_nil, ssum_nil]; grind
  | cons x xs ih => simp only [List.map_cons, ssum_cons, ih]; grind

theorem ssum_map_mul_left (c : Rat) (l : List V3) (f : V3 → Rat) :
    ssum (l.map (fun x => c * f x)) = c * ssum (l.map f) := by
  induction l with
  | nil => simp only [List.map_nil, ssum_nil]; grind
  | cons x xs ih => simp only [List.map_cons, ssum_cons, ih]; grind

theorem ssum_map_const (l : List V3) (f : V3 → Rat) (c : Rat) (h : ∀ x, f x = c) :
    ssum (l.map f) = (l.length : Rat) * c := by
  induction l with
  | nil => simp only [List.map_nil, ssum_nil, List.length_nil]; grind
  | cons x xs ih =>
    simp only [List.map_cons, ssum_cons, ih, List.length_cons, h]
    grind

theorem vsum_map_add (l : List V3) (f g : V3 → V3) :
    vsum (l.map (fun x => V3.add (f x) (g x))) =
      V3.add (vsum (l.map f)) (vsum (l.map g)) := by
  induction l with
  | nil => simp only [List.map_nil, vsum_nil, V3.zero_add]
  | cons x xs ih =>
    simp only [List.map_cons, vsum_cons, ih]
    rw [V3.add_comm4]

theorem vsum_map_smul (c : Rat) (l : List V3) (f : V3 → V3) :
    vsum (l.map (fun x => V3.smul c (f x))) = V3.smul c (vsum (l.map f)) := by
  induction l with
  | nil => simp only [List.map_nil, vsum_nil, V3.smul_zero]
  | cons x xs ih =>
    simp only [List.map_cons, vsum_cons, ih]
    rw [V3.smul_add]

theorem vsum_map_const (l : List V3) (c : V3) :
    vsum (l.map (fun _ => c)) = V3.smul (l.length : Rat) c := by
  induction l with
  | nil => simp only [List.map_nil, vsum_nil, List.length_nil]; exact (V3.zero_smul c).symm
  | cons x xs ih =>
    simp only [List.map_cons, vsum_cons, ih, List.length_cons]
    apply V3.ext <;> simp only [V3.add, V3.smul] <;> grind

theorem vsum_map_sub_const (a : V3) (l : List V3) :
    vsum (l.map (fun x => V3.sub x a)) = V3.sub (vsum l) (V3.smul (l.length : Rat) a) := by
  induction l with
  | nil =>
    simp only [List.map_nil, vsum_nil, List.length_nil]
    apply V3.ext <;> simp only [V3.sub, V3.smul, V3.zero] <;> grind
  | cons x xs ih =>
    simp only [List.map_cons, vsum_cons, ih, List.length_cons]
    apply V3.ext <;> simp only [V3.add, V3.sub, V3.smul] <;> grind

theorem dot_vsum_right (u : V3) (l : List V3) :
    V3.dot u (vsum l) = ssum (l.map (fun v => V3.dot u v)) := by
  induction l with
  | nil => simp only [vsum_nil, V3.dot_zero_right, List.map_nil, ssum_nil]
  | cons x xs ih =>
    simp only [vsum_cons, List.map_cons, ssum_cons]
    rw [V3.dot_add_right, ih]

theorem dot_vsum_left (l : List V3) (u : V3) :
    V3.dot (vsum l) u = ssum (l.map (fun v => V3.dot v u)) := by
  induction l with
  | nil => simp only [vsum_nil, V3.dot_zero_left, List.map_nil, ssum_nil]
  | cons x xs ih =>
    simp only [vsum_cons, List.map_cons, ssum_cons]
    rw [V3.dot_add_left, ih]

theorem smul_one (u : V3) : V3.smul 1 u = u := by
  cases u; simp only [V3.smul]; grind

end Foundation
