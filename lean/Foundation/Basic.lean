/-
Core invariants for the pair-force model.

We model 3-vectors over an abstract additive commutative group `G`
(a structure extending `Add`, `Neg` and `Zero` with commutativity,
defined below since the plain Lean 4 core has no `AddCommGroup`),
so the invariant holds for rationals, reals, or any other abelian
group of force values.

`pairForces f` returns the pair of forces exchanged by a particle
pair: `(f, -f)`. The theorem states the componentwise sum is zero,
i.e. internal forces cancel and linear momentum is conserved.
-/

namespace Foundation

/-- Minimal additive commutative group structure used for force values. -/
class AddCommGroup (A : Type) extends Add A, Neg A, Zero A where
  add_comm : ∀ a b : A, a + b = b + a
  add_zero : ∀ a : A, a + 0 = a
  neg_add : ∀ a : A, a + (-a) = 0

variable {G : Type} [AddCommGroup G]

/-- A 3-vector of quantities from an additive commutative group. -/
structure Vec3 (G : Type) [AddCommGroup G] where
  x : G
  y : G
  z : G

/-- Componentwise addition. -/
def Vec3.add (a b : Vec3 G) : Vec3 G := ⟨a.x + b.x, a.y + b.y, a.z + b.z⟩

/-- Componentwise negation. -/
def Vec3.neg (a : Vec3 G) : Vec3 G := ⟨-a.x, -a.y, -a.z⟩

/-- Zero vector. -/
def Vec3.zero (G : Type) [AddCommGroup G] : Vec3 G := ⟨0, 0, 0⟩

/-- The two forces a particle pair exchanges: `(f, -f)`. -/
def pairForces (forceOnFirst : Vec3 G) : Vec3 G × Vec3 G :=
  (forceOnFirst, forceOnFirst.neg)

/-- Componentwise sum of the force pair is the zero vector. -/
theorem pair_forces_conserve_momentum (f : Vec3 G) :
    (pairForces f).1.add (pairForces f).2 = Vec3.zero G := by
  cases f with
  | mk fx fy fz =>
    simp only [pairForces, Vec3.add, Vec3.neg, Vec3.zero]
    have h := AddCommGroup.neg_add (A := G) fx
    have h2 := AddCommGroup.neg_add (A := G) fy
    have h3 := AddCommGroup.neg_add (A := G) fz
    exact (by rw [h, h2, h3])

end Foundation
