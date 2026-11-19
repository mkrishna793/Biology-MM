/-
Balanced Ternary and Symplectic Reversible Dynamics in Lean 4.

This module formalizes:
1. Balanced ternary digit type `Trit` (-1, 0, +1) and involutive negation.
2. Dual-rail wire representation `DualRail` with 0-cycle rail-swap negation.
3. Balanced Ternary Half Adder (`btha`) and Full Adder (`btfa`) with exact
   algebraic radix-3 conservation laws.
4. Carry-free pair accumulator invariant on `CarryPair` representing `val = s + 3 * c`.
5. Exact integer Störmer-Verlet Kick-Drift-Kick (KDK) symplectic reversibility,
   proving zero-information-loss step inversion and n-step time-reversal invariance.
6. Multi-degree-of-freedom / list / map extensions and Hamiltonian energy invariants.
-/

import Init.Data.Int
import Init.Data.List

namespace Foundation

/-!
### 1. Balanced Ternary Digit Type (Trit)
-/

/-- Balanced ternary digit: neg (-1), zero (0), pos (+1). -/
inductive Trit where
  | neg : Trit
  | zero : Trit
  | pos : Trit
  deriving DecidableEq, Repr

namespace Trit

/-- Integer valuation: -1, 0, +1. -/
def toInt : Trit → Int
  | neg  => -1
  | zero =>  0
  | pos  =>  1

/-- Symmetric trit negation: neg ↔ pos, zero ↔ zero. -/
def negate : Trit → Trit
  | neg  => pos
  | zero => zero
  | pos  => neg

/-- Negation is an involution: negate (negate t) = t. -/
theorem trit_neg_involutive (t : Trit) : t.negate.negate = t := by
  cases t <;> rfl

/-- Negation inverts the integer valuation: (negate t).toInt = -t.toInt. -/
theorem trit_neg_toInt (t : Trit) : t.negate.toInt = -t.toInt := by
  cases t <;> rfl

end Trit

theorem trit_neg_involutive (t : Trit) : t.negate.negate = t :=
  Trit.trit_neg_involutive t

theorem trit_neg_toInt (t : Trit) : t.negate.toInt = -t.toInt :=
  Trit.trit_neg_toInt t

/-!
### 2. Dual-Rail Wire Representation
-/

/-- Dual-rail wire with positive (p) and negative (n) rails.
    Valid condition: both rails cannot be simultaneously asserted high (true). -/
structure DualRail where
  p : Bool
  n : Bool
  valid : ¬(p = true ∧ n = true)
  deriving DecidableEq, Repr

namespace DualRail

/-- 0-cycle negation via rail swap (n, p). -/
def neg (w : DualRail) : DualRail :=
  ⟨w.n, w.p, fun h => w.valid ⟨h.2, h.1⟩⟩

/-- Dual-rail negation is involutive: w.neg.neg = w. -/
theorem dual_rail_neg_involutive (w : DualRail) : w.neg.neg = w := by
  cases w
  rfl

/-- Integer valuation of dual-rail wire: +1 for positive rail, -1 for negative rail, 0 for idle. -/
def toInt (w : DualRail) : Int :=
  (if w.p then 1 else 0) - (if w.n then 1 else 0)

/-- Rail swap negates the integer value: w.neg.toInt = -w.toInt. -/
theorem dual_rail_neg_toInt (w : DualRail) : w.neg.toInt = -w.toInt := by
  rcases w with ⟨p, n, _⟩
  dsimp [neg, toInt]
  omega

/-- Inject a Trit into its DualRail wire representation. -/
def ofTrit (t : Trit) : DualRail :=
  match t with
  | Trit.pos  => ⟨true, false, fun h => by cases h.2⟩
  | Trit.zero => ⟨false, false, fun h => by cases h.1⟩
  | Trit.neg  => ⟨false, true, fun h => by cases h.1⟩

/-- Project a DualRail wire back to a Trit. -/
def toTrit (w : DualRail) : Trit :=
  match w.p, w.n with
  | true, false => Trit.pos
  | false, true => Trit.neg
  | false, false => Trit.zero
  | true, true => Trit.zero

theorem dual_rail_of_to_trit (t : Trit) : (DualRail.ofTrit t).toTrit = t := by
  cases t <;> rfl

theorem dual_rail_of_trit_toInt (t : Trit) : (DualRail.ofTrit t).toInt = t.toInt := by
  cases t <;> rfl

end DualRail

theorem dual_rail_neg_involutive (w : DualRail) : w.neg.neg = w :=
  DualRail.dual_rail_neg_involutive w

theorem dual_rail_neg_toInt (w : DualRail) : w.neg.toInt = -w.toInt :=
  DualRail.dual_rail_neg_toInt w

/-!
### 3. Balanced Ternary Half Adder and Full Adder
-/

/-- Balanced Ternary Half Adder: returns (sum, carry) such that sum + 3 * carry = a + b. -/
def btha (a b : Trit) : Trit × Trit :=
  match a, b with
  | Trit.pos,  Trit.pos  => (Trit.neg,  Trit.pos)   -- 1 + 1 = 2 = -1 + 3*(1)
  | Trit.pos,  Trit.zero => (Trit.pos,  Trit.zero)  -- 1 + 0 = 1 = 1 + 3*(0)
  | Trit.pos,  Trit.neg  => (Trit.zero, Trit.zero)  -- 1 - 1 = 0 = 0 + 3*(0)
  | Trit.zero, Trit.pos  => (Trit.pos,  Trit.zero)  -- 0 + 1 = 1 = 1 + 3*(0)
  | Trit.zero, Trit.zero => (Trit.zero, Trit.zero)  -- 0 + 0 = 0 = 0 + 3*(0)
  | Trit.zero, Trit.neg  => (Trit.neg,  Trit.zero)  -- 0 - 1 = -1 = -1 + 3*(0)
  | Trit.neg,  Trit.pos  => (Trit.zero, Trit.zero)  -- -1 + 1 = 0 = 0 + 3*(0)
  | Trit.neg,  Trit.zero => (Trit.neg,  Trit.zero)  -- -1 + 0 = -1 = -1 + 3*(0)
  | Trit.neg,  Trit.neg  => (Trit.pos,  Trit.neg)   -- -1 - 1 = -2 = 1 + 3*(-1)

/-- Algebraic conservation of half-adder: (btha a b).1.toInt + 3 * (btha a b).2.toInt = a.toInt + b.toInt. -/
theorem btha_conservation (a b : Trit) :
    (btha a b).1.toInt + 3 * (btha a b).2.toInt = a.toInt + b.toInt := by
  cases a <;> cases b <;> rfl

/-- Balanced Ternary Full Adder: returns (sum, carry) such that sum + 3 * carry = a + b + c. -/
def btfa (a b c : Trit) : Trit × Trit :=
  match a, b, c with
  -- sum = 3 -> 0 + 3*(1)
  | Trit.pos,  Trit.pos,  Trit.pos  => (Trit.zero, Trit.pos)
  -- sum = 2 -> -1 + 3*(1)
  | Trit.pos,  Trit.pos,  Trit.zero => (Trit.neg,  Trit.pos)
  | Trit.pos,  Trit.zero, Trit.pos  => (Trit.neg,  Trit.pos)
  | Trit.zero, Trit.pos,  Trit.pos  => (Trit.neg,  Trit.pos)
  -- sum = 1 -> 1 + 3*(0)
  | Trit.pos,  Trit.pos,  Trit.neg  => (Trit.pos,  Trit.zero)
  | Trit.pos,  Trit.zero, Trit.zero => (Trit.pos,  Trit.zero)
  | Trit.pos,  Trit.neg,  Trit.pos  => (Trit.pos,  Trit.zero)
  | Trit.zero, Trit.pos,  Trit.zero => (Trit.pos,  Trit.zero)
  | Trit.zero, Trit.zero, Trit.pos  => (Trit.pos,  Trit.zero)
  | Trit.neg,  Trit.pos,  Trit.pos  => (Trit.pos,  Trit.zero)
  -- sum = 0 -> 0 + 3*(0)
  | Trit.pos,  Trit.zero, Trit.neg  => (Trit.zero, Trit.zero)
  | Trit.pos,  Trit.neg,  Trit.zero => (Trit.zero, Trit.zero)
  | Trit.zero, Trit.pos,  Trit.neg  => (Trit.zero, Trit.zero)
  | Trit.zero, Trit.zero, Trit.zero => (Trit.zero, Trit.zero)
  | Trit.zero, Trit.neg,  Trit.pos  => (Trit.zero, Trit.zero)
  | Trit.neg,  Trit.pos,  Trit.zero => (Trit.zero, Trit.zero)
  | Trit.neg,  Trit.zero, Trit.pos  => (Trit.zero, Trit.zero)
  -- sum = -1 -> -1 + 3*(0)
  | Trit.pos,  Trit.neg,  Trit.neg  => (Trit.neg,  Trit.zero)
  | Trit.zero, Trit.zero, Trit.neg  => (Trit.neg,  Trit.zero)
  | Trit.zero, Trit.neg,  Trit.zero => (Trit.neg,  Trit.zero)
  | Trit.neg,  Trit.pos,  Trit.neg  => (Trit.neg,  Trit.zero)
  | Trit.neg,  Trit.zero, Trit.zero => (Trit.neg,  Trit.zero)
  | Trit.neg,  Trit.neg,  Trit.pos  => (Trit.neg,  Trit.zero)
  -- sum = -2 -> 1 + 3*(-1)
  | Trit.zero, Trit.neg,  Trit.neg  => (Trit.pos,  Trit.neg)
  | Trit.neg,  Trit.zero, Trit.neg  => (Trit.pos,  Trit.neg)
  | Trit.neg,  Trit.neg,  Trit.zero => (Trit.pos,  Trit.neg)
  -- sum = -3 -> 0 + 3*(-1)
  | Trit.neg,  Trit.neg,  Trit.neg  => (Trit.zero, Trit.neg)

/-- Algebraic conservation of full-adder: (btfa a b c).1.toInt + 3 * (btfa a b c).2.toInt = a.toInt + b.toInt + c.toInt. -/
theorem btfa_conservation (a b c : Trit) :
    (btfa a b c).1.toInt + 3 * (btfa a b c).2.toInt = a.toInt + b.toInt + c.toInt := by
  cases a <;> cases b <;> cases c <;> rfl

/-!
### 4. Carry-Free Pair Accumulator Invariant
-/

/-- Redundant radix-3 carry-free accumulator state: val = s + 3 * c. -/
structure CarryPair where
  s : Int
  c : Int
  deriving DecidableEq, Repr

namespace CarryPair

/-- Integer valuation: sum + 3 * carry. -/
def val (p : CarryPair) : Int :=
  p.s + 3 * p.c

/-- Local accumulator step with incoming increments. -/
def step (p : CarryPair) (ds dc : Int) : CarryPair :=
  ⟨p.s + ds, p.c + dc⟩

theorem carry_free_conservation (p : CarryPair) (s' c' x : Int)
    (h : s' + 3 * c' = p.s + 3 * p.c + x) :
    (CarryPair.mk s' c').val = p.val + x := by
  dsimp [CarryPair.val] at *
  omega

end CarryPair

/-- Conservation of carry-free decomposition for arbitrary inputs. -/
theorem carry_free_conservation (s c s' c' x : Int)
    (h : s' + 3 * c' = s + 3 * c + x) :
    (CarryPair.mk s' c').val = (CarryPair.mk s c).val + x := by
  dsimp [CarryPair.val] at *
  omega

/-- Accumulator step conservation: when ds + 3 * dc = x, updating (s, c) by (ds, dc) preserves val + x. -/
theorem carry_free_step_conservation (p : CarryPair) (x ds dc : Int)
    (h : ds + 3 * dc = x) :
    (p.step ds dc).val = p.val + x := by
  dsimp [CarryPair.step, CarryPair.val] at *
  omega

/-!
### 5. Störmer-Verlet Kick-Drift-Kick (KDK) Exact Symplectic Reversibility
-/

/-- Canonical phase-space state: position q and momentum p. -/
structure PhaseState where
  q : Int
  p : Int
  deriving DecidableEq, Repr

/-- Forward Störmer-Verlet KDK step:
    1. half-kick: p1 = s.p + F s.q
    2. drift:     q1 = s.q + V p1
    3. half-kick: p2 = p1 + F q1 -/
def verletKDK_forward (F V : Int → Int) (s : PhaseState) : PhaseState :=
  let p1 := s.p + F s.q
  let q1 := s.q + V p1
  let p2 := p1 + F q1
  ⟨q1, p2⟩

/-- Reverse Störmer-Verlet KDK step (running time backwards):
    1. undo half-kick: p1 = s.p - F s.q
    2. undo drift:     q0 = s.q - V p1
    3. undo half-kick: p0 = p1 - F q0 -/
def verletKDK_reverse (F V : Int → Int) (s : PhaseState) : PhaseState :=
  let p1 := s.p - F s.q
  let q0 := s.q - V p1
  let p0 := p1 - F q0
  ⟨q0, p0⟩

/-- Exact reversibility: reverse (forward s) = s for ANY force F and drift map V.
    This guarantees zero information loss without storing intermediate states. -/
theorem verletKDK_exact_reversible (F V : Int → Int) (s : PhaseState) :
    verletKDK_reverse F V (verletKDK_forward F V s) = s := by
  dsimp [verletKDK_forward, verletKDK_reverse]
  rcases s with ⟨q, p⟩
  have h1 : p + F q + F (q + V (p + F q)) - F (q + V (p + F q)) = p + F q := by omega
  rw [h1]
  have h2 : q + V (p + F q) - V (p + F q) = q := by omega
  rw [h2]
  have h3 : p + F q - F q = p := by omega
  rw [h3]

/-- Exact invertibility: forward (reverse s) = s. -/
theorem verletKDK_reverse_forward (F V : Int → Int) (s : PhaseState) :
    verletKDK_forward F V (verletKDK_reverse F V s) = s := by
  dsimp [verletKDK_forward, verletKDK_reverse]
  rcases s with ⟨q, p⟩
  have h1 : p - F q - F (q - V (p - F q)) + F (q - V (p - F q)) = p - F q := by omega
  have h2 : q - V (p - F q) + V (p - F q) = q := by omega
  have h3 : p - F q + F q = p := by omega
  rw [h1, h2, h3]

/-- Iterated n-step execution of an endomorphism. -/
def stepN {α : Type} (f : α → α) : Nat → α → α
  | 0, s     => s
  | n + 1, s => f (stepN f n s)

theorem stepN_succ {α : Type} (f : α → α) (n : Nat) (s : α) :
    stepN f (n + 1) s = stepN f n (f s) := by
  induction n generalizing s with
  | zero => rfl
  | succ n ih =>
    dsimp [stepN]
    congr 1
    exact ih s

/-- Generic n-step inversion lemma: if g (f s) = s, then g^n (f^n s) = s. -/
theorem stepN_inv {α : Type} (f g : α → α) (h_inv : ∀ s, g (f s) = s) :
    ∀ (n : Nat) (s : α), stepN g n (stepN f n s) = s := by
  intro n
  induction n with
  | zero => intro s; rfl
  | succ n ih =>
    intro s
    have h_f : stepN f (n + 1) s = f (stepN f n s) := rfl
    rw [h_f]
    rw [stepN_succ g n (f (stepN f n s))]
    rw [h_inv (stepN f n s)]
    exact ih s

/-- N-step Störmer-Verlet exact reversibility: reverse^n (forward^n s) = s. -/
theorem verletKDK_n_steps_reversible (F V : Int → Int) :
    ∀ (n : Nat) (s : PhaseState),
      stepN (verletKDK_reverse F V) n (stepN (verletKDK_forward F V) n s) = s := by
  intro n s
  exact stepN_inv (verletKDK_forward F V) (verletKDK_reverse F V)
    (verletKDK_exact_reversible F V) n s

/-- N-step reverse-forward invertibility: forward^n (reverse^n s) = s. -/
theorem verletKDK_n_steps_reverse_forward (F V : Int → Int) :
    ∀ (n : Nat) (s : PhaseState),
      stepN (verletKDK_forward F V) n (stepN (verletKDK_reverse F V) n s) = s := by
  intro n s
  exact stepN_inv (verletKDK_reverse F V) (verletKDK_forward F V)
    (verletKDK_reverse_forward F V) n s

/-!
### 6. Vector / List / Map Extensions & Hamiltonian Energy Invariants
-/

/-- Multi-particle list forward step. -/
def verletList_forward (F V : Int → Int) : List PhaseState → List PhaseState :=
  List.map (verletKDK_forward F V)

/-- Multi-particle list reverse step. -/
def verletList_reverse (F V : Int → Int) : List PhaseState → List PhaseState :=
  List.map (verletKDK_reverse F V)

theorem verletList_exact_reversible (F V : Int → Int) (xs : List PhaseState) :
    verletList_reverse F V (verletList_forward F V xs) = xs := by
  dsimp [verletList_reverse, verletList_forward]
  rw [List.map_map]
  have h_comp : (verletKDK_reverse F V ∘ verletKDK_forward F V) = id := by
    funext s
    exact verletKDK_exact_reversible F V s
  rw [h_comp, List.map_id]

theorem verletList_reverse_forward (F V : Int → Int) (xs : List PhaseState) :
    verletList_forward F V (verletList_reverse F V xs) = xs := by
  dsimp [verletList_forward, verletList_reverse]
  rw [List.map_map]
  have h_comp : (verletKDK_forward F V ∘ verletKDK_reverse F V) = id := by
    funext s
    exact verletKDK_reverse_forward F V s
  rw [h_comp, List.map_id]

/-- Arbitrary degree-of-freedom forward step. -/
def verletMap_forward {α : Type} (F V : Int → Int) (s : α → PhaseState) : α → PhaseState :=
  fun i => verletKDK_forward F V (s i)

/-- Arbitrary degree-of-freedom reverse step. -/
def verletMap_reverse {α : Type} (F V : Int → Int) (s : α → PhaseState) : α → PhaseState :=
  fun i => verletKDK_reverse F V (s i)

theorem verletMap_exact_reversible {α : Type} (F V : Int → Int) (s : α → PhaseState) :
    verletMap_reverse F V (verletMap_forward F V s) = s := by
  funext i
  exact verletKDK_exact_reversible F V (s i)

theorem verletMap_reverse_forward {α : Type} (F V : Int → Int) (s : α → PhaseState) :
    verletMap_forward F V (verletMap_reverse F V s) = s := by
  funext i
  exact verletKDK_reverse_forward F V (s i)

theorem verletMap_n_steps_reversible {α : Type} (F V : Int → Int) :
    ∀ (n : Nat) (s : α → PhaseState),
      stepN (verletMap_reverse F V) n (stepN (verletMap_forward F V) n s) = s := by
  intro n s
  exact stepN_inv (verletMap_forward F V) (verletMap_reverse F V)
    (verletMap_exact_reversible F V) n s

/-- Hamiltonian radix-3 energy of phase state: H(q, p) = q + 3 * p. -/
def hamiltonianEnergy (s : PhaseState) : Int :=
  s.q + 3 * s.p

/-- Hamiltonian addition conservation: H(Q + dQ, P + dP) = H(Q, P) + X when dQ + 3 * dP = X. -/
theorem hamiltonian_energy_conservation (Q P X dQ dP : Int)
    (h_local : dQ + 3 * dP = X) :
    hamiltonianEnergy ⟨Q + dQ, P + dP⟩ = hamiltonianEnergy ⟨Q, P⟩ + X := by
  dsimp [hamiltonianEnergy]
  omega

/-- Hamiltonian subtraction conservation: H(Q - dQ, P - dP) = H(Q, P) - X when dQ + 3 * dP = X. -/
theorem hamiltonian_subtraction (Q P X dQ dP : Int)
    (h_local : dQ + 3 * dP = X) :
    hamiltonianEnergy ⟨Q - dQ, P - dP⟩ = hamiltonianEnergy ⟨Q, P⟩ - X := by
  dsimp [hamiltonianEnergy]
  omega

end Foundation
