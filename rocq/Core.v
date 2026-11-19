From Stdlib Require Import QArith Lia.

Record Vec3 := {
  vx : Q;
  vy : Q;
  vz : Q
}.

(* Pointwise (Qeq) equality on vectors: two vectors are equal when their
   components are equal as rationals. *)
Definition veq (a b : Vec3) : Prop :=
  vx a == vx b /\ vy a == vy b /\ vz a == vz b.

Definition zero : Vec3 := {| vx := 0; vy := 0; vz := 0 |}.

Definition vadd (a b : Vec3) : Vec3 :=
  {| vx := vx a + vx b;
     vy := vy a + vy b;
     vz := vz a + vz b |}.

Definition vneg (a : Vec3) : Vec3 :=
  {| vx := - vx a;
     vy := - vy a;
     vz := - vz a |}.

Definition pair_forces (force_on_first : Vec3) : Vec3 * Vec3 :=
  (force_on_first, vneg force_on_first).

Lemma q_add_opp : forall x : Q, x + (- x) == 0.
Proof.
  intros x. unfold Qeq, Qplus, Qopp. simpl. lia.
Qed.

Theorem pair_forces_conserve_momentum (f : Vec3) :
  veq (vadd (fst (pair_forces f)) (snd (pair_forces f))) zero.
Proof.
  destruct f as [x y z].
  unfold pair_forces, vadd, vneg, zero, veq; simpl.
  repeat split; apply q_add_opp.
Qed.
