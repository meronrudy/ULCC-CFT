From Coq Require Import Reals.
Local Open Scope R_scope.

Record ST2 := {
  comp : nat -> nat -> R ;
  symm : forall i j, comp i j = comp j i
}.
Definition Tensor2 := ST2.

Definition zero2 : ST2.
refine {| comp := fun _ _ => 0%R;
          symm := _ |}.
intros i j; reflexivity.
Defined.

Definition add2 (A B:ST2) : ST2.
refine {| comp := fun i j => A.(comp) i j + B.(comp) i j;
          symm := _ |}.
intros i j; rewrite (symm A i j). rewrite (symm B i j). reflexivity.
Defined.

Definition scale2 (a:R) (A:ST2) : ST2.
refine {| comp := fun i j => a * A.(comp) i j;
          symm := _ |}.
intros i j; rewrite (symm A i j); reflexivity.
Defined.

Definition approx_equal (eps:R) (A B:ST2) : Prop :=
  forall i j, Rabs (A.(comp) i j - B.(comp) i j) <= eps.

Definition with_error_bound (E:Type) := R.

Definition Pi_est (T:Tensor2) : ST2 := T.
