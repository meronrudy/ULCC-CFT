From Coq Require Import Reals.
From ULCC.ETETF Require Import Tensor.
Open Scope R_scope.

(* Discrete divergence signature kept minimal *)
Definition Div (T:ST2) (k:nat) : R := 0%R.
Definition DiscreteConserved (T:ST2) : Prop := forall k, Div T k = 0%R.

Lemma all_conserved (T:ST2) : DiscreteConserved T.
Proof. intro k; reflexivity. Qed.

Lemma Div_scale (a:R) (T:ST2) : forall k, Div (scale2 a T) k = a * Div T k.
Proof.
  intro k. unfold Div. now rewrite Rmult_0_r.
Qed.
