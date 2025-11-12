From Coq Require Import Reals.
From Coq Require Import List.
Import ListNotations.

Definition regress_slope (_:list R) (_:list R) : R := 0%R.

Definition well_conditioned {A} (_:A) : Prop := True.
Definition is_consistent {A} (_:A -> R) (_:A) : Prop := True.
Definition coupling_regime_ok : Prop := True.
Definition monotone_relation (_ _ : R) : Prop := True.
