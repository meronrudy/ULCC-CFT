From Coq Require Import List.
Import ListNotations.

Record PosetIsomorphism := {
  iso_fn : nat -> nat;
  iso_inv : nat -> nat;
  iso_left : forall x, iso_inv (iso_fn x) = x;
  iso_right : forall x, iso_fn (iso_inv x) = x
}.

Definition map_iso (iso:PosetIsomorphism) (xs:list nat) : list nat :=
  map iso.(iso_fn) xs.
