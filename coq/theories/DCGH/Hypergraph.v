From Coq Require Import Reals List.
Import ListNotations.
From ULCC.UEG Require Import Posets.
From ULCC.ETETF Require Import Tensor.

Definition EdgeId := nat.
Definition EdgeWeights := EdgeId -> R.
Definition Path := list EdgeId.

Definition apply_iso (iso:PosetIsomorphism) (w:EdgeWeights) : EdgeWeights :=
  fun e => w (iso_inv iso e).

Definition weights_to_tensor (_:EdgeWeights) : ST2 := zero2.
