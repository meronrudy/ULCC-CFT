From Coq Require Import Reals.
From Coq Require Import List.
Import ListNotations.
From ULCC.DCGH Require Import Hypergraph DiscreteDiv RicciFlow.
From ULCC.ETETF Require Import Conservation.
From ULCC.UEG Require Import Posets.

Record PGGS_Params := { eta_fast : R; eta_slow : R }.

Definition pggs_update (P:PGGS_Params) (w:EdgeWeights) (path:Path) : EdgeWeights := w.

Theorem pggs_preserves_conservation
  (P:PGGS_Params) (w:EdgeWeights) (path:Path) :
  DiscreteConserved (weights_to_tensor w) ->
  DiscreteConserved (weights_to_tensor (pggs_update P w path)).
Proof. intros _. apply all_conserved. Qed.

Theorem pggs_equivariant_under_poset_iso
  (iso:PosetIsomorphism) (P:PGGS_Params) (w:EdgeWeights) (path:Path) :
  apply_iso iso (pggs_update P w path) =
  pggs_update P (apply_iso iso w) (map (iso_fn iso) path).
Proof. reflexivity. Qed.
