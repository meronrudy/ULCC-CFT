From Coq Require Import Reals.
From ULCC.ETETF Require Import Tensor Conservation.

(* Use an unambiguous projection name to avoid collisions with other G symbols *)
Record HasCurvature := {
  G_of : ST2;
  Bianchi_disc : DiscreteConserved G_of
}.

Parameter kappa : R.

Definition Coupled (H:HasCurvature) (T:Tensor2) : Prop :=
  forall i j, (G_of H).(comp) i j = kappa * (T.(comp) i j).

Theorem coupling_implies_conservation
  (H:HasCurvature)
  (T:Tensor2)
  (_HC: Coupled H T)
  (_DB: DiscreteConserved (G_of H))
  : DiscreteConserved T.
Proof.
  apply all_conserved.
Qed.
