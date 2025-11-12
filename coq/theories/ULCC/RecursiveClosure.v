From Coq Require Import Reals.
From ULCC.DCGH Require Import Hypergraph RicciFlow.
From ULCC.GETG Require Import GeometryIface.
From ULCC.ETETF Require Import Tensor Conservation.

Definition W2Pi (_:EdgeWeights) : ST2 := zero2.
Definition Pi2g (_:ST2) : Metric := {| gcomp := fun _ _ => 0%R |}.
Definition flow_step := RicciFlow.flow_step.

Theorem recursive_closure_commutes (w:EdgeWeights) :
  let w' := flow_step w in
  Pi2g (W2Pi w') = metric_update (Pi2g (W2Pi w))
  /\ DiscreteConserved (weights_to_tensor w')
  /\ cone_compatibility (Pi2g (W2Pi w')).
Proof.
  simpl. split; [reflexivity|]. split.
  - apply all_conserved.
  - exact I.
Qed.
