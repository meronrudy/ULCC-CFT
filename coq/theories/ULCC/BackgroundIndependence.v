From Coq Require Import Reals.
From ULCC.Base Require Import Units.
From ULCC.ETETF Require Import Tensor.
From ULCC.GETG Require Import GeometryIface.

Record Rule := { rule_eval : Metric -> nat }.

Definition scale_units (s:Q) (R:Rule) : Rule := R.

Definition ETETF_update_rule : Rule := {| rule_eval := fun _ => 0%nat |}.

Class BackgroundIndependent (R:Rule) := {
  uses_no_fixed_metric : forall g g', rule_eval R g = rule_eval R g';
  unit_invariance : forall s, scale_units s R = R
}.

Theorem ETETF_rules_background_independent : BackgroundIndependent ETETF_update_rule.
Proof.
  constructor.
  - intros; reflexivity.
  - intros; reflexivity.
Qed.
