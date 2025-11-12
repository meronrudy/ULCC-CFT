From Coq Require Import Reals.

Record Metric := { gcomp : nat -> nat -> R }.

Definition metric_update (m:Metric) : Metric := m.
Definition cone_compatibility (_:Metric) : Prop := True.
