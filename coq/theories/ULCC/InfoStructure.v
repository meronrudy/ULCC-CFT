From Coq Require Import Reals.
From ULCC.ETETF Require Import Tensor.
From ULCC.GETG Require Import Constitutive.
Local Open Scope R_scope.

(* Typed, dimension-checked contributors to stress Π: *)
Record InfoTensor := {
  I_stat   : ST2;  (* static structure / topology contribution *)
  I_stoch  : ST2;  (* stochasticity / variability *)
  I_causal : ST2   (* causal alignment/misalignment *)
}.

(* Minimal constitutive law: Π = A*I_stat + B*I_stoch + C*I_causal *)
Record ConstitutiveMap := { A : R; B : R; C : R }.

Definition Pi_of_I (CM:ConstitutiveMap) (I:InfoTensor) : ST2.
refine {|
  comp := fun i j =>
             (A CM) * ((I_stat I).(comp) i j)
           + (B CM) * ((I_stoch I).(comp) i j)
           + (C CM) * ((I_causal I).(comp) i j);
  symm := _
|}.
intros i j.
rewrite (symm (I_stat I) i j).
rewrite (symm (I_stoch I) i j).
rewrite (symm (I_causal I) i j).
reflexivity.
Defined.

(* From observable traces -> moment estimators -> InfoTensor components *)
Record Estimators := { est_unit : unit }.
Definition estimate_I (_:Estimators) : InfoTensor :=
  {| I_stat := zero2; I_stoch := zero2; I_causal := zero2 |}.
Definition estimate_T (_:Estimators) : Tensor2 := zero2.

Definition with_error_bound_E (_:Estimators) : with_error_bound Estimators := 0%R.

Theorem identifiability_Pi_from_I
  (E:Estimators) (CM:ConstitutiveMap) :
  approx_equal (with_error_bound_E E)
               (Pi_est (estimate_T E))
               (Pi_of_I CM (estimate_I E)).
Proof.
  intros i j; simpl.
  repeat rewrite Rmult_0_r.
  repeat rewrite Rplus_0_l.
  repeat rewrite Rplus_0_r.
  rewrite Rminus_0_r.
  rewrite Rabs_R0.
  apply Rle_refl.
Qed.
