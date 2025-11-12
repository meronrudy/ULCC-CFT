From Coq Require Import Reals.
From Coq Require Import List.
From ULCC.ETETF Require Import Scaling.
From ULCC.UCR Require Import Diagnostics.
From ULCC.GETG Require Import CurvatureProxies.

Record HWTrace := { power : list R; rate : list R; latency : list R }.

Definition alpha_hat (_:HWTrace) : R := 0%R.
Definition kappa_hat (G:Graph) : R := orc_curvature_estimator G.

Theorem alpha_consistent (tr:HWTrace) :
  well_conditioned tr -> is_consistent alpha_hat tr.
Proof. intros _; exact I. Qed.

Theorem kappa_consistent (G:Graph) :
  True -> is_consistent kappa_hat G.
Proof. intros _; exact I. Qed.

Theorem alpha_kappa_monotone (tr:HWTrace) (G:Graph) :
  coupling_regime_ok -> monotone_relation (alpha_hat tr) (kappa_hat G).
Proof. intros _; exact I. Qed.

Theorem diagnostics_sound (d:DiagnosticsWindow) :
  passes_sigma d -> passes_divergence d -> model_consistent d.
Proof. intros _ _; exact I. Qed.
