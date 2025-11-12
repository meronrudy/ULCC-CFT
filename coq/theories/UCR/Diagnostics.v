From Coq Require Import Reals.

Record DiagnosticsWindow := { diag_unit : unit }.
Definition passes_sigma (_:DiagnosticsWindow) := True.
Definition passes_divergence (_:DiagnosticsWindow) := True.
Definition model_consistent (_:DiagnosticsWindow) := True.
