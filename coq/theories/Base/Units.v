From Coq Require Import Bool ZArith Reals.
Open Scope Z_scope.
Open Scope R_scope.

Record Dim : Type := { L : Z ; M : Z ; Tm : Z }.

Definition Dimless : Dim :=
  {| L := 0%Z ; M := 0%Z ; Tm := 0%Z |}.

Definition dim_eq (d1 d2 : Dim) : bool :=
  andb (Z.eqb (L d1) (L d2))
       (andb (Z.eqb (M d1) (M d2)) (Z.eqb (Tm d1) (Tm d2))).

Record Q : Type := { dim : Dim ; val : R }.

Definition is_dimless (x:Q) : bool :=
  dim_eq (dim x) Dimless.

Definition qlog (x:Q) (_H : is_dimless x = true) : Q :=
  {| dim := Dimless ; val := ln (val x) |}.
