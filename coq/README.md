# ULCC Coq Subproject

## Prereqs (opam)
- coq 8.17.x (or current LTS)
- coquelicot, mathcomp-ssreflect, mathcomp-algebra, mathcomp-finmap, stdpp, hb
- quickchick (optional), coq-interval (optional)

## Build
make -j all

## Coqdoc (ULCC modules)
make coqdoc
