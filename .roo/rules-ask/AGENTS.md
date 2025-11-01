# AGENTS.md

This file provides guidance to agents when working with code in this repository.

Project Documentation Rules (Non-Obvious Only):
- Document the import-name ↔ folder-name mismatch and required symlinks prominently in usage examples.
- Note intentional stubs: [parallel_transport_identity()](ulcc-ddg/transport.py:5)=1.0 and [holonomy_loop()](ulcc_ddg/holonomy.py:7)=0.0; contrast with spec-first’s full holonomy [holonomy()](spec-first/geom/holonomy.py:15).
- Emphasize determinism policy (rng via np.random.default_rng) and boundary-domain constraints for θ ∈ (0,1).
- Use tests as canonical examples; cite single-test invocation with hyphenated path:
  pytest -q [ulcc-core/tests/test_core.py::test_coord_roundtrip](ulcc-core/tests/test_core.py:18).