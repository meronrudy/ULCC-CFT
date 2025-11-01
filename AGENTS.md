# AGENTS.md

This file provides guidance to agents when working with code in this repository.

Non-obvious, project-specific rules:
- Import name vs folder mismatch: before tests/runs create symlinks in repo root:
  ln -s ulcc-core ulcc_core; ln -s ulcc-ddg ulcc_ddg; ln -s pggs-toy ulcc_pggs
  Recreate inside Docker too.
- Tests rely on symlinks. All tests: make test (runs pytest -q). Single test example:
  pytest -q [ulcc-core/tests/test_core.py::test_coord_roundtrip](ulcc-core/tests/test_core.py:18)
- Docker gotcha: docker/Dockerfile must add symlinks or imports fail. Add:
  RUN ln -s /app/ulcc-core /app/ulcc_core && ln -s /app/ulcc-ddg /app/ulcc_ddg && ln -s /app/pggs-toy /app/ulcc_pggs
- Determinism: thread rng through PGGS; tests assume rng=np.random.default_rng(0). See [pggs_toy.sampler.sample_paths()](pggs-toy/sampler.py:7).
- Domain guards are API: θ must be strictly in (0,1); avoid boundaries. See [theta_to_phi()](ulcc-core/coords.py:4), [fisher_metric_bernoulli()](ulcc-core/fisher.py:4), [gamma_theta()](ulcc-core/christoffel.py:4).
- Intentional stubs (do not “improve” semantics):
  [parallel_transport_identity()](ulcc-ddg/transport.py:5) → 1.0; [holonomy_loop()](ulcc_ddg/holonomy.py:7) → 0.0 under identity transport; [guide_score()](pggs-toy/guide.py:2) ≈ identity with floor 1e-8.
- Types/aliases contract:
  Scalars return native float (e.g., [ngd_step()](ulcc-core/dynamics.py:16)); vectors return np.ndarray float (e.g., [geodesic_rhs()](ulcc-core/dynamics.py:6)); keep alias “Gamma” (see [ulcc-core/dynamics.py](ulcc-core/dynamics.py:3)).
- Control protocol must follow staged order:
  [GCU.shadow_apply()](control/gcu.py:82) → [GCU.quick_check()](control/gcu.py:102) → [GCU.quiesce()](control/gcu.py:138) → [GCU.commit()](control/gcu.py:165) → [GCU.verify()](control/gcu.py:184) → [GCU.rollback()](control/gcu.py:217).
- Pack schema nuances (Phase A permissive):
  CRC present when enabled [GCU._validate_pack_crc()](control/gcu.py:307); noc vs link mismatch warns (status) not fails [GCU._validate_pack_schema()](control/gcu.py:295).
- Guardrails expect [H,W,4] finite weights, per-tile sum > 0, shapes match: [_validate_weight_tensor()](control/guardrails.py:125).
- Harness emits JSON-native packs; convert numpy via [_to_native()](harness/control_loop.py:20).
- Telemetry windows fixed/aligned; percentile uses nearest-rank: [_percentile_nearest_rank()](telemetry/aggregator.py:211).
- Spec-first artifacts: make spec-first-artifacts (JUnit XML, feedback, env.json) and make spec-first-archive.