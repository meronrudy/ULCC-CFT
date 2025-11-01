# E4 Integration and Benchmarking Guide

This guide documents the end-to-end slow-plane → pack → guardrails → control apply (A/B quick‑check) flow, and how to benchmark morphogenetic adaptations against fast‑plane proxy metrics.

Banner image used in README: put the provided image at:
- docs/images/e4_overview.png
You can replace it with any PNG; the README references this path.

## Components and click-through references

- Control loop utility: [python.control_loop.control_apply_cycle()](harness/control_loop.py:125)
- Guardrails validation: [python.guardrails.validate_reconfig_pack()](control/guardrails.py:18)
- GCU stages:
  - [python.GCU.shadow_apply()](control/gcu.py:82) → [python.GCU.quick_check()](control/gcu.py:102) → [python.GCU.quiesce()](control/gcu.py:138) → [python.GCU.commit()](control/gcu.py:165) → [python.GCU.verify()](control/gcu.py:184) → optional [python.GCU.rollback()](control/gcu.py:217)
- Packer: [python.make_reconfig_pack()](slow_plane/packer/make.py:1)
- Field solver: [python.solve_field()](slow_plane/field/solver.py:387)
- Geometry update: [python.update_geometry()](slow_plane/geometry/update.py:328)

## What control_apply_cycle() does

Given a metric grid g and PGGS artifacts (U, J, B), the utility:
1) Solves field to produce Φ and ∇Φ via [python.solve_field()](slow_plane/field/solver.py:387).
2) Applies geometry update to obtain g_next + trust-region meta via [python.update_geometry()](slow_plane/geometry/update.py:328).
3) Packs a ReconfigPack with CRC-32C via [python.make_reconfig_pack()](slow_plane/packer/make.py:442).
4) Validates safety via [python.guardrails.validate_reconfig_pack()](control/guardrails.py:18).
5) Executes GCU atomic apply: [python.GCU.shadow_apply()](control/gcu.py:82) → [python.GCU.quick_check()](control/gcu.py:102) → [python.GCU.quiesce()](control/gcu.py:138) → [python.GCU.commit()](control/gcu.py:165) → [python.GCU.verify()](control/gcu.py:184), with optional rollback on verify failure.

Return schema on success:
- commit_id, pack_crc32c, trust_region_meta, quick_meta, verify_meta

Return schema on failures:
- {'stage': 'guardrails' | 'quick_check' | 'quiesce' | 'verify', 'ok': False, ...}

## Running the integration tests

Minimal integration checks:
```bash
.venv/bin/python -m pytest -q tests/control/test_control_integration.py
```

These tests validate:
- Happy path end-to-end success.
- Guardrails blocking on malformed routing weights.
- Verify failure triggering the rollback path.

## Quick benchmarking

Run a single E4 benchmark and emit a Markdown report:
- Runner: [harness/e4_bench.py](harness/e4_bench.py:1)
- Emits: harness/e4_bench_report.md

```bash
.venv/bin/python -m harness.e4_bench --grid 8x8 --iterations 50 --md harness/e4_bench_report.md
```

## Comprehensive morphogenesis sweep

Large-scale, deterministic sweep across grid sizes and geometry configs, with proxy fast‑plane metrics:
- Driver: [harness/e4_morphogenesis_bench.py](harness/e4_morphogenesis_bench.py:1)
- Output:
  - Markdown report: harness/reports/e4_morphogenesis_report.md
  - JSON summary printed to stdout (for automation)

Example:
```bash
.venv/bin/python -m harness.e4_morphogenesis_bench \
  --grids 4,8,16,32 \
  --iterations 100 \
  --fp-cycles 2000 \
  --period-us 1000000 \
  --out harness/reports/e4_morphogenesis_report.md
```

What’s inside the report:
- Slow‑plane stage overheads (PGGS / Field / Geometry / Pack and overhead % of period).
- Control‑loop timing stats (min/p50/mean/p95/max), acceptance ratios, residual norms; baseline vs adaptive trust‑region settings.
- Minimal fast‑plane proxy metrics: produced flits, served requests, avg/p95 MC latency, total energy proxy, max tile temperature.
- Comparative deltas vs baseline (p95) to identify degradation thresholds or improvements.

## Reproducibility and determinism

- All slow‑plane numerics are deterministic (no RNG required) given identical inputs.
- The packer produces stable CRC via canonicalized JSON and float rounding.
- Guardrails default to permissive fairness in Phase A for tiny synthetic fixtures; adjust [python.guardrails.GuardrailConfig()](control/guardrails.py:8) per needs.
- GCU quick-check uses a structural sanity screen and enforces regression budgets based on [python.GCUConfig.max_sim_regression_pct](control/gcu.py:33).

## Adding the README banner image

- Save the provided image to:
  - docs/images/e4_overview.png
- The README already references this path. If you rename it, update the README banner:
  ```markdown
  ![Morphogenesis Overview](docs/images/e4_overview.png)
  ```

## Troubleshooting

- Missing symlinks for import-name vs folder-name:
  - Run `make test` or manually `ln -s ulcc-core ulcc_core; ln -s ulcc-ddg ulcc_ddg; ln -s pggs-toy ulcc_pggs`.
- CRC missing:
  - Ensure packer config has `crc_enable=True`, see [python.PackerConfig](slow_plane/packer/make.py:77).
- Guardrails failing on routing:
  - Check weights tensors have shape [H,W,4] and non-negative entries; see [python.guardrails._validate_weight_tensor()](control/guardrails.py:125).
- Quiesce not entered:
  - Lower `max_vc_depth` in telemetry below [python.GCUConfig.quiesce_vc_drain_threshold](control/gcu.py:33).

## References

- Control-loop utility: [harness/control_loop.py](harness/control_loop.py:1)
- Benchmarks: [harness/e4_bench.py](harness/e4_bench.py:1), [harness/e4_morphogenesis_bench.py](harness/e4_morphogenesis_bench.py:1)
- Guardrails: [control/guardrails.py](control/guardrails.py:1)
- GCU: [control/gcu.py](control/gcu.py:1)
- Packer: [slow_plane/packer/make.py](slow_plane/packer/make.py:1)
- Field / Geometry: [slow_plane/field/solver.py](slow_plane/field/solver.py:1), [slow_plane/geometry/update.py](slow_plane/geometry/update.py:1)