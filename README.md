# ULCC Phase-0: Minimal Reproducible Implementation

![Morphogenesis Overview](docs/images/e4_overview.png)

This repository provides a compact, testable skeleton for the **Universal Law of Curved Computation (ULCC)** and **PGGS**.
It focuses on the Bernoulli/Gaussian toy cases, a minimal **discrete differential geometry (DDG)** scaffold, and a **PGGS** toy sampler.

The Phase-A “E4 integration” is implemented end-to-end:
- Slow-plane field solve and geometry update
- Deterministic packer with CRC-32C
- Guardrails validation
- Control Apply via GCU (shadow → quick-check → quiesce → commit → verify)
- Benchmarks and comprehensive morphogenesis report generation

See: [harness/control_loop.py](harness/control_loop.py:1), [tests/control/test_control_integration.py](tests/control/test_control_integration.py:1).

> Image note: place the attached image at docs/images/e4_overview.png (created if missing) so the banner renders locally and on Git hosting.

## Layout
```
ulcc-core/         # smooth geometry: Fisher metric, Christoffel, dynamics, coords
ulcc-ddg/          # discrete differential geometry scaffolding (metric graph, transport, holonomy)
pggs-toy/          # noncommutative hypergraph attribution (toy path-integral sampler)
harness/           # control loop, benchmarks, runners, reports
docs/              # guides and reports
docker/            # dockerization
```

## Quick start

- Symlink rule: Make targets enforce the import-name vs folder-name rule from [AGENTS.md](AGENTS.md). "make test" will create the required symlinks automatically.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make test
```

Manual fallback (if not using make):

```bash
ln -s ulcc-core ulcc_core; ln -s ulcc-ddg ulcc_ddg; ln -s pggs-toy ulcc_pggs
pytest -q
```

### Docker

Build and run tests in Docker:

```bash
docker build -t ulcc:dev -f docker/Dockerfile . && docker run --rm ulcc:dev
```

## E4 Integration

- Control-loop utility: [harness/control_loop.py](harness/control_loop.py:1) exposes [python.control_loop.control_apply_cycle()](harness/control_loop.py:125). It executes:
  1) Field solve → 2) Geometry update (trust-region meta) → 3) Packer (CRC-32C) → 4) Guardrails → 5) GCU shadow/quick/quiesce/commit/verify (optional rollback).
- Guardrails: [control/guardrails.py](control/guardrails.py:1)
- GCU emulator: [control/gcu.py](control/gcu.py:1)
- Packer: [slow_plane/packer/make.py](slow_plane/packer/make.py:1)
- Field & Geometry: [slow_plane/field/solver.py](slow_plane/field/solver.py:1), [slow_plane/geometry/update.py](slow_plane/geometry/update.py:1)

Smoke tests for E4 integration:
```bash
.venv/bin/python -m pytest -q tests/control/test_control_integration.py
```

## Benchmarks

Fast E4 benchmark (single configuration):
- Runner: [harness/e4_bench.py](harness/e4_bench.py:1)
- Report helper: [python.e4_bench.write_markdown_report()](harness/e4_bench.py:347)

Run:
```bash
.venv/bin/python -m harness.e4_bench --grid 8x8 --iterations 50 --md harness/e4_bench_report.md
```

Comprehensive Morphogenesis Sweep (baseline vs adaptive configs across grids):
- Driver: [harness/e4_morphogenesis_bench.py](harness/e4_morphogenesis_bench.py:1)
- Generates a comparative Markdown report per grid with:
  - Slow-plane overhead (PGGS/Field/Geometry/Pack, overhead %)
  - Control-loop timing stats (min/p50/mean/p95/max) + trust-region acceptance/residuals
  - Minimal fast-plane proxy metrics (produced, served, avg/p95 latency, power, thermal)
  - Deltas vs baseline (p95)

Run:
```bash
.venv/bin/python -m harness.e4_morphogenesis_bench \
  --grids 4,8,16,32 \
  --iterations 100 \
  --fp-cycles 2000 \
  --period-us 1000000 \
  --out harness/reports/e4_morphogenesis_report.md
```

Artifacts:
- Morphogenesis report: harness/reports/e4_morphogenesis_report.md
- Single-bench report: harness/e4_bench_report.md

## Guides

- E4 Integration and Benchmarking Guide: [docs/Guides/E4-Integration-Guide.md](docs/Guides/E4-Integration-Guide.md)

## Design goals
- Deterministic, small, CPU-only; tests run in minutes.
- Invariance checks for coordinate transforms.
- Clean path to expand into full DDG + PGGS experiments.

## License
Apache-2.0
