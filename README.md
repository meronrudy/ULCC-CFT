# Universal Law of Curved Computation (ULCC) and Computational Field Theory (CFT)
A Minimal, Reproducible Reference Implementation with PGGS and Discrete Differential Geometry

![E4 Morphogenesis Overview](docs/images/e4_overview.png)

Abstract
This repository presents a compact, testable reference for the Universal Law of Curved Computation (ULCC) and its operationalization via Computational Field Theory (CFT). We instantiate a small but complete stack: (i) smooth information geometry over probabilistic models (Fisher metric, Christoffel symbols, natural-gradient dynamics), (ii) a discrete differential geometry (DDG) scaffold for transport and holonomy on metric graphs, (iii) a toy noncommutative hypergraph sampler (PGGS), and (iv) an E4 “morphogenesis” integration harness implementing a deterministic control protocol with telemetry, guardrails, packing, and report generation. The implementation is CPU-only, deterministic, and fully regression-tested; all components are built to be readable, modifiable, and pedagogically useful for researchers. While the scope targets Bernoulli/Gaussian exemplars and graph-based DDG stubs, the architecture admits natural extension to richer models, manifolds, and pipelines.

Keywords
ULCC; CFT; Information Geometry; Fisher Metric; Natural Gradient; Discrete Differential Geometry; Parallel Transport; Holonomy; PGGS; Control Theory; Morphogenesis; Reproducibility

1. Introduction
Computational processes are geometrically constrained by their dynamics, resources, and information structure. The Universal Law of Curved Computation (ULCC) posits that computation unfolds on curved spaces induced by statistical structure, physical constraints, and control. Computational Field Theory (CFT) provides a unified lens: fields encode computational state, geometry encodes lawful evolution, and control orchestrates morphogenesis—structured adaptation under constraints.

This repository offers a minimal reference implementation that is small enough to audit in full but broad enough to demonstrate end-to-end integration:

- Smooth information geometry core (Fisher metric, Christoffel symbols, natural-gradient flow).
- Discrete differential geometry (metric graphs with transport/holonomy stubs that preserve invariants).
- PGGS toy sampler over noncommutative hypergraphs.
- Deterministic E4 control loop (shadow → quick-check → quiesce → commit → verify), guardrails, packer with CRC-32C, and reproducible benchmarks/reports.

Core components (code-level)
- Smooth geometry (ULCC core):
  - [ulcc_core/fisher.py](ulcc_core/fisher.py:1)
  - [ulcc_core/christoffel.py](ulcc_core/christoffel.py:1)
  - [ulcc_core/coords.py](ulcc_core/coords.py:1)
  - [ulcc_core/dynamics.py](ulcc_core/dynamics.py:1)
- Discrete differential geometry:
  - [ulcc_ddg/metric_graph.py](ulcc_ddg/metric_graph.py:1)
  - [ulcc_ddg/transport.py](ulcc_ddg/transport.py:1)
  - [ulcc_ddg/holonomy.py](ulcc_ddg/holonomy.py:1)
- PGGS toy sampler and algebra:
  - [pggs-toy/algebra.py](pggs-toy/algebra.py:1)
  - [pggs-toy/sampler.py](pggs-toy/sampler.py:1)
  - [pggs-toy/action.py](pggs-toy/action.py:1)
- E4 integration harness, control, telemetry:
  - [harness/control_loop.py](harness/control_loop.py:1)
  - [control/gcu.py](control/gcu.py:1)
  - [control/guardrails.py](control/guardrails.py:1)
  - [slow_plane/packer/make.py](slow_plane/packer/make.py:1)
  - [slow_plane/field/solver.py](slow_plane/field/solver.py:1)
  - [slow_plane/geometry/update.py](slow_plane/geometry/update.py:1)
  - [telemetry/aggregator.py](telemetry/aggregator.py:1)

Spec-first mirror (didactic versions)
For pedagogy and unit tests, a parallel “spec-first” layer mirrors core geometry and dynamics:
- [spec-first/geom/fisher.py](spec-first/geom/fisher.py:1)
- [spec-first/geom/christoffel.py](spec-first/geom/christoffel.py:1)
- [spec-first/geom/transport.py](spec-first/geom/transport.py:1)
- [spec-first/dynamics/geodesic.py](spec-first/dynamics/geodesic.py:1)
- [spec-first/tests/formal/test_fisher_empirical.py](spec-first/tests/formal/test_fisher_empirical.py:1), etc.

2. Background and Related Work
Information Geometry: Statistical models form Riemannian manifolds with metric given by the Fisher information. Natural gradient (Amari) yields steepest descent with respect to this metric, often improving stability and invariance.
Differential Geometry in Computation: Transport, curvature, and holonomy quantify how local constraints accumulate over global structures. DDG offers discrete analogues suitable for graphs and meshes.
Noncommutative Hypergraphs and PGGS: Computation over structured, ordered interactions (paths) benefits from hypergraph formalization; PGGS instantiates a toy sampler designed for determinism and testing, exposing control of randomness and attribution structure.

This implementation aligns with these themes and composes them into a controlled morphogenesis loop (E4), connecting statistical geometry (ULCC), field-theoretic interpretation (CFT), and concrete control/telemetry.

3. Formalism and Implementation

3.1 Statistical Manifolds and Fisher Geometry
We parameterize families such as Bernoulli and Gaussian and compute:
- Fisher metric g(θ).
- Christoffel symbols Γ(θ) derived from g(θ).
- Natural-gradient dynamics and geodesic RHS.

Code: [ulcc_core/fisher.py](ulcc_core/fisher.py:1), [ulcc_core/christoffel.py](ulcc_core/christoffel.py:1), [ulcc_core/dynamics.py](ulcc_core/dynamics.py:1), coordinate transforms in [ulcc_core/coords.py](ulcc_core/coords.py:1).

Design rules enforced by tests and guards:
- θ strictly in (0,1) for Bernoulli; boundary avoidance (domain guards).
- Scalars returned as native float; vectors as NumPy float arrays.
- Alias preservation (Gamma) for clarity.

3.2 Discrete Differential Geometry (DDG) Scaffold
We provide a minimal metric graph structure and explicit stubs for transport and holonomy to isolate invariants and enable controlled experiments:
- Metric graph: [ulcc_ddg/metric_graph.py](ulcc_ddg/metric_graph.py:1)
- Transport: [ulcc_ddg/transport.py](ulcc_ddg/transport.py:1)
- Holonomy: [ulcc_ddg/holonomy.py](ulcc_ddg/holonomy.py:1)

Stubs intentionally fix transport to identity and holonomy to zero under identity transport. This preserves baseline invariants and supports regression tests; it is a starting point for richer discrete connections.

3.3 PGGS Toy Sampler on Hypergraphs
The toy PGGS layer exercises noncommutative path sampling with a deterministic RNG threading discipline (np.random.default_rng(0) in tests). This ensures exact reproducibility and simplifies attribution analyses.
- Algebra/spec: [pggs-toy/algebra.py](pggs-toy/algebra.py:1)
- Sampler: [pggs-toy/sampler.py](pggs-toy/sampler.py:1)
- Action/guide: [pggs-toy/action.py](pggs-toy/action.py:1)

3.4 E4 Morphogenesis: Control, Guardrails, Packing
The E4 loop orchestrates field solve, geometry update, pack/validate, and controlled apply:
- Control loop: [harness/control_loop.py](harness/control_loop.py:1)
- Guardrails (shape, finiteness, per-tile sum, contract checks): [control/guardrails.py](control/guardrails.py:1)
- GCU protocol (shadow → quick-check → quiesce → commit → verify → rollback): [control/gcu.py](control/gcu.py:1)
- Deterministic packer with CRC-32C: [slow_plane/packer/make.py](slow_plane/packer/make.py:1)
- Field solver and trust-region geometry update: [slow_plane/field/solver.py](slow_plane/field/solver.py:1), [slow_plane/geometry/update.py](slow_plane/geometry/update.py:1)
- Telemetry windowing and aggregation (nearest-rank percentiles, aligned windows): [telemetry/aggregator.py](telemetry/aggregator.py:1)

4. Methods

4.1 Smooth-Core Demonstrations
We implement empirical checks that Fisher metrics and connections behave under reparameterization and that natural-gradient steps respect invariants. Spec-first tests cover exact small cases for Bernoulli/Gaussian:
- [spec-first/tests/formal/test_fisher_empirical.py](spec-first/tests/formal/test_fisher_empirical.py:1)
- [spec-first/tests/formal/test_christoffel_dg.py](spec-first/tests/formal/test_christoffel_dg.py:1)
- [spec-first/tests/formal/test_transport_order.py](spec-first/tests/formal/test_transport_order.py:1)

4.2 DDG Baselines
Identity transport and zero holonomy define the “flat baseline”. This isolates control/telemetry behavior and pack/guardrail correctness from geometric complexity. Later work can substitute nontrivial connections and curvature.

4.3 PGGS Sampling Protocol
The sampler uses a threaded RNG to ensure determinism across runs and across control cycles. Tests validate variance, attribution properties, and reproducibility.

4.4 E4 Control Protocol
We formalize the staged apply sequence and validate via integration tests. Shape/contract violations are surfaced by guardrails before any commit, and the GCU emulator simulates realistic orchestration states.

5. Experiments

5.1 Fast E4 Benchmark
A single-configuration benchmark produces a concise performance and overhead snapshot:
- Runner: [harness/e4_bench.py](harness/e4_bench.py:1)
- Markdown report output: harness/e4_bench_report.md

5.2 Morphogenesis Sweep
We sweep grids and iterations to study overheads and latency distributions, comparing baseline vs adaptive configurations:
- Driver: [harness/e4_morphogenesis_bench.py](harness/e4_morphogenesis_bench.py:1)
- Output: [harness/reports/e4_morphogenesis_report.md](harness/reports/e4_morphogenesis_report.md:1)

5.3 Telemetry and Contracts
Telemetry frames and validators ensure window alignment, schema adherence, and nearest-rank percentile semantics:
- [telemetry/frame.py](telemetry/frame.py:1)
- [telemetry/validator.py](telemetry/validator.py:1)

6. Results (Representative)
- Overheads: The slow-plane (Field/Geometry/Pack) overhead is quantified relative to total cycle time; CRC-32C pack adds minimal deterministic cost.
- Control timings: Shadow/quick-check/quiesce/commit/verify latencies summarized with min/p50/p95/max; acceptance rates and residuals reported for trust-region steps.
- Fast-plane proxies: Minimal produced/served, p95 latency, power, and thermal proxy metrics recorded for comparative analysis.

See generated artifacts:
- Single-bench report: [harness/e4_bench_report.md](harness/e4_bench_report.md:1)
- Morphogenesis report: [harness/reports/e4_morphogenesis_report.md](harness/reports/e4_morphogenesis_report.md:1)

7. Reproducibility and Testing
We emphasize deterministic, CPU-only tests that run in minutes:
- Acceptance and integration tests: [tests/control/test_control_integration.py](tests/control/test_control_integration.py:1), [tests/acceptance/test_phaseA_e1_acceptance.py](tests/acceptance/test_phaseA_e1_acceptance.py:1)
- Benchmarks smoke tests: [tests/benchmarks/test_e4_bench.py](tests/benchmarks/test_e4_bench.py:1), [tests/benchmarks/test_e4_bench_report.py](tests/benchmarks/test_e4_bench_report.py:1)
- Fast-plane determinism and scheduling: [tests/fast_plane/test_determinism_seed.py](tests/fast_plane/test_determinism_seed.py:1), [tests/fast_plane/test_scheduler_priority_preemption.py](tests/fast_plane/test_scheduler_priority_preemption.py:1), etc.
- Telemetry validations: [tests/telemetry/test_telemetry_validations.py](tests/telemetry/test_telemetry_validations.py:1)

Agent and coding rules (non-obvious, enforced by CI and docs):
- See [AGENTS.md](AGENTS.md:1) for import-name symlinks, domain guards for θ∈(0,1), intentional stubs, type discipline, RNG threading, and control protocol order.

8. Discussion
This implementation validates that a principled geometric view of computation (ULCC) can be embodied in a compact, deterministically testable system. The CFT perspective serves as an organizing scaffold: smooth statistical geometry for local laws, DDG for structured transport/holonomy in discrete substrates, and a control loop (E4) to regulate evolution with explicit guardrails and telemetry.

Limitations and opportunities:
- Smooth core presently targets simple families (Bernoulli, Gaussian) but the interfaces generalize.
- DDG transport/holonomy are intentionally conservative stubs; extending to nontrivial connections and curvature is the natural next step.
- PGGS is a toy sampler designed for determinism; richer guide/action semantics can be layered with the same RNG discipline.

9. Conclusion
ULCC and CFT offer a compositional architecture for lawful computation on curved spaces. This repository demonstrates a minimal, research-grade baseline where every moving part is transparent, testable, and extensible. We hope this serves as a foundation for more ambitious experimental programs linking geometry, control, and complex computation.

Appendix A: Quick Start
Environment
- Python 3.9+ (recommend venv), CPU-only.

Make-driven workflow (creates import symlinks per AGENTS.md):
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make test
```

Manual fallback:
```bash
ln -s ulcc-core ulcc_core; ln -s ulcc-ddg ulcc_ddg; ln -s pggs-toy ulcc_pggs
pytest -q
```

Docker:
```bash
docker build -t ulcc:dev -f docker/Dockerfile . && docker run --rm ulcc:dev
```

Appendix B: Repository Layout (selected)
- ULCC smooth core:
  - [ulcc_core/](ulcc_core/__init__.py:1) — Fisher metric, Christoffel, dynamics, coords
- DDG scaffold:
  - [ulcc_ddg/](ulcc_ddg/__init__.py:1) — metric graph, transport, holonomy
- PGGS toy:
  - [pggs-toy/](pggs-toy/__init__.py:1) — algebra, sampler, action
- Harness and control:
  - [harness/](harness/control_loop.py:1), [control/](control/gcu.py:1)
- Slow plane (field, geometry, packer):
  - [slow_plane/](slow_plane/__init__.py:1)
- Telemetry:
  - [telemetry/](telemetry/__init__.py:1)
- Spec-first pedagogy:
  - [spec-first/](spec-first/README.md:1)
- Docs and images:
  - [docs/Guides/E4-Integration-Guide.md](docs/Guides/E4-Integration-Guide.md:1), [docs/images/e4_overview.PNG](docs/images/e4_overview.PNG:1)
- Papers:
  - [CFT-ULCC-papers-pdf/CFT Computational Field Theory Explained.pdf](CFT-ULCC-papers-pdf/CFT Computational Field Theory Explained.pdf:1)
  - [CFT-ULCC-papers-pdf/CFT Unified Computational Framework Synthesis.pdf](CFT-ULCC-papers-pdf/CFT Unified Computational Framework Synthesis.pdf:1)
  - [CFT-ULCC-papers-pdf/ULCC: PGGS Formalization and Integration.pdf](CFT-ULCC-papers-pdf/ULCC: PGGS Formalization and Integration.pdf:1)
  - [CFT-ULCC-papers-pdf/CFT Roadmap: Theory, Code, Publications.pdf](CFT-ULCC-papers-pdf/CFT Roadmap: Theory, Code, Publications.pdf:1)
  - [CFT-ULCC-papers-pdf/Dynamic Causal Geometry Framework.pdf](CFT-ULCC-papers-pdf/Dynamic Causal Geometry Framework.pdf:1)
  - [CFT-ULCC-papers-pdf/Law of Computational Geometry.pdf](CFT-ULCC-papers-pdf/Law of Computational Geometry.pdf:1)

Appendix C: Design Rules (selection; see AGENTS.md for full)
- Import-name vs folder-name symlinks created at repo root and inside Docker.
- RNG determinism: thread numpy.default_rng through PGGS.
- Domain guards: θ ∈ (0,1) strictly for relevant APIs.
- Stub semantics are intentional (identity transport, zero holonomy).
- Types: scalars → float; vectors → np.ndarray[float]; preserve alias “Gamma”.
- Control protocol order is strict: shadow → quick-check → quiesce → commit → verify → rollback.
- Pack/guardrails validate shape, finiteness, and per-tile weight sums; numpy→JSON-native conversion before externalization.

Acknowledgments
We thank contributors and reviewers whose feedback shaped this minimal, testable baseline.

Citation
Please cite via [CITATION.cff](CITATION.cff:1).

License
Apache-2.0 (see [LICENSE](LICENSE:1)).
