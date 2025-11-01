# CFT+ULCC Spec-first Scaffold

Purpose
- Background-independent computational geometry system with dual timescales:
  - Fast: causal field dynamics on a fixed metric g_t (data-driven causal updates).
  - Slow: geometric evolution of g via CFE/Ricci-flow-style updates steered by the causal field Π(𝕀).
- This repository subtree is a spec-first scaffold. It encodes architecture, invariants, and validation targets before algorithms.

## Architecture Overview

Directories and roles:
- geom/: metric and differential geometry scaffolds
  - [spec-first/geom/fisher.py](spec-first/geom/fisher.py): Fisher metric (SPD).
  - [spec-first/geom/christoffel.py](spec-first/geom/christoffel.py): Levi–Civita connection.
  - [spec-first/geom/transport.py](spec-first/geom/transport.py): Parallel transport.
  - [spec-first/geom/holonomy.py](spec-first/geom/holonomy.py): Loop holonomy.
  - [spec-first/geom/cfe_update.py](spec-first/geom/cfe_update.py): CFE/Ricci-flow metric update.
  - [spec-first/geom/curvature.py](spec-first/geom/curvature.py): Curvature tensors/operators.
  - [spec-first/geom/operators.py](spec-first/geom/operators.py): grad/div/Laplace–Beltrami (discrete).
  - [spec-first/geom/reparameterize.py](spec-first/geom/reparameterize.py): coordinate transforms.
- dynamics/: variational and flow dynamics
  - [spec-first/dynamics/lagrangian.py](spec-first/dynamics/lagrangian.py): action/Euler–Lagrange.
  - [spec-first/dynamics/geodesic.py](spec-first/dynamics/geodesic.py): geodesic RHS.
  - [spec-first/dynamics/overdamped.py](spec-first/dynamics/overdamped.py): overdamped/NGD limit.
- field/: metric-aware field equations
  - [spec-first/field/wave.py](spec-first/field/wave.py): wave equation on discrete manifold.
- pggs/: probabilistic geometric graph simulator
  - [spec-first/pggs/algebra.py](spec-first/pggs/algebra.py), [spec-first/pggs/hypergraph.py](spec-first/pggs/hypergraph.py), [spec-first/pggs/guide.py](spec-first/pggs/guide.py), [spec-first/pggs/sampler.py](spec-first/pggs/sampler.py), [spec-first/pggs/export.py](spec-first/pggs/export.py).
- utils/: small utilities (e.g., logging stub).
- tests/: formal tests and fixtures (to be added next).
- notebooks/: exploration notebooks (empty keeper added).
- examples/: runnable stubs, e.g., [spec-first/examples/run_feedback_loop.py](spec-first/examples/run_feedback_loop.py).

## Invariants (spec-level)

Geometry
- Fisher metric is symmetric and positive-definite (SPD); results are coordinate-free under reparameterizations.
- Levi-Civita compatibility (∇g = 0) and torsion-free connection; parallel transport preserves inner product.
- Flat-loop holonomy ≈ identity (within target tolerance).
- Discrete operators respect adjointness (grad/div) and self-adjointness of Laplace–Beltrami under metric inner product.

Dynamics
- Lagrangian dynamics extremize action (Euler–Lagrange conditions hold).
- Overdamped limit reduces to natural gradient descent; energy is monotone where applicable.
- Geodesic evolution uses the Levi–Civita connection derived from the current metric.

Field
- Wave equation is posed on a metric-aware discrete manifold; time stepping is well-posed (CFL-like).

PGGS
- Noncommutativity is captured where required; symmetric B and divergence-free current checks included.
- Guided sampling reduces variance versus uniform baseline.
- Determinism: samplers must accept and use an explicit rng.

CFE / Geometry Update
- Residual for the Causal Field Equation decreases over steps.
- Ricci-flow-based update remains consistent with the Π(𝕀) source.

## Module Mapping (theory → files)

- Riemannian metric (statistical manifold): [spec-first/geom/fisher.py](spec-first/geom/fisher.py)
- Levi–Civita and operators: [spec-first/geom/christoffel.py](spec-first/geom/christoffel.py), [spec-first/geom/operators.py](spec-first/geom/operators.py)
- Parallel transport and holonomy: [spec-first/geom/transport.py](spec-first/geom/transport.py), [spec-first/geom/holonomy.py](spec-first/geom/holonomy.py)
- Curvature: [spec-first/geom/curvature.py](spec-first/geom/curvature.py)
- Reparameterizations: [spec-first/geom/reparameterize.py](spec-first/geom/reparameterize.py)
- Lagrangian / geodesic / overdamped dynamics: [spec-first/dynamics/lagrangian.py](spec-first/dynamics/lagrangian.py), [spec-first/dynamics/geodesic.py](spec-first/dynamics/geodesic.py), [spec-first/dynamics/overdamped.py](spec-first/dynamics/overdamped.py)
- Metric-aware field equation: [spec-first/field/wave.py](spec-first/field/wave.py)
- PGGS algebra/structures/sampling/export: files under [spec-first/pggs/](spec-first/pggs/__init__.py)

## Validation Matrix Summary (targets)

- Reparameterization invariance error: < 1e-5.
- Flat-loop holonomy deviation from identity: < 1e-6.
- Overdamped vs NGD agreement: ≤ 1% (relative).
- CFE residual reduction: ≥ 10× decrease per step (relative).
- PGGS guided variance reduction vs uniform: ≥ 50%.

These are measurable acceptance criteria to be enforced in tests under spec-first/tests/formal/.

## Determinism and Reproducibility

- Always pass an explicit rng; tests seed with `np.random.default_rng(0)`.
- Prefer structure-preserving DDG discretizations to maintain invariants (adjointness, SPD).
- Avoid global state and implicit seeds in sampling or geometry updates.

## Dependencies

This scaffold expects the following Python packages (pinned in [requirements.txt](requirements.txt)):
- numpy (numerics)
- sympy (symbolics/verification)
- deal (contracts/specs)
- pytest (tests)
- networkx (graphs/hypergraphs)
- jupyter (notebooks)

## Environment Notes

- The ulcc_* packages and symlink gotchas described in [AGENTS.md](AGENTS.md) do not apply to this subtree: spec-first is standalone and does not import the legacy packages.
- CI/Makefile targets for spec-first will be added in a follow-up task; do not modify the existing root Makefile or Dockerfile in this subtask.

## Usage Roadmap

1) Implement the module internals per the above invariants.
2) Add formal tests under spec-first/tests/formal/ matching the Validation Matrix.
3) Provide demonstration notebooks under spec-first/notebooks/.
4) Wire an example feedback loop in [spec-first/examples/run_feedback_loop.py](spec-first/examples/run_feedback_loop.py).

## Acceptance Criteria and Validation Matrix

- Reparameterization invariance: numeric error < 1e-5 → [spec-first/tests/formal/test_invariants.py](spec-first/tests/formal/test_invariants.py)
- Flat holonomy error: < 1e-6 → [spec-first/tests/formal/test_holonomy_flat.py](spec-first/tests/formal/test_holonomy_flat.py)
- Overdamped limit agreement: < 1% vs NGD → [spec-first/tests/formal/test_overdamped_limit.py](spec-first/tests/formal/test_overdamped_limit.py)
- CFE residual reduction: ≥ 10× per step → [spec-first/tests/formal/test_cfe_residual.py](spec-first/tests/formal/test_cfe_residual.py)
- PGGS variance reduction: ≥ 50% guided vs uniform → [spec-first/tests/formal/test_pggs_variance.py](spec-first/tests/formal/test_pggs_variance.py)
- Wave energy conservation (J=0): relative drift ≤ 1e-3 → [spec-first/tests/formal/test_wave.py](spec-first/tests/formal/test_wave.py)
- End-to-end PGGS→CFE geometry update: ≥ 10× residual reduction, nontrivial metric change → [spec-first/tests/formal/test_end_to_end.py](spec-first/tests/formal/test_end_to_end.py)

## Determinism and RNG Policy

- All stochastic routines accept an explicit numpy Generator (np.random.Generator); tests use np.random.default_rng(0) to ensure repeatability in CI and locally.
- No global RNG use in library code; seeding is controlled by callers/tests for deterministic behavior.
- Modules already adhering: [spec-first/pggs/sampler.py](spec-first/pggs/sampler.py), [spec-first/tests/formal/test_pggs_variance.py](spec-first/tests/formal/test_pggs_variance.py).
