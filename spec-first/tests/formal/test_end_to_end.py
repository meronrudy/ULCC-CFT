"""
End-to-end integration: PGGS → CFE → geometry update.

This deterministic, NumPy-only test ties the pipeline:
PGGS (guide potentials) → sampler (softmax) → export (J, B) → curvature projection (I)
→ CFE residual + single backtracked update → ≥10× residual reduction and metric change.

Design notes
- We compute r0 with the trace-free operator for inspection (stability-oriented residual).
- We enforce the ≥10× reduction using the simple Einstein operator in the update step,
  consistent with the residual-minimization contract of cfe_update.cfe_update_step_residual().
"""

import sys
import os
import numpy as np

# Ensure 'spec-first' is on sys.path so imports like `from pggs...` resolve
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pggs.sampler import softmax_neg_potential
from pggs.guide import CausalAtlas
from pggs.export import export_results
from geom.cfe_update import cfe_residual, cfe_update_step_residual
from geom.curvature import (
    project_information_tensor,
    einstein_operator_tracefree,
    einstein_operator_simple,
)


def test_end_to_end_pggs_to_cfe_geometry_update():
    # 1) Deterministic discrete paths
    paths = list(range(9))

    # 2) Build a CausalAtlas with low potentials on {0,1,2} to emulate "hotspots"
    atlas = CausalAtlas()
    for i in paths:
        atlas.update(i, 0.0 if i in {0, 1, 2} else 3.0)

    # 3) Proposal weights via softmax of negative potential (temperature=1.0)
    q = softmax_neg_potential(paths, atlas, temperature=1.0)
    assert q.shape == (len(paths),)
    assert np.all(q > 0), "Softmax must yield strictly positive probabilities"
    assert np.isclose(np.sum(q), 1.0), "Probabilities must sum to 1"

    # 4) Deterministic mapping from path index i to s_i ∈ R^3: s_i = e_{i % 3}
    E = np.eye(3, dtype=np.float64)
    samples = [E[i % 3].copy() for i in paths]

    # 5) Export (J, B) using normalized weights q; verify B is symmetric
    J, B = export_results(samples, q)
    assert J.shape == (3,)
    assert B.shape == (3, 3)
    assert np.allclose(B, B.T, rtol=1e-12, atol=1e-12), "B must be symmetric"

    # 6) Initial geometry g0 = I_3
    g0 = np.eye(3, dtype=np.float64)

    # 7) Information-Structure tensor as symmetric projection Π(B)
    I = project_information_tensor(B)
    assert np.allclose(I, I.T, rtol=1e-12, atol=1e-12), "Projected tensor must be symmetric"

    # 8) Residual with trace-free operator (stability-oriented)
    kappa = 0.5
    r0_tracefree = cfe_residual(g0, I, kappa=kappa, operator=einstein_operator_tracefree)
    # Not asserted here; used to connect the end-to-end pipeline (PGGS → CFE).

    # 9) Single residual-minimization step.
    # Use the simple operator to enforce ≥10× reduction (target_factor=0.1).
    # α=0.95 is aggressive but safe given backtracking; no RNG used.
    g1 = cfe_update_step_residual(
        g0,
        I,
        kappa=kappa,
        alpha=0.95,
        operator=einstein_operator_simple,
        target_factor=0.1,
    )

    # 10) Post-update checks
    # - Compute residuals with the same operator used in the acceptance (simple)
    r0 = cfe_residual(g0, I, kappa=kappa, operator=einstein_operator_simple)
    r1 = cfe_residual(g1, I, kappa=kappa, operator=einstein_operator_simple)

    # - ≥10× reduction in Frobenius norm
    norm_r0 = float(np.linalg.norm(r0, ord="fro"))
    norm_r1 = float(np.linalg.norm(r1, ord="fro"))
    assert norm_r0 > 0.0
    assert norm_r1 <= 0.1 * norm_r0 + 1e-12

    # - Nontrivial geometry change
    delta_g = float(np.linalg.norm(g1 - g0, ord="fro"))
    assert delta_g > 0.0

    # - Updated geometry remains symmetric (numerically)
    assert np.allclose(g1, g1.T, rtol=1e-12, atol=1e-12)