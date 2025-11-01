"""Formal test for CFE residual reduction (≥10× per step)."""

import os
import sys
import numpy as np

# Ensure 'spec-first/' is importable so 'from geom.cfe_update import ...' works.
THIS_DIR = os.path.dirname(__file__)
SPEC_FIRST_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if SPEC_FIRST_ROOT not in sys.path:
    sys.path.insert(0, SPEC_FIRST_ROOT)

from geom.cfe_update import cfe_residual, cfe_update_step_residual
from geom.curvature import einstein_operator_simple


def _make_spd(n: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((n, n))
    return A.T @ A + 0.5 * np.eye(n, dtype=float)


def test_cfe_residual_reduction_ge_10x() -> None:
    """
    Using the simple surrogate operator G(g)=g and alpha=0.95, one step yields
    r_new = (1 - alpha) * r_old, so ||r_new||_F = 0.05 ||r_old||_F ≤ 0.1 ||r_old||_F.
    """
    rng = np.random.default_rng(0)
    n = 4
    g0 = _make_spd(n, rng)
    I0 = rng.standard_normal((n, n))

    r0 = cfe_residual(g0, I0, kappa=1.0, operator=einstein_operator_simple)
    norm0 = float(np.linalg.norm(r0, ord="fro"))
    assert norm0 > 0.0

    g1 = cfe_update_step_residual(
        g0, I0, kappa=1.0, alpha=0.95,
        operator=einstein_operator_simple,
        target_factor=0.1,
    )

    r1 = cfe_residual(g1, I0, kappa=1.0, operator=einstein_operator_simple)
    norm1 = float(np.linalg.norm(r1, ord="fro"))
    ratio = norm1 / norm0
    assert ratio <= 0.1 + 1e-12