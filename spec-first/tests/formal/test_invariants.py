"""Formal invariants tests for Fisher information metrics.

Deterministic tests:
- Symmetry and PD for Bernoulli Fisher.
- Reparameterization invariance under phi=logit(theta).
- Empirical Fisher approximates analytic Fisher for Bernoulli.
"""

import os
import sys
import numpy as np

# Ensure 'spec-first/' is importable so 'from geom.fisher import ...' works.
THIS_DIR = os.path.dirname(__file__)
SPEC_FIRST_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if SPEC_FIRST_ROOT not in sys.path:
    sys.path.insert(0, SPEC_FIRST_ROOT)

from geom.fisher import fisher_bernoulli, empirical_fisher


def test_metric_symmetry_pd_bernoulli() -> None:
    theta = 0.3
    g = fisher_bernoulli(theta)
    assert g.shape == (1, 1)
    # Symmetry
    assert np.allclose(g, g.T, atol=1e-12)
    # Positive definiteness
    w = np.linalg.eigvalsh(g)
    assert np.all(w > 0)


def _sigmoid(phi: float) -> float:
    return 1.0 / (1.0 + np.exp(-phi))


def test_reparameterization_invariance_bernoulli() -> None:
    theta = 0.3
    # logit and logistic maps
    phi = np.log(theta / (1.0 - theta))
    J = _sigmoid(phi) * (1.0 - _sigmoid(phi))  # dθ/dφ = σ(φ)(1−σ(φ)) = θ(1−θ)

    # Fisher in θ-param
    F_theta = fisher_bernoulli(theta)[0, 0]

    # Fisher in φ-param analytically: Var_x[x - θ] = θ(1−θ)
    F_phi_analytic = theta * (1.0 - theta)

    # Reparameterization rule: F_φ = J^T F_θ J (scalar => J^2 F_θ)
    F_phi_from_transform = (J ** 2) * F_theta

    assert np.allclose(F_phi_from_transform, F_phi_analytic, atol=1e-5, rtol=1e-5)


def test_empirical_matches_analytic_bernoulli() -> None:
    theta = 0.3
    N = 10_000
    rng = np.random.default_rng(0)
    x = rng.binomial(n=1, p=theta, size=N)

    # Score for Bernoulli w.r.t θ: s_i = (x_i − θ) / (θ (1 − θ))
    scores = (x - theta) / (theta * (1.0 - theta))

    F_emp = empirical_fisher(scores)
    F_emp_scalar = float(F_emp[0, 0])

    F_analytic = 1.0 / (theta * (1.0 - theta))

    assert np.isclose(F_emp_scalar, F_analytic, rtol=1e-2, atol=1e-3)