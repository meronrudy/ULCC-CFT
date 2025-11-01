"""Formal tests for reparameterization utilities and pullback metric."""

import os
import sys
import numpy as np

# Ensure 'spec-first/' is importable so 'from geom.reparameterize import ...' works.
THIS_DIR = os.path.dirname(__file__)
SPEC_FIRST_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if SPEC_FIRST_ROOT not in sys.path:
    sys.path.insert(0, SPEC_FIRST_ROOT)

from geom.fisher import fisher_bernoulli
from geom.reparameterize import reparam_theta_to_phi, pullback_metric


def test_pullback_metric_scalar_bernoulli() -> None:
    # Scalar Bernoulli case
    theta = 0.3
    g_theta = fisher_bernoulli(theta)  # shape (1,1)

    # Reparameterize θ -> φ via logit, and get J = dθ/dφ
    phi, J = reparam_theta_to_phi(np.array([theta], dtype=float))  # shapes (1,), (1,)

    # Pull back metric using diagonal J (1-D accepted)
    g_phi = pullback_metric(g_theta, J)

    # Expected: scalar pullback g_φ = (J^2) * g_θ
    expected = (J[0] ** 2) * g_theta
    assert g_phi.shape == (1, 1)
    assert np.allclose(g_phi, expected, atol=1e-10, rtol=1e-10)


def test_pullback_metric_vector_diag_bernoulli() -> None:
    # Vector Bernoulli parameters; diagonal metric in θ-coordinates
    thetas = np.array([0.2, 0.3, 0.7], dtype=float)
    g_diag = np.array([fisher_bernoulli(t)[0, 0] for t in thetas], dtype=float)
    g_theta = np.diag(g_diag)

    # Reparameterize componentwise and get diagonal Jacobian entries
    phi, J_vec = reparam_theta_to_phi(thetas)

    # Pull back under diagonal J
    g_phi = pullback_metric(g_theta, J_vec)

    # Expected diagonal entries: (J_i^2) * g_θ,ii
    expected_diag = (J_vec ** 2) * g_diag

    # Check symmetry and diagonal relationship
    assert np.allclose(g_phi, g_phi.T, atol=1e-12, rtol=1e-12)
    assert np.allclose(np.diag(g_phi), expected_diag, atol=1e-10, rtol=1e-10)