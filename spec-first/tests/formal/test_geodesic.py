"""Formal tests for geodesic solver on flat manifolds (Euclidean metric, zero connection)."""

import os
import sys
import numpy as np

# Ensure 'spec-first/' is importable so 'from dynamics.geodesic import ...' works.
THIS_DIR = os.path.dirname(__file__)
SPEC_FIRST_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if SPEC_FIRST_ROOT not in sys.path:
    sys.path.insert(0, SPEC_FIRST_ROOT)

from dynamics.geodesic import shoot_geodesic, geodesic_step


def _discrete_path_length(path: np.ndarray, g: np.ndarray) -> float:
    """
    Compute discrete path length L(path) = sum_k sqrt(dθ_k^T g dθ_k),
    where dθ_k = path[k+1] - path[k].
    """
    diffs = path[1:] - path[:-1]  # shape (m, n)
    # quadratic form per segment
    sq = np.einsum("bi,ij,bj->b", diffs, g, diffs)  # shape (m,)
    return float(np.sum(np.sqrt(sq)))


def test_geodesic_flat_reaches_target() -> None:
    # Setup: flat space with identity metric and zero connection
    n = 3
    g = np.eye(n, dtype=float)
    Gamma = np.zeros((n, n, n), dtype=float)

    theta0 = np.array([0.0, 1.0, -0.5], dtype=float)
    theta1 = np.array([1.0, -2.0, 0.5], dtype=float)
    T = 1.0
    steps = 200

    path, v0 = shoot_geodesic(theta0, theta1, g, Gamma, T=T, steps=steps, tol=1e-10)

    # Endpoints must match exactly (flat case, constant velocity integration)
    assert np.allclose(path[0], theta0, atol=1e-12)
    assert np.allclose(path[-1], theta1, atol=1e-9)

    # Straightness: successive finite differences should be constant
    diffs = path[1:] - path[:-1]
    max_dev = np.max(np.linalg.norm(diffs - diffs[0], axis=1))
    assert max_dev <= 1e-12


def test_geodesic_flat_straight_length_minimal() -> None:
    # Setup: same flat manifold
    n = 3
    g = np.eye(n, dtype=float)
    Gamma = np.zeros((n, n, n), dtype=float)

    theta0 = np.array([0.0, 1.0, -0.5], dtype=float)
    theta1 = np.array([1.0, -2.0, 0.5], dtype=float)
    T = 1.0
    steps = 200
    dt = T / steps

    # Integrate the straight geodesic via geodesic_step with v0 = (theta1 - theta0)/T
    v0 = (theta1 - theta0) / T
    path_g = np.empty((steps + 1, n), dtype=float)
    path_g[0] = theta0
    theta = theta0.copy()
    v = v0.copy()
    for k in range(steps):
        theta, v = geodesic_step(theta, v, g, Gamma, dt)
        path_g[k + 1] = theta

    # Construct a small sinusoidal perturbation that vanishes at endpoints
    eps = 1e-3
    t_grid = np.linspace(0.0, 1.0, steps + 1)
    e1 = np.zeros(n, dtype=float)
    e1[0] = 1.0  # unit vector along first coordinate
    delta = (eps * np.sin(2.0 * np.pi * t_grid))[:, None] * e1[None, :]
    path_p = path_g + delta

    # Discrete path length under metric g
    L_g = _discrete_path_length(path_g, g)
    L_p = _discrete_path_length(path_p, g)

    # Straight path should be strictly shorter by a clear margin
    assert (L_p - L_g) >= 1e-6