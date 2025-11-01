"""Deterministic test: high-γ Lagrangian reduces to NGD (≤1% deviation)."""

import os
import sys
import numpy as np

# Ensure 'spec-first/' is importable so 'from dynamics.lagrangian import ...' works.
THIS_DIR = os.path.dirname(__file__)
SPEC_FIRST_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if SPEC_FIRST_ROOT not in sys.path:
    sys.path.insert(0, SPEC_FIRST_ROOT)

from dynamics.lagrangian import lagrangian_step
from dynamics.overdamped import ngd_step


def test_overdamped_limit_agreement() -> None:
    # Quadratic potential V(theta) = 0.5 * theta^T A theta with diag A.
    A = np.diag([1.0, 2.0, 0.5])

    def gradV(theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        return A @ theta

    n = 3
    g = np.eye(n, dtype=float)
    Gamma = np.zeros((n, n, n), dtype=float)

    theta0 = np.array([1.0, -0.5, 2.0], dtype=float)
    v0 = np.zeros(n, dtype=float)

    gamma = 100.0
    dt = 0.01
    steps = 500
    lr = dt / gamma

    # Second-order with friction
    theta_lagr = theta0.copy()
    v = v0.copy()
    for _ in range(steps):
        theta_lagr, v = lagrangian_step(theta_lagr, v, g, gradV, Gamma, gamma, dt)

    # Overdamped NGD
    theta_ngd = theta0.copy()
    for _ in range(steps):
        theta_ngd = ngd_step(theta_ngd, g, gradV, lr)

    # Relative agreement within 1%
    diff = theta_lagr - theta_ngd
    rel = float(np.linalg.norm(diff) / max(1e-12, np.linalg.norm(theta_ngd)))
    assert rel <= 0.01 + 1e-12