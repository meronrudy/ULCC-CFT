"""Overdamped (natural-gradient) dynamics.

This module implements a single Natural Gradient Descent (NGD) step for
coordinates theta with respect to a Riemannian metric g and potential V
with gradient gradV.

Invariants and notes:
- For sufficiently small lr, energy decreases monotonically.
- Deterministic; numpy-only.
"""

from typing import Callable
import numpy as np


def ngd_step(
    theta: np.ndarray,
    g: np.ndarray,
    gradV: Callable[[np.ndarray], np.ndarray],
    lr: float = 0.01,
) -> np.ndarray:
    """
    Perform one Natural Gradient Descent (NGD) step:
        delta = -lr * inv(g) @ gradV(theta)
        theta_new = theta + delta

    Validation:
    - g symmetric positive-definite (eigvalsh(g) > 0); shapes consistent.
    - gradV(theta) returns shape (n,).
    - Monotonicity: grad(theta)·delta ≤ 1e-12 (numerical slack).

    Returns:
        theta_new
    """
    theta = np.asarray(theta, dtype=float)
    g = np.asarray(g, dtype=float)

    if theta.ndim != 1:
        raise ValueError("theta must be a 1-D array.")
    n = theta.shape[0]
    if g.shape != (n, n):
        raise ValueError(f"g must have shape ({n}, {n}); got {g.shape}.")
    if not (lr > 0 and np.isfinite(lr)):
        raise ValueError("lr must be a positive finite float.")
    if not np.all(np.isfinite(g)):
        raise ValueError("g must contain finite values.")
    if not np.allclose(g, g.T, atol=1e-12, rtol=0.0):
        raise ValueError("g must be symmetric.")
    eig = np.linalg.eigvalsh(g)
    if np.min(eig) <= 0.0:
        raise ValueError("g must be positive-definite (eigvalsh(g) > 0).")

    grad = np.asarray(gradV(theta), dtype=float)
    if grad.shape != (n,):
        raise ValueError(f"gradV(theta) must return shape ({n},); got {grad.shape}.")
    if not np.all(np.isfinite(grad)):
        raise ValueError("gradV(theta) must be finite.")

    # Natural gradient direction
    nat_dir = np.linalg.solve(g, grad)
    delta = -lr * nat_dir

    # Monotonicity check in the local quadratic model
    inner = float(grad.T @ delta)
    if inner > 1e-12:
        raise ValueError(
            f"Monotonicity violated: grad(theta)·delta = {inner} > 1e-12. "
            "Reduce lr or ensure g is SPD."
        )

    return theta + delta
