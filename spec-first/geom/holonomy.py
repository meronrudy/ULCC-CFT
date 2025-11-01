"""Discrete holonomy computation.

Invariants
- For flat connections (Γ=0), holonomy equals identity and curvature ≈ 0.
- Curvature measure: Frobenius norm of (T − I), where T is the composed transport.
"""

from __future__ import annotations

import numpy as np

from geom.transport import step_transport_matrix


def holonomy(
    loop: list[np.ndarray],
    Gamma: np.ndarray,
    g: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Compose discrete parallel transports around a closed loop and measure deviation from identity.

    Parameters
    ----------
    loop : list[np.ndarray]
        Sequence of steps forming a closed loop; each step has shape (n,).
    Gamma : np.ndarray
        Connection coefficients Γ^a_{bc}, shape (n, n, n).
    g : np.ndarray
        Metric tensor (SPD), shape (n, n). Used for validation.

    Returns
    -------
    (T, curv) : (np.ndarray, float)
        - T: composed transport matrix of shape (n, n).
        - curv: Frobenius norm ||T − I||_F as a scalar curvature measure.

    Raises
    ------
    ValueError
        If shapes are inconsistent or g is not SPD.
    """
    Gamma = np.asarray(Gamma, dtype=float)
    g = np.asarray(g, dtype=float)

    if Gamma.ndim != 3 or Gamma.shape[0] != Gamma.shape[1] or Gamma.shape[0] != Gamma.shape[2]:
        raise ValueError("Gamma must have shape (n, n, n).")
    n = Gamma.shape[0]

    if g.ndim != 2 or g.shape != (n, n):
        raise ValueError("g must be a square (n, n) matrix matching Gamma.")

    # Validate SPD metric
    if not np.allclose(g, g.T, atol=1e-12):
        raise ValueError("g must be symmetric within tolerance.")
    g_sym = 0.5 * (g + g.T)
    w = np.linalg.eigvalsh(g_sym)
    if not np.all(w > 0.0):
        raise ValueError("g must be positive-definite (eigvals > 0).")

    # Compose transport around the loop
    T = np.eye(n, dtype=float)
    for step in loop:
        step = np.asarray(step, dtype=float)
        if step.ndim != 1 or step.shape[0] != n:
            raise ValueError("Each loop step must be a 1D vector of length n.")
        P = step_transport_matrix(Gamma, step)
        # Pre-multiply to respect application order on vectors: v_out = P_k ... P_1 v
        T = P @ T

    curv = float(np.linalg.norm(T - np.eye(n, dtype=float), ord="fro"))
    return T, curv
