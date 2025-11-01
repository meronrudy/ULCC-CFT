"""Reparameterization and coordinate transforms.

Deterministic NumPy helpers with validation.
"""

import numpy as np

_ATOL = 1e-12


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Elementwise logistic σ(x) = 1 / (1 + exp(-x)) computed in a numerically stable way.

    Inputs must be finite. Works elementwise on arrays and scalars; returns array
    of the same shape and dtype=float.
    """
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("sigmoid input must be finite.")
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def logit(theta: np.ndarray) -> np.ndarray:
    """
    Elementwise logit φ = log(θ / (1 − θ)) with strict domain θ ∈ (0, 1).

    Inputs must be finite and strictly between 0 and 1.
    """
    t = np.asarray(theta, dtype=float)
    if not np.all(np.isfinite(t)):
        raise ValueError("logit input must be finite.")
    if np.any((t <= 0.0) | (t >= 1.0)):
        raise ValueError("logit is defined only for θ in the open interval (0,1).")
    # Use log1p for numerical stability: log(t) - log(1-t)
    return np.log(t) - np.log1p(-t)


def jacobian_logit_to_theta(phi: np.ndarray) -> np.ndarray:
    """
    Elementwise Jacobian dθ/dφ for θ = σ(φ): J = σ(φ) · (1 − σ(φ)).
    """
    p = np.asarray(phi, dtype=float)
    if not np.all(np.isfinite(p)):
        raise ValueError("jacobian input must be finite.")
    s = sigmoid(p)
    return s * (1.0 - s)


def pullback_metric(g_theta: np.ndarray, J: np.ndarray) -> np.ndarray:
    """
    Compute pullback of a metric under a diagonal reparameterization.

    g_phi = J^T g_theta J

    Supports:
    - Scalar case: g_theta is 1x1 and J is 0-D, 1-element 1-D, or 1x1 array.
    - Diagonal/vectorized case: J is a 1-D array giving the diagonal of the Jacobian.

    Both g_theta and the resulting g_phi are validated/symmetrized within a small tolerance.
    """
    g = np.asarray(g_theta, dtype=float)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError("g_theta must be a square 2D matrix.")
    if not np.allclose(g, g.T, atol=_ATOL):
        raise ValueError("g_theta must be symmetric within tolerance.")

    n = g.shape[0]
    J_arr = np.asarray(J, dtype=float)
    # Build a diagonal Jacobian matrix
    if J_arr.ndim == 0:
        if n != 1:
            raise ValueError("0-D Jacobian only valid for 1x1 metric.")
        J_mat = np.array([[float(J_arr)]], dtype=float)
    elif J_arr.ndim == 1:
        if J_arr.shape[0] != n:
            raise ValueError("Length of 1-D J must equal metric dimension.")
        J_mat = np.diag(J_arr)
    elif J_arr.ndim == 2:
        if J_arr.shape != (n, n):
            raise ValueError("2-D J must have same shape as g_theta.")
        # Require diagonal within tolerance
        if not np.allclose(J_arr, np.diag(np.diag(J_arr)), atol=_ATOL):
            raise ValueError("Only diagonal Jacobians are supported in pullback_metric.")
        J_mat = J_arr
    else:
        raise ValueError("J must be 0-D, 1-D, or 2-D diagonal.")

    g_phi = J_mat.T @ g @ J_mat
    # Numerical symmetrization
    g_phi = 0.5 * (g_phi + g_phi.T)
    return g_phi


def reparam_theta_to_phi(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Map θ ∈ (0,1)^n to φ via logit and return the Jacobian of θ with respect to φ.

    Returns
    -------
    (phi, J)
        phi = logit(theta) with the same shape as theta
        J = dθ/dφ evaluated at phi, elementwise σ(φ)(1 − σ(φ)) with same shape as theta
    """
    t = np.asarray(theta, dtype=float)
    phi = logit(t)
    J = jacobian_logit_to_theta(phi)
    return phi, J
