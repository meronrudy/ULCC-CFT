"""Parallel transport primitives.

Invariants
- Preserves inner-product length under metric g (exact in flat connection Γ=0)
- Reduces to identity over trivial (flat) loops
- First-order accurate for small steps

This module is deterministic and uses no RNG.
"""

from __future__ import annotations

import numpy as np


def step_transport_matrix(Gamma: np.ndarray, step: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Discrete transport operator for a single step with selectable approximation order.

    Definitions
    -----------
    Let A = Γ • s denote the contraction over the last index of Γ with the step s:
        A[a, b] = Γ[a, b, c] * s[c]  (implemented via einsum "abc,c->ab")

    - First-order (order=1):   T1 = I − A
    - Second-order (order=2):  T2 ≈ I − A + 0.5 * (A @ A)

    Parameters
    ----------
    Gamma : np.ndarray
        Connection coefficients Γ^a_{bc} with shape (n, n, n), ordered as Γ[a, b, c].
    step : np.ndarray
        Displacement vector (shape (n,)) for the discrete step in coordinates.
    order : int, optional
        Approximation order in {1, 2}. Defaults to 1 for backward compatibility.

    Returns
    -------
    np.ndarray
        Transport matrix T of shape (n, n) with dtype float.

    Raises
    ------
    ValueError
        If shapes are inconsistent, inputs are non-finite, or order is not 1 or 2.

    Notes
    -----
    - Shapes and broadcasting follow the existing first-order implementation; no changes
      beyond adding the 'order' argument.
    - The second-order update improves length preservation for sufficiently small steps,
      while keeping computational cost low (single matrix product A @ A).
    """
    Gamma = np.asarray(Gamma, dtype=float)
    step = np.asarray(step, dtype=float)

    if Gamma.ndim != 3 or Gamma.shape[0] != Gamma.shape[1] or Gamma.shape[0] != Gamma.shape[2]:
        raise ValueError("Gamma must have shape (n, n, n).")
    if step.ndim != 1 or step.shape[0] != Gamma.shape[0]:
        raise ValueError("step must be a 1D vector of length n matching Gamma.")
    if not (np.all(np.isfinite(Gamma)) and np.all(np.isfinite(step))):
        raise ValueError("Gamma and step must contain only finite values.")
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2.")

    n = Gamma.shape[0]
    # A = contraction over the last axis of Gamma with step
    A = np.einsum("abc,c->ab", Gamma, step)

    I = np.eye(n, dtype=float)
    if order == 1:
        T = I - A
    else:  # order == 2
        T = I - A + 0.5 * (A @ A)

    return T


def parallel_transport(
    v: np.ndarray,
    path: list[np.ndarray],
    Gamma: np.ndarray,
    g: np.ndarray,
    atol: float = 1e-12,
    order: int = 1,
) -> np.ndarray:
    """
    Apply discrete parallel transport of vector v along a path with selectable order.

    Transport rules
    ---------------
    For each step s in `path`, form A(s) = Γ • s with A[a, b] = Γ[a, b, c] * s[c], then:
      - order=1: update with T1(s) = I − A(s)
      - order=2: update with T2(s) ≈ I − A(s) + 0.5 * (A(s) @ A(s))
    Apply v ← T( s_k ) @ ... @ T( s_1 ) @ v.

    Invariant (checked)
    -------------------
    Preserves inner-product length under metric g; exact when Γ=0.
    We assert |v_out^T g v_out − v^T g v| ≤ atol.

    Parameters
    ----------
    v : np.ndarray
        Initial vector, shape (n,).
    path : list[np.ndarray]
        Sequence of step vectors, each of shape (n,).
    Gamma : np.ndarray
        Connection coefficients, shape (n, n, n) ordered as Γ[a, b, c].
    g : np.ndarray
        Metric tensor (should be SPD), shape (n, n).
    atol : float
        Absolute tolerance for length preservation check.
    order : int, optional
        Approximation order in {1, 2}. Defaults to 1 for backward compatibility.

    Returns
    -------
    np.ndarray
        Transported vector after applying all steps.

    Raises
    ------
    ValueError
        On shape/domain violations or non-finite inputs.
    AssertionError
        If the inner-product length deviates from the initial length beyond atol.

    Notes
    -----
    - The update is first-order accurate for order=1, and second-order accurate for order=2.
    - Shapes and dtype semantics match the prior implementation; only the 'order' argument
      is added.
    """
    v = np.asarray(v, dtype=float)
    Gamma = np.asarray(Gamma, dtype=float)
    g = np.asarray(g, dtype=float)

    if v.ndim != 1:
        raise ValueError("v must be a 1D vector.")
    if Gamma.ndim != 3 or Gamma.shape[0] != Gamma.shape[1] or Gamma.shape[0] != Gamma.shape[2]:
        raise ValueError("Gamma must have shape (n, n, n).")
    n = Gamma.shape[0]
    if v.shape[0] != n:
        raise ValueError("v must have length n matching Gamma.")
    if g.ndim != 2 or g.shape != (n, n):
        raise ValueError("g must be a square (n, n) matrix matching Gamma.")
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(Gamma)) and np.all(np.isfinite(g))):
        raise ValueError("Inputs v, Gamma, and g must contain only finite values.")
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2.")

    # Metric validation: symmetry and PD
    if not np.allclose(g, g.T, atol=1e-12):
        raise ValueError("g must be symmetric within tolerance.")
    g_sym = 0.5 * (g + g.T)
    w = np.linalg.eigvalsh(g_sym)
    if not np.all(w > 0.0):
        raise ValueError("g must be positive-definite (eigvals > 0).")

    # Initial inner-product length
    v_out = v.copy()
    length0 = float(v_out @ (g_sym @ v_out))

    # Apply transport along the path
    for step in path:
        step = np.asarray(step, dtype=float)
        if step.ndim != 1 or step.shape[0] != n:
            raise ValueError("Each step in path must be a 1D vector of length n.")
        if not np.all(np.isfinite(step)):
            raise ValueError("Each step vector must contain only finite values.")
        T = step_transport_matrix(Gamma, step, order=order)
        v_out = T @ v_out

    # Length preservation check (exact when Γ=0; approx otherwise)
    length1 = float(v_out @ (g_sym @ v_out))
    assert abs(length1 - length0) <= atol, (
        "Parallel transport must preserve inner-product length under metric g "
        f"within tolerance; |Δ|={abs(length1 - length0)} > {atol}"
    )

    return v_out
