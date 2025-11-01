"""Curvature helpers (discrete/DDG-friendly, minimal placeholders).

This module provides small, deterministic utilities needed by the CFE update
routines and formal tests. Implementations are intentionally simple and
documented; they validate square shapes and use only NumPy.

Functions
- symmetrize(A): 0.5 * (A + A.T)
- project_information_tensor(I): symmetric projection Π(𝕀) = 0.5 * (I + I.T)
- einstein_operator_simple(g): surrogate Einstein operator G(g) = g (returns symmetrized g)
- einstein_operator_tracefree(g): G(g) = g − mean(g)·I, where mean(g) = trace(g)/n
- ricci_tensor_naive(g): returns zeros_like(g) (placeholder; not used in current tests)
- scalar_curvature(g, Ric): R = trace(g^{-1} · Ric) with SPD checks on g
- einstein_from_ricci(g, Ric): G = Ric − 0.5 · R · g

Notes
- These helpers are minimal surrogates sufficient for validation-oriented tests.
- When SPD is required (e.g., scalar_curvature), we symmetrize g and check PD.
"""
from __future__ import annotations

import numpy as np


def _validate_square(M: np.ndarray, name: str) -> None:
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"{name} must be a square 2D array; got shape {M.shape}.")


def symmetrize(A: np.ndarray) -> np.ndarray:
    """
    Symmetrize a square matrix.

    Definition
    S = 0.5 * (A + A.T)

    Parameters
    ----------
    A : np.ndarray
        Square matrix.

    Returns
    -------
    np.ndarray
        Symmetric matrix S.

    Raises
    ------
    ValueError
        If A is not square.
    """
    _validate_square(A, "A")
    A = np.asarray(A, dtype=float)
    return 0.5 * (A + A.T)


def project_information_tensor(I: np.ndarray) -> np.ndarray:
    """
    Project an information tensor to its symmetric part Π(𝕀).

    This is the natural projection used in our discrete setting:
    Π(𝕀) = 0.5 * (I + I.T)

    Parameters
    ----------
    I : np.ndarray
        Square information tensor.

    Returns
    -------
    np.ndarray
        Symmetric projection of I.

    Raises
    ------
    ValueError
        If I is not square.
    """
    return symmetrize(I)


def einstein_operator_simple(g: np.ndarray) -> np.ndarray:
    """
    Structure-preserving surrogate Einstein operator: G(g) = g.

    For numerical stability and invariants, we return the symmetrized input.

    Parameters
    ----------
    g : np.ndarray
        Metric-like matrix.

    Returns
    -------
    np.ndarray
        Symmetric matrix equal to g (up to symmetrization).

    Raises
    ------
    ValueError
        If g is not square.
    """
    return symmetrize(g)


def einstein_operator_tracefree(g: np.ndarray) -> np.ndarray:
    """
    Trace-free surrogate Einstein operator.

    Definition (using diagonal mean)
    - Let n be the dimension and μ = trace(g)/n.
    - Return G(g) = g − μ · I_n.

    This removes the average diagonal part, yielding a trace-free matrix
    with respect to the identity basis.

    Parameters
    ----------
    g : np.ndarray
        Square matrix.

    Returns
    -------
    np.ndarray
        Symmetric, trace-adjusted matrix.

    Raises
    ------
    ValueError
        If g is not square.
    """
    g_sym = symmetrize(g)
    n = g_sym.shape[0]
    mu = float(np.trace(g_sym)) / float(n)
    return g_sym - mu * np.eye(n, dtype=float)


def ricci_tensor_naive(g: np.ndarray) -> np.ndarray:
    """
    Naive placeholder for the Ricci tensor in DDG context.

    Returns
    -------
    np.ndarray
        Matrix of zeros with the same shape as g.

    Notes
    -----
    - This is a stand-in not used by current tests. Proper discrete Ricci
      requires additional geometric structure (e.g., metric graph/mesh).
    """
    _validate_square(g, "g")
    g = np.asarray(g, dtype=float)
    return np.zeros_like(g)


def _assert_spd(g: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    """Internal: symmetrize and assert positive-definite."""
    _validate_square(g, "g")
    g = symmetrize(g)
    w = np.linalg.eigvalsh(g)
    if not np.all(w > 0.0):
        raise ValueError("g must be symmetric positive-definite (eigvals > 0).")
    return g


def scalar_curvature(g: np.ndarray, Ric: np.ndarray) -> float:
    """
    Compute scalar curvature surrogate: R = trace(g^{-1} · Ric).

    Preconditions
    - g must be SPD (validated after symmetrization).
    - Ric is symmetrized before use.

    Parameters
    ----------
    g : np.ndarray
        Metric tensor, expected SPD.
    Ric : np.ndarray
        Ricci-like tensor of the same shape as g.

    Returns
    -------
    float
        Scalar curvature R.

    Raises
    ------
    ValueError
        If shapes mismatch, inputs are not square, or g is not SPD.
    """
    _validate_square(g, "g")
    _validate_square(Ric, "Ric")
    if g.shape != Ric.shape:
        raise ValueError("g and Ric must have the same shape.")
    g_spd = _assert_spd(g)
    Ric_sym = symmetrize(Ric)
    R = float(np.trace(np.linalg.inv(g_spd) @ Ric_sym))
    return R


def einstein_from_ricci(g: np.ndarray, Ric: np.ndarray) -> np.ndarray:
    """
    Construct surrogate Einstein tensor from Ricci and metric.

    Definition
    G = Ric − 0.5 · R · g
    where R = scalar_curvature(g, Ric).

    Parameters
    ----------
    g : np.ndarray
        Metric tensor (SPD expected).
    Ric : np.ndarray
        Ricci-like tensor.

    Returns
    -------
    np.ndarray
        Symmetric Einstein-like tensor.

    Raises
    ------
    ValueError
        If shapes mismatch or g is not SPD.
    """
    _validate_square(g, "g")
    _validate_square(Ric, "Ric")
    if g.shape != Ric.shape:
        raise ValueError("g and Ric must have the same shape.")
    g_sym = _assert_spd(g)
    Ric_sym = symmetrize(Ric)
    R = scalar_curvature(g_sym, Ric_sym)
    return symmetrize(Ric_sym - 0.5 * R * g_sym)
