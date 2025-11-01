"""Deterministic NumPy-only distribution fixtures for tests.

Functions
- bernoulli_scores(theta, N, rng): score samples s_i = (x_i - theta) / (theta * (1 - theta))
- quadratic_potential_matrix(diag): positive diagonal matrix A for V(θ) = 0.5 θ^T A θ

All helpers validate inputs, are deterministic given rng, and avoid global RNG.
"""
from __future__ import annotations

from typing import Iterable
import numpy as np


def _require_generator(rng: np.random.Generator) -> None:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator instance")


def _require_int(name: str, val: int) -> None:
    if not isinstance(val, (int, np.integer)):
        raise TypeError(f"{name} must be an integer; got type {type(val).__name__}.")


def bernoulli_scores(theta: float, N: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate Bernoulli score samples s_i = (x_i − θ) / (θ (1−θ)) with x_i ~ Bernoulli(θ).

    Deterministic given the supplied rng (no global state).

    Parameters
    ----------
    theta : float
        Success probability, required to be in the open interval (0, 1).
    N : int
        Number of samples, must be ≥ 1.
    rng : np.random.Generator
        NumPy random generator providing determinism.

    Returns
    -------
    np.ndarray
        1D array of shape (N,) with dtype float64.

    Raises
    ------
    TypeError
        If N is not an integer or rng is not a Generator.
    ValueError
        If theta ∉ (0,1) or N < 1.
    """
    _require_int("N", N)
    if not (0.0 < float(theta) < 1.0):
        raise ValueError("theta must be in the open interval (0, 1)")
    if N <= 0:
        raise ValueError("N must be ≥ 1")
    _require_generator(rng)

    x = rng.binomial(n=1, p=float(theta), size=int(N)).astype(np.float64)
    denom = float(theta) * (1.0 - float(theta))
    s = (x - float(theta)) / denom
    return s.astype(np.float64, copy=False)


def quadratic_potential_matrix(diag: Iterable[float] | np.ndarray) -> np.ndarray:
    """
    Construct a positive diagonal matrix A for the quadratic potential
        V(θ) = 0.5 * θ^T A θ.

    Parameters
    ----------
    diag : Iterable[float] | np.ndarray
        Positive diagonal entries. Must be 1D with all entries > 0.

    Returns
    -------
    np.ndarray
        A diagonal matrix with dtype float64 and shape (d, d).

    Raises
    ------
    ValueError
        If diag is empty, not 1D, or contains non-positive entries.
    """
    d = np.asarray(diag, dtype=np.float64)
    if d.ndim != 1 or d.size == 0:
        raise ValueError("diag must be a non-empty 1D array-like")
    if not np.all(d > 0.0):
        raise ValueError("all diagonal entries must be strictly positive")
    return np.diag(d).astype(np.float64, copy=False)


__all__ = ["bernoulli_scores", "quadratic_potential_matrix"]