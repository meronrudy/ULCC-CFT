"""Deterministic NumPy-only graph fixtures for tests.

Functions
- path_graph_adjacency(n): symmetric adjacency for path graph P_n
- cycle_graph_incidence(n): incidence matrix for directed cycle C_n
- basis_vectors(d): standard basis in R^d

All functions validate shapes and inputs; no global RNG used.
"""
from __future__ import annotations

from typing import List
import numpy as np


def _require_int(name: str, val: int) -> None:
    if not isinstance(val, (int, np.integer)):
        raise TypeError(f"{name} must be an integer; got type {type(val).__name__}.")


def path_graph_adjacency(n: int) -> np.ndarray:
    """
    Construct the n×n symmetric adjacency matrix for the path graph P_n.

    Definition
    - Vertices: {0, 1, ..., n-1}
    - Edges: (i, i+1) for i = 0..n-2
    - A[i, j] = 1 iff (i, j) is an edge, else 0. A is symmetric.

    Parameters
    ----------
    n : int
        Number of vertices. Must be ≥ 1.

    Returns
    -------
    np.ndarray
        An (n, n) float64 symmetric adjacency matrix.

    Raises
    ------
    TypeError
        If n is not an integer.
    ValueError
        If n < 1.
    """
    _require_int("n", n)
    if n <= 0:
        raise ValueError("n must be ≥ 1")
    A = np.zeros((n, n), dtype=np.float64)
    if n >= 2:
        i = np.arange(n - 1)
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    return A


def cycle_graph_incidence(n: int) -> np.ndarray:
    """
    Construct the m×n incidence matrix for the directed cycle C_n with consistent orientation.

    Orientation
    - Edge i is (i → (i+1) mod n).
    - Each row corresponds to an edge and contains -1 at the tail, +1 at the head.

    Parameters
    ----------
    n : int
        Number of vertices (and edges). Must be ≥ 3.

    Returns
    -------
    np.ndarray
        An (n, n) float64 incidence matrix with rows indexed by edges.

    Raises
    ------
    TypeError
        If n is not an integer.
    ValueError
        If n < 3.
    """
    _require_int("n", n)
    if n < 3:
        raise ValueError("n must be ≥ 3 for a cycle graph")
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        tail = i
        head = (i + 1) % n
        M[i, tail] = -1.0
        M[i, head] = +1.0
    return M


def basis_vectors(d: int) -> List[np.ndarray]:
    """
    Return the standard basis {e_0, ..., e_{d-1}} of R^d as 1D float64 arrays.

    Parameters
    ----------
    d : int
        Dimension (d ≥ 1).

    Returns
    -------
    list[np.ndarray]
        List of length d; each element is shape (d,) with a single 1 at its index.

    Raises
    ------
    TypeError
        If d is not an integer.
    ValueError
        If d < 1.
    """
    _require_int("d", d)
    if d <= 0:
        raise ValueError("d must be ≥ 1")
    E = np.eye(d, dtype=np.float64)
    return [E[i].copy() for i in range(d)]


__all__ = ["path_graph_adjacency", "cycle_graph_incidence", "basis_vectors"]