"""Discrete geometric operator utilities: Laplacian and divergence.

All functions are deterministic NumPy implementations with input validation.
"""

import numpy as np

_ATOL = 1e-12


def laplacian_from_adjacency(A: np.ndarray) -> np.ndarray:
    """
    Graph Laplacian; constant vector is in nullspace for connected graphs.

    Given a weighted, undirected adjacency matrix A (square, symmetric within atol,
    nonnegative entries; zero diagonal allowed), return the combinatorial Laplacian
    L = D − A, where D = diag(A.sum(axis=1)) is the degree matrix.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Adjacency A must be a square 2D array.")
    if not np.all(np.isfinite(A)):
        raise ValueError("Adjacency A must have finite entries.")
    if not np.allclose(A, A.T, atol=_ATOL):
        raise ValueError("Adjacency A must be symmetric within tolerance.")
    if np.any(A < -_ATOL):
        raise ValueError("Adjacency A must be elementwise nonnegative (within tolerance).")
    deg = np.sum(A, axis=1)
    D = np.diag(deg)
    L = D - A
    return L


def divergence_node(J: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Discrete node divergence of a scalar field J on the vertices of a graph with adjacency A.

    Computes div = L @ J where L is the graph Laplacian derived from A via
    laplacian_from_adjacency(A). The result is a 1D vector of shape (n,).

    Invariant: divergence of a constant field is zero on any graph
    with symmetric adjacency (L · 1 = 0).
    """
    L = laplacian_from_adjacency(A)
    n = L.shape[0]
    J_arr = np.asarray(J, dtype=float)
    if J_arr.ndim == 2:
        if J_arr.shape == (n, 1):
            J_vec = J_arr[:, 0]
        elif J_arr.shape == (1, n):
            J_vec = J_arr[0, :]
        else:
            raise ValueError("J must have shape (n,), (n,1), or (1,n).")
    elif J_arr.ndim == 1:
        if J_arr.shape[0] != n:
            raise ValueError("J must have length equal to the number of nodes.")
        J_vec = J_arr
    else:
        raise ValueError("J must be a 1D vector or 2D single-row/column.")
    div = L @ J_vec
    return np.asarray(div, dtype=float)


def divergence_from_incidence(inc: np.ndarray, edge_flux: np.ndarray) -> np.ndarray:
    """
    Incidence-based divergence; sum of divergences equals zero for closed systems.

    Convention: if inc has shape (m, n) where rows index oriented edges and
    columns index nodes, and edge_flux has shape (m,), the node-wise divergence
    is computed as

        div = -inc.T @ edge_flux

    With consistent edge orientations and no sources/sinks, np.sum(div) == 0.
    """
    B = np.asarray(inc, dtype=float)
    if B.ndim != 2:
        raise ValueError("incidence matrix must be 2D with shape (m, n).")
    if not np.all(np.isfinite(B)):
        raise ValueError("incidence matrix must have finite entries.")
    m, n = B.shape
    f = np.asarray(edge_flux, dtype=float)
    if f.ndim == 2:
        if f.shape == (m, 1):
            f_vec = f[:, 0]
        elif f.shape == (1, m):
            f_vec = f[0, :]
        else:
            raise ValueError("edge_flux must have shape (m,), (m,1), or (1,m).")
    elif f.ndim == 1:
        if f.shape[0] != m:
            raise ValueError("edge_flux length must equal number of edges (m).")
        f_vec = f
    else:
        raise ValueError("edge_flux must be 1D or 2D single-row/column.")
    if not np.all(np.isfinite(f_vec)):
        raise ValueError("edge_flux must have finite entries.")
    div = - (B.T @ f_vec)
    return np.asarray(div, dtype=float)
