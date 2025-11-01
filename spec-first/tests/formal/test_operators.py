"""Formal tests for discrete divergence and Laplacian operators."""

import os
import sys
import numpy as np

# Ensure 'spec-first/' is importable so 'from geom.operators import ...' works.
THIS_DIR = os.path.dirname(__file__)
SPEC_FIRST_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if SPEC_FIRST_ROOT not in sys.path:
    sys.path.insert(0, SPEC_FIRST_ROOT)

from geom.operators import divergence_node, divergence_from_incidence


def _path_graph_adjacency(n: int) -> np.ndarray:
    """
    Build the adjacency matrix A for the path graph on n nodes with unit weights.
    Edges: (0,1), (1,2), ..., (n-2, n-1)
    """
    A = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    return A


def _cycle_incidence(n: int) -> np.ndarray:
    """
    Build incidence matrix inc of shape (m, n) for the cycle graph C_n with
    a consistent orientation. Edge i is oriented i -> (i+1) mod n.

    Row for edge i has:
      -1 at column i (tail), +1 at column (i+1) mod n (head).
    """
    m = n  # number of edges in a cycle equals number of nodes
    inc = np.zeros((m, n), dtype=float)
    for i in range(n):
        j = (i + 1) % n
        inc[i, i] = -1.0
        inc[i, j] = +1.0
    return inc


def test_divergence_constant_zero_on_path_graph() -> None:
    n = 5
    A = _path_graph_adjacency(n)
    J = np.ones(n, dtype=float)
    div = divergence_node(J, A)
    assert div.shape == (n,)
    # Constant vector must be in Laplacian nullspace: L @ 1 = 0
    norm = float(np.linalg.norm(div))
    assert norm <= 1e-12


def test_divergence_sum_zero_from_incidence_on_cycle() -> None:
    n = 7
    inc = _cycle_incidence(n)
    edge_flux = np.ones(inc.shape[0], dtype=float)
    div = divergence_from_incidence(inc, edge_flux)
    assert div.shape == (n,)
    # Global conservation: sum of divergences equals zero
    total = float(np.sum(div))
    assert abs(total) <= 1e-12