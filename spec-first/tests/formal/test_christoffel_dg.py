"""Formal tests for discrete metric derivative and Christoffel consistency."""

import os
import sys
import numpy as np
import pytest

THIS_DIR = os.path.dirname(__file__)
SPEC_FIRST_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if SPEC_FIRST_ROOT not in sys.path:
    sys.path.insert(0, SPEC_FIRST_ROOT)

from geom.christoffel import discrete_metric_derivative, christoffel


def _path_graph_edges(N: int) -> np.ndarray:
    a = np.arange(N - 1, dtype=int)
    b = a + 1
    return np.stack([a, b], axis=1)


def _grid_3x3_coords_edges():
    xs = np.linspace(-1.0, 1.0, 3)
    ys = np.linspace(-1.0, 1.0, 3)
    coords = []
    for r in range(3):
        for c in range(3):
            coords.append([xs[c], ys[r]])
    coords = np.asarray(coords, dtype=float)
    edges = []
    for r in range(3):
        for c in range(3):
            i = r * 3 + c
            if c + 1 < 3:
                edges.append([i, r * 3 + (c + 1)])
            if r + 1 < 3:
                edges.append([i, (r + 1) * 3 + c])
    edges = np.asarray(edges, dtype=int)
    return coords, edges


def test_dg_1d_line_derivative_accuracy():
    N = 21
    x = np.linspace(-1.0, 1.0, N)
    coords = x[:, None]
    alpha = 0.3
    g_nodes = (1.0 + alpha * x)[:, None, None]
    edges = _path_graph_edges(N)
    dg = discrete_metric_derivative(g_nodes, coords, edges)
    interior = np.arange(1, N - 1, dtype=int)
    est = dg[interior, 0, 0, 0]
    err = np.abs(est - alpha)
    assert np.max(err) <= 5e-3


def test_christoffel_1d_consistency():
    N = 21
    x = np.linspace(-1.0, 1.0, N)
    coords = x[:, None]
    alpha = 0.3
    g_nodes = (1.0 + alpha * x)[:, None, None]
    edges = _path_graph_edges(N)
    dg_nodes = discrete_metric_derivative(g_nodes, coords, edges)
    interior = np.arange(1, N - 1, dtype=int)
    gammas = []
    for i in interior:
        g_i = g_nodes[i, :, :].reshape(1, 1)
        dg_i = dg_nodes[i, :, :, :].reshape(1, 1, 1)
        Gamma_i = christoffel(g_i, dg_i)
        gammas.append(Gamma_i[0, 0, 0])
    gammas = np.asarray(gammas)
    analytic = 0.5 * (alpha / (1.0 + alpha * x[interior]))
    assert np.max(np.abs(gammas - analytic)) <= 5e-3


def test_dg_2d_grid_least_squares_center_node():
    coords, edges = _grid_3x3_coords_edges()
    N = coords.shape[0]
    d = 2
    x = coords[:, 0]
    y = coords[:, 1]
    g_nodes = np.zeros((N, d, d), dtype=float)
    g_nodes[:, 0, 0] = 1.0 + 0.2 * x
    g_nodes[:, 1, 1] = 1.0 + 0.1 * y
    dg = discrete_metric_derivative(g_nodes, coords, edges)
    center = 1 * 3 + 1
    dg_c = dg[center]
    assert np.allclose(dg_c[0, 0, 0], 0.2, atol=2e-2)
    assert np.allclose(dg_c[1, 1, 1], 0.1, atol=2e-2)
    off = []
    for k in range(2):
        off.append(abs(dg_c[k, 0, 1]))
        off.append(abs(dg_c[k, 1, 0]))
    assert max(off) <= 2e-2


def test_validation_non_symmetric_metric_raises():
    N, d = 2, 2
    coords = np.zeros((N, d))
    coords[1, 0] = 1.0
    edges = np.array([[0, 1]], dtype=int)
    g_nodes = np.zeros((N, d, d), dtype=float)
    for i in range(N):
        g_nodes[i] = np.eye(d)
    g_nodes[0, 0, 1] = 0.1  # break symmetry without mirroring (1,0)
    with pytest.raises(ValueError):
        _ = discrete_metric_derivative(g_nodes, coords, edges)


def test_validation_non_spd_metric_raises():
    N, d = 2, 2
    coords = np.zeros((N, d))
    coords[1, 0] = 1.0
    edges = np.array([[0, 1]], dtype=int)
    g_nodes = np.zeros((N, d, d), dtype=float)
    g_nodes[0] = np.array([[0.0, 0.0], [0.0, 0.0]])
    g_nodes[1] = np.eye(d)
    with pytest.raises(ValueError):
        _ = discrete_metric_derivative(g_nodes, coords, edges)


def test_validation_invalid_edge_indices_raises():
    N, d = 3, 1
    coords = np.linspace(0.0, 1.0, N)[:, None]
    g_nodes = np.ones((N, d, d), dtype=float)
    edges = np.array([[0, 3]], dtype=int)  # 3 is out-of-range
    with pytest.raises(ValueError):
        _ = discrete_metric_derivative(g_nodes, coords, edges)


def test_validation_d_not_equal_p_raises():
    N, d = 3, 2
    coords = np.zeros((N, d + 1))
    g_nodes = np.array([np.eye(d) for _ in range(N)], dtype=float)
    edges = np.array([[0, 1], [1, 2]], dtype=int)
    with pytest.raises(ValueError):
        _ = discrete_metric_derivative(g_nodes, coords, edges)


def test_output_shape_and_symmetry():
    coords, edges = _grid_3x3_coords_edges()
    N = coords.shape[0]
    d = 2
    x = coords[:, 0]
    y = coords[:, 1]
    g_nodes = np.zeros((N, d, d), dtype=float)
    g_nodes[:, 0, 0] = 1.0 + 0.2 * x
    g_nodes[:, 1, 1] = 1.0 + 0.1 * y
    dg = discrete_metric_derivative(g_nodes, coords, edges)
    assert dg.shape == (N, d, d, d)
    assert np.allclose(dg, np.swapaxes(dg, 2, 3), atol=1e-12)