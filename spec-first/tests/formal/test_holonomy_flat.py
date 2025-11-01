"""Formal tests for flat holonomy and parallel transport length preservation."""

import os
import sys
import numpy as np

# Ensure 'spec-first/' is importable so 'from geom.transport import ...' works.
THIS_DIR = os.path.dirname(__file__)
SPEC_FIRST_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if SPEC_FIRST_ROOT not in sys.path:
    sys.path.insert(0, SPEC_FIRST_ROOT)

from geom.transport import parallel_transport
from geom.holonomy import holonomy


def test_flat_holonomy_identity() -> None:
    n = 3
    g = np.eye(n, dtype=float)
    Gamma = np.zeros((n, n, n), dtype=float)

    dx = np.array([1.0, 0.0, 0.0], dtype=float)
    dy = np.array([0.0, 1.0, 0.0], dtype=float)
    loop = [dx, dy, -dx, -dy]

    T, curv = holonomy(loop, Gamma, g)
    assert np.allclose(T, np.eye(n), atol=1e-12)
    assert curv < 1e-12


def test_parallel_transport_length_preservation() -> None:
    n = 3
    g = np.eye(n, dtype=float)
    Gamma = np.zeros((n, n, n), dtype=float)

    v = np.array([1.0, 2.0, -0.5], dtype=float)
    path = [
        np.array([0.3, -0.1, 0.0], dtype=float),
        np.array([0.0, 0.2, -0.2], dtype=float),
        np.array([-0.3, -0.1, 0.3], dtype=float),
    ]

    v2 = parallel_transport(v, path, Gamma, g, atol=1e-12)
    # Length preservation and identity action in flat connection
    assert np.isclose(v2 @ (g @ v2), v @ (g @ v), atol=1e-12)
    assert np.allclose(v2, v, atol=1e-12)