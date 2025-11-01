"""Formal tests for selectable-order parallel transport (step and path)."""

import os
import sys
import numpy as np

# Ensure 'spec-first/' is importable so 'from geom.transport import ...' works.
THIS_DIR = os.path.dirname(__file__)
SPEC_FIRST_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if SPEC_FIRST_ROOT not in sys.path:
    sys.path.insert(0, SPEC_FIRST_ROOT)

from geom.transport import step_transport_matrix, parallel_transport  # noqa: E402


def test_flat_space_identity_orders() -> None:
    """With Γ=0 and g=I, transport should be exactly identity for any step and both orders."""
    n = 5
    rng = np.random.default_rng(0)
    g = np.eye(n, dtype=float)
    Gamma = np.zeros((n, n, n), dtype=float)
    v = rng.standard_normal(n)

    # Random small path
    path = [rng.standard_normal(n) for _ in range(3)]

    # Path transport should not change v exactly in flat connection
    v1 = parallel_transport(v, path, Gamma, g, atol=0.0, order=1)
    v2 = parallel_transport(v, path, Gamma, g, atol=0.0, order=2)
    np.testing.assert_allclose(v1, v, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(v2, v, rtol=0.0, atol=0.0)

    # Step transport matrices are exactly identity
    for s in path:
        T1 = step_transport_matrix(Gamma, s, order=1)
        T2 = step_transport_matrix(Gamma, s, order=2)
        np.testing.assert_allclose(T1, np.eye(n), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(T2, np.eye(n), rtol=0.0, atol=0.0)


def test_curved_second_order_improves_length_preservation() -> None:
    """Second-order update improves length preservation for small steps in a curved (synthetic) connection."""
    n = 6
    rng = np.random.default_rng(0)
    g = np.eye(n, dtype=float)

    # Small magnitude synthetic connection and step
    Gamma = 1e-3 * rng.standard_normal((n, n, n))
    s = 1e-2 * rng.standard_normal(n)
    v = rng.standard_normal(n)

    # Transport via a single step matrix
    T1 = step_transport_matrix(Gamma, s, order=1)
    T2 = step_transport_matrix(Gamma, s, order=2)

    v1 = T1 @ v
    v2 = T2 @ v

    norm_v = float(np.linalg.norm(v))
    err1 = abs(float(np.linalg.norm(v1)) - norm_v)
    err2 = abs(float(np.linalg.norm(v2)) - norm_v)

    # Require a measurable improvement; relative ≥ 1e-5 (tuned for cross-env stability)
    # Guard against division by zero if err1 is numerically 0
    bound = (1.0 - 1e-5) * err1 + 1e-18
    assert err2 <= bound, f"Second-order did not sufficiently improve length error: err1={err1}, err2={err2}"

    # Ensure the two transport matrices are actually different for nonzero Γ and s
    D = T2 - T1
    diff_norm = float(np.linalg.norm(D))
    assert diff_norm > 1e-12, f"Expected T2 != T1; ||T2-T1|| = {diff_norm}"


def test_api_backward_compatibility_and_distinct_order_behavior() -> None:
    """API: omitting order equals order=1; order=2 yields a distinct matrix for nonzero Γ and s."""
    n = 4
    rng = np.random.default_rng(0)
    Gamma = 1e-3 * rng.standard_normal((n, n, n))
    s = 1e-2 * rng.standard_normal(n)

    # step_transport_matrix default equals order=1
    T_default = step_transport_matrix(Gamma, s)
    T_1 = step_transport_matrix(Gamma, s, order=1)
    np.testing.assert_allclose(T_default, T_1, rtol=0.0, atol=0.0)

    # order=2 yields a different matrix (very small difference but nonzero)
    T_2 = step_transport_matrix(Gamma, s, order=2)
    diff = float(np.linalg.norm(T_2 - T_1))
    assert diff > 1e-12, f"Expected order=2 matrix to differ from order=1; ||Δ||={diff}"

    # parallel_transport default equals order=1 on a short path
    g = np.eye(n, dtype=float)
    v = rng.standard_normal(n)
    path = [1e-2 * rng.standard_normal(n), 1e-2 * rng.standard_normal(n)]
    v_def = parallel_transport(v, path, Gamma, g, atol=1e-3)
    v_ord1 = parallel_transport(v, path, Gamma, g, atol=1e-3, order=1)
    np.testing.assert_allclose(v_def, v_ord1, rtol=1e-15, atol=1e-15)


def test_determinism_fixed_seed() -> None:
    """With fixed seeds, repeated calls return identical results."""
    n = 7
    rng = np.random.default_rng(0)
    Gamma = 1e-3 * rng.standard_normal((n, n, n))
    s = 1e-2 * rng.standard_normal(n)
    v = rng.standard_normal(n)
    g = np.eye(n, dtype=float)
    path = [1e-2 * rng.standard_normal(n) for _ in range(3)]

    # Step-level determinism
    T2_a = step_transport_matrix(Gamma, s, order=2)
    T2_b = step_transport_matrix(Gamma, s, order=2)
    np.testing.assert_allclose(T2_a, T2_b, rtol=0.0, atol=0.0)

    # Path-level determinism
    v2_a = parallel_transport(v, path, Gamma, g, atol=1e-3, order=2)
    v2_b = parallel_transport(v, path, Gamma, g, atol=1e-3, order=2)
    np.testing.assert_allclose(v2_a, v2_b, rtol=0.0, atol=0.0)