# Ensure project root is on sys.path for direct pytest invocation of a nested module
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import pytest

from slow_plane.field import FieldConfig, solve_field
# Access internal helpers for targeted tests where allowed by scope
from slow_plane.field.solver import cfl_dt_max, enforce_spd, laplace_g, grad_g  # noqa: F401


def _make_identity_metric(H: int, W: int) -> np.ndarray:
    g = np.zeros((H, W, 2, 2), dtype=np.float64)
    g[..., 0, 0] = 1.0
    g[..., 1, 1] = 1.0
    g[..., 0, 1] = 0.0
    g[..., 1, 0] = 0.0
    return g


def _single_source(H: int, W: int, y: int, x: int, mag: float = 1.0) -> np.ndarray:
    J = np.zeros((H, W), dtype=np.float64)
    J[y, x] = float(mag)
    return J


def _norm_corr(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a - float(a.mean())
    b0 = b - float(b.mean())
    na = np.linalg.norm(a0.ravel())
    nb = np.linalg.norm(b0.ravel())
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a0.ravel(), b0.ravel()) / (na * nb))


def test_shape_and_determinism_leapfrog():
    H = W = 16
    g = _make_identity_metric(H, W)
    J = _single_source(H, W, H // 2, W // 2, mag=1.0)

    # CFL-safe dt: for identity metric, cfl_dt_max ≈ 0.5
    dt_max = cfl_dt_max(g)
    cfg = FieldConfig(
        method="leapfrog",
        dt=min(0.3, 0.8 * dt_max),
        steps=50,
        cfl_safety=0.8,
        boundary="neumann",
        grad_clip=0.0,
    )

    out1 = solve_field(g, J, cfg)
    out2 = solve_field(g, J, cfg)

    phi1, grad1, meta1 = out1["phi"], out1["grad"], out1["meta"]
    phi2, grad2, meta2 = out2["phi"], out2["grad"], out2["meta"]

    # Shapes
    assert phi1.shape == (H, W)
    assert grad1.shape == (H, W, 2)

    # Determinism: byte-identical
    assert phi1.dtype == np.float64 and grad1.dtype == np.float64
    assert phi2.dtype == np.float64 and grad2.dtype == np.float64
    assert phi1.tobytes() == phi2.tobytes()
    assert grad1.tobytes() == grad2.tobytes()

    # CFL passed
    assert meta1["cfl_passed"] is True
    assert meta2["cfl_passed"] is True


def test_cfl_check_flag():
    H = W = 16
    g = _make_identity_metric(H, W)
    J = np.zeros((H, W), dtype=np.float64)

    dt_max = cfl_dt_max(g)
    # Intentionally choose dt slightly above cfl_safety*dt_max to trip flag
    cfg = FieldConfig(method="leapfrog", dt=0.51 * dt_max, steps=5, cfl_safety=0.5)

    out = solve_field(g, J, cfg)
    assert out["meta"]["cfl_passed"] is False


def test_boundary_conditions_dirichlet_vs_neumann():
    H = W = 32
    g = _make_identity_metric(H, W)
    # Source near a corner to excite boundary behavior
    J = _single_source(H, W, 2, 2, mag=1.0)

    dt_max = cfl_dt_max(g)
    dt = 0.4 * dt_max

    cfg_neu = FieldConfig(method="leapfrog", dt=dt, steps=60, boundary="neumann")
    cfg_dir = FieldConfig(method="leapfrog", dt=dt, steps=60, boundary="dirichlet")

    phi_neu = solve_field(g, J, cfg_neu)["phi"]
    phi_dir = solve_field(g, J, cfg_dir)["phi"]

    # Amplitude near borders should be more suppressed under Dirichlet
    border_mask = np.zeros_like(phi_neu, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    amp_neu = float(np.mean(np.abs(phi_neu[border_mask])))
    amp_dir = float(np.mean(np.abs(phi_dir[border_mask])))

    assert amp_dir <= amp_neu


def test_cg_steady_state_and_correlation_with_leapfrog():
    H = W = 24
    g = _make_identity_metric(H, W)
    J = _single_source(H, W, H // 2, W // 2, mag=1.0)

    # CG steady-state
    cfg_cg = FieldConfig(method="cg", max_cg_iters=500, cg_tol=1e-6, boundary="neumann")
    out_cg = solve_field(g, J, cfg_cg)
    phi_cg, meta_cg = out_cg["phi"], out_cg["meta"]
    assert meta_cg["iters"] <= cfg_cg.max_cg_iters
    assert meta_cg["residual"] >= 0.0

    # Late-time leapfrog snapshot (shape similarity)
    dt_max = cfl_dt_max(g)
    cfg_lf = FieldConfig(method="leapfrog", dt=0.4 * dt_max, steps=120, boundary="neumann")
    phi_lf = solve_field(g, J, cfg_lf)["phi"]

    corr = _norm_corr(phi_cg, phi_lf)
    assert corr > 0.9


def test_metric_spd_clamp_and_no_nans():
    H = W = 12
    g = _make_identity_metric(H, W)
    # Introduce slight non-SPD: negative small eig along one tile and asymmetric noise
    g[3, 4] = np.array([[1.0, 0.2], [0.6, -1e-8]], dtype=np.float64)  # negative tiny eig
    g[5, 2] = np.array([[0.9, 0.5], [0.1, 1.1]], dtype=np.float64)    # asymmetric

    g_spd = enforce_spd(g, 1e-6)
    # Eigenvalues strictly positive
    for y in range(H):
        for x in range(W):
            w = np.linalg.eigvalsh(0.5 * (g_spd[y, x] + g_spd[y, x].T))
            assert np.all(w > 0.0)

    J = _single_source(H, W, H // 2, W // 2, mag=0.5)
    cfg = FieldConfig(method="cg", max_cg_iters=200, cg_tol=1e-5)
    out = solve_field(g, J, cfg)  # solver will internally SPD-project
    phi = out["phi"]
    assert np.all(np.isfinite(phi))


def test_gradient_clipping_bounds_norm():
    H = W = 20
    g = _make_identity_metric(H, W)
    # Stronger source to drive large gradients
    J = _single_source(H, W, H // 2, W // 2, mag=5.0)

    dt_max = cfl_dt_max(g)
    clip = 0.05
    cfg = FieldConfig(method="leapfrog", dt=0.3 * dt_max, steps=80, grad_clip=clip)

    out = solve_field(g, J, cfg)
    phi = out["phi"]

    # Compute gradients; must respect clipping within tolerance
    grads = grad_g(phi, g, boundary=cfg.boundary)
    norms = np.sqrt(grads[..., 0] ** 2 + grads[..., 1] ** 2)
    assert float(np.max(norms)) <= clip + 1e-6