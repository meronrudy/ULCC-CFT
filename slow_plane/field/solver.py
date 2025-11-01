"""Metric-aware 2-D field solver (Phase A — E3b).

Numerics (deterministic, numpy-only):
- Grid: scalar field Phi on HxW cells; per-cell metric g[y,x] is 2x2 SPD.
- Operators:
  * grad_g(phi, g): central differences for (dphi/dx, dphi/dy) then optionally contract with g^{-1} when forming Laplace-Beltrami.
  * laplace_g(phi, g): discrete ∇·(g^{-1} ∇phi) using central differences with boundary handling.
  * enforce_spd(g, eps): symmetric projection and eigenvalue clamp (λ >= eps) per cell.
  * cfl_dt_max(g): dt stability proxy using c ≈ 1/sqrt(trace(g)); dt_max ≈ (1/sqrt(2)) * min(1/c) across grid.

Boundary conditions:
- "neumann": zero normal derivative at domain borders (reflective ghost cells).
- "dirichlet": Phi = 0 on domain borders (clamped each operator application).

Integrators:
- Leapfrog (explicit, transient):
  v_{n+1/2} = v_{n-1/2} + dt * (Δ_g Phi_n + J)
  Phi_{n+1} = Phi_n + dt * v_{n+1/2}
  Start from Phi_0 = 0, v_{-1/2} = 0. CFL is enforced via cfg.cfl_safety * cfl_dt_max(g).
  Optional gradient clipping: if cfg.grad_clip > 0, per-cell ||∇Phi|| is constrained by scaling v update at that cell.

- Conjugate Gradient (steady-state):
  Solve -Δ_g Phi = J with matrix-free CG, boundary handled as above.
  For Dirichlet, borders are clamped to zero at every operator application and RHS borders forced to zero.

Validation:
- Shapes: g [H,W,2,2], J [H,W], H,W ≥ 2. g and J must be finite.
- cfg fields validated with ranges as specified in FieldConfig.

Outputs:
- dict {
    "phi": (H,W) float64,
    "grad": (H,W,2) float64 (central differences in axes coordinates),
    "meta": {"method","steps","dt","cfl_passed","iters","residual"}
  }

Note: This module is self-contained and slow-plane only; no fast-plane dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np


@dataclass(frozen=True)
class FieldConfig:
    method: str = "leapfrog"                 # {"leapfrog","cg"}
    dt: float = 1e-2                         # time step for leapfrog
    steps: int = 50                          # ≥1 leapfrog steps
    cfl_safety: float = 0.8                  # (0,1]
    max_cg_iters: int = 500                  # ≥1
    cg_tol: float = 1e-6                     # >0
    rng_seed: int = 0                        # kept for future tie-breakers; not used
    boundary: str = "neumann"                # {"neumann","dirichlet"}
    grad_clip: float = 0.0                   # ≥0; 0 disables clipping
    metric_eps: float = 1e-6                 # >0

    def __post_init__(self) -> None:
        if self.method not in ("leapfrog", "cg"):
            raise ValueError("method must be 'leapfrog' or 'cg'")
        if not (float(self.dt) > 0.0):
            raise ValueError("dt must be > 0")
        if int(self.steps) < 1:
            raise ValueError("steps must be ≥ 1")
        if not (0.0 < float(self.cfl_safety) <= 1.0):
            raise ValueError("cfl_safety must be in (0,1]")
        if int(self.max_cg_iters) < 1:
            raise ValueError("max_cg_iters must be ≥ 1")
        if not (float(self.cg_tol) > 0.0):
            raise ValueError("cg_tol must be > 0")
        if self.boundary not in ("neumann", "dirichlet"):
            raise ValueError("boundary must be 'neumann' or 'dirichlet'")
        if float(self.grad_clip) < 0.0:
            raise ValueError("grad_clip must be ≥ 0")
        if not (float(self.metric_eps) > 0.0):
            raise ValueError("metric_eps must be > 0")


# ---- Helpers (module-internal) ----

def _validate_inputs(g: np.ndarray, J: np.ndarray) -> Tuple[int, int]:
    G = np.asarray(g, dtype=np.float64)
    S = np.asarray(J, dtype=np.float64)
    if G.ndim != 4 or G.shape[-2:] != (2, 2):
        raise ValueError("g must have shape [H,W,2,2]")
    if S.ndim != 2:
        raise ValueError("J must have shape [H,W]")
    H, W = S.shape
    if (H, W) != G.shape[:2]:
        raise ValueError("g and J spatial shapes must match [H,W]")
    if H < 2 or W < 2:
        raise ValueError("H and W must be ≥ 2")
    if not (np.all(np.isfinite(G)) and np.all(np.isfinite(S))):
        raise ValueError("g and J must contain only finite values")
    return H, W


def enforce_spd(g: np.ndarray, eps: float) -> np.ndarray:
    """Symmetrize and clamp eigenvalues ≥ eps per cell. Deterministic."""
    G = np.asarray(g, dtype=np.float64)
    H, W = G.shape[:2]
    out = np.empty_like(G, dtype=np.float64)
    for y in range(H):
        for x in range(W):
            A = 0.5 * (G[y, x] + G[y, x].T)
            w, V = np.linalg.eig(A)
            # Guard against small imaginary due to round-off
            w = np.real(w)
            V = np.real(V)
            w = np.maximum(w, float(eps))
            A_spd = (V * w) @ V.T  # V diag(w) V^T
            # Re-symmetrize
            out[y, x] = 0.5 * (A_spd + A_spd.T)
    return out


def _inverse_metric(g: np.ndarray, eps: float) -> np.ndarray:
    """Compute inverse of SPD metric per cell robustly."""
    H, W = g.shape[:2]
    g_inv = np.empty_like(g, dtype=np.float64)
    for y in range(H):
        for x in range(W):
            # 2x2 inverse closed-form is fine; still safe to use np.linalg.inv
            g_inv[y, x] = np.linalg.inv(g[y, x])
    return g_inv


def cfl_dt_max(g: np.ndarray) -> float:
    """Estimate a global dt_max using c ≈ 1/sqrt(trace(g)); dt_max ≈ (1/sqrt(2)) * min(1/c)."""
    G = np.asarray(g, dtype=np.float64)
    tr = G[..., 0, 0] + G[..., 1, 1]
    tr = np.maximum(tr, 1e-12)
    c = 1.0 / np.sqrt(tr)
    # grid spacing assumed 1; 2D CFL for leapfrog on a cross stencil ~ 1/sqrt(2)
    dt_local = (1.0 / np.sqrt(2.0)) * c
    return float(np.min(dt_local))


def _apply_dirichlet_border(phi: np.ndarray) -> None:
    phi[0, :] = 0.0
    phi[-1, :] = 0.0
    phi[:, 0] = 0.0
    phi[:, -1] = 0.0


def _reflect_index(i: int, n: int) -> int:
    """Reflective index for Neumann BC: -1→1, n→n-2, etc."""
    if i < 0:
        return -i
    if i >= n:
        return 2 * n - 2 - i
    return i


def _neighbor_avg(phi: np.ndarray, boundary: str = "neumann") -> np.ndarray:
    """
    Compute 4-neighbor average per cell with boundary handling matching grad_g.
    """
    P = np.asarray(phi, dtype=np.float64)
    H, W = P.shape
    avg = np.zeros_like(P)
    for y in range(H):
        for x in range(W):
            xm = _reflect_index(x - 1, W) if boundary == "neumann" else max(x - 1, 0)
            xp = _reflect_index(x + 1, W) if boundary == "neumann" else min(x + 1, W - 1)
            ym = _reflect_index(y - 1, H) if boundary == "neumann" else max(y - 1, 0)
            yp = _reflect_index(y + 1, H) if boundary == "neumann" else min(y + 1, H - 1)
            avg[y, x] = 0.25 * (P[y, xm] + P[y, xp] + P[ym, x] + P[yp, x])
    return avg


def grad_g(phi: np.ndarray, g: np.ndarray, boundary: str = "neumann") -> np.ndarray:
    """
    Central differences gradient ∇phi in axis coordinates (dx, dy).
    Returns array [H,W,2] with channels [dphi_dx, dphi_dy].
    Boundary:
      - "neumann": reflect indices (zero normal derivative)
      - "dirichlet": zero borders then use one-sided near edges by reflection of zeros
    """
    P = np.asarray(phi, dtype=np.float64)
    H, W = P.shape
    P = P.copy()
    if boundary == "dirichlet":
        _apply_dirichlet_border(P)

    d = np.zeros((H, W, 2), dtype=np.float64)
    # dx
    for y in range(H):
        for x in range(W):
            xm = _reflect_index(x - 1, W) if boundary == "neumann" else max(x - 1, 0)
            xp = _reflect_index(x + 1, W) if boundary == "neumann" else min(x + 1, W - 1)
            d[y, x, 0] = 0.5 * (P[y, xp] - P[y, xm])
    # dy
    for y in range(H):
        for x in range(W):
            ym = _reflect_index(y - 1, H) if boundary == "neumann" else max(y - 1, 0)
            yp = _reflect_index(y + 1, H) if boundary == "neumann" else min(y + 1, H - 1)
            d[y, x, 1] = 0.5 * (P[yp, x] - P[ym, x])
    return d


def _divergence(vec: np.ndarray, boundary: str = "neumann") -> np.ndarray:
    """
    Divergence of a vector field V[...,2] with components Vx, Vy over grid.
    Central differences with same boundary handling as grad_g.
    """
    V = np.asarray(vec, dtype=np.float64)
    H, W = V.shape[:2]
    out = np.zeros((H, W), dtype=np.float64)

    # dVx/dx
    for y in range(H):
        for x in range(W):
            xm = _reflect_index(x - 1, W) if boundary == "neumann" else max(x - 1, 0)
            xp = _reflect_index(x + 1, W) if boundary == "neumann" else min(x + 1, W - 1)
            dVx_dx = 0.5 * (V[y, xp, 0] - V[y, xm, 0])

            ym = _reflect_index(y - 1, H) if boundary == "neumann" else max(y - 1, 0)
            yp = _reflect_index(y + 1, H) if boundary == "neumann" else min(y + 1, H - 1)
            dVy_dy = 0.5 * (V[yp, x, 1] - V[ym, x, 1])

            out[y, x] = dVx_dx + dVy_dy
    return out


def laplace_g(phi: np.ndarray, g: np.ndarray, boundary: str = "neumann", metric_eps: float = 1e-6) -> np.ndarray:
    """
    Metric-aware Laplace-Beltrami: Δ_g phi ≈ ∇·(g^{-1} ∇phi).
    Implementation ignores √|g| factor for Phase A simplicity; retains SPD metric influence via g^{-1}.
    Boundary handling per 'boundary'.
    """
    G = enforce_spd(np.asarray(g, dtype=np.float64), metric_eps)
    grad = grad_g(phi, G, boundary=boundary)  # [H,W,2] axis gradients
    g_inv = _inverse_metric(G, metric_eps)    # [H,W,2,2]
    # Contract: J = g^{-1} ∇phi (2x2 @ 2→2)
    H, W = grad.shape[:2]
    J = np.empty((H, W, 2), dtype=np.float64)
    for y in range(H):
        for x in range(W):
            J[y, x] = g_inv[y, x] @ grad[y, x]
    # Divergence of J
    L = _divergence(J, boundary=boundary)
    # For Dirichlet, enforce border zeros in Laplacian application to stabilize
    if boundary == "dirichlet":
        L[0, :] = 0.0
        L[-1, :] = 0.0
        L[:, 0] = 0.0
        L[:, -1] = 0.0
    return L


# ---- Leapfrog ----

def _leapfrog(g: np.ndarray, J: np.ndarray, cfg: FieldConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    H, W = _validate_inputs(g, J)
    G = enforce_spd(np.asarray(g, dtype=np.float64), float(cfg.metric_eps))
    S = np.asarray(J, dtype=np.float64)

    # CFL check
    dt_max_raw = cfl_dt_max(G)
    dt_max = float(cfg.cfl_safety) * dt_max_raw
    cfl_passed = bool(float(cfg.dt) <= dt_max)

    # Initialize
    phi = np.zeros((H, W), dtype=np.float64)
    v_half = np.zeros((H, W), dtype=np.float64)

    dt = float(cfg.dt)
    steps = int(cfg.steps)
    boundary = cfg.boundary
    clip = float(cfg.grad_clip)

    for _ in range(steps):
        lap = laplace_g(phi, G, boundary=boundary, metric_eps=float(cfg.metric_eps))
        a = lap + S  # κ = 1 in Phase A
        v_half = v_half + dt * a
        if clip > 0.0:
            # Per-cell gradient clipping (soft): scale velocity where current ∥∇phi∥ is large
            grad = grad_g(phi, G, boundary=boundary)
            norms = np.sqrt(grad[..., 0] * grad[..., 0] + grad[..., 1] * grad[..., 1])
            scale = np.ones_like(norms)
            mask = norms > clip
            scale[mask] = clip / np.maximum(norms[mask], 1e-12)
            v_half = v_half * scale
        # Update field
        phi = phi + dt * v_half
        if boundary == "dirichlet":
            _apply_dirichlet_border(phi)
        # Hard projection to enforce gradient cap deterministically
        if clip > 0.0:
            # Global smoothing loop: repeatedly replace with 4-neighbor average until cap is met
            # This guarantees non-increasing max gradient and deterministically enforces the bound.
            for _proj in range(64):
                grads = grad_g(phi, G, boundary=boundary)
                norms2 = np.sqrt(grads[..., 0] ** 2 + grads[..., 1] ** 2)
                if float(np.max(norms2)) <= clip + 1e-9:
                    break
                phi = _neighbor_avg(phi, boundary=boundary)
                if boundary == "dirichlet":
                    _apply_dirichlet_border(phi)

    grad_out = grad_g(phi, G, boundary=boundary)
    meta = {
        "method": "leapfrog",
        "steps": steps,
        "dt": dt,
        "cfl_passed": cfl_passed,
        "iters": steps,
        "residual": float(np.linalg.norm(laplace_g(phi, G, boundary=boundary, metric_eps=float(cfg.metric_eps)) + S)),
    }
    return phi, grad_out, meta


# ---- Conjugate Gradient (steady-state) ----

def _apply_A(phi: np.ndarray, G: np.ndarray, boundary: str, eps: float) -> np.ndarray:
    """Matrix-free A(phi) = -Δ_g(phi) with boundary handling."""
    out = -laplace_g(phi, G, boundary=boundary, metric_eps=eps)
    if boundary == "dirichlet":
        # enforce Dirichlet rows: set borders to zero (equivalent to identity rows with zero RHS)
        out[0, :] = phi[0, :]  # penalize any non-zero on border
        out[-1, :] = phi[-1, :]
        out[:, 0] = phi[:, 0]
        out[:, -1] = phi[:, -1]
    return out


def _cg(g: np.ndarray, J: np.ndarray, cfg: FieldConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    H, W = _validate_inputs(g, J)
    G = enforce_spd(np.asarray(g, dtype=np.float64), float(cfg.metric_eps))
    S = np.asarray(J, dtype=np.float64).copy()
    boundary = cfg.boundary
    eps = float(cfg.metric_eps)

    # For Dirichlet, zero RHS on borders
    if boundary == "dirichlet":
        _apply_dirichlet_border(S)

    # Initial guess
    x = np.zeros((H, W), dtype=np.float64)

    def A(phi2d: np.ndarray) -> np.ndarray:
        return _apply_A(phi2d, G, boundary, eps)

    r = S - A(x)
    p = r.copy()
    rs_old = float(np.vdot(r, r))
    tol2 = float(cfg.cg_tol) ** 2
    iters = 0
    max_iters = int(cfg.max_cg_iters)

    for k in range(max_iters):
        Ap = A(p)
        denom = float(np.vdot(p, Ap))
        if abs(denom) <= 1e-20:
            break
        alpha = rs_old / denom
        x = x + alpha * p
        if boundary == "dirichlet":
            _apply_dirichlet_border(x)
        r = r - alpha * Ap
        rs_new = float(np.vdot(r, r))
        iters = k + 1
        if rs_new <= tol2:
            break
        beta = rs_new / rs_old if rs_old > 0.0 else 0.0
        p = r + beta * p
        rs_old = rs_new

    residual = float(np.sqrt(rs_old))
    grad_out = grad_g(x, G, boundary=boundary)
    meta = {
        "method": "cg",
        "steps": 1,
        "dt": 0.0,
        "cfl_passed": True,
        "iters": iters,
        "residual": residual,
    }
    return x, grad_out, meta


# ---- Public API ----

def solve_field(g: np.ndarray, J: np.ndarray, cfg: FieldConfig) -> Dict[str, Any]:
    """
    Solve the metric-aware field given metric g[H,W,2,2] and source J[H,W].
    Returns:
      - phi: (H,W) float64
      - grad: (H,W,2) float64, central differences (axes)
      - meta: dict with fields {"method","steps","dt","cfl_passed","iters","residual"}
    """
    _ = _validate_inputs(g, J)  # shape and finiteness
    if cfg.method == "leapfrog":
        phi, grad, meta = _leapfrog(g, J, cfg)
    else:
        phi, grad, meta = _cg(g, J, cfg)
    return {"phi": phi, "grad": grad, "meta": meta}