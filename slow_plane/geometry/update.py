"""Geometry update (Phase A — E3c): single damped CFE/Ricci-like step with SPD projection,
condition-number clamp, and simple trust-region acceptance with hysteresis.

Residual model (fixed Phase A constants):
- R ≈ α1 * (∇∇Φ sym) + α2 * (div B)·I + α3 * (J - mean(J))·I + α3_abs * |J - mean(J)|·I + α4 * (ΔU)·I
  where I is 2x2 identity, applied per cell.
- Constants (documented and intentionally simple):
    α1 = 1.0
    α2 = 0.10
    α3 = 0.05
    α3_abs = 0.20
    α4 = 0.10

Trust-region and acceptance:
- Candidate: g_cand = Π_SPD,cond( sym(g - η·γ·R) )
- Let Δg = g_cand - g; use mean Frobenius norms across grid as proxies:
    delta_norm = mean(||Δg||_F)
    residual_norm = mean(||R||_F)
- Predicted decrease proxy: pred_drop = (η·γ) * residual_norm
- Actual change proxy:   act_drop = delta_norm
- Accept if: act_drop ≤ trust_radius AND (act_drop / max(pred_drop,eps)) ≥ accept_ratio_min
- If accepted and ratio is high (≥0.9) and hysteresis_left == 0, slightly increase radius (×1.2).
- If rejected, shrink radius (×0.5) and set hysteresis_left = cfg.hysteresis.
- On acceptance with hysteresis_left > 0, decrement by 1 and do not increase the radius.

Boundary handling:
- Hessian/second derivatives: central differences on interior; zero second derivative at borders.
- Divergence of B: if dict {"N","S","E","W"} or tensor [H,W,4] with channel order [N,E,S,W].
  Build vector field V = (E - W, N - S). Compute forward-difference divergence with zero-flux
  at domain borders (differences that would step out-of-bounds are treated as 0).

Determinism:
- Pure numpy, no randomness. All loops are deterministic and stable under identical inputs.

Inputs are validated for shapes, finiteness; g is SPD-projected with eigenvalue clamp and condition bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, Union
import numpy as np


# ---- Public configuration ----

@dataclass(frozen=True)
class GeometryConfig:
    # Step and damping
    step_size: float = 0.2            # η > 0
    damping: float = 0.7              # γ in (0,1]
    # SPD and condition constraints
    cond_max: float = 1e5             # ≥ 1
    spd_eps: float = 1e-6             # > 0
    # Trust region
    trust_radius: float = 0.25        # τ > 0 (mean Frobenius of Δg)
    accept_ratio_min: float = 0.5     # in (0,1)
    hysteresis: int = 2               # integer ≥ 0

    def __post_init__(self) -> None:
        if not (float(self.step_size) > 0.0):
            raise ValueError("step_size must be > 0")
        if not (0.0 < float(self.damping) <= 1.0):
            raise ValueError("damping must be in (0,1]")
        if not (float(self.cond_max) >= 1.0):
            raise ValueError("cond_max must be ≥ 1")
        if not (float(self.spd_eps) > 0.0):
            raise ValueError("spd_eps must be > 0")
        if not (float(self.trust_radius) > 0.0):
            raise ValueError("trust_radius must be > 0")
        if not (0.0 < float(self.accept_ratio_min) < 1.0):
            raise ValueError("accept_ratio_min must be in (0,1)")
        if int(self.hysteresis) < 0:
            raise ValueError("hysteresis must be ≥ 0")


# ---- Constants ----

_ALPHA1 = 1.0
_ALPHA2 = 0.10
_ALPHA3 = 0.05
_ALPHA3_ABS = 0.20
_ALPHA4 = 0.10

_EPS = 1e-12


# ---- Utilities ----

def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.swapaxes(-1, -2))


def ensure_spd_and_cond(g: np.ndarray, eps: float, cond_max: float) -> np.ndarray:
    """
    Symmetrize, clamp eigenvalues ≥ eps, and cap condition number by clipping
    eigenvalues to [lam_min, lam_min*cond_max] per cell deterministically.
    """
    G = np.asarray(g, dtype=np.float64)
    if G.ndim != 4 or G.shape[-2:] != (2, 2):
        raise ValueError("ensure_spd_and_cond: g must have shape [H,W,2,2]")
    H, W = G.shape[:2]
    out = np.empty_like(G, dtype=np.float64)
    for y in range(H):
        for x in range(W):
            A = _sym(G[y, x])
            w, V = np.linalg.eigh(A)
            w = np.real(w)
            V = np.real(V)
            # Clamp min
            w = np.maximum(w, float(eps))
            # Cap condition number by clipping max eigenvalue(s)
            lam_min = float(np.min(w))
            lam_max_bound = lam_min * float(cond_max)
            w = np.minimum(w, lam_max_bound)
            A_proj = (V * w) @ V.T
            out[y, x] = _sym(A_proj)
    return out


def _validate_inputs(
    g: np.ndarray,
    phi: np.ndarray,
    grad: np.ndarray,
    U: np.ndarray,
    J: np.ndarray,
    B: Union[Dict[str, np.ndarray], np.ndarray],
) -> Tuple[int, int]:
    G = np.asarray(g, dtype=np.float64)
    PH = np.asarray(phi, dtype=np.float64)
    GR = np.asarray(grad, dtype=np.float64)
    UU = np.asarray(U, dtype=np.float64)
    JJ = np.asarray(J, dtype=np.float64)

    if G.ndim != 4 or G.shape[-2:] != (2, 2):
        raise ValueError("g must have shape [H,W,2,2]")
    H, W = G.shape[:2]
    if PH.shape != (H, W):
        raise ValueError("phi must have shape [H,W]")
    if GR.shape != (H, W, 2):
        raise ValueError("grad must have shape [H,W,2]")
    if UU.shape != (H, W):
        raise ValueError("U must have shape [H,W]")
    if JJ.shape != (H, W):
        raise ValueError("J must have shape [H,W]")

    if isinstance(B, dict):
        for k in ("N", "S", "E", "W"):
            if k not in B:
                raise ValueError("B dict must contain keys 'N','S','E','W'")
            arr = np.asarray(B[k], dtype=np.float64)
            if arr.shape != (H, W):
                raise ValueError("B[%r] must have shape [H,W]" % k)
    else:
        BB = np.asarray(B, dtype=np.float64)
        if BB.ndim != 3 or BB.shape != (H, W, 4):
            raise ValueError("B tensor must have shape [H,W,4] with channels [N,E,S,W]")
    # Finiteness
    if not (np.all(np.isfinite(G)) and np.all(np.isfinite(PH)) and np.all(np.isfinite(GR))
            and np.all(np.isfinite(UU)) and np.all(np.isfinite(JJ))):
        raise ValueError("inputs must be finite")
    return H, W


def _hessian_sym(phi: np.ndarray) -> np.ndarray:
    """Per-cell symmetric Hessian H = [[dxx, dxy],[dxy, dyy]]. Zero at borders."""
    P = np.asarray(phi, dtype=np.float64)
    Hh, Ww = P.shape
    Hs = np.zeros((Hh, Ww, 2, 2), dtype=np.float64)
    for y in range(Hh):
        for x in range(Ww):
            if 0 < x < Ww - 1 and 0 < y < Hh - 1:
                dxx = P[y, x + 1] - 2.0 * P[y, x] + P[y, x - 1]
                dyy = P[y + 1, x] - 2.0 * P[y, x] + P[y - 1, x]
                dxy = 0.25 * (
                    P[y + 1, x + 1]
                    - P[y + 1, x - 1]
                    - P[y - 1, x + 1]
                    + P[y - 1, x - 1]
                )
            else:
                dxx = 0.0
                dyy = 0.0
                dxy = 0.0
            Hs[y, x, 0, 0] = dxx
            Hs[y, x, 1, 1] = dyy
            Hs[y, x, 0, 1] = dxy
            Hs[y, x, 1, 0] = dxy
    return Hs


def _laplacian_5pt(arr: np.ndarray) -> np.ndarray:
    """5-point Laplacian Δ with zero second-derivative at borders (set to 0 on borders)."""
    A = np.asarray(arr, dtype=np.float64)
    H, W = A.shape
    L = np.zeros((H, W), dtype=np.float64)
    for y in range(H):
        for x in range(W):
            if 0 < x < W - 1 and 0 < y < H - 1:
                L[y, x] = (A[y, x + 1] - 2.0 * A[y, x] + A[y, x - 1]) + (
                    A[y + 1, x] - 2.0 * A[y, x] + A[y - 1, x]
                )
            else:
                L[y, x] = 0.0
    return L


def _divergence_B(B: Union[Dict[str, np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Build vector field V=(Vx,Vy) with Vx = E - W, Vy = N - S, then forward-diff divergence:
      div = (Vx[y,x+1]-Vx[y,x]) + (Vy[y+1,x]-Vy[y,x])
    Differences that step out of bounds are treated as zero (zero-flux at border).
    """
    if isinstance(B, dict):
        N = np.asarray(B["N"], dtype=np.float64)
        S = np.asarray(B["S"], dtype=np.float64)
        E = np.asarray(B["E"], dtype=np.float64)
        W = np.asarray(B["W"], dtype=np.float64)
    else:
        BB = np.asarray(B, dtype=np.float64)
        # channel order [N,E,S,W]
        N = BB[..., 0]
        E = BB[..., 1]
        S = BB[..., 2]
        W = BB[..., 3]
    H, Wd = N.shape
    Vx = E - W
    Vy = N - S
    div = np.zeros((H, Wd), dtype=np.float64)
    for y in range(H):
        for x in range(Wd):
            dVx = 0.0 if x == Wd - 1 else (Vx[y, x + 1] - Vx[y, x])
            dVy = 0.0 if y == H - 1 else (Vy[y + 1, x] - Vy[y, x])
            div[y, x] = dVx + dVy
    return div


# ---- Residual and step ----

def cfe_residual(
    g: np.ndarray,
    phi: np.ndarray,
    U: np.ndarray,
    J: np.ndarray,
    B: Union[Dict[str, np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    Compute residual R per-cell 2x2:
      R = α1*(∇∇Φ sym) + [ α2*(div B) + α3*(J - mean(J)) + α4*(ΔU) ] * I
    """
    _ = g  # reserved for future coupling; not used in Phase A residual
    Hs = _hessian_sym(phi)             # [H,W,2,2]
    divB = _divergence_B(B)            # [H,W]
    Jc = np.asarray(J, dtype=np.float64) - float(np.asarray(J, dtype=np.float64).mean())
    absJ = np.abs(Jc)
    dU = _laplacian_5pt(U)
    H, W = divB.shape
    I2 = np.eye(2, dtype=np.float64)
    R = np.empty((H, W, 2, 2), dtype=np.float64)
    for y in range(H):
        for x in range(W):
            scalar = _ALPHA2 * divB[y, x] + _ALPHA3 * Jc[y, x] + _ALPHA3_ABS * absJ[y, x] + _ALPHA4 * dU[y, x]
            R[y, x] = _ALPHA1 * Hs[y, x] + scalar * I2
    return R


def apply_step(g: np.ndarray, R: np.ndarray, cfg: GeometryConfig) -> np.ndarray:
    """Symmetrize and apply damped step, then enforce SPD and condition."""
    s = float(cfg.step_size) * float(cfg.damping)
    g_sym = _sym(np.asarray(g, dtype=np.float64))
    trial = g_sym - s * np.asarray(R, dtype=np.float64)
    return ensure_spd_and_cond(trial, float(cfg.spd_eps), float(cfg.cond_max))


def _mean_frobenius(A: np.ndarray) -> float:
    """Mean Frobenius norm over grid of 2x2 tensors."""
    T = np.asarray(A, dtype=np.float64)
    H, W = T.shape[:2]
    acc = 0.0
    for y in range(H):
        for x in range(W):
            acc += float(np.linalg.norm(T[y, x], ord="fro"))
    return acc / float(H * W)


def accept_with_trust(
    g: np.ndarray,
    g_cand: np.ndarray,
    R: np.ndarray,
    cfg: GeometryConfig,
    trust_radius: float,
    hysteresis_left: int,
) -> Tuple[bool, float, int, float]:
    """
    Return (accepted, new_trust_radius, new_hysteresis_left, accept_ratio)
    - Proxy predicted drop = (η·γ) * mean||R||_F
    - Actual change = mean||Δg||_F
    """
    s = float(cfg.step_size) * float(cfg.damping)
    delta = np.asarray(g_cand, dtype=np.float64) - np.asarray(g, dtype=np.float64)
    delta_norm = _mean_frobenius(delta)
    res_norm = _mean_frobenius(R)
    pred_drop = s * res_norm
    ratio = float(delta_norm / max(pred_drop, _EPS))

    accepted = (delta_norm <= float(trust_radius)) and (ratio >= float(cfg.accept_ratio_min))

    new_tr = float(trust_radius)
    new_hyst = int(hysteresis_left)

    if accepted:
        if new_hyst > 0:
            new_hyst = new_hyst - 1
            # inhibit radius increase while cooling down
        else:
            if ratio >= 0.9:
                new_tr = new_tr * 1.2
    else:
        # rejection: shrink radius and reset hysteresis counter
        new_tr = max(new_tr * 0.5, _EPS)
        new_hyst = int(cfg.hysteresis)

    return accepted, new_tr, new_hyst, ratio


# ---- Public API ----

def update_geometry(
    g: np.ndarray,
    phi: np.ndarray,
    grad: np.ndarray,
    U: np.ndarray,
    J: np.ndarray,
    B: Union[Dict[str, np.ndarray], np.ndarray],
    cfg: GeometryConfig,
) -> Dict[str, Any]:
    """
    Perform a single geometry update step with trust-region acceptance.
    Inputs:
      - g: [H,W,2,2] SPD (or will be projected)
      - phi: [H,W]
      - grad: [H,W,2] (validated; not used by residual in Phase A)
      - U: [H,W]
      - J: [H,W]
      - B: dict {"N","S","E","W"} or tensor [H,W,4] with channels [N,E,S,W]
    Outputs:
      - dict {"g_next": [H,W,2,2], "meta": {...}}
        meta:
          - accepted: bool
          - accept_ratio: float
          - residual_norm: float (mean Frobenius of R)
          - trust_radius: float (possibly updated)
          - hysteresis_left: int
    """
    H, W = _validate_inputs(g, phi, grad, U, J, B)
    # Ensure SPD/cond on the current metric before computing anything
    g0 = ensure_spd_and_cond(np.asarray(g, dtype=np.float64), float(cfg.spd_eps), float(cfg.cond_max))

    # Residual
    R = cfe_residual(g0, phi, U, J, B)
    res_norm = _mean_frobenius(R)

    # Candidate
    g_cand = apply_step(g0, R, cfg)
    # Trust check
    accepted, tr_new, hyst_new, ratio = accept_with_trust(
        g0, g_cand, R, cfg, float(cfg.trust_radius), int(cfg.hysteresis)
    )

    g_next = g_cand if accepted else g0

    # Final safety: enforce SPD+cond again to be strict
    g_next = ensure_spd_and_cond(g_next, float(cfg.spd_eps), float(cfg.cond_max))

    meta = {
        "accepted": bool(accepted),
        "accept_ratio": float(ratio),
        "residual_norm": float(res_norm),
        "trust_radius": float(tr_new),
        "hysteresis_left": int(hyst_new),
    }
    return {"g_next": g_next, "meta": meta}