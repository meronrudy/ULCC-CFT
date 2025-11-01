"""Causal Field Equation (CFE) / Ricci-flow style geometric updates.

Deterministic, minimal implementations used by formal tests.

Key concepts
- Residual: r(g; I) = G(g) − κ · Π(I), where Π is symmetric projection and
  G is a surrogate Einstein operator.
- Update step: backtracking on step size to ensure residual reduction.

Defaults
- Residual uses trace-free operator by default for regularization.
- Update step uses the simple operator G(g)=g, yielding an affine contraction
  towards κ·Π(I). With α=0.95 and target_factor=0.1, we achieve ≥10× reduction.

All inputs/outputs are symmetrized; shapes are validated.
"""
from __future__ import annotations

import numpy as np

from geom.curvature import (
    symmetrize,
    project_information_tensor,
    einstein_operator_tracefree,
    einstein_operator_simple,
)


def _validate_same_square(g: np.ndarray, I: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(g, dtype=float)
    I = np.asarray(I, dtype=float)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError("g must be a square 2D array.")
    if I.ndim != 2 or I.shape[0] != I.shape[1]:
        raise ValueError("I must be a square 2D array.")
    if g.shape != I.shape:
        raise ValueError("g and I must have the same shape.")
    return g, I


def cfe_residual(
    g: np.ndarray,
    I: np.ndarray,
    kappa: float = 1.0,
    operator=einstein_operator_tracefree,
) -> np.ndarray:
    """
    Compute the CFE residual r = operator(g) − kappa · Π(I).

    - Inputs are symmetrized.
    - Output residual is symmetrized.
    - operator defaults to einstein_operator_tracefree for stability.

    Parameters
    ----------
    g : np.ndarray
        Metric-like matrix.
    I : np.ndarray
        Information/source tensor.
    kappa : float
        Coupling constant κ.
    operator : callable
        Surrogate Einstein operator, mapping (n,n) -> (n,n).

    Returns
    -------
    np.ndarray
        Symmetric residual matrix.
    """
    g, I = _validate_same_square(g, I)
    g_sym = symmetrize(g)
    I_proj = project_information_tensor(I)
    res = operator(g_sym) - float(kappa) * I_proj
    return symmetrize(res)


def cfe_update_step_residual(
    g: np.ndarray,
    I: np.ndarray,
    kappa: float = 1.0,
    alpha: float = 0.95,
    operator=einstein_operator_simple,
    target_factor: float | None = 0.1,
    max_backtracks: int = 10,
) -> np.ndarray:
    """
    Perform a single residual-minimization step with backtracking.

    Update rule (trial)
        g_new = symmetrize(g - α · r_old)
    where r_old = cfe_residual(g, I, κ; operator).

    Acceptance
    - If target_factor is provided (default 0.1), accept when
        ||r_new||_F ≤ target_factor · ||r_old||_F    (≥10× reduction for 0.1).
    - Else require strict decrease: ||r_new||_F < ||r_old||_F.
    - Backtracking halves α upon rejection, up to `max_backtracks`.

    Invariants asserted
    - On return, the acceptance condition holds.

    Parameters
    ----------
    g : np.ndarray
        Current iterate (square).
    I : np.ndarray
        Information/source tensor (square of same shape).
    kappa : float
        Coupling constant κ.
    alpha : float
        Initial step size (0 < α ≤ 1 recommended).
    operator : callable
        Operator used inside the residual and acceptance (default: simple).
    target_factor : float | None
        Desired reduction factor per step; set None for monotone decrease only.
    max_backtracks : int
        Maximum number of backtracking halvings of α.

    Returns
    -------
    np.ndarray
        The accepted updated matrix g_new (symmetrized).

    Raises
    ------
    AssertionError
        If acceptance condition cannot be met within max_backtracks.
    ValueError
        If shapes are inconsistent.
    """
    g, I = _validate_same_square(g, I)

    # Compute initial residual and its Frobenius norm
    r_old = cfe_residual(g, I, kappa=kappa, operator=operator)
    norm_old = float(np.linalg.norm(r_old, ord="fro"))

    # If already solved within numerical precision, return symmetrized g
    if norm_old == 0.0:
        return symmetrize(g)

    alpha_cur = float(alpha)
    g_new = symmetrize(g)
    accepted = False

    for _ in range(max_backtracks + 1):
        # Trial update with current step size
        g_trial = symmetrize(g - alpha_cur * r_old)
        r_new = cfe_residual(g_trial, I, kappa=kappa, operator=operator)
        norm_new = float(np.linalg.norm(r_new, ord="fro"))

        if target_factor is not None:
            if norm_new <= (target_factor * norm_old):
                g_new = g_trial
                accepted = True
                break
        else:
            if norm_new < norm_old:
                g_new = g_trial
                accepted = True
                break

        # Backtrack
        alpha_cur *= 0.5

    # Enforce acceptance
    if not accepted:
        # Final attempt already in g_new if any improved; assert condition regardless.
        # Compute final residual for assertion clarity.
        r_final = cfe_residual(g_new, I, kappa=kappa, operator=operator)
        norm_final = float(np.linalg.norm(r_final, ord="fro"))
        if target_factor is not None:
            assert norm_final <= target_factor * norm_old, (
                "Backtracking failed to achieve the requested residual reduction "
                f"({norm_final} > {target_factor} * {norm_old})."
            )
        else:
            assert norm_final < norm_old, (
                "Backtracking failed to reduce the residual strictly "
                f"({norm_final} >= {norm_old})."
            )
    else:
        # Also assert the acceptance condition here to document invariants
        r_final = cfe_residual(g_new, I, kappa=kappa, operator=operator)
        norm_final = float(np.linalg.norm(r_final, ord="fro"))
        if target_factor is not None:
            assert norm_final <= target_factor * norm_old
        else:
            assert norm_final < norm_old

    return g_new


def ricci_flow_step(g: np.ndarray, I: np.ndarray, kappa: float = 1.0) -> np.ndarray:
    """
    Forward Euler step for a surrogate Ricci-flow/CFE-like evolution.

    Minimal placeholder consistent with residual dynamics:
        g_{t+Δt} = g_t − r_simple(g_t; I)
    where r_simple uses operator G(g)=g. Thus, Δt=1 and
        r_simple(g; I) = g − κ·Π(I)  ⇒  g_{t+Δt} = κ·Π(I).

    This is intentionally simple and deterministic; it is not used by
    the current formal tests but provided for completeness.

    Parameters
    ----------
    g : np.ndarray
        Current metric-like matrix.
    I : np.ndarray
        Information/source tensor.
    kappa : float
        Coupling constant κ.

    Returns
    -------
    np.ndarray
        Updated, symmetrized matrix.
    """
    g, I = _validate_same_square(g, I)
    # One explicit Euler step with the simple operator residual
    step = cfe_residual(g, I, kappa=kappa, operator=einstein_operator_simple)
    return symmetrize(g - step)
