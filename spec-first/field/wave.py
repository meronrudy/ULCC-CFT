"""Wave equation on a metric-equipped discrete manifold.

Invariants
- Well-posed time stepping (CFL-like stability constraints)
- Discrete energy boundedness (where applicable)

Implements metric-aware discrete wave dynamics with deterministic numpy-only functions.
"""

from __future__ import annotations

import numpy as np

__all__ = ["dAlembertian", "wave_leapfrog_step", "wave_energy"]


def _validate_spd(g: np.ndarray) -> np.ndarray:
    """
    Validate that g is a finite symmetric positive-definite (SPD) matrix.
    Returns the array as float dtype.
    Raises ValueError on violation.
    """
    G = np.asarray(g, dtype=float)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError("g must be a square (n,n) matrix")
    if not np.all(np.isfinite(G)):
        raise ValueError("g must contain only finite values")
    if not np.allclose(G, G.T, atol=1e-12, rtol=0.0):
        raise ValueError("g must be symmetric")
    w = np.linalg.eigvalsh(G)
    if not np.all(w > 0.0):
        raise ValueError("g must be positive definite (eigvalsh > 0)")
    return G


def _validate_vector(x: np.ndarray, name: str) -> np.ndarray:
    """
    Validate that x is a finite 1-D vector and return float array view/copy.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def dAlembertian(phi: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Discrete d’Alembertian on a metric-equipped finite set; constant-mode nullspace mimics structure-preserving DDG; deterministic.
    Implementation:
    - Validate phi is 1-D (n,), g is (n,n) SPD, finite.
    - invg = inv(g)
    - Return invg @ (phi - phi.mean())  # removes constant mode
    """
    phi = _validate_vector(phi, "phi")
    G = _validate_spd(g)
    n = phi.shape[0]
    if G.shape[0] != n:
        raise ValueError("g shape does not match phi length")
    invg = np.linalg.inv(G)
    phi_zero_mean = phi - float(np.mean(phi))
    out = invg @ phi_zero_mean
    out = np.asarray(out, dtype=float).reshape(n)
    if not np.all(np.isfinite(out)):
        raise ValueError("dAlembertian produced non-finite values")
    return out


def wave_leapfrog_step(
    phi: np.ndarray,
    pi_half: np.ndarray,
    J: np.ndarray,
    g: np.ndarray,
    kappaC: float = 1.0,
    dt: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Invariant: approximately conserves discrete energy for J=0 under small dt; first-order consistent in time for the staggered state; deterministic.

    Leapfrog update (staggered):
      - Acceleration a = kappaC*J − dAlembertian(phi, g)
      - pi_half_new = pi_half + dt * a
      - phi_new = phi + dt * pi_half_new
    """
    phi = _validate_vector(phi, "phi")
    pi_half = _validate_vector(pi_half, "pi_half")
    J = _validate_vector(J, "J")
    G = _validate_spd(g)

    n = phi.shape[0]
    if pi_half.shape[0] != n or J.shape[0] != n:
        raise ValueError("phi, pi_half, and J must have the same length")
    if G.shape[0] != n:
        raise ValueError("g shape does not match state dimension")
    if not np.isfinite(kappaC):
        raise ValueError("kappaC must be finite")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a positive finite scalar")

    a = float(kappaC) * J - dAlembertian(phi, G)
    pi_half_new = pi_half + dt * a
    phi_new = phi + dt * pi_half_new
    return np.asarray(phi_new, dtype=float), np.asarray(pi_half_new, dtype=float)


def wave_energy(phi: np.ndarray, pi_half: np.ndarray, g: np.ndarray) -> float:
    """
    Invariant: E ≈ const for J=0 with stable dt; used in tests.
    Energy functional:
      - Kinetic: 0.5 * (pi_half · pi_half)
      - Potential: 0.5 * phi^T [dAlembertian(phi, g)]  # uses operator form; nullspace-safe
    """
    phi = _validate_vector(phi, "phi")
    pi_half = _validate_vector(pi_half, "pi_half")
    G = _validate_spd(g)
    n = phi.shape[0]
    if pi_half.shape[0] != n or G.shape[0] != n:
        raise ValueError("shape mismatch among phi, pi_half, and g")

    K = 0.5 * float(np.dot(pi_half, pi_half))
    Lphi = dAlembertian(phi, G)
    V = 0.5 * float(np.dot(phi, Lphi))
    E = float(K + V)
    if not np.isfinite(E):
        raise ValueError("Computed energy is not finite")
    return E
