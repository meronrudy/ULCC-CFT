"""Geodesic dynamics and simple shooting solver.

Invariants
- Uses Levi–Civita connection from metric (Γ assumed compatible with g).
- Semi-implicit (symplectic-Euler) integrator preserves form of geodesic flow.
- Deterministic; numpy-only; first-order accurate.

Notes
- Reduces to straight line for flat case (Γ=0) with constant metric g.
"""

from __future__ import annotations

import numpy as np

__all__ = ["geodesic_step", "shoot_geodesic"]


def _validate_common(theta: np.ndarray, v: np.ndarray, g: np.ndarray, Gamma: np.ndarray) -> int:
    """Validate shapes/values common to geodesic routines; return dimension n."""
    theta = np.asarray(theta, dtype=float)
    v = np.asarray(v, dtype=float)
    g = np.asarray(g, dtype=float)
    Gamma = np.asarray(Gamma, dtype=float)

    if theta.ndim != 1:
        raise ValueError("theta must be 1-D.")
    if v.ndim != 1:
        raise ValueError("v must be 1-D.")
    n = theta.shape[0]
    if v.shape != (n,):
        raise ValueError(f"v must have shape ({n},); got {v.shape}.")
    if g.shape != (n, n):
        raise ValueError(f"g must have shape ({n}, {n}); got {g.shape}.")
    if Gamma.shape != (n, n, n):
        raise ValueError(f"Γ must have shape ({n}, {n}, {n}); got {Gamma.shape}.")

    if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(v))):
        raise ValueError("theta and v must contain finite values.")
    if not np.all(np.isfinite(g)):
        raise ValueError("g must contain finite values.")
    if not np.all(np.isfinite(Gamma)):
        raise ValueError("Γ must contain finite values.")

    if not np.allclose(g, g.T, atol=1e-12, rtol=0.0):
        raise ValueError("g must be symmetric.")
    eig = np.linalg.eigvalsh(g)
    if np.min(eig) <= 0.0:
        raise ValueError("g must be positive-definite (eigvalsh(g) > 0).")
    return n


def geodesic_step(
    theta: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    Γ: np.ndarray,
    dt: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform one semi-implicit (symplectic-Euler) step of the geodesic ODE:
        a^a = - Γ^a_{bc} v^b v^c
        v_new = v + dt * a
        theta_new = theta + dt * v_new

    Validation:
    - theta, v are 1-D with same length n; finite.
    - g has shape (n, n), symmetric positive-definite (eigvalsh(g) > 0); finite.
    - Γ has shape (n, n, n); finite.
    - dt > 0 and finite.

    Invariant:
    - Path extremizes length functional for γ=0, V=0; reduces to straight line
      when Γ=0 with constant g. Deterministic; first-order accurate.

    Returns:
        theta_new, v_new
    """
    # Validate inputs
    _ = _validate_common(theta, v, g, Γ)
    if not (dt > 0 and np.isfinite(dt)):
        raise ValueError("dt must be a positive finite float.")

    # Ensure arrays
    theta = np.asarray(theta, dtype=float)
    v = np.asarray(v, dtype=float)
    Γ = np.asarray(Γ, dtype=float)

    # Acceleration: a^a = - Γ^a_{bc} v^b v^c
    acc = -np.einsum("abc,b,c->a", Γ, v, v)
    v_new = v + dt * acc
    theta_new = theta + dt * v_new
    return theta_new, v_new


def shoot_geodesic(
    theta0: np.ndarray,
    theta1: np.ndarray,
    g: np.ndarray,
    Γ: np.ndarray,
    T: float = 1.0,
    steps: int = 200,
    tol: float = 1e-8,
    max_iters: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple shooting method to find initial velocity v0 so that integrating the
    geodesic from theta0 reaches theta1 at time T.

    Algorithm:
    - Initialize v0 = (theta1 - theta0) / T.
    - For k in 0..max_iters:
        * Integrate for 'steps' with dt = T/steps using geodesic_step.
        * Compute error e = theta_T - theta1.
        * If ||e|| <= tol: return (path_thetas, v0).
        * Update v0 <- v0 - e / T  (linearized correction; exact for Γ=0).
    - After loop, assert ||e|| <= 10*tol and return latest (path_thetas, v0).

    Validation:
    - theta0, theta1 are 1-D with same length n; finite.
    - g shape (n, n), symmetric positive-definite; finite.
    - Γ shape (n, n, n); finite.
    - T > 0 finite; steps >= 1 integer; tol >= 0 finite; max_iters >= 0 integer.

    Designed to be exact in the flat Euclidean case (Γ=0, g=I) with the initial
    guess; iterative refinement for mild curvature; deterministic.

    Returns:
        path_thetas: array of shape (steps+1, n) including endpoints
        v0: the (refined) initial velocity that produced the path
    """
    theta0 = np.asarray(theta0, dtype=float)
    theta1 = np.asarray(theta1, dtype=float)
    g = np.asarray(g, dtype=float)
    Γ = np.asarray(Γ, dtype=float)

    if theta0.ndim != 1 or theta1.ndim != 1:
        raise ValueError("theta0 and theta1 must be 1-D.")
    n = theta0.shape[0]
    if theta1.shape != (n,):
        raise ValueError(f"theta1 must have shape ({n},); got {theta1.shape}.")
    if g.shape != (n, n):
        raise ValueError(f"g must have shape ({n}, {n}); got {g.shape}.")
    if Γ.shape != (n, n, n):
        raise ValueError(f"Γ must have shape ({n}, {n}, {n}); got {Γ.shape}.")

    if not (np.all(np.isfinite(theta0)) and np.all(np.isfinite(theta1))):
        raise ValueError("theta0 and theta1 must be finite.")
    if not np.all(np.isfinite(g)):
        raise ValueError("g must contain finite values.")
    if not np.all(np.isfinite(Γ)):
        raise ValueError("Γ must contain finite values.")
    if not np.allclose(g, g.T, atol=1e-12, rtol=0.0):
        raise ValueError("g must be symmetric.")
    eig = np.linalg.eigvalsh(g)
    if np.min(eig) <= 0.0:
        raise ValueError("g must be positive-definite (eigvalsh(g) > 0).")

    if not (T > 0 and np.isfinite(T)):
        raise ValueError("T must be a positive finite float.")
    if not (isinstance(steps, int) and steps >= 1):
        raise ValueError("steps must be an integer >= 1.")
    if not (tol >= 0 and np.isfinite(tol)):
        raise ValueError("tol must be a nonnegative finite float.")
    if not (isinstance(max_iters, int) and max_iters >= 0):
        raise ValueError("max_iters must be an integer >= 0.")

    dt = T / float(steps)

    v0 = (theta1 - theta0) / T
    last_path = None
    e = None
    for _ in range(max_iters + 1):
        # Integrate path with current v0
        path = np.empty((steps + 1, n), dtype=float)
        path[0] = theta0
        theta = theta0.copy()
        v = v0.copy()
        for k in range(steps):
            theta, v = geodesic_step(theta, v, g, Γ, dt)
            path[k + 1] = theta

        last_path = path
        e = path[-1] - theta1
        err = float(np.linalg.norm(e, ord=2))
        if err <= tol:
            return path, v0
        # Linearized correction: exact in flat case
        v0 = v0 - e / T

    # After iterations, ensure acceptable error (flat cases must pass)
    assert e is not None and last_path is not None  # for type checkers
    final_err = float(np.linalg.norm(e, ord=2))
    if final_err > 10.0 * tol:
        raise AssertionError(
            f"Shooting did not converge sufficiently: ||e|| = {final_err} > 10*tol"
        )
    return last_path, v0
