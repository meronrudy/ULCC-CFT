"""Second-order Lagrangian dynamics with friction and potential.

Invariants and notes:
- For γ=0 and V≡0, trajectories extremize the action (Euler–Lagrange).
- Reduces to geodesic motion when gradV ≡ 0 and γ = 0.
- Overdamped limit (γ ≫ 1): behavior approaches Natural Gradient Descent with step lr ≈ dt/γ.
- Deterministic; numpy-only.
"""

from typing import Callable
import numpy as np


def lagrangian_step(
    theta: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    gradV: Callable[[np.ndarray], np.ndarray],
    Γ: np.ndarray,
    γ: float = 0.0,
    dt: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform one semi-implicit symplectic Euler step for second-order Lagrangian dynamics
    with Rayleigh friction and potential forces on a Riemannian manifold.

    Equations (structure-preserving, first-order accurate):
        acc = -einsum('abc,b,c->a', Γ, v, v) - γ * v - inv(g) @ gradV(theta)
        v_new = v + dt * acc
        theta_new = theta + dt * v_new

    Validation:
    - theta, v are 1-D and same length n.
    - g has shape (n, n) and is symmetric positive-definite (eigvalsh(g) > 0).
    - Γ has shape (n, n, n).
    - gradV(theta) returns shape (n,).

    Invariants/Notes:
    - For γ=0 and V≡0, the scheme extremizes action; when gradV≡0 and γ=0, it reduces to geodesic motion.
    - In the overdamped limit γ ≫ 1, behavior approaches NGD with learning rate lr ≈ dt/γ.
    - Deterministic; numpy-only.

    Returns:
        (theta_new, v_new)
    """
    # Coerce inputs
    theta = np.asarray(theta, dtype=float)
    v = np.asarray(v, dtype=float)
    g = np.asarray(g, dtype=float)
    Γ = np.asarray(Γ, dtype=float)

    # Shape validations
    if theta.ndim != 1 or v.ndim != 1:
        raise ValueError("theta and v must be 1-D arrays.")
    n = theta.shape[0]
    if v.shape[0] != n:
        raise ValueError(f"theta and v must have the same length; got {theta.shape[0]} and {v.shape[0]}.")
    if g.shape != (n, n):
        raise ValueError(f"g must have shape ({n}, {n}); got {g.shape}.")
    if Γ.shape != (n, n, n):
        raise ValueError(f"Γ must have shape ({n}, {n}, {n}); got {Γ.shape}.")

    # Finite checks
    if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(v)) and np.all(np.isfinite(g)) and np.all(np.isfinite(Γ))):
        raise ValueError("All inputs must be finite.")

    # SPD metric validation
    if not np.allclose(g, g.T, atol=1e-12, rtol=0.0):
        raise ValueError("g must be symmetric.")
    eigvals = np.linalg.eigvalsh(g)
    if np.min(eigvals) <= 0.0:
        raise ValueError("g must be positive-definite (eigvalsh(g) > 0).")

    # Step parameters validation
    if not np.isfinite(γ):
        raise ValueError("γ must be a finite float.")
    if not (dt > 0 and np.isfinite(dt)):
        raise ValueError("dt must be a positive finite float.")

    # Gradient evaluation and validation
    grad = np.asarray(gradV(theta), dtype=float)
    if grad.shape != (n,):
        raise ValueError(f"gradV(theta) must return shape ({n},); got {grad.shape}.")
    if not np.all(np.isfinite(grad)):
        raise ValueError("gradV(theta) must be finite.")

    # Compute acceleration
    geodesic_term = np.einsum('abc,b,c->a', Γ, v, v)  # Γ^a_{bc} v^b v^c
    nat_grad = np.linalg.solve(g, grad)               # inv(g) @ gradV(theta) without explicit inverse
    acc = -geodesic_term - γ * v - nat_grad

    # Semi-implicit symplectic Euler update
    v_new = v + dt * acc
    theta_new = theta + dt * v_new
    return theta_new, v_new
