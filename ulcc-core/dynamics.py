"""Geodesic and natural-gradient dynamics for Bernoulli."""
import numpy as np
from .christoffel import gamma_theta as Gamma
from .fisher import fisher_metric_inv_bernoulli as g_inv

def geodesic_rhs(t: float, y: np.ndarray) -> np.ndarray:
    """First-order ODE for state y = [theta, v] where v = dtheta/dt.
    dtheta/dt = v
    dv/dt = - Γ(theta) * v^2
    """
    theta, v = float(y[0]), float(y[1])
    dtheta = v
    dv = - Gamma(theta) * (v**2)
    return np.array([dtheta, dv], dtype=float)

def ngd_step(theta: float, gradL: float, lr: float) -> float:
    """One natural-gradient step in parameter θ with scalar metric inverse."""
    return float(theta - lr * g_inv(theta) * gradL)

def overdamped_second_order(theta: float, v: float, gradL: float, gamma: float, dt: float) -> tuple[float, float]:
    """Discretized second-order dynamics with friction gamma:
    dtheta = v
    dv = -Gamma(theta) v^2 - gamma v - g^{-1}(theta) gradL
    Returns (theta_next, v_next).
    """
    theta_next = theta + dt * v
    v_next = v + dt * ( - Gamma(theta) * (v**2) - gamma * v - g_inv(theta) * gradL )
    return float(theta_next), float(v_next)
