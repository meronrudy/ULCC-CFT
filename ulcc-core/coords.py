"""Coordinate transforms for Bernoulli."""
import numpy as np

def theta_to_phi(theta: float) -> float:
    """φ = arcsin(2θ - 1). Domain θ in (0,1), φ in (-π/2, π/2)."""
    if not (0.0 < theta < 1.0):
        raise ValueError("theta must be in (0,1)")
    return np.arcsin(2.0*theta - 1.0)

def phi_to_theta(phi: float) -> float:
    """Inverse transform: θ = (sin φ + 1)/2."""
    return 0.5*(np.sin(phi) + 1.0)
