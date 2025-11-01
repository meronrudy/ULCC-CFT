"""Christoffel symbols for simple 1D manifolds (Bernoulli)."""
from .fisher import fisher_metric_bernoulli as g

def gamma_theta(theta: float) -> float:
    """Return Γ^θ_{θθ} for Bernoulli scalar manifold.
    Γ = (2θ - 1) / (2 θ (1-θ))
    """
    if not (0.0 < theta < 1.0):
        raise ValueError("theta must be in (0,1)")
    return (2.0*theta - 1.0) / (2.0 * theta * (1.0 - theta))
