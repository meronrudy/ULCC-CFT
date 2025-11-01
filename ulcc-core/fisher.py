"""Fisher information metrics for simple families."""
import numpy as np

def fisher_metric_bernoulli(theta: float) -> float:
    """Return scalar Fisher metric g(theta) for Bernoulli(theta).

    g(theta) = 1 / (theta * (1 - theta))
    Domain: 0 < theta < 1
    """
    if not (0.0 < theta < 1.0):
        raise ValueError("theta must be in (0,1)")
    return 1.0 / (theta * (1.0 - theta))

def fisher_metric_inv_bernoulli(theta: float) -> float:
    """Inverse metric g^{-1}(theta) = theta(1-theta)."""
    if not (0.0 < theta < 1.0):
        raise ValueError("theta must be in (0,1)")
    return theta * (1.0 - theta)

def fisher_metric_diag_gaussian(var: np.ndarray) -> np.ndarray:
    """Diagonal Fisher metric for N(mu, diag(var)).
    For a diagonal Gaussian with fixed mean, the Fisher metric in variance
    parameters (sigma^2) is: g = 1/(2 var^2) elementwise.
    This is a placeholder; Phase-0 uses Bernoulli primarily.
    """
    var = np.asarray(var, dtype=float)
    if np.any(var <= 0):
        raise ValueError("variances must be positive")
    return 1.0 / (2.0 * (var ** 2))
