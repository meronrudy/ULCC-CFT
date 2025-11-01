"""PGGS export utilities.

Invariants
- Pure export utilities; no mutation of model state
- B symmetric; ∇·J check deferred to geom/operators in a later task.
"""

from typing import Sequence, Tuple
import numpy as np


def export_results(samples: Sequence[Sequence[float]], weights: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute weighted mean current J in R^d and the causal flux tensor B.

    Inputs
    ------
    samples : sequence of 1D numeric vectors, consistent length d
    weights : 1D array-like positive weights

    Returns
    -------
    (J, B) : tuple
        J is a (d,) float64 ndarray; B is a (d,d) float64 symmetric ndarray.

    Notes
    -----
    We normalize weights internally to sum to 1 to avoid scale dependence.
    """
    S = [np.asarray(s, dtype=np.float64) for s in samples]
    if len(S) == 0:
        raise ValueError("samples must be non-empty")
    d = S[0].shape
    if len(d) != 1:
        raise ValueError("each sample must be a 1D vector")
    d0 = d[0]
    for s in S:
        if s.ndim != 1 or s.shape[0] != d0:
            raise ValueError("all samples must be 1D vectors of identical length")
    W = np.asarray(weights, dtype=np.float64)
    if W.ndim != 1 or W.shape[0] != len(S):
        raise ValueError("weights must be a 1D array-like matching number of samples")
    if not np.all(W > 0):
        raise ValueError("weights must be strictly positive")
    W = W / np.sum(W)
    A = np.vstack(S)  # (n, d)
    J = W @ A         # (d,)
    B0 = np.outer(J, J)
    B = 0.5 * (B0 + B0.T)
    if not np.allclose(B, B.T, rtol=1e-12, atol=1e-12):
        raise AssertionError("B must be symmetric")
    return J.astype(np.float64, copy=False), B.astype(np.float64, copy=False)


__all__ = ["export_results"]
