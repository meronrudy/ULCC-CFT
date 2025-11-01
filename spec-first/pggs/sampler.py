"""PGGS sampler utilities.

Invariants
- Deterministic sampling when rng is provided (no hidden global state)
- Tests must seed with np.random.default_rng(0)
- Guided importance sampling variance reduction acceptance criterion (≥50% vs uniform).
"""

from typing import Any, Callable, Sequence, Tuple
import numpy as np


def softmax_neg_potential(keys: Sequence[Any], atlas: Any, temperature: float = 1.0) -> np.ndarray:
    """
    Compute proposal probabilities q_i ∝ exp(-(U_i)/temperature).

    Stabilized via subtracting the maximum prior to exponentiation.

    Parameters
    ----------
    keys : sequence
        Keys to evaluate.
    atlas : object
        Provides potential(key) -> float.
    temperature : float
        Positive temperature scaling.

    Returns
    -------
    np.ndarray
        Normalized probabilities (float64) summing to 1.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    N = len(keys)
    if N == 0:
        raise ValueError("keys must be non-empty")
    U = np.array([float(atlas.potential(k)) for k in keys], dtype=np.float64)
    scale = -U / float(temperature)
    shift = scale - np.max(scale)
    exps = np.exp(shift)
    q = exps / np.sum(exps)
    return q.astype(np.float64, copy=False)


def guided_estimator(
    paths: Sequence[Any],
    f: Callable[[Any], float],
    atlas: Any,
    n_samples: int,
    rng: np.random.Generator,
    temperature: float = 1.0,
) -> float:
    """
    Importance-sampled estimator of the uniform mean under proposal q.

    The target distribution p is uniform over paths (p=1/N). We sample indices
    from q(keys) and use weights w = p/q to compute the estimate.

    Returns
    -------
    float
        Guided IS estimate of E_p[f(X)].
    """
    N = len(paths)
    if N == 0:
        raise ValueError("paths must be non-empty")
    q = softmax_neg_potential(paths, atlas, temperature=temperature)
    idx = rng.choice(N, size=int(n_samples), replace=True, p=q)
    vals = np.array([float(f(paths[i])) for i in idx], dtype=np.float64)
    w = (1.0 / N) / q[idx]
    est = np.mean(w * vals)
    return float(est)


def uniform_estimator(
    paths: Sequence[Any],
    f: Callable[[Any], float],
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    """
    Uniform Monte Carlo estimator of the mean E_p[f(X)] with p=Uniform(paths).
    """
    N = len(paths)
    if N == 0:
        raise ValueError("paths must be non-empty")
    idx = rng.integers(0, N, size=int(n_samples), dtype=np.int64)
    vals = np.array([float(f(paths[i])) for i in idx], dtype=np.float64)
    return float(np.mean(vals))


def compare_estimator_variance(
    paths: Sequence[Any],
    f: Callable[[Any], float],
    atlas: Any,
    n_samples: int,
    repeats: int,
    rng: np.random.Generator,
    temperature: float = 1.0,
) -> Tuple[float, float]:
    """
    Compare replicate variances of uniform vs guided estimators.

    The acceptance criterion is that guided variance should be at most 50% of
    the uniform variance on the provided model.

    Returns
    -------
    (var_uniform, var_guided) : tuple of floats
        Population variances over the replicate estimates (ddof=0).
    """
    est_u = np.empty(int(repeats), dtype=np.float64)
    est_g = np.empty(int(repeats), dtype=np.float64)
    # Create independent sub-generators for repeatability.
    seeds = rng.integers(0, np.iinfo(np.int64).max, size=(int(repeats), 2), dtype=np.int64)
    for i in range(int(repeats)):
        rg_u = np.random.default_rng(int(seeds[i, 0]))
        rg_g = np.random.default_rng(int(seeds[i, 1]))
        est_u[i] = uniform_estimator(paths, f, n_samples, rg_u)
        est_g[i] = guided_estimator(paths, f, atlas, n_samples, rg_g, temperature=temperature)
    var_u = float(np.var(est_u, ddof=0))
    var_g = float(np.var(est_g, ddof=0))
    return var_u, var_g


__all__ = [
    "softmax_neg_potential",
    "guided_estimator",
    "uniform_estimator",
    "compare_estimator_variance",
]
