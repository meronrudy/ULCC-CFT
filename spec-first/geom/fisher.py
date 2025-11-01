"""Fisher information metric implementations.

Purpose
- Concrete implementations of analytic and empirical Fisher–Rao information metrics.

Invariants enforced via asserts
- Symmetry: g == g.T within atol
- Positive-definiteness: eigvals(g) > 0 within tolerance

Notes
- Functions are pure/deterministic. No global RNG access here.
- Domain validation is performed (e.g., θ ∈ (0,1) for Bernoulli, Σ SPD for Gaussian).
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Union


def _assert_spd(g: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    """
    Assert that a matrix is symmetric positive-definite (SPD).

    Symmetry: g == g.T within atol
    Positive-definiteness: all eigenvalues > 0 (within tolerance)

    Returns the array 'g' to allow inline usage.
    """
    g = np.asarray(g, dtype=float)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "Fisher metric must be a square 2D matrix."
    # Symmetry
    assert np.allclose(g, g.T, atol=atol), "Fisher metric must be symmetric within tolerance."
    # Positive-definite (use eigvalsh for symmetric matrices)
    w = np.linalg.eigvalsh(0.5 * (g + g.T))
    assert np.all(w > atol), f"Fisher metric must be positive-definite; eigenvalues={w}."
    return g


@dataclass(frozen=True)
class FisherMetric:
    """
    Container for a Fisher information metric.

    Fields
    - params: parameter vector at which the metric is evaluated
    - g: SPD metric tensor (matrix)

    Invariants (asserted on construction)
    - g is symmetric positive-definite (SPD) within tolerance.
    """
    params: np.ndarray
    g: np.ndarray

    def __post_init__(self) -> None:
        params = np.asarray(self.params, dtype=float)
        g = np.asarray(self.g, dtype=float)
        _assert_spd(g)
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "g", g)


def fisher_bernoulli(theta: float, atol: float = 1e-12) -> np.ndarray:
    """
    Analytic Fisher information for Bernoulli(θ) with θ ∈ (0,1).

    Returns
    - 1x1 SPD matrix with value 1 / (θ (1 − θ)).

    Invariants enforced
    - Symmetry and positive-definiteness via internal asserts.
    """
    theta = float(theta)
    if not (0.0 < theta < 1.0):
        raise ValueError("Bernoulli parameter θ must lie in the open interval (0,1).")
    g = np.array([[1.0 / (theta * (1.0 - theta))]], dtype=float)
    return _assert_spd(g, atol=atol)


def fisher_gaussian_mean(Sigma: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    """
    Analytic Fisher information for a multivariate Gaussian with mean μ as the only
    parameter and known covariance Σ (constant, SPD).

    Result
    - Fisher metric equals Σ^{-1}.

    Domain validation
    - Σ must be square, symmetric positive-definite (SPD).
    """
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Σ must be a square 2D array.")
    # Symmetrize and check SPD of Σ
    Sigma_sym = 0.5 * (Sigma + Sigma.T)
    w = np.linalg.eigvalsh(Sigma_sym)
    if not np.all(w > atol):
        raise ValueError("Σ must be symmetric positive-definite (SPD).")
    g = np.linalg.inv(Sigma_sym)
    return _assert_spd(g, atol=atol)


def fisher_gaussian_full(Sigma: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    """
    Full Fisher information for a multivariate Gaussian parameterized by
    the concatenation [μ, vec(Σ)], where vec(Σ) is the full (non-redundant)
    column-stacking of the covariance matrix entries.

    Structure
    - Mean block: F_{μμ} = Σ^{-1}.
    - Covariance block: F_{ΣΣ} = 0.5 · kron(Σ^{-1}, Σ^{-1}).
    - Cross blocks F_{μΣ} and F_{Σμ} are zero under this parameterization.

    Notes
    - This uses the standard Gaussian Fisher–Rao result for mean and covariance.
    - For a symmetric parameterization using only unique entries of Σ (vech),
      the covariance block would differ; we document and return the vec(Σ) form.
    """
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Σ must be a square 2D array.")
    Sigma_sym = 0.5 * (Sigma + Sigma.T)
    w = np.linalg.eigvalsh(Sigma_sym)
    if not np.all(w > atol):
        raise ValueError("Σ must be symmetric positive-definite (SPD).")
    d = Sigma_sym.shape[0]
    P = np.linalg.inv(Sigma_sym)  # Σ^{-1}
    cov_block = 0.5 * np.kron(P, P)  # block for vec(Σ)
    zeros_top_right = np.zeros((d, d * d), dtype=float)
    zeros_bottom_left = np.zeros((d * d, d), dtype=float)
    g = np.block([[P, zeros_top_right],
                  [zeros_bottom_left, cov_block]])
    return _assert_spd(g, atol=atol)


def _ensure_spd(A: np.ndarray, eps: float = 1e-8, max_tries: int = 5, atol: float = 1e-12) -> np.ndarray:
    """
    Symmetrize and ensure SPD by adding minimal diagonal jitter.

    Parameters
    - A: 2D array
    - eps: base jitter added to the diagonal; adaptively increased by x10 if needed
    - max_tries: number of jitter increases to attempt
    - atol: minimal eigenvalue threshold for positive definiteness

    Returns
    - Symmetric positive-definite matrix as float ndarray.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input must be a square 2D array.")
    G = 0.5 * (A + A.T)
    d = G.shape[0]
    jitter = float(eps)
    for _ in range(max_tries + 1):
        w = np.linalg.eigvalsh(G)
        if np.all(w > atol):
            return G
        G = G + jitter * np.eye(d, dtype=float)
        jitter *= 10.0
    # Final hard assertion
    return _assert_spd(G, atol=atol)


def empirical_fisher_from_scores(scores: np.ndarray, eps: float = 1e-8, max_tries: int = 5, atol: float = 1e-12) -> np.ndarray:
    """
    Empirical Fisher from per-sample score vectors (covariance-of-scores, mean-free).

    Parameters
    - scores: array-like of shape (N, d) or (N,), each row is the gradient of log-likelihood wrt parameters.
    - eps: base diagonal jitter added to enforce positive definiteness (default 1e-8).
    - max_tries: adaptive jitter escalation attempts if matrix is numerically singular.
    - atol: eigenvalue tolerance for SPD checks.

    Returns
    - ndarray float of shape (d, d) that is symmetric positive-definite.

    Behavior
    - Computes mean-free covariance: F = (1/N) * sum_i (s_i - mean_s)(s_i - mean_s)^T.
    - Enforces symmetry via (A + A.T)/2 and adds ε·I adaptively to ensure SPD.
    """
    S = np.asarray(scores, dtype=float)
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    if S.ndim != 2:
        raise ValueError("scores must be a 1D or 2D array with shape (N, d).")
    N, d = S.shape
    if N <= 0 or d <= 0:
        raise ValueError("scores must contain at least one sample (N>0) and dimension (d>0).")
    if not np.all(np.isfinite(S)):
        raise ValueError("scores must have all finite values.")
    mean = S.mean(axis=0, keepdims=True)
    Sc = S - mean
    G = (Sc.T @ Sc) / float(N)
    return _ensure_spd(G, eps=eps, max_tries=max_tries, atol=atol)


def empirical_fisher(scores: np.ndarray, eps: float = 1e-8, max_tries: int = 5, atol: float = 1e-12) -> np.ndarray:
    """
    Backward-compatible wrapper for empirical_fisher_from_scores.
    See empirical_fisher_from_scores(...) for details.
    """
    return empirical_fisher_from_scores(scores, eps=eps, max_tries=max_tries, atol=atol)


def empirical_fisher_from_data(
    data: Union[np.ndarray, Iterable],
    score_fn: Callable[[np.ndarray], np.ndarray],
    batch_size: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    eps: float = 1e-8,
    max_tries: int = 5,
    atol: float = 1e-12,
) -> np.ndarray:
    """
    Convenience API to compute empirical Fisher from a dataset and a score function.

    Parameters
    - data: array-like of observations (shape (N, ...) ) or an iterable yielding observations.
    - score_fn: callable taking a batch (np.ndarray of shape (B, ...)) and returning per-sample scores
      as an array of shape (B, d) or (B,).
      If the callable accepts an 'rng' keyword, it will be passed the provided rng.
    - batch_size: optional mini-batch size. If None, processes all data in one batch if array-like.
      Iterables are batched into lists of up to batch_size, then converted to arrays.
    - rng: numpy Generator for determinism policy (default np.random.default_rng(0)); forwarded to score_fn if supported.
    - eps, max_tries, atol: SPD enforcement parameters; see empirical_fisher_from_scores.

    Returns
    - ndarray float of shape (d, d) representing the empirical Fisher matrix.

    Implementation notes
    - Uses a numerically stable streaming/parallel covariance accumulation (Chan–Golub–LeVeque)
      so that the full score matrix need not be held in memory.
    - Computes mean-free covariance with 1/N normalization.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Helper to call score_fn with rng if supported
    def _call_with_rng(fn, batch):
        try:
            return fn(batch, rng=rng)
        except TypeError:
            return fn(batch)

    # Create batch generator
    def _batches_from_array(arr: np.ndarray):
        if batch_size is None or batch_size <= 0 or arr.shape[0] <= batch_size:
            yield arr
        else:
            for i in range(0, arr.shape[0], batch_size):
                yield arr[i : i + batch_size]

    def _batches_from_iterable(it: Iterable):
        if batch_size is None or batch_size <= 0:
            buf = list(it)
            if buf:
                yield np.asarray(buf)
        else:
            buf = []
            for item in it:
                buf.append(item)
                if len(buf) >= batch_size:
                    yield np.asarray(buf)
                    buf = []
            if buf:
                yield np.asarray(buf)

    # Select batching strategy
    if isinstance(data, np.ndarray):
        batch_iter = _batches_from_array(data)
    else:
        batch_iter = _batches_from_iterable(data)

    n = 0  # total samples
    d: Optional[int] = None
    mean = None  # running mean (d,)
    M2 = None    # sum of outer products about the mean (d,d)

    for batch in batch_iter:
        scores_batch = _call_with_rng(score_fn, np.asarray(batch))
        Sb = np.asarray(scores_batch, dtype=float)
        if Sb.ndim == 1:
            Sb = Sb.reshape(-1, 1)
        if Sb.ndim != 2:
            raise ValueError("score_fn must return an array of shape (B, d) or (B,).")
        if not np.all(np.isfinite(Sb)):
            raise ValueError("score_fn returned non-finite values.")
        B, d_batch = Sb.shape
        if B == 0:
            continue
        if d is None:
            d = d_batch
            mean = np.zeros((d,), dtype=float)
            M2 = np.zeros((d, d), dtype=float)
        elif d_batch != d:
            raise ValueError("score dimension changed across batches.")

        # Batch statistics
        mb = Sb.mean(axis=0)              # (d,)
        Xc = Sb - mb                      # (B,d)
        Sb_centered = Xc.T @ Xc           # (d,d)

        # Parallel/streaming covariance combination (Chan–Golub–LeVeque)
        n_new = n + B
        delta = mb - mean                 # (d,)
        mean = mean + (B / float(n_new)) * delta
        M2 = M2 + Sb_centered + (n * B / float(n_new)) * np.outer(delta, delta)
        n = n_new

    if n == 0:
        raise ValueError("No samples were provided to empirical_fisher_from_data.")

    G = M2 / float(n)
    return _ensure_spd(G, eps=eps, max_tries=max_tries, atol=atol)
