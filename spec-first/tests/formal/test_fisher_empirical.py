import os
import sys
import numpy as np

# Ensure 'spec-first/' is importable so 'from geom.fisher import ...' works.
THIS_DIR = os.path.dirname(__file__)
SPEC_FIRST_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if SPEC_FIRST_ROOT not in sys.path:
    sys.path.insert(0, SPEC_FIRST_ROOT)

# Import APIs under test
from geom.fisher import (
    fisher_bernoulli,
    empirical_fisher_from_scores,
    empirical_fisher_from_data,
)


def _rel_err(a: float, b: float) -> float:
    return float(abs(a - b) / abs(b))


def test_bernoulli_empirical_matches_analytic():
    # Deterministic RNG per project policy
    rng = np.random.default_rng(0)
    theta = 0.3
    N = 20000  # chosen to meet ≤5% relative error reliably and run fast

    # Generate Bernoulli samples deterministically
    x = (rng.random(N) < theta).astype(float)

    # Per-sample exact score wrt theta: (x - theta) / (theta (1 - theta))
    s = (x - theta) / (theta * (1.0 - theta))  # shape (N,)

    G = empirical_fisher_from_scores(s)  # should be 1x1 SPD
    assert G.shape == (1, 1)
    g_hat = float(G[0, 0])

    g_true = float(1.0 / (theta * (1.0 - theta)))
    assert _rel_err(g_hat, g_true) <= 0.05

    # Symmetry and SPD
    assert np.allclose(G, G.T, atol=1e-12)
    w = np.linalg.eigvalsh(0.5 * (G + G.T))
    assert np.all(w > 0.0)
    assert float(w.min()) >= 1e-12


def test_gaussian_mean_only_unit_variance():
    rng = np.random.default_rng(0)
    mu = 0.5
    sigma = 1.0
    N = 10000  # runtime-friendly and accurate

    x = rng.normal(loc=mu, scale=sigma, size=N).astype(float)

    # Score wrt mu is (x - mu) / sigma^2; here sigma^2=1
    s = (x - mu)

    G = empirical_fisher_from_scores(s)
    assert G.shape == (1, 1)
    g_hat = float(G[0, 0])

    g_true = 1.0 / (sigma ** 2)
    assert _rel_err(g_hat, g_true) <= 0.03

    assert np.allclose(G, G.T, atol=1e-12)
    w = np.linalg.eigvalsh(G)
    assert np.all(w > 0.0)
    assert float(w.min()) >= 1e-12


def test_centering_mean_free_covariance_used():
    rng = np.random.default_rng(0)
    N = 5000
    # Non-zero mean scores (Gaussian scores shifted by constant)
    base = rng.normal(size=N)
    c = 3.0
    s_shift = base + c  # mean != 0

    # Empirical Fisher should be mean-free covariance
    G = empirical_fisher_from_scores(s_shift)
    assert G.shape == (1, 1)
    G_val = float(G[0, 0])

    # Non-centered "covariance"
    G_uncentered = float((s_shift.T @ s_shift) / float(N))

    # Centered covariance explicitly
    m = float(s_shift.mean())
    Sc = s_shift - m
    G_centered = float((Sc.T @ Sc) / float(N))

    # The implemented result equals centered covariance, and differs from uncentered
    assert np.allclose(G_val, G_centered, rtol=0.0, atol=1e-12)
    assert not np.allclose(G_val, G_uncentered, rtol=0.0, atol=1e-6)


def test_spd_enforced_with_collinear_features_and_jitter():
    rng = np.random.default_rng(0)
    N = 4096
    s = rng.normal(size=N)
    # Two perfectly collinear dimensions - singular covariance without jitter
    S = np.stack([s, 2.0 * s], axis=1)  # shape (N,2)

    G = empirical_fisher_from_scores(S)
    assert G.shape == (2, 2)
    assert np.allclose(G, G.T, atol=1e-12)

    w = np.linalg.eigvalsh(G)
    # After jitter, all eigenvalues are strictly positive and above numerical floor
    assert np.all(w > 0.0)
    assert float(w.min()) >= 1e-12


def test_dataset_api_mini_batch_and_determinism_array():
    rng = np.random.default_rng(0)
    mu = -0.7
    sigma = 1.0
    N = 8000
    data = rng.normal(loc=mu, scale=sigma, size=N).astype(float)

    # Score function that accepts rng (passed but not used) and returns per-sample scores
    def score_mu(batch: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
        return (batch - mu) / (sigma ** 2)

    # Use mini-batching to exercise streaming accumulation
    G1 = empirical_fisher_from_data(data, score_mu, batch_size=257, rng=np.random.default_rng(0))
    G2 = empirical_fisher_from_data(data, score_mu, batch_size=257, rng=np.random.default_rng(0))

    # Determinism: identical results with same data and batching
    assert G1.shape == (1, 1) and G2.shape == (1, 1)
    assert np.array_equal(G1, G2)

    # Matches analytic target within tolerance
    g_hat = float(G1[0, 0])
    g_true = 1.0 / (sigma ** 2)
    assert _rel_err(g_hat, g_true) <= 0.03


def test_dataset_api_iterator_and_determinism_iterator():
    rng = np.random.default_rng(0)
    theta = 0.3
    N = 16000
    # Build a deterministic list so re-iteration preserves order
    xs = ((rng.random(N) < theta).astype(float)).tolist()

    def score_theta(batch: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
        return (batch - theta) / (theta * (1.0 - theta))

    # Provide as iterator (list is iterable) with mini-batching
    G1 = empirical_fisher_from_data(xs, score_theta, batch_size=511, rng=np.random.default_rng(0))
    G2 = empirical_fisher_from_data(xs, score_theta, batch_size=511, rng=np.random.default_rng(0))

    assert G1.shape == (1, 1) and G2.shape == (1, 1)
    # Deterministic across runs
    assert np.array_equal(G1, G2)

    g_hat = float(G1[0, 0])
    g_true = float(1.0 / (theta * (1.0 - theta)))
    assert _rel_err(g_hat, g_true) <= 0.05