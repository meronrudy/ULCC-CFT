"""
Formal test: guided importance sampling achieves ≥50% variance reduction vs uniform baseline.
"""
import sys
import os
import numpy as np

# Ensure 'spec-first' is on sys.path so `import pggs...` resolves
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pggs.sampler import compare_estimator_variance
from pggs.guide import CausalAtlas


def test_guided_importance_sampling_variance_reduction():
    """
    Construct a toy model with rare large contributions and verify
    guided importance sampling reduces variance by at least 50% vs uniform.
    """
    paths = list(range(100))

    def f(i: int) -> float:
        # Rare, high-impact events in first 5 indices
        return 10.0 if i < 5 else 0.1

    # True mean for reference: (5*10 + 95*0.1)/100 = 0.595
    atlas = CausalAtlas()
    for i in paths:
        atlas.update(i, 0.0 if i < 5 else 3.0)

    rng = np.random.default_rng(0)
    n_samples = 200
    repeats = 300

    var_uniform, var_guided = compare_estimator_variance(
        paths, f, atlas, n_samples, repeats, rng
    )

    assert var_uniform > 0.0
    assert var_guided <= 0.5 * var_uniform