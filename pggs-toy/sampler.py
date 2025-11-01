"""Importance sampler for toy PGGS."""
from typing import List, Sequence, Tuple
import numpy as np
from .hypergraph import Hyperedge
from .guide import guide_score

def sample_paths(edges: Sequence[Tuple[Hyperedge, float]], L: int, num: int, rng=None):
    """Sample `num` paths of length L with probs ∝ guide_score(weight).
    Returns (paths, probs).
    """
    rng = np.random.default_rng() if rng is None else rng
    weights = np.array([guide_score(w) for (_, w) in edges], dtype=float)
    probs = weights / weights.sum()
    idx = rng.choice(len(edges), size=(num, L), replace=True, p=probs)
    paths = [[edges[k][0] for k in row] for row in idx]
    return paths, probs
