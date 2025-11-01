from ulcc_pggs.sampler import sample_paths
from ulcc_pggs.hypergraph import Hyperedge
import numpy as np

def test_probabilities_sum_to_one():
    edges = [(Hyperedge((0,),1), 0.1), (Hyperedge((1,),2), 0.9)]
    _, probs = sample_paths(edges, L=2, num=5, rng=np.random.default_rng(0))
    assert np.isclose(probs.sum(), 1.0)
    assert probs[1] > probs[0]  # higher weight ⇒ higher prob
