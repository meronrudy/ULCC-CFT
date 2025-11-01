"""Fixtures package for spec-first test data.

Re-exports convenience constructors for tests and examples.
Import-safe: no side effects.
"""

from .graphs import path_graph_adjacency, cycle_graph_incidence, basis_vectors
from .distributions import bernoulli_scores, quadratic_potential_matrix

__all__ = [
    "path_graph_adjacency",
    "cycle_graph_incidence",
    "basis_vectors",
    "bernoulli_scores",
    "quadratic_potential_matrix",
]
