"""PGGS hypergraph module.

Invariants
- edges unique; causal order preserved.
"""

from typing import Any, Iterable, List, Set, Tuple, Generator, Optional
import itertools


class Hypergraph:
    """
    Minimal typed hypergraph structure with unique edges.
    Invariant: edges unique; causal order preserved.
    """

    def __init__(self) -> None:
        self.nodes: Set[Any] = set()
        self.hyperedges: List[Tuple[Any, ...]] = []

    def add_edge(self, nodes: Iterable[Any]) -> Tuple[Any, ...]:
        """
        Add a hyperedge defined by an iterable of nodes.

        The edge is stored as a sorted unique tuple for deterministic ordering.

        Raises
        ------
        AssertionError
            If the edge already exists.
        ValueError
            If the provided iterable is empty.
        """
        uniq = set(nodes)
        if len(uniq) == 0:
            raise ValueError("Hyperedge must contain at least one node")
        edge = tuple(sorted(uniq))
        if edge in self.hyperedges:
            raise AssertionError("Edge already present")
        self.hyperedges.append(edge)
        self.nodes.update(edge)
        return edge

    def edges(self) -> List[Tuple[Any, ...]]:
        """Return a list of current hyperedges."""
        return list(self.hyperedges)

    def paths(
        self, max_length: Optional[int] = None
    ) -> Generator[Tuple[Tuple[Any, ...], ...], None, None]:
        """
        Generate simple permutations of edges up to max_length.

        Deterministic ordering using sorted tuples to ease testing.
        """
        edges_sorted = sorted(self.hyperedges)
        n = len(edges_sorted)
        if n == 0:
            return
        Lmax = n if (max_length is None) else max(0, min(max_length, n))
        for L in range(1, Lmax + 1):
            for perm in itertools.permutations(edges_sorted, L):
                yield perm


__all__ = ["Hypergraph"]
