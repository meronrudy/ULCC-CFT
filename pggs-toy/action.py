"""ULCC-inspired action along a path (toy)."""
from typing import List
from .hypergraph import Hyperedge

def action_along_path(path: List[Hyperedge], guide: float=1.0) -> float:
    """Toy action: path length divided by guide."""
    return len(path) / float(guide)
