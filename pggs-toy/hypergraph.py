"""Hypergraph primitives."""
from dataclasses import dataclass
from typing import Tuple, List

@dataclass(frozen=True)
class Hyperedge:
    sources: Tuple[int, ...]
    target: int

def hyperpaths_of_length_one(nodes: int) -> List[Hyperedge]:
    return [Hyperedge((i,), j) for i in range(nodes) for j in range(nodes) if i != j]
