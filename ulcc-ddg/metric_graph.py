"""Minimal metric-graph scaffold for DDG tests."""
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass(frozen=True)
class Edge:
    u: int
    v: int

@dataclass
class MetricGraph:
    edges: Dict[Edge, float]  # edge length

    def length(self, e: Edge) -> float:
        return float(self.edges[e])
