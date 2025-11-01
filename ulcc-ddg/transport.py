"""Placeholder for discrete parallel transport."""
from typing import List, Callable, Dict
import numpy as np
from .metric_graph import Edge, MetricGraph

def parallel_transport_identity(path: List[Edge]) -> float:
    """Phase-0: trivial transport returning scalar 1.0 for any path."""
    return 1.0

def parallel_transport_rotation(angle_map: Dict[Edge, float]) -> Callable[[int, int, np.ndarray], np.ndarray]:
    """
    Return a transport function T(u, v, vec) that rotates a 2D vector by the angle assigned
    to directed edge (u, v) in angle_map. If an edge is missing, 0.0 is used.
    """
    def T(u: int, v: int, vec: np.ndarray) -> np.ndarray:
        phi = float(angle_map.get(Edge(u, v), 0.0))
        c = np.cos(phi)
        s = np.sin(phi)
        x = float(vec[0])
        y = float(vec[1])
        return np.array([c * x - s * y, s * x + c * y], dtype=float)
    return T
