"""Holonomy estimator stub."""
from typing import List, Callable
import numpy as np
from .metric_graph import Edge
from .transport import parallel_transport_identity

def holonomy_loop(loop: List[Edge]) -> float:
    """Phase-0: identity transport ⇒ zero holonomy."""
    t = parallel_transport_identity(loop)
    return 0.0 if t == 1.0 else float('nan')

def holonomy_loop_with_transport(loop: List[Edge], transport_fn: Callable[[int, int, np.ndarray], np.ndarray]) -> float:
    """
    Compose transports around the loop starting from v0=[1,0] and return
    the absolute rotation angle between v_final and v0, wrapped to (-pi, pi].
    """
    v0 = np.array([1.0, 0.0], dtype=float)
    v = v0
    for e in loop:
        v = transport_fn(e.u, e.v, v)
    angle = np.arctan2(v[1], v[0]) - np.arctan2(v0[1], v0[0])
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return float(abs(angle))
