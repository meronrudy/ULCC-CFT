"""Causal atlas guide (toy)."""
def guide_score(edge_weight: float) -> float:
    """Monotone map used as importance weight; Phase-0 uses identity."""
    return float(max(edge_weight, 1e-8))
