"""PGGS guide module.

Invariants
- Guided variance reduction vs uniform baseline
- Invariant: expected variance ↓ with refinement.
"""

from typing import Any, Dict


class CausalAtlas:
    """
    Dictionary-backed guidance potential U(key) used to skew proposals.

    Invariant: expected variance ↓ with refinement.
    """

    def __init__(self) -> None:
        self.U: Dict[Any, float] = {}

    def update(self, key: Any, potential: float) -> None:
        """Update/store the potential value for a key."""
        self.U[key] = float(potential)

    def potential(self, key: Any) -> float:
        """Return the potential for a key, defaulting to 0.0."""
        return float(self.U.get(key, 0.0))


__all__ = ["CausalAtlas"]
