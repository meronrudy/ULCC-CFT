"""Phase A — E3c geometry update public API.

Exports:
- GeometryConfig
- update_geometry

Helper internals (also exported for tests and diagnostics):
- ensure_spd_and_cond
- cfe_residual
- apply_step
- accept_with_trust
"""

from .update import (
    GeometryConfig,
    update_geometry,
    ensure_spd_and_cond,
    cfe_residual,
    apply_step,
    accept_with_trust,
)

__all__ = [
    "GeometryConfig",
    "update_geometry",
    "ensure_spd_and_cond",
    "cfe_residual",
    "apply_step",
    "accept_with_trust",
]