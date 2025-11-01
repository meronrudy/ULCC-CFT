"""Phase A — E3d: Packer public API.

Re-exports:
- PackerConfig: configuration dataclass for deterministic packing behavior.
- make_reconfig_pack: compile (g, Phi, grad, U, J, B, geom_meta) → ReconfigPack dict.
"""

from .make import PackerConfig, make_reconfig_pack

__all__ = ["PackerConfig", "make_reconfig_pack"]