"""
PGGS slow-plane package (Phase A — E3a)

Exposes:
- PGGSConfig, AtlasU, SourcesJ, FluxB dataclasses
- run_pggs(frames, cfg) - deterministic attribution pipeline

See sim/API_SURFACE.md and sim/DATA_CONTRACTS.md for contracts.
"""
from .pipeline import (
    PGGSConfig as PGGSConfig,
    AtlasU as AtlasU,
    SourcesJ as SourcesJ,
    FluxB as FluxB,
    run_pggs as run_pggs,
)
