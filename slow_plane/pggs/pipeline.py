from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

from telemetry.frame import TelemetryFrame
from .utils import (
    combine_seeds,
    grid_shape_from_frames,
    tile_index_maps,
    make_activity_arrays,
    deterministic_subsets,
    ema_update,
    clip_l2,
    divergence_2d,
    normalize_zero_sum,
    neighbor_diffs,
    safe_scale,
    ensure_float64,
)

@dataclass(frozen=True)
class PGGSConfig:
    rng_seed: int = 0
    batch_size: int = 8         # ≥1
    n_batches: int = 16         # ≥1
    smoothing_alpha: float = 0.3  # in [0,1]
    grad_norm_clip: float = 0.0   # ≥0, 0 = no clip
    min_perturb_budget: int = 0   # placeholder for future guard
    max_perturb_budget: int = 0   # placeholder for future guard
    tile_scope: str = "all"       # "all" | "subset"

    def __post_init__(self) -> None:
        if int(self.batch_size) < 1:
            raise ValueError("batch_size must be ≥ 1")
        if int(self.n_batches) < 1:
            raise ValueError("n_batches must be ≥ 1")
        if not (0.0 <= float(self.smoothing_alpha) <= 1.0):
            raise ValueError("smoothing_alpha must be in [0,1]")
        if float(self.grad_norm_clip) < 0.0:
            raise ValueError("grad_norm_clip must be ≥ 0")
        if self.tile_scope not in ("all", "subset"):
            raise ValueError("tile_scope must be 'all' or 'subset'")

@dataclass(frozen=True)
class AtlasU:
    U: np.ndarray  # shape (H, W), float64

@dataclass(frozen=True)
class SourcesJ:
    J: np.ndarray  # shape (H, W), float64

@dataclass(frozen=True)
class FluxB:
    B: Dict[str, np.ndarray]  # keys: N,S,E,W; each shape (H, W), float64

def _kpi_baseline(flit_tx: np.ndarray) -> float:
    # produced_flits_total proxy baseline across grid
    return float(flit_tx.sum())

def _attribution_shapley_proxy(w: int, h: int, subsets: List[np.ndarray], flit_tx: np.ndarray, q_p99: np.ndarray, power: np.ndarray, alpha: float, clip: float) -> np.ndarray:
    """
    Deterministic "virtual" perturbation:
    - For each subset S, compute delta proxy vs baseline within the window:
      delta = mean( flit_tx[S] + beta*q_p99[S] + gamma*power[S] ) - global_mean
    - Accumulate marginal per-tile: delta/|S| for i in S
    - EMA over batches to stabilize
    """
    H = h
    W = w
    N = W * H
    U = np.zeros((H, W), dtype=np.float64)
    # Precompute combined per-tile score proxy
    # Weights chosen conservatively; keep scale reasonable
    beta = 0.1
    gamma = 0.2
    score = ensure_float64(flit_tx) + beta * ensure_float64(q_p99) + gamma * ensure_float64(power)
    global_mean = float(score.mean())

    for b, S in enumerate(subsets):
        contrib = np.zeros((H, W), dtype=np.float64)
        vals = score.ravel()[S]
        if vals.size == 0:
            delta = 0.0
        else:
            delta = float(vals.mean() - global_mean)
        if S.size > 0:
            per = delta / float(S.size)
            # assign per-tile marginal
            contrib.ravel()[S] = per
        # gradient norm clip per-batch before EMA
        contrib = clip_l2(contrib, clip)
        U = ema_update(U, contrib, alpha)

    return U

def _estimate_J(U: np.ndarray, frames: List[TelemetryFrame]) -> np.ndarray:
    """Estimate sources as normalized divergence of U, nudged by MC served requests if present."""
    J_div = divergence_2d(U)
    # MC served requests proxy: use total activations to scale magnitude if non-zero
    total_act = 0
    for fr in frames:
        for mm in fr.memory_metrics:
            total_act += int(mm.activations)
    if total_act > 0:
        # scale divergence gently
        J = J_div / max(total_act, 1)
    else:
        J = J_div
    # Normalize to near-zero sum
    J = normalize_zero_sum(J)
    return J

def _estimate_B(U: np.ndarray, flit_tx: np.ndarray) -> Dict[str, np.ndarray]:
    """Flux as oriented differences of U scaled by activity proxy."""
    diffs = neighbor_diffs(U)
    # Build a scale from flit_tx normalized to [0,1] by its max
    mx = float(np.max(flit_tx)) if flit_tx.size > 0 else 0.0
    if mx > 0.0:
        scale = (flit_tx / mx)
    else:
        scale = np.ones_like(flit_tx, dtype=np.float64)
    B = {k: safe_scale(v, scale) for (k, v) in diffs.items()}
    return B

def run_pggs(frames: List[TelemetryFrame], cfg: PGGSConfig) -> Dict[str, Any]:
    """
    Deterministic PGGS pipeline:
    - Validates inputs and derives grid_shape
    - Aggregates per-tile proxies over frames
    - Forms deterministic virtual perturbation batches and computes attribution via a Shapley-like proxy
    - Applies EMA smoothing and gradient norm clipping
    - Estimates sources J and flux B
    - Returns artifacts and meta
    Error handling per sim/API_SURFACE.md: raises ValueError for invalid inputs.
    """
    if not isinstance(frames, list) or len(frames) == 0:
        raise ValueError("run_pggs: frames must be a non-empty list")
    # Basic grid validation
    w, h = grid_shape_from_frames(frames)

    # Seed folding: combine cfg.rng_seed with frame rng_seeds deterministically
    seed = combine_seeds(int(cfg.rng_seed), (int(fr.rng_seed) for fr in frames))

    # Aggregate activity proxies
    flit_tx, q_p99, power = make_activity_arrays(frames, w, h)

    # Deterministic batch subsets
    subsets = deterministic_subsets(w, h, int(cfg.batch_size), int(cfg.n_batches), seed)

    # Attribution
    U = _attribution_shapley_proxy(w, h, subsets, flit_tx, q_p99, power, float(cfg.smoothing_alpha), float(cfg.grad_norm_clip))

    # Sources
    J = _estimate_J(U, frames)

    # Flux
    B = _estimate_B(U, flit_tx)

    meta = {
        "batch_size": int(cfg.batch_size),
        "n_batches": int(cfg.n_batches),
        "rng_seed": int(seed),
        "frames_used": int(len(frames)),
        "grid_shape": {"width": w, "height": h},
        "kpi": "produced_flits_total",
    }

    return {
        "U": AtlasU(U=U),
        "J": SourcesJ(J=J),
        "B": FluxB(B=B),
        "meta": meta,
    }
