"""Phase A — E3d: Packer

This module compiles slow-plane outputs (g, Φ, ∇Φ, U, J, B) and geometry meta into a
deterministic ReconfigPack dictionary suitable for downstream consumption by control/fast planes
in later phases. It does not perform any I/O or invoke other subsystems.

Phase A mapping rationale:
- NoC routing weights: favor directions aligned with negative ∇Φ (toward lower potential) and
  positive flux B. We convert both cues into directional preferences (N,E,S,W), add them, then
  apply a numerically stable softmax with temperature routing_temp. We then clip weights to at least
  routing_clip_min and renormalize to sum exactly 1 per tile.
- MC priorities ("policy words"): we map J to priorities by a grid-wise softmax with temperature
  mc_softmax_temp to emphasize relatively higher J tiles, then affine-scale the probabilities into
  [mc_min_priority, mc_max_priority].
- CAT masks: Phase A heuristic. If cat_policy="uniform", enable exactly floor(cat_num_ways/2) ways
  via the lowest-order bits. If "u_weighted", enable k(y,x) ≈ floor(p(U[y,x]) * cat_num_ways),
  where p(U) is the rank-percentile of U over the grid. This yields more enabled ways where U is high.
- CPU affinities: either provided tile→cpu map (validated) or row-major modulo n_cpus.
- NUMA policies: fixed "local_first" for all tiles in Phase A.
- DVFS states: quantize either ||∇Φ|| or normalized U into the discrete domain cfg.dvfs_levels.
  Lower metric → lower DVFS level; higher metric → higher DVFS level.
- Trust-region meta: pass-through from geometry update meta without transformation.
- CRC-32C: compute over a canonical JSON of the whole pack excluding the 'crc32c' field itself,
  using the Castagnoli polynomial. The canonical encoder uses sorted keys and compact separators,
  with floats rounded to 10 decimal places to ensure stability.

Determinism:
- Pure numpy and Python stdlib. No randomness. Stable softmax and clipping. Canonical JSON encoding.
- Strict input validation (shapes, finiteness). Integer/float dtypes are enforced from config.

Public API:
- PackerConfig: dataclass of knobs with validated defaults.
- make_reconfig_pack(...): builds and returns the ReconfigPack as a Python dict.

Output schema (Phase A):
{
  "version": 1,
  "noc_tables": {
    "weights": float32 [H,W,4] (N,E,S,W), non-negative, per-tile sums == 1 ± 1e-6,
    "vc_credits": int32 scalar or [H,W,2] (ESC,MAIN) — Phase A provides scalar 8
  },
  "link_weights": float32 [H,W,4] (duplicate of noc_tables.weights),
  "mc_policy_words": float32 [H,W] in [mc_min_priority, mc_max_priority],
  "cat_masks": int32 [H,W] bitmasks; uniform enables floor(cat_num_ways/2) ways; u_weighted varies with U,
  "cpu_affinities": int32 [H,W] cpu ids in [0, n_cpus-1],
  "numa_policies": str [H,W] entries in {"local_first"},
  "dvfs_states": int32 [H,W] values in cfg.dvfs_levels,
  "trust_region_meta": {
    "accepted": bool,
    "accept_ratio": float,
    "residual_norm": float,
    "trust_radius": float,
    "hysteresis_left": int
  },
  "crc32c": uint32 (Castagnoli)
}

Constraints:
- Only numpy and Python stdlib.
- All float tables use cfg.weight_dtype (default float32).
- All integer tables use cfg.policy_int_dtype (default int32).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import math
import numpy as np


# --------------------
# Configuration
# --------------------

@dataclass(frozen=True)
class PackerConfig:
    # Routing / NoC
    routing_temp: float = 0.50                  # > 0
    routing_clip_min: float = 1e-6              # >= 0
    # Memory controller priority words
    mc_softmax_temp: float = 0.75               # > 0
    mc_min_priority: float = 0.0                # >= 0
    mc_max_priority: float = 1.0                # >= mc_min_priority
    # CAT
    cat_num_ways: int = 8                       # >= 1
    cat_policy: str = "uniform"                 # {"uniform","u_weighted"}
    # Scheduler / CPU topology
    n_cpus: int = 64                            # >= 1
    cpu_grid_map: Optional[np.ndarray] = None   # int32 [H,W] or None
    # DVFS
    dvfs_levels: List[int] = None               # non-empty list of ints
    dvfs_from: str = "grad_phi"                 # {"grad_phi","U"}
    # Dtypes
    weight_dtype: Any = np.float32
    policy_int_dtype: Any = np.int32
    # Integrity
    crc_enable: bool = True

    def __post_init__(self) -> None:
        if not (float(self.routing_temp) > 0.0):
            raise ValueError("routing_temp must be > 0")
        if float(self.routing_clip_min) < 0.0:
            raise ValueError("routing_clip_min must be >= 0")
        if not (float(self.mc_softmax_temp) > 0.0):
            raise ValueError("mc_softmax_temp must be > 0")
        if float(self.mc_min_priority) < 0.0:
            raise ValueError("mc_min_priority must be >= 0")
        if not (float(self.mc_max_priority) >= float(self.mc_min_priority)):
            raise ValueError("mc_max_priority must be >= mc_min_priority")
        if int(self.cat_num_ways) < 1:
            raise ValueError("cat_num_ways must be >= 1")
        if self.cat_policy not in ("uniform", "u_weighted"):
            raise ValueError("cat_policy must be 'uniform' or 'u_weighted'")
        if int(self.n_cpus) < 1:
            raise ValueError("n_cpus must be >= 1")
        if self.cpu_grid_map is not None:
            A = np.asarray(self.cpu_grid_map)
            if A.ndim != 2:
                raise ValueError("cpu_grid_map must have shape [H,W]")
        if self.dvfs_levels is None or len(self.dvfs_levels) == 0:
            raise ValueError("dvfs_levels must be a non-empty list of ints")
        if self.dvfs_from not in ("grad_phi", "U"):
            raise ValueError("dvfs_from must be 'grad_phi' or 'U'")
        # dtype sanity (not exhaustive)
        _ = np.dtype(self.weight_dtype)
        _ = np.dtype(self.policy_int_dtype)


# --------------------
# Canonical JSON and CRC-32C (Castagnoli)
# --------------------

def _round_float(x: float, places: int = 10) -> float:
    # Stable rounding to limit platform-dependent repr differences
    return float(round(float(x), places))


def _to_canonical_obj(obj: Any) -> Any:
    """
    Convert numpy arrays and scalars to Python types; round floats for stability.
    Recurses dicts/lists/tuples. Leaves strings/ints/bools as-is.
    """
    # numpy arrays
    if isinstance(obj, np.ndarray):
        # Convert to native Python container first, then recurse to round floats.
        if np.issubdtype(obj.dtype, np.integer):
            return obj.astype(int).tolist()
        # for floats, objects, strings: tolist then recurse for rounding/normalization
        return _to_canonical_obj(obj.tolist())

    # dicts: process values
    if isinstance(obj, dict):
        return {str(k): _to_canonical_obj(v) for k, v in obj.items()}

    # lists/tuples: recurse elements
    if isinstance(obj, (list, tuple)):
        return [_to_canonical_obj(v) for v in obj]

    # numpy scalars
    if isinstance(obj, (np.floating,)):
        return _round_float(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # native floats
    if isinstance(obj, float):
        return _round_float(obj)

    # passthrough for int, bool, str, None
    return obj


def _canonical_json_bytes(obj: Any) -> bytes:
    """
    Deterministic canonical JSON:
    - Convert numpy/scalars to JSON-safe types with float rounding
    - sort_keys=True
    - separators=(',', ':')
    - ensure_ascii=False
    """
    canon = _to_canonical_obj(obj)
    return json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _make_crc32c_table() -> Tuple[int, ...]:
    poly = 0x82F63B78  # reversed 0x1EDC6F41
    table: List[int] = []
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
        table.append(crc & 0xFFFFFFFF)
    return tuple(table)


_CRC32C_TABLE = _make_crc32c_table()


def _crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for b in data:
        idx = (crc ^ b) & 0xFF
        crc = (_CRC32C_TABLE[idx] ^ (crc >> 8)) & 0xFFFFFFFF
    return crc ^ 0xFFFFFFFF


# --------------------
# Internals: directional mappings and tables
# --------------------

def _validate_inputs(
    g: np.ndarray,
    phi: np.ndarray,
    grad: np.ndarray,
    U: np.ndarray,
    J: np.ndarray,
    B: Union[Dict[str, np.ndarray], np.ndarray],
) -> Tuple[int, int]:
    G = np.asarray(g, dtype=np.float64)
    PH = np.asarray(phi, dtype=np.float64)
    GR = np.asarray(grad, dtype=np.float64)
    UU = np.asarray(U, dtype=np.float64)
    JJ = np.asarray(J, dtype=np.float64)
    if G.ndim != 4 or G.shape[-2:] != (2, 2):
        raise ValueError("g must have shape [H,W,2,2]")
    H, W = G.shape[:2]
    if PH.shape != (H, W):
        raise ValueError("phi must have shape [H,W]")
    if GR.shape != (H, W, 2):
        raise ValueError("grad must have shape [H,W,2]")
    if UU.shape != (H, W):
        raise ValueError("U must have shape [H,W]")
    if JJ.shape != (H, W):
        raise ValueError("J must have shape [H,W]")
    # B format
    if isinstance(B, dict):
        for k in ("N", "E", "S", "W"):
            if k not in B:
                raise ValueError("B dict must contain keys 'N','E','S','W'")
            arr = np.asarray(B[k], dtype=np.float64)
            if arr.shape != (H, W):
                raise ValueError(f"B['{k}'] must have shape [H,W]")
    else:
        BB = np.asarray(B, dtype=np.float64)
        if BB.ndim != 3 or BB.shape != (H, W, 4):
            raise ValueError("B tensor must have shape [H,W,4] with channels [N,E,S,W]")
    # finiteness
    if not (np.all(np.isfinite(G)) and np.all(np.isfinite(PH)) and np.all(np.isfinite(GR))
            and np.all(np.isfinite(UU)) and np.all(np.isfinite(JJ))):
        raise ValueError("inputs must be finite")
    return H, W


def _flux_to_dirs(B: Union[Dict[str, np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Normalize B to non-negative directional hints [H,W,4] in order [N,E,S,W], clipping negatives to 0.
    If per-tile sum is 0, leave zeros (the softmax downstream will regularize with grad component).
    """
    if isinstance(B, dict):
        N = np.clip(np.asarray(B["N"], dtype=np.float64), 0.0, None)
        E = np.clip(np.asarray(B["E"], dtype=np.float64), 0.0, None)
        S = np.clip(np.asarray(B["S"], dtype=np.float64), 0.0, None)
        W = np.clip(np.asarray(B["W"], dtype=np.float64), 0.0, None)
        out = np.stack([N, E, S, W], axis=-1)
    else:
        BB = np.asarray(B, dtype=np.float64).copy()
        BB = np.maximum(BB, 0.0)
        # Assume channels order [N,E,S,W]
        out = BB
    return out


def _grad_to_dirs(grad: np.ndarray) -> np.ndarray:
    """
    Map ∇Φ to directional preferences [H,W,4] in [N,E,S,W] using projected positive components
    of the negative gradient -∇Φ onto the cardinal directions:
    N: max(gy, 0); S: max(-gy, 0); E: max(-gx, 0); W: max(gx, 0)
    """
    G = np.asarray(grad, dtype=np.float64)
    gx = G[..., 0]
    gy = G[..., 1]
    N = np.maximum(gy, 0.0)
    S = np.maximum(-gy, 0.0)
    E = np.maximum(-gx, 0.0)
    W = np.maximum(gx, 0.0)
    return np.stack([N, E, S, W], axis=-1)


def _normalize_softmax(x: np.ndarray, temp: float, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax over axis with temperature scaling.
    """
    X = np.asarray(x, dtype=np.float64)
    T = float(temp)
    T = T if T > 1e-12 else 1e-12
    Xs = X / T
    m = np.max(Xs, axis=axis, keepdims=True)
    z = np.exp(Xs - m)
    s = np.sum(z, axis=axis, keepdims=True)
    s = np.maximum(s, 1e-20)
    out = z / s
    return out


def _compose_noc_weights(
    grad_dirs: np.ndarray,
    flux_dirs: np.ndarray,
    cfg: PackerConfig,
) -> np.ndarray:
    """
    Combine grad- and flux-based preferences additively, softmax with temperature, then clip and renormalize.
    Ensures per-tile sum == 1 and values within [routing_clip_min, 1].
    """
    prefs = np.asarray(grad_dirs, dtype=np.float64) + np.asarray(flux_dirs, dtype=np.float64)
    soft = _normalize_softmax(prefs, float(cfg.routing_temp), axis=-1)
    # enforce per-direction minimum via baseline + residual distribution
    clip_min = float(cfg.routing_clip_min)
    if clip_min > 0.0:
        K = soft.shape[-1]
        baseline = clip_min * K
        if baseline >= 1.0 - 1e-12:
            # If the minimums saturate the simplex, fall back to uniform
            soft = np.full_like(soft, 1.0 / K, dtype=np.float64)
        else:
            resid = 1.0 - baseline
            denom = np.sum(soft, axis=-1, keepdims=True)
            denom = np.maximum(denom, 1e-20)
            pnorm = soft / denom
            soft = clip_min + resid * pnorm
    # final exact renormalization safeguard
    s = np.sum(soft, axis=-1, keepdims=True)
    s = np.maximum(s, 1e-12)
    soft = soft / s
    return soft.astype(cfg.weight_dtype, copy=False)


def _mc_priorities(J: np.ndarray, cfg: PackerConfig) -> np.ndarray:
    """
    Map J [H,W] to [mc_min_priority, mc_max_priority] via a grid-wise softmax with temperature.
    """
    S = np.asarray(J, dtype=np.float64)
    flat = S.reshape(-1, 1)
    probs = _normalize_softmax(flat, float(cfg.mc_softmax_temp), axis=0).reshape(S.shape)
    lo = float(cfg.mc_min_priority)
    hi = float(cfg.mc_max_priority)
    out = lo + (hi - lo) * probs
    return out.astype(cfg.weight_dtype, copy=False)


def _cat_masks(U: np.ndarray, cfg: PackerConfig) -> np.ndarray:
    """
    Produce per-tile bitmasks with cat_num_ways bits.
    - uniform: enable exactly floor(W/2) lowest-order ways.
    - u_weighted: enable floor(p * W) where p is percentile rank of U in [0,1].
    """
    H, W = np.asarray(U, dtype=np.float64).shape
    WAYS = int(cfg.cat_num_ways)
    if cfg.cat_policy == "uniform":
        k = int(math.floor(WAYS / 2))
        mask_val = (1 << k) - 1 if k > 0 else 0
        out = np.full((H, W), mask_val, dtype=cfg.policy_int_dtype)
        return out
    # u_weighted
    X = np.asarray(U, dtype=np.float64)
    # percentile rank p in [0,1] using argsort ranks
    flat = X.ravel()
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(order.size)
    p = (ranks.astype(np.float64) / max(flat.size - 1, 1)).reshape(X.shape)
    # Base selection proportional to percentile
    k_base = np.floor(p * WAYS).astype(int)
    k_base = np.clip(k_base, 0, WAYS)
    # Enforce at least half the ways enabled to meet Phase A heuristic
    k_min = WAYS // 2
    k_arr = np.maximum(k_base, k_min)
    # Build mask with lowest k bits set
    masks = np.zeros((H, W), dtype=np.int64)
    for y in range(H):
        for x in range(W):
            k = int(k_arr[y, x])
            masks[y, x] = (1 << k) - 1 if k > 0 else 0
    return masks.astype(cfg.policy_int_dtype, copy=False)


def _cpu_affinity_map(H: int, W: int, cfg: PackerConfig) -> np.ndarray:
    n = int(cfg.n_cpus)
    if cfg.cpu_grid_map is not None:
        M = np.asarray(cfg.cpu_grid_map)
        if M.shape != (H, W):
            raise ValueError("cpu_grid_map shape must match [H,W]")
        if np.any(M < 0) or np.any(M >= n):
            raise ValueError("cpu_grid_map entries must be in [0, n_cpus-1]")
        return M.astype(cfg.policy_int_dtype, copy=False)
    # row-major modulo assignment
    ids = np.arange(H * W, dtype=np.int64).reshape(H, W) % n
    return ids.astype(cfg.policy_int_dtype, copy=False)


def _dvfs_states(U: np.ndarray, grad: np.ndarray, cfg: PackerConfig) -> np.ndarray:
    """
    Quantize either ||∇Φ|| or normalized U into discrete cfg.dvfs_levels.
    """
    levels = np.array(cfg.dvfs_levels, dtype=np.int64)
    L = levels.size
    if L == 0:
        raise ValueError("dvfs_levels must be non-empty")
    if cfg.dvfs_from == "grad_phi":
        G = np.asarray(grad, dtype=np.float64)
        mag = np.sqrt(G[..., 0] * G[..., 0] + G[..., 1] * G[..., 1])
        M = mag
    else:
        X = np.asarray(U, dtype=np.float64)
        # normalize to [0,1]
        mn = float(np.min(X))
        mx = float(np.max(X))
        if mx - mn <= 1e-20:
            M = np.zeros_like(X)
        else:
            M = (X - mn) / (mx - mn)
    # map M in [0,∞) (or [0,1]) to indices 0..L-1 via linear scaling on [0, p95]
    # Clamp to [0,1] by robust percentile to reduce outlier impact for grad case
    p95 = float(np.percentile(M, 95)) if M.size > 0 else 0.0
    scale = p95 if p95 > 1e-12 else (float(np.max(M)) if float(np.max(M)) > 1e-12 else 1.0)
    R = np.clip(M / scale, 0.0, 1.0)
    idx = np.minimum((R * L).astype(int), L - 1)
    states = levels[idx]
    return states.astype(cfg.policy_int_dtype, copy=False)


# --------------------
# Public API
# --------------------

def make_reconfig_pack(
    g: np.ndarray,
    phi: np.ndarray,
    grad: np.ndarray,
    U: np.ndarray,
    J: np.ndarray,
    B: Union[Dict[str, np.ndarray], np.ndarray],
    geom_meta: Dict[str, Any],
    cfg: PackerConfig,
) -> Dict[str, Any]:
    """
    Build a deterministic, schema-compliant ReconfigPack.

    Inputs:
      - g: ndarray [H,W,2,2] SPD (e.g., g_next)
      - phi: ndarray [H,W]
      - grad: ndarray [H,W,2]
      - U: ndarray [H,W]
      - J: ndarray [H,W]
      - B: dict {"N","E","S","W"} or ndarray [H,W,4] with channels (N,E,S,W)
      - geom_meta: dict from slow_plane.geometry.update.update_geometry(...)[ "meta" ]
      - cfg: PackerConfig

    Returns:
      dict with fields as described in the module docstring.
    """
    H, W = _validate_inputs(g, phi, grad, U, J, B)

    # Directional preferences
    flux_dirs = _flux_to_dirs(B)            # [H,W,4] non-negative
    grad_dirs = _grad_to_dirs(grad)         # [H,W,4] non-negative (from -∇Φ projections)

    # NoC weights
    weights = _compose_noc_weights(grad_dirs, flux_dirs, cfg)  # dtype weight_dtype [H,W,4]

    # Validate sums and bounds
    sums = np.sum(weights.astype(np.float64), axis=-1)
    if not np.all(np.isfinite(weights)):
        raise ValueError("computed NoC weights contain non-finite values")
    if not np.all((weights >= 0.0) & (weights <= 1.0 + 1e-6)):
        raise ValueError("computed NoC weights out of [0,1] bounds")
    if not np.allclose(sums, 1.0, atol=1e-6):
        # Renormalize defensively if numerics drifted beyond tolerance
        weights = (weights.astype(np.float64) / np.maximum(sums[..., None], 1e-12)).astype(cfg.weight_dtype, copy=False)

    # Duplicate for link-level compatibility
    link_weights = weights.copy()

    # MC priorities
    mc_policy_words = _mc_priorities(J, cfg)  # dtype weight_dtype [H,W]

    # CAT masks
    cat_masks = _cat_masks(U, cfg)            # dtype policy_int_dtype [H,W]

    # CPU affinities
    cpu_affinities = _cpu_affinity_map(H, W, cfg)  # dtype policy_int_dtype [H,W]

    # NUMA policies (strings)
    numa_policies = np.full((H, W), "local_first", dtype=object)

    # DVFS states
    dvfs_states = _dvfs_states(U, grad, cfg)  # dtype policy_int_dtype [H,W]

    # Trust region meta: pass-through with required fields; validate presence
    tr_fields = ("accepted", "accept_ratio", "residual_norm", "trust_radius", "hysteresis_left")
    tr_meta: Dict[str, Any] = {}
    for k in tr_fields:
        if k not in geom_meta:
            raise ValueError(f"geom_meta missing required key: {k}")
        tr_meta[k] = geom_meta[k]

    noc_tables: Dict[str, Any] = {
        "weights": weights,
        # Phase A: provide scalar 8 credits
        "vc_credits": np.array(8, dtype=cfg.policy_int_dtype).item(),  # JSON/scalar friendly
    }

    pack: Dict[str, Any] = {
        "version": 1,
        "noc_tables": noc_tables,
        "link_weights": link_weights,
        "mc_policy_words": mc_policy_words,
        "cat_masks": cat_masks,
        "cpu_affinities": cpu_affinities,
        "numa_policies": numa_policies,
        "dvfs_states": dvfs_states,
        "trust_region_meta": tr_meta,
    }

    if cfg.crc_enable:
        # Compute CRC over canonical JSON of the pack excluding crc32c
        payload_bytes = _canonical_json_bytes(pack)
        crc = _crc32c(payload_bytes)
        pack["crc32c"] = int(crc)

    return pack