from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, List
import numpy as np

_EPS = 1e-12

def combine_seeds(base_seed: int, frame_seeds: Iterable[int]) -> int:
    """Deterministically combine seeds via xor-folding with a fixed avalanche."""
    s = np.uint64(base_seed)
    for fs in frame_seeds:
        x = np.uint64(int(fs))
        # mix: xorshift-like avalanche
        x ^= (x << np.uint64(13)) & np.uint64(0xFFFFFFFFFFFFFFFF)
        x ^= (x >> np.uint64(7))
        x ^= (x << np.uint64(17)) & np.uint64(0xFFFFFFFFFFFFFFFF)
        s ^= x
    return int(s & np.uint64(0x7FFFFFFF))  # fit into 31-bit for numpy RNG portability

def grid_shape_from_frames(frames) -> Tuple[int,int]:
    if not frames:
        raise ValueError("run_pggs: frames list must be non-empty")
    w = int(frames[0].grid_shape.get('width', 0))
    h = int(frames[0].grid_shape.get('height', 0))
    if w <= 0 or h <= 0:
        raise ValueError("run_pggs: invalid grid_shape in frames[0]")
    for fr in frames[1:]:
        w2 = int(fr.grid_shape.get('width', 0))
        h2 = int(fr.grid_shape.get('height', 0))
        if w2 != w or h2 != h:
            raise ValueError("run_pggs: inconsistent grid_shape across frames")
    return (w, h)

def tile_index_maps(w: int, h: int) -> Tuple[Dict[int, Tuple[int,int]], Dict[Tuple[int,int], int]]:
    """Row-major id map: id = y*w + x"""
    id_to_xy: Dict[int, Tuple[int,int]] = {}
    xy_to_id: Dict[Tuple[int,int], int] = {}
    for y in range(h):
        for x in range(w):
            tid = y * w + x
            id_to_xy[tid] = (x, y)
            xy_to_id[(x, y)] = tid
    return id_to_xy, xy_to_id

def make_activity_arrays(frames, w: int, h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate per-tile proxies across frames deterministically.
    Returns arrays with shape (h, w): flit_tx_sum, queue_p99_mean, power_mean
    """
    flit_tx = np.zeros((h, w), dtype=np.float64)
    q_p99_acc = np.zeros((h, w), dtype=np.float64)
    q_p99_cnt = np.zeros((h, w), dtype=np.float64)
    power_acc = np.zeros((h, w), dtype=np.float64)
    power_cnt = np.zeros((h, w), dtype=np.float64)

    for fr in frames:
        for tm in fr.tile_metrics:
            tid = int(tm.tile_id)
            y, x = divmod(tid, w)
            flit_tx[y, x] += float(tm.flit_tx)
            q_p99_acc[y, x] += float(tm.queue_depth_p99)
            q_p99_cnt[y, x] += 1.0
            power_acc[y, x] += float(tm.power_pu)
            power_cnt[y, x] += 1.0

    q_p99 = np.divide(q_p99_acc, np.maximum(q_p99_cnt, 1.0), out=np.zeros_like(q_p99_acc), where=q_p99_cnt > 0)
    power = np.divide(power_acc, np.maximum(power_cnt, 1.0), out=np.zeros_like(power_acc), where=power_cnt > 0)
    return flit_tx, q_p99, power

def deterministic_subsets(w: int, h: int, batch_size: int, n_batches: int, seed: int) -> List[np.ndarray]:
    """Produce n_batches deterministic subsets of linear tile indices length batch_size each."""
    N = w * h
    idx = np.arange(N, dtype=np.int32)
    rng = np.random.default_rng(seed)
    # One global permutation, then stride windows for batches to reduce RNG use and ensure determinism
    perm = rng.permutation(idx)
    out = []
    for b in range(n_batches):
        start = (b * batch_size) % N
        # cycle through perm deterministically
        sel = np.take(perm, np.arange(start, start + batch_size) % N, mode='wrap')
        out.append(sel)
    return out

def ema_update(prev: np.ndarray, delta: np.ndarray, alpha: float) -> np.ndarray:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("smoothing_alpha must be in [0,1]")
    return (1.0 - alpha) * prev + alpha * delta

def clip_l2(vec: np.ndarray, max_norm: float) -> np.ndarray:
    if max_norm is None or max_norm <= 0.0:
        return vec
    n = np.linalg.norm(vec.ravel(), ord=2)
    if n <= max_norm + _EPS:
        return vec
    return vec * (max_norm / (n + _EPS))

def divergence_2d(U: np.ndarray) -> np.ndarray:
    """Simple 2D divergence proxy using forward diffs with zero-flux boundaries."""
    h, w = U.shape
    div = np.zeros_like(U, dtype=np.float64)
    # East-West
    div[:, :-1] += U[:, :-1] - U[:, 1:]
    div[:, 1:]  += U[:, 1:]  - U[:, :-1]
    # North-South
    div[:-1, :] += U[:-1, :] - U[1:, :]
    div[1:, :]  += U[1:, :]  - U[:-1, :]
    return div

def normalize_zero_sum(J: np.ndarray) -> np.ndarray:
    s = float(J.sum())
    if abs(s) < _EPS:
        return J
    Jc = J - s / J.size
    return Jc

def neighbor_diffs(U: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute oriented neighbor differences of U: N,S,E,W arrays with same shape as U (padding zeros at borders)."""
    h, w = U.shape
    out: Dict[str, np.ndarray] = {
        "N": np.zeros_like(U, dtype=np.float64),
        "S": np.zeros_like(U, dtype=np.float64),
        "E": np.zeros_like(U, dtype=np.float64),
        "W": np.zeros_like(U, dtype=np.float64),
    }
    out["N"][1:, :] = U[:-1, :] - U[1:, :]
    out["S"][:-1, :] = U[1:, :] - U[:-1, :]
    out["E"][:, :-1] = U[:, 1:] - U[:, :-1]
    out["W"][:, 1:] = U[:, :-1] - U[:, 1:]
    return out

def safe_scale(arr: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return arr * scale

def ensure_float64(x: np.ndarray) -> np.ndarray:
    if x.dtype != np.float64:
        return x.astype(np.float64, copy=False)
    return x
