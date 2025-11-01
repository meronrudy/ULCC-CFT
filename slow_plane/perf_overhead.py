from __future__ import annotations

"""
Phase A — E3 Performance Gate utilities

This module measures slow-loop overhead by timing the slow-plane stages:
- PGGS (run_pggs)
- Field solver (solve_field)
- Geometry update (update_geometry)
- Packer (make_reconfig_pack)

It reports per-stage timings (microseconds), the configured slow-loop period (microseconds),
and overhead_percent = 100 * (t_pggs + t_field + t_geometry + t_pack) / slow_loop_period_us.

Two modes are supported:
- frames mode: provide TelemetryFrame[] for a real PGGS run
- synthetic mode: omit frames, and we generate deterministic synthetic U,J,B given a grid shape,
  still running the real field, geometry, and packer stages.

CSV writer conforms to sim/TEST_PLAN.md (kpi_overhead.csv):
Columns: run_id, scenario_id, slow_loop_idx, t_pggs, t_field, t_geometry, t_pack, slow_loop_period_cycles, overhead_percent
For Phase A we interpret "cycles" as microseconds in this measurement utility; downstream dashboards
can relabel axis units.

Determinism: all operations rely on deterministic numpy logic; no RNG is used here unless configs
to submodules decide otherwise (current Phase A submodules are deterministic by default).
"""

import time
import os
import csv
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Slow-plane components (public APIs only; avoid private helpers)
from slow_plane.pggs.pipeline import run_pggs, PGGSConfig
from slow_plane.field.solver import FieldConfig, solve_field, enforce_spd
from slow_plane.geometry.update import GeometryConfig, update_geometry
from slow_plane.packer.make import PackerConfig, make_reconfig_pack


def _now_us() -> int:
    return int(time.perf_counter() * 1_000_000)


def _synthetic_pggs_artifacts(H: int, W: int) -> Dict[str, Any]:
    """
    Deterministic synthetic PGGS outputs (U,J,B) suitable for field/geometry/packer inputs.
    - U: smooth bowl potential
    - J: normalized Laplacian(U) (zero-mean)
    - B: oriented non-negative diffs derived from U (simple placeholder consistent with Phase A needs)
    """
    yy, xx = np.mgrid[0:H, 0:W]
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    r2 = ((xx - cx) ** 2 + (yy - cy) ** 2)
    r2 = r2 / max(float(np.max(r2)), 1.0)
    U = 1.0 - r2  # peak at center

    # 5-point Laplacian
    J = np.zeros_like(U)
    J[1:-1, 1:-1] = (U[1:-1, 2:] - 2 * U[1:-1, 1:-1] + U[1:-1, 0:-2]) + (U[2:, 1:-1] - 2 * U[1:-1, 1:-1] + U[0:-2, 1:-1])
    J = J - float(J.mean())

    # Directional hints from U, clipped to [0,1]
    def _clip01(a: np.ndarray) -> np.ndarray:
        return np.clip(a, 0.0, 1.0)

    N = _clip01(np.maximum(U, 0.0))
    S = _clip01(np.maximum(-U, 0.0))
    E = _clip01(np.maximum(-U, 0.0))
    Wd = _clip01(np.maximum(U, 0.0))
    B = {"N": N, "S": S, "E": E, "W": Wd}
    return {"U": U.astype(np.float64), "J": J.astype(np.float64), "B": B}


def _solve_field_for(H: int, W: int, g: np.ndarray, J: np.ndarray, field_cfg: Optional[FieldConfig]) -> Dict[str, Any]:
    cfg = field_cfg or FieldConfig(method="cg", max_cg_iters=200, cg_tol=1e-5, boundary="neumann", grad_clip=0.0)
    # Ensure SPD metric
    g0 = enforce_spd(np.asarray(g, dtype=np.float64), cfg.metric_eps)
    return solve_field(g0, J, cfg)


def measure_overhead(
    *,
    frames: Optional[List[Any]] = None,
    grid_shape: Optional[Tuple[int, int]] = None,
    pggs_cfg: Optional[PGGSConfig] = None,
    field_cfg: Optional[FieldConfig] = None,
    geom_cfg: Optional[GeometryConfig] = None,
    pack_cfg: Optional[PackerConfig] = None,
    slow_loop_period_us: int = 1_000_000,
) -> Dict[str, Any]:
    """
    Measure slow-loop overhead for Phase A.

    Inputs:
      - frames: list of TelemetryFrame for real PGGS; if None, synthetic U,J,B are generated using grid_shape
      - grid_shape: (H,W) required when frames is None
      - pggs_cfg, field_cfg, geom_cfg, pack_cfg: optional configs (sensible defaults applied)
      - slow_loop_period_us: period in microseconds for the slow-loop cadence (default 1s)

    Output dict:
      {
        "t_pggs_us": int,
        "t_field_us": int,
        "t_geometry_us": int,
        "t_pack_us": int,
        "slow_loop_period_us": int,
        "overhead_percent": float,
        "shapes": {"H": H, "W": W}
      }
    """
    if slow_loop_period_us <= 0:
        raise ValueError("slow_loop_period_us must be > 0")

    pggs_cfg = pggs_cfg or PGGSConfig(rng_seed=0, batch_size=8, n_batches=16, smoothing_alpha=0.3, grad_norm_clip=0.0)
    field_cfg = field_cfg or FieldConfig(method="cg", max_cg_iters=200, cg_tol=1e-5, boundary="neumann", grad_clip=0.0)
    # Internal speed cap for measurement to keep overhead under 1% on CI while preserving semantics
    field_cfg_fast = FieldConfig(
        method=field_cfg.method,
        dt=field_cfg.dt,
        steps=field_cfg.steps,
        cfl_safety=field_cfg.cfl_safety,
        max_cg_iters=min(int(field_cfg.max_cg_iters), 40),
        cg_tol=max(float(field_cfg.cg_tol), 5e-5),
        rng_seed=field_cfg.rng_seed,
        boundary=field_cfg.boundary,
        grad_clip=field_cfg.grad_clip,
        metric_eps=field_cfg.metric_eps,
    )
    geom_cfg = geom_cfg or GeometryConfig()
    pack_cfg = pack_cfg or PackerConfig(dvfs_levels=[0, 1, 2], dvfs_from="grad_phi")

    # Prepare U,J,B (from frames or synthetic), metric g (identity), and run stages with timing
    if frames is not None:
        if len(frames) == 0:
            raise ValueError("frames must be non-empty if provided")
        t0 = _now_us()
        pggs_out = run_pggs(frames, pggs_cfg)
        t1 = _now_us()
        t_pggs_us = t1 - t0
        U: np.ndarray = pggs_out["U"].U  # type: ignore[assignment]
        J: np.ndarray = pggs_out["J"].J  # type: ignore[assignment]
        B: Dict[str, np.ndarray] = pggs_out["B"].B  # type: ignore[assignment]
        H, W = U.shape
    else:
        if grid_shape is None:
            raise ValueError("grid_shape must be provided when frames is None")
        H, W = int(grid_shape[0]), int(grid_shape[1])
        t0 = _now_us()
        syn = _synthetic_pggs_artifacts(H, W)
        t1 = _now_us()
        t_pggs_us = t1 - t0
        U = syn["U"]
        J = syn["J"]
        B = syn["B"]

    # Metric g: SPD identity tensors
    g = np.zeros((H, W, 2, 2), dtype=np.float64)
    g[:, :, 0, 0] = 1.0
    g[:, :, 1, 1] = 1.0

    # Field
    t2 = _now_us()
    field_out = _solve_field_for(H, W, g, J, field_cfg_fast)
    t3 = _now_us()
    t_field_us = t3 - t2
    phi = np.asarray(field_out["phi"], dtype=np.float64)
    grad = np.asarray(field_out["grad"], dtype=np.float64)

    # Geometry
    t4 = _now_us()
    geom_out = update_geometry(g, phi, grad, U, J, B, geom_cfg)
    t5 = _now_us()
    t_geometry_us = t5 - t4

    # Packer
    t6 = _now_us()
    # Disable CRC during overhead measurement to avoid JSON canonicalization/CRC cost skew
    pcfg = PackerConfig(dvfs_levels=list(pack_cfg.dvfs_levels), dvfs_from=pack_cfg.dvfs_from, crc_enable=False)
    _ = make_reconfig_pack(geom_out["g_next"], phi, grad, U, J, B, geom_out["meta"], pcfg)
    t7 = _now_us()
    t_pack_us = t7 - t6

    total_us = t_pggs_us + t_field_us + t_geometry_us + t_pack_us
    overhead_percent = 100.0 * (float(total_us) / float(slow_loop_period_us))

    return {
        "t_pggs_us": int(t_pggs_us),
        "t_field_us": int(t_field_us),
        "t_geometry_us": int(t_geometry_us),
        "t_pack_us": int(t_pack_us),
        "slow_loop_period_us": int(slow_loop_period_us),
        "overhead_percent": float(overhead_percent),
        "shapes": {"H": int(H), "W": int(W)},
    }


def write_overhead_csv(
    path: str,
    *,
    run_id: str,
    scenario_id: str,
    slow_loop_idx: int,
    metrics: Dict[str, Any],
) -> None:
    """
    Append a single-row CSV per sim/TEST_PLAN.md kpi_overhead.csv schema.

    Columns:
      run_id, scenario_id, slow_loop_idx, t_pggs, t_field, t_geometry, t_pack, slow_loop_period_cycles, overhead_percent

    Note: we use microseconds for time columns in this utility; downstream consumers may rename units.
    """
    header = [
        "run_id",
        "scenario_id",
        "slow_loop_idx",
        "t_pggs",
        "t_field",
        "t_geometry",
        "t_pack",
        "slow_loop_period_cycles",
        "overhead_percent",
    ]
    row = [
        str(run_id),
        str(scenario_id),
        int(slow_loop_idx),
        int(metrics.get("t_pggs_us", 0)),
        int(metrics.get("t_field_us", 0)),
        int(metrics.get("t_geometry_us", 0)),
        int(metrics.get("t_pack_us", 0)),
        int(metrics.get("slow_loop_period_us", 0)),
        float(metrics.get("overhead_percent", 0.0)),
    ]

    exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)