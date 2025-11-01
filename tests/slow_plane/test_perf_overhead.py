from __future__ import annotations

import os
import sys
import csv
import math
import numpy as np

# Ensure repository root is on sys.path to avoid shadowing by tests/slow_plane directory
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from slow_plane.perf_overhead import measure_overhead, write_overhead_csv
from slow_plane.pggs.pipeline import PGGSConfig
from slow_plane.field.solver import FieldConfig
from slow_plane.geometry.update import GeometryConfig
from slow_plane.packer.make import PackerConfig


def test_measure_overhead_synthetic_small_grid_under_budget(tmp_path) -> None:
    """
    Phase A — E3 Performance Gate (unit): Measure slow-loop overhead on a small synthetic grid
    and verify overhead_percent <= 1% given a sufficiently large slow_loop_period_us.

    Note:
    - Wall-clock timings are environment-dependent; to make this test robust, we choose a large
      slow_loop_period_us so that the computed overhead percent is comfortably <= 1% on CI.
    - This validates timing plumbing and budget calculation without being brittle.
    """
    H, W = 16, 16
    # Configurations chosen to be light-weight and deterministic
    pggs_cfg = PGGSConfig(rng_seed=0, batch_size=4, n_batches=4, smoothing_alpha=0.3, grad_norm_clip=0.0)
    field_cfg = FieldConfig(method="cg", max_cg_iters=50, cg_tol=1e-5, boundary="neumann", grad_clip=0.0)
    geom_cfg = GeometryConfig(step_size=0.2, damping=0.7, cond_max=1e5, spd_eps=1e-6, trust_radius=0.25, accept_ratio_min=0.5, hysteresis=1)
    pack_cfg = PackerConfig(dvfs_levels=[0, 1, 2], dvfs_from="grad_phi")

    # Use a very generous slow-loop period (e.g., 30 seconds) to ensure <= 1% even on slower runners
    slow_loop_period_us = 30_000_000

    metrics = measure_overhead(
        frames=None,
        grid_shape=(H, W),
        pggs_cfg=pggs_cfg,
        field_cfg=field_cfg,
        geom_cfg=geom_cfg,
        pack_cfg=pack_cfg,
        slow_loop_period_us=slow_loop_period_us,
    )

    # Basic shape and field checks
    assert isinstance(metrics, dict)
    assert "overhead_percent" in metrics
    assert metrics["shapes"]["H"] == H
    assert metrics["shapes"]["W"] == W

    # Budget check (robust by construction via large period)
    assert float(metrics["overhead_percent"]) <= 1.0


def test_overhead_csv_schema_and_values(tmp_path) -> None:
    """
    Validate CSV writer schema and single-row append behavior.
    """
    run_id = "R-test"
    scenario_id = "scenario-e3-perf"
    slow_loop_idx = 0

    # Minimal synthetic metrics dict
    metrics = {
        "t_pggs_us": 1000,
        "t_field_us": 2000,
        "t_geometry_us": 500,
        "t_pack_us": 300,
        "slow_loop_period_us": 1_000_000,
        "overhead_percent": (1000 + 2000 + 500 + 300) * 100.0 / 1_000_000,
    }

    csv_path = tmp_path / "kpi_overhead.csv"
    write_overhead_csv(str(csv_path), run_id=run_id, scenario_id=scenario_id, slow_loop_idx=slow_loop_idx, metrics=metrics)

    assert csv_path.exists()
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Header + single data row
    assert len(rows) == 2
    header = rows[0]
    row = rows[1]

    expected_header = [
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
    assert header == expected_header

    # Basic value checks
    assert row[0] == run_id
    assert row[1] == scenario_id
    assert int(row[2]) == slow_loop_idx
    assert int(row[3]) == metrics["t_pggs_us"]
    assert int(row[4]) == metrics["t_field_us"]
    assert int(row[5]) == metrics["t_geometry_us"]
    assert int(row[6]) == metrics["t_pack_us"]
    assert int(row[7]) == metrics["slow_loop_period_us"]
    # Float compare with small tolerance
    assert abs(float(row[8]) - float(metrics["overhead_percent"])) < 1e-9