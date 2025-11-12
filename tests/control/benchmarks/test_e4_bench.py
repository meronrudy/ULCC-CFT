from __future__ import annotations

import math

from harness.e4_bench import bench_slow_plane_overhead, bench_control_apply_cycle  # [python.import()](harness/e4_bench.py:1)


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def test_bench_slow_plane_overhead_smoke():
    res = bench_slow_plane_overhead(grid_shape=(4, 4), slow_loop_period_us=1_000_000)  # [python.e4_bench.bench_slow_plane_overhead()](harness/e4_bench.py:105)
    # Required keys
    for k in ("t_pggs_us", "t_field_us", "t_geometry_us", "t_pack_us", "overhead_percent", "shapes"):
        assert k in res, f"missing key: {k}"

    assert _is_num(res["t_pggs_us"])
    assert _is_num(res["t_field_us"])
    assert _is_num(res["t_geometry_us"])
    assert _is_num(res["t_pack_us"])
    assert _is_num(res["overhead_percent"])

    shapes = res.get("shapes", {})
    assert shapes.get("H") == 4
    assert shapes.get("W") == 4


def test_bench_control_apply_cycle_smoke():
    ctl = bench_control_apply_cycle(grid_shape=(2, 2), iterations=5)  # [python.e4_bench.bench_control_apply_cycle()](harness/e4_bench.py:132)
    assert ctl["ok_count"] == 5
    assert ctl["fail_count"] == 0

    stats = ctl.get("stats", {})
    for k in ("min_us", "max_us", "mean_us", "p50_us", "p95_us"):
        assert k in stats, f"missing stat: {k}"
        assert _is_num(stats[k])

    timings = ctl.get("timings_us", [])
    assert isinstance(timings, list)
    assert len(timings) == 5
    assert all(_is_num(t) for t in timings)