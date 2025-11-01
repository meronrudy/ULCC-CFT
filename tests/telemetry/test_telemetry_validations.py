from telemetry.validator import validate_frame, filter_and_count
from telemetry.frame import (
    TelemetryFrame,
    TileMetrics,
    LinkMetrics,
    MemoryMetrics,
    SchedulerMetrics,
    PowerThermal,
    MemoryQueuesDepth,
)
from typing import List


def _mk_valid_frame(cycle: int) -> TelemetryFrame:
    tiles: List[TileMetrics] = [
        TileMetrics(
            tile_id=0,
            flit_tx=1,
            flit_rx=1,
            vc_depth_avg=[0.0, 0.0],
            vc_depth_p99=[0.0, 0.0],
            queue_depth_avg=0.0,
            queue_depth_p99=0.0,
            service_time_avg=0.0,
            service_time_var=0.0,
            stalls={"credit_starve": 0, "hazard": 0, "mc_block": 0},
            mpki=0.0,
            ipc=0.0,
            temp_c=50.0,
            power_pu=0.5,
        )
    ]
    links: List[LinkMetrics] = []
    mms: List[MemoryMetrics] = [
        MemoryMetrics(
            mc_id=0,
            queues=MemoryQueuesDepth(rdq_depth_avg=0.0, wrq_depth_avg=0.0),
            fr_fcfs_hit=0,
            activations=0,
            precharges=0,
            bandwidth_util=0.0,
            read_latency_avg=0.0,
            read_latency_p99=0.0,
            throttles=0,
        )
    ]
    sched = SchedulerMetrics(
        runnable_tasks_avg=0.0,
        runnable_tasks_p99=0.0,
        migrations=0,
        preemptions=0,
        affinity_violations=0,
    )
    power = PowerThermal(tdp_proxy_pu=0.5, thermal_ceiling_hits=0, dvfs_state_counts={})
    fr = TelemetryFrame(
        schema_version="1.0.0",
        frame_id=f"F{cycle:016d}",
        run_id="run-val",
        rng_seed=0,
        grid_shape={"width": 1, "height": 1},
        cycle_window=1,
        t_start_cycle=cycle,
        t_end_cycle=cycle + 1,
        sampling_mode="windowed",
        tile_metrics=tiles,
        link_metrics=links,
        memory_metrics=mms,
        scheduler_metrics=sched,
        power_thermal=power,
        anomalies=[],
        meta={},
    ).with_crc()
    return fr


def test_validate_frame_ok():
    fr = _mk_valid_frame(0)
    ok, reason = validate_frame(fr)
    assert ok is True
    assert reason is None


def test_validate_crc_mismatch_reason():
    fr = _mk_valid_frame(10)
    # Corrupt CRC explicitly
    fr.frame_crc32c = (fr.frame_crc32c or 0) ^ 0xFFFFFFFF
    ok, reason = validate_frame(fr)
    assert ok is False
    assert reason == "crc_mismatch"


def test_validate_bounds_violation_reason():
    fr = _mk_valid_frame(20)
    # Make a tile out of bounds temperature
    fr.tile_metrics[0].temp_c = 200.0
    # Need to update CRC to keep the check path meaningful; but validate_frame maps bounds first
    fr = fr.with_crc()
    ok, reason = validate_frame(fr)
    assert ok is False
    assert reason == "bounds_violation"


def test_validate_missing_required_field_reason():
    fr = _mk_valid_frame(30)
    # Make grid invalid (width=0) to trigger "must" message
    fr.grid_shape = {"width": 0, "height": 1}
    fr = fr.with_crc()
    ok, reason = validate_frame(fr)
    assert ok is False
    assert reason == "missing_required_field"


def test_filter_and_count_mixture():
    valid = _mk_valid_frame(0)

    crc_bad = _mk_valid_frame(1)
    crc_bad.frame_crc32c = (crc_bad.frame_crc32c or 0) ^ 0xFFFFFFFF

    bounds_bad = _mk_valid_frame(2)
    bounds_bad.tile_metrics[0].temp_c = 200.0
    bounds_bad = bounds_bad.with_crc()

    missing_bad = _mk_valid_frame(3)
    missing_bad.grid_shape = {"width": 0, "height": 1}
    missing_bad = missing_bad.with_crc()

    frames = [valid, crc_bad, bounds_bad, missing_bad]
    kept, stats = filter_and_count(frames)
    assert stats["total"] == 4
    assert stats["valid"] == 1
    assert stats["crc_fail"] == 1
    assert stats["bounds_fail"] == 1
    assert stats["missing"] == 1
    assert len(kept) == 1