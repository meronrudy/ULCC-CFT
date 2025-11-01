from telemetry.aggregator import TelemetryAggregator, WindowSpec
from telemetry.validator import filter_and_count
from telemetry.frame import (
    TelemetryFrame,
    TileMetrics,
    LinkMetrics,
    MemoryMetrics,
    SchedulerMetrics,
    PowerThermal,
    MemoryQueuesDepth,
)
from typing import List, Dict


def _make_tile(tile_id: int, flit_tx: int, q_p99: float = 1.0, temp: float = 50.0, pwr: float = 0.5) -> TileMetrics:
    return TileMetrics(
        tile_id=tile_id,
        flit_tx=flit_tx,
        flit_rx=flit_tx,  # symmetric for simplicity
        vc_depth_avg=[0.0, 0.0],
        vc_depth_p99=[0.0, 0.0],
        queue_depth_avg=q_p99 / 2.0,
        queue_depth_p99=q_p99,
        service_time_avg=0.0,
        service_time_var=0.0,
        stalls={"credit_starve": 0, "hazard": 0, "mc_block": 0},
        mpki=0.0,
        ipc=0.0,
        temp_c=temp,
        power_pu=pwr,
    )


def _make_frame(cycle: int, w: int, h: int, flits_per_tile: int, with_mc: bool) -> TelemetryFrame:
    # Single-cycle window
    tiles = [ _make_tile(tid, flits_per_tile, q_p99=1.0 + (tid % 3), temp=45.0 + (tid % 5), pwr=0.5) for tid in range(w * h) ]
    link_metrics: List[LinkMetrics] = []  # not used in E2 tests
    memory_metrics: List[MemoryMetrics] = []
    if with_mc:
        # One MC with activations and latencies
        memory_metrics.append(
            MemoryMetrics(
                mc_id=0,
                queues=MemoryQueuesDepth(rdq_depth_avg=0.0, wrq_depth_avg=0.0),
                fr_fcfs_hit=0,
                activations=2,  # served requests proxy
                precharges=0,
                bandwidth_util=0.5,
                read_latency_avg=200.0,
                read_latency_p99=600.0,
                throttles=0,
            )
        )

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
        run_id="run-agg",
        rng_seed=0,
        grid_shape={"width": w, "height": h},
        cycle_window=1,
        t_start_cycle=cycle,
        t_end_cycle=cycle + 1,
        sampling_mode="windowed",
        tile_metrics=tiles,
        link_metrics=link_metrics,
        memory_metrics=memory_metrics,
        scheduler_metrics=sched,
        power_thermal=power,
        anomalies=[],
        meta={"scenario_id": "scenario-fixed"},
    ).with_crc()
    return fr


def test_windowing_and_kpis_basic():
    # 4x4 mesh, 30 cycles, interval 1 (one frame per cycle)
    w, h = 4, 4
    T = 30
    frames = []
    for t in range(T):
        # flits per tile grows slowly to get non-zero throughput
        flits = 1 + (t % 3)
        frames.append(_make_frame(t, w, h, flits_per_tile=flits, with_mc=True))

    # Validate then filter
    valid_frames, stats = filter_and_count(frames)
    assert stats["total"] == T
    assert stats["valid"] == T

    # Aggregate windows of 10 cycles, stride 10
    agg = TelemetryAggregator(WindowSpec(window_cycles=10, stride_cycles=10, align_to=0))
    for fr in valid_frames:
        agg.push(fr)

    windows = []
    while agg.ready():
        windows.append(agg.next_window())

    assert len(windows) == T // 10  # floor(30/10) == 3

    # produced_flits_total should be non-zero and reasonably consistent
    for wrec in windows:
        assert wrec.cycles == 10
        assert wrec.produced_flits_total > 0

    # Memory: served requests >= 0 and avg latency finite when >0
    for wrec in windows:
        assert wrec.mc_served_requests >= 0
        if wrec.mc_served_requests > 0:
            assert wrec.mc_avg_latency is not None
            assert wrec.mc_avg_latency >= 0.0

    # Power energy monotonic non-decreasing across windows (constant power)
    energies = [wrec.power_total_energy for wrec in windows]
    for i in range(1, len(energies)):
        assert energies[i] >= energies[i - 1]


def test_windowing_without_mc_and_models_defaults():
    # No MC present; power/thermal still exist per-frame, but allow aggregator defaults
    w, h = 2, 2
    T = 20
    frames = [_make_frame(t, w, h, flits_per_tile=1, with_mc=False) for t in range(T)]

    valid_frames, stats = filter_and_count(frames)
    assert stats["valid"] == T

    agg = TelemetryAggregator(WindowSpec(window_cycles=10, stride_cycles=10, align_to=0))
    for fr in valid_frames:
        agg.push(fr)

    windows = []
    while agg.ready():
        windows.append(agg.next_window())

    assert len(windows) == 2
    for wrec in windows:
        # No MC → latencies None, served requests 0
        assert wrec.mc_served_requests >= 0
        if wrec.mc_served_requests == 0:
            assert wrec.mc_avg_latency is None
            assert wrec.mc_p95_latency is None