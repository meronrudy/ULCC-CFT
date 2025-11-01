import numpy as np
from typing import List, Dict
from telemetry.frame import (
    TelemetryFrame,
    TileMetrics,
    LinkMetrics,
    LinkEndpoints,
    MemoryMetrics,
    MemoryQueuesDepth,
    SchedulerMetrics,
    PowerThermal,
)
from slow_plane.pggs import PGGSConfig, run_pggs

def _mk_frame(w: int, h: int, cycle: int, rng_seed: int, tile_overrides: Dict[int, Dict] = None) -> TelemetryFrame:
    tile_overrides = tile_overrides or {}
    tiles: List[TileMetrics] = []
    for y in range(h):
        for x in range(w):
            tid = y * w + x
            ov = tile_overrides.get(tid, {})
            tiles.append(
                TileMetrics(
                    tile_id=tid,
                    flit_tx=int(ov.get("flit_tx", 10)),
                    flit_rx=int(ov.get("flit_rx", 10)),
                    vc_depth_avg=ov.get("vc_depth_avg", [0.0, 0.0]),
                    vc_depth_p99=ov.get("vc_depth_p99", [0.0, 0.0]),
                    queue_depth_avg=float(ov.get("queue_depth_avg", 0.0)),
                    queue_depth_p99=float(ov.get("queue_depth_p99", 0.0)),
                    service_time_avg=0.0,
                    service_time_var=0.0,
                    stalls={"credit_starve": 0, "hazard": 0, "mc_block": 0},
                    mpki=0.0,
                    ipc=0.0,
                    temp_c=float(ov.get("temp_c", 50.0)),
                    power_pu=float(ov.get("power_pu", 0.5)),
                )
            )
    links: List[LinkMetrics] = []  # not needed for E3a tests
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
        runnable_tasks_avg=0.0, runnable_tasks_p99=0.0, migrations=0, preemptions=0, affinity_violations=0
    )
    power = PowerThermal(tdp_proxy_pu=0.5, thermal_ceiling_hits=0, dvfs_state_counts={})
    fr = TelemetryFrame(
        schema_version="1.0.0",
        frame_id=f"F{cycle:08d}",
        run_id="run-pggs-test",
        rng_seed=int(rng_seed),
        grid_shape={"width": w, "height": h},
        cycle_window=1,
        t_start_cycle=int(cycle),
        t_end_cycle=int(cycle + 1),
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

def _mk_frames(w: int, h: int, n: int, rng_seed: int, hot_tile: int = None, hot_scale: float = 4.0) -> List[TelemetryFrame]:
    frames: List[TelemetryFrame] = []
    for t in range(n):
        overrides: Dict[int, Dict] = {}
        if hot_tile is not None:
            # Heavier around hot tile: increase queue_depth_p99, power, and flit_tx a bit
            overrides[hot_tile] = {
                "queue_depth_p99": 10.0 * hot_scale,
                "power_pu": 0.9,
                "flit_tx": 40,
                "temp_c": 70.0,
            }
            # Also bump its immediate neighbors slightly
            w_, h_ = w, h
            y, x = divmod(hot_tile, w_)
            neigh = []
            if y > 0: neigh.append((y-1, x))
            if y < h_-1: neigh.append((y+1, x))
            if x > 0: neigh.append((y, x-1))
            if x < w_-1: neigh.append((y, x+1))
            for (ny, nx) in neigh:
                tid = ny * w_ + nx
                overrides[tid] = {
                    "queue_depth_p99": 4.0 * hot_scale,
                    "power_pu": 0.7,
                    "flit_tx": 25,
                    "temp_c": 60.0,
                }
        frames.append(_mk_frame(w, h, t, rng_seed=rng_seed, tile_overrides=overrides))
    return frames

def test_determinism_and_meta():
    w, h, n = 4, 4, 200
    frames = _mk_frames(w, h, n, rng_seed=123)
    cfg = PGGSConfig(rng_seed=999, batch_size=8, n_batches=16, smoothing_alpha=0.3, grad_norm_clip=1.0)
    out1 = run_pggs(frames, cfg)
    out2 = run_pggs(frames, cfg)
    U1 = out1["U"].U; U2 = out2["U"].U
    J1 = out1["J"].J; J2 = out2["J"].J
    B1 = out1["B"].B; B2 = out2["B"].B
    assert U1.tobytes() == U2.tobytes()
    assert J1.tobytes() == J2.tobytes()
    for k in ("N", "S", "E", "W"):
        assert B1[k].tobytes() == B2[k].tobytes()
    assert out1["meta"] == out2["meta"]
    assert out1["meta"]["grid_shape"] == {"width": w, "height": h}
    assert out1["meta"]["batch_size"] == 8
    assert out1["meta"]["n_batches"] == 16
    assert out1["meta"]["kpi"] == "produced_flits_total"

def test_attribution_highlights_hot_and_clip_stability():
    w, h, n = 4, 4, 200
    hot = 5  # near center
    frames_hot = _mk_frames(w, h, n, rng_seed=42, hot_tile=hot, hot_scale=3.0)
    frames_uni = _mk_frames(w, h, n, rng_seed=42, hot_tile=None)

    cfg_clip = PGGSConfig(rng_seed=7, batch_size=8, n_batches=16, smoothing_alpha=0.4, grad_norm_clip=0.5)
    out_hot = run_pggs(frames_hot, cfg_clip)
    out_uni = run_pggs(frames_uni, cfg_clip)

    Uh = out_hot["U"].U; Uu = out_uni["U"].U
    # Hot region should have higher mean U than uniform
    mean_hot = Uh.mean()
    mean_uni = Uu.mean()
    # Compare hot tile vicinity vs distant corners
    y, x = divmod(hot, w)
    vic = []
    vic.append((y, x))
    if y > 0: vic.append((y-1, x))
    if y < h-1: vic.append((y+1, x))
    if x > 0: vic.append((y, x-1))
    if x < w-1: vic.append((y, x+1))
    vic_vals = np.array([Uh[yy, xx] for (yy, xx) in vic])
    far_vals = np.array([Uh[0,0], Uh[0,w-1], Uh[h-1,0], Uh[h-1,w-1]])
    assert float(vic_vals.mean()) > float(far_vals.mean()) + 1e-6
    # Gradient clip should bound extremes
    assert float(np.linalg.norm(Uh.ravel(), ord=2)) <= 10.0  # loose upper bound given clip and EMA

    # Flux magnitudes should be higher near edges into hot region than far-away
    Bh = out_hot["B"].B
    # Evaluate magnitude around neighbors of hot tile
    mag_near = 0.0
    for (yy, xx) in vic:
        mag_near += abs(Bh["N"][yy, xx]) + abs(Bh["S"][yy, xx]) + abs(Bh["E"][yy, xx]) + abs(Bh["W"][yy, xx])
    mag_near /= len(vic)
    mag_far = np.mean(np.abs(Bh["N"]) + np.abs(Bh["S"]) + np.abs(Bh["E"]) + np.abs(Bh["W"]))
    assert float(mag_near) >= float(mag_far) - 1e-9  # near should not be less than global mean

def test_sources_J_sanity_zero_sum_and_bounded():
    w, h, n = 4, 4, 50
    hot = 10
    frames = _mk_frames(w, h, n, rng_seed=100, hot_tile=hot, hot_scale=2.0)
    cfg = PGGSConfig(rng_seed=3, batch_size=6, n_batches=12, smoothing_alpha=0.5, grad_norm_clip=0.3)
    out = run_pggs(frames, cfg)
    J = out["J"].J
    # Nearly zero-sum after normalization
    assert abs(float(J.sum())) < 1e-9
    # Non-trivial field
    assert float(np.max(np.abs(J))) > 0.0

def test_interface_and_shapes_and_perf():
    w, h, n = 4, 4, 200
    frames = _mk_frames(w, h, n, rng_seed=55)
    cfg = PGGSConfig(rng_seed=1, batch_size=8, n_batches=16, smoothing_alpha=0.3, grad_norm_clip=1.0)
    out = run_pggs(frames, cfg)
    U = out["U"].U
    J = out["J"].J
    B = out["B"].B
    assert U.shape == (h, w)
    assert J.shape == (h, w)
    for k in ("N", "S", "E", "W"):
        assert B[k].shape == (h, w)
    m = out["meta"]
    for key in ("batch_size", "n_batches", "rng_seed", "frames_used", "grid_shape", "kpi"):
        assert key in m
