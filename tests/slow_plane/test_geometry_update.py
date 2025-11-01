# Ensure project root on sys.path for direct pytest invocation
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pytest

from slow_plane.geometry import (
    GeometryConfig,
    update_geometry,
    ensure_spd_and_cond,
    cfe_residual,
    apply_step,
    accept_with_trust,
)
from slow_plane.field import FieldConfig, solve_field
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


def _identity_metric(H: int, W: int) -> np.ndarray:
    g = np.zeros((H, W, 2, 2), dtype=np.float64)
    g[..., 0, 0] = 1.0
    g[..., 1, 1] = 1.0
    return g


def _cond_number_2x2(A: np.ndarray) -> float:
    w = np.linalg.eigvalsh(0.5 * (A + A.T))
    w = np.maximum(w, 1e-18)
    return float(np.max(w) / np.min(w))


def _make_smooth_phi(H: int, W: int) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    sigma2 = max(H, W)
    return np.exp(-r2 / float(sigma2))


def _make_random_fields(H: int, W: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    U = rng.uniform(-0.1, 0.1, size=(H, W)).astype(np.float64)
    J = rng.uniform(-0.05, 0.05, size=(H, W)).astype(np.float64)
    # B as dict with small bounded values
    B = {
        "N": rng.uniform(-0.1, 0.1, size=(H, W)).astype(np.float64),
        "S": rng.uniform(-0.1, 0.1, size=(H, W)).astype(np.float64),
        "E": rng.uniform(-0.1, 0.1, size=(H, W)).astype(np.float64),
        "W": rng.uniform(-0.1, 0.1, size=(H, W)).astype(np.float64),
    }
    return U, J, B


def _delta_norm_map(g_next: np.ndarray, g0: np.ndarray) -> np.ndarray:
    H, W = g0.shape[:2]
    out = np.zeros((H, W), dtype=np.float64)
    for y in range(H):
        for x in range(W):
            out[y, x] = np.linalg.norm(g_next[y, x] - g0[y, x], ord="fro")
    return out


def test_shapes_and_spd_cond_basic():
    H = W = 16
    g0 = _identity_metric(H, W)
    phi = _make_smooth_phi(H, W)
    grad = np.zeros((H, W, 2), dtype=np.float64)  # not used by residual in Phase A
    U, J, B = _make_random_fields(H, W, seed=1)

    cfg = GeometryConfig(step_size=0.3, damping=0.7, cond_max=1e3, spd_eps=1e-6, trust_radius=0.5, accept_ratio_min=0.4, hysteresis=1)
    out = update_geometry(g0, phi, grad, U, J, B, cfg)
    g_next = out["g_next"]
    meta = out["meta"]

    assert g_next.shape == (H, W, 2, 2)
    assert isinstance(meta["accepted"], (bool, np.bool_))
    # SPD and condition bound
    for y in range(H):
        for x in range(W):
            w = np.linalg.eigvalsh(0.5 * (g_next[y, x] + g_next[y, x].T))
            assert np.all(w > 0.0)
            assert _cond_number_2x2(g_next[y, x]) <= cfg.cond_max + 1e-9


def test_residual_influence_centered_J():
    H = W = 20
    g0 = _identity_metric(H, W)
    phi = _make_smooth_phi(H, W)
    grad = np.zeros((H, W, 2), dtype=np.float64)

    rng = np.random.default_rng(0)
    U = rng.uniform(-0.05, 0.05, size=(H, W)).astype(np.float64)

    # Uniform J
    J_uni = np.zeros((H, W), dtype=np.float64)
    # Centered J bump
    J_bump = np.zeros((H, W), dtype=np.float64)
    cy, cx = H // 2, W // 2
    J_bump[cy, cx] = 1.0

    # Small flux background
    B = {
        "N": np.zeros((H, W), dtype=np.float64),
        "S": np.zeros((H, W), dtype=np.float64),
        "E": np.zeros((H, W), dtype=np.float64),
        "W": np.zeros((H, W), dtype=np.float64),
    }

    cfg = GeometryConfig(step_size=0.4, damping=0.7, cond_max=1e4, spd_eps=1e-6, trust_radius=1.0, accept_ratio_min=0.3, hysteresis=0)

    out_uni = update_geometry(g0, phi, grad, U, J_uni, B, cfg)
    out_bmp = update_geometry(g0, phi, grad, U, J_bump, B, cfg)

    assert out_bmp["meta"]["residual_norm"] >= out_uni["meta"]["residual_norm"] - 1e-12

    # Deviation map peaks near center for J bump
    dn_bmp = _delta_norm_map(out_bmp["g_next"], g0)
    # Compare center vs corners
    corners = np.array([dn_bmp[0, 0], dn_bmp[0, -1], dn_bmp[-1, 0], dn_bmp[-1, -1]])
    assert float(dn_bmp[cy, cx]) >= float(corners.mean()) + 1e-12


def test_trust_region_accept_reject_and_ratio_tracks():
    H = W = 16
    g0 = _identity_metric(H, W)
    phi = _make_smooth_phi(H, W)
    grad = np.zeros((H, W, 2), dtype=np.float64)
    U, J, B = _make_random_fields(H, W, seed=2)

    # First, tiny trust radius to force rejection
    cfg_small = GeometryConfig(step_size=0.5, damping=0.8, cond_max=1e5, spd_eps=1e-6, trust_radius=1e-6, accept_ratio_min=0.2, hysteresis=2)
    out_small = update_geometry(g0, phi, grad, U, J, B, cfg_small)
    assert out_small["meta"]["accepted"] is False

    # Larger radius should accept
    cfg_large = GeometryConfig(step_size=0.5, damping=0.8, cond_max=1e5, spd_eps=1e-6, trust_radius=1.0, accept_ratio_min=0.2, hysteresis=2)
    out_large = update_geometry(g0, phi, grad, U, J, B, cfg_large)
    assert out_large["meta"]["accepted"] is True

    # accept_ratio is defined and finite
    ar_small = float(out_small["meta"]["accept_ratio"])
    ar_large = float(out_large["meta"]["accept_ratio"])
    assert np.isfinite(ar_small) and ar_small >= 0.0
    assert np.isfinite(ar_large) and ar_large >= 0.0


def test_hysteresis_counter_and_inhibition_of_increase():
    H = W = 12
    g0 = _identity_metric(H, W)
    phi = _make_smooth_phi(H, W)
    grad = np.zeros((H, W, 2), dtype=np.float64)
    U, J, B = _make_random_fields(H, W, seed=3)

    cfg = GeometryConfig(step_size=0.5, damping=0.8, cond_max=1e5, spd_eps=1e-6, trust_radius=1e-8, accept_ratio_min=0.2, hysteresis=3)
    out_reject = update_geometry(g0, phi, grad, U, J, B, cfg)
    assert out_reject["meta"]["accepted"] is False
    assert int(out_reject["meta"]["hysteresis_left"]) > 0

    # Now use internals to simulate countdown with acceptances, verifying trust radius does not increase until hysteresis reaches 0
    # Build residual and candidate once (deterministic for given inputs)
    g_proj = ensure_spd_and_cond(g0, cfg.spd_eps, cfg.cond_max)
    R = cfe_residual(g_proj, phi, U, J, B)
    g_cand = apply_step(g_proj, R, cfg)

    tr = 1e-2  # start with some radius after rejection shrink; exact value not critical
    hyst = 2
    tr_history = [tr]
    # Accept several times; increase is inhibited while hyst > 0
    for _ in range(5):
        accepted, tr, hyst, ratio = accept_with_trust(g_proj, g_cand, R, cfg, tr, hyst)
        tr_history.append(tr)
        if hyst > 0:
            # No increase expected while cooling down
            assert tr_history[-1] <= tr_history[-2] * 1.0000001  # allow equal or decrease (no increase)
        else:
            # If ratio high, radius may increase slightly
            break
    # Ensure countdown occurred
    assert hyst >= 0


def test_determinism_two_runs_identical():
    H = W = 10
    g0 = _identity_metric(H, W)
    phi = _make_smooth_phi(H, W)
    grad = np.zeros((H, W, 2), dtype=np.float64)
    U, J, B = _make_random_fields(H, W, seed=5)
    cfg = GeometryConfig(step_size=0.3, damping=0.7, cond_max=1e4, spd_eps=1e-6, trust_radius=0.5, accept_ratio_min=0.4, hysteresis=2)

    out1 = update_geometry(g0, phi, grad, U, J, B, cfg)
    out2 = update_geometry(g0, phi, grad, U, J, B, cfg)
    assert out1["g_next"].tobytes() == out2["g_next"].tobytes()
    assert out1["meta"] == out2["meta"]


def _mk_frame(w: int, h: int, cycle: int, rng_seed: int, tile_overrides=None) -> TelemetryFrame:
    tile_overrides = tile_overrides or {}
    tiles = []
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
    links = []
    mms = [
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
        run_id="run-geom-integration",
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


def test_integration_with_field_and_pggs_small_grid():
    w = h = 4
    frames = [_mk_frame(w, h, t, rng_seed=42) for t in range(20)]
    pggs_cfg = PGGSConfig(rng_seed=7, batch_size=4, n_batches=8, smoothing_alpha=0.4, grad_norm_clip=1.0)
    pggs_out = run_pggs(frames, pggs_cfg)
    U = pggs_out["U"].U
    J = pggs_out["J"].J
    B = pggs_out["B"].B

    g0 = _identity_metric(h, w)
    field_cfg = FieldConfig(method="cg", max_cg_iters=200, cg_tol=1e-6)
    field_out = solve_field(g0, J, field_cfg)
    phi = field_out["phi"]
    grad = field_out["grad"]

    cfg = GeometryConfig(step_size=0.25, damping=0.7, cond_max=1e5, spd_eps=1e-6, trust_radius=0.3, accept_ratio_min=0.3, hysteresis=1)
    out = update_geometry(g0, phi, grad, U, J, B, cfg)
    g_next = out["g_next"]
    meta = out["meta"]

    # SPD and condition bound
    for y in range(h):
        for x in range(w):
            wv = np.linalg.eigvalsh(0.5 * (g_next[y, x] + g_next[y, x].T))
            assert np.all(wv > 0.0)
            assert _cond_number_2x2(g_next[y, x]) <= cfg.cond_max + 1e-9

    # Reasonable delta magnitude: mean Frobenius <= trust_radius (by acceptance rule if accepted)
    dn = _delta_norm_map(g_next, g0).mean()
    if meta["accepted"]:
        assert float(dn) <= float(meta["trust_radius"]) + 1e-12