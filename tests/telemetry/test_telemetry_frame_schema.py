import pytest

from fast_plane.types import VC, MeshShape, NoCParams, Message
from fast_plane.noc import build_mesh
from telemetry import TelemetryEmitter, TelemetryFrame, compute_crc32c
from fast_plane.power import PowerProxy, PowerConfig
from fast_plane.thermal import ThermalRC, ThermalConfig


def _mk_params(w=4, h=4, buf=8, rlat=1, llat=1):
    return NoCParams(
        mesh=MeshShape(width=w, height=h),
        buffer_depth_per_vc=buf,
        vcs=2,
        router_pipeline_latency=rlat,
        link_latency=llat,
        link_width_bytes=16,
        flit_bytes=16,
        esc_vc_id=int(VC.ESC),
        rng_seed=0,
    )


def _attach_power_thermal(noc):
    # Simple per-tile power and thermal models
    cfg_pwr = PowerConfig(e_flit_main=0.01, e_flit_esc=0.01, e_router_xbar=0.01, e_core_issue=0.0, sampling_window_cycles=1)
    cfg_th = ThermalConfig(r_th=100.0, c_th=1000.0, t_amb=25.0)
    for t in noc.tiles:
        noc.register_power_model(t.tile_id, PowerProxy(cfg_pwr))
        noc.register_thermal_model(t.tile_id, ThermalRC(cfg_th))


def _inject_initial_messages(noc):
    # Ensure some activity deterministically: symmetric messages like other tests
    mesh = noc.params.mesh
    for tid in range(mesh.width * mesh.height):
        dst = mesh.width * mesh.height - 1 - tid
        # Keep within LOCAL buffer capacity headroom
        noc.inject_message(Message(msg_id=1000 + tid * 4 + 0, src=tid, dst=dst, vc=VC.MAIN, size_flits=2))
        noc.inject_message(Message(msg_id=2000 + tid * 4 + 0, src=tid, dst=dst, vc=VC.ESC, size_flits=1))


def _basic_frame_checks(fr: TelemetryFrame):
    ok, err = fr.validate()
    assert ok, f"Frame failed validate: {err}"
    # Recompute CRC-32C and compare
    recomputed = compute_crc32c(fr.to_bytes())
    assert recomputed == fr.frame_crc32c, "CRC-32C mismatch on recompute"
    # Bounds and required fields
    w = fr.grid_shape["width"]
    h = fr.grid_shape["height"]
    assert w > 0 and h > 0
    # tile_ids range
    for tm in fr.tile_metrics:
        assert 0 <= tm.tile_id < (w * h)
        assert tm.flit_tx >= 0 and tm.flit_rx >= 0
        assert tm.queue_depth_avg >= 0.0 and tm.queue_depth_p99 >= 0.0
        assert -20.0 <= tm.temp_c <= 125.0
        assert 0.0 <= tm.power_pu <= 2.0
        assert all(v >= 0.0 for v in tm.vc_depth_avg)
        assert all(v >= 0.0 for v in tm.vc_depth_p99)
    # power_thermal bounds
    assert 0.0 <= fr.power_thermal.tdp_proxy_pu <= 2.0


def test_telemetry_frames_validate_and_crc_and_time_monotonic():
    params = _mk_params(4, 4, buf=8)
    noc = build_mesh(params)
    _attach_power_thermal(noc)

    # Register TelemetryEmitter with sampling each cycle
    emitter = TelemetryEmitter(sampling_interval_cycles=1)
    noc.register_telemetry_emitter(emitter)

    # Inject some initial messages to generate activity
    _inject_initial_messages(noc)

    T = 200
    noc.step(T)

    # Ensure frames captured
    assert len(emitter.frames) >= T // emitter.interval

    # Validate a subset of frames and time monotonicity per tile (by frame index, since we capture every cycle)
    prev_by_run = {}
    for idx, fr in enumerate(emitter.frames[:50]):
        _basic_frame_checks(fr)
        run_id = fr.run_id
        if run_id in prev_by_run:
            prev = prev_by_run[run_id]
            ok, err = fr.validate(previous=prev)
            assert ok, f"Window monotonicity failed: {err}"
            # time strictly increasing (start and end advance by 1 due to windowed cycle_window=1)
            assert fr.t_start_cycle == prev.t_start_cycle + 1
            assert fr.t_end_cycle == prev.t_end_cycle + 1
        prev_by_run[run_id] = fr


def test_crc_tamper_detection():
    params = _mk_params(4, 4)
    noc = build_mesh(params)
    emitter = TelemetryEmitter(sampling_interval_cycles=1)
    noc.register_telemetry_emitter(emitter)
    noc.step(3)
    assert len(emitter.frames) >= 1
    fr = emitter.frames[0]

    # Tamper: mutate a field without updating CRC
    fr.tile_metrics[0].flit_tx += 1

    ok, err = fr.validate()
    assert not ok and err == "crc_mismatch", "Tamper should be detected via CRC mismatch"


def test_determinism_first_k_frames_byte_identical():
    params = _mk_params(4, 4)
    noc_a = build_mesh(params)
    noc_b = build_mesh(params)

    em_a = TelemetryEmitter(sampling_interval_cycles=1)
    em_b = TelemetryEmitter(sampling_interval_cycles=1)
    noc_a.register_telemetry_emitter(em_a)
    noc_b.register_telemetry_emitter(em_b)

    # Identical initial injections
    _inject_initial_messages(noc_a)
    _inject_initial_messages(noc_b)

    K = 20
    noc_a.step(K)
    noc_b.step(K)

    # Compare first K frames (or available if fewer)
    n = min(len(em_a.frames), len(em_b.frames), K)
    assert n > 0
    for i in range(n):
        a_bytes = em_a.frames[i].to_bytes()
        b_bytes = em_b.frames[i].to_bytes()
        assert a_bytes == b_bytes, f"Frame bytes diverged at {i}"