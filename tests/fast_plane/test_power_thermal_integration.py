import copy

from fast_plane.types import VC, MeshShape, NoCParams
from fast_plane.noc import build_mesh
from fast_plane.cores import TokenBucketProducer, Consumer
from fast_plane.power import PowerConfig, PowerProxy
from fast_plane.thermal import ThermalConfig, ThermalRC


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


def _build_system():
    params = _mk_params()
    noc = build_mesh(params)

    # Attach consumer at (0,1) and a steady producer at (0,0) to drive MAIN traffic
    cons = Consumer(service_rate_flits_per_cycle=16.0, sink_latency_cycles=0)
    noc.register_consumer((0, 1), cons)

    prod = TokenBucketProducer(
        rate_tokens_per_cycle=1.0,
        burst_size_flits=64,
        message_size_flits=1,
        dst_selection=(0, 1),
        vc=VC.MAIN,
        rng_seed=0,
    )
    noc.register_producer((0, 0), prod)

    # Power/Thermal on (0,0) (source) and (0,1) (first hop) and a far idle tile (3,3)
    pw_cfg = PowerConfig(
        e_flit_main=1.5,
        e_flit_esc=0.5,
        e_router_xbar=0.25,
        e_core_issue=0.0,
        sampling_window_cycles=1,
    )
    th_cfg = ThermalConfig(r_th=12.0, c_th=60.0, t_amb=25.0, t_max=None)

    pw_src = PowerProxy(pw_cfg)
    th_src = ThermalRC(th_cfg)
    noc.register_power_model((0, 0), pw_src)
    noc.register_thermal_model((0, 0), th_src)

    pw_nbr = PowerProxy(pw_cfg)
    th_nbr = ThermalRC(th_cfg)
    noc.register_power_model((0, 1), pw_nbr)
    noc.register_thermal_model((0, 1), th_nbr)

    # Far idle tile: also attach models to assert non-negativity and boundedness vs ambient
    pw_idle = PowerProxy(pw_cfg)
    th_idle = ThermalRC(th_cfg)
    noc.register_power_model((3, 3), pw_idle)
    noc.register_thermal_model((3, 3), th_idle)

    return noc, ((0, 0), (0, 1), (3, 3))


def test_power_thermal_determinism_and_bounds_and_relative_behavior():
    noc_a, tiles_a = _build_system()
    noc_b, tiles_b = _build_system()
    assert tiles_a == tiles_b

    T = 300
    # Two identical runs, but step grouping different to stress determinism
    for _ in range(T):
        noc_a.step(1)
    noc_b.step(T)

    cnt_a = noc_a.get_counters()
    cnt_b = noc_b.get_counters()

    # Determinism: energy/temp trajectories final state equal
    def read_state(cnt, xy):
        tid = cnt["mesh"]["w"] * xy[1] + xy[0]
        pw_by = cnt.get("power", {}).get("by_tile", {})
        th_by = cnt.get("thermal", {}).get("by_tile", {})
        # Missing records should appear only if models weren't registered; we registered them.
        p = pw_by[tid]["power_inst"]
        e = pw_by[tid]["energy_accum"]
        t = th_by[tid]["temp"]
        tmax = th_by[tid]["max_temp_seen"]
        return (float(p), float(e), float(t), float(tmax))

    src_xy, nbr_xy, idle_xy = tiles_a
    a_src = read_state(cnt_a, src_xy)
    b_src = read_state(cnt_b, src_xy)
    a_nbr = read_state(cnt_a, nbr_xy)
    b_nbr = read_state(cnt_b, nbr_xy)
    a_idle = read_state(cnt_a, idle_xy)
    b_idle = read_state(cnt_b, idle_xy)

    assert a_src == b_src
    assert a_nbr == b_nbr
    assert a_idle == b_idle

    # Non-negativity and ambient bound (temps >= ambient)
    t_amb = 25.0
    for st in (a_src, a_nbr, a_idle):
        _, _, t, tmax = st
        assert t >= t_amb - 1e-9
        assert tmax >= t - 1e-9

    # Relative behavior: tiles on hot path (src or neighbor) hotter than far idle
    assert a_src[2] >= a_idle[2] - 1e-9
    assert a_nbr[2] >= a_idle[2] - 1e-9

    # Also compare energy totals vs idle
    assert a_src[1] >= a_idle[1] - 1e-9
    assert a_nbr[1] >= a_idle[1] - 1e-9