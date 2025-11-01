import math

from fast_plane.types import VC, MeshShape, NoCParams
from fast_plane.noc import build_mesh
from fast_plane.cores import TokenBucketProducer, Consumer
from fast_plane.power import PowerConfig, PowerProxy
from fast_plane.thermal import ThermalConfig, ThermalRC


def _mk_params(w=2, h=2, buf=8, rlat=0, llat=0):
    # Use zero latencies for stable cycle-by-cycle behavior
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


def test_thermal_rc_converges_to_amb_plus_rth_times_power():
    params = _mk_params()
    noc = build_mesh(params)

    # Source (0,0) sends steady MAIN flits to (0,1)
    dst_xy = (0, 1)
    cons = Consumer(service_rate_flits_per_cycle=16.0, sink_latency_cycles=0)
    noc.register_consumer(dst_xy, cons)

    prod = TokenBucketProducer(
        rate_tokens_per_cycle=1.0,  # 1 message per cycle
        burst_size_flits=64,
        message_size_flits=1,       # 1 flit per message for constant per-cycle flits
        dst_selection=dst_xy,
        vc=VC.MAIN,
        rng_seed=0,
    )
    noc.register_producer((0, 0), prod)

    # Power: only MAIN flit energy contributes; set e_flit_main = P_target (EU/cycle) for 1 flit/cycle
    P_target = 3.0
    pw_cfg = PowerConfig(
        e_flit_main=P_target,
        e_flit_esc=0.0,
        e_router_xbar=0.0,
        e_core_issue=0.0,
        sampling_window_cycles=1,
    )
    pw = PowerProxy(pw_cfg)
    noc.register_power_model((0, 0), pw)

    # Thermal RC on source tile with stable RC parameters
    r_th = 10.0     # cycles per energy
    c_th = 50.0     # energy per temp
    t_amb = 20.0
    th_cfg = ThermalConfig(r_th=r_th, c_th=c_th, t_amb=t_amb)
    th = ThermalRC(th_cfg)
    noc.register_thermal_model((0, 0), th)

    # Run long enough to approach steady-state
    T = 2000
    noc.step(T)

    # Steady-state temperature ~ t_amb + r_th * P_target
    t_ss = t_amb + r_th * P_target
    t_final = th.temp

    assert t_final >= t_amb - 1e-9
    assert abs(t_final - t_ss) <= 0.1 * t_ss + 1e-6, f"final {t_final} expected near {t_ss}"

    # Monotone-type convergence without overshoot for these parameters (weak check)
    # Since we don't record history, check that another long run doesn't change much.
    prev = t_final
    noc.step(T)
    assert th.temp >= prev - 1e-6
    assert abs(th.temp - t_ss) <= 0.1 * t_ss + 1e-6