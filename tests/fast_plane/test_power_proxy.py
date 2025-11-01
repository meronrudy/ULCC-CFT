import math

from fast_plane.types import VC, MeshShape, NoCParams
from fast_plane.noc import build_mesh
from fast_plane.cores import TokenBucketProducer, Consumer
from fast_plane.power import PowerConfig, PowerProxy


def _mk_params(w=4, h=4, buf=8, rlat=0, llat=0):
    # Zero latencies to maximize per-cycle link sends deterministically for this test
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


def test_power_energy_accumulates_from_main_flits_only_when_router_xbar_and_core_issue_zero():
    params = _mk_params()
    noc = build_mesh(params)

    # Attach consumer at (0,1) to drain
    cons = Consumer(service_rate_flits_per_cycle=16.0, sink_latency_cycles=0)
    noc.register_consumer((0, 1), cons)

    # Producer at (0,0) -> (0,1), MAIN VC, one flit per cycle (msg_size=1 makes tokenized steady stream)
    prod = TokenBucketProducer(
        rate_tokens_per_cycle=1.0,
        burst_size_flits=16,
        message_size_flits=1,
        dst_selection=(0, 1),
        vc=VC.MAIN,
        rng_seed=0,
    )
    noc.register_producer((0, 0), prod)

    # Power proxy on source tile; isolate link flit energy
    cfg = PowerConfig(
        e_flit_main=2.0,   # energy per MAIN flit
        e_flit_esc=0.0,
        e_router_xbar=0.0,
        e_core_issue=0.0,
        sampling_window_cycles=1,
    )
    pw = PowerProxy(cfg)
    src_tid = noc.mesh.tile_id(noc.mesh.coord_of(0))  # tid of (0,0) is 0
    noc.register_power_model((0, 0), pw)

    # Sanity: zero traffic -> zero energy
    before = pw.energy_accum
    assert before == 0.0

    T = 200
    noc.step(T)

    # Expected energy = e_flit_main * total MAIN credits_used at the source router
    cnt = noc.get_counters()
    # Router counters by tile added as non-breaking additive key
    rbt = cnt["router_by_tile"]
    main_credits_used = int(rbt[src_tid]["credits_used"][int(VC.MAIN)])
    expected_energy = cfg.e_flit_main * main_credits_used

    assert pw.energy_accum == expected_energy, f"energy_accum={pw.energy_accum} expected={expected_energy}"

    # Additional sanity: with zero additional steps, energy should not change
    e_after = pw.energy_accum
    noc.step(0)
    assert pw.energy_accum == e_after