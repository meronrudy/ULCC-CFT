import copy
import pytest

from fast_plane.types import VC, MeshShape, NoCParams
from fast_plane.noc import build_mesh
from fast_plane.cores import TokenBucketProducer, Consumer


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


def _setup_system(params):
    noc = build_mesh(params)
    # Attach one consumer at far corner and one producer at origin
    src_xy = (0, 0)
    dst_xy = (params.mesh.width - 1, params.mesh.height - 1)
    dst_tid = dst_xy[1] * params.mesh.width + dst_xy[0]

    cons = Consumer(service_rate_flits_per_cycle=4.0, sink_latency_cycles=0)
    noc.register_consumer(dst_xy, cons)

    prod = TokenBucketProducer(
        rate_tokens_per_cycle=1.0,
        burst_size_flits=16,
        message_size_flits=4,
        dst_selection=dst_tid,
        vc=VC.MAIN,
        rng_seed=0,
    )
    noc.register_producer(src_xy, prod)
    return noc


def test_producer_consumer_and_router_counters_deterministic():
    """
    With identical config and fixed ordering, two runs must yield identical:
      - producer counters
      - consumer counters
      - router aggregate counters and occupancy
    """
    params = _mk_params()
    noc_a = _setup_system(params)
    noc_b = _setup_system(params)

    T = 100
    for _ in range(T):
        noc_a.step(1)
    # Step B with grouped cycles to also exercise grouping determinism
    noc_b.step(T)

    cnt_a = noc_a.get_counters()
    cnt_b = noc_b.get_counters()

    # Compare aggregate router counters
    assert cnt_a["agg"] == cnt_b["agg"], f"Router aggregates differ: {cnt_a['agg']} vs {cnt_b['agg']}"
    # Compare occupancy snapshots
    assert cnt_a["occupancy"] == cnt_b["occupancy"], "Occupancy snapshots differ"
    # Compare producer/consumer counters
    assert cnt_a["producer"] == cnt_b["producer"], f"Producer counters differ: {cnt_a['producer']} vs {cnt_b['producer']}"
    assert cnt_a["consumer"] == cnt_b["consumer"], f"Consumer counters differ: {cnt_a['consumer']} vs {cnt_b['consumer']}"