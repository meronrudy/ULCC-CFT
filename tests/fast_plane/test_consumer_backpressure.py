import pytest

from fast_plane.types import VC, Coord, MeshShape, NoCParams, Message
from fast_plane.noc import build_mesh
from fast_plane.cores import TokenBucketProducer, Consumer


def _mk_params(w=2, h=1, buf=6, rlat=1, llat=1):
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


def _assert_invariants(noc):
    for t in noc.tiles:
        t.router.assert_credit_invariants()


def test_consumer_slow_service_and_producer_deferral_stability():
    """
    Consumer with service_rate < inbound rate creates sustained sink backlog.
    Ensure:
      - No credit invariant violations
      - Producer observes deferrals (LOCAL headroom gating)
      - Consumer queue grows beyond zero (backlog)
    """
    params = _mk_params(w=2, h=1, buf=6, rlat=1, llat=1)
    noc = build_mesh(params)

    src_xy = (0, 0)
    dst_xy = (1, 0)
    src_tid = params.mesh.tile_id(Coord(*src_xy))
    dst_tid = params.mesh.tile_id(Coord(*dst_xy))

    # Slow consumer at destination (drain slower than potential inbound)
    cons = Consumer(service_rate_flits_per_cycle=0.25, sink_latency_cycles=0)
    noc.register_consumer(dst_xy, cons)

    # Aggressive producer to create pressure; message size 4 flits
    prod = TokenBucketProducer(
        rate_tokens_per_cycle=2.0,   # 2 flits/cycle potential
        burst_size_flits=8,
        message_size_flits=4,
        dst_selection=dst_tid,
        vc=VC.MAIN,
        rng_seed=0,
    )
    noc.register_producer(src_xy, prod)

    # Prefill source LOCAL buffer to hasten deferral
    noc.inject_message(Message(msg_id=7000, src=src_tid, dst=dst_tid, vc=VC.MAIN, size_flits=4))

    T = 100
    for _ in range(T):
        noc.step(1)
        _assert_invariants(noc)

    counters = noc.get_counters()
    # Producer must have experienced deferrals at some point
    assert int(counters["producer"]["dropped_or_deferred_events"]) >= 1

    # Consumer backlog should be positive (slow drain)
    qocc = counters["consumer"]["queue_occupancies"].get(dst_tid, 0)
    assert qocc > 0

    # Safety: no negative credits counted anywhere (sanity already via invariants)
    agg = counters["agg"]
    for vc in (VC.ESC, VC.MAIN):
        assert agg["credit_underflow"][vc] == 0