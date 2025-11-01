import pytest

from fast_plane.types import VC, MeshShape, NoCParams, Message, Coord
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


def _assert_invariants(noc):
    # Router credit invariants must hold each cycle
    for t in noc.tiles:
        t.router.assert_credit_invariants()


def test_token_bucket_basic_deterministic_injection_and_delivery():
    """
    Configure a 4x4 mesh. One producer at (0,0) targeting (3,3), MAIN VC.
    rate=1.0 flits/cycle, message_size_flits=4, burst_size_flits >= 4.
    Run for T cycles; total produced_flits ~= rate*T within one message boundary.
    Delivered to consumer without deadlock.
    """
    params = _mk_params(w=4, h=4, buf=8, rlat=1, llat=1)
    noc = build_mesh(params)
    mesh = params.mesh

    src_xy = (0, 0)
    dst_xy = (3, 3)
    dst_tid = mesh.tile_id(Coord(dst_xy[0], dst_xy[1]))

    # Consumer at destination with high drain rate
    consumer = Consumer(service_rate_flits_per_cycle=16.0, sink_latency_cycles=0)
    noc.register_consumer(dst_xy, consumer)

    # Producer with deterministic bucket
    producer = TokenBucketProducer(
        rate_tokens_per_cycle=1.0,
        burst_size_flits=8,
        message_size_flits=4,
        dst_selection=dst_tid,
        vc=VC.MAIN,
        rng_seed=0,
    )
    noc.register_producer(src_xy, producer)

    T = 60
    for _ in range(T):
        noc.step(1)
        _assert_invariants(noc)

    counters = noc.get_counters()
    produced_flits = int(counters["producer"]["produced_flits"])
    # Within one message boundary (4 flits)
    assert abs(produced_flits - int(1.0 * T)) <= 4, f"Produced {produced_flits} vs expected ~{T}"

    consumed = int(counters["consumer"]["consumed_flits"])
    # Allow in-flight slack based on path length and per-hop pipeline:
    # slack ~= (link_latency + router_pipeline_latency) * manhattan_distance + one_message
    man = abs(dst_xy[0] - src_xy[0]) + abs(dst_xy[1] - src_xy[1])
    slack = (params.link_latency + params.router_pipeline_latency) * man + 4
    assert consumed + slack >= produced_flits, f"Consumer delivered {consumed} < produced {produced_flits} - slack({slack})"


def test_producer_defers_when_local_headroom_insufficient():
    """
    Verify no flit injection occurs when LOCAL buffer lacks headroom for a full message.
    Tokens should defer (not lost), and a deferral event is counted.
    """
    params = _mk_params(w=4, h=4, buf=8, rlat=1, llat=1)
    noc = build_mesh(params)
    mesh = params.mesh

    src_xy = (0, 0)
    src_tid = mesh.tile_id(Coord(src_xy[0], src_xy[1]))
    dst_xy = (3, 3)
    dst_tid = mesh.tile_id(Coord(dst_xy[0], dst_xy[1]))

    # High-rate producer to ensure tokens >= msg_size on first step
    producer = TokenBucketProducer(
        rate_tokens_per_cycle=10.0,  # accumulate quickly
        burst_size_flits=8,
        message_size_flits=4,
        dst_selection=dst_tid,
        vc=VC.MAIN,
        rng_seed=0,
    )
    noc.register_producer(src_xy, producer)

    # Pre-fill LOCAL input buffer at source with two messages (8 flits total, equals buffer cap here)
    noc.inject_message(Message(msg_id=10_000, src=src_tid, dst=dst_tid, vc=VC.MAIN, size_flits=4))
    noc.inject_message(Message(msg_id=10_001, src=src_tid, dst=dst_tid, vc=VC.MAIN, size_flits=4))

    # One cycle: routers will drain at most one flit from LOCAL; still insufficient headroom for a 4-flit message
    noc.step(1)
    _assert_invariants(noc)

    counters = noc.get_counters()
    prod = counters["producer"]
    produced_flits = int(prod["produced_flits"])
    deferred = int(prod["dropped_or_deferred_events"])

    # Should have deferred without injecting a new message in that cycle
    assert deferred >= 1, "Producer did not count a deferral under insufficient headroom"
    # In the immediate cycle, no new flits should have been produced by the producer itself
    assert produced_flits == 0, f"Unexpected produced flits despite full/local headroom: {produced_flits}"