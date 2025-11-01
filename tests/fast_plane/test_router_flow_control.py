import pytest

from fast_plane.types import VC, MeshShape, NoCParams, Message
from fast_plane.noc import build_mesh


def _mk_params(w=2, h=1, buf=4, rlat=1, llat=1):
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


def _assert_no_negative_output_credits(noc):
    for t in noc.tiles:
        r = t.router
        for d, op in r._out.items():
            for vc in (VC.ESC, VC.MAIN):
                assert op.credits[vc] >= 0, "Output credits must not go negative"


def test_credit_invariant_and_no_send_without_credit():
    # Topology: 2x1 (tile 0 -- tile 1)
    params = _mk_params(w=2, h=1, buf=4, rlat=1, llat=1)
    noc = build_mesh(params)

    # Sanity: initial invariants hold
    for t in noc.tiles:
        t.router.assert_credit_invariants()

    # Inject one ESC and one MAIN message from tile 0 -> tile 1
    esc_msg = Message(msg_id=1, src=0, dst=1, vc=VC.ESC, size_flits=3)
    main_msg = Message(msg_id=2, src=0, dst=1, vc=VC.MAIN, size_flits=3)
    noc.inject_message(esc_msg)
    noc.inject_message(main_msg)

    # Step cycles until delivery is expected; check invariants each cycle
    for _ in range(12):
        noc.step(1)
        for t in noc.tiles:
            t.router.assert_credit_invariants()
        _assert_no_negative_output_credits(noc)

    # Both messages should be fully delivered (6 flits total)
    delivered = sum(len(t.router.delivered_flits) for t in noc.tiles)
    assert delivered >= esc_msg.size_flits + main_msg.size_flits

    # Output credits used must be balanced by grants over time (no underflow counted)
    agg = noc.get_counters()["agg"]
    for vc in (VC.ESC, VC.MAIN):
        # Cannot strongly assert equality due to in-flight reservations, but underflow must be zero
        assert agg["credit_underflow"][vc] == 0