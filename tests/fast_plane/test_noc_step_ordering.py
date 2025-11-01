import copy
import pytest

from fast_plane.types import VC, MeshShape, NoCParams, Message
from fast_plane.noc import build_mesh


def _mk_params(w=3, h=3, buf=6, rlat=1, llat=1):
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


def _inject_symmetric_messages(noc):
    mesh = noc.params.mesh
    # Inject a few deterministic messages per tile on both VCs to ensure activity
    for tid in range(mesh.width * mesh.height):
        # Mirror destination across the grid
        dst = mesh.width * mesh.height - 1 - tid
        # Keep within LOCAL buffer capacity headroom
        for k in range(2):
            noc.inject_message(Message(msg_id=1000 + tid * 8 + k, src=tid, dst=dst, vc=VC.MAIN, size_flits=2))
        for k in range(1):
            noc.inject_message(Message(msg_id=2000 + tid * 8 + k, src=tid, dst=dst, vc=VC.ESC, size_flits=1))


def _snapshot(noc):
    # Build a compact deterministic snapshot: delivered counts per tile and aggregate counters
    delivered_counts = [len(t.router.delivered_flits) for t in noc.tiles]
    agg = noc.get_counters()["agg"]
    # Convert VC-keyed dicts to tuples (esc, main) of ints for stable comparison
    def vc_tuple(d):
        return (int(d[VC.ESC]), int(d[VC.MAIN]))
    return {
        "delivered": tuple(delivered_counts),
        "enq": vc_tuple(agg["enqueues"]),
        "deq": vc_tuple(agg["dequeues"]),
        "cu": vc_tuple(agg["credits_used"]),
        "cg": vc_tuple(agg["credits_granted"]),
        "cufo": vc_tuple(agg["credit_underflow"]),
    }


def test_step_grouping_and_repeatability():
    params = _mk_params()
    # Build two identical meshes
    noc_a = build_mesh(params)
    noc_b = build_mesh(params)

    _inject_symmetric_messages(noc_a)
    _inject_symmetric_messages(noc_b)

    # Advance 20 cycles in two ways: (1) 20 individual steps, (2) one step with cycles=20
    for _ in range(20):
        noc_a.step(1)
    noc_b.step(20)

    snap_a = _snapshot(noc_a)
    snap_b = _snapshot(noc_b)

    assert snap_a == snap_b, f"Inconsistent results for step grouping: {snap_a} vs {snap_b}"

    # Determinism across rebuild with same config and injections
    noc_c = build_mesh(params)
    _inject_symmetric_messages(noc_c)
    noc_c.step(20)
    snap_c = _snapshot(noc_c)

    assert snap_b == snap_c, "Deterministic repeatability violated for identical config and injections"