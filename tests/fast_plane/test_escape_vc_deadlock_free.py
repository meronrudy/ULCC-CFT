import pytest

from fast_plane.types import VC, MeshShape, NoCParams, Message
from fast_plane.noc import build_mesh


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


def _manhattan(a_id, b_id, mesh: MeshShape):
    ax, ay = a_id % mesh.width, a_id // mesh.width
    bx, by = b_id % mesh.width, b_id // mesh.width
    return abs(ax - bx) + abs(ay - by)


def test_escape_progress_under_main_saturation():
    # 4x4 mesh with moderate buffering
    params = _mk_params(w=4, h=4, buf=6, rlat=1, llat=1)
    noc = build_mesh(params)
    mesh = params.mesh

    # Adversarial: saturate MAIN VC by filling LOCAL input buffers everywhere with many messages
    # Destination pattern: send MAIN toward the opposite corner to increase contention
    for tid in range(mesh.width * mesh.height):
        # leave src 0 (our ESC source) slightly lighter to avoid immediate LOCAL overflow in tests
        count = params.buffer_depth_per_vc if tid != 0 else params.buffer_depth_per_vc - 1
        dst = mesh.width * mesh.height - 1 - tid  # mirror index
        for k in range(count):
            noc.inject_message(Message(msg_id=10_000 + tid * 32 + k, src=tid, dst=dst, vc=VC.MAIN, size_flits=1))

    # ESC message from (0,0) to (3,3): DO/XY must guarantee progress and drain.
    src = 0
    dst = mesh.width * mesh.height - 1
    esc_flits = 3
    noc.inject_message(Message(msg_id=42, src=src, dst=dst, vc=VC.ESC, size_flits=esc_flits))

    # Bounded progress budget: <= 10x Manhattan distance + pipeline/link overhead
    man = _manhattan(src, dst, mesh)
    budget = 10 * man + 10

    delivered = 0
    cycles = 0
    while cycles <= budget and delivered < esc_flits:
        noc.step(1)
        # Count delivered esc flits at destination tile
        delivered = sum(1 for f in noc.tiles[-1].router.delivered_flits if f.msg_id == 42 and f.vc == VC.ESC)
        cycles += 1

        # Invariants should hold each cycle
        for t in noc.tiles:
            t.router.assert_credit_invariants()

    assert delivered == esc_flits, f"ESC did not drain under MAIN saturation within budget {budget} (got {delivered})"