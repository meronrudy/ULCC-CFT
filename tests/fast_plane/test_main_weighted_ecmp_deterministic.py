import math
import pytest

from fast_plane.types import VC, MeshShape, NoCParams, Message, RouterWeights, Flit
from fast_plane.noc import build_mesh


def _mk_params(w=3, h=3, buf=8, rlat=0, llat=0):
    # Use zero latencies to simplify path-independent sampling of routing choice
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


def test_weighted_ecmp_distribution_and_determinism():
    params = _mk_params()
    mesh = params.mesh
    # Router 0 at (0,0). Destination (2,2) yields admissible next hops {EAST, SOUTH}.
    tile_id = 0
    dst = mesh.width * mesh.height - 1  # (2,2)
    weights = {
        tile_id: RouterWeights(east=3, south=1, north=0, west=0, local=0),
        # Other routers fallback to DO routing due to missing weights
    }
    noc = build_mesh(params, weights=weights)
    r0 = noc.tiles[tile_id].router

    # Sample many distinct flow identifiers (msg_id) and tally first-hop choices
    trials = 2000
    east = 0
    south = 0

    for i in range(trials):
        fl = Flit(
            msg_id=1000 + i,
            src=tile_id,
            dst=dst,
            seq=0,
            size_flits=1,
            is_head=True,
            is_tail=True,
            vc=VC.MAIN,
        )
        d = r0._route_main(fl)  # private but stable for tests in E1a
        if d.name == "EAST":
            east += 1
        elif d.name == "SOUTH":
            south += 1
        else:
            pytest.fail(f"Unexpected admissible direction {d}")

        # Deterministic: the same flit id must always map to the same next hop
        d_again = r0._route_main(fl)
        assert d_again == d

    # Distribution should reflect weights approximately 3:1 for EAST:SOUTH
    ratio = east / max(1, south)
    assert 2.0 <= ratio <= 4.0, f"Weighted ECMP ratio off: EAST={east}, SOUTH={south}, ratio={ratio:.2f}"


def test_main_fallback_to_do_when_weights_missing():
    params = _mk_params()
    noc = build_mesh(params, weights={})  # no weights for any router
    r0 = noc.tiles[0].router  # at (0,0)
    dst = noc.tiles[-1].tile_id  # (2,2)

    fl = Flit(
        msg_id=7,
        src=0,
        dst=dst,
        seq=0,
        size_flits=1,
        is_head=True,
        is_tail=True,
        vc=VC.MAIN,
    )
    # DO/XY from (0,0) to (2,2) must pick EAST first
    d = r0._route_main(fl)
    assert d.name == "EAST"