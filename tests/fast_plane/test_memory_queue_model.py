import math
import pytest

from fast_plane.types import VC, MeshShape, NoCParams
from fast_plane.noc import build_mesh
from fast_plane import CacheMissProcess, MemoryController, MCConfig, Coord


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
    for t in noc.tiles:
        t.router.assert_credit_invariants()


def _run_with_mode(mode: str, T: int = 20000):
    """
    Build a 4x4 mesh, MC at (3,3), two cache processes at (0,0) and (0,3),
    and run for T cycles. Return NoC counters.
    """
    params = _mk_params(w=4, h=4, buf=8, rlat=1, llat=1)
    noc = build_mesh(params)

    # MC config per spec with FR-FCFS window
    mcfg = MCConfig(
        bank_count=4,
        channel_count=1,
        rows_per_bank=64,
        window_size=8,
        t_row_hit=10,
        t_row_miss=30,
        t_bus=2,
        mode=mode,
    )
    mc = MemoryController(mcfg)
    noc.register_memory_controller((3, 3), mc)

    # Two cache-miss processes (MAIN VC) sending to MC tile
    c0 = CacheMissProcess(mpki=8.0, ipc=1.0, message_size_flits=4, vc=VC.MAIN, mc_tile=(3, 3), enable=True)
    c1 = CacheMissProcess(mpki=8.0, ipc=1.0, message_size_flits=4, vc=VC.MAIN, mc_tile=(3, 3), enable=True)
    noc.register_cache_process((0, 0), c0)
    noc.register_cache_process((0, 3), c1)

    # Advance simulation
    for _ in range(T):
        noc.step(1)
        # Invariants are heavy; check sparsely to keep runtime reasonable
        if _ % 100 == 0:
            _assert_invariants(noc)

    return noc.get_counters()


def test_frfcfs_vs_fifo_latency_comparison():
    """
    FR-FCFS should achieve lower or equal average service latency than FIFO
    under identical workload and configuration.
    """
    cnt_fr = _run_with_mode("FRFCFS", T=20000)
    cnt_ff = _run_with_mode("FIFO", T=20000)

    fr_avg = float(cnt_fr["mc"]["avg_latency_cycles"])
    ff_avg = float(cnt_ff["mc"]["avg_latency_cycles"])

    # Basic sanity
    assert fr_avg > 0.0 and ff_avg > 0.0

    # FR-FCFS should be no worse than FIFO and typically better by a small margin.
    assert fr_avg <= ff_avg + 1e-9, f"FR-FCFS avg {fr_avg} > FIFO avg {ff_avg}"

    # Require a small but strict improvement: either at least 1 cycle, or 5% reduction.
    improved = (ff_avg - fr_avg) >= 1.0 or (fr_avg <= 0.95 * ff_avg)
    assert improved, f"Expected FR-FCFS to improve over FIFO. FR={fr_avg:.3f}, FIFO={ff_avg:.3f}"


def _run_bank_conflict(rows_per_bank: int, T: int = 15000):
    """
    Single-channel, single-bank setup to emphasize row behavior.
    One cache process driving requests to MC.
    """
    params = _mk_params(w=2, h=2, buf=8, rlat=1, llat=1)
    noc = build_mesh(params)

    mcfg = MCConfig(
        bank_count=1,
        channel_count=1,
        rows_per_bank=rows_per_bank,
        window_size=8,
        t_row_hit=6,
        t_row_miss=24,
        t_bus=2,
        mode="FRFCFS",
    )
    mc = MemoryController(mcfg)
    mc_xy = (1, 1)
    noc.register_memory_controller(mc_xy, mc)

    # One cache source aimed at MC, slightly higher rate for signal
    c = CacheMissProcess(mpki=12.0, ipc=1.0, message_size_flits=4, vc=VC.MAIN, mc_tile=mc_xy, enable=True)
    noc.register_cache_process((0, 0), c)

    for _ in range(T):
        noc.step(1)
        if _ % 200 == 0:
            _assert_invariants(noc)

    return noc.get_counters()["mc"]


def test_bank_conflict_behavior_rows_per_bank_effect():
    """
    With very small rows_per_bank, requests tend to alternate rows causing
    more row misses. Increasing rows_per_bank should reduce the miss rate.
    """
    small = _run_bank_conflict(rows_per_bank=2, T=12000)
    large = _run_bank_conflict(rows_per_bank=64, T=12000)

    sm_total = max(1, int(small["served_requests"]))
    lg_total = max(1, int(large["served_requests"]))

    sm_miss = int(small["row_misses"])
    sm_hit = int(small["row_hits"])
    lg_miss = int(large["row_misses"])
    lg_hit = int(large["row_hits"])

    # Under high alternation, misses dominate
    assert sm_miss > sm_hit, f"Expected more row misses with tiny rows_per_bank; got hits={sm_hit}, misses={sm_miss}"

    sm_miss_rate = sm_miss / sm_total
    lg_miss_rate = lg_miss / lg_total

    # Increasing rows_per_bank reduces miss rate (creates more opportunities for hits)
    assert lg_miss_rate <= sm_miss_rate + 1e-9, f"rows_per_bank increase did not reduce miss rate: small={sm_miss_rate:.3f}, large={lg_miss_rate:.3f}"


def test_invariants_and_stability_under_moderate_injection():
    """
    Ensure no deadlock and invariants hold while MC queue depth remains bounded
    under moderate injection. Queue may grow transiently but should not explode.
    """
    params = _mk_params(w=3, h=3, buf=8, rlat=1, llat=1)
    noc = build_mesh(params)

    # Moderately fast MC
    mcfg = MCConfig(
        bank_count=2,
        channel_count=1,
        rows_per_bank=32,
        window_size=8,
        t_row_hit=8,
        t_row_miss=20,
        t_bus=2,
        mode="FRFCFS",
    )
    mc = MemoryController(mcfg)
    mc_xy = (2, 2)
    noc.register_memory_controller(mc_xy, mc)

    # Two modest cache sources
    c0 = CacheMissProcess(mpki=6.0, ipc=1.0, message_size_flits=4, vc=VC.MAIN, mc_tile=mc_xy, enable=True)
    c1 = CacheMissProcess(mpki=4.0, ipc=1.0, message_size_flits=4, vc=VC.MAIN, mc_tile=mc_xy, enable=True)
    noc.register_cache_process((0, 0), c0)
    noc.register_cache_process((0, 2), c1)

    T = 6000
    for _ in range(T):
        noc.step(1)
        _assert_invariants(noc)

    mc_stats = noc.get_counters()["mc"]
    enq = int(mc_stats["enqueued_requests"])
    served = int(mc_stats["served_requests"])
    qd = int(mc_stats["queue_depth"])

    # Basic forward progress
    assert enq > 0 and served > 0 and served <= enq

    # Bounded queue depth under moderate load (heuristic bound)
    assert qd < 128, f"MC queue appears unbounded for moderate traffic: depth={qd}"