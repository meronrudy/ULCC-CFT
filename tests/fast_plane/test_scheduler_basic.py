import pytest

from fast_plane import (
    VC,
    MeshShape,
    NoCParams,
    build_mesh,
    TokenBucketProducer,
    Consumer,
    Scheduler,
    Task,
)


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


def _setup_two_tasks_same_priority_distinct_tiles():
    """Builds a 4x4 mesh with two producers on distinct tiles and a scheduler."""
    params = _mk_params()
    noc = build_mesh(params)

    # Dest consumer at (2,2) drains traffic
    consumer = Consumer(service_rate_flits_per_cycle=16.0, sink_latency_cycles=0)
    noc.register_consumer((2, 2), consumer)

    # Producers on distinct tiles
    prod0 = TokenBucketProducer(
        rate_tokens_per_cycle=1.0,
        burst_size_flits=8,
        message_size_flits=4,
        dst_selection=(2, 2),
        vc=VC.MAIN,
        rng_seed=0,
    )
    prod1 = TokenBucketProducer(
        rate_tokens_per_cycle=1.0,
        burst_size_flits=8,
        message_size_flits=4,
        dst_selection=(2, 2),
        vc=VC.MAIN,
        rng_seed=0,
    )
    tile_a = (0, 0)
    tile_b = (3, 3)
    noc.register_producer(tile_a, prod0)
    noc.register_producer(tile_b, prod1)

    # Scheduler with same priority tasks, distinct affinities
    sched = Scheduler(quantum_cycles=4, default_tile_id=0)
    t0 = Task(task_id=1, priority=5, affinity=[tile_a], enabled=True)
    t1 = Task(task_id=2, priority=5, affinity=[tile_b], enabled=True)
    sched.register_task(t0)
    sched.register_task(t1)
    # Bind producers to their tiles
    sched.bind_task_producer(1, 0, prod0)  # tile_id for (0,0) is 0
    sched.bind_task_producer(2, 15, prod1)  # tile_id for (3,3) in 4x4 is 15

    # Attach scheduler
    noc.register_scheduler(sched)

    return noc, sched


def test_two_tasks_distinct_affinity_run_concurrently_and_deterministically():
    noc_a, sched_a = _setup_two_tasks_same_priority_distinct_tiles()
    noc_b, sched_b = _setup_two_tasks_same_priority_distinct_tiles()

    T = 60
    noc_a.step(T)
    noc_b.step(T)

    cnt_a = noc_a.get_counters()
    cnt_b = noc_b.get_counters()

    # Both tasks should be mapped on their tiles in the final assignment
    runmap_a = cnt_a["scheduler"]["tasks_running_per_tile"]
    runmap_b = cnt_b["scheduler"]["tasks_running_per_tile"]
    # Expect both tile 0 and 15 to be present (4x4 mesh corners)
    assert 0 in runmap_a and 15 in runmap_a, f"Unexpected running map: {runmap_a}"
    assert 0 in runmap_b and 15 in runmap_b, f"Unexpected running map: {runmap_b}"

    # Producers emitted some traffic on both tiles
    prod_by_tile_a = cnt_a["producer"]["by_tile"]
    prod_by_tile_b = cnt_b["producer"]["by_tile"]
    for tid in (0, 15):
        assert tid in prod_by_tile_a and prod_by_tile_a[tid]["produced_flits"] > 0
        assert tid in prod_by_tile_b and prod_by_tile_b[tid]["produced_flits"] > 0

    # Deterministic: full counters should match across identical runs
    assert cnt_a["agg"] == cnt_b["agg"]
    assert cnt_a["occupancy"] == cnt_b["occupancy"]
    assert cnt_a["producer"] == cnt_b["producer"]
    assert cnt_a["consumer"] == cnt_b["consumer"]
    # Scheduler counters identical as well
    assert cnt_a["scheduler"] == cnt_b["scheduler"]