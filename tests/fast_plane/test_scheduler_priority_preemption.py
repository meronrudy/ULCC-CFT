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


def _setup_preemption_case():
    """
    Build a 4x4 mesh.
    Two tasks share the same affinity tile (0,0), one low priority, one high priority.
    High priority is enabled later to force a preemption at the next quantum boundary.
    """
    params = _mk_params()
    noc = build_mesh(params)

    # Consumer at destination to drain traffic
    dst_xy = (2, 2)
    consumer = Consumer(service_rate_flits_per_cycle=32.0, sink_latency_cycles=0)
    noc.register_consumer(dst_xy, consumer)

    # Two producers on the same tile (0,0), same dst, MAIN VC
    src_xy = (0, 0)
    prod_lo = TokenBucketProducer(
        rate_tokens_per_cycle=2.0,
        burst_size_flits=16,
        message_size_flits=4,
        dst_selection=dst_xy,
        vc=VC.MAIN,
        rng_seed=0,
    )
    prod_hi = TokenBucketProducer(
        rate_tokens_per_cycle=2.0,
        burst_size_flits=16,
        message_size_flits=4,
        dst_selection=dst_xy,
        vc=VC.MAIN,
        rng_seed=0,
    )
    noc.register_producer(src_xy, prod_lo)
    noc.register_producer(src_xy, prod_hi)

    # Scheduler with quantum=4 cycles, default tile 0
    sched = Scheduler(quantum_cycles=4, default_tile_id=0)

    # Tasks: low prio enabled initially, high prio disabled initially
    t_lo = Task(task_id=1, priority=1, affinity=[src_xy], enabled=True)
    t_hi = Task(task_id=2, priority=10, affinity=[src_xy], enabled=False)
    sched.register_task(t_lo)
    sched.register_task(t_hi)

    # Bind producers to tile_id 0 (for (0,0))
    sched.bind_task_producer(1, 0, prod_lo)
    sched.bind_task_producer(2, 0, prod_hi)

    noc.register_scheduler(sched)

    return noc, sched, prod_lo, prod_hi


def test_priority_preemption_occurs_at_quantum_boundary_and_is_deterministic():
    noc, sched, prod_lo, prod_hi = _setup_preemption_case()

    # Phase 1: only low priority runs for a few quanta
    noc.step(8)  # two quanta of 4 cycles
    cnt1 = noc.get_counters()
    runmap1 = cnt1["scheduler"]["tasks_running_per_tile"]
    # On tile 0 we expect the low-priority task (id=1) to be running
    assert runmap1.get(0, None) == 1

    # Record produced flits so far by low producer
    prod_lo_before = int(noc.get_counters()["producer"]["by_tile"][0]["produced_flits"])

    # Phase 2: enable the high-priority task; preemption should occur at next boundary
    sched.set_task_enabled(2, True)
    # Advance less than a quantum; still old assignment until boundary crossed
    noc.step(2)
    runmap_mid = noc.get_counters()["scheduler"]["tasks_running_per_tile"]
    assert runmap_mid.get(0, None) == 1, "Preemption should not occur mid-quantum"

    # Cross the boundary
    noc.step(2)
    cnt2 = noc.get_counters()
    runmap2 = cnt2["scheduler"]["tasks_running_per_tile"]
    assert runmap2.get(0, None) == 2, "High-priority task should preempt at quantum boundary"

    # Scheduler counters reflect a preemption and at least one context switch
    sched_cnt = cnt2["scheduler"]
    assert int(sched_cnt["context_switches"]) >= 1
    assert int(sched_cnt["preemptions"]) >= 1

    # Producer gating: after preemption, low producer should be disabled; high enabled
    assert getattr(prod_lo, "_sched_enabled") is False
    assert getattr(prod_hi, "_sched_enabled") is True

    # Determinism: repeat from a fresh system yields identical scheduler counters after identical steps
    noc2, sched2, prod_lo2, prod_hi2 = _setup_preemption_case()
    noc2.step(8)
    sched2.set_task_enabled(2, True)
    noc2.step(2)
    noc2.step(2)
    cnt2b = noc2.get_counters()
    assert cnt2["scheduler"] == cnt2b["scheduler"]