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


def test_multi_tile_affinity_round_robin_deterministic():
    """
    One task with affinity to three tiles should be scheduled in deterministic
    round-robin order across those tiles, switching only at quantum boundaries.
    Only the producer on the selected tile should be enabled within a quantum.
    """
    params = _mk_params()
    noc = build_mesh(params)

    # Drain consumer near center to avoid backpressure effects dominating
    dst_xy = (2, 2)
    consumer = Consumer(service_rate_flits_per_cycle=64.0, sink_latency_cycles=0)
    noc.register_consumer(dst_xy, consumer)

    # Three tiles in first row of a 4x4 mesh: tile_ids 0,1,2
    tiles = [(0, 0), (1, 0), (2, 0)]
    prods = []
    for xy in tiles:
        p = TokenBucketProducer(
            rate_tokens_per_cycle=1.0,
            burst_size_flits=16,
            message_size_flits=4,
            dst_selection=dst_xy,
            vc=VC.MAIN,
            rng_seed=0,
        )
        noc.register_producer(xy, p)
        # Scheduler will gate these
        prods.append(p)

    # Scheduler with small quantum to observe rotation frequently
    quantum = 3
    sched = Scheduler(quantum_cycles=quantum, default_tile_id=0)
    task = Task(task_id=7, priority=5, affinity=tiles, enabled=True)
    sched.register_task(task)

    # Bind producers by their tile ids (row-major: (0,0)->0, (1,0)->1, (2,0)->2)
    sched.bind_task_producer(7, 0, prods[0])
    sched.bind_task_producer(7, 1, prods[1])
    sched.bind_task_producer(7, 2, prods[2])

    noc.register_scheduler(sched)

    # Run for 6 quanta and record which tile hosts the task each quantum
    per_quantum_host = []
    total_cycles = quantum * 6
    for cyc in range(total_cycles):
        noc.step(1)
        # On boundaries, sample current assignment
        if (cyc + 1) % quantum == 0:
            runmap = noc.get_counters()["scheduler"]["tasks_running_per_tile"]
            # Determine which of the three tiles has the task
            host = None
            for tid in (0, 1, 2):
                if runmap.get(tid, None) == 7:
                    host = tid
                    break
            per_quantum_host.append(host)

            # Within a quantum, exactly one producer should be enabled
            enabled_flags = [getattr(p, "_sched_enabled") for p in prods]
            assert sum(1 for e in enabled_flags if e) == 1, f"Expected one enabled producer, got {enabled_flags}"

    # Expect rotation 0 -> 1 -> 2 -> 0 -> 1 -> 2
    assert per_quantum_host[:3] == [0, 1, 2], f"Unexpected rotation start: {per_quantum_host}"
    assert per_quantum_host[3:6] == [0, 1, 2], f"Unexpected rotation continuation: {per_quantum_host}"

    # Deterministic repeat
    noc2 = build_mesh(params)
    consumer2 = Consumer(service_rate_flits_per_cycle=64.0, sink_latency_cycles=0)
    noc2.register_consumer(dst_xy, consumer2)
    prods2 = []
    for xy in tiles:
        p2 = TokenBucketProducer(
            rate_tokens_per_cycle=1.0,
            burst_size_flits=16,
            message_size_flits=4,
            dst_selection=dst_xy,
            vc=VC.MAIN,
            rng_seed=0,
        )
        noc2.register_producer(xy, p2)
        prods2.append(p2)
    sched2 = Scheduler(quantum_cycles=quantum, default_tile_id=0)
    t2 = Task(task_id=7, priority=5, affinity=tiles, enabled=True)
    sched2.register_task(t2)
    sched2.bind_task_producer(7, 0, prods2[0])
    sched2.bind_task_producer(7, 1, prods2[1])
    sched2.bind_task_producer(7, 2, prods2[2])
    noc2.register_scheduler(sched2)

    per_quantum_host_2 = []
    for cyc in range(total_cycles):
        noc2.step(1)
        if (cyc + 1) % quantum == 0:
            runmap2 = noc2.get_counters()["scheduler"]["tasks_running_per_tile"]
            host2 = None
            for tid in (0, 1, 2):
                if runmap2.get(tid, None) == 7:
                    host2 = tid
                    break
            per_quantum_host_2.append(host2)

    assert per_quantum_host == per_quantum_host_2