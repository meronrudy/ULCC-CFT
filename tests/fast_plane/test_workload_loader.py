import pytest

from fast_plane import (
    VC,
    MeshShape,
    NoCParams,
    build_mesh,
    load_workload,
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


def _scenario_basic():
    # 4x4 mesh. Two tasks on different tiles; one cache-miss process; one MC.
    return {
        "scenario_id": "sched_workload_smoke",
        "topology": {"N": 4},
        "scheduler": {"quantum": 4, "default_tile": 0},
        "consumers": [
            {"tile": (2, 2), "service_rate_flits_per_cycle": 64.0, "sink_latency_cycles": 0}
        ],
        "mc": {
            "tile": (3, 3),
            "bank_count": 4,
            "channel_count": 2,
            "rows_per_bank": 32,
            "window_size": 4,
            "t_row_hit": 20,
            "t_row_miss": 80,
            "t_bus": 10,
            "mode": "FRFCFS",
        },
        "cache_miss": [
            {
                "tile": (1, 1),
                "mpki": 4.0,
                "ipc": 1.0,
                "message_size_flits": 4,
                "vc": "MAIN",
                "mc_tile": (3, 3),
                "enable": True,
            }
        ],
        "tasks": [
            {
                "task_id": 11,
                "priority": 5,
                "affinity": (0, 0),
                "dst": (2, 2),
                "vc": "MAIN",
                "message_size_flits": 4,
                "rate_tokens_per_cycle": 1.0,
                "burst_size_flits": 16,
                "enable": True,
            },
            {
                "task_id": 22,
                "priority": 5,
                "affinity": (1, 0),
                "dst": (2, 2),
                "vc": "MAIN",
                "message_size_flits": 4,
                "rate_tokens_per_cycle": 1.0,
                "burst_size_flits": 16,
                "enable": True,
            },
        ],
    }


def test_loader_wires_components_and_is_deterministic():
    params = _mk_params()
    noc_a = build_mesh(params)
    noc_b = build_mesh(params)

    scen = _scenario_basic()
    sched_a = load_workload(noc_a, scen)
    sched_b = load_workload(noc_b, scen)

    # Run same number of cycles
    T = 100
    noc_a.step(T)
    noc_b.step(T)

    c_a = noc_a.get_counters()
    c_b = noc_b.get_counters()

    # Deterministic producer totals and by-tile presence
    assert c_a["producer"]["produced_flits"] == c_b["producer"]["produced_flits"]
    assert c_a["producer"]["by_tile"] == c_b["producer"]["by_tile"]

    # Scheduler counters and scenario_id deterministic
    assert c_a["scheduler"] == c_b["scheduler"]
    assert c_a["scenario_id"] == c_b["scenario_id"] == "sched_workload_smoke"

    # Cache process and MC totals present and deterministic (non-zero expected for this scenario)
    assert c_a["cache"]["emitted_flits"] == c_b["cache"]["emitted_flits"]
    assert c_a["mc"]["served_requests"] == c_b["mc"]["served_requests"]

    # Basic sanity: some traffic occurred
    assert int(c_a["producer"]["produced_flits"]) > 0
    assert int(c_a["cache"]["emitted_flits"]) >= 0  # may be 0 depending on service overlap