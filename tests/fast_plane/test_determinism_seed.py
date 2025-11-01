import pytest

from fast_plane import VC, MeshShape, NoCParams, build_mesh
from fast_plane.workload import load_workload
from telemetry import TelemetryEmitter


def _mk_params(seed=123, w=4, h=4, buf=8, rlat=1, llat=1):
    return NoCParams(
        mesh=MeshShape(width=w, height=h),
        buffer_depth_per_vc=buf,
        vcs=2,
        router_pipeline_latency=rlat,
        link_latency=llat,
        link_width_bytes=16,
        flit_bytes=16,
        esc_vc_id=int(VC.ESC),
        rng_seed=int(seed),
    )


def _scenario_4x4(seed=123):
    # Minimal deterministic scenario: two producers sending to adjacent consumers, single scheduler.
    return {
        "scenario_id": "determinism_seed_smoke",
        "rng_seed": int(seed),
        "topology": {"N": 4},
        "scheduler": {"quantum": 4, "default_tile": 0},
        "consumers": [
            {"tile": (1, 0), "service_rate_flits_per_cycle": 64.0, "sink_latency_cycles": 0},
            {"tile": (2, 0), "service_rate_flits_per_cycle": 64.0, "sink_latency_cycles": 0},
        ],
        "tasks": [
            {
                "task_id": 101,
                "priority": 5,
                "affinity": (0, 0),
                "dst": (1, 0),
                "vc": "MAIN",
                "message_size_flits": 4,
                "rate_tokens_per_cycle": 1.0,
                "burst_size_flits": 16,
                "enable": True,
            },
            {
                "task_id": 202,
                "priority": 5,
                "affinity": (1, 1),
                "dst": (2, 0),
                "vc": "MAIN",
                "message_size_flits": 4,
                "rate_tokens_per_cycle": 1.0,
                "burst_size_flits": 16,
                "enable": True,
            },
        ],
    }


def test_determinism_with_fixed_seed_counters_and_telemetry():
    params = _mk_params(seed=123)
    scen = _scenario_4x4(seed=123)

    # Two independent NoC instances
    noc_a = build_mesh(params)
    noc_b = build_mesh(params)

    # Wire workload
    load_workload(noc_a, scen)
    load_workload(noc_b, scen)

    # Attach telemetry with same interval for both
    interval = 5
    em_a = TelemetryEmitter(sampling_interval_cycles=interval)
    em_b = TelemetryEmitter(sampling_interval_cycles=interval)
    noc_a.register_telemetry_emitter(em_a)
    noc_b.register_telemetry_emitter(em_b)

    # Run same number of cycles; step A one-by-one, B in chunks to exercise grouping determinism
    T = 100
    for _ in range(T):
        noc_a.step(1)
    noc_b.step(T)

    # Router/link/producer/consumer/cache/mc aggregates must match
    c_a = noc_a.get_counters()
    c_b = noc_b.get_counters()

    # Compare router aggregates and occupancy
    assert c_a["agg"] == c_b["agg"]
    assert c_a["occupancy"] == c_b["occupancy"]

    # Producer/consumer totals and by-tile presence
    assert c_a["producer"] == c_b["producer"]
    assert c_a["consumer"] == c_b["consumer"]

    # Scheduler counters identical
    assert c_a["scheduler"] == c_b["scheduler"]

    # Scenario ID and mesh
    assert c_a["scenario_id"] == c_b["scenario_id"] == "determinism_seed_smoke"
    assert c_a["mesh"] == c_b["mesh"] == {"w": 4, "h": 4}

    # Telemetry: byte-identical first K frames
    K = min(5, len(em_a.frames), len(em_b.frames))
    assert K > 0, "expected telemetry frames to be emitted"
    for i in range(K):
        ba = em_a.frames[i].to_bytes()
        bb = em_b.frames[i].to_bytes()
        assert ba == bb, f"telemetry frame {i} differs"

    # Sanity: some throughput occurred
    assert int(c_a["producer"]["produced_flits"]) > 0