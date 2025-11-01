import os
import math
import pathlib
import pytest

from fast_plane import VC, MeshShape, NoCParams, build_mesh
from fast_plane.workload import load_workload_from_json, load_workload
from telemetry import TelemetryEmitter
from telemetry.frame import TelemetryFrame

# Helpers

def _load_e1_scenario():
    # Load the provided 8x8 Phase A scenario (deterministic rng_seed=0)
    cfg_path = pathlib.Path("configs/phaseA_e1_small.json")
    scenario = load_workload_from_json(str(cfg_path))
    # Ensure rng_seed=0 for deterministic replay
    scenario["rng_seed"] = int(scenario.get("rng_seed", 0))
    return scenario

def _mk_params_from_scenario(scenario: dict) -> NoCParams:
    topo = scenario.get("topology", {}) or {}
    if "N" in topo:
        w = h = int(topo["N"])
    elif "mesh" in topo:
        w = int(topo["mesh"][0]); h = int(topo["mesh"][1])
    else:
        raise ValueError("topology must contain N or mesh [w,h]")
    return NoCParams(
        mesh=MeshShape(width=w, height=h),
        buffer_depth_per_vc=8,
        vcs=2,
        router_pipeline_latency=1,
        link_latency=1,
        link_width_bytes=16,
        flit_bytes=16,
        esc_vc_id=int(VC.ESC),
        rng_seed=int(scenario.get("rng_seed", 0)),
    )

def _build_noc_for_scenario(scenario: dict):
    params = _mk_params_from_scenario(scenario)
    noc = build_mesh(params)
    # Optionally persist the seed in metadata for emitters
    if hasattr(noc, "set_rng_seed"):
        noc.set_rng_seed(params.rng_seed)  # type: ignore[attr-defined]
    return noc

def _attach_telemetry(noc, interval: int = 10) -> TelemetryEmitter:
    em = TelemetryEmitter(sampling_interval_cycles=max(1, int(interval)))
    noc.register_telemetry_emitter(em)
    return em

def _assert_credit_invariants(noc) -> None:
    # Reuse invariants similar to tests/fast_plane/test_router_flow_control.py
    for t in noc.tiles:
        r = t.router
        r.assert_credit_invariants()
        for _d, op in r._out.items():
            for vc in (VC.ESC, VC.MAIN):
                assert op.credits[vc] >= 0, "Output credits must not go negative"

def _kpi_assertions(counters: dict) -> None:
    produced = int(counters.get("producer", {}).get("produced_flits", 0))
    assert produced > 0, "Expected some flits to be produced"
    mc = counters.get("mc", {}) or {}
    # If MC is present in scenario, require service > 0 and finite latency
    total_served = int(mc.get("served_requests", 0))
    if mc.get("by_tile"):
        assert total_served > 0, "Expected memory controller to serve some requests"
        avg_lat = float(mc.get("avg_latency_cycles", 0.0))
        assert math.isfinite(avg_lat), "Expected finite average latency for MC"

def _reduced_determinism_view(counters: dict) -> dict:
    # Select stable aggregates for equality comparison across runs
    keys = [
        "agg",
        "occupancy",
        "producer",
        "consumer",
        "cache",
        "mc",
        "scheduler",
        "router_by_tile",
        "mesh",
        "scenario_id",
    ]
    return {k: counters.get(k) for k in keys}

def _validate_sampled_telemetry(em: TelemetryEmitter, sample_k: int = 20) -> None:
    if not em.frames:
        # No telemetry captured is acceptable only if sampling interval skipped all cycles;
        # but for acceptance we expect some frames.
        pytest.fail("Expected telemetry frames to be emitted")
    prev_by_run = {}
    n = min(len(em.frames), max(1, int(sample_k)))
    for i in range(n):
        fr: TelemetryFrame = em.frames[i]
        ok, err = fr.validate(previous=prev_by_run.get(fr.run_id))
        assert ok, f"Telemetry frame {i} failed validate: {err}"
        prev_by_run[fr.run_id] = fr

# Tests

def test_phaseA_e1_acceptance_smoke():
    # Gating and cycle window per spec
    T_fast = int(os.getenv("E1_ACCEPT_CYCLES_FAST", "2000"))
    interval = int(os.getenv("E1_TELEMETRY_INTERVAL", "10"))

    scenario = _load_e1_scenario()

    # Build NoC and wire workload (scheduler quantum=8 is in the config)
    noc = _build_noc_for_scenario(scenario)
    load_workload(noc, scenario)

    # Attach telemetry with modest interval to limit overhead
    em = _attach_telemetry(noc, interval=interval)

    # Run for the fast acceptance window
    noc.step(T_fast)

    # No deadlock symptoms: run completed; counters show progress
    counters = noc.get_counters()
    _kpi_assertions(counters)

    # NoC-level invariants: credit invariants remain valid
    _assert_credit_invariants(noc)

    # Telemetry schema/CRC validation on a subset of frames
    _validate_sampled_telemetry(em, sample_k=20)

def test_phaseA_e1_determinism_long():
    # Long-run determinism only when explicitly requested
    T_long = int(os.getenv("E1_ACCEPT_CYCLES", "0"))
    if T_long < 5000:
        pytest.skip("Long-run determinism gated by E1_ACCEPT_CYCLES (set to >=5000, e.g., 10000)")

    interval = int(os.getenv("E1_TELEMETRY_INTERVAL", "10"))
    scenario = _load_e1_scenario()

    # Two independent NoC instances with identical config and seed
    noc_a = _build_noc_for_scenario(scenario)
    noc_b = _build_noc_for_scenario(scenario)

    load_workload(noc_a, scenario)
    load_workload(noc_b, scenario)

    em_a = _attach_telemetry(noc_a, interval=interval)
    em_b = _attach_telemetry(noc_b, interval=interval)

    # Run for the long window
    noc_a.step(T_long)
    noc_b.step(T_long)

    c_a = noc_a.get_counters()
    c_b = noc_b.get_counters()

    # Compare aggregates for equality
    assert _reduced_determinism_view(c_a) == _reduced_determinism_view(c_b), "Counters diverged between identical runs"

    # Telemetry: compare first K frames' canonical bytes for identical match
    k = min(50, len(em_a.frames), len(em_b.frames))
    assert k > 0, "Expected telemetry frames to be emitted in long-run test"
    for i in range(k):
        a_bytes = em_a.frames[i].to_bytes()
        b_bytes = em_b.frames[i].to_bytes()
        assert a_bytes == b_bytes, f"Telemetry frame bytes diverged at index {i}"