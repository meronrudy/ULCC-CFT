from telemetry import TelemetryEmitter, TelemetryService
from telemetry.frame import TelemetryFrame, TileMetrics, LinkMetrics, MemoryMetrics, SchedulerMetrics, PowerThermal, MemoryQueuesDepth
from typing import Dict, Any, List


class _Params:
    def __init__(self, rng_seed: int) -> None:
        self.rng_seed = rng_seed


class _MockNoC:
    """
    Minimal NoC stub sufficient for TelemetryEmitter.capture():
    - current_cycle()
    - params.rng_seed
    - tiles (list of objects with .tile_id)
    - get_counters() -> dict with mesh, router_by_tile, occupancy, power, thermal, scheduler, mc
    """

    class _Tile:
        def __init__(self, tid: int) -> None:
            self.tile_id = tid

    def __init__(self, w: int, h: int, rng_seed: int) -> None:
        self._w = w
        self._h = h
        self._cycle = 0
        self.params = _Params(rng_seed)
        # 4x4 tile ids
        self.tiles = [self._Tile(tid) for tid in range(w * h)]
        # Cumulative counters for router per tile
        self._credits_used = {tid: {0: 0, 1: 0} for tid in range(w * h)}
        self._dequeues = {tid: {0: 0, 1: 0} for tid in range(w * h)}
        self._power = {tid: {"power_inst": 0.5} for tid in range(w * h)}
        self._thermal = {tid: {"temp": 50.0} for tid in range(w * h)}
        self._mc = {}  # no MC by default

    def current_cycle(self) -> int:
        return self._cycle

    def step(self, cycles: int = 1) -> None:
        # advance counters deterministically
        for _ in range(cycles):
            self._cycle += 1
            inc = 1
            for tid in self._credits_used:
                self._credits_used[tid][0] += inc
                self._credits_used[tid][1] += inc
                self._dequeues[tid][0] += inc
                self._dequeues[tid][1] += inc

    def get_counters(self) -> Dict[str, Any]:
        occupancy: Dict[int, Dict[str, Dict[str, int]]] = {}
        for tid in range(self._w * self._h):
            # deterministic small queue depths
            occupancy[tid] = {"p0": {"vc0": 1, "vc1": 2}}
        return {
            "mesh": {"w": self._w, "h": self._h},
            "scenario_id": "scenario-fixed",
            "router_by_tile": {
                tid: {
                    "credits_used": self._credits_used[tid],
                    "dequeues": self._dequeues[tid],
                }
                for tid in range(self._w * self._h)
            },
            "occupancy": occupancy,
            "power": {"by_tile": self._power},
            "thermal": {"by_tile": self._thermal},
            "scheduler": {"run_queue_len_current": 0, "run_queue_len_max": 0, "preemptions": 0},
            "mc": {"by_tile": self._mc},
        }


def test_micro_perturbation_service_and_tagging():
    svc = TelemetryService()
    pid1 = svc.start_perturbation({"target": "tile:0", "magnitude": 0.1})
    pid2 = svc.start_perturbation({"target": "tile:1", "magnitude": 0.2})
    # Deterministic ids
    assert pid1 == 1
    assert pid2 == 2

    # List order stable and ascending by id
    lst = svc.list_perturbations()
    assert [rec["perturbation_id"] for rec in lst] == [1, 2]

    # Attach service to emitter and capture frames with tagging
    noc = _MockNoC(4, 4, rng_seed=0)
    em = TelemetryEmitter(sampling_interval_cycles=1)
    em.telemetry_service = svc

    # Capture a few cycles
    for _ in range(3):
        em.capture(noc)
        noc.step(1)

    assert len(em.frames) >= 2
    # All frames should carry smallest active id (1)
    for fr in em.frames:
        assert fr.meta.get("perturbation_id") == 1

    # Stop smallest; now smallest active should be 2
    assert svc.stop_perturbation(pid1) is True
    em.capture(noc)
    assert em.frames[-1].meta.get("perturbation_id") == 2

    # Stop remaining; tag becomes -1
    assert svc.stop_perturbation(pid2) is True
    em.capture(noc)
    assert em.frames[-1].meta.get("perturbation_id") == -1