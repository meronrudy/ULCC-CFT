# Telemetry package init for E1/E2 (exports per sim/DATA_CONTRACTS.md and E2 additions)
from __future__ import annotations
from typing import Dict, List, Tuple, Optional

from .frame import (
    TelemetryFrame,
    TileMetrics,
    LinkMetrics,
    MemoryMetrics,
    SchedulerMetrics,
    PowerThermal,
    Anomaly,
    crc32c,
    compute_crc32c,
    crc32c_bytes,
    MemoryQueuesDepth,
)

# E2 aggregator and CSV exports
from .aggregator import TelemetryAggregator, WindowSpec, AggregatedMetrics
from .csv import write_aggregates_csv

__all__ = [
    # Frames and schema
    "TelemetryFrame",
    "TileMetrics",
    "LinkMetrics",
    "MemoryMetrics",
    "SchedulerMetrics",
    "PowerThermal",
    "Anomaly",
    # CRC helpers
    "crc32c",
    "compute_crc32c",
    "crc32c_bytes",
    # Emitter and service
    "TelemetryEmitter",
    "TelemetryService",
    # E2 Aggregation/CSV
    "TelemetryAggregator",
    "WindowSpec",
    "AggregatedMetrics",
    "write_aggregates_csv",
]

class TelemetryService:
    """
    Minimal micro-perturbation service (E2 hooks only, no behavioral effect).
    - Deterministic id assignment starting at 1, incrementing by 1.
    - Active set maintained as an ordered dict behavior (by id).
    - current_perturbation_id():
        * returns the smallest active id if any active
        * returns -1 if none active (reserved default)
    - Specs are user-provided dicts stored verbatim.
    """

    def __init__(self) -> None:
        self._next_id: int = 1
        # id -> spec
        self._active: Dict[int, Dict] = {}

    def start_perturbation(self, spec: Dict) -> int:
        pid = self._next_id
        self._next_id += 1
        # store a shallow copy for determinism safety
        self._active[pid] = dict(spec) if isinstance(spec, dict) else {"spec": spec}
        return pid

    def stop_perturbation(self, pid: int) -> bool:
        return self._active.pop(int(pid), None) is not None

    def list_perturbations(self) -> List[Dict]:
        # Stable ascending id order
        out: List[Dict] = []
        for pid in sorted(self._active.keys()):
            rec = dict(self._active[pid])
            rec["perturbation_id"] = pid
            out.append(rec)
        return out

    def current_perturbation_id(self) -> int:
        if not self._active:
            return -1
        return min(self._active.keys())

class TelemetryEmitter:
    """
    Minimal per-cycle Telemetry emitter with CRC-32C integrity.
    - sampling_interval_cycles: capture once every N cycles (default 1).
    - frames: captured TelemetryFrame objects in order.
    - For E1, link_metrics may be empty and memory metrics minimal; fields not applicable are set to conservative defaults per DATA_CONTRACTS.
    """

    def __init__(self, sampling_interval_cycles: int = 1) -> None:
        if sampling_interval_cycles < 1:
            raise ValueError("sampling_interval_cycles must be >= 1")
        self.interval: int = int(sampling_interval_cycles)
        self.frames: List[TelemetryFrame] = []
        # Optional E2 micro-perturbation service (metadata tagging only)
        self.telemetry_service: Optional[TelemetryService] = None
        # Previous cumulative counters by tile to compute windowed deltas
        # Keys: tid -> {"credits_used": int(sum over VCs), "dequeues": int(sum over VCs)}
        self._prev_by_tile: Dict[int, Dict[str, int]] = {}
        self._last_frame_by_run: Dict[str, TelemetryFrame] = {}

    def _init_prev_if_needed(self, counters: Dict[str, object]) -> None:
        if self._prev_by_tile:
            return
        router_by_tile: Dict[int, Dict[str, Dict[int, int]]] = counters.get("router_by_tile", {})  # type: ignore
        for tid, rec in router_by_tile.items():
            cu = int(rec["credits_used"].get(0, 0)) + int(rec["credits_used"].get(1, 0))
            deq = int(rec["dequeues"].get(0, 0)) + int(rec["dequeues"].get(1, 0))
            self._prev_by_tile[int(tid)] = {"credits_used": cu, "dequeues": deq}

    def _tile_metrics_from(self, tid: int, counters: Dict[str, object]) -> TileMetrics:
        router_by_tile = counters["router_by_tile"]  # type: ignore
        occ_by_tile = counters["occupancy"]  # type: ignore
        power_by_tile = counters.get("power", {}).get("by_tile", {})  # type: ignore
        thermal_by_tile = counters.get("thermal", {}).get("by_tile", {})  # type: ignore

        rrec = router_by_tile[tid]
        cu_now = int(rrec["credits_used"].get(0, 0)) + int(rrec["credits_used"].get(1, 0))
        deq_now = int(rrec["dequeues"].get(0, 0)) + int(rrec["dequeues"].get(1, 0))
        prev = self._prev_by_tile.get(tid, {"credits_used": cu_now, "dequeues": deq_now})
        flit_tx = max(0, cu_now - int(prev["credits_used"]))
        flit_rx = max(0, deq_now - int(prev["dequeues"]))

        # Instantaneous queue depths: average across all ports/VCs
        occ = occ_by_tile.get(tid, {})
        depths: List[int] = []
        for _port, rec in occ.items():
            for _vc_name, depth in rec.items():
                try:
                    depths.append(int(depth))
                except Exception:
                    pass
        q_avg = float(sum(depths) / len(depths)) if depths else 0.0
        q_p99 = float(max(depths)) if depths else 0.0

        # VC depths: not tracked per VC in averages for E1; expose zeros of length 2
        vc_depth_avg = [0.0, 0.0]
        vc_depth_p99 = [0.0, 0.0]

        # Service time proxies: not modeled in E1; zeros
        service_time_avg = 0.0
        service_time_var = 0.0

        # Stalls proxy
        stalls = {"credit_starve": 0, "hazard": 0, "mc_block": 0}

        # MPKI / IPC proxies from producer/consumer are not exposed here; zeros
        mpki = 0.0
        ipc = 0.0

        # Power/Thermal per-tile if present
        pwr = float(power_by_tile.get(tid, {}).get("power_inst", 0.0))
        th = float(thermal_by_tile.get(tid, {}).get("temp", 0.0))

        return TileMetrics(
            tile_id=int(tid),
            flit_tx=int(flit_tx),
            flit_rx=int(flit_rx),
            vc_depth_avg=vc_depth_avg,
            vc_depth_p99=vc_depth_p99,
            queue_depth_avg=float(q_avg),
            queue_depth_p99=float(q_p99),
            service_time_avg=float(service_time_avg),
            service_time_var=float(service_time_var),
            stalls={k: int(v) for k, v in stalls.items()},
            mpki=float(mpki),
            ipc=float(ipc),
            temp_c=float(th),
            power_pu=float(pwr),
        )

    def capture(self, noc: "fast_plane.noc.NoC") -> None:  # type: ignore[name-defined]
        cycle = int(noc.current_cycle())
        if cycle % self.interval != 0:
            return

        counters = noc.get_counters()
        self._init_prev_if_needed(counters)

        # Frame header/meta
        mesh = counters["mesh"]  # type: ignore
        w = int(mesh["w"]); h = int(mesh["h"])
        grid_shape = {"width": w, "height": h}
        run_id = str(counters.get("scenario_id") or "run")
        rng_seed = int(noc.params.rng_seed)

        # Per-tile metrics
        tile_ids = [t.tile_id for t in noc.tiles]
        tmetrics: List[TileMetrics] = []
        for tid in tile_ids:
            tmetrics.append(self._tile_metrics_from(tid, counters))

        # Update prev snapshot for next window
        router_by_tile = counters["router_by_tile"]  # type: ignore
        for tid in tile_ids:
            rrec = router_by_tile[tid]
            cu_now = int(rrec["credits_used"].get(0, 0)) + int(rrec["credits_used"].get(1, 0))
            deq_now = int(rrec["dequeues"].get(0, 0)) + int(rrec["dequeues"].get(1, 0))
            self._prev_by_tile[tid] = {"credits_used": cu_now, "dequeues": deq_now}

        # Link metrics (not tracked in E1): empty
        link_metrics: List[LinkMetrics] = []

        # Memory metrics (optional in E1): generate entries for MC tiles if present in counters
        memory_metrics: List[MemoryMetrics] = []
        mc_by_tile = counters.get("mc", {}).get("by_tile", {})  # type: ignore
        for mc_tid, rec in mc_by_tile.items():
            memory_metrics.append(
                MemoryMetrics(
                    mc_id=int(mc_tid),
                    queues=MemoryQueuesDepth(rdq_depth_avg=0.0, wrq_depth_avg=0.0),
                    fr_fcfs_hit=int(rec.get("row_hits", 0)),
                    activations=int(rec.get("activations", 0)),
                    precharges=int(rec.get("precharges", 0)),
                    bandwidth_util=float(rec.get("bandwidth_util", 0.0)),
                    read_latency_avg=float(rec.get("avg_latency_cycles", 0.0)),
                    read_latency_p99=float(rec.get("p95_latency_cycles", 0.0)),
                    throttles=0,
                )
            )

        # Scheduler metrics (minimal)
        sched = counters.get("scheduler", {})  # type: ignore
        scheduler_metrics = SchedulerMetrics(
            runnable_tasks_avg=float(sched.get("run_queue_len_current", 0)),
            runnable_tasks_p99=float(sched.get("run_queue_len_max", 0)),
            migrations=0,
            preemptions=int(sched.get("preemptions", 0)),
            affinity_violations=0,
        )

        # Power/Thermal rollup
        power_by_tile = counters.get("power", {}).get("by_tile", {})  # type: ignore
        if power_by_tile:
            p_mean = 0.0
            if power_by_tile:
                total = 0.0; n = 0
                for rec in power_by_tile.values():
                    total += float(rec.get("power_inst", 0.0)); n += 1
                p_mean = float(total / max(1, n))
            power_thermal = PowerThermal(
                tdp_proxy_pu=float(p_mean),
                thermal_ceiling_hits=0,
                dvfs_state_counts={},
            )
        else:
            power_thermal = PowerThermal(tdp_proxy_pu=0.0, thermal_ceiling_hits=0, dvfs_state_counts={})

        # Determine deterministic perturbation id (E2 hook)
        pid = -1
        svc = getattr(self, "telemetry_service", None)
        if svc is not None:
            try:
                pid = int(svc.current_perturbation_id())
            except Exception:
                pid = -1

        frame = TelemetryFrame(
            schema_version="1.0.0",
            frame_id=f"F{cycle:016d}",
            run_id=run_id,
            rng_seed=rng_seed,
            grid_shape=grid_shape,
            cycle_window=1,
            t_start_cycle=cycle,
            t_end_cycle=cycle + 1,
            sampling_mode="windowed",
            tile_metrics=tmetrics,
            link_metrics=link_metrics,
            memory_metrics=memory_metrics,
            scheduler_metrics=scheduler_metrics,
            power_thermal=power_thermal,
            anomalies=[],
            meta={"scenario_id": run_id, "perturbation_id": pid},
        ).with_crc()

        # Validate CRC and simple bounds before storing
        prev = self._last_frame_by_run.get(run_id)
        ok, err = frame.validate(previous=prev)
        if not ok:
            # Mark anomaly but still store for inspection
            frame.anomalies.append(Anomaly(code="crc_mismatch", severity="error", detail=str(err)))
        self.frames.append(frame)
        self._last_frame_by_run[run_id] = frame