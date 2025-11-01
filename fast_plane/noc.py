from __future__ import annotations

# E1a/E1b/E1c/E1d: Deterministic NoC mesh with ESC (DO/XY) and MAIN (weighted ECMP) VCs,
# plus an optional per-cycle scheduler (E1d) for task placement and time slicing.
# - Mesh builder wires Routers and Links with per-VC credits.
# - step(): fixed global ordering per cycle for determinism:
#       1) advance all Links
#       2) tick all Routers
#       3) per-tile in row-major:
#            a) deliver new LOCAL flits to Consumer (if present) then consumer.step()
#       4) per-tile in row-major:
#            b) call producers (in insertion order) producer.step()
#       5) per-tile in row-major:
#            c) call cache-miss processes (in insertion order) cache_proc.step()
#       6) scheduler.step() if registered (advances quanta and enables/disables producers for next window)
#   Final deterministic order per cycle:
#     links -> routers -> deliver LOCAL -> consumer.step() -> producers.step() -> cache_procs.step() -> scheduler.step()
# - inject_message(): segments into flits and enqueues at source LOCAL port.
# - try_inject_message(): like inject_message but returns False instead of asserting when insufficient headroom.
# - get_counters(): returns router counters, producer/consumer metrics, cache/mc aggregates, and scheduler stats if present.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# Optional Phase A power/thermal proxies (registered dynamically)
# Avoid hard dependency when unused.
try:
    from fast_plane.power import PowerProxy
except Exception:  # pragma: no cover - tests will import when present
    PowerProxy = None  # type: ignore
try:
    from fast_plane.thermal import ThermalRC
except Exception:  # pragma: no cover
    ThermalRC = None  # type: ignore

from fast_plane.types import (
    VC,
    Coord,
    MeshShape,
    Message,
    Flit,
    RouterWeights,
    NoCParams,
    NoCConfig,  # alias
    Dir,
)
from fast_plane.router import Router
from fast_plane.link import Link
from fast_plane.cores import TokenBucketProducer, Consumer


@dataclass
class _Tile:
    router: Router
    coord: Coord
    tile_id: int


class NoC:
    def __init__(self, params: NoCConfig, weights: Optional[Dict[int, RouterWeights]] = None) -> None:
        params.validate()
        self.params = params
        self.mesh = params.mesh
        self.tiles: List[_Tile] = []
        self.links: List[Link] = []
        # Build routers
        for y in range(self.mesh.height):
            for x in range(self.mesh.width):
                c = Coord(x, y)
                tid = self.mesh.tile_id(c)
                w = weights.get(tid) if isinstance(weights, dict) else None
                r = Router(tile_id=tid, coord=c, mesh=self.mesh, params=params, weights=w)
                self.tiles.append(_Tile(router=r, coord=c, tile_id=tid))
        # Index by tile_id
        self._rid: Dict[int, Router] = {t.tile_id: t.router for t in self.tiles}

        # Wire links for 4-neighborhood with shared link object per undirected edge
        def at(x: int, y: int) -> Optional[_Tile]:
            if 0 <= x < self.mesh.width and 0 <= y < self.mesh.height:
                tid = self.mesh.tile_id(Coord(x, y))
                return _Tile(router=self._rid[tid], coord=Coord(x, y), tile_id=tid)
            return None

        for y in range(self.mesh.height):
            for x in range(self.mesh.width):
                a = at(x, y)
                assert a is not None
                # East neighbor
                b = at(x + 1, y)
                if b is not None:
                    link = Link(self.params.link_latency, a.router, b.router, Dir.EAST)
                    self.links.append(link)
                    a.router.connect_neighbor(Dir.EAST, link)
                    b.router.connect_neighbor(Dir.WEST, link)
                    # Initialize output credits to downstream input buffer capacities (per VC)
                    cap = self.params.buffer_depth_per_vc
                    a.router.set_initial_output_credits(Dir.EAST, cap)
                    b.router.set_initial_output_credits(Dir.WEST, cap)
                # South neighbor
                b = at(x, y + 1)
                if b is not None:
                    link = Link(self.params.link_latency, a.router, b.router, Dir.SOUTH)
                    self.links.append(link)
                    a.router.connect_neighbor(Dir.SOUTH, link)
                    b.router.connect_neighbor(Dir.NORTH, link)
                    cap = self.params.buffer_depth_per_vc
                    a.router.set_initial_output_credits(Dir.SOUTH, cap)
                    b.router.set_initial_output_credits(Dir.NORTH, cap)

        # E1b/E1c: tile-attached producers, consumers, cache-miss processes, and optional memory controller
        self._producers: Dict[int, List[TokenBucketProducer]] = {t.tile_id: [] for t in self.tiles}
        self._consumer: Dict[int, Optional[Consumer]] = {t.tile_id: None for t in self.tiles}
        self._cache_procs: Dict[int, List[object]] = {t.tile_id: [] for t in self.tiles}  # CacheMissProcess instances
        self._mc: Dict[int, Optional[object]] = {t.tile_id: None for t in self.tiles}      # MemoryController instance
        # Track per-tile delivered_flits index forwarded to consumer
        self._delivered_idx: Dict[int, int] = {t.tile_id: 0 for t in self.tiles}
        # Global cycle counter (used by MC timing, etc.)
        self._cycle: int = 0
        # E1d: optional scheduler and scenario id
        self._scheduler = None  # type: Optional[object]
        self._scenario_id: Optional[str] = None

        # E1e: per-tile power/thermal model attachments
        self._power: Dict[int, Any] = {t.tile_id: None for t in self.tiles}
        self._thermal: Dict[int, Any] = {t.tile_id: None for t in self.tiles}

        # E1e: previous snapshots for per-cycle deltas (router counters)
        # - credits_used per VC => link flits sent (by VC)
        # - dequeues per VC => successful xbar transfers (incl LOCAL)
        self._prev_router: Dict[int, Dict[str, Dict[int, int]]] = {}
        for t in self.tiles:
            r = t.router
            self._prev_router[t.tile_id] = {
                "credits_used": {int(VC.ESC): int(r.counters["credits_used"][VC.ESC]),
                                 int(VC.MAIN): int(r.counters["credits_used"][VC.MAIN])},
                "dequeues": {int(VC.ESC): int(r.counters["dequeues"][VC.ESC]),
                             int(VC.MAIN): int(r.counters["dequeues"][VC.MAIN])},
            }

        # E1e: previous snapshots of producer produced_flits per tile
        self._prev_prod_flits: Dict[int, int] = {t.tile_id: 0 for t in self.tiles}

        # E1f: optional telemetry emitter
        self._telemetry_emitter: Optional[object] = None

    # --- Registration API (E1b) ---

    def _tile_id_of(self, tile_xy: Union[int, Tuple[int, int]]) -> int:
        if isinstance(tile_xy, int):
            tid = tile_xy
        else:
            x, y = tile_xy
            tid = self.mesh.tile_id(Coord(x, y))
        if tid not in self._rid:
            raise ValueError(f"tile id {tid} out of range")
        return tid

    def register_producer(self, tile_xy: Union[int, Tuple[int, int]], producer: TokenBucketProducer) -> None:
        tid = self._tile_id_of(tile_xy)
        # attach and append in insertion order
        producer._attach(self, tid)
        self._producers[tid].append(producer)

    def register_consumer(self, tile_xy: Union[int, Tuple[int, int]], consumer: Consumer) -> None:
        tid = self._tile_id_of(tile_xy)
        if self._consumer[tid] is not None:
            raise AssertionError("Only one consumer allowed per tile")
        consumer._attach(self, tid)
        self._consumer[tid] = consumer

    # --- E1c registrations ---
    def register_cache_process(self, tile_xy: Union[int, Tuple[int, int]], cache_proc: Any) -> None:
        """
        Attach a CacheMissProcess to a tile. Multiple processes per tile are allowed and
        will be stepped in insertion order after producers each cycle.
        """
        tid = self._tile_id_of(tile_xy)
        # Late binding to avoid strong typing dependency
        cache_proc._attach(self, tid)
        self._cache_procs[tid].append(cache_proc)

    def register_memory_controller(self, tile_xy: Union[int, Tuple[int, int]], mc: Any) -> None:
        """
        Attach a MemoryController at the given tile. The MC occupies the tile's consumer slot
        (receives LOCAL-delivered flits) and will be stepped during the consumer phase.
        """
        tid = self._tile_id_of(tile_xy)
        if self._consumer[tid] is not None:
            raise AssertionError("Only one consumer allowed per tile (MC occupies consumer slot)")
        mc._attach(self, tid)
        self._consumer[tid] = mc  # MC acts as the tile consumer
        self._mc[tid] = mc

    # --- E1e power/thermal registration ---
    def register_power_model(self, tile_xy: Union[int, Tuple[int, int]], power_proxy: Any) -> None:
        """
        Attach a PowerProxy to a tile. The proxy's step(activity) will be called once per cycle
        after all visible activity for the cycle is known, before thermal.
        """
        tid = self._tile_id_of(tile_xy)
        self._power[tid] = power_proxy

    def register_thermal_model(self, tile_xy: Union[int, Tuple[int, int]], thermal_rc: Any) -> None:
        """
        Attach a ThermalRC to a tile. The model's step(power_inst) will be called immediately
        after the power model step on the same tile (if present).
        """
        tid = self._tile_id_of(tile_xy)
        self._thermal[tid] = thermal_rc

    # --- E1d scheduler registration ---
    def register_scheduler(self, sched: Any) -> None:
        """
        Attach exactly one scheduler to the NoC. The scheduler will be stepped once per cycle
        after cache processes to compute the next cycle's producer enable/disable window.
        """
        if self._scheduler is not None:
            raise AssertionError("Only one scheduler can be registered")
        # Attach and allow bootstrap placement before first step
        if hasattr(sched, "_attach"):
            sched._attach(self)
        self._scheduler = sched
        if hasattr(sched, "bootstrap"):
            sched.bootstrap()

    # --- External API ---

    def current_cycle(self) -> int:
        """Return the current global cycle count."""
        return self._cycle

    def set_scenario_id(self, scenario_id: str) -> None:
        """Optional: set a scenario id for determinism tracking and introspection."""
        self._scenario_id = str(scenario_id)

    # E1g optional helper: record RNG seed for metadata/future use (no randomness introduced)
    def set_rng_seed(self, seed: int) -> None:
        """Record deterministic seed in NoC params; does not alter step ordering."""
        try:
            self.params.rng_seed = int(seed)
        except Exception:
            self.params.rng_seed = 0

    def inject_message(self, msg: Message) -> None:
        """
        Segment message into flits and enqueue at source LOCAL input.
        Asserts sufficient LOCAL capacity to hold the full message (E1a simplification).
        """
        src_router = self._rid[msg.src]
        # Check capacity on LOCAL VC for this message size
        local_port = src_router._in[Dir.LOCAL]  # internal, for tests and E1a
        vcq = local_port.vcq[msg.vc]
        if vcq.free < msg.size_flits:
            raise AssertionError("Insufficient LOCAL buffer capacity to inject message")
        for i in range(msg.size_flits):
            fl = Flit(
                msg_id=msg.msg_id,
                src=msg.src,
                dst=msg.dst,
                seq=i,
                size_flits=msg.size_flits,
                is_head=(i == 0),
                is_tail=(i == msg.size_flits - 1),
                vc=msg.vc,
            )
            ok = src_router.enqueue(Dir.LOCAL, fl)
            if not ok:
                raise AssertionError("Failed to enqueue flit on LOCAL port despite capacity check")

    def try_inject_message(self, msg: Message) -> bool:
        """
        Attempt to enqueue a full message at the source LOCAL input.
        Returns False if insufficient headroom; on success returns True.
        """
        src_router = self._rid[msg.src]
        local_port = src_router._in[Dir.LOCAL]
        vcq = local_port.vcq[msg.vc]
        if vcq.free < msg.size_flits:
            return False
        for i in range(msg.size_flits):
            fl = Flit(
                msg_id=msg.msg_id,
                src=msg.src,
                dst=msg.dst,
                seq=i,
                size_flits=msg.size_flits,
                is_head=(i == 0),
                is_tail=(i == msg.size_flits - 1),
                vc=msg.vc,
            )
            if not src_router.enqueue(Dir.LOCAL, fl):
                # Should not happen given free check; treat as failure without partial injection
                return False
        return True

    def register_telemetry_emitter(self, emitter: object) -> None:
        """
        Attach a single TelemetryEmitter (E1f). The emitter must provide capture(noc).
        """
        if self._telemetry_emitter is not None:
            raise AssertionError("Only one telemetry emitter can be registered in E1")
        # No strong typing to avoid import cycle
        if not hasattr(emitter, "capture"):
            raise ValueError("emitter must have a capture(noc) method")
        self._telemetry_emitter = emitter

    def step(self, cycles: int = 1) -> None:
        """
        Advance the entire NoC by N cycles.

        Fixed ordering per cycle for determinism (see module header):
          1) Links tick
          2) Routers tick
          3) Per-tile (row-major): deliver new LOCAL flits to Consumer then consumer.step()
          4) Per-tile (row-major): step producers in insertion order
          5) Per-tile (row-major): step cache-miss processes in insertion order
          6) Scheduler: step once if registered (enables/disables producers for next cycle window)
        """
        if cycles < 1:
            return
        for _ in range(cycles):
            # 1) Advance links first (deliveries and credits)
            for link in self.links:
                link.tick()
            # 2) Then routers: pipeline advance, route, arbitrate, xbar, send
            for t in self.tiles:
                t.router.tick()
            # 3) Consumers: deliver newly arrived LOCAL flits then step
            for t in self.tiles:
                tid = t.tile_id
                cons = self._consumer.get(tid)
                if cons is not None:
                    delivered = t.router.delivered_flits
                    last = self._delivered_idx[tid]
                    if last < len(delivered):
                        cons.on_flits_delivered(delivered[last:])
                        self._delivered_idx[tid] = len(delivered)
                    cons.step()
            # 4) Producers: per tile, in insertion order
            for t in self.tiles:
                tid = t.tile_id
                plist = self._producers.get(tid, [])
                for p in plist:
                    # E1d: allow scheduler to gate producer execution deterministically
                    if getattr(p, "_sched_enabled", True):
                        p.step()
            # 5) Cache-miss processes: per tile, in insertion order
            for t in self.tiles:
                tid = t.tile_id
                clist = self._cache_procs.get(tid, [])
                for cp in clist:
                    cp.step()
            # 6) Scheduler: advance quanta and program enable/disable for next cycle window
            if self._scheduler is not None:
                self._scheduler.step()

            # 7) Power/Thermal (E1e): after all visible activity is known
            for t in self.tiles:
                tid = t.tile_id
                r = t.router

                # Router deltas this cycle
                prev_ru = self._prev_router[tid]["credits_used"]
                prev_deq = self._prev_router[tid]["dequeues"]
                cur_ru_esc = int(r.counters["credits_used"][VC.ESC])
                cur_ru_main = int(r.counters["credits_used"][VC.MAIN])
                cur_deq_esc = int(r.counters["dequeues"][VC.ESC])
                cur_deq_main = int(r.counters["dequeues"][VC.MAIN])

                d_ru_esc = max(0, cur_ru_esc - prev_ru[int(VC.ESC)])
                d_ru_main = max(0, cur_ru_main - prev_ru[int(VC.MAIN)])
                d_deq_total = max(0, (cur_deq_esc - prev_deq[int(VC.ESC)]) + (cur_deq_main - prev_deq[int(VC.MAIN)]))

                # Update snapshots for next cycle
                self._prev_router[tid]["credits_used"][int(VC.ESC)] = cur_ru_esc
                self._prev_router[tid]["credits_used"][int(VC.MAIN)] = cur_ru_main
                self._prev_router[tid]["dequeues"][int(VC.ESC)] = cur_deq_esc
                self._prev_router[tid]["dequeues"][int(VC.MAIN)] = cur_deq_main

                # Producer injected flits delta this cycle (core issue estimate)
                prod_list = self._producers.get(tid, [])
                cur_prod_flits = 0
                for p in prod_list:
                    cur_prod_flits += int(getattr(p, "counters", {}).get("produced_flits", 0))
                d_core_issue = max(0, cur_prod_flits - self._prev_prod_flits[tid])
                self._prev_prod_flits[tid] = cur_prod_flits

                # Power model
                pw = self._power.get(tid)
                if pw is not None and hasattr(pw, "step"):
                    activity = {
                        "flits_tx_main": float(d_ru_main),
                        "flits_tx_esc": float(d_ru_esc),
                        "xbar_switches": float(d_deq_total),
                        "core_issue_est": float(d_core_issue),
                    }
                    pw.step(activity)  # type: ignore[call-arg]
                    # Thermal model consumes instantaneous power from power proxy if present
                    th = self._thermal.get(tid)
                    if th is not None and hasattr(th, "step"):
                        p_inst = float(getattr(pw, "power_inst", 0.0))
                        th.step(p_inst)  # type: ignore[call-arg]
                else:
                    # If no power model but thermal exists, feed zero power
                    th = self._thermal.get(tid)
                    if th is not None and hasattr(th, "step"):
                        th.step(0.0)  # type: ignore[call-arg]

            # 7.5) Telemetry capture after power/thermal, before cycle increments
            if self._telemetry_emitter is not None:
                try:
                    # Sampling interval handled by emitter
                    self._telemetry_emitter.capture(self)  # type: ignore[attr-defined]
                except Exception:
                    # Telemetry must not perturb NoC determinism; swallow errors in E1
                    pass

            # Advance global cycle at end of this tick
            self._cycle += 1

    def get_counters(self) -> Dict[str, object]:
        """
        Introspection for tests:
          - per-router counters aggregates and occupancy
          - producer/consumer aggregates and by-tile snapshots
          - cache/memory-controller aggregates and by-tile snapshots (E1c)
        """
        agg = {
            "enqueues": {VC.ESC: 0, VC.MAIN: 0},
            "dequeues": {VC.ESC: 0, VC.MAIN: 0},
            "credits_used": {VC.ESC: 0, VC.MAIN: 0},
            "credits_granted": {VC.ESC: 0, VC.MAIN: 0},
            "credit_underflow": {VC.ESC: 0, VC.MAIN: 0},
        }
        occ: Dict[int, Dict[str, Dict[str, int]]] = {}
        for t in self.tiles:
            r = t.router
            for k in agg.keys():
                if isinstance(r.counters.get(k), dict):
                    for vc in (VC.ESC, VC.MAIN):
                        agg[k][vc] += r.counters[k][vc]  # type: ignore
            occ[t.tile_id] = r.occupancy_snapshot()

        # Producer aggregates
        prod_totals = {"produced_messages": 0, "produced_flits": 0, "dropped_or_deferred_events": 0}
        prod_by_tile: Dict[int, Dict[str, int]] = {}
        for t in self.tiles:
            tid = t.tile_id
            totals = {"produced_messages": 0, "produced_flits": 0, "dropped_or_deferred_events": 0}
            for p in self._producers.get(tid, []):
                for k in totals.keys():
                    totals[k] += int(p.counters.get(k, 0))
            for k in prod_totals.keys():
                prod_totals[k] += totals[k]
            if any(v != 0 for v in totals.values()):
                prod_by_tile[tid] = totals

        # Consumer aggregates (non-MC consumers)
        cons_total = 0
        cons_occ: Dict[int, int] = {}
        for t in self.tiles:
            c = self._consumer.get(t.tile_id)
            # If tile has an MC, it may not expose consumer counters; guard via hasattr
            if c is not None and hasattr(c, "counters") and "consumed_flits" in getattr(c, "counters", {}):
                cons_total += int(c.counters.get("consumed_flits", 0))
                cons_occ[t.tile_id] = int(c.counters.get("queue_occupancy", 0))

        # Cache-miss process aggregates
        cache_totals = {"emitted_requests": 0, "emitted_flits": 0, "deferred_events": 0}
        cache_by_tile: Dict[int, Dict[str, int]] = {}
        for t in self.tiles:
            tid = t.tile_id
            ctot = {"emitted_requests": 0, "emitted_flits": 0, "deferred_events": 0}
            for cp in self._cache_procs.get(tid, []):
                ctot["emitted_requests"] += int(cp.counters.get("emitted_requests", 0))
                ctot["emitted_flits"] += int(cp.counters.get("emitted_flits", 0))
                ctot["deferred_events"] += int(cp.counters.get("deferred_events", 0))
            for k in cache_totals.keys():
                cache_totals[k] += ctot[k]
            if any(v != 0 for v in ctot.values()):
                cache_by_tile[tid] = ctot

        # Memory controller aggregates
        mc_by_tile: Dict[int, Dict[str, float]] = {}
        mc_totals = {
            "enqueued_requests": 0,
            "row_hits": 0,
            "row_misses": 0,
            "served_requests": 0,
            "avg_latency_cycles": 0.0,
            "p95_latency_cycles": 0.0,
            "queue_depth": 0,
        }
        latency_sum_total = 0.0
        p95_list: List[float] = []
        for t in self.tiles:
            mc = self._mc.get(t.tile_id)
            if mc is None:
                continue
            stats = mc.metrics() if hasattr(mc, "metrics") else {}
            if not stats:
                continue
            mc_by_tile[t.tile_id] = {
                "enqueued_requests": float(stats.get("enqueued_requests", 0)),
                "row_hits": float(stats.get("row_hits", 0)),
                "row_misses": float(stats.get("row_misses", 0)),
                "served_requests": float(stats.get("served_requests", 0)),
                "avg_latency_cycles": float(stats.get("avg_latency_cycles", 0.0)),
                "p95_latency_cycles": float(stats.get("p95_latency_cycles", 0.0)),
                "queue_depth": float(stats.get("queue_depth", 0)),
            }
            mc_totals["enqueued_requests"] += int(stats.get("enqueued_requests", 0))
            mc_totals["row_hits"] += int(stats.get("row_hits", 0))
            mc_totals["row_misses"] += int(stats.get("row_misses", 0))
            served = int(stats.get("served_requests", 0))
            mc_totals["served_requests"] += served
            mc_totals["queue_depth"] += int(stats.get("queue_depth", 0))
            latency_sum_total += float(stats.get("latency_sum", 0.0))
            p95_list.append(float(stats.get("p95_latency_cycles", 0.0)))
        if mc_totals["served_requests"] > 0:
            mc_totals["avg_latency_cycles"] = float(latency_sum_total / mc_totals["served_requests"])
        mc_totals["p95_latency_cycles"] = float(max(p95_list) if p95_list else 0.0)

        # Scheduler counters (if attached)
        sched_stats: Dict[str, object] = {}
        if self._scheduler is not None and hasattr(self._scheduler, "get_counters"):
            try:
                sched_stats = self._scheduler.get_counters()  # type: ignore
            except Exception:
                sched_stats = {}

        # E1e: expose minimal per-router counters by tile for tests (non-breaking additive key)
        router_by_tile: Dict[int, Dict[str, Dict[int, int]]] = {}
        for t in self.tiles:
            r = t.router
            router_by_tile[t.tile_id] = {
                "enqueues": {int(VC.ESC): int(r.counters["enqueues"][VC.ESC]),
                             int(VC.MAIN): int(r.counters["enqueues"][VC.MAIN])},
                "dequeues": {int(VC.ESC): int(r.counters["dequeues"][VC.ESC]),
                             int(VC.MAIN): int(r.counters["dequeues"][VC.MAIN])},
                "credits_used": {int(VC.ESC): int(r.counters["credits_used"][VC.ESC]),
                                 int(VC.MAIN): int(r.counters["credits_used"][VC.MAIN])},
                "credits_granted": {int(VC.ESC): int(r.counters["credits_granted"][VC.ESC]),
                                    int(VC.MAIN): int(r.counters["credits_granted"][VC.MAIN])},
                "credit_underflow": {int(VC.ESC): int(r.counters["credit_underflow"][VC.ESC]),
                                     int(VC.MAIN): int(r.counters["credit_underflow"][VC.MAIN])},
            }

        # E1e power/thermal counters (omit blocks if no models registered)
        power_by_tile: Dict[int, Dict[str, object]] = {}
        thermal_by_tile: Dict[int, Dict[str, object]] = {}
        total_energy = 0.0
        for t in self.tiles:
            tid = t.tile_id
            pw = self._power.get(tid)
            th = self._thermal.get(tid)
            if pw is not None and hasattr(pw, "energy_accum"):
                rec = {
                    "energy_accum": float(getattr(pw, "energy_accum", 0.0)),
                    "power_inst": float(getattr(pw, "power_inst", 0.0)),
                    "last_activity": dict(getattr(pw, "last_activity", {})),
                }
                power_by_tile[tid] = rec
                total_energy += float(rec["energy_accum"])
            if th is not None and hasattr(th, "temp"):
                thermal_by_tile[tid] = {
                    "temp": float(getattr(th, "temp", 0.0)),
                    "max_temp_seen": float(getattr(th, "max_temp_seen", 0.0)),
                }

        return {
            "agg": agg,
            "occupancy": occ,
            "mesh": {"w": self.mesh.width, "h": self.mesh.height},
            "producer": {"by_tile": prod_by_tile, **prod_totals},
            "consumer": {"consumed_flits": cons_total, "queue_occupancies": cons_occ},
            "cache": {"by_tile": cache_by_tile, **cache_totals},
            "mc": {"by_tile": mc_by_tile, **mc_totals},
            "scheduler": sched_stats,
            "router_by_tile": router_by_tile,
            "power": {"by_tile": power_by_tile, "total_energy": float(total_energy)} if power_by_tile else {},
            "thermal": {"by_tile": thermal_by_tile} if thermal_by_tile else {},
            "scenario_id": self._scenario_id,
        }


# Builder entry point
def build_mesh(params: NoCConfig, weights: Optional[Dict[int, RouterWeights]] = None) -> NoC:
    return NoC(params, weights)