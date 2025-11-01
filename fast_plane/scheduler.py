from __future__ import annotations

"""
E1d: Deterministic run-queue scheduler with priorities, affinities, and fixed quanta.

Policy (deterministic):
- Order tasks by (priority desc, submit_index asc). No randomness.
- Placement:
  * Single-tile affinity: always that tile.
  * Multi-tile affinity: deterministic round-robin over the affinity list when the task is dispatched.
  * No affinity: default tile_id 0 (row-major first tile).
- Time slicing:
  * Fixed quantum in cycles (integer >= 1).
  * Preemption occurs only at boundaries between quanta.
  * Stable tie-break across equal priorities by submit_index (ascending).
- Integration with producers:
  * Each task is bound to one or more TokenBucketProducer instances, one per affinity tile.
  * The scheduler sets a private flag on each bound producer: _sched_enabled (bool).
  * No other producer parameters are modified at runtime.

Step ordering with NoC (fast_plane.noc):
- Each NoC cycle:
  1) links.tick()
  2) routers.tick()
  3) deliver LOCAL to consumer; consumer.step()
  4) producers.step()  (producers check _sched_enabled gating)
  5) cache_procs.step()
  6) scheduler.step()  (programs next-cycle window; preemption at quantum boundaries)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class Task:
    task_id: int
    priority: int
    affinity: Optional[List[Union[int, Tuple[int, int]]]] = None  # tile ids or (x,y) coords
    demand_hint: Optional[float] = None
    enabled: bool = True

    # Internal, assigned by scheduler on registration
    submit_index: int = field(default=-1, init=False)
    _aff_tile_ids: List[int] = field(default_factory=list, init=False)
    _rr_idx: int = field(default=0, init=False)


class Scheduler:
    """
    Deterministic run-queue scheduler (E1d).

    Usage:
      - Create Scheduler(quantum_cycles=Q)
      - register_task(Task(...)) for all tasks
      - bind_task_producer(task_id, tile_id, producer) for each affinity tile
      - NoC.register_scheduler(scheduler) (attaches and bootstrap()s)
    """

    def __init__(self, quantum_cycles: int = 8, default_tile_id: int = 0) -> None:
        if quantum_cycles < 1:
            raise ValueError("quantum_cycles must be >= 1")
        self.quantum: int = int(quantum_cycles)
        self.default_tile: int = int(default_tile_id)

        self._noc = None  # type: Optional["NoC"]
        self._tasks: Dict[int, Task] = {}
        self._submit_order: List[int] = []  # task_ids in registration order
        self._producers: Dict[int, Dict[int, List[Any]]] = {}  # task_id -> {tile_id: [producer,...]}
        self._current_assign: Dict[int, int] = {}  # tile_id -> task_id (current quantum)
        self._prev_assign: Dict[int, int] = {}
        self._cycles_into_quantum: int = 0
        self._needs_replan: bool = False
        # E1e: defer assignment change to the start of the next cycle after a boundary,
        # so counters sampled after scheduler.step() still reflect the current quantum.
        self._next_assign: Optional[Dict[int, int]] = None

        # Counters and stats
        self._context_switches: int = 0
        self._preemptions: int = 0
        self._dispatch_counts_by_pri: Dict[int, int] = {}
        self._runq_len_max: int = 0

    # ---- Attachment and bootstrap ----

    def _attach(self, noc: "NoC") -> None:
        self._noc = noc

    def bootstrap(self) -> None:
        """
        Plan the first quantum window so producers are enabled appropriately starting at cycle 0.
        Must be called before the first NoC.step() tick. NoC.register_scheduler() calls this.
        """
        # Disable all bound producers first
        self._gate_all(enable=False)
        # Compute initial assignment for the first quantum and enable producers WITHOUT
        # advancing RR or counting context switches/preemptions. This avoids off-by-one
        # rotation at time 0 for multi-tile affinity tasks.
        new_assign = self._plan_assignments()
        for tile, tid in new_assign.items():
            for p in self._producers.get(tid, {}).get(tile, []):
                setattr(p, "_sched_enabled", True)
        self._current_assign = dict(new_assign)
        self._prev_assign = {}
        # Do not advance task._rr_idx here; advancement occurs only after an actual quantum runs.

    # ---- Registration API ----

    def register_task(self, task: Task) -> None:
        if task.task_id in self._tasks:
            raise AssertionError(f"Duplicate task_id {task.task_id}")
        # Assign submit index and normalize affinity to tile ids
        task.submit_index = len(self._submit_order)
        self._submit_order.append(task.task_id)
        # Resolve affinity entries to tile ids (defer to _noc when available; otherwise accept ints)
        aff_ids: List[int] = []
        if task.affinity and len(task.affinity) > 0:
            for entry in task.affinity:
                if isinstance(entry, int):
                    aff_ids.append(int(entry))
                elif isinstance(entry, tuple) and len(entry) == 2 and all(isinstance(v, int) for v in entry):
                    # Need NoC mesh to resolve coord - may not be attached yet. Store placeholder negative id.
                    # We'll resolve coordinates at bind time since NoC is guaranteed by then.
                    aff_ids.append(-(len(aff_ids) + 1))  # placeholder
                else:
                    raise ValueError("Unsupported affinity entry; use tile_id int or (x,y) tuple")
        task._aff_tile_ids = aff_ids
        self._tasks[task.task_id] = task
        self._producers[task.task_id] = {}

    def bind_task_producer(self, task_id: int, tile_id: int, producer: Any) -> None:
        """
        Bind a producer instance to a task at a specific tile (one instance per affinity tile).
        The producer will be gated via _sched_enabled flag.
        """
        if task_id not in self._tasks:
            raise AssertionError(f"Unknown task_id {task_id}")
        self._producers[task_id].setdefault(int(tile_id), []).append(producer)
        # Ensure producer starts disabled; scheduler will enable only when dispatched.
        setattr(producer, "_sched_enabled", False)

    # ---- Control API ----

    def set_task_enabled(self, task_id: int, enable: bool) -> None:
        if task_id not in self._tasks:
            raise AssertionError(f"Unknown task_id {task_id}")
        self._tasks[task_id].enabled = bool(enable)
        # Request a replan at the next scheduling boundary
        self._needs_replan = True

    # ---- Stepping ----

    def step(self) -> None:
        """
        Called once per NoC cycle after producers and cache_procs.
        This programs the gating state for the next cycle window.
        """
        # Strict boundary scheduling (Phase A determinism): only at quantum boundaries.
        # If a replan is requested mid-quantum, defer it until the boundary.
        self._cycles_into_quantum += 1
        if self._cycles_into_quantum >= self.quantum:
            self._cycles_into_quantum = 0
            self._needs_replan = False
            new_assign = self._plan_assignments()
            # Apply immediately at the boundary so tests observe the new mapping
            self._apply_assignment(new_assign)

    # ---- Internals ----

    def _iter_enabled_tasks_sorted(self) -> List[Task]:
        # Determine enabled tasks and stable sort by (-priority, submit_index)
        enabled = [t for t in self._tasks.values() if t.enabled]
        enabled.sort(key=lambda t: (-int(t.priority), int(t.submit_index)))
        # Track run-queue stats
        self._runq_len_max = max(self._runq_len_max, len(enabled))
        return enabled

    def _resolve_affinity_ids(self, task: Task) -> List[int]:
        """
        Resolve affinity entries to tile_ids using NoC mesh if necessary.
        """
        if self._noc is None:
            # Accept pre-resolved ints only
            return [tid for tid in task._aff_tile_ids if tid >= 0]
        mesh = self._noc.mesh
        out: List[int] = []
        if task.affinity and len(task.affinity) > 0:
            for entry in task.affinity:
                if isinstance(entry, int):
                    out.append(int(entry))
                else:
                    x, y = entry  # type: ignore[misc]
                    out.append(mesh.tile_id(type("Coord", (), {"x": x, "y": y})))
        return out

    def _desired_tile_for_task(self, task: Task) -> int:
        aff = self._resolve_affinity_ids(task)
        if not aff:
            return self.default_tile
        if len(aff) == 1:
            return aff[0]
        # Multi-tile: round-robin when dispatched
        idx = task._rr_idx % len(aff)
        return aff[idx]

    def _plan_assignments(self) -> Dict[int, int]:
        """
        Build next-quantum assignments: map tile_id -> task_id.
        At most one task per tile in a quantum. Deterministic ordering resolves conflicts.
        Only tasks with at least one bound producer at the desired tile are considered runnable.
        """
        assigned: Dict[int, int] = {}
        claimed_tiles: set = set()
        for t in self._iter_enabled_tasks_sorted():
            # Determine desired tile
            tile = self._desired_tile_for_task(t)
            # Skip if producer is not bound at that tile (loader must have bound per-tile producer)
            if tile not in self._producers.get(t.task_id, {}):
                continue
            if tile in claimed_tiles:
                # Tile already taken this quantum; task waits
                continue
            assigned[tile] = t.task_id
            claimed_tiles.add(tile)
        return assigned

    def _gate_all(self, enable: bool) -> None:
        for _tid, by_tile in self._producers.items():
            for _tile, plist in by_tile.items():
                for p in plist:
                    setattr(p, "_sched_enabled", bool(enable))

    def _apply_assignment(self, new_assign: Dict[int, int]) -> None:
        """
        Apply tile assignments for the next quantum:
          - Disable all producers, then enable only those for assigned (tile, task).
          - Update counters: context switches, preemptions, per-priority dispatch.
          - Advance round-robin index for multi-tile tasks that were actually dispatched.
        """
        # Counters: detect switches and preemptions
        for tile, new_tid in new_assign.items():
            old_tid = self._current_assign.get(tile)
            if old_tid is not None and old_tid != new_tid:
                self._context_switches += 1
                # Preemption if new priority > old priority
                old_pri = int(self._tasks[old_tid].priority)
                new_pri = int(self._tasks[new_tid].priority)
                if new_pri > old_pri:
                    self._preemptions += 1

        # Program gating: off -> on for assigned
        self._gate_all(enable=False)
        for tile, tid in new_assign.items():
            for p in self._producers.get(tid, {}).get(tile, []):
                setattr(p, "_sched_enabled", True)

        # Advance RR only for multi-tile tasks that actually ran
        for tile, tid in new_assign.items():
            task = self._tasks[tid]
            aff = self._resolve_affinity_ids(task)
            if len(aff) > 1:
                task._rr_idx = (task._rr_idx + 1) % len(aff)

        # Update per-priority dispatch counts
        for tile, tid in new_assign.items():
            pri = int(self._tasks[tid].priority)
            self._dispatch_counts_by_pri[pri] = int(self._dispatch_counts_by_pri.get(pri, 0)) + 1

        # Roll assignments
        self._prev_assign = dict(self._current_assign)
        self._current_assign = dict(new_assign)

    # ---- Introspection ----

    def get_counters(self) -> Dict[str, object]:
        runq_len = len([t for t in self._tasks.values() if t.enabled])
        # Export a minimal scheduler counters block
        return {
            "tasks_submitted": int(len(self._tasks)),
            "run_queue_len_current": int(runq_len),
            "run_queue_len_max": int(self._runq_len_max),
            "context_switches": int(self._context_switches),
            "preemptions": int(self._preemptions),
            "tasks_running_per_tile": dict(self._current_assign),
            "dispatch_counts_by_priority": dict(self._dispatch_counts_by_pri),
        }