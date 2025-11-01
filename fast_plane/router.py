from __future__ import annotations

# E1a: Input-buffered router with ESC (DO/XY) and MAIN (weighted ECMP) VCs.
# - Per-VC FIFO input queues on 5 ports (N, S, E, W, LOCAL).
# - Credit-based flow control per VC/output with backpressure.
# - Single-cycle crossbar model with simple round-robin arbitration.
# - Router pipeline latency parameterized; link latency handled by Link.
#
# Determinism:
# - Fixed port ordering and persistent round-robin pointers.
# - MAIN weighted ECMP uses deterministic weighted choice over admissible next hops.
#
# Invariants:
# - Per input buffer: occupancy + free_credits == capacity (asserted on enqueue/dequeue paths).

from dataclasses import dataclass, field
from typing import Dict, Deque, List, Optional, Tuple
from collections import deque

from fast_plane.types import (
    VC,
    Dir,
    Coord,
    MeshShape,
    Flit,
    RouterWeights,
    NoCParams,
    deterministic_weighted_choice,
)


def _port_order() -> List[Dir]:
    # Deterministic input port iteration order
    return [Dir.LOCAL, Dir.NORTH, Dir.SOUTH, Dir.EAST, Dir.WEST]


def _out_order() -> List[Dir]:
    # Deterministic output port iteration order (no need for LOCAL first)
    return [Dir.NORTH, Dir.SOUTH, Dir.EAST, Dir.WEST, Dir.LOCAL]


@dataclass
class _VCQ:
    q: Deque[Flit]
    capacity: int
    free: int  # free credits from sender perspective (capacity - occupancy)

    def assert_invariant(self) -> None:
        occ = len(self.q)
        assert occ + self.free == self.capacity, f"credit invariant broken: occ={occ} free={self.free} cap={self.capacity}"


@dataclass
class _InPort:
    vcq: Dict[VC, _VCQ]


@dataclass
class _OutPort:
    # Credits available to send toward neighbor's input (per VC)
    credits: Dict[VC, int]
    # RR pointer over input port indices (shared across VCs; ESC priority handled in arbiter)
    rr_ptr: int = 0
    # Router pipeline stages for this output (list length equals router_pipeline_latency)
    # Each stage entry holds a Flit or None. When latency=0 pipeline is not used.
    pipeline: List[Optional[Flit]] = field(default_factory=list)


class Router:
    """
    Input-buffered router with two VCs: ESC and MAIN.

    Ports:
      - Inputs: one FIFO per VC for each Dir in {N, S, E, W, LOCAL}
      - Outputs: one lane per Dir in {N, S, E, W}; LOCAL delivery is immediate.

    Flow control:
      - Output credits per VC indicate downstream free slots in peer's input buffer.
      - On dequeue from an input buffer, one slot frees and a credit is returned along the
        reverse link (if non-LOCAL input).

    Pipeline:
      - router_pipeline_latency cycles from dequeue to link injection; modeled as a per-output pipeline.
      - When latency==0, flits are sent directly to the link (subject to link stage-0 availability).
    """

    def __init__(
        self,
        tile_id: int,
        coord: Coord,
        mesh: MeshShape,
        params: NoCParams,
        weights: Optional[RouterWeights] = None,
    ) -> None:
        params.validate()
        self.tile_id = tile_id
        self.coord = coord
        self.mesh = mesh
        self.params = params
        self.pipe_len = max(0, int(params.router_pipeline_latency))
        self.weights = weights  # MAIN VC weights for this router (may be None)
        # Neighbor link wiring filled by NoC after construction: out/in share the same object per dir
        self.neighbors: Dict[Dir, "Link"] = {}

        # Input per-VC queues (including LOCAL)
        self._in: Dict[Dir, _InPort] = {}
        for d in _port_order():
            self._in[d] = _InPort(
                vcq={
                    VC.ESC: _VCQ(deque(), params.buffer_depth_per_vc, params.buffer_depth_per_vc),
                    VC.MAIN: _VCQ(deque(), params.buffer_depth_per_vc, params.buffer_depth_per_vc),
                }
            )

        # Outputs (no LOCAL egress pipeline; LOCAL delivery immediate)
        self._out: Dict[Dir, _OutPort] = {}
        for d in [Dir.NORTH, Dir.SOUTH, Dir.EAST, Dir.WEST]:
            self._out[d] = _OutPort(
                credits={VC.ESC: 0, VC.MAIN: 0},
                rr_ptr=0,
                pipeline=[None for _ in range(self.pipe_len)],
            )
        # Local-output RR pointer (since LOCAL has no _OutPort entry)
        self._rr_local = 0

        # Counters (minimal for tests)
        self.counters = {
            "enqueues": {VC.ESC: 0, VC.MAIN: 0},
            "dequeues": {VC.ESC: 0, VC.MAIN: 0},
            "credits_used": {VC.ESC: 0, VC.MAIN: 0},     # output credits consumed
            "credits_granted": {VC.ESC: 0, VC.MAIN: 0},  # credits arrived from downstream
            "credit_underflow": {VC.ESC: 0, VC.MAIN: 0}, # attempts without credit (should stay 0)
        }

        # Delivery inbox (LOCAL consumption) for tests
        self.delivered_flits: List[Flit] = []

    # ---- Wiring helpers (called by NoC) ----

    def connect_neighbor(self, dir_out: Dir, link: "Link") -> None:
        self.neighbors[dir_out] = link

    def set_initial_output_credits(self, dir_out: Dir, esc_main_capacity: int) -> None:
        if dir_out not in self._out:
            return
        self._out[dir_out].credits[VC.ESC] = esc_main_capacity
        self._out[dir_out].credits[VC.MAIN] = esc_main_capacity

    # ---- External events from Link ----

    def enqueue(self, in_dir: Dir, flit: Flit) -> bool:
        """Arrival from link or LOCAL injection. Respect input buffer capacity."""
        port = self._in[in_dir]
        vcq = port.vcq[flit.vc]
        if vcq.free == 0:
            # No space
            return False
        vcq.q.append(flit)
        vcq.free -= 1
        vcq.assert_invariant()
        self.counters["enqueues"][flit.vc] += 1
        return True

    def on_credit(self, out_dir: Dir, vc: VC, count: int) -> None:
        """Credit arrival from downstream peer for given output direction."""
        if out_dir not in self._out:
            # LOCAL or invalid; ignore defensively
            return
        self._out[out_dir].credits[vc] += int(count)
        self.counters["credits_granted"][vc] += int(count)

    # ---- Routing policies ----

    def _route_escape(self, dst_coord: Coord) -> Dir:
        # DO/XY routing: strictly X first, then Y.
        if self.coord.x < dst_coord.x:
            return Dir.EAST
        if self.coord.x > dst_coord.x:
            return Dir.WEST
        if self.coord.y < dst_coord.y:
            return Dir.SOUTH
        if self.coord.y > dst_coord.y:
            return Dir.NORTH
        return Dir.LOCAL

    def _admissible_next_hops(self, dst_coord: Coord) -> List[Dir]:
        # Return dirs that reduce Manhattan distance; LOCAL if already at dst
        dirs: List[Dir] = []
        if self.coord.x < dst_coord.x:
            dirs.append(Dir.EAST)
        elif self.coord.x > dst_coord.x:
            dirs.append(Dir.WEST)
        if self.coord.y < dst_coord.y:
            dirs.append(Dir.SOUTH)
        elif self.coord.y > dst_coord.y:
            dirs.append(Dir.NORTH)
        if not dirs:
            dirs.append(Dir.LOCAL)
        return dirs

    def _route_main(self, flit: Flit) -> Dir:
        # Weighted ECMP over admissible next hops; fallback to DO if weights missing/zero.
        dst_coord = self.mesh.coord_of(flit.dst)
        admissible = self._admissible_next_hops(dst_coord)
        # If no weights provided, DO fallback
        if self.weights is None:
            return self._route_escape(dst_coord)
        wmap = self.weights.as_dict()
        # Restrict to admissible
        restricted: Dict[Dir, int] = {d: int(wmap.get(d, 0)) for d in admissible}
        total = sum(max(0, w) for w in restricted.values())
        if total == 0:
            return self._route_escape(dst_coord)
        # Deterministic seed per-flow and router
        seed = (
            (self.tile_id & 0xFFFF)
            ^ ((flit.msg_id & 0xFFFFFFFF) << 16)
            ^ ((flit.src & 0xFFFF) << 1)
            ^ ((flit.dst & 0xFFFF) << 3)
        ) & 0xFFFFFFFFFFFFFFFF
        return deterministic_weighted_choice(restricted, seed)

    # ---- Arbitration and crossbar ----

    def _hol(self, in_dir: Dir, vc: VC) -> Optional[Flit]:
        q = self._in[in_dir].vcq[vc].q
        return q[0] if q else None

    def _pop(self, in_dir: Dir, vc: VC) -> Flit:
        port = self._in[in_dir]
        vcq = port.vcq[vc]
        fl = vcq.q.popleft()
        vcq.free += 1
        vcq.assert_invariant()
        self.counters["dequeues"][vc] += 1
        # Return credit on reverse link for non-LOCAL input
        if in_dir != Dir.LOCAL:
            link = self.neighbors.get(in_dir)
            if link is not None:
                link.return_credit_from(self, vc, 1)
        return fl

    def _can_send_out(self, out_dir: Dir, vc: VC) -> bool:
        if out_dir == Dir.LOCAL:
            return True
        if out_dir not in self._out:
            return False
        # When pipeline length > 0, stage-0 must be free; when 0, link must accept now
        if self.pipe_len > 0:
            if self._out[out_dir].pipeline and self._out[out_dir].pipeline[0] is not None:
                return False
        else:
            # zero-latency router pipeline: require link stage-0 availability
            link = self.neighbors.get(out_dir)
            if link is None or not link.can_send_from(self):
                return False
        # Need output credit
        return self._out[out_dir].credits[vc] > 0

    def _commit_send(self, in_dir: Dir, out_dir: Dir, vc: VC) -> bool:
        """
        Perform dequeue from input and either enqueue into router pipeline stage-0 or
        send directly to link for latency==0. Returns True on success.
        """
        if not self._can_send_out(out_dir, vc):
            return False

        fl = self._pop(in_dir, vc)  # frees input slot and triggers credit return upstream

        if out_dir == Dir.LOCAL:
            self.delivered_flits.append(fl)
            return True

        # Consume output credit now (reserve for router pipeline)
        self._out[out_dir].credits[vc] -= 1
        self.counters["credits_used"][vc] += 1

        if self.pipe_len > 0:
            # Stage into pipeline stage-0
            self._out[out_dir].pipeline[0] = fl
        else:
            # Send immediately to link
            link = self.neighbors[out_dir]
            link.send_from(self, fl)
        return True

    def _arbitrate_one_dir(self, out_dir: Dir, vc_priority: List[VC]) -> None:
        """
        For a given output dir, pick at most one (in_dir, vc) using RR over in_dirs and
        VC priority (ESC before MAIN).
        """
        in_dirs = _port_order()
        used = False
        # LOCAL has no _OutPort entry; keep a dedicated RR pointer
        if out_dir == Dir.LOCAL:
            start = self._rr_local % len(in_dirs)
        else:
            start = self._out[out_dir].rr_ptr % len(in_dirs)
        # Try ESC first then MAIN
        for vc in vc_priority:
            if used:
                break
            for k in range(len(in_dirs)):
                idx = (start + k) % len(in_dirs)
                in_dir = in_dirs[idx]
                hol = self._hol(in_dir, vc)
                if hol is None:
                    continue
                # Compute desired next hop for this HOL flit
                if vc == VC.ESC:
                    nxt = self._route_escape(self.mesh.coord_of(hol.dst))
                else:
                    nxt = self._route_main(hol)

                if nxt != out_dir:
                    continue
                # Can we send?
                if self._commit_send(in_dir, out_dir, vc):
                    if out_dir == Dir.LOCAL:
                        self._rr_local = (idx + 1) % len(in_dirs)
                    else:
                        self._out[out_dir].rr_ptr = (idx + 1) % len(in_dirs)
                    used = True
                    break
        if not used:
            # Advance pointer slowly to prevent bias even if no send
            if out_dir == Dir.LOCAL:
                self._rr_local = (self._rr_local + 1) % len(in_dirs)
            else:
                self._out[out_dir].rr_ptr = (self._out[out_dir].rr_ptr + 1) % len(in_dirs)

    def _advance_pipelines_and_send(self) -> None:
        # First, try to send from the last stage to link (if any)
        for out_dir, op in self._out.items():
            if out_dir == Dir.LOCAL or self.pipe_len == 0:
                continue
            if not op.pipeline:
                continue
            last_idx = self.pipe_len - 1
            fl = op.pipeline[last_idx]
            if fl is not None:
                link = self.neighbors.get(out_dir)
                if link is not None and link.can_send_from(self):
                    link.send_from(self, fl)
                    op.pipeline[last_idx] = None
            # Shift pipeline towards output
            for i in range(self.pipe_len - 1, 0, -1):
                if op.pipeline[i] is None and op.pipeline[i - 1] is not None:
                    op.pipeline[i] = op.pipeline[i - 1]
                    op.pipeline[i - 1] = None
            # stage-0 remains for new arbitration fill

    # ---- Public tick ----

    def tick(self) -> None:
        """
        Pipeline stage sequence:
          1) advance pipelines and attempt sends to links (for latency > 0)
          2) route + arbitrate per output dir; ESC has priority over MAIN
        """
        # Step 1: pipelines
        if self.pipe_len > 0:
            self._advance_pipelines_and_send()

        # Step 2: arbitration per direction; VC priority ESC -> MAIN
        for out_dir in _out_order():
            if out_dir == Dir.LOCAL:
                # LOCAL output collects deliveries implicitly via commit_send
                # Allow arbitration targeting LOCAL (drain when at dst)
                self._arbitrate_one_dir(out_dir, [VC.ESC, VC.MAIN])
            else:
                self._arbitrate_one_dir(out_dir, [VC.ESC, VC.MAIN])

    # ---- Debug helpers ----

    def occupancy_snapshot(self) -> Dict[str, Dict[str, int]]:
        snap: Dict[str, Dict[str, int]] = {}
        for d in _port_order():
            sd = {}
            for vc in (VC.ESC, VC.MAIN):
                sd[f"vc{int(vc)}_occ"] = len(self._in[d].vcq[vc].q)
            snap[str(d.name)] = sd  # type: ignore
        return snap

    def assert_credit_invariants(self) -> None:
        for d in _port_order():
            for vc in (VC.ESC, VC.MAIN):
                self._in[d].vcq[vc].assert_invariant()