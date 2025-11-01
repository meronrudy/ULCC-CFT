from __future__ import annotations

# E1a: Bidirectional point-to-point link with latency pipeline and credit return path.
# One flit per cycle capacity; per-VC credit messages return on the reverse direction.

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from fast_plane.types import VC, Flit, Dir


def opposite(d: Dir) -> Dir:
    if d == Dir.NORTH:
        return Dir.SOUTH
    if d == Dir.SOUTH:
        return Dir.NORTH
    if d == Dir.EAST:
        return Dir.WEST
    if d == Dir.WEST:
        return Dir.EAST
    return Dir.LOCAL


@dataclass
class _DirPipes:
    # Flit pipeline stages for one direction (from src to dst).
    flit_stages: List[Optional[Flit]]
    # Credit pipeline stages (dict of VC -> count) going reverse (dst to src).
    credit_stages: List[Dict[VC, int]]


class Link:
    """
    Discrete-time latency link with symmetric pipelines in both directions.
    - One flit/cycle capacity (serialization simplified for E1a).
    - Credit return path modeled as a pipeline of same latency in reverse direction.
    """

    def __init__(self, latency: int, router_a: "Router", router_b: "Router", dir_a_to_b: Dir) -> None:
        assert latency >= 0
        self.latency = latency
        self.a = router_a
        self.b = router_b
        # Orientation: sending A -> B uses this direction from A's perspective.
        self.dir_a_to_b = dir_a_to_b
        # Pipelines for A->B (flit) and B->A (flit). Credit pipes mirror reverse.
        self._ab = _DirPipes(
            flit_stages=[None for _ in range(latency)],
            credit_stages=[{ } for _ in range(latency)]
        )
        self._ba = _DirPipes(
            flit_stages=[None for _ in range(latency)],
            credit_stages=[{ } for _ in range(latency)]
        )

    # --- Send path ---

    def can_send_from(self, router: "Router") -> bool:
        """Stage-0 of the appropriate direction must be empty to accept a new flit."""
        if self.latency == 0:
            return True
        if router is self.a:
            return self._ab.flit_stages[0] is None
        elif router is self.b:
            return self._ba.flit_stages[0] is None
        else:
            raise AssertionError("Router not an endpoint of this Link")

    def send_from(self, router: "Router", flit: Flit) -> None:
        """Enqueue a flit into stage-0 of the proper direction."""
        if self.latency == 0:
            # Deliver immediately to the other endpoint's input buffer.
            if router is self.a:
                self._deliver_to(self.b, opposite(self.dir_a_to_b), flit)
            elif router is self.b:
                self._deliver_to(self.a, self.dir_a_to_b, flit)
            else:
                raise AssertionError("Router not an endpoint of this Link")
            return

        if router is self.a:
            if self._ab.flit_stages[0] is not None:
                raise AssertionError("Link A->B stage0 busy")
            self._ab.flit_stages[0] = flit
        elif router is self.b:
            if self._ba.flit_stages[0] is not None:
                raise AssertionError("Link B->A stage0 busy")
            self._ba.flit_stages[0] = flit
        else:
            raise AssertionError("Router not an endpoint of this Link")

    # --- Credit return path ---

    def return_credit_from(self, router: "Router", vc: VC, count: int = 1) -> None:
        """
        Schedule a credit to be delivered to the other endpoint (reverse direction).
        Credits move through the credit pipeline and increment the sender's output
        credits upon arrival.
        """
        if count <= 0:
            return
        if self.latency == 0:
            # Immediate credit arrival to the other endpoint.
            if router is self.a:
                # Credit A -> B arrives at B for out_dir opposite(dir_a_to_b)
                self._credit_arrive(self.b, opposite(self.dir_a_to_b), vc, count)
            elif router is self.b:
                # Credit B -> A arrives at A for out_dir dir_a_to_b
                self._credit_arrive(self.a, self.dir_a_to_b, vc, count)
            else:
                raise AssertionError("Router not an endpoint of this Link")
            return

        if router is self.a:
            # Queue credit to flow A -> B
            stage0 = self._ab.credit_stages[0]
            stage0[vc] = stage0.get(vc, 0) + count
        elif router is self.b:
            # Queue credit to flow B -> A
            stage0 = self._ba.credit_stages[0]
            stage0[vc] = stage0.get(vc, 0) + count
        else:
            raise AssertionError("Router not an endpoint of this Link")

    # --- Tick advance ---

    def tick(self) -> None:
        """
        Advance both flit and credit pipelines by one cycle and perform deliveries.
        Delivery order: move deepest stage to sink first to avoid same-cycle reuse.
        """
        # Move A->B flits
        if self.latency > 0:
            flit_ab_out = self._ab.flit_stages[-1]
            # Shift flit pipeline towards output
            for i in range(self.latency - 1, 0, -1):
                self._ab.flit_stages[i] = self._ab.flit_stages[i - 1]
            self._ab.flit_stages[0] = None

            if flit_ab_out is not None:
                # Deliver into B on input port opposite(dir_a_to_b)
                self._deliver_to(self.b, opposite(self.dir_a_to_b), flit_ab_out)

            # Move B->A flits
            flit_ba_out = self._ba.flit_stages[-1]
            for i in range(self.latency - 1, 0, -1):
                self._ba.flit_stages[i] = self._ba.flit_stages[i - 1]
            self._ba.flit_stages[0] = None

            if flit_ba_out is not None:
                # Deliver into A on input port dir_a_to_b (since opposite of opposite)
                self._deliver_to(self.a, self.dir_a_to_b, flit_ba_out)

            # Move credits: A<-B along credit_stages of _ba, and B<-A along credit_stages of _ab
            credit_ba_out = self._ba.credit_stages[-1]
            for i in range(self.latency - 1, 0, -1):
                self._ba.credit_stages[i] = self._ba.credit_stages[i - 1]
            self._ba.credit_stages[0] = {}

            for vc, cnt in credit_ba_out.items():
                # Credits arriving at A for direction A->B (dir_a_to_b)
                self._credit_arrive(self.a, self.dir_a_to_b, vc, cnt)

            credit_ab_out = self._ab.credit_stages[-1]
            for i in range(self.latency - 1, 0, -1):
                self._ab.credit_stages[i] = self._ab.credit_stages[i - 1]
            self._ab.credit_stages[0] = {}

            for vc, cnt in credit_ab_out.items():
                # Credits arriving at B for direction B->A (opposite(dir_a_to_b))
                self._credit_arrive(self.b, opposite(self.dir_a_to_b), vc, cnt)

    # --- Helpers ---

    def _deliver_to(self, router: "Router", in_dir: Dir, flit: Flit) -> None:
        ok = router.enqueue(in_dir, flit)
        if not ok:
            # This should never happen if send respected credit; assert hard in debug.
            raise AssertionError("Downstream input buffer overflow on flit delivery")

    def _credit_arrive(self, router: "Router", out_dir: Dir, vc: VC, count: int) -> None:
        router.on_credit(out_dir, vc, count)