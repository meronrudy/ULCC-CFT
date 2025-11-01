from __future__ import annotations

"""
Deterministic token-bucket producers and simple consumers for fast_plane.

Design notes (E1b scope):
- No stochastic interarrival: dst selection can be fixed or round-robin over a provided list.
- TokenBucketProducer accumulates fractional tokens deterministically; bucket capacity is burst_size_flits.
  When tokens >= message_size_flits and the source LOCAL input buffer has headroom for the full message,
  the producer injects exactly one message per attempt (may loop to emit multiple messages if enough tokens).
  On insufficient headroom the attempt is deferred (tokens are retained up to the burst cap).
- Consumer enqueues flits that arrive at LOCAL delivery and drains them deterministically at a fixed
  service_rate_flits_per_cycle. An optional sink_latency_cycles models a service pipeline.

Step ordering relative to NoC.step():
- NoC ticks links, then routers, then for each tile: consumers ingest and drain, then producers inject.
  This ordering is fixed and deterministic; it is documented in fast_plane.noc.
"""

from typing import List, Optional, Sequence, Tuple, Union

from fast_plane.types import VC, Message, Coord, Flit


class TokenBucketProducer:
    """
    Deterministic token-bucket producer.

    Parameters:
    - rate_tokens_per_cycle: float >= 0.0; tokens added per cycle.
    - burst_size_flits: int >= 1; bucket capacity in flits (tokens).
    - message_size_flits: int >= 1; size of each message in flits.
    - dst_selection: either a fixed destination (tile_id int or (x,y) coord),
      or a deterministic round-robin list of destinations [tile_id|(x,y), ...].
    - vc: VC.ESC or VC.MAIN (default MAIN).
    - rng_seed: reserved for future stochastic sources; unused in E1b.

    Counters:
    - produced_messages, produced_flits, dropped_or_deferred_events (deferrals due to no headroom).
    """

    def __init__(
        self,
        rate_tokens_per_cycle: float,
        burst_size_flits: int,
        message_size_flits: int,
        dst_selection: Union[int, Tuple[int, int], Sequence[Union[int, Tuple[int, int]]]],
        vc: VC = VC.MAIN,
        rng_seed: int = 0,
    ) -> None:
        if rate_tokens_per_cycle < 0.0:
            raise ValueError("rate_tokens_per_cycle must be >= 0")
        if burst_size_flits < 1:
            raise ValueError("burst_size_flits must be >= 1")
        if message_size_flits < 1:
            raise ValueError("message_size_flits must be >= 1")
        self.rate = float(rate_tokens_per_cycle)
        self.burst = int(burst_size_flits)
        self.msg_size = int(message_size_flits)
        self.vc = vc
        self.rng_seed = int(rng_seed)

        # Destination mode and storage (resolved to tile ids upon attach).
        # Check whether dst_selection is sequence (round robin) or fixed.
        if isinstance(dst_selection, (list, tuple)) and not (
            isinstance(dst_selection, tuple) and len(dst_selection) == 2 and all(isinstance(x, int) for x in dst_selection)
        ):
            # Treat as round-robin sequence
            self._dst_mode = "round_robin"
            self._dst_spec_seq = list(dst_selection)  # type: ignore[arg-type]
            if len(self._dst_spec_seq) == 0:
                raise ValueError("dst_selection sequence must not be empty")
        else:
            self._dst_mode = "fixed"
            self._dst_spec_fixed = dst_selection  # type: ignore[assignment]

        # Runtime state
        self._noc = None  # set by _attach
        self._tile_id: Optional[int] = None
        self._rr_idx = 0
        self._dst_ids_seq: Optional[List[int]] = None
        self._dst_id_fixed: Optional[int] = None
        self._tokens: float = 0.0
        self._next_msg_id: int = 1

        self.counters = {
            "produced_messages": 0,
            "produced_flits": 0,
            "dropped_or_deferred_events": 0,
        }

    def _attach(self, noc: "NoC", tile_id: int) -> None:
        # Local import to avoid circular type import
        self._noc = noc
        self._tile_id = tile_id
        # Resolve destinations to tile_ids deterministically
        if self._dst_mode == "fixed":
            sel = self._dst_spec_fixed
            if isinstance(sel, int):
                self._dst_id_fixed = sel
            elif isinstance(sel, tuple) and len(sel) == 2:
                x, y = sel
                self._dst_id_fixed = noc.mesh.tile_id(Coord(x, y))
            else:
                raise ValueError("Unsupported fixed dst_selection type")
        else:
            seq = []
            for item in self._dst_spec_seq:  # type: ignore[attr-defined]
                if isinstance(item, int):
                    seq.append(item)
                elif isinstance(item, tuple) and len(item) == 2:
                    x, y = item
                    seq.append(noc.mesh.tile_id(Coord(x, y)))
                else:
                    raise ValueError("Unsupported dst_selection entry in sequence")
            self._dst_ids_seq = seq

    def _next_dst(self) -> int:
        assert self._noc is not None and self._tile_id is not None
        if self._dst_mode == "fixed":
            assert self._dst_id_fixed is not None
            return self._dst_id_fixed
        assert self._dst_ids_seq is not None and len(self._dst_ids_seq) > 0
        d = self._dst_ids_seq[self._rr_idx % len(self._dst_ids_seq)]
        self._rr_idx = (self._rr_idx + 1) % len(self._dst_ids_seq)
        return d

    def step(self) -> None:
        """Accumulate tokens and attempt deterministic injection."""
        assert self._noc is not None and self._tile_id is not None
        # Accrue tokens up to burst cap
        self._tokens = min(self.burst, self._tokens + self.rate)
        # Attempt to inject as many messages as tokens and headroom allow.
        # Stop after first deferral to avoid unproductive busy looping.
        while self._tokens >= self.msg_size:
            dst_id = self._next_dst()
            msg = Message(
                msg_id=self._next_msg_id,
                src=self._tile_id,
                dst=dst_id,
                vc=self.vc,
                size_flits=self.msg_size,
            )
            ok = self._noc.try_inject_message(msg)
            if ok:
                self._next_msg_id += 1
                self._tokens -= self.msg_size
                self.counters["produced_messages"] += 1
                self.counters["produced_flits"] += self.msg_size
                # continue loop if enough tokens remain
            else:
                self.counters["dropped_or_deferred_events"] += 1
                break


class Consumer:
    """
    Deterministic sink that drains delivered flits at a fixed service rate.

    Parameters:
    - service_rate_flits_per_cycle: float >= 0.0
    - sink_latency_cycles: int >= 0; if > 0 a simple pipeline is modeled.

    Counters:
    - consumed_flits
    - queue_occupancy (pending + in-service)
    """

    def __init__(self, service_rate_flits_per_cycle: float, sink_latency_cycles: int = 0) -> None:
        if service_rate_flits_per_cycle < 0.0:
            raise ValueError("service_rate_flits_per_cycle must be >= 0")
        if sink_latency_cycles < 0:
            raise ValueError("sink_latency_cycles must be >= 0")
        self.service_rate = float(service_rate_flits_per_cycle)
        self.lat = int(sink_latency_cycles)
        self._noc = None  # type: Optional["NoC"]
        self._tile_id: Optional[int] = None

        self._pending: int = 0  # number of flits waiting to start service
        self._tokens: float = 0.0
        self._pipeline: List[int] = [0 for _ in range(self.lat)] if self.lat > 0 else []

        self.counters = {
            "consumed_flits": 0,
            "queue_occupancy": 0,
        }

    def _attach(self, noc: "NoC", tile_id: int) -> None:
        self._noc = noc
        self._tile_id = tile_id

    def on_flits_delivered(self, flits: List["Flit"]) -> None:
        # We only need the count to model queueing deterministically.
        self._pending += len(flits)

    def _advance_pipeline(self) -> None:
        if self.lat <= 0:
            return
        completed = self._pipeline[-1] if self._pipeline else 0
        if completed:
            self.counters["consumed_flits"] += int(completed)
        # shift right
        for i in range(self.lat - 1, 0, -1):
            self._pipeline[i] = self._pipeline[i - 1]
        if self.lat > 0:
            self._pipeline[0] = 0

    def step(self) -> None:
        # Complete any in-flight service first
        if self.lat > 0:
            self._advance_pipeline()

        # Accrue service tokens
        self._tokens += self.service_rate
        take = int(self._tokens)
        if take > 0 and self._pending > 0:
            n = min(take, self._pending)
            self._tokens -= n
            self._pending -= n
            if self.lat > 0:
                self._pipeline[0] += n
            else:
                self.counters["consumed_flits"] += n

        # Update occupancy counter
        occ = self._pending + (sum(self._pipeline) if self.lat > 0 else 0)
        self.counters["queue_occupancy"] = int(occ)