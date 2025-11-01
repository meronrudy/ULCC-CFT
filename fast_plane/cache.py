from __future__ import annotations

"""
E1c: CacheMissProcess

Deterministic, MPKI-driven cache-miss generator per tile.

Units and model:
- mpki: misses per 1000 instructions (float >= 0)
- ipc: instructions per cycle (float >= 0)
- message_size_flits: flits per memory request message (int >= 1)
- vc: fast_plane.types.VC to use for injection (default MAIN)
- mc_tile: (x, y) coordinates of the memory controller tile (destination)
- enable: enable/disable traffic generation

Rate model:
- Expected misses per cycle = (mpki / 1000.0) * ipc (dimensionless “miss tokens” per cycle).
- Deterministic token accumulator (no RNG). When tokens >= 1.0, emit exactly one full memory
  request message of size message_size_flits flits, and decrement tokens by 1.0.
- Injection path uses NoC.try_inject_message() against the source LOCAL input buffer. If insufficient
  headroom for the full message, the attempt is deferred (tokens are retained) and counted.

NoC integration and ordering:
- NoC will call cache_procs.step() after producers:
    links → routers → deliver LOCAL → consumer.step() → producers.step() → cache_procs.step()
"""

from typing import Optional, Tuple

from fast_plane.types import VC, Message, Coord


class CacheMissProcess:
    """
    Deterministic MPKI-based cache-miss traffic source.

    Counters (ints except where noted):
    - emitted_requests: number of successfully injected memory requests (messages)
    - emitted_flits: total flits injected by this process
    - deferred_events: number of deferrals due to insufficient LOCAL headroom
    - tokens_accumulated: current token bucket value (float; miss-events)
    """

    def __init__(
        self,
        mpki: float,
        ipc: float,
        message_size_flits: int = 4,
        vc: VC = VC.MAIN,
        mc_tile: Tuple[int, int] = (0, 0),
        enable: bool = True,
    ) -> None:
        if mpki < 0.0:
            raise ValueError("mpki must be >= 0")
        if ipc < 0.0:
            raise ValueError("ipc must be >= 0")
        if message_size_flits < 1:
            raise ValueError("message_size_flits must be >= 1")

        self.mpki = float(mpki)
        self.ipc = float(ipc)
        self.rate = (self.mpki / 1000.0) * self.ipc  # miss-events per cycle
        self.msg_size = int(message_size_flits)
        self.vc = vc
        self.mc_tile_xy = (int(mc_tile[0]), int(mc_tile[1]))
        self.enable = bool(enable)

        self._noc: Optional["NoC"] = None
        self._tile_id: Optional[int] = None
        self._mc_tile_id: Optional[int] = None
        self._tokens: float = 0.0
        self._next_msg_id: int = 1

        self.counters = {
            "emitted_requests": 0,
            "emitted_flits": 0,
            "deferred_events": 0,
            "tokens_accumulated": 0.0,
        }

    def _attach(self, noc: "NoC", tile_id: int) -> None:
        # Late import type via string to avoid circular deps
        self._noc = noc
        self._tile_id = tile_id
        self._mc_tile_id = noc.mesh.tile_id(Coord(self.mc_tile_xy[0], self.mc_tile_xy[1]))

    def step(self) -> None:
        if not self.enable:
            return
        assert self._noc is not None and self._tile_id is not None and self._mc_tile_id is not None

        # Accumulate deterministic miss tokens
        self._tokens += self.rate
        self.counters["tokens_accumulated"] = float(self._tokens)

        # Emit at most as many requests as whole tokens allow; stop at first deferral
        while self._tokens >= 1.0:
            msg = Message(
                msg_id=self._next_msg_id,
                src=self._tile_id,
                dst=self._mc_tile_id,
                vc=self.vc,
                size_flits=self.msg_size,
            )
            ok = self._noc.try_inject_message(msg)
            if ok:
                self._tokens -= 1.0
                self._next_msg_id += 1
                self.counters["emitted_requests"] += 1
                self.counters["emitted_flits"] += self.msg_size
            else:
                # Insufficient LOCAL headroom; defer and retain tokens
                self.counters["deferred_events"] += 1
                break