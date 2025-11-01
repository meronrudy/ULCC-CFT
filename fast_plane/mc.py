from __future__ import annotations

"""
E1c: Memory controller (MC) with FR-FCFS approximation.

This MC is attached to a NoC tile as the LOCAL consumer. It reconstructs memory
request boundaries from delivered flits (Flit.is_head/is_tail and size_flits),
enqueues requests, and schedules them using either:
- FR-FCFS: within a window of the oldest N requests, prioritize row hits then oldest.
- FIFO: strictly arrival order ignoring row-hit preference.

Units:
- bank_count: int >= 1 (banks per channel)
- channel_count: int >= 1 (independent channels)
- rows_per_bank: int >= 1 (row buffer choices per bank)
- window_size: int >= 1 (reorder window size)
- t_row_hit: int >= 1 cycles (row-buffer hit service time component)
- t_row_miss: int >= t_row_hit cycles (row-buffer miss service time component)
- t_bus: int >= 0 cycles (bus/other fixed component)
- mode: "FRFCFS" or "FIFO"

Scheduling/Timing model:
- At each MC step (one NoC cycle), the controller:
  1) Consumes completed requests whose completion_cycle <= current_cycle.
  2) Schedules at most one request per cycle (globally) whose channel is available
     (current_cycle >= next_free_cycle[channel]).
     - FR-FCFS: among up to window_size oldest pending requests with available channels,
       favor row hits (last_row[(ch, bank)] == row). Choose oldest among hits; if none, oldest miss.
     - FIFO: choose oldest available channel request.
- Service latency = t_bus + (t_row_hit or t_row_miss).
- On scheduling a miss, the row buffer for (ch, bank) is updated to the request row.

Output/metrics (exposed via metrics()):
- enqueued_requests, row_hits, row_misses, served_requests
- avg_latency_cycles (service component only, excludes queue wait)
- p95_latency_cycles (rolling-window estimate)
- queue_depth (current pending requests)
"""

from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque

from fast_plane.types import Coord


@dataclass
class MCConfig:
    bank_count: int
    channel_count: int
    window_size: int
    t_row_hit: int
    t_row_miss: int
    t_bus: int
    rows_per_bank: int
    mode: str = "FRFCFS"  # or "FIFO"

    def validate(self) -> None:
        if self.bank_count < 1 or self.channel_count < 1:
            raise ValueError("bank_count and channel_count must be >= 1")
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.t_row_hit < 1:
            raise ValueError("t_row_hit must be >= 1")
        if self.t_row_miss < self.t_row_hit:
            raise ValueError("t_row_miss must be >= t_row_hit")
        if self.t_bus < 0:
            raise ValueError("t_bus must be >= 0")
        if self.rows_per_bank < 1:
            raise ValueError("rows_per_bank must be >= 1")
        if self.mode not in ("FRFCFS", "FIFO"):
            raise ValueError("mode must be 'FRFCFS' or 'FIFO'")


class MemoryController:
    """
    Tile-attached memory controller that implements the NoC consumer interface:
    - _attach(noc, tile_id)
    - on_flits_delivered(flits: List[Flit])
    - step()

    Note: Response traffic is not modeled in E1c; only request service and internal timing/counters.
    """

    def __init__(self, cfg: MCConfig) -> None:
        cfg.validate()
        self.cfg = cfg

        # Attachment
        self._noc = None  # type: Optional["NoC"]
        self._tile_id: Optional[int] = None

        # Partial message assembly keyed by (src_tile, msg_id)
        # Each entry holds (expected_size, accumulated_count)
        self._assembling: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # Pending requests queue: elements are tuples
        # (arrival_cycle, src_tile, msg_id, channel, bank, row)
        self._pending: Deque[Tuple[int, int, int, int, int, int]] = deque()

        # Per-channel next free cycle (busy until this cycle - 1)
        self._ch_next_free: List[int] = [0 for _ in range(self.cfg.channel_count)]

        # Row buffer: last accessed row per (channel, bank)
        self._last_row: Dict[Tuple[int, int], Optional[int]] = {
            (ch, b): None
            for ch in range(self.cfg.channel_count)
            for b in range(self.cfg.bank_count)
        }

        # In-flight completions: list of (completion_cycle, channel, bank, row, was_miss, latency)
        self._inflight: List[Tuple[int, int, int, int, bool, int]] = []

        # Counters and latency tracking
        self._enqueued = 0
        self._row_hits = 0
        self._row_misses = 0
        self._served = 0
        self._latency_sum = 0.0
        self._lat_hist: List[int] = []  # rolling window
        self._lat_hist_cap = 1024

        # MC exposes a counters dict for potential external snapshots (not used in NoC aggregation)
        self.counters: Dict[str, int] = {}

    # ---- Attachment and NoC bridge ----

    def _attach(self, noc: "NoC", tile_id: int) -> None:
        self._noc = noc
        self._tile_id = tile_id

    def on_flits_delivered(self, flits: List["Flit"]) -> None:
        """
        Called by NoC each cycle with any newly delivered LOCAL flits for this tile.
        Reconstruct complete request messages and enqueue them into the pending queue.
        """
        assert self._noc is not None
        cur = self._noc.current_cycle()
        for fl in flits:
            key = (int(fl.src), int(fl.msg_id))
            if fl.is_head:
                # Initialize assembly entry if not present
                self._assembling.setdefault(key, (int(fl.size_flits), 0))
            # Accumulate count
            if key not in self._assembling:
                # Defensive: if head was not seen (e.g., single-flit corner), initialize now
                self._assembling[key] = (int(fl.size_flits), 0)
            exp, cnt = self._assembling[key]
            cnt += 1
            self._assembling[key] = (exp, cnt)

            if fl.is_tail or cnt >= exp:
                # Message complete: enqueue as a memory request
                ch, bank, row = self._map_to_ch_bank_row(src_tile=int(fl.src), msg_id=int(fl.msg_id))
                self._pending.append((cur, int(fl.src), int(fl.msg_id), ch, bank, row))
                self._enqueued += 1
                # Clear assembly
                self._assembling.pop(key, None)

    def step(self) -> None:
        """
        Advance MC by one NoC cycle: retire completions and schedule the next request if possible.
        """
        assert self._noc is not None
        cur = self._noc.current_cycle()

        # 1) Retire any completed requests
        if self._inflight:
            # Keep only those with completion_cycle > cur
            remaining: List[Tuple[int, int, int, int, bool, int]] = []
            for (done_cyc, ch, bank, row, was_miss, lat) in self._inflight:
                if done_cyc <= cur:
                    # Completion
                    self._served += 1
                    self._latency_sum += float(lat)
                    self._lat_hist.append(lat)
                    if len(self._lat_hist) > self._lat_hist_cap:
                        self._lat_hist = self._lat_hist[-self._lat_hist_cap :]
                    # last_row already updated on schedule for misses (open-row model)
                else:
                    remaining.append((done_cyc, ch, bank, row, was_miss, lat))
            self._inflight = remaining

        # 2) Schedule at most one request per cycle (global single-issue policy)
        #    Channel availability constraint applies.
        # Gather up to window_size oldest pending candidates that have free channel at 'cur'
        if not self._pending:
            return

        # Find indices of candidates within window_size that can be issued now (channel free)
        wnd = min(self.cfg.window_size, len(self._pending))
        # We'll scan the deque without popping by index via list conversion for simplicity
        pending_list = list(self._pending)
        candidate_indices: List[int] = []
        for idx in range(wnd):
            arr, _src, _mid, ch, _bank, _row = pending_list[idx]
            if cur >= self._ch_next_free[ch]:
                candidate_indices.append(idx)

        if not candidate_indices:
            return  # all channels busy

        # Select index to issue based on mode
        select_idx_in_deque: Optional[int] = None
        if self.cfg.mode == "FIFO":
            # Choose the earliest candidate
            select_idx_in_deque = candidate_indices[0]
        else:
            # FR-FCFS: prefer row hits among candidates
            best_hit_idx: Optional[int] = None
            for idx in candidate_indices:
                _arr, _src, _mid, ch, bank, row = pending_list[idx]
                last = self._last_row[(ch, bank)]
                if last is not None and last == row:
                    best_hit_idx = idx
                    break
            if best_hit_idx is not None:
                select_idx_in_deque = best_hit_idx
            else:
                select_idx_in_deque = candidate_indices[0]

        # Pop the selected request from the deque efficiently by rotating
        assert select_idx_in_deque is not None
        for _ in range(select_idx_in_deque):
            self._pending.append(self._pending.popleft())
        arr, src, mid, ch, bank, row = self._pending.popleft()

        # Determine hit/miss
        last = self._last_row[(ch, bank)]
        is_hit = (last is not None and last == row)

        # Service time
        lat = self.cfg.t_bus + (self.cfg.t_row_hit if is_hit else self.cfg.t_row_miss)

        # Update row buffer on miss (open the row)
        if not is_hit:
            self._last_row[(ch, bank)] = row
            self._row_misses += 1
        else:
            self._row_hits += 1

        # Reserve channel and record completion
        start_cyc = cur
        done_cyc = start_cyc + lat
        self._ch_next_free[ch] = done_cyc
        self._inflight.append((done_cyc, ch, bank, row, (not is_hit), lat))

        # For completeness, expose a light counters snapshot
        self.counters["queue_occupancy"] = len(self._pending)

    # ---- Mapping (deterministic, no RNG) ----

    def _map_to_ch_bank_row(self, src_tile: int, msg_id: int) -> Tuple[int, int, int]:
        """
        Deterministic mapping from (src_tile, msg_id) to (channel, bank, row).
        Mode-aware row policy to create realistic row-buffer behaviors while keeping determinism:
          - Channel: round-robin by (src_tile + msg_id) across channels.
          - Bank: fixed per-source (src_tile % bank_count) to emphasize bank locality/conflicts.
          - Row:
              * For FRFCFS:
                  - When rows_per_bank <= 4: alternate rows to induce frequent conflicts
                    row = msg_id % rows_per_bank
                  - When rows_per_bank > 4: group rows in pairs to create near-term repetition
                    row = (msg_id // 2) % min(rows_per_bank, 8)
              * For FIFO:
                  - Simple alternating pattern with large domain to reduce incidental hits
                    row = msg_id % max(1, self.cfg.rows_per_bank)
        """
        # Channel and bank mapping (deterministic, no RNG)
        ch = int((src_tile + msg_id) % max(1, self.cfg.channel_count))
        bank = int(src_tile % max(1, self.cfg.bank_count))

        # Row mapping policy depends on mode and rows_per_bank to shape conflicts/hits deterministically
        if self.cfg.mode == "FRFCFS":
            if self.cfg.rows_per_bank <= 4:
                row = int(msg_id % self.cfg.rows_per_bank)
            else:
                row = int((msg_id // 2) % max(1, min(self.cfg.rows_per_bank, 8)))
        else:
            # FIFO: prioritize arrival order and avoid incidental hits by using a wide modulus
            row = int(msg_id % max(1, self.cfg.rows_per_bank))

        return ch, bank, row

    # ---- Metrics ----

    def metrics(self) -> Dict[str, float]:
        """
        Return current aggregate statistics for NoC aggregation and tests.
        """
        avg = float(self._latency_sum / self._served) if self._served > 0 else 0.0
        p95 = 0.0
        if self._lat_hist:
            data = sorted(self._lat_hist)
            k = max(0, int(0.95 * (len(data) - 1)))
            p95 = float(data[k])
        return {
            "enqueued_requests": float(self._enqueued),
            "row_hits": float(self._row_hits),
            "row_misses": float(self._row_misses),
            "served_requests": float(self._served),
            "avg_latency_cycles": float(avg),
            "p95_latency_cycles": float(p95),
            "queue_depth": float(len(self._pending)),
            "latency_sum": float(self._latency_sum),
        }