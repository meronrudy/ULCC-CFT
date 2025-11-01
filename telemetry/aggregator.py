"""
TelemetryAggregator for fixed, deterministic sampling windows (Phase A — E2)

Design choices (per sim/DATA_CONTRACTS.md and sim/TEST_PLAN.md):
- Windowing is cycle-aligned and deterministic. No randomness.
- Frames are assumed valid and non-overlapping within a run (see TelemetryFrame.validate()).
- Missing optional subsystems:
  • Memory metrics: if absent in a window → mc_* fields set to None (latencies) or 0 (served requests).
  • Power/thermal: if per-tile power_pu not present → power_* set to 0.0 and temps set to 0.0.
  • Escape VC / XBar switches not modeled in E1 → flits_tx_main == produced_flits_total, flits_tx_esc == 0, xbar_switches == 0.
  • queue_depth_p95_proxy derived from instantaneous tile queue_depth_p99 samples in the window:
    deterministic percentile approximation (nearest-rank with clamp). If no samples, set to 0.0.

Deterministic computations:
- produced_flits_total = sum over frames of sum over tiles flit_tx
- mc_served_requests = sum over frames of sum over memory activations
- mc_avg_latency = weighted mean over memory controllers by activations (read_latency_avg)
- mc_p95_latency = weighted mean over memory controllers by activations (read_latency_p99)
- power_total_energy = sum over frames of (mean over tiles power_pu) * cycle_window
- power_avg_power = power_total_energy / total_cycles in window
- max_temp_any_tile = max over frames of max over tiles temp_c
- avg_temp_all_tiles = mean of all tile temps across all frames and tiles
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Iterable

from .frame import TelemetryFrame


@dataclass(frozen=True)
class WindowSpec:
    window_cycles: int
    stride_cycles: int
    align_to: int = 0

    def __post_init__(self) -> None:
        if int(self.window_cycles) < 1:
            raise ValueError("window_cycles must be ≥ 1")
        if int(self.stride_cycles) < 1:
            raise ValueError("stride_cycles must be ≥ 1")
        if int(self.align_to) < 0:
            raise ValueError("align_to must be ≥ 0")


@dataclass(frozen=True)
class AggregatedMetrics:
    # Window identity
    cycles: int
    start_cycle: int
    end_cycle: int
    # Throughput and switching
    produced_flits_total: int
    flits_tx_main: int
    flits_tx_esc: int
    xbar_switches: int
    # Memory latency/throughput (if present)
    mc_served_requests: int
    mc_avg_latency: Optional[float]
    mc_p95_latency: Optional[float]
    # Power/Thermal rollups (if present)
    power_total_energy: float
    power_avg_power: float
    max_temp_any_tile: float
    avg_temp_all_tiles: float
    # Queueing proxy
    queue_depth_p95_proxy: float


class TelemetryAggregator:
    def __init__(self, window_spec: WindowSpec) -> None:
        self.spec = window_spec
        self._frames: List[TelemetryFrame] = []
        self._win_start: Optional[int] = None  # start cycle of next window

    def reset(self) -> None:
        self._frames.clear()
        self._win_start = None

    def push(self, frame: TelemetryFrame) -> None:
        # Append in arrival order; frames expected validated already in tests.
        self._frames.append(frame)
        # Initialize first window to aligned boundary not after first frame
        if self._win_start is None:
            # Align the first window start to the largest multiple of stride not after first frame start,
            # but not less than align_to.
            first_start = int(self._frames[0].t_start_cycle)
            align = int(self.spec.align_to)
            stride = int(self.spec.stride_cycles)
            if first_start < align:
                self._win_start = align
            else:
                # floor to stride grid, but ≥ align_to
                delta = first_start - align
                self._win_start = align + (delta // stride) * stride

    def ready(self) -> bool:
        if self._win_start is None:
            return False
        win_end = self._win_start + int(self.spec.window_cycles)
        # Ready when we have at least one frame whose t_end_cycle ≥ win_end
        for fr in reversed(self._frames):
            if int(fr.t_end_cycle) >= win_end:
                return True
        return False

    def next_window(self) -> AggregatedMetrics:
        if not self.ready():
            raise RuntimeError("next_window() called before ready()")
        assert self._win_start is not None
        wstart = int(self._win_start)
        wend = wstart + int(self.spec.window_cycles)

        # Select frames within [wstart, wend)
        frames = _slice_frames_by_cycle(self._frames, wstart, wend)

        # Compute KPIs
        total_cycles = sum(int(f.cycle_window) for f in frames)
        # Throughput
        produced_flits_total = 0
        queue_p99_samples: List[float] = []
        temps: List[float] = []
        max_temp = 0.0

        # Memory aggregations
        mc_served = 0
        lat_avg_num = 0.0
        lat_avg_den = 0
        lat_p95_num = 0.0
        lat_p95_den = 0

        # Power energy and avg
        power_energy = 0.0  # proxy energy in (p.u. * cycles)

        for f in frames:
            # Per-tile
            if f.tile_metrics:
                # flits
                for tm in f.tile_metrics:
                    produced_flits_total += int(tm.flit_tx)
                    queue_p99_samples.append(float(tm.queue_depth_p99))
                    temps.append(float(tm.temp_c))
                    if tm.temp_c > max_temp:
                        max_temp = float(tm.temp_c)
                # power: use mean tile power * cycle_window
                mean_pwr = sum(float(tm.power_pu) for tm in f.tile_metrics) / float(len(f.tile_metrics))
                power_energy += mean_pwr * float(f.cycle_window)
            # Memory
            for mm in f.memory_metrics:
                act = int(mm.activations)
                mc_served += act
                if act > 0:
                    lat_avg_num += float(mm.read_latency_avg) * act
                    lat_avg_den += act
                    lat_p95_num += float(mm.read_latency_p99) * act
                    lat_p95_den += act

        mc_avg_latency = (lat_avg_num / lat_avg_den) if lat_avg_den > 0 else None
        mc_p95_latency = (lat_p95_num / lat_p95_den) if lat_p95_den > 0 else None

        power_avg_power = (power_energy / float(total_cycles)) if total_cycles > 0 else 0.0

        avg_temp_all = (sum(temps) / float(len(temps))) if temps else 0.0
        q_p95 = _percentile_nearest_rank(queue_p99_samples, 95.0) if queue_p99_samples else 0.0

        # Compose
        agg = AggregatedMetrics(
            cycles=total_cycles,
            start_cycle=wstart,
            end_cycle=wend,
            produced_flits_total=produced_flits_total,
            flits_tx_main=produced_flits_total,
            flits_tx_esc=0,
            xbar_switches=0,
            mc_served_requests=mc_served,
            mc_avg_latency=mc_avg_latency,
            mc_p95_latency=mc_p95_latency,
            power_total_energy=float(power_energy),
            power_avg_power=float(power_avg_power),
            max_temp_any_tile=float(max_temp),
            avg_temp_all_tiles=float(avg_temp_all),
            queue_depth_p95_proxy=float(q_p95),
        )

        # Advance window
        self._win_start = wstart + int(self.spec.stride_cycles)
        # Drop fully consumed frames preceding new window start to keep buffer bounded
        self._drop_prefix_before(self._win_start)

        return agg

    # Injected via helper below (kept out of public API)
    def _drop_prefix_before(self, cycle: int) -> None:  # type: ignore[empty-body]
        pass


def _slice_frames_by_cycle(frames: List[TelemetryFrame], start: int, end: int) -> List[TelemetryFrame]:
    """Return frames with t_start_cycle ≥ start and t_end_cycle ≤ end.
    Assumes frames are non-overlapping and cycle_window small (E1 uses 1).
    """
    out: List[TelemetryFrame] = []
    for f in frames:
        ts = int(f.t_start_cycle)
        te = int(f.t_end_cycle)
        if ts >= start and te <= end:
            out.append(f)
    return out


def _percentile_nearest_rank(samples: Iterable[float], p: float) -> float:
    """Deterministic nearest-rank percentile. p in [0,100]."""
    data = sorted(float(x) for x in samples)
    n = len(data)
    if n == 0:
        return 0.0
    if p <= 0.0:
        return float(data[0])
    if p >= 100.0:
        return float(data[-1])
    # nearest-rank (ceil(p/100 * n)), 1-indexed
    r = (p / 100.0) * n
    idx = int(r)
    if r - idx > 0.0:
        idx += 1
    idx = max(1, min(n, idx))
    return float(data[idx - 1])


def _frames_before(frames: List[TelemetryFrame], cycle: int) -> int:
    """Count frames whose t_end_cycle ≤ cycle."""
    c = 0
    for f in frames:
        if int(f.t_end_cycle) <= cycle:
            c += 1
        else:
            break
    return c


def _agg_drop_prefix_before(self: TelemetryAggregator, cycle: int) -> None:
    k = _frames_before(self._frames, cycle)
    if k > 0:
        del self._frames[:k]


# Bind helper as method without exposing in __all__
TelemetryAggregator._drop_prefix_before = _agg_drop_prefix_before  # type: ignore[attr-defined]