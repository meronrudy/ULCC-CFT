"""
CSV emitter for AggregatedMetrics windows (Phase A — E2)

Deterministic behavior:
- Stable column order (explicit header list, not dict iteration).
- Fixed float formatting with 6 decimal places.
- None values serialized as empty string.

This module intentionally has no external dependencies beyond stdlib.
"""

from __future__ import annotations
import csv
from typing import Iterable, List, TextIO

from .aggregator import AggregatedMetrics


# Stable, explicit header order
_HEADER: List[str] = [
    "cycles",
    "start_cycle",
    "end_cycle",
    "produced_flits_total",
    "flits_tx_main",
    "flits_tx_esc",
    "xbar_switches",
    "mc_served_requests",
    "mc_avg_latency",
    "mc_p95_latency",
    "power_total_energy",
    "power_avg_power",
    "max_temp_any_tile",
    "avg_temp_all_tiles",
    "queue_depth_p95_proxy",
]


def _fmt(val) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        # Fixed precision for determinism
        return f"{val:.6f}"
    return str(val)


def write_aggregates_csv(windows: Iterable[AggregatedMetrics], fp: TextIO) -> int:
    """
    Write AggregatedMetrics rows to CSV with deterministic header and formatting.

    Returns:
        int: number of rows written (excluding header).
    """
    writer = csv.writer(fp, lineterminator="\n")
    writer.writerow(_HEADER)

    count = 0
    for w in windows:
        row = [
            _fmt(w.cycles),
            _fmt(w.start_cycle),
            _fmt(w.end_cycle),
            _fmt(w.produced_flits_total),
            _fmt(w.flits_tx_main),
            _fmt(w.flits_tx_esc),
            _fmt(w.xbar_switches),
            _fmt(w.mc_served_requests),
            _fmt(w.mc_avg_latency),
            _fmt(w.mc_p95_latency),
            _fmt(w.power_total_energy),
            _fmt(w.power_avg_power),
            _fmt(w.max_temp_any_tile),
            _fmt(w.avg_temp_all_tiles),
            _fmt(w.queue_depth_p95_proxy),
        ]
        writer.writerow(row)
        count += 1
    return count