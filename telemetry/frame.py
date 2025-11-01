"""
TelemetryFrame and CRC-32C (Castagnoli) per sim/DATA_CONTRACTS.md

Canonical serialization for CRC:
- Use JSON with sorted keys and no whitespace (separators=(',', ':'), ensure_ascii=False).
- Encode as UTF-8 bytes.
- Exclude the 'frame_crc32c' field from the payload when computing/validating CRC.
- The canonical object root is {"telemetry_frame": <frame_dict>}.

CRC parameters:
- Polynomial: 0x1EDC6F41 (Castagnoli), reflected representation 0x82F63B78 for table-driven impl.
- Init: 0xFFFFFFFF, XOR-out: 0xFFFFFFFF, input bytes processed LSB-first (reflected).
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict, replace
from typing import List, Dict, Optional, Any, Tuple
import json

# ---- CRC-32C (Castagnoli) implementation ----
def _make_crc32c_table() -> Tuple[int, ...]:
    poly = 0x82F63B78  # reversed 0x1EDC6F41
    table: List[int] = []
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
        table.append(crc & 0xFFFFFFFF)
    return tuple(table)

_CRC32C_TABLE: Tuple[int, ...] = _make_crc32c_table()

def crc32c(data: bytes) -> int:
    """Compute CRC-32C (Castagnoli) over data bytes."""
    crc = 0xFFFFFFFF
    for b in data:
        idx = (crc ^ b) & 0xFF
        crc = (_CRC32C_TABLE[idx] ^ (crc >> 8)) & 0xFFFFFFFF
    return crc ^ 0xFFFFFFFF

# Top-level helpers required by API
def compute_crc32c(payload: bytes) -> int:
    return crc32c(payload)

def crc32c_bytes(payload: bytes) -> int:
    return crc32c(payload)

def canonical_encode(obj: Any) -> bytes:
    """Canonical JSON encoding for CRC purposes (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')

# ---- Data contracts (subset for E1) ----
@dataclass
class TileMetrics:
    tile_id: int
    flit_tx: int
    flit_rx: int
    vc_depth_avg: List[float]
    vc_depth_p99: List[float]
    queue_depth_avg: float
    queue_depth_p99: float
    service_time_avg: float  # cycles
    service_time_var: float  # cycles^2
    stalls: Dict[str, int]
    mpki: float
    ipc: float
    temp_c: float
    power_pu: float

@dataclass
class LinkEndpoints:
    from_tile: int
    to_tile: int
    vc_count: int

@dataclass
class LinkMetrics:
    link_id: int
    endpoints: LinkEndpoints
    utilization: float  # 0..1
    flit_errors: int
    backpressure_ratio: float  # 0..1
    credit_underflows: int

@dataclass
class MemoryQueuesDepth:
    rdq_depth_avg: float
    wrq_depth_avg: float

@dataclass
class MemoryMetrics:
    mc_id: int
    queues: MemoryQueuesDepth
    fr_fcfs_hit: int
    activations: int
    precharges: int
    bandwidth_util: float  # 0..1
    read_latency_avg: float  # cycles
    read_latency_p99: float  # cycles
    throttles: int

@dataclass
class SchedulerMetrics:
    runnable_tasks_avg: float
    runnable_tasks_p99: float
    migrations: int
    preemptions: int
    affinity_violations: int

@dataclass
class PowerThermal:
    tdp_proxy_pu: float  # 0..2
    thermal_ceiling_hits: int
    dvfs_state_counts: Dict[str, int]

@dataclass
class Anomaly:
    code: str  # enum [crc_mismatch, deadlock_suspect, ...]
    severity: str  # enum [info, warn, error]
    detail: str

@dataclass
class TelemetryFrame:
    schema_version: str
    frame_id: str
    run_id: str
    rng_seed: int
    grid_shape: Dict[str, int]  # {"width": u16, "height": u16}
    cycle_window: int  # u32
    t_start_cycle: int  # u64
    t_end_cycle: int  # u64
    sampling_mode: str  # "instantaneous" | "windowed"
    tile_metrics: List[TileMetrics]
    link_metrics: List[LinkMetrics]
    memory_metrics: List[MemoryMetrics]
    scheduler_metrics: SchedulerMetrics
    power_thermal: PowerThermal
    anomalies: List[Anomaly] = field(default_factory=list)
    frame_crc32c: Optional[int] = None
    meta: Dict[str, str] = field(default_factory=dict)

    def to_dict(self, include_crc: bool = True) -> Dict[str, Any]:
        d = asdict(self)
        if not include_crc:
            d.pop('frame_crc32c', None)
        return {'telemetry_frame': d}

    def to_bytes(self) -> bytes:
        """Canonical bytes for CRC: JSON UTF-8 of to_dict(include_crc=False) with sorted keys."""
        return canonical_encode(self.to_dict(include_crc=False))

    def compute_crc32c(self) -> int:
        d = self.to_dict(include_crc=False)
        return crc32c(canonical_encode(d))

    def with_crc(self) -> 'TelemetryFrame':
        crc = self.compute_crc32c()
        return replace(self, frame_crc32c=crc)

    def validate(self, previous: Optional['TelemetryFrame'] = None) -> Tuple[bool, Optional[str]]:
        # Consistency of window
        if self.t_end_cycle != self.t_start_cycle + int(self.cycle_window):
            return (False, "t_end_cycle must equal t_start_cycle + cycle_window")
        # Bounds
        w = int(self.grid_shape.get('width', 0))
        h = int(self.grid_shape.get('height', 0))
        if w <= 0 or h <= 0:
            return (False, "grid_shape must have positive width and height")
        for lm in self.link_metrics:
            if not (0.0 <= lm.utilization <= 1.0):
                return (False, "link.utilization out of bounds")
            if not (0.0 <= lm.backpressure_ratio <= 1.0):
                return (False, "link.backpressure_ratio out of bounds")
        for tm in self.tile_metrics:
            if not (-20.0 <= tm.temp_c <= 125.0):
                return (False, "tile.temp_c out of bounds")
            if not (0.0 <= tm.power_pu <= 2.0):
                return (False, "tile.power_pu out of bounds")
            if any(v < 0.0 for v in tm.vc_depth_avg):
                return (False, "vc_depth_avg negative")
            if any(v < 0.0 for v in tm.vc_depth_p99):
                return (False, "vc_depth_p99 negative")
        for mm in self.memory_metrics:
            if not (0.0 <= mm.bandwidth_util <= 1.0):
                return (False, "memory.bandwidth_util out of bounds")
            if mm.read_latency_avg < 0.0 or mm.read_latency_p99 < 0.0:
                return (False, "memory.latencies must be non-negative")
        if not (0.0 <= self.power_thermal.tdp_proxy_pu <= 2.0):
            return (False, "power_thermal.tdp_proxy_pu out of bounds")
        # Integrity
        expected = self.compute_crc32c()
        if self.frame_crc32c is None or int(self.frame_crc32c) != expected:
            return (False, "crc_mismatch")
        # Window monotonicity (no overlap)
        if previous is not None and previous.run_id == self.run_id:
            if previous.t_end_cycle > self.t_start_cycle:
                return (False, "sampling windows overlap within a run")
        return (True, None)

# Known-answer test for CRC-32C: "123456789" -> 0xE3069283
def _selftest_crc32c() -> None:
    kat = b"123456789"
    val = crc32c(kat)
    assert val == 0xE3069283, f"CRC-32C self-test failed, got {val:08X}"

# Run KAT on import to avoid silent mismatches
_selftest_crc32c()