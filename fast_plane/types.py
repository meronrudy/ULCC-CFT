# Fast-plane core types for E1 (JSON configs per sim/LAYOUT.md; API per sim/API_SURFACE.md)
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Tuple, Optional, List


class VC(IntEnum):
    """Virtual channels. VC0 is reserved Escape per E1."""
    ESC = 0
    MAIN = 1

    @property
    def is_escape(self) -> bool:
        return self is VC.ESC


class Dir(IntEnum):
    """Mesh directions including LOCAL port."""
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    LOCAL = 4

    @staticmethod
    def delta(d: "Dir") -> Tuple[int, int]:
        if d == Dir.NORTH:
            return (0, -1)
        if d == Dir.SOUTH:
            return (0, 1)
        if d == Dir.EAST:
            return (1, 0)
        if d == Dir.WEST:
            return (-1, 0)
        return (0, 0)

    @staticmethod
    def from_offset(dx: int, dy: int) -> "Dir":
        if dx > 0:
            return Dir.EAST
        if dx < 0:
            return Dir.WEST
        if dy > 0:
            return Dir.SOUTH
        if dy < 0:
            return Dir.NORTH
        return Dir.LOCAL


@dataclass(frozen=True)
class Coord:
    x: int
    y: int

    def add(self, d: Dir) -> "Coord":
        dx, dy = Dir.delta(d)
        return Coord(self.x + dx, self.y + dy)


@dataclass(frozen=True)
class Message:
    """Message metadata segmented into flits."""
    msg_id: int
    src: int
    dst: int
    vc: VC
    size_flits: int  # total flits including head/tail

    def __post_init__(self) -> None:
        if self.size_flits < 1:
            raise ValueError("size_flits must be >= 1")


@dataclass(frozen=True)
class Flit:
    """Atomic flow unit on links; carries message association and VC."""
    msg_id: int
    src: int
    dst: int
    seq: int  # 0..size_flits-1
    size_flits: int
    is_head: bool
    is_tail: bool
    vc: VC

    def __post_init__(self) -> None:
        if self.seq < 0:
            raise ValueError("flit.seq must be non-negative")
        if self.size_flits < 1:
            raise ValueError("flit.size_flits must be >= 1")


@dataclass
class StepOutcome:
    """Aggregate outcome for fast-plane step() per sim/API_SURFACE.md."""
    flits_transferred: int
    avg_queue_depth: Dict[str, float]
    stalls_by_cause: Dict[str, int]
    deadlock_flag: bool


@dataclass
class MeshShape:
    width: int
    height: int

    def tile_id(self, c: Coord) -> int:
        return c.y * self.width + c.x

    def coord_of(self, tile_id: int) -> Coord:
        x = tile_id % self.width
        y = tile_id // self.width
        return Coord(x, y)

    def in_bounds(self, c: Coord) -> bool:
        return 0 <= c.x < self.width and 0 <= c.y < self.height


@dataclass
class RouterWeights:
    """Weighted ECMP next hop probabilities for MAIN VC. If absent, fall back to DO (XY)."""
    north: int = 0
    south: int = 0
    east: int = 0
    west: int = 0
    local: int = 0

    def as_dict(self) -> Dict[Dir, int]:
        return {
            Dir.NORTH: int(self.north),
            Dir.SOUTH: int(self.south),
            Dir.EAST: int(self.east),
            Dir.WEST: int(self.west),
            Dir.LOCAL: int(self.local),
        }


@dataclass
class NoCParams:
    """NoC timing and buffer parameters."""
    mesh: MeshShape
    buffer_depth_per_vc: int = 8  # flits
    vcs: int = 2  # must be 2 in E1: ESC and MAIN
    router_pipeline_latency: int = 1  # cycles from input to output
    link_latency: int = 1  # per-hop link latency in cycles
    link_width_bytes: int = 16  # serialization width (E1 proxy)
    flit_bytes: int = 16  # 1 flit per cycle at width 16B for simplicity
    esc_vc_id: int = int(VC.ESC)  # escape VC index
    rng_seed: int = 0  # deterministic seed for any randomized policies (reserved)

    def validate(self) -> None:
        if self.vcs < 2:
            raise ValueError("E1 requires at least 2 VCs (ESC and MAIN)")
        if self.buffer_depth_per_vc < 1:
            raise ValueError("buffer_depth_per_vc must be >= 1")
        if self.flit_bytes <= 0 or self.link_width_bytes <= 0:
            raise ValueError("positive widths required")
        if self.esc_vc_id != int(VC.ESC):
            raise ValueError("escape VC id must be 0 in E1")


@dataclass
class TokenBucketParams:
    """Producer/consumer token bucket parameters."""
    rate_tokens_per_cycle: float
    burst_size: int
    msg_size_flits: int

    def validate(self) -> None:
        if self.rate_tokens_per_cycle < 0.0:
            raise ValueError("rate must be non-negative")
        if self.burst_size < 1:
            raise ValueError("burst_size must be >= 1")
        if self.msg_size_flits < 1:
            raise ValueError("msg_size_flits must be >= 1")


@dataclass
class MPKIParams:
    """Cache/memory miss model parameters."""
    mpki: float  # misses per 1k instructions
    ipc_proxy: float  # instructions per cycle proxy

    def validate(self) -> None:
        if self.mpki < 0.0:
            raise ValueError("mpki must be non-negative")
        if self.ipc_proxy < 0.0:
            raise ValueError("ipc_proxy must be non-negative")


@dataclass
class MCParams:
    """Memory controller timing proxy (FR-FCFS approximation)."""
    bank_count: int = 8
    channel_count: int = 2
    t_row_hit: int = 40
    t_row_miss: int = 200
    t_bus: int = 20
    reorder_window: int = 8  # small window for FR-FCFS reordering

    def validate(self) -> None:
        if self.bank_count < 1 or self.channel_count < 1:
            raise ValueError("bank/channel counts must be >= 1")
        if min(self.t_row_hit, self.t_row_miss, self.t_bus) < 1:
            raise ValueError("timings must be >= 1 cycle")


# Utility: deterministic weighted choice without external RNG dependency
def deterministic_weighted_choice(weights: Dict[Dir, int], seed: int) -> Dir:
    """Pick a Dir deterministically from weights and seed by modular hashing."""
    total = sum(max(0, w) for w in weights.values())
    if total == 0:
        # degenerate: choose LOCAL if available else NORTH by convention
        return Dir.LOCAL if weights.get(Dir.LOCAL, 0) else Dir.NORTH
    # Mix seed to get stable position in [0, total-1]
    # Use a simple LCG-like scramble to avoid dependency on random module.
    x = (seed ^ 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 30)) & 0xFFFFFFFFFFFFFFFF
    pos = int(x % total)
    cumulative = 0
    for d in (Dir.NORTH, Dir.SOUTH, Dir.EAST, Dir.WEST, Dir.LOCAL):
        w = max(0, int(weights.get(d, 0)))
        if pos < cumulative + w:
            return d
        cumulative += w
    return Dir.LOCAL


# === E1a additional minimal aliases/types for NoC ===
from typing import NewType

# A port is identified by (tile_id, direction)
PortID = Tuple[int, Dir]

# Credits represent available buffer slots for a given VC
Credit = NewType("Credit", int)

@dataclass(frozen=True)
class RouterCoords:
    tile_id: int
    coord: Coord

# Per-router routing weights map for MAIN VC
RoutingWeights = Dict[int, RouterWeights]

# Backwards-compatible alias for config naming used in docs/tests
NoCConfig = NoCParams