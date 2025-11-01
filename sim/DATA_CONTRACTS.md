# DATA CONTRACTS: Phase A Simulator

Overview

This document defines normative data contracts for Phase A artifacts: GeometryTable, TelemetryFrame, and ReconfigPack. It specifies types, units, ranges, precision, tiling, integrity, and validation rules. These contracts are language-agnostic and intended to be serialized in JSON or CBOR with canonical ordering. Cross-references: conceptual numerics in [spec-first/field/wave.py](spec-first/field/wave.py), [spec-first/geom/cfe_update.py](spec-first/geom/cfe_update.py), and attribution sampling in [spec-first/pggs/sampler.py](spec-first/pggs/sampler.py). Control-plane usage is defined in [sim/CONTROL_PROTOCOL.md](sim/CONTROL_PROTOCOL.md); API bindings in [sim/API_SURFACE.md](sim/API_SURFACE.md).

Serialization and integrity

- Canonical encoding: CBOR preferred for runtime; JSON for logs and CI baselines. Keys must use lower_snake_case.
- Endianness: integers are big-endian when emitted in binary blobs; text encodings unaffected.
- Integrity:
  - Tile-level CRC: CRC-32C (Castagnoli) polynomial 0x1EDC6F41 for GeometryTable tiles and ReconfigPack sections.
  - Frame-level CRC: same polynomial across the full buffer excluding crc field.
- Versioning: semver string major.minor.patch per schema; parsers must reject unknown major versions and may accept compatible minor versions.

Common scalar types

- u8, u16, u32: unsigned integer widths
- i16, i32: signed integer widths
- f32, f64: IEEE-754 floats
- q format: fixed point qi.f where total bits equals i plus f, two’s complement. Example: q4.28 in a 32-bit lane.

Coordinate frames and units

- Time: cycles (fast plane), microseconds (control quick-check), seconds (wall clock in reports)
- Thermal: degrees Celsius
- Power proxy: arbitrary unit p.u. normalized to 1.0 at TDP proxy; define baseline in harness
- Geometry tensors: dimensionless in Phase A; SPD constraints apply numerically not physically

1) GeometryTable

Purpose

- Provide tiled metric tensors g and their inverse g_inv, plus local Christoffel symbols for a 2D metric-aware stencil. Produced by slow_plane.geometry and consumed by slow_plane.packer and fast_plane policy compilation.

High-level structure

- geometry_table
  - schema_version: semver string
  - grid_shape: {width: u16, height: u16}
  - tile_shape: {tw: u8, th: u8} tile width and height in cells; all tiles uniform size except right or bottom edge partials
  - atlas_u_id: opaque id referencing PGGS atlas
  - cond_number_bound: f32 greater or equal to 1.0 applied post-projection
  - tiles: [GeometryTile]
  - checksum: crc32c over canonical CBOR of tiles
  - meta: {generator: string, created_at_utc: string ISO8601, run_id: string}

GeometryTile

- tile_id: u32 enumerated row-major from 0
- origin: {x: u16, y: u16} grid coordinates upper-left of tile in cells
- extent: {w: u8, h: u8} actual extent for edge tiles
- encoding: enum [cholesky_q4_28, full_f32] default cholesky_q4_28
- tensors: depends on encoding
  - cholesky_q4_28:
    - l11: [q4.28] length equals extent.w times extent.h
    - l21: [q4.28]
    - l22: [q4.28]
    - normalization: f32 scale s greater than 0.0 such that reconstructed g equals L times L transpose divided by s
  - full_f32:
    - g00: [f32], g01: [f32], g11: [f32] symmetric 2 by 2 per cell
- g_inv_hint: optional packed inverse metric with same encoding as tensors; may be omitted and derived at consume time; if present must satisfy SPD and condition bound
- christoffel: optional compact per-cell tuple gamma00_0 gamma01_0 gamma11_0 gamma00_1 gamma01_1 gamma11_1 as f32 or q4.28 matching encoding; may be omitted and derived
- cond_number_est: f32 estimated condition number per tile
- spd_pass: bool indicating SPD projection acceptance
- tile_crc32c: u32 CRC-32C over tile payload excluding tile_crc32c

Encoding and numerics

- SPD guarantee: tiles must reconstruct SPD matrices after decoding. For cholesky encoding, SPD is guaranteed by construction; consumers shall verify positive diagonals and numerical stability.
- Condition clamp: max cond_number_est less than or equal to cond_number_bound; if above bound, spd_projection with damping must be applied upstream per [spec-first/geom/cfe_update.py](spec-first/geom/cfe_update.py).
- Bounds:
  - q4.28 representable numeric range approximately [-8, 8); recommended normalization ensures l11 and l22 in [2^-4, 2^3].
  - full_f32 paths must clamp eigenvalues to [lambda_min, lambda_max] where lambda_min is greater than 1e-6 and lambda_max is less than or equal to 1e6 times lambda_min.
- Monotonicity: not applicable; however update deltas must satisfy trust_region_radius constraints in control (see [sim/CONTROL_PROTOCOL.md](sim/CONTROL_PROTOCOL.md)).

Validation rules

- For every cell:
  - det(g) greater than 0.0
  - g equals g transpose within 1e-6 relative tolerance
  - if g_inv_hint present: norm2 of g times g_inv minus identity less than or equal to 1e-4
- Tile CRC must match recomputed CRC
- Topology: number of tiles equals ceil(width divided by tw) times ceil(height divided by th)
- Required fields present for declared encoding

JSON-like schema

```json
{
  "geometry_table": {
    "schema_version": "1.0.0",
    "grid_shape": {"width": 256, "height": 256},
    "tile_shape": {"tw": 16, "th": 16},
    "atlas_u_id": "U:2025-10-31T11:00:00Z:abc",
    "cond_number_bound": 1e5,
    "tiles": [
      {
        "tile_id": 0,
        "origin": {"x": 0, "y": 0},
        "extent": {"w": 16, "h": 16},
        "encoding": "cholesky_q4_28",
        "tensors": {
          "l11": ["q4.28", "..."],
          "l21": ["q4.28", "..."],
          "l22": ["q4.28", "..."],
          "normalization": 0.5
        },
        "g_inv_hint": null,
        "christoffel": null,
        "cond_number_est": 1200.0,
        "spd_pass": true,
        "tile_crc32c": 305419896
      }
    ],
    "checksum": 267242409
  }
}
```

2) TelemetryFrame

Purpose

- Provide per-frame measurements of the fast plane and related proxies. Source of truth for PGGS, field, and control quick-check guardrails.

High-level structure

- telemetry_frame
  - schema_version: semver string
  - frame_id: string unique within run
  - run_id: string
  - rng_seed: u64 deterministic seed used during collection
  - grid_shape: {width: u16, height: u16}
  - cycle_window: u32 number of cycles covered by this frame
  - t_start_cycle: u64 inclusive
  - t_end_cycle: u64 exclusive; equals t_start_cycle plus cycle_window
  - sampling_mode: enum [instantaneous, windowed] windowed for aggregates
  - tile_metrics: [TileMetrics]
  - link_metrics: [LinkMetrics]
  - memory_metrics: [MemoryMetrics]
  - scheduler_metrics: SchedulerMetrics
  - power_thermal: PowerThermal
  - anomalies: [Anomaly] optional
  - frame_crc32c: u32
  - meta: {workload_tag: string, notes: string}

TileMetrics

- tile_id: u32
- flit_tx: u64 total flits transmitted during window
- flit_rx: u64 total flits received during window
- vc_depth_avg: [f32] average depth per VC in flits
- vc_depth_p99: [f32] 99th percentile depth per VC in flits
- queue_depth_avg: f32
- queue_depth_p99: f32
- service_time_avg: f32 cycles
- service_time_var: f32 cycles squared
- stalls: {credit_starve: u64, hazard: u64, mc_block: u64}
- mpki: f32 misses per kilo-instructions
- ipc: f32 instructions per cycle proxy
- temp_c: f32
- power_pu: f32

LinkMetrics

- link_id: u32
- endpoints: {from_tile: u32, to_tile: u32, vc_count: u8}
- utilization: f32 in 0 to 1 average over window
- flit_errors: u32 bit-errors or drop count
- backpressure_ratio: f32 in 0 to 1
- credit_underflows: u32

MemoryMetrics

- mc_id: u16
- queues: {rdq_depth_avg: f32, wrq_depth_avg: f32}
- fr_fcfs_hit: u64 row hits
- activations: u64
- precharges: u64
- bandwidth_util: f32 in 0 to 1
- read_latency_avg: f32 cycles
- read_latency_p99: f32 cycles
- throttles: u32 DVFS or policy throttling events

SchedulerMetrics

- runnable_tasks_avg: f32
- runnable_tasks_p99: f32
- migrations: u32
- preemptions: u32
- affinity_violations: u32 placements outside policy

PowerThermal

- tdp_proxy_pu: f32 in 0 to 2
- thermal_ceiling_hits: u32
- dvfs_state_counts: {state_id: u8 to u64}

Anomaly

- code: enum [crc_mismatch, deadlock_suspect, livelock_suspect, thermal_exceedance, power_exceedance, sampling_gap]
- severity: enum [info, warn, error]
- detail: string

Validation rules

- Monotonic fields: flit_tx flit_rx stalls and MC counters are non-decreasing across frames with non-overlapping windows; if windowed, report deltas and monotonicity constraint applies to cumulative roll-up not per-frame.
- Bounds:
  - utilization and backpressure_ratio in 0 to 1
  - temps in [-20, 125]
  - power_pu in 0 to 2
  - latencies and service times non-negative
- Integrity: frame_crc32c over canonical encoding excluding frame_crc32c
- Consistency: t_end_cycle equals t_start_cycle plus cycle_window; windows must not overlap within a run unless sampling_mode equals instantaneous

JSON-like schema

```json
{
  "telemetry_frame": {
    "schema_version": "1.0.0",
    "frame_id": "F00001",
    "run_id": "R42",
    "rng_seed": 0,
    "grid_shape": {"width": 8, "height": 8},
    "cycle_window": 10000,
    "t_start_cycle": 100000,
    "t_end_cycle": 110000,
    "sampling_mode": "windowed",
    "tile_metrics": [{"tile_id": 0, "flit_tx": 12345, "flit_rx": 12000, "vc_depth_avg": [1.2, 0.5], "vc_depth_p99": [3.8, 1.9], "queue_depth_avg": 2.3, "queue_depth_p99": 7.1, "service_time_avg": 18.0, "service_time_var": 5.0, "stalls": {"credit_starve": 10, "hazard": 2, "mc_block": 4}, "mpki": 4.7, "ipc": 0.9, "temp_c": 65.0, "power_pu": 0.82}],
    "link_metrics": [{"link_id": 17, "endpoints": {"from_tile": 2, "to_tile": 3, "vc_count": 2}, "utilization": 0.45, "flit_errors": 0, "backpressure_ratio": 0.2, "credit_underflows": 0}],
    "memory_metrics": [{"mc_id": 0, "queues": {"rdq_depth_avg": 8.1, "wrq_depth_avg": 2.0}, "fr_fcfs_hit": 100, "activations": 40, "precharges": 38, "bandwidth_util": 0.72, "read_latency_avg": 220.0, "read_latency_p99": 700.0, "throttles": 0}],
    "scheduler_metrics": {"runnable_tasks_avg": 14.2, "runnable_tasks_p99": 31.0, "migrations": 2, "preemptions": 6, "affinity_violations": 0},
    "power_thermal": {"tdp_proxy_pu": 0.91, "thermal_ceiling_hits": 0, "dvfs_state_counts": {"0": 8000, "1": 2000}},
    "anomalies": [],
    "frame_crc32c": 254395613
  }
}
```

3) ReconfigPack

Purpose

- Atomic configuration bundle for transitioning from current geometry and policies to a new configuration. Produced by slow_plane.packer and applied by control via shadow-apply, quick-check, quiesce, commit, verify, or rollback per [sim/CONTROL_PROTOCOL.md](sim/CONTROL_PROTOCOL.md).

High-level structure

- reconfig_pack
  - schema_version: semver string
  - pack_id: string unique
  - parent_commit_id: string or null
  - proposal_epoch: u64 monotonically increasing per run
  - geometry_ref: {geometry_table_id: string, cond_number_bound: f32}
  - trust_region_meta: {max_delta_norm: f32 in 0 to 1, accepted_prev: bool}
  - noc: NoCConfig
  - memory: MemoryConfig
  - cpu_mem: CpuMemConfig
  - dvfs: DVFSConfig
  - pr: PartialReconfig optional
  - apply_barrier: {vc_drain_threshold_flits: u16, timeout_us: u32}
  - guardrails_hint: GuardrailsHint optional
  - pack_crc32c: u32

NoCConfig

- mesh_shape: {width: u16, height: u16}
- router_tables: [RouterTable] sorted by tile_id
- vc_credits: [VCCredit] per tile vc
- link_weights: [LinkWeight] optional

RouterTable

- tile_id: u32
- next_hop_weights: {north: u16, south: u16, east: u16, west: u16, local: u16} weights for stochastic or weighted ECMP routing
- min_turn_penalty: u8 optional
- deadlock_escapes: [u8] virtual channels reserved for escape per deadlock-free routing constraint

VCCredit

- tile_id: u32
- vc_id: u8
- credits: u16 in 1 to 4096

LinkWeight

- link_id: u32
- weight: u16 in 1 to 65535

MemoryConfig

- mc_policy_words: [u32] bitfields for FR-FCFS approximations and refresh controls
- channel_throttle: [{channel_id: u8, max_bw_pu: f32 in 0 to 1}]
- cat_masks: [{l3_partition_id: u8, mask: u64 bitmask}] Intel CAT-like cache allocation tokens
- num_mc: u8 for validation

CpuMemConfig

- cpu_affinity: [{task_class: string, allowed_tiles: [u32]}]
- numa_policy: [{policy_id: u8, mode: enum [local_first, interleave, pinned], tiles: [u32]}]

DVFSConfig

- domains: [{domain_id: u8, tiles: [u32], state: u8, min_state: u8, max_state: u8}]
- transitions_allowed: bool

PartialReconfig

- bitstreams: [{region_id: u8, url: string or null, sha256: string, size_bytes: u32}]
- apply_mode: enum [staged, immediate]
- size_limit_bytes: u32 maximum total size; must be less than or equal to 32 MiB in Phase A

GuardrailsHint

- throughput_min_increase_pu: f32 desired improvement in p.u.
- tail_latency_p99_max_increase_pu: f32 allowed degradation in p.u.
- thermal_ceiling_allowance: u8 allowed hits during quick-check

Validation rules

- Atomicity: applying the pack must be all-or-nothing; control must treat pack_crc32c mismatch as fatal and reject.
- Ranges:
  - vc credits in 1 to 4096
  - link and hop weights in 1 to 65535
  - channel throttles in 0 to 1
  - trust_region_meta.max_delta_norm in 0 to 1
- Deadlock guard: router_tables must include escape VCs; control must verify deadlock-free turn model before commit.
- Compatibility: mesh_shape must match fast plane; geometry_ref must refer to a GeometryTable with matching grid and tile shapes or a documented projection.
- Versioning: incompatible major version changes require control to reject.
- CRC: pack_crc32c computed over entire canonical encoding excluding pack_crc32c; section-level CRCs recommended for noc memory cpu_mem dvfs pr subsections.

JSON-like schema

```json
{
  "reconfig_pack": {
    "schema_version": "1.0.0",
    "pack_id": "P:2025-10-31:001",
    "parent_commit_id": null,
    "proposal_epoch": 12,
    "geometry_ref": {"geometry_table_id": "G:abc", "cond_number_bound": 100000.0},
    "trust_region_meta": {"max_delta_norm": 0.25, "accepted_prev": true},
    "noc": {
      "mesh_shape": {"width": 8, "height": 8},
      "router_tables": [{"tile_id": 0, "next_hop_weights": {"north": 1, "south": 1, "east": 2, "west": 2, "local": 1}, "min_turn_penalty": 0, "deadlock_escapes": [0]}],
      "vc_credits": [{"tile_id": 0, "vc_id": 0, "credits": 16}],
      "link_weights": [{"link_id": 17, "weight": 10}]
    },
    "memory": {
      "mc_policy_words": [4294967295],
      "channel_throttle": [{"channel_id": 0, "max_bw_pu": 0.9}],
      "cat_masks": [{"l3_partition_id": 1, "mask": 65535}],
      "num_mc": 2
    },
    "cpu_mem": {
      "cpu_affinity": [{"task_class": "A", "allowed_tiles": [0, 1, 2]}],
      "numa_policy": [{"policy_id": 0, "mode": "local_first", "tiles": [0, 1, 2]}]
    },
    "dvfs": {
      "domains": [{"domain_id": 0, "tiles": [0, 1, 2, 3], "state": 1, "min_state": 0, "max_state": 3}],
      "transitions_allowed": true
    },
    "pr": {
      "bitstreams": [{"region_id": 0, "url": "file://bs0.bit", "sha256": "abc...", "size_bytes": 1048576}],
      "apply_mode": "staged",
      "size_limit_bytes": 33554432
    },
    "apply_barrier": {"vc_drain_threshold_flits": 4, "timeout_us": 2000},
    "guardrails_hint": {"throughput_min_increase_pu": 0.05, "tail_latency_p99_max_increase_pu": 0.02, "thermal_ceiling_allowance": 0},
    "pack_crc32c": 123456789
  }
}
```

Validation procedures

- GeometryTable.validate
  - decode tensors per encoding; reconstruct g; assert SPD and condition; verify optional g_inv_hint; verify CRC; verify coverage and tiling.
- TelemetryFrame.validate
  - window and time checks; bounds checks; monotonicity checks for cumulative counters; CRC verification.
- ReconfigPack.validate
  - schema version compat; mesh compatibility; deadlock-escape presence; range checks; trust-region compatibility with last commit geometry; CRC verification.

Operational constraints

- SPD projection and condition clamp must occur upstream during GeomUpdate; ReconfigPack consumers must not silently re-project.
- Trust-region: control enforces max_delta_norm across successive GeometryTable updates; exceeding bound triggers reject and optional rollback.
- Atomicity envelope: a pack must be applied under shadow then quiesced and committed; partial application is invalid and must be rolled back per [sim/CONTROL_PROTOCOL.md](sim/CONTROL_PROTOCOL.md).

References

- Field solver CFL and stability guidance: [spec-first/field/wave.py](spec-first/field/wave.py)
- CFE update SPD projection and damping: [spec-first/geom/cfe_update.py](spec-first/geom/cfe_update.py)
- PGGS sampling determinism hooks: [spec-first/pggs/sampler.py](spec-first/pggs/sampler.py)
- API calls producing and consuming these artifacts: [sim/API_SURFACE.md](sim/API_SURFACE.md)