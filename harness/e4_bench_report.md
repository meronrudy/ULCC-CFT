# E4 Benchmark Report

## Configuration
- Grid: 8 x 8
- Iterations: 50
- slow_loop_period_us: 1000000

## Slow-plane Overhead
- t_pggs_us: 144.000
- t_field_us: 47939.000
- t_geometry_us: 2349.000
- t_pack_us: 284.000
- overhead_percent: 5.072%

| Stage   | Time (us) |
|---------|-----------:|
| PGGS    | 144.000 |
| Field   | 47939.000 |
| Geometry| 2349.000 |
| Pack    | 284.000 |

## Control Loop Timing Stats
- ok_count: 50
- fail_count: 0

| Metric | Value (us) |
|--------|-----------:|
| min    | 222489.208 |
| p50    | 226689.875 |
| mean   | 232736.794 |
| p95    | 243810.791 |
| max    | 434364.541 |

## Notes
- Deterministic artifacts and timing; results depend only on grid, iterations, and configs.
- Percentiles use nearest-rank method, consistent with telemetry.aggregator._percentile_nearest_rank().
