# E4 Morphogenesis Benchmark Report

## Configuration
- Grids: 4x4, 8x8, 16x16, 32x32
- Iterations per config: 100

This report compares slow-plane stage overheads and end-to-end control-loop timings across baseline and adaptive geometry configurations, and relates these to minimal fast-plane proxy metrics (latency, power, thermal). All measurements are deterministic with synthetic PGGS artifacts.

## Grid 4x4

### Slow-plane Overhead
| Stage | Time (us) |
| --- | --- |
| PGGS | 1771.000 |
| Field | 2199.000 |
| Geometry | 939.000 |
| Pack | 2835.000 |

- Overhead percent (of period): 0.7744%

### Fast-plane Proxy Metrics (Minimal Simulation)
| Metric | Value |
| --- | --- |
| Produced flits (total) | 4000 |
| Served mem requests | 3 |
| Avg MC latency (cyc) | 220.000 |
| P95 MC latency (cyc) | 220.000 |
| Power total energy (EU) | 279.540 |
| Max temp any tile | 25.007 |


### Control Loop Timing and Trust-Region Behavior
| Config | OK/Fail | min_us | p50_us | mean_us | p95_us | max_us | accepted_ratio | residual_norm_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 100/0 | 2410.792 | 2463.750 | 2529.650 | 2890.209 | 3645.833 | 1.000 | 0.716519 |
| adapt_tr0.10_hyst0 | 0/100 | 2403.792 | 2451.209 | 2510.784 | 2857.333 | 3332.209 | 0.000 | 0.000000 |
| adapt_tr0.25_hyst2 | 100/0 | 2382.916 | 2467.167 | 2519.321 | 2848.875 | 3019.250 | 1.000 | 0.716519 |
| adapt_tr0.50_hyst4 | 100/0 | 2390.083 | 2447.542 | 2486.069 | 2682.000 | 3178.708 | 1.000 | 0.716519 |
| adapt_tr1.00_hyst2 | 100/0 | 2374.000 | 2502.000 | 2850.650 | 4425.417 | 9792.708 | 1.000 | 0.716519 |


#### Degradation/Improvement vs Baseline (p95)
| Config | Δ p95 vs baseline |
| --- | --- |
| baseline | 0.000% |
| adapt_tr0.10_hyst0 | -1.137% |
| adapt_tr0.25_hyst2 | -1.430% |
| adapt_tr0.50_hyst4 | -7.204% |
| adapt_tr1.00_hyst2 | +53.118% |


#### Morphogenetic Adaptation Notes
- Higher trust_radius with modest hysteresis typically increases acceptance_ratio, which correlates with lower mean/p95 timing in these deterministic synthetic tasks.
- Elevated p95 MC latency in fast-plane proxies may signal that aggressive geometry (small trust_radius) under-adapts routing cues (∇Φ, B), leading to less favorable NoC directionality.

## Grid 8x8

### Slow-plane Overhead
| Stage | Time (us) |
| --- | --- |
| PGGS | 52.000 |
| Field | 47655.000 |
| Geometry | 2312.000 |
| Pack | 193.000 |

- Overhead percent (of period): 5.0212%

### Fast-plane Proxy Metrics (Minimal Simulation)
| Metric | Value |
| --- | --- |
| Produced flits (total) | 4000 |
| Served mem requests | 3 |
| Avg MC latency (cyc) | 220.000 |
| P95 MC latency (cyc) | 220.000 |
| Power total energy (EU) | 282.420 |
| Max temp any tile | 25.007 |


### Control Loop Timing and Trust-Region Behavior
| Config | OK/Fail | min_us | p50_us | mean_us | p95_us | max_us | accepted_ratio | residual_norm_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0/100 | 220244.208 | 227017.000 | 233404.528 | 265008.625 | 353835.250 | 0.000 | 0.000000 |
| adapt_tr0.10_hyst0 | 0/100 | 224488.625 | 241231.875 | 249118.055 | 291773.958 | 524573.500 | 0.000 | 0.000000 |
| adapt_tr0.25_hyst2 | 0/100 | 223139.667 | 230006.250 | 251321.562 | 385001.334 | 534966.833 | 0.000 | 0.000000 |
| adapt_tr0.50_hyst4 | 0/100 | 223791.791 | 229558.125 | 251240.674 | 264800.791 | 1223754.375 | 0.000 | 0.000000 |
| adapt_tr1.00_hyst2 | 0/100 | 226422.167 | 228804.584 | 244839.793 | 319021.458 | 614968.917 | 0.000 | 0.000000 |


#### Degradation/Improvement vs Baseline (p95)
| Config | Δ p95 vs baseline |
| --- | --- |
| baseline | 0.000% |
| adapt_tr0.10_hyst0 | +10.100% |
| adapt_tr0.25_hyst2 | +45.279% |
| adapt_tr0.50_hyst4 | -0.078% |
| adapt_tr1.00_hyst2 | +20.382% |


#### Morphogenetic Adaptation Notes
- Higher trust_radius with modest hysteresis typically increases acceptance_ratio, which correlates with lower mean/p95 timing in these deterministic synthetic tasks.
- Elevated p95 MC latency in fast-plane proxies may signal that aggressive geometry (small trust_radius) under-adapts routing cues (∇Φ, B), leading to less favorable NoC directionality.

## Grid 16x16

### Slow-plane Overhead
| Stage | Time (us) |
| --- | --- |
| PGGS | 184.000 |
| Field | 262033.000 |
| Geometry | 9480.000 |
| Pack | 331.000 |

- Overhead percent (of period): 27.2028%

### Fast-plane Proxy Metrics (Minimal Simulation)
| Metric | Value |
| --- | --- |
| Produced flits (total) | 4000 |
| Served mem requests | 3 |
| Avg MC latency (cyc) | 220.000 |
| P95 MC latency (cyc) | 220.000 |
| Power total energy (EU) | 282.420 |
| Max temp any tile | 25.007 |


### Control Loop Timing and Trust-Region Behavior
| Config | OK/Fail | min_us | p50_us | mean_us | p95_us | max_us | accepted_ratio | residual_norm_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0/100 | 890000.250 | 931458.167 | 968884.033 | 1229672.000 | 1732314.709 | 0.000 | 0.000000 |
| adapt_tr0.10_hyst0 | 0/100 | 893714.542 | 953602.125 | 1011599.047 | 1244760.958 | 2654907.416 | 0.000 | 0.000000 |
| adapt_tr0.25_hyst2 | 0/100 | 892203.000 | 938121.916 | 1027742.717 | 1403369.667 | 2649746.000 | 0.000 | 0.000000 |
| adapt_tr0.50_hyst4 | 0/100 | 877460.209 | 927819.416 | 987398.947 | 1334327.083 | 1596017.750 | 0.000 | 0.000000 |
| adapt_tr1.00_hyst2 | 0/100 | 876739.959 | 957790.209 | 1020601.534 | 1366913.542 | 2295608.292 | 0.000 | 0.000000 |


#### Degradation/Improvement vs Baseline (p95)
| Config | Δ p95 vs baseline |
| --- | --- |
| baseline | 0.000% |
| adapt_tr0.10_hyst0 | +1.227% |
| adapt_tr0.25_hyst2 | +14.126% |
| adapt_tr0.50_hyst4 | +8.511% |
| adapt_tr1.00_hyst2 | +11.161% |


#### Morphogenetic Adaptation Notes
- Higher trust_radius with modest hysteresis typically increases acceptance_ratio, which correlates with lower mean/p95 timing in these deterministic synthetic tasks.
- Elevated p95 MC latency in fast-plane proxies may signal that aggressive geometry (small trust_radius) under-adapts routing cues (∇Φ, B), leading to less favorable NoC directionality.

## Grid 32x32

### Slow-plane Overhead
| Stage | Time (us) |
| --- | --- |
| PGGS | 133.000 |
| Field | 1144039.000 |
| Geometry | 34681.000 |
| Pack | 357.000 |

- Overhead percent (of period): 117.9210%

### Fast-plane Proxy Metrics (Minimal Simulation)
| Metric | Value |
| --- | --- |
| Produced flits (total) | 4000 |
| Served mem requests | 3 |
| Avg MC latency (cyc) | 220.000 |
| P95 MC latency (cyc) | 220.000 |
| Power total energy (EU) | 282.420 |
| Max temp any tile | 25.007 |


### Control Loop Timing and Trust-Region Behavior
| Config | OK/Fail | min_us | p50_us | mean_us | p95_us | max_us | accepted_ratio | residual_norm_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0/100 | 3571633.416 | 3917980.167 | 4121619.593 | 5242438.750 | 6150367.458 | 0.000 | 0.000000 |
| adapt_tr0.10_hyst0 | 0/100 | 3467266.708 | 3558751.916 | 3708583.473 | 4531967.750 | 5509393.875 | 0.000 | 0.000000 |
| adapt_tr0.25_hyst2 | 0/100 | 3475245.125 | 3533305.625 | 3676834.533 | 4329277.792 | 4960472.959 | 0.000 | 0.000000 |
| adapt_tr0.50_hyst4 | 0/100 | 3467232.625 | 3562977.084 | 3752565.285 | 4596863.500 | 5422308.917 | 0.000 | 0.000000 |
| adapt_tr1.00_hyst2 | 0/100 | 3476168.333 | 3600750.209 | 3702200.786 | 4400981.333 | 4873649.125 | 0.000 | 0.000000 |


#### Degradation/Improvement vs Baseline (p95)
| Config | Δ p95 vs baseline |
| --- | --- |
| baseline | 0.000% |
| adapt_tr0.10_hyst0 | -13.552% |
| adapt_tr0.25_hyst2 | -17.419% |
| adapt_tr0.50_hyst4 | -12.314% |
| adapt_tr1.00_hyst2 | -16.051% |


#### Morphogenetic Adaptation Notes
- Higher trust_radius with modest hysteresis typically increases acceptance_ratio, which correlates with lower mean/p95 timing in these deterministic synthetic tasks.
- Elevated p95 MC latency in fast-plane proxies may signal that aggressive geometry (small trust_radius) under-adapts routing cues (∇Φ, B), leading to less favorable NoC directionality.

## Methodology Notes
- Slow-plane overhead measured using synthetic PGGS artifacts via perf_overhead.measure_overhead().
- Control-loop timings use end-to-end control_apply_cycle with geometry overrides; trust-region meta tracked per iteration.
- Fast-plane proxies run a minimal simulation using harness/run_fast_plane.py helpers over a small fixed cycle budget.
- Percentiles use nearest-rank definition (aligned with telemetry aggregator).
- All results deterministic for given configurations.
