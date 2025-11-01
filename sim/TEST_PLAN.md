# TEST PLAN AND KPIs: Phase A Simulator

Purpose

- Define measurable, testable acceptance and verification criteria for Phase A.
- Specify KPI calculations, sampling windows, and confidence intervals.
- Provide acceptance tests, synthetic workloads, baselines, and a comprehensive test matrix.
- Document CI entry points and reporting schemas for reproducible evaluation.

References

- APIs: [sim/API_SURFACE.md](sim/API_SURFACE.md)
- Data contracts: [sim/DATA_CONTRACTS.md](sim/DATA_CONTRACTS.md)
- Control protocol: [sim/CONTROL_PROTOCOL.md](sim/CONTROL_PROTOCOL.md)
- End-to-end scaffold: [spec-first/tests/formal/test_end_to_end.py](spec-first/tests/formal/test_end_to_end.py)
- Numerics inspiration: [spec-first/field/wave.py](spec-first/field/wave.py), [spec-first/geom/cfe_update.py](spec-first/geom/cfe_update.py), [spec-first/pggs/sampler.py](spec-first/pggs/sampler.py)
- RNG, domain guards, symlink rules: [AGENTS.md](AGENTS.md)
- Example stability study context: [ulcc-expts/bernoulli_vs_ngd.py](ulcc-expts/bernoulli_vs_ngd.py:5)

Phase A acceptance gates

- Performance improvement: for a selected workload, either throughput improves by greater than or equal to 10 percent OR tail latency p99 improves by greater than or equal to 10 percent versus the fixed-policy baseline.
- Reconfig overhead: less than or equal to 1 percent of simulated time per slow-loop cycle (PGGS plus field plus geometry plus pack plus control).
- Stability: geometry updates converge with bounded oscillation (norm of successive geometry deltas decreases or remains within a small trust region envelope).
- Safety: deadlock equals 0, livelock equals 0; rollback_rate less than or equal to 20 percent during exploratory Phase A runs (informative threshold).
- Guardrails: thermal ceiling not exceeded; power proxy within documented limits.

KPI definitions and calculations

Notation

- All calculations are language-agnostic. Inputs derive from TelemetryFrame series and ControlFrame logs.
- Let frames be an ordered list of TelemetryFrame records as defined in [sim/DATA_CONTRACTS.md](sim/DATA_CONTRACTS.md).
- Let windows be non-overlapping sampling intervals unless explicitly sliding.

Throughput

- Definition: average flit throughput per cycle over window; optionally normalized to baseline.
- Formula:
  - throughput_per_frame equals sum over tiles of flit_tx divided by cycle_window.
  - throughput_pu equals throughput_per_frame divided by baseline_throughput_per_frame when normalized.
- Aggregation: mean across evaluation window; report mean and 95 percent CI.
- Units: flits per cycle and p.u.

Tail latency p99

- Definition: latency p99 proxy, derived from read_latency_p99 for memory-bound workloads, or queue_depth_p99 plus service_time_avg heuristics for compute or NoC-bound scenarios.
- Formula:
  - memory_lat_p99 equals weighted average over memory controllers of read_latency_p99 with weights equal to activations.
  - noc_lat_proxy_p99 equals p99 over tiles of (queue_depth_p99 plus service_time_avg).
  - tail_latency_p99 equals memory_lat_p99 if memory bound else noc_lat_proxy_p99.
- Aggregation: median across evaluation window; report median and 95 percent CI.
- Units: cycles.

Reconfig overhead

- Definition: fraction of simulated time consumed by slow-plane and control per slow-loop.
- Formula:
  - overhead equals (t_pggs plus t_field plus t_geometry plus t_pack plus t_control) divided by slow_loop_period_cycles.
- Report: average overhead in percent with 95 percent CI.

Rollback rate

- Definition: fraction of proposals that result in rollback.
- Formula: rollback_rate equals rollbacks divided by proposals in percent.
- Report: point estimate and Wilson 95 percent interval.

Deadlock/livelock

- Deadlock: deadlock_flag equals true in any frame OR zero progress for D consecutive frames with D greater than or equal to 3 triggers failure.
- Livelock: progress oscillates without forward completion; proxy detected by cyclic channel dependency graph without drain combined with stable nonzero utilization over L greater than or equal to 5 frames.

Power proxy and thermal

- Power proxy:
  - power_pu_mean equals average over tiles of power_pu.
  - tdp_proxy_pu equals power_thermal.tdp_proxy_pu (global).
- Thermal:
  - thermal_ceiling_hits equals sum of thermal_ceiling_hits; must be 0 for acceptance windows.
  - max_temp_c equals max over tiles of temp_c; must be less than or equal to thermal_ceiling.

Stability of geometry updates

- Delta norm per slow-loop k:
  - delta_k equals Frobenius norm over all tiles of (g_k minus g_{k-1}) divided by Frobenius norm of g_{k-1}.
- Stability metric:
  - stability_norm equals exponential moving average of delta_k with alpha equals 0.3.
  - Acceptance: stability_norm decreases or remains less than or equal to trust_region_meta.max_delta_norm for last M equals 5 iterations.

Confidence intervals

- Continuous metrics (throughput, latency, overhead):
  - Nonparametric bootstrap with B equals 1000 resamples over frames.
  - Report 2.5 percentile and 97.5 percentile.
- Proportions (rollback_rate):
  - Wilson interval with z equals 1.96.
- Determinism:
  - For each scenario, evaluate across S equals 3 seeds; report across-seed mean and CI separately from within-run bootstrap CI.
  - Seeds are fixed and documented in configs for CI reproducibility per [AGENTS.md](AGENTS.md).

Workloads and baselines

Synthetic workload definitions

- W1 Uniform NoC: uniform random traffic among all tiles; token buckets configured to saturate NoC at approximately 60 percent.
- W2 Hotspot NoC: 10 percent of tiles act as sinks; 90 percent as sources; stress on central links.
- W3 Stride Memory: sequential read streams with MPKI approximately 4 to 6; FR-FCFS favorable.
- W4 Random Memory: random read streams with MPKI approximately 20; FR-FCFS less effective.
- W5 Mixed Compute-IO: 70 percent compute-bound tasks with low MPKI approximately 1; 30 percent memory-bound with MPKI approximately 15; includes NUMA affinity preferences.

Baseline policies

- Routing: deterministic dimension-ordered (escape VC always enabled), static equal weights; no geometry-driven routing.
- Memory controller: FR-FCFS approximation with static timing; no throttling beyond safety.
- CAT: default full-share mask.
- Scheduler: fixed affinities by task class; no dynamic adjustments.
- DVFS: fixed nominal state; transitions disabled.

Evaluation scenarios

- Acceptance scenario A:
  - Grid 8 by 8; buffers 8 flits per VC; 2 VCs with one escape VC; memory channels 2; workload W3 Random Memory.
  - Target: greater than or equal to 10 percent throughput improvement OR tail-latency p99 improvement vs baseline.
- Acceptance scenario B:
  - Grid 8 by 8; buffers 16; 3 VCs; workload W5 Mixed; ensure reconfig overhead less than or equal to 1 percent.

Test matrix

- Grid sizes: 4 by 4, 8 by 8, 12 by 12.
- Buffer depths per VC: 4, 8, 16.
- VC count: 2 with one escape, 3 with one escape.
- Memory channels: 1, 2, 4.
- Workloads: W1, W2, W3, W4, W5.
- Perturbation budgets per slow-loop: 8, 16, 32 perturbations.
- Slow-loop cadence (cycles): 10k, 50k, 100k.
- Trust-region radius: 0.1, 0.25, 0.4.
- Damping lambda: 0.3, 0.5, 0.7.
- Field method: leapfrog with CFL 0.8, conjugate-gradient with tolerance 1e-4.

Sampling windows and schedules

- Telemetry sampling_period cycles: default equals 10k cycles for 8 by 8 grids; scale proportional to grid size.
- Verification window (control verify): 10 times quick-check window, capped to less than or equal to 1 percent of slow-loop cadence per [sim/CONTROL_PROTOCOL.md](sim/CONTROL_PROTOCOL.md).
- KPI aggregation windows: exclusion of warmup frames (first 10 percent of run cycles) from KPI calculations.

Acceptance tests

- AT1 E2E improvement:
  - Run baseline and adaptive configurations on scenario A with three seeds.
  - Pass if throughput_mean_pu greater than or equal to 1.10 OR tail_latency_p99_mean less than or equal to 0.90 of baseline with 95 percent CI not crossing the threshold.
- AT2 Overhead bound:
  - Log t_pggs, t_field, t_geometry, t_pack, t_control per slow-loop; compute overhead; pass if mean less than or equal to 1 percent and upper 95 percent CI less than or equal to 2 percent.
- AT3 Safety:
  - Pass if deadlock equals 0, livelock equals 0; thermal_ceiling_hits equals 0; no frame anomalies with severity equals error.
- AT4 Stability:
  - Pass if stability_norm decreasing or bounded less than or equal to trust_region radius across last M equals 5 slow-loops; and projected_condition_number less than or equal to cond_number_bound.
- AT5 Rollback discipline:
  - During sweep runs (matrix), rollback_rate less than or equal to 20 percent; otherwise diagnostics must include quick-check calibration evidence and model-fidelity error logs.

CI and Make targets (document-only stubs)

- make sim:
  - Runs harness default scenario with single seed and writes metrics to artifacts directory.
- make sim-test:
  - Runs a minimal matrix: grid 4 by 4, buffers 8, workload W1 and W3, 1 seed.
- Pytest entry points:
  - pytest -q tests/test_acceptance.py::test_phase_a_e2e
  - pytest -q tests/test_guardrails.py::test_deadlock_free_quick_check
  - pytest -q tests/test_overhead.py::test_slow_loop_overhead_bound
- Artifacts directory layout:
  - artifacts/run_{run_id}/frames/*.cbor (TelemetryFrame CBOR)
  - artifacts/run_{run_id}/control/*.jsonl (ControlFrame)
  - artifacts/run_{run_id}/kpi/*.csv (aggregated KPIs)
  - artifacts/reports/*.html or *.md summaries

Metrics and dashboards

CSV schemas

- kpi_throughput.csv
  - Columns: run_id, scenario_id, seed, window_start_cycle, window_end_cycle, throughput_flits_per_cycle, throughput_pu, ci_low, ci_high
- kpi_latency.csv
  - Columns: run_id, scenario_id, seed, tail_latency_p99_cycles, ci_low, ci_high
- kpi_overhead.csv
  - Columns: run_id, scenario_id, slow_loop_idx, t_pggs, t_field, t_geometry, t_pack, t_control, slow_loop_period_cycles, overhead_percent
- kpi_safety.csv
  - Columns: run_id, scenario_id, deadlock_events, livelock_events, thermal_ceiling_hits, max_temp_c
- kpi_stability.csv
  - Columns: run_id, scenario_id, slow_loop_idx, delta_norm, stability_norm, projected_condition_number
- kpi_rollbacks.csv
  - Columns: run_id, scenario_id, proposals, rollbacks, rollback_rate_percent, ci_low, ci_high

Required plots

- Throughput over time: line plot of throughput_pu with commit points marked.
- Tail latency CDF: per scenario overlay for baseline vs adaptive.
- Overhead bars: stacked bars of t_pggs plus t_field plus t_geometry plus t_pack plus t_control as percent of period.
- Stability trend: delta_norm and stability_norm per slow-loop.
- Quick-check vs verify deltas: scatter plot with y equals verify delta, x equals quick-check delta.

Validation checks (pre flight)

- TelemetryFrame.validate must pass (CRC, bounds, monotonicity) before KPI ingestion.
- ReconfigPack.validate must pass before any control stage per [sim/DATA_CONTRACTS.md](sim/DATA_CONTRACTS.md).
- Control state transitions must follow [sim/CONTROL_PROTOCOL.md](sim/CONTROL_PROTOCOL.md); quick_check_report and verify_report must be present for each commit or rollback.

Determinism and seeds

- Seeds: define SEEDS equals [0, 1, 2] per scenario; embedded in run metadata and file paths for reproducibility.
- All randomized components must document rng_seed in TelemetryFrame and ControlFrame headers per [AGENTS.md](AGENTS.md).

Risk-based tests and mitigations

- Deadlock regression:
  - Inject adversarial weight maps lacking escapes; ensure control rejects with guardrail_violation before commit.
- Geometry instability:
  - Increase damping to 0.7 and reduce trust_region radius to 0.1 in stress scenarios; confirm stability_norm decreases.
- PGGS noise:
  - Reduce perturbation budget to 8; expect insufficient_signal and fallback; verify pack not produced and prior atlas reused.

Reporting structure

- Summary report (Markdown):
  - Executive summary of acceptance gates with pass fail.
  - Scenario-by-scenario KPI tables.
  - Plots listed above and raw CSV links.
  - Control decisions timeline with proposal and commit ids.

Exit criteria for Phase A (test perspective)

- AT1 to AT4 pass in CI on default scenario set with all seeds.
- No guardrail failures in logs across acceptance runs.
- All CSV schemas produced and validated; dashboards render required plots without missing data.