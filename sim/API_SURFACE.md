# API SURFACE: Phase A Simulator Architecture and Interfaces

Overview

This document specifies the Phase A module boundaries, typed cross-module interfaces, timing contracts, units, error semantics, and mapping from CFT IR operations to simulator calls. It aligns with repository practices in [AGENTS.md](AGENTS.md) and integrates with spec-first references in [spec-first/field/wave.py](spec-first/field/wave.py), [spec-first/geom/cfe_update.py](spec-first/geom/cfe_update.py), and [spec-first/pggs/sampler.py](spec-first/pggs/sampler.py).

Scope and cadence

- Fast plane: discrete-time cycle-accurate at NoC-link granularity; step size equals 1 cycle. Determinism enforced via fixed RNG seeds passed from the harness.
- Slow plane: executes every K cycles (configurable), with algorithmic steps for PGGS, field solve, and a single damped CFE geometry update per slow-loop iteration.
- Control plane: orchestrates shadow-apply, micro A B quick-check on a micro-horizon, quiesce, commit, verify, and rollback.

Module boundaries and responsibilities

cft_ir

- Role: host-agnostic IR for expressing actions and queries over the simulator.
- Types: GeomOp, FieldOp, DynOp, PGGSOp, GeomUpdateOp. Serialized as stable JSON records for persistency and CI.
- Semantics: IR records are idempotent requests; all operations are side-effect free until Control applies a ReconfigPack.

fast_plane

- Role: NoC mesh with credit-based flow control, routing table weights, simplified cores and accelerators as token-bucket producers consumers, cache memory as queuing model with MPKI-driven misses and FR-FCFS approximation, and a task-graph scheduler with affinities and priorities.
- Exposes: step, load_workload, set_policies, snapshot_telemetry.
- Determinism: controlled by harness-provided seeds and configuration. No hidden global state.

telemetry

- Role: collect PMU counters per tile and link, flit counts, queue depths, service times, MPKI estimates, memory controller stats, plus hooks for micro-perturbations.
- Exposes: capture_frame, reset, inject_perturbation.

slow_plane

- Role: PGGS attribution and sampling, field solver on a metric-aware 2D stencil, geometry update via damped CFE step with SPD projection and condition number clamp, and packer for compiling geometry to SoC knobs.
- Submodules: pggs, field, geometry, packer.

control

- Role: GCU emulator implementing shadow-apply, micro A B quick-check, quiesce, commit, verify, rollback, and watchdogs.
- Exposes: propose_reconfig, quick_check, quiesce_and_commit, verify, rollback, status.

policies

- Role: pluggable policy providers for routers, memory controller arbitration, CAT masks, scheduler affinities, and DVFS governors. Policy changes flow through Control after packing.

harness

- Role: scenario runner, seed control, metrics, reporting, and CI entry points. Defines workload timelines and slow-loop cadences for acceptance testing.

Cross-module API matrix

- Frequency classes
  - Per-cycle: fast-plane step, link credit updates, queues.
  - Per-slow-iteration: telemetry capture, PGGS, field solve, geometry update, pack build, control apply.
  - Out-of-band: load workload, set policies, save reports.
- Data contracts
  - TelemetryFrame, GeometryTable, ReconfigPack are defined normatively in [sim/DATA_CONTRACTS.md](sim/DATA_CONTRACTS.md).

Typed interfaces

Notation

- Types are language-agnostic. Cardinality uses [] for lists and {} for maps. Units and bounds are explicit. Errors return a code and message; no exceptions are assumed at the boundary.

cft_ir

IR records

- FieldOp
  - Inputs: sources_J ref, atlas_U ref, solver_params id, cadence slow_loop_step.
  - Output: field_solution_id ref to Phi and grad_Phi artifacts.
  - Errors: invalid_units, missing_dependency.
- PGGSOp
  - Inputs: telemetry_frame_id, budget perturb_n, rng_seed, attribution_mode shapley_or_gradient_proxy.
  - Output: atlas_U id, sources_J id, flux_B id.
  - Errors: insufficient_signal, budget_exhausted.
- GeomUpdateOp
  - Inputs: field_solution_id, damping in 0 to 1, cond_number_clamp min 1.0, trust_region_radius in 0 to 1, spd_projection on off.
  - Output: geometry_table_id and update_meta id.
  - Errors: spd_violation, trust_region_reject, cfl_violation.
- GeomOp
  - Inputs: geometry_table_id, policy_overrides optional.
  - Output: reconfig_pack_id.
  - Errors: pack_infeasible.
- DynOp
  - Inputs: duration_cycles, telemetry_sampling_period, slow_loop_period.
  - Output: run_id and summary metrics id.
  - Errors: deadline_miss.

fast_plane

step

- Purpose: advances NoC and cores by N cycles.
- Inputs
  - cycles N integer greater or equal to 1
  - dt cycle_time seconds per cycle constant over the run
  - rng_seed integer optional, used for stochastic service times if enabled
- Output
  - StepOutcome
    - flits_transferred total per step
    - avg_queue_depth per queue
    - stalls_by_cause map cause to count including credit_starve, hazard, mc_block
    - deadlock_flag boolean
- Errors
  - deadlock_detected if progress 0 for more than D cycles

load_workload

- Inputs
  - workload_spec ref id from harness; includes task graph, MPKI models, token bucket rates per core
- Output
  - ok boolean
- Errors
  - invalid_workload

snapshot_telemetry

- Inputs
  - sampling_window cycles integer; when 0, snapshot instantaneous
- Output
  - TelemetryFrame as defined in [sim/DATA_CONTRACTS.md](sim/DATA_CONTRACTS.md)
- Errors
  - sampling_error

set_policies

- Inputs
  - effective_policy_words derived from a ReconfigPack
- Output
  - applied boolean
- Errors
  - policy_conflict, guardrail_violation

telemetry

capture_frame

- Inputs
  - window_cycles integer greater or equal to 1
  - include_histograms boolean
- Output
  - TelemetryFrame
- Errors
  - integrity_failure crc_mismatch

inject_perturbation

- Inputs
  - target id tile or link
  - magnitude relative 0 to 1, duration cycles
- Output
  - perturbation_id
- Errors
  - out_of_range, unsafe_power

slow_plane.pggs

run_attribution

- Inputs
  - frames [TelemetryFrame] ordered, rng_seed integer, budget perturbations integer, proxy_mode shapley gradient
- Output
  - atlas_U, sources_J, flux_B all returned as identifiers and artifact metadata
- Errors
  - insufficient_variance, budget_exhausted
- Notes
  - Deterministic given frames and rng_seed. Aligns conceptually with [spec-first/pggs/sampler.py](spec-first/pggs/sampler.py).

slow_plane.field

solve

- Inputs
  - atlas_U from PGGS, sources_J from PGGS, grid_shape WxH, boundary_condition type, method leapfrog or cg, cfl_safety in 0.5 to 0.95 for explicit schemes
- Output
  - field_solution containing Phi scalar field and grad_Phi vector field with units as per [spec-first/field/wave.py](spec-first/field/wave.py)
- Errors
  - cfl_violation, divergence_detected
- Notes
  - Metric-aware stencil; verify CFL condition each call. Use double precision internally; export float32 fields plus checksums.

slow_plane.geometry

cfe_update

- Inputs
  - field_solution, damping lambda in 0 to 1, cond_clamp min 1.0 max 1e6, trust_region_radius in 0 to 1, spd_projection boolean
- Output
  - GeometryUpdate
    - geometry_table updated SPD tensors, inverse tensors, local Christoffel tiles
    - update_meta norms, acceptance_flag, projected_condition_number
- Errors
  - spd_violation_postproject, trust_region_reject
- Notes
  - Numerics align with [spec-first/geom/cfe_update.py](spec-first/geom/cfe_update.py) and stability rules in [AGENTS.md](AGENTS.md).

slow_plane.packer

build

- Inputs
  - geometry_table, policy_overrides optional
- Output
  - ReconfigPack including routing tables, VC credits, link weights, MC policy words, CAT masks, CPU affinities, NUMA policies, DVFS states, trust-region meta
- Errors
  - pack_infeasible, serialization_error

control

propose_reconfig

- Inputs
  - pack ReconfigPack, apply_mode shadow or commit, quick_check_horizon_us microseconds integer
- Output
  - proposal_id, accepted boolean
- Errors
  - guardrail_violation, quick_check_fail, watchdog_timeout

quick_check

- Inputs
  - proposal_id, horizon_us integer, acceptance_sla throughput tail_latency thresholds
- Output
  - quick_check_report including deltas versus baseline
- Errors
  - model_fidelity_error

quiesce_and_commit

- Inputs
  - proposal_id, vc_drain_threshold flits, timeout_us integer
- Output
  - commit_id
- Errors
  - quiesce_timeout, stale_proposal

verify

- Inputs
  - commit_id, canary_probes spec
- Output
  - verify_ok boolean
- Errors
  - post_commit_sla_violation

rollback

- Inputs
  - from_commit_id or proposal_id
- Output
  - rollback_id
- Errors
  - rollback_conflict

status

- Output
  - state enum idle shadow_applied committed rolling_back failed

policies

register_policies

- Inputs
  - router_policy_id, mc_policy_id, scheduler_policy_id, dvfs_policy_id
- Output
  - ok boolean

harness

run

- Inputs
  - scenario id, duration_cycles integer, rng_seed integer, sampling_period cycles, slow_loop_period cycles, acceptance thresholds
- Output
  - run_id, metrics_path ref, artifacts_path ref
- Errors
  - acceptance_failure if Phase A gates not met

IR op to simulator call mapping

- DynOp maps to harness.run which drives fast_plane.step and orchestrates slow-plane cadence with telemetry.capture_frame.
- PGGSOp maps to slow_plane.pggs.run_attribution over a batch of TelemetryFrame instances.
- FieldOp maps to slow_plane.field.solve producing Phi and grad_Phi artifact ids.
- GeomUpdateOp maps to slow_plane.geometry.cfe_update returning a GeometryUpdate with SPD guarantees.
- GeomOp maps to slow_plane.packer.build producing a ReconfigPack; Control then handles application lifecycle.

Timing and performance budgets

- Slow-loop overhead target: less than or equal to 1 percent of simulated time per slow-loop cycle including PGGS, field, geometry, pack, and control quick-check.
- Fast plane must report deadlock free operation; deadlock_flag must be false; otherwise Control triggers rollback and aborts run.

Units and coordinate frames

- Time units: cycles in fast plane, microseconds in control quick-check, seconds for wall-clock reporting.
- Fields: Phi is scalar potential; grad_Phi is vector with grid components; atlas_U captures chart parameters used by the metric-aware stencil.

Error taxonomy

- invalid_units
- missing_dependency
- insufficient_signal
- budget_exhausted
- cfl_violation
- divergence_detected
- spd_violation
- spd_violation_postproject
- trust_region_reject
- pack_infeasible
- serialization_error
- guardrail_violation
- quick_check_fail
- quiesce_timeout
- stale_proposal
- post_commit_sla_violation
- watchdog_timeout
- deadlock_detected
- sampling_error
- integrity_failure

Determinism and guards

- RNG: all randomized components accept rng_seed from harness; reproducibility is mandatory and audited in CI.
- Domain constraints: geometry tensors must remain SPD with condition number within configured clamp; service models must avoid zero-time or infinite-time pathologies.
- Symlink rule: respect import namespace indirection described in [AGENTS.md](AGENTS.md) during integration, though documents here remain language agnostic.

Acceptance hooks

- The harness computes throughput and latency percentiles, reconfig overhead, rollback rate, deadlock livelock status, power proxy, thermal ceilings, and stability norm of successive geometry deltas as specified in [sim/TEST_PLAN.md](sim/TEST_PLAN.md).
- Control exposes SLA thresholds to quick_check and verify to gate commit or trigger rollback.

References

- Field solver conceptual reference: [spec-first/field/wave.py](spec-first/field/wave.py)
- Geometry update conceptual reference: [spec-first/geom/cfe_update.py](spec-first/geom/cfe_update.py)
- PGGS sampling conceptual reference: [spec-first/pggs/sampler.py](spec-first/pggs/sampler.py)
- Stability invariants and repository rules: [AGENTS.md](AGENTS.md)