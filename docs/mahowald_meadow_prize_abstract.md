# ULCC-CFT: Dual-Timescale Computational Field Theory for Neuromorphic Adaptive Control

Submission type: 4-page abstract (Mahowald Meadow Prize – Neuromorphic Engineering)

Team and affiliation (template)
- Team: __________________________________________
- Institution/Company: ____________________________
- Contact (email): ________________________________
- Track: Neuromorphic algorithms and infrastructure

Keywords: neuromorphic control, computational field theory, information geometry, deterministic networks-on-chip, adaptive systems, safety, invariance, energy awareness

Page 1 — Motivation and Framing

Neuromorphic systems increasingly face safety-critical deployment contexts (robotics, autonomous platforms, edge inference near actuation). Practical adaptive control must satisfy three invariants simultaneously: operational safety (no deadlocks/livelocks; guarded reconfiguration), structural invariance (stable geometry of information flow under adaptation), and energy awareness (power/thermal ceilings respected). ULCC-CFT addresses these with a dual-timescale architecture that fuses deterministic control on a fast plane with geometric learning on a slow plane.

CFT lens and dual-timescale introduction:
We model adaptation via Computational Field Theory (CFT): signals induce sources J on an information-geometric manifold whose metric g evolves by damped CFE steps; control derives hardware/software policy updates by packing the updated geometry into a ReconfigPack. A fast-plane executes the workload on a discrete NoC with deterministic scheduling; a slow-plane runs PGGS attribution, field solving, and geometry update every K cycles; a control-plane enforces a provable apply protocol (shadow→quick-check→quiesce→commit→verify→rollback) with CRC gating and watchdogs. The public API, timing, and data contracts are defined in [sim/API_SURFACE.md](sim/API_SURFACE.md) and the workplan in [sim/ROADMAP_PHASE_A.md](sim/ROADMAP_PHASE_A.md).

Deterministic control + geometric learning (what is different):
Unlike gradient-only or heuristic runtime tuners, ULCC-CFT separates concerns: the control-plane is strictly deterministic and auditable (seeded, state-machine protocol), while the learning plane computes geometry from telemetry using information geometry and field solvers. This separation allows hard guarantees (no surprise mutations; bounded overhead) alongside principled learning dynamics.

<!-- pagebreak -->

Page 2–3 — Method and Implementation

System overview (fast-plane, slow-plane, control-plane)
- Fast-plane: discrete-time, cycle-accurate NoC with credit-based flow control and a deterministic scheduler. Policy changes are applied only via committed ReconfigPacks. See [fast_plane/scheduler.py](fast_plane/scheduler.py) and the acceptance suite in [tests/fast_plane/test_token_bucket_producer.py](tests/fast_plane/test_token_bucket_producer.py).
- Slow-plane: PGGS attribution produces atlas U, sources J, and flux B; a metric-aware field solver computes Φ and ∇Φ; a single damped CFE geometry step updates g with SPD projection and condition clamps; packer compiles geometry and policy words to a ReconfigPack. See [slow_plane/pggs/pipeline.py](slow_plane/pggs/pipeline.py), [slow_plane/field/solver.py](slow_plane/field/solver.py), [slow_plane/geometry/update.py](slow_plane/geometry/update.py), and [slow_plane/packer/make.py](slow_plane/packer/make.py).
- Control-plane: a Geometry Control Unit (GCU) emulator implements the atomic apply protocol with CRC validation, quick-check micro-horizons, quiesce thresholds, commit/verify, and rollback. See [control/gcu.py](control/gcu.py) and the integration wrapper [harness/control_loop.py](harness/control_loop.py).

System diagram (from positioning strategy)
```mermaid
flowchart LR
  subgraph Fast-plane (per-cycle)
    Producers -->|flits| NoC[(Mesh NoC)]
    Scheduler[[Deterministic Scheduler]] --> Producers
    NoC --> Telemetry[TelemetryFrames]
  end
  subgraph Slow-plane (every K cycles)
    Telemetry --> PGGS[PGGS attribution: U,J,B]
    PGGS --> Field[Field solve: Φ, ∇Φ]
    Field --> Geometry[CFE geometry update (SPD, clamp)]
    Geometry --> Packer[ReconfigPack]
  end
  subgraph Control-plane
    Shadow[shadow_apply] --> QC[quick_check (μs)]
    QC --> Quiesce[quiesce (VC drain)]
    Quiesce --> Commit[commit]
    Commit --> Verify[verify (canaries)]
    Verify -- pass --> Active[Active config]
    Verify -- fail --> Rollback[rollback]
  end
  Packer --> Shadow
  Active -->|policies| NoC
```

Key technical components (with code references)
- Deterministic scheduler and producer gating: the run-queue is strictly ordered and producers are gated via a per-producer flag. See [python.Scheduler()](fast_plane/scheduler.py:49), its boundary step [python.Scheduler.step()](fast_plane/scheduler.py:153), mass gating helper [python.Scheduler._gate_all()](fast_plane/scheduler.py:227), and assignment application [python.Scheduler._apply_assignment()](fast_plane/scheduler.py:233).
- Discrete NoC and backpressure correctness: acceptance tests exercise token-bucket producers and credit invariants in [tests/fast_plane/test_token_bucket_producer.py](tests/fast_plane/test_token_bucket_producer.py).
- GCU apply protocol with CRC: configurable CRC enforcement and stage methods are in [python.GCUConfig()](control/gcu.py:33), [python.GCU.shadow_apply()](control/gcu.py:82), [python.GCU.quick_check()](control/gcu.py:102), [python.GCU.quiesce()](control/gcu.py:138), [python.GCU.commit()](control/gcu.py:165), [python.GCU.verify()](control/gcu.py:184), [python.GCU.rollback()](control/gcu.py:217), with CRC presence enforced by [python.GCU._validate_pack_crc()](control/gcu.py:307).
- Slow-plane numerics and packer: field solver and geometry update APIs are defined in [slow_plane/field/solver.py](slow_plane/field/solver.py) and [slow_plane/geometry/update.py](slow_plane/geometry/update.py); packer compiles to ReconfigPack in [slow_plane/packer/make.py](slow_plane/packer/make.py). Overhead and profiling utilities live in [python.measure_overhead()](slow_plane/perf_overhead.py:85) and CSV writer [python.write_overhead_csv()](slow_plane/perf_overhead.py:186).
- Control-loop integration: the end-to-end control application path is implemented as [python.control_apply_cycle()](harness/control_loop.py:125), which ties geometry outputs to packer, guardrails, and GCU.
- Policy guardrails and safety: guardrails validate packs against deadlock risks, fairness floors, and thermal/power ceilings as specified in [sim/CONTROL_PROTOCOL.md](sim/CONTROL_PROTOCOL.md) and implemented in [control/guardrails.py](control/guardrails.py) with CRC-checked telemetry contracts in [sim/DATA_CONTRACTS.md](sim/DATA_CONTRACTS.md).

Implementation reality (evidence in repository)
- Working control emulator with CRC validation: CRC presence is required when enabled (default True) in [python.GCUConfig()](control/gcu.py:33) and enforced by [python.GCU._validate_pack_crc()](control/gcu.py:307); commits are verified via canary probes in [python.GCU.verify()](control/gcu.py:184).
- Deterministic scheduler and producer gating: deterministic step boundary and gating semantics are enforced in [python.Scheduler.step()](fast_plane/scheduler.py:153), [python.Scheduler._gate_all()](fast_plane/scheduler.py:227), and [python.Scheduler._apply_assignment()](fast_plane/scheduler.py:233).
- Overhead measurement utilities: end-to-end slow-loop overhead is measured by [python.measure_overhead()](slow_plane/perf_overhead.py:85) and exported via [python.write_overhead_csv()](slow_plane/perf_overhead.py:186); tests in [tests/slow_plane/test_perf_overhead.py](tests/slow_plane/test_perf_overhead.py) gate against the ≤1% budget.
- Acceptance tests cover reality and determinism: smoke and long-run determinism tests are defined in [python.test_phaseA_e1_acceptance_smoke()](tests/acceptance/test_phaseA_e1_acceptance.py:105) and [python.test_phaseA_e1_determinism_long()](tests/acceptance/test_phaseA_e1_acceptance.py:132). Additional control-plane integration tests are provided in [tests/control/test_control_integration.py](tests/control/test_control_integration.py) and [tests/control/test_gcu.py](tests/control/test_gcu.py).
- Power/Thermal proxies: fast-plane energy awareness hooks are available in [fast_plane/power.py](fast_plane/power.py), with policy application mediated by ReconfigPack DVFS states.

Performance methodology (≥10% improvement; ≤1% overhead)
- Baseline: run an acceptance scenario with fixed policies (routing, CAT, MC arbitration, scheduler affinities, DVFS) using the harness and record throughput and p99 tail-latency metrics per [sim/TEST_PLAN.md](sim/TEST_PLAN.md).
- Adaptive run: enable the slow-loop every K cycles; invoke the control loop [python.control_apply_cycle()](harness/control_loop.py:125) to produce and apply ReconfigPacks with micro quick-checks and quiesce windows.
- Improvement criterion: compute delta(throughput) and delta(p99) between adaptive and baseline; accept if improvement ≥10% in either metric while preserving safety invariants (no deadlock/livelock, guardrails pass).
- Overhead criterion: measure slow-loop time shares with [python.measure_overhead()](slow_plane/perf_overhead.py:85) for the chosen (H,W,K) and confirm overhead_percent ≤ 1.0; export CSV via [python.write_overhead_csv()](slow_plane/perf_overhead.py:186) for CI dashboards.
- Determinism: seed control is enforced across planes; see API rules in [sim/API_SURFACE.md](sim/API_SURFACE.md) and repository rules in [AGENTS.md](AGENTS.md).

<!-- pagebreak -->

Page 4 — Impact and Hardware Translation

Neuromorphic applications and benefits
- Robust adaptive control for on-robot inference/control loops: deterministic apply protocol prevents unsafe mid-flight parameter changes while geometric learning optimizes flow under constraints.
- Event-driven sensing and spiking fabrics: the fast-plane abstraction maps to neuromorphic interconnects; slow-plane geometry acts as an energy-aware routing and scheduling field.
- Continual learning at the edge: information geometry stabilizes updates via SPD projections and trust-region bounds while respecting power/thermal budgets.

Hardware translation path via ReconfigPack and GCU semantics
- ReconfigPack is the hardware/software contract that carries routing tables, link weights, VC credits, MC policy words, CAT masks, CPU affinities, DVFS states, and trust-region metadata per [sim/DATA_CONTRACTS.md](sim/DATA_CONTRACTS.md). Control applies packs using the deterministic protocol in [sim/CONTROL_PROTOCOL.md](sim/CONTROL_PROTOCOL.md) and the emulator in [control/gcu.py](control/gcu.py).
- A practical path to hardware: (1) compile geometry to pack with [slow_plane/packer/make.py](slow_plane/packer/make.py); (2) validate with guardrails; (3) apply via [python.GCU.shadow_apply()](control/gcu.py:82)→[python.GCU.quick_check()](control/gcu.py:102)→[python.GCU.quiesce()](control/gcu.py:138)→[python.GCU.commit()](control/gcu.py:165)→[python.GCU.verify()](control/gcu.py:184) or rollback via [python.GCU.rollback()](control/gcu.py:217). This sequencing is compatible with reconfigurable fabrics and control MCUs.
- ReconfigPack to hardware control: router table updates, DVFS states, and CAT masks can be programmed through vendor APIs or microcode under the same apply sequencing; CRC fields and watchdogs provide auditability.

Broader applicability (SoCs, datacenters, edge)
- SoCs: on-die mesh NoCs with per-core DVFS and last-level cache partitioning benefit from deterministic control; the geometry acts as a unifying field for routing and resource allocation.
- Datacenters: leaf/spine fabrics and accelerator pods can adopt the same dual-timescale pattern; telemetry-driven geometry steers load and pacing, while control quick-check windows protect SLAs.
- Edge devices: tight power/thermal envelopes and safety constraints align well with bounded-overhead slow-loops and CRC-gated reconfiguration.

Future milestones and limitations
- Near-term milestones (Phase A→B): enrich field and PGGS backends; extend packers to vendor-specific control blocks; add multi-atlas geometry; calibrate quick-check fidelity; formalize canary probes. See [sim/ROADMAP_PHASE_A.md](sim/ROADMAP_PHASE_A.md) for current DoD and gates.
- Limitations: Phase A provides proxies for power/thermal and simplified memory models; physical DRAM timing, coherence, and PR bitstreams are out of scope for now as noted in [sim/ROADMAP_PHASE_A.md](sim/ROADMAP_PHASE_A.md).

Evaluation criteria mapping (explicit)
- Novelty: deterministic control-plane plus information-geometric learning-plane separation; CFT-derived geometry drives policy compilation (ReconfigPack) rather than direct knob twiddling; control quick-check micro-horizons provide a new, auditable safety envelope.
- Reality: working emulator and end-to-end control loop with CRC validation and watchdogs ([control/gcu.py](control/gcu.py), [python.control_apply_cycle()](harness/control_loop.py:125)); deterministic NoC scheduler and producer gating ([fast_plane/scheduler.py](fast_plane/scheduler.py)); acceptance tests for fast-plane and control ([tests/acceptance/test_phaseA_e1_acceptance.py](tests/acceptance/test_phaseA_e1_acceptance.py)).
- Performance: methodology and tooling to demonstrate ≥10% improvement with ≤1% slow-loop overhead ([python.measure_overhead()](slow_plane/perf_overhead.py:85), [tests/slow_plane/test_perf_overhead.py](tests/slow_plane/test_perf_overhead.py)); deterministic RNG/seed propagation per [sim/API_SURFACE.md](sim/API_SURFACE.md).
- Impact: neuromorphic safety (deterministic apply), invariance (SPD geometry clamps), energy awareness (DVFS/power proxies [fast_plane/power.py](fast_plane/power.py)); translation path to hardware via ReconfigPack semantics ([sim/DATA_CONTRACTS.md](sim/DATA_CONTRACTS.md)).

References and traceability
- Architecture and APIs: [sim/API_SURFACE.md](sim/API_SURFACE.md); Control protocol: [sim/CONTROL_PROTOCOL.md](sim/CONTROL_PROTOCOL.md); Data contracts: [sim/DATA_CONTRACTS.md](sim/DATA_CONTRACTS.md).
- Roadmap and acceptance gates: [sim/ROADMAP_PHASE_A.md](sim/ROADMAP_PHASE_A.md); Test plan and KPIs: [sim/TEST_PLAN.md](sim/TEST_PLAN.md); Repository rules: [AGENTS.md](AGENTS.md).

PDF note: This markdown uses explicit page breaks (<!-- pagebreak -->) and a native Mermaid diagram; it is ready for md→pdf conversion (≤4 MB) with any Mermaid-capable pipeline.