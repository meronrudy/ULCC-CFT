
from __future__ import annotations

"""
Lightweight deterministic power proxy for Phase A.

Units (proxies, not physical):
- Energy unit: arbitrary "energy units" (EU) per event (flit/switch/core-issue equivalent).
- Power unit: EU per cycle (since dt = 1 cycle per step).
- Energy accumulates as a simple sum of per-cycle power (EU) with dt=1 implicit.

Activity fields (per cycle):
- flits_tx_main: number of MAIN VC flits transmitted on links (credits consumed) this cycle.
- flits_tx_esc: number of ESC VC flits transmitted on links this cycle.
- xbar_switches: number of successful crossbar transfers (including LOCAL deliveries) this cycle.
- core_issue_est: producer-side flit-equivalents injected into LOCAL this cycle.

Determinism:
- No randomness; purely arithmetic accumulation.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PowerConfig:
    # Energy per event (proxy units)
    e_flit_main: float
    e_flit_esc: float
    e_router_xbar: float
    e_core_issue: float = 0.0
    # Optional sampling window (cycles) for reporting; accumulation remains per-cycle
    sampling_window_cycles: int = 1

    def __post_init__(self) -> None:
        if self.sampling_window_cycles < 1:
            raise ValueError("sampling_window_cycles must be >= 1")
        for v in (self.e_flit_main, self.e_flit_esc, self.e_router_xbar, self.e_core_issue):
            if v < 0.0:
                raise ValueError("energy coefficients must be non-negative")


class PowerProxy:
    """
    Per-tile power proxy.

    step(activity) computes instantaneous dynamic power proxy:
      p_dyn = e_flit_main*flits_tx_main
            + e_flit_esc*flits_tx_esc
            + e_router_xbar*xbar_switches
            + e_core_issue*core_issue_est

    State:
      - power_inst: last computed instantaneous power (EU/cycle)
      - energy_accum: total accumulated energy (EU) since creation or last reset
      - last_activity: last activity dict for introspection
    """

    def __init__(self, config: PowerConfig) -> None:
        self.cfg = config
        self.power_inst: float = 0.0
        self.energy_accum: float = 0.0
        self._window_ctr: int = 0
        self.last_activity: Dict[str, float] = {
            "flits_tx_main": 0.0,
            "flits_tx_esc": 0.0,
            "xbar_switches": 0.0,
            "core_issue_est": 0.0,
        }

    def step(self, activity: Dict[str, float]) -> None:
        # Normalize and clamp to non-negative
        fm = float(max(0.0, activity.get("flits_tx_main", 0.0)))
        fe = float(max(0.0, activity.get("flits_tx_esc", 0.0)))
        xs = float(max(0.0, activity.get("xbar_switches", 0.0)))
        ci = float(max(0.0, activity.get("core_issue_est", 0.0)))
        # Deterministic arithmetic
        p = (
            self.cfg.e_flit_main * fm
            + self.cfg.e_flit_esc * fe
            + self.cfg.e_router_xbar * xs
            + self.cfg.e_core_issue * ci
        )
        self.power_inst = float(p)
        # Accumulate energy with dt = 1 cycle
        self.energy_accum += self.power_inst
        self._window_ctr = (self._window_ctr + 1) % self.cfg.sampling_window_cycles
        # Stash last activity for counters
        self.last_activity = {
            "flits_tx_main": fm,
            "flits_tx_esc": fe,
            "xbar_switches": xs,
            "core_issue_est": ci,
        }

    def reset_energy(self) -> None:
        self.energy_accum = 0.0
        self._window_ctr = 0


class PowerAggregator:
    """
    Optional Phase A aggregator: holds per-tile PowerProxy instances and provides totals.
    """

    def __init__(self) -> None:
        self.by_tile: Dict[int, PowerProxy] = {}

