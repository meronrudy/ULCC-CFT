from __future__ import annotations

"""
Lightweight deterministic thermal RC proxy for Phase A.

Model (discrete one-pole RC with dt = 1 cycle):
  T_{k+1} = T_k + (1/c_th) * (power - (T_k - t_amb) / r_th)

Units (proxies, not physical):
- r_th: cycles per energy (proxy).
- c_th: energy per temperature (proxy).
- t_amb: ambient baseline temperature (proxy units).
- Temperature evolves deterministically per cycle.

Constraints:
- r_th > 0, c_th > 0 for stability in this proxy.
- t_max (optional): clamp temperature at or below this bound.
"""

from dataclasses import dataclass


@dataclass
class ThermalConfig:
    r_th: float
    c_th: float
    t_amb: float
    t_init: float | None = None
    t_max: float | None = None

    def __post_init__(self) -> None:
        if self.r_th <= 0.0:
            raise ValueError("r_th must be > 0")
        if self.c_th <= 0.0:
            raise ValueError("c_th must be > 0")
        if self.t_init is None:
            self.t_init = float(self.t_amb)


class ThermalRC:
    """
    Per-tile thermal RC proxy.
    """

    def __init__(self, config: ThermalConfig) -> None:
        self.cfg = config
        self._t: float = float(self.cfg.t_init if self.cfg.t_init is not None else self.cfg.t_amb)
        self.max_temp_seen: float = float(self._t)

    @property
    def temp(self) -> float:
        return self._t

    def step(self, power: float) -> None:
        # Discrete RC update with dt = 1
        # p drives heating; (T - Tamb)/Rth drives cooling
        p = float(power)
        t = self._t
        dt_term = (p - (t - self.cfg.t_amb) / self.cfg.r_th) / self.cfg.c_th
        t_next = t + dt_term
        # Optional clamp
        if self.cfg.t_max is not None and t_next > self.cfg.t_max:
            t_next = float(self.cfg.t_max)
        self._t = float(t_next)
        if self._t > self.max_temp_seen:
            self.max_temp_seen = float(self._t)