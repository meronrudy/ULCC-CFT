from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math


@dataclass
class GuardrailConfig:
    thermal_ceiling_c: float = 95.0
    power_proxy_limit: float = 1.0
    fairness_min_share: float = 0.05
    delta_g_norm_max: float = 0.15
    link_weight_min: float = 0.0
    weight_sum_tol: float = 1e-3


def validate_reconfig_pack(
    pack: Dict[str, Any],
    telemetry: Optional[Dict[str, Any]] = None,
    prev_meta: Optional[Dict[str, Any]] = None,
    cfg: Optional[GuardrailConfig] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate a Phase A ReconfigPack-like dict against basic guardrails.

    Returns:
        (ok, meta) where meta contains:
          - 'reasons': list[str] of failure reasons (empty if ok)
          - 'checks': dict[str, bool] per guard category
    """
    _cfg = cfg or GuardrailConfig()
    reasons: List[str] = []
    checks: Dict[str, bool] = {
        "routing": True,
        "fairness": True,
        "thermal": True,
        "power": True,
        "trust": True,
    }

    # --- Routing weights safety ---
    routing_ok, routing_reason = _check_routing_weights(pack, _cfg)
    if not routing_ok:
        checks["routing"] = False
        reasons.append(f"routing: {routing_reason}")

    # --- Fairness floor (MC) ---
    fairness_ok, fairness_reason = _check_fairness_floor(pack, _cfg)
    if not fairness_ok:
        checks["fairness"] = False
        reasons.append(f"fairness: {fairness_reason}")

    # --- Thermal/power guardrails (optional, only if provided) ---
    if telemetry is not None and isinstance(telemetry, dict):
        # Thermal
        if "temp_max" in telemetry and _is_number(telemetry["temp_max"]):
            if float(telemetry["temp_max"]) > _cfg.thermal_ceiling_c:
                checks["thermal"] = False
                reasons.append(
                    f"thermal: temp_max {float(telemetry['temp_max'])} exceeds ceiling {(_cfg.thermal_ceiling_c)}"
                )
        # Power
        if "power_proxy_avg" in telemetry and _is_number(telemetry["power_proxy_avg"]):
            if float(telemetry["power_proxy_avg"]) > _cfg.power_proxy_limit:
                checks["power"] = False
                reasons.append(
                    f"power: power_proxy_avg {float(telemetry['power_proxy_avg'])} exceeds limit {(_cfg.power_proxy_limit)}"
                )

    # --- Trust-region bounds (optional) ---
    tri = pack.get("trust_region_meta")
    if isinstance(tri, dict):
        delta_val = tri.get("delta_norm", None)
        if not _is_number(delta_val):
            delta_val = tri.get("delta_g_norm", None)
        if _is_number(delta_val):
            if float(delta_val) > _cfg.delta_g_norm_max:
                checks["trust"] = False
                reasons.append(
                    f"trust: delta_norm {float(delta_val)} exceeds bound {(_cfg.delta_g_norm_max)}"
                )

    ok = all(checks.values())
    meta: Dict[str, Any] = {"reasons": reasons, "checks": checks}
    return ok, meta


# -------------------------
# Internal helper routines
# -------------------------

def _check_routing_weights(pack: Dict[str, Any], cfg: GuardrailConfig) -> Tuple[bool, str]:
    """
    Ensure:
      - noc_tables.weights and link_weights.weights exist
      - both are nested lists with shape [H, W, 4]
      - all weights are finite and >= cfg.link_weight_min
      - per-tile directional weights sum to > 0 (with tolerance)
      - the two tensors have equal [H, W, 4] shape
    """
    noc = pack.get("noc_tables")
    link = pack.get("link_weights")
    if not isinstance(noc, dict) or not isinstance(link, dict):
        return False, "missing or invalid noc_tables/link_weights containers"

    nw = noc.get("weights")
    lw = link.get("weights")
    if not isinstance(nw, list) or not isinstance(lw, list):
        return False, "missing weights list under noc_tables or link_weights"

    ok_nw, shape_nw, reason_nw = _validate_weight_tensor(nw, cfg)
    if not ok_nw:
        return False, f"noc_tables invalid: {reason_nw}"
    ok_lw, shape_lw, reason_lw = _validate_weight_tensor(lw, cfg)
    if not ok_lw:
        return False, f"link_weights invalid: {reason_lw}"

    if shape_nw != shape_lw:
        return False, f"shape mismatch between noc_tables {shape_nw} and link_weights {shape_lw}"

    return True, ""


def _validate_weight_tensor(weights: List[Any], cfg: GuardrailConfig) -> Tuple[bool, Tuple[int, int, int], str]:
    """
    Validate a [H, W, 4] list-of-lists-of-lists tensor.
    Returns: (ok, (H, W, 4), reason_if_not_ok)
    """
    if not isinstance(weights, list):
        return False, (0, 0, 0), "weights must be a list"

    H = len(weights)
    if H == 0:
        return False, (0, 0, 0), "empty H dimension"

    if not isinstance(weights[0], list):
        return False, (0, 0, 0), "weights[0] must be a list"
    W = len(weights[0])
    if W == 0:
        return False, (H, 0, 0), "empty W dimension"

    # Ensure rectangular [H, W]
    for h in range(H):
        row = weights[h]
        if not isinstance(row, list) or len(row) != W:
            return False, (H, W, 0), "non-rectangular HxW"

    # Validate the last dimension and numeric constraints
    for h in range(H):
        for w in range(W):
            cell = weights[h][w]
            if not isinstance(cell, list) or len(cell) != 4:
                return False, (H, W, 0), "last dimension must be of length 4 per tile"
            s = 0.0
            for k in range(4):
                v = cell[k]
                if not _is_number(v) or not math.isfinite(float(v)):
                    return False, (H, W, 4), "non-finite or non-numeric weight detected"
                if float(v) < cfg.link_weight_min:
                    return False, (H, W, 4), f"weight {float(v)} < link_weight_min {cfg.link_weight_min}"
                s += float(v)
            if s <= cfg.weight_sum_tol:
                return False, (H, W, 4), f"per-tile weight sum {s} not > 0"
    return True, (H, W, 4), ""


def _check_fairness_floor(pack: Dict[str, Any], cfg: GuardrailConfig) -> Tuple[bool, str]:
    """
    For mc_policy_words [H, W] numeric:
      - Compute total S. If S == 0, pass.
      - Else each tile's share s_ij / S must be >= cfg.fairness_min_share
    """
    mc = pack.get("mc_policy_words")
    if not isinstance(mc, list) or len(mc) == 0:
        # If not present/empty, treat as pass for Phase A permissiveness
        return True, ""
    H = len(mc)
    if not isinstance(mc[0], list):
        return True, ""  # treat as pass if not 2D; spec here is permissive
    W = len(mc[0])
    # Rectangular check
    for h in range(H):
        row = mc[h]
        if not isinstance(row, list) or len(row) != W:
            return True, ""  # permissive
    # Sum and min-share
    total = 0.0
    for h in range(H):
        for w in range(W):
            v = mc[h][w]
            if _is_number(v) and math.isfinite(float(v)):
                total += float(v)
            # Non-numeric treated as 0 contribution

    if abs(total) <= 0.0:
        return True, ""  # no demand => pass

    min_share = 1.0
    for h in range(H):
        for w in range(W):
            v = mc[h][w]
            x = float(v) if _is_number(v) and math.isfinite(float(v)) else 0.0
            share = x / total if total != 0.0 else 0.0
            if share < min_share:
                min_share = share

    if min_share + 0.0 < cfg.fairness_min_share:
        return False, f"min share {min_share:.6f} < fairness_min_share {cfg.fairness_min_share:.6f}"
    return True, ""


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float))

