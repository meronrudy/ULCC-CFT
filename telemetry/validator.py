"""
Telemetry frame validator utilities (Phase A — E2)

Functions:
- validate_frame(frame) -> (ok: bool, reason: str|None)
  Wraps TelemetryFrame.validate() and maps errors to friendly reason codes:
    * "crc_mismatch" -> "crc_mismatch"
    * bounds-related messages (contains "out of bounds", "negative", "overlap") -> "bounds_violation"
    * required/missing fields (contains "must", "missing", "None", "invalid") -> "missing_required_field"
    * otherwise, returns the raw reason string (lowercased) when available

- filter_and_count(frames) -> (valid_frames, stats)
  Filters frames by validate_frame and returns a tuple:
    valid_frames: list of frames with ok == True
    stats: dict with counts:
      {
        "total": int,
        "valid": int,
        "crc_fail": int,
        "bounds_fail": int,
        "missing": int
      }

Notes:
- This module does not mutate frames and has no side effects.
- Aggregator remains independent; tests can feed only valid frames produced here.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional

from .frame import TelemetryFrame


def _map_reason(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    r = str(raw).strip()
    # Fast-path canonical code from TelemetryFrame.validate
    if r == "crc_mismatch":
        return "crc_mismatch"

    rl = r.lower()
    # Bounds violations
    if ("out of bounds" in rl) or ("negative" in rl):
        return "bounds_violation"

    # Overlap / monotonic window errors count as bounds/consistency violations
    if "overlap" in rl:
        return "bounds_violation"

    # Missing/required field style errors
    if ("must" in rl) or ("missing" in rl) or ("invalid" in rl) or ("none" in rl):
        return "missing_required_field"

    # Fall back to normalized message
    return rl


def validate_frame(frame: TelemetryFrame) -> Tuple[bool, Optional[str]]:
    """
    Validate a single TelemetryFrame and return (ok, reason).
    reason is a canonical short string or None when ok is True.
    """
    ok, msg = frame.validate(previous=None)
    if ok:
        return True, None
    return False, _map_reason(msg)


def filter_and_count(frames: List[TelemetryFrame]) -> Tuple[List[TelemetryFrame], Dict[str, int]]:
    """
    Filter frames by validate_frame and return (valid_frames, stats).
    """
    valid: List[TelemetryFrame] = []
    stats: Dict[str, int] = {
        "total": 0,
        "valid": 0,
        "crc_fail": 0,
        "bounds_fail": 0,
        "missing": 0,
    }

    for f in frames:
        stats["total"] += 1
        ok, reason = validate_frame(f)
        if ok:
            valid.append(f)
            stats["valid"] += 1
        else:
            if reason == "crc_mismatch":
                stats["crc_fail"] += 1
            elif reason == "bounds_violation":
                stats["bounds_fail"] += 1
            elif reason == "missing_required_field":
                stats["missing"] += 1
            else:
                # Unclassified errors: attribute to bounds for conservative triage
                stats["bounds_fail"] += 1

    return valid, stats