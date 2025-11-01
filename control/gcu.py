from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional, Tuple


# Types
ReconfigPack = Dict[str, Any]
MetricFn = Callable[[ReconfigPack], Tuple[bool, Dict[str, Any]]]
VerifyFn = Callable[[ReconfigPack], Tuple[bool, Dict[str, Any]]]


class GCUStateError(RuntimeError):
    pass


class GCUPackError(ValueError):
    pass


class ApplyStatus(Enum):
    IDLE = auto()
    SHADOWED = auto()
    QUIESCED = auto()
    COMMITTED = auto()
    ROLLED_BACK = auto()


@dataclass
class GCUConfig:
    quick_check_horizon_us: int = 10
    quiesce_vc_drain_threshold: int = 2
    verify_timeout_us: int = 1_000
    apply_timeout_us: int = 2_000
    enable_crc_check: bool = True
    # Minimal A/B thresholds for quick_check
    max_sim_regression_pct: float = 0.0  # phase A: disallow regression in quick A/B


@dataclass
class GCUStatus:
    status: ApplyStatus = ApplyStatus.IDLE
    shadow_ticket_id: Optional[str] = None
    commit_id: Optional[str] = None
    last_crc32c: Optional[int] = None
    last_error: Optional[str] = None
    timestamps_us: Dict[str, int] = field(default_factory=dict)


class GCU:
    """
    Geometry Control Unit (emulator) for Phase A.

    Implements the atomic apply protocol:
      shadow_apply → quick_check → quiesce → commit → verify → (ok | rollback)

    All methods are synchronous and deterministic given identical inputs.
    Timing is tracked with a monotonic microsecond clock for watchdogs.
    """

    def __init__(self, cfg: Optional[GCUConfig] = None) -> None:
        self.cfg = cfg or GCUConfig()
        self._active_pack: Optional[ReconfigPack] = None
        self._shadow_pack: Optional[ReconfigPack] = None
        self._prev_pack: Optional[ReconfigPack] = None
        self._status = GCUStatus()
        self._monotonic0_us = self._now_us()

    # --- Time/ID utilities ---

    def _now_us(self) -> int:
        return int(time.monotonic_ns() // 1_000)

    def _mk_id(self, prefix: str) -> str:
        return f"{prefix}_{self._now_us()}"

    # --- Public API ---

    def shadow_apply(self, pack: ReconfigPack) -> str:
        """
        Stage 1: Validate and stage a ReconfigPack as shadow configuration.

        Returns:
            shadow_ticket_id (str)
        """
        self._ensure_state(allowed={ApplyStatus.IDLE, ApplyStatus.COMMITTED, ApplyStatus.ROLLED_BACK})
        self._validate_pack_schema(pack)
        if self.cfg.enable_crc_check:
            self._validate_pack_crc(pack)

        self._shadow_pack = self._deepcopy_pack(pack)
        self._status.status = ApplyStatus.SHADOWED
        self._status.shadow_ticket_id = self._mk_id("shadow")
        self._status.timestamps_us["shadow"] = self._now_us()
        self._status.last_crc32c = pack.get("crc32c")
        self._status.last_error = None
        return self._status.shadow_ticket_id  # type: ignore[return-value]

    def quick_check(self, metric_fn: Optional[MetricFn] = None) -> Dict[str, Any]:
        """
        Stage 2: Perform a fast A/B screen over a tiny horizon without mutating active state.

        metric_fn:
            Optional callable that receives the shadow pack and returns (pass: bool, meta: dict).
            If None, a default structural check is used.

        Returns:
            result dict with keys: {passed: bool, horizon_us: int, meta: dict}
        """
        self._ensure_state(allowed={ApplyStatus.SHADOWED})
        if self._shadow_pack is None:
            raise GCUStateError("quick_check: missing shadow pack")

        horizon = int(self.cfg.quick_check_horizon_us)
        start = self._now_us()
        # Default check: basic structural sanity + optional CRC presence
        if metric_fn is None:
            passed, meta = self._default_quick_check(self._shadow_pack)
        else:
            passed, meta = metric_fn(self._shadow_pack)

        # Enforce no-regression policy if provided by metric_fn
        reg = float(meta.get("regression_pct", 0.0))
        if passed and reg > self.cfg.max_sim_regression_pct:
            # regression exceeds allowed budget -> fail fast
            passed = False
            meta["reason"] = f"regression {reg:.3f}% exceeds budget {self.cfg.max_sim_regression_pct:.3f}%"

        # Simulate tiny compute horizon
        while self._now_us() - start < horizon:
            pass  # busy-wait to approximate the intended horizon in us

        return {"passed": passed, "horizon_us": horizon, "meta": meta}

    def quiesce(self, telemetry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Stage 3: Enter quiesce window. For Phase A, emulate VC drain thresholds.

        telemetry:
            Optional telemetry snapshot dict with keys like { 'max_vc_depth': int }.

        Returns:
            dict with {entered: bool, meta: dict}
        """
        self._ensure_state(allowed={ApplyStatus.SHADOWED})
        if self._shadow_pack is None:
            raise GCUStateError("quiesce: missing shadow pack")

        max_depth = 0
        if telemetry and isinstance(telemetry.get("max_vc_depth"), int):
            max_depth = int(telemetry["max_vc_depth"])

        entered = max_depth <= self.cfg.quiesce_vc_drain_threshold
        meta = {"max_vc_depth": max_depth, "threshold": self.cfg.quiesce_vc_drain_threshold}
        if entered:
            self._status.status = ApplyStatus.QUIESCED
            self._status.timestamps_us["quiesce"] = self._now_us()
        else:
            self._status.last_error = f"VC depth {max_depth} > threshold {self.cfg.quiesce_vc_drain_threshold}"
        return {"entered": entered, "meta": meta}

    def commit(self) -> str:
        """
        Stage 4: Commit the shadow configuration atomically.

        Returns:
            commit_id (str)
        """
        self._ensure_state(allowed={ApplyStatus.QUIESCED})
        if self._shadow_pack is None:
            raise GCUStateError("commit: missing shadow pack")

        self._prev_pack = self._deepcopy_pack(self._active_pack) if self._active_pack is not None else None
        self._active_pack = self._shadow_pack
        self._shadow_pack = None
        self._status.status = ApplyStatus.COMMITTED
        self._status.commit_id = self._mk_id("commit")
        self._status.timestamps_us["commit"] = self._now_us()
        return self._status.commit_id  # type: ignore[return-value]

    def verify(self, verify_fn: Optional[VerifyFn] = None, timeout_us: Optional[int] = None) -> Dict[str, Any]:
        """
        Stage 5: Verify post-commit behavior using canary probes.

        verify_fn:
            Optional callable receiving the ACTIVE pack, returns (pass: bool, meta: dict).
            If None, a default canary probe is used.

        timeout_us:
            Optional timeout override in microseconds.

        Returns:
            dict with {passed: bool, meta: dict}
        """
        self._ensure_state(allowed={ApplyStatus.COMMITTED})
        if self._active_pack is None:
            raise GCUStateError("verify: missing active pack")

        deadline = self._now_us() + int(timeout_us or self.cfg.verify_timeout_us)
        if verify_fn is None:
            passed, meta = self._default_verify(self._active_pack)
        else:
            passed, meta = verify_fn(self._active_pack)

        # emulate time consumption but respect timeout
        while self._now_us() < deadline:
            # quick exit if verify_fn already computed a result
            break

        if not passed:
            self._status.last_error = meta.get("reason", "verify failed")
        return {"passed": passed, "meta": meta}

    def rollback(self) -> Dict[str, Any]:
        """
        Stage 6: Roll back to previous active configuration if available.

        Returns:
            dict with {rolled_back: bool, reason: Optional[str]}
        """
        self._ensure_state(allowed={ApplyStatus.COMMITTED})
        if self._prev_pack is None:
            self._status.status = ApplyStatus.ROLLED_BACK
            self._status.timestamps_us["rollback"] = self._now_us()
            return {"rolled_back": False, "reason": "no previous pack"}

        # swap back
        self._active_pack, self._prev_pack = self._prev_pack, None
        self._status.status = ApplyStatus.ROLLED_BACK
        self._status.timestamps_us["rollback"] = self._now_us()
        return {"rolled_back": True, "reason": None}

    def watchdog(self, start_us: int, stage: str) -> Dict[str, Any]:
        """
        Watchdog helper to validate stage deadlines.

        Args:
            start_us: microsecond timestamp when the stage began
            stage: one of {"apply", "verify"} for Phase A

        Returns:
            dict {ok: bool, elapsed_us: int, deadline_us: int}
        """
        now = self._now_us()
        elapsed = now - int(start_us)
        if stage == "apply":
            deadline = self.cfg.apply_timeout_us
        elif stage == "verify":
            deadline = self.cfg.verify_timeout_us
        else:
            deadline = self.cfg.apply_timeout_us
        ok = elapsed <= deadline
        if not ok:
            self._status.last_error = f"watchdog: stage {stage} elapsed {elapsed}us > deadline {deadline}us"
        return {"ok": ok, "elapsed_us": elapsed, "deadline_us": deadline}

    # --- Introspection ---

    @property
    def status(self) -> GCUStatus:
        return self._status

    @property
    def active_pack(self) -> Optional[ReconfigPack]:
        return self._deepcopy_pack(self._active_pack) if self._active_pack is not None else None

    @property
    def shadow_pack(self) -> Optional[ReconfigPack]:
        return self._deepcopy_pack(self._shadow_pack) if self._shadow_pack is not None else None

    # --- Internal helpers ---

    def _ensure_state(self, *, allowed: set[ApplyStatus]) -> None:
        if self._status.status not in allowed:
            raise GCUStateError(f"Invalid state: {self._status.status.name}; allowed: {[s.name for s in allowed]}")

    def _validate_pack_schema(self, pack: ReconfigPack) -> None:
        required_top = [
            "version",
            "noc_tables",
            "link_weights",
            "mc_policy_words",
            "cat_masks",
            "cpu_affinities",
            "dvfs_states",
            "trust_region_meta",
        ]
        for k in required_top:
            if k not in pack:
                raise GCUPackError(f"ReconfigPack missing required key: {k}")
        # Basic alias check for Phase A (per DATA_CONTRACTS)
        if pack["link_weights"] is not None and pack["noc_tables"] is not None:
            # We accept equality by value; exact object identity is not required.
            try:
                lw = json.dumps(pack["link_weights"], sort_keys=True, separators=(",", ":"))
                nt = json.dumps(pack["noc_tables"], sort_keys=True, separators=(",", ":"))
                if lw != nt:
                    # Only warn via status meta; remain permissive in Phase A.
                    self._status.last_error = "link_weights differs from noc_tables; treating as non-fatal in Phase A"
            except Exception:
                # if non-serializable types, skip strict compare in Phase A
                pass

    def _validate_pack_crc(self, pack: ReconfigPack) -> None:
        if "crc32c" not in pack:
            raise GCUPackError("ReconfigPack missing crc32c while CRC check enabled")
        # Best-effort canonicalization to re-hash selective fields for sanity
        # Note: packer is authoritative for CRC generation; here we only enforce presence.
        if not isinstance(pack.get("crc32c"), int):
            raise GCUPackError("ReconfigPack field crc32c must be int")

    def _default_quick_check(self, pack: ReconfigPack) -> Tuple[bool, Dict[str, Any]]:
        # Structural checks
        noc = pack.get("noc_tables")
        link = pack.get("link_weights")
        ok = noc is not None and link is not None
        meta: Dict[str, Any] = {"has_noc": noc is not None, "has_link": link is not None}
        # If available, check dimensions match [H,W,4]
        try:
            nw = noc.get("weights") if isinstance(noc, dict) else None
            lw = link.get("weights") if isinstance(link, dict) else None
            if nw is not None and lw is not None:
                ok = ok and (self._shape_4d(nw) == self._shape_4d(lw)) and self._shape_4d(nw)[-1] == 4
                meta["weights_shape"] = self._shape_4d(nw)
        except Exception as e:
            ok = False
            meta["reason"] = f"quick_check structure error: {e}"
        meta.setdefault("regression_pct", 0.0)
        return ok, meta

    def _default_verify(self, pack: ReconfigPack) -> Tuple[bool, Dict[str, Any]]:
        # Canary probe: verify essential fields are non-empty and trust_region_meta claims acceptance
        tri = pack.get("trust_region_meta", {})
        accepted = bool(tri) and bool(tri.get("accepted", True))
        meta: Dict[str, Any] = {"tri": tri}
        return accepted, meta

    @staticmethod
    def _shape_4d(x: Any) -> Tuple[int, int, int, int]:
        # Accepts lists/nested structures; returns shape or raises.
        if not isinstance(x, list):
            raise TypeError("weights must be nested lists in Phase A")
        h = len(x)
        if h == 0:
            return (0, 0, 0, 0)
        w = len(x[0])
        c = len(x[0][0])
        d = len(x[0][0][0]) if c > 0 and isinstance(x[0][0][0], list) else 0
        # We expect [H, W, 1, 4] or [H, W, 4] depending on packer representation.
        if d == 0:
            # Interpret as [H, W, 4]
            return (h, w, 1, c)
        return (h, w, c, d)

    @staticmethod
    def _deepcopy_pack(pack: Optional[ReconfigPack]) -> Optional[ReconfigPack]:
        if pack is None:
            return None
        # Use JSON round-trip for deterministic deep copy of basic types
        try:
            return json.loads(json.dumps(pack, sort_keys=True, separators=(",", ":")))
        except Exception:
            # Fallback to naive copy
            return json.loads(json.dumps({k: v for k, v in pack.items()}))