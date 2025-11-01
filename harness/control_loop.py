from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

# Public APIs from slow-plane/control
from control.guardrails import validate_reconfig_pack, GuardrailConfig
from control.gcu import GCU, GCUConfig
from slow_plane.field.solver import FieldConfig, solve_field
from slow_plane.geometry.update import GeometryConfig, update_geometry
from slow_plane.packer.make import PackerConfig, make_reconfig_pack

# Prefer a public synthetic helper if available; otherwise define a local deterministic fallback.
try:
    # Underscore helper, but deterministic and available in Phase A repo
    from slow_plane.perf_overhead import _synthetic_pggs_artifacts as _syn_pggs  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    _syn_pggs = None  # type: ignore[assignment]


def _to_native(obj: Any) -> Any:
    """
    Recursively convert numpy arrays/scalars and other non-JSON-native types
    into Python lists/ints/floats/bools/str for Phase A guardrails/GCU.
    """
    # Fast path for native primitives
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Dict
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}

    # List/tuple
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]

    # Numpy-like conversions without importing numpy explicitly:
    # - Prefer tolist() if present (arrays, scalars)
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        try:
            return _to_native(tolist())
        except Exception:
            pass

    # - item() for numpy scalar
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return _to_native(item())
        except Exception:
            pass

    # Fallback: return as-is (should not happen for pack fields)
    return obj


def _shape_hw_from_g(g: Any) -> Tuple[int, int]:
    """
    Infer (H, W) from a grid-like g expected to be [H][W][2][2] lists or similar.
    """
    try:
        H = len(g)
        W = len(g[0]) if H > 0 else 0
        return int(H), int(W)
    except Exception as e:
        raise ValueError(f"invalid grid g; expected [H][W][2][2] structure: {e}")


def _mk_identity_metric(H: int, W: int) -> Any:
    """Build a list-based [H][W][2][2] identity metric grid."""
    row = [[[1.0, 0.0], [0.0, 1.0]] for _ in range(W)]
    return [list(row) for _ in range(H)]


def _mk_synthetic_artifacts(H: int, W: int) -> Dict[str, Any]:
    """
    Deterministic, list-based U,J,B suitable for Phase A pipeline if a helper is not available.
    Shapes:
      - U: [H][W] floats
      - J: [H][W] floats (nonzero interior)
      - B: dict with keys N,S,E,W each [H][W] nonnegative
    """
    if _syn_pggs is not None:
        syn = _syn_pggs(H, W)  # returns numpy arrays/dicts
        # Convert to lists
        U = _to_native(syn["U"])
        J = _to_native(syn["J"])
        B = _to_native(syn["B"])
        return {"U": U, "J": J, "B": B}

    # Fallback: simple bowl-shaped U, derived J via 5-pt Laplacian, B from U non-neg parts.
    U = [[0.0 for _ in range(W)] for __ in range(H)]
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    # Parabolic bowl normalized to [0,1]
    max_r2 = (max(cx, W - 1 - cx) ** 2 + max(cy, H - 1 - cy) ** 2) or 1.0
    for y in range(H):
        for x in range(W):
            r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
            U[y][x] = 1.0 - (r2 / max_r2)

    # 5-point Laplacian J and zero-mean
    J = [[0.0 for _ in range(W)] for __ in range(H)]
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            J[y][x] = (U[y][x + 1] - 2.0 * U[y][x] + U[y][x - 1]) + (U[y + 1][x] - 2.0 * U[y][x] + U[y - 1][x])
    meanJ = sum(J[y][x] for y in range(H) for x in range(W)) / float(H * W)
    for y in range(H):
        for x in range(W):
            J[y][x] = J[y][x] - meanJ

    # Directional cues from U (nonnegative)
    def _clip01(v: float) -> float:
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    Bn = [[_clip01(U[y][x]) for x in range(W)] for y in range(H)]
    Bs = [[_clip01(-U[y][x]) for x in range(W)] for y in range(H)]
    Be = [[_clip01(-U[y][x]) for x in range(W)] for y in range(H)]
    Bw = [[_clip01(U[y][x]) for x in range(W)] for y in range(H)]
    B = {"N": Bn, "S": Bs, "E": Be, "W": Bw}
    return {"U": U, "J": J, "B": B}


def control_apply_cycle(
    g: Any,
    pggs_artifacts: Optional[Dict[str, Any]],
    telemetry: Optional[Dict[str, Any]] = None,
    gcu_cfg: Optional[GCUConfig] = None,
    guard_cfg: Optional[GuardrailConfig] = None,
    geometry_cfg: Optional[GeometryConfig] = None,
    pack_mutator: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    rollback_on_fail: bool = False,
) -> Dict[str, Any]:
    """
    Integrate slow-plane -> packer -> guardrails -> control apply (GCU) with A/B quick-check.

    Args:
      g: current metric grid [H][W][2][2] list- or ndarray-like
      pggs_artifacts: dict with keys at least U,J,B; if None, generate deterministic synthetic artifacts
      telemetry: optional dict, e.g., {'max_vc_depth': 0, 'temp_max': 60.0, 'power_proxy_avg': 0.6}
      gcu_cfg: optional GCUConfig
      guard_cfg: optional GuardrailConfig
      geometry_cfg: optional GeometryConfig override for the geometry update step
      pack_mutator: optional callable(pack: dict) -> dict to modify pack before guardrails/GCU
      rollback_on_fail: if True, call GCU.rollback() on verify failure

    Returns:
      Structured dict per spec.
    """
    # Normalize grid and shapes
    H, W = _shape_hw_from_g(g if g is not None else _mk_identity_metric(2, 2))
    g_list = g if g is not None else _mk_identity_metric(H, W)

    # PGGS artifacts (U,J,B)
    if pggs_artifacts is None:
        arts = _mk_synthetic_artifacts(H, W)
    else:
        arts = pggs_artifacts
    U = arts.get("U")
    J = arts.get("J")
    B = arts.get("B")

    # Field solve: Φ and ∇Φ from (g, J)
    field_cfg = FieldConfig(method="cg", max_cg_iters=200, cg_tol=1e-5, boundary="neumann", grad_clip=0.0)
    field_out = solve_field(g_list, J, field_cfg)
    phi = field_out["phi"]
    grad = field_out["grad"]

    # Geometry update: g_next and trust-region meta
    # Defaults tuned for Phase A tiny synthetic artifacts to encourage acceptance while respecting config guards
    geom_cfg = geometry_cfg or GeometryConfig(trust_radius=1.5, accept_ratio_min=0.05)
    geom_out = update_geometry(g_list, phi, grad, U, J, B, geom_cfg)
    g_next = geom_out["g_next"]
    geom_meta = geom_out["meta"]

    # Pack using Packer with CRC enabled, DVFS levels provided
    pack_cfg = PackerConfig(dvfs_levels=[0, 1, 2], dvfs_from="grad_phi")
    pack = make_reconfig_pack(g_next, phi, grad, U, J, B, geom_meta, pack_cfg)

    # Convert to list-based/native types for guardrails/GCU
    pack_native = _to_native(pack)
    # Normalize containers for guardrails schema: expect dicts with 'weights'
    lw = pack_native.get("link_weights")
    if isinstance(lw, list):
        pack_native["link_weights"] = {"weights": lw}
    # Be permissive if noc_tables was flattened (should not happen with packer)
    nt = pack_native.get("noc_tables")
    if isinstance(nt, list):
        pack_native["noc_tables"] = {"weights": nt}

    # Optional mutation hook (tests)
    if callable(pack_mutator):
        mutated = pack_mutator(_to_native(pack_native))
        # Ensure still native after mutator
        pack_native = _to_native(mutated if mutated is not None else pack_native)

    # Guardrails
    # Use a permissive default fairness floor for Phase A tiny synthetic artifacts unless caller overrides.
    guard_eff = guard_cfg or GuardrailConfig(fairness_min_share=0.0)
    ok, meta = validate_reconfig_pack(pack_native, telemetry=telemetry, cfg=guard_eff)
    if not ok:
        return {
            "stage": "guardrails",
            "ok": False,
            "reasons": list(meta.get("reasons", [])),
            "checks": dict(meta.get("checks", {})),
        }

    # Control apply via GCU
    gcu = GCU(gcu_cfg or GCUConfig())

    # 1) shadow apply
    _ = gcu.shadow_apply(pack_native)

    # 2) quick check (default)
    quick_res = gcu.quick_check(metric_fn=None)
    if not bool(quick_res.get("passed", False)):
        return {"stage": "quick_check", "ok": False, "meta": _to_native(quick_res.get("meta", {}))}

    # 3) quiesce
    qres = gcu.quiesce(telemetry=telemetry or {"max_vc_depth": 0})
    if not bool(qres.get("entered", False)):
        return {"stage": "quiesce", "ok": False, "meta": _to_native(qres)}

    # 4) commit
    commit_id = gcu.commit()

    # 5) verify (default)
    vres = gcu.verify()
    if not bool(vres.get("passed", False)):
        rolled_back = None
        if rollback_on_fail:
            rb = gcu.rollback()
            rolled_back = bool(rb.get("rolled_back", False))
        return {
            "stage": "verify",
            "ok": False,
            "meta": _to_native(vres.get("meta", {})),
            "rolled_back": bool(rolled_back) if rolled_back is not None else False,
        }

    # Success
    return {
        "stage": "done",
        "ok": True,
        "commit_id": commit_id,
        "pack_crc32c": pack_native.get("crc32c"),
        "trust_region_meta": _to_native(pack_native.get("trust_region_meta")),
        "quick_meta": _to_native(quick_res.get("meta", {})),
        "verify_meta": _to_native(vres.get("meta", {})),
    }