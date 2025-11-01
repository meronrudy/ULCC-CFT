import copy

from control.guardrails import validate_reconfig_pack, GuardrailConfig


def _mk_weights(H=2, W=2):
    # Shape [H, W, 4] lists, normalized per-tile
    return [[[0.25, 0.25, 0.25, 0.25] for _ in range(W)] for __ in range(H)]


def _mk_pack(H=2, W=2):
    weights = _mk_weights(H, W)
    noc_tables = {"weights": weights}
    link_weights = {"weights": copy.deepcopy(weights)}
    pack = {
        "version": 1,
        "noc_tables": noc_tables,
        "link_weights": link_weights,
        "mc_policy_words": [[0 for _ in range(W)] for __ in range(H)],
        "cat_masks": [[0 for _ in range(W)] for __ in range(H)],
        "cpu_affinities": [[0 for _ in range(W)] for __ in range(H)],
        "dvfs_states": [[0 for _ in range(W)] for __ in range(H)],
        "trust_region_meta": {"accepted": True, "delta_norm": 0.05},
        "crc32c": 123,
    }
    return pack


def _has_reason(reasons, needle):
    s = " | ".join(reasons).lower()
    return needle in s


def test_validate_passes_on_nominal_pack():
    pack = _mk_pack()
    ok, meta = validate_reconfig_pack(pack)
    assert ok is True
    assert isinstance(meta, dict)
    checks = meta.get("checks", {})
    # All checks should be True when no telemetry provided and nominal pack
    assert all(checks.get(k, False) for k in ["routing", "fairness", "thermal", "power", "trust"])


def test_routing_guard_violates_on_negative_weight():
    pack = _mk_pack()
    # Inject a negative weight in noc_tables
    pack["noc_tables"]["weights"][0][0][0] = -0.1
    ok, meta = validate_reconfig_pack(pack)
    assert ok is False
    assert _has_reason(meta.get("reasons", []), "routing")


def test_fairness_floor_enforced_on_skew():
    H, W = 2, 2
    pack = _mk_pack(H, W)
    # Make one tile extremely large, others zero (skewed)
    mc = [[0 for _ in range(W)] for __ in range(H)]
    mc[0][0] = 1_000_000
    pack["mc_policy_words"] = mc
    # Use default fairness_min_share = 0.05, so min share 0.0 should fail
    ok, meta = validate_reconfig_pack(pack)
    assert ok is False
    assert _has_reason(meta.get("reasons", []), "fairness")


def test_thermal_and_power_guardrails():
    pack = _mk_pack()
    telemetry = {"temp_max": 96.5, "power_proxy_avg": 1.10}
    ok, meta = validate_reconfig_pack(pack, telemetry=telemetry)
    assert ok is False
    reasons = meta.get("reasons", [])
    assert _has_reason(reasons, "thermal")
    assert _has_reason(reasons, "power")


def test_trust_region_delta_bound():
    pack = _mk_pack()
    # Set delta beyond default bound 0.15
    pack["trust_region_meta"]["delta_norm"] = 0.25
    ok, meta = validate_reconfig_pack(pack)
    assert ok is False
    assert _has_reason(meta.get("reasons", []), "trust")


def test_shape_mismatch_between_noc_and_link_weights():
    # NOC is 2x2, make link weights 1x2
    pack = _mk_pack(H=2, W=2)
    pack["link_weights"]["weights"] = _mk_weights(H=1, W=2)
    ok, meta = validate_reconfig_pack(pack)
    assert ok is False
    assert _has_reason(meta.get("reasons", []), "routing")