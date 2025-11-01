import copy

from control.guardrails import GuardrailConfig
from harness.control_loop import control_apply_cycle


def _mk_identity_metric(H=2, W=2):
    # Return [H][W][2][2] identity tiles as nested Python lists
    row = [[[1.0, 0.0], [0.0, 1.0]] for _ in range(W)]
    return [list(row) for _ in range(H)]


def _deterministic_pggs_artifacts(H=2, W=2):
    """
    Deterministic small PGGS-like artifacts:
      - U: smooth increasing pattern
      - J: small zero-mean source with nonzero entries
      - B: nonnegative directional cues dict with keys N,S,E,W
    Shapes as plain lists.
    """
    U = [[0.1, 0.2], [0.3, 0.4]]
    # zero-mean J with small magnitude
    J = [[0.01, 0.0], [0.0, -0.01]]
    # nonnegative directional hints
    N = [[0.2, 0.2], [0.2, 0.2]]
    S = [[0.0, 0.0], [0.0, 0.0]]
    E = [[0.1, 0.1], [0.1, 0.1]]
    W = [[0.0, 0.0], [0.0, 0.0]]
    B = {"N": N, "S": S, "E": E, "W": W}
    return {"U": U, "J": J, "B": B}


def test_integration_success_cycle():
    g = _mk_identity_metric(2, 2)
    pggs_artifacts = _deterministic_pggs_artifacts(2, 2)
    telemetry = {"max_vc_depth": 0, "temp_max": 50.0, "power_proxy_avg": 0.5}

    res = control_apply_cycle(g, pggs_artifacts, telemetry=telemetry)

    assert res["ok"] is True
    assert res["stage"] == "done"
    assert isinstance(res.get("commit_id"), str)
    assert "trust_region_meta" in res
    assert isinstance(res["trust_region_meta"], dict)


def test_integration_guardrails_block_on_bad_routing():
    g = _mk_identity_metric(2, 2)
    pggs_artifacts = _deterministic_pggs_artifacts(2, 2)
    telemetry = {"max_vc_depth": 0, "temp_max": 50.0, "power_proxy_avg": 0.5}

    guard_cfg = GuardrailConfig(link_weight_min=0.25)  # require every dir weight >= 0.25

    def mutate_pack(pack: dict) -> dict:
        # Force an invalid directional weight below threshold on a single tile and keep alias in link_weights
        p = copy.deepcopy(pack)
        p["noc_tables"]["weights"][0][0][0] = 0.0  # N dir at (0,0)
        if isinstance(p.get("link_weights"), dict) and isinstance(p["link_weights"].get("weights"), list):
            p["link_weights"]["weights"][0][0][0] = 0.0
        return p

    res = control_apply_cycle(
        g,
        pggs_artifacts,
        telemetry=telemetry,
        guard_cfg=guard_cfg,
        pack_mutator=mutate_pack,
    )

    assert res["ok"] is False
    assert res["stage"] == "guardrails"
    reasons = " | ".join(res.get("reasons", [])).lower()
    assert "routing" in reasons


def test_integration_rollback_path_on_verify_failure():
    g = _mk_identity_metric(2, 2)
    pggs_artifacts = _deterministic_pggs_artifacts(2, 2)
    telemetry = {"max_vc_depth": 0, "temp_max": 50.0, "power_proxy_avg": 0.5}

    # First successful cycle (establish prior apply; subsequent cycle uses fresh GCU but we only require boolean presence)
    first = control_apply_cycle(g, pggs_artifacts, telemetry=telemetry)
    assert first["ok"] is True

    def flip_acceptance(pack: dict) -> dict:
        # Force verify failure by marking trust_region_meta as not accepted
        p = copy.deepcopy(pack)
        p["trust_region_meta"] = {"accepted": False}
        return p

    res = control_apply_cycle(
        g,
        pggs_artifacts,
        telemetry=telemetry,
        pack_mutator=flip_acceptance,
        rollback_on_fail=True,
    )

    assert res["ok"] is False
    assert res["stage"] == "verify"
    assert "rolled_back" in res and isinstance(res["rolled_back"], bool)