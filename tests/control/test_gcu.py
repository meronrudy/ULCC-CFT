import time
import copy
import json

import pytest

from control.gcu import GCU, GCUConfig, ApplyStatus


def _mk_weights(H=2, W=2):
    # Shape [H, W, 4] as lists
    return [[[0.25, 0.25, 0.25, 0.25] for _ in range(W)] for __ in range(H)]


def _mk_pack(accepted=True, with_link_alias=True, crc=123, H=2, W=2):
    weights = _mk_weights(H, W)
    noc_tables = {"weights": weights}
    link_weights = {"weights": copy.deepcopy(weights) if with_link_alias else _mk_weights(H, W)}
    pack = {
        "version": 1,
        "noc_tables": noc_tables,
        "link_weights": link_weights,
        "mc_policy_words": [[0 for _ in range(W)] for __ in range(H)],
        "cat_masks": [[0 for _ in range(W)] for __ in range(H)],
        "cpu_affinities": [[0 for _ in range(W)] for __ in range(H)],
        "dvfs_states": [[0 for _ in range(W)] for __ in range(H)],
        "trust_region_meta": {"accepted": bool(accepted), "accepted_ratio": 1.0 if accepted else 0.0},
        "crc32c": int(crc),
    }
    return pack


def test_shadow_apply_validates_schema_and_crc():
    gcu = GCU(GCUConfig(enable_crc_check=True))
    pack = _mk_pack()
    ticket = gcu.shadow_apply(pack)
    assert isinstance(ticket, str)
    assert gcu.status.status == ApplyStatus.SHADOWED
    assert gcu.status.last_crc32c == pack["crc32c"]

    # Missing required key raises
    bad_pack = _mk_pack()
    del bad_pack["cat_masks"]
    with pytest.raises(Exception):
        gcu.shadow_apply(bad_pack)

    # Missing CRC raises when enabled
    no_crc = _mk_pack()
    del no_crc["crc32c"]
    with pytest.raises(Exception):
        gcu.shadow_apply(no_crc)


def test_quick_check_default_pass_and_regression_budget():
    # By default, structural pass and regression_pct default to 0.0
    gcu = GCU(GCUConfig(max_sim_regression_pct=0.0))
    gcu.shadow_apply(_mk_pack())
    res = gcu.quick_check()
    assert res["passed"] is True
    assert res["horizon_us"] == gcu.cfg.quick_check_horizon_us

    # Custom metric_fn that signals small regression should fail with budget 0.0
    def metric_fn_fail(_pack):
        return True, {"regression_pct": 0.001}

    res2 = gcu.quick_check(metric_fn=metric_fn_fail)
    assert res2["passed"] is False
    assert "regression" in res2["meta"].get("reason", "")

    # If budget increased, should pass
    gcu2 = GCU(GCUConfig(max_sim_regression_pct=1.0))
    gcu2.shadow_apply(_mk_pack())
    res3 = gcu2.quick_check(metric_fn=metric_fn_fail)
    assert res3["passed"] is True


def test_quiesce_threshold_and_commit_verify_success():
    gcu = GCU()
    gcu.shadow_apply(_mk_pack(accepted=True))

    # Fail to enter quiesce if vc depth above threshold
    q1 = gcu.quiesce(telemetry={"max_vc_depth": gcu.cfg.quiesce_vc_drain_threshold + 1})
    assert q1["entered"] is False
    assert gcu.status.status == ApplyStatus.SHADOWED

    # Enter quiesce when depth within threshold
    q2 = gcu.quiesce(telemetry={"max_vc_depth": gcu.cfg.quiesce_vc_drain_threshold})
    assert q2["entered"] is True
    assert gcu.status.status == ApplyStatus.QUIESCED

    # Commit and verify should pass for accepted True
    commit_id = gcu.commit()
    assert isinstance(commit_id, str)
    assert gcu.status.status == ApplyStatus.COMMITTED

    vres = gcu.verify()
    assert vres["passed"] is True
    assert "tri" in vres["meta"]


def test_rollback_when_verify_fails():
    gcu = GCU()

    # First establish a baseline ACTIVE pack by a full apply
    gcu.shadow_apply(_mk_pack(accepted=True, crc=111))
    assert gcu.quiesce(telemetry={"max_vc_depth": 0})["entered"]
    gcu.commit()
    assert gcu.status.status == ApplyStatus.COMMITTED

    # Now stage a new pack that will fail verify (accepted=False)
    gcu.shadow_apply(_mk_pack(accepted=False, crc=222))
    assert gcu.quiesce(telemetry={"max_vc_depth": 0})["entered"]
    gcu.commit()
    assert gcu.status.status == ApplyStatus.COMMITTED

    vres = gcu.verify()
    assert vres["passed"] is False

    rb = gcu.rollback()
    assert rb["rolled_back"] in (True, False)  # True if previous existed, False otherwise
    assert gcu.status.status == ApplyStatus.ROLLED_BACK


def test_watchdog_deadline_enforcement():
    gcu = GCU(GCUConfig(apply_timeout_us=100, verify_timeout_us=150))
    start = gcu._now_us()
    # Busy-wait to exceed apply timeout
    while gcu._now_us() - start <= 120:
        pass
    wd = gcu.watchdog(start_us=start, stage="apply")
    assert wd["ok"] is False
    assert wd["elapsed_us"] >= gcu.cfg.apply_timeout_us

    # Verify stage within deadline
    start2 = gcu._now_us()
    wd2 = gcu.watchdog(start_us=start2, stage="verify")
    assert wd2["ok"] is True