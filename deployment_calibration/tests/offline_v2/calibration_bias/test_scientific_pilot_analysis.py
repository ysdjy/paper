"""Sanity tests for the scientific pilot analysis (EXPLORATORY / PILOT — NOT CONFIRMATORY). Fast unit checks:
the leakage guard blocks hidden fields, features split K0/K1 correctly, block grouping is used, and the
oracle/best-single computation is well-formed. Does not launch Isaac; skips gracefully if the 306 data is
absent on this machine."""

from __future__ import annotations

import pytest

from deployment_calibration.analysis.offline_v2.calibration_bias import pilot_data_loader as DL
from deployment_calibration.analysis.offline_v2.calibration_bias import pilot_features as FT
from deployment_calibration.analysis.offline_v2.calibration_bias import pilot_statistics as ST


def _data_or_skip():
    try:
        return DL.resolve_data_path()
    except FileNotFoundError:
        pytest.skip("306 exploratory data not present on this machine")


def test_leakage_guard_blocks_hidden_fields():
    bad = ["residual_bias_y", "actual_bias_y", "eff_signed", "abs_eff", "oracle_action",
           "true_handle_error_at_close_local_y", "secret_deployment_state.x", "nominal_bias_y", "block_seed"]
    res = FT.leakage_check(bad)
    assert res["ok"] is False
    assert len(res["violations"]) == len(bad)                    # every hidden name is caught


def test_leakage_guard_allows_clean_features():
    res = FT.leakage_check(["offset", "abs_offset", "tcp_pos_0", "probe_phase_goal_error",
                            "probe_skill_elapsed_time", "target_tolerance"])
    assert res["ok"] is True and res["violations"] == []


def test_static_features_carry_no_hidden_state():
    path = _data_or_skip()
    ep = DL.load_episodes(path)[0]
    feats = FT.static_features(ep)
    assert FT.leakage_check(feats.keys())["ok"]                  # K0 is leakage-clean
    assert "offset" in feats and "abs_offset" in feats


def test_k1_strictly_extends_k0_with_probe_history():
    path = _data_or_skip()
    ep = DL.load_episodes(path)[0]
    k0 = set(FT.static_features(ep))
    k1_extra = set(FT.probe_history_features(ep))
    assert k1_extra and k0.isdisjoint(k1_extra)                  # probe history is additive, disjoint from K0
    assert all(n.startswith("probe_") for n in k1_extra)
    assert FT.leakage_check(k1_extra)["ok"]


def test_capability_map_is_18x17_and_probe_present():
    path = _data_or_skip()
    cap = DL.build_capability_map(DL.load_episodes(path))
    assert len(cap["blocks"]) == 18 and len(cap["offsets"]) == 17
    assert all(cap["probe"][b] is not None for b in cap["blocks"])          # emulated probe (offset 0) exists
    assert all((b, o) in cap["cells"] for b in cap["blocks"] for o in cap["offsets"])


def test_block_bootstrap_is_wellformed():
    ci = ST.block_bootstrap_ci([1, 1, 0, 1, 0, 1], n_boot=1000, seed=0)
    assert ci["lo"] <= ci["point"] <= ci["hi"] and ci["n_blocks"] == 6
    d = ST.paired_block_bootstrap_ci([1, 1, 1], [1, 1, 1], n_boot=500)
    assert d["point"] == 0.0                                     # identical arms -> zero difference


def test_oracle_and_best_single_are_bounded():
    path = _data_or_skip()
    cap = DL.build_capability_map(DL.load_episodes(path))
    succ = {(b, o): DL.success(cap["cells"][(b, o)]) for b in cap["blocks"] for o in cap["offsets"]}
    oracle = sum(1 for b in cap["blocks"] if any(succ[(b, o)] for o in cap["offsets"]))
    assert 0 <= oracle <= len(cap["blocks"])
    # oracle must be >= any single fixed offset's success count (definitional)
    for o in cap["offsets"]:
        assert oracle >= sum(succ[(b, o)] for b in cap["blocks"])
