"""Frozen band-edge analysis estimators tested on controllable synthetic data (synthetic-only)."""

import numpy as np

from deployment_calibration.offline_v2.calibration_bias import band_edge as BE
from deployment_calibration.offline_v2.calibration_bias.residual_nuisance import draw_block_residuals


def _synth(center=0.03, scale=0.004, seed=1, hard=False):
    cfg = BE.DEFAULT_BAND_EDGE
    res = draw_block_residuals(cfg.n_blocks, seed=seed)
    rng = np.random.default_rng(seed + 100)
    recs = []
    for b in range(cfg.n_blocks):
        r = res[b]
        for o in cfg.offsets:
            eff_s = 0.0 + r + o
            eff = abs(eff_s)
            if hard:
                p = 1.0 if eff <= 0.02 else 0.0
            else:
                p = 1.0 / (1.0 + np.exp((eff - center) / scale))
            recs.append({"block_id": b, "offset": o, "residual": r, "actual_bias": r,
                         "eff": round(eff, 4), "eff_signed": round(eff_s, 4),
                         "success": int(rng.random() < p), "collision_confounded": False})
    return recs


def test_logistic_recovers_center_and_scale():
    fit = BE.fit_logistic_edge(_synth(center=0.03, scale=0.004))
    assert fit["converged"]
    assert abs(fit["center"] - 0.03) < 0.006
    assert 0.001 < fit["scale"] < 0.012


def test_isotonic_monotone_and_crosses_half():
    iso = BE.fit_isotonic_edge(_synth())
    p = iso["success_pred"]
    assert all(p[i] >= p[i + 1] - 1e-9 for i in range(len(p) - 1))   # non-increasing
    assert iso["edge_center_cross0.5"] is not None
    assert 0.015 < iso["edge_center_cross0.5"] < 0.045


def test_block_bootstrap_ci_contains_truth():
    boot = BE.block_bootstrap_edge(_synth(center=0.03, scale=0.004), n_boot=300)
    lo, hi = boot["center_ci"]
    assert lo <= 0.03 <= hi
    assert boot["n_blocks"] == 18


def test_eff03_variation_detected_on_soft_edge():
    e = BE.eff03_variation(_synth(center=0.03, scale=0.004))
    assert e["available"] and e["has_both_labels"]


def test_residual_label_variation_soft_edge():
    rlv = BE.residual_label_variation(_synth(center=0.03, scale=0.004))
    assert rlv["any_block_level_variation"]


def test_symmetry_not_severe_on_symmetric_synth():
    s = BE.symmetry_test(_synth())
    assert not s["severe_asymmetry"]


def test_exit_criteria_pass_on_good_synth():
    recs = _synth(center=0.03, scale=0.004)
    fit = BE.fit_logistic_edge(recs)
    rlv = BE.residual_label_variation(recs)
    e = BE.eff03_variation(recs)
    res = BE.evaluate_exit_criteria(
        eff025_035_both_labels=e["has_both_labels"],
        residual_block_variation=rlv["any_block_level_variation"],
        edge_center=fit["center"], edge_scale=fit["scale"], edge_estimable=fit["converged"],
        severe_asymmetry=False, collision_sensor_available=True, instrumentation_complete=True,
        result_collision_dominated=False)
    assert res["verdict"] == "PASS_TO_V4"


def test_exit_criteria_modify_on_hard_threshold():
    # a perfectly hard band -> no soft edge, no operating-region variance -> MODIFY
    recs = _synth(hard=True)
    rlv = BE.residual_label_variation(recs)
    e = BE.eff03_variation(recs)
    res = BE.evaluate_exit_criteria(
        eff025_035_both_labels=e["has_both_labels"],
        residual_block_variation=rlv["any_block_level_variation"],
        edge_center=0.02, edge_scale=1e-6, edge_estimable=True,
        severe_asymmetry=False, collision_sensor_available=True, instrumentation_complete=True,
        result_collision_dominated=False)
    assert res["verdict"] == "MODIFY_RESIDUAL_OR_DESIGN"
    assert res["modify_triggers"]["success_still_hard_threshold"]


def test_exit_criteria_modify_when_collision_dominated():
    res = BE.evaluate_exit_criteria(
        eff025_035_both_labels=True, residual_block_variation=True, edge_center=0.03, edge_scale=0.004,
        edge_estimable=True, severe_asymmetry=False, collision_sensor_available=True,
        instrumentation_complete=True, result_collision_dominated=True)
    assert res["verdict"] == "MODIFY_RESIDUAL_OR_DESIGN"
