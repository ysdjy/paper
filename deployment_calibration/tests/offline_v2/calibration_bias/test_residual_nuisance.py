"""Tests for the v3 block-level residual calibration nuisance (schema + residual-aware validation)."""

import numpy as np
import pytest

from deployment_calibration.offline_v2.calibration_bias import design_validation as DV
from deployment_calibration.offline_v2.calibration_bias import power as PW
from deployment_calibration.offline_v2.calibration_bias import residual_nuisance as RN

OFFSETS = [-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06]
TRAIN, VAL, TEST = [-0.04, -0.02, 0.0, 0.02, 0.04], [-0.01, 0.01], [-0.03, 0.03]


def test_residual_draws_in_range_zero_mean():
    draws = RN.draw_block_residuals(200, seed=0)
    assert all(-0.01 <= d <= 0.01 for d in draws)          # truncation respected
    assert abs(np.mean(draws)) < 0.002                     # ~zero mean
    assert RN.residual_std_effective() < 0.005             # truncated SD below nominal sigma


def test_actual_bias_and_guard():
    assert abs(RN.actual_bias(-0.03, 0.005) - (-0.025)) < 1e-12
    RN.assert_residual_not_in_x({"episode_id": "e", "x": {"tcp_pos": [0, 0, 0]}})  # clean
    with pytest.raises(ValueError, match="residual/actual"):
        RN.assert_residual_not_in_x({"episode_id": "e", "x": {"residual_bias_y": 0.005}})
    with pytest.raises(ValueError):
        RN.assert_residual_not_in_x({"episode_id": "e", "x": {"actual_bias_y": -0.03}})


def test_actual_bias_stays_compensable():
    # extreme nominal +/-0.04 + residual +/-0.01 -> actual +/-0.05, still inside the 7-offset bank
    comp = RN.actual_bias_in_compensable_range(TRAIN + VAL + TEST, OFFSETS, RN.DEFAULT_RESIDUAL, band=0.02)
    assert comp["ok"]
    assert comp["max_actual_bias"] <= 0.05 + 1e-9


def test_residual_split_validation_passes_and_no_common_offset():
    r = DV.validate_confirmatory_split_residual(TRAIN, VAL, TEST, OFFSETS, RN.DEFAULT_RESIDUAL)
    assert r["ok"]
    assert r["compensable_over_residual_support"]["ok"]
    assert r["no_common_offset_any_residual"]                       # best-single can't cover both test biases
    assert r["best_single_max_test_success_over_residual"] <= 0.5 + 1e-9
    assert r["state_aware_test_success_upper"] == 1.0
    assert r["max_achievable_success_only_gain_residual"] >= 0.15


def test_residual_split_rejects_flawed_v1_test():
    # v1 test {-0.01,+0.01}: offset 0 covers both even under residual -> reject
    r = DV.validate_confirmatory_split_residual(TRAIN, [-0.03, 0.03], [-0.01, 0.01], OFFSETS,
                                                RN.DEFAULT_RESIDUAL)
    assert not r["ok"]


def test_residual_power_supports_floor():
    r = PW.power_at_residual(9, n_sims=150, n_boot=400, seed=3)
    assert r["power"] >= 0.8                                        # >= frozen target at N=9
    assert 0.3 < r["mean_gain"] < 0.75
