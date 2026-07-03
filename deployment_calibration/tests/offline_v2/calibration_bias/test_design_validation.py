"""Pre-data confirmatory-split validation tests (the v1 flaw / v2 fix).

Enforces: held-out test levels must be bracketed by train; the per-test-bias success-offset sets must
have an EMPTY intersection (no single offset covers all test biases); if such a common offset exists,
validation fails.
"""

import pytest

from deployment_calibration.offline_v2.calibration_bias import design_validation as DV

OFFSETS = [-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06]
TRAIN = [-0.04, -0.02, 0.0, 0.02, 0.04]


def test_v1_split_rejected_common_offset():
    # v1 test {-0.01, +0.01}: offset 0.0 satisfies the band for both -> best-single covers both -> reject
    r = DV.validate_confirmatory_split(TRAIN, [-0.03, 0.03], [-0.01, 0.01], OFFSETS)
    assert not r["ok"]
    assert r["common_success_offset_across_test"] == [0.0]
    assert not r["empty_intersection"]
    assert r["max_achievable_success_only_gain"] == 0.0
    assert "success band for EVERY test bias" in r["reason"]


def test_v2_split_accepted_empty_intersection():
    r = DV.validate_confirmatory_split(TRAIN, [-0.01, 0.01], [-0.03, 0.03], OFFSETS)
    assert r["ok"]
    assert r["common_success_offset_across_test"] == []
    assert r["empty_intersection"]
    assert r["all_test_bracketed"]
    assert r["max_achievable_success_only_gain"] == 0.5
    # explicit success sets: -0.03 -> {+0.02,+0.04}; +0.03 -> {-0.04,-0.02}
    assert r["test_success_offset_sets"]["-0.03"] == [0.02, 0.04]
    assert r["test_success_offset_sets"]["0.03"] == [-0.04, -0.02]


def test_unbracketed_test_rejected():
    # test bias +0.05 lies outside the train range -> extrapolation -> reject
    r = DV.validate_confirmatory_split(TRAIN, [0.0], [0.05], OFFSETS)
    assert not r["ok"]
    assert not r["all_test_bracketed"]


def test_common_band_candidate_forces_failure():
    # a single test bias trivially has a common offset with itself -> reject (no discrimination)
    r = DV.validate_confirmatory_split(TRAIN, [0.0], [0.0], OFFSETS)
    assert not r["ok"]
    assert len(r["common_success_offset_across_test"]) > 0


def test_predicted_success_offsets_geometry():
    assert DV.predicted_success_offsets(-0.03, OFFSETS) == {0.02, 0.04}
    assert DV.predicted_success_offsets(0.0, OFFSETS) == {-0.02, 0.0, 0.02}
