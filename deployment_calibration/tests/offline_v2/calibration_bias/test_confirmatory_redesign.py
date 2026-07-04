"""Tests for the offline confirmatory candidate-bank redesign (hard-edge geometry). Pure python."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from deployment_calibration.offline_v2.calibration_bias import confirmatory_redesign as R
from deployment_calibration.offline_v2.calibration_bias import run_confirmatory_redesign as RUN

_REPO = Path(__file__).resolve().parents[4]
DATA = _REPO / "deployment_calibration" / "data" / "band_edge_full_306_v1" / "episodes.jsonl"
EP_SHA = "131750e158032e53e5c8daab6e94530c70de1224f05b8fc79a7497009524c20e"


def test_hard_threshold_and_sensitivity_interval():
    # success is a hard step; sensitivity anchors bracket the identified interval
    assert R.success(0.0, 0.0, 0.03) and not R.success(0.0, 0.0, 0.05)
    assert R.TAU_LOWER < R.TAU_POINT < R.TAU_UPPER
    assert R.TAU_ISOTONIC in R.TAU_SENSITIVITY


def test_design_A_nondegenerate_and_robust():
    d = R.evaluate_design(R.CANDIDATE_BANKS["A_three_point"])
    assert d["best_single_offset"] == 0.0
    assert d["state_aware_test_success"] == 1.0
    assert d["best_single_test_success"] < 1.0        # best-single straddles the edge
    assert d["expected_success_gain"] >= 0.15
    assert d["per_block_gain_nondegenerate"]
    assert d["mixed_best_single_labels"]              # residual flips labels in the primary contrast
    assert not d["mixed_state_aware_labels"]          # state-aware deterministically succeeds
    assert not d["common_robust_action_exists"]
    assert d["all_states_compensable"]


def test_design_C_best_single_is_thin_tie_fragile():
    # current 7-point bank: best-single 0 wins by only a thin margin over +-0.02 -> fragile -> rejected
    margin = RUN._top2_margin(R.CANDIDATE_BANKS["C_current_seven_point"])
    assert margin < 0.15
    assert not RUN._stable_margin(None, R.CANDIDATE_BANKS["C_current_seven_point"])
    # forcing best-single to -0.02 (the near-tie alternative) is degenerate at test
    per = [(nb, 1 if R.success(nb, r, -0.02) else 0) for nb in R.TEST_NOMINALS for r in R.EMPIRICAL_RESIDUALS]
    from collections import defaultdict
    by = defaultdict(list)
    for nb, s in per:
        by[nb].append(s)
    assert all(len(set(v)) == 1 for v in by.values())   # deterministic per test nominal -> degenerate


def test_design_A_margin_wide():
    assert RUN._top2_margin(R.CANDIDATE_BANKS["A_three_point"]) >= 0.15


def test_best_single_uses_train_val_only_not_test():
    # perturb the TEST outcomes conceptually: best_single_offset must not depend on test nominals
    off1, _ = R.best_single_offset(R.CANDIDATE_BANKS["A_three_point"])
    # recompute with a different test set (should be identical: best-single ignores test)
    import inspect
    src = inspect.getsource(R.best_single_offset)
    assert "TEST_NOMINALS" not in src      # selection never references the test set
    assert off1 == 0.0


def test_common_action_check():
    c = R.common_robust_action(R.CANDIDATE_BANKS["A_three_point"])
    assert not c["robust_common_action_exists"]        # no offset robustly covers BOTH test biases


def test_compensability():
    assert R.all_states_compensable(R.CANDIDATE_BANKS["A_three_point"])["all_compensable"]


def test_per_block_gain_variance_positive_A():
    d = R.evaluate_design(R.CANDIDATE_BANKS["A_three_point"])
    assert d["per_block_gain_variance"] > 0.0


def test_residual_operating_cell_label_variation_A():
    # at the test biases, the best-single (offset 0) success varies across blocks (residual flips it)
    bs = 0.0
    for nb in R.TEST_NOMINALS:
        labels = set(1 if R.success(nb, r, bs) else 0 for r in R.EMPIRICAL_RESIDUALS)
        assert len(labels) == 2                       # both success and failure present


def test_probe_K1_fixed_first_probe_semantics():
    pr = R.probe_analysis(R.CANDIDATE_BANKS["A_three_point"], (-0.04, 0.04))
    assert pr["K_selected_success"][0] < pr["K_selected_success"][1]   # K1 > K0
    assert pr["K_selected_success"][1] == 1.0
    assert pr["first_probe_distinguishes_test_nominals"]
    # K2 does not regress vs K1 (redundant on the two test biases)
    assert pr["K_selected_success"][2] >= pr["K_selected_success"][1] - 1e-9


def test_deterministic_output():
    a = R.evaluate_design(R.CANDIDATE_BANKS["A_three_point"])
    b = R.evaluate_design(R.CANDIDATE_BANKS["A_three_point"])
    assert a["expected_success_gain"] == b["expected_success_gain"]
    assert a["per_block_gain_variance"] == b["per_block_gain_variance"]


def test_verdict_selects_design_A():
    rows = RUN.evaluate_all()
    a = next(r for r in rows if r["design_id"] == "A_three_point")
    c = next(r for r in rows if r["design_id"] == "C_current_seven_point")
    assert a["all_criteria_met"]
    assert not c["all_criteria_met"]


@pytest.mark.skipif(not DATA.exists(), reason="306 data not present")
def test_no_source_data_mutation():
    before = hashlib.sha256(DATA.read_bytes()).hexdigest()
    RUN.evaluate_all()
    after = hashlib.sha256(DATA.read_bytes()).hexdigest()
    assert before == after == EP_SHA
