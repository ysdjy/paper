"""Unit tests for offline_v2.metrics — the AUROC contract in particular."""

import math
import warnings

import numpy as np

from deployment_calibration.offline_v2 import metrics as M


def test_auroc_constant_is_half():
    # A constant predictor must score 0.5, NOT an argsort artefact.
    y = [0, 1, 0, 1, 1, 0]
    p = [0.3] * 6
    assert abs(M.auroc(y, p) - 0.5) < 1e-9


def test_auroc_perfect_order_is_one():
    y = [0, 0, 1, 1]
    p = [0.1, 0.2, 0.8, 0.9]
    assert abs(M.auroc(y, p) - 1.0) < 1e-9


def test_auroc_perfect_reverse_is_zero():
    y = [0, 0, 1, 1]
    p = [0.9, 0.8, 0.2, 0.1]
    assert abs(M.auroc(y, p) - 0.0) < 1e-9


def test_auroc_single_class_is_nan_with_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v = M.auroc([1, 1, 1], [0.2, 0.7, 0.9])
        assert math.isnan(v)
        assert any("single-class" in str(x.message) for x in w)


def test_auroc_ties_partial():
    # two of the scores tie across classes -> AUROC between 0 and 1, tie-aware
    y = [0, 1, 0, 1]
    p = [0.5, 0.5, 0.2, 0.9]
    v = M.auroc(y, p)
    assert 0.0 <= v <= 1.0
    # matches sklearn reference exactly
    from sklearn.metrics import roc_auc_score
    assert abs(v - roc_auc_score(y, p)) < 1e-9


def test_numpy_fallback_matches_sklearn_random():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 2, size=25)
        if y.min() == y.max():
            continue
        p = rng.random(25)
        from sklearn.metrics import roc_auc_score
        assert abs(M._auroc_numpy(y.astype(float), p) - roc_auc_score(y, p)) < 1e-9


def test_brier_defined_single_class():
    assert abs(M.brier([1, 1], [1.0, 1.0]) - 0.0) < 1e-12
    assert abs(M.brier([1, 1], [0.0, 0.0]) - 1.0) < 1e-12


def test_macro_f1_failure_rewards_minority():
    # all-success predictor on imbalanced data: macro-F1 penalizes missing failure class
    y = [1, 1, 1, 0]
    p = [0.9, 0.9, 0.9, 0.9]  # predicts success everywhere
    v = M.macro_f1_failure(y, p)
    assert v < 0.6  # failure-class F1 is 0 -> macro pulled down


def test_balanced_accuracy_and_confusion():
    y = ["a", "a", "b", "b"]
    p = ["a", "b", "b", "b"]
    assert abs(M.balanced_accuracy(y, p) - 0.75) < 1e-9
    lab, cm = M.confusion_matrix(y, p)
    assert lab == ["a", "b"]
    assert cm.tolist() == [[1, 1], [0, 2]]
