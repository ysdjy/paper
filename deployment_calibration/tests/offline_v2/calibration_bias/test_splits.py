"""Tests for bias-level split: partition, pairwise disjointness (levels/sessions/seeds), interpolation."""

import pytest

from deployment_calibration.offline_v2.calibration_bias import splits as S
from deployment_calibration.offline_v2.calibration_bias import synthetic


def _eps():
    return synthetic.make_synthetic(replicates=2)


def test_split_partition_and_pairwise_disjoint():
    eps = _eps()
    lv = S.level_values(eps)
    split = S.split_bias_levels(lv, n_test_interior=2, n_val_interior=1)
    man = S.build_manifest(eps, split)
    a = man["audit"]
    assert a["ok"]
    assert a["levels_pairwise_disjoint"]
    assert a["partition_ok"]
    assert a["sessions_disjoint"]
    assert a["nuisance_seeds_disjoint"]


def test_test_levels_are_interior_and_bracketed():
    eps = _eps()
    lv = S.level_values(eps)
    split = S.split_bias_levels(lv, n_test_interior=2, n_val_interior=1)
    man = S.build_manifest(eps, split)
    assert man["audit"]["interpolation_ok"]
    # every test level bracketed by train on both sides
    assert all(man["audit"]["test_levels_bracketed_by_train"].values())
    # test levels are not the extremes
    vals = sorted(lv.values())
    test_vals = [lv[t] for t in split["test"]]
    assert min(vals) not in test_vals and max(vals) not in test_vals


def test_requires_at_least_five_levels():
    import numpy as np
    eps = synthetic.make_synthetic(
        bias_levels=[(f"B{i}", float(v)) for i, v in enumerate(np.linspace(-0.02, 0.02, 4))],
        replicates=2)
    lv = S.level_values(eps)
    with pytest.raises(ValueError, match=">=5 bias levels"):
        S.split_bias_levels(lv)
