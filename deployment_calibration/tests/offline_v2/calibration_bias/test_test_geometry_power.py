"""Tests for the test-geometry learned-selector power study (offline). No Isaac, no data generation."""

import numpy as np
import pytest

from deployment_calibration.offline_v2.calibration_bias import test_geometry_power as T
from deployment_calibration.offline_v2.calibration_bias import learned_selector_power as L


# ---------------- frozen geometry ----------------
def test_primary_test_nominal_is_pm_035():
    assert T.TEST_NOMINALS_PRIMARY == (-0.035, 0.035)


def test_sensitivity_nominals_do_not_replace_primary():
    # sensitivity anchors are distinct from the primary and are the pre-registered +-0.034 / +-0.036
    assert T.TEST_NOMINALS_SENS_TIGHT == (-0.034, 0.034)
    assert T.TEST_NOMINALS_SENS_WIDE == (-0.036, 0.036)
    assert T.TEST_NOMINALS_PRIMARY not in (T.TEST_NOMINALS_SENS_TIGHT, T.TEST_NOMINALS_SENS_WIDE)


def test_bank_and_train_val_frozen_from_prior_phase():
    assert T.BANK == (-0.04, 0.0, 0.04)
    assert T.TRAIN_NOMINALS == (-0.04, -0.02, 0.0, 0.02, 0.04)
    assert T.VAL_NOMINALS == (-0.01, 0.01)
    assert T.PROBE_OFFSET == -0.04
    assert T.DEEPSETS_HP == dict(d_hid=32, d_emb=16, lr=1e-2, max_epochs=200, patience=25, l2=1e-4)
    assert T.MODEL_SEEDS == (1103, 2207, 3301, 4409, 5519)


# ---------------- support ----------------
def test_test_actual_support_inside_train_support():
    lo_tr, hi_tr = min(T.TRAIN_NOMINALS) + L.RESIDUAL_LO, max(T.TRAIN_NOMINALS) + L.RESIDUAL_HI
    for tn in T.TEST_NOMINALS_PRIMARY:
        assert tn + L.RESIDUAL_LO >= lo_tr - 1e-12
        assert tn + L.RESIDUAL_HI <= hi_tr + 1e-12


# ---------------- geometry audit ----------------
def test_geometry_audit_passes():
    a = T.geometry_audit()
    assert a["GEOMETRY_AUDIT_PASSED"] is True
    assert a["state_aware"]["all_support_success"] is True
    assert a["best_single"]["offset_is_zero_and_crosses_edge"] is True
    assert a["no_robust_common_action"]["no_common_action"] is True
    assert a["probe"]["deterministic_1bit"] is True
    assert a["test_support_within_train"] is True


def test_state_aware_succeeds_over_full_support():
    for tn in T.TEST_NOMINALS_PRIMARY:
        off = min(T.BANK, key=lambda o: abs(tn + o))
        for tau in (T.TAU_LOWER, T.TAU_POINT, T.TAU_UPPER, T.TAU_ISO):
            for r in np.linspace(L.RESIDUAL_LO, L.RESIDUAL_HI, 21):
                assert L.succ(tn, r, off, tau) == 1


def test_best_single_crosses_edge_non_degenerate():
    a = T.geometry_audit()
    for tau_key, v in a["best_single"]["per_tau"].items():
        assert abs(v["best_single_offset"]) < 1e-9          # offset 0
        for frac in v["test_success_frac"].values():
            assert 0.02 < frac < 0.98                        # genuinely straddles the edge


def test_no_robust_common_action():
    a = T.geometry_audit()
    for v in a["no_robust_common_action"]["per_tau"].values():
        assert v["max_common"] < 0.90


def test_probe_is_deterministic_1bit():
    # over the whole residual support, probe -0.04 succeeds at +0.035 and fails at -0.035, every tau
    for tau in (T.TAU_LOWER, T.TAU_POINT, T.TAU_UPPER, T.TAU_ISO):
        for r in np.linspace(L.RESIDUAL_LO, L.RESIDUAL_HI, 21):
            assert L.succ(0.035, r, T.PROBE_OFFSET, tau) == 1
            assert L.succ(-0.035, r, T.PROBE_OFFSET, tau) == 0


# ---------------- selection isolation / leakage ----------------
def test_best_single_uses_train_val_only_and_is_zero():
    rng = np.random.default_rng(3)
    tr, va, te = T.make_dataset(12, 9, 12, T.TAU_POINT, rng, T.TEST_NOMINALS_PRIMARY)
    off = L.best_single(tr, va, T.TAU_POINT)          # signature takes only train+val
    assert off == 0.0


def test_test_blocks_disjoint_from_train_val():
    rng = np.random.default_rng(4)
    tr, va, te = T.make_dataset(9, 6, 9, T.TAU_POINT, rng, T.TEST_NOMINALS_PRIMARY)
    tr_ids = {s["block_id"] for s in tr}; va_ids = {s["block_id"] for s in va}
    te_ids = {s["block_id"] for s in te}
    assert tr_ids.isdisjoint(va_ids) and tr_ids.isdisjoint(te_ids) and va_ids.isdisjoint(te_ids)


def test_test_sessions_only_use_primary_nominals():
    rng = np.random.default_rng(5)
    _, _, te = T.make_dataset(9, 6, 9, T.TAU_POINT, rng, T.TEST_NOMINALS_PRIMARY)
    assert {round(s["nominal"], 6) for s in te} == {-0.035, 0.035}


def test_one_residual_per_block_shared_across_nominals():
    rng = np.random.default_rng(6)
    _, _, te = T.make_dataset(9, 6, 9, T.TAU_POINT, rng, T.TEST_NOMINALS_PRIMARY)
    by_block = {}
    for s in te:
        by_block.setdefault(s["block_id"], set()).add(round(s["residual"], 12))
    assert all(len(v) == 1 for v in by_block.values())


def test_k1_history_is_the_fixed_probe_offset():
    rng = np.random.default_rng(7)
    tr, _, _ = T.make_dataset(9, 6, 9, T.TAU_POINT, rng, T.TEST_NOMINALS_PRIMARY)
    for s in tr:
        for _, H in s["candidates"]:
            assert len(H) == 1 and H[0]["theta"]["grasp_offset_local_y"] == T.PROBE_OFFSET


# ---------------- structural vs learned, bootstrap, power event ----------------
def test_struct_gain_larger_than_prior_phase():
    # the whole point: structural gain at +-0.035 is >> the ~0.18 of the +-0.03 geometry
    rep = T.run_replicate(9, 6, 9, T.TAU_POINT, 0, T.TEST_NOMINALS_PRIMARY)
    assert rep["mean_struct_gain"] > 0.4
    assert rep["oracle_test_success"] >= 0.99
    assert rep["best_single_test_success"] < 0.6


def test_block_bootstrap_full_resample_deterministic():
    g = [0.5, 0.5, 0.5, 0.5]
    lo, hi = L.block_bootstrap_ci(g, np.random.default_rng(0), n_boot=200)
    assert abs(lo - 0.5) < 1e-9 and abs(hi - 0.5) < 1e-9      # degenerate -> exact


def test_power_event_all_conditions_required():
    good = {"gain_block": [0.4, 0.6, 0.5], "mean_gain": 0.5, "oracle_test_success": 1.0,
            "best_single_test_success": 0.45, "seed_gains": [0.5, 0.5, 0.5, 0.5, 0.5]}
    assert L.power_event(good, ci_low=0.3)["event"] is True
    # fails if CI low below threshold
    assert L.power_event(good, ci_low=0.10)["event"] is False
    # fails if a seed goes negative
    bad = dict(good, seed_gains=[0.5, 0.5, 0.5, 0.5, -0.1])
    assert L.power_event(bad, ci_low=0.3)["event"] is False
    # fails if oracle below 0.85
    bad2 = dict(good, oracle_test_success=0.5)
    assert L.power_event(bad2, ci_low=0.3)["event"] is False


def test_struct_power_event_uses_oracle_gain():
    rep = {"struct_gain_block": [0.5, 0.6, 0.55], "mean_struct_gain": 0.55, "oracle_test_success": 1.0,
           "best_single_test_success": 0.45}
    assert T.struct_power_event(rep, ci_low=0.4) is True
    assert T.struct_power_event(rep, ci_low=0.1) is False


def test_model_seeds_all_used():
    rep = T.run_replicate(9, 6, 9, T.TAU_POINT, 1, T.TEST_NOMINALS_PRIMARY)
    assert len(rep["seed_gains"]) == len(T.MODEL_SEEDS)


def test_deterministic_rerun():
    a = T.run_replicate(9, 6, 9, T.TAU_POINT, 42, T.TEST_NOMINALS_PRIMARY)
    b = T.run_replicate(9, 6, 9, T.TAU_POINT, 42, T.TEST_NOMINALS_PRIMARY)
    assert a["mean_gain"] == b["mean_gain"] and a["k1_test_success"] == b["k1_test_success"]


# ---------------- probe cost ----------------
def test_net_voi_horizon_formula():
    from deployment_calibration.offline_v2.calibration_bias import run_test_geometry_power as R
    nv = R.amortized_net_voi(0.5)
    cost = 0.02 * 16.58
    assert abs(nv["net_voi_1task"] - (0.5 - cost)) < 1e-9
    assert abs(nv["horizon"]["10"] - (10 * 0.5 - cost)) < 1e-9
    assert nv["horizon"]["2"] > nv["horizon"]["1"]


# ---------------- translation equivalence anchor ----------------
def test_translation_equivalence_306_hash_unchanged():
    import hashlib, os
    p = "deployment_calibration/data/band_edge_full_306_v1/episodes.jsonl"
    if not os.path.exists(p):
        pytest.skip("306 data not present in this checkout")
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    assert h == "131750e158032e53e5c8daab6e94530c70de1224f05b8fc79a7497009524c20e"
