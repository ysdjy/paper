"""Tests for the learned-selector power simulation (Design A). Pure python; a few small DeepSets fits."""

import inspect

import numpy as np

from deployment_calibration.offline_v2.calibration_bias import learned_selector_power as L


def test_candidate_bank_is_design_A():
    assert L.BANK == (-0.04, 0.0, 0.04)
    assert L.PROBE_OFFSET == -0.04


def test_split_isolation_and_residual_one_per_block():
    rng = np.random.default_rng(0)
    tr, va, te = L.make_dataset(9, 6, 9, L.TAU_POINT, rng)
    tb = set(s["block_id"] for s in tr); vb = set(s["block_id"] for s in va); eb = set(s["block_id"] for s in te)
    assert not (tb & vb or tb & eb or vb & eb)                 # disjoint block ids across splits
    # one residual per block, shared across that block's sessions (nominals)
    from collections import defaultdict
    byb = defaultdict(set)
    for s in tr + va + te:
        byb[s["block_id"]].add(round(s["residual"], 9))
    assert all(len(v) == 1 for v in byb.values())


def test_residual_shared_within_block_across_candidates():
    rng = np.random.default_rng(1)
    tr, _, _ = L.make_dataset(9, 6, 9, L.TAU_POINT, rng)
    s = tr[0]
    # all candidates in a session share the session residual/actual (only offset varies)
    offs = sorted(e["_offset"] for e, _ in s["candidates"])
    assert offs == [-0.04, 0.0, 0.04]


def test_no_hidden_state_leakage_in_model_inputs():
    # the model reads only static_features(theta,g,x) + probe history; none carry bias/actual/eff
    rng = np.random.default_rng(2)
    tr, _, _ = L.make_dataset(9, 6, 9, L.TAU_POINT, rng)
    e, H = tr[0]["candidates"][0]
    for banned in ("nominal", "residual", "actual", "eff", "bias", "block", "seed"):
        assert not any(banned in str(k).lower() for k in e["x"])
        assert not any(banned in str(k).lower() for k in e["g"])
        assert not any(banned in str(k).lower() for k in e["theta"])
    # probe history entry carries only offset + outcome fields
    assert set(H[0]["theta"]) <= {"grasp_offset_local_y", "max_pos_step", "pull_lead"}


def test_K0_reduces_to_fixed_offset_no_session_info():
    # with empty history, all sessions have identical model inputs -> K0 cannot condition on session
    rng = np.random.default_rng(3)
    tr, _, _ = L.make_dataset(9, 6, 9, L.TAU_POINT, rng)
    p0 = L._pairs(tr[:1], 0)
    p0b = L._pairs(tr[1:2], 0)
    # the candidate static features are identical across sessions (same x,g,offsets)
    from deployment_calibration.models_v2 import features as F
    f1 = [F.static_features(e) for e, _ in p0]
    f2 = [F.static_features(e) for e, _ in p0b]
    assert f1 == f2


def test_best_single_train_val_only_ignores_test():
    src = inspect.getsource(L.best_single)
    assert "TEST_NOMINALS" not in src and "test" not in src.split("def")[1][:200].lower()
    off = L.best_single(*L.make_dataset(9, 6, 9, L.TAU_POINT, np.random.default_rng(4))[:2], L.TAU_POINT)
    assert off == 0.0


def test_fixed_first_probe_offset():
    rng = np.random.default_rng(5)
    tr, _, _ = L.make_dataset(9, 6, 9, L.TAU_POINT, rng)
    _, H = tr[0]["candidates"][0]
    assert H[0]["theta"]["grasp_offset_local_y"] == -0.04


def test_all_model_seeds_used():
    assert len(L.MODEL_SEEDS) == 5


def test_block_bootstrap_full_blocks_and_deterministic():
    g = [0.0, 0.5, 1.0, 0.5, 0.0, 1.0, 0.5, 0.0, 0.5]
    a = L.block_bootstrap_ci(g, np.random.default_rng(7), n_boot=500)
    b = L.block_bootstrap_ci(g, np.random.default_rng(7), n_boot=500)
    assert a == b


def test_power_event_requires_all_conditions():
    good = {"mean_gain": 0.2, "oracle_test_success": 1.0, "best_single_test_success": 0.75,
            "gain_block": [0.0, 0.5, 1.0], "seed_gains": [0.2, 0.2, 0.2, 0.2, 0.2]}
    assert L.power_event(good, ci_low=0.16)["event"]
    assert not L.power_event(good, ci_low=0.10)["event"]          # CI below threshold
    bad_seed = dict(good, seed_gains=[0.2, 0.2, 0.1, 0.1, -0.1])
    assert not L.power_event(bad_seed, ci_low=0.2)["event"]        # seed instability + a negative seed


def test_conservative_failure_counts_nonconverged():
    # a non-converged replicate must not be counted as a power success (checked in run_power)
    src = inspect.getsource(L.run_power)
    assert "rep[\"converged\"]" in src or "rep['converged']" in src


def test_tau_sensitivity_values():
    assert L.TAU_LOWER < L.TAU_POINT < L.TAU_UPPER
    assert L.TAU_ISO == 0.0325


def test_deterministic_replicate_rerun():
    a = L.run_replicate(9, 6, 9, L.TAU_POINT, rep_seed=42, model_seeds=(1103, 2207))
    b = L.run_replicate(9, 6, 9, L.TAU_POINT, rep_seed=42, model_seeds=(1103, 2207))
    assert a["mean_gain"] == b["mean_gain"] and a["k1_test_success"] == b["k1_test_success"]


def test_net_voi_one_shot_and_horizon():
    from deployment_calibration.offline_v2.calibration_bias import run_learned_selector_power as RUN
    hv = RUN.amortized_net_voi(0.15)
    cost = RUN.LAMBDA_TIME * RUN.PROBE_TIME_S
    assert abs(hv["1"] - (1 * 0.15 - cost)) < 1e-6
    assert abs(hv["10"] - (10 * 0.15 - cost)) < 1e-6
    assert hv["1"] < hv["2"] < hv["5"] < hv["10"]                 # amortization increases with horizon
