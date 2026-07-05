"""Tests for confirmatory preregistration v4 (offline). No Isaac, no generator, no data."""

import json
import os

import pytest

from deployment_calibration.offline_v2.calibration_bias import preregistration_v4 as P
from deployment_calibration.models_v2 import features as FEAT


def _docs():
    return P._docs_dir()


@pytest.fixture(scope="module")
def pre():
    return P.preregistration_dict()


# ---------------- frozen design ----------------
def test_exact_nominals(pre):
    d = pre["design"]
    assert tuple(d["train_nominals"]) == (-0.04, -0.02, 0.0, 0.02, 0.04)
    assert tuple(d["val_nominals"]) == (-0.01, 0.01)
    assert tuple(d["test_nominals"]) == (-0.035, 0.035)


def test_design_a_bank_and_probe(pre):
    assert tuple(pre["design"]["candidate_bank"]) == (-0.04, 0.0, 0.04)
    assert pre["design"]["probe_offset"] == -0.04 and pre["design"]["K"] == 1


def test_blocks_9_6_9(pre):
    b = pre["design"]["blocks"]
    assert (b["train"], b["val"], b["test"]) == (9, 6, 9)


def test_session_and_trial_counts(pre):
    tc = pre["trial_counts"]
    assert tc["sessions_train"] == 45 and tc["sessions_val"] == 12 and tc["sessions_test"] == 18
    assert tc["sessions_total"] == 75
    assert tc["trials_per_session"] == 4
    assert tc["full_task_trials_total"] == 300


def test_residual_one_per_block_shared_semantics(pre):
    r = pre["design"]["residual"]
    assert r["sigma"] == 0.005 and r["support"] == [-0.01, 0.01]
    # spec text asserts one residual per block, constant across trials in the block
    txt = open(os.path.join(_docs(), "preregistration_v4.md")).read()
    assert "one draw per block" in txt and "constant" in txt


def test_test_actual_support_within_train(pre):
    lo_tr, hi_tr = pre["design"]["train_actual_support"]
    for tn, (lo, hi) in pre["design"]["test_actual_support"].items():
        assert lo >= lo_tr - 1e-12 and hi <= hi_tr + 1e-12


# ---------------- probe vs candidate separation ----------------
def test_probe_and_candidate_are_separate_trials(pre):
    tc = pre["trial_counts"]
    assert tc["probe_trials_per_session"] == 1
    assert tc["candidate_trials_per_session"] == 3
    assert tc["trials_per_session"] == tc["probe_trials_per_session"] + tc["candidate_trials_per_session"]


# ---------------- model freeze ----------------
def test_model_hyperparameters_and_seeds(pre):
    m = pre["model"]
    assert m["hyperparameters"] == {"d_hid": 32, "d_emb": 16, "lr": 1e-2,
                                    "max_epochs": 200, "patience": 25, "l2": 1e-4}
    assert tuple(m["model_seeds"]) == (1103, 2207, 3301, 4409, 5519)
    assert m["no_hyperparameter_search_on_confirmatory"] is True


def test_model_hp_matches_power_sim():
    from deployment_calibration.offline_v2.calibration_bias import learned_selector_power as L
    assert P.DEEPSETS_HP == L.DEEPSETS_HP
    assert tuple(P.MODEL_SEEDS) == tuple(L.MODEL_SEEDS)


# ---------------- history allowlist / secret denylist ----------------
def test_history_allowlist_matches_feature_code(pre):
    assert P.verify_allowlist_matches_code() is True
    assert pre["history_allowlist"] == list(FEAT.PROBE_KEYS)
    assert len(pre["history_allowlist"]) == FEAT.PROBE_DIM == 8


def test_static_allowlist_dim(pre):
    assert len(pre["static_allowlist"]) == FEAT.STATIC_DIM == 8


def test_secret_denylist_covers_hidden(pre):
    dl = set(pre["secret_denylist"])
    for k in ["nominal_bias", "residual_bias", "actual_bias", "eff_signed", "abs_eff",
              "nuisance_block_id", "block_seed", "residual_seed", "split_identity",
              "future_candidate_outcomes", "oracle_action", "hidden_state_class_label",
              "secret_deployment_state", "hidden_state_id", "damping_eff"]:
        assert k in dl


def test_k0_k1_capacity_matched(pre):
    cap = pre["secondary"]["k0_k1_capacity"]
    assert "DeepSets" in cap and "same" in cap.lower()
    assert "empty history" in cap


# ---------------- best-single ----------------
def test_best_single_train_val_only_not_hardcoded(pre):
    bs = pre["best_single"]
    assert "train + validation" in bs["data"] and "ONLY" in bs["data"]
    assert bs["hardcoded_in_confirmatory"] is False
    assert bs["exploratory_expectation"] == 0.0
    # fix1: no secret tie-break
    rule = " ".join(bs["rule"]).lower()
    assert "tau" not in rule and "eff" not in rule


# ---------------- primary / CI / gate ----------------
def test_primary_comparator_and_threshold(pre):
    pr = pre["primary"]
    assert "best-single" in pr["comparator"] and "B2_K1" in pr["comparator"]
    assert pr["statistical_unit"] == "test nuisance block"
    assert ">= 0.15" in pr["primary_pass"]
    for bad in ["point estimate as primary", "CI>0 as primary", "episode-level bootstrap",
                "lowering 0.15", "changing the comparator"]:
        assert bad in pr["not_allowed"]


def test_bootstrap_frozen(pre):
    b = pre["bootstrap"]
    assert b["n_boot"] == 2000 and b["ci"] == 0.95
    assert "test nuisance block" in b["resample_unit"]


def test_seed_aggregation_then_block(pre):
    txt = open(os.path.join(_docs(), "confirmatory_v4_analysis_plan.md")).read()
    # per test block: average over the 2 test nominals AND over the 5 model seeds; unit is the block
    assert "mean over {2 test nominals}" in txt and "mean over {5 seeds}" in txt
    assert pre["primary"]["statistical_unit"] == "test nuisance block"


def test_seed_gate_primary(pre):
    g = pre["primary"]["plus_gate"]
    assert g[">=4/5 seed point gains >= 0.15"] is True and g["no seed point gain < 0"] is True
    assert "PRIMARY" in g["role"]


# ---------------- collision / technical invalidation ----------------
def test_collision_rule_frozen(pre):
    c = pre["collision_rule"]
    assert c["numeric_threshold_N"] == 0.0
    assert "intention-to-analyze" in c["primary"]
    assert c["confirmatory_data_not_used_to_set_threshold"] is True
    assert "non-finger" in c["monitored_scope"]


def test_technical_invalidation_frozen(pre):
    t = pre["technical_invalid"]
    assert t["never_rerun_on_task_outcome"] is True
    assert t["resample_residual_or_nuisance_on_retry"] is False
    assert t["replace_block_allowed"] is False
    assert isinstance(t["max_technical_invalid_trials"], int)
    # legitimate failures never invalid
    assert any("timeout" in x for x in t["legitimate_failures_never_invalid"])


# ---------------- seal/unseal ----------------
def test_seal_unseal_order(pre):
    s = pre["seal_unseal"]
    # fix1: unique scheme, steps 0..8; model-freeze (Step 5) precedes test-manifest generation (Step 6)
    assert pre["seal_scheme"] == "SCHEME_2_MODEL_FREEZE_BEFORE_TEST_GENERATION"
    assert s[0].startswith("Step 0")
    i5 = next(i for i, x in enumerate(s) if x.startswith("Step 5"))
    i6 = next(i for i, x in enumerate(s) if x.startswith("Step 6"))
    assert i5 < i6
    for bad in ["change threshold", "change bank", "change seeds", "retrain"]:
        assert bad in pre["post_unseal_forbidden"]


# ---------------- data isolation / claim scope ----------------
def test_no_exploratory_data_reuse(pre):
    di = pre["data_isolation"]
    assert di["306_explore_not_in_confirmatory"] is True
    assert di["capability_map_not_in_training"] is True
    assert di["synthetic_power_data_not_in_training"] is True
    assert di["test_inaccessible_until_model_frozen"] is True


def test_claim_scope_wording(pre):
    cs = pre["claim_scope"]
    assert cs["probe_term"] == "repeated-task deployment diagnostic full-task trial"
    for bad in ["one-shot online adaptation", "low-cost internal query", "non-destructive probe",
                "unseen hidden-state generalization"]:
        assert bad in cs["forbidden_terms"]
    assert cs["primary_claim"] == "diagnostic history improves subsequent action-selection success"
    assert cs["net_value_is_secondary"] is True


def test_status_and_no_authorization(pre):
    assert pre["status"] == "PREREGISTRATION_V4_FIX2_READY_FOR_FINAL_C_REAUDIT"
    for k in ["no confirmatory generator", "no confirmatory data", "no run authorization"]:
        assert k in pre["hard_constraints"]


# ---------------- markdown / json consistency ----------------
def test_emitted_json_matches_module(tmp_path):
    P.emit(str(tmp_path))
    j = json.load(open(os.path.join(str(tmp_path), "preregistration_v4.json")))
    assert j["trial_counts"]["full_task_trials_total"] == 300
    assert j["design"]["test_nominals"] == [-0.035, 0.035]
    assert j["status"] == "PREREGISTRATION_V4_FIX2_READY_FOR_FINAL_C_REAUDIT"


def test_docs_json_present_and_consistent(pre):
    docs = _docs()
    for fn in ["preregistration_v4.json", "confirmatory_v4_config.json",
               "confirmatory_v4_audit_checklist.json", "preregistration_v4.md",
               "confirmatory_v4_schema.md", "confirmatory_v4_analysis_plan.md",
               "confirmatory_v4_manifest_spec.md", "confirmatory_v4_seal_unseal_protocol.md",
               "confirmatory_v4_claim_scope.md"]:
        assert os.path.exists(os.path.join(docs, fn)), fn
    j = json.load(open(os.path.join(docs, "preregistration_v4.json")))
    assert j["trial_counts"]["sessions_total"] == 75
    # markdown states the same headline count
    md = open(os.path.join(docs, "preregistration_v4.md")).read()
    assert "300 full-task trials" in md or "75×4 = 300" in md


def test_306_hash_unchanged():
    import hashlib
    p = "deployment_calibration/data/band_edge_full_306_v1/episodes.jsonl"
    if not os.path.exists(p):
        pytest.skip("306 data not present")
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == \
        "131750e158032e53e5c8daab6e94530c70de1224f05b8fc79a7497009524c20e"
