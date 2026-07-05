"""Post-fix PASS tests for preregistration v4 FIX1 (offline). No Isaac, no generator, no data.

These encode the four blocker fixes (A/B/C/D) + minor issues as ordinary PASS assertions. They intentionally
mirror the requirements of Claude C's strict-xfail audit tests; C's file is preserved unmodified as historical
evidence (its xfails now XPASS-strict, which is the intended signal that the blockers are fixed).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from deployment_calibration.offline_v2.calibration_bias import preregistration_v4 as P
from deployment_calibration.offline_v2.calibration_bias import learned_selector_power as L
from deployment_calibration.offline_v2.calibration_bias import test_geometry_power as T
from deployment_calibration.models_v2 import features as FEAT

_DOCS = Path(P._docs_dir())


def _cfg():
    return json.loads((_DOCS / "confirmatory_v4_config.json").read_text())


def _pre():
    return json.loads((_DOCS / "preregistration_v4.json").read_text())


# ---------------- status ----------------
def test_status_is_fix1():
    assert P.STATUS == "PREREGISTRATION_V4_BLOCK_STATE_KAT_GATE_READY_FOR_C_FINAL_REAUDIT"
    assert _pre()["status"] == P.STATUS
    assert _pre()["power_recertification_required"] is False


# ================= BLOCKER A: no secret best-single tie-break =================
def test_best_single_rule_has_no_secret():
    rule = " ".join(_cfg()["best_single"]["rule"]).lower()
    assert "eff" not in rule and "|eff|" not in rule and "tau" not in rule
    fb = " ".join(_cfg()["best_single"]["forbidden_inputs"]).lower()
    for k in ("actual bias", "tau-|eff|", "oracle", "test outcomes"):
        assert k in fb


def test_best_single_legal_matches_rule_and_is_zero():
    # legal rule: max mean success -> min|offset| -> bank order; on the frozen design -> offset 0
    rng = np.random.default_rng(0)
    tr, va, _ = T.make_dataset(9, 6, 9, L.TAU_POINT, rng, (-0.035, 0.035))
    assert L.best_single_legal(tr, va, L.TAU_POINT) == 0.0


def test_best_single_legal_no_secret_in_source():
    import re
    src = (Path(P.__file__).resolve().parent / "learned_selector_power.py").read_text()
    body = src.split("def best_single_legal")[1].split("\ndef ")[0]
    code = re.sub(r'""".*?"""', "", body, flags=re.S)          # strip the docstring prose
    # the illegal secret tie-break computed a tau-|eff| margin over actual_bias; the legal code must not.
    # (`tau` as the success-label threshold passed to succ() is legal; the secret was the MARGIN.)
    assert "margins" not in code
    assert "tau -" not in code and "tau-" not in code
    assert "eff" not in code
    # sanity: the legacy function DOES contain the illegal margin (proves the check discriminates)
    legacy = src.split("def best_single(")[1].split("\ndef ")[0]
    assert "margins" in legacy


def test_selection_invariance_proof_present_and_clean():
    inv = json.loads((_DOCS / "best_single_tiebreak_invariance_v1.json").read_text())
    assert inv["step1_tie_count"] == 0
    assert inv["selection_mismatch_count"] == 0
    assert inv["all_replicates_offset_zero"] is True
    assert inv["power_recertification_required"] is False
    assert inv["total_replicates_checked"] == 4500


def test_selection_invariance_recompute_small():
    # independently re-verify a slice: old rule == new rule, and step-1 unique, over a few seeds
    for tau in (0.0342, 0.03425, 0.0343):
        for s in range(25):
            rng = np.random.default_rng(1000 + s)
            tr, va, _ = T.make_dataset(9, 6, 9, tau, rng, (-0.035, 0.035))
            old = L.best_single(tr, va, tau)
            new = L.best_single_legal(tr, va, tau)
            assert old == new == 0.0


# ================= BLOCKER B: seeds frozen =================
def test_all_15_seeds_are_exact_ints():
    fs = _cfg()["frozen_seeds"]
    assert len(fs) == 15
    assert all(isinstance(v, int) for v in fs.values())
    # top-level convenience ints also present (audit reads these)
    for name in ("master_seed", "residual_seed", "bootstrap_seed"):
        assert isinstance(_cfg()[name], int)


def test_seed_derivation_recomputes():
    fs = _cfg()["frozen_seeds"]
    for label, val in fs.items():
        assert P.seed(label) == val
    # exact known values
    assert fs["master_seed"] == 6341914557047805261
    assert fs["bootstrap_seed"] == 9014517173581927929


def test_seed_derivation_rule_frozen_in_config():
    c = _cfg()
    assert "sha256" in c["seed_derivation"] and "2**63" in c["seed_derivation"]
    assert c["seed_root"] == "confirmatory-v4|2bf7217907f24ae54a08db71bbdcf624b110ccf0"


def test_label_changes_seed():
    assert P.seed("master_seed") != P.seed("bootstrap_seed")
    assert P.seed("train_residual_seed") != P.seed("test_residual_seed")


def test_subseed_stable_and_identity_addressed():
    ds = P.seed("train_residual_seed")
    assert P.subseed(ds, "block=03") == P.subseed(ds, "block=03")           # stable
    assert P.subseed(ds, "block=03") != P.subseed(ds, "block=04")           # identity-addressed
    # order independence: computing in any order yields the same set
    a = {i: P.subseed(ds, f"block={i:02d}") for i in range(9)}
    b = {i: P.subseed(ds, f"block={i:02d}") for i in reversed(range(9))}
    assert a == b


def test_bootstrap_seed_frozen_before_unseal():
    b = _cfg()["bootstrap"]
    assert b["bootstrap_seed_frozen_before_unseal"] is True
    assert b["bootstrap_seed"] == P.seed("bootstrap_seed")


# ================= BLOCKER C: unique seal scheme =================
def test_seal_scheme_unique_no_either_or():
    pre = _pre()
    assert pre["seal_scheme"] == "SCHEME_2_MODEL_FREEZE_BEFORE_TEST_GENERATION"
    seal = " ".join(pre["seal_unseal"]).lower()
    assert "(or " not in seal and " or generate" not in seal and "sealed (or" not in seal
    assert "either" not in seal


def test_test_manifest_only_after_model_freeze():
    ac = _pre()["access_control"]
    assert ac["test_manifest_only_after_step5"] is True
    assert ac["test_outcomes_nonexistent_before_step6"] is True
    assert ac["early_test_access_verdict"] == "EXPERIMENT_INVALID_EARLY_TEST_ACCESS"
    # step 5 (freeze analysis) precedes step 6 (generate test manifest) in the ordered list
    steps = _pre()["seal_unseal"]
    idx5 = next(i for i, s in enumerate(steps) if s.startswith("Step 5"))
    idx6 = next(i for i, s in enumerate(steps) if s.startswith("Step 6"))
    assert idx5 < idx6


# ================= BLOCKER D: model-failure semantics =================
def test_model_failure_separate_namespace():
    c = _cfg()
    tfh = c["model"]["training_failure_handling"].lower()
    assert "technical-invalid" not in tfh and "section 17" not in tfh
    assert c["model"]["all_seeds_must_converge"] is True
    ic = c["invalid_counters"]
    assert "runtime_trial_invalid_count" in ic and "model_fit_failure_count" in ic
    assert "300" in ic["runtime_trial_invalid_count"]


def test_all_5_seeds_required_no_drop():
    mf = _cfg()["model_fit"]
    assert mf["all_seeds_must_converge"] is True
    assert mf["no_seed_replacement"] is True
    assert mf["no_dropping_bad_seed"] is True
    assert mf["missing_or_invalid_seed_makes_experiment_invalid"] is True
    assert mf["invalid_verdict"] == "EXPERIMENT_INVALID_MODEL_FIT"


def test_retry_at_most_one_deterministic():
    mf = _cfg()["model_fit"]
    assert "at most 1" in mf["retry"]
    for k in ("same data hash", "same model seed", "same hyperparameters"):
        assert k in mf["retry_constraints"]


def test_seed_gate_requires_all_five_valid():
    gate = _pre()["primary"]["plus_gate"]
    assert "all 5 model seeds exist and are technically valid" in gate["precondition"]


def test_runtime_invalid_budget_excludes_model_fit():
    ti = _cfg()["technical_invalid"]
    assert "RUNTIME full-task trials ONLY" in ti["scope"]
    assert ti["max_technical_invalid_trials"] == 15


# ================= minor issues =================
def test_probe_fields_required_no_zero_fill():
    pre = _pre()
    assert pre["history_fields_required"] is True
    assert pre["missing_probe_field_verdict"] == "TECHNICAL_INVALID_SCHEMA"
    assert pre["history_allowlist"] == list(FEAT.PROBE_KEYS) and len(FEAT.PROBE_KEYS) == 8


def test_explicit_hyperparameters_required():
    m = _cfg()["model"]
    assert m["must_pass_all_hyperparameters_explicitly"] is True
    assert m["constructor_defaults_differ"]["max_epochs_default"] == 300
    assert m["constructor_defaults_differ"]["patience_default"] == 30
    assert m["hyperparameters"]["max_epochs"] == 200 and m["hyperparameters"]["patience"] == 25


def test_band_edge_supersession_present_original_untouched():
    assert (_DOCS / "band_edge_exit_verdict_supersession_v1.md").exists()
    sup = json.loads((_DOCS / "band_edge_exit_verdict_supersession_v1.json").read_text())
    assert sup["superseded_file_unchanged"] is True
    assert sup["no_data_recomputation"] is True
    # original still says its old verdict for +-0.03
    orig = json.loads((_DOCS / "band_edge_exit_verdict_v1.json").read_text())
    assert orig["verdict"] == "MODIFY_RESIDUAL_OR_DESIGN"


# ================= unchanged design + consistency =================
def test_design_unchanged():
    d = _cfg()["design"]
    assert d["candidate_bank"] == [-0.04, 0.0, 0.04]
    assert d["test_nominals"] == [-0.035, 0.035]
    assert _cfg()["trial_counts"]["full_task_trials_total"] == 300
    assert _cfg()["trial_counts"]["sessions_total"] == 75


def test_model_hp_matches_power_sim():
    assert P.DEEPSETS_HP == L.DEEPSETS_HP
    assert tuple(P.MODEL_SEEDS) == tuple(L.MODEL_SEEDS)


def test_no_generator_or_data_produced():
    # only docs + module; no manifest instance / generator / runtime data under the calibration_bias dirs
    names = [p.name for p in _DOCS.iterdir()]
    assert not any("manifest_instance" in n or "confirmatory_data" in n for n in names)
    hc = _pre()["hard_constraints"]
    assert "no confirmatory generator" in hc and "no run authorization" in hc


def test_md_json_consistency():
    P.emit()  # re-emit, must match module
    j = _pre()
    assert j["frozen_seeds"]["master_seed"] == 6341914557047805261
    md = (_DOCS / "preregistration_v4.md").read_text()
    assert "PREREGISTRATION_V4_BLOCK_STATE_KAT_GATE_READY_FOR_C_FINAL_REAUDIT" in md
    assert "6341914557047805261" in (_DOCS / "confirmatory_v4_manifest_spec.md").read_text()


def test_306_hash_unchanged():
    p = "deployment_calibration/data/band_edge_full_306_v1/episodes.jsonl"
    if not os.path.exists(p):
        pytest.skip("306 data not present")
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == \
        "131750e158032e53e5c8daab6e94530c70de1224f05b8fc79a7497009524c20e"
