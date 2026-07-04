"""Independent adversarial audit tests (Claude C) for confirmatory preregistration v4.

Read-only. Do NOT modify B's v4 files, runtime, data, or models. These tests encode Claude C's
independent findings; several are EXPECTED-FAIL assertions that document open blockers (marked
xfail) so the audit is machine-checkable and regressions are caught after B fixes them.

Run: pytest deployment_calibration/tests/offline_v2/calibration_bias/test_preregistration_v4_audit_c.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_DOCS = Path(__file__).resolve().parents[4] / "docs" / "offline_v2" / "calibration_bias"


def _cfg():
    return json.loads((_DOCS / "confirmatory_v4_config.json").read_text())


def _prereg():
    return json.loads((_DOCS / "preregistration_v4.json").read_text())


# ---------------------------------------------------------------- design integrity (PASS)
def test_trial_count_300():
    tc = _cfg()["trial_counts"]
    assert tc["sessions_total"] == 75
    assert tc["full_task_trials_total"] == 300
    assert tc["trials_per_session"] == 4  # 1 probe + 3 candidates
    assert 9 * 5 + 6 * 2 + 9 * 2 == 75


def test_test_actual_support_inside_train_support():
    d = _cfg()["design"]
    tr_lo, tr_hi = d["train_actual_support"]
    for _, (lo, hi) in d["test_actual_support"].items():
        assert tr_lo <= lo and hi <= tr_hi, "test actual support must be inside train support"


def test_candidate_bank_and_probe():
    d = _cfg()["design"]
    assert d["candidate_bank"] == [-0.04, 0.0, 0.04]
    assert d["probe_offset"] == -0.04 and d["K"] == 1
    assert d["test_nominals"] == [-0.035, 0.035]


# ---------------------------------------------------------------- feature allowlist == real code (PASS)
def test_history_allowlist_equals_real_probe_vector():
    from deployment_calibration.models_v2 import features as F
    assert list(F.PROBE_KEYS) == [
        "theta.grasp_offset_local_y", "theta.max_pos_step", "theta.pull_lead",
        "success", "task_outcome_error", "skill_elapsed_time",
        "pull_phase_duration", "final_joint_position",
    ]
    assert _cfg()["history_allowlist"] == list(F.PROBE_KEYS)
    assert F.PROBE_DIM == 8 and F.STATIC_DIM == 8


def test_no_secret_field_reachable_by_features():
    from deployment_calibration.models_v2 import features as F
    e = {"theta": {"grasp_offset_local_y": 0.0, "max_pos_step": 0.02, "pull_lead": 0.08},
         "g": {"target_open_position": 0.2, "target_tolerance": 0.02},
         "x": {"initial_mechanism_joint_pos": 0.0, "gripper_width": 0.08, "member": "sektion_cabinet"},
         "secret_deployment_state": {"actual_bias_y": 0.033}}
    feats = F.static_features(e)
    assert 0.033 not in feats  # secret must not leak into the static vector


# ---------------------------------------------------------------- model / loss == real code (PASS)
def test_loss_is_bce_plus_mse_plus_mse_not_2x():
    src = (Path(__file__).resolve().parents[3] / "models_v2" / "torch_models.py").read_text()
    assert "return bce + mse_e + mse_t" in src, "loss must be BCE + MSE + MSE (weight 1 each)"
    assert "2 * mse" not in src and "2*mse" not in src


# ---------------------------------------------------------------- primary statistics (PASS)
def test_primary_unit_and_threshold():
    p = _cfg()["primary"]
    assert p["statistical_unit"] == "test nuisance block"
    assert "CI_lower(Delta_primary) >= 0.15" in p["primary_pass"]
    b = _cfg()["bootstrap"]
    assert b["n_boot"] == 2000 and "test nuisance block" in b["resample_unit"]
    assert "episode-level bootstrap" in " ".join(p["not_allowed"])


# ---------------------------------------------------------------- power / v4 consistency (PASS)
def test_power_cert_uses_pm035_and_reports_learned_separately():
    v = json.loads((_DOCS / "test_geometry_power_verdict_v1.json").read_text())
    assert v["verdict"] == "POWER_SUFFICIENT_FOR_PREREG_V4"
    assert v["primary_geometry"] == [-0.035, 0.035]
    d = v["9_6_9_per_tau"]["0.03425"]
    assert d["learned_power"] >= 0.85 and d["passes"], "learned (not just struct) power must clear the bar"
    assert v["gain_threshold"] == 0.15 and v["threshold_not_lowered"]


def test_learned_power_module_uses_real_deepsets():
    src = (Path(__file__).resolve().parents[3] / "offline_v2" / "calibration_bias"
           / "learned_selector_power.py").read_text()
    assert "from deployment_calibration.models_v2 import DeepSets" in src


# ---------------------------------------------------------------- BLOCKER A: secret tie-break
@pytest.mark.xfail(reason="BLOCKER_SECRET_TIEBREAK: best-single tie-break reads secret actual_bias (tau-|eff|)",
                   strict=True)
def test_best_single_rule_has_no_secret_tiebreak():
    rule = " ".join(_cfg()["best_single"]["rule"]).lower()
    assert "eff" not in rule and "|eff|" not in rule and "tau" not in rule, \
        "best-single must not use tau-|eff| (secret actual_bias)"


def test_best_single_secret_tiebreak_is_dead_code():
    """Documents that the illegal tie-break never fires -> fix requires NO power recert."""
    import numpy as np
    from deployment_calibration.offline_v2.calibration_bias import learned_selector_power as L
    from deployment_calibration.offline_v2.calibration_bias import test_geometry_power as T
    tau = L.TAU_POINT
    ties = 0
    for rep in range(100):
        rng = np.random.default_rng(rep)
        tr, va, _ = T.make_dataset(9, 6, 9, tau, rng, (-0.035, 0.035))
        rows = [(o, float(np.mean([L.succ(s["nominal"], s["residual"], o, tau) for s in tr + va])))
                for o in L.BANK]
        mx = max(r[1] for r in rows)
        if sum(1 for r in rows if abs(r[1] - mx) < 1e-9) > 1:
            ties += 1
        assert L.best_single(tr, va, tau) == 0.0
    assert ties == 0, "step-1 is always unique -> the secret tie-break is dead code"


# ---------------------------------------------------------------- BLOCKER B: seeds not frozen
@pytest.mark.xfail(reason="BLOCKER_SEEDS_NOT_FROZEN: master/residual/order/bootstrap seeds have no frozen "
                          "value or derivation rule (only model_seeds are frozen)", strict=True)
def test_all_randomization_seeds_frozen():
    cfg = _cfg()
    # model seeds are frozen; the run-randomization seeds must ALSO have concrete values or a rule
    for name in ("master_seed", "residual_seed", "bootstrap_seed"):
        assert isinstance(cfg.get(name), int) or isinstance(cfg.get("seed_derivation"), str), \
            f"{name} must be a frozen integer or derived by a frozen rule before the run"


# ---------------------------------------------------------------- BLOCKER C: seal scheme not unique
@pytest.mark.xfail(reason="BLOCKER_SEAL_SCHEME_NOT_UNIQUE: seal step keeps an either/or", strict=True)
def test_seal_scheme_is_unique():
    seal = " ".join(_prereg()["seal_unseal"]).lower()
    # the either/or is phrased "keep test SEALED (or generate only after models frozen)"
    either_or = ("(or " in seal) or (" or generate" in seal) or ("sealed (or" in seal)
    assert not either_or, "seal/unseal must freeze exactly one scheme (no 'A or B')"


# ---------------------------------------------------------------- BLOCKER D: model-failure semantics
@pytest.mark.xfail(reason="BLOCKER_MODEL_FAILURE_SEMANTICS: model-training failure conflated with the "
                          "300-trial technical-invalid count; no all-5-seeds-must-converge rule; "
                          "seed gate >=4/5 can silently pass a missing seed", strict=True)
def test_model_training_failure_separated_from_trial_invalid():
    cfg = _cfg()
    tfh = cfg["model"]["training_failure_handling"].lower()
    assert "technical-invalid" not in tfh and "section 17" not in tfh, \
        "a model-training failure is an analysis-stage event, not one of the 300 runtime trials"
    # and the config must state all 5 seeds are required (no silent drop)
    assert cfg["model"].get("all_seeds_must_converge") is True


# ---------------------------------------------------------------- collision rule (PASS/minor)
def test_collision_is_secondary_intention_to_analyze():
    cr = _cfg()["collision_rule"]
    assert cr["numeric_threshold_N"] == 0.0
    assert "intention-to-analyze" in cr["primary"]
    assert cr["confirmatory_data_not_used_to_set_threshold"] is True


# ---------------------------------------------------------------- claim scope (PASS)
def test_claim_scope_forbidden_terms_absent_from_primary_claim():
    import re
    txt = re.sub(r"\s+", " ", (_DOCS / "confirmatory_v4_claim_scope.md").read_text().lower())
    # the doc must explicitly forbid these over-claims (whitespace-normalized for line wraps)
    for term in ("unseen hidden-state", "one-shot online", "non-destructive", "arbitrary",
                 "continuous action"):
        assert term in txt, f"claim-scope doc must explicitly disclaim '{term}'"
