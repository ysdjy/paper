"""Claude C second-pass (fix1) re-audit tests — independent, read-only.

Encodes Claude C's re-audit findings on preregistration v4 FIX1 (commit 91a41e5). PASS tests confirm
the resolved blockers (B seeds, C seal, D model-failure) and unchanged power inputs. strict-xfail tests
document the OPEN items after fix1:
  * BLOCKER_PRODUCTION_BEST_SINGLE_STILL_SECRET_DEPENDENT
  * canonical planned_identity format not fully frozen.
Do NOT modify B's fix1 files or C's first-audit file. Run in env_isaaclab (torch).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[4]
_DOCS = _ROOT / "docs" / "offline_v2" / "calibration_bias"
_SEED_ROOT = "confirmatory-v4|2bf7217907f24ae54a08db71bbdcf624b110ccf0"


def _cfg():
    return json.loads((_DOCS / "confirmatory_v4_config.json").read_text())


def _seed(label):
    return int(hashlib.sha256(f"{_SEED_ROOT}|{label}".encode()).hexdigest()[:16], 16) % (2 ** 63 - 1)


# ============================ Blocker A (rule fixed) ============================
def test_best_single_rule_is_observable_only_and_no_secret_margin():
    bs = _cfg()["best_single"]
    rule = " ".join(bs["rule"]).lower()
    assert "eff" not in rule and "tau" not in rule and "margin" not in rule
    fb = " ".join(bs["forbidden_inputs"]).lower()
    for s in ("nominal", "residual", "actual", "eff_signed", "abs_eff", "tau-|eff|", "oracle", "test"):
        assert s in fb


def test_selection_invariance_4500_recompute():
    """Independently reconfirm B's 4500-replicate no-change claim (subset per config)."""
    import numpy as np
    from deployment_calibration.offline_v2.calibration_bias import learned_selector_power as L
    tie = mism = nonzero = tot = 0
    for (tb, vb, teb) in [(9, 6, 9), (12, 9, 12), (18, 9, 18)]:
        for tau in (L.TAU_LOWER, L.TAU_POINT, L.TAU_UPPER):
            for s in range(60):
                rng = np.random.default_rng(s)
                tr, va, _ = L.make_dataset(tb, vb, teb, tau, rng)
                rows = [(o, float(np.mean([L.succ(x["nominal"], x["residual"], o, tau) for x in tr + va])))
                        for o in L.BANK]
                mx = max(r[1] for r in rows)
                if sum(1 for r in rows if abs(r[1] - mx) < 1e-9) > 1:
                    tie += 1
                if L.best_single(tr, va, tau) != L.best_single_legal(tr, va, tau):
                    mism += 1
                if L.best_single_legal(tr, va, tau) != 0.0:
                    nonzero += 1
                tot += 1
    assert (tie, mism, nonzero) == (0, 0, 0), f"ties/mismatch/nonzero = {(tie, mism, nonzero)}"


# ---- NEW BLOCKER: production best-single still secret-dependent ----
@pytest.mark.xfail(reason="BLOCKER_PRODUCTION_BEST_SINGLE_STILL_SECRET_DEPENDENT: best_single_legal "
                          "reconstructs the label via succ(nominal,residual,tau) (secret hidden state); "
                          "no observed-y.success-only production interface is frozen", strict=True)
def test_production_best_single_reads_only_observed_outcomes():
    from deployment_calibration.offline_v2.calibration_bias import learned_selector_power as L
    import inspect
    src = inspect.getsource(L.best_single_legal)
    # a legal production selector must NOT read the secret hidden state to compute success
    assert 's["nominal"]' not in src and 's["residual"]' not in src and "tau" not in src, \
        "production best-single must read observed y.success, not reconstruct from secret nominal/residual/tau"


# ============================ Blocker B (seeds frozen) — RESOLVED ============================
def test_all_15_seeds_recompute_exactly():
    frozen = _cfg()["frozen_seeds"]
    assert frozen["master_seed"] == 6341914557047805261
    assert frozen["bootstrap_seed"] == 9014517173581927929
    assert len(frozen) == 15
    for label, val in frozen.items():
        assert _seed(label) == val, f"{label} mismatch"
    ints = [v for v in frozen.values() if isinstance(v, int)]
    assert len(set(ints)) == len(ints), "seed collision"


def test_seed_derivation_rule_and_subseed_frozen_in_config():
    cfg = _cfg()
    assert cfg["seed_root"] == _SEED_ROOT
    assert "sha256" in cfg["seed_derivation"] and "2**63 - 1" in cfg["seed_derivation"]
    assert "subseed(domain_seed, planned_identity)" in cfg["subseed_derivation"]


# ---- OPEN sub-issue: canonical planned_identity format not pinned ----
@pytest.mark.xfail(reason="canonical planned_identity string format (padding + planned_episode_id "
                          "template) is only exemplified ('block=03' vs rule 'block=<i>'), not frozen",
                   strict=True)
def test_canonical_planned_identity_format_frozen():
    s = json.dumps(_cfg())
    assert "planned_identity_format" in s or "identity_template" in s, \
        "a canonical planned_identity template (exact fields, order, padding) must be frozen"


# ============================ Blocker C (seal unique) — RESOLVED ============================
def test_seal_scheme_unique_and_no_either_or():
    p = json.loads((_DOCS / "preregistration_v4.json").read_text())
    assert p["seal_scheme"] == "SCHEME_2_MODEL_FREEZE_BEFORE_TEST_GENERATION"
    seal = " ".join(p["seal_unseal"]).lower()
    assert "(or " not in seal and " or generate" not in seal
    forb = " ".join(p["post_unseal_forbidden"]).lower()
    assert "reselect test seed" in forb and "generate multiple test manifests and pick" in forb


def test_test_manifest_only_after_model_freeze_and_early_access_invalid():
    p = json.loads((_DOCS / "preregistration_v4.json").read_text())
    steps = p["seal_unseal"]
    # Step 5 freezes models/analysis; Step 6 generates the test manifest -> ordering holds
    assert any("FREEZE ANALYSIS" in s for s in steps)
    assert any("GENERATE TEST MANIFEST" in s and "after Step 5" in s for s in steps)
    s = json.dumps(p)
    assert "EXPERIMENT_INVALID_EARLY_TEST_ACCESS" in s


# ============================ Blocker D (model failure) — RESOLVED ============================
def test_model_fit_failure_separate_namespace_and_all5_required():
    cfg = _cfg()
    mf = cfg["model_fit"]
    assert mf["all_seeds_must_converge"] is True
    assert mf["no_seed_replacement"] and mf["no_dropping_bad_seed"]
    assert mf["missing_or_invalid_seed_makes_experiment_invalid"] is True
    assert mf["invalid_verdict"] == "EXPERIMENT_INVALID_MODEL_FIT"
    assert "SEPARATE" in mf["namespace"] and "runtime_trial_invalid_count" in mf["namespace"]
    ti = cfg["technical_invalid"]
    assert "model-fit failures" in ti["scope"] and "do NOT count here" in ti["scope"]


def test_retry_at_most_one_infrastructural_only():
    mf = _cfg()["model_fit"]
    assert "at most 1 deterministic retry" in mf["retry"]
    assert "infrastructural" in mf["retry"].lower()
    for c in ("same model seed", "same hyperparameters", "same code commit"):
        assert c in mf["retry_constraints"]


# ============================ minors + power-unchanged ============================
def test_probe_allowlist_fields_required_no_silent_zero_fill():
    cfg = _cfg()
    assert cfg["history_allowlist"] == [
        "theta.grasp_offset_local_y", "theta.max_pos_step", "theta.pull_lead",
        "success", "task_outcome_error", "skill_elapsed_time",
        "pull_phase_duration", "final_joint_position"]
    ti = " ".join(cfg["technical_invalid"]["invalid_conditions"]).lower()
    assert "probe allowlist field" in ti


def test_power_inputs_unchanged_by_fix1():
    d = _cfg()["design"]
    assert d["candidate_bank"] == [-0.04, 0.0, 0.04]
    assert d["test_nominals"] == [-0.035, 0.035]
    assert d["blocks"] == {"train": 9, "val": 6, "test": 9}
    v = json.loads((_DOCS / "test_geometry_power_verdict_v1.json").read_text())
    assert v["verdict"] == "POWER_SUFFICIENT_FOR_PREREG_V4"


def test_band_edge_original_verdict_untouched():
    old = json.loads((_DOCS / "band_edge_exit_verdict_v1.json").read_text())
    assert old["verdict"] == "MODIFY_RESIDUAL_OR_DESIGN"  # superseded, not overwritten
    assert (_DOCS / "band_edge_exit_verdict_supersession_v1.json").exists()
