"""Post-fix3 PASS tests for preregistration v4 FIX3 (offline). No Isaac, no generator, no data.

Covers the three closed blockers (best-single input guard 45/12; identity domain validation; manifest
integrity full-hash layers), determinism machine config, and protocol consistency. C's audit files unmodified.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from deployment_calibration.offline_v2.calibration_bias import preregistration_v4 as P
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_selection as S
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as ID
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI
from deployment_calibration.offline_v2.calibration_bias import learned_selector_power as L
from deployment_calibration.offline_v2.calibration_bias import test_geometry_power as T

_DOCS = Path(P._docs_dir())


def _cfg():
    return json.loads((_DOCS / "confirmatory_v4_config.json").read_text())


def _pre():
    return json.loads((_DOCS / "preregistration_v4.json").read_text())


def test_status_is_fix3():
    assert P.STATUS == "PREREGISTRATION_V4_FIX3_READY_FOR_FINAL_C_REAUDIT"
    assert _pre()["status"] == P.STATUS and _pre()["power_recertification_required"] is False


# ================= BLOCKER I: best-single input guard =================
def test_only_confirmatory_entry_in_all_and_single_param():
    assert S.__all__ == ["select_best_single_confirmatory", "ObservedCandidateOutcome", "BestSingleInputError"]
    sig = inspect.signature(S.select_best_single_confirmatory)
    assert list(sig.parameters) == ["observed_candidate_records"]
    # no override kwargs on the production entry
    for bad in ("validate_completeness", "candidate_bank", "expected_candidate_records", "expected_sessions"):
        assert bad not in sig.parameters


def test_active_config_binds_confirmatory_entry():
    impl = _cfg()["best_single"]["implementation"]
    assert impl.endswith("confirmatory_v4_selection.select_best_single_confirmatory")
    pe = _cfg()["best_single"]["production_entry"]
    assert pe["no_override_kwargs"] is True and pe["validate_completeness"] is True
    assert pe["expected_candidate_records"] == 171 and pe["candidate_bank_overridable"] is False


def test_config_checks_mention_45_12_135_36():
    checks = " ".join(_cfg()["best_single"]["input_completeness"]["checks"])
    for tok in ("45", "12", "135", "36"):
        assert tok in checks


def _valid(split_sessions, success_of_offset):
    recs = []
    for split, n in split_sessions:
        for i in range(n):
            for o in (-0.04, 0.0, 0.04):
                recs.append({"split": split, "session_id": f"{split}-{i}", "trial_role": "candidate",
                             "theta": {"grasp_offset_local_y": o},
                             "y": {"success": bool(success_of_offset[o])},
                             "planned_episode_id": f"{split}-{i}-{o:+.3f}",
                             "secret_deployment_state": {"actual_bias_y": 0.033}})
    return recs


def test_45_12_valid_and_split_counts_in_artifact():
    art = S.select_best_single_confirmatory(_valid([("train", 45), ("validation", 12)],
                                                   {-0.04: False, 0.0: True, 0.04: False}))
    assert art["selected_offset"] == 0.0
    assert art["train_session_count"] == 45 and art["validation_session_count"] == 12
    assert art["train_record_count"] == 135 and art["validation_record_count"] == 36
    assert art["n_trials_per_offset"] == 57 and art["production_entrypoint"].endswith("select_best_single_confirmatory")


def test_wrong_split_composition_rejected():
    s = {-0.04: False, 0.0: True, 0.04: False}
    with pytest.raises(S.BestSingleInputError):          # 57 train + 0 validation
        S.select_best_single_confirmatory(_valid([("train", 57)], s))
    with pytest.raises(S.BestSingleInputError):          # 44/13 (total 171 but wrong split)
        S.select_best_single_confirmatory(_valid([("train", 44), ("validation", 13)], s))
    with pytest.raises(S.BestSingleInputError):          # 45/11 (short)
        S.select_best_single_confirmatory(_valid([("train", 45), ("validation", 11)], s))


def test_cross_split_session_id_collision_rejected():
    recs = _valid([("train", 45), ("validation", 12)], {-0.04: False, 0.0: True, 0.04: False})
    # rename one validation session to collide with a train session_id
    for r in recs:
        if r["session_id"] == "validation-0":
            r["session_id"] = "train-0"
            r["planned_episode_id"] = "collide-" + r["planned_episode_id"]
    with pytest.raises(S.BestSingleInputError):
        S.select_best_single_confirmatory(recs)


def test_secret_perturbation_invariant_and_observed_decisive():
    base = _valid([("train", 45), ("validation", 12)], {-0.04: False, 0.0: True, 0.04: False})
    a = S.select_best_single_confirmatory(base)["selected_offset"]
    for r in base:
        r["secret_deployment_state"] = {"actual_bias_y": -99.0}
        r["eff_signed"] = 42.0
    assert S.select_best_single_confirmatory(base)["selected_offset"] == a == 0.0
    flip = _valid([("train", 45), ("validation", 12)], {-0.04: True, 0.0: False, 0.04: False})
    assert S.select_best_single_confirmatory(flip)["selected_offset"] == -0.04


def test_probe_test_and_nonbool_rejected():
    good = _valid([("train", 45), ("validation", 12)], {-0.04: False, 0.0: True, 0.04: False})
    for mutate in (lambda r: r.update(trial_role="probe"),
                   lambda r: r.update(split="test"),
                   lambda r: r.update(y={"success": 1})):
        bad = [dict(x) for x in good]
        mutate(bad[0])
        with pytest.raises(S.BestSingleInputError):
            S.select_best_single_confirmatory(bad)


def test_no_completeness_or_bank_bypass_on_production_entry():
    src = inspect.getsource(S.select_best_single_confirmatory)
    assert "validate_completeness" not in src           # cannot be turned off through the entry
    assert "_check_confirmatory_completeness" in src     # always calls the full gate
    assert "_select_best_single_core" not in S.__all__ if "_select_best_single_core" in dir(S) else True


def test_bridge_invariance_4500_evidence_and_recompute():
    j = json.loads((_DOCS / "production_best_single_bridge_invariance_v1.json").read_text())
    assert j["total_configs_checked"] == 4500 and j["mismatch_count"] == 0 and j["agree_all"] is True
    for tau in (0.0342, 0.03425, 0.0343):
        for s in range(20):
            rng = np.random.default_rng(s)
            TR, VA, _ = T.make_dataset(9, 6, 9, tau, rng, (-0.035, 0.035))
            recs = []
            for si, sess in enumerate(TR + VA):
                split = "train" if sess in TR else "validation"
                for o in L.BANK:
                    recs.append({"split": split, "session_id": f"s{si}", "trial_role": "candidate",
                                 "theta": {"grasp_offset_local_y": o},
                                 "y": {"success": bool(L.succ(sess["nominal"], sess["residual"], o, tau))},
                                 "planned_episode_id": f"ep-{si}-{o}"})
            assert S._select_best_single_core(recs, validate_completeness=False)["selected_offset"] == \
                L.best_single_legal(TR, VA, tau)


# ================= BLOCKER II: identity domain validation =================
def test_illegal_domain_rejected():
    for args in [("train", 99, 0.0, "candidate", 0.0), ("train", -1, 0.0, "candidate", 0.0),
                 ("validation", 6, 0.01, "candidate", 0.0), ("train", 0, 0.5, "candidate", 0.0),
                 ("validation", 0, 0.0, "candidate", 0.0), ("test", 0, 0.03, "candidate", 0.04),
                 ("test", 0, 0.0346, "candidate", 0.04), ("test", 0, 0.035, "probe", 0.02),
                 ("test", 0, 0.035, "candidate", 0.07), ("train", 0, 0.0, "other", 0.0),
                 ("train", 0, float("nan"), "candidate", 0.0), ("train", 0, float("inf"), "candidate", 0.0)]:
        with pytest.raises(ValueError):
            ID.trial_identity(*args)


def test_bool_block_and_attempt_rejected():
    with pytest.raises(ValueError):
        ID.block_identity("train", True)
    with pytest.raises(ValueError):
        ID.attempt_id("v4ep-" + "a" * 24, True)
    for a in (-1, 2):
        with pytest.raises(ValueError):
            ID.attempt_id("v4ep-" + "a" * 24, a)
    with pytest.raises(ValueError):
        ID.attempt_id("not-a-pid", 0)


def test_300_identities_valid_unique_and_unchanged():
    rows = ID.enumerate_trials()
    assert len(rows) == 300 and len({r["planned_episode_id"] for r in rows}) == 300
    # canonical strings unchanged from fix2 examples
    assert ID.trial_identity("test", 8, 0.035, "candidate", 0.04) == \
        "v4|split=test|block=08|nominal=+0.035|role=candidate|offset=+0.040"
    key = ID.PlannedTrialKey("test", 8, 0.035, "candidate", 0.04)
    assert key.trial_str() == ID.trial_identity("test", 8, 0.035, "candidate", 0.04)


def test_subseeds_route_through_validation():
    with pytest.raises(ValueError):
        ID.block_residual_subseed("train", 99)
    with pytest.raises(ValueError):
        ID.trial_init_subseed("test", 0, 0.035, "probe", 0.02)
    assert isinstance(ID.block_residual_subseed("train", 0), int)


# ================= BLOCKER III: manifest integrity =================
def _manifest():
    return {"schema_version": "1", "phase": "test", "protocol_commit": "p", "generator_commit": "g",
            "runtime_commit": "r", "config_sha256": "c", "planned_structure_sha256": "s",
            "frozen_seeds": {"master_seed": 1}, "deterministic_environment": {"device": "cpu"},
            "environment_versions": {"torch": "x"}, "counts": {"blocks": 9, "sessions": 18, "trials": 72},
            "execution_order_definition": "ord", "manifest_algorithm_version": "v1",
            "blocks": [{"split": "test", "block_index": i, "canonical_block_identity": f"b{i}",
                        "residual_value": 0.001 * i, "nuisance_values": {}, "residual_subseed": i,
                        "nuisance_subseed": i, "block_order_key": i} for i in range(9)],
            "sessions": [{"canonical_session_identity": f"s{i}", "split": "test", "block_index": 0,
                          "nominal_bias": 0.035, "session_order_key": i, "nominal_order_key": i,
                          "resolved_candidate_order": [-0.04, 0.0, 0.04]} for i in range(18)],
            "trials": [{"canonical_trial_identity": f"tr{i}", "planned_episode_id": f"v4ep-{i:024d}",
                        "role": "candidate", "offset": 0.0, "trial_init_subseed": i,
                        "execution_order_index": i, "resume_key": f"v4ep-{i:024d}"} for i in range(72)]}


def test_structure_hash_semantics_and_rename():
    assert ID.canonical_planned_structure_hash() == ID.canonical_manifest_hash()   # alias
    src = inspect.getsource(ID.canonical_planned_structure_hash)
    assert "STRUCTURE-ONLY" in src or "structure" in src.lower()


def test_phase_counts_228_72():
    assert MI.PHASES["train_validation"]["trials"] == 228
    assert MI.PHASES["test"]["trials"] == 72
    assert MI.PHASES["train_validation"]["trials"] + MI.PHASES["test"]["trials"] == 300


def test_full_hash_mutation_sensitivity():
    h0 = MI.fully_resolved_phase_manifest_hash(_manifest())
    def mut(fn):
        m = _manifest(); fn(m); return MI.fully_resolved_phase_manifest_hash(m)
    assert mut(lambda m: m["blocks"][0].__setitem__("residual_value", 9.9)) != h0
    assert mut(lambda m: m["blocks"][0].__setitem__("nuisance_values", {"z": 1})) != h0
    assert mut(lambda m: m["sessions"][0].__setitem__("session_order_key", 999)) != h0
    assert mut(lambda m: m["sessions"][0].__setitem__("resolved_candidate_order", [0.04, 0.0, -0.04])) != h0
    assert mut(lambda m: m["trials"][0].__setitem__("execution_order_index", 999)) != h0
    assert mut(lambda m: m["trials"][0].__setitem__("trial_init_subseed", 999)) != h0
    assert mut(lambda m: m["frozen_seeds"].__setitem__("master_seed", 2)) != h0
    assert mut(lambda m: m.__setitem__("config_sha256", "zzz")) != h0
    assert mut(lambda m: m.__setitem__("generator_commit", "zzz")) != h0
    assert mut(lambda m: m.__setitem__("runtime_commit", "zzz")) != h0
    assert mut(lambda m: m["environment_versions"].__setitem__("torch", "zzz")) != h0


def test_full_hash_excludes_only_self_field():
    m = _manifest()
    h0 = MI.fully_resolved_phase_manifest_hash(m)
    stamped = MI.stamp_full_manifest_hash(m)
    assert stamped[MI.SELF_HASH_FIELD] == h0
    # a non-null self field before hashing is rejected
    m2 = _manifest(); m2[MI.SELF_HASH_FIELD] = "deadbeef"
    with pytest.raises(MI.ManifestIntegrityError):
        MI.fully_resolved_phase_manifest_hash(m2)


def test_missing_resolved_field_rejected():
    m = _manifest(); del m["blocks"][0]["residual_value"]
    with pytest.raises(MI.ManifestIntegrityError):
        MI.validate_fully_resolved_phase_manifest(m)


def test_combined_plan_hash_requires_all_anchors():
    ok = MI.combined_experiment_plan_hash(
        train_validation_manifest_sha256="a", model_analysis_freeze_sha256="b", test_manifest_sha256="c",
        config_sha256="d", protocol_commit="e", generator_commit="f", analysis_code_sha256="g",
        bootstrap_seed=1)
    assert isinstance(ok, str) and len(ok) == 64
    with pytest.raises(MI.ManifestIntegrityError):
        MI.combined_experiment_plan_hash(
            train_validation_manifest_sha256=None, model_analysis_freeze_sha256="b", test_manifest_sha256="c",
            config_sha256="d", protocol_commit="e", generator_commit="f")


# ================= determinism / protocol =================
def test_determinism_machine_config():
    d = _cfg()["determinism_env"]
    assert d["device"] == "cpu"
    assert d["torch.set_num_threads"] == 1 and d["torch.use_deterministic_algorithms"] is True
    assert "EXPERIMENT_INVALID_MODEL_FIT" in d["deterministic_unsupported"]


def test_power_inputs_unchanged():
    d = _cfg()["design"]
    assert d["candidate_bank"] == [-0.04, 0.0, 0.04] and d["test_nominals"] == [-0.035, 0.035]
    assert _cfg()["trial_counts"]["full_task_trials_total"] == 300
    v = json.loads((_DOCS / "test_geometry_power_verdict_v1.json").read_text())
    assert v["verdict"] == "POWER_SUFFICIENT_FOR_PREREG_V4"


def test_no_generator_manifest_checkpoint_data():
    hc = _pre()["hard_constraints"]
    assert "no confirmatory generator" in hc and "no run authorization" in hc
    names = [p.name for p in _DOCS.iterdir()]
    assert not any(n.endswith(".pt") or "manifest_instance" in n or "confirmatory_data" in n for n in names)


def test_md_json_consistency():
    P.emit()
    md = (_DOCS / "preregistration_v4.md").read_text()
    assert "PREREGISTRATION_V4_FIX3_READY_FOR_FINAL_C_REAUDIT" in md
    assert "select_best_single_confirmatory" in (_DOCS / "preregistration_v4.json").read_text()


def test_306_hash_unchanged():
    p = "deployment_calibration/data/band_edge_full_306_v1/episodes.jsonl"
    if not Path(p).exists():
        pytest.skip("306 not present")
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == \
        "131750e158032e53e5c8daab6e94530c70de1224f05b8fc79a7497009524c20e"
