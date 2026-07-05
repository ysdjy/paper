"""Claude C fix3 final re-audit tests — independent, read-only.

Confirms the resolved parts (single production entry + 45/12; identity domain strict; hash layering) and
documents the remaining GO-blocking gaps found by executing the fix3 modules:
  * best-single OFFSET domain not strict (round(.,3) smuggles +0.0004 / '0.0' / -0.0396 / +0.0404)   (3.3)
  * full-manifest validator is shape-only: a 228-duplicate / wrong-composition / bad-PID manifest passes (5.4/4.1)
  * combined_experiment_plan_hash leaves analysis_code_sha256 / bootstrap_seed optional               (5.6)
  * active manifest_spec retains stale semantics (block=<i>, science==frozen manifest_hash, manifest_hash) (5.1/7)
Do NOT modify B files or prior C audits. Run in env_isaaclab (torch).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[4]
_DOCS = _ROOT / "docs" / "offline_v2" / "calibration_bias"


def _cfg():
    return json.loads((_DOCS / "confirmatory_v4_config.json").read_text())


# ============ Blocker I core RESOLVED: single entry, 45/12, invariance ============
def test_single_production_entry_and_all():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_selection as S
    assert S.__all__ == ["select_best_single_confirmatory", "ObservedCandidateOutcome", "BestSingleInputError"]
    assert _cfg()["best_single"]["implementation"].endswith("select_best_single_confirmatory")


def _valid_4512(secret=None, m04=-0.04, zero=0.0, p04=0.04):
    recs = []
    for split, n in (("train", 45), ("validation", 12)):
        for i in range(n):
            for off in (m04, zero, p04):
                r = {"split": split, "session_id": f"{split}-{i}", "trial_role": "candidate",
                     "theta": {"grasp_offset_local_y": off}, "y": {"success": bool(off == 0.0)},
                     "planned_episode_id": f"{split}-{i}-{off!r}"}
                if secret:
                    r["secret_deployment_state"] = {"actual_bias_y": secret}
                recs.append(r)
    return recs


def test_45_12_composition_enforced_and_invariant():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_selection as S
    import copy, random
    a = S.select_best_single_confirmatory(_valid_4512(secret=0.03))
    assert a["train_session_count"] == 45 and a["validation_session_count"] == 12
    assert a["train_record_count"] == 135 and a["validation_record_count"] == 36
    b = _valid_4512(secret=-99.0)
    random.Random(3).shuffle(b)
    art = S.select_best_single_confirmatory(b)
    assert art["selected_offset"] == a["selected_offset"] == 0.0
    assert art["selection_artifact_hash"] == a["selection_artifact_hash"]  # order+secret invariant
    # wrong composition rejected
    bad = [r for r in _valid_4512() if r["split"] == "train"]  # 45 train, 0 val
    bad += bad[:36]  # pad to 171 records but 0 validation
    with pytest.raises(S.BestSingleInputError):
        S.select_best_single_confirmatory(bad[:171])


# ---- GAP 3.3: offset domain not strict ----
@pytest.mark.xfail(reason="BLOCKER_BEST_SINGLE_OFFSET_DOMAIN_NOT_STRICT: _round_offset=round(.,3) smuggles "
                          "+0.0004->0.0, '0.0'->0.0, -0.0396->-0.04, +0.0404->0.04 into the bank",
                   strict=True)
def test_offset_domain_is_strict():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_selection as S
    for m04, z, p04, desc in [(-0.04, 0.0004, 0.04, "+0.0004"), (-0.04, "0.0", 0.04, "num-string"),
                              (-0.0396, 0.0, 0.0404, "near-bank")]:
        with pytest.raises(S.BestSingleInputError):
            S.select_best_single_confirmatory(_valid_4512(m04=m04, zero=z, p04=p04))


# ============ Blocker II RESOLVED: identity domain strict ============
def test_identity_domain_validation_rejects_out_of_domain():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as ID
    for args in [("train", 99, 0.0, "candidate", 0.0), ("validation", 6, 0.0, "candidate", 0.0),
                 ("train", 0, 0.5, "candidate", 0.0), ("test", 0, 0.0346, "candidate", 0.0),
                 ("test", 0, 0.035, "probe", 0.02), ("train", 0, 0.0, "candidate", 0.07),
                 ("train", -1, 0.0, "candidate", 0.0)]:
        with pytest.raises(ValueError):
            ID.trial_identity(*args)
    # 300 canonical + unique unchanged
    rows = ID.enumerate_trials()
    assert len(rows) == 300 and len({r["planned_episode_id"] for r in rows}) == 300


# ============ Blocker III: layering OK; validator/combined/stale gaps ============
def test_structure_hash_renamed_and_layers_defined():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI
    assert set(("planned_structure_sha256", "train_validation_manifest_sha256", "test_manifest_sha256",
                "model_analysis_freeze_sha256", "combined_experiment_plan_sha256")).issubset(MI.HASH_LAYERS)
    assert MI.PHASES["train_validation"]["trials"] == 228 and MI.PHASES["test"]["trials"] == 72


def test_full_hash_covers_resolved_fields():
    """Changing a resolved scientific field changes the full hash (coverage OK)."""
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI
    m = _bogus_manifest()
    h1 = MI.fully_resolved_phase_manifest_hash(m)
    m2 = _bogus_manifest(); m2["blocks"][0]["residual_value"] = 0.00777
    assert MI.fully_resolved_phase_manifest_hash(m2) != h1


# ---- GAP 5.4 + 4.1: full validator is shape-only ----
@pytest.mark.xfail(reason="BLOCKER_FULL_MANIFEST_VALIDATOR_SHALLOW: validate_fully_resolved_phase_manifest "
                          "checks only key-presence + list length; a 228-duplicate / all-train / "
                          "PID!=identity manifest passes and yields a 'valid' full hash (no uniqueness, "
                          "composition, identity<->PID recompute, referential, subseed checks)", strict=True)
def test_full_validator_rejects_duplicate_and_inconsistent_manifest():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI
    with pytest.raises(MI.ManifestIntegrityError):
        MI.validate_fully_resolved_phase_manifest(_bogus_manifest())


# ---- GAP 5.6: combined hash optional required fields ----
@pytest.mark.xfail(reason="BLOCKER_COMBINED_HASH_OPTIONAL_FIELDS: analysis_code_sha256 and bootstrap_seed "
                          "default to None and are not required", strict=True)
def test_combined_hash_requires_analysis_and_bootstrap():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI
    with pytest.raises(MI.ManifestIntegrityError):
        MI.combined_experiment_plan_hash(train_validation_manifest_sha256="a", model_analysis_freeze_sha256="b",
                                         test_manifest_sha256="c", config_sha256="d", protocol_commit="e",
                                         generator_commit="f")  # no analysis_code_sha256 / bootstrap_seed


# ---- GAP 5.1/7: stale active manifest-spec semantics ----
@pytest.mark.xfail(reason="BLOCKER_ACTIVE_MANIFEST_SPEC_STALE: manifest_spec retains 'block=<i>' subseed "
                          "form, 'science_manifest_sha256 == frozen manifest_hash', and manifest_fields "
                          "still lists singular 'manifest_hash'", strict=True)
def test_active_manifest_spec_has_no_stale_semantics():
    spec = (_DOCS / "confirmatory_v4_manifest_spec.md").read_text()
    assert 'block=<i>' not in spec
    assert "== frozen `manifest_hash`" not in spec
    assert '"manifest_hash"' not in json.dumps(_cfg())


# ============ determinism / power unchanged (PASS) ============
def test_determinism_machine_frozen():
    s = json.dumps(_cfg())
    for f in ("set_num_threads", "set_num_interop_threads", "use_deterministic_algorithms",
              "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert f in s


def test_power_inputs_unchanged():
    d = _cfg()["design"]
    assert d["candidate_bank"] == [-0.04, 0.0, 0.04] and d["test_nominals"] == [-0.035, 0.035]
    assert d["blocks"] == {"train": 9, "val": 6, "test": 9}
    v = json.loads((_DOCS / "test_geometry_power_verdict_v1.json").read_text())
    assert v["verdict"] == "POWER_SUFFICIENT_FOR_PREREG_V4"


# ---- shared bogus-manifest builder ----
def _bogus_manifest():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI
    blk = {k: ("train" if k == "split" else (0 if k == "block_index" else "x")) for k in MI.REQUIRED_BLOCK}
    sess = {k: ("train" if k == "split" else (0 if k == "block_index" else "x")) for k in MI.REQUIRED_SESSION}
    tri = {k: "DUP" for k in MI.REQUIRED_TRIAL}
    tri["canonical_trial_identity"] = "v4|split=train|block=00|nominal=+0.000|role=candidate|offset=+0.000"
    tri["planned_episode_id"] = "v4ep-000000000000000000000000"  # inconsistent with identity
    m = {k: "x" for k in MI.REQUIRED_TOP_LEVEL}
    m["phase"] = "train_validation"
    m["counts"] = {"blocks": 15, "sessions": 57, "trials": 228}
    m["blocks"] = [dict(blk) for _ in range(15)]      # all train, all identical
    m["sessions"] = [dict(sess) for _ in range(57)]
    m["trials"] = [dict(tri) for _ in range(228)]     # all duplicate
    return m
