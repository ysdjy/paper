"""B-owned regression for the FINAL-001/002 completion (offline). No Isaac, no generator, no data."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import pytest

from deployment_calibration.offline_v2.calibration_bias import preregistration_v4 as P
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as ID
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI

_DOCS = Path(P._docs_dir())


def _cfg():
    return json.loads((_DOCS / "confirmatory_v4_config.json").read_text())


def test_status_completion():
    assert P.STATUS == "PREREGISTRATION_V4_BLOCK_STATE_KAT_GATE_READY_FOR_C_FINAL_REAUDIT"
    assert json.loads((_DOCS / "preregistration_v4.json").read_text())["power_recertification_required"] is False


# ==================== FINAL-001: canonical storage order ====================
def test_canonical_storage_sequences_lengths():
    for phase, (nb, ns, nt) in (("train_validation", (15, 57, 228)), ("test", (9, 18, 72))):
        assert len(ID.canonical_phase_block_identities(phase)) == nb
        assert len(ID.canonical_phase_session_identities(phase)) == ns
        assert len(ID.canonical_phase_trial_identities(phase)) == nt
        # storage order distinct from execution order
        assert ID.canonical_phase_trial_identities(phase) != ID.resolve_phase_execution_plan(phase)


def test_storage_sequences_are_frozen_order():
    # blocks: split rank -> block asc
    assert ID.canonical_phase_block_identities("train_validation")[0] == ID.block_identity("train", 0)
    assert ID.canonical_phase_block_identities("train_validation")[9] == ID.block_identity("validation", 0)
    # sessions use sorted(NOMINALS)
    ss = ID.canonical_phase_session_identities("test")
    assert ss[0] == ID.session_identity("test", 0, -0.035) and ss[1] == ID.session_identity("test", 0, 0.035)


def test_reference_manifests_valid_and_reproducible():
    for phase in ("train_validation", "test"):
        m = MI.reference_phase_manifest(phase)
        MI.validate_fully_resolved_phase_manifest(m)
        assert MI.fully_resolved_phase_manifest_hash(m) == \
            MI.fully_resolved_phase_manifest_hash(MI.reference_phase_manifest(phase))       # reproducible
        # lists already in canonical storage order
        assert tuple(b["canonical_block_identity"] for b in m["blocks"]) == \
            ID.canonical_phase_block_identities(phase)
        assert tuple(t["canonical_trial_identity"] for t in m["trials"]) == \
            ID.canonical_phase_trial_identities(phase)


def test_reference_storage_and_execution_separated():
    m = MI.reference_phase_manifest("test")
    # trials list is in STORAGE order; execution_order_index follows the EXECUTION plan (not the list index)
    plan = {tid: i for i, tid in enumerate(ID.resolve_phase_execution_plan("test"))}
    assert [t["execution_order_index"] for t in m["trials"]] == \
        [plan[t["canonical_trial_identity"]] for t in m["trials"]]
    # they genuinely differ (storage list index != execution index for at least one trial)
    assert any(i != t["execution_order_index"] for i, t in enumerate(m["trials"]))


def test_reordered_lists_are_rejected():
    m = MI.reference_phase_manifest("test")
    for what, msg in (("blocks", "blocks list not in canonical storage order"),
                      ("sessions", "sessions list not in canonical storage order"),
                      ("trials", "trials list not in canonical storage order")):
        mm = copy.deepcopy(m)
        random.Random(7).shuffle(mm[what])
        with pytest.raises(MI.ManifestIntegrityError) as ei:
            MI.validate_fully_resolved_phase_manifest(mm)
        assert msg in str(ei.value)


def test_reordered_trials_keeping_exec_index_rejected():
    m = MI.reference_phase_manifest("test")
    random.Random(9).shuffle(m["trials"])            # exec indices intact, just list reordered
    with pytest.raises(MI.ManifestIntegrityError):
        MI.validate_fully_resolved_phase_manifest(m)


def test_config_describes_seed_recomputed_and_storage_order():
    checks = " ".join(_cfg()["deep_manifest_validator"]["checks"]).lower()
    assert "resolved_candidate_order == id.resolve_candidate_order" in checks
    assert "resolved_candidate_order is a bank permutation" not in checks
    assert "canonical storage order" in checks
    assert "execution_order_index == index in id.resolve_phase_execution_plan" in checks
    dl = _cfg()["planned_identity_format"]["domain_labels"]
    assert "block_order" in dl


# ==================== FINAL-002: config description sync ====================
def test_config_describes_exact_final002_checks():
    checks = " ".join(_cfg()["deep_manifest_validator"]["checks"]).lower()
    assert "config/commit hex formats" not in checks and "device==cpu" not in checks
    assert "config_sha256 == canonical_config_sha256" in checks
    assert "schema_version == confirmatory_v4_phase_manifest_v1" in checks
    assert "deterministic_environment == frozen" in checks
    assert "frozen at generator step 0" in checks


def test_final002_code_unchanged_smoke():
    assert MI.PHASE_MANIFEST_SCHEMA_VERSION == "confirmatory_v4_phase_manifest_v1"
    assert len(MI.canonical_config_sha256()) == 64
    m = MI.reference_phase_manifest("test")
    for mut in (lambda mm: mm.__setitem__("config_sha256", "0" * 64),
                lambda mm: mm.__setitem__("schema_version", "x"),
                lambda mm: mm["deterministic_environment"].__setitem__("device", "cuda")):
        mm = copy.deepcopy(m); mut(mm)
        with pytest.raises(MI.ManifestIntegrityError):
            MI.validate_fully_resolved_phase_manifest(mm)


# ==================== unchanged design/power ====================
def test_design_power_unchanged():
    d = _cfg()["design"]
    assert d["candidate_bank"] == [-0.04, 0.0, 0.04] and d["test_nominals"] == [-0.035, 0.035]
    assert d["blocks"] == {"train": 9, "val": 6, "test": 9}
    assert json.loads((_DOCS / "test_geometry_power_verdict_v1.json").read_text())["verdict"] \
        == "POWER_SUFFICIENT_FOR_PREREG_V4"


def test_final_003_004_005_unchanged():
    fj = json.loads((_DOCS / "final_001_002_completion.json").read_text())["frozen_issue_status"]
    assert fj["FINAL-003"] == "unchanged_resolved" and fj["FINAL-004"] == "unchanged_resolved" \
        and fj["FINAL-005"] == "unchanged_resolved"
