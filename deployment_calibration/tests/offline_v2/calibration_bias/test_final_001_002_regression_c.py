"""Claude C FINAL-001/FINAL-002 final regression — independent, read-only.

Regresses ONLY FINAL-001 and FINAL-002 after B's completion patch. All tests PASS (both issues resolved).
Covers the 10 required checks: list-shuffle rejection (blocks/sessions/trials, incl. trials-with-complete-
exec-index), reproducible reference JSON/hash, storage!=execution separation, no stale candidate-order /
FINAL-002 config descriptions, exact-value checks still valid, and design/power invariants unchanged.
Do NOT modify B files or prior audits. Run in env_isaaclab (torch).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[4]
_DOCS = _ROOT / "docs" / "offline_v2" / "calibration_bias"


def _cfg():
    return json.loads((_DOCS / "confirmatory_v4_config.json").read_text())


def _MI():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI
    return MI


def _ID():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as ID
    return ID


# ============================ FINAL-001 ============================
def test_01_canonical_storage_identities_sizes():
    ID = _ID()
    for phase, (nb, ns, nt) in (("train_validation", (15, 57, 228)), ("test", (9, 18, 72))):
        b = ID.canonical_phase_block_identities(phase)
        s = ID.canonical_phase_session_identities(phase)
        t = ID.canonical_phase_trial_identities(phase)
        assert (len(b), len(s), len(t)) == (nb, ns, nt)
        assert len(set(b)) == nb and len(set(s)) == ns and len(set(t)) == nt


def test_02_shuffle_blocks_rejected():
    MI = _MI()
    m = MI.reference_phase_manifest("train_validation")
    MI.validate_fully_resolved_phase_manifest(m)
    m2 = copy.deepcopy(m); random.Random(1).shuffle(m2["blocks"])
    with pytest.raises(MI.ManifestIntegrityError, match="blocks list not in canonical storage order"):
        MI.validate_fully_resolved_phase_manifest(m2)


def test_03_shuffle_sessions_rejected():
    MI = _MI()
    m = MI.reference_phase_manifest("train_validation")
    m2 = copy.deepcopy(m); random.Random(2).shuffle(m2["sessions"])
    with pytest.raises(MI.ManifestIntegrityError, match="sessions list not in canonical storage order"):
        MI.validate_fully_resolved_phase_manifest(m2)


def test_04_shuffle_trials_rejected_even_with_complete_exec_index():
    MI = _MI()
    m = MI.reference_phase_manifest("train_validation")
    m2 = copy.deepcopy(m); random.Random(3).shuffle(m2["trials"])
    # execution_order_index travels with the rows -> still a complete 0..227 set, yet rejected on storage order
    assert sorted(t["execution_order_index"] for t in m2["trials"]) == list(range(228))
    with pytest.raises(MI.ManifestIntegrityError, match="trials list not in canonical storage order"):
        MI.validate_fully_resolved_phase_manifest(m2)


def test_05_reference_reproducible_json_and_hash():
    MI = _MI()
    a = MI.reference_phase_manifest("train_validation")
    b = MI.reference_phase_manifest("train_validation")
    assert MI._canonical(a) == MI._canonical(b)
    assert MI.fully_resolved_phase_manifest_hash(a) == MI.fully_resolved_phase_manifest_hash(b)


def test_06_storage_order_distinct_from_execution_order():
    MI, ID = _MI(), _ID()
    m = MI.reference_phase_manifest("train_validation")
    storage = tuple(t["canonical_trial_identity"] for t in m["trials"])
    execu = tuple(t["canonical_trial_identity"] for t in sorted(m["trials"], key=lambda t: t["execution_order_index"]))
    assert storage == ID.canonical_phase_trial_identities("train_validation")
    assert execu == ID.resolve_phase_execution_plan("train_validation")
    assert storage != execu  # the two orders genuinely differ


def test_07_config_describes_seed_order_and_storage_not_permutation():
    d = _cfg()
    checks = " ".join(d["deep_manifest_validator"]["checks"]).lower()
    assert "resolved_candidate_order == id.resolve_candidate_order" in checks
    assert "resolved_candidate_order is a bank permutation" not in checks
    assert "resolve_phase_execution_plan" in checks
    assert "canonical storage order" in checks
    # explicit storage != execution distinction, and block_order in domain labels
    assert "storage_order_vs_execution_order" in json.dumps(d).lower()
    assert "block_order" in json.dumps(d.get("identity_template", {})).lower() or \
        json.dumps(d).lower().count("block_order") > 0


# ============================ FINAL-002 ============================
def test_08_config_description_reflects_exact_checks_not_stale():
    checks = " ".join(_cfg()["deep_manifest_validator"]["checks"]).lower()
    assert "config/commit hex formats" not in checks and "device==cpu" not in checks
    assert "canonical_config_sha256" in checks
    assert "confirmatory_v4_phase_manifest_v1" in checks
    assert "7-key" in checks and "no gpu" in checks


def test_09_exact_value_checks_still_valid():
    MI = _MI()
    p = os.path.join(MI._repo_root(), "docs", "offline_v2", "calibration_bias", "confirmatory_v4_config.json")
    assert MI.canonical_config_sha256() == hashlib.sha256(open(p, "rb").read()).hexdigest()
    m = MI.reference_phase_manifest("test")

    def rejects(mut):
        mm = copy.deepcopy(m); mut(mm)
        with pytest.raises(MI.ManifestIntegrityError):
            MI.validate_fully_resolved_phase_manifest(mm)
    rejects(lambda mm: mm.__setitem__("config_sha256", "0" * 64))
    rejects(lambda mm: mm.__setitem__("schema_version", "x"))
    rejects(lambda mm: mm["deterministic_environment"].__setitem__("torch_set_num_threads", True))
    rejects(lambda mm: mm["deterministic_environment"].__setitem__("device", "cuda"))
    rejects(lambda mm: mm["deterministic_environment"].__setitem__("extra", 1))


# ============================ invariants (FINAL-003/004/005 + design/power) ============================
def test_10_design_power_and_final_003_004_005_invariant():
    d = _cfg()["design"]
    assert d["candidate_bank"] == [-0.04, 0.0, 0.04] and d["test_nominals"] == [-0.035, 0.035]
    assert d["blocks"] == {"train": 9, "val": 6, "test": 9}
    assert json.loads((_DOCS / "test_geometry_power_verdict_v1.json").read_text())["verdict"] \
        == "POWER_SUFFICIENT_FOR_PREREG_V4"
    # FINAL-004 disclaimer still present; FINAL-005 snapshot commit fields still exact
    import re
    cs = re.sub(r"\s+", " ", (_DOCS / "confirmatory_v4_claim_scope.md").read_text().lower())
    assert "auxiliary multitask" in cs and "success only" in cs
    sm = json.loads((_DOCS / "final_snapshot_v1" / "snapshot_manifest.json").read_text())
    assert sm["final_head_commit"] == "b983b7e9e638e2ed03589e2bd061139b073ce53f"
