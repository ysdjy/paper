"""Claude C one-shot full-repository audit tests (final snapshot v1) — independent, read-only.

Encodes the FROZEN issue list from the one-shot audit of the final snapshot
(HEAD b983b7e, fix4 source f8fedd45). PASS = verified-clean / resolved; strict-xfail = the frozen
open issues (FINAL-001..005). Do NOT modify B files or prior C audits. Run in env_isaaclab (torch).
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[4]
_DOCS = _ROOT / "docs" / "offline_v2" / "calibration_bias"


def _cfg():
    return json.loads((_DOCS / "confirmatory_v4_config.json").read_text())


# ==================== VERIFIED PASS (resolved fix3 blockers + core) ====================
def test_offset_domain_strict():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_selection as S
    def recs(zero):
        out = []
        for split, n in (("train", 45), ("validation", 12)):
            for i in range(n):
                for off in (-0.04, zero, 0.04):
                    out.append({"split": split, "session_id": f"{split}-{i}", "trial_role": "candidate",
                                "theta": {"grasp_offset_local_y": off}, "y": {"success": off == 0.0},
                                "planned_episode_id": f"{split}-{i}-{off!r}"})
        return out
    for bad in (0.0004, "0.0", True, float("nan")):
        with pytest.raises(S.BestSingleInputError):
            S.select_best_single_confirmatory(recs(bad))
    assert S.select_best_single_confirmatory(recs(0.0))["selected_offset"] == 0.0


def test_single_production_entry_no_bypass():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_selection as S
    assert S.__all__ == ["select_best_single_confirmatory", "ObservedCandidateOutcome", "BestSingleInputError"]
    sig = inspect.signature(S.select_best_single_confirmatory)
    assert list(sig.parameters) == ["observed_candidate_records"]  # one param, no override kwargs
    assert _cfg()["best_single"]["implementation"].endswith("select_best_single_confirmatory")


def test_deep_validator_rejects_bogus_and_accepts_valid():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI
    blk = {k: ("train" if k == "split" else (0 if k == "block_index" else "x")) for k in MI.REQUIRED_BLOCK}
    sess = {k: ("train" if k == "split" else (0 if k == "block_index" else "x")) for k in MI.REQUIRED_SESSION}
    tri = {k: "DUP" for k in MI.REQUIRED_TRIAL}
    m = {k: "x" for k in MI.REQUIRED_TOP_LEVEL}
    m["phase"] = "train_validation"
    m["counts"] = {"blocks": 15, "sessions": 57, "trials": 228}
    m["blocks"] = [dict(blk) for _ in range(15)]
    m["sessions"] = [dict(sess) for _ in range(57)]
    m["trials"] = [dict(tri) for _ in range(228)]
    with pytest.raises(MI.ManifestIntegrityError):
        MI.validate_fully_resolved_phase_manifest(m)


def test_combined_hash_all_fields_required():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI
    sig = inspect.signature(MI.combined_experiment_plan_hash)
    assert all(p.default is inspect._empty for p in sig.parameters.values())  # no optional field


def test_identity_domain_strict_and_selection_success_only():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as ID
    for args in [("train", 99, 0.0, "candidate", 0.0), ("test", 0, 0.0346, "candidate", 0.0),
                 ("test", 0, 0.035, "probe", 0.02)]:
        with pytest.raises(ValueError):
            ID.trial_identity(*args)
    plan = (_DOCS / "confirmatory_v4_analysis_plan.md").read_text().lower()
    assert "max p_success" in plan and "selected success" in plan  # selection + primary use success


def test_power_inputs_unchanged_and_claim_success_bounded():
    d = _cfg()["design"]
    assert d["candidate_bank"] == [-0.04, 0.0, 0.04] and d["test_nominals"] == [-0.035, 0.035]
    assert d["blocks"] == {"train": 9, "val": 6, "test": 9}
    assert json.loads((_DOCS / "test_geometry_power_verdict_v1.json").read_text())["verdict"] \
        == "POWER_SUFFICIENT_FOR_PREREG_V4"
    import re
    cs = re.sub(r"\s+", " ", (_DOCS / "confirmatory_v4_claim_scope.md").read_text().lower())
    assert "unseen hidden-state" in cs and "non-destructive" in cs  # over-claims disclaimed


# ==================== FROZEN OPEN ISSUES (strict-xfail) ====================
@pytest.mark.xfail(reason="FINAL-001 pre_generator_blocker: no canonical subseed->permutation order "
                          "resolution; validator accepts ANY bank permutation / unique int, not the "
                          "seed-determined order", strict=True)
def test_FINAL_001_order_resolution_frozen():
    import deployment_calibration.offline_v2.calibration_bias.confirmatory_v4_identity as ID
    import deployment_calibration.offline_v2.calibration_bias.confirmatory_v4_manifest_integrity as MI
    src = inspect.getsource(ID) + inspect.getsource(MI)
    # a frozen order resolver mapping a subseed to a deterministic permutation must exist
    assert "resolve_candidate_order" in src or "order_permutation" in src or "def resolve_order" in src


@pytest.mark.xfail(reason="FINAL-002 pre_manifest_blocker: config_sha256/schema_version/determinism_env "
                          "are format/presence-only in the validator (knowable-now values not exact-checked)",
                   strict=True)
def test_FINAL_002_frozen_value_checks():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI
    src = inspect.getsource(MI.validate_fully_resolved_phase_manifest)
    assert "canonical_config_sha256(" in src or "expected_config_sha256" in src


@pytest.mark.xfail(reason="FINAL-003 docs(nonblocking): manifest_spec §2.3 retains stale singular "
                          "manifest_hash/config_hash/code_commit field names", strict=True)
def test_FINAL_003_no_stale_manifest_field_names():
    spec = (_DOCS / "confirmatory_v4_manifest_spec.md").read_text()
    assert "manifest_hash " not in spec and "config_hash " not in spec and "code_commit " not in spec


@pytest.mark.xfail(reason="FINAL-004 science/docs(nonblocking): prereg does not explicitly state the "
                          "error/time heads are auxiliary and multidimensional-prediction superiority is "
                          "not claimed/identified (primary = success only)", strict=True)
def test_FINAL_004_multidim_disclaimer_present():
    txt = " ".join((p.read_text() for p in _DOCS.glob("confirmatory_v4_*.md"))).lower()
    assert "auxiliary" in txt and ("error/time head" in txt or "multidimensional prediction is not" in txt)


@pytest.mark.xfail(reason="FINAL-005 docs/traceability(nonblocking): snapshot_manifest.snapshot_commit is "
                          "the phase-1 metadata-payload commit (1abaa72), not the final audited HEAD "
                          "(b983b7e); field name is ambiguous and no final_head_commit is recorded",
                   strict=True)
def test_FINAL_005_snapshot_commit_names_final_head():
    sm = json.loads((_DOCS / "final_snapshot_v1" / "snapshot_manifest.json").read_text())
    assert sm.get("final_head_commit", "").startswith("b983b7e") or \
        sm.get("snapshot_commit", "").startswith("b983b7e")
