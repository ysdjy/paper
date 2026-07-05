"""Claude C frozen-issue regression FINAL-001..005 — independent, read-only.

Regresses ONLY the five frozen issues (no scope expansion, no new IDs). PASS = the resolved parts;
strict-xfail = the parts still INCOMPLETE after the batch fix:
  * FINAL-001 INCOMPLETE: reordering the blocks/sessions/trials LISTS still validates but changes the full
    manifest hash (no canonical storage order / hash normalization) -> manifest not uniquely determined
    (§3.5); the config validator-checks description still says 'resolved_candidate_order is a bank
    permutation' (§3.2), not the seed-recomputed exact order.
  * FINAL-002 INCOMPLETE: the active config validator-checks description (line ~631) still states
    'config/commit hex formats; ...; device==cpu' (§4.5), omitting the now-implemented exact config-hash /
    schema-version / full-determinism-contract checks (code is correct; description is stale).
Do NOT modify B files or prior audits. Run in env_isaaclab (torch).
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[4]
_DOCS = _ROOT / "docs" / "offline_v2" / "calibration_bias"


def _cfg():
    return json.loads((_DOCS / "confirmatory_v4_config.json").read_text())


# ============================ FINAL-001 (resolved parts) ============================
def test_final001_resolvers_deterministic_and_plan_sizes():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as ID
    tv = ID.resolve_phase_execution_plan("train_validation")
    te = ID.resolve_phase_execution_plan("test")
    assert len(tv) == 228 and len(set(tv)) == 228
    assert len(te) == 72 and len(set(te)) == 72
    assert tv == ID.resolve_phase_execution_plan("train_validation")  # byte-identical re-run
    # permutation-independent resolve_order
    fwd = ID.resolve_block_order("train")
    rev = ID.resolve_order(list(reversed(list(ID.BLOCKS["train"]))),
                           key_fn=lambda bi: ID.block_order_key("train", bi), canonical_index_fn=lambda bi: bi)
    assert fwd == rev


def test_final001_validator_recomputes_order_from_seed():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI
    m = MI.reference_phase_manifest("test")
    MI.validate_fully_resolved_phase_manifest(m)  # valid reference passes

    def rejects(mut):
        mm = copy.deepcopy(m); mut(mm)
        with pytest.raises(MI.ManifestIntegrityError):
            MI.validate_fully_resolved_phase_manifest(mm)
    rejects(lambda mm: mm["blocks"][0].__setitem__("block_order_key", mm["blocks"][0]["block_order_key"] + 1))
    rejects(lambda mm: mm["sessions"][0].__setitem__(
        "resolved_candidate_order", list(reversed(mm["sessions"][0]["resolved_candidate_order"]))))

    def swap(mm):
        a, b = mm["trials"][0], mm["trials"][1]
        a["execution_order_index"], b["execution_order_index"] = b["execution_order_index"], a["execution_order_index"]
    rejects(swap)


# ---- FINAL-001 INCOMPLETE (§3.5 + §3.2) ----
@pytest.mark.xfail(reason="FINAL-001 INCOMPLETE (§3.5): reordering the blocks/sessions/trials lists still "
                          "validates but changes the full manifest hash -> manifest not uniquely "
                          "determined; no canonical storage order / hash normalization", strict=True)
def test_final001_manifest_uniquely_determined_by_content():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI
    m = MI.reference_phase_manifest("test")
    h0 = MI.fully_resolved_phase_manifest_hash(m)
    m2 = copy.deepcopy(m)
    random.Random(1).shuffle(m2["blocks"]); random.Random(2).shuffle(m2["sessions"]); random.Random(3).shuffle(m2["trials"])
    try:
        MI.validate_fully_resolved_phase_manifest(m2)
        validated = True
    except MI.ManifestIntegrityError:
        validated = False
    h2 = MI.fully_resolved_phase_manifest_hash(m2) if validated else None
    # PASS condition (desired): reordered lists are rejected, OR hash normalizes to the same value.
    assert (not validated) or (h2 == h0)


@pytest.mark.xfail(reason="FINAL-001 §3.2: active config validator-checks description still says "
                          "'resolved_candidate_order is a bank permutation' (not the seed-recomputed "
                          "exact order)", strict=True)
def test_final001_config_describes_seed_recomputed_order():
    checks = " ".join(_cfg()["deep_manifest_validator"]["checks"]).lower()
    assert "resolved_candidate_order == id.resolve_candidate_order" in checks or \
        "seed-recomputed" in checks or "seed-determined candidate order" in checks
    assert "resolved_candidate_order is a bank permutation" not in checks


# ============================ FINAL-002 ============================
def test_final002_exact_value_checks_enforced():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI
    import hashlib, os
    p = os.path.join(MI._repo_root(), "docs", "offline_v2", "calibration_bias", "confirmatory_v4_config.json")
    assert MI.canonical_config_sha256() == hashlib.sha256(open(p, "rb").read()).hexdigest()
    assert MI.PHASE_MANIFEST_SCHEMA_VERSION == "confirmatory_v4_phase_manifest_v1"
    m = MI.reference_phase_manifest("test")

    def rejects(mut):
        mm = copy.deepcopy(m); mut(mm)
        with pytest.raises(MI.ManifestIntegrityError):
            MI.validate_fully_resolved_phase_manifest(mm)
    rejects(lambda mm: mm.__setitem__("config_sha256", "0" * 64))
    rejects(lambda mm: mm.__setitem__("schema_version", "x"))
    rejects(lambda mm: mm["deterministic_environment"].__setitem__("torch_set_num_threads", True))  # True-for-1
    rejects(lambda mm: mm["deterministic_environment"].__setitem__("device", "cuda"))
    rejects(lambda mm: mm["deterministic_environment"].__setitem__("extra", 1))


# ---- FINAL-002 INCOMPLETE (§4.5) ----
@pytest.mark.xfail(reason="FINAL-002 INCOMPLETE (§4.5): active config validator-checks description still "
                          "states 'config/commit hex formats; ...; device==cpu', omitting the implemented "
                          "exact config-hash / schema-version / full-determinism-contract checks", strict=True)
def test_final002_config_description_reflects_exact_checks():
    checks = " ".join(_cfg()["deep_manifest_validator"]["checks"]).lower()
    assert "config/commit hex formats" not in checks and "device==cpu" not in checks
    assert ("config_sha256 == canonical_config_sha256" in checks or "exact config" in checks)


# ============================ FINAL-003 (RESOLVED) ============================
def test_final003_no_stale_manifest_field_entries():
    spec = (_DOCS / "confirmatory_v4_manifest_spec.md").read_text()
    import re
    # no singular field-list entries (allow meta-references in headings that mention 'no singular manifest_hash')
    for line in spec.splitlines():
        if re.match(r"^\s*[`\-|]*\s*(manifest_hash|config_hash|code_commit)\b", line) and "no singular" not in line \
                and "FINAL-003" not in line:
            pytest.fail(f"stale field entry: {line}")
    assert "runtime_commit" in spec and "planned_structure_sha256" in spec


# ============================ FINAL-004 (RESOLVED) ============================
def test_final004_multitask_auxiliary_disclaimer():
    import re
    cs = re.sub(r"\s+", " ", (_DOCS / "confirmatory_v4_claim_scope.md").read_text().lower())
    assert "auxiliary multitask" in cs           # error/time heads auxiliary
    assert "success only" in cs                  # selection + primary use success only
    assert "multidimensional" in cs              # explicit multidimensional-prediction disclaimer
    assert "outperforms a success-only" in cs    # no claim of superiority over success-only


# ============================ FINAL-005 (RESOLVED) ============================
def test_final005_snapshot_commit_disambiguated():
    sm = json.loads((_DOCS / "final_snapshot_v1" / "snapshot_manifest.json").read_text())
    assert "snapshot_commit" not in sm or sm.get("snapshot_commit") is None
    assert sm["metadata_payload_commit"] == "1abaa72d43cb7c6ff6251c5b9ca2db16856cbea6"
    assert sm["final_head_commit"] == "b983b7e9e638e2ed03589e2bd061139b073ce53f"
    assert sm["audited_snapshot_commit"] == "b983b7e9e638e2ed03589e2bd061139b073ce53f"


# ============================ design/power invariants unchanged ============================
def test_design_and_power_invariants_unchanged():
    d = _cfg()["design"]
    assert d["candidate_bank"] == [-0.04, 0.0, 0.04] and d["test_nominals"] == [-0.035, 0.035]
    assert d["blocks"] == {"train": 9, "val": 6, "test": 9}
    assert json.loads((_DOCS / "test_geometry_power_verdict_v1.json").read_text())["verdict"] \
        == "POWER_SUFFICIENT_FOR_PREREG_V4"
