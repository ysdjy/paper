"""Claude C validator-description final regression — independent, read-only.

Narrow-scope regression of the ONLY remaining root cause: the duplicate/stale validator-description block.
All tests PASS -> FINAL-001 §2.5 and FINAL-002 §4.5 active-description criteria are met (a single
authoritative block; no stale phrasing in ANY active validator description; correct semantics present).
Scans BOTH active machine files (confirmatory_v4_config.json and preregistration_v4.json) so a second
stale block anywhere is caught. Do NOT modify B files or prior audits.
"""

from __future__ import annotations

import json
from pathlib import Path

_DOCS = Path(__file__).resolve().parents[4] / "docs" / "offline_v2" / "calibration_bias"
_ACTIVE = ("confirmatory_v4_config.json", "preregistration_v4.json")

_STALE = (
    "resolved_candidate_order is a bank permutation",
    "execution_order_index non-bool int, unique, complete 0..n-1",
    "config/commit hex formats",
    "device==cpu",
)
_REQUIRED_SEMANTICS = (
    "candidate_order_keys == id.candidate_order_key_map",
    "resolved_candidate_order == id.resolve_candidate_order",
    "execution_order_index == index in id.resolve_phase_execution_plan",
    "canonical storage order",
    "config_sha256 == canonical_config_sha256",
    "confirmatory_v4_phase_manifest_v1",
    "7-key",
    "no gpu",
    "frozen at generator step 0",
)


def _load(fn):
    return json.loads((_DOCS / fn).read_text())


def _all_manifest_validator_check_lists(d):
    """Every FULL manifest-validator checks list (>=8 items). The best_single.input_completeness list is a
    different validator and is excluded by its path."""
    out = []

    def walk(o, path=""):
        if isinstance(o, dict):
            if isinstance(o.get("checks"), list) and len(o["checks"]) >= 8 and "input_completeness" not in path:
                out.append((path, o["checks"]))
            for k, v in o.items():
                walk(v, path + "." + k)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, path + f"[{i}]")

    walk(d)
    return out


# ---- §2.1 exactly one authoritative manifest-validator checks list ----
def test_21_exactly_one_authoritative_validator_checks_list():
    for fn in _ACTIVE:
        d = _load(fn)
        lists = _all_manifest_validator_check_lists(d)
        assert len(lists) == 1, f"{fn}: expected 1 manifest-validator checks list, got {[p for p, _ in lists]}"
        assert lists[0][0] == ".deep_manifest_validator", f"{fn}: {lists[0][0]}"
        assert d["deep_manifest_validator"].get("authoritative") is True


# ---- §2.2 stale duplicate block removed ----
def test_22_stale_deep_validation_block_removed():
    for fn in _ACTIVE:
        mi = _load(fn).get("manifest_integrity", {})
        assert "deep_validation" not in mi, f"{fn}: manifest_integrity.deep_validation must be absent"
        removed = mi.get("deep_validation_removed")
        if removed is not None:  # allowed only as an explanatory string, never a checks list
            assert not (isinstance(removed, dict) and "checks" in removed), f"{fn}: removed marker has checks"


# ---- §2.3 no old semantics anywhere in the active machine config ----
def test_23_no_stale_semantics_in_active_config():
    for fn in _ACTIVE:
        blob = json.dumps(_load(fn)).lower()
        for s in _STALE:
            assert s not in blob, f"{fn}: stale phrasing present -> {s!r}"


# ---- §2.4 the authoritative block carries the correct semantics ----
def test_24_authoritative_block_has_correct_semantics():
    for fn in _ACTIVE:
        checks = " ".join(_load(fn)["deep_manifest_validator"]["checks"]).lower()
        for g in _REQUIRED_SEMANTICS:
            assert g in checks, f"{fn}: missing authoritative semantics -> {g!r}"


# ---- §4 invariance ----
def test_4_design_power_and_003_004_005_invariant():
    d = _load("confirmatory_v4_config.json")["design"]
    assert d["candidate_bank"] == [-0.04, 0.0, 0.04] and d["test_nominals"] == [-0.035, 0.035]
    assert d["blocks"] == {"train": 9, "val": 6, "test": 9}
    assert json.loads((_DOCS / "test_geometry_power_verdict_v1.json").read_text())["verdict"] \
        == "POWER_SUFFICIENT_FOR_PREREG_V4"
    import re
    cs = re.sub(r"\s+", " ", (_DOCS / "confirmatory_v4_claim_scope.md").read_text().lower())
    assert "auxiliary multitask" in cs  # FINAL-004 unchanged
    sm = json.loads((_DOCS / "final_snapshot_v1" / "snapshot_manifest.json").read_text())
    assert sm["final_head_commit"] == "b983b7e9e638e2ed03589e2bd061139b073ce53f"  # FINAL-005 unchanged
