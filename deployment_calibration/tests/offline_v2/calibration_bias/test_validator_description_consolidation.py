"""B-owned regression for the validator-description consolidation (offline). No Isaac, no generator, no data.

Root cause fixed: the active config had TWO validator description blocks (deep_manifest_validator +
manifest_integrity.deep_validation.checks). Now exactly one authoritative checks list; the reference block
carries none.
"""

from __future__ import annotations

import json
from pathlib import Path

from deployment_calibration.offline_v2.calibration_bias import preregistration_v4 as P

_DOCS = Path(P._docs_dir())


def _cfg():
    return json.loads((_DOCS / "confirmatory_v4_config.json").read_text())


def _pre():
    return json.loads((_DOCS / "preregistration_v4.json").read_text())


def test_status_consolidated():
    assert P.STATUS == "PREREGISTRATION_V4_BLOCK_STATE_KAT_GATE_READY_FOR_C_FINAL_REAUDIT"
    assert _pre()["power_recertification_required"] is False


def _validator_check_lists(cfg):
    """All validator-description 'checks' lists in the config (deep_manifest_validator +
    manifest_integrity.deep_validation), mirroring Claude C's aggregation."""
    out = []
    if "checks" in cfg.get("deep_manifest_validator", {}):
        out.append(("deep_manifest_validator.checks", cfg["deep_manifest_validator"]["checks"]))
    dv = cfg.get("manifest_integrity", {}).get("deep_validation", {})
    if "checks" in dv:
        out.append(("manifest_integrity.deep_validation.checks", dv["checks"]))
    return out


# ---- 6.1 exactly one authoritative full checks list, at deep_manifest_validator.checks ----
def test_exactly_one_validator_checks_list():
    for cfg in (_cfg(), _pre()):
        lists = _validator_check_lists(cfg)
        assert len(lists) == 1, f"expected 1 validator checks list, got {[p for p, _ in lists]}"
        assert lists[0][0] == "deep_manifest_validator.checks"
    assert _cfg()["deep_manifest_validator"].get("authoritative") is True


# ---- 6.2 Scheme B: the old duplicate deep_validation block is removed entirely ----
def test_deep_validation_removed_scheme_b():
    for cfg in (_cfg(), _pre()):
        mi = cfg["manifest_integrity"]
        assert "deep_validation" not in mi                 # no second block at all
        assert "deep_validation_removed" in mi             # documented removal
    # the authoritative block absorbed the phase split composition (nothing lost)
    assert "phase_split_composition" in _cfg()["deep_manifest_validator"]


# ---- 6.3 no stale phrases anywhere in active validator description text ----
def test_no_stale_phrases_in_any_active_validator_block():
    for cfg in (_cfg(), _pre()):
        txt = " ".join(c for _, lst in _validator_check_lists(cfg) for c in lst).lower()
        # include the whole manifest_integrity + deep_manifest_validator objects for safety
        txt += " " + json.dumps(cfg.get("deep_manifest_validator", {})).lower()
        txt += " " + json.dumps(cfg.get("manifest_integrity", {})).lower()
        assert "resolved_candidate_order is a bank permutation" not in txt
        assert "config/commit hex formats" not in txt
        assert "device==cpu" not in txt


# ---- 6.4 correct semantics present in the single authoritative list ----
def test_authoritative_list_has_correct_semantics():
    checks = " ".join(_cfg()["deep_manifest_validator"]["checks"]).lower()
    for phrase in ("id.resolve_candidate_order", "id.resolve_phase_execution_plan", "canonical storage order",
                   "canonical_config_sha256", "confirmatory_v4_phase_manifest_v1", "frozen",
                   "generator step 0"):
        assert phrase in checks, phrase
    assert "storage_order_vs_execution_order" in json.dumps(_cfg()).lower()


def test_authoritative_note_present():
    assert "authoritative validator description = deep_manifest_validator" in \
        json.dumps(_cfg()["deep_manifest_validator"]).lower()


# ---- design/power unchanged ----
def test_design_power_unchanged():
    d = _cfg()["design"]
    assert d["candidate_bank"] == [-0.04, 0.0, 0.04] and d["test_nominals"] == [-0.035, 0.035]
    assert d["blocks"] == {"train": 9, "val": 6, "test": 9}
    assert json.loads((_DOCS / "test_geometry_power_verdict_v1.json").read_text())["verdict"] \
        == "POWER_SUFFICIENT_FOR_PREREG_V4"
