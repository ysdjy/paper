"""B-owned regression for the one-shot batch fix FINAL-001..005 (offline). No Isaac, no generator, no data.

One test group per frozen issue. C's one-shot audit file is NOT modified; this is the B-owned evidence that
the five frozen issues are resolved.
"""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest

from deployment_calibration.offline_v2.calibration_bias import preregistration_v4 as P
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as ID
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI

_DOCS = Path(P._docs_dir())


def _cfg():
    return json.loads((_DOCS / "confirmatory_v4_config.json").read_text())


def test_status_is_batch_fix():
    assert P.STATUS == "PREREGISTRATION_V4_VALIDATOR_DESCRIPTION_CONSOLIDATED_READY_FOR_C_REGRESSION"
    assert json.loads((_DOCS / "preregistration_v4.json").read_text())["power_recertification_required"] is False


# ==================== FINAL-001: frozen seed -> order resolution ====================
def test_final_001_resolvers_exist():
    for fn in ("resolve_order", "resolve_block_order", "resolve_session_order", "resolve_candidate_order",
               "resolve_phase_execution_plan", "block_order_key", "candidate_item_order_key",
               "candidate_order_key_map"):
        assert hasattr(ID, fn), fn


def test_final_001_no_random_or_hash_in_resolvers():
    src = "\n".join(inspect.getsource(getattr(ID, f)) for f in
                    ("resolve_order", "resolve_block_order", "resolve_session_order", "resolve_candidate_order",
                     "resolve_phase_execution_plan"))
    for bad in ("random.", "np.random", "shuffle", "hash(", ".items()", ".keys()", ".values()"):
        assert bad not in src, f"resolver uses forbidden ordering source: {bad}"


def test_final_001_resolve_order_permutation_stable_and_collision_tiebreak():
    items = list(range(9))
    r1 = ID.resolve_order(items, key_fn=lambda i: (i * 7) % 5, canonical_index_fn=lambda i: i)
    r2 = ID.resolve_order(list(reversed(items)), key_fn=lambda i: (i * 7) % 5, canonical_index_fn=lambda i: i)
    assert r1 == r2                                   # permutation-independent
    # collision (all key 0) -> canonical index order
    assert ID.resolve_order([3, 1, 2], key_fn=lambda i: 0, canonical_index_fn=lambda i: i) == (1, 2, 3)
    with pytest.raises(ValueError):                   # duplicate item
        ID.resolve_order([1, 1], key_fn=lambda i: i, canonical_index_fn=lambda i: i)
    with pytest.raises(ValueError):                   # bool key
        ID.resolve_order([1], key_fn=lambda i: True, canonical_index_fn=lambda i: i)


def test_final_001_plans_228_72_probe_first_and_deterministic():
    tv = ID.resolve_phase_execution_plan("train_validation")
    te = ID.resolve_phase_execution_plan("test")
    assert len(tv) == 228 and len(te) == 72
    assert len(set(tv)) == 228 and len(set(te)) == 72
    assert ID.resolve_phase_execution_plan("test") == te            # deterministic (byte-identical)
    for i in range(0, len(te), 4):                                  # each session: probe first
        assert "role=probe" in te[i]


def test_final_001_validator_rejects_order_tampers():
    def bad(fn):
        m = MI.reference_phase_manifest("test"); fn(m)
        with pytest.raises(MI.ManifestIntegrityError):
            MI.validate_fully_resolved_phase_manifest(m)
    # a genuinely different candidate order (not the resolved one)
    s0 = MI.reference_phase_manifest("test")["sessions"][0]
    alt = list(reversed(s0["resolved_candidate_order"]))
    if tuple(alt) == tuple(s0["resolved_candidate_order"]):
        alt = [s0["resolved_candidate_order"][1], s0["resolved_candidate_order"][0], s0["resolved_candidate_order"][2]]
    bad(lambda m: m["sessions"][0].__setitem__("resolved_candidate_order", list(alt)))
    bad(lambda m: m["blocks"][0].__setitem__("block_order_key", 123))
    bad(lambda m: m["sessions"][0]["candidate_order_keys"].__setitem__(
        list(m["sessions"][0]["candidate_order_keys"])[0], 1))
    # swap two execution indices (still 0..N-1) -> != resolved plan -> INVALID
    def swap(m):
        m["trials"][1]["execution_order_index"], m["trials"][2]["execution_order_index"] = \
            m["trials"][2]["execution_order_index"], m["trials"][1]["execution_order_index"]
    bad(swap)


def test_final_001_config_documents_order_resolution():
    orr = json.loads((_DOCS / "preregistration_v4.json").read_text())["order_resolution"]
    assert "resolve_phase_execution_plan" in orr["resolvers"]
    assert "execution_order_index == plan index" in orr["execution_order"]


# ==================== FINAL-002: exact frozen integrity values ====================
def test_final_002_validator_uses_canonical_config_sha256():
    src = inspect.getsource(MI.validate_fully_resolved_phase_manifest)
    assert "canonical_config_sha256(" in src


def test_final_002_config_hash_exact_checked():
    m = MI.reference_phase_manifest("test")
    MI.validate_fully_resolved_phase_manifest(m)          # passes with the real config hash
    assert m["config_sha256"] == MI.canonical_config_sha256() and len(m["config_sha256"]) == 64
    m["config_sha256"] = "a" * 64                         # wrong (but valid-format) -> INVALID
    with pytest.raises(MI.ManifestIntegrityError):
        MI.validate_fully_resolved_phase_manifest(m)


def test_final_002_schema_version_exact():
    assert MI.PHASE_MANIFEST_SCHEMA_VERSION == "confirmatory_v4_phase_manifest_v1"
    m = MI.reference_phase_manifest("test"); m["schema_version"] = "x"
    with pytest.raises(MI.ManifestIntegrityError):
        MI.validate_fully_resolved_phase_manifest(m)


def test_final_002_determinism_contract_exact():
    contract = MI.deterministic_environment_contract()
    assert contract == _cfg()["deterministic_environment_manifest_contract"]
    # missing key
    m = MI.reference_phase_manifest("test"); del m["deterministic_environment"]["OMP_NUM_THREADS"]
    with pytest.raises(MI.ManifestIntegrityError):
        MI.validate_fully_resolved_phase_manifest(m)
    # extra key
    m = MI.reference_phase_manifest("test"); m["deterministic_environment"]["extra"] = 1
    with pytest.raises(MI.ManifestIntegrityError):
        MI.validate_fully_resolved_phase_manifest(m)
    # wrong type (bool for int)
    m = MI.reference_phase_manifest("test"); m["deterministic_environment"]["torch_set_num_threads"] = True
    with pytest.raises(MI.ManifestIntegrityError):
        MI.validate_fully_resolved_phase_manifest(m)
    # GPU
    m = MI.reference_phase_manifest("test"); m["deterministic_environment"]["device"] = "cuda"
    with pytest.raises(MI.ManifestIntegrityError):
        MI.validate_fully_resolved_phase_manifest(m)


def test_final_002_freeze_source_table_documented():
    spec = (_DOCS / "confirmatory_v4_manifest_spec.md").read_text()
    assert "Freeze-source table" in spec
    assert "generator Step 0" in spec and "canonical_config_sha256()" in spec


# ==================== FINAL-003: stale field names removed ====================
def test_final_003_no_stale_manifest_field_names():
    spec = (_DOCS / "confirmatory_v4_manifest_spec.md").read_text()
    assert "manifest_hash " not in spec and "config_hash " not in spec and "code_commit " not in spec
    for k in ("planned_structure_sha256", "train_validation_manifest_sha256", "test_manifest_sha256",
              "combined_experiment_plan_sha256", "runtime_commit"):
        assert k in spec


# ==================== FINAL-004: multitask disclaimer ====================
def test_final_004_disclaimer_present():
    txt = " ".join(p.read_text() for p in _DOCS.glob("confirmatory_v4_*.md")).lower()
    assert "auxiliary" in txt and "error/time" in txt
    assert "success-only predictor" in txt
    md = (_DOCS / "preregistration_v4.md").read_text().lower()
    assert "auxiliary" in md and "success only" in md


# ==================== FINAL-005: snapshot commit disambiguation ====================
def test_final_005_snapshot_commit_disambiguated():
    sm = json.loads((_DOCS / "final_snapshot_v1" / "snapshot_manifest.json").read_text())
    assert sm["final_head_commit"].startswith("b983b7e")
    assert sm["audited_snapshot_commit"].startswith("b983b7e")
    assert sm["metadata_payload_commit"].startswith("1abaa72")
    assert "snapshot_commit" not in sm


# ==================== unchanged science / no artifacts ====================
def test_power_and_design_unchanged():
    d = _cfg()["design"]
    assert d["candidate_bank"] == [-0.04, 0.0, 0.04] and d["test_nominals"] == [-0.035, 0.035]
    assert _cfg()["trial_counts"]["full_task_trials_total"] == 300
    assert json.loads((_DOCS / "test_geometry_power_verdict_v1.json").read_text())["verdict"] \
        == "POWER_SUFFICIENT_FOR_PREREG_V4"


def test_no_generator_manifest_checkpoint_data():
    hc = json.loads((_DOCS / "preregistration_v4.json").read_text())["hard_constraints"]
    assert "no confirmatory generator" in hc and "no run authorization" in hc
    names = [p.name for p in _DOCS.iterdir()]
    assert not any(n.endswith(".pt") or "manifest_instance" in n or "confirmatory_data" in n for n in names)
