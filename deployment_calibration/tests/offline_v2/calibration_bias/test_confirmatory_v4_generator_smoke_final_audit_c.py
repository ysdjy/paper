"""Claude C generator+smoke final gate — independent, read-only.

Regresses the frozen GEN-B-001..007 findings and the kept-invariant generator/smoke behaviour. All PASS.
Do NOT modify A/B files or frozen active modules. Run in env_isaaclab (numpy+torch).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile

import pytest


def _GEN():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_generator as GEN
    return GEN


def _MI():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI
    return MI


def _ID():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as ID
    return ID


_FULLENV = dict(python_implementation="CPython", python_version="3.11.15", numpy_version="1.26.0",
                torch_version="2.7.0+cu128", os_system="Linux", os_release="6.8", machine="x86_64",
                isaac_status="NOT_IMPORTED_NOT_LAUNCHED", gpu_status="NOT_USED_CPU_SMOKE",
                execution_mode="SMOKE_ONLY")


def _build(phase="test", commits=None, env=None):
    GEN = _GEN()
    return GEN.build_phase_manifest_in_memory(
        phase, auth=GEN.GeneratorAuthorization(smoke_only=True),
        commits=commits or GEN.smoke_placeholder_commits(),
        environment_versions=GEN.EnvironmentVersionContext(env or _FULLENV))


# ============================ GEN-B-001 ============================
def test_gen_b_001_immutable_snapshot_and_hash_binding():
    GEN, MI = _GEN(), _MI()
    g = _build("test")
    m = g.unsealed_manifest
    m["blocks"][0]["residual_value"] = 9.9
    m["trials"][0]["execution_order_index"] = 999
    m["counts"]["trials"] = 1
    m2 = g.unsealed_manifest
    assert m2["blocks"][0]["residual_value"] != 9.9 and m2["counts"]["trials"] == 72
    assert MI.fully_resolved_phase_manifest_hash(g.unsealed_manifest) == g.full_manifest_sha256
    kr = g.kat_report; kr["passed"] = False
    assert g.kat_report["passed"] is True


# ============================ GEN-B-002 ============================
def test_gen_b_002_envelope_public_immutable_no_secret():
    GEN = _GEN()
    g = _build("test")
    tid = g.unsealed_manifest["trials"][0]["canonical_trial_identity"]
    env = GEN.build_trial_execution_envelope("test", tid, auth=GEN.GeneratorAuthorization(smoke_only=True))
    p = env.public_trial_spec
    p["residual_bias"] = 1.0; p["nested"] = {"actual_bias": 2.0}
    p2 = env.public_trial_spec
    assert "residual_bias" not in p2 and "nested" not in p2
    pub = " ".join(p2.keys()).lower()
    assert not any(t in pub for t in ("residual", "actual_bias", "nominal_bias", "eff", "secret", "damping"))
    with pytest.raises(GEN.PublicPayloadLeakageError):
        GEN.assert_no_secret_in_public({"residual_bias": 1.0})


# ============================ GEN-B-003 ============================
def test_gen_b_003_no_unfrozen_scientific_value_assigned():
    GEN = _GEN()
    # Definitive check: the public envelope carries EXACTLY the frozen identity/order fields -- no synthetic
    # unfrozen scientific value (e.g. target_open_position=0.20) is created. (The only 'target_open_position'
    # mention in the module is a docstring note of its removal, which is not an assignment.)
    g = _build("test")
    tid = g.unsealed_manifest["trials"][0]["canonical_trial_identity"]
    p = GEN.build_trial_execution_envelope("test", tid, auth=GEN.GeneratorAuthorization(smoke_only=True)).public_trial_spec
    assert set(p) == {"canonical_trial_identity", "planned_episode_id", "resume_key", "role", "offset"}


# ============================ GEN-B-004 ============================
def test_gen_b_004_smoke_commit_context():
    GEN = _GEN()
    real = hashlib.sha1(b"x").hexdigest()
    with pytest.raises(GEN.SmokeCommitContextError):
        _build("test", commits=GEN.CommitContext(real, real, real))
    for bad in (GEN.CommitContext("A" * 40, "b" * 40, "c" * 40),   # uppercase
                GEN.CommitContext("a" * 39, "b" * 40, "c" * 40),   # short
                GEN.CommitContext("z" * 40, "b" * 40, "c" * 40)):  # non-hex
        with pytest.raises((GEN.CommitContextError, GEN.SmokeCommitContextError)):
            _build("test", commits=bad)
    _build("test")  # placeholder a/b/c*40 accepted


# ============================ GEN-B-005 (public builder gate) ============================
def test_gen_b_005_provenance_gate_at_public_builder():
    GEN = _GEN()
    assert set(GEN.SMOKE_ENV_VERSION_KEYS) | set(GEN.SMOKE_ENV_FIXED) == set(_FULLENV) and len(_FULLENV) == 10
    for bad in ({"numpy_version": "1.26.0"},                                   # single key
                {k: v for k, v in _FULLENV.items() if k != "torch_version"},   # missing
                dict(_FULLENV, extra="1"),                                     # extra
                dict(_FULLENV, numpy_version=""),                              # empty value
                dict(_FULLENV, execution_mode="FORMAL")):                      # wrong marker
        with pytest.raises(GEN.EnvironmentProvenanceError):
            _build("test", env=bad)
    # gate runs BEFORE _build_unsealed_manifest (cannot bypass CLI for a partial-provenance hash)
    orig = GEN._build_unsealed_manifest
    GEN._build_unsealed_manifest = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("BUILD_CALLED"))
    try:
        with pytest.raises(GEN.EnvironmentProvenanceError):
            _build("test", env={"numpy_version": "1.26.0"})
    finally:
        GEN._build_unsealed_manifest = orig
    # a changed allowed version string changes the full hash
    assert _build("test", env=dict(_FULLENV, python_version="3.12.0")).full_manifest_sha256 \
        != _build("test").full_manifest_sha256


# ============================ GEN-B-006 ============================
def test_gen_b_006_strict_selection_and_candidate_order():
    GEN, ID = _GEN(), _ID()
    for bad in (True, "0.0", None, float("nan"), 0.0004):
        with pytest.raises(GEN.SelectionOrderError):
            GEN.canonical_selected_offset(bad)
    assert GEN.canonical_selected_offset(0.04) == 0.04
    for bad in ("a" * 63, "A" * 64, "g" * 64):
        with pytest.raises(GEN.SelectionOrderError):
            GEN._validate_evidence_hash(bad)
    order = (-0.04, 0.0, 0.04)
    s = GEN.SessionRunController("s0", resolved_candidate_order=order)
    s.probe_ready(); s.probe_complete(); s.freeze_selection(0.04, "a" * 64); s.candidates_ready()
    with pytest.raises(GEN.SelectionOrderError):
        s.session_complete()   # before all candidates
    pid = lambda off: ID.planned_episode_id(ID.trial_identity("test", 0, 0.035, "candidate", off))
    s.record_candidate_complete(-0.04, planned_episode_id=pid(-0.04), attempt_index=0)
    with pytest.raises(GEN.SelectionOrderError):
        s.record_candidate_complete(0.04, planned_episode_id=pid(0.04), attempt_index=0)  # out of order
    with pytest.raises(GEN.SelectionOrderError):
        s.record_candidate_complete(-0.04, planned_episode_id=pid(-0.04), attempt_index=0)  # dup
    with pytest.raises(ValueError):
        s.record_candidate_complete(0.0, planned_episode_id=pid(0.0), attempt_index=2)  # attempt>1
    with pytest.raises(GEN.SelectionOrderError):
        s.offer_candidate_outcome_to_selector(1)
    s.record_candidate_complete(0.0, planned_episode_id=pid(0.0), attempt_index=0)
    s.record_candidate_complete(0.04, planned_episode_id=pid(0.04), attempt_index=1)
    s.session_complete()
    assert s.state == "SESSION_COMPLETE"


# ============================ GEN-B-007 ============================
def test_gen_b_007_atomic_summary():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_generator_cli as CLI
    d = tempfile.mkdtemp()
    p = os.path.join(d, "SMOKE_ONLY_x_summary.json")
    payload = {"a": 1, "write_complete": True}
    CLI._atomic_write_smoke_summary(p, payload)
    assert json.load(open(p)) == payload            # read-back equal
    CLI._atomic_write_smoke_summary(p, payload)      # idempotent
    with pytest.raises(CLI.SmokeArtifactCollisionError):
        CLI._atomic_write_smoke_summary(p, {"a": 2})  # differing -> collision, never overwrite
    assert json.load(open(p)) == payload            # original intact
    assert not any(f.startswith(".SMOKE_ONLY_tmp_") for f in os.listdir(d))  # no leftover temp


# ============================ regression (kept invariants) ============================
def test_regression_kat_blockstate_formalwriter_counts():
    import inspect
    GEN, MI = _GEN(), _MI()
    src = inspect.getsource(GEN)
    assert "open(" not in src                                   # writes no files
    assert "_residual_value_from_subseed_unchecked" not in src  # never calls unchecked core
    g_tv, g_te = _build("train_validation"), _build("test")
    assert g_tv.trial_count == 228 and g_te.trial_count == 72
    # formal writer locked, zero side effect
    with pytest.raises(GEN.FormalGenerationNotAuthorized):
        GEN.write_formal_phase_manifest_atomic("test", g_te, os.path.join(tempfile.mkdtemp(), "f.json"),
                                               auth=GEN.GeneratorAuthorization(smoke_only=True))
    # KAT report bound into the phase
    assert g_te.kat_report["known_answer_version"] == "pcg64_normal_floathex_kat_v1"


def test_regression_power_invariants():
    from pathlib import Path
    docs = Path(__file__).resolve().parents[4] / "docs" / "offline_v2" / "calibration_bias"
    d = json.loads((docs / "confirmatory_v4_config.json").read_text())["design"]
    assert d["candidate_bank"] == [-0.04, 0.0, 0.04] and d["test_nominals"] == [-0.035, 0.035]
    assert d["blocks"] == {"train": 9, "val": 6, "test": 9}
    assert json.loads((docs / "test_geometry_power_verdict_v1.json").read_text())["verdict"] \
        == "POWER_SUFFICIENT_FOR_PREREG_V4"
