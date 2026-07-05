"""Reverse-property tests proving GEN-B-001..007 are CLOSED (offline; no Isaac). Each test asserts the
FIXED behaviour (the opposite of the frozen finding-demonstration in test_confirmatory_v4_generator_audit_b.py)."""

from __future__ import annotations

import hashlib
import inspect
import json

import pytest

from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_generator as GEN
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_generator_cli as CLI
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as ID
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI
from deployment_calibration.offline_v2.calibration_bias import preregistration_v4 as PRE

AUTH = GEN.GeneratorAuthorization(smoke_only=True)
ENV = GEN.EnvironmentVersionContext({"numpy": "1.26.0"})


def _build(phase, commits=None):
    return GEN.build_phase_manifest_in_memory(phase, auth=AUTH,
                                              commits=commits or GEN.smoke_placeholder_commits(),
                                              environment_versions=ENV)


# ---------------- GEN-B-001: immutable manifest/kat snapshot bound to hash ----------------
def test_gen_b_001_generated_phase_is_immutable_snapshot():
    g = _build("test")
    m1 = g.unsealed_manifest
    m1["blocks"][0]["residual_value"] = 0.00777              # mutate the returned copy
    m1["trials"][0]["execution_order_index"] = 999
    m1["counts"]["blocks"] = 123
    m2 = g.unsealed_manifest                                  # fresh copy-on-read -> mutation did NOT persist
    assert m2["blocks"][0]["residual_value"] != 0.00777
    assert m2["trials"][0]["execution_order_index"] != 999
    assert m2["counts"]["blocks"] == 9
    # hash is bound to the internal snapshot and matches a re-hash of a fresh copy
    assert MI.fully_resolved_phase_manifest_hash(g.unsealed_manifest) == g.full_manifest_sha256
    # kat report copy cannot pollute internal
    k = g.kat_report; k["passed"] = False
    assert g.kat_report["passed"] is True
    with pytest.raises((AttributeError, Exception)):
        g.full_manifest_sha256 = "x"                         # frozen dataclass


# ---------------- GEN-B-002: envelope immutable; post-scan injection impossible ----------------
def test_gen_b_002_envelope_public_secret_immutable():
    tid = ID.canonical_phase_trial_identities("test")[0]
    env = GEN.build_trial_execution_envelope("test", tid, auth=AUTH)
    p1 = env.public_trial_spec
    p1["residual_bias"] = 0.009                               # inject secret into the returned copy
    p1["nested"] = {"actual_bias": 1}
    p2 = env.public_trial_spec                                # fresh copy -> injection did not persist
    assert "residual_bias" not in p2 and "nested" not in p2
    GEN.assert_no_secret_in_public(p2)                        # still clean
    s1 = env.secret_environment_spec; s1["x"] = 1
    assert "x" not in env.secret_environment_spec


# ---------------- GEN-B-003: no unfrozen target_open_position value ----------------
def test_gen_b_003_no_unfrozen_target_value():
    src = inspect.getsource(GEN)
    assert '"target_open_position": 0.20' not in src
    assert "target_open_position = 0.20" not in src
    tid = ID.canonical_phase_trial_identities("test")[0]
    env = GEN.build_trial_execution_envelope("test", tid, auth=AUTH)
    assert "target_open_position" not in env.public_trial_spec
    assert not hasattr(PRE, "TARGET_OPEN_POSITION")


# ---------------- GEN-B-004: smoke build requires placeholder commits ----------------
def test_gen_b_004_smoke_requires_placeholder_commits():
    _build("test", commits=GEN.smoke_placeholder_commits())  # placeholder accepted
    real = GEN.CommitContext("1" * 40, "2" * 40, "3" * 40)
    with pytest.raises(GEN.SmokeCommitContextError):
        _build("test", commits=real)                         # real-looking rejected
    # rejection produced no object/hash (exception before build) — nothing to assert beyond raise
    assert not GEN.commit_context_is_smoke_placeholder(real)


# ---------------- GEN-B-005: environment provenance contract + immutability ----------------
def test_gen_b_005_env_context_immutable_and_rejects_bad():
    for bad in ({}, {"numpy": 1}, {"numpy": ""}, {"numpy": True}, {"numpy": None}, {"": "x"}, {"numpy": ["1"]}):
        with pytest.raises(GEN.EnvironmentProvenanceError):
            GEN.EnvironmentVersionContext(bad)
    src = {"numpy": "1.26.0"}
    ctx = GEN.EnvironmentVersionContext(src)
    src["numpy"] = "changed"                                  # caller mutation
    assert ctx.values["numpy"] == "1.26.0"                    # context unaffected (copied)


def test_gen_b_005_smoke_provenance_ten_key_contract():
    full = {k: "v" for k in GEN.SMOKE_ENV_VERSION_KEYS}
    full.update(GEN.SMOKE_ENV_FIXED)
    GEN.validate_smoke_environment_provenance(GEN.EnvironmentVersionContext(full))     # ok
    # missing a key
    miss = dict(full); miss.pop("machine")
    with pytest.raises(GEN.EnvironmentProvenanceError):
        GEN.validate_smoke_environment_provenance(GEN.EnvironmentVersionContext(miss))
    # extra key
    extra = dict(full); extra["extra"] = "v"
    with pytest.raises(GEN.EnvironmentProvenanceError):
        GEN.validate_smoke_environment_provenance(GEN.EnvironmentVersionContext(extra))
    # wrong fixed marker
    wrong = dict(full); wrong["execution_mode"] = "FORMAL"
    with pytest.raises(GEN.EnvironmentProvenanceError):
        GEN.validate_smoke_environment_provenance(GEN.EnvironmentVersionContext(wrong))


def test_gen_b_005_cli_env_versions_complete_and_no_silent_drop():
    csrc = inspect.getsource(CLI._env_versions)
    assert "except Exception:\n" not in csrc or "pass" not in csrc      # no silent drop
    assert "UNAVAILABLE:" in csrc and "platform." in csrc
    env = CLI._env_versions()
    v = env.values
    assert set(v) == set(GEN.SMOKE_ENV_VERSION_KEYS)
    assert v["execution_mode"] == "SMOKE_ONLY" and v["isaac_status"] == "NOT_IMPORTED_NOT_LAUNCHED"
    assert v["gpu_status"] == "NOT_USED_CPU_SMOKE"
    assert all(isinstance(x, str) and x for x in v.values())
    # environment enters the manifest + hash
    g = GEN.build_phase_manifest_in_memory("test", auth=AUTH, commits=GEN.smoke_placeholder_commits(),
                                           environment_versions=env)
    assert g.unsealed_manifest["environment_versions"] == v


# ---------------- GEN-B-006: strict selection + candidate completion ----------------
def test_gen_b_006_strict_selection_domain():
    ev = hashlib.sha256(b"e").hexdigest()
    for bad in (False, True, "0.0", "nan", None, 0.0004, 0.041, float("nan"), float("inf")):
        c = GEN.SessionRunController("s"); c.probe_ready(); c.probe_complete()
        with pytest.raises(GEN.SelectionOrderError):
            c.freeze_selection(bad, ev)
    # bank value near-canonical accepted -> exact frozen float
    c = GEN.SessionRunController("s"); c.probe_ready(); c.probe_complete()
    c.freeze_selection(0.0 + 1e-13, ev)
    assert c._selected_offset in ID.CANDIDATE_BANK


def test_gen_b_006_evidence_hash_strict():
    good = hashlib.sha256(b"x").hexdigest()
    for bad in ("", None, 123, good.upper(), good[:-1], good + "a", "zz" + good[2:]):
        c = GEN.SessionRunController("s"); c.probe_ready(); c.probe_complete()
        with pytest.raises(GEN.SelectionOrderError):
            c.freeze_selection(0.0, bad)


def test_gen_b_006_candidate_completion_order_and_attempts():
    ev = hashlib.sha256(b"e").hexdigest()
    order = tuple(ID.resolve_candidate_order("test", 0, 0.035))
    pid = "v4ep-" + "a" * 24
    c = GEN.SessionRunController("s", resolved_candidate_order=order)
    c.probe_ready(); c.probe_complete(); c.freeze_selection(order[0], ev)
    # bad resolved order rejected
    bad = GEN.SessionRunController("s", resolved_candidate_order=(0.0, 0.0, 0.0))
    bad.probe_ready(); bad.probe_complete(); bad.freeze_selection(0.0, ev)
    with pytest.raises(GEN.SelectionOrderError):
        bad.candidates_ready()
    c.candidates_ready()
    # out-of-order completion rejected
    with pytest.raises(GEN.SelectionOrderError):
        c.record_candidate_complete(order[1], planned_episode_id=pid, attempt_index=0)
    c.record_candidate_complete(order[0], planned_episode_id=pid, attempt_index=0)
    # duplicate rejected
    with pytest.raises(GEN.SelectionOrderError):
        c.record_candidate_complete(order[0], planned_episode_id=pid, attempt_index=1)
    # attempt domain: bool / 2 / -1 rejected
    for bad_att in (True, 2, -1):
        with pytest.raises(Exception):
            c.record_candidate_complete(order[1], planned_episode_id=pid, attempt_index=bad_att)
    c.record_candidate_complete(order[1], planned_episode_id=pid, attempt_index=1)
    c.record_candidate_complete(order[2], planned_episode_id=pid, attempt_index=0)
    c.session_complete()
    assert c.state == "SESSION_COMPLETE"
    with pytest.raises(GEN.SelectionOrderError):              # candidate outcomes never fed to selector
        c.offer_candidate_outcome_to_selector()


# ---------------- GEN-B-007: atomic smoke summary ----------------
def test_gen_b_007_atomic_write_readback_idempotent_collision(tmp_path, monkeypatch):
    monkeypatch.setattr(CLI, "_smoke_dir", lambda: str(tmp_path / "smoke"))
    src = inspect.getsource(CLI._atomic_write_smoke_summary)
    assert "os.replace" in src and "fsync" in src            # temp + fsync + rename
    s1 = CLI.run_smoke("test")
    path = s1["summary_path"]
    assert s1["write_complete"] is True and "_summary_path" not in s1
    on_disk = json.loads(open(path).read())
    assert on_disk == s1                                      # return == disk
    # idempotent: identical content re-write succeeds, still one file
    s2 = CLI.run_smoke("test")
    assert s2 == s1 and len(list((tmp_path / "smoke").iterdir())) == 1
    # collision: differing content refused, original preserved, no temp residue
    with pytest.raises(CLI.SmokeArtifactCollisionError):
        CLI._atomic_write_smoke_summary(path, {"different": True})
    assert json.loads(open(path).read()) == s1
    assert not any(p.name.startswith(".SMOKE_ONLY_tmp_") for p in (tmp_path / "smoke").iterdir())


def test_gen_b_007_failure_leaves_no_target_or_temp(tmp_path, monkeypatch):
    monkeypatch.setattr(CLI, "_smoke_dir", lambda: str(tmp_path / "smoke"))
    # force a read-back/replace failure by monkeypatching os.replace to raise
    import os as _os
    real_replace = _os.replace
    monkeypatch.setattr(CLI.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        CLI._atomic_write_smoke_summary(str(tmp_path / "smoke" / "SMOKE_ONLY_test_summary.json"), {"a": 1})
    monkeypatch.setattr(CLI.os, "replace", real_replace)
    d = tmp_path / "smoke"
    # no target file, no leftover temp
    assert not (d / "SMOKE_ONLY_test_summary.json").exists()
    assert not any(p.name.startswith(".SMOKE_ONLY_tmp_") for p in d.iterdir()) if d.exists() else True
