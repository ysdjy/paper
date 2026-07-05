"""Tests for the KAT-gated confirmatory v4 generator (offline; no Isaac). Covers §24.1-24.6."""

from __future__ import annotations

import json
import os

import pytest

from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as ID
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_block_state as BS
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI
from deployment_calibration.offline_v2.calibration_bias import preregistration_v4 as PRE
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_generator as GEN
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_generator_cli as CLI

SMOKE = GEN.GeneratorAuthorization(smoke_only=True)
COMMITS = GEN.smoke_placeholder_commits()


def full_smoke_env_mapping():
    """Complete 10-key smoke provenance (the public builder now enforces the exact contract, GEN-B-005)."""
    return {"python_implementation": "CPython", "python_version": "3.10.0", "numpy_version": "1.26.0",
            "torch_version": "2.7.0+cu128", "os_system": "Linux", "os_release": "test", "machine": "x86_64",
            "isaac_status": "NOT_IMPORTED_NOT_LAUNCHED", "gpu_status": "NOT_USED_CPU_SMOKE",
            "execution_mode": "SMOKE_ONLY"}


FULL_ENV_MAPPING = full_smoke_env_mapping()
ENV = GEN.EnvironmentVersionContext(FULL_ENV_MAPPING)


def _build(phase):
    return GEN.build_phase_manifest_in_memory(phase, auth=SMOKE, commits=COMMITS, environment_versions=ENV)


# ----------------------------------------------------------------- 24.1 authorization
def test_authorization_requires_smoke_only():
    GEN.require_generator_authorization(SMOKE)                       # ok
    with pytest.raises(GEN.FormalGenerationNotAuthorized):
        GEN.require_generator_authorization(GEN.GeneratorAuthorization(smoke_only=False))
    for kw in ("formal_manifest_generation_authorized", "confirmatory_data_generation_authorized",
               "confirmatory_run_authorized"):
        with pytest.raises(GEN.FormalGenerationNotAuthorized):
            GEN.require_generator_authorization(GEN.GeneratorAuthorization(smoke_only=True, **{kw: True}))


def test_formal_writer_zero_side_effect(tmp_path):
    out = tmp_path / "formal_manifest.json"
    g = _build("test")
    with pytest.raises(GEN.FormalGenerationNotAuthorized):
        GEN.write_formal_phase_manifest_atomic("test", g, str(out), auth=SMOKE)
    assert not out.exists()                                          # no file, no dir side effects
    assert list(tmp_path.iterdir()) == []


def test_commit_context_validation():
    GEN.validate_commit_context(COMMITS)
    assert GEN.commit_context_is_smoke_placeholder(COMMITS)
    with pytest.raises(GEN.CommitContextError):
        GEN.validate_commit_context(GEN.CommitContext("x" * 40, "b" * 40, "c" * 40))   # non-hex
    with pytest.raises(GEN.CommitContextError):
        GEN.validate_commit_context(GEN.CommitContext("a" * 39, "b" * 40, "c" * 40))   # wrong length


# ----------------------------------------------------------------- 24.2 mandatory KAT
def test_generator_source_never_calls_unchecked_core():
    src = open(GEN.__file__).read()
    assert "_residual_value_from_subseed_unchecked" not in src
    assert "resolved_block_state" in src and "require_known_answer_compatibility" in src


def test_kat_failure_blocks_build(monkeypatch):
    def boom():
        raise BS.BlockStateCompatibilityError("KAT forced fail")
    monkeypatch.setattr(BS, "require_known_answer_compatibility", boom)
    with pytest.raises(BS.BlockStateCompatibilityError):
        _build("test")


def test_kat_failure_writes_no_smoke_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(CLI, "_smoke_dir", lambda: str(tmp_path / "smoke"))

    def boom():
        raise BS.BlockStateCompatibilityError("KAT forced fail")
    monkeypatch.setattr(BS, "require_known_answer_compatibility", boom)
    with pytest.raises(BS.BlockStateCompatibilityError):
        CLI.run_smoke("test")
    assert not (tmp_path / "smoke").exists()                         # no summary/hash/file on KAT failure


def test_build_runs_kat_and_reports():
    g = _build("test")
    assert g.kat_report["passed"] is True
    assert g.kat_report["known_answer_version"] == BS.KNOWN_ANSWER_VERSION
    assert g.kat_report["vectors_checked"] == len(BS.KNOWN_ANSWER_VECTORS)


# ----------------------------------------------------------------- 24.3 phase construction
@pytest.mark.parametrize("phase,exp", [("train_validation", (15, 57, 228)), ("test", (9, 18, 72))])
def test_phase_counts_and_deep_valid(phase, exp):
    g = _build(phase)
    m = g.unsealed_manifest
    assert (len(m["blocks"]), len(m["sessions"]), len(m["trials"])) == exp
    assert m["counts"] == {"blocks": exp[0], "sessions": exp[1], "trials": exp[2]}
    MI.validate_fully_resolved_phase_manifest(m)                     # deep validator accepts
    assert g.trial_count == exp[2]


@pytest.mark.parametrize("phase", ["train_validation", "test"])
def test_storage_and_execution_order(phase):
    m = _build(phase).unsealed_manifest
    assert tuple(b["canonical_block_identity"] for b in m["blocks"]) == ID.canonical_phase_block_identities(phase)
    assert tuple(s["canonical_session_identity"] for s in m["sessions"]) == ID.canonical_phase_session_identities(phase)
    assert tuple(t["canonical_trial_identity"] for t in m["trials"]) == ID.canonical_phase_trial_identities(phase)
    plan = {tid: i for i, tid in enumerate(ID.resolve_phase_execution_plan(phase))}
    for t in m["trials"]:
        assert t["execution_order_index"] == plan[t["canonical_trial_identity"]]
    # probe-first per session (in execution order)
    by_sess = {}
    for t in m["trials"]:
        by_sess.setdefault(t["session_ref"], []).append(t)
    for sref, tl in by_sess.items():
        probe = [t for t in tl if t["role"] == "probe"]
        cands = [t for t in tl if t["role"] == "candidate"]
        assert len(probe) == 1 and len(cands) == 3
        assert min(c["execution_order_index"] for c in cands) > probe[0]["execution_order_index"]


@pytest.mark.parametrize("phase", ["train_validation", "test"])
def test_block_state_from_frozen_sampler(phase):
    m = _build(phase).unsealed_manifest
    for b in m["blocks"]:
        bst = BS.resolved_block_state(b["split"], b["block_index"])
        assert b["residual_value"] == bst["residual_value"]
        assert b["nuisance_values"] == {} == bst["nuisance_values"]
        assert b["residual_subseed"] == bst["residual_subseed"]


@pytest.mark.parametrize("phase", ["train_validation", "test"])
def test_independent_build_equals_reference(phase):
    """§7: independent construction must EQUAL MI.reference_phase_manifest (equality check, not wrapping)."""
    mine = _build(phase).unsealed_manifest
    ref = MI.reference_phase_manifest(phase, protocol_commit="a" * 40, generator_commit="b" * 40,
                                      runtime_commit="c" * 40,
                                      environment_versions=full_smoke_env_mapping())
    assert ID.canonical_json(mine) == ID.canonical_json(ref)


# ----------------------------------------------------------------- 24.4 determinism
@pytest.mark.parametrize("phase", ["train_validation", "test"])
def test_deterministic_rebuild(phase):
    a, b = _build(phase), _build(phase)
    assert ID.canonical_json(a.unsealed_manifest) == ID.canonical_json(b.unsealed_manifest)
    assert a.full_manifest_sha256 == b.full_manifest_sha256


def test_mutated_block_state_rejected():
    m = _build("test").unsealed_manifest
    bad = json.loads(json.dumps(m))
    bad["blocks"][0]["residual_value"] = 0.009999    # not the frozen sampler output
    with pytest.raises(MI.ManifestIntegrityError):
        MI.validate_fully_resolved_phase_manifest(bad)


# ----------------------------------------------------------------- 24.5 leakage / selection
def test_envelope_public_secret_split():
    tid = ID.canonical_phase_trial_identities("test")[0]
    env = GEN.build_trial_execution_envelope("test", tid, auth=SMOKE)
    GEN.assert_no_secret_in_public(env.public_trial_spec)            # no denylist key
    pub_keys = set(env.public_trial_spec)
    assert "nominal_bias" not in pub_keys and "residual_bias" not in pub_keys
    assert env.secret_environment_spec["residual_bias"] == BS.residual_value(*_split_block(tid))
    assert env.public_trial_spec["planned_episode_id"].startswith("v4ep-")


def _split_block(tid):
    r = next(r for r in ID.enumerate_trials() if r["canonical_trial_identity"] == tid)
    return r["split"], r["block_index"]


def test_leakage_scan_catches_injected_secret():
    for key in ("residual_bias", "actual_bias", "eff_signed", "abs_eff", "oracle_action", "hidden_state_id"):
        with pytest.raises(GEN.PublicPayloadLeakageError):
            GEN.assert_no_secret_in_public({"role": "probe", "nested": {key: 1}})


def test_session_state_machine_probe_first_and_selection_freeze():
    import hashlib
    ev = hashlib.sha256(b"probe-evidence").hexdigest()              # valid 64-hex evidence
    order = tuple(ID.resolve_candidate_order("test", 0, -0.035))    # frozen resolved order (bank permutation)
    pid = "v4ep-" + "0" * 24
    c = GEN.SessionRunController("s0", resolved_candidate_order=order)
    c.probe_ready(); c.probe_complete()
    with pytest.raises(GEN.SelectionOrderError):                    # candidate outcome never reaches the selector
        c.offer_candidate_outcome_to_selector(0.0, True)
    c.freeze_selection(0.0, ev)                                     # frozen on probe evidence, before candidates
    assert c.state == "SELECTION_FROZEN"
    c.candidates_ready()
    with pytest.raises(GEN.SelectionOrderError):                    # cannot complete before recording candidates
        c.session_complete()
    for i, off in enumerate(order):
        aid = c.record_candidate_complete(off, planned_episode_id=pid, attempt_index=0)
        assert aid.endswith("attempt=00")
    c.session_complete()
    assert c.state == "SESSION_COMPLETE"
    # cannot freeze before probe complete
    c2 = GEN.SessionRunController("s1"); c2.probe_ready()
    with pytest.raises(GEN.SelectionOrderError):
        c2.freeze_selection(0.0, ev)
    assert c.attempt_id(pid, 1).endswith("attempt=01")


# ----------------------------------------------------------------- 24.6 smoke isolation
def test_run_smoke_writes_only_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(CLI, "_smoke_dir", lambda: str(tmp_path / "smoke"))
    s = CLI.run_smoke("test")
    files = list((tmp_path / "smoke").iterdir())
    assert len(files) == 1 and files[0].name == "SMOKE_ONLY_test_summary.json"
    data = json.loads(files[0].read_text())
    # summary carries NO full lists / no formal anchor naming
    for forbidden in ("blocks", "sessions", "trials"):
        assert not isinstance(data.get(forbidden), list)
    assert data["formal_manifest_written"] is False and data["confirmatory_records_written"] is False
    assert data["not_a_formal_manifest_anchor"] is True and data["kat_passed"] is True
    assert data["deterministic_rebuild_match"] is True


def test_cli_refuses_without_smoke_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(CLI, "_smoke_dir", lambda: str(tmp_path / "smoke"))
    assert CLI.main(["--phase", "test"]) == 2                        # no --smoke-only
    assert CLI.main(["--smoke-only", "--phase", "test", "--formal"]) == 2
    assert CLI.main(["--smoke-only", "--phase", "test", "--output", "x.json"]) == 2
    assert not (tmp_path / "smoke").exists()                         # zero side effects on refusal


def test_cli_smoke_ok(monkeypatch, tmp_path):
    monkeypatch.setattr(CLI, "_smoke_dir", lambda: str(tmp_path / "smoke"))
    assert CLI.main(["--smoke-only", "--phase", "train_validation"]) == 0
    assert (tmp_path / "smoke" / "SMOKE_ONLY_train_validation_summary.json").exists()
