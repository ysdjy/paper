"""Tests for the confirmatory v4 FORMAL manifest writer + real commit-freeze interface (offline; no Isaac).

Covers §17.1-17.8: the current C gate cannot authorize generation, the authorization loader / commit-freeze /
formal-env contracts, the Step-5 test early-access guard, the stamp pure function, and the atomic one-shot
bundle writer with path safety. Per §16/§18 this suite NEVER builds a full 228/72 formal payload through the
production builder success path and NEVER writes under the repo's artifacts/formal tree — the atomic writer is
exercised only with an explicitly-marked TEST_ONLY payload in pytest tmp_path.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_formal_manifest_writer as FW
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_formal_writer_cli as CLI
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI

A1 = "1" * 40                      # stand-in audited writer / generator commit
PROTO, RUNTIME, AUTHC = "d" * 40, "e" * 40, "f" * 40
H64 = "a" * 64
TEST_ONLY = "TEST_ONLY_NON_SCIENTIFIC_PAYLOAD"


# --------------------------------------------------------------------------- helpers
def tv_auth_raw(**over):
    raw = {"verdict": FW.TRAIN_VALIDATION_VERDICT, "audited_writer_commit": A1,
           "authorized_phase": "train_validation", "protocol_commit": PROTO, "generator_commit": A1,
           "runtime_commit": RUNTIME,
           "authorized_output_relpath": "artifacts/formal/confirmatory_v4/train_validation",
           "authorizes_formal_manifest_generation": True, "authorizes_confirmatory_data_generation": False,
           "authorizes_confirmatory_run": False, "one_shot": True, "power_recertification_required": False}
    raw.update(over)
    return raw


def mk_test_auth_raw(**over):
    raw = tv_auth_raw(verdict=FW.TEST_VERDICT, authorized_phase="test",
                      authorized_output_relpath="artifacts/formal/confirmatory_v4/test",
                      model_analysis_freeze_sha256="b" * 64, train_validation_manifest_sha256="c" * 64,
                      analysis_code_sha256="d" * 64)
    raw.update(over)
    return raw


def write_json(tmp_path, raw, name="auth.json"):
    p = Path(tmp_path) / name
    p.write_text(json.dumps(raw), encoding="utf-8")
    return str(p)


def load_tv(tmp_path, **over):
    return FW.load_and_verify_formal_manifest_authorization(
        write_json(tmp_path, tv_auth_raw(**over)), authorization_commit=AUTHC, expected_writer_commit=A1)


def mint(raw, *, phase, m_freeze=None, tv_manifest=None, analysis_code=None):
    """Construct a token-authorized VerifiedFormalManifestAuthorization directly (white-box) for guard tests
    that must FAIL before any build. Not reachable through the public loader."""
    canonical = FW._canon(raw)
    evidence = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return FW.VerifiedFormalManifestAuthorization(
        authorization_verdict=raw["verdict"], authorization_evidence_sha256=evidence,
        authorization_commit=AUTHC, audited_writer_commit=raw["audited_writer_commit"],
        authorized_phase=phase, protocol_commit=raw["protocol_commit"], generator_commit=raw["generator_commit"],
        runtime_commit=raw["runtime_commit"], authorized_output_relpath=raw["authorized_output_relpath"],
        authorizes_formal_manifest_generation=True, authorizes_confirmatory_data_generation=False,
        authorizes_confirmatory_run=False, one_shot=True,
        test_prerequisite_model_analysis_freeze_sha256=m_freeze,
        test_prerequisite_train_validation_manifest_sha256=tv_manifest,
        test_prerequisite_analysis_code_sha256=analysis_code,
        _raw_authorization_canonical_json=canonical, _verified_loader_version=FW.VERIFIED_LOADER_VERSION,
        _loader_token=FW._AUTHORIZATION_LOADER_TOKEN)


def good_freeze(auth):
    return FW.FormalCommitFreeze(protocol_commit=auth.protocol_commit, generator_commit=auth.generator_commit,
                                 runtime_commit=auth.runtime_commit, authorization_commit=auth.authorization_commit,
                                 authorization_evidence_sha256=auth.authorization_evidence_sha256)


def formal_env():
    return FW.FormalManifestEnvironmentContext({
        "python_implementation": "CPython", "python_version": "3.11.0", "numpy_version": "1.26.0",
        "torch_version": "2.7.0+cu128", "os_system": "Linux", "os_release": "x", "machine": "x86_64",
        "isaac_status": "NOT_IMPORTED_NOT_LAUNCHED", "gpu_status": "NOT_USED_MANIFEST_GENERATION",
        "execution_mode": "FORMAL_MANIFEST_GENERATION"})


# =============================================================== §17.1 current C gate cannot authorize generation
def test_current_c_gate_authorizes_writer_impl_not_generation():
    root = Path(FW.__file__).resolve().parents[3]
    c_gate = root / CLI.C_GATE_RELPATH
    gate = json.loads(c_gate.read_text(encoding="utf-8"))
    assert gate["authorizes_formal_manifest_writer_implementation"] is True
    assert gate["authorizes_formal_manifest_generation"] is False
    # the generation loader REFUSES the current gate
    with pytest.raises(FW.FormalManifestGenerationNotAuthorized):
        FW.load_and_verify_formal_manifest_authorization(
            str(c_gate), authorization_commit="0" * 40, expected_writer_commit="0" * 40)
    # no formal artifact directory exists as a side effect
    assert not (root / "artifacts" / "formal").exists()


def test_cli_preflight_reports_generation_locked(capsys):
    rc = CLI.main(["--preflight-only"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "FORMAL_WRITER_IMPLEMENTED_GENERATION_LOCKED" in out
    assert "--force" not in out


def test_cli_refuses_without_preflight_flag(capsys):
    rc = CLI.main([])
    assert rc == 2
    assert "FORMAL_WRITER_IMPLEMENTED_GENERATION_LOCKED" not in capsys.readouterr().out


# =============================================================== §17.2 authorization loader
def test_loader_accepts_wellformed_train_validation(tmp_path):
    auth = load_tv(tmp_path)
    assert auth.authorized_phase == "train_validation"
    assert auth.authorizes_formal_manifest_generation is True
    assert FW._is_64hex(auth.authorization_evidence_sha256)
    assert auth.authorization_evidence_sha256 == hashlib.sha256(
        auth._raw_authorization_canonical_json.encode()).hexdigest()


def test_loader_accepts_wellformed_test(tmp_path):
    auth = FW.load_and_verify_formal_manifest_authorization(
        write_json(tmp_path, mk_test_auth_raw()), authorization_commit=AUTHC, expected_writer_commit=A1)
    assert auth.authorized_phase == "test"
    assert FW._is_64hex(auth.test_prerequisite_model_analysis_freeze_sha256)


@pytest.mark.parametrize("mutate,exc", [
    (lambda r: r.pop("protocol_commit"), FW.FormalAuthorizationIntegrityError),      # missing field
    (lambda r: r.update(verdict="GENERATOR_SMOKE_GATE_PASS"), FW.FormalManifestGenerationNotAuthorized),  # wrong verdict
    (lambda r: r.update(authorized_phase="test"), FW.FormalAuthorizationIntegrityError),  # verdict/phase (two phases)
    (lambda r: r.update(authorizes_formal_manifest_generation=False), FW.FormalManifestGenerationNotAuthorized),
    (lambda r: r.update(authorizes_confirmatory_data_generation=True), FW.FormalAuthorizationIntegrityError),
    (lambda r: r.update(authorizes_confirmatory_run=True), FW.FormalAuthorizationIntegrityError),
    (lambda r: r.update(authorized_output_relpath="artifacts/formal/confirmatory_v4/test"), FW.FormalAuthorizationIntegrityError),
    (lambda r: r.update(protocol_commit="zz" + "d" * 38), FW.FormalAuthorizationIntegrityError),   # non-40hex
    (lambda r: r.update(one_shot=False), FW.FormalAuthorizationIntegrityError),
])
def test_loader_rejects_bad_train_validation(tmp_path, mutate, exc):
    raw = tv_auth_raw()
    mutate(raw)
    with pytest.raises(exc):
        FW.load_and_verify_formal_manifest_authorization(
            write_json(tmp_path, raw), authorization_commit=AUTHC, expected_writer_commit=A1)


def test_loader_rejects_writer_commit_mismatch(tmp_path):
    with pytest.raises(FW.FormalAuthorizationIntegrityError):
        FW.load_and_verify_formal_manifest_authorization(
            write_json(tmp_path, tv_auth_raw()), authorization_commit=AUTHC, expected_writer_commit="9" * 40)


def test_loader_rejects_generator_ne_writer(tmp_path):
    with pytest.raises(FW.FormalAuthorizationIntegrityError):
        FW.load_and_verify_formal_manifest_authorization(
            write_json(tmp_path, tv_auth_raw(generator_commit="7" * 40)),
            authorization_commit=AUTHC, expected_writer_commit=A1)


def test_loader_rejects_test_missing_or_bad_step5(tmp_path):
    for over in ({"model_analysis_freeze_sha256": None}, {"train_validation_manifest_sha256": "xyz"},
                 {"analysis_code_sha256": "a" * 63}):
        raw = mk_test_auth_raw()
        raw.update(over)
        raw = {k: v for k, v in raw.items() if v is not None}
        with pytest.raises(FW.FormalAuthorizationIntegrityError):
            FW.load_and_verify_formal_manifest_authorization(
                write_json(tmp_path, raw), authorization_commit=AUTHC, expected_writer_commit=A1)


# =============================================================== §17.3 commit freeze
def test_commit_freeze_ok(tmp_path):
    auth = load_tv(tmp_path)
    FW.validate_formal_commit_freeze(good_freeze(auth), auth)          # no raise


def test_commit_freeze_rejects_smoke_placeholder(tmp_path):
    auth = load_tv(tmp_path)
    bad = FW.FormalCommitFreeze("a" * 40, "b" * 40, "c" * 40, auth.authorization_commit,
                                auth.authorization_evidence_sha256)
    with pytest.raises(FW.FormalCommitFreezeError):
        FW.validate_formal_commit_freeze(bad, auth)


def test_commit_freeze_rejects_inconsistent_with_auth(tmp_path):
    auth = load_tv(tmp_path)
    bad = FW.FormalCommitFreeze("9" * 40, auth.generator_commit, auth.runtime_commit,
                                auth.authorization_commit, auth.authorization_evidence_sha256)
    with pytest.raises(FW.FormalCommitFreezeError):
        FW.validate_formal_commit_freeze(bad, auth)


def test_commit_verifiers_with_mock_resolvers(tmp_path):
    # commit does not exist
    with pytest.raises(FW.FormalCommitFreezeError):
        FW.verify_commit_exists("/repo", A1, resolver=lambda root, sha: False)
    FW.verify_commit_exists("/repo", A1, resolver=lambda root, sha: True)      # exists
    # dirty worktree
    with pytest.raises(FW.FormalCommitFreezeError):
        FW.verify_clean_worktree("/repo", resolver=lambda root: False)
    FW.verify_clean_worktree("/repo", resolver=lambda root: True)
    # never auto-substitutes HEAD: HEAD != auth commit and not a descendant -> reject
    with pytest.raises(FW.FormalCommitFreezeError):
        FW.verify_head_or_ancestor_policy("/repo", authorization_commit=AUTHC,
                                          head_resolver=lambda root: "9" * 40,
                                          ancestor_resolver=lambda root, a, h: False)
    # HEAD == auth commit -> ok
    FW.verify_head_or_ancestor_policy("/repo", authorization_commit=AUTHC, head_resolver=lambda root: AUTHC,
                                      ancestor_resolver=lambda root, a, h: False)
    # HEAD is a descendant -> ok
    FW.verify_head_or_ancestor_policy("/repo", authorization_commit=AUTHC, head_resolver=lambda root: "9" * 40,
                                      ancestor_resolver=lambda root, a, h: True)


# =============================================================== §17.4 formal env
def test_formal_env_exact_contract_and_immutable():
    env = formal_env()
    assert set(env.values) == set(FW.FORMAL_ENV_VERSION_KEYS)
    assert env.values["execution_mode"] == "FORMAL_MANIFEST_GENERATION"
    assert env.values["gpu_status"] == "NOT_USED_MANIFEST_GENERATION"
    with pytest.raises(AttributeError):
        env._canonical_json = "x"


def test_formal_env_rejects_bad():
    base = formal_env().values
    for bad in (dict(base, execution_mode="SMOKE_ONLY"),                       # smoke marker
                {k: base[k] for k in list(base)[:9]},                          # partial (missing 1)
                dict(base, extra="1"),                                         # extra key
                dict(base, machine=""),                                        # empty value
                dict(base, os_release=True),                                   # bool value
                dict(base, gpu_status="NOT_USED_CPU_SMOKE")):                  # wrong fixed marker
        with pytest.raises(FW.FormalEnvironmentError):
            FW.FormalManifestEnvironmentContext(bad)


# =============================================================== §17.5 test early access (build NOT called)
def test_test_early_access_missing_step5(monkeypatch):
    monkeypatch.setattr(FW.GEN, "_build_unsealed_manifest",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("build must not run")))
    auth = mint(mk_test_auth_raw(), phase="test")            # token auth, NO step-5 prerequisites provided
    with pytest.raises(FW.EarlyTestAccessError):
        FW.prepare_authorized_formal_phase_in_memory(
            authorization=auth, commits=good_freeze(auth), environment_versions=formal_env())


def test_test_early_access_wrong_verdict(monkeypatch):
    monkeypatch.setattr(FW.GEN, "_build_unsealed_manifest",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("build must not run")))
    raw = mk_test_auth_raw(verdict=FW.TRAIN_VALIDATION_VERDICT)     # test phase but train_validation verdict
    auth = mint(raw, phase="test", m_freeze="b" * 64, tv_manifest="c" * 64, analysis_code="d" * 64)
    with pytest.raises(FW.EarlyTestAccessError):
        FW.prepare_authorized_formal_phase_in_memory(
            authorization=auth, commits=good_freeze(auth), environment_versions=formal_env())


def test_test_early_access_wrong_output_path(monkeypatch):
    monkeypatch.setattr(FW.GEN, "_build_unsealed_manifest",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("build must not run")))
    raw = mk_test_auth_raw(authorized_output_relpath="artifacts/formal/confirmatory_v4/train_validation")
    auth = mint(raw, phase="test", m_freeze="b" * 64, tv_manifest="c" * 64, analysis_code="d" * 64)
    with pytest.raises(FW.EarlyTestAccessError):
        FW.prepare_authorized_formal_phase_in_memory(
            authorization=auth, commits=good_freeze(auth), environment_versions=formal_env())


def test_hand_constructed_authorization_without_token_refused():
    with pytest.raises(FW.FormalAuthorizationIntegrityError):
        FW.VerifiedFormalManifestAuthorization(
            authorization_verdict=FW.TRAIN_VALIDATION_VERDICT, authorization_evidence_sha256=H64,
            authorization_commit=AUTHC, audited_writer_commit=A1, authorized_phase="train_validation",
            protocol_commit=PROTO, generator_commit=A1, runtime_commit=RUNTIME,
            authorized_output_relpath="artifacts/formal/confirmatory_v4/train_validation",
            authorizes_formal_manifest_generation=True, authorizes_confirmatory_data_generation=False,
            authorizes_confirmatory_run=False, one_shot=True,
            test_prerequisite_model_analysis_freeze_sha256=None,
            test_prerequisite_train_validation_manifest_sha256=None,
            test_prerequisite_analysis_code_sha256=None,
            _raw_authorization_canonical_json="{}", _verified_loader_version=FW.VERIFIED_LOADER_VERSION)


# =============================================================== §17.6 stamp pure function (TEST_ONLY payload)
def test_stamp_pure_function_test_only_payload():
    payload = {"marker": TEST_ONLY, "phase": "train_validation", "value": 3, "list": [1, 2, 3]}
    stamped, full, sjson, sfile = FW._stamp_manifest(payload, validate=lambda m: None, full_hash=lambda m: H64)
    assert stamped[MI.SELF_HASH_FIELD] == H64                       # flat self-hash field stamped
    assert full == H64                                             # scientific anchor
    assert sfile == hashlib.sha256(sjson.encode()).hexdigest()     # stamped FILE hash
    assert sfile != full                                          # file hash is separate from full hash
    # deterministic canonical bytes
    _, _, sjson2, _ = FW._stamp_manifest(payload, validate=lambda m: None, full_hash=lambda m: H64)
    assert sjson == sjson2
    # the original payload is not mutated by stamping
    assert MI.SELF_HASH_FIELD not in payload


def test_stamp_rejects_non_finite():
    payload = {"marker": TEST_ONLY, "bad": float("nan")}
    with pytest.raises(ValueError):
        FW._stamp_manifest(payload, validate=lambda m: None, full_hash=lambda m: H64)


# =============================================================== §17.7 atomic bundle helper + path safety
def _test_only_files():
    body = json.dumps({"marker": TEST_ONLY}).encode("utf-8")
    return {"phase_manifest.json": body,
            "phase_manifest.sha256": (hashlib.sha256(body).hexdigest() + "  phase_manifest.json\n").encode(),
            "write_receipt.json": b'{"marker":"TEST_ONLY_NON_SCIENTIFIC_PAYLOAD"}',
            "SEALED": b'{"write_complete":true}'}


def test_atomic_bundle_write_readback_and_one_shot(tmp_path):
    final = tmp_path / "artifacts/formal/confirmatory_v4/train_validation"
    files = _test_only_files()
    FW._atomic_write_bundle(final, files)
    assert sorted(p.name for p in final.iterdir()) == sorted(files)
    for name, data in files.items():
        assert (final / name).read_bytes() == data
    # one-shot: a second write is refused even with identical content
    with pytest.raises(FW.FormalManifestAlreadyExists):
        FW._atomic_write_bundle(final, files)
    # existing bundle untouched; no staging residue in parent
    assert (final / "SEALED").read_bytes() == files["SEALED"]
    assert not any(p.name.startswith(".formal_staging_") for p in (tmp_path / "artifacts/formal/confirmatory_v4").iterdir())


def test_atomic_bundle_failure_leaves_no_final_dir(tmp_path, monkeypatch):
    final = tmp_path / "artifacts/formal/confirmatory_v4/train_validation"
    monkeypatch.setattr(FW.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        FW._atomic_write_bundle(final, _test_only_files())
    assert not final.exists()
    parent = tmp_path / "artifacts/formal/confirmatory_v4"
    assert not any(p.name.startswith(".formal_staging_") for p in parent.iterdir()) if parent.exists() else True


def test_path_safety_rejects_traversal_and_absolute(tmp_path):
    for bad in ("/tmp/x", "../../x", "artifacts/formal/confirmatory_v4/../../../x"):
        with pytest.raises(FW.FormalPathSafetyError):
            FW._resolve_contained_final_dir(tmp_path, bad, "train_validation")
    # phase / directory mismatch
    with pytest.raises(FW.FormalPathSafetyError):
        FW._resolve_contained_final_dir(tmp_path, "artifacts/formal/confirmatory_v4/test", "train_validation")


def test_path_safety_rejects_symlink_escape(tmp_path):
    repo = tmp_path / "repo"
    (repo / "artifacts/formal").mkdir(parents=True)
    outside = tmp_path / "evil"
    outside.mkdir()
    os.symlink(outside, repo / "artifacts/formal/confirmatory_v4")     # subroot symlinked outside repo
    with pytest.raises(FW.FormalPathSafetyError):
        FW._resolve_contained_final_dir(repo, "artifacts/formal/confirmatory_v4/train_validation", "train_validation")


def test_path_safety_accepts_clean_relpath(tmp_path):
    final = FW._resolve_contained_final_dir(tmp_path, "artifacts/formal/confirmatory_v4/train_validation",
                                            "train_validation")
    assert final.name == "train_validation"
    assert not final.exists()
