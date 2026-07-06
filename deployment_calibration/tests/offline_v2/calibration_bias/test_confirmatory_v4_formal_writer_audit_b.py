"""Claude B one-shot, read-only, adversarial audit of the confirmatory v4 FORMAL manifest writer + real
commit-freeze interface (A1 anchor e1a5a67, A2 report 0f28df9).

PASS-region tests confirm the parts that ARE correct (strict formal-env context, exact commit-freeze match,
path-safety rejections, current-C-gate rejection + preflight-only CLI, HEAD/ancestor resolver wiring, baseline
regression classification). The `test_FW_B_00x_*` tests DEMONSTRATE the frozen findings on the CURRENT code:
they assert the present (vulnerable) behaviour so a future A fix flips them.

Constraints honoured: no real 228/72 formal phase is constructed (TEST_ONLY payloads via monkeypatch), nothing
is written under the repo `artifacts/formal` tree (only pytest tmp_path), no Isaac, no formal artifact. Do NOT
modify A's files.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import json
import os
import subprocess
from pathlib import Path

import pytest

from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_formal_manifest_writer as FW
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_formal_writer_cli as CLI
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_generator as GEN
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI

A1_WRITER_COMMIT = "e1a5a67bc10110b157798b028480116eb0627582"
A2_REPORT_COMMIT = "0f28df9499d46efdb1965e7b528c2775e1646644"
SOURCE_C_GATE_COMMIT = "f41e36620b721455263fb8489bc906096a2f7442"
SUB = FW.FORMAL_ARTIFACT_SUBROOT
WC = "e" * 40


def _canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha_t(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def _full_env_values():
    return {"python_implementation": "CPython", "python_version": "3.10.0", "numpy_version": "1.26.0",
            "torch_version": "2.3.0", "os_system": "Linux", "os_release": "6.8.0", "machine": "x86_64",
            "isaac_status": "NOT_IMPORTED_NOT_LAUNCHED", "gpu_status": "NOT_USED_MANIFEST_GENERATION",
            "execution_mode": "FORMAL_MANIFEST_GENERATION"}


def _auth_dict(phase="train_validation", extra=None, wc=WC):
    raw = {"verdict": FW.TEST_VERDICT if phase == "test" else FW.TRAIN_VALIDATION_VERDICT,
           "authorized_phase": phase, "authorizes_formal_manifest_generation": True,
           "authorizes_confirmatory_data_generation": False, "authorizes_confirmatory_run": False,
           "one_shot": True, "power_recertification_required": False,
           "authorized_output_relpath": f"{SUB}/{phase}", "audited_writer_commit": wc,
           "protocol_commit": "1" * 40, "generator_commit": wc, "runtime_commit": "3" * 40}
    if extra:
        raw.update(extra)
    return raw


def _mint_from_tmp(tmp_path, phase="train_validation", extra=None, authorization_commit="f" * 40):
    p = tmp_path / "auth.json"
    p.write_text(json.dumps(_auth_dict(phase, extra)))
    return FW.load_and_verify_formal_manifest_authorization(
        str(p), authorization_commit=authorization_commit, expected_writer_commit=WC)


def _test_only_stamp():
    """A monkeypatch stub for FW._stamp_manifest that never touches the real MI validator/hasher."""
    stamped = {MI.SELF_HASH_FIELD: "d" * 64, "TEST_ONLY": True}
    sj = _canon(stamped)
    return lambda manifest: (stamped, "d" * 64, sj, _sha_t(sj))


# ============================================================ PASS region (verified-correct behaviour)
def test_anchor_and_diff_integrity():
    """§2: A1 = writer+cli+test+doc additions; A2 = report json only; A2 does not touch A1 writer source."""
    root = Path(FW.__file__).resolve().parents[3]

    def files(a, b):
        out = subprocess.run(["git", "-C", str(root), "diff", "--name-only", a, b],
                             capture_output=True, text=True)
        return set(x for x in out.stdout.split("\n") if x)

    a1 = files(SOURCE_C_GATE_COMMIT, A1_WRITER_COMMIT)
    a2 = files(A1_WRITER_COMMIT, A2_REPORT_COMMIT)
    assert any("confirmatory_v4_formal_manifest_writer.py" in f for f in a1)
    assert a2 == {"docs/offline_v2/calibration_bias/confirmatory_v4_formal_writer_implementation_report.json"}
    # A2 must not modify A1 writer source or tests
    assert not any("confirmatory_v4_formal_manifest_writer.py" in f for f in a2)
    assert not any("test_confirmatory_v4_formal_manifest_writer.py" in f for f in a2)


def test_formal_environment_context_is_strict():
    good = FW.FormalManifestEnvironmentContext(_full_env_values())
    assert good.values["execution_mode"] == "FORMAL_MANIFEST_GENERATION"
    assert good.values is not good.values                       # copy-on-read
    with pytest.raises(AttributeError):
        good.foo = 1                                            # immutable
    # partial / extra / empty / bool / SMOKE_ONLY marker are all rejected
    for bad in ({k: v for k, v in _full_env_values().items() if k != "machine"},
                {**_full_env_values(), "extra": "x"}, {},
                {**_full_env_values(), "python_version": True}):
        with pytest.raises(FW.FormalEnvironmentError):
            FW.FormalManifestEnvironmentContext(bad)
    smoke = {**_full_env_values(), "execution_mode": "SMOKE_ONLY"}
    with pytest.raises(FW.FormalEnvironmentError):
        FW.FormalManifestEnvironmentContext(smoke)


def test_commit_freeze_exact_match_and_smoke_rejected(tmp_path):
    auth = _mint_from_tmp(tmp_path)
    ok = FW.FormalCommitFreeze("1" * 40, WC, "3" * 40, "f" * 40, auth.authorization_evidence_sha256)
    FW.validate_formal_commit_freeze(ok, auth)                  # exact agreement passes
    bad = FW.FormalCommitFreeze("9" * 40, WC, "3" * 40, "f" * 40, auth.authorization_evidence_sha256)
    with pytest.raises(FW.FormalCommitFreezeError):
        FW.validate_formal_commit_freeze(bad, auth)             # protocol mismatch
    smoke = FW.FormalCommitFreeze("a" * 40, "b" * 40, "c" * 40, "f" * 40, auth.authorization_evidence_sha256)
    with pytest.raises(FW.FormalCommitFreezeError):
        FW.validate_formal_commit_freeze(smoke, auth)           # placeholder rejected


def test_path_safety_rejects_absolute_traversal_and_phase_mismatch():
    with pytest.raises(FW.FormalPathSafetyError):
        FW._resolve_contained_final_dir("/tmp/x", "/abs/path", "train_validation")
    with pytest.raises(FW.FormalPathSafetyError):
        FW._resolve_contained_final_dir("/tmp/x", f"{SUB}/../../etc", "train_validation")
    with pytest.raises(FW.FormalPathSafetyError):
        FW._resolve_contained_final_dir("/tmp/x", f"{SUB}/test", "train_validation")  # phase mismatch


def test_current_c_gate_rejected_and_cli_is_preflight_only():
    root = Path(FW.__file__).resolve().parents[3]
    gate = root / "docs/offline_v2/calibration_bias/confirmatory_v4_generator_smoke_final_audit_c.json"
    with pytest.raises((FW.FormalManifestGenerationNotAuthorized, FW.FormalAuthorizationIntegrityError)):
        FW.load_and_verify_formal_manifest_authorization(str(gate), authorization_commit="0" * 40,
                                                         expected_writer_commit="0" * 40)
    csrc = inspect.getsource(CLI)
    for flag in ('"--force"', '"--write"', '"--phase"', '"--formal"', '"--output"'):
        assert f"add_argument({flag}" not in csrc              # no dangerous CLI option is defined
    assert "write_authorized_formal_manifest_bundle_atomic" not in csrc   # CLI cannot write a bundle


def test_head_ancestor_policy_resolver_wiring():
    # HEAD == auth commit -> pass; unrelated HEAD (not descendant) -> reject
    FW.verify_head_or_ancestor_policy("/x", authorization_commit="a" * 40,
                                      head_resolver=lambda r: "a" * 40)
    with pytest.raises(FW.FormalCommitFreezeError):
        FW.verify_head_or_ancestor_policy("/x", authorization_commit="a" * 40,
                                          head_resolver=lambda r: "b" * 40,
                                          ancestor_resolver=lambda r, a, h: False)


def test_baseline_failure_classification_no_writer_regression():
    """The 34 full-suite failures = 25 historical-C reaudit + 9 audit-b GEN-fix flips; NONE imports the
    formal writer, so the writer introduces zero active regressions."""
    tdir = Path(__file__).parent
    historical_c = ["test_block_state_sampler_reaudit_c.py", "test_final_001_002_regression_c.py",
                    "test_frozen_issue_regression_final_001_005_c.py", "test_one_shot_full_audit_c.py",
                    "test_preregistration_v4_audit_c.py", "test_preregistration_v4_fix1_reaudit_c.py",
                    "test_preregistration_v4_fix2_final_reaudit_c.py",
                    "test_preregistration_v4_fix3_final_reaudit_c.py"]
    audit_b_flips = ["test_confirmatory_v4_generator_audit_b.py"]
    for f in historical_c + audit_b_flips:
        assert (tdir / f).exists()
        assert "confirmatory_v4_formal" not in (tdir / f).read_text()   # no failing file touches the writer


# ============================================================ FROZEN FINDINGS (demonstrated on current code)
def test_FW_B_001_arbitrary_local_json_mints_authorization(tmp_path):
    """§5 BLOCKING: any local file with correct fields + a decorative caller-supplied 40-hex
    authorization_commit mints a VerifiedFormalManifestAuthorization. No binding to a git-pinned in-repo
    authorization artifact; the evidence hash is self-referential over the file's own bytes."""
    auth = _mint_from_tmp(tmp_path, authorization_commit="f" * 40)   # f*40 never resolved against git
    assert isinstance(auth, FW.VerifiedFormalManifestAuthorization)
    assert auth.authorizes_formal_manifest_generation is True
    assert auth.authorization_commit == "f" * 40                     # accepted without existence/content check


def test_FW_B_002_duplicate_json_key_last_wins_mints(tmp_path):
    """§6 BLOCKING: json.load has no object_pairs_hook; a duplicate key (false then true) silently mints."""
    body = ('{"verdict":"%s","authorized_phase":"train_validation",'
            '"authorizes_formal_manifest_generation":false,'
            '"authorizes_formal_manifest_generation":true,'
            '"authorizes_confirmatory_data_generation":false,"authorizes_confirmatory_run":false,'
            '"one_shot":true,"power_recertification_required":false,'
            '"authorized_output_relpath":"%s/train_validation","audited_writer_commit":"%s",'
            '"protocol_commit":"%s","generator_commit":"%s","runtime_commit":"%s"}'
            % (FW.TRAIN_VALIDATION_VERDICT, SUB, WC, "1" * 40, WC, "3" * 40))
    p = tmp_path / "dup.json"
    p.write_text(body)
    auth = FW.load_and_verify_formal_manifest_authorization(str(p), authorization_commit="f" * 40,
                                                            expected_writer_commit=WC)
    assert auth.authorizes_formal_manifest_generation is True        # last-key-wins accepted


def test_FW_B_003_hand_built_authorization_via_public_token_passes_reverify():
    """§7 BLOCKING: _AUTHORIZATION_LOADER_TOKEN is a public module attribute; a hand-built object with a
    self-consistent evidence hash passes _reverify_authorization_integrity (token is treated as trust root)."""
    raw = _auth_dict()
    cj = _canon(raw)
    hb = FW.VerifiedFormalManifestAuthorization(
        authorization_verdict=raw["verdict"], authorization_evidence_sha256=_sha_t(cj),
        authorization_commit="a" * 40, audited_writer_commit=WC, authorized_phase="train_validation",
        protocol_commit="1" * 40, generator_commit=WC, runtime_commit="3" * 40,
        authorized_output_relpath=raw["authorized_output_relpath"],
        authorizes_formal_manifest_generation=True, authorizes_confirmatory_data_generation=False,
        authorizes_confirmatory_run=False, one_shot=True,
        test_prerequisite_model_analysis_freeze_sha256=None,
        test_prerequisite_train_validation_manifest_sha256=None,
        test_prerequisite_analysis_code_sha256=None, _raw_authorization_canonical_json=cj,
        _verified_loader_version=FW.VERIFIED_LOADER_VERSION, _loader_token=FW._AUTHORIZATION_LOADER_TOKEN)
    FW._reverify_authorization_integrity(hb)                         # no exception -> accepted


def test_FW_B_004_replace_mutation_uncrosschecked_passes_reverify(tmp_path):
    """§8 BLOCKING: _reverify only cross-checks verdict + 3 bools + one_shot vs raw JSON; a dataclasses.replace
    that mutates audited_writer_commit/protocol/runtime (raw JSON unchanged) is not detected."""
    auth = _mint_from_tmp(tmp_path)
    mutated = dataclasses.replace(auth, audited_writer_commit="9" * 40, protocol_commit="7" * 40,
                                  runtime_commit="8" * 40)
    FW._reverify_authorization_integrity(mutated)                   # accepted despite raw/field divergence
    assert json.loads(mutated._raw_authorization_canonical_json)["protocol_commit"] == "1" * 40
    assert mutated.protocol_commit == "7" * 40                       # object diverges from evidence-bound raw


def test_FW_B_005_repo_root_none_skips_all_git_verification(tmp_path, monkeypatch):
    """§9 BLOCKING: repo_root=None (the default) skips commit-existence / clean-tree / HEAD-ancestor checks."""
    auth = _mint_from_tmp(tmp_path)
    env = FW.FormalManifestEnvironmentContext(_full_env_values())
    freeze = FW.FormalCommitFreeze("1" * 40, WC, "3" * 40, "f" * 40, auth.authorization_evidence_sha256)
    monkeypatch.setattr(GEN, "_build_unsealed_manifest",
                        lambda ph, cc, ev: {"TEST_ONLY": True, "environment_versions": env.values})
    monkeypatch.setattr(FW, "_stamp_manifest", _test_only_stamp())
    # if git verification were mandatory, a bogus repo/commit would fail; repo_root=None bypasses it entirely
    prepared = FW.prepare_authorized_formal_phase_in_memory(
        authorization=auth, commits=freeze, environment_versions=env, repo_root=None)
    assert prepared.phase == "train_validation"                     # built with zero repo checks


def test_FW_B_006_any_descendant_accepted_and_no_writer_blob_pinning():
    """§10 BLOCKING: verify_head_or_ancestor_policy accepts ANY descendant of the auth commit, and nothing
    compares the currently-imported writer/generator source bytes to the A1 blob."""
    # a modified successor commit (HEAD) that is merely a descendant is accepted
    FW.verify_head_or_ancestor_policy("/x", authorization_commit="a" * 40,
                                      head_resolver=lambda r: "b" * 40,
                                      ancestor_resolver=lambda r, a, h: True)   # descendant -> accepted
    src = inspect.getsource(FW)
    assert "cat-file" in src and "blob" not in src                  # only commit existence, never a blob pin
    assert "getsource" not in src                                   # module never hashes its own source


def test_FW_B_007_fake_step5_hashes_build_test_phase(tmp_path, monkeypatch):
    """§12 BLOCKING: the test-phase Step-5 gate only checks the 3 prerequisite hashes are 64-hex FORMAT;
    fabricated a*64/b*64/c*64 with zero Step-5 artifacts on disk build a test formal phase."""
    auth = _mint_from_tmp(tmp_path, phase="test",
                          extra={"model_analysis_freeze_sha256": "a" * 64,
                                 "train_validation_manifest_sha256": "b" * 64,
                                 "analysis_code_sha256": "c" * 64})
    env = FW.FormalManifestEnvironmentContext(_full_env_values())
    freeze = FW.FormalCommitFreeze("1" * 40, WC, "3" * 40, "f" * 40, auth.authorization_evidence_sha256)
    monkeypatch.setattr(GEN, "_build_unsealed_manifest",
                        lambda ph, cc, ev: {"TEST_ONLY": True, "environment_versions": env.values})
    monkeypatch.setattr(FW, "_stamp_manifest", _test_only_stamp())
    prepared = FW.prepare_authorized_formal_phase_in_memory(
        authorization=auth, commits=freeze, environment_versions=env, repo_root=None)
    assert prepared.phase == "test"                                 # no EarlyTestAccessError; no artifact checked


def test_FW_B_008_and_009_hand_built_prepared_garbage_written_under_arbitrary_root(tmp_path):
    """§14 + §16 BLOCKING: the writer never re-runs MI deep-validation nor recomputes the scientific full hash;
    a hand-built PreparedFormalPhase with a GARBAGE payload + internally-consistent hashes is written as a
    full SEALED bundle, under an arbitrary output_root (no binding to a verified repo root)."""
    auth = _mint_from_tmp(tmp_path)
    garbage = {"totally": "bogus", "not_a_manifest": True}
    stamped = dict(garbage)
    stamped[MI.SELF_HASH_FIELD] = "d" * 64
    sj = _canon(stamped)
    fake = FW.PreparedFormalPhase(
        phase="train_validation", _unstamped_manifest_canonical_json=_canon(garbage),
        _stamped_manifest_canonical_json=sj, full_manifest_sha256="d" * 64, stamped_file_sha256=_sha_t(sj),
        authorization_evidence_sha256=auth.authorization_evidence_sha256, protocol_commit="1" * 40,
        generator_commit=WC, runtime_commit="3" * 40)
    outroot = tmp_path / "arbitrary_root"                           # NOT the repo root
    outroot.mkdir()
    receipt = FW.write_authorized_formal_manifest_bundle_atomic(fake, authorization=auth, output_root=str(outroot))
    final = Path(receipt.final_dir)
    assert final.exists() and set(p.name for p in final.iterdir()) == set(FW.BUNDLE_FILES)
    written = json.loads((final / "phase_manifest.json").read_text())
    assert written["not_a_manifest"] is True                        # garbage accepted as a formal manifest
    assert str(outroot) in receipt.final_dir                        # wrote under an arbitrary root


def test_FW_B_010_post_rename_failure_leaves_final_dir(tmp_path, monkeypatch):
    """§17.2 BLOCKING: a failure AFTER os.replace (parent fsync / final read-back) raises to the caller while
    the fully-populated final bundle dir remains -> ambiguous commit, future FORMAL_MANIFEST_ALREADY_EXISTS."""
    auth = _mint_from_tmp(tmp_path)
    garbage = {"x": 1}
    stamped = dict(garbage)
    stamped[MI.SELF_HASH_FIELD] = "d" * 64
    sj = _canon(stamped)
    prepared = FW.PreparedFormalPhase("train_validation", _canon(garbage), sj, "d" * 64, _sha_t(sj),
                                      auth.authorization_evidence_sha256, "1" * 40, WC, "3" * 40)
    outroot = tmp_path / "root2"
    outroot.mkdir()
    orig = FW._fsync_path
    state = {"n": 0}

    def boom(path):
        state["n"] += 1
        if state["n"] >= 2:                                         # 1st = staging (pre-rename); 2nd = parent (post)
            raise OSError("simulated post-rename parent fsync failure")
        return orig(path)

    monkeypatch.setattr(FW, "_fsync_path", boom)
    final = outroot / f"{SUB}/train_validation"
    with pytest.raises(OSError):
        FW.write_authorized_formal_manifest_bundle_atomic(prepared, authorization=auth, output_root=str(outroot))
    assert final.exists()                                           # BLOCKING: caller sees failure, final remains
    assert set(p.name for p in final.iterdir()) == set(FW.BUNDLE_FILES)


def test_FW_B_011_os_replace_can_replace_empty_competitor(tmp_path):
    """§17.3 BLOCKING: the one-shot claim rests on non-atomic exists() checks + os.replace, which replaces an
    EMPTY competitor directory (not a no-replace exclusive claim)."""
    src = inspect.getsource(FW._atomic_write_bundle)
    assert "os.replace" in src                                      # rename-based, not exclusive-create
    assert "O_EXCL" not in src and "RENAME_NOREPLACE" not in src and "flock" not in src and "mkdir(str(final" not in src
    # empirical: os.replace onto an empty dir succeeds (replaces the competitor)
    s = tmp_path / "s"
    s.mkdir()
    (s / "f").write_text("x")
    t = tmp_path / "t"
    t.mkdir()                                                       # empty competitor final dir
    os.replace(str(s), str(t))                                      # succeeds -> competitor replaced
    assert (t / "f").exists()


def test_FW_B_012_parent_dir_residue_after_failure_nonblocking(tmp_path, monkeypatch):
    """§17.1 NONBLOCKING: a pre-rename failure leaves the empty parent chain artifacts/formal/confirmatory_v4/
    behind (no final phase dir, no staging) -> empty residue, does not poison the one-shot lock."""
    auth = _mint_from_tmp(tmp_path)
    stamped = {MI.SELF_HASH_FIELD: "d" * 64}
    sj = _canon(stamped)
    prepared = FW.PreparedFormalPhase("train_validation", _canon({}), sj, "d" * 64, _sha_t(sj),
                                      auth.authorization_evidence_sha256, "1" * 40, WC, "3" * 40)
    outroot = tmp_path / "root3"
    outroot.mkdir()
    monkeypatch.setattr(FW, "_bundle_payloads", lambda prepared, a: {"phase_manifest.json": "NOT_BYTES"})
    with pytest.raises(FW.FormalBundleWriteError):
        FW.write_authorized_formal_manifest_bundle_atomic(prepared, authorization=auth, output_root=str(outroot))
    chain = outroot / "artifacts" / "formal" / "confirmatory_v4"
    assert chain.exists()                                           # empty parent chain left behind
    assert not (chain / "train_validation").exists()               # but no final phase dir (one-shot intact)


def test_FW_B_013_writer_exceptions_lack_verdict_attribute():
    """§19 NONBLOCKING: writer exception classes carry no `.verdict`, unlike the project convention (GEN/MI
    exceptions expose a class-level verdict), so machine gates classifying by `.verdict` cannot classify them."""
    for exc in (FW.FormalManifestGenerationNotAuthorized, FW.FormalAuthorizationIntegrityError,
                FW.FormalCommitFreezeError, FW.EarlyTestAccessError, FW.FormalPathSafetyError,
                FW.FormalManifestAlreadyExists, FW.FormalBundleWriteError):
        assert not hasattr(exc, "verdict")
    # the surrounding convention DOES expose .verdict
    assert getattr(GEN.FormalGenerationNotAuthorized, "verdict", None) is not None


def test_FW_B_014_receipt_and_sealed_have_no_selfhash():
    """§18 NONBLOCKING: write_receipt.json / SEALED carry no self-verifiable hash or signature, so they can be
    edited post-write undetectably (subsumed once payload validation is fixed)."""
    src = inspect.getsource(FW._bundle_payloads)
    assert "receipt_sha256" not in src and "sealed_sha256" not in src and "signature" not in src
