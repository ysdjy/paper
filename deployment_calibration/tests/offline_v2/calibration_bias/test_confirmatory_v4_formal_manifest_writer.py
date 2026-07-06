"""Tests for the confirmatory v4 FORMAL manifest writer, updated for the HARDENED git-pinned interface
(FW-B-001..014). The exhaustive adversarial coverage lives in test_confirmatory_v4_formal_writer_fw_b_fix.py;
this file keeps the durable behavioural checks (formal env contract, path safety, stamping, CLI preflight,
current-C-gate lock) under the safe interface. Offline; no Isaac; no formal artifact under the repo tree.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_formal_manifest_writer as FW
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_formal_writer_cli as CLI
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI

SUB = FW.FORMAL_ARTIFACT_SUBROOT
TEST_ONLY = "TEST_ONLY_NON_SCIENTIFIC_PAYLOAD"


def full_formal_env():
    return {"python_implementation": "CPython", "python_version": "3.11.0", "numpy_version": "1.26.0",
            "torch_version": "2.7.0+cu128", "os_system": "Linux", "os_release": "x", "machine": "x86_64",
            "isaac_status": "NOT_IMPORTED_NOT_LAUNCHED", "gpu_status": "NOT_USED_MANIFEST_GENERATION",
            "execution_mode": "FORMAL_MANIFEST_GENERATION"}


# ------------------------------------------------------------------ formal environment contract
def test_formal_env_exact_10_key_and_markers():
    env = FW.FormalManifestEnvironmentContext(full_formal_env())
    assert set(env.values) == set(FW.FORMAL_ENV_VERSION_KEYS)
    assert env.values["execution_mode"] == "FORMAL_MANIFEST_GENERATION"
    assert env.values["gpu_status"] == "NOT_USED_MANIFEST_GENERATION"
    with pytest.raises(AttributeError):
        env._canonical_json = "x"


@pytest.mark.parametrize("mut", [
    lambda d: d.__setitem__("execution_mode", "SMOKE_ONLY"),
    lambda d: d.pop("machine"),
    lambda d: d.__setitem__("extra", "x"),
    lambda d: d.__setitem__("os_release", ""),
    lambda d: d.__setitem__("python_version", True),
    lambda d: d.__setitem__("gpu_status", "NOT_USED_CPU_SMOKE"),
])
def test_formal_env_rejects_bad(mut):
    d = full_formal_env()
    mut(d)
    with pytest.raises(FW.FormalEnvironmentError):
        FW.FormalManifestEnvironmentContext(d)


# ------------------------------------------------------------------ path safety
def test_path_safety_rejects_absolute_traversal_phase_mismatch(tmp_path):
    for bad in ("/abs/path", f"{SUB}/../../etc", f"{SUB}/test"):
        with pytest.raises(FW.FormalPathSafetyError):
            FW._resolve_contained_final_dir(tmp_path, bad, "train_validation")


def test_path_safety_accepts_clean_relpath(tmp_path):
    final = FW._resolve_contained_final_dir(tmp_path, f"{SUB}/train_validation", "train_validation")
    assert final.name == "train_validation" and not final.exists()


# ------------------------------------------------------------------ stamp hash layering
def test_stamp_pure_function_hash_layering():
    payload = {"marker": TEST_ONLY, "value": 3}
    h64 = "a" * 64
    stamped, full, sjson, sfile = FW._stamp_manifest(payload, validate=lambda m: None, full_hash=lambda m: h64)
    assert stamped[MI.SELF_HASH_FIELD] == h64 and full == h64          # scientific anchor / flat self field
    assert sfile == hashlib.sha256(sjson.encode()).hexdigest() and sfile != full   # file hash is separate
    assert MI.SELF_HASH_FIELD not in payload                          # original not mutated


def test_stamp_rejects_non_finite():
    with pytest.raises(ValueError):
        FW._stamp_manifest({"marker": TEST_ONLY, "bad": float("nan")}, validate=lambda m: None,
                           full_hash=lambda m: "a" * 64)


# ------------------------------------------------------------------ TEST_ONLY atomic bundle
def _test_only_files():
    body = json.dumps({"marker": TEST_ONLY}).encode()
    return {"phase_manifest.json": body,
            "phase_manifest.sha256": (hashlib.sha256(body).hexdigest() + "  phase_manifest.json\n").encode(),
            "write_receipt.json": b'{"marker":"TEST_ONLY_NON_SCIENTIFIC_PAYLOAD"}',
            "SEALED": b'{"write_complete":true}', "bundle_index.json": b'{"files":{}}'}


def test_atomic_write_and_one_shot(tmp_path):
    final = tmp_path / SUB / "train_validation"
    files = _test_only_files()
    FW._test_only_atomic_write_bundle(final, files)
    assert set(p.name for p in final.iterdir()) == set(files)
    with pytest.raises(FW.FormalManifestAlreadyExists):
        FW._test_only_atomic_write_bundle(final, files)


# ------------------------------------------------------------------ current C gate lock + CLI preflight
def test_public_local_loader_is_locked_and_current_gate_rejected():
    root = Path(FW.__file__).resolve().parents[3]
    gate = root / "docs/offline_v2/calibration_bias/confirmatory_v4_generator_smoke_final_audit_c.json"
    with pytest.raises((FW.FormalManifestGenerationNotAuthorized, FW.FormalAuthorizationIntegrityError)):
        FW.load_and_verify_formal_manifest_authorization(str(gate), authorization_commit="0" * 40,
                                                         expected_writer_commit="0" * 40)


def test_cli_preflight_reports_generation_locked(capsys):
    rc = CLI.main(["--preflight-only"])
    out = capsys.readouterr().out
    assert rc == 0 and "FORMAL_WRITER_IMPLEMENTED_GENERATION_LOCKED" in out and "--force" not in out


def test_cli_refuses_without_preflight_flag(capsys):
    assert CLI.main([]) == 2
    assert "FORMAL_WRITER_IMPLEMENTED_GENERATION_LOCKED" not in capsys.readouterr().out


def test_no_formal_artifacts_dir_in_repo():
    root = Path(FW.__file__).resolve().parents[3]
    assert not (root / "artifacts" / "formal").exists()
