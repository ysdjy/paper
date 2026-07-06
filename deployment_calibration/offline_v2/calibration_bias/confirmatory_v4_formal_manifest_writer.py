"""Confirmatory v4 FORMAL manifest writer + real commit-freeze interface (offline; no Isaac, no GPU).

HARDENED after Claude B's adversarial audit (FW-B-001..014). The production trust root is NOT a local JSON
path, a private Python token, or a caller boolean — it is a **real git commit** plus the committed bytes of a
fixed allowlisted authorization artifact that pins the hardened writer commit (F1), verified against the actual
repository on every production entry:

  authorization = git show <authorization_commit>:<fixed allowlist relpath>
                  + the artifact pins F1 (audited_writer_commit == generator_commit == writer_implementation_commit)
                  + F1 is an ancestor of the authorization commit
                  + HEAD == authorization_commit, clean worktree
                  + the writer/generator/science-module blobs are byte-identical across F1, the authorization
                    commit, and the working tree (no code drift under a clean tree)

A `VerifiedFormalManifestAuthorization` can only be produced by `load_and_verify_formal_manifest_authorization_
from_git`, and every production entry re-verifies it against git (`reverify_authorization_against_git`), so a
hand-built object, a `dataclasses.replace`/`object.__setattr__` mutation, an arbitrary/duplicate-key JSON, a
dirty tree, a drifted writer blob, or a changed committed artifact are all rejected. The private
`_AUTHORIZATION_LOADER_TOKEN` is an internal misuse guard ONLY, never a trust root.

Fail-closed: the current repo contains no such authorization artifact, so every production generation entry
fails before constructing a phase and before any filesystem effect. Nothing under `artifacts/formal/` is
created and no Isaac/GPU is used. The public local-path loader is permanently locked.
"""

from __future__ import annotations

import copy
import ctypes
import errno
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_generator as GEN
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_block_state as BS
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI

VERIFIED_LOADER_VERSION = "confirmatory_v4_formal_manifest_authorization_loader_v2_git_pinned"
AUTHORIZATION_SCHEMA_VERSION = "confirmatory_v4_formal_manifest_authorization_v1"
RECEIPT_SCHEMA_VERSION = "confirmatory_v4_formal_manifest_receipt_v1"
BUNDLE_SCHEMA_VERSION = "confirmatory_v4_formal_manifest_bundle_v1"
MODEL_FREEZE_SCHEMA_VERSION = "confirmatory_v4_model_analysis_freeze_v1"
ANALYSIS_CODE_FREEZE_SCHEMA_VERSION = "confirmatory_v4_analysis_code_freeze_v1"

FORMAL_ENV_VERSION_KEYS = ("python_implementation", "python_version", "numpy_version", "torch_version",
                           "os_system", "os_release", "machine", "isaac_status", "gpu_status", "execution_mode")
FORMAL_ENV_FIXED = {"execution_mode": "FORMAL_MANIFEST_GENERATION",
                    "isaac_status": "NOT_IMPORTED_NOT_LAUNCHED",
                    "gpu_status": "NOT_USED_MANIFEST_GENERATION"}

TRAIN_VALIDATION_VERDICT = "AUTHORIZE_TRAIN_VALIDATION_MANIFEST_GENERATION"
TEST_VERDICT = "AUTHORIZE_TEST_MANIFEST_GENERATION"
_VERDICT_PHASE = {TRAIN_VALIDATION_VERDICT: "train_validation", TEST_VERDICT: "test"}
FORMAL_ARTIFACT_SUBROOT = "artifacts/formal/confirmatory_v4"

# FW-B-001: authorization artifacts live at FIXED allowlisted repo relpaths; a caller cannot pass a path.
_AUTH_DIR = "docs/offline_v2/calibration_bias/authorizations"
AUTHORIZATION_RELPATHS = {
    "train_validation": f"{_AUTH_DIR}/confirmatory_v4_train_validation_manifest_authorization.json",
    "test": f"{_AUTH_DIR}/confirmatory_v4_test_manifest_authorization.json",
}

# FW-B-006/008.3: the writer + generator + every imported science module must be byte-identical across F1, the
# authorization commit, and the working tree (clean). Extended docs blobs may be appended in future.
_CB = "deployment_calibration/offline_v2/calibration_bias/"
CRITICAL_BLOB_RELPATHS = (
    _CB + "confirmatory_v4_formal_manifest_writer.py",
    _CB + "confirmatory_v4_formal_writer_cli.py",
    _CB + "confirmatory_v4_generator.py",
    _CB + "confirmatory_v4_identity.py",
    _CB + "confirmatory_v4_block_state.py",
    _CB + "confirmatory_v4_manifest_integrity.py",
    _CB + "preregistration_v4.py",
)

# FW-B-007: Step-5 formal artifacts at fixed relpaths (verified for test-phase generation; not created here).
TRAIN_VALIDATION_BUNDLE_RELPATH = f"{FORMAL_ARTIFACT_SUBROOT}/train_validation"
MODEL_FREEZE_RELPATH = f"{FORMAL_ARTIFACT_SUBROOT}/model_analysis_freeze.json"
ANALYSIS_CODE_FREEZE_RELPATH = f"{FORMAL_ARTIFACT_SUBROOT}/analysis_code_freeze.json"

BUNDLE_FILES = ("phase_manifest.json", "phase_manifest.sha256", "write_receipt.json", "SEALED",
                "bundle_index.json")
_INDEXED_BUNDLE_FILES = ("phase_manifest.json", "phase_manifest.sha256", "write_receipt.json", "SEALED")

_GIT_SHA_LEN = 40
_HASH_HEX_LEN = 64
_SMOKE_PLACEHOLDER = GEN.SMOKE_PLACEHOLDER            # ("a"*40, "b"*40, "c"*40)

# FW-B-003: internal misuse guard ONLY — NOT a trust root. Production always re-verifies against git.
_AUTHORIZATION_LOADER_TOKEN = object()


# ----------------------------------------------------------------------------- errors (FW-B-013: stable verdicts)
class FormalManifestGenerationNotAuthorized(RuntimeError):
    verdict = "FORMAL_MANIFEST_GENERATION_NOT_AUTHORIZED"


class FormalAuthorizationIntegrityError(ValueError):
    verdict = "FORMAL_AUTHORIZATION_INTEGRITY_ERROR"


class FormalCommitFreezeError(ValueError):
    verdict = "FORMAL_COMMIT_FREEZE_ERROR"


class FormalEnvironmentError(ValueError):
    verdict = "FORMAL_ENVIRONMENT_ERROR"


class EarlyTestAccessError(RuntimeError):
    verdict = "EXPERIMENT_INVALID_EARLY_TEST_ACCESS"


class FormalPathSafetyError(ValueError):
    verdict = "FORMAL_PATH_SAFETY_ERROR"


class FormalManifestAlreadyExists(RuntimeError):
    verdict = "FORMAL_MANIFEST_ALREADY_EXISTS"


class FormalBundleWriteError(RuntimeError):
    verdict = "FORMAL_BUNDLE_WRITE_ERROR"


class FormalAtomicNoReplaceUnavailable(RuntimeError):
    verdict = "FORMAL_ATOMIC_NOREPLACE_UNAVAILABLE"


# ----------------------------------------------------------------------------- canonical helpers
def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _is_40hex(s) -> bool:
    return isinstance(s, str) and len(s) == _GIT_SHA_LEN and all(c in "0123456789abcdef" for c in s)


def _is_64hex(s) -> bool:
    return isinstance(s, str) and len(s) == _HASH_HEX_LEN and all(c in "0123456789abcdef" for c in s)


def _strict_bool(v, what: str) -> bool:
    if type(v) is not bool:
        raise FormalAuthorizationIntegrityError(f"{what} must be a JSON boolean, got {type(v).__name__}")
    return v


# FW-B-002: reject duplicate JSON object keys (no last-key-wins control surface).
def _reject_duplicate_object_pairs(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise FormalAuthorizationIntegrityError(f"duplicate JSON key {k!r} is forbidden")
        out[k] = v
    return out


def _strict_json_loads(text: str):
    return json.loads(text, object_pairs_hook=_reject_duplicate_object_pairs)


# ============================================================================= git backend (injectable)
class GitReader:
    """Thin git backend. Production constructs it from the verified repo root; tests inject a TEST_ONLY reader
    (e.g. a real tmp git repo or an in-memory fake). It is a *backend*, never a skip switch."""

    def __init__(self, repo_root):
        self.repo_root = str(Path(repo_root).resolve())

    def _run(self, *args) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", self.repo_root, *args], capture_output=True)

    def is_repo(self) -> bool:
        return self._run("rev-parse", "--git-dir").returncode == 0

    def commit_exists(self, sha) -> bool:
        return _is_40hex(sha) and self._run("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0

    def show_bytes(self, commit, relpath) -> Optional[bytes]:
        res = self._run("show", f"{commit}:{relpath}")
        return res.stdout if res.returncode == 0 else None

    def head(self) -> str:
        return self._run("rev-parse", "HEAD").stdout.decode("utf-8", "replace").strip()

    def is_ancestor(self, ancestor, descendant) -> bool:
        return self._run("merge-base", "--is-ancestor", ancestor, descendant).returncode == 0

    def worktree_clean(self) -> bool:
        res = self._run("status", "--porcelain")
        return res.returncode == 0 and not res.stdout.strip()

    def worktree_bytes(self, relpath) -> Optional[bytes]:
        p = Path(self.repo_root) / relpath
        if p.is_symlink() or not p.is_file():
            return None
        return p.read_bytes()


# ============================================================================= §3 immutable verified objects
@dataclass(frozen=True)
class VerifiedFormalRepositoryContext:
    repo_root_realpath: str
    authorization_commit: str
    authorization_relpath: str
    authorization_blob_sha256: str
    writer_implementation_commit: str
    current_head: str
    verified_critical_blob_map_sha256: str


@dataclass(frozen=True)
class VerifiedFormalManifestAuthorization:
    authorization_schema_version: str
    authorization_verdict: str
    authorization_evidence_sha256: str
    authorization_commit: str
    audited_writer_commit: str
    authorized_phase: Literal["train_validation", "test"]
    protocol_commit: str
    generator_commit: str
    runtime_commit: str
    authorized_output_relpath: str
    authorizes_formal_manifest_generation: bool
    authorizes_confirmatory_data_generation: bool
    authorizes_confirmatory_run: bool
    one_shot: bool
    power_recertification_required: bool
    test_prerequisite_model_analysis_freeze_sha256: Optional[str]
    test_prerequisite_train_validation_manifest_sha256: Optional[str]
    test_prerequisite_analysis_code_sha256: Optional[str]
    repository_context: VerifiedFormalRepositoryContext
    committed_authorization_canonical_json: str
    committed_authorization_blob_sha256: str
    _verified_loader_version: str
    _loader_token: object = field(repr=False, default=None)

    def __post_init__(self):
        if self._loader_token is not _AUTHORIZATION_LOADER_TOKEN:
            raise FormalAuthorizationIntegrityError(
                "VerifiedFormalManifestAuthorization can only be produced by the git-pinned loader "
                "(missing loader token); hand construction is refused")


# ============================================================================= schema validation
def _validate_authorization_schema(raw: dict, authorized_phase: str) -> None:
    if not isinstance(raw, dict):
        raise FormalAuthorizationIntegrityError("authorization must be a JSON object")
    if raw.get("authorization_schema_version") != AUTHORIZATION_SCHEMA_VERSION:
        raise FormalAuthorizationIntegrityError(
            f"authorization_schema_version must be {AUTHORIZATION_SCHEMA_VERSION!r}")
    verdict = raw.get("verdict")
    if verdict not in _VERDICT_PHASE:
        raise FormalManifestGenerationNotAuthorized(
            f"verdict {verdict!r} does not authorize formal manifest generation")
    phase = _VERDICT_PHASE[verdict]
    if phase != authorized_phase or raw.get("authorized_phase") != authorized_phase:
        raise FormalAuthorizationIntegrityError("verdict/authorized_phase/requested phase disagree")
    common = {"authorization_schema_version", "verdict", "audited_writer_commit", "authorized_phase",
              "protocol_commit", "generator_commit", "runtime_commit", "authorized_output_relpath",
              "authorizes_formal_manifest_generation", "authorizes_confirmatory_data_generation",
              "authorizes_confirmatory_run", "one_shot", "power_recertification_required"}
    test_extra = {"model_analysis_freeze_sha256", "train_validation_manifest_sha256", "analysis_code_sha256"}
    expected = common | test_extra if phase == "test" else common
    keys = set(raw)
    if keys != expected:
        missing = expected - keys
        unknown = keys - expected
        raise FormalAuthorizationIntegrityError(
            f"authorization key set mismatch (missing={sorted(missing)}, unknown={sorted(unknown)})")
    for k in ("audited_writer_commit", "protocol_commit", "generator_commit", "runtime_commit"):
        if not _is_40hex(raw[k]):
            raise FormalAuthorizationIntegrityError(f"{k} must be 40 lowercase hex")
    _strict_bool(raw["authorizes_formal_manifest_generation"], "authorizes_formal_manifest_generation")
    _strict_bool(raw["authorizes_confirmatory_data_generation"], "authorizes_confirmatory_data_generation")
    _strict_bool(raw["authorizes_confirmatory_run"], "authorizes_confirmatory_run")
    _strict_bool(raw["one_shot"], "one_shot")
    _strict_bool(raw["power_recertification_required"], "power_recertification_required")
    if raw["authorizes_formal_manifest_generation"] is not True:
        raise FormalManifestGenerationNotAuthorized("authorizes_formal_manifest_generation must be true")
    if raw["authorizes_confirmatory_data_generation"] is not False:
        raise FormalAuthorizationIntegrityError("authorizes_confirmatory_data_generation must be false")
    if raw["authorizes_confirmatory_run"] is not False:
        raise FormalAuthorizationIntegrityError("authorizes_confirmatory_run must be false")
    if raw["one_shot"] is not True:
        raise FormalAuthorizationIntegrityError("one_shot must be true")
    if raw["power_recertification_required"] is not False:
        raise FormalAuthorizationIntegrityError("power_recertification_required must be false")
    if raw["authorized_output_relpath"] != f"{FORMAL_ARTIFACT_SUBROOT}/{phase}":
        raise FormalAuthorizationIntegrityError("authorized_output_relpath inconsistent with phase")
    if raw["generator_commit"] != raw["audited_writer_commit"]:
        raise FormalAuthorizationIntegrityError("generator_commit must equal audited_writer_commit")
    if phase == "test":
        for k in test_extra:
            if not _is_64hex(raw[k]):
                raise FormalAuthorizationIntegrityError(f"test authorization {k} must be 64 lowercase hex")


# ============================================================================= §4/§8 git-pinned repo verification
def _verify_repository_and_build_context(git_reader: GitReader, *, authorization_commit: str,
                                         authorized_phase: str, expected_writer_commit: str):
    """Verify the repository, HEAD==authorization_commit, clean tree, writer-commit ancestry, and byte-identical
    critical blobs across F1 / authorization commit / working tree. Returns (context, auth_artifact_bytes)."""
    if authorized_phase not in AUTHORIZATION_RELPATHS:
        raise FormalAuthorizationIntegrityError(f"unknown authorized_phase {authorized_phase!r}")
    if not _is_40hex(authorization_commit):
        raise FormalCommitFreezeError("authorization_commit must be 40 lowercase hex")
    if not _is_40hex(expected_writer_commit):
        raise FormalCommitFreezeError("expected_writer_commit must be 40 lowercase hex")
    if not git_reader.is_repo():
        raise FormalCommitFreezeError(f"{git_reader.repo_root} is not a git repository")
    if not git_reader.commit_exists(authorization_commit):
        raise FormalCommitFreezeError(f"authorization commit {authorization_commit} does not exist")
    if not git_reader.commit_exists(expected_writer_commit):
        raise FormalCommitFreezeError(f"writer commit {expected_writer_commit} does not exist")
    # F1 must be an ancestor of the authorization commit (auth may only ADD authorization/audit files)
    if not git_reader.is_ancestor(expected_writer_commit, authorization_commit):
        raise FormalCommitFreezeError("writer_implementation_commit is not an ancestor of the authorization commit")
    if not git_reader.worktree_clean():
        raise FormalCommitFreezeError("worktree is not clean (formal generation requires a clean tree)")
    head = git_reader.head()
    if not _is_40hex(head):
        raise FormalCommitFreezeError(f"could not resolve a 40-hex HEAD, got {head!r}")
    if head != authorization_commit:                                     # FW-B-006: exact pin, no descendant
        raise FormalCommitFreezeError("HEAD must equal the pinned authorization commit (no arbitrary descendant)")
    blob_map = {}
    for rel in CRITICAL_BLOB_RELPATHS:
        b_f1 = git_reader.show_bytes(expected_writer_commit, rel)
        b_auth = git_reader.show_bytes(authorization_commit, rel)
        b_work = git_reader.worktree_bytes(rel)
        if b_f1 is None or b_auth is None or b_work is None:
            raise FormalCommitFreezeError(f"critical blob missing for {rel}")
        if not (b_f1 == b_auth == b_work):
            raise FormalCommitFreezeError(f"critical blob differs across F1/auth/worktree for {rel}")
        blob_map[rel] = _sha256_bytes(b_f1)
    blob_map_sha = _sha256_text(_canon(blob_map))
    auth_rel = AUTHORIZATION_RELPATHS[authorized_phase]
    auth_bytes = git_reader.show_bytes(authorization_commit, auth_rel)
    if auth_bytes is None:
        raise FormalAuthorizationIntegrityError(
            f"authorization artifact {auth_rel} is not present in commit {authorization_commit}")
    ctx = VerifiedFormalRepositoryContext(
        repo_root_realpath=git_reader.repo_root, authorization_commit=authorization_commit,
        authorization_relpath=auth_rel, authorization_blob_sha256=_sha256_bytes(auth_bytes),
        writer_implementation_commit=expected_writer_commit, current_head=head,
        verified_critical_blob_map_sha256=blob_map_sha)
    return ctx, auth_bytes


_CRITICAL_AUTH_FIELDS = (
    ("authorization_schema_version", "authorization_schema_version"),
    ("verdict", "authorization_verdict"),
    ("audited_writer_commit", "audited_writer_commit"),
    ("protocol_commit", "protocol_commit"),
    ("generator_commit", "generator_commit"),
    ("runtime_commit", "runtime_commit"),
    ("authorized_output_relpath", "authorized_output_relpath"),
    ("authorizes_formal_manifest_generation", "authorizes_formal_manifest_generation"),
    ("authorizes_confirmatory_data_generation", "authorizes_confirmatory_data_generation"),
    ("authorizes_confirmatory_run", "authorizes_confirmatory_run"),
    ("one_shot", "one_shot"),
    ("power_recertification_required", "power_recertification_required"),
    ("model_analysis_freeze_sha256", "test_prerequisite_model_analysis_freeze_sha256"),
    ("train_validation_manifest_sha256", "test_prerequisite_train_validation_manifest_sha256"),
    ("analysis_code_sha256", "test_prerequisite_analysis_code_sha256"),
)


def _crosscheck_every_field(raw: dict, auth: VerifiedFormalManifestAuthorization) -> None:
    """FW-B-004: re-parse the committed JSON and compare EVERY critical field to the dataclass; reject any
    divergence (defeats dataclasses.replace / object.__setattr__ mutations of a loader-minted object)."""
    if raw.get("authorized_phase") != auth.authorized_phase:
        raise FormalAuthorizationIntegrityError("authorized_phase diverges from committed artifact")
    for raw_key, attr in _CRITICAL_AUTH_FIELDS:
        if getattr(auth, attr) != raw.get(raw_key, None):
            raise FormalAuthorizationIntegrityError(f"field {attr!r} diverges from committed artifact")


# ============================================================================= §4.2 git-pinned loader
def load_and_verify_formal_manifest_authorization_from_git(
        *, repo_root, authorization_commit: str, authorized_phase: Literal["train_validation", "test"],
        expected_writer_commit: str, git_reader: Optional[GitReader] = None) -> VerifiedFormalManifestAuthorization:
    """The ONLY production way to mint a VerifiedFormalManifestAuthorization. Reads the authorization artifact
    from a FIXED allowlisted relpath via `git show <authorization_commit>:<relpath>` (committed bytes only —
    never the working tree), binds the evidence hash to those bytes, verifies the repository/HEAD/clean/critical
    blobs, and requires the artifact to pin `expected_writer_commit` (== generator == audited writer)."""
    gr = git_reader if git_reader is not None else GitReader(repo_root)
    ctx, auth_bytes = _verify_repository_and_build_context(
        gr, authorization_commit=authorization_commit, authorized_phase=authorized_phase,
        expected_writer_commit=expected_writer_commit)
    raw = _strict_json_loads(auth_bytes.decode("utf-8"))
    _validate_authorization_schema(raw, authorized_phase)
    if raw["audited_writer_commit"] != expected_writer_commit:
        raise FormalAuthorizationIntegrityError("audited_writer_commit != expected_writer_commit (F1 pin)")
    evidence = _sha256_bytes(auth_bytes)
    committed_canonical = _canon(raw)
    is_test = authorized_phase == "test"
    return VerifiedFormalManifestAuthorization(
        authorization_schema_version=raw["authorization_schema_version"], authorization_verdict=raw["verdict"],
        authorization_evidence_sha256=evidence, authorization_commit=authorization_commit,
        audited_writer_commit=raw["audited_writer_commit"], authorized_phase=authorized_phase,
        protocol_commit=raw["protocol_commit"], generator_commit=raw["generator_commit"],
        runtime_commit=raw["runtime_commit"], authorized_output_relpath=raw["authorized_output_relpath"],
        authorizes_formal_manifest_generation=True, authorizes_confirmatory_data_generation=False,
        authorizes_confirmatory_run=False, one_shot=True, power_recertification_required=False,
        test_prerequisite_model_analysis_freeze_sha256=raw.get("model_analysis_freeze_sha256") if is_test else None,
        test_prerequisite_train_validation_manifest_sha256=raw.get("train_validation_manifest_sha256") if is_test else None,
        test_prerequisite_analysis_code_sha256=raw.get("analysis_code_sha256") if is_test else None,
        repository_context=ctx, committed_authorization_canonical_json=committed_canonical,
        committed_authorization_blob_sha256=evidence, _verified_loader_version=VERIFIED_LOADER_VERSION,
        _loader_token=_AUTHORIZATION_LOADER_TOKEN)


def load_and_verify_formal_manifest_authorization(*_a, **_k):
    """PERMANENTLY LOCKED (FW-B-001): the old local-path loader could mint an authorization from any file. The
    production trust root is now a git-pinned artifact; use `load_and_verify_formal_manifest_authorization_from_
    git`. This public entry always fail-closes."""
    raise FormalManifestGenerationNotAuthorized(
        "the local-path authorization loader is permanently locked; authorization must come from a git-pinned "
        "artifact via load_and_verify_formal_manifest_authorization_from_git")


def reverify_authorization_against_git(auth: VerifiedFormalManifestAuthorization, *,
                                       git_reader: Optional[GitReader] = None) -> dict:
    """FW-B-003/004: re-verify a loader-minted authorization against git on every production entry — the token
    and dataclass fields are NOT trusted. Re-runs the full repository/blob verification, re-reads the committed
    artifact, and cross-checks every field. Returns the re-parsed committed mapping."""
    if not isinstance(auth, VerifiedFormalManifestAuthorization):
        raise FormalAuthorizationIntegrityError("authorization must be a VerifiedFormalManifestAuthorization")
    if auth._loader_token is not _AUTHORIZATION_LOADER_TOKEN:
        raise FormalAuthorizationIntegrityError("authorization missing loader token")
    if auth._verified_loader_version != VERIFIED_LOADER_VERSION:
        raise FormalAuthorizationIntegrityError("authorization loader version mismatch")
    rc = auth.repository_context
    if not isinstance(rc, VerifiedFormalRepositoryContext):
        raise FormalAuthorizationIntegrityError("authorization repository_context missing")
    gr = git_reader if git_reader is not None else GitReader(rc.repo_root_realpath)
    new_ctx, auth_bytes = _verify_repository_and_build_context(
        gr, authorization_commit=rc.authorization_commit, authorized_phase=auth.authorized_phase,
        expected_writer_commit=auth.audited_writer_commit)
    if new_ctx != rc:
        raise FormalAuthorizationIntegrityError("repository context changed since authorization was minted")
    evidence = _sha256_bytes(auth_bytes)
    if evidence != auth.committed_authorization_blob_sha256 or evidence != auth.authorization_evidence_sha256:
        raise FormalAuthorizationIntegrityError("committed authorization bytes changed since minting")
    raw = _strict_json_loads(auth_bytes.decode("utf-8"))
    _validate_authorization_schema(raw, auth.authorized_phase)
    if _canon(raw) != auth.committed_authorization_canonical_json:
        raise FormalAuthorizationIntegrityError("committed authorization canonical JSON changed")
    if raw["audited_writer_commit"] != rc.writer_implementation_commit:
        raise FormalAuthorizationIntegrityError("audited_writer_commit != pinned writer_implementation_commit")
    _crosscheck_every_field(raw, auth)
    return raw


# ============================================================================= §8/§9 commit freeze
@dataclass(frozen=True)
class FormalCommitFreeze:
    protocol_commit: str
    generator_commit: str
    runtime_commit: str
    authorization_commit: str
    authorization_evidence_sha256: str


def validate_formal_commit_freeze(freeze: FormalCommitFreeze,
                                  auth: VerifiedFormalManifestAuthorization) -> None:
    """All commit fields 40-hex, non-empty, not a smoke placeholder, and EXACTLY equal to the verified
    authorization (incl. generator==audited_writer==pinned writer commit, and authorization_commit/evidence from
    the repository context). Never auto-substitutes HEAD."""
    if not isinstance(freeze, FormalCommitFreeze):
        raise FormalCommitFreezeError("freeze must be a FormalCommitFreeze")
    for name in ("protocol_commit", "generator_commit", "runtime_commit", "authorization_commit"):
        v = getattr(freeze, name)
        if not _is_40hex(v):
            raise FormalCommitFreezeError(f"{name} must be 40 lowercase hex, got {v!r}")
    if (freeze.protocol_commit, freeze.generator_commit, freeze.runtime_commit) == _SMOKE_PLACEHOLDER:
        raise FormalCommitFreezeError("formal freeze must not use the smoke placeholder a/b/c*40 commits")
    for name in ("protocol_commit", "generator_commit", "runtime_commit"):
        if getattr(freeze, name) in _SMOKE_PLACEHOLDER:
            raise FormalCommitFreezeError(f"{name} must not be a smoke placeholder commit")
    if not _is_64hex(freeze.authorization_evidence_sha256):
        raise FormalCommitFreezeError("authorization_evidence_sha256 must be 64 lowercase hex")
    if freeze.protocol_commit != auth.protocol_commit:
        raise FormalCommitFreezeError("protocol_commit != authorization")
    if freeze.generator_commit != auth.generator_commit:
        raise FormalCommitFreezeError("generator_commit != authorization")
    if freeze.runtime_commit != auth.runtime_commit:
        raise FormalCommitFreezeError("runtime_commit != authorization")
    if freeze.authorization_commit != auth.authorization_commit:
        raise FormalCommitFreezeError("authorization_commit != authorization")
    if freeze.authorization_commit != auth.repository_context.authorization_commit:
        raise FormalCommitFreezeError("authorization_commit != repository_context")
    if freeze.authorization_evidence_sha256 != auth.authorization_evidence_sha256:
        raise FormalCommitFreezeError("authorization_evidence_sha256 != authorization")
    if freeze.generator_commit != auth.audited_writer_commit:
        raise FormalCommitFreezeError("generator_commit must equal audited_writer_commit")
    if freeze.generator_commit != auth.repository_context.writer_implementation_commit:
        raise FormalCommitFreezeError("generator_commit must equal the pinned writer_implementation_commit")


def verify_head_or_ancestor_policy(repo_root, *, authorization_commit: str,
                                   head_resolver=None, ancestor_resolver=None) -> None:
    """FW-B-006: the pinned policy is HEAD == authorization_commit (an arbitrary descendant is NOT accepted; a
    future successor requires a NEW authorization). `ancestor_resolver` is kept for signature compatibility."""
    if not _is_40hex(authorization_commit):
        raise FormalCommitFreezeError("authorization_commit must be 40 lowercase hex")
    head = head_resolver(str(repo_root)) if head_resolver is not None else GitReader(repo_root).head()
    if not _is_40hex(head):
        raise FormalCommitFreezeError(f"could not resolve a 40-hex HEAD, got {head!r}")
    if head != authorization_commit:
        raise FormalCommitFreezeError("HEAD must equal the pinned authorization commit (no arbitrary descendant)")


# ============================================================================= §9 formal environment
class FormalManifestEnvironmentContext:
    """Immutable formal-manifest environment provenance, distinct from smoke. Exact 10-key set, all non-empty
    str, the three fixed FORMAL markers, copy-on-read canonical snapshot; refuses the SMOKE_ONLY marker."""
    __slots__ = ("_canonical_json",)

    def __init__(self, values):
        if not isinstance(values, dict):
            raise FormalEnvironmentError("formal environment must be a dict")
        d = dict(values)
        if set(d) != set(FORMAL_ENV_VERSION_KEYS):
            raise FormalEnvironmentError(
                f"formal environment key set != exact {FORMAL_ENV_VERSION_KEYS} (got {sorted(d)})")
        for k, v in d.items():
            if isinstance(v, bool) or not isinstance(v, str) or not v:
                raise FormalEnvironmentError(f"formal environment[{k!r}] must be a non-empty str, got {v!r}")
        for k, expect in FORMAL_ENV_FIXED.items():
            if d[k] != expect:
                raise FormalEnvironmentError(f"formal environment[{k!r}] must be {expect!r}, got {d[k]!r}")
        if d["execution_mode"] == "SMOKE_ONLY":
            raise FormalEnvironmentError("formal environment must not reuse the SMOKE_ONLY marker")
        object.__setattr__(self, "_canonical_json", _canon(d))

    def __setattr__(self, *_a):
        raise AttributeError("FormalManifestEnvironmentContext is immutable")

    @property
    def values(self) -> dict:
        return json.loads(self._canonical_json)


# ============================================================================= stamping
def _stamp_manifest(manifest: dict, *,
                    validate=MI.validate_fully_resolved_phase_manifest,
                    full_hash=MI.fully_resolved_phase_manifest_hash):
    """Deep-validate + full-hash + flat self-hash stamp + canonical serialize. Returns
    (stamped_dict, full_manifest_sha256, stamped_canonical_json, stamped_file_sha256)."""
    validate(manifest)
    h = full_hash(manifest)
    if not _is_64hex(h):
        raise FormalBundleWriteError("full manifest hash is not 64 lowercase hex")
    stamped = copy.deepcopy(manifest)
    stamped[MI.SELF_HASH_FIELD] = h
    check = copy.deepcopy(stamped)
    check.pop(MI.SELF_HASH_FIELD, None)
    if _canon(check) != _canon(manifest):
        raise FormalBundleWriteError("stamping altered the manifest payload")
    stamped_json = _canon(stamped)
    return stamped, h, stamped_json, _sha256_text(stamped_json)


# ============================================================================= §10 bundle re-verification + Step-5
def verify_formal_manifest_bundle(repo_root, phase: Literal["train_validation", "test"]):
    """Re-verify a sealed formal bundle on disk: exactly 5 files, bundle_index covers the 4 indexed files, the
    manifest deep-validates and its scientific full hash matches its stamped self field and the sidecar/receipt/
    SEALED cross-fields. Returns a dict with full_manifest_sha256 + bundle_index_sha256."""
    final_dir = _resolve_contained_final_dir(repo_root, f"{FORMAL_ARTIFACT_SUBROOT}/{phase}", phase)
    if final_dir.is_symlink() or not final_dir.is_dir():
        raise FormalBundleWriteError(f"formal bundle directory missing for {phase}")
    names = set(p.name for p in final_dir.iterdir())
    if names != set(BUNDLE_FILES):
        raise FormalBundleWriteError(f"formal bundle file set mismatch for {phase}: {sorted(names)}")
    raw_files = {}
    for name in BUNDLE_FILES:
        fp = final_dir / name
        if fp.is_symlink() or not fp.is_file():
            raise FormalBundleWriteError(f"bundle file {name} missing or is a symlink")
        raw_files[name] = fp.read_bytes()
    index = _strict_json_loads(raw_files["bundle_index.json"].decode("utf-8"))
    if index.get("bundle_schema_version") != BUNDLE_SCHEMA_VERSION or not isinstance(index.get("files"), dict):
        raise FormalBundleWriteError("bundle_index.json schema invalid")
    if set(index["files"]) != set(_INDEXED_BUNDLE_FILES):
        raise FormalBundleWriteError("bundle_index does not index exactly the 4 core files")
    for name in _INDEXED_BUNDLE_FILES:
        if index["files"][name] != _sha256_bytes(raw_files[name]):
            raise FormalBundleWriteError(f"bundle_index hash mismatch for {name}")
    bundle_index_sha256 = _sha256_bytes(raw_files["bundle_index.json"])
    stamped = _strict_json_loads(raw_files["phase_manifest.json"].decode("utf-8"))
    if stamped.get("phase") != phase:
        raise FormalBundleWriteError("phase_manifest phase mismatch")
    self_hash = stamped.get(MI.SELF_HASH_FIELD)
    recomputed = MI.fully_resolved_phase_manifest_hash(stamped)   # MI deep-validates + pops the self field
    if recomputed != self_hash:
        raise FormalBundleWriteError("phase_manifest scientific full hash mismatch")
    stamped_file_sha = _sha256_bytes(raw_files["phase_manifest.json"])
    if raw_files["phase_manifest.sha256"].decode("utf-8") != f"{stamped_file_sha}  phase_manifest.json\n":
        raise FormalBundleWriteError("phase_manifest.sha256 sidecar mismatch")
    receipt = _strict_json_loads(raw_files["write_receipt.json"].decode("utf-8"))
    sealed = _strict_json_loads(raw_files["SEALED"].decode("utf-8"))
    for obj, label in ((receipt, "receipt"), (sealed, "SEALED")):
        if obj.get("full_manifest_sha256") != self_hash or obj.get("stamped_file_sha256") != stamped_file_sha \
                or obj.get("phase") != phase:
            raise FormalBundleWriteError(f"{label} cross-field mismatch")
    return {"full_manifest_sha256": self_hash, "stamped_file_sha256": stamped_file_sha,
            "bundle_index_sha256": bundle_index_sha256}


def _verify_step5_prerequisites(repo_root, auth: VerifiedFormalManifestAuthorization) -> None:
    """FW-B-007: before a test phase is built, require the sealed train_validation bundle + model-analysis
    freeze + analysis-code freeze to EXIST and match the authorization's Step-5 hashes on disk."""
    root = Path(repo_root)
    try:
        tv = verify_formal_manifest_bundle(root, "train_validation")
    except (FormalBundleWriteError, FormalPathSafetyError) as exc:
        raise EarlyTestAccessError(f"EXPERIMENT_INVALID_EARLY_TEST_ACCESS: train_validation bundle invalid: {exc}")
    if tv["full_manifest_sha256"] != auth.test_prerequisite_train_validation_manifest_sha256:
        raise EarlyTestAccessError(
            "EXPERIMENT_INVALID_EARLY_TEST_ACCESS: train_validation_manifest_sha256 does not match the sealed bundle")
    mf_path = root / MODEL_FREEZE_RELPATH
    if mf_path.is_symlink() or not mf_path.is_file():
        raise EarlyTestAccessError("EXPERIMENT_INVALID_EARLY_TEST_ACCESS: model_analysis_freeze.json missing")
    mf_bytes = mf_path.read_bytes()
    if _sha256_bytes(mf_bytes) != auth.test_prerequisite_model_analysis_freeze_sha256:
        raise EarlyTestAccessError("EXPERIMENT_INVALID_EARLY_TEST_ACCESS: model_analysis_freeze raw hash mismatch")
    mf = _strict_json_loads(mf_bytes.decode("utf-8"))
    if set(mf) != {"freeze_schema_version", "train_validation_manifest_sha256", "analysis_code_sha256",
                   "write_complete"}:
        raise EarlyTestAccessError("EXPERIMENT_INVALID_EARLY_TEST_ACCESS: model_analysis_freeze schema invalid")
    if mf["freeze_schema_version"] != MODEL_FREEZE_SCHEMA_VERSION or mf["write_complete"] is not True \
            or mf["train_validation_manifest_sha256"] != auth.test_prerequisite_train_validation_manifest_sha256 \
            or mf["analysis_code_sha256"] != auth.test_prerequisite_analysis_code_sha256:
        raise EarlyTestAccessError("EXPERIMENT_INVALID_EARLY_TEST_ACCESS: model_analysis_freeze cross-field mismatch")
    ac_path = root / ANALYSIS_CODE_FREEZE_RELPATH
    if ac_path.is_symlink() or not ac_path.is_file():
        raise EarlyTestAccessError("EXPERIMENT_INVALID_EARLY_TEST_ACCESS: analysis_code_freeze.json missing")
    ac_bytes = ac_path.read_bytes()
    if _sha256_bytes(ac_bytes) != auth.test_prerequisite_analysis_code_sha256:
        raise EarlyTestAccessError("EXPERIMENT_INVALID_EARLY_TEST_ACCESS: analysis_code_freeze raw hash mismatch")
    ac = _strict_json_loads(ac_bytes.decode("utf-8"))
    if set(ac) != {"schema_version", "files", "write_complete"} \
            or ac.get("schema_version") != ANALYSIS_CODE_FREEZE_SCHEMA_VERSION or ac.get("write_complete") is not True:
        raise EarlyTestAccessError("EXPERIMENT_INVALID_EARLY_TEST_ACCESS: analysis_code_freeze schema invalid")
    files = ac["files"]
    if not isinstance(files, list) or not files:
        raise EarlyTestAccessError("EXPERIMENT_INVALID_EARLY_TEST_ACCESS: analysis_code_freeze files must be non-empty")
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"} or not _is_64hex(entry.get("sha256")):
            raise EarlyTestAccessError("EXPERIMENT_INVALID_EARLY_TEST_ACCESS: analysis_code_freeze entry invalid")
        rel = entry["path"]
        if not isinstance(rel, str) or not rel or os.path.isabs(rel) or ".." in Path(rel).parts \
                or rel.startswith(("/", "\\")):
            raise EarlyTestAccessError("EXPERIMENT_INVALID_EARLY_TEST_ACCESS: analysis_code path unsafe")
        if rel.startswith(FORMAL_ARTIFACT_SUBROOT) or "test" in Path(rel).parts or "data" in Path(rel).parts:
            raise EarlyTestAccessError("EXPERIMENT_INVALID_EARLY_TEST_ACCESS: analysis_code path references outcome/data")
        fp = root / rel
        if fp.is_symlink() or not fp.is_file():
            raise EarlyTestAccessError(f"EXPERIMENT_INVALID_EARLY_TEST_ACCESS: analysis file missing {rel}")
        if _sha256_bytes(fp.read_bytes()) != entry["sha256"]:
            raise EarlyTestAccessError(f"EXPERIMENT_INVALID_EARLY_TEST_ACCESS: analysis file bytes changed {rel}")


# ============================================================================= §10/§11 prepared phase
@dataclass(frozen=True)
class PreparedFormalPhase:
    phase: Literal["train_validation", "test"]
    _unstamped_manifest_canonical_json: str
    _stamped_manifest_canonical_json: str
    full_manifest_sha256: str
    stamped_file_sha256: str
    authorization_evidence_sha256: str
    protocol_commit: str
    generator_commit: str
    runtime_commit: str
    repository_context_json: str
    formal_environment_json: str
    commit_freeze_json: str
    authorization_blob_sha256: str

    @property
    def unstamped_manifest(self) -> dict:
        return json.loads(self._unstamped_manifest_canonical_json)

    @property
    def stamped_manifest(self) -> dict:
        return json.loads(self._stamped_manifest_canonical_json)

    @property
    def stamped_manifest_bytes(self) -> bytes:
        return self._stamped_manifest_canonical_json.encode("utf-8")


def _repo_context_canonical(rc: VerifiedFormalRepositoryContext) -> str:
    return _canon({"repo_root_realpath": rc.repo_root_realpath, "authorization_commit": rc.authorization_commit,
                   "authorization_relpath": rc.authorization_relpath,
                   "authorization_blob_sha256": rc.authorization_blob_sha256,
                   "writer_implementation_commit": rc.writer_implementation_commit,
                   "current_head": rc.current_head,
                   "verified_critical_blob_map_sha256": rc.verified_critical_blob_map_sha256})


def _commit_freeze_canonical(freeze: FormalCommitFreeze) -> str:
    return _canon({"protocol_commit": freeze.protocol_commit, "generator_commit": freeze.generator_commit,
                   "runtime_commit": freeze.runtime_commit, "authorization_commit": freeze.authorization_commit,
                   "authorization_evidence_sha256": freeze.authorization_evidence_sha256})


def prepare_authorized_formal_phase_in_memory(
        *, authorization: VerifiedFormalManifestAuthorization, commits: FormalCommitFreeze,
        environment_versions: FormalManifestEnvironmentContext,
        git_reader: Optional[GitReader] = None) -> PreparedFormalPhase:
    """Strict construction of an authorized formal phase (in memory only; nothing is written). Repo root comes
    ONLY from the git-verified authorization context — there is no repo_root/skip argument (FW-B-005)."""
    reverify_authorization_against_git(authorization, git_reader=git_reader)        # FW-B-003/004/005/006
    phase = authorization.authorized_phase
    if phase not in ("train_validation", "test"):
        raise FormalAuthorizationIntegrityError(f"unknown authorized_phase {phase!r}")
    if authorization.authorizes_formal_manifest_generation is not True:
        raise FormalManifestGenerationNotAuthorized("FORMAL_MANIFEST_GENERATION_NOT_AUTHORIZED")
    repo_root = authorization.repository_context.repo_root_realpath
    if phase == "test":
        if authorization.authorization_verdict != TEST_VERDICT:
            raise EarlyTestAccessError("EXPERIMENT_INVALID_EARLY_TEST_ACCESS: test phase requires the TEST verdict")
        if authorization.authorized_output_relpath != f"{FORMAL_ARTIFACT_SUBROOT}/test":
            raise EarlyTestAccessError("EXPERIMENT_INVALID_EARLY_TEST_ACCESS: test output path is not the test dir")
        for val in (authorization.test_prerequisite_model_analysis_freeze_sha256,
                    authorization.test_prerequisite_train_validation_manifest_sha256,
                    authorization.test_prerequisite_analysis_code_sha256):
            if not _is_64hex(val):
                raise EarlyTestAccessError(
                    "EXPERIMENT_INVALID_EARLY_TEST_ACCESS: missing Step-5 model/analysis freeze prerequisite")
        _verify_step5_prerequisites(repo_root, authorization)                       # FW-B-007 real artifacts
    else:
        if authorization.authorization_verdict != TRAIN_VALIDATION_VERDICT:
            raise FormalAuthorizationIntegrityError("train_validation phase requires the TRAIN_VALIDATION verdict")
    validate_formal_commit_freeze(commits, authorization)
    if not isinstance(environment_versions, FormalManifestEnvironmentContext):
        raise FormalEnvironmentError("environment_versions must be a FormalManifestEnvironmentContext")
    BS.require_known_answer_compatibility()                                          # mandatory KAT
    commit_ctx = GEN.CommitContext(commits.protocol_commit, commits.generator_commit, commits.runtime_commit)
    manifest = GEN._build_unsealed_manifest(phase, commit_ctx, environment_versions)
    if manifest["environment_versions"].get("execution_mode") == "SMOKE_ONLY":
        raise FormalEnvironmentError("constructed manifest carries SMOKE_ONLY marker")
    stamped, full_hash, stamped_json, stamped_file_sha = _stamp_manifest(manifest)
    return PreparedFormalPhase(
        phase=phase, _unstamped_manifest_canonical_json=_canon(manifest),
        _stamped_manifest_canonical_json=stamped_json, full_manifest_sha256=full_hash,
        stamped_file_sha256=stamped_file_sha, authorization_evidence_sha256=authorization.authorization_evidence_sha256,
        protocol_commit=commits.protocol_commit, generator_commit=commits.generator_commit,
        runtime_commit=commits.runtime_commit,
        repository_context_json=_repo_context_canonical(authorization.repository_context),
        formal_environment_json=_canon(environment_versions.values),
        commit_freeze_json=_commit_freeze_canonical(commits),
        authorization_blob_sha256=authorization.committed_authorization_blob_sha256)


# ============================================================================= §12/§13/§14/§16 bundle + write
@dataclass(frozen=True)
class FormalWriteReceipt:
    phase: str
    full_manifest_sha256: str
    stamped_file_sha256: str
    bundle_index_sha256: str
    authorization_evidence_sha256: str
    authorization_commit: str
    protocol_commit: str
    generator_commit: str
    runtime_commit: str
    final_dir: str
    committed: bool
    write_complete: bool
    post_commit_verification_passed: bool
    post_commit_warning: Optional[str]


def _resolve_contained_final_dir(output_root, relpath: str, phase: str) -> Path:
    """§14 path safety: relpath must be relative, no '..', exactly artifacts/formal/confirmatory_v4/<phase>,
    dir name == phase, and the longest existing prefix must resolve inside the repo root (no symlink escape)."""
    if not isinstance(relpath, str) or not relpath:
        raise FormalPathSafetyError("authorized_output_relpath must be a non-empty relative path")
    if os.path.isabs(relpath) or relpath.startswith(("/", "\\")):
        raise FormalPathSafetyError("authorized_output_relpath must not be absolute")
    if ".." in Path(relpath).parts:
        raise FormalPathSafetyError("authorized_output_relpath must not contain '..'")
    if relpath != f"{FORMAL_ARTIFACT_SUBROOT}/{phase}":
        raise FormalPathSafetyError(f"authorized_output_relpath must be exactly {FORMAL_ARTIFACT_SUBROOT}/{phase}")
    root = Path(output_root).resolve()
    final_dir = root / relpath
    if final_dir.name != phase:
        raise FormalPathSafetyError("final directory name must equal the phase")
    existing = final_dir
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    resolved_existing = existing.resolve()
    if resolved_existing != root and root not in resolved_existing.parents:
        raise FormalPathSafetyError("authorized output path escapes the repo root (symlink/containment)")
    return final_dir


def _build_bundle_files(prepared: PreparedFormalPhase, auth: VerifiedFormalManifestAuthorization):
    """FW-B-014: build the 5 fixed bundle files. bundle_index.json indexes the 4 core files by sha256; its own
    sha256 (bundle_index_sha256) is returned for external pinning, avoiding a self-reference cycle."""
    manifest_bytes = prepared.stamped_manifest_bytes
    sidecar_bytes = f"{prepared.stamped_file_sha256}  phase_manifest.json\n".encode("utf-8")
    receipt_core = {
        "receipt_schema_version": RECEIPT_SCHEMA_VERSION, "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "phase": prepared.phase, "full_manifest_sha256": prepared.full_manifest_sha256,
        "stamped_file_sha256": prepared.stamped_file_sha256,
        "authorization_evidence_sha256": prepared.authorization_evidence_sha256,
        "authorization_commit": auth.authorization_commit, "protocol_commit": prepared.protocol_commit,
        "generator_commit": prepared.generator_commit, "runtime_commit": prepared.runtime_commit,
        "one_shot": True, "formal_manifest_written": True, "confirmatory_data_authorized": False,
        "confirmatory_run_authorized": False}
    receipt_bytes = (_canon(receipt_core) + "\n").encode("utf-8")
    sealed_core = {"bundle_schema_version": BUNDLE_SCHEMA_VERSION, "phase": prepared.phase,
                   "full_manifest_sha256": prepared.full_manifest_sha256,
                   "stamped_file_sha256": prepared.stamped_file_sha256, "write_complete": True}
    sealed_bytes = (_canon(sealed_core) + "\n").encode("utf-8")
    core = {"phase_manifest.json": manifest_bytes, "phase_manifest.sha256": sidecar_bytes,
            "write_receipt.json": receipt_bytes, "SEALED": sealed_bytes}
    index = {"bundle_schema_version": BUNDLE_SCHEMA_VERSION, "phase": prepared.phase,
             "files": {name: _sha256_bytes(core[name]) for name in _INDEXED_BUNDLE_FILES}}
    index_bytes = (_canon(index) + "\n").encode("utf-8")
    files = dict(core)
    files["bundle_index.json"] = index_bytes
    return files, _sha256_bytes(index_bytes)


# --- atomic no-replace primitive (FW-B-011) ----------------------------------------------------------------
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


def _renameat2_noreplace(src, dst) -> None:
    """Atomic exclusive directory claim via renameat2(RENAME_NOREPLACE). EEXIST -> FileExistsError; the syscall
    being unavailable -> FormalAtomicNoReplaceUnavailable (fail-closed; NEVER falls back to os.replace)."""
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.renameat2.restype = ctypes.c_int
        libc.renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    except (OSError, AttributeError) as exc:
        raise FormalAtomicNoReplaceUnavailable(f"renameat2 unavailable: {exc}")
    ctypes.set_errno(0)
    res = libc.renameat2(_AT_FDCWD, os.fsencode(str(src)), _AT_FDCWD, os.fsencode(str(dst)),
                         ctypes.c_uint(_RENAME_NOREPLACE))
    if res != 0:
        e = ctypes.get_errno()
        if e == errno.EEXIST:
            raise FileExistsError(e, "target exists")
        if e in (errno.ENOSYS, errno.EINVAL, errno.ENOTTY, errno.EOPNOTSUPP):
            raise FormalAtomicNoReplaceUnavailable(f"renameat2(RENAME_NOREPLACE) not supported (errno {e})")
        raise OSError(e, os.strerror(e))


def _fsync_path(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _make_dirs_tracked(target: Path):
    """Create target's directory chain, returning the list of directories actually created (deepest last)."""
    created = []
    stack = []
    cur = target
    while not cur.exists():
        stack.append(cur)
        if cur == cur.parent:
            break
        cur = cur.parent
    for d in reversed(stack):
        try:
            os.mkdir(str(d))
            created.append(d)
        except FileExistsError:
            pass
    return created


def _cleanup_dir(path: Path) -> None:
    try:
        for child in path.iterdir():
            if child.is_dir() and not child.is_symlink():
                _cleanup_dir(child)
            else:
                child.unlink()
        os.rmdir(str(path))
    except OSError:
        pass


def _atomic_bundle_core(final_dir: Path, files: dict) -> dict:
    """Shared one-shot atomic claim. Writes files into a same-parent staging dir (flush+fsync each, fsync dir,
    read-back verify), then atomically claims `final_dir` via renameat2(RENAME_NOREPLACE). Commit point = a
    successful rename. Pre-commit failure -> no final dir, staging + newly-created parents removed. Post-commit
    failure (parent fsync / final read-back) -> returns a warning, never raises with a look-formal final dir."""
    final_dir = Path(final_dir)
    parent = final_dir.parent
    created_parents = _make_dirs_tracked(parent)
    staging = Path(tempfile.mkdtemp(prefix=".formal_staging_", dir=str(parent)))
    committed = False
    try:
        for name, data in files.items():
            if not isinstance(data, (bytes, bytearray)):
                raise FormalBundleWriteError(f"bundle file {name} must be bytes")
            fpath = staging / name
            with open(fpath, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
        _fsync_path(staging)
        for name, data in files.items():
            with open(staging / name, "rb") as fh:
                if fh.read() != data:
                    raise FormalBundleWriteError(f"read-back mismatch for {name} before claim")
        try:
            _renameat2_noreplace(staging, final_dir)                 # atomic exclusive claim = COMMIT POINT
        except FileExistsError:
            raise FormalManifestAlreadyExists(
                f"FORMAL_MANIFEST_ALREADY_EXISTS: {final_dir} already exists (one-shot; no replace)")
        committed = True
        staging = None
        post_ok, warn = True, None
        try:
            _fsync_path(parent)
            for name, data in files.items():
                with open(final_dir / name, "rb") as fh:
                    if fh.read() != data:
                        raise FormalBundleWriteError(f"post-commit read-back mismatch for {name}")
        except OSError as exc:
            post_ok, warn = False, f"post-commit verification failed (bundle IS committed): {exc}"
        return {"post_commit_verification_passed": post_ok, "post_commit_warning": warn}
    except (FormalManifestAlreadyExists, FormalAtomicNoReplaceUnavailable, FormalBundleWriteError):
        raise
    except OSError as exc:
        raise FormalBundleWriteError(f"FORMAL_BUNDLE_WRITE_ERROR: {exc}") from exc
    finally:
        if not committed:
            if staging is not None and Path(staging).exists():
                _cleanup_dir(Path(staging))
            for d in reversed(created_parents):                     # FW-B-012: remove newly-created empty parents
                try:
                    os.rmdir(str(d))
                except OSError:
                    pass


def _test_only_atomic_write_bundle(final_dir, files) -> dict:
    """TEST_ONLY low-level atomic bundle writer (no authorization / repo binding). Production code must NOT call
    this; use write_authorized_formal_manifest_bundle_atomic."""
    return _atomic_bundle_core(Path(final_dir), files)


def _writer_revalidate_prepared(prepared: PreparedFormalPhase) -> None:
    """FW-B-008: the writer does NOT trust the prepared payload/hashes. Re-parse the un-stamped snapshot, re-run
    MI deep validation, recompute the scientific full hash, re-stamp, and byte-compare against the prepared
    snapshot. A hand-built garbage payload fails MI validation here, before any filesystem effect."""
    if not isinstance(prepared, PreparedFormalPhase):
        raise FormalBundleWriteError("prepared must be a PreparedFormalPhase")
    manifest = prepared.unstamped_manifest
    try:
        stamped, full_hash, stamped_json, stamped_file_sha = _stamp_manifest(manifest)
    except MI.ManifestIntegrityError as exc:                  # garbage payload fails MI deep validation here
        raise FormalBundleWriteError(f"FORMAL_BUNDLE_WRITE_ERROR: prepared payload fails deep validation: {exc}")
    if full_hash != prepared.full_manifest_sha256:
        raise FormalBundleWriteError("recomputed full_manifest_sha256 disagrees with prepared")
    if stamped_json != prepared._stamped_manifest_canonical_json:
        raise FormalBundleWriteError("recomputed stamped manifest disagrees with prepared")
    if stamped_file_sha != prepared.stamped_file_sha256:
        raise FormalBundleWriteError("recomputed stamped_file_sha256 disagrees with prepared")
    if _sha256_bytes(prepared.stamped_manifest_bytes) != prepared.stamped_file_sha256:
        raise FormalBundleWriteError("stamped bytes hash disagrees with prepared stamped_file_sha256")


def write_authorized_formal_manifest_bundle_atomic(
        prepared: PreparedFormalPhase, *, authorization: VerifiedFormalManifestAuthorization,
        git_reader: Optional[GitReader] = None) -> FormalWriteReceipt:
    """Production one-shot atomic bundle write. Re-verifies the authorization against git, re-validates the
    prepared payload with MI (not just hashes), binds the output root to the verified repo (no output_root
    argument), and performs an atomic no-replace claim. In the current repo no authorization can be minted, so
    this is unreachable via any real call chain."""
    reverify_authorization_against_git(authorization, git_reader=git_reader)
    if not isinstance(prepared, PreparedFormalPhase):
        raise FormalBundleWriteError("prepared must be a PreparedFormalPhase")
    if prepared.phase != authorization.authorized_phase:
        raise FormalBundleWriteError("prepared.phase != authorization.authorized_phase")
    if prepared.authorization_evidence_sha256 != authorization.authorization_evidence_sha256 \
            or prepared.authorization_blob_sha256 != authorization.committed_authorization_blob_sha256:
        raise FormalBundleWriteError("prepared authorization evidence mismatch")
    if prepared.repository_context_json != _repo_context_canonical(authorization.repository_context):
        raise FormalBundleWriteError("prepared repository context disagrees with authorization")
    if (prepared.protocol_commit != authorization.protocol_commit
            or prepared.generator_commit != authorization.generator_commit
            or prepared.runtime_commit != authorization.runtime_commit):
        raise FormalBundleWriteError("prepared commit fields disagree with authorization")
    _writer_revalidate_prepared(prepared)                                            # FW-B-008 deep re-validation
    repo_root = authorization.repository_context.repo_root_realpath                   # FW-B-009 bound root
    final_dir = _resolve_contained_final_dir(repo_root, authorization.authorized_output_relpath, prepared.phase)
    files, bundle_index_sha256 = _build_bundle_files(prepared, authorization)
    status = _atomic_bundle_core(final_dir, files)
    return FormalWriteReceipt(
        phase=prepared.phase, full_manifest_sha256=prepared.full_manifest_sha256,
        stamped_file_sha256=prepared.stamped_file_sha256, bundle_index_sha256=bundle_index_sha256,
        authorization_evidence_sha256=prepared.authorization_evidence_sha256,
        authorization_commit=authorization.authorization_commit, protocol_commit=prepared.protocol_commit,
        generator_commit=prepared.generator_commit, runtime_commit=prepared.runtime_commit,
        final_dir=str(final_dir), committed=True, write_complete=True,
        post_commit_verification_passed=status["post_commit_verification_passed"],
        post_commit_warning=status["post_commit_warning"])


__all__ = [
    "VERIFIED_LOADER_VERSION", "AUTHORIZATION_SCHEMA_VERSION", "RECEIPT_SCHEMA_VERSION", "BUNDLE_SCHEMA_VERSION",
    "MODEL_FREEZE_SCHEMA_VERSION", "ANALYSIS_CODE_FREEZE_SCHEMA_VERSION", "FORMAL_ENV_VERSION_KEYS",
    "FORMAL_ENV_FIXED", "TRAIN_VALIDATION_VERDICT", "TEST_VERDICT", "FORMAL_ARTIFACT_SUBROOT",
    "AUTHORIZATION_RELPATHS", "CRITICAL_BLOB_RELPATHS", "BUNDLE_FILES",
    "FormalManifestGenerationNotAuthorized", "FormalAuthorizationIntegrityError", "FormalCommitFreezeError",
    "FormalEnvironmentError", "EarlyTestAccessError", "FormalPathSafetyError", "FormalManifestAlreadyExists",
    "FormalBundleWriteError", "FormalAtomicNoReplaceUnavailable", "GitReader",
    "VerifiedFormalRepositoryContext", "VerifiedFormalManifestAuthorization",
    "load_and_verify_formal_manifest_authorization_from_git", "load_and_verify_formal_manifest_authorization",
    "reverify_authorization_against_git", "FormalCommitFreeze", "validate_formal_commit_freeze",
    "verify_head_or_ancestor_policy", "FormalManifestEnvironmentContext", "PreparedFormalPhase",
    "prepare_authorized_formal_phase_in_memory", "verify_formal_manifest_bundle", "FormalWriteReceipt",
    "write_authorized_formal_manifest_bundle_atomic",
]
