"""Confirmatory v4 FORMAL manifest writer + real commit-freeze interface (offline; no Isaac, no GPU).

Authorization boundary
-----------------------
Claude C's generator-smoke final gate authorises *implementing* this writer, NOT running it: the current gate
carries `authorizes_formal_manifest_generation=false`. This module therefore implements the complete future
success path (real commit freeze, formal environment provenance, independent phase construction, deep
validation, full-hash stamping, atomic one-shot bundle write, receipt/evidence binding) while remaining
**fail-closed**: a `VerifiedFormalManifestAuthorization` can only be minted by `load_and_verify_...`, which
requires a future, independently-signed C authorization file with `authorizes_formal_manifest_generation=true`
and the exact phase-specific verdict. No file in the current repository satisfies that, so every production
generation entry fails BEFORE any formal phase is constructed and BEFORE any filesystem effect. There is no
`--force`, environment-variable, boolean-flag, or hand-constructed-dataclass bypass.

This module writes nothing at import time and performs no filesystem effect until an authorized, verified,
atomic bundle write is explicitly requested with a verified authorization (unreachable in the current repo).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_generator as GEN
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_block_state as BS
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI

VERIFIED_LOADER_VERSION = "confirmatory_v4_formal_manifest_authorization_loader_v1"
RECEIPT_SCHEMA_VERSION = "confirmatory_v4_formal_manifest_receipt_v1"

# Fixed formal-environment markers. Formal manifest generation is an OFFLINE pure step: no Isaac, no GPU.
FORMAL_ENV_VERSION_KEYS = ("python_implementation", "python_version", "numpy_version", "torch_version",
                           "os_system", "os_release", "machine", "isaac_status", "gpu_status", "execution_mode")
FORMAL_ENV_FIXED = {"execution_mode": "FORMAL_MANIFEST_GENERATION",
                    "isaac_status": "NOT_IMPORTED_NOT_LAUNCHED",
                    "gpu_status": "NOT_USED_MANIFEST_GENERATION"}

TRAIN_VALIDATION_VERDICT = "AUTHORIZE_TRAIN_VALIDATION_MANIFEST_GENERATION"
TEST_VERDICT = "AUTHORIZE_TEST_MANIFEST_GENERATION"
_VERDICT_PHASE = {TRAIN_VALIDATION_VERDICT: "train_validation", TEST_VERDICT: "test"}
FORMAL_ARTIFACT_SUBROOT = "artifacts/formal/confirmatory_v4"

BUNDLE_FILES = ("phase_manifest.json", "phase_manifest.sha256", "write_receipt.json", "SEALED")

_GIT_SHA_RE_LEN = 40
_HASH_HEX_LEN = 64
_SMOKE_PLACEHOLDER = GEN.SMOKE_PLACEHOLDER            # ("a"*40, "b"*40, "c"*40)

# Private token: a VerifiedFormalManifestAuthorization can only be constructed by the loader. Not a crypto
# boundary — it prevents ordinary business code from fabricating an authorization by passing True.
_AUTHORIZATION_LOADER_TOKEN = object()


# ----------------------------------------------------------------------------- errors
class FormalManifestGenerationNotAuthorized(RuntimeError):
    """Raised (as FORMAL_MANIFEST_GENERATION_NOT_AUTHORIZED) whenever a formal phase would be constructed
    without a verified, generation-authorizing C authorization. Fail-closed default."""


class FormalAuthorizationIntegrityError(ValueError):
    """The authorization file / object failed schema or evidence-integrity verification."""


class FormalCommitFreezeError(ValueError):
    """The real commit-freeze context is malformed, is a smoke placeholder, or disagrees with the auth."""


class FormalEnvironmentError(ValueError):
    """The formal environment provenance context is malformed or reuses a smoke marker."""


class EarlyTestAccessError(RuntimeError):
    """EXPERIMENT_INVALID_EARLY_TEST_ACCESS — a test manifest was requested before the Step-5 model/analysis
    freeze prerequisites were provided (test manifest must not exist before train/validation freeze)."""


class FormalPathSafetyError(ValueError):
    """The authorized output path is unsafe (absolute / traversal / symlink escape / phase mismatch)."""


class FormalManifestAlreadyExists(RuntimeError):
    """FORMAL_MANIFEST_ALREADY_EXISTS — formal bundle write is one-shot create; a second write is refused even
    if byte-identical. There is no overwrite / resume / force."""


class FormalBundleWriteError(RuntimeError):
    """An atomic bundle write failed after staging; no final directory is left behind."""


# ----------------------------------------------------------------------------- canonical helpers
def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _is_40hex(s) -> bool:
    return isinstance(s, str) and len(s) == _GIT_SHA_RE_LEN and all(c in "0123456789abcdef" for c in s)


def _is_64hex(s) -> bool:
    return isinstance(s, str) and len(s) == _HASH_HEX_LEN and all(c in "0123456789abcdef" for c in s)


def _require_true(value, what: str) -> None:
    if value is not True:
        raise FormalAuthorizationIntegrityError(f"{what} must be exactly true")


def _require_false(value, what: str) -> None:
    if value is not False:
        raise FormalAuthorizationIntegrityError(f"{what} must be exactly false")


# ============================================================================= §6 authorization evidence
@dataclass(frozen=True)
class VerifiedFormalManifestAuthorization:
    """Immutable, loader-minted proof that a specific formal phase generation is authorized. Carries the
    loader's process evidence (`_raw_authorization_canonical_json`, `authorization_evidence_sha256`,
    `_verified_loader_version`) so a downstream writer can re-verify it was produced by the loader and was not
    hand-fabricated. Construction requires the private loader token."""
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
    test_prerequisite_model_analysis_freeze_sha256: Optional[str]
    test_prerequisite_train_validation_manifest_sha256: Optional[str]
    test_prerequisite_analysis_code_sha256: Optional[str]
    _raw_authorization_canonical_json: str
    _verified_loader_version: str
    _loader_token: object = field(repr=False, default=None)

    def __post_init__(self):
        if self._loader_token is not _AUTHORIZATION_LOADER_TOKEN:
            raise FormalAuthorizationIntegrityError(
                "VerifiedFormalManifestAuthorization can only be created by load_and_verify_formal_manifest_"
                "authorization (missing loader token) — hand construction is refused")


def _reverify_authorization_integrity(auth: VerifiedFormalManifestAuthorization) -> dict:
    """Re-derive the evidence hash from the stored canonical JSON and re-parse the critical fields. Raises if
    the object was tampered with after loading. Returns the re-parsed raw mapping."""
    if not isinstance(auth, VerifiedFormalManifestAuthorization):
        raise FormalAuthorizationIntegrityError("authorization must be a VerifiedFormalManifestAuthorization")
    if auth._loader_token is not _AUTHORIZATION_LOADER_TOKEN:
        raise FormalAuthorizationIntegrityError("authorization missing loader token")
    if auth._verified_loader_version != VERIFIED_LOADER_VERSION:
        raise FormalAuthorizationIntegrityError("authorization loader version mismatch")
    recomputed = _sha256_text(auth._raw_authorization_canonical_json)
    if recomputed != auth.authorization_evidence_sha256:
        raise FormalAuthorizationIntegrityError(
            "authorization_evidence_sha256 does not match sha256(_raw_authorization_canonical_json)")
    raw = json.loads(auth._raw_authorization_canonical_json)
    # re-parse critical fields and cross-check against the frozen dataclass copy. NOTE: the verdict->phase
    # SEMANTIC gate (e.g. a test phase must carry the TEST verdict) is enforced by
    # prepare_authorized_formal_phase_in_memory as EarlyTestAccess, not here, so that a test-phase request with
    # the wrong verdict/path is reported as EXPERIMENT_INVALID_EARLY_TEST_ACCESS rather than a generic error.
    if raw.get("verdict") != auth.authorization_verdict:
        raise FormalAuthorizationIntegrityError("verdict re-parse mismatch")
    _require_true(auth.authorizes_formal_manifest_generation, "authorizes_formal_manifest_generation")
    _require_false(auth.authorizes_confirmatory_data_generation, "authorizes_confirmatory_data_generation")
    _require_false(auth.authorizes_confirmatory_run, "authorizes_confirmatory_run")
    _require_true(auth.one_shot, "one_shot")
    return raw


def load_and_verify_formal_manifest_authorization(
        authorization_json_path, *, authorization_commit: str,
        expected_writer_commit: str) -> VerifiedFormalManifestAuthorization:
    """The ONLY way to mint a VerifiedFormalManifestAuthorization. Reads + schema-validates the authorization
    JSON, requires the exact phase-specific verdict and `authorizes_formal_manifest_generation=true`, pins the
    audited writer commit, and binds an evidence hash over the canonical bytes. Rejects the current C
    generator-smoke gate (its `authorizes_formal_manifest_generation=false`)."""
    if not _is_40hex(authorization_commit):
        raise FormalAuthorizationIntegrityError("authorization_commit must be 40 lowercase hex")
    if not _is_40hex(expected_writer_commit):
        raise FormalAuthorizationIntegrityError("expected_writer_commit must be 40 lowercase hex")
    with open(authorization_json_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise FormalAuthorizationIntegrityError("authorization JSON must be an object")

    verdict = raw.get("verdict")
    if verdict not in _VERDICT_PHASE:
        # This is where the current C gate (verdict GENERATOR_SMOKE_GATE_PASS) is refused: it authorizes writer
        # IMPLEMENTATION, never actual generation.
        raise FormalManifestGenerationNotAuthorized(
            f"verdict {verdict!r} does not authorize formal manifest generation "
            f"(need one of {sorted(_VERDICT_PHASE)})")
    phase = _VERDICT_PHASE[verdict]

    # generation must be explicitly authorized; confirmatory data/run must be explicitly refused
    if raw.get("authorizes_formal_manifest_generation") is not True:
        raise FormalManifestGenerationNotAuthorized(
            "authorizes_formal_manifest_generation is not true (writer stays locked)")
    if raw.get("authorizes_confirmatory_data_generation") is not False:
        raise FormalAuthorizationIntegrityError("authorizes_confirmatory_data_generation must be false")
    if raw.get("authorizes_confirmatory_run") is not False:
        raise FormalAuthorizationIntegrityError("authorizes_confirmatory_run must be false")
    if raw.get("one_shot") is not True:
        raise FormalAuthorizationIntegrityError("one_shot must be true")
    if raw.get("power_recertification_required") is not False:
        raise FormalAuthorizationIntegrityError("power_recertification_required must be false")

    # phase / verdict / path consistency — one authorization authorizes EXACTLY one phase
    if raw.get("authorized_phase") != phase:
        raise FormalAuthorizationIntegrityError(
            f"authorized_phase {raw.get('authorized_phase')!r} inconsistent with verdict phase {phase!r}")
    expected_relpath = f"{FORMAL_ARTIFACT_SUBROOT}/{phase}"
    if raw.get("authorized_output_relpath") != expected_relpath:
        raise FormalAuthorizationIntegrityError(
            f"authorized_output_relpath must be {expected_relpath!r}")

    # commits
    for k in ("audited_writer_commit", "protocol_commit", "generator_commit", "runtime_commit"):
        if not _is_40hex(raw.get(k)):
            raise FormalAuthorizationIntegrityError(f"{k} must be 40 lowercase hex")
    if raw["audited_writer_commit"] != expected_writer_commit:
        raise FormalAuthorizationIntegrityError("audited_writer_commit != expected_writer_commit")
    if raw["generator_commit"] != raw["audited_writer_commit"]:
        raise FormalAuthorizationIntegrityError("generator_commit must equal audited_writer_commit")

    # test phase carries the Step-5 model/analysis freeze prerequisites (all 64 hex)
    m_freeze = tv_manifest = analysis_code = None
    if phase == "test":
        m_freeze = raw.get("model_analysis_freeze_sha256")
        tv_manifest = raw.get("train_validation_manifest_sha256")
        analysis_code = raw.get("analysis_code_sha256")
        for name, val in (("model_analysis_freeze_sha256", m_freeze),
                          ("train_validation_manifest_sha256", tv_manifest),
                          ("analysis_code_sha256", analysis_code)):
            if not _is_64hex(val):
                raise FormalAuthorizationIntegrityError(f"test authorization {name} must be 64 lowercase hex")

    canonical = _canon(raw)
    evidence = _sha256_text(canonical)
    return VerifiedFormalManifestAuthorization(
        authorization_verdict=verdict, authorization_evidence_sha256=evidence,
        authorization_commit=authorization_commit, audited_writer_commit=raw["audited_writer_commit"],
        authorized_phase=phase, protocol_commit=raw["protocol_commit"],
        generator_commit=raw["generator_commit"], runtime_commit=raw["runtime_commit"],
        authorized_output_relpath=raw["authorized_output_relpath"],
        authorizes_formal_manifest_generation=True, authorizes_confirmatory_data_generation=False,
        authorizes_confirmatory_run=False, one_shot=True,
        test_prerequisite_model_analysis_freeze_sha256=m_freeze,
        test_prerequisite_train_validation_manifest_sha256=tv_manifest,
        test_prerequisite_analysis_code_sha256=analysis_code,
        _raw_authorization_canonical_json=canonical, _verified_loader_version=VERIFIED_LOADER_VERSION,
        _loader_token=_AUTHORIZATION_LOADER_TOKEN)


# ============================================================================= §8 real commit freeze
@dataclass(frozen=True)
class FormalCommitFreeze:
    protocol_commit: str
    generator_commit: str
    runtime_commit: str
    authorization_commit: str
    authorization_evidence_sha256: str


def validate_formal_commit_freeze(freeze: FormalCommitFreeze,
                                  auth: VerifiedFormalManifestAuthorization) -> None:
    """All commit fields 40-hex, none empty, none a smoke placeholder, and every field EXACTLY equal to the
    verified authorization (incl. generator_commit == audited_writer_commit and evidence binding)."""
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
    # exact agreement with the verified authorization
    if freeze.protocol_commit != auth.protocol_commit:
        raise FormalCommitFreezeError("protocol_commit != authorization")
    if freeze.generator_commit != auth.generator_commit:
        raise FormalCommitFreezeError("generator_commit != authorization")
    if freeze.runtime_commit != auth.runtime_commit:
        raise FormalCommitFreezeError("runtime_commit != authorization")
    if freeze.authorization_commit != auth.authorization_commit:
        raise FormalCommitFreezeError("authorization_commit != authorization")
    if freeze.authorization_evidence_sha256 != auth.authorization_evidence_sha256:
        raise FormalCommitFreezeError("authorization_evidence_sha256 != authorization")
    if freeze.generator_commit != auth.audited_writer_commit:
        raise FormalCommitFreezeError("generator_commit must equal audited_writer_commit")


# --- repository verifiers (resolver-injectable so tests never depend on a future commit existing) -----------
def _git(repo_root, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, text=True)


def verify_commit_exists(repo_root, sha: str, *, resolver: Optional[Callable[[str, str], bool]] = None) -> None:
    if not _is_40hex(sha):
        raise FormalCommitFreezeError(f"commit {sha!r} must be 40 lowercase hex")
    if resolver is not None:
        if not resolver(str(repo_root), sha):
            raise FormalCommitFreezeError(f"commit {sha} not found in {repo_root}")
        return
    res = _git(repo_root, "cat-file", "-e", f"{sha}^{{commit}}")
    if res.returncode != 0:
        raise FormalCommitFreezeError(f"commit {sha} does not exist in {repo_root}")


def verify_clean_worktree(repo_root, *, resolver: Optional[Callable[[str], bool]] = None) -> None:
    if resolver is not None:
        if not resolver(str(repo_root)):
            raise FormalCommitFreezeError("worktree is not clean")
        return
    res = _git(repo_root, "status", "--porcelain")
    if res.returncode != 0:
        raise FormalCommitFreezeError("git status failed")
    if res.stdout.strip():
        raise FormalCommitFreezeError("worktree is not clean (formal generation requires a clean tree)")


def verify_head_or_ancestor_policy(repo_root, *, authorization_commit: str,
                                   head_resolver: Optional[Callable[[str], str]] = None,
                                   ancestor_resolver: Optional[Callable[[str, str, str], bool]] = None) -> None:
    """HEAD must equal the authorization commit or be an explicitly-allowed successor (authorization_commit is
    an ancestor of HEAD). Never silently substitutes the current git HEAD for the pinned commit."""
    if not _is_40hex(authorization_commit):
        raise FormalCommitFreezeError("authorization_commit must be 40 lowercase hex")
    head = head_resolver(str(repo_root)) if head_resolver is not None else \
        _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    if not _is_40hex(head):
        raise FormalCommitFreezeError(f"could not resolve a 40-hex HEAD, got {head!r}")
    if head == authorization_commit:
        return
    if ancestor_resolver is not None:
        ok = ancestor_resolver(str(repo_root), authorization_commit, head)
    else:
        ok = _git(repo_root, "merge-base", "--is-ancestor", authorization_commit, head).returncode == 0
    if not ok:
        raise FormalCommitFreezeError(
            "HEAD is neither the authorization commit nor a descendant of it (pinned-commit policy)")


# ============================================================================= §9 formal environment
class FormalManifestEnvironmentContext:
    """Immutable formal-manifest environment provenance, distinct from the smoke context. Exact 10-key set,
    all non-empty str, the three fixed FORMAL markers, copy-on-read canonical snapshot. Refuses to reuse the
    SMOKE_ONLY execution mode. Enters the phase-manifest full hash via `.values`."""
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
        if d["execution_mode"] == "SMOKE_ONLY":                 # defensive; already excluded by fixed marker
            raise FormalEnvironmentError("formal environment must not reuse the SMOKE_ONLY marker")
        object.__setattr__(self, "_canonical_json", _canon(d))

    def __setattr__(self, *_a):
        raise AttributeError("FormalManifestEnvironmentContext is immutable")

    @property
    def values(self) -> dict:
        return json.loads(self._canonical_json)


# ============================================================================= §10 phase-specific builder
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

    @property
    def unstamped_manifest(self) -> dict:
        return json.loads(self._unstamped_manifest_canonical_json)

    @property
    def stamped_manifest(self) -> dict:
        return json.loads(self._stamped_manifest_canonical_json)

    @property
    def stamped_manifest_bytes(self) -> bytes:
        return self._stamped_manifest_canonical_json.encode("utf-8")


def _stamp_manifest(manifest: dict, *,
                    validate: Callable[[dict], None] = MI.validate_fully_resolved_phase_manifest,
                    full_hash: Callable[[dict], str] = MI.fully_resolved_phase_manifest_hash):
    """Deep-validate + full-hash + flat self-hash stamp + canonical serialize. Returns
    (stamped_dict, full_manifest_sha256, stamped_canonical_json, stamped_file_sha256). The scientific anchor
    (full_manifest_sha256, over the UN-stamped payload) is kept separate from stamped_file_sha256 (bytes of
    the serialized stamped file). Injectable validate/hash keep this unit-testable with a TEST_ONLY payload."""
    validate(manifest)
    h = full_hash(manifest)
    if not _is_64hex(h):
        raise FormalBundleWriteError("full manifest hash is not 64 lowercase hex")
    stamped = copy.deepcopy(manifest)
    stamped[MI.SELF_HASH_FIELD] = h
    # verify the stamped flat self-field equals the recomputed hash over the un-stamped payload
    check = copy.deepcopy(stamped)
    check.pop(MI.SELF_HASH_FIELD, None)
    if _canon(check) != _canon(manifest):
        raise FormalBundleWriteError("stamping altered the manifest payload")
    stamped_json = _canon(stamped)                              # allow_nan=False rejects non-finite values
    stamped_file_sha = _sha256_text(stamped_json)
    if stamped[MI.SELF_HASH_FIELD] != h:
        raise FormalBundleWriteError("stamped self-hash field mismatch")
    return stamped, h, stamped_json, stamped_file_sha


def prepare_authorized_formal_phase_in_memory(
        *, authorization: VerifiedFormalManifestAuthorization, commits: FormalCommitFreeze,
        environment_versions: FormalManifestEnvironmentContext,
        repo_root=None, commit_resolver=None, clean_resolver=None,
        head_resolver=None, ancestor_resolver=None) -> PreparedFormalPhase:
    """Strict 14-step construction of an authorized formal phase (in memory only; nothing is written). Every
    gate precedes the phase construction, so an unauthorized or inconsistent request never builds a manifest.
    repo verifiers accept injectable resolvers so tests never depend on a real future commit."""
    # 1 verify authorization object integrity
    _reverify_authorization_integrity(authorization)
    # 2 verify authorization permits exactly one KNOWN phase
    phase = authorization.authorized_phase
    if phase not in ("train_validation", "test"):
        raise FormalAuthorizationIntegrityError(f"unknown authorized_phase {phase!r}")
    # 2b generation must be authorized (fail-closed default for everything else)
    if authorization.authorizes_formal_manifest_generation is not True:
        raise FormalManifestGenerationNotAuthorized("FORMAL_MANIFEST_GENERATION_NOT_AUTHORIZED")
    if phase == "test":
        # Step-5 gate: a test manifest must not be constructible before the model/analysis freeze. Wrong
        # verdict, wrong output path, or a missing 64-hex prerequisite => EXPERIMENT_INVALID_EARLY_TEST_ACCESS.
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
    else:  # train_validation
        if authorization.authorization_verdict != TRAIN_VALIDATION_VERDICT:
            raise FormalAuthorizationIntegrityError("train_validation phase requires the TRAIN_VALIDATION verdict")
    # 3 verify commit freeze exact match
    validate_formal_commit_freeze(commits, authorization)
    # 4 verify repository state (only if a repo_root is supplied; pinned-commit policy, never auto-HEAD)
    if repo_root is not None:
        verify_commit_exists(repo_root, commits.generator_commit, resolver=commit_resolver)
        verify_commit_exists(repo_root, commits.protocol_commit, resolver=commit_resolver)
        verify_commit_exists(repo_root, commits.runtime_commit, resolver=commit_resolver)
        verify_clean_worktree(repo_root, resolver=clean_resolver)
        verify_head_or_ancestor_policy(repo_root, authorization_commit=commits.authorization_commit,
                                       head_resolver=head_resolver, ancestor_resolver=ancestor_resolver)
    # 5 verify formal env
    if not isinstance(environment_versions, FormalManifestEnvironmentContext):
        raise FormalEnvironmentError("environment_versions must be a FormalManifestEnvironmentContext")
    env_values = environment_versions.values
    if env_values.get("execution_mode") != "FORMAL_MANIFEST_GENERATION":
        raise FormalEnvironmentError("formal environment must carry execution_mode=FORMAL_MANIFEST_GENERATION")
    # reject any smoke-only leakage into the commit context (§11)
    if (commits.protocol_commit, commits.generator_commit, commits.runtime_commit) == _SMOKE_PLACEHOLDER:
        raise FormalCommitFreezeError("formal build refuses smoke placeholder commits")
    # 6 mandatory BS KAT (first real construction work)
    BS.require_known_answer_compatibility()
    # 7 independently construct phase from frozen modules (reuse the C-gate-approved pure construction)
    commit_ctx = GEN.CommitContext(commits.protocol_commit, commits.generator_commit, commits.runtime_commit)
    manifest = GEN._build_unsealed_manifest(phase, commit_ctx, environment_versions)
    # guard: the constructed manifest must not carry any smoke marker
    if manifest["environment_versions"].get("execution_mode") == "SMOKE_ONLY":
        raise FormalEnvironmentError("constructed manifest carries SMOKE_ONLY marker")
    # 8-13 deep validate -> full hash -> stamp -> verify -> serialize -> stamped file hash
    stamped, full_hash, stamped_json, stamped_file_sha = _stamp_manifest(manifest)
    # 14 return immutable PreparedFormalPhase
    return PreparedFormalPhase(
        phase=phase, _unstamped_manifest_canonical_json=_canon(manifest),
        _stamped_manifest_canonical_json=stamped_json, full_manifest_sha256=full_hash,
        stamped_file_sha256=stamped_file_sha, authorization_evidence_sha256=authorization.authorization_evidence_sha256,
        protocol_commit=commits.protocol_commit, generator_commit=commits.generator_commit,
        runtime_commit=commits.runtime_commit)


# ============================================================================= §12/§13/§14 atomic bundle write
@dataclass(frozen=True)
class FormalWriteReceipt:
    phase: str
    full_manifest_sha256: str
    stamped_file_sha256: str
    authorization_evidence_sha256: str
    authorization_commit: str
    protocol_commit: str
    generator_commit: str
    runtime_commit: str
    final_dir: str
    write_complete: bool


def _phase_manifest_sha256_line(stamped_file_sha: str) -> str:
    # sha256sum format: "<hash><two spaces><filename>\n"
    return f"{stamped_file_sha}  phase_manifest.json\n"


def _bundle_payloads(prepared: PreparedFormalPhase,
                     auth: VerifiedFormalManifestAuthorization) -> dict:
    receipt = {
        "receipt_schema_version": RECEIPT_SCHEMA_VERSION, "phase": prepared.phase,
        "full_manifest_sha256": prepared.full_manifest_sha256,
        "stamped_file_sha256": prepared.stamped_file_sha256,
        "authorization_evidence_sha256": prepared.authorization_evidence_sha256,
        "authorization_commit": auth.authorization_commit, "protocol_commit": prepared.protocol_commit,
        "generator_commit": prepared.generator_commit, "runtime_commit": prepared.runtime_commit,
        "one_shot": True, "formal_manifest_written": True,
        "confirmatory_data_authorized": False, "confirmatory_run_authorized": False,
    }
    sealed = {"phase": prepared.phase, "full_manifest_sha256": prepared.full_manifest_sha256,
              "stamped_file_sha256": prepared.stamped_file_sha256, "write_complete": True}
    return {
        "phase_manifest.json": prepared.stamped_manifest_bytes,
        "phase_manifest.sha256": _phase_manifest_sha256_line(prepared.stamped_file_sha256).encode("utf-8"),
        "write_receipt.json": (_canon(receipt) + "\n").encode("utf-8"),
        "SEALED": (_canon(sealed) + "\n").encode("utf-8"),
    }


def _resolve_contained_final_dir(output_root, relpath: str, phase: str) -> Path:
    """§14 path safety: relpath must be relative, no '..', land inside output_root/artifacts/formal/
    confirmatory_v4/<phase>, and not escape via symlink."""
    if not isinstance(relpath, str) or not relpath:
        raise FormalPathSafetyError("authorized_output_relpath must be a non-empty relative path")
    if os.path.isabs(relpath) or relpath.startswith(("/", "\\")):
        raise FormalPathSafetyError("authorized_output_relpath must not be absolute")
    parts = Path(relpath).parts
    if ".." in parts:
        raise FormalPathSafetyError("authorized_output_relpath must not contain '..'")
    if relpath != f"{FORMAL_ARTIFACT_SUBROOT}/{phase}":
        raise FormalPathSafetyError(
            f"authorized_output_relpath must be exactly {FORMAL_ARTIFACT_SUBROOT}/{phase}")
    root = Path(output_root).resolve()
    final_dir = root / relpath
    if final_dir.name != phase:
        raise FormalPathSafetyError("final directory name must equal the phase")
    # symlink-escape containment: the longest existing prefix of the target must resolve inside the repo root
    existing = final_dir
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    resolved_existing = existing.resolve()
    if resolved_existing != root and root not in resolved_existing.parents:
        raise FormalPathSafetyError("authorized output path escapes the repo root (symlink/containment)")
    return final_dir


def _fsync_path(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_bundle(final_dir: Path, files: dict) -> None:
    """Low-level one-shot atomic bundle write. final_dir must NOT exist. Writes all files into a unique staging
    dir under the same parent (flush+fsync each), fsyncs the staging dir, reads back + verifies bytes, atomically
    renames staging->final, fsyncs the parent, re-reads the final dir. Any failure removes the staging dir and
    leaves no final dir and no existing file changed."""
    final_dir = Path(final_dir)
    if final_dir.exists():
        raise FormalManifestAlreadyExists(
            f"FORMAL_MANIFEST_ALREADY_EXISTS: {final_dir} already exists (one-shot create; no overwrite)")
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".formal_staging_", dir=str(parent)))
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
        # read-back verify every file's bytes match
        for name, data in files.items():
            with open(staging / name, "rb") as fh:
                got = fh.read()
            if got != data:
                raise FormalBundleWriteError(f"read-back mismatch for {name} before rename")
        if final_dir.exists():                                  # last-moment collision guard
            raise FormalManifestAlreadyExists(f"FORMAL_MANIFEST_ALREADY_EXISTS: {final_dir}")
        os.replace(str(staging), str(final_dir))                # atomic dir rename (same filesystem/parent)
        staging = None
        _fsync_path(parent)
        # final read-back verify
        for name, data in files.items():
            with open(final_dir / name, "rb") as fh:
                if fh.read() != data:
                    raise FormalBundleWriteError(f"final read-back mismatch for {name}")
    finally:
        if staging is not None and Path(staging).exists():
            for child in Path(staging).iterdir():
                try:
                    child.unlink()
                except OSError:
                    pass
            try:
                os.rmdir(str(staging))
            except OSError:
                pass


def write_authorized_formal_manifest_bundle_atomic(
        prepared: PreparedFormalPhase, *, authorization: VerifiedFormalManifestAuthorization,
        output_root) -> FormalWriteReceipt:
    """Production one-shot atomic bundle write. Re-verifies the authorization + the prepared<->auth<->commit<->
    hash binding, enforces path containment, and refuses to overwrite an existing bundle. In the current repo no
    VerifiedFormalManifestAuthorization can be minted, so this is unreachable via any real call chain."""
    _reverify_authorization_integrity(authorization)
    if authorization.authorizes_formal_manifest_generation is not True:
        raise FormalManifestGenerationNotAuthorized("FORMAL_MANIFEST_GENERATION_NOT_AUTHORIZED")
    if not isinstance(prepared, PreparedFormalPhase):
        raise FormalBundleWriteError("prepared must be a PreparedFormalPhase")
    if prepared.phase != authorization.authorized_phase:
        raise FormalBundleWriteError("prepared.phase != authorization.authorized_phase")
    if prepared.authorization_evidence_sha256 != authorization.authorization_evidence_sha256:
        raise FormalBundleWriteError("prepared authorization evidence mismatch")
    if (prepared.protocol_commit != authorization.protocol_commit
            or prepared.generator_commit != authorization.generator_commit
            or prepared.runtime_commit != authorization.runtime_commit):
        raise FormalBundleWriteError("prepared commit fields disagree with authorization")
    # re-verify the stamped bytes hash and the self-hash binding
    if _sha256_bytes(prepared.stamped_manifest_bytes) != prepared.stamped_file_sha256:
        raise FormalBundleWriteError("stamped_file_sha256 does not match the prepared bytes")
    if prepared.stamped_manifest.get(MI.SELF_HASH_FIELD) != prepared.full_manifest_sha256:
        raise FormalBundleWriteError("stamped self-hash field does not match full_manifest_sha256")
    final_dir = _resolve_contained_final_dir(output_root, authorization.authorized_output_relpath, prepared.phase)
    files = _bundle_payloads(prepared, authorization)
    _atomic_write_bundle(final_dir, files)
    return FormalWriteReceipt(
        phase=prepared.phase, full_manifest_sha256=prepared.full_manifest_sha256,
        stamped_file_sha256=prepared.stamped_file_sha256,
        authorization_evidence_sha256=prepared.authorization_evidence_sha256,
        authorization_commit=authorization.authorization_commit, protocol_commit=prepared.protocol_commit,
        generator_commit=prepared.generator_commit, runtime_commit=prepared.runtime_commit,
        final_dir=str(final_dir), write_complete=True)


__all__ = [
    "VERIFIED_LOADER_VERSION", "RECEIPT_SCHEMA_VERSION", "FORMAL_ENV_VERSION_KEYS", "FORMAL_ENV_FIXED",
    "TRAIN_VALIDATION_VERDICT", "TEST_VERDICT", "FORMAL_ARTIFACT_SUBROOT", "BUNDLE_FILES",
    "FormalManifestGenerationNotAuthorized", "FormalAuthorizationIntegrityError", "FormalCommitFreezeError",
    "FormalEnvironmentError", "EarlyTestAccessError", "FormalPathSafetyError", "FormalManifestAlreadyExists",
    "FormalBundleWriteError", "VerifiedFormalManifestAuthorization",
    "load_and_verify_formal_manifest_authorization", "FormalCommitFreeze", "validate_formal_commit_freeze",
    "verify_commit_exists", "verify_clean_worktree", "verify_head_or_ancestor_policy",
    "FormalManifestEnvironmentContext", "PreparedFormalPhase", "prepare_authorized_formal_phase_in_memory",
    "FormalWriteReceipt", "write_authorized_formal_manifest_bundle_atomic",
]
