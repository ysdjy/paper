"""Confirmatory v4 phase-manifest GENERATOR (offline; KAT-gated; smoke-only authorized).

Builds the IN-MEMORY fully-resolved phase manifest for a phase (train_validation | test) from the FROZEN
active modules ONLY -- it re-defines no scientific constant:
  PRE = preregistration_v4              (frozen seeds, SECRET_DENYLIST, PRIMARY_SUCCESS)
  ID  = confirmatory_v4_identity        (identities, subseeds, storage/execution order, structure hash)
  BS  = confirmatory_v4_block_state     (UNIQUE frozen block-state sampler; mandatory KAT gate)
  MI  = confirmatory_v4_manifest_integrity (schema/config/env, DEEP validator, full-manifest hash)

Closes GEN-B-001..007 (Claude B generator+smoke audit):
  001 GeneratedPhase is an IMMUTABLE canonical-JSON snapshot (copy-on-read; hash bound to the snapshot).
  002 TrialExecutionEnvelope public/secret are IMMUTABLE snapshots (post-scan injection impossible).
  003 no unfrozen `target_open_position=0.20` in the general envelope (removed; not a frozen value).
  004 smoke build REQUIRES the placeholder commit context (machine-unmistakable from a formal freeze).
  005 EnvironmentVersionContext validates + copies (immutable); the 10-key smoke provenance contract is a
      CLI-layer gate (`validate_smoke_environment_provenance`), keeping build permissive for reference eq.
  006 strict selection domain + evidence-hash + candidate-completion FSM (candidate outcomes never fed in).
  007 CLI writes the SMOKE_ONLY summary atomically (temp+fsync+os.replace; read-back; idempotent; collision).

Hard rules kept: KAT is the first real work; block state ONLY via BS.resolved_block_state (unchecked core
never called); MI.reference_phase_manifest is NOT wrapped; formal writer refuses with zero side effects; no
Isaac, no model, no confirmatory data, no formal manifest written by this module (this module writes no files).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from deployment_calibration.offline_v2.calibration_bias import preregistration_v4 as PRE
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as ID
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_block_state as BS
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI

_GIT_SHA_LEN = 40
SMOKE_PLACEHOLDER = ("a" * 40, "b" * 40, "c" * 40)   # protocol / generator / runtime (smoke only)
EXECUTION_ORDER_DEFINITION = ("split order -> resolve_block_order -> resolve_session_order -> probe first "
                              "-> resolve_candidate_order")
_ABS_TOL = 1e-12

# GEN-B-005: frozen smoke provenance contract (a CLI-layer guarantee; not a scientific-design parameter).
SMOKE_ENV_VERSION_KEYS = ("python_implementation", "python_version", "numpy_version", "torch_version",
                          "os_system", "os_release", "machine", "isaac_status", "gpu_status", "execution_mode")
SMOKE_ENV_FIXED = {"execution_mode": "SMOKE_ONLY", "isaac_status": "NOT_IMPORTED_NOT_LAUNCHED",
                   "gpu_status": "NOT_USED_CPU_SMOKE"}


# ----------------------------------------------------------------------- errors
class FormalGenerationNotAuthorized(RuntimeError):
    verdict = "GENERATOR_FORMAL_MODE_NOT_AUTHORIZED"


class CommitContextError(ValueError):
    verdict = "GENERATOR_COMMIT_CONTEXT_INVALID"


class SmokeCommitContextError(CommitContextError):
    verdict = "GENERATOR_SMOKE_COMMIT_CONTEXT_REQUIRED"


class PublicPayloadLeakageError(ValueError):
    verdict = "GENERATOR_PUBLIC_PAYLOAD_LEAKAGE"


class SelectionOrderError(RuntimeError):
    verdict = "GENERATOR_SELECTION_ORDER_VIOLATION"


class EnvironmentProvenanceError(ValueError):
    verdict = "GENERATOR_SMOKE_ENVIRONMENT_PROVENANCE_INVALID"


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


# ----------------------------------------------------------------------- immutable env context (GEN-B-005)
class EnvironmentVersionContext:
    """Immutable environment-version provenance. Validates + deep-copies at construction (rejects empty
    mapping / non-str keys / bool / non-str / empty-string values); caller mutation cannot affect it. The
    EXACT 10-key smoke contract is enforced separately by `validate_smoke_environment_provenance`."""
    __slots__ = ("_canonical_json",)

    def __init__(self, values: Mapping[str, str]):
        if not isinstance(values, Mapping):
            raise EnvironmentProvenanceError(f"environment_versions must be a mapping, got {type(values).__name__}")
        d = dict(values)
        if not d:
            raise EnvironmentProvenanceError("environment_versions must be a non-empty mapping")
        for k, v in d.items():
            if not isinstance(k, str) or not k:
                raise EnvironmentProvenanceError(f"environment_versions key must be a non-empty str, got {k!r}")
            if isinstance(v, bool) or not isinstance(v, str) or not v:
                raise EnvironmentProvenanceError(f"environment_versions[{k!r}] must be a non-empty str, got {v!r}")
        object.__setattr__(self, "_canonical_json", _canon(d))

    def __setattr__(self, *_a):
        raise AttributeError("EnvironmentVersionContext is immutable")

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> "EnvironmentVersionContext":
        return cls(values)

    @property
    def values(self) -> dict:
        return json.loads(self._canonical_json)


def validate_smoke_environment_provenance(env: "EnvironmentVersionContext") -> None:
    """GEN-B-005: enforce the EXACT 10-key smoke provenance contract (key set + non-empty str values + the
    three fixed markers). Enforced by the PUBLIC `build_phase_manifest_in_memory` entry point itself (and by
    the CLI smoke path): a caller cannot bypass the CLI and obtain a phase hash with partial provenance."""
    if not isinstance(env, EnvironmentVersionContext):
        raise EnvironmentProvenanceError("env must be an EnvironmentVersionContext")
    v = env.values
    if set(v) != set(SMOKE_ENV_VERSION_KEYS):
        raise EnvironmentProvenanceError(f"smoke environment key set != frozen {SMOKE_ENV_VERSION_KEYS} "
                                         f"(got {sorted(v)})")
    for k in SMOKE_ENV_VERSION_KEYS:
        if not isinstance(v[k], str) or not v[k]:
            raise EnvironmentProvenanceError(f"smoke environment[{k!r}] must be a non-empty str")
    for k, expect in SMOKE_ENV_FIXED.items():
        if v[k] != expect:
            raise EnvironmentProvenanceError(f"smoke environment[{k!r}] must be {expect!r}, got {v[k]!r}")


# ----------------------------------------------------------------------- dataclasses
@dataclass(frozen=True)
class GeneratorAuthorization:
    smoke_only: bool
    formal_manifest_generation_authorized: bool = False
    confirmatory_data_generation_authorized: bool = False
    confirmatory_run_authorized: bool = False


@dataclass(frozen=True)
class CommitContext:
    protocol_commit: str
    generator_commit: str
    runtime_commit: str


@dataclass(frozen=True)
class GeneratedPhase:
    """GEN-B-001: immutable snapshot. The validated manifest + kat report are stored as canonical JSON; the
    `unsealed_manifest`/`kat_report` properties return an INDEPENDENT deep copy each access, so a caller can
    never mutate the internal payload that `full_manifest_sha256` is bound to."""
    phase: Literal["train_validation", "test"]
    _manifest_canonical_json: str
    full_manifest_sha256: str
    trial_count: int
    smoke_only: bool
    _kat_report_canonical_json: str

    @property
    def unsealed_manifest(self) -> dict:
        return json.loads(self._manifest_canonical_json)      # fresh copy-on-read

    @property
    def kat_report(self) -> dict:
        return json.loads(self._kat_report_canonical_json)

    @property
    def canonical_manifest_json(self) -> str:
        return self._manifest_canonical_json


@dataclass(frozen=True)
class TrialExecutionEnvelope:
    """GEN-B-002: immutable snapshot; public/secret returned as fresh copies each access."""
    canonical_trial_identity: str
    planned_episode_id: str
    resume_key: str
    execution_order_index: int
    _public_trial_json: str
    _secret_environment_json: str
    smoke_only: bool

    @property
    def public_trial_spec(self) -> dict:
        return json.loads(self._public_trial_json)

    @property
    def secret_environment_spec(self) -> dict:
        return json.loads(self._secret_environment_json)


# ----------------------------------------------------------------------- gates
def require_generator_authorization(auth: GeneratorAuthorization) -> None:
    if not isinstance(auth, GeneratorAuthorization):
        raise FormalGenerationNotAuthorized("authorization must be a GeneratorAuthorization")
    if auth.smoke_only is not True:
        raise FormalGenerationNotAuthorized("smoke_only must be True (no default formal mode)")
    if (auth.formal_manifest_generation_authorized or auth.confirmatory_data_generation_authorized
            or auth.confirmatory_run_authorized):
        raise FormalGenerationNotAuthorized(
            "formal manifest / confirmatory data / confirmatory run are NOT authorized in this phase")


def require_formal_authorization(auth: GeneratorAuthorization) -> None:
    if not (isinstance(auth, GeneratorAuthorization) and auth.formal_manifest_generation_authorized
            and not auth.smoke_only):
        raise FormalGenerationNotAuthorized(
            "formal phase-manifest writing is NOT authorized (smoke-only phase); refusing with zero side effects")


def _is_40_hex(s) -> bool:
    return isinstance(s, str) and len(s) == _GIT_SHA_LEN and all(c in "0123456789abcdef" for c in s)


def validate_commit_context(ctx: CommitContext) -> None:
    if not isinstance(ctx, CommitContext):
        raise CommitContextError("commit context must be a CommitContext")
    for name in ("protocol_commit", "generator_commit", "runtime_commit"):
        v = getattr(ctx, name)
        if not _is_40_hex(v):
            raise CommitContextError(f"{name} must be 40 lowercase hex, got {v!r}")


def commit_context_is_smoke_placeholder(ctx: CommitContext) -> bool:
    return (ctx.protocol_commit, ctx.generator_commit, ctx.runtime_commit) == SMOKE_PLACEHOLDER


def smoke_placeholder_commits() -> CommitContext:
    return CommitContext(*SMOKE_PLACEHOLDER)


def require_smoke_commit_context(ctx: CommitContext) -> None:
    """GEN-B-004: under smoke authorization the commit context MUST be the placeholder (a real-looking commit
    would make a smoke manifest machine-indistinguishable from a future formal freeze)."""
    if not commit_context_is_smoke_placeholder(ctx):
        raise SmokeCommitContextError(
            "smoke-only build requires the placeholder commit context (a/b/c*40); real commit freezing is a "
            "separate future gate not open in this phase")


# ----------------------------------------------------------------------- production builder
def _phase_splits(phase: str):
    if phase not in MI.PHASE_SPLITS:
        raise ValueError(f"unknown phase {phase!r}")
    return MI.PHASE_SPLITS[phase]


def _build_unsealed_manifest(phase: str, commits: CommitContext,
                             environment_versions: EnvironmentVersionContext) -> dict:
    """Independent construction (NOT MI.reference_phase_manifest) from the frozen modules. Blocks/sessions in
    canonical STORAGE order; block state from the frozen sampler; execution_order_index from the seed plan."""
    splits = _phase_splits(phase)
    blocks = []
    for split in splits:
        for bi in ID.BLOCKS[split]:
            bst = BS.resolved_block_state(split, bi)               # frozen sampler (KAT-gated); no placeholder
            blocks.append({
                "split": split, "block_index": bi,
                "canonical_block_identity": ID.block_identity(split, bi),
                "residual_value": bst["residual_value"], "nuisance_values": bst["nuisance_values"],
                "residual_subseed": bst["residual_subseed"], "nuisance_subseed": bst["nuisance_subseed"],
                "block_order_key": ID.block_order_key(split, bi),
            })
    sessions = []
    for split in splits:
        for bi in ID.BLOCKS[split]:
            for nom in sorted(ID.NOMINALS[split]):
                sessions.append({
                    "canonical_session_identity": ID.session_identity(split, bi, nom),
                    "split": split, "block_index": bi, "nominal_bias": nom,
                    "session_order_key": ID.session_order_subseed(split, bi, nom),
                    "nominal_order_key": ID.nominal_order_subseed(split, bi, nom),
                    "candidate_order_keys": ID.candidate_order_key_map(split, bi, nom),
                    "resolved_candidate_order": list(ID.resolve_candidate_order(split, bi, nom)),
                })
    meta = {r["canonical_trial_identity"]: r for r in ID.enumerate_trials() if r["split"] in splits}
    exec_index = {tid: i for i, tid in enumerate(ID.resolve_phase_execution_plan(phase))}
    trials = []
    for tid in ID.canonical_phase_trial_identities(phase):
        r = meta[tid]
        pid = ID.planned_episode_id(tid)
        trials.append({
            "canonical_trial_identity": tid, "planned_episode_id": pid,
            "session_ref": ID.session_identity(r["split"], r["block_index"], r["nominal"]),
            "role": r["role"], "offset": r["offset"],
            "trial_init_subseed": ID.trial_init_subseed(r["split"], r["block_index"], r["nominal"],
                                                        r["role"], r["offset"]),
            "execution_order_index": exec_index[tid], "resume_key": pid,
        })
    exp = MI.PHASES[phase]
    return {
        "schema_version": MI.PHASE_MANIFEST_SCHEMA_VERSION, "phase": phase,
        "protocol_commit": commits.protocol_commit, "generator_commit": commits.generator_commit,
        "runtime_commit": commits.runtime_commit, "config_sha256": MI.canonical_config_sha256(),
        "planned_structure_sha256": ID.canonical_planned_structure_hash(), "frozen_seeds": PRE.frozen_seeds(),
        "deterministic_environment": MI.deterministic_environment_contract(),
        "environment_versions": dict(environment_versions.values),
        "counts": {"blocks": exp["blocks"], "sessions": exp["sessions"], "trials": exp["trials"]},
        "execution_order_definition": EXECUTION_ORDER_DEFINITION,
        "manifest_algorithm_version": MI.MANIFEST_ALGORITHM_VERSION,
        "blocks": blocks, "sessions": sessions, "trials": trials,
    }


def build_phase_manifest_in_memory(phase: Literal["train_validation", "test"], *,
                                   auth: GeneratorAuthorization, commits: CommitContext,
                                   environment_versions: EnvironmentVersionContext) -> GeneratedPhase:
    """Frozen order: auth gate -> mandatory KAT -> 40-hex commit -> smoke-placeholder commit ->
    EXACT 10-key smoke provenance gate -> build -> DEEP validate -> full hash -> IMMUTABLE snapshot."""
    require_generator_authorization(auth)                      # 1
    kat = BS.require_known_answer_compatibility()              # 2  MANDATORY KAT (first real work)
    validate_commit_context(commits)                          # 3  40-hex format
    require_smoke_commit_context(commits)                     # 3b GEN-B-004 placeholder-only under smoke
    if not isinstance(environment_versions, EnvironmentVersionContext):
        raise EnvironmentProvenanceError("environment_versions must be an EnvironmentVersionContext")
    validate_smoke_environment_provenance(environment_versions)   # 3c GEN-B-005 EXACT 10-key provenance
    #     enforced at the PUBLIC builder itself (not merely the CLI): a caller cannot bypass the CLI and
    #     obtain a phase hash with partial provenance. Runs BEFORE any manifest construction / hashing.
    manifest = _build_unsealed_manifest(phase, commits, environment_versions)   # 4-7
    MI.validate_fully_resolved_phase_manifest(manifest)       # 8  (re-runs KAT + recomputes residual/nuisance)
    full_hash = MI.fully_resolved_phase_manifest_hash(manifest)   # 9
    return GeneratedPhase(phase=phase, _manifest_canonical_json=_canon(manifest),   # 10 immutable snapshot
                          full_manifest_sha256=full_hash, trial_count=len(manifest["trials"]),
                          smoke_only=bool(auth.smoke_only), _kat_report_canonical_json=_canon(dict(kat)))


# ----------------------------------------------------------------------- public/secret envelope (GEN-B-002/003)
def _denylist_tokens() -> set:
    toks = set()
    for entry in PRE.SECRET_DENYLIST:
        toks.add(str(entry).split(" ")[0].strip())
    toks.update({"nominal_bias", "residual_bias", "actual_bias", "eff_signed", "abs_eff",
                 "residual_value", "residual_subseed", "nuisance_subseed", "nuisance_values",
                 "future_candidate_outcomes", "oracle_action", "hidden_state_id",
                 "secret_deployment_state", "block_seed", "residual_seed", "tau_minus_abs_eff"})
    return toks


def assert_no_secret_in_public(public: Mapping[str, Any]) -> None:
    forbidden = _denylist_tokens()

    def scan(obj, path="public"):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in forbidden:
                    raise PublicPayloadLeakageError(f"secret/denylist key {k!r} present at {path}")
                scan(v, f"{path}.{k}")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                scan(v, f"{path}[{i}]")

    scan(public)


def build_trial_execution_envelope(phase: Literal["train_validation", "test"], canonical_trial_identity: str,
                                   *, auth: GeneratorAuthorization) -> TrialExecutionEnvelope:
    """Public (runtime-legal) vs secret (audit-only) split for one trial, both frozen as immutable snapshots.
    Public carries ONLY frozen identity/order fields (GEN-B-003: no unfrozen target level); the block hidden
    state lives ONLY in secret. KAT-gated via the block state."""
    require_generator_authorization(auth)
    BS.require_known_answer_compatibility()
    splits = _phase_splits(phase)
    meta = {r["canonical_trial_identity"]: r for r in ID.enumerate_trials() if r["split"] in splits}
    r = meta.get(canonical_trial_identity)
    if r is None:
        raise ValueError(f"trial identity {canonical_trial_identity!r} not in phase {phase!r}")
    pid = ID.planned_episode_id(canonical_trial_identity)
    exec_index = {tid: i for i, tid in enumerate(ID.resolve_phase_execution_plan(phase))}
    bst = BS.resolved_block_state(r["split"], r["block_index"])
    public = {
        "canonical_trial_identity": canonical_trial_identity, "planned_episode_id": pid, "resume_key": pid,
        "role": r["role"], "offset": r["offset"],
    }
    assert_no_secret_in_public(public)                         # scan the mutable payload, THEN freeze
    secret = {
        "split": r["split"], "block_index": r["block_index"], "nominal_bias": r["nominal"],
        "residual_bias": bst["residual_value"], "nuisance_values": bst["nuisance_values"],
        "residual_subseed": bst["residual_subseed"], "nuisance_subseed": bst["nuisance_subseed"],
    }
    return TrialExecutionEnvelope(canonical_trial_identity=canonical_trial_identity, planned_episode_id=pid,
                                  resume_key=pid, execution_order_index=exec_index[canonical_trial_identity],
                                  _public_trial_json=_canon(public), _secret_environment_json=_canon(secret),
                                  smoke_only=bool(auth.smoke_only))


# ----------------------------------------------------------------------- strict selection helpers (GEN-B-006)
def canonical_selected_offset(offset) -> float:
    """Strict: reject bool / non-numeric (incl. numeric strings) / non-finite; accept only within abs_tol
    1e-12 of a frozen candidate-bank offset; return the exact frozen float."""
    if isinstance(offset, bool) or not isinstance(offset, (int, float)):
        raise SelectionOrderError(f"selected offset must be a real number, got {offset!r}")
    fv = float(offset)
    if not math.isfinite(fv):
        raise SelectionOrderError(f"selected offset must be finite, got {offset!r}")
    for a in ID.CANDIDATE_BANK:
        if math.isclose(fv, a, rel_tol=0.0, abs_tol=_ABS_TOL):
            return a
    raise SelectionOrderError(f"selected offset {offset!r} not in candidate bank {ID.CANDIDATE_BANK}")


def _validate_evidence_hash(evidence_hash) -> str:
    if not isinstance(evidence_hash, str) or len(evidence_hash) != 64 \
            or any(c not in "0123456789abcdef" for c in evidence_hash):
        raise SelectionOrderError("evidence_hash must be a 64-char lowercase hex sha256")
    return evidence_hash


# ----------------------------------------------------------------------- session run-state interface (no runtime)
SESSION_STATES = ("PLANNED", "PROBE_READY", "PROBE_COMPLETE", "SELECTION_FROZEN", "CANDIDATES_READY",
                  "SESSION_COMPLETE")


@dataclass
class SessionRunController:
    """PURE state interface (no Isaac): probe-first, selection frozen (from probe evidence) BEFORE any
    candidate outcome, and candidate COMPLETION recorded in the frozen resolved order. Candidate OUTCOMES are
    never fed to the selector."""
    session_id: str
    resolved_candidate_order: tuple = ()
    state: str = "PLANNED"
    _selected_offset: float | None = field(default=None)
    _evidence_hash: str | None = field(default=None)
    _completed: list = field(default_factory=list)

    def probe_ready(self):
        self._require("PLANNED"); self.state = "PROBE_READY"

    def probe_complete(self):
        self._require("PROBE_READY"); self.state = "PROBE_COMPLETE"

    def freeze_selection(self, selected_offset, evidence_hash):
        """Frozen from probe evidence ONLY, before any candidate outcome. Strict offset + evidence hash."""
        self._require("PROBE_COMPLETE")
        self._selected_offset = canonical_selected_offset(selected_offset)   # GEN-B-006 strict
        self._evidence_hash = _validate_evidence_hash(evidence_hash)
        self.state = "SELECTION_FROZEN"

    def candidates_ready(self):
        self._require("SELECTION_FROZEN")
        order = tuple(canonical_selected_offset(o) for o in self.resolved_candidate_order)
        if len(order) != len(ID.CANDIDATE_BANK) or sorted(order) != sorted(ID.CANDIDATE_BANK):
            raise SelectionOrderError(f"resolved_candidate_order {order} is not a permutation of the bank")
        self.state = "CANDIDATES_READY"

    def record_candidate_complete(self, offset, *, planned_episode_id: str, attempt_index: int) -> str:
        """Record ONE candidate execution completion (identity only; NO outcome). Must be the next expected
        candidate in the frozen resolved order; attempt in {0,1}; returns the attempt id."""
        self._require("CANDIDATES_READY")
        off = canonical_selected_offset(offset)
        if len(self._completed) >= len(self.resolved_candidate_order):
            raise SelectionOrderError("all candidates already completed")
        expected = canonical_selected_offset(self.resolved_candidate_order[len(self._completed)])
        if not math.isclose(off, expected, rel_tol=0.0, abs_tol=_ABS_TOL):
            raise SelectionOrderError(f"candidate {off} out of resolved order; expected {expected}")
        if any(math.isclose(off, c, rel_tol=0.0, abs_tol=_ABS_TOL) for c in self._completed):
            raise SelectionOrderError(f"candidate {off} already completed")
        aid = ID.attempt_id(planned_episode_id, attempt_index)   # validates PID + attempt in {0,1}
        self._completed.append(off)
        return aid

    def session_complete(self):
        self._require("CANDIDATES_READY")
        if len(self._completed) != len(ID.CANDIDATE_BANK):
            raise SelectionOrderError(f"session_complete requires all {len(ID.CANDIDATE_BANK)} candidates "
                                      f"completed, got {len(self._completed)}")
        self.state = "SESSION_COMPLETE"

    def offer_candidate_outcome_to_selector(self, *_a, **_k):
        raise SelectionOrderError("candidate outcomes may NOT enter the selector (frozen before candidates)")

    def attempt_id(self, planned_episode_id: str, attempt_index: int) -> str:
        return ID.attempt_id(planned_episode_id, attempt_index)

    def _require(self, expected):
        if self.state != expected:
            raise SelectionOrderError(f"session {self.session_id}: state {self.state} != required {expected}")


# ----------------------------------------------------------------------- formal writer (LOCKED)
def write_formal_phase_manifest_atomic(phase, generated: GeneratedPhase, output_path, *,
                                       auth: GeneratorAuthorization) -> None:
    """LOCKED: refuses without formal authorization BEFORE any filesystem effect (no dir/temp/write/overwrite).
    The future atomic implementation is intentionally absent. NOTE: this module writes no files (no `open`)."""
    require_formal_authorization(auth)   # always raises in this phase -> zero side effects below
    raise FormalGenerationNotAuthorized("unreachable: formal writing not authorized")


__all__ = ["GeneratorAuthorization", "CommitContext", "EnvironmentVersionContext", "GeneratedPhase",
           "TrialExecutionEnvelope", "require_generator_authorization", "require_formal_authorization",
           "validate_commit_context", "commit_context_is_smoke_placeholder", "smoke_placeholder_commits",
           "require_smoke_commit_context", "validate_smoke_environment_provenance", "SMOKE_ENV_VERSION_KEYS",
           "SMOKE_ENV_FIXED", "build_phase_manifest_in_memory", "build_trial_execution_envelope",
           "assert_no_secret_in_public", "canonical_selected_offset", "SessionRunController", "SESSION_STATES",
           "write_formal_phase_manifest_atomic", "FormalGenerationNotAuthorized", "CommitContextError",
           "SmokeCommitContextError", "PublicPayloadLeakageError", "SelectionOrderError",
           "EnvironmentProvenanceError", "EXECUTION_ORDER_DEFINITION"]
