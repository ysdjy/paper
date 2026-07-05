"""Confirmatory v4 phase-manifest GENERATOR (offline; KAT-gated; smoke-only authorized).

Builds the IN-MEMORY fully-resolved phase manifest for a phase (train_validation | test) from the FROZEN
active modules ONLY -- it re-defines no scientific constant:
  PRE = preregistration_v4              (frozen seeds, SECRET_DENYLIST)
  ID  = confirmatory_v4_identity        (identities, subseeds, storage/execution order, structure hash)
  BS  = confirmatory_v4_block_state     (UNIQUE frozen block-state sampler; mandatory KAT gate)
  MI  = confirmatory_v4_manifest_integrity (schema/config/env, DEEP validator, full-manifest hash)
  SEL = confirmatory_v4_selection       (best-single selection freeze interface)

Hard rules honoured here:
  * the FIRST operation of any phase build / smoke is BS.require_known_answer_compatibility() (mandatory KAT);
  * block state ONLY via BS.resolved_block_state(...); the unchecked core is never called;
  * MI.reference_phase_manifest is NOT wrapped/returned (independent construction; tests assert equality);
  * the formal writer refuses without formal authorization (zero side effects);
  * no Isaac, no model, no confirmatory data, no formal manifest written by this module.
"""

from __future__ import annotations

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


# ----------------------------------------------------------------------- errors
class FormalGenerationNotAuthorized(RuntimeError):
    verdict = "GENERATOR_FORMAL_MODE_NOT_AUTHORIZED"


class CommitContextError(ValueError):
    verdict = "GENERATOR_COMMIT_CONTEXT_INVALID"


class PublicPayloadLeakageError(ValueError):
    verdict = "GENERATOR_PUBLIC_PAYLOAD_LEAKAGE"


class SelectionOrderError(RuntimeError):
    verdict = "GENERATOR_SELECTION_ORDER_VIOLATION"


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
class EnvironmentVersionContext:
    values: Mapping[str, str]


@dataclass(frozen=True)
class GeneratedPhase:
    phase: Literal["train_validation", "test"]
    unsealed_manifest: Mapping[str, Any]
    full_manifest_sha256: str
    trial_count: int
    smoke_only: bool
    kat_report: Mapping[str, Any]


@dataclass(frozen=True)
class TrialExecutionEnvelope:
    canonical_trial_identity: str
    planned_episode_id: str
    resume_key: str
    execution_order_index: int
    public_trial_spec: Mapping[str, Any]
    secret_environment_spec: Mapping[str, Any]
    smoke_only: bool


# ----------------------------------------------------------------------- gates
def require_generator_authorization(auth: GeneratorAuthorization) -> None:
    """Current phase authorizes ONLY smoke_only=true with all formal flags false."""
    if not isinstance(auth, GeneratorAuthorization):
        raise FormalGenerationNotAuthorized("authorization must be a GeneratorAuthorization")
    if auth.smoke_only is not True:
        raise FormalGenerationNotAuthorized("smoke_only must be True (no default formal mode)")
    if (auth.formal_manifest_generation_authorized or auth.confirmatory_data_generation_authorized
            or auth.confirmatory_run_authorized):
        raise FormalGenerationNotAuthorized(
            "formal manifest / confirmatory data / confirmatory run are NOT authorized in this phase")


def require_formal_authorization(auth: GeneratorAuthorization) -> None:
    """Guards the formal writer. There is NO formal authorization now -> always refuses."""
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


# ----------------------------------------------------------------------- production builder
def _phase_splits(phase: str):
    if phase not in MI.PHASE_SPLITS:
        raise ValueError(f"unknown phase {phase!r}")
    return MI.PHASE_SPLITS[phase]


def _build_unsealed_manifest(phase: str, commits: CommitContext,
                             environment_versions: EnvironmentVersionContext) -> dict:
    """Independent construction (NOT MI.reference_phase_manifest) from the frozen modules. Blocks/sessions
    in canonical STORAGE order; block state from the frozen sampler; execution_order_index from the seed-
    derived plan; nothing hashed here."""
    splits = _phase_splits(phase)

    # ---- blocks (storage order: split rank -> block_index asc) ----
    blocks = []
    for split in splits:
        for bi in ID.BLOCKS[split]:
            bst = BS.resolved_block_state(split, bi)                 # frozen sampler (KAT-gated); no placeholder
            blocks.append({
                "split": split, "block_index": bi,
                "canonical_block_identity": ID.block_identity(split, bi),
                "residual_value": bst["residual_value"], "nuisance_values": bst["nuisance_values"],
                "residual_subseed": bst["residual_subseed"], "nuisance_subseed": bst["nuisance_subseed"],
                "block_order_key": ID.block_order_key(split, bi),
            })

    # ---- sessions (storage order: split -> block asc -> nominal numeric asc) ----
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

    # ---- trials (storage order == canonical_phase_trial_identities; execution index from the seed plan) ----
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
    """Frozen order: auth gate -> mandatory KAT -> commit context -> build -> DEEP validate -> full hash."""
    require_generator_authorization(auth)                      # 1
    kat = BS.require_known_answer_compatibility()              # 2  MANDATORY KAT (first real work)
    validate_commit_context(commits)                          # 3
    manifest = _build_unsealed_manifest(phase, commits, environment_versions)   # 4-7
    MI.validate_fully_resolved_phase_manifest(manifest)       # 8  (re-runs KAT + recomputes residual/nuisance)
    full_hash = MI.fully_resolved_phase_manifest_hash(manifest)   # 9  (validates again, then hashes)
    return GeneratedPhase(phase=phase, unsealed_manifest=manifest, full_manifest_sha256=full_hash,
                          trial_count=len(manifest["trials"]), smoke_only=bool(auth.smoke_only),
                          kat_report=dict(kat))


# ----------------------------------------------------------------------- public/secret envelope
def _denylist_tokens() -> set:
    """Forbidden key tokens derived from the ACTIVE PRE.SECRET_DENYLIST (first word of each entry) plus the
    canonical secret keys; a public payload key matching any token is leakage."""
    toks = set()
    for entry in PRE.SECRET_DENYLIST:
        toks.add(str(entry).split(" ")[0].strip())
    toks.update({"nominal_bias", "residual_bias", "actual_bias", "eff_signed", "abs_eff",
                 "residual_value", "residual_subseed", "nuisance_subseed", "nuisance_values",
                 "future_candidate_outcomes", "oracle_action", "hidden_state_id",
                 "secret_deployment_state", "block_seed", "residual_seed", "tau_minus_abs_eff"})
    return toks


def assert_no_secret_in_public(public: Mapping[str, Any]) -> None:
    """Recursively reject any ACTIVE-denylist key anywhere in the public payload."""
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
    """Public (model/runtime-legal) vs secret (audit-only) split for one trial. Public carries NO denylist
    field; the block hidden state (residual + nuisance) lives ONLY in secret. KAT-gated (via block state)."""
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
        "task": {"target_open_position": 0.20, "target_tolerance": PRE.PRIMARY_SUCCESS["target_tolerance"]},
    }
    assert_no_secret_in_public(public)
    secret = {
        "split": r["split"], "block_index": r["block_index"], "nominal_bias": r["nominal"],
        "residual_bias": bst["residual_value"], "nuisance_values": bst["nuisance_values"],
        "residual_subseed": bst["residual_subseed"], "nuisance_subseed": bst["nuisance_subseed"],
    }
    return TrialExecutionEnvelope(canonical_trial_identity=canonical_trial_identity, planned_episode_id=pid,
                                  resume_key=pid, execution_order_index=exec_index[canonical_trial_identity],
                                  public_trial_spec=public, secret_environment_spec=secret,
                                  smoke_only=bool(auth.smoke_only))


# ----------------------------------------------------------------------- session run-state interface (no runtime)
SESSION_STATES = ("PLANNED", "PROBE_READY", "PROBE_COMPLETE", "SELECTION_FROZEN", "CANDIDATES_READY",
                  "SESSION_COMPLETE")


@dataclass
class SessionRunController:
    """PURE state interface (no Isaac): enforces probe-first and selection-frozen-before-any-candidate-outcome.
    Candidate outcomes may NEVER be handed to the selector; selection freezes on probe evidence only."""
    session_id: str
    state: str = "PLANNED"
    _selected_offset: float | None = field(default=None)
    _evidence_hash: str | None = field(default=None)

    def probe_ready(self):
        self._require("PLANNED"); self.state = "PROBE_READY"

    def probe_complete(self):
        self._require("PROBE_READY"); self.state = "PROBE_COMPLETE"

    def freeze_selection(self, selected_offset: float, evidence_hash: str):
        """Selection is frozen from probe evidence ONLY, BEFORE any candidate outcome exists."""
        self._require("PROBE_COMPLETE")
        if not any(abs(float(selected_offset) - o) <= 1e-12 for o in ID.CANDIDATE_BANK):
            raise SelectionOrderError(f"selected_offset {selected_offset} not in candidate bank")
        self._selected_offset = float(selected_offset); self._evidence_hash = str(evidence_hash)
        self.state = "SELECTION_FROZEN"

    def candidates_ready(self):
        self._require("SELECTION_FROZEN"); self.state = "CANDIDATES_READY"

    def session_complete(self):
        self._require("CANDIDATES_READY"); self.state = "SESSION_COMPLETE"

    def offer_candidate_outcome_to_selector(self, *_a, **_k):
        raise SelectionOrderError("candidate outcomes may NOT enter the selector (frozen before candidates)")

    def attempt_id(self, planned_episode_id: str, attempt_index: int) -> str:
        return ID.attempt_id(planned_episode_id, attempt_index)   # attempt in {0,1}

    def _require(self, expected):
        if self.state != expected:
            raise SelectionOrderError(f"session {self.session_id}: state {self.state} != required {expected}")


# ----------------------------------------------------------------------- formal writer (LOCKED)
def write_formal_phase_manifest_atomic(phase, generated: GeneratedPhase, output_path, *,
                                       auth: GeneratorAuthorization) -> None:
    """Formal manifest writer. LOCKED: refuses without formal authorization BEFORE any filesystem effect
    (no dir, no temp, no write, no overwrite). The future atomic implementation is intentionally absent."""
    require_formal_authorization(auth)   # always raises in this phase -> zero side effects below
    raise FormalGenerationNotAuthorized("unreachable: formal writing not authorized")


__all__ = ["GeneratorAuthorization", "CommitContext", "EnvironmentVersionContext", "GeneratedPhase",
           "TrialExecutionEnvelope", "require_generator_authorization", "require_formal_authorization",
           "validate_commit_context", "commit_context_is_smoke_placeholder", "smoke_placeholder_commits",
           "build_phase_manifest_in_memory", "build_trial_execution_envelope", "assert_no_secret_in_public",
           "SessionRunController", "SESSION_STATES", "write_formal_phase_manifest_atomic",
           "FormalGenerationNotAuthorized", "CommitContextError", "PublicPayloadLeakageError",
           "SelectionOrderError", "EXECUTION_ORDER_DEFINITION"]
