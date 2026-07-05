"""Full fully-resolved phase-manifest integrity for confirmatory v4 (offline; pure functions; NO manifest
instance, NO data).

FIX3 / BLOCKER_MANIFEST_INTEGRITY_HASH_STRUCTURE_ONLY.

`confirmatory_v4_identity.canonical_planned_structure_hash()` covers ONLY planned identities + episode ids
(structure). Under seal Scheme 2 the runtime records must anchor to a hash that also covers the resolved
residual/nuisance values, execution order, seeds, deterministic environment, and commits. This module
provides those FULL, phase-specific manifest hashes plus the combined experiment-plan hash — as pure
functions/validators over an in-memory manifest dict. It builds/writes NO real manifest.

Phases (Scheme 2):
    train_validation : 15 blocks (9 train + 6 val), 57 sessions, 228 trials  (probe+3 candidates)
    test             :  9 blocks,                    18 sessions,  72 trials
    total            : 300 trials
"""

from __future__ import annotations

import copy
import hashlib
import json

PHASES = {
    "train_validation": {"blocks": 15, "sessions": 57, "trials": 228},
    "test": {"blocks": 9, "sessions": 18, "trials": 72},
}
STRUCTURE_HASH_SEMANTICS = ("planned identities + planned_episode_ids ONLY; excludes residual/nuisance/"
                            "execution-order/seeds/environment/generator-commit; NOT a full-manifest anchor")
MANIFEST_ALGORITHM_VERSION = "confirmatory_v4_full_manifest_v1"
SELF_HASH_FIELD = "integrity.full_manifest_sha256"

# required coverage (documented + test-checked): every resolved scientific field enters the hash
REQUIRED_TOP_LEVEL = ("schema_version", "phase", "protocol_commit", "generator_commit", "runtime_commit",
                      "config_sha256", "planned_structure_sha256", "frozen_seeds", "deterministic_environment",
                      "environment_versions", "counts", "execution_order_definition",
                      "manifest_algorithm_version")
REQUIRED_BLOCK = ("split", "block_index", "canonical_block_identity", "residual_value", "nuisance_values",
                  "residual_subseed", "nuisance_subseed", "block_order_key")
REQUIRED_SESSION = ("canonical_session_identity", "split", "block_index", "nominal_bias",
                    "session_order_key", "nominal_order_key", "resolved_candidate_order")
REQUIRED_TRIAL = ("canonical_trial_identity", "planned_episode_id", "role", "offset", "trial_init_subseed",
                  "execution_order_index", "resume_key")


class ManifestIntegrityError(ValueError):
    verdict = "EXPERIMENT_INVALID_MANIFEST_INTEGRITY"


def canonical_planned_structure_hash() -> str:
    """Re-exported structure-only hash (see confirmatory_v4_identity). NOT a full-manifest anchor."""
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as ID
    return ID.canonical_planned_structure_hash()


def _canonical(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha(payload) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def validate_fully_resolved_phase_manifest(manifest) -> None:
    """Structural/coverage validation. Raises ManifestIntegrityError. Ensures NO resolved scientific field
    is absent (every required top-level/block/session/trial key present), phase counts match, and the only
    permitted place for the self-hash is SELF_HASH_FIELD (missing or null at hashing time)."""
    if not isinstance(manifest, dict):
        raise ManifestIntegrityError("manifest must be a dict")
    phase = manifest.get("phase")
    if phase not in PHASES:
        raise ManifestIntegrityError(f"unknown phase {phase!r}")
    for k in REQUIRED_TOP_LEVEL:
        if k not in manifest:
            raise ManifestIntegrityError(f"missing top-level field {k!r}")
    exp = PHASES[phase]
    blocks = manifest.get("blocks", [])
    sessions = manifest.get("sessions", [])
    trials = manifest.get("trials", [])
    if manifest.get("counts") != {"blocks": exp["blocks"], "sessions": exp["sessions"],
                                  "trials": exp["trials"]}:
        raise ManifestIntegrityError(f"counts != frozen {exp} for phase {phase}")
    if len(blocks) != exp["blocks"] or len(sessions) != exp["sessions"] or len(trials) != exp["trials"]:
        raise ManifestIntegrityError("resolved block/session/trial list lengths != phase counts")
    for b in blocks:
        for k in REQUIRED_BLOCK:
            if k not in b:
                raise ManifestIntegrityError(f"block missing {k!r}")
    for s in sessions:
        for k in REQUIRED_SESSION:
            if k not in s:
                raise ManifestIntegrityError(f"session missing {k!r}")
    for t in trials:
        for k in REQUIRED_TRIAL:
            if k not in t:
                raise ManifestIntegrityError(f"trial missing {k!r}")
    # self-hash field must be absent or null when hashing
    if manifest.get(SELF_HASH_FIELD) not in (None,):
        raise ManifestIntegrityError(f"{SELF_HASH_FIELD} must be null/absent before hashing")


def fully_resolved_phase_manifest_hash(manifest) -> str:
    """Validate, then hash EVERYTHING except the self-hash field. Deep-copied; canonical JSON (sorted keys,
    tight separators, ensure_ascii=False, allow_nan=False). Nothing scientific is excluded."""
    validate_fully_resolved_phase_manifest(manifest)
    payload = copy.deepcopy(manifest)
    payload.pop(SELF_HASH_FIELD, None)      # exclude ONLY the self field
    return _sha(payload)


def stamp_full_manifest_hash(manifest) -> dict:
    """Return a copy with SELF_HASH_FIELD set to the full hash of the rest (offline helper; no file I/O)."""
    h = fully_resolved_phase_manifest_hash(manifest)
    out = copy.deepcopy(manifest)
    out[SELF_HASH_FIELD] = h
    return out


def combined_experiment_plan_hash(*, train_validation_manifest_sha256, model_analysis_freeze_sha256,
                                  test_manifest_sha256, config_sha256, protocol_commit, generator_commit,
                                  analysis_code_sha256=None, bootstrap_seed=None) -> str:
    """Step-6 combined plan anchor over the two phase manifests + model/analysis freeze + commits + config."""
    payload = {
        "protocol_commit": protocol_commit,
        "config_sha256": config_sha256,
        "generator_commit": generator_commit,
        "train_validation_manifest_sha256": train_validation_manifest_sha256,
        "model_analysis_freeze_sha256": model_analysis_freeze_sha256,
        "test_manifest_sha256": test_manifest_sha256,
        "analysis_code_sha256": analysis_code_sha256,
        "bootstrap_seed": bootstrap_seed,
    }
    for k, v in payload.items():
        if v is None and k in ("train_validation_manifest_sha256", "model_analysis_freeze_sha256",
                               "test_manifest_sha256", "config_sha256", "protocol_commit", "generator_commit"):
            raise ManifestIntegrityError(f"combined plan hash requires {k}")
    return _sha(payload)


HASH_LAYERS = {
    "planned_structure_sha256": STRUCTURE_HASH_SEMANTICS,
    "train_validation_manifest_sha256": "full fully-resolved train/validation phase manifest (228 trials)",
    "test_manifest_sha256": "full fully-resolved test phase manifest (72 trials)",
    "model_analysis_freeze_sha256": "10 model hashes + best-single artifact + feature/analysis code + bootstrap seed",
    "combined_experiment_plan_sha256": "Step-6 anchor over both phase manifests + model/analysis freeze + commits",
    "per_record_anchor": "each runtime record.science_manifest_sha256 == its PHASE full-manifest hash",
    "final_analysis_checks": ["train/val records match train_validation_manifest_sha256",
                              "test records match test_manifest_sha256",
                              "model hashes match model_analysis_freeze",
                              "combined_experiment_plan_sha256 matches -> else EXPERIMENT_INVALID_MANIFEST_INTEGRITY"],
}
