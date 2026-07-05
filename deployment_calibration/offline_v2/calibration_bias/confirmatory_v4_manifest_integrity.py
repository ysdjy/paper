"""Full fully-resolved phase-manifest integrity for confirmatory v4 (offline; pure functions; NO manifest
instance, NO data).

FIX4 / BLOCKER_FULL_MANIFEST_VALIDATOR_SHALLOW + BLOCKER_COMBINED_HASH_OPTIONAL_FIELDS.

`confirmatory_v4_identity.canonical_planned_structure_hash()` covers ONLY planned identities + episode ids
(structure). Under seal Scheme 2 the runtime records anchor to a FULL phase-manifest hash that also covers
the resolved residual/nuisance values, execution order, seeds, deterministic environment, and commits. The
full hash may be computed ONLY after DEEP scientific validation (composition, uniqueness, identity<->PID
recompute, referential integrity, block-shared residual/nuisance, subseed recompute, execution order,
planned-structure subset, frozen seeds/config/version). A shape-only manifest (e.g. 228 duplicate trials)
is rejected. No real manifest is built or written.

Phases (Scheme 2):
    train_validation : 15 blocks (9 train + 6 val), 57 sessions, 228 trials  (probe + 3 candidates)
    test             :  9 blocks,                    18 sessions,  72 trials
    total            : 300 trials
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re

PHASES = {
    "train_validation": {"blocks": 15, "sessions": 57, "trials": 228},
    "test": {"blocks": 9, "sessions": 18, "trials": 72},
}
PHASE_SPLITS = {"train_validation": ("train", "validation"), "test": ("test",)}
PHASE_BLOCK_SPLIT_COUNTS = {"train_validation": {"train": 9, "validation": 6}, "test": {"test": 9}}
PHASE_SESSION_SPLIT_COUNTS = {"train_validation": {"train": 45, "validation": 12}, "test": {"test": 18}}
PHASE_TRIAL_SPLIT_COUNTS = {"train_validation": {"train": 180, "validation": 48}, "test": {"test": 72}}

STRUCTURE_HASH_SEMANTICS = ("planned identities + planned_episode_ids ONLY; excludes residual/nuisance/"
                            "execution-order/seeds/environment/generator-commit; NOT a full-manifest anchor")
MANIFEST_ALGORITHM_VERSION = "confirmatory_v4_full_manifest_v1"
SELF_HASH_FIELD = "integrity.full_manifest_sha256"          # flat top-level key; nested {"integrity":{}} forbidden
FROZEN_BOOTSTRAP_SEED = 9014517173581927929
RESIDUAL_LO, RESIDUAL_HI = -0.01, 0.01
_ABS_TOL = 1e-12
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

REQUIRED_TOP_LEVEL = ("schema_version", "phase", "protocol_commit", "generator_commit", "runtime_commit",
                      "config_sha256", "planned_structure_sha256", "frozen_seeds", "deterministic_environment",
                      "environment_versions", "counts", "execution_order_definition",
                      "manifest_algorithm_version")
ALLOWED_TOP_LEVEL = set(REQUIRED_TOP_LEVEL) | {"blocks", "sessions", "trials", SELF_HASH_FIELD}
REQUIRED_BLOCK = ("split", "block_index", "canonical_block_identity", "residual_value", "nuisance_values",
                  "residual_subseed", "nuisance_subseed", "block_order_key")
REQUIRED_SESSION = ("canonical_session_identity", "split", "block_index", "nominal_bias",
                    "session_order_key", "nominal_order_key", "resolved_candidate_order")
REQUIRED_TRIAL = ("canonical_trial_identity", "planned_episode_id", "session_ref", "role", "offset",
                  "trial_init_subseed", "execution_order_index", "resume_key")


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


def _err(msg):
    raise ManifestIntegrityError(msg)


def _int_nonbool(v, what):
    if isinstance(v, bool) or not isinstance(v, int):
        _err(f"{what} must be a non-bool int, got {v!r}")
    return v


def _finite(v, what):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        _err(f"{what} must be a real number, got {v!r}")
    if not math.isfinite(float(v)):
        _err(f"{what} must be finite, got {v!r}")
    return float(v)


def _recurse_finite(obj, path="root"):
    if isinstance(obj, float) and not math.isfinite(obj):
        _err(f"non-finite numeric at {path}")
    if isinstance(obj, dict):
        for k, v in obj.items():
            _recurse_finite(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _recurse_finite(v, f"{path}[{i}]")


def _isclose(a, b):
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=_ABS_TOL)


def validate_fully_resolved_phase_manifest(manifest) -> None:
    """DEEP scientific integrity validation. Raises ManifestIntegrityError on ANY violation. Recomputes every
    canonical identity, planned_episode_id, and subseed from confirmatory_v4_identity + the frozen seeds;
    checks phase split composition, global uniqueness, referential integrity, per-session probe+3-candidate
    structure, block-shared residual (finite in [-0.01,+0.01]), execution-order completeness, the planned-
    structure subset, and the frozen seeds/config/version. Only after this may the full hash be computed."""
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as ID
    from deployment_calibration.offline_v2.calibration_bias import preregistration_v4 as PRE

    if not isinstance(manifest, dict):
        _err("manifest must be a dict")
    if "integrity" in manifest:                              # nested self-hash object forbidden
        _err("nested 'integrity' object forbidden; use the flat key 'integrity.full_manifest_sha256'")
    for k in manifest:                                       # strict: no unexpected top-level keys
        if k not in ALLOWED_TOP_LEVEL:
            _err(f"unexpected top-level field {k!r}")
    for k in REQUIRED_TOP_LEVEL:
        if k not in manifest:
            _err(f"missing top-level field {k!r}")
    phase = manifest["phase"]
    if phase not in PHASES:
        _err(f"unknown phase {phase!r}")
    exp = PHASES[phase]
    blocks, sessions, trials = manifest.get("blocks", []), manifest.get("sessions", []), manifest.get("trials", [])
    if manifest["counts"] != {"blocks": exp["blocks"], "sessions": exp["sessions"], "trials": exp["trials"]}:
        _err(f"counts != frozen {exp} for phase {phase}")
    if len(blocks) != exp["blocks"] or len(sessions) != exp["sessions"] or len(trials) != exp["trials"]:
        _err("resolved block/session/trial list lengths != phase counts")
    if manifest.get(SELF_HASH_FIELD) is not None:
        _err(f"{SELF_HASH_FIELD} must be null/absent before hashing")

    # ---- strict per-item schema (no unexpected fields) + finite ----
    for name, items, req in (("block", blocks, REQUIRED_BLOCK), ("session", sessions, REQUIRED_SESSION),
                             ("trial", trials, REQUIRED_TRIAL)):
        for it in items:
            if not isinstance(it, dict):
                _err(f"{name} must be a dict")
            for k in req:
                if k not in it:
                    _err(f"{name} missing {k!r}")
            for k in it:
                if k not in req:
                    _err(f"{name} unexpected field {k!r}")
    _recurse_finite(manifest)

    # ---- top-level frozen values ----
    if manifest["manifest_algorithm_version"] != MANIFEST_ALGORITHM_VERSION:
        _err("manifest_algorithm_version mismatch")
    if manifest["frozen_seeds"] != PRE.frozen_seeds():
        _err("frozen_seeds != preregistered frozen seeds")
    if not _SHA256_RE.match(str(manifest["config_sha256"])):
        _err("config_sha256 not 64 lowercase hex")
    if not _SHA256_RE.match(str(manifest["planned_structure_sha256"])):
        _err("planned_structure_sha256 not 64 lowercase hex")
    if manifest["planned_structure_sha256"] != ID.canonical_planned_structure_hash():
        _err("planned_structure_sha256 != canonical_planned_structure_hash()")
    for c in ("protocol_commit", "generator_commit", "runtime_commit"):
        if not _GIT_SHA_RE.match(str(manifest[c])):
            _err(f"{c} not a 40-hex git sha")
    if not isinstance(manifest["deterministic_environment"], dict) or \
            str(manifest["deterministic_environment"].get("device", "")).lower() != "cpu":
        _err("deterministic_environment.device must be 'cpu'")

    # ---- blocks: composition, uniqueness, identity recompute, residual/subseeds ----
    blk_by_key = {}
    blk_ident, blk_order = set(), set()
    blk_split_count = {}
    for b in blocks:
        split, bi = b["split"], b["block_index"]
        if split not in PHASE_SPLITS[phase]:
            _err(f"block split {split!r} not allowed in phase {phase}")
        _int_nonbool(bi, "block_index")
        if bi not in ID.BLOCKS.get(split, range(0)):
            _err(f"block_index {bi} out of range for split {split!r}")
        if b["canonical_block_identity"] != ID.block_identity(split, bi):
            _err(f"canonical_block_identity mismatch for {split} {bi}")
        key = (split, bi)
        if key in blk_by_key:
            _err(f"duplicate block {key}")
        if b["canonical_block_identity"] in blk_ident:
            _err("duplicate canonical_block_identity")
        blk_ident.add(b["canonical_block_identity"])
        _int_nonbool(b["block_order_key"], "block_order_key")
        if b["block_order_key"] in blk_order:
            _err("duplicate block_order_key")
        blk_order.add(b["block_order_key"])
        res = _finite(b["residual_value"], "residual_value")
        if not (RESIDUAL_LO - _ABS_TOL <= res <= RESIDUAL_HI + _ABS_TOL):
            _err(f"residual_value {res} out of [{RESIDUAL_LO},{RESIDUAL_HI}]")
        if not isinstance(b["nuisance_values"], dict):
            _err("nuisance_values must be a dict")
        if b["residual_subseed"] != ID.block_residual_subseed(split, bi):
            _err(f"residual_subseed mismatch for {split} {bi}")
        if b["nuisance_subseed"] != ID.block_nuisance_subseed(split, bi):
            _err(f"nuisance_subseed mismatch for {split} {bi}")
        blk_by_key[key] = b
        blk_split_count[split] = blk_split_count.get(split, 0) + 1
    if blk_split_count != PHASE_BLOCK_SPLIT_COUNTS[phase]:
        _err(f"block split composition {blk_split_count} != {PHASE_BLOCK_SPLIT_COUNTS[phase]}")

    # ---- sessions: composition, uniqueness, identity/subseed recompute, referential to block ----
    sess_by_ident = {}
    sess_keyset, sess_order = set(), set()
    sess_split_count = {}
    for s in sessions:
        split, bi, nom = s["split"], s["block_index"], s["nominal_bias"]
        if (split, bi) not in blk_by_key:
            _err(f"session references missing block {(split, bi)}")
        _finite(nom, "nominal_bias")
        if not any(_isclose(nom, n) for n in ID.NOMINALS.get(split, ())):
            _err(f"nominal_bias {nom} not in {split} nominal set")
        ident = ID.session_identity(split, bi, nom)
        if s["canonical_session_identity"] != ident:
            _err(f"canonical_session_identity mismatch for {split} {bi} {nom}")
        if ident in sess_by_ident:
            _err("duplicate canonical_session_identity")
        key = (split, bi, round(float(nom), 3))
        if key in sess_keyset:
            _err(f"duplicate (split,block,nominal) {key}")
        sess_keyset.add(key)
        if s["session_order_key"] != ID.session_order_subseed(split, bi, nom):
            _err("session_order_key mismatch")
        if s["nominal_order_key"] != ID.nominal_order_subseed(split, bi, nom):
            _err("nominal_order_key mismatch")
        if s["session_order_key"] in sess_order:
            _err("duplicate session_order_key")
        sess_order.add(s["session_order_key"])
        rco = s["resolved_candidate_order"]
        if not isinstance(rco, list) or len(rco) != len(ID.CANDIDATE_BANK):
            _err("resolved_candidate_order must be a length-3 list")
        rco_c = [_canon_bank(v) for v in rco]
        if sorted(rco_c) != sorted(ID.CANDIDATE_BANK):
            _err("resolved_candidate_order is not a bank permutation")
        sess_by_ident[ident] = s
        sess_split_count[split] = sess_split_count.get(split, 0) + 1
    if sess_split_count != PHASE_SESSION_SPLIT_COUNTS[phase]:
        _err(f"session split composition {sess_split_count} != {PHASE_SESSION_SPLIT_COUNTS[phase]}")

    # ---- trials: uniqueness, identity<->PID recompute, referential, per-session structure, exec order ----
    tri_ident, tri_pid, exec_idx = set(), set(), []
    per_session = {}
    tri_split_count = {}
    planned_trial_idents = set()
    for t in trials:
        sref = t["session_ref"]
        sess = sess_by_ident.get(sref)
        if sess is None:
            _err(f"trial references missing session {sref!r}")
        split, bi, nom = sess["split"], sess["block_index"], sess["nominal_bias"]
        role = t["role"]
        if role not in ("probe", "candidate"):
            _err(f"illegal trial role {role!r}")
        off = _canon_bank(t["offset"]) if role == "candidate" else _canon_probe(t["offset"])
        tid = ID.trial_identity(split, bi, nom, role, off)
        if t["canonical_trial_identity"] != tid:
            _err(f"canonical_trial_identity mismatch ({sref}, {role}, {off})")
        pid = ID.planned_episode_id(tid)
        if t["planned_episode_id"] != pid:
            _err("planned_episode_id != planned_episode_id(canonical_trial_identity)")
        if t["resume_key"] != pid:
            _err("resume_key != planned_episode_id")
        if tid in tri_ident:
            _err("duplicate canonical_trial_identity")
        tri_ident.add(tid)
        if pid in tri_pid:
            _err("duplicate planned_episode_id")
        tri_pid.add(pid)
        if t["trial_init_subseed"] != ID.trial_init_subseed(split, bi, nom, role, off):
            _err("trial_init_subseed mismatch")
        _int_nonbool(t["execution_order_index"], "execution_order_index")
        exec_idx.append(t["execution_order_index"])
        per_session.setdefault(sref, []).append((role, off, t["execution_order_index"]))
        tri_split_count[split] = tri_split_count.get(split, 0) + 1
        planned_trial_idents.add(tid)
    if tri_split_count != PHASE_TRIAL_SPLIT_COUNTS[phase]:
        _err(f"trial split composition {tri_split_count} != {PHASE_TRIAL_SPLIT_COUNTS[phase]}")

    # per-session 1 probe(-0.04) + 3 candidates(bank); probe before candidates in execution order
    for sref, tl in per_session.items():
        if len(tl) != 4:
            _err(f"session {sref} has {len(tl)} trials, expected 4 (probe + 3 candidates)")
        probes = [x for x in tl if x[0] == "probe"]
        cands = [x for x in tl if x[0] == "candidate"]
        if len(probes) != 1 or not _isclose(probes[0][1], ID.PROBE_OFFSET):
            _err(f"session {sref} must have exactly one probe at -0.04")
        if sorted(_canon_bank(c[1]) for c in cands) != sorted(ID.CANDIDATE_BANK):
            _err(f"session {sref} candidate offsets != bank")
        if min(c[2] for c in cands) <= probes[0][2]:
            _err(f"session {sref}: probe must execute before its candidates")

    # execution order: unique, complete 0..N-1
    n = len(trials)
    if sorted(exec_idx) != list(range(n)):
        _err(f"execution_order_index must be a complete unique 0..{n - 1}")

    # planned-structure subset: the trials are exactly this phase's slice of the 300 planned identities
    phase_planned = {r["canonical_trial_identity"] for r in ID.enumerate_trials()
                     if r["split"] in PHASE_SPLITS[phase]}
    if planned_trial_idents != phase_planned:
        _err("trial identities are not exactly the phase subset of the 300 planned structure")


def _canon_bank(v):
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as ID
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        _err(f"offset must be a finite number, got {v!r}")
    for a in ID.CANDIDATE_BANK:
        if _isclose(v, a):
            return a
    _err(f"offset {v!r} not in candidate bank")


def _canon_probe(v):
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as ID
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        _err(f"probe offset must be a finite number, got {v!r}")
    if not _isclose(v, ID.PROBE_OFFSET):
        _err(f"probe offset {v!r} != -0.04")
    return ID.PROBE_OFFSET


def fully_resolved_phase_manifest_hash(manifest) -> str:
    """DEEP-validate, then hash EVERYTHING except the flat self-hash field. Deep-copied; canonical JSON
    (sorted keys, tight separators, ensure_ascii=False, allow_nan=False). Nothing scientific is excluded."""
    validate_fully_resolved_phase_manifest(manifest)
    payload = copy.deepcopy(manifest)
    payload.pop(SELF_HASH_FIELD, None)
    return _sha(payload)


def stamp_full_manifest_hash(manifest) -> dict:
    h = fully_resolved_phase_manifest_hash(manifest)
    out = copy.deepcopy(manifest)
    out[SELF_HASH_FIELD] = h
    return out


def combined_experiment_plan_hash(*, train_validation_manifest_sha256, model_analysis_freeze_sha256,
                                  test_manifest_sha256, config_sha256, protocol_commit, generator_commit,
                                  analysis_code_sha256, bootstrap_seed) -> str:
    """Step-6 combined plan anchor. FIX4: ALL fields REQUIRED (no None/empty), sha256 fields 64 lowercase hex,
    commits 40 lowercase hex, bootstrap_seed the frozen non-bool int."""
    sha_fields = {"train_validation_manifest_sha256": train_validation_manifest_sha256,
                  "model_analysis_freeze_sha256": model_analysis_freeze_sha256,
                  "test_manifest_sha256": test_manifest_sha256, "config_sha256": config_sha256,
                  "analysis_code_sha256": analysis_code_sha256}
    for k, v in sha_fields.items():
        if not isinstance(v, str) or not _SHA256_RE.match(v):
            _err(f"{k} must be 64 lowercase hex, got {v!r}")
    for k, v in (("protocol_commit", protocol_commit), ("generator_commit", generator_commit)):
        if not isinstance(v, str) or not _GIT_SHA_RE.match(v):
            _err(f"{k} must be a 40-hex git sha, got {v!r}")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        _err(f"bootstrap_seed must be a non-bool int, got {bootstrap_seed!r}")
    if bootstrap_seed != FROZEN_BOOTSTRAP_SEED:
        _err(f"bootstrap_seed must equal the frozen {FROZEN_BOOTSTRAP_SEED}")
    payload = {"protocol_commit": protocol_commit, "config_sha256": config_sha256,
               "generator_commit": generator_commit,
               "train_validation_manifest_sha256": train_validation_manifest_sha256,
               "model_analysis_freeze_sha256": model_analysis_freeze_sha256,
               "test_manifest_sha256": test_manifest_sha256,
               "analysis_code_sha256": analysis_code_sha256, "bootstrap_seed": bootstrap_seed}
    return _sha(payload)


HASH_LAYERS = {
    "planned_structure_sha256": STRUCTURE_HASH_SEMANTICS,
    "train_validation_manifest_sha256": "full deep-validated train/validation phase manifest (228 trials)",
    "test_manifest_sha256": "full deep-validated test phase manifest (72 trials)",
    "model_analysis_freeze_sha256": "10 model hashes + best-single artifact + feature/analysis code + bootstrap seed",
    "combined_experiment_plan_sha256": "Step-6 anchor over both phase manifests + model/analysis freeze + commits",
    "per_record_anchor": "each runtime record.science_manifest_sha256 == its PHASE full-manifest hash",
    "final_analysis_checks": ["train/val records match train_validation_manifest_sha256",
                              "test records match test_manifest_sha256",
                              "model hashes match model_analysis_freeze",
                              "combined_experiment_plan_sha256 matches -> else EXPERIMENT_INVALID_MANIFEST_INTEGRITY"],
}
