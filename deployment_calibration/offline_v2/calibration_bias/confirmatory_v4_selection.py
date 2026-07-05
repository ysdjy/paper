"""Production best-single selector for confirmatory v4 (offline; observed-outcome-only; NO secret).

FIX3 / BLOCKER_BEST_SINGLE_INPUT_GUARD_INCOMPLETE (also FIX2 BLOCKER_PRODUCTION_BEST_SINGLE_STILL_SECRET_
DEPENDENT).

The ONLY confirmatory production entry point is `select_best_single_confirmatory(observed_candidate_records)`:
one business parameter, no override kwargs, all frozen design constants hardcoded, full completeness gate
always on (exactly 45 train + 12 validation sessions -> 135 + 36 = 171 candidate records). It reads ONLY
observed candidate outcomes (`y.success`) and never touches nominal / residual / actual bias / eff / tau /
oracle / a success model / test records.

The generic, override-capable logic is PRIVATE (`_select_best_single_core`, `_project_records`,
`_check_confirmatory_completeness`) and exists only for the offline power bridge / tests. It is NOT in
`__all__`, the active config does NOT reference it, and the generator spec forbids calling it. A thin
`select_best_single` alias to the core is retained for bridge/legacy tests only (also not in `__all__`).

Allowlisted input fields (everything else, incl. secret audit fields, is projected away first):
    split, session_id, trial_role, theta.grasp_offset_local_y, y.success, planned_episode_id
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

__all__ = ["select_best_single_confirmatory", "ObservedCandidateOutcome", "BestSingleInputError"]

SELECTION_RULE_VERSION = "confirmatory_v4_best_single_observed_v1"
COMPLETENESS_GATE_VERSION = "confirmatory_v4_completeness_45_12_135_36_171_v1"
PRODUCTION_ENTRYPOINT = ("deployment_calibration.offline_v2.calibration_bias.confirmatory_v4_selection."
                         "select_best_single_confirmatory")
CANDIDATE_BANK = (-0.04, 0.0, 0.04)

# FROZEN confirmatory split composition (NOT overridable through the production entry)
EXPECTED_TRAIN_SESSIONS = 45
EXPECTED_VAL_SESSIONS = 12
EXPECTED_SESSIONS = EXPECTED_TRAIN_SESSIONS + EXPECTED_VAL_SESSIONS                 # 57
EXPECTED_TRAIN_RECORDS = EXPECTED_TRAIN_SESSIONS * len(CANDIDATE_BANK)              # 135
EXPECTED_VAL_RECORDS = EXPECTED_VAL_SESSIONS * len(CANDIDATE_BANK)                  # 36
EXPECTED_CANDIDATE_RECORDS = EXPECTED_SESSIONS * len(CANDIDATE_BANK)                # 171


class BestSingleInputError(ValueError):
    """Raised on any completeness/leakage violation -> EXPERIMENT_INVALID_BEST_SINGLE_INPUT."""
    verdict = "EXPERIMENT_INVALID_BEST_SINGLE_INPUT"


@dataclass(frozen=True)
class ObservedCandidateOutcome:
    """Leakage-safe projection of one executed train/validation CANDIDATE trial."""
    split: Literal["train", "validation"]
    session_id: str
    planned_episode_id: str
    offset: float
    success: bool


def _round_offset(x) -> float:
    return round(float(x), 3)


def _project_records(records) -> list:
    """Allowlist-project raw records -> ObservedCandidateOutcome. Reads ONLY the 6 legal fields.

    Rejects (BestSingleInputError) any record that is not a train/validation candidate trial or whose
    observed success is not a Python bool. Secret audit fields on the raw record are never read."""
    out = []
    for i, r in enumerate(records):
        try:
            split = r["split"]
            session_id = r["session_id"]
            role = r["trial_role"]
            offset = _round_offset(r["theta"]["grasp_offset_local_y"])
            pid = r["planned_episode_id"]
            success = r["y"]["success"]
        except (KeyError, TypeError) as e:
            raise BestSingleInputError(f"record {i} missing allowlisted field: {e}")
        if role != "candidate":
            raise BestSingleInputError(f"record {i} trial_role={role!r} (only 'candidate'; probe/test excluded)")
        if split not in ("train", "validation"):
            raise BestSingleInputError(f"record {i} split={split!r} (only train/validation; test forbidden)")
        if not isinstance(success, bool):     # strict Python bool (rejects 0/1/np.bool_)
            raise BestSingleInputError(f"record {i} y.success is not a Python bool: {success!r}")
        out.append(ObservedCandidateOutcome(split, session_id, pid, offset, success))
    return out


def _check_confirmatory_completeness(proj):
    """Exact frozen 45/12 -> 135/36 -> 171 guard. No warnings, no dropped records, no denominator change."""
    bank = tuple(sorted(_round_offset(o) for o in CANDIDATE_BANK))
    if len(proj) != EXPECTED_CANDIDATE_RECORDS:
        raise BestSingleInputError(f"expected {EXPECTED_CANDIDATE_RECORDS} records, got {len(proj)}")
    seen_pid, seen_sso = set(), set()
    by_ss = {}                                 # (split, session_id) -> [offsets]
    sid_splits = {}                            # session_id -> {splits}
    train_recs = val_recs = 0
    for o in proj:
        if o.split == "train":
            train_recs += 1
        else:
            val_recs += 1
        if o.planned_episode_id in seen_pid:
            raise BestSingleInputError(f"duplicate planned_episode_id {o.planned_episode_id}")
        seen_pid.add(o.planned_episode_id)
        sso = (o.split, o.session_id, o.offset)
        if sso in seen_sso:
            raise BestSingleInputError(f"duplicate (split, session_id, offset) {sso}")
        seen_sso.add(sso)
        by_ss.setdefault((o.split, o.session_id), []).append(o.offset)
        sid_splits.setdefault(o.session_id, set()).add(o.split)
    if train_recs != EXPECTED_TRAIN_RECORDS:
        raise BestSingleInputError(f"train records {train_recs} != {EXPECTED_TRAIN_RECORDS}")
    if val_recs != EXPECTED_VAL_RECORDS:
        raise BestSingleInputError(f"validation records {val_recs} != {EXPECTED_VAL_RECORDS}")
    for sid, splits in sid_splits.items():     # session_id must be GLOBALLY unique (no cross-split collision)
        if len(splits) > 1:
            raise BestSingleInputError(f"session_id {sid!r} appears in multiple splits {splits}")
    train_sessions = [k for k in by_ss if k[0] == "train"]
    val_sessions = [k for k in by_ss if k[0] == "validation"]
    if len(train_sessions) != EXPECTED_TRAIN_SESSIONS:
        raise BestSingleInputError(f"train sessions {len(train_sessions)} != {EXPECTED_TRAIN_SESSIONS}")
    if len(val_sessions) != EXPECTED_VAL_SESSIONS:
        raise BestSingleInputError(f"validation sessions {len(val_sessions)} != {EXPECTED_VAL_SESSIONS}")
    if len(by_ss) != EXPECTED_SESSIONS:
        raise BestSingleInputError(f"total sessions {len(by_ss)} != {EXPECTED_SESSIONS}")
    for k, offs in by_ss.items():
        if len(offs) != len(bank):
            raise BestSingleInputError(f"session {k} has {len(offs)} candidates, expected {len(bank)}")
        if tuple(sorted(_round_offset(x) for x in offs)) != bank:
            raise BestSingleInputError(f"session {k} offset set != bank {bank}")


def _check_structure_only(proj, candidate_bank):
    """Design-agnostic per-session structure check (bridge only; not the confirmatory gate)."""
    bank = tuple(sorted(_round_offset(x) for x in candidate_bank))
    by_ss, seen = {}, set()
    for o in proj:
        k = (o.split, o.session_id, _round_offset(o.offset))
        if k in seen:
            raise BestSingleInputError(f"duplicate (split,session,offset) {k}")
        seen.add(k)
        by_ss.setdefault((o.split, o.session_id), []).append(o.offset)
    for k, offs in by_ss.items():
        if tuple(sorted(_round_offset(x) for x in offs)) != bank:
            raise BestSingleInputError(f"session {k} offset set != bank")


def _count_select(proj, candidate_bank):
    """Observed-count selection: max observed_success_count -> min|offset| -> earliest in bank order."""
    bank = [_round_offset(o) for o in candidate_bank]
    n_per = len(proj) // len(bank)
    counts = {o: 0 for o in bank}
    for o in proj:
        if o.success:
            counts[_round_offset(o.offset)] += 1
    max_c = max(counts[o] for o in bank)
    tie1 = [o for o in bank if counts[o] == max_c]
    min_abs = min(abs(o) for o in tie1)
    tie2 = [o for o in tie1 if abs(abs(o) - min_abs) < 1e-12]
    selected = min(tie2, key=lambda o: bank.index(o))
    artifact = {
        "n_trials_per_offset": n_per,
        "success_count_per_offset": {f"{o:+.3f}": counts[o] for o in bank},
        "mean_success_per_offset": {f"{o:+.3f}": (counts[o] / n_per if n_per else 0.0) for o in bank},
        "tie_set_after_step1": [f"{o:+.3f}" for o in tie1],
        "tie_set_after_step2": [f"{o:+.3f}" for o in tie2],
        "selected_offset": selected,
        "selection_rule_version": SELECTION_RULE_VERSION,
    }
    return selected, artifact


def _canonical_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _finalize(artifact, proj, raw_records, candidate_bank):
    proj_rows = sorted([[o.split, o.session_id, o.planned_episode_id, f"{_round_offset(o.offset):+.3f}",
                         bool(o.success)] for o in proj])
    raw_rows = sorted([[r["split"], r["session_id"], r["trial_role"],
                        f"{_round_offset(r['theta']['grasp_offset_local_y']):+.3f}",
                        bool(r["y"]["success"]), r["planned_episode_id"]] for r in raw_records])
    artifact["input_projection_hash"] = _canonical_hash(proj_rows)
    artifact["input_record_set_hash"] = _canonical_hash(raw_rows)
    artifact["candidate_bank"] = [_round_offset(o) for o in candidate_bank]
    artifact["selection_artifact_hash"] = _canonical_hash({k: artifact[k] for k in artifact})
    return artifact


# ============================ THE ONLY PRODUCTION ENTRY (one parameter, no overrides) ============================
def select_best_single_confirmatory(observed_candidate_records):
    """Frozen confirmatory production best-single. Observed outcomes ONLY; all design constants hardcoded;
    full 45/12 -> 135/36 -> 171 completeness gate always on; no bypass kwargs.

    Returns the selection artifact (dict). Raises BestSingleInputError -> EXPERIMENT_INVALID_BEST_SINGLE_INPUT
    on any completeness/leakage violation. Offset 0 is NOT hardcoded (it is the observed-count outcome)."""
    proj = _project_records(observed_candidate_records)
    _check_confirmatory_completeness(proj)
    selected, artifact = _count_select(proj, CANDIDATE_BANK)
    artifact.update({
        "train_session_count": EXPECTED_TRAIN_SESSIONS,
        "validation_session_count": EXPECTED_VAL_SESSIONS,
        "train_record_count": EXPECTED_TRAIN_RECORDS,
        "validation_record_count": EXPECTED_VAL_RECORDS,
        "completeness_gate_version": COMPLETENESS_GATE_VERSION,
        "production_entrypoint": PRODUCTION_ENTRYPOINT,
    })
    return _finalize(artifact, proj, observed_candidate_records, CANDIDATE_BANK)


# ============================ PRIVATE generic core (bridge / tests ONLY; not in __all__) ============================
def _select_best_single_core(records, *, candidate_bank=CANDIDATE_BANK, validate_completeness=True,
                             expected_candidate_records=EXPECTED_CANDIDATE_RECORDS):
    """Generic override-capable selection for the offline power bridge and tests. NOT a production entry:
    the active config and generator spec must never reference it."""
    proj = _project_records(records)
    if validate_completeness:
        _check_confirmatory_completeness(proj)
    else:
        _check_structure_only(proj, candidate_bank)
    _selected, artifact = _count_select(proj, candidate_bank)
    return _finalize(artifact, proj, records, candidate_bank)


# deprecated bridge/legacy alias (NOT in __all__, NOT the production entry)
select_best_single = _select_best_single_core
