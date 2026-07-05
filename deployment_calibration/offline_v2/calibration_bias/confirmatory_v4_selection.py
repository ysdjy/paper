"""Production best-single selector for confirmatory v4 (offline; observed-outcome-only; NO secret).

FIX2 / BLOCKER_PRODUCTION_BEST_SINGLE_STILL_SECRET_DEPENDENT.

The simulation helper `learned_selector_power.best_single_legal` reconstructs the success label from the
SECRET hidden state via `succ(nominal, residual, offset, tau)`. That is a `SIMULATION_ONLY_REFERENCE` and
must NOT be the confirmatory production implementation. This module is the frozen production selector: it
reads ONLY observed candidate outcomes (`y.success`) and never touches nominal / residual / actual bias /
eff / tau / oracle / a success model / test records.

Allowlisted input fields (everything else, incl. secret audit fields, is projected away first):
    split, session_id, trial_role, theta.grasp_offset_local_y, y.success, planned_episode_id
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

SELECTION_RULE_VERSION = "confirmatory_v4_best_single_observed_v1"
CANDIDATE_BANK = (-0.04, 0.0, 0.04)

# frozen confirmatory input completeness (train 45 + validation 12 sessions, 3 candidates each)
EXPECTED_TRAIN_SESSIONS = 45
EXPECTED_VAL_SESSIONS = 12
EXPECTED_SESSIONS = EXPECTED_TRAIN_SESSIONS + EXPECTED_VAL_SESSIONS       # 57
EXPECTED_CANDIDATE_RECORDS = EXPECTED_SESSIONS * len(CANDIDATE_BANK)       # 171


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


def _project(records) -> list:
    """Allowlist-project raw records -> ObservedCandidateOutcome. Reads ONLY the 6 legal fields.

    Rejects (BestSingleInputError) any record that is not a train/validation candidate trial or whose
    observed success is not a bool. Secret audit fields present on the raw record are never read."""
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
            raise BestSingleInputError(f"record {i} trial_role={role!r} (only 'candidate' allowed; probe/"
                                       "test excluded)")
        if split not in ("train", "validation"):
            raise BestSingleInputError(f"record {i} split={split!r} (only train/validation; test forbidden)")
        if not isinstance(success, bool):
            raise BestSingleInputError(f"record {i} y.success is not bool: {success!r}")
        out.append(ObservedCandidateOutcome(split, session_id, pid, offset, success))
    return out


def _check_completeness(proj, candidate_bank, expected_records):
    bank = tuple(_round_offset(o) for o in candidate_bank)
    if len(proj) != expected_records:
        raise BestSingleInputError(f"expected {expected_records} candidate records, got {len(proj)}")
    by_session = {}
    seen_pid = set()
    seen_session_offset = set()
    for o in proj:
        if o.planned_episode_id in seen_pid:
            raise BestSingleInputError(f"duplicate planned_episode_id {o.planned_episode_id}")
        seen_pid.add(o.planned_episode_id)
        key = (o.session_id, o.offset)
        if key in seen_session_offset:
            raise BestSingleInputError(f"duplicate (session_id, offset) {key}")
        seen_session_offset.add(key)
        by_session.setdefault(o.session_id, []).append(o)
    if len(by_session) != expected_records // len(bank):
        raise BestSingleInputError(f"expected {expected_records // len(bank)} sessions, "
                                   f"got {len(by_session)}")
    for sid, recs in by_session.items():
        if len(recs) != len(bank):
            raise BestSingleInputError(f"session {sid} has {len(recs)} candidates, expected {len(bank)}")
        if tuple(sorted(_round_offset(r.offset) for r in recs)) != tuple(sorted(bank)):
            raise BestSingleInputError(f"session {sid} offset set != bank {bank}")


def _core_select(proj, candidate_bank):
    """Observed-count selection. Denominators are equal (one trial per offset per session), so we rank by
    integer observed-success COUNT. Returns (selected_offset, artifact_partial)."""
    bank = [_round_offset(o) for o in candidate_bank]
    n_per = len(proj) // len(bank)
    counts = {o: 0 for o in bank}
    for o in proj:
        off = _round_offset(o.offset)
        if o.success:
            counts[off] += 1
    max_c = max(counts[o] for o in bank)
    tie1 = [o for o in bank if counts[o] == max_c]
    min_abs = min(abs(o) for o in tie1)
    tie2 = [o for o in tie1 if abs(abs(o) - min_abs) < 1e-12]
    selected = min(tie2, key=lambda o: bank.index(o))       # earliest in bank order
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
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def select_best_single(observed_candidate_records, *, candidate_bank=CANDIDATE_BANK,
                       validate_completeness=True, expected_candidate_records=EXPECTED_CANDIDATE_RECORDS):
    """Frozen production best-single. Observed outcomes ONLY. No tau/nominal/residual/secret.

    Returns the selection artifact (dict) incl. hashes. Raises BestSingleInputError on any completeness or
    leakage violation. `validate_completeness=False` (used only by the power bridge) skips the fixed-171
    confirmatory gate but keeps per-session structural checks."""
    proj = _project(observed_candidate_records)
    if validate_completeness:
        _check_completeness(proj, candidate_bank, expected_candidate_records)
    else:
        # still enforce per-session 3-candidate structure and uniqueness (design-agnostic)
        by_s = {}
        seen = set()
        for o in proj:
            k = (o.session_id, _round_offset(o.offset))
            if k in seen:
                raise BestSingleInputError(f"duplicate (session,offset) {k}")
            seen.add(k)
            by_s.setdefault(o.session_id, []).append(o)
        bank = tuple(sorted(_round_offset(x) for x in candidate_bank))
        for sid, recs in by_s.items():
            if tuple(sorted(_round_offset(r.offset) for r in recs)) != bank:
                raise BestSingleInputError(f"session {sid} offset set != bank")
    selected, artifact = _core_select(proj, candidate_bank)
    # provenance hashes
    proj_rows = sorted([[o.split, o.session_id, o.planned_episode_id, f"{_round_offset(o.offset):+.3f}",
                         bool(o.success)] for o in proj])
    raw_rows = sorted([[r["split"], r["session_id"], r["trial_role"],
                        f"{_round_offset(r['theta']['grasp_offset_local_y']):+.3f}",
                        bool(r["y"]["success"]), r["planned_episode_id"]] for r in observed_candidate_records])
    artifact["input_projection_hash"] = _canonical_hash(proj_rows)
    artifact["input_record_set_hash"] = _canonical_hash(raw_rows)
    artifact["candidate_bank"] = [_round_offset(o) for o in candidate_bank]
    body = {k: artifact[k] for k in artifact}
    artifact["selection_artifact_hash"] = _canonical_hash(body)
    return artifact
