"""Independent-sample audit for the calibration-bias stage (offline).

Extends the damping-stage replicate audit with calibration-specific checks:
  * session-level NUISANCE PROVENANCE: does each session carry a distinct nuisance seed, and is
    the seed varied within a bias level (so replicates are not identical re-runs)?
  * replicate variance per physical cell (bias x target x offset).
  * near-duplicate outcome detection across sessions (deterministic clones).
  * effective sample count (independent samples vs technical repeats).

Blocker rule: if sessions at the same bias level are still near-deterministic clones (discrete
outcomes identical AND continuous jitter ~0 AND seeds not varied), emit blocker=True — formal CIs
may NOT be narrowed by session count and GO is not allowed.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .schema import DEFAULT_FIELDS, FieldMap, ROLE_CANDIDATE, bias_level_of, offset_key

_CONT = ("final_joint_position", "task_outcome_error", "skill_elapsed_time")


def audit(episodes, fm: FieldMap = DEFAULT_FIELDS) -> dict:
    cands = [e for e in episodes if e.get(fm.episode_role) == ROLE_CANDIDATE]
    # cell = (bias_level, target, offset) across replicate sessions
    cells = defaultdict(list)
    for e in cands:
        cells[(str(bias_level_of(e, fm)), e.get(fm.target_id), offset_key(e, fm))].append(e)

    n_cells = len(cells)
    reps = sorted({len(v) for v in cells.values()}) if cells else []
    succ_identical = fr_identical = 0
    cont_std = {f: [] for f in _CONT}
    for v in cells.values():
        if len({bool(e[fm.y]["success"]) for e in v}) == 1:
            succ_identical += 1
        if len({e[fm.y].get("failure_reason") for e in v}) == 1:
            fr_identical += 1
        for f in _CONT:
            cont_std[f].append(float(np.std([float(e[fm.y][f]) for e in v])))

    # nuisance provenance: seeds distinct across sessions? varied within a bias level?
    by_level_seeds = defaultdict(set)
    session_seed = {}
    for e in cands:
        s = e.get(fm.nuisance_seed)
        if s is not None:
            by_level_seeds[str(bias_level_of(e, fm))].add(s)
            session_seed[e.get(fm.session_id)] = s
    seeds_present = len(session_seed) > 0
    seeds_varied_within_level = all(len(v) > 1 for v in by_level_seeds.values()) if by_level_seeds else False
    distinct_session_seeds = len(set(session_seed.values())) if session_seed else 0

    # near-duplicate outcome detection: sessions whose full outcome vectors are ~identical
    dup = _near_duplicate_sessions(cands, fm)

    max_final_std = max(cont_std["final_joint_position"]) if cont_std["final_joint_position"] else 0.0
    discrete_deterministic = n_cells > 0 and succ_identical == n_cells and fr_identical == n_cells
    near_deterministic = discrete_deterministic and max_final_std < 0.005

    n_bias = len({str(bias_level_of(e, fm)) for e in cands})
    n_sessions = len({e.get(fm.session_id) for e in cands})

    blocker = bool(near_deterministic and (not seeds_present or not seeds_varied_within_level))
    return {
        "n_cells": n_cells, "replicates_per_cell": reps,
        "success_identical_cells": succ_identical, "failure_reason_identical_cells": fr_identical,
        "continuous_within_cell_max_std": {f: (max(cont_std[f]) if cont_std[f] else 0.0) for f in _CONT},
        "nuisance_provenance": {
            "seeds_present": seeds_present,
            "distinct_session_seeds": distinct_session_seeds,
            "seeds_varied_within_bias_level": seeds_varied_within_level,
        },
        "near_duplicate_sessions": dup,
        "effective_sample_sizes": {
            "n_bias_levels": n_bias, "n_unique_cells": n_cells, "n_sessions": n_sessions,
            "n_technical_repeats_per_cell": reps,
            "n_independent_sessions_estimate": distinct_session_seeds if seeds_present else n_bias,
        },
        "discrete_deterministic": discrete_deterministic,
        "near_deterministic": near_deterministic,
        "blocker": blocker,
        "blocker_reason": (
            "Sessions at the same bias level are near-deterministic clones and nuisance seeds are "
            "absent or not varied within a level -> replicates are technical repeats. Formal CIs may "
            "NOT be narrowed by session count; GO is not allowed until independent nuisance variation "
            "is added." if blocker else None),
        "resampling_recommendation": (
            "Bootstrap over independent nuisance seeds / bias-level blocks, not raw sessions; label CIs "
            "exploratory if seeds are not varied." if near_deterministic else
            "Session-level bootstrap defensible; still report per-cell descriptive stats."),
    }


def _near_duplicate_sessions(cands, fm: FieldMap, tol=1e-4) -> dict:
    """Group sessions by a rounded outcome fingerprint; report clusters with >1 session."""
    sig = defaultdict(set)
    for e in cands:
        y = e[fm.y]
        fp = (e.get(fm.target_id), offset_key(e, fm),
              bool(y["success"]), round(float(y["task_outcome_error"]), 4),
              round(float(y["final_joint_position"]), 4))
        sig[fp].add(e.get(fm.session_id))
    # a cell fingerprint shared by multiple sessions => near-duplicate at that cell
    dup_cells = sum(1 for v in sig.values() if len(v) > 1)
    return {"n_cell_fingerprints": len(sig), "n_fingerprints_shared_by_multiple_sessions": dup_cells}
