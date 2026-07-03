"""Replicate-independence audit (offline_v2).

Deterministic sim resets can make the N replicate sessions per (damping) TECHNICAL repeats rather
than independent statistical samples. Treating them as independent would fake-narrow every CI.

This module quantifies, per physical cell (damping x target x candidate_id):
  * whether success / failure_reason are identical across replicates (discrete determinism)
  * within-cell std / range of continuous outputs (final_pos, task_error, time, pull duration)
and reports the effective sample sizes:
  * n_unique_hidden_states   (distinct damping values)
  * n_unique_conditions      (distinct damping x target x candidate cells)
  * n_technical_repeats      (replicates per cell)
  * n_independent_sessions   (upper bound: distinct hidden-state draws)
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

_CONT = ("final_joint_position", "task_outcome_error", "skill_elapsed_time")


def _pull(e):
    return float(e["y"].get("phase_durations", {}).get("PULL", 0.0))


def audit_replicates(episodes) -> dict:
    cands = [e for e in episodes if e.get("episode_role") == "candidate"]
    # cell key: (damping, target_id, candidate_id) across replicate sessions
    cells = defaultdict(list)
    for e in cands:
        d = float((e.get("secret_deployment_state") or {}).get("damping"))
        cells[(d, e.get("target_id"), e.get("candidate_id"))].append(e)

    n_cells = len(cells)
    reps_per_cell = sorted({len(v) for v in cells.values()})
    succ_identical = 0
    fr_identical = 0
    cont_std = {f: [] for f in _CONT}
    cont_range = {f: [] for f in _CONT}
    pull_std = []
    for key, v in cells.items():
        if len({bool(e["y"]["success"]) for e in v}) == 1:
            succ_identical += 1
        if len({e["y"]["failure_reason"] for e in v}) == 1:
            fr_identical += 1
        for f in _CONT:
            arr = np.array([float(e["y"][f]) for e in v])
            cont_std[f].append(float(arr.std()))
            cont_range[f].append(float(arr.max() - arr.min()))
        parr = np.array([_pull(e) for e in v])
        pull_std.append(float(parr.std()))

    dampings = sorted({float((e.get("secret_deployment_state") or {}).get("damping")) for e in cands})
    targets = sorted({e.get("target_id") for e in cands})
    sessions = sorted({e["session_id"] for e in cands})
    replicate_ids = sorted({e.get("replicate_id") for e in cands})

    # heuristic verdict
    discrete_deterministic = (succ_identical == n_cells and fr_identical == n_cells)
    max_final_std = max(cont_std["final_joint_position"]) if cont_std["final_joint_position"] else 0.0
    near_deterministic = discrete_deterministic and max_final_std < 0.01

    return {
        "n_cells": n_cells,
        "replicates_per_cell": reps_per_cell,
        "success_identical_cells": succ_identical,
        "failure_reason_identical_cells": fr_identical,
        "continuous_within_cell": {
            f: {"max_std": max(cont_std[f]) if cont_std[f] else 0.0,
                "mean_std": float(np.mean(cont_std[f])) if cont_std[f] else 0.0,
                "max_range": max(cont_range[f]) if cont_range[f] else 0.0}
            for f in _CONT},
        "pull_duration_max_within_cell_std": max(pull_std) if pull_std else 0.0,
        "effective_sample_sizes": {
            "n_unique_hidden_states": len(dampings),
            "n_unique_conditions_cells": n_cells,
            "n_unique_target_damping_contexts": len(dampings) * len(targets),
            "n_technical_repeats_per_cell": reps_per_cell,
            "n_sessions_total": len(sessions),
            "n_independent_sessions_upper_bound": len(dampings),  # only damping is the random draw
            "n_replicates_per_level": len(replicate_ids),
        },
        "discrete_deterministic": discrete_deterministic,
        "near_deterministic": near_deterministic,
        "resampling_recommendation": (
            "Replicates are technical repeats (discrete outcomes identical, continuous jitter sub-cm). "
            "Do NOT treat the 9 sessions as 9 independent samples. Report descriptive stats per unique "
            "condition / matched group; bootstrap over target/condition blocks and label CIs EXPLORATORY."
            if near_deterministic else
            "Replicates show non-trivial variation; session-level bootstrap is defensible but still "
            "report per-condition descriptive stats."),
    }
