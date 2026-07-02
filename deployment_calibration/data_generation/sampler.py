"""Candidate (reset_index, drawer, g, theta) sampling for open_drawer round-1.

Pure python + numpy (no Isaac). A "plan" is a list of episode specs. We use a deterministic RNG so
the whole dataset is reproducible from a seed. For the candidate-selection evaluation we also expose
``candidate_group`` (same x,g -> N theta candidates) so regret can be computed.
"""

from __future__ import annotations

import numpy as np

from contracts.episode_schema import THETA_RANGES, TARGET_OPEN_LEVELS


def _uniform(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def sample_theta(rng) -> dict:
    r = THETA_RANGES
    gy = _uniform(rng, *r["grasp_offset_local_xyz_y"])
    return {
        "max_pos_step": round(_uniform(rng, *r["max_pos_step"]), 4),
        "pull_lead": round(_uniform(rng, *r["pull_lead"]), 4),
        "pre_grasp_clearance": round(_uniform(rng, *r["pre_grasp_clearance"]), 4),
        "approach_line_lead": round(_uniform(rng, *r["approach_line_lead"]), 4),
        "grasp_offset_local_xyz": [0.0, round(gy, 4), 0.0],
    }


def build_plan(seed: int, n_conditions: int, candidates_per_condition: int,
               drawers=("top_drawer",), reset_index_base: int = 0) -> list[dict]:
    """n_conditions distinct (reset_index, drawer, g); each with C theta candidates.

    Episodes that share (reset_index, drawer, g) form one candidate group for ranking/regret.
    """
    rng = np.random.default_rng(seed)
    plan = []
    gid = 0
    for c in range(n_conditions):
        reset_index = reset_index_base + c
        drawer = drawers[c % len(drawers)]
        g_target = float(TARGET_OPEN_LEVELS[rng.integers(0, len(TARGET_OPEN_LEVELS))])
        group_id = f"cond_{gid:04d}"
        gid += 1
        for k in range(candidates_per_condition):
            plan.append({
                "reset_index": reset_index,
                "drawer": drawer,
                "g": {"target_open_position": g_target, "drawer_name": drawer},
                "theta": sample_theta(rng),
                "candidate_group": group_id,
                "candidate_k": k,
            })
    return plan
