"""Shared featurizers for models_v2. Model-legal fields ONLY (never damping/secret/hidden)."""

from __future__ import annotations

import numpy as np

THETA_KEYS = ("grasp_offset_local_y", "max_pos_step", "pull_lead")

# per-probe outcome features (order matters; used by DeepSets / GRU too)
PROBE_KEYS = (
    "theta.grasp_offset_local_y", "theta.max_pos_step", "theta.pull_lead",
    "success", "task_outcome_error", "skill_elapsed_time",
    "pull_phase_duration", "final_joint_position",
)
PROBE_DIM = len(PROBE_KEYS)


def static_features(e: dict) -> list:
    """Candidate decision inputs a deployable model may read: theta, g, observable x."""
    th = e["theta"]
    g = e["g"]
    x = e.get("x", {})
    f = [float(th[k]) for k in THETA_KEYS]
    f += [float(g.get("target_open_position", 0.0)),
          float(g.get("target_tolerance", 0.0)),
          float(x.get("initial_mechanism_joint_pos", 0.0)),
          float(x.get("gripper_width", 0.0)),
          1.0 if x.get("member") == "sektion_cabinet" else 0.0]
    return f


STATIC_DIM = 8  # len(static_features(...))


def probe_vector(h: dict) -> list:
    th = h.get("theta", {})
    return [
        float(th.get("grasp_offset_local_y", 0.0)),
        float(th.get("max_pos_step", 0.0)),
        float(th.get("pull_lead", 0.0)),
        float(h.get("success", 0.0)),
        float(h.get("task_outcome_error", 0.0)),
        float(h.get("skill_elapsed_time", 0.0)),
        float(h.get("pull_phase_duration", 0.0)),
        float(h.get("final_joint_position", 0.0)),
    ]


def history_matrix(H: list) -> np.ndarray:
    """(K, PROBE_DIM) matrix (empty -> (0, PROBE_DIM))."""
    if not H:
        return np.zeros((0, PROBE_DIM), dtype=float)
    return np.array([probe_vector(h) for h in H], dtype=float)


def history_mean(H: list) -> list:
    """Weak mean-pool summary of history (for B2Mean)."""
    M = history_matrix(H)
    if M.shape[0] == 0:
        return [0.0] * PROBE_DIM + [0.0]
    return list(M.mean(0)) + [float(M.shape[0])]


def targets(e: dict):
    y = e["y"]
    return (1.0 if y["success"] else 0.0,
            float(y["task_outcome_error"]),
            float(y["skill_elapsed_time"]))
