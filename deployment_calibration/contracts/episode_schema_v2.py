"""Episode contract v2 for drawer deployment calibration.

One execution: (x, g, theta, H, z_secret) -> y.
z_secret is for data audit / Oracle ONLY -- it MUST NOT enter ordinary model input.

Does not modify episode_schema.py (v1 kept as legacy). See docs/paper_experiment_contract_v2.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

CONTRACT_VERSION = "open_drawer_v2"

# --- theta: round-1 keeps ONLY 3 sampled dims; the rest are FIXED for efficiency ---
THETA_RANGES = {
    "grasp_offset_local_y": (-0.06, 0.06),   # edge-grasp axis (drawer LINK local +Y)
    "max_pos_step": (0.008, 0.035),          # DLS IK per-step cap
    "pull_lead": (0.03, 0.14),               # pull carrot distance
}
THETA_FIXED = {
    "pre_grasp_clearance": 0.12,
    "approach_line_lead": 0.03,
    "reach_timeout": 16.0,
    "pull_timeout": 16.0,
    "close_duration": 1.0,
    "settle_duration": 0.5,
}
THETA_SAMPLED_KEYS = tuple(THETA_RANGES.keys())

TARGET_OPEN_LEVELS = (0.12, 0.20, 0.28)      # 2-3 representative g values
TARGET_TOLERANCE = 0.02

FAILURE_REASONS = ["NONE", "REACH_TIMEOUT", "PULL_TIMEOUT", "POSITION_TIMEOUT",
                   "HANDLE_DETACHED", "INVALID_PARAM", "EPISODE_EXCEPTION", "OTHER"]

# episode_role values
ROLE_PROBE = "probe"
ROLE_CANDIDATE = "candidate"

# fields that are deploy-observable (allowed in x). Everything else about the mechanism is hidden.
X_OBSERVABLE_KEYS = [
    "mechanism_id", "drawer_name", "robot_joint_pos", "tcp_pos", "tcp_quat",
    "gripper_width", "initial_mechanism_joint_pos", "member",
]


def normalize_failure(reason: str | None) -> str:
    if not reason:
        return "NONE"
    r = str(reason).upper()
    for k in FAILURE_REASONS:
        if k in r:
            return k
    if "DRAWER_OPEN_TIMEOUT" in r:
        return "PULL_TIMEOUT"
    if "DRAWER_CLOSE_TIMEOUT" in r:
        return "PULL_TIMEOUT"
    return "OTHER"


@dataclass
class HistoryEntry:
    """One probe result visible in H (decision-time-legal only)."""
    g: dict[str, Any]
    theta: dict[str, Any]
    success: bool
    failure_reason: str
    task_outcome_error: float
    skill_elapsed_time: float
    pull_phase_duration: float
    handle_relative_error: float
    mechanism_id: str
    probe_index: int = -1


@dataclass
class EpisodeV2:
    # identity
    episode_id: str
    contract_version: str
    skill: str                       # "open_drawer"
    seed: int
    session_id: str                  # deployment session (fixed hidden state within)
    mechanism_id: str                # e.g. "cabinet:top_drawer"
    drawer_name: str
    episode_role: str                # ROLE_PROBE | ROLE_CANDIDATE
    reset_index: int                 # groups identical initial conditions
    order_in_session: int

    # decision inputs
    x: dict[str, Any]                # observable initial state (X_OBSERVABLE_KEYS)
    g: dict[str, Any]                # {target_open_position, target_tolerance}
    theta: dict[str, Any]            # effective params actually used (3 sampled + fixed)
    history_cutoff: int              # number of probe results legally visible before this decision

    # outcome
    y: dict[str, Any]                # outcomes (see build_outcome)

    # audit / oracle ONLY -- never in model input
    secret_deployment_state: dict[str, Any] = field(default_factory=dict)
    hidden_state_id: str = ""        # e.g. "damping_L1"

    # grouping
    candidate_group: str = ""
    candidate_index: int = -1
    probe_index: int = -1

    # provenance
    scene_version: str = ""
    git_commit: str = ""
    dirty_worktree: bool = False
    full_reset_verified: bool = False
    requested_parameters: dict[str, Any] = field(default_factory=dict)
    effective_parameter_traces: list[dict[str, Any]] = field(default_factory=list)
    controller: dict[str, Any] = field(default_factory=dict)
    trajectory_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # hard guard: secret must not leak into x
        for k in list(d["x"].keys()):
            if "damping" in k or "secret" in k or "hidden" in k:
                raise ValueError(f"x contains privileged key '{k}' -- leakage")
        return _json_safe(d)


def build_x(mechanism_id, drawer_name, member, robot_joint_pos, tcp_pos, tcp_quat,
            gripper_width, initial_mechanism_joint_pos) -> dict[str, Any]:
    return {
        "mechanism_id": mechanism_id, "drawer_name": drawer_name, "member": member,
        "robot_joint_pos": list(robot_joint_pos), "tcp_pos": list(tcp_pos), "tcp_quat": list(tcp_quat),
        "gripper_width": float(gripper_width),
        "initial_mechanism_joint_pos": float(initial_mechanism_joint_pos),
    }


def build_outcome(success, failure_reason, final_joint_position, task_outcome_error, overshoot,
                  skill_elapsed_time, phase_durations, handle_detached, handle_relative_error,
                  command_tracking_error, phase_goal_error, wall_clock_time) -> dict[str, Any]:
    return {
        "success": bool(success), "failure_reason": normalize_failure(failure_reason),
        "final_joint_position": float(final_joint_position), "task_outcome_error": float(task_outcome_error),
        "overshoot": float(overshoot), "skill_elapsed_time": float(skill_elapsed_time),
        "phase_durations": dict(phase_durations or {}), "handle_detached": bool(handle_detached),
        "handle_relative_error": float(handle_relative_error),
        "command_tracking_error": float(command_tracking_error), "phase_goal_error": float(phase_goal_error),
        "wall_clock_time": float(wall_clock_time),
    }


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return obj
    return obj
