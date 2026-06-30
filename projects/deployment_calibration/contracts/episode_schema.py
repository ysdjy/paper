"""Frozen episode schema for deployment-conditioned plan evaluation (paper round-1, open_drawer).

Implements `docs/paper_experiment_contract_v1.md`. One executed candidate = (x, g, theta, H) -> y,
with the four variable groups kept STRICTLY separate. Pure python (no Isaac), so it is importable in
both the data-generation entry and the offline evaluation.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

CONTRACT_VERSION = "open_drawer_v1"

# theta search ranges (frozen, contract section 1) -- effective params only (close_duration dropped)
THETA_RANGES = {
    "max_pos_step": (0.008, 0.035),
    "pull_lead": (0.03, 0.14),
    "pre_grasp_clearance": (0.06, 0.18),
    "approach_line_lead": (0.01, 0.06),
    "grasp_offset_local_xyz_y": (-0.06, 0.06),  # sampled on Y (edge-grasp axis); x,z held 0 in round-1
}
TARGET_OPEN_LEVELS = (0.10, 0.18, 0.26, 0.34)
TARGET_TOLERANCE = 0.02

FAILURE_REASONS = ["NONE", "REACH_TIMEOUT", "PULL_TIMEOUT", "HANDLE_DETACHED", "INVALID_PARAM", "OTHER"]


def normalize_failure(reason: str | None) -> str:
    if not reason:
        return "NONE"
    r = reason.upper()
    if "DRAWER_OPEN_TIMEOUT" in r or "PULL" in r:
        return "PULL_TIMEOUT"
    if "POSITION_TIMEOUT" in r or "REACH" in r:
        return "REACH_TIMEOUT"
    if "HANDLE_DETACH" in r:
        return "HANDLE_DETACHED"
    if "INVALID_EXPERIMENT_PARAMETER" in r:
        return "INVALID_PARAM"
    if r in FAILURE_REASONS:
        return r
    return "OTHER"


@dataclass
class Episode:
    episode_id: str
    contract_version: str
    skill: str
    seed: int
    session_id: str            # deployment session (a contiguous run sharing drift state)
    reset_index: int           # groups identical initial conditions (for split + pairing)
    order_in_session: int
    # four variable groups
    x: dict[str, Any]          # initial_state (deploy-available + privileged flagged)
    g: dict[str, Any]          # task target
    theta: dict[str, Any]      # execution params (effective value actually used)
    y: dict[str, Any]          # outcomes
    # provenance
    requested_parameters: dict[str, Any] = field(default_factory=dict)
    effective_parameter_traces: list[dict[str, Any]] = field(default_factory=list)
    controller: dict[str, Any] = field(default_factory=dict)
    trajectory_file: str | None = None
    full_reset: bool = True

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    @property
    def success(self) -> bool:
        return bool(self.y.get("success"))


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def load_episodes(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
