"""Serializable skill result model."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import torch

from runtime.skill_types import ExecutionStatus, SkillType


def _tensor_to_list(value: torch.Tensor | None) -> list[float] | None:
    if value is None:
        return None
    return value.detach().cpu().reshape(-1).tolist()


@dataclass
class SkillResult:
    request_id: str
    skill_type: SkillType
    target_name: str | None
    success: bool
    final_status: ExecutionStatus
    failure_reason: str | None = None
    elapsed_time: float = 0.0
    final_tcp_pose: torch.Tensor | None = None
    final_object_pose: torch.Tensor | None = None
    position_error: float | None = None
    orientation_error: float | None = None
    gripper_width: float | None = None
    state_history: list[dict[str, Any]] = field(default_factory=list)
    # --- stage-1 experiment extension (all optional, backward compatible) ---
    # ``outcomes`` holds the multi-dimensional execution result (object errors, settling time,
    # tracking error, drawer position error, handle_detached, etc.) per the stage-1 schema.
    # ``legacy_reached`` is a DEBUG-only flag: did the old state machine reach the end-of-open-gripper
    # position. It must NEVER be used as ``success`` (success is the verified object/mechanism state).
    outcomes: dict[str, Any] = field(default_factory=dict)
    legacy_reached: bool | None = None
    task_target: dict[str, Any] = field(default_factory=dict)
    requested_parameters: dict[str, Any] = field(default_factory=dict)
    effective_parameters: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "skill_type": self.skill_type.value,
            "target_name": self.target_name,
            "success": self.success,
            "final_status": self.final_status.value,
            "failure_reason": self.failure_reason,
            "elapsed_time": self.elapsed_time,
            "final_tcp_pose": _tensor_to_list(self.final_tcp_pose),
            "final_object_pose": _tensor_to_list(self.final_object_pose),
            "position_error": self.position_error,
            "orientation_error": self.orientation_error,
            "gripper_width": self.gripper_width,
            "state_history": self.state_history,
            "outcomes": self.outcomes,
            "legacy_reached": self.legacy_reached,
            "task_target": self.task_target,
            "requested_parameters": self.requested_parameters,
            "effective_parameters": self.effective_parameters,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any], device: str | torch.device = "cpu") -> "SkillResult":
        tcp_pose = data.get("final_tcp_pose")
        object_pose = data.get("final_object_pose")
        return cls(
            request_id=str(data["request_id"]),
            skill_type=SkillType(data["skill_type"]),
            target_name=data.get("target_name"),
            success=bool(data["success"]),
            final_status=ExecutionStatus(data["final_status"]),
            failure_reason=data.get("failure_reason"),
            elapsed_time=float(data.get("elapsed_time", 0.0)),
            final_tcp_pose=torch.tensor(tcp_pose, dtype=torch.float32, device=device) if tcp_pose else None,
            final_object_pose=torch.tensor(object_pose, dtype=torch.float32, device=device) if object_pose else None,
            position_error=data.get("position_error"),
            orientation_error=data.get("orientation_error"),
            gripper_width=data.get("gripper_width"),
            state_history=list(data.get("state_history", [])),
            outcomes=dict(data.get("outcomes", {})),
            legacy_reached=data.get("legacy_reached"),
            task_target=dict(data.get("task_target", {})),
            requested_parameters=dict(data.get("requested_parameters", {})),
            effective_parameters=list(data.get("effective_parameters", [])),
        )

    @classmethod
    def from_json(cls, payload: str, device: str | torch.device = "cpu") -> "SkillResult":
        return cls.from_dict(json.loads(payload), device=device)
