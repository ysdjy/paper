"""Serializable skill request model."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import torch

from runtime.skill_types import SkillType


def _pose_to_list(pose: torch.Tensor | None) -> list[float] | None:
    if pose is None:
        return None
    return pose.detach().cpu().reshape(-1).tolist()


@dataclass
class SkillRequest:
    request_id: str
    skill_type: SkillType
    source_object: str | None
    destination_type: str | None = None
    destination_object: str | None = None
    target_pose: torch.Tensor | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "skill_type": self.skill_type.value,
            "source_object": self.source_object,
            "destination_type": self.destination_type,
            "destination_object": self.destination_object,
            "target_pose": _pose_to_list(self.target_pose),
            "parameters": self.parameters,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any], device: str | torch.device = "cpu") -> "SkillRequest":
        pose = data.get("target_pose")
        return cls(
            request_id=str(data["request_id"]),
            skill_type=SkillType(data["skill_type"]),
            source_object=data.get("source_object"),
            destination_type=data.get("destination_type"),
            destination_object=data.get("destination_object"),
            target_pose=torch.tensor(pose, dtype=torch.float32, device=device) if pose is not None else None,
            parameters=dict(data.get("parameters", {})),
        )

    @classmethod
    def from_json(cls, payload: str, device: str | torch.device = "cpu") -> "SkillRequest":
        return cls.from_dict(json.loads(payload), device=device)
