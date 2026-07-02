# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Unified 6D object-pose entry schema shared by sim-GT and real FoundationPose.

Pure python. Both the sim ground-truth adapter (stage 1) and the real
FoundationPose runner (stage 2) emit a list of these, so the pi0.5 observation
adapter and the skill-runtime debug pose source consume ONE format regardless
of where the pose came from.

Entry JSON contract:
    {
      "name": "cube_1",
      "position": [x, y, z],          # world frame, meters
      "quat": [x, y, z, w],           # world frame
      "confidence": 1.0,
      "pose_in_camera": [[..4x4..]] or null,
      "mesh_path": "..." or null,
      "mask_path": "..." or null
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PoseEntry:
    name: str
    position: list[float]                     # [x, y, z] world, meters
    quat: list[float]                         # [x, y, z, w] world
    confidence: float = 1.0
    pose_in_camera: list[list[float]] | None = None   # 4x4 or None
    mesh_path: str | None = None
    mask_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "position": [float(v) for v in self.position],
            "quat": [float(v) for v in self.quat],
            "confidence": float(self.confidence),
            "pose_in_camera": self.pose_in_camera,
            "mesh_path": self.mesh_path,
            "mask_path": self.mask_path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PoseEntry":
        return cls(
            name=data["name"],
            position=list(data["position"]),
            quat=list(data["quat"]),
            confidence=float(data.get("confidence", 1.0)),
            pose_in_camera=data.get("pose_in_camera"),
            mesh_path=data.get("mesh_path"),
            mask_path=data.get("mask_path"),
        )


@dataclass
class PoseResult:
    """A full perception output: a list of PoseEntry plus provenance."""

    objects: list[PoseEntry] = field(default_factory=list)
    source: str = "sim_gt"        # "sim_gt" | "foundationpose" | "mock"
    frame: str = "world"          # frame the `position`/`quat` live in

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "frame": self.frame,
            "objects": [o.to_dict() for o in self.objects],
        }

    def names(self) -> list[str]:
        return [o.name for o in self.objects]


def validate_entry(data: dict) -> list[str]:
    """Return a list of problems with a candidate pose-entry dict (empty = ok)."""
    problems = []
    for key in ("name", "position", "quat", "confidence"):
        if key not in data:
            problems.append(f"missing key: {key}")
    if "position" in data and len(data["position"]) != 3:
        problems.append("position must have 3 elements")
    if "quat" in data and len(data["quat"]) != 4:
        problems.append("quat must have 4 elements [x,y,z,w]")
    return problems
