# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Per-episode frame buffer for V1 teleop collection. STATUS: ready.

Pure-python (stdlib only) so it is unit-testable without Isaac / h5py / numpy.
The driver appends one :class:`Frame`-shaped dict per recorded sim step; on
success it hands the buffered episode to :mod:`hdf5_writer`.

Frame schema (per step) — matches teleop_collection/data_format/demo_hdf5_schema.md:
    {
      "timestamp": float,
      "step_id": int,
      "robot": {joint_pos[7], joint_vel[7], ee_pose[7] (x,y,z,qx,qy,qz,qw), gripper_width},
      "teleop": {raw_q[7], filtered_q[7]},
      "action": {joint_target[7], gripper},
      "images": {"wrist_rgb": HxWx3|None, "wrist_depth": HxW|None},
      "objects": [{"name", "pos":[3], "quat_wxyz":[4]}, ...],
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class FrameBuilder:
    """Assemble one record dict from raw values (no numpy needed).

    All vector args accept lists or anything with ``.tolist()`` (torch / numpy).
    """

    @staticmethod
    def _as_list(x, n: int | None = None) -> list[float]:
        if x is None:
            return [] if n is None else [0.0] * n
        try:
            x = x.detach().cpu().tolist()
        except AttributeError:
            try:
                x = x.tolist()
            except AttributeError:
                x = list(x)
        x = [float(v) for v in x]
        if n is not None:
            if len(x) < n:
                x = x + [0.0] * (n - len(x))
            x = x[:n]
        return x

    @classmethod
    def build(
        cls,
        *,
        step_id: int,
        timestamp: float,
        joint_pos,
        joint_vel,
        ee_pose,
        gripper_width: float,
        raw_q,
        filtered_q,
        joint_target,
        gripper: float,
        images: dict | None = None,
        objects: list[dict] | None = None,
    ) -> dict:
        return {
            "timestamp": float(timestamp),
            "step_id": int(step_id),
            "robot": {
                "joint_pos": cls._as_list(joint_pos, 7),
                "joint_vel": cls._as_list(joint_vel, 7),
                "ee_pose": cls._as_list(ee_pose, 7),
                "gripper_width": float(gripper_width),
            },
            "teleop": {
                "raw_q": cls._as_list(raw_q, 7),
                "filtered_q": cls._as_list(filtered_q, 7),
            },
            "action": {
                "joint_target": cls._as_list(joint_target, 7),
                "gripper": float(gripper),
            },
            "images": images or {},
            "objects": objects or [],
        }


@dataclass
class EpisodeRecorder:
    """Accumulates frames for the current episode + episode-level metadata."""

    episode_id: str = ""
    task_instruction: str = ""
    skill_type: str = ""
    target_name: str = ""
    scene_registry: str = ""
    seed: int = 0
    record_images: bool = False
    record_depth: bool = False
    record_objects: bool = False
    frames: list[dict] = field(default_factory=list)

    def reset(self, *, episode_id: str, task_instruction: str, skill_type: str,
              target_name: str, scene_registry: str, seed: int) -> None:
        self.episode_id = episode_id
        self.task_instruction = task_instruction
        self.skill_type = skill_type
        self.target_name = target_name
        self.scene_registry = scene_registry
        self.seed = seed
        self.frames = []

    def append(self, frame: dict) -> None:
        self.frames.append(frame)

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def num_steps(self) -> int:
        return len(self.frames)

    def episode_meta(self, success: bool) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_instruction": self.task_instruction,
            "skill_type": self.skill_type,
            "target_name": self.target_name,
            "scene_registry": self.scene_registry,
            "seed": int(self.seed),
            "success": bool(success),
            "num_steps": self.num_steps,
            "has_images": self.record_images,
            "has_depth": self.record_depth,
            "has_objects": self.record_objects,
        }
