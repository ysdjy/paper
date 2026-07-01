# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""The Franka Skill Scene V1 contract.

This module is the SINGLE SOURCE OF TRUTH for what the V1 scene *is*: which
base task, which control mode, which robot, which objects exist, and which
skills the state machine is expected to support. Every downstream consumer
(skill runtime, teleop collection, pi0.5 training, FoundationPose perception)
imports the constants from here instead of hard-coding its own list.

Pure python, no Isaac / torch imports — safe to import from any venv and from
``scripts/check_project_layout.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .v1_task_ids import V1_BASE_TASK_ID, V1_CONTROL_MODE

SCHEMA_VERSION = "franka-scene-v1"
SCENE_NAME = "Franka Skill Scene V1"

# --- Objects ----------------------------------------------------------------
# Exactly the manipulable / interactable objects present in the V1 base env
# cfg (stack_joint_pos_env_cfg.py). These names are the env scene-entity names
# (env.unwrapped.scene[<name>]) AND the semantic-tag class names.
V1_OBJECTS: tuple[str, ...] = (
    "cube_1",
    "cube_2",
    "cube_3",
    "knife",
    "cabinet",
    "microwave",
    "coffee_machine",
)

# Grasp/place targets the skill UI exposes (subset of V1_OBJECTS that is a
# free rigid body the gripper can pick).
V1_GRASP_TARGETS: tuple[str, ...] = ("cube_1", "cube_2", "cube_3", "knife")

# Articulated appliances and their interaction skill family.
V1_ARTICULATIONS: dict[str, str] = {
    "cabinet": "drawer",       # open_drawer / close_drawer (ik_pull)
    "microwave": "door",       # open_door / close_door (ik)
    "coffee_machine": "prop",  # static prop for now (no skill yet)
}

# Drawer sub-targets exposed by the cabinet (top/middle work; bottom locked).
V1_DRAWER_TARGETS: tuple[str, ...] = ("top_drawer", "middle_drawer", "bottom_drawer")
V1_FUNCTIONAL_DRAWERS: tuple[str, ...] = ("top_drawer", "middle_drawer")

# Revolute-door targets (open_door / close_door).
V1_DOOR_TARGETS: tuple[str, ...] = ("microwave",)

# --- Skills -----------------------------------------------------------------
V1_SKILLS: tuple[str, ...] = (
    "grasp",
    "place",
    "open_drawer",
    "close_drawer",
    "open_door",
    "close_door",
)

# Skill -> backend the V1 mainline runs it with.
V1_SKILL_BACKENDS: dict[str, str] = {
    "grasp": "joint_ik",
    "place": "joint_ik",
    "open_drawer": "ik_pull",
    "close_drawer": "ik_pull",
    "open_door": "ik",
    "close_door": "ik",
}

# --- Consumers --------------------------------------------------------------
# The four downstream modules that consume the V1 scene.
V1_CONSUMERS: tuple[str, ...] = (
    "skill_runtime",
    "teleop_collection",
    "pi05_training",
    "perception_foundationpose",
)

# --- Sensors ----------------------------------------------------------------
# The TWO wrist cameras that are part of the shared V1 scene (so every consumer
# loads the same instrumented scene). Mount geometry / intrinsics live in
# ``sensors/d435/d435_config.py`` (source of truth); here we only name them.
#   foundationpose_d435_rgbd : RGB-D, FoundationPose            (wrist, panda_hand)
#   vla_libero_eye_in_hand   : RGB, VLA/pi0.5 `wrist_image`     (wrist, panda_hand)
#   vla_front_static         : RGB, VLA/pi0.5 `image` main view (fixed, env root)
V1_SENSORS: tuple[str, ...] = (
    "foundationpose_d435_rgbd",
    "vla_libero_eye_in_hand",
    "vla_front_static",
)
# The two wrist cameras hang off this Franka link; the front camera is static.
V1_WRIST_CAMERA_PARENT = "panda_hand"
# LeRobot / pi0.5 image-field -> V1 camera mapping (see pi05 dataset_mapping).
V1_IMAGE_FIELD_CAMERA = {
    "image": "vla_front_static",          # main third-person view
    "wrist_image": "vla_libero_eye_in_hand",
}


@dataclass(frozen=True)
class SceneContract:
    """Immutable description of the V1 scene contract."""

    schema_version: str = SCHEMA_VERSION
    scene_name: str = SCENE_NAME
    task_id: str = V1_BASE_TASK_ID
    control_mode: str = V1_CONTROL_MODE
    robot: str = "Franka"
    objects: tuple[str, ...] = V1_OBJECTS
    skills: tuple[str, ...] = V1_SKILLS
    consumers: tuple[str, ...] = V1_CONSUMERS
    sensors: tuple[str, ...] = V1_SENSORS

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "scene_name": self.scene_name,
            "task_id": self.task_id,
            "control_mode": self.control_mode,
            "robot": self.robot,
            "objects": list(self.objects),
            "skills": list(self.skills),
            "consumers": list(self.consumers),
            "sensors": list(self.sensors),
        }


CONTRACT = SceneContract()


def validate_objects(objects: list[str]) -> list[str]:
    """Return the names in ``objects`` that are NOT part of the V1 contract."""
    return [name for name in objects if name not in V1_OBJECTS]


def validate_skills(skills: list[str]) -> list[str]:
    """Return the names in ``skills`` that are NOT part of the V1 contract."""
    return [name for name in skills if name not in V1_SKILLS]


def validate_sensors(sensors: list[str]) -> list[str]:
    """Return the names in ``sensors`` that are NOT part of the V1 contract."""
    return [name for name in sensors if name not in V1_SENSORS]
