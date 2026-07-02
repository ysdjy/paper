# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Franka V1 scene contract package.

Single source of truth for the V1 scene: task id, control mode, object list,
skills, consumers, and the active-scene registry.
"""

from .scene_contract import (  # noqa: F401
    CONTRACT,
    SCENE_NAME,
    SCHEMA_VERSION,
    SceneContract,
    V1_CONSUMERS,
    V1_DOOR_TARGETS,
    V1_DRAWER_TARGETS,
    V1_FUNCTIONAL_DRAWERS,
    V1_GRASP_TARGETS,
    V1_IMAGE_FIELD_CAMERA,
    V1_OBJECTS,
    V1_SENSORS,
    V1_SKILLS,
    V1_WRIST_CAMERA_PARENT,
    validate_objects,
    validate_sensors,
    validate_skills,
)
from .scene_registry import (  # noqa: F401
    DEFAULT_REGISTRY_PATH,
    SceneRegistry,
    authorize_scene_write,
    default_sensors,
    load_registry,
    resolve_active_scene,
    revoke_scene_write,
    save_registry,
    update_active_scene,
)
from .init_region import (  # noqa: F401
    INIT_CORNER_NAMES,
    INIT_REGION_OBJECTS,
    load_init_region,
    point_in_region,
    sample_in_region,
)
from .v1_task_ids import (  # noqa: F401
    V1_BASE_TASK_ID,
    V1_CONTROL_MODE,
    V1_JOINT_POS_TASK_ID,
    is_v1_task,
)
