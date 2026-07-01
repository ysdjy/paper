# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Two simulated wrist cameras for the shared V1 scene.

``foundationpose_d435_rgbd`` (RGB-D, FoundationPose) and
``vla_libero_eye_in_hand`` (RGB, VLA/pi0.5, LIBERO-aligned). ``d435_config`` is
pure python; ``d435_scene_cfg`` imports Isaac lazily.
"""

from . import d435_config as config  # noqa: F401
from .d435_config import (  # noqa: F401
    D435_BODY_PRIM,
    FP_CAM_NAME,
    FP_CAM_PRIM,
    FP_PROFILE,
    FRONT_CAM_NAME,
    FRONT_CAM_PRIM,
    FRONT_PROFILE,
    PROFILES,
    V1_CAMERA_GROUP,
    V1_CAMERA_NAMES,
    VLA_CAM_NAME,
    VLA_CAM_PRIM,
    VLA_PROFILE,
    VLA_SUPPORTED_RESOLUTIONS,
    default_sensors_block,
    get_profile,
    robosuite_eye_in_hand_to_isaac_offset,
)
from .d435_observation_adapter import (  # noqa: F401
    D435ObservationAdapter,
    WristCameraAdapter,
    logical_image_dict,
)


def attach_wrist_cameras(env_cfg, **kwargs):
    """Lazy proxy to :func:`d435_scene_cfg.attach_wrist_cameras` (Isaac deferred)."""
    from .d435_scene_cfg import attach_wrist_cameras as _attach

    return _attach(env_cfg, **kwargs)


def attach_d435(env_cfg, **kwargs):
    """Back-compat proxy to the old single-camera attach."""
    from .d435_scene_cfg import attach_d435 as _attach

    return _attach(env_cfg, **kwargs)
