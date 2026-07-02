# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""D435 RGB-D observation adapter — MOVED.

The canonical implementation now lives in the shared sensors package:

    franka_v1_skill_lab/sensors/d435/d435_observation_adapter.py

so the SAME wrist-D435 camera is used by every V1 consumer (skill runtime,
teleop, pi0.5, FoundationPose). This module re-exports it for backwards
compatibility with the older ``perception_foundationpose.d435`` import path.

See sensors/d435/README.md for the mount geometry, intrinsics, and frame
contract, and sensors/docs/d435_foundationpose_contract.md for how this feeds
FoundationPose.
"""

from __future__ import annotations

from franka_v1_skill_lab.sensors.d435.d435_observation_adapter import (  # noqa: F401
    D435ObservationAdapter,
    logical_image_dict,
)

# Legacy alias: the old stub exposed an ``RGBDFrame`` dataclass. The unified
# adapter uses a plain dict frame (documented in sensors/d435/README.md); keep a
# tiny shim so old code that imported the name does not break at import time.
from dataclasses import dataclass


@dataclass
class RGBDFrame:  # pragma: no cover - compatibility shim only
    rgb: object
    depth: object
    K: object
    width: int
    height: int
    T_base_camera: object | None = None
