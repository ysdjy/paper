"""Microwave door config for the state-machine runtime.

Single source of truth lives in the task package so the env cfg (source/) and this runtime read the
SAME geometry:
    stackpkg.franka.microwave_door_config

See that module for the calibration notes and the asset caveat (door physically opens only ~10deg
on this asset). Calibrate with entries/debug_microwave_door_calib.py.
"""

from __future__ import annotations

from isaaclab_tasks.manager_based.manipulation.stack.config.franka.microwave_door_config import (  # noqa: F401
    CLOSE_SUCCESS_ANGLE,
    COFFEE_LEVER_JOINT,
    COFFEE_LEVER_LINK,
    DOOR_LINK,
    DOOR_TARGETS,
    HANDLE_OFFSET_LOCAL,
    HINGE_JOINT,
    MICROWAVE_SCALE,
    OPEN_SUCCESS_ANGLE,
)


def door_config(target: str) -> dict:
    if target not in DOOR_TARGETS:
        raise KeyError(f"unknown door target '{target}'; available={list(DOOR_TARGETS)}")
    return DOOR_TARGETS[target]
