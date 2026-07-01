# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical target_drawer -> joint / link / handle config for the custom Cabinet_44853.

Single source of truth, importable from both the RL env (this package) and the state-machine
runtime (scripts/.../skill_runtime/drawer_target_config.py re-exports this).

Confirmed by scripts/environments/state_machine/debug_drawer_joint_scan.py:
    top_drawer    = joint_0 / link_0 (z~0.684)  -> opens (world -X), functional
    middle_drawer = joint_2 / link_2 (z~0.391)  -> opens (world -X), functional
    bottom_drawer = joint_1 / link_1 (z~0.111)  -> LOCKED (penetration at closed), NOT functional yet
All prismatic, axis=Z(local), limits [0,0.8] -> closed=0, open_direction=+1. Gripper +1 open / -1 close.
"""

from __future__ import annotations

CABINET_USD_SCALE = 0.62

# Front-face / graspable handle offset in each drawer LINK's local frame, calibrated by
# debug_drawer_handle_calib.py (handle = mesh AABB min world-X face center, drawers open toward -X).
# This SAME offset is applied in training (drawer_frames FrameTransformer + custom_drawer_mdp via the
# sensor) and deployment (SelectedDrawerObsAdapter), so handle poses match across train/deploy.
HANDLE_LOCAL_OFFSET = {
    # front-face (handle) offsets in each drawer LINK local frame, calibrated by
    # debug_drawer_handle_calib.py on the rotated cabinet (front = min world-Y). All three resolve to
    # the same front face (world x~0.272, y~0.53) at descending heights, so grasp poses are consistent.
    "top_drawer": (-0.0733, -0.0053, 0.0395),
    "middle_drawer": (0.0306, 0.0389, 0.6824),
    "bottom_drawer": (0.0741, 0.0220, 0.6737),  # now at the front face (was 0,0,0 = link origin/back)
}

DRAWER_TARGETS = {
    "top_drawer": {
        "display_name_zh": "上抽屉",
        "joint_name": "joint_0",
        "link_name": "link_0",
        "handle_frame": "top_drawer_handle",
        "handle_offset": HANDLE_LOCAL_OFFSET["top_drawer"],
        "success_threshold": 0.20,
        "closed_pos": 0.0,
        "open_direction": 1,
        "functional": True,
    },
    "middle_drawer": {
        "display_name_zh": "中抽屉",
        "joint_name": "joint_2",
        "link_name": "link_2",
        "handle_frame": "middle_drawer_handle",
        "handle_offset": HANDLE_LOCAL_OFFSET["middle_drawer"],
        "success_threshold": 0.20,
        "closed_pos": 0.0,
        "open_direction": 1,
        "functional": True,
    },
    "bottom_drawer": {
        "display_name_zh": "下抽屉",
        "joint_name": "joint_1",
        "link_name": "link_1",
        "handle_frame": "bottom_drawer_handle",
        "handle_offset": HANDLE_LOCAL_OFFSET["bottom_drawer"],
        "success_threshold": 0.20,
        "closed_pos": 0.0,
        "open_direction": 1,
        "functional": False,  # locked in current asset; fix collision/closed-offset before training
    },
}

# order used by the RL env's drawer_frames FrameTransformer + selected-drawer index sampling
FUNCTIONAL_DRAWERS = [k for k, v in DRAWER_TARGETS.items() if v["functional"]]
DEFAULT_TARGET = "top_drawer"


def get_drawer_config(target_drawer: str) -> dict:
    if target_drawer not in DRAWER_TARGETS:
        raise KeyError(f"unknown target_drawer '{target_drawer}'; valid={list(DRAWER_TARGETS)}")
    return DRAWER_TARGETS[target_drawer]
