# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical calibrated geometry for the microwave revolute door.

Single source of truth (mirrors custom_drawer_config.py for the cabinet) so the env cfg (this
package) and the skill runtime (projects/franka_skill_state_machine/runtime, which re-exports this)
read the SAME numbers. Calibrated with the project's debug_microwave_door_calib.py at the deployed
microwave pose (stack_joint_pos_env_cfg.py: prim /Microwave, yaw -90deg, scale 0.35).

Facts: the door is articulation body ``link_0`` driven by revolute ``joint_0``; link_0's body origin
is the hinge and the world hinge axis is +Z (vertical swing door); the graspable free edge sits at a
constant world-metric offset in link_0's body frame.

ASSET CAVEAT: this microwave's door+body collision hulls overlap, so the door physically opens only
~10deg before contact stops it (like the locked bottom drawer). The skill is general/correct; full
opening needs the collision fixed or a different asset.

PLACEMENT CAVEAT: at the current microwave pose (world ~(0.67,-0.61), yaw -90deg) the door handle is
at world ~(0.48,-0.44,0.13), which is ~7cm beyond the Franka's comfortable -Y reach (the arm reaches
a natural pose but stalls ~0.08m short at MOVE_TO_PRE_GRASP). Move the microwave closer / more central
in the SceneLayoutModule (e.g. nearer +Y and -X) so the handle is reachable, then re-sync the cfg.
"""

from __future__ import annotations

MICROWAVE_SCALE = 0.35
DOOR_LINK = "link_0"
HINGE_JOINT = "joint_0"

# Handle (door free edge, mid height) offset in link_0's body frame, WORLD-METRIC (used by the obs
# adapter via combine_frame_transforms(link0_pos_w, link0_quat_w, this)).
# NOTE: link_0 local +X maps to world -X, so to move the handle/collider by +DX in WORLD X we shift
# the local-X component by -DX. 2026-06-12: shifted +0.0225 m in world +X (local X 0.4571 -> 0.4346)
# to recenter the narrowed collider on the graspable part of the bar (per user GUI review). Both the
# collision proxy and the IK grasp target read this, so they stay coincident.
# 2026-06-12 (Y refine): local +Z maps to world +Y, so to shrink the collider's world-Y extent toward
# the microwave (keep the near-microwave half, drop the outer half) we move its center -0.01125 m in
# world Y by lowering local-Z 0.0309 -> 0.01965 (matches halving the world-Y size below, with the
# near-microwave face held fixed).
HANDLE_OFFSET_LOCAL = (0.4346, -0.1843, 0.01965)

# Same offset / scale for the collision-proxy child prim (its authored local translate is multiplied
# by the parent /Microwave scale at spawn).
HANDLE_PROXY_LOCAL_OFFSET = tuple(v / MICROWAVE_SCALE for v in HANDLE_OFFSET_LOCAL)

# Thin vertical handle bar (world-metric local-frame dims, x,y,z). The proxy is authored in link_0's
# BODY frame; calibration (debug_microwave_handle_proxy.py) shows that frame is rotated so local +Y is
# the world-vertical (the hinge axis = world +Z = link_0 local +Y) and local +Z is world-horizontal.
# Therefore the bar's LONG (vertical) dimension must be on local Y, NOT local Z. The graspable handle
# bar is ~0.045 m thick and stands ~0.18 m tall along the door's free edge.
#   local X (=world -X, door normal/standoff): 0.0225 m (thin; halved from 0.045 per user GUI review)
#   local Y (=world +Z, vertical):             0.180 m  (the bar length, runs up the free edge)
#   local Z (=world +Y, along door width):     0.0225 m (thin; halved from 0.045, outer half dropped,
#                                               near-microwave half kept -- see offset shift above)
# (Originally (0.045,0.045,0.18) laid the 0.18 m bar HORIZONTALLY along world Y; swapping Y/Z stood it
#  upright, then the world-X and world-Y dims were each halved to hug the real graspable bar.)
HANDLE_PROXY_WORLD_SIZE = (0.0225, 0.18, 0.0225)
HANDLE_PROXY_SIZE = tuple(v / MICROWAVE_SCALE for v in HANDLE_PROXY_WORLD_SIZE)

# Door-open success angle (rad). Modest because the asset caps physical open; raise after a fix.
OPEN_SUCCESS_ANGLE = 0.14   # ~8 deg
CLOSE_SUCCESS_ANGLE = 0.03  # ~1.7 deg

# ---- Fridge (Fridge_10797) revolute door -------------------------------------------------------
# Calibrated with the project's debug_fridge_scene.py at the DEPLOYED fridge pose. In the V1 scene the
# scene member is still named "microwave" but holds the fridge (scene_props.replace_microwave_with_fridge:
# prim /Microwave -> fridge.usd, scale 0.5, world ~(0.046,-0.904)). Facts from the scan:
#   * door = articulation body link_1, driven by revolute joint_1 (limit 0..pi); link_0/base are static.
#   * world hinge axis = +Z (vertical); link local +Y = world up (= URDF joint_1 axis (0,1,0)).
#   * joint+ OPENS the door via a +Z world rotation -- OPPOSITE the microwave's -Z -- so open_sweep_sign=+1.
#   * the graspable handle is a real VERTICAL BAR mesh on the door's free edge -- NOT the slab edge.
#     Per-mesh AABB (debug_handle_meshes.py, fabric off): the bar mesh center is world (0.216,-0.627,
#     0.542), size (0.040,0.056,0.433), spanning z[0.325,0.758]; the door SLAB is the big (0.45,0.125,
#     0.74) mesh. An earlier whole-link-AABB estimate put the grasp at the slab mid-height z=0.388 --
#     ~15 cm BELOW the bar's grippable middle -- so the handle offset is taken from the bar center.
# The fridge has real (convex-decomposed) door+body collision, so unlike the microwave it swings open
# well past 10deg; accept a solid swing as success.
FRIDGE_DOOR_LINK = "link_1"
FRIDGE_HINGE_JOINT = "joint_1"
FRIDGE_HANDLE_OFFSET_LOCAL = (-0.4029, -0.2165, 0.0299)   # bar center (0.216,-0.627,0.542) in link_1 frame
FRIDGE_OPEN_SUCCESS_ANGLE = 1.047   # ~60 deg (fridge opens freely toward pi; accept a solid swing)
FRIDGE_CLOSE_SUCCESS_ANGLE = 0.05   # ~3 deg

# ---- Coffee machine (CoffeeMachine_103046) horizontally-swinging lever -------------------------
# The user placed the coffee machine in reach and set the grasp pose (grasp_poses.json:
# handle_coffee_lever, link_4, local pos (-0.08,0,0.05)). From the source URDF the lever is link_4,
# driven by REVOLUTE joint_4 (axis local -Z, limit -0.768..+1.571). The machine stands upright (only
# yaw -90 deg), so the joint axis is WORLD-VERTICAL -> the lever swings HORIZONTALLY, i.e. a vertical
# hinge exactly like the door. The door skill auto-locates the lever from link_4's meshes; the offset
# below is only a fallback (= the user's saved grasp pose, link-local).
COFFEE_LEVER_LINK = "link_5"   # 实测：joint_5/link_5 才是绕世界Z水平摆动的把手(joint_4 是绕+Y竖直的旋钮)
COFFEE_LEVER_JOINT = "joint_5"
COFFEE_HANDLE_OFFSET_LOCAL = (-0.08, 0.0, 0.05)
COFFEE_OPEN_SUCCESS_ANGLE = 0.52    # ~30 deg horizontal swing (joint_4 reaches up to ~+90 deg)
COFFEE_CLOSE_SUCCESS_ANGLE = 0.05   # ~3 deg

DOOR_TARGETS = {
    "microwave": {
        "asset_name": "microwave",
        "joint_name": HINGE_JOINT,
        "link_name": DOOR_LINK,
        "handle_offset": HANDLE_OFFSET_LOCAL,
        "open_success_angle": OPEN_SUCCESS_ANGLE,
        "close_success_angle": CLOSE_SUCCESS_ANGLE,
        "open_sweep_sign": -1.0,   # microwave: joint+ opens via a world -Z rotation
    },
    "fridge": {
        # Scene member is still keyed "microwave" (the fridge spawns in its place); asset_name must be
        # that scene key, while joint/link/offset are the fridge's.
        "asset_name": "microwave",
        "joint_name": FRIDGE_HINGE_JOINT,
        "link_name": FRIDGE_DOOR_LINK,
        "handle_offset": FRIDGE_HANDLE_OFFSET_LOCAL,
        "open_success_angle": FRIDGE_OPEN_SUCCESS_ANGLE,
        "close_success_angle": FRIDGE_CLOSE_SUCCESS_ANGLE,
        "open_sweep_sign": 1.0,    # fridge: joint+ opens via a world +Z rotation (opposite microwave)
    },
    # 注：咖啡机拉手【不再】作为 DOOR_TARGETS 条目(以前会让抓取面板多出一个重复的 handle_microwave_coffee，
    # 和 handles.py 里硬编码的 coffee_lever 指向同一个 coffee_machine/link_4)。咖啡机操作走 controller 的
    # _on_operate_coffee(直接用 joint_4 + handle_coffee_lever)，不需要这里再配一份。
}
