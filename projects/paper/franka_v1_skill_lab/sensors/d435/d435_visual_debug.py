# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Print-friendly debug for the two wrist cameras (``--show_camera_debug``).

Pure python at import time. Includes a camera-axis report so you can confirm the
VLA eye-in-hand camera is NOT pointing straight down.
"""

from __future__ import annotations

from . import d435_config as cfg
from .d435_observation_adapter import WristCameraAdapter


def _view_axes(quat_wxyz, convention: str):
    """Return (view, up, right) unit axes of the camera in the parent frame.

    ``convention`` follows IsaacLab: 'ros' (look +Z, +Y down), 'opengl' (look
    -Z, +Y up), 'world' (look +X). The returned ``view`` is the optical axis.
    """
    R = cfg.quat_to_matrix(quat_wxyz)
    col = lambda j: (R[0][j], R[1][j], R[2][j])
    x, y, z = col(0), col(1), col(2)
    neg = lambda v: tuple(-c for c in v)
    if convention == "opengl":
        return neg(z), y, x          # look -Z, up +Y
    if convention == "world":
        return x, z, neg(y)          # look +X
    return z, neg(y), x              # 'ros': look +Z, up -Y


def camera_axis_lines(profile) -> list[str]:
    view, up, right = _view_axes(profile.offset_rot, profile.convention)
    straight_down = abs(view[2] + 1.0) < 0.15  # view ≈ (0,0,-1) in parent frame
    return [
        f"{profile.name}: convention={profile.convention} "
        f"{'(approx pose)' if profile.pose_is_approximate else ''}",
        f"    offset pos      = {tuple(round(v,4) for v in profile.offset_pos)}",
        f"    offset quat_wxyz= {tuple(round(v,4) for v in profile.offset_rot)}",
        f"    view  axis      = {tuple(round(v,3) for v in view)}",
        f"    up    axis      = {tuple(round(v,3) for v in up)}",
        f"    right axis      = {tuple(round(v,3) for v in right)}",
        f"    straight-down?  = {straight_down}  (True only if optical axis ≈ world -Z)",
    ]


def static_summary_lines() -> list[str]:
    lines = [
        f"camera group    : {cfg.V1_CAMERA_GROUP}",
        f"D435 body       : {cfg.D435_BODY_PRIM} (visual only, HIDDEN by default, no collider)",
    ]
    for name in cfg.V1_CAMERA_NAMES:
        p = cfg.PROFILES[name]
        intr = p.intrinsics()
        lines.append(
            f"{p.name}: consumer={p.consumer} kind={p.kind} "
            f"res={intr['width']}x{intr['height']} fx={intr['fx']:.1f} depth={p.depth} prim={p.prim_path}"
        )
    return lines


def print_camera_axes(tag: str = "cam_axes") -> None:
    """Offline: print each camera's 3 axes (no Isaac needed)."""
    print(f"[{tag}] ---- wrist camera axes ----", flush=True)
    for name in cfg.V1_CAMERA_NAMES:
        for line in camera_axis_lines(cfg.PROFILES[name]):
            print(f"[{tag}]   {line}", flush=True)
    print(f"[{tag}] ---------------------------", flush=True)


def print_camera_debug(env=None, camera_name: str | None = None, tag: str = "cam") -> None:
    """Print the wrist-camera summary + live availability/pose/intrinsics."""
    print(f"[{tag}] ---- wrist cameras ----", flush=True)
    for line in static_summary_lines():
        print(f"[{tag}]   {line}", flush=True)

    names = [camera_name] if camera_name else list(cfg.V1_CAMERA_NAMES)
    for name in names:
        adapter = WristCameraAdapter(env=env, camera_name=name)
        available = adapter.is_available()
        print(f"[{tag}]   {name}: live={'AVAILABLE' if available else 'not attached (need --enable_cameras)'}", flush=True)
        if available:
            intr = adapter.intrinsics()
            pose = adapter.camera_pose_world()
            print(
                f"[{tag}]     intrinsics fx={intr['fx']:.1f} fy={intr['fy']:.1f} "
                f"cx={intr['cx']:.1f} cy={intr['cy']:.1f} ({intr['width']}x{intr['height']})",
                flush=True,
            )
            print(
                f"[{tag}]     pose_w pos={['%.3f' % v for v in pose['position']]} "
                f"quat_wxyz={['%.3f' % v for v in pose['quat_wxyz']]}",
                flush=True,
            )
    print(f"[{tag}] ---------------------------", flush=True)
