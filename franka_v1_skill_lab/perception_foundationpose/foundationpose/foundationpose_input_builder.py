# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Build the FoundationPose input bundle from a wrist-D435 frame + tracked objects.

STATUS: ready (pure python over a frame dict + object list; no Isaac needed when
given a frame). The frame comes from
``sensors/d435/d435_observation_adapter.py`` (live ``capture()`` or offline
``mock_frame()``); the object list comes from the V1 contract / a PoseResult.

Output (see sensors/docs/d435_foundationpose_contract.md):

    {
      "camera": {
        "frame_id", "rgb_path", "depth_path",
        "intrinsics": {fx, fy, cx, cy, width, height},
        "camera_pose_world": {"position": [...], "quat_wxyz": [...]}
      },
      "objects": [
        {"name", "mesh_path", "mask_path", "pose_source"}
      ]
    }

Images are passed by reference (paths) — in V1 they usually live in memory on
the frame, so ``rgb_path`` / ``depth_path`` are ``None`` unless ``dump_dir`` is
given, in which case the arrays are written as .npy for a real FoundationPose
process running in its own conda env.
"""

from __future__ import annotations

from pathlib import Path

# Default per-object mesh paths (relative to repo root). Real FoundationPose
# needs a textured mesh per tracked object; these point at the SAPIEN/USD
# assets already in the scene. cubes are primitives (no mesh file) -> None.
DEFAULT_MESH_PATHS = {
    "cube_1": None,
    "cube_2": None,
    "cube_3": None,
    "knife": "SapienAssetPipeline/usd_assets/Knife_101054/knife.usd",
    "cabinet": "SapienAssetPipeline/usd_assets/Cabinet_44853/cabinet.usd",
    "microwave": "SapienAssetPipeline/usd_assets/Microwave_7320/microwave_flattened.usd",
    "coffee_machine": "SapienAssetPipeline/usd_assets/CoffeeMachine_103046/coffeemachine.usd",
}

DEFAULT_TRACKED = ("cube_1", "cube_2", "cube_3", "knife")


def _maybe_dump(array, path: Path) -> str | None:
    if array is None:
        return None
    try:
        import numpy as np

        np.save(str(path), np.asarray(array))
        return str(path)
    except Exception:
        return None


def build_foundationpose_input(
    frame: dict,
    objects=DEFAULT_TRACKED,
    mesh_paths: dict | None = None,
    pose_source: str = "sim_gt_or_foundationpose",
    dump_dir: str | None = None,
) -> dict:
    """Assemble the FoundationPose input bundle.

    Args:
        frame: an RGB-D frame dict (from the wrist-D435 adapter).
        objects: iterable of tracked object names, or a list of ``PoseEntry``
            dicts (each with a ``name``) — both are accepted.
        mesh_paths: per-object mesh path overrides (merged over the defaults).
        pose_source: provenance string written into each object entry.
        dump_dir: if given, write rgb/depth arrays as .npy there and reference
            them via ``rgb_path`` / ``depth_path``.

    Returns:
        The FoundationPose input dict.
    """
    meshes = dict(DEFAULT_MESH_PATHS)
    if mesh_paths:
        meshes.update(mesh_paths)

    rgb_path = depth_path = None
    if dump_dir is not None:
        d = Path(dump_dir)
        d.mkdir(parents=True, exist_ok=True)
        rgb_path = _maybe_dump(frame.get("rgb"), d / "wrist_d435_rgb.npy")
        depth_path = _maybe_dump(frame.get("depth"), d / "wrist_d435_depth.npy")

    # Accept either a list of names or a list of PoseEntry-like dicts.
    names = []
    for o in objects:
        if isinstance(o, dict):
            names.append(o.get("name"))
        else:
            names.append(o)

    obj_entries = []
    for name in names:
        if name is None:
            continue
        obj_entries.append(
            {
                "name": name,
                "mesh_path": meshes.get(name),
                "mask_path": None,  # sim/real mask placeholder (None until a segmenter fills it)
                "pose_source": pose_source,
            }
        )

    depth = frame.get("depth")
    cam_name = frame.get("name", frame.get("frame_id", "foundationpose_d435_rgbd"))
    return {
        "camera": {
            "name": cam_name,
            "frame_id": frame.get("frame_id", cam_name),
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "has_rgb": frame.get("rgb") is not None,
            "has_depth": depth is not None,
            "intrinsics": frame.get("intrinsics", {}),
            "camera_pose_world": frame.get("camera_pose_world", {}),
        },
        "objects": obj_entries,
    }


def warn_if_missing_depth(bundle: dict) -> list[str]:
    """Return human-readable warnings (never raise) about incomplete input."""
    warnings = []
    cam = bundle.get("camera", {})
    if not cam.get("has_depth", False):
        warnings.append(
            "depth missing: FoundationPose needs RGB-D. Launch the sim with "
            "--enable_cameras (and without --no_depth) to render wrist depth."
        )
    if not cam.get("has_rgb", False):
        warnings.append("rgb missing: no color frame on the D435 input.")
    for o in bundle.get("objects", []):
        if o.get("mesh_path") is None and not o["name"].startswith("cube"):
            warnings.append(f"object {o['name']}: no mesh_path (real FoundationPose needs a mesh).")
    return warnings
