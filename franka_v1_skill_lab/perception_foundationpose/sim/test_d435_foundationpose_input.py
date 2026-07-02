#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Build + validate a FoundationPose input bundle from the wrist RGB-D camera.

Pure python (no Isaac, no GPU) — uses the offline mock frame of the
``foundationpose_d435_rgbd`` camera plus the sim-GT object list, so the
camera/object contract can be exercised anywhere and a sample bundle printed.

Run (offline, or under ./isaaclab.sh — both work):
    python projects/franka_v1_skill_lab/perception_foundationpose/sim/test_d435_foundationpose_input.py \
        --camera foundationpose_d435_rgbd
    # optional: --out /tmp/fp_input.json  --dump_dir /tmp/fp_imgs  --no_depth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECTS_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECTS_DIR))

from franka_v1_skill_lab.perception_foundationpose.foundationpose.foundationpose_input_builder import (  # noqa: E402
    build_foundationpose_input,
    warn_if_missing_depth,
)
from franka_v1_skill_lab.perception_foundationpose.sim.sim_gt_pose_as_foundationpose import (  # noqa: E402
    _demo_result,
)
from franka_v1_skill_lab.sensors.d435.d435_observation_adapter import WristCameraAdapter  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="wrist RGB-D -> FoundationPose input bundle (offline).")
    ap.add_argument("--camera", default="foundationpose_d435_rgbd",
                    help="Which wrist camera frame to build from (FoundationPose uses the RGB-D one).")
    ap.add_argument("--num_envs", type=int, default=1, help="(accepted for command parity; unused offline)")
    ap.add_argument("--task", default=None, help="(accepted for parity; unused offline)")
    ap.add_argument("--scene_registry", default=None, help="(accepted for parity; unused offline)")
    ap.add_argument("--no_depth", action="store_true", help="Simulate a frame with no depth (warning path).")
    ap.add_argument("--out", default=None, help="Write the bundle JSON here.")
    ap.add_argument("--dump_dir", default=None, help="Dump rgb/depth .npy here and reference them.")
    args = ap.parse_args()

    print("=== wrist RGB-D -> FoundationPose input test ===")
    if args.camera != "foundationpose_d435_rgbd":
        print(f"  NOTE: FoundationPose expects the RGB-D camera; got --camera {args.camera}.")
    frame = WristCameraAdapter.mock_frame(args.camera, with_depth=not args.no_depth)
    pose_result = _demo_result()
    tracked = [o.name for o in pose_result.objects]

    bundle = build_foundationpose_input(frame, objects=tracked, pose_source="sim_gt", dump_dir=args.dump_dir)

    cam = bundle["camera"]
    assert cam["name"] == args.camera, cam
    assert cam["frame_id"] == args.camera, cam
    for k in ("fx", "fy", "cx", "cy", "width", "height"):
        assert k in cam["intrinsics"], cam["intrinsics"]
    assert "position" in cam["camera_pose_world"] and "quat_wxyz" in cam["camera_pose_world"], cam
    assert len(bundle["objects"]) == len(tracked), bundle["objects"]
    print(f"  camera: name={cam['name']} has_rgb={cam['has_rgb']} has_depth={cam['has_depth']}")
    print(f"  intrinsics: {cam['intrinsics']}")
    print(f"  objects: {[o['name'] for o in bundle['objects']]}")

    for w in warn_if_missing_depth(bundle):
        print(f"  WARNING: {w}")

    payload = json.dumps(bundle, indent=2)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
        print(f"  wrote {args.out}")
    else:
        print("\n--- sample bundle ---")
        print(payload)

    print("\nOK — FoundationPose input bundle is well-formed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
