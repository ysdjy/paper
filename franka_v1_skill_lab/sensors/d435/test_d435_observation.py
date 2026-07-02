#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Test / capture the two wrist cameras.

Two modes:

* OFFLINE (default, no Isaac, no GPU) — validates the camera config + adapter
  contract for BOTH cameras and the LIBERO axis (not straight down). This is the
  smoke-test entry:
      python projects/franka_v1_skill_lab/sensors/d435/test_d435_observation.py

* LIVE (needs Isaac + a GPU) — pass ``--save_debug_outputs`` to launch the V1
  scene, attach both cameras, capture ``--camera`` and write debug files:
      ./isaaclab.sh -p .../test_d435_observation.py --num_envs 1 \
          --camera foundationpose_d435_rgbd --save_debug_outputs
      ./isaaclab.sh -p .../test_d435_observation.py --num_envs 1 \
          --camera vla_libero_eye_in_hand --vla_camera_resolution 128 128 --save_debug_outputs

FoundationPose camera depth is REQUIRED: if the renderer did not produce it,
``capture()`` raises ``Depth output is not available for foundationpose_d435_rgbd.``
rather than silently returning None.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECTS_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECTS_DIR))

DEBUG_OUT_DIR = Path(__file__).resolve().parent / "debug_outputs"


# ===========================================================================
# OFFLINE contract test
# ===========================================================================
def _run_offline() -> int:
    from franka_v1_skill_lab.sensors.d435 import d435_config as cfg
    from franka_v1_skill_lab.sensors.d435.d435_observation_adapter import (
        WristCameraAdapter,
        logical_image_dict,
    )
    from franka_v1_skill_lab.sensors.d435.d435_visual_debug import camera_axis_lines

    print("=== V1 cameras contract test (offline) ===")

    # three cameras exist with the right roles
    assert set(cfg.V1_CAMERA_NAMES) == {
        "foundationpose_d435_rgbd", "vla_libero_eye_in_hand", "vla_front_static"
    }, cfg.V1_CAMERA_NAMES
    assert cfg.FP_PROFILE.kind == "rgbd" and cfg.FP_PROFILE.depth is True
    assert cfg.VLA_PROFILE.kind == "rgb" and cfg.VLA_PROFILE.depth is False
    print("  three cameras present (FP=rgbd, VLA wrist=rgb, front=rgb) OK")

    # front camera: static (NOT on panda_hand), role=image, 256x256
    fr = cfg.FRONT_PROFILE
    assert fr.static is True and fr.parent == "env_root", (fr.static, fr.parent)
    assert "panda_hand" not in fr.prim_path, fr.prim_path
    assert fr.role == "image", fr.role
    fi = fr.intrinsics()
    assert fi["width"] == 256 and fi["height"] == 256, fi
    assert "images.front_rgb" in fr.image_keys and "images.image" in fr.image_keys, fr.image_keys
    print("  front camera OK (static under env root, role=image, 256x256 -> LeRobot `image`)")
    # the two wrist cameras ARE on panda_hand
    assert "panda_hand" in cfg.VLA_PROFILE.prim_path and "panda_hand" in cfg.FP_PROFILE.prim_path
    print("  wrist cameras OK (both on panda_hand)")

    # FP intrinsics 640x480
    fp = cfg.FP_PROFILE.intrinsics()
    assert fp["width"] == 640 and fp["height"] == 480, fp
    print("  FP intrinsics 640x480 fx=%.1f OK" % fp["fx"])

    # VLA supports 128 and 224
    v128 = cfg.VLA_PROFILE.intrinsics(128, 128)
    v224 = cfg.VLA_PROFILE.intrinsics(224, 224)
    assert v128["width"] == 128 and v224["width"] == 224, (v128, v224)
    assert v224["fx"] > v128["fx"], (v128, v224)
    print("  VLA intrinsics 128 (fx=%.1f) + 224 (fx=%.1f) OK" % (v128["fx"], v224["fx"]))

    # LIBERO pose is NOT straight down
    for line in camera_axis_lines(cfg.VLA_PROFILE):
        if "straight-down?" in line:
            assert "False" in line, line
    assert cfg.VLA_PROFILE.libero_compatible is True
    print("  VLA pose aligned to LIBERO eye_in_hand, not straight down OK")

    # FP mock frame has rgb + depth + intrinsics + pose; depth key maps
    fpf = WristCameraAdapter.mock_frame("foundationpose_d435_rgbd")
    for k in ("name", "rgb", "depth", "intrinsics", "camera_pose_world", "frame_id"):
        assert k in fpf, k
    assert fpf["rgb"] is not None and fpf["depth"] is not None
    fp_keys = logical_image_dict(fpf)
    assert "images.wrist_depth" in fp_keys, fp_keys
    print("  FP mock frame OK (rgb+depth -> images.wrist_depth)")

    # VLA mock frame -> the three LIBERO-compatible rgb keys
    vf = WristCameraAdapter.mock_frame("vla_libero_eye_in_hand")
    vk = logical_image_dict(vf)
    for key in ("images.wrist_rgb", "images.eye_in_hand_rgb", "images.robot0_eye_in_hand_rgb"):
        assert key in vk, vk
    assert vf["depth"] is None
    print("  VLA mock frame OK (rgb -> wrist_rgb / eye_in_hand_rgb / robot0_eye_in_hand_rgb)")

    # registry descriptor shape
    d = cfg.default_sensors_block()
    assert d["foundationpose_d435_rgbd"]["consumer"] == "foundationpose"
    assert d["vla_libero_eye_in_hand"]["consumer"] == "vla_pi05"
    assert d["foundationpose_d435_rgbd"]["visible_body"] is False
    print("  registry descriptors OK (consumers + visible_body=false)")

    print("\nOK — wrist cameras contract is consistent.")
    print("(For live RGB-D capture run this under ./isaaclab.sh with --save_debug_outputs.)")
    return 0


# ===========================================================================
# LIVE capture (Isaac)
# ===========================================================================
def _save_rgb(rgb, path: Path) -> None:
    import numpy as np

    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    try:
        from PIL import Image

        Image.fromarray(arr).save(str(path))
    except Exception as exc:
        print(f"[test_cam] PIL unavailable ({exc}); saving RGB as .npy.", flush=True)
        np.save(str(path.with_suffix(".npy")), arr)


def _save_depth(depth, npy_path: Path, png_path: Path) -> None:
    import numpy as np

    d = np.asarray(depth, dtype=np.float32)
    np.save(str(npy_path), d)
    finite = np.isfinite(d)
    vis = np.zeros(d.shape, dtype=np.uint8)
    if finite.any():
        lo, hi = float(d[finite].min()), float(d[finite].max())
        vis = (np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    try:
        from PIL import Image

        Image.fromarray(vis).save(str(png_path))
    except Exception as exc:
        print(f"[test_cam] PIL unavailable ({exc}); depth_visual skipped (raw in {npy_path.name}).", flush=True)


def _run_live(argv: list[str]) -> int:
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Capture a wrist camera to debug files.")
    parser.add_argument("--task", default="Isaac-Stack-Cube-Franka-JointPolicy-v0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scene_registry", default=None, help="Active V1 scene registry (provenance).")
    parser.add_argument("--camera", default="foundationpose_d435_rgbd",
                        choices=["foundationpose_d435_rgbd", "vla_libero_eye_in_hand", "vla_front_static"])
    parser.add_argument("--vla_camera_resolution", type=int, nargs=2, default=None, metavar=("W", "H"))
    parser.add_argument("--front_resolution", type=int, nargs=2, default=None, metavar=("W", "H"),
                        help="render the front camera at this resolution (use high res for FoundationPose: "
                             "the front cam is ~2.4m away, so cubes are tiny at 256x256). e.g. 1280 960.")
    parser.add_argument("--warmup_steps", type=int, default=12)
    parser.add_argument("--save_debug_outputs", action="store_true")
    parser.add_argument("--out_dir", default=str(DEBUG_OUT_DIR))
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)

    if not getattr(args, "enable_cameras", False):
        args.enable_cameras = True
        print("[test_cam] auto-enabled --enable_cameras (required to render the camera).", flush=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym  # noqa: E402
    import torch  # noqa: E402

    import isaaclab_tasks  # noqa: F401,E402
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

    from franka_v1_skill_lab.sensors.d435.d435_observation_adapter import WristCameraAdapter  # noqa: E402
    from franka_v1_skill_lab.sensors.d435.d435_scene_cfg import attach_wrist_cameras  # noqa: E402
    from franka_v1_skill_lab.sensors.d435.d435_visual_debug import print_camera_debug  # noqa: E402

    # provenance
    try:
        from franka_v1_skill_lab.scene import resolve_active_scene

        info = resolve_active_scene(args.scene_registry)
        print(f"[test_cam] active scene sensors={sorted((info.get('sensors') or {}).keys())}", flush=True)
    except Exception as exc:
        print(f"[test_cam] scene registry resolve skipped ({exc}).", flush=True)

    env_cfg = parse_env_cfg(
        args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric
    )
    env_cfg.seed = args.seed
    if getattr(env_cfg, "events", None) is not None and hasattr(env_cfg.events, "randomize_cube_positions"):
        env_cfg.events.randomize_cube_positions = None

    vla_res = tuple(args.vla_camera_resolution) if args.vla_camera_resolution else None
    # If the FRONT camera is the capture target, render depth on it too so it can
    # feed FoundationPose (front view is RGB-only by default).
    want_front_depth = args.camera == "vla_front_static"
    front_res = tuple(args.front_resolution) if args.front_resolution else None
    attach_wrist_cameras(env_cfg, enable_cameras=True, show_body=False,
                         vla_resolution=vla_res, enable_front_depth=want_front_depth,
                         front_resolution=front_res)
    print("[test_cam] both wrist cameras attached.", flush=True)

    env = gym.make(args.task, cfg=env_cfg)
    env.reset(seed=args.seed)

    action_dim = env.action_space.shape[-1]
    hold = torch.zeros((args.num_envs, action_dim), device=env.unwrapped.device)
    for _ in range(max(1, args.warmup_steps)):
        with torch.inference_mode():
            env.step(hold)

    print_camera_debug(env, camera_name=args.camera, tag="test_cam")

    adapter = WristCameraAdapter(env, camera_name=args.camera)
    # require depth for any camera we save for FoundationPose (front cam now renders it)
    frame = adapter.capture(require_depth=True)  # raises if depth is missing (never silent None)

    if args.save_debug_outputs:
        import json

        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = (_PROJECTS_DIR.parent / out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        cam = args.camera

        if frame.get("rgb") is not None:
            _save_rgb(frame["rgb"], out_dir / f"{cam}_rgb.png")
            print(f"[test_cam] wrote {cam}_rgb.png", flush=True)
        if frame.get("depth") is not None:
            _save_depth(frame["depth"], out_dir / f"{cam}_depth.npy", out_dir / f"{cam}_depth_visual.png")
            print(f"[test_cam] wrote {cam}_depth.npy + {cam}_depth_visual.png", flush=True)
        else:
            print(f"[test_cam] note: {cam} has no depth (RGB-only camera).", flush=True)

        (out_dir / f"{cam}_intrinsics.json").write_text(json.dumps(frame["intrinsics"], indent=2) + "\n")
        (out_dir / f"{cam}_camera_pose_world.json").write_text(
            json.dumps(frame["camera_pose_world"], indent=2) + "\n"
        )
        print(f"[test_cam] wrote {cam}_intrinsics.json + {cam}_camera_pose_world.json", flush=True)
        print(f"[test_cam] DONE -> {out_dir}", flush=True)

    env.close()
    simulation_app.close()
    return 0


def main() -> int:
    argv = sys.argv[1:]
    live = "--save_debug_outputs" in argv
    if live:
        return _run_live(argv)
    return _run_offline()


if __name__ == "__main__":
    raise SystemExit(main())
