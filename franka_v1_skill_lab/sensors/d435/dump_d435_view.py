# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Capture the wrist-D435 view (RGB + depth) to image files.

This is the easy way to actually SEE what the simulated wrist D435 records: it
launches the V1 scene with the camera attached (sensors.d435.attach_d435),
steps a few frames so the renderer fills the camera buffers, captures one frame
via D435ObservationAdapter, and writes:

    <out_dir>/wrist_d435_rgb.png          (color)
    <out_dir>/wrist_d435_depth.png        (depth, viridis-ish colormap)
    <out_dir>/wrist_d435_depth.npy        (raw float32 meters)
    <out_dir>/wrist_d435_frame.json       (intrinsics + camera pose)

Runs headless (no display needed) — perfect for a workstation over SSH.

Run:
    ./isaaclab.sh -p projects/franka_v1_skill_lab/sensors/d435/dump_d435_view.py \
        --num_envs 1 --task Isaac-Stack-Cube-Franka-JointPolicy-v0 \
        --enable_cameras --headless \
        --out_dir projects/franka_v1_skill_lab/data/d435_view

Note: ``--enable_cameras`` is REQUIRED (the RGB-D camera only renders with it).
The script adds it automatically if you forget.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# Make franka_v1_skill_lab importable (projects/ on path).
_PROJECTS_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECTS_DIR))

parser = argparse.ArgumentParser(description="Capture the wrist D435 view to image files.")
parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-JointPolicy-v0", help="V1 base task id.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (use 1).")
parser.add_argument("--seed", type=int, default=1, help="Deterministic scene seed.")
parser.add_argument("--camera", default="foundationpose_d435_rgbd",
                    choices=["foundationpose_d435_rgbd", "vla_libero_eye_in_hand"],
                    help="Which wrist camera to capture.")
parser.add_argument("--no_depth", action="store_true", default=False, help="Capture RGB only (no depth).")
parser.add_argument("--warmup_steps", type=int, default=12, help="Sim steps before capture (fill render buffers).")
parser.add_argument(
    "--out_dir",
    type=str,
    default="projects/franka_v1_skill_lab/data/d435_view",
    help="Where to write the captured images.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# The wrist D435 only renders with camera rendering enabled — force it on.
if not getattr(args_cli, "enable_cameras", False):
    args_cli.enable_cameras = True
    print("[dump_d435] auto-enabled --enable_cameras (required to render the camera).", flush=True)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything that needs the Isaac app below."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

from franka_v1_skill_lab.sensors.d435 import d435_config as d435cfg  # noqa: E402
from franka_v1_skill_lab.sensors.d435.d435_observation_adapter import WristCameraAdapter  # noqa: E402
from franka_v1_skill_lab.sensors.d435.d435_scene_cfg import attach_wrist_cameras  # noqa: E402
from franka_v1_skill_lab.sensors.d435.d435_visual_debug import print_camera_debug  # noqa: E402


def _save_rgb(rgb, path: Path) -> bool:
    try:
        import numpy as np
        from PIL import Image

        arr = np.asarray(rgb)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        Image.fromarray(arr).save(str(path))
        return True
    except Exception as exc:
        print(f"[dump_d435] could not write RGB png ({exc}); see the .npy fallback.", flush=True)
        try:
            import numpy as np

            np.save(str(path.with_suffix(".npy")), np.asarray(rgb))
        except Exception:
            pass
        return False


def _save_depth(depth, png_path: Path, npy_path: Path) -> None:
    import numpy as np

    d = np.asarray(depth, dtype=np.float32)
    np.save(str(npy_path), d)
    # colorize finite depth to 0..255 for a quick look
    finite = np.isfinite(d)
    vis = np.zeros(d.shape, dtype=np.uint8)
    if finite.any():
        lo = float(d[finite].min())
        hi = float(d[finite].max())
        rng = max(hi - lo, 1e-6)
        norm = np.clip((d - lo) / rng, 0.0, 1.0)
        vis = (norm * 255.0).astype(np.uint8)
    try:
        from PIL import Image

        Image.fromarray(vis).save(str(png_path))
    except Exception as exc:
        print(f"[dump_d435] could not write depth png ({exc}); raw depth is in {npy_path.name}.", flush=True)


def main() -> None:
    out_dir = Path(args_cli.out_dir)
    if not out_dir.is_absolute():
        out_dir = (_PROJECTS_DIR.parent / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.seed = args_cli.seed
    if getattr(env_cfg, "events", None) is not None and hasattr(env_cfg.events, "randomize_cube_positions"):
        env_cfg.events.randomize_cube_positions = None

    attach_wrist_cameras(env_cfg, enable_cameras=True, show_body=False, enable_fp_depth=not args_cli.no_depth)
    print("[dump_d435] both wrist cameras attached (D435 body hidden).", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset(seed=args_cli.seed)

    # Hold still (zero action -> default joint targets) while the renderer fills
    # the camera buffers over a few steps.
    action_dim = env.action_space.shape[-1]
    hold = torch.zeros((args_cli.num_envs, action_dim), device=env.unwrapped.device)
    for _ in range(max(1, args_cli.warmup_steps)):
        with torch.inference_mode():
            env.step(hold)

    print_camera_debug(env, camera_name=args_cli.camera, tag="dump_d435")

    adapter = WristCameraAdapter(env, camera_name=args_cli.camera)
    if not adapter.is_available():
        print(
            "[dump_d435] ERROR: wrist D435 not available on the live scene. "
            "Did the camera attach? (it needs --enable_cameras).",
            flush=True,
        )
        env.close()
        return

    frame = adapter.capture()

    # --- write outputs ------------------------------------------------------
    import json

    if frame.get("rgb") is not None:
        rgb_path = out_dir / "wrist_d435_rgb.png"
        if _save_rgb(frame["rgb"], rgb_path):
            print(f"[dump_d435] wrote RGB   : {rgb_path}", flush=True)
    else:
        print("[dump_d435] WARNING: no RGB on the frame.", flush=True)

    if frame.get("depth") is not None:
        depth_png = out_dir / "wrist_d435_depth.png"
        depth_npy = out_dir / "wrist_d435_depth.npy"
        _save_depth(frame["depth"], depth_png, depth_npy)
        print(f"[dump_d435] wrote depth : {depth_png} (+ raw {depth_npy.name})", flush=True)
    else:
        print(
            "[dump_d435] WARNING: depth is None — depth only renders with --enable_cameras "
            "and without --no_depth.",
            flush=True,
        )

    meta = {
        "frame_id": frame["frame_id"],
        "intrinsics": frame["intrinsics"],
        "camera_pose_world": frame["camera_pose_world"],
        "timestamp": frame["timestamp"],
    }
    meta_path = out_dir / "wrist_d435_frame.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[dump_d435] wrote meta  : {meta_path}", flush=True)
    print(f"[dump_d435] DONE -> {out_dir}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
