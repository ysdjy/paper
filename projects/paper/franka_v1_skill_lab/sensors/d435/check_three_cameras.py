# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Render + verify all THREE V1 cameras and save images for manual inspection.

Launches the V1 scene, attaches the shared cameras (front static + two wrist),
warms up rendering, captures each, and writes:

    franka_d435_foundationpose/outputs/camera_view_checks/<timestamp>/
        front_image.png        (vla_front_static     -> LeRobot `image`)
        wrist_image.png        (vla_libero_eye_in_hand-> LeRobot `wrist_image`)
        d435_rgb.png           (foundationpose_d435_rgbd RGB)
        d435_depth_vis.png     (foundationpose_d435_rgbd depth, colorized)
        camera_check_report.md

Also checks prim existence + that the front camera is NOT a child of panda_hand
(static) while the two wrist cameras ARE.

Run (headless, no UI needed):
    ./isaaclab.sh -p projects/franka_v1_skill_lab/sensors/d435/check_three_cameras.py \
        --num_envs 1 --task Isaac-Stack-Cube-Franka-JointPolicy-v0 --headless
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

_PROJECTS_DIR = Path(__file__).resolve().parents[3]   # projects/
_REPO_ROOT = _PROJECTS_DIR.parent                      # IsaacLab/
sys.path.insert(0, str(_PROJECTS_DIR))

import argparse  # noqa: E402

parser = argparse.ArgumentParser(description="Render + verify the three V1 cameras.")
parser.add_argument("--task", default="Isaac-Stack-Cube-Franka-JointPolicy-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--warmup_steps", type=int, default=14)
parser.add_argument("--scene_registry", default=None)
parser.add_argument(
    "--out_dir", default=None,
    help="Output dir (default: franka_d435_foundationpose/outputs/camera_view_checks/<timestamp>).",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Cameras must render.
if not getattr(args_cli, "enable_cameras", False):
    args_cli.enable_cameras = True
    print("[check3] auto-enabled --enable_cameras.", flush=True)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

from franka_v1_skill_lab.sensors.d435 import d435_config as cfg  # noqa: E402
from franka_v1_skill_lab.sensors.d435.d435_observation_adapter import WristCameraAdapter  # noqa: E402
from franka_v1_skill_lab.sensors.d435.d435_scene_cfg import attach_wrist_cameras  # noqa: E402

ENV_NS = "/World/envs/env_0"


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _save_rgb(rgb, path: Path) -> bool:
    if rgb is None:
        return False
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    try:
        from PIL import Image

        Image.fromarray(arr).save(str(path))
        return True
    except Exception:
        np.save(str(path.with_suffix(".npy")), arr)
        return True


def _save_depth_vis(depth, path: Path) -> bool:
    if depth is None:
        return False
    d = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(d)
    vis = np.zeros(d.shape, dtype=np.uint8)
    if finite.any():
        lo, hi = float(d[finite].min()), float(d[finite].max())
        vis = (np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    try:
        from PIL import Image

        Image.fromarray(vis).save(str(path))
        return True
    except Exception:
        np.save(str(path.with_suffix(".npy")), d)
        return True


def _prim_parent_ok(stage, prim_path: str, must_be_under: str | None, must_not_be_under: str | None):
    prim = stage.GetPrimAtPath(prim_path)
    exists = prim.IsValid()
    ok = exists
    if exists and must_be_under and must_be_under not in prim_path:
        ok = False
    if exists and must_not_be_under and must_not_be_under in prim_path:
        ok = False
    return exists, ok


def main() -> None:
    out_dir = Path(args_cli.out_dir) if args_cli.out_dir else (
        _REPO_ROOT / "franka_d435_foundationpose" / "outputs" / "camera_view_checks" / _timestamp()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.seed = args_cli.seed
    if getattr(env_cfg, "events", None) is not None and hasattr(env_cfg.events, "randomize_cube_positions"):
        env_cfg.events.randomize_cube_positions = None

    attach_wrist_cameras(env_cfg, enable_cameras=True, show_body=False, enable_front=True)
    print("[check3] attached 3 cameras (front static + 2 wrist).", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset(seed=args_cli.seed)

    action_dim = env.action_space.shape[-1]
    hold = torch.zeros((args_cli.num_envs, action_dim), device=env.unwrapped.device)
    for _ in range(max(1, args_cli.warmup_steps)):
        with torch.inference_mode():
            env.step(hold)

    import omni.usd  # noqa: E402

    stage = omni.usd.get_context().get_stage()

    front = WristCameraAdapter(env, cfg.FRONT_CAM_NAME)
    wrist = WristCameraAdapter(env, cfg.VLA_CAM_NAME)
    d435 = WristCameraAdapter(env, cfg.FP_CAM_NAME)

    front_frame = front.capture(require_depth=False)
    wrist_frame = wrist.capture(require_depth=False)
    d435_frame = d435.capture()  # depth REQUIRED -> raises if not rendered

    saved = {}
    saved["front_image.png"] = _save_rgb(front_frame["rgb"], out_dir / "front_image.png")
    saved["wrist_image.png"] = _save_rgb(wrist_frame["rgb"], out_dir / "wrist_image.png")
    saved["d435_rgb.png"] = _save_rgb(d435_frame["rgb"], out_dir / "d435_rgb.png")
    saved["d435_depth_vis.png"] = _save_depth_vis(d435_frame["depth"], out_dir / "d435_depth_vis.png")

    # prim/parent checks
    fp_path = cfg.FP_CAM_PRIM.replace("{ENV_REGEX_NS}", ENV_NS)
    vla_path = cfg.VLA_CAM_PRIM.replace("{ENV_REGEX_NS}", ENV_NS)
    front_path = cfg.FRONT_CAM_PRIM.replace("{ENV_REGEX_NS}", ENV_NS)
    front_exists, front_static_ok = _prim_parent_ok(stage, front_path, None, "panda_hand")
    vla_exists, vla_wrist_ok = _prim_parent_ok(stage, vla_path, "panda_hand", None)
    fp_exists, fp_wrist_ok = _prim_parent_ok(stage, fp_path, "panda_hand", None)

    # report
    lines = [
        f"# V1 three-camera check — {_timestamp()}",
        "",
        f"task: {args_cli.task}",
        "",
        "## Cameras",
        "| name | role | LeRobot field | parent | static | prim exists | render |",
        "|------|------|---------------|--------|--------|-------------|--------|",
        f"| {cfg.FRONT_CAM_NAME} | image | `image` | env_root | {cfg.FRONT_PROFILE.static} | {front_exists} | {saved['front_image.png']} |",
        f"| {cfg.VLA_CAM_NAME} | wrist_image | `wrist_image` | panda_hand | {cfg.VLA_PROFILE.static} | {vla_exists} | {saved['wrist_image.png']} |",
        f"| {cfg.FP_CAM_NAME} | depth | (FoundationPose) | panda_hand | {cfg.FP_PROFILE.static} | {fp_exists} | rgb={saved['d435_rgb.png']} depth={saved['d435_depth_vis.png']} |",
        "",
        "## Checks",
        f"- front camera NOT under panda_hand (static): {front_static_ok}",
        f"- wrist VLA under panda_hand: {vla_wrist_ok}",
        f"- d435 under panda_hand (aligned with wrist): {fp_wrist_ok}",
        "",
        "## Intrinsics / pose",
        f"- front:  K={front.intrinsics()}  pose={front.camera_pose_world()}",
        f"- wrist:  K={wrist.intrinsics()}  pose={wrist.camera_pose_world()}",
        f"- d435:   K={d435.intrinsics()}  pose={d435.camera_pose_world()}",
        "",
        "## Mapping",
        "- LeRobot `image`       <- vla_front_static",
        "- LeRobot `wrist_image` <- vla_libero_eye_in_hand",
        "- FoundationPose RGB-D  <- foundationpose_d435_rgbd",
    ]
    (out_dir / "camera_check_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[check3] saved images + report -> {out_dir}", flush=True)
    for k, v in saved.items():
        print(f"[check3]   {k}: {'OK' if v else 'MISSING'}", flush=True)
    print(f"[check3] front static OK={front_static_ok} | wrist on hand OK={vla_wrist_ok} | d435 on hand OK={fp_wrist_ok}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
