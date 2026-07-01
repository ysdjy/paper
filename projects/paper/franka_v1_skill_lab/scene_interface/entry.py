#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""scene_interface 启动 entry / 自检 demo（其他模块也可照此启动场景）。

这是 entry 用法：脚本自己建 AppLauncher（在 import isaac 之前），再把它传给
SceneSession.launch。库用法见 example_consumer.py（让 SceneSession 自己起 app）。

Run:
    ./isaaclab.sh -p projects/franka_v1_skill_lab/scene_interface/entry.py \
        --headless --enable_cameras --reset_mode region --enable_fp --steps 60 --num_resets 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECTS_DIR = Path(__file__).resolve().parents[2]   # projects/
sys.path.insert(0, str(_PROJECTS_DIR))

from franka_v1_skill_lab.scene import V1_BASE_TASK_ID  # noqa: E402
from franka_v1_skill_lab.scene_interface import ResetMode, SceneConfig  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="SceneSession 启动 / 自检 demo（V1）。")
    ap.add_argument("--task", default=V1_BASE_TASK_ID)
    ap.add_argument("--reset_mode", default="region", choices=[m.value for m in ResetMode])
    ap.add_argument("--enable_fp", action="store_true", help="渲染 depth 并填充 FoundationPose 块。")
    ap.add_argument("--no_cameras", action="store_true", help="don't attach cameras (control loop only)")
    ap.add_argument("--load_latest_scene", action="store_true",
                    help="apply saved scene_v1_latest object poses (default: base cfg poses)")
    ap.add_argument("--control_hz", type=float, default=50.0)
    ap.add_argument("--steps", type=int, default=60, help="每个 reset 后保持当前绝对关节的步数。")
    ap.add_argument("--num_resets", type=int, default=2)
    ap.add_argument("--layout_base", type=int, default=0, help="reset_index = layout_base + i。")
    ap.add_argument("--seed", type=int, default=1)
    return ap


def main() -> int:
    ap = build_arg_parser()
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    if not args.no_cameras:
        args.enable_cameras = True  # 同 eval：否则相机被丢

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    rc = _run(args, app_launcher)
    simulation_app.close()
    return rc


def _run(args, app_launcher) -> int:
    from franka_v1_skill_lab.scene_interface import SceneSession

    cfg = SceneConfig(
        task_id=args.task,
        device=args.device,
        headless=args.headless,
        control_hz=args.control_hz,
        enable_cameras=not args.no_cameras,
        enable_fp=args.enable_fp,
        reset_mode=ResetMode(args.reset_mode),
        load_latest_scene=args.load_latest_scene,
        seed=args.seed,
    )
    session = SceneSession.launch(cfg, _app_launcher=app_launcher)
    try:
        for r in range(args.num_resets):
            obs = session.reset(reset_index=args.layout_base + r)
            _print_obs(obs, f"reset {r}")
            for name, h in obs.handles.items():
                p = h["position"]
                print(f"[entry]   handle {name}: base_pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
                      f"calibrated={h['calibrated']} functional={h['functional']}", flush=True)
            for _ in range(args.steps):
                # demo：发“当前绝对关节”=保持不动；验证绝对语义（机械臂不应漂移）。
                obs = session.step(joint_target=obs.raw.arm_joint_pos[:7], gripper=0)
            _print_obs(obs, f"reset {r} after {args.steps} steps")
            if obs.foundationpose is not None:
                b = obs.foundationpose.fp_input(which="wrist")
                print(f"[entry] FP wrist bundle: has_rgb={b['camera']['has_rgb']} "
                      f"has_depth={b['camera']['has_depth']} intr={b['camera']['intrinsics']} "
                      f"objects={len(b['objects'])}", flush=True)
    finally:
        session.close()
    return 0


def _print_obs(obs, tag: str) -> None:
    p = obs.pi05
    img = list(p.image.shape) if p.image is not None else None
    wimg = list(p.wrist_image.shape) if p.wrist_image is not None else None
    fp_views = "none" if obs.foundationpose is None else list(obs.foundationpose.views)
    print(
        f"[entry] {tag}: state8={len(p.state8)}d image={img} wrist_image={wimg} "
        f"fp_views={fp_views} sim_time={obs.sim_time:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
