#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""最小参考消费者（库用法）：其他模块照这样用 scene_interface。

与 entry.py 不同：这里**不自己建 AppLauncher**，由 SceneSession.launch 内部起 app
（纯库用法）。仍需在 Isaac python 下跑：

    ./isaaclab.sh -p projects/franka_v1_skill_lab/scene_interface/example_consumer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # projects/

from franka_v1_skill_lab.scene_interface import ResetMode, SceneConfig, SceneSession  # noqa: E402


def main() -> int:
    cfg = SceneConfig(headless=True, enable_cameras=True, enable_fp=True, reset_mode=ResetMode.REGION)
    with SceneSession.launch(cfg) as session:        # AppLauncher 在 launch 内部起
        obs = session.reset(reset_index=0)
        print("[example] reset state8 =", [round(v, 4) for v in obs.pi05.state8], flush=True)
        print("[example] image =", None if obs.pi05.image is None else obs.pi05.image.shape,
              "wrist_image =", None if obs.pi05.wrist_image is None else obs.pi05.wrist_image.shape, flush=True)

        # pi0.5 推理位点：把 obs.pi05.image / wrist_image / state8 发给 policy server，
        # 拿回 7 维绝对关节 q_des + grip(0/1)，再 session.step。这里用“保持当前关节”代替策略，
        # 验证绝对关节语义（机械臂不应漂移）。
        for _ in range(50):
            obs = session.step(joint_target=obs.pi05.state8[:7], gripper=0)
        print("[example] after 50 hold-steps state8 =", [round(v, 4) for v in obs.pi05.state8], flush=True)

        # FoundationPose 位点：同一相机的 RGBD（与 VLA-RGB 同视角）。
        if obs.foundationpose is not None:
            for which in ("front", "wrist"):
                b = obs.foundationpose.fp_input(which=which)
                cam = b["camera"]
                print(f"[example] FP {which}: has_rgb={cam['has_rgb']} has_depth={cam['has_depth']} "
                      f"objects={[o['name'] for o in b['objects']]}", flush=True)
            print("[example] object GT poses:", obs.foundationpose.object_gt_poses, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
