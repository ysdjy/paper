#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Headless smoke test of the sorting pick-and-place: grasp the banana, place it into the fruits
basket (KLT_3), using the same open-loop runners the GUI's "Grasp (Z-approach)" / "Place to Basket"
buttons use. Reports whether the banana ends up inside the basket and the place orientation (z-down).

Run:
    ./isaaclab.sh -p projects/franka_v1_skill_lab/scene_interface/test_banana_sort.py --headless
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECTS_DIR))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="Prop_011_banana")
    ap.add_argument("--basket", default="Prop_KLT_3")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    app = AppLauncher(args)
    rc = _run(args, app)
    app.app.close()
    return rc


def _run(args, app) -> int:
    import torch

    from franka_v1_skill_lab.scene import V1_BASE_TASK_ID
    from franka_v1_skill_lab.scene_interface import ResetMode, SceneConfig, SceneMode, SceneSession

    cfg = SceneConfig(
        mode=SceneMode.TEST, task_id=V1_BASE_TASK_ID, device=args.device, headless=True,
        enable_cameras=False, enable_fp=False, apply_saved_camera_offsets=False,
        load_latest_scene=True, replace_microwave_with_fridge=True, refine_handle_collisions=True,
        add_robot_stand=True, lock_knife=True,
        control_hz=50.0, reset_mode=ResetMode.STATIC, seed=1,
    )
    session = SceneSession.launch(cfg, _app_launcher=app)
    env, provider = session.env, session.provider
    session.reset()

    import isaaclab.utils.math as mu
    from franka_v1_skill_lab.skill_runtime._legacy import ensure_legacy_on_path

    ensure_legacy_on_path()
    from runtime.ik_joint_adapter import IKJointAdapter
    from runtime.scene_state_provider import PoseState
    from state_machine.skill_test_controller import _GraspPoseRunner, _PlacePoseRunner

    adapter = IKJointAdapter(env)
    scene = provider.scene
    dev = adapter.device

    def P(*a):
        print(*a, flush=True)

    # --- saved grasp poses (reference_local) ---
    gp_path = _PROJECTS_DIR / "franka_v1_skill_lab/scene/saved_scenes/v1_active/grasp_poses.json"
    poses = json.loads(gp_path.read_text())["poses"]

    def world_from_saved(name):
        e = poses[name]
        member, link = e["member"], e.get("link")
        if member not in scene.keys():
            return None
        asset = scene[member]
        if link is None:
            rp, rq = asset.data.root_pos_w[0], asset.data.root_quat_w[0]
        else:
            names = list(asset.data.body_names)
            i = next((k for k, n in enumerate(names) if link in n), 0)
            rp, rq = asset.data.body_pos_w[0, i], asset.data.body_quat_w[0, i]
        lp = torch.tensor(e["pos"], dtype=torch.float32, device=dev).reshape(1, 3)
        lq = torch.tensor(e["quat"], dtype=torch.float32, device=dev).reshape(1, 4)
        wp = rp.reshape(1, 3) + mu.quat_apply(rq.reshape(1, 4), lp)
        wq = mu.quat_mul(rq.reshape(1, 4), lq)
        return PoseState(wp[0], wq[0])

    grasp_pose = world_from_saved(args.object)
    basket_pos = scene[args.basket].data.root_pos_w[0]
    obj_member = poses[args.object]["member"]
    obj0 = scene[obj_member].data.root_pos_w[0].clone()
    P(f"[test] object={args.object} world0=({obj0[0]:.3f},{obj0[1]:.3f},{obj0[2]:.3f})")
    P(f"[test] grasp pose=({grasp_pose.pos_w[0]:.3f},{grasp_pose.pos_w[1]:.3f},{grasp_pose.pos_w[2]:.3f})")
    P(f"[test] basket={args.basket} world=({basket_pos[0]:.3f},{basket_pos[1]:.3f},{basket_pos[2]:.3f})")

    clock = [0.0]

    def drive(runner, max_steps=1200, label=""):
        for _ in range(max_steps):
            st = provider.get_state()
            q, grip = runner.step(st, 0.02)
            if runner.done:
                act = provider.make_hold_joint_action(st, grip)
            elif q is None:
                act = provider.make_hold_joint_action(st, grip)
            else:
                act = provider.make_joint_action_from_q_des(q, grip)
            env.step(act)
            clock[0] += 0.02
            provider.set_sim_time(clock[0])
            if runner.done:
                P(f"[test] {label} done (phase={runner.phase})")
                return True
        P(f"[test] {label} did NOT finish (phase={runner.phase})")
        return False

    P("\n=== GRASP banana ===")
    drive(_GraspPoseRunner(adapter, grasp_pose, device=dev), label="grasp")
    held = scene[obj_member].data.root_pos_w[0].clone()
    tcp = provider.get_state().robot.tcp_pose
    lifted = float(held[2] - obj0[2])
    P(f"[test] after grasp: object z gain={lifted:.3f}m (>0.02 = gripped & lifted)")

    P("\n=== PLACE into basket (z-down) ===")
    pr = _PlacePoseRunner(adapter, basket_pos, device=dev)
    drive(pr, label="place")
    # check place orientation was z-down: TCP +Z . world-down
    tcpq = provider.get_state().robot.tcp_pose.quat_w
    zc = mu.quat_apply(tcpq.reshape(1, 4), torch.tensor([[0.0, 0.0, 1.0]], device=dev))[0]
    zdown = float(zc[2])   # -1 = perfectly down
    final = scene[obj_member].data.root_pos_w[0].clone()
    dxy = float(((final[0] - basket_pos[0]) ** 2 + (final[1] - basket_pos[1]) ** 2) ** 0.5)
    P(f"\n[test] RESULT: object final=({final[0]:.3f},{final[1]:.3f},{final[2]:.3f}) "
      f"basket=({basket_pos[0]:.3f},{basket_pos[1]:.3f},{basket_pos[2]:.3f})")
    P(f"[test] object planar dist to basket center = {dxy*100:.1f}cm  (small = in/near basket)")
    P(f"[test] place TCP +Z . world_down = {-zdown:.2f}  (1.0 = perfectly z-down)")
    P(f"[test] VERDICT: {'IN BASKET' if dxy < 0.12 else 'NOT in basket'}, "
      f"z-down={'YES' if -zdown > 0.9 else 'NO (%.2f)' % -zdown}")
    session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
