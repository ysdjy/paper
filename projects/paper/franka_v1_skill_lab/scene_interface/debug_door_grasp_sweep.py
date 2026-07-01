#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Sweep the door grasp orientation (and other redundancy knobs) in ONE Isaac session to find the
configuration that opens the fridge door WIDEST without the wrist twisting / losing grip.

For each candidate config it resets the arm to home + the door to closed, runs OpenDoorIKSkill to
completion, and records the MAX door angle reached and the peak wrist tilt during the pull. Prints a
ranked summary so we can pick the best grasp pose / redundancy branch.

Run:
    ./isaaclab.sh -p projects/franka_v1_skill_lab/scene_interface/debug_door_grasp_sweep.py --headless
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_PROJECTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECTS_DIR))


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rolls", default="0,45,90,135,180,225,270,315",
                    help="comma list of grasp_roll_deg to try")
    ap.add_argument("--pitch", default="0", help="comma list of approach_pitch_deg to try")
    ap.add_argument("--max_steps", type=int, default=1400, help="per-attempt step budget")
    return ap


def main() -> int:
    ap = build_arg_parser()
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    app_launcher = AppLauncher(args)
    rc = _run(args, app_launcher)
    app_launcher.app.close()
    return rc


def _run(args, app_launcher) -> int:
    import torch

    from franka_v1_skill_lab.scene import V1_BASE_TASK_ID
    from franka_v1_skill_lab.scene_interface import ResetMode, SceneConfig, SceneMode, SceneSession

    cfg = SceneConfig(
        mode=SceneMode.TEST, task_id=V1_BASE_TASK_ID, device=args.device, headless=True,
        enable_cameras=False, enable_fp=False, apply_saved_camera_offsets=False,
        free_microwave_door=False, load_latest_scene=True, add_microwave_stand=False,
        replace_microwave_with_fridge=True, lock_knife=True, spawn_init_markers=False,
        refine_handle_collisions=True, control_hz=50.0, reset_mode=ResetMode.STATIC, seed=args.seed,
    )
    session = SceneSession.launch(cfg, _app_launcher=app_launcher)
    env, provider = session.env, session.provider
    session.reset()

    from franka_v1_skill_lab.skill_runtime._legacy import ensure_legacy_on_path

    ensure_legacy_on_path()
    from runtime.base_skill import set_speed_scale
    from runtime.ik_joint_adapter import IKJointAdapter
    from runtime.skill_request import SkillRequest
    from runtime.skill_types import SkillType
    from skills.microwave_door_skill import DoorIKConfig, OpenDoorIKSkill

    set_speed_scale(1.0)  # diagnostic: skill runs at its own (speed-decoupled) rate anyway
    adapter = IKJointAdapter(env)
    robot = provider.scene["robot"]
    art = provider.scene["microwave"]   # fridge articulation (member kept "microwave")
    jids, _ = art.find_joints("joint_1")
    arm_ids = adapter._joint_ids
    dt = float(env_cfg_dt(env))

    def P(*a):
        print(*a, flush=True)

    clock = {"t": 0.0}

    def tick():
        clock["t"] += dt
        provider.set_sim_time(clock["t"])

    def reset_world():
        # arm -> default home; door -> closed; settle
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        q = art.data.joint_pos.clone(); q[:, jids[0]] = 0.0
        art.write_joint_state_to_sim(q, torch.zeros_like(q))
        for _ in range(25):
            st = provider.get_state()
            env.step(provider.make_hold_joint_action(st, 1.0))
            tick()

    def door_deg():
        return math.degrees(float(art.data.joint_pos[0, jids[0]]))

    def run_attempt(roll, pitch, seed=None, label="") -> dict:
        reset_world()
        if seed is not None:
            # perturb the arm into a different REDUNDANT initial posture (then let the approach IK
            # converge from there -> a different null-space branch for the grasp/pull).
            q = robot.data.joint_pos.clone()
            for ji, off in seed.items():
                q[0, arm_ids[ji]] = q[0, arm_ids[ji]] + math.radians(off)
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            for _ in range(15):
                st = provider.get_state()
                env.step(provider.make_hold_joint_action(st, 1.0))
                tick()
        cfg_door = DoorIKConfig()
        cfg_door.grasp_roll_deg = float(roll)
        cfg_door.approach_pitch_deg = float(pitch)
        cfg_door.max_regrips = 0       # isolate the SINGLE-grasp reach of this config
        req = SkillRequest(request_id=f"sweep_{roll}_{pitch}", skill_type=SkillType.OPEN_DOOR,
                           source_object=None, destination_type="door", destination_object="fridge")
        skill = OpenDoorIKSkill(req, env, adapter)
        skill.cfg = cfg_door
        skill.ee_log_every = 0         # quiet
        st = provider.get_state()
        skill.start(st)
        max_ang = 0.0
        peak_tilt = 0.0
        gripped = False
        for i in range(args.max_steps):
            st = provider.get_state()
            cmd = skill.step(st, dt)
            tick()
            # peak wrist tilt while gripping (orientation deviation from level)
            if skill.runtime.state in ("SWEEP", "VERIFY", "SETTLE"):
                gripped = True
                try:
                    tcp = st.robot.tcp_pose
                    import isaaclab.utils.math as mu
                    xaxis = mu.quat_apply(tcp.quat_w.unsqueeze(0),
                                          torch.tensor([[1.0, 0.0, 0.0]], device=tcp.quat_w.device))[0]
                    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(abs(xaxis[2]))))))
                    peak_tilt = max(peak_tilt, tilt)
                except Exception:
                    pass
            max_ang = max(max_ang, door_deg())
            if cmd.raw_joint_action is not None:
                action = provider.make_joint_action_from_raw(cmd.raw_joint_action)
            elif cmd.joint_target is not None:
                action = provider.make_joint_action_from_q_des(cmd.joint_target, cmd.gripper_command)
            else:
                action = provider.make_hold_joint_action(st, cmd.gripper_command)
            env.step(action)
            if skill.status.value in ("succeeded", "failed", "stopped"):
                # keep stepping a few to capture settle, then break
                pass
            if skill.runtime.state in ("SUCCEEDED", "FAILED"):
                break
        return {"roll": roll, "pitch": pitch, "max_deg": round(max_ang, 1),
                "final_deg": round(door_deg(), 1), "peak_tilt": round(peak_tilt, 1),
                "gripped": gripped, "state": skill.runtime.state}

    # Fixed best roll (180 per the orientation sweep); vary the REDUNDANT initial arm posture instead.
    seeds = [
        ("home",        None),
        ("j1-30",       {0: -30}),
        ("j1+30",       {0: +30}),
        ("j3-50",       {2: -50}),
        ("j3+50",       {2: +50}),
        ("j6-40",       {5: -40}),
        ("j6-70",       {5: -70}),
        ("j1-30,j3+40", {0: -30, 2: +40}),
        ("j1+30,j3-40", {0: +30, 2: -40}),
        ("j4-30",       {3: -30}),
        ("elbowA",      {0: -25, 2: +40, 5: -50}),
        ("elbowB",      {0: +25, 2: -40, 5: -30}),
    ]
    P("\n================ DOOR REDUNDANCY SWEEP (roll=180) ================")
    results = []
    for label, seed in seeds:
        r = run_attempt(180.0, 0.0, seed=seed, label=label)
        r["label"] = label
        results.append(r)
        P(f"[sweep] seed={label:14s} -> MAX={r['max_deg']:5.1f}deg final={r['final_deg']:5.1f}deg "
          f"peak_tilt={r['peak_tilt']:5.1f}deg gripped={r['gripped']} end={r['state']}")
    results.sort(key=lambda r: r["max_deg"], reverse=True)
    P("\n---- ranked by max open angle ----")
    for r in results[:12]:
        P(f"  seed={r.get('label',''):14s} MAX={r['max_deg']:5.1f}deg  peak_tilt={r['peak_tilt']:5.1f}deg")
    if results:
        best = results[0]
        P(f"\n[BEST] seed={best.get('label','')} -> {best['max_deg']}deg (peak_tilt {best['peak_tilt']}deg)")
    P("================ END ================\n")
    session.close()
    return 0


def env_cfg_dt(env):
    try:
        return env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation
    except Exception:
        return 0.02


if __name__ == "__main__":
    raise SystemExit(main())
