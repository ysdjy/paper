#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Closed-loop pi0.5 eval in IsaacLab: stack blue cube on red cube. STATUS: ready (Isaac/GPU + server).

Runs the fine-tuned pi0.5 policy IN THE LOOP in the V1 JointPolicy env. Each step:
  obs (front_rgb + wrist_rgb + state[joint_pos7 + gripper_width1] + prompt)
    --HTTP--> v1_joint_policy_server (.venv_openpi)  --> joint_targets[7] + gripper
  -> make_joint_action_from_q_des -> env.step -> record front frame -> check stack.

Reuses the collection env/cameras/provider/success-check. Talks to the policy
server with stdlib urllib only (the OpenPI model stays in .venv_openpi). Saves a
GIF of the front camera per rollout and a JSON summary with success rate.

Prereq: start the server first (separate terminal / background), e.g.
    pi05_isaacsim_baseline/.venv_openpi/bin/python \
        projects/franka_v1_skill_lab/pi05_training/v1_joint_policy_server.py \
        --ckpt <.../3000> --port 8010

Run:
    ./isaaclab.sh -p projects/franka_v1_skill_lab/pi05_training/eval_pi05_closed_loop_v1.py \
        --headless --enable_cameras --policy_port 8010 --num_rollouts 5 --max_steps 400 \
        --out_dir projects/franka_v1_skill_lab/data/pi05_eval
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

_PROJECTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECTS_DIR))

from franka_v1_skill_lab.scene import V1_BASE_TASK_ID  # noqa: E402

# success thresholds (same geometric check as the collector; cube_size 0.0406)
SUCCESS_XY = 0.030
SUCCESS_DZ_MIN = 0.030
SUCCESS_DZ_MAX = 0.052
SETTLE_VEL = 0.04
SUCCESS_STABLE_FRAMES = 5
PROMPT = "stack the blue cube on top of the red cube"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Closed-loop pi0.5 stack eval (V1).")
    ap.add_argument("--task", default=V1_BASE_TASK_ID)
    ap.add_argument("--scene_registry", default=None)
    ap.add_argument("--policy_host", default="127.0.0.1")
    ap.add_argument("--policy_port", type=int, default=8010)
    ap.add_argument("--policy_timeout", type=float, default=20.0)
    ap.add_argument("--num_rollouts", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--source_cube", default="cube_1")
    ap.add_argument("--target_cube", default="cube_2")
    ap.add_argument("--control_hz", type=float, default=50.0)
    ap.add_argument("--arm_stiffness", type=float, default=400.0)
    ap.add_argument("--arm_damping", type=float, default=80.0)
    ap.add_argument("--replan_every", type=int, default=1,
                    help="Re-query the policy every N steps; execute the action chunk in between (1=closed-loop).")
    ap.add_argument("--out_dir", default="projects/franka_v1_skill_lab/data/pi05_eval")
    ap.add_argument("--video_stride", type=int, default=2, help="Save every Nth front frame to the GIF.")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--layout_base_index", type=int, default=1000,
                    help="reset_index = layout_base_index + rollout. Use 1 with --seed 1 to replay TRAINING "
                         "cube layouts (diagnostic: separates generalization gap from interface bugs); "
                         "default 1000 = held-out layouts.")
    return ap


def main() -> int:
    parser = build_arg_parser()
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True  # need rendered cameras for the policy obs + video

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    rc = _run(args, simulation_app)
    simulation_app.close()
    return rc


def _run(args, simulation_app) -> int:
    import numpy as np
    import torch
    from PIL import Image

    from franka_v1_skill_lab.teleop_collection.recording.isaac_teleop_driver import build_teleop_env

    base_url = f"http://{args.policy_host}:{args.policy_port}"

    # --- wait for the policy server ----------------------------------------
    def _health() -> dict | None:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            return None

    print(f"[eval] waiting for policy server at {base_url} ...", flush=True)
    h = None
    for _ in range(60):
        h = _health()
        if h is not None:
            break
        time.sleep(2)
    if h is None:
        print(f"[eval] ERROR: policy server not reachable at {base_url}. Start v1_joint_policy_server.py first.",
              file=sys.stderr)
        return 2
    print(f"[eval] server up: {h}", flush=True)

    decimation = max(1, round(100.0 / max(1.0, args.control_hz)))
    torch.manual_seed(args.seed)
    env, env_cfg, cam_attached = build_teleop_env(
        task_id=args.task, num_envs=1, device=args.device,
        use_fabric=not getattr(args, "disable_fabric", False),
        enable_wrist_d435=True, enable_depth=False, free_microwave_door=True, seed=args.seed,
        decimation=decimation, arm_stiffness=args.arm_stiffness, arm_damping=args.arm_damping,
    )
    if not cam_attached:
        print("[eval] ERROR: cameras not attached (need --enable_cameras + GPU).", file=sys.stderr)
        env.close()
        return 3

    from runtime.scene_state_provider import SceneStateProvider
    from runtime.simple_scene_layout import SimpleSceneLayoutManager
    from franka_v1_skill_lab.sensors.d435.d435_observation_adapter import WristCameraAdapter

    provider = SceneStateProvider(env)
    layout = SimpleSceneLayoutManager(env=env, base_seed=args.seed)
    front_cam = WristCameraAdapter(env, camera_name="vla_front_static")
    wrist_cam = WristCameraAdapter(env, camera_name="vla_libero_eye_in_hand")

    sim_dt = env_cfg.sim.dt * env_cfg.decimation
    clock = {"t": 0.0}
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    def _b64_png(rgb: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(np.asarray(rgb)[..., :3].astype(np.uint8)).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def infer(front_rgb, wrist_rgb, state8) -> dict | None:
        payload = json.dumps({
            "state": [float(v) for v in state8],
            "image": _b64_png(front_rgb),
            "wrist_image": _b64_png(wrist_rgb),
            "prompt": PROMPT,
        }).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/infer", data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=args.policy_timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:
            print(f"[eval] infer error: {exc}", flush=True)
            return None

    def get_obs():
        st = provider.get_state()
        fr = front_cam.capture(require_depth=False).get("rgb")
        wr = wrist_cam.capture(require_depth=False).get("rgb")
        jp = provider.arm_joint_pos(st).detach().cpu().tolist()
        state8 = jp[:7] + [float(st.robot.gripper_width)]
        return st, fr, wr, state8

    def settle(n, gripper=1.0):
        for _ in range(n):
            s = provider.get_state()
            env.step(provider.make_hold_joint_action(s, gripper))
            clock["t"] += sim_dt
            provider.set_sim_time(clock["t"])

    def stacked_ok(st) -> bool:
        c1 = st.objects.get(args.source_cube); c2 = st.objects.get(args.target_cube)
        if c1 is None or c2 is None:
            return False
        p1, p2 = c1.pose.pos_w, c2.pose.pos_w
        dz = float((p1[2] - p2[2]).item())
        dxy = float(torch.norm(p1[:2] - p2[:2]).item())
        v1 = 0.0 if c1.lin_vel_w is None else float(torch.norm(c1.lin_vel_w).item())
        v2 = 0.0 if c2.lin_vel_w is None else float(torch.norm(c2.lin_vel_w).item())
        return dxy < SUCCESS_XY and SUCCESS_DZ_MIN < dz < SUCCESS_DZ_MAX and v1 < SETTLE_VEL and v2 < SETTLE_VEL

    results = []
    for ro in range(args.num_rollouts):
        env.reset(seed=args.seed)
        try:
            provider.reset_cabinet_joint("joint_0", 0.0)
        except Exception:
            pass
        layout.reset_layout(reset_index=args.layout_base_index + ro)  # 1000+=held-out; (seed 1, base 1)=training layouts
        settle(8, gripper=1.0)
        provider.set_sim_time(clock["t"])

        frames = []
        chunk = None; chunk_i = 0; ok_run = 0; success = False
        for step in range(args.max_steps):
            st, fr, wr, state8 = get_obs()
            if fr is not None and step % args.video_stride == 0:
                frames.append(np.asarray(fr)[..., :3].astype(np.uint8))
            # re-query policy every replan_every steps; reuse chunk in between
            if chunk is None or chunk_i >= len(chunk) or step % args.replan_every == 0:
                resp = infer(fr, wr, state8)
                if resp is None:
                    env.step(provider.make_hold_joint_action(st, 1.0)); clock["t"] += sim_dt; continue
                chunk = resp.get("chunk") or [resp["joint_targets"] + [resp["gripper"]]]
                chunk_i = 0
            act = chunk[chunk_i]; chunk_i += 1
            q_des = act[:7]
            g = float(act[7]) if len(act) > 7 else 1.0
            gripper_cmd = 1.0 if g < 0.5 else -1.0   # model 0=open/1=close -> env +1=open/-1=close
            env.step(provider.make_joint_action_from_q_des(q_des, gripper_cmd))
            clock["t"] += sim_dt; provider.set_sim_time(clock["t"])

            ok_run = ok_run + 1 if stacked_ok(provider.get_state()) else 0
            if ok_run >= SUCCESS_STABLE_FRAMES:
                success = True
                break

        # settle + final frame
        settle(20, gripper=1.0)
        ffr = front_cam.capture(require_depth=False).get("rgb")
        if ffr is not None:
            frames.append(np.asarray(ffr)[..., :3].astype(np.uint8))
        success = success or stacked_ok(provider.get_state())

        gif = out_dir / f"rollout_{ro:02d}_{'success' if success else 'fail'}.gif"
        if frames:
            imgs = [Image.fromarray(f) for f in frames]
            imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=80, loop=0)
        results.append({"rollout": ro, "success": success, "steps": step + 1, "gif": str(gif)})
        print(f"[eval] rollout {ro}: success={success} steps={step+1} -> {gif.name}", flush=True)

    env.close()
    n_succ = sum(r["success"] for r in results)
    summary = {
        "task": args.task, "checkpoint": "step-3000 pi05_franka_v1_stack",
        "num_rollouts": args.num_rollouts, "successes": n_succ,
        "success_rate": n_succ / max(1, args.num_rollouts), "rollouts": results,
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\n[eval] DONE. success {n_succ}/{args.num_rollouts} "
          f"({100*summary['success_rate']:.0f}%). summary -> {out_dir/'eval_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
