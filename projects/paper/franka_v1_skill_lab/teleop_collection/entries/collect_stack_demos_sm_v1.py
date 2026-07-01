#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Autonomous STATE-MACHINE data collection: stack blue cube_1 on red cube_2. STATUS: ready (Isaac/GPU).

Unlike ``collect_teleop_demos_joint_v1.py`` (human/GELLO/mock driven), this entry
drives the Franka with the V1 **skill state machine** (grasp -> place), records
every frame in the SAME LeRobot-compatible HDF5 format, auto-detects stack
success, resets with RANDOMIZED cube positions, and loops until N successful
demos are collected (default 20). The resulting dataset is meant for pi0.5
fine-tuning.

Task: grasp blue cube (cube_1) and place it on top of red cube (cube_2).
  - blue  = cube_1  diffuse (0.1, 0.25, 0.9)
  - red   = cube_2  diffuse (0.9, 0.1, 0.08)

Control: joint action (q_des from internal DLS IK), task
``Isaac-Stack-Cube-Franka-JointPolicy-v0``.

Recording (LeRobot strict, via the shared recorder/writer):
  obs/joint_pos(7) joint_vel(7) gripper_width(1) ee_pose(7)
  obs/images/front_rgb(256x256x3 -> LeRobot `image`)  wrist_rgb(128x128x3 -> `wrist_image`)
  obs/objects/<name>(7)   actions/joint_target(7) gripper_command(1)

Per episode the cubes are repositioned by SimpleSceneLayoutManager (random,
min-separation), so each of the N demos is a different layout.

Run (headless, 20 successes, with cameras):
    ./isaaclab.sh -p projects/franka_v1_skill_lab/teleop_collection/entries/collect_stack_demos_sm_v1.py \
        --headless --enable_cameras --num_success 20 \
        --scene_registry projects/franka_v1_skill_lab/scene/saved_scenes/v1_active/scene_v1_registry.json \
        --out_dir projects/franka_v1_skill_lab/data/stack_demos_hdf5 --seed 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_PROJECTS_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECTS_DIR))

from franka_v1_skill_lab.scene import V1_BASE_TASK_ID  # noqa: E402

# --- task geometry (cube_size = 0.0406; see stack_joint_pos_env_cfg) ---------
CUBE_SIZE = 0.0406
CUBE_HALF = CUBE_SIZE / 2.0           # 0.0203 = OBJECT_SUPPORT_OFFSET_Z for cubes
# Place surface = top of cube_2 (its center z + half). The place skill then puts
# cube_1's center at surface_z + support_offset(0.0203) + clearance(0.002).
STACK_SURFACE_OFFSET_Z = CUBE_HALF    # place onto cube_2's top face

# Success thresholds for "cube_1 stacked on cube_2" (geometric, tuned to 0.0406 cubes).
SUCCESS_XY = 0.030                    # cube_1/cube_2 xy alignment (m)
SUCCESS_DZ_MIN = 0.030               # cube_1 center above cube_2 by ~one cube (0.0406)
SUCCESS_DZ_MAX = 0.052
SETTLE_VEL = 0.03                     # m/s: both cubes considered settled below this
SUCCESS_STABLE_FRAMES = 5            # consecutive settled+stacked frames required


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="State-machine stack-cube collection (V1).")
    ap.add_argument("--task", default=V1_BASE_TASK_ID)
    ap.add_argument("--scene_registry", default=None)
    ap.add_argument("--num_success", type=int, default=20, help="Target number of SUCCESSFUL demos.")
    ap.add_argument("--max_attempts", type=int, default=80, help="Hard cap on episode attempts.")
    ap.add_argument("--source_cube", default="cube_1", help="Cube to grasp (blue).")
    ap.add_argument("--target_cube", default="cube_2", help="Cube to stack onto (red).")
    ap.add_argument("--instruction", default="stack the blue cube on top of the red cube")
    # cameras / recording
    ap.add_argument("--enable_wrist_d435", action="store_true", default=True,
                    help="Attach + record front/wrist cameras (LeRobot image/wrist_image). On by default.")
    ap.add_argument("--no_camera", action="store_true", help="Disable cameras (debug only; breaks LeRobot images).")
    ap.add_argument("--record_depth", action="store_true", help="Also record FoundationPose wrist depth.")
    ap.add_argument("--record_objects", action="store_true", default=True, help="Record sim-GT object poses.")
    ap.add_argument("--out_dir", default="projects/franka_v1_skill_lab/data/stack_demos_hdf5")
    # control precision (same knobs as the teleop collector)
    ap.add_argument("--control_hz", type=float, default=50.0)
    ap.add_argument("--arm_stiffness", type=float, default=400.0)
    ap.add_argument("--arm_damping", type=float, default=80.0)
    ap.add_argument("--speed", type=float, default=2.0, help="Skill motion speed scale (global step scale).")
    # phase budgets (sim steps)
    ap.add_argument("--grasp_max_steps", type=int, default=900)
    ap.add_argument("--place_max_steps", type=int, default=900)
    ap.add_argument("--settle_steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dry_run", action="store_true", help="Print plan + thresholds, do not launch Isaac.")
    return ap


def main() -> int:
    parser = build_arg_parser()
    pre_args, _ = parser.parse_known_args()
    if pre_args.dry_run:
        plan = {
            "task_id": pre_args.task,
            "control_mode": "joint",
            "skill_sequence": ["grasp(source_cube)", "place(source_cube -> top of target_cube)"],
            "source_cube(blue)": pre_args.source_cube,
            "target_cube(red)": pre_args.target_cube,
            "num_success": pre_args.num_success,
            "instruction": pre_args.instruction,
            "place_surface_z": "cube_2_local.z + %.4f (cube top face)" % STACK_SURFACE_OFFSET_Z,
            "success_check": {
                "xy<": SUCCESS_XY, "dz_range": [SUCCESS_DZ_MIN, SUCCESS_DZ_MAX],
                "settle_vel<": SETTLE_VEL, "stable_frames": SUCCESS_STABLE_FRAMES,
            },
            "lerobot_images": {"front_rgb->image": "256x256", "wrist_rgb->wrist_image": "128x128"},
        }
        print("[stack_collect] plan:")
        print(json.dumps(plan, indent=2))
        print("[stack_collect] --dry_run: not launching Isaac.")
        return 0

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    want_camera = args.enable_wrist_d435 and not args.no_camera
    if want_camera:
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    rc = _run(args, simulation_app, want_camera)
    simulation_app.close()
    return rc


def _run(args, simulation_app, want_camera: bool) -> int:
    import torch

    from franka_v1_skill_lab.scene import resolve_active_scene
    from franka_v1_skill_lab.teleop_collection.recording.episode_recorder import EpisodeRecorder, FrameBuilder
    from franka_v1_skill_lab.teleop_collection.recording.hdf5_writer import TeleopHDF5Writer
    from franka_v1_skill_lab.teleop_collection.recording.isaac_teleop_driver import (
        build_teleop_env,
        capture_objects,
        ee_pose_xyzw,
    )

    scene_reg = args.scene_registry or ""
    try:
        info = resolve_active_scene(args.scene_registry)
        print(f"[stack_collect] V1 scene: task={info['task_id']} usd_exists={info['usd_exists']}", flush=True)
    except Exception as exc:  # pragma: no cover
        print(f"[stack_collect] scene registry resolve skipped ({exc}).", flush=True)

    decimation = max(1, round(100.0 / max(1.0, args.control_hz)))
    control_hz_eff = 100.0 / decimation
    print(f"[stack_collect] control_hz~{control_hz_eff:.0f} (decimation={decimation}) "
          f"arm_PD={args.arm_stiffness}/{args.arm_damping} speed={args.speed}", flush=True)

    torch.manual_seed(args.seed)
    env, env_cfg, cam_attached = build_teleop_env(
        task_id=args.task, num_envs=1, device=args.device,
        use_fabric=not getattr(args, "disable_fabric", False),
        enable_wrist_d435=want_camera, enable_depth=args.record_depth,
        free_microwave_door=True, seed=args.seed,
        decimation=decimation, arm_stiffness=args.arm_stiffness, arm_damping=args.arm_damping,
    )

    # legacy skill-machine modules (build_teleop_env put the legacy root on sys.path)
    from runtime.base_skill import set_speed_scale
    from runtime.ik_joint_adapter import IKJointAdapter
    from runtime.scene_state_provider import SceneStateProvider
    from runtime.simple_scene_layout import SimpleSceneLayoutManager
    from runtime.skill_request import SkillRequest
    from runtime.skill_types import ExecutionStatus, SkillType
    from runtime.target_registry import TargetRegistry
    from state_machine.skill_executor import JointBackendConfig, SkillExecutor

    set_speed_scale(args.speed)
    provider = SceneStateProvider(env)
    layout_manager = SimpleSceneLayoutManager(env=env, base_seed=args.seed)
    registry = TargetRegistry(env.unwrapped.device, cube_grasp_z_offset=0.0)
    adapter = IKJointAdapter(env)
    backend = JointBackendConfig(
        mode="joint", grasp_backend="joint_ik", place_backend="joint_ik", drawer_backend="none",
        adapter=adapter, arm_joint_ids=provider._arm_joint_ids,
    )

    # camera adapters (front = LeRobot image, vla = wrist_image, fp = depth)
    front_adapter = vla_adapter = fp_adapter = None
    if cam_attached:
        try:
            from franka_v1_skill_lab.sensors.d435.d435_observation_adapter import WristCameraAdapter

            front_adapter = WristCameraAdapter(env, camera_name="vla_front_static")
            vla_adapter = WristCameraAdapter(env, camera_name="vla_libero_eye_in_hand")
            if args.record_depth:
                fp_adapter = WristCameraAdapter(env, camera_name="foundationpose_d435_rgbd")
        except Exception as exc:  # pragma: no cover
            print(f"[stack_collect] camera adapters unavailable ({exc}); recording without images.", flush=True)
            cam_attached = False

    recorder = EpisodeRecorder(
        record_images=cam_attached, record_depth=args.record_depth and cam_attached, record_objects=args.record_objects,
    )
    writer = TeleopHDF5Writer(args.out_dir, task_id=args.task, scene_registry=scene_reg)

    sim_dt = env_cfg.sim.dt * env_cfg.decimation
    state_holder = {"sim_time": 0.0}

    # ---- helpers -----------------------------------------------------------
    def settle(steps: int, gripper: float | None = 1.0):
        for _ in range(steps):
            st = provider.get_state()
            env.step(provider.make_hold_joint_action(st, gripper))
            state_holder["sim_time"] += sim_dt
            provider.set_sim_time(state_holder["sim_time"])

    def reset_episode(reset_index: int):
        env.reset(seed=args.seed)
        try:
            provider.reset_cabinet_joint("joint_0", 0.0)
        except Exception:
            pass
        layout_manager.reset_layout(reset_index=reset_index)
        settle(8, gripper=1.0)

    def command_to_action(command, st):
        if command.control_mode == "joint":
            if command.raw_joint_action is not None:
                return provider.make_joint_action_from_raw(command.raw_joint_action)
            if command.joint_target is not None:
                return provider.make_joint_action_from_q_des(command.joint_target, command.gripper_command)
            return provider.make_hold_joint_action(st, None)
        return provider.make_action(command.tcp_pose_w, command.gripper_command)

    def capture_frame(command):
        """Record one frame AFTER env.step (obs reflects applied action)."""
        post = provider.get_state()
        images: dict = {}
        if cam_attached:
            if front_adapter is not None:
                try:
                    fr = front_adapter.capture(require_depth=False)
                    if fr.get("rgb") is not None:
                        images["front_rgb"] = fr["rgb"]
                except Exception:
                    pass
            if vla_adapter is not None:
                try:
                    fr = vla_adapter.capture()
                    if fr.get("rgb") is not None:
                        images["wrist_rgb"] = fr["rgb"]
                except Exception:
                    pass
            if args.record_depth and fp_adapter is not None:
                try:
                    fr = fp_adapter.capture()
                    if fr.get("depth") is not None:
                        images["wrist_depth"] = fr["depth"]
                except Exception:
                    pass
        q_des = command.joint_target
        if q_des is not None:
            q_des = q_des.detach().cpu().tolist()
        else:
            q_des = provider.arm_joint_pos(post).detach().cpu().tolist()
        gripper_norm = 0.0 if command.gripper_command > 0 else 1.0  # 0=open .. 1=closed
        objects = capture_objects(post) if recorder.record_objects else []
        frame = FrameBuilder.build(
            step_id=recorder.num_steps, timestamp=time.time(),
            joint_pos=provider.arm_joint_pos(post),
            joint_vel=post.robot.joint_vel[provider._arm_joint_ids],
            ee_pose=ee_pose_xyzw(post), gripper_width=post.robot.gripper_width,
            raw_q=q_des, filtered_q=q_des, joint_target=q_des,
            gripper=gripper_norm, images=images, objects=objects,
        )
        recorder.append(frame)

    def run_skill(executor, request, max_steps: int) -> bool:
        """Drive + RECORD one skill to terminal; return True iff SUCCEEDED."""
        st = provider.get_state()
        res = executor.start(request, st)
        if res is not None and not res.success:
            print(f"[stack_collect]   {request.skill_type.value} rejected: {res.failure_reason}", flush=True)
            return False
        for _ in range(max_steps):
            provider.set_sim_time(state_holder["sim_time"])
            st = provider.get_state()
            command = executor.step(st, sim_dt)
            env.step(command_to_action(command, st))
            capture_frame(command)
            state_holder["sim_time"] += sim_dt
            if executor.status == ExecutionStatus.SUCCEEDED:
                return True
            if executor.status in (ExecutionStatus.FAILED, ExecutionStatus.NOT_IMPLEMENTED, ExecutionStatus.STOPPED):
                print(f"[stack_collect]   {request.skill_type.value} ended status={executor.status.value}", flush=True)
                return False
        print(f"[stack_collect]   {request.skill_type.value} TIMEOUT ({max_steps} steps)", flush=True)
        return False

    def target_surface_xyz(st) -> list[float]:
        tgt = st.objects[args.target_cube].pose.pos_w
        local = (tgt - st.env_origin_w).detach().cpu().tolist()
        return [float(local[0]), float(local[1]), float(local[2]) + STACK_SURFACE_OFFSET_Z]

    def stacked_ok(st) -> bool:
        c1 = st.objects.get(args.source_cube)
        c2 = st.objects.get(args.target_cube)
        if c1 is None or c2 is None:
            return False
        p1 = c1.pose.pos_w
        p2 = c2.pose.pos_w
        dz = float((p1[2] - p2[2]).item())
        dxy = float(torch.norm((p1[:2] - p2[:2])).item())
        v1 = 0.0 if c1.lin_vel_w is None else float(torch.norm(c1.lin_vel_w).item())
        v2 = 0.0 if c2.lin_vel_w is None else float(torch.norm(c2.lin_vel_w).item())
        return (
            dxy < SUCCESS_XY and SUCCESS_DZ_MIN < dz < SUCCESS_DZ_MAX
            and v1 < SETTLE_VEL and v2 < SETTLE_VEL
        )

    # ---- collection loop ---------------------------------------------------
    successes = 0
    attempt = 0
    print(f"[stack_collect] target {args.num_success} successful '{args.source_cube}->on->{args.target_cube}' demos.",
          flush=True)

    while successes < args.num_success and attempt < args.max_attempts and simulation_app.is_running():
        attempt += 1
        reset_episode(reset_index=attempt)  # different cube layout each attempt
        eid = f"stack_{successes + 1:04d}_att{attempt:03d}_{int(time.time())}"
        recorder.reset(
            episode_id=eid, task_instruction=args.instruction,
            skill_type="stack", target_name=f"{args.source_cube}->{args.target_cube}",
            scene_registry=scene_reg, seed=args.seed,
        )
        executor = SkillExecutor(registry, backend=backend)
        print(f"[stack_collect] attempt {attempt} (have {successes}/{args.num_success}) eid={eid}", flush=True)

        # phase 1: grasp blue cube
        grasp_req = SkillRequest(
            request_id=f"grasp_{eid}", skill_type=SkillType.GRASP, source_object=args.source_cube,
        )
        if not run_skill(executor, grasp_req, args.grasp_max_steps):
            print("[stack_collect]   grasp failed -> discard", flush=True)
            continue
        if executor.held_object is None or executor.held_object.object_name != args.source_cube:
            print("[stack_collect]   grasp ok but no held-object context -> discard", flush=True)
            continue

        # phase 2: place on top of red cube (surface from LIVE cube_2 pose)
        st = provider.get_state()
        place_req = SkillRequest(
            request_id=f"place_{eid}", skill_type=SkillType.PLACE, source_object=args.source_cube,
            destination_type="point", destination_object=args.target_cube,
            parameters={"target_frame": "env_local", "target_surface_xyz": target_surface_xyz(st)},
        )
        placed = run_skill(executor, place_req, args.place_max_steps)

        # phase 3: settle + record + success check
        for _ in range(args.settle_steps):
            provider.set_sim_time(state_holder["sim_time"])
            st = provider.get_state()
            env.step(provider.make_hold_joint_action(st, 1.0))  # gripper open, hold posture
            # synthesize a "hold" command frame so settle is in the trajectory
            class _Hold:
                control_mode = "joint"; raw_joint_action = None
                joint_target = provider.arm_joint_pos(st); gripper_command = 1.0
            capture_frame(_Hold())
            state_holder["sim_time"] += sim_dt

        # require a few consecutive stable+stacked frames
        ok_frames = 0
        for _ in range(SUCCESS_STABLE_FRAMES + 5):
            provider.set_sim_time(state_holder["sim_time"])
            st = provider.get_state()
            env.step(provider.make_hold_joint_action(st, 1.0))
            ok_frames = ok_frames + 1 if stacked_ok(st) else 0
            state_holder["sim_time"] += sim_dt
            if ok_frames >= SUCCESS_STABLE_FRAMES:
                break
        success = ok_frames >= SUCCESS_STABLE_FRAMES

        meta = recorder.episode_meta(success)
        if success:
            name = writer.write_episode(recorder.frames, meta)
            successes += 1
            print(f"[stack_collect]   SUCCESS #{successes}: {name} steps={meta['num_steps']} placed={placed}", flush=True)
        else:
            print(f"[stack_collect]   discard (placed={placed}, stack check failed, steps={meta['num_steps']})", flush=True)

    path = writer.close()
    env.close()

    manifest = {
        "hdf5": path, "num_demos": writer.num_demos, "task_id": args.task,
        "scene_registry": scene_reg, "controller": "skill_state_machine",
        "skill_sequence": ["grasp", "place"], "source_cube": args.source_cube, "target_cube": args.target_cube,
        "num_success": successes, "attempts": attempt, "had_camera": cam_attached,
        "recorded_objects": recorder.record_objects, "instruction": args.instruction,
    }
    man_path = Path(args.out_dir) / "last_session.json"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[stack_collect] DONE. successes={successes}/{args.num_success} attempts={attempt} hdf5={path}", flush=True)
    print(f"[stack_collect] manifest -> {man_path}", flush=True)
    return 0 if successes >= args.num_success else 4


if __name__ == "__main__":
    raise SystemExit(main())
