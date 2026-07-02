#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Collect joint-action teleop demos for the V1 scene -> pi0.5 dataset. STATUS: ready.

Records (observation, joint-action) pairs while a human teleoperates the Franka
in ``Isaac-Stack-Cube-Franka-JointPolicy-v0`` via GELLO (or a hardware-free
mock), and writes HDF5 demos consumed by ``pi05_training`` (-> LeRobot).

Pipeline per sim step:
    GELLO read  ->  JointTeleopSafety (EMA + clamp + limits + e-stop)
                ->  q_des[7] + gripper  ->  make_joint_action_from_q_des
                ->  env.step  ->  capture obs  ->  (if RECORDING) buffer frame

Episode lifecycle is an event-driven state machine
(recording/keyboard_episode_controller). Events come from the keyboard in the
GUI, or from a deterministic ``ScriptedEventSource`` (headless / mock), so the
mock smoke test runs end-to-end with NO hardware and NO human.

Smoke (no Isaac):
    python .../collect_teleop_demos_joint_v1.py --dry_run

Mock collection (Isaac, auto-records one episode then exits):
    ./isaaclab.sh -p .../collect_teleop_demos_joint_v1.py \
        --num_envs 1 --task Isaac-Stack-Cube-Franka-JointPolicy-v0 \
        --scene_registry .../scene_v1_registry.json \
        --teleop_device mock --enable_wrist_d435 --record_objects \
        --out_dir projects/franka_v1_skill_lab/data/teleop_raw_hdf5 \
        --max_steps_per_episode 300 --seed 1
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

# Franka "ready" home posture (7 arm joints) used by the H / home request.
HOME_Q = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741]

RECORD_SCHEMA = {
    "obs": {
        "joint_pos": "(T,7) f32", "joint_vel": "(T,7) f32",
        "gripper_width": "(T,1) f32", "ee_pose": "(T,7) f32 [x,y,z,qx,qy,qz,qw]",
        "objects/<name>": "(T,7) [x,y,z,qw,qx,qy,qz] (--record_objects)",
        "images/front_rgb": "(T,256,256,3) u8 -> LeRobot `image` (static front cam, --enable_wrist_d435)",
        "images/wrist_rgb": "(T,H,W,3) u8 -> LeRobot `wrist_image` (eye-in-hand, --enable_wrist_d435)",
        "images/wrist_depth": "(T,H,W) f32 (--record_depth, FoundationPose cam)",
    },
    "actions": {"joint_target": "(T,7) f32 << the action", "gripper_command": "(T,1) f32 0=open..1=closed"},
    "teleop": {"raw_q": "(T,7) f32", "filtered_q": "(T,7) f32"},
    "demo_attrs": {
        "task_id": V1_BASE_TASK_ID, "control_mode": "joint", "scene_schema": "franka-scene-v1",
        "episode_id": "str", "task_instruction": "str", "skill_type": "str",
        "target_name": "str", "success": "bool", "num_steps": "int", "seed": "int",
    },
}


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Collect joint teleop demos (V1).")
    ap.add_argument("--num_envs", type=int, default=1, help="Number of envs (collection supports 1).")
    ap.add_argument("--task", default=V1_BASE_TASK_ID, help="Gym task id (V1 base joint-policy task).")
    ap.add_argument("--scene_registry", default=None, help="V1 scene registry JSON (provenance).")
    ap.add_argument("--teleop_device", default="mock", choices=["gello", "mock", "keyboard"])
    ap.add_argument("--mock_gello", action="store_true", help="Alias for --teleop_device mock.")
    ap.add_argument("--gello_config", default=None, help="GELLO yaml (real device).")
    ap.add_argument("--enable_wrist_d435", action="store_true", help="Attach + record the sim wrist D435 RGB.")
    ap.add_argument("--record_depth", action="store_true", help="Also record wrist depth (needs D435).")
    ap.add_argument("--no_camera", action="store_true", help="Force no camera even if --enable_wrist_d435.")
    ap.add_argument("--record_objects", action="store_true", help="Record sim-GT object poses.")
    ap.add_argument("--record_foundationpose_objects", action="store_true",
                    help="(stub) record FoundationPose object poses instead of sim GT.")
    # task labelling
    ap.add_argument("--task_config", default=None, help="teleop_tasks.yaml (instruction / skill / target).")
    ap.add_argument("--task_catalog_id", default=None, help="Which task entry from --task_config to stamp.")
    ap.add_argument("--instruction", default=None, help="Override task_instruction string.")
    ap.add_argument("--skill_type", default=None, help="Override skill_type label.")
    ap.add_argument("--target_name", default=None, help="Override target_name label.")
    # episode / output
    ap.add_argument("--out_dir", default="projects/franka_v1_skill_lab/data/teleop_raw_hdf5")
    ap.add_argument("--episode_id_prefix", default="ep")
    ap.add_argument("--num_episodes", type=int, default=1, help="Episodes to auto-collect (scripted source).")
    ap.add_argument("--max_steps_per_episode", type=int, default=300)
    ap.add_argument("--success_only", action="store_true", help="Only export success episodes (failed -> failed file).")
    ap.add_argument("--save_failed", action="store_true", help="Also save failed episodes to a separate file.")
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=1)
    # --- control precision / responsiveness ----------------------------------
    ap.add_argument("--control_hz", type=float, default=50.0,
                    help="Control rate. Sets env decimation=round(100/control_hz): 50->2 (50Hz), 100->1 (100Hz). "
                         "Base task is 20Hz (sluggish for teleop).")
    ap.add_argument("--arm_stiffness", type=float, default=400.0,
                    help="Franka arm joint PD stiffness (base task 80 = soft tracking; 400 = HIGH_PD, tight).")
    ap.add_argument("--arm_damping", type=float, default=80.0, help="Franka arm joint PD damping.")
    ap.add_argument("--gello_hz", type=float, default=100.0, help="Real GELLO serial read rate (Hz).")
    # safety / smoothing
    ap.add_argument("--lowpass_alpha", type=float, default=0.5,
                    help="EMA on raw GELLO reads (0..1, higher=snappier/less lag). 1.0 = no smoothing.")
    ap.add_argument("--max_joint_vel", type=float, default=4.0,
                    help="Max arm joint speed (rad/s). Per-step clamp = max_joint_vel/control_hz.")
    ap.add_argument("--max_step_rad", type=float, default=None,
                    help="Override the per-step joint clamp directly (else derived from --max_joint_vel).")
    # control flow
    ap.add_argument("--auto", action="store_true", help="Force scripted (auto) collection.")
    ap.add_argument("--keyboard", action="store_true", help="Force keyboard control (GUI).")
    ap.add_argument("--dry_run", action="store_true", help="Validate + print schema, do not launch Isaac.")
    return ap


def _load_task_label(args) -> dict:
    """Resolve task_instruction / skill_type / target_name from yaml + overrides."""
    label = {"task_instruction": "Teleoperate the Franka.", "skill_type": "teleop", "target_name": ""}
    if args.task_config and Path(args.task_config).is_file():
        import yaml

        catalog = yaml.safe_load(Path(args.task_config).read_text(encoding="utf-8")) or {}
        tasks = {t["task_id"]: t for t in catalog.get("tasks", [])}
        tid = args.task_catalog_id or catalog.get("default_task_id")
        entry = tasks.get(tid)
        if entry:
            label.update(
                task_instruction=entry.get("instruction", label["task_instruction"]),
                skill_type=entry.get("skill_type", label["skill_type"]),
                target_name=entry.get("target_name", label["target_name"]),
            )
        elif tid is not None:
            print(f"[collect] WARNING: task_catalog_id {tid!r} not in {args.task_config}; using defaults.", flush=True)
    if args.instruction:
        label["task_instruction"] = args.instruction
    if args.skill_type:
        label["skill_type"] = args.skill_type
    if args.target_name:
        label["target_name"] = args.target_name
    return label


def main() -> int:
    parser = build_arg_parser()

    # --dry_run is fully offline (no Isaac) -> handle before AppLauncher.
    pre_args, _ = parser.parse_known_args()
    if pre_args.dry_run:
        print("[collect] planned HDF5 record schema:")
        print(json.dumps(RECORD_SCHEMA, indent=2))
        label = _load_task_label(pre_args)
        print("[collect] resolved task label:", json.dumps(label))
        print("[collect] --dry_run: not launching Isaac. Contract above is the recording target.")
        return 0

    # --- launch Isaac --------------------------------------------------------
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.mock_gello:
        args.teleop_device = "mock"
    want_camera = args.enable_wrist_d435 and not args.no_camera
    if want_camera:
        args.enable_cameras = True  # the renderer needs this for the D435

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    rc = _run(args, simulation_app, want_camera)
    simulation_app.close()
    return rc


def _run(args, simulation_app, want_camera: bool) -> int:
    import torch

    from franka_v1_skill_lab.scene import resolve_active_scene
    from franka_v1_skill_lab.teleop_collection.gello.gello_reader import make_gello_source
    from franka_v1_skill_lab.teleop_collection.gello.gello_to_franka_mapper import GelloToFrankaMapper
    from franka_v1_skill_lab.teleop_collection.gello.joint_teleop_safety import JointTeleopSafety
    from franka_v1_skill_lab.teleop_collection.recording.episode_recorder import EpisodeRecorder, FrameBuilder
    from franka_v1_skill_lab.teleop_collection.recording.hdf5_writer import TeleopHDF5Writer
    from franka_v1_skill_lab.teleop_collection.recording.isaac_teleop_driver import (
        CarbKeyboardSource,
        build_teleop_env,
        capture_objects,
        ee_pose_xyzw,
    )
    from franka_v1_skill_lab.teleop_collection.recording.keyboard_episode_controller import (
        CollectionStateMachine,
        ScriptedEventSource,
    )

    # scene provenance (never fatal)
    scene_reg = args.scene_registry or ""
    try:
        info = resolve_active_scene(args.scene_registry)
        print(f"[collect] V1 scene: task={info['task_id']} usd_exists={info['usd_exists']}", flush=True)
    except Exception as exc:  # pragma: no cover
        print(f"[collect] scene registry resolve skipped ({exc}).", flush=True)

    # control-rate + clamp: base sim is 100 Hz physics; decimation sets control rate.
    decimation = max(1, round(100.0 / max(1.0, args.control_hz)))
    control_hz_eff = 100.0 / decimation
    max_step_rad = args.max_step_rad if args.max_step_rad is not None else args.max_joint_vel / control_hz_eff
    print(f"[collect] control_hz~{control_hz_eff:.0f} (decimation={decimation}) "
          f"max_step_rad={max_step_rad:.4f} (~{max_step_rad * control_hz_eff:.1f} rad/s) "
          f"lowpass_alpha={args.lowpass_alpha} arm_PD={args.arm_stiffness}/{args.arm_damping}", flush=True)

    torch.manual_seed(args.seed)
    env, env_cfg, cam_attached = build_teleop_env(
        task_id=args.task, num_envs=args.num_envs, device=args.device,
        use_fabric=not getattr(args, "disable_fabric", False),
        enable_wrist_d435=want_camera, enable_depth=args.record_depth,
        free_microwave_door=True, seed=args.seed,
        decimation=decimation, arm_stiffness=args.arm_stiffness, arm_damping=args.arm_damping,
    )

    # legacy real impl: importable now that build_teleop_env put it on sys.path
    from runtime.scene_state_provider import SceneStateProvider

    provider = SceneStateProvider(env)

    # camera capture (optional): front RGB (LeRobot `image`) from the static
    # front camera, wrist RGB (`wrist_image`) from the VLA eye-in-hand camera,
    # and depth (if --record_depth) from the FoundationPose camera.
    front_adapter = vla_adapter = fp_adapter = None
    if cam_attached:
        try:
            from franka_v1_skill_lab.sensors.d435.d435_observation_adapter import WristCameraAdapter

            front_adapter = WristCameraAdapter(env, camera_name="vla_front_static")
            vla_adapter = WristCameraAdapter(env, camera_name="vla_libero_eye_in_hand")
            fp_adapter = WristCameraAdapter(env, camera_name="foundationpose_d435_rgbd")
        except Exception as exc:  # pragma: no cover
            print(f"[collect] camera adapters unavailable ({exc}); recording without images.", flush=True)
            cam_attached = False

    # teleop source + mapper
    source = make_gello_source(args.teleop_device, config_path=args.gello_config, hz=args.gello_hz)
    safety = JointTeleopSafety(lowpass_alpha=args.lowpass_alpha, max_step_rad=max_step_rad)
    mapper = GelloToFrankaMapper(safety=safety)

    label = _load_task_label(args)
    if args.record_foundationpose_objects:
        print("[collect] NOTE: --record_foundationpose_objects is a stub; recording sim-GT object poses instead.",
              flush=True)

    # recorder + writers
    recorder = EpisodeRecorder(
        record_images=cam_attached, record_depth=args.record_depth and cam_attached,
        record_objects=args.record_objects or args.record_foundationpose_objects,
    )
    writer = TeleopHDF5Writer(args.out_dir, task_id=args.task, scene_registry=scene_reg)
    failed_writer = None  # lazy

    # state machine + event source
    sm = CollectionStateMachine()
    use_scripted = args.auto or args.headless or (args.teleop_device == "mock" and not args.keyboard)
    if use_scripted:
        ev_source = ScriptedEventSource(episode_len=args.max_steps_per_episode, num_episodes=args.num_episodes)
        print(f"[collect] event source = SCRIPTED (auto {args.num_episodes} ep x {args.max_steps_per_episode} steps).",
              flush=True)
    else:
        try:
            ev_source = CarbKeyboardSource()
            print("[collect] event source = KEYBOARD. Keys: SPACE pause | S success | F fail | "
                  "D discard | R reset | N next | E e-stop | H home | Q quit | 1-6 skill.", flush=True)
        except Exception as exc:
            print(f"[collect] keyboard unavailable ({exc}); falling back to scripted.", flush=True)
            ev_source = ScriptedEventSource(episode_len=args.max_steps_per_episode, num_episodes=args.num_episodes)

    episode_counter = {"n": 0}

    def prepare_next_episode():
        episode_counter["n"] += 1
        eid = f"{args.episode_id_prefix}_{episode_counter['n']:04d}_{int(time.time())}"
        recorder.reset(
            episode_id=eid, task_instruction=label["task_instruction"],
            skill_type=label["skill_type"], target_name=label["target_name"],
            scene_registry=scene_reg, seed=args.seed,
        )
        st = provider.get_state()
        mapper.reset(provider.arm_joint_pos(st).detach().cpu().tolist())
        sm.prepare_episode()

    def reset_scene():
        env.reset(seed=args.seed)
        try:
            provider.reset_cabinet_joint("joint_0", 0.0)
        except Exception:
            pass

    def flush_episode(success: bool):
        nonlocal failed_writer
        meta = recorder.episode_meta(success)
        if success:
            name = writer.write_episode(recorder.frames, meta)
            print(f"[collect] saved {name} success={success} steps={meta['num_steps']} -> {writer.path.name}",
                  flush=True)
        elif args.save_failed:
            if failed_writer is None:
                failed_writer = TeleopHDF5Writer(
                    Path(args.out_dir) / "failed", task_id=args.task,
                    scene_registry=scene_reg, update_latest=False,
                )
            failed_writer.write_episode(recorder.frames, meta)
            print(f"[collect] saved FAILED episode steps={meta['num_steps']} -> {failed_writer.path}", flush=True)
        else:
            print(f"[collect] failed episode discarded (steps={meta['num_steps']}; --save_failed off).", flush=True)

    prepare_next_episode()

    sim_dt = env_cfg.sim.dt * env_cfg.decimation
    sim_time = 0.0
    home_target = None
    step_count = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            provider.set_sim_time(sim_time)
            state = provider.get_state()

            quit_now = False
            for event, payload in ev_source.poll(recorder.num_steps):
                tr = sm.handle(event, payload)
                if tr.home_requested:
                    home_target = list(HOME_Q)
                if tr.reset_scene:
                    reset_scene()
                if tr.flush_success:
                    flush_episode(True)
                if tr.flush_failed:
                    flush_episode(False)
                if tr.begin_recording:
                    print(f"[collect] RECORDING episode {recorder.episode_id}", flush=True)
                if tr.prepare_next:
                    prepare_next_episode()
                if tr.should_quit:
                    quit_now = True
            if quit_now or sm.is_terminal():
                break

            # decide the joint action for this step
            mapped = None
            last_raw_q = [0.0] * 7
            if home_target is not None:
                q_des = mapper.map(home_target, 1.0)["q_des"]
                if max(abs(a - b) for a, b in zip(q_des, home_target)) < 0.02:
                    home_target = None  # reached
                action = provider.make_joint_action_from_q_des(q_des, 1.0)
            elif sm.is_following():
                # Track the teleop device (RECORDING drives + records; PREPARE drives
                # so the human can align before pressing START).
                mapper.release_estop()
                q_raw, grip = source.read()
                mapped = mapper.map(q_raw, grip)
                last_raw_q = list(q_raw)[:7]
                action = provider.make_joint_action_from_q_des(mapped["q_des"], mapped["gripper_cmd"])
            else:
                # PAUSED / e-stop / pending / resetting -> freeze current posture.
                action = provider.make_hold_joint_action(state, None)

            env.step(action)

            # record frame AFTER stepping (obs reflects the applied action's result)
            if sm.is_recording() and mapped is not None:
                post = provider.get_state()
                images: dict = {}
                if cam_attached and vla_adapter is not None:
                    if front_adapter is not None:
                        try:
                            front_frame = front_adapter.capture(require_depth=False)
                            if front_frame.get("rgb") is not None:
                                images["front_rgb"] = front_frame["rgb"]   # LeRobot `image`
                        except Exception:
                            pass
                    try:
                        vla_frame = vla_adapter.capture()  # VLA cam has no depth -> no raise
                        if vla_frame.get("rgb") is not None:
                            images["wrist_rgb"] = vla_frame["rgb"]         # LeRobot `wrist_image`
                    except Exception:
                        pass
                    if args.record_depth and fp_adapter is not None:
                        try:
                            fp_frame = fp_adapter.capture()  # FP cam: depth required (raises if missing)
                            if fp_frame.get("depth") is not None:
                                images["wrist_depth"] = fp_frame["depth"]
                        except Exception:
                            pass
                objects = capture_objects(post) if recorder.record_objects else []
                rec_frame = FrameBuilder.build(
                    step_id=recorder.num_steps, timestamp=time.time(),
                    joint_pos=provider.arm_joint_pos(post),
                    joint_vel=post.robot.joint_vel[provider._arm_joint_ids],
                    ee_pose=ee_pose_xyzw(post), gripper_width=post.robot.gripper_width,
                    raw_q=last_raw_q, filtered_q=mapped["q_des"], joint_target=mapped["q_des"],
                    gripper=mapped["gripper_norm"], images=images, objects=objects,
                )
                recorder.append(rec_frame)

            sim_time += sim_dt
            step_count += 1

    source.close()
    if isinstance(ev_source, CarbKeyboardSource):
        ev_source.close()
    path = writer.close()
    if failed_writer is not None:
        failed_writer.close()
    env.close()

    # session manifest
    manifest = {
        "hdf5": path, "num_demos": writer.num_demos, "task_id": args.task,
        "scene_registry": scene_reg, "teleop_device": args.teleop_device,
        "episodes_saved": sm.episodes_saved, "episodes_discarded": sm.episodes_discarded,
        "had_camera": cam_attached, "recorded_objects": recorder.record_objects,
    }
    man_path = Path(args.out_dir) / "last_session.json"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[collect] DONE. demos={writer.num_demos} hdf5={path}", flush=True)
    print(f"[collect] manifest -> {man_path}", flush=True)
    if writer.num_demos == 0:
        print("[collect] WARNING: no successful demos were saved.", flush=True)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
