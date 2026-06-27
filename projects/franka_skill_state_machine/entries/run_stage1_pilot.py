# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Stage-1 batch pilot runner (num_envs=1).

Launches Isaac ONCE and runs many skill episodes in a loop, logging an Episode JSONL + per-episode
NPZ trajectory under ``experiments/stage1/<run_id>/``. Variable groups are kept separate:
initial_state (x) / task_target (g) / execution_parameters (theta) / outcomes (y).

Place episode:  reset -> grasp -> confirm grasp -> capture place initial_state -> place.
                grasp failure => setup_failure=true (NOT counted as a place failure; re-sampled).
Drawer episode: reset (correct joint: top=joint_0, middle=joint_2) -> open_drawer.

Samplers:
  --sampler fixed          all default params (regression). --episodes N.
  --sampler one_param      single-parameter sweep: --param NAME --values v1,v2,..  --repeats R
                           (vec3 param e.g. grasp_offset_local_xyz: --values "0,0,0;0,0.03,0;0,-0.03,0")
                           Paired design: repeat r uses the SAME reset_index across all levels.

Examples:
  # default regression
  ./isaaclab.sh -p projects/franka_skill_state_machine/entries/run_stage1_pilot.py \
      --skill place --episodes 10 --seed 42 --sampler fixed \
      --output_dir projects/franka_skill_state_machine/experiments/stage1 --headless
  # place sensitivity on descend step
  ./isaaclab.sh -p .../run_stage1_pilot.py --skill place --sampler one_param \
      --param descend_max_position_step --values 0.003,0.006,0.009 --repeats 5 --seed 42 --headless
"""

from __future__ import annotations

import argparse
import math
import os as _os
import sys as _sys
import time

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stage-1 batch pilot runner for place / open_drawer.")
parser.add_argument("--skill", type=str, required=True, choices=["place", "open_drawer"])
parser.add_argument("--episodes", type=int, default=10, help="Episodes for --sampler fixed.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--sampler", type=str, default="fixed", choices=["fixed", "one_param"])
parser.add_argument("--param", type=str, default=None, help="Parameter name for --sampler one_param.")
parser.add_argument("--values", type=str, default=None,
                    help="Comma list of scalar levels, or ';'-separated 'x,y,z' for a vec3 param.")
parser.add_argument("--repeats", type=int, default=5, help="Repeats per level for one_param.")
parser.add_argument("--drawer", type=str, default="top_drawer", choices=["top_drawer", "middle_drawer"])
parser.add_argument("--place_object", type=str, default="cube_1", choices=["cube_1", "cube_2", "cube_3"])
parser.add_argument("--place_point", type=str, default="point_a")
parser.add_argument("--initial_drawer_open", type=float, default=0.0, help="Initial drawer joint position (m).")
parser.add_argument("--base_reset_index", type=int, default=0)
parser.add_argument("--output_dir", type=str,
                    default="projects/franka_skill_state_machine/experiments/stage1")
parser.add_argument("--run_id", type=str, default=None)
parser.add_argument("--max_setup_retries", type=int, default=6, help="Re-sample attempts when grasp setup fails.")
parser.add_argument("--max_skill_steps", type=int, default=1800, help="Per-skill hard step cap.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import json
from pathlib import Path

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from runtime.base_skill import set_speed_scale
from runtime.drawer_target_config import DRAWER_TARGETS
from runtime.episode_logger import EpisodeLogger
from runtime.ik_joint_adapter import IKJointAdapter
from runtime.scene_state_provider import SceneStateProvider
from runtime.simple_scene_layout import SimpleSceneLayoutManager
from runtime.skill_request import SkillRequest
from runtime.skill_types import ExecutionStatus, FailureReason, SkillType
from runtime.target_registry import TargetRegistry
from runtime.trajectory_logger import TrajectoryLogger, measured_angular_speed, measured_speed
from state_machine.skill_executor import JointBackendConfig, SkillExecutor

TASK_ID = "Isaac-Stack-Cube-Franka-JointPolicy-v0"
TERMINAL = {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.STOPPED, ExecutionStatus.NOT_IMPLEMENTED}
REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_PLACE_POINTS = {
    "point_a": [0.42, 0.10, 0.00],
    "point_b": [0.55, 0.10, 0.00],
    "point_c": [0.68, 0.10, 0.00],
}

# first-round sensitivity ranges (documented in design doc; for parameter_ranges.json)
PARAMETER_RANGES = {
    "place": {
        "release_clearance": [0.0, 0.02, 0.04],
        "descend_max_position_step": [0.003, 0.006, 0.009],
        "move_max_position_step": [0.006, 0.012, 0.018],
        "open_duration": [0.25, 0.45, 0.90],
        "settle_after_release_duration": [0.2, 0.5, 1.0],
    },
    "open_drawer": {
        "max_pos_step": [0.010, 0.020, 0.030],
        "pull_lead": [0.04, 0.08, 0.12],
        "grasp_offset_local_xyz": ["0,0,0", "0,0.03,0", "0,-0.03,0"],
        "close_duration": [0.5, 1.0, 1.5],
        "target_open_position": [0.10, 0.20, 0.30],  # task-target sensitivity
    },
}


def _parse_values(skill: str, param: str, raw: str):
    """Return a list of levels. vec3 params -> list[list[float]]; scalars -> list[float]."""
    is_vec3 = param.endswith("_xyz")
    if is_vec3:
        out = []
        for chunk in raw.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            out.append([float(x) for x in chunk.split(",")])
        return out
    return [float(x) for x in raw.split(",")]


def _build_specs():
    """Return list of dicts: {episode_index, level_label, reset_index, params{}}."""
    base = args_cli.base_reset_index
    specs = []
    if args_cli.sampler == "fixed":
        for i in range(args_cli.episodes):
            specs.append({"level": "default", "reset_index": base + i, "params": {}})
    else:  # one_param
        if not args_cli.param or args_cli.values is None:
            raise ValueError("--sampler one_param requires --param and --values")
        levels = _parse_values(args_cli.skill, args_cli.param, args_cli.values)
        for level in levels:
            for r in range(args_cli.repeats):
                specs.append({
                    "level": f"{args_cli.param}={level}",
                    "reset_index": base + r,           # paired across levels
                    "params": {args_cli.param: level},
                })
    return specs


def _reset_drawer(provider, joint_name: str, value: float):
    """Reset all functional drawer joints closed, then set the target joint to ``value``."""
    for cfg in DRAWER_TARGETS.values():
        try:
            provider.reset_cabinet_joint(cfg["joint_name"], 0.0)
        except Exception:
            pass
    provider.reset_cabinet_joint(joint_name, value)


def _settle(env, provider, n=5):
    state = provider.get_state()
    action = provider.make_hold_joint_action(state, 1.0)
    for _ in range(n):
        env.step(action)


def _controller_meta(env_cfg, control_dt):
    return {
        "type": "joint_position_with_dls_ik",
        "control_dt": control_dt,
        "control_frequency_hz": round(1.0 / control_dt, 2),
        "ik_command_type": "pose",
        "ik_relative": False,
        "ik_method": "dls",
        "ik_max_joint_step": 0.20,
        "physics_dt": env_cfg.sim.dt,
        "decimation": env_cfg.decimation,
        "action_dim": 8,
        "task_id": TASK_ID,
    }


def _pose_list(pose):
    return [float(v) for v in pose.as_pose_tensor().tolist()]


def _run_skill_loop(executor, provider, env, sim_time_ref, control_dt, tl, skill_type, ctx):
    """Drive one already-started skill to terminal, recording the trajectory. Returns (status, sim_time)."""
    sim_time = sim_time_ref
    skill_start = sim_time
    prev_tcp_pos = None
    prev_tcp_quat = None
    steps = 0
    while simulation_app.is_running() and steps < args_cli.max_skill_steps:
        provider.set_sim_time(sim_time)
        state = provider.get_state()
        skill = executor.active_skill
        with torch.no_grad():
            command = executor.step(state, control_dt)
            action = _command_to_action(provider, command, state)
            _record_step(tl, skill, skill_type, state, prev_tcp_pos, prev_tcp_quat,
                         skill_start, sim_time, control_dt, ctx)
            env.step(action)
        prev_tcp_pos = [float(v) for v in state.robot.tcp_pose.pos_w.tolist()]
        prev_tcp_quat = [float(v) for v in state.robot.tcp_pose.quat_w.tolist()]
        sim_time += control_dt
        steps += 1
        if executor.status in TERMINAL:
            break
    return executor.status, sim_time


def _command_to_action(provider, command, state):
    if command.control_mode == "joint":
        if command.raw_joint_action is not None:
            return provider.make_joint_action_from_raw(command.raw_joint_action)
        if command.joint_target is not None:
            return provider.make_joint_action_from_q_des(command.joint_target, command.gripper_command)
        return provider.make_hold_joint_action(state, None)
    return provider.make_action(command.tcp_pose_w, command.gripper_command)


def _record_step(tl, skill, skill_type, state, prev_tcp_pos, prev_tcp_quat,
                 skill_start, sim_time, control_dt, ctx):
    tel = getattr(skill, "last_telemetry", {}) or {}
    tcp_pos = [float(v) for v in state.robot.tcp_pose.pos_w.tolist()]
    tcp_quat = [float(v) for v in state.robot.tcp_pose.quat_w.tolist()]
    row = {
        "sim_time": sim_time,
        "elapsed_time": sim_time - skill_start,
        "skill_state": tel.get("skill_state", ""),
        "joint_position": [float(v) for v in state.robot.joint_pos.tolist()],
        "joint_velocity": [float(v) for v in state.robot.joint_vel.tolist()],
        "tcp_position": tcp_pos,
        "tcp_orientation": tcp_quat,
        "target_tcp_position": tel.get("target_tcp_position"),
        "target_tcp_orientation": tel.get("target_tcp_orientation"),
        "tcp_position_error": tel.get("tcp_position_error"),
        "tcp_orientation_error": tel.get("tcp_orientation_error"),
        "measured_tcp_linear_speed": measured_speed(prev_tcp_pos, tcp_pos, control_dt),
        "measured_tcp_angular_speed": measured_angular_speed(prev_tcp_quat, tcp_quat, control_dt),
        "gripper_command": tel.get("gripper_command"),
        "contact_available": False,
        "contact_force": float("nan"),
        "collision_available": False,
        "collision_flag": float("nan"),
    }
    if skill_type == SkillType.PLACE:
        obj = state.objects.get(ctx.get("object_name", ""))
        if obj is not None:
            row["object_position"] = [float(v) for v in obj.pose.pos_w.tolist()]
            row["object_orientation"] = [float(v) for v in obj.pose.quat_w.tolist()]
            if obj.lin_vel_w is not None:
                row["object_linear_velocity"] = [float(v) for v in obj.lin_vel_w.tolist()]
            if obj.ang_vel_w is not None:
                row["object_angular_velocity"] = [float(v) for v in obj.ang_vel_w.tolist()]
            tt = ctx.get("task_target_pos")
            if tt is not None:
                row["object_position_error"] = float(
                    sum((float(obj.pose.pos_w[i]) - tt[i]) ** 2 for i in range(3)) ** 0.5)
    else:  # open_drawer
        row["drawer_joint_position"] = tel.get("drawer_joint_position")
        row["handle_position"] = tel.get("handle_position")
        row["target_handle_grasp_position"] = tel.get("target_handle_grasp_position")
        row["handle_relative_position_error"] = tel.get("handle_relative_position_error")
        row["drawer_progress"] = tel.get("drawer_progress")
        jv = ctx.get("drawer_joint_vel_fn")
        if jv is not None:
            row["drawer_joint_velocity"] = jv()
    tl.add(**row)


def _make_grasp_request(cube: str) -> SkillRequest:
    return SkillRequest(request_id=f"grasp_{cube}_{time.time_ns()}", skill_type=SkillType.GRASP, source_object=cube)


def run_place_episode(executor, provider, env, layout_manager, spec, control_dt, env_cfg, el, episode_id, sim_time):
    """reset -> grasp -> confirm -> place. Returns (record, sim_time)."""
    cube = args_cli.place_object
    point = args_cli.place_point
    point_xyz = DEFAULT_PLACE_POINTS.get(point, DEFAULT_PLACE_POINTS["point_a"])
    setup_failure = False
    grasp_status = None
    for attempt in range(args_cli.max_setup_retries):
        ridx = spec["reset_index"] + attempt * 1000
        layout_manager.reset_layout(reset_index=ridx)
        _reset_drawer(provider, "joint_0", 0.0)
        _settle(env, provider)
        provider.set_sim_time(sim_time)
        state = provider.get_state()
        executor.reset()
        executor.start(_make_grasp_request(cube), state)
        grasp_status, sim_time = _run_skill_loop(
            executor, provider, env, sim_time, control_dt, TrajectoryLogger(), SkillType.GRASP, {})
        if grasp_status == ExecutionStatus.SUCCEEDED and executor.held_object is not None:
            break
        setup_failure = True
    if executor.held_object is None:
        rec = EpisodeLogger.build_episode_record(
            episode_id=episode_id, seed=args_cli.seed, skill="place", env_id=0,
            controller=_controller_meta(env_cfg, control_dt),
            initial_state={}, task_target={"target_surface_xyz": point_xyz, "point_name": point},
            requested_parameters=spec["params"], effective_parameters=[],
            outcomes={"setup_failure": True, "grasp_status": None if grasp_status is None else grasp_status.value},
            success=False, failure_reason=FailureReason.SETUP_GRASP_FAILED.value, trajectory_file=None,
            extra={"level": spec["level"], "setup_failure": True})
        return rec, sim_time

    # capture place initial state (x)
    provider.set_sim_time(sim_time)
    state = provider.get_state()
    held = executor.held_object
    obj = state.objects.get(cube)
    initial_state = {
        "initial_robot_joint_position": [float(v) for v in state.robot.joint_pos.tolist()],
        "initial_tcp_pose": _pose_list(state.robot.tcp_pose),
        "initial_object_pose": None if obj is None else _pose_list(obj.pose),
        "object_to_tcp_transform": {
            "pos": [float(v) for v in held.object_to_tcp_pos.tolist()],
            "quat": [float(v) for v in held.object_to_tcp_quat.tolist()],
        },
        "object_type": cube,
        "grasp_setup_attempts": attempt + 1,
    }
    place_params = dict(spec["params"])
    place_req = SkillRequest(
        request_id=f"place_{point}_{time.time_ns()}", skill_type=SkillType.PLACE, source_object=None,
        destination_type="point", destination_object=point,
        parameters={"target_frame": "env_local", "target_surface_xyz": list(point_xyz), **place_params})
    executor.start(place_req, state)
    skill = executor.active_skill
    plan = getattr(getattr(skill, "runtime", None), "plan", None)
    task_target_pos = None
    if plan is not None:
        task_target_pos = [float(v) for v in plan.task_target_pose.pos_w.tolist()]
    tl = TrajectoryLogger()
    ctx = {"object_name": cube, "task_target_pos": task_target_pos}
    status, sim_time = _run_skill_loop(executor, provider, env, sim_time, control_dt, tl, SkillType.PLACE, ctx)
    res = executor.last_result
    traj_rel = el.trajectory_relpath(episode_id)
    tl.save(el.trajectory_abspath(episode_id))
    rec = EpisodeLogger.build_episode_record(
        episode_id=episode_id, seed=args_cli.seed, skill="place", env_id=0,
        controller=_controller_meta(env_cfg, control_dt),
        initial_state=initial_state,
        task_target=(res.task_target if res else {}) | {"target_surface_xyz": point_xyz, "point_name": point},
        requested_parameters=(res.requested_parameters if res else spec["params"]),
        effective_parameters=(res.effective_parameters if res else []),
        outcomes=(res.outcomes if res else {}),
        success=bool(res.success) if res else False,
        failure_reason=(res.failure_reason if res else "no_result"),
        trajectory_file=traj_rel,
        extra={"level": spec["level"], "setup_failure": setup_failure, "legacy_reached": res.legacy_reached if res else None})
    return rec, sim_time


def run_drawer_episode(executor, provider, env, layout_manager, spec, control_dt, env_cfg, el, episode_id, sim_time):
    drawer = args_cli.drawer
    joint_name = DRAWER_TARGETS[drawer]["joint_name"]
    layout_manager.reset_layout(reset_index=spec["reset_index"])
    _reset_drawer(provider, joint_name, float(args_cli.initial_drawer_open))
    _settle(env, provider)
    provider.set_sim_time(sim_time)
    state = provider.get_state()
    executor.reset()
    params = dict(spec["params"])
    req = SkillRequest(
        request_id=f"open_drawer_{drawer}_{time.time_ns()}", skill_type=SkillType.OPEN_DRAWER, source_object=None,
        destination_type="drawer", destination_object=drawer, parameters=params)
    executor.start(req, state)
    skill = executor.active_skill
    cabinet = provider.scene["cabinet"]
    jid = skill.obs_adapter._joint_id if getattr(skill, "obs_adapter", None) is not None else None
    initial_drawer = float(cabinet.data.joint_pos[0, jid]) if jid is not None else float(args_cli.initial_drawer_open)
    handle_pose = None
    try:
        h = skill._handle_pos()
        handle_pose = [float(v) for v in h.tolist()]
    except Exception:
        pass
    initial_state = {
        "initial_robot_joint_position": [float(v) for v in state.robot.joint_pos.tolist()],
        "initial_tcp_pose": _pose_list(state.robot.tcp_pose),
        "initial_drawer_position": initial_drawer,
        "drawer_name": drawer,
        "handle_pose": handle_pose,
    }
    ctx = {"drawer_joint_vel_fn": (lambda: float(cabinet.data.joint_vel[0, jid])) if jid is not None else None}
    tl = TrajectoryLogger()
    status, sim_time = _run_skill_loop(executor, provider, env, sim_time, control_dt, tl, SkillType.OPEN_DRAWER, ctx)
    res = executor.last_result
    traj_rel = el.trajectory_relpath(episode_id)
    tl.save(el.trajectory_abspath(episode_id))
    rec = EpisodeLogger.build_episode_record(
        episode_id=episode_id, seed=args_cli.seed, skill="open_drawer", env_id=0,
        controller=_controller_meta(env_cfg, control_dt),
        initial_state=initial_state,
        task_target=(res.task_target if res else {"target_open_position": 0.20, "drawer_name": drawer}),
        requested_parameters=(res.requested_parameters if res else spec["params"]),
        effective_parameters=(res.effective_parameters if res else []),
        outcomes=(res.outcomes if res else {}),
        success=bool(res.success) if res else False,
        failure_reason=(res.failure_reason if res else "no_result"),
        trajectory_file=traj_rel,
        extra={"level": spec["level"], "setup_failure": False})
    return rec, sim_time


def main():
    torch.manual_seed(args_cli.seed)
    set_speed_scale(1.0)  # fixed for reproducibility
    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env_cfg.seed = args_cli.seed
    if getattr(env_cfg, "events", None) is not None and hasattr(env_cfg.events, "randomize_cube_positions"):
        env_cfg.events.randomize_cube_positions = None
    # ik_pull: free-sliding drawer (no spring fighting the gripper)
    if hasattr(env_cfg.scene, "cabinet") and hasattr(env_cfg.scene.cabinet, "actuators"):
        if "drawers" in env_cfg.scene.cabinet.actuators:
            env_cfg.scene.cabinet.actuators["drawers"].stiffness = 0.0
            env_cfg.scene.cabinet.actuators["drawers"].damping = 2.0
    env_cfg.episode_length_s = 1.0e9
    if getattr(env_cfg, "terminations", None) is not None:
        for _t in ("cube_1_dropping", "cube_2_dropping", "cube_3_dropping", "success", "cubes_stacked"):
            if hasattr(env_cfg.terminations, _t):
                setattr(env_cfg.terminations, _t, None)

    env = gym.make(TASK_ID, cfg=env_cfg)
    env.reset(seed=args_cli.seed)

    provider = SceneStateProvider(env)
    layout_manager = SimpleSceneLayoutManager(env=env, base_seed=args_cli.seed)
    registry = TargetRegistry(env.unwrapped.device)
    adapter = IKJointAdapter(env)
    backend = JointBackendConfig(
        mode="joint", grasp_backend="joint_ik", place_backend="joint_ik", drawer_backend="ik_pull",
        adapter=adapter, drawer_env=env, arm_joint_ids=provider._arm_joint_ids)
    executor = SkillExecutor(registry, backend=backend,
                             log_path="logs/skill_tests/stage1_pilot_executor.jsonl")

    control_dt = env_cfg.sim.dt * env_cfg.decimation
    run_id = args_cli.run_id or f"{args_cli.skill}_{args_cli.sampler}_seed{args_cli.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    el = EpisodeLogger(args_cli.output_dir, run_id)
    specs = _build_specs()
    metadata = {
        "run_id": run_id, "skill": args_cli.skill, "sampler": args_cli.sampler, "seed": args_cli.seed,
        "episodes_planned": len(specs), "param": args_cli.param, "values": args_cli.values,
        "repeats": args_cli.repeats, "drawer": args_cli.drawer, "place_object": args_cli.place_object,
        "place_point": args_cli.place_point, "speed_scale": 1.0,
    }
    el.write_metadata(metadata, PARAMETER_RANGES.get(args_cli.skill, {}), _controller_meta(env_cfg, control_dt),
                      run_command=" ".join(_sys.argv), repo_root=REPO_ROOT)
    done = el.completed_episode_ids()
    print(f"[pilot] run_id={run_id} skill={args_cli.skill} planned={len(specs)} resume_skip={len(done)}", flush=True)

    sim_time = 0.0
    n_succ = 0
    for i, spec in enumerate(specs):
        episode_id = f"{args_cli.skill}_{i:06d}"
        if episode_id in done:
            continue
        t0 = time.time()
        try:
            if args_cli.skill == "place":
                rec, sim_time = run_place_episode(executor, provider, env, layout_manager, spec,
                                                  control_dt, env_cfg, el, episode_id, sim_time)
            else:
                rec, sim_time = run_drawer_episode(executor, provider, env, layout_manager, spec,
                                                   control_dt, env_cfg, el, episode_id, sim_time)
        except Exception as exc:  # an episode error must NOT abort the batch
            import traceback
            traceback.print_exc()
            rec = EpisodeLogger.build_episode_record(
                episode_id=episode_id, seed=args_cli.seed, skill=args_cli.skill, env_id=0,
                controller=_controller_meta(env_cfg, control_dt), initial_state={},
                task_target={}, requested_parameters=spec["params"], effective_parameters=[],
                outcomes={"exception": str(exc)}, success=False, failure_reason="EPISODE_EXCEPTION",
                trajectory_file=None, extra={"level": spec["level"]})
        el.log_episode(rec)
        n_succ += int(bool(rec.get("success")))
        print(f"[pilot] {episode_id} level={spec['level']} success={rec.get('success')} "
              f"reason={rec.get('failure_reason')} wall={time.time()-t0:.1f}s "
              f"({i+1}/{len(specs)} succ={n_succ})", flush=True)

    print(f"[pilot] DONE run_id={run_id} success {n_succ}/{len(specs)} -> {el.run_dir}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
