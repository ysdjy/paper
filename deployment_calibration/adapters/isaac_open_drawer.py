"""[LEGACY v1 -- kept, do not delete] Isaac execution backend adapter for open_drawer (paper round-1).

Superseded by adapters/articulated_drawer_v2.py (member-aware: desktop cabinet + Sektion; reads the
frozen scene/paper_scene_v2 registry). This v1 is cabinet-only (hardcoded scene["cabinet"]) and is
retained for reproducing round-1 v1 data. New experiments must use the v2 adapter.

Isaac execution backend adapter for open_drawer episodes (paper round-1).

Reuses the verified franka_skill_state_machine backend (SkillExecutor + OpenDrawerIKSkill + DLS IK +
SceneStateProvider) as the *physical executor* only. Adds:
  * FULL per-episode reset (robot -> default joint state, objects/cabinet -> deterministic layout,
    drawer joint -> initial open) so episodes are INDEPENDENT (fixes bias R1);
  * clean, contract-correct labels: separates phase_goal_error / command_tracking_error /
    task_outcome_error (R3); separates sim/skill/phase/wall time (R4).

Functions take already-constructed Isaac handles (env/provider/executor/...) so this module has no
top-level Isaac import beyond torch and is only used after the app is launched.
"""

from __future__ import annotations

import math
import time

import torch

from runtime.drawer_target_config import DRAWER_TARGETS
from runtime.skill_request import SkillRequest
from runtime.skill_types import ExecutionStatus, SkillType
from runtime.trajectory_logger import TrajectoryLogger, measured_angular_speed, measured_speed
from contracts.episode_schema import normalize_failure


def reset_full(env, provider, layout_manager, reset_index: int, drawer_joint: str, initial_open: float):
    """Independent-episode reset: robot -> default joints, scene -> deterministic layout, drawer set."""
    robot = provider.scene["robot"]
    djp = robot.data.default_joint_pos.clone()
    djv = torch.zeros_like(robot.data.default_joint_vel)
    robot.write_joint_state_to_sim(djp, djv)
    if hasattr(robot, "set_joint_position_target"):
        try:
            robot.set_joint_position_target(djp)
        except Exception:
            pass
    layout_manager.reset_layout(reset_index=reset_index)
    for cfg in DRAWER_TARGETS.values():
        try:
            provider.reset_cabinet_joint(cfg["joint_name"], 0.0)
        except Exception:
            pass
    provider.reset_cabinet_joint(drawer_joint, float(initial_open))
    # settle holding the default pose (gripper open) so writes take effect & velocities damp
    state = provider.get_state()
    action = provider.make_hold_joint_action(state, 1.0)
    for _ in range(8):
        env.step(action)


def _phase_durations(history: list[dict]) -> dict[str, float]:
    """Duration spent in each state from the skill transition log (sim time)."""
    out: dict[str, float] = {}
    for i, rec in enumerate(history):
        state_name = rec.get("to")
        t0 = rec.get("time")
        if state_name is None or t0 is None:
            continue
        t1 = history[i + 1]["time"] if i + 1 < len(history) else None
        if t1 is None:
            continue
        out[state_name] = out.get(state_name, 0.0) + max(0.0, float(t1) - float(t0))
    return out


def _workspace_region(handle_xy) -> str:
    sx = "px" if handle_xy[0] >= 0 else "nx"
    sy = "py" if handle_xy[1] >= 0 else "ny"
    return f"{sx}_{sy}"


def run_open_drawer_episode(env, provider, executor, adapter, control_dt, drawer_name, g, theta,
                            sim_time, max_steps=1800):
    """Run ONE open_drawer episode (skill already gets a freshly reset scene). Returns (x, y, traj, sim_time)."""
    params = dict(theta)
    params["target_open_position"] = float(g["target_open_position"])
    req = SkillRequest(
        request_id=f"open_drawer_{drawer_name}_{time.time_ns()}",
        skill_type=SkillType.OPEN_DRAWER, source_object=None,
        destination_type="drawer", destination_object=drawer_name, parameters=params)

    provider.set_sim_time(sim_time)
    state = provider.get_state()
    executor.reset()
    executor.start(req, state)
    skill = executor.active_skill

    cabinet = provider.scene["cabinet"]
    jid = skill.obs_adapter._joint_id if getattr(skill, "obs_adapter", None) is not None else None
    initial_drawer = float(cabinet.data.joint_pos[0, jid]) if jid is not None else float("nan")
    handle_pose = None
    try:
        h = skill._handle_pos()
        handle_pose = [float(v) for v in h.tolist()]
    except Exception:
        pass
    x = {
        "initial_robot_joint_position": [float(v) for v in state.robot.joint_pos.tolist()],
        "initial_tcp_pose": [float(v) for v in state.robot.tcp_pose.as_pose_tensor().tolist()],
        "gripper_width": float(state.robot.gripper_width),
        "drawer_name": drawer_name,
        "initial_drawer_position": initial_drawer,
        "handle_pose": handle_pose,                      # privileged_in_sim
        "handle_pose_privileged_in_sim": True,
        "workspace_region": _workspace_region(handle_pose) if handle_pose else None,
    }

    tl = TrajectoryLogger()
    skill_start = sim_time
    wall0 = time.time()
    prev_p = prev_q = None
    cmd_err_max = 0.0
    cmd_err_sum = 0.0
    cmd_err_n = 0
    steps = 0
    while steps < max_steps:
        provider.set_sim_time(sim_time)
        s = provider.get_state()
        skill_ref = executor.active_skill
        with torch.no_grad():
            command = executor.step(s, control_dt)
            # command_tracking_error: TCP vs the bounded per-step command (R3 distinct metric)
            cmd_pose = getattr(getattr(skill_ref, "runtime", None), "last_command_pose", None)
            if cmd_pose is not None:
                ce = float(torch.linalg.norm(s.robot.tcp_pose.pos_w - cmd_pose.pos_w))
                cmd_err_max = max(cmd_err_max, ce); cmd_err_sum += ce; cmd_err_n += 1
            tel = getattr(skill_ref, "last_telemetry", {}) or {}
            tcp_p = [float(v) for v in s.robot.tcp_pose.pos_w.tolist()]
            tcp_q = [float(v) for v in s.robot.tcp_pose.quat_w.tolist()]
            jv = float(cabinet.data.joint_vel[0, jid]) if jid is not None else float("nan")
            tl.add(sim_time=sim_time, elapsed_time=sim_time - skill_start,
                   skill_state=tel.get("skill_state", ""),
                   joint_position=[float(v) for v in s.robot.joint_pos.tolist()],
                   joint_velocity=[float(v) for v in s.robot.joint_vel.tolist()],
                   tcp_position=tcp_p, tcp_orientation=tcp_q,
                   target_tcp_position=tel.get("target_tcp_position"),
                   target_tcp_orientation=tel.get("target_tcp_orientation"),
                   tcp_position_error=tel.get("tcp_position_error"),
                   tcp_orientation_error=tel.get("tcp_orientation_error"),
                   measured_tcp_linear_speed=measured_speed(prev_p, tcp_p, control_dt),
                   measured_tcp_angular_speed=measured_angular_speed(prev_q, tcp_q, control_dt),
                   gripper_command=tel.get("gripper_command"),
                   drawer_joint_position=tel.get("drawer_joint_position"),
                   drawer_joint_velocity=jv,
                   handle_position=tel.get("handle_position"),
                   target_handle_grasp_position=tel.get("target_handle_grasp_position"),
                   handle_relative_position_error=tel.get("handle_relative_position_error"),
                   drawer_progress=tel.get("drawer_progress"),
                   contact_available=False, contact_force=float("nan"),
                   collision_available=False, collision_flag=float("nan"))
            env.step(_to_action(provider, command, s))
        prev_p, prev_q = tcp_p, tcp_q
        sim_time += control_dt
        steps += 1
        if executor.status in (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.STOPPED):
            break

    res = executor.last_result
    o = res.outcomes if res else {}
    final_drawer = o.get("final_drawer_position")
    target = float(g["target_open_position"])
    y = {
        "success": bool(res.success) if res else False,
        "failure_reason": normalize_failure(res.failure_reason if res else "OTHER"),
        "final_drawer_position": final_drawer,
        "task_outcome_error": (abs(final_drawer - target) if final_drawer is not None else None),
        "drawer_overshoot": (final_drawer - target if final_drawer is not None else None),
        "skill_elapsed_time": (o.get("elapsed_time")),                 # sim time
        "phase_durations": _phase_durations(getattr(getattr(skill, "runtime", None), "history", []) or []),
        "wall_clock_time": time.time() - wall0,
        "command_tracking_error_max": cmd_err_max if cmd_err_n else None,
        "command_tracking_error_mean": (cmd_err_sum / cmd_err_n) if cmd_err_n else None,
        "phase_goal_error_max": o.get("maximum_tcp_tracking_error"),   # R3: was mislabeled "tracking"
        "phase_goal_error_mean": o.get("mean_tcp_tracking_error"),
        "handle_relative_error_max": o.get("maximum_handle_relative_error"),
        "handle_relative_error_mean": o.get("mean_handle_relative_error"),
        "handle_detached": o.get("handle_detached"),
        "target_reached_time": o.get("target_reached_time"),
        "timeout": o.get("timeout"),
        "n_steps": steps,
        "contact_available": False, "maximum_contact_force": None,
    }
    prov = {
        "requested_parameters": res.requested_parameters if res else {},
        "effective_parameter_traces": res.effective_parameters if res else [],
    }
    return x, y, prov, tl, sim_time


def _to_action(provider, command, state):
    if command.control_mode == "joint":
        if command.raw_joint_action is not None:
            return provider.make_joint_action_from_raw(command.raw_joint_action)
        if command.joint_target is not None:
            return provider.make_joint_action_from_q_des(command.joint_target, command.gripper_command)
        return provider.make_hold_joint_action(state, None)
    return provider.make_action(command.tcp_pose_w, command.gripper_command)
