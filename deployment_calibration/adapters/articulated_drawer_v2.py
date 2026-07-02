"""Member-aware articulated-drawer adapter (v2).

Resolves drawer_name -> member -> articulation -> joint -> link -> handle pose from the FROZEN
scene/paper_scene_v2/mechanism_registry.json (NOT the volatile shared drawer_target_config).
Supports desktop cabinet + Sektion drawers. Legacy v1 adapter (isaac_open_drawer.py) is kept.

Provides:
  load_mechanism_registry(path) -> dict
  MechanismSpec(...)                              resolved mechanism record
  reset_full_v2(env, provider, spec, initial_open) -> invariants dict (Stage-0 checks)
  set_drawer_damping_v2(env, spec, damping)      -> {"requested":.., "effective":..} (Stage-3, verified)
  read_drawer_damping_v2(env, spec)              -> float
  run_drawer_episode_v2(...)                     -> (x, y, provenance, trajectory_logger, sim_time)

theta (3 sampled): grasp_offset_local_y, max_pos_step, pull_lead. Fixed params from THETA_FIXED.
Handle pose comes from the frozen paper single source (override_grasp_local).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from runtime.skill_request import SkillRequest
from runtime.skill_types import ExecutionStatus, SkillType

from contracts.episode_schema_v2 import (
    THETA_FIXED, build_outcome, build_x, normalize_failure,
)

_PAPER = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = _PAPER / "scene" / "paper_scene_v2" / "mechanism_registry.json"
DEFAULT_GRASP = _PAPER / "scene" / "paper_scene_v2" / "grasp_poses.json"


def load_mechanism_registry(path: str | Path = DEFAULT_REGISTRY) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))["mechanisms"]


def load_paper_handles(path: str | Path = DEFAULT_GRASP) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))["poses"]


@dataclass
class MechanismSpec:
    drawer_name: str
    member: str
    joint_name: str
    link_name: str
    mechanism_id: str
    handle_local_pos: list
    handle_local_quat: list

    @classmethod
    def resolve(cls, drawer_name: str, registry: dict, handles: dict) -> "MechanismSpec":
        r = registry[drawer_name]
        h = handles.get(f"handle_{drawer_name}", {})
        return cls(drawer_name=drawer_name, member=r["member"], joint_name=r["joint_name"],
                   link_name=r["link_name"], mechanism_id=f"{r['member']}:{drawer_name}",
                   handle_local_pos=h.get("pos") or r.get("handle_local_pos"),
                   handle_local_quat=h.get("quat") or r.get("handle_local_quat"))


def _joint_id(asset, joint_name):
    jn = list(asset.data.joint_names)
    if joint_name in jn:
        return jn.index(joint_name)
    ids, _ = asset.find_joints(joint_name)
    return int(ids[0]) if ids else None


def read_drawer_damping_v2(env, spec: MechanismSpec) -> float:
    a = env.unwrapped.scene[spec.member]
    ji = _joint_id(a, spec.joint_name)
    return float(a.data.joint_damping[0, ji]) if (ji is not None and hasattr(a.data, "joint_damping")) else float("nan")


def set_drawer_damping_v2(env, spec: MechanismSpec, damping: float) -> dict:
    """Set the drawer joint's actuator damping and READ BACK the effective value (verification).
    Sets stiffness 0 so the drawer stays free (pure physical pull) while damping resists motion."""
    a = env.unwrapped.scene[spec.member]
    ji = _joint_id(a, spec.joint_name)
    dev = a.data.joint_pos.device
    n = a.num_instances
    a.write_joint_stiffness_to_sim(torch.zeros((n, 1), device=dev), joint_ids=[ji])
    a.write_joint_damping_to_sim(torch.full((n, 1), float(damping), device=dev), joint_ids=[ji])
    eff = read_drawer_damping_v2(env, spec)
    return {"requested": float(damping), "effective": eff, "joint": spec.joint_name, "member": spec.member}


def _zero_member_velocity(a):
    a.write_joint_state_to_sim(a.data.joint_pos.clone(), torch.zeros_like(a.data.joint_vel))


def reset_full_v2(env, provider, spec: MechanismSpec, initial_open: float = 0.0,
                  settle_steps: int = 6, executor=None) -> dict:
    """Full independent reset: robot->default, target member drawer joint->initial_open, other joints of
    that member->0, all velocities 0, executor reset. Returns verified invariants for Stage-0."""
    scene = provider.scene
    robot = scene["robot"]
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.joint_vel))
    a = scene[spec.member]
    jp = a.data.joint_pos.clone()
    jv = torch.zeros_like(a.data.joint_vel)
    # zero all joints of this member, then set the target drawer joint to initial_open
    for jn in a.data.joint_names:
        ji = _joint_id(a, jn)
        jp[:, ji] = 0.0
    ti = _joint_id(a, spec.joint_name)
    jp[:, ti] = float(initial_open)
    a.write_joint_state_to_sim(jp, jv)
    if executor is not None:
        executor.reset()
    # settle a few hold frames
    for _ in range(settle_steps):
        st = provider.get_state()
        env.step(provider.make_hold_joint_action(st, 1.0))
    # verify invariants
    st = provider.get_state()
    inv = {
        "robot_joint_pos": [round(float(v), 6) for v in robot.data.joint_pos[0].tolist()],
        "robot_joint_vel_absmax": round(float(robot.data.joint_vel[0].abs().max()), 6),
        "drawer_joint_pos": round(float(a.data.joint_pos[0, ti]), 6),
        "drawer_joint_vel": round(float(a.data.joint_vel[0, ti]), 6),
        "tcp_pos": [round(float(v), 6) for v in st.robot.tcp_pose.pos_w.tolist()],
        "tcp_quat": [round(float(v), 6) for v in st.robot.tcp_pose.quat_w.tolist()],
        "gripper_width": round(float(getattr(st.robot, "gripper_width", 0.0)), 6),
    }
    return inv


def _theta_to_params(g: dict, theta: dict, spec: MechanismSpec) -> dict:
    p = dict(THETA_FIXED)
    p["max_pos_step"] = float(theta["max_pos_step"])
    p["pull_lead"] = float(theta["pull_lead"])
    p["grasp_offset_local_xyz"] = [0.0, float(theta["grasp_offset_local_y"]), 0.0]
    p["target_open_position"] = float(g["target_open_position"])
    p["drawer_link"] = spec.link_name
    if spec.handle_local_pos and spec.handle_local_quat:
        p["override_grasp_local"] = {"pos": list(spec.handle_local_pos), "quat": list(spec.handle_local_quat)}
    return p


def _phase_durations(history: list[dict]) -> dict:
    """Duration in each state from the skill transition log (sim time). Mirrors v1."""
    out: dict = {}
    for i, rec in enumerate(history or []):
        sn, t0 = rec.get("to"), rec.get("time")
        if sn is None or t0 is None:
            continue
        t1 = history[i + 1]["time"] if i + 1 < len(history) else None
        if t1 is None:
            continue
        out[sn] = out.get(sn, 0.0) + max(0.0, float(t1) - float(t0))
    return {k: round(v, 4) for k, v in out.items()}


def run_drawer_episode_v2(env, provider, executor, adapter, control_dt, spec: MechanismSpec,
                          g: dict, theta: dict, sim_time: float = 0.0, max_steps: int = 1800):
    """Run ONE member-aware open_drawer episode. Scene must already be freshly reset. Returns
    (x, y, provenance, TrajectoryLogger, sim_time). Mirrors the working v1 interface."""
    from runtime.trajectory_logger import TrajectoryLogger, measured_angular_speed, measured_speed

    req = SkillRequest(
        request_id=f"open_drawer_{spec.drawer_name}_{time.time_ns()}",
        skill_type=SkillType.OPEN_DRAWER, source_object=None,
        destination_type="drawer", destination_object=spec.drawer_name,
        parameters=_theta_to_params(g, theta, spec))
    provider.set_sim_time(sim_time)
    state = provider.get_state()
    executor.reset()
    executor.start(req, state)
    skill = executor.active_skill

    member_asset = provider.scene[spec.member]
    jid = skill.obs_adapter._joint_id if getattr(skill, "obs_adapter", None) is not None else _joint_id(member_asset, spec.joint_name)
    initial_drawer = float(member_asset.data.joint_pos[0, jid]) if jid is not None else float("nan")
    x = build_x(spec.mechanism_id, spec.drawer_name, spec.member,
                [float(v) for v in state.robot.joint_pos.tolist()],
                [float(v) for v in state.robot.tcp_pose.pos_w.tolist()],
                [float(v) for v in state.robot.tcp_pose.quat_w.tolist()],
                float(getattr(state.robot, "gripper_width", 0.0)), initial_drawer)

    tl = TrajectoryLogger()
    skill_start = sim_time
    wall0 = time.time()
    prev_p = prev_q = None
    cmd_err_sum = 0.0
    cmd_err_n = 0
    steps = 0
    while steps < max_steps:
        provider.set_sim_time(sim_time)
        s = provider.get_state()
        skill_ref = executor.active_skill
        with torch.no_grad():
            command = executor.step(s, control_dt)
            cmd_pose = getattr(getattr(skill_ref, "runtime", None), "last_command_pose", None)
            if cmd_pose is not None:
                cmd_err_sum += float(torch.linalg.norm(s.robot.tcp_pose.pos_w - cmd_pose.pos_w)); cmd_err_n += 1
            tel = getattr(skill_ref, "last_telemetry", {}) or {}
            tcp_p = [float(v) for v in s.robot.tcp_pose.pos_w.tolist()]
            tcp_q = [float(v) for v in s.robot.tcp_pose.quat_w.tolist()]
            jv = float(member_asset.data.joint_vel[0, jid]) if jid is not None else float("nan")
            tl.add(sim_time=sim_time, elapsed_time=sim_time - skill_start,
                   skill_state=tel.get("skill_state", ""),
                   joint_position=[float(v) for v in s.robot.joint_pos.tolist()],
                   joint_velocity=[float(v) for v in s.robot.joint_vel.tolist()],
                   tcp_position=tcp_p, tcp_orientation=tcp_q,
                   tcp_position_error=tel.get("tcp_position_error"),
                   measured_tcp_linear_speed=measured_speed(prev_p, tcp_p, control_dt),
                   measured_tcp_angular_speed=measured_angular_speed(prev_q, tcp_q, control_dt),
                   drawer_joint_position=tel.get("drawer_joint_position"),
                   drawer_joint_velocity=jv,
                   handle_relative_position_error=tel.get("handle_relative_position_error"))
            env.step(_to_action(provider, command, s))
        prev_p, prev_q = tcp_p, tcp_q
        sim_time += control_dt
        steps += 1
        if executor.status in (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.STOPPED):
            break

    res = executor.last_result
    o = res.outcomes if res else {}
    final_drawer = o.get("final_drawer_position")
    if final_drawer is None:
        final_drawer = float(member_asset.data.joint_pos[0, jid]) if jid is not None else float("nan")
    target = float(g["target_open_position"])
    task_err = abs(final_drawer - target) if final_drawer is not None else float("nan")
    y = build_outcome(
        success=bool(res.success) if res else False,
        failure_reason=(res.failure_reason if res else "OTHER"),
        final_joint_position=final_drawer if final_drawer is not None else float("nan"),
        task_outcome_error=task_err,
        overshoot=max(0.0, final_drawer - target) if final_drawer is not None else 0.0,
        skill_elapsed_time=o.get("elapsed_time") or (steps * control_dt),
        phase_durations=_phase_durations(getattr(getattr(skill, "runtime", None), "history", []) or []),
        handle_detached=bool(o.get("handle_detached")),
        handle_relative_error=o.get("mean_handle_relative_error") or 0.0,
        command_tracking_error=(cmd_err_sum / cmd_err_n) if cmd_err_n else 0.0,
        phase_goal_error=o.get("mean_tcp_tracking_error") or 0.0,
        wall_clock_time=time.time() - wall0)
    prov = {"requested_parameters": (res.requested_parameters if res else req.parameters),
            "effective_parameter_traces": (res.effective_parameters if res else []),
            "initial_drawer": initial_drawer, "n_steps": steps}
    return x, y, prov, tl, sim_time


def _to_action(provider, command, state):
    if getattr(command, "control_mode", None) == "joint":
        if getattr(command, "raw_joint_action", None) is not None:
            return provider.make_joint_action_from_raw(command.raw_joint_action)
        if getattr(command, "joint_target", None) is not None:
            return provider.make_joint_action_from_q_des(command.joint_target, command.gripper_command)
        return provider.make_hold_joint_action(state, None)
    return provider.make_action(command.tcp_pose_w, command.gripper_command)
