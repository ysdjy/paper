"""Band-edge instrumented episode runner (v1, Isaac). Mirrors the verified `run_drawer_episode_v2` and
adds the six instrumentation groups + collision monitor + the CLOSE->PULL true-handle snapshot.

Injection reuses the verified `handle_calibration_bias_v1.biased_spec` (perceived handle only). The
snapshot uses the TRUE (unbiased) handle pose. This module is driven only by the generator's SMOKE modes
in this phase; it never runs the full 306 by itself.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

_PAPER = Path(__file__).resolve().parents[2]
for _p in (_PAPER, _PAPER / "deployment_calibration", _PAPER / "franka_skill_state_machine",
           Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import isaaclab.utils.math as math_utils

from band_edge_instrumentation_core_v1 import (
    JointMarginTracker, IKFailureCounter, ClampCounter, CloseSnapshot, failure_phase_from_history)
from band_edge_instrumentation_runtime_v1 import InstrumentedIKAdapter, CollisionMonitor


@dataclass
class BandEdgeHarness:
    env: object
    provider: object
    executor: object
    adapter: object
    control_dt: float
    reg: dict
    handles: dict
    scene: object
    _mod: object = field(default=None)
    sim_time: float = 0.0

    def spec(self, drawer):
        return self._mod.MechanismSpec.resolve(drawer, self.reg, self.handles)

    def reset(self, spec, initial_open=0.0):
        return self._mod.reset_full_v2(self.env, self.provider, spec, initial_open=initial_open,
                                       executor=self.executor)

    def set_damping(self, spec, d):
        return self._mod.set_drawer_damping_v2(self.env, spec, d)

    def read_damping(self, spec):
        return self._mod.read_drawer_damping_v2(self.env, spec)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


def launch_band_edge_scene(app_launcher, device="cuda:0", speed_scale=5.0) -> BandEdgeHarness:
    """Mirror of `_drawer_harness_v2.launch_drawer_scene` but with the collision monitor ENABLED and an
    InstrumentedIKAdapter (IK-failure + clamp counting)."""
    from franka_v1_skill_lab.scene import V1_BASE_TASK_ID
    from franka_v1_skill_lab.scene_interface import ResetMode, SceneConfig, SceneMode, SceneSession
    from runtime.base_skill import set_speed_scale
    from runtime.target_registry import TargetRegistry
    from state_machine.skill_executor import JointBackendConfig, SkillExecutor
    from skills.open_drawer_skill import OpenDrawerIKConfig
    from skills.close_drawer_skill import CloseDrawerIKConfig
    import adapters.articulated_drawer_v2 as adv

    cfg = SceneConfig(
        mode=SceneMode.TEST, task_id=V1_BASE_TASK_ID, device=device, headless=True,
        enable_cameras=False, enable_fp=False, free_microwave_door=False, load_latest_scene=True,
        add_microwave_stand=False, replace_microwave_with_fridge=False, lock_knife=True,
        enable_collision_monitor=True, spawn_init_markers=False, refine_handle_collisions=True,
        apply_saved_camera_offsets=False, reset_mode=ResetMode.STATIC, seed=1,
        disable_auto_reset=True, exclude_members=("microwave", "dishwasher"), hidden_members=())
    session = SceneSession.launch(cfg, _app_launcher=app_launcher)
    env, provider = session.env, session.provider
    control_dt = float(getattr(session, "_sim_dt", 0.02)) or 0.02
    set_speed_scale(speed_scale)
    registry = TargetRegistry(env.unwrapped.device)
    adapter = InstrumentedIKAdapter(env)                      # <-- instrumented IK
    backend = JointBackendConfig(
        mode="joint", grasp_backend="joint_ik", place_backend="joint_ik", drawer_backend="ik_pull",
        adapter=adapter, drawer_env=env, arm_joint_ids=provider._arm_joint_ids, drawer_joint_name="joint_0",
        drawer_open_ik_config=OpenDrawerIKConfig(use_turn_to_face=False, start_from_current=True),
        drawer_close_ik_config=CloseDrawerIKConfig(use_turn_to_face=False, start_from_current=True))
    executor = SkillExecutor(registry, log_path="logs/skill_tests/band_edge_v1.jsonl", backend=backend)
    return BandEdgeHarness(env=env, provider=provider, executor=executor, adapter=adapter,
                           control_dt=control_dt, reg=adv.load_mechanism_registry(),
                           handles=adv.load_paper_handles(), scene=provider.scene, _mod=adv)


def _true_handle_world(provider, spec):
    """TRUE (unbiased) handle world pose from the link pose + spec.handle_local_pos/quat."""
    a = provider.scene[spec.member]
    from adapters.articulated_drawer_v2 import _joint_id  # noqa
    # link index by name
    bnames = list(a.data.body_names)
    li = next((i for i, n in enumerate(bnames) if spec.link_name == n),
              next((i for i, n in enumerate(bnames) if spec.link_name in n), 0))
    link_pos = a.data.body_pos_w[0, li]
    link_quat = a.data.body_quat_w[0, li]
    hp = torch.tensor(spec.handle_local_pos, dtype=torch.float32, device=link_pos.device)
    hq = torch.tensor(spec.handle_local_quat, dtype=torch.float32, device=link_pos.device)
    gp, gq = math_utils.combine_frame_transforms(link_pos.unsqueeze(0), link_quat.unsqueeze(0),
                                                 hp.unsqueeze(0), hq.unsqueeze(0))
    return gp[0], gq[0]


def run_band_edge_episode(H: BandEdgeHarness, spec, *, actual_bias_y, offset_y, g, robot_joint_delta,
                          damping=3.0, max_steps=1800, arm_ids=None):
    """One instrumented band-edge episode. Returns (x, y, instrumentation_dict, damping_eff, damping_post)."""
    from adapters.handle_calibration_bias_v1 import biased_spec
    from adapters.articulated_drawer_v2 import _theta_to_params, _to_action, _phase_durations, _joint_id
    from contracts.episode_schema_v2 import build_x, build_outcome
    from runtime.skill_request import SkillRequest
    from runtime.skill_types import ExecutionStatus, SkillType

    provider, env, executor, adapter = H.provider, H.env, H.executor, H.adapter
    bspec = biased_spec(spec, actual_bias_y)                  # inject actual bias into PERCEIVED handle

    # reset + nuisance (robot joint perturbation) + damping
    H.reset(bspec)
    robot = provider.scene["robot"]
    if robot_joint_delta is not None and arm_ids is not None:
        q = robot.data.default_joint_pos.clone()
        for j, jid in enumerate(arm_ids):
            q[:, jid] = q[:, jid] + float(robot_joint_delta[j])
        robot.write_joint_state_to_sim(q, torch.zeros_like(robot.data.joint_vel))
        for _ in range(4):
            st = provider.get_state()
            env.step(provider.make_hold_joint_action(st, 1.0))
    setinfo = H.set_damping(bspec, damping)

    theta = {"grasp_offset_local_y": float(offset_y), "max_pos_step": 0.02, "pull_lead": 0.08}
    params = _theta_to_params(g, theta, bspec)
    req = SkillRequest(request_id=f"band_edge_{time.time_ns()}", skill_type=SkillType.OPEN_DRAWER,
                       source_object=None, destination_type="drawer", destination_object=spec.drawer_name,
                       parameters=params)

    # instruments
    margins = JointMarginTracker(); ikc = IKFailureCounter(); clamps = ClampCounter()
    snap = CloseSnapshot(); coll = CollisionMonitor(H.scene)
    phase_holder = {"phase": "IDLE"}
    adapter.attach_instruments(ikc, clamps, lambda: phase_holder["phase"])
    lower = [float(v) for v in adapter._joint_lower.tolist()]
    upper = [float(v) for v in adapter._joint_upper.tolist()]

    provider.set_sim_time(0.0)
    state = provider.get_state()
    executor.reset(); executor.start(req, state)
    skill = executor.active_skill
    member_asset = provider.scene[spec.member]
    jid = _joint_id(member_asset, spec.joint_name)
    initial_drawer = float(member_asset.data.joint_pos[0, jid]) if jid is not None else float("nan")
    x = build_x(spec.mechanism_id, spec.drawer_name, spec.member,
                [float(v) for v in state.robot.joint_pos.tolist()],
                [float(v) for v in state.robot.tcp_pose.pos_w.tolist()],
                [float(v) for v in state.robot.tcp_pose.quat_w.tolist()],
                float(getattr(state.robot, "gripper_width", 0.0)), initial_drawer)

    sim_time = 0.0; steps = 0; prev_phase = "IDLE"
    wall0 = time.time()
    while steps < max_steps:
        provider.set_sim_time(sim_time)
        s = provider.get_state()
        skill_ref = executor.active_skill
        with torch.no_grad():
            command = executor.step(s, H.control_dt)
            tel = getattr(skill_ref, "last_telemetry", {}) or {}
            phase = tel.get("skill_state", prev_phase) or prev_phase
            phase_holder["phase"] = phase
            # 5.1 joint margin (7 arm joints)
            arm_q = [float(s.robot.joint_pos[i]) for i in range(7)]
            margins.update(arm_q, lower, upper, phase)
            # 6 collision
            coll.update(phase)
            # 5.5 CLOSE_GRIPPER -> PULL snapshot (single control step at the transition)
            if prev_phase == "CLOSE_GRIPPER" and phase == "PULL" and not snap.captured:
                th_pos, th_quat = _true_handle_world(provider, spec)
                tcp_pos = s.robot.tcp_pose.pos_w
                err3d = float(torch.linalg.norm(tcp_pos - th_pos))
                # signed local-Y of TCP-vs-true-handle in the handle frame
                rel_pos, _ = math_utils.subtract_frame_transforms(
                    th_pos.unsqueeze(0), th_quat.unsqueeze(0), tcp_pos.unsqueeze(0),
                    s.robot.tcp_pose.quat_w.unsqueeze(0))
                snap.capture(err3d=err3d, err_local_y=float(rel_pos[0, 1]),
                             tcp_pose7=[float(v) for v in tcp_pos.tolist()] + [float(v) for v in s.robot.tcp_pose.quat_w.tolist()],
                             true_handle_pose7=[float(v) for v in th_pos.tolist()] + [float(v) for v in th_quat.tolist()],
                             gripper_width=float(getattr(s.robot, "gripper_width", 0.0)),
                             gripper_cmd=float(getattr(command, "gripper_command", 0.0) or 0.0))
            env.step(_to_action(provider, command, s))
        prev_phase = phase
        sim_time += H.control_dt; steps += 1
        if executor.status in (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.STOPPED):
            break

    res = executor.last_result
    o = res.outcomes if res else {}
    final_drawer = o.get("final_drawer_position")
    if final_drawer is None:
        final_drawer = float(member_asset.data.joint_pos[0, jid]) if jid is not None else float("nan")
    target = float(g["target_open_position"])
    y = build_outcome(
        success=bool(res.success) if res else False,
        failure_reason=(res.failure_reason if res else "OTHER"),
        final_joint_position=final_drawer if final_drawer is not None else float("nan"),
        task_outcome_error=abs(final_drawer - target) if final_drawer is not None else float("nan"),
        overshoot=max(0.0, final_drawer - target) if final_drawer is not None else 0.0,
        skill_elapsed_time=o.get("elapsed_time") or (steps * H.control_dt),
        phase_durations=_phase_durations(getattr(getattr(skill, "runtime", None), "history", []) or []),
        handle_detached=bool(o.get("handle_detached")),
        handle_relative_error=o.get("mean_handle_relative_error") or 0.0,
        command_tracking_error=0.0, phase_goal_error=o.get("mean_tcp_tracking_error") or 0.0,
        wall_clock_time=time.time() - wall0)

    fp = failure_phase_from_history(getattr(getattr(skill, "runtime", None), "history", []) or [],
                                    bool(res.success) if res else False)
    instrumentation = {**margins.result(), **ikc.result(), **clamps.result(), **fp, **snap.result(),
                       **coll.result()}
    return x, y, instrumentation, setinfo["effective"], H.read_damping(bspec)
