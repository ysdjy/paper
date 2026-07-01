"""Low-frequency terminal debug logger for the joint-action entries.

Prints a compact one-line summary every ``every_steps`` control steps (default 30, ~0.5 s at 60 Hz
control) instead of flooding every step. It reports the active skill / backend / target, whether the
command sent to the env is a joint or tcp command, the action stats, and skill-specific diagnostics
(IK errors for grasp/place, drawer/obs info for the learned drawer policy). It also emits a WARNING
if a skill in the joint path emits a non-joint command.
"""

from __future__ import annotations

import math

import torch

DRAWER_DISPLAY = {
    "bottom_drawer": "下抽屉(bottom_drawer)",
    "middle_drawer": "中抽屉(middle_drawer)",
    "top_drawer": "上抽屉(top_drawer)",
}


def _stats(t: torch.Tensor) -> str:
    t = t.detach().float().reshape(-1)
    return f"shape={tuple(t.shape)} min={float(t.min()):.3f} max={float(t.max()):.3f}"


class JointDebugLogger:
    def __init__(self, every_steps: int = 30):
        self.every_steps = max(1, int(every_steps))

    def maybe_log(self, step_count: int, executor, command, state) -> None:
        if step_count % self.every_steps != 0:
            return
        self.log(step_count, executor, command, state)

    def log(self, step_count: int, executor, command, state) -> None:
        active = executor.active_skill
        request = getattr(active, "request", None)
        skill_name = "idle" if request is None else request.skill_type.value
        backend = getattr(active, "backend", None)
        runtime = getattr(active, "runtime", None)

        # target description
        if skill_name == "grasp":
            target = f"object={getattr(request, 'source_object', None)}"
        elif skill_name == "place":
            xyz = getattr(request, "parameters", {}).get("target_surface_xyz") if request else None
            target = f"point={getattr(request, 'destination_object', None)} xyz={xyz}"
        elif skill_name in ("open_drawer", "close_drawer"):
            dest = getattr(request, "destination_object", None)
            target = f"drawer={DRAWER_DISPLAY.get(dest, dest)}"
        else:
            target = "none"

        # output type + action stats
        if command.control_mode == "joint":
            if command.raw_joint_action is not None:
                out = f"OUTPUT=joint(raw_joint_action) {_stats(command.raw_joint_action)}"
            elif command.joint_target is not None:
                out = f"OUTPUT=joint(q_des) {_stats(command.joint_target)}"
            else:
                out = "OUTPUT=joint(hold)"
            out += f" gripper={command.gripper_command:+.1f}"
        else:
            out = "OUTPUT=tcp_pose"
            print(
                f"[JointDebug][WARNING] skill={skill_name} backend={backend} emitted a NON-joint "
                f"command (control_mode={command.control_mode}) in the joint path!",
                flush=True,
            )

        extra = ""
        if skill_name in ("grasp", "place") and command.tcp_pose_w is not None:
            cur = state.robot.tcp_pose
            tgt = command.tcp_pose_w
            pos_err = float(torch.linalg.norm(cur.pos_w - tgt.pos_w))
            ori = getattr(runtime, "final_error_ori", None)
            extra = (
                f" target_tcp={[round(float(v),3) for v in tgt.pos_w.tolist()]}"
                f" current_tcp={[round(float(v),3) for v in cur.pos_w.tolist()]}"
                f" pos_err={pos_err:.4f}"
                f" ori_err_deg={(math.degrees(ori) if ori is not None else float('nan')):.2f}"
                f" ik_success={getattr(active, 'last_ik_success', None)}"
            )
        elif skill_name in ("open_drawer", "close_drawer"):
            cabinet = state.objects.get("cabinet")
            djn = getattr(runtime, "drawer_joint_name", "joint_0")
            djp = None if cabinet is None else cabinet.joint_pos.get(djn)
            obs_shape = getattr(runtime, "obs_shape", None)
            act_shape = getattr(runtime, "action_shape", None)
            extra = (
                f" drawer_joint={djn} drawer_joint_pos={djp}"
                f" drawer_joint_target={command.drawer_joint_target}"
            )
            if backend == "official_joint_policy":
                rel = self._rel_ee_drawer(executor, state)
                extra += (
                    f" rel_ee_drawer(handle-tcp)={rel} obs_shape={list(obs_shape) if obs_shape else None}"
                    f" action_shape={list(act_shape) if act_shape else None}"
                    f" drawer_joint_target_is_None={command.drawer_joint_target is None}"
                )

        print(
            f"[JointDebug] step={step_count} active_skill={skill_name} backend={backend} "
            f"status={executor.status.value} state={getattr(active, 'current_state', 'IDLE')} "
            f"{target} | {out}{extra}",
            flush=True,
        )

    def _rel_ee_drawer(self, executor, state):
        adapter = getattr(executor.backend, "drawer_obs_adapter", None)
        if adapter is None:
            return None
        try:
            handle = adapter._handle_pos_w()[adapter.env_id]
            tcp = state.robot.tcp_pose.pos_w
            rel = (handle - tcp).detach().cpu().tolist()
            return [round(float(v), 3) for v in rel]
        except Exception:
            return None
