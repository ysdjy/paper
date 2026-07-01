# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""UI#1 —— Franka 控制面板（控制算法在此）。用于测可达性。

任务空间(点动 ±X/±Y/±Z + 绕工具轴旋转 / 绝对 x/y/z+rpy+Go，都走 DLS IK)、关节空间(7 滑块)、
回 home(末端垂直向下)、夹爪开/合。按钮回调只改内部状态，主循环每帧调 `compute(state)` 产出
`ControlIntent`(绝对 q_des + gripper) 或 None(无手动指令)。

顶层 import torch/isaac，**只能在 AppLauncher 之后被 import**（由 test_mode_ui entry 负责）。
omni.ui 文本全 ASCII。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import isaaclab.utils.math as math_utils
from runtime.scene_state_provider import PoseState

HOME_Q = (0.028, -0.330, -0.092, -2.223, -0.030, 1.912, 0.734)   # 末端竖直朝下 + 比原 home 抬高 ~15cm
JOG_STEPS = [("1 cm", 0.01), ("2 cm", 0.02), ("5 cm", 0.05)]
ROT_STEPS = [("5 deg", 5.0), ("10 deg", 10.0), ("15 deg", 15.0)]
POS_TOL = 2.0e-3        # m，IK 到位判据
ORI_TOL = 2.0e-2        # rad
IK_TIMEOUT_FRAMES = 600  # 单个目标超时未收敛则放弃
JOINT_MAX_STEP = 0.20   # rad，关节/home 软逼近步长


@dataclass
class ControlIntent:
    q_des: torch.Tensor   # 7 绝对关节弧度
    gripper: float        # +1 open / -1 close
    source: str           # "ik" | "joint" | "home"


class FrankaControlPanel:
    def __init__(self, env, provider, adapter):
        self.env = env
        self.provider = provider
        self.adapter = adapter
        self.robot = env.unwrapped.scene["robot"]
        self.device = self.robot.device
        self.env_id = getattr(provider, "env_id", 0)
        # home 关节配置：优先读保存的(saved_scenes/v1_active/robot_home.json)，没有才用默认 HOME_Q。
        from . import robot_home as _rh
        self._robot_home = _rh
        self._home_q = torch.tensor(_rh.load_home_q(HOME_Q), dtype=torch.float32, device=self.device)
        # 关节滑块范围 = adapter 已解析的软限位
        self._j_lo = adapter._joint_lower.detach().cpu().tolist()
        self._j_hi = adapter._joint_upper.detach().cpu().tolist()

        # 控制状态
        self._mode = "idle"                 # idle / ik / joint / home
        self._target_pose: PoseState | None = None
        self._target_q: torch.Tensor | None = None
        self._gripper = 1.0                 # latched，默认开
        self._jog_step = JOG_STEPS[0][1]
        self._rot_step = ROT_STEPS[0][1]
        self._last_tcp: PoseState | None = None
        self._ik_frames = 0
        self._pos_err = float("nan")
        self._ori_err = float("nan")
        self._owned_by_skill = False

        # 实时同步：_auto_sync 勾选时每帧把字段/滑块刷成真实状态；用户一开始编辑字段就 _editing=True
        # 暂停同步(避免输入被覆盖)，点 Go/Apply/Home 或手动 Sync 后恢复。_suppress_sync 用于在程序化
        # set_value 时不把"被同步刷新"误判成用户编辑。
        self._auto_sync = True
        self._editing = False
        self._suppress_sync = False

        self._build_window()

    # ------------------------------------------------------------------ UI
    def _build_window(self):
        import omni.ui as ui

        self.ui = ui
        self.window = ui.Window("Franka Control (TEST)", width=440, height=640)
        self._abs_fields = {}
        self._joint_sliders = []
        self._status = {}
        with self.window.frame:
            with ui.VStack(spacing=6, height=0):
                # 实时同步：勾选时每帧把关节滑块 + 任务空间字段刷成机器人【当前真实状态】，
                # 方便基于真值修改；要手填一个目标值时取消勾选(字段就不再被覆盖)，填好再 Go/Apply。
                with ui.HStack(spacing=6, height=22):
                    self._autosync_cb = ui.CheckBox(width=20)
                    self._autosync_cb.model.set_value(True)
                    self._autosync_cb.model.add_value_changed_fn(self._on_autosync_toggle)
                    ui.Label("Auto-sync fields/sliders <- live state (pauses while you edit)")
                ui.Label("Task-space jog (DLS IK)")
                self._jog_step_model = ui.ComboBox(0, *[l for l, _ in JOG_STEPS]).model
                self._jog_step_model.add_item_changed_fn(
                    lambda m, i: setattr(self, "_jog_step", JOG_STEPS[m.get_item_value_model().as_int][1]))
                with ui.HStack(spacing=4, height=24):
                    for ax, sgn, lab in [(0, 1, "+X"), (0, -1, "-X"), (1, 1, "+Y"),
                                         (1, -1, "-Y"), (2, 1, "+Z"), (2, -1, "-Z")]:
                        ui.Button(lab, clicked_fn=lambda a=ax, s=sgn: self._jog_pos(a, s))
                self._rot_step_model = ui.ComboBox(0, *[l for l, _ in ROT_STEPS]).model
                self._rot_step_model.add_item_changed_fn(
                    lambda m, i: setattr(self, "_rot_step", ROT_STEPS[m.get_item_value_model().as_int][1]))
                with ui.HStack(spacing=4, height=24):
                    for ax, sgn, lab in [(0, 1, "+Rx"), (0, -1, "-Rx"), (1, 1, "+Ry"),
                                         (1, -1, "-Ry"), (2, 1, "+Rz"), (2, -1, "-Rz")]:
                        ui.Button(lab, clicked_fn=lambda a=ax, s=sgn: self._jog_rot(a, s))
                ui.Separator()
                ui.Label("Absolute pose (base frame; xyz m, rpy deg)")
                for name in ("x", "y", "z", "roll", "pitch", "yaw"):
                    with ui.HStack(spacing=6, height=22):
                        ui.Label(name, width=60)
                        self._abs_fields[name] = ui.FloatField().model
                        self._abs_fields[name].add_value_changed_fn(lambda m: self._on_field_edit())
                with ui.HStack(spacing=6, height=24):
                    ui.Button("Go (abs)", clicked_fn=self._go_abs)
                    ui.Button("Sync fields <- current TCP", clicked_fn=self._resume_and_sync_abs)
                ui.Separator()
                ui.Label("Joint space (rad)")
                for i in range(7):
                    with ui.HStack(spacing=6, height=22):
                        ui.Label(f"J{i+1}", width=40)
                        s = ui.FloatSlider(min=float(self._j_lo[i]), max=float(self._j_hi[i]))
                        s.model.add_value_changed_fn(lambda m: self._on_field_edit())
                        self._joint_sliders.append(s.model)
                with ui.HStack(spacing=6, height=24):
                    ui.Button("Apply joints", clicked_fn=self._apply_joints)
                    ui.Button("Sync sliders <- current", clicked_fn=self._resume_and_sync_joints)
                ui.Separator()
                with ui.HStack(spacing=6, height=24):
                    ui.Button("Home", clicked_fn=self._go_home)
                    # 把机器人现在的关节配置设为新的 home(并存盘)：先用滑块/IK/jog 把臂摆到想要的 home
                    # 姿态，再点这个；之后 Home 按钮 + 抽屉/门技能都回到这个新 home。
                    ui.Button("Set Home = current (save)", clicked_fn=self._set_home_current)
                self._status["home"] = ui.Label("home: (default)")
                with ui.HStack(spacing=6, height=24):
                    ui.Button("Gripper Open", clicked_fn=lambda: self._set_gripper(1.0))
                    ui.Button("Gripper Close", clicked_fn=lambda: self._set_gripper(-1.0))
                ui.Separator()
                for key in ("mode", "source", "pos_err", "ori_err_deg", "gripper", "owner"):
                    self._status[key] = ui.Label(f"{key}:")

    # ------------------------------------------------------------------ button callbacks (set state)
    def _root_pose(self):
        return self.robot.data.root_pos_w[self.env_id], self.robot.data.root_quat_w[self.env_id]

    def _jog_pos(self, axis: int, sign: int):
        if self._owned_by_skill or self._last_tcp is None:
            return
        pos = self._last_tcp.pos_w.clone()
        pos[axis] = pos[axis] + sign * self._jog_step
        self._set_ik_target(PoseState(pos, self._last_tcp.quat_w.clone()))

    def _jog_rot(self, axis: int, sign: int):
        if self._owned_by_skill or self._last_tcp is None:
            return
        ang = torch.zeros(3, device=self.device)
        ang[axis] = sign * self._rot_step * torch.pi / 180.0
        dq = math_utils.quat_from_euler_xyz(ang[0:1], ang[1:2], ang[2:3])[0]
        # tool-axis rotation: q_new = q_current * dq
        q_new = math_utils.quat_mul(self._last_tcp.quat_w.unsqueeze(0), dq.unsqueeze(0))[0]
        self._set_ik_target(PoseState(self._last_tcp.pos_w.clone(), q_new))

    def _go_abs(self):
        if self._owned_by_skill:
            return
        f = self._abs_fields
        pos_b = torch.tensor([f["x"].get_value_as_float(), f["y"].get_value_as_float(),
                              f["z"].get_value_as_float()], dtype=torch.float32, device=self.device)
        rpy = [f["roll"].get_value_as_float(), f["pitch"].get_value_as_float(), f["yaw"].get_value_as_float()]
        r = torch.tensor([rpy[0]], device=self.device) * torch.pi / 180.0
        p = torch.tensor([rpy[1]], device=self.device) * torch.pi / 180.0
        y = torch.tensor([rpy[2]], device=self.device) * torch.pi / 180.0
        quat_b = math_utils.quat_from_euler_xyz(r, p, y)[0]
        root_pos, root_quat = self._root_pose()
        pos_w, quat_w = math_utils.combine_frame_transforms(
            root_pos.unsqueeze(0), root_quat.unsqueeze(0), pos_b.unsqueeze(0), quat_b.unsqueeze(0))
        self._set_ik_target(PoseState(pos_w[0], quat_w[0]))
        self._editing = False   # 已执行 -> 恢复实时同步

    def _sync_abs_from_tcp(self):
        if self._last_tcp is None:
            return
        root_pos, root_quat = self._root_pose()
        pos_b, quat_b = math_utils.subtract_frame_transforms(
            root_pos.unsqueeze(0), root_quat.unsqueeze(0),
            self._last_tcp.pos_w.unsqueeze(0), self._last_tcp.quat_w.unsqueeze(0))
        p = pos_b[0].detach().cpu().tolist()
        self._suppress_sync = True   # 程序化刷新，别触发 _on_field_edit 误判成用户编辑
        try:
            self._abs_fields["x"].set_value(float(p[0]))
            self._abs_fields["y"].set_value(float(p[1]))
            self._abs_fields["z"].set_value(float(p[2]))
            r, pi, ya = math_utils.euler_xyz_from_quat(quat_b)
            self._abs_fields["roll"].set_value(float(r[0]) * 180.0 / 3.14159265)
            self._abs_fields["pitch"].set_value(float(pi[0]) * 180.0 / 3.14159265)
            self._abs_fields["yaw"].set_value(float(ya[0]) * 180.0 / 3.14159265)
        except Exception:
            pass
        finally:
            self._suppress_sync = False

    def _apply_joints(self):
        if self._owned_by_skill:
            return
        q = torch.tensor([m.get_value_as_float() for m in self._joint_sliders],
                         dtype=torch.float32, device=self.device)
        self._target_q = q
        self._mode = "joint"
        self._editing = False   # 已执行 -> 恢复实时同步

    def _sync_joints_from_current(self):
        q = self.provider.arm_joint_pos(self.provider.get_state()).detach().cpu().tolist()
        self._suppress_sync = True
        try:
            for i, m in enumerate(self._joint_sliders):
                m.set_value(float(q[i]))
        finally:
            self._suppress_sync = False

    def _go_home(self):
        if self._owned_by_skill:
            return
        self._target_q = self._home_q.clone()
        self._mode = "home"
        self._editing = False   # 已执行 -> 恢复实时同步

    def _set_home_current(self):
        """把机器人【当前】手臂关节配置设为新的 home，并存盘(robot_home.json)。
        Home 按钮立即生效；抽屉/门技能下次启动从同一文件读到这个 home。"""
        try:
            q = self.provider.arm_joint_pos(self.provider.get_state()).detach().reshape(-1)[:7]
            self._home_q = q.clone().to(self.device)
            ok = self._robot_home.save_home_q(q.detach().cpu().tolist())
            ql = [round(float(v), 3) for v in q.detach().cpu().tolist()]
            if "home" in getattr(self, "_status", {}):
                self._status["home"].text = f"home {'saved' if ok else 'SET (save FAILED)'}: {ql}"
            print(f"[FrankaControlPanel] home set to current joints {ql} (saved={ok})", flush=True)
        except Exception as exc:  # pragma: no cover
            print(f"[FrankaControlPanel] set home failed: {exc}", flush=True)

    # --- 实时同步辅助 ---
    def _on_field_edit(self):
        """用户手动改了某个字段/滑块 -> 暂停自动同步，免得输入被每帧刷新覆盖。"""
        if not self._suppress_sync:
            self._editing = True

    def _on_autosync_toggle(self, m):
        self._auto_sync = bool(m.get_value_as_bool())
        self._editing = False   # 重新勾选时恢复同步

    def _resume_and_sync_abs(self):
        self._editing = False
        self._sync_abs_from_tcp()

    def _resume_and_sync_joints(self):
        self._editing = False
        self._sync_joints_from_current()

    def set_ik_target(self, pose: PoseState):
        """供外部面板(如 Grasp Pose)请求机器人运动到一个世界位姿(走 UI#1 的 IK 执行路径)。"""
        if not self._owned_by_skill:
            self._set_ik_target(pose)
            self._editing = False

    def _set_gripper(self, g: float):
        self._gripper = float(g)

    def _set_ik_target(self, pose: PoseState):
        self._target_pose = pose
        self._mode = "ik"
        self._ik_frames = 0

    # ------------------------------------------------------------------ per-frame
    def compute(self, state) -> ControlIntent | None:
        self._last_tcp = PoseState(state.robot.tcp_pose.pos_w.clone(), state.robot.tcp_pose.quat_w.clone())
        if self._owned_by_skill:
            return None

        if self._mode == "ik" and self._target_pose is not None:
            res = self.adapter.solve(self._target_pose)
            self._ik_frames += 1
            if not res.success:
                self._mode = "idle"
                return None
            self._pos_err, self._ori_err = res.position_error, res.orientation_error
            if (res.position_error < POS_TOL and res.orientation_error < ORI_TOL) or self._ik_frames > IK_TIMEOUT_FRAMES:
                self._mode = "idle"
            return ControlIntent(res.q_des, self._gripper, "ik")

        if self._mode in ("joint", "home") and self._target_q is not None:
            q_cur = self.provider.arm_joint_pos(state).to(self.device)
            delta = torch.clamp(self._target_q - q_cur, -JOINT_MAX_STEP, JOINT_MAX_STEP)
            q_next = q_cur + delta
            if torch.max(torch.abs(self._target_q - q_cur)) < 1.0e-3:
                self._mode = "idle"
            return ControlIntent(q_next, self._gripper, self._mode)

        return None

    # ------------------------------------------------------------------ arbitration hooks / status
    def notify_skill_owns_robot(self):
        if not self._owned_by_skill:
            self._owned_by_skill = True
            self._mode = "idle"
            self._target_pose = None
            self._target_q = None

    def notify_skill_released(self):
        self._owned_by_skill = False
        self._mode = "idle"
        self._target_pose = None
        self._target_q = None

    def current_gripper(self) -> float:
        return self._gripper

    def active_target_pose(self) -> PoseState | None:
        return self._target_pose if self._mode == "ik" else None

    def update_status(self):
        if not hasattr(self, "_status"):
            return
        # 实时同步：刷字段/滑块成真实状态；用户正在编辑(_editing)或取消勾选时暂停，避免覆盖输入。
        if getattr(self, "_auto_sync", True) and not getattr(self, "_editing", False):
            try:
                self._sync_abs_from_tcp()
                self._sync_joints_from_current()
            except Exception:
                pass
        vals = {
            "mode": self._mode,
            "source": "skill" if self._owned_by_skill else "ui1",
            "pos_err": f"{self._pos_err:.4f}" if self._pos_err == self._pos_err else "n/a",
            "ori_err_deg": f"{self._ori_err * 180.0 / 3.14159265:.2f}" if self._ori_err == self._ori_err else "n/a",
            "gripper": "open" if self._gripper > 0 else "close",
            "owner": "SKILL (UI#2)" if self._owned_by_skill else "UI#1 manual",
        }
        for k, lab in self._status.items():
            if k in vals:   # "home" 标签由 _set_home_current 单独更新，这里跳过(否则 KeyError)
                lab.text = f"{k}: {vals[k]}"
