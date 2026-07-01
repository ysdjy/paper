# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""GELLO -> test_mode controller 适配器。

把 GELLO 遥操作模块(gello_isaac_teleop.GelloFrankaTeleop)包成 test_mode_ui 的 --controller，
这样可以在【最新测试场景】(冰箱替换 + 碰撞细化 + 目标可视化那一套)里直接用 GELLO 遥操作 Franka，
而不是 run_gello_teleop_demo.py 的纯净 --task 场景。

用法(串口权限用 sg dialout 包装；shell 已在 dialout 组可去掉)::

    sg dialout -c './isaaclab.sh -p projects/franka_v1_skill_lab/scene_interface/test_mode_ui.py \\
        --controller franka_v1_skill_lab.scene_interface.gello_controller:GelloTeleopController'

test_mode_ui 默认就加载最新保存场景(load_latest_scene = not --base_scene)并替换冰箱，所以上面这条
就是"在最新测试场景里 GELLO 遥操作"。可选环境变量::

    GELLO_CONFIG=<yaml>          覆盖默认 gello_franka.yaml
    GELLO_NO_GRIPPER=1           关掉夹爪控制(保持张开)
    GELLO_MAX_JOINT_VEL=2.5      跟随速度上限(rad/s，嫌慢调大)
    GELLO_SMOOTHING_TAU=0.08     平滑时间常数(s，嫌滞后调小)

controller 契约(test_mode_ui 调用)：__init__(session) / build_window() / on_reset() /
step(session)->action。step 返回 GELLO 合成的整段 env 动作(关节目标 + 夹爪)驱动机器人；
GELLO 没起来时返回 None，交还给 UI/hold，场景照常跑。

本模块顶层不 import isaac/gello(保持可被 test_mode 安全 import)；全部延迟到 __init__(launch 之后)。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _envflag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


class GelloTeleopController:
    """把 GelloFrankaTeleop 接进 test_mode 的外部 controller。"""

    def __init__(self, session):
        self.session = session
        self.env = session.env
        self.teleop = None
        self._win = None
        self._status_label = None

        # 让 gello_isaac_teleop 可 import：projects/gello_franka_teleop 上 sys.path。
        # __file__ = .../projects/franka_v1_skill_lab/scene_interface/gello_controller.py
        # parents[2] = .../projects
        projects_dir = Path(__file__).resolve().parents[2]
        gello_root = projects_dir / "gello_franka_teleop"
        if str(gello_root) not in sys.path:
            sys.path.insert(0, str(gello_root))
        cfg_path = os.environ.get("GELLO_CONFIG", str(gello_root / "configs" / "gello_franka.yaml"))

        try:
            from gello_isaac_teleop import GelloFrankaTeleop, GelloTeleopConfig

            self.teleop = GelloFrankaTeleop(
                self.env,
                GelloTeleopConfig(
                    gello_config=cfg_path,
                    enable_gripper=not _envflag("GELLO_NO_GRIPPER"),
                    max_joint_vel=float(os.environ.get("GELLO_MAX_JOINT_VEL", "2.5")),
                    smoothing_tau=float(os.environ.get("GELLO_SMOOTHING_TAU", "0.08")),
                ),
            )
            self.teleop.start()   # 打开设备，q_cmd 以机器人当前位姿起步(无突跳)
            print(f"[gello] teleop started (config={cfg_path}); GELLO drives the Franka in the test scene.", flush=True)
        except Exception as exc:  # pragma: no cover - hardware/serial dependent
            print(f"[gello] FAILED to start teleop: {exc}", flush=True)
            print("[gello] hints: run via  sg dialout -c '...'  for serial permission; "
                  "check 'port' in gello_franka.yaml; ensure GELLO is connected.", flush=True)
            self.teleop = None

    # ----------------------------------------------------------- controller hooks
    def build_window(self):
        if self.teleop is None:
            return
        try:
            import omni.ui as ui

            self._win = ui.Window("GELLO Teleop", width=340, height=120)
            with self._win.frame:
                with ui.VStack(spacing=6, height=0):
                    ui.Label("GELLO is driving the Franka (test scene).")
                    self._status_label = ui.Label("status: starting...")
        except Exception as exc:  # pragma: no cover
            print(f"[gello] build_window failed: {exc}", flush=True)

    def on_reset(self):
        # env.reset 后机器人回到初始位姿；把 q_cmd 重新对齐到新位姿，避免下一步突跳。
        if self.teleop is not None:
            try:
                self.teleop.reseat()
            except Exception as exc:  # pragma: no cover
                print(f"[gello] reseat failed: {exc}", flush=True)

    def step(self, session):
        if self.teleop is None:
            return None
        try:
            action = self.teleop.step()
        except Exception as exc:  # pragma: no cover
            print(f"[gello] step failed: {exc}", flush=True)
            return None
        self._update_status()
        return action

    # ----------------------------------------------------------- internals
    def _update_status(self):
        if self._status_label is None:
            return
        try:
            t = self.teleop.telemetry()
            self._status_label.text = (
                f"loop {float(t.get('real_loop_hz', 0)):.0f}Hz  "
                f"read {float(t.get('read_hz', 0)):.0f}Hz  "
                f"gripper {t.get('gripper_state', '?')}"
            )
        except Exception:
            pass
