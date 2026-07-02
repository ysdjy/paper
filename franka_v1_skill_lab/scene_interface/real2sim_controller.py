# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Real Franka -> test_mode 仿真 Franka 跟随 controller 适配器。

把 real_franka_to_sim_minimal 的 ZMQ 状态流(真机 q)接进 test_mode_ui 的 --controller，这样可以在
【最新 V1 测试场景】(冰箱替换 + 碰撞细化 + 目标可视化那套)里让真机驱动仿真 Franka——而不是
run_real2sim_latest_scene.py 加载的 SceneLayoutModule 旧场景。

严格单向 Real->Sim：本 controller 只【订阅】真机状态，绝不连真机、绝不回发命令、不控真实夹爪。

夹爪：真机端不流式发布夹爪(README 已说明)，所以仿真夹爪在这里单独控——走 env 动作的夹爪维度
(和技能抓取完全同一套机制，所以一定能动)，由 "Real2Sim Follow" 窗口的 Open/Close 按钮切换，
不依赖会失焦的键盘。默认张开。

用法(先开真机 streamer 或 fake_stream 发布到 5555)::

    ./isaaclab.sh -p projects/franka_v1_skill_lab/scene_interface/test_mode_ui.py \\
        --controller franka_v1_skill_lab.scene_interface.real2sim_controller:Real2SimController

可选环境变量::

    REAL2SIM_ZMQ=tcp://127.0.0.1:5555   订阅端点(远程发布端换成其 IP)
    REAL2SIM_TIMEOUT=0.2                 断流阈值(s)：超过则暂停跟随(返回 None -> hold)

controller 契约：__init__(session)/build_window()/on_reset()/step(session)->action。
顶层不 import isaac/torch/zmq；全部延迟到 __init__/step(launch 之后)。
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque


class _ZmqStateSubscriber:
    """后台线程订阅真机状态流，维护最新 q(7 臂关节)。schema 同 zero_torque_state_streamer。"""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self._latest_q = None
        self._latest_seq = None
        self._last_recv = None
        self._recv_times = deque(maxlen=200)
        self._lock = threading.Lock()
        self._stop = False
        self._thread = None

    def start(self):
        import zmq

        self._zmq = zmq
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        zmq = self._zmq
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.connect(self.endpoint)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.setsockopt(zmq.RCVTIMEO, 200)   # ms；让线程能定期检查 _stop
        while not self._stop:
            try:
                raw = sock.recv()
            except Exception:
                continue
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            q = msg.get("q", None)
            if isinstance(q, (list, tuple)) and len(q) >= 7:
                now = time.monotonic()
                with self._lock:
                    self._latest_q = [float(v) for v in q[:7]]
                    self._latest_seq = msg.get("sequence")
                    self._last_recv = now
                    self._recv_times.append(now)

    def latest(self, timeout: float):
        """返回 (q_or_None, seq, age_s)。q 太旧(>timeout)时返回 None。"""
        with self._lock:
            if self._last_recv is None:
                return None, None, None
            age = time.monotonic() - self._last_recv
            q = self._latest_q if age <= timeout else None
            return q, self._latest_seq, age

    def recv_hz(self) -> int:
        with self._lock:
            now = time.monotonic()
            return sum(1 for t in self._recv_times if now - t <= 1.0)

    def stop(self):
        self._stop = True


class Real2SimController:
    """真机状态流 -> 仿真 Franka 跟随(在 test_mode 的 V1 测试场景里)。"""

    def __init__(self, session):
        self.session = session
        self.provider = session.provider
        self.endpoint = os.environ.get("REAL2SIM_ZMQ", "tcp://127.0.0.1:5555")
        self.timeout = float(os.environ.get("REAL2SIM_TIMEOUT", "0.2"))
        self._gripper = 1.0   # +1=open, -1=close（仿真侧，真机夹爪不流式）
        self._win = None
        self._status_label = None
        self._sub = None

        import torch

        self._torch = torch
        try:
            self._sub = _ZmqStateSubscriber(self.endpoint).start()
            print(f"[real2sim] subscribing real-robot state at {self.endpoint}; sim Franka will follow.", flush=True)
            print("[real2sim] one-way Real->Sim: never connects/commands the real robot.", flush=True)
        except Exception as exc:  # pragma: no cover - depends on pyzmq/runtime
            print(f"[real2sim] FAILED to subscribe ({exc}); is pyzmq installed in env_isaaclab?", flush=True)
            self._sub = None

    # ----------------------------------------------------------- controller hooks
    def build_window(self):
        try:
            import omni.ui as ui

            self._win = ui.Window("Real2Sim Follow", width=360, height=150)
            with self._win.frame:
                with ui.VStack(spacing=6, height=0):
                    ui.Label("Sim Franka follows the real robot (hand-drag the real arm).")
                    with ui.HStack(spacing=8, height=28):
                        ui.Button("Gripper Open", clicked_fn=lambda: self._set_gripper(1.0))
                        ui.Button("Gripper Close", clicked_fn=lambda: self._set_gripper(-1.0))
                    self._status_label = ui.Label("status: waiting for stream...")
        except Exception as exc:  # pragma: no cover
            print(f"[real2sim] build_window failed: {exc}", flush=True)

    def on_reset(self):
        # 纯跟随，无内部状态需要重置。
        pass

    def step(self, session):
        if self._sub is None:
            return None
        q, seq, age = self._sub.latest(self.timeout)
        self._update_status(q, seq, age)
        if q is None:
            return None   # 无数据/断流 -> 交回 hold，不用旧 q 继续驱动
        q_t = self._torch.tensor(q, dtype=self._torch.float32, device=self.provider.device)
        return self.provider.make_joint_action_from_q_des(q_t, self._gripper)

    # ----------------------------------------------------------- internals
    def _set_gripper(self, v: float):
        self._gripper = float(v)

    def _update_status(self, q, seq, age):
        if self._status_label is None:
            return
        try:
            hz = self._sub.recv_hz()
            g = "open" if self._gripper > 0 else "close"
            if q is None:
                self._status_label.text = f"status: NO/STALE stream  recv {hz}Hz  gripper {g}"
            else:
                self._status_label.text = (
                    f"following seq={seq}  recv {hz}Hz  age {float(age) * 1000:.0f}ms  gripper {g}"
                )
        except Exception:
            pass
