# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""相机位姿调整面板：点动 wrist(末端) / front(前面) 相机的局部位姿(平移+绕轴旋转)。

直接改 camera prim 的 USD 局部 xform（wrist = FP+VLA 两 prim 同步）。current_offsets() 给保存用。
omni.ui 文本全 ASCII；pxr/omni 延迟 import。
"""

from __future__ import annotations

from . import camera_offsets as co

POS_STEPS = [("1 cm", 0.01), ("2 cm", 0.02), ("5 cm", 0.05)]
ROT_STEPS = [("2 deg", 2.0), ("5 deg", 5.0), ("10 deg", 10.0)]


class CameraPanel:
    def __init__(self):
        import omni.usd

        self.stage = omni.usd.get_context().get_stage()
        # 逻辑相机 -> 它要驱动的 prim 列表 + 当前目标局部位姿(从第一个 prim 读)
        self._groups = {
            "wrist": list(co.WRIST_CAM_PRIMS),
            "front": [co.FRONT_CAM_PRIM],
            "left": [co.LEFT_CAM_PRIM],
            "right": [co.RIGHT_CAM_PRIM],
            "top": [co.TOP_CAM_PRIM],
        }
        self._target = {}   # name -> (pos, quat)
        for name, prims in self._groups.items():
            pose = co.read_local_pose(self.stage, prims[0])
            self._target[name] = pose if pose is not None else ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        self.selected = "wrist"
        self.pos_step = POS_STEPS[0][1]
        self.rot_step = ROT_STEPS[1][1]
        self._build_window()

    def _build_window(self):
        import omni.ui as ui

        self.ui = ui
        self.window = ui.Window("Camera Pose (live)", width=360, height=300)
        self._status = {}
        with self.window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("Select camera")
                self._cam_names = ["wrist", "front", "left", "right", "top"]
                self._cam_model = ui.ComboBox(0, *self._cam_names).model
                self._cam_model.add_item_changed_fn(self._on_cam_changed)
                ui.Label("Position step")
                self._pos_model = ui.ComboBox(0, *[l for l, _ in POS_STEPS]).model
                self._pos_model.add_item_changed_fn(
                    lambda m, i: setattr(self, "pos_step", POS_STEPS[m.get_item_value_model().as_int][1]))
                with ui.HStack(spacing=4, height=24):
                    for ax, sgn, lab in [(0, 1, "+X"), (0, -1, "-X"), (1, 1, "+Y"),
                                         (1, -1, "-Y"), (2, 1, "+Z"), (2, -1, "-Z")]:
                        ui.Button(lab, clicked_fn=lambda a=ax, s=sgn: self._nudge_pos(a, s))
                ui.Label("Rotation step")
                self._rot_model = ui.ComboBox(1, *[l for l, _ in ROT_STEPS]).model
                self._rot_model.add_item_changed_fn(
                    lambda m, i: setattr(self, "rot_step", ROT_STEPS[m.get_item_value_model().as_int][1]))
                with ui.HStack(spacing=4, height=24):
                    for ax, sgn, lab in [(0, 1, "+Rx"), (0, -1, "-Rx"), (1, 1, "+Ry"),
                                         (1, -1, "-Ry"), (2, 1, "+Rz"), (2, -1, "-Rz")]:
                        ui.Button(lab, clicked_fn=lambda a=ax, s=sgn: self._nudge_rot(a, s))
                self._status["sel"] = ui.Label("camera:")
                self._status["pose"] = ui.Label("pose:")

    def _on_cam_changed(self, model, item):
        idx = model.get_item_value_model().as_int
        self.selected = self._cam_names[idx] if 0 <= idx < len(self._cam_names) else "wrist"

    def _apply(self, name):
        pos, quat = self._target[name]
        for p in self._groups[name]:
            co.set_local_pose(self.stage, p, pos, quat)

    def _nudge_pos(self, axis: int, sign: int):
        pos, quat = self._target[self.selected]
        pos = list(pos)
        pos[axis] += sign * self.pos_step
        self._target[self.selected] = (tuple(pos), quat)
        self._apply(self.selected)

    def _nudge_rot(self, axis: int, sign: int):
        from pxr import Gf

        pos, quat = self._target[self.selected]
        ax = [Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1)][axis]
        dq = Gf.Rotation(ax, sign * self.rot_step).GetQuat()      # Gf.Quatd
        cur = Gf.Quatd(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        new = dq * cur
        nq = (new.GetReal(), *[float(v) for v in new.GetImaginary()])
        self._target[self.selected] = (pos, tuple(float(v) for v in nq))
        self._apply(self.selected)

    def current_offsets(self) -> dict:
        """{'wrist':(pos,quat),'front':(pos,quat)}，给保存用。"""
        return {k: v for k, v in self._target.items()}

    def update_status(self):
        if not hasattr(self, "_status"):
            return
        pos, quat = self._target[self.selected]
        self._status["sel"].text = f"camera: {self.selected}"
        self._status["pose"].text = ("pose: pos(%.3f,%.3f,%.3f) quat(%.3f,%.3f,%.3f,%.3f)"
                                     % (pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]))
