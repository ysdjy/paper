# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Skill Waypoints 面板：给每个【动作技能】编辑一串【任务空间途径点】(世界系绝对 pose + 绕行半径)。

用途：技能执行(尤其在技能之间过渡)时，机器人先按顺序【绕行】这些途径点再去做接触动作，用来避开
已经打开的抽屉等障碍(用户手放途径点 = 手动指定安全路径)。与 grasp_pose_panel 平级、结构一致：

  * 选技能(skill_id，如 open_drawer:sektion_top_drawer) -> 选该技能的第几个途径点 -> 在下面点动它的
    pose(XYZ / 旋转)、拖它的【绕行半径】滑条；"Add @TCP"把机器人当前 TCP 位姿抓成一个新途径点。
  * 每帧把选中技能的途径点用坐标轴 marker 画出来(选中的更大)，半径越大画得越大。
  * Save 写到 saved_scenes/v1_active/skill_waypoints.json；状态机 controller 每次起技能前读它、
    用 _WaypointMover 依次绕行(绕行半径=到点判定容差=不必精确到达=blend)。

pose 存【世界系绝对】值(用户选的坐标系)。torch/omni 延迟 import(post-launch)。
"""

from __future__ import annotations

import json
from pathlib import Path

WAYPOINTS_FILE_NAME = "skill_waypoints.json"

# 可编辑途径点的动作技能(与状态机 controller 的 skill_id 约定一致)。
SKILL_IDS = [
    "open_drawer:middle_drawer", "close_drawer:middle_drawer",
    "open_drawer:top_drawer", "close_drawer:top_drawer",
    "open_drawer:bottom_drawer", "close_drawer:bottom_drawer",
    "open_drawer:sektion_top_drawer", "close_drawer:sektion_top_drawer",
    "open_drawer:sektion_bottom_drawer", "close_drawer:sektion_bottom_drawer",
    "operate_coffee",
]

_DEFAULT_RADIUS = 0.08          # 默认绕行半径(m)：到点容差，越大=离途径点越远就切下一段(blend 越圆)
_POS_STEPS = [0.005, 0.01, 0.02, 0.05, 0.1]
_ROT_STEPS_DEG = [1.0, 5.0, 10.0, 30.0, 90.0]


class SkillWaypointsPanel:
    def __init__(self, env, provider, headless: bool = False):
        self.env = env
        self.provider = provider
        self.scene = provider.scene
        self.env_id = 0
        self.headless = headless
        self.pos_step = 0.02
        self.rot_step_deg = 5.0

        # data: {skill_id: [ {"pos":[x,y,z], "quat":[w,x,y,z], "radius": r}, ... ]}
        self.waypoints: dict[str, list] = self._load()
        for sid in SKILL_IDS:
            self.waypoints.setdefault(sid, [])
        self.sel_skill = SKILL_IDS[0]
        self.sel_idx = 0
        self.last_saved = "-"

        try:
            from .target_markers import TargetPoseMarkers
            self._markers = TargetPoseMarkers(enabled=not headless, prefix="skillwp")
        except Exception:
            self._markers = None
        self._models = {}
        self._status = {}
        if not headless:
            self._build_window()

    # ------------------------------------------------------------------ storage
    def _path(self) -> Path:
        from franka_v1_skill_lab.scene.scene_registry import V1_ACTIVE_DIR
        return Path(V1_ACTIVE_DIR) / WAYPOINTS_FILE_NAME

    def _load(self) -> dict:
        try:
            p = self._path()
            if p.is_file():
                d = json.loads(p.read_text(encoding="utf-8"))
                out = {}
                for sid, lst in (d.get("skills") or {}).items():
                    items = []
                    for w in lst:
                        e = {"kind": w.get("kind", "task"),
                             "pos": [float(x) for x in w.get("pos", [0.0, 0.0, 0.0])],
                             "quat": [float(x) for x in w.get("quat", [0.0, 1.0, 0.0, 0.0])],
                             "radius": float(w.get("radius", _DEFAULT_RADIUS))}
                        if w.get("joint_pos"):
                            e["joint_pos"] = [float(x) for x in w["joint_pos"]]
                        items.append(e)
                    out[sid] = items
                return out
        except Exception as exc:  # pragma: no cover
            print(f"[skill_wp] load failed: {exc}", flush=True)
        return {}

    def save(self):
        try:
            p = self._path()
            data = {"frame": "world_absolute", "quat_order": "wxyz",
                    "note": "Per-skill transit via-points; robot rounds each within its radius (blend).",
                    "skills": {sid: lst for sid, lst in self.waypoints.items() if lst}}
            p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.last_saved = p.name
            print(f"[skill_wp] saved {sum(len(v) for v in data['skills'].values())} waypoints -> {p}", flush=True)
            self._refresh_status()
        except Exception as exc:  # pragma: no cover
            print(f"[skill_wp] save failed: {exc}", flush=True)

    # ------------------------------------------------------------------ helpers
    def _cur(self) -> list:
        return self.waypoints.setdefault(self.sel_skill, [])

    def _tcp_pose(self):
        st = self.provider.get_state()
        p = st.robot.tcp_pose.pos_w.reshape(3).tolist()
        q = st.robot.tcp_pose.quat_w.reshape(4).tolist()
        return [float(v) for v in p], [float(v) for v in q]

    def _add_at_tcp(self):
        pos, quat = self._tcp_pose()
        self._cur().append({"kind": "task", "pos": pos, "quat": quat, "radius": _DEFAULT_RADIUS})
        self.sel_idx = len(self._cur()) - 1
        self._rebuild_wp_combo()
        self._refresh_status()
        print(f"[skill_wp] {self.sel_skill}: added TASK waypoint #{self.sel_idx} @ TCP "
              f"({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) r={_DEFAULT_RADIUS}", flush=True)

    def _add_at_joints(self):
        """把机器人【当前关节坐标】存成一个【关节空间途径点】(严格到达,无半径)。用于必须靠转底座才能到的
        姿态(如'正对把手'),任务空间点做不到底座回转,故用关节坐标严格复现。pos/quat 仅供可视化。"""
        st = self.provider.get_state()
        try:
            jp = [float(v) for v in st.robot.joint_pos[self.provider._arm_joint_ids].reshape(-1).tolist()]
        except Exception as exc:
            print(f"[skill_wp] read arm joints failed: {exc}", flush=True)
            return
        p = [float(v) for v in st.robot.tcp_pose.pos_w.reshape(3).tolist()]
        q = [float(v) for v in st.robot.tcp_pose.quat_w.reshape(4).tolist()]
        self._cur().append({"kind": "joint", "joint_pos": jp, "pos": p, "quat": q, "radius": 0.0})
        self.sel_idx = len(self._cur()) - 1
        self._rebuild_wp_combo()
        self._refresh_status()
        print(f"[skill_wp] {self.sel_skill}: added JOINT waypoint #{self.sel_idx} (strict) "
              f"j=[{', '.join(f'{v:.2f}' for v in jp)}]", flush=True)

    def _delete(self):
        lst = self._cur()
        if 0 <= self.sel_idx < len(lst):
            lst.pop(self.sel_idx)
            self.sel_idx = max(0, min(self.sel_idx, len(lst) - 1))
            self._rebuild_wp_combo()
            self._refresh_status()

    def _sel_wp(self):
        lst = self._cur()
        if 0 <= self.sel_idx < len(lst):
            return lst[self.sel_idx]
        return None

    def _to_task_if_joint(self, w):
        """在笛卡尔空间编辑(点动位置/旋转/半径)一个 joint 途径点时,把它转成 task 途径点——因为你是在按坐标
        摆它,就按你摆的走。注意:转成 task 后【不再有底座回转】(joint 点才严格复现转身),转身角大的柜子需保留
        joint 点(用 Add @ joints 重设);正前方小转身的橱柜用 task 点即可。"""
        if w.get("kind") == "joint":
            w["kind"] = "task"
            w.pop("joint_pos", None)
            if float(w.get("radius", 0.0)) <= 0.0:
                w["radius"] = _DEFAULT_RADIUS
            print("[skill_wp] JOINT waypoint -> converted to task (edited in Cartesian; base-turn no longer "
                  "forced -- use 'Add @ joints' if you need the base rotation)", flush=True)

    def _nudge(self, axis: int, sign: int):
        w = self._sel_wp()
        if w is not None:
            self._to_task_if_joint(w)
            w["pos"][axis] += sign * self.pos_step
            self._refresh_status()

    def _rot(self, axis: int, sign: int):
        w = self._sel_wp()
        if w is None:
            return
        self._to_task_if_joint(w)
        try:
            from pxr import Gf

            ax = [Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1)][axis]
            dq = Gf.Rotation(ax, sign * self.rot_step_deg).GetQuat()
            q = w["quat"]
            cur = Gf.Quatd(q[0], Gf.Vec3d(q[1], q[2], q[3]))
            new = (cur * dq).GetNormalized()   # 绕自身轴后旋
            im = new.GetImaginary()
            w["quat"] = [new.GetReal(), float(im[0]), float(im[1]), float(im[2])]
            self._refresh_status()
        except Exception as exc:
            print(f"[skill_wp] rot failed: {exc}", flush=True)

    def _set_radius(self, r: float):
        w = self._sel_wp()
        if w is not None and float(r) > 0.0:
            self._to_task_if_joint(w)   # 给 joint 点拖了半径=当成绕行 task 点
        if w is not None:
            w["radius"] = max(0.0, float(r))
            self._refresh_status()

    # ------------------------------------------------------------------ ui
    def _build_window(self):
        import omni.ui as ui

        self.window = ui.Window("Skill Waypoints (transit via-points)", width=430, height=440)
        with self.window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("Skill (pick one, edit its transit via-points)")
                self._models["skill"] = ui.ComboBox(0, *SKILL_IDS).model
                self._models["skill"].add_item_changed_fn(self._on_skill_changed)
                with ui.HStack(spacing=6, height=24):
                    ui.Label("Via-point #", width=90)
                    self._wp_frame = ui.Frame(height=24)
                    self._rebuild_wp_combo()
                with ui.HStack(spacing=6, height=26):
                    ui.Button("Add @ TCP (task)", clicked_fn=self._add_at_tcp)
                    ui.Button("Add @ joints (strict)", clicked_fn=self._add_at_joints)
                    ui.Button("Delete", clicked_fn=self._delete)
                ui.Separator()
                ui.Label("Position jog (world XYZ)")
                self._models["pstep"] = ui.ComboBox(1, *[f"{s:.3f}" for s in _POS_STEPS]).model
                self._models["pstep"].add_item_changed_fn(self._on_pstep)
                for axis, lab in ((0, "X"), (1, "Y"), (2, "Z")):
                    with ui.HStack(spacing=6, height=24):
                        ui.Label(f"{lab}", width=20)
                        ui.Button(f"-{lab}", clicked_fn=lambda a=axis: self._nudge(a, -1))
                        ui.Button(f"+{lab}", clicked_fn=lambda a=axis: self._nudge(a, +1))
                ui.Label("Rotation jog (about own axes)")
                self._models["rstep"] = ui.ComboBox(1, *[f"{s:.0f}deg" for s in _ROT_STEPS_DEG]).model
                self._models["rstep"].add_item_changed_fn(self._on_rstep)
                for axis, lab in ((0, "Rx"), (1, "Ry"), (2, "Rz")):
                    with ui.HStack(spacing=6, height=24):
                        ui.Label(f"{lab}", width=20)
                        ui.Button(f"-{lab}", clicked_fn=lambda a=axis: self._rot(a, -1))
                        ui.Button(f"+{lab}", clicked_fn=lambda a=axis: self._rot(a, +1))
                with ui.HStack(spacing=6, height=22):
                    ui.Label("Bypass radius (m)", width=140)
                    self._radius_label = ui.Label(f"{_DEFAULT_RADIUS:.3f}")
                rs = ui.FloatSlider(min=0.0, max=0.4)
                rs.model.set_value(_DEFAULT_RADIUS)
                rs.model.add_value_changed_fn(lambda m: self._on_radius_slider(m))
                self._models["radius"] = rs.model
                with ui.HStack(spacing=6, height=26):
                    ui.Button("Save", clicked_fn=self.save)
                    ui.Button("Reload", clicked_fn=self._reload)
                self._status["info"] = ui.Label("-")
                self._refresh_status()

    def _rebuild_wp_combo(self):
        if getattr(self, "_wp_frame", None) is None:
            return
        import omni.ui as ui
        n = len(self._cur())
        items = [str(i) for i in range(n)] or ["(none)"]
        self._wp_frame.clear()
        with self._wp_frame:
            m = ui.ComboBox(min(self.sel_idx, max(0, n - 1)), *items).model
            m.add_item_changed_fn(self._on_wp_changed)
            self._models["wp"] = m

    def _on_skill_changed(self, model, *_):
        idx = int(model.get_item_value_model().get_value_as_int())
        if 0 <= idx < len(SKILL_IDS):
            self.sel_skill = SKILL_IDS[idx]
            self.sel_idx = 0
            self._rebuild_wp_combo()
            self._sync_radius_slider()
            self._refresh_status()

    def _on_wp_changed(self, model, *_):
        self.sel_idx = int(model.get_item_value_model().get_value_as_int())
        self._sync_radius_slider()
        self._refresh_status()

    def _on_pstep(self, model, *_):
        self.pos_step = _POS_STEPS[int(model.get_item_value_model().get_value_as_int())]

    def _on_rstep(self, model, *_):
        self.rot_step_deg = _ROT_STEPS_DEG[int(model.get_item_value_model().get_value_as_int())]

    def _on_radius_slider(self, model):
        self._set_radius(float(model.get_value_as_float()))
        if getattr(self, "_radius_label", None) is not None:
            self._radius_label.text = f"{float(model.get_value_as_float()):.3f}"

    def _sync_radius_slider(self):
        w = self._sel_wp()
        if w is not None and self._models.get("radius") is not None:
            self._models["radius"].set_value(float(w.get("radius", _DEFAULT_RADIUS)))
            if getattr(self, "_radius_label", None) is not None:
                self._radius_label.text = f"{float(w.get('radius', _DEFAULT_RADIUS)):.3f}"

    def _reload(self):
        self.waypoints = self._load()
        for sid in SKILL_IDS:
            self.waypoints.setdefault(sid, [])
        self.sel_idx = 0
        self._rebuild_wp_combo()
        self._refresh_status()

    def _refresh_status(self):
        if self._status.get("info") is None:
            return
        w = self._sel_wp()
        n = len(self._cur())
        if w is None:
            self._status["info"].text = f"{self.sel_skill}: {n} via-points | saved: {self.last_saved}"
        elif w.get("kind") == "joint":
            self._status["info"].text = (f"{self.sel_skill} #{self.sel_idx}/{n}  [JOINT strict] "
                                         f"(saved joint coords, exact reach) | saved: {self.last_saved}")
        else:
            p = w["pos"]
            self._status["info"].text = (f"{self.sel_skill} #{self.sel_idx}/{n}  [task] "
                                         f"pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) r={w['radius']:.3f} "
                                         f"| saved: {self.last_saved}")

    # ------------------------------------------------------------------ per-frame
    def apply(self):
        """每帧把选中技能的途径点画成坐标轴 marker(选中的更大、半径越大越大)。"""
        if self.headless or self._markers is None or not getattr(self._markers, "enabled", False):
            return
        try:
            lst = self._cur()
            if not lst:
                self._markers.clear()
                return
            poses, names = [], []
            for i, w in enumerate(lst):
                poses.append([*w["pos"], *w["quat"]])
                names.append(f"wp_{self.sel_skill}_{i}")
            al = [0.10 if i == self.sel_idx else 0.06 for i in range(len(lst))]
            # TargetPoseMarkers.show 用单一 axis_length；分两批画(选中/非选中)以体现大小差异
            sel = [p for i, p in enumerate(poses) if i == self.sel_idx]
            oth = [p for i, p in enumerate(poses) if i != self.sel_idx]
            seln = [n for i, n in enumerate(names) if i == self.sel_idx]
            othn = [n for i, n in enumerate(names) if i != self.sel_idx]
            if oth:
                self._markers.show(oth, names=othn, use_arrows=True, axis_length=0.06)
            if sel:
                self._markers.show(sel, names=seln, use_arrows=True, axis_length=0.11)
        except Exception as exc:  # pragma: no cover
            print(f"[skill_wp] marker draw failed: {exc}", flush=True)

    def update_status(self):
        pass
