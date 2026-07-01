# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""被抓取位姿编辑面板 —— 给每个【可抓目标】定义/可视化/调整抓取 pose 并保存。

可抓目标分两类，统一成 entry：
  * 桌面刚体物体（Prop_*、cube_1/2/3、knife）：参考系 = 物体 root；
  * 家电把手（抽屉拉手 top/middle/bottom、咖啡机把手、冰箱门把手）：参考系 = 家电对应 link，
    抽屉/门一动把手 marker 跟着动。

每个目标的抓取位姿存在【参考系局部坐标】里，每帧换算成世界系用坐标轴 marker 画出来（选中的更大）。
保存到 grasp_poses.json（含 member/link/frame，供抓取技能换算世界系并执行）。只在非 headless 建 UI。
顶层只 stdlib；torch/pxr/isaac 延迟到方法体内 import。
"""

from __future__ import annotations

import json

TOP_DOWN_QUAT_WXYZ = (0.0, 1.0, 0.0, 0.0)   # 默认抓取朝向：自上而下（TCP +Z 朝下）
POS_STEPS = [("1 cm", 0.01), ("2 cm", 0.02), ("5 cm", 0.05)]
ROT_STEPS = [("5 deg", 5.0), ("15 deg", 15.0), ("45 deg", 45.0)]
_EXTRA_GRASPABLE = ("cube_1", "cube_2", "cube_3", "knife")   # 桌面刚体（非 Prop_）
GRASP_FILE_NAME = "grasp_poses.json"


class GraspPosePanel:
    def __init__(self, env, provider, headless: bool = False, franka_panel=None):
        self.env = env
        self.scene = env.unwrapped.scene
        self.provider = provider
        self.device = self.scene.device
        self.env_id = getattr(provider, "env_id", 0)
        self.headless = bool(headless)
        self.franka_panel = franka_panel   # 用它的 IK 路径让机器人运动到选中的抓取位姿
        self._link_idx_cache: dict = {}

        # name -> {"member": str, "link": str|None, "default_pos":[3], "default_quat":[4], "kind": "object"|"handle"}
        self.entries = {}
        self.entries.update(self._build_object_entries())
        self.entries.update(self._build_handle_entries())
        self.objects = sorted(self.entries.keys())

        # 每个目标的抓取位姿（参考系局部）：name -> {"pos":[3], "quat":[4] wxyz}
        saved = self._load_saved()
        self.grasp_local: dict[str, dict] = {}
        for name, e in self.entries.items():
            self.grasp_local[name] = saved.get(name) or {"pos": list(e["default_pos"]), "quat": list(e["default_quat"])}

        self.selected = self.objects[0] if self.objects else None
        self.pos_step = POS_STEPS[0][1]
        self.rot_step = ROT_STEPS[1][1]
        self.visible = True
        self.last_saved = "None"

        from .target_markers import TargetPoseMarkers
        self._markers = TargetPoseMarkers(enabled=not self.headless, prefix="grasp")
        if not self.headless:
            self._build_window()

    # ---------------------------------------------------------------- entry build
    def _asset(self, name):
        try:
            return self.scene[name]
        except Exception:
            return None

    def _link_idx(self, asset, link_name):
        """解析 link 在 articulation body_names 里的下标。

        精确匹配 -> 子串匹配 -> 回退到 body 0（和状态机侧 SelectedDrawerObsAdapter 的解析一致）。
        关键：以前找不到就返回 None，导致 _build_handle_entries 把整个把手【静默跳过】——把手就不
        出现在抓取面板下拉里、没法编辑、也没有 override 传给技能（咖啡机把手/抽屉把手都中招）。
        现在最差也回退到 body 0 并打日志，保证把手一定出现、可调，且与技能侧用同一个 link。
        只有 body_names 完全拿不到（空）才返回 None。
        """
        key = (id(asset), link_name)
        if key in self._link_idx_cache:
            return self._link_idx_cache[key]
        idx = None
        try:
            names = list(getattr(asset.data, "body_names", []))
            idx = next((i for i, n in enumerate(names) if n == link_name), None)
            if idx is None:
                idx = next((i for i, n in enumerate(names) if link_name in n), None)
            if idx is None and names:
                print(f"[grasp_panel] link '{link_name}' not in body_names {names} -> fallback body 0 "
                      f"(adjust the grasp pose manually onto the real handle).", flush=True)
                idx = 0
        except Exception as exc:
            print(f"[grasp_panel] _link_idx('{link_name}') failed: {exc}", flush=True)
            idx = None
        self._link_idx_cache[key] = idx
        return idx

    def _on_table(self, asset) -> bool:
        try:
            p = asset.data.root_pos_w[self.env_id]
            return all(abs(float(v)) < 50.0 for v in p) and -2.0 < float(p[2]) < 5.0
        except Exception:
            return True

    def _build_object_entries(self) -> dict:
        out = {}
        try:
            for k in self.scene.keys():
                k = str(k)
                if not (k.startswith("Prop_") or k in _EXTRA_GRASPABLE):
                    continue
                a = self._asset(k)
                if a is None or not hasattr(a, "write_root_pose_to_sim"):
                    continue
                if not self._on_table(a):   # 跳过被锁走/移出场景的（如 lock_knife 把刀挪到 z=-5e6）
                    continue
                out[k] = {"member": k, "link": None, "kind": "object",
                          "default_pos": [0.0, 0.0, 0.0], "default_quat": list(TOP_DOWN_QUAT_WXYZ)}
        except Exception:
            pass
        return out

    def _fridge_active(self) -> bool:
        try:
            mw = self._asset("microwave")
            up = getattr(getattr(getattr(mw, "cfg", None), "spawn", None), "usd_path", "") or ""
            return "fridge" in str(up).lower()
        except Exception:
            return False

    def _build_handle_entries(self) -> dict:
        out = {}
        try:
            from .handles import build_handle_specs
            specs = build_handle_specs()
        except Exception as exc:  # pragma: no cover
            print(f"[grasp_panel] build_handle_specs failed: {exc}", flush=True)
            return out
        fridge = self._fridge_active()
        for s in specs:
            nm = s.get("name", "")
            # 冰箱/微波炉同一成员：换成冰箱时门把手在 link_1(microwave_fridge)，否则 link_0(microwave_door)
            if nm == "microwave_door" and fridge:
                continue
            if nm == "microwave_fridge" and not fridge:
                continue
            member, link = s.get("asset"), s.get("link")
            a = self._asset(member)
            if a is None or self._link_idx(a, link) is None:
                continue
            key = "handle_" + ("fridge" if nm == "microwave_fridge" else nm)
            dquat = list(TOP_DOWN_QUAT_WXYZ)
            if s.get("grasp_into"):   # 抽屉把手：默认朝向让 TCP +Z 指向【抽屉里】(而非默认的自上而下)
                gi = self._grasp_into_local_quat(member, link)
                if gi is not None:
                    dquat = gi
            out[key] = {"member": member, "link": link, "kind": "handle",
                        "default_pos": [float(v) for v in s.get("offset", (0.0, 0.0, 0.0))],
                        "default_quat": dquat}
        return out

    def _grasp_into_local_quat(self, member: str, link: str):
        """默认抽屉抓取朝向(link 局部四元数)：TCP +Z = 把手指向【柜体内部】的水平方向(=抽屉里)，+Y 朝上，
        +X 沿把手横杆。这样夹爪从外面水平接近把手、z 轴朝抽屉里(用户要求),不再朝外。"""
        try:
            import isaaclab.utils.math as mu
            import torch

            a = self._asset(member)
            li = self._link_idx(a, link)
            if a is None or li is None:
                return None
            hp = a.data.body_pos_w[self.env_id, li]
            hq = a.data.body_quat_w[self.env_id, li]
            root = a.data.root_pos_w[self.env_id]
            d = (root - hp).clone(); d[2] = 0.0    # 把手 -> 柜体根 的水平方向 = 指向抽屉里
            n = float(torch.linalg.norm(d))
            z = d / n if n > 1e-6 else torch.tensor([1.0, 0.0, 0.0], device=d.device)
            up = torch.tensor([0.0, 0.0, 1.0], device=d.device)
            x = torch.linalg.cross(up, z); nx = float(torch.linalg.norm(x))
            x = x / nx if nx > 1e-6 else torch.tensor([1.0, 0.0, 0.0], device=d.device)
            y = torch.linalg.cross(z, x)
            R = torch.stack((x, y, z), dim=1)
            qw = mu.quat_from_matrix(R.unsqueeze(0))[0]                       # 世界系抓取朝向
            ql = mu.quat_mul(mu.quat_inv(hq).unsqueeze(0), qw.unsqueeze(0))[0]  # 换到 link 局部系
            return [float(v) for v in ql.tolist()]
        except Exception as exc:  # pragma: no cover
            print(f"[grasp_panel] grasp_into default failed for {member}/{link}: {exc}", flush=True)
            return None

    # ---------------------------------------------------------------- file io
    def _grasp_path(self):
        from pathlib import Path

        from franka_v1_skill_lab.scene.scene_registry import V1_ACTIVE_DIR
        return Path(V1_ACTIVE_DIR) / GRASP_FILE_NAME

    def _load_saved(self) -> dict:
        try:
            p = self._grasp_path()
            if p.is_file():
                d = json.loads(p.read_text(encoding="utf-8"))
                return {k: {"pos": [float(x) for x in v["pos"]], "quat": [float(x) for x in v["quat"]]}
                        for k, v in (d.get("poses") or {}).items()}
        except Exception as exc:  # pragma: no cover
            print(f"[grasp_panel] load grasp_poses.json failed: {exc}", flush=True)
        return {}

    def save(self):
        try:
            p = self._grasp_path()
            poses = {}
            for name, g in self.grasp_local.items():
                e = self.entries.get(name, {})
                poses[name] = {
                    "pos": [float(v) for v in g["pos"]], "quat": [float(v) for v in g["quat"]],
                    "member": e.get("member"), "link": e.get("link"), "kind": e.get("kind"),
                }
            data = {"frame": "reference_local", "quat_order": "wxyz",
                    "note": "world = ref_world(member, link) compose local; link=null -> member root.",
                    "poses": poses}
            p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.last_saved = f"{len(poses)} poses -> {p.name}"
            print(f"[grasp_panel] saved {len(poses)} grasp poses -> {p}", flush=True)
        except Exception as exc:  # pragma: no cover
            self.last_saved = f"FAILED: {exc}"
            print(f"[grasp_panel] save failed: {exc}", flush=True)

    # ---------------------------------------------------------------- math
    def _ref_world_pose(self, name):
        """参考系世界位姿：object -> 成员 root；handle -> 家电对应 link 的世界位姿。"""
        e = self.entries[name]
        asset = self.scene[e["member"]]
        if e["link"] is None:
            return asset.data.root_pos_w[self.env_id], asset.data.root_quat_w[self.env_id]
        idx = self._link_idx(asset, e["link"])
        return asset.data.body_pos_w[self.env_id, idx], asset.data.body_quat_w[self.env_id, idx]

    def _world_grasp(self, name):
        """world = ref_world ∘ local。返回 7 元 [x,y,z, qw,qx,qy,qz]。"""
        import torch
        import isaaclab.utils.math as mu

        rpos, rquat = self._ref_world_pose(name)
        g = self.grasp_local[name]
        lp = torch.tensor(g["pos"], dtype=torch.float32, device=self.device).reshape(1, 3)
        lq = torch.tensor(g["quat"], dtype=torch.float32, device=self.device).reshape(1, 4)
        wpos = rpos.reshape(1, 3) + mu.quat_apply(rquat.reshape(1, 4), lp)
        wquat = mu.quat_mul(rquat.reshape(1, 4), lq)
        p = wpos[0].detach().cpu().tolist()
        q = wquat[0].detach().cpu().tolist()
        return [float(p[0]), float(p[1]), float(p[2]), float(q[0]), float(q[1]), float(q[2]), float(q[3])]

    # ---------------------------------------------------------------- per-frame
    def apply(self):
        if self.headless or not getattr(self._markers, "enabled", False):
            return
        if not self.visible:
            self._markers.clear()
            return
        for name in self.objects:
            try:
                p7 = self._world_grasp(name)
            except Exception:
                continue
            al = 0.12 if name == self.selected else 0.06
            self._markers.show([p7], names=[f"grasp_{name}"], axis_length=al)
        self._draw_ee()   # 机器人末端 TCP 位姿 marker（区别于抓取 marker，更大）

    def _draw_ee(self):
        """画机器人末端执行器(TCP)的当前世界位姿坐标轴。"""
        try:
            eef = self.scene["ee_frame"]
            ep = eef.data.target_pos_w[self.env_id, 0]
            eq = eef.data.target_quat_w[self.env_id, 0]
            p7 = [float(v) for v in ep.detach().cpu().tolist()] + [float(v) for v in eq.detach().cpu().tolist()]
            self._markers.show([p7], names=["ee_tcp"], axis_length=0.10)
        except Exception:
            pass

    # ---------------------------------------------------------------- edits (reference-local frame)
    def _nudge(self, axis: int, sign: int):
        if self.selected:
            self.grasp_local[self.selected]["pos"][axis] += sign * self.pos_step

    def _rot(self, axis: int, sign: int):
        if not self.selected:
            return
        from pxr import Gf

        ax = [Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1)][axis]
        dq = Gf.Rotation(ax, sign * self.rot_step).GetQuat()
        q = self.grasp_local[self.selected]["quat"]
        cur = Gf.Quatd(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        new = cur * dq   # 后乘 = 绕抓取位姿自身轴
        self.grasp_local[self.selected]["quat"] = [new.GetReal(), *[float(v) for v in new.GetImaginary()]]

    def _reset_selected(self):
        if self.selected:
            e = self.entries[self.selected]
            self.grasp_local[self.selected] = {"pos": list(e["default_pos"]), "quat": list(e["default_quat"])}

    def _move_robot_to_selected(self):
        """让机器人运动到当前选中(已微调)抓取位姿。用 Franka 控制面板的 IK 目标执行(主循环驱动)。"""
        if not self.selected or self.franka_panel is None or not hasattr(self.franka_panel, "set_ik_target"):
            print("[grasp_panel] move robot: no franka panel / no selection", flush=True)
            return
        try:
            import torch
            from runtime.scene_state_provider import PoseState

            p7 = self._world_grasp(self.selected)
            pos = torch.tensor(p7[:3], dtype=torch.float32, device=self.device)
            quat = torch.tensor(p7[3:], dtype=torch.float32, device=self.device)
            self.franka_panel.set_ik_target(PoseState(pos, quat))
            print(f"[grasp_panel] moving robot to grasp pose of '{self.selected}'", flush=True)
        except Exception as exc:  # pragma: no cover
            print(f"[grasp_panel] move robot failed: {exc}", flush=True)

    # ---------------------------------------------------------------- UI
    def _build_window(self):
        import omni.ui as ui

        self.ui = ui
        self.window = ui.Window("Grasp Pose (objects + handles)", width=420, height=380)
        self._status = {}
        with self.window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("Select grasp target (object or handle)")
                self._obj_model = ui.ComboBox(0, *self.objects).model
                self._obj_model.add_item_changed_fn(self._on_obj_changed)
                ui.Label("Position step")
                self._pos_model = ui.ComboBox(0, *[l for l, _ in POS_STEPS]).model
                self._pos_model.add_item_changed_fn(
                    lambda m, i: setattr(self, "pos_step", POS_STEPS[m.get_item_value_model().as_int][1]))
                with ui.HStack(spacing=4, height=24):
                    for ax, sgn, lab in [(0, 1, "+X"), (0, -1, "-X"), (1, 1, "+Y"),
                                         (1, -1, "-Y"), (2, 1, "+Z"), (2, -1, "-Z")]:
                        ui.Button(lab, clicked_fn=lambda a=ax, s=sgn: self._nudge(a, s))
                ui.Label("Rotation step (about grasp's own axes)")
                self._rot_model = ui.ComboBox(1, *[l for l, _ in ROT_STEPS]).model
                self._rot_model.add_item_changed_fn(
                    lambda m, i: setattr(self, "rot_step", ROT_STEPS[m.get_item_value_model().as_int][1]))
                with ui.HStack(spacing=4, height=24):
                    for ax, sgn, lab in [(0, 1, "+Roll"), (0, -1, "-Roll"), (1, 1, "+Pitch"),
                                         (1, -1, "-Pitch"), (2, 1, "+Yaw"), (2, -1, "-Yaw")]:
                        ui.Button(lab, clicked_fn=lambda a=ax, s=sgn: self._rot(a, s))
                with ui.HStack(spacing=6, height=24):
                    ui.Button("Reset (default)", clicked_fn=self._reset_selected)
                    ui.Button("Save grasp poses", clicked_fn=self.save)
                    self._vis_cb = ui.CheckBox(width=20)
                    self._vis_cb.model.set_value(True)
                    self._vis_cb.model.add_value_changed_fn(lambda m: setattr(self, "visible", bool(m.get_value_as_bool())))
                    ui.Label("Show")
                # 让机器人运动到当前选中(且已微调)的抓取位姿——走 Franka 控制面板的 IK 执行路径
                ui.Button("Move robot to this pose", clicked_fn=self._move_robot_to_selected)
                for key in ("selected", "pose", "saved"):
                    self._status[key] = ui.Label(f"{key}:")

    def _on_obj_changed(self, model, item):
        idx = model.get_item_value_model().as_int
        if 0 <= idx < len(self.objects):
            self.selected = self.objects[idx]

    def update_status(self):
        if not hasattr(self, "_status") or not self.selected:
            return
        e = self.entries.get(self.selected, {})
        g = self.grasp_local.get(self.selected, {})
        pos = g.get("pos", [0, 0, 0])
        quat = g.get("quat", [1, 0, 0, 0])
        ref = e.get("member", "?") + (f"/{e['link']}" if e.get("link") else " (root)")
        self._status["selected"].text = f"selected: {self.selected}  [{e.get('kind','?')}]  ref={ref}"
        self._status["pose"].text = ("local pos=(%.3f, %.3f, %.3f)  quat=(%.2f, %.2f, %.2f, %.2f)"
                                     % (pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]))
        self._status["saved"].text = f"saved: {self.last_saved}  ({len(self.objects)} targets)"
