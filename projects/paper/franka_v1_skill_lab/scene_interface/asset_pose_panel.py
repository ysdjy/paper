# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""资产位姿实时编辑面板 —— 物理运行中调整资产位置/朝向（机器人同时能动）。

候选：microwave / cabinet / coffee_machine（带物理，用 write_root_pose_to_sim 钉）+ InitCorner_0..3
（黄框，AssetBase 无物理，改 USD 局部 xform）。不含 cube/knife（它们由区域随机化管，不需手动调）。
被点动调过的资产进入 held，主循环每帧把它固定在目标位姿；Release 解除。微波炉与台子联动。

顶层只 stdlib；torch/pxr/isaac 延迟 import。omni.ui 文本全 ASCII。
"""

from __future__ import annotations

from . import camera_offsets as _co   # 复用 read_local_pose / set_local_pose（USD 局部 xform）

ASSET_CANDIDATES = ["robot", "microwave", "cabinet", "coffee_machine", "dishwasher", "knife",
                    "sektion_cabinet",   # 右下角白色 Sektion 橱柜(articulation,2抽屉2门可开)——移到机器人可达处
                    "InitCorner_0", "InitCorner_1", "InitCorner_2", "InitCorner_3"]
POS_STEPS = [("1 cm", 0.01), ("5 cm", 0.05), ("10 cm", 0.10)]
YAW_STEPS = [("5 deg", 5.0), ("15 deg", 15.0), ("45 deg", 45.0)]
ENV_NS = "/World/envs/env_0"


class AssetPosePanel:
    def __init__(self, env, provider):
        import omni.usd

        self.env = env
        self.scene = env.unwrapped.scene
        self.provider = provider
        self.device = self.scene.device
        self.env_id = getattr(provider, "env_id", 0)
        self.stage = omni.usd.get_context().get_stage()

        self.entries = {}   # name -> {"kind": "rigid"|"xform", "prim": path(for xform)}
        for name in ASSET_CANDIDATES:
            if not self._present(name):
                continue
            asset = self._asset(name)
            if asset is not None and hasattr(asset, "write_root_pose_to_sim"):
                self.entries[name] = {"kind": "rigid", "prim": None}
            else:
                self.entries[name] = {"kind": "xform", "prim": f"{ENV_NS}/{name}"}
        # 动态额外资产(add_usd_asset 加的 Prop_ 水果)：当刚体处理，可移动 + 可缩放 + 可保存。
        try:
            for k in self.scene.keys():
                k = str(k)
                if k.startswith("Prop_") and k not in self.entries:
                    asset = self._asset(k)
                    if asset is not None and hasattr(asset, "write_root_pose_to_sim"):
                        self.entries[k] = {"kind": "rigid", "prim": None}   # 物理刚体
                    else:
                        self.entries[k] = {"kind": "xform", "prim": f"{ENV_NS}/{k}"}   # 静态道具(USD xform)
        except Exception:
            pass
        self.assets = list(self.entries.keys())
        self.selected = self.assets[0] if self.assets else None
        self.pos_step = POS_STEPS[1][1]
        self.yaw_step = YAW_STEPS[1][1]
        self.held: dict[str, dict] = {}   # name -> {"pos": t3/list, "quat": t4/list}
        self.scale_target: dict[str, float] = {}   # name -> 统一缩放(改 USD xformOp:scale；保存+重载生效)
        self.scale_step = 1.1                       # 每次 x/ / 这个倍数

        self._stand_z = None
        if self._present("microwave_stand"):
            try:
                self._stand_z = float(self.scene["microwave_stand"].data.root_pos_w[self.env_id][2])
            except Exception:
                self._stand_z = None
        self._dw_stand_z = None
        if self._present("dishwasher_stand"):
            try:
                self._dw_stand_z = float(self.scene["dishwasher_stand"].data.root_pos_w[self.env_id][2])
            except Exception:
                self._dw_stand_z = None
        # 洗碗机缩放(给台子本体中心偏移用)；读不到回退 (0.4,0.4,0.4)
        self._dw_scale = (0.4, 0.4, 0.4)
        try:
            sc = getattr(getattr(self._asset("dishwasher").cfg, "spawn", None), "scale", None)
            if sc:
                self._dw_scale = tuple(float(v) for v in sc)
        except Exception:
            pass
        # microwave 成员是否其实是冰箱(spawn usd 含 'fridge')：只为下拉框显示名，不改任何内部名。
        self._fridge_active = False
        try:
            mw = self._asset("microwave")
            up = getattr(getattr(getattr(mw, "cfg", None), "spawn", None), "usd_path", "") or ""
            self._fridge_active = "fridge" in str(up).lower()
        except Exception:
            self._fridge_active = False

        self._robot_stand_z = None
        if self._present("robot_stand"):
            try:
                self._robot_stand_z = float(self.scene["robot_stand"].data.root_pos_w[self.env_id][2])
            except Exception:
                self._robot_stand_z = None
        self._build_window()

    # --- helpers ---
    def _present(self, name) -> bool:
        try:
            return name in self.scene.keys()
        except Exception:
            return False

    def _asset(self, name):
        try:
            return self.scene[name]
        except Exception:
            return None

    def _dw_stand_offset(self, quat_wxyz):
        """洗碗机台子相对洗碗机原点的世界 XY 偏移(随当前朝向)；复用 scene_props 的算法。"""
        try:
            from .scene_props import dw_stand_offset_world
            return dw_stand_offset_world(quat_wxyz, self._dw_scale)
        except Exception:
            return (0.0, 0.0)

    def _display_name(self, name) -> str:
        """下拉框显示名：microwave 成员若已被换成冰箱(spawn usd 含 'fridge')则显示 'fridge'。

        仅改显示，内部仍用成员名 'microwave'(选中索引→self.assets，不受影响)。
        按 usd_path 识别(不靠 prim 名)：prim 仍叫 /Microwave，但资产其实是冰箱。
        """
        if name == "microwave" and getattr(self, "_fridge_active", False):
            return "fridge"
        return str(name)

    def _kind(self, name) -> str:
        return self.entries.get(name, {}).get("kind", "rigid")

    def _read_pose(self, name):
        """返回 (pos, quat_wxyz)。rigid -> world；xform -> USD 局部。"""
        if self._kind(name) == "rigid":
            a = self.scene[name]
            return (a.data.root_pos_w[self.env_id].clone(), a.data.root_quat_w[self.env_id].clone())
        pose = _co.read_local_pose(self.stage, self.entries[name]["prim"])
        return pose if pose is not None else ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))

    def _ensure_target(self, name) -> dict:
        if name not in self.held:
            pos, quat = self._read_pose(name)
            self.held[name] = {"pos": pos, "quat": quat}
        return self.held[name]

    # --- UI ---
    def _build_window(self):
        import omni.ui as ui

        self.ui = ui
        self.window = ui.Window("Asset Pose (live)", width=380, height=320)
        self._status = {}
        with self.window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("Select asset")
                self._asset_model = ui.ComboBox(0, *[self._display_name(a) for a in self.assets]).model
                self._asset_model.add_item_changed_fn(self._on_asset_changed)
                ui.Label("Position step")
                self._pos_model = ui.ComboBox(1, *[l for l, _ in POS_STEPS]).model
                self._pos_model.add_item_changed_fn(
                    lambda m, i: setattr(self, "pos_step", POS_STEPS[m.get_item_value_model().as_int][1]))
                with ui.HStack(spacing=4, height=24):
                    for ax, sgn, lab in [(0, 1, "+X"), (0, -1, "-X"), (1, 1, "+Y"),
                                         (1, -1, "-Y"), (2, 1, "+Z"), (2, -1, "-Z")]:
                        ui.Button(lab, clicked_fn=lambda a=ax, s=sgn: self._nudge(a, s))
                ui.Label("Yaw step (about world Z)")
                self._yaw_model = ui.ComboBox(1, *[l for l, _ in YAW_STEPS]).model
                self._yaw_model.add_item_changed_fn(
                    lambda m, i: setattr(self, "yaw_step", YAW_STEPS[m.get_item_value_model().as_int][1]))
                with ui.HStack(spacing=6, height=24):
                    ui.Button("+Yaw", clicked_fn=lambda: self._yaw(1))
                    ui.Button("-Yaw", clicked_fn=lambda: self._yaw(-1))
                ui.Label("Scale (size; saved + applied on reload)")
                with ui.HStack(spacing=6, height=24):
                    ui.Button("Scale +", clicked_fn=lambda: self._scale(1))
                    ui.Button("Scale -", clicked_fn=lambda: self._scale(-1))
                    ui.Button("Scale = 1", clicked_fn=lambda: self._scale(0))
                with ui.HStack(spacing=6, height=24):
                    ui.Button("Release (unpin)", clicked_fn=self._release)
                    ui.Button("Sync target <- current", clicked_fn=self._sync)
                    ui.Button("Release All", clicked_fn=self.clear)
                for key in ("selected", "held"):
                    self._status[key] = ui.Label(f"{key}:")

    def _on_asset_changed(self, model, item):
        idx = model.get_item_value_model().as_int
        if 0 <= idx < len(self.assets):
            self.selected = self.assets[idx]

    # --- edits (work for both kinds; pos/quat are tensors for rigid, tuples for xform) ---
    def _nudge(self, axis: int, sign: int):
        if not self.selected:
            return
        t = self._ensure_target(self.selected)
        pos = t["pos"]
        if hasattr(pos, "clone"):      # tensor (rigid)
            pos = pos.clone()
            pos[axis] = pos[axis] + sign * self.pos_step
        else:                          # tuple (xform)
            pos = list(pos)
            pos[axis] += sign * self.pos_step
            pos = tuple(pos)
        t["pos"] = pos

    def _yaw(self, sign: int):
        if not self.selected:
            return
        from pxr import Gf

        t = self._ensure_target(self.selected)
        dq = Gf.Rotation(Gf.Vec3d(0, 0, 1), sign * self.yaw_step).GetQuat()
        q = t["quat"]
        if hasattr(q, "clone"):        # tensor (rigid)
            ql = [float(v) for v in q.detach().cpu().tolist()]
        else:
            ql = [float(v) for v in q]
        cur = Gf.Quatd(ql[0], ql[1], ql[2], ql[3])
        new = dq * cur
        nq = (new.GetReal(), *[float(v) for v in new.GetImaginary()])
        if hasattr(q, "clone"):
            import torch
            t["quat"] = torch.tensor(nq, dtype=torch.float32, device=self.device)
        else:
            t["quat"] = tuple(float(v) for v in nq)

    def _prim_path(self, name) -> str:
        return f"{ENV_NS}/{name}"

    def _read_scale(self, name) -> float:
        """读 prim 当前 xformOp:scale 的 x 分量(统一缩放)。读不到返回 1.0。"""
        try:
            from pxr import UsdGeom

            prim = self.stage.GetPrimAtPath(self._prim_path(name))
            for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                if op.GetName() == "xformOp:scale" and op.Get() is not None:
                    return float(op.Get()[0])
        except Exception:
            pass
        return 1.0

    def _set_uniform_scale(self, name, s: float) -> None:
        """设 prim 的 xformOp:scale=(s,s,s)。缺则新增。视觉立即缩放；物理 collider 在重载后生效。"""
        try:
            from pxr import Gf, UsdGeom

            prim = self.stage.GetPrimAtPath(self._prim_path(name))
            xf = UsdGeom.Xformable(prim)
            ops = {op.GetName(): op for op in xf.GetOrderedXformOps()}
            if "xformOp:scale" in ops:
                ops["xformOp:scale"].Set(Gf.Vec3d(float(s), float(s), float(s)))
            else:
                xf.AddScaleOp().Set(Gf.Vec3d(float(s), float(s), float(s)))
        except Exception:
            pass

    def _scale(self, sign: int):
        """sign>0 放大 / <0 缩小 / ==0 复位 1.0。"""
        if not self.selected:
            return
        cur = self.scale_target.get(self.selected, self._read_scale(self.selected))
        if sign > 0:
            cur *= self.scale_step
        elif sign < 0:
            cur /= self.scale_step
        else:
            cur = 1.0
        cur = max(0.05, min(20.0, cur))
        self.scale_target[self.selected] = cur
        self._set_uniform_scale(self.selected, cur)

    def _release(self):
        if self.selected:
            self.held.pop(self.selected, None)

    def _sync(self):
        if self.selected:
            pos, quat = self._read_pose(self.selected)
            self.held[self.selected] = {"pos": pos, "quat": quat}

    def clear(self):
        self.held.clear()

    # --- per-frame ---
    def apply(self):
        # 缩放：每帧把目标 scale 写回 USD xformOp:scale（保证 Save 捕获到，且抵抗 fabric 复位）。
        for name, s in self.scale_target.items():
            self._set_uniform_scale(name, s)
        if not self.held:
            return
        import torch

        env_ids = torch.tensor([self.env_id], dtype=torch.long, device=self.device)
        zero_vel = torch.zeros((1, 6), dtype=torch.float32, device=self.device)
        for name, t in self.held.items():
            try:
                if self._kind(name) == "rigid":
                    a = self.scene[name]
                    pos = t["pos"] if hasattr(t["pos"], "reshape") else torch.tensor(t["pos"], device=self.device)
                    quat = t["quat"] if hasattr(t["quat"], "reshape") else torch.tensor(t["quat"], device=self.device)
                    root = torch.cat((pos.reshape(1, 3).float(), quat.reshape(1, 4).float()), dim=-1).to(self.device)
                    a.write_root_pose_to_sim(root, env_ids=env_ids)
                    if hasattr(a, "write_root_velocity_to_sim"):
                        a.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
                else:   # xform (InitCorner)
                    _co.set_local_pose(self.stage, self.entries[name]["prim"], t["pos"], t["quat"])
            except Exception:
                continue
        # 微波炉台子跟随微波炉（同 XY + yaw，保持自身 z；实时读底座 z 作兜底）
        if "microwave" in self.held and self._present("microwave_stand"):
            try:
                mt = self.held["microwave"]
                mp = mt["pos"]
                mx = float(mp[0]); my = float(mp[1])
                mq = mt["quat"]
                ql = mq.detach().cpu().tolist() if hasattr(mq, "detach") else list(mq)
                stand = self.scene["microwave_stand"]
                if self._stand_z is None:
                    self._stand_z = float(stand.data.root_pos_w[self.env_id][2])
                spos = torch.tensor([mx, my, self._stand_z], dtype=torch.float32, device=self.device)
                quat = torch.tensor([float(v) for v in ql], dtype=torch.float32, device=self.device)
                root = torch.cat((spos.reshape(1, 3), quat.reshape(1, 4)), dim=-1)
                stand.write_root_pose_to_sim(root, env_ids=env_ids)
                if hasattr(stand, "write_root_velocity_to_sim"):
                    stand.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
            except Exception:
                pass
        # 洗碗机台子跟随洗碗机（同 XY + yaw，保持自身 z）
        # 注意：不依赖 __init__ 时读到的 self._dw_stand_z（那次读失败=None 会让跟随永久跳过）；
        # 这里实时读底座当前 z 作兜底，保证只要底座存在就一定跟随。
        if "dishwasher" in self.held and self._present("dishwasher_stand"):
            try:
                dt = self.held["dishwasher"]
                dp = dt["pos"]
                dq = dt["quat"]
                ql = dq.detach().cpu().tolist() if hasattr(dq, "detach") else list(dq)
                stand = self.scene["dishwasher_stand"]
                sz = self._dw_stand_z
                if sz is None:
                    sz = float(stand.data.root_pos_w[self.env_id][2])
                    self._dw_stand_z = sz   # 缓存，后续帧直接用
                # 台子只盖本体：用同一函数把本体中心偏移(随当前朝向)加到洗碗机 XY 上
                ox, oy = self._dw_stand_offset(ql)
                spos = torch.tensor([float(dp[0]) + ox, float(dp[1]) + oy, sz],
                                    dtype=torch.float32, device=self.device)
                quat = torch.tensor([float(v) for v in ql], dtype=torch.float32, device=self.device)
                root = torch.cat((spos.reshape(1, 3), quat.reshape(1, 4)), dim=-1)
                stand.write_root_pose_to_sim(root, env_ids=env_ids)
                if hasattr(stand, "write_root_velocity_to_sim"):
                    stand.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
            except Exception:
                pass
        # 机器人底座跟随机器人（绑定：同 XY + yaw，保持自身 z=支架顶；机器人坐在它上面一起动）
        if "robot" in self.held and self._present("robot_stand"):
            try:
                rt = self.held["robot"]
                rp = rt["pos"]
                rq = rt["quat"]
                ql = rq.detach().cpu().tolist() if hasattr(rq, "detach") else list(rq)
                stand = self.scene["robot_stand"]
                if self._robot_stand_z is None:
                    self._robot_stand_z = float(stand.data.root_pos_w[self.env_id][2])
                spos = torch.tensor([float(rp[0]), float(rp[1]), self._robot_stand_z], dtype=torch.float32, device=self.device)
                quat = torch.tensor([float(v) for v in ql], dtype=torch.float32, device=self.device)
                root = torch.cat((spos.reshape(1, 3), quat.reshape(1, 4)), dim=-1)
                stand.write_root_pose_to_sim(root, env_ids=env_ids)
                if hasattr(stand, "write_root_velocity_to_sim"):
                    stand.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
            except Exception:
                pass

    def update_status(self):
        if not hasattr(self, "_status"):
            return
        sc = self.scale_target.get(self.selected) if self.selected else None
        sc_txt = f"  scale={sc:.2f}" if sc is not None else ""
        self._status["selected"].text = (
            f"selected: {self._display_name(self.selected) if self.selected else '-'} "
            f"({self._kind(self.selected) if self.selected else '-'}){sc_txt}")
        self._status["held"].text = f"held: {sorted(self._display_name(k) for k in self.held.keys())}"
