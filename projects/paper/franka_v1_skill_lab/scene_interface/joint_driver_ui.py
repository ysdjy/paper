#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Joint driver + collision viz on the SAVED scene USD (faithful, GUI/GPU).

Opens your saved scene directly (scene_v1_latest.usd) like the layout editor, so the
viewport shows EXACTLY your scene: the three appliances where you placed them, the four
yellow InitCorner markers, MicrowaveStand, etc. Then runs physics and lets you drive the
cabinet drawers / microwave door / coffee lever with sliders, and toggle collider display.

- Physical drive (default): set_joint_position_target -> collision-limited (door ~10deg,
  bottom drawer locked) = the TRUE physical range.
- Teleport (checkbox): write_joint_state_to_sim -> bypass collision to see the geometric
  range (does NOT mean physically reachable).

Run (GUI, do NOT pass --headless):
    ./isaaclab.sh -p projects/franka_v1_skill_lab/scene_interface/joint_driver_ui.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECTS_DIR = Path(__file__).resolve().parents[2]   # projects/
sys.path.insert(0, str(_PROJECTS_DIR))

# articulation prim + drivable joints (+ actuator gains so set_joint_position_target drives).
# labels must be ASCII (CJK renders as tofu in omni.ui).
JOINT_GROUPS = [
    {"name": "cabinet", "prim": "/World/envs/env_0/Cabinet",
     "joints_expr": ["joint_0", "joint_1", "joint_2"], "stiffness": 10.0, "damping": 1.0,
     "labels": {"joint_0": "top_drawer", "joint_2": "middle_drawer", "joint_1": "bottom_drawer (locked)"}},
    # "microwave" 成员可能是微波炉(门=joint_0)或被替换的冰箱(门=joint_1)。两个都列，谁在场景谁能动。
    {"name": "microwave", "prim": "/World/envs/env_0/Microwave",
     "joints_expr": ["joint_.*"], "stiffness": 100.0, "damping": 10.0,
     "labels": {"joint_0": "door_j0(microwave)", "joint_1": "door_j1(fridge)"}},
    {"name": "coffee_machine", "prim": "/World/envs/env_0/CoffeeMachine",
     "joints_expr": ["joint_.*"], "stiffness": 100.0, "damping": 10.0,
     "labels": {"joint_5": "coffee_handle"}},   # 只驱动 joint_5（用户选定的把手）
    # 洗碗机：joint_1 = 碗架滑轨(prismatic)。滑条范围按关节限位自动取(约 -0.576~0.136，
    # 负方向=拉出)。测试用：物理模式驱动会 respect 碰撞、Teleport 模式直接置位、Show colliders 可视化。
    {"name": "dishwasher", "prim": "/World/envs/env_0/Dishwasher",
     "joints_expr": ["joint_1"], "stiffness": 100.0, "damping": 10.0,
     "labels": {"joint_1": "dishwasher_rack (open/close)"}},
]


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Joint driver + collision viz on the saved scene USD.")
    ap.add_argument("--usd", default=None, help="scene USD to open (default: active scene_v1_latest).")
    ap.add_argument("--no_collision_vis", action="store_true", help="don't show colliders at startup")
    return ap


def main() -> int:
    ap = build_arg_parser()
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    rc = _run(args, simulation_app)
    simulation_app.close()
    return rc


def _resolve_usd(args) -> str:
    if args.usd:
        p = Path(args.usd).expanduser()
        if not p.is_absolute():
            p = _PROJECTS_DIR.parent / p
    else:
        from franka_v1_skill_lab.scene.scene_registry import resolve_active_scene

        p = Path(resolve_active_scene()["usd"])
    if not p.is_file():
        raise FileNotFoundError(f"scene USD not found: {p}")
    return str(p.resolve())


def _enable_collision_vis(on: bool = True) -> None:
    import carb

    s = carb.settings.get_settings()
    s.set_int("/persistent/physics/visualizationDisplayColliders", 2 if on else 0)
    s.set_bool("/persistent/physics/visualizationDisplayColliderNormals", False)


def _prim_world_pose(stage, prim_path):
    """Existing prim's world (pos, quat_wxyz); None if absent."""
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = m.ExtractTranslation()
    q = m.ExtractRotationQuat()
    im = q.GetImaginary()
    return (float(t[0]), float(t[1]), float(t[2])), (float(q.GetReal()), float(im[0]), float(im[1]), float(im[2]))


def _run(args, simulation_app) -> int:
    import omni.usd

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sim.utils.stage import open_stage

    usd = _resolve_usd(args)
    open_stage(usd)
    print(f"[joint_driver] opened saved scene: {usd}", flush=True)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device))
    stage = omni.usd.get_context().get_stage()

    # Wrap each appliance articulation on its EXISTING prim. init_state = the prim's current
    # saved pose, so sim.reset() keeps appliances where you placed them (not teleported to origin).
    arts: dict = {}
    for g in JOINT_GROUPS:
        prim_path = g["prim"]
        pose = _prim_world_pose(stage, prim_path)
        # microwave 成员可能已存成 /Fridge(冰箱替换后改了 prim 名)：回退试 Fridge。
        if pose is None and "/Microwave" in prim_path:
            alt = prim_path.replace("/Microwave", "/Fridge")
            pose = _prim_world_pose(stage, alt)
            if pose is not None:
                prim_path = alt
        if pose is None:
            print(f"[joint_driver] WARN: prim not found, skipping: {g['prim']}", flush=True)
            continue
        pos, rot = pose
        cfg = ArticulationCfg(
            prim_path=prim_path,
            spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(pos=pos, rot=rot),
            actuators={"all": ImplicitActuatorCfg(
                joint_names_expr=g["joints_expr"], stiffness=g["stiffness"], damping=g["damping"],
                effort_limit_sim=87.0, velocity_limit_sim=100.0,
            )},
        )
        try:
            arts[g["name"]] = Articulation(cfg)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[joint_driver] WARN: cannot wrap {g['name']} at {g['prim']}: {exc}", flush=True)

    sim.reset()
    _enable_collision_vis(not args.no_collision_vis)

    driver = JointDriver(arts)
    print("[joint_driver] drivable joints:", flush=True)
    for e in driver.entries:
        st = f"present jid={e['jid']} range=[{e['lo']:.3f},{e['hi']:.3f}]" if e["present"] else "NOT found"
        print(f"  - {e['group']}/{e['joint']} ({e['label']}): {st}", flush=True)

    window = None if args.headless else JointDriverWindow(driver, collision_on=not args.no_collision_vis)
    sim_dt = sim.get_physics_dt()

    while simulation_app.is_running():
        if window is not None:
            window.apply()
        for art in arts.values():
            art.write_data_to_sim()
        sim.step()
        for art in arts.values():
            art.update(sim_dt)
        if window is not None:
            window.update()

    return 0


class JointDriver:
    """Resolve labeled joints (id/limits) per wrapped Articulation; drive via target or teleport."""

    def __init__(self, arts: dict):
        import torch

        self.torch = torch
        self.entries: list[dict] = []
        for g in JOINT_GROUPS:
            art = arts.get(g["name"])
            for jname, label in g["labels"].items():
                e = {"group": g["name"], "art": art, "joint": jname, "label": label,
                     "present": False, "jid": None, "lo": 0.0, "hi": 0.8}
                if art is not None:
                    names = list(getattr(art.data, "joint_names", []))
                    jid = names.index(jname) if jname in names else None
                    if jid is None:
                        try:
                            ids, _ = art.find_joints(jname)
                            jid = int(ids[0]) if ids else None
                        except Exception:
                            jid = None
                    if jid is not None:
                        e["present"] = True
                        e["jid"] = jid
                        try:
                            lim = art.data.joint_pos_limits[0, jid]
                            lo, hi = float(lim[0]), float(lim[1])
                            if hi > lo:
                                e["lo"], e["hi"] = lo, hi
                        except Exception:
                            pass
                self.entries.append(e)

    def current(self, e: dict) -> float:
        if not e["present"]:
            return 0.0
        return float(e["art"].data.joint_pos[0, e["jid"]].detach().cpu())

    def drive(self, e: dict, value: float, teleport: bool = False) -> None:
        if not e["present"]:
            return
        torch = self.torch
        art = e["art"]
        jid = torch.tensor([e["jid"]], dtype=torch.long, device=art.device)
        env_ids = torch.tensor([0], dtype=torch.long, device=art.device)
        t = torch.tensor([[float(value)]], device=art.device)
        if teleport:
            v0 = torch.zeros_like(t)
            art.write_joint_state_to_sim(t, v0, joint_ids=jid, env_ids=env_ids)
        else:
            art.set_joint_position_target(t, joint_ids=jid, env_ids=env_ids)


class JointDriverWindow:
    def __init__(self, driver: JointDriver, collision_on: bool = True):
        import omni.ui as ui

        self.ui = ui
        self.driver = driver
        self.rows: list[dict] = []
        self.window = ui.Window("Joint Driver + Collision", width=560, height=560)
        with self.window.frame:
            with ui.ScrollingFrame(
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            ):
              with ui.VStack(spacing=6, height=0):
                with ui.HStack(spacing=8, height=24):
                    self.teleport_cb = ui.CheckBox(width=20)
                    ui.Label("Teleport (bypass collision)", width=240)
                    self.collision_cb = ui.CheckBox(width=20)
                    self.collision_cb.model.set_value(bool(collision_on))
                    ui.Label("Show colliders")
                self.collision_cb.model.add_value_changed_fn(
                    lambda m: _enable_collision_vis(bool(m.get_value_as_bool()))
                )
                with ui.HStack(spacing=6, height=24):
                    ui.Button("All Closed", clicked_fn=self._all_closed)
                    ui.Button("Sync sliders <- current", clicked_fn=self._sync_from_current)
                ui.Separator()
                for e in self.driver.entries:
                    if not e["present"]:
                        ui.Label(f"{e['label']}  ({e['group']}/{e['joint']}): NOT found")
                        continue
                    with ui.HStack(spacing=6, height=26):
                        ui.Label(f"{e['label']}", width=170)
                        slider = ui.FloatSlider(min=e["lo"], max=e["hi"])
                        slider.model.set_value(self.driver.current(e))
                        cur = ui.Label("", width=70)
                    self.rows.append({"entry": e, "slider": slider.model, "cur": cur})

    def _all_closed(self):
        for r in self.rows:
            lo, hi = r["entry"]["lo"], r["entry"]["hi"]
            closed = 0.0 if (lo <= 0.0 <= hi) else lo
            r["slider"].set_value(closed)

    def _sync_from_current(self):
        for r in self.rows:
            r["slider"].set_value(self.driver.current(r["entry"]))

    def apply(self):
        teleport = bool(self.teleport_cb.model.get_value_as_bool())
        for r in self.rows:
            self.driver.drive(r["entry"], float(r["slider"].get_value_as_float()), teleport=teleport)

    def update(self):
        for r in self.rows:
            r["cur"].text = f"now {self.driver.current(r['entry']):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
