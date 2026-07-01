#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""TEST 模式 entry：碰撞可视化 + 资产实时摆放 + 机器人/关节驱动 + 保存场景（GUI/GPU）。

窗口：
- Franka Control：点动/绝对 IK、关节、home、夹爪（测可达性）。
- Asset Pose：物理运行中实时摆放资产（微波炉带台子联动），点动 ±X/±Y/±Z/±Yaw。
- Joint Driver：驱动柜子三抽屉 / 微波炉门 / 咖啡机把手（位置目标 / teleport）+ 碰撞显示开关。
- Scene：Reset layout（区域随机 cubes/knife）/ Save as latest scene（存为 scene_v1_latest，机器人不存）。

场景用你保存的家电位置；黄色 InitCorner 方框会 spawn（可视 + 保存保留区域）；微波炉下有台子。
use_fabric=False 以便保存时导出能抓到移动后的资产位姿。所有 print / omni.ui 文本 ASCII。

Run（GUI，不要 --headless）:
    ./isaaclab.sh -p projects/franka_v1_skill_lab/scene_interface/test_mode_ui.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECTS_DIR = Path(__file__).resolve().parents[2]   # projects/
sys.path.insert(0, str(_PROJECTS_DIR))

from franka_v1_skill_lab.scene import V1_BASE_TASK_ID  # noqa: E402
from franka_v1_skill_lab.scene_interface import ResetMode, SceneConfig, SceneMode  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Scene TEST mode: live asset edit + robot/joint drive + save.")
    ap.add_argument("--task", default=V1_BASE_TASK_ID)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--control_hz", type=float, default=50.0)
    ap.add_argument("--base_scene", action="store_true",
                    help="use base cfg appliance poses instead of saved scene_v1_latest.")
    ap.add_argument("--no_collision_vis", action="store_true")
    ap.add_argument("--pengzhuang", action="store_true",
                    help="开启碰撞体可视化(绿/红 collider overlay)。默认关闭；加此开关才显示。")
    ap.add_argument("--no_refine", action="store_true", help="skip handle collision refinement (convexDecomposition).")
    ap.add_argument("--no_fridge", action="store_true", help="keep microwave instead of replacing it with the fridge.")
    ap.add_argument("--keep_fridge", action="store_true",
                    help="保留冰箱(microwave 成员)。默认【删除】冰箱(用户要求)。")
    ap.add_argument("--keep_dishwasher", action="store_true",
                    help="保留洗碗机。默认【删除】洗碗机(用户要求)。")
    ap.add_argument("--no_collision_monitor", action="store_true",
                    help="skip the robot ContactSensor / collision monitoring.")
    ap.add_argument("--no_cameras", action="store_true", help="don't attach the two cameras / image stream.")
    ap.add_argument("--no_stream", action="store_true", help="don't ZMQ-publish camera frames to the viewer.")
    ap.add_argument("--stream_addr", default="tcp://*:5557", help="ZMQ PUB addr for the image viewer.")
    ap.add_argument("--controller", default=None,
                    help="external test controller 'pkg.module:ClassName' (state-machine/VLA plug-in). "
                         "It gets the SceneSession and drives the robot each frame.")
    ap.add_argument("--describe", action="store_true",
                    help="add the 'Scene Perception (GPT / Qwen)' panel: pick a backend, one click sends "
                         "the RGB view(s) and shows objects/relations. Needs cameras + the chosen server "
                         "running (scene_describer/ for GPT, perception_qwen/ for Qwen).")
    ap.add_argument("--describe_addr", default="http://127.0.0.1:5599",
                    help="GPT scene_describer server address for the perception panel.")
    ap.add_argument("--qwen_addr", default="http://127.0.0.1:5601",
                    help="Qwen perception_qwen server address for the perception panel.")
    ap.add_argument("--add_asset", default=None,
                    help="extra USD(s) to spawn into the latest scene: comma-separated local paths or "
                         "http/https/omniverse:// URLs. Spread in a row in front of the robot.")
    ap.add_argument("--add_asset_scale", type=float, default=1.0,
                    help="uniform scale applied to every --add_asset item.")
    ap.add_argument("--add_asset_rigid", action="store_true",
                    help="wrap added assets as rigid bodies (physics). Default: static visual props moved via USD "
                         "xform (SimReady cloud props use deferred payloads, so rigid-wrap warns and does not attach).")
    ap.add_argument("--no_props", action="store_true",
                    help="skip the DEFAULT_PROP_URLS (lemon/orange/pomegranate) loaded by default.")
    ap.add_argument("--no_robot_stand", action="store_true",
                    help="don't place the robot on the Isaac stand (stand top aligned to robot base).")
    ap.add_argument("--no_isaac_props", action="store_true",
                    help="don't load the downloaded Isaac props (YCB/mugs/cabinet...) into the +Y parking grid.")
    # paper: UIs OFF by default (default="none"). Open panels explicitly with --ui grasp,scene,...
    # or the individual flags (--grasp/--asset/...). Use --ui all to restore the original all-on.
    ap.add_argument("--ui", default="none",
                    help="选择加载哪些内置 UI 面板(逗号分隔)，按需精简界面。可用键: "
                         "franka(机器人关节/任务空间控制) / asset(资产位姿编辑,移动场景物体) / "
                         "grasp(抓取位姿编辑,物体+把手) / joint(家电关节驱动:抽屉/门/咖啡机) / "
                         "camera(相机视角调整,需相机) / scene(场景重置+保存) / viz(可视化开关:碰撞/箭头)。 "
                         "paper 默认 'none'=都不加(UI 默认关闭)；'all'=全部；技能测试UI用 --controller。")
    # 单独的开关写法(和 --ui 等价，二选一)：写了任意一个就【只加】这些面板，方便多行命令里逐行注释取舍。
    ui_grp = ap.add_argument_group("UI 面板单独开关 (等价于 --ui，写了任意一个就只加这些)")
    ui_grp.add_argument("--franka", action="store_true", help="机器人控制面板(关节/任务空间读写+执行)")
    ui_grp.add_argument("--asset", action="store_true", help="资产位姿编辑面板(移动/旋转/缩放场景物体)")
    ui_grp.add_argument("--grasp", action="store_true", help="抓取位姿编辑面板(物体+把手；抽屉技能取这里的pose)")
    ui_grp.add_argument("--waypoints", action="store_true", help="技能途径点编辑面板(每个技能的过渡绕行途径点)")
    ui_grp.add_argument("--joint", action="store_true", help="家电关节驱动面板(抽屉/门/咖啡机)")
    ui_grp.add_argument("--camera", action="store_true", help="相机视角调整面板(需相机)")
    ui_grp.add_argument("--scene", action="store_true", help="场景重置+保存面板")
    ui_grp.add_argument("--viz", action="store_true", help="可视化开关面板(碰撞体/目标箭头)")
    return ap


# 内置 UI 面板键(供 --ui 选择)：键 -> 中文说明
_UI_KEYS = {
    "franka": "机器人控制(关节/任务空间读写+执行)",
    "asset": "资产位姿编辑(实时移动/旋转/缩放场景物体)",
    "grasp": "抓取位姿编辑(桌面物体+把手的抓取pose；抽屉技能取这里的pose)",
    "waypoints": "技能途径点编辑(每个技能的过渡绕行途径点:世界系pose+绕行半径)",
    "joint": "家电关节驱动(抽屉/门/咖啡机关节目标)",
    "camera": "相机视角调整(需相机)",
    "scene": "场景重置 + 保存最新场景",
    "viz": "可视化开关(碰撞体/目标箭头)",
}


def _parse_ui_selection(ui_arg: str) -> set:
    """把 --ui 的逗号串解析成要加载的面板键集合。'all'=全部，'none'=空。未知键告警并忽略。"""
    raw = (ui_arg or "all").strip().lower()
    if raw in ("all", ""):
        return set(_UI_KEYS)
    if raw == "none":
        return set()
    sel = {k.strip() for k in raw.split(",") if k.strip()}
    unknown = sel - set(_UI_KEYS)
    if unknown:
        print(f"[test_mode] WARNING: 未知 --ui 键 {sorted(unknown)}；可用: {sorted(_UI_KEYS)}", flush=True)
    return sel & set(_UI_KEYS)


# 默认往最新场景加载的 SimReady 道具。已下载到本地 USD 资产目录(含依赖材质/纹理，自包含)。
# 用 _base.usd(纯几何，无缩略图 rig)。--no_props 跳过；--add_asset 覆盖这份默认。
# 这 4 个水果本来就在桌面上(用户摆放并保存进 manifest)：spawn 后 apply_latest_scene_to_cfg
# 会用最新保存的 pose 覆盖回桌面位置(lemon/orange z≈0.76)。--no_props 可跳过。
_SRP = "SapienAssetPipeline/usd_assets/SimReadyProps/"
DEFAULT_PROP_URLS: list = [
    _SRP + "lemon_01/lemon_01_base.usd",
    _SRP + "orange_01/orange_01_base.usd",
    _SRP + "orange_02/orange_02_base.usd",
    _SRP + "pomegranate01/pomegranate01_base.usd",
]

# 下载到本地的 Isaac 官方道具(含材质)。默认加载到 +Y 侧停车网格(远离主场景)，供编辑时取用。
# (相对路径, 中文名, 是否内置物理刚体)。rigid=True -> RigidObject(可抓/掉落)；False -> 静态道具。
_IP = "SapienAssetPipeline/usd_assets/IsaacProps/"
ISAAC_PROPS = [
    # 用户要求删除 pitcher 附近地面的 4 个物品：芥末瓶(洗洁剂)/糖盒(盒子)/大厨咖啡罐(杯)/500ml烧杯(杯)。
    (_IP + "Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd", "番茄汤罐头", True),
    (_IP + "Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd", "饼干盒", True),
    # rigid=True -> 物理刚体(RigidObject)：可在 Asset Pose 面板拖动(走物理路径)。静态(rigid=False)件
    # 用 USD xform 移动会被 Fabric 复位 -> 拖不动；这些都在任务4加了碰撞+刚体，故设 True 可拖。
    (_IP + "Props/YCB/Axis_Aligned/011_banana.usd", "香蕉", True),
    (_IP + "Props/YCB/Axis_Aligned/051_large_clamp.usd", "大号夹钳", True),
    (_IP + "Props/YCB/Axis_Aligned/037_scissors.usd", "剪刀", True),
    (_IP + "Props/YCB/Axis_Aligned/025_mug.usd", "马克杯(YCB)", True),
    (_IP + "Props/YCB/Axis_Aligned/007_tuna_fish_can.usd", "金枪鱼罐头", True),
    (_IP + "Props/YCB/Axis_Aligned/024_bowl.usd", "碗", True),
    (_IP + "Props/YCB/Axis_Aligned/010_potted_meat_can.usd", "午餐肉罐头", True),
    # 注：原始 KLT 周转箱(Prop_small_KLT_visual_collision)已从这里移除——只保留下方额外复制的
    # 3 份 KLT(Prop_KLT_1/2/3)，避免多出第 4 个在地面上的篮子。
    # 注：机器人支架(Stand)不在此停放——机器人已坐在专门的 RobotStand 上(add_robot_stand)，
    # 这里再放一个会重复/悬空，故移除。需要单独支架道具时用 --add_asset 显式加。
    (_IP + "Props/Food/mac_n_cheese_centered.usd", "芝士通心粉盒", True),
    # SM_Mug_A2 已移除(用户要求删除 Prop_SM_Mug_A2)。
    (_IP + "Props/Mugs/SM_Mug_B1.usd", "马克杯B1", True),
    (_IP + "Props/Mugs/SM_Mug_C1.usd", "马克杯C1", True),
    (_IP + "Props/Mugs/SM_Mug_D1.usd", "马克杯D1", True),
    # Sektion 橱柜【已改为在 env cfg 里作为 articulation 'sektion_cabinet' 接入】(抽屉是真关节、可被
    # 打开抽屉技能拉开)，故这里不再作为静态道具重复添加，避免一个场景里出现两个白柜。
    # (_IP + "Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd", "Sektion橱柜", False),
]

# 道具初始朝向覆盖(wxyz)，按 USD 文件名匹配。默认空：已摆放的道具朝向以保存场景(manifest)为准，
# 这里只给【全新场景】里某些默认倒扣的道具一个翻正初值。add_usd_asset 的 ground 已是 rot-aware。
PROP_ROT_OVERRIDE: dict = {}


def main() -> int:
    ap = build_arg_parser()
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    if not args.no_cameras:
        args.enable_cameras = True          # 两相机渲染需要（同 eval）
    app_launcher = AppLauncher(args)            # 进程内唯一 AppLauncher
    simulation_app = app_launcher.app
    rc = _run(args, app_launcher, simulation_app)
    simulation_app.close()
    return rc


class _ScenePanel:
    """Reset/Save 控制面板（缓冲式：按钮置 flag，主循环 pop 处理）。"""

    def __init__(self):
        import omni.ui as ui

        self._reset = False
        self._save = False
        self.window = ui.Window("Scene (TEST)", width=320, height=130)
        with self.window.frame:
            with ui.VStack(spacing=8, height=0):
                ui.Button("Reset to initial (all objects)", clicked_fn=self._on_reset)
                ui.Button("Save as latest scene", clicked_fn=self._on_save)
                self._status = ui.Label("status: ready")

    def _on_reset(self):
        self._reset = True

    def _on_save(self):
        self._save = True

    def pop_reset(self) -> bool:
        v = self._reset
        self._reset = False
        return v

    def pop_save(self) -> bool:
        v = self._save
        self._save = False
        return v

    def set_status(self, s: str):
        self._status.text = "status: " + s


class _VizPanel:
    """可视化开关面板：碰撞体 / 目标箭头 / 黄框 区域标记。调试者自选。"""

    def __init__(self, session, show_arrows: bool = True, show_colliders: bool = True, show_markers: bool = False):
        import omni.ui as ui

        self.session = session
        self.show_arrows = show_arrows
        session.set_collision_visible(show_colliders)
        session.set_markers_visible(show_markers)
        self.window = ui.Window("Visualization", width=280, height=120)
        with self.window.frame:
            with ui.VStack(spacing=6, height=0):
                self._cb_coll = self._row(ui, "Show colliders", show_colliders,
                                          lambda v: self.session.set_collision_visible(v))
                self._cb_arrow = self._row(ui, "Show target arrows", show_arrows,
                                           lambda v: setattr(self, "show_arrows", v))
                self._cb_mark = self._row(ui, "Show region markers", show_markers,
                                          lambda v: self.session.set_markers_visible(v))

    def _row(self, ui, label, default, on_change):
        with ui.HStack(spacing=8, height=22):
            cb = ui.CheckBox(width=20)
            cb.model.set_value(bool(default))
            cb.model.add_value_changed_fn(lambda m: on_change(bool(m.get_value_as_bool())))
            ui.Label(label)
        return cb


def _load_controller(spec: str, session):
    """加载外部测试控制器 'pkg.module:ClassName'，实例化为 Class(session)。失败返回 None。"""
    import importlib

    try:
        mod_path, _, cls_name = spec.partition(":")
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        ctrl = cls(session)
        print(f"[test_mode] external controller loaded: {spec}", flush=True)
        return ctrl
    except Exception as exc:
        print(f"[test_mode] WARNING: failed to load controller '{spec}': {exc}", flush=True)
        return None


def _run(args, app_launcher, simulation_app) -> int:
    import torch

    from franka_v1_skill_lab.scene_interface import SceneSession

    use_cam = not args.no_cameras
    import re

    def _prop_name(path):
        stem = path.rstrip("/").split("/")[-1].rsplit(".", 1)[0]
        if stem.endswith("_base"):
            stem = stem[:-5]
        return "Prop_" + re.sub(r"[^A-Za-z0-9_]", "_", stem)

    items = []
    # (1) 水果道具(SimReady)：机器人前方一排
    if args.add_asset:
        urls = [u.strip() for u in args.add_asset.split(",") if u.strip()]
        fruit_rigid = bool(args.add_asset_rigid)
    elif not args.no_props:
        urls = list(DEFAULT_PROP_URLS)
        fruit_rigid = True         # 默认水果(_base 已加碰撞+刚体)：当刚体 spawn -> 可抓/可掉落/可拖动
    else:
        urls = []
        fruit_rigid = bool(args.add_asset_rigid)
    for i, url in enumerate(urls):
        items.append({
            "name": _prop_name(url), "usd_path": url,
            "pos": (0.45, -0.20 + 0.12 * i, 0.0),
            "scale": (float(args.add_asset_scale),) * 3,
            "rigid": fruit_rigid,
            "ground": True,        # 按 bbox 落地，避免水果悬空
        })
    # (2) Isaac 官方道具：+Y 侧停车网格(远离主场景)，按各自是否有物理设 rigid
    if not args.no_isaac_props:
        for i, (path, _zh, rigid) in enumerate(ISAAC_PROPS):
            col, row = i % 6, i // 6
            rot = next((r for k, r in PROP_ROT_OVERRIDE.items() if path.endswith(k)), (1.0, 0.0, 0.0, 0.0))
            items.append({
                "name": _prop_name(path), "usd_path": path,
                "pos": (-1.0 + col * 0.45, 1.6 + row * 0.45, 0.0),
                "rot": rot,            # 翻正倒扣的道具(如碗)；默认单位四元数
                "scale": (1.0, 1.0, 1.0),
                "rigid": bool(rigid),
                "ground": True,        # 按 bbox 落地，避免停放道具(篮子/碗等)悬空
            })
        # L 形前台桌(云端下载, metersPerUnit=0.01 -> scale 0.01 得真实 ~2m)。kinematic+三角网格碰撞:
        # 静态不掉落、精确贴形不穿模、可在面板拖动定位。抽屉是整体网格不可开(资产限制)。
        items.append({
            "name": "Prop_L_Desk",
            "usd_path": "SapienAssetPipeline/usd_assets/CloudProps/L_Desk/L_Desk.usd",
            "pos": (1.8, 1.8, 0.0), "scale": (0.01, 0.01, 0.01),
            "rigid": False, "ground": True,   # 静态：de-instance 注册精确三角网格碰撞(不穿模)
        })
        # 额外复制 3 份 KLT 周转箱（独立命名 Prop_KLT_1/2/3），停在 +Y 网格里，可在编辑器里再摆/堆叠测试。
        _KLT = _IP + "Props/KLT_Bin/small_KLT_visual_collision.usd"
        _n0 = len(items)
        for j in range(3):
            col, row = (_n0 + j) % 6, (_n0 + j) // 6
            items.append({
                "name": f"Prop_KLT_{j+1}", "usd_path": _KLT,
                "pos": (-1.0 + col * 0.45, 1.6 + row * 0.45, 0.0),
                "scale": (1.0, 1.0, 1.0), "rigid": True, "ground": True,
            })
    extra_assets = tuple(items)
    if items:
        print(f"[test_mode] spawning {len(items)} extra assets: {[it['name'] for it in items]}", flush=True)
    cfg = SceneConfig(
        mode=SceneMode.TEST, task_id=args.task, device=args.device, headless=args.headless,
        enable_cameras=use_cam, enable_fp=use_cam,   # 两相机 RGBD（front+wrist，同视角）
        free_microwave_door=False,           # 门 stiffness 保留 -> 可被关节面板位置驱动
        load_latest_scene=not args.base_scene,
        add_microwave_stand=False,                     # 微波炉换成冰箱(落地)，不用台子
        replace_microwave_with_fridge=not args.no_fridge,
        lock_knife=True,                               # 锁死小刀关节
        enable_collision_monitor=not args.no_collision_monitor,   # 机器人各连杆 ContactSensor

        spawn_init_markers=True,
        refine_handle_collisions=not args.no_refine,   # spawn 时把把手 link 设 convexDecomposition(物理 init 前，不破坏 view)
        apply_saved_camera_offsets=use_cam,
        control_hz=args.control_hz, reset_mode=ResetMode.STATIC, seed=args.seed,
        extra_assets=extra_assets,
        add_robot_stand=not args.no_robot_stand,
        # 默认删除冰箱(microwave 成员)+洗碗机(用户要求)；--keep_fridge / --keep_dishwasher 可保留。
        exclude_members=tuple(
            ([] if args.keep_fridge else ["microwave"]) + ([] if args.keep_dishwasher else ["dishwasher"])),
        # 静态场景：关掉物体随机化 + 自动 reset(不随机跳位、不自动重置)。
        # 三个方块(cube_1蓝/cube_2红/cube_3绿)放在桌面上(位置来自 manifest，可在 Asset Pose 面板微调)；
        # 不再隐藏它们。小刀(knife)同样保留在桌面。
        disable_auto_reset=True,
        hidden_members=(),
    )
    session = SceneSession.launch(cfg, _app_launcher=app_launcher)   # 内部 ensure_legacy_on_path
    env, provider, env_cfg = session.env, session.provider, session.env_cfg

    # 启动后把保存的相机局部位姿设回 camera prim（所见即所得，不经 offset/convention）
    if use_cam and cfg.apply_saved_camera_offsets:
        try:
            from franka_v1_skill_lab.scene_interface import camera_offsets

            camera_offsets.apply_saved_offsets_runtime(env)
        except Exception as exc:  # pragma: no cover
            print(f"[test_mode] WARNING: apply saved camera offsets failed: {exc}", flush=True)

    # --- AppLauncher + legacy 之后才 import（顶层带 isaac）---
    from franka_v1_skill_lab.skill_runtime._legacy import ensure_legacy_on_path

    ensure_legacy_on_path()
    from runtime.ik_joint_adapter import IKJointAdapter

    from franka_v1_skill_lab.scene_interface import camera_offsets, scene_saver
    from franka_v1_skill_lab.scene_interface.asset_pose_panel import AssetPosePanel
    from franka_v1_skill_lab.scene_interface.camera_panel import CameraPanel
    from franka_v1_skill_lab.scene_interface.grasp_pose_panel import GraspPosePanel
    from franka_v1_skill_lab.scene_interface.skill_waypoints_panel import SkillWaypointsPanel
    from franka_v1_skill_lab.scene_interface.franka_control_panel import FrankaControlPanel
    from franka_v1_skill_lab.scene_interface.image_publisher import ImagePublisher
    from franka_v1_skill_lab.scene_interface.joint_driver_ui import (
        JointDriver, JointDriverWindow, _enable_collision_vis,
    )
    from franka_v1_skill_lab.scene_interface.reset_strategies import build_reset_strategy

    adapter = IKJointAdapter(env)
    # 碰撞体可视化：默认关闭，仅 --pengzhuang 开启（--no_collision_vis 仍可强制关）。
    show_coll = args.pengzhuang and not args.no_collision_vis
    if not args.headless and show_coll:
        _enable_collision_vis(True)

    # 按 --ui / 单独开关 选择加载哪些面板（headless 时一律不建 UI）。
    # 单独开关(--grasp/--scene/...)优先：写了任意一个就【只加】这些；都没写才看 --ui(默认 all)。
    _individual = {k for k in _UI_KEYS if getattr(args, k, False)}
    ui_set = set() if args.headless else (_individual or _parse_ui_selection(args.ui))

    def _want(key: str) -> bool:
        return (not args.headless) and (key in ui_set)

    franka = FrankaControlPanel(env, provider, adapter) if _want("franka") else None
    asset_panel = AssetPosePanel(env, provider) if _want("asset") else None
    grasp_panel = (GraspPosePanel(env, provider, headless=args.headless, franka_panel=franka)
                   if _want("grasp") else None)
    wp_panel = SkillWaypointsPanel(env, provider, headless=args.headless) if _want("waypoints") else None
    arts = {n: provider.scene[n] for n in ("cabinet", "microwave", "coffee_machine", "dishwasher") if n in provider.scene.keys()}
    jdriver = JointDriver(arts)
    jwindow = JointDriverWindow(jdriver, collision_on=show_coll) if _want("joint") else None
    cam_panel = CameraPanel() if (use_cam and _want("camera")) else None
    scene_panel = _ScenePanel() if _want("scene") else None
    viz_panel = _VizPanel(session, show_colliders=show_coll) if _want("viz") else None
    if not args.headless:
        print(f"[test_mode] UI panels loaded: {sorted(ui_set) or '(none)'}", flush=True)

    # GPT 场景描述面板（--describe）：点按钮把两路 RGB + 状态发给 VLM server，拿回物品描述/关系。
    describe_panel = None
    if args.describe and not args.headless and use_cam:
        try:
            from franka_v1_skill_lab.scene_describer.describe_panel import DescribePanel

            describe_panel = DescribePanel(session, gpt_addr=args.describe_addr, qwen_addr=args.qwen_addr)
            print(f"[test_mode] perception panel on -> GPT {args.describe_addr} | Qwen {args.qwen_addr}",
                  flush=True)
        except Exception as exc:  # pragma: no cover
            print(f"[test_mode] WARNING: describe panel init failed: {exc}", flush=True)
    elif args.describe and not use_cam:
        print("[test_mode] WARNING: --describe needs cameras; ignored (--no_cameras set).", flush=True)

    # 外部测试控制器（状态机 / VLA 等模块把控制接进来）。它持有 session，每帧 step(session)->action|None。
    controller = _load_controller(args.controller, session) if args.controller else None
    # 把 Grasp Pose 面板接给控制器：抓取目标列表 + 抓取位姿用面板的【实时】值（build_window 前接好）。
    if controller is not None and grasp_panel is not None and hasattr(controller, "attach_grasp_panel"):
        try:
            controller.attach_grasp_panel(grasp_panel)
        except Exception as exc:  # pragma: no cover
            print(f"[test_mode] controller.attach_grasp_panel failed: {exc}", flush=True)
    # 把技能途径点面板接给控制器：起技能前按面板里编辑的途径点绕行(避开已打开的抽屉等)。
    if controller is not None and wp_panel is not None and hasattr(controller, "attach_waypoints_panel"):
        try:
            controller.attach_waypoints_panel(wp_panel)
        except Exception as exc:  # pragma: no cover
            print(f"[test_mode] controller.attach_waypoints_panel failed: {exc}", flush=True)
    if controller is not None and hasattr(controller, "build_window"):
        try:
            controller.build_window()
        except Exception as exc:  # pragma: no cover
            print(f"[test_mode] controller.build_window failed: {exc}", flush=True)

    # 把 GUI 视口对准当前场景：以机器人为焦点(场景中心)，从斜后上方看过去，进去就框住整个场景。
    if not args.headless:
        try:
            rb = provider.scene["robot"].data.root_pos_w[0].detach().cpu().tolist()
            target = (float(rb[0]), float(rb[1]), 0.4)
            eye = (float(rb[0]) - 2.0, float(rb[1]) - 2.0, float(rb[2]) + 1.8)
            env.unwrapped.sim.set_camera_view(eye, target)
            print(f"[test_mode] viewport framed on scene @ robot {[round(v,2) for v in rb]}", flush=True)
        except Exception as exc:  # pragma: no cover
            print(f"[test_mode] WARNING: set viewport view failed: {exc}", flush=True)

    publisher = None
    if use_cam and not args.no_stream:
        try:
            publisher = ImagePublisher(addr=args.stream_addr)
        except Exception as exc:  # pragma: no cover
            print(f"[test_mode] WARNING: image publisher init failed: {exc}", flush=True)

    region_reset = build_reset_strategy(ResetMode.REGION, args.seed)
    reset_index = 0

    def _settle(n: int = 5):
        provider.reset_cabinet_joint("joint_0", 0.0)
        for _ in range(n):
            env.step(provider.make_hold_joint_action(provider.get_state(), 1.0))

    def _reset_to_initial(idx: int = 0):
        """场景重置：所有物体(方块/刀/道具/家电/底座)回到初始(保存场景)位置，家电关节归零(抽屉/门关上)。

        不随机化。env.reset() 后逐个把刚体/铰接体写回 cfg 的 init_state(覆盖任何随机化事件)，
        家电再写回默认关节(关上)。机器人由 env.reset 复位，跳过避免冲突。"""
        import torch

        env.reset(seed=args.seed)
        scene = env.unwrapped.scene
        dev = env.unwrapped.device
        env_ids = torch.tensor([0], device=dev)
        try:
            origin = scene.env_origins[0]
        except Exception:
            origin = torch.zeros(3, device=dev)
        scfg = getattr(getattr(session, "env_cfg", None), "scene", None)
        zero6 = torch.zeros((1, 6), device=dev)
        for name in list(scene.keys()):
            name = str(name)
            if name == "robot":            # 机器人由 env.reset 复位，避免冲突
                continue
            member = getattr(scfg, name, None) if scfg is not None else None
            init = getattr(member, "init_state", None) if member is not None else None
            asset = scene[name]
            if init is None or not hasattr(asset, "write_root_pose_to_sim"):
                continue
            try:
                pos = torch.tensor([float(v) for v in init.pos], dtype=torch.float32, device=dev) + origin
                rot = torch.tensor([float(v) for v in getattr(init, "rot", (1.0, 0.0, 0.0, 0.0))],
                                   dtype=torch.float32, device=dev)
                root = torch.cat((pos.reshape(1, 3), rot.reshape(1, 4)), dim=-1)
                asset.write_root_pose_to_sim(root, env_ids=env_ids)
                if hasattr(asset, "write_root_velocity_to_sim"):
                    asset.write_root_velocity_to_sim(zero6, env_ids=env_ids)
                if hasattr(asset, "write_joint_state_to_sim") and hasattr(asset.data, "default_joint_pos"):
                    asset.write_joint_state_to_sim(asset.data.default_joint_pos, asset.data.default_joint_vel,
                                                   env_ids=env_ids)
            except Exception as exc:  # pragma: no cover
                print(f"[test_mode] reset '{name}' failed: {exc}", flush=True)
                continue
        provider.reset_cabinet_joint("joint_0", 0.0)
        # 机器人复位到最新保存的 home(robot_home.json)：env.reset 不会把 default 写进真实关节
        # (实测会停在≈0)，且上面跳过了 robot，所以这里显式写 home，和 SceneSession.reset 一致。
        # 否则进场景机器人不在 home，要手点 UI 的 Home 按钮才过去。
        try:
            session._apply_saved_home()
        except Exception as exc:  # pragma: no cover
            print(f"[test_mode] apply saved home failed: {exc}", flush=True)
        if asset_panel is not None:
            asset_panel.clear()
        _settle()
        print("[test_mode] scene reset -> all objects restored to initial positions "
              "(robot at saved home).", flush=True)

    def _do_save():
        robot = provider.scene["robot"]
        try:   # 机器人复位 default（姿态不保存）
            robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        except Exception:
            pass
        for _ in range(2):
            env.step(provider.make_hold_joint_action(provider.get_state(), 1.0))
        sensors = None
        if cam_panel is not None:
            try:
                # 从【实际相机 prim】读当前位姿(而不是 cam_panel 的内部缓存)——这样无论你用相机面板
                # 按钮还是直接在视口里挪相机，存盘记录的都是真实当前视角，不会和 manifest 里的 prim
                # xform 不一致导致重载后被旧默认值覆盖。
                import omni.usd as _ou
                _stage = _ou.get_context().get_stage()
                sensors = camera_offsets.offsets_to_sensors(camera_offsets.current_offsets_from_stage(_stage))
            except Exception as exc:  # pragma: no cover
                print(f"[test_mode] WARNING: build camera sensors failed: {exc}", flush=True)
        # 一起把【抓取位姿】存盘(grasp_poses.json)：保存场景时把修改后的目标 pose 一并持久化，
        # 状态机/技能下次启动从存盘读取(GraspPosePanel.__init__ load -> 控制器用它的实时值)。
        if grasp_panel is not None:
            try:
                grasp_panel.save()
            except Exception as exc:  # pragma: no cover
                print(f"[test_mode] WARNING: save grasp poses failed: {exc}", flush=True)
        try:
            info = scene_saver.save_current_stage_as_latest(env=env, note="Saved by test_mode_ui.", sensors=sensors)
            msg = f"saved -> {Path(info['usd']).name} ({len(info['objects'])} objects) + grasp poses"
            print(f"[test_mode] {msg}", flush=True)
        except Exception as exc:
            msg = f"SAVE FAILED: {exc}"
            print(f"[test_mode] {msg}", flush=True)
        if scene_panel is not None:
            scene_panel.set_status(msg)

    sim_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    sim_time = 0.0
    provider.set_sim_time(sim_time)
    _reset_to_initial(reset_index)
    _extra_ui = ([f"controller={args.controller}"] if controller is not None else []) + \
                (["describe"] if describe_panel is not None else [])
    print(f"[test_mode] ready. UI panels: {sorted(ui_set) or '(none)'}"
          + (f" + {_extra_ui}" if _extra_ui else ""), flush=True)

    while simulation_app.is_running():
        with torch.inference_mode():
            provider.set_sim_time(sim_time)
            state = provider.get_state()

            intent = franka.compute(state) if franka is not None else None

            if scene_panel is not None:
                if scene_panel.pop_reset():
                    reset_index += 1
                    _reset_to_initial(reset_index)
                    if controller is not None and hasattr(controller, "on_reset"):
                        try:
                            controller.on_reset()
                        except Exception:
                            pass
                    state = provider.get_state()
                if scene_panel.pop_save():
                    _do_save()
                    state = provider.get_state()

            # 机器人动作优先级：外部控制器 > UI#1 手动 IK/关节 > 保持
            ctrl_action = None
            if controller is not None:
                try:
                    ctrl_action = controller.step(session)
                except Exception as exc:  # pragma: no cover
                    print(f"[test_mode] controller.step failed: {exc}", flush=True)
                    ctrl_action = None
            if ctrl_action is not None:
                actions = ctrl_action
            elif intent is not None:
                actions = provider.make_joint_action_from_q_des(intent.q_des, intent.gripper)
            else:
                g = franka.current_gripper() if franka is not None else None
                actions = provider.make_hold_joint_action(state, g)

            if jwindow is not None:       # 家电关节目标（抽屉/门/把手）
                jwindow.apply()
            if asset_panel is not None:   # 钉住被点动的资产（微波炉+台子联动）
                asset_panel.apply()
            env.step(actions)

            if grasp_panel is not None:   # 画每个物体的抓取位姿 marker（读 step 后的物体世界位姿）
                grasp_panel.apply()
            if wp_panel is not None:      # 画选中技能的途径点 marker
                wp_panel.apply()

            # UI#1 当前 IK 目标箭头（受 Visualization 面板的 "Show target arrows" 控制）
            show_arrows = viz_panel.show_arrows if viz_panel is not None else True
            tp = franka.active_target_pose() if (franka is not None and show_arrows) else None
            if tp is not None:
                session.show_target_poses([tp], names=["ui1_target"])
            else:
                session.clear_target_poses()
            if franka is not None:
                franka.update_status()
            if asset_panel is not None:
                asset_panel.update_status()
            if grasp_panel is not None:
                grasp_panel.update_status()
            if jwindow is not None:
                jwindow.update()
            if cam_panel is not None:
                cam_panel.update_status()
            if describe_panel is not None:    # 处理「描述当前帧」按钮 -> 抓帧+后台发 GPT
                describe_panel.update()

            # 发图给独立查看器（节流）：front/wrist 的 RGB + colorized depth
            if publisher is not None and publisher.due():
                try:
                    obs = session.observe()
                    for view in ("front", "wrist"):
                        cv = obs.cameras.get(view)
                        if cv is None:
                            continue
                        publisher.send_rgb(f"{view}_rgb", cv.rgb)
                        publisher.send_depth(f"{view}_depth", cv.depth)
                except Exception as exc:  # pragma: no cover
                    print(f"[test_mode] publish WARNING: {exc}", flush=True)

            sim_time += sim_dt

    if publisher is not None:
        publisher.close()
    session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
