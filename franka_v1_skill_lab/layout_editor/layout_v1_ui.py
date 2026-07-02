# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Franka V1 layout editor — STATUS: ready (needs GPU / Isaac Sim to run).

Adapted from ``SceneLayoutModule/scene_layout_ui.py``. Same idea (load the base
task scene or the latest saved scene, keep the robot still, let the user move /
add prims, then save), but specialised for the V1 contract:

  * loads the V1 base task ``Isaac-Stack-Cube-Franka-JointPolicy-v0`` by default;
  * can re-open the latest saved V1 USD with ``--load_latest_v1``;
  * "Save V1" writes the CANONICAL names ``scene_v1_latest.usd`` +
    ``scene_v1_latest.json`` into ``scene/saved_scenes/v1_active/`` (and a
    timestamped backup), then updates ``scene_v1_registry.json`` so every
    downstream consumer immediately sees the new active scene.

Run:
    ./isaaclab.sh -p projects/franka_v1_skill_lab/layout_editor/layout_v1_ui.py \
        --num_envs 1 --task Isaac-Stack-Cube-Franka-JointPolicy-v0

See layout_editor/README.md for the V0 -> V1 workflow.
"""

from __future__ import annotations

"""Launch Omniverse Toolkit first."""

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

# Make the V1 scene-contract package importable (projects/ on path).
_PROJECTS_DIR = Path(__file__).resolve().parents[2]   # projects/
sys.path.insert(0, str(_PROJECTS_DIR))

parser = argparse.ArgumentParser(description="Franka V1 scene layout editor.")
parser.set_defaults(disable_fabric=True)
parser.add_argument("--disable_fabric", action="store_true", dest="disable_fabric", help="Use USD I/O for layout editing.")
parser.add_argument("--enable_fabric", action="store_false", dest="disable_fabric", help="Enable Fabric for advanced debugging.")
parser.add_argument("--disable_collision_debug_vis", action="store_true", default=False, help="Disable collider overlays.")
parser.add_argument("--show_colliders", action="store_true", default=False,
                    help="Show collider overlays at startup. Default OFF (toggle anytime with the 'Show colliders' checkbox).")
parser.add_argument("--show_init_region", action="store_true", default=False,
                    help="Show the 4 yellow InitCorner markers (cube random-init rectangle). Default HIDDEN.")
parser.add_argument(
    "--property_edit",
    action="store_true",
    default=False,
    help="Minimal STATIC property-editing mode: load the current saved scene, edit assets via Isaac's "
    "native Property panel (position/orient/scale), and the UI shows ONLY Save + a collision-viz toggle "
    "(no robot control, no add/move UI, no reach/init markers). Implies --load_latest_v1 if no scene given.",
)
parser.add_argument("--seed", type=int, default=1, help="Deterministic scene seed.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. This layout UI supports 1.")
parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-JointPolicy-v0", help="V1 base task id.")
parser.add_argument("--load_usd", type=str, default=None, help="Open a specific saved USD instead of the base task.")
parser.add_argument(
    "--load_latest_v1",
    "--load_latest_saved",
    dest="load_latest_v1",
    action="store_true",
    default=False,
    help="Open the active V1 USD from the registry (alias: --load_latest_saved).",
)
parser.add_argument(
    "--enable_wrist_cameras",
    "--enable_wrist_d435",
    dest="enable_wrist_cameras",
    action="store_true",
    default=False,
    help="Attach the two wrist cameras (foundationpose_d435_rgbd + vla_libero_eye_in_hand). "
    "The D435 body mesh is hidden by default; cameras render only with --enable_cameras. "
    "(alias: --enable_wrist_d435).",
)
parser.add_argument(
    "--show_camera_body",
    action="store_true",
    default=False,
    help="Make the D435 housing mesh visible (default: hidden).",
)
parser.add_argument(
    "--hide_camera_body",
    action="store_true",
    default=False,
    help="Force-hide the D435 housing mesh (default behaviour; explicit for the docs' command).",
)
parser.add_argument(
    "--vla_camera_resolution",
    type=int,
    nargs=2,
    default=None,
    metavar=("W", "H"),
    help="VLA eye-in-hand resolution, e.g. --vla_camera_resolution 128 128 or 224 224.",
)
parser.add_argument(
    "--no_depth",
    action="store_true",
    default=False,
    help="When cameras render, disable the FoundationPose camera depth stream (RGB only).",
)
parser.add_argument(
    "--registry",
    type=str,
    default=None,
    help="Path to scene_v1_registry.json (default: scene/saved_scenes/v1_active/scene_v1_registry.json).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# property-edit 模式：没指定加载哪个场景时，默认加载当前最新保存场景(= test_mode 里看到的完整场景)。
if args_cli.property_edit and not args_cli.load_usd and not args_cli.load_latest_v1:
    args_cli.load_latest_v1 = True

# Pass the FULL parsed namespace (not just headless) so app-launcher flags like
# --enable_cameras actually take effect. Without this, --enable_wrist_d435 +
# --enable_cameras would spawn the D435 CameraCfg while rendering stays off,
# which Isaac rejects with "A camera was spawned without the --enable_cameras flag".
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import carb  # noqa: E402
import gymnasium as gym  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab.sim.utils.stage import open_stage  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

from franka_v1_skill_lab.scene import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    SCHEMA_VERSION,
    V1_OBJECTS,
    authorize_scene_write,
    revoke_scene_write,
    update_active_scene,
)
from franka_v1_skill_lab.scene.scene_registry import V1_ACTIVE_DIR  # noqa: E402
from franka_v1_skill_lab.sensors.d435 import d435_config as d435cfg  # noqa: E402
from franka_v1_skill_lab.sensors.d435.d435_scene_cfg import attach_wrist_cameras  # noqa: E402


def _cameras_render_enabled() -> bool:
    """True if the rendered cameras should be added (needs --enable_cameras)."""
    return bool(args_cli.enable_wrist_cameras and getattr(args_cli, "enable_cameras", False))


def _show_body() -> bool:
    """D435 housing visible? Hidden by default; --show_camera_body overrides."""
    if args_cli.hide_camera_body:
        return False
    return bool(args_cli.show_camera_body)


def _vla_resolution() -> tuple[int, int]:
    if args_cli.vla_camera_resolution:
        return int(args_cli.vla_camera_resolution[0]), int(args_cli.vla_camera_resolution[1])
    return d435cfg.VLA_PROFILE.width, d435cfg.VLA_PROFILE.height


def _camera_sensors_block(stage=None) -> dict:
    """Sensors block (all three cameras) to stamp into the saved registry.

    stage 给定时：用【当前 stage 上相机 prim 的局部位姿】覆盖各相机的 offset。否则用的是 profile
    里写死的默认 offset —— 那会把用户在 viewport 里调好的相机姿态又冲回默认（静态编辑器一保存
    前视相机就复位到最初位置的 bug 根因）。test_mode 的 _do_save 早就这么做了，这里补上。
    """
    vw, vh = _vla_resolution()
    enabled = bool(args_cli.enable_wrist_cameras)
    block = {
        d435cfg.FP_CAM_NAME: d435cfg.FP_PROFILE.descriptor(enabled=enabled, visible_body=_show_body()),
        d435cfg.VLA_CAM_NAME: d435cfg.VLA_PROFILE.descriptor(
            enabled=enabled, visible_body=_show_body(), width=vw, height=vh
        ),
        d435cfg.FRONT_CAM_NAME: d435cfg.FRONT_PROFILE.descriptor(enabled=enabled, visible_body=False),
    }
    if stage is not None:
        try:
            from franka_v1_skill_lab.scene_interface import camera_offsets

            offs = camera_offsets.current_offsets_from_stage(stage)

            def _ovr(name, pose):
                if name in block and pose is not None:
                    pos, quat = pose
                    conv = (block[name].get("offset") or {}).get("convention", "opengl")
                    block[name]["offset"] = {
                        "pos": [float(v) for v in pos],
                        "rot_wxyz": [float(v) for v in quat],
                        "convention": conv,
                    }
            if "front" in offs:
                _ovr(d435cfg.FRONT_CAM_NAME, offs["front"])
            if "wrist" in offs:
                _ovr(d435cfg.FP_CAM_NAME, offs["wrist"])
                _ovr(d435cfg.VLA_CAM_NAME, offs["wrist"])
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[layout_v1] live camera offset capture failed (using profile defaults): {exc}", flush=True)
    return block


def _refresh_wrist_camera_viewport(env) -> None:
    """Force the viewport to re-read the wrist-camera transforms.

    Replicates the manual fix (nudge a camera's orient by a hair, then set it
    back): with the timeline paused the viewport caches the camera pose and only
    re-reads it when the prim's transform attribute is RE-AUTHORED. Re-rendering
    alone is not enough. For each wrist camera we perturb its ``xformOp:orient``
    by 1e-5, render, then restore the exact value — emitting the USD change
    notification that snaps the camera viewport to the configured pose.
    """
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    prim_paths = [
        d435cfg.FP_CAM_PRIM.replace("{ENV_REGEX_NS}", ENV_NS),
        d435cfg.VLA_CAM_PRIM.replace("{ENV_REGEX_NS}", ENV_NS),
        d435cfg.FRONT_CAM_PRIM.replace("{ENV_REGEX_NS}", ENV_NS),
    ]
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            print(f"[layout_v1] refresh: camera prim not found yet: {path}", flush=True)
            continue
        xform = UsdGeom.Xformable(prim)
        op = next((o for o in xform.GetOrderedXformOps() if "orient" in o.GetName()), None)
        if op is None:
            continue
        q = op.Get()
        if q is None:
            continue
        q_type = type(q)            # Gf.Quatf or Gf.Quatd (preserve precision)
        im = q.GetImaginary()
        perturbed = q_type(q.GetReal() + 1.0e-5, im[0], im[1], im[2])
        op.Set(perturbed)
        env.unwrapped.sim.render()
        simulation_app.update()
        op.Set(q)                   # restore the exact configured orientation

ENV_NS = "/World/envs/env_0"
DEFAULT_DYNAMIC_ASSETS = {
    "CoffeeMachine_103046": "SapienAssetPipeline/usd_assets/CoffeeMachine_103046/coffeemachine.usd",
    "Cabinet_44853": "SapienAssetPipeline/usd_assets/Cabinet_44853/cabinet.usd",
    "Fridge_12252": "SapienAssetPipeline/usd_assets/Fridge_12252/fridge.usd",
    "Knife_101054": "SapienAssetPipeline/usd_assets/Knife_101054/knife.usd",
    "Microwave_7320": "SapienAssetPipeline/usd_assets/Microwave_7320/microwave_referenceable.usd",
}

ACTIVE_USD_NAME = "scene_v1_latest.usd"
ACTIVE_MANIFEST_NAME = "scene_v1_latest.json"
# 上一版本的固定名备份（覆盖 latest 之前先把旧的 latest 退避到这里，方便回滚）
PREVIOUS_USD_NAME = "scene_v1_previous.usd"
PREVIOUS_MANIFEST_NAME = "scene_v1_previous.json"
# 与最新 USD 同级目录的“如何复现”文档（保存时自动生成）
REPRODUCE_DOC_NAME = "scene_v1_latest.reproduce.md"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _registry_path() -> Path:
    return Path(args_cli.registry).resolve() if args_cli.registry else DEFAULT_REGISTRY_PATH


def _active_dir() -> Path:
    return _registry_path().parent if args_cli.registry else V1_ACTIVE_DIR


def set_collision_debug_visualization(on: bool) -> None:
    """开/关碰撞体可视化叠加（2=显示所有 collider，0=关）。"""
    settings = carb.settings.get_settings()
    settings.set_int("/persistent/physics/visualizationDisplayColliders", 2 if on else 0)
    settings.set_bool("/persistent/physics/visualizationDisplayColliderNormals", False)


def enable_collision_debug_visualization() -> None:
    set_collision_debug_visualization(True)


def _timestamp() -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{int((time.time() % 1) * 1000):03d}"


def _safe_prim_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def _discover_usd_assets() -> dict[str, Path]:
    repo = _repo_root()
    assets: dict[str, Path] = {}
    for name, relative in DEFAULT_DYNAMIC_ASSETS.items():
        path = repo / relative
        if path.is_file():
            assets[name] = path
    for usd_file in sorted((repo / "SapienAssetPipeline" / "usd_assets").glob("*/*.usd")):
        assets.setdefault(usd_file.parent.name, usd_file)
    return assets


def _pause_timeline_for_layout() -> None:
    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        if hasattr(timeline, "pause"):
            timeline.pause()
        else:
            timeline.stop()


def _next_free_prim_path(stage: Usd.Stage, base_path: str) -> str:
    if not stage.GetPrimAtPath(base_path).IsValid():
        return base_path
    index = 1
    while stage.GetPrimAtPath(f"{base_path}_{index:02d}").IsValid():
        index += 1
    return f"{base_path}_{index:02d}"


def _add_usd_reference(asset_name: str, usd_path: Path) -> str:
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open.")
    base_name = _safe_prim_name(asset_name)
    prim_path = _next_free_prim_path(stage, f"{ENV_NS}/{base_name}")
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(str(usd_path.resolve()))
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.45, 0.25, 0.05))
    xform.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    xform.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    print(f"[layout_v1] added {usd_path} -> {prim_path}", flush=True)
    return prim_path


INIT_REGION_DEFAULT = [(0.40, -0.10), (0.60, -0.10), (0.60, 0.10), (0.40, 0.10)]
INIT_MARKER_Z = 0.03
INIT_MARKER_NAMES = [f"InitCorner_{i}" for i in range(4)]


def _spawn_reach_workspace():
    """Draw the Franka reachable-workspace as circle outlines (BasisCurves) centered on the base, at the
    ground/cube height and the handle working height, so the user can judge if objects are in reach.

    Franka Panda max reach ~0.85 m from the shoulder (~z=0.33); 'comfortable' grasp reach ~0.65 m (the
    door tests succeeded around 0.6 m, failed at 0.83 m). Rings: ORANGE = max, GREEN = comfortable.
    """
    import math as _m

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    scope = "/World/Visuals/ReachWorkspace"
    rings = [  # (name, z, radius, color)
        ("ground_max", 0.05, 0.78, Gf.Vec3f(1.0, 0.45, 0.0)),
        ("ground_ok", 0.05, 0.62, Gf.Vec3f(0.1, 0.9, 0.2)),
        ("handle_max", 0.45, 0.82, Gf.Vec3f(1.0, 0.45, 0.0)),
        ("handle_ok", 0.45, 0.65, Gf.Vec3f(0.1, 0.9, 0.2)),
    ]
    n = 72
    for name, z, r, color in rings:
        path = f"{scope}/{name}"
        if stage.GetPrimAtPath(path).IsValid():
            continue
        curve = UsdGeom.BasisCurves.Define(stage, path)
        curve.CreateTypeAttr().Set("linear")
        curve.CreateWrapAttr().Set("periodic")
        pts = [Gf.Vec3f(r * _m.cos(2 * _m.pi * i / n), r * _m.sin(2 * _m.pi * i / n), z) for i in range(n)]
        curve.CreatePointsAttr(pts)
        curve.CreateCurveVertexCountsAttr([n])
        curve.CreateWidthsAttr([0.01] * n)
        curve.SetWidthsInterpolation("vertex")
        curve.CreateDisplayColorAttr([color])
    print("[layout_v1] reach workspace drawn: ORANGE=max(~0.8m) GREEN=comfortable(~0.65m), at ground & "
          "handle height. Keep the door handle / cubes inside the GREEN ring for reliable grasping.",
          flush=True)


def _spawn_init_region_markers():
    """Spawn 4 draggable corner markers (bright yellow cubes) that define the cube random-init
    rectangle. Drag them in the viewport; Save V1 records their world positions, from which the cube
    randomization x/y range is rebuilt. If markers already exist (loaded from a saved scene), keep them."""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    for i, (x, y) in enumerate(INIT_REGION_DEFAULT):
        p = f"{ENV_NS}/{INIT_MARKER_NAMES[i]}"
        if stage.GetPrimAtPath(p).IsValid():
            continue
        cube = UsdGeom.Cube.Define(stage, p)
        cube.GetSizeAttr().Set(0.04)
        cube.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.85, 0.0)])
        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), INIT_MARKER_Z))
    print("[layout_v1] init-region markers ready: drag InitCorner_0..3 (yellow) to set the cube "
          "random-init rectangle; Save V1 records them.", flush=True)


def _hide_init_markers(stage, visible: bool = False) -> int:
    """默认隐藏 4 个 InitCorner 黄方块(用户暂时不需要)。仍保留 prim(保存/区域采样不受影响),
    只切换可见性。visible=True(--show_init_region)时显示。返回处理的 marker 数。"""
    if stage is None:
        return 0
    n = 0
    for nm in INIT_MARKER_NAMES:
        prim = stage.GetPrimAtPath(f"{ENV_NS}/{nm}")
        if not prim.IsValid():
            continue
        img = UsdGeom.Imageable(prim)
        img.MakeVisible() if visible else img.MakeInvisible()
        n += 1
    if n:
        print(f"[layout_v1] InitCorner markers {'shown' if visible else 'hidden'} ({n}).", flush=True)
    return n


def _author_robot_joint_positions(stage, robot_path: str, home) -> int:
    """把 home 的 7 个手臂关节角写进机器人 USD 关节(drive target + PhysxJointState, 单位=度)。

    这样保存到 USD 后,重新加载/物理初始化时机器人从这些关节角摆位(= home),与测试模式一致。
    返回成功写入的关节数。"""
    import math

    from pxr import PhysxSchema, Usd, UsdPhysics

    name_to_val = {f"panda_joint{i + 1}": float(home[i]) for i in range(7)}
    robot = stage.GetPrimAtPath(robot_path)
    if not robot.IsValid():
        print(f"[layout_v1] robot prim not found: {robot_path}", flush=True)
        return 0
    n = 0
    for p in Usd.PrimRange(robot):
        nm = p.GetName()
        if nm not in name_to_val or p.GetTypeName() != "PhysicsRevoluteJoint":
            continue
        deg = math.degrees(name_to_val[nm])   # USD 角度关节单位是【度】
        try:
            drive = UsdPhysics.DriveAPI.Get(p, "angular")
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(p, "angular")
            ta = drive.GetTargetPositionAttr()
            (ta.Set(deg) if ta else drive.CreateTargetPositionAttr(deg))
            js = PhysxSchema.JointStateAPI.Get(p, "angular")
            if not js:
                js = PhysxSchema.JointStateAPI.Apply(p, "angular")
            pa = js.GetPositionAttr()
            (pa.Set(deg) if pa else js.CreatePositionAttr(deg))
            n += 1
        except Exception as exc:  # pragma: no cover
            print(f"[layout_v1] author joint {nm} failed: {exc}", flush=True)
    return n


def _set_robot_home(stage, env=None) -> None:
    """机器人默认停到最新保存的 home(robot_home.json)。

    - 两条路径都把 home 写进 USD 关节(drive target + joint state) -> 保存/重载/测试一致;
    - base task 路径(env 存在)再用 write_joint_state_to_sim 立刻写进物理 -> 编辑器里当场可见。
    没保存过 home 则不动(保持场景原姿态)。"""
    from franka_v1_skill_lab.scene_interface.robot_home import saved_home_q

    home = saved_home_q()
    if home is None:
        print("[layout_v1] no saved robot home (robot_home.json); robot keeps scene pose.", flush=True)
        return
    n = _author_robot_joint_positions(stage, f"{ENV_NS}/Robot", home)
    if env is not None:
        try:
            import torch

            robot = env.unwrapped.scene["robot"]
            jn = list(robot.data.joint_names)
            ids = [jn.index(f"panda_joint{i + 1}") for i in range(7)]
            pos = robot.data.joint_pos[0:1].clone()
            vel = robot.data.joint_vel[0:1].clone()
            for k, jid in enumerate(ids):
                pos[0, jid] = float(home[k])
                vel[0, jid] = 0.0
            env_ids = torch.tensor([0], device=robot.device)
            robot.write_joint_state_to_sim(pos, vel, env_ids=env_ids)
            robot.set_joint_position_target(pos, env_ids=env_ids)
            env.unwrapped.sim.render()
        except Exception as exc:  # pragma: no cover
            print(f"[layout_v1] write robot home to sim failed: {exc}", flush=True)
    print(f"[layout_v1] robot home applied (authored {n} USD joints"
          f"{', + live sim' if env is not None else ''}) <- robot_home.json "
          f"{tuple(round(v, 4) for v in home)}.", flush=True)


def _value_to_json(value):
    if value is None:
        return None
    if isinstance(value, (Gf.Vec3d, Gf.Vec3f, Gf.Vec3h)):
        return [float(value[0]), float(value[1]), float(value[2])]
    if isinstance(value, (Gf.Quatd, Gf.Quatf, Gf.Quath)):
        return [float(value.GetReal()), *[float(item) for item in value.GetImaginary()]]
    try:
        return [float(item) for item in value]
    except TypeError:
        return str(value)


def _get_xform_ops(prim: Usd.Prim) -> dict:
    xform = UsdGeom.Xformable(prim)
    result = {"translate": [0.0, 0.0, 0.0], "orient_wxyz": [1.0, 0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]}
    for op in xform.GetOrderedXformOps():
        value = _value_to_json(op.Get())
        if op.GetName() == "xformOp:translate":
            result["translate"] = value
        elif op.GetName() == "xformOp:orient":
            result["orient_wxyz"] = value
        elif op.GetName() == "xformOp:scale":
            result["scale"] = value
    return result


def _get_world_transform(prim: Usd.Prim) -> dict:
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = matrix.ExtractTranslation()
    q = matrix.ExtractRotationQuat()
    return {
        "translation": [float(t[0]), float(t[1]), float(t[2])],
        "rotation_wxyz": [float(q.GetReal()), *[float(v) for v in q.GetImaginary()]],
    }


def _subtree_asset_refs(prim: Usd.Prim) -> list[str]:
    refs: list[str] = []
    for sub in Usd.PrimRange(prim):
        for spec in sub.GetPrimStack():
            for r in spec.referenceList.GetAddedOrExplicitItems():
                if r.assetPath:
                    refs.append(r.assetPath)
            for r in spec.payloadList.GetAddedOrExplicitItems():
                if r.assetPath:
                    refs.append(r.assetPath)
    # de-dup, keep order
    seen = set()
    out = []
    for a in refs:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def _iter_layout_prims(stage: Usd.Stage) -> list[Usd.Prim]:
    roots = []
    env_prim = stage.GetPrimAtPath(ENV_NS)
    if env_prim.IsValid():
        roots.extend(child for child in env_prim.GetChildren() if child.IsA(UsdGeom.Xformable))
    world_prim = stage.GetPrimAtPath("/World")
    if world_prim.IsValid():
        for child in world_prim.GetChildren():
            path = child.GetPath().pathString
            if path.startswith("/World/envs") or path.startswith("/World/Visuals"):
                continue
            if child.IsA(UsdGeom.Xformable):
                roots.append(child)
    # also stage-root appliances (e.g. /coffeemachine, /microwave_flattened)
    for child in stage.GetPseudoRoot().GetChildren():
        name = child.GetName()
        if name in {"World", "Render", "Replicator", "physicsScene", "GroundPlane", "light"}:
            continue
        if child.IsA(UsdGeom.Xformable):
            roots.append(child)
    return roots


def _save_v1(active_dir: Path) -> tuple[Path, Path]:
    """Export the stage to scene_v1_latest.usd + .json (plus timestamped backups)."""
    active_dir.mkdir(parents=True, exist_ok=True)
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open.")
    _pause_timeline_for_layout()

    # 导出前剥掉运行时调试 marker（抓取/末端坐标轴 /Visuals/SkillRuntime），否则会 bake 进 USD
    # 变成残留死几何（静态编辑器里能看到一堆已删物体的彩色坐标轴，且不跟随移动）。
    try:
        from franka_v1_skill_lab.scene_interface.scene_saver import strip_debug_markers
        strip_debug_markers(stage)
    except Exception as _exc:  # pragma: no cover
        print(f"[layout_v1] strip_debug_markers skipped: {_exc}", flush=True)

    # 把用户在【官方属性框】里改过的把手碰撞方块(位置/尺寸)读回来，回写给抓取技能：
    # 位置 -> grasp_poses.json(技能直接用)，尺寸 -> grasp_blocks.json(测试模式重生成时用)。所见即所得。
    try:
        from franka_v1_skill_lab.scene_interface.scene_props import (
            read_grasp_blocks_from_stage, save_grasp_blocks)
        _blocks = read_grasp_blocks_from_stage(stage, env_root=ENV_NS)
        if _blocks:
            save_grasp_blocks(_blocks)
    except Exception as _exc:  # pragma: no cover
        print(f"[layout_v1] persist grasp blocks skipped: {_exc}", flush=True)

    ts = _timestamp()
    # canonical + timestamped backup
    usd_canonical = (active_dir / ACTIVE_USD_NAME).resolve()
    usd_backup = (active_dir / f"scene_v1_{ts}.usd").resolve()
    json_canonical = (active_dir / ACTIVE_MANIFEST_NAME).resolve()

    # 覆盖 latest 之前，把上一版的 latest 退避成固定名 previous（防止改错可一键回滚）。
    # 用固定名，所以 previous 永远只保留“紧邻上一个版本”，与无限增长的时间戳备份互补。
    if usd_canonical.is_file():
        shutil.copy2(usd_canonical, active_dir / PREVIOUS_USD_NAME)
        if json_canonical.is_file():
            shutil.copy2(json_canonical, active_dir / PREVIOUS_MANIFEST_NAME)
        print(f"[layout_v1] 已备份上一版本 -> {PREVIOUS_USD_NAME} (+json)", flush=True)

    if not stage.Export(str(usd_canonical)):
        raise RuntimeError(f"Failed to export USD: {usd_canonical}")
    stage.Export(str(usd_backup))

    objects = []
    for prim in _iter_layout_prims(stage):
        objects.append(
            {
                "path": prim.GetPath().pathString,
                "name": prim.GetName(),
                "type": prim.GetTypeName(),
                "xform": _get_xform_ops(prim) if prim.IsA(UsdGeom.Xformable) else None,
                "world_transform": _get_world_transform(prim),
                "asset_refs": _subtree_asset_refs(prim),
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": args_cli.task,
        "control_mode": "joint",
        "env_root": ENV_NS,
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "contract_objects": list(V1_OBJECTS),
        "sensors": _camera_sensors_block(stage),
        "restore_contract": {
            "preferred_restore": "Load scene_v1_latest.usd directly.",
            "programmatic_restore": "Recreate each object at path; apply xform translate/orient_wxyz/scale.",
        },
        "objects": objects,
    }
    json_backup = (active_dir / f"scene_v1_{ts}.json").resolve()
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    json_canonical.write_text(payload, encoding="utf-8")
    json_backup.write_text(payload, encoding="utf-8")

    # 与最新场景同级，生成“如何复现”的中文文档（直接加载命令 + 程序化重建清单）。
    reproduce_path = _write_reproduce_doc(active_dir, usd_canonical, json_canonical, objects, manifest, ts)

    print(f"[layout_v1] 已保存 USD : {usd_canonical}", flush=True)
    print(f"[layout_v1] 已保存 JSON: {json_canonical}", flush=True)
    print(f"[layout_v1] 已生成复现文档: {reproduce_path}", flush=True)
    return usd_canonical, json_canonical


def _fmt_vec(values, digits: int = 5) -> str:
    if not values:
        return "n/a"
    return "(" + ", ".join(f"{float(v):.{digits}f}" for v in values) + ")"


def _repo_rel(path_str: str) -> str:
    """把绝对 asset 路径尽量转成相对仓库根的短路径，便于阅读/迁移。"""
    try:
        return str(Path(path_str).resolve().relative_to(_repo_root()))
    except Exception:
        return path_str


def _write_reproduce_doc(
    active_dir: Path,
    usd_canonical: Path,
    json_canonical: Path,
    objects: list[dict],
    manifest: dict,
    ts: str,
) -> Path:
    """在 active_dir 下写一份中文“如何复现最新场景”的 Markdown。

    复用 _save_v1 已经构建好的 objects（含 path / xform / asset_refs），不另外开 stage。
    内容包含：首选的直接加载命令、程序化重建说明、逐物体的 pos/rot/scale/asset 清单、
    外部 asset 依赖列表，以及回滚到 previous 的方法。
    """
    repo = _repo_root()
    try:
        usd_rel = usd_canonical.relative_to(repo)
    except ValueError:
        usd_rel = usd_canonical

    lines: list[str] = []
    lines.append("# 如何复现最新场景 (scene_v1_latest)")
    lines.append("")
    lines.append("> 本文件由 `layout_editor/layout_v1_ui.py` 在每次 **Save V1** 时自动生成，")
    lines.append("> 始终描述与它同级的 `scene_v1_latest.usd`。请勿手动编辑——下次保存会被覆盖。")
    lines.append("")
    lines.append(f"- 生成时间戳: `{ts}`")
    lines.append(f"- 最新场景 USD: `{usd_rel}`")
    lines.append(f"- 配套 manifest: `{json_canonical.name}`")
    lines.append(f"- 基础任务 (base task): `{manifest.get('task')}`")
    lines.append(f"- 控制模式: `{manifest.get('control_mode')}`  | up 轴: `{manifest.get('up_axis')}`"
                 f"  | meters_per_unit: `{manifest.get('meters_per_unit')}`")
    lines.append("")
    lines.append("## 方式一：直接加载最新 USD（推荐）")
    lines.append("")
    lines.append("```bash")
    lines.append("./isaaclab.sh -p projects/franka_v1_skill_lab/layout_editor/layout_v1_ui.py \\")
    lines.append("    --num_envs 1 --load_latest_saved")
    lines.append("```")
    lines.append("")
    lines.append("其他模块通过场景 registry 自动读取这同一个最新 USD：")
    lines.append("")
    lines.append("```python")
    lines.append("from franka_v1_skill_lab.scene.scene_registry import resolve_active_scene")
    lines.append("info = resolve_active_scene()   # -> {'usd': '.../scene_v1_latest.usd', ...}")
    lines.append("```")
    lines.append("")
    lines.append("## 方式二：程序化重建（USD 丢失/损坏时）")
    lines.append("")
    lines.append("在基础任务场景上，按下表对每个 prim 重新创建并施加 `translate / orient(wxyz) / scale`，")
    lines.append("引用对应的外部 asset。变换为相对各自父级的局部 xform（与 USD 中一致）。")
    lines.append("")
    lines.append("| prim 路径 | 类型 | 位置 (x,y,z) | 旋转 (w,x,y,z) | 缩放 | 外部 asset |")
    lines.append("|-----------|------|--------------|----------------|------|------------|")
    for obj in objects:
        xf = obj.get("xform") or {}
        pos = _fmt_vec(xf.get("translate"), 5)
        rot = _fmt_vec(xf.get("orient_wxyz"), 6)
        scale = _fmt_vec(xf.get("scale"), 5)
        refs = obj.get("asset_refs") or []
        asset = ", ".join(f"`{_repo_rel(r)}`" for r in refs) if refs else "—"
        lines.append(f"| `{obj.get('path')}` | {obj.get('type')} | {pos} | {rot} | {scale} | {asset} |")
    lines.append("")

    # 外部 asset 依赖汇总（去重）
    all_refs: list[str] = []
    seen = set()
    for obj in objects:
        for r in obj.get("asset_refs") or []:
            rel = _repo_rel(r)
            if rel not in seen:
                seen.add(rel)
                all_refs.append(rel)
    lines.append("## 外部 asset 依赖")
    lines.append("")
    if all_refs:
        lines.append("复现时以下 asset 必须可用（路径相对仓库根）：")
        lines.append("")
        for rel in all_refs:
            lines.append(f"- `{rel}`")
    else:
        lines.append("（本场景未引用外部 USD asset。）")
    lines.append("")
    lines.append("## 回滚到上一版本")
    lines.append("")
    lines.append("每次保存会先把旧的 latest 退避为固定名 previous。若本次改动有误，可回滚：")
    lines.append("")
    lines.append("```bash")
    lines.append(f"cd {active_dir}")
    lines.append(f"cp {PREVIOUS_USD_NAME} {ACTIVE_USD_NAME}")
    lines.append(f"cp {PREVIOUS_MANIFEST_NAME} {ACTIVE_MANIFEST_NAME}")
    lines.append("```")
    lines.append("")
    lines.append(f"> 时间戳历史备份 `scene_v1_<时间戳>.usd` 也在本目录，可加载任意一版。")
    lines.append("")

    out_path = (active_dir / REPRODUCE_DOC_NAME).resolve()
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


class LayoutWindow:
    def __init__(self, active_dir: Path, loaded_usd: Path | None = None,
                 minimal: bool = False, collision_on: bool = True):
        import omni.ui as ui

        self.ui = ui
        self.active_dir = active_dir
        # 本次会话最初加载的 USD（--load_latest_saved / --load_usd 时有值）。
        # “恢复初始状态”优先回到它；base-task 启动时为 None，则回退到磁盘上的 scene_v1_latest.usd。
        self.loaded_usd = loaded_usd
        # minimal=True（--property_edit）：只留 Save + Restore + 碰撞可视化开关，去掉加资产/移动那些 UI。
        # 用户通过 Isaac 原生属性面板改 prim 的 translate/orient/scale，这里只负责保存与可视化。
        self.minimal = minimal
        self.asset_names = sorted(_discover_usd_assets()) or list(DEFAULT_DYNAMIC_ASSETS.keys())
        self.selected_asset_name = self.asset_names[0]
        self.asset_entry_model = ui.SimpleStringModel(self.selected_asset_name)
        self.status_labels: dict[str, object] = {}
        self.last_saved = "None"
        self.last_added = "None"
        self.last_restore = "None"
        title = "Franka V1 Property Editor (static)" if minimal else "Franka V1 Scene Layout"
        self.window = ui.Window(title, width=540, height=300 if minimal else 460)
        with self.window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("Scene V1 (static edit: use the native Property panel for pos/orient/scale)"
                         if minimal else "Scene V1")
                with ui.HStack(spacing=6):
                    ui.Button("Save V1 (USD+JSON+registry)", clicked_fn=self.save_v1)
                    ui.Button("Restore (reload latest, discard edits)", clicked_fn=self.restore_initial)
                ui.Label("Restore = reload scene_v1_latest.usd from disk, discard unsaved edits (no restart needed).")
                # 碰撞体可视化开关（可视化控制 UI）
                with ui.HStack(spacing=6, height=24):
                    self.collision_cb = ui.CheckBox(width=20)
                    self.collision_cb.model.set_value(bool(collision_on))
                    self.collision_cb.model.add_value_changed_fn(
                        lambda m: set_collision_debug_visualization(bool(m.get_value_as_bool())))
                    ui.Label("Show colliders (collision visualization)")
                if not minimal:
                    ui.Label("Add USD Asset")
                    self.asset_model = ui.ComboBox(0, *self.asset_names).model
                    self.asset_model.add_item_changed_fn(self._on_asset_changed)
                    ui.StringField(model=self.asset_entry_model)
                    with ui.HStack(spacing=6):
                        ui.Button("Refresh Assets", clicked_fn=self.refresh_assets)
                        ui.Button("Add Asset", clicked_fn=self.add_selected_asset)
                    ui.Label("Move/add objects in the viewport, then click Save V1.")
                    ui.Label("Yellow InitCorner_0..3 = cube random-init rectangle; drag them, Save V1 records it.")
                else:
                    ui.Label("Robot is NOT controllable here. Select a prim, edit it in the Property panel, then Save.")
                keys = ("task", "active_dir", "last_saved", "last_restore", "registry") if minimal else (
                    "task", "active_dir", "selected_asset", "last_added", "last_saved",
                    "last_restore", "registry", "front_camera", "vla_camera", "fp_camera", "camera_body")
                for key in keys:
                    self.status_labels[key] = ui.Label(f"{key}:")

    def _on_asset_changed(self, model, item):
        index = model.get_item_value_model().as_int
        if 0 <= index < len(self.asset_names):
            self.selected_asset_name = self.asset_names[index]
            self.asset_entry_model.set_value(self.selected_asset_name)

    def refresh_assets(self):
        asset_map = _discover_usd_assets()
        self.asset_names = sorted(asset_map)
        self.selected_asset_name = self.asset_names[0] if self.asset_names else ""
        self.asset_entry_model.set_value(self.selected_asset_name)

    def add_selected_asset(self):
        asset_map = _discover_usd_assets()
        asset_key = self.asset_entry_model.as_string.strip() or self.selected_asset_name
        usd_path = Path(asset_key).expanduser()
        if not usd_path.is_absolute():
            usd_path = _repo_root() / usd_path
        asset_name = usd_path.parent.name if usd_path.is_file() else asset_key
        usd_path = usd_path if usd_path.is_file() else asset_map.get(asset_key)
        if usd_path is None:
            self.last_added = f"missing asset: {asset_key}"
            return
        self.last_added = _add_usd_reference(asset_name, usd_path)

    def save_v1(self):
        # 只有这里（用户点击 Save V1）才授权写入最新场景；写完立即撤销，
        # 任何其他模块/脚本在未授权时调用 registry 写 API 都会被 PermissionError 拒绝。
        authorize_scene_write(reason="layout_v1_ui Save V1 按钮")
        try:
            _stage = omni.usd.get_context().get_stage()
            usd, manifest = _save_v1(self.active_dir)
            reg = update_active_scene(
                ACTIVE_USD_NAME,
                ACTIVE_MANIFEST_NAME,
                registry_path=_registry_path(),
                note=f"Saved by layout_v1_ui at {_timestamp()} from task {args_cli.task}.",
                sensors=_camera_sensors_block(_stage),
            )
        finally:
            revoke_scene_write()
        self.last_saved = f"{usd.name} (+json, registry+sensors updated)"
        print(f"[layout_v1] registry updated: {reg}", flush=True)

    def restore_initial(self):
        """重新加载磁盘上的最新场景，丢弃当前未保存的改动（改错时无需关掉重启）。

        目标优先级：本次加载的 USD -> 磁盘上的 scene_v1_latest.usd。重新 open_stage 后，
        运行时画的参考圈 / init 角标会消失（它们不写进 USD），所以再 spawn 一次。
        """
        target = self.loaded_usd or (self.active_dir / ACTIVE_USD_NAME)
        target = Path(target).resolve()
        if not target.is_file():
            self.last_restore = f"无法恢复：找不到 {target.name}（还没保存过 latest？）"
            print(f"[layout_v1] 恢复失败：{target} 不存在", flush=True)
            return
        if not open_stage(str(target)):
            self.last_restore = f"恢复失败：打不开 {target.name}"
            print(f"[layout_v1] 恢复失败：open_stage 返回 False ({target})", flush=True)
            return
        _pause_timeline_for_layout()
        _spawn_reach_workspace()       # 参考圈不在 USD 里，重新画
        _spawn_init_region_markers()   # init 角标若已存于加载的 USD 则跳过，否则重建
        self.last_restore = f"已恢复到 {target.name}（未保存改动已丢弃）"
        print(f"[layout_v1] 已恢复初始状态 <- {target}", flush=True)

    def update(self):
        stage = omni.usd.get_context().get_stage()
        if not args_cli.enable_wrist_cameras:
            front_status = "not attached (launch with --enable_wrist_cameras)"
            fp_status = "not attached"
            vla_status = "not attached"
            body_status = "n/a"
        else:
            rendering = _cameras_render_enabled()
            mode = "rendering" if rendering else "config only (add --enable_cameras to render)"
            fr = d435cfg.FRONT_PROFILE.intrinsics()
            front_status = f"{mode} | RGB {fr['width']}x{fr['height']} | STATIC front (env root) -> LeRobot image"
            vw, vh = _vla_resolution()
            vla_status = f"{mode} | RGB {vw}x{vh} | wrist eye_in_hand (panda_hand) -> LeRobot wrist_image"
            fp = d435cfg.FP_PROFILE.intrinsics()
            depth = "off" if (rendering and args_cli.no_depth) else "on"
            fp_status = f"{mode} | RGB-D depth={depth} {fp['width']}x{fp['height']} | wrist (panda_hand) -> FoundationPose"
            body_status = "visible" if _show_body() else "hidden (default)"
        values = {
            "task": args_cli.task,
            "active_dir": str(self.active_dir),
            "selected_asset": self.selected_asset_name,
            "last_added": self.last_added,
            "last_saved": self.last_saved,
            "last_restore": self.last_restore,
            "registry": str(_registry_path()),
            "front_camera": front_status,
            "vla_camera": vla_status,
            "fp_camera": fp_status,
            "camera_body": body_status,
        }
        for key, label in self.status_labels.items():
            label.text = f"{key}: {values[key]}"


def _resolve_load_usd() -> Path | None:
    if args_cli.load_usd:
        p = Path(args_cli.load_usd).expanduser()
        if not p.is_absolute():
            p = _repo_root() / p
        if not p.is_file():
            raise FileNotFoundError(f"Saved USD does not exist: {p}")
        return p.resolve()
    if args_cli.load_latest_v1:
        p = _active_dir() / ACTIVE_USD_NAME
        if not p.is_file():
            raise FileNotFoundError(f"No active V1 USD yet: {p} (save one first).")
        return p.resolve()
    return None


def main():
    if args_cli.num_envs != 1:
        raise ValueError("layout_v1_ui supports --num_envs 1.")

    active_dir = _active_dir()
    env = None
    load_usd = _resolve_load_usd()
    if load_usd is not None:
        if not open_stage(str(load_usd)):
            raise RuntimeError(f"Failed to open saved USD stage: {load_usd}")
        print(f"[layout_v1] loaded USD: {load_usd}", flush=True)
    else:
        env_cfg = parse_env_cfg(
            args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        )
        env_cfg.seed = args_cli.seed
        if getattr(env_cfg, "events", None) is not None and hasattr(env_cfg.events, "randomize_cube_positions"):
            env_cfg.events.randomize_cube_positions = None
        env_cfg.viewer.eye = (2.0, -2.0, 1.4)
        env_cfg.viewer.lookat = (0.45, 0.0, 0.15)
        if args_cli.enable_wrist_cameras:
            vw, vh = _vla_resolution()
            attach_wrist_cameras(
                env_cfg,
                enable_cameras=_cameras_render_enabled(),
                show_body=_show_body(),
                enable_fp_depth=not args_cli.no_depth,
                enable_vla_depth=False,
                vla_resolution=(vw, vh),
            )
            rendering = "rendering RGB-D + RGB" if _cameras_render_enabled() else "config only (add --enable_cameras to render)"
            body = "body VISIBLE" if _show_body() else "body hidden"
            print(
                f"[layout_v1] wrist cameras attached: foundationpose_d435_rgbd + "
                f"vla_libero_eye_in_hand ({vw}x{vh}) | {rendering} | {body}",
                flush=True,
            )
        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset(seed=args_cli.seed)
    _pause_timeline_for_layout()

    # 碰撞体可视化【默认关闭】；仅 --show_colliders 时启动即开（随时可用 'Show colliders' 勾选框开关）。
    if not args_cli.headless and args_cli.show_colliders and not args_cli.disable_collision_debug_vis:
        enable_collision_debug_visualization()

    # Make the camera viewport show the correct pose immediately. The camera
    # prim's USD xform is correct from spawn, BUT with the timeline paused the
    # viewport caches the camera pose and only re-reads it when the prim's
    # transform attribute is re-authored (the manual fix is: nudge the camera
    # orient, then set it back). We replicate that programmatically below.
    if env is not None and args_cli.enable_wrist_cameras and not args_cli.headless:
        try:
            for _ in range(8):                       # let the camera initialize callback run
                _pause_timeline_for_layout()
                env.unwrapped.sim.render()
                simulation_app.update()
            _refresh_wrist_camera_viewport(env)       # re-author orient -> viewport snaps to pose
            for _ in range(3):
                env.unwrapped.sim.render()
                simulation_app.update()
            print("[layout_v1] wrist-camera viewport refreshed (orient re-authored).", flush=True)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[layout_v1] camera viewport refresh skipped ({exc}).", flush=True)

    # 把 4 个把手碰撞方块(抽屉 top/middle/bottom + 咖啡机拉手)生成成【可在官方属性框编辑】的 prim。
    # base task 与 load_usd 两条路径都装：load_usd 时若 USD 里已有方块则保留(不覆盖用户编辑)。
    try:
        from franka_v1_skill_lab.scene_interface.scene_props import install_editable_grasp_blocks
        install_editable_grasp_blocks(omni.usd.get_context().get_stage(), env_root=ENV_NS)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[layout_v1] install editable grasp blocks skipped: {exc}", flush=True)

    # 机器人默认停到最新保存的 home(robot_home.json)：base task 写 sim(立即可见)，
    # 两条路径都把 home 写进 USD 关节(drive target + joint state),保证保存/重载/测试一致。
    try:
        _set_robot_home(omni.usd.get_context().get_stage(), env)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[layout_v1] set robot home skipped: {exc}", flush=True)

    # property-edit 模式只编辑当前场景，不往里塞编辑器辅助物(够标环/初始化角标)，保持场景干净。
    if not args_cli.property_edit:
        _spawn_reach_workspace()      # orange/green reach rings to judge what the Franka can grasp
        _spawn_init_region_markers()  # 4 draggable yellow corners defining the cube random-init rectangle

    # 默认隐藏 4 个 InitCorner 方块(用户暂时不需要)；--show_init_region 可恢复显示。
    # 必须在 _spawn_init_region_markers 之后(否则刚生成又被显示出来)，且对 load_usd 里已有的也生效。
    try:
        _hide_init_markers(omni.usd.get_context().get_stage(),
                           visible=bool(getattr(args_cli, "show_init_region", False)))
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[layout_v1] hide init markers skipped: {exc}", flush=True)

    _collision_on = (not args_cli.headless and args_cli.show_colliders
                     and not args_cli.disable_collision_debug_vis)
    window = None if args_cli.headless else LayoutWindow(
        active_dir, loaded_usd=load_usd, minimal=args_cli.property_edit, collision_on=_collision_on)

    while simulation_app.is_running():
        _pause_timeline_for_layout()
        if window is not None:
            window.update()
        simulation_app.update()

    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
