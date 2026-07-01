# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""把最新保存场景 (scene_v1_latest) 的物体位姿/缩放覆盖到 env cfg。

为什么需要：env 由 base task cfg 构建，家电(cabinet/microwave/coffee)的位姿写死在
`stack_joint_pos_env_cfg.py`。用户在布局编辑器里挪动/缩放家电并存进 `scene_v1_latest.usd`
后，base cfg 是 STALE 的。本模块在建 env 前，用最新 manifest 的 xform 覆盖 cfg 里对应物体的
`init_state.pos/rot` 和 `spawn.scale`，使运行场景 == 最新保存场景（单一可信源）。

纯 python（json + resolve_active_scene）；只做属性赋值，不 import isaac/torch。

局限：manager 环境只会 spawn cfg 里已有的成员；用户在编辑器里**新增**的 prim（如
MicrowaveStand、InitCorner 标定块、相机）不在 base cfg 里，不会被 spawn。本模块只同步
已存在成员的位姿/缩放；新增 prim 需要在 base cfg 里加成员（另议）。
"""

from __future__ import annotations

# manifest 物体名 -> env_cfg.scene 成员属性名
NAME_TO_CFG = {
    "Cabinet": "cabinet",
    "Microwave": "microwave",
    "Fridge": "microwave",   # 微波炉成员换成冰箱后 prim 叫 Fridge；存档里是 Fridge 时也回填 microwave 成员
    "CoffeeMachine": "coffee_machine",
    "Dishwasher": "dishwasher",
    "SektionCabinet": "sektion_cabinet",   # Sektion 橱柜(articulation)：存档 prim 名 SektionCabinet -> 回填 sektion_cabinet 位姿
    # 旧存档里 Sektion 是 GUI 加的静态 prop(Prop_sektion_cabinet_instanceable)；把它的保存位姿也回填到
    # 可动的 sektion_cabinet articulation，这样动态场景的橱柜就和静态场景对齐(不再用 cfg 默认远处生成位)。
    "Prop_sektion_cabinet_instanceable": "sektion_cabinet",
    "Robot": "robot",
    "Knife": "knife",
    "Cube_1": "cube_1",
    "Cube_2": "cube_2",
    "Cube_3": "cube_3",
}

# 这些 manifest 物体不映射到 cfg 成员（机器人/地面/灯/相机/标定块/编辑器新增 prim 等），静默跳过。
_IGNORE_UNMAPPED = {
    "Robot", "GroundPlane", "light", "vla_front_static", "vla_left_static", "vla_right_static",
    "vla_top_static", "vla_libero_eye_in_hand",
    "foundationpose_d435_rgbd", "InitCorner_0", "InitCorner_1", "InitCorner_2", "InitCorner_3",
    "OmniverseKit_Persp", "OmniverseKit_Front", "OmniverseKit_Top", "OmniverseKit_Right",
}


# manifest 相机名 -> env_cfg.scene 成员属性名（静态第三人称相机，可在 spawn 前写 offset）。
# 腕部相机挂在 panda_hand 上、随臂动，不在此处覆盖（其 offset 是手系局部量，保持 cfg）。
_CAMERA_TO_CFG = {"vla_front_static": "vla_front_static", "vla_left_static": "vla_left_static",
                  "vla_right_static": "vla_right_static", "vla_top_static": "vla_top_static"}


def apply_latest_camera_to_cfg(env_cfg, registry_path=None, verbose: bool = True) -> list:
    """在 spawn 前把最新 manifest 里静态相机的视角写进 CameraCfg.offset。

    必须在建 env 之前调用：IsaacLab Camera 的 data.pos_w/quat_w_world 在 spawn 时从
    CameraCfg.offset 取定，运行时用 set_local_pose/set_world_poses 挪 prim 只动渲染、
    不动这份缓存 → 渲染与上报外参 desync → FoundationPose 的 GT 投影/世界系换算全错。
    在 cfg 阶段写 offset 则渲染与外参从一开始就一致。返回应用到的相机名列表。
    """
    manifest, _ = _active_manifest_dict(registry_path)
    scene = getattr(env_cfg, "scene", None)
    applied = []
    xf_by_name = {o.get("name"): (o.get("xform") or {}) for o in manifest.get("objects", [])}
    for mname, attr in _CAMERA_TO_CFG.items():
        xf = xf_by_name.get(mname)
        member = getattr(scene, attr, None)
        off = getattr(member, "offset", None) if member is not None else None
        if not xf or off is None:
            continue
        t = xf.get("translate")
        r = xf.get("orient_wxyz")
        if t is not None and len(t) == 3:
            off.pos = tuple(float(v) for v in t)
        if r is not None and len(r) == 4:
            off.rot = tuple(float(v) for v in r)
            off.convention = "opengl"   # manifest xform 是 USD 局部姿态（opengl 相机约定）
        applied.append(attr)
    if verbose and applied:
        print(f"[scene_sync] applied latest camera pose(s) to cfg (pre-spawn): {applied}", flush=True)
    return applied


def _active_manifest_dict(registry_path=None) -> tuple[dict, str]:
    """读取当前 active 场景的 manifest（dict）+ 路径。"""
    import json
    from pathlib import Path

    from franka_v1_skill_lab.scene.scene_registry import resolve_active_scene

    info = resolve_active_scene(registry_path)
    mpath = info.get("manifest")
    if not mpath or not Path(mpath).is_file():
        raise FileNotFoundError(f"active 场景 manifest 不存在：{mpath}")
    return json.loads(Path(mpath).read_text(encoding="utf-8")), mpath


def apply_latest_scene_to_cfg(env_cfg, registry_path=None, verbose: bool = True) -> dict:
    """用最新 manifest 覆盖 env_cfg 里物体的 init_state.pos/rot + spawn.scale。

    返回 {"manifest": path, "applied": [...], "skipped_unmapped": [...], "missing_cfg": [...]}。
    任何单个物体失败不致命（记入 skipped），保证 env 仍能构建。
    """
    manifest, mpath = _active_manifest_dict(registry_path)
    scene = getattr(env_cfg, "scene", None)
    applied, skipped, missing = [], [], []

    for obj in manifest.get("objects", []):
        name = obj.get("name")
        xf = obj.get("xform") or {}
        attr = NAME_TO_CFG.get(name)
        if attr is None:
            # 动态额外资产(add_usd_asset 加的水果道具)：prim 名 == cfg 成员名，约定 Prop_ 前缀。
            # 成员已在 cfg_hook 里先 spawn 出来，这里覆盖其保存的 pos/rot/scale。
            if name.startswith("Prop_") and getattr(scene, name, None) is not None:
                attr = name
            else:
                if name not in _IGNORE_UNMAPPED:
                    skipped.append(name)
                continue
        member = getattr(scene, attr, None)
        if member is None:
            missing.append(f"{name}->{attr}")
            continue
        t = xf.get("translate")
        r = xf.get("orient_wxyz")
        s = xf.get("scale")
        changed = []
        init_state = getattr(member, "init_state", None)
        if init_state is not None and t is not None and len(t) == 3:
            init_state.pos = tuple(float(v) for v in t)
            changed.append("pos")
        if init_state is not None and r is not None and len(r) == 4:
            init_state.rot = tuple(float(v) for v in r)
            changed.append("rot")
        spawn = getattr(member, "spawn", None)
        if spawn is not None and hasattr(spawn, "scale") and s is not None and len(s) == 3:
            spawn.scale = tuple(float(v) for v in s)
            changed.append("scale")
        applied.append({"name": name, "cfg": attr, "changed": changed})

    summary = {"manifest": mpath, "applied": applied, "skipped_unmapped": skipped, "missing_cfg": missing}
    if verbose:
        names = ", ".join(f"{a['cfg']}({'+'.join(a['changed'])})" for a in applied)
        print(f"[scene_sync] applied latest scene: {mpath}", flush=True)
        print(f"[scene_sync]   overridden: {names or '(none)'}", flush=True)
        if skipped:
            print(f"[scene_sync]   skipped unmapped (editor-added, no cfg member -> NOT spawned): {skipped}", flush=True)
        if missing:
            print(f"[scene_sync]   cfg missing member: {missing}", flush=True)
    return summary
