# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""相机位姿(局部 xform)读/写 + 保存到 registry sensors + 重载应用。

为避开 CameraCfg.offset 的 convention 换算坑：统一用「相机 prim 的 USD 局部 xform」作为唯一表示。
- 调整：直接改 prim 的 xformOp:translate/orient（camera_panel 做）。
- 保存：读 wrist(FP) + front 相机 prim 当前局部位姿，写进 registry sensors.<name>.offset(pos+rot_wxyz)。
- 重载：test_mode 启动后把保存的局部位姿直接设回 prim（不经 offset/convention），所见即所得。

wrist 逻辑相机 = FP + VLA 两个 prim(同位置)，一起调/存/载。顶层只 stdlib + pxr 延迟 import。
"""

from __future__ import annotations

ENV_NS = "/World/envs/env_0"
WRIST_CAM_PRIMS = [
    f"{ENV_NS}/Robot/panda_hand/foundationpose_d435_rgbd",
    f"{ENV_NS}/Robot/panda_hand/vla_libero_eye_in_hand",
]
FRONT_CAM_PRIM = f"{ENV_NS}/vla_front_static"
LEFT_CAM_PRIM = f"{ENV_NS}/vla_left_static"
RIGHT_CAM_PRIM = f"{ENV_NS}/vla_right_static"
TOP_CAM_PRIM = f"{ENV_NS}/vla_top_static"
# registry sensors 里写 offset 的相机名
WRIST_SENSOR_NAMES = ["foundationpose_d435_rgbd", "vla_libero_eye_in_hand"]
FRONT_SENSOR_NAMES = ["vla_front_static"]
LEFT_SENSOR_NAMES = ["vla_left_static"]
RIGHT_SENSOR_NAMES = ["vla_right_static"]
TOP_SENSOR_NAMES = ["vla_top_static"]


def read_local_pose(stage, prim_path):
    """读 prim 局部 (pos[3], quat_wxyz[4])。prim 不存在返回 None。"""
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    xf = UsdGeom.Xformable(prim)
    ops = {op.GetName(): op for op in xf.GetOrderedXformOps()}
    pos = (0.0, 0.0, 0.0)
    quat = (1.0, 0.0, 0.0, 0.0)
    if "xformOp:translate" in ops and ops["xformOp:translate"].Get() is not None:
        t = ops["xformOp:translate"].Get()
        pos = (float(t[0]), float(t[1]), float(t[2]))
    if "xformOp:orient" in ops and ops["xformOp:orient"].Get() is not None:
        q = ops["xformOp:orient"].Get()
        quat = (float(q.GetReal()), *[float(v) for v in q.GetImaginary()])
    return pos, quat


def set_local_pose(stage, prim_path, pos, quat_wxyz) -> bool:
    """设 prim 局部 xformOp:translate/orient（保留 scale），缺则新增。"""
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False
    xf = UsdGeom.Xformable(prim)
    ops = {op.GetName(): op for op in xf.GetOrderedXformOps()}
    if "xformOp:translate" in ops:
        ops["xformOp:translate"].Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    else:
        xf.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    qf = [float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])]
    if "xformOp:orient" in ops:
        cur = ops["xformOp:orient"].Get()
        q = Gf.Quatf(*qf) if isinstance(cur, Gf.Quatf) else Gf.Quatd(*qf)
        ops["xformOp:orient"].Set(q)
    else:
        xf.AddOrientOp().Set(Gf.Quatd(*qf))
    return True


def current_offsets_from_stage(stage) -> dict:
    """读 wrist(FP prim) + front 当前局部位姿，给保存用。返回 {'wrist':(pos,quat),'front':(pos,quat)}。"""
    out = {}
    w = read_local_pose(stage, WRIST_CAM_PRIMS[0])
    f = read_local_pose(stage, FRONT_CAM_PRIM)
    l = read_local_pose(stage, LEFT_CAM_PRIM)
    r = read_local_pose(stage, RIGHT_CAM_PRIM)
    tp = read_local_pose(stage, TOP_CAM_PRIM)
    if w is not None:
        out["wrist"] = w
    if f is not None:
        out["front"] = f
    if l is not None:
        out["left"] = l
    if r is not None:
        out["right"] = r
    if tp is not None:
        out["top"] = tp
    return out


def offsets_to_sensors(offsets: dict, registry_path=None) -> dict:
    """把 {'wrist':(pos,quat),'front':(pos,quat)} 合并进 registry sensors 的 offset 字段，返回 sensors dict。"""
    from franka_v1_skill_lab.scene import default_sensors
    from franka_v1_skill_lab.scene.scene_registry import load_registry

    reg = load_registry(registry_path)
    sensors = dict(reg.sensors) if reg.sensors else dict(default_sensors())

    def _set(name, pose):
        pos, quat = pose
        blk = dict(sensors.get(name, {}))
        conv = (blk.get("offset") or {}).get("convention", "opengl")
        blk["offset"] = {"pos": [float(v) for v in pos], "rot_wxyz": [float(v) for v in quat], "convention": conv}
        sensors[name] = blk

    if "wrist" in offsets:
        for nm in WRIST_SENSOR_NAMES:
            _set(nm, offsets["wrist"])
    if "front" in offsets:
        for nm in FRONT_SENSOR_NAMES:
            _set(nm, offsets["front"])
    if "left" in offsets:
        for nm in LEFT_SENSOR_NAMES:
            _set(nm, offsets["left"])
    if "right" in offsets:
        for nm in RIGHT_SENSOR_NAMES:
            _set(nm, offsets["right"])
    if "top" in offsets:
        for nm in TOP_SENSOR_NAMES:
            _set(nm, offsets["top"])
    return sensors


def apply_saved_offsets_runtime(env, registry_path=None) -> int:
    """test_mode 启动后：把 registry sensors 里保存的相机局部位姿直接设回 camera prim。返回应用数。"""
    import omni.usd

    from franka_v1_skill_lab.scene.scene_registry import load_registry

    reg = load_registry(registry_path)
    sensors = reg.sensors or {}
    stage = omni.usd.get_context().get_stage()
    n = 0
    name_to_prims = {nm: [p] for nm, p in zip(["vla_front_static"], [FRONT_CAM_PRIM])}
    name_to_prims["vla_left_static"] = [LEFT_CAM_PRIM]
    name_to_prims["vla_right_static"] = [RIGHT_CAM_PRIM]
    name_to_prims["vla_top_static"] = [TOP_CAM_PRIM]
    name_to_prims["foundationpose_d435_rgbd"] = [WRIST_CAM_PRIMS[0]]
    name_to_prims["vla_libero_eye_in_hand"] = [WRIST_CAM_PRIMS[1]]
    for name, prims in name_to_prims.items():
        off = (sensors.get(name) or {}).get("offset")
        if not off:
            continue
        pos = off.get("pos")
        quat = off.get("rot_wxyz")
        if not (pos and quat and len(pos) == 3 and len(quat) == 4):
            continue
        for p in prims:
            if set_local_pose(stage, p, pos, quat):
                n += 1
    if n:
        print(f"[camera_offsets] applied saved camera local poses to {n} prims.", flush=True)
    return n
