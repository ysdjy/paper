# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""场景道具：微波炉台子(kinematic，可跟随)、InitCorner 黄框、碰撞体细化。

- add_microwave_stand: 悬空微波炉下加一个 kinematic RigidObject 台子(可被 write_root_pose 移动 -> 跟随微波炉)。
- spawn_init_markers: 从 active manifest 读 4 个 InitCorner 点，spawn 黄色 visual-only cube(可见 + 保存时保留区域定义)。
- refine_collisions: 运行时对各 link 的 mesh collider 设差异化 approximation(把手 convexDecomposition、门/抽屉 boundingCube、咖啡机放杯处禁碰撞)。实验性，可能需物理重解析才生效。

cfg 加工函数(add_microwave_stand/spawn_init_markers)是属性赋值，在 gym.make 之前的 cfg hook 里调用；
refine_collisions 需在 spawn 之后(env 存在)调用。isaac 全部延迟 import。
"""

from __future__ import annotations

# 实测自 microwave_flattened.usd 的局部包围盒（未缩放，单位 m）
MW_MIN_Z_LOCAL = -0.4343
MW_X_SIZE_LOCAL = 0.914
MW_Y_SIZE_LOCAL = 1.603
TABLE_TOP_Z = 0.0
FOOTPRINT_FACTOR = 0.8
MIN_STAND_HEIGHT = 0.03


def add_microwave_stand(env_cfg, table_top_z: float = TABLE_TOP_Z, color=(0.40, 0.40, 0.42)) -> bool:
    """在微波炉下方加一个 kinematic、带碰撞的台子(可运行时移动 -> 跟随微波炉)。返回是否添加。"""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    mw = getattr(getattr(env_cfg, "scene", None), "microwave", None)
    if mw is None or getattr(mw, "init_state", None) is None:
        return False
    pos = tuple(float(v) for v in mw.init_state.pos)
    scale = tuple(float(v) for v in getattr(getattr(mw, "spawn", None), "scale", (1.0, 1.0, 1.0)))

    mw_bottom_z = pos[2] + MW_MIN_Z_LOCAL * scale[2]
    height = mw_bottom_z - table_top_z
    if height < MIN_STAND_HEIGHT:
        return False

    fx = MW_X_SIZE_LOCAL * scale[0] * FOOTPRINT_FACTOR
    fy = MW_Y_SIZE_LOCAL * scale[1] * FOOTPRINT_FACTOR
    center_z = table_top_z + height / 2.0

    env_cfg.scene.microwave_stand = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/MicrowaveStand",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(pos[0], pos[1], center_z)),
        spawn=sim_utils.CuboidCfg(
            size=(float(fx), float(fy), float(height)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0, friction_combine_mode="max",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color), roughness=0.7),
        ),
    )
    print(f"[scene_props] microwave stand (kinematic) added: pos=({pos[0]:.3f},{pos[1]:.3f},{center_z:.3f}) "
          f"size=({fx:.3f},{fy:.3f},{height:.3f})", flush=True)
    return True


# 实测自 Dishwasher_12085/dishwasher.usd 的局部包围盒（未缩放，单位 m）
DW_MIN_Z_LOCAL = -0.5281
DW_X_SIZE_LOCAL = 1.4865
DW_Y_SIZE_LOCAL = 1.1543
# 只取【本体 link_0】的水平包围盒（碗架 link_1 沿 local -X 拉出，不算进台子）。实测 link_0 bbox：
DW_BODY_X_LOCAL = (-0.0589, 0.7266)   # 本体在 local X 的范围（拉出方向；台子只盖这段）
DW_BODY_Y_LOCAL = (-0.5305, 0.5464)   # 本体在 local Y 的范围


def _rot_vec_by_quat(q_wxyz, v):
    """用四元数(w,x,y,z)旋转向量 v(3,)，返回旋转后 (x,y,z)。纯 python。"""
    w, x, y, z = (float(c) for c in q_wxyz)
    vx, vy, vz = (float(c) for c in v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return (rx, ry, rz)


def dw_stand_offset_world(rot_wxyz, scale):
    """洗碗机台子相对洗碗机原点的世界系 XY 偏移：把【本体中心】(local)缩放后旋到世界。

    台子做成只盖本体(link_0)，本体中心不在洗碗机原点(原点偏向碗架侧)，所以台子要平移这个偏移。
    面板里跟随移动时也用同一函数，保证移动/旋转后台子始终对准本体。"""
    lcx = (DW_BODY_X_LOCAL[0] + DW_BODY_X_LOCAL[1]) / 2.0 * float(scale[0])
    lcy = (DW_BODY_Y_LOCAL[0] + DW_BODY_Y_LOCAL[1]) / 2.0 * float(scale[1])
    ox, oy, _ = _rot_vec_by_quat(rot_wxyz, (lcx, lcy, 0.0))
    return float(ox), float(oy)


DW_PAD_HEIGHT = 0.05   # 洗碗机下方垫子固定 5cm（用户要求）


def add_dishwasher_stand(env_cfg, pad_height: float = DW_PAD_HEIGHT, table_top_z: float = TABLE_TOP_Z,
                         color=(0.40, 0.40, 0.42)) -> bool:
    """在洗碗机下放一个台子，台子高度【随洗碗机当前高度自适应】：顶面贴洗碗机底面、底面落在台面。

    不再把洗碗机强行压到离地 5cm（那样台子永远只有 5cm、洗碗机被拉低）。改为：读洗碗机当前
    (保存场景/cfg 决定的) z，算出它的底面到台面的整段间隙，台子就做这么高，正好把这段空隙填满，
    洗碗机停在原来的高度上。pad_height 仅作【最小间隙】——若洗碗机太低(底面低于台面+5cm，会卡
    门板滑轨)，才把它抬到底面=台面+5cm。台子 kinematic、与洗碗机同 XY+朝向、可运行时跟随。"""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    dw = getattr(getattr(env_cfg, "scene", None), "dishwasher", None)
    if dw is None or getattr(dw, "init_state", None) is None:
        return False
    pos = tuple(float(v) for v in dw.init_state.pos)
    rot = tuple(float(v) for v in getattr(dw.init_state, "rot", (1.0, 0.0, 0.0, 0.0)))
    scale = tuple(float(v) for v in getattr(getattr(dw, "spawn", None), "scale", (1.0, 1.0, 1.0)))

    # 洗碗机当前底面（世界 z）= 原点 z + 局部最低点*缩放
    dw_bottom = pos[2] + DW_MIN_Z_LOCAL * scale[2]
    gap = dw_bottom - table_top_z
    min_gap = float(pad_height)   # 最小间隙：低于此会卡门板滑轨
    if gap < min_gap:
        # 太低/穿台面：抬洗碗机使底面=台面+min_gap，保留 x/y/朝向
        target_z = table_top_z + min_gap - DW_MIN_Z_LOCAL * scale[2]
        dw.init_state.pos = (pos[0], pos[1], target_z)
        dw_bottom = table_top_z + min_gap
        gap = min_gap
        print(f"[scene_props] dishwasher too low, raised to z={target_z:.3f} (bottom={dw_bottom:.3f})", flush=True)

    height = float(gap)                          # 台子高度 = 台面到洗碗机底面整段间隙
    center_z = table_top_z + height / 2.0        # 顶面贴洗碗机底面、底面落台面
    # 台子只盖【本体 link_0】水平投影：尺寸取本体 bbox(不含碗架拉出段)，位置平移到本体中心。
    fx = (DW_BODY_X_LOCAL[1] - DW_BODY_X_LOCAL[0]) * scale[0] * 0.92   # 局部 X(拉出轴)只盖本体
    fy = (DW_BODY_Y_LOCAL[1] - DW_BODY_Y_LOCAL[0]) * scale[1] * 0.92
    ox, oy = dw_stand_offset_world(rot, scale)
    sx, sy = pos[0] + ox, pos[1] + oy

    env_cfg.scene.dishwasher_stand = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/DishwasherStand",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(sx, sy, center_z), rot=rot),
        spawn=sim_utils.CuboidCfg(
            size=(float(fx), float(fy), height),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0, friction_combine_mode="max",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color), roughness=0.7),
        ),
    )
    print(f"[scene_props] dishwasher kept at z={dw.init_state.pos[2]:.3f} (bottom={dw_bottom:.3f}); "
          f"body-only stand h={height:.3f} at ({sx:.3f},{sy:.3f},{center_z:.3f}) "
          f"size=({fx:.3f},{fy:.3f},{height:.3f})", flush=True)
    return True


# Isaac Stand 几何竖直范围(local frame)：原点在顶面，向下延伸 0.618。
# 量自 usd_assets/IsaacProps/Props/Mounts/Stand/stand_instanceable.usd 的 bbox(z: -0.618~0)。
# 换资产时下面会用 USD 实测，量不到才回退这两个常量。
_STAND_LOCAL_MIN_Z = -0.618
_STAND_LOCAL_MAX_Z = 0.0


def _usd_local_z_extent(up: str) -> tuple[float, float]:
    """返回 USD 默认 prim 在自身 local frame 的 (min_z, max_z)。失败回退 stand 常量。"""
    try:
        from pxr import Usd, UsdGeom
        st = Usd.Stage.Open(up)
        dp = st.GetDefaultPrim()
        bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        rng = bc.ComputeWorldBound(dp).ComputeAlignedRange()
        return float(rng.GetMin()[2]), float(rng.GetMax()[2])
    except Exception:
        return _STAND_LOCAL_MIN_Z, _STAND_LOCAL_MAX_Z


def _usd_grounded_z(up: str, rot_wxyz, scale) -> float:
    """资产落地的世界 z：把【缩放+旋转后】包围盒最低点抬到 z=0，返回应设的 pos.z。

    顺序与运行时一致：world = T + R*(S*p_local)。对 8 个角点先缩放、再旋转，取最低 z，
    pos.z = -min_z。这样【翻转过的资产(如倒扣的碗绕 X 转 180°)】也能正确贴地，不悬空/不下陷。
    旋转为单位四元数时退化成 -min_z_local*scale_z（与旧逻辑一致，向后兼容）。失败回退旧算法。"""
    try:
        from pxr import Usd, UsdGeom, Gf
        st = Usd.Stage.Open(up)
        dp = st.GetDefaultPrim()
        bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        rng = bc.ComputeWorldBound(dp).ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        q = Gf.Quatd(float(rot_wxyz[0]), float(rot_wxyz[1]), float(rot_wxyz[2]), float(rot_wxyz[3]))
        rot = Gf.Rotation(q)
        minz = None
        for cx in (mn[0], mx[0]):
            for cy in (mn[1], mx[1]):
                for cz in (mn[2], mx[2]):
                    sp = Gf.Vec3d(cx * float(scale[0]), cy * float(scale[1]), cz * float(scale[2]))
                    wz = rot.TransformDir(sp)[2]
                    minz = wz if minz is None else min(minz, wz)
        return -float(minz)
    except Exception:
        min_z, _ = _usd_local_z_extent(up)
        return -float(min_z) * float(scale[2])


def add_robot_stand(env_cfg, usd_path: str = "SapienAssetPipeline/usd_assets/IsaacProps/Stand/stand_instanceable.usd") -> bool:
    """把 Isaac 支架落地放在机器人当前 XY 处，并把机器人基座抬高到支架顶面 -> 机器人坐在支架上。

    支架 USD 原点在顶面、向下延伸；若按机器人基座(z≈0)直接摆，整座支架会埋到地面以下。
    正确摆法(geom 实测 min_z/max_z)：支架底面落到机器人当前所在的地面 rz，机器人基座抬到支架顶面：
        stand_top  = rz + (max_z - min_z)      # 支架高度
        stand.pos.z = rz - min_z               # 顶面(原点)落到 stand_top，底面落到 rz
        robot.pos.z = rz + (max_z - min_z)     # 机器人坐到支架顶
    机器人 XY 不变；只抬高 Z(其余场景物体不动)。支架是 KINEMATIC 刚体(可被 write_root_pose_to_sim
    移动 -> 在 asset_pose_panel 里跟随机器人移动/旋转，保持相对位置)。
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    scene = getattr(env_cfg, "scene", None)
    robot = getattr(scene, "robot", None) if scene is not None else None
    if robot is None or getattr(robot, "init_state", None) is None:
        print("[scene_props] add_robot_stand: no robot, skip.", flush=True)
        return False
    rpos = tuple(float(v) for v in robot.init_state.pos)
    rrot = tuple(float(v) for v in getattr(robot.init_state, "rot", (1.0, 0.0, 0.0, 0.0)))

    from pathlib import Path

    up = str(usd_path)
    if not (up.startswith(("http://", "https://", "omniverse://")) or Path(up).is_absolute()):
        up = str((_repo_root() / up).resolve())
    if up.startswith("/") and not Path(up).is_file():
        print(f"[scene_props] add_robot_stand WARNING: usd not found: {up}", flush=True)
        return False

    min_z, max_z = _usd_local_z_extent(up)
    height = max_z - min_z
    floor_z = TABLE_TOP_Z                   # 实际地面(0)；不用 rpos[2]，否则恢复后已抬高的机器人会被二次抬高
    stand_pos = (rpos[0], rpos[1], floor_z - min_z)   # 顶面落到 floor_z+height，底面落到 floor_z
    robot_top = (rpos[0], rpos[1], floor_z + height)  # 机器人基座抬到支架顶(幂等：始终=floor+height)

    # stand USD 是 instanceable -> rigid_props 加不到实例代理(报"no rigid body")。用自定义 spawn func
    # 先去实例化再加 kinematic RigidBodyAPI，使其成为可 write_root_pose 移动的 kinematic 刚体。
    stand_spawn = sim_utils.UsdFileCfg(usd_path=up)
    stand_spawn.func = spawn_kinematic_deinstanced
    scene.robot_stand = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RobotStand",
        init_state=RigidObjectCfg.InitialStateCfg(pos=stand_pos, rot=rrot),  # 与机器人同朝向
        spawn=stand_spawn,
    )
    robot.init_state.pos = robot_top        # 抬高机器人基座，坐到支架顶
    print(f"[scene_props] robot stand (kinematic) grounded at XY={rpos[:2]} height={height:.3f}: "
          f"stand.pos={tuple(round(v,3) for v in stand_pos)}, robot raised to z={robot_top[2]:.3f}", flush=True)
    return True


def spawn_init_markers(env_cfg, z: float = 0.03, size: float = 0.04, color=(1.0, 0.85, 0.0)) -> int:
    """从 active manifest spawn 黄色 visual-only InitCorner 方块。返回数量。

    优先用 manifest 里每个角【保存的 (x,y,z)】（保留用户抬到桌面的高度）；没有保存高度时回退到
    区域 CCW 的 (x,y) + 默认 z。这样用户把方块放桌面上、Save 后重载不再默认落地。"""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    def _mk(name, x, y, zz):
        setattr(env_cfg.scene, name, AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/" + name,
            init_state=AssetBaseCfg.InitialStateCfg(pos=(float(x), float(y), float(zz))),
            spawn=sim_utils.CuboidCfg(
                size=(size, size, size), visible=False,   # 默认隐藏(用户暂不需要)；仍 spawn 以便保存保留区域+可toggle
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color)),
            ),   # no collision_props -> visual only
        ))

    try:
        from franka_v1_skill_lab.scene.init_region import load_init_corner_poses, load_init_region

        poses = load_init_corner_poses()   # {name:(x,y,z)} 保存的高度
    except Exception as exc:  # pragma: no cover
        print(f"[scene_props] WARNING: cannot load InitCorner region ({exc}); markers skipped.", flush=True)
        return 0
    if poses:   # 有保存位姿：按名字 spawn，保留各角身份 + 高度
        for name, (x, y, zz) in poses.items():
            _mk(name, x, y, zz)
        print(f"[scene_props] spawned {len(poses)} InitCorner markers at saved (x,y,z).", flush=True)
        return len(poses)
    try:        # 回退：区域 CCW (x,y) + 默认 z
        region = load_init_region()
    except Exception as exc:  # pragma: no cover
        print(f"[scene_props] WARNING: cannot load InitCorner region ({exc}); markers skipped.", flush=True)
        return 0
    for i, (x, y) in enumerate(region):
        _mk(f"InitCorner_{i}", x, y, z)
    print(f"[scene_props] spawned {len(region)} InitCorner markers (default z={z}).", flush=True)
    return len(region)


def _count_rigid_bodies(usd_path: str) -> int:
    """数 USD 里 RigidBodyAPI 的数量(用于判断单刚体/多体/纯视觉)。失败返回 -1。"""
    try:
        from pxr import Usd, UsdPhysics

        st = Usd.Stage.Open(str(usd_path))   # 默认加载 payload，计数准确
        return sum(1 for p in st.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI))
    except Exception:
        return -1


def add_usd_asset(env_cfg, name: str, usd_path: str, pos=(0.4, 0.0, 0.1),
                  rot=(1.0, 0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0),
                  rigid: bool = True, mass: float = 0.2, ground: bool = False) -> bool:
    """往场景加一个 USD 资产(本地路径 或 云端直链 http/https/omniverse://)。

    rigid=True -> 可被物理推动/抓取的刚体(加 rigid+mass+collision props)；False -> 静态可见资产
    (AssetBaseCfg，保留资产自带物理/碰撞)。云端 USD 由 omni.client 解析(需联网)。失败不致命，返回 False。
    ground=True -> 按资产 bbox 把底面落到地面(z=0)，覆盖 pos.z；避免静态道具悬空(rigid 的也省去掉落)。
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

    scene = getattr(env_cfg, "scene", None)
    if scene is None:
        return False
    try:
        from pathlib import Path

        # 本地相对路径 -> 绝对(相对仓库根)；http/https/omniverse:// 原样。
        up = str(usd_path)
        if not (up.startswith(("http://", "https://", "omniverse://")) or Path(up).is_absolute()):
            up = str((_repo_root() / up).resolve())
        spawn = sim_utils.UsdFileCfg(usd_path=up, scale=tuple(float(v) for v in scale))
        p = tuple(float(v) for v in pos)
        r = tuple(float(v) for v in rot)
        if ground:
            # 把资产底面落到地面 z=0（考虑 rot：翻转过的资产也正确贴地，不悬空/不下陷）。
            p = (p[0], p[1], _usd_grounded_z(up, r, scale))
        # RigidObjectCfg 要求【恰好 1 个刚体】。0=纯视觉/单碰撞 -> 静态；多个=articulation -> 固定铰接体。
        n_rb = _count_rigid_bodies(up)
        if rigid and n_rb != 1:
            print(f"[scene_props] add_usd_asset: '{name}' has {n_rb} rigid bodies (not 1) "
                  f"-> spawning as {'articulation' if n_rb > 1 else 'static'}.", flush=True)
            rigid = False
        if rigid:
            # 资产自带 RigidBodyAPI/碰撞/质量(如 YCB 单体)；直接引用为 RigidObject(可抓/可 write_root_pose)。
            # 限制解穿插速度 + 提高 solver 迭代：万一在静态编辑器里把两个物体摆得碰撞体重叠，
            # 动态场景物理一开也只会【缓慢分开】而不会爆飞到乱七八糟的位置(避免“乱码”)。
            spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            )
            cfg = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                init_state=RigidObjectCfg.InitialStateCfg(pos=p, rot=r),
                spawn=spawn,
            )
        elif n_rb > 1:
            # 多体 articulation(如 Sektion 橱柜)：spawn 成【固定 articulation】-> 可 write_root_pose 移动
            # (像 cabinet/微波炉那样)、不掉落；关节用轻 actuator held 住不耷拉。
            from isaaclab.actuators import ImplicitActuatorCfg
            from isaaclab.assets import ArticulationCfg

            spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True, enabled_self_collisions=False)
            cfg = ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                init_state=ArticulationCfg.InitialStateCfg(pos=p, rot=r),
                spawn=spawn,
                actuators={"hold": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=100.0, damping=10.0)},
            )
        else:
            # 静态件(纯视觉/单碰撞)：de-instance 让 instanced-proxy 碰撞体变成真 prim -> PhysX 注册静态碰撞
            # (否则碰撞体红色"未激活"、动态物体穿模)。spawn 时(物理 init 前)运行，安全。
            spawn.func = spawn_usd_deinstanced
            cfg = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                init_state=AssetBaseCfg.InitialStateCfg(pos=p, rot=r),
                spawn=spawn,
            )
        setattr(scene, name, cfg)
        print(f"[scene_props] added asset '{name}' from {usd_path} (rigid={rigid}, pos={p})", flush=True)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[scene_props] add_usd_asset WARNING: '{name}' from {usd_path} failed: {exc}", flush=True)
        return False


def spawn_kinematic_deinstanced(prim_path: str, cfg, translation=None, orientation=None):
    """spawn USD -> 去实例化 -> 在根上加 KINEMATIC RigidBodyAPI + MassAPI。让 instanceable 的纯视觉
    资产(如机器人支架)也能成为可 write_root_pose 移动的 kinematic 刚体(否则实例代理上加不了 rigid
    body -> 'no rigid body' 崩)。返回 spawn 的 prim。"""
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    from pxr import UsdPhysics

    prim = spawn_from_usd(prim_path, cfg, translation, orientation)
    try:
        from isaaclab.sim.utils import find_matching_prims

        stage = prim.GetStage()
        tops = find_matching_prims(prim_path, stage) or [prim]
        for top in tops:
            if top is None or not top.IsValid():
                continue
            _deinstance_subtree(top, include_root=True)
            rb = UsdPhysics.RigidBodyAPI.Apply(top)
            rb.CreateKinematicEnabledAttr().Set(True)
            UsdPhysics.MassAPI.Apply(top).CreateMassAttr().Set(10.0)
        print(f"[scene_props] kinematic-deinstanced {prim_path} -> movable kinematic body", flush=True)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[scene_props] kinematic-deinstance WARNING: {prim_path}: {exc}", flush=True)
    return prim


def spawn_usd_deinstanced(prim_path: str, cfg, translation=None, orientation=None):
    """静态道具的 spawn func：spawn 后把整棵子树去实例化，让 instanced-proxy 碰撞体变成真 prim。
    这样它们的【静态碰撞】被 PhysX 注册(动态物体撞得到=绿色)，而不是穿模(红色未激活)。
    spawn 在物理 tensor view 创建之前 -> 改 collider 安全。失败不致命(照常 spawn)。"""
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd

    prim = spawn_from_usd(prim_path, cfg, translation, orientation)
    try:
        from isaaclab.sim.utils import find_matching_prims

        stage = prim.GetStage()
        tops = find_matching_prims(prim_path, stage) or [prim]
        total = 0
        for top in tops:
            if top is not None and top.IsValid():
                total += _deinstance_subtree(top, include_root=True)
        print(f"[scene_props] de-instanced static prop {prim_path}: {total} prim(s) -> static colliders active", flush=True)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[scene_props] de-instance static prop WARNING: {prim_path}: {exc}", flush=True)
    return prim


# ---------------------------------------------------------------------------
# 把手碰撞细化（spawn 时，物理 init 之前 -> 不破坏 physics view）
# ---------------------------------------------------------------------------
# 含把手的可动 link，spawn 后立即设 convexDecomposition(中等)，让把手有真实形状。
# 通过自定义 spawn func 实现：spawn 原 prim -> 立即改 approximation。此时还没创建物理 tensor view，
# 所以重建 collider 安全(运行时改会使 view 失效 -> 整个 env 崩，已验证)。
_HANDLE_LINKS_BY_ASSET = {
    "coffeemachine": ["link_4", "link_5", "link_15"],  # link_4/5 把手 + link_15 机身/机架(全 SDF,放杯凹槽开放)
    "microwave": ["link_0"],                  # 微波炉门(含门把手)
    "cabinet": ["link_0", "link_1", "link_2"],  # 三抽屉(含抽屉拉手) -> SDF 保持抽屉内部中空
    "fridge": ["link_0", "link_1"],           # link_0 身体(SDF 保持内部中空) + link_1 门(box 代理)
    # 洗碗机：link_0 机身 + link_1 碗架(滑轨)。默认凸包把机身内部填实->碗架卡死滑不动+穿模。
    # 两件都是高精度真网格(机身 591v/碗架格栅 1030v) -> SDF 贴合真实形状、机身中空、碗架可滑出。
    "dishwasher": ["link_0", "link_1"],
}


FRIDGE_MIN_Z_LOCAL = -0.746   # 实测自 Fridge_10797/fridge.usd 局部 bbox 底部 z（未缩放）


def _repo_root():
    # depth-robust: search upward for the IsaacLab repo marker. The paper clone lives one dir deeper
    # than the original franka_v1_skill_lab, so a hardcoded parents[N] would resolve to projects/ not
    # the repo root (breaking shared-asset paths).
    from pathlib import Path

    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "isaaclab.sh").exists():
            return p
    return here.parents[3]


def replace_microwave_with_fridge(env_cfg, usd_path: str, scale=(0.5, 0.5, 0.5), ground_z: bool = True) -> bool:
    """把 microwave 成员的 spawn 换成冰箱 USD（保留成员名/位姿，改 usd+scale，删微波炉门把手代理）。

    保留成员名 "microwave" 以免破坏 scene_sync(按 manifest "Microwave" 项写位姿)和下游引用——
    冰箱因此自动继承微波炉在最新场景里的位姿。

    ground_z=True：把冰箱 z 压到落地(用于全新替换，微波炉默认 z 会让冰箱悬空)。
    ground_z=False：保留当前 z(用于已保存场景——用户可能把冰箱放在柜子上等高位，不能强行落地)。
    """
    from pathlib import Path

    scene = getattr(env_cfg, "scene", None)
    mw = getattr(scene, "microwave", None)
    spawn = getattr(mw, "spawn", None) if mw is not None else None
    if spawn is None or not hasattr(spawn, "usd_path"):
        print("[scene_props] replace fridge: no microwave member, skip.", flush=True)
        return False
    p = Path(usd_path)
    if not p.is_absolute():
        p = _repo_root() / usd_path
    if not p.is_file():
        print(f"[scene_props] replace fridge WARNING: fridge usd not found: {p} (convert it first).", flush=True)
        return False
    spawn.usd_path = str(p.resolve())
    if hasattr(spawn, "scale"):
        spawn.scale = tuple(float(v) for v in scale)
    # 注意：prim 路径保持 /Microwave 不变。曾试过改名成 /Fridge，headless 一切正常(prim 存在、
    # 14/14 网格可见)，但 GUI 实时视口对"改名 + 去实例化的 instanceable prim"渲染会丢几何(冰箱看不见)。
    # 改名纯属外观，不值得这风险 -> 只在资产面板把【显示名】改成 fridge(按 usd_path 识别)，成员/prim 全不动。
    # 冰箱继承微波炉位姿会悬空(微波炉 z=0.44)；ground_z 时用冰箱实测 bbox 把 z 设成落地(底部到 z=0)。
    # fridge.usd 局部 bbox 底部 z=-0.746(未缩放) -> 落地 z = 0.746 * scale_z。保留 x/y/rot。
    # ground_z=False(已保存场景)时不动 z：保留用户存的高位(如放在柜子上)。
    init_state = getattr(mw, "init_state", None)
    if ground_z and init_state is not None and getattr(init_state, "pos", None) is not None:
        x, y, _ = (float(v) for v in init_state.pos)
        base_z = -FRIDGE_MIN_Z_LOCAL * float(scale[2])
        init_state.pos = (x, y, base_z)
        print(f"[scene_props] fridge grounded to z={base_z:.3f} (no saved pose).", flush=True)
    elif not ground_z:
        cz = float(init_state.pos[2]) if (init_state is not None and init_state.pos is not None) else 0.0
        print(f"[scene_props] fridge keeps saved z={cz:.3f} (placed by user).", flush=True)
    try:
        if hasattr(spawn, "semantic_tags"):
            spawn.semantic_tags = [("class", "refrigerator")]
    except Exception:
        pass
    # 微波炉门把手代理(几何/偏移按微波炉算)对冰箱无效 -> 禁用(置 None，scene manager 跳过)
    if hasattr(scene, "microwave_handle_proxy"):
        scene.microwave_handle_proxy = None
    print(f"[scene_props] replaced microwave -> fridge: {p.name} scale={tuple(scale)}", flush=True)
    return True


def install_collision_monitor(env_cfg) -> bool:
    """Add a ContactSensor over ALL robot links so a runtime monitor can read collision forces.

    Enables contact reporting on the robot spawn and attaches one sensor matching every Robot rigid
    body; ``scene['robot_contact'].data.net_forces_w[env, link]`` is the net contact force on that link
    (non-zero => it is touching something). Lets the skill monitor catch the robot BODY accidentally
    bumping the scene (e.g. an arm/wrist link knocking the fridge door shut on retreat) while not
    grasping. Returns True if installed.
    """
    from isaaclab.sensors import ContactSensorCfg

    scene = getattr(env_cfg, "scene", None)
    robot = getattr(scene, "robot", None)
    if robot is None or getattr(robot, "spawn", None) is None:
        print("[scene_props] collision monitor: no robot, skip.", flush=True)
        return False
    robot.spawn.activate_contact_sensors = True   # enable PhysX contact reporting on the robot bodies
    scene.robot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,        # every physics step
        history_length=0,
        track_air_time=False,
    )
    print("[scene_props] collision monitor: robot contact sensor installed "
          "(scene['robot_contact'].data.net_forces_w).", flush=True)
    return True


# 小刀刀片关节 joint_0 行程(来自 knife.usd)：[-167.4°, 0°] = [-2.921, 0] rad。
# 0° 附近 = 刀片展开(打开)；接近下限 = 刀片收进刀柄(合住)。base cfg 默认 -0.2(几乎全开)。
# -2.85 (≈-163°) 已经收拢看着是闭合，且离硬下限 -2.921 留 0.07 rad 余量，避免顶死硬限位反弹回弹开。
KNIFE_JOINT_CLOSED_RAD = -2.85


def lock_knife(env_cfg, stiffness: float = 3000.0, damping: float = 200.0, effort: float = 300.0,
               closed: bool = True) -> bool:
    """锁死小刀关节：把 knife 的 actuator 拉成【极硬位置驱动】，刀片永久保持在合住角不动。

    closed=True 时把刀片 init 关节角设到“合住”(KNIFE_JOINT_CLOSED_RAD)，再用很高的 stiffness/effort
    把它牢牢钉在那里——即使被机器人碰到/重力作用也不会自动弹开(用户要求永久闭合)。
    注意 base cfg 里 blade_lock 的 effort_limit_sim=0(无力矩)，必须强制给力矩否则驱动不动 = 锁不住。
    """
    scene = getattr(env_cfg, "scene", None)
    knife = getattr(scene, "knife", None)
    acts = getattr(knife, "actuators", None) if knife is not None else None
    if not acts:
        print("[scene_props] lock knife: no knife actuators, skip.", flush=True)
        return False
    # 刀片合住：把 init_state.joint_pos 的 joint_0 设到收拢角(高 stiffness 会把它驱动并保持在这里)。
    if closed:
        ist = getattr(knife, "init_state", None)
        jp = dict(getattr(ist, "joint_pos", {}) or {}) if ist is not None else None
        if jp is not None:
            # 覆盖所有匹配 joint_0 的键(防 base cfg 用别的写法)，确保默认关节角=合住角。
            jp = {(k if k != "joint_0" else "joint_0"): v for k, v in jp.items()}
            jp["joint_0"] = float(KNIFE_JOINT_CLOSED_RAD)
            ist.joint_pos = jp
            print(f"[scene_props] knife blade set to CLOSED (joint_0={KNIFE_JOINT_CLOSED_RAD:.2f} rad).", flush=True)
    for a in acts.values():
        a.stiffness = float(stiffness)
        a.damping = float(damping)
        # 强制给力矩(base 是 0=无力矩，驱动不动)：无论原值是不是 0 都覆盖成够大的力矩，保证锁得住。
        if hasattr(a, "effort_limit_sim"):
            a.effort_limit_sim = float(effort)
        if hasattr(a, "velocity_limit_sim"):
            a.velocity_limit_sim = 2.0
    print(f"[scene_props] knife blade LOCKED CLOSED (stiffness={stiffness:.0f}, effort={effort:.0f}).", flush=True)
    return True


def _handle_links_for(usd_path: str) -> list[str]:
    up = (usd_path or "").lower()
    for key, links in _HANDLE_LINKS_BY_ASSET.items():
        if key in up:
            return links
    return []


def spawn_usd_refined(prim_path: str, cfg, translation=None, orientation=None):
    """自定义 spawn func：spawn 原 USD 后，对把手 link 设 convexDecomposition(中等精度)。

    在物理 tensor view 创建之前运行(scene 构建阶段)，所以改 collider 安全。失败不致命：
    打印 warning，env 照常构建(只是该 link 没细化)。
    """
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd

    # spawn_from_usd 被 @clone 装饰：prim_path 可能是正则命名空间 (如 /World/envs/env_.*/Cabinet)，
    # 它内部解析+克隆到各 env 并返回一个 prim。所以这里【不能】用 prim_path 字符串去 GetPrimAtPath
    # (正则不是合法 SdfPath -> null prim -> GetChildren 崩)；要用 find_matching_prims 拿到各 env
    # 下的【具体】prim 再处理。
    prim = spawn_from_usd(prim_path, cfg, translation, orientation)
    stage = prim.GetStage()
    try:
        from isaaclab.sim.utils import find_matching_prims

        tops = find_matching_prims(prim_path, stage)
    except Exception:
        tops = []
    if not tops:
        tops = [prim]

    # 关键：SAPIEN 资产 make_instanceable=true，collider 在实例化原型里。PhysX 的碰撞 overlay
    # (visualizationDisplayColliders) 不渲染实例代理(instance proxy)的 collider，而且实例代理
    # 也改不动 approximation。所以把【整个家电 prim】(root-inclusive)去实例化 —— 去实例化后所有
    # collider 变成真实 prim：overlay 能显示，modify 也能改。逐个 env 的具体 prim 处理。
    total = 0
    for top in tops:
        if top is not None and top.IsValid():
            total += _deinstance_subtree(top, include_root=True)
    print(
        f"[scene_props] de-instanced {prim_path}: {total} prim(s) across {len(tops)} env(s) "
        "-> colliders now visible", flush=True,
    )

    links = _handle_links_for(getattr(cfg, "usd_path", ""))
    if not links:
        return prim
    usd_path = getattr(cfg, "usd_path", "")
    try:
        from isaaclab.sim import schemas
        from isaaclab.sim.schemas import schemas_cfg

        for top in tops:
            if top is None or not top.IsValid():
                continue
            base = str(top.GetPath())
            for link in links:
                link_path = f"{base}/{link}"
                if not stage.GetPrimAtPath(link_path).IsValid():
                    print(f"[scene_props] handle refine: link not found {link_path}", flush=True)
                    continue
                method = _link_collision_method(usd_path, link)
                try:
                    if method == "door_boxes":
                        # 冰箱门：轴对齐 box 代理(门板平整无倾角 + 把手竖杆留缝)，夹爪可抓。
                        nb = _install_replacement_boxes(stage, link_path, _FRIDGE_DOOR_BOXES)
                        print(f"[scene_props] {link_path} -> {nb} box(es) (door panel+handle)", flush=True)
                    elif method == "body_boxes":
                        # 冰箱身体：碰撞 mesh 是低多边形实心箱(8+28面),任何 approximation 都填实。
                        # 换成 5 面墙盒(底/顶/背/左/右),前面(+Z,门口)敞开 -> 夹爪能进、东西落底板、真中空。
                        nb = _install_replacement_boxes(stage, link_path, _FRIDGE_BODY_BOXES)
                        print(f"[scene_props] {link_path} -> {nb} wall box(es) (hollow, front +Z open)", flush=True)
                    elif method == "coffee_body_boxes":
                        # 咖啡机机身：只留 后/左/底 墙盒，前面(+Z)和右面(-X)敞开 -> 右前方放杯区可进。
                        nb = _install_replacement_boxes(stage, link_path, _COFFEE_BODY_BOXES)
                        print(f"[scene_props] {link_path} -> {nb} wall box(es) (front +Z / right -X open)", flush=True)
                    elif method == "sdf":
                        # 真网格件(咖啡机/微波炉/抽屉)：默认 convexHull 把凹处填成实心凸包->挡机器人。
                        # SDF 精确贴合真实表面(凹凸都还原),不填洞、不挡操作,且能用于动态 articulation link。
                        # modify_* 是 @apply_nested：作用到 link 下所有 collider mesh。
                        schemas.modify_mesh_collision_properties(
                            link_path, schemas_cfg.SDFMeshPropertiesCfg(sdf_resolution=256)
                        )
                        print(f"[scene_props] {link_path} -> sdf (tight fit to true shape)", flush=True)
                    elif method == "trimesh":
                        # 固定底座的机身/机架(静态体)：三角网格(approximation="none")精确贴合真实形状,
                        # 不把放杯凹槽等凹处填实, 且最轻(无 SDF 网格/无凸分解计算)。仅适用静态体。
                        schemas.modify_mesh_collision_properties(
                            link_path, schemas_cfg.TriangleMeshPropertiesCfg()
                        )
                        print(f"[scene_props] {link_path} -> trimesh/none (exact static shape, cup nook open)", flush=True)
                    else:
                        schemas.modify_mesh_collision_properties(
                            link_path,
                            schemas_cfg.ConvexDecompositionPropertiesCfg(max_convex_hulls=16, hull_vertex_limit=32),
                        )
                        print(f"[scene_props] {link_path} -> convexDecomposition", flush=True)
                except Exception as exc:  # pragma: no cover
                    print(f"[scene_props] handle refine WARNING: {link_path} failed: {exc}", flush=True)

            # 抽屉把手太滑 -> 在【场景编辑里保存的把手 pose 位置】加一个小高摩擦碰撞块, 夹爪好夹。
            if "cabinet" in (usd_path or "").lower():
                blocks = _grasp_block_local_offsets()
                for link, (local_pos, hname) in blocks.items():
                    lp = f"{base}/{link}"
                    if stage.GetPrimAtPath(lp).IsValid():
                        # 半边长取用户在属性框改后存的(grasp_blocks.json)，否则默认。
                        half = grasp_block_half(hname, (0.018, 0.018, 0.030))
                        _install_grasp_block(stage, lp, local_pos, half=half, name="GraspBlock")
            # 咖啡机拉手也太滑(细杆)，旋转时夹爪会打滑 -> 在保存的把手 pose(link_5，水平摆动杠杆)加高摩擦块。
            if "coffeemachine" in (usd_path or "").lower():
                cpos = _coffee_block_local_offset()
                lp = f"{base}/link_5"
                if cpos is not None and stage.GetPrimAtPath(lp).IsValid():
                    half = grasp_block_half("handle_coffee_lever", (0.020, 0.020, 0.030))
                    _install_grasp_block(stage, lp, cpos, half=half,
                                         friction=2.0, name="GraspBlock")
    except Exception as exc:  # pragma: no cover
        print(f"[scene_props] handle refine import failed: {exc}", flush=True)

    # Dishwasher slider limit: tighten lowerLimit so the rack cannot be pulled fully out and clip
    # the body (self-collision is off so an over-travel would pass the rack through the shell = 穿模).
    if "dishwasher" in (getattr(cfg, "usd_path", "") or "").lower():
        try:
            from pxr import Usd
            for top in tops:
                if top is None or not top.IsValid():
                    continue
                n_lim = 0
                for p in Usd.PrimRange(top):
                    if p.GetTypeName() == "PhysicsPrismaticJoint":
                        attr = p.GetAttribute("physics:lowerLimit")
                        if attr and attr.Get() is not None and float(attr.Get()) < -0.35:
                            attr.Set(-0.35)
                            n_lim += 1
                print(f"[scene_props] dishwasher {top.GetPath()}: {n_lim} slider lowerLimit clamped to -0.35", flush=True)
        except Exception as exc:  # pragma: no cover
            print(f"[scene_props] dishwasher limit WARNING: {exc}", flush=True)
    return prim


def _link_collision_method(usd_path: str, link: str) -> str:
    """决定某 link 的碰撞细化方式：

    - "door_boxes": 冰箱门 link_1 -> 门板平整盒 + 把手竖杆盒(留缝可抓)。抓取把手=高精度。
    - "body_boxes": 冰箱身体 link_0(碰撞 mesh=低多边形实心箱) -> 5 面墙盒，前面敞开 -> 真中空可进。
    - "sdf":        其余真网格件(咖啡机/微波炉门/抽屉) -> SDF 精确贴合真实形状，不把凹处填实、
                    不挡机器人操作，复杂度适中。这就是"其他地方尽量贴合原形状(不大量增加复杂度)"。
    - "convexDecomposition": 兜底(一般用不到)。

    全局原则(用户要求)：抓取门把手=高精度(box 留缝)；其余处=贴合真实形状、低复杂度(SDF)。
    注：各资产碰撞源【都是完整真网格】(面数==视觉)，只是默认 approximation 为空被当 convexHull
    处理->填凹挡操作；改 SDF 即贴合。唯独冰箱身体 mesh 本身是实心低多边形箱，必须显式墙盒。
    """
    up = (usd_path or "").lower()
    if "fridge" in up:
        return "door_boxes" if link == "link_1" else "body_boxes"   # link_0 实心箱 -> 墙盒中空
    # 咖啡机机身 link_15：SDF 也没能让放杯区可进(动态 link 上 SDF 仍贴外壳)。按用户要求改墙盒，
    # 只保留 后(-Z)/左(+X)/底，前面(+Z)和右面(-X)整面敞开 -> 机器人从右前方进、放杯区无碰撞。
    if "coffeemachine" in up and link == "link_15":
        return "coffee_body_boxes"
    # 柜子抽屉把手是细薄特征：SDF 分辨率不够时等值面向外鼓 -> 碰撞跑到把手外(中/上抽屉明显)。
    # convexDecomposition 用紧凑凸块贴把手、不鼓，且抽屉开关技能本来就用它验证过。
    if "cabinet" in up:
        return "convexDecomposition"
    return "sdf"   # 咖啡机把手 link_4/5、微波炉门(可动真网格)：SDF 贴合真实表面


# 冰箱门 link_1 局部坐标(未缩放；prim 上的 0.5 缩放会自动应用)下的代理碰撞盒。
# 实测自 fridge.usd：门板 door_frame_9 X[-0.86,0.03] Y[-1.48,0]；把手 handle_10
# X[-0.84,-0.77] Y[-0.87,0] Z 凸到 0.113。门板前表面取 Z=0.005(平整无倾角)，把手竖杆取
# Z[0.075,0.113]，两者之间 0.005->0.075 留缝(未缩放 0.07 -> 实际 ~3.5cm)给手指穿过。
# 抓不住就调这里：把手 z 下界调小=缝更大；x/z 范围=杆更粗/细。
_FRIDGE_DOOR_BOXES = {
    "door_panel_proxy":  {"x": (-0.86, 0.03),  "y": (-1.48, 0.0), "z": (-0.071, 0.005)},
    "door_handle_proxy": {"x": (-0.85, -0.76), "y": (-0.92, 0.05), "z": (0.075, 0.118)},
}

# 冰箱身体 link_0 局部坐标(未缩放)下的 5 面墙盒。实测身体外形 X[-0.44,0.44] Y[-0.75,0.77]
# Z[-0.50,0.46]；Y=竖直，门在 +Z 面(门 center Z=0.483) -> 前面(+Z)敞开，夹爪从这里进。
# 墙厚 ~0.06(未缩放 -> 实际 ~3cm)。东西放进去落在 body_floor 上。内部净空 X[-0.38,0.38]
# Y[-0.69,0.71] Z[-0.44,0.46](缩放后 ~0.38x0.70x0.45m)足够夹爪+方块。
_FRIDGE_BODY_BOXES = {
    "body_floor": {"x": (-0.44, 0.44),  "y": (-0.75, -0.69), "z": (-0.50, 0.46)},
    "body_top":   {"x": (-0.44, 0.44),  "y": (0.71, 0.77),   "z": (-0.50, 0.46)},
    "body_back":  {"x": (-0.44, 0.44),  "y": (-0.75, 0.77),  "z": (-0.50, -0.44)},
    "body_left":  {"x": (-0.44, -0.38), "y": (-0.75, 0.77),  "z": (-0.50, 0.46)},
    "body_right": {"x": (0.38, 0.44),   "y": (-0.75, 0.77),  "z": (-0.50, 0.46)},
    # 前面 (+Z, z~0.46) 不放墙 -> 门口敞开
}

# 咖啡机机身 link_15 局部坐标(未缩放)墙盒。实测 X[-0.52,0.52] Y[-0.68,0.69](Y 竖直)
# Z[-0.59,0.61]；link_4/5(旋钮/出杯口)在 +Z -> 前面=+Z；放杯口(link_5)在 -X -> 右=-X。
# 只保留 后(-Z)/左(+X)/底(-Y)；前(+Z)、右(-X)、顶(+Y)整面敞开 -> 机器人从右前方进、放杯区无碰撞。
# 若"右边还堵"=右其实是+X：把 body_left 改成留 -X、敞开 +X(对调下面 left 的 x 范围)。
_COFFEE_BODY_BOXES = {
    "body_back":  {"x": (-0.52, 0.52), "y": (-0.68, 0.69),  "z": (-0.59, -0.53)},  # 后 -Z
    "body_left":  {"x": (0.46, 0.52),  "y": (-0.68, 0.69),  "z": (-0.59, 0.61)},   # 左 +X
    "body_floor": {"x": (-0.52, 0.52), "y": (-0.68, -0.62), "z": (-0.59, 0.61)},   # 底 -Y
}


def _install_replacement_boxes(stage, link_path: str, boxes: dict) -> int:
    """禁用 link 原有网格碰撞，换成一组轴对齐 box(在 link 局部系，必然无倾角)。返回新建 box 数。

    box 设为不可见(visibility=invisible)，但 PhysX collider overlay 仍会画出来(便于调试)。
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    link = stage.GetPrimAtPath(link_path)
    if not link.IsValid():
        return 0

    # 1) 关掉 link 下所有原网格碰撞(去实例化后可编辑)
    for p in Usd.PrimRange(link):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)

    # 2) 加 box
    n = 0
    for name, b in boxes.items():
        xr, yr, zr = b["x"], b["y"], b["z"]
        cube = UsdGeom.Cube.Define(stage, f"{link_path}/{name}")
        cube.GetSizeAttr().Set(2.0)   # 立方体 ±1，再用 scale 缩到半边长
        cx, cy, cz = (xr[0] + xr[1]) / 2, (yr[0] + yr[1]) / 2, (zr[0] + zr[1]) / 2
        hx, hy, hz = (xr[1] - xr[0]) / 2, (yr[1] - yr[0]) / 2, (zr[1] - zr[0]) / 2
        xf = UsdGeom.Xformable(cube)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(float(cx), float(cy), float(cz)))
        xf.AddScaleOp().Set(Gf.Vec3f(float(hx), float(hy), float(hz)))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()   # 渲染不显示；collider overlay 仍可见
        n += 1
    return n


def _grasp_block_local_offsets() -> dict:
    """读 grasp_poses.json，返回 {drawer_link: (local_pos, name)}：把手块要放的 link 局部位置(米)。
    link_0=handle_top_drawer, link_1=handle_bottom_drawer, link_2=handle_middle_drawer。"""
    import json
    from pathlib import Path

    want = {"handle_top_drawer": "link_0", "handle_bottom_drawer": "link_1",
            "handle_middle_drawer": "link_2"}
    out = {}
    try:
        p = Path(__file__).resolve().parents[2] / (
            "franka_v1_skill_lab/scene/saved_scenes/v1_active/grasp_poses.json")
        poses = json.loads(p.read_text(encoding="utf-8")).get("poses", {})
        for hname, link in want.items():
            e = poses.get(hname)
            if e is not None and e.get("link") == link:
                out[link] = (tuple(float(v) for v in e["pos"]), hname)
    except Exception as exc:  # pragma: no cover
        print(f"[scene_props] grasp-block offsets read failed: {exc}", flush=True)
    return out


def _coffee_block_local_offset():
    """读 grasp_poses.json 的 handle_coffee_lever(link_4)局部位置(米)；用于在咖啡拉手处放抓取块。"""
    import json
    from pathlib import Path

    try:
        p = Path(__file__).resolve().parents[2] / (
            "franka_v1_skill_lab/scene/saved_scenes/v1_active/grasp_poses.json")
        e = json.loads(p.read_text(encoding="utf-8")).get("poses", {}).get("handle_coffee_lever")
        if e is not None and e.get("link") == "link_5":
            return tuple(float(v) for v in e["pos"])
    except Exception as exc:  # pragma: no cover
        print(f"[scene_props] coffee block offset read failed: {exc}", flush=True)
    return None


def _install_grasp_block(stage, link_path: str, local_pos, half=(0.018, 0.018, 0.030),
                         friction: float = 1.4, name: str = "GraspBlock") -> bool:
    """在 link 下、把手的【link 局部米制位置】放一个小高摩擦碰撞块(夹爪有平面+摩擦可夹牢，解决太滑)。
    link 有非均匀缩放 -> 子节点 translate/scale 会被缩放，所以这里用累计缩放 S 反补(t=pos/S)。
    放完打印块世界位 vs 目标把手世界位，便于核对放置是否正确。"""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

    link = stage.GetPrimAtPath(link_path)
    if not link.IsValid():
        print(f"[scene_props] grasp-block: link not found {link_path}", flush=True)
        return False
    m = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(link)
    sx = Gf.Vec3d(m[0][0], m[0][1], m[0][2]).GetLength() or 1.0
    sy = Gf.Vec3d(m[1][0], m[1][1], m[1][2]).GetLength() or 1.0
    sz = Gf.Vec3d(m[2][0], m[2][1], m[2][2]).GetLength() or 1.0
    cube = UsdGeom.Cube.Define(stage, f"{link_path}/{name}")
    cube.GetSizeAttr().Set(2.0)                                  # ±1, 再用 scale 缩到半边长
    xf = UsdGeom.Xformable(cube); xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(local_pos[0] / sx, local_pos[1] / sy, local_pos[2] / sz))
    xf.AddScaleOp().Set(Gf.Vec3f(half[0] / sx, half[1] / sy, half[2] / sz))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    cube.GetDisplayColorAttr().Set([Gf.Vec3f(0.9, 0.2, 0.2)])    # 可见红块，便于目视
    # 高摩擦物理材质，绑到块上(physics purpose)
    try:
        mat = UsdShade.Material.Define(stage, f"{link_path}/{name}_mat")
        UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        pm = UsdPhysics.MaterialAPI(mat.GetPrim())
        pm.CreateStaticFrictionAttr().Set(float(friction))
        pm.CreateDynamicFrictionAttr().Set(float(friction))
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(
            mat, bindingStrength=UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics")
    except Exception as exc:  # pragma: no cover
        print(f"[scene_props] grasp-block friction material failed: {exc}", flush=True)
    # 核对：块世界位 vs 目标(link 世界 ∘ 局部偏移)
    try:
        bw = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
            cube.GetPrim()).ExtractTranslation()
        tgt = m.Transform(Gf.Vec3d(local_pos[0] / sx, local_pos[1] / sy, local_pos[2] / sz))
        print(f"[scene_props] grasp-block @ {link_path}/{name}: world=({bw[0]:.3f},{bw[1]:.3f},{bw[2]:.3f}) "
              f"target=({tgt[0]:.3f},{tgt[1]:.3f},{tgt[2]:.3f}) scale=({sx:.2f},{sy:.2f},{sz:.2f})", flush=True)
    except Exception:
        pass
    return True


# ---------------------------------------------------------------------------
# 可在【官方属性框】编辑的把手碰撞方块（静态编辑器）
# ---------------------------------------------------------------------------
# 4 个抓取碰撞方块：抽屉 top/middle/bottom + 咖啡机拉手。member/prim/link 与测试模式
# (spawn_usd_refined 里的 _grasp_block_local_offsets / _coffee_block_local_offset) 完全一致。
# 抽屉 link 映射：top->link_0, middle->link_2, bottom->link_1；咖啡拉手->link_5(水平摆杆)。
# 默认半边长(米, 世界尺度)：用户可在属性框改 scale，保存时回写到 grasp_blocks.json。
GRASP_BLOCK_NAME = "GraspBlock"
_GRASP_BLOCK_SPECS = [
    {"hname": "handle_top_drawer",    "member": "cabinet",        "prim": "Cabinet",       "link": "link_0", "half": (0.018, 0.018, 0.030), "friction": 1.4},
    {"hname": "handle_middle_drawer", "member": "cabinet",        "prim": "Cabinet",       "link": "link_2", "half": (0.018, 0.018, 0.030), "friction": 1.4},
    {"hname": "handle_bottom_drawer", "member": "cabinet",        "prim": "Cabinet",       "link": "link_1", "half": (0.018, 0.018, 0.030), "friction": 1.4},
    {"hname": "handle_coffee_lever",  "member": "coffee_machine", "prim": "CoffeeMachine", "link": "link_5", "half": (0.020, 0.020, 0.030), "friction": 2.0},
]


def _v1_active_dir():
    from pathlib import Path
    return Path(__file__).resolve().parents[2] / "franka_v1_skill_lab/scene/saved_scenes/v1_active"


def load_grasp_block_sizes() -> dict:
    """读 grasp_blocks.json -> {hname: {"half":[3], "link":str}}。缺失返回 {}。"""
    import json
    try:
        p = _v1_active_dir() / "grasp_blocks.json"
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8")).get("blocks", {}) or {}
    except Exception as exc:  # pragma: no cover
        print(f"[scene_props] load grasp_blocks.json failed: {exc}", flush=True)
    return {}


def grasp_block_half(hname: str, default) -> tuple:
    """取某把手方块的半边长：优先 grasp_blocks.json(用户在属性框改后存的)，否则用 default。"""
    e = load_grasp_block_sizes().get(hname)
    if e and isinstance(e.get("half"), (list, tuple)) and len(e["half"]) == 3:
        return tuple(float(v) for v in e["half"])
    return tuple(float(v) for v in default)


def _grasp_pose_local_pos(hname: str, link: str):
    """读 grasp_poses.json 里某把手的参考系局部 pos(米)。link 不匹配/缺失返回 None。"""
    import json
    try:
        p = _v1_active_dir() / "grasp_poses.json"
        e = json.loads(p.read_text(encoding="utf-8")).get("poses", {}).get(hname)
        if e is not None and e.get("link") == link:
            return tuple(float(v) for v in e["pos"])
    except Exception as exc:  # pragma: no cover
        print(f"[scene_props] grasp pose pos read failed for {hname}: {exc}", flush=True)
    return None


def install_editable_grasp_blocks(stage, env_root: str = "/World/envs/env_0") -> list:
    """在【静态编辑器】里把 4 个把手碰撞方块生成成【可在官方属性框编辑】的 prim。

    SAPIEN 资产是 instanceable -> link 子节点是只读 instance proxy(属性框改不动、也加不了子 prim)。
    所以先把家电整体去实例化(include_root)，再在对应 link 下放红色高摩擦碰撞块(复用 _install_grasp_block)。
    方块位置取 grasp_poses.json、半边长取 grasp_blocks.json(没有则用默认)。返回 [{hname, prim_path, link}]。
    """
    created = []
    for prim_name in {s["prim"] for s in _GRASP_BLOCK_SPECS}:
        top = stage.GetPrimAtPath(f"{env_root}/{prim_name}")
        if top.IsValid():
            _deinstance_subtree(top, include_root=True)
    for s in _GRASP_BLOCK_SPECS:
        link_path = f"{env_root}/{s['prim']}/{s['link']}"
        if not stage.GetPrimAtPath(link_path).IsValid():
            print(f"[scene_props] editable grasp block: link not found {link_path}", flush=True)
            continue
        bp = f"{link_path}/{GRASP_BLOCK_NAME}"
        if stage.GetPrimAtPath(bp).IsValid():
            # 已存在(从已存 USD 加载)：保留用户在属性框里改过的 transform，不重建覆盖。
            created.append({"hname": s["hname"], "prim_path": bp, "link": s["link"]})
            continue
        pos = _grasp_pose_local_pos(s["hname"], s["link"])
        if pos is None:
            print(f"[scene_props] editable grasp block: no grasp pose for {s['hname']}, skip", flush=True)
            continue
        half = grasp_block_half(s["hname"], s["half"])
        if _install_grasp_block(stage, link_path, pos, half=half, friction=s["friction"], name=GRASP_BLOCK_NAME):
            created.append({"hname": s["hname"], "prim_path": f"{link_path}/{GRASP_BLOCK_NAME}", "link": s["link"]})
    print(f"[scene_props] installed {len(created)}/{len(_GRASP_BLOCK_SPECS)} editable grasp block(s) "
          "(select in viewport, edit in the native Property panel, then Save).", flush=True)
    return created


def read_grasp_blocks_from_stage(stage, env_root: str = "/World/envs/env_0") -> dict:
    """从 stage 读回各把手碰撞方块当前 transform，换算回 link 局部米制 pos + 半边长(米)。

    _install_grasp_block 写的是 translate=pos/linkScale、scale=half/linkScale(cube size=2)，
    这里按 link 世界缩放反推：pos = translate*linkScale, half = scale*linkScale*(size/2)。
    返回 {hname: {"member","link","pos":[3],"half":[3]}}。供保存时回写。
    """
    from pxr import Gf, Usd, UsdGeom

    out = {}
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for s in _GRASP_BLOCK_SPECS:
        link_path = f"{env_root}/{s['prim']}/{s['link']}"
        bp = f"{link_path}/{GRASP_BLOCK_NAME}"
        cube_prim = stage.GetPrimAtPath(bp)
        link_prim = stage.GetPrimAtPath(link_path)
        if not (cube_prim.IsValid() and link_prim.IsValid()):
            continue
        m = cache.GetLocalToWorldTransform(link_prim)
        sx = Gf.Vec3d(m[0][0], m[0][1], m[0][2]).GetLength() or 1.0
        sy = Gf.Vec3d(m[1][0], m[1][1], m[1][2]).GetLength() or 1.0
        sz = Gf.Vec3d(m[2][0], m[2][1], m[2][2]).GetLength() or 1.0
        ops = {op.GetName(): op for op in UsdGeom.Xformable(cube_prim).GetOrderedXformOps()}
        t = ops.get("xformOp:translate")
        sc = ops.get("xformOp:scale")
        tv = t.Get() if t is not None else None
        scv = sc.Get() if sc is not None else None
        if tv is None or scv is None:
            continue
        size = 2.0
        try:
            sa = UsdGeom.Cube(cube_prim).GetSizeAttr().Get()
            if sa:
                size = float(sa)
        except Exception:
            pass
        pos = [float(tv[0]) * sx, float(tv[1]) * sy, float(tv[2]) * sz]
        half = [abs(float(scv[0]) * sx * (size / 2.0)),
                abs(float(scv[1]) * sy * (size / 2.0)),
                abs(float(scv[2]) * sz * (size / 2.0))]
        out[s["hname"]] = {"member": s["member"], "link": s["link"], "pos": pos, "half": half}
    return out


def save_grasp_blocks(blocks: dict) -> bool:
    """把读回的方块 pos 写进 grasp_poses.json(对应把手 entry 的 pos)，half 写进 grasp_blocks.json。

    这样：抓取技能(读 grasp_poses.json)用新位置；测试模式重生成方块时(grasp_block_half)用新尺寸。所见即所得。
    """
    import json

    if not blocks:
        return False
    base = _v1_active_dir()
    ok = True
    # 1) grasp_poses.json: 更新/新建把手 entry 的 pos
    try:
        gp = base / "grasp_poses.json"
        doc = json.loads(gp.read_text(encoding="utf-8")) if gp.is_file() else {
            "frame": "reference_local", "poses": {}}
        poses = doc.setdefault("poses", {})
        for hname, b in blocks.items():
            e = poses.get(hname)
            if e is None:
                e = {"kind": "handle", "member": b["member"], "link": b["link"],
                     "pos": [0.0, 0.0, 0.0], "quat": [0.0, 1.0, 0.0, 0.0]}
                poses[hname] = e
            e["pos"] = [float(v) for v in b["pos"]]
            e["link"] = b["link"]
        gp.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        print(f"[scene_props] grasp block positions -> {gp.name} ({len(blocks)} handle(s))", flush=True)
    except Exception as exc:  # pragma: no cover
        print(f"[scene_props] write grasp_poses.json failed: {exc}", flush=True)
        ok = False
    # 2) grasp_blocks.json: 记录 half(尺寸)
    try:
        sp = base / "grasp_blocks.json"
        doc2 = json.loads(sp.read_text(encoding="utf-8")) if sp.is_file() else {"blocks": {}}
        bl = doc2.setdefault("blocks", {})
        for hname, b in blocks.items():
            bl[hname] = {"half": [float(v) for v in b["half"]], "link": b["link"]}
        sp.write_text(json.dumps(doc2, indent=1) + "\n", encoding="utf-8")
        print(f"[scene_props] grasp block sizes -> {sp.name} ({len(blocks)} handle(s))", flush=True)
    except Exception as exc:  # pragma: no cover
        print(f"[scene_props] write grasp_blocks.json failed: {exc}", flush=True)
        ok = False
    return ok


def _deinstance_subtree(prim, include_root: bool = False) -> int:
    """把(prim 自身+)子树里所有 instanceable 的 prim 去实例化，返回去实例化的数量。

    include_root=True 时连 prim 自身也去实例化(必须：若顶层 prim 是 instance，其 link 子节点
    就是只读 instance proxy，改不了 collider 也显示不了)。去实例化后 GetChildren() 才会返回
    可编辑的组合子节点。
    """
    n = 0
    stack = [(prim, include_root)]
    while stack:
        cur, do_self = stack.pop()
        if do_self:
            try:
                if cur.IsInstanceable():
                    cur.SetInstanceable(False)
                    n += 1
            except Exception:
                pass
        for child in cur.GetChildren():
            stack.append((child, True))
    return n


def install_handle_refine(env_cfg) -> int:
    """把 cabinet/microwave/coffee_machine/dishwasher 的 spawn.func 换成 spawn_usd_refined。返回替换数。"""
    n = 0
    for name in ("cabinet", "microwave", "coffee_machine", "dishwasher"):
        member = getattr(getattr(env_cfg, "scene", None), name, None)
        spawn = getattr(member, "spawn", None)
        if spawn is not None and hasattr(spawn, "usd_path"):
            spawn.func = spawn_usd_refined
            n += 1
    if n:
        print(f"[scene_props] handle refine spawn-func installed on {n} appliances.", flush=True)
    return n
