# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""brain_interface.py —— 「大脑」(高层编排 / LLM Agent) 调用本场景模块的统一接口。

把整个 V1 场景模块封装成大脑能直接用的三个职责清晰的类（聚合在 BrainInterface 下）：

    ScenePerception  感知类：读仿真器里的一切（相机 RGB/Depth/RGBD、物体位姿、关节、
                     夹爪、把手位姿、FoundationPose 估计/输入）。只读，无副作用。

    SkillControl     控制类：给状态机下指令（= 大脑给状态机下命令）。和状态机对齐——
                     选技能 / 选目标物体 / 选目标位置 / 夹爪 / 暂停 / 继续 / 停止 / 中止。
                     底层复用 SkillExecutor（与 SkillTestController 完全同一套 backend 接线）。

    TaskCognition    认知类（= 你没考虑到的那层）：大脑闭环真正需要、但既不属于「读传感器」
                     也不属于「下技能指令」的东西——
                       1) 能力发现：有哪些技能 / 哪些可选目标 / 每个技能要什么参数+成功判据；
                       2) 前置校验：can_execute(skill, target) 这条命令现在能不能执行 + 为什么不能；
                       3) 执行反馈：技能跑到哪一步 / 成没成功 / 失败原因（闭环必需）；
                       4) 回合管理：reset(布局随机) / 保存场景为最新 / settle 稳定；
                       5) 可视化调试：目标位姿箭头 / 碰撞体 / 黄色标记块开关。

聚合入口 BrainInterface 持有三者 + 共享的 SceneSession / SkillExecutor，并提供：
    - tick()                推进一个仿真步（驱动当前技能 -> 动作 -> env.step），返回状态快照
    - run_skill(...)        阻塞式跑完一个技能并返回结果（大脑最常用的「下一条指令并等结果」）

两种用法：
    1) 自管理（headless 大脑 / 脚本）::

           from franka_v1_skill_lab.scene_interface.brain_interface import BrainInterface
           from franka_v1_skill_lab.scene_interface.config import SceneConfig, SceneMode

           brain = BrainInterface.launch(SceneConfig(mode=SceneMode.TEST, enable_cameras=True))
           brain.reset()
           rgb = brain.perception.get_rgb("front")
           res = brain.run_skill("open_door", target="microwave")   # 阻塞跑完
           print(brain.cognition.last_result())
           brain.close()

    2) 作为 test_mode 的 --controller（复用现有两行启动 + UI）::

           ./isaaclab.sh -p projects/franka_v1_skill_lab/scene_interface/test_mode_ui.py \
               --controller franka_v1_skill_lab.scene_interface.brain_interface:BrainInterface

       test_mode 会 BrainInterface(session)，每帧 step(session)->action 驱动机器人；
       此时大脑可在另一个线程/进程持有同一个 brain 对象下指令（control/cognition）。

约定（全程一致）：
    - 关节：7 维绝对关节弧度 q_des（与 robot.data.joint_pos[:7] 同序）。
    - 夹爪：对外用 0=open / 1=close（pi0.5/LeRobot 约定），内部统一换算成 env 的 +1/-1。
    - 四元数：一律 wxyz。
    - 坐标系：位姿默认世界系 frame="world"，可传 frame="base" 转到机器人基座系。

本模块顶层不 import isaac/torch/runtime（保持 plain venv 可 import）；所有重依赖延迟到
BrainInterface 实例化之后（那时 AppLauncher 已起、isaac 可用）。
"""

from __future__ import annotations

from typing import Any

from .config import SceneConfig
from .session import SceneSession, pi05_gripper_to_env_cmd

# ---- 对外暴露的可选目标（ASCII；与 SkillTestController 一致，单一可信源在状态机 target 配置）----
GRASP_TARGETS = ("cube_1", "cube_2", "cube_3", "knife")
DRAWER_TARGETS = ("middle_drawer", "top_drawer", "bottom_drawer")
DOOR_TARGETS = ("microwave",)          # 成员名仍是 microwave（即便被替换成冰箱）；门关节由门技能解析
PLACE_POINTS = ("point_a",)            # 默认放置点；真正落点由 target_pose / target_surface_xyz 决定
_DEFAULT_PLACE_XYZ = (0.45, 0.0, 0.06)

# 技能目录：大脑用来「发现能力」。needs_target=该技能是否必须指定目标；target_kind=目标取值域。
_SKILL_CATALOG = [
    {"skill": "grasp", "target_kind": "graspable", "needs_target": True,
     "params": {}, "desc": "Close gripper on a graspable object and lift it.",
     "success": "object grasped (gripper closed on it, object follows TCP)"},
    {"skill": "place", "target_kind": "place_point", "needs_target": False,
     "params": {"target_surface_xyz": "[x,y,z] env-local; default (0.45,0,0.06)"},
     "desc": "Release the currently held object at a support-surface point.",
     "success": "held object released near target_surface_xyz"},
    {"skill": "open_drawer", "target_kind": "drawer", "needs_target": True,
     "params": {"drawer_link": "link_1 (default)"},
     "desc": "Grasp drawer handle and pull the cabinet drawer open (pure-physical).",
     "success": "drawer prismatic joint past open threshold"},
    {"skill": "close_drawer", "target_kind": "drawer", "needs_target": True,
     "params": {"drawer_link": "link_1 (default)"},
     "desc": "Grasp drawer handle and push the cabinet drawer closed.",
     "success": "drawer prismatic joint near zero"},
    {"skill": "open_door", "target_kind": "door", "needs_target": True,
     "params": {}, "desc": "Grasp door handle and swing the revolute door open.",
     "success": "door hinge angle past open success angle"},
    {"skill": "close_door", "target_kind": "door", "needs_target": True,
     "params": {}, "desc": "Grasp door handle and swing the revolute door closed.",
     "success": "door hinge angle near closed"},
]
_TARGETS_BY_KIND = {
    "graspable": GRASP_TARGETS,
    "drawer": DRAWER_TARGETS,
    "door": DOOR_TARGETS,
    "place_point": PLACE_POINTS,
}
_CATALOG_BY_SKILL = {c["skill"]: c for c in _SKILL_CATALOG}


# =====================================================================================
#  内部共享核心：持有 session / provider / executor / dt 等，三个 facade 类共用。
# =====================================================================================
class _BrainCore:
    """三个 facade 类共享的运行时句柄 + 底层原语。实例化时（launch 之后）才 import 重依赖。"""

    def __init__(self, session: SceneSession):
        import os

        import torch

        # 旧状态机根目录上 sys.path（session 已做过一次，这里再保险一次）。
        from franka_v1_skill_lab.skill_runtime._legacy import ensure_legacy_on_path
        ensure_legacy_on_path()

        from runtime.base_skill import set_speed_scale
        from runtime.ik_joint_adapter import IKJointAdapter
        from runtime.skill_request import SkillRequest
        from runtime.skill_types import SkillType
        from runtime.target_registry import TargetRegistry
        from state_machine.skill_executor import JointBackendConfig, SkillExecutor

        self.torch = torch
        self.session = session
        self.env = session.env
        self.provider = session.provider
        self.dt = float(getattr(session, "_sim_dt", 0.02))
        self.SkillType = SkillType
        self.SkillRequest = SkillRequest

        self.registry = TargetRegistry(self.env.unwrapped.device)
        adapter = IKJointAdapter(self.env)
        backend = JointBackendConfig(
            mode="joint",
            grasp_backend="joint_ik",
            place_backend="joint_ik",
            drawer_backend="ik_pull",          # 纯物理抓+拉/推（无策略、无关节-目标作弊）
            adapter=adapter,
            drawer_env=self.env,
            arm_joint_ids=self.provider._arm_joint_ids,
            drawer_joint_name="joint_0",
        )
        self.executor = SkillExecutor(
            self.registry, log_path="logs/skill_brain/results.jsonl", backend=backend
        )
        applied = set_speed_scale(float(os.environ.get("SKILL_TEST_SPEED", "5.0")))
        print(f"[brain] skill speed scale = {applied:.1f}x", flush=True)

        # 控制状态（mirror SkillTestController）
        self._pending = None                   # 待启动的 SkillRequest（compute_action 里真正 start）
        self._want_stop = False
        self._manual = None                    # 一次性手动关节透传 (q7, gripper)
        self._freed: set = set()               # 已经在运行时 free 过的 (asset, joint)

    # --- skill 名/枚举互转 -------------------------------------------------------------
    def coerce_skill(self, skill):
        if isinstance(skill, str):
            try:
                return self.SkillType(skill.strip().lower())
            except ValueError as exc:
                raise ValueError(
                    f"unknown skill '{skill}'; valid={[c['skill'] for c in _SKILL_CATALOG]}"
                ) from exc
        return skill

    # --- 把 SkillCommand 翻成 env 动作（mirror SkillTestController._command_to_action）----
    def command_to_action(self, command, state):
        p = self.provider
        if command.control_mode == "joint":
            if command.raw_joint_action is not None:
                return p.make_joint_action_from_raw(command.raw_joint_action)
            if command.joint_target is not None:
                return p.make_joint_action_from_q_des(command.joint_target, command.gripper_command)
            return p.make_hold_joint_action(state, command.gripper_command)
        return p.make_hold_joint_action(state, command.gripper_command)

    # --- 运行时 free 抽屉关节（纯物理拉/推；门技能自己 free 铰链）-----------------------
    def prepare_joint_for(self, req) -> None:
        if req.skill_type not in (self.SkillType.OPEN_DRAWER, self.SkillType.CLOSE_DRAWER):
            return
        from runtime.drawer_target_config import joint_name_for
        try:
            joint_name = joint_name_for(req.destination_object or "middle_drawer")
        except Exception as exc:
            print(f"[brain] WARN: cannot resolve drawer joint: {exc}", flush=True)
            return
        self.free_joint("cabinet", joint_name)

    def free_joint(self, asset_name: str, joint_name: str, damping: float = 2.0) -> None:
        if (asset_name, joint_name) in self._freed:
            return
        torch = self.torch
        try:
            asset = self.provider.scene[asset_name]
            ids, _ = asset.find_joints(joint_name)
            n, dev = asset.num_instances, asset.device
            asset.write_joint_stiffness_to_sim(torch.zeros((n, len(ids)), device=dev), joint_ids=ids)
            asset.write_joint_damping_to_sim(
                torch.full((n, len(ids)), float(damping), device=dev), joint_ids=ids
            )
            self._freed.add((asset_name, joint_name))
            print(f"[brain] freed {asset_name}/{joint_name} (runtime only, not saved)", flush=True)
        except Exception as exc:
            print(f"[brain] WARN: could not free {asset_name}/{joint_name}: {exc}", flush=True)


# =====================================================================================
#  1) 感知类
# =====================================================================================
class ScenePerception:
    """读仿真器里的一切（只读、无副作用）。位姿默认世界系，可 frame='base' 转机器人基座系。"""

    def __init__(self, core: _BrainCore):
        self._c = core

    # ---- 全量观测 -------------------------------------------------------------------
    def observe(self):
        """一次拿一帧完整观测（pi05 块 + raw 块 + 两相机 + foundationpose 块 + 把手）。重。"""
        return self._c.session.observe()

    def get_state(self):
        """底层 SceneState（控制/认知类内部用；含 robot + objects 的 tensor 量）。"""
        return self._c.provider.get_state()

    # ---- 相机：两路（front 第三人称静态 / wrist 腕部随手）----------------------------
    def _cam(self, camera: str):
        s = self._c.session
        return s.front_cam if str(camera).startswith("f") else s.wrist_cam

    def get_rgb(self, camera: str = "front"):
        """RGB，HxWx3 uint8（None 表示没挂相机）。camera ∈ {'front','wrist'}。"""
        cam = self._cam(camera)
        return None if cam is None else cam.capture(require_depth=False).get("rgb")

    def get_depth(self, camera: str = "wrist"):
        """深度，HxW float32（米）。需要 enable_fp=True 渲染深度。"""
        cam = self._cam(camera)
        return None if cam is None else cam.capture(require_depth=True).get("depth")

    def get_rgbd(self, camera: str = "wrist"):
        """同视角 (rgb, depth) 一起返回（FoundationPose 用）。"""
        cam = self._cam(camera)
        if cam is None:
            return None, None
        f = cam.capture(require_depth=True)
        return f.get("rgb"), f.get("depth")

    def get_camera_intrinsics(self, camera: str = "front") -> dict | None:
        """{fx,fy,cx,cy,width,height}。"""
        cam = self._cam(camera)
        return None if cam is None else cam.intrinsics()

    def get_camera_pose(self, camera: str = "front") -> dict | None:
        """相机光心世界位姿 {position[3], quat_wxyz[4]}。"""
        cam = self._cam(camera)
        return None if cam is None else cam.camera_pose_world()

    # ---- 物体位姿（cube_1/2/3、knife）-----------------------------------------------
    def get_object_pose(self, name: str, frame: str = "world") -> dict | None:
        """物体根位姿 {position[3], quat_wxyz[4], frame}。frame ∈ {'world','base'}。未知物体->None。"""
        st = self.get_state()
        obj = st.objects.get(name)
        if obj is None:
            return None
        pos = [float(v) for v in obj.pose.pos_w.detach().cpu().tolist()]
        quat = [float(v) for v in obj.pose.quat_w.detach().cpu().tolist()]   # wxyz
        return self._maybe_to_base(pos, quat, frame, name)

    def list_object_poses(self, frame: str = "world") -> dict:
        """所有被跟踪物体的位姿 {name: {position, quat_wxyz, frame}}。"""
        st = self.get_state()
        out = {}
        for name in st.objects:
            obj = st.objects[name]
            pos = [float(v) for v in obj.pose.pos_w.detach().cpu().tolist()]
            quat = [float(v) for v in obj.pose.quat_w.detach().cpu().tolist()]
            out[name] = self._maybe_to_base(pos, quat, frame, name)
        return out

    # ---- 机器人：末端位姿 / 夹爪 / 关节 ---------------------------------------------
    def get_ee_pose(self, frame: str = "world") -> dict:
        """末端 TCP 位姿 {position[3], quat_wxyz[4], frame}。"""
        st = self.get_state()
        pos = [float(v) for v in st.robot.tcp_pose.pos_w.detach().cpu().tolist()]
        quat = [float(v) for v in st.robot.tcp_pose.quat_w.detach().cpu().tolist()]
        return self._maybe_to_base(pos, quat, frame, "ee")

    def get_gripper_width(self) -> float:
        """两指开度（米；两指关节位置之和）。"""
        return float(self.get_state().robot.gripper_width)

    def get_arm_joints(self) -> list:
        """7 维手臂关节弧度（与 q_des 同序）。"""
        st = self.get_state()
        return [float(v) for v in self._c.provider.arm_joint_pos(st).detach().cpu().tolist()]

    def get_robot_joints(self) -> dict:
        """{'pos':[9], 'vel':[9], 'arm':[7], 'gripper_width':float}。"""
        st = self.get_state()
        return {
            "pos": [float(v) for v in st.robot.joint_pos.detach().cpu().tolist()],
            "vel": [float(v) for v in st.robot.joint_vel.detach().cpu().tolist()],
            "arm": self.get_arm_joints(),
            "gripper_width": float(st.robot.gripper_width),
        }

    # ---- 家电关节（抽屉行程 / 门角度 / 咖啡把手）-----------------------------------
    def get_articulation_joint(self, asset: str, joint: str) -> dict | None:
        """读某家电某关节 {asset, joint, pos, vel}（cabinet/microwave(fridge)/coffee_machine）。"""
        try:
            a = self._c.provider.scene[asset]
            ids, _ = a.find_joints(joint)
            jid = ids[0]
            return {
                "asset": asset, "joint": joint,
                "pos": float(a.data.joint_pos[0, jid].detach().cpu()),
                "vel": float(a.data.joint_vel[0, jid].detach().cpu()),
            }
        except Exception:
            return None

    # ---- 把手/拉手/门把手（门技能抓取目标）-----------------------------------------
    def get_handle_poses(self, frame: str = "base") -> dict:
        """门/抽屉/咖啡把手位姿。frame='base'(默认,机器人基座系) 或 'world'。

        返回 {name: {asset, link, calibrated, functional, position[3], quat_wxyz[4]}}，
        name ∈ {microwave_door, top_drawer, middle_drawer, bottom_drawer, coffee_lever}（存在时）。
        """
        raw = self._c.session.handle_poses_in_base()
        if frame == "base":
            return raw
        out = {}
        for name, h in raw.items():
            d = dict(h)
            d["position"] = h.get("position_world", h.get("position"))
            d["quat_wxyz"] = h.get("quat_wxyz_world", h.get("quat_wxyz"))
            out[name] = d
        return out

    # ---- FoundationPose：输入打包 + 估计位姿（可注入真实 FP）------------------------
    def get_foundationpose_input(self, which: str = "wrist", dump_dir: str | None = None) -> dict | None:
        """把同视角 RGBD + 内外参 + 物体清单打包成 FoundationPose 进程的输入 bundle。"""
        obs = self._c.session.observe()
        if obs.foundationpose is None:
            return None
        return obs.foundationpose.fp_input(which=which, dump_dir=dump_dir)

    def set_pose_estimator(self, fn) -> None:
        """注入真实的 FoundationPose 估计器：fn(name, fp_input) -> {position, quat_wxyz} 或 4x4。

        不注入时 get_estimated_pose 退化为仿真 GT（pose_source='sim_gt'）。
        """
        self._estimator = fn

    def get_estimated_pose(self, name: str, which: str = "wrist") -> dict | None:
        """物体的感知估计位姿。注入了 estimator 用 FoundationPose；否则返回仿真 GT。"""
        fn = getattr(self, "_estimator", None)
        if fn is not None:
            bundle = self.get_foundationpose_input(which=which)
            est = fn(name, bundle)
            if isinstance(est, dict):
                est.setdefault("pose_source", "foundationpose")
                return est
            return {"pose_4x4": est, "pose_source": "foundationpose"}
        gt = self.get_object_pose(name, frame="world")
        if gt is not None:
            gt["pose_source"] = "sim_gt"
        return gt

    # ---- 内部：世界系 -> 机器人基座系 ----------------------------------------------
    def _maybe_to_base(self, pos, quat, frame, _name) -> dict:
        if frame != "base":
            return {"position": pos, "quat_wxyz": quat, "frame": "world"}
        robot = self._c.env.unwrapped.scene["robot"]
        bp = [float(v) for v in robot.data.root_pos_w[0].detach().cpu().tolist()]
        bq = [float(v) for v in robot.data.root_quat_w[0].detach().cpu().tolist()]  # wxyz
        pos_b, quat_b = _world_to_base(pos, quat, bp, bq)
        return {"position": pos_b, "quat_wxyz": quat_b, "frame": "base"}


# =====================================================================================
#  2) 控制类（大脑 -> 状态机）
# =====================================================================================
class SkillControl:
    """给状态机下指令：选技能 / 选目标物体 / 选目标位置 / 夹爪 / 暂停 / 继续 / 停止 / 中止。

    两种下指令方式：
      - 一步到位：command_skill(skill, target=..., target_pose=..., params=...)
      - 分步选择：select_skill(...); set_target_object(...); set_target_pose(...); start()
    指令只是「入队」，真正的 executor.start 在下一次 tick()/step() 里用当帧 SceneState 执行。
    """

    def __init__(self, core: _BrainCore):
        self._c = core
        self._sel_skill = None
        self._sel_target = None
        self._sel_pose = None
        self._sel_params: dict = {}

    # ---- 分步选择 -------------------------------------------------------------------
    def select_skill(self, skill) -> None:
        """选技能：'grasp'|'place'|'open_drawer'|'close_drawer'|'open_door'|'close_door' 或 SkillType。"""
        self._sel_skill = self._c.coerce_skill(skill)

    def set_target_object(self, name: str) -> None:
        """选目标物体/抽屉/门：如 'cube_1' / 'middle_drawer' / 'microwave'。"""
        self._sel_target = name

    def set_target_pose(self, xyz) -> None:
        """选目标位置（仅 place 用，env-local [x,y,z]）。其他技能位姿由场景几何自动求解。"""
        self._sel_pose = tuple(float(v) for v in xyz)

    def set_params(self, **params) -> None:
        """设置技能附加参数（如 drawer_link）。"""
        self._sel_params.update(params)

    def start(self) -> str:
        """用已选 skill/target/pose/params 入队一条指令，返回 request_id。"""
        if self._sel_skill is None:
            raise ValueError("no skill selected; call select_skill(...) first")
        rid = self.command_skill(
            self._sel_skill, target=self._sel_target,
            target_pose=self._sel_pose, params=self._sel_params or None,
        )
        self._sel_pose = None
        self._sel_params = {}
        return rid

    # ---- 一步到位 -------------------------------------------------------------------
    def command_skill(self, skill, target: str | None = None,
                      target_pose=None, params: dict | None = None) -> str:
        """下一条技能指令（入队）。返回 request_id。下一帧 tick() 真正启动。"""
        st = self._c.coerce_skill(skill)
        req = self._build_request(st, target, target_pose, params)
        self._c._pending = req
        self._c._want_stop = False
        return req.request_id

    # ---- 运行时控制：暂停 / 继续 / 停止 / 中止 -------------------------------------
    def pause(self) -> None:
        """暂停当前技能（冻结在当前姿态；之后 tick 发 hold，不推进技能）。"""
        self._c.executor.pause(self._c.provider.get_state())

    def resume(self) -> bool:
        """继续被暂停的技能。返回是否成功 resume。"""
        return bool(self._c.executor.resume(self._c.provider.get_state()))

    def stop(self) -> None:
        """停止当前技能（= pause，软停，保留 held_object）。"""
        self._c.executor.stop(self._c.provider.get_state())

    def abort(self) -> None:
        """硬中止：丢弃当前技能并清空状态机（不保留 held 上下文），回到 idle。"""
        self._c.executor.reset()
        self._c._pending = None
        self._c._want_stop = True

    def is_busy(self) -> bool:
        """是否有技能在跑或在待启动。"""
        ex = self._c.executor
        return ex.active_skill is not None or self._c._pending is not None

    # ---- 低层透传：直接发绝对关节（无技能时由大脑自己伺服）-------------------------
    def command_joints(self, q_des7, gripper=1) -> None:
        """入队一次性手动关节动作（7 维绝对弧度 + 夹爪 0/1）。仅当无技能在跑时生效。"""
        self._c._manual = (q_des7, gripper)

    # ---- 每帧：算出当前应发的 env 动作（tick/step 调用；返回 None=由调用方 hold）----
    def compute_action(self, state):
        c = self._c
        if c._want_stop:
            c.executor.reset()
            c._want_stop = False
            return None
        if c._pending is not None:
            req = c._pending
            c._pending = None
            c.prepare_joint_for(req)        # free 目标抽屉关节（门技能自己 free）
            c.executor.start(req, state)
        if c.executor.active_skill is None and c.executor.held_object is None:
            if c._manual is not None:
                q7, g = c._manual
                c._manual = None
                return c.provider.make_joint_action_from_q_des(q7, pi05_gripper_to_env_cmd(g))
            return None
        command = c.executor.step(state, c.dt)
        return c.command_to_action(command, state)

    # ---- 内部：构造 SkillRequest（generalize SkillTestController._make_request）-----
    def _build_request(self, skill_type, target, target_pose, params):
        import time
        c = self._c
        ST, Req = c.SkillType, c.SkillRequest
        stamp = time.time_ns()
        p = dict(params or {})
        if skill_type == ST.PLACE:
            xyz = list(target_pose) if target_pose is not None else list(_DEFAULT_PLACE_XYZ)
            p.setdefault("target_frame", "env_local")
            p["target_surface_xyz"] = xyz
            return Req(request_id=f"{skill_type.value}_{stamp}", skill_type=skill_type,
                       source_object=None, destination_type="point",
                       destination_object=target or "point_a", parameters=p)
        if skill_type in (ST.OPEN_DRAWER, ST.CLOSE_DRAWER):
            p.setdefault("drawer_link", "link_1")
            return Req(request_id=f"{skill_type.value}_{target or 'middle_drawer'}_{stamp}",
                       skill_type=skill_type, source_object=None, destination_type="drawer",
                       destination_object=target or "middle_drawer", parameters=p)
        if skill_type in (ST.OPEN_DOOR, ST.CLOSE_DOOR):
            return Req(request_id=f"{skill_type.value}_{target or 'microwave'}_{stamp}",
                       skill_type=skill_type, source_object=None, destination_type="door",
                       destination_object=target or "microwave", parameters=p or {})
        # GRASP
        return Req(request_id=f"{skill_type.value}_{target or 'cube_1'}_{stamp}",
                   skill_type=skill_type, source_object=target or "cube_1", parameters=p or {})


# =====================================================================================
#  3) 认知类（能力发现 + 前置校验 + 执行反馈 + 回合 + 可视化）—— 你没考虑到的那层
# =====================================================================================
class TaskCognition:
    """大脑闭环真正需要、却既非「读传感器」也非「下技能指令」的一层。"""

    def __init__(self, core: _BrainCore):
        self._c = core

    # ---- 能力发现：大脑用来知道「我能做什么、对谁做、要什么参数」----------------
    def list_skills(self) -> list:
        """[{skill, target_kind, needs_target, params, desc, success}, ...]。"""
        return [dict(c) for c in _SKILL_CATALOG]

    def list_objects(self) -> dict:
        """{'graspable':[...],'drawer':[...],'door':[...],'place_point':[...]}。"""
        return {k: list(v) for k, v in _TARGETS_BY_KIND.items()}

    def list_targets(self, skill) -> list:
        """某技能的合法目标取值域。"""
        st = self._c.coerce_skill(skill)
        kind = _CATALOG_BY_SKILL[st.value]["target_kind"]
        return list(_TARGETS_BY_KIND.get(kind, ()))

    def describe_skill(self, skill) -> dict:
        """某技能的说明 + 参数 + 成功判据。"""
        st = self._c.coerce_skill(skill)
        return dict(_CATALOG_BY_SKILL[st.value])

    def can_execute(self, skill, target: str | None = None) -> dict:
        """前置校验：这条命令现在能不能执行。返回 {ok, reason}。"""
        try:
            st = self._c.coerce_skill(skill)
        except ValueError as exc:
            return {"ok": False, "reason": str(exc)}
        cat = _CATALOG_BY_SKILL[st.value]
        holding = self.is_holding()
        if st.value == "place":
            if holding is None:
                return {"ok": False, "reason": "place needs a held object; nothing is grasped"}
            return {"ok": True, "reason": "ok"}
        if cat["needs_target"]:
            valid = _TARGETS_BY_KIND[cat["target_kind"]]
            if target is None:
                return {"ok": False, "reason": f"{st.value} needs a target in {list(valid)}"}
            if target not in valid:
                return {"ok": False, "reason": f"unknown target '{target}'; valid={list(valid)}"}
        if st.value == "grasp" and holding is not None:
            return {"ok": False, "reason": f"already holding '{holding}'; place/abort first"}
        return {"ok": True, "reason": "ok"}

    # ---- 执行反馈：闭环必需 --------------------------------------------------------
    def skill_status(self) -> dict:
        """实时状态 {active, phase, runtime_status, held_object, busy}。"""
        ex = self._c.executor
        return {
            "active": ex.active_skill is not None,
            "phase": ex.current_state_name,
            "runtime_status": ex.runtime_status,
            "held_object": getattr(ex.held_object, "object_name", None),
            "busy": ex.active_skill is not None or self._c._pending is not None,
        }

    def is_holding(self) -> str | None:
        """当前夹着的物体名（没有则 None）。"""
        return getattr(self._c.executor.held_object, "object_name", None)

    def last_result(self) -> dict | None:
        """上一条技能的结果（成功/失败/原因/误差/耗时）。None=还没跑过。"""
        r = self._c.executor.last_result
        if r is None:
            return None
        return {
            "request_id": getattr(r, "request_id", None),
            "skill": getattr(getattr(r, "skill_type", None), "value", None),
            "target": getattr(r, "target_name", None),
            "success": bool(getattr(r, "success", False)),
            "status": getattr(getattr(r, "final_status", None), "value", None),
            "failure_reason": getattr(r, "failure_reason", None),
            "elapsed": getattr(r, "elapsed_time", None),
            "position_error": getattr(r, "position_error", None),
            "orientation_error": getattr(r, "orientation_error", None),
            "gripper_width": getattr(r, "gripper_width", None),
        }

    # ---- 回合管理 -----------------------------------------------------------------
    def reset(self, reset_mode: str | None = None, seed: int | None = None):
        """重置场景（按 reset_mode 随机布局）+ 清空状态机。返回 reset 后的 Observation。"""
        s = self._c.session
        if reset_mode is not None:
            from .config import ResetMode
            from .reset_strategies import build_reset_strategy
            sd = s.config.seed if seed is None else seed
            s._reset_strategy = build_reset_strategy(ResetMode(reset_mode), sd)
        self._c.executor.reset()
        self._c._pending = None
        self._c._want_stop = False
        return s.reset(seed=seed)

    def settle(self, steps: int = 8):
        """保持当前姿态空跑若干步让物理/渲染稳定。"""
        self._c.session._settle(int(steps))

    def save_scene(self, note: str | None = None) -> dict:
        """把当前布局保存为「最新场景」（scene_v1_latest），机器人姿态不存。返回保存信息/错误。"""
        try:
            from .scene_saver import save_current_stage_as_latest
            return save_current_stage_as_latest(
                env=self._c.session.env, note=note or "Saved by BrainInterface."
            )
        except Exception as exc:
            print(f"[brain] WARN: save_scene failed: {exc}", flush=True)
            return {"error": str(exc)}

    # ---- 可视化 / 调试开关 --------------------------------------------------------
    def show_targets(self, poses, names=None, use_arrows: bool = True, axis_length: float = 0.08) -> None:
        """画一组目标位姿箭头（headless 静默）。poses: PoseState/(pos,quat)/7元/Nx7。"""
        self._c.session.show_target_poses(poses, names=names, use_arrows=use_arrows, axis_length=axis_length)

    def clear_targets(self) -> None:
        self._c.session.clear_target_poses()

    def set_collision_visible(self, on: bool) -> None:
        """开/关碰撞体可视化。"""
        self._c.session.set_collision_visible(on)

    def set_markers_visible(self, on: bool) -> None:
        """开/关黄色 InitCorner 区域标记块。"""
        self._c.session.set_markers_visible(on)


# =====================================================================================
#  聚合入口
# =====================================================================================
class BrainInterface:
    """大脑的统一句柄。持有 .perception / .control / .cognition + 共享 session/executor。

    也直接实现了 test_mode 的 controller hook（__init__(session) / build_window / on_reset /
    step(session)->action），所以可被 ``--controller ...:BrainInterface`` 直接加载。
    """

    def __init__(self, session: SceneSession, *, owns_session: bool = False):
        self.session = session
        self._core = _BrainCore(session)
        self.perception = ScenePerception(self._core)
        self.control = SkillControl(self._core)
        self.cognition = TaskCognition(self._core)
        self._owns_session = owns_session

    # ---- 创建 ----------------------------------------------------------------------
    @classmethod
    def launch(cls, config: SceneConfig) -> "BrainInterface":
        """自建并启动场景（自管理 AppLauncher），返回 brain。库/脚本用法。"""
        session = SceneSession.launch(config)
        return cls(session, owns_session=True)

    @classmethod
    def attach(cls, session: SceneSession) -> "BrainInterface":
        """复用已有 session（如 test_mode 已起的场景）。不接管 session 生命周期。"""
        return cls(session, owns_session=False)

    # ---- controller hook（test_mode 每帧调用）------------------------------------
    def build_window(self) -> None:    # noqa: D401 - optional UI; brain is headless by default
        """test_mode 期望的可选 UI 钩子；大脑无 UI，留空。"""

    def on_reset(self) -> None:
        """test_mode 布局 reset 后调用：丢弃当前技能。"""
        self._core.executor.reset()
        self._core._pending = None
        self._core._want_stop = False

    def step(self, session: SceneSession):
        """test_mode 每帧调用：返回一帧 env 动作或 None（None=交回 UI/hold）。"""
        return self.control.compute_action(self._core.provider.get_state())

    # ---- 自管理循环（库/脚本用法）------------------------------------------------
    def tick(self) -> dict:
        """推进一个仿真步：算动作 -> env.step（无动作则 hold）。返回 skill_status 快照。"""
        state = self._core.provider.get_state()
        action = self.control.compute_action(state)
        if action is None:
            self._core.session.hold()
        else:
            self._core.env.step(action)
            self._core.session._advance_clock()
        return self.cognition.skill_status()

    def run_skill(self, skill, target: str | None = None, target_pose=None,
                  params: dict | None = None, max_steps: int = 3000, settle: int = 20) -> dict | None:
        """阻塞跑完一个技能：下指令 -> 循环 tick 到技能结束 -> settle。返回 last_result()。

        大脑最常用的「下一条指令并等结果」。需要边跑边感知/可打断时，改用 command_skill + 自己 tick。
        """
        self.control.command_skill(skill, target=target, target_pose=target_pose, params=params)
        for _ in range(int(max_steps)):
            self.tick()
            if self._core.executor.active_skill is None and self._core._pending is None:
                break
        for _ in range(int(settle)):
            self._core.session.hold()
        return self.cognition.last_result()

    # ---- 部署对接：观测 dict + 命令 dict + policy 驱动 -----------------------------
    def observation_dict(self, frame: str = "base", include_images: bool = True) -> dict:
        """给大脑的一帧观测（JSON 友好；include_images 时图像是 numpy 数组，进程内零拷贝）。

        大脑只需读这一个 dict 即可决策；字段格式见 BRAIN_INTEGRATION_SPEC.md。
        """
        p = self.perception
        d = {
            "sim_time": float(getattr(self._core.session, "_clock", 0.0)),
            "ee_pose": p.get_ee_pose(frame),
            "gripper_width": p.get_gripper_width(),
            "arm_joints": p.get_arm_joints(),
            "objects": p.list_object_poses(frame),
            "handles": p.get_handle_poses("base" if frame == "base" else "world"),
            "skill_status": self.cognition.skill_status(),
            "is_holding": self.cognition.is_holding(),
        }
        if include_images:
            d["rgb_front"] = p.get_rgb("front")
            d["rgb_wrist"] = p.get_rgb("wrist")
            d["depth_wrist"] = p.get_depth("wrist")
        return d

    def apply_command(self, cmd: dict | None) -> dict:
        """把大脑输出的一条命令 dict 分发到接口。返回 {ok, ...}。命令 schema 见 SPEC §8。

        op: command|pause|resume|stop|abort|reset|joints|noop|halt
        """
        if not cmd:
            return {"ok": True, "op": "noop"}
        op = cmd.get("op", "command")
        try:
            if op in ("noop", "halt"):
                return {"ok": True, "op": op}
            if op == "command":
                chk = self.cognition.can_execute(cmd.get("skill"), cmd.get("target"))
                if not chk["ok"]:
                    return {"ok": False, "op": op, "reason": chk["reason"]}
                rid = self.control.command_skill(
                    cmd.get("skill"), target=cmd.get("target"),
                    target_pose=cmd.get("target_pose"), params=cmd.get("params"),
                )
                return {"ok": True, "op": op, "request_id": rid}
            if op == "pause":
                self.control.pause(); return {"ok": True, "op": op}
            if op == "resume":
                return {"ok": bool(self.control.resume()), "op": op}
            if op == "stop":
                self.control.stop(); return {"ok": True, "op": op}
            if op == "abort":
                self.control.abort(); return {"ok": True, "op": op}
            if op == "reset":
                self.reset(reset_mode=cmd.get("reset_mode"), seed=cmd.get("seed"))
                return {"ok": True, "op": op}
            if op == "joints":
                self.control.command_joints(cmd["q_des"], cmd.get("gripper", 1))
                return {"ok": True, "op": op}
            return {"ok": False, "op": op, "reason": f"unknown op '{op}'"}
        except Exception as exc:
            return {"ok": False, "op": op, "reason": str(exc)}

    def run_policy(self, policy, *, max_steps: int = 100000, obs_frame: str = "base",
                   include_images: bool = True) -> dict:
        """部署入口：循环 [build obs -> policy.act(obs) -> apply_command -> tick]，直到 policy 发 halt。

        policy 只需实现 ``act(obs: dict) -> cmd_dict | None``（None=本帧不下新命令，继续推进）。
        返回 {steps, last_result}。
        """
        steps = 0
        for _ in range(int(max_steps)):
            obs = self.observation_dict(frame=obs_frame, include_images=include_images)
            cmd = policy.act(obs)
            ack = self.apply_command(cmd)
            if cmd and cmd.get("op") == "halt":
                break
            self.tick()
            steps += 1
            obs = None  # noqa: F841 - 让大数组尽早可回收
            _ = ack
        return {"steps": steps, "last_result": self.cognition.last_result()}

    # ---- 直通便捷 ------------------------------------------------------------------
    def reset(self, reset_mode: str | None = None, seed: int | None = None):
        return self.cognition.reset(reset_mode=reset_mode, seed=seed)

    def observe(self):
        return self.session.observe()

    def close(self) -> None:
        """关闭（仅当 owns_session 时真正关 session/AppLauncher）。"""
        if self._owns_session:
            self.session.close()

    def __enter__(self) -> "BrainInterface":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# =====================================================================================
#  小工具：世界系 -> 机器人基座系（wxyz 四元数）
# =====================================================================================
def _q_conj(q):
    return (q[0], -q[1], -q[2], -q[3])


def _q_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _q_rot(q, v):
    """用单位四元数 q(wxyz) 旋转向量 v(3)。"""
    qv = (0.0, v[0], v[1], v[2])
    r = _q_mul(_q_mul(q, qv), _q_conj(q))
    return [r[1], r[2], r[3]]


def _world_to_base(pos_w, quat_w, base_pos, base_quat):
    """把世界系位姿 (pos_w, quat_w wxyz) 转到 base 位姿 (base_pos, base_quat wxyz) 定义的基座系。"""
    base_inv = _q_conj(base_quat)   # 单位四元数逆 = 共轭
    d = [pos_w[i] - base_pos[i] for i in range(3)]
    pos_b = _q_rot(base_inv, d)
    quat_b = _q_mul(base_inv, quat_w)
    return pos_b, list(quat_b)
