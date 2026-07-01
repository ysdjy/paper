# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""SceneSession —— V1 场景的进程内 facade（无 ZMQ/ROS，像正常启动 isaacsim 一样）。

其他模块（pi0.5 采集/训练/推理、FoundationPose、技能状态机）通过它：
    launch -> reset -> observe -> step -> close
启动底层 Isaac 场景、读三块观测、发**绝对关节 + 夹爪**指令。

isaac / torch / build_teleop_env 全部延迟到 launch() 方法体内 import，所以
``from ...scene_interface import SceneSession`` 本身在 plain venv 也不炸，且满足
“AppLauncher 必须在 import isaac 之前启动” 的约束。
"""

from __future__ import annotations

from .config import ResetMode, SceneConfig
from .handles import build_handle_specs, read_handles_in_base
from .observation import (
    CameraView,
    FoundationPoseBlock,
    Observation,
    Pi05Block,
    RawBlock,
)
from .reset_strategies import build_reset_strategy

# 进程内只允许一个“自管理 AppLauncher”的 session（SimulationApp 是进程级单例）。
_OWNED_APP_STARTED = False


def pi05_gripper_to_env_cmd(g) -> float:
    """夹爪约定换算（唯一一处）：LeRobot/pi0.5 0=open / 1=close -> env +1=open / -1=close。

    也接受字符串 "open"/"close"。
    """
    if isinstance(g, str):
        return 1.0 if g.strip().lower().startswith("o") else -1.0
    return 1.0 if float(g) < 0.5 else -1.0


class SceneSession:
    """V1 场景的统一接口。用 ``SceneSession.launch(cfg)`` 创建。"""

    # ------------------------------------------------------------------ launch
    @classmethod
    def launch(cls, config: SceneConfig, *, _app_launcher=None) -> "SceneSession":
        """启动场景。``_app_launcher`` 给定时复用调用方已建好的 AppLauncher（entry 用）。"""
        global _OWNED_APP_STARTED

        owns_app = _app_launcher is None
        if owns_app:
            if _OWNED_APP_STARTED:
                raise RuntimeError(
                    "本进程已存在一个自管理的 SceneSession（SimulationApp 进程级单例）。"
                    "一个进程只能启动一个；多任务请用 num_envs>1 或多进程。"
                )
            app_launcher = _make_app_launcher(config)
            _OWNED_APP_STARTED = True
        else:
            app_launcher = _app_launcher

        try:
            self = cls._build(config, app_launcher, owns_app)
        except Exception:
            if owns_app:
                _OWNED_APP_STARTED = False
                try:
                    app_launcher.app.close()
                except Exception:
                    pass
            raise
        return self

    @classmethod
    def _build(cls, config, app_launcher, owns_app) -> "SceneSession":
        # --- AppLauncher 之后才 import isaac 相关 ---
        from franka_v1_skill_lab.teleop_collection.recording.isaac_teleop_driver import (
            build_teleop_env,
            ee_pose_xyzw,
        )
        from franka_v1_skill_lab.skill_runtime._legacy import ensure_legacy_on_path

        ensure_legacy_on_path()
        from runtime.scene_state_provider import SceneStateProvider  # noqa: E402
        from franka_v1_skill_lab.sensors.d435 import d435_config as dcfg
        from franka_v1_skill_lab.sensors.d435.d435_observation_adapter import WristCameraAdapter

        want_cam = config.wants_cameras()
        decimation = max(1, round(100.0 / max(1.0, config.control_hz)))
        cam_res = tuple(config.camera_resolution)

        # 建 env 前的 cfg 加工：可选覆盖最新保存场景的物体位姿，并（按需）给悬空微波炉加台子。
        cfg_hook = None
        if (config.load_latest_scene or config.add_microwave_stand
                or config.spawn_init_markers or config.refine_handle_collisions
                or config.replace_microwave_with_fridge or config.lock_knife
                or config.enable_collision_monitor or config.extra_assets
                or config.add_robot_stand or config.exclude_members
                or config.disable_auto_reset or config.hidden_members):
            def cfg_hook(env_cfg, _cfg=config):
                # 删除被排除的成员(置 None 不 spawn)——必须最先做，后续 replace_fridge/add_*_stand/
                # handle_refine 都会跳过 None 成员。同时把【挂在该家电 prim 下的子成员】(如门把手代理
                # microwave_handle_proxy)也置 None，否则它们会试图 spawn 到已不存在的父 prim 上而报错。
                _scene = getattr(env_cfg, "scene", None)
                for _m in (_cfg.exclude_members or ()):
                    _member = getattr(_scene, _m, None) if _scene is not None else None
                    if _member is None:
                        continue
                    _pp = getattr(_member, "prim_path", "") or ""
                    setattr(_scene, _m, None)
                    print(f"[scene_session] excluded scene member '{_m}' (not spawned).", flush=True)
                    if _pp:   # 清掉挂在该家电 prim 下的子成员(把手代理等)
                        _child = _pp.rstrip("/") + "/"
                        for _attr, _mm in list(vars(_scene).items()):
                            _mpp = getattr(_mm, "prim_path", None)
                            if _attr != _m and isinstance(_mpp, str) and _child in _mpp:
                                setattr(_scene, _attr, None)
                                print(f"[scene_session]   also removed child member '{_attr}' (under {_m}).", flush=True)
                # extra_assets 必须在 apply_latest_scene 之前 spawn：这样成员已存在，最新 manifest 里
                # 保存的水果 pose/scale 才能被 apply_latest_scene_to_cfg 覆盖回来(持久化复现)。
                if _cfg.extra_assets:
                    try:
                        from .scene_props import add_usd_asset

                        for a in _cfg.extra_assets:
                            add_usd_asset(env_cfg, **dict(a))
                    except Exception as exc:  # pragma: no cover - defensive
                        print(f"[scene_session] WARNING: add extra assets failed: {exc}", flush=True)
                # 先恢复最新场景(含机器人 XY/yaw)，再放底座 -> 底座跟到机器人的【保存】位置。
                _mw_pos_saved = False   # 保存场景里是否给了微波炉/冰箱成员一个位置(决定冰箱替换是否落地)
                if _cfg.load_latest_scene:
                    try:
                        from .scene_sync import apply_latest_camera_to_cfg, apply_latest_scene_to_cfg

                        _summary = apply_latest_scene_to_cfg(env_cfg, registry_path=_cfg.scene_registry_path)
                        _mw_pos_saved = any(
                            a.get("cfg") == "microwave" and "pos" in a.get("changed", [])
                            for a in (_summary or {}).get("applied", [])
                        )
                        if _cfg.apply_saved_camera_offsets:
                            # 在 spawn 前写相机 offset（一致），取代运行时挪相机(desync)。
                            apply_latest_camera_to_cfg(env_cfg, registry_path=_cfg.scene_registry_path)
                    except Exception as exc:  # pragma: no cover - defensive
                        print(f"[scene_session] WARNING: apply latest scene failed, base cfg used: {exc}", flush=True)
                # 抬高机器人底座初始 Z：原来太低，运动时容易碰到桌面/物体。env 可调，默认 +10cm；0 关闭。
                try:
                    import os as _os
                    _lift = float(_os.environ.get("V1_ROBOT_Z_LIFT", "0.10"))
                    _robot = getattr(getattr(env_cfg, "scene", None), "robot", None)
                    _ist = getattr(_robot, "init_state", None) if _robot is not None else None
                    if _ist is not None and abs(_lift) > 1e-6:
                        _p = tuple(float(v) for v in _ist.pos)
                        _ist.pos = (_p[0], _p[1], _p[2] + _lift)
                        print(f"[scene_session] robot init Z lifted +{_lift:.2f} -> {_ist.pos}", flush=True)
                except Exception as exc:  # pragma: no cover
                    print(f"[scene_session] robot Z lift failed: {exc}", flush=True)
                # 机器人 home：用 Franka 控制面板「Set Home」存的关节配置(robot_home.json)。
                # reset 事件 set_default_joint_pose 决定每次 reset 后手臂停在哪，默认是旧的硬编码 home。
                # 这里把它(及机器人 init_state 手臂关节)改成保存的 home，并关掉 reset 时的高斯抖动，
                # 这样【任何入口】(含不带 --franka)reset 后都停在用户保存的 home，不再回到旧 home。
                try:
                    from .robot_home import saved_home_q
                    _home = saved_home_q()
                    if _home is not None:
                        _events = getattr(env_cfg, "events", None)
                        _evt = getattr(_events, "init_franka_arm_pose", None) if _events is not None else None
                        _params = getattr(_evt, "params", None) if _evt is not None else None
                        if isinstance(_params, dict) and "default_pose" in _params:
                            _old = list(_params["default_pose"])
                            _grip = _old[7:] if len(_old) > 7 else [0.04, 0.04]
                            _params["default_pose"] = [float(v) for v in _home] + [float(v) for v in _grip]
                            print(f"[scene_session] reset home <- saved robot_home.json {tuple(round(v,4) for v in _home)}", flush=True)
                        # 关掉 reset 抖动，保证停在精确 home（否则每次 reset ±0.02rad 噪声）
                        if _events is not None and hasattr(_events, "randomize_franka_joint_state"):
                            _events.randomize_franka_joint_state = None
                            print("[scene_session] disabled randomize_franka_joint_state (exact home).", flush=True)
                        # 机器人 init_state 手臂关节也设成 home（spawn 时即对，避免第一帧跳）
                        _robot2 = getattr(getattr(env_cfg, "scene", None), "robot", None)
                        _ist2 = getattr(_robot2, "init_state", None) if _robot2 is not None else None
                        _jp = getattr(_ist2, "joint_pos", None) if _ist2 is not None else None
                        if isinstance(_jp, dict):
                            for _i in range(7):
                                _k = f"panda_joint{_i+1}"
                                if _k in _jp:
                                    _jp[_k] = float(_home[_i])
                except Exception as exc:  # pragma: no cover
                    print(f"[scene_session] apply saved home failed: {exc}", flush=True)
                # 静态场景：关掉物体随机化 + episode 超时 + 会触发自动 reset 的 termination（cube 掉落/堆叠成功）。
                # 让交互/采集场景保持静止，物体不再每次 reset 随机跳位。
                if _cfg.disable_auto_reset:
                    _events = getattr(env_cfg, "events", None)
                    if _events is not None and hasattr(_events, "randomize_cube_positions"):
                        _events.randomize_cube_positions = None
                        print("[scene_session] disabled randomize_cube_positions event.", flush=True)
                    if hasattr(env_cfg, "episode_length_s"):
                        env_cfg.episode_length_s = 1.0e9
                    _terms = getattr(env_cfg, "terminations", None)
                    if _terms is not None:
                        for _t in ("cube_1_dropping", "cube_2_dropping", "cube_3_dropping",
                                   "success", "cubes_stacked", "time_out"):
                            if hasattr(_terms, _t):
                                setattr(_terms, _t, None)
                        print("[scene_session] disabled auto-reset terminations.", flush=True)
                # 隐藏 contract 成员（cube_1/2/3）：不能置 None（obs/term 引用会崩），改把 init_state 挪到
                # 场景外远处落地，相机看不到。必须在 apply_latest 之后覆盖，否则被 manifest 位姿盖回。
                if _cfg.hidden_members:
                    _scene2 = getattr(env_cfg, "scene", None)
                    for _i, _hm in enumerate(_cfg.hidden_members):
                        _mem = getattr(_scene2, _hm, None) if _scene2 is not None else None
                        _ist = getattr(_mem, "init_state", None) if _mem is not None else None
                        if _ist is not None:
                            _ist.pos = (100.0, 100.0 + 0.3 * _i, 0.05)
                            print(f"[scene_session] hid member '{_hm}' off-scene at {_ist.pos}.", flush=True)
                if _cfg.add_robot_stand:
                    try:
                        from .scene_props import add_robot_stand

                        add_robot_stand(env_cfg, _cfg.robot_stand_usd)
                    except Exception as exc:  # pragma: no cover - defensive
                        print(f"[scene_session] WARNING: add robot stand failed: {exc}", flush=True)
                # 替换微波炉->冰箱：必须在 apply_latest_scene 之后(否则 scale 被 manifest 覆盖)、
                # install_handle_refine 之前(refine 按 usd_path 含 "fridge" 匹配 link_1)。
                # ground_z=只有【没存过位置】时才把冰箱落地；存过(如用户把冰箱放柜子上)就保留保存的 z。
                if _cfg.replace_microwave_with_fridge:
                    try:
                        from .scene_props import replace_microwave_with_fridge

                        replace_microwave_with_fridge(
                            env_cfg, _cfg.fridge_usd, _cfg.fridge_scale, ground_z=not _mw_pos_saved)
                    except Exception as exc:  # pragma: no cover - defensive
                        print(f"[scene_session] WARNING: replace microwave with fridge failed: {exc}", flush=True)
                if _cfg.lock_knife:
                    try:
                        from .scene_props import lock_knife

                        lock_knife(env_cfg)
                    except Exception as exc:  # pragma: no cover - defensive
                        print(f"[scene_session] WARNING: lock knife failed: {exc}", flush=True)
                if _cfg.enable_collision_monitor:
                    try:
                        from .scene_props import install_collision_monitor

                        install_collision_monitor(env_cfg)
                    except Exception as exc:  # pragma: no cover - defensive
                        print(f"[scene_session] WARNING: install collision monitor failed: {exc}", flush=True)
                # 台子：替换冰箱时跳过（台子按微波炉 bbox 算，对冰箱错；冰箱落地不用台子）
                if _cfg.add_microwave_stand and not _cfg.replace_microwave_with_fridge:
                    try:
                        from .scene_props import add_microwave_stand

                        add_microwave_stand(env_cfg)   # 自动判断：微波炉悬空才加
                    except Exception as exc:  # pragma: no cover - defensive
                        print(f"[scene_session] WARNING: add microwave stand failed: {exc}", flush=True)
                try:
                    from .scene_props import add_dishwasher_stand

                    add_dishwasher_stand(env_cfg)   # 填补洗碗机离地间隙的跟随台子(有洗碗机才加)
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"[scene_session] WARNING: add dishwasher stand failed: {exc}", flush=True)
                if _cfg.spawn_init_markers:
                    try:
                        from .scene_props import spawn_init_markers

                        spawn_init_markers(env_cfg)
                    except Exception as exc:  # pragma: no cover - defensive
                        print(f"[scene_session] WARNING: spawn init markers failed: {exc}", flush=True)
                if _cfg.refine_handle_collisions:
                    try:
                        from .scene_props import install_handle_refine

                        install_handle_refine(env_cfg)   # 换 spawn.func，spawn 时细化把手
                    except Exception as exc:  # pragma: no cover - defensive
                        print(f"[scene_session] WARNING: install handle refine failed: {exc}", flush=True)

        env, env_cfg, cam_attached = build_teleop_env(
            task_id=config.task_id,
            num_envs=config.num_envs,
            device=config.device,
            use_fabric=config.use_fabric,
            enable_wrist_d435=want_cam,
            enable_depth=False,  # 旧独立 FP 相机不出深度（FP 改从 front/wrist 同视角读）
            free_microwave_door=config.free_microwave_door,
            seed=config.seed,
            decimation=decimation,
            arm_stiffness=config.arm_stiffness,
            arm_damping=config.arm_damping,
            # 两相机模型：front+wrist 都出 depth、都用 640x480，保证 VLA-RGB 与 FP-RGBD 同视角
            enable_vla_depth=config.enable_fp,
            enable_front_depth=config.enable_fp,
            enable_left_depth=True,   # 左相机始终出 RGB+深度（用户要求）
            enable_right_depth=True,  # 右相机始终出 RGB+深度（用户要求）
            enable_top_depth=True,    # 俯视相机始终出 RGB+深度（用户要求）
            vla_resolution=cam_res,
            front_resolution=cam_res,
            left_resolution=cam_res,
            right_resolution=cam_res,
            top_resolution=cam_res,
            cfg_hook=cfg_hook,
        )
        if want_cam and not cam_attached:
            env.close()
            raise RuntimeError(
                "请求了相机但未挂载：需要用 --enable_cameras 启动 + GPU。"
                "（库用法下 launch 会自动设 enable_cameras；请确认在 GPU 机器上运行。）"
            )

        self = cls.__new__(cls)
        self.config = config
        self._app_launcher = app_launcher
        self._owns_app = owns_app
        self.env = env
        self.env_cfg = env_cfg
        self.provider = SceneStateProvider(env)
        self._ee_pose_xyzw = ee_pose_xyzw
        self._dcfg = dcfg
        self.front_cam = WristCameraAdapter(env, camera_name=dcfg.FRONT_CAM_NAME) if want_cam else None
        self.wrist_cam = WristCameraAdapter(env, camera_name=dcfg.VLA_CAM_NAME) if want_cam else None
        self.left_cam = WristCameraAdapter(env, camera_name=dcfg.LEFT_CAM_NAME) if want_cam else None
        self.right_cam = WristCameraAdapter(env, camera_name=dcfg.RIGHT_CAM_NAME) if want_cam else None
        self.top_cam = WristCameraAdapter(env, camera_name=dcfg.TOP_CAM_NAME) if want_cam else None
        self._reset_strategy = build_reset_strategy(config.reset_mode, config.seed)
        self._handle_specs = build_handle_specs()   # 把手规格表，缓存一次（含 config import）
        self._target_markers = None                 # 懒构造的目标位姿箭头接口
        self._sim_dt = float(env_cfg.sim.dt * env_cfg.decimation)
        self._clock = 0.0
        self._closed = False
        print(
            f"[scene_session] launched task={config.task_id} device={config.device} "
            f"control_hz~{1.0/self._sim_dt:.1f} cameras={'on' if want_cam else 'off'} "
            f"fp={'on' if config.enable_fp else 'off'} reset_mode={config.reset_mode.value}",
            flush=True,
        )
        return self

    # ------------------------------------------------------------------ control loop
    def reset(self, *, reset_index: int = 0, seed: int | None = None, settle_steps: int | None = None) -> Observation:
        seed = self.config.seed if seed is None else seed
        self.env.reset(seed=seed)
        try:
            self.provider.reset_cabinet_joint("joint_0", 0.0)
        except Exception:
            pass
        self._reset_strategy.apply(self.env, self.provider, seed=seed, reset_index=reset_index)
        self._apply_saved_home()
        self._settle(self.config.settle_steps if settle_steps is None else settle_steps)
        self.provider.set_sim_time(self._clock)
        return self.observe()

    def _apply_saved_home(self) -> None:
        """把机器人手臂写到 Franka 面板「Set Home」存的 home(robot_home.json)。

        env.reset 事件只改了 default_joint_pos，但本场景的 reset 不会把它写进真实关节(实测 reset 后
        手臂≈0 而非 home)，所以这里在 settle 前显式 write_joint_state_to_sim 到 home。这样【任何入口】
        (含不带 --franka)reset 后手臂都停在保存的 home；没存过(saved_home_q=None)则保持原行为不动。
        """
        try:
            from .robot_home import saved_home_q

            home = saved_home_q()
            if home is None:
                return
            import torch

            robot = self.env.unwrapped.scene["robot"]
            arm_ids = self.provider._arm_joint_ids
            env_ids = torch.tensor([0], device=self.provider.device)
            pos = robot.data.joint_pos[0:1].clone()
            vel = robot.data.joint_vel[0:1].clone()
            home_t = torch.tensor(home, dtype=pos.dtype, device=pos.device)
            for i, jid in enumerate(arm_ids.tolist()):
                pos[0, jid] = home_t[i]
                vel[0, jid] = 0.0
            robot.write_joint_state_to_sim(pos, vel, env_ids=env_ids)
            robot.set_joint_position_target(pos, env_ids=env_ids)
            print(f"[scene_session] arm written to saved home {tuple(round(v,4) for v in home)}", flush=True)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[scene_session] apply saved home (write) failed: {exc}", flush=True)

    def step(self, joint_target, gripper, *, hold_if_none: bool = True) -> Observation:
        """发一步指令。joint_target=7 维**绝对关节弧度**(q_des)；gripper∈{0=open,1=close}。"""
        if joint_target is None:
            if hold_if_none:
                return self.hold(gripper)
            raise ValueError("joint_target is None (set hold_if_none=True to hold instead)")
        gripper_cmd = pi05_gripper_to_env_cmd(gripper)
        action = self.provider.make_joint_action_from_q_des(joint_target, gripper_cmd)
        self.env.step(action)
        self._advance_clock()
        return self.observe()

    def hold(self, gripper=None) -> Observation:
        """保持当前关节位置一步（gripper=None 时按当前夹爪宽度自动判断开合）。"""
        st = self.provider.get_state()
        g = None if gripper is None else pi05_gripper_to_env_cmd(gripper)
        self.env.step(self.provider.make_hold_joint_action(st, g))
        self._advance_clock()
        return self.observe()

    # ------------------------------------------------------------------ observation
    def observe(self) -> Observation:
        """读一帧完整观测（无副作用）。分组返回 pi05 / foundationpose / raw + cameras。"""
        st = self.provider.get_state()
        scene = self.env.unwrapped.scene
        robot = scene["robot"]
        base_pos = robot.data.root_pos_w[0].detach().cpu().tolist()
        base_quat = robot.data.root_quat_w[0].detach().cpu().tolist()  # (w,x,y,z)

        raw = RawBlock(
            joint_pos=[float(v) for v in st.robot.joint_pos.detach().cpu().tolist()],
            arm_joint_pos=[float(v) for v in self.provider.arm_joint_pos(st).detach().cpu().tolist()],
            joint_vel=[float(v) for v in st.robot.joint_vel.detach().cpu().tolist()],
            tcp_pose=[float(v) for v in self._ee_pose_xyzw(st)],
            gripper_width=float(st.robot.gripper_width),
            base_pose_w=[float(v) for v in (list(base_pos) + list(base_quat))],
        )

        handles = read_handles_in_base(scene, env_id=0, specs=self._handle_specs)

        cameras: dict[str, CameraView] = {}
        if self.front_cam is not None:
            cameras["front"] = self._capture_view("front", self._dcfg.FRONT_CAM_NAME, self.front_cam)
            cameras["wrist"] = self._capture_view("wrist", self._dcfg.VLA_CAM_NAME, self.wrist_cam)
            if self.left_cam is not None and self.left_cam.is_available():
                cameras["left"] = self._capture_view(
                    "left", self._dcfg.LEFT_CAM_NAME, self.left_cam, require_depth=True)
            if self.right_cam is not None and self.right_cam.is_available():
                cameras["right"] = self._capture_view(
                    "right", self._dcfg.RIGHT_CAM_NAME, self.right_cam, require_depth=True)
            if self.top_cam is not None and self.top_cam.is_available():
                cameras["top"] = self._capture_view(
                    "top", self._dcfg.TOP_CAM_NAME, self.top_cam, require_depth=True)

        size = tuple(self.config.vla_rgb_size)
        pi05 = Pi05Block(
            image=_resize_rgb(cameras["front"].rgb, size) if "front" in cameras else None,
            wrist_image=_resize_rgb(cameras["wrist"].rgb, size) if "wrist" in cameras else None,
            state8=raw.arm_joint_pos[:7] + [raw.gripper_width],
        )

        fp = None
        if self.config.enable_fp and cameras:
            fp = FoundationPoseBlock(
                views=cameras,
                object_gt_poses=_object_gt_poses(st, self.config.tracked_objects),
                _tracked=tuple(self.config.tracked_objects),
                _pose_source="sim_gt",
            )

        return Observation(
            pi05=pi05, raw=raw, cameras=cameras, foundationpose=fp, handles=handles, sim_time=self._clock
        )

    get_observation = observe

    # ------------------------------------------------------------------ 目标位姿箭头（USE/TEST 预留接口）
    def show_target_poses(self, poses, *, names=None, use_arrows: bool = True, axis_length: float = 0.08) -> None:
        """画一组目标位姿箭头（其他模块可调，平时不调用就不画）。

        poses: PoseState / (pos3,quat4_wxyz) / 7元序列 / Nx7 tensor，单个或列表。
        names: 可选 N 个稳定 name（同名复用同一 marker）。headless 时静默 no-op。
        """
        self._ensure_target_markers().show(poses, names=names, use_arrows=use_arrows, axis_length=axis_length)

    def clear_target_poses(self) -> None:
        if self._target_markers is not None:
            self._target_markers.clear()

    # ------------------------------------------------------------------ 可视化开关（接入模块/调试者自选）
    def set_collision_visible(self, on: bool) -> None:
        """开/关碰撞体可视化（全局 carb 设置）。"""
        try:
            import carb

            s = carb.settings.get_settings()
            s.set_int("/persistent/physics/visualizationDisplayColliders", 2 if on else 0)
            s.set_bool("/persistent/physics/visualizationDisplayColliderNormals", False)
        except Exception:
            pass

    def set_markers_visible(self, on: bool) -> None:
        """开/关黄色 InitCorner 区域标记方块的显示。"""
        try:
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            for i in range(8):
                prim = stage.GetPrimAtPath(f"/World/envs/env_0/InitCorner_{i}")
                if prim.IsValid():
                    img = UsdGeom.Imageable(prim)
                    img.MakeVisible() if on else img.MakeInvisible()
        except Exception:
            pass

    def _ensure_target_markers(self):
        if self._target_markers is None:
            from .target_markers import TargetPoseMarkers

            self._target_markers = TargetPoseMarkers(enabled=not self.config.headless)
        return self._target_markers

    def handle_poses_in_base(self) -> dict:
        """只读把手/拉手在机器人基坐标系下的 pose（不抓相机，给状态机轻量调用）。

        返回 {name: {asset, link, calibrated, functional, position[3], quat_wxyz[4],
        position_world[3], quat_wxyz_world[4]}}。name ∈ {microwave_door, top_drawer,
        middle_drawer, bottom_drawer, coffee_lever}（资产存在时）。
        """
        return read_handles_in_base(self.env.unwrapped.scene, env_id=0, specs=self._handle_specs)

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        global _OWNED_APP_STARTED
        try:
            self.env.close()
        finally:
            self._closed = True
            if self._owns_app and self._app_launcher is not None:
                try:
                    self._app_launcher.app.close()
                except Exception:
                    pass
                _OWNED_APP_STARTED = False

    def __enter__(self) -> "SceneSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------ internals
    def _capture_view(self, name: str, sensor_name: str, adapter, require_depth=None) -> CameraView:
        rd = self.config.enable_fp if require_depth is None else require_depth
        frame = adapter.capture(require_depth=rd)
        return CameraView(
            name=name,
            sensor_name=sensor_name,
            rgb=frame.get("rgb"),
            depth=frame.get("depth"),
            intrinsics=frame.get("intrinsics", {}),
            camera_pose_world=frame.get("camera_pose_world", {}),
            _frame=frame,
        )

    def _settle(self, n: int, gripper: float = 1.0) -> None:
        for _ in range(int(n)):
            st = self.provider.get_state()
            self.env.step(self.provider.make_hold_joint_action(st, gripper))
            self._advance_clock()

    def _advance_clock(self) -> None:
        self._clock += self._sim_dt
        self.provider.set_sim_time(self._clock)


# --------------------------------------------------------------------- helpers
def _make_app_launcher(config: SceneConfig):
    """库用法下自建完整 namespace 并启动 AppLauncher（必须在 import isaac 之前）。"""
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    ns = parser.parse_args([])           # 带全部默认值，避免缺字段导致 flag 被丢
    ns.headless = bool(config.headless)
    ns.device = config.device
    ns.enable_cameras = config.wants_cameras()
    return AppLauncher(ns)


def _resize_rgb(rgb, size):
    """把 RGB 缩放到 (w,h)（VLA 端用）。size=(w,h)。"""
    if rgb is None:
        return None
    import numpy as np
    from PIL import Image

    w, h = int(size[0]), int(size[1])
    arr = np.asarray(rgb)[..., :3].astype("uint8")
    if arr.shape[1] == w and arr.shape[0] == h:
        return arr
    return np.asarray(Image.fromarray(arr).resize((w, h)))


def _object_gt_poses(state, tracked) -> list[dict]:
    """从场景读跟踪物体的世界系 GT 位姿（pos_w + quat wxyz）。"""
    out: list[dict] = []
    for name in tracked:
        obj = state.objects.get(name)
        if obj is None:
            continue
        pos = obj.pose.pos_w.detach().cpu().tolist()
        quat = obj.pose.quat_w.detach().cpu().tolist()  # (w,x,y,z)
        out.append(
            {
                "name": name,
                "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                "quat_wxyz": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
            }
        )
    return out
