# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac-side glue for V1 teleop: env build, frame capture, carb keyboard. STATUS: ready.

Everything here lazily imports Isaac / gym / carb INSIDE functions, so the module
imports cleanly in a plain venv. It is only *called* after an ``AppLauncher`` has
started the sim (the entry scripts do that at module top, like the skill UI).

Shared by both entries so env setup + action mapping are identical:
  * ``gello_to_isaac_joint_test.py --drive_isaac``  (drive only, no recording)
  * ``collect_teleop_demos_joint_v1.py``            (drive + record)
"""

from __future__ import annotations

# Mirror the skill UI's interactive env tweaks (no auto-reset, deterministic).
_AUTO_RESET_TERMINATIONS = ("cube_1_dropping", "cube_2_dropping", "cube_3_dropping", "success", "cubes_stacked")
_OBJECT_NAMES = ("cube_1", "cube_2", "cube_3", "knife", "cabinet")


def build_teleop_env(
    *,
    task_id: str,
    num_envs: int = 1,
    device: str = "cuda:0",
    use_fabric: bool = True,
    enable_wrist_d435: bool = False,
    enable_depth: bool = False,
    free_microwave_door: bool = True,
    seed: int = 1,
    decimation: int | None = None,
    arm_stiffness: float | None = None,
    arm_damping: float | None = None,
    enable_vla_depth: bool = False,
    enable_front_depth: bool = False,
    enable_left_depth: bool = True,
    enable_right_depth: bool = True,
    enable_top_depth: bool = True,
    vla_resolution=None,
    front_resolution=None,
    left_resolution=None,
    right_resolution=None,
    top_resolution=None,
    cfg_hook=None,
):
    """Build the V1 joint-policy env for teleop. Returns (env, env_cfg, cam_attached).

    Disables episode time-out + cube terminations so the scene never auto-resets
    under the human (matches skill_test_ui_joint). Attaches the wrist D435 when
    requested (requires the app launched with ``enable_cameras=True``).

    Teleop-precision knobs (None = leave the task default):
      * ``decimation`` — physics steps per control step. The base task uses 5
        (sim.dt=0.01 -> 20 Hz control). Lower it to raise the control rate
        (2 -> 50 Hz, 1 -> 100 Hz) for tighter teleop tracking.
      * ``arm_stiffness`` / ``arm_damping`` — Franka arm joint PD gains. The base
        cfg is soft (80 / 4) so q_des is tracked loosely; ~400 / 80 (the Isaac
        Lab HIGH_PD values) tracks the commanded joints far more accurately.
    """
    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    # Put the legacy skill-state-machine root on sys.path so ``runtime.*``
    # (SceneStateProvider et al.) resolves — same bootstrap the V1 skill_runtime uses.
    from franka_v1_skill_lab.skill_runtime._legacy import ensure_legacy_on_path

    ensure_legacy_on_path()

    env_cfg = parse_env_cfg(task_id, device=device, num_envs=num_envs, use_fabric=use_fabric)
    env_cfg.seed = seed

    # --- teleop-precision overrides -----------------------------------------
    if decimation is not None and decimation >= 1:
        env_cfg.decimation = int(decimation)
        # keep rendering cadence sane relative to the new control rate
        if hasattr(env_cfg.sim, "render_interval"):
            env_cfg.sim.render_interval = max(1, int(decimation))
    if (arm_stiffness is not None or arm_damping is not None) and hasattr(env_cfg.scene, "robot"):
        acts = getattr(env_cfg.scene.robot, "actuators", {}) or {}
        for _name, _act in acts.items():
            jn = " ".join(getattr(_act, "joint_names_expr", []) or [])
            if "finger" in jn or "hand" in _name.lower():
                continue  # leave the gripper actuator alone
            if arm_stiffness is not None:
                _act.stiffness = float(arm_stiffness)
            if arm_damping is not None:
                _act.damping = float(arm_damping)
        print(f"[teleop] arm PD gains -> stiffness={arm_stiffness} damping={arm_damping}", flush=True)
    if getattr(env_cfg, "events", None) is not None and hasattr(env_cfg.events, "randomize_cube_positions"):
        env_cfg.events.randomize_cube_positions = None
    env_cfg.episode_length_s = 1.0e9
    if getattr(env_cfg, "terminations", None) is not None:
        for t in _AUTO_RESET_TERMINATIONS:
            if hasattr(env_cfg.terminations, t):
                setattr(env_cfg.terminations, t, None)
    if free_microwave_door and hasattr(env_cfg.scene, "microwave") and hasattr(env_cfg.scene.microwave, "actuators"):
        for _act in env_cfg.scene.microwave.actuators.values():
            _act.stiffness = 0.0
            _act.damping = 2.0
    env_cfg.viewer.eye = (2.0, -2.0, 1.4)
    env_cfg.viewer.lookat = (0.45, 0.0, 0.15)

    cam_attached = False
    if enable_wrist_d435:
        try:
            from franka_v1_skill_lab.sensors.d435.d435_scene_cfg import attach_wrist_cameras

            # Both shared-scene cameras; D435 body stays HIDDEN. RGB for VLA comes
            # from vla_libero_eye_in_hand; depth (if requested) from the FP camera.
            attach_wrist_cameras(
                env_cfg, enable_cameras=True, show_body=False,
                enable_fp_depth=enable_depth, enable_vla_depth=enable_vla_depth,
                enable_front_depth=enable_front_depth,
                enable_left_depth=enable_left_depth,
                enable_right_depth=enable_right_depth,
                enable_top_depth=enable_top_depth,
                vla_resolution=vla_resolution, front_resolution=front_resolution,
                left_resolution=left_resolution, right_resolution=right_resolution,
                top_resolution=top_resolution,
            )
            cam_attached = True
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[teleop] WARNING: could not attach wrist cameras ({exc}); continuing without camera.", flush=True)

    # 建 env 之前的最后一道 cfg 加工（例如把最新保存场景的物体位姿/缩放覆盖进来）。
    if cfg_hook is not None:
        cfg_hook(env_cfg)

    env = gym.make(task_id, cfg=env_cfg)
    env.reset(seed=seed)
    return env, env_cfg, cam_attached


def ee_pose_xyzw(state) -> list[float]:
    """TCP pose as [x,y,z,qx,qy,qz,qw], position in env-local frame."""
    pos = (state.robot.tcp_pose.pos_w - state.env_origin_w).detach().cpu().tolist()
    quat_wxyz = state.robot.tcp_pose.quat_w.detach().cpu().tolist()  # (w,x,y,z)
    w, x, y, z = quat_wxyz[:4]
    return [float(pos[0]), float(pos[1]), float(pos[2]), float(x), float(y), float(z), float(w)]


def capture_objects(state, names=_OBJECT_NAMES) -> list[dict]:
    """GT object poses from the live scene, env-local position, quat (w,x,y,z)."""
    out: list[dict] = []
    origin = state.env_origin_w
    for name in names:
        obj = state.objects.get(name)
        if obj is None:
            continue
        pos = (obj.pose.pos_w - origin).detach().cpu().tolist()
        quat = obj.pose.quat_w.detach().cpu().tolist()
        out.append(
            {
                "name": name,
                "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                "quat_wxyz": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
            }
        )
    return out


class CarbKeyboardSource:
    """Subscribe to omni keyboard events and queue CollectionEvents. STATUS: ready (GUI).

    Mirrors isaaclab.devices.keyboard.Se2Keyboard's subscribe pattern. Each key
    press is mapped via COLLECTION_KEYMAP / SKILL_KEYS into the collection event
    queue; the driver drains it with :meth:`poll` each sim step.
    """

    def __init__(self):
        import weakref

        import carb
        import omni

        from .keyboard_episode_controller import COLLECTION_KEYMAP, SKILL_KEYS

        self._carb = carb
        self._keymap = COLLECTION_KEYMAP
        self._skill_keys = SKILL_KEYS
        self._queue: list[tuple] = []
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_event(event, *args),
        )

    def _on_event(self, event, *args, **kwargs):
        from .keyboard_episode_controller import CollectionEvent

        if event.type != self._carb.input.KeyboardEventType.KEY_PRESS:
            return
        name = event.input.name
        if name in self._skill_keys:
            self._queue.append((CollectionEvent.SELECT_SKILL, self._skill_keys[name]))
        elif name in self._keymap:
            self._queue.append((self._keymap[name], None))

    def poll(self, recorded_steps: int = 0) -> list[tuple]:
        out = self._queue[:]
        self._queue.clear()
        return out

    def close(self):
        try:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
        except Exception:
            pass
