"""Shared drawer experiment harness (v2). Reused by capability-map, sessions, and pilot generators.

Builds the paper drawer scene via SceneSession (same as the GUI/Stage-0), wires the verified
executor + member-aware v2 adapter. Call from a script that already created AppLauncher.

    from _drawer_harness_v2 import launch_drawer_scene, run_id_dir, git_info
    H = launch_drawer_scene(app_launcher, device="cuda:0")
    spec = H.spec("middle_drawer")
    H.reset(spec)
    x, y, prov, tl, H.sim_time = H.run(spec, g, theta, H.sim_time)
    H.close()
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
for _p in (_PAPER, _PAPER / "franka_skill_state_machine", _PAPER / "deployment_calibration"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def git_info():
    def g(*a):
        try:
            return subprocess.check_output(["git", "-C", str(_PAPER), *a], text=True).strip()
        except Exception:
            return ""
    return {"git_commit": g("rev-parse", "HEAD"), "branch": g("branch", "--show-current"),
            "dirty_worktree": bool(g("status", "--short").strip())}


def run_id_dir(run_id: str) -> Path:
    d = _PAPER / "deployment_calibration" / "data" / run_id
    (d / "trajectories").mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class DrawerHarness:
    env: object
    provider: object
    executor: object
    adapter: object
    control_dt: float
    reg: dict
    handles: dict
    sim_time: float = 0.0
    _mod: object = field(default=None)

    def spec(self, drawer_name: str):
        return self._mod.MechanismSpec.resolve(drawer_name, self.reg, self.handles)

    def reset(self, spec, initial_open: float = 0.0):
        return self._mod.reset_full_v2(self.env, self.provider, spec, initial_open=initial_open,
                                       executor=self.executor)

    def set_damping(self, spec, damping: float):
        return self._mod.set_drawer_damping_v2(self.env, spec, damping)

    def read_damping(self, spec):
        return self._mod.read_drawer_damping_v2(self.env, spec)

    def run(self, spec, g, theta, sim_time=None, max_steps=1800):
        st = self.sim_time if sim_time is None else sim_time
        x, y, prov, tl, st = self._mod.run_drawer_episode_v2(
            self.env, self.provider, self.executor, self.adapter, self.control_dt, spec, g, theta,
            sim_time=st, max_steps=max_steps)
        self.sim_time = st
        return x, y, prov, tl, st

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


def launch_drawer_scene(app_launcher, device: str = "cuda:0", speed_scale: float = 5.0) -> DrawerHarness:
    from franka_v1_skill_lab.scene import V1_BASE_TASK_ID
    from franka_v1_skill_lab.scene_interface import ResetMode, SceneConfig, SceneMode, SceneSession
    from runtime.base_skill import set_speed_scale
    from runtime.ik_joint_adapter import IKJointAdapter
    from runtime.target_registry import TargetRegistry
    from state_machine.skill_executor import JointBackendConfig, SkillExecutor
    from skills.open_drawer_skill import OpenDrawerIKConfig
    from skills.close_drawer_skill import CloseDrawerIKConfig
    import adapters.articulated_drawer_v2 as adv

    cfg = SceneConfig(
        mode=SceneMode.TEST, task_id=V1_BASE_TASK_ID, device=device, headless=True,
        enable_cameras=False, enable_fp=False, free_microwave_door=False, load_latest_scene=True,
        add_microwave_stand=False, replace_microwave_with_fridge=False, lock_knife=True,
        enable_collision_monitor=False, spawn_init_markers=False, refine_handle_collisions=True,
        apply_saved_camera_offsets=False, reset_mode=ResetMode.STATIC, seed=1,
        disable_auto_reset=True, exclude_members=("microwave", "dishwasher"), hidden_members=())
    session = SceneSession.launch(cfg, _app_launcher=app_launcher)
    env, provider = session.env, session.provider
    control_dt = float(getattr(session, "_sim_dt", 0.02)) or 0.02
    set_speed_scale(speed_scale)
    registry = TargetRegistry(env.unwrapped.device)
    adapter = IKJointAdapter(env)
    backend = JointBackendConfig(
        mode="joint", grasp_backend="joint_ik", place_backend="joint_ik", drawer_backend="ik_pull",
        adapter=adapter, drawer_env=env, arm_joint_ids=provider._arm_joint_ids, drawer_joint_name="joint_0",
        drawer_open_ik_config=OpenDrawerIKConfig(use_turn_to_face=False, start_from_current=True),
        drawer_close_ik_config=CloseDrawerIKConfig(use_turn_to_face=False, start_from_current=True))
    executor = SkillExecutor(registry, log_path="logs/skill_tests/harness_v2.jsonl", backend=backend)
    return DrawerHarness(env=env, provider=provider, executor=executor, adapter=adapter,
                         control_dt=control_dt, reg=adv.load_mechanism_registry(),
                         handles=adv.load_paper_handles(), _mod=adv)
