"""Build the official Franka open-drawer policy observation from the live scene.

Mirrors ``cabinet_env_cfg.py:ObservationsCfg.PolicyCfg`` (concatenate_terms=True), in order:

    1. joint_pos            = robot joint_pos_rel              (9)
    2. joint_vel            = robot joint_vel_rel              (9)
    3. cabinet_joint_pos    = drawer joint joint_pos_rel       (1)
    4. cabinet_joint_vel    = drawer joint joint_vel_rel       (1)
    5. rel_ee_drawer_distance = handle_pos_w - tcp_pos_w       (3)
    6. actions              = last_action                      (8)
                                                       total = 31

We compute ``handle_pos - tcp_pos`` ourselves: from the ``cabinet_frame`` FrameTransformer
(``drawer_handle_top``) when present, otherwise from the cabinet ``link_1`` body pose combined with
the ``BottomHandleProxy`` local offset.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils

from runtime.drawer_target_config import DRAWER_TARGETS
from runtime.scene_state_provider import SceneState

# bottom-drawer handle proxy local offset on link_1 (see stack_joint_pos_env_cfg.py).
# The cabinet USD is spawned with scale 0.62, which propagates to this child prim's local
# translate, so we scale the authored offset to recover the world offset from link_1.
_HANDLE_PROXY_LOCAL_OFFSET = (0.11946, 0.01491, 1.06183)
_CABINET_SCALE = 0.62


class DrawerObsAdapter:
    def __init__(self, env, env_id: int = 0, drawer_joint_name: str = "joint_0"):
        self.env = env
        self.env_id = env_id
        self.scene = env.unwrapped.scene
        self.device = self.scene.device
        self.drawer_joint_name = drawer_joint_name
        self.cabinet = self.scene["cabinet"]
        names = list(getattr(self.cabinet.data, "joint_names", []))
        if drawer_joint_name in names:
            self._drawer_joint_id = names.index(drawer_joint_name)
        else:
            ids, _ = self.cabinet.find_joints(drawer_joint_name)
            if not ids:
                raise RuntimeError(f"drawer joint '{drawer_joint_name}' not found; available={names}")
            self._drawer_joint_id = int(ids[0])
        self._has_cabinet_frame = "cabinet_frame" in self.scene.keys()
        self.last_obs_dim: int | None = None

    def _handle_pos_w(self) -> torch.Tensor:
        if self._has_cabinet_frame:
            frame = self.scene["cabinet_frame"]
            return frame.data.target_pos_w[:, 0, :]
        # fallback: link_1 body pose + local handle offset
        body_names = list(getattr(self.cabinet.data, "body_names", []))
        link_idx = next((i for i, n in enumerate(body_names) if "link_1" in n), 0)
        link_pos = self.cabinet.data.body_pos_w[:, link_idx]
        link_quat = self.cabinet.data.body_quat_w[:, link_idx]
        offset_local = tuple(v * _CABINET_SCALE for v in _HANDLE_PROXY_LOCAL_OFFSET)
        offset = torch.tensor(offset_local, device=self.device).repeat(link_pos.shape[0], 1)
        handle_pos, _ = math_utils.combine_frame_transforms(link_pos, link_quat, offset)
        return handle_pos

    def build(self, state: SceneState | None = None) -> torch.Tensor:
        robot = self.scene["robot"]
        ee_frame = self.scene["ee_frame"]

        joint_pos_rel = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel_rel = robot.data.joint_vel - robot.data.default_joint_vel

        cab_jp = (
            self.cabinet.data.joint_pos[:, self._drawer_joint_id]
            - self.cabinet.data.default_joint_pos[:, self._drawer_joint_id]
        ).unsqueeze(-1)
        cab_jv = (
            self.cabinet.data.joint_vel[:, self._drawer_joint_id]
            - self.cabinet.data.default_joint_vel[:, self._drawer_joint_id]
        ).unsqueeze(-1)

        tcp_pos_w = ee_frame.data.target_pos_w[:, 0, :]
        handle_pos_w = self._handle_pos_w()
        rel_ee_drawer = handle_pos_w - tcp_pos_w

        last_action = self.env.unwrapped.action_manager.action

        obs = torch.cat(
            (joint_pos_rel, joint_vel_rel, cab_jp, cab_jv, rel_ee_drawer, last_action), dim=-1
        )
        self.last_obs_dim = int(obs.shape[-1])
        return obs


class SelectedDrawerObsAdapter:
    """Deployment-side builder of the 31-d selected-drawer obs, matching the training env.

    The custom RL env (custom_drawer_mdp) builds obs from its drawer_frames FrameTransformer
    (zero offset on link_0 / link_2) + the per-env selected joint. The deployment env
    (Isaac-Stack-Cube-Franka-JointPolicy-v0) has no drawer_frames sensor, so here the selected
    handle = the drawer link body world pose (link origin == zero-offset frame), and the selected
    joint is resolved from the central target->joint config. Order matches training exactly:
        joint_pos_rel(9) | joint_vel_rel(9) | sel_joint_pos(1) | sel_joint_vel(1) |
        (sel_handle - tcp)(3) | last_action(8)  -> 31
    """

    def __init__(self, env, target_drawer: str, env_id: int = 0):
        cfg = DRAWER_TARGETS[target_drawer]
        self.env = env
        self.target_drawer = target_drawer
        self.env_id = env_id
        self.scene = env.unwrapped.scene
        self.device = self.scene.device
        # 抽屉所属场景成员(默认旧 Cabinet_44853='cabinet'；白柜抽屉='sektion_cabinet')。
        self.member = cfg.get("member", "cabinet")
        self.cabinet = self.scene[self.member]
        self.joint_name = cfg["joint_name"]
        self.link_name = cfg["link_name"]
        # handle_offset starts from the config default ONLY as a placeholder; it MUST be overwritten by
        # the single handle-pose source = the UI-adjusted pose (saved grasp_poses.json, or the live
        # override injected via set_handle_pose). No proxy / mesh / computed fallback (deleted).
        self.handle_offset = torch.tensor(cfg["handle_offset"], dtype=torch.float32, device=self.device)

        jnames = list(getattr(self.cabinet.data, "joint_names", []))
        self._joint_id = jnames.index(self.joint_name)
        bnames = list(getattr(self.cabinet.data, "body_names", []))
        self._link_idx = next((i for i, n in enumerate(bnames) if self.link_name == n), None)
        if self._link_idx is None:
            self._link_idx = next((i for i, n in enumerate(bnames) if self.link_name in n), 0)
        self.last_obs_dim: int | None = None
        # SINGLE handle pose source = the UI-adjusted pose. Load the pos+quat the user set in the Grasp
        # Pose panel (persisted to grasp_poses.json). If the skill has a LIVE override it overwrites these
        # via set_handle_pose(). There is NO other handle pose (proxy/mesh/computed sources deleted).
        self.handle_quat = None       # UI grasp orientation (link-local)
        if not self._load_saved_handle_offset():
            raise RuntimeError(
                f"no saved UI handle pose 'handle_{self.target_drawer}' (link={self.link_name}) in "
                "grasp_poses.json; set it in the Grasp Pose panel first (single-source handle pose).")

    def set_handle_pose(self, pos, quat) -> None:
        """Inject the LIVE UI handle pose (link-local pos+quat) so the grasp target AND the tracked
        handle position share ONE source. Called by the skill when parameters carry override_grasp_local
        (the Grasp Pose panel's live value). Everything downstream (grasp/pull/push target, detach
        detection, azimuth, viz) reads handle_offset/handle_quat -> exactly the pose the user set."""
        self.handle_offset = torch.tensor(pos, dtype=torch.float32, device=self.device)
        self.handle_quat = torch.tensor(quat, dtype=torch.float32, device=self.device)

    def _load_saved_handle_offset(self) -> bool:
        """用【场景编辑里保存的 handle_<drawer> 位姿】(link 局部) 作为抓取目标偏移+朝向。用户改了 pose
        下次抓取就用新的，不再自动定位、不新增 pose。返回 True 表示用上了保存值。"""
        import json
        from pathlib import Path

        name = f"handle_{self.target_drawer}"          # handle_top_drawer / handle_middle_drawer / ...
        try:
            p = Path(__file__).resolve().parents[2] / (
                "franka_v1_skill_lab/scene/saved_scenes/v1_active/grasp_poses.json")
            poses = json.loads(p.read_text(encoding="utf-8")).get("poses", {})
            e = poses.get(name)
            if e is None or e.get("link") != self.link_name:
                return False
            self.handle_offset = torch.tensor(e["pos"], dtype=torch.float32, device=self.device)
            if e.get("quat") is not None:
                self.handle_quat = torch.tensor(e["quat"], dtype=torch.float32, device=self.device)
            print(f"[SelectedDrawerObsAdapter] {self.target_drawer} handle from SAVED pose '{name}' "
                  f"link-local pos={[round(float(v),4) for v in self.handle_offset.tolist()]}", flush=True)
            return True
        except Exception as exc:
            print(f"[SelectedDrawerObsAdapter] saved handle load failed ({exc}); falling back", flush=True)
            return False

    def selected_handle_pos_w(self) -> torch.Tensor:
        link_pos = self.cabinet.data.body_pos_w[:, self._link_idx]
        link_quat = self.cabinet.data.body_quat_w[:, self._link_idx]
        offset = self.handle_offset.unsqueeze(0).expand(link_pos.shape[0], -1)
        handle_pos, _ = math_utils.combine_frame_transforms(link_pos, link_quat, offset)
        return handle_pos

    def selected_drawer_joint_pos(self) -> float:
        return float(self.cabinet.data.joint_pos[self.env_id, self._joint_id])

    def build(self, state: SceneState | None = None) -> torch.Tensor:
        robot = self.scene["robot"]
        ee_frame = self.scene["ee_frame"]

        joint_pos_rel = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel_rel = robot.data.joint_vel - robot.data.default_joint_vel

        cab_jp = (
            self.cabinet.data.joint_pos[:, self._joint_id]
            - self.cabinet.data.default_joint_pos[:, self._joint_id]
        ).unsqueeze(-1)
        cab_jv = (
            self.cabinet.data.joint_vel[:, self._joint_id]
            - self.cabinet.data.default_joint_vel[:, self._joint_id]
        ).unsqueeze(-1)

        tcp_pos_w = ee_frame.data.target_pos_w[:, 0, :]
        rel = self.selected_handle_pos_w() - tcp_pos_w

        last_action = self.env.unwrapped.action_manager.action
        obs = torch.cat((joint_pos_rel, joint_vel_rel, cab_jp, cab_jv, rel, last_action), dim=-1)
        self.last_obs_dim = int(obs.shape[-1])
        return obs
