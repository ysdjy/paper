"""General IK-based OPEN-DRAWER skill (stage-1 parameterised; no learned policy, no joint cheating).

Preserved from the original design (task spec section 5.1):
* the handle pose is read LIVE every step (link body pose o link-local handle offset);
* the open direction is derived from the cabinet orientation;
* the drawer is opened ONLY by physically gripping + pulling the handle;
* ``drawer_joint_target`` is always None and ``set_cabinet_joint_target`` is never called;
* top / middle drawers supported (bottom is locked, excluded from experiments).

Stage-1 additions:
* execution parameters come from ``request.parameters`` via :class:`ParamResolver`
  (pre_grasp_clearance / pull_lead / max_pos_step / max_ori_step / approach_line_lead /
  grasp_offset_local_xyz / close_duration / settle_duration / release_duration / reach_timeout /
  pull_timeout), all traced and really entering the geometry;
* ``target_open_position`` is a TASK TARGET (request > DRAWER_TARGETS default 0.20, clamped to the
  joint limit); success = ``actual >= target - tolerance``; ``drawer_overshoot`` is recorded because
  the free-sliding drawer can coast past the target;
* ``grasp_offset_local_xyz`` is added in the drawer LINK local frame (NOT a world offset) before the
  link transform, so it produces accurate / edge / slipped / missed grasps;
* a handle-detachment detector (no contact sensor): sustained large TCP-vs-handle error together with
  no drawer progress while in PULL;
* per-step telemetry + running TCP / handle tracking stats + a full ``outcomes`` dict.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import torch

import isaaclab.utils.math as math_utils

from runtime.base_skill import SkillCommand, PoseState, pose_error, pose_tensor, step_pose
from runtime.drawer_ik_common import HOME_Q_VERTICAL_RAISED, grasp_quat_from_open_dir, open_direction_world
from runtime.drawer_obs_adapter import SelectedDrawerObsAdapter
from runtime.drawer_target_config import DRAWER_TARGETS
from runtime.experiment_params import InvalidExperimentParameter, ParamResolver
from runtime.scene_state_provider import SceneState
from runtime.skill_request import SkillRequest
from runtime.skill_result import SkillResult
from runtime.skill_types import ExecutionStatus, FailureReason, SkillType

# prismatic drawer joint physical range (custom_drawer_config.py): limits [0, 0.8]
DRAWER_JOINT_MIN = 0.0
DRAWER_JOINT_MAX = 0.8


@dataclass
class OpenDrawerIKConfig:
    start_from_current: bool = True
    use_turn_to_face: bool = True
    face_joint_threshold: float = 0.12
    face_timeout: float = 5.0
    pre_grasp_clearance: float = 0.12
    pull_lead: float = 0.08
    max_pos_step: float = 0.020
    max_ori_step: float = math.radians(6.0)
    reach_pos_threshold: float = 0.03
    reach_ori_threshold: float = math.radians(18.0)
    reach_stable_cycles: int = 6
    soft_reach_advance: float = 0.10
    soft_reach_ori: float = math.radians(40.0)
    approach_line_lead: float = 0.03
    close_duration: float = 1.0
    reach_timeout: float = 16.0
    pull_timeout: float = 16.0
    settle_duration: float = 0.5
    release_duration: float = 0.4
    home_joint_threshold: float = 0.12
    home_timeout: float = 6.0
    # 加权 DLS 让关节1优先旋转(reach 阶段把关节1代价调小)。实测对"把手在身后大角度"无效(局部 IK 不会
    # 主动甩底座,只会仰身)，故默认 0=关闭。保留接口备用。
    joint1_face_weight: float = 0.0
    # 【关节1优先旋转的正解】用一段笛卡尔圆弧转身：PRELIFT 抬高后，让 TCP 沿【绕机器人底座的水平圆弧】
    # 从当前方位角扫到把手方位角(home 高度、半径不变)。TCP 绕基座转 -> 普通 IK 被迫转关节1跟随，平滑、
    # 不仰身、不插离散关节路径点。到位后再 MOVE_TO_PRE_GRASP 径向接近把手。
    use_arc_to_face: bool = True
    arc_rate: float = 1.4          # 圆弧扫掠角速度上限(rad/s)
    arc_face_tol: float = 0.14     # 方位角对齐阈值(rad,~8deg) -> 进 reach
    arc_timeout: float = 8.0
    # 【任务空间 via-point blend(方案A)】：PRELIFT 后走一条二次贝塞尔 A(当前)->V(facing 途径点)->G(把手前)，
    # 控制点=V(天然不经过 V=blend 语义);同时关节1由 solve(joint0_cmd) 关节空间硬性带动转身,其余6轴任务空间
    # 跟踪贝塞尔轨迹(底座真转不仰身)。**实测**：对旧柜这种在机器人【斜后方~145°】的抽屉，转身后手腕落在
    # 抓不到的构型/顶到限位(末端朝向差 ~48°)，抓取失败——任务空间控制管不了手臂构型(冗余),这种大角度回转
    # 场景不可靠;关节空间 ARC_TO_FACE 落到已知好构型 faced_q、可靠。故默认【关闭】,仅作可切换实验项(对
    # 正前方/侧向、转身角小的抽屉可用)。优先于 use_arc_to_face。
    use_via_blend: bool = False
    via_blend_rate: float = 0.55   # 贝塞尔参数 s 的推进速度(/s)，约 1/via_blend_rate 秒走完
    via_standoff_radius: float = 0.52  # 途径点 V 到基座的水平半径(m,facing 方向的臂展待命点)
    via_strict: bool = False       # False=不严格到达 V(贝塞尔绕过);True=把 V 当 fine 点插入(严格经过)
    via_timeout: float = 9.0
    # --- task target + verification (stage-1) ---
    target_open_position: float = 0.20        # default == old success_threshold
    target_tolerance: float = 0.02
    # --- handle-detachment detector thresholds (calibrate from default-success runs) ---
    handle_detach_error: float = 0.06         # TCP-vs-live-handle distance (m) considered "lost grip"
    handle_detach_no_progress_window: float = 1.5   # s of no drawer progress
    handle_detach_progress_eps: float = 0.003       # m drawer advance counted as progress


@dataclass
class _Runtime:
    state: str = "IDLE"
    start_time: float = 0.0
    state_start_time: float = 0.0
    stable_count: int = 0
    drawer_control_mode: str = "ik_pull"
    drawer_joint_name: str = ""
    drawer_joint_target: float | None = None  # always None (physical pull only)
    initial_joint_pos: float = 0.0
    current_joint_pos: float = 0.0
    last_command_pose: PoseState | None = None
    final_error_pos: float | None = None
    final_error_ori: float | None = None
    last_failure_message: str | None = None
    handle_detached: bool = False
    target_reached_time: float | None = None
    timed_out: bool = False
    # running stats
    track_err_max: float = 0.0
    track_err_sum: float = 0.0
    track_err_n: int = 0
    handle_err_max: float = 0.0
    handle_err_sum: float = 0.0
    handle_err_n: int = 0
    history: list[dict] = field(default_factory=list)


class OpenDrawerIKSkill:
    backend = "ik_pull"

    def __init__(self, request: SkillRequest, env, ik_adapter, config: OpenDrawerIKConfig | None = None):
        self.request = request
        self.env = env
        self.adapter = ik_adapter
        self.cfg = config or OpenDrawerIKConfig()
        self.target_drawer = request.destination_object or "top_drawer"
        self.status = ExecutionStatus.IDLE
        self.failure_reason = FailureReason.NONE
        cfg = DRAWER_TARGETS.get(self.target_drawer, {})
        self.runtime = _Runtime(drawer_joint_name=cfg.get("joint_name", ""))
        self.obs_adapter: SelectedDrawerObsAdapter | None = None
        self.last_q = None
        self._approach_start = None
        self._approach_end = None
        self._approach_quat = None
        self._settle_target = None    # 开到位瞬间锁定的【实时 TCP 位姿】(SETTLE/RELEASE 都保持它不动)
        self._release_target = None   # RELEASE 阶段锁定的固定位姿(开抽屉到位时捕获一次)
        self._prelift_z = None        # Bug1: 点技能后先竖直抬到的 home 高度(由 request 参数 prelift_z 给)
        self._prelift_target = None
        self._arc_hold_q = None       # ARC_TO_FACE: 转身时锁定的其余关节姿态(只动关节1)
        self._arc_j1 = 0.0
        self._via_A = None            # VIA_BLEND 贝塞尔起点(TCP pos)
        self._via_V = None            # facing 途径点(控制点)
        self._via_G = None            # 终点=把手前 pre-grasp PoseState
        self._via_s = 0.0             # 贝塞尔参数
        self.resolver = ParamResolver(request.parameters)
        self.last_telemetry: dict = {}
        self._eff: dict = {}
        self._param_error: str | None = None
        self._grasp_offset_local = torch.zeros(3)
        self._last_progress_time = 0.0
        self._progress_ref = 0.0
        self.target_open_position = float(cfg.get("success_threshold", 0.20))
        self.initial_handle_pos = None

    @property
    def current_state(self) -> str:
        return self.runtime.state

    # ---- param resolution -------------------------------------------------
    def _resolve_params(self) -> bool:
        c = self.cfg
        at = "open_drawer_skill"
        try:
            r = self.resolver
            e = {}
            e["pre_grasp_clearance"], _ = r.float("pre_grasp_clearance", c.pre_grasp_clearance,
                                                  minimum=0.02, maximum=0.30, hard_minimum=0.0, hard_maximum=0.6, applied_at=at)
            e["pull_lead"], _ = r.float("pull_lead", c.pull_lead,
                                        minimum=0.02, maximum=0.20, hard_minimum=0.0, hard_maximum=0.5, applied_at=at)
            e["max_pos_step"], _ = r.float("max_pos_step", c.max_pos_step,
                                           minimum=0.005, maximum=0.05, hard_minimum=0.001, hard_maximum=0.2, applied_at=at)
            e["max_ori_step"], _ = r.float("max_ori_step", c.max_ori_step,
                                           minimum=math.radians(1.0), maximum=math.radians(15.0), hard_maximum=math.radians(45.0), applied_at=at)
            e["approach_line_lead"], _ = r.float("approach_line_lead", c.approach_line_lead,
                                                 minimum=0.01, maximum=0.10, hard_maximum=0.3, applied_at=at)
            e["close_duration"], _ = r.float("close_duration", c.close_duration,
                                             minimum=0.3, maximum=3.0, hard_minimum=0.0, hard_maximum=10.0, applied_at=at)
            e["settle_duration"], _ = r.float("settle_duration", c.settle_duration,
                                              minimum=0.1, maximum=3.0, hard_minimum=0.0, hard_maximum=10.0, applied_at=at)
            e["release_duration"], _ = r.float("release_duration", c.release_duration,
                                               minimum=0.1, maximum=3.0, hard_minimum=0.0, hard_maximum=10.0, applied_at=at)
            e["reach_timeout"], _ = r.float("reach_timeout", c.reach_timeout, minimum=4.0, maximum=40.0, applied_at=at)
            e["pull_timeout"], _ = r.float("pull_timeout", c.pull_timeout, minimum=4.0, maximum=40.0, applied_at=at)
            off, _ = r.vec3("grasp_offset_local_xyz", [0.0, 0.0, 0.0],
                            minimum=-0.06, maximum=0.06, hard_minimum=-0.2, hard_maximum=0.2, applied_at=at)
            self._grasp_offset_local = torch.tensor(off, dtype=torch.float32, device=self.adapter.device)
            # task target: target_open_position (request > config default), clamped to joint limit
            tgt, _ = r.float("target_open_position", c.target_open_position,
                             minimum=DRAWER_JOINT_MIN + 0.02, maximum=DRAWER_JOINT_MAX,
                             hard_minimum=DRAWER_JOINT_MIN, hard_maximum=DRAWER_JOINT_MAX, applied_at=at + ".task_target")
            self.target_open_position = float(tgt)
            e["target_open_position"] = self.target_open_position
            self._eff = e
            return True
        except InvalidExperimentParameter as exc:
            self._param_error = str(exc)
            return False

    # ---- live scene reads -------------------------------------------------
    def _handle_pos(self) -> torch.Tensor:
        """Live world position of the actual handle bar (link o handle_offset, NO grasp_offset)."""
        return self.obs_adapter.selected_handle_pos_w()[self.adapter.env_id]

    def _cabinet_quat(self) -> torch.Tensor:
        # 用 obs_adapter 解析出的成员(旧 cabinet 或白柜 sektion_cabinet)，不写死 'cabinet'。
        member = getattr(self.obs_adapter, "cabinet", None)
        if member is not None:
            return member.data.root_quat_w[self.adapter.env_id]
        return self.env.unwrapped.scene["cabinet"].data.root_quat_w[self.adapter.env_id]

    def _drawer_pos(self) -> float:
        return self.obs_adapter.selected_drawer_joint_pos()

    def _faced_seed_q(self) -> torch.Tensor:
        robot = self.env.unwrapped.scene["robot"]
        eid = self.adapter.env_id
        base_pos = robot.data.root_pos_w[eid]
        base_quat = robot.data.root_quat_w[eid]
        d = (self._handle_pos() - base_pos)
        d_base = math_utils.quat_apply(math_utils.quat_inv(base_quat).unsqueeze(0), d.unsqueeze(0))[0]
        azimuth = math.atan2(float(d_base[1]), float(d_base[0]))
        seed = self.home_q.clone()
        lo, hi = float(self.adapter._joint_lower[0]), float(self.adapter._joint_upper[0])
        seed[0] = max(lo, min(hi, azimuth))
        print(f"[OpenDrawerIKSkill] turn-to-face: handle azimuth(base) = {math.degrees(azimuth):.1f}deg "
              f"-> joint1 = {math.degrees(float(seed[0])):.1f}deg", flush=True)
        return seed

    def _post_prelift_state(self) -> str:
        if self.cfg.use_via_blend:
            return "VIA_BLEND"
        if self.cfg.use_arc_to_face:
            return "ARC_TO_FACE"
        if self.cfg.use_turn_to_face:
            return "TURN_TO_FACE"
        return "MOVE_TO_PRE_GRASP"

    def _via_blend_command(self, state: SceneState, dt: float):
        """任务空间 via-point blend：TCP 沿二次贝塞尔 A->V(控制点)->G 走(不经过 V)，关节1由 joint0_cmd
        关节空间硬性带动转身、其余6轴任务空间跟踪。返回 (SkillCommand, done_bool)。"""
        robot = self.env.unwrapped.scene["robot"]
        eid = self.adapter.env_id
        tcp = state.robot.tcp_pose
        if self._via_A is None:
            base = robot.data.root_pos_w[eid]
            self._via_G = self._grasp_pose(self._eff["pre_grasp_clearance"])
            self._via_A = tcp.pos_w.clone()
            gxy = self._via_G.pos_w[:2]
            dxy = gxy - base[:2]
            dn = dxy / (torch.linalg.norm(dxy) + 1e-9)
            R = min(float(self.cfg.via_standoff_radius), float(torch.linalg.norm(dxy)) * 0.75)
            z = float(self._prelift_z) if self._prelift_z is not None else float(self._via_A[2])
            V = base.clone()
            V[0] = base[0] + dn[0] * R
            V[1] = base[1] + dn[1] * R
            V[2] = z
            self._via_V = V
            self._via_s = 0.0
            # 关节1从当前角开始朝把手方位角(faced_q[0])带动
            self._arc_j1 = float(state.robot.joint_pos[self.adapter._joint_ids][0])
        a_tgt0 = float(self.faced_q[0])
        j0_now = float(state.robot.joint_pos[self.adapter._joint_ids][0])
        # 一旦 s 走完且关节1已正对：切到【普通 7 自由度 IK】settle 到 G(位置+朝向)，用冗余把手腕摆到能抓的
        # 朝向(joint0-lead 时其余6轴被任务完全定死、手腕姿态不可控,末端朝向会差)。settle 收敛即进 APPROACH。
        if getattr(self, "_via_settling", False) or (self._via_s >= 0.999 and abs(a_tgt0 - j0_now) <= self.cfg.arc_face_tol):
            self._via_settling = True
            target = self._via_G
            cmd = step_pose(tcp, target, self._eff["max_pos_step"], self._eff["max_ori_step"])
            # 零空间把【手腕关节(4,5,6)】拉向 faced_q 的手腕配置(其余关节偏好=当前,不动)，把因转身卡在
            # 限位的手腕挪到能抓的构型，随后任务项才能把末端朝向修到抓取朝向。
            q_pref = state.robot.joint_pos[self.adapter._joint_ids].clone()
            q_pref[4:7] = self.faced_q[4:7]
            ik = self.adapter.solve(cmd, null_gain=0.8, q_pref=q_pref)
            err = pose_error(tcp, target)
            self._last_phase_target = target
            done = err.position < 0.03 and err.orientation < math.radians(18)
            if not ik.success:
                q = self.last_q if self.last_q is not None else state.robot.joint_pos[self.adapter._joint_ids].clone()
                return SkillCommand(tcp, 1.0, self.status, control_mode="joint", joint_target=q,
                                    drawer_joint_target=None), done
            self.last_q = ik.q_des
            return SkillCommand(cmd, 1.0, self.status, control_mode="joint", joint_target=ik.q_des,
                                drawer_joint_target=None), done
        self._via_s = min(1.0, self._via_s + self.cfg.via_blend_rate * max(dt, 1e-3))
        s = self._via_s
        A, V, G = self._via_A, self._via_V, self._via_G.pos_w
        if self.cfg.via_strict:   # 严格经过 V：两段线性 A->V->G
            if s < 0.5:
                u = s / 0.5; P = A + (V - A) * u
            else:
                u = (s - 0.5) / 0.5; P = V + (G - V) * u
        else:                     # 不严格：二次贝塞尔(控制点 V，不经过 V)
            P = (1 - s) ** 2 * A + 2 * (1 - s) * s * V + s ** 2 * G
        # 朝向【随底座转动一起转】(绕世界 Z 预旋 j0_now-faced)：转身过程中手腕相对底座保持中性、不用反向
        # 拧 145°(那会把手腕顶到限位);当 j0_now->faced 时预旋量归零、朝向自然收敛到抓取朝向 G.quat。
        j0 = float(state.robot.joint_pos[self.adapter._joint_ids][0])
        rz = math_utils.quat_from_angle_axis(
            torch.tensor([j0 - a_tgt0], device=self.adapter.device),
            torch.tensor([[0.0, 0.0, 1.0]], device=self.adapter.device))[0]
        quat_co = math_utils.quat_mul(rz.unsqueeze(0), self._via_G.quat_w.unsqueeze(0))[0]
        target = PoseState(P, quat_co)
        # 关节1关节空间带动(限速)
        a_tgt = float(self.faced_q[0])
        max_da = self.cfg.arc_rate * max(dt, 1e-3)
        self._arc_j1 += max(-max_da, min(max_da, a_tgt - self._arc_j1))
        cmd = step_pose(tcp, target, self._eff["max_pos_step"], self._eff["max_ori_step"])
        ik = self.adapter.solve(cmd, joint0_cmd=self._arc_j1)
        if os.environ.get("DRAWER_DBG"):
            self._dbgc = getattr(self, "_dbgc", 0) + 1
            if self._dbgc % 20 == 1:
                j1 = math.degrees(float(state.robot.joint_pos[self.adapter._joint_ids][0]))
                perr = float(torch.linalg.norm(tcp.pos_w - P))
                print(f"[ViaBlendDBG] s={s:.2f} joint1={j1:.1f}deg tcp_vs_bezier={perr*100:.1f}cm", flush=True)
        faced = abs(a_tgt - float(state.robot.joint_pos[self.adapter._joint_ids][0])) <= self.cfg.arc_face_tol
        done = (s >= 0.999 and faced)
        self._last_phase_target = target
        if not ik.success:
            q = self.last_q if self.last_q is not None else state.robot.joint_pos[self.adapter._joint_ids].clone()
            return SkillCommand(tcp, 1.0, self.status, control_mode="joint", joint_target=q,
                                drawer_joint_target=None), done
        self.last_q = ik.q_des
        return SkillCommand(cmd, 1.0, self.status, control_mode="joint", joint_target=ik.q_des,
                            drawer_joint_target=None), done

    def _straighten_grasp_quat(self, quat: torch.Tensor) -> torch.Tensor:
        """把抓取朝向【摆平】：接近轴(+Z)投影到水平面(抽屉是水平接近/滑动的),+Y 保持朝上(手指跨在把手杆
        上下),重建正交朝向。用户面板里若无意把把手 pose 设得上下倾,这里自动纠正成水平抓取,避免斜着抓/斜着
        拉。z 本来就近竖直(投影退化)时保持原样不动。"""
        z = math_utils.quat_apply(quat.reshape(1, 4),
                                  torch.tensor([[0.0, 0.0, 1.0]], device=quat.device))[0].clone()
        z[2] = 0.0
        n = float(torch.linalg.norm(z))
        if n < 1e-4:
            return quat
        z = z / n
        up = torch.tensor([0.0, 0.0, 1.0], device=quat.device)
        x = torch.linalg.cross(up, z); nx = float(torch.linalg.norm(x))
        if nx < 1e-6:
            return quat
        x = x / nx
        y = torch.linalg.cross(z, x)
        R = torch.stack((x, y, z), dim=1)
        return math_utils.quat_from_matrix(R.unsqueeze(0))[0]

    def _lead_dir(self, quat: torch.Tensor, horizontal: bool = False) -> torch.Tensor:
        """站位/后拉的偏移方向 = 抓取位姿的【接近轴 -Z】(TCP +Z 朝抽屉里 -> -Z 朝外)。跟随用户在面板里
        最新设的 pose 朝向:改了 pose 的 z 轴,站位/后拉方向立刻跟着变。

        ``horizontal=True``: 把该方向的【竖直分量清零并重新归一化】,得到纯水平方向。用于【后拉/推入】——
        抽屉是水平滑出的,如果用户设的抓取 pose 的 z 轴有点上/下倾,-Z 就带竖直分量,后拉时 TCP 会沿 Z 抬升/
        下沉,把抽屉拉歪、夹爪从水平把手上滑脱(用户报的 bug)。站位/接近仍按用户角度(非水平),只有拉/推走水平。"""
        az = math_utils.quat_apply(
            quat.reshape(1, 4), torch.tensor([[0.0, 0.0, 1.0]], device=quat.device))[0]
        d = -az
        if horizontal:
            d = d.clone()
            d[2] = 0.0
        return d / (torch.linalg.norm(d) + 1e-9)

    def _grasp_pose(self, lead: float, horizontal_lead: bool = False) -> PoseState:
        """Live grasp/pull target: (handle link-local point + grasp_offset_local) -> world, offset by
        ``lead`` along the GRASP pose approach axis (-Z), so the pre-grasp standoff / pull direction
        track the user's latest handle pose (not the fixed cabinet open direction).

        ``grasp_offset_local_xyz`` is added in the drawer LINK local frame (combined with the base
        handle offset BEFORE the link transform), so it tracks the sliding drawer and yields a real
        edge/slip grasp -- never a fixed world offset (task spec section 5.4).
        """
        open_dir = open_direction_world(self._cabinet_quat())
        eid = self.adapter.env_id
        cab = self.obs_adapter.cabinet
        li = self.obs_adapter._link_idx
        link_pos = cab.data.body_pos_w[eid, li]
        link_quat = cab.data.body_quat_w[eid, li]
        ov = (self.request.parameters or {}).get("override_grasp_local")
        if ov is not None:
            base_local = torch.tensor(ov["pos"], dtype=torch.float32, device=link_pos.device)
            lq = torch.tensor(ov["quat"], dtype=torch.float32, device=link_pos.device)
            total_local = base_local + self._grasp_offset_local
            gpos, gquat = math_utils.combine_frame_transforms(
                link_pos.unsqueeze(0), link_quat.unsqueeze(0), total_local.unsqueeze(0), lq.unsqueeze(0))
            gq = self._straighten_grasp_quat(gquat[0])   # 摆平用户 pose 的上下倾 -> 水平抓取
            return PoseState(gpos[0] + self._lead_dir(gq, horizontal_lead) * lead, gq)
        total_local = self.obs_adapter.handle_offset + self._grasp_offset_local
        gpos, _ = math_utils.combine_frame_transforms(
            link_pos.unsqueeze(0), link_quat.unsqueeze(0), total_local.unsqueeze(0))
        quat = self._straighten_grasp_quat(grasp_quat_from_open_dir(open_dir, link_pos.device))
        return PoseState(gpos[0] + self._lead_dir(quat, horizontal_lead) * lead, quat)

    # ---- skill API --------------------------------------------------------
    def start(self, state: SceneState):
        cfg = DRAWER_TARGETS.get(self.target_drawer)
        self.status = ExecutionStatus.RUNNING
        self.failure_reason = FailureReason.NONE
        if cfg is None:
            self._fail(state, FailureReason.REQUEST_INVALID, f"unknown target_drawer '{self.target_drawer}'")
            return
        self.obs_adapter = SelectedDrawerObsAdapter(self.env, self.target_drawer, env_id=self.adapter.env_id)
        if not self._resolve_params():
            self._fail(state, FailureReason.INVALID_EXPERIMENT_PARAMETER, self._param_error or "bad parameter")
            return
        robot = self.env.unwrapped.scene["robot"]
        self.home_q = torch.tensor(HOME_Q_VERTICAL_RAISED, dtype=torch.float32,
                                   device=robot.data.default_joint_pos.device)
        self.faced_q = self._faced_seed_q()
        pz = (self.request.parameters or {}).get("prelift_z")
        if pz is not None:
            try:
                self._prelift_z = float(pz)
            except Exception:
                self._prelift_z = None
        # cuRobo 已把机器人无碰撞送到把手前(pre-grasp)时,请求带 start_at_approach -> 直接从 APPROACH 起,
        # 跳过 PRELIFT/转身/MOVE_TO_PRE_GRASP(否则会再抬起/重定位、把 cuRobo 的落点破坏掉)。
        if (self.request.parameters or {}).get("start_at_approach"):
            initial_state = "APPROACH"
        elif (self.request.parameters or {}).get("start_at_pregrasp"):
            # 途径点(含 facing joint 点)已完成抬升+转身过渡 -> 跳过内置 PRELIFT/ARC,直接径向接近把手。
            initial_state = "MOVE_TO_PRE_GRASP"
        elif self._prelift_z is not None and self.cfg.start_from_current:
            initial_state = "PRELIFT"
        elif self.cfg.use_via_blend and self.cfg.start_from_current:
            initial_state = "VIA_BLEND"
        elif self.cfg.use_arc_to_face and self.cfg.start_from_current:
            initial_state = "ARC_TO_FACE"
        elif self.cfg.use_turn_to_face:
            initial_state = "TURN_TO_FACE"
        elif self.cfg.start_from_current:
            initial_state = "MOVE_TO_PRE_GRASP"
        else:
            initial_state = "MOVE_TO_HOME"
        self.runtime = _Runtime(
            state=initial_state,
            start_time=state.sim_time,
            state_start_time=state.sim_time,
            drawer_joint_name=cfg["joint_name"],
            last_command_pose=state.robot.tcp_pose,
        )
        self.runtime.initial_joint_pos = self._drawer_pos()
        self.runtime.current_joint_pos = self.runtime.initial_joint_pos
        self.initial_handle_pos = self._handle_pos().clone()
        self._last_progress_time = state.sim_time
        self._progress_ref = self.runtime.initial_joint_pos
        print(f"[OpenDrawerIKSkill] target_open_position={self.target_open_position:.3f} "
              f"initial_drawer={self.runtime.initial_joint_pos:.4f} "
              f"grasp_offset_local={[round(float(v),4) for v in self._grasp_offset_local.tolist()]}", flush=True)
        self._record(state, "IDLE", initial_state)

    def step(self, state: SceneState, dt: float) -> SkillCommand:
        if self.status == ExecutionStatus.IDLE:
            self.start(state)
        if self.status in (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.STOPPED):
            return self._hold(state)

        self.runtime.current_joint_pos = self._drawer_pos()
        gripper = 1.0
        target = None

        if self.runtime.state == "MOVE_TO_HOME":
            return self._home_step(state)
        if self.runtime.state == "TURN_TO_FACE":
            arm_q = state.robot.joint_pos[self.adapter._joint_ids]
            dq1 = abs(float(arm_q[0] - self.faced_q[0]))
            if dq1 <= self.cfg.face_joint_threshold:
                self.runtime.stable_count += 1
                if self.runtime.stable_count >= 3:
                    self._transition(state, "MOVE_TO_PRE_GRASP")
            else:
                self.runtime.stable_count = 0
            if self._state_elapsed(state) > self.cfg.face_timeout:
                self._transition(state, "MOVE_TO_PRE_GRASP")
            self.last_q = self.faced_q.clone()
            self._update_telemetry(state, state.robot.tcp_pose, 1.0)
            return SkillCommand(
                state.robot.tcp_pose, 1.0, self.status, control_mode="joint",
                joint_target=self.faced_q.clone(), drawer_joint_target=None,
            )
        if self.runtime.state == "VIA_BLEND":
            cmd_via, done = self._via_blend_command(state, dt)
            if done or self._state_elapsed(state) > self.cfg.via_timeout:
                self._transition(state, "APPROACH")
            self._update_telemetry(state, self._last_phase_target, 1.0)
            return cmd_via
        if self.runtime.state == "ARC_TO_FACE":
            # 关节空间平滑转身(丝滑版 turn-to-face，无离散停顿)：在【关节空间】朝 faced_q(关节1=把手方位角
            # + 竖直抬起的 reach 起始姿态)逐帧限速插值。关节1的角度差最大 -> 它主导整段运动=关节1优先旋转；
            # TCP 随之绕基座画出平滑圆弧。用关节命令才真能转底座(TCP 目标法会被局部 DLS 用仰身规避)；到 faced_q
            # 的姿态已知可达，reach 不会再有大姿态误差。关节1接近到位(不必精确到达)即【提前融入】reach。
            if self._arc_hold_q is None:
                self._arc_hold_q = state.robot.joint_pos[self.adapter._joint_ids].clone()
            max_step = self.cfg.arc_rate * max(dt, 1e-3)             # 每关节每帧角度上限(关节1差最大->最慢->主导)
            delta = torch.clamp(self.faced_q - self._arc_hold_q, -max_step, max_step)
            self._arc_hold_q = self._arc_hold_q + delta
            q_cmd = self._arc_hold_q.clone()
            self.last_q = q_cmd
            dj1 = abs(float(self.faced_q[0]) - float(state.robot.joint_pos[self.adapter._joint_ids][0]))
            if dj1 <= self.cfg.arc_face_tol or self._state_elapsed(state) > self.cfg.arc_timeout:
                self._transition(state, "MOVE_TO_PRE_GRASP")
            self._update_telemetry(state, state.robot.tcp_pose, 1.0)
            return SkillCommand(
                state.robot.tcp_pose, 1.0, self.status, control_mode="joint",
                joint_target=q_cmd, drawer_joint_target=None,
            )
        if self.runtime.state == "PRELIFT":
            # Bug1: 在当前 XY 竖直抬到 home 高度(prelift_z)，张爪，再去把手前方，避免夹爪斜扫物体。
            if self._prelift_target is None:
                cur = state.robot.tcp_pose
                p = cur.pos_w.clone()
                p[2] = max(float(p[2]), float(self._prelift_z))
                self._prelift_target = PoseState(p, cur.quat_w.clone())
            target = self._prelift_target
            gripper = 1.0
            if abs(float(state.robot.tcp_pose.pos_w[2] - self._prelift_target.pos_w[2])) < 0.03 \
                    or self._state_elapsed(state) > 4.0:
                self._transition(state, self._post_prelift_state())
        elif self.runtime.state == "MOVE_TO_PRE_GRASP":
            target = self._grasp_pose(self._eff["pre_grasp_clearance"])
            gripper = 1.0
            self._advance_when_reached(state, target, "APPROACH", self._eff["reach_timeout"])
        elif self.runtime.state == "APPROACH":
            gp = self._grasp_pose(0.0)                         # LIVE handle every step (no fixed target)
            if self._approach_start is None:
                self._approach_start = state.robot.tcp_pose.pos_w.clone()   # freeze ONLY the line origin
            self._approach_end = gp.pos_w                      # live endpoint tracks the handle
            self._approach_quat = gp.quat_w
            carrot = self._line_setpoint(state.robot.tcp_pose.pos_w, self._approach_start, gp.pos_w)
            target = PoseState(carrot, gp.quat_w)              # straight glide, but toward the LIVE handle
            gripper = 1.0
            self._advance_when_reached(state, gp, "CLOSE_GRIPPER", self._eff["reach_timeout"])
        elif self.runtime.state == "CLOSE_GRIPPER":
            target = self._grasp_pose(0.0)
            gripper = -1.0
            if self._state_elapsed(state) >= self._eff["close_duration"]:
                self._last_progress_time = state.sim_time
                self._progress_ref = self.runtime.current_joint_pos
                self._transition(state, "PULL")
        elif self.runtime.state == "PULL":
            target = self._grasp_pose(self._eff["pull_lead"], horizontal_lead=True)  # 水平后拉,不沿 Z 抬升
            gripper = -1.0
            self._track_pull(state)
            if self.runtime.current_joint_pos >= self.target_open_position - self.cfg.target_tolerance:
                if self.runtime.target_reached_time is None:
                    self.runtime.target_reached_time = max(0.0, state.sim_time - self.runtime.start_time)
                # 开到位的瞬间锁定【当前真实 TCP 位姿】，SETTLE/RELEASE 全程保持它不动。
                # 不要用 _grasp_pose(0.0)(那是把手当前位置，比 TCP 当前位置往把手方向缩了 pull_lead，
                # 会让机器人主动往回拖一小段、把抽屉带回去)。用户要求：开到位后【原地停 0.5s 再松爪】。
                self._settle_target = PoseState(
                    state.robot.tcp_pose.pos_w.clone(), state.robot.tcp_pose.quat_w.clone())
                self._transition(state, "SETTLE")
            elif self.runtime.handle_detached:
                self._fail(state, FailureReason.HANDLE_DETACHED,
                           f"handle lost during pull at drawer={self.runtime.current_joint_pos:.4f}")
            elif self._state_elapsed(state) > self._eff["pull_timeout"]:
                self.runtime.timed_out = True
                self._fail(state, FailureReason.DRAWER_OPEN_TIMEOUT,
                           f"pull did not reach target {self.target_open_position:.3f}: "
                           f"pos={self.runtime.current_joint_pos:.4f}")
        elif self.runtime.state == "SETTLE":
            # 开到位后【原地保持】锁定的 TCP 位姿(夹爪仍闭合)，停 settle_duration(0.5s)，不回拖。
            target = self._settle_target if self._settle_target is not None else self._grasp_pose(0.0)
            gripper = -1.0
            if self._state_elapsed(state) >= self._eff["settle_duration"]:
                self._transition(state, "RELEASE")
        elif self.runtime.state == "RELEASE":
            # 松爪时仍【保持 SETTLE 锁定的固定位姿】：否则张爪瞬间把抽屉带回一点，TCP 跟着把手往里走，
            # 形成"机器人拖着抽屉一起退回去"的反馈环。锁定位姿不动，张开夹爪后抽屉无外力(stiffness0+
            # 阻尼)留在原位，不会被拖回。
            if self._release_target is None:
                base = self._settle_target if self._settle_target is not None else self._grasp_pose(0.0)
                self._release_target = PoseState(base.pos_w.clone(), base.quat_w.clone())
            target = self._release_target
            gripper = 1.0
            if self._state_elapsed(state) >= self._eff["release_duration"]:
                self._succeed(state)

        self._last_phase_target = target
        cmd = self._command(state, target, gripper)
        self._update_telemetry(state, target if target is not None else state.robot.tcp_pose, gripper)
        return cmd

    # ---- pull tracking / detachment --------------------------------------
    def _track_pull(self, state: SceneState):
        handle_w = self._handle_pos()
        tcp = state.robot.tcp_pose.pos_w
        handle_err = float(torch.linalg.norm(tcp - handle_w))
        self.runtime.handle_err_max = max(self.runtime.handle_err_max, handle_err)
        self.runtime.handle_err_sum += handle_err
        self.runtime.handle_err_n += 1
        # drawer progress watchdog
        if self.runtime.current_joint_pos >= self._progress_ref + self.cfg.handle_detach_progress_eps:
            self._progress_ref = self.runtime.current_joint_pos
            self._last_progress_time = state.sim_time
        no_progress = (state.sim_time - self._last_progress_time) > self.cfg.handle_detach_no_progress_window
        not_reached = self.runtime.current_joint_pos < (self.target_open_position - self.cfg.target_tolerance)
        if handle_err > self.cfg.handle_detach_error and no_progress and not_reached:
            self.runtime.handle_detached = True

    def viz_poses(self) -> list:
        out = []
        tgt = getattr(self, "_last_phase_target", None)
        if tgt is not None:
            out.append(("phase_target", tgt))
        try:
            handle = self._handle_pos()
            quat = grasp_quat_from_open_dir(open_direction_world(self._cabinet_quat()), handle.device)
            out.append(("drawer_handle", PoseState(handle, quat)))
        except Exception:
            pass
        return out

    def cancel(self, state: SceneState) -> SkillCommand:
        self.status = ExecutionStatus.STOPPED
        self.failure_reason = FailureReason.CANCELLED_BY_USER
        self._transition(state, "CANCELLED")
        return self._hold(state)

    # ---- outcomes / result -----------------------------------------------
    def outcomes(self, state: SceneState) -> dict:
        final = self._drawer_pos() if self.obs_adapter is not None else self.runtime.current_joint_pos
        mean_track = (self.runtime.track_err_sum / self.runtime.track_err_n) if self.runtime.track_err_n else None
        mean_handle = (self.runtime.handle_err_sum / self.runtime.handle_err_n) if self.runtime.handle_err_n else None
        return {
            "elapsed_time": max(0.0, state.sim_time - self.runtime.start_time),
            "initial_drawer_position": self.runtime.initial_joint_pos,
            "target_drawer_position": self.target_open_position,
            "final_drawer_position": final,
            "drawer_position_error": abs(final - self.target_open_position),
            "drawer_overshoot": final - self.target_open_position,
            "maximum_tcp_tracking_error": self.runtime.track_err_max if self.runtime.track_err_n else None,
            "mean_tcp_tracking_error": mean_track,
            "maximum_handle_relative_error": self.runtime.handle_err_max if self.runtime.handle_err_n else None,
            "mean_handle_relative_error": mean_handle,
            "handle_detached": self.runtime.handle_detached,
            "target_reached_time": self.runtime.target_reached_time,
            "timeout": self.runtime.timed_out,
            "success": self.status == ExecutionStatus.SUCCEEDED,
            "failure_reason": self.failure_reason.value or None,
            "contact_available": False,
            "maximum_contact_force": None,
            "collision_available": False,
            "collision_count": None,
        }

    def result(self, state: SceneState) -> SkillResult:
        return SkillResult(
            request_id=self.request.request_id,
            skill_type=self.request.skill_type,
            target_name=self.target_drawer,
            success=self.status == ExecutionStatus.SUCCEEDED,
            final_status=self.status,
            failure_reason=self.failure_reason.value or None,
            elapsed_time=max(0.0, state.sim_time - self.runtime.start_time),
            final_tcp_pose=pose_tensor(state.robot.tcp_pose),
            position_error=self.runtime.final_error_pos,
            orientation_error=self.runtime.final_error_ori,
            gripper_width=state.robot.gripper_width,
            state_history=self.runtime.history,
            outcomes=self.outcomes(state),
            task_target={"target_open_position": self.target_open_position, "drawer_name": self.target_drawer},
            requested_parameters=self.resolver.requested_subset(),
            effective_parameters=self.resolver.trace_dicts(),
        )

    # ---- helpers ----------------------------------------------------------
    def _line_setpoint(self, c: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        d = b - a
        L = float(torch.linalg.norm(d))
        if L < 1e-6:
            return b.clone()
        u = d / L
        t = float(torch.clamp(torch.dot(c - a, u), 0.0, L))
        s = min(t + self._eff["approach_line_lead"], L)
        return a + s * u

    def _home_step(self, state: SceneState) -> SkillCommand:
        arm_q = state.robot.joint_pos[self.adapter._joint_ids]
        dist = float(torch.linalg.norm(arm_q - self.home_q))
        if dist <= self.cfg.home_joint_threshold:
            self.runtime.stable_count += 1
            if self.runtime.stable_count >= 3:
                self._transition(state, "MOVE_TO_PRE_GRASP")
        else:
            self.runtime.stable_count = 0
        if self._state_elapsed(state) > self.cfg.home_timeout:
            self._transition(state, "MOVE_TO_PRE_GRASP")
        self.last_q = self.home_q.clone()
        return SkillCommand(
            state.robot.tcp_pose, 1.0, self.status, control_mode="joint",
            joint_target=self.home_q.clone(), drawer_joint_target=None,
        )

    # reach 阶段(抬升+接近把手)启用关节1优先旋转的加权 IK；抓住后(CLOSE/PULL/SETTLE/RELEASE)用普通 IK，
    # 避免拉抽屉时关节1还在漂、扭动抓持。
    _FACE_REACH_PHASES = ("PRELIFT", "MOVE_TO_PRE_GRASP", "APPROACH")

    def _reach_joint_weights(self):
        w = float(getattr(self.cfg, "joint1_face_weight", 0.0) or 0.0)
        if w <= 0.0 or self.runtime.state not in self._FACE_REACH_PHASES:
            return None
        if getattr(self, "_jw_cache", None) is None:
            jw = torch.ones(len(self.adapter._joint_ids), device=self.adapter.device)
            jw[0] = w
            self._jw_cache = jw
        return self._jw_cache

    def _command(self, state: SceneState, target: PoseState | None, gripper: float) -> SkillCommand:
        if target is None:
            return self._hold(state, gripper)
        cmd = step_pose(state.robot.tcp_pose, target, self._eff["max_pos_step"], self._eff["max_ori_step"])
        ik = self.adapter.solve(cmd, joint_weights=self._reach_joint_weights())
        if os.environ.get("DRAWER_DBG"):
            self._dbgc = getattr(self, "_dbgc", 0) + 1
            if self._dbgc % 25 == 1:
                j1 = float(state.robot.joint_pos[self.adapter._joint_ids][0])
                print(f"[OpenDrawerDBG] {self.runtime.state} joint1={math.degrees(j1):.1f}deg "
                      f"weighted={self._reach_joint_weights() is not None}", flush=True)
        err = pose_error(state.robot.tcp_pose, target)
        self.runtime.final_error_pos = err.position
        self.runtime.final_error_ori = err.orientation
        self.runtime.last_command_pose = cmd
        # TCP tracking-error stats over the reaching/pulling phases
        if self.runtime.state in ("MOVE_TO_PRE_GRASP", "APPROACH", "PULL"):
            self.runtime.track_err_max = max(self.runtime.track_err_max, err.position)
            self.runtime.track_err_sum += err.position
            self.runtime.track_err_n += 1
        if not ik.success:
            q = self.last_q if self.last_q is not None else state.robot.joint_pos[self.adapter._joint_ids].clone()
            return SkillCommand(state.robot.tcp_pose, gripper, self.status, control_mode="joint", joint_target=q,
                                drawer_joint_target=None)
        self.last_q = ik.q_des
        return SkillCommand(cmd, gripper, self.status, control_mode="joint", joint_target=ik.q_des, drawer_joint_target=None)

    def _hold(self, state: SceneState, gripper: float = -1.0) -> SkillCommand:
        q = self.last_q if self.last_q is not None else state.robot.joint_pos[self.adapter._joint_ids].clone()
        return SkillCommand(state.robot.tcp_pose, gripper, self.status, control_mode="joint", joint_target=q,
                            drawer_joint_target=None)

    def _advance_when_reached(self, state: SceneState, target: PoseState, next_state: str, timeout: float) -> bool:
        err = pose_error(state.robot.tcp_pose, target)
        if err.position <= self.cfg.reach_pos_threshold and err.orientation <= self.cfg.reach_ori_threshold:
            self.runtime.stable_count += 1
            if self.runtime.stable_count >= self.cfg.reach_stable_cycles:
                self._transition(state, next_state)
                return True
        else:
            self.runtime.stable_count = 0
        if self._state_elapsed(state) > timeout:
            if err.position <= self.cfg.soft_reach_advance and err.orientation <= self.cfg.soft_reach_ori:
                print(f"[OpenDrawerIKSkill] soft-advance {self.runtime.state}->{next_state} "
                      f"(pos_err={err.position:.3f} ori_err={math.degrees(err.orientation):.1f}deg)", flush=True)
                self._transition(state, next_state)
                return True
            self._fail(state, FailureReason.POSITION_TIMEOUT,
                       f"{self.runtime.state} reach timeout (pos={err.position:.3f} "
                       f"ori={math.degrees(err.orientation):.1f}deg)")
        return False

    def _state_elapsed(self, state: SceneState) -> float:
        return max(0.0, state.sim_time - self.runtime.state_start_time)

    def _transition(self, state: SceneState, new_state: str):
        old = self.runtime.state
        if old == new_state:
            return
        self.runtime.state = new_state
        self.runtime.state_start_time = state.sim_time
        self.runtime.stable_count = 0
        self._record(state, old, new_state)

    def _update_telemetry(self, state: SceneState, target: PoseState, gripper: float):
        err = pose_error(state.robot.tcp_pose, target)
        handle_w = None
        handle_rel = float("nan")
        try:
            handle_w = self._handle_pos()
            handle_rel = float(torch.linalg.norm(state.robot.tcp_pose.pos_w - handle_w))
        except Exception:
            pass
        self.last_telemetry = {
            "skill_state": self.runtime.state,
            "target_tcp_position": [float(v) for v in target.pos_w.tolist()],
            "target_tcp_orientation": [float(v) for v in target.quat_w.tolist()],
            "tcp_position_error": err.position,
            "tcp_orientation_error": err.orientation,
            "gripper_command": float(gripper),
            "drawer_joint_position": float(self.runtime.current_joint_pos),
            "handle_position": None if handle_w is None else [float(v) for v in handle_w.tolist()],
            "target_handle_grasp_position": [float(v) for v in target.pos_w.tolist()],
            "handle_relative_position_error": handle_rel,
            "drawer_progress": float(self.runtime.current_joint_pos - self.runtime.initial_joint_pos),
        }

    def _record(self, state: SceneState, old: str, new: str):
        rec = {
            "time": round(state.sim_time, 4),
            "skill": self.request.skill_type.value,
            "backend": self.backend,
            "target_drawer": self.target_drawer,
            "drawer_joint_name": self.runtime.drawer_joint_name,
            "from": old,
            "to": new,
            "drawer_joint_pos": round(self.runtime.current_joint_pos, 5),
            "target_open_position": round(self.target_open_position, 5),
            "drawer_joint_target": None,
            "failure_reason": self.failure_reason.value or None,
            "failure_message": self.runtime.last_failure_message,
        }
        self.runtime.history.append(rec)
        print(f"[OpenDrawerIKSkill] {rec}", flush=True)

    def _fail(self, state: SceneState, reason: FailureReason, message: str):
        if self.status == ExecutionStatus.FAILED:
            return
        self.status = ExecutionStatus.FAILED
        self.failure_reason = reason
        self.runtime.last_failure_message = message
        self._transition(state, "FAILED")

    def _succeed(self, state: SceneState):
        self.status = ExecutionStatus.SUCCEEDED
        self.failure_reason = FailureReason.NONE
        self._transition(state, "SUCCEEDED")
        print(f"[OpenDrawerIKSkill] success target={self.target_drawer} drawer_pos={self.runtime.current_joint_pos:.4f}", flush=True)
