"""External test controller that plugs the joint-action skill state machine into the V1 test scene.

Loaded by ``scene_interface/test_mode_ui.py`` via ``--controller
state_machine.skill_test_controller:SkillTestController``. The test-mode entry instantiates this as
``SkillTestController(session)``, optionally calls ``build_window()`` once, ``on_reset()`` on layout
reset, and ``step(session)`` every frame. ``step`` returns a full env action tensor (built from the
skill command) to drive the robot, or ``None`` to hand control back to the UI's manual IK / hold.

PURE-PHYSICAL CONTRACT (mirrors the standalone skill entries): no skill ever commands the door/drawer
joint. The door skill frees its own hinge at runtime; the drawer ik_pull skill does NOT, so this
controller temporarily frees the targeted cabinet drawer joint (stiffness 0 + low damping) when a
drawer skill starts. This is a RUNTIME, IN-MEMORY change to the actuator gain only -- it touches
nothing else in the scene and is never saved (no scene write). With stiffness 0 the per-frame
position target that the test scene's Joint Driver panel pushes becomes inert, so the panel cannot
"hold" the joint against the gripper.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

# Legacy skill-state-machine root is already on sys.path by the time test_mode_ui instantiates us
# (SceneSession._build / test_mode_ui both call ensure_legacy_on_path before this import).
from runtime.collision_monitor import CollisionMonitor
from runtime.drawer_target_config import joint_name_for, member_for, functional_drawers, SEKTION_DRAWERS
from runtime.microwave_door_config import COFFEE_LEVER_JOINT, COFFEE_LEVER_LINK
from runtime.ik_joint_adapter import IKJointAdapter
from runtime.base_skill import set_speed_scale
from runtime.scene_state_provider import PoseState
from runtime.skill_request import SkillRequest
from runtime.target_registry import TargetRegistry
from state_machine.skill_executor import JointBackendConfig, SkillExecutor
from skills.open_drawer_skill import OpenDrawerIKConfig
from skills.close_drawer_skill import CloseDrawerIKConfig
from runtime.skill_types import SkillType, ExecutionStatus

# UI choices (ASCII only -- omni.ui cannot render CJK).
_SKILLS = [
    ("Open Door", SkillType.OPEN_DOOR),
    ("Close Door", SkillType.CLOSE_DOOR),
    ("Open Drawer", SkillType.OPEN_DRAWER),
    ("Close Drawer", SkillType.CLOSE_DRAWER),
    ("Grasp", SkillType.GRASP),
    ("Place", SkillType.PLACE),
]
_GRASP_TARGETS = ["cube_1", "cube_2", "cube_3", "knife"]
# 所有可单独选中操作的抽屉：旧柜 Cabinet_44853(top/middle/bottom) + 右下角白柜 Sektion(top/bottom)。
# 选中任一个后用 Open Drawer / Close Drawer 单独开/关。
_DRAWER_TARGETS = ["middle_drawer", "top_drawer", "bottom_drawer",
                   "sektion_top_drawer", "sektion_bottom_drawer"]
# 抽屉关节阻尼随技能阶段动态切换(stiffness 始终 0，无弹簧)：
#  * PULL/PUSH(机器人主动拖动抽屉)时用【低阻尼】，否则跟手发滞、抽屉跟不上夹爪 -> 报 HANDLE_DETACHED。
#  * 其余阶段(approach/settle/release/空闲)用【高阻尼】，否则松爪/退回时夹爪会把抽屉一起带回去
#    (实测 stiffness0+低阻尼时抽屉本身无外力会停住，但 RELEASE 阶段机器人拖动会把它拽回 0)。
_DRAWER_FREE_DAMPING = 3.0     # 拉/推阶段：能被机器人拉动
_DRAWER_HOLD_DAMPING = 100.0    # 其余阶段：抵抗松爪/退回时的拖拽，抽屉停在开到的位置
# Deployed scene replaces the microwave with the fridge (member still keyed "microwave"); the door
# skill drives the fridge's link_1/joint_1 via the "fridge" door target. "microwave" kept for the old
# asset (--no_fridge).
_DOOR_TARGETS = ["fridge", "microwave"]
# Default place point (env-local xyz). Refine per task flow later.
_DEFAULT_PLACE_XYZ = (0.45, 0.0, 0.06)

# Sorting baskets (KLT bins) -> place target. Label includes the category the user assigned:
#   fruits -> KLT_3 (left), tools/boxes -> KLT_1 (middle), cups -> KLT_2 (right).
# The member name is the scene member read live for the basket world position.
_BASKET_TARGETS = [
    ("KLT_3 (fruits)", "Prop_KLT_3"),
    ("KLT_1 (tools/boxes)", "Prop_KLT_1"),
    ("KLT_2 (cups)", "Prop_KLT_2"),
]

_SKILL_BY_NAME = {s.value: s for _, s in _SKILLS}

# Autorun: env var SKILL_TEST_AUTORUN="open_door:microwave,close_door:microwave,..." runs the listed
# skills in order with no clicks (for headless / unattended verification). Each item is
# "<skill_value>:<target>"; target optional for skills that don't need one.
_AUTORUN_SETTLE_STEPS = 40       # hold steps between skills so physics settles
_AUTORUN_MAX_STEPS = 3000        # ~60s @ 50Hz per skill; abort as TIMEOUT past this


def _parse_sort(spec: str | None):
    """SKILL_TEST_SORT='Prop_011_banana:Prop_KLT_3,Prop_lemon_01:Prop_KLT_3' -> [(obj, basket), ...].

    Headless sorting smoke test: for each pair, grasp the object (open-loop _GraspPoseRunner on its
    saved grasp pose) then place it into the basket (_PlacePoseRunner). Drives the same code the GUI
    buttons use, so it validates the full pick-and-place without clicks.
    """
    if not spec:
        return None
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        obj, _, basket = item.partition(":")
        if obj.strip() and basket.strip():
            out.append((obj.strip(), basket.strip()))
    return out or None


def _parse_autorun(spec: str | None):
    if not spec:
        return None
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, target = item.partition(":")
        st = _SKILL_BY_NAME.get(name.strip())
        if st is None:
            print(f"[SkillTestController] WARN: unknown autorun skill '{name}'", flush=True)
            continue
        out.append((st, target.strip() or None))
    return out or None


class _GraspPoseRunner:
    """把机械臂驱动到一个【任意世界抓取位姿】并合爪，沿【抓取位姿自己的 Z 轴(接近轴)+ 直线笛卡尔】接近/回退：

        standoff: 先退到目标【沿抓取位姿 -Z(接近轴反向)】的 standoff 点，【位置+姿态都对齐】后才进
        descend : 沿直线(standoff->grasp，即沿 +Z 接近轴)笛卡尔前进，姿态恒为抓取朝向
        close   : 在目标 pose 合爪
        retreat : 沿直线(grasp->standoff，沿 -Z)笛卡尔退回

    接近方向 = 抓取位姿的 +Z(TCP 接近轴)在世界系的指向(quat_apply(grasp_quat,[0,0,1]))。所以你把
    抓取 pose 的 Z 设向哪边，机器人就从那个反方向贴着直线进去抓——不是固定从上往下。
    直线靠"跟随胡萝卜"实现：每帧把 IK 目标设成 TCP 在直线上投影再朝前探 lead，TCP 紧贴直线走。
    """

    def __init__(self, adapter, grasp_pose_w: PoseState, *, device, standoff: float = 0.12,
                 lead: float = 0.012, pos_tol: float = 0.006, ori_tol_deg: float = 3.0,
                 speed: float = 1.0, arrival_z=None, retreat: bool = True,
                 sweep_hinge=None, sweep_axis=None, sweep_sign: float = -1.0, sweep_angle: float = 0.0,
                 sweep_rate: float = 0.6, joint_reader=None, joint_target=None):
        import isaaclab.utils.math as mu

        self.adapter = adapter
        self.device = device
        self.retreat = bool(retreat)   # False = 合爪后保持在抓取位姿(不回退)，用于抓住把手等后续操作
        # 合爪后绕【把手的旋转关节轴】rigid 旋转所夹的把手(操作咖啡机杠杆):
        #   sweep_hinge=轴上一点(关节锚点,world), sweep_axis=轴方向(world)。
        #   闭环模式(推荐): 传 joint_reader(读真实关节角)+joint_target -> 先探测旋转方向, 再驱动到目标角,
        #   方向永远正确(不依赖轴符号约定)。开环回退: sweep_sign+sweep_angle。
        self.sweep_hinge = None if sweep_hinge is None else sweep_hinge.reshape(3).to(device).clone()
        if sweep_axis is None:
            self.sweep_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
        else:
            ax = sweep_axis.reshape(3).to(device).float()
            self.sweep_axis = ax / (torch.linalg.norm(ax) + 1e-9)
        self.sweep_sign = float(sweep_sign)
        self.sweep_angle = float(sweep_angle)
        self.sweep_rate = float(sweep_rate)
        self.joint_reader = joint_reader               # callable()->float, 读被操作关节的当前角
        self.joint_target = joint_target               # 目标关节角(闭环)
        self._phi = 0.0          # 已命令的绕轴抓取转角(signed)
        self._jstart = None      # sweep 开始时的关节角
        self._jsign = None       # +phi 对应关节变化的符号(探测得到)
        self._phi_goal = None
        self._swept = 0.0
        self._saved_max_step = None   # sweep 期间临时压低 IK 步长(平滑旋转)，结束恢复
        self._stall_t = 0.0           # 关节失速(手臂到极限拖不动杠杆)累计时长
        self._j_prev = None
        self._grip_pos = None
        self._grip_quat = None
        self.grasp_pos = grasp_pose_w.pos_w.reshape(3).to(device).clone()
        self.grasp_quat = grasp_pose_w.quat_w.reshape(4).to(device).clone()
        # 接近轴 = 抓取位姿 +Z 在世界系；standoff 沿 -Z 退（从接近轴反方向贴近）
        zc = mu.quat_apply(self.grasp_quat.reshape(1, 4), torch.tensor([[0.0, 0.0, 1.0]], device=device))[0]
        zc = zc / torch.linalg.norm(zc)
        self.approach_dir = zc
        self.above_pos = self.grasp_pos - float(standoff) * zc
        # 到达/离开走"高位"：物体正上方、高度 = arrival_z(高于 home 点)。先升到这里再竖直下扎，抓完先升回这里。
        self.arrival_z = None if arrival_z is None else float(arrival_z)
        self.high_pos = None
        if self.arrival_z is not None and self.arrival_z > float(self.grasp_pos[2]):
            self.high_pos = self.grasp_pos.clone(); self.high_pos[2] = self.arrival_z
        # Bug1: 点技能后【先在当前 XY 竖直抬到 home 高度】再水平移到目标上方，避免夹爪斜着扫过物体。
        # lift_pos 在第一帧按【当时 TCP 的 XY】锁定(= 原地直上)，仅当走高位(high_pos 存在)时启用。
        self.lift_pos = None
        # 沿接近轴(Z)直线进退用的"胡萝卜"步长。speed 放大它 -> 沿 Z 接近/回退更快(UI 可调)。
        self.lead = float(lead) * max(0.2, float(speed))
        self.pos_tol = float(pos_tol)
        self.ori_tol = float(ori_tol_deg) * 3.14159265 / 180.0
        self.phase = "lift" if self.high_pos is not None else "standoff"
        self.gripper = 1.0   # 先张开
        self._t = 0.0
        self._total = 0.0
        self._ph_t = 0.0          # 当前 phase 已耗时(检测卡死)
        self._last_phase = self.phase
        self.failed = False
        self.done = False

    def _pos_err(self, c, p) -> float:
        return float(torch.linalg.norm(c - p))

    def _ori_err(self, cq) -> float:
        dot = torch.clamp(torch.abs(torch.dot(cq, self.grasp_quat)), 0.0, 1.0)
        return float(2.0 * torch.acos(dot))   # rad，TCP 当前朝向与抓取朝向夹角

    def _line_setpoint(self, c, a, b):
        """TCP 在直线 a->b 上的投影再朝 b 探 lead，得到下一帧 IK 位置目标（保证贴着直线走）。"""
        d = b - a
        L = float(torch.linalg.norm(d))
        if L < 1e-6:
            return b
        u = d / L
        t = float(torch.clamp(torch.dot(c - a, u), 0.0, L))
        s = min(t + self.lead, L)
        return a + s * u

    def step(self, state, dt):
        """返回 (q_des|None, gripper)。done=True 时表示完成。"""
        c = state.robot.tcp_pose.pos_w.reshape(3).to(self.device)
        cq = state.robot.tcp_pose.quat_w.reshape(4).to(self.device)
        # 卡死检测：某个 phase 超时(IK 够不到/姿态对不齐)就放弃，别无限挂起整条流程。
        self._total += dt
        self._ph_t = self._ph_t + dt if self.phase == self._last_phase else 0.0
        self._last_phase = self.phase
        ph_limit = 7.0 if self.phase in ("lift", "high", "standoff", "approach", "retreat") else 2.0
        if self._ph_t > ph_limit and self.phase not in ("close", "sweep", "done"):
            # 软推进：standoff 超时但已经够近(DLS 在工作空间边缘收敛慢) -> 直接进 approach，让物理贴合，
            # 而不是直接判失败。其它阶段超时才真正放弃。
            if self.phase == "lift":          # 竖直抬升够不到目标高度 -> 直接进 high(别卡死)
                print(f"[GraspPoseRunner] lift soft-advance -> high (z={float(c[2]):.3f})", flush=True)
                self.phase = "high"; self._ph_t = 0.0
            elif self.phase == "standoff" and self._pos_err(c, self.above_pos) < 0.05 and self._ori_err(cq) < 0.26:
                print(f"[GraspPoseRunner] standoff soft-advance (pos_err="
                      f"{self._pos_err(c, self.above_pos)*100:.1f}cm ori_err={self._ori_err(cq)*57.3:.1f}deg)",
                      flush=True)
                self.phase = "approach"; self._ph_t = 0.0
            elif self.phase == "approach" and self._pos_err(c, self.grasp_pos) < 0.025:
                # 已经很接近抓取点(<2.5cm)但收敛慢/到不了 0.8cm 阈值 -> 直接合爪(足够夹住)，别判失败
                print(f"[GraspPoseRunner] approach soft-advance -> close "
                      f"(pos_err={self._pos_err(c, self.grasp_pos)*100:.1f}cm)", flush=True)
                self.phase = "close"; self._t = 0.0; self._ph_t = 0.0
            else:
                print(f"[GraspPoseRunner] STUCK in '{self.phase}' "
                      f"pos_err={self._pos_err(c, self.above_pos if self.phase=='standoff' else self.grasp_pos)*100:.1f}cm "
                      f"ori_err={self._ori_err(cq)*57.3:.1f}deg -> give up", flush=True)
                self.failed = True
                self.done = True
                return None, self.gripper
        if self.phase == "lift":                                 # Bug1: 先在当前 XY 竖直抬到 home 高度
            if self.lift_pos is None:
                self.lift_pos = c.clone(); self.lift_pos[2] = float(self.arrival_z)
            tgt_pos, tgt_quat = self.lift_pos, self.grasp_quat
            if abs(float(c[2] - self.lift_pos[2])) < max(self.pos_tol, 0.03):
                self.phase = "high"                              # 抬到位再水平移到目标上方
        elif self.phase == "high":                               # 先到物体正上方的高位(高于 home 点)
            tgt_pos, tgt_quat = self.high_pos, self.grasp_quat
            if self._pos_err(c, self.high_pos) < max(self.pos_tol, 0.02):
                self.phase = "standoff"
        elif self.phase == "standoff":
            tgt_pos, tgt_quat = self.above_pos, self.grasp_quat   # 到接近轴方向的 standoff 点
            if self._pos_err(c, self.above_pos) < self.pos_tol and self._ori_err(cq) < self.ori_tol:
                self.phase = "approach"                           # 位置+姿态都对齐才沿接近轴前进
        elif self.phase == "approach":
            tgt_pos = self._line_setpoint(c, self.above_pos, self.grasp_pos)   # 沿接近轴直线前进
            tgt_quat = self.grasp_quat
            if self._pos_err(c, self.grasp_pos) < 0.008:
                self.phase = "close"; self._t = 0.0
        elif self.phase == "close":
            tgt_pos, tgt_quat = self.grasp_pos, self.grasp_quat; self.gripper = -1.0; self._t += dt
            if self._t > 0.6:
                closed_loop = self.joint_reader is not None and self.joint_target is not None
                want_sweep = self.sweep_hinge is not None and (closed_loop or abs(self.sweep_angle) > 1e-3)
                if want_sweep:
                    self._grip_pos = c.clone(); self._grip_quat = cq.clone()   # 抓住瞬间的把手位姿
                    self.phase = "sweep"; self._t = 0.0
                    # Bug2: 旋转杠杆时把 IK 每步关节限幅压到很小(0.03 rad)，否则速度滑条把 max_joint_step
                    # 抬到 0.5，rigid 跟踪会让手臂猛跳/甩动("很大问题")。结束(done/timeout)再恢复。
                    try:
                        self._saved_max_step = float(self.adapter.max_joint_step)
                        self.adapter.max_joint_step = min(self._saved_max_step, 0.03)
                    except Exception:
                        self._saved_max_step = None
                    if closed_loop:
                        self._jstart = float(self.joint_reader())
                    rr = float(torch.linalg.norm((self._grip_pos - self.sweep_hinge)[:2]))
                    gw0 = float(getattr(state.robot, "gripper_width", -1.0))
                    print(f"[GraspPoseRunner] grip captured (gw={gw0:.3f}) -> sweep {'(closed-loop to '+format(self.joint_target,'.2f')+')' if closed_loop else 'angle='+format(self.sweep_angle,'.2f')} "
                          f"radius={rr:.3f} axis=({float(self.sweep_axis[0]):.2f},{float(self.sweep_axis[1]):.2f},"
                          f"{float(self.sweep_axis[2]):.2f})", flush=True)
                else:
                    self.phase = "retreat" if self.retreat else "done"
        elif self.phase == "sweep":
            # 绕【把手的关节轴线】rigid 旋转所夹的把手(位置+朝向同转)，物理拖动杠杆。闭环: 探方向后驱动到目标角。
            import isaaclab.utils.math as mu

            self.gripper = -1.0
            self._sweep_total = getattr(self, "_sweep_total", 0.0) + dt
            if self._sweep_total > 14.0:                          # 旋转保险：超时就结束(别无限挂起)
                print(f"[GraspPoseRunner] sweep timeout (phi={self._phi:.2f})", flush=True)
                self.phase = "done"
            if self.joint_reader is not None and self.joint_target is not None:
                cur_j = float(self.joint_reader())
                if self._jsign is None:                          # 探测阶段：+phi 让关节往哪个方向变
                    self._phi += self.sweep_rate * dt
                    if abs(self._phi) >= 0.10:
                        dj = cur_j - self._jstart
                        self._jsign = 1.0 if dj >= 0 else -1.0
                        self._phi_goal = (self.joint_target - self._jstart) / self._jsign
                        print(f"[GraspPoseRunner] sweep dir probed: +phi -> joint {'+' if self._jsign>0 else '-'}; "
                              f"jstart={self._jstart:.2f} goal phi={self._phi_goal:.2f}", flush=True)
                else:                                            # 闭环：夹爪【刚性跟随把手实际转角】+小幅超前施力
                    err = self.joint_target - cur_j              # 关节还差多少
                    # 关键修复(消除夹爪与把手错位/打滑)：夹爪绕锚点的转角 = 把手【当前实际】转角(严格刚性,
                    # 夹爪始终贴在把手上)，再叠加一个【很小的超前量】施加力矩把把手往目标推。旧写法 phi 是自由
                    # P 累加、会冲到 ±1.9 限幅远超把手实际转角 -> 夹爪目标位姿跑到把手前面很远(~0.9rad≈12cm)
                    # -> 中途夹爪与把手明显错位打滑。现在夹爪最多只超前 lead(≈1.4cm),打滑极小。
                    ang_match = self._jsign * (cur_j - self._jstart)   # 与把手当前位姿一致的锚点转角(刚性)
                    lead = 0.10 if abs(err) >= 0.02 else 0.0           # 到位就不再超前(精确贴合、松开前不拽)
                    ang = ang_match + self._jsign * (1.0 if err >= 0 else -1.0) * lead
                    self._phi = max(-1.9, min(1.9, ang))
                    self._dbg = getattr(self, "_dbg", 0) + 1
                    if self._dbg % 20 == 1:
                        gw = float(getattr(state.robot, "gripper_width", -1.0))
                        print(f"[GraspPoseRunner] sweep drive: ang={self._phi:.2f} match={ang_match:.2f} "
                              f"joint={cur_j:.3f}->{self.joint_target:.2f} gw={gw:.3f}", flush=True)
                    # Bug2 失速检测：还在推 phi 但关节几乎不动(手臂/手腕到极限，拖不动杠杆) -> 别一直推到
                    # 限幅甩臂，原地收尾。判定窗口 1.2s、阈值很小，正常转动时每帧关节位移远超阈值。
                    if self._j_prev is not None and abs(cur_j - self._j_prev) < 0.0008:
                        self._stall_t += dt
                    else:
                        self._stall_t = 0.0
                    self._j_prev = cur_j
                    if abs(err) < 0.02:
                        self._t += dt
                        if self._t > 0.4:
                            print(f"[GraspPoseRunner] sweep done: joint={cur_j:.3f} (target {self.joint_target:.2f})",
                                  flush=True)
                            self.phase = "done"
                    else:
                        self._t = 0.0
                        if self._stall_t > 1.2:
                            print(f"[GraspPoseRunner] sweep STALLED at joint={cur_j:.3f} "
                                  f"(arm limit, target {self.joint_target:.2f}) -> stop", flush=True)
                            self.phase = "done"
                ang = self._phi
            else:                                                # 开环回退
                self._swept = min(self.sweep_angle, self._swept + self.sweep_rate * dt)
                ang = self.sweep_sign * self._swept
                if self._swept >= self.sweep_angle - 1e-3:
                    self._t += dt
                    if self._t > 0.5:
                        self.phase = "done"
            rz = mu.quat_from_angle_axis(torch.tensor([ang], device=self.device),
                                         self.sweep_axis.reshape(1, 3))[0]
            r = (self._grip_pos - self.sweep_hinge).reshape(1, 3)   # 关节锚点->把手 的向量, 绕关节轴转
            r_rot = mu.quat_apply(rz.unsqueeze(0), r)[0]
            tgt_pos = self.sweep_hinge + r_rot
            tgt_quat = mu.quat_mul(rz.unsqueeze(0), self._grip_quat.unsqueeze(0))[0]
        elif self.phase == "retreat":
            # 抓完先沿接近轴退回 standoff，再竖直升到高位(高于 home 点)，便于随后高位转移
            retreat_to = self.high_pos if self.high_pos is not None else self.above_pos
            tgt_pos = self._line_setpoint(c, self.grasp_pos, retreat_to)
            tgt_quat = self.grasp_quat; self.gripper = -1.0
            if self._pos_err(c, retreat_to) < 0.03:
                self.phase = "done"
        else:
            if self._saved_max_step is not None:                 # 恢复 sweep 期间压低的 IK 步长
                try:
                    self.adapter.max_joint_step = self._saved_max_step
                except Exception:
                    pass
                self._saved_max_step = None
            self.done = True
            return None, self.gripper
        # (实测：sweep 时开零空间冗余消解反而扰动 rigid 抓持约束、杠杆完全拖不动，故保持 null_gain=0。)
        res = self.adapter.solve(PoseState(tgt_pos, tgt_quat))
        return (res.q_des if getattr(res, "success", False) else None), self.gripper


class _PlacePoseRunner:
    """开环放置：把【夹着的物体】抬起 -> 移到篮子上方 -> 下降到投放高度 -> 松爪让物体掉进篮子。

    与 _GraspPoseRunner 对称：抓取是开环执行器、会 reset 普通 skill 执行器(不记录 held_object)，所以
    放置也用开环执行器(不依赖 held_object)——夹爪【保持闭合】carry 物体过去，只在 release 阶段才张开。
    朝向恒为抓取结束时的 TCP 朝向(物体被怎么夹着就怎么搬)。
    """

    def _zdown_quat(self, cq):
        """放置朝向：TCP +Z 指向【世界 -Z(正下方)】、保留当前水平 yaw(最小化手腕扭转)。"""
        import isaaclab.utils.math as mu

        z = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        cur_x = mu.quat_apply(cq.reshape(1, 4), torch.tensor([[1.0, 0.0, 0.0]], device=self.device))[0]
        xh = cur_x.clone(); xh[2] = 0.0
        if float(torch.linalg.norm(xh)) < 1e-6:
            xh = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        xh = xh / torch.linalg.norm(xh)
        yh = torch.linalg.cross(z, xh)
        R = torch.stack((xh, yh, z), dim=1)
        return mu.quat_from_matrix(R.unsqueeze(0))[0]

    def __init__(self, adapter, basket_pos_w, *, device, lift: float = 0.15,
                 drop_above: float = 0.15, pos_tol: float = 0.03, speed: float = 1.0,
                 carry_z=None, arrival_z=None, transit_via=None, return_to=None):
        self.adapter = adapter
        self.device = device
        self.basket = basket_pos_w.reshape(3).to(device).clone()
        self.lift = float(lift)
        self.drop_above = float(drop_above)
        self.pos_tol = float(pos_tol)
        self.speed = max(0.2, float(speed))
        self.carry_z = None if carry_z is None else float(carry_z)        # 搬运高度(home 高度)
        self.arrival_z = None if arrival_z is None else float(arrival_z)  # 篮子正上方"到达"高度(更高)
        # transit_via / return_to = 显式世界航点(xy 用其值，z 用 carry)：绕开基座圈用。None=不绕/不回位。
        self.transit_via = None if transit_via is None else transit_via.reshape(3).to(device).clone()
        self.return_to = None if return_to is None else return_to.reshape(3).to(device).clone()
        self.phase = "lift"
        self.gripper = -1.0   # 保持夹住，直到 release
        self._cap = False
        self._t = 0.0
        self._total = 0.0
        self.done = False

    def step(self, state, dt):
        c = state.robot.tcp_pose.pos_w.reshape(3).to(self.device)
        cq = state.robot.tcp_pose.quat_w.reshape(4).to(self.device)
        # 安全超时：篮子偏后偏高、可能够不到 -> 别卡死；到点没到位就在当前位张爪投放后结束。
        self._total += dt
        if self._total > 18.0 and self.phase not in ("release", "return", "done"):
            print(f"[PlacePoseRunner] timeout in '{self.phase}' (basket may be unreachable) -> release here",
                  flush=True)
            self._cap = True
            self._quat = self._zdown_quat(cq)
            self._drop = c.clone()   # release at the current pose (don't chase an unreachable basket)
            self.phase = "release"; self._t = 0.0
        if not self._cap:
            self._quat = self._zdown_quat(cq)            # 放置朝向: z 轴朝下(top-down 投放)
            carry_z = self.carry_z if self.carry_z is not None else float(c[2]) + self.lift
            carry_z = max(carry_z, float(self.basket[2]) + self.drop_above)   # 至少高过投放点
            above_z = max(self.arrival_z, carry_z) if self.arrival_z is not None else carry_z
            self._lift_p = c.clone(); self._lift_p[2] = carry_z              # 抬到 carry 高度
            self._above = self.basket.clone(); self._above[2] = above_z      # 篮子上方(更高)
            self._drop = self.basket.clone(); self._drop[2] = float(self.basket[2]) + self.drop_above
            self._via = None
            if self.transit_via is not None:
                self._via = self.transit_via.clone(); self._via[2] = carry_z
            self._ret = None
            if self.return_to is not None:
                self._ret = self.return_to.clone(); self._ret[2] = carry_z
            self._cap = True
        if self.phase == "lift":
            tgt = self._lift_p
            if float(torch.linalg.norm(c - self._lift_p)) < self.pos_tol:
                self.phase = "transit" if self._via is not None else "move"
                if self._via is not None:
                    print(f"[PlacePoseRunner] lift done -> transit AROUND base via "
                          f"({float(self._via[0]):.3f},{float(self._via[1]):.3f})", flush=True)
        elif self.phase == "transit":                    # 绕到基座圈外侧的航点(避开基座轴线), 看 XY 到位
            tgt = self._via
            if float(torch.linalg.norm(c[:2] - self._via[:2])) < max(self.pos_tol, 0.05):
                self.phase = "move"
        elif self.phase == "move":                       # 移到篮子正上方(只看 XY 对齐)
            tgt = self._above
            if float(torch.linalg.norm(c[:2] - self._above[:2])) < self.pos_tol:
                self.phase = "descend"
        elif self.phase == "descend":                    # 下降到投放高度
            tgt = self._drop
            if float(torch.linalg.norm(c - self._drop)) < max(self.pos_tol, 0.04):
                self.phase = "release"; self._t = 0.0
        elif self.phase == "release":                    # 张爪，物体掉进篮子
            tgt = self._drop; self.gripper = 1.0; self._t += dt
            if self._t > 0.6:
                self.phase = "return" if self._ret is not None else "done"
                self._t = 0.0
                if self._ret is not None:
                    print("[PlacePoseRunner] released -> return to staging", flush=True)
        elif self.phase == "return":                     # 回到基座圈外的身前待命点(下一个抓取从高、外侧起步)
            tgt = self._ret; self.gripper = 1.0; self._t += dt
            self._quat = self._zdown_quat(cq)            # 保持 z 朝下、跟随当前 yaw -> 不别扭
            if float(torch.linalg.norm(c[:2] - self._ret[:2])) < max(self.pos_tol, 0.05) or self._t > 6.0:
                self.phase = "done"
        else:
            self.done = True
            return None, self.gripper
        res = self.adapter.solve(PoseState(tgt, self._quat))
        return (res.q_des if getattr(res, "success", False) else None), self.gripper


class _WaypointMover:
    """开环把 TCP 依次【绕行】若干世界系途径点(张爪)。每个途径点带一个【绕行半径】：TCP 进入该半径内就
    切到下一段(不必精确到达=blend/zone 语义)，用来在技能之间过渡时绕开已打开的抽屉等障碍。途径点若存了
    朝向就用它,否则保持当前朝向。done 后交还(随后的技能从最后一个途径点附近起步)。

    waypoints: list of dict {"pos":[x,y,z], "quat":[w,x,y,z]?, "radius": r}  或  裸 pos 张量(兼容)。"""

    def __init__(self, adapter, waypoints, *, device, pos_tol: float = 0.05):
        self.adapter = adapter
        self.device = device
        # 每个 wp = (kind, data, quat, rad)。kind="task": data=世界 pos(绕行半径 blend); kind="joint":
        # data=7 臂关节坐标(严格到达,直接命令关节,能复现底座回转)。
        self.wps = []
        for w in waypoints or []:
            if w is None:
                continue
            if isinstance(w, dict):
                if w.get("kind") == "joint" and w.get("joint_pos"):
                    jp = torch.tensor(w["joint_pos"], dtype=torch.float32, device=device)
                    self.wps.append(("joint", jp, None, 0.0))
                else:
                    pos = torch.tensor(w["pos"], dtype=torch.float32, device=device)
                    quat = (torch.tensor(w["quat"], dtype=torch.float32, device=device)
                            if w.get("quat") else None)
                    self.wps.append(("task", pos, quat, float(w.get("radius", pos_tol))))
            else:
                self.wps.append(("task", w.reshape(3).to(device).clone(), None, pos_tol))
        self.i = 0
        self._t = 0.0
        self.done = False

    def step(self, state, dt):
        c = state.robot.tcp_pose.pos_w.reshape(3).to(self.device)
        cq = state.robot.tcp_pose.quat_w.reshape(4).to(self.device)
        self._t += dt
        if self.i >= len(self.wps):
            self.done = True
            return None, 1.0
        kind, data, quat, rad = self.wps[self.i]
        if kind == "joint":                     # 关节途径点：严格到达(臂关节 L2<0.06)
            reached = float(torch.linalg.norm(
                state.robot.joint_pos[self.adapter._joint_ids] - data)) < 0.06
        else:                                   # 任务途径点：进绕行半径内即可(blend)
            reached = float(torch.linalg.norm(c - data)) < max(rad, 0.02)
        if reached or self._t > 8.0:
            self.i += 1
            self._t = 0.0
            if self.i >= len(self.wps):
                self.done = True
                return None, 1.0
            kind, data, quat, rad = self.wps[self.i]
        if kind == "joint":
            return data.clone(), 1.0            # 直接命令关节坐标(严格复现,含底座回转)
        tq = quat if quat is not None else cq   # 有存朝向就用，否则保持当前朝向
        res = self.adapter.solve(PoseState(data, tq))
        return (res.q_des if getattr(res, "success", False) else None), 1.0


class SkillTestController:
    def __init__(self, session):
        self.s = session
        self.env = session.env
        self.provider = session.provider
        self._dt = float(getattr(session, "_sim_dt", 0.02)) or 0.02   # 防 0(否则基于 dt 的超时全失效->卡死)

        # fast test motion: bump the global skill speed scale (env override, default 5x)
        applied = set_speed_scale(float(os.environ.get("SKILL_TEST_SPEED", "5.0")))
        print(f"[SkillTestController] skill speed scale = {applied:.1f}x", flush=True)

        registry = TargetRegistry(self.env.unwrapped.device)
        adapter = IKJointAdapter(self.env)
        self.adapter = adapter
        self.device = self.env.unwrapped.device
        backend = JointBackendConfig(
            mode="joint",
            grasp_backend="joint_ik",
            place_backend="joint_ik",
            drawer_backend="ik_pull",  # pure-physical grasp+pull/push (no policy, no joint-target cheat)
            adapter=adapter,
            drawer_env=self.env,
            arm_joint_ids=self.provider._arm_joint_ids,
            drawer_joint_name="joint_0",
            # 抽屉技能顺序：PRELIFT(先竖直抬到 home 高度) -> 直接 reach 把手。reach 阶段用【加权 DLS
            # 让关节1(底座)优先旋转】平滑转身正对把手(joint1_face_weight)，不再插 TURN_TO_FACE 中间点
            # (那个是离散关节跳，用户嫌笨)。关节1优先旋转的需求由 IK 约束在 reach 中自然完成。
            drawer_open_ik_config=OpenDrawerIKConfig(use_turn_to_face=False, start_from_current=True),
            drawer_close_ik_config=CloseDrawerIKConfig(use_turn_to_face=False, start_from_current=True),
        )
        self.executor = SkillExecutor(
            registry, log_path="logs/skill_tests/test_mode_results.jsonl", backend=backend
        )
        self.registry = registry

        # collision monitor: reads the robot ContactSensor (scene['robot_contact']); flags the robot
        # BODY (arm/wrist/hand, not fingers) bumping the scene -- e.g. knocking the door on retreat.
        self.collision_monitor = CollisionMonitor(self.provider.scene, env_id=0)
        self._last_collision_set: frozenset = frozenset()

        # UI-driven state
        self._pending: SkillRequest | None = None
        self._want_stop = False
        self._grasp_runner: _GraspPoseRunner | None = None   # 抓取任意 pose 的开环执行器(沿 Z 接近/回退)
        self._place_runner: _PlacePoseRunner | None = None   # 放置进篮子的开环执行器(抬起->上方->松爪)
        self._pre_mover: _WaypointMover | None = None         # 技能前的待命移动(绕开基座圈到目标侧)
        self._pending_after_move: SkillRequest | None = None  # 待命移动完成后再发的技能请求
        self._drawer_queue: list[str] = []                    # 顺序打开白柜两个抽屉的小队列(top->middle)
        self._drawer_gap = 0                                  # 两个抽屉之间的稳定等待帧数
        self._coffee_calib: dict | None = None                # 咖啡把手关节轴实测标定的小状态机
        # home 姿态(竖直向下抬起):放置后回 home，物体间转移走 home 高度的弧线，避免缠绕/关节限位
        # 开环抓取/放置 的运动速度。放大沿接近轴的"胡萝卜"步长 + 抬高 IK 每步关节限幅(真正提速)。
        self._runner_speed = float(os.environ.get("SKILL_TEST_RUNNER_SPEED", "3.0"))
        self._base_max_joint_step = float(getattr(self.adapter, "max_joint_step", 0.20))
        self._apply_speed_to_adapter()
        self._home_extra_z = float(os.environ.get("SKILL_TEST_HOME_EXTRA_Z", "0.15"))  # 篮子上方再抬高(10+5cm)
        # 抓取/放置的"到达"阶段(物体/篮子上方)再比 carry 高度高 _arrival_margin —— 先升到上方再竖直下扎。
        self._arrival_margin = float(os.environ.get("SKILL_TEST_ARRIVAL_MARGIN", "0.06"))
        self._saved_grasp_poses: dict = {}                   # 从 grasp_poses.json 读到的 {name: entry}(无面板时回退)
        self._grasp_panel = None                             # GraspPosePanel 引用：用它的【实时】抓取位姿
        self._wp_panel = None                                # SkillWaypointsPanel 引用：技能过渡的绕行途径点
        self._transited_req = None                           # 已完成途径点绕行的请求(避免重复绕行)
        self._transit = None                                 # CuroboTransit(懒建, False=建失败); None=未建
        self._transit_runner = None                          # 正在执行的 cuRobo 过渡轨迹执行器
        self._transit_stage = None                           # None / "retract"(先抬离) / "curobo"(规划中)
        self._curobo_pregrasp = None                         # 暂存 cuRobo 目标(把手前 pre-grasp)
        # cuRobo 过渡默认【关】(opt-in)：目前操作完抽屉后机器人正处于"刚打开的抽屉障碍盒"里，start-in-
        # collision 导致规划失败,需要先"竖直抬离"的 retract-first 才有用(待做)。设 SKILL_TEST_CUROBO=1 试。
        self._use_curobo = os.environ.get("SKILL_TEST_CUROBO", "") not in ("", "0")
        self._freed: set[tuple[str, str]] = set()  # (asset_name, joint_name) we already freed

        # selections (defaults; overwritten by combo boxes if a window is built)
        self.sel_skill = SkillType.OPEN_DOOR
        self.sel_grasp = _GRASP_TARGETS[0]
        self.sel_drawer = _DRAWER_TARGETS[0]
        self.sel_door = _DOOR_TARGETS[0]

        # omni.ui handles (only when build_window succeeds)
        self._models = {}
        self._status_label = None

        # own marker set for live target-pose visualization (independent of session.show_target_poses,
        # which test_mode's main loop clears every frame for its UI#1 arrow).
        self._markers = None

        # autorun (unattended sequence) state
        self._auto = _parse_autorun(os.environ.get("SKILL_TEST_AUTORUN"))
        self._auto_i = 0
        self._auto_started = False
        self._auto_steps = 0
        self._settle_left = _AUTORUN_SETTLE_STEPS  # settle once before the first skill
        self._auto_results: list[dict] = []
        self._auto_done_logged = False
        if self._auto:
            seq = ", ".join(f"{st.value}:{tgt}" for st, tgt in self._auto)
            print(f"[AUTORUN] sequence ({len(self._auto)}): {seq}", flush=True)

        # sorting smoke test (grasp object -> place into basket), headless via SKILL_TEST_SORT
        self._sort = _parse_sort(os.environ.get("SKILL_TEST_SORT"))
        self._sort_i = 0
        self._sort_done_logged = False
        if self._sort:
            self._saved_grasp_poses = self._load_grasp_poses()
            print(f"[SORT] sequence ({len(self._sort)}): "
                  + ", ".join(f"{o}->{b}" for o, b in self._sort), flush=True)

        # headless coffee-grasp smoke test: SKILL_TEST_COFFEE=1 triggers _on_operate_coffee once
        self._coffee_test = os.environ.get("SKILL_TEST_COFFEE", "") not in ("", "0")
        self._coffee_started = False
        if self._coffee_test:
            self._saved_grasp_poses = self._load_grasp_poses()
            print("[COFFEE] headless grasp test enabled", flush=True)

        # headless 验证：SKILL_TEST_OPEN_BOTH=1 触发"打开两个抽屉"；SKILL_TEST_CLOSE_ALL=1 触发"关闭所有抽屉"
        self._open_both_test = os.environ.get("SKILL_TEST_OPEN_BOTH", "") not in ("", "0")
        self._close_all_test = os.environ.get("SKILL_TEST_CLOSE_ALL", "") not in ("", "0")
        self._open_sektion_test = os.environ.get("SKILL_TEST_OPEN_SEKTION", "") not in ("", "0")
        self._open_both_started = False
        self._close_all_started = False
        self._open_sektion_started = False
        if self._open_both_test:
            print("[OPENBOTH] headless open-both-drawers test enabled", flush=True)
        if self._close_all_test:
            print("[CLOSEALL] headless close-all-drawers test enabled", flush=True)
        if self._open_sektion_test:
            print("[OPENSEKTION] headless open-sektion-drawers test enabled", flush=True)

        # headless 连贯序列测试：SKILL_TEST_SEQUENCE="open_drawer:bottom_drawer,open_drawer:top_drawer"
        # 走抽屉队列(_pending 路径 -> 途径点/cuRobo 无碰撞过渡),依次执行,验证连贯不撞。
        _seq = os.environ.get("SKILL_TEST_SEQUENCE", "")
        if _seq:
            q = []
            for item in _seq.split(","):
                nm, _, tg = item.strip().partition(":")
                st = _SKILL_BY_NAME.get(nm.strip())
                if st is not None and tg.strip():
                    q.append((st, tg.strip()))
            if q:
                self._drawer_queue = q
                print(f"[SEQUENCE] {_seq} ({len(q)} steps)", flush=True)

    # --------------------------------------------------------------- UI window
    def build_window(self):
        import omni.ui as ui

        # 精简到只保留【排序抓放】工作流：保存pose抓取 + 放进篮子 + 速度滑条。
        # (门/抽屉技能、旧cube/knife技能项、Reload按钮 已清理；以后需要再加回来。)
        self.window = ui.Window("Skill Test (sort grasp+place)", width=360, height=220)
        with self.window.frame:
            with ui.VStack(spacing=8, height=0):
                ui.Label("Grasp target (table objects; uses your LIVE edited pose)")
                self._pose_frame = ui.Frame(height=26)
                self._rebuild_pose_combo()
                ui.Button("Grasp (Z-approach)", clicked_fn=self._on_grasp_saved)
                ui.Separator()
                ui.Label("Place basket (carry held object -> drop in)")
                self._models["basket"] = ui.ComboBox(0, *[name for name, _ in _BASKET_TARGETS]).model
                ui.Button("Place to Basket", clicked_fn=self._on_place_basket)
                ui.Separator()
                ui.Label("Drawer (select one, then Open / Close). Incl. white Sektion cabinet.")
                self._models["drawer"] = ui.ComboBox(0, *_DRAWER_TARGETS).model
                with ui.HStack(spacing=8, height=28):
                    ui.Button("Open Drawer", clicked_fn=self._on_open_drawer)
                    ui.Button("Close Drawer", clicked_fn=self._on_close_drawer)
                    ui.Button("Stop", clicked_fn=self._on_stop)
                ui.Separator()
                ui.Label("Coffee lever (grasp + horizontal swing)")
                with ui.HStack(spacing=8, height=28):
                    ui.Button("Operate Coffee", clicked_fn=self._on_operate_coffee)
                    ui.Button("Stop", clicked_fn=self._on_stop)
                ui.Separator()
                with ui.HStack(spacing=8, height=22):
                    ui.Label("Runner speed (grasp/place)", width=220)
                    self._speed_value_label = ui.Label(f"{self._runner_speed:.1f}x")
                sp = ui.FloatSlider(min=0.5, max=8.0)
                sp.model.set_value(self._runner_speed)
                sp.model.add_value_changed_fn(self._on_speed_changed)
                with ui.HStack(spacing=8, height=22):
                    ui.Label("Home height (raise above baskets)", width=220)
                    self._homez_value_label = ui.Label(self._home_z_label())
                hz = ui.FloatSlider(min=0.0, max=0.6)
                hz.model.set_value(self._home_extra_z)
                hz.model.add_value_changed_fn(self._on_home_z_changed)
                self._status_label = ui.Label("status: idle")

    def _on_speed_changed(self, model):
        """UI 滑条改运动速度：放大胡萝卜步长(下次抓/放生效) + 立刻抬高 IK 每步关节限幅(当前也提速)。"""
        try:
            self._runner_speed = max(0.5, float(model.get_value_as_float()))
            self._apply_speed_to_adapter()
            if getattr(self, "_speed_value_label", None) is not None:
                self._speed_value_label.text = f"{self._runner_speed:.1f}x"
        except Exception:
            pass

    def _home_z_label(self) -> str:
        """home 高度滑条旁的读数：抬高量 + 解析出的绝对世界 carry 高度(到达阶段还会再高 margin)。"""
        try:
            z = f"{self._carry_z():.2f}m"
        except Exception:
            z = "?"
        return f"+{self._home_extra_z*100:.0f}cm (z={z})"

    def _on_home_z_changed(self, model):
        """UI 滑条改搬运高度(相对篮子的抬高量)；下次抓/放的 carry/到达高度随之变。"""
        try:
            self._home_extra_z = max(0.0, float(model.get_value_as_float()))
            if getattr(self, "_homez_value_label", None) is not None:
                self._homez_value_label.text = self._home_z_label()
        except Exception:
            pass

    # --------------------------------------------------------------- saved grasp poses
    def _grasp_poses_path(self) -> Path:
        try:
            from franka_v1_skill_lab.scene.scene_registry import V1_ACTIVE_DIR
            return Path(V1_ACTIVE_DIR) / "grasp_poses.json"
        except Exception:
            return Path(__file__).resolve().parents[2] / (
                "franka_v1_skill_lab/scene/saved_scenes/v1_active/grasp_poses.json")

    def _load_grasp_poses(self) -> dict:
        try:
            p = self._grasp_poses_path()
            if p.is_file():
                d = json.loads(p.read_text(encoding="utf-8"))
                return dict(d.get("poses") or {})
        except Exception as exc:  # pragma: no cover
            print(f"[SkillTestController] load grasp_poses.json failed: {exc}", flush=True)
        return {}

    def attach_grasp_panel(self, panel):
        """绑定 GraspPosePanel：抓取目标列表 + 抓取位姿改用它的【实时】值（用户微调后即时生效，不必先存盘）。"""
        self._grasp_panel = panel
        try:
            self._rebuild_pose_combo()
        except Exception:
            pass

    def attach_waypoints_panel(self, panel):
        """绑定 SkillWaypointsPanel：起技能前用它【实时】编辑的途径点绕行(避开已打开的抽屉等)。"""
        self._wp_panel = panel

    def _default_waypoints_for(self, req):
        """把抽屉技能内置的两个过渡点【预填】成可编辑途径点：
          home 点 -> task(机器人基座正上方 carry 高度,自上而下,带绕行半径);
          facing 点 -> joint(HOME_Q_VERTICAL_RAISED + 关节1=正对把手方位角,严格到达,复现底座回转)。
        用户可在面板里删/改/存。返回 [home, facing] 或 None(把手解析不到)。"""
        try:
            import math
            import isaaclab.utils.math as mu
            drawer = req.destination_object or ""
            pw = self._resolve_named_pose("handle_" + drawer)
            if pw is None:
                return None
            robot = self.provider.scene["robot"]
            eid = self.adapter.env_id
            base_pos = robot.data.root_pos_w[eid]
            base_quat = robot.data.root_quat_w[eid]
            d = pw.pos_w - base_pos
            d_base = mu.quat_apply(mu.quat_inv(base_quat).unsqueeze(0), d.unsqueeze(0))[0]
            azimuth = math.atan2(float(d_base[1]), float(d_base[0]))
            from runtime.drawer_ik_common import HOME_Q_VERTICAL_RAISED
            faced = [float(v) for v in HOME_Q_VERTICAL_RAISED]
            lo, hi = float(self.adapter._joint_lower[0]), float(self.adapter._joint_upper[0])
            faced[0] = max(lo, min(hi, azimuth))
            cz = float(self._carry_z())
            home = {"kind": "task", "pos": [float(base_pos[0]), float(base_pos[1]), cz],
                    "quat": [0.0, 1.0, 0.0, 0.0], "radius": 0.10}
            facing = {"kind": "joint", "joint_pos": faced,
                      "pos": [float(pw.pos_w[0]), float(pw.pos_w[1]), cz],   # 仅供画 marker
                      "quat": [float(v) for v in pw.quat_w.tolist()], "radius": 0.0}
            return [home, facing]
        except Exception as exc:
            print(f"[SkillTestController] default waypoints failed: {exc}", flush=True)
            return None

    def _skill_id_for(self, req) -> str | None:
        """把技能请求映射成途径点表的 skill_id(与 SkillWaypointsPanel.SKILL_IDS 一致)。"""
        try:
            if req.skill_type == SkillType.OPEN_DRAWER:
                return f"open_drawer:{req.destination_object}"
            if req.skill_type == SkillType.CLOSE_DRAWER:
                return f"close_drawer:{req.destination_object}"
        except Exception:
            pass
        return None

    def _load_skill_waypoints(self, skill_id: str | None) -> list:
        """取某技能的途径点(world 系 pose+绕行半径)：优先面板【实时】值，无面板回退读 skill_waypoints.json。"""
        if not skill_id:
            return []
        panel = self._wp_panel
        if panel is not None:
            try:
                return [dict(w) for w in panel.waypoints.get(skill_id, [])]
            except Exception:
                pass
        try:
            from franka_v1_skill_lab.scene.scene_registry import V1_ACTIVE_DIR
            p = Path(V1_ACTIVE_DIR) / "skill_waypoints.json"
            if p.is_file():
                d = json.loads(p.read_text(encoding="utf-8"))
                return list((d.get("skills") or {}).get(skill_id, []))
        except Exception as exc:
            print(f"[SkillTestController] load skill_waypoints failed: {exc}", flush=True)
        return []

    # --------------------------------------------------------------- cuRobo 无碰撞过渡
    def _ensure_transit(self):
        """懒建 CuroboTransit(首次~12s warmup)。返回实例或 None(不可用)。"""
        if self._transit is False:
            return None
        if self._transit is not None:
            return self._transit
        if not self._use_curobo:
            self._transit = False
            return None
        try:
            from runtime.curobo_transit import CuroboTransit
            robot = self.provider.scene["robot"]
            print("[SkillTestController] building cuRobo transit planner (warmup ~12s)...", flush=True)
            # test_mode_ui 主循环在 torch.inference_mode() 里，而 cuRobo 优化需要梯度 -> 临时关掉 inference。
            with torch.inference_mode(False), torch.enable_grad():
                self._transit = CuroboTransit(self.env.unwrapped, robot)   # 需要 unwrapped(有 .scene)
            print("[SkillTestController] cuRobo transit ready", flush=True)
        except Exception as exc:
            print(f"[SkillTestController] cuRobo transit unavailable ({exc}); straight-line fallback", flush=True)
            self._transit = False
            return None
        return self._transit

    def _open_drawer_boxes(self):
        """把当前【已打开】的柜子抽屉(joint>0.03)的连杆世界 AABB 作为动态障碍盒返回给 cuRobo。"""
        boxes = []
        try:
            from runtime.curobo_transit import _world_aabb
            stage = self.env.unwrapped.scene.stage
            for dj, link in (("joint_0", "link_0"), ("joint_1", "link_1"), ("joint_2", "link_2")):
                jp = self._joint_angle("cabinet", dj)
                if jp is not None and jp > 0.03:
                    ab = _world_aabb(stage, f"/World/envs/env_0/Cabinet/{link}")
                    if ab is not None:
                        boxes.append((f"open_{link}", ab[0], ab[1]))
            # 白柜(sektion)已打开的抽屉同理
            if "sektion_cabinet" in self.provider.scene.keys():
                for dj, link in (("drawer_top_joint", "drawer_top"), ("drawer_bottom_joint", "drawer_bottom")):
                    jp = self._joint_angle("sektion_cabinet", dj)
                    if jp is not None and jp > 0.03:
                        ab = _world_aabb(stage, f"/World/envs/env_0/SektionCabinet/{link}")
                        if ab is not None:
                            boxes.append((f"open_sektion_{link}", ab[0], ab[1]))
        except Exception as exc:
            print(f"[SkillTestController] open-drawer boxes failed: {exc}", flush=True)
        return boxes

    def _skill_staging_pose(self, req):
        """技能的【过渡落点】：把手正上方 carry 高度、抓取朝向。cuRobo 无碰撞飞到这里,技能再从上方下扎。"""
        try:
            if req.skill_type not in (SkillType.OPEN_DRAWER, SkillType.CLOSE_DRAWER):
                return None
            pw = self._resolve_named_pose("handle_" + (req.destination_object or ""))
            if pw is None:
                return None
            pos = pw.pos_w.clone()
            pos[2] = float(self._carry_z())
            return pos, pw.quat_w.clone()
        except Exception:
            return None

    def _skill_pregrasp_pose(self, req):
        """技能的 pre-grasp(把手前站位)世界位姿：把手 - 抓取接近轴(+Z)方向 * clearance,朝向=抓取朝向。
        cuRobo 无碰撞飞到这里,技能再从 APPROACH 短距滑上把手(不再大范围移动,避免蹭到已开抽屉)。"""
        try:
            if req.skill_type not in (SkillType.OPEN_DRAWER, SkillType.CLOSE_DRAWER):
                return None
            pw = self._resolve_named_pose("handle_" + (req.destination_object or ""))
            if pw is None:
                return None
            import isaaclab.utils.math as mu
            z = mu.quat_apply(pw.quat_w.reshape(1, 4),
                              torch.tensor([[0.0, 0.0, 1.0]], device=pw.pos_w.device))[0]
            z = z / (torch.linalg.norm(z) + 1e-9)
            clearance = float(os.environ.get("SKILL_TEST_CUROBO_CLEARANCE", "0.12"))
            pos = pw.pos_w - z * clearance         # 沿 -Z(朝外)退 clearance
            return pos, pw.quat_w.clone()
        except Exception:
            return None

    def _retract_waypoint(self, state):
        """当前 TCP 正上方的安全高位途径点：先竖直抬到这里,离开刚操作的抽屉,cuRobo 才有非碰撞起点。"""
        c = state.robot.tcp_pose.pos_w
        z = float(self._carry_z()) + 0.10
        return {"pos": [float(c[0]), float(c[1]), max(z, float(c[2]) + 0.05)], "radius": 0.05}

    def _start_curobo_to_pregrasp(self, req, state) -> bool:
        """从(已抬离的)当前位姿用 cuRobo 规划无碰撞轨迹到 pre-grasp;成功则挂上 _transit_runner。"""
        transit = self._transit
        pregrasp = self._curobo_pregrasp
        if transit is None or transit is False or pregrasp is None:
            return False
        try:
            from skills.move_to_pose_skill import MoveToPoseSkill
            boxes = self._open_drawer_boxes()
            with torch.inference_mode(False), torch.enable_grad():
                transit.build_appliance_cuboids(extra=boxes)
                transit.refresh_world()
                runner = MoveToPoseSkill(transit, pregrasp[0], pregrasp[1], gripper=1.0,
                                         label=f"transit->{self._skill_id_for(req)}")
                runner.start(state)
            if runner.status not in (ExecutionStatus.FAILED, ExecutionStatus.STOPPED):
                self._transit_runner = runner
                print(f"[SkillTestController] cuRobo plan OK ({len(boxes)} open-drawer obstacle(s)) "
                      f"-> pre-grasp of {self._skill_id_for(req)}", flush=True)
                return True
            print(f"[SkillTestController] cuRobo plan FAILED for {self._skill_id_for(req)}", flush=True)
        except Exception as exc:
            print(f"[SkillTestController] cuRobo plan error: {exc}", flush=True)
        return False

    def _rebuild_pose_combo(self):
        """重建抓取目标下拉框。有面板时列【所有可抓目标】(桌面物体/碗/把手，实时)；否则回退读 grasp_poses.json。"""
        if self._grasp_panel is not None:
            self._pose_names = list(getattr(self._grasp_panel, "objects", [])) or ["(no targets)"]
        else:
            self._saved_grasp_poses = self._load_grasp_poses()
            self._pose_names = list(self._saved_grasp_poses.keys()) or ["(save poses first)"]
        if getattr(self, "_pose_frame", None) is None:
            return
        import omni.ui as ui
        self._pose_frame.clear()
        with self._pose_frame:
            self._models["pose"] = ui.ComboBox(0, *self._pose_names).model

    def _link_idx(self, asset, link_name):
        try:
            names = list(getattr(asset.data, "body_names", []))
            idx = next((i for i, n in enumerate(names) if n == link_name), None)
            if idx is None:
                idx = next((i for i, n in enumerate(names) if link_name in n), None)
            return idx
        except Exception:
            return None

    def _saved_pose_world(self, entry: dict):
        """把保存的【参考系局部】抓取位姿换算成当前世界位姿：world = ref_world(member,link) ∘ local。"""
        import isaaclab.utils.math as mu

        scene = self.provider.scene
        member, link = entry.get("member"), entry.get("link")
        if not member or member not in scene.keys():
            return None
        asset = scene[member]
        if link is None:
            rpos = asset.data.root_pos_w[0]; rquat = asset.data.root_quat_w[0]
        else:
            idx = self._link_idx(asset, link)
            if idx is None:
                return None
            rpos = asset.data.body_pos_w[0, idx]; rquat = asset.data.body_quat_w[0, idx]
        lp = torch.tensor(entry["pos"], dtype=torch.float32, device=self.device).reshape(1, 3)
        lq = torch.tensor(entry["quat"], dtype=torch.float32, device=self.device).reshape(1, 4)
        wpos = rpos.reshape(1, 3) + mu.quat_apply(rquat.reshape(1, 4), lp)
        wquat = mu.quat_mul(rquat.reshape(1, 4), lq)
        return PoseState(wpos[0], wquat[0])

    def _on_grasp_saved(self):
        names = getattr(self, "_pose_names", [])
        idx = self._combo_index("pose")
        if not names or not (0 <= idx < len(names)):
            self._set_status("no grasp target selected")
            return
        name = names[idx]
        pose_w = None
        # 优先用 GraspPosePanel 的【实时】抓取位姿（用户刚微调的，不必先存盘）
        panel = self._grasp_panel
        if panel is not None and name in getattr(panel, "entries", {}):
            try:
                p7 = panel._world_grasp(name)
                pose_w = PoseState(torch.tensor(p7[:3], dtype=torch.float32, device=self.device),
                                   torch.tensor(p7[3:], dtype=torch.float32, device=self.device))
            except Exception as exc:
                self._set_status(f"live pose '{name}' failed: {exc}")
                return
        elif name in self._saved_grasp_poses:   # 无面板时回退用存盘的
            pose_w = self._saved_pose_world(self._saved_grasp_poses[name])
        if pose_w is None:
            self._set_status(f"cannot resolve grasp target '{name}'")
            return
        self.executor.reset()           # 让出技能执行器，交给开环抓取器（沿 Z 接近/回退）
        self._pending = None
        self._place_runner = None
        self._grasp_runner = _GraspPoseRunner(self.adapter, pose_w, device=self.device, speed=self._runner_speed, arrival_z=self._arrival_z())
        self._set_status(f"grasp (Z-approach): {name}")

    def _basket_world_pos(self, member: str):
        """篮子世界位：优先用保存的参考系位姿(reference_local->world)，回退读场景成员 root。"""
        entry = self._saved_grasp_poses.get(member)
        if entry is None:
            entry = self._load_grasp_poses().get(member)
        if entry is not None:
            pw = self._saved_pose_world(entry)
            if pw is not None:
                return pw.pos_w
        try:
            scene = self.provider.scene
            if member in scene.keys():
                return scene[member].data.root_pos_w[0]
        except Exception:
            pass
        return None

    def _on_place_basket(self):
        """把当前夹着的物体放进所选篮子（开环：抬起->上方->松爪）。"""
        idx = self._combo_index("basket")
        if not (0 <= idx < len(_BASKET_TARGETS)):
            self._set_status("no basket selected")
            return
        label, member = _BASKET_TARGETS[idx]
        basket_pos = self._basket_world_pos(member)
        if basket_pos is None:
            self._set_status(f"cannot resolve basket '{member}'")
            return
        self.executor.reset()
        self._pending = None
        self._grasp_runner = None
        self._place_runner = _PlacePoseRunner(
            self.adapter, basket_pos, device=self.device, speed=self._runner_speed,
            carry_z=self._carry_z(), arrival_z=self._arrival_z())
        print(f"[SkillTestController] place -> {label} at "
              f"({float(basket_pos[0]):.3f},{float(basket_pos[1]):.3f},{float(basket_pos[2]):.3f})", flush=True)
        self._set_status(f"place -> {label}")

    def on_reset(self):
        # Layout was randomized; drop any active skill. Freed joints stay freed (re-freed on next
        # start anyway) -- env.reset does not restore actuator gains, and we never save them.
        self.executor.reset()
        self._pending = None
        self._drawer_queue = []
        self._drawer_gap = 0
        self._want_stop = False

    # --------------------------------------------------------------- callbacks
    def _combo_index(self, key: str) -> int:
        model = self._models.get(key)
        if model is None:
            return 0
        return int(model.get_item_value_model().get_value_as_int())

    def _on_start(self):
        self.sel_skill = _SKILLS[self._combo_index("skill")][1]
        self.sel_grasp = _GRASP_TARGETS[self._combo_index("grasp")]
        self.sel_drawer = _DRAWER_TARGETS[self._combo_index("drawer")]
        self.sel_door = _DOOR_TARGETS[self._combo_index("door")]
        self._pending = self._make_request(self.sel_skill)

    def _on_stop(self):
        self._want_stop = True
        self._pending = None

    def _resolve_named_pose(self, name: str):
        """按名字解析抓取位姿(完整 pos+quat)，始终取【最新值】：
        优先用场景编辑 pose-UI 面板(GraspPosePanel)的【实时】值；没有面板时【每次都重新读】
        grasp_poses.json(用户保存后立刻生效)。不使用任何缓存，保证技能目标 == UI 设置的 pose。"""
        panel = self._grasp_panel
        if panel is not None and name in getattr(panel, "entries", {}):
            try:
                p7 = panel._world_grasp(name)
                print(f"[SkillTestController] '{name}' pose <- LIVE pose-UI panel "
                      f"({p7[0]:.3f},{p7[1]:.3f},{p7[2]:.3f})", flush=True)
                return PoseState(torch.tensor(p7[:3], dtype=torch.float32, device=self.device),
                                 torch.tensor(p7[3:], dtype=torch.float32, device=self.device))
            except Exception as exc:
                print(f"[SkillTestController] live pose '{name}' failed ({exc}); reading grasp_poses.json",
                      flush=True)
        entry = self._load_grasp_poses().get(name)   # 始终重新读盘(取最新保存)
        if entry is not None:
            pw = self._saved_pose_world(entry)
            if pw is not None:
                print(f"[SkillTestController] '{name}' pose <- grasp_poses.json (latest saved) "
                      f"({float(pw.pos_w[0]):.3f},{float(pw.pos_w[1]):.3f},{float(pw.pos_w[2]):.3f})", flush=True)
            return pw
        return None

    def _link_world_pos(self, member: str, link_name: str):
        """读取某 articulation 某 link 的世界位置(用作铰链点)。"""
        asset = self.provider.scene[member]
        names = list(getattr(asset.data, "body_names", []))
        i = next((k for k, n in enumerate(names) if n == link_name), None)
        if i is None:
            i = next((k for k, n in enumerate(names) if link_name in n), 0)
        return asset.data.body_pos_w[0, i].reshape(3).clone()

    def _init_coffee_lever(self, angle: float = -0.45, lower: float = -0.45, upper: float = 0.55):
        """把咖啡机把手关节(joint_5，绕世界Z水平摆动的杠杆)的初始状态设到 angle(默认 -0.45)，限位
        [lower,upper](默认 [-0.45,0.55])，并用刚度保持(操作时会被 _free_joint 释放再旋转)。只做一次。"""
        jname = os.environ.get("SKILL_TEST_COFFEE_JOINT", COFFEE_LEVER_JOINT)
        try:
            asset = self.provider.scene["coffee_machine"]
            ids, _ = asset.find_joints(jname)
            n = asset.num_instances
            dev = asset.device
            # 诊断：打印咖啡机的 articulation body_names，确认 link_4(把手)是否是独立 body
            # —— 抓取面板按 link_4 解析把手 pose，若这里没有 link_4 会回退 body 0(位置不对)。
            try:
                print(f"[SkillTestController] coffee body_names={list(asset.data.body_names)}", flush=True)
            except Exception:
                pass
            # 关节限位不在这里设：joint_5 的限位 [-0.45,0.55] 已烧进 USD(spawn 时读，最稳)。
            # 运行时调 write_joint_position_limit_to_sim 会【整体重推所有关节限位】，而 joint_0~3 是
            # 无限位连续关节([-inf,inf])，PhysX setLimitParams 不接受 ±2π 外的角度 -> 报 4 条错。
            # 既然 USD 已有 joint_5 限位，这里就不再运行时重设，避免那个报错。
            jp = torch.full((n, len(ids)), float(angle), device=dev)
            jv = torch.zeros((n, len(ids)), device=dev)
            asset.write_joint_state_to_sim(jp, jv, joint_ids=ids)
            asset.set_joint_position_target(jp, joint_ids=ids)
            # 关键：把手关节【没有弹簧驱动力】= stiffness 0(否则会被弹回 target/0)，只留阻尼防自由摆动+消惯性。
            # 这样把手出生停在 init 角，被拉到哪就停在哪，不会自己跑回去(用户要求)。
            asset.write_joint_stiffness_to_sim(torch.zeros((n, len(ids)), device=dev), joint_ids=ids)
            asset.write_joint_damping_to_sim(torch.full((n, len(ids)), 20.0, device=dev), joint_ids=ids)
            try:
                asset.data.default_joint_pos[:, ids] = float(angle)   # reset 也回到初始角
            except Exception:
                pass
            self._freed.discard(("coffee_machine", jname))   # 操作时仍可被 _free_joint 走标定/拖动流程
            # 只保留把手关节(jname)可动：其余所有咖啡机关节锁死(高刚度钉在当前角)，相当于刚体——
            # 用户只需要这一个把手关节，其它按钮/小杆都不动。
            try:
                all_ids, _ = asset.find_joints(".*")
                j4 = set(int(i) for i in ids)
                other = [int(i) for i in all_ids if int(i) not in j4]
                if other:
                    cur = asset.data.joint_pos[:, other].clone()
                    asset.set_joint_position_target(cur, joint_ids=other)
                    asset.write_joint_stiffness_to_sim(
                        torch.full((n, len(other)), 1.0e4, device=dev), joint_ids=other)
                    asset.write_joint_damping_to_sim(
                        torch.full((n, len(other)), 1.0e3, device=dev), joint_ids=other)
                    print(f"[SkillTestController] locked {len(other)} other coffee joints "
                          f"(only {jname} movable)", flush=True)
            except Exception as exc:
                print(f"[SkillTestController] lock other coffee joints failed: {exc}", flush=True)
            print(f"[SkillTestController] coffee {jname} init={angle:.2f}, limits=[{lower:.2f},{upper:.2f}] (held)",
                  flush=True)
        except Exception as exc:
            print(f"[SkillTestController] init coffee lever failed: {exc}", flush=True)

    def _revolute_joint_axis_world(self, member: str, joint_name: str, child_link: str):
        """读关节(revolute)的【世界旋转轴】：返回 (anchor_world_xyz, axis_world_unit)。
        从 USD 关节 prim 读 child 端锚点 localPos1/localRot1 + axis，结合 child link 的世界位姿换算。
        这样旋转就是绕【关节轴线】而不是夹持点。失败返回 (None,None)。"""
        try:
            import isaaclab.utils.math as mu
            from pxr import Usd, UsdGeom

            stage = self.env.unwrapped.sim.stage
            root = stage.GetPrimAtPath(f"/World/envs/env_{0}/CoffeeMachine")
            jprim = None
            for p in Usd.PrimRange(root):
                if p.GetName() == joint_name and "Joint" in p.GetTypeName():
                    jprim = p; break
            if jprim is None:
                print(f"[SkillTestController] joint {joint_name} prim not found under CoffeeMachine", flush=True)
                return None, None
            axis_tok = jprim.GetAttribute("physics:axis").Get() or "Z"
            lp1 = jprim.GetAttribute("physics:localPos1").Get()
            lr1 = jprim.GetAttribute("physics:localRot1").Get()
            axis_vec = {"X": [1.0, 0, 0], "Y": [0, 1.0, 0], "Z": [0, 0, 1.0]}[str(axis_tok)]
            lpos1 = torch.tensor([float(lp1[0]), float(lp1[1]), float(lp1[2])], device=self.device)
            w = float(lr1.GetReal()); im = lr1.GetImaginary()
            lq1 = torch.tensor([w, float(im[0]), float(im[1]), float(im[2])], device=self.device)
            link_pos = self._link_world_pos(member, child_link)
            asset = self.provider.scene[member]
            names = list(asset.data.body_names)
            i = next((k for k, n in enumerate(names) if n == child_link), 0)
            link_quat = asset.data.body_quat_w[0, i].reshape(4).to(self.device)
            anchor = link_pos + mu.quat_apply(link_quat.unsqueeze(0), lpos1.unsqueeze(0))[0]
            jworld_q = mu.quat_mul(link_quat.unsqueeze(0), lq1.unsqueeze(0))[0]
            axis_w = mu.quat_apply(jworld_q.unsqueeze(0),
                                   torch.tensor([axis_vec], device=self.device))[0]
            axis_w = axis_w / (torch.linalg.norm(axis_w) + 1e-9)
            print(f"[SkillTestController] {joint_name} world axis: anchor="
                  f"({float(anchor[0]):.3f},{float(anchor[1]):.3f},{float(anchor[2]):.3f}) "
                  f"axis=({float(axis_w[0]):.2f},{float(axis_w[1]):.2f},{float(axis_w[2]):.2f}) "
                  f"(token={axis_tok})", flush=True)
            return anchor, axis_w
        except Exception as exc:
            print(f"[SkillTestController] joint axis read failed: {exc}", flush=True)
            return None, None

    def _step_coffee_calib(self, state):
        """实测【咖啡机所有关节】各自的世界旋转轴：逐个小幅驱动、看其子连杆怎么动来反算。
        然后选【绕世界 Z(水平摆动)】的那个关节作为把手关节，用它的轴/锚点做闭环旋转。
        (标定，不是执行；执行仍由机器人物理夹住拖动)。"""
        import math
        import isaaclab.utils.math as mu

        asset = self.provider.scene["coffee_machine"]
        n_inst = asset.num_instances; dev = asset.device
        bnames = list(asset.data.body_names)
        jnames = list(asset.data.joint_names)
        cal = self._coffee_calib
        cal["t"] += self._dt
        cal["frames"] = cal.get("frames", 0) + 1   # 帧数总超时(即使 dt 异常为0也有效)，避免永久卡住机器人
        hold = self.provider.make_hold_joint_action(state, 1.0)
        if cal["frames"] > 1500:                    # ~30s@50Hz：标定卡住就放弃
            print("[SkillTestController] coffee calib overall TIMEOUT -> abort", flush=True)
            self._coffee_calib = None
            return None

        def child_idx(jname):                      # joint_N -> body link_N 的下标
            ln = jname.replace("joint", "link")
            return next((k for k, nn in enumerate(bnames) if nn == ln), None)

        def set_one(jname, v):
            ids, _ = asset.find_joints(jname)
            # 标定探测需要驱动力：临时给该关节刚度(否则静息态 stiffness=0 的把手关节不会跟随目标、测不到运动)。
            # 标定结束在 "pick" 阶段会 _free_joint 把它重新设回 stiffness 0(自由、无弹簧)。
            asset.write_joint_stiffness_to_sim(torch.full((n_inst, len(ids)), 200.0, device=dev), joint_ids=ids)
            asset.write_joint_damping_to_sim(torch.full((n_inst, len(ids)), 20.0, device=dev), joint_ids=ids)
            asset.set_joint_position_target(torch.full((n_inst, len(ids)), float(v), device=dev), joint_ids=ids)

        def jpos_one(jname):
            ids, _ = asset.find_joints(jname)
            return float(asset.data.joint_pos[0, ids[0]])

        # ---- 阶段：构建关节队列 ----
        if cal["stage"] == 0:
            if os.environ.get("SKILL_TEST_COFFEE_SCAN", "") not in ("", "0"):
                cal["queue"] = [jn for jn in jnames if child_idx(jn) is not None]   # 扫所有关节(诊断,慢)
            else:
                jn = os.environ.get("SKILL_TEST_COFFEE_JOINT", COFFEE_LEVER_JOINT)           # 默认只标定 joint_5(快)
                cal["queue"] = [jn] if child_idx(jn) is not None else \
                    [j for j in jnames if child_idx(j) is not None]
            cal["results"] = []; cal["qi"] = 0; cal["stage"] = "drive"; cal["t"] = 0.0
            print(f"[SkillTestController] coffee joints to probe: {cal['queue']}", flush=True)
            return hold
        if cal["stage"] == "drive":
            if cal["qi"] >= len(cal["queue"]):
                cal["stage"] = "pick"; return hold
            jn = cal["queue"][cal["qi"]]; ci = child_idx(jn)
            cal["jn"] = jn; cal["ci"] = ci
            cal["th0"] = jpos_one(jn)
            cal["p0"] = asset.data.body_pos_w[0, ci].clone()
            cal["q0"] = asset.data.body_quat_w[0, ci].clone()
            set_one(jn, cal["th0"] + 0.10)   # 标定探测幅度调小(0.30->0.10 rad≈5.7°)：测得到轴又几乎看不出把手动
            cal["stage"] = "meas"; cal["t"] = 0.0
            return hold
        if cal["stage"] == "meas":
            jn = cal["jn"]; ci = cal["ci"]
            if abs(jpos_one(jn) - (cal["th0"] + 0.10)) < 0.02 or cal["t"] > 3.5:
                th1 = jpos_one(jn); p1 = asset.data.body_pos_w[0, ci].clone()
                q1 = asset.data.body_quat_w[0, ci].clone()
                dq = mu.quat_mul(q1.reshape(1, 4), mu.quat_conjugate(cal["q0"].reshape(1, 4)))[0]
                w = max(-1.0, min(1.0, float(dq[0]))); ang = 2.0 * math.acos(w)
                axis = (dq[1:4] / (torch.linalg.norm(dq[1:4]) + 1e-9)) if ang > 1e-3 \
                    else torch.tensor([0.0, 0.0, 0.0], device=dev)
                dp = p1 - cal["p0"]
                cal["results"].append({
                    "jn": jn, "ci": ci, "axis": axis, "ang": ang, "dth": th1 - cal["th0"],
                    "zcomp": abs(float(axis[2])), "p0": cal["p0"].clone(), "p1": p1.clone(),
                    "horiz": float((dp[0] ** 2 + dp[1] ** 2) ** 0.5), "vert": abs(float(dp[2]))})
                print(f"[SkillTestController]   {jn}->{bnames[ci]}: axis=({float(axis[0]):.2f},"
                      f"{float(axis[1]):.2f},{float(axis[2]):.2f}) child moved horiz={cal['results'][-1]['horiz']*100:.1f}cm "
                      f"vert={cal['results'][-1]['vert']*100:.1f}cm dth={th1-cal['th0']:.2f}", flush=True)
                set_one(jn, cal["th0"]); cal["stage"] = "restore"; cal["t"] = 0.0
            return hold
        if cal["stage"] == "restore":
            if abs(jpos_one(cal["jn"]) - cal["th0"]) < 0.03 or cal["t"] > 3.0:
                cal["qi"] += 1; cal["stage"] = "drive"; cal["t"] = 0.0
            return hold
        if cal["stage"] == "pick":
            res = [r for r in cal["results"] if r["ang"] > 1e-3]
            if not res:
                print("[SkillTestController] coffee calib: no movable joint measured; abort", flush=True)
                self._coffee_calib = None; return hold
            # 选【轴最接近世界 Z(竖直)】= 子连杆水平摆动的关节
            best = max(res, key=lambda r: r["zcomp"])
            jn = best["jn"]; ci = best["ci"]; axis = best["axis"]
            if float(axis[2]) < 0:
                axis = -axis     # 统一让 +轴朝上(便于方向直观)；闭环探测会再校正符号
            # 锚点：子连杆原点的圆心(绕 axis)
            p0 = best["p0"]; p1 = best["p1"]; ang = best["ang"]
            v = p1 - p0; v = v - torch.dot(v, axis) * axis; vn = float(torch.linalg.norm(v))
            if vn < 1e-4:
                anchor = p0.clone()
            else:
                mid = (p0 + p1) / 2.0; d = (vn / 2.0) / math.tan(abs(ang) / 2.0)
                perp = torch.linalg.cross(axis, v); perp = perp / (torch.linalg.norm(perp) + 1e-9)

                def _rot(p, c):
                    rz = mu.quat_from_angle_axis(torch.tensor([ang], device=dev), axis.reshape(1, 3))[0]
                    return c + mu.quat_apply(rz.unsqueeze(0), (p - c).reshape(1, 3))[0]
                c1 = mid + perp * d; c2 = mid - perp * d
                anchor = c1 if float(torch.linalg.norm(_rot(p0, c1) - p1)) <= \
                    float(torch.linalg.norm(_rot(p0, c2) - p1)) else c2
            # 【消除运行间随机性】优先用 USD 关节定义里的【确定性】锚点/轴(读 joint prim 的 localPos1)，
            # 而不是靠驱动探测的圆拟合——探测位移很小(link_5 原点几乎在轴上)时，圆拟合对噪声极敏感，偶尔会
            # 把锚点算到【夹持点附近】，此时 sweep 就表现成"绕夹持点原地转"(用户看到的现象)。读到就覆盖，
            # 读不到才回退用探测拟合的 anchor。
            try:
                anchor_usd, axis_usd = self._revolute_joint_axis_world("coffee_machine", jn, bnames[ci])
                if anchor_usd is not None and axis_usd is not None:
                    anchor = anchor_usd
                    axis = axis_usd if float(axis_usd[2]) >= 0 else -axis_usd
                    print("[SkillTestController] coffee anchor/axis <- USD joint def (deterministic, "
                          "avoids noisy probe circle-fit)", flush=True)
            except Exception as _exc:
                print(f"[SkillTestController] USD joint-anchor read failed ({_exc}); using probe fit", flush=True)
            self._coffee_axis = axis; self._coffee_anchor = anchor
            self._coffee_joint = jn; self._coffee_link = bnames[ci]
            print(f"[SkillTestController] PICKED lever joint={jn} link={bnames[ci]} "
                  f"axis=({float(axis[0]):.2f},{float(axis[1]):.2f},{float(axis[2]):.2f}) "
                  f"anchor=({float(anchor[0]):.3f},{float(anchor[1]):.3f},{float(anchor[2]):.3f}) "
                  f"({'HORIZONTAL swing about Z' if best['horiz']>best['vert'] else 'VERTICAL swing'})", flush=True)
            # 启动抓取 + 闭环旋转(绕选中关节)
            ids, _ = asset.find_joints(jn)

            def _read_joint():
                return float(asset.data.joint_pos[0, ids[0]])
            self._free_joint("coffee_machine", jn)
            # 咖啡把手是侧向接近(沿 link -Y)，不走"先抬高再下扎"那套(高位绕行反而把手腕摆到限位、够不到)；
            # arrival_z=None 关掉高位段，standoff 也缩短，直接沿接近轴贴近。
            self._grasp_runner = _GraspPoseRunner(
                self.adapter, self._coffee_pose, device=self.device, speed=self._runner_speed,
                arrival_z=None, retreat=False, standoff=0.06,
                sweep_hinge=anchor, sweep_axis=axis,
                joint_reader=_read_joint, joint_target=self._coffee_target)
            print(f"[SkillTestController] coffee: grasp + sweep {jn} to {self._coffee_target:.2f}", flush=True)
            self._coffee_calib = None
            return hold
        self._coffee_calib = None
        return hold

    def _on_operate_coffee(self):
        """操作咖啡机把手：先实测 joint_4 旋转轴(标定)，再用你设的 handle_coffee_lever 位姿抓住，
        绕【实测的真实关节轴】闭环旋转，把 joint_4 从当前角物理拖到目标角。"""
        pose_w = self._resolve_named_pose("handle_coffee_lever")
        if pose_w is None:
            self._set_status("cannot resolve handle_coffee_lever pose")
            print("[SkillTestController] handle_coffee_lever pose not found in panel/grasp_poses.json", flush=True)
            return
        self.executor.reset()
        self._pending = None
        self._place_runner = None
        self._grasp_runner = None
        # 先【实测】joint_4 的真实旋转轴(用关节自身驱动小幅试转、看 link_4 怎么动 —— 这只是测量标定，
        # 真正执行仍由机器人物理夹住把手拖动)。标定完再抓取 + 绕实测轴旋转到目标。
        self._coffee_pose = pose_w
        self._coffee_target = float(os.environ.get("SKILL_TEST_COFFEE_TARGET", "0.55"))
        self._coffee_calib = {"stage": 0, "t": 0.0}
        print("[SkillTestController] coffee: measuring joint_4 axis (drive-probe calibration)...", flush=True)
        self._set_status("coffee: calibrating axis")

    def _drawer_handle_world(self, drawer: str):
        """抽屉把手世界位(用保存的 handle_<drawer> 位姿)。"""
        pw = self._resolve_named_pose(f"handle_{drawer}")
        return pw.pos_w if pw is not None else None

    def _start_drawer(self, skill_type, label: str):
        idx = self._combo_index("drawer")
        self.sel_drawer = _DRAWER_TARGETS[idx] if 0 <= idx < len(_DRAWER_TARGETS) else _DRAWER_TARGETS[0]
        self._grasp_runner = None
        self._place_runner = None
        self.executor.reset()
        self._pending = self._make_request(skill_type)
        print(f"[SkillTestController] {label} -> {self.sel_drawer}", flush=True)
        self._set_status(f"{label} {self.sel_drawer}")

    def _on_open_drawer(self):
        """打开所选抽屉：纯物理 ik_pull 从当前位姿抓把手拉开。"""
        self._start_drawer(SkillType.OPEN_DRAWER, "open drawer")

    def _on_close_drawer(self):
        """关闭所选抽屉：纯物理 ik_pull 从当前位姿抓把手推回。"""
        self._start_drawer(SkillType.CLOSE_DRAWER, "close drawer")

    # 柜子的两个【可用】抽屉：top + middle(bottom 被锁)。复用打开/关闭抽屉技能依次执行。
    _FUNCTIONAL_DRAWERS = ("top_drawer", "middle_drawer")

    def _queue_drawers(self, skill_type, drawers, label: str):
        """把若干抽屉排成队列依次跑同一个技能(open/close)。每个结束后自动发下一个(见 _step_impl)。"""
        self._grasp_runner = None
        self._place_runner = None
        self._pre_mover = None
        self.executor.reset()
        self._pending = None
        self._drawer_queue = [(skill_type, d) for d in drawers]
        self._drawer_gap = 0
        print(f"[SkillTestController] {label}: {list(drawers)}", flush=True)
        self._set_status(f"{label}: {' -> '.join(drawers)}")

    def _on_open_both_drawers(self):
        """依次打开旧柜(Cabinet_44853)两个可用抽屉(top -> middle)：复用 OPEN_DRAWER 技能模块。"""
        self._queue_drawers(SkillType.OPEN_DRAWER, self._FUNCTIONAL_DRAWERS, "open both drawers")

    def _on_open_sektion_drawers(self):
        """依次打开右下角白柜(Sektion)的两个抽屉(top -> bottom)：复用 OPEN_DRAWER 技能模块。
        注意：白柜需先在 GUI 里移到机器人可达处、并标定 handle_sektion_*_drawer 抓取位姿。"""
        self._queue_drawers(SkillType.OPEN_DRAWER, list(SEKTION_DRAWERS), "open sektion drawers")

    def _all_drawers(self):
        """所有可用抽屉：旧柜 top+middle + 白柜 top+bottom(若白柜接入)。"""
        drawers = list(functional_drawers())
        if "sektion_cabinet" in self.provider.scene.keys():
            drawers += list(SEKTION_DRAWERS)
        return drawers

    def _on_close_all_drawers(self):
        """依次关闭所有可用抽屉(旧柜 top+middle + 白柜 top+bottom)：复用 CLOSE_DRAWER 技能模块。"""
        self._queue_drawers(SkillType.CLOSE_DRAWER, self._all_drawers(), "close all drawers")

    # --------------------------------------------------------------- per-frame
    def _sort_step(self, state):
        """Headless sorting test: per (object, basket) pair, grasp then place. Logs each result."""
        if self._sort_i >= len(self._sort):
            if not self._sort_done_logged:
                print(f"[SORT] DONE ({self._sort_i}/{len(self._sort)} pairs run)", flush=True)
                self._sort_done_logged = True
            return None
        obj, basket = self._sort[self._sort_i]

        if self._grasp_runner is not None:                       # ... grasping
            q, grip = self._grasp_runner.step(state, self._dt)
            if self._grasp_runner.done:
                failed = getattr(self._grasp_runner, "failed", False)
                self._grasp_runner = None
                if failed:                                        # 抓取卡死/够不到 -> 跳过此物体
                    print(f"[SORT] {obj}: grasp FAILED (stuck/unreachable), skip", flush=True)
                    self._sort_i += 1
                    return self.provider.make_hold_joint_action(state, grip)
                bpos = self._basket_world_pos(basket)
                if bpos is None:
                    print(f"[SORT] {obj}: cannot resolve basket '{basket}', skip", flush=True)
                    self._sort_i += 1
                    return self.provider.make_hold_joint_action(state, grip)
                print(f"[SORT] {obj} grasped -> placing into {basket}", flush=True)
                self._place_runner = _PlacePoseRunner(
                    self.adapter, bpos, device=self.device, speed=self._runner_speed,
                    carry_z=self._carry_z(), arrival_z=self._arrival_z())
                return self.provider.make_hold_joint_action(state, grip)
            return (self.provider.make_joint_action_from_q_des(q, grip) if q is not None
                    else self.provider.make_hold_joint_action(state, grip))

        if self._place_runner is not None:                       # ... placing
            q, grip = self._place_runner.step(state, self._dt)
            if self._place_runner.done:
                self._place_runner = None
                # verdict: is the object now in/near the basket?
                try:
                    op = self.provider.scene[obj].data.root_pos_w[0]
                    bp = self._basket_world_pos(basket)
                    dxy = float(((op[0] - bp[0]) ** 2 + (op[1] - bp[1]) ** 2) ** 0.5)
                    moved = float(((op[0] - self._sort_obj0[0]) ** 2 + (op[1] - self._sort_obj0[1]) ** 2) ** 0.5) \
                        if getattr(self, "_sort_obj0", None) is not None else -1.0
                    verdict = "IN BASKET" if dxy < 0.13 else "MISSED basket"
                    print(f"[SORT] {obj} -> {basket}: obj final=({op[0]:.3f},{op[1]:.3f},{op[2]:.3f}) "
                          f"basket=({bp[0]:.3f},{bp[1]:.3f}) dist={dxy*100:.1f}cm moved={moved*100:.1f}cm "
                          f"=> {verdict}", flush=True)
                except Exception as exc:
                    print(f"[SORT] {obj} placed (verdict read failed: {exc})", flush=True)
                self._sort_i += 1
                return self.provider.make_hold_joint_action(state, grip)
            return (self.provider.make_joint_action_from_q_des(q, grip) if q is not None
                    else self.provider.make_hold_joint_action(state, grip))

        # neither runner active -> start the grasp for the current pair
        entry = (self._saved_grasp_poses or {}).get(obj)
        pose_w = self._saved_pose_world(entry) if entry is not None else None
        if pose_w is None:
            print(f"[SORT] {obj}: no resolvable grasp pose, skip", flush=True)
            self._sort_i += 1
            return None
        try:
            self._sort_obj0 = self.provider.scene[obj].data.root_pos_w[0].clone()
        except Exception:
            self._sort_obj0 = None
        print(f"[SORT] grasping {obj} at "
              f"({float(pose_w.pos_w[0]):.3f},{float(pose_w.pos_w[1]):.3f},{float(pose_w.pos_w[2]):.3f})", flush=True)
        self._grasp_runner = _GraspPoseRunner(self.adapter, pose_w, device=self.device, speed=self._runner_speed, arrival_z=self._arrival_z())
        return None

    def _apply_speed_to_adapter(self):
        """速度滑条同时抬高 IK 每步关节限幅 -> 真正提速(不止改胡萝卜步长)；上限0.5 rad/步防跳变。"""
        try:
            self.adapter.max_joint_step = min(0.5, self._base_max_joint_step * max(0.5, self._runner_speed))
        except Exception:
            pass

    def _baskets_top_z(self) -> float:
        """所有可解析篮子里最高的 z（放置 carry 高度的基准）。解析不到就用 0.80。"""
        zs = []
        for _label, member in _BASKET_TARGETS:
            bp = self._basket_world_pos(member)
            if bp is not None:
                zs.append(float(bp[2]))
        return max(zs) if zs else 0.80

    def _carry_z(self) -> float:
        """搬运高度 = 最高篮子 + drop_above(0.15) + 额外抬高(滑条)。"""
        return self._baskets_top_z() + 0.15 + self._home_extra_z

    def _arrival_z(self) -> float | None:
        """抓取/放置"到达"阶段(物体/篮子正上方)的高度 = carry 高度 + margin —— 先升到上方再竖直下扎。"""
        try:
            return self._carry_z() + self._arrival_margin
        except Exception:
            return None

    def step(self, session):
        """安全外壳：任何异常都不让机器人卡死——出错就清掉可能卡住的状态、交还控制(返回 None)。"""
        try:
            return self._step_impl(session)
        except Exception as exc:  # pragma: no cover
            import traceback
            print(f"[SkillTestController] step ERROR: {exc}", flush=True)
            traceback.print_exc()
            self._coffee_calib = None
            self._pre_mover = None
            self._grasp_runner = None
            self._pending = None
            return None

    def _step_impl(self, session):
        state = self.provider.get_state()
        self._monitor_collisions(state)
        self._manage_drawer_damping()   # 抽屉阻尼随技能阶段切换(拉/推=低,其余=高)，见 _DRAWER_*_DAMPING

        if not getattr(self, "_coffee_inited", False):
            self._coffee_inited = True
            self._init_coffee_lever(float(os.environ.get("SKILL_TEST_COFFEE_INIT", "-0.45")))
            # 抽屉也【没有弹簧驱动力】：一开始就把三个抽屉关节释放成 stiffness 0 + 阻尼
            # (默认 _DRAWER_FREE_DAMPING)，这样抽屉拉到哪停哪、不会自己弹回关闭(用户要求)。
            for _dj in ("joint_0", "joint_1", "joint_2"):
                self._free_joint("cabinet", _dj, damping=_DRAWER_FREE_DAMPING)
            # 白柜(sektion_cabinet)抽屉关节也释放成 stiffness0+阻尼(若该 articulation 接入了)：拉到哪停哪。
            if "sektion_cabinet" in self.provider.scene.keys():
                try:
                    _sk = self.provider.scene["sektion_cabinet"]
                    print(f"[SEKTION] joints={list(_sk.data.joint_names)} bodies={list(_sk.data.body_names)} "
                          f"root={[round(float(v),3) for v in _sk.data.root_pos_w[0].tolist()]}", flush=True)
                except Exception as _e:
                    print(f"[SEKTION] probe failed: {_e}", flush=True)
                for _dj in ("drawer_top_joint", "drawer_bottom_joint"):
                    self._free_joint("sektion_cabinet", _dj, damping=_DRAWER_FREE_DAMPING)

        if getattr(self, "_open_both_test", False) and not self._open_both_started:
            self._open_both_started = True
            self._on_open_both_drawers()

        if getattr(self, "_close_all_test", False) and not self._close_all_started:
            self._close_all_started = True
            self._on_close_all_drawers()

        if getattr(self, "_open_sektion_test", False) and not self._open_sektion_started:
            self._open_sektion_started = True
            self._on_open_sektion_drawers()

        if getattr(self, "_coffee_test", False) and not self._coffee_started:
            self._coffee_started = True
            self._on_operate_coffee()
        if getattr(self, "_coffee_test", False):
            if self._grasp_runner is not None and self._grasp_runner._last_phase != getattr(self, "_coffee_last", None):
                self._coffee_last = self._grasp_runner._last_phase
            if self._grasp_runner is not None and self._grasp_runner.phase in ("close", "done"):
                try:
                    print(f"[COFFEE] phase={self._grasp_runner.phase} gw={float(state.robot.gripper_width):.3f}",
                          flush=True)
                except Exception:
                    pass

        if self._sort is not None:
            return self._sort_step(state)

        if self._auto is not None:
            return self._auto_step(state)

        if self._want_stop:
            self.executor.reset()
            self._grasp_runner = None
            self._place_runner = None
            self._pre_mover = None
            self._pending_after_move = None
            self._transited_req = None
            self._transit_runner = None
            self._transit_stage = None
            self._curobo_pregrasp = None
            self._coffee_calib = None          # Stop 也要能中断咖啡标定/操作，别卡死
            self._pending = None
            self._drawer_queue = []             # Stop 也清掉"打开两个抽屉"的队列
            self._drawer_gap = 0
            self._want_stop = False
            self._apply_speed_to_adapter()     # 恢复 IK 步长(若 Stop 时正处于 sweep 的压低状态)
            self._set_status("stopped")
            return None

        # 咖啡把手关节轴实测标定(标定完会启动抓取+旋转)
        if self._coffee_calib is not None:
            return self._step_coffee_calib(state)

        # cuRobo 无碰撞过渡执行中：飞完(轨迹执行到位)再真正起技能
        if self._transit_runner is not None:
            with torch.inference_mode(False):
                cmd = self._transit_runner.step(state, self._dt)
            self._set_status("cuRobo transit")
            if self._transit_runner.status in (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED,
                                               ExecutionStatus.STOPPED):
                self._transit_runner = None
                self._transit_stage = None
                self._curobo_pregrasp = None
                req = self._pending_after_move
                self._pending_after_move = None
                self._transited_req = req
                # cuRobo 已把机器人送到 pre-grasp -> 技能从 APPROACH 起(短距滑上把手,不再大范围移动)。
                if req is not None:
                    try:
                        req.parameters = {**(req.parameters or {}), "start_at_approach": True}
                    except Exception:
                        pass
                self._pending = req
                return self.provider.make_hold_joint_action(state, 1.0)
            return self._command_to_action(cmd, state)

        # 技能前的待命移动(绕到基座圈外、目标侧)：移完再发技能请求
        if self._pre_mover is not None:
            q_des, grip = self._pre_mover.step(state, self._dt)
            self._set_status("pre-move")
            if self._pre_mover.done:
                self._pre_mover = None
                if self._transit_stage == "retract":
                    # 已抬离刚操作的抽屉 -> 现在用 cuRobo 规划到 pre-grasp
                    req = self._pending_after_move
                    if self._start_curobo_to_pregrasp(req, state):
                        self._transit_stage = "curobo"
                        return self.provider.make_hold_joint_action(state, grip)
                    # cuRobo 规划失败 -> 直接起技能(退回原行为)
                    self._transit_stage = None
                    self._transited_req = req
                    self._pending = req
                    self._pending_after_move = None
                    return self.provider.make_hold_joint_action(state, grip)
                self._transited_req = self._pending_after_move   # 手动途径点绕行完，别再重复绕行
                self._pending = self._pending_after_move
                self._pending_after_move = None
                return self.provider.make_hold_joint_action(state, grip)
            if q_des is None:
                return self.provider.make_hold_joint_action(state, grip)
            return self.provider.make_joint_action_from_q_des(q_des, grip)

        # 抓取任意保存 pose（开环执行器）优先于普通 skill 执行器
        if self._grasp_runner is not None:
            q_des, grip = self._grasp_runner.step(state, self._dt)
            self._set_status(f"grasp_pose / {self._grasp_runner.phase}")
            if self._grasp_runner.done:
                self._grasp_runner = None
                return self.provider.make_hold_joint_action(state, grip)
            if q_des is None:
                return self.provider.make_hold_joint_action(state, grip)
            return self.provider.make_joint_action_from_q_des(q_des, grip)

        # 放置进篮子（开环执行器）
        if self._place_runner is not None:
            q_des, grip = self._place_runner.step(state, self._dt)
            self._set_status(f"place / {self._place_runner.phase}")
            if self._place_runner.done:
                self._place_runner = None
                return self.provider.make_hold_joint_action(state, grip)
            if q_des is None:
                return self.provider.make_hold_joint_action(state, grip)
            return self.provider.make_joint_action_from_q_des(q_des, grip)

        # 抽屉队列(打开两个/关闭所有)：上一个技能结束(executor 空闲)后，隔 gap 帧自动发下一个。
        if self._drawer_queue and self._pending is None and self.executor.active_skill is None \
                and self._grasp_runner is None and self._place_runner is None and self._pre_mover is None:
            if self._drawer_gap > 0:
                self._drawer_gap -= 1
                return self.provider.make_hold_joint_action(state, 1.0)
            skill_t, nxt = self._drawer_queue.pop(0)
            self.sel_drawer = nxt
            self._pending = self._make_request(skill_t)
            self._drawer_gap = _AUTORUN_SETTLE_STEPS   # 下一个抽屉前留稳定间隔
            print(f"[SkillTestController] drawer-queue: start {skill_t.value} {nxt} "
                  f"(remaining {[d for _, d in self._drawer_queue]})", flush=True)
            self._set_status(f"{skill_t.value} / {nxt}")

        if self._pending is not None:
            req = self._pending
            # 起接触技能前：若该技能配了途径点且【还没绕行过】，先按途径点绕行(避开已打开的抽屉等)，
            # 绕完(_pre_mover.done)会把该 req 标记 _transited_req 再回到这里真正 start。
            if req is not self._transited_req:
                # (1) 途径点绕行。首次把内置 home+facing 预填成面板可编辑途径点;之后用面板/存盘的值。
                skill_id = self._skill_id_for(req)
                wps = self._load_skill_waypoints(skill_id)
                if not wps and skill_id:   # 首次:内置 home+facing 预填为途径点(有面板就写进面板供编辑)
                    defaults = self._default_waypoints_for(req)
                    if defaults:
                        wps = defaults
                        if self._wp_panel is not None:
                            self._wp_panel.waypoints[skill_id] = defaults
                            try:
                                if getattr(self._wp_panel, "sel_skill", None) == skill_id:
                                    self._wp_panel._rebuild_wp_combo()
                            except Exception:
                                pass
                        print(f"[SkillTestController] prefilled {len(defaults)} default waypoints (home+facing) "
                              f"for {skill_id}", flush=True)
                if wps:
                    self._pending = None
                    # 有途径点 -> 途径点【完全定义过渡】(抬升/转身/绕行全由你摆),技能跳过内置 PRELIFT/ARC,
                    # 从 MOVE_TO_PRE_GRASP 起。转身角大的柜子途径点里要含 joint 点才真转底座(纯 task 会仰身)。
                    try:
                        req.parameters = {**(req.parameters or {}), "start_at_pregrasp": True}
                    except Exception:
                        pass
                    self._pre_mover = _WaypointMover(self.adapter, wps, device=self.device)
                    self._pending_after_move = req
                    print(f"[SkillTestController] transit via {len(wps)} waypoint(s) before {skill_id}", flush=True)
                    return self.provider.make_hold_joint_action(state, 1.0)
                # (2) 否则 cuRobo 无碰撞过渡：先竖直抬离(retract)刚操作的抽屉(否则起点在"开抽屉障碍盒"里、
                #     规划不出),再从安全位规划到 pre-grasp,技能随后从 APPROACH 短距滑上把手。
                pregrasp = self._skill_pregrasp_pose(req)
                transit = self._ensure_transit() if pregrasp is not None else None
                if transit is not None and pregrasp is not None:
                    self._pending = None
                    self._curobo_pregrasp = pregrasp
                    self._pre_mover = _WaypointMover(self.adapter, [self._retract_waypoint(state)],
                                                     device=self.device)
                    self._pending_after_move = req
                    self._transit_stage = "retract"
                    print(f"[SkillTestController] cuRobo transit: retract then plan -> {self._skill_id_for(req)}",
                          flush=True)
                    return self.provider.make_hold_joint_action(state, 1.0)
            self._pending = None
            self._transited_req = None
            self._prepare_joint_for(req)  # free targeted drawer joint (door frees itself)
            self.executor.start(req, state)

        # nothing active and not holding -> hand control back to the UI / default hold
        if self.executor.active_skill is None and self.executor.held_object is None:
            self._set_status(self.executor.runtime_status)
            self._draw_target_markers(None)
            return None

        command = self.executor.step(state, self._dt)
        self._set_status(f"{self.executor.current_state_name} / {self.executor.runtime_status}")
        self._draw_target_markers(self.executor.active_skill)
        return self._command_to_action(command, state)

    def _draw_target_markers(self, skill) -> None:
        """Live-visualize the active skill's phase target + handle pose as small coordinate frames."""
        mk = self._ensure_markers()
        if mk is None:
            return
        try:
            if skill is not None and hasattr(skill, "viz_poses"):
                vp = skill.viz_poses()
                if vp:
                    mk.show([p for _, p in vp], names=[n for n, _ in vp],
                            use_arrows=True, axis_length=0.06)
                    return
            mk.clear()
        except Exception as exc:  # pragma: no cover
            print(f"[SkillTestController] marker draw failed: {exc}", flush=True)

    def _ensure_markers(self):
        if self._markers is None:
            try:
                from franka_v1_skill_lab.scene_interface.target_markers import TargetPoseMarkers

                headless = bool(getattr(self.s.config, "headless", False))
                self._markers = TargetPoseMarkers(enabled=not headless, prefix="skilltest")
            except Exception:
                self._markers = False
        return self._markers or None

    # --------------------------------------------------------------- autorun
    def _auto_step(self, state):
        # done with the whole sequence
        if self._auto_i >= len(self._auto):
            if not self._auto_done_logged:
                ok = sum(1 for r in self._auto_results if r["success"])
                print(f"[AUTORUN] DONE: {ok}/{len(self._auto_results)} skills succeeded", flush=True)
                for r in self._auto_results:
                    print(
                        f"[AUTORUN]   {r['skill']}:{r['target']} -> "
                        f"{'OK' if r['success'] else 'FAIL'} ({r['status']}) "
                        f"angle={r['angle']} reason={r['reason']}",
                        flush=True,
                    )
                self._auto_done_logged = True
            return None

        # settle gap between skills (hold)
        if self._settle_left > 0:
            self._settle_left -= 1
            return None

        st, target = self._auto[self._auto_i]

        # start the current skill once
        if not self._auto_started:
            self._cur_skill, self._cur_target = st, target
            req = self._make_autorun_request(st, target)
            self._prepare_joint_for(req)
            print(f"[AUTORUN] start {st.value}:{target} ({self._auto_i + 1}/{len(self._auto)})", flush=True)
            self.executor.start(req, state)
            self._auto_started = True
            self._auto_steps = 0
            if self.executor.active_skill is None:  # immediate failure on start
                self._finish_auto(state)
                return None

        # drive the active skill
        if self.executor.active_skill is None:
            self._finish_auto(state)
            return None

        self._auto_steps += 1
        if self._auto_steps > _AUTORUN_MAX_STEPS:
            print(f"[AUTORUN] TIMEOUT {st.value}:{target} after {self._auto_steps} steps", flush=True)
            self.executor.reset()
            self._finish_auto(state, timeout=True)
            return None

        command = self.executor.step(state, self._dt)
        action = self._command_to_action(command, state)
        if self.executor.active_skill is None:  # skill finished this frame
            self._finish_auto(state)
        return action

    def _finish_auto(self, state, timeout: bool = False):
        st, target = self._cur_skill, self._cur_target
        res = self.executor.last_result
        success = bool(getattr(res, "success", False)) and not timeout
        status = "TIMEOUT" if timeout else (getattr(getattr(res, "final_status", None), "value", "?"))
        reason = getattr(res, "failure_reason", "") or ""
        self._auto_results.append(
            {
                "skill": st.value,
                "target": target,
                "success": success,
                "status": status,
                "reason": reason,
                "angle": self._relevant_angle(st, target),
            }
        )
        print(
            f"[AUTORUN] result {st.value}:{target} -> {'OK' if success else 'FAIL'} "
            f"({status}) angle={self._relevant_angle(st, target)} reason={reason}",
            flush=True,
        )
        self._auto_i += 1
        self._auto_started = False
        self._settle_left = _AUTORUN_SETTLE_STEPS

    def _relevant_angle(self, st: SkillType, target: str | None):
        """Read the joint angle (deg for the door, m for drawers) relevant to the skill, for logging."""
        if st in (SkillType.OPEN_DOOR, SkillType.CLOSE_DOOR):
            # read the door's configured joint (microwave=joint_0, fridge=joint_1) on the scene member
            try:
                from runtime.microwave_door_config import DOOR_TARGETS

                dcfg = DOOR_TARGETS.get(target or self.sel_door, {})
                asset = dcfg.get("asset_name", "microwave")
                joint = dcfg.get("joint_name", "joint_0")
            except Exception:
                asset, joint = "microwave", "joint_0"
            a = self._joint_angle(asset, joint)
            return None if a is None else round(a * 57.2958, 2)  # rad -> deg
        if st in (SkillType.OPEN_DRAWER, SkillType.CLOSE_DRAWER):
            try:
                jn = joint_name_for(target or "middle_drawer")
            except Exception:
                return None
            a = self._joint_angle("cabinet", jn)
            return None if a is None else round(a, 4)  # prismatic metres
        return None

    def _joint_angle(self, asset_name: str, joint_name: str):
        try:
            a = self.provider.scene[asset_name]
            ids, _ = a.find_joints(joint_name)
            return float(a.data.joint_pos[0, ids[0]].detach().cpu())
        except Exception:
            return None

    def _make_autorun_request(self, st: SkillType, target: str | None) -> SkillRequest:
        # set selections so _make_request picks them up, then reuse the same builder as the UI path
        if st == SkillType.GRASP and target:
            self.sel_grasp = target
        elif st in (SkillType.OPEN_DRAWER, SkillType.CLOSE_DRAWER) and target:
            self.sel_drawer = target
        elif st in (SkillType.OPEN_DOOR, SkillType.CLOSE_DOOR) and target:
            self.sel_door = target
        return self._make_request(st)

    # --------------------------------------------------------------- internals
    def _make_request(self, skill_type: SkillType) -> SkillRequest:
        stamp = time.time_ns()
        if skill_type == SkillType.PLACE:
            return SkillRequest(
                request_id=f"{skill_type.value}_{stamp}",
                skill_type=skill_type,
                source_object=None,  # executor fills from held_object
                destination_type="point",
                destination_object="point_a",
                parameters={
                    "target_frame": "env_local",
                    "target_surface_xyz": list(_DEFAULT_PLACE_XYZ),
                },
            )
        if skill_type in (SkillType.OPEN_DRAWER, SkillType.CLOSE_DRAWER):
            params = {"drawer_link": "link_1"}
            # Bug1: 点技能后先竖直抬到 home 高度(UI 可调)再去把手前方，避免夹爪斜扫物体。
            try:
                params["prelift_z"] = self._carry_z()
            except Exception:
                pass
            # 把 Grasp Pose 面板里【实时编辑】的该抽屉把手抓取位姿(link 局部系)传给技能，
            # 让开/关抽屉技能在你定义的 pose 上抓把手(随抽屉滑动跟踪)，而不是用硬编码把手配置。
            panel = self._grasp_panel
            hname = "handle_" + self.sel_drawer   # e.g. handle_top_drawer
            gl = getattr(panel, "grasp_local", {}) if panel is not None else {}
            if hname in gl:
                g = gl[hname]
                params["override_grasp_local"] = {"pos": list(g["pos"]), "quat": list(g["quat"])}
            return SkillRequest(
                request_id=f"{skill_type.value}_{self.sel_drawer}_{stamp}",
                skill_type=skill_type,
                source_object=None,
                destination_type="drawer",
                destination_object=self.sel_drawer,
                parameters=params,
            )
        if skill_type in (SkillType.OPEN_DOOR, SkillType.CLOSE_DOOR):
            return SkillRequest(
                request_id=f"{skill_type.value}_{self.sel_door}_{stamp}",
                skill_type=skill_type,
                source_object=None,
                destination_type="door",
                destination_object=self.sel_door,
            )
        # GRASP
        return SkillRequest(
            request_id=f"{skill_type.value}_{self.sel_grasp}_{stamp}",
            skill_type=skill_type,
            source_object=self.sel_grasp,
        )

    def _prepare_joint_for(self, req: SkillRequest) -> None:
        """Free the targeted cabinet drawer joint at runtime (pure-physical pull/push).

        Door skills free their own hinge in DoorIKSkill._setup_door_free, so we only handle drawers
        here. This writes ONLY the actuator stiffness/damping of the one targeted joint; it is never
        persisted (no scene save).
        """
        if req.skill_type not in (SkillType.OPEN_DRAWER, SkillType.CLOSE_DRAWER):
            return
        target = req.destination_object or "middle_drawer"
        try:
            joint_name = joint_name_for(target)
            member = member_for(target)   # 旧 cabinet 或白柜 sektion_cabinet
        except Exception as exc:
            print(f"[SkillTestController] WARN: cannot resolve drawer joint: {exc}", flush=True)
            return
        self._free_joint(member, joint_name, damping=_DRAWER_FREE_DAMPING)

    def _manage_drawer_damping(self) -> None:
        """抽屉关节阻尼随当前技能阶段切换：PULL/PUSH 用低阻尼(机器人能拉动)，其余用高阻尼(松爪/退回
        时抵抗拖拽、抽屉停在开到的位置)。stiffness 始终 0(无弹簧)。只在阻尼值变化时写一次。"""
        sk = getattr(self.executor, "active_skill", None)
        st = getattr(getattr(sk, "runtime", None), "state", None) if sk is not None else None
        want = _DRAWER_FREE_DAMPING if st in ("PULL", "PUSH") else _DRAWER_HOLD_DAMPING
        if want == getattr(self, "_cur_drawer_damping", None):
            return
        try:
            import torch
            cab = self.provider.scene["cabinet"]
            ids, _ = cab.find_joints("joint_[0-2]")
            n = cab.num_instances
            cab.write_joint_damping_to_sim(
                torch.full((n, len(ids)), float(want), device=cab.device), joint_ids=ids)
            self._cur_drawer_damping = want
        except Exception as exc:  # pragma: no cover
            print(f"[SkillTestController] manage drawer damping failed: {exc}", flush=True)

    def _free_joint(self, asset_name: str, joint_name: str, damping: float = 2.0) -> None:
        if (asset_name, joint_name) in self._freed:
            return
        try:
            asset = self.provider.scene[asset_name]
            ids, _ = asset.find_joints(joint_name)
            n = asset.num_instances
            dev = asset.device
            asset.write_joint_stiffness_to_sim(
                torch.zeros((n, len(ids)), device=dev), joint_ids=ids
            )
            asset.write_joint_damping_to_sim(
                torch.full((n, len(ids)), float(damping), device=dev), joint_ids=ids
            )
            self._freed.add((asset_name, joint_name))
            print(
                f"[SkillTestController] freed {asset_name}/{joint_name} (stiffness=0, damping={damping}) "
                "-- runtime only, not saved",
                flush=True,
            )
        except Exception as exc:
            print(f"[SkillTestController] WARN: could not free {asset_name}/{joint_name}: {exc}", flush=True)

    def _command_to_action(self, command, state):
        """Mirror skill_test_ui_joint._command_to_action for the joint backend."""
        if command.control_mode == "joint":
            if command.raw_joint_action is not None:
                return self.provider.make_joint_action_from_raw(command.raw_joint_action)
            if command.joint_target is not None:
                return self.provider.make_joint_action_from_q_des(
                    command.joint_target, command.gripper_command
                )
            return self.provider.make_hold_joint_action(state, command.gripper_command)
        # ik_pose fallback (should not happen in joint mode)
        return self.provider.make_hold_joint_action(state, command.gripper_command)

    def _monitor_collisions(self, state) -> None:
        """Log when the robot BODY (arm/wrist/hand, not the fingers) touches something. If it happens
        while NOT holding an object and the arm IS moving, that is almost certainly an accidental body
        bump (the door-knock hypothesis), so it is flagged loudly. Logs only on change (no per-frame spam).
        """
        if not self.collision_monitor.available:
            return
        contacts = self.collision_monitor.contacts(body_only=True)
        keyset = frozenset(n for n, _ in contacts)
        if keyset == self._last_collision_set:
            return
        self._last_collision_set = keyset
        if not contacts:
            return
        holding = self.executor.held_object is not None
        try:
            vel = state.robot.joint_vel[self.provider._arm_joint_ids]
            moving = float(torch.linalg.norm(vel)) > 0.05
        except Exception:
            moving = True
        detail = ", ".join(f"{n}={m:.1f}N" for n, m in contacts)
        if not holding and moving:
            msg = f"ACCIDENTAL BODY COLLISION (no object held, arm moving): {detail}"
        elif holding:
            msg = f"body contact while holding (carrying/placing?): {detail}"
        else:
            msg = f"body contact (arm ~stationary): {detail}"
        print(f"[CollisionMonitor] {msg}", flush=True)
        self._set_status("COLLISION: " + detail)

    def _set_status(self, text: str) -> None:
        if self._status_label is not None:
            self._status_label.text = "status: " + text
