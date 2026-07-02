"""IK-based OPEN/CLOSE microwave-door skills (revolute door, arc trajectory).

General, state-machine-callable door skills that mirror the open/close DRAWER skills, but follow a
circular ARC about the door hinge instead of a straight pull/push. The state machine only provides
the target door name (e.g. "microwave"); the hinge position/axis, the live handle pose, and the door
angle are all read live from the articulation, so the same skill works for any revolute door.

Geometry (calibrated, microwave_door_config):
  * door = articulation body ``link_0``; link_0's body origin IS the hinge -> hinge_w = link_0 pos_w.
  * hinge axis (world) = R(link_0_quat) @ local_Y  (= world +Z for the microwave: a vertical door).
  * handle (graspable free edge) = combine_frame_transforms(link_0 pos_w, link_0 quat_w, offset).
  * door outward face normal (world) = R(link_0_quat) @ (0,0,-1).

Opening physically rotates the handle about the hinge so the door joint angle increases (validated:
on this microwave joint+ corresponds to rotating the handle about world -Z). The door is opened ONLY
by physical interaction; the door joint target is never commanded.

Output: joint commands (q_des from the encapsulated DLS IK) for the joint-position env.

ASSET CAVEAT: the current microwave asset's door/body collision hulls overlap, so the door physically
opens only ~8-10deg before contact stops it (open_success_angle is set accordingly). The skill logic
is general; raise the threshold once the asset collision is fixed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

import isaaclab.utils.math as math_utils

from runtime.base_skill import SkillCommand, PoseState, get_speed_scale, pose_error, pose_tensor, step_pose
from runtime.microwave_door_config import DOOR_TARGETS
from runtime.scene_state_provider import SceneState
from runtime.skill_request import SkillRequest
from runtime.skill_result import SkillResult
from runtime.skill_types import ExecutionStatus, FailureReason, SkillType


@dataclass
class DoorIKConfig:
    """All tunable parameters for the open/close revolute-door skill (one place to retune).

    The phase order is: MOVE_TO_HOME -> MOVE_TO_PRE_GRASP -> APPROACH -> CLOSE_GRIPPER -> SWEEP ->
    VERIFY -> RELEASE -> RETREAT -> SUCCEEDED. Each timeout/threshold below belongs to one phase.
    """

    # --- start behaviour ---
    start_from_current: bool = True     # True: begin at MOVE_TO_STANDOFF using the CURRENT arm pose (no
    #                                     detour home). False: legacy -- first drive the arm back to its
    #                                     default home joints for a clean IK seed, then approach.
    use_turn_to_face: bool = True       # FIRST rotate joint 1 so the arm FACES the door, then approach.
    #                                     The local DLS IK won't turn the base on its own -> for a side/rear
    #                                     door it contorts ("leans back") and often can't reach. Pre-facing
    #                                     gives a natural full-reach posture.
    face_joint_threshold: float = 0.12
    face_timeout: float = 5.0
    use_grasp_seed: bool = False        # door reach is wrist-limited; the INITIAL redundant arm posture
    #                                     decides how wide the single-grasp pull gets (sweep: home 54deg
    #                                     vs this seed 79deg). Pre-position to home + these offsets so the
    #                                     grasp inherits a wrist branch that routes around the singularity.
    grasp_seed_offsets_deg: dict = field(default_factory=lambda: {0: 30.0, 2: -40.0})  # {arm_joint_idx: deg}
    seed_joint_threshold: float = 0.15  # rad: reached the seed posture
    seed_timeout: float = 5.0
    # --- pre-grasp / approach geometry ---
    approach_mode: str = "face_on"      # "face_on" (into door face from robot side; correct vertical-bar grasp) | "radial" (side hook)
    standoff_distance: float = 0.20     # MOVE_TO_STANDOFF: first reach this far IN FRONT of the handle
    #                                     (along the door-normal approach line) and ARRIVE ALREADY
    #                                     ALIGNED with the grasp orientation; only then move straight in.
    #                                     Prevents the wrist/forearm from clipping the door on the way in.
    standoff_ori_threshold: float = math.radians(12.0)  # must be this aligned before advancing inward
    pre_grasp_clearance: float = 0.08   # back-off along the approach line before grasping (MOVE_TO_PRE_GRASP)
    grasp_depth: float = 0.0            # APPROACH/CLOSE aim this far PAST the handle (into the door along
    #                                     the approach) so the bar seats deep between the fingers and is
    #                                     actually clamped -- the gripper was stopping ~2cm short and
    #                                     closing on empty air (gw=0) instead of gripping the bar.
    grasp_roll_deg: float = 180.0       # roll of the grasp frame about its approach axis (TCP +Z).
    #                                     Flips the Franka wrist branch (joint6 above vs below joint7).
    #                                     180 -> "6 above 7" (works: catches the bar, opens ~48deg). 0 ->
    #                                     TCP +X up, but the full close then MISSES the bar (door 0deg)
    #                                     because the gripper can't fully seat on the bar at this pose.
    approach_pitch_deg: float = 0.0     # downward tilt of the approach. 0 for face_on -> TCP +Z is
    #                                     HORIZONTAL straight into the door and the bar-axis (TCP +X) is
    #                                     exactly vertical so the fingers straddle the vertical bar.
    arc_lead_deg: float = 6.0           # how far ahead of the live door angle to aim while sweeping (the
    #                                     pull lead, applied to position AND orientation together so the
    #                                     grip stays rigid). Small: just enough to drag the freed hinge;
    #                                     too large slides the gripper ahead of the bar (25deg opened LESS).
    # --- motion limits (scaled globally by base_skill.set_speed_scale) ---
    max_pos_step: float = 0.020
    max_ori_step: float = math.radians(6.0)
    null_space_gain: float = 0.0        # redundancy resolution during IK (bias toward joint centers, TCP
    #                                     unchanged). DISABLED: the 1-D null space here only relieves j5,
    #                                     not the j6 bottleneck, and hurt reach. Kept as a tunable hook.
    # --- reach / settle ---
    reach_pos_threshold: float = 0.03
    reach_stable_cycles: int = 5
    close_duration: float = 1.0         # CLOSE_GRIPPER dwell so the fingers seat on the bar
    # --- per-phase timeouts (s) ---
    reach_timeout: float = 16.0         # MOVE_TO_PRE_GRASP / APPROACH
    sweep_timeout: float = 35.0         # SWEEP (longer: allow slowly dragging the door to its max)
    home_joint_threshold: float = 0.12
    home_timeout: float = 6.0
    # --- verify-open / release / retreat (added) ---
    verify_stable_cycles: int = 5       # door angle must stay past threshold this many steps
    verify_timeout: float = 4.0         # VERIFY budget before giving up the confirmation
    settle_before_release: float = 0.6  # after the pull plateaus, HOLD the door gripped this long so its
    #                                     swing inertia damps out BEFORE the fingers open (else the freed
    #                                     door keeps coasting when released)
    release_duration: float = 0.6       # RELEASE dwell with the gripper opening off the handle
    retreat_distance: float = 0.14      # how far to back the TCP off the handle (along -approach) in RETREAT
    skip_retreat: bool = False          # True: SUCCEED at RELEASE (arm left at the open handle) and let
    #                                     an external cuRobo planner retreat collision-free (no door clip)
    retreat_timeout: float = 8.0        # RETREAT budget; on timeout we still SUCCEED (door is already open)
    # --- collision-free egress (avoid the hand sweeping back THROUGH the open door) ---
    retreat_to_home: bool = True        # after clearing the door, drive the arm back to default home joints
    retreat_clear_distance: float = 0.22  # RETREAT_CLEAR: first back the open gripper this far horizontally
    #                                       TOWARD THE ROBOT BASE -- out of the door's swing region -- before
    #                                       homing, so the wrist/hand does not knock the open door shut.
    # --- DRIVE-ASSIST: open the door by APPLYING A DRIVING FORCE to the hinge joint ---
    # The pure IK grasp-pull is sensitive to the gripper exactly reaching the handle. With drive-assist
    # the skill instead gives the door joint a strong implicit PD actuator and commands the open angle,
    # so the hinge is physically DRIVEN open (collisions respected) while the gripper does a best-effort
    # approach. This is the "apply driving force" path; set False for the pure-physical-interaction mode.
    drive_door_joint: bool = False        # OFF: open/close the door PURELY by physical gripper contact
    #                                       (no joint drive). True only for debugging the mechanism.
    open_target_deg: float = 88.0         # ideal OPEN angle; the gripper keeps PULLING toward it until
    #                                       it either reaches it or the door angle PLATEAUS (gripper hit
    #                                       its pull limit) -- so the robot drags the door to its physical
    #                                       max while still gripping, NOT releasing early to coast.
    pull_to_max: bool = True              # keep gripping + pulling until the door angle plateaus
    pull_plateau_deg: float = 0.5         # if the door gains < this over the window, treat as plateaued
    pull_plateau_steps: int = 100         # consecutive no-progress steps -> door is at its pulled max
    #                                       (raised: a slowly-but-steadily opening door was cut off early)
    pull_min_open_deg: float = 8.0        # don't accept a plateau as success below this (means no grip)
    # --- staged RE-GRIP to beat the single-grasp wrist limit (open wide doors) ---
    max_regrips: int = 0                  # DISABLED: re-approaching from the partly-open door gives a
    #                                       WORSE grip config and the free door drifts back while released,
    #                                       so re-grip spiralled the angle DOWN (48->26->12deg). A single
    #                                       clean grasp+pull reaches farther; keep the hook for future use.
    regrip_below_deg: float = 80.0        # (only used when max_regrips>0) re-grip while below this angle
    regrip_below_deg: float = 80.0        # only re-grip while the door is still below this angle
    free_damping: float = 0.1             # pure-physical: free the hinge (stiffness 0) with this LOW
    #                                       damping so the gripper can swing it. This is NOT a drive
    #                                       torque (no position/velocity target) -- just low friction.
    door_drive_stiffness: float = 400.0   # door-joint P gain (drive torque)
    door_drive_damping: float = 40.0
    door_open_drive_deg: float = 75.0     # commanded open angle when driving (door physically reaches ~here)
    door_drive_ramp_s: float = 4.0        # time to ramp the commanded angle 0 -> target (smooth swing)
    soft_reach_advance: bool = True       # in drive mode, advance past a reach timeout instead of failing
    pure_soft_reach_advance: float = 0.06 # pure-physical mode: advance on reach timeout if within this of
    #                                       the handle (the bar is grippable over a few cm; lets the grip
    #                                       + pull engage instead of failing a couple cm short)
    drive_reach_timeout: float = 6.0      # shorter best-effort reach budget per phase when driving
    # --- PULL -> PUSH switch ---
    # Pulling the handle becomes kinematically awkward once the door has swung out a fair bit (the free
    # edge moves away from the robot). After ``push_switch_angle_deg`` the skill stops pulling and instead
    # PUSHES the door's INNER face near the free edge along the opening tangent, which keeps the arm in a
    # comfortable pose while the door finishes opening.
    push_after_pull: bool = False
    push_switch_angle_deg: float = 20.0   # door-open angle at which to STOP pulling (release + clear)
    push_standoff: float = 0.04           # gripper stands this far on the inner side of the free edge
    push_timeout: float = 10.0
    # PULL -> (release + clear retreat) -> PUSH. Going straight from the outside handle to the inside
    # face scrapes the gripper on the door frame, so at the switch angle we release, back the gripper
    # well clear of the door (away along -approach) and let the driven door keep swinging open, THEN
    # come around to push the inner face.
    clear_retreat_distance: float = 0.16  # how far to back the gripper off the door when clearing
    coast_duration: float = 5.0           # time to stay cleared (gripper open) while the door swings on


@dataclass
class _Runtime:
    state: str = "IDLE"
    start_time: float = 0.0
    state_start_time: float = 0.0
    stable_count: int = 0
    door_control_mode: str = "ik_arc"
    door_joint_name: str = ""
    drawer_joint_target: float | None = None  # never set (physical interaction only)
    initial_angle: float = 0.0
    current_angle: float = 0.0
    last_command_pose: PoseState | None = None
    last_gripper: float = 1.0
    verify_passed: bool = False
    drive_start_time: float = -1.0  # when the door-joint drive ramp started (persists across phases)
    max_angle: float = 0.0          # best (max) door-open angle seen while pulling (for plateau detection)
    peak_angle: float = 0.0         # true running peak open angle (for the relax-tolerant success check)
    plateau_count: int = 0          # consecutive steps with no further opening progress
    regrip_count: int = 0           # how many times we have released + re-approached to beat the wrist limit
    final_error_pos: float | None = None
    final_error_ori: float | None = None
    last_failure_message: str | None = None
    history: list[dict] = field(default_factory=list)


def _round_vec(vec: torch.Tensor, n: int = 4) -> list:
    """Round a 3-vector to a plain python list for debug logging."""
    return [round(float(v), n) for v in vec]


def _rotate_about_z(vec: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate a 3-vector about world +Z by ``angle`` rad (z component unchanged)."""
    c, s = math.cos(angle), math.sin(angle)
    x = c * vec[0] - s * vec[1]
    y = s * vec[0] + c * vec[1]
    return torch.stack((x, y, vec[2]))


def _grasp_quat_vertical_edge(approach: torch.Tensor, device) -> torch.Tensor:
    """TCP quaternion for grasping a VERTICAL handle bar, given the (horizontal) approach direction.

    approach (TCP +Z) = the given direction (gripper moves along it onto the bar);
    bar axis (TCP +X) = world up (the free edge is vertical);
    finger-open axis (TCP +Y) = horizontal, so the fingers straddle the bar and their closing
    contact normal lies in the horizontal plane (it can push the door along the opening tangent).

    We use a FORWARD-ish approach (toward the hinge along the door radial) rather than a face-on +Y
    approach, because the Franka reaches a forward-pointing gripper far more comfortably.
    """
    up = torch.tensor([0.0, 0.0, 1.0], device=device)
    z = approach / torch.linalg.norm(approach)
    x = up - torch.dot(up, z) * z  # up component perpendicular to approach -> bar runs vertical
    nx = torch.linalg.norm(x)
    if float(nx) < 1e-6:
        x = torch.tensor([1.0, 0.0, 0.0], device=device)
    else:
        x = x / nx
    y = torch.linalg.cross(z, x)
    R = torch.stack((x, y, z), dim=1)
    return math_utils.quat_from_matrix(R.unsqueeze(0))[0]


class DoorIKSkill:
    """Shared open/close revolute-door skill; subclasses set the sweep direction + success test."""

    backend = "ik_arc"
    is_open = True

    # Process-level cache of the proxy-calibrated handle offset per (asset, link); populated when the
    # door is closed (open_door) and reused while it is open (close_door / reopen). See
    # _calibrate_handle_offset_from_proxy.
    _HANDLE_OFFSET_CACHE: dict = {}

    def __init__(self, request: SkillRequest, env, ik_adapter, config: DoorIKConfig | None = None):
        self.request = request
        self.env = env
        self.adapter = ik_adapter
        self.cfg = config or DoorIKConfig()
        self.target_door = request.destination_object or "microwave"
        self.status = ExecutionStatus.IDLE
        self.failure_reason = FailureReason.NONE
        dcfg = DOOR_TARGETS.get(self.target_door, {})
        self.asset_name = dcfg.get("asset_name", "microwave")
        self.door_joint_name = dcfg.get("joint_name", "joint_0")
        self.door_link_name = dcfg.get("link_name", "link_0")
        self.handle_offset = torch.tensor(
            dcfg.get("handle_offset", (0.0, 0.0, 0.0)), dtype=torch.float32, device=ik_adapter.env.unwrapped.device
        )
        self.open_success_angle = dcfg.get("open_success_angle", 0.14)
        self.close_success_angle = dcfg.get("close_success_angle", 0.03)
        # Sign of the world-Z rotation that OPENS the door (asset-dependent: microwave opens via -Z,
        # the fridge via +Z). Read from the door config so the skill is not hard-wired to one asset.
        self.open_sweep_sign = float(dcfg.get("open_sweep_sign", -1.0))
        self.runtime = _Runtime(door_joint_name=self.door_joint_name)
        self.last_q = None
        self._asset = None
        self._door_idx = None
        self._joint_id = None
        self._ee_log_i = 0
        self.ee_log_every = 20  # periodic end-effector pose log (steps); 0 disables
        self._pull_quat = None  # wrist orientation frozen at grip-close, reused through the pull arc
        # Rigid-pull capture (set when the grip closes): the gripper then moves as a rigid body about
        # the hinge, so we remember the TCP pose + door angle + hinge at the instant of grip.
        self._grip_tcp_pos = None
        self._grip_tcp_quat = None
        self._grip_angle = 0.0
        self._grip_hinge = None
        self._grip_approach_dir = None
        self._last_phase_target = None  # current phase's TCP target (for live visualization)
        self._settle_pose = None        # frozen hold pose at the pulled-open max (SETTLE/RELEASE)
        self._retreat_clear_target = None  # captured-once horizontal clear target (RETREAT_CLEAR)

    @property
    def current_state(self) -> str:
        return self.runtime.state

    # ---- live scene reads -------------------------------------------------
    def _bind_asset(self):
        self._asset = self.env.unwrapped.scene[self.asset_name]
        bnames = list(self._asset.data.body_names)
        self._door_idx = next((i for i, n in enumerate(bnames) if n == self.door_link_name), None)
        if self._door_idx is None:
            self._door_idx = next((i for i, n in enumerate(bnames) if self.door_link_name in n), 0)
        jnames = list(self._asset.data.joint_names)
        self._joint_id = jnames.index(self.door_joint_name) if self.door_joint_name in jnames else 0

    def _resolve_handle_offset(self):
        """Decide the handle grasp offset for THIS door. Generic-first so swapping the door asset needs
        no manual calibration: auto-locate the handle from the door link's meshes; if that fails, fall
        back to a DoorHandleProxy prim (microwave), then to the config offset."""
        if self._auto_locate_handle():
            return
        self._calibrate_handle_offset_from_proxy()

    def _auto_locate_handle(self) -> bool:
        """GENERALIZATION: find the graspable handle straight from the door link's child meshes, with NO
        per-asset calibration. The handle is taken as the thin mesh whose center is FARTHEST (in the
        horizontal plane) from the vertical hinge axis -- i.e. the bar on the door's free edge. The door
        slab (largest-volume mesh) and sub-cm noise are excluded. The offset is computed in the door
        link body frame (rotation-invariant, so it stays valid as the door swings) and cached.

        Returns True if a handle was located (self.handle_offset set), else False (caller falls back).
        This is the "read the handle vs hinge-axis geometry from sim" path: a new revolute door with a
        free-edge handle works immediately once its joint/link names are known.
        """
        cache_key = (self.asset_name, self.door_link_name)
        cached = DoorIKSkill._HANDLE_OFFSET_CACHE.get(cache_key)
        if cached is not None:
            self.handle_offset = cached.to(self.handle_offset.device).clone()
            print(f"[DoorIKSkill] handle offset from cache: "
                  f"{[round(float(v),4) for v in self.handle_offset.tolist()]}", flush=True)
            return True
        try:
            from pxr import Usd, UsdGeom

            stage = self.env.unwrapped.sim.stage
            eid = self.adapter.env_id
            link_path = f"/World/envs/env_{eid}/{self.asset_name.capitalize()}/{self.door_link_name}"
            prim = stage.GetPrimAtPath(link_path)
            if not prim.IsValid():
                print(f"[DoorIKSkill] auto-locate: door link prim not found ({link_path})", flush=True)
                return False
            bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                  [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True)
            meshes = []
            for p in Usd.PrimRange(prim):
                if not (p.IsA(UsdGeom.Mesh) or p.IsA(UsdGeom.Gprim)):
                    continue
                rng = bb.ComputeWorldBound(p).ComputeAlignedRange()
                if rng.IsEmpty():
                    continue
                mn, mx = rng.GetMin(), rng.GetMax()
                size = (mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])
                ctr = ((mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, (mn[2] + mx[2]) * 0.5)
                meshes.append((ctr, size, size[0] * size[1] * size[2]))
            if not meshes:
                print("[DoorIKSkill] auto-locate: no meshes under door link", flush=True)
                return False
            link_pos, link_quat = self._link0()
            hx, hy = float(link_pos[0]), float(link_pos[1])

            def axis_dist(m):  # horizontal distance of a mesh center from the (vertical) hinge axis
                return ((m[0][0] - hx) ** 2 + (m[0][1] - hy) ** 2) ** 0.5

            slab_vol = max(m[2] for m in meshes)
            cand = [m for m in meshes if m[2] < slab_vol and max(m[1]) > 0.03]  # drop slab + sub-cm noise
            if not cand:
                cand = meshes
            handle = max(cand, key=axis_dist)               # farthest-from-hinge = the free-edge bar
            ctr = handle[0]
            handle_world = torch.tensor([float(ctr[0]), float(ctr[1]), float(ctr[2])],
                                        device=self.handle_offset.device)
            off, _ = math_utils.subtract_frame_transforms(
                link_pos.unsqueeze(0), link_quat.unsqueeze(0), handle_world.unsqueeze(0))
            self.handle_offset = off[0].clone()
            DoorIKSkill._HANDLE_OFFSET_CACHE[cache_key] = self.handle_offset.detach().cpu().clone()
            print(f"[DoorIKSkill] auto-located handle: world="
                  f"({float(ctr[0]):.3f},{float(ctr[1]):.3f},{float(ctr[2]):.3f}) "
                  f"dist_from_hinge_axis={axis_dist(handle):.3f} size={tuple(round(s,3) for s in handle[1])} "
                  f"-> link offset={[round(float(v),4) for v in self.handle_offset.tolist()]}", flush=True)
            return True
        except Exception as exc:  # pragma: no cover
            print(f"[DoorIKSkill] auto-locate handle failed ({exc}); falling back", flush=True)
            return False

    def _calibrate_handle_offset_from_proxy(self):
        """Recompute the handle offset (link_0 body frame) from the ACTUAL DoorHandleProxy prim.

        The config offset in microwave_door_config is world-metric at MICROWAVE_SCALE=0.35 and is applied
        WITHOUT rescaling, so it only matches when the microwave prim scale is exactly 0.35 uniform. The
        user can freely (non-uniformly) rescale the microwave in the layout editor, which moves the real
        handle but NOT the config offset -> the computed grasp point misses the bar (measured ~12.7 cm
        off at scale [0.30,0.25,0.30]). The DoorHandleProxy is a child of the scaled link, so its world
        pose always sits on the real bar. We read it once at start and back out the TRUE offset in the
        link_0 body frame; because the handle is rigid w.r.t. the door, that offset stays valid as the
        door swings. Falls back to the config offset if the proxy prim is absent.
        """
        cache_key = (self.asset_name, self.door_link_name)
        # Reuse a previously-calibrated offset if we have one (calibration is geometric & constant).
        cached = DoorIKSkill._HANDLE_OFFSET_CACHE.get(cache_key)
        # Only (re)calibrate when the door is near CLOSED: the proxy world pose (USD) and the link body
        # pose (physics) only agree when the door is at rest closed; once the door is swung open they
        # diverge by the door angle, so a live read would bake in that error. Door always starts closed
        # for open_door, which populates the cache that close_door / reopen then reuse.
        door_angle = abs(self._door_angle())
        if cached is not None and door_angle > 0.05:
            self.handle_offset = cached.to(self.handle_offset.device).clone()
            print(f"[DoorIKSkill] handle offset from cache (door open {door_angle:.2f}rad): "
                  f"{[round(float(v),4) for v in self.handle_offset.tolist()]}", flush=True)
            return
        try:
            from pxr import Usd, UsdGeom

            stage = self.env.unwrapped.sim.stage
            eid = self.adapter.env_id
            proxy_path = f"/World/envs/env_{eid}/{self.asset_name.capitalize()}/{self.door_link_name}/DoorHandleProxy"
            proxy_prim = stage.GetPrimAtPath(proxy_path)
            if not proxy_prim.IsValid():
                print(f"[DoorIKSkill] handle proxy not found ({proxy_path}); using config offset", flush=True)
                return
            m = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(proxy_prim)
            t = m.ExtractTranslation()
            proxy_w = torch.tensor([float(t[0]), float(t[1]), float(t[2])],
                                   dtype=torch.float32, device=self.handle_offset.device)
            # back out the offset in link_0's body frame (proxy world pos relative to the live link pose)
            link_pos, link_quat = self._link0()
            true_offset, _ = math_utils.subtract_frame_transforms(
                link_pos.unsqueeze(0), link_quat.unsqueeze(0), proxy_w.unsqueeze(0)
            )
            old = [round(float(v), 4) for v in self.handle_offset.tolist()]
            self.handle_offset = true_offset[0].clone()
            DoorIKSkill._HANDLE_OFFSET_CACHE[cache_key] = self.handle_offset.detach().cpu().clone()
            new = [round(float(v), 4) for v in self.handle_offset.tolist()]
            print(f"[DoorIKSkill] handle offset calibrated from proxy prim (door {door_angle:.2f}rad): "
                  f"config={old} -> actual={new}", flush=True)
        except Exception as exc:  # pragma: no cover
            print(f"[DoorIKSkill] WARN handle-proxy calibration failed ({exc}); using config offset", flush=True)

    def _door_pose(self):
        eid = self.adapter.env_id
        return self._asset.data.body_pos_w[eid], self._asset.data.body_quat_w[eid]

    def _link0(self):
        eid = self.adapter.env_id
        return (
            self._asset.data.body_pos_w[eid, self._door_idx],
            self._asset.data.body_quat_w[eid, self._door_idx],
        )

    def _hinge_pos(self) -> torch.Tensor:
        return self._link0()[0]

    def _door_angle(self) -> float:
        return float(self._asset.data.joint_pos[self.adapter.env_id, self._joint_id])

    def _handle_pos(self) -> torch.Tensor:
        link_pos, link_quat = self._link0()
        handle, _ = math_utils.combine_frame_transforms(
            link_pos.unsqueeze(0), link_quat.unsqueeze(0), self.handle_offset.unsqueeze(0)
        )
        return handle[0]

    def _faced_seed_q(self) -> torch.Tensor:
        """Home arm posture with joint 1 rotated to FACE the door handle (its azimuth in the robot base
        frame). Turning the base to face the target avoids the contorted 'lean back' the local DLS IK
        picks for a side/rear door, and usually fixes reachability."""
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
        print(f"[DoorIKSkill] turn-to-face: handle azimuth(base) = {math.degrees(azimuth):.1f}deg "
              f"-> joint1 = {math.degrees(float(seed[0])):.1f}deg", flush=True)
        return seed

    def _radial_dir(self) -> torch.Tensor:
        """Horizontal unit vector from the hinge to the handle (outward along the door)."""
        r = (self._handle_pos() - self._hinge_pos()).clone()
        r[2] = 0.0
        n = torch.linalg.norm(r)
        if float(n) < 1e-6:
            return torch.tensor([-1.0, 0.0, 0.0], device=r.device)
        return r / n

    def _door_into_dir(self) -> torch.Tensor:
        """Horizontal unit vector pointing INTO the door from the robot side (the face-on approach
        direction). The door's robot-facing front normal is +Y for the deployed microwave (the handle
        protrudes toward the robot), so moving onto/into the handle is the -Y direction. We derive it
        robustly as R(link_0_quat) @ (0,0,-1) (= world -Y here) and force it to point AWAY from the
        robot base so it generalizes if the microwave is re-yawed.
        """
        link_pos, link_quat = self._link0()
        n = math_utils.quat_apply(link_quat.unsqueeze(0), torch.tensor([[0.0, 0.0, -1.0]], device=link_quat.device))[0].clone()
        n[2] = 0.0
        m = torch.linalg.norm(n)
        if float(m) < 1e-6:
            return torch.tensor([0.0, -1.0, 0.0], device=link_quat.device)
        n = n / m
        # ensure it points away from the robot base (into the door, the way the gripper drives in)
        try:
            robot = self.env.unwrapped.scene["robot"]
            base = robot.data.root_pos_w[self.adapter.env_id]
            to_robot = (base - self._handle_pos()).clone()
            to_robot[2] = 0.0
            if float(torch.dot(n, to_robot)) > 0.0:
                n = -n
        except Exception:
            pass
        return n

    def _approach_dir(self) -> torch.Tensor:
        """Unit direction the TCP travels to seat on the handle (TCP +Z), pitched down by
        ``approach_pitch_deg`` for a natural Franka wrist.

        Two modes (``DoorIKConfig.approach_mode``):
          * ``"face_on"`` (default): approach straight into the door face from the robot side
            (-outward_normal). The pre-grasp stand-off then sits BETWEEN the robot and the handle, so
            the arm never has to reach across past the door's free edge -> far more reachable.
          * ``"radial"``: original geometry -- approach toward the hinge along the door radial. Kept
            for assets where a side hook is preferable; pre-grasp lands beyond the free edge.
        """
        p = math.radians(self.cfg.approach_pitch_deg)
        down = torch.tensor([0.0, 0.0, -1.0], device=self.handle_offset.device)
        if self.cfg.approach_mode == "radial":
            horiz = -self._radial_dir()
        else:  # face_on: drive straight into the door face from the robot side
            horiz = self._door_into_dir()
        approach = horiz * math.cos(p) + down * math.sin(p)
        return approach / torch.linalg.norm(approach)

    def _grasp_quat(self, approach: torch.Tensor, device=None) -> torch.Tensor:
        """Grasp orientation for the vertical bar, rolled about the approach axis by ``grasp_roll_deg``.

        The base frame has TCP +Z = approach, +X = world up (bar axis). Rolling about TCP +Z (the hand's
        own approach axis) by 180deg flips the Franka wrist to the 'joint6 above joint7' branch, which
        the operator finds natural and which has more joint range to follow the door open."""
        q = _grasp_quat_vertical_edge(approach, approach.device)
        roll = math.radians(getattr(self.cfg, "grasp_roll_deg", 0.0))
        if abs(roll) > 1e-6:
            dev = approach.device
            rz = math_utils.quat_from_angle_axis(
                torch.tensor([roll], device=dev), torch.tensor([[0.0, 0.0, 1.0]], device=dev)
            )
            q = math_utils.quat_mul(q.unsqueeze(0), rz)[0]  # body-frame roll about TCP +Z
        return q

    def _grasp_pose(self, clearance: float) -> PoseState:
        """Handle target with a forward+down approach grasp of the vertical free edge.

        The gripper approaches along ``_approach_dir`` (toward the hinge, pitched down); ``clearance``
        backs the pre-grasp off along -approach so the gripper comes in from the open / upper side.
        """
        handle = self._handle_pos()
        approach = self._approach_dir()
        quat = self._grasp_quat(approach, handle.device)
        return PoseState(handle - approach * clearance, quat)

    def _capture_grip(self, state: SceneState) -> None:
        """Snapshot the gripper pose + door angle + hinge at the instant the grip closes.

        After this, the gripper holds the handle, so the door + gripper form ONE rigid body that
        rotates about the (vertical) hinge. `_pull_target` reproduces exactly that rigid rotation."""
        self._grip_tcp_pos = state.robot.tcp_pose.pos_w.clone()
        self._grip_tcp_quat = state.robot.tcp_pose.quat_w.clone()
        self._grip_angle = self._door_angle()
        self._grip_hinge = self._hinge_pos().clone()
        self._grip_approach_dir = self._door_into_dir().clone()  # horizontal approach dir at grip (for level wrist)
        print(f"[DoorIKSkill] grip captured: tcp_pos={_round_vec(self._grip_tcp_pos,3)} "
              f"door={math.degrees(self._grip_angle):.1f}deg hinge={_round_vec(self._grip_hinge,3)}", flush=True)

    def _pull_target(self, lead_deg: float) -> PoseState:
        """Unified rigid-body pull pose (the correct kinematic model the door demands).

        Once the gripper grips the handle, the handle is a fixed point on the door, so as the door
        rotates about the vertical (world +Z) hinge by the joint travel since grip, the WHOLE captured
        TCP pose (position AND orientation) rotates rigidly about the hinge by that same angle. An
        extra ``lead_deg`` in the opening/closing direction keeps the commanded pose slightly ahead so
        the freed hinge is dragged along (continuous pull torque) without the gripper sliding on the
        bar. Replaces the old ``_arc_target`` which evolved position (arc) and orientation
        (live door normal) independently and so let the gripper slip / ride up-down the handle."""
        if self._grip_tcp_pos is None:
            return self._arc_target()  # not yet gripped -> fall back (shouldn't happen in SWEEP)
        travel = abs(self._door_angle() - self._grip_angle)            # door rotation magnitude since grip
        dev = self._grip_tcp_pos.device
        # ONE rigid rotation for BOTH position and orientation: the gripper is a rigid body clamped on the
        # bar, so it must rotate about the vertical hinge by the SAME angle as the door. We add a SMALL
        # consistent ``lead`` (applied to position AND orientation together) so the commanded pose sits
        # slightly ahead of the live door -> drags the freed hinge open WITHOUT twisting the gripper
        # relative to the bar. (The old code rotated position by travel+lead but orientation by travel
        # only; that 'lead' split slid/twisted the gripper off the handle and the error grew with travel.)
        rot = self._sweep_sign() * (travel + math.radians(lead_deg))
        rel = self._grip_tcp_pos - self._grip_hinge
        pos = self._grip_hinge + _rotate_about_z(rel, rot)               # height Z preserved -> no riding the bar
        # ORIENTATION: rotate the CAPTURED grip quaternion rigidly about the WORLD vertical axis by the
        # same `rot` (world-frame rotation -> pre-multiply). This keeps the gripper's orientation w.r.t.
        # the bar EXACTLY as gripped -- it does NOT rebuild/re-level the pose from the approach dir, which
        # spun the wrist about its own TCP-Z and slid it off the bar (the deviation the operator saw).
        rz = math_utils.quat_from_angle_axis(
            torch.tensor([rot], device=dev), torch.tensor([[0.0, 0.0, 1.0]], device=dev)
        )
        quat = math_utils.quat_mul(rz, self._grip_tcp_quat.unsqueeze(0))[0]
        return PoseState(pos, math_utils.normalize(quat.unsqueeze(0))[0])

    def _arc_target(self) -> PoseState:
        """[LEGACY fallback] Aim a lead angle ahead of the live door angle, along the arc.

        Superseded by `_pull_target` (rigid hinge rotation). Kept only as a pre-grip fallback."""
        hinge = self._hinge_pos()
        handle = self._handle_pos()
        lead = math.radians(self.cfg.arc_lead_deg) * self._sweep_sign()
        target_handle = hinge + _rotate_about_z(handle - hinge, lead)
        quat = self._grasp_quat(self._approach_dir(), handle.device)
        return PoseState(target_handle, quat)

    def _retreat_pose(self) -> PoseState:
        """Clear the (open) handle by lifting the open gripper straight UP off the vertical bar.

        The fingers are already open in RETREAT; lifting along world +Z slides them off the top of the
        bar with NO horizontal force, so the freed door is NOT pushed back closed. (Backing along
        -approach used to keep the fingers hooked on the handle and dragged the door shut.)"""
        handle = self._handle_pos()
        approach = self._approach_dir()
        quat = self._grasp_quat(approach, handle.device)
        up = torch.tensor([0.0, 0.0, 1.0], device=handle.device)
        return PoseState(handle + up * self.cfg.retreat_distance, quat)

    def _toward_base_pose(self, state: SceneState, distance: float) -> PoseState:
        """A pose ``distance`` from the current TCP, moved HORIZONTALLY toward the robot base (height +
        orientation held). Backs the open gripper out of the door's swing region toward the robot side
        before homing, so the hand does not sweep back through the open door."""
        tcp = state.robot.tcp_pose
        try:
            base = self.env.unwrapped.scene["robot"].data.root_pos_w[self.adapter.env_id]
        except Exception:
            base = torch.zeros(3, device=tcp.pos_w.device)
        d = (base - tcp.pos_w).clone()
        d[2] = 0.0
        n = torch.linalg.norm(d)
        d = d / n if float(n) > 1e-6 else torch.tensor([1.0, 0.0, 0.0], device=tcp.pos_w.device)
        return PoseState(tcp.pos_w + d * distance, tcp.quat_w.clone())

    def _open_tangent(self) -> torch.Tensor:
        """Horizontal unit vector along the door's OPENING direction at the free edge.

        Opening rotates the handle about world +Z by a negative angle (see _arc_target/_sweep_sign),
        so a free-edge point moves along d/dtheta(Rz(-theta) r) = (r_y, -r_x, 0)."""
        r = (self._handle_pos() - self._hinge_pos()).clone()
        r[2] = 0.0
        t = torch.stack((r[1], -r[0], torch.zeros((), device=r.device)))
        n = torch.linalg.norm(t)
        if float(n) < 1e-6:
            return torch.tensor([0.0, 1.0, 0.0], device=r.device)
        return t / n

    def _clear_pose(self) -> PoseState:
        """Released stand-off well clear of the door: back the (open) gripper off the current handle
        along -approach by ``clear_retreat_distance`` so it does not scrape the door frame while the
        door keeps swinging open, before coming around to push the inner face."""
        handle = self._handle_pos()
        approach = self._approach_dir()
        quat = self._grasp_quat(approach, handle.device)
        return PoseState(handle - approach * self.cfg.clear_retreat_distance, quat)

    def _push_pose(self) -> PoseState:
        """PUSH the door open from its INNER side near the free edge.

        Once pulling the handle is awkward, the gripper moves just behind the free edge (on the inner
        face, -tangent side) and pushes along the opening tangent (TCP +Z = push direction, TCP +X =
        vertical). Combined with the joint drive this finishes the swing without the arm over-reaching.
        """
        handle = self._handle_pos()
        tangent = self._open_tangent()
        quat = self._grasp_quat(tangent, handle.device)
        return PoseState(handle - tangent * self.cfg.push_standoff, quat)

    # ---- subclass hooks ---------------------------------------------------
    def _sweep_sign(self) -> float:
        """Sign of the world-Z rotation that drives the door in the desired direction.

        Asset-dependent: ``open_sweep_sign`` (from the door config) is the sign that OPENS the door
        (microwave opens via world -Z, the fridge via +Z). Closing uses the opposite sign.
        """
        return self.open_sweep_sign if self.is_open else -self.open_sweep_sign

    def _reached_goal(self) -> bool:
        if self.is_open:
            # accept the PEAK angle reached, not just the live one: a freed door relaxes a few degrees
            # after the pull hits its max, which would otherwise drop it back below the success threshold
            # and fail VERIFY even though the door is clearly open.
            return max(self.runtime.current_angle, self.runtime.peak_angle) >= self.open_success_angle
        return self.runtime.current_angle <= self.close_success_angle

    # ---- skill API --------------------------------------------------------
    def start(self, state: SceneState):
        self.status = ExecutionStatus.RUNNING
        self.failure_reason = FailureReason.NONE
        if self.target_door not in DOOR_TARGETS:
            self._fail(state, FailureReason.REQUEST_INVALID, f"unknown door '{self.target_door}'")
            return
        self._bind_asset()
        self._resolve_handle_offset()
        robot = self.env.unwrapped.scene["robot"]
        self.home_q = robot.data.default_joint_pos[self.adapter.env_id, self.adapter._joint_ids].clone()
        # favourable redundant seed posture for the door pull (home + per-joint offsets)
        self.seed_q = self.home_q.clone()
        for ji, off in self.cfg.grasp_seed_offsets_deg.items():
            if 0 <= int(ji) < self.seed_q.shape[0]:
                self.seed_q[int(ji)] = self.seed_q[int(ji)] + math.radians(float(off))
        self.faced_q = self._faced_seed_q()   # home posture with joint 1 turned to face the door handle
        if self.cfg.use_turn_to_face:
            initial_state = "TURN_TO_FACE"
        elif self.cfg.use_grasp_seed:
            initial_state = "MOVE_TO_SEED"
        elif self.cfg.start_from_current:
            initial_state = "MOVE_TO_STANDOFF"
        else:
            initial_state = "MOVE_TO_HOME"
        self.runtime = _Runtime(
            state=initial_state,
            start_time=state.sim_time,
            state_start_time=state.sim_time,
            door_joint_name=self.door_joint_name,
            last_command_pose=state.robot.tcp_pose,
        )
        self.runtime.initial_angle = self._door_angle()
        self.runtime.current_angle = self.runtime.initial_angle
        if self.cfg.drive_door_joint:
            self._setup_door_drive()
            # with a powered hinge the door reaches the commanded angle, so require most of it as success
            if self.is_open:
                self.open_success_angle = math.radians(self.cfg.door_open_drive_deg) * 0.8
        else:
            self._setup_door_free()  # free-swinging hinge (low friction, NO drive torque)
            if self.is_open:
                # pull toward the full target; relaxation after the peak is tolerated via peak_angle in
                # _reached_goal, so VERIFY still passes once the door has reached the open angle.
                self.open_success_angle = math.radians(self.cfg.open_target_deg)
        self._record(state, "IDLE", initial_state)

    def _setup_door_free(self):
        """Pure-physical mode: free the hinge (stiffness 0) with a LOW damping so the gripper can swing
        it. This applies NO position/velocity target -> not a drive force, just low joint friction."""
        dev = self._asset.data.joint_pos.device
        n = self._asset.num_instances
        try:
            self._asset.write_joint_stiffness_to_sim(torch.zeros((n, 1), device=dev), joint_ids=[self._joint_id])
            self._asset.write_joint_damping_to_sim(
                torch.full((n, 1), float(self.cfg.free_damping), device=dev), joint_ids=[self._joint_id]
            )
        except Exception as exc:  # pragma: no cover - API guard
            print(f"[DoorIKSkill] WARN could not free door joint: {exc}", flush=True)

    def _setup_door_drive(self):
        """Give the door hinge a strong implicit PD actuator so it can be DRIVEN open by force."""
        dev = self._asset.data.joint_pos.device
        n = self._asset.num_instances
        stiff = torch.full((n, 1), float(self.cfg.door_drive_stiffness), device=dev)
        damp = torch.full((n, 1), float(self.cfg.door_drive_damping), device=dev)
        try:
            self._asset.write_joint_stiffness_to_sim(stiff, joint_ids=[self._joint_id])
            self._asset.write_joint_damping_to_sim(damp, joint_ids=[self._joint_id])
        except Exception as exc:  # pragma: no cover - API guard
            print(f"[DoorIKSkill] WARN could not set door drive gains: {exc}", flush=True)

    def _drive_door(self, state: SceneState):
        """Ramp the commanded door angle toward the open target and apply it (the PD actuator drives
        the hinge open with force). The ramp clock persists across phases so the door is not driven
        back closed when SWEEP -> VERIFY -> RELEASE -> RETREAT reset the per-state timer."""
        if self.runtime.drive_start_time < 0.0:
            self.runtime.drive_start_time = state.sim_time
        elapsed = state.sim_time - self.runtime.drive_start_time
        frac = min(1.0, elapsed / max(1e-3, self.cfg.door_drive_ramp_s))
        goal_ang = math.radians(self.cfg.door_open_drive_deg) if self.is_open else 0.0
        target_ang = self.runtime.initial_angle + (goal_ang - self.runtime.initial_angle) * frac
        dev = self._asset.data.joint_pos.device
        cmd = torch.full((self._asset.num_instances, 1), float(target_ang), device=dev)
        try:
            self._asset.set_joint_position_target(cmd, joint_ids=[self._joint_id])
        except Exception as exc:  # pragma: no cover
            print(f"[DoorIKSkill] WARN could not command door target: {exc}", flush=True)

    def step(self, state: SceneState, dt: float) -> SkillCommand:
        if self.status == ExecutionStatus.IDLE:
            self.start(state)
        if self.status in (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.STOPPED):
            return self._hold(state)

        self.runtime.current_angle = self._door_angle()
        if self.is_open:   # track the true PEAK (separate from max_angle, which plateau detection uses)
            self.runtime.peak_angle = max(self.runtime.peak_angle, self.runtime.current_angle)
        gripper = 1.0
        target = None

        # periodic end-effector trajectory log for offline failure analysis
        self._ee_log_i += 1
        if self.ee_log_every and self._ee_log_i % self.ee_log_every == 0:
            tcp = state.robot.tcp_pose
            ep = self.runtime.final_error_pos
            eo = self.runtime.final_error_ori
            # gripper tilt from level: TCP +X should be world vertical (the vertical bar axis). report
            # the angle between the ACTUAL TCP +X and world up, so a non-level wrist shows up directly.
            tilt = None
            try:
                xaxis = math_utils.quat_apply(tcp.quat_w.unsqueeze(0),
                                              torch.tensor([[1.0, 0.0, 0.0]], device=tcp.quat_w.device))[0]
                tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(abs(xaxis[2]))))))
            except Exception:
                pass
            # link6 vs link7 height (operator wants link6 ABOVE link7)
            wrist = ""
            try:
                robot = self.env.unwrapped.scene["robot"]
                bn = list(robot.data.body_names)
                i6 = next((i for i, n in enumerate(bn) if "link6" in n.lower()), None)
                i7 = next((i for i, n in enumerate(bn) if "link7" in n.lower()), None)
                if i6 is not None and i7 is not None:
                    z6 = float(robot.data.body_pos_w[self.adapter.env_id, i6, 2])
                    z7 = float(robot.data.body_pos_w[self.adapter.env_id, i7, 2])
                    wrist = f" link6z={z6:.3f} link7z={z7:.3f} {'6ABOVE7' if z6 > z7 else '6below7'}"
            except Exception:
                pass
            # arm joint angles + which joints are within 8deg of a limit (the reach bottleneck while pulling)
            joints = ""
            try:
                robot = self.env.unwrapped.scene["robot"]
                q = robot.data.joint_pos[self.adapter.env_id, self.adapter._joint_ids]
                lim = robot.data.soft_joint_pos_limits[self.adapter.env_id, self.adapter._joint_ids]
                margins = torch.minimum(q - lim[:, 0], lim[:, 1] - q)
                jdeg = [round(math.degrees(float(v))) for v in q]
                near = [f"j{i+1}={math.degrees(float(margins[i])):.0f}deg"
                        for i in range(len(q)) if float(margins[i]) < math.radians(8.0)]
                joints = f" q={jdeg}" + (f" NEARLIM[{','.join(near)}]" if near else "")
            except Exception:
                pass
            print(f"[DoorEE] t={state.sim_time:6.2f} {self.runtime.state:<14s} "
                  f"tcp={_round_vec(tcp.pos_w, 3)} door={math.degrees(self.runtime.current_angle):6.2f}deg "
                  f"grip={self.runtime.last_gripper:+.0f} gw={state.robot.gripper_width:.3f} "
                  f"perr={('%.3f' % ep) if ep is not None else 'NA'} "
                  f"oerr={('%.1f' % math.degrees(eo)) if eo is not None else 'NA'}deg "
                  f"tilt={('%.1f' % tilt) if tilt is not None else 'NA'}deg{wrist}{joints}", flush=True)

        reach_timeout = self.cfg.drive_reach_timeout if self.cfg.drive_door_joint else self.cfg.reach_timeout

        if self.runtime.state == "MOVE_TO_HOME":
            return self._home_step(state)
        if self.runtime.state == "TURN_TO_FACE":
            arm_q = state.robot.joint_pos[self.adapter._joint_ids]
            dq1 = abs(float(arm_q[0] - self.faced_q[0]))   # joint 1 = "facing" the door
            if dq1 <= self.cfg.face_joint_threshold:
                self.runtime.stable_count += 1
                if self.runtime.stable_count >= 3:
                    self._transition(state, "MOVE_TO_STANDOFF")
            else:
                self.runtime.stable_count = 0
            if self._state_elapsed(state) > self.cfg.face_timeout:
                self._transition(state, "MOVE_TO_STANDOFF")
            self.last_q = self.faced_q.clone()
            return SkillCommand(
                state.robot.tcp_pose, 1.0, self.status, control_mode="joint",
                joint_target=self.faced_q.clone(), drawer_joint_target=None,
            )
        if self.runtime.state == "MOVE_TO_SEED":
            # pre-position the arm to the favourable redundant posture so the grasp inherits a wrist
            # branch that pulls the door far without saturating/twisting the wrist (see sweep results).
            arm_q = state.robot.joint_pos[self.adapter._joint_ids]
            dist = float(torch.linalg.norm(arm_q - self.seed_q))
            if dist <= self.cfg.seed_joint_threshold:
                self.runtime.stable_count += 1
                if self.runtime.stable_count >= 3:
                    self._transition(state, "MOVE_TO_STANDOFF")
            else:
                self.runtime.stable_count = 0
            if self._state_elapsed(state) > self.cfg.seed_timeout:
                self._transition(state, "MOVE_TO_STANDOFF")
            self.last_q = self.seed_q.clone()
            return SkillCommand(
                state.robot.tcp_pose, 1.0, self.status, control_mode="joint",
                joint_target=self.seed_q.clone(), drawer_joint_target=None,
            )
        if self.runtime.state == "MOVE_TO_STANDOFF":
            # Reach a point ~standoff_distance IN FRONT of the handle (along the door normal) and arrive
            # roughly aligned with the grasp orientation BEFORE moving inward, so PRE_GRASP/APPROACH then
            # travel straight along the normal and the wrist/forearm never clips the door on the way in.
            # This is an ALIGNMENT WAYPOINT, not a precise grasp: loose thresholds + soft timeout-advance
            # (the retracted+aligned pose is awkward for IK to nail exactly; good-enough then go in).
            target = self._grasp_pose(self.cfg.standoff_distance)
            gripper = 1.0
            err = pose_error(state.robot.tcp_pose, target)
            if err.position <= 0.06 and err.orientation <= self.cfg.standoff_ori_threshold:
                self.runtime.stable_count += 1
                if self.runtime.stable_count >= 3:
                    self._transition(state, "MOVE_TO_PRE_GRASP")
            else:
                self.runtime.stable_count = 0
            if self._state_elapsed(state) > reach_timeout:
                print(f"[DoorIKSkill] standoff soft-advance (pos_err={err.position:.3f} "
                      f"ori_err={math.degrees(err.orientation):.1f}deg)", flush=True)
                self._transition(state, "MOVE_TO_PRE_GRASP")
        elif self.runtime.state == "MOVE_TO_PRE_GRASP":
            target = self._grasp_pose(self.cfg.pre_grasp_clearance)
            gripper = 1.0
            self._advance_when_reached(state, target, "APPROACH", reach_timeout)
        elif self.runtime.state == "APPROACH":
            target = self._grasp_pose(-self.cfg.grasp_depth)   # aim PAST the handle so the bar seats deep
            gripper = 1.0
            self._advance_when_reached(state, target, "CLOSE_GRIPPER", reach_timeout)
        elif self.runtime.state == "CLOSE_GRIPPER":
            target = self._grasp_pose(-self.cfg.grasp_depth)
            gripper = -1.0
            if self._state_elapsed(state) >= self.cfg.close_duration:
                self._capture_grip(state)
                self._transition(state, "SWEEP")
        elif self.runtime.state == "SWEEP":
            target = self._pull_target(self.cfg.arc_lead_deg)
            gripper = -1.0  # KEEP GRIPPING the handle through the whole pull
            if self.cfg.drive_door_joint:
                self._drive_door(state)
            # track the best opening so far -> detect when pulling can't open it any further
            if self.is_open:
                if self.runtime.current_angle > self.runtime.max_angle + math.radians(self.cfg.pull_plateau_deg):
                    self.runtime.max_angle = self.runtime.current_angle
                    self.runtime.plateau_count = 0
                else:
                    self.runtime.plateau_count += 1
            cur_deg = math.degrees(self.runtime.current_angle)
            if (self.is_open and self.cfg.push_after_pull
                    and cur_deg >= self.cfg.push_switch_angle_deg):
                self._transition(state, "RELEASE_COAST")
            elif self._reached_goal():
                self._transition(state, "VERIFY")
            elif (self.is_open and self.cfg.pull_to_max and cur_deg >= self.cfg.pull_min_open_deg
                    and self.runtime.plateau_count >= self.cfg.pull_plateau_steps):
                # The gripper pulled the door to its physical max (angle plateaued WHILE still gripping).
                # Accept this as the open result and release here -- no relying on inertia/coast.
                if cur_deg < self.cfg.regrip_below_deg and self.runtime.regrip_count < self.cfg.max_regrips:
                    # The pull plateaued because the WRIST saturated (j6 at its limit), not because the
                    # door is fully open. Release and RE-APPROACH the handle at its new angle for a fresh
                    # wrist config, then keep pulling -- staged opening past the single-grasp wrist limit.
                    self.runtime.regrip_count += 1
                    self.runtime.plateau_count = 0
                    self.runtime.max_angle = self.runtime.current_angle
                    self._grip_tcp_pos = None
                    print(f"[DoorIKSkill] pull plateaued ~{cur_deg:.1f}deg (wrist limit) -> RE-GRIP "
                          f"#{self.runtime.regrip_count}/{self.cfg.max_regrips}", flush=True)
                    self._transition(state, "MOVE_TO_PRE_GRASP")
                else:
                    self.runtime.verify_passed = True
                    # freeze the hold pose at the MAX-pulled angle so SETTLE/RELEASE keep the gripper THERE
                    # (resisting any spring-back) instead of following the door if it starts closing.
                    self._settle_pose = self._pull_target(0.0)
                    print(f"[DoorIKSkill] pulled door to max ~{cur_deg:.1f}deg (plateau) while gripping -> settle",
                          flush=True)
                    self._transition(state, "SETTLE")
            elif self._state_elapsed(state) > self.cfg.sweep_timeout:
                reason = FailureReason.DOOR_OPEN_TIMEOUT if self.is_open else FailureReason.DOOR_CLOSE_TIMEOUT
                self._fail(
                    state, reason,
                    f"door sweep timeout {self.target_door}: angle={self.runtime.current_angle:.4f}",
                )
        elif self.runtime.state == "RELEASE_COAST":
            # Released at the switch angle: open the gripper and back well clear of the door, letting the
            # (driven) door keep swinging open, so the gripper never scrapes the door frame on its way
            # from the outside handle to the inside face.
            target = self._clear_pose()
            gripper = 1.0
            if self.cfg.drive_door_joint:
                self._drive_door(state)
            if self._state_elapsed(state) >= self.cfg.coast_duration:
                self._transition(state, "PUSH")
        elif self.runtime.state == "PUSH":
            # Open gripper, reposition behind the free edge on the inner face, push along the opening
            # tangent. The hinge drive (if on) does the heavy lifting; the gripper stays in a reachable
            # pose instead of chasing the receding handle.
            target = self._push_pose()
            gripper = -1.0  # closed fist pushes the inner face more solidly than open fingers
            if self.cfg.drive_door_joint:
                self._drive_door(state)
            if self._reached_goal():
                self._transition(state, "VERIFY")
            elif self._state_elapsed(state) > self.cfg.push_timeout:
                if self.cfg.drive_door_joint:
                    self._transition(state, "VERIFY")  # drive keeps opening; pushing is best-effort
                else:
                    self._fail(state, FailureReason.DOOR_OPEN_TIMEOUT,
                               f"door push timeout {self.target_door}: angle={self.runtime.current_angle:.4f}")
        elif self.runtime.state == "VERIFY":
            # Keep the handle held at the rigid pull pose and confirm the door joint stays past the
            # success threshold for a few consecutive steps (rejects a transient contact spike).
            target = self._pull_target(0.0)
            gripper = -1.0
            if self.cfg.drive_door_joint:
                self._drive_door(state)  # hold the door at the open angle while verifying
            if self._reached_goal():
                self.runtime.stable_count += 1
                if self.runtime.stable_count >= self.cfg.verify_stable_cycles:
                    self.runtime.verify_passed = True
                    self._transition(state, "RELEASE")
            else:
                self.runtime.stable_count = 0
            if self._state_elapsed(state) > self.cfg.verify_timeout:
                # Confirmation budget spent. If the joint is still at goal, accept; else fail honestly.
                if self._reached_goal():
                    self.runtime.verify_passed = True
                    self._transition(state, "RELEASE")
                else:
                    reason = FailureReason.DOOR_OPEN_TIMEOUT if self.is_open else FailureReason.DOOR_CLOSE_TIMEOUT
                    self._fail(state, reason,
                               f"verify timeout {self.target_door}: angle={self.runtime.current_angle:.4f}")
        elif self.runtime.state == "SETTLE":
            # Hold the door at its MAX-pulled angle (frozen pose) with the gripper STILL CLOSED so its
            # swing inertia damps out -- and so the gripper resists any spring-back -- before we let go.
            target = self._settle_pose if self._settle_pose is not None else self._pull_target(0.0)
            gripper = -1.0
            if self.cfg.drive_door_joint:
                self._drive_door(state)
            if self._state_elapsed(state) >= self.cfg.settle_before_release:
                self._transition(state, "RELEASE")
        elif self.runtime.state == "RELEASE":
            # Open the gripper while easing the hand a few cm toward the base so it lifts OFF the bar
            # instead of pressing into the door as the fingers open (removes the release-instant contact).
            if self._retreat_clear_target is None:
                self._retreat_clear_target = self._toward_base_pose(state, self.cfg.retreat_clear_distance)
            target = self._toward_base_pose(state, 0.03)
            gripper = 1.0
            if self.cfg.drive_door_joint:
                self._drive_door(state)  # keep the door held open as the gripper lets go
            if self._state_elapsed(state) >= self.cfg.release_duration:
                if self.cfg.skip_retreat:
                    # Leave the arm AT the open handle (door at its pulled-open angle) and SUCCEED; an
                    # external collision-free planner (cuRobo) does the retreat so no link clips the door.
                    self._succeed(state)
                else:
                    self._transition(state, "RETREAT_CLEAR")
        elif self.runtime.state == "RETREAT_CLEAR":
            # Egress step 1: back the OPEN gripper horizontally TOWARD THE ROBOT BASE, out of the door's
            # swing region, so the wrist/hand does not sweep back through the open door (which knocked it
            # shut). Target captured once at entry so it does not chase a drifting handle.
            if self._retreat_clear_target is None:
                self._retreat_clear_target = self._toward_base_pose(state, self.cfg.retreat_clear_distance)
            target = self._retreat_clear_target
            gripper = 1.0
            err = pose_error(state.robot.tcp_pose, target)
            if err.position <= self.cfg.reach_pos_threshold:
                self.runtime.stable_count += 1
                if self.runtime.stable_count >= self.cfg.reach_stable_cycles:
                    self._transition(state, "RETREAT_HOME" if self.cfg.retreat_to_home else "RELEASE_DONE")
            else:
                self.runtime.stable_count = 0
            if self._state_elapsed(state) > self.cfg.retreat_timeout:
                self._transition(state, "RETREAT_HOME" if self.cfg.retreat_to_home else "RELEASE_DONE")
        elif self.runtime.state == "RELEASE_DONE":
            # clear of the door, not homing -> succeed where we are
            self._succeed(state)
        elif self.runtime.state == "RETREAT_HOME":
            # Egress step 2: now clear of the door, drive the arm back to its default home joints.
            gripper = 1.0
            arm_q = state.robot.joint_pos[self.adapter._joint_ids]
            dist = float(torch.linalg.norm(arm_q - self.home_q))
            self.last_q = self.home_q.clone()
            if dist <= self.cfg.home_joint_threshold or self._state_elapsed(state) > self.cfg.retreat_timeout:
                self._succeed(state)
            self.runtime.last_gripper = gripper
            self._last_phase_target = None
            return SkillCommand(
                state.robot.tcp_pose, gripper, self.status, control_mode="joint",
                joint_target=self.home_q.clone(), drawer_joint_target=None,
            )

        self.runtime.last_gripper = gripper
        self._last_phase_target = target
        return self._command(state, target, gripper)

    def viz_poses(self) -> list:
        """(name, PoseState) pairs for live visualization: current phase TCP target + door handle.

        Stable names so the markers reuse the same prims (no per-frame churn)."""
        out = []
        tgt = getattr(self, "_last_phase_target", None)
        if tgt is not None:
            out.append(("phase_target", tgt))  # stable name -> one reused marker that follows the phase
        try:
            handle = self._handle_pos()
            quat = self._grasp_quat(self._approach_dir(), handle.device)
            out.append(("door_handle", PoseState(handle, quat)))
        except Exception:
            pass
        return out

    def cancel(self, state: SceneState) -> SkillCommand:
        self.status = ExecutionStatus.STOPPED
        self.failure_reason = FailureReason.CANCELLED_BY_USER
        self._transition(state, "CANCELLED")
        return self._hold(state)

    def result(self, state: SceneState) -> SkillResult:
        return SkillResult(
            request_id=self.request.request_id,
            skill_type=self.request.skill_type,
            target_name=self.target_door,
            success=self.status == ExecutionStatus.SUCCEEDED,
            final_status=self.status,
            failure_reason=self.failure_reason.value or None,
            elapsed_time=max(0.0, state.sim_time - self.runtime.start_time),
            final_tcp_pose=pose_tensor(state.robot.tcp_pose),
            position_error=self.runtime.final_error_pos,
            orientation_error=self.runtime.final_error_ori,
            gripper_width=state.robot.gripper_width,
            state_history=self.runtime.history,
        )

    # ---- helpers ----------------------------------------------------------
    def _home_step(self, state: SceneState) -> SkillCommand:
        arm_q = state.robot.joint_pos[self.adapter._joint_ids]
        dist = float(torch.linalg.norm(arm_q - self.home_q))
        if dist <= self.cfg.home_joint_threshold:
            self.runtime.stable_count += 1
            if self.runtime.stable_count >= 3:
                self._transition(state, "MOVE_TO_STANDOFF")
        else:
            self.runtime.stable_count = 0
        if self._state_elapsed(state) > self.cfg.home_timeout:
            self._transition(state, "MOVE_TO_STANDOFF")
        self.last_q = self.home_q.clone()
        return SkillCommand(
            state.robot.tcp_pose, 1.0, self.status, control_mode="joint",
            joint_target=self.home_q.clone(), drawer_joint_target=None,
        )

    def _command(self, state: SceneState, target: PoseState | None, gripper: float) -> SkillCommand:
        if target is None:
            return self._hold(state, gripper)
        # Decouple the door motion from the GLOBAL speed scale (UI default 5x): pre-divide the per-step
        # limits by the scale so step_pose's internal *scale cancels and the door always advances at its
        # own small, ACCURATE step. A big per-step jump (e.g. 30deg at 5x) exceeds what one IK/DLS step
        # can track -> the wrist lags the command and the gripper visibly twists/slips off the bar and
        # the door barely opens. The door is precision-critical, so it runs slow regardless of the UI speed.
        scale = max(1.0, float(get_speed_scale()))
        cmd = step_pose(state.robot.tcp_pose, target,
                        self.cfg.max_pos_step / scale, self.cfg.max_ori_step / scale)
        ik = self.adapter.solve(cmd, null_gain=self.cfg.null_space_gain)
        err = pose_error(state.robot.tcp_pose, target)
        self.runtime.final_error_pos = err.position
        self.runtime.final_error_ori = err.orientation
        self.runtime.last_command_pose = cmd
        if not ik.success:
            q = self.last_q if self.last_q is not None else state.robot.joint_pos[self.adapter._joint_ids].clone()
            return SkillCommand(state.robot.tcp_pose, gripper, self.status, control_mode="joint",
                                joint_target=q, drawer_joint_target=None)
        self.last_q = ik.q_des
        return SkillCommand(cmd, gripper, self.status, control_mode="joint", joint_target=ik.q_des,
                            drawer_joint_target=None)

    def _hold(self, state: SceneState, gripper: float = -1.0) -> SkillCommand:
        q = self.last_q if self.last_q is not None else state.robot.joint_pos[self.adapter._joint_ids].clone()
        return SkillCommand(state.robot.tcp_pose, gripper, self.status, control_mode="joint",
                            joint_target=q, drawer_joint_target=None)

    def _advance_when_reached(self, state: SceneState, target: PoseState, next_state: str, timeout: float,
                              ori_threshold: float | None = None) -> bool:
        err = pose_error(state.robot.tcp_pose, target)
        ori_ok = (ori_threshold is None) or (err.orientation <= ori_threshold)
        if err.position <= self.cfg.reach_pos_threshold and ori_ok:
            self.runtime.stable_count += 1
            if self.runtime.stable_count >= self.cfg.reach_stable_cycles:
                self._transition(state, next_state)
                return True
        else:
            self.runtime.stable_count = 0
        if self._state_elapsed(state) > timeout:
            if self.cfg.drive_door_joint and self.cfg.soft_reach_advance:
                # Best-effort grasp: the hinge is driven open by force, so a slow/short reach is not
                # fatal -- advance and let the door drive open while the gripper holds its closest pose.
                print(f"[DoorIKSkill] soft-advance {self.runtime.state}->{next_state} "
                      f"(reach err={err.position:.3f}, drive-assist on)", flush=True)
                self._transition(state, next_state)
                return True
            # Pure-physical mode: if the gripper is within a grip's reach of the handle bar, advance and
            # let CLOSE_GRIPPER + SWEEP engage the bar (the door-angle plateau/verify is the real test).
            if err.position <= self.cfg.pure_soft_reach_advance:
                print(f"[DoorIKSkill] soft-advance {self.runtime.state}->{next_state} "
                      f"(reach err={err.position:.3f}, within {self.cfg.pure_soft_reach_advance:.3f})", flush=True)
                self._transition(state, next_state)
                return True
            self._fail(state, FailureReason.POSITION_TIMEOUT, f"{self.runtime.state} reach timeout (err={err.position:.3f})")
        return False

    def _state_elapsed(self, state: SceneState) -> float:
        return max(0.0, state.sim_time - self.runtime.state_start_time)

    def _transition(self, state: SceneState, new_state: str):
        old = self.runtime.state
        if old == new_state:
            return
        elapsed = self._state_elapsed(state)  # time spent in the state we are leaving
        self.runtime.state = new_state
        self.runtime.state_start_time = state.sim_time
        self.runtime.stable_count = 0
        self._record(state, old, new_state, elapsed)

    def _record(self, state: SceneState, old: str, new: str, elapsed: float = 0.0):
        tgt = self.runtime.last_command_pose
        target_pos = _round_vec(tgt.pos_w) if tgt is not None else None
        rec = {
            "time": round(state.sim_time, 4),
            "skill": self.request.skill_type.value,
            "backend": self.backend,
            "target_door": self.target_door,
            "door_joint_name": self.runtime.door_joint_name,
            "from": old,
            "to": new,
            "elapsed_in_prev": round(elapsed, 3),
            "door_angle": round(self.runtime.current_angle, 5),
            "open_thresh": round(self.open_success_angle, 4),
            "target_pos": target_pos,
            "gripper": round(self.runtime.last_gripper, 2),
            "verify_passed": self.runtime.verify_passed,
            "failure_reason": self.failure_reason.value or None,
            "failure_message": self.runtime.last_failure_message,
        }
        self.runtime.history.append(rec)
        print(f"[DoorIKSkill] {rec}", flush=True)

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
        print(f"[DoorIKSkill] success {self.request.skill_type.value} target={self.target_door} "
              f"angle={self.runtime.current_angle:.4f}", flush=True)


class OpenDoorIKSkill(DoorIKSkill):
    backend = "ik_arc"
    is_open = True


class CloseDoorIKSkill(DoorIKSkill):
    backend = "ik_arc"
    is_open = False
