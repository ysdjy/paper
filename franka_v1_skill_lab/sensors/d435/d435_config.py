# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Single source of truth for the TWO wrist cameras in the shared V1 scene.

PURE PYTHON (no Isaac / torch / numpy at import time) so it imports from any
venv and from the offline tests. The Isaac-side spawn lives in
:mod:`d435_scene_cfg`.

The V1 scene fixes **two distinct logical cameras** on the Franka wrist — they
are NOT one camera reused:

1. ``foundationpose_d435_rgbd`` — RGB **+ depth**, 640x480, simulates a real
   Intel RealSense D435. Consumer: FoundationPose 6-DoF pose estimation.
2. ``vla_libero_eye_in_hand`` — RGB (optional depth), 128x128 default (also
   224x224), pose aligned to robosuite/LIBERO ``robot0_eye_in_hand``. Consumer:
   pi0.5 / VLA / LIBERO-style imitation learning.

Both mount **directly** on ``panda_hand`` (the spawner requires an existing
parent prim, and an extra ``V1_Cameras`` Xform would have to be created first —
so we flatten and keep clear names instead):

    {ENV_REGEX_NS}/Robot/panda_hand/
        D435_Body                 (visible body — HIDDEN by default)
        foundationpose_d435_rgbd
        vla_libero_eye_in_hand

The D435 housing mesh is **invisible by default** (show it with
``--show_camera_body``); it never has a collider or rigid body, so Franka
dynamics are unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# prim paths
# ---------------------------------------------------------------------------
WRIST_PARENT_LINK = "panda_hand"
WRIST_MOUNT_PARENT = "{ENV_REGEX_NS}/Robot/panda_hand"
# Cameras mount directly on panda_hand (an extra Xform parent would need to be
# created before the spawner runs; flattening keeps the same semantics).
V1_CAMERA_GROUP = WRIST_MOUNT_PARENT

FP_CAM_NAME = "foundationpose_d435_rgbd"
VLA_CAM_NAME = "vla_libero_eye_in_hand"
FRONT_CAM_NAME = "vla_front_static"
LEFT_CAM_NAME = "vla_left_static"
RIGHT_CAM_NAME = "vla_right_static"
TOP_CAM_NAME = "vla_top_static"

FP_CAM_PRIM = WRIST_MOUNT_PARENT + "/" + FP_CAM_NAME
VLA_CAM_PRIM = WRIST_MOUNT_PARENT + "/" + VLA_CAM_NAME
D435_BODY_PRIM = WRIST_MOUNT_PARENT + "/D435_Body"

# Fixed third-person front camera. Mounted DIRECTLY under the env root (which
# exists — an extra "Sensors" Xform would need creating first, like V1_Cameras
# did), so it is static and does NOT move with panda_hand.
ENV_ROOT = "{ENV_REGEX_NS}"
FRONT_CAM_PRIM = ENV_ROOT + "/" + FRONT_CAM_NAME
# Fixed third-person LEFT camera (faces the robot from its left side). Like the
# front cam it is mounted directly under the env root, so it is static.
LEFT_CAM_PRIM = ENV_ROOT + "/" + LEFT_CAM_NAME
# Fixed third-person RIGHT camera (faces the robot from its right side). Static.
RIGHT_CAM_PRIM = ENV_ROOT + "/" + RIGHT_CAM_NAME
# Fixed TOP-DOWN camera (directly above the robot, looks straight down). Static.
TOP_CAM_PRIM = ENV_ROOT + "/" + TOP_CAM_NAME

# Visible D435 housing cuboid (HIDDEN by default). Visual only.
D435_BODY_LOCAL_POS = (0.06, 0.0, 0.025)
D435_BODY_LOCAL_ROT = (1.0, 0.0, 0.0, 0.0)  # (w, x, y, z)
D435_BODY_SIZE = (0.028, 0.090, 0.025)      # (depth, width, height) meters
D435_BODY_COLOR = (0.08, 0.08, 0.10)

# Data-type strings the renderer understands.
RGB_DATA_TYPE = "rgb"
DEPTH_DATA_TYPE = "distance_to_image_plane"
MASK_DATA_TYPE = "semantic_segmentation"


# ---------------------------------------------------------------------------
# tiny quaternion helpers (w, x, y, z) — pure python, no numpy
# ---------------------------------------------------------------------------
def quat_mul(q1, q2):
    """Hamilton product q1 ⊗ q2, both (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def quat_rotate(q, v):
    """Rotate 3-vector v by quaternion q (w, x, y, z)."""
    w, x, y, z = q
    vx, vy, vz = v
    # t = 2 * cross(q.xyz, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    # v' = v + w*t + cross(q.xyz, t)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def quat_normalize(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    return (w / n, x / n, y / n, z / n)


def quat_to_matrix(q):
    """Return the 3x3 rotation matrix (list of rows) for q (w, x, y, z)."""
    w, x, y, z = quat_normalize(q)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


# ---------------------------------------------------------------------------
# LIBERO / robosuite Panda eye-in-hand reference
# ---------------------------------------------------------------------------
# These are the exact robosuite (MuJoCo) parameters for the Panda
# ``robot0_eye_in_hand`` camera that LIBERO uses. Source: robosuite Panda robot
# model (gripper mount `right_hand` body + eye_in_hand camera). MuJoCo camera
# convention == OpenGL (looks down -Z, +Y up), which we map to Isaac via
# CameraCfg.OffsetCfg(convention="opengl").
LIBERO_REFERENCE = {
    "source": "robosuite Panda robot0_eye_in_hand (MuJoCo/OpenGL camera convention)",
    "camera_name": "robot0_eye_in_hand",
    "parent_body": "right_hand",
    "parent_local_pos": (0.0, 0.0, 0.1065),
    "parent_local_quat_wxyz": (0.924, 0.0, 0.0, -0.383),  # Rz(-45°)
    "camera_local_pos": (0.05, 0.0, 0.0),
    "camera_local_quat_wxyz": (0.0, 0.707108, 0.707108, 0.0),
    "fovy_deg": 75.0,
    "default_image_size": (128, 128),
    "mujoco_camera_convention": "opengl",
}

# Alignment from robosuite `right_hand` frame to Isaac `panda_hand` frame.
# APPROXIMATE: defaulted to identity. robosuite's right_hand z-axis points along
# the gripper approach direction, as does Isaac's panda_hand z-axis, so the
# eye-in-hand view direction (along +approach) is preserved under identity. If a
# precise hand-frame offset is calibrated later, set this quaternion and the VLA
# camera pose updates everywhere. See sensors/docs/d435_foundationpose_contract.md.
RIGHTHAND_TO_PANDAHAND_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)
LIBERO_POSE_IS_APPROXIMATE = True

# ---------------------------------------------------------------------------
# ACTIVE wrist-camera mount (GUI-tuned by the user, 2026-06-12)
# ---------------------------------------------------------------------------
# The user hand-tuned the VLA camera in the layout editor and read back the
# camera prim's USD local transform (relative to panda_hand). BOTH wrist cameras
# now share this exact pose, so the FoundationPose depth image is captured from
# the SAME viewpoint as the VLA RGB. Because a USD camera is OpenGL-convention,
# CameraCfg.OffsetCfg(convention="opengl") writes this rotation onto the prim
# verbatim (camera.py converts origin->opengl, an identity when origin=opengl),
# so the spawned prim matches the GUI values 1:1.
#
# quat order is wxyz (USD / IsaacLab native, what xformOp:orient stores).
WRIST_CAMERA_LOCAL_POS = (0.04414, 0.00482, 0.00429)
_RAW_WRIST_ROT = (0.06663, 0.70859, 0.7012, 0.04215)  # (w, x, y, z) from GUI
WRIST_CAMERA_LOCAL_ROT_WXYZ = quat_normalize(_RAW_WRIST_ROT)
WRIST_CAMERA_CONVENTION = "opengl"


def robosuite_eye_in_hand_to_isaac_offset(align_quat=RIGHTHAND_TO_PANDAHAND_QUAT_WXYZ):
    """Convert the robosuite eye-in-hand local pose to an Isaac camera offset.

    Composes ``T(parent) ∘ T(camera_local)`` (both relative to ``right_hand``),
    then applies the ``right_hand → panda_hand`` alignment. The returned rotation
    is in the **OpenGL/MuJoCo** camera convention, so pass it straight to
    ``CameraCfg.OffsetCfg(pos=..., rot=..., convention="opengl")``.

    Returns:
        (pos: tuple[3], quat_wxyz: tuple[4], convention: str)
    """
    p1 = LIBERO_REFERENCE["parent_local_pos"]
    q1 = LIBERO_REFERENCE["parent_local_quat_wxyz"]
    p2 = LIBERO_REFERENCE["camera_local_pos"]
    q2 = LIBERO_REFERENCE["camera_local_quat_wxyz"]

    # camera pose in right_hand frame
    pos_rh = tuple(a + b for a, b in zip(p1, quat_rotate(q1, p2)))
    quat_rh = quat_normalize(quat_mul(q1, q2))

    # right_hand -> panda_hand alignment
    pos_ph = quat_rotate(align_quat, pos_rh)
    quat_ph = quat_normalize(quat_mul(align_quat, quat_rh))
    return pos_ph, quat_ph, LIBERO_REFERENCE["mujoco_camera_convention"]


# ---------------------------------------------------------------------------
# camera profile
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CameraProfile:
    """Everything one V1 wrist camera needs — pure data."""

    name: str
    prim_path: str
    consumer: str               # "foundationpose" | "vla_pi05"
    kind: str                   # "rgbd" | "rgb"
    width: int
    height: int
    focal_length: float
    horizontal_aperture: float
    clipping_range: tuple
    offset_pos: tuple
    offset_rot: tuple           # (w, x, y, z)
    convention: str             # "ros" | "opengl" | "world"
    depth: bool
    image_keys: tuple
    frame_id: str
    parent: str = WRIST_PARENT_LINK   # "panda_hand" (wrist cams) | "env_root" (static front cam)
    static: bool = False              # True = fixed in the scene, does NOT move with the arm
    role: str = "wrist_image"         # "image" | "wrist_image" | "depth" (LeRobot/pi0.5 role)
    libero_compatible: bool = False
    pose_is_approximate: bool = False
    notes: str = ""

    # --- derived intrinsics -------------------------------------------------
    def intrinsics(self, width: int | None = None, height: int | None = None) -> dict:
        """Nominal pinhole intrinsics for a (possibly overridden) resolution."""
        w = int(width or self.width)
        h = int(height or self.height)
        fx = self.focal_length * w / self.horizontal_aperture
        fy = fx  # square pixels (vertical aperture scaled to keep pixel aspect 1)
        return {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(w / 2.0),
            "cy": float(h / 2.0),
            "width": w,
            "height": h,
        }

    def intrinsics_matrix(self, width: int | None = None, height: int | None = None):
        k = self.intrinsics(width, height)
        return [[k["fx"], 0.0, k["cx"]], [0.0, k["fy"], k["cy"]], [0.0, 0.0, 1.0]]

    def data_types(self, want_depth: bool | None = None, want_mask: bool = False) -> list:
        depth = self.depth if want_depth is None else (want_depth and self.depth or want_depth)
        dt = [RGB_DATA_TYPE]
        if depth:
            dt.append(DEPTH_DATA_TYPE)
        if want_mask:
            dt.append(MASK_DATA_TYPE)
        return dt

    def descriptor(self, enabled: bool = True, visible_body: bool = False,
                   width: int | None = None, height: int | None = None) -> dict:
        """The registry ``sensors.<name>`` block."""
        intr = self.intrinsics(width, height)
        return {
            "enabled": bool(enabled),
            "type": "rgbd" if self.kind == "rgbd" else "rgb",
            "role": self.role,
            "consumer": self.consumer,
            "parent": self.parent,
            "static": bool(self.static),
            "rgb": True,
            "depth": bool(self.depth),
            "resolution": [intr["width"], intr["height"]],
            "visible_body": bool(visible_body),
            "frame_id": self.frame_id,
            "prim_path": self.prim_path,
            "intrinsics": intr,
            "offset": {"pos": list(self.offset_pos), "rot_wxyz": list(self.offset_rot), "convention": self.convention},
            "image_keys": list(self.image_keys),
            "libero_compatible": bool(self.libero_compatible),
            "pose_is_approximate": bool(self.pose_is_approximate),
            "consumer_modules": ["skill_runtime", "teleop_collection", "pi05_training", "perception_foundationpose"],
        }


# ---------------------------------------------------------------------------
# the two concrete profiles
# ---------------------------------------------------------------------------
# FoundationPose D435 RGB-D — the proven wrist-camera offset (ROS convention)
# that looks forward + slightly down onto the grasp workspace.
FP_PROFILE = CameraProfile(
    name=FP_CAM_NAME,
    prim_path=FP_CAM_PRIM,
    consumer="foundationpose",
    kind="rgbd",
    width=640,
    height=480,
    focal_length=20.0,            # fx = 20*640/20.955 ≈ 611 (≈ real D435)
    horizontal_aperture=20.955,
    clipping_range=(0.05, 3.0),
    # SAME pose as the VLA camera (GUI-tuned) so the depth view matches the RGB view.
    offset_pos=WRIST_CAMERA_LOCAL_POS,
    offset_rot=WRIST_CAMERA_LOCAL_ROT_WXYZ,
    convention=WRIST_CAMERA_CONVENTION,
    depth=True,
    image_keys=("images.wrist_depth",),  # FP camera owns the depth key for pi0.5
    frame_id=FP_CAM_NAME,
    parent=WRIST_PARENT_LINK,
    static=False,
    role="depth",
    notes="Simulated Intel RealSense D435 RGB-D for FoundationPose. Shares the GUI-tuned wrist pose with the VLA camera.",
)

# VLA / LIBERO eye-in-hand. Pose is the user's GUI-tuned wrist mount (read from
# the layout editor). The LIBERO/robosuite reference + conversion stay available
# for documentation (robosuite_eye_in_hand_to_isaac_offset()), but the ACTIVE
# offset is the measured one.
VLA_PROFILE = CameraProfile(
    name=VLA_CAM_NAME,
    prim_path=VLA_CAM_PRIM,
    consumer="vla_pi05",
    kind="rgb",
    width=128,
    height=128,
    focal_length=13.654,          # fovy ≈ 75° at aperture 20.955 (LIBERO eye_in_hand)
    horizontal_aperture=20.955,
    clipping_range=(0.02, 3.0),
    offset_pos=WRIST_CAMERA_LOCAL_POS,
    offset_rot=WRIST_CAMERA_LOCAL_ROT_WXYZ,
    convention=WRIST_CAMERA_CONVENTION,   # "opengl"
    depth=False,
    image_keys=("images.wrist_rgb", "images.eye_in_hand_rgb", "images.robot0_eye_in_hand_rgb"),
    frame_id=VLA_CAM_NAME,
    parent=WRIST_PARENT_LINK,
    static=False,
    role="wrist_image",           # LeRobot/pi0.5 `wrist_image`
    libero_compatible=True,
    pose_is_approximate=False,    # measured in the GUI, not the approximate LIBERO conversion
    notes="VLA eye-in-hand. Pose GUI-tuned by the user; FOV from LIBERO robot0_eye_in_hand (fovy 75deg).",
)

# Fixed third-person FRONT camera -> LeRobot/pi0.5 `image` (main view).
# Pose GUI-tuned by the user (read back from the layout editor as the camera
# prim's USD local transform). convention="opengl" writes it onto the prim
# verbatim (USD cameras are OpenGL-convention), so the spawned camera matches the
# GUI values 1:1. Resolution 256x256 matches the converter's image_resize
# (dataset_mapping_isaaclab_franka.yaml). Static (under the env root), so it does
# NOT move with the arm.
FRONT_RGB_KEY = "images.front_rgb"     # V1 logical key
LEROBOT_IMAGE_KEY = "images.image"     # LeRobot field alias
FRONT_CAM_LOCAL_POS = (2.4, 0.0, 0.87)
FRONT_CAM_LOCAL_ROT_WXYZ = quat_normalize((0.56666, 0.42297, 0.42297, 0.56666))  # (w,x,y,z) from GUI
FRONT_CAM_CONVENTION = "opengl"
FRONT_PROFILE = CameraProfile(
    name=FRONT_CAM_NAME,
    prim_path=FRONT_CAM_PRIM,
    consumer="vla_pi05",
    kind="rgb",
    width=256,
    height=256,
    focal_length=24.0,            # fovx ≈ 47°
    horizontal_aperture=20.955,
    clipping_range=(0.05, 8.0),
    offset_pos=FRONT_CAM_LOCAL_POS,
    offset_rot=FRONT_CAM_LOCAL_ROT_WXYZ,
    convention=FRONT_CAM_CONVENTION,
    depth=False,
    image_keys=(FRONT_RGB_KEY, LEROBOT_IMAGE_KEY),
    frame_id=FRONT_CAM_NAME,
    parent="env_root",
    static=True,
    role="image",                 # LeRobot/pi0.5 `image`
    notes="Fixed third-person front camera (pose GUI-tuned by the user). Maps to LeRobot `image`.",
)

# Fixed third-person LEFT camera -> RGB + depth, faces the robot from its left
# (+Y) side. Same intrinsics/resolution as the front camera ("配置和前视相机一模一样"),
# but with depth ENABLED (the user wants this view to output an RGB image AND a
# depth map). Default pose: the front camera's pose rotated 90deg about world-Z
# around the robot, so it looks straight at the robot from the left at the same
# distance/elevation as the front cam. Static (under env root), GUI-tunable +
# saved like the front cam. The user can fine-tune the pose in the camera panel.
LEFT_RGB_KEY = "images.left_rgb"
LEFT_CAM_LOCAL_POS = (2.18, 4.04, 2.33)
LEFT_CAM_LOCAL_ROT_WXYZ = quat_normalize((-0.001788, 0.003727, 0.501148, 0.865352))  # (w,x,y,z)
LEFT_CAM_CONVENTION = "opengl"
LEFT_PROFILE = CameraProfile(
    name=LEFT_CAM_NAME,
    prim_path=LEFT_CAM_PRIM,
    consumer="vla_pi05",
    kind="rgbd",                  # RGB + depth
    width=256,
    height=256,
    focal_length=24.0,            # same lens as the front camera
    horizontal_aperture=20.955,
    clipping_range=(0.05, 8.0),
    offset_pos=LEFT_CAM_LOCAL_POS,
    offset_rot=LEFT_CAM_LOCAL_ROT_WXYZ,
    convention=LEFT_CAM_CONVENTION,
    depth=True,
    image_keys=(LEFT_RGB_KEY, "images.left_depth"),
    frame_id=LEFT_CAM_NAME,
    parent="env_root",
    static=True,
    role="left_image",
    notes="Fixed third-person LEFT camera (RGB+depth), faces the robot from its left side. Same config as the front cam, depth enabled.",
)

# Fixed third-person RIGHT camera -> RGB + depth, faces the robot from its right
# (-Y) side. Same config as front/left; pose = front cam rotated -90deg about
# world-Z around the robot (mirror of the left cam), so it looks straight at the
# robot from the right at the same distance/elevation. Static, GUI-tunable + saved.
RIGHT_RGB_KEY = "images.right_rgb"
RIGHT_CAM_LOCAL_POS = (2.06, -0.02, 2.33)
RIGHT_CAM_LOCAL_ROT_WXYZ = quat_normalize((0.865352, 0.501148, -0.003727, 0.001788))  # (w,x,y,z)
RIGHT_CAM_CONVENTION = "opengl"
RIGHT_PROFILE = CameraProfile(
    name=RIGHT_CAM_NAME,
    prim_path=RIGHT_CAM_PRIM,
    consumer="vla_pi05",
    kind="rgbd",                  # RGB + depth
    width=256,
    height=256,
    focal_length=24.0,            # same lens as the front/left cameras
    horizontal_aperture=20.955,
    clipping_range=(0.05, 8.0),
    offset_pos=RIGHT_CAM_LOCAL_POS,
    offset_rot=RIGHT_CAM_LOCAL_ROT_WXYZ,
    convention=RIGHT_CAM_CONVENTION,
    depth=True,
    image_keys=(RIGHT_RGB_KEY, "images.right_depth"),
    frame_id=RIGHT_CAM_NAME,
    parent="env_root",
    static=True,
    role="right_image",
    notes="Fixed third-person RIGHT camera (RGB+depth), faces the robot from its right side. Same config as the front/left cams, depth enabled.",
)

# Fixed TOP-DOWN camera -> RGB + depth, directly above the robot looking straight
# down (-Z). Same lens/resolution/depth as the other third-person cams. Identity
# rotation = an OpenGL camera looks down world -Z (image-up = world +Y). Centered
# over the robot base so the robot + its workspace are framed from directly above.
# Static (under env root), GUI-tunable + saved like the other static cams.
TOP_RGB_KEY = "images.top_rgb"
TOP_CAM_LOCAL_POS = (2.12, 2.01, 3.0)
TOP_CAM_LOCAL_ROT_WXYZ = (1.0, 0.0, 0.0, 0.0)   # identity -> straight down
TOP_CAM_CONVENTION = "opengl"
TOP_PROFILE = CameraProfile(
    name=TOP_CAM_NAME,
    prim_path=TOP_CAM_PRIM,
    consumer="vla_pi05",
    kind="rgbd",                  # RGB + depth
    width=256,
    height=256,
    focal_length=24.0,            # same lens as the front/left/right cameras
    horizontal_aperture=20.955,
    clipping_range=(0.05, 8.0),
    offset_pos=TOP_CAM_LOCAL_POS,
    offset_rot=TOP_CAM_LOCAL_ROT_WXYZ,
    convention=TOP_CAM_CONVENTION,
    depth=True,
    image_keys=(TOP_RGB_KEY, "images.top_depth"),
    frame_id=TOP_CAM_NAME,
    parent="env_root",
    static=True,
    role="top_image",
    notes="Fixed TOP-DOWN camera (RGB+depth), directly above the robot looking straight down. Same config as the other third-person cams, depth enabled.",
)

# Supported VLA resolutions (CLI --vla_camera_resolution).
VLA_SUPPORTED_RESOLUTIONS = ((128, 128), (224, 224))

PROFILES = {FP_CAM_NAME: FP_PROFILE, VLA_CAM_NAME: VLA_PROFILE,
            FRONT_CAM_NAME: FRONT_PROFILE, LEFT_CAM_NAME: LEFT_PROFILE,
            RIGHT_CAM_NAME: RIGHT_PROFILE, TOP_CAM_NAME: TOP_PROFILE}
V1_CAMERA_NAMES = (FP_CAM_NAME, VLA_CAM_NAME, FRONT_CAM_NAME, LEFT_CAM_NAME, RIGHT_CAM_NAME, TOP_CAM_NAME)


def get_profile(name: str) -> CameraProfile:
    if name not in PROFILES:
        raise KeyError(f"unknown V1 camera {name!r}; expected one of {list(PROFILES)}")
    return PROFILES[name]


def default_sensors_block(show_body: bool = False) -> dict:
    """The full ``sensors`` block for the registry (all three cameras)."""
    return {
        FP_CAM_NAME: FP_PROFILE.descriptor(enabled=True, visible_body=show_body),
        VLA_CAM_NAME: VLA_PROFILE.descriptor(enabled=True, visible_body=show_body),
        FRONT_CAM_NAME: FRONT_PROFILE.descriptor(enabled=True, visible_body=False),
        LEFT_CAM_NAME: LEFT_PROFILE.descriptor(enabled=True, visible_body=False),
        RIGHT_CAM_NAME: RIGHT_PROFILE.descriptor(enabled=True, visible_body=False),
        TOP_CAM_NAME: TOP_PROFILE.descriptor(enabled=True, visible_body=False),
    }


# ---------------------------------------------------------------------------
# backwards-compat shims (old single-D435 names → FP camera)
# ---------------------------------------------------------------------------
D435_SENSOR_NAME = FP_CAM_NAME
D435_FRAME_ID = FP_CAM_NAME
D435_MOUNT_PARENT = WRIST_MOUNT_PARENT
D435_COLOR_PRIM_PATH = FP_CAM_PRIM
D435_BODY_PRIM_PATH = D435_BODY_PRIM
D435_LOCAL_POS = FP_PROFILE.offset_pos
D435_LOCAL_ROT = FP_PROFILE.offset_rot
D435_CONVENTION = FP_PROFILE.convention
D435_RGB_WIDTH = FP_PROFILE.width
D435_RGB_HEIGHT = FP_PROFILE.height
D435_WRIST_RGB_KEY = "images.wrist_rgb"
D435_WRIST_DEPTH_KEY = "images.wrist_depth"
D435_RGB_DATA_TYPE = RGB_DATA_TYPE
D435_DEPTH_DATA_TYPE = DEPTH_DATA_TYPE
D435_MASK_DATA_TYPE = MASK_DATA_TYPE


def intrinsics_dict() -> dict:
    """Back-compat: nominal FoundationPose-camera intrinsics."""
    return FP_PROFILE.intrinsics()


def intrinsics_matrix() -> list:
    return FP_PROFILE.intrinsics_matrix()


def sensor_descriptor(enabled: bool = True, rgb: bool = True, depth: bool = True,
                      consumer_modules=None) -> dict:
    """Back-compat single-sensor descriptor (FoundationPose camera)."""
    return FP_PROFILE.descriptor(enabled=enabled)
