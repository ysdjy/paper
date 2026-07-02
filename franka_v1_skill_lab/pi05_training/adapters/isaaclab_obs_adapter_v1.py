# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""V1 observation adapter: IsaacLab joint env -> policy observation. STATUS: ready (builder), wrapper (Isaac obs capture).

Builds the unified V1 observation the pi0.5 policy server consumes. The V1
contract is JOINT state (not EE-delta): joint_pos(7), joint_vel(7),
gripper_width(1), ee_pose(7, aux), and optionally object_poses from the
perception layer (sim-GT or FoundationPose) and camera images.

Pure python over lists so it can be unit-tested without Isaac. The Isaac-side
glue (pulling joint_vel / images off the env) reuses
``pi05_isaacsim_baseline/scripts/isaaclab/isaac_obs_utils.py``; this module just
defines and assembles the V1 shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

OBS_SPEC = {
    "joint_pos": 7,
    "joint_vel": 7,
    "gripper_width": 1,
    "ee_pose": 7,          # [x,y,z,qx,qy,qz,qw]
}

# Logical image keys the V1 observation carries. Mapping to LeRobot/pi0.5 fields:
#   image       <- vla_front_static     (images.front_rgb / images.image)
#   wrist_image <- vla_libero_eye_in_hand(images.wrist_rgb / eye_in_hand_rgb / robot0_eye_in_hand_rgb)
#   (depth)     <- foundationpose_d435_rgbd (images.wrist_depth)
FRONT_RGB_KEY = "images.front_rgb"                     # main third-person view (front cam)
LEROBOT_IMAGE_KEY = "images.image"                     # LeRobot `image` alias
WRIST_RGB_KEY = "images.wrist_rgb"                     # project-unified wrist RGB (VLA cam)
EYE_IN_HAND_RGB_KEY = "images.eye_in_hand_rgb"
LIBERO_RGB_KEY = "images.robot0_eye_in_hand_rgb"       # LIBERO-compatible
WRIST_DEPTH_KEY = "images.wrist_depth"                 # FoundationPose cam depth

# LeRobot field -> which logical image key feeds it.
LEROBOT_FIELD_TO_KEY = {"image": FRONT_RGB_KEY, "wrist_image": WRIST_RGB_KEY}


def wrist_images_from_frame(frame: dict | None) -> dict:
    """Map ONE wrist-camera frame to its logical image keys.

    Delegates to ``sensors.d435.logical_image_dict`` so the VLA frame yields the
    three RGB keys (wrist_rgb / eye_in_hand_rgb / robot0_eye_in_hand_rgb) and the
    FoundationPose frame yields ``images.wrist_depth``. Missing modalities are
    skipped (never silently zero-filled).
    """
    if not frame:
        return {}
    try:
        from franka_v1_skill_lab.sensors.d435.d435_observation_adapter import logical_image_dict

        return logical_image_dict(frame)
    except Exception:
        # minimal fallback if the sensors pkg is unavailable
        out = {}
        if frame.get("rgb") is not None:
            out[WRIST_RGB_KEY] = frame["rgb"]
        if frame.get("depth") is not None:
            out[WRIST_DEPTH_KEY] = frame["depth"]
        return out


def wrist_images_from_frames(*frames) -> dict:
    """Merge logical image dicts from several camera frames (VLA + FP)."""
    out: dict = {}
    for f in frames:
        out.update(wrist_images_from_frame(f))
    return out


# Back-compat alias for the previous single-camera helper name.
def wrist_images_from_d435(frame: dict | None) -> dict:
    return wrist_images_from_frame(frame)


@dataclass
class V1Observation:
    joint_pos: list[float]
    joint_vel: list[float]
    gripper_width: float
    ee_pose: list[float]
    object_poses: list[dict] = field(default_factory=list)   # PoseEntry dicts
    images: dict = field(default_factory=dict)               # name -> array
    control_mode: str = "joint"

    def to_dict(self) -> dict:
        return {
            "joint_pos": [float(v) for v in self.joint_pos],
            "joint_vel": [float(v) for v in self.joint_vel],
            "gripper_width": float(self.gripper_width),
            "ee_pose": [float(v) for v in self.ee_pose],
            "object_poses": self.object_poses,
            "images": {k: v for k, v in self.images.items()},
            "control_mode": self.control_mode,
        }


def build_observation(
    joint_pos,
    joint_vel,
    gripper_width: float,
    ee_pose,
    object_poses: list[dict] | None = None,
    images: dict | None = None,
) -> V1Observation:
    """Assemble a V1Observation, validating the joint-state shapes."""

    def _as_list(x, n, name):
        try:
            x = x.detach().cpu().tolist()
        except AttributeError:
            x = list(x)
        if len(x) < n:
            raise ValueError(f"{name} needs {n} elements, got {len(x)}")
        return [float(v) for v in x[:n]]

    return V1Observation(
        joint_pos=_as_list(joint_pos, 7, "joint_pos"),
        joint_vel=_as_list(joint_vel, 7, "joint_vel"),
        gripper_width=float(gripper_width),
        ee_pose=_as_list(ee_pose, 7, "ee_pose"),
        object_poses=list(object_poses or []),
        images=dict(images or {}),
    )


def observation_from_provider(
    provider,
    object_poses: list[dict] | None = None,
    front_frame: dict | None = None,
    vla_frame: dict | None = None,
    fp_frame: dict | None = None,
    d435_frame: dict | None = None,
) -> V1Observation:
    """Build a V1Observation from a SceneStateProvider state.

    Expects ``state.robot`` to expose ``joint_pos``, ``joint_vel`` (or 0s),
    ``gripper_width`` and ``tcp_pose``. Object poses come from the perception
    adapter (already PoseEntry dicts).

    Camera frames are merged into ``images``:
      * ``front_frame`` (vla_front_static)     -> images.front_rgb / images.image  (LeRobot `image`);
      * ``vla_frame`` (vla_libero_eye_in_hand) -> images.wrist_rgb / eye_in_hand_rgb / robot0_eye_in_hand_rgb (`wrist_image`);
      * ``fp_frame`` (foundationpose_d435_rgbd) -> images.wrist_depth;
      * ``d435_frame`` is a back-compat alias for ``fp_frame``.
    """
    state = provider.get_state()
    robot = state.robot
    joint_vel = getattr(robot, "joint_vel", None)
    if joint_vel is None:
        joint_vel = [0.0] * 7
    return build_observation(
        joint_pos=getattr(robot, "joint_pos", [0.0] * 7),
        joint_vel=joint_vel,
        gripper_width=getattr(robot, "gripper_width", 0.0),
        ee_pose=getattr(robot, "tcp_pose", [0.0] * 7),
        object_poses=object_poses,
        images=wrist_images_from_frames(front_frame, vla_frame, fp_frame or d435_frame),
    )


def _main() -> int:
    """Offline dry-run: print the V1 observation schema + a camera's image keys."""
    import argparse
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    ap = argparse.ArgumentParser(description="V1 pi0.5/VLA observation adapter (dry run).")
    ap.add_argument("--scene_registry", default=None, help="(accepted for parity; unused offline)")
    ap.add_argument("--camera", default="vla_libero_eye_in_hand",
                    help="Which wrist camera to preview image keys for.")
    ap.add_argument("--dry_run", action="store_true", help="Print schema + keys, no Isaac.")
    args = ap.parse_args()

    from franka_v1_skill_lab.sensors.d435 import d435_config as cfg
    from franka_v1_skill_lab.sensors.d435.d435_observation_adapter import WristCameraAdapter

    prof = cfg.PROFILES.get(args.camera, cfg.VLA_PROFILE)
    frame = WristCameraAdapter.mock_frame(args.camera)
    img_keys = list(wrist_images_from_frame(frame).keys())

    obs = build_observation(
        joint_pos=[0.0] * 7, joint_vel=[0.0] * 7, gripper_width=0.04,
        ee_pose=[0.4, 0.0, 0.3, 0, 0, 0, 1], images=wrist_images_from_frame(frame),
    )
    lerobot_field = {"image": "image", "wrist_image": "wrist_image", "depth": "(depth, not a LeRobot image field)"}.get(prof.role, "?")
    print("=== V1 pi0.5 / VLA observation (dry run) ===")
    print(f"camera        : {args.camera} (consumer={prof.consumer}, kind={prof.kind}, role={prof.role})")
    print(f"static        : {prof.static} (parent={prof.parent})")
    print(f"LeRobot field : {lerobot_field}   <- this camera")
    print(f"resolution    : {prof.intrinsics()['width']}x{prof.intrinsics()['height']}")
    print(f"image keys    : {img_keys}")
    print(f"obs spec      : {json.dumps(OBS_SPEC)}")
    print(f"obs.images    : {list(obs.to_dict()['images'].keys())}")
    print("control_mode  : joint")
    print("\nLeRobot mapping (pipeline): image <- vla_front_static | wrist_image <- vla_libero_eye_in_hand")
    if prof.libero_compatible:
        print("LIBERO        : images.robot0_eye_in_hand_rgb is the LIBERO-compatible wrist key")
    print("\nOK — observation adapter dry run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
