# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical wrist-camera observation adapter for the V1 scene (two cameras).

Turns a live IsaacLab :class:`~isaaclab.sensors.Camera` (attached via
:func:`d435_scene_cfg.attach_wrist_cameras`) into the frame contract every V1
consumer agrees on. The SAME class reads either camera by name:

  * ``foundationpose_d435_rgbd`` — RGB + depth (FoundationPose). Depth is
    REQUIRED: if the renderer did not produce it, ``capture()`` raises rather
    than silently returning ``None``.
  * ``vla_libero_eye_in_hand``   — RGB (VLA/pi0.5). Depth optional.

Frame dict:
    {
      "name": "<camera name>",
      "consumer": "foundationpose" | "vla_pi05",
      "rgb":   HxWx3 uint8 | None,
      "depth": HxW float32 meters | None,
      "mask":  HxW int | None,
      "intrinsics": {fx, fy, cx, cy, width, height},
      "camera_pose_world": {"position": [x,y,z], "quat_wxyz": [w,x,y,z]},
      "image_keys": [...],
      "frame_id": "<name>",
      "timestamp": float,
    }

numpy/torch handling stays inside methods so the module imports in a plain venv
(the offline test only uses ``mock_frame``).
"""

from __future__ import annotations

from . import d435_config as cfg


# --- helpers ----------------------------------------------------------------
def _to_numpy(x):
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None
    if x is None:
        return None
    detach = getattr(x, "detach", None)
    if detach is not None:
        x = detach().cpu().numpy()
    elif np is not None and not isinstance(x, np.ndarray):
        try:
            x = np.asarray(x)
        except Exception:
            return x
    return x


def logical_image_dict(frame: dict) -> dict:
    """Map a frame to its logical image keys (depends on the camera).

    Keys containing ``depth`` receive the depth array; the rest receive the rgb
    array. Keys whose modality is missing on ``frame`` are skipped.
    """
    out: dict = {}
    keys = frame.get("image_keys") or []
    rgb = frame.get("rgb")
    depth = frame.get("depth")
    for k in keys:
        if "depth" in k:
            if depth is not None:
                out[k] = depth
        else:
            if rgb is not None:
                out[k] = rgb
    return out


class WristCameraAdapter:
    """Capture one wrist camera (by name) from a live IsaacLab env."""

    def __init__(self, env=None, camera_name: str = cfg.FP_CAM_NAME, env_index: int = 0):
        self.env = env
        self.camera_name = camera_name
        self.env_index = env_index
        self.profile = cfg.PROFILES.get(camera_name)

    # --- availability -------------------------------------------------------
    def is_available(self) -> bool:
        if self.env is None:
            return False
        try:
            return self.camera_name in self.env.unwrapped.scene.sensors
        except Exception:
            return False

    def _camera(self):
        return self.env.unwrapped.scene.sensors[self.camera_name]

    # --- intrinsics / pose --------------------------------------------------
    def intrinsics(self) -> dict:
        if self.is_available():
            try:
                cam = self._camera()
                k = _to_numpy(cam.data.intrinsic_matrices)[self.env_index]
                return {
                    "fx": float(k[0, 0]), "fy": float(k[1, 1]),
                    "cx": float(k[0, 2]), "cy": float(k[1, 2]),
                    "width": int(cam.image_shape[1]), "height": int(cam.image_shape[0]),
                }
            except Exception:
                pass
        return self.profile.intrinsics() if self.profile else cfg.intrinsics_dict()

    def camera_pose_world(self) -> dict:
        if self.is_available():
            try:
                cam = self._camera()
                pos = _to_numpy(cam.data.pos_w)[self.env_index]
                quat = _to_numpy(cam.data.quat_w_world)[self.env_index]
                return {
                    "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                    "quat_wxyz": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
                }
            except Exception:
                pass
        return {"position": [0.0, 0.0, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}

    # --- capture ------------------------------------------------------------
    def capture(self, require_depth: bool | None = None) -> dict:
        """Capture one frame from this camera.

        ``require_depth`` defaults to the camera profile's depth flag (so the
        FoundationPose camera requires depth). If depth is required but the
        renderer did not produce it, a RuntimeError is raised (never a silent
        ``None``).
        """
        if not self.is_available():
            raise RuntimeError(
                f"wrist camera {self.camera_name!r} not available on the live scene. "
                "Attach it with sensors.d435.attach_wrist_cameras(env_cfg) and launch "
                "with --enable_cameras, or use WristCameraAdapter.mock_frame(name)."
            )
        prof = self.profile
        want_depth = prof.depth if (require_depth is None and prof is not None) else bool(require_depth)

        cam = self._camera()
        out = cam.data.output
        rgb = depth = mask = None
        if cfg.RGB_DATA_TYPE in out:
            rgb = _to_numpy(out[cfg.RGB_DATA_TYPE])[self.env_index]
            if rgb is not None and getattr(rgb, "shape", (0, 0, 0))[-1] == 4:
                rgb = rgb[..., :3]
        if cfg.DEPTH_DATA_TYPE in out:
            depth = _to_numpy(out[cfg.DEPTH_DATA_TYPE])[self.env_index]
            if depth is not None and getattr(depth, "ndim", 0) == 3:
                depth = depth[..., 0]
        if cfg.MASK_DATA_TYPE in out:
            mask = _to_numpy(out[cfg.MASK_DATA_TYPE])[self.env_index]

        if want_depth and depth is None:
            raise RuntimeError(f"Depth output is not available for {self.camera_name}.")

        try:
            timestamp = float(self.env.unwrapped.sim.current_time)
        except Exception:
            timestamp = 0.0

        return {
            "name": self.camera_name,
            "consumer": prof.consumer if prof else "",
            "rgb": rgb,
            "depth": depth,
            "mask": mask,
            "intrinsics": self.intrinsics(),
            "camera_pose_world": self.camera_pose_world(),
            "image_keys": list(prof.image_keys) if prof else [],
            "frame_id": prof.frame_id if prof else self.camera_name,
            "timestamp": timestamp,
        }

    # --- offline fixture ----------------------------------------------------
    @staticmethod
    def mock_frame(camera_name: str = cfg.FP_CAM_NAME, with_depth: bool | None = None,
                   width: int | None = None, height: int | None = None,
                   with_mask: bool = False) -> dict:
        prof = cfg.PROFILES.get(camera_name, cfg.FP_PROFILE)
        w = int(width or prof.width)
        h = int(height or prof.height)
        depth_on = prof.depth if with_depth is None else with_depth
        rgb = depth = mask = None
        try:
            import numpy as np

            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            if depth_on:
                depth = np.full((h, w), 0.6, dtype=np.float32)
            if with_mask:
                mask = np.zeros((h, w), dtype=np.int32)
        except Exception:
            rgb = [[[0, 0, 0] for _ in range(w)] for _ in range(h)]
            if depth_on:
                depth = [[0.6 for _ in range(w)] for _ in range(h)]
        return {
            "name": camera_name,
            "consumer": prof.consumer,
            "rgb": rgb,
            "depth": depth,
            "mask": mask,
            "intrinsics": prof.intrinsics(w, h),
            "camera_pose_world": {"position": [0.0, 0.0, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
            "image_keys": list(prof.image_keys),
            "frame_id": prof.frame_id,
            "timestamp": 0.0,
        }


# Back-compat alias: the old name defaulted to the (now FoundationPose) camera.
class D435ObservationAdapter(WristCameraAdapter):
    def __init__(self, env=None, sensor_name: str = cfg.FP_CAM_NAME, env_index: int = 0):
        super().__init__(env=env, camera_name=sensor_name, env_index=env_index)
