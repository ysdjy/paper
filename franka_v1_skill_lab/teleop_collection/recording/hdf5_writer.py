# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Write V1 teleop episodes to HDF5. STATUS: ready (needs h5py + numpy).

One HDF5 file per collection *session*; each saved episode is a ``data/demo_<i>``
group. The layout matches teleop_collection/data_format/demo_hdf5_schema.md and
is consumed by pi05_training/adapters/skill_demo_to_lerobot.py.

    <out_dir>/teleop_demos_<stamp>.hdf5
      attrs: schema_version, task_id, control_mode, scene_registry, created_unix
      data/                       attrs: num_demos
        demo_0/
          attrs: episode_id, task_instruction, skill_type, target_name,
                 success, num_steps, seed
          obs/   joint_pos(T,7) joint_vel(T,7) gripper_width(T,1) ee_pose(T,7)
                 objects/<name>(T,7)  [x,y,z,qw,qx,qy,qz]      (optional)
                 images/wrist_rgb(T,H,W,3) u8  wrist_depth(T,H,W) f32  (optional)
          actions/ joint_target(T,7)  gripper_command(T,1)
          teleop/  raw_q(T,7)  filtered_q(T,7)
          timestamps(T,)

h5py / numpy are imported lazily so the module imports in a plain venv (only the
write path needs them).
"""

from __future__ import annotations

import os
from pathlib import Path

SCHEMA_VERSION = "franka-scene-v1"


class TeleopHDF5Writer:
    """Append-style writer: open once per session, ``write_episode`` per demo."""

    def __init__(
        self,
        out_dir: str | Path,
        *,
        task_id: str,
        scene_registry: str = "",
        control_mode: str = "joint",
        stamp: str | None = None,
        update_latest: bool = True,
    ):
        import time

        import h5py  # noqa: F401  (validate availability early)

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
        self.path = self.out_dir / f"teleop_demos_{self._stamp}.hdf5"
        self.update_latest = update_latest
        self._num_demos = 0
        self._task_id = task_id

        self._f = h5py.File(self.path, "w")
        self._f.attrs["schema_version"] = SCHEMA_VERSION
        self._f.attrs["task_id"] = task_id
        self._f.attrs["control_mode"] = control_mode
        self._f.attrs["scene_registry"] = scene_registry or ""
        self._f.attrs["created_unix"] = float(time.time())
        self._data = self._f.create_group("data")

    # ------------------------------------------------------------------ #
    def write_episode(self, frames: list[dict], meta: dict) -> str:
        """Write one episode (list of frame dicts + episode meta) as demo_<i>.

        Returns the demo group name. No-op (returns "") if frames is empty.
        """
        import numpy as np

        if not frames:
            return ""

        T = len(frames)
        demo_name = f"demo_{self._num_demos}"
        g = self._data.create_group(demo_name)
        for k, v in meta.items():
            g.attrs[k] = v

        def stack(getter, dim):
            arr = np.zeros((T, dim), dtype=np.float32)
            for t, fr in enumerate(frames):
                vals = getter(fr)
                arr[t, : len(vals)] = np.asarray(vals[:dim], dtype=np.float32)
            return arr

        obs = g.create_group("obs")
        obs.create_dataset("joint_pos", data=stack(lambda f: f["robot"]["joint_pos"], 7))
        obs.create_dataset("joint_vel", data=stack(lambda f: f["robot"]["joint_vel"], 7))
        obs.create_dataset("ee_pose", data=stack(lambda f: f["robot"]["ee_pose"], 7))
        obs.create_dataset(
            "gripper_width",
            data=stack(lambda f: [f["robot"]["gripper_width"]], 1),
        )

        actions = g.create_group("actions")
        actions.create_dataset("joint_target", data=stack(lambda f: f["action"]["joint_target"], 7))
        actions.create_dataset(
            "gripper_command",
            data=stack(lambda f: [f["action"]["gripper"]], 1),
        )

        teleop = g.create_group("teleop")
        teleop.create_dataset("raw_q", data=stack(lambda f: f["teleop"]["raw_q"], 7))
        teleop.create_dataset("filtered_q", data=stack(lambda f: f["teleop"]["filtered_q"], 7))

        ts = np.asarray([fr["timestamp"] for fr in frames], dtype=np.float64)
        g.create_dataset("timestamps", data=ts)

        # --- optional: object GT poses (assume constant object set) ----------
        if meta.get("has_objects"):
            names = [o["name"] for o in frames[0].get("objects", [])]
            if names:
                obj_grp = obs.create_group("objects")
                for name in names:
                    pose = np.zeros((T, 7), dtype=np.float32)  # [x,y,z, qw,qx,qy,qz]
                    for t, fr in enumerate(frames):
                        match = next((o for o in fr.get("objects", []) if o["name"] == name), None)
                        if match is not None:
                            pose[t, :3] = match.get("pos", [0, 0, 0])[:3]
                            pose[t, 3:7] = match.get("quat_wxyz", [1, 0, 0, 0])[:4]
                    obj_grp.create_dataset(name, data=pose)

        # --- optional: images -------------------------------------------------
        img0 = frames[0].get("images", {})
        have_front = meta.get("has_images") and img0.get("front_rgb") is not None   # LeRobot `image`
        have_rgb = meta.get("has_images") and img0.get("wrist_rgb") is not None     # LeRobot `wrist_image`
        have_depth = meta.get("has_depth") and img0.get("wrist_depth") is not None
        if have_front or have_rgb or have_depth:
            img_grp = obs.create_group("images")

            def _write_rgb(key: str):
                first = np.asarray(frames[0]["images"][key])
                H, W = first.shape[0], first.shape[1]
                rgb = np.zeros((T, H, W, 3), dtype=np.uint8)
                for t, fr in enumerate(frames):
                    im = fr.get("images", {}).get(key)
                    if im is not None:
                        rgb[t] = np.asarray(im)[..., :3].astype(np.uint8)
                img_grp.create_dataset(key, data=rgb, compression="gzip", compression_opts=4)

            if have_front:
                _write_rgb("front_rgb")
            if have_rgb:
                _write_rgb("wrist_rgb")
            if have_depth:
                firstd = np.asarray(frames[0]["images"]["wrist_depth"])
                H, W = firstd.shape[0], firstd.shape[1]
                depth = np.zeros((T, H, W), dtype=np.float32)
                for t, fr in enumerate(frames):
                    im = fr.get("images", {}).get("wrist_depth")
                    if im is not None:
                        depth[t] = np.asarray(im, dtype=np.float32).reshape(H, W)
                img_grp.create_dataset("wrist_depth", data=depth, compression="gzip", compression_opts=4)

        self._num_demos += 1
        self._data.attrs["num_demos"] = self._num_demos
        self._f.flush()
        return demo_name

    # ------------------------------------------------------------------ #
    @property
    def num_demos(self) -> int:
        return self._num_demos

    def close(self) -> str:
        """Close the file and (optionally) refresh ``latest.hdf5``. Returns path."""
        self._data.attrs["num_demos"] = self._num_demos
        self._f.close()
        if self.update_latest and self._num_demos > 0:
            latest = self.out_dir / "latest.hdf5"
            try:
                if latest.exists() or latest.is_symlink():
                    latest.unlink()
                os.symlink(self.path.name, latest)
            except OSError:
                import shutil

                shutil.copyfile(self.path, latest)
        return str(self.path)
