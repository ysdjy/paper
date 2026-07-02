#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Convert V1 joint teleop demos (HDF5) -> normalized dataset (+ LeRobot, best effort). STATUS: ready.

V1 mainline is a JOINT action space:
    observation.state = joint_pos(7) + gripper_width(1)   = 8
    action            = joint_target(7) + gripper_command(1) = 8   (NOT an EE delta)

This reads the HDF5 written by
``teleop_collection/entries/collect_teleop_demos_joint_v1.py`` (layout in
``teleop_collection/data_format/demo_hdf5_schema.md``) and:

  1. ALWAYS writes a normalized intermediate dataset (numpy) — the guaranteed,
     dependency-light output.
  2. Best-effort builds a LeRobot dataset if ``lerobot`` is importable; failure
     here NEVER fails the run (the normalized dataset is still produced).
  3. Writes a conversion report JSON.

Only ``success`` demos are exported by default (``--include_failed`` to override),
so failed/discarded episodes never leak into the training set.

Smoke (no HDF5, no deps): ``--dry_run`` validates the V1 field mapping.
Real:
    python skill_demo_to_lerobot.py \
        --input  .../data/teleop_raw_hdf5/latest.hdf5 \
        --output .../data/lerobot/franka_v1_skill_pi05 \
        --task_config .../configs/teleop_tasks.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# V1 HDF5 field -> LeRobot/normalized key mapping (joint action space).
V1_FIELD_MAPPING = {
    "observation.state": "obs/joint_pos (+ obs/gripper_width)",  # 7 + 1
    "observation.joint_vel": "obs/joint_vel",                     # 7 (aux)
    "observation.ee_pose": "obs/ee_pose",                         # 7 (aux)
    "observation.images.image": "obs/images/front_rgb",           # LeRobot `image` (front, optional)
    "observation.images.wrist": "obs/images/wrist_rgb",           # LeRobot `wrist_image` (optional)
    "observation.images.wrist_depth": "obs/images/wrist_depth",   # optional
    "action": "actions/joint_target (+ actions/gripper_command)",  # 7 + 1  (NOT EE delta)
}
REQUIRED_KEYS = ("observation.state", "action")

DEFAULT_REPORT = "projects/franka_v1_skill_lab/data/reports/convert_latest_report.json"
DEFAULT_NORMALIZED_ROOT = "projects/franka_v1_skill_lab/data/processed/normalized_dataset"


def validate_mapping(mapping: dict) -> list[str]:
    problems = []
    for key in REQUIRED_KEYS:
        if key not in mapping:
            problems.append(f"missing required mapping key: {key}")
    action = mapping.get("action", "")
    if "delta_ee" in action or "ee_delta" in action:
        problems.append("action maps to an EE-delta field — V1 must be a JOINT target")
    return problems


def _episode_instruction(demo_group, task_catalog: dict) -> str:
    instr = demo_group.attrs.get("task_instruction", "")
    if isinstance(instr, bytes):
        instr = instr.decode()
    if instr:
        return str(instr)
    # fall back to catalog default
    return task_catalog.get("_default_instruction", "Perform the manipulation task.")


def convert(
    input_path: str,
    output_path: str,
    *,
    task_catalog: dict,
    include_failed: bool = False,
    image_resize=(256, 256),
    fps: int = 30,
    build_lerobot: bool = True,
    normalized_root: str = DEFAULT_NORMALIZED_ROOT,
) -> dict:
    import h5py
    import numpy as np

    report: dict = {"input": input_path, "episodes": 0, "frames": 0, "skipped_failed": 0, "missing": []}

    f = h5py.File(input_path, "r")
    root = f["data"] if "data" in f else f
    demo_keys = sorted([k for k in root.keys()], key=lambda s: (len(s), s))
    if not demo_keys:
        f.close()
        raise RuntimeError("No demos found under 'data/'.")

    name = os.path.basename(output_path.rstrip("/"))
    norm_dir = os.path.join(normalized_root, name)
    os.makedirs(norm_dir, exist_ok=True)
    img_root = os.path.join(norm_dir, "images")

    all_states, all_actions, all_epidx, episodes_meta, all_instructions = [], [], [], [], []
    have_images = False    # wrist_rgb -> LeRobot `wrist_image`
    have_front = False     # front_rgb -> LeRobot `image`

    ei = 0
    for dk in demo_keys:
        ep = root[dk]
        success = bool(ep.attrs.get("success", True))
        if not success and not include_failed:
            report["skipped_failed"] += 1
            continue
        if "obs/joint_pos" not in ep or "actions/joint_target" not in ep:
            report["missing"].append(f"{dk}: obs/joint_pos or actions/joint_target")
            continue

        joint_pos = np.asarray(ep["obs/joint_pos"], dtype=np.float32)        # (T,7)
        T = joint_pos.shape[0]
        grip_w = np.asarray(ep["obs/gripper_width"], dtype=np.float32).reshape(T, -1)[:, :1]
        state = np.concatenate([joint_pos, grip_w], axis=1)                  # (T,8)

        jt = np.asarray(ep["actions/joint_target"], dtype=np.float32)        # (T,7)
        gcmd = np.asarray(ep["actions/gripper_command"], dtype=np.float32).reshape(T, -1)[:, :1]
        action = np.concatenate([jt, gcmd], axis=1)                          # (T,8)

        all_states.append(state)
        all_actions.append(action)
        all_epidx.append(np.full((T,), ei, dtype=np.int64))
        instr = _episode_instruction(ep, task_catalog)
        all_instructions.append(instr)

        # optional wrist RGB -> PNG per frame
        if "obs/images/wrist_rgb" in ep:
            have_images = True
            try:
                from PIL import Image

                cam_dir = os.path.join(img_root, "wrist")
                os.makedirs(cam_dir, exist_ok=True)
                imgs = ep["obs/images/wrist_rgb"]
                for t in range(T):
                    im = np.asarray(imgs[t])
                    if im.dtype != np.uint8:
                        im = np.clip(im, 0, 255).astype(np.uint8)
                    Image.fromarray(im[..., :3]).resize(image_resize).save(
                        os.path.join(cam_dir, f"ep{ei:04d}_step{t:06d}.png")
                    )
            except Exception as exc:  # noqa
                report.setdefault("image_errors", []).append(f"{dk}: {exc}")

        # optional front RGB -> PNG per frame  (LeRobot `image`, main third-person view)
        if "obs/images/front_rgb" in ep:
            have_front = True
            try:
                from PIL import Image

                cam_dir = os.path.join(img_root, "front")
                os.makedirs(cam_dir, exist_ok=True)
                imgs = ep["obs/images/front_rgb"]
                for t in range(T):
                    im = np.asarray(imgs[t])
                    if im.dtype != np.uint8:
                        im = np.clip(im, 0, 255).astype(np.uint8)
                    Image.fromarray(im[..., :3]).resize(image_resize).save(
                        os.path.join(cam_dir, f"ep{ei:04d}_step{t:06d}.png")
                    )
            except Exception as exc:  # noqa
                report.setdefault("image_errors", []).append(f"{dk} (front): {exc}")

        episodes_meta.append({
            "episode_index": ei, "demo_key": dk, "length": int(T),
            "instruction": instr, "skill_type": str(ep.attrs.get("skill_type", "")),
            "target_name": str(ep.attrs.get("target_name", "")), "success": success,
        })
        ei += 1

    f.close()

    if not all_actions:
        raise RuntimeError("No usable success demos extracted (use --include_failed to include failed).")

    states_cat = np.concatenate(all_states, axis=0)
    actions_cat = np.concatenate(all_actions, axis=0)
    epidx_cat = np.concatenate(all_epidx, axis=0)
    report["episodes"] = len(episodes_meta)
    report["frames"] = int(actions_cat.shape[0])

    np.save(os.path.join(norm_dir, "states.npy"), states_cat)
    np.save(os.path.join(norm_dir, "actions.npy"), actions_cat)
    np.save(os.path.join(norm_dir, "episode_index.npy"), epidx_cat)
    with open(os.path.join(norm_dir, "episodes.jsonl"), "w") as fp:
        for e in episodes_meta:
            fp.write(json.dumps(e) + "\n")
    meta = {
        "name": name, "source_hdf5": input_path, "fps": fps,
        "num_episodes": len(episodes_meta), "num_frames": int(actions_cat.shape[0]),
        "state_dim": int(states_cat.shape[1]), "action_dim": int(actions_cat.shape[1]),
        "action_space": "joint (7 arm joint targets + 1 gripper command)",
        "state_layout": "joint_pos[7] + gripper_width[1]",
        "action_layout": "joint_target[7] + gripper_command[1]",
        "has_images": have_images, "has_front_image": have_front,
        "lerobot_features": {"image": "front_rgb", "wrist_image": "wrist_rgb", "state": "state", "actions": "actions"},
        "field_mapping": V1_FIELD_MAPPING,
    }
    with open(os.path.join(norm_dir, "metadata.json"), "w") as fp:
        json.dump(meta, fp, indent=2)
    report["normalized_dir"] = norm_dir
    report["state_dim"] = int(states_cat.shape[1])
    report["action_dim"] = int(actions_cat.shape[1])
    report["has_images"] = have_images
    report["has_front_image"] = have_front

    if build_lerobot:
        try:
            _build_lerobot(name, output_path, all_states, all_actions, all_instructions,
                           have_images, have_front, img_root, image_resize, fps)
            report["lerobot_dir"] = output_path
            report["lerobot_built"] = True
        except Exception as exc:  # noqa
            report["lerobot_built"] = False
            report["lerobot_error"] = str(exc)

    return report


def _build_lerobot(name, output_path, all_states, all_actions, all_instructions,
                   have_images, have_front, img_root, image_resize, fps):
    """Build a real LeRobotDataset. Raises if lerobot is unavailable (best effort)."""
    import numpy as np

    LeRobotDataset = None
    for modpath in ("lerobot.common.datasets.lerobot_dataset", "lerobot.datasets.lerobot_dataset"):
        try:
            mod = __import__(modpath, fromlist=["LeRobotDataset"])
            LeRobotDataset = getattr(mod, "LeRobotDataset")
            break
        except Exception:
            continue
    if LeRobotDataset is None:
        raise ImportError("lerobot not importable in this environment")

    import shutil

    from PIL import Image

    state_dim = all_states[0].shape[1]
    action_dim = all_actions[0].shape[1]
    H, W = image_resize
    features = {
        "state": {"dtype": "float32", "shape": (state_dim,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (action_dim,), "names": ["actions"]},
    }
    if have_front:
        features["image"] = {"dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"]}
    if have_images:
        features["wrist_image"] = {"dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"]}

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    ds = LeRobotDataset.create(repo_id=name, root=output_path, robot_type="panda", fps=fps, features=features)

    for ei, (states, actions) in enumerate(zip(all_states, all_actions)):
        T = actions.shape[0]
        instr = all_instructions[ei]
        for t in range(T):
            frame = {"state": states[t].astype(np.float32), "actions": actions[t].astype(np.float32), "task": instr}
            if have_front:
                pf = os.path.join(img_root, "front", f"ep{ei:04d}_step{t:06d}.png")
                frame["image"] = (np.asarray(Image.open(pf)) if os.path.exists(pf)
                                  else np.zeros((H, W, 3), np.uint8))
            if have_images:
                p = os.path.join(img_root, "wrist", f"ep{ei:04d}_step{t:06d}.png")
                frame["wrist_image"] = (np.asarray(Image.open(p)) if os.path.exists(p)
                                        else np.zeros((H, W, 3), np.uint8))
            ds.add_frame(frame)
        ds.save_episode()


def main() -> int:
    ap = argparse.ArgumentParser(description="V1 demo HDF5 -> normalized + LeRobot (joint action).")
    ap.add_argument("--input", default=None, help="Input HDF5 (e.g. data/teleop_raw_hdf5/latest.hdf5).")
    ap.add_argument("--output", default="projects/franka_v1_skill_lab/data/lerobot/franka_v1_skill_pi05")
    ap.add_argument("--task_config", default=None, help="configs/teleop_tasks.yaml (instruction fallback).")
    ap.add_argument("--include_failed", action="store_true", help="Also export failed demos (default: success only).")
    ap.add_argument("--no_lerobot", action="store_true", help="Only build the normalized dataset.")
    ap.add_argument("--report", default=DEFAULT_REPORT, help="Conversion report JSON path.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dry_run", action="store_true", help="Validate the V1 mapping, do not convert.")
    args = ap.parse_args()

    print("[skill_demo->lerobot] V1 field mapping (joint action space):")
    print(json.dumps(V1_FIELD_MAPPING, indent=2))
    problems = validate_mapping(V1_FIELD_MAPPING)
    if problems:
        print("MAPPING INVALID:")
        for p in problems:
            print(f"  - {p}")
        return 1

    if args.dry_run or args.input is None:
        print("[skill_demo->lerobot] --dry_run: mapping is V1-valid (joint action). No conversion run.")
        return 0

    task_catalog = {}
    if args.task_config and Path(args.task_config).is_file():
        import yaml

        cat = yaml.safe_load(Path(args.task_config).read_text(encoding="utf-8")) or {}
        tasks = {t["task_id"]: t for t in cat.get("tasks", [])}
        default = tasks.get(cat.get("default_task_id"), {})
        task_catalog = {"_default_instruction": default.get("instruction", "Perform the manipulation task.")}

    report = convert(
        args.input, args.output, task_catalog=task_catalog,
        include_failed=args.include_failed, fps=args.fps, build_lerobot=not args.no_lerobot,
    )
    print(json.dumps(report, indent=2))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[skill_demo->lerobot] report -> {args.report}")
    if report.get("missing"):
        print("[WARN] missing fields:", report["missing"], file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
