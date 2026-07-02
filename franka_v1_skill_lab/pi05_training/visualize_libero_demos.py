#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize expert demos from the LIBERO LeRobot dataset (pi0.5 training data). STATUS: ready (.venv_openpi).

Downloads a FEW selected episodes (door/drawer/microwave "open & put" tasks) from
``physical-intelligence/libero`` and renders each to an mp4 (front view + wrist
view side by side). This is the dataset pi0.5 (pi05_libero) was trained on — it
shows the scenes / behaviours without running the model or installing the LIBERO
sim stack.

Run (OpenPI venv):
    pi05_isaacsim_baseline/.venv_openpi/bin/python \
        projects/franka_v1_skill_lab/pi05_training/visualize_libero_demos.py \
        --out_dir projects/franka_v1_skill_lab/data/libero_demos
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = "physical-intelligence/libero"
# (label, keyword to match in the episode task text)
TARGETS = [
    ("microwave_put_mug", "microwave"),
    ("open_top_drawer_put_bowl", "open the top drawer"),
    ("open_middle_drawer", "open the middle drawer"),
    ("bowl_bottom_drawer", "bottom drawer"),
]


def _to_uint8_hwc(x) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[2] not in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = (a * 255.0).clip(0, 255).astype(np.uint8) if a.max() <= 1.0 + 1e-6 else a.clip(0, 255).astype(np.uint8)
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    return a[..., :3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="projects/franka_v1_skill_lab/data/libero_demos")
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    import imageio.v2 as imageio
    from huggingface_hub import snapshot_download

    LeRobotDataset = None
    for modpath in ("lerobot.common.datasets.lerobot_dataset", "lerobot.datasets.lerobot_dataset"):
        try:
            LeRobotDataset = __import__(modpath, fromlist=["LeRobotDataset"]).LeRobotDataset
            break
        except Exception:
            continue
    if LeRobotDataset is None:
        print("lerobot not importable"); return 2

    meta_dir = Path(snapshot_download(repo_id=REPO, repo_type="dataset", allow_patterns=["meta/*"])) / "meta"
    eps = [json.loads(l) for l in open(meta_dir / "episodes.jsonl")]

    # choose first episode matching each target keyword
    chosen = {}
    for label, kw in TARGETS:
        for e in eps:
            tasks = e.get("tasks") or []
            if any(kw in t.lower() for t in tasks):
                chosen[label] = (int(e["episode_index"]), tasks[0])
                break
    print("[libero_viz] chosen episodes:")
    for label, (ei, task) in chosen.items():
        print(f"  {label}: ep{ei}  \"{task}\"")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    written = []
    for label, (ei, task) in chosen.items():
        print(f"[libero_viz] downloading + rendering ep{ei} ({label}) ...", flush=True)
        try:
            ds = LeRobotDataset(REPO, episodes=[ei])
        except Exception as exc:
            print(f"  load failed: {exc}"); continue
        frames = []
        for i in range(len(ds)):
            f = ds[i]
            img = _to_uint8_hwc(f["image"])
            wrist = _to_uint8_hwc(f["wrist_image"])
            # match heights, concat side by side (front | wrist)
            h = min(img.shape[0], wrist.shape[0])
            combo = np.concatenate([img[:h], wrist[:h]], axis=1)
            frames.append(combo)
        if not frames:
            print("  no frames"); continue
        mp4 = out / f"{label}_ep{ei}.mp4"
        imageio.mimsave(mp4, frames, fps=args.fps, quality=8)
        written.append((label, task, str(mp4), len(frames)))
        print(f"  wrote {mp4} ({len(frames)} frames)", flush=True)

    print("\n[libero_viz] done:")
    for label, task, p, n in written:
        print(f"  {p}  ({n} frames)  \"{task}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
