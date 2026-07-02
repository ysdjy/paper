#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize fine-tuned pi0.5 inference vs ground truth on one episode. STATUS: ready (.venv_openpi).

Runs the fine-tuned policy frame-by-frame over a whole recorded episode and
plots predicted vs ground-truth action for all 8 dims (7 joint targets + gripper),
plus a filmstrip of camera frames at start / grasp / lift / place. Saves a PNG.

Run (OpenPI venv):
    pi05_isaacsim_baseline/.venv_openpi/bin/python \
        projects/franka_v1_skill_lab/pi05_training/visualize_inference_v1_stack.py \
        --ckpt <.../3000> --episode 0 --stride 3 --out <png>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULT_CKPT = (
    "pi05_isaacsim_baseline/policies/checkpoints/pi05_isaaclab_20260613_112349/"
    "pi05_franka_v1_stack/isaaclab_20260613_112349/3000"
)
DEFAULT_NORM = "projects/franka_v1_skill_lab/data/processed/normalized_dataset/franka_v1_stack_pi05"
PROMPT = "stack the blue cube on top of the red cube"
ACTION_DIM = 8
JOINT_LABELS = [f"joint_{i+1}" for i in range(7)] + ["gripper"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi05_franka_v1_stack")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--norm_dir", default=DEFAULT_NORM)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--stride", type=int, default=3, help="Infer every Nth frame (speed).")
    ap.add_argument("--out", default="projects/franka_v1_skill_lab/data/reports/pi05_inference_ep0.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    import openpi.policies.libero_policy as _libero
    from openpi.training import config as _config
    from openpi.policies import policy_config

    # keep the full 8-D action (7 joints + gripper); stock LiberoOutputs slices to 7.
    _libero.LiberoOutputs.__call__ = lambda self, data: {"actions": np.asarray(data["actions"])[:, :ACTION_DIM]}

    print(f"[viz] loading {args.config} @ {args.ckpt}", flush=True)
    cfg = _config.get_config(args.config)
    policy = policy_config.create_trained_policy(cfg, args.ckpt)

    norm = Path(args.norm_dir)
    states = np.load(norm / "states.npy")
    actions = np.load(norm / "actions.npy")
    epidx = np.load(norm / "episode_index.npy")
    front_dir = norm / "images" / "front"
    wrist_dir = norm / "images" / "wrist"

    frames = np.where(epidx == args.episode)[0]
    start = int(frames[0])
    steps = list(range(0, len(frames), args.stride))
    print(f"[viz] episode {args.episode}: {len(frames)} frames, inferring {len(steps)} (stride {args.stride})", flush=True)

    pred, gt, xs = [], [], []
    for s in steps:
        t = int(frames[s])
        fp = front_dir / f"ep{args.episode:04d}_step{s:06d}.png"
        wp = wrist_dir / f"ep{args.episode:04d}_step{s:06d}.png"
        if not fp.exists() or not wp.exists():
            continue
        obs = {
            "observation/state": states[t].astype(np.float32),
            "observation/image": np.asarray(Image.open(fp))[..., :3].astype(np.uint8),
            "observation/wrist_image": np.asarray(Image.open(wp))[..., :3].astype(np.uint8),
            "prompt": PROMPT,
        }
        a = np.asarray(policy.infer(obs)["actions"])[0]
        pred.append(a[:ACTION_DIM]); gt.append(actions[t][:ACTION_DIM]); xs.append(s)
    pred = np.array(pred); gt = np.array(gt); xs = np.array(xs)
    print(f"[viz] inferred {len(xs)} frames. joint L1={np.abs(pred[:,:7]-gt[:,:7]).mean():.4f} "
          f"gripper L1={np.abs(pred[:,7]-gt[:,7]).mean():.4f}", flush=True)

    # filmstrip key frames (front cam)
    key_fracs = [0.0, 0.33, 0.66, 0.99]
    key_steps = [min(int(f * (len(frames) - 1)), len(frames) - 1) for f in key_fracs]
    key_titles = ["start", "grasp", "lift", "place"]

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(4, 4, height_ratios=[1.4, 1, 1, 1], hspace=0.42, wspace=0.28)

    # row 0: camera filmstrip
    for i, (ks, title) in enumerate(zip(key_steps, key_titles)):
        ax = fig.add_subplot(gs[0, i])
        fp = front_dir / f"ep{args.episode:04d}_step{ks:06d}.png"
        if fp.exists():
            ax.imshow(np.asarray(Image.open(fp)))
        ax.set_title(f"{title}  (front cam, step {ks})", fontsize=10)
        ax.axis("off")

    # rows 1-3: 8 action dims (pred vs gt)
    for d in range(ACTION_DIM):
        ax = fig.add_subplot(gs[1 + d // 4, d % 4])
        ax.plot(xs, gt[:, d], color="tab:blue", lw=2.0, label="ground truth")
        ax.plot(xs, pred[:, d], color="tab:red", lw=1.4, ls="--", label="pi0.5 pred")
        ax.set_title(JOINT_LABELS[d], fontsize=10)
        ax.grid(alpha=0.3)
        if d == 0:
            ax.legend(fontsize=8, loc="best")
        if d == 7:
            ax.set_ylabel("0=open .. 1=closed", fontsize=8)
    fig.suptitle(
        f"pi0.5 (step-3000 LoRA) inference vs ground truth — episode {args.episode}\n"
        f"task: \"{PROMPT}\"   |   joint L1={np.abs(pred[:,:7]-gt[:,:7]).mean():.3f} rad, "
        f"gripper L1={np.abs(pred[:,7]-gt[:,7]).mean():.3f}",
        fontsize=13,
    )
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"[viz] saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
