#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Offline open-loop eval of the fine-tuned pi0.5 stack policy. STATUS: ready (.venv_openpi).

Loads the fine-tuned pi0.5 checkpoint and runs inference on REAL recorded
observations (front image + wrist image + 8-D joint state + prompt) drawn from
the training dataset, then compares the predicted joint action against the
ground-truth recorded action. This validates that the model actually learned the
grasp->stack behaviour, without needing the full IsaacLab closed loop.

It feeds the dataset's own `state` (joint_pos[7] + gripper_width[1]) directly to
``policy.infer`` — the exact layout the model was trained on — so it does NOT go
through the EE-led obs path in the HTTP server.

Run (OpenPI venv):
    pi05_isaacsim_baseline/.venv_openpi/bin/python \
        projects/franka_v1_skill_lab/pi05_training/eval_offline_v1_stack.py \
        --ckpt <.../pi05_franka_v1_stack/.../3000> --frames 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEFAULT_CKPT = (
    "pi05_isaacsim_baseline/policies/checkpoints/pi05_isaaclab_20260613_112349/"
    "pi05_franka_v1_stack/isaaclab_20260613_112349/3000"
)
DEFAULT_NORM = "projects/franka_v1_skill_lab/data/processed/normalized_dataset/franka_v1_stack_pi05"
PROMPT = "stack the blue cube on top of the red cube"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi05_franka_v1_stack")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--norm_dir", default=DEFAULT_NORM)
    ap.add_argument("--frames", type=int, default=12, help="How many sampled frames to evaluate.")
    args = ap.parse_args()

    from PIL import Image

    from openpi.training import config as _config
    from openpi.policies import policy_config
    import openpi.policies.libero_policy as _libero

    # The stock LiberoOutputs hardcodes actions[:, :7] (LIBERO action = 6 EE-delta + 1
    # gripper). Our action is 8-D (7 joint targets + 1 gripper), so :7 would DROP the
    # gripper. pi0.5 itself outputs the full padded action; we just keep the first
    # ACTION_DIM dims (no retrain needed — the model already learned 8 dims). Patch the
    # output slice to our real action dim so the gripper survives.
    _ACTION_DIM = 8

    def _outputs8(self, data):
        return {"actions": np.asarray(data["actions"])[:, :_ACTION_DIM]}

    _libero.LiberoOutputs.__call__ = _outputs8
    print(f"[offline_eval] LiberoOutputs patched to keep {_ACTION_DIM} dims (7 joint + gripper).", flush=True)

    print(f"[offline_eval] loading config '{args.config}' + checkpoint {args.ckpt}", flush=True)
    cfg = _config.get_config(args.config)
    policy = policy_config.create_trained_policy(cfg, args.ckpt)
    print("[offline_eval] policy loaded.", flush=True)

    norm = Path(args.norm_dir)
    states = np.load(norm / "states.npy")            # (N, 8) joint_pos[7]+gripper
    actions = np.load(norm / "actions.npy")          # (N, 8) joint_target[7]+gripper_cmd
    epidx = np.load(norm / "episode_index.npy")      # (N,)
    front_dir = norm / "images" / "front"
    wrist_dir = norm / "images" / "wrist"

    # sample frames spread across several episodes (avoid only-first-frames)
    N = states.shape[0]
    sample = np.linspace(int(N * 0.05), int(N * 0.95), args.frames).astype(int)

    # frame -> (episode, step-in-episode) for PNG filename
    ep_start: dict[int, int] = {}
    for i, e in enumerate(epidx.tolist()):
        ep_start.setdefault(int(e), i)

    joint_l1, gripper_err = [], []
    print(f"\n{'frame':>6} {'ep':>3} {'joint_L1(rad)':>13} {'pred_grip':>10} {'gt_grip':>8}", flush=True)
    for t in sample:
        ei = int(epidx[t]); step = int(t - ep_start[ei])
        fp = front_dir / f"ep{ei:04d}_step{step:06d}.png"
        wp = wrist_dir / f"ep{ei:04d}_step{step:06d}.png"
        if not fp.exists() or not wp.exists():
            continue
        obs = {
            "observation/state": states[t].astype(np.float32),
            "observation/image": np.asarray(Image.open(fp))[..., :3].astype(np.uint8),
            "observation/wrist_image": np.asarray(Image.open(wp))[..., :3].astype(np.uint8),
            "prompt": PROMPT,
        }
        res = policy.infer(obs)
        pred = np.asarray(res["actions"])            # (horizon, >=7)
        pred0 = pred[0]
        gt = actions[t]                              # (8,)
        jl1 = float(np.abs(pred0[:7] - gt[:7]).mean())
        joint_l1.append(jl1)
        # gripper: LiberoOutputs slices to 7 dims, so pred may not carry gripper.
        pg = float(pred0[7]) if pred0.shape[0] > 7 else float("nan")
        if pred0.shape[0] > 7:
            gripper_err.append(abs(pg - float(gt[7])))
        print(f"{t:>6} {ei:>3} {jl1:>13.4f} {pg:>10.3f} {float(gt[7]):>8.3f}", flush=True)

    print("\n[offline_eval] summary:", flush=True)
    if joint_l1:
        print(f"  joint action L1 error: mean={np.mean(joint_l1):.4f} rad  max={np.max(joint_l1):.4f} rad "
              f"(over {len(joint_l1)} frames)", flush=True)
    if gripper_err:
        print(f"  gripper error: mean={np.mean(gripper_err):.4f}", flush=True)
    else:
        print("  gripper: pred sliced to 7 dims by LiberoOutputs (gripper not in pred head).", flush=True)
    print("  NOTE: open-loop on TRAINING data — low error => model fit the demos. "
          "True success needs the IsaacLab closed loop.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
