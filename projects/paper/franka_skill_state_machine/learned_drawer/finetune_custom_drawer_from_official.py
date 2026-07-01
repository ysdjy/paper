# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Fine-tune the custom selected-drawer policy from Isaac Lab's official Franka open-drawer PPO checkpoint.

Loads the official checkpoint into an RSL-RL runner for the custom selected-drawer task and continues
training. The custom task's obs (31-d) / action (8-d) / network ([256,128,64]) match the official cfg,
so the weights load directly; if a shape mismatch occurs, this script errors out clearly instead of
silently training from scratch.

Example (sanity):
    ./isaaclab.sh -p scripts/environments/state_machine/finetune_custom_drawer_from_official.py \
        --task Isaac-Open-CustomDrawer-Selected-Franka-v0 --use_published_official_checkpoint \
        --num_envs 128 --max_iterations 5 --headless --seed 1
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fine-tune custom selected-drawer policy from the official checkpoint.")
parser.add_argument("--task", type=str, default="Isaac-Open-CustomDrawer-Selected-Franka-v0")
parser.add_argument("--official_checkpoint_path", type=str, default=None, help="Path to an official rsl_rl checkpoint.pt.")
parser.add_argument("--use_published_official_checkpoint", action="store_true", help="Use the published open-drawer checkpoint.")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--max_iterations", type=int, default=5)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--log_dir", type=str, default=None, help="Override log dir (default logs/rsl_rl/custom_drawer_selected/finetune_<ts>).")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import os
import time

import gymnasium as gym
from packaging import version  # noqa: F401

installed_version = metadata.version("rsl-rl-lib")

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

OFFICIAL_TRAIN_TASK = "Isaac-Open-Drawer-Franka-v0"


def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"

    # resolve official checkpoint FIRST (fail fast, no env build if missing)
    if args_cli.official_checkpoint_path:
        resume_path = retrieve_file_path(args_cli.official_checkpoint_path)
    elif args_cli.use_published_official_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", OFFICIAL_TRAIN_TASK)
        if not resume_path:
            raise RuntimeError(f"No published checkpoint for {OFFICIAL_TRAIN_TASK}; pass --official_checkpoint_path.")
    else:
        raise ValueError("Provide --use_published_official_checkpoint or --official_checkpoint_path.")

    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    env_cfg.seed = args_cli.seed
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_device = getattr(agent_cfg, "device", device) or device

    log_dir = args_cli.log_dir or os.path.abspath(
        os.path.join("logs", "rsl_rl", "custom_drawer_selected", f"finetune_{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    )
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_device)

    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    print(f"[finetune] loading OFFICIAL checkpoint into custom-drawer runner: {resume_path}", flush=True)
    try:
        runner.load(resume_path)
    except (RuntimeError, ValueError) as exc:
        msg = str(exc)
        if "size mismatch" in msg or "shape" in msg or "mismatch" in msg:
            raise RuntimeError(
                "Official checkpoint network/obs/action shape does NOT match the custom selected-drawer "
                f"task. Refusing to silently train from scratch. Underlying error: {msg}"
            ) from exc
        raise

    print(f"[finetune] loaded. continuing training for {args_cli.max_iterations} iterations. log_dir={log_dir}", flush=True)
    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)
    env.close()
    print(f"[finetune] done. checkpoints under {log_dir}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
