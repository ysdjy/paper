# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Export Isaac Lab's official Franka open-drawer PPO checkpoint to a TorchScript ``policy.pt``.

This reuses the rsl_rl checkpoint-loading / export logic (same as
``scripts/reinforcement_learning/rsl_rl/play.py``) but exits immediately after exporting — it does
NOT enter the simulation loop, train, or record video.

Example:
    ./isaaclab.sh -p scripts/environments/state_machine/export_official_drawer_policy.py \
        --task Isaac-Open-Drawer-Franka-Play-v0 \
        --use_pretrained_checkpoint \
        --output_path logs/policies/official_open_drawer_policy.pt \
        --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export the official Franka open-drawer PPO policy to TorchScript.")
parser.add_argument("--task", type=str, default="Isaac-Open-Drawer-Franka-Play-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (kept tiny: export only).")
parser.add_argument("--seed", type=int, default=1, help="Environment seed.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the published pretrained checkpoint.")
parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path (overrides pretrained).")
parser.add_argument(
    "--output_path",
    type=str,
    default="logs/policies/official_open_drawer_policy.pt",
    help="Where to write the exported TorchScript policy.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest follows after the app is launched."""

import importlib.metadata as metadata
import os
import shutil

import gymnasium as gym
from packaging import version

installed_version = metadata.version("rsl-rl-lib")

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg


def main():
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    device = args_cli.device if args_cli.device is not None else "cuda:0"

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    # convert deprecated (pre-5.0) runner cfg layout to the structure rsl-rl expects
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    agent_device = getattr(agent_cfg, "device", device) or device

    # resolve the checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            raise RuntimeError(
                f"No published pretrained checkpoint available for '{train_task_name}'. "
                "Provide one with --checkpoint PATH instead."
            )
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        raise ValueError("Provide --use_pretrained_checkpoint or --checkpoint PATH.")

    # build env + runner
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    print(f"[export] loading checkpoint: {resume_path}", flush=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_device)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    runner.load(resume_path)

    out_path = os.path.abspath(args_cli.output_path)
    out_dir = os.path.dirname(out_path)
    out_name = os.path.basename(out_path)
    os.makedirs(out_dir, exist_ok=True)

    if version.parse(installed_version) >= version.parse("4.0.0"):
        runner.export_policy_to_jit(path=out_dir, filename=out_name)
    else:
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic
        normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=out_dir, filename=out_name)

    # export_*_to_jit may write into a different filename; make sure the requested path exists
    if not os.path.isfile(out_path):
        produced = os.path.join(out_dir, "policy.pt")
        if os.path.isfile(produced) and produced != out_path:
            shutil.copyfile(produced, out_path)

    print(f"[export] wrote TorchScript policy to: {out_path}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
