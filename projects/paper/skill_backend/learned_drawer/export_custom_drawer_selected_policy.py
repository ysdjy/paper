# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Export a trained custom selected-drawer policy to TorchScript ``policy.pt`` (then exit).

Reuses the rsl_rl runner load/export path (no training, no sim loop). Loads a custom-drawer
checkpoint (``--checkpoint``, or the latest under logs/rsl_rl/custom_drawer_selected) and writes a
TorchScript policy usable by the state machine's ``custom_selected_policy`` drawer backend.

    ./isaaclab.sh -p scripts/environments/state_machine/export_custom_drawer_selected_policy.py \
        --num_envs 1 --output_path logs/policies/custom_drawer_selected_policy.pt --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export the custom selected-drawer policy to TorchScript.")
parser.add_argument("--task", type=str, default="Isaac-Open-CustomDrawer-Selected-Franka-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a custom-drawer model_*.pt (default: latest).")
parser.add_argument("--output_path", type=str, default="logs/policies/custom_drawer_selected_policy.pt")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

import glob
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

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

EXPERIMENT_DIR = os.path.join("logs", "rsl_rl", "custom_drawer_selected")


def _latest_checkpoint() -> str:
    candidates = glob.glob(os.path.join(EXPERIMENT_DIR, "**", "model_*.pt"), recursive=True)
    if not candidates:
        raise RuntimeError(
            f"No checkpoints under {EXPERIMENT_DIR}. Train first (train.py or "
            "finetune_custom_drawer_from_official.py) or pass --checkpoint PATH."
        )
    return max(candidates, key=os.path.getmtime)


def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else _latest_checkpoint()

    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    env_cfg.seed = args_cli.seed
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    agent_device = getattr(agent_cfg, "device", device) or device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    print(f"[export] loading custom-drawer checkpoint: {resume_path}", flush=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_device)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    runner.load(resume_path)

    out_path = os.path.abspath(args_cli.output_path)
    out_dir, out_name = os.path.dirname(out_path), os.path.basename(out_path)
    os.makedirs(out_dir, exist_ok=True)

    if version.parse(installed_version) >= version.parse("4.0.0"):
        runner.export_policy_to_jit(path=out_dir, filename=out_name)
    else:
        policy_nn = runner.alg.policy if version.parse(installed_version) >= version.parse("2.3.0") else runner.alg.actor_critic
        normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=out_dir, filename=out_name)

    if not os.path.isfile(out_path):
        produced = os.path.join(out_dir, "policy.pt")
        if os.path.isfile(produced) and produced != out_path:
            shutil.copyfile(produced, out_path)

    print(f"[export] wrote TorchScript policy to: {out_path}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
