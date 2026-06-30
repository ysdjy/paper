# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Generate open_drawer episodes for deployment-conditioned plan evaluation (paper round-1).

FULL reset per episode (independent episodes, fixes bias R1). Reuses the verified skill backend.
Writes <output_dir>/<run_id>/episodes.jsonl + trajectories/<id>.npz + metadata.

Usage:
  ./isaaclab.sh -p projects/deployment_calibration/data_generation/generate_open_drawer.py \
      --n_conditions 12 --candidates 3 --seed 7 --drawers top_drawer \
      --run_id kill_test --output_dir projects/deployment_calibration/data --headless
"""

from __future__ import annotations

import argparse
import os as _os
import sys as _sys
import time

# make both this paper package and the paper-owned (frozen) skill backend importable.
# Isolation: the paper imports its OWN skill_backend copy, NOT the shared franka_skill_state_machine,
# so platform/perception-project churn cannot infect paper experiments. The SCENE (gym task + assets
# in source/) stays shared on purpose.
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PAPER = _os.path.dirname(_os.path.dirname(_HERE))                             # projects/paper
_sys.path.insert(0, _os.path.dirname(_HERE))                                   # projects/paper/deployment_calibration
_sys.path.insert(0, _os.path.join(_PAPER, "skill_backend"))                    # projects/paper/skill_backend

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--n_conditions", type=int, default=12)
parser.add_argument("--candidates", type=int, default=3)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--drawers", type=str, default="top_drawer", help="comma list: top_drawer[,middle_drawer]")
parser.add_argument("--initial_drawer_open", type=float, default=0.0)
parser.add_argument("--reset_index_base", type=int, default=0)
parser.add_argument("--run_id", type=str, default=None)
parser.add_argument("--output_dir", type=str, default="projects/deployment_calibration/data")
parser.add_argument("--max_steps", type=int, default=1800)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import json
from pathlib import Path

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from runtime.base_skill import set_speed_scale
from runtime.ik_joint_adapter import IKJointAdapter
from runtime.scene_state_provider import SceneStateProvider
from runtime.simple_scene_layout import SimpleSceneLayoutManager
from runtime.target_registry import TargetRegistry
from runtime.episode_logger import EpisodeLogger, git_commit
from state_machine.skill_executor import JointBackendConfig, SkillExecutor

from contracts.episode_schema import Episode, CONTRACT_VERSION
from data_generation.sampler import build_plan
from adapters.isaac_open_drawer import reset_full, run_open_drawer_episode
from runtime.drawer_target_config import DRAWER_TARGETS

TASK_ID = "Isaac-Stack-Cube-Franka-JointPolicy-v0"


def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "isaaclab.sh").exists():
            return p
    return start.parents[4]


REPO_ROOT = _find_repo_root(Path(__file__).resolve())


def main():
    torch.manual_seed(args_cli.seed)
    set_speed_scale(1.0)
    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env_cfg.seed = args_cli.seed
    if getattr(env_cfg, "events", None) is not None and hasattr(env_cfg.events, "randomize_cube_positions"):
        env_cfg.events.randomize_cube_positions = None
    if hasattr(env_cfg.scene, "cabinet") and hasattr(env_cfg.scene.cabinet, "actuators"):
        if "drawers" in env_cfg.scene.cabinet.actuators:
            env_cfg.scene.cabinet.actuators["drawers"].stiffness = 0.0
            env_cfg.scene.cabinet.actuators["drawers"].damping = 2.0
    env_cfg.episode_length_s = 1.0e9
    if getattr(env_cfg, "terminations", None) is not None:
        for _t in ("cube_1_dropping", "cube_2_dropping", "cube_3_dropping", "success", "cubes_stacked"):
            if hasattr(env_cfg.terminations, _t):
                setattr(env_cfg.terminations, _t, None)

    env = gym.make(TASK_ID, cfg=env_cfg)
    env.reset(seed=args_cli.seed)
    provider = SceneStateProvider(env)
    layout_manager = SimpleSceneLayoutManager(env=env, base_seed=args_cli.seed)
    registry = TargetRegistry(env.unwrapped.device)
    adapter = IKJointAdapter(env)
    backend = JointBackendConfig(mode="joint", grasp_backend="joint_ik", place_backend="joint_ik",
                                 drawer_backend="ik_pull", adapter=adapter, drawer_env=env,
                                 arm_joint_ids=provider._arm_joint_ids)
    executor = SkillExecutor(registry, backend=backend, log_path="logs/skill_tests/depcal_executor.jsonl")

    control_dt = env_cfg.sim.dt * env_cfg.decimation
    drawers = tuple(d.strip() for d in args_cli.drawers.split(",") if d.strip())
    plan = build_plan(args_cli.seed, args_cli.n_conditions, args_cli.candidates,
                      drawers=drawers, reset_index_base=args_cli.reset_index_base)
    run_id = args_cli.run_id or f"open_drawer_{time.strftime('%Y%m%d_%H%M%S')}"
    el = EpisodeLogger(args_cli.output_dir, run_id)
    controller = {"type": "joint_position_with_dls_ik", "control_dt": control_dt,
                  "control_frequency_hz": round(1.0 / control_dt, 2), "ik_method": "dls",
                  "ik_max_joint_step": 0.20, "physics_dt": env_cfg.sim.dt, "decimation": env_cfg.decimation,
                  "task_id": TASK_ID, "full_reset_per_episode": True}
    el.write_metadata({"run_id": run_id, "skill": "open_drawer", "contract": CONTRACT_VERSION,
                       "seed": args_cli.seed, "n_conditions": args_cli.n_conditions,
                       "candidates": args_cli.candidates, "drawers": list(drawers),
                       "episodes_planned": len(plan)},
                      {"theta_ranges": "see contracts/episode_schema.py THETA_RANGES"},
                      controller, run_command=" ".join(_sys.argv), repo_root=REPO_ROOT)
    done = el.completed_episode_ids()
    session_id = f"sess_{args_cli.seed}_{run_id}"
    print(f"[depcal] run_id={run_id} planned={len(plan)} resume_skip={len(done)} commit={git_commit(REPO_ROOT)[:12]}", flush=True)

    sim_time = 0.0
    n_succ = 0
    for i, spec in enumerate(plan):
        episode_id = f"open_drawer_{i:06d}"
        if episode_id in done:
            continue
        t0 = time.time()
        drawer = spec["drawer"]
        joint_name = DRAWER_TARGETS[drawer]["joint_name"]
        try:
            reset_full(env, provider, layout_manager, spec["reset_index"], joint_name, args_cli.initial_drawer_open)
            x, y, prov, tl, sim_time = run_open_drawer_episode(
                env, provider, executor, adapter, control_dt, drawer, spec["g"], spec["theta"],
                sim_time, max_steps=args_cli.max_steps)
            tl.save(el.trajectory_abspath(episode_id))
            ep = Episode(
                episode_id=episode_id, contract_version=CONTRACT_VERSION, skill="open_drawer",
                seed=args_cli.seed, session_id=session_id, reset_index=spec["reset_index"],
                order_in_session=i, x=x, g=spec["g"], theta=spec["theta"], y=y,
                requested_parameters=prov["requested_parameters"],
                effective_parameter_traces=prov["effective_parameter_traces"],
                controller=controller, trajectory_file=el.trajectory_relpath(episode_id), full_reset=True)
            rec = ep.to_dict()
            rec["candidate_group"] = spec["candidate_group"]
            rec["candidate_k"] = spec["candidate_k"]
        except Exception as exc:
            import traceback
            traceback.print_exc()
            rec = {"episode_id": episode_id, "skill": "open_drawer", "success": False,
                   "failure_reason": "EPISODE_EXCEPTION", "error": str(exc),
                   "candidate_group": spec["candidate_group"], "reset_index": spec["reset_index"]}
        el.log_episode(rec)
        n_succ += int(bool(rec.get("success") or (rec.get("y", {}) or {}).get("success")))
        succ = rec.get("success", (rec.get("y", {}) or {}).get("success"))
        print(f"[depcal] {episode_id} drawer={drawer} g={spec['g']['target_open_position']} "
              f"succ={succ} reason={(rec.get('y',{}) or {}).get('failure_reason', rec.get('failure_reason'))} "
              f"wall={time.time()-t0:.1f}s ({i+1}/{len(plan)})", flush=True)

    print(f"[depcal] DONE {run_id} -> {el.run_dir}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
