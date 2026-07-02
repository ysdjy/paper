#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Stage-2 teleop: GELLO q[0:7] -> Franka joint target (with safety). STATUS: ready.

Two modes:
  * default (no Isaac): wire a GELLO source through the joint-teleop safety
    pipeline and print the resulting Franka joint targets. With ``--mock_gello``
    it runs with NO hardware and NO Isaac, so it doubles as the teleop smoke test.
  * ``--drive_isaac``: launch ``Isaac-Stack-Cube-Franka-JointPolicy-v0`` and feed
    the safe joint targets into the live env via
    ``SceneStateProvider.make_joint_action_from_q_des`` for ``--steps`` steps.
    No recording (that is collect_teleop_demos_joint_v1.py).

Run (hardware-free smoke):
    python .../gello_to_isaac_joint_test.py --mock_gello --steps 30

Run (mock -> live Isaac, headless):
    ./isaaclab.sh -p .../gello_to_isaac_joint_test.py --drive_isaac --mock_gello \
        --task Isaac-Stack-Cube-Franka-JointPolicy-v0 --num_envs 1 --steps 60 --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECTS_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECTS_DIR))

from franka_v1_skill_lab.scene import V1_BASE_TASK_ID  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="GELLO -> Franka joint target (stage 2).")
    ap.add_argument("--mock_gello", action="store_true", help="Use a synthetic GELLO source (no hardware).")
    ap.add_argument("--no_hardware", action="store_true", help="Alias for --mock_gello.")
    ap.add_argument("--teleop_device", default=None, choices=[None, "gello", "mock", "keyboard"])
    ap.add_argument("--gello_config", default=None, help="GELLO yaml (real device).")
    ap.add_argument("--steps", type=int, default=30, help="Number of control steps.")
    ap.add_argument("--lowpass_alpha", type=float, default=0.5)
    ap.add_argument("--max_joint_vel", type=float, default=4.0, help="Max arm joint speed (rad/s).")
    ap.add_argument("--max_step_rad", type=float, default=None, help="Override per-step clamp (else from --max_joint_vel).")
    ap.add_argument("--control_hz", type=float, default=50.0, help="Control rate (sets env decimation).")
    ap.add_argument("--arm_stiffness", type=float, default=400.0, help="Franka arm PD stiffness (HIGH_PD=400).")
    ap.add_argument("--arm_damping", type=float, default=80.0, help="Franka arm PD damping.")
    ap.add_argument("--gello_hz", type=float, default=100.0, help="Real GELLO serial read rate (Hz).")
    ap.add_argument("--drive_isaac", action="store_true", help="Drive the live JointPolicy env with safe targets.")
    ap.add_argument("--task", default=V1_BASE_TASK_ID)
    ap.add_argument("--scene_registry", default=None)
    ap.add_argument("--num_envs", type=int, default=1)
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=1)
    return ap


def _make_source(args):
    from franka_v1_skill_lab.teleop_collection.gello.gello_reader import make_gello_source

    use_mock = args.mock_gello or args.no_hardware or args.teleop_device in ("mock", "keyboard")
    device = "mock" if use_mock else (args.teleop_device or "gello")
    return make_gello_source(device, config_path=args.gello_config, hz=getattr(args, "gello_hz", 100.0)), use_mock


def _mapping_only(args) -> int:
    """Original hardware-free mapping smoke (no Isaac)."""
    from franka_v1_skill_lab.teleop_collection.gello.joint_teleop_safety import JointTeleopSafety

    try:
        source, use_mock = _make_source(args)
    except NotImplementedError as exc:
        print(f"[gello->isaac] real source unavailable: {exc}", file=sys.stderr)
        print("[gello->isaac] re-run with --mock_gello for a hardware-free smoke test.", file=sys.stderr)
        return 2

    control_hz_eff = 100.0 / max(1, round(100.0 / max(1.0, args.control_hz)))
    max_step_rad = args.max_step_rad if args.max_step_rad is not None else args.max_joint_vel / control_hz_eff
    safety = JointTeleopSafety(lowpass_alpha=args.lowpass_alpha, max_step_rad=max_step_rad)
    q0, _ = source.read()
    safety.reset(q0)

    print(f"[gello->isaac] source={'mock' if use_mock else 'real'} steps={args.steps} max_step_rad={max_step_rad:.4f}")
    print("step |               raw q[0:7]                |            safe franka target")
    max_jump = 0.0
    prev = None
    for i in range(args.steps):
        q_raw, gripper = source.read()
        q_safe = safety.step(q_raw)
        if prev is not None:
            max_jump = max(max_jump, max(abs(a - b) for a, b in zip(q_safe, prev)))
        prev = q_safe
        if i < 5 or i == args.steps - 1:
            raw_s = " ".join(f"{v:+.2f}" for v in q_raw)
            safe_s = " ".join(f"{v:+.2f}" for v in q_safe)
            print(f"{i:4d} | {raw_s} | {safe_s}  grip={gripper:.2f}")

    source.close()
    if max_jump > max_step_rad + 1e-6:
        print(f"FAIL — per-step jump {max_jump:.4f} exceeds max_step_rad {max_step_rad:.4f}")
        return 1
    print(f"OK — max per-step joint jump {max_jump:.4f} <= max_step_rad {max_step_rad:.4f}")
    return 0


def _drive_isaac(parser, argv) -> int:
    """Launch Isaac and feed safe GELLO targets into the live env."""
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch

    from franka_v1_skill_lab.teleop_collection.gello.gello_to_franka_mapper import GelloToFrankaMapper
    from franka_v1_skill_lab.teleop_collection.gello.joint_teleop_safety import JointTeleopSafety
    from franka_v1_skill_lab.teleop_collection.recording.isaac_teleop_driver import build_teleop_env

    rc = 0
    try:
        source, use_mock = _make_source(args)
    except NotImplementedError as exc:
        print(f"[gello->isaac] real source unavailable: {exc}", file=sys.stderr)
        simulation_app.close()
        return 2

    decimation = max(1, round(100.0 / max(1.0, args.control_hz)))
    control_hz_eff = 100.0 / decimation
    max_step_rad = args.max_step_rad if args.max_step_rad is not None else args.max_joint_vel / control_hz_eff
    print(f"[gello->isaac] control_hz~{control_hz_eff:.0f} (decimation={decimation}) "
          f"max_step_rad={max_step_rad:.4f} arm_PD={args.arm_stiffness}/{args.arm_damping}", flush=True)

    env, env_cfg, _ = build_teleop_env(
        task_id=args.task, num_envs=args.num_envs, device=args.device,
        use_fabric=not getattr(args, "disable_fabric", False), seed=args.seed,
        decimation=decimation, arm_stiffness=args.arm_stiffness, arm_damping=args.arm_damping,
    )
    from runtime.scene_state_provider import SceneStateProvider

    provider = SceneStateProvider(env)
    mapper = GelloToFrankaMapper(safety=JointTeleopSafety(lowpass_alpha=args.lowpass_alpha, max_step_rad=max_step_rad))
    state = provider.get_state()
    mapper.reset(provider.arm_joint_pos(state).detach().cpu().tolist())

    print(f"[gello->isaac] DRIVE source={'mock' if use_mock else 'real'} steps={args.steps}", flush=True)
    i = 0
    while simulation_app.is_running() and i < args.steps:
        with torch.inference_mode():
            q_raw, grip = source.read()
            mapped = mapper.map(q_raw, grip)
            env.step(provider.make_joint_action_from_q_des(mapped["q_des"], mapped["gripper_cmd"]))
            i += 1
    source.close()
    env.close()
    print(f"[gello->isaac] OK — drove {i} steps into {args.task}", flush=True)
    simulation_app.close()
    return rc


def main() -> int:
    parser = build_arg_parser()
    pre_args, _ = parser.parse_known_args()
    if pre_args.drive_isaac:
        return _drive_isaac(parser, sys.argv[1:])
    return _mapping_only(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
