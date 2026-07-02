"""Stage-0 runtime validation: 5x default-param open_drawer per candidate drawer, full reset each.

Builds the paper drawer scene via the SAME harness as the GUI (SceneSession) and drives the
member-aware v2 adapter. Records reset invariants + episode labels, checks consistency, selects the
stable drawers (>=4/5 success + reset stable). Writes a run dir + docs/paper_stage0_drawer_validation_v2.md.

    ./isaaclab.sh -p projects/paper/deployment_calibration/data_generation/generate_stage0_validation_v2.py \
        --headless [--reps 5] [--drawers top_drawer,middle_drawer,sektion_top_drawer,sektion_bottom_drawer]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER / "franka_skill_state_machine"))
sys.path.insert(0, str(_PAPER / "deployment_calibration"))

from franka_v1_skill_lab.scene import V1_BASE_TASK_ID          # noqa: E402
from franka_v1_skill_lab.scene_interface import ResetMode, SceneConfig, SceneMode  # noqa: E402
from isaaclab.app import AppLauncher                            # noqa: E402

CANDIDATES = ["top_drawer", "middle_drawer", "sektion_top_drawer", "sektion_bottom_drawer"]
DEFAULT_THETA = {"grasp_offset_local_y": 0.0, "max_pos_step": 0.020, "pull_lead": 0.08}
DEFAULT_G = {"target_open_position": 0.20, "target_tolerance": 0.02}
THRESH = {"robot_joint": 1e-3, "tcp": 2e-3, "drawer": 1e-4, "handle": 2e-3}

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--drawers", type=str, default=",".join(CANDIDATES))
ap.add_argument("--max_steps", type=int, default=1800)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def _git(*a):
    try:
        return subprocess.check_output(["git", "-C", str(_PAPER), *a], text=True).strip()
    except Exception:
        return ""


def _maxdiff(rows, key):
    import numpy as np
    vals = [r[key] for r in rows if isinstance(r.get(key), list)]
    if len(vals) < 2:
        return 0.0
    a = np.array(vals, dtype=float)
    return float(np.max(np.max(a, axis=0) - np.min(a, axis=0)))


def main() -> int:
    import numpy as np
    import torch  # noqa
    from franka_v1_skill_lab.scene_interface import SceneSession
    from runtime.base_skill import set_speed_scale
    from runtime.ik_joint_adapter import IKJointAdapter
    from runtime.target_registry import TargetRegistry
    from state_machine.skill_executor import JointBackendConfig, SkillExecutor
    from skills.open_drawer_skill import OpenDrawerIKConfig
    from skills.close_drawer_skill import CloseDrawerIKConfig
    from adapters.articulated_drawer_v2 import (
        MechanismSpec, load_mechanism_registry, load_paper_handles,
        reset_full_v2, run_drawer_episode_v2, read_drawer_damping_v2,
    )

    cfg = SceneConfig(
        mode=SceneMode.TEST, task_id=V1_BASE_TASK_ID, device=args.device, headless=True,
        enable_cameras=False, enable_fp=False, free_microwave_door=False, load_latest_scene=True,
        add_microwave_stand=False, replace_microwave_with_fridge=False, lock_knife=True,
        enable_collision_monitor=False, spawn_init_markers=False, refine_handle_collisions=True,
        apply_saved_camera_offsets=False, reset_mode=ResetMode.STATIC, seed=1,
        disable_auto_reset=True, exclude_members=("microwave", "dishwasher"), hidden_members=())
    session = SceneSession.launch(cfg, _app_launcher=app_launcher)
    env, provider = session.env, session.provider
    control_dt = float(getattr(session, "_sim_dt", 0.02)) or 0.02
    set_speed_scale(5.0)
    registry = TargetRegistry(env.unwrapped.device)
    adapter = IKJointAdapter(env)
    backend = JointBackendConfig(
        mode="joint", grasp_backend="joint_ik", place_backend="joint_ik", drawer_backend="ik_pull",
        adapter=adapter, drawer_env=env, arm_joint_ids=provider._arm_joint_ids, drawer_joint_name="joint_0",
        drawer_open_ik_config=OpenDrawerIKConfig(use_turn_to_face=False, start_from_current=True),
        drawer_close_ik_config=CloseDrawerIKConfig(use_turn_to_face=False, start_from_current=True))
    executor = SkillExecutor(registry, log_path="logs/skill_tests/stage0_v2.jsonl", backend=backend)

    reg = load_mechanism_registry()
    handles = load_paper_handles()
    drawers = [d.strip() for d in args.drawers.split(",") if d.strip()]

    run_id = f"stage0_drawer_validation_v2_{time.strftime('%Y%m%d_%H%M%S')}"
    outdir = _PAPER / "deployment_calibration" / "data" / run_id
    (outdir / "trajectories").mkdir(parents=True, exist_ok=True)
    ep_f = open(outdir / "episodes.jsonl", "w")
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--short").strip())

    per_drawer = {}
    sim_time = 0.0
    for dn in drawers:
        if dn not in reg:
            per_drawer[dn] = {"error": "not in mechanism_registry"}
            continue
        spec = MechanismSpec.resolve(dn, reg, handles)
        invs, eps = [], []
        for rep in range(args.reps):
            inv = reset_full_v2(env, provider, spec, initial_open=0.0, executor=executor)
            inv["damping_effective"] = read_drawer_damping_v2(env, spec)
            invs.append(inv)
            x, y, prov, tl, sim_time = run_drawer_episode_v2(
                env, provider, executor, adapter, control_dt, spec, DEFAULT_G, DEFAULT_THETA,
                sim_time=sim_time, max_steps=args.max_steps)
            eid = f"{dn}_{rep:02d}"
            try:
                tl.save(str(outdir / "trajectories" / f"{eid}.npz"))
            except Exception:
                pass
            rec = {"episode_id": eid, "drawer_name": dn, "mechanism_id": spec.mechanism_id, "rep": rep,
                   "reset_invariants": inv, "x": x, "g": DEFAULT_G, "theta": DEFAULT_THETA, "y": y,
                   "contract_version": "open_drawer_v2", "git_commit": commit, "dirty_worktree": dirty,
                   "full_reset_verified": True}
            ep_f.write(json.dumps(rec) + "\n"); ep_f.flush()
            eps.append(rec)
            print(f"[stage0] {eid}: success={y['success']} final={y['final_joint_position']:.4f} "
                  f"reason={y['failure_reason']} err={y['task_outcome_error']:.4f} damp={inv['damping_effective']:.2f}",
                  flush=True)
        # invariants across reps + labels
        n_succ = sum(1 for e in eps if e["y"]["success"])
        label_ok = all(
            (e["y"]["success"] == (e["y"]["final_joint_position"] >= DEFAULT_G["target_open_position"] - DEFAULT_G["target_tolerance"]))
            for e in eps if e["y"]["final_joint_position"] == e["y"]["final_joint_position"])  # nan-safe
        per_drawer[dn] = {
            "member": spec.member, "reps": args.reps, "n_success": n_succ,
            "success_rate": round(n_succ / max(1, args.reps), 3),
            "reset_robot_joint_maxdiff": _maxdiff(invs, "robot_joint_pos"),
            "reset_tcp_maxdiff": _maxdiff(invs, "tcp_pos"),
            "reset_drawer_maxdiff": float(np.ptp([i["drawer_joint_pos"] for i in invs])) if len(invs) > 1 else 0.0,
            "reset_robot_vel_absmax": max(i["robot_joint_vel_absmax"] for i in invs),
            "reset_drawer_vel_absmax": max(abs(i["drawer_joint_vel"]) for i in invs),
            "label_consistent": bool(label_ok),
            "final_positions": [round(e["y"]["final_joint_position"], 4) for e in eps],
            "elapsed_mean": round(float(np.mean([e["y"]["skill_elapsed_time"] for e in eps])), 2),
        }
        d = per_drawer[dn]
        d["reset_stable"] = (d["reset_robot_joint_maxdiff"] <= THRESH["robot_joint"] and
                             d["reset_tcp_maxdiff"] <= THRESH["tcp"] and
                             d["reset_drawer_maxdiff"] <= THRESH["drawer"])
        d["STABLE_FOR_EXPERIMENT"] = (n_succ >= 4 and d["reset_stable"] and d["label_consistent"])
        print(f"[stage0] === {dn}: succ={n_succ}/{args.reps} reset_stable={d['reset_stable']} "
              f"label_ok={label_ok} STABLE={d['STABLE_FOR_EXPERIMENT']}", flush=True)
    ep_f.close()

    meta = {"run_id": run_id, "contract_version": "open_drawer_v2", "scene_version": "paper_scene_v2",
            "git_commit": commit, "dirty_worktree": dirty, "reps": args.reps,
            "default_theta": DEFAULT_THETA, "default_g": DEFAULT_G, "thresholds": THRESH,
            "drawers": drawers, "per_drawer": per_drawer, "control_dt": control_dt,
            "run_command": " ".join(sys.argv), "captured_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(meta, open(outdir / "metadata.json", "w"), indent=1)
    json.dump({"run_id": run_id, "per_drawer": per_drawer}, open(outdir / "summary.json", "w"), indent=1)
    print(f"[stage0] DONE -> {outdir}", flush=True)
    stable = [d for d, v in per_drawer.items() if v.get("STABLE_FOR_EXPERIMENT")]
    print(f"[stage0] STABLE drawers: {stable}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    sys.exit(rc)
