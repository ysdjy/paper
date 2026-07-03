"""Exploratory calibration-bias capability map (v1). NOT a confirmatory paper run.

Hidden state = controller handle local-Y calibration bias (fixed per session). Grid: 5 bias x 7 candidate
grasp offsets x 3 INDEPENDENT-seed sessions (+ K fixed probes/session). Purpose: locate a usable bias
range and offset window, check whether the best offset moves with bias, whether a robust generalist
offset exists, and whether replicates are genuine independent samples (the damping stage's failure).

Replicate independence: every episode gets an independent per-episode NUISANCE draw (small robot initial
joint perturbation recorded in x; target_open_position jitter recorded in g), seeded independently of the
bias level. bias is injected ONLY into the perceived handle (handle_calibration_bias_v1); the true handle
geometry and the frozen registry are untouched. git provenance captured BEFORE output -> dirty=false.

    ./isaaclab.sh -p .../generate_calibration_bias_capability_map_v1.py --headless \
        --config .../configs/calibration_bias_capability_map_v1.yaml [--max_sessions N]  # N for safety smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

_PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PAPER / "deployment_calibration"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--run_id", default=None)
ap.add_argument("--max_sessions", type=int, default=0, help=">0 limits sessions (nuisance safety smoke)")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def _load_yaml(p):
    import yaml
    return yaml.safe_load(Path(p).read_text())


_RESET_THRESH = {"vel": 5e-2}


def _verify_reset(inv) -> bool:
    try:
        return (abs(float(inv["drawer_joint_pos"])) <= 1e-3
                and abs(float(inv["drawer_joint_vel"])) <= _RESET_THRESH["vel"]
                and float(inv["robot_joint_vel_absmax"]) <= _RESET_THRESH["vel"])
    except Exception:
        return False


def main() -> int:
    import torch
    from _drawer_harness_v2 import launch_drawer_scene, run_id_dir, git_info
    from adapters.handle_calibration_bias_v1 import (
        biased_spec, secret_provenance, bias_level_id, BIAS_INJECTION_VERSION, BIAS_AXIS)

    gi = git_info()
    runtime_design_commit = gi["git_commit"]
    cfg = _load_yaml(args.config)
    dc_root = _PAPER / "deployment_calibration"
    grid_path = dc_root / cfg["offset_grid"]
    grid_bytes = grid_path.read_bytes()
    grid_sha = hashlib.sha256(grid_bytes).hexdigest()
    grid = json.loads(grid_bytes)
    grid_version = grid["grid_version"]

    drawer = cfg["drawer"]
    biases = [float(b) for b in cfg.get("bias_levels_y", grid["bias_levels_y"])]
    cand = grid["candidate_offsets"]
    probes = grid.get("probe_offsets", [])
    reps = int(cfg["sessions_per_bias"])
    robust = cfg["robust_theta"]
    damping = float(cfg.get("damping", 3.0))
    base_target = float(cfg.get("target_open_position", 0.20))
    max_steps = int(cfg.get("max_steps", 1800))
    nz = cfg.get("nuisance", {})
    rj_sigma = float(nz.get("robot_joint_sigma", 0.015))
    rj_cap = float(nz.get("robot_joint_cap", 0.03))
    tgt_jit = float(nz.get("target_jitter", 0.005))
    base_seed = int(cfg.get("base_session_seed", 74010))

    # sessions: 5 bias x reps; session_seed derived from a running counter (INDEPENDENT of bias level)
    sessions = []
    sidx = 0
    for b in biases:
        for r in range(reps):
            sessions.append({"session_id": f"sess_{sidx:03d}", "drawer": drawer, "bias_y": b,
                             "hidden_state_id": bias_level_id(b), "replicate_id": r,
                             "session_seed": base_seed + sidx})   # counter-based, not bias-derived
            sidx += 1
    if args.max_sessions and args.max_sessions > 0:
        sessions = sessions[:args.max_sessions]

    H = launch_drawer_scene(app_launcher, device=args.device)
    spec = H.spec(drawer)
    provider = H.provider
    robot = provider.scene["robot"]
    arm_ids = list(provider._arm_joint_ids)
    dev = robot.data.joint_pos.device

    run_id = args.run_id or f"{cfg.get('run_prefix','calibration_bias_capability_map')}_v1_{time.strftime('%Y%m%d_%H%M%S')}"
    outdir = run_id_dir(run_id)
    ep_f = open(outdir / "episodes.jsonl", "w")

    def apply_nuisance(seed):
        """Small robot arm-joint perturbation (recorded in x via build_x) + target jitter. Returns the
        applied deltas for provenance. Seed is independent of the bias level."""
        rng = np.random.default_rng(seed)
        djoint = np.clip(rng.normal(0.0, rj_sigma, size=len(arm_ids)), -rj_cap, rj_cap)
        q = robot.data.joint_pos.clone()
        for j, jid in enumerate(arm_ids):
            q[:, jid] = q[:, jid] + float(djoint[j])
        robot.write_joint_state_to_sim(q, torch.zeros_like(robot.data.joint_vel))
        for _ in range(3):
            st = provider.get_state()
            H.env.step(provider.make_hold_joint_action(st, 1.0))
        tj = float(rng.uniform(-tgt_jit, tgt_jit))
        return djoint.tolist(), tj

    damping_verify, reset_verify = [], []
    t0 = time.time(); n = 0
    common0 = dict(offset_grid_version=grid_version, offset_grid_sha256=grid_sha,
                   bias_injection_version=BIAS_INJECTION_VERSION, bias_axis=BIAS_AXIS,
                   runtime_design_commit=runtime_design_commit)
    total = len(sessions) * (len(probes) + len(cand))
    for sess in sessions:
        b = sess["bias_y"]
        bspec = biased_spec(spec, b)
        r = sess["replicate_id"]
        sec = secret_provenance(b)
        matched_group_id = f"{drawer}__r{r}"     # spans bias levels; NO bias in the id
        order = 0
        # ---- fixed probes (identical offsets across ALL sessions; do NOT read bias) ----
        for pi, p in enumerate(probes):
            inv = H.reset(bspec); reset_verify.append(_verify_reset(inv))
            djoint, tj = apply_nuisance(sess["session_seed"] * 100000 + order)   # perturb AFTER reset
            setinfo = H.set_damping(bspec, damping); damping_verify.append(abs(setinfo["requested"] - setinfo["effective"]))
            g = {"target_open_position": base_target + tj, "target_tolerance": 0.02}
            th = {"grasp_offset_local_y": float(p["grasp_offset_local_y"]),
                  "max_pos_step": float(robust["max_pos_step"]), "pull_lead": float(robust["pull_lead"])}
            x, y, prov, tl, _ = H.run(bspec, g, th, max_steps=max_steps)
            post = H.read_damping(bspec)
            rec = _mk(sess, spec, "probe", th, g, x, y, order, gi, reset_verify[-1], sec,
                      probe_index=pi, probe_id=p["probe_id"], candidate_grasp_offset_y=float(p["grasp_offset_local_y"]),
                      matched_group_id=matched_group_id, replicate_id=r,
                      nuisance_seed=sess["session_seed"] * 100000 + order, nuisance_robot_joint_delta=djoint,
                      nuisance_target_jitter=tj, damping_eff=setinfo["effective"], damping_post=post, **common0)
            ep_f.write(json.dumps(rec) + "\n"); ep_f.flush(); order += 1; n += 1
        # ---- candidate offsets ----
        for ci, c in enumerate(cand):
            inv = H.reset(bspec); reset_verify.append(_verify_reset(inv))
            djoint, tj = apply_nuisance(sess["session_seed"] * 100000 + order)
            setinfo = H.set_damping(bspec, damping); damping_verify.append(abs(setinfo["requested"] - setinfo["effective"]))
            g = {"target_open_position": base_target + tj, "target_tolerance": 0.02}
            th = {"grasp_offset_local_y": float(c["grasp_offset_local_y"]),
                  "max_pos_step": float(robust["max_pos_step"]), "pull_lead": float(robust["pull_lead"])}
            x, y, prov, tl, _ = H.run(bspec, g, th, max_steps=max_steps)
            post = H.read_damping(bspec)
            rec = _mk(sess, spec, "candidate", th, g, x, y, order, gi, reset_verify[-1], sec,
                      candidate_group=sess["session_id"], candidate_index=ci, candidate_id=c["candidate_id"],
                      candidate_grasp_offset_y=float(c["grasp_offset_local_y"]),
                      matched_group_id=matched_group_id, replicate_id=r, history_cutoff=len(probes),
                      nuisance_seed=sess["session_seed"] * 100000 + order, nuisance_robot_joint_delta=djoint,
                      nuisance_target_jitter=tj, damping_eff=setinfo["effective"], damping_post=post, **common0)
            ep_f.write(json.dumps(rec) + "\n"); ep_f.flush(); order += 1; n += 1
        print(f"[calib-map] {sess['session_id']} bias={b:+.3f} r={r} done "
              f"({n}/{total} eps, {(time.time()-t0)/60:.1f} min)", flush=True)
    ep_f.close()

    dv = np.array(damping_verify)
    meta = {"run_id": run_id, "contract_version": "open_drawer_v2", "scene_version": "paper_scene_v2",
            "experiment": "calibration_bias_capability_map_v1", "config": cfg,
            "offset_grid_version": grid_version, "offset_grid_sha256": grid_sha, "offset_grid": grid,
            "bias_levels_y": biases, "sessions_per_bias": reps, "n_probes": len(probes),
            "candidate_offsets": [c["grasp_offset_local_y"] for c in cand],
            "candidate_ids": [c["candidate_id"] for c in cand],
            "nuisance_distribution": {"robot_joint_sigma": rj_sigma, "robot_joint_cap": rj_cap,
                                      "robot_joints": "panda_joint.*", "target_jitter_uniform": tgt_jit,
                                      "per_episode_seed": "session_seed*100000 + order_in_session",
                                      "seed_independent_of_bias": True},
            "n_sessions": len(sessions), "n_episodes": n, "damping": damping,
            "damping_set_maxerr": float(dv.max()) if len(dv) else 0.0,
            "damping_verified": bool(len(dv) and dv.max() < 1e-3),
            "full_reset_all_verified": bool(all(reset_verify)) if reset_verify else False,
            "n_reset_verify_fail": int(sum(1 for v in reset_verify if not v)),
            "sessions": sessions, "runtime_design_commit": runtime_design_commit,
            "run_command": " ".join(sys.argv), "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "wall_clock_minutes": round((time.time() - t0) / 60.0, 2),
            "env": {"python": sys.version.split()[0], "platform": platform.platform(),
                    "torch": _torch_version()}, **gi}
    json.dump(meta, open(outdir / "metadata.json", "w"), indent=1)
    print(f"[calib-map] DONE -> {outdir}  n_eps={n} damping_verified={meta['damping_verified']} "
          f"full_reset_all_verified={meta['full_reset_all_verified']} dirty_worktree={gi['dirty_worktree']}", flush=True)
    H.close()
    return 0


def _torch_version():
    try:
        import torch
        return torch.__version__
    except Exception:
        return ""


def _mk(sess, spec, role, theta, g, x, y, order, gi, frv, sec, **kw):
    rec = {"episode_id": f"{sess['session_id']}_{role}_{order:03d}", "session_id": sess["session_id"],
           "mechanism_id": spec.mechanism_id, "drawer_name": sess["drawer"], "episode_role": role,
           "order_in_session": order, "x": x, "g": g, "theta": theta, "y": y,
           "contract_version": "open_drawer_v2", "full_reset_verified": bool(frv), **sec, **gi}
    rec.update(kw)
    return rec


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    sys.exit(rc)
