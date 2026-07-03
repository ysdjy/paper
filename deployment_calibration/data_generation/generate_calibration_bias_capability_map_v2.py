"""Exploratory calibration-bias capability map (v2). NOT a confirmatory paper run.

v2 fixes the v1 candidate-group nuisance CONFOUND: v1 drew nuisance per-episode, so within a candidate
group the 7 candidates (and 2 probes) had DIFFERENT initial conditions -> candidates differed in more than
theta. v2 uses BLOCK-LEVEL (session/replicate) nuisance:

  * One nuisance context per replicate block r = {robot arm-joint perturbation vector, target jitter}.
  * ALL 9 episodes in a session (2 probes + 7 candidates) reuse that SAME context -> within a candidate
    group only theta (grasp offset) differs.
  * The SAME replicate r reuses the SAME context across ALL 5 bias levels -> a matched block: within a
    matched_group (drawer__r{r}) the x/g conditions are identical across bias; only the hidden bias differs.
  * 3 blocks use 3 independent seeds drawn from a MASTER RNG (not bias-ordered).
  * The 15 sessions are executed in a randomized order; canonical order, execution order, block_id and
    seeds are saved in metadata.

Bias is injected into the perceived handle only (handle_calibration_bias_v1); true geometry / frozen
registry / shared adapter untouched. git provenance captured BEFORE output -> dirty=false.

    ./isaaclab.sh -p .../generate_calibration_bias_capability_map_v2.py --headless \
        --config .../configs/calibration_bias_capability_map_v2.yaml
    # stratified safety smoke: --config .../configs/calibration_bias_stratified_smoke_v2.yaml
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
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def _load_yaml(p):
    import yaml
    return yaml.safe_load(Path(p).read_text())


def _verify_reset(inv) -> bool:
    try:
        return (abs(float(inv["drawer_joint_pos"])) <= 1e-3
                and abs(float(inv["drawer_joint_vel"])) <= 5e-2
                and float(inv["robot_joint_vel_absmax"]) <= 5e-2)
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
    subset = cfg.get("candidate_offset_ids")            # optional (smoke): restrict candidate offsets
    if subset:
        cand = [c for c in cand if c["candidate_id"] in subset]
    probes = grid.get("probe_offsets", []) if cfg.get("run_probes", True) else []
    reps = int(cfg["sessions_per_bias"])
    robust = cfg["robust_theta"]
    damping = float(cfg.get("damping", 3.0))
    base_target = float(cfg.get("target_open_position", 0.20))
    max_steps = int(cfg.get("max_steps", 1800))
    nz = cfg.get("nuisance", {})
    rj_sigma = float(nz.get("robot_joint_sigma", 0.015))
    rj_cap = float(nz.get("robot_joint_cap", 0.03))
    tgt_jit = float(nz.get("target_jitter", 0.005))
    master_seed = int(cfg.get("master_seed", 20260703))

    # ---- BLOCK-LEVEL nuisance: one context per replicate block, seeds from an independent master RNG ----
    master = np.random.default_rng(master_seed)
    block_seeds = [int(master.integers(1, 2**31 - 1)) for _ in range(reps)]
    nuisance_contexts = {}
    for r in range(reps):
        brng = np.random.default_rng(block_seeds[r])
        delta = np.clip(brng.normal(0.0, rj_sigma, size=7), -rj_cap, rj_cap)   # 7 panda arm joints
        jitter = float(brng.uniform(-tgt_jit, tgt_jit))
        nuisance_contexts[r] = {"block_id": r, "block_seed": block_seeds[r],
                                "robot_joint_delta": [float(v) for v in delta.tolist()],
                                "target_jitter": jitter}

    # ---- sessions: 5 bias x reps (canonical), then randomize execution order ----
    canonical = []
    sidx = 0
    for b in biases:
        for r in range(reps):
            canonical.append({"session_id": f"sess_{sidx:03d}", "drawer": drawer, "bias_y": b,
                              "hidden_state_id": bias_level_id(b), "replicate_id": r, "block_id": r,
                              "block_seed": block_seeds[r]})
            sidx += 1
    exec_order = list(range(len(canonical)))
    master.shuffle(exec_order)                              # randomized execution order (independent of bias)
    sessions_exec = [canonical[i] for i in exec_order]

    H = launch_drawer_scene(app_launcher, device=args.device)
    spec = H.spec(drawer)
    provider = H.provider
    robot = provider.scene["robot"]
    arm_ids = list(provider._arm_joint_ids)

    run_id = args.run_id or f"{cfg.get('run_prefix','calibration_bias_capability_map')}_v2_{time.strftime('%Y%m%d_%H%M%S')}"
    outdir = run_id_dir(run_id)
    ep_f = open(outdir / "episodes.jsonl", "w")

    def apply_block_nuisance(ctx):
        """Set robot to default + block delta (reproducible, base = default not current), settle."""
        delta = ctx["robot_joint_delta"]
        q = robot.data.default_joint_pos.clone()
        for j, jid in enumerate(arm_ids):
            q[:, jid] = q[:, jid] + float(delta[j])
        robot.write_joint_state_to_sim(q, torch.zeros_like(robot.data.joint_vel))
        for _ in range(4):
            st = provider.get_state()
            H.env.step(provider.make_hold_joint_action(st, 1.0))

    damping_verify, reset_verify = [], []
    t0 = time.time(); n = 0
    # NB: bias_injection_version / bias_axis are provided by secret_provenance() (sec); do NOT repeat them
    # here or the record dict() gets duplicate kwargs.
    common0 = dict(offset_grid_version=grid_version, offset_grid_sha256=grid_sha,
                   nuisance_level="replicate_block", runtime_design_commit=runtime_design_commit)
    total = len(sessions_exec) * (len(probes) + len(cand))
    for exec_pos, sess in enumerate(sessions_exec):
        b = sess["bias_y"]; r = sess["replicate_id"]
        bspec = biased_spec(spec, b)
        ctx = nuisance_contexts[r]
        sec = secret_provenance(b)
        matched_group_id = f"{drawer}__r{r}"           # spans bias; NO bias in the id
        g = {"target_open_position": base_target + ctx["target_jitter"], "target_tolerance": 0.02}
        nz_prov = dict(block_id=r, block_seed=ctx["block_seed"],
                       nuisance_robot_joint_delta=ctx["robot_joint_delta"],
                       nuisance_target_jitter=ctx["target_jitter"], execution_position=exec_pos)
        order = 0
        episodes = [("probe", pi, p) for pi, p in enumerate(probes)] + \
                   [("candidate", ci, c) for ci, c in enumerate(cand)]
        for role, idx, item in episodes:
            inv = H.reset(bspec); reset_verify.append(_verify_reset(inv))
            apply_block_nuisance(ctx)                   # SAME context for every episode in this session
            setinfo = H.set_damping(bspec, damping); damping_verify.append(abs(setinfo["requested"] - setinfo["effective"]))
            off = float(item["grasp_offset_local_y"])
            th = {"grasp_offset_local_y": off, "max_pos_step": float(robust["max_pos_step"]),
                  "pull_lead": float(robust["pull_lead"])}
            x, y, prov, tl, _ = H.run(bspec, dict(g), th, max_steps=max_steps)
            post = H.read_damping(bspec)
            extra = dict(candidate_grasp_offset_y=off, matched_group_id=matched_group_id, replicate_id=r,
                         damping_eff=setinfo["effective"], damping_post=post, **nz_prov, **common0, **sec)
            if role == "probe":
                extra.update(probe_index=idx, probe_id=item["probe_id"])
            else:
                extra.update(candidate_group=sess["session_id"], candidate_index=idx,
                             candidate_id=item["candidate_id"], history_cutoff=len(probes))
            rec = _mk(sess, spec, role, th, dict(g), x, y, order, gi, reset_verify[-1], extra)
            ep_f.write(json.dumps(rec) + "\n"); ep_f.flush(); order += 1; n += 1
        print(f"[calib-map-v2] exec#{exec_pos:02d} {sess['session_id']} bias={b:+.3f} r={r} done "
              f"({n}/{total} eps, {(time.time()-t0)/60:.1f} min)", flush=True)
    ep_f.close()

    dv = np.array(damping_verify)
    meta = {"run_id": run_id, "contract_version": "open_drawer_v2", "scene_version": "paper_scene_v2",
            "experiment": "calibration_bias_capability_map_v2", "config": cfg,
            "offset_grid_version": grid_version, "offset_grid_sha256": grid_sha, "offset_grid": grid,
            "bias_levels_y": biases, "sessions_per_bias": reps, "n_probes": len(probes),
            "candidate_offsets": [c["grasp_offset_local_y"] for c in cand],
            "candidate_ids": [c["candidate_id"] for c in cand],
            "nuisance_level": "replicate_block", "master_seed": master_seed,
            "nuisance_contexts": nuisance_contexts,
            "nuisance_distribution": {"robot_joint_sigma": rj_sigma, "robot_joint_cap": rj_cap,
                                      "robot_joints": "panda_joint.* (7)", "target_jitter_uniform": tgt_jit,
                                      "scheme": "one draw per replicate block; shared by all 9 eps in a "
                                                "session and by the same replicate across all bias levels",
                                      "block_seeds_from": "master RNG (independent of bias)"},
            "canonical_session_order": [s["session_id"] for s in canonical],
            "execution_order_session_ids": [s["session_id"] for s in sessions_exec],
            "execution_order_indices": exec_order,
            "sessions": canonical, "n_sessions": len(canonical), "n_episodes": n, "damping": damping,
            "damping_set_maxerr": float(dv.max()) if len(dv) else 0.0,
            "damping_verified": bool(len(dv) and dv.max() < 1e-3),
            "full_reset_all_verified": bool(all(reset_verify)) if reset_verify else False,
            "n_reset_verify_fail": int(sum(1 for v in reset_verify if not v)),
            "runtime_design_commit": runtime_design_commit,
            "run_command": " ".join(sys.argv), "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "wall_clock_minutes": round((time.time() - t0) / 60.0, 2),
            "env": {"python": sys.version.split()[0], "platform": platform.platform(),
                    "torch": _torch_version()}, **gi}
    json.dump(meta, open(outdir / "metadata.json", "w"), indent=1)
    print(f"[calib-map-v2] DONE -> {outdir}  n_eps={n} damping_verified={meta['damping_verified']} "
          f"full_reset_all_verified={meta['full_reset_all_verified']} dirty_worktree={gi['dirty_worktree']}", flush=True)
    H.close()
    return 0


def _torch_version():
    try:
        import torch
        return torch.__version__
    except Exception:
        return ""


def _mk(sess, spec, role, theta, g, x, y, order, gi, frv, extra):
    rec = {"episode_id": f"{sess['session_id']}_{role}_{order:03d}", "session_id": sess["session_id"],
           "mechanism_id": spec.mechanism_id, "drawer_name": sess["drawer"], "episode_role": role,
           "order_in_session": order, "x": x, "g": g, "theta": theta, "y": y,
           "contract_version": "open_drawer_v2", "full_reset_verified": bool(frv), **gi}
    rec.update(extra)
    return rec


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    sys.exit(rc)
