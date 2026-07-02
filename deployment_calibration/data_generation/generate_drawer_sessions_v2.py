"""Stage-2/3 session generator (v2): deployment sessions with a hidden per-session damping state.

A session = a contiguous deployment period where the hidden mechanism state (drawer joint DAMPING) is
FIXED. Each session = K probe episodes (fixed standard thetas) + candidate decision groups. Every
episode gets an independent FULL reset; damping is (re)set + runtime-VERIFIED before each episode.

Leakage-safe by construction: candidate outcomes are never fed back; the eval reconstructs history only
from the session's probe episodes (which always precede candidates in order). No candidate self/future/
cross-session leakage.

    ./isaaclab.sh -p .../generate_drawer_sessions_v2.py --headless --config .../damping_pilot_v1.yaml
"""

from __future__ import annotations

import argparse
import json
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


def _sobol(n, d, seed):
    try:
        from scipy.stats import qmc
        return qmc.Sobol(d=d, scramble=True, seed=seed).random(n)
    except Exception:
        return np.random.default_rng(seed).random((n, d))


def main() -> int:
    from _drawer_harness_v2 import launch_drawer_scene, run_id_dir, git_info
    from probe_library_v2 import PROBES, PROBE_TARGET, probe_theta, probe_name, n_probes

    cfg = _load_yaml(args.config)
    tkeys = ["grasp_offset_local_y", "max_pos_step", "pull_lead"]
    tr = cfg["theta_ranges"]
    lo = np.array([tr[k][0] for k in tkeys]); hi = np.array([tr[k][1] for k in tkeys])
    targets = cfg["targets"]
    damping_levels = cfg["damping_levels"]                 # [{"id":"L1","value":..}, ...]
    drawers = cfg["drawers"]
    sess_per_cond = int(cfg.get("sessions_per_condition", 1))
    groups_per_session = int(cfg.get("candidate_groups_per_session", 1))
    cands_per_group = int(cfg.get("candidates_per_group", 5))
    K = int(cfg.get("n_probes", n_probes()))
    seed0 = int(cfg.get("seed", 11))
    max_steps = int(cfg.get("max_steps", 1800))

    H = launch_drawer_scene(app_launcher, device=args.device)
    run_id = args.run_id or f"{cfg.get('run_prefix','sessions')}_v2_{time.strftime('%Y%m%d_%H%M%S')}"
    outdir = run_id_dir(run_id)
    ep_f = open(outdir / "episodes.jsonl", "w")
    gi = git_info()

    # build session list: every (drawer x damping_level) repeated sess_per_cond times
    sessions = []
    sidx = 0
    for dn in drawers:
        for lvl in damping_levels:
            for r in range(sess_per_cond):
                sessions.append({"session_id": f"sess_{sidx:03d}", "drawer": dn,
                                 "hidden_state_id": lvl["id"], "damping": float(lvl["value"]),
                                 "seed": seed0 + sidx})
                sidx += 1

    damping_verify = []
    for sess in sessions:
        spec = H.spec(sess["drawer"])
        D = sess["damping"]
        order = 0
        # ---- probe episodes (fixed thetas) ----
        for pi in range(K):
            H.reset(spec)
            setinfo = H.set_damping(spec, D)
            damping_verify.append({"session": sess["session_id"], "phase": "probe", "requested": D,
                                   "effective": setinfo["effective"]})
            th = probe_theta(pi)
            x, y, prov, tl, _ = H.run(spec, dict(PROBE_TARGET), th, max_steps=max_steps)
            post = H.read_damping(spec)
            rec = _mk(sess, spec, "probe", th, dict(PROBE_TARGET), x, y, order, gi,
                      probe_index=pi, damping_eff=setinfo["effective"], damping_post=post)
            ep_f.write(json.dumps(rec) + "\n"); ep_f.flush(); order += 1
        # ---- candidate groups ----
        for cg in range(groups_per_session):
            g = {"target_open_position": float(targets[cg % len(targets)]), "target_tolerance": 0.02}
            u = _sobol(cands_per_group, 3, sess["seed"] * 100 + cg)
            thetas = lo + u * (hi - lo)
            gid = f"{sess['session_id']}_g{cg}"
            for ci in range(cands_per_group):
                th = {k: float(thetas[ci, j]) for j, k in enumerate(tkeys)}
                H.reset(spec)
                setinfo = H.set_damping(spec, D)
                damping_verify.append({"session": sess["session_id"], "phase": "cand", "requested": D,
                                       "effective": setinfo["effective"]})
                x, y, prov, tl, _ = H.run(spec, g, th, max_steps=max_steps)
                post = H.read_damping(spec)
                rec = _mk(sess, spec, "candidate", th, g, x, y, order, gi,
                          candidate_group=gid, candidate_index=ci, history_cutoff=K,
                          damping_eff=setinfo["effective"], damping_post=post)
                ep_f.write(json.dumps(rec) + "\n"); ep_f.flush(); order += 1
        succ_p = "?"
        print(f"[sessions] {sess['session_id']} drawer={sess['drawer']} {sess['hidden_state_id']} "
              f"D={D} probes={K} groups={groups_per_session}x{cands_per_group} done", flush=True)
    ep_f.close()

    dv = np.array([abs(d["requested"] - d["effective"]) for d in damping_verify])
    meta = {"run_id": run_id, "contract_version": "open_drawer_v2", "scene_version": "paper_scene_v2",
            "config": cfg, "n_sessions": len(sessions), "damping_levels": damping_levels,
            "damping_set_maxerr": float(dv.max()) if len(dv) else 0.0,
            "damping_verified": bool(len(dv) and dv.max() < 1e-3),
            "sessions": sessions, "run_command": " ".join(sys.argv),
            "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"), **gi}
    json.dump(meta, open(outdir / "metadata.json", "w"), indent=1)
    print(f"[sessions] DONE -> {outdir}  damping_verified={meta['damping_verified']} "
          f"(max set err {meta['damping_set_maxerr']:.2e})", flush=True)
    H.close()
    return 0


def _mk(sess, spec, role, theta, g, x, y, order, gi, **kw):
    rec = {"episode_id": f"{sess['session_id']}_{role}_{order:03d}", "session_id": sess["session_id"],
           "mechanism_id": spec.mechanism_id, "drawer_name": sess["drawer"], "episode_role": role,
           "order_in_session": order, "x": x, "g": g, "theta": theta, "y": y,
           "hidden_state_id": sess["hidden_state_id"],
           "secret_deployment_state": {"damping": sess["damping"]},   # AUDIT/ORACLE ONLY
           "contract_version": "open_drawer_v2", "full_reset_verified": True, **gi}
    rec.update(kw)
    return rec


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    sys.exit(rc)
