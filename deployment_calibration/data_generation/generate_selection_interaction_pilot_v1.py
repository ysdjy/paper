"""Selection-interaction pilot generator (v1).

Tests whether the hidden per-session drawer DAMPING changes not only outcome probability but WHICH
candidate theta is optimal. Uses a FIXED, MATCHED candidate bank (selection_interaction_candidate_bank_v1
.json): the same candidate_id maps to the same theta across every damping level and every repeat session
-- candidate theta is NEVER redrawn from a session seed or the hidden state.

Structure (see selection_interaction_pilot_v1.yaml):
  9 sessions = 3 damping levels x 3 replicates.
  Each session = 3 fixed probes (probe_library_v2) + 3 target groups x 5 matched candidates = 18 episodes.
  Total = 162 episodes. 27 within-session candidate groups; 9 cross-damping matched groups.

Every episode gets an independent FULL reset; damping is (re)set + runtime-VERIFIED before each episode
and re-read AFTER (damping_post). git provenance is captured BEFORE any output is created so a clean
committed tree yields dirty_worktree=false.

    ./isaaclab.sh -p .../generate_selection_interaction_pilot_v1.py --headless \
        --config .../configs/selection_interaction_pilot_v1.yaml
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


# reset-verification thresholds (match test_reset_invariants_v2.py).
_RESET_THRESH = {"drawer": 1e-3, "vel": 5e-2}


def _verify_reset(inv: dict, initial_open: float = 0.0) -> bool:
    try:
        return (abs(float(inv["drawer_joint_pos"]) - float(initial_open)) <= _RESET_THRESH["drawer"]
                and abs(float(inv["drawer_joint_vel"])) <= _RESET_THRESH["vel"]
                and float(inv["robot_joint_vel_absmax"]) <= _RESET_THRESH["vel"])
    except Exception:
        return False


def main() -> int:
    from _drawer_harness_v2 import launch_drawer_scene, run_id_dir, git_info
    from probe_library_v2 import PROBE_TARGET, probe_theta, probe_name, n_probes

    # --- provenance FIRST, before any output dir exists, so a clean tree reads dirty_worktree=false ---
    gi = git_info()
    runtime_design_commit = gi["git_commit"]

    cfg = _load_yaml(args.config)
    dc_root = _PAPER / "deployment_calibration"
    bank_path = dc_root / cfg["candidate_bank"]
    bank_bytes = bank_path.read_bytes()
    bank_sha256 = hashlib.sha256(bank_bytes).hexdigest()
    bank = json.loads(bank_bytes)
    bank_version = bank["candidate_bank_version"]

    drawer = cfg["drawer"]
    if bank["drawer"] != drawer:
        raise SystemExit(f"bank drawer {bank['drawer']} != config drawer {drawer}")
    damping_levels = cfg["damping_levels"]                 # [{"id":..,"value":..}, ...]
    replicates = int(cfg["replicates_per_level"])
    targets_order = cfg["targets_order"]
    K = int(cfg.get("n_probes", n_probes()))
    max_steps = int(cfg.get("max_steps", 1800))

    # per-target candidate lists (in bank file order == candidate_order)
    tkeys = bank["theta_keys"]
    per_target = {}
    for tid in targets_order:
        t = bank["targets"][tid]
        per_target[tid] = {"g": {"target_open_position": float(t["target_open_position"]),
                                 "target_tolerance": float(t["target_tolerance"])},
                           "candidates": t["candidates"]}

    # --- build session list: 3 damping levels x 3 replicates; replicate_id spans damping (matched) ---
    sessions = []
    sidx = 0
    for lvl in damping_levels:
        for r in range(replicates):
            sessions.append({"session_id": f"sess_{sidx:03d}", "drawer": drawer,
                             "hidden_state_id": lvl["id"], "damping": float(lvl["value"]),
                             "damping_level_id": lvl["id"], "replicate_id": r})
            sidx += 1

    H = launch_drawer_scene(app_launcher, device=args.device)
    run_id = args.run_id or f"{cfg.get('run_prefix','selection_interaction_pilot')}_v1_{time.strftime('%Y%m%d_%H%M%S')}"
    outdir = run_id_dir(run_id)
    ep_f = open(outdir / "episodes.jsonl", "w")

    damping_verify = []
    reset_verify = []
    t_start = time.time()
    n_written = 0
    for sess in sessions:
        spec = H.spec(sess["drawer"])
        D = sess["damping"]
        r = sess["replicate_id"]
        order = 0
        common = dict(candidate_bank_version=bank_version, candidate_bank_sha256=bank_sha256,
                      runtime_design_commit=runtime_design_commit)
        # ---- fixed probe episodes ----
        for pi in range(K):
            inv = H.reset(spec)
            frv = _verify_reset(inv); reset_verify.append(frv)
            setinfo = H.set_damping(spec, D)
            damping_verify.append(abs(setinfo["requested"] - setinfo["effective"]))
            th = probe_theta(pi)
            x, y, prov, tl, _ = H.run(spec, dict(PROBE_TARGET), th, max_steps=max_steps)
            post = H.read_damping(spec)
            rec = _mk(sess, spec, "probe", th, dict(PROBE_TARGET), x, y, order, gi, frv,
                      probe_index=pi, probe_name=probe_name(pi),
                      damping_eff=setinfo["effective"], damping_post=post, **common)
            ep_f.write(json.dumps(rec) + "\n"); ep_f.flush(); order += 1; n_written += 1
        # ---- matched candidate groups (one per target) ----
        for tid in targets_order:
            g = per_target[tid]["g"]
            group_id = f"{sess['session_id']}__{tid}"              # within-session group (same hidden state)
            matched_group_id = f"{drawer}__{tid}__r{r}"            # spans damping; NO damping in the id
            for ci, cand in enumerate(per_target[tid]["candidates"]):
                th = {k: float(cand["theta"][k]) for k in tkeys}
                inv = H.reset(spec)
                frv = _verify_reset(inv); reset_verify.append(frv)
                setinfo = H.set_damping(spec, D)
                damping_verify.append(abs(setinfo["requested"] - setinfo["effective"]))
                x, y, prov, tl, _ = H.run(spec, g, th, max_steps=max_steps)
                post = H.read_damping(spec)
                rec = _mk(sess, spec, "candidate", th, g, x, y, order, gi, frv,
                          candidate_group=group_id, candidate_index=ci,
                          candidate_id=cand["candidate_id"], candidate_archetype=cand["archetype"],
                          target_id=tid, matched_group_id=matched_group_id, replicate_id=r,
                          history_cutoff=K, damping_eff=setinfo["effective"], damping_post=post, **common)
                ep_f.write(json.dumps(rec) + "\n"); ep_f.flush(); order += 1; n_written += 1
        el = time.time() - t_start
        print(f"[sel-pilot] {sess['session_id']} D={D} r={r} done "
              f"({n_written}/{len(sessions)*(K+len(targets_order)*5)} eps, {el/60:.1f} min elapsed)", flush=True)
    ep_f.close()

    dv = np.array(damping_verify)
    meta = {
        "run_id": run_id, "contract_version": "open_drawer_v2", "scene_version": "paper_scene_v2",
        "experiment": "selection_interaction_pilot_v1", "config": cfg,
        "candidate_bank_version": bank_version, "candidate_bank_sha256": bank_sha256,
        "candidate_bank_path": str(bank_path.relative_to(_PAPER)), "candidate_bank": bank,
        "targets_and_candidate_ids": {tid: [c["candidate_id"] for c in per_target[tid]["candidates"]]
                                      for tid in targets_order},
        "n_sessions": len(sessions), "n_episodes": n_written, "damping_levels": damping_levels,
        "replicates_per_level": replicates, "n_probes": K, "targets_order": targets_order,
        "damping_set_maxerr": float(dv.max()) if len(dv) else 0.0,
        "damping_verified": bool(len(dv) and dv.max() < 1e-3),
        "full_reset_all_verified": bool(all(reset_verify)) if reset_verify else False,
        "n_reset_verify_fail": int(sum(1 for v in reset_verify if not v)),
        "sessions": sessions, "runtime_design_commit": runtime_design_commit,
        "run_command": " ".join(sys.argv), "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_clock_minutes": round((time.time() - t_start) / 60.0, 2),
        "env": {"python": sys.version.split()[0], "platform": platform.platform(),
                "torch": _torch_version()},
        **gi}
    json.dump(meta, open(outdir / "metadata.json", "w"), indent=1)
    print(f"[sel-pilot] DONE -> {outdir}  n_eps={n_written} "
          f"damping_verified={meta['damping_verified']} full_reset_all_verified={meta['full_reset_all_verified']} "
          f"dirty_worktree={gi['dirty_worktree']}", flush=True)
    H.close()
    return 0


def _torch_version():
    try:
        import torch
        return torch.__version__
    except Exception:
        return ""


def _mk(sess, spec, role, theta, g, x, y, order, gi, full_reset_verified, **kw):
    rec = {"episode_id": f"{sess['session_id']}_{role}_{order:03d}", "session_id": sess["session_id"],
           "mechanism_id": spec.mechanism_id, "drawer_name": sess["drawer"], "episode_role": role,
           "order_in_session": order, "x": x, "g": g, "theta": theta, "y": y,
           "hidden_state_id": sess["hidden_state_id"],
           "secret_deployment_state": {"damping": sess["damping"]},   # AUDIT/ORACLE ONLY
           "contract_version": "open_drawer_v2", "full_reset_verified": bool(full_reset_verified), **gi}
    rec.update(kw)
    return rec


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    sys.exit(rc)
