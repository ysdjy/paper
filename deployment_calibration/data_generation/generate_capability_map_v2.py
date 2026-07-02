"""Stage-1 capability map: Sobol-sample theta x targets over the stable drawers, full reset each.

Finds where open_drawer succeeds/fails as a function of (grasp_offset_local_y, max_pos_step, pull_lead)
so we can freeze a formal candidate theta space with a 20-80% success band. Baseline damping only.

    ./isaaclab.sh -p projects/paper/deployment_calibration/data_generation/generate_capability_map_v2.py \
        --headless [--config .../drawer_capability_map_v2.yaml] [--n 50]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PAPER / "deployment_calibration"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=str(_PAPER / "deployment_calibration" / "configs" / "drawer_capability_map_v2.yaml"))
ap.add_argument("--n", type=int, default=None, help="override n_params_per_drawer")
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
        import numpy as np
        rng = np.random.default_rng(seed)
        return rng.random((n, d))


def main() -> int:
    import numpy as np
    from _drawer_harness_v2 import launch_drawer_scene, run_id_dir, git_info

    cfg = _load_yaml(args.config)
    tr = cfg["theta_ranges"]
    keys = ["grasp_offset_local_y", "max_pos_step", "pull_lead"]
    lo = np.array([tr[k][0] for k in keys]); hi = np.array([tr[k][1] for k in keys])
    targets = cfg["targets"]
    n = int(args.n or cfg["n_params_per_drawer"])
    reps = int(cfg.get("reps", 1))
    seed = int(cfg.get("seed", 7))

    H = launch_drawer_scene(app_launcher, device=args.device)
    run_id = args.run_id or f"capability_map_v2_{time.strftime('%Y%m%d_%H%M%S')}"
    outdir = run_id_dir(run_id)
    ep_f = open(outdir / "episodes.jsonl", "w")
    gi = git_info()

    per_drawer = {}
    for dn in cfg["drawers"]:
        spec = H.spec(dn)
        u = _sobol(n, 3, seed)
        thetas = lo + u * (hi - lo)
        rows = []
        for i in range(n):
            th = {k: float(thetas[i, j]) for j, k in enumerate(keys)}
            g = {"target_open_position": float(targets[i % len(targets)]), "target_tolerance": 0.02}
            for rep in range(reps):
                H.reset(spec)
                x, y, prov, tl, _ = H.run(spec, g, th, max_steps=cfg.get("max_steps", 1800))
                eid = f"{dn}_{i:03d}_{rep}"
                rec = {"episode_id": eid, "drawer_name": dn, "mechanism_id": spec.mechanism_id,
                       "theta": th, "g": g, "y": y, "contract_version": "open_drawer_v2",
                       "point_index": i, "rep": rep, **gi}
                ep_f.write(json.dumps(rec) + "\n"); ep_f.flush()
                rows.append(rec)
            if (i + 1) % 10 == 0:
                sr = np.mean([r["y"]["success"] for r in rows])
                print(f"[capmap] {dn} {i+1}/{n} running success_rate={sr:.2f}", flush=True)
        sr = float(np.mean([r["y"]["success"] for r in rows]))
        finals = [r["y"]["final_joint_position"] for r in rows]
        per_drawer[dn] = {"n": len(rows), "success_rate": round(sr, 3),
                          "final_min": round(float(np.nanmin(finals)), 4),
                          "final_max": round(float(np.nanmax(finals)), 4),
                          "n_fail_reasons": {}}
        for r in rows:
            fr = r["y"]["failure_reason"]
            per_drawer[dn]["n_fail_reasons"][fr] = per_drawer[dn]["n_fail_reasons"].get(fr, 0) + 1
        print(f"[capmap] === {dn}: success_rate={sr:.2f} finals=[{per_drawer[dn]['final_min']},"
              f"{per_drawer[dn]['final_max']}] reasons={per_drawer[dn]['n_fail_reasons']}", flush=True)
    ep_f.close()
    meta = {"run_id": run_id, "contract_version": "open_drawer_v2", "scene_version": "paper_scene_v2",
            "config": cfg, "n_params_per_drawer": n, "reps": reps, "per_drawer": per_drawer,
            "run_command": " ".join(sys.argv), "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"), **gi}
    json.dump(meta, open(outdir / "metadata.json", "w"), indent=1)
    print(f"[capmap] DONE -> {outdir}", flush=True)
    H.close()
    return 0


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    sys.exit(rc)
