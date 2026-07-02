"""Tiny damping micro-experiment: pick 3 hidden-state levels that span low/mid/high without degeneracy.

Sweeps a few damping values on the pilot drawers with the fixed nominal probe, verifying damping is set
+ not overridden, and records final position / pull duration / success so we can choose 3 non-degenerate
levels for the pilot. ~ (len(values) x drawers x reps) episodes.

    ./isaaclab.sh -p .../damping_micro_sweep_v2.py --headless \
        [--values 3,20,60,120,200] [--drawers middle_drawer,sektion_top_drawer] [--reps 2]
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
ap.add_argument("--values", default="3,20,60,120,200")
ap.add_argument("--drawers", default="middle_drawer,sektion_top_drawer")
ap.add_argument("--reps", type=int, default=2)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main() -> int:
    from _drawer_harness_v2 import launch_drawer_scene, run_id_dir, git_info
    from probe_library_v2 import PROBE_TARGET, probe_theta

    values = [float(v) for v in args.values.split(",")]
    drawers = [d.strip() for d in args.drawers.split(",")]
    H = launch_drawer_scene(app_launcher, device=args.device)
    run_id = f"damping_micro_sweep_v2_{time.strftime('%Y%m%d_%H%M%S')}"
    outdir = run_id_dir(run_id)
    rows = []
    for dn in drawers:
        spec = H.spec(dn)
        for D in values:
            for r in range(args.reps):
                H.reset(spec)
                si = H.set_damping(spec, D)
                th = probe_theta(0)
                x, y, prov, tl, _ = H.run(spec, dict(PROBE_TARGET), th)
                post = H.read_damping(spec)
                rows.append({"drawer": dn, "damping_req": D, "damping_eff": si["effective"],
                             "damping_post": post, "success": y["success"],
                             "final": round(y["final_joint_position"], 4),
                             "pull_dur": round(y.get("phase_durations", {}).get("PULL", 0.0), 3),
                             "reason": y["failure_reason"]})
                print(f"[micro] {dn} D={D} eff={si['effective']:.2f} post={post:.2f} "
                      f"succ={y['success']} final={y['final_joint_position']:.4f} reason={y['failure_reason']}",
                      flush=True)
    json.dump({"run_id": run_id, "rows": rows, **git_info()}, open(outdir / "sweep.json", "w"), indent=1)

    # summarize per (drawer, D)
    print("\n[micro] === summary (drawer, D): success_rate mean_final mean_pull ===", flush=True)
    agg = {}
    for row in rows:
        k = (row["drawer"], row["damping_req"])
        agg.setdefault(k, []).append(row)
    for k in sorted(agg):
        rr = agg[k]
        sr = np.mean([x["success"] for x in rr]); mf = np.mean([x["final"] for x in rr]); mp = np.mean([x["pull_dur"] for x in rr])
        print(f"[micro] {k[0]:20s} D={k[1]:6.0f}  succ={sr:.2f} final={mf:.4f} pull={mp:.2f}", flush=True)
    verified = all(abs(r["damping_req"] - r["damping_eff"]) < 1e-3 and abs(r["damping_req"] - r["damping_post"]) < 1e-3 for r in rows)
    print(f"[micro] damping_set_and_persist_verified = {verified}", flush=True)
    print(f"[micro] DONE -> {outdir}", flush=True)
    H.close()
    return 0


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    sys.exit(rc)
