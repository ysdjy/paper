"""Isaac smoke that VERIFIES the handle calibration-bias injection end-to-end (v1).

Small matrix (default bias {-0.04,0,+0.04} x offset {-0.04,0,+0.04} = 9 eps). For each cell it captures,
at t0 (skill just started, before any motion):
  * perceived handle (obs_adapter.handle_offset, link-local) -> must be true + bias in Y,
  * TRUE link world pose (cab.body_pos_w[link]) -> must be identical across bias (unchanged geometry),
  * commanded grasp world pose (skill._grasp_pose(0.0)) -> world shift must equal (bias+offset) mapped
    to world along the handle local-Y axis,
then runs the episode to completion for the physical outcome (mis-compensated grasp should degrade).

Writes bias_injection_verify.json under the run dir. `test_calibration_bias_injection_v1.py` asserts on it.

    ./isaaclab.sh -p .../verify_calibration_bias_injection_v1.py --headless \
        --config .../configs/calibration_bias_injection_smoke_v1.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
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


def main() -> int:
    import torch
    from _drawer_harness_v2 import launch_drawer_scene, run_id_dir, git_info
    import adapters.articulated_drawer_v2 as adv
    from adapters.handle_calibration_bias_v1 import (
        biased_spec, verify_perceived_shift, bias_level_id, BIAS_INJECTION_VERSION)
    from runtime.skill_request import SkillRequest
    from runtime.skill_types import ExecutionStatus, SkillType

    cfg = _load_yaml(args.config)
    drawer = cfg.get("drawer", "middle_drawer")
    biases = [float(b) for b in cfg["bias_levels_y"]]
    offsets = [float(o) for o in cfg["candidate_offsets_y"]]
    robust = cfg.get("robust_theta", {"max_pos_step": 0.02, "pull_lead": 0.08})
    target = float(cfg.get("target_open_position", 0.20))
    max_steps = int(cfg.get("max_steps", 1800))
    damping = float(cfg.get("damping", 3.0))
    gi = git_info()

    H = launch_drawer_scene(app_launcher, device=args.device)
    spec = H.spec(drawer)
    control_dt = H.control_dt
    env, provider, executor = H.env, H.provider, H.executor

    def _grasp_at_start(the_spec, g, theta):
        """Reset, set damping, start skill; capture geometry at t0 WITHOUT stepping; then run to end."""
        H.reset(the_spec)
        setinfo = H.set_damping(the_spec, damping)
        params = adv._theta_to_params(g, theta, the_spec)
        req = SkillRequest(request_id=f"verify_{time.time_ns()}", skill_type=SkillType.OPEN_DRAWER,
                           source_object=None, destination_type="drawer", destination_object=drawer,
                           parameters=params)
        provider.set_sim_time(0.0)
        state = provider.get_state()
        executor.reset(); executor.start(req, state)
        skill = executor.active_skill
        cab = skill.obs_adapter.cabinet
        li = skill.obs_adapter._link_idx
        eid = skill.adapter.env_id
        true_link_w = [float(v) for v in cab.data.body_pos_w[eid, li].tolist()]
        perceived_local = [float(v) for v in skill.obs_adapter.handle_offset.tolist()]
        gp = skill._grasp_pose(0.0)
        grasp_world = [float(v) for v in gp.pos_w.tolist()]
        return dict(setinfo_eff=setinfo["effective"], true_link_w=true_link_w,
                    perceived_local=perceived_local, grasp_world=grasp_world,
                    override=params.get("override_grasp_local"))

    outdir = run_id_dir(args.run_id or f"calibration_bias_injection_smoke_v1_{time.strftime('%Y%m%d_%H%M%S')}")
    records = []
    theta = {"grasp_offset_local_y": 0.0, "max_pos_step": float(robust["max_pos_step"]),
             "pull_lead": float(robust["pull_lead"])}
    g = {"target_open_position": target, "target_tolerance": 0.02}

    # baseline grasp-world at (bias=0, offset=0) for the shift reference
    base = _grasp_at_start(spec, g, {**theta, "grasp_offset_local_y": 0.0})
    base_grasp = np.array(base["grasp_world"])
    base_true_link = np.array(base["true_link_w"])

    for b in biases:
        bspec = biased_spec(spec, b)
        vshift = verify_perceived_shift(spec, bspec, b)
        for o in offsets:
            th = {**theta, "grasp_offset_local_y": float(o)}
            cap = _grasp_at_start(bspec, g, th)
            # now run the SAME cell to completion for the outcome
            x, y, prov, tl, _ = H.run(bspec, g, th, max_steps=max_steps)
            # geometry deltas
            gw = np.array(cap["grasp_world"])
            dshift = float(np.linalg.norm(gw - base_grasp))               # world move vs (0,0) baseline
            expected_local_shift = abs(b + o)                              # |bias+offset| in local-Y
            true_link_drift = float(np.linalg.norm(np.array(cap["true_link_w"]) - base_true_link))
            perceived_dy = float(cap["perceived_local"][1] - base["perceived_local"][1])
            rec = dict(bias_y=b, bias_level_id=bias_level_id(b), offset_y=o,
                       spec_shift_ok=vshift["ok"], spec_measured_dy=vshift["measured_dy"],
                       perceived_local_dy=perceived_dy, true_link_drift=true_link_drift,
                       grasp_world_shift_vs_base=dshift, expected_local_shift=expected_local_shift,
                       true_handle_local=list(spec.handle_local_pos),
                       perceived_handle_local=list(bspec.handle_local_pos),
                       success=bool(y["success"]), final_joint_position=y["final_joint_position"],
                       failure_reason=y["failure_reason"], task_outcome_error=y["task_outcome_error"],
                       handle_relative_error=y["handle_relative_error"], x_keys=sorted(x.keys()))
            records.append(rec)
            print(f"[verify] bias={b:+.3f} off={o:+.3f} perceived_dy={perceived_dy:+.4f} "
                  f"true_link_drift={true_link_drift:.2e} grasp_world_shift={dshift:.4f} "
                  f"(exp |b+o|={expected_local_shift:.3f}) succ={y['success']} "
                  f"final={y['final_joint_position']:.3f}", flush=True)

    meta = {"experiment": "calibration_bias_injection_smoke_v1", "drawer": drawer,
            "bias_levels_y": biases, "candidate_offsets_y": offsets, "damping": damping,
            "robust_theta": robust, "target_open_position": target,
            "bias_injection_version": BIAS_INJECTION_VERSION,
            "run_command": " ".join(sys.argv), "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "records": records, **gi}
    json.dump(meta, open(outdir / "bias_injection_verify.json", "w"), indent=1)
    print(f"[verify] DONE -> {outdir/'bias_injection_verify.json'}  n={len(records)} "
          f"dirty_worktree={gi['dirty_worktree']}", flush=True)
    H.close()
    return 0


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    sys.exit(rc)
