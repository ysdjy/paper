"""Band-edge experiment generator (v1). The FORMAL plan is 18 blocks x 17 offsets = 306 episodes, no
probes. In THIS phase the full 306 run is NOT authorized; only manifest generation and SMOKE subsets run.

Modes:
  --mode manifest_only : write the frozen 306 manifest (NO Isaac).
  --mode smoke         : run a SMOKE subset (blocks/offsets from --config), tag smoke_only=true, validate
                          every record, write episodes + manifest. (Isaac.)
  --mode full          : the 306 run — GATED. Refuses unless --i_have_explicit_user_approval is passed
                          (which this phase never passes). No confirmatory generator, no confirmatory data.

    # manifest only (no Isaac):
    python .../generate_calibration_bias_band_edge_v1.py --mode manifest_only --run_id band_edge_manifest_v1
    # clear-zone smoke (Isaac):
    ./isaaclab.sh -p .../generate_calibration_bias_band_edge_v1.py --headless --mode smoke \
        --config .../configs/calibration_bias_band_edge_clearzone_smoke_v1.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
_DC = _PAPER / "deployment_calibration"
for _p in (_DC, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import band_edge_plan_v1 as PLAN            # noqa: E402
import band_edge_validators_v1 as VAL        # noqa: E402


def _load_yaml(p):
    import yaml
    return yaml.safe_load(Path(p).read_text())


def _pc_from_cfg(cfg, smoke_only):
    bs = cfg.get("blocks_subset")
    os_ = cfg.get("offsets_subset")
    return PLAN.PlanConfig(
        master_seed=int(cfg.get("master_seed", PLAN.PlanConfig.master_seed)),
        target_open_position=float(cfg.get("target_open_position", 0.20)),
        damping=float(cfg.get("damping", 3.0)), drawer=cfg.get("drawer", "middle_drawer"),
        smoke_only=bool(smoke_only),
        blocks_subset=tuple(bs) if bs else None,
        offsets_subset=tuple(os_) if os_ else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["manifest_only", "smoke", "full"], required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--i_have_explicit_user_approval", action="store_true")
    # AppLauncher args are added only when Isaac is needed
    if "--mode" in sys.argv and "manifest_only" in sys.argv:
        args, _ = ap.parse_known_args()
        return _run_manifest_only(args)

    # full run is gated
    pre, _ = ap.parse_known_args()
    if pre.mode == "full" and not pre.i_have_explicit_user_approval:
        print("[band-edge] FULL 306-episode run is NOT AUTHORIZED in this phase. "
              "Refusing. (manifest is generable via --mode manifest_only.)", flush=True)
        return 2

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        rc = _run_isaac(args, app_launcher)
    finally:
        simulation_app.close()
    return rc


def _run_manifest_only(args):
    pc = PLAN.PlanConfig()
    m = PLAN.build_plan(pc)
    vp = VAL.validate_plan(m)
    outdir = _DC / "data" / (args.run_id or f"band_edge_manifest_v1_{time.strftime('%Y%m%d_%H%M%S')}")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "manifest.json").write_text(json.dumps(m, indent=1))
    print(f"[band-edge] manifest_only -> {outdir}/manifest.json  planned={m['planned_episode_count']} "
          f"(18x17=306), no_probes={not m['design']['run_probes']}, validate_plan={vp['ok']} "
          f"config_sha256={m['config_sha256'][:16]}...", flush=True)
    if not vp["ok"]:
        print("PLAN ERRORS:", vp["errors"]); return 1
    return 0


def _run_isaac(args, app_launcher):
    from _drawer_harness_v2 import run_id_dir, git_info
    from band_edge_runtime_v1 import launch_band_edge_scene, run_band_edge_episode

    if args.mode == "full" and not args.i_have_explicit_user_approval:
        print("[band-edge] FULL run not authorized. Refusing.", flush=True); return 2
    cfg = _load_yaml(args.config)
    smoke_only = args.mode == "smoke"
    pc = _pc_from_cfg(cfg, smoke_only)
    manifest = PLAN.build_plan(pc)
    if args.mode == "full":
        # (unreachable without approval flag) — full 306
        if manifest["planned_episode_count"] != 306:
            print("[band-edge] refusing full run: plan != 306"); return 2
    gi = git_info()
    run_id = args.run_id or f"{'band_edge_smoke' if smoke_only else 'band_edge'}_v1_{time.strftime('%Y%m%d_%H%M%S')}"
    outdir = run_id_dir(run_id)
    manifest["run_id"] = run_id
    manifest["git_commit"] = gi["git_commit"]; manifest["dirty_worktree"] = gi["dirty_worktree"]
    manifest["branch"] = gi["branch"]
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1))

    H = launch_band_edge_scene(app_launcher, device=args.device)
    arm_ids = list(H.provider._arm_joint_ids)
    spec = H.spec(pc.drawer)
    blk_by_id = {b["block_id"]: b for b in manifest["blocks"]}
    ep_f = open(outdir / "episodes.jsonl", "w")
    records = []
    t0 = time.time(); n = 0; sensor_ok = None
    for pe in manifest["planned_episodes"]:
        b = blk_by_id[pe["block_id"]]
        g = {"target_open_position": pc.target_open_position + b["nuisance_target_jitter"],
             "target_tolerance": 0.02}
        x, y, instr, deff, dpost = run_band_edge_episode(
            H, spec, actual_bias_y=b["actual_bias_y"], offset_y=pe["grasp_offset_local_y"], g=g,
            robot_joint_delta=b["nuisance_robot_joint_delta"], damping=pc.damping, arm_ids=arm_ids)
        sensor_ok = instr.get("contact_sensor_available")
        rec = {"episode_id": f"{run_id}_b{b['block_id']:02d}_{pe['offset_id']}",
               "episode_role": "candidate", "smoke_only": bool(smoke_only),
               "block_id": b["block_id"], "nuisance_block_id": b["block_id"],
               "offset_id": pe["offset_id"], "candidate_id": pe["offset_id"],
               "x": x, "g": g, "theta": {"grasp_offset_local_y": pe["grasp_offset_local_y"],
                                         "max_pos_step": 0.02, "pull_lead": 0.08},
               "y": y,
               "secret_deployment_state": {"nominal_bias_y": b["nominal_bias_y"],
                                           "residual_bias_y": b["residual_bias_y"],
                                           "actual_bias_y": b["actual_bias_y"]},
               "nuisance_robot_joint_delta": b["nuisance_robot_joint_delta"],
               "nuisance_target_jitter": b["nuisance_target_jitter"], "block_seed": b["block_seed"],
               "residual_seed": b["residual_seed"], "eff_signed": pe["eff_signed"], "abs_eff": pe["abs_eff"],
               "damping_eff": deff, "damping_post": dpost, "config_sha256": manifest["config_sha256"],
               "code_commit": gi["git_commit"], **instr}
        vr = VAL.validate_record_full(rec)
        rec["_validation_ok"] = vr["ok"]
        if not vr["ok"]:
            print(f"[band-edge] RECORD INVALID {rec['episode_id']}: {vr}", flush=True)
        ep_f.write(json.dumps(rec) + "\n"); ep_f.flush(); records.append(rec); n += 1
        print(f"[band-edge] {n}/{len(manifest['planned_episodes'])} b{b['block_id']} {pe['offset_id']} "
              f"abseff={pe['abs_eff']:.3f} succ={y['success']} contact={instr['max_unintended_contact_force_N']:.2f}N "
              f"({(time.time()-t0)/60:.1f}min)", flush=True)
    ep_f.close()

    # smoke-level validators
    mb = VAL.validate_matched_block(records)
    dup = VAL.reject_duplicate_episodes(records)
    npb = VAL.assert_no_probes(records)
    summary = {"run_id": run_id, "mode": args.mode, "smoke_only": smoke_only, "n_episodes": n,
               "contact_sensor_available": sensor_ok, "matched_block_ok": mb["ok"],
               "matched_block_errors": mb["errors"][:8], "duplicate_ok": dup["ok"], "no_probes_ok": npb["ok"],
               "all_records_valid": all(r["_validation_ok"] for r in records),
               "wall_clock_minutes": round((time.time() - t0) / 60, 2), **gi}
    (outdir / "smoke_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[band-edge] DONE {run_id} n={n} sensor_available={sensor_ok} matched_block={mb['ok']} "
          f"dup_ok={dup['ok']} no_probes={npb['ok']} all_valid={summary['all_records_valid']} "
          f"dirty={gi['dirty_worktree']} -> {outdir}", flush=True)
    H.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
