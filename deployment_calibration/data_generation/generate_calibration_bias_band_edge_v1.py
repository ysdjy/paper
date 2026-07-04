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
    # reliability / test controls
    ap.add_argument("--stop_after_new_episodes", type=int, default=0,
                    help="test-only planned stop after N NEW episodes -> IN_PROGRESS (forbidden for full)")
    ap.add_argument("--inject_invalid_record_at", default=None,
                    help="test-only: corrupt the record at this planned_episode_id to prove fail-fast "
                         "(smoke/test only; forbidden for full)")
    ap.add_argument("--allow_dirty", action="store_true",
                    help="allow a dirty source tree (smoke only; a full run always refuses dirty)")
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


def _build_record(pe, b, run_id, smoke_only, sci_sha, gi, x, y, instr, deff, dpost, config_sha):
    return {"episode_id": f"{run_id}__{pe['planned_episode_id']}",
            "planned_episode_id": pe["planned_episode_id"],
            "episode_role": "candidate", "smoke_only": bool(smoke_only),
            "block_id": b["block_id"], "nuisance_block_id": b["block_id"],
            "offset_id": pe["offset_id"], "candidate_id": pe["offset_id"],
            "x": x, "g": {"target_open_position": pe.get("_g_target"), "target_tolerance": 0.02},
            "theta": {"grasp_offset_local_y": pe["grasp_offset_local_y"], "max_pos_step": 0.02, "pull_lead": 0.08},
            "y": y,
            "secret_deployment_state": {"nominal_bias_y": b["nominal_bias_y"],
                                        "residual_bias_y": b["residual_bias_y"], "actual_bias_y": b["actual_bias_y"]},
            "nuisance_robot_joint_delta": b["nuisance_robot_joint_delta"],
            "nuisance_target_jitter": b["nuisance_target_jitter"], "block_seed": b["block_seed"],
            "residual_seed": b["residual_seed"], "eff_signed": pe["eff_signed"], "abs_eff": pe["abs_eff"],
            "damping_eff": deff, "damping_post": dpost, "config_sha256": config_sha,
            "science_manifest_sha256": sci_sha, "code_commit": gi["git_commit"], **instr}


def _run_isaac(args, app_launcher):
    import traceback
    from _drawer_harness_v2 import run_id_dir, git_info
    from band_edge_runtime_v1 import launch_band_edge_scene, run_band_edge_episode
    import band_edge_reliability_v1 as REL

    is_full = args.mode == "full"
    smoke_only = args.mode == "smoke"
    if is_full and not args.i_have_explicit_user_approval:
        print("[band-edge] FULL run not authorized. Refusing.", flush=True); return 2
    # test-only controls are forbidden on the formal full path
    if is_full and (args.stop_after_new_episodes or args.inject_invalid_record_at or args.allow_dirty):
        print("[band-edge] test-only controls (stop_after/inject_invalid/allow_dirty) forbidden for full run.",
              flush=True); return 2

    cfg = _load_yaml(args.config)
    pc = _pc_from_cfg(cfg, smoke_only)
    manifest = PLAN.build_plan(pc)
    if is_full and manifest["planned_episode_count"] != 306:
        print("[band-edge] refusing full run: plan != 306"); return 2
    sci_sha = REL.science_manifest_sha256(manifest)
    gi = git_info()
    run_id = args.run_id or f"{'band_edge_smoke' if smoke_only else 'band_edge'}_v1_{time.strftime('%Y%m%d_%H%M%S')}"
    outdir = run_id_dir(run_id)
    config_sha = manifest["config_sha256"]

    # ---- dirty-source refusal (full run must start clean; only its own output dir may differ) ----
    if is_full:
        try:
            REL.assert_clean_source_tree_for_full_run(_PAPER, outdir,
                                                      extra_allowed=("deployment_calibration/data/",))
        except RuntimeError as ex:
            REL.write_run_status(outdir, REL.INVALID, reason=str(ex))
            print(f"[band-edge] {ex}", flush=True); return 3
    elif not args.allow_dirty and gi["dirty_worktree"]:
        print("[band-edge] note: dirty worktree in smoke mode (allowed; pass --allow_dirty to silence).", flush=True)

    # ---- manifest-first identity: freeze/verify the run manifest ----
    man_path = outdir / "manifest.json"
    if man_path.exists():
        existing = json.loads(man_path.read_text())
        if REL.science_manifest_sha256(existing) != sci_sha or existing.get("config_sha256") != config_sha:
            REL.write_run_status(outdir, REL.INVALID, reason="existing manifest science/config hash mismatch")
            print("[band-edge] INVALID: existing run manifest does not match the frozen plan.", flush=True)
            return 3
        manifest = existing
    else:
        m = dict(manifest); m["run_id"] = run_id; m["science_manifest_sha256"] = sci_sha
        m["git_commit"] = gi["git_commit"]; m["branch"] = gi["branch"]
        REL.write_json_atomic(man_path, m); manifest = m
        REL.write_json_atomic(outdir / "run_metadata.json",
                              {"run_id": run_id, "mode": args.mode, "smoke_only": smoke_only,
                               "config_sha256": config_sha, "science_manifest_sha256": sci_sha,
                               "source_commit": gi["git_commit"], "master_seed": manifest["master_seed"],
                               "planned_episode_count": manifest["planned_episode_count"],
                               "output_directory": str(outdir), "created_at": REL._ts(), **gi})
        REL.write_run_status(outdir, REL.PLANNED, planned=manifest["planned_episode_count"])

    # ---- resume: load + verify committed records; build completed set (no resample, no re-exec) ----
    try:
        completed, existing_recs = REL.load_and_verify_completed(outdir, manifest, sci_sha)
    except ValueError as ex:
        REL.write_run_status(outdir, REL.INVALID, reason="existing records inconsistent", detail=str(ex)[:2000])
        (outdir / "errors").mkdir(exist_ok=True)
        REL.write_json_atomic(outdir / "errors" / "resume_invalid.json", {"detail": str(ex)})
        print(f"[band-edge] INVALID: existing records inconsistent -> {ex}", flush=True); return 3
    planned = manifest["planned_episodes"]
    remaining = [pe for pe in planned if pe["planned_episode_id"] not in completed]
    REL.append_resume_log(outdir, {"source_commit": gi["git_commit"], "science_manifest_sha256": sci_sha,
                                    "completed": len(completed), "remaining": len(remaining),
                                    "planned": len(planned)})
    print(f"[band-edge] resume: {len(completed)} completed, {len(remaining)} remaining of {len(planned)} "
          f"(mode={args.mode})", flush=True)
    REL.write_run_status(outdir, REL.IN_PROGRESS, completed=len(completed), remaining=len(remaining))

    H = launch_band_edge_scene(app_launcher, device=args.device)
    arm_ids = list(H.provider._arm_joint_ids)
    spec = H.spec(pc.drawer)
    blk_by_id = {b["block_id"]: b for b in manifest["blocks"]}
    seen_ids = set(completed)
    t0 = time.time(); new_done = 0
    stop_after = int(args.stop_after_new_episodes or 0)
    for pe in remaining:
        if stop_after and new_done >= stop_after:
            print(f"[band-edge] test-only planned stop after {new_done} new episodes -> IN_PROGRESS.", flush=True)
            REL.rebuild_episodes_jsonl(outdir, manifest)
            REL.write_run_status(outdir, REL.IN_PROGRESS, completed=len(seen_ids), remaining=len(planned) - len(seen_ids),
                                 stopped_for_test=True)
            H.close(); return 0
        # running guard: membership + no duplicate + count
        if (pe["block_id"], pe["offset_id"]) not in {(p["block_id"], p["offset_id"]) for p in planned} \
                or pe["planned_episode_id"] in seen_ids or len(seen_ids) >= len(planned):
            _fail(outdir, REL, pe, "running guard: unplanned/duplicate/over-count", None, len(seen_ids))
            H.close(); return 4
        b = blk_by_id[pe["block_id"]]
        g_target = pc.target_open_position + b["nuisance_target_jitter"]
        pe = {**pe, "_g_target": g_target}
        try:
            x, y, instr, deff, dpost = run_band_edge_episode(
                H, spec, actual_bias_y=b["actual_bias_y"], offset_y=pe["grasp_offset_local_y"],
                g={"target_open_position": g_target, "target_tolerance": 0.02},
                robot_joint_delta=b["nuisance_robot_joint_delta"], damping=pc.damping, arm_ids=arm_ids)
        except Exception:
            _fail(outdir, REL, pe, "episode exception", traceback.format_exc(), len(seen_ids))
            H.close(); return 4
        rec = _build_record(pe, b, run_id, smoke_only, sci_sha, gi, x, y, instr, deff, dpost, config_sha)
        # test-only invalid injection (smoke/test only)
        if args.inject_invalid_record_at and pe["planned_episode_id"] == args.inject_invalid_record_at:
            if is_full:
                _fail(outdir, REL, pe, "invalid-injection forbidden on full run", None, len(seen_ids)); H.close(); return 4
            rec["minimum_joint_limit_margin_rad"] = -1.0   # deterministic schema violation
        # FAIL-FAST: validate BEFORE persisting; never write an invalid record
        vr = VAL.validate_record_full(rec)
        if not vr["ok"]:
            _fail(outdir, REL, pe, "record failed validate_record_full",
                  json.dumps(vr), len(seen_ids), errors=vr)
            H.close(); return 4
        REL.atomic_write_record(outdir, rec["episode_id"], rec)   # crash-safe commit AFTER validation
        seen_ids.add(pe["planned_episode_id"]); new_done += 1
        print(f"[band-edge] {len(seen_ids)}/{len(planned)} b{b['block_id']} {pe['offset_id']} "
              f"abseff={pe['abs_eff']:.3f} succ={y['success']} contact={instr['max_unintended_contact_force_N']:.2f}N "
              f"({(time.time()-t0)/60:.1f}min)", flush=True)
    H.close()

    # ---- rebuild episodes.jsonl (manifest order) + HARD self-check controls exit ----
    REL.rebuild_episodes_jsonl(outdir, manifest)
    all_recs = list(REL.committed_records(outdir).values())
    sc = REL.hard_self_check(all_recs, manifest)
    REL.write_json_atomic(outdir / "self_check.json",
                          {"ok": sc["ok"], "errors": sc["errors"], "counts": sc["counts"],
                           "contact_sensor_available": all(r.get("contact_sensor_available") for r in all_recs),
                           "wall_clock_minutes": round((time.time() - t0) / 60, 2)})
    if not sc["ok"]:
        REL.write_run_status(outdir, REL.INVALID, self_check_errors=sc["errors"][:10])
        print(f"[band-edge] INVALID self-check: {sc['errors'][:5]}", flush=True); return 5
    REL.write_run_status(outdir, REL.COMPLETE, n_episodes=len(all_recs))
    REL.append_resume_log(outdir, {"final_status": REL.COMPLETE, "completed": len(all_recs)})
    print(f"[band-edge] COMPLETE {run_id} n={len(all_recs)} self_check=OK -> {outdir}", flush=True)
    return 0


def _fail(outdir, REL, pe, reason, tb, completed_count, errors=None):
    (Path(outdir) / "errors").mkdir(exist_ok=True)
    payload = {"planned_episode_id": pe.get("planned_episode_id"), "reason": reason,
               "validation_errors": errors, "traceback": tb, "completed_count": completed_count,
               "time": REL._ts()}
    REL.write_json_atomic(Path(outdir) / "errors" / f"fail_{pe.get('planned_episode_id','x')}.json", payload)
    REL.write_run_status(outdir, REL.FAILED, reason=reason,
                         failed_episode=pe.get("planned_episode_id"), completed=completed_count)
    REL.append_resume_log(outdir, {"final_status": REL.FAILED, "reason": reason,
                                   "failed_episode": pe.get("planned_episode_id")})
    print(f"[band-edge] FAILED at {pe.get('planned_episode_id')}: {reason}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
