"""Tests for the band-edge plan/manifest + validators (v1, pure python, no Isaac).

Covers (from the required-tests list): residual matches frozen distribution; one residual per block; 17
offsets share residual; block independence; matched nuisance; randomized order reproducible under master
seed; resume does not resample; duplicate rejection; plan == 306 & no probes; secret not in x/g/theta/H;
observable nuisance keys rejected; smoke excluded from formal analysis.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_DC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_DC))
sys.path.insert(0, str(_DC / "data_generation"))

import band_edge_plan_v1 as PLAN          # noqa: E402
import band_edge_validators_v1 as VAL      # noqa: E402
from offline_v2.calibration_bias import band_edge as BE            # noqa: E402
from offline_v2.calibration_bias import residual_nuisance as RN    # noqa: E402


def _mk_record(block_id, offset, actual_bias, *, smoke=False, robot_q=None, jitter=0.001):
    """A schema-complete synthetic band-edge record (documents the exact runtime output shape)."""
    rq = robot_q or [0.1, -1.0, 0.02, -2.8, 0.02, 1.76, 0.89, 0.04, 0.04]
    return {
        "episode_id": f"be_b{block_id:02d}_{PLAN.offset_id(offset)}", "episode_role": "candidate",
        "smoke_only": smoke, "block_id": block_id, "nuisance_block_id": block_id,
        "offset_id": PLAN.offset_id(offset), "candidate_id": PLAN.offset_id(offset),
        "x": {"mechanism_id": "cabinet:middle_drawer", "drawer_name": "middle_drawer", "member": "cabinet",
              "robot_joint_pos": rq, "tcp_pos": [2.3, 2.0, 1.08], "tcp_quat": [0, 1, 0, 0],
              "gripper_width": 0.08, "initial_mechanism_joint_pos": 0.0},
        "g": {"target_open_position": 0.20 + jitter, "target_tolerance": 0.02},
        "theta": {"grasp_offset_local_y": offset, "max_pos_step": 0.02, "pull_lead": 0.08},
        "y": {"success": abs(actual_bias + offset) <= 0.02, "failure_reason": "NONE",
              "final_joint_position": 0.19, "handle_detached": False},
        "secret_deployment_state": {"nominal_bias_y": 0.0, "residual_bias_y": round(actual_bias, 6),
                                    "actual_bias_y": round(actual_bias, 6)},
        "nuisance_robot_joint_delta": [0.001] * 7, "nuisance_target_jitter": jitter,
        # six instrumentation groups
        "minimum_joint_limit_margin_rad": 0.35, "minimum_joint_limit_margin_normalized": 0.12,
        "minimum_joint_limit_joint_index": 3, "minimum_joint_limit_phase": "APPROACH",
        "ik_solve_count": 120, "ik_failure_count": 0, "max_consecutive_ik_failures": 0,
        "ik_failure_phase_counts": {}, "joint_step_clamp_count": 2, "joint_limit_clamp_count": 0,
        "failure_phase": "NONE", "failure_transition": "NONE",
        "true_handle_error_at_close_3d": 0.012, "true_handle_error_at_close_local_y": -0.004,
        "tcp_pose_at_close": [2.3, 2.0, 1.08, 0, 1, 0, 0], "true_handle_pose_at_close": [2.31, 2.0, 1.08, 0, 1, 0, 0],
        "gripper_width_at_close": 0.03, "gripper_command_at_close": 0.0,
        "max_unintended_contact_force_N": 0.0, "unintended_contact_frame_count": 0,
        "first_unintended_contact_phase": "NONE", "contact_force_by_phase": {}, "contact_sensor_available": True,
    }


def check_plan(fails):
    pc = PLAN.PlanConfig()
    m = PLAN.build_plan(pc)
    # exactly 306, 18x17, no probes
    if m["planned_episode_count"] != 306:
        fails.append(f"plan not 306 ({m['planned_episode_count']})")
    if m["design"]["run_probes"] is not False:
        fails.append("plan has probes")
    if list(m["design"]["offsets"]) != list(BE.OFFSET_GRID):
        fails.append("plan offsets != frozen grid")
    vp = VAL.validate_plan(m)
    if not vp["ok"]:
        fails.append(f"validate_plan: {vp['errors']}")
    # residual matches frozen distribution: every block residual within [-0.01, 0.01], from B's sampler
    res = [b["residual_bias_y"] for b in m["blocks"]]
    if not all(-0.01 <= r <= 0.01 for r in res):
        fails.append("residual outside frozen support")
    ref = RN.draw_block_residuals(18, m["residual_seed"], BE.DEFAULT_BAND_EDGE.residual)
    if [round(r, 9) for r in res] != [round(r, 9) for r in ref]:
        fails.append("residual not from B's draw_block_residuals (second algorithm?)")
    # one residual per block; 17 offsets share it
    for b in m["blocks"]:
        eps = [e for e in m["planned_episodes"] if e["block_id"] == b["block_id"]]
        if len(eps) != 17:
            fails.append(f"block {b['block_id']} has {len(eps)} offsets != 17")
        for e in eps:
            if abs((e["eff_signed"] - e["grasp_offset_local_y"]) - b["actual_bias_y"]) > 1e-5:
                fails.append(f"block {b['block_id']} offset uses wrong actual bias"); break
    # block independence: block seeds distinct
    if len({b["block_seed"] for b in m["blocks"]}) != 18:
        fails.append("block seeds not distinct")
    # randomized order reproducible under master seed
    m2 = PLAN.build_plan(pc)
    if m["randomized_block_order"] != m2["randomized_block_order"] or \
       [b["randomized_offset_order"] for b in m["blocks"]] != [b["randomized_offset_order"] for b in m2["blocks"]]:
        fails.append("randomized order not reproducible under master seed")
    # different master seed -> different order (sanity)
    m3 = PLAN.build_plan(PLAN.PlanConfig(master_seed=pc.master_seed + 1))
    if m3["randomized_block_order"] == m["randomized_block_order"]:
        fails.append("different master seed gave identical block order (suspicious)")
    # config sha present
    if m["config_sha256"] != PLAN.config_sha256():
        fails.append("config sha mismatch")


def check_resume(fails):
    pc = PLAN.PlanConfig()
    with tempfile.TemporaryDirectory() as d:
        mp = Path(d) / "manifest.json"
        m1, built1 = PLAN.load_or_build(mp, pc)
        m2, built2 = PLAN.load_or_build(mp, pc)   # resume: must load, not resample
        if not built1 or built2:
            fails.append("resume: second call should NOT rebuild")
        if json.dumps(m1, sort_keys=True) != json.dumps(m2, sort_keys=True):
            fails.append("resume: manifest changed on reload (resampled)")


def check_validators(fails):
    # matched-block: 17 offsets, same block residual/nuisance -> OK
    blk = [_mk_record(0, o, 0.004) for o in BE.OFFSET_GRID]
    r = VAL.validate_matched_block(blk)
    if not r["ok"]:
        fails.append(f"matched-block false-negative: {r['errors']}")
    # perturb one episode's residual -> must fail
    bad = [dict(e) for e in blk]
    bad[3] = dict(bad[3]); bad[3]["secret_deployment_state"] = dict(bad[3]["secret_deployment_state"])
    bad[3]["secret_deployment_state"]["actual_bias_y"] = 0.009
    if VAL.validate_matched_block(bad)["ok"]:
        fails.append("matched-block missed a within-block bias change")

    # schema+leakage full record OK
    rec = _mk_record(0, 0.02, 0.004)
    vr = VAL.validate_record_full(rec)
    if not vr["ok"]:
        fails.append(f"validate_record_full false-negative: {vr}")

    # secret in x -> leakage fail
    leak = _mk_record(0, 0.0, 0.004); leak["x"] = dict(leak["x"]); leak["x"]["actual_bias_y"] = 0.004
    if VAL.validate_leakage(leak)["ok"]:
        fails.append("leakage missed actual_bias_y in x")
    # observable nuisance key in a model channel -> fail
    obs = _mk_record(0, 0.0, 0.004); obs["x"] = dict(obs["x"]); obs["x"]["observed_sensor_noise_level"] = 0.01
    if VAL.validate_leakage(obs)["ok"]:
        fails.append("observable-nuisance gate missed a disabled key in x")

    # no probes
    if not VAL.assert_no_probes(blk)["ok"]:
        fails.append("assert_no_probes false-positive")
    probed = blk + [dict(_mk_record(0, 0.0, 0.004), episode_role="probe")]
    if VAL.assert_no_probes(probed)["ok"]:
        fails.append("assert_no_probes missed a probe")

    # duplicate rejection
    dup = blk + [blk[0]]
    if VAL.reject_duplicate_episodes(dup)["ok"]:
        fails.append("duplicate rejection missed a repeat")

    # smoke excluded from formal analysis
    mixed = [_mk_record(0, 0.0, 0.004, smoke=True), _mk_record(0, 0.02, 0.004, smoke=False)]
    kept = VAL.exclude_smoke(mixed)
    if len(kept) != 1 or kept[0]["smoke_only"]:
        fails.append("exclude_smoke did not drop smoke_only records")


def main() -> int:
    fails = []
    check_plan(fails)
    check_resume(fails)
    check_validators(fails)
    for f in fails:
        print("FAIL:", f)
    print(f"[test_band_edge_plan_v1] {'PASS' if not fails else str(len(fails))+' FAIL'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
