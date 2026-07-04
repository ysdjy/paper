"""Reliability tests (v1, pure python, no Isaac): hard self-check failures, dirty-source classification,
resume verify (completed keys + inconsistency rejection), and atomic per-episode persistence round-trip.
pytest-discoverable (def test_*)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_DC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_DC))
sys.path.insert(0, str(_DC / "data_generation"))

import band_edge_plan_v1 as PLAN            # noqa: E402
import band_edge_reliability_v1 as REL      # noqa: E402


def _smoke_manifest():
    pc = PLAN.PlanConfig(smoke_only=True, blocks_subset=(0, 1), offsets_subset=(-0.05, 0.0, 0.05))
    m = PLAN.build_plan(pc)
    m["science_manifest_sha256"] = REL.science_manifest_sha256(m)
    return m


def _mk_record(manifest, pe, sci_sha, *, smoke=True):
    b = next(x for x in manifest["blocks"] if x["block_id"] == pe["block_id"])
    return {
        "episode_id": f"run__{pe['planned_episode_id']}", "planned_episode_id": pe["planned_episode_id"],
        "episode_role": "candidate", "smoke_only": smoke, "block_id": b["block_id"],
        "nuisance_block_id": b["block_id"], "offset_id": pe["offset_id"], "candidate_id": pe["offset_id"],
        "x": {"mechanism_id": "cabinet:middle_drawer", "drawer_name": "middle_drawer", "member": "cabinet",
              "robot_joint_pos": [0.0] * 9, "tcp_pos": [2.3, 2, 1.08], "tcp_quat": [0, 1, 0, 0],
              "gripper_width": 0.08, "initial_mechanism_joint_pos": 0.0},
        "g": {"target_open_position": 0.2 + b["nuisance_target_jitter"], "target_tolerance": 0.02},
        "theta": {"grasp_offset_local_y": pe["grasp_offset_local_y"], "max_pos_step": 0.02, "pull_lead": 0.08},
        "y": {"success": pe["abs_eff"] <= 0.02, "failure_reason": "NONE"},
        "secret_deployment_state": {"nominal_bias_y": b["nominal_bias_y"], "residual_bias_y": b["residual_bias_y"],
                                    "actual_bias_y": b["actual_bias_y"]},
        "nuisance_robot_joint_delta": b["nuisance_robot_joint_delta"],
        "nuisance_target_jitter": b["nuisance_target_jitter"],
        "config_sha256": manifest["config_sha256"], "science_manifest_sha256": sci_sha,
        # six instrumentation groups
        "minimum_joint_limit_margin_rad": 0.3, "minimum_joint_limit_margin_normalized": 0.1,
        "minimum_joint_limit_joint_index": 3, "minimum_joint_limit_phase": "APPROACH",
        "ik_solve_count": 100, "ik_failure_count": 0, "max_consecutive_ik_failures": 0,
        "ik_failure_phase_counts": {}, "joint_step_clamp_count": 1, "joint_limit_clamp_count": 0,
        "failure_phase": "NONE", "failure_transition": "NONE",
        "true_handle_error_at_close_3d": 0.01, "true_handle_error_at_close_local_y": -0.003,
        "tcp_pose_at_close": [2.3, 2, 1.08, 0, 1, 0, 0], "true_handle_pose_at_close": [2.31, 2, 1.08, 0, 1, 0, 0],
        "gripper_width_at_close": 0.03, "gripper_command_at_close": 0.0,
        "max_unintended_contact_force_N": 0.0, "unintended_contact_frame_count": 0,
        "first_unintended_contact_phase": "NONE", "contact_force_by_phase": {}, "contact_sensor_available": True,
    }


def test_hard_self_check_clean_and_faults():
    m = _smoke_manifest(); sci = m["science_manifest_sha256"]
    recs = [_mk_record(m, pe, sci) for pe in m["planned_episodes"]]
    assert REL.hard_self_check(recs, m)["ok"], REL.hard_self_check(recs, m)["errors"]
    # duplicate
    assert not REL.hard_self_check(recs + [recs[0]], m)["ok"]
    # missing episode
    assert not REL.hard_self_check(recs[:-1], m)["ok"]
    # unexpected probe
    bad = [dict(r) for r in recs]; bad[0] = {**bad[0], "episode_role": "probe"}
    assert not REL.hard_self_check(bad, m)["ok"]
    # matched-block mismatch (change one residual within a block)
    bad = [dict(r) for r in recs]
    bad[0] = {**bad[0], "secret_deployment_state": {**bad[0]["secret_deployment_state"], "actual_bias_y": 0.777}}
    assert not REL.hard_self_check(bad, m)["ok"]
    # invalid schema (negative count)
    bad = [dict(r) for r in recs]; bad[0] = {**bad[0], "ik_solve_count": -5}
    assert not REL.hard_self_check(bad, m)["ok"]
    # smoke flag inconsistent with run (this run is smoke_only=True; flip one to False)
    bad = [dict(r) for r in recs]; bad[0] = {**bad[0], "smoke_only": False}
    assert not REL.hard_self_check(bad, m)["ok"]


def test_dirty_source_classification():
    allowed = ["deployment_calibration/data/band_edge_run/"]
    # clean
    assert REL._classify_porcelain([], allowed)["clean_for_full"]
    # tracked source modified -> offending
    r = REL._classify_porcelain([" M deployment_calibration/data_generation/generate_calibration_bias_band_edge_v1.py"], allowed)
    assert not r["clean_for_full"] and r["offending"]
    # frozen config modified -> offending
    r = REL._classify_porcelain([" M deployment_calibration/offline_v2/calibration_bias/band_edge_characterization_config_v1.json"], allowed)
    assert not r["clean_for_full"]
    # only designated run output -> allowed (resume preflight passes)
    r = REL._classify_porcelain(["?? deployment_calibration/data/band_edge_run/records/b00_o+0.000.json"], allowed)
    assert r["clean_for_full"] and not r["offending"]


def test_atomic_persist_and_resume_verify():
    m = _smoke_manifest(); sci = m["science_manifest_sha256"]
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        REL.write_json_atomic(out / "manifest.json", m)
        # commit first 3 records atomically
        for pe in m["planned_episodes"][:3]:
            rec = _mk_record(m, pe, sci)
            REL.atomic_write_record(out, rec["episode_id"], rec)
        completed, recs = REL.load_and_verify_completed(out, m, sci)
        assert len(completed) == 3 and len(recs) == 3
        # remaining computed correctly
        remaining = [pe for pe in m["planned_episodes"] if pe["planned_episode_id"] not in completed]
        assert len(remaining) == len(m["planned_episodes"]) - 3
        # rebuild episodes.jsonl in manifest order (only committed)
        n = REL.rebuild_episodes_jsonl(out, m)
        assert n == 3
        # corrupt a committed record (residual mismatch) -> resume must reject (INVALID)
        bad_pe = m["planned_episodes"][0]
        bad = _mk_record(m, bad_pe, sci)
        bad["secret_deployment_state"]["actual_bias_y"] = 0.999
        REL.atomic_write_record(out, bad["episode_id"], bad)
        try:
            REL.load_and_verify_completed(out, m, sci)
            assert False, "expected ValueError for inconsistent record"
        except ValueError:
            pass


def test_science_manifest_sha_stable_ignores_provenance():
    m = _smoke_manifest()
    a = REL.science_manifest_sha256(m)
    m2 = dict(m); m2["git_commit"] = "deadbeef"; m2["dirty_worktree"] = True; m2["run_id"] = "x"
    assert REL.science_manifest_sha256(m2) == a   # volatile provenance excluded
    m3 = dict(m); m3["master_seed"] = m["master_seed"] + 1
    assert REL.science_manifest_sha256(m3) != a   # science change -> different hash
