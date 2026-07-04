"""Unit tests for the PURE instrumentation cores (v1, no Isaac): joint margin, clamps, IK failure,
failure phase, CLOSE snapshot null/availability; plus a full assembled record passing B's schema."""

from __future__ import annotations

import sys
from pathlib import Path

_DC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_DC))
sys.path.insert(0, str(_DC / "data_generation"))

import band_edge_instrumentation_core_v1 as C   # noqa: E402
import band_edge_validators_v1 as VAL            # noqa: E402


def check_joint_margin(fails):
    lower = [-2.0] * 7; upper = [2.0] * 7
    q = [0.0] * 7; q[4] = 1.9   # joint 4 closest to upper -> margin 0.1
    r, n, j = C.joint_margins(q, lower, upper)
    if not (abs(r - 0.1) < 1e-9 and j == 4 and abs(n - 0.1 / 4.0) < 1e-9):
        fails.append(f"joint_margins wrong: r={r} n={n} j={j}")
    t = C.JointMarginTracker()
    t.update([0.0] * 7, lower, upper, "APPROACH")            # margin 2.0
    t.update(q, lower, upper, "PULL")                        # margin 0.1 at PULL, joint 4
    res = t.result()
    if res["minimum_joint_limit_joint_index"] != 4 or res["minimum_joint_limit_phase"] != "PULL":
        fails.append(f"tracker min not at PULL/j4: {res}")
    if not (0.0 <= res["minimum_joint_limit_margin_normalized"] <= 1.0):
        fails.append("normalized margin out of [0,1]")
    # negative margin: frozen field clamped to >=0, flag set, signed value preserved
    t2 = C.JointMarginTracker(); t2.update([2.5] + [0.0] * 6, lower, upper, "APPROACH")   # q past soft limit
    r2 = t2.result()
    if not r2["joint_limit_margin_negative_flag"]:
        fails.append("negative margin not flagged")
    if r2["minimum_joint_limit_margin_rad"] < 0:
        fails.append("frozen margin field not clamped to >=0")
    if r2["minimum_joint_limit_margin_rad_signed"] >= 0:
        fails.append("signed margin field should preserve the negative value")


def check_clamps(fails):
    lower = [-2.0] * 3; upper = [2.0] * 3; max_step = 0.2
    # step clamp: delta saturated at max_step
    s, l = C.detect_clamps([0.0, 0, 0], [0.2, 0, 0], lower, upper, max_step)
    if not s or l:
        fails.append(f"step-clamp detect wrong: s={s} l={l}")
    # limit clamp: q_des on the upper limit, small delta
    s, l = C.detect_clamps([1.95, 0, 0], [2.0, 0, 0], lower, upper, max_step)
    if s or not l:
        fails.append(f"limit-clamp detect wrong: s={s} l={l}")
    # neither
    s, l = C.detect_clamps([0.0, 0, 0], [0.05, 0, 0], lower, upper, max_step)
    if s or l:
        fails.append(f"no-clamp misdetected: s={s} l={l}")
    cc = C.ClampCounter()
    cc.update([0.0, 0, 0], [0.2, 0, 0], lower, upper, max_step)   # step
    cc.update([1.95, 0, 0], [2.0, 0, 0], lower, upper, max_step)  # limit
    r = cc.result()
    if r["joint_step_clamp_count"] != 1 or r["joint_limit_clamp_count"] != 1:
        fails.append(f"clamp counters wrong / merged: {r}")


def check_ik_failure(fails):
    ik = C.IKFailureCounter()
    for ok, ph in [(True, "APPROACH"), (False, "APPROACH"), (False, "APPROACH"), (True, "PULL"),
                   (False, "PULL")]:
        ik.update(ok, ph)
    r = ik.result()
    if r["ik_solve_count"] != 5 or r["ik_failure_count"] != 3 or r["max_consecutive_ik_failures"] != 2:
        fails.append(f"ik counts wrong: {r}")
    if r["ik_failure_phase_counts"] != {"APPROACH": 2, "PULL": 1}:
        fails.append(f"ik phase counts wrong: {r['ik_failure_phase_counts']}")


def check_failure_phase(fails):
    if C.failure_phase_from_history([], True)["failure_phase"] != "NONE":
        fails.append("success should give NONE failure phase")
    hist = [{"from": "IDLE", "to": "APPROACH", "time": 0}, {"from": "APPROACH", "to": "CLOSE_GRIPPER", "time": 1},
            {"from": "CLOSE_GRIPPER", "to": "FAILED", "time": 2}]
    r = C.failure_phase_from_history(hist, False)
    if r["failure_phase"] != "CLOSE_GRIPPER" or r["failure_transition"] != "CLOSE_GRIPPER->FAILED":
        fails.append(f"failure phase wrong: {r}")


def check_close_snapshot(fails):
    s = C.CloseSnapshot()
    r = s.result()
    if r["close_snapshot_available"] or r["true_handle_error_at_close_3d"] is not None:
        fails.append("uncaptured snapshot should be null + available False")
    if "not reached" not in r["close_snapshot_reason"]:
        fails.append("uncaptured snapshot missing reason")
    s.capture(err3d=0.01, err_local_y=-0.004, tcp_pose7=[0, 0, 0, 1, 0, 0, 0],
              true_handle_pose7=[0.01, 0, 0, 1, 0, 0, 0], gripper_width=0.03, gripper_cmd=0.0)
    r = s.result()
    if not r["close_snapshot_available"] or r["true_handle_error_at_close_3d"] != 0.01:
        fails.append("captured snapshot fields wrong")
    s.capture(err3d=9.9, err_local_y=9.9, tcp_pose7=[9] * 7, true_handle_pose7=[9] * 7,
              gripper_width=9, gripper_cmd=9)   # must be ignored
    if s.result()["true_handle_error_at_close_3d"] != 0.01:
        fails.append("second capture not ignored")


def check_full_record_schema(fails):
    """Assemble the 6 groups (from cores) into a record and pass B's frozen validate_record."""
    jm = C.JointMarginTracker(); jm.update([0.0] * 7, [-2] * 7, [2] * 7, "APPROACH")
    ik = C.IKFailureCounter(); ik.update(True, "APPROACH")
    cc = C.ClampCounter(); cc.update([0.0] * 7, [0.05] * 7, [-2] * 7, [2] * 7, 0.2)
    fp = C.failure_phase_from_history([], True)
    sn = C.CloseSnapshot(); sn.capture(err3d=0.012, err_local_y=-0.003, tcp_pose7=[2.3, 2, 1.08, 0, 1, 0, 0],
                                       true_handle_pose7=[2.31, 2, 1.08, 0, 1, 0, 0], gripper_width=0.03, gripper_cmd=0.0)
    coll = {"max_unintended_contact_force_N": 0.0, "unintended_contact_frame_count": 0,
            "first_unintended_contact_phase": "NONE", "contact_force_by_phase": {}, "contact_sensor_available": True}
    rec = {"episode_id": "be_test", "episode_role": "candidate", "smoke_only": True,
           "block_id": 0, "nuisance_block_id": 0, "offset_id": "o+0.020",
           "x": {"mechanism_id": "cabinet:middle_drawer", "drawer_name": "middle_drawer", "member": "cabinet",
                 "robot_joint_pos": [0.0] * 9, "tcp_pos": [2.3, 2, 1.08], "tcp_quat": [0, 1, 0, 0],
                 "gripper_width": 0.08, "initial_mechanism_joint_pos": 0.0},
           "g": {"target_open_position": 0.2}, "theta": {"grasp_offset_local_y": 0.02},
           "y": {"success": True, "failure_reason": "NONE"},
           "secret_deployment_state": {"nominal_bias_y": 0.0, "residual_bias_y": 0.003, "actual_bias_y": 0.003},
           **jm.result(), **ik.result(), **cc.result(), **fp, **sn.result(), **coll}
    r = VAL.validate_record_full(rec)
    if not r["ok"]:
        fails.append(f"assembled record fails schema/leakage: {r}")


def test_band_edge_instrumentation_cores():   # pytest-discoverable
    fails = []
    for fn in (check_joint_margin, check_clamps, check_ik_failure, check_failure_phase,
               check_close_snapshot, check_full_record_schema):
        fn(fails)
    assert not fails, fails


def main() -> int:
    fails = []
    for fn in (check_joint_margin, check_clamps, check_ik_failure, check_failure_phase,
               check_close_snapshot, check_full_record_schema):
        fn(fails)
    for f in fails:
        print("FAIL:", f)
    print(f"[test_band_edge_instrumentation_core_v1] {'PASS' if not fails else str(len(fails))+' FAIL'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
