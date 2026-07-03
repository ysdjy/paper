"""PURE instrumentation cores for the band-edge experiment (v1) — NO Isaac imports (unit-testable).

The Isaac-coupled wrappers (InstrumentedIKAdapter, CollisionMonitor) live in
`band_edge_instrumentation_runtime_v1.py` and reuse these cores. Field definitions are frozen by
`offline_v2/calibration_bias/band_edge_instrumentation.py`.
"""

from __future__ import annotations


# 5.1 joint-limit margin
def joint_margins(q, lower, upper):
    """min over joints of min(q-lower, upper-q); returns (min_margin_rad, min_normalized, joint_index)."""
    best_rad, best_norm, best_j = float("inf"), float("inf"), -1
    for j in range(len(q)):
        m = min(float(q[j]) - float(lower[j]), float(upper[j]) - float(q[j]))
        rng = float(upper[j]) - float(lower[j])
        norm = m / rng if rng > 0 else 0.0
        if m < best_rad:
            best_rad, best_norm, best_j = m, norm, j
    return best_rad, best_norm, best_j


class JointMarginTracker:
    def __init__(self):
        self.min_rad = float("inf"); self.min_norm = float("inf")
        self.joint_index = -1; self.phase = "NONE"; self.negative_seen = False

    def update(self, q, lower, upper, phase):
        r, n, j = joint_margins(q, lower, upper)
        if r < 0:
            self.negative_seen = True
        if r < self.min_rad:
            self.min_rad, self.min_norm, self.joint_index, self.phase = r, n, j, phase

    def result(self):
        # The frozen schema field range is [0, joint_range] and validate_record enforces >= 0. A joint can
        # momentarily sit a hair past a SOFT limit (~1e-5 rad), giving a tiny negative raw margin. We report
        # the frozen field clamped to its [0, joint_range] range, and preserve the true value + the event via
        # `minimum_joint_limit_margin_rad_signed` (audit) and `joint_limit_margin_negative_flag`. No info lost.
        return {
            "minimum_joint_limit_margin_rad": None if self.min_rad == float("inf") else round(max(0.0, self.min_rad), 6),
            "minimum_joint_limit_margin_rad_signed": None if self.min_rad == float("inf") else round(self.min_rad, 6),
            "minimum_joint_limit_margin_normalized": None if self.min_norm == float("inf") else round(max(0.0, min(1.0, self.min_norm)), 6),
            "minimum_joint_limit_joint_index": int(self.joint_index),
            "minimum_joint_limit_phase": self.phase,
            "joint_limit_margin_negative_flag": bool(self.negative_seen),
        }


# 5.2 IK failure
class IKFailureCounter:
    def __init__(self):
        self.solve = 0; self.fail = 0; self.cur_run = 0; self.max_run = 0
        self.phase_counts = {}

    def update(self, success: bool, phase: str):
        self.solve += 1
        if not success:
            self.fail += 1; self.cur_run += 1
            self.max_run = max(self.max_run, self.cur_run)
            self.phase_counts[phase] = self.phase_counts.get(phase, 0) + 1
        else:
            self.cur_run = 0

    def result(self):
        return {"ik_solve_count": int(self.solve), "ik_failure_count": int(self.fail),
                "max_consecutive_ik_failures": int(self.max_run),
                "ik_failure_phase_counts": dict(self.phase_counts)}


# 5.3 two DISTINCT clamps (never merged)
def detect_clamps(q_curr, q_des, lower, upper, max_step, eps=1e-9):
    """step_clamp: applied |delta| saturates at max_step (raw step exceeded the cap).
    limit_clamp: q_des sits on a soft joint limit (q_des exceeded the limit after step clamp)."""
    step_clamped = any(abs(float(q_des[j]) - float(q_curr[j])) >= max_step - eps for j in range(len(q_des)))
    limit_clamped = any(float(q_des[j]) <= float(lower[j]) + eps or float(q_des[j]) >= float(upper[j]) - eps
                        for j in range(len(q_des)))
    return bool(step_clamped), bool(limit_clamped)


class ClampCounter:
    def __init__(self):
        self.step = 0; self.limit = 0

    def update(self, q_curr, q_des, lower, upper, max_step):
        s, l = detect_clamps(q_curr, q_des, lower, upper, max_step)
        if s:
            self.step += 1
        if l:
            self.limit += 1

    def result(self):
        return {"joint_step_clamp_count": int(self.step), "joint_limit_clamp_count": int(self.limit)}


# 5.4 failure phase
def failure_phase_from_history(history, succeeded: bool):
    if succeeded or not history:
        return {"failure_phase": "NONE", "failure_transition": "NONE"}
    last_to = None
    for rec in history:
        to = rec.get("to")
        if to == "FAILED":
            return {"failure_phase": rec.get("from") or last_to or "NONE",
                    "failure_transition": f"{rec.get('from', last_to or 'NONE')}->FAILED"}
        if to is not None:
            last_to = to
    return {"failure_phase": last_to or "NONE", "failure_transition": f"{last_to or 'NONE'}->FAILED"}


# 5.5/5.6 CLOSE_GRIPPER->PULL snapshot (TRUE handle, null+reason if unreached)
class CloseSnapshot:
    def __init__(self):
        self.captured = False
        self.data = {
            "true_handle_error_at_close_3d": None, "true_handle_error_at_close_local_y": None,
            "tcp_pose_at_close": None, "true_handle_pose_at_close": None,
            "gripper_width_at_close": None, "gripper_command_at_close": None,
            "close_snapshot_available": False, "close_snapshot_reason": "CLOSE_GRIPPER->PULL not reached",
        }

    def capture(self, *, err3d, err_local_y, tcp_pose7, true_handle_pose7, gripper_width, gripper_cmd):
        if self.captured:
            return
        self.captured = True
        self.data = {
            "true_handle_error_at_close_3d": round(float(err3d), 6),
            "true_handle_error_at_close_local_y": round(float(err_local_y), 6),
            "tcp_pose_at_close": [round(float(v), 6) for v in tcp_pose7],
            "true_handle_pose_at_close": [round(float(v), 6) for v in true_handle_pose7],
            "gripper_width_at_close": round(float(gripper_width), 6),
            "gripper_command_at_close": round(float(gripper_cmd), 6),
            "close_snapshot_available": True, "close_snapshot_reason": "captured",
        }

    def result(self):
        return dict(self.data)
