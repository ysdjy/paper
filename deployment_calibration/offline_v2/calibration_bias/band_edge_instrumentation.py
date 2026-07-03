"""Frozen instrumentation schema + collision-sensor plan for the band-edge experiment (offline).

Defines the EXACT fields A must record (units, ranges, definitions) so the band-edge data can exclude
each competing failure mechanism (joint-limit / IK-unreachability / collision / empty-grasp / offset-miss
/ PULL-slip). Mirrors Claude C's field-by-field feasibility audit. B does not implement runtime; this is
the frozen contract a runtime-implementation phase must satisfy, checked by a schema/record validator.
"""

from __future__ import annotations

PHASES = ("ARC_TO_FACE", "MOVE_TO_PRE_GRASP", "APPROACH", "CLOSE_GRIPPER", "PULL", "SETTLE", "RELEASE")

# secret / audit-only bias fields (a deployable model reads NONE of these)
SECRET_BIAS_FIELDS = ("nominal_bias_y", "residual_bias_y", "actual_bias_y")

# each spec: name -> (dtype, unit, range, one-line definition, is_model_legal)
INSTRUMENTATION_FIELDS = {
    # 5.1 joint-limit margin
    "minimum_joint_limit_margin_rad": ("float", "rad", "[0, joint_range]",
        "per-episode min over ALL IK steps and 7 arm joints of min(q-q_lower, q_upper-q) vs soft_joint_pos_limits", False),
    "minimum_joint_limit_margin_normalized": ("float", "unitless", "[0, 1]",
        "same margin divided by (q_upper - q_lower) for that joint", False),
    "minimum_joint_limit_joint_index": ("int", "index", "[0, 6]",
        "arm-joint index achieving the episode-min margin", False),
    "minimum_joint_limit_phase": ("str", "phase", "PHASES",
        "skill phase in which the episode-min margin occurred", False),
    # 5.2 IK failure
    "ik_solve_count": ("int", "count", "[0, inf)", "IKJointAdapter.solve() calls in the episode", False),
    "ik_failure_count": ("int", "count", "[0, inf)", "solve() returns with success==False", False),
    "max_consecutive_ik_failures": ("int", "count", "[0, inf)", "longest run of consecutive IK failures", False),
    "ik_failure_phase_counts": ("dict", "count", "phase->int", "IK failures grouped by skill phase", False),
    # 5.3 two DISTINCT clamps (never merged)
    "joint_step_clamp_count": ("int", "count", "[0, inf)",
        "steps where raw IK step exceeded max_joint_step (step clamp)", False),
    "joint_limit_clamp_count": ("int", "count", "[0, inf)",
        "steps where q_des still exceeded soft joint limits AFTER step clamp (limit clamp)", False),
    # 5.4 failure phase
    "failure_phase": ("str", "phase", "PHASES|NONE", "last active skill state before entering FAILED", False),
    "failure_transition": ("str", "transition", "e.g. APPROACH->FAILED", "the terminal state transition", False),
    # 5.5 true handle error at CLOSE_GRIPPER->PULL
    "true_handle_error_at_close_3d": ("float", "m", "[0, inf)",
        "TCP-vs-TRUE-handle 3D error at the CLOSE_GRIPPER->PULL control step (true, not perceived)", False),
    "true_handle_error_at_close_local_y": ("float", "m", "(-inf, inf)",
        "signed local-Y component of the same true error", False),
    "tcp_pose_at_close": ("list", "m,quat", "len 7", "TCP pose (pos3+quat4) at the CLOSE->PULL step", False),
    "true_handle_pose_at_close": ("list", "m,quat", "len 7", "TRUE handle pose (pos3+quat4) at the same step", False),
    # 5.6 gripper at close
    "gripper_width_at_close": ("float", "m", "[0, ~0.08]", "actual gripper width at end of CLOSE_GRIPPER", False),
    "gripper_command_at_close": ("float", "m", "[0, ~0.08]", "commanded gripper width at the same instant", False),
    # 6 collision / contact
    "max_unintended_contact_force_N": ("float", "N", "[0, inf)",
        "max contact force on arm/hand-vs-cabinet EXCLUDING finger-handle grasp contact", False),
    "unintended_contact_frame_count": ("int", "count", "[0, inf)", "steps with unintended contact > 0", False),
    "first_unintended_contact_phase": ("str", "phase", "PHASES|NONE", "phase of first unintended contact", False),
    "contact_force_by_phase": ("dict", "N", "phase->float", "max unintended contact force per phase", False),
    "contact_sensor_available": ("bool", "flag", "{true,false}", "whether the contact sensor was active", False),
}

REQUIRED_FIELD_GROUPS = {
    "joint_limit_margin": ["minimum_joint_limit_margin_rad", "minimum_joint_limit_margin_normalized",
                           "minimum_joint_limit_joint_index", "minimum_joint_limit_phase"],
    "ik_failure": ["ik_solve_count", "ik_failure_count", "max_consecutive_ik_failures",
                   "ik_failure_phase_counts"],
    "clamps": ["joint_step_clamp_count", "joint_limit_clamp_count"],
    "failure_phase": ["failure_phase", "failure_transition"],
    "handle_error_at_close": ["true_handle_error_at_close_3d", "true_handle_error_at_close_local_y",
                              "tcp_pose_at_close", "true_handle_pose_at_close"],
    "gripper_at_close": ["gripper_width_at_close", "gripper_command_at_close"],
    "collision": ["max_unintended_contact_force_N", "unintended_contact_frame_count",
                  "first_unintended_contact_phase", "contact_force_by_phase", "contact_sensor_available"],
}

# ---- collision-sensor plan (frozen mechanism; numeric threshold chosen AFTER band data, never from
#      confirmatory data) ----
COLLISION_SENSOR_PLAN = {
    "mechanism": "Isaac Lab / PhysX ContactSensor (or equivalent contact-report API)",
    "monitored_pairs": [
        "Franka non-gripper arm links vs cabinet bodies",
        "Franka hand body vs cabinet bodies",
    ],
    "excluded_pairs": ["Franka finger vs handle (intended grasp contact)"],
    "raw_quantities": ["max_unintended_contact_force_N", "unintended_contact_frame_count",
                       "first_unintended_contact_phase", "contact_force_by_phase", "contact_sensor_available"],
    "sensor_validation_smoke": {
        "step_1_clean_baseline": "run a known collision-free trajectory -> expect max force ~0",
        "step_2_intentional_contact_positive_control": "drive the arm into the cabinet -> expect large force",
        "step_3_discrimination": "confirm the sensor separates step_1 from step_2 (force gap >> noise)",
        "must_pass_before": "the 306-episode band-edge run",
    },
    "threshold_selection_rule": "the numeric unintended-contact force threshold is chosen from the "
        "band-edge data distribution (e.g. above the clean-baseline noise floor from step_1) AFTER the "
        "band run, and frozen BEFORE confirmatory v4. It must NOT use any confirmatory data.",
}


def field_spec(name: str):
    return INSTRUMENTATION_FIELDS.get(name)


def validate_record(record: dict) -> dict:
    """Validate one (future) band-edge episode record against the frozen schema. Read-only / structural.

    Checks: all six instrumentation groups present with correct dtype/basic range; secret bias fields
    present in the audit block; NO secret/residual/actual bias inside x (model input); collision fields
    present (availability may be true or false but the field must exist)."""
    errors = []

    def dtype_ok(val, dtype):
        return {"float": (int, float), "int": int, "str": str, "dict": dict, "list": (list, tuple),
                "bool": bool}.get(dtype, object)

    for group, fields in REQUIRED_FIELD_GROUPS.items():
        for f in fields:
            if f not in record:
                errors.append(f"missing instrumentation field '{f}' (group {group})")
                continue
            dtype = INSTRUMENTATION_FIELDS[f][0]
            if not isinstance(record[f], dtype_ok(record[f], dtype)) and record[f] is not None:
                errors.append(f"field '{f}' expected {dtype}, got {type(record[f]).__name__}")

    # secret bias fields must live in an audit block, not in x
    x = record.get("x", {})
    for k in x:
        low = str(k).lower()
        if any(s in low for s in ("bias", "residual", "actual_bias", "nominal_bias", "secret")):
            errors.append(f"x carries privileged key '{k}'")
    audit = record.get("secret_deployment_state") or record.get("audit") or {}
    for sf in SECRET_BIAS_FIELDS:
        if sf not in audit and sf not in record:
            errors.append(f"missing secret/audit bias field '{sf}'")

    # basic non-negativity on count/force fields
    for f in ("ik_solve_count", "ik_failure_count", "joint_step_clamp_count", "joint_limit_clamp_count",
              "minimum_joint_limit_margin_rad", "max_unintended_contact_force_N",
              "unintended_contact_frame_count"):
        if isinstance(record.get(f), (int, float)) and record[f] < 0:
            errors.append(f"field '{f}' must be >= 0, got {record[f]}")
    if isinstance(record.get("minimum_joint_limit_margin_normalized"), (int, float)):
        v = record["minimum_joint_limit_margin_normalized"]
        if not (0.0 <= v <= 1.0):
            errors.append(f"normalized margin must be in [0,1], got {v}")

    return {"ok": not errors, "errors": errors}


def schema_summary() -> dict:
    return {"n_fields": len(INSTRUMENTATION_FIELDS), "groups": list(REQUIRED_FIELD_GROUPS),
            "secret_bias_fields": list(SECRET_BIAS_FIELDS),
            "collision_sensor_plan": COLLISION_SENSOR_PLAN}
