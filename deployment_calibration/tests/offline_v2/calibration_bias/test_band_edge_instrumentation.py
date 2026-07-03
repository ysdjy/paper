"""Frozen instrumentation-schema + collision-plan tests."""

import copy

from deployment_calibration.offline_v2.calibration_bias import band_edge_instrumentation as BI


def _good_record():
    return {
        "x": {"tcp_pos": [0, 0, 0], "gripper_width": 0.08},
        "secret_deployment_state": {"nominal_bias_y": 0.0, "residual_bias_y": 0.004, "actual_bias_y": 0.004},
        "minimum_joint_limit_margin_rad": 0.31, "minimum_joint_limit_margin_normalized": 0.12,
        "minimum_joint_limit_joint_index": 3, "minimum_joint_limit_phase": "APPROACH",
        "ik_solve_count": 120, "ik_failure_count": 0, "max_consecutive_ik_failures": 0,
        "ik_failure_phase_counts": {},
        "joint_step_clamp_count": 4, "joint_limit_clamp_count": 0,
        "failure_phase": "NONE", "failure_transition": "NONE",
        "true_handle_error_at_close_3d": 0.012, "true_handle_error_at_close_local_y": -0.004,
        "tcp_pose_at_close": [0, 0, 0, 0, 1, 0, 0], "true_handle_pose_at_close": [0, 0, 0, 0, 1, 0, 0],
        "gripper_width_at_close": 0.02, "gripper_command_at_close": 0.0,
        "max_unintended_contact_force_N": 0.0, "unintended_contact_frame_count": 0,
        "first_unintended_contact_phase": "NONE", "contact_force_by_phase": {},
        "contact_sensor_available": True,
    }


def test_all_six_groups_defined():
    groups = BI.REQUIRED_FIELD_GROUPS
    assert set(groups) == {"joint_limit_margin", "ik_failure", "clamps", "failure_phase",
                           "handle_error_at_close", "gripper_at_close", "collision"}
    # two DISTINCT clamp counters, never merged
    assert "joint_step_clamp_count" in groups["clamps"]
    assert "joint_limit_clamp_count" in groups["clamps"]


def test_every_field_has_definition_and_unit():
    for name, spec in BI.INSTRUMENTATION_FIELDS.items():
        dtype, unit, rng, definition, legal = spec
        assert dtype in ("float", "int", "str", "dict", "list", "bool")
        assert definition and unit
        assert legal is False        # NO instrumentation field is model-legal


def test_good_record_validates():
    r = BI.validate_record(_good_record())
    assert r["ok"], r["errors"]


def test_reject_residual_in_x():
    rec = _good_record()
    rec["x"]["residual_bias_y"] = 0.004
    r = BI.validate_record(rec)
    assert not r["ok"]
    assert any("privileged" in e for e in r["errors"])


def test_reject_actual_bias_in_x():
    rec = _good_record()
    rec["x"]["actual_bias_y"] = 0.004
    assert not BI.validate_record(rec)["ok"]


def test_reject_missing_instrumentation_field():
    rec = _good_record()
    del rec["joint_limit_clamp_count"]
    r = BI.validate_record(rec)
    assert not r["ok"]
    assert any("joint_limit_clamp_count" in e for e in r["errors"])


def test_reject_missing_secret_bias():
    rec = _good_record()
    rec["secret_deployment_state"] = {"nominal_bias_y": 0.0}   # residual/actual missing
    r = BI.validate_record(rec)
    assert not r["ok"]
    assert any("residual_bias_y" in e or "actual_bias_y" in e for e in r["errors"])


def test_collision_field_present_even_if_unavailable():
    rec = _good_record()
    rec["contact_sensor_available"] = False   # allowed value, field still required to exist
    assert BI.validate_record(rec)["ok"]


def test_collision_sensor_plan_concrete():
    plan = BI.COLLISION_SENSOR_PLAN
    assert "ContactSensor" in plan["mechanism"]
    assert plan["monitored_pairs"] and plan["excluded_pairs"]
    smoke = plan["sensor_validation_smoke"]
    assert "step_1_clean_baseline" in smoke and "step_2_intentional_contact_positive_control" in smoke
    assert "confirmatory" in plan["threshold_selection_rule"].lower()
