"""Pilot feature builders + leakage guard (EXPLORATORY / PILOT — NOT CONFIRMATORY).

K0 = static, pre-probe visible fields + the candidate offset (the action). K1 = K0 + emulated-probe execution
history (the offset=0.0 episode's `y`). The hidden state (residual/actual/nominal bias, true handle error,
eff_signed/abs_eff, secret_deployment_state, other candidates' outcomes) is NEVER a feature — only a label.
"""

from __future__ import annotations

# Names that must never appear in any K0/K1 feature (substring match on the flattened feature key).
LEAKAGE_DENYLIST = (
    "residual_bias", "actual_bias", "nominal_bias", "eff_signed", "abs_eff", "secret_deployment_state",
    "true_handle_error", "true_handle_pose", "oracle", "candidate_outcome", "future", "test_label",
    "residual_seed", "block_seed",
)

# K1 probe-history features drawn from the probe episode's execution outcome `y` (post-probe, pre-candidate).
PROBE_HISTORY_KEYS = ("task_outcome_error", "phase_goal_error", "handle_relative_error", "final_joint_position",
                      "overshoot", "skill_elapsed_time", "command_tracking_error", "handle_detached")


def _num(v, default=0.0) -> float:
    if isinstance(v, bool):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def static_features(ep: dict) -> dict:
    """K0: visible static state + goal + the candidate offset (action). No probe outcome, no hidden state."""
    x, g, th = ep["x"], ep["g"], ep["theta"]
    off = round(float(th["grasp_offset_local_y"]), 3)
    feats = {
        "offset": off,                                                  # the ACTION being scored (allowed)
        "abs_offset": abs(off),                                         # lets a linear model see the symmetric band
        "offset_sq": off * off,
        "max_pos_step": _num(th.get("max_pos_step")),
        "pull_lead": _num(th.get("pull_lead")),
        "target_open_position": _num(g.get("target_open_position")),
        "target_tolerance": _num(g.get("target_tolerance")),
        "gripper_width": _num(x.get("gripper_width")),
        "initial_mechanism_joint_pos": _num(x.get("initial_mechanism_joint_pos")),
    }
    for i, v in enumerate(x.get("tcp_pos", [])):
        feats[f"tcp_pos_{i}"] = _num(v)
    for i, v in enumerate(x.get("robot_joint_pos", [])[:7]):
        feats[f"joint_{i}"] = _num(v)
    return feats


def probe_history_features(probe_ep: dict) -> dict:
    """K1 extra: the emulated-probe execution outcome (offset=0.0 episode's `y`)."""
    y = probe_ep["y"]
    feats = {f"probe_{k}": _num(y.get(k)) for k in PROBE_HISTORY_KEYS}
    for ph, dur in (y.get("phase_durations") or {}).items():
        feats[f"probe_phasedur_{ph}"] = _num(dur)
    return feats


def leakage_check(feature_names) -> dict:
    """Return {ok, violations}. A feature name containing any denylist token is a leak."""
    violations = []
    for name in feature_names:
        low = str(name).lower()
        for token in LEAKAGE_DENYLIST:
            if token in low:
                violations.append({"feature": name, "matched": token})
    return {"ok": not violations, "violations": violations, "n_features": len(list(feature_names))}
