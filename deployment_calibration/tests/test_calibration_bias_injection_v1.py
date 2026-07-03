"""Tests for the handle calibration-bias injection (v1).

Part A (pure python, always runs): the adapter algebra -- bias=0 identity, sign/magnitude, additive
compensation formula, bias-not-in-x guard, spec-copy leaves the original perceived source untouched,
level-id encoding.

Part B (runs only if a bias_injection_smoke run exists): asserts the Isaac end-to-end verify --
perceived handle shifts by exactly bias, TRUE link geometry is unchanged across bias, world grasp shift
== |bias+offset|, bias=0 reproduces success, mis-compensated grasp degrades, x carries no bias key.

Run: python .../deployment_calibration/tests/test_calibration_bias_injection_v1.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PAPER / "deployment_calibration"))
DATA = _PAPER / "deployment_calibration" / "data"

from adapters.handle_calibration_bias_v1 import (  # noqa: E402
    CalibrationBiasConfig, biased_spec, verify_perceived_shift, commanded_grasp_local_y,
    ideal_compensation_offset_y, secret_provenance, assert_no_bias_in_x, bias_level_id)


@dataclass
class _Spec:
    """Minimal stand-in for MechanismSpec (dataclass with the fields the adapter touches)."""
    drawer_name: str = "middle_drawer"
    member: str = "cabinet"
    joint_name: str = "joint_2"
    link_name: str = "drawer_link"
    mechanism_id: str = "cabinet:middle_drawer"
    handle_local_pos: list = None
    handle_local_quat: list = None


def _mk_spec():
    return _Spec(handle_local_pos=[0.0246, 0.0247, 0.657], handle_local_quat=[1.0, 0.0, 0.0, 0.0])


def check_algebra(fails):
    spec = _mk_spec()
    truey = spec.handle_local_pos[1]

    # 1) bias=0 is identity (same object, unchanged pose)
    if biased_spec(spec, 0.0) is not spec:
        fails.append("bias=0 is not identity (should return same spec object)")
    if not CalibrationBiasConfig().enabled is False or CalibrationBiasConfig().effective_bias_y != 0.0:
        fails.append("default CalibrationBiasConfig not OFF")
    if CalibrationBiasConfig(enabled=False, bias_y=0.04).effective_bias_y != 0.0:
        fails.append("disabled config leaks bias_y")

    # 2) +/- bias shifts perceived local-Y by exactly bias; x/z/quat unchanged; original untouched
    for b in (-0.04, -0.02, 0.02, 0.04):
        bs = biased_spec(spec, b)
        v = verify_perceived_shift(spec, bs, b)
        if not v["ok"]:
            fails.append(f"verify_perceived_shift failed for bias {b}: {v}")
        if abs((bs.handle_local_pos[1] - truey) - b) > 1e-12:
            fails.append(f"perceived Y shift != bias for {b}")
        if bs.handle_local_pos[0] != spec.handle_local_pos[0] or bs.handle_local_pos[2] != spec.handle_local_pos[2]:
            fails.append(f"bias {b} changed X or Z")
        # original spec untouched (true perceived source preserved)
        if spec.handle_local_pos[1] != truey:
            fails.append("biased_spec mutated the original spec.handle_local_pos")

    # 3) additive compensation formula and ideal offset
    for b in (-0.04, 0.0, 0.03):
        for o in (-0.06, 0.0, 0.05):
            got = commanded_grasp_local_y(truey, b, o)
            if abs(got - (truey + b + o)) > 1e-12:
                fails.append(f"commanded_grasp_local_y wrong for b={b} o={o}")
        if abs(ideal_compensation_offset_y(b) - (-b)) > 1e-12:
            fails.append(f"ideal_compensation_offset_y wrong for {b}")
        # ideal offset makes commanded == true
        if abs(commanded_grasp_local_y(truey, b, ideal_compensation_offset_y(b)) - truey) > 1e-12:
            fails.append(f"ideal offset does not recover true handle for bias {b}")

    # 4) bias must not be reachable from x; secret provenance carries it
    for k in ("mechanism_id", "drawer_name", "robot_joint_pos", "tcp_pos", "gripper_width", "member"):
        pass
    try:
        assert_no_bias_in_x({"mechanism_id": "x", "robot_joint_pos": [0], "handle_bias_local_y": 0.04})
        fails.append("assert_no_bias_in_x did not catch a bias key")
    except ValueError:
        pass
    assert_no_bias_in_x({"mechanism_id": "cabinet:middle_drawer", "drawer_name": "middle_drawer",
                         "robot_joint_pos": [0.1], "tcp_pos": [1, 2, 3], "gripper_width": 0.08,
                         "member": "cabinet", "initial_mechanism_joint_pos": 0.0})
    sp = secret_provenance(0.04)
    if sp["secret_deployment_state"].get("handle_bias_local_y") != 0.04:
        fails.append("secret_provenance missing bias")
    assert_no_bias_in_x_secret = "bias" in json.dumps(sp).lower()
    if not assert_no_bias_in_x_secret:
        fails.append("secret_provenance should record bias (audit-only)")

    # 5) level-id encoding sign-explicit + unique
    ids = {b: bias_level_id(b) for b in (-0.04, -0.02, 0.0, 0.02, 0.04)}
    if len(set(ids.values())) != len(ids):
        fails.append(f"bias_level_id not unique: {ids}")
    if ids[0.0] != "bias_z000" or ids[-0.04] != "bias_m040" or ids[0.04] != "bias_p040":
        fails.append(f"bias_level_id encoding wrong: {ids}")


def check_smoke(fails):
    runs = sorted(DATA.glob("calibration_bias_injection_smoke_v1_*"))
    if not runs:
        print("no injection smoke run yet; algebra checks only")
        return
    run = runs[-1]
    meta = json.loads((run / "bias_injection_verify.json").read_text())
    recs = meta["records"]
    print(f"checking smoke {run.name}: {len(recs)} cells")
    by = {(r["bias_y"], r["offset_y"]): r for r in recs}
    for r in recs:
        # spec-level shift ok
        if not r["spec_shift_ok"]:
            fails.append(f"smoke: spec shift not ok at bias={r['bias_y']}")
        # true link geometry unchanged across bias (drift ~ 0)
        if r["true_link_drift"] > 2e-3:
            fails.append(f"smoke: TRUE link moved (drift {r['true_link_drift']:.2e}) at bias={r['bias_y']} -- injection touched geometry")
        # perceived handle shifted by exactly bias (vs the bias=0 baseline perceived)
        if abs(r["perceived_local_dy"] - r["bias_y"]) > 1e-6:
            fails.append(f"smoke: perceived dy {r['perceived_local_dy']} != bias {r['bias_y']}")
        # world grasp shift magnitude == |bias+offset| (handle local-Y mapped to world, unit axis)
        if abs(r["grasp_world_shift_vs_base"] - r["expected_local_shift"]) > 3e-3:
            fails.append(f"smoke: world grasp shift {r['grasp_world_shift_vs_base']:.4f} != |b+o| "
                         f"{r['expected_local_shift']:.4f} at bias={r['bias_y']} off={r['offset_y']}")
        # bias not in x
        if any("bias" in k.lower() or "secret" in k.lower() or "hidden" in k.lower() for k in r["x_keys"]):
            fails.append(f"smoke: x carries a privileged key: {r['x_keys']}")
    # bias=0, offset=0 should succeed (old behaviour reproduced)
    z = by.get((0.0, 0.0))
    if z is not None and not z["success"]:
        fails.append("smoke: bias=0 offset=0 failed -- old behaviour not reproduced")
    # physical direction: for a nonzero bias, the ideal compensating offset (-bias) should do at least as
    # well (>= final position) as the uncompensated offset 0.
    for b in {r["bias_y"] for r in recs if r["bias_y"] != 0.0}:
        comp = by.get((b, round(-b, 6))) or by.get((b, -b))
        unc = by.get((b, 0.0))
        if comp and unc and comp["final_joint_position"] + 1e-6 < unc["final_joint_position"]:
            fails.append(f"smoke: compensating offset {-b} worse than 0 at bias {b} -- wrong direction")


def main() -> int:
    fails: list = []
    check_algebra(fails)
    check_smoke(fails)
    for f in fails[:40]:
        print("FAIL:", f)
    ok = not fails
    print(f"[test_calibration_bias_injection_v1] {'PASS' if ok else str(len(fails))+' FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
