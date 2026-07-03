"""Hidden handle calibration-bias injection (v1) -- stage-specific, DEFAULT OFF.

The hidden deployment state for this stage is the CONTROLLER'S handle local-Y calibration bias:

    perceived_handle_local_y = true_handle_local_y + bias_y
    commanded_grasp_local_y  = perceived_handle_local_y + candidate_grasp_offset_y
    => ideal compensation:     candidate_grasp_offset_y ~= -bias_y

Injection is NON-INVASIVE and touches nothing the old experiments use:
  * The bias is added ONLY to a COPY of the resolved MechanismSpec's `handle_local_pos` (the pose the
    controller believes the handle is at). The skill turns that into `override_grasp_local`, i.e. the
    obs_adapter's PERCEIVED handle -- never the real link geometry.
  * The TRUE handle geometry is the articulation link (`body_pos_w[link]`), resolved by name and read
    from sim; it is NOT derived from `handle_local_pos`, so it is unaffected.
  * The frozen scene registry / grasp_poses.json on disk are NOT modified.
  * bias_y is written ONLY to audit-only `secret_deployment_state` / `hidden_state_id`; it structurally
    cannot enter `x` (build_x uses the X_OBSERVABLE_KEYS whitelist) -- `assert_no_bias_in_x` re-checks.

Contract note: this needs no contract change -- `secret_deployment_state` is a free audit-only dict and
`hidden_state_id` is a free string; both are additive. (See docs/calibration_bias_injection_design_v1.md.)

Default OFF: enabled=False or bias_y==0.0 -> `biased_spec` returns the spec unchanged, so any run that
does not opt in is byte-for-byte the old behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

BIAS_INJECTION_VERSION = "handle_calibration_bias_v1"
BIAS_AXIS = "handle_local_y"   # drawer LINK local +Y (same frame as grasp_offset_local_y)
_SECRET_KEY = "handle_bias_local_y"


@dataclass(frozen=True)
class CalibrationBiasConfig:
    enabled: bool = False
    bias_y: float = 0.0

    @property
    def effective_bias_y(self) -> float:
        return float(self.bias_y) if self.enabled else 0.0


def bias_level_id(bias_y: float) -> str:
    """Stable, sign-explicit hidden-state id, e.g. -0.04 -> 'bias_m040', +0.02 -> 'bias_p020'."""
    mm = int(round(abs(float(bias_y)) * 1000))
    sign = "z" if mm == 0 else ("m" if bias_y < 0 else "p")
    return f"bias_{sign}{mm:03d}"


def biased_spec(spec, bias_y: float):
    """Return a COPY of `spec` whose perceived `handle_local_pos` is shifted by `bias_y` along local-Y.
    bias_y == 0 -> returns `spec` unchanged (identity; old behaviour preserved exactly)."""
    b = float(bias_y)
    if b == 0.0:
        return spec
    if spec.handle_local_pos is None:
        raise ValueError("biased_spec: spec.handle_local_pos is None; cannot inject bias")
    p = list(spec.handle_local_pos)
    p[1] = float(p[1]) + b
    return replace(spec, handle_local_pos=p)


def verify_perceived_shift(spec, biased, bias_y: float, tol: float = 1e-9) -> dict:
    """Runtime verify of the PERCEIVED-handle injection at the spec level (pre-sim).
    Confirms: (a) local-Y shifted by exactly bias_y; (b) X/Z unchanged; (c) the ORIGINAL spec's
    handle_local_pos is untouched (true perceived source preserved); (d) quat unchanged."""
    b = float(bias_y)
    tp, pp = spec.handle_local_pos, biased.handle_local_pos
    dy = float(pp[1]) - float(tp[1])
    ok_y = abs(dy - b) <= tol
    ok_xz = abs(float(pp[0]) - float(tp[0])) <= tol and abs(float(pp[2]) - float(tp[2])) <= tol
    ok_quat = list(spec.handle_local_quat or []) == list(biased.handle_local_quat or [])
    # b==0 must be identity (same object / same values)
    ok_identity = (b != 0.0) or (list(pp) == list(tp))
    return {"requested_bias_y": b, "measured_dy": dy, "ok_shift": bool(ok_y),
            "ok_xz_unchanged": bool(ok_xz), "ok_quat_unchanged": bool(ok_quat),
            "ok_zero_is_identity": bool(ok_identity),
            "ok": bool(ok_y and ok_xz and ok_quat and ok_identity)}


def commanded_grasp_local_y(true_handle_local_y: float, bias_y: float, candidate_offset_y: float) -> float:
    """The frozen algebra (link-local Y): commanded = true + bias + offset. Ideal offset = -bias."""
    return float(true_handle_local_y) + float(bias_y) + float(candidate_offset_y)


def ideal_compensation_offset_y(bias_y: float) -> float:
    return -float(bias_y)


def secret_provenance(bias_y: float) -> dict[str, Any]:
    """Audit/oracle-ONLY provenance for an episode. Never merge into x."""
    return {"secret_deployment_state": {_SECRET_KEY: float(bias_y)},
            "hidden_state_id": bias_level_id(bias_y),
            "bias_injection_version": BIAS_INJECTION_VERSION, "bias_axis": BIAS_AXIS}


def assert_no_bias_in_x(x: dict) -> None:
    """Guard: the observable state x must not carry the bias or any privileged key."""
    for k in x.keys():
        kl = str(k).lower()
        if "bias" in kl or "secret" in kl or "hidden" in kl or "damping" in kl or "perceived" in kl:
            raise ValueError(f"x contains privileged key '{k}' -- calibration-bias leakage")
