# Handle calibration-bias injection — design (v1)

Exploratory stage. The hidden deployment state pivots from the damping stage's scalar joint damping to a
**controller handle-pose calibration bias** — a perception/calibration error in where the controller
*believes* the handle is, along the drawer link-local +Y axis.

## Frozen symbols
```
perceived_handle_local_y = true_handle_local_y + bias_y
commanded_grasp_local_y  = perceived_handle_local_y + candidate_grasp_offset_y
ideal compensation:        candidate_grasp_offset_y ≈ -bias_y
```
- `bias_y` is **fixed within a deployment session** (the hidden state).
- `bias_y` is **audit-only**: it lives in `secret_deployment_state.handle_bias_local_y` and `hidden_state_id`,
  never in `x`, `g`, or `theta`.
- The candidate decision variable is `candidate_grasp_offset_y` (= `theta.grasp_offset_local_y`).

## Why this is a real perception/calibration error (not moving the handle)
The injection adds `bias_y` **only to a copy of the resolved `MechanismSpec.handle_local_pos`** — the pose
the controller turns into `override_grasp_local`, i.e. the obs_adapter's *perceived* handle. In
`open_drawer_skill._grasp_pose`, the commanded grasp is
`combine_frame_transforms(true_link_pose, perceived_handle_local + grasp_offset_local)`. So:
- The controller **aims** the gripper at `perceived + offset` in the true link frame.
- The **true handle** is the articulation link (`body_pos_w[link]`), resolved by name and read from sim —
  **not** derived from `handle_local_pos`, so it is physically **unchanged**.
- The gripper must physically contact the **real** handle proxy to pull the free drawer (stiffness 0). A
  mis-compensated grasp (`offset ≠ -bias`) lands `|bias+offset|` off the real handle → edge-miss →
  `HANDLE_DETACHED`/`POSITION_TIMEOUT`. Outcome labels come from the **real** drawer joint (ground truth).

This reproduces exactly a calibration/perception bias: the policy is confidently wrong about the handle
location and physically misses unless its grasp offset compensates.

## Injection is non-invasive and default-OFF
- `handle_calibration_bias_v1.biased_spec(spec, bias_y)` returns a dataclass copy; `bias_y == 0` returns
  the spec **unchanged** (identity → old behaviour byte-for-byte). No opt-in run is affected.
- No file, frozen scene registry, `grasp_poses.json`, or shared adapter (`articulated_drawer_v2.py`) is
  modified. The damping-stage data/scene/skills/reports/tag are untouched.

## Contract capacity — no change needed
`secret_deployment_state` is a free audit-only dict and `hidden_state_id` a free string (both additive);
`build_x` emits only `X_OBSERVABLE_KEYS`, so `bias_y` **structurally cannot** enter `x`. Verified by
`assert_no_bias_in_x` and the injection test. Therefore **no `contract_change_request`** is filed; the
frozen contract (`open_drawer_v2`) is not edited.

## Runtime verification
- Spec-level (pre-sim): `verify_perceived_shift` — local-Y shifted by exactly `bias_y`, X/Z/quat
  unchanged, original spec untouched, `bias=0` identity.
- End-to-end (Isaac smoke, `verify_calibration_bias_injection_v1.py`, config
  `calibration_bias_injection_smoke_v1.yaml`, 3 bias × 3 offset): at t0 captures the perceived handle
  (= true + bias), the **true link world pose** (must be constant across bias), and the commanded grasp
  world pose (world shift must equal `|bias+offset|` mapped to the handle local-Y axis); then runs each
  cell to completion to confirm mis-compensation degrades and the compensating offset recovers.
- `test_calibration_bias_injection_v1.py` asserts the algebra always and the smoke properties when present.

## Verification checklist (maps to the 7 required properties)
1. `bias=0` ≡ old behaviour — `biased_spec` identity + smoke (0,0) succeeds. ✓
2. ± bias → correct direction & magnitude of the commanded-grasp shift along handle local-Y — smoke
   `grasp_world_shift == |bias+offset|`, `perceived_dy == bias`. ✓
3. true handle pose unchanged — smoke `true_link_drift ≈ 0`. ✓
4. bias not in `x` — `assert_no_bias_in_x` + smoke `x_keys` check. ✓
5. candidate offset + bias add per the formula — `commanded_grasp_local_y` test. ✓
6. reset restores the session bias — bias is a pure function of the session config (re-applied to a fresh
   spec copy every episode); nothing persists in sim state to restore. ✓ (re-derived each reset)
7. old damping / matched-bank tests do not regress — run in the freeze step. ✓
