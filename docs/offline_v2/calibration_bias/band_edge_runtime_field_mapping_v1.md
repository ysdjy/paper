# Band-edge runtime field mapping v1 (verified) + schema v1.1 additive clarification

Claude B, pre-run audit. Verified mapping of each frozen schema field to its runtime source, from
first-hand reading of `band_edge_instrumentation_core_v1.py`, `band_edge_instrumentation_runtime_v1.py`,
and `band_edge_runtime_v1.py`, cross-checked against the 6 clear-zone smoke records. Also publishes an
**additive** schema v1.1 clarification for the joint-margin fields (allowed pre-run because the raw signed
information is complete — see the compliance audit §5).

## Field → runtime source (all present on every smoke record)
| group | frozen field(s) | runtime source | verified |
|---|---|---|---|
| 5.1 joint margin | `minimum_joint_limit_margin_rad`, `..._normalized`, `..._joint_index`, `..._phase` | `JointMarginTracker.update(arm_q[0:7], adapter._joint_lower/_upper (soft limits), phase)` every control step; episode-min | ✓ (6/6) |
| 5.1 (additive) | `minimum_joint_limit_margin_rad_signed`, `joint_limit_margin_negative_flag` | raw signed min tracked before clip; flag when signed<0 | ✓ (signed ≈ −1e−5, flag True) |
| 5.2 IK | `ik_solve_count`, `ik_failure_count`, `max_consecutive_ik_failures`, `ik_failure_phase_counts` | `InstrumentedIKAdapter.solve()` → `IKFailureCounter.update(res.success, phase)` | ✓ (timeout ≠ IK failure) |
| 5.3 clamps | `joint_step_clamp_count`, `joint_limit_clamp_count` | `detect_clamps(q_curr, res.q_des, limits, max_joint_step)` per successful solve; two separate counters | ✓ (both can fire) |
| 5.4 failure phase | `failure_phase`, `failure_transition` | `failure_phase_from_history(skill.runtime.history, success)` — last state before `FAILED` | ✓ |
| 5.5 close snapshot | `true_handle_error_at_close_3d`, `..._local_y`, `tcp_pose_at_close`, `true_handle_pose_at_close` | at the `CLOSE_GRIPPER→PULL` step; `_true_handle_world` = link pose ⊕ `spec.handle_local_pos` (**TRUE, not biased**); `null` + `close_snapshot_reason` if unreached | ✓ (0 → captured; +0.05 → null+reason) |
| 5.6 gripper at close | `gripper_width_at_close`, `gripper_command_at_close` | same control step | ✓ |
| 6 collision | `max_unintended_contact_force_N`, `unintended_contact_frame_count`, `first_unintended_contact_phase`, `contact_force_by_phase`, `contact_sensor_available` (+ `contact_sensor_backend`) | `CollisionMonitor(ContactSensor.net_forces_w, non-finger links)` per step | ✓ (sensor available, backend recorded) |
| secret/audit | `nominal_bias_y`, `residual_bias_y`, `actual_bias_y` | `secret_deployment_state` only | ✓ (never in x/g/theta/H) |

Bias injection uses `handle_calibration_bias_v1.biased_spec(spec, actual_bias_y)` for the **perceived**
handle (what the skill sees); the close snapshot and true-handle error use the **unbiased** link pose —
correctly separating the injected hidden state from the ground-truth diagnostic.

## Schema v1.1 — additive clarification (joint-limit margin)
This is an **additive** clarification of `band_edge_characterization_schema_v1.md`; it introduces **no**
runtime change and does **not** modify any frozen field. It documents the two audit fields the runtime
adds so the semantics are unambiguous BEFORE the run (per the audit's
`ACCEPT_ADDITIVE_SIGNED_DIAGNOSTIC`).

| field | dtype | unit | range | definition |
|---|---|---|---|---|
| `minimum_joint_limit_margin_rad` | float | rad | **[0, joint_range]** | non-negative distance-to-limit summary (the original frozen field); clamped to its declared range so a sub-µrad soft-limit numerical excursion does not violate the schema |
| `minimum_joint_limit_margin_rad_signed` | float | rad | (−ε, joint_range] | **raw signed** minimum margin **before** clipping; negative iff a joint momentarily sat past a soft limit |
| `joint_limit_margin_negative_flag` | bool | flag | {true,false} | true iff `minimum_joint_limit_margin_rad_signed < 0` |

Interpretation rule (frozen here, pre-run): a **real** joint-limit involvement is indicated only when the
signed margin is negative by a physically meaningful amount **and** `joint_limit_clamp_count > 0`; a
sub-µrad negative signed margin with zero limit-clamps is a soft-limit numerical artifact, **not** a
joint-limit failure. This distinction is what excludes "joint-limit failure" as a band-edge cause.

## Mechanism-exclusion coverage (verified fields → excludable mechanism)
| mechanism | excluded by (present in runtime) |
|---|---|
| joint-limit failure | signed margin < 0 (meaningful) **and** `joint_limit_clamp_count` > 0 |
| IK local unreachability | `ik_failure_count` > 0 / step-clamp saturating without progress |
| cabinet collision | `max_unintended_contact_force_N` above the (post-run, pre-frozen) threshold |
| empty grasp (grasped air) | `gripper_width_at_close` ≈ open **and** `true_handle_error_at_close_3d` large |
| off-centre grasp | `true_handle_error_at_close_3d` large but bounded (e.g. smoke −0.05 → 0.056 m) |
| PULL slip | `handle_detached` + `handle_relative_error` + `failure_phase == PULL` |

All six competing mechanisms are excludable from the recorded fields **once the run reliability fixes
(compliance audit §10) land** — the fields themselves are complete and correct.
